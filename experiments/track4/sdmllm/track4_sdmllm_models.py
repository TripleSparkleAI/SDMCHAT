"""SDMLLM S0 models: a Qwen-style decoder yardstick and the SDM language model (address, read, emit, roll back).

<claudes_code_comments>
** Function List **
RMSNorm - root-mean-square norm with a learned scale
rope_tables(T, hd, theta) - cos/sin tables for rotary position embedding
apply_rope(x, cos, sin) - rotate query/key pairs
QwenBlock - pre-norm block: GQA attention with QK-norm and RoPE, SwiGLU MLP
QwenLM - tied-embedding decoder, the matched transformer yardstick
SwiGLU - gated MLP
causal_ema(x, decay) - exponential moving average over time, computed in parallel
SdmStore - hard locations (keys, values), top-candidate read with a soft sigmoid cut-off
SdmLM - the SDM language model: context address -> iterated store reads -> readout -> tied head
count_params(model) - embedding vs non-embedding parameter counts
flops_per_token(model, T) - forward matmul FLOPs per token, split head / non-head
build(arm, V, cfg) - construct the model for a named arm
grad_scale(x, alpha) - identity forward, gradient multiplied by alpha (alpha 0 = stop-gradient)
NgramStore - fixed n-gram hash address, learned value rows (the address-lever arm)

** Technical Review **
- Both models share the tied 129,280 x d token embedding. At d=256 that table is 33.1M parameters, far more
  than the non-embedding parameters (~2-3M), and the tied head costs 2*V*d = 66M FLOPs per token, about ten
  times the rest of either model. Both facts are reported, never hidden.
- QwenLM: RMSNorm pre-norm, grouped-query attention (n_kv < n_head), per-head RMSNorm on q and k (Qwen3 style),
  RoPE theta 10,000, SwiGLU MLP, final RMSNorm, logits = h E^T.
- SdmLM is not a transformer. Per position t it forms an address from the context only:
  features f_t = [n(e_t), n(e_{t-1}), n(e_{t-2}), n(e_{t-3}), n(ema_0.8(e)_t), n(ema_0.97(e)_t)], a residual
  stream x = W_x f. Each hop: query q = BN(W_q^h n(x)) (a non-affine batch norm, as in product-key
  memory layers, so queries spread over the locations instead of collapsing onto a few); similarity s_m = q . k_m / sqrt(d_a) to every hard
  location m; candidates = top 2k; the cut-off theta = k-th largest s (detached), firing weight
  w_m = sigmoid((s_m - theta) / softness); read = sum_m w_m v_m / k; x <- x + read. The read feeds the next
  hop's address (SOFTSDM's rounds). After the hops a SwiGLU readout and the tied head give the next token.
  The emitted token then enters e_{t+1}: that is the roll-back of the generator loop.
- The soft cut-off follows SOFTSDM (settle-rs/src/softsdm.rs): a location fires with probability
  sigmoid((t - d)/w) with the threshold placed so the expected firing count is k, and the write/read gain is
  1/(expected count). softness -> 0 is the hard SDM cut; large softness spreads weight over all 2k candidates.
- Store modes (arms): 'learned' (keys and values trained), 'frozen' (values frozen random, keys trained),
  'none' (no store: the address goes straight to the readout, the 245 control), 'mlp' (each read replaced by
  a dense SwiGLU with the store's parameter count). Eval-time ablations on a trained store: 'shuffle_keys'
  (permute key rows, breaking key-value pairing) and 'zero_read'.
- EMA features use a T x T lower-triangular decay matrix (parallel, exact), normalised by the weight sum.
- SDMLLMSTORE additions (defaults reproduce S0 exactly): value_init 's0' (randn * 0.02 sqrt(k)) or 'zero'
  (values start at 0, so the store starts as a no-op and keys get gradient only once values grow); qgrad alpha
  scales the gradient that flows from the store's query back into the residual stream x (and so into wx and the
  embedding). alpha 1 = S0; alpha 0 = the address features are trained only by the readout path, while W_q, the
  keys and the values still train. A pilot on the S0 checkpoints showed the hard cut-off sends a gradient of norm
  2.2-2.4 into the embedding through this path (no-store model: 0.33), which the global clip at 1.0 then applies to
  every parameter.
- NgramStore (arm sdm_ngram): the address is a fixed multiplicative hash of the last n token ids (one table per n,
  default n = 2 and 3), the value rows are learned, zero-initialised; x <- x + sum_n T_n[h_n(context)]. Rows per
  table are chosen so the table parameters equal the S0 store's keys + values + query maps (3,856 x 256 x 2).
  A unigram-only table (orders (1,), 7,712 rows) is the control: same parameters, no context in the address.
  Ablations: 'shuffle_keys' permutes the bucket-to-row map, 'zero_read' drops the read.
</claudes_code_comments>
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


def rope_tables(T, hd, theta=10000.0, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    fr = torch.outer(t, inv)
    return fr.cos(), fr.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    return torch.stack((y1, y2), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class QwenBlock(nn.Module):
    def __init__(self, d, n_head, n_kv, f):
        super().__init__()
        self.hd = d // n_head
        self.n_head, self.n_kv = n_head, n_kv
        self.n1 = RMSNorm(d)
        self.q = nn.Linear(d, n_head * self.hd, bias=False)
        self.k = nn.Linear(d, n_kv * self.hd, bias=False)
        self.v = nn.Linear(d, n_kv * self.hd, bias=False)
        self.o = nn.Linear(n_head * self.hd, d, bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, f)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        h = self.n1(x)
        q = self.qn(self.q(h).view(B, T, self.n_head, self.hd)).transpose(1, 2)
        k = self.kn(self.k(h).view(B, T, self.n_kv, self.hd)).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.n_head // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class QwenLM(nn.Module):
    kind = "qwen"

    def __init__(self, V, d=256, n_layer=4, n_head=4, n_kv=2, f=688, T=256):
        super().__init__()
        self.V, self.d, self.T = V, d, T
        self.n_layer, self.n_head, self.n_kv, self.f = n_layer, n_head, n_kv, f
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([QwenBlock(d, n_head, n_kv, f) for _ in range(n_layer)])
        self.nf = RMSNorm(d)
        cos, sin = rope_tables(T, d // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in n:
                nn.init.normal_(p, std=0.02)
            if n.endswith("o.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))

    def hidden(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.nf(x)

    def forward(self, idx):
        return self.hidden(idx) @ self.emb.weight.t()


def causal_ema(x, decay):
    """x: (B, T, d). Returns the normalised causal exponential moving average along T."""
    T = x.shape[1]
    t = torch.arange(T, device=x.device)
    lag = (t[:, None] - t[None, :]).float()
    Wm = torch.where(lag >= 0, decay ** lag.clamp(min=0), torch.zeros_like(lag))
    Wm = Wm / Wm.sum(1, keepdim=True)
    return torch.einsum("ts,bsd->btd", Wm.to(x.dtype), x)


def grad_scale(x, alpha):
    if alpha == 1.0:
        return x
    xd = x.detach()
    return xd if alpha == 0.0 else xd + alpha * (x - xd)


class SdmStore(nn.Module):
    def __init__(self, M, d_a, d_v, k=32, softness=0.5, frozen_values=False, seed=0, value_init="s0"):
        super().__init__()
        g = torch.Generator().manual_seed(1234 + seed)
        self.M, self.d_a, self.d_v, self.k, self.softness = M, d_a, d_v, k, softness
        self.keys = nn.Parameter(torch.randn(M, d_a, generator=g))
        vals = torch.randn(M, d_v, generator=g) * 0.02 * math.sqrt(k)
        if value_init == "zero":
            vals = torch.zeros_like(vals)
        if frozen_values:
            self.register_buffer("values", vals)
        else:
            self.values = nn.Parameter(vals)
        self.ablate = None  # None | 'shuffle_keys' | 'zero_read'
        self.register_buffer("perm", torch.randperm(M, generator=g), persistent=False)
        self.last_stats = {}
        self.collect_stats = True  # False on the CUDA trainer: the stats force host syncs and graph breaks

    def forward(self, q):
        """q: (N, d_a) -> read (N, d_v)."""
        if self.ablate == "zero_read":
            return q.new_zeros(q.shape[0], self.d_v)
        keys = self.keys[self.perm] if self.ablate == "shuffle_keys" else self.keys
        kn = F.normalize(keys, dim=-1) * math.sqrt(self.d_a)
        s = (q @ kn.t()) / math.sqrt(self.d_a)  # (N, M)
        c = min(2 * self.k, self.M)
        sv, si = s.topk(c, dim=-1)
        theta = sv[:, self.k - 1: self.k].detach()
        w = torch.sigmoid((sv - theta) / self.softness)
        vals = self.values[si]  # (N, c, d_v)
        read = torch.einsum("nc,ncd->nd", w.to(vals.dtype), vals) / self.k
        if not self.collect_stats:
            return read
        with torch.no_grad():
            self.last_stats = {"fire_mass": round(float(w.float().sum(-1).mean()), 3), "sim_top": round(float(sv[:, 0].float().mean()), 3),
                               "distinct_topk_frac": round(float(torch.unique(si[:, : self.k]).numel()) / self.M, 4)}
            self.last_si = si
        return read


class NgramStore(nn.Module):
    def __init__(self, rows, d, orders=(2, 3), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(4321 + seed)
        self.orders, self.rows, self.d = tuple(orders), rows, d
        self.tables = nn.ParameterList([nn.Parameter(torch.zeros(rows, d)) for _ in self.orders])
        mult = torch.randint(1, 2**30, (len(self.orders), max(self.orders)), generator=g) * 2 + 1
        self.register_buffer("mult", mult, persistent=False)
        self.register_buffer("perm", torch.randperm(rows, generator=g), persistent=False)
        self.ablate = None
        self.last_stats = {}
        self.collect_stats = True

    def forward(self, idx):
        if self.ablate == "zero_read":
            return None
        B, T = idx.shape
        out = 0
        for j, n in enumerate(self.orders):
            h = torch.zeros_like(idx)
            for i in range(n):
                ids = idx if i == 0 else F.pad(idx, (i, 0))[:, :T]
                h = (h + ids * self.mult[j, i]) % 2147483647
            h = h % self.rows
            if self.ablate == "shuffle_keys":
                h = self.perm[h]
            out = out + self.tables[j][h]
        if not self.collect_stats:
            return out
        with torch.no_grad():
            self.last_stats = {"read_rms": round(float(out.float().pow(2).mean().sqrt()), 4)}
        return out


class SdmLM(nn.Module):
    kind = "sdm"

    def __init__(self, V, d=256, d_a=256, M=3600, k=32, hops=2, softness=0.5, f=688, store="learned",
                 n_back=4, decays=(0.8, 0.97), seed=0, mlp_f=None, value_init="s0", qgrad=1.0,
                 ngram_orders=(2, 3)):
        super().__init__()
        self.qgrad = qgrad
        self.V, self.d, self.d_a, self.M, self.k, self.hops, self.store_mode = V, d, d_a, M, k, hops, store
        self.n_back, self.decays, self.f = n_back, decays, f
        self.emb = nn.Embedding(V, d)
        nfeat = n_back + len(decays)
        self.fnorms = nn.ModuleList([RMSNorm(d) for _ in range(nfeat)])
        self.wx = nn.Linear(nfeat * d, d, bias=False)
        self.qnorms = nn.ModuleList([RMSNorm(d) for _ in range(hops)])
        if store in ("learned", "frozen"):
            self.wq = nn.ModuleList([nn.Linear(d, d_a, bias=False) for _ in range(hops)])
            # query batch-normalisation (as in product-key memory layers): spreads queries over the locations
            self.qbn = nn.ModuleList([nn.BatchNorm1d(d_a, affine=False) for _ in range(hops)])
            self.store = SdmStore(M, d_a, d, k=k, softness=softness, frozen_values=(store == "frozen"), seed=seed,
                                  value_init=value_init)
        elif store == "mlp":
            self.mlps = nn.ModuleList([SwiGLU(d, mlp_f) for _ in range(hops)])
        elif store == "ngram":
            self.store = NgramStore(M, d, orders=ngram_orders, seed=seed)
        self.rnorm = RMSNorm(d)
        self.readout = SwiGLU(d, f)
        self.nf = RMSNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in n and "store" not in n:
                nn.init.normal_(p, std=0.02)

    def features(self, idx):
        e = self.emb(idx)
        B, T, D = e.shape
        feats = []
        for j in range(self.n_back):
            ej = e if j == 0 else F.pad(e, (0, 0, j, 0))[:, :T]
            feats.append(self.fnorms[j](ej))
        for i, dc in enumerate(self.decays):
            feats.append(self.fnorms[self.n_back + i](causal_ema(e, dc)))
        return torch.cat(feats, -1)

    def hidden(self, idx):
        x = self.wx(self.features(idx))
        B, T, D = x.shape
        self.hop_stats = []
        if self.store_mode == "ngram":
            r = self.store(idx)
            if r is not None:
                x = x + r.to(x.dtype)
            if self.store.collect_stats:
                self.hop_stats.append(dict(self.store.last_stats))
        for h in range(self.hops if self.store_mode != "ngram" else 0):
            if self.store_mode in ("learned", "frozen"):
                q = self.qbn[h](self.wq[h](self.qnorms[h](grad_scale(x, self.qgrad))).reshape(B * T, self.d_a).float()).to(x.dtype)
                x = x + self.store(q).view(B, T, D)
                if self.store.collect_stats:
                    self.hop_stats.append(dict(self.store.last_stats))
            elif self.store_mode == "mlp":
                x = x + self.mlps[h](self.qnorms[h](x))
        x = x + self.readout(self.rnorm(x))
        return self.nf(x)

    def forward(self, idx):
        return self.hidden(idx) @ self.emb.weight.t()


def count_params(model):
    emb = model.emb.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    buffers_store = 0
    if getattr(model, "store_mode", None) == "frozen":
        buffers_store = model.store.values.numel()
    return {"embedding": emb, "non_embedding_trainable": total - emb, "frozen_store_values": buffers_store,
            "total_trainable": total}


def flops_per_token(model, T):
    """Forward matmul FLOPs per token (2 per multiply-add); attention uses the mean context T/2."""
    d, V = model.d, model.V
    head = 2 * V * d
    if model.kind == "qwen":
        hd = d // model.n_head
        per_layer = 2 * (d * d + 2 * d * model.n_kv * hd + d * d) + 2 * 3 * d * model.f
        attn = 2 * 2 * (T / 2) * d
        body = model.n_layer * (per_layer + attn)
    else:
        nfeat = model.n_back + len(model.decays)
        body = 2 * nfeat * d * d + 2 * 3 * d * model.f + 2 * len(model.decays) * (T / 2) * d
        if model.store_mode in ("learned", "frozen"):
            per_hop = 2 * d * model.d_a + 2 * model.M * model.d_a + 2 * 2 * model.k * d
            body += model.hops * per_hop
        elif model.store_mode == "mlp":
            body += model.hops * 2 * 3 * d * model.mlps[0].gate.out_features
    return {"head": head, "body": body, "total": head + body}


def build(arm, V, cfg):
    d = cfg.get("d", 256)
    T = cfg.get("T", 256)
    seed = cfg.get("seed", 0)
    if arm == "qwen":
        return QwenLM(V, d=d, n_layer=cfg.get("n_layer", 4), n_head=cfg.get("n_head", 4), n_kv=cfg.get("n_kv", 2),
                      f=cfg.get("f", 688), T=T)
    store = {"sdm": "learned", "sdm_frozen": "frozen", "sdm_nostore": "none", "sdm_mlp": "mlp",
             "sdm_big": "learned", "sdm_ngram": "ngram"}[arm]
    M = cfg.get("M_big", 16384) if arm == "sdm_big" else cfg.get("M", 3600)
    if arm == "sdm_ngram":
        # rows per table: the sdm store's keys + values + query maps, split over the tables
        M = cfg.get("ngram_rows", 3856 * 2 // len(cfg.get("ngram_orders", (2, 3))))
    mlp_f = None
    if store == "mlp":
        # a dense SwiGLU per hop with the store's parameter count: 3 d f = M (d_a + d) / hops + d d_a
        store_params_per_hop = cfg.get("M", 3600) * (cfg.get("d_a", 256) + d) / cfg.get("hops", 2) + d * cfg.get("d_a", 256)
        mlp_f = int(round(store_params_per_hop / (3 * d)))
    return SdmLM(V, d=d, d_a=cfg.get("d_a", 256), M=M, k=cfg.get("k", 32), hops=cfg.get("hops", 2),
                 softness=cfg.get("softness", 0.5), f=cfg.get("f", 688), store=store, seed=seed, mlp_f=mlp_f,
                 value_init=cfg.get("value_init", "s0"), qgrad=cfg.get("qgrad", 1.0),
                 ngram_orders=tuple(cfg.get("ngram_orders", (2, 3))), n_back=cfg.get("n_back", 4),
                 decays=tuple(cfg.get("decays", (0.8, 0.97))))
