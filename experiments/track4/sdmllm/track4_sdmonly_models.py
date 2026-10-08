"""SDMONLY models: a language model whose body is a stack of sparse memory reads and nothing else.

<claudes_code_comments>
** Function List **
ProductKeyStore.forward(q, collect, slot) - product-key sparse read: wake about k of n_sub^2 locations, sum their values
ProductKeyStore.enable_row_grad(n_tokens, slots, device) - opt-in: hand the optimiser per-row gradients (lane SPARSEROWS)
StoreBank.forward(h, q) - the store for hop h (one shared store, or one per hop)
SdmMix.weights(x) - per position, the 2k hard-locations it addresses and their weights (sparse)
SdmMix._chunk(...) - one chunk: densify only that chunk's weights, write-then-read, carry the counters
SdmMix.forward(x) - the mixing layer: write earlier positions into locations, read at the current one
SdmMix._run_eager(x) / _run(x) - the same layer outside torch.compile (windows over 1,024) / inside it
SdmMixFast.locate(q) - the exact product-key top-2k searched over the staircase of rank pairs only
SdmMixFast.weights(x) - SdmMix.weights with that search, in slices of positions (memory only)
SdmMixFast.forward(x) / _run(x) - the same layer from sorted index lists (track4_sdmonly_mixfast.mix_scan), eager
SdmOnlyLM.features(idx) - back-token embeddings and causal moving averages, each RMS-normalised
SdmOnlyLM.hidden(idx) - features -> working vector -> hops of sparse reads (or dense control) -> optional readout
SdmOnlyLM.forward(idx) - hidden states times the tied embedding (the cleanup against the token table)
count_store(model) - parameters in the stores and their query maps
flops_per_token(model, T) - forward matmul FLOPs per token, head and body
matched_dense_f(d, d_a, n_sub, k, heads) - SwiGLU width whose per-hop FLOPs equal one sparse read's
build_sdmonly(arm, V, cfg) - arm name -> model
selftest() - shapes, exact top-k against brute force, ablations, gradient reach, causality

** Technical Review **
- The shape (lane SDMONLY, 2026-10-03): context features as in SdmLM (n_back previous token embeddings plus causal
  moving averages, concatenated, one linear map to d), then `hops` sparse reads x <- x + read_h(x), then the tied
  head. No SwiGLU readout by default (readout_f 0). The only non-linearity in the body is which locations wake.
- ProductKeyStore follows product-key memory layers (Lample et al. 2019, "Large Memory Layers with Product Keys";
  Berges et al. 2024, "Memory Layers at Scale"): the query is split in two halves, each half is scored against
  n_sub sub-keys, and a location (i, j) scores s1[i] + s2[j]. The top 2k of the n_sub^2 locations lie inside the
  top-2k by top-2k grid of the two halves, so the search is exact and costs 2 n_sub scores instead of n_sub^2.
- It keeps the SDMLLMSTORE fixes measured on SdmStore: a soft cut-off w = sigmoid((s - theta) / softness) with theta
  the k-th score (detached) over 2k candidates, gain 1/k, values initialised at zero, and the query gradient into x
  scaled by qgrad (0 stops it). Queries are batch-normalised without affine terms so they spread over locations.
- Scores: each half query has unit variance per dimension after batch norm and each sub-key is unit length, so a
  half score is about N(0, 1); the sum is divided by sqrt(2) so softness means what it means in SdmStore.
- SdmMix (2026-10-03, the all-SDM answer to attention): a Kanerva memory written and read at run time. Each position
  writes a value into the hard-locations its address picks; each later position reads the locations its own address
  picks, holding only what earlier positions wrote, faded per head (decay 1.0 = never fades). Chunked: inside a chunk
  one masked product, between chunks only the counters are carried, so a token's cost does not grow with the window.
  Windows over 1,024 run this layer eagerly because Triton failed to compile the chunk loop at T 2,048 and 4,096. Weights stay
  sparse (2k per position) and only one chunk is densified at a time; in training, a window longer than one chunk
  recomputes each chunk in backward (checkpoint), so memory grows with the window only through the sparse indices.
- SdmMixFast (lane MIXERFAST, 2026-10-03; opt-in mix_impl="fast", default "dense" so no past number moves): the
  same parameters and the same maths as SdmMix, but positions meet only through the index lists. Every (location,
  position, weight) triple is sorted by location (stable, so positions stay in order) and a segmented faded scan
  gives what each location holds when a position reads it (track4_sdmonly_mixfast.py). No (L, n_sub^2) array is
  built, so a token costs about 2k x dh x (scan tile) per head whatever n_sub and the window are. A custom autograd
  Function with an exact backward (gradcheck in float64); the self-test checks output and every gradient against
  the dense layer, fading and never-fading heads, several dense chunk sizes, B 3. It always runs eagerly (a sort and
  a data-dependent loop) and outside autocast (float32), while the dense layer's products ran in bf16 under autocast.
- heads > 1 gives each head its own sub-keys and all heads one shared value table; the read is the mean over heads.
- Arms: `sdmonly` (the model), `sdmonly_dense` (each read replaced by a SwiGLU whose FLOPs match one read: the
  matched-compute control), `sdmonly_none` (no reads: the floor). Parameter names of every store start with
  "store." so the S0 trainer puts them in the store optimiser group.
- Ablations for scoring: store.ablate = 'zero_read' (reads return zero) or 'shuffle_keys' (the value rows are
  permuted, so addressing is kept and content is scrambled).
- row_grad (lane SPARSEROWS, default off, training only): the value rows are read detached and a zero tensor `tap`
  (N, d_v) is added to the read. The read is the same number, the sub-keys and the query map get the same gradient,
  and the value table gets none from autograd: tap.grad is dLoss/dread, and the store keeps the weights and row ids
  of the step, so track4_sdmonly_optim.py can form the gradient of only the woken rows without an M by d tensor.
  With row_grad off the forward and backward are the lines they were before the switch existed.
- Wave-2 options, all off by default (the default model's parameters and forward are unchanged): `untie` (a separate
  input token table; the head stays `emb`), `hop_mlp` (a SwiGLU of that width after every read), `ng_rows` and
  `ng_orders` (exact n-gram hash tables, S0's NgramStore, read once before the first hop; they live in the store
  group and obey the ablations), `conj` (2 or 3: element-wise products of the last 2 or 3 normalised token vectors
  as extra features). `rt_write` K: the run-time write. Each earlier position in the window writes the vector of
  the token that followed it into the K locations its own address woke (an address map of its own, product keys,
  no stored values); a later position reads the sum over the locations it wakes, through a zero-initialised map.
  This is Kanerva's write done at run time inside the context window, and the model's only path past the 8 back
  tokens and the moving averages. Locations are hashed into `rt_buckets` for the overlap product. `rt_shuffle` is
  its control: the same write with the written vectors shuffled across positions. `graded`: a woken location's
  weight is its score minus the cut-off (zero at the k-th location), so a read is a k-sparse ReLU layer whose
  hidden units are locations; with `qgrad 1` the gradient reaches the features through the address.
Docs: TRAINING_SDMONLY_HOW_WE_TRAIN.md · runs_sdmonly/PREREG_SDMONLY_SMALL_2026-10-03.md
</claudes_code_comments>
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track4_sdmllm_models import NgramStore, RMSNorm, SwiGLU, causal_ema, grad_scale  # noqa: E402
from track4_sdmonly_mixfast import mix_scan  # noqa: E402


class ProductKeyStore(nn.Module):
    def __init__(self, n_sub, d_a, d_v, k=32, softness=0.25, heads=1, seed=0):
        super().__init__()
        assert d_a % 2 == 0
        g = torch.Generator().manual_seed(1234 + seed)
        self.n_sub, self.M, self.d_a, self.d_v, self.k, self.softness, self.heads = n_sub, n_sub * n_sub, d_a, d_v, k, softness, heads
        self.keys1 = nn.Parameter(torch.randn(heads, n_sub, d_a // 2, generator=g))
        self.keys2 = nn.Parameter(torch.randn(heads, n_sub, d_a // 2, generator=g))
        self.values = nn.Parameter(torch.zeros(self.M, d_v))
        self.register_buffer("perm", torch.randperm(self.M, generator=g), persistent=False)
        self.ablate = None  # None | 'zero_read' | 'shuffle_keys'
        self.last_stats = {}
        self.last_si = None
        self.row_grad = False  # lane SPARSEROWS; see enable_row_grad
        self.graded = False  # wave 5: weights grow with the score instead of switching on
        self.row_taps, self.row_w, self.row_rows = [], [], []

    def enable_row_grad(self, n_tokens, slots=1, device=None):
        """Opt-in. One zero tap of (n_tokens, d_v) per call of this store in a forward (slots > 1 for a shared store)."""
        device = device or self.values.device
        self.row_taps = [torch.zeros(n_tokens, self.d_v, device=device, requires_grad=True) for _ in range(slots)]
        self.row_w, self.row_rows = [None] * slots, [None] * slots
        self.row_grad = True

    def locate(self, q):
        """q: (N, heads, d_a) -> scores (N, heads, c) descending, location ids (N, heads, c)."""
        half = self.d_a // 2
        s1 = torch.einsum("nhd,hsd->nhs", q[..., :half], F.normalize(self.keys1, dim=-1).to(q.dtype))
        s2 = torch.einsum("nhd,hsd->nhs", q[..., half:], F.normalize(self.keys2, dim=-1).to(q.dtype))
        c1 = min(2 * self.k, self.n_sub)  # 2k per half, so the top 2k of the grid are the true top 2k
        v1, i1 = s1.topk(c1, dim=-1)
        v2, i2 = s2.topk(c1, dim=-1)
        comb = ((v1[..., :, None] + v2[..., None, :]) / math.sqrt(2.0)).flatten(-2)
        c = min(2 * self.k, c1 * c1)
        sv, sc = comb.topk(c, dim=-1)
        a = torch.div(sc, c1, rounding_mode="floor")
        si = i1.gather(-1, a) * self.n_sub + i2.gather(-1, sc - a * c1)
        return sv, si

    def forward(self, q, collect=True, slot=0):
        """q: (N, heads, d_a) -> read (N, d_v)."""
        if self.ablate == "zero_read":
            return q.new_zeros(q.shape[0], self.d_v)
        sv, si = self.locate(q)
        kk = min(self.k, sv.shape[-1])
        theta = sv[..., kk - 1: kk].detach()
        if self.graded:  # a location's weight is how far its score is above the cut-off (a k-sparse ReLU layer)
            w = (sv - theta).clamp(min=0)
        else:
            w = torch.sigmoid((sv - theta) / self.softness)
        rows = self.perm[si] if self.ablate == "shuffle_keys" else si
        if self.row_grad and self.training and self.ablate is None:
            vals = self.values.detach()[rows]
            read = torch.einsum("nhc,nhcd->nd", w.to(vals.dtype), vals) / (self.k * self.heads) + self.row_taps[slot]
            self.row_w[slot], self.row_rows[slot] = w.detach(), rows
        else:
            vals = self.values[rows]  # (N, heads, c, d_v)
            read = torch.einsum("nhc,nhcd->nd", w.to(vals.dtype), vals) / (self.k * self.heads)
        if collect:
            with torch.no_grad():
                self.last_stats = {"fire_mass": round(float(w.float().sum(-1).mean()), 3),
                                   "sim_top": round(float(sv[..., 0].float().mean()), 3),
                                   "distinct_topk_frac": round(float(torch.unique(si[..., :kk]).numel()) / self.M, 5)}
                self.last_si = si
        return read


class StoreBank(nn.Module):
    """One store shared by every hop, or one store per hop. `ablate` and `collect_stats` reach every store."""

    def __init__(self, stores, ngram=None):
        super().__init__()
        self.banks = nn.ModuleList(stores)
        self.ngram = ngram  # optional exact n-gram hash tables read before the first hop (wave 2)
        self.collect_stats = True

    def set_ablate(self, mode):
        for b in self.banks:
            b.ablate = mode
        if self.ngram is not None:
            self.ngram.ablate = mode

    def of(self, h):
        return self.banks[h if len(self.banks) > 1 else 0]

    def forward(self, h, q):
        return self.of(h)(q, collect=self.collect_stats, slot=h if len(self.banks) == 1 else 0)


class SdmMix(nn.Module):
    """One SDM mixing layer: a Kanerva memory written and read at run time, in parallel over the window.

    Each position t picks the 2k best of n_sub^2 hard-locations by product keys (a weight per location, the same
    soft cut-off as ProductKeyStore), WRITES a learned value of itself into them, and READS the locations its own
    address picks, holding only what EARLIER positions wrote (s < t). Head h fades old writes by decay[h] per
    token (1.0 = never). The read is normalised by the total weight it gathered, so it is a mean of counters.
    Inside a chunk the write-then-read is one masked product; between chunks only the counters (n_sub^2 x d per
    head) are carried, so the cost of a token does not grow with the window."""

    def __init__(self, d, heads=4, n_sub=64, d_a=64, k=32, softness=0.25, decays=(1.0, 0.999, 0.99, 0.9), chunk=256,
                 seed=0):
        super().__init__()
        assert d % heads == 0 and len(decays) == heads
        self.d, self.h, self.dh, self.k, self.chunk, self.d_a = d, heads, d // heads, k, chunk, d_a
        self.M = n_sub * n_sub
        self.norm = RMSNorm(d)
        self.wa = nn.Linear(d, heads * d_a, bias=False)
        self.bn = nn.BatchNorm1d(heads * d_a, affine=False)
        self.addr = ProductKeyStore(n_sub, d_a, 1, k=k, softness=softness, heads=heads, seed=seed)
        self.addr.values.requires_grad_(False)  # only the address machinery is used; the content is written at run time
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.register_buffer("decays", torch.tensor(decays, dtype=torch.float32), persistent=False)

    def weights(self, x):
        """x (B, T, d) -> SPARSE location weights: indices si and weights w, each (B, h, T, 2k)."""
        B, T, _ = x.shape
        q = self.bn(self.wa(self.norm(x)).reshape(B * T, -1).float()).view(B * T, self.h, self.d_a)
        sv, si = self.addr.locate(q)
        kk = min(self.k, sv.shape[-1])
        w = torch.sigmoid((sv - sv[..., kk - 1: kk].detach()) / self.addr.softness) / kk
        c = si.shape[-1]
        return si.view(B, T, self.h, c).transpose(1, 2), w.float().view(B, T, self.h, c).transpose(1, 2)

    def _chunk(self, si_c, w_c, Vc, mem, cnt, lg):
        """One chunk: densify its weights (B, h, L, M) only here, write-then-read, carry the counters."""
        B, h, L, _ = si_c.shape
        Ac = torch.zeros(B, h, L, self.M, device=Vc.device, dtype=torch.float32).scatter_add_(3, si_c, w_c)
        pos = torch.arange(L, device=Vc.device, dtype=torch.float32)
        age = pos[:, None] - pos[None, :]  # t - s
        dec = torch.exp(lg * age.clamp(min=0)) * (age > 0).float()  # strictly earlier, faded by age
        ov = torch.matmul(Ac, Ac.transpose(-1, -2)) * dec  # (B, h, L, L)
        fade_in = torch.exp(lg * (pos + 1)[None, None, :, None])  # fade of the carried counters by position t
        num = torch.matmul(ov, Vc) + torch.matmul(Ac, mem) * fade_in
        den = ov.sum(-1, keepdim=True) + (Ac * cnt[:, :, None, :]).sum(-1, keepdim=True) * fade_in
        tail = torch.exp(lg * (L - 1 - pos)[None, None, :, None])  # fade of each write by the chunk's end
        mem = mem * torch.exp(lg * L) + torch.matmul((Ac * tail).transpose(-1, -2), Vc)
        cnt = cnt * torch.exp(lg * L)[..., 0] + (Ac * tail).sum(2)
        return num / (den + 1e-6), mem, cnt

    def forward(self, x):
        if x.shape[1] > 1024:  # Triton fails to compile the unrolled chunk loop at T 2,048 and 4,096 (PassManager::run failed,
            return self._run_eager(x)  # w10_cn_mix4_T4096_200M); long windows run this layer eagerly, short ones as before
        return self._run(x)

    @torch.compiler.disable
    def _run_eager(self, x):
        return self._run(x)

    def _run(self, x):
        B, T, D = x.shape
        si, w = self.weights(x)  # (B, h, T, 2k) each: never a dense T x M grid for the whole window
        V = self.wv(self.norm(x)).float().view(B, T, self.h, self.dh).transpose(1, 2)  # (B, h, T, dh)
        outs = []
        mem = V.new_zeros(B, self.h, self.M, self.dh)
        cnt = V.new_zeros(B, self.h, self.M)
        lg = torch.log(self.decays.clamp(min=1e-6))[None, :, None, None]
        ckpt = torch.is_grad_enabled() and T > self.chunk  # long windows: recompute each chunk in backward
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            args = (si[:, :, c0:c1], w[:, :, c0:c1], V[:, :, c0:c1], mem, cnt, lg)
            if ckpt:
                o, mem, cnt = torch.utils.checkpoint.checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                o, mem, cnt = self._chunk(*args)
            outs.append(o)
        out = torch.cat(outs, 2)
        return self.wo(out.transpose(1, 2).reshape(B, T, D).to(x.dtype))


class SdmMixFast(SdmMix):
    """The same layer, the same parameters and the same maths as SdmMix, computed from the index lists (lane
    MIXERFAST): every (location, position, weight) triple is sorted by location, and a segmented faded scan gives
    what each location holds when a position reads it. Nothing n_sub^2 wide is built, so a token's cost does not grow
    with n_sub^2 or with the window. `chunk` is unused; `scan_tile` and `scan_piece` set the scan's tile and the tiles
    per batched product (memory only, the result does not depend on them)."""

    def __init__(self, *args, scan_tile=32, scan_piece=8192, scan_max_triples=1 << 23, locate_rows=16384, **kw):
        super().__init__(*args, **kw)
        self.scan_tile, self.scan_piece, self.scan_max_triples = scan_tile, scan_piece, scan_max_triples
        self.locate_rows = locate_rows

    def locate(self, q):
        """ProductKeyStore.locate with the same result and less work: the two half-score lists are sorted, so a
        pair (a, b) of ranks can be among the top c of the grid only if (a + 1)(b + 1) <= c (every pair (a', b') with
        a' <= a and b' <= b scores at least as high). That staircase is about c ln c pairs (303 for c 64) instead of
        the full (2k)^2 grid (4,096)."""
        st = self.addr
        half = st.d_a // 2
        s1 = torch.einsum("nhd,hsd->nhs", q[..., :half], F.normalize(st.keys1, dim=-1).to(q.dtype))
        s2 = torch.einsum("nhd,hsd->nhs", q[..., half:], F.normalize(st.keys2, dim=-1).to(q.dtype))
        c1 = min(2 * st.k, st.n_sub)
        v1, i1 = s1.topk(c1, dim=-1)
        v2, i2 = s2.topk(c1, dim=-1)
        c = min(2 * st.k, c1 * c1)
        if getattr(self, "_stair", None) is None or self._stair[0] != (c1, c) or self._stair[1].device != q.device:
            ab = [(a_, b_) for a_ in range(c1) for b_ in range(c1) if (a_ + 1) * (b_ + 1) <= c]
            self._stair = ((c1, c), torch.tensor([x[0] for x in ab], device=q.device),
                           torch.tensor([x[1] for x in ab], device=q.device))
        ra, rb = self._stair[1], self._stair[2]
        comb = (v1.index_select(-1, ra) + v2.index_select(-1, rb)) / math.sqrt(2.0)  # index_select: a cheap backward
        sv, sc = comb.topk(c, dim=-1)
        si = i1.gather(-1, ra[sc]) * st.n_sub + i2.gather(-1, rb[sc])
        return sv, si

    def weights(self, x):
        """As SdmMix.weights, but the product-key search runs over slices of `locate_rows` positions, so its
        (positions, heads, (2k)^2) candidate grid never exists for a whole long window. Same numbers."""
        B, T, _ = x.shape
        q = self.bn(self.wa(self.norm(x)).reshape(B * T, -1).float()).view(B * T, self.h, self.d_a)
        parts = [self.locate(q[i:i + self.locate_rows]) for i in range(0, B * T, self.locate_rows)]
        sv, si = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        kk = min(self.k, sv.shape[-1])
        w = torch.sigmoid((sv - sv[..., kk - 1: kk].detach()) / self.addr.softness) / kk
        c = si.shape[-1]
        return si.view(B, T, self.h, c).transpose(1, 2), w.float().view(B, T, self.h, c).transpose(1, 2)

    @torch.compiler.disable  # a sort and a data-dependent carry loop; runs eagerly inside a compiled model
    def forward(self, x):
        return self._run(x)

    def _run(self, x):
        B, T, D = x.shape
        si, w = self.weights(x)  # (B, h, T, 2k) each
        V = self.wv(self.norm(x)).float().view(B, T, self.h, self.dh).transpose(1, 2).contiguous()
        lg = torch.log(self.decays.clamp(min=1e-6))
        nd = mix_scan(w.contiguous(), V, si.contiguous(), lg, self.M, self.scan_tile, self.scan_piece, self.scan_max_triples)
        out = nd[..., :-1] / (nd[..., -1:] + 1e-6)
        return self.wo(out.transpose(1, 2).reshape(B, T, D).to(x.dtype))


class SdmOnlyLM(nn.Module):
    kind = "sdmonly"

    def __init__(self, V, d=256, d_a=256, n_sub=256, k=32, hops=4, softness=0.25, heads=1, n_back=8,
                 decays=(0.5, 0.8, 0.9, 0.97, 0.99), qgrad=0.0, readout_f=0, share_store=False, body="sdm",
                 dense_f=None, seed=0, untie=False, hop_mlp=0, ng_rows=0, ng_orders=(2, 3), conj=0,
                 rt_write=0, rt_buckets=4096, rt_shuffle=False, graded=False, mix_layers=0, mix_heads=4,
                 mix_n_sub=64, mix_d_a=64, mix_k=32, mix_decays=(1.0, 0.999, 0.99, 0.9), mix_f=0, mix_chunk=256,
                 mix_impl="dense"):
        super().__init__()
        assert mix_impl in ("dense", "fast")
        self.mix_layers, self.mix_impl = mix_layers, mix_impl
        self.graded = graded
        self.untie, self.hop_mlp, self.ng_rows, self.conj = untie, hop_mlp, ng_rows, conj
        self.rt_write, self.rt_buckets, self.rt_shuffle = rt_write, rt_buckets, rt_shuffle
        self.V, self.d, self.d_a, self.n_sub, self.k, self.hops, self.heads = V, d, d_a, n_sub, k, hops, heads
        self.n_back, self.decays, self.qgrad, self.f, self.store_mode = n_back, tuple(decays), qgrad, readout_f, body
        self.M = n_sub * n_sub
        self.emb = nn.Embedding(V, d)
        if untie:  # a separate input table; the output head stays self.emb
            self.in_emb = nn.Embedding(V, d)
        nfeat = n_back + len(decays) + max(0, conj - 1)
        self.nfeat = nfeat
        self.fnorms = nn.ModuleList([RMSNorm(d) for _ in range(nfeat)])
        self.wx = nn.Linear(nfeat * d, d, bias=False)
        self.qnorms = nn.ModuleList([RMSNorm(d) for _ in range(hops)])
        if body == "sdm":
            self.wq = nn.ModuleList([nn.Linear(d, heads * d_a, bias=False) for _ in range(hops)])
            self.qbn = nn.ModuleList([nn.BatchNorm1d(heads * d_a, affine=False) for _ in range(hops)])
            n_store = 1 if share_store else hops
            ngram = NgramStore(ng_rows, d, orders=tuple(ng_orders), seed=seed) if ng_rows else None
            if ngram is not None:
                ngram.collect_stats = False
            self.store = StoreBank([ProductKeyStore(n_sub, d_a, d, k=k, softness=softness, heads=heads, seed=seed + 17 * i)
                                    for i in range(n_store)], ngram=ngram)
            for b in self.store.banks:
                b.graded = graded
        elif body == "dense":
            self.dense_f = dense_f or matched_dense_f(d, d_a, n_sub, k, heads)
            self.mlps = nn.ModuleList([SwiGLU(d, self.dense_f) for _ in range(hops)])
        if hop_mlp:  # a small dense layer after every read (dense beside sparse)
            self.hnorms = nn.ModuleList([RMSNorm(d) for _ in range(hops)])
            self.hmlps = nn.ModuleList([SwiGLU(d, hop_mlp) for _ in range(hops)])
        if mix_layers:  # SDM mixing layers (the all-SDM answer to attention), each followed by a dense layer
            mix_cls = SdmMixFast if mix_impl == "fast" else SdmMix  # same parameters, same maths (lane MIXERFAST)
            self.mixes = nn.ModuleList([mix_cls(d, mix_heads, mix_n_sub, mix_d_a, mix_k, softness, tuple(mix_decays),
                                               mix_chunk, seed=seed + 501 * i) for i in range(mix_layers)])
            self.mix_f = mix_f or 4 * d
            self.mnorms = nn.ModuleList([RMSNorm(d) for _ in range(mix_layers)])
            self.mmlps = nn.ModuleList([SwiGLU(d, self.mix_f) for _ in range(mix_layers)])
        if rt_write:
            # the run-time write (Kanerva's write, inside the context window): every earlier position writes the
            # token that followed it into the locations its address woke; a later position reads those locations.
            self.rt_qnorm = RMSNorm(d)
            self.rt_wq = nn.Linear(d, d_a, bias=False)
            self.rt_bn = nn.BatchNorm1d(d_a, affine=False)
            self.rt_addr = ProductKeyStore(n_sub, d_a, 1, k=rt_write, softness=softness, heads=1, seed=seed + 991)
            self.rt_addr.values.requires_grad_(False)  # addresses only; the written content lives in the window
            self.rt_cnorm = RMSNorm(d)
            self.rt_out = nn.Linear(d, d, bias=False)
        if readout_f:
            self.rnorm = RMSNorm(d)
            self.readout = SwiGLU(d, readout_f)
        self.nf = RMSNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        if untie:
            nn.init.normal_(self.in_emb.weight, std=0.02)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in n and not n.startswith("store.") and not n.startswith("rt_addr.") \
                    and ".addr." not in n:
                nn.init.normal_(p, std=0.02)
        for mx in getattr(self, "mixes", []):
            nn.init.zeros_(mx.wo.weight)  # a new mixing layer starts as a no-op, like the store values
        if rt_write:
            nn.init.zeros_(self.rt_out.weight)  # the written read starts at zero, like the store values

    def rt_read(self, idx, x):
        """The run-time write and read. Position s (s < t) wrote the vector of token s+1 into its woken locations;
        position t reads the sum over its own woken locations. Causal: token s+1 is known at t whenever s < t."""
        B, T, D = x.shape
        q = self.rt_wq(self.rt_qnorm(grad_scale(x, self.qgrad))).reshape(B * T, self.d_a)
        q = self.rt_bn(q.float()).to(x.dtype).view(B * T, 1, self.d_a)
        sv, si = self.rt_addr.locate(q)  # (N, 1, c)
        kk = min(self.rt_write, sv.shape[-1])
        w = torch.sigmoid((sv - sv[..., kk - 1: kk].detach()) / self.rt_addr.softness)
        A = x.new_zeros(B, T, self.rt_buckets, dtype=torch.float32)
        A.scatter_add_(2, (si.view(B, T, -1) % self.rt_buckets), (w.view(B, T, -1) / kk).float())
        overlap = torch.bmm(A, A.transpose(1, 2)).tril(-1)  # (B, T, T): how much t's locations hold of s's write
        e = (self.in_emb if self.untie else self.emb)(idx)
        content = torch.cat([e[:, 1:], e.new_zeros(B, 1, D)], 1)  # what followed position s
        if self.rt_shuffle:  # the control: the same write with the written vectors shuffled across positions
            perm = torch.stack([torch.randperm(T, device=x.device) for _ in range(B)])
            content = torch.gather(content, 1, perm[..., None].expand(-1, -1, D))
        r = torch.bmm(overlap.to(content.dtype), self.rt_cnorm(content))
        return self.rt_out(r)

    def features(self, idx):
        e = (self.in_emb if self.untie else self.emb)(idx)
        T = e.shape[1]
        feats = []
        for j in range(self.n_back):
            ej = e if j == 0 else F.pad(e, (0, 0, j, 0))[:, :T]
            feats.append(self.fnorms[j](ej))
        for i, dc in enumerate(self.decays):
            feats.append(self.fnorms[self.n_back + i](causal_ema(e, dc)))
        # conjunction features: the element-wise product of the last 2 (and 3, ...) normalised token vectors
        prod = feats[0]
        for c in range(1, self.conj):
            prod = prod * feats[c]
            feats.append(self.fnorms[self.n_back + len(self.decays) + c - 1](prod))
        return torch.cat(feats, -1)

    def hidden(self, idx):
        x = self.wx(self.features(idx))
        B, T, D = x.shape
        self.hop_stats = []
        if self.store_mode == "sdm" and self.store.ngram is not None:
            r = self.store.ngram(idx)
            if r is not None:
                x = x + r.to(x.dtype)
        for h in range(self.hops):
            if self.store_mode == "sdm":
                q = self.wq[h](self.qnorms[h](grad_scale(x, self.qgrad))).reshape(B * T, self.heads * self.d_a)
                q = self.qbn[h](q.float()).to(x.dtype).view(B * T, self.heads, self.d_a)
                x = x + self.store(h, q).view(B, T, D)
                if self.store.collect_stats:
                    self.hop_stats.append(dict(self.store.of(h).last_stats))
            elif self.store_mode == "dense":
                x = x + self.mlps[h](self.qnorms[h](x))
            if self.hop_mlp:
                x = x + self.hmlps[h](self.hnorms[h](x))
        if self.rt_write:
            x = x + self.rt_read(idx, x)
        for i in range(self.mix_layers):
            x = x + self.mixes[i](x)
            x = x + self.mmlps[i](self.mnorms[i](x))
        if self.f:
            x = x + self.readout(self.rnorm(x))
        return self.nf(x)

    def forward(self, idx):
        return self.hidden(idx) @ self.emb.weight.t()


def count_store(model):
    if model.store_mode != "sdm":
        return 0
    return sum(p.numel() for p in model.store.parameters()) + sum(p.numel() for p in model.wq.parameters())


def read_flops(d, d_a, n_sub, k, heads):
    """One sparse read: the query map, two half-score matmuls against n_sub sub-keys, and the weighted value sum."""
    return 2 * d * heads * d_a + 2 * heads * 2 * n_sub * (d_a // 2) + 2 * heads * 2 * k * d


def matched_dense_f(d, d_a, n_sub, k, heads):
    return max(8, int(round(read_flops(d, d_a, n_sub, k, heads) / (2 * 3 * d))))


def flops_per_token(model, T):
    d, V = model.d, model.V
    if model.kind != "sdmonly":
        from track4_sdmllm_models import flops_per_token as base
        return base(model, T)
    head = 2 * V * d
    nfeat = model.nfeat
    body = 2 * nfeat * d * d + 2 * len(model.decays) * (T / 2) * d + 2 * 3 * d * model.f
    body += model.hops * 2 * 3 * d * model.hop_mlp
    if model.mix_layers:  # per layer: address, half scores, the write-read product, value and out maps, dense layer
        mx = model.mixes[0]
        L = min(T, mx.chunk)
        per = 2 * d * mx.h * mx.d_a + 2 * mx.h * 2 * int(round(mx.M ** 0.5)) * (mx.d_a // 2) + 2 * 2 * (L / 2) * mx.M \
            + 2 * 2 * (L / 2) * d + 2 * 2 * d * d + 2 * 3 * d * model.mix_f
        if model.mix_impl == "fast":  # the scan's tile product and the gather / index_add, per triple, instead of L x M
            per += 2 * mx.h * 2 * mx.k * (mx.dh + 1) * (mx.scan_tile + 2) - 2 * 2 * (L / 2) * mx.M - 2 * 2 * (L / 2) * d
        body += model.mix_layers * per
    if model.rt_write:  # address map, half scores, the overlap product over the window, the written read
        body += 2 * d * model.d_a + 2 * 2 * model.n_sub * (model.d_a // 2) + 2 * (T / 2) * model.rt_buckets + 2 * (T / 2) * d + 2 * d * d
    if model.store_mode == "sdm":
        body += model.hops * read_flops(d, model.d_a, model.n_sub, model.k, model.heads)
    elif model.store_mode == "dense":
        body += model.hops * 2 * 3 * d * model.dense_f
    return {"head": head, "body": body, "total": head + body}


def build_sdmonly(arm, V, cfg):
    body = {"sdmonly": "sdm", "sdmonly_dense": "dense", "sdmonly_none": "none"}[arm]
    return SdmOnlyLM(V, d=cfg.get("d", 256), d_a=cfg.get("d_a", 256), n_sub=cfg.get("n_sub", 256), k=cfg.get("k", 32),
                     hops=cfg.get("hops", 4), softness=cfg.get("softness", 0.25), heads=cfg.get("heads", 1),
                     n_back=cfg.get("n_back", 8), decays=tuple(cfg.get("decays", (0.5, 0.8, 0.9, 0.97, 0.99))),
                     qgrad=cfg.get("qgrad", 0.0), readout_f=cfg.get("readout_f", 0),
                     share_store=cfg.get("share_store", False), body=body, dense_f=cfg.get("dense_f"),
                     seed=cfg.get("seed", 0), untie=cfg.get("untie", False), hop_mlp=cfg.get("hop_mlp", 0),
                     ng_rows=cfg.get("ng_rows", 0), ng_orders=tuple(cfg.get("ng_orders", (2, 3))), conj=cfg.get("conj", 0),
                     rt_write=cfg.get("rt_write", 0), rt_buckets=cfg.get("rt_buckets", 4096),
                     rt_shuffle=cfg.get("rt_shuffle", False), graded=cfg.get("graded", False),
                     mix_layers=cfg.get("mix_layers", 0), mix_heads=cfg.get("mix_heads", 4),
                     mix_n_sub=cfg.get("mix_n_sub", 64), mix_d_a=cfg.get("mix_d_a", 64), mix_k=cfg.get("mix_k", 32),
                     mix_decays=tuple(cfg.get("mix_decays", (1.0, 0.999, 0.99, 0.9))), mix_f=cfg.get("mix_f", 0),
                     mix_chunk=cfg.get("mix_chunk", 256), mix_impl=cfg.get("mix_impl", "dense"))


def selftest():
    torch.manual_seed(0)
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    V = 500
    # 1. exact top-k: the product-key search equals a brute-force search over all n_sub^2 locations
    st = ProductKeyStore(n_sub=12, d_a=16, d_v=8, k=5, heads=2)
    q = torch.randn(40, 2, 16)
    sv, si = st.locate(q)
    k1, k2 = F.normalize(st.keys1, dim=-1), F.normalize(st.keys2, dim=-1)
    full = (torch.einsum("nhd,hsd->nhs", q[..., :8], k1)[..., :, None]
            + torch.einsum("nhd,hsd->nhs", q[..., 8:], k2)[..., None, :]).flatten(-2) / math.sqrt(2.0)
    bv, bi = full.topk(sv.shape[-1], dim=-1)
    check("product-key top scores equal brute force", torch.allclose(sv, bv, atol=1e-5))
    check("product-key top locations equal brute force", bool((si[..., :5].sort(-1).values == bi[..., :5].sort(-1).values).all()))
    # 2. shapes and the zero-initialised read
    m = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=3, n_back=4, decays=(0.8, 0.97))
    idx = torch.randint(1, V, (3, 20))
    out = m(idx)
    check("logits shape", out.shape == (3, 20, V))
    none = build_sdmonly("sdmonly_none", V, {"d": 32, "hops": 3, "n_back": 4, "decays": (0.8, 0.97)})
    none.load_state_dict({k: v for k, v in m.state_dict().items() if k in none.state_dict()})
    check("zero-initialised values read nothing (equals the no-read floor)", torch.allclose(out, none(idx), atol=1e-5))
    # 3. gradient reaches values, sub-keys and query maps; with qgrad 0 the query sends none into wx
    for b in m.store.banks:
        nn.init.normal_(b.values, std=0.1)
    loss = F.cross_entropy(m(idx).reshape(-1, V), torch.randint(0, V, (60,)))
    loss.backward()
    check("values receive gradient", float(m.store.banks[0].values.grad.abs().sum()) > 0)
    check("sub-keys receive gradient", float(m.store.banks[0].keys1.grad.abs().sum()) > 0)
    check("query maps receive gradient", float(m.wq[0].weight.grad.abs().sum()) > 0)
    check("only the woken rows receive gradient (sparse write)", float((m.store.banks[0].values.grad.abs().sum(-1) > 0).float().mean()) < 1.0)
    # 4. ablations
    m.eval()
    with torch.no_grad():
        base = m(idx)
        m.store.set_ablate("zero_read")
        z = m(idx)
        m.store.set_ablate("shuffle_keys")
        s = m(idx)
        m.store.set_ablate(None)
    check("zero_read changes the output once values are non-zero", not torch.allclose(base, z, atol=1e-5))
    check("shuffle_keys changes the output", not torch.allclose(base, s, atol=1e-5))
    # 5. causality: a change at position 12 leaves positions 0 to 11 as they were
    with torch.no_grad():
        idx2 = idx.clone()
        idx2[:, 12] = (idx2[:, 12] + 7) % V
        check("causal: earlier positions do not see a later token", torch.allclose(m(idx)[:, :12], m(idx2)[:, :12], atol=1e-5))
        check("causal control: later positions do change", not torch.allclose(m(idx)[:, 12:], m(idx2)[:, 12:], atol=1e-5))
    # 6. the dense control's FLOPs match one read; store parameter names carry the trainer's prefix
    f = matched_dense_f(256, 256, 256, 32, 1)
    check("dense control FLOPs within 2% of one read", abs(2 * 3 * 256 * f - read_flops(256, 256, 256, 32, 1)) / read_flops(256, 256, 256, 32, 1) < 0.02)
    names = [n for n, _ in m.named_parameters() if "keys" in n or "values" in n]
    check("every store parameter name starts with 'store.'", all(n.startswith("store.") for n in names) and len(names) > 0)
    shared = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=3, share_store=True)
    check("share_store keeps one value table", len(shared.store.banks) == 1 and shared(idx).shape == (3, 20, V))
    # 7. the wave-2 options build, run, stay causal, and default to off
    opt = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=2, n_back=4, decays=(0.8, 0.97), untie=True, hop_mlp=24,
                    ng_rows=64, ng_orders=(2, 3), conj=3)
    for b in opt.store.banks:
        nn.init.normal_(b.values, std=0.1)
    for t in opt.store.ngram.tables:
        nn.init.normal_(t, std=0.1)
    opt.eval()
    with torch.no_grad():
        o1 = opt(idx)
        idx3 = idx.clone()
        idx3[:, 12] = (idx3[:, 12] + 7) % V
        check("wave-2 options: logits shape", o1.shape == (3, 20, V))
        check("wave-2 options: causal", torch.allclose(o1[:, :12], opt(idx3)[:, :12], atol=1e-5))
        opt.store.set_ablate("zero_read")
        check("wave-2 options: zero_read also silences the n-gram tables", not torch.allclose(o1, opt(idx), atol=1e-5))
        opt.store.set_ablate(None)
    # 8. the run-time write: zero at the start, causal, and it reads what was written
    rt = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=2, n_back=4, decays=(0.8, 0.97), rt_write=4, rt_buckets=64)
    plain = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=2, n_back=4, decays=(0.8, 0.97))
    plain.load_state_dict({k: v for k, v in rt.state_dict().items() if k in plain.state_dict()})
    rt.eval(); plain.eval()
    with torch.no_grad():
        check("run-time write: zero-initialised output map leaves the model unchanged", torch.allclose(rt(idx), plain(idx), atol=1e-5))
        nn.init.normal_(rt.rt_out.weight, std=0.5)
        r1 = rt(idx)
        check("run-time write: changes the output once its map is non-zero", not torch.allclose(r1, plain(idx), atol=1e-5))
        check("run-time write: causal", torch.allclose(r1[:, :12], rt(idx3)[:, :12], atol=1e-5))
        check("run-time write: position 0 has nothing to read", torch.allclose(r1[:, 0], plain(idx)[:, 0], atol=1e-5))
    rt.train()
    F.cross_entropy(rt(idx).reshape(-1, V), torch.randint(0, V, (60,))).backward()
    check("run-time write: its address map receives gradient", float(rt.rt_wq.weight.grad.abs().sum()) > 0)
    gr = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=2, n_back=4, decays=(0.8, 0.97), graded=True, qgrad=1.0)
    for b in gr.store.banks:
        nn.init.normal_(b.values, std=0.1)
    F.cross_entropy(gr(idx).reshape(-1, V), torch.randint(0, V, (60,))).backward()
    check("graded read: runs, and with qgrad 1 the feature map receives gradient through the address",
          float(gr.wx.weight.grad.abs().sum()) > 0 and float(gr.store.banks[0].keys1.grad.abs().sum()) > 0)
    # 9. the SDM mixer: causal, chunked equals unchunked, and it carries context past the back tokens
    mx = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=0, n_back=2, decays=(0.9,), body="none", mix_layers=2,
                   mix_heads=4, mix_n_sub=6, mix_d_a=8, mix_k=4, mix_chunk=20)
    mx.eval()
    with torch.no_grad():
        check("SDM mixer: logits shape", mx(idx).shape == (3, 20, V))
        for m_ in mx.mixes:
            nn.init.normal_(m_.wo.weight, std=0.3)
        m1 = mx(idx)
        check("SDM mixer: causal", torch.allclose(m1[:, :12], mx(idx3)[:, :12], atol=1e-4))
        for m_ in mx.mixes:
            m_.chunk = 7
        m2 = mx(idx)
        check("SDM mixer: chunked (7) equals one chunk (20)", torch.allclose(m1, m2, atol=1e-4))
        for m_ in mx.mixes:
            m_.chunk = 20
        idx4 = idx.clone()
        idx4[:, 2] = (idx4[:, 2] + 11) % V
        check("SDM mixer: a change at position 2 reaches position 19 (beyond the 2 back tokens)",
              not torch.allclose(m1[:, 19], mx(idx4)[:, 19], atol=1e-5))
    mx.train()
    F.cross_entropy(mx(idx).reshape(-1, V), torch.randint(0, V, (60,))).backward()
    check("SDM mixer: its address map and value map receive gradient",
          float(mx.mixes[0].wa.weight.grad.abs().sum()) > 0 and float(mx.mixes[0].wv.weight.grad.abs().sum()) > 0)
    sm = SdmMix(32, 4, 6, 8, 4, 0.25, (1.0, 0.99, 0.9, 0.5), chunk=7)
    nn.init.normal_(sm.wo.weight, std=0.3)
    xg = torch.randn(2, 20, 32, requires_grad=True)
    y1 = sm(xg)
    y1.square().sum().backward()
    g1 = xg.grad.clone()
    xg.grad = None
    sm.chunk = 20
    y2 = sm(xg)
    y2.square().sum().backward()
    check("SDM mixer: checkpointed chunks (7) give the same output and gradient as one chunk (20)",
          torch.allclose(y1, y2, atol=1e-5) and torch.allclose(g1, xg.grad, atol=1e-4))
    # 10. the fast mixer (lane MIXERFAST): same output and gradients as the dense one, causal, long reach
    for decs in ((1.0, 0.99, 0.9, 0.5), (1.0, 1.0, 1.0, 1.0)):
        for ch in (5, 16, 37):
            torch.manual_seed(7)
            dn = SdmMix(32, 4, 6, 8, 4, 0.25, decs, chunk=ch, seed=3)
            nn.init.normal_(dn.wo.weight, std=0.3)
            fs = SdmMixFast(32, 4, 6, 8, 4, 0.25, decs, chunk=ch, seed=3, scan_tile=3, scan_piece=5,
                            scan_max_triples=37 * 8 * 2, locate_rows=29)  # several row slices and search slices
            fs.load_state_dict(dn.state_dict())
            xa = torch.randn(3, 37, 32, requires_grad=True)
            xb = xa.detach().clone().requires_grad_(True)
            ya, yb = dn(xa), fs(xb)
            gy = torch.randn_like(ya)
            (ya * gy).sum().backward()
            (yb * gy).sum().backward()
            pa, pb = dict(dn.named_parameters()), dict(fs.named_parameters())
            gp_ok = all((pa[n].grad is None) == (pb[n].grad is None) and
                        (pa[n].grad is None or torch.allclose(pa[n].grad, pb[n].grad, atol=1e-4, rtol=1e-3)) for n in pa)
            tag = "never-fading" if min(decs) == 1.0 else "fading"
            check(f"SDM mixer fast = dense ({tag}, dense chunk {ch}, B 3, T 37): output",
                  torch.allclose(ya, yb, atol=1e-5, rtol=1e-4))
            check(f"SDM mixer fast = dense ({tag}, dense chunk {ch}, B 3, T 37): gradients of x and every parameter",
                  torch.allclose(xa.grad, xb.grad, atol=1e-5, rtol=1e-3) and gp_ok)
    fl = SdmMixFast(32, 4, 20, 8, 6, 0.25, (1.0, 1.0, 1.0, 1.0))
    ql = torch.randn(500, 4, 8)
    sva, sia = fl.addr.locate(ql)
    svb, sib = fl.locate(ql)
    check("SDM mixer fast: the staircase search equals the full product-key search (scores and locations)",
          torch.allclose(sva, svb) and bool((sia.sort(-1).values == sib.sort(-1).values).all()))
    mf = SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=0, n_back=2, decays=(0.9,), body="none", mix_layers=2,
                   mix_heads=4, mix_n_sub=6, mix_d_a=8, mix_k=4, mix_chunk=20, mix_impl="fast")
    mf.load_state_dict(mx.state_dict())
    mf.eval(); mx.eval()
    with torch.no_grad():
        for m_ in mf.mixes:
            m_.scan_tile = 4
        f1 = mf(idx)
        check("SDM mixer fast: the whole model equals the dense-mixer model", torch.allclose(f1, mx(idx), atol=1e-4))
        check("SDM mixer fast: causal", torch.allclose(f1[:, :12], mf(idx3)[:, :12], atol=1e-4))
        check("SDM mixer fast: a change at position 2 reaches position 19",
              not torch.allclose(f1[:, 19], mf(idx4)[:, 19], atol=1e-5))
    check("SDM mixer: default mix_impl is dense", isinstance(build_sdmonly("sdmonly_none", V, {"d": 32, "mix_layers": 1}).mixes[0], SdmMix)
          and not isinstance(build_sdmonly("sdmonly_none", V, {"d": 32, "mix_layers": 1}).mixes[0], SdmMixFast))
    names2 = [n for n, _ in opt.named_parameters()]
    check("n-gram tables sit in the store group", any(n.startswith("store.ngram.tables") for n in names2))
    check("untied input table exists and the default model has none", "in_emb.weight" in names2 and not hasattr(m, "in_emb"))
    check("default model's parameter names are unchanged by the options", m.nfeat == 6 and not m.hop_mlp and m.store.ngram is None)
    print(f"{sum(ok)} of {len(ok)} checks pass")
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
    print("NEXT -> python3 track4_sdmonly_models.py --selftest")
