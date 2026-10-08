"""ONESDM models: a residual stack whose only path between positions is a Kanerva memory written and read at run time.

<claudes_code_comments>
** Function List **
pk_address(z, keys1, keys2, k) - exact product-key top-k: scores (mean of two half cosines) and slot ids
RunTimeSdm.decay() - per-head fade, fixed buffer or learned sigmoid
RunTimeSdm.prepare(x) - addresses, soft weights, write gate and values of every position (no position mixing yet)
RunTimeSdm._chunk(...) - one chunk in parallel: densify its weights, causal write-then-read, carry the state
    (optionally only within a segment: the reset at BOS)
RunTimeSdm.forward(x, reset) - the chunked parallel form (training and long evaluation)
RunTimeSdm.forward_sequential(x, reset) - the token-by-token recurrence (the reference, and the decode step)
RunTimeSdm.finish(r, B, T) - mean or sum read, optional per-head norm, the zero-initialised output map
TrainedSdm.address(x) - exact product-key top-k of a learned query, softmax weights over the k picked slots
TrainedSdm.forward(x) - per token: weighted sum of the picked slots' values (embedding_bag), output map back to d
trained_sdm_params(d, n_sub, heads, d_a) - weight count of one TrainedSdm
matched_sdm_n_sub(d, mlp_f, heads, d_a) - n_sub whose TrainedSdm weights are closest to one SwiGLU plus its norm
Archive.slots(z) - pointer-SDM slot ids of a compressed key or query (discrete, no gradient)
Archive.candidates_parallel(Sa, Sb, T) - per position: the last P writers of each read slot, plus the recent window
Archive.read(...) - score candidates against stored compressed keys, keep the top few, weight, fetch exact vectors
Archive.forward(x) / forward_sequential(x) - the parallel form and the ring-buffer recurrence (must agree)
OneSdmLM.enfold(idx) / unfold(x) - token to vector, vector to token scores (tied table by default)
OneSdmLM.project(h) - normalised state to token scores (the tied table, or the head when untied)
OneSdmLM.residual(idx, sequential) - the residual stack: memory read added, archive read added, MLP (or TrainedSdm) added
OneSdmLM.hidden(idx, sequential) - nf(residual): the normalised state the head reads (the text trainers call this)
OneSdmLM.forward(idx, sequential) - logits = project(hidden)
OneSdmLM.memory_parameters() - names of the memory and archive parameters (their own optimiser group)
causal_ema_linear(x, decay, chunk) - the baseline's moving average in linear time (equals causal_ema)
use_linear_ema() - point track4_sdmonly_models at causal_ema_linear (runtime only, no file edit)
lm_kwargs(cfg) - merged cfg and the OneSdmLM keyword arguments
allsdm_n_sub(arm, c) - trained-table n_sub: matched to the MLP, or x sdm_big_mult (default 4, so 16x slots) for _big
build_onesdm(arm, V, cfg) - arm name -> model (onesdm, onesdm_archive, plain, onesdm_allsdm, onesdm_allsdm_big,
    sdmonly, yardstick_transformer)
param_count(model) - non-embedding and embedding parameter counts
decode_state_bytes(model, T) - bytes a model must keep to continue after T tokens (fixed for the SDM models)
body_violations(model) - every SwiGLU, attention module or body Linear wider than d (empty for the allsdm arms)
selftest() - addressing, shapes, causality with leaking controls, zero start, gate, chunked = sequential, archive,
    and the allsdm arms (top-k, sparse = dense read, zero start, structure, gradient reach, causality, weights)

** Technical Review **
- THE LAW (lane SDMONLY): all SDM, no transformer. The only operation that mixes positions here is RunTimeSdm (a
  Kanerva memory with a fixed number M = n_sub^2 of hard locations per head, written and read while reading the
  text) and, when switched on, the Archive (a pointer SDM over a growing buffer). There is no softmax over the
  window and no attention block. The transformer in build_onesdm is the YARDSTICK only (QwenLM, imported).
- ENFOLD = the token table (token to vector). THE COMPLICATE = the residual stream. UNFOLD = the tied head.
- Each layer: x <- x + Mem(x); x <- x + Archive(x) when on; x <- x + SwiGLU(RMSNorm(x)).
- Mem(x): h = RMSNorm(x). Per head: write address from W_k h, read address from W_q h, value W_v h, write gate
  g = sigmoid(W_g h + b) in [0, 1] (filler can learn to write weakly). Addressing is PRODUCT KEYS: each address is
  split in two halves, each half is L2-normalised and scored by cosine against n_sub unit sub-keys, a slot (i, j)
  scores the mean of the two half cosines, and the exact top-k of the n_sub^2 grid is found from the top-k of each
  half (2 n_sub scores per token, not n_sub^2). Weights are softmax(scale * score) over those k slots (scale learned
  per head). Why product keys over sign-of-projection (LSH bit addressing): (a) the search is exact and cheap at any
  M, (b) the soft weights over k slots give the address a gradient, where a sign hash gives none, (c) it is the
  addressing every earlier SDMONLY store and mixer used, so results compare. Queries and keys are L2-normalised,
  never batch-normalised: batch statistics would mix future positions into earlier outputs during training.
- State per head: mem (M x dh) and cnt (M). After reading at t, the write is mem <- lam mem + g a_t v_t^T,
  cnt <- lam cnt + g a_t. The read at t uses the state BEFORE t's own write, so a token sees strictly earlier writes,
  a write at s reaching t faded lam^(t-1-s). lam 0 makes a previous-token head (only s = t-1 survives), lam 1 never
  fades. read_mode 'mean': num / (den + eps), a convex mix of the values written where the read looks (bounded,
  zero where nothing was written); 'sum': num.
- Chunked parallel form: inside a chunk of L tokens the weights are densified to (L, M), the overlap of reads with
  earlier writes in the chunk is one (L, L) product masked by lam^(t-1-s) for s < t, the carried state is read
  through lam^(t - c0), and the state is advanced by lam^L and the chunk's faded writes. Only the state crosses
  chunks, so a token's cost is fixed (about L*M per head) whatever the window. In training, a window longer than
  one chunk recomputes each chunk in backward (checkpoint). forward_sequential is the same model as a recurrence
  with sparse gathers and scatters (a different code path), and the self-test checks the two agree.
- addr_bias S (default 0, so a zero learned bias): a learned vector per head added to both the write key and the
  read query, initialised N(0, S^2). With S well above the projections' scale every token wakes the same slots at
  init, so each head starts as a fading average (a fade-0 head as an exact previous-token head) and has to learn
  to address by content. It exists because a fade-0 head needs its read at t to overlap the write at t - 1 while
  both addresses come from different tokens: with M 1,024 and k 16 that overlap is about 0.25 slots at random init.
- out_norm divides each head's read by sqrt(mean(r^2) + 0.01), an RMS floor of 0.1, so a near-empty read (the first
  tokens, or a slot nobody wrote) is not scaled up to full size; with a 1e-6 floor the input gradient at width 32
  measured about 800 in the self-test, which is the climbing-gradient hazard this lane was asked to avoid.
- Stability flags, each from what failed before (the mixer led early, then fell behind with climbing gradients):
  zero-initialised output map (the model starts as a plain residual MLP stack, checked exactly), L2-normalised
  queries and keys, mem_in_grad (scales the gradient the memory sends back into the residual stream), a separate
  optimiser group with its own learning-rate multiplier and its own clip (the puzzle trainer), out_norm (per-head
  RMSNorm on the read), learn_decay (default fixed fades).
- Archive (OFF by default; DeepSeek-style): each token keeps a compressed key k_s and a compressed exact vector c_s
  (2 d_c floats per token, a buffer that grows with the text). Its position is written as a POINTER into a pointer
  SDM: a ring of the last P writers in each of the k slots its key wakes. A read wakes k slots by its query, takes
  their k*P pointers plus the last W positions (the exact recent window), drops duplicates, scores each candidate by
  q_t . k_s / sqrt(d_c), keeps the top n_fetch, and returns their soft-weighted (or straight-through hard) exact
  vectors through a zero-initialised map. Exactly causal: candidates are strictly earlier positions. The parallel
  form finds "the last P writers of slot m before t" by one sort and two searchsorted calls; forward_sequential keeps
  real ring buffers; the self-test checks they agree. NOT TRAINED WELL: which pointers come back is discrete. The
  pointer addresses are computed from the same compressed key and query as the scores, so they move only because
  the scoring trains those maps; the slot choice itself gets no gradient. Scoring a handful of fetched candidates
  with a softmax is a small fixed-size soft selection (the brief allows it); its cost per token is fixed at
  k*P + W candidates.
- THE ALL-SDM ARMS (lane SDMEVERYLAYER, 2026-10-07): ffn="sdm" puts a TrainedSdm where each SwiGLU was. Every
  layer is then x <- x + Mem(x) (run-time SDM, exactly as onesdm) and x <- x + TrainedSdm(x). No attention, no MLP,
  no Linear in the body wider than d (body_violations checks it). TrainedSdm, per token and per head: q = W_q
  RMSNorm(x) (heads * d_a <= d), exact product-key top-k over n_sub^2 slots (pk_address, the same addressing as
  RunTimeSdm), weights softmax(scale * score) over the k picked slots (scale learned per head), read = the weighted
  sum of those slots' value rows (dh = d / heads wide; F.embedding_bag, so no (N, h, k, dh) tensor is built), the
  heads concatenated and mapped back to d by W_o. Values start at ZERO, so at init the layer adds exactly nothing
  (checked exactly against the same model with every TrainedSdm removed, ffn="none"). The top-k pick is hard;
  gradient reaches the values (the picked rows), and the sub-keys and the query map through the picked scores.
  onesdm_allsdm sizes n_sub so the table's weights match the SwiGLU's (n_sub 41 at d 256, MLP 688: 1,681 slots per
  head x 4 heads x width 64). onesdm_allsdm_big multiplies n_sub by 4 (16x the slots) with the same k, so the
  values touched per token are the same (matched compute, about 10x the weights). The values are 3-D, so the puzzle
  trainer puts them in the no-decay group; W_q and W_o get weight decay like any body matrix.
- RESET AT BOS (lane TEXTWIRE, off by default): every forward starts from an empty memory, so each training window
  begins empty. With reset_at_bos the memory is also emptied before each BOS token reads, so the tokens from a BOS on
  give exactly the output they give when fed alone from an empty memory. The chunked form gives each position a
  segment number (the count of BOS tokens up to it) and lets a write reach a read only in the same segment; the carried
  state reaches a position only if no BOS lies between the chunk start and it. The recurrence empties the state at a
  BOS. The text trainer's self-test checks both forms against the alone-fed suffix. Not built for the archive.
- hidden(idx) is nf(residual(idx)), the convention of every text trainer here (the chunked loss and the fine-tune tool
  compute logits as hidden @ E^T); forward(idx) = project(hidden(idx)) gives the same logits as before.
- Baselines: 'sdmonly' is today's SdmOnlyLM (8 back tokens, 5 fading averages, product-key store, no run-time
  memory); its batch norm uses batch statistics in training, so its causality is checked in eval mode. Its
  causal_ema builds a T x T matrix; use_linear_ema() swaps in a linear-time version with the same numbers so the
  16k cost runs. 'plain' is OneSdmLM with the memory off (the no-memory floor). 'yardstick_transformer' is QwenLM.
Docs: ONESDM_README.md · IDEAS_SDMONLY_NEXT_2026-10-05.md (DECIDED: the middle path) · track4_sdmonly_models.py
(ProductKeyStore, the trained store TrainedSdm follows, without its batch-normalised queries, which are not causal)
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
from track4_sdmllm_models import QwenLM, RMSNorm, SwiGLU, causal_ema, grad_scale  # noqa: E402
import track4_sdmonly_models as SO  # noqa: E402


def pk_address(z, keys1, keys2, k):
    """z (N, h, d_a), keys (h, n_sub, d_a/2) -> scores (N, h, kk) descending, slot ids (N, h, kk) in [0, n_sub^2)."""
    half = z.shape[-1] // 2
    n_sub = keys1.shape[1]
    s1 = torch.einsum("nhd,hsd->nhs", F.normalize(z[..., :half], dim=-1), F.normalize(keys1, dim=-1).to(z.dtype))
    s2 = torch.einsum("nhd,hsd->nhs", F.normalize(z[..., half:], dim=-1), F.normalize(keys2, dim=-1).to(z.dtype))
    c1 = min(k, n_sub)
    v1, i1 = s1.topk(c1, dim=-1)
    v2, i2 = s2.topk(c1, dim=-1)
    comb = ((v1[..., :, None] + v2[..., None, :]) * 0.5).flatten(-2)
    kk = min(k, c1 * c1)
    sv, sc = comb.topk(kk, dim=-1)
    a = torch.div(sc, c1, rounding_mode="floor")
    si = i1.gather(-1, a) * n_sub + i2.gather(-1, sc - a * c1)
    return sv, si


class RunTimeSdm(nn.Module):
    """A Kanerva memory with M = n_sub^2 hard locations per head, written and read while the text is read."""

    def __init__(self, d, heads=4, n_sub=32, d_a=32, k=16, decays=(0.0, 0.8, 0.99, 1.0), learn_decay=False,
                 write_gate=True, gate_bias=0.0, read_mode="mean", out_norm=True, addr_scale=8.0, learn_scale=True,
                 chunk=128, read_eps=1e-3, mem_in_grad=1.0, out_norm_eps=1e-2, addr_bias=0.0, seed=0):
        super().__init__()
        assert d % heads == 0 and d_a % 2 == 0 and len(decays) == heads and read_mode in ("mean", "sum")
        g = torch.Generator().manual_seed(4321 + seed)
        self.d, self.h, self.dh, self.k, self.d_a, self.n_sub, self.M = d, heads, d // heads, k, d_a, n_sub, n_sub * n_sub
        self.chunk, self.read_mode, self.read_eps, self.mem_in_grad = chunk, read_mode, read_eps, mem_in_grad
        self.write_gate, self.out_norm, self.learn_decay = write_gate, out_norm, learn_decay
        self.out_norm_eps = out_norm_eps  # default 0.01, an RMS floor of 0.1: a near-empty read is not blown up
        self.enabled = True
        self._leak = False  # TEST ONLY: lets a position read writes from later in its chunk (the causality control)
        self.norm = RMSNorm(d)
        self.wk = nn.Linear(d, heads * d_a, bias=False)
        self.wq = nn.Linear(d, heads * d_a, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wg = nn.Linear(d, heads, bias=True)
        self.wo = nn.Linear(d, d, bias=False)
        self.keys1 = nn.Parameter(torch.randn(heads, n_sub, d_a // 2, generator=g))
        self.keys2 = nn.Parameter(torch.randn(heads, n_sub, d_a // 2, generator=g))
        # one learned bias per head, added to BOTH the write key and the read query: a large one makes every token
        # address the same slots at init (the head starts as a fading average and learns to address by content)
        self.addr_b = nn.Parameter(torch.randn(heads, d_a, generator=g) * addr_bias)
        self.log_scale = nn.Parameter(torch.full((heads,), math.log(addr_scale)), requires_grad=learn_scale)
        lam = torch.tensor(decays, dtype=torch.float32)
        if learn_decay:
            self.decay_logit = nn.Parameter(torch.logit(lam.clamp(1e-4, 1 - 1e-4)))
        else:
            self.register_buffer("decays_fixed", lam, persistent=True)
        if out_norm:
            self.onorm_w = nn.Parameter(torch.ones(heads, self.dh))
        self.gate_bias0 = gate_bias

    def reset_init(self):
        nn.init.zeros_(self.wo.weight)  # the read starts at zero: the model starts as the plain residual MLP stack
        nn.init.zeros_(self.wg.weight)
        nn.init.constant_(self.wg.bias, self.gate_bias0)

    def decay(self):
        return torch.sigmoid(self.decay_logit) if self.learn_decay else self.decays_fixed

    def prepare(self, x):
        B, T, _ = x.shape
        hx = self.norm(grad_scale(x, self.mem_in_grad)).float()
        scale = self.log_scale.exp()[None, :, None]
        kz = self.wk(hx).view(B * T, self.h, self.d_a) + self.addr_b
        qz = self.wq(hx).view(B * T, self.h, self.d_a) + self.addr_b
        svw, siw = pk_address(kz, self.keys1, self.keys2, self.k)
        svr, sir = pk_address(qz, self.keys1, self.keys2, self.k)
        ww = torch.softmax(svw * scale, -1)
        wr = torch.softmax(svr * scale, -1)
        gate = torch.sigmoid(self.wg(hx)) if self.write_gate else hx.new_ones(B, T, self.h)
        ww = ww * gate.reshape(B * T, self.h, 1)  # the gate scales every weight of the write
        c = siw.shape[-1]

        def bh(t):
            return t.view(B, T, self.h, c).transpose(1, 2)  # (B, h, T, k)
        V = self.wv(hx).view(B, T, self.h, self.dh).transpose(1, 2)  # (B, h, T, dh)
        self.last_gate = gate.detach()
        return bh(siw), bh(ww), bh(sir), bh(wr), V

    def _chunk(self, siw, ww, sir, wr, Vc, mem, cnt, lam, seg=None, seg_prev=None):
        """seg (B, L): segment number of each position (BOS count so far), or None; seg_prev (B,): the segment of the
        position just before the chunk. With seg, a write reaches a read only inside one segment (reset at BOS)."""
        B, h, L, _ = siw.shape
        A = torch.zeros(B, h, L, self.M, device=Vc.device, dtype=torch.float32).scatter_add(3, siw, ww)
        Rq = torch.zeros(B, h, L, self.M, device=Vc.device, dtype=torch.float32).scatter_add(3, sir, wr)
        pos = torch.arange(L, device=Vc.device, dtype=torch.float32)
        e = pos[:, None] - 1 - pos[None, :]  # t - 1 - s
        mask = (e >= 0).float() if not self._leak else torch.ones_like(e)
        lam_ = lam[None, :, None, None]
        D = torch.pow(lam_, e.clamp(min=0)) * mask  # lam^(t-1-s) for s < t
        fade_in = torch.pow(lam_, pos[None, None, :, None])  # the carried state, faded (t - c0) times
        tail = torch.pow(lam_, (L - 1 - pos)[None, None, :, None])  # each write's fade by the chunk's end
        keep = 1.0
        if seg is not None:
            D = D * (seg[:, :, None] == seg[:, None, :]).float()[:, None]  # (B, 1, L, L): same segment only
            fade_in = fade_in * (seg == seg_prev[:, None]).float()[:, None, :, None]  # no BOS since the chunk began
            tail = tail * (seg == seg[:, -1:]).float()[:, None, :, None]  # writes after the chunk's last BOS survive
            keep = (seg[:, -1] == seg_prev).float()[:, None, None]  # the carried state survives a chunk with no BOS
        ov = torch.matmul(Rq, A.transpose(-1, -2)) * D  # (B, h, L, L)
        num = torch.matmul(ov, Vc) + torch.matmul(Rq, mem) * fade_in
        den = ov.sum(-1, keepdim=True) + torch.matmul(Rq, cnt[..., None]) * fade_in
        At = A * tail
        keep_m = keep if seg is None else keep[..., None]  # (B, 1, 1, 1) for mem, (B, 1, 1) for cnt
        mem = mem * torch.pow(lam_, float(L)) * keep_m + torch.matmul(At.transpose(-1, -2), Vc)
        cnt = cnt * torch.pow(lam[None, :, None], float(L)) * keep + At.sum(2)
        return num, den, mem, cnt

    def finish(self, num, den, B, T, dtype):
        r = num / (den + self.read_eps) if self.read_mode == "mean" else num
        if self.out_norm:
            r = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + self.out_norm_eps) * self.onorm_w[None, :, None, :]
        return self.wo(r.transpose(1, 2).reshape(B, T, self.d)).to(dtype)

    def forward(self, x, reset=None):
        """reset (B, T) bool or None: where True the memory is emptied BEFORE that position reads (a BOS starts a
        new document). Every call starts from an empty memory: no state is carried from one window to the next."""
        B, T, D = x.shape
        if not self.enabled:
            return torch.zeros_like(x)
        siw, ww, sir, wr, V = self.prepare(x)
        lam = self.decay()
        mem = V.new_zeros(B, self.h, self.M, self.dh)
        cnt = V.new_zeros(B, self.h, self.M)
        ckpt = torch.is_grad_enabled() and T > self.chunk
        seg = None if reset is None else torch.cumsum(reset.long(), 1)  # (B, T): BOS count up to and including t
        nums, dens = [], []
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            args = (siw[:, :, c0:c1], ww[:, :, c0:c1], sir[:, :, c0:c1], wr[:, :, c0:c1], V[:, :, c0:c1], mem, cnt, lam)
            if seg is not None:  # at c0 = 0 the state is empty, so seg_prev only has to differ from nothing
                args = args + (seg[:, c0:c1], seg[:, c0 - 1] if c0 > 0 else torch.zeros_like(seg[:, 0]))
            if ckpt:
                n_, d_, mem, cnt = torch.utils.checkpoint.checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                n_, d_, mem, cnt = self._chunk(*args)
            nums.append(n_)
            dens.append(d_)
        self.last_state = (mem.detach(), cnt.detach())
        return self.finish(torch.cat(nums, 2), torch.cat(dens, 2), B, T, x.dtype)

    def forward_sequential(self, x, reset=None):
        """The recurrence, token by token, with sparse gathers and scatters: (empty at a reset,) read, then write."""
        B, T, D = x.shape
        if not self.enabled:
            return torch.zeros_like(x)
        siw, ww, sir, wr, V = self.prepare(x)
        lam = self.decay()[None, :, None]
        mem = V.new_zeros(B, self.h, self.M, self.dh)
        cnt = V.new_zeros(B, self.h, self.M)
        nums, dens = [], []
        for t in range(T):
            if reset is not None:
                live = (~reset[:, t]).to(mem.dtype)[:, None, None]  # 0 where position t is a BOS
                mem, cnt = mem * live[..., None], cnt * live
            rows = sir[:, :, t]  # (B, h, k)
            got = mem.gather(2, rows[..., None].expand(-1, -1, -1, self.dh))
            nums.append((wr[:, :, t, :, None] * got).sum(2))
            dens.append((wr[:, :, t] * cnt.gather(2, rows)).sum(-1, keepdim=True))
            wrow = siw[:, :, t]
            add = torch.zeros_like(mem).scatter_add(2, wrow[..., None].expand(-1, -1, -1, self.dh),
                                                    ww[:, :, t, :, None] * V[:, :, t, None, :])
            mem = mem * lam[..., None] + add
            cnt = cnt * lam + torch.zeros_like(cnt).scatter_add(2, wrow, ww[:, :, t])
        self.last_state = (mem.detach(), cnt.detach())
        return self.finish(torch.stack(nums, 2), torch.stack(dens, 2), B, T, x.dtype)


class Archive(nn.Module):
    """OFF by default. Exact compressed vectors in a growing buffer, found through pointers kept in an SDM."""

    def __init__(self, d, d_c=64, n_sub=16, k=4, slot_keep=4, window=16, n_fetch=4, weighting="soft", seed=0):
        super().__init__()
        assert weighting in ("soft", "st") and d_c % 2 == 0
        g = torch.Generator().manual_seed(8765 + seed)
        self.d, self.d_c, self.n_sub, self.M, self.k = d, d_c, n_sub, n_sub * n_sub, k
        self.P, self.W, self.n_fetch, self.weighting = slot_keep, window, n_fetch, weighting
        self._leak = False  # TEST ONLY: adds the NEXT position to the window (the causality control)
        self.norm = RMSNorm(d)
        self.wk = nn.Linear(d, d_c, bias=False)
        self.wq = nn.Linear(d, d_c, bias=False)
        self.wc = nn.Linear(d, d_c, bias=False)
        self.wo = nn.Linear(d_c, d, bias=False)
        # the pointer SDM's sub-keys: fixed (the slot choice is discrete, so they could get no gradient anyway)
        self.register_buffer("keys1", torch.randn(1, n_sub, d_c // 2, generator=g))
        self.register_buffer("keys2", torch.randn(1, n_sub, d_c // 2, generator=g))

    def reset_init(self):
        nn.init.zeros_(self.wo.weight)

    @torch.no_grad()
    def slots(self, z):
        B, T, _ = z.shape
        return pk_address(z.reshape(B * T, 1, self.d_c).float(), self.keys1, self.keys2, self.k)[1].view(B, T, -1)

    def _window(self, T, device):
        t = torch.arange(T, device=device)[:, None]
        r = torch.arange(self.W, device=device)[None, :]
        w = t - 1 - r
        if self._leak:
            w = torch.cat([w, t + 1], 1)
        return torch.where((w >= 0) & (w < T), w, torch.full_like(w, -1))  # (T, W)

    @torch.no_grad()
    def candidates_parallel(self, Sa, Sb, T):
        """Sa, Sb (B, T, k) write and read slots -> candidate positions (B, T, C), -1 where empty, sorted."""
        B, _, k = Sa.shape
        dev = Sa.device
        base = (torch.arange(B, device=dev)[:, None, None] * self.M + Sa) * (T + 1)
        keys = (base + torch.arange(T, device=dev)[None, :, None]).reshape(-1)
        srt, order = keys.sort()
        spos = torch.arange(T, device=dev)[None, :, None].expand(B, T, k).reshape(-1)[order]
        seg = (torch.arange(B, device=dev)[:, None, None] * self.M + Sb) * (T + 1)  # (B, T, k)
        e = torch.searchsorted(srt, (seg + torch.arange(T, device=dev)[None, :, None]).reshape(-1)).view(B, T, k)
        s0 = torch.searchsorted(srt, seg.reshape(-1)).view(B, T, k)
        idx = e[..., None] - 1 - torch.arange(self.P, device=dev)  # (B, T, k, P): the last P writers before t
        ok = idx >= s0[..., None]
        cand = torch.where(ok, spos[idx.clamp(min=0)], torch.full_like(idx, -1)).reshape(B, T, k * self.P)
        cand = torch.cat([cand, self._window(T, dev)[None].expand(B, -1, -1)], -1)
        return self._dedupe(cand)

    @staticmethod
    def _dedupe(cand):
        cand = cand.sort(-1).values
        dup = torch.zeros_like(cand, dtype=torch.bool)
        dup[..., 1:] = cand[..., 1:] == cand[..., :-1]
        return torch.where(dup, torch.full_like(cand, -1), cand)

    def read(self, q, kz, cz, cand):
        """q (B, T, d_c); kz, cz (B, S, d_c) stored keys and vectors; cand (B, T, C) positions into S (or -1)."""
        B, T, C = cand.shape
        valid = cand >= 0
        ci = cand.clamp(min=0)
        kg = kz.gather(1, ci.reshape(B, T * C, 1).expand(-1, -1, self.d_c)).view(B, T, C, self.d_c)
        sc = (q[:, :, None, :] * kg).sum(-1) / math.sqrt(self.d_c)
        sc = sc.masked_fill(~valid, -1e9)
        nf = min(self.n_fetch, C)
        top, ti = sc.topk(nf, -1)
        tvalid = valid.gather(-1, ti)
        w = torch.softmax(top, -1) * tvalid.float()
        if self.weighting == "st":  # hard choice of the best in the forward, the soft gradient in the backward
            hard = F.one_hot(top.argmax(-1), nf).float() * tvalid.float()
            w = hard + w - w.detach()
        pos = ci.gather(-1, ti)
        cg = cz.gather(1, pos.reshape(B, T * nf, 1).expand(-1, -1, self.d_c)).view(B, T, nf, self.d_c)
        self.last_n_valid = valid.sum(-1).float().mean().detach()
        return (w[..., None] * cg).sum(2)

    def _proj(self, x):
        hx = self.norm(x).float()
        return self.wk(hx), self.wq(hx), self.wc(hx)

    def forward(self, x):
        B, T, _ = x.shape
        kz, qz, cz = self._proj(x)
        cand = self.candidates_parallel(self.slots(kz), self.slots(qz), T)
        return self.wo(self.read(qz, kz, cz, cand)).to(x.dtype)

    def forward_sequential(self, x):
        """Ring buffers of pointers, token by token: read (window + the P pointers of each woken slot), then write."""
        B, T, _ = x.shape
        kz, qz, cz = self._proj(x)
        Sa, Sb = self.slots(kz), self.slots(qz)
        ring = torch.full((B, self.M, self.P), -1, dtype=torch.long, device=x.device)
        ptr = torch.zeros(B, self.M, dtype=torch.long, device=x.device)
        win = self._window(T, x.device)
        bi = torch.arange(B, device=x.device)[:, None]
        outs = []
        for t in range(T):
            got = ring[bi, Sb[:, t]].reshape(B, -1)  # (B, k * P)
            cand = self._dedupe(torch.cat([got, win[t][None].expand(B, -1)], -1))
            outs.append(self.read(qz[:, t:t + 1], kz, cz, cand[:, None]))
            m = Sa[:, t]  # (B, k): this position's pointer goes into each of its slots' rings
            p = ptr[bi, m]
            ring[bi, m, p] = t
            ptr[bi, m] = (p + 1) % self.P
        return self.wo(torch.cat(outs, 1)).to(x.dtype)


class TrainedSdm(nn.Module):
    """The MLP replacement: a trained product-key table, read once per token. It never mixes positions."""

    def __init__(self, d, n_sub, heads=4, d_a=32, k=32, scale=8.0, learn_scale=True, seed=0):
        super().__init__()
        # heads * d_a <= d: the query map never widens the stream, so no dense hidden expansion hides in here
        assert d % heads == 0 and d_a % 2 == 0 and heads * d_a <= d
        g = torch.Generator().manual_seed(9876 + seed)
        self.d, self.h, self.dh, self.d_a, self.k = d, heads, d // heads, d_a, k
        self.n_sub, self.M = n_sub, n_sub * n_sub
        self.norm = RMSNorm(d)
        self.wq = nn.Linear(d, heads * d_a, bias=False)
        self.keys1 = nn.Parameter(torch.randn(heads, n_sub, d_a // 2, generator=g))
        self.keys2 = nn.Parameter(torch.randn(heads, n_sub, d_a // 2, generator=g))
        # one table per head: M slots of width dh; the heads' reads are concatenated back to width d
        self.values = nn.Parameter(torch.zeros(heads, self.M, self.dh))
        self.log_scale = nn.Parameter(torch.full((heads,), math.log(scale)), requires_grad=learn_scale)
        self.wo = nn.Linear(d, d, bias=False)

    def reset_init(self):
        # ZERO START: every value is zero, so every read is exactly zero and wo(0) = 0. At init the layer adds
        # nothing to the residual (the same rule as RunTimeSdm's zero output map). The self-test checks it exactly.
        nn.init.zeros_(self.values)

    def address(self, x):
        """x (B, T, d) -> soft weights (B*T, h, k) and slot ids (B*T, h, k). Exact product-key top-k."""
        B, T, _ = x.shape
        q = self.wq(self.norm(x).float()).float().view(B * T, self.h, self.d_a)
        # the top-k PICK is hard (no gradient through which slots wake); the gradient reaches the query map and
        # the sub-keys only through the scores of the slots that were picked, via these soft weights
        sv, si = pk_address(q, self.keys1, self.keys2, self.k)
        w = torch.softmax(sv * self.log_scale.exp()[None, :, None], -1)
        return w, si

    def forward(self, x):
        B, T, D = x.shape
        w, si = self.address(x)
        kk = si.shape[-1]
        flat = (si + torch.arange(self.h, device=si.device)[None, :, None] * self.M).reshape(-1, kk)
        # a weighted sum of the picked rows without building (N, h, k, dh): gradient reaches the rows and the weights
        r = F.embedding_bag(flat, self.values.reshape(-1, self.dh).float(), per_sample_weights=w.reshape(-1, kk).float(),
                            mode="sum")
        self.last_si = si.detach()
        return self.wo(r.view(B, T, D)).to(x.dtype)


def trained_sdm_params(d, n_sub, heads, d_a):
    """Parameter count of one TrainedSdm: norm, query map, sub-keys, value tables, scales, output map."""
    return d + d * heads * d_a + heads * n_sub * d_a + heads * n_sub * n_sub * (d // heads) + heads + d * d


def matched_sdm_n_sub(d, mlp_f, heads, d_a):
    """The n_sub whose TrainedSdm weight count is closest to one SwiGLU(d, mlp_f) plus its RMSNorm."""
    target = 3 * d * mlp_f + d
    return min(range(2, 4097), key=lambda n: abs(trained_sdm_params(d, n, heads, d_a) - target))


class OneSdmLM(nn.Module):
    kind = "onesdm"

    def __init__(self, V, d=256, layers=4, mlp_f=688, mem=True, heads=4, n_sub=32, d_a=32, k=16,
                 decays=(0.0, 0.8, 0.99, 1.0), learn_decay=False, write_gate=True, gate_bias=0.0, read_mode="mean",
                 out_norm=True, addr_scale=8.0, learn_scale=True, chunk=128, mem_in_grad=1.0, out_norm_eps=1e-2,
                 read_eps=1e-3, addr_bias=0.0, archive=False,
                 arc_layers=None, arc_d_c=64, arc_n_sub=16, arc_k=4, arc_slot_keep=4, arc_window=16, arc_n_fetch=4,
                 arc_weighting="soft", tie=True, ffn="mlp", sdm_n_sub=0, sdm_heads=4, sdm_d_a=0, sdm_k=32,
                 sdm_scale=8.0, sdm_learn_scale=True, reset_at_bos=False, bos_id=0, seed=0):
        super().__init__()
        assert ffn in ("mlp", "sdm", "none")
        assert not (reset_at_bos and archive), "the archive has no reset at BOS"
        self.V, self.d, self.n_layers, self.mlp_f, self.tie = V, d, layers, mlp_f, tie
        # reset_at_bos: every run-time memory is emptied at each BOS token inside a window (off by default)
        self.reset_at_bos, self.bos_id = reset_at_bos, bos_id
        self.ffn = ffn
        self.has_mem, self.has_arc = mem, archive
        self.emb = nn.Embedding(V, d)
        self.mems = nn.ModuleList([RunTimeSdm(d, heads, n_sub, d_a, k, tuple(decays), learn_decay, write_gate,
                                              gate_bias, read_mode, out_norm, addr_scale, learn_scale, chunk,
                                              read_eps=read_eps, mem_in_grad=mem_in_grad, out_norm_eps=out_norm_eps,
                                              addr_bias=addr_bias,
                                              seed=seed + 31 * i) for i in range(layers)])
        for m in self.mems:
            m.enabled = mem
        self.arc_layers = tuple(range(1, layers)) if arc_layers is None else tuple(arc_layers)
        if archive:
            self.arcs = nn.ModuleDict({str(i): Archive(d, arc_d_c, arc_n_sub, arc_k, arc_slot_keep, arc_window,
                                                       arc_n_fetch, arc_weighting, seed=seed + 7 * i)
                                       for i in self.arc_layers})
        if ffn == "mlp":  # the per-token step of the onesdm arm: a SwiGLU MLP (built exactly as before)
            self.mnorms = nn.ModuleList([RMSNorm(d) for _ in range(layers)])
            self.mlps = nn.ModuleList([SwiGLU(d, mlp_f) for _ in range(layers)])
        elif ffn == "sdm":  # the allsdm arms: a trained product-key table in place of every MLP
            self.sdm_d_a = sdm_d_a or min(32, (d // sdm_heads) // 2 * 2)
            self.sdm_n_sub = sdm_n_sub or matched_sdm_n_sub(d, mlp_f, sdm_heads, self.sdm_d_a)
            self.sdms = nn.ModuleList([TrainedSdm(d, self.sdm_n_sub, sdm_heads, self.sdm_d_a, sdm_k, sdm_scale,
                                                  sdm_learn_scale, seed=seed + 53 * i) for i in range(layers)])
        self.nf = RMSNorm(d)
        if not tie:
            self.head = nn.Linear(d, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in n and "onorm_w" not in n and "addr_b" not in n:
                out_map = n.endswith("down.weight") or (n.startswith("sdms.") and n.endswith("wo.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers) if out_map else 0.02)
        for m in self.mems:
            m.reset_init()
        for s in getattr(self, "sdms", []):
            s.reset_init()
        for a in getattr(self, "arcs", {}).values():
            a.reset_init()

    def enfold(self, idx):
        return self.emb(idx)

    def project(self, h):
        return h @ self.emb.weight.t() if self.tie else self.head(h)

    def unfold(self, x):
        return self.project(self.nf(x))

    def residual(self, idx, sequential=False):
        x = self.enfold(idx)
        reset = (idx == self.bos_id) if self.reset_at_bos else None
        for i in range(self.n_layers):
            m = self.mems[i]
            if m.enabled:
                x = x + (m.forward_sequential(x, reset) if sequential else m(x, reset))
            if self.has_arc and str(i) in self.arcs:
                a = self.arcs[str(i)]
                x = x + (a.forward_sequential(x) if sequential else a(x))
            if self.ffn == "mlp":
                x = x + self.mlps[i](self.mnorms[i](x))
            elif self.ffn == "sdm":
                x = x + self.sdms[i](x)  # per token: the same in the parallel and the sequential form
        return x

    def hidden(self, idx, sequential=False):
        """The final normalised state the tied head reads (the text trainers' convention: logits = hidden @ E^T)."""
        return self.nf(self.residual(idx, sequential))

    def forward(self, idx, sequential=False):
        return self.project(self.hidden(idx, sequential))

    def memory_parameters(self):
        return {n for n, _ in self.named_parameters() if n.startswith("mems.") or n.startswith("arcs.")}

    def set_leak(self, on):
        for m in self.mems:
            m._leak = on
        for a in getattr(self, "arcs", {}).values():
            a._leak = on


def causal_ema_linear(x, decay, chunk=256):
    """Same numbers as causal_ema (a normalised causal moving average) in O(T * chunk) instead of O(T^2)."""
    B, T, D = x.shape
    out = []
    S = x.new_zeros(B, D)
    pos = torch.arange(chunk, device=x.device, dtype=torch.float32)
    lag = pos[:, None] - pos[None, :]
    Wm = torch.where(lag >= 0, torch.pow(torch.tensor(decay, device=x.device), lag.clamp(min=0)), torch.zeros_like(lag))
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        L = c1 - c0
        carry = torch.pow(torch.tensor(decay, device=x.device), pos[:L] + 1)
        y = torch.einsum("ts,bsd->btd", Wm[:L, :L].to(x.dtype), x[:, c0:c1]) + carry[None, :, None].to(x.dtype) * S[:, None]
        tpos = torch.arange(c0, c1, device=x.device, dtype=torch.float32)
        Z = (1 - torch.pow(torch.tensor(decay, device=x.device), tpos + 1)) / (1 - decay) if decay < 1 else tpos + 1
        out.append(y / Z[None, :, None].to(x.dtype))
        S = y[:, -1]
    return torch.cat(out, 1)


def use_linear_ema():
    SO.causal_ema = causal_ema_linear  # SdmOnlyLM.features looks this name up in its own module at call time


DEFAULTS = dict(d=256, layers=4, mlp_f=688, heads=4, n_sub=32, d_a=32, k=16)
ALLSDM_ARMS = ("onesdm_allsdm", "onesdm_allsdm_big")


def lm_kwargs(cfg):
    """cfg -> (merged cfg, the OneSdmLM keyword arguments it carries)."""
    c = {**DEFAULTS, **cfg}
    mem_keys = ("heads", "n_sub", "d_a", "k", "learn_decay", "write_gate", "gate_bias", "read_mode", "out_norm",
                "addr_scale", "learn_scale", "chunk", "mem_in_grad", "out_norm_eps", "read_eps", "addr_bias", "tie", "seed",
                "reset_at_bos", "bos_id")
    arc_keys = ("arc_layers", "arc_d_c", "arc_n_sub", "arc_k", "arc_slot_keep", "arc_window", "arc_n_fetch", "arc_weighting")
    sdm_keys = ("sdm_heads", "sdm_d_a", "sdm_k", "sdm_scale", "sdm_learn_scale")
    kw = {kk: c[kk] for kk in mem_keys + arc_keys + sdm_keys if kk in c}
    if "decays" in c:
        kw["decays"] = tuple(c["decays"])
    return c, kw


def allsdm_n_sub(arm, c):
    """n_sub of the trained table: matched to the MLP's weights, or 4x that (16x the slots) for the big arm."""
    heads = c.get("sdm_heads", 4)
    d_a = c.get("sdm_d_a", 0) or min(32, (c["d"] // heads) // 2 * 2)
    n = c.get("sdm_n_sub", 0) or matched_sdm_n_sub(c["d"], c["mlp_f"], heads, d_a)
    return n * c.get("sdm_big_mult", 4) if arm == "onesdm_allsdm_big" else n


def build_onesdm(arm, V, cfg):
    """onesdm | onesdm_archive | plain (memory off) | onesdm_allsdm | onesdm_allsdm_big | sdmonly | yardstick_transformer."""
    c, kw = lm_kwargs(cfg)
    if arm in ("onesdm", "onesdm_archive", "plain"):
        return OneSdmLM(V, d=c["d"], layers=c["layers"], mlp_f=c["mlp_f"], mem=arm != "plain",
                        archive=arm == "onesdm_archive", **kw)
    if arm in ALLSDM_ARMS:  # the run-time memory exactly as onesdm has it; a trained SDM in place of every MLP
        return OneSdmLM(V, d=c["d"], layers=c["layers"], mlp_f=c["mlp_f"], mem=True, archive=False, ffn="sdm",
                        sdm_n_sub=allsdm_n_sub(arm, c), **kw)
    if arm == "sdmonly":
        use_linear_ema()
        return SO.SdmOnlyLM(V, d=c["d"], d_a=c.get("sdm_d_a", 64), n_sub=c.get("sdm_n_sub", 32), k=c["k"],
                            hops=c["layers"], hop_mlp=c.get("sdm_hop_mlp", 512), seed=c.get("seed", 0))
    if arm == "yardstick_transformer":  # a YARDSTICK only: softmax attention is not allowed in the model itself
        return QwenLM(V, d=c["d"], n_layer=c["layers"], n_head=c.get("yard_heads", 4), n_kv=c.get("yard_kv", 2),
                      f=c["mlp_f"], T=c.get("max_len", 20000))
    raise ValueError(arm)


def param_count(model):
    emb = model.emb.weight.numel()
    return {"non_embedding": sum(p.numel() for p in model.parameters()) - emb, "embedding": emb}


def decode_state_bytes(model, T):
    """Bytes kept to continue decoding after T tokens (float32), excluding the weights."""
    d = model.d
    if isinstance(model, OneSdmLM):
        b = 0
        if model.has_mem:
            m = model.mems[0]
            b += model.n_layers * m.h * m.M * (m.dh + 1) * 4
        if model.has_arc:
            for a in model.arcs.values():
                b += T * 2 * a.d_c * 4 + a.M * a.P * 8  # the growing buffer, the fixed pointer rings
        return b
    if isinstance(model, SO.SdmOnlyLM):
        return (model.n_back + len(model.decays)) * d * 4  # the back tokens and one running sum per fade
    if isinstance(model, QwenLM):
        return model.n_layer * 2 * T * model.n_kv * (d // model.n_head) * 4  # the key-value cache
    raise ValueError(type(model))


def body_violations(model):
    """Everything in a model that is attention or an MLP: a SwiGLU, an attention module, or a Linear in the layer
    body (anything but the token table and the head) whose out_features exceed the width d (a dense expansion)."""
    bad = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if isinstance(mod, (SwiGLU, nn.MultiheadAttention)) or "Attention" in cls or "Attn" in cls or "Qwen" in cls:
            bad.append(f"{name or '<root>'}: {cls}")
        elif isinstance(mod, nn.Linear) and name != "head" and mod.out_features > model.d:
            bad.append(f"{name}: Linear {mod.in_features} -> {mod.out_features}")
    return bad


def _causal_ok(model, idx, p, train_mode):
    model.train(train_mode)
    with torch.no_grad():
        idx2 = idx.clone()
        idx2[:, p] = (idx2[:, p] + 5) % model.V
        a, b = model(idx), model(idx2)
    return torch.allclose(a[:, :p], b[:, :p], atol=1e-5), not torch.allclose(a[:, p:], b[:, p:], atol=1e-5)


def selftest():
    torch.manual_seed(0)
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name, flush=True)

    def rel(a, b):
        return float((a - b).abs().max() / (a.abs().max() + 1e-12))

    V = 50
    # 1. product-key addressing equals a brute-force search over all n_sub^2 slots
    k1, k2 = torch.randn(2, 9, 6), torch.randn(2, 9, 6)
    z = torch.randn(30, 2, 12)
    sv, si = pk_address(z, k1, k2, 5)
    full = ((torch.einsum("nhd,hsd->nhs", F.normalize(z[..., :6], dim=-1), F.normalize(k1, dim=-1))[..., :, None]
             + torch.einsum("nhd,hsd->nhs", F.normalize(z[..., 6:], dim=-1), F.normalize(k2, dim=-1))[..., None, :]) * 0.5).flatten(-2)
    bv, bi = full.topk(5, -1)
    check("product-key top-k equals brute force (scores and slots)",
          torch.allclose(sv, bv, atol=1e-6) and bool((si.sort(-1).values == bi.sort(-1).values).all()))
    cfg = dict(d=32, layers=3, mlp_f=48, heads=4, n_sub=6, d_a=8, k=5, decays=(0.0, 0.7, 0.95, 1.0), chunk=7)
    idx = torch.randint(0, V, (2, 23))
    # 2. shapes and the zero start: the memory is exactly a no-op at init (equals the plain stack)
    m = build_onesdm("onesdm", V, cfg)
    pl = build_onesdm("plain", V, cfg)
    pl.load_state_dict(m.state_dict())
    with torch.no_grad():
        out = m(idx)
        check("logits shape", out.shape == (2, 23, V))
        check("zero start: memory on at init gives the plain stack's output exactly", torch.equal(out, pl(idx)))
    # give the memory a non-zero output map, a non-trivial gate and learned decays for the remaining tests
    ml = build_onesdm("onesdm", V, {**cfg, "learn_decay": True})
    for mm in list(m.mems) + list(ml.mems):
        nn.init.normal_(mm.wo.weight, std=0.3)
        nn.init.normal_(mm.wg.weight, std=0.5)
    with torch.no_grad():
        check("memory changes the output once its output map is non-zero", not torch.allclose(m(idx), pl(idx), atol=1e-5))
    # 3. causality, train and eval mode, with a leaking control that must fail
    for mode in (True, False):
        c_ok, later = _causal_ok(m, idx, 12, mode)
        check(f"causal ({'train' if mode else 'eval'} mode): earlier outputs ignore a later token; later ones change", c_ok and later)
    m.set_leak(True)
    c_ok, _ = _causal_ok(m, idx, 12, False)
    check("causality control: the leaking memory FAILS the causality check", not c_ok)
    m.set_leak(False)
    # 4. chunked parallel = token-by-token recurrence (outputs and gradients), fixed and learned fades
    for model, tag in ((m, "fixed fades incl. 0 and 1"), (ml, "learned fades")):
        model.train()
        e1 = model.enfold(idx).detach().requires_grad_(True)
        e2 = e1.detach().clone().requires_grad_(True)
        mm = model.mems[0]
        y1, y2 = mm(e1), mm.forward_sequential(e2)
        gy = torch.randn_like(y1)
        (y1 * gy).sum().backward()
        g1 = {n: p.grad.clone() for n, p in mm.named_parameters() if p.grad is not None}
        mm.zero_grad()
        (y2 * gy).sum().backward()
        g2 = {n: p.grad.clone() for n, p in mm.named_parameters() if p.grad is not None}
        check(f"chunked (7) = sequential ({tag}): output", torch.allclose(y1, y2, atol=1e-5, rtol=1e-4))
        check(f"chunked (7) = sequential ({tag}): gradient of input and every parameter",
              rel(e1.grad, e2.grad) < 1e-5 and g1.keys() == g2.keys() and all(rel(g1[n], g2[n]) < 1e-5 for n in g1))
        mm.zero_grad()
    with torch.no_grad():
        m.eval()
        ref = m(idx, sequential=True)
        res = []
        for ch in (1, 5, 23, 64):
            for mm in m.mems:
                mm.chunk = ch
            res.append(torch.allclose(m(idx), ref, atol=1e-5, rtol=1e-4))
        check("whole model: chunks 1, 5, 23, 64 all equal the sequential recurrence", all(res))
        for mm in m.mems:
            mm.chunk = 7
    # 5. the write gate: a gate near zero for one token removes its write; reads at the next token change
    with torch.no_grad():
        mm = m.mems[0]
        x = torch.randn(1, 10, 32)
        base = mm(x)
        check("write gate lies in [0, 1]", float(mm.last_gate.min()) >= 0 and float(mm.last_gate.max()) <= 1)
        mm.write_gate = False
        ungated = mm(x)
        mm.write_gate = True
        check("write gate changes the read (gate on vs gate fixed at 1)", not torch.allclose(base, ungated, atol=1e-5))
        nn.init.zeros_(mm.wg.weight)
        nn.init.constant_(mm.wg.bias, -30.0)  # gate about 1e-13: nothing is written
        y0 = mm(x)
        check("write gate at about zero: nothing is written, every read is (near) zero before the out map",
              float(y0.abs().max()) < 1e-6)
        nn.init.normal_(mm.wg.weight, std=0.5)
        nn.init.zeros_(mm.wg.bias)
    # 6. the previous-token head: fade 0 with one shared address for every token reads exactly the last write
    with torch.no_grad():
        r = RunTimeSdm(8, heads=1, n_sub=4, d_a=4, k=4, decays=(0.0,), write_gate=False, out_norm=False, chunk=3)
        nn.init.zeros_(r.wq.weight)
        nn.init.zeros_(r.wk.weight)  # every address is the same: k slots, weight 1/k each
        nn.init.eye_(r.wo.weight)
        xx = torch.randn(1, 9, 8)
        Vv = r.prepare(xx)[-1]
        y = r(xx)
        den = 1.0 / r.k  # sum over k slots of (1/k read weight) x (1/k written count)
        want = Vv[0, 0, :-1] * den / (den + r.read_eps)  # what position t - 1 wrote, through the mean read
        check("fade 0 is a previous-token head: position t reads the value written at t - 1 (and t 0 reads zero)",
              torch.allclose(y[0, 1:], want, atol=1e-5) and float(y[0, 0].abs().max()) == 0.0)
        rb = RunTimeSdm(64, heads=1, n_sub=8, d_a=16, k=8, decays=(0.0,), write_gate=False, out_norm=False, addr_bias=100.0)
        nn.init.eye_(rb.wo.weight)
        xb = torch.randn(1, 30, 64)
        yb, Vb = rb(xb), rb.prepare(xb)[-1]
        cos = F.cosine_similarity(yb[0, 1:], Vb[0, 0, :-1], dim=-1)
        check("addr_bias 100: at init the fade-0 head is a previous-token head (cosine to v[t-1] > 0.99 everywhere)",
              float(cos.min()) > 0.99)
    # 7. the archive: parallel = ring buffers, causal, zero start, leaking control fails
    acfg = {**cfg, "arc_d_c": 8, "arc_n_sub": 4, "arc_k": 3, "arc_slot_keep": 2, "arc_window": 3, "arc_n_fetch": 3}
    ma = build_onesdm("onesdm_archive", V, acfg)
    mb = build_onesdm("onesdm", V, cfg)
    mb.load_state_dict({kk: v for kk, v in ma.state_dict().items() if kk in mb.state_dict()})
    with torch.no_grad():
        check("archive: zero start equals the model without it", torch.equal(ma(idx), mb(idx)))
        for a in ma.arcs.values():
            nn.init.normal_(a.wo.weight, std=0.3)
        for mm in ma.mems:
            nn.init.normal_(mm.wo.weight, std=0.3)
        a1 = ma.arcs["1"]
        xx = torch.randn(2, 40, 32)
        kz, qz, cz = a1._proj(xx)
        cp = a1.candidates_parallel(a1.slots(kz), a1.slots(qz), 40)
        check("archive: every candidate is strictly earlier", bool(((cp < torch.arange(40)[None, :, None]) | (cp < 0)).all()))
        check("archive: parallel = ring-buffer recurrence (one layer)", torch.allclose(a1(xx), a1.forward_sequential(xx), atol=1e-5))
        check("archive: parallel = recurrence (whole model)", torch.allclose(ma(idx), ma(idx, sequential=True), atol=1e-5, rtol=1e-4))
    for mode in (True, False):
        c_ok, later = _causal_ok(ma, idx, 12, mode)
        check(f"archive model causal ({'train' if mode else 'eval'} mode)", c_ok and later)
    ma.set_leak(True)
    for mm in ma.mems:
        mm._leak = False  # only the archive leaks in this control
    for a in ma.arcs.values():
        a.n_fetch = 64  # fetch every candidate, so the leaked next position is always used
    c_ok, _ = _causal_ok(ma, idx, 12, False)
    check("archive causality control: a window that includes the next position FAILS", not c_ok)
    ma.set_leak(False)
    for a in ma.arcs.values():
        a.n_fetch = 3
    for wmode in ("soft", "st"):
        for a in ma.arcs.values():
            a.weighting = wmode
        ma.train()
        ma.zero_grad()
        F.cross_entropy(ma(idx).reshape(-1, V), idx.reshape(-1)).backward()
        a1 = ma.arcs["1"]
        check(f"archive ({wmode}): compressed key, query and vector maps receive gradient",
              all(float(getattr(a1, n).weight.grad.abs().sum()) > 0 for n in ("wk", "wq", "wc", "wo")))
    # 8. baselines: today's SdmOnlyLM (linear EMA equals the original; causal in eval), the yardstick
    xe = torch.randn(2, 600, 8)
    check("linear-time EMA equals causal_ema (fades 0.5 and 0.99, T 600)",
          all(torch.allclose(causal_ema(xe, dc), causal_ema_linear(xe, dc), atol=1e-5) for dc in (0.5, 0.99)))
    so = build_onesdm("sdmonly", V, cfg)
    for b in so.store.banks:
        nn.init.normal_(b.values, std=0.1)
    c_ok, later = _causal_ok(so, idx, 12, False)
    check("sdmonly baseline causal in eval mode (its batch norm uses batch statistics in training)", c_ok and later)
    ys = build_onesdm("yardstick_transformer", V, {**cfg, "max_len": 64})
    c_ok, later = _causal_ok(ys, idx, 12, False)
    check("yardstick transformer causal", c_ok and later)
    pc = {a: param_count(build_onesdm(a, 388, {"max_len": 64}))["non_embedding"] for a in
          ("onesdm", "onesdm_archive", "plain", "sdmonly", "yardstick_transformer")}
    print("non-embedding parameters at the default width 256:", pc, flush=True)
    check("default sizes are matched within 30% (onesdm vs sdmonly vs yardstick)",
          max(pc["onesdm"], pc["sdmonly"], pc["yardstick_transformer"]) / min(pc["onesdm"], pc["sdmonly"], pc["yardstick_transformer"]) < 1.3)
    md = build_onesdm("onesdm", 388, {})
    check("decode state is the same at 1k and 16k tokens for onesdm; the yardstick's grows 16x",
          decode_state_bytes(md, 1024) == decode_state_bytes(md, 16384)
          and decode_state_bytes(ys, 16384) == 16 * decode_state_bytes(ys, 1024))
    # 9. the all-SDM arms: a trained SDM (TrainedSdm) in place of every MLP, the run-time SDM unchanged
    ts = TrainedSdm(32, n_sub=12, heads=2, d_a=8, k=7)
    xt = torch.randn(3, 11, 32)
    with torch.no_grad():
        w_t, si_t = ts.address(xt)
        q_t = ts.wq(ts.norm(xt).float()).view(33, 2, 8)
        full_t = ((torch.einsum("nhd,hsd->nhs", F.normalize(q_t[..., :4], dim=-1), F.normalize(ts.keys1, dim=-1))[..., :, None]
                   + torch.einsum("nhd,hsd->nhs", F.normalize(q_t[..., 4:], dim=-1), F.normalize(ts.keys2, dim=-1))[..., None, :])
                  * 0.5).flatten(-2)
        bv_t, bi_t = full_t.topk(7, -1)
    check("TrainedSdm: product-key top-k slots equal a brute-force top-k over all n_sub^2 slots",
          bool((si_t.sort(-1).values == bi_t.sort(-1).values).all()))
    with torch.no_grad():
        nn.init.normal_(ts.values, std=1.0)
        dense = torch.zeros(33, 2, ts.M).scatter(2, bi_t, torch.softmax(bv_t * ts.log_scale.exp()[None, :, None], -1))
        want_t = ts.wo(torch.einsum("nhm,hmd->nhd", dense, ts.values).reshape(3, 11, 32))
    check("TrainedSdm: the sparse read equals a dense softmax-over-brute-force-top-k reference",
          torch.allclose(ts(xt), want_t, atol=1e-5))
    scfg = {**cfg, "sdm_heads": 4, "sdm_d_a": 8, "sdm_k": 5}
    sa = build_onesdm("onesdm_allsdm", V, scfg)
    c_s, kw_s = lm_kwargs(scfg)
    sn = OneSdmLM(V, d=c_s["d"], layers=c_s["layers"], mlp_f=c_s["mlp_f"], ffn="none", **kw_s)
    miss = sn.load_state_dict(sa.state_dict(), strict=False)
    with torch.no_grad():
        check("zero start: onesdm_allsdm at init equals the same model with every TrainedSdm removed (exactly)",
              torch.equal(sa(idx), sn(idx)) and all(kk.startswith("sdms.") for kk in miss.unexpected_keys)
              and not miss.missing_keys and len(miss.unexpected_keys) > 0)
    big_s = build_onesdm("onesdm_allsdm_big", V, scfg)
    viol = {a: body_violations(mo) for a, mo in (("onesdm_allsdm", sa), ("onesdm_allsdm_big", big_s))}
    for a in ALLSDM_ARMS:  # and at the default width 256
        viol[a + "_default"] = body_violations(build_onesdm(a, 388, {}))
    check(f"structure: the allsdm arms (test and default sizes) hold no attention, no MLP, no dense expansion ({viol})",
          not any(viol.values()))
    check("structure control: the same check FAILS on onesdm (its SwiGLU MLPs)", len(body_violations(m)) > 0)
    for mm in sa.mems:
        nn.init.normal_(mm.wo.weight, std=0.3)
        nn.init.normal_(mm.wg.weight, std=0.5)
    with torch.no_grad():
        for s in sa.sdms:
            nn.init.normal_(s.values, std=0.3)
    sa.train()
    sa.zero_grad()
    F.cross_entropy(sa(idx).reshape(-1, V), idx.reshape(-1)).backward()
    s0 = sa.sdms[0]
    gsum = {n: float(getattr(s0, n).grad.abs().sum()) if getattr(s0, n).grad is not None else 0.0
            for n in ("values", "keys1", "keys2")}
    gsum["wq"] = float(s0.wq.weight.grad.abs().sum()) if s0.wq.weight.grad is not None else 0.0
    check(f"gradient: values, both sub-key sets and the query map all receive gradient ({gsum})",
          all(v > 0 for v in gsum.values()))
    for mo, tag in ((sa, "onesdm_allsdm"), (big_s, "onesdm_allsdm_big")):
        if mo is big_s:
            for mm in mo.mems:
                nn.init.normal_(mm.wo.weight, std=0.3)
            with torch.no_grad():
                for s in mo.sdms:
                    nn.init.normal_(s.values, std=0.3)
        for mode in (True, False):
            c_ok, later = _causal_ok(mo, idx, 12, mode)
            check(f"{tag} causal ({'train' if mode else 'eval'} mode)", c_ok and later)
    pc["onesdm_allsdm"] = param_count(build_onesdm("onesdm_allsdm", 388, {}))["non_embedding"]
    pc["onesdm_allsdm_big"] = param_count(build_onesdm("onesdm_allsdm_big", 388, {}))["non_embedding"]
    print(f"non-embedding parameters: onesdm {pc['onesdm']:,} | onesdm_allsdm {pc['onesdm_allsdm']:,} | "
          f"onesdm_allsdm_big {pc['onesdm_allsdm_big']:,} | yardstick {pc['yardstick_transformer']:,}", flush=True)
    check("weights: onesdm_allsdm is within 10% of onesdm",
          abs(pc["onesdm_allsdm"] - pc["onesdm"]) / pc["onesdm"] < 0.10)
    print(f"{sum(ok)} of {len(ok)} checks pass", flush=True)
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
    print("NEXT -> python3 track4_onesdm_models.py --selftest")
