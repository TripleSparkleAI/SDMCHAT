"""SDMONLY fast mixer core: the run-time Kanerva write and read of SdmMix, computed from the index lists.

<claudes_code_comments>
** Function List **
seg_scan_excl(seg, pos, lgv, x, tile, piece) - per segment, the faded sum of the earlier entries (exclusive scan)
_sorted_triples(w, V, si, lg, M) - sort every (location, position, weight) triple by location, then position
_rows(G, T, c, max_triples) - slices of rows (batch row x head) that bound the triples held at once
MixScan.forward(ctx, w, V, si, lg_head, M, tile, piece, max_triples) - num and den of every read, from the triples
MixScan.backward(ctx, g) - recompute the forward scan, run the same scan backwards in time, form dw and dV
mix_scan(w, V, si, lg_head, M, tile, piece, max_triples) - the entry point SdmMixFast calls
selftest() - the scan against a brute-force sum, and a float64 gradcheck of MixScan

** Technical Review **
- The maths it computes (one head, one batch row; the dense SdmMix computes the same thing chunk by chunk):
    A[t, m] = sum_c w[t, c] [si[t, c] = m]                       (2k non-zero entries per position)
    num[t]  = sum_{s<t} decay^(t-s) (sum_m A[t, m] A[s, m]) V[s]
    den[t]  = sum_{s<t} decay^(t-s)  sum_m A[t, m] A[s, m]
  The read is num / (den + 1e-6), formed by the caller.
- Positions t and s meet only through a location both of them picked. So every (location m, position t, weight w)
  triple is listed once (n = B x heads x T x 2k triples), sorted by (batch row and head, location) with a STABLE
  sort, which keeps positions ascending inside each location. A run of equal keys is a SEGMENT: the history of one
  location. For triple i in a segment, R_i = sum over the earlier triples j of the segment of decay^(p_i - p_j) w_j
  [V[p_j], 1] is exactly what location m holds when position p_i reads it (only earlier writes; a position picks a
  location at most once, so "earlier in the segment" equals "earlier in time"). Then [num, den][t] = sum_c w[t, c]
  R_(t, c), an index_add over the triples. No n_sub^2-wide array is built anywhere.
- seg_scan_excl does the segmented, faded, exclusive scan in two levels, with no division and so no overflow:
  (1) tiles of `tile` consecutive triples: inside a tile, P[i, j] = [same segment] [j < i] exp(lg |p_i - p_j|),
      and E = P @ x is one batched matmul (bmm), done in pieces of `piece` tiles to bound memory;
  (2) across tiles: a segment that continues from tile k-1 into tile k carries the state at tile k-1's last triple.
      Carries obey c_k = A_k c_(k-1) + b_k (A_k non-zero only when tile k-1 is wholly inside that one segment),
      solved by a Hillis-Steele scan over the tiles, which stops as soon as every chain has reached its start.
  Every factor is exp(lg x |gap|) <= 1, so a never-fading head (lg 0) and a strongly fading one are both exact.
- Backward: num and den are linear in V and bilinear in w. dL/d[V,1][s] = sum_c w[s, c] Q_(s, c) and
  dL/dw[t, c] = R_(t, c) . g[t] + Q_(t, c) . [V, 1][t], where Q is the same scan run backwards in time with values
  w_j g[p_j] (g = the incoming gradient of [num, den]). The backward recomputes R and runs one reversed scan;
  nothing of size n x dh is kept between forward and backward (the sort is redone in backward).
- Rows (one batch row and one head) never share a segment, so forward and backward run over slices of rows with at
  most `max_triples` triples each (default 2^23); a 262,144-token window then holds one head's triples at a time.
- Autocast is switched off inside, so the scan runs in the dtype it is given (float32 in training, float64 in the
  gradcheck). The dense SdmMix ran its products under bf16 autocast in training; this core is at least as precise.
- Cost per position and head: about 2k x (dh + 1) x (tile + a few) multiply-adds plus a sort of 2k keys; it does not
  depend on n_sub^2 or on the window, except the sort's log factor and the carry scan's log of the longest chain.
Docs: MIXERFAST_RESULTS_2026-10-03.md
</claudes_code_comments>
"""
import sys

import torch


def seg_scan_excl(seg, pos, lgv, x, tile=32, piece=8192):
    """seg (n,) int64, equal values contiguous; pos (n,) positions, monotone inside a segment (either direction);
    lgv (n,) log-decay per entry (<= 0); x (n, D). Returns E (n, D) with
    E_i = sum over the earlier entries j of i's segment of exp(lgv_i |pos_i - pos_j|) x_j."""
    n, D = x.shape
    if n == 0:
        return x.clone()
    npad = -(-n // tile) * tile
    if npad != n:
        pad = npad - n
        seg = torch.cat([seg, seg.new_full((pad,), -1)])
        pos = torch.cat([pos, pos.new_zeros(pad)])
        lgv = torch.cat([lgv, lgv.new_zeros(pad)])
        x = torch.cat([x, x.new_zeros(pad, D)])
    nt = npad // tile
    sT, pT, lT, xT = seg.view(nt, tile), pos.view(nt, tile), lgv.view(nt, tile), x.view(nt, tile, D)
    E = torch.empty(nt, tile, D, dtype=x.dtype, device=x.device)
    lower = torch.ones(tile, tile, dtype=torch.bool, device=x.device).tril(-1)
    for a0 in range(0, nt, piece):  # level 1: inside each tile, one masked faded product
        a1 = min(nt, a0 + piece)
        s, p, lg = sT[a0:a1], pT[a0:a1], lT[a0:a1]
        P = torch.exp(lg[:, :, None] * (p[:, :, None] - p[:, None, :]).abs())
        P = P.masked_fill_(~((s[:, :, None] == s[:, None, :]) & lower), 0.0)
        torch.bmm(P, xT[a0:a1], out=E[a0:a1])
    if nt == 1:
        return E.view(npad, D)[:n]
    # level 2: carries between tiles. c_k = the state of tile k-1's last entry, when tile k starts in its segment.
    first, last, plast, llast = sT[:, 0], sT[:, -1], pT[:, -1], lT[:, -1]
    s_last = E[:, -1] + xT[:, -1]  # inclusive state at each tile's last entry, from inside the tile only
    gate = torch.zeros(nt, dtype=x.dtype, device=x.device)
    gate[1:] = (last[:-1] == first[1:]).to(x.dtype)
    whole = (first == last).to(x.dtype)
    A = torch.zeros(nt, dtype=x.dtype, device=x.device)
    if nt > 2:
        A[2:] = gate[2:] * whole[1:-1] * torch.exp(llast[1:-1] * (plast[1:-1] - plast[:-2]).abs())
    b = torch.zeros(nt, D, dtype=x.dtype, device=x.device)
    b[1:] = gate[1:, None] * s_last[:-1]
    d = 1
    while d < nt:  # Hillis-Steele over the affine maps c -> A c + b; an entry whose A is 0 has reached its start
        A[:d] = 0
        if not bool((A != 0).any()):
            break
        b = torch.cat([b[:d], b[d:] + A[d:, None] * b[:-d]])
        A = torch.cat([A[:d], A[d:] * A[:-d]])
        d *= 2
    pprev = torch.cat([plast.new_zeros(1), plast[:-1]])
    hit = (sT == first[:, None]).to(x.dtype) * torch.exp(lT * (pT - pprev[:, None]).abs())
    E = E + hit[:, :, None] * b[:, None, :]
    return E.view(npad, D)[:n]


def _sorted_triples(w, V, si, lg, M):
    """w, si (G, T, c); V (G, T, dh); lg (G,) log-decay per row (a row = one batch row and head)."""
    G, T, c = w.shape
    dev = w.device
    g = torch.arange(G, device=dev).view(G, 1, 1)
    key = (g * M + si).reshape(-1)
    order = torch.sort(key, stable=True).indices  # stable: positions stay ascending inside each location
    seg = key[order]
    tpos = torch.arange(T, device=dev).view(1, T, 1).expand(G, T, c).reshape(-1)[order]
    gp = (g * T).expand(G, T, c).reshape(-1)[order] + tpos
    lgv = lg.to(w.dtype).view(G, 1, 1).expand(G, T, c).reshape(-1)[order]
    ws = w.reshape(-1)[order]
    Vaug = torch.cat([V, V.new_ones(G, T, 1)], -1).reshape(G * T, -1)
    return order, seg, tpos.to(w.dtype), gp, lgv, ws, Vaug


def _rows(G, T, c, max_triples):
    step = max(1, max_triples // (T * c))  # rows (batch row x head) per slice; segments never cross rows
    return [(g0, min(G, g0 + step)) for g0 in range(0, G, step)]


class MixScan(torch.autograd.Function):
    """[num, den] (B, h, T, dh + 1) of the run-time write and read; gradients for w and V. The rows (batch row x
    head) are processed in slices of at most `max_triples` triples, which bounds memory and changes no number."""

    @staticmethod
    def forward(ctx, w, V, si, lg_head, M, tile, piece, max_triples):
        B, h, T, c = w.shape
        G, dh = B * h, V.shape[-1]
        w2, V2, si2 = w.reshape(G, T, c), V.reshape(G, T, dh), si.reshape(G, T, c)
        lg = lg_head.to(w.dtype)[None].expand(B, h).reshape(G)
        out = w.new_empty(G, T, dh + 1)
        with torch.autocast(device_type=w.device.type, enabled=False):
            for g0, g1 in _rows(G, T, c, max_triples):
                order, seg, pos, gp, lgv, ws, Vaug = _sorted_triples(w2[g0:g1], V2[g0:g1], si2[g0:g1], lg[g0:g1], M)
                R = seg_scan_excl(seg, pos, lgv, ws[:, None] * Vaug[gp], tile, piece)
                out[g0:g1] = Vaug.new_zeros(Vaug.shape).index_add_(0, gp, ws[:, None] * R).view(g1 - g0, T, dh + 1)
                del R, order, seg, pos, gp, lgv, ws, Vaug
        ctx.save_for_backward(w, V, si, lg_head)
        ctx.M, ctx.tile, ctx.piece, ctx.max_triples = M, tile, piece, max_triples
        return out.view(B, h, T, dh + 1)

    @staticmethod
    def backward(ctx, gout):
        w, V, si, lg_head = ctx.saved_tensors
        B, h, T, c = w.shape
        G, dh = B * h, V.shape[-1]
        w2, V2, si2 = w.reshape(G, T, c), V.reshape(G, T, dh), si.reshape(G, T, c)
        lg = lg_head.to(w.dtype)[None].expand(B, h).reshape(G)
        gout2 = gout.reshape(G, T, dh + 1).to(w.dtype)
        dw, dV = torch.empty_like(w2), torch.empty_like(V2)
        with torch.autocast(device_type=w.device.type, enabled=False):
            for g0, g1 in _rows(G, T, c, ctx.max_triples):
                order, seg, pos, gp, lgv, ws, Vaug = _sorted_triples(w2[g0:g1], V2[g0:g1], si2[g0:g1], lg[g0:g1], ctx.M)
                Gs = gout2[g0:g1].reshape((g1 - g0) * T, dh + 1)[gp]
                Vs = Vaug[gp]
                R = seg_scan_excl(seg, pos, lgv, ws[:, None] * Vs, ctx.tile, ctx.piece)
                dws = (R * Gs).sum(-1)
                del R
                Q = seg_scan_excl(seg.flip(0), pos.flip(0), lgv.flip(0), (ws[:, None] * Gs).flip(0),
                                  ctx.tile, ctx.piece).flip(0)  # the same scan, backwards in time
                dws = dws + (Q * Vs).sum(-1)
                del Gs, Vs
                dwr = torch.empty_like(dws)
                dwr[order] = dws
                dw[g0:g1] = dwr.view(g1 - g0, T, c)
                dVaug = torch.zeros_like(Vaug).index_add_(0, gp, ws[:, None] * Q)
                dV[g0:g1] = dVaug[:, :dh].view(g1 - g0, T, dh)
                del Q, dVaug, order, seg, pos, gp, lgv, ws, Vaug
        return dw.view(B, h, T, c), dV.view(B, h, T, dh), None, None, None, None, None, None


try:  # the fused kernels (CUDA only); without Triton every call takes the torch path above
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _seg_fwd_kernel(starts, lens, pos, gp, dest, ws, lgv, V, Y, D: tl.constexpr, BLOCK: tl.constexpr):
        """One program per segment (one location of one row). Walks it in time: R = state (only earlier writes),
        Y[row] += w R (atomic: the 2k triples of a position meet there), then state += w [V[row], 1]. Lane D of the
        state is the denominator."""
        sid = tl.program_id(0)
        start = tl.load(starts + sid).to(tl.int64)
        n = tl.load(lens + sid)
        offs = tl.arange(0, BLOCK)
        lane = offs < D
        lane1 = offs < D + 1
        lg = tl.load(lgv + start)
        p_prev = tl.load(pos + start)
        state = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(0, n):
            i = start + j
            p = tl.load(pos + i)
            state = state * tl.exp(lg * (p - p_prev).to(tl.float32))
            wv = tl.load(ws + i)
            row = tl.load(gp + i).to(tl.int64)
            tl.atomic_add(Y + row * (D + 1) + offs, wv * state, mask=lane1, sem="relaxed")
            v = tl.load(V + row * D + offs, mask=lane, other=0.0)
            v = tl.where(offs == D, 1.0, v)
            state = state + wv * v
            p_prev = p

    @triton.jit
    def _seg_bwd_kernel(starts, lens, pos, gp, dest, ws, lgv, V, Gr, DW, ZQ, D: tl.constexpr, BLOCK: tl.constexpr):
        """Pass 1 (forward in time): DW[dest] = R . G[row]. Pass 2 (backward in time): Q = what later readers of
        this location will pull, DW[dest] += Q . [V[row], 1], ZQ[row] += w Q (atomic, = dV), Q += w G[row]."""
        sid = tl.program_id(0)
        start = tl.load(starts + sid).to(tl.int64)
        n = tl.load(lens + sid)
        offs = tl.arange(0, BLOCK)
        lane = offs < D
        lane1 = offs < D + 1
        lg = tl.load(lgv + start)
        p_prev = tl.load(pos + start)
        state = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(0, n):
            i = start + j
            p = tl.load(pos + i)
            state = state * tl.exp(lg * (p - p_prev).to(tl.float32))
            wv = tl.load(ws + i)
            row = tl.load(gp + i).to(tl.int64)
            g = tl.load(Gr + row * (D + 1) + offs, mask=lane1, other=0.0)
            tl.store(DW + tl.load(dest + i).to(tl.int64), tl.sum(state * g, 0))
            v = tl.load(V + row * D + offs, mask=lane, other=0.0)
            v = tl.where(offs == D, 1.0, v)
            state = state + wv * v
            p_prev = p
        q = tl.zeros([BLOCK], dtype=tl.float32)
        p_next = tl.load(pos + start + n - 1)
        for jj in range(0, n):
            i = start + (n - 1 - jj)
            p = tl.load(pos + i)
            q = q * tl.exp(lg * (p_next - p).to(tl.float32))
            wv = tl.load(ws + i)
            row = tl.load(gp + i).to(tl.int64)
            d = tl.load(dest + i).to(tl.int64)
            v = tl.load(V + row * D + offs, mask=lane, other=0.0)
            v = tl.where(offs == D, 1.0, v)
            tl.store(DW + d, tl.load(DW + d) + tl.sum(q * v, 0))
            tl.atomic_add(ZQ + row * D + offs, wv * q, mask=lane, sem="relaxed")
            g = tl.load(Gr + row * (D + 1) + offs, mask=lane1, other=0.0)
            q = q + wv * g
            p_next = p


def _segments(w, si, lg, M):
    """Sorted triples for the kernels: segment starts and lengths, and per triple its position, row, original index
    (dest), weight and log-decay. Rows are (batch row x head); dest indexes the (G, T, c) layout."""
    G, T, c = w.shape
    dev = w.device
    g = torch.arange(G, device=dev).view(G, 1, 1)
    key = (g * M + si).reshape(-1)
    if G * M < 2**31:  # a 32-bit radix sort is about twice as fast; same order
        key = key.to(torch.int32)
    order = torch.sort(key, stable=True).indices
    seg = key[order]
    n = seg.numel()
    new = torch.ones(n, dtype=torch.bool, device=dev)
    new[1:] = seg[1:] != seg[:-1]
    starts = new.nonzero().flatten()
    lens = torch.diff(starts, append=starts.new_tensor([n]))
    tri_t = torch.div(order, c, rounding_mode="floor")  # (g T + t) for each sorted triple
    pos = (tri_t % T).to(torch.int32)
    lgv = lg.to(torch.float32)[torch.div(tri_t, T, rounding_mode="floor")]
    return starts.to(torch.int64), lens.to(torch.int32), pos, tri_t, order, w.reshape(-1)[order].float(), lgv


class MixScanTriton(torch.autograd.Function):
    """MixScan with the per-segment walk fused into one Triton kernel each way (float32, CUDA)."""

    @staticmethod
    def forward(ctx, w, V, si, lg_head, M, max_triples):
        B, h, T, c = w.shape
        G, dh = B * h, V.shape[-1]
        w2, V2, si2 = w.reshape(G, T, c), V.reshape(G, T, dh).float().contiguous(), si.reshape(G, T, c)
        lg = lg_head.float()[None].expand(B, h).reshape(G)
        out = torch.empty(G, T, dh + 1, device=w.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(dh + 1)
        ctx.segs = []  # the sort is kept for backward (about 28 bytes a triple), not redone
        for g0, g1 in _rows(G, T, c, max_triples):
            sg = _segments(w2[g0:g1], si2[g0:g1], lg[g0:g1], M)
            ctx.segs.append(sg)
            starts, lens, pos, gp, dest, ws, lgv = sg
            Y = torch.zeros((g1 - g0) * T, dh + 1, device=w.device, dtype=torch.float32)
            _seg_fwd_kernel[(starts.numel(),)](starts, lens, pos, gp, dest, ws, lgv, V2[g0:g1], Y, D=dh, BLOCK=BLOCK)
            out[g0:g1] = Y.view(g1 - g0, T, dh + 1)
            del Y
        ctx.save_for_backward(w, V, si, lg_head)
        ctx.M, ctx.max_triples = M, max_triples
        return out.view(B, h, T, dh + 1)

    @staticmethod
    def backward(ctx, gout):
        w, V, si, lg_head = ctx.saved_tensors
        B, h, T, c = w.shape
        G, dh = B * h, V.shape[-1]
        w2, V2, si2 = w.reshape(G, T, c), V.reshape(G, T, dh).float().contiguous(), si.reshape(G, T, c)
        lg = lg_head.float()[None].expand(B, h).reshape(G)
        gout2 = gout.reshape(G, T, dh + 1).float().contiguous()
        dw = torch.empty(G, T, c, device=w.device, dtype=torch.float32)
        dV = torch.empty(G, T, dh, device=w.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(dh + 1)
        for (g0, g1), sg in zip(_rows(G, T, c, ctx.max_triples), ctx.segs):
            starts, lens, pos, gp, dest, ws, lgv = sg
            DW = torch.empty((g1 - g0) * T * c, device=w.device, dtype=torch.float32)
            ZQ = torch.zeros((g1 - g0) * T, dh, device=w.device, dtype=torch.float32)
            _seg_bwd_kernel[(starts.numel(),)](starts, lens, pos, gp, dest, ws, lgv, V2[g0:g1], gout2[g0:g1], DW, ZQ,
                                               D=dh, BLOCK=BLOCK)
            dw[g0:g1] = DW.view(g1 - g0, T, c)
            dV[g0:g1] = ZQ.view(g1 - g0, T, dh)
            del DW, ZQ
        ctx.segs = None
        return dw.view(B, h, T, c).to(w.dtype), dV.view(B, h, T, dh).to(V.dtype), None, None, None, None


def mix_scan(w, V, si, lg_head, M, tile=32, piece=8192, max_triples=1 << 23, backend="auto"):
    """backend 'auto' = the Triton kernels for float32 CUDA tensors when Triton is present, else the torch scan."""
    if backend == "triton" or (backend == "auto" and HAVE_TRITON and w.is_cuda and w.dtype == torch.float32):
        return MixScanTriton.apply(w, V, si, lg_head, M, max_triples)
    return MixScan.apply(w, V, si, lg_head, M, tile, piece, max_triples)


def selftest():
    torch.manual_seed(0)
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    # the scan against a brute-force double loop, with long segments that cross many tiles and short ones
    n, D = 300, 5
    seg = torch.sort(torch.randint(0, 6, (n,))).values
    pos = torch.zeros(n)
    for s in seg.unique():  # ascending, distinct positions inside each segment
        idx = (seg == s).nonzero().flatten()
        pos[idx] = torch.sort(torch.randperm(1000)[: idx.numel()]).values.float()
    lgv = torch.log(torch.tensor([1.0, 0.99, 0.9])[seg % 3])
    x = torch.randn(n, D, dtype=torch.float64)
    ref = torch.zeros(n, D, dtype=torch.float64)
    for i in range(n):
        for j in range(i):
            if seg[j] == seg[i]:
                ref[i] += torch.exp(lgv[i].double() * (pos[i] - pos[j]).abs().double()) * x[j]
    for tile in (1, 4, 7, 32, 512):
        E = seg_scan_excl(seg, pos.double(), lgv.double(), x, tile=tile, piece=3)
        check(f"scan equals brute force (tile {tile})", torch.allclose(E, ref, atol=1e-9, rtol=1e-9))
    Er = seg_scan_excl(seg.flip(0), pos.double().flip(0), lgv.double().flip(0), x.flip(0), tile=4, piece=2).flip(0)
    ref2 = torch.zeros(n, D, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if seg[j] == seg[i]:
                ref2[i] += torch.exp(lgv[i].double() * (pos[i] - pos[j]).abs().double()) * x[j]
    check("reversed scan equals brute force (later entries)", torch.allclose(Er, ref2, atol=1e-9, rtol=1e-9))
    # gradcheck of the autograd function in float64, distinct locations per position as the product-key search gives
    B, h, T, c, M, dh = 2, 2, 9, 3, 7, 4
    si = torch.stack([torch.randperm(M)[:c] for _ in range(B * h * T)]).view(B, h, T, c)
    w = torch.rand(B, h, T, c, dtype=torch.float64, requires_grad=True)
    V = torch.randn(B, h, T, dh, dtype=torch.float64, requires_grad=True)
    lg = torch.log(torch.tensor([1.0, 0.8], dtype=torch.float64))
    check("gradcheck of MixScan (float64, tile 2)",
          torch.autograd.gradcheck(lambda a, b: mix_scan(a, b, si, lg, M, 2, 3), (w, V), eps=1e-6, atol=1e-6))
    check("gradcheck of MixScan in row slices (max_triples 30: one row per slice)",
          torch.autograd.gradcheck(lambda a, b: mix_scan(a, b, si, lg, M, 2, 3, 30), (w, V), eps=1e-6, atol=1e-6))
    if HAVE_TRITON and torch.cuda.is_available():  # the fused kernels against the torch scan, float32 on the GPU
        B, h, T, c, M, dh = 3, 4, 300, 16, 40, 64
        si = torch.stack([torch.randperm(M)[:c] for _ in range(B * h * T)]).view(B, h, T, c).cuda()
        lg = torch.log(torch.tensor([1.0, 0.999, 0.99, 0.9])).cuda()
        w = torch.rand(B, h, T, c, device="cuda", requires_grad=True)
        V = torch.randn(B, h, T, dh, device="cuda", requires_grad=True)
        w2, V2 = w.detach().clone().requires_grad_(True), V.detach().clone().requires_grad_(True)
        o1 = mix_scan(w, V, si, lg, M, backend="torch")
        o2 = mix_scan(w2, V2, si, lg, M, max_triples=T * c * 5, backend="triton")
        gg = torch.randn_like(o1)
        (o1 * gg).sum().backward()
        (o2 * gg).sum().backward()
        check("Triton kernels = torch scan: output (GPU, float32, 4 decays, row slices)",
              torch.allclose(o1, o2, atol=1e-4, rtol=1e-4))
        check("Triton kernels = torch scan: gradients of w and V",
              torch.allclose(w.grad, w2.grad, atol=1e-3, rtol=1e-4) and torch.allclose(V.grad, V2.grad, atol=1e-3, rtol=1e-4))
    print(f"{sum(ok)} of {len(ok)} checks pass")
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
    print("NEXT -> python3 track4_sdmonly_mixfast.py --selftest")
