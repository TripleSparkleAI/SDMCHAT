"""Fused tied-head cross-entropy for the SDMLLM trainer on CUDA (lane SPARK24): never materialises the full
B*T x V logits, and computes the gradient during the forward pass, chunk by chunk.

<claudes_code_comments>
** Function List **
_chunk(h, E, t, n) - one chunk: logits (bf16 matmul, fp32 softmax), loss sum, grad wrt h, grad wrt E
FusedTiedCE - autograd Function: forward returns the mean CE and stores grad_h, grad_E; backward scales them
fused_ce(h, E, tgt, chunk) - mean next-token cross-entropy of hidden states h (N, d) against the tied table E (V, d)
selftest() - loss and gradients equal the plain path (fp32 logits from a bf16 matmul) within bf16 tolerance

** Technical Review **
- The plain S0 path writes B*T*V fp32 logits (4.2 GB at B 32, T 256, V 129,280) and cross-entropy then reads and
  writes them several times; on the GB10 the step is bound by that memory traffic (26.6k tok/s measured).
- Here each chunk of `chunk` positions computes logits = bf16(h) @ bf16(E)^T, upcasts to fp32 (exactly as S0's
  autocast matmul followed by .float()), takes logsumexp, and forms the logit gradient (softmax - onehot) / N in
  fp32, cast to bf16 for the two gradient matmuls: grad_h = G @ E and grad_E += G^T @ h (accumulated in fp32).
- The math is the textbook CE gradient; the only difference from autograd on the plain path is where bf16
  rounding happens in the backward matmuls, which the selftest bounds.
</claudes_code_comments>
"""
import torch
import torch.nn.functional as F


def _chunk(h, E, t, n):
    lg = (h.to(torch.bfloat16) @ E.to(torch.bfloat16).t()).float()
    lse = torch.logsumexp(lg, -1)
    loss = (lse - lg.gather(1, t[:, None]).squeeze(1)).sum()
    g = torch.exp(lg - lse[:, None])
    g.scatter_add_(1, t[:, None], torch.full_like(lse[:, None], -1.0))
    g = (g / n).to(torch.bfloat16)
    gh = (g @ E.to(torch.bfloat16)).float()
    gE = (g.t() @ h.to(torch.bfloat16)).float()
    return loss, gh, gE


_CHUNK_FN = {"eager": _chunk}


def chunk_fn(compiled):
    if compiled and "compiled" not in _CHUNK_FN:
        _CHUNK_FN["compiled"] = torch.compile(_chunk, dynamic=False)
    return _CHUNK_FN["compiled" if compiled else "eager"]


class FusedTiedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, E, tgt, chunk, compiled=False):
        n = h.shape[0]
        f = chunk_fn(compiled)
        gh = torch.empty(h.shape, dtype=torch.float32, device=h.device)
        gE = torch.zeros(E.shape, dtype=torch.float32, device=E.device)
        tot = torch.zeros((), dtype=torch.float32, device=h.device)
        for i in range(0, n, chunk):
            l, a, b = f(h[i:i + chunk], E, tgt[i:i + chunk], n)
            tot += l
            gh[i:i + chunk] = a
            gE += b
        ctx.save_for_backward(gh, gE)
        ctx.hdtype, ctx.Edtype = h.dtype, E.dtype
        return tot / n

    @staticmethod
    def backward(ctx, go):
        gh, gE = ctx.saved_tensors
        return (gh * go).to(ctx.hdtype), (gE * go).to(ctx.Edtype), None, None, None


def ce_chunked(h, E, tgt, chunk=2048):
    """Mean CE of the tied head over chunks, each recomputed in backward (checkpoint); same math as the plain path."""
    from torch.utils.checkpoint import checkpoint
    h = h.reshape(-1, h.shape[-1])
    tgt = tgt.reshape(-1)

    def part(hc, tc):
        return F.cross_entropy((hc @ E.t()).float(), tc, reduction="sum")
    tot = 0
    for i in range(0, h.shape[0], chunk):
        tot = tot + checkpoint(part, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
    return tot / h.shape[0]


def fused_ce(h, E, tgt, chunk=4096, compiled=False):
    h = h.reshape(-1, h.shape[-1])
    with torch.autocast(device_type=h.device.type, enabled=False):
        return FusedTiedCE.apply(h, E, tgt.reshape(-1), chunk, compiled)


def selftest():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    V, d, N = 129280, 256, 4096
    E = (torch.randn(V, d, device=dev) * 0.02).requires_grad_()
    h = (torch.randn(N, d, device=dev)).requires_grad_()
    t = torch.randint(0, V, (N,), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l1 = F.cross_entropy((h @ E.t()).float(), t)
    l1.backward()
    g1h, g1E = h.grad.clone(), E.grad.clone()
    h.grad = E.grad = None
    l2 = fused_ce(h, E, t, chunk=1024, compiled=True)
    l2.backward()
    rel = lambda a, b: float((a - b).norm() / b.norm())
    print("loss", float(l1), float(l2), "abs diff", abs(float(l1) - float(l2)))
    print("grad_h rel", rel(h.grad, g1h), "grad_E rel", rel(E.grad, g1E))
    ok = abs(float(l1) - float(l2)) < 1e-4 and rel(h.grad, g1h) < 1e-2 and rel(E.grad, g1E) < 1e-2
    # negative control: a wrong target must change the loss
    l3 = fused_ce(h.detach(), E.detach(), (t + 1) % V)
    ok = ok and abs(float(l3) - float(l2)) > 1e-3
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
