"""Throughput of the SDMLLM training step on the GB10 under different loss paths (lane SPARK24).

<claudes_code_comments>
** Function List **
step_fn(kind, model) - one training step's loss for a loss path: 'plain' (S0: fp32 logits + F.cross_entropy),
                       'chunked' (hidden @ E^T and CE in token chunks under checkpoint), 'compiled' (torch.compile
                       of hidden -> loss), 'liger' (Liger fused linear cross-entropy, if installed)
main() - time each path for a few steps on random tokens, and check its loss equals the plain loss

** Technical Review **
- The tied output layer (129,280 x d) dominates: plain fp32 logits are B*T*V*4 bytes = 4.2 GB per step at B 32,
  T 256, and cross-entropy reads and writes them several times, so the step is memory-bound on the logits.
- Every path must give the same loss on the same batch (to ~1e-3 nats with bf16 matmuls); the check is printed.
</claudes_code_comments>
"""
import os, sys, time, argparse
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track4_sdmllm_models import build
V = 129280


def ce_chunked(h, E, tgt, chunk=2048):
    h = h.reshape(-1, h.shape[-1]); tgt = tgt.reshape(-1)
    def part(hc, tc):
        return F.cross_entropy((hc @ E.t()).float(), tc, reduction="sum")
    tot = 0
    for i in range(0, h.shape[0], chunk):
        tot = tot + checkpoint(part, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
    return tot / h.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="sdm_ngram"); ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--B", type=int, default=32); ap.add_argument("--paths", default="plain,chunked,compiled,compiled_chunked,liger")
    a = ap.parse_args()
    dev = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    cfg = {"d": a.d, "ngram_orders": [1], "qgrad": 0.0, "value_init": "zero"}
    xb = torch.randint(0, V, (a.B, 257), device=dev)
    ref = None
    for path in a.paths.split(","):
        torch.manual_seed(0)
        m = build(a.arm, V, cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        if path == "liger":
            try:
                from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            except Exception as e:
                print(path, "unavailable:", type(e).__name__, e); continue
            lf = LigerFusedLinearCrossEntropyLoss()
            def loss_fn(x, t, m=m): return lf(m.emb.weight, m.hidden(x).reshape(-1, a.d), t.reshape(-1))
        elif path == "plain":
            def loss_fn(x, t, m=m): return F.cross_entropy(m(x).float().reshape(-1, V), t.reshape(-1))
        elif path == "chunked":
            def loss_fn(x, t, m=m): return ce_chunked(m.hidden(x), m.emb.weight, t)
        elif path == "compiled":
            def _f(x, t, m=m): return F.cross_entropy((m.hidden(x) @ m.emb.weight.t()).float().reshape(-1, V), t.reshape(-1))
            loss_fn = torch.compile(_f)
        elif path.startswith("fused") or path.startswith("cfused"):
            from track4_sdmllm24_fused_ce import fused_ce
            comp = path.startswith("c")
            ch = int(path[6 if comp else 5:] or 4096)
            def loss_fn(x, t, m=m, ch=ch, comp=comp): return fused_ce(m.hidden(x), m.emb.weight, t, ch, comp)
        elif path == "compiled_chunked":
            loss_fn = torch.compile(lambda x, t, m=m: ce_chunked(m.hidden(x), m.emb.weight, t))
        times = []
        for i in range(10):
            torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(xb[:, :-1], xb[:, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); times.append(time.time() - t0)
            if i == 0:
                l0 = float(loss)
        if ref is None: ref = l0
        dt = sorted(times[3:])[len(times[3:]) // 2]
        print(f"{path:18s} {a.B * 256 / dt:9.0f} tok/s  {dt:.3f} s/step  first-step loss {l0:.5f} (plain {ref:.5f})  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
