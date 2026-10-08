"""MIXERFAST bench: training-step speed and peak memory of the SDM mixer, dense against fast, on one GPU.

<claudes_code_comments>
** Function List **
stamp() - GPU, driver, torch, load, GPU utilisation and time, for every number this prints
build(impl, ema, d) - the wave-11 cnmix body (sdmonly_none, hop MLP 192, untie, conj 2, 4 mixers, 128 x 128
                      locations, k 32, 4 never-fading heads) with the chosen mixer ("none" = no mixer, the
                      reference); ema False drops the moving averages
one(a) - time warmup + n steps of forward, backward and AdamW at one (impl, T, B); prints one JSON line
main() - parse; --one runs a single point, otherwise runs every point in its own process (OOM-safe)

** Technical Review **
- Random tokens (no data needed), vocabulary 129,280, the trainer's loss path (`--loss chunked`: compiled
  hidden -> chunked tied-head cross-entropy, track4_sdmllm24_fused_ce), bf16 autocast, AdamW. The dense mixer
  compiles inside that loss for windows up to 2,048 and runs eagerly beyond, as in training; the fast mixer always
  runs eagerly (torch.compiler.disable).
- `--no-ema` drops the five causal moving averages: track4_sdmllm_models.causal_ema builds a T x T matrix per decay,
  which cannot reach 64k tokens (17 GB per decay at T 65,536). That limit is outside the mixer and is reported.
- Each point runs in a fresh process; a CUDA out-of-memory error is reported as "OOM", not a crash.
- tok/s = timed steps x B x T / wall time of the timed steps (after torch.cuda.synchronize); peak memory =
  torch.cuda.max_memory_allocated over warmup and timed steps.
Docs: MIXERFAST_RESULTS_2026-10-03.md
</claudes_code_comments>
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def stamp():
    out = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load1": round(os.getloadavg()[0], 2),
           "ncpu": os.cpu_count()}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        out["gpu"], out["driver"], out["gpu_util"], out["gpu_mem_used"] = [x.strip() for x in q.splitlines()[0].split(",")]
    except Exception as e:  # noqa: BLE001
        out["gpu"] = f"unknown ({e.__class__.__name__})"
    import torch
    out["torch"], out["cuda"] = torch.__version__, torch.version.cuda
    return out


def build(impl, ema, d=256):
    import track4_sdmonly_models as M
    cfg = {"d": d, "hops": 4, "hop_mlp": 192, "untie": True, "conj": 2, "mix_layers": 0 if impl == "none" else 4,
           "mix_n_sub": 128, "mix_k": 32, "mix_heads": 4, "mix_d_a": 64, "mix_decays": (1.0, 1.0, 1.0, 1.0),
           "mix_impl": "dense" if impl == "none" else impl,
           "decays": (0.5, 0.8, 0.9, 0.97, 0.99) if ema else ()}
    return M.build_sdmonly("sdmonly_none", 129280, cfg)


def one(a):
    import torch
    from track4_sdmllm24_fused_ce import ce_chunked
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    torch.manual_seed(0)
    rec = {"impl": a.impl, "T": a.T, "B": a.B, "ema": not a.no_ema, "stamp_start": stamp()}
    try:
        model = build(a.impl, not a.no_ema).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.compile(lambda x, t: ce_chunked(model.hidden(x), model.emb.weight, t, 2048))
        torch.cuda.reset_peak_memory_stats()

        def step():
            xb = torch.randint(1, 129280, (a.B, a.T + 1), device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = loss_fn(xb[:, :-1], xb[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return loss

        t0 = time.time()
        for _ in range(a.warmup):
            step()
        torch.cuda.synchronize()
        rec["warmup_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        for _ in range(a.steps):
            loss = step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        rec.update({"tok_per_s": round(a.steps * a.B * a.T / dt, 1), "s_per_step": round(dt / a.steps, 4),
                    "steps": a.steps, "loss": round(float(loss), 4),
                    "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    except torch.cuda.OutOfMemoryError as e:
        rec.update({"tok_per_s": "OOM", "error": str(e).splitlines()[0][:200],
                    "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    rec["stamp_end"] = stamp()
    print("RESULT " + json.dumps(rec), flush=True)


def main():
    p = argparse.ArgumentParser(description="Example: python3 %(prog)s --Ts 256,1024,4096 --impls dense,fast")
    p.add_argument("--one", action="store_true", help="run one point in this process")
    p.add_argument("--impl", default="fast", choices=["dense", "fast", "none"], help="none = the same body, no mixer")
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--B", type=int, default=0, help="0 = max(1, tokens_per_step // T)")
    p.add_argument("--tokens-per-step", type=int, default=16384)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--no-ema", action="store_true", help="drop the T x T moving-average features")
    p.add_argument("--Ts", default="256,1024,4096,16384,65536,262144")
    p.add_argument("--impls", default="dense,fast")
    p.add_argument("--out", default=None, help="append RESULT lines to this file")
    a = p.parse_args()
    if a.one:
        a.B = a.B or max(1, a.tokens_per_step // a.T)
        one(a)
        return
    for T in [int(x) for x in a.Ts.split(",")]:
        for impl in a.impls.split(","):
            cmd = [sys.executable, os.path.abspath(__file__), "--one", "--impl", impl, "--T", str(T),
                   "--tokens-per-step", str(a.tokens_per_step), "--warmup", str(a.warmup), "--steps", str(a.steps)]
            if a.B:
                cmd += ["--B", str(a.B)]
            if a.no_ema:
                cmd.append("--no-ema")
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
            line = lines[-1] if lines else "RESULT " + json.dumps(
                {"impl": impl, "T": T, "tok_per_s": "FAILED", "rc": r.returncode, "tail": r.stderr[-400:]})
            print(line, flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(line + "\n")
    print("NEXT -> write the table into MIXERFAST_RESULTS_2026-10-03.md")


if __name__ == "__main__":
    main()
