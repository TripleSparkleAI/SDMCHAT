"""SDMLLM S0 trainer: one arm, one seed, standard next-token pretraining, resumable, stamped.

<claudes_code_comments>
** Function List **
stamp() - load average, cpu count, power mode, macOS build, torch version, device
load_split(name) - memmap a uint32 token shard
val_windows(arr, T, half) - fixed contiguous windows over the test (or calib) half of validation
eval_bpb(model, ...) - bits per byte over the fixed windows, BOS targets excluded
lr_at(step, total, peak, warm, floor) - linear warmup then cosine decay to floor * peak
train(args) - the loop: AdamW, grad clip, checkpoint every N steps, resume from the last checkpoint
main() - CLI

** Technical Review **
- Batches are B windows of T+1 tokens at offsets drawn from a seeded numpy Generator over the train shard;
  the draw for step s is a pure function of (seed, s), so a resumed run sees exactly the batches a straight
  run would have seen. Every arm with the same seed sees the SAME tokens in the SAME order.
- bpb = sum over scored positions of CE nats / (ln 2 * sum of token_bytes[target]); a position whose target
  is BOS (id 0) is not scored. Validation windows are non-overlapping T+1 slices of the val shard's second
  half (TEST) by default; the first half (CALIB) is reserved for tuning the count-store floors.
- Mixed precision: bf16 autocast on MPS for matmuls, fp32 master weights and fp32 cross-entropy.
- AdamW betas (0.9, 0.95), weight decay 0.1 on 2-D non-embedding weights only, grad clip 1.0,
  warmup 2% then cosine to 10% of peak.
- SDMLLMSTORE options (defaults reproduce S0): the store's keys and values sit in their own AdamW group with
  --store-lr-mult (lr multiplier) and --store-wd (S0: 0.1, because they are 2-D); --value-init and --qgrad pass
  to the model. SDMLLM_DATA points at the gitignored shards (read-only). At the end the trainer also writes the
  per-window TEST nats and bytes to checkpoints/<run>/test_per_window.npz, the input of the paired comparison.
- Checkpoint (model, optimizer, step, config) every --ckpt-every steps to checkpoints/<run>/last.pt; the
  run log and the final result json carry the stamp at start and end (load changes during a run are logged).
</claudes_code_comments>
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data"))
sys.path.insert(0, HERE)
from track4_sdmllm_models import build, count_params, flops_per_token  # noqa: E402

V = 129280


def stamp():
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        l1, l5, l15 = os.getloadavg()
        out["load"] = [round(l1, 2), round(l5, 2), round(l15, 2)]
    except OSError:
        out["load"] = None
    out["ncpu"] = os.cpu_count()
    try:
        pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
        out["powermode"] = next((ln.split()[-1] for ln in pm.splitlines() if "powermode" in ln), "unknown")
        out["macos_build"] = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    except Exception:
        out["powermode"] = "unknown"
    out["torch"] = torch.__version__
    out["host"] = platform.node()
    if torch.cuda.is_available():
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0)
        try:
            out["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            out["driver"] = "unknown"
        try:
            out["mem_available_gb"] = round(int(next(ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemAvailable"))) / 1048576, 1)
        except Exception:
            pass
    return out


def load_split(name):
    return np.memmap(os.path.join(DATA, f"{name}.u32"), dtype=np.uint32, mode="r")


def pick_device(cpu=False):
    if cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def val_windows(arr, T, half="test", max_windows=None):
    n = len(arr)
    lo, hi = {"test": (n // 2, n), "calib": (0, n // 2), "all": (0, n)}[half]
    starts = np.arange(lo, hi - (T + 1), T + 1)
    if max_windows:
        starts = starts[:max_windows]
    return starts


@torch.no_grad()
def eval_bpb(model, arr, starts, T, tb, device, bs=32):
    model.eval()
    cap = int(os.environ.get("SDM_EVAL_TOKENS", "8192"))  # a very wide model (width 4,096) needs a smaller eval batch
    if T * bs > 2 * cap:  # long windows: a 32 x 1024 batch of 129k-vocab logits is 15.8 GiB; keep about `cap` tokens per batch
        bs = max(1, cap // T)  # with the default cap, windows up to 512 keep bs=32, so every earlier number is unchanged
    tot_nats, tot_bytes, tot_tok = 0.0, 0, 0
    for i in range(0, len(starts), bs):
        s = starts[i:i + bs]
        x = np.stack([arr[a:a + T + 1] for a in s]).astype(np.int64)
        xb = torch.from_numpy(x).to(device)
        inp, tgt = xb[:, :-1], xb[:, 1:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
            logits = model(inp)
        nll = F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1), reduction="none")
        mask = tgt.reshape(-1) != 0
        tot_nats += float(nll[mask].sum())
        tot_bytes += int(tb[tgt.reshape(-1)[mask]].sum())
        tot_tok += int(mask.sum())
    model.train()
    if device.type == "cuda":
        # an eval batch's full-vocabulary logits are several GB; hand the cached blocks back so a run sharing the
        # unified memory of the Spark (lane SDMCHATS, beside HEADCLIMB) returns to its training footprint. No number moves.
        logits = nll = None
        torch.cuda.empty_cache()
    return {"bpb": tot_nats / (math.log(2) * tot_bytes), "nats_per_token": tot_nats / tot_tok,
            "tokens": tot_tok, "bytes": tot_bytes}


def lr_at(step, total, peak, warm, floor=0.1):
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def batch_for(step, seed, arr, B, T):
    g = np.random.default_rng([seed, step])
    offs = g.integers(0, len(arr) - (T + 1), size=B)
    return np.stack([arr[a:a + T + 1] for a in offs]).astype(np.int64)


def train(a):
    device = pick_device(a.cpu)
    run = a.run or f"{a.arm}_s{a.seed}_{a.tokens // 10**6}M"
    ck_dir = os.path.join(os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints")), run)
    os.makedirs(ck_dir, exist_ok=True)
    out_dir = os.environ.get("SDMLLM_RUNS", os.path.join(HERE, "runs"))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{run}.log")
    cfg = {"d": a.d, "T": a.T, "seed": a.seed, "M": a.M, "k": a.k, "hops": a.hops, "softness": a.softness,
           "n_layer": a.n_layer, "M_big": a.M_big, "value_init": a.value_init, "qgrad": a.qgrad,
           "ngram_orders": [int(x) for x in a.ngram_orders.split(",")], "f": a.f, "train_name": a.train_name,
           "n_back": a.n_back, "decays": [float(x) for x in a.decays.split(",")]}
    torch.manual_seed(a.seed)
    model = build(a.arm, V, cfg).to(device)
    if a.init:
        ck0 = torch.load(a.init, map_location=device)
        model.load_state_dict(ck0["model"])
        print(json.dumps({"event": "init", "from": a.init, "step": ck0.get("step")}), flush=True)
    tb = torch.from_numpy(np.fromfile(os.path.join(DATA, "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
    train_arr = load_split(a.train_name)
    val_arr = load_split("val")
    starts = val_windows(val_arr, a.T, "test", a.val_windows)
    tokens_per_step = a.B * a.T
    total = a.tokens // tokens_per_step
    warm = max(1, int(0.02 * total))
    decay, no_decay, store_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("store."):
            store_p.append(p)
        else:
            (decay if p.dim() == 2 and "emb" not in n else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": 0.1, "lr_mult": 1.0}, {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}]
    if store_p:
        groups.append({"params": store_p, "weight_decay": a.store_wd, "lr_mult": a.store_lr_mult})
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    step = 0
    last = os.path.join(ck_dir, "last.pt")
    if os.path.exists(last) and not a.fresh:
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    if a.compile:
        model = torch.compile(model)
    if a.loss == "chunked":
        # lane SPARK24: the tied head's fp32 logits (B*T*V*4 bytes) bound the CUDA step; compute hidden -> CE in
        # checkpointed chunks under torch.compile instead. Same math (bf16 matmul, fp32 log-softmax), never the full logits.
        from track4_sdmllm24_fused_ce import ce_chunked
        if getattr(model, "store", None) is not None:
            model.store.collect_stats = False
        _m = model
        loss_fn = torch.compile(lambda x, t: ce_chunked(_m.hidden(x), _m.emb.weight, t, a.ce_chunk))
    else:
        def loss_fn(x, t):
            return F.cross_entropy(model(x).float().reshape(-1, V), t.reshape(-1))
    info = {"run": run, "arm": a.arm, "seed": a.seed, "cfg": cfg, "B": a.B, "T": a.T, "tokens_target": a.tokens,
            "steps": total, "lr": a.lr, "loss_path": a.loss, "store_lr_mult": a.store_lr_mult, "store_wd": a.store_wd, "params": count_params(model), "flops_per_token_fwd": flops_per_token(model, a.T),
            "device": str(device), "stamp_start": stamp(), "resumed_from_step": step, "init": a.init,
            "train_name": a.train_name}
    logf = open(log_path, "a")
    logf.write(json.dumps({"event": "start", **info}) + "\n")
    logf.flush()
    print(json.dumps(info))
    t0 = time.time()
    tok_seen = step * tokens_per_step
    curve = []
    while step < total:
        lr = lr_at(step, total, a.lr, warm)
        for gpar in opt.param_groups:
            gpar["lr"] = lr * gpar.get("lr_mult", 1.0)
        xb = torch.from_numpy(batch_for(step, a.seed, train_arr, a.B, a.T)).to(device)
        opt.zero_grad(set_to_none=True)
        if a.accum <= 1:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
                loss = loss_fn(xb[:, :-1], xb[:, 1:])
            loss.backward()
        else:  # gradient accumulation: the same B windows in `accum` micro-batches (batch-norm statistics are per micro-batch)
            loss = 0.0
            for mb in xb.chunk(a.accum):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
                    lm = loss_fn(mb[:, :-1], mb[:, 1:]) / a.accum
                lm.backward()
                loss = loss + lm.detach()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tok_seen += tokens_per_step
        if step % a.log_every == 0 or step == total:
            dt = time.time() - t0
            rec = {"event": "step", "step": step, "loss": round(float(loss), 4), "lr": lr, "gnorm": round(float(gn), 3),
                   "tok_per_s": round(a.log_every * tokens_per_step / dt, 1), "load": os.getloadavg()[0]}
            if getattr(model, "hop_stats", None):
                rec["hop_stats"] = model.hop_stats
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
            t0 = time.time()
        if step % a.eval_every == 0 and step < total:
            ev = eval_bpb(model, val_arr, starts[: a.quick_val], a.T, tb, device)
            curve.append({"step": step, "tokens": tok_seen, "bpb_quick": ev["bpb"]})
            logf.write(json.dumps({"event": "eval_quick", "step": step, **ev}) + "\n")
            logf.flush()
            print("eval", step, ev, flush=True)
        if step % a.ckpt_every == 0 or step == total:
            torch.save({"model": getattr(model, "_orig_mod", model).state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg, "arm": a.arm},
                       last + ".tmp")
            os.replace(last + ".tmp", last)
    if getattr(model, "store", None) is not None:
        getattr(model, "_orig_mod", model).store.collect_stats = True
    res = {**info, "stamp_end": stamp(), "curve": curve}
    res["val_test"] = eval_bpb(model, val_arr, starts, a.T, tb, device)
    for name in a.extra_val:
        ex = load_split(name)
        res.setdefault("val_extra", {})[name] = eval_bpb(model, ex, val_windows(ex, a.T, "all", 4000), a.T, tb, device)
        print("extra", name, json.dumps(res["val_extra"][name]), flush=True)
    if a.val_windows is None:
        from track4_sdmllm_paired_window_comparison import per_window
        model.eval()
        pn, pb = per_window(model, val_arr, starts, a.T, tb, device)
        np.savez(os.path.join(ck_dir, "test_per_window.npz"), nats=pn, bytes=pb)
        model.train()
    model = getattr(model, "_orig_mod", model)
    if a.arm == "sdm_ngram":
        for ab in ("shuffle_keys", "zero_read"):
            model.store.ablate = ab
            res[f"val_test_ablate_{ab}"] = eval_bpb(model, val_arr, starts, a.T, tb, device)
        model.store.ablate = None
    if a.arm in ("sdm", "sdm_frozen", "sdm_big"):
        for ab in ("shuffle_keys", "zero_read"):
            model.store.ablate = ab
            res[f"val_test_ablate_{ab}"] = eval_bpb(model, val_arr, starts, a.T, tb, device)
        model.store.ablate = None
        # location usage: how many distinct locations the top-k selects over the test windows (hop 0)
        used = torch.zeros(model.M, dtype=torch.float32, device=device)
        with torch.no_grad():
            for i in range(0, min(len(starts), 64), 16):
                x = np.stack([val_arr[s:s + a.T] for s in starts[i:i + 16]]).astype(np.int64)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
                    model(torch.from_numpy(x).to(device))
                sel = model.store.last_si[:, : model.k].reshape(-1)
                used.index_add_(0, sel, torch.ones(sel.shape[0], device=device))
        res["locations_used_fraction_last_hop"] = float((used > 0).float().mean())
    logf.write(json.dumps({"event": "final", "val_test": res["val_test"]}) + "\n")
    logf.close()
    json.dump(res, open(os.path.join(out_dir, f"{run}.result.json"), "w"), indent=1)
    print("FINAL", run, json.dumps(res["val_test"]))
    return res


def bench(a):
    """Throughput of a few steps on random tokens (no data needed)."""
    device = pick_device(a.cpu)
    sync = torch.cuda.synchronize if device.type == "cuda" else torch.mps.synchronize
    for arm in a.bench_arms.split(","):
        torch.manual_seed(0)
        model = build(arm, V, {"d": a.d, "T": a.T, "M": a.M, "k": a.k, "hops": a.hops, "n_layer": a.n_layer, "f": a.f,
                               "ngram_orders": [int(x) for x in a.ngram_orders.split(",")], "qgrad": a.qgrad}).to(device)
        if a.compile:
            model = torch.compile(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        xb = torch.randint(0, V, (a.B, a.T + 1), device=device)
        for i in range(8):
            if i == 3:
                sync()
                t0 = time.time()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(xb[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, V), xb[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sync()
        dt = (time.time() - t0) / 5
        print(arm, f"{a.B * a.T / dt:.0f} tok/s", f"{dt:.3f} s/step", stamp()["load"], flush=True)


def main():
    p = argparse.ArgumentParser(description="SDMLLM S0: train one arm. Example: python3 %(prog)s --arm sdm --seed 0 --tokens 30000000")
    p.add_argument("--arm", default="qwen", choices=["qwen", "sdm", "sdm_frozen", "sdm_nostore", "sdm_mlp", "sdm_big", "sdm_ngram"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", type=int, default=30_000_000)
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="split each batch of B windows into this many micro-batches")
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--f", type=int, default=688, help="SwiGLU hidden size (readout, and qwen MLP)")
    p.add_argument("--n-back", type=int, default=4, help="SDM family: how many previous token embeddings are features")
    p.add_argument("--decays", default="0.8,0.97", help="SDM family: decays of the causal moving-average features")
    p.add_argument("--train-name", default="train", help="train shard name under SDMLLM_DATA (train_big on the Spark)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    p.add_argument("--loss", default="plain", choices=["plain", "chunked"],
                   help="plain = S0 (fp32 logits + cross_entropy); chunked = compiled checkpointed chunks (CUDA)")
    p.add_argument("--ce-chunk", type=int, default=2048)
    p.add_argument("--init", default=None, help="start from this checkpoint's weights (fresh optimizer and schedule)")
    p.add_argument("--extra-val", action="append", default=[], help="also score bpb on this shard (all of it, up to 4,000 windows)")
    p.add_argument("--M", type=int, default=3600)
    p.add_argument("--M_big", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--softness", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--store-lr-mult", type=float, default=1.0, help="lr multiplier for store keys and values (S0: 1)")
    p.add_argument("--store-wd", type=float, default=0.1, help="weight decay on store keys and values (S0: 0.1)")
    p.add_argument("--value-init", default="s0", choices=["s0", "zero"])
    p.add_argument("--ngram-orders", default="2,3", help="sdm_ngram: context lengths hashed, one table each")
    p.add_argument("--qgrad", type=float, default=1.0, help="gradient scale from the store query into x (S0: 1, 0 = stop)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--quick-val", type=int, default=64)
    p.add_argument("--val-windows", type=int, default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--bench-arms", default="qwen,sdm,sdm_nostore,sdm_mlp")
    a = p.parse_args()
    if a.bench:
        bench(a)
    else:
        train(a)
    print("NEXT -> python3 track4_sdmllm_evaluate_floors_and_samples.py")


if __name__ == "__main__":
    main()
