"""ONESDM puzzles: synthetic tasks only a memory can solve, a trainer, an evaluator and a cost bench.

<claudes_code_comments>
** Function List **
Vocab - token layout: PAD, SEP, QUERY, BOS, then key, value and noise alphabets
make_recall(g, B, n_pairs, gap, n_queries, voc) - puzzle A: key-value pairs, noise, then queried keys
make_copy(g, B, L, voc) - puzzle B: a random sequence, SEP, the same sequence again
make_blurry(g, B, n_pairs, key_len, gap, n_queries, corrupt, voc) - puzzle C: multi-symbol keys, corrupted at query time
oracle_accuracy(idx, tgt, puzzle) - a lookup solver over the tokens (the data check: 1.0 on clean data)
sample_batch(a, g, voc, train) - one training batch of the chosen puzzle at a random gap or length
eval_setting(model, a, voc, puzzle, setting, device) - accuracy at one gap / length / corruption level
param_groups(model, a) - AdamW groups (decay, no-decay, memory) and the optional Muon body matrices
lr_at(step, total, peak) - 2% warmup, cosine to 10% of peak
grad_norm(params) - L2 norm of the gradients of a parameter list
train_puzzle(a, puzzle, arm, seed, device) - train one arm on one puzzle, evaluate every setting, write a JSON
cost_bench(a, device) - wall time per token and decode state per model at sequence lengths 1k, 4k, 16k
stamps(device) - host, torch, device, driver, load and git sha for every JSON
parse(argv) - flags; main() - dispatch
selftest() - data checks, oracle, causal masking of targets, positive controls (onesdm, onesdm_allsdm), tiny cost runs

** Technical Review **
- Vocabulary (puzzle-only, not the LM tokenizer): 0 PAD, 1 SEP, 2 QUERY, 3 BOS, then n_keys key symbols, n_vals
  value symbols, n_noise noise symbols (default 128 each, V 388). The three alphabets are disjoint, so a write gate
  can learn from a token's identity alone that noise is not worth writing. That makes the puzzles a test of holding
  and finding, not of telling noise from data; overlapping alphabets is a harder variant left untested.
- A, RECALL AT DISTANCE: BOS, n_pairs (key value) pairs with distinct keys, `gap` noise tokens, then n_queries
  (QUERY key value) triples over keys drawn from the pairs. Scored: the next-token prediction at each queried key
  must be its value (argmax over the whole vocabulary). TRAINED at gaps uniform in [0, --train-gap] (default 256),
  TESTED at 100, 1,000, 4,000 and 16,000: the long gaps are never seen in training (length generalisation).
- B, COPY THE REPEAT: BOS, L random symbols, SEP, the same L symbols. Scored: every prediction of the second copy
  (the first symbol predicted at SEP, then each next one). Trained at L uniform in [8, --train-len] (default 128),
  tested at 128, 512 and 2,048.
- C, BLURRY CUE: like A, but a key is key_len symbols (default 4) and at query time round(c * key_len) of them are
  replaced by other key symbols. Scored at the last key symbol. Trained at corruption --train-corrupt (default 0) and
  gaps uniform in [0, --train-gap]; tested at corruption 0, 0.25, 0.5, 0.75 and gap --blurry-gap (default 100).
- Loss: cross-entropy on the scored positions only. Better is HIGHER accuracy. Chance for A and C is about
  1/n_vals (0.8%); for B it is about 1/(V - 4).
- Optimiser: AdamW (betas 0.9, 0.95; weight decay 0.1 on 2-D non-embedding non-memory matrices). The memory and
  archive parameters (names mems.* and arcs.*) are their own group: learning rate x --mem-lr-mult, no weight decay,
  and when --mem-clip > 0 they are clipped to that norm before the global clip (--clip). --opt muon gives the 2-D body
  matrices Muon (track4_sdmonly_optim.Muon) and keeps AdamW for the rest. Every log line records the gradient norm of
  the memory parameters and of the rest, before clipping, because climbing memory gradients sank the last mixer.
- COST: for each model, B 1, random tokens at T 1,024, 4,096 and 16,384: the forward pass's wall time per token (best
  of --cost-reps after one warm run), the CUDA peak memory when on a GPU, and the bytes a model must keep to continue
  decoding (decode_state_bytes: fixed for the SDM models, growing with T for the yardstick's key-value cache and for
  the archive's buffer). Better is FLAT per-token time and state. On the CPU no peak-memory figure is taken.
- ARMS: the default campaign is ARMS (onesdm, onesdm_archive, plain, sdmonly, yardstick_transformer). The all-SDM
  arms onesdm_allsdm and onesdm_allsdm_big (track4_onesdm_models.TrainedSdm in place of every MLP) run only when
  named in --arms. Their flags: --sdm-n-sub (0 = matched to the MLP), --sdm-heads, --sdm-d-a (0 = min(32, d/heads)),
  --sdm-k, --sdm-scale, --sdm-big-mult (n_sub multiplier for _big, default 4 = 16x the slots).
- Every run writes one JSON with its full config, the stamps, the parameter counts, the training curve and every
  evaluated setting. Folder: ONESDM_RUNS (default runs_onesdm/ beside this file).
Docs: ONESDM_README.md · track4_onesdm_models.py
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

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track4_onesdm_models as OM  # noqa: E402
import track4_sdmonly_optim as O  # noqa: E402

PAD, SEP, QUERY, BOS = 0, 1, 2, 3
ARMS = ["onesdm", "onesdm_archive", "plain", "sdmonly", "yardstick_transformer"]  # the default campaign
ALL_ARMS = ARMS + list(OM.ALLSDM_ARMS)  # the all-SDM arms run only when named in --arms


class Vocab:
    def __init__(self, n_keys=128, n_vals=128, n_noise=128):
        self.nk, self.nv, self.nn_ = n_keys, n_vals, n_noise
        self.k0, self.v0, self.n0 = 4, 4 + n_keys, 4 + n_keys + n_vals
        self.V = self.n0 + n_noise


def _rand(g, lo, hi, shape):
    return torch.randint(lo, hi, shape, generator=g)


def make_recall(g, B, n_pairs, gap, n_queries, voc):
    keys = torch.rand(B, voc.nk, generator=g).argsort(-1)[:, :n_pairs] + voc.k0
    vals = _rand(g, voc.v0, voc.v0 + voc.nv, (B, n_pairs))
    pairs = torch.stack([keys, vals], -1).reshape(B, 2 * n_pairs)
    noise = _rand(g, voc.n0, voc.V, (B, gap))
    qi = torch.rand(B, n_pairs, generator=g).argsort(-1)[:, :n_queries]
    qk, qv = keys.gather(1, qi), vals.gather(1, qi)
    qs = torch.stack([torch.full_like(qk, QUERY), qk, qv], -1).reshape(B, 3 * n_queries)
    idx = torch.cat([torch.full((B, 1), BOS), pairs, noise, qs], 1)
    tgt = torch.full_like(idx, -100)
    start = 1 + 2 * n_pairs + gap
    for j in range(n_queries):
        tgt[:, start + 3 * j + 1] = qv[:, j]  # at the queried key, predict its value
    return idx, tgt


def make_copy(g, B, L, voc):
    seq = _rand(g, voc.k0, voc.V, (B, L))
    idx = torch.cat([torch.full((B, 1), BOS), seq, torch.full((B, 1), SEP), seq], 1)
    tgt = torch.full_like(idx, -100)
    tgt[:, L + 1: 2 * L + 1] = seq  # at SEP predict seq[0]; at copy position i predict seq[i + 1]
    return idx, tgt


def make_blurry(g, B, n_pairs, key_len, gap, n_queries, corrupt, voc):
    keys = _rand(g, voc.k0, voc.k0 + voc.nk, (B, n_pairs, key_len))
    vals = _rand(g, voc.v0, voc.v0 + voc.nv, (B, n_pairs))
    pairs = torch.cat([keys, vals[..., None]], -1).reshape(B, n_pairs * (key_len + 1))
    noise = _rand(g, voc.n0, voc.V, (B, gap))
    qi = torch.rand(B, n_pairs, generator=g).argsort(-1)[:, :n_queries]
    qk = keys.gather(1, qi[..., None].expand(-1, -1, key_len)).clone()
    qv = vals.gather(1, qi)
    nc = int(round(corrupt * key_len))
    if nc:
        which = torch.rand(B, n_queries, key_len, generator=g).argsort(-1)[..., :nc]
        shift = _rand(g, 1, voc.nk, (B, n_queries, nc))  # a different key symbol, never the original
        old = qk.gather(-1, which) - voc.k0
        qk.scatter_(-1, which, (old + shift) % voc.nk + voc.k0)
    qs = torch.cat([torch.full((B, n_queries, 1), QUERY), qk, qv[..., None]], -1).reshape(B, n_queries * (key_len + 2))
    idx = torch.cat([torch.full((B, 1), BOS), pairs, noise, qs], 1)
    tgt = torch.full_like(idx, -100)
    start = 1 + n_pairs * (key_len + 1) + gap
    for j in range(n_queries):
        tgt[:, start + j * (key_len + 2) + key_len] = qv[:, j]
    return idx, tgt


def oracle_accuracy(idx, tgt, puzzle, key_len=1):
    """Solve from the tokens with a dictionary (exact match, earliest pairs). 1.0 on clean data; checks the layout."""
    hit = tot = 0
    for b in range(idx.shape[0]):
        row, tr = idx[b].tolist(), tgt[b].tolist()
        for p, t in enumerate(tr):
            if t < 0:
                continue
            tot += 1
            if puzzle == "copy":
                L = (len(row) - 2) // 2
                hit += int(row[p - L] == t)
            else:
                cue = tuple(row[p - key_len + 1: p + 1])
                for s in range(1, len(row) - key_len):
                    if tuple(row[s: s + key_len]) == cue and s + key_len < p - key_len and row[s - 1] != QUERY:
                        hit += int(row[s + key_len] == t)
                        break
    return hit / max(1, tot)


def sample_batch(a, g, voc, puzzle, train=True, setting=None, B=None):
    B = B or a.B
    if puzzle == "recall":
        gap = setting if setting is not None else int(_rand(g, 0, a.train_gap + 1, (1,)))
        return make_recall(g, B, a.n_pairs, gap, a.n_queries, voc)
    if puzzle == "copy":
        L = setting if setting is not None else int(_rand(g, 8, a.train_len + 1, (1,)))
        return make_copy(g, B, L, voc)
    if setting is None:
        gap, corrupt = int(_rand(g, 0, a.train_gap + 1, (1,))), a.train_corrupt
    else:
        gap, corrupt = a.blurry_gap, setting
    return make_blurry(g, B, a.n_pairs, a.key_len, gap, a.n_queries, corrupt, voc)


def settings_of(a, puzzle):
    return {"recall": a.eval_gaps, "copy": a.eval_lens, "blurry": a.eval_corrupts}[puzzle]


@torch.no_grad()
def eval_setting(model, a, voc, puzzle, setting, device, seed=12345):
    model.eval()
    g = torch.Generator().manual_seed(seed + int(setting * 1000 if isinstance(setting, float) else setting))
    probe, _ = sample_batch(a, g, voc, puzzle, setting=setting, B=1)
    T = probe.shape[1]
    bs = max(1, min(a.eval_n, a.eval_tokens // T))
    hit = tot = 0
    t0 = time.time()
    done = 0
    while done < a.eval_n:
        b = min(bs, a.eval_n - done)
        idx, tgt = sample_batch(a, g, voc, puzzle, setting=setting, B=b)
        idx, tgt = idx.to(device), tgt.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.bf16 and device.type == "cuda"):
            logits = model(idx)
        m = tgt >= 0
        hit += int((logits.argmax(-1)[m] == tgt[m]).sum())
        tot += int(m.sum())
        done += b
    return {"setting": setting, "T": T, "accuracy": hit / max(1, tot), "n_scored": tot, "n_sequences": done,
            "seconds": round(time.time() - t0, 3)}


def param_groups(model, a):
    mem = OM.OneSdmLM.memory_parameters(model) if isinstance(model, OM.OneSdmLM) else set()
    dec, nodec, memp, muon = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in mem:
            memp.append(p)
        elif p.dim() == 2 and "emb" not in n:
            (muon if a.opt == "muon" else dec).append(p)
        else:
            nodec.append(p)
    groups = [{"params": dec, "weight_decay": 0.1, "lr_mult": 1.0}, {"params": nodec, "weight_decay": 0.0, "lr_mult": 1.0}]
    if memp:
        groups.append({"params": memp, "weight_decay": 0.0, "lr_mult": a.mem_lr_mult})
    adam = torch.optim.AdamW([gr for gr in groups if gr["params"]], lr=a.lr, betas=(0.9, 0.95))
    mu = O.Muon(muon, lr=a.muon_lr, lr_mult=a.muon_lr / a.lr) if muon else None
    return adam, mu, memp


def lr_at(step, total, peak):
    warm = max(1, int(0.02 * total))
    if step < warm:
        return peak * (step + 1) / warm
    pr = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pr)))


def grad_norm(params):
    gs = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.0


def stamps(device):
    s = {"host": platform.node(), "torch": torch.__version__, "device": str(device), "python": platform.python_version(),
         "loadavg": list(os.getloadavg()), "cpu_count": os.cpu_count(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if device.type == "cuda":
        s["cuda_device"] = torch.cuda.get_device_name(device)
        s["cuda_runtime"] = torch.version.cuda
        try:
            s["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            s["driver"] = "unknown"
    try:
        s["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, timeout=10).stdout.strip()
    except Exception:
        s["git_sha"] = "unknown"
    return s


def model_cfg(a, max_len):
    c = {"d": a.d, "layers": a.layers, "mlp_f": a.mlp_f, "heads": a.heads, "n_sub": a.n_sub, "d_a": a.d_a, "k": a.k,
         "decays": [float(x) for x in a.decays.split(",")], "learn_decay": a.learn_decay, "write_gate": not a.no_gate,
         "gate_bias": a.gate_bias, "read_mode": a.read_mode, "out_norm": not a.no_out_norm, "chunk": a.chunk,
         "mem_in_grad": a.mem_in_grad, "addr_bias": a.addr_bias, "arc_d_c": a.arc_d_c, "arc_n_sub": a.arc_n_sub, "arc_k": a.arc_k,
         "arc_slot_keep": a.arc_slot_keep, "arc_window": a.arc_window, "arc_n_fetch": a.arc_n_fetch,
         "arc_weighting": a.arc_weighting, "max_len": max_len, "sdm_n_sub": a.sdm_n_sub, "sdm_heads": a.sdm_heads,
         "sdm_d_a": a.sdm_d_a, "sdm_k": a.sdm_k, "sdm_scale": a.sdm_scale, "sdm_big_mult": a.sdm_big_mult}
    if a.arc_layers:
        c["arc_layers"] = [int(x) for x in a.arc_layers.split(",")]
    return c


def max_len_needed(a, voc):
    g = torch.Generator().manual_seed(0)
    out = 0
    for pz in a.puzzles:
        for st in settings_of(a, pz):
            out = max(out, sample_batch(a, g, voc, pz, setting=st, B=1)[0].shape[1])
        # the longest TRAINING batch too: blurry trains at gaps up to --train-gap but is tested at --blurry-gap, so
        # sizing from the eval settings alone left the yardstick's position table short (250 > 181 on the Spark)
        if pz == "recall":
            out = max(out, make_recall(g, 1, a.n_pairs, a.train_gap, a.n_queries, voc)[0].shape[1])
        elif pz == "copy":
            out = max(out, make_copy(g, 1, a.train_len, voc)[0].shape[1])
        else:
            out = max(out, make_blurry(g, 1, a.n_pairs, a.key_len, a.train_gap, a.n_queries, a.train_corrupt, voc)[0].shape[1])
    return out + 16


def train_puzzle(a, puzzle, arm, seed, device, voc):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    cfg = {**model_cfg(a, max_len_needed(a, voc)), "seed": seed}
    model = OM.build_onesdm(arm, voc.V, cfg).to(device)
    adam, mu, memp = param_groups(model, a)
    others = [p for p in model.parameters() if p.requires_grad and all(p is not q for q in memp)]
    curve = []
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        lr = lr_at(step, a.steps, a.lr)
        for gr in adam.param_groups:
            gr["lr"] = lr * gr["lr_mult"]
        if mu:
            mu.param_groups[0]["lr"] = lr * mu.param_groups[0]["lr_mult"]
        idx, tgt = sample_batch(a, g, voc, puzzle)
        idx, tgt = idx.to(device), tgt.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.bf16 and device.type == "cuda"):
            logits = model(idx)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        adam.zero_grad(set_to_none=True)
        if mu:
            mu.zero_grad()
        loss.backward()
        gm, go = grad_norm(memp), grad_norm(others)
        if a.mem_clip > 0 and memp:
            torch.nn.utils.clip_grad_norm_(memp, a.mem_clip)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), a.clip)
        adam.step()
        if mu:
            mu.step()
        if not math.isfinite(float(loss.detach())):
            curve.append({"step": step, "loss": float(loss), "event": "non-finite loss, stopped"})
            break
        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                m = tgt >= 0
                acc = float((logits.argmax(-1)[m] == tgt[m]).float().mean())
            rec = {"step": step, "loss": round(loss.item(), 4), "train_acc": round(acc, 4), "grad_mem": round(gm, 4),
                   "grad_rest": round(go, 4), "lr": lr, "T": idx.shape[1], "sec": round(time.time() - t0, 1)}
            curve.append(rec)
            print(json.dumps({"puzzle": puzzle, "arm": arm, "seed": seed, **rec}), flush=True)
    train_sec = time.time() - t0
    evals = [eval_setting(model, a, voc, puzzle, st, device) for st in settings_of(a, puzzle)]
    for e in evals:
        print(json.dumps({"puzzle": puzzle, "arm": arm, "seed": seed, "eval": e}), flush=True)
    res = {"lane": "ONESDMSTREAM", "kind": "onesdm_puzzle", "puzzle": puzzle, "arm": arm, "seed": seed,
           "label": a.label, "model_cfg": cfg, "params": OM.param_count(model), "vocab": vars(voc),
           "train": {k: getattr(a, k) for k in ("steps", "B", "lr", "mem_lr_mult", "mem_clip", "clip", "opt", "muon_lr",
                                                 "bf16", "train_gap", "train_len", "train_corrupt", "n_pairs", "n_queries",
                                                 "key_len", "blurry_gap", "eval_n")},
           "train_seconds": round(train_sec, 1), "curve": curve, "eval": evals, "stamps": stamps(device)}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"onesdm_{a.label}_{puzzle}_{arm}_s{seed}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({"wrote": path}), flush=True)
    return res


@torch.no_grad()
def cost_bench(a, device, voc):
    out = []
    for arm in a.arms:
        cfg = {**model_cfg(a, max(a.cost_lengths) + 16), "seed": 0}
        model = OM.build_onesdm(arm, voc.V, cfg).to(device).eval()
        for T in a.cost_lengths:
            idx = torch.randint(4, voc.V, (1, T), generator=torch.Generator().manual_seed(T)).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            model(idx)  # warm
            best = float("inf")
            for _ in range(a.cost_reps):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(idx)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
            rec = {"arm": arm, "T": T, "sec_per_token": best / T, "sec_total": best,
                   "decode_state_bytes": OM.decode_state_bytes(model, T),
                   "decode_state_bytes_per_token": OM.decode_state_bytes(model, T) / T,
                   "peak_mem_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "params": OM.param_count(model)}
            out.append(rec)
            print(json.dumps(rec), flush=True)
    res = {"lane": "ONESDMSTREAM", "kind": "onesdm_cost", "label": a.label, "model_cfg": model_cfg(a, max(a.cost_lengths) + 16),
           "reps": a.cost_reps, "rows": out, "stamps": stamps(device),
           "note": "forward pass over the whole sequence, B 1, best of reps after one warm run; on the CPU no peak memory"}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"onesdm_{a.label}_cost.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({"wrote": path}), flush=True)
    return res


def ints(s):
    return [int(x) for x in s.split(",") if x]


def parse(argv=None):
    p = argparse.ArgumentParser(
        description="ONESDM puzzles. Examples:\n"
                    "  python3 %(prog)s --selftest\n"
                    "  python3 %(prog)s --puzzles recall --arms onesdm,plain --steps 3000\n"
                    "  python3 %(prog)s --cost --arms onesdm,sdmonly,yardstick_transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--cost", action="store_true", help="run the cost bench instead of the puzzles")
    p.add_argument("--puzzles", default="recall,copy,blurry")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--seeds", default="0")
    p.add_argument("--label", default="run", help="a name carried into every output file")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--out", default=os.environ.get("ONESDM_RUNS", os.path.join(HERE, "runs_onesdm")))
    g = p.add_argument_group("model (all arms share the width, depth and MLP)")
    g.add_argument("--d", type=int, default=256)
    g.add_argument("--layers", type=int, default=4)
    g.add_argument("--mlp-f", type=int, default=688)
    g.add_argument("--heads", type=int, default=4)
    g.add_argument("--n-sub", type=int, default=32, help="hard locations per head = n_sub squared")
    g.add_argument("--d-a", type=int, default=32)
    g.add_argument("--k", type=int, default=16, help="slots each write and read touches")
    g.add_argument("--decays", default="0.0,0.8,0.99,1.0", help="one fade per head (0 = previous token, 1 = never)")
    g.add_argument("--learn-decay", action="store_true")
    g.add_argument("--no-gate", action="store_true", help="write gate fixed at 1")
    g.add_argument("--gate-bias", type=float, default=0.0)
    g.add_argument("--read-mode", default="mean", choices=["mean", "sum"])
    g.add_argument("--no-out-norm", action="store_true")
    g.add_argument("--chunk", type=int, default=128)
    g.add_argument("--addr-bias", type=float, default=0.0, help="init std of a learned per-head bias shared by key and query")
    g.add_argument("--mem-in-grad", type=float, default=1.0, help="gradient scale from the memory into the residual")
    g.add_argument("--arc-layers", default="", help="archive layers (default: every layer but the first)")
    g.add_argument("--arc-d-c", type=int, default=64)
    g.add_argument("--arc-n-sub", type=int, default=16)
    g.add_argument("--arc-k", type=int, default=4)
    g.add_argument("--arc-slot-keep", type=int, default=4)
    g.add_argument("--arc-window", type=int, default=16)
    g.add_argument("--arc-n-fetch", type=int, default=4)
    g.add_argument("--arc-weighting", default="soft", choices=["soft", "st"])
    g.add_argument("--sdm-n-sub", type=int, default=0, help="allsdm arms: trained-table n_sub (0 = matched to the MLP's weights)")
    g.add_argument("--sdm-heads", type=int, default=4, help="allsdm arms: trained-table heads")
    g.add_argument("--sdm-d-a", type=int, default=0, help="allsdm arms: query width per head (0 = min(32, d / heads))")
    g.add_argument("--sdm-k", type=int, default=32, help="allsdm arms: slots each token reads per head")
    g.add_argument("--sdm-scale", type=float, default=8.0, help="allsdm arms: initial softmax scale (learned)")
    g.add_argument("--sdm-big-mult", type=int, default=4, help="onesdm_allsdm_big: n_sub multiplier (4 = 16x the slots)")
    t = p.add_argument_group("training")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--B", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--mem-lr-mult", type=float, default=1.0)
    t.add_argument("--mem-clip", type=float, default=0.0, help="clip the memory parameters' gradient to this norm first")
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--opt", default="adamw", choices=["adamw", "muon"])
    t.add_argument("--muon-lr", type=float, default=0.02)
    t.add_argument("--bf16", action="store_true", help="bf16 autocast on CUDA (the memory itself always runs in float32)")
    t.add_argument("--log-every", type=int, default=100)
    z = p.add_argument_group("puzzles")
    z.add_argument("--n-keys", type=int, default=128)
    z.add_argument("--n-vals", type=int, default=128)
    z.add_argument("--n-noise", type=int, default=128)
    z.add_argument("--n-pairs", type=int, default=8)
    z.add_argument("--n-queries", type=int, default=4)
    z.add_argument("--key-len", type=int, default=4)
    z.add_argument("--train-gap", type=int, default=256)
    z.add_argument("--train-len", type=int, default=128)
    z.add_argument("--train-corrupt", type=float, default=0.0)
    z.add_argument("--blurry-gap", type=int, default=100)
    z.add_argument("--eval-gaps", default="100,1000,4000,16000")
    z.add_argument("--eval-lens", default="128,512,2048")
    z.add_argument("--eval-corrupts", default="0,0.25,0.5,0.75")
    z.add_argument("--eval-n", type=int, default=128, help="sequences per evaluated setting")
    z.add_argument("--eval-tokens", type=int, default=65536, help="tokens per evaluation batch (memory only)")
    c = p.add_argument_group("cost")
    c.add_argument("--cost-lengths", default="1024,4096,16384")
    c.add_argument("--cost-reps", type=int, default=3)
    a = p.parse_args(argv)
    a.puzzles = [x for x in a.puzzles.split(",") if x]
    a.arms = [x for x in a.arms.split(",") if x]
    assert all(x in ALL_ARMS for x in a.arms), a.arms
    assert all(x in ("recall", "copy", "blurry") for x in a.puzzles), a.puzzles
    a.seeds = ints(a.seeds)
    a.eval_gaps, a.eval_lens = ints(a.eval_gaps), ints(a.eval_lens)
    a.eval_corrupts = [float(x) for x in a.eval_corrupts.split(",") if x]
    a.cost_lengths = ints(a.cost_lengths)
    return a


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def selftest():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name, flush=True)

    t_all = time.time()
    torch.manual_seed(0)
    voc = Vocab()
    g = torch.Generator().manual_seed(0)
    # 1. layouts: lengths, targets in place, the oracle solves clean data exactly
    idx, tgt = make_recall(g, 4, 8, 100, 4, voc)
    check("recall: length 1 + 2*pairs + gap + 3*queries, 4 scored answers per row",
          idx.shape == (4, 1 + 16 + 100 + 12) and bool(((tgt >= 0).sum(1) == 4).all()))
    check("recall: every answer is a value symbol and the queried key sits at the scored position",
          bool(((tgt[tgt >= 0] >= voc.v0) & (tgt[tgt >= 0] < voc.n0)).all())
          and bool(((idx[tgt >= 0] >= voc.k0) & (idx[tgt >= 0] < voc.v0)).all()))
    check("recall: the dictionary oracle scores 1.0", oracle_accuracy(idx, tgt, "recall") == 1.0)
    idx, tgt = make_copy(g, 3, 20, voc)
    check("copy: oracle 1.0 and 20 scored positions per row", oracle_accuracy(idx, tgt, "copy") == 1.0 and bool(((tgt >= 0).sum(1) == 20).all()))
    clean_i, clean_t = make_blurry(torch.Generator().manual_seed(5), 4, 8, 4, 30, 4, 0.0, voc)
    blur_i, blur_t = make_blurry(torch.Generator().manual_seed(5), 4, 8, 4, 30, 4, 0.5, voc)
    check("blurry: clean cue oracle 1.0", oracle_accuracy(clean_i, clean_t, "blurry", key_len=4) == 1.0)
    diff = (clean_i != blur_i).sum(1)
    check("blurry: corruption 0.5 changes exactly 2 of 4 symbols in each of 4 queries (8 per row), nothing else",
          bool((diff == 8).all()) and torch.equal(clean_t, blur_t))
    check("blurry: corrupted symbols stay in the key alphabet",
          bool(((blur_i[clean_i != blur_i] >= voc.k0) & (blur_i[clean_i != blur_i] < voc.v0)).all()))
    # 2. the targets never peek: an answer token never sits at or before its scored position
    idx, tgt = make_recall(g, 2, 4, 10, 2, voc)
    pos = (tgt[0] >= 0).nonzero().flatten()
    check("recall: the answer is the NEXT token of each scored position", all(int(idx[0, p + 1]) == int(tgt[0, p]) for p in pos))
    # 3. positive control: a tiny recall puzzle a memory model learns and the no-memory stack cannot
    small = Vocab(8, 8, 8)
    a = parse(["--d", "64", "--layers", "2", "--mlp-f", "128", "--heads", "2", "--n-sub", "8", "--d-a", "8", "--k", "8",
               "--decays", "0.0,1.0", "--steps", "300", "--B", "32", "--lr", "3e-3", "--n-pairs", "1", "--n-queries", "1",
               "--train-gap", "12", "--eval-gaps", "12", "--eval-n", "512", "--puzzles", "recall", "--log-every", "1000",
               "--label", "selftest", "--out", os.path.join(os.environ.get("TMPDIR", "/tmp"), "onesdm_selftest")])
    dev = torch.device("cpu")
    t0 = time.time()
    r_mem = train_puzzle(a, "recall", "onesdm", 0, dev, small)
    r_none = train_puzzle(a, "recall", "plain", 0, dev, small)
    am, an = r_mem["eval"][0]["accuracy"], r_none["eval"][0]["accuracy"]
    print(f"positive control (CPU, 300 steps, {time.time() - t0:.0f} s): memory {am:.3f}, no memory {an:.3f}, chance 0.125", flush=True)
    check("positive control: the memory model learns one-pair recall (accuracy > 0.9)", am > 0.9)
    check("positive control: the no-memory stack cannot (accuracy < 0.3)", an < 0.3)
    check("positive control: the JSON carries config, curve, eval and stamps",
          all(k in r_mem for k in ("model_cfg", "curve", "eval", "stamps", "params")) and len(r_mem["curve"]) > 0)
    # 3b. the same control for the fully-SDM model: run-time SDM in place of attention, trained SDM in place of the MLP
    t0 = time.time()
    r_all = train_puzzle(a, "recall", "onesdm_allsdm", 0, dev, small)
    aa = r_all["eval"][0]["accuracy"]
    print(f"positive control (CPU, 300 steps, {time.time() - t0:.0f} s): onesdm_allsdm {aa:.3f}, chance 0.125", flush=True)
    check("positive control: onesdm_allsdm learns one-pair recall (accuracy > 0.9)", aa > 0.9)
    # 4. a tiny cost run: the SDM model's decode state is flat, the yardstick's grows
    a2 = parse(["--cost", "--arms", "onesdm,yardstick_transformer", "--cost-lengths", "64,256", "--cost-reps", "1",
                "--d", "32", "--layers", "1", "--mlp-f", "48", "--heads", "4", "--n-sub", "6", "--d-a", "8", "--k", "4",
                "--label", "selftest", "--out", a.out])
    rows = cost_bench(a2, dev, small)["rows"]
    st = {(r["arm"], r["T"]): r["decode_state_bytes"] for r in rows}
    check("cost: onesdm decode state equal at T 64 and 256; the yardstick's grows 4x",
          st[("onesdm", 64)] == st[("onesdm", 256)] and st[("yardstick_transformer", 256)] == 4 * st[("yardstick_transformer", 64)])
    a2b = parse(["--cost", "--arms", "onesdm_allsdm,onesdm_allsdm_big", "--cost-lengths", "64,256", "--cost-reps", "1",
                 "--d", "32", "--layers", "1", "--mlp-f", "48", "--heads", "4", "--n-sub", "6", "--d-a", "8", "--k", "4",
                 "--label", "selftest_allsdm", "--out", a.out])
    rows_b = cost_bench(a2b, dev, small)["rows"]
    st_b = {(r["arm"], r["T"]): r["decode_state_bytes"] for r in rows_b}
    check("cost: both allsdm arms run in the bench and their decode state is equal at T 64 and 256",
          len(rows_b) == 4 and all(st_b[(arm, 64)] == st_b[(arm, 256)] for arm in OM.ALLSDM_ARMS))
    # 5. the length every model is built for covers every TRAINING batch, not only the eval settings (the blurry
    # yardstick died on the Spark: trained at gap up to 256, built for the eval gap of 100)
    # one puzzle a run, as the chain runs them, or the 16,000 recall eval gap hides the gap for the others
    voc3 = Vocab()
    fits = []
    for pz in ("recall", "copy", "blurry"):
        a3 = parse(["--puzzles", pz, "--label", "selftest", "--out", a.out])
        need = max_len_needed(a3, voc3)
        longest = max(sample_batch(a3, torch.Generator().manual_seed(s), voc3, pz, B=1)[0].shape[1] for s in range(200))
        fits.append(f"{pz} {longest}<={need}" if need >= longest else f"{pz} {longest}>{need} SHORT")
    check(f"max_len_needed covers the longest of 200 sampled training batches, each puzzle alone ({', '.join(fits)})",
          not any("SHORT" in f for f in fits))
    print(f"{sum(ok)} of {len(ok)} checks pass ({time.time() - t_all:.0f} s)", flush=True)
    return all(ok)


def main():
    a = parse()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    device = pick_device(a.device)
    voc = Vocab(a.n_keys, a.n_vals, a.n_noise)
    if a.cost:
        cost_bench(a, device, voc)
        return
    for puzzle in a.puzzles:
        for arm in a.arms:
            for seed in a.seeds:
                train_puzzle(a, puzzle, arm, seed, device, voc)
    print("NEXT -> python3 track4_onesdm_puzzles.py --cost --arms " + ",".join(a.arms))


if __name__ == "__main__":
    main()
