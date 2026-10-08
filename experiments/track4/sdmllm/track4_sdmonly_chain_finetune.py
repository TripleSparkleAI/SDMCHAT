"""SDMONLY chain fine-tune: SDM BASE -> SDM CHAT -> WEIRD LITTLE GUY, for the SDM-only arms, with a row mix in every batch.

<claudes_code_comments>
** Function List **
resolve_shard(spec, registry, data) - a shard name or name=path -> its token file, loss mask and document starts
Shard.window(rng, T) - one window of T+1 tokens and its loss mask (a document start for part of the rows)
alloc_rows(fracs, B) - rows per shard in one batch: floor of frac x B, the rest by largest remainder
mix_batch(shards, rows, seed, rnd, step, T, doc_frac) - the batch for (seed, round, step): tokens, mask, row source, offsets
infer_build_cfg(ck, result_json) - the full build_sdmonly config of a checkpoint, from its own cfg and its weight shapes
load_init(path, device, result_json) - a checkpoint or run folder -> (model, arm, build_cfg, source path)
token_nll(model, x, m, loss, chunk) - per-token cross-entropy over the masked targets (plain or checkpointed chunks)
masked_bpb(model, sh, T, tb, device, max_windows) - bits per byte over a masked shard's scored tokens
score(model, a, names, ...) - every --extra-val shard: S0's eval_bpb for plain shards, masked_bpb for masked ones
make_opt(model, a) - AdamW with three groups: decay, no decay, the store (own lr multiplier and weight decay)
lr_fn(a) - the schedule: S0's cosine or the SDMONLY trainer's warmup-constant-decay
save_ck(path, ...) - atomic checkpoint with the full build cfg inside
run_chain(a) - round 0 (the init, scored), then rounds 1..N, each resumable, scored and checkpointed
print_table(runs_dir, run, names) - the per-round scores table from the result files
parse(argv) - flags
selftest() - row mix fractions, seeded batches, loss falls, checkpoint round trip, rebuild from shapes, resume equals straight
main() - CLI

** Technical Review **
- What it fine-tunes: any checkpoint of the SDM-only family (arms sdmonly, sdmonly_dense, sdmonly_none) from the
  SDMONLY trainer (track4_sdmonly_train.py, i.e. S0's train()) or from this file. The model is rebuilt from the
  checkpoint alone: a checkpoint written here carries `build_cfg` (every build_sdmonly argument); an S0-trainer
  checkpoint carries the S0 cfg (d, k, hops, softness, n_back, decays, qgrad, seed), and the rest (d_a, n_sub,
  heads, readout_f, share_store, dense_f) is read off the weight shapes, then cross-checked against the run's
  result json ("sdmonly" block) when one is found. load_state_dict is strict, so a wrong guess cannot load.
- The row mix: --mix guy:0.5,chat_mix:0.25,train_big:0.25. Fractions must sum to 1. Rows per batch are floor(f x B)
  with the remainder given by largest remainder (ties to the earlier shard), so at B 32 those fractions are exactly
  16, 8 and 8 rows in EVERY batch, not on average. The offsets of step s in round r come from
  numpy default_rng([seed, r, s, shard_index]), so a resumed run sees the batches a straight run would have seen.
- Shards: a name resolves to $SDMLLM_DATA/<name>.u32; `--shard name=path` registers any other .u32 file. A shard
  with <stem>_mask.u8 beside it is MASKED (the guy's prep: only his replies and the BOS that ends a document are
  in the loss) and with <stem>_starts.i64 it starts --doc-start-frac of its rows at a document start (the old
  guy recipe, 0.7). A plain shard trains on every target, as the S0 trainer does.
- Loss: mean cross-entropy over all loss tokens of the batch (so a web row of 256 targets weighs more than a guy row
  of fewer). bf16 autocast for matmuls, fp32 cross-entropy, clip 1.0. --loss chunked recomputes the tied head in
  checkpointed chunks over the selected positions only (CUDA memory); plain forms the logits of the selected rows.
  The log records each shard's share of loss tokens and its own mean loss.
- Optimiser: AdamW (0.9, 0.95), weight decay 0.1 on 2-D non-embedding weights, the store parameters in their own
  group (--store-lr-mult, --store-wd), as S0. Each round runs its own schedule over --tokens (warmup --warmup,
  then cosine to 10% or wsd), with the peak multiplied by --round-lr-mult per round. The optimiser state is
  carried from round to round.
- Rounds and files: <SDMLLM_CKPT>/<run>/round_<r>/last.pt (saved every --ckpt-every steps, so a round resumes),
  <run>/last.pt (a copy of the newest finished round, the input of the next link of the chain),
  <SDMLLM_RUNS>/<run>.round_<r>.result.json (round 0 is the init, scored before any step) and <run>.log.
  A round whose result json exists and whose checkpoint is at its last step is skipped on a re-run.
- Scores after each round (--extra-val): `val` uses S0's TEST windows of val.u32 and S0.eval_bpb; any other plain
  shard uses S0.eval_bpb over windows of the whole shard (S0's extra-val rule); a masked shard uses masked_bpb
  (non-overlapping windows from the start, only loss tokens scored). bpb = nats / (ln 2 x UTF-8 bytes of the
  scored targets). With --val-windows unset the full TEST per-window file is also written per round, for the
  paired comparison tool.
- Every result json carries S0.stamp() at start and end (load, power mode, driver, torch, host).
Docs: TRAINING_SDMONLY_HOW_WE_TRAIN.md · PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md (steps 3 and 4)
</claudes_code_comments>
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SDMLLM_RUNS", os.path.join(HERE, "runs_sdmonly"))
import track4_sdmllm_train_one_arm as S0  # noqa: E402
import track4_sdmonly_models as M  # noqa: E402
from track4_sdmonly_train import wsd_lr  # noqa: E402

BOS = 0
ARMS = ("sdmonly", "sdmonly_dense", "sdmonly_none")
EXAMPLE = """examples:
  # SDM CHAT from SDM BASE: one round, half chat_mix half web text
  python3 track4_sdmonly_chain_finetune.py --init ck/base_run --run chat1 --mix chat_mix:0.5,train_big:0.5 \\
      --tokens 200000000 --lr 1e-3 --extra-val val --extra-val chat_test
  # WEIRD LITTLE GUY from SDM CHAT: three rounds, his corpus with web and chat rows in every batch
  python3 track4_sdmonly_chain_finetune.py --init ck/chat1 --run guy1 --rounds 3 \\
      --shard guy=~/settle24/weirdguy/prep/train.u32 --shard guy_test=~/settle24/weirdguy/prep/test.u32 \\
      --mix guy:0.5,chat_mix:0.25,train_big:0.25 --tokens 4000000 --lr 5e-4 \\
      --extra-val guy_test --extra-val val --extra-val chat_test
  # the table of an existing chain
  python3 track4_sdmonly_chain_finetune.py --table guy1
  python3 track4_sdmonly_chain_finetune.py --selftest"""


# ---------------------------------------------------------------- shards and the row mix
class Shard:
    def __init__(self, name, path):
        self.name, self.path = name, path
        self.ids = np.memmap(path, dtype=np.uint32, mode="r")
        stem = path[:-4] if path.endswith(".u32") else path
        mp, sp = stem + "_mask.u8", stem + "_starts.i64"
        self.mask = np.memmap(mp, dtype=np.uint8, mode="r") if os.path.exists(mp) else None
        self.starts = np.fromfile(sp, dtype=np.int64) if os.path.exists(sp) else None
        if self.mask is not None and len(self.mask) != len(self.ids):
            raise ValueError(f"shard {name}: mask has {len(self.mask)} entries, tokens {len(self.ids)}")
        self.masked = self.mask is not None

    def window(self, rng, T, doc_frac):
        n = len(self.ids)
        if self.starts is not None and len(self.starts) and rng.random() < doc_frac:
            a = int(self.starts[rng.integers(len(self.starts))])
        else:
            a = int(rng.integers(0, n - T - 1))
        a = min(a, n - T - 1)
        x = np.asarray(self.ids[a:a + T + 1]).astype(np.int64)
        m = np.asarray(self.mask[a:a + T + 1]).astype(np.bool_) if self.masked else np.ones(T + 1, dtype=np.bool_)
        return x, m, a

    def info(self):
        return {"name": self.name, "path": self.path, "tokens": int(len(self.ids)), "masked": self.masked,
                "loss_tokens": int(self.mask.sum()) if self.masked else int(len(self.ids)),
                "docs": int(len(self.starts)) if self.starts is not None else None}


def resolve_shard(spec, registry, data):
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        name = spec
        path = registry.get(name) or os.path.join(data, f"{name}.u32")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"shard {name}: {path} not found (register it with --shard {name}=PATH)")
    return Shard(name, path)


def parse_mix(s):
    out = []
    for part in s.split(","):
        name, frac = part.rsplit(":", 1)
        out.append((name.strip(), float(frac)))
    tot = sum(f for _, f in out)
    if abs(tot - 1.0) > 1e-6 or any(f < 0 for _, f in out):
        raise ValueError(f"--mix fractions must be non-negative and sum to 1, got {tot}")
    return out


def alloc_rows(fracs, B):
    raw = [f * B for f in fracs]
    rows = [int(math.floor(r + 1e-9)) for r in raw]
    order = sorted(range(len(fracs)), key=lambda i: (-(raw[i] - rows[i]), i))
    for i in order[: B - sum(rows)]:
        rows[i] += 1
    return rows


def mix_batch(shards, rows, seed, rnd, step, T, doc_frac):
    xs, ms, src, offs = [], [], [], []
    for si, (sh, n) in enumerate(zip(shards, rows)):
        g = np.random.default_rng([seed, rnd, step, si])
        for _ in range(n):
            x, m, a = sh.window(g, T, doc_frac)
            xs.append(x)
            ms.append(m)
            src.append(si)
            offs.append(a)
    return np.stack(xs), np.stack(ms), np.array(src), np.array(offs)


# ---------------------------------------------------------------- the model from a checkpoint
def _n_index(sd, prefix):
    idx = {int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)}
    return max(idx) + 1 if idx else 0


def infer_build_cfg(ck, result_json=None):
    if ck.get("build_cfg"):
        return dict(ck["build_cfg"])
    sd, cfg, arm = ck["model"], ck.get("cfg", {}), ck["arm"]
    if arm not in ARMS:
        raise ValueError(f"arm {arm} is not an SDM-only arm {ARMS}")
    out = {k: cfg[k] for k in ("k", "softness", "n_back", "qgrad", "seed") if k in cfg}
    out["decays"] = [float(x) for x in cfg.get("decays", (0.5, 0.8, 0.9, 0.97, 0.99))]
    out["d"] = int(sd["emb.weight"].shape[1])
    out["hops"] = _n_index(sd, "qnorms.")
    out["n_back"] = _n_index(sd, "fnorms.") - len(out["decays"])
    out["readout_f"] = int(sd["readout.gate.weight"].shape[0]) if "readout.gate.weight" in sd else 0
    if arm == "sdmonly":
        heads, n_sub, half = sd["store.banks.0.keys1"].shape
        out.update(heads=int(heads), n_sub=int(n_sub), d_a=int(2 * half),
                   share_store=_n_index(sd, "store.banks.") == 1 and out["hops"] > 1)
    if arm == "sdmonly_dense":
        out["dense_f"] = int(sd["mlps.0.gate.weight"].shape[0])
    if result_json and os.path.exists(result_json):
        extra = json.load(open(result_json)).get("sdmonly", {})
        for key in ("d_a", "n_sub", "heads", "readout_f", "share_store"):
            if key in extra and key in out and extra[key] != out[key]:
                raise ValueError(f"{key}: weights say {out[key]}, {result_json} says {extra[key]}")
    return out


def find_ck(path):
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path
    if os.path.isfile(os.path.join(path, "last.pt")):
        return os.path.join(path, "last.pt")
    rounds = sorted((int(d.split("_")[1]), d) for d in os.listdir(path) if d.startswith("round_"))
    for _, d in reversed(rounds):
        if os.path.isfile(os.path.join(path, d, "last.pt")):
            return os.path.join(path, d, "last.pt")
    raise FileNotFoundError(f"no checkpoint at {path}")


def load_init(path, device, result_json=None, V=None):
    p = find_ck(path)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if result_json is None:
        run = os.path.basename(os.path.dirname(p))
        guess = os.path.join(os.environ["SDMLLM_RUNS"], f"{run}.result.json")
        result_json = guess if os.path.exists(guess) else None
    bc = infer_build_cfg(ck, result_json)
    V = V or int(ck["model"]["emb.weight"].shape[0])
    model = M.build_sdmonly(ck["arm"], V, bc)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device), ck["arm"], bc, p


# ---------------------------------------------------------------- loss and scores
def token_nll(model, x, m, loss="plain", chunk=2048):
    from torch.utils.checkpoint import checkpoint
    h = model.hidden(x[:, :-1])
    sel = m[:, 1:].reshape(-1)
    hs = h.reshape(-1, h.shape[-1])[sel]
    t = x[:, 1:].reshape(-1)[sel]
    E = model.emb.weight
    if loss == "plain":
        return F.cross_entropy((hs @ E.t()).float(), t, reduction="none"), sel

    def part(hc, tc):
        return F.cross_entropy((hc @ E.t()).float(), tc, reduction="none")
    return torch.cat([checkpoint(part, hs[i:i + chunk], t[i:i + chunk], use_reentrant=False)
                      for i in range(0, hs.shape[0], chunk)]), sel


@torch.no_grad()
def masked_bpb(model, sh, T, tb, device, max_windows=None, bs=16):
    model.eval()
    starts = np.arange(0, len(sh.ids) - (T + 1), T + 1)
    if max_windows:
        starts = starts[:max_windows]
    nats, nbytes, ntok = 0.0, 0, 0
    for i in range(0, len(starts), bs):
        s = starts[i:i + bs]
        x = torch.from_numpy(np.stack([sh.ids[a:a + T + 1] for a in s]).astype(np.int64)).to(device)
        mm = torch.from_numpy(np.stack([sh.mask[a:a + T + 1] for a in s]).astype(np.bool_)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
            nll, sel = token_nll(model, x, mm, "plain")
        t = x[:, 1:].reshape(-1)[sel]
        nats += float(nll.sum())
        nbytes += int(tb[t].sum())
        ntok += int(sel.sum())
    model.train()
    return {"bpb": nats / (math.log(2) * max(1, nbytes)), "nats_per_token": nats / max(1, ntok), "tokens": ntok,
            "bytes": nbytes, "windows": int(len(starts)), "rule": "masked: only loss tokens"}


def score(model, a, shards_val, tb, device, ck_dir=None):
    out = {}
    for name, sh in shards_val.items():
        if sh.masked:
            out[name] = masked_bpb(model, sh, a.T, tb, device, a.val_max_windows)
        else:
            half, cap = ("test", a.val_windows) if name == "val" else ("all", a.val_max_windows)
            starts = S0.val_windows(sh.ids, a.T, half, cap)
            out[name] = {**S0.eval_bpb(model, sh.ids, starts, a.T, tb, device),
                         "windows": int(len(starts)), "rule": f"S0 eval_bpb, {half} windows"}
            if name == "val" and a.val_windows is None and ck_dir:
                from track4_sdmllm_paired_window_comparison import per_window
                model.eval()
                pn, pb = per_window(model, sh.ids, starts, a.T, tb, device)
                np.savez(os.path.join(ck_dir, "test_per_window.npz"), nats=pn, bytes=pb)
                model.train()
        print(f"[chain] score {name}: bpb {out[name]['bpb']:.5f} ({out[name]['windows']} windows)", flush=True)
    model.train()
    return out


# ---------------------------------------------------------------- training
def make_opt(model, a):
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
    return torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)


def lr_fn(a):
    if a.sched == "wsd":
        return wsd_lr(a.wsd_decay)
    return lambda step, total, peak, warm: S0.lr_at(step, total, peak, warm, 0.1)


def save_ck(path, model, opt, step, total, arm, bc, a, rnd, init_path):
    cfg = {**bc, "T": a.T, "train_name": "mix:" + a.mix}
    obj = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg, "build_cfg": bc, "arm": arm,
           "chain": {"round": rnd, "steps_total": total, "init": init_path, "mix": a.mix, "tokens": a.tokens,
                     "lr": a.lr, "seed": a.seed}}
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def run_chain(a, V=None):
    device = S0.pick_device(a.cpu)
    data = os.path.expanduser(os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data")))
    registry = {}
    for spec in a.shard:
        n, p = spec.split("=", 1)
        registry[n] = p
    mix = parse_mix(a.mix)
    shards = [resolve_shard(n, registry, data) for n, _ in mix]
    rows = alloc_rows([f for _, f in mix], a.B)
    vals = {}
    for spec in a.extra_val:
        sh = resolve_shard(spec, registry, data)
        vals[sh.name] = sh
    tb = torch.from_numpy(np.fromfile(os.path.expanduser(a.token_bytes or os.path.join(data, "token_bytes.i32")),
                                      dtype=np.int32).astype(np.int64)).to(device)
    ck_root = os.path.join(os.path.expanduser(os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints"))), a.run)
    runs_dir = os.path.expanduser(os.environ["SDMLLM_RUNS"])
    os.makedirs(ck_root, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    logf = open(os.path.join(runs_dir, f"{a.run}.log"), "a")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    torch.manual_seed(a.seed)
    model, arm, bc, init_path = load_init(a.init, device, a.init_result, V)
    model.train()
    if getattr(model, "store", None) is not None:
        model.store.collect_stats = False  # the per-read stats (a torch.unique per hop) are not needed while training
    opt = make_opt(model, a)
    lr_at = lr_fn(a)
    tps = a.B * a.T
    total = max(1, a.tokens // tps)
    warm = max(1, int(a.warmup * total))
    mix_info = {"mix": [{"shard": n, "frac": f, "rows_per_batch": r} for (n, f), r in zip(mix, rows)],
                "shards": [s.info() for s in shards]}
    log({"event": "start", "run": a.run, "init": init_path, "arm": arm, "build_cfg": bc, **mix_info,
         "rounds": a.rounds, "steps_per_round": total, "device": str(device), "stamp": S0.stamp()})
    if a.fresh:
        for r in range(0, a.rounds + 1):
            for f in (os.path.join(runs_dir, f"{a.run}.round_{r}.result.json"), os.path.join(ck_root, f"round_{r}", "last.pt")):
                if os.path.exists(f):
                    os.remove(f)
        if os.path.exists(os.path.join(ck_root, "last.pt")):
            os.remove(os.path.join(ck_root, "last.pt"))

    r0 = os.path.join(runs_dir, f"{a.run}.round_0.result.json")
    if not os.path.exists(r0):
        d0 = os.path.join(ck_root, "round_0")
        os.makedirs(d0, exist_ok=True)
        res0 = {"run": a.run, "round": 0, "what": "the init, before any fine-tune step", "init": init_path, "arm": arm,
                "build_cfg": bc, "stamp_start": S0.stamp(), "val": score(model, a, vals, tb, device, d0)}
        res0["stamp_end"] = S0.stamp()
        json.dump(res0, open(r0, "w"), indent=1)

    done_steps = 0
    for rnd in range(1, a.rounds + 1):
        rdir = os.path.join(ck_root, f"round_{rnd}")
        os.makedirs(rdir, exist_ok=True)
        last = os.path.join(rdir, "last.pt")
        res_path = os.path.join(runs_dir, f"{a.run}.round_{rnd}.result.json")
        step = 0
        if os.path.exists(last):
            ck = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            step = ck["step"]
            if step >= total and os.path.exists(res_path):
                log({"event": "round_skip", "round": rnd, "reason": "finished earlier"})
                continue
        peak = a.lr * (a.round_lr_mult ** (rnd - 1))
        stamp0, t_round, t0 = S0.stamp(), time.time(), time.time()
        curve, speeds, share_logged, last_log_step = [], [], False, step
        log({"event": "round_start", "round": rnd, "resume_step": step, "steps": total, "peak_lr": peak})
        while step < total:
            lr = lr_at(step, total, peak, warm)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            x, m, src, _ = mix_batch(shards, rows, a.seed, rnd, step, a.T, a.doc_start_frac)
            x, m = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
                nll, sel = token_nll(model, x, m, a.loss, a.ce_chunk)
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            done_steps += 1
            if step % a.log_every == 0 or step == total or not share_logged:
                tok_src = torch.from_numpy(np.repeat(src, a.T)).to(device)[sel]
                nd = nll.detach().float()
                per = {}
                for si, sh in enumerate(shards):
                    k = tok_src == si
                    per[sh.name] = {"loss": round(float(nd[k].mean()), 4) if bool(k.any()) else None,
                                    "loss_token_share": round(float(k.float().mean()), 4)}
                dt = time.time() - t0
                rec = {"event": "step", "round": rnd, "step": step, "loss": round(float(loss.detach()), 4), "lr": lr,
                       "gnorm": round(float(gn), 3), "per_shard": per}
                if share_logged:
                    rec["tok_per_s"] = round((step - last_log_step) * tps / max(dt, 1e-9), 1)
                    speeds.append(rec["tok_per_s"])
                log(rec)
                curve.append({"step": step, "loss": rec["loss"]})
                share_logged, t0, last_log_step = True, time.time(), step
            if step % a.ckpt_every == 0 or step == total:
                save_ck(last, model, opt, step, total, arm, bc, a, rnd, init_path)
            if a.stop_after_steps and done_steps >= a.stop_after_steps and step < total:
                log({"event": "stopped_early", "round": rnd, "step": step, "why": "--stop-after-steps (a resume test)"})
                save_ck(last, model, opt, step, total, arm, bc, a, rnd, init_path)
                logf.close()
                return None
        res = {"run": a.run, "round": rnd, "init": init_path, "arm": arm, "build_cfg": bc, **mix_info,
               "tokens": total * tps, "steps": total, "B": a.B, "T": a.T, "peak_lr": peak, "warmup_steps": warm,
               "sched": a.sched, "store_lr_mult": a.store_lr_mult, "store_wd": a.store_wd, "loss_path": a.loss,
               "seed": a.seed, "doc_start_frac": a.doc_start_frac, "device": str(device), "curve": curve,
               "median_tok_per_s": float(np.median(speeds)) if speeds else None,
               "seconds_train": round(time.time() - t_round, 1), "stamp_start": stamp0}
        res["val"] = score(model, a, vals, tb, device, rdir)
        res["stamp_end"] = S0.stamp()
        json.dump(res, open(res_path, "w"), indent=1)
        shutil.copyfile(last, os.path.join(ck_root, "last.pt.tmp"))
        os.replace(os.path.join(ck_root, "last.pt.tmp"), os.path.join(ck_root, "last.pt"))
        log({"event": "round_end", "round": rnd, "val": {k: round(v["bpb"], 5) for k, v in res["val"].items()}})
    logf.close()
    return print_table(runs_dir, a.run)


def print_table(runs_dir, run):
    rows, names = [], []
    r = 0
    while os.path.exists(os.path.join(runs_dir, f"{run}.round_{r}.result.json")):
        res = json.load(open(os.path.join(runs_dir, f"{run}.round_{r}.result.json")))
        rows.append(res)
        for n in res.get("val", {}):
            if n not in names:
                names.append(n)
        r += 1
    head = f"{'round':<6}{'tokens':>12}{'final loss':>12}" + "".join(f"{n:>14}" for n in names)
    print(f"\n{run}: bits per byte after each round (round 0 = the init)\n{head}")
    for res in rows:
        loss = res["curve"][-1]["loss"] if res.get("curve") else None
        line = f"{res['round']:<6}{res.get('tokens', 0):>12,}{(f'{loss:.4f}' if loss else '-'):>12}"
        line += "".join(f"{res['val'][n]['bpb']:>14.5f}" if n in res.get("val", {}) else f"{'-':>14}" for n in names)
        print(line)
    return rows


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description="SDMONLY chain fine-tune: an SDM-only checkpoint, fine-tuned on a row mix of "
                                            "shards, over one or more rounds, each scored and checkpointed.",
                                epilog=EXAMPLE, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("what and where")
    g.add_argument("--init", help="checkpoint last.pt, or a run folder (its last.pt, else its newest round_N/last.pt)")
    g.add_argument("--init-result", default=None, help="result json of the init run, cross-checked against the weights")
    g.add_argument("--run", help="name of this fine-tune; folders <SDMLLM_CKPT>/<run>/ and <SDMLLM_RUNS>/<run>.*")
    g.add_argument("--mix", default="train:1.0", help="name:frac,... rows of every batch per shard; fractions sum to 1")
    g.add_argument("--shard", action="append", default=[], help="name=path of a .u32 shard (a _mask.u8 beside it = masked)")
    g.add_argument("--extra-val", action="append", default=[], help="shard scored after each round (val = S0 TEST windows)")
    g.add_argument("--token-bytes", default=None, help="token_bytes.i32 (default $SDMLLM_DATA/token_bytes.i32)")
    g = p.add_argument_group("training")
    g.add_argument("--tokens", type=int, default=2_000_000, help="tokens per round")
    g.add_argument("--rounds", type=int, default=1)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--round-lr-mult", type=float, default=1.0, help="peak lr of round r = lr x this^(r-1)")
    g.add_argument("--store-lr-mult", type=float, default=3.0)
    g.add_argument("--store-wd", type=float, default=0.0)
    g.add_argument("--sched", default="cosine", choices=["cosine", "wsd"])
    g.add_argument("--wsd-decay", type=float, default=0.2)
    g.add_argument("--warmup", type=float, default=0.02, help="warmup fraction of each round")
    g.add_argument("--B", type=int, default=32)
    g.add_argument("--T", type=int, default=256)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--doc-start-frac", type=float, default=0.7, help="rows of a shard with _starts.i64 that start a document")
    g.add_argument("--loss", default="plain", choices=["plain", "chunked"])
    g.add_argument("--ce-chunk", type=int, default=2048)
    g = p.add_argument_group("logging, scoring, resume")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--ckpt-every", type=int, default=250)
    g.add_argument("--val-windows", type=int, default=None, help="cap on val TEST windows (unset = all, and the per-window file)")
    g.add_argument("--val-max-windows", type=int, default=1000, help="cap on windows for every other scored shard")
    g.add_argument("--fresh", action="store_true", help="drop this run's earlier round files and start again")
    g.add_argument("--stop-after-steps", type=int, default=0, help="stop after this many steps (to test resume)")
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--table", default=None, metavar="RUN", help="print the per-round table of RUN and exit")
    g.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if not (a.selftest or a.table) and not (a.init and a.run):
        p.error("--init and --run are required (or --table RUN, or --selftest)")
    return a


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name, flush=True)

    Vs = 512
    tmp = tempfile.mkdtemp(prefix="chain_selftest_")
    data, ckd, rund = (os.path.join(tmp, d) for d in ("data", "ck", "runs"))
    for d in (data, ckd, rund):
        os.makedirs(d)
    i = np.arange(60_000)
    (1 + (i * 7) % 97).astype(np.uint32).tofile(os.path.join(data, "web.u32"))            # a learnable pattern
    (1 + (i * 11) % 89).astype(np.uint32).tofile(os.path.join(data, "chat.u32"))
    g_ids = (1 + (i[:20_000] * 5) % 61).astype(np.uint32)
    g_starts = np.arange(0, 20_000, 50, dtype=np.int64)
    g_mask = ((i[:20_000] % 50) >= 20).astype(np.uint8)                                   # replies only
    g_ids[g_starts] = BOS
    g_ids.tofile(os.path.join(data, "guy.u32"))
    g_mask.tofile(os.path.join(data, "guy_mask.u8"))
    g_starts.tofile(os.path.join(data, "guy_starts.i64"))
    (1 + (i[:30_000] * 7) % 97).astype(np.uint32).tofile(os.path.join(data, "val.u32"))
    np.ones(Vs, dtype=np.int32).tofile(os.path.join(data, "token_bytes.i32"))
    os.environ.update(SDMLLM_DATA=data, SDMLLM_CKPT=ckd, SDMLLM_RUNS=rund)

    # 1. the row mix: exact counts every batch, rows are true windows of their shard, masks are the shard's
    check("alloc 0.5/0.25/0.25 of 32 rows = 16/8/8", alloc_rows([0.5, 0.25, 0.25], 32) == [16, 8, 8])
    check("alloc thirds of 32 rows sums to 32 (11/11/10)", alloc_rows([1 / 3, 1 / 3, 1 / 3], 32) == [11, 11, 10])
    sh = [resolve_shard(n, {}, data) for n in ("guy", "chat", "web")]
    rows = alloc_rows([0.5, 0.25, 0.25], 32)
    exact, true_rows, mask_ok = True, True, True
    for step in range(20):
        x, m, src, offs = mix_batch(sh, rows, 0, 1, step, 32, 0.7)
        exact &= [int((src == k).sum()) for k in range(3)] == [16, 8, 8]
        for r in range(len(src)):
            s = sh[src[r]]
            true_rows &= bool((x[r] == np.asarray(s.ids[offs[r]:offs[r] + 33])).all())
            want = np.asarray(s.mask[offs[r]:offs[r] + 33]).astype(bool) if s.masked else np.ones(33, bool)
            mask_ok &= bool((m[r] == want).all())
    check("every batch of 20 holds exactly 16 guy, 8 chat, 8 web rows", exact)
    check("every row is the shard's own window at its recorded offset", true_rows)
    check("masked rows carry the shard's mask, plain rows train on every target", mask_ok)
    docs = [int(o) in set(g_starts.tolist()) for st in range(50) for o in mix_batch(sh, rows, 0, 1, st, 32, 0.7)[3][:16]]
    check(f"about 70% of guy rows start a document ({sum(docs) / len(docs):.2f})", 0.6 < sum(docs) / len(docs) < 0.8)
    # 2. seeded batches
    b1, b2 = mix_batch(sh, rows, 3, 2, 17, 32, 0.7), mix_batch(sh, rows, 3, 2, 17, 32, 0.7)
    b3 = mix_batch(sh, rows, 3, 2, 18, 32, 0.7)
    check("same (seed, round, step) gives the same batch", all((p == q).all() for p, q in zip(b1, b2)))
    check("the next step gives a different batch", not (b1[0] == b3[0]).all())
    # 3. a tiny base checkpoint in the S0 format (no build_cfg: shapes must be read off the weights)
    torch.manual_seed(0)
    bcfg = {"d": 32, "d_a": 16, "n_sub": 12, "k": 5, "hops": 3, "heads": 2, "n_back": 4, "decays": [0.8, 0.97],
            "softness": 0.25, "qgrad": 0.0, "readout_f": 24, "seed": 0}
    base = M.build_sdmonly("sdmonly", Vs, bcfg)
    for b in base.store.banks:
        torch.nn.init.normal_(b.values, std=0.05)
    s0cfg = {k: bcfg[k] for k in ("d", "k", "hops", "softness", "n_back", "decays", "qgrad", "seed")}
    os.makedirs(os.path.join(ckd, "base"))
    torch.save({"model": base.state_dict(), "opt": None, "step": 0, "cfg": s0cfg, "arm": "sdmonly"},
               os.path.join(ckd, "base", "last.pt"))
    rebuilt, _, rbc, _ = load_init(os.path.join(ckd, "base"), torch.device("cpu"), V=Vs)
    idx = torch.randint(1, Vs, (2, 30))
    base.eval()
    rebuilt.eval()
    with torch.no_grad():
        check("S0-format checkpoint: shapes give the build cfg, identical logits", torch.equal(base(idx), rebuilt(idx)))
    check("inferred cfg: n_sub 12, d_a 16, heads 2, readout 24, 3 stores",
          (rbc["n_sub"], rbc["d_a"], rbc["heads"], rbc["readout_f"], rbc["share_store"]) == (12, 16, 2, 24, False))
    for arm, extra in (("sdmonly_dense", {"dense_f": 40}), ("sdmonly", {"share_store": True})):
        mm = M.build_sdmonly(arm, Vs, {**bcfg, **extra})
        ckx = {"model": mm.state_dict(), "cfg": s0cfg, "arm": arm}
        mm2 = M.build_sdmonly(arm, Vs, infer_build_cfg(ckx))
        mm2.load_state_dict(ckx["model"], strict=True)
        mm.eval()
        mm2.eval()
        with torch.no_grad():
            check(f"rebuild from shapes: {arm} {extra}", torch.equal(mm(idx), mm2(idx)))
    # 4. a two-round chain on the CPU: loss falls, rounds scored, checkpoint round trip
    common = ["--init", os.path.join(ckd, "base"), "--mix", "guy:0.5,chat:0.25,web:0.25", "--tokens", str(40 * 8 * 32),
              "--B", "8", "--T", "32", "--lr", "3e-3", "--rounds", "2", "--log-every", "10", "--ckpt-every", "7",
              "--extra-val", "guy", "--extra-val", "val", "--extra-val", "chat", "--val-max-windows", "40",
              "--val-windows", "40", "--cpu"]
    S0V, threads = S0.V, torch.get_num_threads()
    S0.V = Vs
    torch.set_num_threads(1)  # multi-threaded CPU backward is not bit-deterministic; one thread is
    try:
        rows_t = run_chain(parse(common + ["--run", "straight"]), V=Vs)
        log = [json.loads(ln) for ln in open(os.path.join(rund, "straight.log"))]
        losses = [r["loss"] for r in log if r.get("event") == "step" and r["round"] == 1]
        check(f"round 1 loss falls ({losses[0]:.3f} -> {losses[-1]:.3f})", losses[-1] < losses[0] - 0.3)
        check("rounds 0, 1, 2 scored on guy, val, chat",
              len(rows_t) == 3 and all(set(r["val"]) == {"guy", "val", "chat"} for r in rows_t))
        check("the guy's own score falls over the chain", rows_t[2]["val"]["guy"]["bpb"] < rows_t[0]["val"]["guy"]["bpb"])
        ck = torch.load(os.path.join(ckd, "straight", "last.pt"), weights_only=False)
        check("every checkpoint written carries the full build cfg", ck["build_cfg"] == {**rbc, **ck["build_cfg"]}
              and set(rbc) <= set(ck["build_cfg"]))
        m2, _, _, _ = load_init(os.path.join(ckd, "straight"), torch.device("cpu"), V=Vs)
        m1 = M.build_sdmonly("sdmonly", Vs, rbc)
        m1.load_state_dict(ck["model"])
        m1.eval()
        m2.eval()
        with torch.no_grad():
            check("round trip: rebuilt from the checkpoint's own cfg, identical logits", torch.equal(m1(idx), m2(idx)))
        # 5. resume: stop part way through round 1, start again, end on the same weights as the straight run
        run_chain(parse(common + ["--run", "resumed", "--stop-after-steps", "23"]), V=Vs)
        run_chain(parse(common + ["--run", "resumed"]), V=Vs)
        ckr = torch.load(os.path.join(ckd, "resumed", "last.pt"), weights_only=False)
        same = all(torch.equal(ck["model"][k], ckr["model"][k]) for k in ck["model"])
        check("a resumed chain ends on the same weights as a straight one", same)
        run_chain(parse(common + ["--run", "seed1", "--seed", "1"]), V=Vs)
        cks = torch.load(os.path.join(ckd, "seed1", "last.pt"), weights_only=False)
        check("control: the same chain with seed 1 ends on different weights",
              not all(torch.equal(ck["model"][k], cks["model"][k]) for k in ck["model"]))
        # 6. chain the next link: init from the finished run folder
        rows_n = run_chain(parse(["--init", os.path.join(ckd, "straight"), "--run", "next", "--mix", "guy:1.0", "--tokens",
                                  str(5 * 8 * 32), "--B", "8", "--T", "32", "--extra-val", "guy", "--cpu",
                                  "--val-max-windows", "40"]), V=Vs)
        check("the next link starts where the last one ended (its round 0 = the last round)",
              abs(rows_n[0]["val"]["guy"]["bpb"] - masked_bpb(m2, resolve_shard("guy", {}, data), 32,
                                                             torch.ones(Vs, dtype=torch.int64), torch.device("cpu"), 40)["bpb"]) < 1e-9)
    finally:
        S0.V = S0V
        torch.set_num_threads(threads)
    if not os.environ.get("CHAIN_SELFTEST_KEEP"):
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("kept", tmp)
    print(f"{sum(ok)} of {len(ok)} checks pass")
    return all(ok)


def main():
    a = parse()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.table:
        print_table(os.path.expanduser(os.environ["SDMLLM_RUNS"]), a.table)
        return
    if run_chain(a) is None:
        print("NEXT -> stopped early on purpose: run the same command again to resume from the last checkpoint")
        return
    print(f"NEXT -> chain the next link: --init {os.path.join(os.environ.get('SDMLLM_CKPT', 'checkpoints'), a.run)}"
          f" (or print this table again: --table {a.run})")


if __name__ == "__main__":
    main()
