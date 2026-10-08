"""SDMONLY fine-tune: an SdmOnlyLM or onesdm checkpoint (SDM BASE, or a fine-tune of it) -> SDM CHAT or THE WEIRD LITTLE GUY.

<claudes_code_comments>
** Function List **
data_dir() - $SDMLLM_DATA (read at call time), else data/ beside this file
shard(spec, need_mask) - a shard name, path or name=path -> the chain tool's Shard; refuses a GUY shard with no mask
shard_path(spec) - a shard name or path -> its .u32 path (used for val, read as a plain memmap)
fingerprint(arr) - sha256 of the first and last million tokens (16 hex digits), recorded per shard
find_ck(path) - a checkpoint file, a run folder's last.pt, or its newest round_N/last.pt (the chain tool's rule)
ck_body_opt(opt_sd) - which body optimiser wrote a checkpoint's optimiser state: muon, adamw or None
load_checkpoint(path, device) - rebuild the model from the checkpoint's own cfg (cfg["onesdm"] for the onesdm arms),
    strict load -> (model, ck, path)
build_opt(model, a, body_opt) - the trainer's optimiser (build_optimizer, or for an onesdm model the trainer's onesdm split)
load_opt_state(opt, sd) - load a checkpoint's optimiser state, then put back this run's lr_mult and weight decay
autocast(device) - bf16 autocast on cuda and mps, off on cpu
nll_selected(hidden_fn, E, x, m, loss, chunk) - cross-entropy of the masked targets only (plain or checkpointed chunks)
test_bpb / stream_bpb / masked_bpb - TEST of val.u32, a plain held-out stream, a masked held-out shard (one rule)
evaluate(model, ev, a, tb, device, quick) - every held-out set of this task
short(ev) - {set: bpb rounded} for the log
train_step(model, hidden_fn, opt, x, m, a) - one step: accum micro-batches, mean over all loss tokens of the batch
task_mix(a) - the row mix of the task: chat + web, or guy + chat + web
run(a) - load, score before, fine-tune, score at intervals, checkpoint, score after, write the result json
parse(argv) - flags
selftest() - CPU, about a minute: the real trainer writes the base, then chat, guy-on-chat, mask, round trip, resume,
    and an onesdm base (reset at BOS) taking one chat and one guy step
main() - CLI

** Technical Review **
- What it loads: any checkpoint of the SDM-only family written by track4_sdmonly_train.py, by the DDP trainer
  (lane DDPRECIPE layout: the same keys plus "ddp", and cfg["data"]), or by this file. Those checkpoints carry the
  FULL build config in ck["cfg"] (the trainer's build hook adds extra_cfg: untie, hop_mlp, conj, share_store, d_a,
  n_sub, the mixer...), so the model is M.build_sdmonly(arm, V, ck["cfg"]) and load_state_dict(strict=True). The
  chain tool's infer_build_cfg reads only a few cfg keys plus weight shapes and does NOT know untie, hop_mlp or conj,
  so it cannot rebuild the base run (d 1536, hop MLP 6,144, 16 hops, untie, conj 2): this file exists for that.
  An older checkpoint whose cfg lacks extra_cfg falls back to the chain tool's shape inference, and says so.
  The file is memory-mapped (torch.load mmap=True): a 9.9 GB base checkpoint is not read into RAM twice.
- The optimiser is the trainer's: track4_sdmonly_optim.build_optimizer, with --body-opt auto taking what the
  checkpoint's optimiser state shows (a Composite with a "muon" part = the base run's Muon on the 2-D body matrices;
  AdamW on the embedding/head, the norms and the store, store lr x3, no store weight decay, AdamW (0.9, 0.95), clip
  1.0). By default (--opt-state load) the checkpoint's optimiser state is loaded (Adam moments and step counts,
  Muon momentum), and then this run's lr_mult and weight decay are put back, because torch's load_state_dict
  replaces the group dicts with the saved ones. A state that does not fit (row-wise Adam, other parts) is not
  loaded and the run says so and starts the optimiser fresh.
- Learning rate: --lr is the AdamW peak (default 1e-3 for chat, 5e-4 for guy: a third and a sixth of the base's
  3e-3); --muon-lr defaults to lr x (0.02 / 3e-3), the base recipe's own ratio, so both optimisers move by the same
  fraction of their pretraining rate. Schedule: the trainer's wsd_lr - linear warmup over --warmup of the steps,
  constant, then linear to zero over the last --cooldown fraction.
- The row mix (the chain tool's Shard / alloc_rows / mix_batch, imported, not copied): every batch holds EXACTLY
  floor(frac x B) rows of each shard (largest remainder for the rest), drawn from default_rng([seed, 1, step, i]),
  so a resumed run sees the batches a straight run would have seen. A shard with <stem>_mask.u8 beside it is masked
  (the guy: his replies and the BOS that ends a document); with <stem>_starts.i64, --doc-start-frac of its rows begin
  at a document start. Chat and web shards train on every target, as the base did. The GUY shards are REFUSED
  without their mask file (a bare .u32 would silently train on the visitor's turns).
  --task chat: chat:(1 - web_frac) + web:web_frac (default chat_train 0.5 + train_big 0.5; --chat chat_mix
  --web-frac 0 is the old pre-mixed file). --task guy: guy:(1 - chat_frac - web_frac) + chat + web (default
  0.5 / 0.25 / 0.25). The shares and each shard's share of LOSS tokens are logged.
- Loss: mean cross-entropy over all loss tokens of the whole batch (sum over micro-batches / total count, so --accum
  changes memory only, never the maths; batch-norm statistics are per micro-batch, as in the trainer). bf16 autocast
  on cuda/mps, fp32 cross-entropy. --loss chunked (default) forms the tied head's logits in checkpointed chunks over
  the selected positions only; plain forms them at once.
- Bits per byte, ONE rule everywhere, the trainer's (S0.eval_bpb): nats over scored targets / (ln 2 x their UTF-8
  bytes, token_bytes.i32); a target that is BOS (id 0, 0 bytes) is never scored. TEST = S0.val_windows(val, T,
  "test") + S0.eval_bpb, i.e. the trainer's own function on the trainer's own windows: 3,892 windows / 995,323 tokens
  at T 256, 488 windows / 998,390 tokens at T 2,048 (the base run's T, the default here). chat_test = the trainer's
  extra-val rule (S0.val_windows(arr, T, "all", 4000) + S0.eval_bpb). --eval-T scores at another window than the
  training T. The plan's gate numbers (1.40285, 1.06386) come from older width-768 runs: compare at their window. The guy's test = non-overlapping windows from
  the start, scored on his mask AND not BOS (the chain tool also scores the BOS; this file does not, so the guy's
  number has the same rule as TEST). Scored before step 1, every --eval-every steps (first --quick-windows windows),
  and after the last step (all windows); the after TEST also writes test_per_window.npz for the paired tool.
- Checkpoints: <SDMLLM_CKPT>/<run>/last.pt, the trainer's keys (model, opt, step, cfg, arm) so every tool that reads
  a trainer checkpoint reads these (track4_sdmonly_generate.py, score_ablations, this file with --task guy), plus
  "finetune" (task, mix, B, T, accum, seed, lr, steps, shard fingerprints, lineage = every link back to the base).
  cfg["data"] becomes this fine-tune's data record, so the pretraining trainer's resume check refuses to resume a
  fine-tune as if it were pretraining. "step" counts fine-tune steps. A re-run resumes from last.pt when the
  fine-tune record matches, and refuses (naming the field) when it does not; --fresh starts again.
- Single GPU. The base needed one 2,048 window per micro-batch per 32 GB card, so on such a card use --B 64 --accum 64
  (the base's 131,072 tokens a step). --compile compiles the body (model.hidden) for the fixed micro-batch shape; the
  selftest does not exercise it.
- THE ONESDM ARMS (lane TEXTWIRE): a checkpoint of onesdm, onesdm_allsdm, onesdm_allsdm_big, plain or
  yardstick_transformer written by track4_sdmonly_train.py carries its whole build in cfg["onesdm"]; it is rebuilt by
  track4_sdmonly_train.build_text_model and loaded strictly. Its optimiser is the trainer's onesdm split
  (build_onesdm_optimizer: Muon on the body matrices, the memory tables in the AdamW store group, --store-lr-mult /
  --store-wd), so the init's optimiser state loads. cfg["onesdm"] (with reset_at_bos) rides into every fine-tuned
  checkpoint, so chat and the guy inherit the base's build. --task chat and --task guy run on it unchanged.
- THE VOCABULARY IS THE DATA'S: run() refuses a checkpoint whose embedding rows differ from the entries of
  token_bytes.i32, then sets that V in S0 and the paired scorer (the trainer's set_vocab). Before lane TEXTWIRE the
  check compared with S0's hard-coded 129,280.
- NOT here: generation and the looping-replies gate (track4_sdmonly_generate.py reads these checkpoints), RL, DDP.
Docs: PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md (steps 3 and 4) · track4_sdmonly_chain_finetune.py (the row mix)
</claudes_code_comments>
"""
import argparse
import copy
import hashlib
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
import track4_sdmonly_optim as O  # noqa: E402
from track4_sdmonly_train import wsd_lr  # noqa: E402
import track4_sdmonly_train as TR  # noqa: E402  (lane TEXTWIRE: the onesdm arms' build and optimiser split)
import track4_onesdm_models as OS  # noqa: E402
from track4_sdmllm_models import QwenLM  # noqa: E402
import track4_sdmonly_chain_finetune as CH  # noqa: E402  (Shard, alloc_rows, mix_batch, infer_build_cfg)

BOS = 0
ARMS = ("sdmonly", "sdmonly_dense", "sdmonly_none")
BASE_LR, BASE_MUON_LR = 3e-3, 0.02  # the trainer's defaults; the base run's line sets neither, so it used these
EXAMPLE = """examples:
  # SDM CHAT from SDM BASE on one 32 GB GPU: half chat, half web rows in every batch, 131,072 tokens a step
  python3 track4_sdmonly_finetune.py --task chat --init checkpoints/true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B \\
      --run sdm_chat_1 --tokens 200000000 --B 64 --accum 64
  # THE WEIRD LITTLE GUY on SDM CHAT: his corpus (with its _mask.u8 and _starts.i64), chat and web rows in every batch
  python3 track4_sdmonly_finetune.py --task guy --init checkpoints/sdm_chat_1 --run guy_on_chat_1 \\
      --guy private/wlg_v2/guy2_train.u32 --guy-test private/wlg_v2/guy2_test.u32 --tokens 20000000 --B 64 --accum 64
  # the control: the same GUY straight off SDM BASE
  python3 track4_sdmonly_finetune.py --task guy --init checkpoints/true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B \\
      --run guy_on_base_1 --guy private/wlg_v2/guy2_train.u32 --guy-test private/wlg_v2/guy2_test.u32 \\
      --tokens 20000000 --B 64 --accum 64
  # the gate (CPU, about a minute)
  python3 track4_sdmonly_finetune.py --selftest
data: shard names resolve to $SDMLLM_DATA/<name>.u32 (val, train_big, chat_train, chat_test, token_bytes.i32);
      anything with a "/" or ending in .u32 is a path. Folders: $SDMLLM_CKPT/<run>/ and $SDMLLM_RUNS/<run>.*"""


# ---------------------------------------------------------------- data
def data_dir():
    return os.path.expanduser(os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data")))


def shard(spec, need_mask=False):
    if "=" in spec:
        name, path = spec.split("=", 1)
    elif "/" in spec or spec.endswith(".u32"):
        path = spec
        name = os.path.basename(spec)[:-4] if spec.endswith(".u32") else os.path.basename(spec)
    else:
        name, path = spec, os.path.join(data_dir(), f"{spec}.u32")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"shard {name}: {path} not found")
    sh = CH.Shard(name, path)
    if need_mask and not sh.masked:
        stem = path[:-4] if path.endswith(".u32") else path
        raise SystemExit(f"shard {name}: {stem}_mask.u8 is missing. The GUY shards need their loss mask (his replies "
                         f"only); without it every token, the visitor's included, would be trained on. Copy the mask "
                         f"(and _starts.i64) from the corpus folder beside the .u32.")
    return sh


def fingerprint(arr):
    m = 1 << 20
    h = hashlib.sha256(np.ascontiguousarray(arr[:m]).tobytes())
    h.update(np.ascontiguousarray(arr[max(0, len(arr) - m):]).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- checkpoints
def find_ck(path):
    return CH.find_ck(path)


def ck_body_opt(opt_sd):
    if not opt_sd:
        return None
    if "sdmonly_composite" in opt_sd:
        if "rowadam" in opt_sd["parts"]:
            return "rowadam"
        return "muon" if "muon" in opt_sd["parts"] else "adamw"
    return "adamw"


def load_checkpoint(path, device):
    p = find_ck(path)
    try:
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError:  # a file not in the zip format cannot be memory-mapped
        ck = torch.load(p, map_location="cpu", weights_only=False)
    arm = ck.get("arm")
    if arm not in ARMS and arm not in TR.ONESDM_ARMS:
        raise SystemExit(f"{p}: arm {arm!r} is not an SDM-only arm {ARMS} nor an onesdm arm {TR.ONESDM_ARMS}")
    sd = ck["model"]
    V = int(sd["emb.weight"].shape[0])  # run() checks it against the data's token_bytes.i32
    cfg = dict(ck["cfg"])
    if arm in TR.ONESDM_ARMS:  # the run-time SDM arms: the checkpoint's cfg["onesdm"] is the whole build
        if "onesdm" not in cfg:
            raise SystemExit(f"{p}: an {arm} checkpoint without cfg['onesdm'] (not written by track4_sdmonly_train.py)")
        model = TR.build_text_model(arm, V, cfg)
        model.load_state_dict(sd, strict=True)
        ck["cfg"], ck["_rebuilt_from"] = cfg, "cfg"
        return model.to(device), ck, p
    model = M.build_sdmonly(arm, V, cfg)
    try:
        model.load_state_dict(sd, strict=True)
        how = "cfg"
    except RuntimeError as e:  # an old checkpoint whose cfg lacks the extra build keys: read them off the shapes
        bc = CH.infer_build_cfg(ck)
        model = M.build_sdmonly(arm, V, bc)
        model.load_state_dict(sd, strict=True)
        cfg = {**cfg, **bc}
        how = f"shapes (cfg alone failed: {str(e).splitlines()[0][:120]})"
    ck["cfg"] = cfg
    ck["_rebuilt_from"] = how
    return model.to(device), ck, p


def build_opt(model, a, body_opt):
    if isinstance(model, (OS.OneSdmLM, QwenLM)):  # the onesdm arms: the trainer's own split (TR.onesdm_group)
        return TR.build_onesdm_optimizer(model, a.lr, a.store_lr_mult, a.store_wd, body_opt, a.muon_lr, a.muon_wd,
                                         betas=(0.9, 0.95), eps=1e-8)
    return O.build_optimizer(model, a.lr, a.store_lr_mult, a.store_wd, body_opt, "dense", a.muon_lr, a.muon_wd,
                             betas=(0.9, 0.95), eps=1e-8)


def load_opt_state(opt, sd):
    hyper = [(g.get("lr_mult", 1.0), g.get("weight_decay", 0.0)) for g in opt.param_groups]
    opt.load_state_dict(sd)
    for g, (lm, wd) in zip(opt.param_groups, hyper):
        g["lr_mult"], g["weight_decay"] = lm, wd


# ---------------------------------------------------------------- loss and scores
def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda"))


def nll_selected(hidden_fn, E, x, m, loss="chunked", chunk=2048):
    """Per-token cross-entropy of the targets x[:, 1:] whose mask m[:, 1:] is set; the others are never formed."""
    h = hidden_fn(x[:, :-1])
    sel = m[:, 1:].reshape(-1)
    hs = h.reshape(-1, h.shape[-1])[sel]
    t = x[:, 1:].reshape(-1)[sel]
    if loss == "plain":
        return F.cross_entropy((hs @ E.t()).float(), t, reduction="none"), sel
    from torch.utils.checkpoint import checkpoint

    def part(hc, tc):
        return F.cross_entropy((hc @ E.t()).float(), tc, reduction="none")
    if hs.shape[0] == 0:
        return hs.sum(-1), sel
    return torch.cat([checkpoint(part, hs[i:i + chunk], t[i:i + chunk], use_reentrant=False)
                      for i in range(0, hs.shape[0], chunk)]), sel


def test_bpb(model, val, T, tb, device, max_windows=None):
    starts = S0.val_windows(val, T, "test", max_windows)
    r = S0.eval_bpb(model, val, starts, T, tb, device)
    return {**r, "windows": int(len(starts)), "T": T, "rule": "S0.eval_bpb on S0.val_windows(val, T, 'test')"}


def stream_bpb(model, arr, T, tb, device, max_windows):
    starts = S0.val_windows(arr, T, "all", max_windows)
    r = S0.eval_bpb(model, arr, starts, T, tb, device)
    return {**r, "windows": int(len(starts)), "T": T, "rule": "S0.eval_bpb on S0.val_windows(arr, T, 'all')"}


@torch.no_grad()
def masked_bpb(model, sh, T, tb, device, max_windows=None):
    model.eval()
    starts = np.arange(0, len(sh.ids) - (T + 1), T + 1)
    if max_windows:
        starts = starts[:max_windows]
    cap = int(os.environ.get("SDM_EVAL_TOKENS", "8192"))
    bs = 32 if T * 32 <= 2 * cap else max(1, cap // T)  # S0.eval_bpb's batch rule
    nats, nbytes, ntok = 0.0, 0, 0
    for i in range(0, len(starts), bs):
        s = starts[i:i + bs]
        x = torch.from_numpy(np.stack([sh.ids[a:a + T + 1] for a in s]).astype(np.int64)).to(device)
        mm = torch.from_numpy(np.stack([sh.mask[a:a + T + 1] for a in s]).astype(np.bool_)).to(device)
        mm[:, 1:] &= x[:, 1:] != BOS  # the trainer's rule: a BOS target is never scored
        with autocast(device):
            nll, sel = nll_selected(model.hidden, model.emb.weight, x, mm, "plain")
        t = x[:, 1:].reshape(-1)[sel]
        nats += float(nll.float().sum())
        nbytes += int(tb[t].sum())
        ntok += int(sel.sum())
    model.train()
    return {"bpb": nats / (math.log(2) * max(1, nbytes)), "nats_per_token": nats / max(1, ntok), "tokens": ntok,
            "bytes": nbytes, "windows": int(len(starts)), "T": T, "rule": "masked: the shard's mask AND target != BOS"}


def evaluate(model, ev, a, tb, device, quick=False):
    out = {}
    q = a.quick_windows if quick else None
    T = a.eval_T or a.T
    out["test"] = test_bpb(model, ev["test"], T, tb, device, q or a.val_windows)
    out["chat_test"] = stream_bpb(model, ev["chat_test"].ids, T, tb, device, q or a.chat_test_windows)
    if "guy_test" in ev:
        out["guy_test"] = masked_bpb(model, ev["guy_test"], T, tb, device, q or a.guy_test_windows)
    model.train()
    return out


def short(ev):
    return {k: round(v["bpb"], 5) for k, v in ev.items()}


# ---------------------------------------------------------------- training
def train_step(model, hidden_fn, opt, x, m, a):
    n_all = int(m[:, 1:].sum())
    opt.zero_grad(set_to_none=True)
    nlls, sels, tot = [], [], 0.0
    for xs, ms in zip(x.chunk(a.accum), m.chunk(a.accum)):
        with autocast(x.device):
            nll, sel = nll_selected(hidden_fn, model.emb.weight, xs, ms, a.loss, a.ce_chunk)
        (nll.sum() / max(1, n_all)).backward()
        tot += float(nll.detach().float().sum())
        nlls.append(nll.detach().float())
        sels.append(sel)
    gn = O.clip_grad_norm_(list(model.parameters()), a.clip, opt)
    opt.step()
    return tot / max(1, n_all), float(gn), torch.cat(nlls), torch.cat(sels)


def task_mix(a):
    if a.task == "chat":
        mix = [("chat", a.chat, 1.0 - a.web_frac, False), ("web", a.web, a.web_frac, False)]
    else:
        mix = [("guy", a.guy, 1.0 - a.chat_frac - a.web_frac, True), ("chat", a.chat, a.chat_frac, False),
               ("web", a.web, a.web_frac, False)]
    if any(f < 0 for _, _, f, _ in mix) or abs(sum(f for _, _, f, _ in mix) - 1.0) > 1e-9:
        raise SystemExit(f"row shares must be non-negative and sum to 1: {[(r, f) for r, _, f, _ in mix]}")
    return [x for x in mix if x[2] > 0]


def run(a):
    device = S0.pick_device(a.cpu)
    ck_dir = os.path.join(os.path.expanduser(os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints"))), a.run)
    runs_dir = os.path.expanduser(os.environ["SDMLLM_RUNS"])
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    last = os.path.join(ck_dir, "last.pt")
    before_path = os.path.join(runs_dir, f"{a.run}.before.json")
    if a.fresh:
        for f in (last, before_path, os.path.join(runs_dir, f"{a.run}.result.json")):
            if os.path.exists(f):
                os.remove(f)
    logf = open(os.path.join(runs_dir, f"{a.run}.log"), "a")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    # the init: the base (or an earlier fine-tune); its cfg fixes the model, its T is the default window
    torch.manual_seed(a.seed)
    resume = os.path.exists(last)
    model, ck, init_path = load_checkpoint(last if resume else a.init, device)
    if not resume:
        init_ck_step = ck.get("step")
    a.T = a.T or int(ck["cfg"]["T"])
    mix = task_mix(a)
    shards = [shard(spec, need) for _, spec, _, need in mix]
    rows = CH.alloc_rows([f for _, _, f, _ in mix], a.B)
    if a.B % a.accum:
        raise SystemExit(f"--accum {a.accum} must divide --B {a.B}")
    ev = {"test": np.memmap(shard_path(a.val), dtype=np.uint32, mode="r"), "chat_test": shard(a.chat_test)}
    if a.task == "guy":
        ev["guy_test"] = shard(a.guy_test, need_mask=True)
    tb = torch.from_numpy(np.fromfile(os.path.expanduser(a.token_bytes or os.path.join(data_dir(), "token_bytes.i32")),
                                      dtype=np.int32).astype(np.int64)).to(device)
    # the vocabulary is the data's (one token_bytes entry per id); the model must have exactly that many rows
    if int(model.emb.weight.shape[0]) != len(tb):
        raise SystemExit(f"{init_path}: vocabulary {int(model.emb.weight.shape[0])}, but token_bytes.i32 has {len(tb)} "
                         f"entries: this checkpoint was trained on another tokenizer's shards")
    TR.set_vocab(len(tb))
    tps = a.B * a.T
    total = max(1, a.tokens // tps)
    warm = max(1, int(a.warmup * total))
    lr_at = wsd_lr(a.cooldown)
    record = {"task": a.task, "mix": [{"role": r, "shard": sh.name, "path": os.path.abspath(sh.path), "frac": f,
                                       "rows_per_batch": n, "masked": sh.masked, "tokens": int(len(sh.ids)),
                                       "fingerprint": fingerprint(sh.ids)}
                                      for (r, _, f, _), sh, n in zip(mix, shards, rows)],
              "B": a.B, "T": a.T, "accum": a.accum, "seed": a.seed, "tokens": a.tokens, "steps_total": total,
              "warmup_steps": warm, "cooldown_frac": a.cooldown, "lr": a.lr, "muon_lr": a.muon_lr, "muon_wd": a.muon_wd,
              "store_lr_mult": a.store_lr_mult, "store_wd": a.store_wd, "clip": a.clip, "loss": a.loss,
              "doc_start_frac": a.doc_start_frac, "eval_T": a.eval_T or a.T}
    key = ("task", "mix", "B", "T", "seed", "tokens", "lr", "muon_lr")
    step = 0
    if resume:
        old = ck.get("finetune") or {}
        diff = {k: (old.get(k), record.get(k)) for k in key if old.get(k) != record.get(k)}
        if diff:
            raise SystemExit(f"resume refused: {last} was written with other settings {json.dumps(diff)[:600]}. "
                             f"Match them, or --fresh.")
        body_opt, step = old["body_opt"], int(ck["step"])
        init_path, base_record = old["init"], old
    else:
        detected = ck_body_opt(ck.get("opt"))
        body_opt = a.body_opt if a.body_opt != "auto" else ("muon" if detected == "muon" else "adamw")
        base_record = None
    opt = build_opt(model, a, body_opt)
    opt_note = "fresh"
    if resume:
        load_opt_state(opt, ck["opt"])
        opt_note = "resumed"
    elif a.opt_state == "load" and ck.get("opt"):
        if ck_body_opt(ck["opt"]) == body_opt:
            try:
                load_opt_state(opt, ck["opt"])
                opt_note = "loaded from the init"
            except (ValueError, KeyError, RuntimeError) as e:
                opt = build_opt(model, a, body_opt)
                opt_note = f"fresh (the init's optimiser state did not fit: {str(e)[:160]})"
        else:
            opt_note = f"fresh (the init's optimiser is {ck_body_opt(ck['opt'])}, this run uses {body_opt})"
    if resume:
        lineage = ck["finetune"]["lineage"]
    else:
        prev = (ck.get("finetune") or {}).get("lineage") or [{"what": "base", "path": os.path.abspath(init_path),
                                                              "step": init_ck_step, "arm": ck["arm"]}]
        lineage = prev + [{"what": a.task, "run": a.run, "init": os.path.abspath(init_path), "init_step": init_ck_step}]
    record.update(body_opt=body_opt, init=os.path.abspath(init_path), lineage=lineage)
    base_cfg = {k: v for k, v in ck["cfg"].items() if k != "data"}
    data_rec = {"finetune": True, "task": a.task, "B_global": a.B, "T": a.T, "train_n": None,
                "fingerprint": "+".join(m["fingerprint"] for m in record["mix"]), "shards": record["mix"]}
    cfg_out = {**base_cfg, "train_name": f"finetune:{a.task}", "data": data_rec}
    arm = ck["arm"]
    del ck
    if getattr(model, "store", None) is not None:
        model.store.collect_stats = False
    hidden_fn = torch.compile(model.hidden) if a.compile else model.hidden
    model.train()
    log({"event": "start", "run": a.run, "resume_step": step, "init": os.path.abspath(init_path), "arm": arm,
         "rebuilt_from": "checkpoint cfg" if not resume else "resume", "body_opt": body_opt, "opt_state": opt_note,
         "device": str(device), "params": S0.count_params(model), **{k: record[k] for k in record if k != "lineage"},
         "stamp": S0.stamp()})

    def save(path):
        rec = {**record, "steps_done": step}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg_out, "arm": arm,
                    "finetune": rec}, path + ".tmp")
        os.replace(path + ".tmp", path)

    if os.path.exists(before_path):
        before = json.load(open(before_path))
    else:
        before = {"stamp": S0.stamp(), "val": evaluate(model, ev, a, tb, device)}
        json.dump(before, open(before_path, "w"), indent=1)
    log({"event": "before", **short(before["val"])})
    curve, quick, speeds, t0, last_step, stamp0, t_run = [], [], [], time.time(), step, S0.stamp(), time.time()
    done = 0
    while step < total:
        lr = lr_at(step, total, a.lr, warm)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        xn, mn, src, _ = CH.mix_batch(shards, rows, a.seed, 1, step, a.T, a.doc_start_frac)
        x, m = torch.from_numpy(xn).to(device), torch.from_numpy(mn).to(device)
        loss, gn, nll, sel = train_step(model, hidden_fn, opt, x, m, a)
        step += 1
        done += 1
        if step % a.log_every == 0 or step == total or done == 1:
            tok_src = torch.from_numpy(np.repeat(src, a.T)).to(device)[sel]
            per = {}
            for i, (sh, (role, _, _, _)) in enumerate(zip(shards, mix)):
                k = tok_src == i
                per[role] = {"loss": round(float(nll[k].mean()), 4) if bool(k.any()) else None,
                             "loss_token_share": round(float(k.float().mean()), 4)}
            rec = {"event": "step", "step": step, "loss": round(loss, 4), "lr": lr, "gnorm": round(gn, 3), "per_role": per}
            if done > 1:
                rec["tok_per_s"] = round((step - last_step) * tps / max(time.time() - t0, 1e-9), 1)
                speeds.append(rec["tok_per_s"])
            log(rec)
            curve.append({"step": step, "loss": rec["loss"]})
            t0, last_step = time.time(), step
        if step % a.eval_every == 0 and step < total:
            q = evaluate(model, ev, a, tb, device, quick=True)
            quick.append({"step": step, **short(q)})
            log({"event": "eval_quick", "step": step, "windows": {k: v["windows"] for k, v in q.items()}, **short(q)})
        if step % a.ckpt_every == 0 or step == total:
            save(last)
        if a.stop_after_steps and done >= a.stop_after_steps and step < total:
            save(last)
            log({"event": "stopped_early", "step": step, "why": "--stop-after-steps (a resume test)"})
            logf.close()
            return None
    after = evaluate(model, ev, a, tb, device)
    if a.val_windows is None:
        from track4_sdmllm_paired_window_comparison import per_window
        model.eval()
        eT = a.eval_T or a.T
        pn, pb = per_window(model, ev["test"], S0.val_windows(ev["test"], eT, "test"), eT, tb, device)
        np.savez(os.path.join(ck_dir, "test_per_window.npz"), nats=pn, bytes=pb)
        model.train()
    res = {"run": a.run, "finetune": record, "arm": arm, "cfg": cfg_out, "before": before["val"], "after": after,
           "delta_bpb": {k: round(after[k]["bpb"] - before["val"][k]["bpb"], 6) for k in after},
           "curve": curve, "quick_evals": quick, "median_tok_per_s": float(np.median(speeds)) if speeds else None,
           "seconds_this_session": round(time.time() - t_run, 1), "device": str(device), "opt_state": opt_note,
           "stamp_start": stamp0, "stamp_end": S0.stamp(), "checkpoint": os.path.abspath(last)}
    json.dump(res, open(os.path.join(runs_dir, f"{a.run}.result.json"), "w"), indent=1)
    log({"event": "final", "before": short(before["val"]), "after": short(after), "delta": res["delta_bpb"]})
    logf.close()
    return res


def shard_path(spec):
    if "/" in spec or spec.endswith(".u32"):
        return os.path.expanduser(spec)
    return os.path.join(data_dir(), f"{spec}.u32")


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description="SDMONLY fine-tune: an SdmOnlyLM checkpoint -> SDM CHAT (--task chat) or "
                                            "THE WEIRD LITTLE GUY (--task guy), on one GPU, scored before and after.",
                                epilog=EXAMPLE, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("what")
    g.add_argument("--task", choices=["chat", "guy"])
    g.add_argument("--init", help="checkpoint file, or a run folder (its last.pt): the base, or SDM CHAT for the guy")
    g.add_argument("--run", help="this fine-tune's name: $SDMLLM_CKPT/<run>/last.pt, $SDMLLM_RUNS/<run>.*")
    g = p.add_argument_group("data (a name = $SDMLLM_DATA/<name>.u32; a '/' or .u32 = a path)")
    g.add_argument("--chat", default="chat_train", help="chat rows (default chat_train; chat_mix is pre-mixed 50/50 web)")
    g.add_argument("--chat-test", default="chat_test", help="held-out chat stream")
    g.add_argument("--web", default="train_big", help="web rows, against forgetting")
    g.add_argument("--web-frac", type=float, default=None, help="share of rows that are web (chat 0.5, guy 0.25)")
    g.add_argument("--chat-frac", type=float, default=0.25, help="guy only: share of rows that are chat")
    g.add_argument("--guy", help="guy only: guy2_train.u32 (its _mask.u8 beside it is required, _starts.i64 used)")
    g.add_argument("--guy-test", help="guy only: guy2_test.u32 (its _mask.u8 beside it is required)")
    g.add_argument("--val", default="val", help="TEST is the second half of this shard (the trainer's rule)")
    g.add_argument("--token-bytes", default=None, help="token_bytes.i32 (default $SDMLLM_DATA/token_bytes.i32)")
    g = p.add_argument_group("training")
    g.add_argument("--tokens", type=int, default=None, help="tokens to train (chat 200M, guy 20M)")
    g.add_argument("--B", type=int, default=64, help="windows a step (the base: 64)")
    g.add_argument("--T", type=int, default=0, help="window (0 = the checkpoint's T; the base: 2,048)")
    g.add_argument("--accum", type=int, default=1, help="micro-batches a step (memory only; 32 GB at T 2,048: --accum B)")
    g.add_argument("--lr", type=float, default=None, help="AdamW peak (chat 1e-3, guy 5e-4; the base used 3e-3)")
    g.add_argument("--muon-lr", type=float, default=None, help="Muon peak (default lr x 0.02/3e-3, the base's ratio)")
    g.add_argument("--muon-wd", type=float, default=0.0)
    g.add_argument("--body-opt", default="auto", choices=["auto", "adamw", "muon"],
                   help="auto = what the checkpoint's optimiser state shows (the base: muon)")
    g.add_argument("--opt-state", default="load", choices=["load", "fresh"], help="load the init's optimiser state")
    g.add_argument("--store-lr-mult", type=float, default=3.0)
    g.add_argument("--store-wd", type=float, default=0.0)
    g.add_argument("--warmup", type=float, default=0.02, help="warmup fraction of the steps")
    g.add_argument("--cooldown", type=float, default=0.1, help="linear decay to zero over this last fraction")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--doc-start-frac", type=float, default=0.7, help="rows of a shard with _starts.i64 that start a doc")
    g.add_argument("--loss", default="chunked", choices=["chunked", "plain"])
    g.add_argument("--ce-chunk", type=int, default=2048)
    g.add_argument("--compile", action="store_true", help="torch.compile the body for the micro-batch shape (CUDA)")
    g = p.add_argument_group("scoring, logging, resume")
    g.add_argument("--eval-T", type=int, default=0, help="window of every score (0 = the training T; 256 gives "
                                                          "the 995,323-token TEST)")
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--quick-windows", type=int, default=64, help="windows per set for the in-run evals")
    g.add_argument("--val-windows", type=int, default=None, help="cap on TEST windows (unset = all + the per-window file)")
    g.add_argument("--chat-test-windows", type=int, default=4000, help="the trainer's extra-val cap")
    g.add_argument("--guy-test-windows", type=int, default=None, help="cap on the guy's test windows (unset = all)")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--ckpt-every", type=int, default=500)
    g.add_argument("--stop-after-steps", type=int, default=0, help="stop after this many steps (to test resume)")
    g.add_argument("--fresh", action="store_true", help="drop this run's checkpoint and scores and start again")
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return a
    if not (a.task and a.init and a.run):
        p.error("--task, --init and --run are required (or --selftest)")
    if a.task == "guy" and not (a.guy and a.guy_test):
        p.error("--task guy needs --guy and --guy-test")
    a.web_frac = a.web_frac if a.web_frac is not None else (0.5 if a.task == "chat" else 0.25)
    a.tokens = a.tokens or (200_000_000 if a.task == "chat" else 20_000_000)
    a.lr = a.lr or (1e-3 if a.task == "chat" else 5e-4)
    a.muon_lr = a.muon_lr or a.lr * BASE_MUON_LR / BASE_LR
    return a


# ---------------------------------------------------------------- selftest
def selftest():
    import subprocess
    import tempfile
    t_start = time.time()
    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)

    tmp = tempfile.mkdtemp(prefix="sdmonly_finetune_selftest_")
    data, ckd, rund = (os.path.join(tmp, d) for d in ("data", "ck", "runs"))
    for d in (data, ckd, rund):
        os.makedirs(d)
    V = S0.V
    i = np.arange(80_000)
    web = (1 + (i * 7) % 97).astype(np.uint32)                     # web: a learnable cycle over ids 1..97
    web[::64] = BOS
    web.tofile(os.path.join(data, "train.u32"))
    web.tofile(os.path.join(data, "train_big.u32"))
    web[:20_000].tofile(os.path.join(data, "val.u32"))
    chat = (300 + (i * 11) % 89).astype(np.uint32)                 # chat: another cycle, ids 300..388
    chat[::60] = BOS
    chat.tofile(os.path.join(data, "chat_train.u32"))
    chat[:6_000].tofile(os.path.join(data, "chat_test.u32"))
    gdir = os.path.join(data, "guy")
    os.makedirs(gdir)
    for split, n in (("train", 20_000), ("test", 3_000)):
        ids = (600 + (i[:n] * 5) % 61).astype(np.uint32)            # the guy: ids 600..660, docs of 50
        starts = np.arange(0, n, 50, dtype=np.int64)
        ids[starts] = BOS
        mask = ((i[:n] % 50) >= 20).astype(np.uint8)                # his replies: positions 20..49 of each doc
        mask[starts[1:]] = 1                                         # the BOS that ends a doc is in the loss
        ids.tofile(os.path.join(gdir, f"guy2_{split}.u32"))
        mask.tofile(os.path.join(gdir, f"guy2_{split}_mask.u8"))
        starts.tofile(os.path.join(gdir, f"guy2_{split}_starts.i64"))
    tb = np.ones(V, dtype=np.int32)
    tb[0] = 0                                                         # BOS is 0 bytes, as in the real table
    tb.tofile(os.path.join(data, "token_bytes.i32"))
    env_keys = ("SDMLLM_DATA", "SDMLLM_RUNS", "SDMLLM_CKPT")
    saved_env = {k: os.environ.get(k) for k in env_keys}
    os.environ.update(SDMLLM_DATA=data, SDMLLM_RUNS=rund, SDMLLM_CKPT=ckd)
    threads = torch.get_num_threads()
    try:
        # 1. the BASE, written by the real single-device trainer: the base run's options (Muon, untie, conj 2, hop
        #    MLP, one shared store, wsd) at a tiny width
        cmd = [sys.executable, os.path.join(HERE, "track4_sdmonly_train.py"), "--arm", "sdmonly", "--run", "base",
               "--tokens", str(4 * 16 * 12), "--B", "4", "--T", "16", "--d", "16", "--d-a", "16", "--n-sub", "16",
               "--k", "4", "--hops", "2", "--share-store", "--untie", "--conj", "2", "--hop-mlp", "24",
               "--body-opt", "muon", "--sched", "wsd", "--cpu", "--fresh", "--log-every", "4", "--eval-every", "1000",
               "--ckpt-every", "6", "--val-windows", "300"]
        r = subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True)
        check("the trainer writes a tiny base checkpoint (Muon, untie, conj 2, hop MLP, shared store)", r.returncode == 0,
              r.stderr[-400:] if r.returncode else "")
        base_res = json.load(open(os.path.join(rund, "base.result.json")))
        torch.set_num_threads(1)  # multi-threaded CPU backward is not bit-deterministic; one thread is
        cpu = torch.device("cpu")
        tbt = torch.from_numpy(tb.astype(np.int64))
        val = np.memmap(os.path.join(data, "val.u32"), dtype=np.uint32, mode="r")
        bm, bck, _ = load_checkpoint(os.path.join(ckd, "base"), cpu)
        check("rebuilt from the checkpoint's own cfg, strict load", bck["_rebuilt_from"] == "cfg" and bm.untie
              and bm.conj == 2 and bm.hop_mlp == 24 and len(bm.store.banks) == 1)
        tb_mine = test_bpb(bm, val, 16, tbt, cpu, 300)["bpb"]
        check("TEST bpb by this file equals the trainer's own TEST bpb", abs(tb_mine - base_res["val_test"]["bpb"]) < 1e-6,
              f"{tb_mine:.6f} vs {base_res['val_test']['bpb']:.6f}")
        try:
            CH.load_init(os.path.join(ckd, "base"), cpu, V=V)
            chain_ok = True
        except Exception:
            chain_ok = False
        check("control: the chain tool's loader cannot rebuild this base (it knows no untie / hop MLP / conj)", not chain_ok)
        check("the base's optimiser state is read as Muon", ck_body_opt(bck["opt"]) == "muon")
        # the DDPRECIPE layout: the same keys plus "ddp" and cfg["data"]
        ddp_ck = {k: bck[k] for k in ("model", "opt", "step", "cfg", "arm")}
        ddp_ck["cfg"] = {**bck["cfg"], "data": {"train_name": "train", "train_n": 80000, "fingerprint": "x", "B_global": 4,
                                                "accum_global": 1, "micro_B": 4, "T": 16, "data_n_flag": None}}
        ddp_ck["ddp"] = {"world_size": 2, "B": 2, "accum": 1, "B_global": 4}
        os.makedirs(os.path.join(ckd, "base_ddp"))
        torch.save(ddp_ck, os.path.join(ckd, "base_ddp", "last.pt"))
        dm, _, _ = load_checkpoint(os.path.join(ckd, "base_ddp"), cpu)
        idx = torch.randint(1, 700, (2, 16))
        bm.eval()
        dm.eval()
        with torch.no_grad():
            check("a DDP-trainer checkpoint (\"ddp\" key, cfg data record) loads to identical logits", torch.equal(bm(idx), dm(idx)))
        # the optimiser: Muon on the body, the init's moments loaded, this run's lr_mult put back
        a0 = parse(["--task", "chat", "--init", "x", "--run", "x"])
        check("default rates: chat lr 1e-3, Muon lr 1e-3 x 0.02/3e-3", a0.lr == 1e-3 and abs(a0.muon_lr - 0.02 / 3) < 1e-12)
        a0 = parse(["--task", "chat", "--init", "x", "--run", "x", "--lr", "1e-3", "--muon-lr", "5e-3", "--store-lr-mult", "2"])
        opt = build_opt(bm, a0, "muon")
        load_opt_state(opt, bck["opt"])
        mu = opt.parts["muon"]
        check("Muon momentum loaded from the base", all(torch.equal(b, s) for b, s in zip(mu.bufs, bck["opt"]["parts"]["muon"]["bufs"])))
        check("lr_mult is this run's after the load, not the base's (Muon 5, store 2; the base had 6.67 and 3)",
              abs(mu.param_groups[0]["lr_mult"] - 5.0) < 1e-9 and opt.parts["adamw"].param_groups[-1]["lr_mult"] == 2.0
              and bck["opt"]["parts"]["adamw"]["param_groups"][-1]["lr_mult"] == 3.0)

        # 2. SDM CHAT from the base: few steps, loss falls, chat_test falls, web rows in every batch
        common = ["--B", "8", "--T", "16", "--log-every", "5", "--eval-every", "20", "--ckpt-every", "15",
                  "--quick-windows", "8", "--chat-test-windows", "200", "--val-windows", "300", "--cpu", "--fresh"]
        chat_res = run(parse(["--task", "chat", "--init", os.path.join(ckd, "base"), "--run", "chat", "--tokens",
                              str(8 * 16 * 40), "--lr", "3e-3"] + common))
        steps = [json.loads(ln) for ln in open(os.path.join(rund, "chat.log")) if '"event": "step"' in ln]
        check(f"chat: the training loss falls ({steps[0]['loss']} -> {steps[-1]['loss']})", steps[-1]["loss"] < steps[0]["loss"] - 1.0)
        check("chat: chat_test bpb falls", chat_res["after"]["chat_test"]["bpb"] < chat_res["before"]["chat_test"]["bpb"] - 0.5,
              f"{chat_res['before']['chat_test']['bpb']:.3f} -> {chat_res['after']['chat_test']['bpb']:.3f}")
        check("chat: half the rows are web, and web loss tokens are logged",
              [m["rows_per_batch"] for m in chat_res["finetune"]["mix"]] == [4, 4] and steps[-1]["per_role"]["web"]["loss_token_share"] > 0.4)
        check("chat: before TEST = the base's TEST", abs(chat_res["before"]["test"]["bpb"] - tb_mine) < 1e-9)
        check("chat: the optimiser state came from the base", "loaded" in chat_res["opt_state"], chat_res["opt_state"])

        # 3. save / load round trip: the reloaded checkpoint scores exactly what the run scored
        cm, cck, _ = load_checkpoint(os.path.join(ckd, "chat"), cpu)
        a_c = parse(["--task", "chat", "--init", "x", "--run", "x"] + common[:-2])
        a_c.T = 16
        ev_c = {"test": val, "chat_test": shard("chat_test")}
        e2 = evaluate(cm, ev_c, a_c, tbt, cpu)
        check("round trip: TEST and chat_test of the reloaded checkpoint equal the run's",
              e2["test"]["bpb"] == chat_res["after"]["test"]["bpb"] and e2["chat_test"]["bpb"] == chat_res["after"]["chat_test"]["bpb"])
        a_c.eval_T = 8
        e8 = evaluate(cm, ev_c, a_c, tbt, cpu)
        check("--eval-T 8 scores the trainer's T-8 windows (the same as S0.val_windows at T 8)",
              e8["test"]["T"] == 8 and e8["test"]["windows"] == min(300, len(S0.val_windows(val, 8, "test")))
              and e8["test"]["bpb"] != e2["test"]["bpb"])
        check("the checkpoint keeps the trainer's keys and the full build cfg",
              {"model", "opt", "step", "cfg", "arm"} <= set(cck) and cck["cfg"]["untie"] and cck["cfg"]["hop_mlp"] == 24)
        import track4_sdmonly_generate as G
        gm, _ = G.load_checkpoint(find_ck(os.path.join(ckd, "chat")), cpu)
        cm.eval()
        with torch.no_grad():
            check("the generation tool loads the fine-tuned checkpoint to identical logits", torch.equal(gm(idx), cm(idx)))

        # 4. the mask: a target outside the mask sends no gradient
        gsh = shard(os.path.join(gdir, "guy2_train.u32"), need_mask=True)
        offs = [0, 20, 100, 125]                                # last targets at doc positions 32, 2, 32, 7: in, out, in, out
        xs = np.stack([np.asarray(gsh.ids[o:o + 33]).astype(np.int64) for o in offs])
        ms = np.stack([np.asarray(gsh.mask[o:o + 33]).astype(np.bool_) for o in offs])
        x, mk = torch.from_numpy(xs), torch.from_numpy(ms)
        cm.train()
        h_holder = {}

        def hid(z):
            h = cm.hidden(z)
            h.retain_grad()
            h_holder["h"] = h
            return h
        nll, _ = nll_selected(hid, cm.emb.weight, x, mk, "chunked", 7)
        nll.sum().backward()
        hg = h_holder["h"].grad.abs().sum(-1)
        out_of_mask = ~mk[:, 1:]
        check("mask: zero gradient at every position whose target is outside the mask, non-zero inside",
              bool((hg[out_of_mask] == 0).all()) and bool((hg[~out_of_mask] > 0).all()) and bool(out_of_mask.any()))

        def grads(xx):
            cm.zero_grad(set_to_none=True)
            n, _ = nll_selected(cm.hidden, cm.emb.weight, xx, mk, "plain")
            n.sum().backward()
            return [p.grad.clone() for p in cm.parameters() if p.grad is not None]
        r_out = int(np.nonzero(~ms[:, -1])[0][0]) if (~ms[:, -1]).any() else None
        if r_out is None:
            check("mask: a row with its last target out of the mask exists", False)
        else:
            x2 = x.clone()
            x2[r_out, -1] = 777                               # a target-only token, outside the mask
            g1, g2 = grads(x), grads(x2)
            check("mask: changing a target outside the mask leaves every gradient bit-identical",
                  all(torch.equal(p, q) for p, q in zip(g1, g2)))
        r_in = int(np.nonzero(ms[:, -1])[0][0]) if ms[:, -1].any() else None
        if r_in is not None:
            x3 = x.clone()
            x3[r_in, -1] = 777
            g3 = grads(x3)
            check("control: changing a target inside the mask changes the gradient",
                  not all(torch.equal(p, q) for p, q in zip(g1, g3)))

        # 5. THE WEIRD LITTLE GUY on SDM CHAT
        gcommon = ["--task", "guy", "--guy", os.path.join(gdir, "guy2_train.u32"), "--guy-test",
                   os.path.join(gdir, "guy2_test.u32"), "--tokens", str(8 * 16 * 30), "--lr", "3e-3"] + common
        guy_res = run(parse(gcommon + ["--init", os.path.join(ckd, "chat"), "--run", "guy"]))
        check("guy: a chat checkpoint loads as the guy's init; its before TEST and chat_test = chat's after",
              guy_res["before"]["test"]["bpb"] == chat_res["after"]["test"]["bpb"]
              and guy_res["before"]["chat_test"]["bpb"] == chat_res["after"]["chat_test"]["bpb"])
        check("guy: his test bpb falls", guy_res["after"]["guy_test"]["bpb"] < guy_res["before"]["guy_test"]["bpb"] - 0.5,
              f"{guy_res['before']['guy_test']['bpb']:.3f} -> {guy_res['after']['guy_test']['bpb']:.3f}")
        check("guy: rows 4 guy / 2 chat / 2 web in every batch",
              [m["rows_per_batch"] for m in guy_res["finetune"]["mix"]] == [4, 2, 2])
        lin = torch.load(os.path.join(ckd, "guy", "last.pt"), map_location="cpu", weights_only=False)["finetune"]["lineage"]
        check("guy: lineage base -> chat -> guy", [x["what"] for x in lin] == ["base", "chat", "guy"])
        # 6. resume: stop part way, run again, end on the same weights as the straight run
        run(parse(gcommon + ["--init", os.path.join(ckd, "chat"), "--run", "guy_resumed", "--stop-after-steps", "11"]))
        rres = run(parse([x for x in gcommon if x != "--fresh"] + ["--init", os.path.join(ckd, "chat"), "--run", "guy_resumed"]))
        s1 = torch.load(os.path.join(ckd, "guy", "last.pt"), map_location="cpu", weights_only=False)["model"]
        s2 = torch.load(os.path.join(ckd, "guy_resumed", "last.pt"), map_location="cpu", weights_only=False)["model"]
        check("resume: the resumed run ends on the straight run's weights", all(torch.equal(s1[k], s2[k]) for k in s1)
              and rres["after"]["guy_test"]["bpb"] == guy_res["after"]["guy_test"]["bpb"])
        try:
            run(parse([x for x in gcommon if x != "--fresh"] + ["--init", os.path.join(ckd, "chat"), "--run", "guy_resumed",
                                                                  "--B", "4"]))
            refused = False
        except SystemExit as e:
            refused = "resume refused" in str(e)
        check("resume with another --B is refused", refused)
        # 7. control off the BASE also runs (the plan's control GUY)
        ctrl = run(parse(gcommon + ["--init", os.path.join(ckd, "base"), "--run", "guy_ctrl", "--tokens", str(8 * 16 * 5)]))
        check("control GUY off the base: before TEST = the base's TEST", abs(ctrl["before"]["test"]["bpb"] - tb_mine) < 1e-9)
        # 8. a guy shard without its mask is refused
        bare = os.path.join(tmp, "bare")
        os.makedirs(bare)
        shutil.copyfile(os.path.join(gdir, "guy2_train.u32"), os.path.join(bare, "guy2_train.u32"))
        try:
            run(parse([x if x != os.path.join(gdir, "guy2_train.u32") else os.path.join(bare, "guy2_train.u32") for x in gcommon]
                      + ["--init", os.path.join(ckd, "chat"), "--run", "guy_bare"]))
            refused = False
        except SystemExit as e:
            refused = "_mask.u8 is missing" in str(e)
        check("a GUY shard without its _mask.u8 is refused", refused)
        # 9. lane TEXTWIRE: an onesdm base (run-time SDM, reset at BOS) from the real trainer, then chat and guy steps
        cmd = [sys.executable, os.path.join(HERE, "track4_sdmonly_train.py"), "--arm", "onesdm", "--run", "base_onesdm",
               "--tokens", str(4 * 16 * 6), "--B", "4", "--T", "16", "--d", "16", "--layers", "2", "--mlp-f", "32",
               "--mem-heads", "4", "--mem-n-sub", "4", "--mem-d-a", "8", "--mem-k", "4", "--chunk", "8",
               "--rt-reset-at-bos", "--cpu", "--fresh", "--log-every", "3", "--eval-every", "1000", "--val-windows", "300"]
        r = subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True)

        def attempt(argv):  # a refusal or an error is a FAIL of the check, not the end of the selftest
            try:
                return run(parse(argv)), ""
            except (SystemExit, Exception) as e:
                return None, f"{type(e).__name__}: {str(e)[:300]}"
        oc, err = attempt(["--task", "chat", "--init", os.path.join(ckd, "base_onesdm"), "--run", "chat_onesdm",
                           "--tokens", str(8 * 16)] + common) if r.returncode == 0 else (None, r.stderr[-400:])
        ock = torch.load(os.path.join(ckd, "chat_onesdm", "last.pt"), map_location="cpu", weights_only=False) if oc else {}
        check("onesdm: the tool loads a tiny onesdm checkpoint and runs one chat step",
              oc is not None and oc["arm"] == "onesdm" and oc["finetune"]["steps_total"] == 1 and ock.get("step") == 1
              and math.isfinite(oc["curve"][0]["loss"]) and ock.get("cfg", {}).get("onesdm", {}).get("reset_at_bos")
              and oc["opt_state"] == "loaded from the init", err or oc["opt_state"])
        og, err = attempt(["--task", "guy", "--guy", os.path.join(gdir, "guy2_train.u32"), "--guy-test",
                           os.path.join(gdir, "guy2_test.u32"), "--tokens", str(8 * 16), "--init",
                           os.path.join(ckd, "chat_onesdm"), "--run", "guy_onesdm"] + common) if oc else (None, "no chat")
        check("onesdm: one guy step on the onesdm chat checkpoint (lineage base -> chat -> guy)",
              og is not None and og["arm"] == "onesdm" and math.isfinite(og["curve"][0]["loss"])
              and [x["what"] for x in og["finetune"]["lineage"]] == ["base", "chat", "guy"], err)
    finally:
        torch.set_num_threads(threads)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if os.environ.get("FINETUNE_SELFTEST_KEEP"):
        print("kept", tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(ok)} of {len(ok)} checks pass in {time.time() - t_start:.0f} s")
    print("NEXT -> python3 track4_sdmonly_finetune.py --help")
    return all(ok)


def main():
    a = parse()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    res = run(a)
    if res is None:
        print("NEXT -> stopped early on purpose: run the same command again to resume from last.pt")
        return
    ck = os.path.join(os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints")), a.run)
    if a.task == "chat":
        print(f"NEXT -> chat with it: python3 track4_sdmonly_generate.py --help (checkpoint {ck}/last.pt); "
              f"then the guy on top: --task guy --init {ck}")
    else:
        print(f"NEXT -> probe him: python3 track4_sdmonly_generate.py --help (checkpoint {ck}/last.pt); "
              f"another round: --task guy --init {ck} --run <new name>")


if __name__ == "__main__":
    main()
