"""SDMONLY multi-GPU trainer: one SDM-only model trained across the GPUs of ONE box with DistributedDataParallel.

<claudes_code_comments>
** Function List **
own_parser() - this file's own flags
parse(argv) - this file's flags, then every flag of track4_sdmonly_train.py; refuses a flag this loop cannot run, by name
global_offsets(step, seed, n, world, B, T) - the seeded window offsets of the whole global batch (world x B rows)
rank_batch(step, seed, arr, B, T, rank, world) - this rank's B rows of the global batch
scaled_lr(a, world) - the peak learning rate after the optional --lr-scale rule
LossModule.forward(x, t) - model -> mean next-token cross-entropy (plain, or the chunked tied head)
rank_device(a, rank) - cpu, cuda:(rank mod GPUs), or mps
pick_backend(a, device, world) - nccl when every rank has its own GPU, gloo otherwise
s0_cfg(a) - the S0 trainer's cfg fields
build_all(a, rank, world, device, state) - model, DDP wrapper, optimiser (the single trainer's build_optimizer), tables
sync_table_rows(tables, world) - average only the touched rows of each value table across ranks
dense_table_bytes(model) - bytes of every value table
bn_modules(model) / bn_snapshot(bns) - the batch-norm layers with running statistics, and their state before a step
bn_combine(bns, snaps, rank, world) - set every rank's running statistics to what one device would hold after the step
state_digest(model, opt) - bit-exact integer digest of every parameter, buffer and Muon momentum buffer
check_ranks(model, opt, rank, world, step) - all-gather the digests; rank 0 logs, every rank stops if they differ
train_step(a, net, model, opt, tables, bns, xb, world, rank, device, step, dump) - one step: micro-batches, no_sync, clip
autocast(device) - bf16 autocast on cuda and mps
rank_rates(rank, world, value, device) - one number per rank, gathered
prepare_cooldown_ddp(a, rank, world) - the cooldown fork: rank 0 copies the source checkpoint, every rank gets its schedule
save_ckpt(path, model, opt, step, cfg, a, world, extra) - the single trainer's checkpoint layout plus a "ddp" record
train(a, rank, world, device) - the loop; rank 0 logs, scores, checkpoints and writes the result json
bench(a, rank, world, device) - the recipe's step on random tokens: tokens a second per rank and in total
run_rank(rank, world, a, init_method) - one rank: join the group, then train (or bench) and leave the group
_spawned(rank, world, argv, port) - entry point of a spawned rank
launch(argv) - one command: under torchrun run this rank; otherwise spawn `--world` ranks and wait
selftest(only) - the gate, CPU and gloo, in named sections (see the list in the function)
main() - CLI

** Technical Review **
- The single-device trainer is track4_sdmonly_train.py, which runs the S0 trainer's train() loop. That loop is one
  function, so this file carries its own copy of the step and reuses every part that is a function: S0.batch_for's
  draw rule, S0.eval_bpb, S0.lr_at, S0.val_windows, S0.stamp, the trainer's wsd_lr, cooldown_lr, prepare_cooldown,
  extra_cfg, opt_info, score_ablations and data record, the optimiser module's build_optimizer and clip_grad_norm_,
  and ce_chunked. bf16 autocast, clip 1.0, 2% warmup, checkpoint keys and result keys are the single trainer's.
- THE RECIPE runs here: --arm sdmonly and sdmonly_none, --d, --d-a, --n-sub, --k, --hops, --hop-mlp, --share-store,
  --untie, --conj, --qgrad, --graded, --store-lr-mult, --body-opt muon (--muon-lr, --muon-wd), --accum, --sched
  cosine|wsd (--wsd-decay), --loss chunked, --init-ckpt with --cooldown-tokens (the cooldown fork), --keep-at,
  --ckpt-every, resume from last.pt, --data-n. Every other model-shape flag passes to the model unchanged.
  REFUSED BY NAME: --arm qwen (this loop builds SDM-only models), --store-opt sparse (its row gradients live outside
  p.grad, so DDP would not average them), --rt-shuffle (the forward draws torch random numbers per rank, so no run
  at another world size can reproduce it), --accum that does not divide --B. A trainer flag added later that this
  loop does not know is refused too, until it is implemented here.
- GLOBAL BATCH. For step s the offsets are default_rng([seed, s]).integers(0, n - (T + 1), size=world * B); rank r
  takes rows [r * B, (r + 1) * B). With world 1 this is S0.batch_for exactly; with world W it is the batch one device
  draws at B' = W * B (numpy draws the offsets one after another, so the first B of a longer draw equal the
  shorter draw). Steps = tokens // (world * B * T); warmup is 2% of those steps.
- THE EQUIVALENCE, exact in maths: world W, --B B, --accum A here equals the single trainer at --B W*B --accum W*A.
  Each rank cuts its B rows into A micro-batches of B/A rows; the single trainer cuts W*B rows into W*A micro-batches
  of the same size, and rank r's micro-batch i IS the single trainer's micro-batch r*A + i. So every micro-batch sees
  the same rows and the same batch-norm statistics. The loss of a micro-batch is divided by A; DDP averages the ranks,
  so the gradient is the sum over all W*A micro-batches divided by W*A, as on one device. Example: the run line
  `--B 512 --accum 64` on one GPU is `--world 2 --B 256 --accum 32` here.
- ACCUMULATION: every micro-batch but the last runs inside DDP's no_sync(), so gradients add up locally and are
  all-reduced once per step. --count-allreduce counts DDP's bucket all-reduces through a comm hook and logs them per
  step (the self-test uses it to prove one round per step).
- MUON under DDP: DDP averages p.grad during the last backward, before the optimiser runs, so Muon reads the same
  averaged gradient on every rank, and its Newton-Schulz step is deterministic, so every rank takes the identical
  update. check_ranks() proves it: at every checkpoint and at the end it all-gathers a bit-exact digest of every
  parameter, buffer and Muon momentum buffer; rank 0 logs {"event": "rank_check", "identical": ...} and every rank
  stops the run if they differ.
- BATCH NORM. Each hop's query passes a BatchNorm with no affine terms. In training mode it normalises with the
  micro-batch's own statistics (the same rows on one device, see above). Its RUNNING statistics, used by evaluation
  and saved in checkpoints, are updated once per micro-batch: r <- f r + (1 - f) s, f = 1 - momentum. After each step
  bn_combine() sets every rank to what one device would hold after its W*A sequential updates: with c_r updates on
  rank r and A_r its result from the common start r0, R = f^(sum c) r0 + sum_r f^(sum_{s>r} c_s) (A_r - f^(c_r) r0),
  one all-reduce of the (small) buffers, computed in float64. --sync-bn instead converts to SyncBatchNorm (statistics
  over the global micro-batch, so it equals one device at accum 1 only) and skips bn_combine.
- CHECKPOINTS ARE PORTABLE, in both directions, between this file at any world size and track4_sdmonly_train.py.
  last.pt and keep_<tokens>.pt hold exactly the single trainer's keys - "model" (the plain model's state dict: no
  DDP, compile or loss-module prefix), "opt" (build_optimizer's state dict: AdamW, or the composite with Muon's
  momentum buffers), "step", "cfg" (the S0 cfg, every build setting, and cfg["data"]), "arm" (and "tokens" in a keep
  file) - plus "ddp" {world_size, B, accum, B_global}, which the single trainer ignores. The schedule position is a
  pure function of the step and of the step count (tokens // global batch tokens); the data position is a pure
  function of (seed, step, train length, global batch). Training reads no torch random numbers (rt_shuffle is
  refused), so no RNG state is stored and none is needed.
- RESUME CHECK (shared with the single trainer): cfg["data"] records train_n, a fingerprint of the train file,
  B_global and T. A resume refuses when any differs: a run moved to another machine must use a train file of the
  SAME length and content, or pass --data-n <recorded n> to draw from the first n tokens of a longer file whose
  fingerprint still matches. A world size or B that keeps world x B is allowed (2 x 256 resumes as 1 x 512).
- LEARNING RATE. --lr is the peak rate as given. --lr-scale sqrt or linear rescales it by (world * B /
  --lr-ref-batch) ** 0.5 or ** 1 (default none). The result json records lr (the rate used), lr_flag and lr_scale.
- VALUE TABLES. A store's table is n_sub^2 x d and a step touches only the rows that woke. --table-sync dense (the
  default) lets DDP all-reduce the whole gradient; --table-sync rows keeps the tables out of DDP's reducer and, after
  the last micro-batch, all-reduces a row mask then only the touched rows. Equal to the dense average (self-test).
- COMPILE. The chunked loss is compiled as in the single trainer (the whole hidden -> loss graph). torch.compile's
  DDP optimiser, which splits that graph at gradient-bucket boundaries to overlap the all-reduce with backward, is
  OFF by default: split graphs fuse differently, the float sums run in another order and world 1 stops being
  bit-equal to the single trainer (measured: 8.8e-5 on the embedding after 6 steps of the self-test recipe). With
  --accum 32 the all-reduce runs once per 32 micro-batches, so the overlap it buys is at most one micro-batch's
  backward. --ddp-graph-split turns it back on.
- Evaluation, checkpoints, the per-window TEST file, the result json and the ablations run on rank 0 only; the other
  ranks wait in their next collective (the process group timeout is 60 minutes). Log `tok_per_s` is the total over
  ranks. Launch: `--world N` spawns the ranks itself, or under torchrun RANK and WORLD_SIZE are read. Rank r uses
  cuda:(r mod GPU count); more ranks than GPUs is a correctness run only (gloo). MPS runs world 1 only.
Docs: TRAINING_SDMONLY_HOW_WE_TRAIN.md · SDMONLY_CHANNEL.md (lanes MULTIGPU, DDPRECIPE) · lanes/LANE_DDPRECIPE.md
</claudes_code_comments>
"""
import argparse
import copy
import datetime
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track4_sdmonly_train as TR  # noqa: E402  (sets the SDMLLM_RUNS default, imports the S0 trainer)
import track4_sdmonly_models as M  # noqa: E402
import track4_sdmonly_optim as O  # noqa: E402

S0 = TR.S0

# the fields of track4_sdmonly_train.parse() this loop implements; a field outside this set that differs from its
# default is a flag this loop does not run yet, and is refused
KNOWN = {"arm", "seed", "tokens", "B", "accum", "T", "d", "d_a", "n_sub", "heads", "k", "hops", "softness", "share_store",
         "readout_f", "untie", "hop_mlp", "ng_rows", "ng_orders", "conj", "rt_write", "rt_buckets", "mix_layers",
         "mix_heads", "mix_n_sub", "mix_d_a", "mix_k", "mix_decays", "mix_f", "mix_chunk", "mix_impl", "graded", "rt_shuffle",
         "dense_f", "n_back", "decays", "qgrad", "lr", "store_lr_mult", "store_wd", "sched", "wsd_decay", "train_name",
         "compile", "loss", "ce_chunk", "init", "extra_val", "log_every", "eval_every", "ckpt_every", "quick_val",
         "val_windows", "run", "fresh", "cpu", "store_opt", "body_opt", "muon_lr", "muon_wd", "init_ckpt",
         "cooldown_tokens", "keep_at", "data_n", "M", "M_big", "n_layer", "f", "value_init", "ngram_orders"}

EXAMPLE = """examples:
  # the recipe on 2 GPUs of one box: global batch 512 windows, micro-batches of 8 (= one GPU at --B 512 --accum 64)
  python3 track4_sdmonly_train_ddp.py --world 2 --arm sdmonly --seed 0 --tokens 2000000000 --run NAME \\
      --train-name train_big --loss chunked --d 1536 --hop-mlp 6144 --hops 16 --share-store --untie --conj 2 \\
      --body-opt muon --B 256 --accum 32 --eval-every 5000 --ckpt-every 1000 --log-every 100
  # the same under torchrun
  torchrun --nproc_per_node 2 track4_sdmonly_train_ddp.py --arm sdmonly ... (same flags, no --world)
  # resume that run on ONE GPU elsewhere (copy checkpoints/NAME/ and use a train file of the same length)
  python3 track4_sdmonly_train.py --arm sdmonly ... --run NAME --B 512 --accum 64   (no --fresh)
  # speed only, random tokens, no data files
  python3 track4_sdmonly_train_ddp.py --bench 30 --world 2 --arm sdmonly --loss chunked --d 1536 --hop-mlp 6144 \\
      --hops 16 --share-store --untie --conj 2 --body-opt muon --B 256 --accum 32
  # the gate (CPU, gloo); --only takes section names
  python3 track4_sdmonly_train_ddp.py --selftest
"""


def own_parser():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="SDMONLY multi-GPU: train one SDM-only arm across the GPUs of one box. "
                                            "Flags below are this file's; every flag of track4_sdmonly_train.py "
                                            "(printed after them) is accepted too. --B and --accum are PER RANK.",
                                epilog=EXAMPLE)
    p.add_argument("-h", "--help", action="store_true")
    p.add_argument("--world", type=int, default=0, help="ranks to spawn; 0 = one per visible GPU (1 on CPU or MPS)")
    p.add_argument("--table-sync", default="dense", choices=["dense", "rows"],
                   help="dense = DDP all-reduces whole value tables; rows = all-reduce only the rows a step touched")
    p.add_argument("--sync-bn", action="store_true",
                   help="SyncBatchNorm over the ranks (equals one device only at --accum 1; default: per micro-batch)")
    p.add_argument("--lr-scale", default="none", choices=["none", "sqrt", "linear"],
                   help="none = --lr is used as given (default); sqrt / linear scale it by global batch / --lr-ref-batch")
    p.add_argument("--lr-ref-batch", type=int, default=32, help="the batch (in windows) that --lr was tuned at")
    p.add_argument("--backend", default="auto", choices=["auto", "nccl", "gloo"])
    p.add_argument("--no-loss-compile", action="store_true",
                   help="chunked loss without torch.compile (the single trainer always compiles it)")
    p.add_argument("--ddp-graph-split", action="store_true",
                   help="let torch.compile split the graph at DDP bucket boundaries (overlaps the all-reduce with "
                        "backward; float sums then run in another order, so world 1 is no longer bit-equal)")
    p.add_argument("--count-allreduce", action="store_true",
                   help="count DDP's gradient all-reduces through a comm hook and log them per step")
    p.add_argument("--bench", type=int, default=0, metavar="STEPS", help="time STEPS steps on random tokens and exit")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--only", default="", help="--selftest: run only these comma-separated sections")
    return p


def parse(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    p = own_parser()
    own, rest = p.parse_known_args(argv)
    if own.help:
        p.print_help()
        print("\n---- flags of track4_sdmonly_train.py (all accepted here) ----")
        TR.parse(["-h"])
    a = TR.parse(rest)
    ref = TR.parse([])
    unknown = [k for k in vars(a) if k not in KNOWN and getattr(a, k) != getattr(ref, k, None)]
    if unknown:
        sys.exit(f"track4_sdmonly_train_ddp.py does not implement trainer flag(s) {unknown} yet. "
                 f"Run without them, or use the single-device trainer.")
    if a.arm not in ("sdmonly", "sdmonly_dense", "sdmonly_none"):
        sys.exit(f"--arm {a.arm} is refused: this loop builds the SDM-only models (sdmonly, sdmonly_dense, sdmonly_none).")
    if a.store_opt == "sparse":
        sys.exit("--store-opt sparse is refused: its row gradients live outside p.grad, so DDP would not average them "
                 "across ranks. Use --store-opt dense here, or the single-device trainer.")
    if a.rt_shuffle:
        sys.exit("--rt-shuffle is refused: its forward draws torch random numbers per rank, so no run at another world "
                 "size (or a resumed run) can reproduce it. Use the single-device trainer.")
    if a.accum < 1 or a.B % a.accum:
        sys.exit(f"--accum {a.accum} must divide --B {a.B} (every micro-batch the same size, as the equivalence needs).")
    for k, v in vars(own).items():
        setattr(a, k, v)
    a.argv = argv
    return a


def global_offsets(step, seed, n, world, B, T):
    return np.random.default_rng([seed, step]).integers(0, n - (T + 1), size=world * B)


def rank_batch(step, seed, arr, B, T, rank, world):
    offs = global_offsets(step, seed, len(arr), world, B, T)[rank * B:(rank + 1) * B]
    return np.stack([arr[o:o + T + 1] for o in offs]).astype(np.int64)


def scaled_lr(a, world):
    ratio = world * a.B / a.lr_ref_batch
    return a.lr * {"none": 1.0, "sqrt": ratio ** 0.5, "linear": ratio}[a.lr_scale]


class LossModule(nn.Module):
    """The model plus its loss, so DDP's forward covers the whole graph. Parameter names gain the prefix 'model.'."""

    def __init__(self, model, loss, chunk):
        super().__init__()
        self.model, self.loss, self.chunk = model, loss, chunk

    def forward(self, x, t):
        if self.loss == "chunked":
            from track4_sdmllm24_fused_ce import ce_chunked
            return ce_chunked(self.model.hidden(x), self.model.emb.weight, t, self.chunk)
        return F.cross_entropy(self.model(x).float().reshape(-1, self.model.V), t.reshape(-1))


def rank_device(a, rank):
    if a.cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pick_backend(a, device, world):
    if a.backend != "auto":
        return a.backend
    return "nccl" if device.type == "cuda" and world <= torch.cuda.device_count() else "gloo"


def s0_cfg(a):
    return {"d": a.d, "T": a.T, "seed": a.seed, "M": a.M, "k": a.k, "hops": a.hops, "softness": a.softness,
            "n_layer": a.n_layer, "M_big": a.M_big, "value_init": a.value_init, "qgrad": a.qgrad,
            "ngram_orders": [int(x) for x in a.ngram_orders.split(",")], "f": a.f, "train_name": a.train_name,
            "n_back": a.n_back, "decays": [float(x) for x in a.decays.split(",")]}


def build_all(a, rank, world, device, state=None):
    """Returns model (plain), net (what the loop calls), opt, tables (value tables synced by hand, or []), cfg."""
    import torch._dynamo
    torch._dynamo.config.optimize_ddp = bool(a.ddp_graph_split)  # off: the compiled graph is the single trainer's
    cfg = {**s0_cfg(a), **TR.extra_cfg(a)}  # the single trainer's checkpoint cfg (its build hook adds extra_cfg)
    torch.manual_seed(a.seed)
    model = M.build_sdmonly(a.arm, S0.V, cfg).to(device)
    if state is not None:
        model.load_state_dict(state)
    if a.sync_bn and world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # the single trainer's optimiser: S0's AdamW groups, or the composite with Muon on the 2-D body matrices
    opt = O.build_optimizer(model, a.lr, a.store_lr_mult, a.store_wd, a.body_opt, a.store_opt, a.muon_lr, a.muon_wd,
                            n_tokens=a.B * a.T, betas=(0.9, 0.95), eps=1e-8)
    core = torch.compile(model) if a.compile else model
    if a.loss == "chunked" and getattr(model, "store", None) is not None:
        model.store.collect_stats = False
    lossmod = LossModule(core, a.loss, a.ce_chunk)
    if a.loss == "chunked" and not a.no_loss_compile:  # as the S0 trainer: the hidden -> chunked CE graph is compiled
        lossmod = torch.compile(lossmod)
    tables = []
    if device.type == "mps":
        return model, lossmod, opt, tables, cfg
    if a.table_sync == "rows" and world > 1:
        tables = [p for n, p in model.named_parameters() if n.startswith("store.") and n.endswith(".values")]
    for p in tables:  # DDP's reducer takes the parameters that require grad at construction; the tables stay out
        p.requires_grad_(False)
    net = nn.parallel.DistributedDataParallel(lossmod, device_ids=[device.index] if device.type == "cuda" else None,
                                              broadcast_buffers=False,
                                              # the no-read arm keeps per-hop norms it never calls
                                              find_unused_parameters=(a.arm == "sdmonly_none"))
    for p in tables:
        p.requires_grad_(True)
    if a.count_allreduce:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        def counting_hook(state, bucket):
            state["n"] += 1
            return default_hooks.allreduce_hook(None, bucket)
        net.allreduce_count = {"n": 0}
        net.register_comm_hook(net.allreduce_count, counting_hook)
    return model, net, opt, tables, cfg


def sync_table_rows(tables, world):
    """Average each value table's gradient over ranks, sending only rows that any rank touched. Returns
    (bytes all-reduced, rows sent, rows total)."""
    sent, rows_sent, rows_all = 0, 0, 0
    for p in tables:
        g = p.grad
        mask = (g != 0).any(1).to(torch.int32)
        dist.all_reduce(mask)
        idx = mask.nonzero().squeeze(1)
        rows = g[idx]
        dist.all_reduce(rows)
        g[idx] = rows / world
        sent += mask.numel() * 4 + rows.numel() * 4
        rows_sent += idx.numel()
        rows_all += g.shape[0]
    return sent, rows_sent, rows_all


def dense_table_bytes(model):
    return sum(p.numel() * 4 for n, p in model.named_parameters() if n.startswith("store.") and n.endswith(".values"))


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]


def bn_snapshot(bns):
    return [(m.running_mean.clone(), m.running_var.clone(), int(m.num_batches_tracked)) for m in bns]


@torch.no_grad()
def bn_combine(bns, snaps, rank, world):
    """Running statistics as one device would hold them after updating with rank 0's micro-batches, then rank 1's, ..."""
    if not bns:
        return
    dev = bns[0].running_mean.device
    cnt = torch.zeros(len(bns), world, dtype=torch.float64, device=dev)
    for j, (m, (_, _, nb0)) in enumerate(zip(bns, snaps)):
        cnt[j, rank] = int(m.num_batches_tracked) - nb0
    dist.all_reduce(cnt)
    cnt = cnt.cpu().tolist()
    parts = []
    for j, (m, (rm0, rv0, _)) in enumerate(zip(bns, snaps)):
        assert m.momentum is not None, "a cumulative-average batch norm is not combined here"
        f = 1.0 - m.momentum
        later = sum(cnt[j][rank + 1:])
        for buf, b0 in ((m.running_mean, rm0), (m.running_var, rv0)):
            parts.append(((buf.double() - f ** cnt[j][rank] * b0.double()) * f ** later).reshape(-1))
    flat = torch.cat(parts)
    dist.all_reduce(flat)
    off = 0
    for j, (m, (rm0, rv0, nb0)) in enumerate(zip(bns, snaps)):
        f = 1.0 - m.momentum
        tot = sum(cnt[j])
        for buf, b0 in ((m.running_mean, rm0), (m.running_var, rv0)):
            n = buf.numel()
            buf.copy_((f ** tot * b0.double() + flat[off:off + n].view_as(buf)).to(buf.dtype))
            off += n
        m.num_batches_tracked.fill_(nb0 + int(round(tot)))


@torch.no_grad()
def state_digest(model, opt):
    """Two integers per tensor from its bit pattern (a plain sum and a position-weighted sum, int64, wrapping), so two
    ranks agree only if every bit of every parameter, buffer and Muon momentum buffer agrees."""
    tensors = list(model.parameters()) + list(model.buffers())
    muon = getattr(opt, "parts", {}).get("muon")
    if muon is not None:
        tensors += list(muon.bufs)
    out = []
    for t in tensors:
        x = t.detach().reshape(-1)
        if x.dtype.is_floating_point:
            x = x.float().contiguous().view(torch.int32)
        x = x.to(torch.int64)
        s, w = torch.zeros((), dtype=torch.int64, device=x.device), torch.zeros((), dtype=torch.int64, device=x.device)
        for i in range(0, x.numel(), 1 << 22):
            c = x[i:i + (1 << 22)]
            s += c.sum()
            w += (c * (torch.arange(i, i + c.numel(), device=x.device) % 9973 + 1)).sum()
        out += [s, w]
    return torch.stack(out) if out else torch.zeros(0, dtype=torch.int64)


def check_ranks(model, opt, rank, world, step):
    """Every rank must hold bit-identical weights and Muon buffers. Returns the record (rank 0) or None."""
    if world == 1:
        return None
    d = state_digest(model, opt)
    got = [torch.zeros_like(d) for _ in range(world)]
    dist.all_gather(got, d)
    same = all(torch.equal(got[0], g) for g in got[1:])
    rec = {"event": "rank_check", "step": step, "identical": same, "tensors": int(d.numel() // 2)}
    if not same:
        bad = sorted({int(i) // 2 for g in got[1:] for i in (g != got[0]).nonzero().reshape(-1).tolist()})
        rec["differing_tensors"] = bad[:20]
        if rank == 0:
            print(json.dumps(rec), flush=True)
        raise SystemExit(f"rank check failed at step {step}: the ranks' weights differ (tensors {bad[:20]})")
    return rec if rank == 0 else None


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda"))


def train_step(a, net, model, opt, tables, bns, xb, world, rank, device, step=None, dump=None):
    """One optimiser step on this rank's B rows. Returns (loss: the mean over this rank's rows, grad norm, sync)."""
    opt.zero_grad(set_to_none=True)
    combine = bool(bns) and world > 1 and not a.sync_bn
    snaps = bn_snapshot(bns) if combine else None
    mbs = xb.chunk(a.accum) if a.accum > 1 else [xb]
    ddp = isinstance(net, nn.parallel.DistributedDataParallel)
    loss = 0.0
    for i, mb in enumerate(mbs):
        ctx = net.no_sync() if (ddp and i < len(mbs) - 1) else nullcontext()
        with ctx:
            with autocast(device):
                lm = net(mb[:, :-1], mb[:, 1:])
                if a.accum > 1:
                    lm = lm / a.accum
            lm.backward()
        loss = loss + lm.detach() if a.accum > 1 else lm.detach()
    sync = sync_table_rows(tables, world) if tables else (0, 0, 0)
    if combine:
        bn_combine(bns, snaps, rank, world)
    if dump and rank == 0 and (dump[1] is None or step in dump[1]):  # the averaged gradient, before the clip
        torch.save({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None},
                   os.path.join(dump[0], f"grad_{step}.pt"))
    gn = O.clip_grad_norm_(list(model.parameters()), 1.0, opt)
    opt.step()
    return loss, gn, sync


def rank_rates(rank, world, value, device):
    v = torch.zeros(world, device=device if device.type != "mps" else "cpu")
    v[rank] = value
    if world > 1:
        dist.all_reduce(v)
    return [round(float(x), 1) for x in v]


def prepare_cooldown_ddp(a, rank, world):
    """TR.prepare_cooldown at the global batch: rank 0 checks and copies the source; every rank gets the schedule."""
    vals = [None, None, None]
    if rank == 0:
        b = copy.copy(a)
        b.B = world * a.B  # the fork's step count is in global-batch tokens
        TR.prepare_cooldown(b)
        vals = [b.tokens, b.cooldown_step0, b.cooldown_peak]
    if world > 1:
        dist.broadcast_object_list(vals, src=0)
    a.tokens, a.cooldown_step0, a.cooldown_peak = vals
    a.fresh = False  # the loop resumes from the copied checkpoint
    return TR.cooldown_lr(a.cooldown_step0, a.cooldown_peak)


def save_ckpt(path, model, opt, step, cfg, a, world, extra=None):
    """The single trainer's layout (model, opt, step, cfg, arm [, tokens]) plus the "ddp" record it ignores."""
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg, "arm": a.arm,
                **(extra or {}), "ddp": {"world_size": world, "B": a.B, "accum": a.accum, "B_global": world * a.B}},
               path + ".tmp")
    os.replace(path + ".tmp", path)


def train(a, rank, world, device):
    main_rank = rank == 0
    run = a.run
    ck_dir = os.path.join(os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints")), run)
    out_dir = os.environ["SDMLLM_RUNS"]
    last = os.path.join(ck_dir, "last.pt")
    if main_rank:
        os.makedirs(ck_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
    lr_at = TR.wsd_lr(a.wsd_decay) if a.sched == "wsd" else S0.lr_at
    if a.init_ckpt:
        lr_at = prepare_cooldown_ddp(a, rank, world)
    train_arr = TR.train_array(a)
    rec = TR.data_record(a, train_arr, world * a.B, world * a.accum)
    state, opt_state, step = None, None, 0
    if a.init:
        ck0 = torch.load(a.init, map_location=device)
        state = ck0["model"]
        if main_rank:
            print(json.dumps({"event": "init", "from": a.init, "step": ck0.get("step")}), flush=True)
    if os.path.exists(last) and not a.fresh:
        ck = torch.load(last, map_location=device)
        old = ck.get("cfg", {}).get("data")
        if old is None and "ddp" in ck:  # a checkpoint of this file written before the data record existed
            was = ck["ddp"]["world_size"] * ck["ddp"]["B"]
            if was != world * a.B:
                sys.exit(f"resume refused: {last} was written at global batch {was}; this run is {world} x {a.B}. "
                         f"The batches of a step differ. Use --fresh or match it.")
        if old is not None or rank == 0:  # with no record only rank 0 prints the note
            TR.check_resume_data(old, rec, last)
        state, opt_state, step = ck["model"], ck["opt"], ck["step"]
        del ck
    model, net, opt, tables, cfg = build_all(a, rank, world, device, state)
    cfg["data"] = rec
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    bns = bn_modules(model)
    peak = scaled_lr(a, world)
    tokens_per_step = world * a.B * a.T
    total = a.tokens // tokens_per_step
    warm = max(1, int(0.02 * total))
    keep = {math.ceil(t / tokens_per_step): t for t in a.keep_at}
    dump = TR.grad_dump_spec()  # SDMONLY_DUMP_GRADS, tests only
    backend = dist.get_backend() if dist.is_initialized() else "none"
    if main_rank:
        tb = torch.from_numpy(np.fromfile(os.path.join(S0.DATA, "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
        val_arr = S0.load_split("val")
        starts = S0.val_windows(val_arr, a.T, "test", a.val_windows)
        ngpu = torch.cuda.device_count() if device.type == "cuda" else 0
        info = {"run": run, "arm": a.arm, "seed": a.seed, "cfg": cfg, "B": a.B, "T": a.T, "tokens_target": a.tokens,
                "steps": total, "lr": peak, "loss_path": a.loss, "store_lr_mult": a.store_lr_mult, "store_wd": a.store_wd,
                "params": S0.count_params(model), "flops_per_token_fwd": M.flops_per_token(model, a.T),
                "device": str(device), "stamp_start": S0.stamp(), "resumed_from_step": step, "init": a.init,
                "train_name": a.train_name,
                "world_size": world, "B_global": world * a.B, "accum": a.accum, "accum_global": world * a.accum,
                "single_device_equivalent": f"--B {world * a.B} --accum {world * a.accum}",
                "lr_flag": a.lr, "lr_scale": a.lr_scale, "lr_ref_batch": a.lr_ref_batch, "table_sync": a.table_sync,
                "sync_bn": bool(a.sync_bn and world > 1), "backend": backend, "gpus_visible": ngpu,
                "ranks_share_a_gpu": bool(ngpu and world > ngpu), "table_bytes_dense": dense_table_bytes(model),
                "trainer": "track4_sdmonly_train_ddp.py"}
        logf = open(os.path.join(out_dir, f"{run}.log"), "a")
        logf.write(json.dumps({"event": "start", **info}) + "\n")
        logf.flush()
        print(json.dumps(info), flush=True)
        if info["ranks_share_a_gpu"]:
            print(f"NOTE {world} ranks share {ngpu} GPU(s): a correctness run, NOT a speed measurement.", flush=True)

    def log(rec_):
        if main_rank and rec_:
            logf.write(json.dumps(rec_) + "\n")
            logf.flush()
            print(json.dumps(rec_), flush=True)

    t0 = time.time()
    tok_seen = step * tokens_per_step
    curve, rate_log, rank_checks = [], [], []
    sync_acc = [0, 0, 0]
    ar0 = net.allreduce_count["n"] if hasattr(net, "allreduce_count") else 0
    while step < total:
        if main_rank and step in keep and step > 0:  # as the single trainer: the state after `step` steps
            path = os.path.join(ck_dir, f"keep_{keep[step]}.pt")
            if not os.path.exists(path):
                save_ckpt(path, model, opt, step, cfg, a, world, {"tokens": step * tokens_per_step})
                log({"event": "keep", "path": path, "step": step, "tokens": step * tokens_per_step})
        lr = lr_at(step, total, peak, warm)
        for gpar in opt.param_groups:
            gpar["lr"] = lr * gpar.get("lr_mult", 1.0)
        xb = torch.from_numpy(rank_batch(step, a.seed, train_arr, a.B, a.T, rank, world)).to(device)
        loss, gn, s = train_step(a, net, model, opt, tables, bns, xb, world, rank, device, step, dump)
        sync_acc = [x + y for x, y in zip(sync_acc, s)]
        step += 1
        tok_seen += tokens_per_step
        if step % a.log_every == 0 or step == total:
            dt = time.time() - t0
            lv = torch.stack([torch.as_tensor(loss, dtype=torch.float32, device=device if device.type != "mps" else "cpu")])
            if world > 1:
                dist.all_reduce(lv)
            rates = rank_rates(rank, world, a.log_every * a.B * a.T / dt, device)
            if main_rank:
                r = {"event": "step", "step": step, "loss": round(float(lv) / world, 4), "loss_rank0": round(float(loss), 4),
                     "lr": lr, "gnorm": round(float(gn), 3), "tok_per_s": round(sum(rates), 1), "tok_per_s_rank": rates,
                     "load": os.getloadavg()[0]}
                if tables and sync_acc[2]:
                    r["table_sync_mb"] = round(sync_acc[0] / a.log_every / 1e6, 2)
                    r["table_rows_frac"] = round(sync_acc[1] / sync_acc[2], 4)
                if hasattr(net, "allreduce_count"):
                    r["allreduce_calls"] = net.allreduce_count["n"] - ar0
                    ar0 = net.allreduce_count["n"]
                if getattr(model, "hop_stats", None):
                    r["hop_stats"] = model.hop_stats
                rate_log.append((sum(rates), rates))
                log(r)
            sync_acc = [0, 0, 0]
            t0 = time.time()
        if main_rank and step % a.eval_every == 0 and step < total:
            ev = S0.eval_bpb(model, val_arr, starts[: a.quick_val], a.T, tb, device)
            curve.append({"step": step, "tokens": tok_seen, "bpb_quick": ev["bpb"]})
            log({"event": "eval_quick", "step": step, **ev})
        if step % a.ckpt_every == 0 or step == total:
            rc = check_ranks(model, opt, rank, world, step)
            if rc:
                rank_checks.append(rc)
                log(rc)
            if main_rank:
                save_ckpt(last, model, opt, step, cfg, a, world)
    if not main_rank:
        return None
    if getattr(model, "store", None) is not None:
        model.store.collect_stats = True
    res = {**info, "stamp_end": S0.stamp(), "curve": curve, "rank_checks": rank_checks}
    tail = rate_log[len(rate_log) // 10:]
    if tail:
        res["tok_per_s_total"] = float(np.median([t for t, _ in tail]))
        res["tok_per_s_per_rank"] = [float(x) for x in np.median(np.array([r for _, r in tail]), axis=0)]
    res["val_test"] = S0.eval_bpb(model, val_arr, starts, a.T, tb, device)
    for name in a.extra_val:
        ex = S0.load_split(name)
        res.setdefault("val_extra", {})[name] = S0.eval_bpb(model, ex, S0.val_windows(ex, a.T, "all", 4000), a.T, tb, device)
        print("extra", name, json.dumps(res["val_extra"][name]), flush=True)
    if a.val_windows is None:
        from track4_sdmllm_paired_window_comparison import per_window
        model.eval()
        pn, pb = per_window(model, val_arr, starts, a.T, tb, device)
        np.savez(os.path.join(ck_dir, "test_per_window.npz"), nats=pn, bytes=pb)
        model.train()
    if TR.opt_info(a):
        res["sdmonly_opt"] = TR.opt_info(a)
    logf.write(json.dumps({"event": "final", "val_test": res["val_test"]}) + "\n")
    logf.close()
    res_path = os.path.join(out_dir, f"{run}.result.json")
    json.dump(res, open(res_path, "w"), indent=1)
    print("FINAL", run, json.dumps(res["val_test"]), flush=True)
    return res_path


def bench(a, rank, world, device):
    """The recipe's step (accum, no_sync, batch-norm combine, Muon, clip) on random tokens; no data files. The first
    third of the steps is warmup and is not timed."""
    model, net, opt, tables, _ = build_all(a, rank, world, device)
    bns = bn_modules(model)
    g = torch.Generator().manual_seed(1000 + rank)
    xb = torch.randint(1, S0.V, (a.B, a.T + 1), generator=g).to(device)
    warm = max(3, a.bench // 3)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda *_: None)
    acc = [0, 0, 0]
    for gpar in opt.param_groups:
        gpar["lr"] = a.lr * gpar.get("lr_mult", 1.0)
    for i in range(warm + a.bench):
        if i == warm:
            if world > 1:
                dist.barrier()
            sync()
            t0 = time.time()
            acc = [0, 0, 0]
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        _, _, s = train_step(a, net, model, opt, tables, bns, xb, world, rank, device)
        acc = [x + y for x, y in zip(acc, s)]
    sync()
    dt = time.time() - t0
    rates = rank_rates(rank, world, a.bench * a.B * a.T / dt, device)
    rc = check_ranks(model, opt, rank, world, warm + a.bench)
    if rank == 0:
        touched = [float((b.values.grad != 0).any(1).float().mean()) for b in model.store.banks
                   if b.values.grad is not None] if a.arm == "sdmonly" else []
        ngpu = torch.cuda.device_count() if device.type == "cuda" else 0
        out = {"event": "bench", "world_size": world, "B": a.B, "accum": a.accum, "T": a.T, "B_global": world * a.B,
               "single_device_equivalent": f"--B {world * a.B} --accum {world * a.accum}", "arm": a.arm, "d": a.d,
               "n_sub": a.n_sub, "hops": a.hops, "hop_mlp": a.hop_mlp, "share_store": a.share_store, "untie": a.untie,
               "conj": a.conj, "body_opt": a.body_opt, "loss_path": a.loss, "compile": a.compile,
               "table_sync": a.table_sync, "steps_timed": a.bench, "tok_per_s_total": round(sum(rates), 1),
               "tok_per_s_per_rank": rates, "s_per_step": round(dt / a.bench, 4),
               "params": S0.count_params(model), "table_bytes_dense_mb": round(dense_table_bytes(model) / 1e6, 1),
               "table_sync_mb_per_step": round(acc[0] / a.bench / 1e6, 2) if tables else None,
               "table_rows_touched_frac_per_store": [round(t, 4) for t in touched],
               "peak_mem_gb_rank0": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None,
               "ranks_identical": rc["identical"] if rc else None,
               "note_rows": "random tokens: the touched fraction on real text may differ",
               "backend": dist.get_backend() if dist.is_initialized() else "none", "gpus_visible": ngpu,
               "ranks_share_a_gpu": bool(ngpu and world > ngpu), "device": str(device), "stamp": S0.stamp()}
        print(json.dumps(out), flush=True)
        if out["ranks_share_a_gpu"]:
            print("NOTE ranks share a GPU: this is NOT a speed measurement.", flush=True)
        print("NEXT -> repeat with --world 1 (and the single-device equivalent --B/--accum); compare tok_per_s_total",
              flush=True)


def run_rank(rank, world, a, init_method=None):
    device = rank_device(a, rank)
    if device.type == "mps" and world > 1:
        sys.exit("MPS runs world 1 only. Use --cpu for a multi-rank correctness test on this machine.")
    if device.type == "cpu" and world > 1 and "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))  # a set OMP_NUM_THREADS (the clamp) is kept
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if device.type != "mps":
        kw = {"init_method": init_method, "rank": rank, "world_size": world} if init_method else {}
        dist.init_process_group(pick_backend(a, device, world), timeout=datetime.timedelta(minutes=60), **kw)
    try:
        if a.bench:
            bench(a, rank, world, device)
            return
        res_path = train(a, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        if a.arm == "sdmonly":
            TR.score_ablations(a, res_path)
        print("NEXT -> python3 track4_sdmonly_summary.py", flush=True)


def _spawned(rank, world, argv, port):
    run_rank(rank, world, parse(argv), f"tcp://127.0.0.1:{port}")


def launch(argv=None):
    a = parse(argv)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # under torchrun
        run_rank(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), a)
        return
    world = a.world or (torch.cuda.device_count() if (torch.cuda.is_available() and not a.cpu) else 1)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    if world == 1:
        run_rank(0, 1, a, f"tcp://127.0.0.1:{port}")
        return
    import torch.multiprocessing as mp
    mp.spawn(_spawned, args=(world, a.argv, port), nprocs=world, join=True)


# --------------------------------------------------------------------------------------------------------------------
SECTIONS = ["batch", "refuse", "parity1", "parity2", "ranks", "accum", "resume", "length", "fork", "rows", "bench",
            "summary"]


def selftest(only=""):
    """CPU only, gloo, made-up token shards in a temp folder. Runs the real single-device trainer as the reference.
    Sections (run a subset with --only a,b):
      batch    the global batch: world 1 = S0.batch_for; world W rows joined = one device at W x B; micro-batch order
      refuse   flags this loop cannot run are refused by name; an unknown future flag is refused
      parity1  world 1 on the recipe flags equals track4_sdmonly_train.py EXACTLY (losses, bpb, every tensor, keep files)
      parity2  world 2 x B 4 x accum 2 against one device at B 8 x accum 4: the gradient from the same state equal to
               float rounding, then 12 steps of losses, TEST bpb and parameters; world 4 x B 2 as a third float order
      ranks    Muon under DDP: every rank holds bit-identical weights and Muon buffers at every checkpoint
      accum    one gradient all-reduce round per step whatever --accum is (no_sync on all but the last micro-batch)
      resume   a 2-rank checkpoint resumes on 1 rank and a 1-rank checkpoint on 2 ranks: the first gradient after the
               move equals the unmoved run's from the same state, and steps 7..12 equal the straight 1-rank run
      length   a resume on a train file of another length or content is refused by both trainers; --data-n reads a prefix
      fork     the cooldown fork under DDP equals the single trainer's fork from the same checkpoint
      rows     rows table sync equals dense sync; the chunked loss equals the plain loss; --lr-scale
      bench    the bench mode runs the recipe and reports tokens a second
      summary  track4_sdmonly_summary.py reads the DDP results
    """
    import shutil
    want = set(x for x in only.split(",") if x) or set(SECTIONS)
    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)

    tmp = tempfile.mkdtemp(prefix="sdmonly_ddp_selftest_")
    data = os.path.join(tmp, "data")
    data_long = os.path.join(tmp, "data_long")
    os.makedirs(data)
    os.makedirs(data_long)
    g = np.random.default_rng(7)
    tr = g.integers(1, S0.V, size=6000).astype(np.uint32)
    va = g.integers(1, S0.V, size=800).astype(np.uint32)
    tbytes = g.integers(1, 8, size=S0.V).astype(np.int32)
    for dd, t in ((data, tr), (data_long, np.concatenate([tr, g.integers(1, S0.V, size=500).astype(np.uint32)]))):
        t.tofile(os.path.join(dd, "train.u32"))
        va.tofile(os.path.join(dd, "val.u32"))
        tbytes.tofile(os.path.join(dd, "token_bytes.i32"))
    base_env = {**os.environ, "SDMLLM_DATA": data, "SDMLLM_RUNS": os.path.join(tmp, "runs"), "SDMLLM_CKPT": os.path.join(tmp, "ck")}
    # one thread count for every process, single trainer and ranks alike
    base_env["OMP_NUM_THREADS"] = "2"
    for k in ("RANK", "WORLD_SIZE", "SDMONLY_DUMP_GRADS"):
        base_env.pop(k, None)
    grads = os.path.join(tmp, "grads")
    common = ["--T", "16", "--cpu", "--log-every", "1", "--eval-every", "6", "--ckpt-every", "6", "--seed", "3"]
    # the recipe on a tiny model: memory on, hop SwiGLUs, one shared store, untied input table, conj 2, graded reads
    # with the query gradient on, store lr x3, Muon on the body, wsd, the compiled chunked loss
    recipe = ["--arm", "sdmonly", "--d", "32", "--d-a", "16", "--n-sub", "16", "--k", "4", "--hops", "2", "--hop-mlp", "32",
              "--share-store", "--untie", "--conj", "2", "--qgrad", "1", "--graded", "--store-lr-mult", "3",
              "--body-opt", "muon", "--muon-lr", "0.05", "--muon-wd", "0.01", "--lr", "1e-2", "--sched", "wsd",
              "--wsd-decay", "0.5", "--loss", "chunked", "--ce-chunk", "48"]
    G, STEPS, KEEP = 8, 12, 6  # global batch 8 windows, 12 steps, keep-checkpoint at step 6
    tok = ["--tokens", str(G * 16 * STEPS), "--keep-at", str(G * 16 * KEEP)]

    def go(script, run, extra, expect_fail=False, env=None, fresh=True, dump=None):
        cmd = [sys.executable, os.path.join(HERE, script), "--run", run] + common + (["--fresh"] if fresh else []) + extra
        e = dict(env or base_env)
        if dump is not None:  # save the pre-clip gradient at these steps
            e["SDMONLY_DUMP_GRADS"] = os.path.join(grads, run) + "@" + ",".join(str(x) for x in dump)
        r = subprocess.run(cmd, env=e, capture_output=True, text=True)
        if (r.returncode != 0) != expect_fail:
            print("---- command:", " ".join(cmd[1:]))
            print(r.stdout[-3000:], r.stderr[-3000:])
        if expect_fail or r.returncode != 0:
            return r
        res = json.load(open(os.path.join(tmp, "runs", f"{run}.result.json")))
        lines = [json.loads(ln) for ln in open(os.path.join(tmp, "runs", f"{run}.log"))]
        start = max(i for i, x in enumerate(lines) if x.get("event") == "start")  # this run's part of an appended log
        steps = [x for x in lines[start:] if x.get("event") == "step"]
        rc = [x for x in lines[start:] if x.get("event") == "rank_check"]
        return res, steps, rc

    def ck(run, name="last.pt"):
        return torch.load(os.path.join(tmp, "ck", run, name), map_location="cpu", weights_only=False)

    def sd_diff(c1, c2, how="max", ints=True):
        """over every float tensor of two state dicts: max |a - b| (how 'max'), max |a - b| / max |a| ('rel'), or
        ||a - b|| / ||a|| ('l2'); inf when the keys or (ints) an integer tensor (batch-norm counts) differ"""
        a1, a2 = c1["model"], c2["model"]
        if set(a1) != set(a2) or (ints and any(not torch.equal(a1[k], a2[k]) for k in a1 if not a1[k].is_floating_point())):
            return float("inf")
        out = 0.0
        for k in a1:
            if a1[k].is_floating_point():
                d = (a1[k].double() - a2[k].double())
                if how == "max":
                    v = float(d.abs().max())
                elif how == "rel":
                    v = float(d.abs().max()) / max(float(a1[k].abs().max()), 1e-12)
                else:
                    v = float(d.norm()) / max(float(a1[k].double().norm()), 1e-12)
                out = max(out, v)
        return out

    def gdiff(run1, run2, step):
        """max over parameters of max |g1 - g2| / max |g1| for the gradients two runs saved at `step`"""
        try:
            g1 = torch.load(os.path.join(grads, run1, f"grad_{step}.pt"), weights_only=False)
            g2 = torch.load(os.path.join(grads, run2, f"grad_{step}.pt"), weights_only=False)
        except FileNotFoundError:
            return float("inf")
        if set(g1) != set(g2):
            return float("inf")
        out = 0.0
        for k in g1:
            m = float(g1[k].abs().max())
            d = float((g1[k].double() - g2[k].double()).abs().max())
            out = max(out, d / m if m > 0 else (0.0 if d == 0 else float("inf")))
        return out

    def ok_run(r):
        return isinstance(r, tuple)

    def maxdiff(s1, s2, n):
        return max(abs(x["loss"] - y["loss"]) for x, y in zip(s1, s2)) if len(s1) == len(s2) == n else float("inf")

    TOL_G, TOL_LOSS, TOL_BPB, TOL_L2 = 1e-5, 2e-4, 1e-4, 5e-2
    print(f"TOLERANCES  (world 1 is held to EXACT equality)\n"
          f"  gradient from the same state: {TOL_G} of each tensor's largest gradient. The maths is identical; float32 "
          f"sums run in another order (each rank sums its own micro-batches and DDP averages the ranks, against one "
          f"sequential sum; the compiled graph scales the loss at another point). Measured: about 1e-7.\n"
          f"  over 12 steps: logged loss {TOL_LOSS} absolute (the log rounds to 1e-4), TEST bpb {TOL_BPB}, every "
          f"parameter and buffer {TOL_L2} relative in L2 per tensor, batch-norm counts exact. Why the parameters get a "
          f"looser bound than the gradient: Adam's first update of an element is +-0.44 lr whatever the gradient's size, "
          f"so a gradient component within float noise of zero takes either sign, and the model carries that "
          f"difference forward (measured on this recipe: one such flip moved wq.0 by 2% in L2; without one, 1e-5). "
          f"The controls (other rows, other micro-batches) differ by far more.", flush=True)

    # ---- batch
    if "batch" in want:
        arr = np.memmap(os.path.join(data, "train.u32"), dtype=np.uint32, mode="r")
        check("batch: world 1, the rank's batch is S0.batch_for's batch",
              all(np.array_equal(rank_batch(s, 3, arr, 4, 16, 0, 1), S0.batch_for(s, 3, arr, 4, 16)) for s in range(5)))
        union = np.concatenate([rank_batch(9, 3, arr, 4, 16, r, 3) for r in range(3)])
        check("batch: world 3, the ranks' rows joined are the single-device batch at 3 x B",
              np.array_equal(union, S0.batch_for(9, 3, arr, 12, 16)))
        mb = [c for r in range(2) for c in np.array_split(rank_batch(9, 3, arr, 4, 16, r, 2), 2)]
        ref_mb = np.array_split(S0.batch_for(9, 3, arr, 8, 16), 4)
        check("batch: rank r micro-batch i is the single device's micro-batch r*A+i (A 2, world 2)",
              all(np.array_equal(x, y) for x, y in zip(mb, ref_mb)))
        check("batch: ranks draw different rows",
              not np.array_equal(rank_batch(9, 3, arr, 4, 16, 0, 3), rank_batch(9, 3, arr, 4, 16, 1, 3)))

    # ---- refuse
    if "refuse" in want:
        for flags, word in ((["--store-opt", "sparse"], "--store-opt sparse is refused"),
                            (["--rt-shuffle", "--rt-write", "4"], "--rt-shuffle is refused"),
                            (["--arm", "qwen"], "--arm qwen is refused"),
                            (["--B", "6", "--accum", "4"], "must divide --B")):
            r = go("track4_sdmonly_train_ddp.py", "ddp_refused", ["--world", "1", "--tokens", "1024"] + flags, expect_fail=True)
            out = (r.stdout + r.stderr).strip()
            check(f"refuse: {' '.join(flags)} is refused by name", r.returncode != 0 and word in out,
                  out.splitlines()[-1][:120] if out else "")
        saved = set(KNOWN)
        KNOWN.discard("conj")  # pretend the loop had not implemented --conj: the generic guard must refuse it
        try:
            parse(["--conj", "2"])
            refused = False
        except SystemExit as e:
            refused = "does not implement" in str(e) and "conj" in str(e)
        KNOWN.clear()
        KNOWN.update(saved)
        check("refuse: a trainer flag this loop does not know is refused by name", refused)

    # ---- parity1: world 1 against the single-device trainer, the recipe, accum 2
    if "parity1" in want:
        r_s = go("track4_sdmonly_train.py", "p1_single", recipe + tok + ["--B", "8", "--accum", "2"])
        r_d = go("track4_sdmonly_train_ddp.py", "p1_ddp_w1", recipe + tok + ["--B", "8", "--accum", "2", "--world", "1"])
        if ok_run(r_s) and ok_run(r_d):
            (ref, ref_s, _), (w1, w1_s, _) = r_s, r_d
            dl = maxdiff(ref_s, w1_s, STEPS)
            check("parity1: every logged loss equals the single-device trainer's", dl == 0.0, f"max abs diff {dl}")
            db = abs(ref["val_test"]["bpb"] - w1["val_test"]["bpb"])
            check("parity1: TEST bpb equals the single-device trainer's", db == 0.0, f"abs diff {db:.3e}")
            check("parity1: the ablation scores match",
                  ref["val_test_ablate_shuffle_keys"] == w1["val_test_ablate_shuffle_keys"]
                  and ref["val_test_ablate_zero_read"] == w1["val_test_ablate_zero_read"])
            c1, c2 = ck("p1_single"), ck("p1_ddp_w1")
            dp = sd_diff(c1, c2)
            check("parity1: the final checkpoints hold the same keys and identical tensors", dp == 0.0, f"max {dp}")
            check("parity1: the checkpoint layout is the single trainer's (plus 'ddp'), cfg equal",
                  set(c2) == set(c1) | {"ddp"} and c1["cfg"] == c2["cfg"] and c1["arm"] == c2["arm"] and c1["step"] == c2["step"],
                  f"single {sorted(c1)} ddp {sorted(c2)}")
            p1, p2 = c1["opt"].get("parts", {}), c2["opt"].get("parts", {})
            check("parity1: the optimiser state has Muon's momentum buffers in the same layout, equal",
                  "muon" in p1 and sorted(p1) == sorted(p2) and len(p1["muon"]["bufs"]) == len(p2["muon"]["bufs"])
                  and all(torch.equal(x, y) for x, y in zip(p1["muon"]["bufs"], p2["muon"]["bufs"])),
                  f"single parts {sorted(p1)}, ddp parts {sorted(p2)}")
            check("parity1: the loss falls", w1_s[-1]["loss"] < w1_s[0]["loss"], f"{w1_s[0]['loss']} -> {w1_s[-1]['loss']}")
            check("parity1: the result json carries the single trainer's keys and the DDP fields",
                  set(ref) <= set(w1) and w1["world_size"] == 1 and w1["B_global"] == 8 and "tok_per_s_total" in w1,
                  f"missing {set(ref) - set(w1)}")
            ks, kd = ck("p1_single", f"keep_{G * 16 * KEEP}.pt"), ck("p1_ddp_w1", f"keep_{G * 16 * KEEP}.pt")
            check("parity1: --keep-at writes the same keep checkpoint (step, tokens, weights)",
                  ks["step"] == kd["step"] == KEEP and ks["tokens"] == kd["tokens"] and sd_diff(ks, kd) == 0.0)
        else:
            check("parity1: both runs finished", False)

    # ---- parity2 (+ ranks, accum, resume, fork): world 2 x B 4 x accum 2 against one device at B 8 x accum 4
    if want & {"parity2", "ranks", "accum", "resume", "fork", "summary"}:
        r8 = go("track4_sdmonly_train.py", "p2_single_b8", recipe + tok + ["--B", "8", "--accum", "4"], dump=[0, KEEP])
        w2 = go("track4_sdmonly_train_ddp.py", "p2_ddp_w2", recipe + tok + ["--B", "4", "--accum", "2", "--world", "2",
                                                                            "--count-allreduce"], dump=[0, KEEP])
        both = ok_run(r8) and ok_run(w2)
        if "parity2" in want:
            if both:
                (ref, ref_s, _), (dd, dd_s, _) = r8, w2
                gd = gdiff("p2_single_b8", "p2_ddp_w2", 0)
                check("parity2: the averaged gradient of step 1 (same initial state) equals one device's",
                      gd <= TOL_G, f"max relative diff {gd:.1e}")
                dl = maxdiff(ref_s, dd_s, STEPS)
                check("parity2: world 2 x B 4 x accum 2 = one device at B 8 x accum 4: logged losses, 12 steps",
                      dl <= TOL_LOSS, f"max abs diff {dl:.1e}")
                db = abs(ref["val_test"]["bpb"] - dd["val_test"]["bpb"])
                check("parity2: TEST bpb (eval uses the combined batch-norm running statistics)", db <= TOL_BPB, f"abs diff {db:.2e}")
                c_s, c_d = ck("p2_single_b8"), ck("p2_ddp_w2")
                dp = sd_diff(c_s, c_d, "l2")
                check("parity2: every parameter and buffer after 12 steps, batch-norm counts exact", dp <= TOL_L2,
                      f"max relative L2 diff {dp:.1e}, max abs diff {sd_diff(c_s, c_d):.1e}")
                w4 = go("track4_sdmonly_train_ddp.py", "p2_ddp_w4", recipe + tok + ["--B", "2", "--accum", "1", "--world", "4"],
                        dump=[0])
                if ok_run(w4):
                    g4 = gdiff("p2_single_b8", "p2_ddp_w4", 0)
                    d4 = sd_diff(c_s, ck("p2_ddp_w4"), "l2")
                    check("parity2: world 4 x B 2 x accum 1 (the same maths, a third float order) agrees too",
                          g4 <= TOL_G and d4 <= TOL_L2 and maxdiff(ref_s, w4[1], STEPS) <= TOL_LOSS,
                          f"gradient {g4:.1e}, parameters L2 {d4:.1e}")
                else:
                    check("parity2: the world 4 run finished", False)
                check("parity2: the result says which single-device flags it equals",
                      dd["single_device_equivalent"] == "--B 8 --accum 4" and dd["world_size"] == 2)
                r4 = go("track4_sdmonly_train.py", "p2_single_b4", recipe + ["--tokens", str(4 * 16 * STEPS), "--B", "4", "--accum", "2"],
                        dump=[0])
                c4 = sd_diff(ck("p2_single_b4"), c_d, "l2", ints=False) if ok_run(r4) else 0.0
                g4c = gdiff("p2_single_b4", "p2_ddp_w2", 0)
                check("parity2 control: one device at B 4 (rank 0's rows only) is far from world 2 x B 4",
                      c4 > 10 * TOL_L2 and g4c > 1e3 * TOL_G, f"parameters L2 {c4:.2e}, gradient {g4c:.2e}")
                rm = go("track4_sdmonly_train.py", "p2_single_b8_a2", recipe + tok + ["--B", "8", "--accum", "2"], dump=[0])
                cm = sd_diff(ck("p2_single_b8_a2"), c_d, "l2", ints=False) if ok_run(rm) else 0.0
                gmc = gdiff("p2_single_b8_a2", "p2_ddp_w2", 0)
                check("parity2 control: one device at B 8 x accum 2 (other micro-batches, other batch-norm groups) is far",
                      cm > 10 * TOL_L2 and gmc > 1e3 * TOL_G, f"parameters L2 {cm:.2e}, gradient {gmc:.2e}")
            else:
                check("parity2: both runs finished", False)
        if "ranks" in want:
            rcs = w2[2] if ok_run(w2) else []
            check("ranks: Muon under DDP, every rank bit-identical at every checkpoint",
                  len(rcs) == STEPS // 6 and all(r["identical"] for r in rcs), f"{[r['identical'] for r in rcs]}")
            n_bufs = len(ck("p2_ddp_w2")["opt"].get("parts", {}).get("muon", {}).get("bufs", [])) if ok_run(w2) else 0
            check("ranks: the digest covers parameters, buffers and Muon's buffers",
                  bool(rcs) and n_bufs > 0 and rcs[0]["tensors"] > n_bufs)
        if "accum" in want:
            w2a1 = go("track4_sdmonly_train_ddp.py", "p2_ddp_w2_a1", recipe + ["--tokens", str(G * 16 * STEPS), "--B", "4",
                                                                              "--world", "2", "--count-allreduce"], dump=[0])
            if ok_run(w2) and ok_run(w2a1):
                c2 = [r["allreduce_calls"] for r in w2[1]]
                c1 = [r["allreduce_calls"] for r in w2a1[1]]
                check("accum: one all-reduce round per step at accum 2 (as many bucket calls as at accum 1)",
                      len(c2) == STEPS and c2 == c1 and c2[0] > 0, f"accum 2 {c2[:3]}..., accum 1 {c1[:3]}...")
                r8a2 = go("track4_sdmonly_train.py", "p2_single_b8_acc2", recipe + ["--tokens", str(G * 16 * STEPS), "--B", "8",
                                                                                    "--accum", "2"], dump=[0])
                ga = gdiff("p2_single_b8_acc2", "p2_ddp_w2_a1", 0)
                la = maxdiff(r8a2[1], w2a1[1], STEPS) if ok_run(r8a2) else float("inf")
                check("accum: world 2 x B 4 x accum 1 equals one device at B 8 x accum 2 (gradient, 12 losses)",
                      ga <= TOL_G and la <= TOL_LOSS, f"gradient {ga:.1e}, losses {la:.1e}")
            else:
                check("accum: runs finished", False)

        # ---- resume: 2 -> 1 and 1 -> 2 from the keep checkpoint at step 6
        if "resume" in want and both:
            kname = f"keep_{G * 16 * KEEP}.pt"
            for src, dst, script, flags, label in (
                    ("p2_ddp_w2", "res_21", "track4_sdmonly_train.py", ["--B", "8", "--accum", "4"], "2 -> 1"),
                    ("p2_single_b8", "res_12", "track4_sdmonly_train_ddp.py", ["--B", "4", "--accum", "2", "--world", "2"], "1 -> 2")):
                os.makedirs(os.path.join(tmp, "ck", dst), exist_ok=True)
                shutil.copyfile(os.path.join(tmp, "ck", src, kname), os.path.join(tmp, "ck", dst, "last.pt"))
                rr = go(script, dst, recipe + ["--tokens", str(G * 16 * STEPS)] + flags, fresh=False, dump=[KEEP])
                if not ok_run(rr):
                    check(f"resume {label}: the run finished", False)
                    continue
                res, st, _ = rr
                gk = gdiff(src, dst, KEEP)
                check(f"resume {label}: the first gradient after the move equals the unmoved run's (same state)",
                      res["resumed_from_step"] == KEEP and gk <= TOL_G, f"from step {res['resumed_from_step']}, max relative diff {gk:.1e}")
                ref_tail = [x for x in r8[1] if x["step"] > KEEP]
                dl = maxdiff(ref_tail, st, STEPS - KEEP)
                db = abs(res["val_test"]["bpb"] - r8[0]["val_test"]["bpb"])
                dp = sd_diff(ck(dst), ck("p2_single_b8"), "l2")
                check(f"resume {label}: steps {KEEP + 1}..{STEPS} equal the straight 1-rank run (losses, TEST bpb, parameters)",
                      dl <= TOL_LOSS and db <= TOL_BPB and dp <= TOL_L2, f"losses {dl:.1e}, bpb {db:.1e}, parameters L2 {dp:.1e}")
        elif "resume" in want:
            check("resume: source runs finished", False)

        # ---- fork: the cooldown fork from the keep checkpoint (the constant phase of wsd)
        if "fork" in want and both:
            kpath = os.path.join(tmp, "ck", "p2_ddp_w2", f"keep_{G * 16 * KEEP}.pt")
            fk = ["--init-ckpt", kpath, "--cooldown-tokens", str(G * 16 * 4)]
            fs = go("track4_sdmonly_train.py", "fork_single", recipe + tok[:2] + fk + ["--B", "8", "--accum", "4"], dump=[KEEP])
            fd = go("track4_sdmonly_train_ddp.py", "fork_ddp", recipe + tok[:2] + fk + ["--B", "4", "--accum", "2", "--world", "2"],
                    dump=[KEEP])
            if ok_run(fs) and ok_run(fd):
                dl = maxdiff(fs[1], fd[1], 4)
                gf = gdiff("fork_single", "fork_ddp", KEEP)
                check("fork: the DDP cooldown fork runs steps 7..10 like the single trainer's fork (gradient, losses)",
                      [x["step"] for x in fd[1]] == [7, 8, 9, 10] and dl <= TOL_LOSS and gf <= TOL_G,
                      f"losses {dl:.1e}, gradient {gf:.1e}")
                check("fork: the learning rate falls linearly to zero on both",
                      [x["lr"] for x in fs[1]] == [x["lr"] for x in fd[1]] and fd[1][-1]["lr"] > 0
                      and abs(fd[1][0]["lr"] / fd[1][-1]["lr"] - 4.0) < 1e-9, f"{[x['lr'] for x in fd[1]]}")
                df = sd_diff(ck("fork_single"), ck("fork_ddp"), "l2")
                check("fork: final parameters equal", df <= TOL_L2, f"max relative L2 diff {df:.1e}")
            else:
                check("fork: both forks finished", False)

    # ---- length: a resume on a train file of another length is refused; --data-n reads the recorded prefix
    if "length" in want:
        env_long = {**base_env, "SDMLLM_DATA": data_long}
        r1 = go("track4_sdmonly_train.py", "len_src", recipe + ["--tokens", str(G * 16 * 6), "--B", "8", "--accum", "4"])
        r2 = go("track4_sdmonly_train_ddp.py", "len_src_ddp", recipe + ["--tokens", str(G * 16 * 6), "--B", "4", "--accum", "2",
                                                                       "--world", "2"])
        more = ["--tokens", str(G * 16 * 8)]
        if ok_run(r1) and ok_run(r2):
            for script, run, flags in (("track4_sdmonly_train.py", "len_src_ddp", ["--B", "8", "--accum", "4"]),
                                       ("track4_sdmonly_train_ddp.py", "len_src", ["--B", "4", "--accum", "2", "--world", "2"])):
                r = go(script, run, recipe + more + flags, expect_fail=True, env=env_long, fresh=False)
                check(f"length: {script} refuses a resume on a longer train file",
                      r.returncode != 0 and "resume refused" in (r.stdout + r.stderr) and "train_n" in (r.stdout + r.stderr))
            r = go("track4_sdmonly_train.py", "len_src_ddp", recipe + more + ["--B", "4", "--accum", "2"], expect_fail=True, fresh=False)
            check("length: a resume at another global batch is refused",
                  r.returncode != 0 and "resume refused" in (r.stdout + r.stderr) and "B_global" in (r.stdout + r.stderr))
            ok1 = go("track4_sdmonly_train.py", "len_src_ddp", recipe + more + ["--B", "8", "--accum", "4", "--data-n", "6000"],
                     env=env_long, fresh=False)
            check("length: --data-n <recorded n> on the longer file resumes", ok_run(ok1) and ok1[0]["resumed_from_step"] == 6)
            bad = os.path.join(tmp, "data_other")
            os.makedirs(bad, exist_ok=True)
            other = tr.copy()
            other[100] += 1
            other.tofile(os.path.join(bad, "train.u32"))
            va.tofile(os.path.join(bad, "val.u32"))
            tbytes.tofile(os.path.join(bad, "token_bytes.i32"))
            r = go("track4_sdmonly_train_ddp.py", "len_src", recipe + more + ["--B", "4", "--accum", "2", "--world", "2"],
                   expect_fail=True, env={**base_env, "SDMLLM_DATA": bad}, fresh=False)
            check("length: a train file of the same length but other content is refused (fingerprint)",
                  r.returncode != 0 and "fingerprint" in (r.stdout + r.stderr))
        else:
            check("length: source runs finished", False)

    # ---- rows: rows sync = dense sync; chunked = plain; lr scale (plain AdamW)
    if "rows" in want:
        small = ["--arm", "sdmonly", "--d", "16", "--d-a", "16", "--n-sub", "16", "--k", "4", "--hops", "2", "--lr", "3e-3",
                 "--tokens", str(8 * 16 * 12), "--B", "4", "--world", "2"]
        dn = go("track4_sdmonly_train_ddp.py", "ddp_w2_dense", small + ["--table-sync", "dense"])
        rw = go("track4_sdmonly_train_ddp.py", "ddp_w2_rows", small + ["--table-sync", "rows"])
        if ok_run(dn) and ok_run(rw):
            dl = maxdiff(dn[1], rw[1], 12)
            check("rows: rows sync equals dense sync (losses, TEST bpb)",
                  dl <= 1e-4 and abs(dn[0]["val_test"]["bpb"] - rw[0]["val_test"]["bpb"]) < 1e-6, f"max abs diff {dl:.1e}")
            check("rows: rows sync sends fewer bytes than the dense tables",
                  0 < rw[1][-1]["table_sync_mb"] * 1e6 < rw[0]["table_bytes_dense"] and 0 < rw[1][-1]["table_rows_frac"] < 1)
            ch = go("track4_sdmonly_train_ddp.py", "ddp_w2_chunked", small + ["--loss", "chunked", "--ce-chunk", "24",
                                                                              "--no-loss-compile"])
            check("rows: the chunked loss under DDP equals the plain loss",
                  ok_run(ch) and abs(dn[0]["val_test"]["bpb"] - ch[0]["val_test"]["bpb"]) < 1e-4)
            lr2 = go("track4_sdmonly_train_ddp.py", "ddp_w2_sqrt", small + ["--lr-scale", "sqrt", "--lr-ref-batch", "4"])
            check("rows: --lr-scale sqrt: peak = lr x sqrt(global batch / ref)",
                  ok_run(lr2) and abs(lr2[0]["lr"] - 3e-3 * 2 ** 0.5) < 1e-12 and dn[0]["lr"] == 3e-3)
        else:
            check("rows: runs finished", False)

    # ---- bench: the recipe's step on random tokens
    if "bench" in want:
        cmd = [sys.executable, os.path.join(HERE, "track4_sdmonly_train_ddp.py"), "--bench", "3", "--world", "2",
               "--T", "16", "--cpu", "--seed", "3"] + recipe + ["--B", "4", "--accum", "2"]
        r = subprocess.run(cmd, env=base_env, capture_output=True, text=True)
        out = next((json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith('{"event": "bench"')), None)
        if out is None:
            print(r.stdout[-2000:], r.stderr[-2000:])
        check("bench: runs the recipe (Muon, accum, chunked) and reports tokens a second per rank and in total",
              out is not None and out["body_opt"] == "muon" and out["accum"] == 2 and out["loss_path"] == "chunked"
              and out["tok_per_s_total"] > 0 and len(out["tok_per_s_per_rank"]) == 2 and out["ranks_identical"] is True
              and out["single_device_equivalent"] == "--B 8 --accum 4", json.dumps(out)[:160] if out else "")

    # ---- summary
    if "summary" in want:
        import track4_sdmonly_summary as SM
        runs = SM.load_runs(os.path.join(tmp, "runs"), "p2_")
        rows = SM.table(runs, os.path.join(tmp, "runs"), os.path.join(tmp, "ck"), "p2_single_b8") if runs else []
        by = {r["run"]: r for r in rows}
        check("summary: track4_sdmonly_summary reads the DDP results",
              "p2_ddp_w2" in by and by["p2_ddp_w2"]["tps"] and by["p2_ddp_w2"]["paired"] is not None)
    if os.path.isdir(grads):
        shutil.rmtree(grads)  # the gradient files are large (two embedding tables a step); the rest stays for reading
    print(f"{sum(ok)} of {len(ok)} checks pass   (files in {tmp})")
    print("NEXT -> python3 track4_sdmonly_train_ddp.py --help")
    return all(ok) and bool(ok)


def main():
    if "--selftest" in sys.argv[1:]:
        own, _ = own_parser().parse_known_args(sys.argv[1:])
        sys.exit(0 if selftest(own.only) else 1)
    launch()


if __name__ == "__main__":
    main()
