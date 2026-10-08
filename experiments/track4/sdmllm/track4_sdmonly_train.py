"""SDMONLY trainer: the S0 trainer's loop, with the SDM-only arms, the run-time SDM (onesdm) arms, a WSD schedule option
and their ablations.

<claudes_code_comments>
** Function List **
wsd_lr(decay_frac) - builds lr(step): linear warmup, constant, then a linear decay to zero over the last fraction
cooldown_lr(step0, peak) - builds lr(step): linear from peak at step0 to zero at the last step (the cooldown fork)
parse() - the S0 trainer's flags plus the SDM-only ones
extra_cfg(a) - the SDM-only build settings that ride with the S0 cfg
opt_info(a) - the opt-in optimiser and fork settings, empty on the default path
fades_for(spec, heads, p) - one fade per memory head (the README's four, repeated for 8, 12 ... heads)
mlp_width(a) - --mlp-f, or 8/3 d rounded up to 16
onesdm_cfg(a) - the full build_onesdm cfg of an onesdm arm (rides in every checkpoint as cfg["onesdm"])
eager_body(model, a) - keep an onesdm body out of the compiled chunked loss unless --compile-body
build_text_model(arm, V, cfg) - arm + a checkpoint's cfg -> the model (this trainer and the fine-tune tool)
onesdm_group(name, p, body_opt) - THE SPLIT RULE: Muon for body matrices, AdamW for tables, gains, table and head
onesdm_assign_groups(model, body_opt) - {parameter name: group label} under that rule
build_onesdm_optimizer(model, lr, ...) - AdamW groups (+ Muon) for the onesdm arms, in the Composite the loop speaks
onesdm_flops_per_token(model, T) - analytic forward FLOPs of the chunked dense form
text_flops_per_token(model, T) - dispatch: onesdm, else the SDM-only / S0 rule
data_vocab() / set_vocab(V) - the vocabulary from token_bytes.i32; set it in S0 and the paired per-window scorer
check_ids(arr, V, name) - refuse a shard with a token id >= V
s0_cfg(a) - the cfg S0.train would build, plus the build settings (for the bench and the size report)
make_optimizer(a, model, lr) - the onesdm split or track4_sdmonly_optim.build_optimizer
bench(a, V) - --bench-steps: real training steps on random tokens at the real shape; tokens/s and peak memory
param_report(a, V) - --param-report: embedding and non-embedding weights by part, built on the meta device
with_stop(lr_fn, a, cap) - --stop-after-steps: write last.pt and stop (resume tests)
selftest() - CPU: every onesdm arm trains, vocab from data, TEST, resume, the split, the reset at BOS, bench, report
score_ablations(a, res_path) - reload the final checkpoint, score zero_read and shuffle_keys, count locations used
_Ns - a read-through stand-in for a module with a few names replaced (used for S0's `torch` name only)
prepare_cooldown(a) - copy the source checkpoint into the fork's folder, set the step count and the schedule
with_keep(lr_fn, a, cap) - wrap lr(step) so named keep-checkpoints are written at the --keep-at token counts
fingerprint(arr) - sha256 of the first and last million tokens of a train array (16 hex digits)
data_record(a, arr, B_global, accum_global) - what fixes a run's batches: train length, fingerprint, global batch, T
check_resume_data(old, new, path) - refuse a resume whose batches would differ from the checkpoint's
peek_data(path) - the data record inside a checkpoint's cfg, read without loading the tensors
train_array(a) - the train shard, cut to --data-n tokens when that flag is given
grad_dump_spec() - SDMONLY_DUMP_GRADS=DIR[@s1,s2]: where (and at which steps) to save the pre-clip gradients
main(argv) - put the S0 names back on return; _main: vocabulary, the tools, then patch builder/schedule/optimiser and train

** Technical Review **
- This file adds no training loop. It imports track4_sdmllm_train_one_arm (the S0 trainer), replaces three names
  in that module (build, flops_per_token and, for --sched wsd, lr_at) and calls its train(). So batches, the
  optimiser groups, bf16 autocast, the chunked loss, checkpoints, resume, the stamps and the bpb scoring are the
  S0 trainer's, byte for byte, and a run here is comparable with every earlier SDMLLM run.
- Arms: sdmonly (sparse reads only), sdmonly_dense (each read replaced by a SwiGLU of equal FLOPs),
  sdmonly_none (no reads). The model is in track4_sdmonly_models.py.
- Schedules: cosine (S0: 2% warmup, cosine to 10% of peak) or wsd (2% warmup, constant, linear to zero over the
  last --wsd-decay fraction; Hagele et al. 2024, the schedule SmolLM2 and SmolLM3 use).
- Defaults are the SDMLLMSTORE fixes carried over: 8 back tokens, 5 moving averages, softness 0.25, the query
  gradient stopped, store learning rate x3, no weight decay on the store.
- After train() returns, the final checkpoint is reloaded and scored with every store returning zero
  (zero_read) and with every value table permuted (shuffle_keys); the fraction of locations any test token
  woke is counted per hop. These go into the result json under the same keys the S0 trainer uses.
- OPT-IN, lane SPARSEROWS (none of this runs unless its flag is given; with no flag main() does what it did before):
  * --store-opt sparse: only the woken rows of each value table are updated (RowAdam in track4_sdmonly_optim.py).
    A row not woken in a step gets no update, no momentum decay and no weight decay; dense AdamW gives it all three.
  * --body-opt muon: Muon for the 2-D body matrices (wx, wq[h], readout, the dense control's mlps) at --muon-lr;
    AdamW stays for the embedding/head, the norms and the store.
  * The S0 loop builds its optimiser with `torch.optim.AdamW` and clips with `torch.nn.utils.clip_grad_norm_`.
    For these two flags (and --keep-at) the S0 module's own name `torch` is replaced, for the length of train(),
    by a stand-in that reads through to torch except for those two names. torch itself is never modified.
  * --init-ckpt CKPT --cooldown-tokens N (the cooldown fork): start from a checkpoint of the constant phase of a
    wsd run, with its model AND optimiser state, and decay the learning rate linearly to zero over N tokens. The
    fork continues the source's step count, so it reads the batches the source would have read next. The peak is
    the learning rate stored in the checkpoint. The fork needs its own --run name and the source's model flags.
  * --keep-at T1,T2: the run also writes checkpoints/<run>/keep_<T>.pt when it has seen T tokens (rounded up to a
    whole step), in the format of last.pt, so one constant run can be decayed from several token counts.
- --mix-impl fast (lane MIXERFAST, default dense): the SDM mixer computed from sorted index lists
  (SdmMixFast, track4_sdmonly_mixfast.py); same parameters and maths, a per-token cost that does not grow with the
  window or with n_sub^2. The setting rides in cfg as "mix_impl".
- RESUME DATA CHECK (lane DDPRECIPE): a batch is drawn from (seed, step, train length n, global batch, T). Every
  checkpoint's cfg carries cfg["data"] = {train_name, train_n, fingerprint, B_global, accum_global, micro_B, T}.
  A resume (and the cooldown fork) refuses when train_n, the fingerprint, B_global or T differ from this run, so a
  run moved to another machine (a rented box -> the Spark, either trainer) never silently reads other batches.
  A changed micro-batch (accum) is allowed and printed: it changes only how batch-norm statistics are grouped.
  --data-n N draws offsets over the first N tokens of a longer file; the fingerprint must still match. A checkpoint
  written before this record existed resumes with a printed NOTE and no check. track4_sdmonly_train_ddp.py writes
  the same record and runs the same check, so a checkpoint moves between the two trainers in both directions.
- SDMONLY_DUMP_GRADS=DIR[@0,6] (environment, for tests; off when unset): before the clip of each listed step
  (every step when no list), the gradient of every parameter is saved to DIR/grad_<step>.pt by name. The DDP trainer
  saves the same file on rank 0 after DDP's average, so the two trainers' gradients can be compared step by step.
- Output folders: SDMLLM_RUNS (default runs_sdmonly/) and SDMLLM_CKPT (default checkpoints/).
- THE ONESDM ARMS (lane TEXTWIRE): --arm onesdm | onesdm_allsdm | onesdm_allsdm_big | plain (memory off) |
  yardstick_transformer, built by track4_onesdm_models.build_onesdm from cfg["onesdm"] (onesdm_cfg). Width is --d;
  --layers, --mlp-f, --mem-heads/--mem-n-sub/--mem-d-a/--mem-k, --fades, --chunk, --addr-bias, --learn-decay,
  --read-mode and --sdm-* default to ONESDM_README.md. The batches, the schedule (cosine or --sched wsd), the
  checkpoint with optimiser state, resume and TEST are the S0 loop's, unchanged. --body-opt defaults to muon for
  these arms (adamw for the old ones). THE SPLIT (onesdm_group): Muon takes every nn.Linear weight of the body with
  both sides >= 16; the memory sub-keys and trained-SDM value tables go to AdamW's store group (--store-lr-mult,
  --store-wd); the token table, norm gains, biases, address biases, scales and fades go to AdamW without decay; a thin
  Linear (the write gate, heads x d) goes to AdamW with decay 0.1. The yardstick uses the same rule.
- RUN-TIME MEMORY AND WINDOWS: every training window starts with an EMPTY run-time memory (each forward builds its
  state from zero; nothing crosses windows or batch rows). --rt-reset-at-bos also empties it at every BOS (id 0)
  inside a window, so each document reads only its own writes. It is recorded in cfg["onesdm"]["reset_at_bos"], so a
  fine-tune inherits it.
- THE VOCABULARY COMES FROM THE DATA: V = the number of entries in $SDMLLM_DATA/token_bytes.i32 (129,280 for the
  DeepSeek shards, 32,768 for the trained BPE). It is set in S0.V and the paired scorer's copy before anything is
  built; the first and last million tokens of train and val are checked to be < V. Nothing hard-codes it.
- hidden(): the chunked loss and the fine-tune tool form logits as model.hidden(x) @ emb^T, so OneSdmLM.hidden returns
  the final normalised state (track4_onesdm_models). With --loss chunked an onesdm body runs eager inside the compiled
  loss (eager_body) unless --compile-body: inductor's CPU code for it fails to compile here (torch 2.8) and it is
  untested on CUDA.
- --bench-steps N: N real steps (forward, backward, clip, optimiser) at the real arm, shape, B, T, accum and loss path
  on random tokens; prints tokens/s (median and best, the first 2 steps untimed), CUDA peak memory and the token count
  a day at the median rate (a projection, labelled). --param-report builds on the meta device and prints the counts.
  Both take V from the data, or --vocab N when no data is present. --stop-after-steps N checkpoints and stops.
Docs: TRAINING_SDMONLY_HOW_WE_TRAIN.md
</claudes_code_comments>
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SDMLLM_RUNS", os.path.join(HERE, "runs_sdmonly"))
import track4_sdmllm_train_one_arm as S0  # noqa: E402
import track4_sdmonly_models as M  # noqa: E402
import track4_sdmonly_optim as O  # noqa: E402
import track4_onesdm_models as OS  # noqa: E402  (lane TEXTWIRE: the run-time SDM models)

_LOAD_SPLIT = S0.load_split  # the unpatched loader (main() may patch S0.load_split for --data-n)

# lane TEXTWIRE: the run-time SDM arms of track4_onesdm_models.build_onesdm. plain = the same stack, memory off;
# yardstick_transformer = its QwenLM yardstick, sized by the same --d / --layers / --mlp-f
ONESDM_ARMS = ["onesdm", "onesdm_allsdm", "onesdm_allsdm_big", "plain", "yardstick_transformer"]
RESET_ARMS = ("onesdm", "onesdm_allsdm", "onesdm_allsdm_big")  # the arms with a run-time memory to reset
ARMS = ["sdmonly", "sdmonly_dense", "sdmonly_none", "qwen"] + ONESDM_ARMS  # qwen: the 4-layer transformer yardstick
README_FADES = (0.0, 0.8, 0.99, 1.0)  # ONESDM_README.md: one fade per head, 4 heads


def wsd_lr(decay_frac):
    def lr_at(step, total, peak, warm, floor=0.0):
        if step < warm:
            return peak * (step + 1) / warm
        start = int(total * (1.0 - decay_frac))
        if step < start:
            return peak
        return peak * max(0.0, (total - step) / max(1, total - start))
    return lr_at


def cooldown_lr(step0, peak):
    def lr_at(step, total, _peak, warm, floor=0.0):
        return peak * max(0.0, (total - step) / max(1, total - step0))
    return lr_at


def parse(argv=None):
    p = argparse.ArgumentParser(description="SDMONLY: train one SDM-only arm. Example: python3 %(prog)s --arm sdmonly "
                                            "--tokens 20000000 --hops 4 --n-sub 256 --run sdmonly_center")
    p.add_argument("--arm", default="sdmonly", choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", type=int, default=20_000_000)
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-batches per step (memory only)")
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--d-a", type=int, default=256, help="address width per head (split in two halves)")
    p.add_argument("--n-sub", type=int, default=256, help="sub-keys per half; locations = n_sub squared")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--hops", type=int, default=4)
    p.add_argument("--softness", type=float, default=0.25)
    p.add_argument("--share-store", action="store_true", help="one value table for every hop")
    p.add_argument("--readout-f", type=int, default=0, help="SwiGLU readout width after the hops (0 = none)")
    p.add_argument("--untie", action="store_true", help="a separate input token table (the head stays tied to emb)")
    p.add_argument("--hop-mlp", type=int, default=0, help="SwiGLU width after every read (0 = none)")
    p.add_argument("--ng-rows", type=int, default=0, help="rows per exact n-gram hash table read before hop 1 (0 = none)")
    p.add_argument("--ng-orders", default="2,3", help="context lengths of the n-gram hash tables")
    p.add_argument("--conj", type=int, default=0, help="2 or 3: products of the last 2 or 3 token vectors as features")
    p.add_argument("--rt-write", type=int, default=0, help="run-time write: locations each position writes to (0 = off)")
    p.add_argument("--rt-buckets", type=int, default=4096)
    p.add_argument("--qwen-layers", type=int, default=4, help="the transformer yardstick's depth (default 4)")
    p.add_argument("--qwen-f", type=int, default=0, help="its MLP width (0 = 688, the width-256 shape; use 8/3 d to scale)")
    p.add_argument("--mix-layers", type=int, default=0, help="SDM mixing layers (run-time Kanerva memory), each + a dense layer")
    p.add_argument("--mix-heads", type=int, default=4)
    p.add_argument("--mix-n-sub", type=int, default=64, help="hard-locations per mixer head = n_sub squared")
    p.add_argument("--mix-d-a", type=int, default=64)
    p.add_argument("--mix-k", type=int, default=32)
    p.add_argument("--mix-decays", default="1.0,0.999,0.99,0.9", help="one fade per head (1.0 = never)")
    p.add_argument("--mix-f", type=int, default=0, help="dense layer width after each mixer (0 = 4 d)")
    p.add_argument("--mix-chunk", type=int, default=256)
    p.add_argument("--mix-impl", default="dense", choices=["dense", "fast"],
                   help="fast = the same mixer from sorted index lists (lane MIXERFAST); dense = chunked one-hot products")
    p.add_argument("--graded", action="store_true", help="a location's weight is its score above the cut-off (k-sparse ReLU)")
    p.add_argument("--rt-shuffle", action="store_true", help="control: shuffle the written vectors across positions")
    p.add_argument("--dense-f", type=int, default=None, help="sdmonly_dense: SwiGLU width (default: FLOPs of one read)")
    p.add_argument("--n-back", type=int, default=8)
    p.add_argument("--decays", default="0.5,0.8,0.9,0.97,0.99")
    p.add_argument("--qgrad", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--store-lr-mult", type=float, default=3.0)
    p.add_argument("--store-wd", type=float, default=0.0)
    p.add_argument("--sched", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--wsd-decay", type=float, default=0.2)
    p.add_argument("--train-name", default="train")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--loss", default="plain", choices=["plain", "chunked"])
    p.add_argument("--ce-chunk", type=int, default=2048)
    p.add_argument("--init", default=None)
    p.add_argument("--extra-val", action="append", default=[])
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--quick-val", type=int, default=64)
    p.add_argument("--val-windows", type=int, default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--cpu", action="store_true")
    g = p.add_argument_group("opt-in (lane SPARSEROWS); the defaults leave the run exactly as it was")
    g.add_argument("--store-opt", default="dense", choices=["dense", "sparse"],
                   help="sparse = update only the woken rows of each value table (row-wise Adam)")
    g.add_argument("--body-opt", default=None, choices=["adamw", "muon"],
                   help="muon = Muon for the 2-D body matrices (default: muon for the onesdm arms, adamw for the rest)")
    g.add_argument("--muon-lr", type=float, default=0.02, help="Muon's peak learning rate (it follows the run's schedule)")
    g.add_argument("--muon-wd", type=float, default=0.0)
    g.add_argument("--init-ckpt", default=None, help="cooldown fork: a checkpoint (model and optimiser) from a constant phase")
    g.add_argument("--cooldown-tokens", type=int, default=0, help="cooldown fork: linear decay to zero over this many tokens")
    g.add_argument("--keep-at", default="", help="comma list of token counts at which to write keep_<tokens>.pt")
    p.add_argument("--data-n", type=int, default=None,
                   help="draw batch offsets over the first N tokens of the train file (to resume on a longer file)")
    g = p.add_argument_group("the onesdm arms (lane TEXTWIRE; build_onesdm in track4_onesdm_models.py, defaults from "
                             "ONESDM_README.md; the width is --d)")
    g.add_argument("--layers", type=int, default=4, help="residual layers (each: run-time memory, then MLP or trained SDM)")
    g.add_argument("--mlp-f", type=int, default=0, help="SwiGLU width (0 = 8/3 d rounded up to 16: 688 at d 256, 2,048 at 768)")
    g.add_argument("--mem-heads", type=int, default=4, help="run-time memory heads per layer")
    g.add_argument("--mem-n-sub", type=int, default=32, help="sub-keys per half; slots per head = n_sub squared (32: 1,024)")
    g.add_argument("--mem-d-a", type=int, default=32, help="address width per head")
    g.add_argument("--mem-k", type=int, default=16, help="slots each write and each read wakes per head")
    g.add_argument("--fades", default=None, help="one fade per memory head (default 0,0.8,0.99,1, repeated for 8, 12 ... heads)")
    g.add_argument("--chunk", type=int, default=128, help="tokens per chunk of the parallel write-then-read")
    g.add_argument("--addr-bias", type=float, default=0.0, help="init scale of the shared address bias per head")
    g.add_argument("--learn-decay", action="store_true", help="learned fades (default fixed)")
    g.add_argument("--read-mode", default="mean", choices=["mean", "sum"])
    g.add_argument("--rt-reset-at-bos", action="store_true",
                   help="empty every run-time memory at each BOS token inside a window (default off: a window starts "
                        "empty and its documents share the memory)")
    g.add_argument("--sdm-n-sub", type=int, default=0, help="trained SDM table: sub-keys per half (0 = matched to the MLP's weights)")
    g.add_argument("--sdm-heads", type=int, default=4)
    g.add_argument("--sdm-d-a", type=int, default=0, help="trained SDM address width per head (0 = min(32, d / heads))")
    g.add_argument("--sdm-k", type=int, default=32, help="trained SDM slots read per token per head")
    g.add_argument("--sdm-big-mult", type=int, default=4, help="onesdm_allsdm_big: n_sub x this (4 = 16x the slots, same k)")
    g.add_argument("--yard-heads", type=int, default=4, help="yardstick_transformer: attention heads")
    g.add_argument("--yard-kv", type=int, default=2, help="yardstick_transformer: key-value heads")
    g.add_argument("--compile-body", action="store_true",
                   help="--loss chunked compiles the loss; by default an onesdm body runs eager inside it (its torch.compile "
                        "is untested on CUDA and fails on the CPU here); this flag compiles the body too")
    g = p.add_argument_group("tools (lane TEXTWIRE)")
    g.add_argument("--bench-steps", type=int, default=0,
                   help="run N training steps on random tokens at this B, T, accum and shape; print tokens/s and peak memory")
    g.add_argument("--param-report", action="store_true", help="print embedding and non-embedding weights of this arm and shape, exit")
    g.add_argument("--vocab", type=int, default=0,
                   help="--bench-steps / --param-report only: vocabulary size when no data is present (training always "
                        "takes it from token_bytes.i32)")
    g.add_argument("--stop-after-steps", type=int, default=0, help="checkpoint and stop after N steps of this session (resume tests)")
    g.add_argument("--selftest", action="store_true", help="CPU checks of the onesdm wiring (about a minute)")
    a = p.parse_args(argv)
    if a.selftest:
        return a
    onesdm = a.arm in ONESDM_ARMS
    a.body_opt = a.body_opt or ("muon" if onesdm else "adamw")
    if a.rt_reset_at_bos and a.arm not in RESET_ARMS:
        p.error(f"--rt-reset-at-bos needs a run-time memory: one of {RESET_ARMS}")
    if a.vocab and not (a.bench_steps or a.param_report):
        p.error("--vocab is for --bench-steps and --param-report; training takes the vocabulary from token_bytes.i32")
    if onesdm:
        a.fades_list = fades_for(a.fades, a.mem_heads, p)
    if bool(a.init_ckpt) != bool(a.cooldown_tokens):
        p.error("--init-ckpt and --cooldown-tokens go together")
    if a.init_ckpt and not a.run:
        p.error("the cooldown fork needs its own --run name")
    if a.store_opt == "sparse" and a.arm != "sdmonly":
        p.error("--store-opt sparse needs --arm sdmonly (the other arms have no store)")
    a.keep_at = sorted(int(x) for x in a.keep_at.split(",") if x)
    # fields the S0 train() reads for its cfg record; unused by the SDM-only arms
    a.M, a.M_big, a.n_layer, a.f, a.value_init, a.ngram_orders = a.n_sub * a.n_sub, 0, 0, a.readout_f, "zero", "1"
    if onesdm:  # the S0 cfg record carries the onesdm depth and MLP width; the full build is cfg["onesdm"]
        a.n_layer, a.f = a.layers, mlp_width(a)
    a.run = a.run or f"{a.arm}_s{a.seed}_{a.tokens // 10**6}M"
    return a


def extra_cfg(a):
    return {"d_a": a.d_a, "n_sub": a.n_sub, "heads": a.heads, "readout_f": a.readout_f, "share_store": a.share_store,
            "dense_f": a.dense_f, "sched": a.sched, "wsd_decay": a.wsd_decay, "untie": a.untie, "hop_mlp": a.hop_mlp,
            "ng_rows": a.ng_rows, "ng_orders": [int(x) for x in a.ng_orders.split(",")], "conj": a.conj,
            "rt_write": a.rt_write, "rt_buckets": a.rt_buckets, "rt_shuffle": a.rt_shuffle, "graded": a.graded,
            "mix_layers": a.mix_layers, "mix_heads": a.mix_heads, "mix_n_sub": a.mix_n_sub, "mix_d_a": a.mix_d_a,
            "mix_k": a.mix_k, "mix_decays": [float(v) for v in a.mix_decays.split(",")], "mix_f": a.mix_f,
            "mix_chunk": a.mix_chunk, "mix_impl": a.mix_impl}


def opt_info(a):
    out = {}
    if a.store_opt != "dense" or a.body_opt != "adamw":
        out.update({"store_opt": a.store_opt, "body_opt": a.body_opt})
        if a.body_opt == "muon":
            out.update({"muon_lr": a.muon_lr, "muon_wd": a.muon_wd})
    if a.init_ckpt:
        out.update({"init_ckpt": a.init_ckpt, "cooldown_tokens": a.cooldown_tokens, "cooldown_from_step": a.cooldown_step0,
                    "cooldown_peak_lr": a.cooldown_peak})
    if a.keep_at:
        out["keep_at"] = a.keep_at
    return out


# ---------------------------------------------------------------- the onesdm arms (lane TEXTWIRE)
def fades_for(spec, heads, p=None):
    """One fade per memory head. Default: the README's 0, 0.8, 0.99, 1 (4 heads), repeated for 8, 12 ... heads."""
    if spec:
        f = [float(x) for x in spec.split(",")]
    elif heads % len(README_FADES) == 0:
        f = list(README_FADES) * (heads // len(README_FADES))
    else:
        f = []
    if len(f) != heads:
        msg = f"--fades needs one fade per memory head ({heads}); got {spec or 'the 4 defaults'}"
        if p is not None:
            p.error(msg)
        raise SystemExit(msg)
    return f


def mlp_width(a):
    """--mlp-f, or 8/3 d rounded up to a multiple of 16 (688 at d 256, 1,376 at 512, 2,048 at 768)."""
    return a.mlp_f or -(-a.d // 6) * 16


def onesdm_cfg(a):
    """The full build_onesdm cfg of an onesdm arm. It rides in every checkpoint as cfg["onesdm"], so the fine-tune
    tool and any reader rebuild the model from the checkpoint alone. The S0 cfg keys k, decays and the rest are the
    sdmonly store's and are not read by these arms."""
    return {"d": a.d, "layers": a.layers, "mlp_f": mlp_width(a), "heads": a.mem_heads, "n_sub": a.mem_n_sub,
            "d_a": a.mem_d_a, "k": a.mem_k, "decays": list(a.fades_list), "chunk": a.chunk, "addr_bias": a.addr_bias,
            "learn_decay": a.learn_decay, "read_mode": a.read_mode, "reset_at_bos": bool(a.rt_reset_at_bos), "bos_id": 0,
            "sdm_n_sub": a.sdm_n_sub, "sdm_heads": a.sdm_heads, "sdm_d_a": a.sdm_d_a, "sdm_k": a.sdm_k,
            "sdm_big_mult": a.sdm_big_mult, "yard_heads": a.yard_heads, "yard_kv": a.yard_kv,
            "max_len": max(a.T, 20000), "tie": True, "seed": a.seed}


def eager_body(model, a):
    """--loss chunked compiles hidden -> loss as one graph. For an onesdm model the body is kept out of that graph
    (torch.compiler.disable on model.hidden) unless --compile-body: inductor's CPU code for it does not compile here
    (torch 2.8, a VecMask cast error), and it has not been tried on CUDA. Only the call is wrapped; no weight changes."""
    if isinstance(model, OS.OneSdmLM) and a.loss == "chunked" and not a.compile_body:
        model.hidden = torch.compiler.disable(model.hidden)
    return model


def build_text_model(arm, V, cfg):
    """arm + the cfg a checkpoint carries -> the model. The trainer's build hook and the fine-tune tool both use it."""
    if arm in ONESDM_ARMS:
        return OS.build_onesdm(arm, V, cfg["onesdm"])
    if arm == "qwen":
        from track4_sdmllm_models import build as build_s0
        return build_s0("qwen", V, cfg)
    return M.build_sdmonly(arm, V, cfg)


def onesdm_group(name, p, body_opt):
    """THE SPLIT RULE for the onesdm arms (and their yardstick):
    - Muon: the 2-D weight matrices of the body, i.e. every nn.Linear weight whose two sides are both at least 16.
    - AdamW, no weight decay: the token table (and a head, if untied), every norm gain, bias, address bias, scale and
      fade (1-D, or 2-D gains such as onorm_w and addr_b, which are not Linear weights).
    - AdamW, the store group (--store-lr-mult, --store-wd): the memory sub-keys (keys1, keys2) and the trained SDM's
      value tables (values). They are tables, not maps, so they stay on AdamW whatever --body-opt says.
    - AdamW with weight decay 0.1: a thin Linear weight (a side under 16, such as the write gate wg: heads x d)."""
    leaf = name.rsplit(".", 1)[-1]
    if name.startswith("emb.") or name.startswith("head."):
        return "adamw_nodecay"
    if leaf in ("keys1", "keys2", "values"):
        return "adamw_store"
    if leaf == "weight" and p.dim() == 2:
        return "muon" if body_opt == "muon" and min(p.shape) >= 16 else "adamw_decay"
    return "adamw_nodecay"


def onesdm_assign_groups(model, body_opt="muon"):
    return {n: onesdm_group(n, p, body_opt) for n, p in model.named_parameters() if p.requires_grad}


def build_onesdm_optimizer(model, lr, store_lr_mult=3.0, store_wd=0.0, body_opt="muon", muon_lr=0.02, muon_wd=0.0,
                           betas=(0.9, 0.95), eps=1e-8):
    """AdamW groups (decay 0.1 / no decay / the store group) and, for body_opt muon, Muon on the body matrices, in the
    Composite the S0 loop and track4_sdmonly_optim already speak (param_groups with lr_mult, state_dict, clip)."""
    by = {g: [] for g in O.GROUPS}
    for n, p in model.named_parameters():
        if p.requires_grad:
            by[onesdm_group(n, p, body_opt)].append(p)
    groups = [{"params": by["adamw_decay"], "weight_decay": 0.1, "lr_mult": 1.0},
              {"params": by["adamw_nodecay"], "weight_decay": 0.0, "lr_mult": 1.0},
              {"params": by["adamw_store"], "weight_decay": store_wd, "lr_mult": store_lr_mult}]
    adamw = torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=betas, eps=eps)
    if body_opt == "adamw":
        return adamw
    parts = {"adamw": adamw}
    if by["muon"]:
        parts["muon"] = O.Muon(by["muon"], lr=muon_lr, lr_mult=muon_lr / lr, weight_decay=muon_wd)
    return O.Composite(parts)


def onesdm_flops_per_token(model, T):
    """Forward FLOPs per token (2 per multiply-add), analytic, of the chunked dense form the trainer runs."""
    d, V = model.d, model.V
    body = 0
    for i in range(model.n_layers):
        m = model.mems[i]
        if m.enabled:
            L = min(T, m.chunk)
            body += 2 * d * (2 * m.h * m.d_a + 2 * d + m.h)  # wk, wq, wv, wo, wg
            body += 2 * 2 * m.h * m.n_sub * m.d_a  # the half scores of the key and of the query
            body += m.h * (2 * L * m.M + 2 * L * m.dh + 4 * m.M * m.dh + 2 * m.M)  # overlap, its read, state read and write
        if model.ffn == "mlp":
            body += 2 * 3 * d * model.mlp_f
        elif model.ffn == "sdm":
            s_ = model.sdms[i]
            body += 2 * d * s_.h * s_.d_a + 2 * s_.h * s_.n_sub * s_.d_a + 2 * s_.h * s_.k * s_.dh + 2 * d * d
    head = 2 * V * d
    return {"head": head, "body": body, "total": head + body}


def text_flops_per_token(model, T):
    if isinstance(model, OS.OneSdmLM):
        return onesdm_flops_per_token(model, T)
    return M.flops_per_token(model, T)  # sdmonly arms, and QwenLM through the S0 rule


def data_vocab():
    """The vocabulary of the data: the number of entries of token_bytes.i32 (one per token id), or None."""
    path = os.path.join(S0.DATA, "token_bytes.i32")
    return os.path.getsize(path) // 4 if os.path.exists(path) else None


def set_vocab(V):
    """Every scorer reads V as a module global at call time; the paired per-window scorer keeps its own copy."""
    import track4_sdmllm_paired_window_comparison as PW
    S0.V = PW.V = int(V)


def check_ids(arr, V, name):
    """Refuse a shard with a token id >= V (its first and last million tokens are read)."""
    m = 1 << 20
    hi = max(int(np.asarray(arr[:m]).max(initial=0)), int(np.asarray(arr[max(0, len(arr) - m):]).max(initial=0)))
    if hi >= V:
        raise SystemExit(f"{name}: token id {hi} >= V {V} (token_bytes.i32). The shard and token_bytes.i32 are from "
                         f"different tokenizers.")


def s0_cfg(a):
    """The cfg dict S0.train builds (same keys), plus the SDM-only and onesdm build settings: for --bench-steps and
    --param-report, which run without S0.train."""
    cfg = {"d": a.d, "T": a.T, "seed": a.seed, "M": a.M, "k": a.k, "hops": a.hops, "softness": a.softness,
           "n_layer": a.n_layer, "M_big": a.M_big, "value_init": a.value_init, "qgrad": a.qgrad,
           "ngram_orders": [int(x) for x in a.ngram_orders.split(",")], "f": a.f, "train_name": a.train_name,
           "n_back": a.n_back, "decays": [float(x) for x in a.decays.split(",")]}
    cfg.update(extra_cfg(a))
    if a.arm in ONESDM_ARMS:
        cfg["onesdm"] = onesdm_cfg(a)
    return cfg


def make_optimizer(a, model, lr, betas=(0.9, 0.95), eps=1e-8):
    if a.arm in ONESDM_ARMS:
        return build_onesdm_optimizer(model, lr, a.store_lr_mult, a.store_wd, a.body_opt, a.muon_lr, a.muon_wd, betas, eps)
    return O.build_optimizer(model, lr, a.store_lr_mult, a.store_wd, a.body_opt, a.store_opt, a.muon_lr, a.muon_wd,
                             n_tokens=a.B * a.T, betas=betas, eps=eps)


def bench(a, V):
    """--bench-steps N: N real training steps (forward, backward, clip, optimiser step) at this arm, shape, B, T and
    accum, on random tokens, with the S0 loop's loss path (plain or the compiled chunked loss) and bf16 autocast on
    CUDA. The first steps (up to 2) are warm-up and not timed. Prints tokens/s and the CUDA peak memory."""
    device = S0.pick_device(a.cpu)
    torch.manual_seed(a.seed)
    model = eager_body(build_text_model(a.arm, V, s0_cfg(a)), a).to(device)
    opt = make_optimizer(a, model, a.lr)
    run_m = torch.compile(model) if a.compile else model
    if a.loss == "chunked":
        from track4_sdmllm24_fused_ce import ce_chunked
        if getattr(model, "store", None) is not None:
            model.store.collect_stats = False
        loss_fn = torch.compile(lambda x, t: ce_chunked(model.hidden(x), model.emb.weight, t, a.ce_chunk))
    else:
        def loss_fn(x, t):
            return torch.nn.functional.cross_entropy(run_m(x).float().reshape(-1, V), t.reshape(-1))
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (torch.mps.synchronize if device.type == "mps" else (lambda: None))
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    before = [p.detach().clone() for p in model.parameters()]
    g = torch.Generator().manual_seed(a.seed)
    times, losses = [], []
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda"))
    for _ in range(a.bench_steps):
        xb = torch.randint(1, V, (a.B, a.T + 1), generator=g).to(device)
        sync()
        t0 = time.time()
        for gr in opt.param_groups:
            gr["lr"] = a.lr * gr.get("lr_mult", 1.0)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for mb in xb.chunk(a.accum):
            with torch.autocast(**amp):
                lm = loss_fn(mb[:, :-1], mb[:, 1:]) / a.accum
            lm.backward()
            tot += float(lm.detach())
        O.clip_grad_norm_(list(model.parameters()), 1.0, opt)
        opt.step()
        sync()
        times.append(time.time() - t0)
        losses.append(tot)
    warm = min(2, max(0, len(times) - 1))
    timed = times[warm:]
    med = float(np.median(timed))
    tps = a.B * a.T
    rec = {"event": "bench", "arm": a.arm, "V": V, "d": a.d, "B": a.B, "T": a.T, "accum": a.accum, "loss_path": a.loss,
           "compile": a.compile, "compile_body": a.compile_body, "body_opt": a.body_opt, "device": str(device), "steps": len(times),
           "warm_steps_not_timed": warm, "sec_per_step_median": round(med, 4), "sec_per_step_min": round(min(timed), 4),
           "tok_per_s_median": round(tps / med, 1), "tok_per_s_best": round(tps / min(timed), 1),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None,
           "projected_tokens_24h_at_median": int(tps / med * 86400), "losses": losses,
           "weights_moved": any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())),
           "params": S0.count_params(model), "stamp": S0.stamp()}
    if a.arm in ONESDM_ARMS:
        rec["onesdm"] = onesdm_cfg(a)
    print(json.dumps(rec), flush=True)
    return rec


def param_report(a, V):
    """--param-report: the arm built on the meta device (no memory is allocated), its weights counted by part."""
    cfg = s0_cfg(a)
    try:
        with torch.device("meta"):
            model = build_text_model(a.arm, V, cfg)
    except Exception:  # a builder that does not run on the meta device is built on the CPU
        model = build_text_model(a.arm, V, cfg)
    emb = model.emb.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    parts = {}
    for n, p in model.named_parameters():
        key = n.split(".")[0]
        if key == "sdms" and n.endswith(".values"):
            key = "sdms.values"
        parts[key] = parts.get(key, 0) + p.numel()
    rec = {"event": "param_report", "arm": a.arm, "V": V, "d": a.d, "non_embedding": total - emb, "embedding": emb,
           "total": total, "by_part": parts}
    if isinstance(model, OS.OneSdmLM):
        rec.update(layers=model.n_layers, mlp_f=model.mlp_f, mem_slots_per_head=model.mems[0].M)
        if model.ffn == "sdm":
            rec.update(sdm_n_sub=model.sdm_n_sub, sdm_slots_per_head=model.sdms[0].M)
    print(json.dumps(rec), flush=True)
    return rec


class _StopEarly(Exception):
    pass


def with_stop(lr_fn, a, cap):
    """--stop-after-steps N: after N steps of this session, write last.pt in the S0 format and stop (resume tests)."""
    st = {}

    def lr_at(step, total, peak, warm, *rest):
        st.setdefault("start", step)
        if step - st["start"] >= a.stop_after_steps and step < total:
            last = os.path.join(ckpt_dir(a), "last.pt")
            torch.save({"model": cap["model"].state_dict(), "opt": cap["opt"].state_dict(), "step": step,
                        "cfg": cap["cfg"], "arm": a.arm}, last + ".tmp")
            os.replace(last + ".tmp", last)
            raise _StopEarly(step)
        return lr_fn(step, total, peak, warm, *rest)
    return lr_at


@torch.no_grad()
def score_ablations(a, res_path):
    device = S0.pick_device(a.cpu)
    ck = torch.load(os.path.join(os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints")), a.run, "last.pt"),
                    map_location=device)
    model = M.build_sdmonly(a.arm, S0.V, {**ck["cfg"], **extra_cfg(a)}).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tb = torch.from_numpy(np.fromfile(os.path.join(S0.DATA, "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
    val = S0.load_split("val")
    starts = S0.val_windows(val, a.T, "test", a.val_windows)
    res = json.load(open(res_path))
    res["sdmonly"] = {**extra_cfg(a), "store_params": M.count_store(model)}
    for ab in ("zero_read", "shuffle_keys"):
        model.store.set_ablate(ab)
        res[f"val_test_ablate_{ab}"] = S0.eval_bpb(model, val, starts, a.T, tb, device)
    model.store.set_ablate(None)
    model.eval()
    used = [torch.zeros(model.M, device=device) for _ in range(model.hops)]
    for i in range(0, min(len(starts), 64), 16):
        x = np.stack([val[s:s + a.T] for s in starts[i:i + 16]]).astype(np.int64)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
            model(torch.from_numpy(x).to(device))
        # the stats of hop h are the last ones its store saw; with one shared store only the last hop is counted
        for h in range(model.hops if len(model.store.banks) > 1 else 1):
            sel = model.store.banks[h].last_si[..., : model.k].reshape(-1)
            used[h].index_add_(0, sel, torch.ones(sel.shape[0], device=device))
    res["locations_used_fraction_per_store"] = [float((u > 0).float().mean()) for u in used[: len(model.store.banks)]]
    json.dump(res, open(res_path, "w"), indent=1)
    print("ABLATE", a.run, json.dumps({k: round(res[k]["bpb"], 5) for k in res if k.startswith("val_test_ablate_")}),
          "used", [round(u, 4) for u in res["locations_used_fraction_per_store"]], flush=True)


class _Ns:
    """Reads through to `base`, except for the names given. Stands in for S0's `torch` name during train()."""

    def __init__(self, base, **over):
        self.__dict__.update(_base=base, **over)

    def __getattr__(self, name):
        return getattr(self._base, name)


def ckpt_dir(a):
    return os.path.join(os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints")), a.run)


def prepare_cooldown(a):
    """Put the source checkpoint in the fork's folder (unless the fork already has one), fix the step count."""
    ck = torch.load(a.init_ckpt, map_location="cpu")
    step0, peak = int(ck["step"]), O.peak_lr_from_opt_state(ck["opt"])
    mine = {"d": a.d, "T": a.T, "seed": a.seed, "M": a.M, "k": a.k, "hops": a.hops, "softness": a.softness,
            "qgrad": a.qgrad, "train_name": a.train_name, "n_back": a.n_back, "decays": [float(x) for x in a.decays.split(",")]}
    if a.arm in ONESDM_ARMS:
        mine["onesdm"] = onesdm_cfg(a)
    diff = {k: (ck["cfg"].get(k), v) for k, v in mine.items() if ck["cfg"].get(k) != v}
    if diff or ck.get("arm") != a.arm:
        raise SystemExit(f"cooldown fork: flags differ from the checkpoint (checkpoint, this run): arm {ck.get('arm')}/{a.arm} {diff}")
    custom = a.store_opt != "dense" or a.body_opt != "adamw"
    if ("sdmonly_composite" in ck["opt"]) != custom:
        raise SystemExit("cooldown fork: the checkpoint's optimiser state and this run's --store-opt/--body-opt differ; "
                         "give the fork the source run's optimiser flags")
    last = os.path.join(ckpt_dir(a), "last.pt")
    os.makedirs(ckpt_dir(a), exist_ok=True)
    if os.path.exists(last) and os.path.samefile(last, a.init_ckpt):
        raise SystemExit("cooldown fork: --run names the source run's own folder; give the fork its own --run")
    if a.fresh or not os.path.exists(last):
        shutil.copyfile(a.init_ckpt, last + ".tmp")
        os.replace(last + ".tmp", last)
    a.fresh = False  # train() must resume from the copied checkpoint
    n_steps = math.ceil(a.cooldown_tokens / (a.B * a.T))
    a.tokens = (step0 + n_steps) * a.B * a.T
    a.cooldown_step0, a.cooldown_peak = step0, peak
    print(json.dumps({"event": "cooldown_fork", "from": a.init_ckpt, "from_step": step0, "peak_lr": peak,
                      "cli_lr": a.lr, "cooldown_steps": n_steps, "last_step": step0 + n_steps}), flush=True)
    return cooldown_lr(step0, peak)


def with_keep(lr_fn, a, cap):
    """lr(step) is called once at the top of every step, when the model and optimiser hold the state after `step` steps."""
    steps = {math.ceil(t / (a.B * a.T)): t for t in a.keep_at}

    def lr_at(step, total, peak, warm, *rest):
        if step in steps and step > 0:
            path = os.path.join(ckpt_dir(a), f"keep_{steps[step]}.pt")
            if not os.path.exists(path):
                torch.save({"model": cap["model"].state_dict(), "opt": cap["opt"].state_dict(), "step": step,
                            "cfg": cap["cfg"], "arm": a.arm, "tokens": step * a.B * a.T}, path + ".tmp")
                os.replace(path + ".tmp", path)
                print(json.dumps({"event": "keep", "path": path, "step": step, "tokens": step * a.B * a.T}), flush=True)
        return lr_fn(step, total, peak, warm, *rest)
    return lr_at


def fingerprint(arr):
    m = 1 << 20
    h = hashlib.sha256(np.ascontiguousarray(arr[:m]).tobytes())
    h.update(np.ascontiguousarray(arr[max(0, len(arr) - m):]).tobytes())
    return h.hexdigest()[:16]


def data_record(a, arr, B_global, accum_global):
    return {"train_name": a.train_name, "train_n": int(len(arr)), "fingerprint": fingerprint(arr), "B_global": int(B_global),
            "accum_global": int(accum_global), "micro_B": int(B_global // max(1, accum_global)), "T": int(a.T),
            "data_n_flag": a.data_n}


def check_resume_data(old, new, path):
    """old: the record in a checkpoint (or None); new: this run's. Exits on a difference that changes the batches."""
    if not old:
        print(json.dumps({"event": "resume_data_check", "path": path, "note": "this checkpoint carries no data record "
                          "(written before lane DDPRECIPE); the train length and global batch cannot be checked"}), flush=True)
        return
    keys = ("train_n", "fingerprint", "B_global", "T")
    diff = {k: (old.get(k), new.get(k)) for k in keys if old.get(k) != new.get(k)}
    if diff:
        hint = ""
        if "train_n" in diff and new["train_n"] > old["train_n"]:
            hint = f" This train file is longer; --data-n {old['train_n']} draws from its first {old['train_n']} tokens."
        raise SystemExit(f"resume refused: {path} was written with {json.dumps({k: v[0] for k, v in diff.items()})}; this run has "
                         f"{json.dumps({k: v[1] for k, v in diff.items()})}. The batches of a step would differ.{hint} "
                         f"Match the checkpoint, or use --fresh.")
    if old.get("micro_B") != new.get("micro_B"):
        print(json.dumps({"event": "resume_data_check", "path": path, "note": "the micro-batch differs (batch-norm statistics "
                          "are grouped differently); the batches are the same", "micro_B": [old.get("micro_B"), new.get("micro_B")]}), flush=True)


def peek_data(path):
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError:  # a file not in the zip format cannot be memory-mapped
        ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("cfg", {}).get("data")


def train_array(a):
    arr = _LOAD_SPLIT(a.train_name)
    if a.data_n:
        if a.data_n > len(arr):
            raise SystemExit(f"--data-n {a.data_n} is longer than the train file ({len(arr)} tokens)")
        arr = arr[: a.data_n]
    return arr


def grad_dump_spec():
    v = os.environ.get("SDMONLY_DUMP_GRADS")
    if not v:
        return None
    d, _, st = v.partition("@")
    os.makedirs(d, exist_ok=True)
    return d, ({int(x) for x in st.split(",") if x} or None)


# ---------------------------------------------------------------- selftest (lane TEXTWIRE)
def selftest():
    """CPU, about a minute: the onesdm arms through the real S0 loop on a tiny 300-token shard."""
    import tempfile
    import torch.nn.functional as F
    t_start = time.time()
    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)

    tmp = tempfile.mkdtemp(prefix="sdmonly_train_selftest_")
    data, ckd, rund = (os.path.join(tmp, d) for d in ("data", "ck", "runs"))
    for d in (data, ckd, rund):
        os.makedirs(d)
    V = 300
    i = np.arange(40_000)
    web = (1 + (i * 7) % 97).astype(np.uint32)  # a learnable cycle over ids 1..97, a BOS every 64 tokens
    web[::64] = 0
    web.tofile(os.path.join(data, "train.u32"))
    web[:12_000].tofile(os.path.join(data, "val.u32"))
    tb = np.ones(V, dtype=np.int32)
    tb[0] = 0  # BOS has no bytes, as in the real table
    tb.tofile(os.path.join(data, "token_bytes.i32"))
    saved_env = {k: os.environ.get(k) for k in ("SDMLLM_RUNS", "SDMLLM_CKPT")}
    saved_data, saved_F, threads = S0.DATA, S0.F, torch.get_num_threads()
    os.environ.update(SDMLLM_RUNS=rund, SDMLLM_CKPT=ckd)
    S0.DATA = data
    torch.set_num_threads(1)  # multi-threaded CPU backward is not bit-deterministic; one thread is
    rec_loss = []

    def ce(*args, **kw):  # records the training loss at full precision (the log rounds it to 4 places)
        out = F.cross_entropy(*args, **kw)
        if torch.is_grad_enabled() and kw.get("reduction", "mean") == "mean":
            rec_loss.append(float(out.detach()))
        return out
    S0.F = _Ns(F, cross_entropy=ce)
    tiny = ["--d", "32", "--layers", "2", "--mlp-f", "64", "--mem-heads", "4", "--mem-n-sub", "6", "--mem-d-a", "8",
            "--mem-k", "4", "--chunk", "16", "--sdm-heads", "4", "--sdm-d-a", "8", "--sdm-k", "4", "--B", "8", "--T", "32",
            "--cpu", "--log-every", "5", "--eval-every", "100000", "--ckpt-every", "100000", "--val-windows", "40"]
    try:
        # 1. every onesdm arm trains through the S0 loop and its loss falls
        for arm in ONESDM_ARMS:
            rec_loss.clear()
            main(["--arm", arm, "--run", f"st_{arm}", "--tokens", str(8 * 32 * 30), "--fresh"] + tiny)
            first, last5 = rec_loss[0], float(np.mean(rec_loss[-5:]))
            check(f"{arm}: trains on CPU at a tiny shape and the loss falls", len(rec_loss) == 30 and last5 < first - 1.0,
                  f"{first:.3f} -> {last5:.3f} over {len(rec_loss)} steps")
        # 2. the vocabulary comes from the data: token_bytes.i32 has 300 entries
        ck = torch.load(os.path.join(ckd, "st_onesdm", "last.pt"), map_location="cpu", weights_only=False)
        check("vocabulary from the data: a 300-token shard builds a 300-row embedding",
              tuple(ck["model"]["emb.weight"].shape) == (300, 32), str(tuple(ck["model"]["emb.weight"].shape)))
        # 3. TEST from the trainer = S0.eval_bpb on the same windows, of the model rebuilt from the checkpoint alone
        res = json.load(open(os.path.join(rund, "st_onesdm.result.json")))
        S0.V = V
        mdl = build_text_model(ck["arm"], V, ck["cfg"])
        mdl.load_state_dict(ck["model"])
        val = np.memmap(os.path.join(data, "val.u32"), dtype=np.uint32, mode="r")
        ev = S0.eval_bpb(mdl, val, S0.val_windows(val, 32, "test", 40), 32, torch.from_numpy(tb.astype(np.int64)),
                         torch.device("cpu"))
        check("TEST from the trainer equals S0.eval_bpb on the same windows (model rebuilt from the checkpoint)",
              abs(ev["bpb"] - res["val_test"]["bpb"]) < 1e-9 and ev["tokens"] == res["val_test"]["tokens"],
              f"{ev['bpb']:.9f} vs {res['val_test']['bpb']:.9f}")
        # 4. checkpoint, stop, resume = one unbroken run (onesdm_allsdm: Muon, the AdamW store group, the tables)
        base = ["--arm", "onesdm_allsdm", "--tokens", str(8 * 32 * 12)] + tiny
        rec_loss.clear()
        main(base + ["--run", "st_straight", "--fresh"])
        straight = list(rec_loss)
        main(base + ["--run", "st_broken", "--fresh", "--stop-after-steps", "6"])
        rec_loss.clear()
        main(base + ["--run", "st_broken"])
        resumed = list(rec_loss)
        rel = max(abs(x - y) / abs(y) for x, y in zip(resumed, straight[6:])) if len(resumed) == 6 else float("inf")
        check("checkpoint then resume equals an unbroken run: the losses of steps 7..12 within 1e-6 relative",
              len(straight) == 12 and rel <= 1e-6, f"max relative difference {rel:.2e}")
        # 5. the optimiser split
        a = parse(["--arm", "onesdm_allsdm"] + tiny)
        mdl = build_text_model("onesdm_allsdm", V, s0_cfg(a))
        opt = make_optimizer(a, mdl, 3e-3)
        seen = [id(q) for g in opt.param_groups for q in g["params"]]
        params = [id(q) for q in mdl.parameters()]
        check("the Muon / AdamW split holds every parameter in exactly one group",
              sorted(seen) == sorted(params) and len(set(seen)) == len(seen) and "muon" in opt.parts,
              f"{len(seen)} entries, {len(params)} parameters")
        adamw_store = [g for g in opt.parts["adamw"].param_groups if g["lr_mult"] == a.store_lr_mult]
        store_ids = {id(q) for g in adamw_store for q in g["params"]}
        tables = [(n, q) for n, q in mdl.named_parameters() if n.endswith(".values") or n.endswith(".keys1")]
        check("the memory value tables and sub-keys are in the AdamW store group, none in Muon",
              len(tables) > 0 and all(id(q) in store_ids for _, q in tables)
              and not any(id(q) in {id(x) for x in opt.parts["muon"].params} for _, q in tables), f"{len(tables)} tables")
        # 6. --rt-reset-at-bos
        x = torch.randint(1, V, (2, 40), generator=torch.Generator().manual_seed(5))
        x[0, 19] = 0  # one BOS that sits across the chunk boundary at 16 (chunk 16)
        x[1, 5] = 0
        x[1, 33] = 0  # two BOS in row 1, the last one in the third chunk

        def built(flags):
            torch.manual_seed(0)
            m_ = build_text_model("onesdm", V, s0_cfg(parse(["--arm", "onesdm"] + tiny + flags)))
            for mm in m_.mems:  # a non-zero output map and gate, so the memory changes the output
                torch.nn.init.normal_(mm.wo.weight, std=0.3)
                torch.nn.init.normal_(mm.wg.weight, std=0.5)
            return m_.eval()
        mr, mo = built(["--rt-reset-at-bos"]), built([])
        mo.load_state_dict(mr.state_dict())
        with torch.no_grad():
            full = mr(x)
            dev_on = max(float((full[0, 19:] - mr(x[:1, 19:])[0]).abs().max()),
                         float((full[1, 33:] - mr(x[1:, 33:])[0]).abs().max()))
            full_off = mo(x)
            dev_off = float((full_off[0, 19:] - mo(x[:1, 19:])[0]).abs().max())
            dev_seq = float((mr(x, sequential=True) - full).abs().max())
        check("--rt-reset-at-bos on: tokens from a BOS on give the output of the same tokens fed alone", dev_on < 1e-5,
              f"max difference {dev_on:.2e}")
        check("--rt-reset-at-bos off (the control): they do not", dev_off > 1e-3, f"max difference {dev_off:.2e}")
        check("--rt-reset-at-bos: the token-by-token recurrence equals the chunked form", dev_seq < 1e-5,
              f"max difference {dev_seq:.2e}")
        # 7. the bench runs real training steps, and the size report counts the built model
        bres = main(["--arm", "onesdm", "--bench-steps", "3"] + tiny)
        check("--bench-steps runs real steps: forward, backward and the optimiser move the weights",
              bres["weights_moved"] and bres["tok_per_s_median"] > 0, f"{bres['tok_per_s_median']} tok/s on CPU")
        bch = main(["--arm", "onesdm_allsdm", "--bench-steps", "1", "--loss", "chunked"] + tiny)
        bpl = main(["--arm", "onesdm_allsdm", "--bench-steps", "1"] + tiny)
        check("--loss chunked (the Spark path: logits = hidden @ E^T) gives the plain loss on the same batch",
              abs(bch["losses"][0] - bpl["losses"][0]) < 1e-4, f"{bch['losses'][0]} vs {bpl['losses'][0]}")
        pr = main(["--arm", "onesdm_allsdm", "--param-report"] + tiny)
        pc = OS.param_count(build_text_model("onesdm_allsdm", V, s0_cfg(parse(["--arm", "onesdm_allsdm"] + tiny))))
        check("--param-report equals the built model's counts", pr["non_embedding"] == pc["non_embedding"]
              and pr["embedding"] == pc["embedding"] == V * 32, f"{pr['non_embedding']} / {pr['embedding']}")
    finally:
        S0.DATA, S0.F = saved_data, saved_F
        torch.set_num_threads(threads)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if os.environ.get("TRAIN_SELFTEST_KEEP"):
        print("kept", tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(ok)} of {len(ok)} checks pass in {time.time() - t_start:.0f} s")
    print("NEXT -> python3 track4_sdmonly_finetune.py --selftest")
    return all(ok)


def main(argv=None):
    """Train one arm (returns the result dict), or --bench-steps / --param-report (returns their record). The S0
    module's names this function replaces are put back when it returns, so it can run several times in one process."""
    a = parse(argv)
    if a.selftest:
        return selftest()
    saved = {n: getattr(S0, n) for n in ("build", "flops_per_token", "lr_at", "load_split", "torch", "V")}
    try:
        return _main(a)
    finally:
        for n, v in saved.items():
            setattr(S0, n, v)
        set_vocab(saved["V"])


def _main(a):
    # THE VOCABULARY COMES FROM THE DATA: token_bytes.i32 holds one entry per token id (129,280 for the DeepSeek
    # shards, 32,768 for the trained BPE ones). It sets S0.V, which every model build and scorer reads.
    V = data_vocab()
    if a.param_report or a.bench_steps:
        V = a.vocab or V
        if not V:
            raise SystemExit(f"no {os.path.join(S0.DATA, 'token_bytes.i32')}: give --vocab N for a bench or a size report")
        set_vocab(V)
        return param_report(a, V) if a.param_report else bench(a, V)
    if not V:
        raise SystemExit(f"{os.path.join(S0.DATA, 'token_bytes.i32')} not found: the vocabulary is read from it")
    set_vocab(V)
    onesdm = a.arm in ONESDM_ARMS
    ex = extra_cfg(a)
    custom_opt = a.store_opt != "dense" or a.body_opt != "adamw"
    cap = {}

    def build(arm, V, cfg):
        cfg.update(ex)  # the full build settings ride in every checkpoint's cfg and in the result json
        cfg["data"] = rec  # what fixes the batches; a resume on another machine is checked against it
        if arm == "qwen":
            cfg.update({"n_layer": a.qwen_layers, "f": a.qwen_f or 688})  # defaults = every earlier yardstick
        if onesdm:
            # each training window starts with an EMPTY run-time memory: nothing is carried from one window (or batch
            # row) to the next. With --rt-reset-at-bos it is also emptied at every BOS inside a window.
            cfg["onesdm"] = onesdm_cfg(a)
        cap["model"], cap["cfg"] = eager_body(build_text_model(arm, V, cfg), a), cfg
        return cap["model"]

    if a.data_n:
        S0.load_split = lambda name: train_array(a) if name == a.train_name else _LOAD_SPLIT(name)
    arr = train_array(a)
    check_ids(arr, V, a.train_name)
    check_ids(_LOAD_SPLIT("val"), V, "val")
    rec = data_record(a, arr, a.B, a.accum)
    S0.build = build
    S0.flops_per_token = text_flops_per_token
    if a.sched == "wsd":
        S0.lr_at = wsd_lr(a.wsd_decay)
    if a.init_ckpt:
        S0.lr_at = prepare_cooldown(a)
    last = os.path.join(ckpt_dir(a), "last.pt")
    if os.path.exists(last) and not a.fresh:
        check_resume_data(peek_data(last), rec, last)
    if a.keep_at:
        S0.lr_at = with_keep(S0.lr_at, a, cap)
    dump = grad_dump_spec()
    if dump:
        _lr = S0.lr_at

        def lr_seen(step, *rest):
            cap["step"] = step
            return _lr(step, *rest)
        S0.lr_at = lr_seen
    if a.stop_after_steps:
        S0.lr_at = with_stop(S0.lr_at, a, cap)
    if custom_opt or onesdm or a.keep_at or dump or a.stop_after_steps:
        def adamw(groups, lr, betas, eps):
            if onesdm or custom_opt:  # the onesdm arms always take their own split (onesdm_group)
                cap["opt"] = make_optimizer(a, cap["model"], lr, betas, eps)
            else:
                cap["opt"] = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
            return cap["opt"]

        def clip(parameters, max_norm):
            if dump and (dump[1] is None or cap["step"] in dump[1]):
                torch.save({n: q.grad.detach().clone() for n, q in cap["model"].named_parameters() if q.grad is not None},
                           os.path.join(dump[0], f"grad_{cap['step']}.pt"))
            return O.clip_grad_norm_(list(parameters), max_norm, cap.get("opt"))

        S0.torch = _Ns(torch, optim=_Ns(torch.optim, AdamW=adamw),
                       nn=_Ns(torch.nn, utils=_Ns(torch.nn.utils, clip_grad_norm_=clip)))
    os.makedirs(os.environ["SDMLLM_RUNS"], exist_ok=True)
    print(json.dumps({"event": "sdmonly", "arm": a.arm, "V": V, **ex, **opt_info(a),
                      **({"onesdm": onesdm_cfg(a)} if onesdm else {})}), flush=True)
    try:
        S0.train(a)
    except _StopEarly as e:
        print(json.dumps({"event": "stopped_early", "step": e.args[0], "checkpoint": last,
                          "why": "--stop-after-steps; run again without it (and without --fresh) to resume"}), flush=True)
        return {"stopped_at": e.args[0]}
    res_path = os.path.join(os.environ["SDMLLM_RUNS"], f"{a.run}.result.json")
    if a.arm == "sdmonly":
        score_ablations(a, res_path)
    if opt_info(a) or onesdm:
        res = json.load(open(res_path))
        if opt_info(a):
            res["sdmonly_opt"] = opt_info(a)
        if onesdm:
            res["vocab"] = {"V": V, "from": os.path.join(S0.DATA, "token_bytes.i32")}
        json.dump(res, open(res_path, "w"), indent=1)
    print("NEXT -> python3 track4_sdmonly_summary.py")
    return json.load(open(res_path))


if __name__ == "__main__":
    out = main()
    if out is False:
        sys.exit(1)
