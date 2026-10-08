# A_q05 nanochat commits, PRs and dev/LOG.md since 2026-08-01

## (a) commits since 2026-08-01

- Query: GET https://api.github.com/repos/karpathy/nanochat/commits?since=2026-08-01T00:00:00Z&per_page=100
- Fetch: 2026-10-03T10:48:24.427945Z 200
- Result: EMPTY (0 commits). The latest commit on master is 2026-07-03T22:54:57Z 'clean up fragile code'.

## (b) widened: commits since 2026-06-01

- Fetch: 2026-10-03T10:48:32.724383Z 200
- Hits: 25 (all dated 2026-07-02 or 2026-07-03)

```
92d63d4e 2026-07-03T22:54:57Z | clean up fragile code
eb16d017 2026-07-03T22:49:35Z | add inference benchmarking
f8a85a5f 2026-07-03T20:18:36Z | add a few tests
a9d0a862 2026-07-03T19:57:47Z | unify the two optimizer implementations into one and prevent bugs like the one that just happened where i updated one and didnt update the other
f4f69f9d 2026-07-03T19:39:09Z | harden and shorten the execution sandbox
f527f761 2026-07-03T19:27:43Z | fix deprecated commit on torchao
a5a3a33c 2026-07-03T19:22:02Z | delete datasets dependency bye
41865401 2026-07-03T18:59:45Z | nuke huggingface and its giant dependency footprint from orbit. datasets still todo
950a1dc6 2026-07-03T18:48:32Z | delete spurious 'cute' bloat that doesn't matter to a serious research stack
da32e1d6 2026-07-03T18:06:54Z | remove a ton of UI bloat, we need to make nanochat small and sexy and minimal and forkable. code is ~free now, if you want UI get your favorite LLM to add it. nukes ~1000 LOC from orbit nice
21a7774e 2026-07-03T17:44:26Z | Validate checkpoint file format in find_last_step
6facdeef 2026-07-03T17:40:03Z | Improve error handling for batch size alignment
79aaf1d5 2026-07-03T17:39:32Z | Remove inherited parameters from scripts
2fc63c96 2026-07-03T17:38:16Z | Merge pull request #540 from svlandeg/fix/kernel
dbe6e2aa 2026-07-03T17:36:18Z | Small doc fixes for consistency
3616a029 2026-07-03T17:34:12Z | add a comment clarifying that RoPE here rotates by -theta (transpose of the textbook convention), which is functionally equivalent. Prompted by #489, thank you @qdrk
2ce972a3 2026-07-03T17:30:47Z | fix token_bytes calculation to use raw token bytes instead of round-tripping through str, which corrupts tokens that are not valid standalone UTF-8 (e.g. bytes >= 0x80 became 3-byte U+FFFD). 193/32768 tokens were affected, so bpb of new runs will read ~0.05% higher than old runs. Thank you @ellenjxu #693
5e43f1eb 2026-07-03T17:26:07Z | docs: note that MPS handles bf16 fine on recent macOS (opt in via NANOCHAT_DTYPE=bfloat16, ~25% memory savings). Thank you @eaglstun #793 and @ifsheldon #685
ca366eb1 2026-07-03T17:26:01Z | allocate the inference KV cache in COMPUTE_DTYPE instead of hardcoding cuda->bf16, fixing bf16 inference (e.g. NANOCHAT_DTYPE=bfloat16 on MPS) and deleting an old hack. Thank you @mparrett #795
e02b3395 2026-07-03T17:25:55Z | fix mixed-dtype ops in the optimizer so that bf16 params (wte, value_embeds) work on MPS, which unlike CUDA refuses to promote dtypes. AdamW math now runs explicitly in fp32; verified bit-identical on CUDA, where inductor was already upcasting in registers. Thank you @mparrett for the report and investigation in #741
4e014d31 2026-07-03T04:01:27Z | adding MuonEq and Muon+, both seem to be very slight improvements at both d12 and d24. contributed by @jn2clark
9ec05d97 2026-07-02T22:45:44Z | fix link
fbb50ad1 2026-07-02T22:43:50Z | rename fwe to climbmix
07e21031 2026-07-02T16:49:15Z | Merge branch 'master' into feat/small_fixes
f10bd751 2026-07-02T02:46:01Z | delete the whole report thing i think it was a bad idea it just bloats the code. doing this subtracts 540 LOC
```

## (c) dev/LOG.md

- Query: GET https://api.github.com/repos/karpathy/nanochat/commits?path=dev/LOG.md
- Last LOG.md change: 2026-05-05 'tried and failed at DyT'. No LOG.md change in the window.
- Verbatim latest entry heading: '## 2026-05-05: DyT for d12 pretraining (negative)'
- Verbatim: "Every variation of the idea that was attempted, including after a bunch of parameter tuning did not outperform the baseline d12 model on master, even with steps on the x-axis. In addition, the throughput (tokens per second) was ~10% lower."
- Verbatim (2026-03-02 SoftCap tuning): "Tried 5..30. 5 was terrible, the rest of them were all about equal with the exception of 20, which was the best. Minor but solid improvement: val loss improved by ~1e-3 (0.716 -> 0.715)."

## (d) PRs, newest 40, all states

- Query: GET https://api.github.com/repos/karpathy/nanochat/pulls?state=all&sort=created&direction=desc&per_page=40
- Fetch: 2026-10-03T10:48:35.027688Z 200

```
859 2026-09-25 open merged=- msjgriffiths | Enable nonblocking CPU->GPU data transfers during training
857 2026-09-17 open merged=- marcinbogdanski | Fix LR divergence across ranks in SFT
855 2026-09-13 open merged=- dijia111223 | Add numerical KV cache equivalence test + CPU CI smoke workflow
854 2026-09-11 open merged=- nihir27 | New record: 91.7 minutes. A 300M-row hashed n-gram table in host RAM + d24 model → 0.70367 val bpb, CORE 0.2578
853 2026-09-11 open merged=- 1023097618 | Fix FA3 backend selection on Blackwell GPUs
852 2026-09-08 open merged=- AshutoshBhonsle1 | Added functionality to pass desired dataset size in MB as an argument using -s
851 2026-09-08 closed merged=- lingwei-gu | Batch size ramp for larger models: 1/4 -> 1/2 -> full over the first 10% of tokens
850 2026-09-08 closed merged=- lingwei-gu | Batch size ramp for larger models: 1/4 -> 1/2 -> full over the first 10% of tokens
845 2026-08-31 open merged=- KaiEureka | eval: reject models with future-token leakage
843 2026-08-30 open merged=- sinameraji | optim: keep async collective inputs and Work handles alive until wait()
842 2026-08-30 open merged=- smiletrl | fix timeout context manager
841 2026-08-28 open merged=- noQbot | Remove unused TaskSequence class
839 2026-08-28 open merged=- eaglstun | Report a real peak memory number on CPU/MPS instead of 0.00MiB
837 2026-08-27 open merged=- isubuz | Stop requantizing fp8 weight; actually pin dataloader buffers
836 2026-08-22 open merged=- ankel | Support amd (rocm) for training
835 2026-08-21 closed merged=- RodrickKamsiyonna | Qwen
834 2026-08-19 closed merged=- ankel | fix(sft): pass max_tokens to render_conversation so conversations fit
833 2026-08-17 open merged=- PabloRaka | feat(attention): Add multi-hardware support for FA2, AMD ROCm (MI300X/CDNA), and SDPA fallback
832 2026-08-16 open merged=- lucaGazzola | overlap dataloader with gpu compute via background thread prefetch
831 2026-08-16 open merged=- smiletrl | Fix wandb api key length issue by upgrade wandb to 0.28.2
830 2026-08-15 closed merged=- giovannizinzi | Speed up the 8xH100 GPT-2 run to 81.8 minutes
829 2026-08-15 closed merged=- eliotli | Speed up GPU/ROCm inference with fused QKV, last-token logits, and torch.compile
828 2026-08-11 open merged=- LaurenceLong | Smooth the squared-ReLU activation
827 2026-08-11 closed merged=- nash635 | add wp2 to wp5 works
826 2026-08-09 open merged=- Qwertyemma | Fix SDPA fallback: sliding window doesn't reduce memory (fixes #825)
823 2026-08-03 closed merged=- Kecro88 | Claude/sig reg nanogpt n4r3wc
822 2026-08-03 open merged=- anupamme | fix: upgrade gitpython to 3.1.47 (CVE-2026-42284)
821 2026-08-03 open merged=- anupamme | fix: upgrade gitpython to 3.1.47 (CVE-2026-42215)
818 2026-08-01 open merged=- marcinbogdanski | SFT: Fix off-by-one EMA debias
817 2026-07-31 closed merged=- marcinbogdanski | SFT: Fix NaN caused by conversations trimmed to hard-coded 2048
816 2026-07-31 open merged=- marcinbogdanski | SFT: Fix incorrect progress accounting and LR scaling.
815 2026-07-30 open merged=- Xiaomu-Lin | fix: count Float8Linear in num_matmul_params (--fp8 understates FLOPs/token and MFU ~12x)
814 2026-07-29 open merged=- tonydzi | read-arxiv-paper skill: handle the URL and archive shapes arxiv actually serves
813 2026-07-23 open merged=- helloaidank | Small typo: removing redundant head_dim from gpt file
812 2026-07-22 closed merged=- 828Tina | Lxy
811 2026-07-21 open merged=- parasol-aser | Fix SFT dataloader buffer clog that leads to NaN loss
809 2026-07-17 open merged=- Osamaali313 | Fix wrong worked example in short_window comment
808 2026-07-13 open merged=- parasol-aser | chat_rl: support resuming an interrupted RL run
807 2026-07-13 open merged=- jurca | 94 minutes - Widening embedding after first transformer block
806 2026-07-11 open merged=- veryfansome | Add counterfactual controls + response logging to the CORE eval
```

No PR in this list is merged.

## (e) PR bodies read (verbatim excerpts)

### PR 854: New record: 91.7 minutes. A 300M-row hashed n-gram table in host RAM + d24 model → 0.70367 val bpb, CORE 0.2578 (created 2026-09-11T11:22:50Z, state open)

- Fetch: 2026-10-03T10:48:42.055812Z 200
- URL: https://github.com/karpathy/nanochat/pull/854

```
Hashed bigram/trigram lookup tables in host RAM, injected into the attention
values of four layers.

- One Engram table in `/dev/shm`, mapped by all eight ranks and updated lock-free (Hogwild!). No VRAM.
- Host IO hidden: rows for the next micro-step are prefetched and updates are written back asynchronously. +1.6% step time at d24.
- Row-wise optimiser on the GPU: state is one scalar per row and Engram layer, kept in VRAM.

| | this PR | Run 6 |
|---|---|---|
| val bpb | **0.703668** | 0.71800 |
| CORE | **0.2578** | 0.262634 |
| total_training_time | **91.74 min** | 99 mins |

Individual runs on the 8x H100 node:

| attempt | val bpb | CORE | total_training_time |
|---|---|---|---|
| 1 | 0.703674 | 0.2592 | 91.56 min |
| 2 | 0.703734 | 0.2557 | 91.63 min |
| 3 | 0.703596 | 0.2584 | 92.03 min |
| mean | 0.703668 | 0.2578 | 91.74 min |

## Launch

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --window-pattern=SSSL \
    --fp8 \
    --device-batch-size=16 \
    --num-iterations=5000 \
    --model-tag=d24-engram \
    --eval-every=500 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-every=999999 \
    --core-metric-max-per-task=-1 \
    --engram-table-size=300000000 \
    --engram-layers=3,7,11,15 \
    --engram-rank=128 \
    --engram-orders=2,3 \
    --engram-order-weights=2,2 \
    --engram-lr=0.2 \
    --engram-gate-channels=12 \
    --engram-accum-beta=0.99 \
    --engram-decay=0.001
```

72 GiB table in `/dev/shm`, released at the end of the run. The checkpoint
holds the model only (4.2 GB): the table is not saved, so the checkpoint is
not a runnable model on its own. Persisting the table is a 72 GB write and is
left out of this PR. Reported numbers are `Minimum validation bpb`, `CORE metric` and
`Total training time` from the log, as for the other leaderboard runs.

## What it does

Each token hashes its last two and three token ids, with two salted hashes per
order, into four rows of one bf16 table. A row is 128 values: a 32-dim slice for
each of the four engram layers. A layer takes its 32-dim slice from each of the
token's four rows and concatenates them (bigram A, bigram B, trigram A, trigram
B) into a 128-dim embedding, gates it on the first 12 residual channels,
projects it to model width and adds it to
```

### PR 850: Batch size ramp for larger models: 1/4 -> 1/2 -> full over the first 10% of tokens (created 2026-09-08T01:51:21Z, state closed)

- Fetch: 2026-10-03T10:48:42.841858Z 200
- URL: https://github.com/karpathy/nanochat/pull/850

```
**Problem.** The auto batch-size rule was validated at d12 (2^19) and d26 (2^20) and extrapolates to 2^21 for d32+. A d34 trained on ClimbMix at 2^21 (current master, everything else default, 8xH100, one pass over the corpus) was 4-6x less token-efficient than the d24 speedrun: the d24 reaches val bpb 0.7185 in 5.8B tokens, this d34 needed 34B, and CORE was flat at ~0.30 from step 14k to 24k.

| tokens | 4.2B | 12.6B | 25.2B | 37.7B | 50.3B |
|---|---|---|---|---|---|
| val bpb | 0.7529 | 0.7276 | 0.7192 | 0.7169 | 0.7143 |

The critical batch size is small early in training and grows as the loss falls; a single large batch from step 0 wastes most of the early tokens.

**Change.** `--batch-ramp=1` (default off): train at 1/4 of the batch for the first 3% of the token budget and 1/2 for the next 7%, then the full batch, with sqrt LR scaling per stage. Same token budget, ~16% more optimizer steps. The auto batch is also capped at 2^20, the largest validated value. For d34 this gives 262,144 -> 524,288 -> 1,048,576.

**Status.** The d34 ramp run (ClimbMix, one epoch) is in progress; I will post its curve at matched token counts here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

```

### PR 830: Speed up the 8xH100 GPT-2 run to 81.8 minutes (created 2026-08-15T23:53:27Z, state closed)

- Fetch: 2026-10-03T10:48:43.656107Z 200
- URL: https://github.com/karpathy/nanochat/pull/830

```
## Result

This is my next speedrun attempt, following the d22 + muon attempt in [nanochat #733](https://github.com/karpathy/nanochat/pull/733). It comes with a few speedups. Its main tradeoff is the new `liger-kernel` dependency. At the speedrun's 49,152-token vocabulary (the repository default remains 32,768), its efficient Triton cross-entropy cuts loss memory enough to use device batch 32 without OOMing.

| | Run 6 (current nanochat) | this PR |
|---|---:|---:|
| recipe | d24, ratio 8 | d22, ratio 9.4 + fused CE + selective RMSNorm scales |
| tokenizer vocabulary | 32,768 | 49,152 (speedrun override; default unchanged) |
| training time | 99.0 min | **81.835 +/- 0.138 min** |
| canonical val BPB | 0.71800 (N=5) | **0.718786 +/- 0.000054 (N=6)** |
| 22-task CORE | 0.262634 (N=5) | **0.261814 +/- 0.002295 (N=6)** |

Relative to Run 6, pretraining is **17.3% faster**. Val BPB differs by +0.000786 and CORE by -0.000820; CORE is within the benchmark's observed run-to-run noise, and all six runs clear the GPT-2 floor of 0.256525.

## What changed, why, and what it bought

| Change | Why | Measured effect / tradeoff |
|---|---|---|
| Restore pinned staging for `cuda:N` devices | DDP passes indexed CUDA devices, but the loader only pinned for the literal string `cuda`. | Saves **7.30 s** on average in full-recipe matches; no data-order or CPU-path change. |
| Reuse Muon communication workspaces | Shape-stable reduce-scatter/all-gather tensors were reallocated every step. | A matched 600-step screen fell from **376.96 s to 261.07 s**; long allocator stalls fell from **322 to 1**. Buffers remain outside checkpoint state. |
| Use a 49,152-token speedrun vocabulary | A larger tokenizer represents the same text with fewer tokens, so a fixed token budget covers more source text. | In an earlier controlled d22 sweep, mean CORE rose from **0.2539 to 0.2646** versus 32,768 (N=5 each); 65,536 added no CORE and was slower. The gain was not robust at d20/d26, so this remains a speedrun-only override; the repository default stays 32,768. |
| Optional Liger cross entropy | At the 49k vocabulary and device batch 32, each GPU produces ~6 GiB BF16 logits and native CE creates another ~12 GiB FP32 copy. | Device batch 32 / GA1 fits. Native CE OOMs at that shape; batch 16 / GA2 is **4.8% slower**. Evaluation and inference stay native. |
| Learn scales on MLP-
```

### PR 828: Smooth the squared-ReLU activation (created 2026-08-11T12:29:20Z, state open)

- Fetch: 2026-10-03T10:48:44.444148Z 200
- URL: https://github.com/karpathy/nanochat/pull/828

```
This replaces the hard squared-ReLU in the MLP with a parameter-free algebraic smoothing:

$$
\phi_a(x) = \frac{1}{2}x\left(x + \sqrt{x^2 + a^2}\right), \qquad a = 0.2
$$

As $a \to 0$, it recovers squared-ReLU. The code change is deliberately one line: no new parameters, state, or configuration.

I ran a matched depth-12, 1000-step, seed-42 BF16 comparison on an RTX 4090. Source, data order, optimizer, batch schedule, and evaluation protocol were held fixed.

| step | squared-ReLU | smoothed squared-ReLU |
| ---: | ---: | ---: |
| 850 | 1.071650 | 1.071477 |
| 900 | 1.064026 | 1.063833 |
| 950 | 1.057777 | 1.057588 |
| 1000 | 1.053912 | 1.053747 |

The smoothed version was lower at all four preregistered late checkpoints. Median throughput was 87,876 vs. 88,235.5 tokens/s (99.59%), and peak memory was effectively unchanged (6297.46 vs. 6297.59 MiB).

This is still one seed at one depth, so I am opening it as a draft rather than treating it as a robust default-recipe win. The next useful check would be the reference miniseries / 8xH100 speedrun.

Disclosure: I used an LLM to help with experiment orchestration and the write-up; I reviewed the code and results.
```

### PR 807: 94 minutes - Widening embedding after first transformer block (created 2026-07-13T08:16:59Z, state open)

- Fetch: 2026-10-03T10:48:45.282164Z 200
- URL: https://github.com/karpathy/nanochat/pull/807

```
This branch narrows the token embedding matrix and the first transformer block to half width. The submission's configuration is set in `runs/speedrun.sh` while `scripts/base_train.py` has been configured for a slightly faster compute target similar to the current `karpathy:master`.

This is a wall-clock speedup at the GPT-2 bar driven by parameter savings in the first block. The IsoFLOP study (40 runs, below) confirms the change is quality-neutral per FLOP, i.e. no regression is being hidden by the faster time.

While these changes are based on an older `master` from May 5, 2026, I believe, based on my experiments, that this modification is compatible with multiple architectural designs. This is based on earlier experiments on the December 2025 codebase, though I have not benchmarked this patch on current `master`.

## Proposed Results

The respective W&B runs are linked below.

| run | val_bpb  | core_score | train_time_min |
| --- | -------- | ---------- | -------------- |
| 0   | 0.719156 | 0.2653     | 94.63          |
| 1   | 0.719385 | 0.2557     | 94.63          |
| 2   | 0.719291 | 0.2689     | 94.61          |
| 3   | 0.719261 | 0.2688     | 94.48          |
| 4   | 0.719219 | 0.2605     | 94.51          |
| 5   | 0.719213 | 0.2628     | 94.62          |

These runs at W&B:
- https://wandb.ai/martin-jurca-seznam-cz/nanochat/runs/tof2kyyr?nw=nwusermartinjurca
- https://wandb.ai/martin-jurca-seznam-cz/nanochat/runs/gp1372g1?nw=nwusermartinjurca
- https://wandb.ai/martin-jurca-seznam-cz/nanochat/runs/02ysivf0?nw=nwusermartinjurca
- https://wandb.ai/martin-jurca-seznam-cz/nanochat/runs/fj6tcwoi?nw=nwusermartinjurca
- https://wandb.ai/martin-jurca-seznam-cz/nanochat/runs/5xi8oyvg?nw=nwusermartinjurca
- https://wandb.ai/martin-jurca-seznam-cz/nanochat/runs/zw9rs537?nw=nwusermartinjurca

Means:

| Metric         | Mean   |
| -------------- | ------ |
| core_score     | 0.2637 |
| val_bpb        | 0.7193 |
| train_time_min | 94.58  |

## Changes

The model's embedding dimension is halved and brought back to full width right after the first transformer block. This is implemented using the new `WideningBlock` replacing an ordinary `Block` as the lowest model block. The `WideningBlock` works identically to regular `Block` with sole exception: after applying the MLP it widens the residual stream by concatenating it with itse
```

