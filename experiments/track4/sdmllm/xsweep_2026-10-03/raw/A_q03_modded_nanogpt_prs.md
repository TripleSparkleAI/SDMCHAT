# A_q03 modded-nanogpt pull requests, newest 40, all states

- Query: GET https://api.github.com/repos/KellerJordan/modded-nanogpt/pulls?state=all&sort=created&direction=desc&per_page=40
- Tool: python3 urllib
- Fetch: STATUS 200 UTC 2026-10-03T10:46:56.168209Z
- Hits: 40

```
379 2026-10-02 open merged=- NathanGodey | Track 1: CPLM on #360 (-4.57s, -11.25% same hardware)
378 2026-10-02 open merged=- josusanmartin | Track 1: Document-local copy feature (-1.17 s, -2.9% vs #360; stacked on #375)
377 2026-10-01 open merged=- jn2clark | Track 3 anchor gradient + fp32 embed 2580 steps (n=16)
376 2026-09-29 open merged=- cyrusghane | Track 1: Document-local copy mixture at the final validation (-15 extension steps, ~-0.8s)
375 2026-09-29 open merged=- daniel-monroe | Track 1: Normalize tokens in n-gram embeddings to improve information density (-0.42s, -1.02%, -15 steps)
374 2026-09-28 closed merged=2026-09-28 ClassicLarry | README: update for record #360 (ANVIL2)
373 2026-09-28 closed merged=2026-09-28 ClassicLarry | Track 1: refactor the ANVIL2 trainer into a readable package (no record change)
372 2026-09-28 closed merged=- ldmberman | Track 1: Adaptive softmax (-2.3s, +35 steps)
371 2026-09-25 open merged=- romeerp | Track 1: Sampled LM-head gradients and truncated attention backward (-0.76s, -1.88%)
370 2026-09-23 open merged=- orange4664 | Track 3: IsoMuon — Muon with a noise-calibrated diagonal response metric; 3.28 in 3190 steps (n=8)
369 2026-09-20 open merged=- jeffreycider | Track 3: improve Muon with MaxEntSlop momentum and Defazio-style decay. 3060 steps (-100 from PR#357)
368 2026-09-20 open merged=- jeffreycider | Track 3: improve MuonH with MaxEntSlop momentum and a fixed weight-parallel fraction. 3010 steps (-55 from PR#359)
367 2026-09-17 open merged=- hermabr | Track 1: Longest exact-match retrieval (40.0s -> 21.56s / -46.1%)
366 2026-09-15 open merged=- nihir27 | Track 1: Hashed n-gram rows from host RAM (-6.16s)
364 2026-09-05 closed merged=- dnhkng | Endgame EMA weight blending (port of Track 3 Tail-EMA readout)
363 2026-09-03 open merged=- isubuz | Track 1: FP8 MLP input gradient and kernel tuning by AI System Warpscale (-0.45s)
362 2026-09-01 open merged=- jingru-lee | Track 3: Muon-NSR reaches 3.28 in 3250 steps (n=8)
361 2026-09-01 closed merged=- jingru-lee | Track 3: Muon-NSR reaches 3.28 in 3250 steps (n=8)
360 2026-08-31 closed merged=2026-09-28 devenpzak | New Record: 0.665 minutes (39.9 seconds): ANVIL2, Sampled-softmax, New Embedding Table, Full-stack fp8 (-34.0s, -46% same hardware)
359 2026-08-28 open merged=- jacknzheng | Track3: K-maxwell momentum muonh decay - 3065 steps (n=8)
358 2026-08-27 closed merged=- matevz-kovacic | Track 1: Norm CSE, FP8 up-proj quantize (-0.729s)
357 2026-08-27 open merged=- jacknzheng | Track 3: K-Maxwell momentum 3160-step on a Muon tuned baseline result (n=8)
356 2026-08-22 closed merged=- jacknzheng | Track3 kmaxwell
353 2026-08-14 open merged=- konstmish | Tune AdamW baseline to 4950 steps
351 2026-08-10 open merged=- Yufei-Gu-451 | MuonH fast-slow-decay schedule: 3125 steps (n=20, sig=0.00450)
350 2026-08-06 closed merged=2026-09-18 jvarho | Track 1: Canonical token masking (-0.8s?, -15 steps)
349 2026-08-06 closed merged=- devenpzak | New record: 1.079 min (64.72s) — new ANVIL optimizer + full-stack fp8 and schedule overhaul (−9.3s / −12.6% same-hardware)
348 2026-07-29 open merged=- jon123boss | Track 1: Topological Layer Dropout (-0.2, +8 steps)
347 2026-07-28 open merged=- deepmatmul | Track 1: terminal-blended readout TailEMA (1.261 min, -0.895%)
346 2026-07-24 open merged=- katop1234 | New record: 1384 steps / 1.312 min — Ember optimizer on embedding tables + earlier context window extension
345 2026-07-22 open merged=- mangocrazz | Track 3: MuonH power-0.4 schedule reaches 3.28 at 3150 steps
344 2026-07-21 closed merged=2026-09-18 Glitchfix | Track 1 (1.167 min on 8xH100, -5.15% same-node): Reduced-width QK and packed FP8 attention
343 2026-07-20 open merged=- zzp1012 | Track 3 (Per-optimizer SOTA): MuonH with minus-sqrt LR schedule — 3175 steps (n=9)
342 2026-07-18 closed merged=2026-08-02 Mister-dev-oss | Track 1: Add FP8 MLP down projection (-1.04s)
341 2026-07-17 open merged=- jn2clark | Track 3 bi-maxwell kfac 2600 steps (n=12)
340 2026-07-15 open merged=- orange4664 | Track 3: bi-Maxwell momentum on the tuned Muon baseline -- 3210 steps (n=8)
339 2026-07-14 open merged=- orange4664 | Track 3: Bi-Maxwell dual-timescale momentum -- 2635 steps (n=8)
338 2026-07-14 open merged=- jn2clark | Track 3: Projected Tail-EMA, 2645 steps (n=19)
337 2026-07-13 closed merged=2026-08-02 jvarho | Track 1: Prefix token prediction (-0.78s, -15 steps)
336 2026-07-11 open merged=- vfedosov77 | Improve shifted keys
```

## Primary-source reads of PR bodies (verbatim excerpts)

### PR 379: Track 1: CPLM on #360 (-4.57s, -11.25% same hardware)

- Fetch: 2026-10-03T10:47:03.515483Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/379
- State: open, merged_at=None

```
# New record: CPLM on record #92 — 36.009 s (0.600 min)

Paper coming soon! :)

GPT-2 (124M-class) on FineWeb10B, 8×H100-80GB, track 1 (≤ 3.28 val CE). Built on record #92 (ANVIL2, [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360)); everything in #92 is unchanged except the output distribution and the step count.

**Result (8×H100 SXM, same node, interleaved with record #92):**

| | runs | train time (mean ± sd) | final val CE (mean ± sd) |
|---|---|---|---|
| **this PR (CPLM, 1050 steps)** | **8** | **36.009 ± 0.039 s** | **3.2769 ± 0.0016** |
| record #92 (1194 steps), same node, interleaved | 13 | 40.575 ± 0.024 s | 3.2765 ± 0.0016 |
| **delta** | | **−4.566 s (11.25 %)** | +0.33 millinats |

- One-sided t-test vs 3.28: t = −5.39, **p = 0.0005** (7 dof). All runs of the shipped configuration count; none were excluded.
- Record #92 is 39.9 s as published; on this node it measured 40.575 s, so this node is ~1.7 % slower than #92's. The speedup above is the same-node comparison (rule 4).
- Logs: `this_pr/` (8 runs) and `baseline/` (13 runs) in `records/track_1_short/2026-10_CPLM/`.

**Cost of CPLM over #92** (analytic estimates from the shapes, per rank):

| | added | relative |
|---|---|---|
| parameters | 196,737 (2 × 128 × 768 + 128 + 1) | +0.35 % of the transformer blocks' matrices (~56 M); +0.15 % with embed + lm_head (~133 M); ~0 % of the 65 B total with the n-gram table |
| training memory | ≈ 0.1 GB at the largest batch (49,152 tokens/rank): pointer q/k and their pre-norm inputs saved for backward, fp32 dq/dk transients, ~3 MB of params + grads + optimizer state. The pointer kernel is flash-style, so no score matrix is stored | ≈ +0.2 % of #92's 48.4 GB peak |
| inference (validation) memory | ≈ 0.13–0.4 GB per 262,144-token validation batch (bf16 q/k plus fp32 norm temporaries); 8192-token band, still flash-style | ≤ +0.8 % of #92's peak |
| compute | ≤ 1.4 MFLOP/token forward (q/k projections 0.4 M + pointer scores ≤ 1.0 M at the full 2 × 2048 band; document masking makes it less in practice), ≤ 4.3 MFLOP/token for training | ≤ ~0.8 % of the model's ~0.5 GFLOP/token training cost |
| wall time per step | measured +1.3 % per step on 4×GH200; consistent with ~1 % here | |

The per-step overhead is small next to the 12 % fewer steps (1194 → 1050), which nets out to the measured 11.25 %.

## Change

The next-token distribution becomes a mixture of the LM softmax and a pointer over the document's previous tokens:

    p(y) = p_lm(y) · (1 − α(1 − a_sink)) + α · p_copy(y)

- **Gate α**: the softmax 
```

### PR 378: Track 1: Document-local copy feature (-1.17 s, -2.9% vs #360; stacked on #375)

- Fetch: 2026-10-03T10:47:16.436841Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/378
- State: open, merged_at=None

```
Each position gets a **candidate next token**: the token that followed the longest earlier exact match of its (normalized) context in the **same document**, fed to the model as an extra input during training. This combines #376's signal (document-local repeats, used there at the final validation) with #367's retrieval injection (the matched continuation as a model input). Stacked on open PR #375, whose two commits it contains unchanged.

## Results (8xH100 SXM, one node, interleaved, unseeded)

| | runs | steps | val loss (mean +- 95% CI) | train time (s) |
|---|---|---|---|---|
| **this PR** | 8 | 1132 | **3.27548 +- 0.00057** | **39.181 +- 0.099** |
| record #360 (master), same node | 4 | 1194 | 3.27540 (the 2 unaffected runs; see notes) | 40.355 +- 0.155 |
| **delta** | | -62 | | **-1.174 s (-2.91%)**; -1.116 s (-2.77%) against the 2 unaffected #360 runs alone |

- Mean val loss <= 3.28: one-sided t-test over all 8 runs, t = -18.65, df = 7, **p = 1.6e-07**.
- Faster than #360: one-sided Welch t-test on all runs, t = -18.26, df = 7.3, **p = 1.1e-07**.
- #375 alone measured -0.42 s on its own node, and it also lowered val loss by about 2 millinats, which this PR spends too. #375 alone was not run on this node, so the copy feature's share is roughly 0.5 to 0.75 s.

All 13 runs (including one launch that crashed at startup) are listed in execution order with their full logs in `records/track_1_short/2026-10-02_DocCopy/`, together with the details below.

## What it does

- **Matcher** (`track_1_short/doc_copy.py`, inside the compiled forward, from `input_seq` alone):
  - For 10 match lengths (1, 2, 3, 4, 6, 8, 12, 16, 24, 32), each position's context is hashed (a rolling hash over #375's normalized ids) into a 64-bit key mixed with the length and the document index.
  - One stable sort finds each position's latest earlier occurrence of the same key. The longest match wins, and the candidate is the raw token after it.
  - The bucket (31 values) encodes the length and how often the context occurred before (1, 2, 3+).
- **Model.** `copy_vec = embed(candidate) * scale[bucket] + bucket_embed[bucket]`, added at the input (after the smear) and before the final norm with two learned gains. This follows #367's retrieval injection (its gain init 0.5 and 1.0) at two of its three sites. The new tensors are 31 scales, 31 x 768 bucket embeddings and 2 gains. The scales and bucket embeddings start at zero, so the feature starts switched off.
- **Causality.** The match must end strictly before t, and the code enforces it (`prev < t`). The runs were made without the expli
```

### PR 375: Track 1: Normalize tokens in n-gram embeddings to improve information density (-0.42s, -1.02%, -15 steps)

- Fetch: 2026-10-03T10:47:04.261217Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/375
- State: open, merged_at=None

```
 Normalize tokens in n-gram embeddings to improve information density (-0.42s, -1.02%, -15 steps)

Following Deepseek's [Engram](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf) paper, this PR normalizes tokens prior to obtaining the hashes for the n-gram embeddings by removing punctuation, whitespace, and capitalization from their textual representation. The idea is to improve information density by hashing similar n-grams together and reducing collisions from unrelated n-grams. This enables us to shed ~15 steps while reducing val loss against master (though based on the validation loss reduction of two millinats).

I was not able to achieve any further gain from more aggressive normalization methods (e.g., removing suffixes and punctuation tokens) and believe there is little gain left in this direction. However, after this normalization, 64.6% of bigram slots and 8% of trigram slots are never hit in a full training run, suggesting an avenue to reduce memory consumption at equal validation loss or reduce validation loss at equal memory consumption.

Results are on 8×H100. There are 7 runs of this PR and 7 of master commit 4ea6b93.

## Results (8×H100)

Summary (mean ± 95% CI half-width):

| | Steps | Val loss | Train time (s) |
|---|---|---|---|
| **This PR** | 1179 | **3.27350 ± 0.00148** | **40.588 ± 0.198** |
| Master | 1194 | 3.27554 ± 0.00143 | 41.008 ± 0.062 |

| Test | Result |
|---|---|
| This PR's mean val loss ≤ 3.28 (one-sided t) | t = 12.5, **p = 1.9e-5** |
| Val loss, PR vs master (Welch, one-sided) | p = 0.016 |
| Train time, PR vs master (Welch, one-sided) | p = 0.0008 |
| Time reduction | **0.420 s (1.02%)** |

| Run | Val loss: this PR | Val loss: master | Train time (s): this PR | Train time (s): master |
|---|---|---|---|---|
| 1 | 3.2718 | 3.2775 | 40.508 | 40.889 |
| 2 | 3.2761 | 3.2773 | 40.383 | 41.039 |
| 3 | 3.2731 | 3.2760 | 40.965 | 40.955 |
| 4 | 3.2725 | 3.2737 | 40.800 | 41.080 |
| 5 | 3.2733 | 3.2743 | 40.557 | 41.003 |
| 6 | 3.2753 | 3.2741 | 40.441 | 41.022 |
| 7 | 3.2724 | 3.2759 | 40.462 | 41.069 |




## Methodology

| | |
|---|---|
| Hardware | One 8× H100 80GB HBM3 (SXM) node for all 14 runs, one session |
| Order | Interleaved: PR, master, PR, master, … |
| Seeds | Unseeded (random init); every run counted |
| Environment | The repo's `Dockerfile` and `run.sh` command; data at `data/fineweb10B` |


## Abridged runs (RTX PRO 6000 Blackwell Server Edition)

Before the full runs, I ran abridged experiments on one NVIDIA RTX PRO 6000 Blackwell Server Edi
```

### PR 372: Track 1: Adaptive softmax (-2.3s, +35 steps)

- Fetch: 2026-10-03T10:47:04.884393Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/372
- State: closed, merged_at=None

```
The new version takes 65.3 seconds to reach the target loss at step 1325.

Count token occurrences in the first 20M tokens inside the timed training run and for most tokens compute softmax over only the most frequent 18430 tokens + two tail cluster logits. Put the other tokens into the two tail clusters, 16384 and 15443 tokens large. For target tokens that fall into these two tail clusters, compute a cluster logit as part of the first softmax plus additional within-cluster softmax terms.

Ablations included head clusters with 16382 and 20478 tokens and a 12286 head cluster with full softmax from stage 2 onward.

I have also run the combined version with https://github.com/KellerJordan/modded-nanogpt/pull/367 and https://github.com/KellerJordan/modded-nanogpt/pull/366 (see the checked out logs):


Variant | The first probe under 3.28 | Step 688
-- | -- | --
[Hashed n-gram](https://github.com/KellerJordan/modded-nanogpt/pull/366), [Longest exact-match](https://github.com/KellerJordan/modded-nanogpt/pull/367), Adaptive softmax | step 660, 3.2765, 45.2 s | 3.2604, 47.2 s
[Longest exact-match](https://github.com/KellerJordan/modded-nanogpt/pull/367), Adaptive softmax | step 670, 3.2780, **41.4 s** | 3.2684, 42.7 s
[Hashed n-gram](https://github.com/KellerJordan/modded-nanogpt/pull/366), Adaptive softmax | step 688, 3.2790, 50.3 s | 3.2790, 50.3 s


```

### PR 371: Track 1: Sampled LM-head gradients and truncated attention backward (-0.76s, -1.88%)

- Fetch: 2026-10-03T10:47:05.597698Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/371
- State: open, merged_at=None

```
Adds two backward approximations on top of #360. Both reduce work per update
for part of training, then switch back to the original backward for the finish.

**LM-head weight gradient.** Split the logit gradient into positive and
negative entries. The positive part is dense, so sample one activation/gradient
pair per group of four token rows and multiply its contribution by four. This
reduces the dense dW GEMM's reduction dimension by 4×. Negative entries occur
only at target positions and are accumulated separately without sampling.
The head's input gradient is unchanged, so all tokens still contribute to
training the preceding layers.

**Attention backward.** Some heads can use a shorter backward window with
little change to their Q/K/V gradients. At steps 592 and 593, compare windows
of 64, 128 and 256 tokens against the original backward. For each eligible
head, choose the cheapest window with measured relative L2 error ≤0.2 for
all three gradients on both steps and every rank; otherwise keep the full
window. The kernel skips distant backward tiles while using the original
forward output and softmax normalization. This biases the gradient; retained
attention probabilities are not renormalized.

Head sampling runs on steps 320–1106 and attention truncation on 594–1106.
The final 87 updates use the original backward. Calibration is included in
training time. The forward pass, 1,194-update schedule and full-vocabulary
validation are unchanged from #360.

Both methods combined, compared with unmodified #360 on the same 8×H100 SXM node
(mean ± sample SD):

| Run | N | Time (s) | Validation loss | p (mean < 3.28) |
|---|---:|---:|---:|---:|
| #360 | 6 | 40.342 ± 0.184 | 3.279000 ± 0.004737 | 0.3136 |
| This PR | 12 | 39.583 ± 0.266 | 3.278705 ± 0.001276 | 0.00242 |

This saves **0.759 s (1.88%)** and passes the one-sided loss test at p < 0.01.

```

### PR 367: Track 1: Longest exact-match retrieval (40.0s -> 21.56s / -46.1%)

- Fetch: 2026-10-03T10:47:12.606537Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/367
- State: open, merged_at=None

```
Author: Herman Brunborg (X: [@hermanbrunborg](https://x.com/hermanbrunborg))

In this PR, we train a second model for longest exact-match retrieval, trained on the full 10.3B tokens using the CPU. It contains (lossy) 6-grams, 10-grams and 20-grams. The exact matches are provided as hints for the LM-model. We train the retrieval model in parallel with the language model in around 10s.

|  | runs | wall (training section) | final val CE |
| --- | --- | --- | --- |
| **this PR** | 18 | **21.555 +/- 0.026 s** | **3.27489 +/- 0.00261** |
| record #92 (ANVIL2, [#360](https://github.com/KellerJordan/modded-nanogpt/pull/360)), same machine, in between record runs | 9 | 40.001 +/- 0.040 s | 3.27927 +/- 0.00373 |
| **delta** |  | **-18.45 s, -46.1%, 1.86x** |  |

p-value: 1.1e-7

Current on top of #374 

AI use disclosure: AI agents wrote virtually all code in the PR and during exploration, but almost all of the ideation, exploration and profiling was done by hand.  I found AI to be incredibly useful in implementing ablations, making diagrams and implementing even vague ideas, but only around -1s came from auto research/AI ideas.
```

### PR 366: Track 1: Hashed n-gram rows from host RAM (-6.16s)

- Fetch: 2026-10-03T10:47:06.258267Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/366
- State: open, merged_at=None

```
# Track 1: Hashed n-gram rows from host RAM (-6.16s)


This is a port of my nanochat submission ([karpathy/nanochat#854](https://github.com/karpathy/nanochat/pull/854)) to the modded-nanogpt
speedrun: the same idea (hashed n-gram rows in host RAM injected into attention values, trained with a row-wise optimiser and async table updates).

## How this differs from the nanochat version

- Hashing, gathering and write-back run on dedicated worker threads and the copies are explicitly scheduled to avoid contention.
- No up-projection; n-gram values are injected directly into a single head's attention values. 
- Triton kernels for the row update and the row gather.
- 8 out of 11 layers using 80M rows vs 4 out of 24 layers using 300M rows.

## Results

12 candidate and 12 baseline runs, interleaved on the same 8x H100 node. Each run is the standard
command in its own checkout:

```
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Baseline = upstream master ecbb586, unmodified (1,285 steps); candidate = this branch (1,140 steps).

| | Baseline | This PR |
|---|---|---|
| Mean val loss | 3.2777 | 3.2785 |
| Mean train time | 74.03 s | 67.87 s |
| p(mean<3.28) | 0.0010 | 0.0047 |



## Logs, environment and per-run values

Logs: `records/track_1_short/2026-09-15_EngramHostTable/this_pr/` and `baseline/`.

Environment: 8x H100, PyTorch 2.10.0+cu128, Triton 3.6.0, CUDA 12.8. The table needs ~41 GB of
host RAM in `/dev/shm`.


Baseline, val loss and train times:

```python
[3.2769, 3.2756, 3.2759, 3.2768, 3.2801, 3.2820, 3.2760, 3.2781, 3.2771, 3.2792, 3.2787, 3.2754]

[73.944, 73.981, 74.109, 74.102, 74.101, 74.016, 73.980, 74.094, 74.001, 74.065, 73.980, 73.955]
```

This PR, val loss and train times:

```python
[3.2766, 3.2787, 3.2776, 3.2785, 3.2796, 3.2782, 3.2760, 3.2817, 3.2781, 3.2786, 3.2775, 3.2810]

[67.965, 68.062, 67.918, 67.827, 67.738, 67.815, 67.723, 67.846, 67.858, 68.089, 67.714, 67.870]
```

## Acknowledgements

I would like to thank [Oriole Networks](https://oriolenetworks.com/), especially my colleagues Alessandro
Ottino and Robin Matzner, for the discussions and support.

```

### PR 346: New record: 1384 steps / 1.312 min — Ember optimizer on embedding tables + earlier context window extension

- Fetch: 2026-10-03T10:47:13.393002Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/346
- State: open, merged_at=None

```
# New record: 1384 steps / 1.312 min — Ember optimizer on embedding tables + earlier context window extension

## What changed (one file, +113/−35 vs the current record)

1. **Ember on `value_embeds` and `embed`** ([github.com/katop1234/ember](https://github.com/katop1234/ember)):
   Adam's dense per-coordinate second moment on the two embedding tables is replaced by a
   factored reconstruction `v = outer(r, c) / s` (per-row EMA × globally-pooled per-column
   EMA), with β₁=0 (no momentum buffer). Optimizer state on 46% of the model's parameters
   drops from ~1.86 GB to ~1 MB; distributed cost is one ~768-float `all_reduce` per table
   per optimizer step (~3 KB total) on a dedicated side communicator; stats use
   contiguous reductions only (no atomics), so
   updates are deterministic at fixed world size. `value_embeds` lr ramps 0.64×→1.0× over
   updates 500–690; `embed` runs at a flat 0.5×, cold-started at the untie. `lm_head`
   keeps dense Adam. Measured peak VRAM drops 2.9 GB/GPU (paired runs),
   though at this scale that's a side benefit rather than a speed lever.
2. **Attention windows step to (8,20) at step 1300** instead of at the 1380 extension
   (stage split in `TRAINING_STAGES`; YaRN fires at 1300; the extension-stage transition
   and final-val ws=20 extension become no-ops).
3. **`num_extension_iterations` 10 → 4**: total steps 1390 → 1384. The Muon momentum
   cooldown stays anchored to the 1390-step clock (the run stops partway down the same
   ramp it was tuned on).

## Result (n=20, logs attached, unmodified `run.sh`, byte-identical source in every log)

- Final val loss: **mean 3.27863, sd 0.00157**, 15/20 individually ≤ 3.28 —
  `ttest_1samp(finals, 3.28, 'less')`: **t = −3.89, p = 0.00049** (bar: p < 0.01).
- Timing (measured, not derived): **mean 78.73 s** (min 78.58, max 78.84) across both
  machines. On 8×H100 from PrimeIntellect the mean is **78.68 s vs 78.80 s / 79.10 s for stock on the same box**
  (baseline logs attached in `baseline/`), i.e. faster than the same-hardware stock
  baseline as well as the current record row's 1.320 min. (Comparison base: the 79.2 s /
  1390-step stock code in this repo, i.e. the README row for record 84.)

## Engineering notes

- **State is small enough to skip sharding.** Row stats slice with the shard; column/
  global stats are replicated. Checkpoints no longer depend on world size, and the
  untie's Adam state move (all-gather + transpose + reshard, ~300 MB) is deleted
  outright.
- **NCCL runs collectives in issue order per communicator.** The 3 KB stats all-reduce
  queued behind the
```

### PR 370: Track 3: IsoMuon — Muon with a noise-calibrated diagonal response metric; 3.28 in 3190 steps (n=8)

- Fetch: 2026-10-03T10:47:14.100781Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/370
- State: open, merged_at=None

```
## TL;DR
IsoMuon takes Muon's Newton–Schulz polar decomposition in a diagonal response metric calibrated by the gradient's own sampling-noise variance: each output row and input column is weighted by the square root of its relative noise (bounded), so noisier channels are damped more. It adds two vector EMAs per matrix and negligible compute. Its two constants (λ = 0.5, c = 2) were chosen in a few single-seed runs during development (see the ablations) and were not tuned on this benchmark. On the tuned Muon + aux AdamW baseline (result #36, 3250 steps) it reaches 3.28 at **3190 steps** (mean 3.27842, n=8 non-cherry-picked seeds 0–7, `(3.28 − mean)·√8 = 0.00446 ≥ 0.004`). With bi-Maxwell momentum (PR #339) in the same script, IsoMuon keeps 93% of its gain and the two changes are close to additive (8 paired seeds per arm, 2026-09-29 update).

## What is new
Muon's orthogonalization moves every singular direction of the update at unit speed, whether or not its coordinates are noisy. In a language model the gradient of a hidden matrix is dominated by label-sampling noise, and its variance is strongly non-uniform across output channels (rows) and input channels (columns). IsoMuon measures that variance at no extra cost from the micro-batch gradients that every step already computes, and whitens the momentum update by a diagonal metric: each channel's relative noise raised to the power λ and clipped to [1/c, c],

    B = clamp((row_heat / mean)^λ, 1/c, c),  A = clamp((col_heat / mean)^λ, 1/c, c),  λ = 0.5, c = 2
    D = B^-1/2 · polar(B^-1/2 U A^-1/2) · A^-1/2,   ‖D‖_F re-aligned to ‖polar(U)‖_F

where `row_heat`/`col_heat` are EMAs (β = 0.95) of the row/column means of `Var_k[g_k]` over the micro-batches `k` of the step. With λ = 0 the metric is the identity and IsoMuon reduces to Muon up to the final Frobenius re-alignment, a scalar step-size factor. With λ = 0.5 each channel's damping is proportional to the standard deviation of its gradient noise rather than to its variance.

The estimator does not depend on the number of GPUs: the per-micro-batch squared-gradient row/column sums are all-reduced together with the gradient, so 1, 2, 4 or 8 GPUs give the same expected noise estimates.

## Origin: the endpoint-metric polar decomposition (EMP)

IsoMuon grew out of an earlier experiment of ours that was never submitted on its own, so we describe it here.
It was run on an older setup: a Muon trainer derived from the track-3 baseline script `train_gpt_simple.py`, run for 3500 steps on one A800 per run.

Muon's polar decomposition fixes the singular directions of 
```

### PR 369: Track 3: improve Muon with MaxEntSlop momentum and Defazio-style decay. 3060 steps (-100 from PR#357)

- Fetch: 2026-10-03T10:47:15.738577Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/369
- State: open, merged_at=None

```
## I'm sorry this method is ugly

> Hi reader, sorry this method is still slop. It is more principled than k-maxwell from #357 and #359.
> It is not yet a complete theory of momentum, nor would I necessarily recommend it yet outside nanogpt.
> I'm PRing this since the gains appear to be legitimate and I want others in the community to be able
> to build on top of these findings while I figure out why this works. -- @jeffreycider

## How we changed momentum

Single-decay EMA is less expressive than the optimal momentum implementation. Every momentum implementation assigns some weight $w_{t-k}$ to each lagged gradient $g_{t-k}$. Let's call $w_{t-k}$ the momentum *kernel*.

Hu et al found a win with Bi-maxwell, which sets the kernel as the linear combo of 2 EMA momentum buffers with different decays; @jacknzheng and I previously developed an extreme extension $k$-maxwell — a linear combo of $k$ momentum buffers — as a proof of concept that there is room for improvement via both added kernel expressivity and kernel adaptivity.

This latest method sets a new SOTA and with far fewer hyperparameters: 8 instead of 3k. Instead of parameterizing the kernel as a sum of many EMA buffers and tuning their decays / linear combo weights, we tune 4 sufficient statistics for the momentum kernel distribution and solve for the weights directly before pretraining begins. On Muon I also use Defazio-style weight decay: the weight-decay coefficient follows the learning-rate schedule, so decay anneals with the step size instead of staying fixed. Here is a gif showing how these three momentum kernels differ:

![momentum kernels over training](https://raw.githubusercontent.com/jeffreycider/modded-nanogpt/track3-muon-maxentslop/records/track_3_optimization/results/20260921_muon_maxentslop_3060/momentum_kernels.gif)

## Aspirations

My original goal was to distill the essence of why k-maxwell provides pretraining gains, but I accidentally got empirical wins before I could develop a robust theory of momentum.

I solve for the maximum entropy momentum kernel given the sufficient statistics. This is an arbitrary design choice. I am also not happy with the fact that we still have to tune a starting momentum kernel and an ending momentum kernel to linearly anneal between them. To voice my dissatisfaction at this method, I am calling it MaxEntSlop.

I hope to replace this method in a few weeks. This PR is to communicate the existing empirical win to the nanogpt community in case I fail.

## Result

![validation loss over training, mean and 95% confidence interval over eight seeds](https:/
```

### PR 377: Track 3 anchor gradient + fp32 embed 2580 steps (n=16)

- Fetch: 2026-10-03T10:47:14.944586Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/377
- State: open, merged_at=None

```
# Anchor-extrapolated gradient + fp32 master embedding, 2580 steps

## Result

Sixteen fixed GH200 seeds pass Track 3 at **2580 optimizer steps**.

The score is:

`margin = (3.28 - mean_loss) * sqrt(number_of_seeds)`

| Step | Seeds | Mean loss | Margin | Required margin | Result |
|---:|---:|---:|---:|---:|:---|
| 2575 | 16 | 3.27926000 | 0.00296000 | 0.00400000 | fail |
| 2580 | 16 | 3.27896875 | 0.00412500 | 0.00400000 | pass |

The maximum passing mean for 16 seeds is `3.27900000`, so the measured mean
clears it by `0.00003125`.

Per-seed results are in `summary.tsv`. Raw logs are `GH200_seed0.txt` through
`GH200_seed15.txt`.

At the same step as PR #341 (2600), the mean is `3.27778313` versus
`3.27877000` (n=12), a difference of `0.00099`. That is not pairwise
significant at these seed counts.

## Method

The trainer is the PR #341 trainer with two changes to the late phase and a
retuned readout.

**Anchor-extrapolated gradient.** Let `e` be the Tail-EMA of the weights
(`e = e + (w - e) / 100`, from step 2040). From step 2100, each step's single
forward-backward pass is taken at

`y = w + g_t * (w - e)`

and the resulting gradient is applied to `w` by the unchanged optimizer. `g_t`
ramps linearly from `0` at step 2100 to `0.45` at step 2200. This is a
Nesterov-style lookahead whose displacement is the drift away from the weight
average used at readout.

**fp32 master embedding.** The token embedding is a bf16 parameter with bf16
Adam state. By step 2000 its entries reach `|w| ~ 9`, where most tail Adam
updates are below half a bf16 ulp and round away. The optimizer keeps an fp32
master copy and fp32 Adam state for it and writes the bf16 rounding of the
master to the parameter. The parameter dtype and forward pass are unchanged.
The embedding is included in the anchor `e` (readout blend `0`).

**Readout.** Tail-EMA `tau` is `100`. The fixed blends are `0.80` for
first-block matrices, `0.55` for other block matrices, `1.00` for auxiliary
parameters, and `0.50` for the output projection.

The submitted trainer SHA-256 is
`6ddd8ab6500740817bdf8d6817f197e76f1f17ddb1364baa6b0e24458c4fb6ec`.

## Reproduce

```bash
STOP_STEP=2580 torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/results/20260930_anchor_fp32embed_2580/train_gpt_anchor_fp32embed_2580.py \
  --seed 0
```
```

