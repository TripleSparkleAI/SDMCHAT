# C_q02 modded-nanogpt PRs after #360 touching head, softmax, embeddings, output distribution

- query: GET https://api.github.com/repos/KellerJordan/modded-nanogpt/pulls?state=all&sort=created&direction=desc&per_page=40 then /pulls/N for 372 371 346 377 379 367 376 378 375 366
- tool: python3 urllib, GitHub REST API unauthenticated
- fetched (UTC, HTTP status): 2026-10-03T10:47:32.778909Z 200

## PR list (verbatim number state created merged author title)

- #379 open created 2026-10-02 merged  NathanGodey | Track 1: CPLM on #360 (-4.57s, -11.25% same hardware)
- #378 open created 2026-10-02 merged  josusanmartin | Track 1: Document-local copy feature (-1.17 s, -2.9% vs #360; stacked on #375)
- #377 open created 2026-10-01 merged  jn2clark | Track 3 anchor gradient + fp32 embed 2580 steps (n=16)
- #376 open created 2026-09-29 merged  cyrusghane | Track 1: Document-local copy mixture at the final validation (-15 extension steps, ~-0.8s)
- #375 open created 2026-09-29 merged  daniel-monroe | Track 1: Normalize tokens in n-gram embeddings to improve information density (-0.42s, -1.02%, -15 steps)
- #374 closed created 2026-09-28 merged 2026-09-28 ClassicLarry | README: update for record #360 (ANVIL2)
- #373 closed created 2026-09-28 merged 2026-09-28 ClassicLarry | Track 1: refactor the ANVIL2 trainer into a readable package (no record change)
- #372 closed created 2026-09-28 merged  ldmberman | Track 1: Adaptive softmax (-2.3s, +35 steps)
- #371 open created 2026-09-25 merged  romeerp | Track 1: Sampled LM-head gradients and truncated attention backward (-0.76s, -1.88%)
- #370 open created 2026-09-23 merged  orange4664 | Track 3: IsoMuon — Muon with a noise-calibrated diagonal response metric; 3.28 in 3190 steps (n=8)
- #369 open created 2026-09-20 merged  jeffreycider | Track 3: improve Muon with MaxEntSlop momentum and Defazio-style decay. 3060 steps (-100 from PR#357)
- #368 open created 2026-09-20 merged  jeffreycider | Track 3: improve MuonH with MaxEntSlop momentum and a fixed weight-parallel fraction. 3010 steps (-55 from PR#359)
- #367 open created 2026-09-17 merged  hermabr | Track 1: Longest exact-match retrieval (40.0s -> 21.56s / -46.1%)
- #366 open created 2026-09-15 merged  nihir27 | Track 1: Hashed n-gram rows from host RAM (-6.16s)
- #364 closed created 2026-09-05 merged  dnhkng | Endgame EMA weight blending (port of Track 3 Tail-EMA readout)
- #363 open created 2026-09-03 merged  isubuz | Track 1: FP8 MLP input gradient and kernel tuning by AI System Warpscale (-0.45s)
- #362 open created 2026-09-01 merged  jingru-lee | Track 3: Muon-NSR reaches 3.28 in 3250 steps (n=8)
- #361 closed created 2026-09-01 merged  jingru-lee | Track 3: Muon-NSR reaches 3.28 in 3250 steps (n=8)
- #360 closed created 2026-08-31 merged 2026-09-28 devenpzak | New Record: 0.665 minutes (39.9 seconds): ANVIL2, Sampled-softmax, New Embedding Table, Full-stack fp8 (-34.0s, -46% same hardware)
- #359 open created 2026-08-28 merged  jacknzheng | Track3: K-maxwell momentum muonh decay - 3065 steps (n=8)
- #358 closed created 2026-08-27 merged  matevz-kovacic | Track 1: Norm CSE, FP8 up-proj quantize (-0.729s)
- #357 open created 2026-08-27 merged  jacknzheng | Track 3: K-Maxwell momentum 3160-step on a Muon tuned baseline result (n=8)
- #356 closed created 2026-08-22 merged  jacknzheng | Track3 kmaxwell
- #353 open created 2026-08-14 merged  konstmish | Tune AdamW baseline to 4950 steps
- #351 open created 2026-08-10 merged  Yufei-Gu-451 | MuonH fast-slow-decay schedule: 3125 steps (n=20, sig=0.00450)
- #350 closed created 2026-08-06 merged 2026-09-18 jvarho | Track 1: Canonical token masking (-0.8s?, -15 steps)
- #349 closed created 2026-08-06 merged  devenpzak | New record: 1.079 min (64.72s) — new ANVIL optimizer + full-stack fp8 and schedule overhaul (−9.3s / −12.6% same-hardware)
- #348 open created 2026-07-29 merged  jon123boss | Track 1: Topological Layer Dropout (-0.2, +8 steps)
- #347 open created 2026-07-28 merged  deepmatmul | Track 1: terminal-blended readout TailEMA (1.261 min, -0.895%)
- #346 open created 2026-07-24 merged  katop1234 | New record: 1384 steps / 1.312 min — Ember optimizer on embedding tables + earlier context window extension
- #345 open created 2026-07-22 merged  mangocrazz | Track 3: MuonH power-0.4 schedule reaches 3.28 at 3150 steps
- #344 closed created 2026-07-21 merged 2026-09-18 Glitchfix | Track 1 (1.167 min on 8xH100, -5.15% same-node): Reduced-width QK and packed FP8 attention
- #343 open created 2026-07-20 merged  zzp1012 | Track 3 (Per-optimizer SOTA): MuonH with minus-sqrt LR schedule — 3175 steps (n=9)
- #342 closed created 2026-07-18 merged 2026-08-02 Mister-dev-oss | Track 1: Add FP8 MLP down projection (-1.04s)
- #341 open created 2026-07-17 merged  jn2clark | Track 3 bi-maxwell kfac 2600 steps (n=12)
- #340 open created 2026-07-15 merged  orange4664 | Track 3: bi-Maxwell momentum on the tuned Muon baseline -- 3210 steps (n=8)
- #339 open created 2026-07-14 merged  orange4664 | Track 3: Bi-Maxwell dual-timescale momentum -- 2635 steps (n=8)
- #338 open created 2026-07-14 merged  jn2clark | Track 3: Projected Tail-EMA, 2645 steps (n=19)
- #337 closed created 2026-07-13 merged 2026-08-02 jvarho | Track 1: Prefix token prediction (-0.78s, -15 steps)
- #336 open created 2026-07-11 merged  vfedosov77 | Improve shifted keys

## #372 Track 1: Adaptive softmax (-2.3s, +35 steps)
- fetched 2026-10-03T10:47:39.766361Z 200; state closed; created 2026-09-28T13:16:18Z; closed 2026-10-01T21:31:02Z; merged None

The new version takes 65.3 seconds to reach the target loss at step 1325.

Count token occurrences in the first 20M tokens inside the timed training run and for most tokens compute softmax over only the most frequent 18430 tokens + two tail cluster logits. Put the other tokens into the two tail clusters, 16384 and 15443 tokens large. For target tokens that fall into these two tail clusters, compute a cluster logit as part of the first softmax plus additional within-cluster softmax terms.

Ablations included head clusters with 16382 and 20478 tokens and a 12286 head cluster with full softmax from stage 2 onward.

I have also run the combined version with https://github.com/KellerJordan/modded-nanogpt/pull/367 and https://github.com/KellerJordan/modded-nanogpt/pull/366 (see the checked out logs):


Variant | The first probe under 3.28 | Step 688
-- | -- | --
[Hashed n-gram](https://github.com/KellerJordan/modded-nanogpt/pull/366), [Longest exact-match](https://github.com/KellerJordan/modded-nanogpt/pull/367), Adaptive softmax | step 660, 3.2765, 45.2 s | 3.2604, 47.2 s
[Longest exact-match](https://github.com/KellerJordan/modded-nanogpt/pull/367), Adaptive softmax | step 670, 3.2780, **41.4 s** | 3.2684, 42.7 s
[Hashed n-gram](https://github.com/KellerJordan/modded-nanogpt/pull/366), Adaptive softmax | step 688, 3.2790, 50.3 s | 3.2790, 50.3 s



### comment ldmberman 2026-10-01T21:31:02Z

The new sampled softmax approach beats adaptive softmax at the first glance.

Also, I mistakenly changed the evaluation procedure in this patch.

## #371 Track 1: Sampled LM-head gradients and truncated attention backward (-0.76s, -1.88%)
- fetched 2026-10-03T10:47:46.456847Z 200; state open; created 2026-09-25T18:45:25Z; closed None; merged None

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


### comment varunneal 2026-09-28T16:08:50Z

Really cool

## #379 Track 1: CPLM on #360 (-4.57s, -11.25% same hardware)
- fetched 2026-10-03T10:47:49.513204Z 200; state open; created 2026-10-02T17:38:54Z; closed None; merged None

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

- **Gate α**: the softmax mass of a `<copy>` slot, vocab id 50257 (one of lm_head's existing padding rows), under the same softcap as the other logits. No new output parameters; `p_lm` is renormalized over the real vocabulary.
- **Pointer `p_copy`**: one head (d = 128) over the final hidden state. It attends with document masking over previous tokens (a 2048-token band in training) plus a learned sink key. `p_copy(y)` is the attention mass on previous positions holding token `y`, so the pointer only copies tokens already seen (causal). The sink mass `a_sink` goes back to the LM.
- **QK-norm**: queries and keys are RMS-normed, with a learnable query gain. Without it the pointer logits blow up on some seeds.
- **Kernels**: in training the mixture runs inside #92's fused fp8 CE kernel (`perf/kernels/cplm_cross_entropy.py`), including the sampled-softmax stages. The pointer is a fused flash-style Triton kernel (`cplm_copy.py`).
- **Validation**: the same mixture over #92's full-softmax, canonically masked logits (`<copy>` is exempt from the mask), with an 8192-token pointer band.
- **Steps**: 1194 → 1050 (`NUM_SCHEDULED_ITERATIONS` 1122 → 978; the 72 growth + extension steps are unchanged). Stage boundaries scale with the schedule as in #92.

`CPLM=0 NUM_SCHEDULED_ITERATIONS=1122 ./run.sh` runs record #92 from the same tree; this is how the baseline pool was produced.

## Validity of the probability model

The README defines the target as a valid probability model over the val set. The mixture sums to 1 over the vocabulary: (1 − α + α·a_sink) on the LM side plus α(1 − a_sink) on the pointer side. Pointer mass that lands on canonically masked tokens is dropped, so on feasible tokens the mixture sums to *at most* 1, which can only raise the loss. The validation NLL is computed as `−log(p + 1e−9)`. That is not exactly normalized: the floor adds at most 50257 × 1e−9 of mass. The normalized model `q = (p + 1e−9) / Z` scores at most `log(1 + 5.03e−5)` = **0.00005 nats** above the reported value, which does not change any reported mean at 4 decimals or the significance result.

## Rules checklist

1. **Data pipelines untouched**: the token streams, data loader and validation tokens are byte-identical to #92. Only the model's output distribution changes. The longer pointer band at evaluation (8192 vs 2048 in training) falls under "evaluation at any sequence length".
2. **Mean val

## #376 Track 1: Document-local copy mixture at the final validation (-15 extension steps, ~-0.8s)
- fetched 2026-10-03T10:48:02.392631Z 200; state open; created 2026-09-29T23:47:16Z; closed None; merged None

At the final validation only, mix the model's next-token distribution with an old, tried-and-true trick: if the last few tokens already occurred earlier in the same document, assign some probability on whatever followed them last time. The mixing weights (31 of them, one per match-length × distance bucket) are fitted by maximum likelihood on training tokens, on the clock, right before the final validation; λ = 0 is always available, so on the tokens it is fitted on the mixture can only match or beat the model, and nothing is learned from validation. On identical weights this is worth 0.0031 val loss, which I spend on a 15-step shorter extension schedule (1290 → 1275), netting ~7 s on 1×H100 and a projected ~0.8 s on the leaderboard's 8×H100 at iso-loss.

```
                          Runs  Steps  Hardware   Time μ  Time σ  Time +/-    Loss μ   Loss σ  Loss +/-   Loss p
  baseline-1290              1   1290  1×H100      508.8       -      0.0   3.27637        -   0.00000        -
  baseline-1275 (no mix)     6   1275  1×H100      502.4     2.5     -6.4   3.28028  0.00054  +0.00391   0.8714
  this PR                    6   1275  1×H100      503.8     2.5     -5.0   3.27715  0.00053  +0.00078   0.0000
  this PR                    1   1275  8×H100 †    179.6       -        -   3.27746        -         -        -
```

"baseline-1275 (no mix)" and "this PR" refer to the same six runs: every run prints both losses from the same forward passes, so the gain carries no seed noise: +0.00312 ± 0.00003 over all eight runs on this code. The step cut fails the bar on its own (p = 0.87) and clears it with the mixture (p = 2.3e-5). Time μ includes the fit in the "this PR" rows and excludes it in the others. Loss p is my one-sided t-test against 3.28; the 8×H100 row is one run (model alone 3.28060).

Hardware: 1×H100 on Modal for the statistics, plus one 8×H100 Modal (due to compute constraints...) node (†) that runs 140 ms/step against the leaderboard's ~52, so its absolute time does not transfer; the per-step and fit costs do. The wall-clock change decomposes into (a) −15 extension steps at 566 ms (1×H100) / 327 ms (that 8×H100 node) / ~76 ms (leaderboard node, from the record's logs), and (b) +1 fit: 1.38 s on 1×H100, 0.44 s across 8 GPUs (one eval forward per rank on a validation-sized batch, plus the matcher and one all-reduce). Within each run that is 15 × 566 − 1384 = 7.1 s (1.4%) on 1×H100 and 15 × 327 − 440 = 4.5 s (2.4%) on the 8×H100 node; with ~76 ms steps and a 0.3–0.44 s fit it should be 0.7–0.85 s (1.0–1.3%) on the leaderboard node. Matching itself costs ~0.2 s per validation pass, off the clock. Nothing enters the compiled graph: the mixture runs eagerly on the per-token loss the eval forward already returns.

The mixture: for each position, find the longest suffix of the context (lengths 1, 2, 3, 4, 6, 8, 12, 16, 24, 32) that occurs earlier in the same document, take its most recent occurrence and let `c` be the token that followed it. Then P′(x) = (1 − λ_b) · P_model(x) + λ_b · [x = c], with `b` the bucket of (match length, distance ≤768 / ≤2560 / >2560 tokens, i.e. the final short and long attention windows). `c` and `b` depend only on the inputs up to the current position, never on the target, and P′ sums to one, so it is still a valid probability model of the validation tokens. The matcher is exact (it ranks token pairs, suffix-array style; no hashing) and never crosses a document boundary. The weights are fitted on the first 2.1M tokens of the last train shard (which a standard 9-shard run never reaches), read by the unmodified loader, with eval-mode forward passes only; the fit sits after the canonical mask is in place, so it sees the distribution validation will score. `COPY_MIX=0` restores master.

The (alleged) gain comes from, essentially, not copying so much as from context length. 79% of it comes from matches more than 2560 tokens back (1.6% of validation tokens), where no attention layer can see the source; there, for length-32 matches, the copy rule is right 95% of the time while the model's loss is 3.33 nats (0.42 with the mixture). Within 768 tokens the model already copies well and the mixture adds ~0.0004. The fitted weights say the same: λ ≈ 0.001 for length-1 matches nearby, 0.37 for length-32 nearby, and 0.16 / 0.42 / 0.74 for lengths 4 / 8 / 32 beyond 2560. It is additive with #350's canonical masking (+0.00307 on record #89's weights, +0.00312 here).

I also tried and discarded:

- Re-tuning the eval windows on the same weights: long windows of 16 / 26 / 32 blocks were all worse than 20 (+0.0004 to +0.0006), and widening the short window was far worse (+0.006 at 8 blocks, +0.022 at 10).
- Fancier mixtures, rescored offline on one run's per-token losses and ranked on train tokens only: extra distance edges ≤ +0.000005, a "previous occurrence agrees" feature +0.0002 for 3× the buckets, fitting the weights on validation itself (an oracle) +0.00005 (length-only buckets

## #378 Track 1: Document-local copy feature (-1.17 s, -2.9% vs #360; stacked on #375)
- fetched 2026-10-03T10:48:03.049567Z 200; state open; created 2026-10-02T01:59:33Z; closed None; merged None

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
- **Causality.** The match must end strictly before t, and the code enforces it (`prev < t`). The runs were made without the explicit check, which only matters under a 64-bit key collision across match lengths (about 1e-4 per run); on realistic data both versions are bit-identical. Nothing is fitted on validation data.
- **Cost.** Three prefix sums, one sort and some scatters and gathers, inside the step's CUDA graph, with no host syncs. Averaged over the run a step (this PR, #375 included) costs +2.4%, and 62 fewer steps more than pay for it.
- **Schedule.** `num_scheduled_iterations` 1060 (#375: 1107, #360: 1122).
- **Self-check.** `SELF_CHECK_REL` goes from 5e-4 to 2e-3. It was raised during development after a 5.2e-4 gap in a configuration that also had an in-graph training-stream memory, which turned out to be a real captured-graph bug and is not in this PR. This PR's configuration was not tried at 5e-4.

## Relation to other open PRs

- **#376** (@cyrusghane) introduced document-local copying to the speedrun. It mixes a fitted copy distribution into the output at the final validation; here the candidate is an input feature learned during training. The two may partly stack.
- **#367** (@hermabr, exact-match retrieval over the training set): this PR reuses its retrieval injection, but retrieves only from the current document, with no training-set index and no second model.
- **#375** (@daniel-monroe, n-gram token normalization): stacked. The matcher also hashes its normalized ids.

## There is more here, and an ask

**Step margin.** The mean val loss is 4.5 millinats under the bar with sd 0.0007, leaving about 3.8 millinats of headroom for the p < 0.01 test on this node. That is roughly 10 to 15 more scheduled steps, or 0.35 to 0.5 s. A good first try is `NUM_SCHEDULED_ITERATIONS=1050` over at least 8 runs.

**Stream memory.** A second feature, a hashed memory of the training stream, was promising on one GPU. Its 8-GPU tests are inconclusive (see the environment note below).

**How this was made.** One person worked with Claude Opus 5.5 (Anthropic), which wrote the code and ran the experiments. Ideas were screened on a single RTX 3090 proxy of the record, then on a few rented H100 hours, building on #367 and #376 on top of #375.

It took little domain expertise: the limiting factor was compute for validation runs, not know-how. It does not feel like this record is close to its ceiling. With GPU time to validate ideas, we think anyone could keep pushing it down.

**An ask.** If anyo

## #367 Track 1: Longest exact-match retrieval (40.0s -> 21.56s / -46.1%)
- fetched 2026-10-03T10:48:01.739110Z 200; state open; created 2026-09-17T02:00:23Z; closed None; merged None

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

## #375 Track 1: Normalize tokens in n-gram embeddings to improve information density (-0.42s, -1.02%, -15 steps)
- fetched 2026-10-03T10:48:03.662288Z 200; state open; created 2026-09-29T03:59:19Z; closed None; merged None

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

Before the full runs, I ran abridged experiments on one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU, with 1/8 the batch size and 1/8 the n-gram slots, otherwise identical to master. Normalization lowered val loss by **6.2 ± 1.5 millinats** on average over 16 seeds. Seeds 12 to 15 were not run.

| Seed | Baseline | Normalized | Diff |
|---|---|---|---|
| 0 | 3.7816 | 3.7734 | −0.0082 |
| 1 | 3.7822 | 3.7746 | −0.0076 |
| 2 | 3.7929 | 3.7823 | −0.0106 |
| 3 | 3.7822 | 3.7759 | −0.0063 |
| 4 | 3.7820 | 3.7746 | −0.0074 |
| 5 | 3.7908 | 3.7953 | +0.0045 |
| 6 | 3.7893 | 3.7826 | −0.0067 |
| 7 | 3.7846 | 3.7773 | −0.0073 |
| 8 | 3.7838 | 3.7712 | −0.0126 |
| 9 | 3.7841 | 3.7793 | −0.0048 |
| 10 | 3.7950 | 3.7832 | −0.0118 |
| 11 | 3.7959 | 3.8066 | +0.0107 |
| 16 | 3.7951 | 3.7887 | −0.0064 |
| 17 | 3.7766 | 3.7707 | −0.0059 |
| 18 | 3.7880 | 3.7804 | −0.0076 |
| 19 | 3.7855 | 3.7750 | −0.0105 |
| **Mean** | | | **−0.0062 ± 0.0015** |


## #366 Track 1: Hashed n-gram rows from host RAM (-6.16s)
- fetched 2026-10-03T10:48:04.333790Z 200; state open; created 2026-09-15T14:41:23Z; closed None; merged None

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


## #346 New record: 1384 steps / 1.312 min — Ember optimizer on embedding tables + earlier context window extension
- fetched 2026-10-03T10:47:47.310517Z 200; state open; created 2026-07-24T05:40:58Z; closed None; merged None

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
  queued behind the 386 MB param all-gathers, costing 2.3%/step in an early version.
  Fix: a dedicated `dist.new_group()` — same bytes, different queue.
- **Deterministic stats.** Contiguous reductions only, no atomics: optimizer state is
  bitwise identical at fixed world size (verified 2-rank == 1-rank). Since every rank
  must hold identical column stats after the all-reduce, a ~3 KB hash doubles as a
  cross-rank desync check.
- **One all-reduce, exact.** The pooled stats are sums of g², which split exactly
  across ranks — a single 3 KB collective gives every rank the exact global statistics.

## Transparency notes

- The 20 logs come from two cloud 8×H100 nodes (10 + 10, both at ~56.9 ms/step reference
  pace, run on 07/18 and 07/20). The embedded source is byte-identical across all 20
  logs and to the file in this PR. The two batches' means differ by 0.23 milli
  (Welch p = 0.75); each 10-run batch alone lands p ≈ 0.011–0.015, so we attach the full
  set rather than a lucky half.
- One run shows a transient val excursion at step 500 (4.45 vs ~4.15 cohort) that fully
  recovers by step 750 and finishes normally; noted for completeness.
- Everything (17-arm sweeps, paired confirms at 1× and 8×, negative results included) is
  documented in the linked repo's findings if anyone wants the full trail.


### comment katop1234 2026-07-29T05:31:01Z

Congrats on #315/#317 — will rebase onto the new record and post updated numbers once the pending window-related PRs settle.

### comment ClassicLarry 2026-09-18T05:23:52Z

I am going to pass on this PR for now. With only 0.28s gain for a new optimizer, higher loss (requires more than 10 runs to hit p value <0.01), and based on an older branch that would need to be revalidated. The PR drops 6 steps to get the speedup, and its not super clear to me I can't get a similar-ish speedup from dropping 6 steps on the baseline and taking a slightly higher loss. Will come back and revisit at a future time once the open PRs are merged to see if there's a cleaner integration point with bigger wins.

## #377 Track 3 anchor gradient + fp32 embed 2580 steps (n=16)
- fetched 2026-10-03T10:47:48.179230Z 200; state open; created 2026-10-01T05:42:56Z; closed None; merged None

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