# BASEDATA · open training sets and recipes for a small base model, and the fastest way to train ours · 2026-10-04

Lane BASEDATA, a research lane. Nothing was trained, nothing was rented and nothing was spent. The Vast.ai
prices below come from read-only `vastai search offers` calls. Every web claim was read from a primary page
on 2026-10-03 (UTC) unless it says otherwise.

Labels used throughout:
- **MEASURED (ours)**: read from our own run logs in this folder.
- **MEASURED (elsewhere)**: a number another group published, read from their primary page.
- **SCALED**: our arithmetic on a measured number.
- **ESTIMATE**: a judgement, not a measurement. It must be measured before money is spent against it.
- **UNVERIFIED**: seen only second-hand, or the primary page could not be read.

This extends `RESEARCH_OPENDATA_MODEL_2026-10-01.md` (lane OPENDATA). That lane covered K2 Horizon and
TxT360-v2 in full. They are not repeated here.

## 0. Our model, as it actually ran

The brief describes SdmLM with 4 back-tokens and 2 averages. **The champion run used a different recipe.**
The start event of `runs_sdmchats/sdmchats_SWL_d768_s0_600M.log` (sdmwide768) reads: `n_back 8`, decays
`0.5, 0.8, 0.9, 0.97, 0.99`, softness 0.25, `value_init zero`, `qgrad 0`, store learning-rate multiplier 3.

| quantity | value | source |
|---|---|---|
| embedding (tied, 129,280 x 768) | 99,287,040 params | MEASURED (ours), run start event |
| everything else | 16,515,840 params | same |
| forward FLOPs per token, head | 198.6M (86.7%) | same, `flops_per_token_fwd` |
| forward FLOPs per token, body | 30.5M (13.3%) | same |
| batch | B 32 x T 256 = 8,192 tokens per step | same |
| optimiser | AdamW (0.9, 0.95), wd 0.1 on 2-D non-embedding, clip 1.0, 2% warmup, cosine to 10% | `track4_sdmllm_train_one_arm.py` header |
| tokens | 600M of FineWeb-Edu `sample/10BT` | `track4_sdmllm24_train_big_provenance.json` |
| box and speed | Spark GB10, median 31,380 tok/s, 11.0 h | MEASURED (ours), log |
| TEST | 1.40285 bpb on 3,892 FineWeb-Edu windows | result JSON |

```
   one training token, by forward FLOPs (sdmwide768)

   head  129,280-way softmax  ████████████████████████████████████████  86.7%
   body  mix + 2 store hops   ██████                                     13.3%

   => anything that speeds the HEAD speeds the run. Muon, attention tricks and
      body kernels can only touch the small bar.
```

**This shape decides most of the answers below.** Our model costs what a ~115M-parameter model costs per
token, but only 16.5M of those parameters do the thinking. Most published speed recipes target the body of
a transformer. Ours is mostly a vocabulary projection.

## 1. The best open pretraining datasets for small base models

### What the controlled comparisons say

| dataset | size | licence, access | evidence at small scale |
|---|---|---|---|
| **Nemotron-ClimbMix** (NVIDIA) | 400B GPT-2 tokens; about **335B DeepSeek tokens** (SCALED, see below) | **CC-BY-NC-4.0**, ungated | nanochat's biggest single speedup (below) |
| **Nemotron-CC v1 HQ / v2.1** (NVIDIA) | v1 6.3T; v2.1 2,544.8B tokens (2.5T new English web) | v2.1: CC-BY-4.0 except named subsets, **gated** behind the "NVIDIA Data Agreement for Model Training" | open-sci-ref: best of 8 sets, 0.13B to 1.7B |
| **DCLM-Baseline 1.0** | 3.8T (per the open-sci-ref summary) | CC-BY-4.0, ungated | open-sci-ref: second |
| **FineWeb-Edu** (what we use) | 1.3T GPT-2 tokens, about 1.22T DeepSeek (OPENDATA, SCALED) | ODC-By, ungated | open-sci-ref: third; nanochat's baseline until 2026-03 |
| **FinePDFs-Edu** (HF) | 350B+ tokens | ODC-By, ungated | half of HF's Smol-Data mix (below) |
| **HF Smol-Data mixes** (2026-02-13) | 100B: FinePDFs-Edu 50 / DCLM 30 / FineWeb-Edu 20 | ODC-By, ungated | won a 70M / 1B-token study; lost in nanochat (below) |
| **Ultra-FineWeb** (OpenBMB) | not read | Apache-2.0, ungated, card modified 2026-08-20 | authors report it beats FineWeb-Edu from scratch at small scale (UNVERIFIED, search summary only) |
| **FineMath, Stack-Edu** (HF) | per SmolLM2 paper | FineMath ODC-By; Stack-Edu tag empty | math and code, used from step 0 in SmolLM2-135M/360M |

Hub metadata (created, modified, licence tag, gate) was read from `huggingface.co/api/datasets/<id>` on
2026-10-03 for every row above.

- **open-sci-ref-0.01** (arXiv 2509.09009, v1 2025-09-10, v3 2025-12-30) trained 0.13B to 1.7B models on
  8 open sets up to 1T tokens. Quoted: *"training on NemoTron-CC HQ consistently outperforms other reference
  datasets, followed by DCLM-baseline and FineWeb-Edu."* MEASURED (elsewhere).
- **nanochat switched to ClimbMix on 2026-03-04** (`dev/LOG.md`, `dev/LEADERBOARD.md`, commit `324e69c`).
  Time to GPT-2 fell **2 h 46 m to 2 h 1 m, a 27% cut**, and the model shrank from d26 to d24. MEASURED
  (elsewhere). Karpathy's own record of what failed before it, from the same LOG:
  - FineWeb instead of FineWeb-Edu (2026-02-17): d26 CORE 0.2602 to **0.2241**.
  - The FinePDFs 50 / DCLM 30 / FineWeb-Edu 20 mix (2026-02-17): d26 0.2602 to **0.2549**; d18 0.199 to
    **0.192**.
  - Olmo 3 `dolma3_mix-6T` (2026-01-15): d16 CORE 15.5 to **13.8**, and he found documents as short as "5".
  - His X post (2026-03-05, read via search, X not fetched): *"ClimbMix worked really well out of the box
    (to the point that I am slightly suspicious about goodharting, though reading the paper it seems ~ok)."*
    UNVERIFIED as to exact wording.
- **The 1-billion-token challenge** (HF blog, codelion, 2025-11-03): 70M GPT-2 at 1B tokens, over 50
  experiments. The 50/30/20 FinePDFs/DCLM/FineWeb-Edu mix won, and *"static mixing is 1.8x better on
  validation"* than their curriculum. This is **the study closest to our scale**, and it is the one HF
  packaged as Smol-Data. MEASURED (elsewhere).
- **The two disagree, and the reason matters for us.** codelion scored perplexity on held-out text from the
  mix plus FineWiki. nanochat scored CORE (22 benchmark tasks). Neither scored bpb on FineWeb-Edu text.
- **CLIMB paper** (arXiv 2504.13161, v1 2025-04-17, v2 2025-11-30): a 1B model on 400B ClimbMix tokens is
  2.0% above Llama-3.2-1B, from an equal-budget mixture search over 20 clusters of 1.2T tokens. MEASURED
  (elsewhere).
- **Newer and narrower.** Edu-QuRating (arXiv 2609.09425, 2026-09-08, CC-BY-4.0) re-scores 322.25M
  FineWeb-Edu-Fortified documents with distilled pairwise judges and reports *"higher observed aggregate
  accuracy across nine benchmarks than the FineWeb-Edu baseline"*. Its abstract gives no model size or
  token budget. MixtureVitae (arXiv 2509.25531, v5 2026-01-12) is permissive-first; at 1.7B / 300B it beats
  FineWeb-Edu and approaches DCLM (search summary, UNVERIFIED). **I found no 2026 English corpus with a
  published win over ClimbMix or Nemotron-CC HQ at small scale.**

### ClimbMix at our tokenizer, measured in this lane

I downloaded `karpathy/climbmix-400b-shuffle` shard 0 (92.3 MB parquet, one `text` column, 86,016 docs,
revision `915333b4`) and tokenized row groups 0, 42 and 83 (3,072 docs) with our DeepSeek V4 tokenizer.
- **4.832 bytes per DeepSeek token** (MEASURED, ours). FineWeb-Edu gave 4.77 (S0 provenance: 312.6 MB in
  65.5M tokens).
- About **51.3M DeepSeek tokens per shard**. Over 6,543 shards that is **about 335B tokens** (SCALED from 3
  row groups of 1 shard).
- That is 30 times more tokens than any recommendation below needs.

### ⚠ Two warnings before switching data

1. **Our yardstick is FineWeb-Edu.** TEST is 3,892 FineWeb-Edu windows. Training on FineWeb-Edu is
   in-distribution for that metric, so any other dataset will probably score worse on it even if it makes a
   better model. nanochat met this: its LEADERBOARD says val_bpb after the ClimbMix switch is *"NOT
   comparable"* with earlier runs. **Add a second held-out TEST set from the new data before switching**,
   and report both.
2. **ClimbMix is non-commercial.** NVIDIA's card says CC BY-NC 4.0. Karpathy's re-shuffle carries an
   `mit` tag, but a re-shuffle cannot relicense the text. Fine for research; not for a model we sell. The
   ODC-By sets (FineWeb-Edu, FinePDFs-Edu, Smol-Data) and CC-BY DCLM have no such limit.

## 2. Mixtures and curricula for 50M to 500M parameter models

| recipe | model | mixture | schedule | source, date |
|---|---|---|---|---|
| SmolLM2-135M / 360M | 135M on 2T, 360M on 4T | **single stage**: DCLM filtered by the FineWeb-Edu classifier (score 0 dropped, 1 and 2 downsampled) + FineWeb-Edu, with Stack-Edu, InfiMM-WebMath, FineMath and Cosmopedia **from the start** | WSD, **20% decay**, peak LR 3e-3 | arXiv 2502.02737 §6, 2025-02-04 |
| SmolLM2-1.7B | 11T | 4 stages; web 60% FineWeb-Edu / 40% DCLM; math and code rise in stage 4 | WSD, 10% decay | same §4 |
| SmolLM3-3B | 11.1T | stage 1 web 85 / code 12 / math 3; stage 2 75/15/10; decay stage 63/24/13 | WSD, linear decay to 0 over final 10% | HF blog, 2025-07-08 |
| nanochat d24 | ratio 8 tokens per param | ClimbMix only, one stage | Muon + AdamW | `dev/LEADERBOARD.md`, 2026-03-14 |
| codelion 70M | 1B tokens | 50/30/20 static mix; curriculum lost | not read | HF blog, 2025-11-03 |

- **The small-model finding is consistent.** SmolLM2's own words: the 135M and 360M *"benefited from a
  single-stage training approach with consistently high-quality data"*, unlike the 1.7B. codelion found the
  same at 70M. MEASURED (elsewhere). ⇒ **At our size, pick one good mix and keep it; do not build a
  multi-stage curriculum.**
- **The 60/40 web split.** SmolLM2 Table 1 (350B tokens per arm): FineWeb-Edu wins MMLU 37.5 vs 35.5 and
  ARC 57.5 vs 53.5; DCLM wins HellaSwag 62.3 vs 60.1 and CommonsenseQA 40.1 vs 36.2; 60/40 keeps most of
  both. MEASURED (elsewhere).
- **Annealing (the high-quality tail).** MiniCPM (arXiv 2404.06395, 2024-04-09) introduced WSD so a model
  can be decayed on better data at the end. SmolLM2 evaluated math and code datasets by annealing a 3T-token
  checkpoint to LR 0 on 60B tokens of the candidate plus 40B of the old mix. SmolLM3 raises code and math in
  its final 10%. For us, the natural tail is **FineWeb-Edu itself**: decay on the distribution TEST uses.
  ESTIMATE, owed a measurement.

## 3. Tokens per parameter, and what 600M tokens leaves behind

The guidance disagrees by a factor of 20, because each group counts "parameters" differently:

| source | ratio | counted against | at our model |
|---|---|---|---|
| Chinchilla (2022) | 20 | all params | 20 x 115.8M = **2.3B** |
| nanochat `base_train.py` default (2026) | 10.5 | non-embedding + lm_head (their `num_scaling_params`) | 10.5 x 115.8M = **1.2B** (our head is tied, so it is the embedding) |
| MiniCPM (2024) | **192** on average | non-embedding (their Table 8 sizes) | 192 x 16.5M = **3.2B** |
| SmolLM2-135M (2025) | ~14,800 | all params | deliberately over-trained for cheap inference |

The MiniCPM figure was read from the PDF: *"the data size should be 192 times larger than the model size on
average, as opposed to 20 times in Hoffmann et al."* MEASURED (elsewhere).

- **600M tokens is 5.2 tokens per parameter, or 36 per non-embedding parameter.** Every compute-optimal rule
  above wants 1.2B to 3.2B. **Our model runs in a browser**, where inference cost dominates, so
  over-training past compute-optimal is the right goal, exactly as SmolLM2 does. ⇒ **The target is 3B to
  10B tokens, not 600M.**
- **What our own runs show, and what they do not.** On the RTX 5090 box (2026-10-01), `VC` (d1024, 2.5B
  tokens) scored 1.41699 and `VB` (d768, 3B tokens) 1.43842. SWL (d768, 600M tokens) scored **1.40285**,
  better than both. MEASURED (ours). **That is a recipe effect, not a data effect:** VB and VC are the
  unigram-ngram arm with 4 back-tokens and 2 averages; SWL is the learned store with 8 back-tokens, 5
  averages and the store fixes. ⇒ **We have no measurement of how SWL's recipe scales with tokens.** Every
  run used a cosine schedule fixed to its horizon, so no curve can be read across horizons.
- **The cheapest way to buy that curve: one WSD run with branch cooldowns.** Hägele et al. (arXiv
  2405.18392, NeurIPS 2024 spotlight) show constant LR plus a cooldown matches cosine and lets one run serve
  many horizons. Run SWL at constant LR to 4.8B tokens, and fork a 20% cooldown at 0.6B, 1.2B, 2.4B and 4.8B.
  The four cooldowns cost about 0.2 x (0.6 + 1.2 + 2.4 + 4.8) = 1.8B extra tokens. SCALED. That gives four
  points on one curve for about 1.4 runs' compute.

## 4. Training-speed recipes that matter on one box

| recipe | measured gain | where | does it fit us? |
|---|---|---|---|
| **Better data** (ClimbMix) | **27%** less time to the same CORE | nanochat, 2026-03-04 | yes, if the licence and the yardstick allow (§1) |
| **Muon** on hidden matrices | speedrun 31.4 to 24.9 min (record 3, 2024-10-04); README: *"~1.5x better sample-efficiency, <2% wallclock overhead"* | modded-nanogpt, 124M | **partly**: only our 16.5M body takes Muon. The embedding/head stays on AdamW in every implementation read |
| Muon, independently re-tested | **1.4x at 0.1B, 1.1x at 1.2B**, shrinking with scale | arXiv 2509.02046 (Wen et al., 2025) | the honest prior: modest, and smaller on our head-dominated model |
| **WSD schedule** | matches cosine; reusable checkpoints | arXiv 2405.18392; SmolLM2/3 | yes; free |
| **Batch-size rule** | B_opt ∝ D^0.383 (10x tokens ⇒ ~2.4x batch) | nanochat 2026-02-05, citing Cerebras "Power Lines" arXiv 2505.13738 | yes, when tokens rise 5 to 15x, raise B from 32 toward 64 or 96 and retune LR |
| **FP8, all linears** (torchao tensorwise) | 630K to 740K tok/s = **1.17x**, about **1.05x** at matched quality; **slower at d12** | nanochat 2026-02-02 | ESTIMATE: our head is one large GEMM, the case FP8 suits; must be measured |
| **FP8 on lm_head only** | about **1%** at d12 | nanochat 2026-01-13 | their head is a small share; ours is 87%. ESTIMATE |
| **Sampled softmax** over a shared candidate set, early training | part of record 92: 1.126 to **0.665 min**, bundled with 5 other changes | modded-nanogpt, 2026-08-30, PR #360 | **the recipe aimed at our bottleneck**; gain not separable. Must be measured |
| **Sparse row-wise embedding updates** | in record 92 | same | relevant: we have 99M embedding rows; tied head makes the gradient dense, so only half applies |
| **Hashed n-gram embedding table** | record 62: 1.748 to **1.655 min** (2026-01-19) | modded-nanogpt | we already have `NgramStore`; nanochat **reverted** its version (2026-01-28): gain vanished in wall clock at d25 |
| **Cut / chunked cross-entropy** | Gemma 2 loss memory 24 GB to 1 MB, speed kept | arXiv 2411.09009 (ICLR 2025) | **already done**: our `chunked` loss path |
| **torch.compile** | FP8 without compile is **4x slower** | nanochat 2026-02-02 | already used on CUDA |
| **BOS-aligned packing** | crops 39.4% of tokens at T=2048 with greedy crop; BestFit-Crop 34.6% | nanochat 2026-01-13 | quality, not speed; at T=256 our windows cross documents. Optional |
| Skip AdamW every other step | +2% tok/s, slightly worse per wall clock, **not adopted** | nanochat 2026-02-03 | no |

The current modded-nanogpt record is **#92, 0.665 min (39.9 s) on 8xH100, 2026-08-30, @DevenPzak**, using
*"under 330M tokens"*. Read from the raw README on 2026-10-03. A press release claiming 24.90 s
(AutoTrust, 2026-09-29) is self-reported and not on the record table. UNVERIFIED.

**Muon placement for SdmLM** (from the Muon, DeepSpeed and nanochat implementations read, and the
optax/Muon embedding issue): Muon for `wx`, `wq[h]`, the readout SwiGLU, and possibly the store keys; AdamW
for `emb` (which is also the head), the norms, and the store values (a lookup table, like an embedding). Write
a test that asserts each parameter's group, because a wrong split still trains and the loss still falls.

## 5. Which Vast.ai box gives the most tokens per dollar

### What we have measured on rented hardware

All on one 4x RTX 5090 box (host `41ac7b578b34`, driver 570.181, torch 2.8.0+cu128, 128 CPUs), four
independent arms at once, 2026-10-01. Medians over steps after step 1,000, MEASURED (ours):

| run | arm | d | median tok/s | model TFLOP/s (3 x forward) |
|---|---|---|---|---|
| `vast_VS_sdmread_d512_s0_3B` | learned store | 512 | 174,548 | not computed |
| `vast_VB_d768_s0_3B` | ngram unigram | 768 | 152,293 | 98.5 |
| `vast_VC_d1024_s0_2500M` | ngram unigram | 1024 | 115,708 | 102.3 |
| `sdmchats_SWL_d768_s0_600M` (Spark GB10) | learned store | 768 | 31,380 | 21.6 |

- **The 5090 already runs our model at about 47 to 49% of its 209.5 TFLOP/s** dense BF16 peak with FP32
  accumulate (the figure from Puget Systems and the HN thread on NVIDIA's whitepaper; UNVERIFIED against the
  PDF itself). The chunked loss recomputes the head, so true hardware utilisation is higher still.
- **One 5090 is about 4.9 times the Spark** (152k against 31k tok/s). The two rows are different arms
  (ngram unigram against learned store), so the ratio is approximate.
- **SWL's recipe has not been timed on a 5090.** ESTIMATE: 110k to 150k tok/s; use 130k.

### Live prices, single GPU, 2026-10-03 09:20 UTC

Filter: `rentable=true reliability>0.97 direct_port_count>=1`, single-GPU offers, `dph_total`.

| GPU | offers | min $/h | median $/h | dense BF16 TFLOP/s (fp32 acc) | peak TFLOP/s per median $ |
|---|---|---|---|---|---|
| RTX 5090 | 61 | 0.406 | 0.668 | 209.5 | 314 |
| RTX 4090 | 62 | 0.333 | 0.470 | 165.2 | 351 |
| A100 SXM4 | 27 | 0.403 | 0.734 | 312 | 425 |
| H100 SXM | 8 | 1.735 | 2.137 | ~989 | 463 |
| H200 | 3 | 3.554 | 5.120 | ~989 | 193 |
| B200 | 4 | 7.507 | 8.216 | ~2,250 | 274 |
| RTX PRO 6000 (WS) | 25 | 1.073 | 1.359 | not verified | n/a |

Multi-GPU offers the same day: 4x 5090 min $1.87, median $2.40 (37 offers); 8x 5090 min $3.52, median
$5.12 (29); 8x 4090 min $3.20 (16); 8x B200 min $62.50 (2); 8x H100 SXM **none** under the filter.

### Tokens per dollar for our model (SCALED and ESTIMATE)

Anchored on the measured 5090 rate of 152k tok/s, and assuming the same fraction of peak on each card:

| GPU | tok/s if same MFU | tokens per $ at median | tokens per $ at min |
|---|---|---|---|
| RTX 5090 | 152k (MEASURED) | **0.82B** | **1.35B** |
| RTX 4090 | 120k | 0.92B | 1.30B |
| A100 SXM4 | 226k | 1.11B | 2.02B |
| H100 SXM | 718k | 1.21B | 1.49B |
| B200 | 1.63M | 0.71B | 0.78B |

- **Why "same MFU" is optimistic for the big cards.** At B 32 x T 256 the head GEMM (8,192 x 768 x
  129,280) is large enough to fill an H100, but the body is small matmuls plus a top-k over 3,600 rows, and
  those stay launch-bound. ESTIMATE: an H100 reaches 40 to 70% of the 5090's MFU unless B rises, and raising
  B changes the training (§4). On those terms the H100 lands at 0.5 to 0.85B tokens per dollar, at or below
  the 5090.
- **Why the A100 deserves a 10-minute test.** It has 1.5 times the 5090's dense BF16 rate (consumer cards
  run BF16 with FP32 accumulate at half speed) at a similar price. It has no FP8. ESTIMATE.
- **FP8 changes the ranking on paper.** The 5090's FP8 dense rate with FP32 accumulate is 419 TFLOP/s,
  twice its BF16; the A100 has none. Our head is the FP8-friendly part. nanochat measured only 1.05x at
  matched quality at d24, so treat this as a maybe.
- **One big box versus many small ones.** Our model fits a 24 GB card many times over. Data-parallel across
  8 PCIe GPUs means all-reducing 116M bf16 gradients (232 MB) every step: at 8,192 tokens per GPU and 152k
  tok/s that is about 18.5 steps a second, or about 7.5 GB/s of ring traffic per GPU (SCALED). That is
  feasible over PCIe but not free. ESTIMATE: 70 to 90% scaling at B 32 per GPU, better at B 128. **For seeds
  and ablations, independent arms per GPU (as on 2026-10-01) scale perfectly.**
- ⚠ **Our own throughput must be measured on each candidate before a long run.** Every non-5090 row above is
  arithmetic, not a measurement.

## 6. Recommendation: three options

```
   tokens ─────────────────────────────────────────────────────────▶
   0.6B (today)      3B                 6B                     10B+
   ●─────────────────●──────────────────●──────────────────────●
   SWL 1.40285       A: same data,      B: better data,         C: speed work
                     WSD, measured box  Muon body, WSD tail     on the head first
```

### Option A · the measured path (cheapest, keeps TEST comparable)
- **Data:** FineWeb-Edu, the 3.018B tokens already tokenized in `train_big` (sha256 `624b04a9…`). No new
  data work.
- **Budget:** 3B tokens, with WSD and branch cooldowns at 0.6, 1.2 and 2.4B (§3).
- **Optimiser:** current AdamW recipe; only the schedule changes (warmup 2%, constant, 20% linear decay).
- **Box:** one RTX 5090. At the ESTIMATE of 130k tok/s: 3B + 0.84B of cooldowns = 3.84B tokens, 8.2 h,
  **$3.33 at $0.406/h to $5.48 at $0.668/h**. SCALED from the 2026-10-03 prices.
- **What it buys:** the first token-scaling curve for SWL, on the yardstick we trust.

### Option B · the best-data path (the strongest base we can make)
- **Data:** for research, **ClimbMix** (nanochat's 27% win; non-commercial). For anything we might ship, HF
  **Smol-Data 50/30/20** (ODC-By), or FineWeb-Edu 60 / DCLM 40 after SmolLM2. Decay the last 20% on
  FineWeb-Edu so TEST stays meaningful. **Hold out a new-data TEST set first.**
- **Budget:** 6B tokens.
- **Optimiser:** Muon on the body matrices, AdamW on the embedding/head and store values; WSD 20% decay;
  raise B to 64 and retune LR (the D^0.383 rule gives about 1.9x batch for 5x tokens).
- **Box:** 4x RTX 5090 (min $1.87/h), one data-parallel run, or one A100 if the 10-minute test wins.
  ESTIMATE: 4 x 130k x 0.8 = 416k tok/s, 4.0 h, **about $7.50 to $10**, plus a tokenizing pass on the box's
  CPUs (6B tokens is about 29 GB of text and 24 GB of uint32 shards).
- **Risk:** the data switch may raise FineWeb-Edu TEST bpb while improving everything else (§1).

### Option C · fix the head first, then scale
- **Why:** 86.7% of our FLOPs are the 129,280-way head. The two head recipes with published gains are
  sampled softmax early in training (modded-nanogpt #92) and FP8 on the projection. Neither gain has been
  measured on a model shaped like ours.
- **Work:** a short engineering lane, about a day, then Option A or B on whatever speedup holds.
- **The bigger lever is out of scope here:** a smaller vocabulary would cut the head directly, and Tao et
  al. (arXiv 2407.13623) find smaller models want smaller vocabularies. Our DeepSeek tokenizer is fixed for
  project reasons, so this is noted, not recommended.

### Before any of them: a 10-minute bench, about $1
Rent one each of 5090, A100 SXM4 and H100 SXM for 10 minutes, run SWL's exact config for 2,000 steps, and
read the median tok/s with the run's own load, driver and GPU stamps. That turns every ESTIMATE in §5 into a
measurement. **My expectation (ESTIMATE): the 5090 or the A100 wins tokens per dollar; the H100 wins only
wall clock.**

## Sources

All read 2026-10-03 UTC unless marked.
- modded-nanogpt raw README and record table: https://github.com/KellerJordan/modded-nanogpt
- nanochat README, `dev/LEADERBOARD.md`, `dev/LOG.md`, `nanochat/dataset.py`: https://github.com/karpathy/nanochat
- Karpathy on X, 2026-03-05: https://x.com/karpathy/status/2029701092347630069 (via search; X not fetched)
- Shizhe Diao on X: https://x.com/shizhediao/status/2029370289461575741 (via search)
- ClimbMix: https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix ·
  https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle (shard 0 downloaded and tokenized) ·
  CLIMB paper https://arxiv.org/abs/2504.13161
- Nemotron-CC v2.1 card: https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1
- open-sci-ref-0.01: https://arxiv.org/abs/2509.09009
- SmolLM2: https://arxiv.org/abs/2502.02737 (HTML v1 read for §2 and §6)
- SmolLM3 blog: https://huggingface.co/blog/smollm3
- Smol-Data collection and mix: https://huggingface.co/collections/HuggingFaceFW/smol-data ·
  https://huggingface.co/datasets/HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT
- The 1 Billion Token Challenge: https://huggingface.co/blog/codelion/optimal-dataset-mixing
- Edu-QuRating: https://arxiv.org/abs/2609.09425 · MixtureVitae: https://arxiv.org/abs/2509.25531 (via search)
- MiniCPM (PDF read): https://arxiv.org/abs/2404.06395
- Hägele et al., schedules: https://arxiv.org/abs/2405.18392
- Wen et al., optimisers: https://arxiv.org/abs/2509.02046
- Tao et al., vocabulary scaling: https://arxiv.org/abs/2407.13623
- Cut Cross-Entropy: https://arxiv.org/abs/2411.09009
- Muon placement: https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/ and the search results on
  embedding and head handling (search summary)
- RTX 5090 BF16 rate: https://www.pugetsystems.com/labs/articles/nvidia-geforce-rtx-5090-amp-5080-ai-review/ ·
  https://news.ycombinator.com/item?id=44997957 (via search)
- AutoTrust 24.90 s claim: https://manilatimes.net/2026/09/29/tmt-newswire/pr-newswire/singapores-autotrust-posts-top-marks-on-public-tests-of-recursive-self-improving-ai-with-a-sliver-of-rivals-funding/2434417 (via search)
- Hub API (`/api/datasets/<id>`, `/api/collections/HuggingFaceFW/smol-data`) and `vastai search offers`, both
  read-only, 2026-10-03
- Local: `runs_sdmchats/sdmchats_SWL_d768_s0_600M.{log,result.json}`, `runs_vast/vast_V*_s0_*.{log,result.json}`,
  `runs_vast/box_logs/probe_sdm_d512.out`, `track4_sdmllm24_train_big_provenance.json`,
  `track4_sdmllm_train_one_arm.py`, `track4_sdmllm_models.py`, `RESEARCH_OPENDATA_MODEL_2026-10-01.md`
