# PRIORRECIPES · how recent small language models are trained, and what our SDM-only trainer does differently · 2026-10-03

This document compares the training recipe of the SDM-only model (`track4_sdmonly_models.py`, trained by
`track4_sdmonly_train.py` through the S0 loop in `track4_sdmllm_train_one_arm.py`) with the newest public
small-LM recipes and with the memory-layer literature. It extends `RESEARCH_BASEDATA_2026-10-04.md`, which
already covers datasets, token budgets, Muon in general, WSD, FP8, the sampled-softmax record and box prices.
Those are cited here, not repeated. Nothing was trained and nothing was run on the Spark.

Labels:
- **MEASURED (ours)**: from our own run logs or reports, with the path.
- **MEASURED (elsewhere)**: a result another group published, with the URL.
- **READ (elsewhere)**: a setting read from another group's source code or config, with the URL. A setting is
  not a measurement; it is what the recipe does.
- **ARITHMETIC**: our arithmetic on numbers labelled elsewhere in this document.
- **UNVERIFIED**: seen only in a search summary, or the page could not be read unambiguously.

Web pages were read on 2026-10-03 (UTC). Where a page was read through a summarising fetch rather than as raw
source or rendered PDF, a single number could have changed in that step; those are marked "(fetch)". Local
papers were read from the PDF text already on disk.

## 0. The model this recipe trains

| quantity | d 256, 4 hops, n_sub 256 | d 768, 4 hops, n_sub 256 | label |
|---|---|---|---|
| head forward FLOPs per token (2 V d) | 66,191,360 | 198,574,080 | ARITHMETIC, from `flops_per_token()` in `track4_sdmonly_models.py` |
| body forward FLOPs per token | 3,211,264 | 18,808,832 | same |
| head share of forward FLOPs | **95.4%** | **91.4%** | same |
| tied table V x d | 33,095,680 | 99,287,040 | same |
| store parameters (values, keys, query maps) | 67,633,152 | 202,375,168 | same, `count_store()` |
| dense body parameters (wx, norms) | about 0.86M | about 7.7M | ARITHMETIC, total minus table minus store |
| training speed on the Spark | 46,700 tok/s | 22,800 tok/s | MEASURED (ours), `SDMONLY_CHANNEL.md` JIMOTHY block |

```
   one training token, forward FLOPs, SDM-only d 256

   head  129,280-way tied softmax  ██████████████████████████████████████████████  95.4%
   body  features + 4 sparse reads ██                                              4.6%

   the earlier shape (sdmwide768, RESEARCH_BASEDATA §0): head 86.7%, body 13.3%
```

Two consequences shape every answer below.
1. **A recipe change that touches only the body's dense matrices touches about 1% of the parameters at d 256.**
   Muon, matrix initialisation and matrix weight decay act on `wx` and the query maps only. ARITHMETIC.
2. **The tied table is three things at once**: the input embedding, the output head, and (through the back-token
   and moving-average features) the whole context representation. Every recent recipe gives the input table and
   the output head different settings. A tied table cannot take both.

## 1. Recipe elements: the newest recipes, our trainer, the gap

"nanochat" is `karpathy/nanochat` master, `scripts/base_train.py` and `nanochat/gpt.py`
(https://github.com/karpathy/nanochat, read 2026-10-03, fetch; the same lines were also read from our local
clone at commit `92d63d4`, 2026-07-03, `wikis/WIKI_NANOCHAT/_sources/nanochat-repo/`, and agree).
"modded #92" is the current modded-nanogpt record, 0.665 min, 2026-08-30, @DevenPzak, PR #360
(https://github.com/KellerJordan/modded-nanogpt, `records/track_1_short/`, read 2026-10-03, fetch).
"Ours" is the default path of `track4_sdmonly_train.py`, which hands its loop to `track4_sdmllm_train_one_arm.py`
(read in the working tree on 2026-10-03; the opt-in flags added by lane SPARSEROWS are listed separately).

| element | nanochat | modded #92 | ours today | gap |
|---|---|---|---|---|
| matrix optimiser | Muon, lr 0.02, momentum 0.95 (warmed 0.85 to 0.97 over 400 steps), 5 Newton-Schulz steps (READ) | ANVIL (a Muon descendant: two momentum rails 0.85 and 0.98, six quintic maps), lr 0.023, momentum 0.95 (READ, fetch) | AdamW lr 3e-3, betas (0.9, 0.95), eps 1e-8 on `wx` and `wq` | small: about 1.1M dense body params at d 256. Opt-in `--body-opt muon` exists, unmeasured |
| output head | untied `lm_head`, Adam lr 0.008 x (d/768)^-0.5, betas (0.8, 0.96), eps 1e-10, wd 0.01 (READ) | Adam lr 0.008, betas (0.5, 0.95), base wd 0.005 with `wd_mul` 150 (READ, fetch) | tied with the input, lr 3e-3, betas (0.9, 0.95), **no weight decay** (the name filter `"emb" not in n` puts it in the no-decay group) | the table gets one lr and one beta pair for two jobs |
| input embedding | untied `wte`, Adam lr 0.3 x (d/768)^-0.5, betas (0.8, 0.995), wd 0.001 (READ) | embed initialised as a copy of the head, then split late in training (READ, fetch; the split stage is our reading of `SPLIT_EMBED_STAGE = 4`) | the same tied table | nanochat's input lr is **37.5x** its head lr (0.3 / 0.008, ARITHMETIC) |
| extra per-token input tables | value embeddings on alternating layers, lr 0.5 x embedding lr (READ); "the models **love** Value Embeddings" (MEASURED elsewhere, nanochat `dev/LOG.md` 2026-01-17, quoted in `wikis/WIKI_NANOCHAT/09-DEVLOG_the-negative-results-ledger.md` §2.6) | value embeddings at layers 1, 2, 8, 10; a hashed n-gram table of 84,602,880 rows, zero-init, its own sparse Adam (READ, fetch) | none | our own record says this helps us too (§2, row "untie") |
| schedule | warmup 40 steps, constant, linear warmdown over the last **65%** to 0.05 of peak (READ) | no warmup in `get_lr`, linear decay over the last **80%** to a floor of 0.30 x the stage multiplier (READ, fetch) | 2% warmup, cosine to 10% of peak; `--sched wsd`: 2% warmup, constant, linear to zero over the last 20% | the decay fraction is open: 20% (ours, SmolLM2), 65%, 80%. Wave 1 tests WSD 20% only |
| weight decay | Muon: "cautious" wd 0.28, scaled by sqrt(B/B_ref) x (D_ref/D), cosine-decayed over training; head 0.01, input 0.001 (READ) | ANVIL cautious wd 2.25; Adam base 0.005 (READ, fetch) | 0.1 on `wx` and `wq`; 0 on the table, the norms and the store | ours sets none on the head and a flat 0.1 on matrices |
| batch | B_ref 524,288 tokens at d12; auto batch `B_ref x ratio^0.383` (Power Lines); lr scaled by sqrt(B/B_ref) (READ) | 131,072 tokens a step in stage 0, rising to 393,216 in stage 2 (8 GPUs) (READ, fetch; ARITHMETIC 8 x 2048 x 8 and 24 x 2048 x 8) | **8,192 tokens a step** (B 32 x T 256) | **64x** below nanochat's reference batch (ARITHMETIC) |
| sequence length | 2,048 | 896, then 2,048, then 3,072 | 256 windows, random offsets, crossing document boundaries | our model has no attention; T sets only the reach of the moving averages and the O(T^2) average matrix in `causal_ema` |
| initialisation | `wte` N(0, 0.8); `lm_head` N(0, 0.001); input matrices uniform with std n_embd^-0.5; **both output projections zero** (READ) | `lm_head` N(0, 0.005); the MLP output projection zero (READ, fetch) | table N(0, 0.02); every 2-D body weight N(0, 0.02); **store values zero** | our zero values already play the zero-output-projection role: the model starts as the no-read model |
| norms | parameter-free RMSNorm (READ) | not checked | RMSNorm with a learned weight (`track4_sdmllm_models.py` line 63); BatchNorm without affine on each query | minor |
| logit cap | `15 * tanh(logits / 15)` (READ); a sweep of 5 to 30: "5 was terrible, the rest of them were all about equal with the exception of 20, which was the best" (MEASURED elsewhere, nanochat `dev/LOG.md` 2026-03-02, quoted in `wikis/WIKI_NANOCHAT/03-MODEL_architecture-by-depth-and-the-single-dial.md` §6) | `23 * sigmoid((z + 5) / 7.5)` (READ, fetch) | **none** | cheap, unmeasured on us |
| head loss | full softmax | sampled softmax over 10,240 to 24,576 candidates for about 98.7% of steps, uncorrected (READ, fetch; §2) | exact fused chunked cross-entropy (`track4_sdmllm24_fused_ce.py`) | §2 |
| precision | one global bf16 dtype, no autocast (READ, local clone) | FP8 head and MLP (READ, fetch) | bf16 autocast, fp32 weights, fp32 cross-entropy | FP8: see `RESEARCH_BASEDATA_2026-10-04.md` §4 |
| gradient clip | not checked | not checked | global norm 1.0 | none known |
| evaluation | `val_bpb` | val loss | bits per byte on fixed FineWeb-Edu TEST windows, BOS targets excluded | aligned with nanochat |

**What the table says, in four lines.**
- Our trainer is a 2023-style AdamW recipe applied to a model that is 95% output head.
- Every recent recipe splits the token table's settings in two (input and output), and we cannot, because the
  table is tied.
- Our batch is 64x smaller than nanochat's reference. For a head-bound model, a larger batch changes no FLOPs per
  token and makes the head GEMM larger.
- We have no logit cap and no z-loss. Both recent recipes cap their logits.

### The nanochat Adam rules, transferred to our shape (ARITHMETIC, not measured)

nanochat scales Adam learning rates by `(d/768)^-0.5` and by `sqrt(B/B_ref)`. Applied to d 256 and
8,192 tokens a step:

```
   head lr    0.008 x (256/768)^-0.5 x (8,192/524,288)^0.5 = 0.008 x 1.732 x 0.125 = 0.0017
   input lr   0.3   x 1.732 x 0.125                                                = 0.065
   ours, tied                                                                        0.003
              ├── 0.0017 head ──●── 0.003 ours ─────────────────────────────── 0.065 input
```

Our tied lr sits just above nanochat's head rule and 22x below its input rule. nanochat's own comment on these
rules reads: *"Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling
hoping it ~works for Muon too."* (READ, local clone `scripts/base_train.py`, quoted in
`wikis/WIKI_NANOCHAT/03-MODEL_architecture-by-depth-and-the-single-dial.md` §7). The transfer above is a pointer,
not a recommendation.

### What failed in nanochat that bears on us (MEASURED elsewhere, `wikis/WIKI_NANOCHAT/09-DEVLOG_the-negative-results-ledger.md`)
- **Context-aware gating of an Engram-style hash memory did not help** at 100M to 1B: *"Gating didn't help at our
  scale"*, and modded-nanogpt found *"simple direct addition to the residual stream outperformed by a decent
  margin."* (§2.2)
- **The bigram hash table helped per step and was removed one day later**: *"At larger scale (d25), the
  improvement was tiny and disappeared entirely when measured by wall clock time."* (§2.2)
- **Hyperparameters tuned at d12 did not transfer to d20**: *"The elaborate fine-tuning that won at d12 actively
  hurts at d20"*, and the improvement shrank from about 0.002 at d12 to about 0.0007 at d20 (§2.7). Our 20M-token,
  d 256 arms are a proxy in exactly this sense.
- SwiGLU lost to ReLU² at iso-param and iso-FLOP (§2.4). Our dense control and readout use SwiGLU.

## 2. The output layer: every option to cut its cost or improve it

At d 256 the head is 95.4% of forward FLOPs (§0). The training head also costs two backward GEMMs of the same
size, so about 3 x 66.2M = 199M FLOPs per token against about 9.6M for the body. ARITHMETIC.

| option | published result | suits a tied 129,280-way head? | cost to try |
|---|---|---|---|
| **Exact fused chunked CE** (never materialise the logits) | Cut Cross-Entropy: loss memory 24 GB to 1 MB on Gemma 2 2B; at 8,192 tokens, loss+grad memory 164 MB and 145 ms against torch.compile 16,000 MB and 143 ms (MEASURED elsewhere, https://arxiv.org/abs/2411.09009, fetch). Liger: "increase multi-GPU training throughput by 20% and reduces memory usage by 60%" (MEASURED elsewhere, https://github.com/linkedin/Liger-Kernel README, fetch) | yes, and **already ours**: `track4_sdmllm24_fused_ce.py`; plain path 26.6k tok/s on the GB10 at d 768 (MEASURED ours, that file's header) | done |
| **Gradient filtering inside the fused CE**: skip softmax entries below ε = 2^-12 in the two backward GEMMs | CCE: the backward "without gradient filtering it is 3.4x (356 ms) longer" (MEASURED elsewhere, same URL, fetch). The filtered time was given second-hand as 100 ms: UNVERIFIED | yes; tying changes nothing. Needs block-sparse matmuls, not a PyTorch `@`. On the GB10 the gain is unknown | an engineering lane, not an arm |
| **Sampled softmax early, full softmax late** | modded #92: candidates = every distinct target in the microbatch plus stride negatives `k * 20011 mod V`; P = 10,240 on steps 0 to 680, 14,336 on 681 to 964, 24,576 on 965 to 1,106, full softmax from step 1,107 of 1,122; **no log-Q correction**, the file says the loss is "biased (the normalizer misses the non-candidate mass)" (READ, fetch). Its gain is bundled with five other changes in one record (`RESEARCH_BASEDATA_2026-10-04.md` §4) | yes. Only candidate rows of the tied table get an output gradient; every row still gets its input gradient. At P = 16,384 the loss GEMMs shrink to 12.7% of the full head (ARITHMETIC). The biased loss means the run must end on the full softmax, and TEST is always scored on the full softmax | one arm (§4, rank 7) |
| **Corrected sampled softmax** (subtract log Q) | TensorFlow's implementation: `true_logits -= log(true_expected_count)`; "generally an underestimate of the full softmax loss" (READ, `tensorflow/python/ops/nn_impl.py`, fetch); Jean et al. 2015 (https://arxiv.org/abs/1412.2007, fetch) | yes | a variant of the arm above |
| **Adaptive softmax** (frequency clusters, small dims for rare tokens) | "2× to 10× speed-ups" (MEASURED elsewhere, https://arxiv.org/pdf/1609.04309 p.1). With adaptive input and tied weights: WikiText-103 test ppl **20.51 tied against 21.74 untied**, 246.9M against 291.3M params; "ADP-T is 34% faster than ADP" (MEASURED elsewhere, https://arxiv.org/pdf/1809.10853 Table 3) | yes, it is defined tied (adaptive input and adaptive softmax share partitions). It changes the model: rare tokens get fewer dimensions. Our vocabulary is not sorted by frequency, so the clusters need a count pass | a model change, owned by the models file; larger than one arm |
| **Low-rank or factorised head** (V x E then E x d) | ALBERT: E = 128 (MEASURED elsewhere, https://arxiv.org/abs/1909.11942). Against it: "models based on less than 1000 hidden dimensions tend to adopt degenerate latent representations in late pretraining" (MEASURED elsewhere, Godey et al., https://arxiv.org/abs/2404.07647, fetch) | poorly. d 256 is already a narrow head; E = 128 would halve head FLOPs (ARITHMETIC) and narrow it further | not proposed |
| **Logit soft-cap** | nanochat 15 tanh; the sweep quoted in §1 (MEASURED elsewhere); Gemma 2 caps the final logits at 30.0 (READ, https://arxiv.org/html/2408.00118, fetch) | yes | one line; one arm (§4, rank 3) |
| **z-loss** | PaLM: `z_loss = 10^-4 · log² Z` (READ, https://arxiv.org/abs/2204.02311 PDF p.10) | yes | one line; a variant of rank 3 |
| **Scaled tied logits** | PaLM: "Because the input and output embedding layers are shared, we scale the pre-softmax output logits by 1/√n", with embeddings N(0, 1) (READ, PaLM PDF p.10). Gemma multiplies the input embedding by √d and reuses the table for the logits (READ, `gemma_pytorch/gemma/model.py`, fetch). muP divides the readout input by the width multiplier and recommends a zero-init readout (READ, `microsoft/mup` `mup/layer.py` and README, fetch) | yes. Ours: `RMSNorm(x) · E^T` with E ~ N(0, 0.02), so initial logits have std about √256 x 0.02 = 0.32 (ARITHMETIC). The RMSNorm weight is a learned per-dimension temperature already | low priority |
| **Untie the input table** (keep E as the head, add a separate input table) | nanochat is untied (READ). MobileLLM at 125M: tying saves 16M params (11.8%) and costs "a 0.2 points drop in average accuracy" (MEASURED elsewhere, https://arxiv.org/html/2402.14905 §2.2.3, fetch). SmolLM2-135M is tied (READ, its `config.json`, fetch). **Ours: a free per-token input vector (the hashed unigram table) beat no-store by -0.0083 / -0.0082 / -0.0075 bpb over three seeds and beat the parameter-matched dense control by -0.0123** (MEASURED ours, `REPORT_SDMLLMSTORE.md` §5, "N-gram hash arms"). That report reads it as *"a finding about the tied embedding, not about memory."* | this is the question itself. Cost: +33.1M params at d 256 (+99.3M at d 768), no extra FLOPs (a gather). Browser bytes: about +33 MB int8 at d 256 against a 107 MB file (ARITHMETIC from the BROWSEREXPORT size formula in `SDMONLY_CHANNEL.md`) | one arm (§4, rank 1) |
| **A smaller vocabulary** | Tao et al.: optimal vocabulary for 33M non-embedding params is about 37K to 43K, and performance "degrades consistently when the vocabulary size goes beyond the optimal configuration" (MEASURED elsewhere, https://arxiv.org/abs/2407.13623, quoted in `wikis/WIKI_ML_LIBRARY/03-training-optimization/small-lm-training-recipes-20-100m.md` §1) | the tokenizer is fixed for project reasons | not proposed |
| **FP8 head** | see `RESEARCH_BASEDATA_2026-10-04.md` §4 | the GB10 is not an H100; unknown | not repeated |

**A zero-cost diagnostic owed before any head work:** count how many of the 129,280 rows ever appear as a
training target. A row that is never a target receives only the softmax denominator's push. The practice is
described in `wikis/WIKI_ML_LIBRARY/03-training-optimization/small-lm-training-recipes-20-100m.md` §1; a 17K-token
distillation corpus touched only 2,261 of 129,280 ids (MEASURED ours, quoted in
`wikis/WIKI_ML_LIBRARY/01-architectures/embedding-parameterization.md`, "Key idea 4"). No count exists for the
65.5M-token `train` shard or the 3.0B-token `train_big` shard. It needs one pass over a shard, no GPU.

## 3. Memory layers: what Memory Layers at Scale and its relatives do that we do not

Sources: Berges et al., *Memory Layers at Scale*, arXiv 2412.09764, PDF text on disk at
`WIKI/research/HARVEST_2026-07-26_parametric-vs-nonparametric-memory_papers/arxiv-2412.09764_berges-2024_memory-layers-at-scale.txt`,
and its official code `facebookresearch/memory` (`lingua/product_key/memory.py`,
`apps/main/configs/pkplus_373m_1024k.yaml`, read 2026-10-03 as source). Lample et al. 2019, arXiv 1907.05242,
PDF text at `WIKI/research/HARVEST_2026-07-29_learned-teacher-free-addressing-rivals_sources/1907.05242_large-memory-layers-with-product-keys.txt`.
Our own harvests: `wikis/WIKI_ML_LIBRARY/07-moe-sparsity/product-key-memory-and-large-memory-layers.md`,
`WIKI/research/HARVEST_sizing_memory-layers_2026-07-19.md`,
`WIKI/research/HARVEST_R8L4_memory-augmented-training-from-scratch-bypass-and-collapse.md`.

| element | Memory+ (Berges 2024) and its code | Lample 2019 | Engram (2601.07372) | ours |
|---|---|---|---|---|
| where memory sits | 3 layers, "centered with a stride of 4 for the 134m models and 8 for the others"; the 373m config uses layers `4,12,20` (READ) | replaces the FFN of layer 4 or 5 of 6 at best; "Putting memory at layer 1 ... gives the worst performance" (MEASURED elsewhere, quoted in R8L4) | layers 2 and 15 (READ, fetch) | every hop is a memory read; no dense layer between hops |
| dense layers kept | yes. "Beyond this point, replacing further FFN layers degrades performance, showing sparse and dense layers are both needed and likely complementary" (MEASURED elsewhere, Berges text lines 296 to 299) | yes, a transformer | yes, a transformer | **none by default** (`--readout-f 0`). Wave 1 arm `p0_A_ro` adds one readout after the hops |
| sharing | one value pool shared by all memory layers; `mem_share_values: True` (READ) | separate | separate | one store per hop by default; wave 1 arm `p0_A_share` shares one |
| output gating | `output = (y ⊙ silu(xᵀW₁))ᵀW₂`; in code `self.value_proj(output * F.silu(self.swilu_projection(input)))` (READ). The paper: "the swilu non-linearity consistently improves results" (MEASURED elsewhere, Berges text line 922) | none | sigmoid gate `σ(RMSNorm(h)ᵀRMSNorm(k)/√d)` (READ, fetch) | none: `x <- x + read` |
| query normalisation | qk-normalisation "when needed", since training "can become unstable, especially for small base models" (Berges text lines 374 to 376); code default `mem_query_batchnorm=False` (READ) | query BatchNorm: at 1M slots usage 25.8% to 80.3% and ppl 19.8 to 18.0; "not necessary for small memories of size 16k and 65k" (MEASURED elsewhere, Lample text lines 434 to 456) | RMSNorm on both sides of the gate | BatchNorm without affine on the query; sub-keys unit-normalised; 65,536 locations a store |
| sizes | 2^10 half keys, 2^20 values; key dim half the model dim; value dim = model dim; `mem_heads 4`, `mem_knn 32` (READ) | 512² slots, 4 heads, k 32 | n-gram hash tables | 256² locations a store, 1 head, k 32, d_a 256 = d |
| value learning rate | `value_fixed_lr=0.001` in a separate AdamW group, base lr 1e-4 in the 373m config: **10x** (READ, ARITHMETIC) | "a higher Adam learning rate of 10⁻³" against 2.5 x 10⁻⁴: **4x** (READ, Lample text lines 352 to 356) | "Adam with a learning rate scaled by 5× and no weight decay" (READ, fetch) | **3x**, weight decay 0 (`--store-lr-mult 3 --store-wd 0`) |
| value initialisation | `normal_(std=v_dim**-0.5)` (READ) | not stated | conv parameters zero, "to strictly preserve the identity mapping" (READ, quoted in R8L4 §1.3) | zero |
| value updates | dense in the paper's framing; their custom EmbeddingBag kernel reaches "3TB/s" against "less than 400GB/s" for PyTorch's, "end-to-end 6x faster" (MEASURED elsewhere, Berges PDF) | sparse Adam on values | sparse | dense AdamW over all 4 x 65,536 x 256 = 67.1M value floats every step (ARITHMETIC); opt-in `--store-opt sparse` (row-wise Adam over woken rows) exists, unmeasured |
| query gradient into the trunk | flows | flows | flows | stopped (`qgrad 0`): MEASURED ours, the flowing gradient cost 0.0235 bpb against S0 sdm (`REPORT_SDMLLMSTORE.md` §5 Block 1) |

### What our own record adds to that table (MEASURED ours, `REPORT_SDMLLMSTORE.md`)
- **The paper settings are harmful while the query gradient flows.** lr x3, wd 0 and zero values with the gradient
  flowing scored 1.72924 against 1.60536 for no-store, with 63% of steps clipped and value rows growing to a max
  norm of 26.6. With the gradient stopped the same knobs are worth 0.0012. Our 3x and wd 0 stand on that test.
- **Softness stopped mattering** once the gradient path was fixed: 0.0010 bpb across softness 0.25 to 8.
- **No context-addressed store beat no-store across seeds** in the old shape (mean -0.0006 over three seeds).

```
   the value learning-rate multiplier, as each group set it

   ours (SDMLLMSTORE)        x3   ███
   Lample 2019               x4   ████
   Engram                    x5   █████
   Memory+ code (373m cfg)   x10  ██████████
                                  only ours was measured against x1 on our model
```

### The bypass and stale-row risks (from our harvests)
- **Stale Adam state on rows nobody woke.** Dense AdamW keeps moving a row after its gradient goes to zero: the
  first moment decays by 0.9 a step and the second by 0.95, so the update `m / √v` shrinks by about 7.7% a step
  and lasts tens of steps (ARITHMETIC from betas (0.9, 0.95)). Bricken's sparse-memory ablation ranked SGD 0.63,
  SGD with momentum 0.54, Adam 0.23, RMSProp 0.20 on a continual-learning task (MEASURED elsewhere, quoted in
  `experiments/track4/sparsestar-distilled/fresh-ground-round/HARVEST_lane5_modern-optimizers-and-the-sdm-no-momentum-split.md`
  §FINDING 2). That task is not language-model pretraining, so the transfer is a hypothesis. Lane SPARSEROWS's
  `--store-opt sparse` is the instrument that tests it.
- **Magnitude bypass.** With no gate, the trunk can only ignore a useless read by shrinking the values and the query
  map. R8L4 names the read-norm ratio `‖read‖ / ‖x‖` per hop as the first-class metric to log. Our trainer logs
  `fire_mass`, `sim_top` and `distinct_topk_frac` per hop and the locations-used fraction after training; it does
  not log the read norm.
- **No memory-specific training recipe exists to import.** *"What is NOT in any of these papers: key/value
  initialization, a memory-specific learning rate, memory weight decay, warmup, or any load-balancing loss"*
  (`wikis/WIKI_ML_LIBRARY/07-moe-sparsity/product-key-memory-and-large-memory-layers.md`, "TRAINING DYNAMICS"). The
  official code since then supplies a value lr and a value init (table above); nothing else has changed.

### Measured gains of Memory+ (MEASURED elsewhere, Berges 2024)
- At 1.3b, NaturalQuestions / TriviaQA: dense 7.76 / 32.64, Memory+ with 1M values 13.68 / 42.89 (PDF Table 1,
  fetch; the same figures in `wikis/WIKI_ML_LIBRARY/07-moe-sparsity/product-key-memory-and-large-memory-layers.md`).
- At 8B, 1T tokens: MMLU 59.68 dense against 63.04 Memory+ (PDF Table 2, fetch).
- "Memory+ improves further over Memory, with performance falling generally between dense models with 2x-4x
  higher compute" (Berges text lines 446 to 447).
- The per-row numbers of Table 3 (placement and gating ablations) and Table 4 (value dim) could not be read
  unambiguously: the extracted text on disk and a rendered read of the PDF give different row alignments.
  UNVERIFIED. The prose conclusions quoted above are unambiguous.

## 4. Ranked changes worth one Spark arm each (PROPOSALS)

Protocol for every row: wave 1's protocol (20M tokens of `train`, B 32, T 256, d 256, seed 0, the centre
`p0_A_c` settings, TEST on S0's windows), one change from the centre, the centre as control. Expected time:
20M tokens at the measured 46,700 tok/s is **about 7.1 minutes of training** (ARITHMETIC), plus the ablation
scoring the trainer already does. A difference counts at 0.003 bpb and paired z of at least 3 (the plan's rule).
None of these duplicates a wave-1 arm (lr, store lr, hops, locations, k, heads, share, softness, qgrad, readout,
WSD and store wd are already sealed there). Each row needs a flag in a file this document does not own; the owner
is named.

| rank | the one change | control | why it is ranked here | expected | owner of the flag |
|---|---|---|---|---|---|
| 1 | **Untie the input table**: a separate V x d input embedding, E stays the head | `p0_A_c` | our own three-seed result: a free per-token input vector bought -0.008 bpb; every recent small recipe gives input and output different settings | a gain of 0.005 or more (ESTIMATE). +33.1M params, same FLOPs, about 7 min | `track4_sdmonly_models.py` (JIMOTHY) |
| 2 | **Silu-gated value projection on each read**: `x <- x + W₂(read ⊙ silu(W₁ norm(x)))`, W₂ zero-init | `p0_A_c` | Memory+'s adopted change; it also gives each hop the dense non-linearity the paper says sparse layers need beside them | small gain (ESTIMATE). Two d x d maps add 4 d² FLOPs a hop: 1.05M per token over 4 hops at d 256, 1.5% of the 69.4M forward FLOPs (ARITHMETIC) | `track4_sdmonly_models.py` |
| 3 | **Logit soft-cap** `15 tanh(z / 15)` on the head (variant: z-loss 1e-4) | `p0_A_c` | both recent recipes cap logits; nanochat's sweep found 15 to 30 about equal and 5 terrible | small at 20M tokens; the case is stability over a 3B-token run (ESTIMATE) | the fused CE and `track4_sdmonly_models.py` |
| 4 | **The table in its own Adam group**: betas (0.8, 0.995), eps 1e-10, lr unchanged | `p0_A_c` | nanochat's input-table settings; the wave-1 lr sweep moves every group together and never the table alone | unknown sign (ESTIMATE) | the S0 trainer's group builder (`track4_sdmllm_train_one_arm.py`, via SPARSEROWS's `track4_sdmonly_optim.py`) |
| 5 | **Row-wise Adam on the values at store lr x10** (`--store-opt sparse --store-lr-mult 10`) | `--store-opt sparse --store-lr-mult 3` | Memory+ code uses 10x; sparse updates remove the stale-row effect; our x3 was measured only against x1 with dense Adam | unknown (ESTIMATE); it also tests whether x3 was a dense-Adam optimum | SPARSEROWS, flags exist |
| 6 | **Batch 4x**: B 128 (32,768 tokens a step), lr x2 by the sqrt rule, same 20M tokens | `p0_A_c` | 64x below nanochat's reference batch; a head-bound model gets bigger GEMMs at no FLOP cost | tok/s up, bpb at 20M tokens probably worse (610 steps against 2,441, ARITHMETIC). Read both columns. The real test is at 100M+ tokens | trainer flag `--B` exists |
| 7 | **Sampled softmax for the first 80%, full softmax for the last 20%**: candidates = batch targets plus stride negatives to P = 16,384, no correction | `p0_A_c` | the one head trick aimed at our bottleneck; at P = 16,384 the loss GEMMs are 12.7% of the full head (ARITHMETIC) | tok/s up, bpb at equal tokens slightly worse; a win only per wall clock (ESTIMATE) | a new loss path beside `track4_sdmllm24_fused_ce.py` |
| 8 | **Muon on the body matrices** (`--body-opt muon`) | `p0_A_c` | Wen et al.: 1.4x over AdamW at 0.1B, 1.1x at 1.2B (MEASURED elsewhere, https://arxiv.org/abs/2509.02046, fetch); but it touches about 1.1M of our 101M params | small (ESTIMATE) | SPARSEROWS, flag exists |
| 9 | **RMS-normalise the query in place of BatchNorm** | `p0_A_c` | Lample: BN unnecessary at 65k locations; Memory+ code ships with BN off; BN makes a token's address depend on its batch until the running stats are frozen | neutral (ESTIMATE); its value is batch-independence for the export | `track4_sdmonly_models.py` |
| 10 | **Long warmdown**: WSD with the last 65% decaying to 0.05 of peak (nanochat) | `p0_A_wsd` (20% to 0) | the decay fraction ranges from 20% to 80% across recipes and is unmeasured on us | unknown; read after `p0_A_wsd` lands | `wsd_lr()` in `track4_sdmonly_train.py` (SPARSEROWS) |

```
   the ten arms, by the evidence behind each

   1 untie input table     ●●●  our 3-seed -0.008 + every recent recipe
   2 silu-gated read       ●●   Memory+ adopted it; nanochat saw gates fail on hashes
   3 logit soft-cap        ●●   two recipes use it; one sweep
   4 table Adam group      ●    one recipe's settings
   5 sparse values x10     ●    one code default
   6 batch x4              ●    a rule from transformers; speed is the point
   7 sampled softmax       ●    one bundled record
   8 Muon body             ●    fair 1.4x, on 1% of our params
   9 query RMSNorm         ○    export hygiene
  10 long warmdown         ○    waits on p0_A_wsd
```

Two zero-run tasks belong before the arms: the target-coverage count of §2, and adding `‖read‖ / ‖x‖` per hop
to the step log (§3).

## 5. What "latest LLM style" should mean for this model

**The default for the long run, adopted now because each part is either already measured on us or is a
no-regret setting:**
- Exact next-token cross-entropy over all 129,280 tokens through the fused chunked CE. TEST in bits per byte on
  the fixed FineWeb-Edu windows, BOS targets excluded. Every run stamped.
- Warmup, a long constant phase and a linear cooldown (WSD), with cooldown forks at 0.6B, 1.2B and 2.4B tokens, as
  `RESEARCH_BASEDATA_2026-10-04.md` §3 proposes and SPARSEROWS's `--init-ckpt --cooldown-tokens` implements.
- AdamW with separate groups for the token table, the body matrices, the store and the norms. The store keeps lr x3
  and weight decay 0, and its query gradient stays stopped: all three are MEASURED (ours).
- Every residual write starts at zero, so the model starts as the no-read model. Values already do; any new
  projection added on the read path (rank 2) must too.
- bf16 autocast with fp32 weights and fp32 cross-entropy. Gradient clip 1.0.

**Unproven until a Spark arm measures it:** untying the table, the silu gate, the logit cap, the table's own
Adam settings, sparse row updates and their learning rate, a larger batch, sampled softmax, Muon, the query norm,
and the cooldown fraction. Each recent recipe uses most of them. None has been tested on a model that is 95%
output head with no attention.

**The transfer warning applies to every row of §4.** nanochat found that settings won at d12 hurt at d20. A
20M-token d 256 win is a reason to test at width (plan step 1), not a reason to adopt at width.

## Sources

Web, read 2026-10-03 (UTC):
- modded-nanogpt README, `train_gpt.py`, `records/track_1_short/{config.py, schedule.py, training.py,
  model/gpt.py, sampled_softmax.py, ngram_table.py, optim/anvil.py}`:
  https://github.com/KellerJordan/modded-nanogpt (fetch). UNVERIFIED: whether value embeddings use `lr_mul 70`
  (the fetch may have mixed them with the n-gram table's settings); how `wd_mul` combines with the base wd.
- nanochat `scripts/base_train.py`, `nanochat/gpt.py`: https://github.com/karpathy/nanochat (fetch), cross-checked
  against the local clone at `wikis/WIKI_NANOCHAT/_sources/nanochat-repo/` (commit `92d63d4`, 2026-07-03).
- Cut Cross-Entropy: https://arxiv.org/abs/2411.09009 (fetch). UNVERIFIED: the filtered backward time of 100 ms.
- Liger Kernel README: https://github.com/linkedin/Liger-Kernel (fetch).
- Adaptive softmax: https://arxiv.org/pdf/1609.04309 (PDF). Adaptive input: https://arxiv.org/pdf/1809.10853 (PDF).
- Sampled softmax: Jean et al. https://arxiv.org/abs/1412.2007 (fetch); TensorFlow `nn_impl.py` (fetch).
  Bengio and Senecal: not opened.
- Godey et al.: https://arxiv.org/abs/2404.07647 (fetch). ALBERT: https://arxiv.org/abs/1909.11942 (abstract).
- PaLM: https://arxiv.org/abs/2204.02311 (PDF p.10). Gemma 2: https://arxiv.org/html/2408.00118 (fetch).
  Gemma code: https://github.com/google/gemma_pytorch `gemma/model.py` (fetch).
  muP: https://github.com/microsoft/mup `mup/layer.py` and README (fetch).
- MobileLLM: https://arxiv.org/html/2402.14905 (fetch). SmolLM2-135M config on Hugging Face (fetch).
- Memory Layers at Scale: https://arxiv.org/pdf/2412.09764 (PDF) and https://github.com/facebookresearch/memory
  (`lingua/product_key/memory.py`, `lingua/optim.py`, `apps/main/configs/pkplus_373m_1024k.yaml`, source).
- Lample et al. 2019: https://arxiv.org/pdf/1907.05242 (PDF).
- Engram: https://arxiv.org/html/2601.07372 (fetch).
- Wen et al., optimisers: https://arxiv.org/abs/2509.02046 (fetch).
- Not used for a ranking because the numbers were search-only (UNVERIFIED): MARS on GPT-2 large, NorMuon at 5.4B,
  Dion at 3B. Schedule-Free AdamW (https://arxiv.org/abs/2405.15682 and its README, fetch) recommends β 0.95 to 0.98
  for long runs and lrs 1x to 10x higher than scheduled runs; its 124M GPT-2 losses were not found in the main
  text, so it is not ranked.

Local:
- `track4_sdmonly_models.py`, `track4_sdmonly_train.py`, `track4_sdmllm_train_one_arm.py`, `track4_sdmllm_models.py`,
  `track4_sdmllm24_fused_ce.py`, `track4_sdmonly_optim.py` (header only), `SDMONLY_CHANNEL.md`,
  `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`, `TRAINING_SDMONLY_HOW_WE_TRAIN.md`, `RESEARCH_BASEDATA_2026-10-04.md`,
  `RESEARCH_OPENDATA_MODEL_2026-10-01.md`, `REPORT_SDMLLMSTORE.md`.
- `wikis/WIKI_NANOCHAT/03-MODEL_architecture-by-depth-and-the-single-dial.md`,
  `wikis/WIKI_NANOCHAT/09-DEVLOG_the-negative-results-ledger.md`,
  `wikis/WIKI_NANOGPT/SUCCESSOR_modded-nanogpt-muon-speedrun.md`.
- `wikis/WIKI_ML_LIBRARY/03-training-optimization/{small-lm-training-recipes-20-100m.md,
  optimizer-benchmarking-reality.md, optimizer-schedule-free.md, regularization-and-stability.md,
  training-memory-precision-and-hp-transfer.md}`, `wikis/WIKI_ML_LIBRARY/01-architectures/embedding-parameterization.md`,
  `wikis/WIKI_ML_LIBRARY/07-moe-sparsity/product-key-memory-and-large-memory-layers.md`.
- `WIKI/research/HARVEST_sizing_memory-layers_2026-07-19.md`,
  `WIKI/research/HARVEST_R8L4_memory-augmented-training-from-scratch-bypass-and-collapse.md`,
  `WIKI/research/HARVEST_2026-07-30_the-optimizer-lever-pdfs-verified.md`,
  `WIKI/research/HARVEST_2026-07-28_logit-caching-and-tied-heads.md`,
  `experiments/track4/sparsestar-distilled/fresh-ground-round/HARVEST_lane5_modern-optimizers-and-the-sdm-no-momentum-split.md`,
  and the paper texts on disk named in §3.
