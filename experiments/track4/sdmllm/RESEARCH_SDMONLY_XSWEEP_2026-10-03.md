# XSWEEP · what the field did between 2026-08-01 and 2026-10-03, and what it changes for the SDM-only model

A field sweep for the SDM-only language model: no attention, a body of product-key memory reads, a tied
129,280-token output layer, trained from scratch on FineWeb-Edu, then chat fine-tuned, meant to run in a
browser. Nothing was trained, rented or committed. Everything here was read on 2026-10-03 (UTC).

## 0. How the sweep ran

- **X was not reachable.** `GET /2/tweets/search/recent` returned HTTP 402 `credits depleted` twice, with the
  rate limit at 449 and then 448 of 450 remaining. That is the credit pool, not a rate limit. The record is
  `xsweep_2026-10-03/raw/X_00_api_status.md` and the two raw JSON responses beside it.
- **The fallback was a web-mirror sweep:** web search, the arXiv Atom API, the GitHub REST API (unauthenticated,
  exhausted at 10:52 UTC, after which GitHub pages were read through raw.githubusercontent.com and the web
  pages), and the Hugging Face Hub API. No x.com post was fetched. Engagement is unknown for every item.
- **Six bands, about 180 raw files**, one per query, each written before it was judged, empties and errors
  included. The band reports carry the full matrices:

| band | report | raw files | subject |
|---|---|---|---|
| A | `xsweep_2026-10-03/band_A_recipes.md` | 23 | small-model recipes, speedruns, optimisers |
| B | `xsweep_2026-10-03/band_B_memory.md` | 39 | memory layers, n-gram tables, attention-free LMs |
| C | `xsweep_2026-10-03/band_C_head.md` | 27 | large-vocabulary output layers |
| D | `xsweep_2026-10-03/band_D_data.md` | 35 | pretraining and chat data |
| E | `xsweep_2026-10-03/band_E_browser.md` | 29 | browser and on-device inference |
| F | `xsweep_2026-10-03/band_F_boxes.md` | 26 | rented Blackwell and Hopper boxes |

- **Independent spot-checks** of the load-bearing claims were made after the bands finished:
  `raw/XS_v01_arxiv_spotcheck.md` (ten abstracts through the arXiv API), `raw/XS_v02_modded_readme_spotcheck.md`
  (the modded-nanogpt record table), and direct reads of modded-nanogpt PR 378 and PR 379 and of the MoME and
  Qwen3.8-Next HTML papers. Each signal below marked "spot-checked" was re-read at its primary source.

**States.** CONFIRMED: the number is in the primary source (paper, repo, PR body, card). For a pull request,
CONFIRMED means the author's own measurement in the PR body; it is not a merge and not a reproduction.
POST CLAIM ONLY: only a post, press release or secondary page says it. UNVERIFIED: the primary was not readable
or not found.

**What did not move in the window.** modded-nanogpt still ends at record #92. Its record row is dated
08/30/26 and its PR #360 was merged 2026-09-28. nanochat had no commit after 2026-07-03. No FineWeb-Edu v2,
no Cosmopedia v3, no Tulu 4. No paper reports chat fine-tuning below 200M parameters.

## 1. Ranked signals

### Tier 1 · changes how we train the model's own parts

**S1. Big sparse tables train best with high learning rates, little or no weight decay and per-row optimiser
state.** Five independent sources point the same way. CONFIRMED.
- modded-nanogpt record #92 (README, spot-checked): *"Sparse row-wise embedding updates: only touched rows are
  exchanged and updated (beta1 = 0, one second-moment value per row), every 4 steps"*.
- MoME, arXiv 2609.15126, 2026-09-14 (spot-checked): the value table uses *"AdamW group, lr 0.2, wd 0.001"*.
- Qwen3.8-Next, arXiv 2608.30320, 2026-08-31 (spot-checked): *"The n-gram embedding table runs on Adam with
  weight decay disabled."*
- nanochat PR 854 (open): a per-row RMSprop on a 300M-row table, 0.703668 against 0.71800 val bpb, 91.74 against
  99 minutes (band A, author's PR body).
- deepseek-ai/Engram issue #9: Engram parameters at *"5× higher learning rate ... and weight_decay=0"*.
  POST CLAIM ONLY (an issue thread).
- Ours: store learning rate x3 and store weight decay 0 are already the recipe. Wave 1 tests x1 and x10 only.

**S2. Adam-style optimisers push rare tokens below their true frequency in a softmax output layer.** arXiv
2609.37535, 2026-09-29. CONFIRMED in the abstract (spot-checked): *"if a token is absent for at least two
consecutive minibatches, its equilibrium probability is strictly below its data frequency for every learning
rate"*, while SGD, AMSGrad, Shampoo and Muon avoid the shift. The paper's experiment is small (band C: width
64, vocabulary 4,096; rare-token log ratio Adam -2.72, AMSGrad -0.14, SGD -0.22). Our tied 129,280-row table
trains with AdamW at beta2 0.95, which is the worst case the abstract describes (a short second-moment time
constant). Leviathan (arXiv 2601.22040, CONFIRMED in the abstract, spot-checked) locates its gains in the same
place: *"gains to be concentrated in rare tokens"*, 9% lower perplexity at 1.2B.

**S3. Copying from earlier in the same document is the largest open speed lever, and one form needs no
attention.** CONFIRMED in the authors' PR bodies, spot-checked, both open and not merged.
- PR 378, 2026-10-02: *"Each position gets a candidate next token: the token that followed the longest earlier
  exact match of its (normalized) context in the same document"*, built from prefix sums and a sort, no
  attention, added at the input and before the final norm. *"-1.174 s (-2.91%)"*, *"-62"* steps.
- PR 379, 2026-10-02, title *"Track 1: CPLM on #360 (-4.57s, -11.25% same hardware)"*: a pointer mixture over
  earlier tokens, *"40.575 ± 0.024 s"* at 1194 steps down to *"36.009 ± 0.039 s"* at 1050 steps.
- PR 376 (band C): 31 mixing weights fitted with no retraining, *"+0.00312 ± 0.00003"* val loss. POST CLAIM ONLY
  as summarised in band C.
- Our model has no copy path at all: its only view of the far past is five moving averages.

**S4. A fixed token-hash table (bigram or n-gram) is a strong memory at small scale, and the address can carry
context.** CONFIRMED.
- MoME (spot-checked, Table 2, 135M model, 3.3B tokens): no memory *"0.8785±0.0010"* val bpb, bigram table
  *"0.8636±0.0009"*, MoME *"0.8611±0.0003"*. MoME gives each token several slots and a learned gate chooses
  between them. Band B reports the hashed bigram also beat the authors' Engram port.
- Qwen3.8-Next (spot-checked): loss *"1.585"* without n-gram embeddings, *"1.541"* with them, on a 25B-A3B
  model, and *"Loss decreases monotonically as the N-gram vocabulary grows, while downstream performance does
  not follow the same trend."*
- FactorEngram, arXiv 2609.35578 (band B): at 340M, WikiText perplexity 23.19 plain, 22.29 Engram, 21.29
  FactorEngram; at 1B, 15.57, 15.60 and 15.57. The perplexity gain was gone at 1B.
- DeepSeek-V4.1-Flash, arXiv 2609.19969 (band B): ships Engram at 196B parameters; no isolated number for it.
- Frozen Memory Is Not Enough, arXiv 2608.17050 (band B): a parameter-matched FFN had better perplexity (7.4
  against 8.7) and lower QA (34.5 against 38.5). This is the split our own record shows: a dense readout wins
  on bits, the memory may win on recall.

**S5. The head can be trained more cheaply, at a measured loss cost.** CONFIRMED.
- Record #92, PR #360 (README and code read in band C at sha 4f5270e5): sampled softmax with every batch target
  plus uniform negatives on a fixed stride, 10,240 then 14,336 then 24,576 candidates of 50,304, full softmax
  only for the last 87 steps, no logQ correction. Band C reads the PR table as about 2.4 millinats of val loss
  for about 8.8 s of wall time. A search summary said the full softmax ran "for the first 100 steps"; the code
  says sampling starts at step 0.
- PR 371 (open): the head's weight gradient from one token row in four, scaled by four, input gradient exact:
  39.583 s against 40.342 s, bundled with an attention-backward change. CONFIRMED in the PR body only.
- PR 372 (closed): adaptive softmax over the 18,430 most frequent tokens; the author wrote that sampled softmax
  *"beats adaptive softmax at the first glance"*.
- Low-rank head, arXiv 2608.16671 (band C): a factorised forward head raised loss by 0.1795, against 0.0586
  for cutting only the backward rank. A low-rank forward head is a bad trade.
- PyTorch 2.13, 2026-07-08 (band F): `nn.LinearCrossEntropyLoss` processes the vocabulary in chunks and supports
  weight tying and z-loss, marked "API Unstable". Its speed against our chunked loss is unmeasured.

### Tier 2 · recipe details and data

**S6. Blend the final weights toward an average of the last steps.** modded-nanogpt PR 364, 2026-09-05, and
optimiser-track row 45: -0.0024 val loss at a blend weight of 0.25 to 0.30. CONFIRMED in the PR body and the
merged row. The PR itself was closed, and nanochat's log says weight averaging did not help there.

**S7. Weight decay and schedule.** arXiv 2607.23777: weight decay scaled down as the learning rate falls reached
the same loss 30% faster, measured with Muon on MoE models. arXiv 2609.04577: the best weight decay grows
roughly as the square root of the overtraining factor. Both CONFIRMED in their abstracts (band A). Neither was
measured on AdamW at our size.

**S8. AI-written text is a growing share of filtered web data.** arXiv 2609.40295, 2026-09-30. CONFIRMED in the
abstract (spot-checked): *"27.5% of tokens from June 2026 web data are labeled as AI-generated by Pangram,
rising to 31.1% by August"*, and for models with large human-text budgets *"AI tokens raise loss almost
immediately"*. Licence CC BY-NC-SA 4.0. Our FineWeb-Edu shards and TEST come from older dumps, so this is a
warning for any newer crawl, not a defect in our current data.

**S9. A fully synthetic corpus trains a 56M model at our width.** SYNTH and Monad, arXiv 2609.37891, 2026-09-29.
CONFIRMED in the abstract (spot-checked): *"At iso-compute, SYNTH outperforms filtered web data"*. Monad's
config (band D): hidden 256, 64 layers, vocabulary 8,192, tied. The comparison is on task accuracy, not web
bpb, and the paper and card disagree on tokens (about 180B against 200B) and width (384 against 256).
Licences CC-BY-4.0 and Apache-2.0.

**S10. Pure FineWeb-Edu held up in nanochat.** nanochat `dev/LOG.md`, entries 2026-01 to 2026-03 (outside the
window). CONFIRMED: CORE 0.2602 against 0.2241 for plain FineWeb, and it also beat a plain-FinePDFs 50/30/20
mix. Small chat fine-tunes are fragile in format: nanochat issue #860 (2026-09-26) reports GSM8K 10.31 to 2.35
after removing 282k short synthetic rows, one run per arm. POST CLAIM ONLY.

**S11. An attention-free model on multi-scale running state beats a same-size transformer on bytes.** Kathleen
Writes, arXiv 2608.04678, 2026-08-05. CONFIRMED in the abstract (spot-checked): *"1.84 vs 2.04 bits/byte at
512 MB with ~0.5M parameters"*. It is the published design closest to our moving-average features. It is
very small and on WikiText-103 bytes, so the result says the idea works, not that it scales.

### Tier 3 · the browser and the rented box

**S12. In WebGPU at batch 1, the number of GPU dispatches sets the speed.** arXiv 2608.08730, 2026-08-09.
CONFIRMED in the abstract (spot-checked): *"the dispatch overhead, not kernel quality, is the bottleneck at
batch size 1"*. Band E reads 24 to 71 µs per dispatch and +53% from cutting 876 dispatches to 564 on Qwen2.5-0.5B.

**S13. Shipped browser models store the token table at 4 bits.** WebLLM's Qwen3-0.6B build (band E, from the
published weight manifest): the 151,936 x 1024 table packed 4-bit with one fp16 scale per 32 values, 77.8 MB
plus 9.7 MB, 24.9% of a 351.5 MB download. CONFIRMED. A plain 4-bit head costs quality: ARCHead, arXiv
2608.02703, reports 1.14 to 1.16 relative perplexity for naive INT4 on the Qwen3-8B head. CONFIRMED in the
abstract (band E).

**S14. One large weight file can fail to store through the browser Cache API.** LatexGen issue #70, 2026-09-14:
`cache.add` failed at 303 MB and 328 MB in one Chromium build; OPFS wrote the same file in 10.4 s. CONFIRMED as
one developer's report. Our d 768 export is one file of about 337 MB. WebLLM's default piece size is 32 MB
(band E, TVM source).

**S15. Multi-GPU RTX PRO 6000 boxes have live collective bugs.** CONFIRMED as reporter issues, none on a Vast host.
- NVIDIA/nccl #2418, 2026-09-17: NCCL 2.26.2 *"reproducibly fails when reaching 524288 bytes"* with *"an
  illegal memory access was encountered"*; 2.31.2 works. Vast's RTX 5 page still states "CUDA 12.8 and PyTorch
  2.7 or greater", the family that ships 2.26.2.
- NVIDIA/nccl #1999 and jax-ml/jax #41182 (2026-10-01, NCCL 2.32.3): P2P hangs, and with P2P disabled one run
  *"completes, but prints correct: False"*.
- pytorch #193752 (POST CLAIM ONLY): six identical resume launches on 4x PRO 6000 printed different eval losses.

**S16. torch.compile on sm_120 has silent bf16 correctness bugs.** pytorch #191433, 2026-07-29, open, labelled
"correctness (silent)": *"BF16 autocast fusion of embeddings, Linear, and RMSNorm silently produces wrong
results"*, max error about 2.2. That is our model's pattern. pytorch #190796 is fixed in 2.11 per the reporter.
CONFIRMED as reporter issues; we have not reproduced either.

**S17. PyTorch moved off CUDA 12.8 wheels.** PyTorch 2.13 blog, 2026-07-08: *"CUDA 13.0 remains the default
build"* and *"the CUDA 12.8/12.9 builds were removed"*. CONFIRMED. Rowwise FP8 finds no cuBLAS kernel on sm_120
(pytorch #192707, CONFIRMED); tensorwise runs.

### Claims seen and not confirmed

- AutoTrust's 24.9 s speedrun from combining two open PRs: their own repository and press release. POST CLAIM ONLY.
- LFM2.5-230M at 1,400 tokens a second in a browser: one X post seen through search, not on Liquid AI's blog.
  POST CLAIM ONLY.
- Ember (arXiv 2607.01455): that Adam's second moment goes stale on rare rows and steps 10^2 to 10^4 times too
  large late in training. In the README, not the abstract. POST CLAIM ONLY.
- AF-Muon for tied embeddings (arXiv 2610.01395): the abstract gives no effect size. UNVERIFIED as a gain.
- Vast prices of about $6.25 an hour for one B200 and $1.10 to $1.99 per RTX PRO 6000 GPU-hour: tracker figures
  seen only in search results. UNVERIFIED.
- The cu130 driver floor (580.65.06 in a search summary): not found on the PyTorch page. UNVERIFIED.

## 2. What this changes for us

Ten changes, most cheap. Each is a proposal for the lane that owns the file; this lane edits no code.

| # | change | source | one-line test |
|---|---|---|---|
| 1 | Store values: per-row second moment with beta1 0, and store learning rate in the embedding range, weight decay 0 | S1: #92, MoME, Qwen3.8-Next, nanochat PR 854 | On `p0_A_c`, arms `--store-opt sparse` at store lr x3, x10, x30 and x60, paired TEST at 20M tokens |
| 2 | Tied table: beta2 0.999 or AMSGrad on the embedding/head group only | S2: arXiv 2609.37535; Leviathan | Arms beta2 0.95 against 0.999 against AMSGrad on the table group; TEST bpb overall and on targets in the rarest 10% of training frequency |
| 3 | Same-document copy candidate as a context feature | S3: PR 378 (no attention) | Add the embedding of the token after the longest earlier exact match in the window to the features; paired TEST at 20M, also scored on positions with a match of 4 or more tokens |
| 4 | Training-free copy mixture at the output | S3: PR 379, PR 376 | On an existing checkpoint, fit per-bucket mixing weights on train windows and score TEST with and without; no training run |
| 5 | A fixed bigram-hash address as one store arm | S4: MoME, Qwen3.8-Next, record 62 | Arm (a) a 65,536-row hashed bigram table added before hop 1; arm (b) hop 1 addressed by the bigram hash instead of product keys; paired TEST against `p0_A_c` |
| 6 | Sampled softmax for the first 90% of tokens, full cross-entropy at the end; no low-rank forward head | S5: #92, PR 371, arXiv 2608.16671 | Candidates = batch targets plus stride negatives over 129,280; tokens a second and TEST at 20M, then at 600M against 1.40285 |
| 7 | Blend final weights toward an average of the last steps | S6: PR 364 | Keep the last 150 steps' running average in the long run; score TEST at blend 0 to 0.4; blend 0 must equal the logged number |
| 8 | Browser file: 4-bit table with one scale per 32 values, 32 MB pieces in OPFS, few dispatches and argmax on the GPU | S12, S13, S14 | 4-bit against int8 per-row: TEST bpb and greedy-token agreement with fp32; load the d 768 export as one file and as pieces, Cache API and OPFS, in Chrome and Safari |
| 9 | Rented-box pre-flight before any long run | S15, S16, S17 | On a fresh box: torch at least 2.11 on cu130, NCCL not 2.26.x, all-reduce checked by value at 256 KiB, 1 MiB and 32 MiB with P2P on and off, compiled against eager gradients on one real batch |
| 10 | Keep FineWeb-Edu as the base and guard TEST; chat half from smol-smoltalk with some short fixed-format rows | S8, S9, S10 | Report TEST bpb split by dump year; one 300M-token arm with 20% SYNTH beside pure FineWeb-Edu at two learning rates each |

**Notes on the list.**
- Change 1 strengthens what SPARSEROWS is already building; the new part is the learning-rate range, which
  every source sets far above our x3. Band B's argument that the 1/k gain makes a value row move 32 times less
  than an embedding row is an inference, not a published result.
- Changes 3 and 4 are the only items here that add a new capability to the model rather than training it
  faster. Our features see the far past only through five moving averages.
- Change 6 has a measured cost on the speedrun of about 2.4 millinats. On our model the head is 87% of forward
  FLOPs, so the speed side may be larger, and that is unmeasured.
- Muon gave nothing new in the window that changes the body plan already written in BASEDATA.

## Sources

Primary pages read 2026-10-03 UTC. Band files list every query and every page.
- modded-nanogpt README and record table: https://github.com/KellerJordan/modded-nanogpt
- modded-nanogpt PRs: https://github.com/KellerJordan/modded-nanogpt/pull/360 · /pull/364 · /pull/371 ·
  /pull/372 · /pull/376 · /pull/378 · /pull/379
- nanochat PR 854 and issue 860: https://github.com/karpathy/nanochat/pull/854 ·
  https://github.com/karpathy/nanochat/issues/860 · `dev/LOG.md`
- arXiv: 2608.30320 (Qwen3.8-Next) · 2609.15126 (MoME) · 2609.35578 (FactorEngram) · 2609.19969
  (DeepSeek-V4.1-Flash) · 2608.17050 (Frozen Memory) · 2609.37535 (rare tokens) · 2601.22040 (Leviathan) ·
  2608.16671 (low-rank heads) · 2607.23777 · 2609.04577 · 2609.40295 (WildAI) · 2609.37891 (SYNTH) ·
  2608.04678 (Kathleen Writes) · 2608.08730 (WebGPU dispatch) · 2608.02703 (ARCHead) · 2607.01455 (Ember) ·
  2610.01395 (AF-Muon)
- Engram issues: https://github.com/deepseek-ai/Engram/issues/9 · /issues/20
- WebLLM Qwen3-0.6B manifest: https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC
- LatexGen issue: https://github.com/OlehZhyhinas/LatexGen/issues/70
- NCCL and P2P: https://github.com/NVIDIA/nccl/issues/2418 · https://github.com/nvidia/nccl/issues/1999 ·
  https://github.com/jax-ml/jax/issues/41182 · https://github.com/pytorch/pytorch/issues/193752
- PyTorch: https://github.com/pytorch/pytorch/issues/191433 · /issues/190796 · /issues/192707 ·
  https://pytorch.org/blog/pytorch-2-13-release-blog/
- Vast: https://docs.vast.ai/rtx-5-series
- Data cards: https://huggingface.co/PleIAs/Monad · https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk ·
  https://huggingface.co/datasets/PleIAs/common_corpus
- Local: `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`, `TRAINING_SDMONLY_HOW_WE_TRAIN.md`,
  `RESEARCH_BASEDATA_2026-10-04.md`, `WIKI/CATCHING_UP_WITH_X_METHOD.md`
