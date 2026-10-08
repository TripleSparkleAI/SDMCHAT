# OPENDATA · the September 2026 model that published its training data on Hugging Face · 2026-10-01

Lane OPENDATA, a research lane. Nothing was trained and nothing was rented. Every figure below is either
read from a primary source (a Hugging Face API response, a model card, a parquet file), measured on a
sample in this lane, or derived by stated arithmetic. Extrapolations carry the label SCALED.

## 1. The answer

**The release is K2 Horizon, from the Institute of Foundation Models (IFM) at MBZUAI, released on
2026-09-03.** It is six models from 0.9B to 375B parameters under Apache-2.0, and its training data sits
on Hugging Face as five public dataset repositories (21.56 TB of parquet in total). The headline repo is
**`IFM/TxT360-v2`**.

- **Best match: confirmed.** It is the only release in the last six weeks found with both new weights and
  a multi-terabyte training corpus published on the Hub.
- **"Complete" is not literally true, and the vendor says so.** IFM's commitment reads as training data
  "where redistribution is possible", and construction recipes where it is not (quoted via the
  tech press, section 2). The 7B card reports 22.9T pretraining tokens, and section 3 estimates the five
  public repos at about 12T DeepSeek tokens (SCALED). The two numbers are not reconciled anywhere public.
- The technical report is listed as "In Progress, end of September 2026" on every card, read 2026-10-01.
  It is not out.

```
   K2 HORIZON  (IFM / MBZUAI, 2026-09-03, Apache-2.0 weights)
   |
   |-- models  0.9B · 3.7B · 7B · 32B · 36B-A4B (MoVA) · 375B-A23B
   |
   |-- data on the Hub, 5 repos, 21.56 TB parquet, all ungated
        |-- IFM/TxT360-v2          5.29 TB  web + web-with-QA      CC-BY-4.0   <- the one for us
        |-- IFM/Pretrain-Behaviors 8.37 TB  synthetic behaviours   Apache-2.0
        |-- IFM/Math-Reasoning     4.46 TB  math traces, dialogue  Apache-2.0
        |-- IFM/Code-Reasoning     3.28 TB  code traces            Apache-2.0
        |-- IFM/SFT-Reasoning      0.16 TB  instruction / chat     Apache-2.0
```

## 2. What it is, in plain words

### The models
Read from the Hugging Face API (`/api/models?author=IFM`) on 2026-10-01:

| repo | created (UTC) | downloads |
|---|---|---|
| IFM/K2-Horizon-0.9B | 2026-09-01 22:55 | 36,589 |
| IFM/K2-Horizon-3.7B | 2026-09-01 23:14 | 16,102 |
| IFM/K2-Horizon-7B | 2026-09-01 23:14 | 25,233 |
| IFM/K2-Horizon-32B | 2026-09-01 23:14 | 6,131 |
| IFM/K2-Horizon-MoVA-36B-A4B | 2026-09-01 23:01 | 19,936 |
| IFM/K2-Horizon-375B-A23B | 2026-09-01 23:25 | 14,242 |

Plus FP8, NVFP4 and GGUF variants, two diffusion "Uno" adapters, and an AMD MXFP4 port of the 375B.

- **Training scale, from the cards.** The 7B and 3.7B cards list pretraining at 1,100,000 steps, 22.9T
  tokens, 8K sequence length. Four midtraining stages add 1.1T + 498B + 110B + 199B tokens and stretch
  the context to 512K. The 375B card lists two pretraining phases of 7.08T and 8.05T tokens.
- **Other artifacts.** Intermediate checkpoints as git branches (for example `pretrain_1100000`), W&B
  training logs, and training code at `github.com/ifm-ai/xllm`. All are marked Available on the cards.
- **Pretraining composition, from the press.** About 17% of the pretraining corpus is
  "problem-solving trajectories with explicit reasoning", and about 10 trillion tokens are synthetic
  from IFM's own pipelines (cellcog.ai, quoting the IFM blog of 2026-09-03). I could not open the IFM
  blog itself (HTTP 403), so these two numbers are second-hand.
- **Compute is not disclosed.** No accelerator count, hours or cost appear in any card or announcement.

### The data
- **Five repos, all ungated, all created 2026-08-24.** Sizes and row counts come from the Hub's
  dataset-server `/size` endpoint (`partial: False`, so these are full counts):

| repo | rows | parquet | in memory | licence | last modified |
|---|---|---|---|---|---|
| IFM/TxT360-v2 | 1,836,021,586 | 5.29 TB | 14.13 TB | CC-BY-4.0 | 2026-09-23 |
| IFM/Pretrain-Behaviors | 1,635,031,234 | 8.37 TB | 20.27 TB | Apache-2.0 | 2026-09-02 |
| IFM/Math-Reasoning | 2,016,081,194 | 4.46 TB | 12.11 TB | Apache-2.0 | 2026-09-02 |
| IFM/Code-Reasoning | 451,546,891 | 3.28 TB | 9.60 TB | Apache-2.0 | 2026-09-02 |
| IFM/SFT-Reasoning | 40,797,817 | 0.16 TB | 0.51 TB | Apache-2.0 | 2026-09-16 |
| **total** | **5,979,478,722** | **21.56 TB** | **56.62 TB** | | |

- **TxT360-v2 at revision `a86bdfb101ebaaf71b445f95fb0f9b9bc2a47511`** has three subsets. I read them
  directly in this lane:
  - `web-high-medium`: 770,472,001 rows, 828 files, 1.66 TB. This is ordinary web text with rich per-document
    metadata: `fineweb_edu_classifier_score`, `fasttext_eli5_score`, `fasttext_preselect_score`, format and
    topic classifier ids, a token count, language and score, buckets, and `quality_category`.
  - `web-high-nltk-qa`: 484,377,491 rows, 756 files, 1.52 TB. Columns are `text` and `token_count` only.
    Each web document is followed by synthetic questions about its own word counts ("What are the most
    common action verbs in this document? ... cancel (3 times) ...").
  - `txt360-qa`: 581,172,094 rows, 1,052 files, 2.12 TB. Each web document is followed by Q:/A: pairs about
    its content. It carries full provenance per row: URL, timestamp, CommonCrawl path, about 20 quality
    signals and per-dump duplicate counts.
  - Information cutoff, from the card: Q4 2024.
- **The cards point at two repos that are not public.** The model cards' metadata name
  `IFM/K2-Horizon-Pretrain-Data` and `IFM/K2-Horizon-Midtrain-Data`. On 2026-10-01 both return an auth
  error from the API, so they are private or do not exist. The cards' own "Data" row links TxT360-v2.
- **The dataset card is thin.** It says subsets "may have undergone source-specific filtering, cleaning,
  deduplication, quality scoring, or synthetic-data generation" and does not say which. It gives no
  token counts and no mixture weights.

### What people said on X and elsewhere
X itself refused this lane's fetch (HTTP 402). These quotes come through search results and the pages
that quote them.
- **@IFM_AI, 2026-09-03**, the launch post: "a connected fleet of six foundation models ranging from
  0.9 billion to 375 billion parameters".
- **@vllm_project**: day-0 vLLM support, and "IFM released intermediate checkpoints, detailed
  data-construction recipes, the training code, and fine-grained logs alongside the weights."
- **@kimmonismus**: "Weights give you the result. The training record shows how they got there. This is
  what open model releases should look like."
- **@testingcatalog** posted similar praise, labelled as a paid partnership.
- **The New Stack, 2026-09-09**: "developers aren't fully convinced". At launch the 32B was a Stage-1
  checkpoint, and several cards said data or code "will be released".
- **Hacker News**: early comments said the repos were "still just placeholders". One commenter listed
  "3.3 Tbyte for code reasoning, 4.5 Tbyte for mathematical reasoning, 8.4 Tbyte of pre-train". Those
  sizes match the API figures above.
- ⚠ **The press is now out of date in one direction.** At launch several reviews said the 375B's data was
  promised. On 2026-10-01 every card I read (7B, 3.7B, 32B, 375B-A23B, MoVA) marks Data and Checkpoints
  **Available**. Those cards were last modified 2026-09-28 to 2026-10-01.

## 3. How many tokens, at our tokenizer

All token counts here use **our** DeepSeek V4 tokenizer (129,280 ids), not K2's. The method was to read
row group 0 of three files (first, middle and last) per TxT360-v2 subset, tokenize 3,000 documents per
subset, and scale by the full parquet size. **SCALED from 9 row groups.**

| subset | text bytes per parquet byte | bytes per DeepSeek token | text | DeepSeek tokens (SCALED) |
|---|---|---|---|---|
| web-high-medium | 2.354 | 4.466 | 3.91 TB | 0.875T |
| web-high-nltk-qa | 2.688 | 4.497 | 4.08 TB | 0.906T |
| txt360-qa | 2.780 | 4.744 | 5.89 TB | 1.242T |
| **TxT360-v2** | | | **13.88 TB** | **3.02T** |

- **Cross-check.** Row count × mean bytes per document gives 3.87, 4.07 and 5.72 TB. These agree with the
  parquet-based figures to within 3%.
- **All five repos (SCALED, coarser).** TxT360-v2's text is 0.982 of its in-memory size. Applying that ratio
  to all 56.62 TB in memory gives about 55.6 TB of text. At 4.5 bytes per token that is about 12.4T
  tokens. At 3.5 bytes per token (code and math may tokenize denser) it is 15.9T. **About 12 to 16T.**
- **For scale.** FineWeb-Edu is 1.3T GPT-2 tokens, per its card. Our S0 shards gave 65.5M DeepSeek tokens
  for 70M GPT-2 tokens, so FineWeb-Edu is about 1.22T DeepSeek tokens. **TxT360-v2 alone is about 2.5×
  FineWeb-Edu.**
- **Edu-score check.** In `web-high-medium`, `fineweb_edu_classifier_score` peaks at 2.44 across the three
  sampled row groups, with 40% of documents at 2.0 or higher and 0.0% at 3.0 or higher. FineWeb-Edu keeps
  documents scoring 3 and up. ⇒ **this subset is a band below FineWeb-Edu's cut**, which matches its
  "Medium-High" label.

## 4. Fit for the next CHATSDM champion

The current champion is `sdmread` (SDMLLMSTORE `sdm_storeopt_qstop_soft0p25_s2_20M`, 1.60233 TEST bpb). Its
size and budget, from `REPORT_SDMLLM_S0.md` and `REPORT_SDMLLMSTORE.md`:
- 2,902,784 non-embedding parameters, plus a tied 129,280 × 256 embedding of 33,095,680 parameters.
  Counting the embedding once gives **N = 35,998,464**.
- **FLOPs per training token = 6N = 2.16e8.**
- Trained on 20M tokens. The useful-budget marker is 20 tokens per parameter: 0.72B tokens counting the
  embedding, or 58M counting only the body.

### Can we do something similar? Yes, and the slice is the right move
At our size, the whole of TxT360-v2 is about 4,200 times the 20-tokens-per-parameter budget
(3.02T / 0.72B). So the question is which slice, not whether to take all of it.

**The slice I would take, in order of fit:**

1. **`txt360-qa`, first.** It is natural web text followed by Q:/A: pairs about that text. It is the
   closest public thing to the "Question: ... Answer:" turns CHATSDM and MAGIC8 already use, and it carries
   URLs, so we can drop any document whose URL is in our FineWeb-Edu TEST docs before training.
2. **`web-high-medium`, filtered on `fineweb_edu_classifier_score >= 2.0`** (about 40% of rows in the
   sample). It keeps the slice near the FineWeb-Edu distribution, so TEST bpb stays comparable.
3. **`SFT-Reasoning/3efforts-pretrain`** (32.2M rows, 115 GB) if a chat flavour is wanted. ⚠ Its schema
   was not inspected in this lane. Read `features` before planning on it.
4. **Skip `web-high-nltk-qa`.** Its appended questions are word-frequency counting tasks. A 2.9M-parameter
   body cannot compute counts over a document, so those tokens teach the model to imitate an answer format
   with wrong numbers.

**How to pick it cheaply.** Parquet is columnar, so read the score column first and fetch `text` only
for the kept rows. Spread the picks over many files (for example every k-th file), because shards follow
source-file order. Hold out by document, as `track4_sdmllm_prepare_fineweb_edu_token_shards.py` does.
Score every arm on the **same 3,873 FineWeb-Edu TEST windows** plus a new held-out TxT360 set, so that a
change of distribution is measured rather than assumed.

### Tokens per hour (SCALED except the M5 row)
Throughput assumptions. The only measurement is the M5 row. The other rows are arithmetic from FLOPs and
bandwidth and are labelled SCALED.
- **M5, measured:** 9.5k to 14.6k tok/s in the S0/STORE `result.json` files, taken under heavy load
  (load 18 to 289 on 18 cores). Use 12k.
- **One 24 GB GPU (RTX 4090 class), SCALED:** 2.16e8 FLOPs per token at 10 to 40 effective TFLOP/s gives
  about 50k to 200k tok/s. The 129,280-way softmax adds about 2 MB of memory traffic per token, which
  caps a ~1 TB/s card near 500k tok/s, so compute binds first.
- **Spark GB10, SCALED:** the measured sequential bandwidth is 230.5 GB/s. At about 2 MB per token that
  caps near 115k tok/s, and S0's assumed 50 TFLOP/s would allow more. Use 30k to 100k tok/s.

| hardware | tokens per hour | 1 hour | 10 hours |
|---|---|---|---|
| M5 (measured 12k tok/s) | 43M | 0.04B | 0.43B |
| one 24 GB GPU (50k to 200k, SCALED) | 180M to 720M | 0.18B to 0.72B | 1.8B to 7.2B |
| Spark GB10 (30k to 100k, SCALED) | 108M to 360M | 0.11B to 0.36B | 1.1B to 3.6B |

- ⇒ **One to ten hours of one GPU covers 0.18B to 7.2B tokens.** That is 0.006% to 0.24% of TxT360-v2,
  and it already reaches or passes the 0.72B useful budget for this model.
- **What a slice costs to move.** 1B DeepSeek tokens of `web-high-medium` is about 4.5 GB of text, about
  1.9 GB of parquet to download, and 4 GB as uint32 shards. The ids need 32 bits because 129,280 is more
  than 65,535. Tokenizing takes about 24 minutes on the M5 at the measured 3.06 MB/s (312.6 MB in 102 s in
  `data/prepare.log`). The 7.2B upper case is about 13.7 GB of parquet, 28.8 GB of shards, and about
  3 hours of tokenizing.
- **A rented 24 GB GPU for 10 hours at $0.30 to $0.60 per hour costs $3 to $6.** The data is free to
  download (ungated).

### What the full thing would take, at our model size (all SCALED)
Arithmetic: FLOPs = 6 × 35,998,464 × D. GPU hours = D / (tokens per second) / 3600. Dollars = hours ×
$0.30 (low end) or × $0.60 (high end). Spark years = hours / 8,766.

| D (tokens) | FLOPs | one 24 GB GPU, hours | dollars at $0.30 to $0.60/h | Spark GB10 |
|---|---|---|---|---|
| TxT360-v2, 3.02T | 6.5e20 | 4,167 to 16,667 | $1,250 to $10,000 | 8,333 to 27,778 h (0.95 to 3.2 years) |
| all five repos, about 12T | 2.6e21 | 16,667 to 66,667 | $5,000 to $40,000 | 3.8 to 12.7 years |
| K2's own pretraining, 22.9T | 4.9e21 | 31,806 to 127,222 | $9,542 to $76,333 | 7.3 to 24.2 years |

The dollar range pairs the fast rate with $0.30 and the slow rate with $0.60, so it is the widest honest
span. Hours divide across GPUs: 100 GPUs take 1/100 of the time at the same total cost.

**The overheads that come with it:**
- download 5.29 TB (TxT360-v2) or 21.56 TB (all five)
- store 12 TB of uint32 shards for 3T tokens, or about 48 TB for 12T
- tokenize 13.9 TB of text: at the M5's 3.06 MB/s that is 52 days for TxT360-v2 alone, so it needs a
  many-core box

⚠ **It would also be a waste at our size.** 3T tokens is about 4,200 times this model's 20-per-parameter
budget. bpb on a 36M-parameter model flattens long before that, so the useful "full thing" for CHATSDM is
a 1 to 7B-token slice and not the corpus.

**For contrast, retraining K2-Horizon-3.7B on its own 22.9T tokens** is 6 × 3.7e9 × 22.9e12 = 5.1e23 FLOPs.
At S0's assumed 400 TFLOP/s effective per H100 that is about 353,000 H100-hours (SCALED). Pretraining
alone, before midtraining, costs about $0.7M to $1.06M at $2 to $3 per H100-hour.

## 5. Cleaner or better documented than FineWeb-Edu?

**Richer metadata, thinner documentation.**

| | FineWeb-Edu (what we use) | TxT360-v2 |
|---|---|---|
| per-document metadata | id, url, dump, edu score, token count | web-high-medium: 4 classifier scores + buckets; txt360-qa: URL, timestamp, CC path, ~20 quality signals, per-dump duplicate counts; nltk-qa: text and token count only |
| written method | a dataset card with the classifier, threshold and ablations, plus its paper | one card, no mixture weights, no token counts, "may have undergone" filtering, technical report pending |
| text | natural web pages | about 58% of TxT360-v2 rows (1.065B of 1.836B) carry synthetic Q/A appended to the page |
| licence | ODC-By | CC-BY-4.0 (the other four repos Apache-2.0) |
| cutoff | per its dumps | Q4 2024 |

- ✅ **Better for CHATSDM in format.** `txt360-qa` already pairs a passage with question and answer turns.
- ⚠ **Worse as a clean language-model yardstick.** The appended Q/A is model-generated, so bpb on it is not
  comparable to bpb on FineWeb-Edu. Keep FineWeb-Edu TEST as the yardstick.
- ⚠ **Possible overlap.** Both are built from CommonCrawl. Drop any TxT360 document whose URL appears in our
  FineWeb-Edu TEST docs before training.

## 6. The other candidates, checked and ruled out

| candidate | date | why not |
|---|---|---|
| AMD Instella-MoE-16B-A3B | weights 2026-07-23 on HF; report arXiv 2609.00791 | outside six weeks; trained on mixtures of existing public datasets, no own corpus repo; ResearchRAIL licence (academic and research only) |
| Marin 535B-A23B (Stanford) | data thread 2026-09-23 | a 25T-token mix from 152 existing HF datasets (23.11T after dedup and decontamination), but the model is still training; no new weights |
| Ai2 OLMo | Olmo 3 (2025-11), Olmo Hybrid (2026-03) | no Olmo release since 2026-08-15; no Olmo 4 found |
| HuggingFace SmolLM | SmolLM3 (2025) | no HF release since 2026-08-15; "SmolLM4" is a forum suggestion only |
| Apertus (Swiss AI) | 1.5 in 2026-07 | no new model; 2.0 planned for 2027 |
| Nvidia Nemotron, EleutherAI, DatologyAI, LLM360 | | no full-corpus model release since 2026-08-15 on their HF orgs (checked by API) |

## 7. Next actions, if the navigator wants the slice
1. Write `track4_sdmllm_prepare_txt360_token_shards.py` beside the FineWeb-Edu preparer, pinned to revision
   `a86bdfb1`, with URL dedup against TEST.
2. Seal a prediction for `sdmread` trained on 0.72B tokens of `txt360-qa` + edu-filtered `web-high-medium`,
   scored on the same FineWeb-Edu TEST windows.
3. Measure tokens per second on the Spark and on one rented 24 GB GPU before any longer run, so that the
   SCALED rows in section 4 become measured ones.

## Sources
- Hugging Face API, read 2026-10-01: `huggingface.co/api/models?author=IFM`,
  `huggingface.co/api/datasets/IFM/TxT360-v2` (and the four sibling repos),
  `datasets-server.huggingface.co/size?dataset=IFM/...`, and the IFM K2 Horizon collection
- [IFM/TxT360-v2 dataset card](https://huggingface.co/datasets/IFM/TxT360-v2)
- Model cards: [K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B),
  [K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B),
  [K2-Horizon-375B-A23B](https://huggingface.co/IFM/K2-Horizon-375B-A23B),
  [K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B)
- [K2 Horizon collection](https://huggingface.co/collections/IFM/k2-horizon)
- [ibl.ai: K2 Horizon, what a fully open model fleet changes](https://ibl.ai/blog/k2-horizon-fully-open-model-fleet-enterprise)
- [cellcog.ai: K2 Horizon](https://cellcog.ai/blog/k2-horizon/)
- [The New Stack, 2026-09-09](https://thenewstack.io/k2-horizon-fully-open/)
- [Hacker News thread 49551760](https://news.ycombinator.com/item?id=49551760)
- [vLLM on X](https://x.com/vllm_project/status/2095523883688587288) ·
  [@kimmonismus on X](https://x.com/kimmonismus/status/2095502148796666271) (both via search; X refused direct fetch)
- [HPCwire/AIwire, 2026-09-03](https://www.hpcwire.com/aiwire/2026/09/03/institute-of-foundation-models-releases-fully-open-k2-horizon-models-with-weights-code-and-training-data/) (fetch refused, 403; cited via search)
- [AMD Instella-MoE Pretrain card](https://huggingface.co/amd/Instella-MoE-16B-A3B-Pretrain) ·
  [Instella-MoE report, arXiv 2609.00791](https://arxiv.org/pdf/2609.00791)
- [Will Held on X, Marin data](https://x.com/WilliamBarrHeld/status/2102850527575097716) (via search)
- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Local: `experiments/track4/sdmllm/REPORT_SDMLLM_S0.md`, `REPORT_SDMLLMSTORE.md`,
  `track4_sdmllm_data_provenance.json`, `data/prepare.log`, `runs*/` `tok_per_s` fields,
  `SETTLE/SETTLE_CAMPAIGN_2026-09-30.md` (the champion row)
