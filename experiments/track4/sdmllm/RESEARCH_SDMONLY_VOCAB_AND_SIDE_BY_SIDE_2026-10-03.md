# SDMONLY · the vocabulary question, and our recipe beside the best current ones · 2026-10-03

Lane VOCABSIDE, a research lane with small measurements. Nothing was trained (three BPE tokenizers were fitted,
which takes seconds), nothing was rented, and no ssh was used. Downloads went to `/private/tmp/vocabside/`
(134 MB) and nothing was written into the repo except this file.

Labels:
- **MEASURED (ours)**: measured in this lane or read from our own logs, with the source named.
- **MEASURED (elsewhere, URL)**: a number another group published, read from their primary page on 2026-10-03.
- **SCALED**: our arithmetic on a measured number.
- **ESTIMATE**: a judgement. It must be measured before anything depends on it.
- **NOT DISCLOSED**: the primary source read does not state it.
- **UNVERIFIED**: seen only in a search summary, or the primary page could not be read in full.

The DeepSeek V4 tokenizer is treated here as one option among several, on equal terms. The question is which
vocabulary suits an attention-free autoregressive model built on sparse memory reads, trained from scratch on
English web text, and run in a browser.

---

## 1. What 129,280 ids cost this model

### 1.1 Parameters and FLOPs, from the code's own functions

Computed with `build_sdmonly("sdmonly", V, cfg)` and `flops_per_token(model, T=256)` imported from
`track4_sdmonly_models.py`. Two shapes: the wave-1 centre (d 256, d_a 256, 4 hops, n_sub 256, k 32) and the
step-1 width (d 768, same store shape). MEASURED (ours, by the code's own counters).

| shape | V | params total | tied table | store + query maps | other | fwd FLOPs/token, head | body | head share |
|---|---|---|---|---|---|---|---|---|
| d 256 | 129,280 | 101,585,408 | 33,095,680 | 67,633,152 | 856,576 | 66.19M | 3.21M | **95.4%** |
| d 256 | 65,536 | 85,266,944 | 16,777,216 | 67,633,152 | 856,576 | 33.55M | 3.21M | 91.3% |
| d 256 | 32,768 | 76,878,336 | 8,388,608 | 67,633,152 | 856,576 | 16.78M | 3.21M | 83.9% |
| d 256 | 16,384 | 72,684,032 | 4,194,304 | 67,633,152 | 856,576 | 8.39M | 3.21M | 72.3% |
| d 768 | 129,280 | 309,343,744 | 99,287,040 | 202,375,168 | 7,681,536 | 198.57M | 18.81M | **91.4%** |
| d 768 | 65,536 | 260,388,352 | 50,331,648 | 202,375,168 | 7,681,536 | 100.66M | 18.81M | 84.3% |
| d 768 | 32,768 | 235,222,528 | 25,165,824 | 202,375,168 | 7,681,536 | 50.33M | 18.81M | 72.8% |
| d 768 | 16,384 | 222,639,616 | 12,582,912 | 202,375,168 | 7,681,536 | 25.17M | 18.81M | 57.2% |

The head term in `flops_per_token` is

    head = 2 · V · d

**In words:** every token multiplies the working vector against every row of the token table, once forward.
Training does this about three times (forward, and two gradient products), so the head's share of training
compute is the same share as in the table.

The store's parameters are large but sparse: a token reads k = 32 of 65,536 rows per hop. So the honest
"thinking" body at d 256 is the 3.21M FLOPs column, and the token table is twenty times larger in compute.

```
   forward FLOPs per token, d 256, wave-1 centre

   V 129,280  head ███████████████████████████████████████████████ 66.2M  body ▌3.2M
   V  65,536  head ████████████████████████ 33.6M                          body ▌3.2M
   V  32,768  head ████████████ 16.8M                                      body ▌3.2M
   V  16,384  head ██████ 8.4M                                             body ▌3.2M
```

### 1.2 Bytes per token on our own text (section 3) turns this into cost per byte

A smaller vocabulary spends more tokens on the same text, so the fair unit is FLOPs per byte:

    FLOPs per byte = FLOPs per token / bytes per token

**In words:** what one byte of text costs the model, so tokenizers that cut text into pieces of different
sizes can be compared. Bytes per token below are MEASURED (section 3) on our held-out text, using the tokenizer
we trained at each size.

| shape | V | bytes/token | fwd FLOPs per byte | against 129,280 |
|---|---|---|---|---|
| d 256 | 129,280 | 4.8143 | 14.42M | 1.00 |
| d 256 | 65,536 | 4.8588 | 7.57M | 0.52 |
| d 256 | 32,768 | 4.6637 | 4.29M | **0.30** |
| d 256 | 16,384 | 4.3637 | 2.66M | 0.18 |
| d 768 | 129,280 | 4.8143 | 45.15M | 1.00 |
| d 768 | 65,536 | 4.8588 | 24.59M | 0.54 |
| d 768 | 32,768 | 4.6637 | 14.83M | **0.33** |
| d 768 | 16,384 | 4.3637 | 10.08M | 0.22 |

### 1.3 Training speed

We hold two measured speeds for this model, both at V 129,280 on the Spark alone (`SDMONLY_CHANNEL.md`,
JIMOTHY OPEN block): d 256 at **46,700 tokens a second** and d 768 at **22,800**. MEASURED (ours). The Spark is
not FLOP-bound at this size: d 768 has 3.13 times the FLOPs of d 256 and runs only 2.05 times slower. So the
speed-up from a smaller head lies between two bounds:

- **lower bound** (SCALED): a two-point fit `time per token = a + b · FLOPs` through the two measured speeds gives
  a = 10.9 µs per token of fixed cost and b = 0.152 µs per million forward FLOPs. A smaller head shrinks only the
  second term.
- **upper bound** (SCALED): time proportional to FLOPs.

| shape | V | tokens/s (fit .. FLOPs-proportional) | bytes/s against 129,280 |
|---|---|---|---|
| d 256 | 65,536 | 60,700 .. 88,200 | x1.31 .. x1.91 |
| d 256 | 32,768 | 71,900 .. 162,100 | **x1.49 .. x3.36** |
| d 256 | 16,384 | 79,100 .. 279,400 | x1.53 .. x5.42 |
| d 768 | 65,536 | 34,500 .. 41,500 | x1.53 .. x1.84 |
| d 768 | 32,768 | 46,800 .. 71,700 | **x1.99 .. x3.05** |
| d 768 | 16,384 | 57,000 .. 112,700 | x2.26 .. x4.48 |

The fit has two points and its fixed term mixes body costs that themselves change with d, so the range is wide.
**One short Spark arm per vocabulary turns it into a measurement** (section 6). On a rented RTX 5090 the model ran
at 47 to 49% of the card's dense BF16 peak (`RESEARCH_BASEDATA_2026-10-04.md` §5), which is close to
FLOP-bound, so there the upper bound is the likelier one. ESTIMATE.

### 1.4 The browser download and the browser's work per token

Lane BROWSEREXPORT's int8 size formula (channel block, 2026-10-03T10:39Z) puts the token table at
`V·d + 4V + 4d` bytes, and their arithmetic puts the whole file at about 107 MB (d 256) and about 337 MB (d 768)
at V 129,280. Changing only V (SCALED from their arithmetic, which is itself not yet measured):

| shape | V | int8 token table | whole file | tokenizer.json (gzip) |
|---|---|---|---|---|
| d 256 | 129,280 | 33.6 MB | ~107 MB | 4.77 MB (1.81 MB) |
| d 256 | 65,536 | 17.0 MB | ~90 MB | 4.71 MB (0.87 MB) |
| d 256 | 32,768 | 8.5 MB | ~82 MB | 2.33 MB (0.42 MB) |
| d 256 | 16,384 | 4.3 MB | ~78 MB | 1.14 MB (0.20 MB) |
| d 768 | 129,280 | 99.8 MB | ~337 MB | as above |
| d 768 | 65,536 | 50.6 MB | ~288 MB | |
| d 768 | 32,768 | 25.3 MB | ~263 MB | |
| d 768 | 16,384 | 12.7 MB | ~250 MB | |

Tokenizer file sizes are MEASURED (ours): our DeepSeek file and the three BPE files trained in section 3.
**The vocabulary alone does not bring d 768 under BROWSEREXPORT's 200 MB budget.** The store (S·M·d values)
dominates there. At d 256 the token table is the second largest part of the file.

The browser's work per token is a different matter. The site engine states in its own header
(`sites/settle-site/src/engine/sdmchat.js`, line 44, read 2026-10-03): *"That loop (129,280 x 256 multiply-adds)
is about 90% of the work per token."* That is the same head term as in 1.1, and it shrinks in proportion to V.

### 1.5 How many rows the model ever trains

| tokenizer | V | distinct ids used on our 2,000 held-out docs | share of the table |
|---|---|---|---|
| DeepSeek V4 (ours) | 129,280 | 50,921 | **39.4%** |
| Qwen3 | 151,669 | 47,490 | 31.3% |
| Gemma 4 | 262,144 | 63,186 | 24.1% |
| GPT-2 | 50,257 | 41,873 | 83.3% |
| SmolLM2 | 49,152 | 42,620 | 86.7% |
| nanochat d32 (published) | 65,536 | 53,929 | 82.3% |
| trained 65,536 | 65,536 | 52,233 | 79.7% |
| trained 32,768 | 32,768 | 31,229 | **95.3%** |
| trained 16,384 | 16,384 | 16,150 | 98.6% |

MEASURED (ours, `/private/tmp/vocabside/distinct_val.json`). On the 65.5M-token train shard, 78,533 of 129,280
DeepSeek ids occur (60.7%; `REPORT_SDMLLM_S0.md` §3). In a tied model every unused row is also an output class the
model must learn to push down, from no positive examples.

---

## 2. What the strongest small models use, and what the papers say

### 2.1 Published choices

| model | size | vocabulary | tied | source, read 2026-10-03 |
|---|---|---|---|---|
| SmolLM2-135M / 360M | 135M, 360M | **49,152**, trained on 70% FineWeb-Edu, 15% Cosmopedia-v2, 8% OpenWebMath, 5% StarCoderData, 2% StackOverflow | yes | arXiv 2502.02737 (HTML v1); `HuggingFaceTB/SmolLM2-135M` config at `93efa2f0` |
| SmolLM3-3B | 3B | **128,256**: "LLama 3.2 tokenizer as is (except for removing `bos_token`)" | yes | huggingface.co/blog/smollm3 (2025-07-08); config at `a07cc9a0` |
| nanochat, published checkpoints | d20, d32, d34 | **65,536** | no | `karpathy/nanochat-d32` (`016dba03`), `-d34` (`c48357d4`), `nanochat-students/base-d20` (`4c6e4e8d`): tokenizer.pkl `n_vocab` |
| nanochat, repo HEAD | d24 speedrun | **32,768**: `--vocab-size ... default=32768`, trained on up to 2B characters | no | github.com/karpathy/nanochat `scripts/tok_train.py`, `nanochat/gpt.py`, HEAD `92d63d4e` (2026-07-03) |
| modded-nanogpt, record 92 | GPT-2 small class | **50,257** (GPT-2 BPE), padded to 50,304 in the head | untied at 2/3 of training | README and `track_1_short/config.py`, `sampled_softmax.py`, HEAD `4ea6b937` (2026-09-28) |
| Qwen3-0.6B | 0.6B | **151,669** (byte-level BPE) | yes | arXiv 2505.09388 §2 and Table 1; config at `c1899de2` |
| Qwen3.5-0.8B (2026) | 0.8B | **248,320** in config (248,070 in tokenizer.json) | yes | `Qwen/Qwen3.5-0.8B` config at `2fc06364` (2026-03-02) |
| Gemma 3 270M | 270M | **262,144**; "170 million embedding parameters due to a large vocabulary size and 100 million for our transformer blocks" | yes | developers.googleblog.com/en/introducing-gemma-3-270m (2025-08-14); `unsloth/gemma-3-270m` mirror (the Google repo is gated) |
| Gemma 4 E2B (2026) | E2B | **262,144** | yes | `google/gemma-4-E2B-it` config at `3e22461f` (2026-07-20) |
| MiniCPM4-0.5B | 0.5B | **73,448** | NOT DISCLOSED in config | `openbmb/MiniCPM4-0.5B` config at `5253c7fc` |
| MiniCPM5-1B (2026) | 1B, 679,552,512 non-embedding | **130,560** | no | `openbmb/MiniCPM5-1B` card and config at `87179e5c` (2026-08-17) |
| nanowhale-100m (2026) | ~110M, "41M embeddings, 69M non-embedding" | **129,280** (DeepSeek-V4 tokenizer) | no | `HuggingFaceTB/nanowhale-100m-base` card at `8ea74cc0` (2026-05-04) |

MEASURED (elsewhere) for every row; the vocabulary sizes were read from the config or tokenizer files we
downloaded. The nanowhale card is the one small 2026 model that uses our exact tokenizer. Its own Limitations
section says: *"110M params with 129K vocab means ~37% of parameters are in embeddings, limiting model capacity."*

**The pattern.** Models built from scratch to be small and English-first (SmolLM2, nanochat, modded-nanogpt)
choose **32k to 65k**. Small members of large families (Qwen, Gemma, SmolLM3, MiniCPM5) inherit the family's
**128k to 262k** tokenizer, because they share it with large siblings, are multilingual, and in Qwen3's case are
distilled from a larger teacher (arXiv 2505.09388 §4.5). Gemma 3 270M pays 63% of its parameters for that.
Our model is from scratch, English, no teacher (`PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` §1), so it sits in the
first group.

### 2.2 What the papers say about vocabulary against model size

- **Tao et al. 2024**, "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies"
  (arXiv 2407.13623, v3 2024-11-01). MEASURED (elsewhere, https://arxiv.org/abs/2407.13623). Abstract:
  *"the optimal vocabulary size depends on the compute budget, with larger models requiring larger
  vocabularies"*. Fitted law (HTML v3): *"N_v^opt ∝ N_nv^γ where γ ≈ 0.83 < 1"*. Table 1 gives 3B non-vocabulary
  parameters → 0.1B vocabulary parameters, 39K to 43K tokens; 7B → 60K to 67K; 70B → 212K to 231K.

      N_v,opt = 0.1B · ( N_nv / 3B ) ^ 0.83          V_opt = N_v,opt / d

  **In words:** the vocabulary's share of parameters should grow more slowly than the rest of the model, so a
  small model wants a small table. Applied to our model (SCALED, an extrapolation of a law fitted on dense
  transformers): d 256 has N_nv = 68.5M, giving V_opt ≈ **17,000** (21,200 with the 39K anchor); d 768 has
  N_nv = 210.1M, giving V_opt ≈ **14,300** (17,900). Most of our N_nv is sparse store rows, so the dense-model
  law may not transfer. ESTIMATE that it points the right way, not at the exact number.

- **Mittal, Gubrani, Kakollu 2026**, "Lifecycle-Optimal Tokenization: Vocabulary Size as a
  Deployment-Regime-Dependent Infrastructure Parameter" (arXiv 2608.11361, 2026-08-11). MEASURED (elsewhere,
  https://arxiv.org/html/2608.11361). *"The training-cost minimum is Vtrain∗=16384"*. At 100M parameters, six GPT
  models at V from 8,192 to 262,144 on 50M tokens: *"BPB is flat across the full V range (<2% spread, best at
  V=16k)"*, 1.344 to 1.368. At 1.3 to 2.3B, *"quality (bits per byte, BPB) is optimized at V=65k"*. And
  *"on-device deployments (B=1) should use V ≈ 32k"*. Their own limits: the large runs were *"only 5k steps"*. A
  search summary reports a critique that the headline batch-shift claim conflicts with their tables
  (UNVERIFIED). This is the closest published study to our case: a browser is a batch-1 device.

- **Limisiewicz et al. 2026**, "Compute Optimal Tokenization" (arXiv 2605.01188, v2 2026-05-26). MEASURED
  (elsewhere, https://arxiv.org/abs/2605.01188): 988 models from 50M to 7B; *"model parameter counts scale
  proportionally to data size measured in bytes, not in tokens"*, and *"the optimal compression rate differs from
  the one obtained with BPE and decreases with compute"*. For us: count the training budget in bytes, which is
  what bpb already does.

- **Huang et al. 2025**, "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling" (arXiv 2501.16975,
  v2 2025-05-23). MEASURED (elsewhere): *"larger input vocabularies consistently enhance model performance,
  regardless of model size"*. Their method decouples the input vocabulary (large, n-gram) from the output one.
  For us this says the cost is in the OUTPUT side. A large input table is cheap (a lookup); a large output table
  is the 2·V·d head. Our own SDMLLMSTORE found the same shape: an extra per-token input vector won by 0.008 bpb in
  all three seeds (`REPORT_SDMLLMSTORE.md`, "It works as an untied input embedding").

- **nanochat's own log** (`dev/LOG.md`, HEAD `92d63d4e`): the digit-grouping regex `\p{N}{1,2}` was validated
  at *"d12, vocab=32K"*: *"`{1,2}` is optimal for vocab size 32K"*. MEASURED (elsewhere).

---

## 3. Bytes per token, measured on our own text

**Method.** All 2,000 held-out documents of `data/val.u32` (sha256 `89fbba2c...`, `track4_sdmllm_data_provenance.json`)
were split at BOS (id 0) and decoded with our tokenizer: 1,998,916 tokens, **9,623,392 UTF-8 bytes** (text
sha256 `59f70289...`). Re-encoding that text with our tokenizer gives 1,998,916 tokens again, the shard's exact
non-BOS count. Each candidate then encoded every document with no special tokens; bytes per token = 9,623,392 /
tokens. A round trip `decode(encode(doc)) == doc` was checked per document. Three tokenizers were trained here:
byte-level BPE with nanochat's split regex, fitted with the `tokenizers` library (0.20.3) on the first 23,276
train documents (114,591,532 bytes), which are disjoint from the held-out documents. Fitting took 4.9 s, 9.1 s and
9.8 s on the M5 (load average about 24, so treat the times as upper bounds). Machine: M5, 2026-10-03T10:49Z.
Scripts: scratchpad `vocab_bpt.py`, `train_bpe.py`, `trim.py`; results in `/private/tmp/vocabside/*.json`.

| tokenizer | vocab | tokens | **bytes/token** | against ours | round-trip fails (docs) |
|---|---|---|---|---|---|
| MiniCPM5-1B | 130,560 | 1,972,312 | **4.8792** | +1.35% | 0 |
| nanochat d32 / d34 (published) | 65,536 | 1,973,124 | **4.8772** | +1.31% | 0 |
| nanochat base-d20 (published) | 65,536 | 1,973,179 | 4.8771 | +1.30% | 0 |
| trained 65,536 (ours, FineWeb-Edu) | 65,536 | 1,980,609 | 4.8588 | +0.92% | 0 |
| **DeepSeek V4 (ours today)** | 129,280 | 1,998,916 | **4.8143** | 0 | 0 |
| nanowhale-100m (same tokenizer) | 129,280 | 1,998,916 | 4.8143 | 0 | 0 |
| SmolLM3 (Llama 3.2) | 128,256 | 2,011,338 | 4.7846 | -0.62% | 0 |
| Gemma 3 270M | 262,145 | 2,042,167 | 4.7123 | -2.12% | 3 |
| Gemma 4 E2B | 262,144 | 2,042,341 | 4.7119 | -2.13% | 3 |
| Qwen3-0.6B | 151,669 | 2,051,099 | 4.6918 | -2.54% | 0 |
| **trained 32,768 (ours, FineWeb-Edu)** | 32,768 | 2,063,475 | **4.6637** | **-3.13%** | 0 |
| Qwen3.5-0.8B | 248,070 | 2,074,501 | 4.6389 | -3.64% | 0 |
| GPT-2 | 50,257 | 2,076,888 | 4.6336 | -3.75% | 0 |
| DeepSeek V4 trimmed to its first 65,536 ids | 65,536 | 2,077,910 | 4.6313 | -3.80% | 12 |
| SmolLM2-135M | 49,152 | 2,095,709 | 4.5920 | -4.62% | 0 |
| MiniCPM4-0.5B | 73,448 | 2,189,005 | 4.3962 | -8.68% | 3 |
| trained 16,384 (ours, FineWeb-Edu) | 16,384 | 2,205,311 | 4.3637 | -9.36% | 0 |
| DeepSeek V4 trimmed to its first 32,768 ids | 32,768 | 2,213,782 | 4.3470 | -9.71% | 31 |
| DeepSeek V4 trimmed to its first 16,384 ids | 16,384 | 2,416,170 | 3.9829 | -17.27% | 47 |

All MEASURED (ours). Notes:
- **A 65k English tokenizer packs our text tighter than DeepSeek's 129k one.** nanochat's published 65,536 gives
  1.3% more bytes per token, with half the table. Most of DeepSeek's extra rows serve Chinese, code and other
  scripts that FineWeb-Edu barely contains (section 1.5: 39% of its ids appear).
- **32k costs 3.1% more tokens than today; 16k costs 9.4% more.** Both are paid back many times over in head
  FLOPs (1.2).
- **"Trimmed DeepSeek"** keeps the first N ids by merge rank. DeepSeek's ids are ordered by merge rank (the
  first merge yields id 259, and ids rise with merge rank for 99.999% of consecutive merges, checked), so a trimmed table's ids are DeepSeek ids.
  It packs English worse than a tokenizer trained on English at the same size (32k: 4.347 against 4.664), because
  the early merges are spread over every language the original was fitted on. The round-trip failures there come
  from decoding at the trimmed boundary (one inspected case: the encoder emitted id 65,535, the same id the full
  tokenizer emits for `.For`, and the trimmed decoder printed a special token); encoding counts are unaffected.
  Cause not fully diagnosed.
- Our trained tokenizers were fitted on the same distribution as the test text, which flatters them. nanochat's
  published 65k (also fitted on FineWeb-Edu, on up to 2B characters) beats our 65k fitted on 115 MB, so the
  flattery is small at this size. The 32k and 16k figures should be read as what a FineWeb-Edu-fitted tokenizer
  of that size gives.

```
   bytes per token on our held-out FineWeb-Edu (higher packs more text per prediction)

   3.9 ┤
       │                                 · DS-trim 16k 3.98
   4.4 ┤      · trained 16k 4.36   · MiniCPM4 4.40
       │                           · SmolLM2 4.59   · GPT-2 4.63
   4.7 ┤      · trained 32k 4.66   · Qwen3 4.69     · Gemma 4.71
       │                           · DeepSeek V4 4.81 (today)
   4.9 ┤      · trained 65k 4.86   · nanochat 65k 4.88   · MiniCPM5 4.88
```

### 3.1 How to keep TEST comparable across tokenizers

bpb is the right score for this, and the S0 report already defines it per byte:

    bpb = ( Σ over scored targets of −ln p(target) ) / ( ln 2 · Σ over scored targets of bytes(target) )

**In words:** total surprise in bits, divided by the number of text bytes it covered. It does not depend on how the
text was cut into tokens, as long as every model scores exactly the same bytes.

The current TEST is "the second half of the val shard by token position, cut into non-overlapping windows of
T + 1 = 257 tokens" (`track4_sdmllm_train_one_arm.py`, `val_windows`). Under another tokenizer that half starts at
a different byte, and each window spans different text. So for a cross-tokenizer comparison:

1. **Fix TEST by document, not by token offset.** TEST = held-out documents 1,000 to 1,999 (the second half by
   document), as text. Record the text's sha256. The bytes come from the text, never from a tokenizer's table.
2. **Re-tokenize those documents with each tokenizer, one document at a time, each starting with that tokenizer's
   BOS.** The BOS target is not scored; every other token is.
3. **Score every byte exactly once.** Either score each document in non-overlapping windows that restart at its
   BOS, or score in windows that do not cross a document boundary. Each model then sees exactly the context of its
   own document. With byte-level BPE the per-token byte counts sum to the document's UTF-8 length, so the
   denominator is identical for every model (assert it).
4. **Give each tokenizer its own token_bytes table** (the `token_bytes.i32` analogue), built from its own vocabulary.
5. **Keep the old protocol beside it for one round.** Score today's champion both ways, so the new TEST has a
   bridge back to every earlier number. Report context in bytes too: 256 tokens are about 1,232 bytes for DeepSeek
   and 1,194 for the trained 32k.

Our model's context is 8 back tokens and 5 moving averages, so a smaller vocabulary also shortens the reach of the
back tokens in bytes (8 tokens: about 38.5 bytes at 4.81, 37.3 at 4.66, 34.9 at 4.36). That is part of what the
comparison measures and should be stated with it.

---

## 4. What we would lose by leaving the DeepSeek tokenizer, and what we would gain

### 4.1 The reasons the record gives for the current choice, as written

- The SETTLE campaign brief for this line of work, `SETTLE/SETTLE_CAMPAIGN_2026-09-30.md` line 36:
  *"an SDM that IS the language model (address the context, read the memory, emit the next token, roll it back
  in), trained the normal modern way: DeepSeek V4 tokenizer (129,280 BPE vocab), standard next-token pretraining,
  against a matched Qwen-style transformer on the same tokens"*. The brief states the choice; it does not give a
  reason.
- `REPORT_SDMLLM_S0.md` §9, under "What a local DeepSeek V4 teacher on the Spark could add later": *"It can supply
  top-k logits for distillation through `--dump-logprobs`. The TWOSTUDENTS pool already holds about 1,528 ds4-flash
  top-128 dumps in this vocabulary."*
- `REPORT_SDMLLM_S0.md` §10, item 6: *"The sealed TWOSTUDENTS preregistration, now that a vocabulary-matched
  pipeline exists."*
- `RESEARCH_BASEDATA_2026-10-04.md` §6, option C: *"Our DeepSeek tokenizer is fixed for project reasons, so this is
  noted, not recommended."* No further reason is given there.
- The root `CLAUDE.md` of this repo, section "THE PIVOT TO NANOCHAT", names ds4-flash as STEP 3 of the teacher
  staircase and states *"STEP 3 remains HELD ⛔"*. Section "THE TEACHER/JUDGE SPLIT" makes ds4-flash *"the PRIME
  JUDGE for everything"*, reached through the Hermes door. Section "THE EMBEDDING-SPACE DEFAULT" says *"the
  addresses default to that model's own captured states"* and notes that *"we hold NO ds4-flash hidden states and
  cannot obtain any"*.

### 4.2 What those reasons mean for SDMONLY

- **Distillation from ds4-flash logits needs a shared vocabulary.** That is the one concrete reason in the record.
  The SDMONLY plan rules it out for this push: *"Three models, each trained from scratch by plain next-token
  prediction. No teacher and no distillation."* (`PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` §1). If TWOSTUDENTS or a
  ds4-flash distillation is ever run on an SDM model, it needs either this tokenizer or a vocabulary mapping.
- **Judging does not need a shared vocabulary.** ds4-flash judges text through the Hermes door, so any tokenizer
  can be judged.
- **The teacher staircase is HELD at STEP 3**, and no hidden-state tap exists, so there is no other ds4-flash
  pipeline that a tokenizer change would break today.

### 4.3 The full ledger

| lose | gain |
|---|---|
| the id match with ds4-flash: the 1,528 top-128 dumps and any future logit distillation (needs a mapping, or a trimmed DeepSeek vocabulary, which packs English worse) | head FLOPs cut 3.9x (32k) at the same d; FLOPs per byte cut to 0.30 (d 256) and 0.33 (d 768) |
| token-level comparability with every earlier SDMLLM checkpoint; the champion `sdmwide768` and its chat and guy fine-tunes cannot be continued | bpb stays comparable (section 3.1), so the score history survives |
| all shards must be re-tokenized: `train` (65.5M), `train_big` (3.018B), `chat_mix`, the guy corpora; new token_bytes tables | ids fit in uint16 at 65,536 or fewer, so shards and data-loader traffic halve (train_big about 12 GB of uint32 today) |
| the site's JS tokenizer is written for DeepSeek's three-regex pre-tokenizer (`sdmchat.js` header); a new tokenizer.json needs that code path changed (site lane's file) | the token table in the browser file shrinks from 33.6 MB to 8.5 MB at d 256; per-token browser work falls with it |
| coverage of Chinese, code and other scripts that DeepSeek's table holds | 95% of a 32k table's rows are used on our text, against 39%; each used row gets about 1.7x more occurrences at 32k (66.1 against 39.3 tokens per used id on the held-out text; SCALED from the 1.5 counts) |
| the "same vocabulary as our big model" story in the paper | a choice that matches Tao et al., the Lifecycle-Optimal study, nanochat HEAD and SmolLM2 for a small English model |

---

## 5. The options, for the navigator

**THIS IS A DECISION FOR THE NAVIGATOR.**

```
   ┌── THE VOCABULARY, side by side (d 256 centre; d 768 in brackets) ─────────────────
   │
   │  A  KEEP 129,280      head 95.4% (91.4%) of FLOPs · table 33.6 MB (99.8)  · 4.814 B/tok
   │                       speed x1.00 · work: none · id match with ds4-flash kept
   │
   │  B  32,768 trained    head 83.9% (72.8%) · table 8.5 MB (25.3)   · 4.664 B/tok (-3.1%)
   │     on FineWeb-Edu    bytes/s x1.49..x3.36 (x1.99..x3.05) SCALED
   │                       work: fit (seconds), re-tokenize shards, new TEST, site tokenizer
   │
   │  C  65,536 adopted    head 91.3% (84.3%) · table 17.0 MB (50.6)  · 4.877 B/tok (+1.3%)
   │     (nanochat's)      bytes/s x1.31..x1.91 (x1.53..x1.84) SCALED
   │                       same work as B; best packing measured; least quality risk
   │
   │  D  KEEP the ids,     sampled softmax early in training (modded-nanogpt #92): training
   │     cut the head        speed only; download, browser work and unused rows unchanged
   │                       or a trimmed DeepSeek table (ids stay DeepSeek's): 32k packs 4.347
   │                         B/tok (-9.7%), worse than B at the same size
   │
   │  each letter is the answer; the measurement that decides it is one paired wave (section 6)
```

| option | training speed | browser size and work | quality evidence | work | risk |
|---|---|---|---|---|---|
| **A** keep 129,280 | x1 (MEASURED 46,700 tok/s d 256, 22,800 d 768, Spark) | table 33.6 MB / 99.8 MB int8; head ~90% of browser work per token (site engine header) | today's champion 1.40285 bpb was made with it (MEASURED, ours); nanowhale-100m's card calls the 129K table at 110M a limit on capacity (MEASURED elsewhere) | none | low; leaves 95% of compute in a table 61% of whose rows are unused |
| **B** 32,768, trained on FineWeb-Edu | x1.49 to x3.36 bytes/s at d 256, x1.99 to x3.05 at d 768 (SCALED) | table 8.5 / 25.3 MB; head work 3.9x less per token, 3.4x less per byte at d 256 | Tao et al. law points to ~14k to 21k at our size (SCALED); Lifecycle-Optimal: flat bpb at 100M (best 16k), 32k for batch-1 devices; nanochat HEAD and its regex validation are at 32k (MEASURED elsewhere); no measurement on our model | fit (10 s), re-tokenize ~3B tokens of shards (CPU hours), token_bytes, new TEST by document, site tokenizer change, int8 export rebuild | medium; the loss of 3.1% bytes per token might cost more bpb than the extra training buys; must be measured |
| **C** 65,536, nanochat's published tokenizer | x1.31 to x1.91 (d 256), x1.53 to x1.84 (d 768) (SCALED) | table 17.0 / 50.6 MB | packs our text best of 19 measured (4.877 B/tok); Lifecycle-Optimal: bpb-optimal at 65k for 1.3 to 2.3B | as B, minus the fitting; tiktoken pickle must be converted to tokenizer.json for the site | low on quality; leaves 91% of d 256 compute in the head |
| **D1** keep tokenizer, sampled softmax | modded-nanogpt: their three head GEMMs and CE shrink "by 2-5x" with 10k to 25k candidates; whole-record gain not separable (MEASURED elsewhere) | unchanged | biased loss early; their run finishes on the full softmax | an engineering lane (candidate sets, a kernel path) | medium; touches only training, and our measured speed is not FLOP-bound at d 256 |
| **D2** keep ids, trim to first 32k or 65k by rank | as B or C | as B or C | packs worse than B or C at equal size (4.347 at 32k, 4.631 at 65k) | as B, plus fixing the trimmed decoder | medium; buys an id subset of DeepSeek's, useful only if a ds4-flash distillation is planned |

**Recommendation (ESTIMATE until the wave in section 6 runs): option B, a 32,768-token byte-level BPE fitted
on FineWeb-Edu with nanochat's split regex,** gated by one paired wave against A and C.

Why:
- The head is 95% of this model's compute at d 256 and 91% at d 768. No body change can buy back what the
  vocabulary costs, and the browser pays the same tax on every generated token.
- Every independent line of evidence for a from-scratch small English model lands between 16k and 65k: the
  vocabulary scaling law (14k to 21k, SCALED), the lifecycle study (16k best at 100M, 32k for batch-1 devices),
  and the from-scratch small models (SmolLM2 49k, nanochat 32k, modded-nanogpt 50k).
- 32k costs only 3.1% more tokens than today on our text, while cutting FLOPs per byte to 0.30 and the token
  table to a quarter. 16k costs 9.4% more tokens for a smaller further gain, and is the right second arm. 65k is
  the low-risk fallback if 32k loses on bpb.
- The one recorded reason to keep DeepSeek's table is ds4-flash logit distillation, which this push excludes.

**The main trade:** about 1.5x to 3.4x more text trained per second and a token table four times smaller, against
re-tokenizing every shard, rebuilding the site's tokenizer, and giving up the id match with ds4-flash.

---

## 6. The test that decides it (a proposal for the coordinator, not run)

One paired wave at the wave-1 protocol (d 256, 20M tokens of training text, centre settings), with only the
tokenizer changed, the training text held fixed in **bytes** (20M DeepSeek tokens = about 96.3M bytes; the same
documents re-tokenized for each arm), scored on TEST-by-document (section 3.1):

| arm | tokenizer | what it answers |
|---|---|---|
| V129k | DeepSeek V4 (today) | the bridge to wave 1 |
| V65k | nanochat published 65,536 | best packing, half the table |
| V32k | trained 32,768 | the recommendation |
| V16k | trained 16,384 | the scaling-law point |

Score: bpb on the same TEST bytes, paired by document; tokens a second and bytes a second on the Spark, stamped.
The plan's rule applies: a difference counts at 0.003 bpb and paired z of 3. **Matching bytes rather than tokens
is the fair budget** (Limisiewicz et al.: scale with bytes). A second budget point at the same wall-clock time
would show the speed gain turned into quality. Work before it can run: re-tokenize the 65.5M-token `train` shard
and the val shard four ways (CPU minutes, MEASURED fitting speed above), four token_bytes tables, and a TEST
scorer that windows by document. ESTIMATE: under a day of one lane's work, four Spark runs of about 7 minutes each
at today's rate.

---

## 7. Side by side with the best training setups

Every cell is read from the primary source named in the column header, on 2026-10-03, unless labelled.

| | **DeepSeek-V4-Flash** (arXiv 2606.19348 v1, 2026-04-26; card `DeepSeek-V4-Flash-0731` at `7872f01b`) | **Qwen3-0.6B / 1.7B** (arXiv 2505.09388 v1) | **SmolLM3-3B** (HF blog 2025-07-08) | **nanochat** (HEAD `92d63d4e`, 2026-07-03) | **modded-nanogpt** (record 92, 2026-08-30; HEAD `4ea6b937`) | **OURS** (`track4_sdmonly_train.py` + `track4_sdmllm_train_one_arm.py`) |
|---|---|---|---|---|---|---|
| tokenizer, vocabulary | DeepSeek-V3 tokenizer plus "a few special tokens", "still remain the vocabulary size to be 128K"; config 129,280; untied | byte-level BPE, 151,669; tied (0.6B, 1.7B) | Llama 3.2, 128k; tied | own BPE (rustbpe), regex `\p{N}{1,2}`, default 32,768 (published checkpoints 65,536); untied | GPT-2 BPE 50,257 (head padded to 50,304); tied, untied at 2/3 of training | DeepSeek V4, 129,280; tied |
| context length | 4K, extended to 16K, 64K, 1M | 4,096 (S1, S2), 32,768 (S3) | 4,096; then 64k on 100B tokens, YaRN to 128k | 2,048; sliding windows "SSSL" | 896, then 2,048, then 3,072 by stage | T = 256 tokens; the model's own context is 8 back tokens plus 5 moving averages |
| data and stages | >32T tokens; web, math, code, long documents; agentic data in mid-training; documents packed from different sources | 36T, 119 languages: S1 >30T general, S2 ~5T STEM/code/reasoning/synthetic, S3 long context | 11.2T: stage 1 web 85 / code 12 / math 3; stage 2 75/15/10; decay 63/24/13 | NVIDIA ClimbMix, one stage, then SFT | FineWeb (GPT-2 tokens), under 330M tokens | FineWeb-Edu sample/10BT, one stage; 20M (wave 1), 600M (champion), 3B planned |
| optimiser, groups | Muon for most matrices; AdamW for embedding, head, RMSNorm weights, mHC biases and gates; AdamW β (0.9, 0.95), ε 1e-20, wd 0.1; Muon momentum 0.95, wd 0.1, update RMS rescaled to 0.18 | NOT DISCLOSED (hyper-parameters "predicted" from scaling laws per stage) | AdamW (0.9, 0.95), wd 0.1, clip 1; no weight decay on embeddings | Muon (momentum 0.95, ns 5, lr 0.02, cautious wd 0.28) for matrices; AdamW for lm_head (lr 0.008·s, β 0.8/0.96), embedding (lr 0.3·s, β 0.8/0.995), scalars; s = (d/768)^-0.5 | "ANVIL" (Muon-derived) for matrices; sparse row-wise embedding updates every 4 steps (β1 0, one second moment per row); embedding and head gradients accumulated 2 steps | AdamW (0.9, 0.95), ε 1e-8, one lr 3e-3 for embedding/head and body; store lr x3; clip 1.0 |
| schedule | warmup 2,000 steps, constant 2.7e-4, cosine to 2.7e-5 near the end | NOT DISCLOSED in detail; "accelerate the learning rate decay" in S2 | WSD: 2,000 warmup, linear decay to 0 over the final 10%; peak 2e-4 | warmup 40 steps, warmdown over the last 65%, final 5% of peak | cooldown over 80% of steps; staged lr multipliers | 2% warmup, cosine to 10% (default); WSD 20% decay as an option |
| weight decay | 0.1 (AdamW and Muon) | NOT DISCLOSED | 0.1; none on embeddings | Muon 0.28 cautious, scaled with batch and horizon; embedding 0.001; head 0.01 | cautious decay tied to lr | 0.1 on 2-D non-embedding; 0 on embedding/head; 0 on the store |
| initialisation | NOT DISCLOSED (mHC gates "initialized to small values") | NOT DISCLOSED | NOT DISCLOSED | embedding N(0, 0.8); head N(0, 0.001); output projections zero; others uniform, std 1/√d | projections zero ("muP-like") | embedding and 2-D weights N(0, 0.02); store values zero; sub-keys N(0, 1) |
| normalisation | RMSNorm; RMSNorm on query heads; mHC residuals | RMSNorm pre-norm, QK-Norm | NOT DISCLOSED in the blog; NoPE every 4th layer | parameter-free RMSNorm; logit softcap 15 | QK-Norm; softcapped logits | RMSNorm on each feature and before the head; BatchNorm without affine on queries; no softcap |
| precision | FP8 training framework (§5.2.1 "existing FP8 training framework"); MoE experts FP4 by quantisation-aware training | NOT DISCLOSED | NOT DISCLOSED in the blog | bf16; FP8 option used by the speedrun | FP8 head, MLP and QKV; bf16 elsewhere | bf16 autocast, fp32 weights, fp32 cross-entropy |
| batch | ramps to 75.5M tokens | NOT DISCLOSED | 2.36M tokens | auto by B ∝ D^0.383; the leaderboard moved to 1M tokens | 131k, 262k, 393k tokens by stage, taper to 328k | 8,192 tokens (32 x 256) |
| tokens per parameter | 32T / 284B total = 113; / 13B active = 2,462 (SCALED) | 36T / 0.6B = 60,000 (SCALED) | 11.2T / 3B = 3,733 (SCALED) | 8 (speedrun) to 12 (default) per scaling parameter (matrices + head) | ~330M tokens, ~2.7 per 124M dense parameters (SCALED; plus a 65B-entry hashed n-gram table) | 0.2 per parameter at 20M (d 256, 101.6M params); 5.2 at 600M (champion, 115.8M); 9.7 for 3B at the SDM-only d 768 shape (309.3M) (SCALED) |
| evaluation | base: MMLU family, BBH, HellaSwag, GSM8K, MATH, HumanEval, LongBench-V2 and more | base: knowledge, reasoning, code, multilingual benchmarks | HellaSwag, ARC, Winogrande, CommonsenseQA, MMLU-CF, GSM8K, MATH, HumanEval+, MBPP+ | CORE (22 tasks) and val bpb | validation loss 3.28 on FineWeb | TEST bpb on held-out FineWeb-Edu, paired by window; zero-read and shuffled-value ablations; no benchmark suite |
| multi-token prediction | depth 1, weight 0.3 then 0.1 | NOT DISCLOSED | NOT DISCLOSED | no | yes (weights 1, 0.5, 0.25, falling by stage) | no |

Two cells need care. DeepSeek's "128K" and its config's 129,280 are the same table with added tokens. Qwen3's
small models were also post-trained by "Strong-to-Weak Distillation" (§4.5), which our no-teacher rule excludes.

### 7.1 Where ours differs from the consensus of the best

```
   the three largest gaps, by how much of our compute they touch

   1 vocabulary      ███████████████████████████████████████ 95% of FLOPs sits in the head
   2 head optimiser  ███████████████████████████████████████ same 95%: one lr, tied, wd 0
   3 batch           ███ 8,192 tokens a step, 16 to 9,200 times smaller than the others
```

1. **The vocabulary is sized for a 284B multilingual model, not for ours.** The from-scratch small models use 32k
   to 65k; our head is 95% of FLOPs at d 256. Worth an arm: **yes**, the wave in section 6.
2. **The token table gets no optimiser treatment of its own.** Every recipe that publishes groups gives the
   embedding and the head their own group: DeepSeek keeps them on AdamW beside Muon, nanochat gives the embedding a
   learning rate 37 times the head's and different betas, modded-nanogpt updates embedding rows sparsely and unties
   the head late. Ours is tied, shares the body's 3e-3, and has no weight decay. Our own SDMLLMSTORE result points
   the same way (an extra per-token input vector won 0.008 bpb in three seeds). Worth arms: **yes**, two cheap ones:
   an untied input table (same as the SDMLLMSTORE winner, now at the SDM-only shape), and a head learning-rate
   multiplier of x0.3 and x3 on the tied table.
3. **The batch is tiny.** 8,192 tokens a step against 131k (modded-nanogpt's first stage) to 75.5M (DeepSeek) elsewhere. At small scale a small batch is not
   wrong in tokens, but it means 366,000 steps for a 3B-token run and leaves GPUs launch-bound. nanochat's rule
   B ∝ D^0.383 gives about 1.9x batch per 5x tokens (BASEDATA §4). Worth an arm: **yes, when the long run is
   planned**: B 64 and B 128 with the learning rate scaled by √(B/32), scored at equal tokens.

Smaller gaps, each already covered or cheap:
- **Muon for the body matrices** (DeepSeek, nanochat, modded-nanogpt): already an opt-in flag (`--body-opt muon`,
  lane SPARSEROWS). Our body is 3.2M FLOPs of 69.4M, so its reach is small at d 256.
- **Long-constant schedules** (DeepSeek constant then cosine, SmolLM3 WSD, nanochat 65% warmdown, modded 80%
  cooldown): the WSD option exists and `p0_A_wsd` is in wave 1.
- **Logit softcap** (nanochat 15, modded-nanogpt): absent in ours; one line in the loss; a cheap arm with a
  129,280-way tied head.
- **Context length**: the others train at 2k to 4k tokens. Our model reads 8 back tokens and moving averages, so T
  only sets how far the averages reach and how documents mix inside a window; not an arm.
- **Multi-stage data**: the small-model evidence favours one stage (BASEDATA §2); not an arm.

---

## Sources

Read 2026-10-03 UTC. Files downloaded to `/private/tmp/vocabside/` at the revisions shown.
- Tao et al., arXiv 2407.13623: https://arxiv.org/abs/2407.13623 and https://arxiv.org/html/2407.13623v3
- Mittal et al., arXiv 2608.11361: https://arxiv.org/abs/2608.11361 and https://arxiv.org/html/2608.11361
- Limisiewicz et al., arXiv 2605.01188: https://arxiv.org/abs/2605.01188
- Huang et al., arXiv 2501.16975: https://arxiv.org/abs/2501.16975
- DeepSeek-V4 report, arXiv 2606.19348 v1 (PDF read in full, §2.4, §4.1, §4.2, §4.3, §5.2.1): https://arxiv.org/abs/2606.19348
- DeepSeek-V4-Flash-0731 card and config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 (revision `7872f01b`)
- Qwen3 report, arXiv 2505.09388 v1 (PDF): https://arxiv.org/abs/2505.09388
- SmolLM3 blog: https://huggingface.co/blog/smollm3
- SmolLM2 paper: https://arxiv.org/html/2502.02737v1
- Gemma 3 270M: https://developers.googleblog.com/en/introducing-gemma-3-270m/
- nanochat: https://github.com/karpathy/nanochat (`README.md`, `dev/LOG.md`, `dev/LEADERBOARD.md`, `nanochat/gpt.py`,
  `nanochat/tokenizer.py`, `scripts/tok_train.py`, `scripts/base_train.py`, `runs/speedrun.sh`)
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt (`README.md`, `train_gpt.py`,
  `track_1_short/config.py`, `track_1_short/sampled_softmax.py`)
- Hugging Face model repos (API and files): `openai-community/gpt2`, `HuggingFaceTB/SmolLM2-135M`,
  `HuggingFaceTB/SmolLM3-3B`, `HuggingFaceTB/nanowhale-100m-base`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3.5-0.8B`,
  `unsloth/gemma-3-270m`, `google/gemma-4-E2B-it`, `openbmb/MiniCPM4-0.5B`, `openbmb/MiniCPM5-1B`,
  `karpathy/nanochat-d32`, `karpathy/nanochat-d34`, `nanochat-students/base-d20`
- Search only (UNVERIFIED): the critique of arXiv 2608.11361's batch-shift claim (pith.science summary).
- Local: `track4_sdmonly_models.py`, `track4_sdmonly_train.py`, `track4_sdmllm_train_one_arm.py`,
  `track4_sdmllm_data_provenance.json`, `track4_sdmllm_tokenizer_provenance.json`, `REPORT_SDMLLM_S0.md`,
  `REPORT_SDMLLMSTORE.md`, `RESEARCH_BASEDATA_2026-10-04.md`, `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`,
  `SDMONLY_CHANNEL.md`, `../../../SETTLE/SETTLE_CAMPAIGN_2026-09-30.md`, `sites/settle-site/src/engine/sdmchat.js`,
  the root `CLAUDE.md`.
