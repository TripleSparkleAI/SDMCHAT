# SDMLLM · Stage S0 · the SDM language model at the DeepSeek V4 vocabulary, 2026-09-30 to 2026-10-01

Lane SDMLLM (lane 4 of the SETTLE week, "SDM to LLM"). M5 only. No Spark, no rental, no paid API.
Code and results: `experiments/track4/sdmllm/`. Sealed predictions and scores:
`SETTLE/SETTLE_CAMPAIGN_2026-09-30.md` (SDMLLM lines).

## 1. The question

Can an SDM be the language model itself? The loop is: address the context, read a memory, emit the
next token, roll the emitted token back into the context. The model is trained the normal modern way
(next-token cross-entropy, AdamW, warmup and cosine) at the DeepSeek V4 vocabulary. It is compared
with a Qwen-style transformer of matched non-embedding size on the same tokens, and with the controls
the track4_245 lesson demands.

## 2. Tokenizer

- Source: the GGUF header of
  `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`. 5,248,121 bytes of
  metadata were read (GGUF v3, 62 keys, 1,328 tensors). The weights were never loaded.
- `tokenizer.ggml.model = gpt2`, `pre = joyai-llm`, 129,280 tokens, 127,741 merges, BOS id 0, EOS id 1.
- The GGUF stores only the pre-tokenizer's name, so the regex, normalizer and decoder come from the
  publisher's `tokenizer.json`. Source: `deepseek-ai/DeepSeek-V4-Flash` at revision `60d8d707`,
  sha256 `8f9f37ca...`.
- Checks: 129,280 of 129,280 token ids identical, 127,741 of 127,741 merges identical and in order.
  Round trip 12 of 12 varied texts plus a 10 kB source file. Encodings equal the publisher's on all of
  them.
- Negative control: dropping the first 1,000 merges changes 12 of 12 encodings.
- The `DeepSeek-V4-Flash-0731` repo's `tokenizer.json` (revision `7872f01b`) is byte identical.
- Provenance: `track4_sdmllm_tokenizer_provenance.json`, `track4_sdmllm_tokenizer_0731_crosscheck.json`.

## 3. Data

- `HuggingFaceFW/fineweb-edu`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, licence odc-by,
  config sample/10BT, file `000_00000.parquet`. Read row group by row group through HfFileSystem.
- Documents were taken in file order until the dataset's own GPT-2 token count reached 70M.
- Train: 64,488 docs, 65,522,263 tokens, 312,597,547 UTF-8 bytes. Held out: the last 2,000 docs,
  2,000,916 tokens.
- CALIB is the first half of the held-out stream and is used only to tune the count stores. TEST is the
  second half: 3,873 windows of 257 tokens, 995,323 scored targets, 4,824,566 bytes.
- Each document is BOS plus its ids. BOS targets are not scored.
- Only 78,533 of the 129,280 ids occur in the training shard.
- Byte identity check: summed token bytes minus UTF-8 bytes is +2,090 on train (7e-6 relative) and +57
  on val. The check is reported, not assumed.

Bits per byte, the headline unit:

    bpb = (sum over scored targets of -ln p(target)) / (ln 2 * sum over scored targets of bytes(target))

Plain reading: the average number of bits the model needs per byte of text. It does not depend on the
tokenizer, so any two models can be compared.

## 4. The models

All neural arms use d = 256, a tied 129,280 x 256 embedding (33,095,680 parameters), context T = 256,
batch 32, 20M training tokens (2,441 steps), AdamW (0.9, 0.95), peak learning rate 3e-3, warmup 2%,
cosine to 10%, weight decay 0.1 on 2-D non-embedding weights, grad clip 1.0, bf16 autocast on MPS.
The batch for step s depends only on (seed, s), so every arm with the same seed sees the same tokens in
the same order.

**(a) The yardstick, qwen:** 4 layers, 4 query heads over 2 key/value heads (grouped-query attention),
per-head RMSNorm on q and k, RoPE theta 10,000, SwiGLU with hidden size 688, pre-norm, final RMSNorm,
tied head. 2,902,784 non-embedding parameters.

**(b) The SDM language model, sdm.** Features from the context only:

    f_t = [ n(e_t), n(e_{t-1}), n(e_{t-2}), n(e_{t-3}), n(ema_0.8(e)_t), n(ema_0.97(e)_t) ]

Plain reading: the last four token embeddings plus two decaying averages of everything so far, each
normalised.

    x = W_x f_t

Plain reading: one linear map turns the features into the model's working vector.

For each of 2 hops, over one shared store of M = 3,600 locations with keys k_m and values v_m:

    q = BN(W_q^h n(x)),   s_m = q . k_m_hat,   C = top 2k of s,   theta = k-th largest s (k = 32)
    w_m = sigmoid((s_m - theta) / softness)  for m in C
    x <- x + (1/k) * sum over m in C of w_m v_m

Plain reading: build a query, score it against every location's key, and let locations fire with a
soft threshold centred at the 32nd best score. Add the firing locations' values, scaled by the expected
firing count as in SOFTSDM, and feed the result into the next hop's query.

    logits = n(x + SwiGLU(n(x))) E^T

Plain reading: a small readout MLP, then the shared embedding table used as the output layer.

The emitted token becomes e_{t+1} for the next step; that is the roll-back. The query batch norm (as in
product-key memory layers) was added before the seal, after a smoke run selected only 2.7% of the
locations. 2,898,432 non-embedding parameters.

**Controls for (b):**
- sdm_nostore: the hops removed, so features go straight to the readout (the 245 control). 924,160
  non-embedding parameters.
- sdm_mlp: each hop's store read replaced by a dense SwiGLU with the store's parameter count. 2,897,920.
- sdm_frozen: store values frozen random, keys trained.
- Eval-time on trained stores: shuffle_keys (permute the key rows) and zero_read.

**Extras, not all sealed in advance:** sdm_big with 16,384 locations (9.44M non-embedding), and sdm at
softness 2.0 (sealed as P11 after seed 0 was seen).

Forward FLOPs per token: qwen 72.5M, of which the tied output layer is 66.2M and the body 6.3M; sdm
72.2M (body 6.0M). The output layer is about ten times the rest of either model. The embedding holds
92% of each model's parameters.

**(c) Floors:** add-one unigram, and a Dirichlet backoff chain up to order 5:

    p_k(y | ctx) = (c(ctx, y) + m_k p_{k-1}(y)) / (c(ctx) + m_k)

Plain reading: trust the order-k counts in proportion to how often that context was seen, and fall back
to the shorter context otherwise. The masses m_k were tuned on a seeded 250k-target CALIB subsample.
TEST was scored once. The count stores use only context inside each window, as the neural models do.

## 5. Results (TEST bpb)

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| qwen | 1.6116 | 1.5580 | 1.5972 | 1.5889 |
| sdm | 1.6302 | 1.6293 | 1.6220 | 1.6272 |
| sdm_nostore | 1.6054 | 1.6061 | 1.6050 | 1.6055 |
| sdm_mlp | 1.6093 | | | |
| sdm_frozen | 1.6074 | | | |
| sdm_big (16,384 locations) | 1.6249 | | | |
| sdm softness 2.0 | 1.6082 | | | |

| floor | all 65.5M train tokens | 20M-token prefix |
|---|---|---|
| unigram | 2.3049 | 2.3068 |
| EXACT1 | 1.7700 | 1.8328 |
| EXACT2 | 1.7062 | 1.7933 |
| EXACT3 | 1.6991 | 1.7888 |
| EXACT5 | 1.6969 | 1.7873 |
| EXACT3, shuffled joint counts (negative control) | 1.7376 | 1.8068 |

Paired differences over the identical 3,873 TEST windows. The standard error comes from 2,000
bootstrap resamples of windows:

    delta = (sum_w nats_A(w) - sum_w nats_B(w)) / (ln 2 * sum_w bytes(w))

Plain reading: the bits per byte by which model A is worse than model B on exactly the same text.

| pair | delta bpb | SE |
|---|---|---|
| sdm minus sdm_nostore, seed 0 | +0.0248 | 0.00026 |
| sdm minus sdm_nostore, seed 1 | +0.0232 | 0.00028 |
| sdm minus sdm_nostore, seed 2 | +0.0170 | 0.00024 |
| sdm minus sdm_mlp | +0.0209 | 0.00030 |
| sdm minus sdm_frozen | +0.0227 | 0.00021 |
| sdm (softness 0.5) minus sdm (softness 2.0) | +0.0220 | 0.00021 |
| sdm_nostore minus sdm softness 2.0 | -0.0028 | 0.00022 |
| sdm_nostore minus sdm_big | -0.0196 | 0.00024 |
| qwen minus sdm_nostore, seeds 0 / 1 / 2 | +0.0062 / -0.0481 / -0.0078 | under 0.0005 |
| qwen minus sdm, seeds 0 / 1 / 2 | -0.0186 / -0.0713 / -0.0248 | under 0.0006 |

Eval-time ablations on trained stores:

| model | intact | shuffle_keys | zero_read |
|---|---|---|---|
| sdm seed 0 | 1.6302 | 1.6534 | 1.6566 |
| sdm seed 1 | 1.6293 | 1.6381 | 1.6393 |
| sdm_big | 1.6249 | 1.6508 | 1.6513 |
| sdm softness 2.0 | 1.6082 | 1.6114 | 1.6131 |
| sdm_frozen | 1.6074 | 1.6076 | 1.6076 |

Location use (distinct locations chosen as top k, last hop, over 64 test windows): sdm 40% and 44%,
sdm_big 9.4%, softness 2.0 57%.

## 6. Verdicts

- **The no-store control wins, in every seed.** The learned store costs 0.017 to 0.025 bpb against the
  same model without it, far outside both the bootstrap error (under 0.0003) and the SDM seed spread
  (0.008). A dense MLP of equal size and a store of frozen random values are both better than the
  learned store. The frozen store is ignored by the trained model. **The track4_245 lesson holds with
  no teacher in the loop:** the address and readout carry the model; the learned store does not.
- **The soft cut-off matters.** Softness 2.0 recovers 0.022 bpb and lands 0.003 from no store. The hard
  end of the dial hurts most, which points at noisy sparse gradients to the store rather than at the
  idea of a read.
- **The SDM architecture as a whole is competitive at this size.** With no attention, the no-store arm
  beats transformer seeds 0 and 2 and trails seed 1 by 0.048. The transformer's seed spread (0.054) is
  about fifty times the SDM family's, consistent with a transformer mid-transition at 20M tokens.
- **The embedding dominates at this vocabulary and budget.** 92% of parameters and 91% of FLOPs sit in
  the tied 129,280-token table. Only 60.7% of ids occur in the training shard. Everything here is
  within about 0.07 bpb, so S0 measures the table more than the body.
- **Neural models beat count stores built on the same amount of text by 0.16 to 0.23 bpb.** Against
  the count store built on the full 65.5M tokens, the SDM model's margin is 0.069.
- **The July SparseStar students** (trained on 16k to 100k tokens of ds4-flash text) score 2.80 to 3.02
  bpb on 400 TEST windows, worse than the unigram floor (`track4_sdmllm_prior_students_result.json`).

## 7. Free-running generation (six prompts, 128 new tokens, seed 0)

Mean statistics over the six prompts (`track4_sdmllm_samples_result.json`):

| model | greedy distinct-2 | greedy repeated-4-gram share | greedy longest repeat | t0.8 distinct-2 | t0.8 repeated-4-gram share |
|---|---|---|---|---|---|
| qwen | 0.180 | 0.749 | 42.2 | 0.919 | 0.001 |
| sdm | 0.176 | 0.756 | 32.7 | 0.919 | 0.007 |
| sdm_nostore | 0.096 | 0.889 | 54.3 | 0.912 | 0.003 |
| sdm_mlp | 0.161 | 0.801 | 45.5 | 0.928 | 0.004 |
| EXACT3 count store | 0.105 | 0.892 | 60.2 | 0.961 | 0.001 |

Excerpts at temperature 0.8 (prompt "The water cycle describes how water moves"):
- sdm: " the atmosphere is the most common form of energy is called energy development. The heat, temperature
  becomes a key to the atmosphere."
- sdm_nostore: " from Earth's surface. They are being seen to be removed and how our earth are the first to be."
- qwen: " in its water, the soil will be an important place for the soil."
- EXACT3: " at a certain number of ways to use the water is one of the few countries with a few minutes."

Greedy decoding loops in every model. The count store falls into "the United States, the United
States". This is the first free-running SDM rollout at a modern vocabulary since the July char-level
stores.

## 8. Stamps

Every run is stamped with load, power mode, macOS build 25F71 and torch 2.8.0. The load average ran
from 18 to 289 on 18 cores. The power mode was 1 (low) at the start of the first two runs and 2 (high)
afterwards. Throughput ranged from 1.4k to 32k tokens/s. No timing here is a claim; bpb does not depend
on timing.

## 9. The staircase (planned, not run)

Training compute is FLOPs = 6 N D, with N counting the tied embedding once as the output layer.
Plain reading: every token touches every weight about six times, counting the forward and backward
passes.

Effective throughput assumptions: H100 bf16 at 400 TFLOP/s (about 40% of peak); Spark GB10 at
50 TFLOP/s (assumed, not measured). This M5 under the week's load reached roughly 1 to 2 TFLOP/s on S0.

| stage | shape | non-embedding | embedding | total N | tokens (20N / 100N) | FLOPs | H100 GPU-hours | GB10 hours |
|---|---|---|---|---|---|---|---|---|
| S1 | d 1024, 16 layers | 0.19B | 0.13B | 0.32B | 6.4B / 32B | 1.2e19 / 6.2e19 | 9 / 43 | 69 / 344 |
| S2 | d 2048, 20 layers | 0.94B | 0.26B | 1.21B | 24B / 121B | 1.8e20 / 8.8e20 | 122 / 609 | 974 / 4,868 |
| 4B | Qwen3-4B shape, d 2560, 36 layers | 3.63B | 0.33B | 3.96B | 79B / 1T | 1.9e21 / 2.4e22 | 1,310 / 16,518 | 10,477 / 132,142 |

- At $2 to $3 per H100-hour: S1 costs $18 to $130, S2 $240 to $1.8k, and the 4B model $2.6k to $3.9k at
  79B tokens or $33k to $50k at 1T tokens.
- On the M5, S1 would take about 100 days. The Spark could carry S1 at the Chinchilla token count (20
  tokens per parameter) in about 3 days after HEADCLIMB ends. S2 and the 4B target need an approved
  rental.
- An SDM version at matched FLOPs could hold more store parameters than the transformer, because a read
  is sparse. S0 says that capacity pays only once the store's own training is fixed.

**What a local DeepSeek V4 teacher on the Spark could add later.**
- The teacher runs on the Spark with SSD streaming (13.4 t/s decode at 2k context was measured on the
  preview checkpoint).
- It can supply top-k logits for distillation through `--dump-logprobs`. The TWOSTUDENTS pool already
  holds about 1,528 ds4-flash top-128 dumps in this vocabulary.
- It can generate teacher text: one million tokens takes about 20.7 hours at the decode rate. The
  prefill rate for scoring existing text is not measured in this lane.
- A hidden-state tap on the DeepSeek path does not exist (the GLM tap wrote zero files there), so
  internal-state distillation is out of reach until one is built.

## 10. Owed, in order

1. Store optimisation: learning rate, no weight decay on keys and values, value initialisation. Memory
   layers use separate settings, and the evidence points at noisy sparse updates.
2. A softness sweep on the language model (0.25 to 8), jointly with lane SOFTSDM.
3. Two more transformer seeds, because its 0.054 spread leaves the yardstick loose.
4. A learned n-gram-hash address arm (the map's lever) at matched parameters.
5. S1 on the Spark after HEADCLIMB, or a priced rental.
6. The sealed TWOSTUDENTS preregistration, now that a vocabulary-matched pipeline exists.

## 11. Sealed predictions, scored

| id | sealed | result | verdict |
|---|---|---|---|
| P1 | qwen 1.50 ±0.12 | 1.589 | hit |
| P2 | sdm 1.66, worse than qwen by 0.10 to 0.25 | 1.627, gap 0.038 | miss |
| P3 | store helps, sdm minus nostore = -0.06 | +0.017 to +0.025 | miss |
| P4 | sdm_mlp within 0.03 of sdm | 0.021, MLP better | hit |
| P5 | frozen between sdm and nostore, about 1.70 | 1.607, in order | split |
| P6 | shuffle and zero cost at least +0.15 | +0.009 to +0.026 | miss |
| P7 | unigram 2.25, EXACT3 1.85 ±0.15, order 5 within 0.05 | 2.305, 1.699, 1.697 | miss by 0.001 on EXACT3 |
| P8 | every neural arm beats EXACT3 by 0.1 | sdm by 0.069 | miss |
| P9 | seed spread under 0.02 | qwen 0.054 | miss |
| P10 | sdm repeats more than qwen, both loop greedy | 0.176 vs 0.180, both loop | hit, marginal |
| P11 | softness 2.0 within 0.01 of 0.5 | 0.022 better | miss |

3 hits, 1 split, 7 misses.

## 12. Files

- `track4_sdmllm_tokenizer_from_gguf_header.py`, `track4_sdmllm_prepare_fineweb_edu_token_shards.py`
- `track4_sdmllm_models.py`, `track4_sdmllm_train_one_arm.py`, `track4_sdmllm_run_queue.sh`
- `track4_sdmllm_count_store_floors.py`, `track4_sdmllm_samples_and_repetition.py`
- `track4_sdmllm_paired_window_comparison.py`, `track4_sdmllm_score_prior_sparsestar_students.py`
- `runs/*.log`, `runs/*.result.json`, `*_result.json`, `*_provenance.json`
- Checkpoints (5.2 GB) and data shards (275 MB) are local and gitignored.
