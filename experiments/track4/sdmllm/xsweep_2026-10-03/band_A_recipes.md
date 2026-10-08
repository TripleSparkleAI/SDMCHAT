# Band A: small-model pretraining speed and recipes (field sweep 2026-10-03)

Window: 2026-08-01 to 2026-10-03, widened to 2026-06-01 where a query was thin.
Fetches ran 2026-10-03 between 10:46 and 10:54 UTC.
The X API returned HTTP 402 (credits depleted), so this band uses the web-mirror fallback: the GitHub REST API, raw.githubusercontent.com, the arXiv Atom API, the Hugging Face Hub API, WebSearch and WebFetch.
The unauthenticated GitHub API hit its rate limit at 10:52 UTC (HTTP 403, "API rate limit exceeded"); later GitHub reads used raw.githubusercontent.com.
Every query has a raw file in `raw/A_qNN_*.md` with the exact query, the fetch time and the verbatim result list.

## 1. Search matrix

"In window" counts arXiv entries published on or after 2026-06-01. "Relevant" counts the entries that bear on a non-attention model with a large tied head and a table-lookup body.

| id | target | tool | hits | relevant |
|---|---|---|---|---|
| q01 | modded-nanogpt README record table | GitHub API | table ends at record 92; 0 new records | 0 new |
| q02 | modded-nanogpt commits since 2026-08-01 | GitHub API | 24 commits | 3 (record merges and the trainer refactor) |
| q03 | modded-nanogpt PRs, newest 40 | GitHub API | 40 PRs | 11 bodies read |
| q04 | modded-nanogpt optimization track (track 3) table | raw README | 46 rows, last dated 2026/06/19; 0 rows in window | 2 open PRs read |
| q05 | nanochat commits, dev/LOG.md and PRs since 2026-08-01 | GitHub API | 0 commits since 2026-08-01; last LOG.md entry 2026-05-05; 40 PRs | 5 PR bodies read |
| q06 | Muon variants | arXiv Atom | 60 in window | 8 |
| q07 | schedule-free, SOAP, Shampoo, Adam-mini, MARS, Lion | arXiv Atom | 33 in window | 2 (most "MARS" hits are unrelated) |
| q08 | embedding initialisation, tied embeddings, output head | arXiv Atom | 22 in window | 3 |
| q09 | z-loss, logit soft-capping, value embeddings | arXiv Atom | 12 in window | 2 |
| q10 | norm placement in small LMs | arXiv Atom | 0 in window (EMPTY) | 0 |
| q11 | muP and u-muP | arXiv Atom | 14 in window | 1 |
| q12 | learning-rate and batch-size scaling | arXiv Atom | 9 in window | 5 |
| q13 | tokens-per-parameter overtraining | arXiv Atom | 24 in window | 4 |
| q14 | large-vocabulary head cost | arXiv Atom | 27 in window | 1 |
| q15 | speedrun records September 2026 | WebSearch | 9 | 3 |
| q16 | x.com mirror for the ANVIL record | WebSearch | 9 + 9 | 0 x.com posts about record 92 |
| q17 | Muon for embeddings and heads | WebSearch | 10 | 2 |
| q18 | AutoTrust 24.9 s claim, Ember, Token Geometry | raw README + WebFetch | 3 primaries read | 3 |
| q19 | primary reads of 6 optimizer and head papers | WebFetch arXiv | 6 abstracts + 1 HTML body | 6 |
| q20 | Hugging Face Hub small-model releases | HF API | 15 + 15 + 2 | 0 |
| q21 | schedule-free 2026 | WebSearch | 10 | 0 in window |
| q22 | optimiser state for sparse table rows | WebSearch | 10 | 2 primary checks; EMPTY for the exact question |

## 2. Ranked signals

States: CONFIRMED means the number was read in the primary source. For a pull request, the primary source is the PR body with its attached logs, so CONFIRMED means "the claimant's own measurement, as written", not "accepted by the maintainers". POST CLAIM ONLY means only a post or a press-style page says it. UNVERIFIED means the primary was not readable or not found.

### S1. Head-gradient sampling and adaptive softmax keep cutting speedrun time after record 92
- **Claim:** two open or closed PRs reduce the cost of the LM head beyond record 92's sampled softmax. PR 371 samples one activation/gradient pair per group of four token rows for the head weight gradient. It reports 0.759 s (1.88%) saved together with a truncated attention backward. PR 372 (closed, not merged) replaces the full softmax with an adaptive softmax over a frequent head of 18,430 tokens plus two tail clusters.
- **Who / date:** romeerp, PR 371, 2026-09-25. ldmberman, PR 372, 2026-09-28.
- **Links:** https://github.com/KellerJordan/modded-nanogpt/pull/371 · https://github.com/KellerJordan/modded-nanogpt/pull/372
- **Primary checked:** both PR bodies (raw/A_q03).
- **State:** CONFIRMED in the PR bodies. Neither is an accepted record. The 0.759 s mixes two methods; the PR gives no split between them.
- **Verbatim (371):** "sample one activation/gradient pair per group of four token rows and multiply its contribution by four. This reduces the dense dW GEMM's reduction dimension by 4×." and "The head's input gradient is unchanged, so all tokens still contribute to training the preceding layers."
- **Verbatim (372):** "The new version takes 65.3 seconds to reach the target loss at step 1325." The title states "(-2.3s, +35 steps)".

### S2. Copying from earlier in the same document is the largest open speed lever, and one form needs no attention
- **Claim:** PR 379 (CPLM) mixes the LM softmax with a pointer over the document's previous tokens and reports 1050 steps against 1194 and an 11.25% same-node speedup on record 92. PR 378 feeds a "candidate next token" found by the longest earlier exact match in the same document as an extra input. It uses hashing and one sort, no attention, and reports 62 fewer steps and a 2.9% speedup. PR 367 trains a separate CPU exact-match retrieval model on the full 10.3B training tokens and reports a 46.1% speedup.
- **Who / date:** NathanGodey, PR 379, 2026-10-02. josusanmartin, PR 378, 2026-10-02. hermabr (Herman Brunborg), PR 367, 2026-09-17.
- **Links:** https://github.com/KellerJordan/modded-nanogpt/pull/379 · https://github.com/KellerJordan/modded-nanogpt/pull/378 · https://github.com/KellerJordan/modded-nanogpt/pull/367
- **Primary checked:** all three PR bodies (raw/A_q03).
- **State:** CONFIRMED in the PR bodies; all three are open. PR 378 says its own share is uncertain: "the copy feature's share is roughly 0.5 to 0.75 s". PR 367 retrieves over the whole training stream with a second model. That is a different kind of method from in-document copying, and its acceptance under the rules is not settled.
- **Verbatim (379):** "| **delta** | | **−4.566 s (11.25 %)** | +0.33 millinats |" and "The next-token distribution becomes a mixture of the LM softmax and a pointer over the document's previous tokens".
- **Verbatim (378):** "Each position gets a **candidate next token**: the token that followed the longest earlier exact match of its (normalized) context in the **same document**, fed to the model as an extra input during training." and "One stable sort finds each position's latest earlier occurrence of the same key."

### S3. Sparsely touched tables train better with per-row optimiser state than with dense Adam state
- **Claim:** three independent recipes drop Adam's per-element second moment for token and n-gram tables. Record 92's README uses beta1 = 0 and one second-moment value per row. nanochat PR 854 trains a 300M-row hashed table with a row-wise RMSprop, one scalar per row, and reports 0.703668 val bpb against 0.71800 in 91.74 min against 99 min. Ember (arXiv 2607.01455) uses a row-by-column factored second moment with no first moment on the embedding and the LM head. Its README claims parity with tuned Adam on the speedrun, and claims that Adam's second moment goes stale on rare rows.
- **Who / date:** record 92 README line (merged 2026-09-28). nihir27, nanochat PR 854, 2026-09-11. Kathan Shah, arXiv 2607.01455, v1 2026-07-01, v3 2026-07-15; Ember speedrun PR 346, 2026-07-24.
- **Links:** https://github.com/KellerJordan/modded-nanogpt · https://github.com/karpathy/nanochat/pull/854 · https://arxiv.org/abs/2607.01455 · https://github.com/katop1234/ember
- **Primary checked:** README line 46 (raw/A_q01, A_q22), PR 854 body (raw/A_q05), the Ember README and the arXiv abstract (raw/A_q18).
- **State:** record 92 recipe CONFIRMED. PR 854 numbers CONFIRMED in the PR body (open PR). Ember parity and the stale-row claim: POST CLAIM ONLY, because they appear in the README and not in the paper abstract, and the paper body was not read.
- **Verbatim (README):** "Sparse row-wise embedding updates: only touched rows are exchanged and updated (beta1 = 0, one second-moment value per row), every 4 steps"
- **Verbatim (854):** "Row-wise optimiser on the GPU: state is one scalar per row and Engram layer, kept in VRAM."
- **Verbatim (Ember README):** "Adam's dense second moment goes stale on rare rows". The same README sentence gives the size as steps 10^2 to 10^4 times too large late in training, against 1 to 3 times for Ember's row statistic (the full sentence is in raw/A_q18).

### S4. Adam-family optimisers bias rare tokens downward in the softmax head
- **Claim:** a theory paper with small-model checks argues that coordinate-wise adaptive optimisers move the training fixed point of rare-token logits below their data frequency. SGD, AMSGrad, Shampoo and Muon do not. The size of the effect depends on the gap between a token's appearances, measured in units of 1/(1 - beta2).
- **Who / date:** Sangsidhya Kar, arXiv 2609.37535, 2026-09-29.
- **Link:** https://arxiv.org/abs/2609.37535
- **Primary checked:** the abstract (raw/A_q19).
- **State:** CONFIRMED as a statement in the abstract. The abstract gives no effect size for a real language model.
- **Verbatim:** "if a token is absent for at least two consecutive minibatches, its equilibrium probability is strictly below its data frequency for every learning rate" and "in the language model, the optimizers with the biased fixed point also fit the generating distribution less well."
- **Relevance:** our tied head has 129,280 rows and we train it with AdamW at beta2 0.95, so most rows are rare in the sense of this paper.

### S5. Tail-EMA weight blending at the end of training lowers validation loss at almost no cost
- **Claim:** blending the final weights toward an EMA of the last about 150 steps lowers val loss by 0.0024 on the speedrun trainer. Optimization-track record 45 adds the same readout.
- **Who / date:** dnhkng, modded-nanogpt PR 364, 2026-09-05 (closed, not merged). Optimization-track record 45, jn2clark, 2026/06/12.
- **Links:** https://github.com/KellerJordan/modded-nanogpt/pull/364 · https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md
- **Primary checked:** PR 364 body and the track 3 table row 45 (raw/A_q04).
- **State:** CONFIRMED in the PR body and the merged track 3 row. The PR author says the wall-clock gain is not shown end to end. nanochat's LOG.md (2026-03-24) records that EMA/SWA "did not help" there, so the result is mixed across codebases.
- **Verbatim (364):** "**-0.0024 at gamma = 0.25-0.30.** Broad optimum: horizon 130-170 x gamma 0.25-0.35 all land within 0.0002" and "gamma is swept **within a single run**, against the same final weights, the same validation data, and the same forward pass".
- **Verbatim (track 3 row 45):** "Setup from #44, plus at the final step blend the weights towards EMA(horizon = 150 steps)".

### S6. Weight decay that also follows the learning-rate schedule
- **Claim:** Apte proposes multiplying the decay coefficient by eta_t / eta_max on top of the usual eta_t coupling. With Muon on MoE models from 72M to 930M at about 600 tokens per active parameter, the paper reports reaching the same loss 30% faster. modded-nanogpt PR 369 uses the same idea ("Defazio-style decay") in a 3060-step optimization-track submission.
- **Who / date:** Anuj Apte, arXiv 2607.23777, 2026-07-26. jeffreycider, PR 369, 2026-09-20 (open).
- **Links:** https://arxiv.org/abs/2607.23777 · https://github.com/KellerJordan/modded-nanogpt/pull/369
- **Primary checked:** the abstract and the HTML body equations (raw/A_q19), and the PR body (raw/A_q03).
- **State:** CONFIRMED in the abstract; measured with Muon on MoE models, not with AdamW.
- **Verbatim:** "reaching the same validation loss 30% faster at our largest scale across models from 72 - 930 million parameters trained at ~600 tokens per active parameter." The body equation, as returned: "W_{t+1}=(1−η_t²λ/η_max)W_t−η_t U_t".
- **Note:** torch AdamW already multiplies decay by the scheduled lr. This proposal adds a second factor.

### S7. Optimiser and hyperparameter preferences change with the overtraining factor
- **Claim:** across 51M to 253M models and overtraining factors from 1x to 256x, the best weight decay scales about as sqrt(OT), the preferred schedule can reverse, and Muon and SOAP keep a roughly constant token-efficiency advantage over AdamW.
- **Who / date:** Katie Everett, Shikai Qiu, arXiv 2609.04577, 2026-09-04.
- **Link:** https://arxiv.org/abs/2609.04577
- **Primary checked:** the abstract (raw/A_q19).
- **State:** CONFIRMED in the abstract. The abstract gives no size for the Muon advantage.
- **Verbatim:** "The preferred learning rate schedule can reverse across the overtraining axis, the best weight decay coefficient scales approximately as sqrt(OT), and longer horizons generally favor longer fixed memory."

### S8. The tuned AdamW baseline needs about 1.5x the steps of tuned Muon on the speedrun task, and a bf16 embedding loses its late updates
- **Claim:** PR 353 tunes AdamW for the optimization track to 4950 steps, against 3250 for the tuned Muon baseline (track 3 row 36). PR 377 notes that a bf16 token embedding reaches |w| about 9, so that most late Adam updates round away, and adds an fp32 master copy.
- **Who / date:** konstmish, PR 353, 2026-08-14 (open). jn2clark, PR 377, 2026-10-01 (open).
- **Links:** https://github.com/KellerJordan/modded-nanogpt/pull/353 · https://github.com/KellerJordan/modded-nanogpt/pull/377
- **Primary checked:** both PR bodies (raw/A_q04, raw/A_q03).
- **State:** CONFIRMED in the PR bodies; neither is merged. The step ratio is for GPT-2 small on that fixed task, not a general constant.
- **Verbatim (353):** "`train_steps`: 3250 -> 4950" and "2D block-matrix AdamW betas: `(0.9, 0.9)`".
- **Verbatim (377):** "By step 2000 its entries reach `|w| ~ 9`, where most tail Adam updates are below half a bf16 ulp and round away. The optimizer keeps an fp32 master copy and fp32 Adam state for it".

### S9. AF-Muon: an AdamW-free Muon variant for tied vocabulary tables
- **Claim:** AF-Muon replaces the auxiliary AdamW on a tied vocabulary table with a support-aware finite-cap update. It reports better mean validation loss than Hybrid Muon across nine tied-token settings from 124M to 1B, with about 20% less optimizer-state memory.
- **Who / date:** Lagzian, Halvachi, Zhang, Lin, Liu, arXiv 2610.01395, 2026-10-01.
- **Link:** https://arxiv.org/abs/2610.01395 (code: https://github.com/arashlagzian/afmoun, not opened)
- **Primary checked:** the abstract (raw/A_q19).
- **State:** CONFIRMED in the abstract. The size of the loss improvement is not in the abstract.
- **Verbatim:** "AF-Muon improves mean validation loss and perplexity over both Hybrid Muon and a SCION-style Sign endpoint."

### S10. No new official record after 92; a 24.9 s claim is self-reported
- **Claim:** the modded-nanogpt track 1 table still ends at record 92 (0.665 minutes, merged 2026-09-28). The GPT-2 Medium table still ends at record 18 (2025-12-31). The optimization track still ends at row 46 (2026/06/19). nanochat has no commit since 2026-07-03, and no LOG.md entry since 2026-05-05. AutoTrust claims 24.90 s by fusing PR 360 with PR 367.
- **Who / date:** AutoTrust AI, 2026-09-25.
- **Link:** https://github.com/AutoTrustAI/nanogpt-speedrun-sota-by-guru
- **Primary checked:** the modded-nanogpt README (raw/A_q01), the commits (raw/A_q02), the nanochat commits and log (raw/A_q05), and the AutoTrust README (raw/A_q18).
- **State:** the absence of new records is CONFIRMED. The 24.9 s figure is POST CLAIM ONLY: it is the claimant's own repository.
- **Verbatim (AutoTrust):** "*Five-seed mean on 8×H100. The result is self-reported and is not yet an accepted leaderboard record.*"

### Lower-signal items, recorded but not ranked
- Head update geometry (arXiv 2608.22253, v2 2026-09-09): its RowNorm makes head steps more stable but raises mean final val loss by 0.0057 to 0.0153 at 190M to 640M (abstract, raw/A_q19). That is a cost for us, not a gain. CONFIRMED in the abstract.
- Z-loss backward geometry (arXiv 2609.16179, 2026-07-31): diagnostic only, with "comparable validation perplexity in low-coefficient regimes". CONFIRMED in the abstract.
- Effective-LR collapse (arXiv 2608.24814) and the nonlinear LR scaling paper (arXiv 2606.29158): both say LR transfer is cleaner in effective LR (LR divided by parameter norm). Abstracts read in the Atom results only.
- nanochat PR 850 (batch ramp 1/4 to 1/2 to full over the first 10% of tokens): closed with no result posted. PR 828 (smoothed squared-ReLU): one seed only. PR 830 (closed): a 49,152-token speedrun vocabulary with a fused cross-entropy kernel; the PR says the vocabulary gain "was not robust at d20/d26".
- PR 375 (normalising tokens before n-gram hashing, from the Engram paper): -0.42 s and -15 steps. It needs a hashed n-gram table, which we do not have.
- Schedule-free (ScheduleFree+, 2605.19095) and the norm-placement query: nothing in the window. The schedule-free summary claims are UNVERIFIED (raw/A_q21).
- Hugging Face Hub: no recipe-bearing small-model release in the window (raw/A_q20).

## 3. Proposed changes (at most 4)

### P1. Tail-EMA readout at the end of training
- **Source:** S5 (modded-nanogpt PR 364; optimization-track row 45).
- **Change:** keep an EMA of all weights over the final part of the run (horizon about 150 steps; scale the horizon if our step count differs a lot) and blend `w = (1 - gamma) * w_final + gamma * w_ema` before evaluation and export.
- **Test:** in one existing run, save `w_final` and the EMA, then evaluate TEST bpb at gamma in {0, 0.1, 0.2, 0.3, 0.4}. Gamma 0 must reproduce the logged TEST bpb exactly.

### P2. Per-row optimiser state for the product-key value tables and the tied head
- **Source:** S3 (record 92 README; nanochat PR 854; Ember) and S4 (arXiv 2609.37535).
- **Change:** for the 65,536-row value tables, and in a second arm also for the tied 129,280-row table, replace AdamW's per-element second moment with one second-moment scalar per row and beta1 = 0. Leave AdamW (0.9, 0.95) on everything else.
- **Test:** two arms at d 512 for 100M tokens with paired seeds and the same lr schedule. Compare TEST bpb, and log the update norm per row, bucketed by row access frequency, to see whether rare rows take oversized steps under AdamW.

### P3. Document-local exact-match copy candidate as an extra context feature
- **Source:** S2 (modded-nanogpt PR 378; the idea it builds on is PR 379).
- **Change:** for each position, hash the last n tokens for several n, find the latest earlier occurrence in the same document, and take the token that followed the longest match as a candidate. Add `embed(candidate) * scale[bucket] + bucket_embed[bucket]` to x, where the bucket encodes the match length. This uses no attention and fits our feature map from recent embeddings and EMAs.
- **Test:** two arms at d 512 for 100M tokens with paired seeds. Report TEST bpb overall and on positions with a match of length 4 or more, so the gain can be attributed.

### P4. Sampled head weight gradient for most of training
- **Source:** S1 (modded-nanogpt PR 371).
- **Change:** in the head backward pass, compute the dense part of the weight gradient from one row in each group of four (scaled by four), accumulate the target-position part exactly, keep the input gradient exact, and switch back to the full backward for the final part of the run. Because the head is about 87% of forward FLOPs, this targets our main cost.
- **Test:** measure tokens per second and TEST bpb at a fixed token budget (d 768, 600M tokens, the champion recipe) against the champion's 1.40285. Accept only if bpb is within seed noise and wall time drops.
