# Band C: large-vocabulary output-layer tricks

Field sweep, 2026-10-03 (UTC 10:46 to 10:57). Window 2026-08-01 to 2026-10-03, widened to 2026-01 where the topic was thin. Older strong primaries are marked with their own dates.

The X API returned HTTP 402 (credits depleted), so this band uses the web-mirror fallback: the GitHub REST API and raw.githubusercontent.com, the arXiv Atom API, the HF Hub raw config files, WebSearch and WebFetch. No X post was fetched. No search surfaced an X post on this topic, so every engagement figure is unknown.

Raw records for every query are in `raw/C_qNN_*.md`. The search plan is `raw/C_q00_matrix_plan.md`.

Our model, for reference: an attention-free LM whose logits are `norm(x) . E^T`, with `E` the 129,280 x d DeepSeek V4 token table tied to the input. At d 768 the table is 99.3M of 115.8M parameters and the head is 86.7% of forward FLOPs. The trainer computes `self.hidden(idx) @ self.emb.weight.t()` with no logit cap (read in `track4_sdmonly_models.py`, `forward`).

## 1. Search matrix

| id | query (short) | tool | returned | in window | relevant |
|---|---|---|---|---|---|
| q01 | modded-nanogpt PR #360 body, files, 15 comments, record README | GitHub API + raw | 1 PR | 1 | 1 |
| q02 | modded-nanogpt PRs after #360 (list of 40, 10 bodies read) | GitHub API | 40 | 20 after #360 | 8 |
| q03 | "sampled softmax" / negative / candidate sampling | arXiv API | 40 | 40 (since 2025-06) | 0 |
| q04 | sampled softmax LM pretraining 2026 | WebSearch | 19 | 2 from 2026 | 0 new |
| q05 | low-rank / factorised / bottleneck LM head | arXiv API | 40 | 40 | 9 |
| q06 | adaptive / hierarchical softmax | arXiv API | 30 | 5 (since 2025-06) | 0 |
| q07 | weight tying / tied / untied embeddings | arXiv API | 40 | 17 | 5 |
| q08 | SmolLM3 tied embeddings and reason | WebSearch | 19 | n/a | 2 |
| q08b | Smol Training Playbook, embedding sharing section | WebFetch (primary) | 1 | n/a | 1 |
| q09 | Gemma 3 270M embedding params, QAT | WebSearch | 9 | n/a | 2 |
| q09b | Google Developers Blog, Gemma 3 270M | WebFetch (primary) | 1 | dated 2025-08-14 | 1 |
| q10 | HF config.json for 14 small models | HF raw | 14 asked, 11 read, 3 HTTP 401 | n/a | 11 |
| q11 | nanochat untied head and learning rates | WebSearch | 10 | n/a | 1 |
| q11b | nanochat/gpt.py at master | raw GitHub (primary) | 1 | last edit 2026-07-03 | 1 |
| q12 | vocabulary size / trimming / pruning, small or on-device | arXiv API | 40 | 26 | 2 |
| q13 | vocabulary trimming and tokenizer transplant | WebSearch | 18 | n/a | 0 for from-scratch training |
| q14 | quantising embedding table or LM head | arXiv API | 21 | 8 | 3 |
| q14b | int4/int8 embedding tables on device | WebSearch | 9 | n/a | 2 |
| q15 | z-loss / logit soft-capping | arXiv API | 11 | 9 (since 2025-06) | 2 |
| q16 | sparse embedding gradients with a tied head | WebSearch | 24 | n/a | 1 (docs) |
| q17 | X mirror: sampled softmax speedrun posts | WebSearch | 15 | 0 X posts | 2 |
| q18 | rare tokens / output embeddings / training | arXiv API | 40 | 27 (since 2026-03) | 1 |
| q19 | record #92 sampled softmax mechanism: blog + code | WebFetch + raw GitHub (primary) | 3 sources | 1 | 1 |
| q20 | Coupled Adam, embedding LR vs vocab size, OEC | arXiv API | 5 | 3 (since 2024-06) | 3 |
| q21 | arXiv HTML checks of three papers | WebFetch (primary) | 3 | 3 | 3 |
| q22 | copy / pointer mixture in the output distribution | arXiv API | 29 | 3 (since 2026-06) | 0 |

Errors logged: the GitHub REST API returned HTTP 403 (unauthenticated rate limit) at 2026-10-03T10:52:53Z, after which raw.githubusercontent.com was used. Three HF repos (google/gemma-3-270m, google/gemma-3-1b-pt, meta-llama/Llama-3.2-1B) returned HTTP 401 because they are gated. One first run of q20 was written by a script another band had overwritten in a shared scratch folder; q20 was re-run with a private script and the file was replaced.

## 2. Ranked signals

States: CONFIRMED means the number is in the primary source as quoted. POST CLAIM ONLY means the number is the author's own, in an open or unreproduced PR or a single unreplicated paper. UNVERIFIED means the claim did not survive or could not be checked against a primary.

**S1. Record #92 sampled softmax has no logQ correction and uses uniform stride negatives plus every batch target.**
- Who: Deven Pietrzak (GitHub devenpzak), modded-nanogpt PR #360. Created 2026-08-31, merged 2026-09-28. Blog by Hyperstition, dated 2026-09-27.
- Link: https://github.com/KellerJordan/modded-nanogpt/pull/360 ; https://hyperstition.cc/training-nanogpt-in-39-9-seconds
- Primary checked: yes, record README, `train_gpt.py` and `triton_kernels.py` at sha 4f5270e5 (raw C_q01, C_q19).
- State: CONFIRMED for the mechanism and the published numbers. The ablation is mostly n=2, as the PR says.
- Verbatim (README): "the training CE is computed against a candidate set (every target in the batch, and every prefix target, plus a per-step duplicate-free stride sweep of negatives) instead of the full 50,304-way softmax: `P = 10,240` from step 0, 14,336 from 681, 24,576 from 961, off at 1101". And: "This biases the training gradient. Validation is always the full 50,304-way softmax".
- Verbatim (code): `_SNS_STRIDE = 20011          # coprime with the vocab, so the negative sweep is a permutation`.
- Verbatim (blog): "For omitted elements of the vocabulary, we keep the gradient at zero in the sampled softmax variant".
- Cost in loss: the per-mechanism table lists the "CE stack (sampled softmax + prefix-CE + fused kernel)" at wallΔ +8.80 s and valΔ -2.4 millinats. With the PR's net formula (8.80 minus 2.4 x 0.164 = 8.41), a negative valΔ reads as the stack costing about 2.4 millinats of val loss at equal steps. The candidate set is about 20% of the vocabulary in the first stage (10,240 of 50,304).
- Correction to a search summary: the search tool said the full softmax runs "for the first 100 steps". The code returns a positive candidate count from step 0. That summary claim is UNVERIFIED and contradicted by the code.

**S2. A later PR samples the LM-head weight gradient and claims a further 0.76 s on top of record #92.**
- Who: romeerp, modded-nanogpt PR #371, opened 2026-09-25, open.
- Link: https://github.com/KellerJordan/modded-nanogpt/pull/371
- Primary checked: yes, PR body (raw C_q02).
- State: POST CLAIM ONLY (open, not reproduced by a maintainer).
- Verbatim: "sample one activation/gradient pair per group of four token rows and multiply its contribution by four. This reduces the dense dW GEMM's reduction dimension by 4×." And: "The head's input gradient is unchanged, so all tokens still contribute to training the preceding layers." Result table: #360 at 40.342 ± 0.184 s (N=6) against this PR at 39.583 ± 0.266 s (N=12), "This saves 0.759 s (1.88%)". The PR's own text pairs this with an attention-backward truncation, so the head part alone is not isolated.

**S3. Adaptive softmax was tried on the speedrun and lost to the sampled softmax, by the author's own first look.**
- Who: ldmberman, modded-nanogpt PR #372, opened 2026-09-28, closed 2026-10-01, not merged.
- Link: https://github.com/KellerJordan/modded-nanogpt/pull/372
- Primary checked: yes, PR body and closing comment (raw C_q02).
- State: UNVERIFIED as a comparison. The author's closing comment also says the eval procedure was changed by mistake.
- Verbatim: "for most tokens compute softmax over only the most frequent 18430 tokens + two tail cluster logits". Closing comment: "The new sampled softmax approach beats adaptive softmax at the first glance." And: "Also, I mistakenly changed the evaluation procedure in this patch."

**S4. A copy-pointer mixture on the output distribution claims 11.25% wall time on top of record #92; a training-free document-local copy mixture claims 0.0031 val loss on identical weights.**
- Who: NathanGodey, PR #379 (CPLM), opened 2026-10-02, open. cyrusghane, PR #376, opened 2026-09-29, open. josusanmartin, PR #378 (copy candidate fed as an input), opened 2026-10-02, open.
- Links: https://github.com/KellerJordan/modded-nanogpt/pull/379 ; https://github.com/KellerJordan/modded-nanogpt/pull/376 ; https://github.com/KellerJordan/modded-nanogpt/pull/378
- Primary checked: yes, PR bodies (raw C_q02).
- State: POST CLAIM ONLY for all three.
- Verbatim (#379): "The next-token distribution becomes a mixture of the LM softmax and a pointer over the document's previous tokens". Result: 36.009 ± 0.039 s (8 runs) against record #92 at 40.575 ± 0.024 s (13 runs) on the same node, "−4.566 s (11.25 %)". Added parameters: "196,737 (2 × 128 × 768 + 128 + 1)".
- Verbatim (#376): "if the last few tokens already occurred earlier in the same document, assign some probability on whatever followed them last time. The mixing weights (31 of them, one per match-length × distance bucket) are fitted by maximum likelihood on training tokens". And: "+0.00312 ± 0.00003 over all eight runs on this code".
- Relevance: these act on the output distribution only. Our model has no attention, so it has no built-in copy path; the size of the gain on our model is unknown.

**S5. Adam-family optimisers bias rare-token output probabilities below their data frequency; AMSGrad and SGD do not, in a toy model.**
- Who: Sangsidhya Kar, arXiv 2609.37535, 2026-09-29.
- Link: http://arxiv.org/abs/2609.37535v1
- Primary checked: yes, abstract and HTML (raw C_q05, C_q21).
- State: CONFIRMED that the numbers are in the paper. The model is a toy: width 64, V=4096 (half unused), a Markov-chain source, untied output layer.
- Verbatim (abstract): "if a token is absent for at least two consecutive minibatches, its equilibrium probability is strictly below its data frequency for every learning rate". Numbers (HTML, β₂=0.95, rare tokens): mean log(p̄/f) Adam -2.72, RMSProp -2.65, AMSGrad -0.14, SGD -0.22; test cross-entropy Adam 3.603, AMSGrad 3.500, SGD 3.467. Mitigations listed: "increasing β₂, increasing the batch size, using a running maximum of the second moment as in AMSGrad, sharing the second moment across the vocabulary as in Coupled Adam". On tying: "With tied embeddings, the common shift from Section 3 now affects the loss because it also shifts the input embeddings."
- Related older primary: Coupled Adam, arXiv 2502.08441, 2025-02-12: "we argue that the second moment in Adam is a cause of anisotropic embeddings" (raw C_q20).

**S6. Leviathan: untie the head and replace the input table with a 2.25M-parameter continuous map; claims 2.1% perplexity at 200M and 9.2% at 1.2B against a tied baseline.**
- Who: Reza T. Batley and Sourav Saha, arXiv 2601.22040, v1 2026-01-29, v2 2026-05-07.
- Link: https://arxiv.org/abs/2601.22040
- Primary checked: yes, abstract and HTML v2 (raw C_q07, C_q21).
- State: POST CLAIM ONLY (one paper, not replicated here).
- Verbatim (abstract): "Leviathan's output head remains untied for a parameter increase of as low as 0.2%." And: "Frequency-stratified analysis reveals gains to be concentrated in rare tokens, where continuous parameterization reduces perplexity by 81%, falling to near zero for the most frequent." HTML: the vocabulary is "factorized into k components using a base-b decomposition", with "shared codebooks" giving "z(i)=∑ᵣ₌₁ᵏCᵣ[iᵣ]", then a per-head projection and a B-spline expansion. Vocab 200,376; 4.2B tokens at 200M.
- Relevance: for us this is parameter-neutral. The untied head keeps the 99.3M output table and the input table shrinks to about 2M.

**S7. A factorised (low-rank) forward head hurts more than reducing the backward rank; the gradient-bottleneck harm claim is contested.**
- Who: arXiv 2608.16671, 2026-08-17 (causal test). The claim it tests: Godey and Artzi, arXiv 2603.10145, 2026-03-10.
- Links: http://arxiv.org/abs/2608.16671v1 ; http://arxiv.org/abs/2603.10145v2
- Primary checked: abstracts (raw C_q05).
- State: CONFIRMED that the numbers are in the abstracts. Scale is small (WikiText-2, byte-level and BPE-8192).
- Verbatim (2608.16671): "At half rank in the larger model, the backward-only loss increase is 0.0586 (95% CI [0.0167, 0.1005]), while the factorized forward head increases loss by 0.1795 ([0.1547, 0.2042])." And: "These results confirm strong geometric compression but do not establish that it is a harmful optimization bottleneck."
- Verbatim (2603.10145): "95-99% of the gradient norm is suppressed by the output layer".
- Relevance: an ALBERT-style factorised head to save parameters has a measured loss cost in this setting.

**S8. Every small model config checked ties its embeddings, including vocabularies of 200k to 262k; the HF playbook's ablation favoured depth over untying.**
- Who: HF Hub config files read 2026-10-03; Smol Training Playbook (undated page, cited by a note dated 2025-11-29).
- Links: raw C_q10; https://huggingfacetb-smol-training-playbook.hf.space/
- Primary checked: yes.
- State: CONFIRMED.
- Config facts: SmolLM3-3B tied, vocab 128,256; Qwen3-0.6B tied, 151,936, hidden 1024; Qwen3.5-0.8B tied, 248,320, hidden 1024; gemma-4-E2B tied, 262,144, hidden 1536; Phi-4-mini tied, 200,064; granite-4.0-350m tied, 100,352; Baguettotron tied, 65,536, hidden 576.
- Verbatim (playbook): "our baseline 1.2B model with tied embeddings achieves comparable performance to the 1.46B untied equivalent on all the benchmarks except for WinoGrande, despite having 18% fewer parameters." And: "increasing model depth provides greater benefits than untying embeddings at equivalent parameter budgets". The search summary said the tied model "beat" the untied one; the primary says "comparable".
- Exception: nanochat unties, with very different init and learning rates for the two tables (S9).

**S9. nanochat keeps separate tables with a 50x learning-rate gap and a 800x init gap, plus a logit softcap of 15.**
- Who: karpathy/nanochat, `nanochat/gpt.py` at master; last edit to the file 2026-07-03 (outside the window).
- Link: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
- Primary checked: yes (raw C_q11b).
- State: CONFIRMED.
- Verbatim: `torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)`, `torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)`, `def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, ...)`, `softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]`.
- Relevance: a tied table cannot take both settings. S5 and the paper in S10 give reasons why the input and output roles pull the shared table differently.

**S10. Tied tables are shaped mainly for output prediction; scaling the input-side gradient shifted the geometry but did not improve perplexity.**
- Who: Lopardo, Harish, Arnett, Gupta, arXiv 2603.26663, 2026-03-27.
- Link: http://arxiv.org/abs/2603.26663v1
- Primary checked: yes, abstract and HTML (raw C_q07, C_q21).
- State: CONFIRMED.
- Verbatim (abstract): "This unembedding bias arises because output gradients dominate early in training." HTML: with 5x input-gradient scaling at step 10,000, WikiText-2 perplexity was 35.71 tied against 36.64 with scaling, "no consistent performance gains".

**S11. Output embedding centering is offered as a replacement for z-loss, on par with soft-capping, with or without tying.**
- Who: arXiv 2601.02031, 2026-01-05.
- Link: http://arxiv.org/abs/2601.02031v3
- Primary checked: abstract (raw C_q07, C_q15).
- State: POST CLAIM ONLY for the comparison (abstract gives no numbers).
- Verbatim: "both variants outperform z-loss in terms of training stability, while being on par with logit soft-capping. This holds true both in the presence and the absence of weight tying."

**S12. Gemma 3 270M is 170M embedding plus 100M blocks, with INT4 QAT checkpoints; Google gives no detail on how the table itself is quantised.**
- Who: Google Developers Blog, 2025-08-14 (outside the window, strong primary).
- Link: https://developers.googleblog.com/en/introducing-gemma-3-270m/
- Primary checked: yes (raw C_q09b).
- State: CONFIRMED for the quoted sentences.
- Verbatim: "a total of 270 million parameters: 170 million embedding parameters due to a large vocabulary size and 100 million for our transformer blocks." And: "Quantization-Aware Trained (QAT) checkpoints are available, enabling you to run the models at INT4 precision with minimal performance degradation".

**S13. Lower-ranked signals, kept for the record.**
- LM head bit allocation: VBQ, arXiv 2607.02893 (2026-07-03), abstract: "the LM head averages 1.09 bits, while the first MLP block keeps ~2.5 bits". POST CLAIM ONLY. For a tied table the head and the input lookup share bits, so this does not transfer directly.
- LM head compression after training: ARCHead, arXiv 2608.02703 (2026-08-03), abstract: "reduces persistent LM-head storage by 3.7-3.9x" and "uses 25.6% of BF16 head storage while attaining 1.007 relative perplexity" on Qwen3-8B-Base. POST CLAIM ONLY.
- Vocabulary size and serving regime: arXiv 2608.11361 (2026-08-11), abstract: "on-device deployments ($B=1$) should use $V \approx 32$k". POST CLAIM ONLY. We keep the DeepSeek V4 tokenizer, so this is context only.
- fp32 master copy for a bf16 embedding: modded-nanogpt PR #377 (2026-10-01, open): "By step 2000 its entries reach `|w| ~ 9`, where most tail Adam updates are below half a bf16 ulp and round away." POST CLAIM ONLY. It matters to us only if the table is stored in bf16 rather than under fp32 autocast.
- Sampled softmax outside the speedrun: q03 and q04 found no 2026 paper using sampled softmax for LM pretraining. The web summary of Chen et al. 2016 (arXiv 1512.04906) says target sampling "makes less progress in terms of perplexity reduction per iteration". Not opened as a primary here, so UNVERIFIED.

## 3. Proposed changes (at most four)

Each test is one arm against the current arm at matched tokens, scored on the TEST windows with the summary table tool, two seeds if the first difference is under the seed noise.

**P1. Shared-candidate sampled softmax for the first 90% of tokens, with row-sparse updates of the tied table.**
- Source: S1 (record #92 mechanism, no logQ term, uniform stride negatives plus all batch targets, full softmax at the end). S2 for a later refinement.
- Why it fits us: the head is 86.7% of our forward FLOPs, and our vocabulary is 2.6x the speedrun's. With a sampled output side, the tied table's gradient touches only the candidate rows plus the input rows, so a row-sparse optimiser on the table becomes possible. Without sampling the output side gives every row a gradient (q16 notes this as the search tool's own reasoning, not a documented fact).
- Cost to watch: record #92 paid about 2.4 millinats of val loss for 8.8 s of wall time (S1).
- Test: add `--sampled-p 16384 --sampled-until 0.9` (new flag: candidate set = unique batch targets plus `(k*S) mod 129280` stride negatives, S coprime to 129,280, then full CE) and compare TEST bpb and tokens per second against the current arm at equal tokens.

**P2. Fix the rare-token second-moment bias on the tied table: raise β₂ for the table group, or use AMSGrad on that group only.**
- Source: S5 (2609.37535, toy model; mitigations quoted from the paper), with Coupled Adam (2502.08441) as the heavier option.
- Why it fits us: 129,280 rows means most rows appear in few batches of 32 x 256 tokens, which is the regime the paper describes. The paper's evidence is a toy model, so the effect size on our model is unknown.
- Test: run the table group with `betas=(0.9, 0.999)` and, as a second arm, `amsgrad=True` on the table group only; report TEST bpb overall and on the rare-token bucket (tokens with under one expected occurrence per batch).

**P3. A training-free document-local copy mixture on the output distribution.**
- Source: S4 (PR #376 for the fitted mixture, PR #379 for a trained pointer). POST CLAIM ONLY on the speedrun.
- Why it fits us: our model has no attention and so no copy path, which is exactly what these mixtures add. #376 needs no retraining: 31 mixing weights, one per match-length x distance bucket, fitted by maximum likelihood on training tokens.
- Test: on an existing checkpoint, find for each position the token that followed the longest earlier exact match in the same window, fit the 31 weights on train windows, and report TEST bpb with and without the mixture. No training run is needed.

**P4. Leviathan-style untie: keep the 129,280 x d output table, replace the input table with a base-b factorised codebook map.**
- Source: S6 (2601.22040, POST CLAIM ONLY, 2.1% perplexity at 200M). S10 explains why a tied table serves the output role first.
- Why it fits us: the change is parameter-neutral for us (the output table stays, the input table falls to about 2M), and the int8 browser size stays about the same. It also frees the head table from the input role, so P2 can tune it for output alone. Risk: our features use 8 back-token embeddings and 5 moving averages, so the input table carries more of our model than it does in a Transformer.
- Test: an arm where `features()` reads token vectors from `sum_r C_r[digit_r(i)]` (k=3, b=51, since 51^3 = 132,651 covers 129,280) followed by a small MLP to d, with `forward()` using a separate untied `nn.Embedding(129280, d)` as the head; compare TEST bpb against the tied arm at equal tokens.

Not proposed, with reasons: a factorised low-rank head (S7 measured it as the larger loss cost); adaptive softmax (S3 lost to sampled softmax on its author's own first look); scaling the input-side gradient (S10 found no perplexity gain). A logit cap or output centering (S9, S11) is cheap and could ride along with any arm above, but logit soft-capping in modded-nanogpt is already known.
