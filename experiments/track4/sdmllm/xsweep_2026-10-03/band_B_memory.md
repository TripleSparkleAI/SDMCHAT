# Band B - memory layers and attention-free language models

- Sweep date: 2026-10-03 (UTC 10:46 to 10:57)
- Window: 2026-08-01 to 2026-10-03, widened to 2026-06-01 where thin. Items outside the window are marked OUTSIDE.
- Route: the X API returned HTTP 402 "credits depleted", so this is the web-mirror fallback. Tools: the arXiv Atom API (python3 urllib, sortBy=submittedDate), WebSearch, WebFetch, the GitHub REST API (unauthenticated) and the Hugging Face Hub API.
- Raw records: `raw/B_q01_*.md` to `raw/B_q39_*.md`, one per query, written before judging.
- Primary checks: every claim marked CONFIRMED below was read at its primary source (arXiv abstract page, arXiv HTML, GitHub page or repository file) through WebFetch or pdftotext. WebFetch returns a model-extracted reading of the page. Numbers are copied from that reading.
- States: CONFIRMED (number found at the primary) · POST CLAIM ONLY (the only source is a post, issue or README by an interested party, not independently checked) · UNVERIFIED (seen only in a search summary).

## 1 · The query matrix

"Returned" is the raw count. "In window" counts items dated 2026-06-01 or later. "Relevant" counts in-window items about LM memory layers, lookup memory or attention-free LMs, after reading titles and abstracts.

| id | query (abridged) | tool | returned | in window | relevant | note |
|---|---|---|---|---|---|---|
| q01 | abs:"memory layer" AND abs:"language model" | arXiv API | 37 | 7 | 0 | all agent-memory systems |
| q02 | abs:"product key" | arXiv API | 20 | 1 | 0 | a database paper |
| q03 | UltraMem / "lookup experts" / PEER | arXiv API | 40 | 40 | 0 | query defect: "PEER" matched "peer review" |
| q04 | Engram AND memory AND language | arXiv API | 26 | 9 | 6 | FactorEngram, Frozen Memory, Tokenizer-Agnostic, TF-Engram, Tensorizing, Engram Adapter |
| q05 | abs:"conditional memory" | arXiv API | 40 | 23 | 4 | MoME, Lngram v2, Memory Is Not Always Needed, CHARM |
| q06 | kNN-LM / nearest neighbor LM | arXiv API | 18 | 1 | 0 | one episodic-memory psycholinguistics paper, not opened |
| q07 | abs:"sparse distributed memory" | arXiv API | 16 | 1 | 1 | rank-order SDM codes, not an LM |
| q08 | abs:"attention-free" | arXiv API | 40 | 15 | 2 | Kathleen Writes, Kathleen Remembers |
| q09 | RWKV / Mamba-3 / xLSTM / DeltaNet (titles) | arXiv API | 40 | 12 | 0 | applications and kernels, no small-LM recipe |
| q10 | n-gram AND embedding AND hash/lookup AND LM | arXiv API | 6 | 4 | 4 | duplicates of q04 plus MolGram |
| q11 | memory layer(s) AND optimizer/lr/sparse | arXiv API | 40 | 30 | 2 | Kathleen Remembers (dup), Memory Layer for recommendation |
| q12 | "embedding layer" AND scaling AND n-gram | arXiv API | 1 | 0 | 0 | EMPTY in window |
| q13 | abs:"memory layers" | arXiv API | 40 | 30 | 2 | same set as q11 |
| q14 | product-key / product keys | arXiv API | 20 | 1 | 0 | |
| q15 | UltraMem | arXiv API | 2 | 0 | 0 | EMPTY in window |
| q16 | lookup experts / lookup table AND LM | arXiv API | 40 | 8 | 0 | LUT quantization papers |
| q17 | "key-value memory" AND language model | arXiv API | 22 | 1 | 1 | LoKiFormer |
| q18 | without attention / MLP-only AND LM | arXiv API | 40 | 40 | 0 | query defect: OR precedence matched generic papers |
| q19 | n-gram embedding / over-tokenized / hashed n-gram | arXiv API | 40 | 28 | 2 | Qwen3.8-Next, Tensorizing (dup) |
| q20 | "parametric memory" AND sparse AND pretraining | arXiv API | 1 | 0 | 0 | EMPTY |
| q21 | "memory layers" product key LM 2026 | WebSearch | 10 | 0 | 0 | FwPKM is January 2026, OUTSIDE |
| q22 | UltraMemV2 / UltraMem follow-up 2026 | WebSearch | 10 | 0 | 0 | EMPTY in window; UltraMemV2 at ICLR 2026 |
| q23 | Engram port nanochat / modded-nanogpt | WebSearch | 10 | 2 | 3 | nanochat LOG (OUTSIDE), MoME, DeepSeek-V4.1-Flash |
| q24 | Mixture of Lookup Experts follow-up 2026 | WebSearch | 9+ | 2 | 1 | Budgeting Bytes (2026-07-29) |
| q25 | Kathleen attention-free byte LM | WebSearch | 10 | 2 | 2 | duplicates of q08 |
| q26 | RWKV-8 / Mamba-3 small LM 2026 | WebSearch | 10+ | 0 | 0 | EMPTY in window |
| q27 | memory value table optimizer / lr tips | WebSearch | 9 | 0 | 0 | all older; used for training-trick context |
| q28 | Large Lookup Layers L3 | WebSearch | 10 | 0 | 0 | L3 is January 2026, OUTSIDE |
| q29 | GitHub repos: engram memory LM, pushed >2026-06-01 | GitHub API | 1 | 1 | 0 | a LoRA-memory repo |
| q30 | GitHub deepseek-ai/Engram issues | GitHub API, then WebFetch | ERROR 403 (rate limit), fallback 16 | 2 | 0 | issues #9 and #20 opened (OUTSIDE) |
| q31 | GitHub repos: product key memory, pushed >2026-06-01 | GitHub API | 13 | 13 | 2 | productkey-init-sdm, lucidrains FwPKM |
| q32 | HF models search "engram" | HF Hub API | 40 | about 30 | 0 | mostly DeepSeek-V4.1-Flash quantizations and LoRA adapters |
| q33 | HF models search "memory-layer" | HF Hub API | 3 | 0 | 0 | EMPTY in window |
| q34 | PEER follow-up 2026 | WebSearch | 10+ | 0 | 0 | EMPTY in window |
| q35 | kNN-LM 2026 | WebSearch | 10+ | 0 | 0 | EMPTY in window |
| q36 | sparse distributed memory LM 2026 | WebSearch | 10+ | 1 | 0 | VaCoAl is a hardware proposal |
| q37 | MLP-only / no attention, FineWeb-Edu, bpb | WebSearch | 10+ | 1 | 0 | Fast Weight Attention is an attention model |
| q38 | per-layer embeddings, browser, on-device | WebSearch | 10+ | several | 1 | gemma4-webgpu-engine buffer limits |
| q39 | site:x.com memory layer / engram / PKM | WebSearch | 10 | 2 | 0 | X not fetched; engagement unknown |

Totals: 39 queries; 10 returned zero relevant in-window items. Two arXiv queries (q03, q18) were defective because of operator precedence and are recorded as such.

## 2 · Ranked signals

### S1 · Qwen3.8-Flash-Next ships hashed n-gram embedding tables off the accelerator; loss falls but downstream does not follow
- Who: Zihan Qiu, Zekun Wang, ... Dayiheng Liu (Qwen team). arXiv 2608.30320, submitted 2026-08-31.
- Link: https://arxiv.org/abs/2608.30320 · https://arxiv.org/html/2608.30320
- Primary checked: yes (abstract and HTML). State: **CONFIRMED**.
- Verbatim (abstract): "a sparse mixture-of-experts model with 125B parameters, 6B activated per token, and additional 51B parameters of n-gram embedding tables held off the accelerator."
- Verbatim (HTML, as extracted): "We place it at Layer 2, allowing host-memory prefetching to overlap with the computation of the first layer." · "The embeddings are further injected into residual stream through contextual gating mechanism." · "The nn-gram embedding table runs on Adam with weight decay disabled."
- Numbers (HTML): on a 25B-A3B model, loss 1.585 without n-gram embeddings and 1.541 with 50x vocabulary scaling; the paper says "downstream performance does not follow the same trend".
- Relevance: a second frontier lab ships a deterministic, token-addressed table beside learned compute, and reports that the loss gain overstates the downstream gain.

### S2 · MoME: a context-gated mixture of slots per token beats a hashed bigram table, which in turn beats a canonical Engram port, at nanochat scale
- Who: Muchen Li, Leonid Sigal, Renjie Liao (UBC, Vector Institute). arXiv 2609.15126, submitted 2026-09-14.
- Link: https://arxiv.org/abs/2609.15126 · https://arxiv.org/html/2609.15126v1
- Primary checked: yes (HTML). State: **CONFIRMED**.
- Verbatim (abstract): "replaces each token's single memory row with a mixture of M slots and uses a learned gate over the hidden state to choose which slots to read at each position."
- Numbers (Table 2, nanochat, iso-parameter, 151M memory): Base 0.8785 val bpb, 0.1447 CORE · Bigram 0.8636, 0.1533 · MoME 0.8611 ± 0.0003, 0.1583 ± 0.0020. Headline config M = 6 slots, K = 2 read.
- Verbatim (Appendix A.6): "At approximately matched total parameters and 6×10^19 training FLOPs on d24 ClimbMix, Bigram achieves lower validation bpb and higher CORE-22 than our canonical Engram adaptation".
- Verbatim (Table 11): "Value-table optimizer: AdamW group, lr 0.2, wd 0.001."
- Relevance: at small scale the cheapest table (a hashed bigram added to the residual) recovers most of the gain, and the memory table is trained at an embedding-table learning rate.

### S3 · DeepSeek-V4.1-Flash ships Engram at 196B parameters and drops the short causal convolution
- Who: DeepSeek-AI. arXiv 2609.19969, submitted 2026-09-17.
- Link: https://arxiv.org/html/2609.19969
- Primary checked: yes (HTML). State: **CONFIRMED** for the design; no ablation number exists in the paper per the extraction.
- Verbatim: "The modules are placed at layers 1 and 14 (zero-indexed) to balance memory usage across training pipeline stages." · "196B Engram parameters evenly across two modules. Each module uses N-gram orders {2,3,4}, with 8 hash heads and a total embedding dimension of 2048 per order. Each head indexes a table of approximately 16M entries." · "First, we omit the short causal convolution because its performance gains do not justify the added complexity." · "optimize the Engram embedding tables, token embedding, and prediction head using a momentum-based update followed by Sinkhorn balancing."
- Hedge: the extraction reports that the paper does not isolate Engram's contribution with a number.

### S4 · FactorEngram: shared-basis n-gram memory with per-basis gating; plain Engram gives no WikiText gain at 1B
- Who: Bowen Yang, Jingbo Zhou, Qinghong Miao, Hua Wu (Baidu, NTU). arXiv 2609.35578, submitted 2026-09-28.
- Link: https://arxiv.org/html/2609.35578v1
- Primary checked: yes (HTML). State: **CONFIRMED**.
- Numbers, 340M / 30B tokens, WikiText PPL: Transformer 23.19 · Engram 22.29 · FactorEngram 21.29. NIAH-3: 22.1% · 15.0% · 58.3%.
- Numbers, 1B / 120B tokens, WikiText PPL: Transformer 15.57 · Engram 15.60 · FactorEngram 15.57.
- Ablation (340M): replacing basis-level gating with scalar gating moved NIAH-3 from 58.3% to 9.4%.
- Verbatim (abstract): "identify insertion before the attention sublayer in the middle layers as an effective configuration."
- Relevance: the n-gram memory gain on perplexity shrank to zero between 340M and 1B in this paper; the long-context recall gain did not.

### S5 · Frozen Memory Is Not Enough: a parameter-matched FFN reader gets better perplexity but worse QA than the memory
- Who: Mingyuan Li, Guangsheng Yu, Xu Wang, Shaoxiong Ji. arXiv 2608.17050 (v1 2026-08-17; v3 dated 2026-09-30 per the extraction).
- Link: https://arxiv.org/html/2608.17050v3
- Primary checked: yes (HTML, Table 5). State: **CONFIRMED**.
- Numbers (Table 5, average QA): full transfer 38.5 · permuted keys 32.5 · random memory 34.6 · no-gate reader 33.7 · parameter-matched FFN control 34.5 "despite superior intrinsic perplexity (7.4 vs 8.7)".
- Verbatim (abstract): "Ablations show that learned memory content and correct addressing both matter, but the transferred table becomes useful only through a reader aligned to the target model."
- Relevance: the same split as our own record (a no-store readout beats the store on bits) appears here, and here the store wins on a recall-style task. Perplexity alone would have picked the wrong arm.

### S6 · Kathleen Writes: an attention-free byte LM with content-gated multi-scale decay beats a parameter-matched transformer at about 0.5M parameters
- Who: George Fountzoulas (single author). arXiv 2608.04678, submitted 2026-08-05.
- Link: https://arxiv.org/html/2608.04678
- Primary checked: yes (HTML). State: **CONFIRMED** for the numbers as reported. Not peer reviewed; very small scale; the transformer baseline at 0.5M parameters is weak by construction.
- Verbatim: "Multi-scale reverb - three banks with content-dependent decay γ∈[0.50,0.90] / [0.90,0.99] / [0.95,0.9995]; recurrence st=γt st−1+(1−γt)vt" (one dash in the source is written here as '-').
- Numbers (WikiText-103 bytes, bits/byte, reverb vs transformer): 2 MB 2.844 vs 3.586 · 8 MB 2.239 vs 3.081 · 32 MB 2.010 vs 2.416 · 128 MB 1.904 vs 2.155 · 512 MB 1.842 vs 2.040. The gap narrows from 0.84 to 0.20 as data grows.
- Verbatim: "a computed (gradient-free) lexicon reaches 94% of a learned embedding table's top-1 accuracy at one fifth of the parameters, but not its perplexity".
- Relevance: the closest published cousin to our context features (causal decaying averages of embeddings). Their decays are content-dependent and reach 0.9995 per byte; ours are fixed and stop at 0.99 per token.

### S7 · Kathleen Remembers: a 25K-parameter holographic notebook adds exact recall to an attention-free trunk
- Who: George Fountzoulas. arXiv 2608.30376, submitted 2026-08-31. Comments field: "Mechanism covered by U.S. Provisional Patent Application No. 64/140,260".
- Link: https://arxiv.org/abs/2608.30376
- Primary checked: abstract only. State: **CONFIRMED** at abstract level.
- Verbatim: "on WikiText-2 bytes the notebook improves prediction of repeated rare words by +0.15-0.27 bits/byte, the gain growing with the distance between mentions and holding zero-shot at 4x training length".
- Relevance: an attention-free model gains on repeated rare words from a fixed-key store. It also supplies a metric (bits on repeated rare words) that isolates the recall our model lacks.

### S8 · Tensorizing Engram: shared CP factors match Engram with fewer table parameters
- Who: Zhou et al. arXiv 2606.08347, submitted 2026-06-06 (widened window).
- Link: https://arxiv.org/html/2606.08347v1
- Primary checked: yes (HTML). State: **CONFIRMED**.
- Numbers (9 layers, vocab 1024, N = 5): GPT 1.251 bpb · Engram 1.209 · TN-gram 1.208; extra parameters 26M (Engram) vs 19M (TN-gram).

### S9 · Lngram v2: a zero-value sink and counterfactual surrogate gradients for a hard discrete address
- Who: Yunao Zheng, Bin Wen, ... Han Li. arXiv 2609.03426 (v1 2026-09-03, v2 2026-09-22).
- Link: https://arxiv.org/abs/2609.03426v2
- Primary checked: abstract only; the mechanism was not read. State: **CONFIRMED** at abstract level.
- Verbatim: "A zero-value Sink and counterfactual surrogate gradients further improve readout selectivity and routing trainability while preserving hard discrete addressing."
- Relevance: our address gradient into x is stopped. This paper names a way to train a hard address without that cut. Experiments are on vision-language models.

### S10 · LoKiFormer: a 64-slot softmax key-value memory speeds pretraining
- Who: Chen et al. arXiv 2608.12419, submitted 2026-08-12.
- Link: https://arxiv.org/html/2608.12419v1
- Primary checked: yes (HTML). State: **CONFIRMED**.
- Numbers: at 10k steps, eval PPL 31.82 baseline vs 29.08 with KMM; abstract claims "converges 1.33x faster". KMM is a full softmax over 64 slots, not a sparse product-key read.

### S11 · Engram repository issues report a 5x learning rate with zero weight decay, and lower loss with worse evaluation at 1.3B dense
- Who: issue #9 by goodluckcwl (2026-01-16, OUTSIDE); issue #20 by fate08301017 (2026-03-05, OUTSIDE).
- Links: https://github.com/deepseek-ai/Engram/issues/9 · https://github.com/deepseek-ai/Engram/issues/20
- State: **POST CLAIM ONLY** (unanswered issues; the reader did not cite code lines).
- Verbatim (#9): "the Engram module appears to be trained with a 5× higher learning rate than the backbone and weight_decay=0".
- Verbatim (#20, as extracted): "loss curve superior to baseline 1.3b model without engram, but evaluation on piqa, siqa, ARC_e, hellaswag datasets showed worse performance than baseline."

### S12 · nanochat Engram-lite: zero-init hashed bigram table, same learning rate as token embeddings, gating did not help
- Who: Andrej Karpathy, nanochat `dev/LOG.md`, entry "2026-01-27: Bigram Hash Embeddings (Engram-lite)" (OUTSIDE).
- Link: https://github.com/karpathy/nanochat/blob/master/dev/LOG.md
- Primary checked: yes (the LOG). State: **CONFIRMED** for the LOG text.
- Verbatim: "zero-init embedding table, learned per-layer lambdas" · "AdamW with same LR as token embeddings" · gating "added parameters and complexity without improvement" · trigrams "No improvement over bigrams alone. Dilutes capacity from more frequent 2-gram patterns" · table size "vocab_size * 5 (~164K entries for 32K vocab)".
- UNVERIFIED: the search summary says Discussion #481 records that the bigram table was not adopted in the speedrun. That discussion was not opened.

### S13 · Product-key factorized initial state for a recurrent "sparse delta memory"
- Who: GitHub user p0rc314in, repository created 2026-08-17, pushed 2026-08-31.
- Link: https://github.com/p0rc314in/productkey-init-sdm
- State: **POST CLAIM ONLY** (README, no paper). Note: "SDM" here means a recurrent sparse delta memory, not a value table.
- Numbers (README, 8-layer model of about 15M parameters, WikiText-103): validation NLL zero init 4.54865 · factorized M0[i, j] = A[i] + B[j] 4.47978 with 57,344 parameters · fully learned 4.47779 with 917,504 parameters.

### S14 · Browser buffer limits for big lookup tables
- Who: Ar5en1c/gemma4-webgpu-engine README (date not shown).
- Link: https://github.com/Ar5en1c/gemma4-webgpu-engine
- State: **POST CLAIM ONLY** (README statement about the WebGPU specification).
- Verbatim: "iOS grants exactly 1 GiB and the specification default is 256 MiB." · "The engine splits the table by vocabulary row and gathers once per slice."
- Our arithmetic: one value table of 65,536 rows x 768 dims is 192 MiB in fp32 and 96 MiB in fp16, so each hop's table fits one default buffer. A hashed table of 5 x 129,280 rows at d 768 would be about 947 MiB in fp16 and would need splitting.

### S15 · Budgeting Bytes: an address-determinism taxonomy for storage-bound decoding
- Who: Hanhaodi Zhang. arXiv 2609.04238, submitted 2026-07-29 (widened window).
- Link: https://arxiv.org/abs/2609.04238
- Primary checked: abstract. State: **CONFIRMED** at abstract level.
- Verbatim: "classifies parameters by when their fetch address becomes known during a token's forward pass (A0: at token sampling; A1: before attention; A2: layerwise data-dependent; A3: always read)." · "Predictability is not speedup".
- Relevance: our product-key reads are class A2; a token-hashed table is class A0 and can be prefetched in a browser before the forward pass starts.

### Lower-ranked, abstract level only
- Memory Is Not Always Needed (arXiv 2608.23982, 2026-08-25): "memory effects vary substantially across inputs, tasks, and injection locations." CONFIRMED (abstract).
- Tokenizer-Agnostic Engram Module (arXiv 2607.29065, 2026-07-31): polynomial byte hashing, "comparable performance". CONFIRMED (abstract).
- TF-Engram (arXiv 2607.07388, 2026-07-08): train-free phrase memory, Qwen3-0.6B downstream "57.6 to 59.4". CONFIRMED (abstract).
- Engram Adapter (arXiv 2608.29327, 2026-08-29): post-hoc Engram on frozen Qwen3, "preserving 99.4%--100.1% of average OOD performance". CONFIRMED (abstract).
- CHARM / CoSE (arXiv 2609.33270, 2026-09-27): compositional sparse task embedding "reduces learned task-memory parameters by over 90%". CONFIRMED (abstract); ARC, not LM.
- UltraMemV2 (arXiv 2508.18756): published at ICLR 2026; no in-window follow-up found. UNVERIFIED beyond the search summary.
- RWKV-8, Mamba-3, xLSTM, PEER, kNN-LM: no in-window primary with small-LM numbers found (q09, q26, q34, q35 EMPTY for this band).

## 3 · Proposed changes (at most four)

### P1 · Test a deterministic token-hash address against our learned address
- Source: S1 (Qwen3.8-Next), S2 (MoME Table 2: Bigram 0.8636 vs base 0.8785 bpb), S3 (DeepSeek-V4.1-Flash), S12 (nanochat zero-init hashed bigram).
- Why: every in-window lab result adds a table addressed by the surface tokens, not by a learned query. Our record says the address decides the outcome. This arm measures a known-good fixed address on our own model.
- Test, two arms on `track4_sdmonly_train.py` against `p0_A_c` at matched steps, paired on TEST windows: (a) add a zero-init hashed bigram table of 65,536 rows, index `(36313*cur XOR 27191*prev) mod 65536`, added to x before hop 1, trained like the value tables; (b) keep 4 hops but replace hop 1's product-key address with that bigram hash, values unchanged, so only the address differs.

### P2 · Extend the store learning-rate sweep into the embedding-table range
- Source: S2 (MoME memory tables at "AdamW group, lr 0.2, wd 0.001"), S12 (nanochat: "same LR as token embeddings"), S1 (Qwen: Adam, weight decay disabled), Lample 2019 (values at 1e-3 vs 2.5e-4, read from the PDF).
- Why: our read divides by k x heads = 32, so one Adam step on a value row moves the read 32 times less than the same step on an embedding row. This is our inference from the read formula, not a published result. The plan sweeps the store multiplier only to x10 (3e-2 absolute).
- Test: run `p0_A_c` with `--store-lr-mult 30` and `--store-lr-mult 60` (0.09 and 0.18 absolute at lr 3e-3), store wd 0, same seed, paired on TEST windows against the x3 arm.

### P3 · Add a slow context average, then a content-gated one
- Source: S6 (Kathleen Writes, decay banks reaching 0.9995 per byte, content-dependent).
- Why: our slowest average is 0.99 per token (about 100 tokens). Their slowest band reaches about 2,000 bytes. At about 4 bytes per token (our assumption, not measured here) that is about 0.998 per token.
- Test: arm (a) `decays=(0.5, 0.8, 0.9, 0.97, 0.99, 0.998)` against `p0_A_c`; arm (b), only if (a) helps, a decay per average computed from the current embedding by a sigmoid mapped into each band's range.

### P4 · Score store arms on recall positions, not only on all-position bits
- Source: S5 (FFN control better perplexity 7.4 vs 8.7, worse QA 34.5 vs 38.5), S1 (Qwen: downstream "does not follow" loss), S7 (Kathleen: repeated rare words), S11 (Engram issue #20, post claim).
- Why: our finding that a no-store readout beats the store is measured on bits over all positions. Two in-window papers report that the bits ranking and the recall ranking of a memory can disagree.
- Test: rerun `track4_sdmllm_paired_window_comparison.py` on existing checkpoints of the store arm and the matched dense control (`body=dense`), reporting bits separately on positions whose target token already appeared earlier in the window, and on targets below a corpus-frequency threshold.
