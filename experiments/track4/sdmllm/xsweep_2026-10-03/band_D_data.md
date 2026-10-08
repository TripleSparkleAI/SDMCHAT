# Band D - data: field sweep, 2026-10-03

Sweep run 2026-10-03 10:47-10:58 UTC. Window 2026-08-01 to 2026-10-03, widened to 2026-06-01. The X API returned HTTP 402 (credits depleted), so every X post below was surfaced by a search engine and not fetched. Engagement for those posts is unknown. Raw logs for every query are in `raw/D_q*.md`, written before any judgement.

Our model, for reference: an attention-free LM of about 16-60M non-embedding parameters with a tied 129,280-token head, pretrained on FineWeb-Edu sample/10BT (600M tokens so far, target 3-10B). TEST is bpb on held-out FineWeb-Edu. A chat fine-tune (SmolTalk-style mix, half web rows) and a philosophy character fine-tune follow.

## 1. Search matrix

"Hits" counts the results the tool returned. "In window" counts results dated on or after 2026-06-01. "Relevant" counts in-window results that bear on this band after reading titles and abstracts.

| file | row | query (exact text in the raw file) | tool | hits | in window | relevant |
|---|---|---|---|---|---|---|
| D_q01 | small-scale ablations | small-scale pretraining data ablation 2026 ... | WebSearch | 10 | 3 | 2 (2608.11859, 2607.08646) |
| D_q02 | small-scale ablations | arXiv abs: "pretraining data" AND small-scale/proxy AND ablation/mixture | arXiv API | 7 | 0 | 0 (EMPTY in window) |
| D_q02b | primary | 13 candidate abstracts by id_list | arXiv API | 13 | 13 | 9 |
| D_q03 | FineWeb versions | FineWeb-Edu v2 OR "FineWeb 2" new release 2026 | WebSearch | 18 | 0 | 0 (no FineWeb-Edu v2) |
| D_q04 | FinePDFs | finepdfs-edu pretraining results small model 2026 | WebSearch | 9 | 2 | 2 (2609.37891, 2608.11859) |
| D_q05 | DCLM successors | DCLM successor dataset 2026 ... | WebSearch | 16 | 0 | 0 (no named successor) |
| D_q06 | synthetic | Cosmopedia v3 OR "synthetic textbooks" ... | WebSearch | 20 | 2 | 1 (2607.28109) |
| D_q07 | synthetic | BeyondWeb rephrased web ... | WebSearch | 9 | 0 | 0 |
| D_q08 | synthetic | Nemotron synthetic pretraining dataset 2026 ... | WebSearch | 18 | 1 | 0 (none aimed below 1B) |
| D_q09 | curricula | arXiv abs: TinyStories OR curriculum AND language model AND pretrain* | arXiv API | 40 | 22 | 4 (2608.13545, 2607.27490, 2609.30535, 2607.10745) |
| D_q10 | chat SFT data | smoltalk2 mix; Gemma 3 270M instruct data; LFM2 350M SFT data | WebSearch | 35 | 2 | 0 |
| D_q10b | primary | HF API metadata for 13 dataset ids | HF API | 11 ok, 2 x 401 | - | - |
| D_q11 | chat SFT data | Qwen3-0.6B / Qwen3.5-0.8B post-training data ... | WebSearch | 30 | 4 | 0 (method only, data unreleased) |
| D_q12 | chat SFT data | chat fine-tuning tiny language model under 200M parameters SFT results 2026 | WebSearch | 29 | 0 | 0 (EMPTY in window) |
| D_q13 | chat SFT data | Tulu 4 OR "Magpie" OR "OpenHermes 3" ... | WebSearch | 48 | 0 | 0 (no Tulu 4 or OpenHermes 3) |
| D_q14 | dedup / quality | deduplication quality classifier small scale 2026 fastText vs embedding | WebSearch | 30 | 0 | 0 |
| D_q15 | books | Common Pile v0.1 OR "Institutional Books" OR "Common Corpus" ... | WebSearch | 37 | 0 | 0 new versions |
| D_q16 | books / philosophy | HF search: philosophy, stanford encyclopedia, gutenberg, public domain books, pg19 | HF API | 97 | 54 | 0 of note (community uploads); "public domain books" EMPTY |
| D_q16c | primary | dataset README licence lines for 7 ids | HF raw | 6 ok, 1 x 401 | - | - |
| D_q17 | community | GitHub search: nanochat dataset / pretraining data / SFT; modded-nanogpt data (since 2026-06-01) | GitHub API | 57 | 57 | 1 (nanochat issue #860) |
| D_q17b | primary | nanochat dev/LOG.md | GitHub raw | 1 file | 0 entries | context only (newest entry 2026-05-05) |
| D_q17c | primary | nanochat issue #860 and comments | GitHub API | 3 posts | 3 | 1 |
| D_q18 | community | x.com nanochat data mixture ablation result tweet 2026 FineWeb-Edu | WebSearch | 20 | 2 | 1 (2606.07597); X posts dated 2026-03 |
| D_q19 | HF recent | HF search: smoltalk, pretraining, synthetic textbook, fineweb, sft small, tiny | HF API | 158 | 125 | 0 of note (community uploads) |
| D_q20 | synthetic | arXiv abs: synthetic AND pretraining AND rephras*/textbook/rewrit* | arXiv API | 40 | 40 | 2 (2609.40295, 2609.30063) |
| D_q22 | synthetic | Pleias Monad 56M SYNTH tiny model results reproduction | WebSearch | 19 | 1 | 1 (2609.37891); no reproduction found |
| D_q22b | primary | PleIAs/Monad config.json and README | HF raw | 1 | - | - |
| D_q23 | curricula | TinyStories successor 2026 ... | WebSearch | 10 | 0 | 0 |
| D_q24 | small-scale results | OpenAI parameter golf FineWeb bpb 16MB ... | WebSearch | 18 | 1 | 1 (2607.01517) |
| D_q25 | bpb / tokenizer | "bits per byte" FineWeb-Edu small model data ablation 2026 tokenizer vocabulary size | WebSearch | 9 | 2 | 1 (2608.18062) |
| D_q26 | synthetic / web | WildAI Pangram AI-generated web text pretraining scaling law FineWeb filter | WebSearch + WebFetch | 10 | 10 | 1 (2609.40295) |
| D_q27 | primary | arXiv html 2609.37891v1 | WebFetch | 1 | 1 | 1 |
| D_q28 | HF recent | newest datasets by author HuggingFaceTB / HuggingFaceFW / PleIAs | HF API | 45 | 7 | 0 new versions of known sets |
| D_q29 | primary | FinePhrase card and arXiv 2604.13977 | HF raw + arXiv API | 2 | 0 | 1 (outside window, not on the known list) |

Empty rows, stated plainly: no FineWeb-Edu v2 (the card's latest changelog entry is v1.4.0, 2025-07-11, and the HF API reports lastModified 2025-07-11), no Cosmopedia v3, no named DCLM successor, no Tulu 4, no OpenHermes 3, no new Magpie release, and no 2026 paper reporting chat SFT results below 200M parameters. The newest HuggingFaceTB dataset is dated 2026-04-08.

## 2. Ranked signals

States: CONFIRMED means the number was read in the primary source (paper abstract, paper body via fetch, dataset card or log). POST CLAIM ONLY means the primary is a single post or issue that was read, but the result has no independent check. UNVERIFIED means only a search-engine summary was seen.

### S1. A fully synthetic corpus trains a 56M model with our width
- **Claim:** Pleias report that "At iso-compute, SYNTH outperforms filtered web data". Their Monad model is 56M parameters, hidden size 256, 64 layers, an 8,192-token vocabulary and tied embeddings. The card reports "close to 30%" on MMLU.
- **Who / date:** Langlais et al. (Pleias), arXiv 2609.37891, published 2026-09-29. Monad card created 2025-11-10.
- **Link:** https://arxiv.org/abs/2609.37891 , https://huggingface.co/PleIAs/Monad
- **Primary checked:** yes (abstract via arXiv API; body via fetch; config.json and README via HF).
- **State:** CONFIRMED for the quoted statements. The comparison is on task accuracy, not on web bpb. In the 600M ablation the two web-trained models needed a separate post-training step ("SmolTalk ... plus the MMLU auxiliary training split ... ~100M tokens over 3 epochs"). The card says 200B training tokens; the paper says about 180B. The paper text read by the fetch tool says d_model 384; config.json says hidden_size 256. No independent reproduction was found.
- **Verbatim:** "We evaluate SYNTH by training a suite of models: a 56M tiny model (Monad), 0.3B-0.6B dense models (Baguettotron), and a 13B / 1B-active MoE. At iso-compute, SYNTH outperforms filtered web data, and our models remain competitive with similarly-sized open-weight baselines." Card: "Monad attains performance on MMLU significantly beyond chance with close to 30% of positive rate."
- **Licence:** SYNTH card `license: cc-by-4.0`, seed text "most of the time CC-BY-SA 4.0". Monad `apache-2.0`.

### S2. FineWeb-style filtering keeps AI-written pages more often, and their share has grown each year
- **Claim:** Pangram labelled 10.1% of FineWeb-filtered tokens AI-generated in June 2024, 16.1% in June 2025 and 27.5% in June 2026. The FineWeb pipeline keeps AI documents 2.3 times as often as human ones. In models of 19.9M to 973M parameters, adding wild AI text lowered human-text loss at 5 tokens per parameter and raised it at 20.
- **Who / date:** Russell et al. (University of Maryland, Pangram Labs), arXiv 2609.40295, published 2026-09-30.
- **Link:** https://arxiv.org/abs/2609.40295
- **Primary checked:** yes (abstract; body sections 2, 4.1, 5.2 and A.1 via fetch).
- **State:** CONFIRMED. The AI labels come from one detector. The paper separates "wild" AI text from synthetic data ("Unlike synthetic data or model-collapse setups, this *wild* AI text comes from many models"), so the result does not transfer directly to SYNTH or FinePhrase.
- **Verbatim:** "In June of 2021, less than 0.1% of FineWeb tokens were AI-generated. By June 2024, it was 10.1%, 16.1% in June 2025, and 27.5% in June 2026." "FineWeb's pipeline keeps AI documents 2.3x as often as human ones, and the DCLM pipeline keeps AI documents 9.8x as often." "With 5 TPP_h, AI decreases loss even at large AI budgets, but at 20 TPP_h, the AI additions increase loss."
- **Why it touches us:** FineWeb-Edu holds dumps up to CC-MAIN-2025-26 (card changelog v1.4.0), and sample-10BT is a random sample of the whole set. Our run is at about 10-37 tokens per parameter now and 50-600 at target.
- **Licence:** WildAI data and code CC BY-NC-SA 4.0 (non-commercial).

### S3. FineWeb-Edu stayed ahead of plain FineWeb, an Olmo mix and a FinePDFs/DCLM/FineWeb-Edu mix in nanochat
- **Claim:** In Karpathy's nanochat log, plain FineWeb dropped CORE from 0.2602 to 0.2241 at d26. The mix `hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT` dropped CORE from 0.2602 to 0.2549 at d26 and from 0.199 to 0.192 at d18. The Olmo 3 mix dropped CORE from 15.5 to 13.8 at d16. ClimbMix later beat FineWeb-Edu.
- **Who / date:** Andrej Karpathy, nanochat `dev/LOG.md`, entries 2026-01-15, 2026-02-17 and 2026-03-04 (outside the window; no data entry falls inside it).
- **Link:** https://github.com/karpathy/nanochat/blob/master/dev/LOG.md
- **Primary checked:** yes (raw file read 2026-10-03 10:50 UTC).
- **State:** CONFIRMED. Metric is CORE, not bpb. One run per arm is shown.
- **Verbatim:** "Tried vanilla fineweb instead of fineweb-edu dataset. Significantly, shockingly worse results: - d26 (GPT-2): CORE 0.2602 → 0.2241". "Tried hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT, a mixture of FinePDFs, DCLM, and FineWeb-EDU. Slightly worse on both model sizes tested".
- **Nuance against the known list:** the mix nanochat tried uses plain FinePDFs. The Smol-Data 50/30/20 set already on our known list uses `finepdfs_edu`. HuggingFaceFW hosts both variants (D_q28), so the nanochat result does not cover the finepdfs_edu variant.
- **Licence:** the HuggingFaceFW mix sets are ODC-By. ClimbMix is `cc-by-nc-4.0` (HF card tag).

### S4. FinePhrase: structured rephrasing of FineWeb-Edu, and it shares documents with our TEST source
- **Claim:** Rephrasing web text into tables, math problems, FAQs and tutorials beat curated web baselines and earlier synthetic methods. Generators larger than 1B gave no extra benefit. FinePhrase is 486B tokens rephrased by SmolLM2-1.7B-Instruct from FineWeb-Edu `sample-350BT`.
- **Who / date:** Niklaus, Penedo, Kydlicek, von Werra, Wolf et al. (Hugging Face), arXiv 2604.13977, published 2026-04-15. Dataset created 2026-02-15. Outside the window and not on the known list.
- **Link:** https://arxiv.org/abs/2604.13977 , https://huggingface.co/datasets/HuggingFaceFW/finephrase
- **Primary checked:** yes (abstract; dataset card).
- **State:** CONFIRMED for the abstract claims. The model sizes used in the paper's ablations were not read.
- **Verbatim:** "Our results reveal that structured output formats, such as tables, math problems, FAQs, and tutorials, consistently outperform both curated web baselines and prior synthetic methods. Notably, increasing the size of the generator model beyond 1B parameters provides no additional benefit." Card: "Source dataset: HuggingFaceFW/fineweb-edu, config sample-350BT, split train".
- **Leakage risk for our TEST:** the FineWeb-Edu card states "`sample-10BT` was sampled from `sample-100BT` which in turn was sampled from `sample-350BT`." So every document in our sample-10BT, held-out rows included, is in FinePhrase's source split. Whether FinePhrase rows carry a source id was not established; the dataset viewer returned HTTP 500.
- **Licence:** ODC-By.

### S5. Short synthetic SFT rows moved GSM8K from 2.35 to 10.31 in a nanochat run
- **Claim:** Removing nanochat's identity and spelling SFT tasks (282k short synthetic rows; mixture 1,071,759 to 789,759 rows) dropped GSM8K from 10.31 to 2.35 on one d24 base checkpoint. Other tasks moved within noise. The reporter reads it as an answer-format failure, not a reasoning one.
- **Who / date:** GitHub user hgt312, nanochat issue #860, 2026-09-26. One reply (mira687, same day) sets out a scorer-based test of the format reading.
- **Link:** https://github.com/karpathy/nanochat/issues/860
- **Primary checked:** yes (issue body and comments).
- **State:** POST CLAIM ONLY. One SFT run per arm, no maintainer reply, no replication.
- **Verbatim:** "GSM8K goes 31 → 136 of 1319 problems. Nothing else moves beyond noise." "the 282k short synthetic rows appear to be what teaches the model to emit a parseable final answer - master's 2.35% looks like a formatting failure, not a reasoning one." (the source's dash is normalised to a hyphen here; the raw log keeps the original)
- **Licence:** nanochat is MIT (repository licence, not re-read here).

### S6. At small scale, hyperparameters are the confound in a data comparison
- **Claim:** Small models from 4M parameters are highly sensitive to hyperparameters, and well-tuned hyperparameters matter more than any other ingredient of the scaling-law recipe. Their runs used the FineWeb-Edu 100B subset.
- **Who / date:** Lourie, Cho, Ullrich, Lotfi, arXiv 2608.11859, published 2026-08-12.
- **Link:** https://arxiv.org/abs/2608.11859
- **Primary checked:** abstract yes. The 4M-268M range and the FineWeb-Edu 100B subset come from a search summary of the PDF (UNVERIFIED).
- **State:** CONFIRMED (abstract).
- **Verbatim:** "Small models are highly sensitive, but hyperparameter sensitivity fades with scale." "By ablating the basic scaling law recipe, we show well-tuned hyperparameters matter more than any other ingredient."
- **Licence:** not a dataset.

### S7. Small mixture experiments mislead when a scarce high-quality set is repeated
- **Claim:** Small proxy runs pick the wrong mixture when the high-quality set's repetition rate changes with budget. Matching the target repetition rate in one experiment at 1/16 of target tokens recovered the mixture within 0.10 of the optimum for a 1.17B model.
- **Who / date:** Zhou, Alazraki, Cao, Rei, arXiv 2606.07597, first published 2026-05-29 (v2 later; just outside the widened window).
- **Link:** https://arxiv.org/abs/2606.07597
- **Primary checked:** abstract yes.
- **State:** CONFIRMED (abstract).
- **Verbatim:** "a primary culprit is a repetition mismatch: because high-quality datasets are small, their repetition rate changes as the training budget grows, shifting the optimal mixture in ways that small-scale proxy experiments do not anticipate."
- **Licence:** not a dataset.

### S8. Parameter Golf: tiny-model FineWeb bpb fell 13.6%, mostly from many small gains
- **Claim:** In OpenAI's 16 MB / 10-minute contest scored on FineWeb validation bpb, the verified best fell from 1.2244 to 1.058 BPB. Individual techniques rarely gained more than 1%.
- **Who / date:** arXiv 2607.01517, published 2026-07-01 (contest ran 2026-03 to 2026-04).
- **Link:** https://arxiv.org/abs/2607.01517
- **Primary checked:** abstract yes. The claim that SP8192 and CaseOps tokenizers reached over 70% adoption came only from a search summary (UNVERIFIED).
- **State:** CONFIRMED (abstract).
- **Verbatim:** "The verified leaderboard score dropped from 1.2244 to 1.058 BPB across three phases, a 13.6% reduction, despite individual techniques rarely improving BPB by more than 1%."
- **Licence:** not a dataset.

### S9. smol-smoltalk is the SmolTalk subset built for models under 1B
- **Claim:** smol-smoltalk keeps shorter Smol-Magpie-Ultra conversations, drops function calling, reduces rewriting and summarisation, and has no advanced math. It was used for SmolLM2-135M-Instruct and SmolLM2-360M-Instruct, with SFT then DPO on UltraFeedback.
- **Who / date:** HuggingFaceTB, card last modified 2025-02-06 (outside the window; context for the chat fine-tune).
- **Link:** https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- **Primary checked:** yes (card).
- **State:** CONFIRMED.
- **Verbatim:** "This is a subset of SmolTalk dataset adapted for smol models with less than 1B parameters." "We include less task specific data compared to SmolTalk (e.g no function calling and less rewriting and summarization examples) since these smaller models have limited capacity".
- **Licence:** `apache-2.0` (card). SmolTalk and SmolTalk2 cards license only their new subsets under Apache 2.0 and refer to the original datasets for the rest.

### S10. Public-domain book sources: one is open for any use, one is gated
- **Claim:** Common Corpus states that all its data "may be used for both commercial and non-commercial purposes". Its OpenCulture part is 967,018,390,906 tokens and includes public domain books and Wikisource. Institutional Books 1.0 is gated on the Hub (`gated: auto`); its README returned HTTP 401, so its terms were not read.
- **Who / date:** PleIAs, card last modified 2026-05-06. Institutional Data Initiative, card last modified 2026-08-19.
- **Link:** https://huggingface.co/datasets/PleIAs/common_corpus , https://huggingface.co/datasets/institutional/institutional-books-1.0
- **Primary checked:** Common Corpus yes. Institutional Books no (401).
- **State:** CONFIRMED for Common Corpus. UNVERIFIED for the Institutional Books licence.
- **Verbatim:** "All data in Common Corpus are either uncopyrighted or freely licensed and may be used for both commercial and non-commercial purposes." "`license`: sharing rights for the content either uncopyrighted (public domain, US federal public domain, CC0 on Wikidata) or various free licenses".
- **Also seen:** Stanford Encyclopedia of Philosophy derived sets on the Hub carry licence tag `other` (AiresPucrs/stanford-encyclopedia-philosophy) or no licence tag (ruggsea/...). Their terms were not verified.

### S11. Lower-weight items, recorded for completeness
- **Book-level organisation of synthetic textbooks** (arXiv 2607.28109, 2026-07-30): replacing natural books in a mid-training mix gave "+1.09 on average". Mid-training at larger scale, Llama3-8B among the models. CONFIRMED (abstract). Licence of the corpus not read.
- **MedLLM, a 0.1B model** (arXiv 2607.27490, 2026-07-29): three-phase pipeline of general pretraining, domain fine-tuning on data "selected from general web data by embedding similarity", and SFT plus DPO. CONFIRMED (abstract).
- **TokEval** (arXiv 2608.18062, 2026-08-18): tokenizer "information-theoretic metrics predict language modeling abilities (Spearman rho up to 0.80)". CONFIRMED (abstract).
- **Self-Play Pretraining with Zero Data** (arXiv 2609.30063, 2026-09-24): a learner trained only on programme-generated byte sequences shows predictable zero-shot loss scaling on natural data. CONFIRMED (abstract). Not a recipe change.
- **Quality-classifier practice:** search summaries say fastText and embedding classifiers reach similar downstream accuracy (Luxical 2512.09015; Apple 2510.00866). These are from 2025 and only summaries were seen. UNVERIFIED.

## 3. Proposed changes to our data recipe (at most 4)

1. **Report TEST bpb by dump and guard it before adding any FineWeb-Edu rephrasing.** Sources: S2 (AI share rising from 10.1% in 2024 to 16.1% in 2025 under FineWeb filtering) and S4 (FinePhrase rephrases `sample-350BT`, which contains every `sample-10BT` document). Test: split the held-out TEST rows by their `dump` field into before-2023 and 2023-2025, report bpb for each, and, before any FinePhrase run, check whether FinePhrase rows can be matched to TEST document ids; if they cannot be matched, do not mix FinePhrase into a run scored on this TEST.

2. **Run one matched synthetic-mix ablation, with a learning-rate pair per arm.** Sources: S1 (SYNTH beats filtered web at iso-compute; Monad has our width), S4 (structured rephrasing beats web baselines), S6 (hyperparameters confound small-scale comparisons). Test: at 300M tokens, train pure FineWeb-Edu against FineWeb-Edu with 20% SYNTH (CC-BY-4.0) at two learning rates each, and compare TEST bpb and the pre-2023 split from change 1. Expect a cost on TEST, because TEST is FineWeb-Edu; the question is whether the chat stage gains more than TEST loses.

3. **Build the chat fine-tune's chat half from smol-smoltalk and add a small slice of short fixed-format rows.** Sources: S9 (smol-smoltalk is the SmolTalk subset built for models under 1B, Apache-2.0) and S5 (282k short synthetic rows moved GSM8K from 2.35 to 10.31 in one nanochat run, POST CLAIM ONLY). Test: two SFT arms on the same base, with and without about 10% short identity and answer-format rows, scored by the share of held-out prompts whose answer parses in the expected format.

4. **Take the philosophy character text from a source whose licence is stated per document, and control repetition.** Sources: S10 (Common Corpus states commercial and non-commercial use is allowed and carries a per-document `license` field; Institutional Books terms were not readable; SEP-derived sets carry `other` or no licence) and S7 (a repeated scarce set shifts the best mixture). Test: filter Common Corpus OpenCulture (or our own public-domain library) to public-domain philosophy works, then compare 1 against 4 passes over it on a held-out philosophy bpb and on general TEST bpb.

Not proposed: switching the base corpus. S3 shows FineWeb-Edu held its place against plain FineWeb, an Olmo mix and the plain-FinePDFs 50/30/20 mix in nanochat, and our TEST is FineWeb-Edu. ClimbMix, the one set that beat it there, is non-commercial (already known).
