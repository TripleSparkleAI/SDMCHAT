# LATESTDATA · the newest open training sets for SDM BASE, SDM CHAT and the philosophy character · 2026-10-03

Lane LATESTDATA, a research lane of the SDMONLY push. Nothing was trained, nothing was rented and nothing was
spent. Every web claim was read from a primary page on 2026-10-03 (UTC): a Hugging Face API response
(`huggingface.co/api/datasets/<id>`, `/api/models/<id>`, `datasets-server.huggingface.co/size`), a dataset or
model card, an arXiv abstract, or a GitHub source file. The sample measurements were taken on the M5 on
2026-10-03 between 10:56Z and 11:10Z, under a load average of about 25 on 18 cores.

Labels used throughout:
- **MEASURED (ours)**: measured in this lane, on rows downloaded to `/private/tmp/latestdata`.
- **MEASURED (elsewhere)**: a number another group published, read from their primary page.
- **SCALED**: our arithmetic on a measured number.
- **ESTIMATE**: a judgement, not a measurement.
- **UNVERIFIED**: seen only second-hand, or the primary page could not be read.

This extends `RESEARCH_BASEDATA_2026-10-04.md` (FineWeb-Edu, ClimbMix, Nemotron-CC, DCLM, Smol-Data,
FinePDFs) and `RESEARCH_OPENDATA_MODEL_2026-10-01.md` (K2 Horizon and TxT360-v2). Those findings are not
repeated here.

## 0. What is new since BASEDATA and OPENDATA

- **MiniCPM5 (OpenBMB) shipped with its training data.** MiniCPM5-1B (model created 2026-05-21) and
  MiniCPM5-2B (2026-09-06, Apache-2.0) name their training sets in the card metadata: Ultra-FineWeb,
  UltraX-Preview, Ultra-FineWeb-L3, UltraData-Math, UltraData-Code, UltraData-SFT-2605, UltraData-SFT-Agent-2609
  and UltraData-RL-2609. The card gives the stages and does not give mixture weights.
- **Ultra-FineWeb gained 2025 Common Crawl.** On 2026-08-20 OpenBMB released `openbmb/Ultra-FineWeb-L1` (six 2025
  dumps, CC-MAIN-2025-30 to 2025-51, "1T+ tokens") and its classifier-selected subset inside
  `openbmb/Ultra-FineWeb` at `data/ultrafineweb_l1_en_hq/`. The card calls it the most recent open web
  pretraining set it knows of, "up to `CC-MAIN-2025-51`".
- **UltraX (2026-07-13, arXiv 2607.08646)**: five web corpora of about 20B tokens each, edited line by line by
  a small model. Apache-2.0.
- **FinePhrase (HF, 2026-03-08)**: 486.4B tokens of FineWeb-Edu rewritten as FAQ, tutorial, table and math
  documents by SmolLM2-1.7B-Instruct. ODC-By. Not covered by BASEDATA.
- **ZGCM-1-7B (2026-09-07, arXiv 2609.13356)** shipped its per-stage data (`zgcagi/ZGCM-1-Data`, about 5.7 TB
  of parquet). It is gated, under licence `other`, and Chinese, math and agent heavy. Not for us.
- **nanowhale-100m (HF, 2026-04-24)** is the closest published analogue to our setting: about 110M parameters
  and **our exact tokenizer** (DeepSeek-V4, vocabulary 129,280), pretrained on ~2.6B tokens of FineWeb-Edu and
  fine-tuned on 72.7M tokens of `HuggingFaceTB/smol-smoltalk`. That is the data we already use.
- **No new chat set beats smol-smoltalk for a model of our size.** The newer sets (smoltalk2, UltraData-SFT,
  IFM SFT-Reasoning, the Apertus 1.5 SFT mix) are built around reasoning traces, tool calls and multilingual
  data. Their useful part for us is a handful of short English subsets of smoltalk2.
- **A defect in our own chat shards (section 4.2).** Our template drops system messages. In the smol-smoltalk
  test split, **every** rewrite and summarize conversation carries its instruction only in the system message,
  so our chat model has been trained to rewrite and summarize text it was never asked to change.

```
   what moved on the Hub since the summer, by creation date
   2026-07-13 ·──── UltraX-Preview          5 x ~20B refined web     Apache-2.0
   2026-08-20 ·──── Ultra-FineWeb-L1 (+HQ)  2025 Common Crawl        Apache-2.0
   2026-08-24 ·──── IFM / TxT360-v2         (OPENDATA)               CC-BY-4.0
   2026-08-27 ·──── MiniCPM5-2B-Base        model + data named
   2026-09-05 ·──── ZGCM-1-Data             7B math/agent, gated     other
   2026-09-06 ·──── MiniCPM5-2B             RL + OPD release
```

## 1. Pretraining sets

### 1.1 Sets released or updated since 2026-07-01, and older sets not yet covered

| repo id (revision read) | released / modified | licence, gate | what it is | size | evidence at small scale |
|---|---|---|---|---|---|
| `openbmb/Ultra-FineWeb` path `data/ultrafineweb_l1_en_hq/` (`02c85641`) | 2026-08-20 | Apache-2.0 plus source terms, ungated | L1-cleaned 2025 Common Crawl (trafilatura 2.0, fastText language id, FineWeb heuristics, PII replacement, per-dump MinHash), then the Ultra-FineWeb fastText classifier | 6,000 files, 478.0 GB parquet; **138B to 152B DeepSeek tokens** (SCALED, section 2) | card: MiniCPM5-1B, 20B tokens, Muon. FineWeb 9.033% macro, L1 9.668%, classifier on L1 **10.379%** (MEASURED elsewhere) |
| `openbmb/Ultra-FineWeb-L1` (`10b9ba18`) | 2026-08-20 | Apache-2.0, ungated | the same six 2025 dumps before classifier selection | 6 configs x 1,000 files, 2.83 TB parquet, 1,144,985,860 rows (dataset-server `/size`) | the 9.668% row above |
| `openbmb/Ultra-FineWeb` `data/ultrafineweb_en/` | 2025-06, modified 2026-08-20 | Apache-2.0 (built on FineWeb, ODC-By) | the older L2 set, classifier-selected FineWeb | 2,048 files, 2,661.4 GB; 1,159,254,991 rows; card "about 1T tokens" | technical report arXiv 2505.05427 (1.2B model, 100B tokens per arm). FinePhrase's ablation at 1.7B ranks it below DCLM (UNVERIFIED in detail, read through the Space page) |
| `openbmb/UltraX-Preview` (`a8852758`) | 2026-07-13 | Apache-2.0, ungated | five corpora edited by a small model with `keep_all`, `remove_all`, `remove_lines`, `replace_str`, `add_line`. Columns hold both `raw_content` and `cleaned_content` | 479 parquet files, 487 GB; config `UltraX-Ultra-FineWeb` is 104 files, 108.9 GB, about **18.3B DeepSeek tokens** (SCALED) | card: 1B MiniCPM, 20B tokens per corpus, about +2.00% relative over raw text, best average on all five corpora (MEASURED elsewhere) |
| `openbmb/Ultra-FineWeb-L3` (`bc3b1ba9`) | 2026-05-28, modified 2026-08-20 | Apache-2.0, ungated | Ultra-FineWeb rewritten as Q&A pairs and in several styles | English 1,258.0 GB parquet (card: 400B+ English tokens) | none read at small scale |
| `HuggingFaceFW/finephrase` (`78cf4a5e`) | 2026-02-15, modified 2026-03-31 | ODC-By, ungated | 339.3M FineWeb-Edu `sample-350BT` documents rewritten by SmolLM2-1.7B-Instruct into `faq`, `tutorial`, `table`, `math` | 27,104 files, 5.16 TB parquet (source text included); 486.4B completion tokens in SmolLM2's tokenizer; **about 431B DeepSeek tokens** (SCALED) | Synthetic Data Playbook (2026-03-08): 1.7B model, 20B tokens. "synthetic-only training falls short of both DCLM and mixed training"; best synthetic share 60% (faq, tutorial) to 80% (math); 1B to 27B rephrasers equal, 270M worse on math and tutorial (MEASURED elsewhere) |
| `PleIAs/SYNTH` (`0d6813a2`) | 2025-11-10, modified 2026-05-06 | CC-BY-4.0 tag; seeds CC-BY-SA, ungated | 79.6M synthetic exercises seeded from 58,698 Wikipedia vital articles; every answer carries a reasoning draft; 20% non-English | 502 files, 236.3 GB; card: 75B tokens in the Pleias tokenizer | card: "state of the art for small models below 350 million parameters", 100B to 200B tokens. FinePhrase's ablation ranks it among the weaker synthetic sets at 1.7B (UNVERIFIED in detail) |
| `zgcagi/ZGCM-1-Data` (`4c6405d0`) | 2026-09-05 | `other`, gated (auto) | the full ZGCM-1-7B recipe: two pretraining stages, three long-context midtraining stages, SFT | 5,740 GB of parquet by file metadata | none at small scale; Chinese and English, math and agent focus |
| `nvidia/Nemotron-CC-v2` | modified 2026-07-07 | gated (manual) | covered by BASEDATA (v2.1) | | what changed on 2026-07-07 was not read (UNVERIFIED) |

`HuggingFaceFW/fineweb-edu` (our yardstick) was last modified 2025-07-11. `nvidia/Nemotron-ClimbMix` was last
modified 2025-10-21. Neither has changed since BASEDATA.

**Searches that found nothing newer and better.** I listed the Hub by trending score (text-generation filter
and unfiltered, 100 each), and by creation date for 37 organisations (HuggingFaceFW, HuggingFaceTB, openbmb,
allenai, nvidia, PleIAs, EleutherAI, LLM360, mlfoundations, common-pile, marin-community, swiss-ai, IFM,
karpathy, microsoft, Qwen, deepseek-ai, LiquidAI, google, BAAI, m-a-p and others). Apart from the rows above,
the new text datasets since 2026-07-01 are SFT, RL and agent sets, benchmarks, or re-tokenised copies of older
corpora. **I found no English web corpus released since 2026-07-01 with a published small-model win over
FineWeb-Edu other than the Ultra-FineWeb L1 result above**, and that result is at 1B parameters, not ours.

### 1.2 Models that shipped with their full data or exact recipe

| model | created | params | data released | fit for us |
|---|---|---|---|---|
| openbmb/MiniCPM5-2B | 2026-09-06 | 2B | Ultra-FineWeb, UltraX, Ultra-FineWeb-L3, UltraData-Math, UltraData-Code (pretraining); UltraData-SFT-2605 (gated), SFT-Agent-2609, RL-2609 | the web part is usable; mixture weights are not published |
| IFM/K2-Horizon (0.9B to 375B) | 2026-09-01 | | covered by OPENDATA | covered by OPENDATA |
| zgcagi/ZGCM-1-7B | 2026-09-07 | 7.39B | `ZGCM-1-Data`, per stage | no: gated, `other`, math and agent data |
| HuggingFaceTB/nanowhale-100m | 2026-04-24 | ~110M (41M embedding) | FineWeb-Edu ~2.6B tokens, then smol-smoltalk ~72.7M tokens (card) | **the closest analogue**: our tokenizer, our size, our two datasets. Held-out perplexity 13.62 base, 12.90 after SFT (MEASURED elsewhere) |
| L20-Edu-135M (arXiv 2606.22189, 2026-06-20) | | 134.5M | 10B FineWeb-Edu tokens, then a 3B-token education, math, code and reasoning mixture (abstract) | the repo ids of the 3B mixture were not read (UNVERIFIED) |

## 2. The top candidates at our tokenizer (MEASURED, ours)

Method, copied from BASEDATA: read three row groups (first, middle, last) of one file through `HfFileSystem`,
only the needed columns, tokenise each document with `data/ds4_v4flash_tokenizer.json`
(`add_special_tokens=False`), and count UTF-8 bytes. The script is a scratch file outside the repo; its output
is `/private/tmp/latestdata/measure_latestdata.json`. About 155 MB crossed the network in this lane.

| set (file read) | docs | bytes per token | doc tokens p10 / median / mean / p90 | docs under 256 tokens | tokens in docs over 256 |
|---|---|---|---|---|---|
| FineWeb-Edu `sample/10BT` (S0 train, from provenance) | 64,488 | **4.776** | n/a | n/a | n/a |
| ClimbMix shard 0 (BASEDATA) | 3,072 | 4.832 | n/a | n/a | n/a |
| Ultra-FineWeb L1-HQ, CC-MAIN-2025-51 part 1 | 1,099 | **5.308** | 215 / 698 / 1,007 / 2,058 | 13.6% | 97.5% |
| UltraX-Ultra-FineWeb, `cleaned_content`, part 1 | 2,949 | 4.898 | 121 / 395 / 762 / 1,521 | 33.1% | 93.1% |
| FinePhrase `faq/000_00000_0` | 3,000 | 4.960 | 89 / 377 / 393 / 643 | 27.2% | 91.1% |
| FinePhrase `tutorial/000_00000_0` | 3,000 | 4.865 | 128 / 370 / 391 / 625 | 25.0% | 90.7% |
| FinePhrase `table/000_00000_0` | 3,000 | 5.728 | 84 / 191 / 241 / 403 | 69.7% | 56.6% |
| FinePhrase `math/000_00000_0` | 3,000 | 4.503 | 37 / 208 / 248 / 472 | 60.2% | 69.8% |

```
   bytes per DeepSeek token (more bytes per token = more text per training token)
   FinePhrase math     4.503  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   FineWeb-Edu         4.776  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   ClimbMix            4.832  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   FinePhrase tutorial 4.865  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   UltraX-UF cleaned   4.898  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   FinePhrase faq      4.960  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   UF L1-HQ 2025       5.308  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   FinePhrase table    5.728  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

**What the sample shows:**
- **Ultra-FineWeb L1-HQ has long documents.** Its median is 698 tokens against UltraX's 395 and FinePhrase's
  370 to 377. At our window of T 256, 97.5% of its tokens sit in documents longer than one window.
- **Every L1-HQ document in the sample is labelled `__label__pos`, with `pred_score` at least 0.5** (p10 0.552,
  median 0.830, p90 0.990). So the HQ subset is the classifier's positive class.
- **UltraX edits almost every document and removes little text.** 98.5% of the 2,949 documents differ from
  their raw form, and the cleaned text is 96.9% of the raw bytes.
- **FinePhrase `table` and `math` are short.** 60% to 70% of their documents fit inside one 256-token window.
- **Bits per byte stays comparable across these sets**, because it divides by bytes, not tokens. A set with
  more bytes per token does give more text per training token.

**Totals at our tokenizer (SCALED from these samples):**
- L1-HQ: 22,869 documents in the file read, times 6,000 files, times a mean of 1,007 tokens, is **138B
  tokens**. By bytes instead (1.687 text bytes per parquet byte, 478.0 GB, 5.308 bytes per token) it is
  **152B**. One file was read, so the range is the honest figure.
- UltraX-Ultra-FineWeb: 230,975 rows per file x 104 files x 762 tokens = **18.3B tokens**.
- FinePhrase: per-config documents (card) times our mean tokens give faq 133.2B, tutorial 132.1B, table 81.6B,
  math 84.0B, **430.9B in all**.

**Overlap with our TEST.** The 2,000 TEST documents (rows 64,488 to 66,487 of FineWeb-Edu
`sample/10BT/000_00000.parquet`) come from two dumps, CC-MAIN-2020-05 and CC-MAIN-2023-14 (MEASURED, ours).
- **L1-HQ** is built from 2025 dumps, so the same document can only recur under the same URL in a later
  crawl. 0 of the 1,099 sampled URLs were TEST URLs. With a sample this small that zero carries no power.
- **FinePhrase rewrites FineWeb-Edu documents, and some of them may be our TEST documents.** FineWeb-Edu holds
  about 1.3T tokens and `sample-350BT` about 27% of them. If the two samples were drawn independently, about
  540 of our 2,000 TEST documents would appear in FinePhrase in rewritten form (ESTIMATE). FinePhrase keeps
  FineWeb-Edu's `id` and `url` columns, so they can be removed exactly. 0 of 12,000 sampled rows matched; the
  expected count in a sample that size is about 0.00002, so that zero says nothing either.
- **Ultra-FineWeb `ultrafineweb_en` (the older L2 set) is built from FineWeb, which contains FineWeb-Edu's
  sources, and has no URL column** (`content`, `score`, `source`). It can only be cleaned against TEST by text
  overlap. That is one reason to prefer the L1-HQ path.

## 3. Chat and instruction sets for a small model

### 3.1 What the newest small instruct models were tuned on

| set (revision) | date | licence | size | format | reasoning or tools | for a model of our size |
|---|---|---|---|---|---|---|
| `HuggingFaceTB/smol-smoltalk` (`f73fe857`, **ours today**) | 2024-11, modified 2025-02-06 | Apache-2.0 | 460,341 train / 24,229 test conversations; 378.9M DeepSeek tokens in train (MEASURED, ours) | `messages` (role, content) | none; 10.7% code (`self-oss-instruct`) in test | built for SmolLM2-135M/360M. Used by nanochat's SFT and by nanowhale-100m |
| `HuggingFaceTB/smoltalk2` (`fc6cc210`) | 2025-07-10, modified 2025-10-31 | new subsets Apache-2.0; others per source | SFT 3.38M rows, 19.3B tokens (card, SmolLM3 tokenizer); Mid 4.8M; Preference 447k | `messages`, `chat_template_kwargs` (system text, tools, thinking flag) | 10 `_think` subsets (1.5M rows); tool calls in 4 subsets | built for SmolLM3-3B. A few short English `no_think` subsets fit us (section 3.2) |
| `openbmb/UltraData-SFT-2605` (`affda6ac`) | 2026-05-21 | Apache-2.0, **gated (auto)** | MiniCPM5's "400B tokens of deep-thinking SFT" | card behind the gate, not read | deep-thinking | no: reasoning traces, and far too large |
| `IFM/SFT-Reasoning` (`4d4dbbe6`) | 2026-08-24 | Apache-2.0 | instruction-following 8,581,751 rows, 43.4 GB; 3efforts-pretrain 32,216,066 rows | flat `text` + `token_count`; the first row reads the constraints, then the model's reasoning about them | yes, inline | no: the reasoning is inside the text and cannot be separated |
| `swiss-ai/Apertus-v1.5-SFT-mix` (`d38f5b98`) | 2026-09-09 | `other` | 3,707,851 rows; 1,045,577 multilingual, 869,593 math, 514,564 code, 220,352 tool calling | saved-dataset format, `messages` | tool definitions injected into 131,557 rows | no: licence `other`, and mostly multilingual, math, code and tools |
| `allenai/tulu-3-sft-personas-instruction-following` | 2024 | ODC-By | 29,970 rows (inside smoltalk2) | messages | none | yes, through smoltalk2 |

nanochat's current chat fine-tune (`scripts/chat_sft.py`, HEAD `92d63d4e`) still uses smol-smoltalk train
(460K rows), MMLU auxiliary train x3 and GSM8K train x4, validated on smol-smoltalk test.

### 3.2 The smoltalk2 subsets that fit a tiny chat model (MEASURED, ours, in our template)

Template as in `track4_sdmllm24_prepare_smoltalk_chat_shards.py`: `User: ...\n` and `Assistant: ...\n` per turn,
[BOS 0] per conversation.

| subset | file | conversations | DeepSeek tokens | bytes per token | median tokens per conversation | system text in `custom_instructions` |
|---|---|---|---|---|---|---|
| `smoltalk_smollm3_everyday_conversations_no_think` | 0.9 MB | 2,260 | 377,701 | 4.620 | 168 (3.82 assistant turns) | 0% |
| `smoltalk_smollm3_explore_instruct_rewriting_no_think` | 5.3 MB | 30,391 | 1,970,630 | 5.129 | 58 | 50.2% |
| `tulu_3_sft_personas_instruction_following_no_think` | 33.2 MB | 29,970 | 11,309,678 | 5.121 | 322 | 0% |
| `smoltalk_smollm3_smol_rewrite_no_think` | 38.5 MB | 53,262 | 15,488,721 | 5.126 | 265 | **100%** |

Not measured here, sizes from the card (SmolLM3 tokenizer): `smoltalk_smollm3_systemchats_30k_no_think`
33,997 rows, 22.06M tokens; `smoltalk_smollm3_smol_summarize_no_think` 96,061 rows, 51.82M tokens;
`smoltalk_smollm3_smol_magpie_ultra_no_think` 406,843 rows, 619.05M tokens, 6 turns and 1,522 tokens on average.

**Exclude for a model of tens of millions of parameters:** every `_think` subset (OpenThoughts3 `_think`
averages 14,763 tokens per example), `OpenThoughts3_1.2M_no_think_no_think` and
`Mixture_of_Thoughts_science_no_think` (math and science answers without their working),
`hermes_function_calling_v1_no_think`, `xlam_traces_no_think` and `smolagents_toolcalling_traces_think` (tool
calls), the two `LongAlign` subsets (15,000 to 18,000 tokens each), `table_gpt`, `s1k`, the multilingual
subsets (our TEST and template are English), and `OpenHermes_2.5_no_think` (its licence was not read; smol-smoltalk
already carries `openhermes-50k`).

### 3.3 A defect in our chat shards: the dropped system message (MEASURED, ours)

`track4_sdmllm24_prepare_smoltalk_chat_shards.py` keeps only `user` and `assistant` messages. In the
smol-smoltalk **test** split, 3,714 of 24,229 conversations (15.3%) open with a system message:

| source in smol-smoltalk test | conversations | with a system message |
|---|---|---|
| `smollm-rewrite-30k` | 1,418 | **1,418** |
| `smol-summarize-20k` | 1,013 | **1,013** |
| `smol-summarize-5k` | 251 | **251** |
| `openhermes-50k` | 2,508 | 953 |
| `explore-instruct-rewrite` | 183 | 79 |
| `smol-magpie-ultra-short` | 14,162 | 0 |
| `self-oss-instruct` | 2,590 | 0 |

The system messages are the task itself: "Rewrite the input text to make it more concise while preserving its
core meaning", "Provide a concise, objective summary of the input text in up to three sentences". With them
dropped, **2,682 test conversations (11.1%) show a user pasting text and the assistant rewriting or summarising
it unasked.** The train split has the same sources (the first 1,000 rows of `train-00000`: 157 with a system
message). smoltalk2's `smol_rewrite` subset is worse still for our template: all 53,262 rows keep their
instruction in `chat_template_kwargs.custom_instructions`, which our template never reads.

**The fix, as nanochat does it** (`nanochat/tokenizer.py`, `render_conversation`): "sometimes the first message
is a system message... just merge it with the second (user) message". nanochat also trains only on the
assistant's tokens (a `mask` of 1 on assistant tokens). CHATGUYCHAIN's trainer already reads a `_mask.u8`
beside a shard, so the re-prepared shard should ship one.

## 4. Philosophy text with a clear licence

| source | licence | what it holds | size |
|---|---|---|---|
| Project Gutenberg catalogue `pg_catalog.csv` (last modified 2026-09-27) + the texts | public domain in the United States; strip the Gutenberg header and footer | English texts with Library of Congress class **B, BC, BD, BH or BJ** (philosophy, logic, metaphysics, aesthetics, ethics; religion BL to BX and psychology BF left out): **688 books** (MEASURED, ours). The broader bookshelf "Category: Philosophy & Ethics" holds 2,519 | 688 x 0.375 MB (mean Gutenberg book) to 688 x 1.04 MB (mean of the 8 books below) = 0.26 to 0.71 GB, **59M to 164M DeepSeek tokens** (SCALED) |
| `sedthh/gutenberg_english` (`28973b04`) | MIT tag on the packaging; texts as above | 48,284 English books from 2023 with LoCC, subjects and bookshelves in `METADATA` | 531 core-B books (856 with BF), 10.7 GB parquet in all |
| `common-pile/project_gutenberg_filtered` (`3cdf6879`) | per-row `license: Public Domain` | 14,354 books, filtered for the Common Pile; metadata has title and URL, no subject. **Only 181 of the 688 core philosophy books are in it** (ids read from the Hub's parquet conversion, MEASURED, ours) | 3.09 GB parquet (dataset-server), stored as 15 `json.gz` files of 7.6 GB |
| `common-pile/pre_1929_books_filtered` (`23f9d96d`) | public domain by date (HathiTrust) | pre-1929 books, metadata has author and a HathiTrust URL, no subject | not measured |
| our own Library of Alexandria | public domain, per its own provenance | 3,149 books; already used by lane GUYCORPUS | GUYCORPUS's measurement |

**Bytes per token for old philosophy prose (MEASURED, ours):** 8 Gutenberg books (Plato's Republic, Hume's
Treatise, Kant's Critique of Pure Reason, Mill's On Liberty, Aristotle's Nicomachean Ethics, Russell's
Problems of Philosophy, Montaigne's Essays, Machiavelli's Prince), header and footer stripped, 8,307,842 bytes
in 1,911,011 tokens: **4.347 bytes per token**, from 4.145 (Montaigne) to 4.688 (Kant).

**Left out:** `AiresPucrs/stanford-encyclopedia-philosophy` (licence `other`; the encyclopedia is copyrighted),
`LisaMegaWatts/philosophy-corpus` (MIT tag, 15.2M lines of humanities text mixed with Wikipedia, no per-line
source), `CristianPelayo/openalex-philosophy` (CC0 metadata, no text), `Saaih/Philosophy-Sets` (CC-BY-NC).

**How much to repeat it.** Sedova et al. (arXiv 2605.12715, v2 2026-05-15), over 2,000 runs from 101M to 805M
parameters: "mixture training tolerates much higher repetition than single-source training: scarce target
corpora can be reused 15-20 times". A 59M to 164M-token corpus mixed with web and chat rows can therefore run
for many rounds before it is over-used (MEASURED elsewhere for their domains; ESTIMATE for ours).

## 5. Preparation practice in the newest recipes

| practice | what the recipes do | source | where we stand |
|---|---|---|---|
| deduplication | MinHash near-duplicate removal **within each Common Crawl dump**, not across dumps | Ultra-FineWeb-L1 card, following FineWeb | inherited from the sources; we add nothing |
| quality classifier | FineWeb-Edu keeps edu score 3 and up; Ultra-FineWeb uses a fastText classifier (HQ = positive class, `pred_score` at least 0.5 in our sample) | FineWeb-Edu card; Ultra-FineWeb card and our sample | inherited |
| cleaning | trafilatura 2.0, mojibake, invisible characters, residual HTML, abnormal lengths; PII replaced with placeholders | Ultra-FineWeb-L1 card | inherited |
| refinement and rewriting | line edits (UltraX); rewriting into FAQ, tutorial, table, math (FinePhrase), mixed 60% to 80% synthetic with natural text, never synthetic alone | UltraX card; Synthetic Data Playbook | not used yet |
| decontamination | benchmarks removed from SFT sets; for us, remove our TEST documents by `id`, by URL, or by text n-grams | smoltalk2 card; this lane | owed for every new source |
| packing | every row starts with BOS; documents placed best-fit to cut cropping; "~35% tokens cropped at T=2048" | nanochat `dataloader.py`; Ding et al. arXiv 2404.10830 (best-fit packing: +4.7% reading comprehension, up to 58.3% less hallucination, relative) | we store [BOS 0] + ids and draw windows at random offsets, so a window may cross a document boundary. At T 256 and medians of 370 to 698 tokens most windows sit inside one document |
| held-out set | the last shard is validation and is always downloaded | nanochat `dataset.py` | we hold out the last 2,000 documents of one FineWeb-Edu file |
| shuffling | corpora published pre-shuffled at shard level (`karpathy/climbmix-400b-shuffle`, the Smol-Data `-shuffled` repos); Apertus shuffles with a stated seed (`20260909`) | Hub repos and cards | one shard per source; CHATGUYCHAIN's `--mix` sets exact rows per batch per source |
| token format | parquet text tokenised on the fly (nanochat); uint16 `.bin` for a 50,257 vocabulary (modded-nanogpt) | GitHub | uint32, because 129,280 > 65,535 |
| streaming | read only the row groups and columns needed | our measurement: FinePhrase's rewritten text is 0.96 MB of each 3.9 MB row group, so a column read moves about 4 times less data | `HfFileSystem` already used by the FineWeb-Edu preparer |
| chat rendering | merge a system message into the first user turn; loss on assistant tokens only | nanochat `tokenizer.py` | **not done** (section 3.3) |

## 6. Recommendation

### 6.1 The base mix for the long run

Every share below is an ESTIMATE until the data wave in section 6.5 has scored it. The plan's step 2 targets
3B tokens; BASEDATA recommends 3B to 10B. The shares are written for **6B tokens** and scale linearly.

```
   LICENCE-CLEAN (ODC-By and Apache-2.0)                   tokens at 6B
   FineWeb-Edu sample/10BT (have it: train_big, 3.018B)  45%  ██████████████████  2.7B
   Ultra-FineWeb L1-HQ, 2025 dumps                       35%  ██████████████      2.1B
   FinePhrase faq + tutorial                             20%  ████████            1.2B

   RESEARCH-ONLY (ClimbMix is CC-BY-NC-4.0)
   ClimbMix 400B shuffle                                 50%  ████████████████████  3.0B
   FineWeb-Edu                                           30%  ████████████          1.8B
   FinePhrase faq + tutorial                             20%  ████████              1.2B
```

**Why this shape:**
- **FineWeb-Edu stays the largest share.** It is in-distribution for TEST A and it is already tokenised.
- **L1-HQ adds 2025 text that no other set holds**, ranked above FineWeb at 1B parameters by its authors, with
  URLs for decontamination. It is the licence-clean answer to ClimbMix.
- **FinePhrase is capped at 20%.** The playbook's best share was 60% to 80% synthetic, at 1.7B parameters and
  20B tokens. Our yardstick is bits per byte on natural web text, and the rewritten text has a different
  shape (FAQ and tutorial layouts). 20% buys the format without moving TEST A far. The data wave decides
  whether more is better.
- **One stage, one mix, to the end** (SmolLM2's finding for 135M and 360M, in BASEDATA). The WSD schedule in the
  trainer allows a second, cheap answer: fork the cooldown at the end of the constant phase twice, once on the
  same mix and once on FineWeb-Edu alone, and keep whichever scores better on both tests.
- **Left out of the base mix:** UltraX (18.3B tokens, edits nearly every document, no URL column; a later arm),
  Ultra-FineWeb-L3 and SYNTH (synthetic Q&A and reasoning drafts; SYNTH is 20% non-English), the older
  `ultrafineweb_en` (no URL column, FineWeb sources overlap TEST A), and everything gated.

### 6.2 The chat mix

```
   SDM CHAT rows, half chat and half web as today (chat_mix)
   smol-smoltalk train, RE-PREPARED                    ~ 340M to 380M tokens
      system message merged into the first user turn
      assistant-only loss mask (_mask.u8)
      self-oss-instruct (code, 10.7% of test) dropped   <- an arm, ESTIMATE
   smoltalk2 English no_think subsets                  ~ 100M tokens (SCALED)
      everyday_conversations     0.38M   explore_instruct_rewriting  1.97M
      tulu_3 personas IF        11.31M   smol_rewrite               15.49M
      systemchats_30k          ~22M      smol_summarize            ~52M
      custom_instructions merged into the first user turn
   web rows from the end of train_big, as today
```

- Licences: smol-smoltalk Apache-2.0; the smoltalk2 new subsets Apache-2.0; tulu-3 personas IF ODC-By;
  everyday-conversations, systemchats and explore-instruct are "per the original dataset" (not read, UNVERIFIED).
- `smoltalk_smollm3_smol_magpie_ultra_no_think` (619M tokens, 6 turns) may share prompts with smol-smoltalk's
  `smol-magpie-ultra-short`. That overlap was not measured (UNVERIFIED). Leave it out, or take a small share
  after deduplicating on the first user message.
- Hold out 2% of each new subset by a hash of its first user message, never trained on.

### 6.3 The philosophy sources for the character

1. **Project Gutenberg, LoCC B, BC, BD, BH, BJ, English: 688 books, 59M to 164M tokens (SCALED).** Select with
   `pg_catalog.csv`; take the text from `sedthh/gutenberg_english` (its 2023 snapshot marks 531 books core-B by
   its own LoCC field) and from Gutenberg's own files for the rest; strip the header and footer. The Common
   Pile's filtered copy holds only 181 of the 688, so it is not the right source here.
2. **Our Library of Alexandria**, through lane GUYCORPUS.
3. Optional, broader: the 2,519 books on the "Philosophy & Ethics" bookshelf, read by a person before use,
   because that shelf includes essays, letters and politics.

### 6.4 The held-out tests

| test | built from | size | purpose |
|---|---|---|---|
| **TEST A** (unchanged) | S0's 2,000 FineWeb-Edu documents, 3,892 windows (`val.u32`, sha256 `89fbba2c...`) | 2,000,916 tokens | comparable with every earlier SDMLLM score |
| **TEST B** (new) | Ultra-FineWeb L1-HQ `CC-MAIN-2025-51/...part-1000-of-1000.parquet`, first 2,000 documents, windows by S0's rule. That file is never trained on, as in nanochat's last-shard rule | about 2.0M tokens (SCALED: 2,000 x 1,007) | 2025 web text, outside FineWeb-Edu |
| CHAT | `chat_test.u32` (smol-smoltalk test, 24,229 conversations), plus the 2% holdouts of section 6.2 | 19.8M tokens + new | instruction following; re-prepared with system messages merged, and scored beside the old file |
| GUY | GUYCORPUS's own test | | the character |

Before any new source is tokenised, remove TEST A from it: FinePhrase by `id` (and `url`), L1-HQ by
`meta.url`. Record the counts removed in the provenance JSON. Report every base model on both TEST A and TEST B.

### 6.5 Before the long run: a data wave on the Spark (PROPOSAL to JIMOTHY)

Five arms of the SDM-only centre (`p0_A_c` settings, d 256), 100M tokens each, scored on TEST A and TEST B:
FineWeb-Edu alone (the control), L1-HQ alone, FinePhrase faq + tutorial alone, the licence-clean mix, and
ClimbMix alone. At the bench rate of 46,700 tokens a second, one arm takes about 36 minutes and the wave about
3.0 hours (SCALED from the d 256 bench recorded in `SDMONLY_CHANNEL.md`). The data for it is 100M tokens per source, minutes to prepare.

### 6.6 Download and tokenising steps, with time and disk on a rented box

Pin every revision. Write `[BOS 0] + ids` as uint32, one shard per source, with a provenance JSON (repo,
revision, files, rows, TEST rows removed, sha256), as `track4_sdmllm_prepare_fineweb_edu_token_shards.py` does.

1. **TEST A ids and URLs.** Read columns `id`, `url` for rows 64,488 to 66,487 of
   `HuggingFaceFW/fineweb-edu@87f09149.../sample/10BT/000_00000.parquet`. Seconds.
2. **TEST B.** Download `openbmb/Ultra-FineWeb@02c85641e3d19a854be2e09139c25adaa9518063`,
   `data/ultrafineweb_l1_en_hq/CC-MAIN-2025-51/ultrafineweb-l1-en-hq-CC-MAIN-2025-51-part-1000-of-1000.parquet`
   (about 80 MB); first 2,000 documents (`content`) to `uf25_test.u32`.
3. **L1-HQ train, 2.1B tokens.** At 22,869 documents and about 23.0M tokens per file, 92 files: 15 or 16 from
   each of the six dumps, spread evenly over parts 0001 to 0999, never part 1000 of CC-MAIN-2025-51. Drop rows
   whose `meta.url` is a TEST A URL. Columns `content`, `meta`. To `uf25_train.u32`.
4. **FinePhrase, 1.2B tokens.** `HuggingFaceFW/finephrase@78cf4a5ed0099214979c094c963e699c19163838`, configs
   `faq/` and `tutorial/` (files `NNN_NNNNN_K.parquet`, 67 row groups of 1,000 rows each). Read only columns
   `rollout_results` (text at `rollout_results[0].text`), `id` and `url`; drop TEST A ids; about 3.06M rows.
   To `fphrase_train.u32`.
5. **Chat.** smol-smoltalk at `f73fe857`, and the six smoltalk2 subsets at `fc6cc210` (242.5 MB of parquet).
   Render with the system message or `custom_instructions` merged into the first user turn; write
   `chat2_train.u32`, `chat2_train_mask.u8`, `chat2_test.u32`.
6. **Philosophy.** `pg_catalog.csv` from gutenberg.org, then the texts of the 688 selected books.

| step | download | text | tokens | uint32 shard | tokenise time, M5 rate | tokenise time, rented box |
|---|---|---|---|---|---|---|
| L1-HQ | 92 files, 7.3 GB | 11.2 GB | 2.1B | 8.4 GB | 13.6 min | 5 to 15 min (ESTIMATE) |
| FinePhrase | 3.2 GB of columns | 5.9 GB | 1.2B | 4.8 GB | 7.2 min | 3 to 8 min (ESTIMATE) |
| TEST B | 80 MB | ~10 MB | ~2.0M | 8 MB | seconds | seconds |
| chat | 1.2 GB | ~2.3 GB | ~0.48B | 1.9 GB + 0.48 GB mask | 2.8 min | 1 to 3 min (ESTIMATE) |
| philosophy | 0.26 to 0.71 GB | same | 59M to 164M | 0.24 to 0.66 GB | under 1 min | under 1 min |

- **M5 rate (MEASURED, ours):** `encode_batch` on 2,112 documents of 4 KB, 8.5 MB: 13.5 to 13.9 MB a second on
  18 cores at load 25. The table's times are SCALED from 13.7 MB a second.
- **Download rate:** not measured; at 25 to 100 MB a second, 11.7 GB takes 2 to 8 minutes (ESTIMATE).
- **Disk:** about 11.7 GB of parquet, 15.6 GB of new shards, and the existing `train_big.u32` (3.018B tokens,
  12.1 GB). **About 40 GB in all**, so a 100 GB disk is ample. The whole preparation runs in under an hour on
  a rented box, before the GPU work starts (ESTIMATE).

## 7. What stayed UNVERIFIED

- The mixture weights of MiniCPM5-2B; the UltraData-SFT-2605 card (gated).
- How FinePhrase's ablation ranks Ultra-FineWeb and SYNTH in detail: read from the Space page as summarised
  text, not from a table.
- Whether 2025 Common Crawl dumps hold more model-written text than earlier dumps: not measured here.
- The overlap between smoltalk2 `smol_magpie_ultra` and smol-smoltalk `smol-magpie-ultra-short`.
- The licences of the everyday-conversations, systemchats and explore-instruct sources.
- What changed in `nvidia/Nemotron-CC-v2` on 2026-07-07.
- The 3B-token mixture of L20-Edu-135M (only the abstract was read).
- Every token total marked SCALED was taken from one file or three row groups.

## Sources

All read 2026-10-03 (UTC).
- Hub API and cards: https://huggingface.co/datasets/openbmb/Ultra-FineWeb ·
  https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1 · https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3 ·
  https://huggingface.co/datasets/openbmb/UltraX-Preview · https://huggingface.co/datasets/openbmb/UltraData-Code ·
  https://huggingface.co/datasets/openbmb/UltraData-SFT-2605 (gated) · https://huggingface.co/openbmb/MiniCPM5-2B ·
  https://huggingface.co/openbmb/MiniCPM5-2B-Base · https://huggingface.co/openbmb/MiniCPM5-1B ·
  https://huggingface.co/datasets/HuggingFaceFW/finephrase · https://huggingface.co/datasets/PleIAs/SYNTH ·
  https://huggingface.co/datasets/zgcagi/ZGCM-1-Data · https://huggingface.co/zgcagi/ZGCM-1-7B ·
  https://huggingface.co/datasets/HuggingFaceTB/smoltalk2 · https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk ·
  https://huggingface.co/datasets/HuggingFaceTB/smoltalk · https://huggingface.co/HuggingFaceTB/nanowhale-100m ·
  https://huggingface.co/HuggingFaceTB/nanowhale-100m-base · https://huggingface.co/datasets/IFM/SFT-Reasoning ·
  https://huggingface.co/datasets/swiss-ai/Apertus-v1.5-SFT-mix ·
  https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following ·
  https://huggingface.co/datasets/sedthh/gutenberg_english · https://huggingface.co/datasets/common-pile/project_gutenberg_filtered ·
  https://huggingface.co/datasets/common-pile/pre_1929_books_filtered · https://huggingface.co/datasets/LisaMegaWatts/philosophy-corpus ·
  https://huggingface.co/datasets/CristianPelayo/openalex-philosophy · https://huggingface.co/datasets/nvidia/Nemotron-CC-v2
- Hub listings: `/api/datasets?sort=trendingScore`, `/api/datasets?author=<org>&sort=createdAt`, `/api/datasets?search=<q>`;
  `datasets-server.huggingface.co/size` and `/first-rows`
- The Synthetic Data Playbook (FinePhrase), 2026-03-08: https://huggingface.co/spaces/HuggingFaceFW/finephrase ·
  https://huggingfacefw-finephrase.hf.space/
- UltraX, arXiv 2607.08646 (v1 2026-07-09): https://arxiv.org/abs/2607.08646
- Ultra-FineWeb, arXiv 2505.05427; UltraData tiers, arXiv 2602.09003 (cited by the cards, not opened)
- ZGCM-1, arXiv 2609.13356 (v1 2026-09-11): https://arxiv.org/abs/2609.13356
- Sedova, Seto, Schluter, Ablin, Scaling Laws for Mixture Pretraining Under Data Constraints, arXiv 2605.12715
  (v2 2026-05-15): https://arxiv.org/abs/2605.12715
- L20-Edu-135M, arXiv 2606.22189 (v1 2026-06-20): https://arxiv.org/abs/2606.22189
- Ding et al., Fewer Truncations Improve Language Modeling, arXiv 2404.10830: https://arxiv.org/abs/2404.10830
- nanochat at HEAD `92d63d4e`: `nanochat/dataset.py`, `nanochat/dataloader.py`, `nanochat/tokenizer.py`,
  `scripts/chat_sft.py`, `tasks/smoltalk.py`: https://github.com/karpathy/nanochat
- Project Gutenberg catalogue: https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv (last modified
  2026-09-27); books 1232, 1497, 3600, 4280, 4705, 5827, 8438, 34901 from https://www.gutenberg.org/cache/epub/
- Local: `track4_sdmllm_data_provenance.json`, `track4_sdmllm24_train_big_provenance.json`,
  `track4_sdmllm24_chat_provenance.json`, `track4_sdmllm24_prepare_smoltalk_chat_shards.py`,
  `track4_sdmllm_prepare_fineweb_edu_token_shards.py`, `RESEARCH_BASEDATA_2026-10-04.md`,
  `RESEARCH_OPENDATA_MODEL_2026-10-01.md`, `SDMONLY_CHANNEL.md`
