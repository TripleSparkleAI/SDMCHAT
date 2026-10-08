# DATA - what SDMCHAT holds back, and how to get it back

The repository carries code, configurations and records. The weights and the token data are too large for git and
are held back for Hugging Face. Nothing here has been uploaded yet; where each file lives today is stated below.

[DATA_FILES.tsv](DATA_FILES.tsv) lists every held-back file that sits on the research laptop (the M5): its family,
its path, its size in bytes, and its sha256. It is written by `SETTLE/tools/sdmchat_data_manifest.sh` in the research
repository, which hashes each file and refuses the two classes listed under "Never uploaded".

## On the research laptop, hashed (DATA_FILES.tsv)

Measured 2026-10-08. Paths are relative to the research repository, which is also the layout of this one.

| family | files | size | where | what it is |
|---|---|---|---|---|
| `vast-weights` | 5 | 28.07 GB | `experiments/track4/sdmllm/runs_vast_sdmonly/*/ck/`, `*/slim/` | the 2.0B-token runs trained on rented Vast boxes: three `true_cn_*` checkpoints (widths 1,024, 1,536, 2,048), `true_sdm_d1536_hm6144_hops16_2B` and its bf16 slim copy |
| `checkpoints` | 106 | 25.06 GB | `experiments/track4/sdmllm/checkpoints/` | every earlier run's `last.pt` (the store-model sweeps, the 300M and 600M chat models, the epic-poem models, the Vast store models), with `test_per_window.npz` scores beside many. Three files end `.partial`: incomplete copies of `model_bf16.pt` for `vast_VBCC4_d768_chat_400M`, `vast_VC_d1024_s0_2500M` and `vast_VS_sdmread_d512_s0_3B` |
| `browser-exports` | 25 | 1.13 GB | `SETTLE/settle-site/public/data/sdmchat/` | the int8 exports the website's chat runs (12 models, each a `.bin` and a `.json`) and the website's copy of the DeepSeek V4 tokenizer |
| `token-shards` | 4 | 0.27 GB | `experiments/track4/sdmllm/data/` | the first FineWeb-Edu shards in the DeepSeek tokenizer: `train.u32` (65,522,263 tokens), `val.u32` (2,000,916 tokens), `token_bytes.i32`, `prepare.log` |
| `tokenizer-copies` | 4 | 0.02 GB | `experiments/track4/sdmllm/data/` | the DeepSeek V4 Flash tokenizer: our export from the model file's header, and the publisher's files for the preview and the 0731 release |
| `vast-arrays` | 383 | 0.02 GB | `experiments/track4/sdmllm/runs_vast_sdmonly/` | per-window TEST scores (`.npz`) of the Vast runs |

Total: 527 files, about 54.6 GB.

## On the Spark only, not hashed from here

These are named in the records but live on the DGX Spark, so DATA_FILES.tsv cannot hash them. Each row gives the
record that names it. Sizes are not stated where no record states them.

| what | where on the Spark | named in |
|---|---|---|
| the FULL base's weights, `last.pt` and `keep_550000000.pt` | `ck/launch_base_fullsdm_d768_L12_T2048_1100M/` | push log, 2026-10-08T09:22:18Z |
| the 2.0B hop base, `step_15258.pt` (sha256 begins `ad16d6c8`, ends `7cb`; the log records only those) | `~/sdmonly_base/true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B/` | push log, 2026-10-06T17:31:17Z |
| SDM CHAT's weights, `sdm_chat_final_base2B_T2048_50M` | the Spark's checkpoint folder | push log, 2026-10-07T03:22:14Z |
| the 32,768-piece shards: `train_big`, `train_big_p200m`, `val`, `token_bytes.i32` | `~/sdmonly_base/data32k/` | every `runs_launch/*.result.json`, field `vocab.from` |
| the 3.0B-token FineWeb-Edu shard in the DeepSeek tokenizer, `train_big.u32` (3,018,298,789 tokens, sha256 `624b04a9...`) | the Spark's data folder | `experiments/track4/sdmllm/track4_sdmllm24_train_big_provenance.json` |
| the chat shards: `chat_train.u32` (378,859,013 tokens, sha256 `b280437d...`), `chat_test.u32` (19,809,072 tokens, sha256 `f88ec2c2...`), `chat_mix.u32` (757,718,026 tokens, sha256 `1a2e1724...`) | the Spark's data folder | `experiments/track4/sdmllm/track4_sdmllm24_chat_provenance.json` |

The full sha256 of each shard is in the provenance file named in its row.

## How to rebuild the data

Every shard is rebuilt from public sources at a pinned revision, and each script checks its output against the
recorded sha256.

1. **The DeepSeek V4 tokenizer.** `track4_sdmllm_tokenizer_from_gguf_header.py` reads it from the model file and
   checks it against the publisher's `tokenizer.json` (`deepseek-ai/DeepSeek-V4-Flash`, revision `60d8d707...`, file
   sha256 `8f9f37ca...`). Record: `track4_sdmllm_tokenizer_provenance.json`.
2. **The first FineWeb-Edu shards.** `track4_sdmllm_prepare_fineweb_edu_token_shards.py` reads
   `HuggingFaceFW/fineweb-edu`, `sample/10BT`, revision `87f09149...`. Record: `track4_sdmllm_data_provenance.json`
   (train sha256 `c2f7bc05...`, val sha256 `89fbba2c...`).
3. **The 3.0B-token shard.** `track4_sdmllm24_extend_fineweb_edu_shards.py` first proves step 2 reproduces
   byte-identically, then appends five more files of the same sample. Record:
   `track4_sdmllm24_train_big_provenance.json`.
4. **The chat shards.** `track4_sdmllm24_prepare_smoltalk_chat_shards.py` reads `HuggingFaceTB/smol-smoltalk`,
   revision `f73fe857...`. Record: `track4_sdmllm24_chat_provenance.json`.
5. **The 32,768-piece shards.** `track4_sdmonly_retoken32k.py --src <shard>.u32 --out data32k/<shard>.u32` re-encodes
   any shard into `tokenizers/trained_bpe_32768.json` losslessly, keeping document boundaries.

All five scripts are in `experiments/track4/sdmllm/`. The weights are rebuilt only by training again (see the
README); there is no other source for them.

## Licences of the sources

| source | licence | what follows for an upload |
|---|---|---|
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`) | ODC-By 1.0 | token shards may be shared with attribution to the dataset |
| smol-smoltalk (`HuggingFaceTB/smol-smoltalk`) | Apache-2.0 | chat shards may be shared with the licence and attribution |
| DeepSeek V4 Flash tokenizer (`deepseek-ai/DeepSeek-V4-Flash`) | MIT, per the publisher's model card | the tokenizer copies may be shared with the MIT notice; check the card at the pinned revision before upload |
| the epic poems (`experiments/track4/epics/`) | public domain translations (Butler, Pope, Dryden, Gummere, Milton and others) | free to share |
| our own trained BPE tokenizers and all weights | ours | the navigator's licence decision applies (LICENCE.md) |

## Never uploaded

These are refused by `sdmchat_data_manifest.sh` and are not in this repository:

- `experiments/track4/sdmllm/data/howl-raw.txt`: a copyrighted poem, not ours to redistribute.
- `experiments/track4/sdmllm/private/` (about 327 MB): the Weird Little Guy's corpora, built from lecture
  transcripts, papers and encyclopedia entries whose licences do not allow redistribution (all rights reserved, or
  terms that forbid copying in bulk).
