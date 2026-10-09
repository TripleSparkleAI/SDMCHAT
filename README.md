<!-- settle-banner -->
```text
 ·● ●●  ● ●  ●● ● ●  ●  ●●●
·   ● ● ●●● ●   ● ● ● ●  ●
 ● ·● ● ●●● ●   ●●● ●●●  ●
· ● ● ● ● ● ●   ● ● ● ●  ●
●···●●· ● ●  ●● ● ● ● ●  ●
↑↓↓↑↓↑↑↑↑↑ ●●●●●●●●
✦ a language model built from sparse distributed memory instead of attention
```

# SDMCHAT

SDMCHAT is a language model built from sparse distributed memory (SDM) instead of attention. It has no
transformer block. Every step that mixes information between positions in the text is a Kanerva memory: the model
writes into it while it reads, and reads back what earlier tokens wrote. The model is trained from scratch, with no
teacher, on FineWeb-Edu web text, and then tuned on chat.

This repository holds the training and inference code, the configurations, every run record and result, the
research notes, and the story of how the model was built. The trained weights and the token data are not here; they
are listed in [DATA.md](DATA.md) and wait for Hugging Face.

> **Licence: not chosen yet.** See [LICENCE.md](LICENCE.md).

## The two shapes

Both shapes are a stack of residual layers over a token table, and both read their next-token scores from the same
table (tied). They differ in what sits after the memory in each layer.

```
   token ──▶ token table ──▶ x
                             │
   every layer:              ├─▶ x + RunTimeSdm(x)      written and read as the text goes by
                             └─▶ x + Body(norm(x))      FULL: a trained SDM table
                                                        PARTIAL: a SwiGLU MLP
                             │
   next-token scores ◀── the same token table ◀── norm(x)
```

- **FULL** (code arm `onesdm_allsdm`): every layer is a run-time SDM, then a trained SDM table where an MLP would be.
  No MLP and no attention anywhere.
- **PARTIAL** (code arm `onesdm`): every layer is a run-time SDM, then a SwiGLU MLP.

The run-time memory keeps a fixed number of hard locations per head (1,024 at the defaults). A token picks its
locations by product keys, writes its value there with a learned gate, and each head forgets at its own rate. The
cost per token does not grow with the length of the text. The full description, with every flag, is
[experiments/track4/sdmllm/ONESDM_README.md](experiments/track4/sdmllm/ONESDM_README.md); the code is
[experiments/track4/sdmllm/track4_onesdm_models.py](experiments/track4/sdmllm/track4_onesdm_models.py).

Two older shapes are in the records too, because the chat models on the SETTLE website are built from them:

- **the store model** (arm `sdm`, `track4_sdmllm_models.py`): a trained store of 3,600 locations read twice per
  token, addressed by the last 8 tokens and five moving averages, then one SwiGLU readout. Nothing is written at run
  time.
- **the hop model** (arm `sdmonly`, `track4_sdmonly_models.py`): a trained store read in several hops, the shape of
  the 2.0B-token base and of the first SDM CHAT.

## The scores

Every score is TEST bits per byte on held-out FineWeb-Edu text. Lower is better. The two tables use different test
text and different tokenizers, so compare rows inside one table only.

**The bake-off and the FULL base** (width 768, 12 layers, window 2,048, the trained 32,768-piece tokenizer, the same
recipe for every row; TEST on 1,031,158 tokens; trained on the DGX Spark, an NVIDIA GB10, driver 580.159.03):

| run | shape | tokens | TEST bpb | record |
|---|---|---|---|---|
| `bk_yard_d768_L12_T2048_50M` | a 12-layer transformer, the yardstick | 50M | 1.22265 | [result](experiments/track4/sdmllm/runs_launch/bk_yard_d768_L12_T2048_50M.result.json) |
| `bk_onesdm_d768_L12_T2048_50M` | PARTIAL | 50M | 1.46405 | [result](experiments/track4/sdmllm/runs_launch/bk_onesdm_d768_L12_T2048_50M.result.json) |
| `bk_allsdm_d768_L12_T2048_50M` | FULL | 50M | 1.54457 | [result](experiments/track4/sdmllm/runs_launch/bk_allsdm_d768_L12_T2048_50M.result.json) |
| `bk_plain_d768_L12_T2048_50M` | the same stack with the memory off | 50M | 1.75751 | [result](experiments/track4/sdmllm/runs_launch/bk_plain_d768_L12_T2048_50M.result.json) |
| `launch_base_fullsdm_d768_L12_T2048_1100M` | FULL | 1.1B | 1.29216 | [result](experiments/track4/sdmllm/runs_launch/launch_base_fullsdm_d768_L12_T2048_1100M.result.json) |

What the table says: at 50M tokens the transformer is far ahead of both SDM shapes, and the run-time memory buys
0.21 to 0.29 of the 0.53 gap between memory off and the transformer. Trained on 22 times the tokens, FULL improves by
0.25241 and still trails the 50M-token transformer. A 1.1B-token transformer is the comparison the record owes next.

**The hop model at 2.0B tokens, and the first SDM CHAT** (width 1,536, 16 hops, window 2,048, the DeepSeek V4
tokenizer; TEST on 998,390 tokens). These rows come from the push log; their result files are not in this
repository yet.

| run | what | TEST bpb | source |
|---|---|---|---|
| `true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B` | the base, memory on | 0.99988 | push log, 2026-10-06T18:20:31Z |
| its twin | the same, memory off | 0.99734 | push log, 2026-10-06T18:20:31Z |
| `sdm_chat_final_base2B_T2048_50M` | SDM CHAT, the base tuned on 50M chat tokens | 1.0091 web, 0.78351 chat | push log, 2026-10-07T03:22:14Z |

The memory and its twin differ by 0.00254, inside the 0.007 noise between seeds, so at 2.0B tokens the memory ties.
The chat tune lowers the chat test from 1.19442 to 0.78351 and costs 0.00922 on web text. The push log is
[experiments/track4/sdmllm/PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md](experiments/track4/sdmllm/PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md).

## How to train it

Training needs a CUDA GPU and the token shards, which are held back (see [DATA.md](DATA.md) for how to rebuild
them). The FULL base above was trained with this command, read back from its result record:

```bash
cd experiments/track4/sdmllm
export SDMLLM_DATA=<folder with train_big.u32, val.u32 and token_bytes.i32 in the 32k tokenizer>
export SDMLLM_CKPT=<checkpoint folder>  SDMLLM_RUNS=<results folder>
python3 track4_sdmonly_train.py --run launch_base_fullsdm_d768_L12_T2048_1100M --arm onesdm_allsdm \
  --d 768 --layers 12 --T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd \
  --compile-body --loss chunked --tokens 1100000000 --train-name train_big --keep-at 550000000
```

For PARTIAL, use `--arm onesdm`. Every option is in `python3 track4_sdmonly_train.py --help`. The bake-off's seven
follow-up shapes are in [runs_launch/sweep.sh](experiments/track4/sdmllm/runs_launch/sweep.sh), and how we train in
general is [TRAINING_SDMONLY_HOW_WE_TRAIN.md](experiments/track4/sdmllm/TRAINING_SDMONLY_HOW_WE_TRAIN.md).

## How to run it

`track4_sdmonly_generate.py` loads a checkpoint of the store or hop model and generates text: one prompt, one chat
reply, or a chat in the terminal, with the website's own sampler ported to Python.

```bash
cd experiments/track4/sdmllm
python3 track4_sdmonly_generate.py --ckpt <checkpoint>/last.pt --tokenizer <ds4_v4flash_tokenizer.json> --chat "What is a river?"
python3 track4_sdmonly_generate.py --ckpt <checkpoint>/last.pt --tokenizer <ds4_v4flash_tokenizer.json> --repl
```

Those models use the DeepSeek V4 tokenizer, which is held back with the data (DATA.md, family
`tokenizer-copies`). **The FULL and PARTIAL shapes have no generation tool yet**: the training script scores them
and writes their checkpoints, and a text generator for them is owed. Their tokenizer,
[tokenizers/trained_bpe_32768.json](experiments/track4/sdmllm/tokenizers/trained_bpe_32768.json), is in this
repository.

The chat on the SETTLE website runs int8 exports of the store models in the browser. That engine is in the
settle-site repository (`src/engine/sdmchat.js`); it runs only the store shape, so FULL needs a port before it can
appear there. Its measurements are under [SETTLE/runs/](SETTLE/runs/).

## Checks you can run without a GPU

These need Python 3 with PyTorch and NumPy, and nothing from DATA.md:

```bash
cd experiments/track4/sdmllm
python3 track4_onesdm_models.py --selftest     # the FULL and PARTIAL models
python3 track4_onesdm_puzzles.py --selftest    # the memory puzzles, with their positive controls
python3 track4_sdmonly_models.py --selftest    # the hop model
python3 track4_sdmonly_optim.py --selftest     # the optimisers
```

## What is where

The repository keeps the research tree's paths, so every path a run record or a note names opens here unchanged.

| path | what it is |
|---|---|
| `experiments/track4/sdmllm/` | the model: code, configs, run records (`runs*/`), notes, the push log, the paper draft |
| `experiments/track4/sdmllm/runs_launch/` | the bake-off and the FULL base |
| `experiments/track4/sdmllm/tokenizers/` | the three trained byte-level BPE tokenizers, with their provenance |
| `experiments/track4/epics/` | the epic-poem corpus and the recite experiments; two of the website's chat models were trained here |
| `SETTLE/runs/sdmchats/`, `sdmscore/`, `sdmnext/` | measurements of the chat models running in the browser |
| `DATA.md`, `DATA_FILES.tsv` | what is held back for Hugging Face: names, sizes, sha256, how to rebuild |

The Weird Little Guy, a character fine-tune of SDM CHAT, is not in this repository: its corpus is built from private
research material that may not be redistributed.

## Where it comes from

This repository is exported from the SETTLE research repository by `SETTLE/tools/export_settle_repos.sh`. Each export
commit names the research commit it was copied from. The source of truth stays in the research repository; change
things there and export again.
