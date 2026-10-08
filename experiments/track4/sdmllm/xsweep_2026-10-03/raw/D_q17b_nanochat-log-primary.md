# D_q17b primary: karpathy/nanochat dev/LOG.md (data sections)
- url: https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LOG.md
- tool: python3 urllib
- fetched: 2026-10-03 10:50:45 UTC, 63,218 bytes
- newest entry in file: 2026-05-05 (no entry in window 2026-08-01..2026-10-03)

## verbatim excerpts

## 2026-03-04: Dataset upgrade: FineWeb-EDU 100B → ClimbMix 400B

Switched the pretraining dataset from FineWeb-EDU 100B to ClimbMix 400B. This is by far the single biggest improvement to nanochat's GPT-2 speedrun time, bringing it down from **2 hours 46 minutes to 2 hours 1 minute** — a 27% reduction.

### What is ClimbMix?

ClimbMix 400B is a curated 400B-token pretraining mixture hosted at `karpathy/climbmix-400b-shuffle` on HuggingFace. It comes form [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix). It is a blend of high-quality web text, code, math, and other sources, designed to be a better general-purpose pretraining dataset than FineWeb-EDU alone.

### What changed

- **Dataset**: `karpathy/fineweb-edu-100b-shuffle` → `karpathy/climbmix-400b-shuffle` (up to 6543 shards available vs the previous 1823 data shards, allowing for longer training in the future)
- **Data directory**: `base_data/` → `base_data_climbmix/` (clean separation from legacy data)
- **Model depth**: d26 → d24. ClimbMix trains more efficiently, so a smaller model reaches GPT-2 capability
- **Shard count**: Only approx 150 data shards (~7B tokens) are now needed for GPT-2 capability
- **Eval tokens**: doubled from 40 to 80 batches for more stable validation loss estimates
- **Legacy fallback**: added a migration warning in `list_parquet_files()` that detects the old `base_data/` directory and falls back gracefully, so existing users see clear upgrade instructions on `git pull`

### Context

This is the sixth attempt at beating FineWeb-EDU on CORE score — the previous five all failed (see entries on 2026-02-17, 2026-02-10, 2026-01-12 below). ClimbMix is the first dataset to convincingly surpass it, and the margin is large enough to also shrink the model from d26 to d24.

---

## 2026-02-17: Pretraining Data: FineWeb (negative)

Tried vanilla fineweb instead of fineweb-edu dataset. Significantly, shockingly worse results:

- d26 (GPT-2): CORE 0.2602 → 0.2241

This is the fifth failed attempt to beat pure FineWeb-EDU on CORE score.

---

## 2026-02-17: Pretraining Data Mixture Experiment (negative)

Tried [hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT](https://huggingface.co/datasets/hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT), a mixture of FinePDFs, DCLM, and FineWeb-EDU. Slightly worse on both model sizes tested:

- d26 (GPT-2): CORE 0.2602 → 0.2549
- d18: CORE 0.199 → 0.192

This is the fourth failed attempt to beat pure FineWeb-EDU on CORE score.

---

## 2026-01-15: Olmo pretraining mix (Negative result)

I attempted to train on the Olmo 3 pretraining dataset [allenai/dolma3_mix-6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T) instead of FineWeb-edu. I ran into a number of [errors and issues](https://huggingface.co/datasets/allenai/dolma3_mix-6T/discussions/2) trying to both download and process the dataset and then noticed some quality issues (e.g. some documents seem to be extremely short, like "5".). I managed to work around these with some sensible hacks (e.g. reject documents less than 100 characters in length) and tried to process the dataset exactly as FineWeb, re-trained the tokenizer and trained a d16 model. The CORE score decreased from 15.5 to 13.8, i.e. the result is quite a bit worse.

I am still looking to try the [DCLM dataset](https://arxiv.org/abs/2406.11794), which according to the paper should be better that FineWeb-edu. I do have some concerns that the same group both prepared the DCLM dataset *and* introduced the CORE score so I'm a bit hesitant in case there was some overfitting to CORE score adjacent data distribution.

Classifying as negative result and reverting back to FineWeb-edu for now.

---
