# D_q17 GitHub REST search: nanochat / modded-nanogpt data issues and PRs
- tool: python3 urllib, GitHub REST /search/issues, unauthenticated
- fetched: 2026-10-03 10:51:06 UTC

## query: `repo:karpathy/nanochat dataset created:>=2026-06-01`
- url: https://api.github.com/search/issues?q=repo%3Akarpathy%2Fnanochat+dataset+created%3A%3E%3D2026-06-01&sort=created&order=desc&per_page=30
- total_count: 10
- [2026-09-17] PR #857 Fix LR divergence across ranks in SFT - https://github.com/karpathy/nanochat/pull/857 (comments 0)
  snippet: Maintainer note: This is already fixed in `experiment_refactor` branch, commit `07620a2`, because SFT moved to prepacked dataset and step-based LR schedule.  In epoch-based SFT, LR can diverge between ranks. LR is calc
- [2026-09-13] PR #855 Add numerical KV cache equivalence test + CPU CI smoke workflow - https://github.com/karpathy/nanochat/pull/855 (comments 0)
  snippet: `tests/test_engine.py` covers the `KVCache` data structure and sampling behaviour using a `MockModel` with uniform logits. It does not verify that the cached decoding path produces the same logits as a full forward pas
- [2026-09-08] PR #852 Added functionality to pass desired dataset size in MB as an argument using -s - https://github.com/karpathy/nanochat/pull/852 (comments 0)
  snippet: For quick tests or small projects, you can use -s to download specific megabytes of the dataset. This enables downloading a limited dataset where size restrictions apply saving the cache storage, such as university proje
- [2026-07-31] PR #817 SFT: Fix NaN caused by conversations trimmed to hard-coded 2048 - https://github.com/karpathy/nanochat/pull/817 (comments 1)
  snippet: Long conversations are currently always trimmed to `max_tokens=2048`, regardless of the `--max-seq-len` parameter.  Relevant chunk in `sft_data_generator_bos_bestfit()`:  ```python ... row_capacity = args.max_seq_l
- [2026-07-21] PR #811 Fix SFT dataloader buffer clog that leads to NaN loss - https://github.com/karpathy/nanochat/pull/811 (comments 1)
  snippet: ## Symptom  Running `chat_sft` on a model pretrained with a short `sequence_len` (e.g. 1024), the loss goes to NaN right after the first optimizer step and stays NaN, while throughput looks healthy:  ``` Step 00000 | Val
- [2026-07-04] PR #800 [WIP] The experiment refactor: nanochat as a miniseries research harness - https://github.com/karpathy/nanochat/pull/800 (comments 5)
  snippet: **Work in progress, changing rapidly. Not ready for review; the code on this branch moves around daily.**  ## Motivation  nanochat is repositioning from "train your own ChatGPT clone for $100" to **the reference pretrain
- [2026-06-23] PR #785 Add MMLU-Pro Evaluation for Improved Reasoning Measurement - https://github.com/karpathy/nanochat/pull/785 (comments 2)
  snippet: Adds MMLU-Pro evaluation support.  This PR integrates MMLU-Pro into the evaluation pipeline, enabling more comprehensive measurement of model reasoning capabilities and providing an additional benchmark alongside exist
- [2026-06-12] PR #775 chat_sft: fix --num-iterations to count optimizer steps, not micro-batches - https://github.com/karpathy/nanochat/pull/775 (comments 3)
  snippet: ## Bug  `--num-iterations` is documented as the number of **optimization steps**:  ```python parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)") ```  But t
- [2026-06-07] PR #764 Fix stale fwe-train/fwe-val labels after ClimbMix migration - https://github.com/karpathy/nanochat/pull/764 (comments 1)
  snippet: `scripts/tok_eval.py` labels its two real-data evaluation samples as `fwe-train` and `fwe-val` (`fwe` = FineWeb-Edu). However, after the FineWeb-Edu-100B → ClimbMix-400B migration, these samples are actually read from Cl
- [2026-06-07] issue #763 Stale fwe-train / fwe-val labels in scripts/tok_eval.py after the FineWeb-Edu → ClimbMix dataset switch - https://github.com/karpathy/nanochat/issues/763 (comments 2)
  snippet: ### Summary `scripts/tok_eval.py` labels its two real-data evaluation samples `fwe-train` and `fwe-val` (`fwe` = FineWeb-Edu), but those samples are now read from the **ClimbMix** dataset. The labels were not updated whe

## query: `repo:karpathy/nanochat pretraining data created:>=2026-06-01`
- url: https://api.github.com/search/issues?q=repo%3Akarpathy%2Fnanochat+pretraining+data+created%3A%3E%3D2026-06-01&sort=created&order=desc&per_page=30
- total_count: 6
- [2026-08-30] PR #843 optim: keep async collective inputs and Work handles alive until wait() - https://github.com/karpathy/nanochat/pull/843 (comments 1)
  snippet: I hit this while pretraining a 1.5B Persian model on nanochat (8×H100, bf16), so the Muon path has had a hard workout. Sharing a latent lifetime bug I found in it.  `MuonAdamW.step()` overlaps communication with comput
- [2026-08-16] PR #832 overlap dataloader with gpu compute via background thread prefetch - https://github.com/karpathy/nanochat/pull/832 (comments 0)
  snippet: The pretraining dataloader (**tokenizing_distributed_data_loader_with_state_bos_bestfit**) was a synchronous generator: every next() call performed parquet reads, tiktoken tokenization, and the best-fit row packing on th
- [2026-08-15] PR #830 Speed up the 8xH100 GPT-2 run to 81.8 minutes - https://github.com/karpathy/nanochat/pull/830 (comments 2)
  snippet: ## Result  This is my next speedrun attempt, following the d22 + muon attempt in [nanochat #733](https://github.com/karpathy/nanochat/pull/733). It comes with a few speedups. Its main tradeoff is the new `liger-kernel`
- [2026-07-04] PR #800 [WIP] The experiment refactor: nanochat as a miniseries research harness - https://github.com/karpathy/nanochat/pull/800 (comments 5)
  snippet: **Work in progress, changing rapidly. Not ready for review; the code on this branch moves around daily.**  ## Motivation  nanochat is repositioning from "train your own ChatGPT clone for $100" to **the reference pretrain
- [2026-06-12] PR #775 chat_sft: fix --num-iterations to count optimizer steps, not micro-batches - https://github.com/karpathy/nanochat/pull/775 (comments 3)
  snippet: ## Bug  `--num-iterations` is documented as the number of **optimization steps**:  ```python parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)") ```  But t
- [2026-06-07] issue #763 Stale fwe-train / fwe-val labels in scripts/tok_eval.py after the FineWeb-Edu → ClimbMix dataset switch - https://github.com/karpathy/nanochat/issues/763 (comments 2)
  snippet: ### Summary `scripts/tok_eval.py` labels its two real-data evaluation samples `fwe-train` and `fwe-val` (`fwe` = FineWeb-Edu), but those samples are now read from the **ClimbMix** dataset. The labels were not updated whe

## query: `repo:karpathy/nanochat SFT created:>=2026-06-01`
- url: https://api.github.com/search/issues?q=repo%3Akarpathy%2Fnanochat+SFT+created%3A%3E%3D2026-06-01&sort=created&order=desc&per_page=30
- total_count: 24
- [2026-09-26] issue #860 Dropping the identity/spelling SFT tasks costs ~8 points of GSM8K - https://github.com/karpathy/nanochat/issues/860 (comments 1)
  snippet: `950a1dc` removed `tasks/customjson.py` and `tasks/spellingbee.py`, which also shrinks the SFT mixture in `scripts/chat_sft.py` from 1,071,759 to 789,759 rows. On 8xH100 that costs most of the model's GSM8K ability.  One
- [2026-09-17] PR #857 Fix LR divergence across ranks in SFT - https://github.com/karpathy/nanochat/pull/857 (comments 0)
  snippet: Maintainer note: This is already fixed in `experiment_refactor` branch, commit `07620a2`, because SFT moved to prepacked dataset and step-based LR schedule.  In epoch-based SFT, LR can diverge between ranks. LR is calc
- [2026-09-03] issue #848 SFT inherits pre-batch-scaling learning rates from base checkpoint - https://github.com/karpathy/nanochat/issues/848 (comments 1)
  snippet: I noticed a possible issue with how learning rates are passed from base training to SFT.  In base_train.py, the learning rates are scaled using batch_lr_scale before being passed to model.setup_optimizer():  unembedding_
- [2026-08-28] PR #841 Remove unused TaskSequence class - https://github.com/karpathy/nanochat/pull/841 (comments 0)
  snippet: Closes #840.  Removes the unused `TaskSequence` class from `tasks/common.py`.  `TaskSequence` has never been referenced since the initial commit. Its only import (`scripts/chat_sft.py`) was already removed in 190d951
- [2026-08-28] issue #840 TaskSequence in tasks/common.py is unused - https://github.com/karpathy/nanochat/issues/840 (comments 0)
  snippet: `tasks/common.py` defines a `TaskSequence(Task)` class (curriculum-style sequential SFT), but nothing in the repo references it:  - No callers in `scripts/` — `chat_sft.py` constructs `TaskMixture` only. - Not imported b
- [2026-08-28] PR #839 Report a real peak memory number on CPU/MPS instead of 0.00MiB - https://github.com/karpathy/nanochat/pull/839 (comments 0)
  snippet: Fixes #838.  **CUDA behavior is unchanged by construction.** `update()` is a no-op there and `peak()` returns `torch.cuda.max_memory_allocated()`, exactly what `get_max_memory` returned before. Nothing about an 8xH100 
- [2026-08-28] issue #838 "Peak memory usage" reports 0.00MiB on CPU and MPS - https://github.com/karpathy/nanochat/issues/838 (comments 0)
  snippet: `base_train.py` and `chat_sft.py` both bind:  ```python get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0 ```  and print it at the end of a run:  ```python print0(f"Peak memory usag
- [2026-08-19] PR #834 fix(sft): pass max_tokens to render_conversation so conversations fit - https://github.com/karpathy/nanochat/pull/834 (comments 1)
  snippet: I was experimenting with nanochat using the small model for quick turn around  when I ran into this issue  ### Symptom:  SFT always collapses with `loss: nan` on step 4. I've tried this on both AMD GPU and CPU, and b
- [2026-08-16] PR #831 Fix wandb api key length issue by upgrade wandb to 0.28.2 - https://github.com/karpathy/nanochat/pull/831 (comments 0)
  snippet: The current wandb version produces below error with latest wandb api key. Upgrading the wandb version to the latest one solves this problem.  ``` Traceback (most recent call last):   File "/Users/rulintang/.local/sha
- [2026-08-15] PR #830 Speed up the 8xH100 GPT-2 run to 81.8 minutes - https://github.com/karpathy/nanochat/pull/830 (comments 2)
  snippet: ## Result  This is my next speedrun attempt, following the d22 + muon attempt in [nanochat #733](https://github.com/karpathy/nanochat/pull/733). It comes with a few speedups. Its main tradeoff is the new `liger-kernel`
- [2026-08-15] PR #829 Speed up GPU/ROCm inference with fused QKV, last-token logits, and torch.compile - https://github.com/karpathy/nanochat/pull/829 (comments 0)
  snippet: ## Summary - Skip the full-vocab `lm_head` on prompt prefixes during generation (last-token logits only); CORE / categorical eval still get full-sequence logits. - Pack Q/K/V into one GEMM for decode (`fuse_qkv`, off by 
- [2026-08-01] PR #818 SFT: Fix off-by-one EMA debias - https://github.com/karpathy/nanochat/pull/818 (comments 0)
  snippet: In train loop, `step += 1` is before EMA debias, causing EMA to have wrong exponents. In `base_train.py` step is incremented correctly after EMA debias.  ```python # State step += 1  # logging # On the first logge
- [2026-07-31] PR #817 SFT: Fix NaN caused by conversations trimmed to hard-coded 2048 - https://github.com/karpathy/nanochat/pull/817 (comments 1)
  snippet: Long conversations are currently always trimmed to `max_tokens=2048`, regardless of the `--max-seq-len` parameter.  Relevant chunk in `sft_data_generator_bos_bestfit()`:  ```python ... row_capacity = args.max_seq_l
- [2026-07-31] PR #816 SFT: Fix incorrect progress accounting and LR scaling. - https://github.com/karpathy/nanochat/pull/816 (comments 2)
  snippet: This PR fixes two tightly interleaved bugs in `chat_sft.py`. They both are related to tracking progress by modifying global variables inside the dataloader `sft_data_generator_bos_bestfit` combined with batch pre-fetch l
- [2026-07-30] PR #815 fix: count Float8Linear in num_matmul_params (--fp8 understates FLOPs/token and MFU ~12x) - https://github.com/karpathy/nanochat/pull/815 (comments 0)
  snippet: ## Problem  With `--fp8`, the reported `bf16_mfu` and `Estimated FLOPs per token` are about 12x too low.  On an 8xH100 d24 speedrun I saw `bf16_mfu: 3.87`, which would imply 4% hardware utilization at 771k tok/sec. 
- [2026-07-21] PR #811 Fix SFT dataloader buffer clog that leads to NaN loss - https://github.com/karpathy/nanochat/pull/811 (comments 1)
  snippet: ## Symptom  Running `chat_sft` on a model pretrained with a short `sequence_len` (e.g. 1024), the loss goes to NaN right after the first optimizer step and stays NaN, while throughput looks healthy:  ``` Step 00000 | Val
- [2026-07-13] PR #808 chat_rl: support resuming an interrupted RL run - https://github.com/karpathy/nanochat/pull/808 (comments 0)
  snippet: If an RL run dies mid-way (SSH drop, cluster walltime), its saved checkpoints can't be used to continue: chat_rl always initializes from SFT and always starts at step 0 / example 0.  This adds two flags, both defaulting 
- [2026-07-10] PR #803 Fix SFT packing NaNs on short contexts - https://github.com/karpathy/nanochat/pull/803 (comments 1)
  snippet: ## Summary  - make SFT best-fit packing truncate the shortest oversized conversation when an empty row cannot otherwise make progress - skip packed batches that have no supervised target tokens after the causal shift - a
- [2026-07-04] PR #800 [WIP] The experiment refactor: nanochat as a miniseries research harness - https://github.com/karpathy/nanochat/pull/800 (comments 5)
  snippet: **Work in progress, changing rapidly. Not ready for review; the code on this branch moves around daily.**  ## Motivation  nanochat is repositioning from "train your own ChatGPT clone for $100" to **the reference pretrain
- [2026-06-27] PR #792 Make device synchronization MPS-aware (fix crash + correct Mac timing) - https://github.com/karpathy/nanochat/pull/792 (comments 0)
  snippet: The training/inference scripts bound synchronize() as:     torch.cuda.synchronize if device_type == "cuda" else lambda: None which has two problems off CUDA:  1. nanochat/engine.py's __main__ self-test called torch.c
- [2026-06-24] PR #786 Match calculator tool output to training format (drop trailing .0) - https://github.com/karpathy/nanochat/pull/786 (comments 2)
  snippet: ## What  When the model calls the calculator tool, `Engine` injects the result back into the context between `<|output_start|>` / `<|output_end|>`. Those output tokens are unsupervised - the model only conditions its con
- [2026-06-20] PR #781 Codex/fix chat sft stability - https://github.com/karpathy/nanochat/pull/781 (comments 0)
  snippet: 
- [2026-06-12] PR #775 chat_sft: fix --num-iterations to count optimizer steps, not micro-batches - https://github.com/karpathy/nanochat/pull/775 (comments 3)
  snippet: ## Bug  `--num-iterations` is documented as the number of **optimization steps**:  ```python parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)") ```  But t
- [2026-06-08] PR #766 Fix/categorical eval dual pass - https://github.com/karpathy/nanochat/pull/766 (comments 0)
  snippet: ## Summary - **Bug**: `run_categorical_eval` called `model(prompt_ids)` directly, completely bypassing the engine. ARC, ARC-Challenge, and MMLU produced identical results at every guidance weight during `iv_eval` sweeps

## query: `repo:KellerJordan/modded-nanogpt data created:>=2026-06-01`
- url: https://api.github.com/search/issues?q=repo%3AKellerJordan%2Fmodded-nanogpt+data+created%3A%3E%3D2026-06-01&sort=created&order=desc&per_page=30
- total_count: 17
- [2026-10-02] PR #379 Track 1: CPLM on #360 (-4.57s, -11.25% same hardware) - https://github.com/KellerJordan/modded-nanogpt/pull/379 (comments 0)
  snippet: # New record: CPLM on record #92 — 36.009 s (0.600 min)  Paper coming soon! :)  GPT-2 (124M-class) on FineWeb10B, 8×H100-80GB, track 1 (≤ 3.28 val CE). Built on record #92 (ANVIL2, [PR #360](https://github.com/Keller
- [2026-10-02] PR #378 Track 1: Document-local copy feature (-1.17 s, -2.9% vs #360; stacked on #375) - https://github.com/KellerJordan/modded-nanogpt/pull/378 (comments 0)
  snippet: Each position gets a **candidate next token**: the token that followed the longest earlier exact match of its (normalized) context in the **same document**, fed to the model as an extra input during training. This combin
- [2026-09-29] PR #375 Track 1: Normalize tokens in n-gram embeddings to improve information density (-0.42s, -1.02%, -15 steps) - https://github.com/KellerJordan/modded-nanogpt/pull/375 (comments 0)
  snippet:  Normalize tokens in n-gram embeddings to improve information density (-0.42s, -1.02%, -15 steps)  Following Deepseek's [Engram](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf) paper, this PR normaliz
- [2026-09-05] PR #364 Endgame EMA weight blending (port of Track 3 Tail-EMA readout) - https://github.com/KellerJordan/modded-nanogpt/pull/364 (comments 0)
  snippet: Blends an EMA of the tail-of-training weights into the final weights at the last step:  `w <- (1 - gamma) * w_final + gamma * w_ema`  ### Prior art  This is the Tail-EMA eval readout from the Track 3 optimization benchma
- [2026-08-31] PR #360 New Record: 0.665 minutes (39.9 seconds): ANVIL2, Sampled-softmax, New Embedding Table, Full-stack fp8 (-34.0s, -46% same hardware) - https://github.com/KellerJordan/modded-nanogpt/pull/360 (comments 15)
  snippet: Created by: Deven Pietrzak (MIT Math. X: [DevenPzak](https://x.com/DevenPzak). Lab: [Hyperstition (fka Social Physics Lab)](https://x.com/hyperstition_cc))  After a few weeks since my last record without touching the P
- [2026-08-28] PR #359 Track3: K-maxwell momentum muonh decay - 3065 steps (n=8) - https://github.com/KellerJordan/modded-nanogpt/pull/359 (comments 0)
  snippet: # Track 3: K-Maxwell on MuonH fast-slow decay, 3065 steps (n=8)  Collaborator: [Jeffrey Cheng](https://github.com/jeffreyscheng) ([@jeffreyscheng](https://github.com/jeffreyscheng)).  [PR #351](https://github.com/Kel
- [2026-08-27] PR #358 Track 1: Norm CSE, FP8 up-proj quantize (-0.729s) - https://github.com/KellerJordan/modded-nanogpt/pull/358 (comments 1)
  snippet: ## Summary  This PR builds on the current Track 1 `master` record (#89, FP8 MLP down projection) and adds three changes to `train_gpt.py`, +37 / -5 lines:  **-0.729 s** mean, measured as a paired same-lease comparis
- [2026-08-27] PR #357 Track 3: K-Maxwell momentum 3160-step on a Muon tuned baseline result (n=8) - https://github.com/KellerJordan/modded-nanogpt/pull/357 (comments 1)
  snippet: # Track 3: annealed K-Maxwell momentum on tuned Muon — 3160 steps (n=8)  Collaborator: [Jeffrey Cheng](https://github.com/jeffreycider) ([@jeffreycider](https://github.com/jeffreycider)).  The tuned Muon + aux AdamW 
- [2026-08-10] PR #351 MuonH fast-slow-decay schedule: 3125 steps (n=20, sig=0.00450) - https://github.com/KellerJordan/modded-nanogpt/pull/351 (comments 0)
  snippet: **Per-optimizer SOTA for MuonH: 3125 steps, 3.2790 (n=20)**  **TL;DR.** This submission improves the MuonH per-optimizer SOTA from **3150 steps** (previous MuonH SOTA, PR #345, power-0.4 cooldown) to **3125 steps** by 
- [2026-08-06] PR #349 New record: 1.079 min (64.72s) — new ANVIL optimizer + full-stack fp8 and schedule overhaul (−9.3s / −12.6% same-hardware) - https://github.com/KellerJordan/modded-nanogpt/pull/349 (comments 13)
  snippet: Created by: Deven Pietrzak ([Social Physics Lab](https://socialphysics.inc), MIT Math)  Previous record: 1.23 min (#89).  This run finishes in **64.72s** — the mean of a 12-run, fully logged pool of the exact shipped
- [2026-07-28] PR #347 Track 1: terminal-blended readout TailEMA (1.261 min, -0.895%) - https://github.com/KellerJordan/modded-nanogpt/pull/347 (comments 1)
  snippet: ## Summary  - Maintain FP32 TailEMA shards for `lm_head` and the split embedding readout, then fuse a 65% terminal blend into the existing compiled Adam update and ordinary all-gather. - Initialize MLP `c_proj` at `0.5 *
- [2026-07-22] PR #345 Track 3: MuonH power-0.4 schedule reaches 3.28 at 3150 steps - https://github.com/KellerJordan/modded-nanogpt/pull/345 (comments 0)
  snippet: ## Summary  This improves the accepted MuonH per-optimizer result #37 from 3250 to 3150 steps.  - MuonH updates 0–99: linear warmup from `1e-5` to `0.03` - updates 100–199: hold at `0.03` - updates 200–3200: `floor + (pe
- [2026-07-18] PR #342 Track 1: Add FP8 MLP down projection (-1.04s) - https://github.com/KellerJordan/modded-nanogpt/pull/342 (comments 2)
  snippet: ## Summary  This PR builds on the current Track 1 `master` record and adds one systems-side change to `train_gpt.py` and `triton_kernels.py`:  **FP8 MLP down projection.** The MLP down-projection forward path now ru
- [2026-07-15] PR #340 Track 3: bi-Maxwell momentum on the tuned Muon baseline -- 3210 steps (n=8) - https://github.com/KellerJordan/modded-nanogpt/pull/340 (comments 0)
  snippet: # Track 3: bi-Maxwell momentum on the tuned Muon baseline, 3210 steps (n=8)  A per-optimizer result in the sense of the track README: the **tuned Muon + aux AdamW baseline** (result #36, 3250 steps) plus **one change to 
- [2026-07-13] PR #337 Track 1: Prefix token prediction (-0.78s, -15 steps) - https://github.com/KellerJordan/modded-nanogpt/pull/337 (comments 5)
  snippet: Adds another auxiliary target to CE loss, like MTP. However, instead of using a future token, it uses the "prefix token" which is defined here^ as the token corresponding to the longest byte prefix of the current token. 
- [2026-06-12] PR #326 Track 3: Sign-based entrywise baselines (Lion, signSGD) - https://github.com/KellerJordan/modded-nanogpt/pull/326 (comments 0)
  snippet: # Track 3: Sign-based entrywise optimizer baselines (Lion, signSGD)  ## Summary  I'm adding the first **sign-based** hidden-matrix optimizers to the track-3 comparison: **Lion** (sign with momentum) and **signSGD** (sign
- [2026-06-09] PR #320 Add hyperball_pema records archive (8 Parallax optimizer benchmarks) - https://github.com/KellerJordan/modded-nanogpt/pull/320 (comments 1)
  snippet: Self-contained archive under `parallax/hyperball_pema/` of the 8 benchmark records: the **4 hyperball optimizers** (MuonH, NorMuonH, SOAP-H, AuroraH) and the **same 4 + projected EMA-Nesterov**, under Parallax local-line

