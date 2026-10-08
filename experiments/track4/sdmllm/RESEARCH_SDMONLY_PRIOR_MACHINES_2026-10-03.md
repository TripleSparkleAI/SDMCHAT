# PRIORMACHINES · what we already learned about machines and CUDA, for the SDM-only training run · 2026-10-03

A research lane over our own record. Nothing was trained, rented or changed. The Spark was read with
`cat`, `ls` and `grep` only. Every number below carries one of three labels:

- **MEASURED (ours)**: read from our own logs or result files; the path is given.
- **MEASURED (elsewhere)**: published by someone else and read from their page, as recorded in our wikis.
- **UNVERIFIED**: a vendor figure, a claim nobody here has checked, or our own arithmetic on measured
  inputs (written "UNVERIFIED, arithmetic"). Arithmetic is not a measurement until a run confirms it.

Companion documents: `RESEARCH_BASEDATA_2026-10-04.md` (data, recipes, Vast prices, tokens per dollar) and
`RESEARCH_OPENDATA_MODEL_2026-10-01.md`. This file does not repeat their price tables.

## 0. The one measurement that shapes everything below

The SDM-only model was benched on the Spark on 2026-10-03, six short runs of 1.6M tokens each, B 32, T 256,
alone on the GPU (load 0.3 to 1.3, MemAvailable 106 to 113 GB). Median tokens a second over steps 60 to 189.
MEASURED (ours): `~/settle24/runs/sdmonly/bench_*.log` and `bench.out` on the Spark; the commands are in
`~/settle24/runs/sdmonly/bench.sh`.

| run | body | trainable params | forward FLOPs per token | tokens a second |
|---|---|---|---|---|
| `bench_d768_dense_h4` | 4 SwiGLU of equal FLOPs | 108,480,000 | 217,915,392 | **40,704** |
| `bench_d768_n256_h4` | 4 hops, own store each, 65,536 rows | 309,343,744 | 217,382,912 | **22,759** |
| `bench_d768_n512_h4_shared` | 4 hops, one shared store, 262,144 rows | 309,212,672 | 217,907,200 | **20,735** |
| `bench_d768_n512_h8_shared` | 8 hops, one shared store | 310,002,176 | 220,921,856 | **15,978** |
| `bench_d768_n512_h4` | 4 hops, own store each, 262,144 rows | 913,585,664 | 217,907,200 | **13,458** |
| `bench_d256_n256_h4` | d 256, 4 hops, 65,536 rows | 101,585,408 | 69,402,624 | **46,671** |

```
   d 768, the same forward FLOPs (217.4M to 217.9M per token), tokens a second on the GB10

   dense SwiGLU      108M params  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  40,704
   store, 65,536/hop 309M params  |@@@@@@@@@@@@@@@@@@@@@@@                   22,759
   one store, 262k   309M params  |@@@@@@@@@@@@@@@@@@@@@                     20,735
   store, 262k/hop   914M params  |@@@@@@@@@@@@@                             13,458

   the FLOPs are flat; the speed falls with the PARAMETER COUNT of the value tables
```

- At equal FLOPs, speed tracks the size of the value tables, not the compute. MEASURED (ours), the table above.
- Going from 309M to 914M trainable parameters (both 262,144-row variants) adds 0.2136 s to every step. That
  is 0.354 ns per extra parameter, or about 81 bytes moved per extra parameter per step at the GB10's measured
  230.5 GB/s ceiling. UNVERIFIED, arithmetic. It is consistent with a dense AdamW update, a dense gradient
  buffer and a gradient-norm clip touching every row of every table on every step, whether or not the row was
  read. The cause has not been profiled.
- So the cost that matters for the rented run is **per step**, not per token. A larger batch spreads it over
  more tokens; sparse row updates remove it. Both are untested on this model (section 3, section 5).

## 1. THE SPARK

### 1.1 What it is

| fact | value | label and source |
|---|---|---|
| GPU | NVIDIA GB10, sm_121, 48 SMs, 24 MB L2 | MEASURED (ours), `wikis/WIKI_HW_SPARK_GB10/00-VERIFIED-FACTS.md` |
| memory | 121.7 GiB usable, one pool shared by CPU and GPU | MEASURED (ours), `.../04-memory-subsystem.md` |
| bandwidth | 230.5 GB/s measured sequential ceiling; 273 GB/s is the datasheet, never reached | MEASURED (ours), `experiments/big25-round24/gb10_block_sweep2.cu` via `00-VERIFIED-FACTS.md` |
| bf16 matmul, warmed | 84.2 TFLOP/s on 4096 cubed; the first, cold iteration reads 8.2 | MEASURED (ours), `00-VERIFIED-FACTS.md` (ledger §66) |
| CPU | 10 Cortex-X925 + 10 Cortex-A725 | MEASURED (ours), `00-VERIFIED-FACTS.md` |
| stack | driver 580.159.03, CUDA 13.0, torch 2.14.1+cu130 in `~/settle24/venv` | MEASURED (ours), start stamps of every `runs_spark24/*.log` |
| power mode | none to read (Linux, no `pmset`); stamps record `powermode: unknown` | MEASURED (ours), the same stamps |

### 1.2 What it does for this model

Medians over steps after step 1,000, read from our logs. MEASURED (ours), `runs_spark24/`, `runs_sdmchats/`.

| run | arm | d | tokens a second | GPU |
|---|---|---|---|---|
| `sdmchats_NWL_d768_s0_600M` | no store | 768 | 38,241 | alone from 02:37Z |
| `sdmchats_SWL_d768_s0_600M` | SDM read, 3,600 locations | 768 | 31,381 | alone from about 13:00Z |
| `spark_WCO_d512_chatonly_100M` | per-token table | 512 | 50,396 | alone |
| `spark_U_s0_300M`, `spark_N_s0_300M` | d 256 arms | 256 | 20,192 and 20,102 | three arms sharing |
| `spark_C_d512_s0_1B` | per-token table | 512 | 16,140 | sharing |

- **The GPU is the limit, and sharing it divides it.** Three d 256 arms together ran at 61k tokens a second
  with the CPU idle and 59k under load 11. MEASURED (ours), `SETTLE/runs/spark24/LANE_SPARK24_STATUS.md`
  (01:45Z) and the 01:50Z amendment in `SETTLE/SETTLE_CAMPAIGN_2026-09-30.md`.
- **A bigger batch bought almost nothing on the plain loss path.** B 32 to B 64, d 256: `sdm_ngram` 27,470 to
  28,013, `sdm_nostore` 27,730 to 28,055, `qwen` 25,942 to 25,941. MEASURED (ours), Spark job logs
  `~/settle24/q/logs/1001010446-0002.log` and `-0003.log`. That is a head-dominated model with a small store;
  the SDM-only tables were not part of it.
- **Tokens a day, alone on the GPU:** SWL's shape 2.71B; SDM-only d 768 with 65,536 rows a hop 1.97B; with
  262,144 rows a hop 1.16B. UNVERIFIED, arithmetic on section 0 and the table above.
- **3B tokens of SDM-only d 768 on the Spark alone:** 36.6 h at 65,536 rows a hop, 61.9 h at 262,144.
  UNVERIFIED, arithmetic.

### 1.3 What it cannot do

- **It is one GPU at about a quarter to a sixth of a rented 5090 for our models.** Matched width d 768:
  NWL (no store) 38,241 on the Spark against VB (per-token table) 152,293 per 5090, a ratio of 3.98. Different
  arms, so the ratio is approximate. MEASURED (ours) inputs; ratio UNVERIFIED, arithmetic.
- **It cannot use inductor's max-autotune GEMM mode.** Every compiled run prints "Not enough SMs to use
  max_autotune_gemm mode". MEASURED (ours), `~/settle24/runs/sdmonly/bench.out` and
  `~/settle24/q/logs/1001010737-0008.log` on the Spark.
- **It has no FP8 path we have measured.** FP8 exists in the silicon (sm_121, 5th-generation tensor cores),
  but the fast kernels target sm_100 or Hopper. `wikis/WIKI_HW_SPARK_GB10/03-tensor-cores-and-precisions.md`
  and `08-kernel-support-gaps-sm121.md`. Untested on our model.
- **It cannot raise a clean out-of-memory error.** See the traps below.

### 1.4 Settings every Spark run must use

| setting | why | source |
|---|---|---|
| `--loss chunked` (compiled, checkpointed chunks of the tied head, `--ce-chunk 2048`) | 2.5 to 2.7 times the plain path and a 1.5 to 1.8 GB peak instead of 13.5 GB; same loss | section 3.1 |
| leave `--compile` off for the model; the chunked loss already compiles `hidden -> loss` | every SPARK24, SDMCHATS and SDMONLY run did this | `SETTLE/tools/spark24-job-queue/sdm_arm.sh`, `track4_sdmonly_spark_arm.sh` |
| store statistics off during training (`collect_stats = False`) | a Python `float()` inside the store's forward made dynamo hit `recompile_limit (8)` and fall back for that frame | Spark job logs `1001010737-0008`, `1001011000-0011`; `track4_sdmllm_train_one_arm.py` near line 194 |
| run through the queue with an honest `--mem` claim, under `nice` | the queue only starts a job that fits MemAvailable minus 8 GB | `SETTLE/tools/spark24-job-queue/README.md` |
| checkpoints every 1,000 to 2,000 steps, written to `.tmp` then renamed | the memory guard cancels a job and it resumes from the last checkpoint | SDMCHATS incident, 02:12Z, campaign file |
| a memory guard beside any long run that shares the box | MemAvailable fell to 11 and 14 GB beside a 76 GB process; the guard cut at step 4,000 and the resume reproduced the loss (5.4087 against 5.4086) | `SETTLE/runs/sdmchats/spark/memguard.py` (LOW 20 GB, HIGH 29 GB in the code) |
| read speed as the median after step 1,000 | the first logged line is low (compile; 10,323 against 46,671) and the last is inflated (63,131), because the last interval is shorter than `log_every` but is divided as if it were not | section 0 logs; `track4_sdmllm_train_one_arm.py` around line 227 |
| stamp load, MemAvailable and driver at start and end | every job log already does; add GPU power draw (the latch below) | `SETTLE/tools/spark24-job-queue/README.md` |

### 1.5 Known traps on the Spark

1. **A memory overrun freezes the box instead of raising an error.** A DGX Spark training run at batch 32
   used 94 GB of 128 GB, left 2.4 GB free and the "system froze rather than raising a clean CUDA OOM error".
   MEASURED (elsewhere), nanochat discussion 710, recorded in `wikis/WIKI_HW_SPARK_GB10/16-nanochat-speedrun-reproduction-on-a-spark.md`.
   Enforce a free-memory floor before the allocation; there is nothing to catch afterwards.
2. **`cudaMalloc` overcommits and `cudaMemGetInfo` reports MemFree.** A large allocation succeeds and the OOM
   killer arrives on first touch. At one reading CUDA reported 31.62 GiB free while MemAvailable was 59.68.
   MEASURED (ours), `WIKI/theory/177-the-sparkport-day-...-2026-09-13.md` §3. Size from `/proc/meminfo`
   MemAvailable, never from `torch.cuda.mem_get_info()`.
3. **A compiled graph is an allocation claimed late.** A recompile at the first evaluation, on top of a
   steady-state training loop, froze a Spark. MEASURED (elsewhere), page 16 above, Issue #14. Our trainer
   evaluates with the uncompiled model and plain full logits, so it does not recompile, but the evaluation
   itself is the next trap.
4. **Evaluation materialises full logits.** `eval_bpb` scores 32 windows at a time with `.float()` logits:
   32 x 256 x 129,280 x 4 bytes = 4.2 GB per batch, plus the bf16 copy. The trainer calls
   `torch.cuda.empty_cache()` afterwards to hand the blocks back. Code: `track4_sdmllm_train_one_arm.py`
   lines 104 to 128. On a 32 GB 5090 shared by two jobs, the plain path ran out of memory ("plain loss OOMed
   beside VCC, chunked used"). MEASURED (ours), `experiments/task-runner/JIMOTHY_VAST_SETTLE_2026-10-01.md`, 11:55Z.
5. **Processes are SIGSTOPped by an unidentified source.** The queue carries a CONT net that SIGCONTs any job
   process left in state T and counts each firing. MEASURED (ours), `~/settle24/code/README.md` on the Spark
   (identical to `SETTLE/tools/spark24-job-queue/README.md`). A run outside the queue has no such net.
6. **Two clocks in one file.** The Spark's `date -u` and its torch warning lines disagree by 10 hours: the
   bench started at about 10:30Z and its compile warnings read `W1003 20:30:09`. MEASURED (ours),
   `~/settle24/runs/sdmonly/bench.out` against the plan's 10:30Z log. Read UTC from `date -u`, never from a
   library's log prefix.
7. **A possible power-state latch.** Other users report a GB10 stuck near 14 W at 96% utilisation, halving
   throughput, cleared only by removing mains power. MEASURED (elsewhere), forum thread 361294, in
   `wikis/WIKI_HW_SPARK_GB10/17-power-state-latch-and-the-driver-stack.md`. Untested on our box. A stamp that
   reads utilisation without power draw cannot see it.
8. **Warm before you time.** The cold first matmul runs at 8.2 TFLOP/s and the warmed one at 84.2. MEASURED
   (ours), `00-VERIFIED-FACTS.md`.

## 2. RENTED BOXES: a checklist from our own incidents

The one rented box this model family has trained on: instance 53644090, machine 42368, 4x RTX 5090 32 GB,
driver 570.181, `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, $2.1719 an hour all-in with a 160 GB disk,
rented 05:32:45Z and destroyed 16:34:29Z on 2026-10-01. MEASURED (ours), `experiments/task-runner/JIMOTHY_VAST_SETTLE_2026-10-01.md`.
11.03 h at that rate is $23.96, which matches the $26.44 to $2.44 credit fall. UNVERIFIED, arithmetic.

```
   THE 2026-10-01 BOX, 05:32Z to 16:34Z, four GPUs

   05:32  rent ─ 50 s to running
   05:45  the home link measured at 0.34 to 0.90 Mbps ── decision: rebuild data on the box
   05:50  data ─────────────────────── 41 min ──────── 06:31 ALL_MATCH (sha256 x 6)
   06:05  anchor gate 1.596861 (M5 1.59706)
   06:33  V1 fired, 4 of 4 GPUs at 97 to 99%
   10:47  two arms finish while nobody watches ── 2 GPUs idle 46 min, about $0.83
   12:30  two GPUs idle again, no sealed work queued
   16:34  ✗ DESTROYED BEFORE THE FINAL PULL ── a deadline read as local time
          lost: three sets of weights. kept: every score.
```

| # | rule | the incident behind it | path |
|---|---|---|---|
| 1 | **Pick the image from the tool, by compute capability.** Blackwell cards (cc 1200: RTX 5090, RTX PRO 6000 Blackwell; cc 1000: B200) need a cu128 image; `pytorch/pytorch:2.6.0-cuda12.4-...` carries no sm_120 kernels. Run `track0_fleet_provisioner.py --image-for <cc>`. | Blackwell offers (RTX 5060 Ti, 5070, 5080) were filtered out of a fleet because the recipe image `2.5.1-cuda12.4` predated them; "that is how one dead 5090 became two". The cu128 tag booted the 5090 box in about 50 s (its first boot by us; `THE_PROVISIONER.md` still says "never booted by us"). | `experiments/task-runner/system_study_vast_fleet/230630-jimb2-rtx50-image-incompat-and-spin-notes.md`; `~/.claude/skills/jimothy/SKILL.md` (sprint gates); `~/.claude/skills/jimothy/docs/THE_PROVISIONER.md` §1; the 05:32Z log line |
| 2 | **Install a compiler.** The runtime image has none, and torch.compile needs one: `apt install gcc g++`. | 05:50Z on the 5090 box. | `JIMOTHY_VAST_SETTLE_2026-10-01.md` |
| 3 | **Never ship the shards over the home link; rebuild them on the box and check every sha256.** | Spark to box 25 MB in 10 min (0.34 Mbps); M5 to box 0.42 Mbps; box to M5 0.52 Mbps; box to Spark 0.90 Mbps. The box pulled Hugging Face at 4.5 MB/s and rebuilt S0's shards, `train_big` (3,018,298,789 tokens, sha256 `624b04a9...`) and the chat shards in 41 min, all six matching. | `JIMOTHY_VAST_SETTLE_2026-10-01.md` 05:45Z, 06:31Z; `runs_vast/box_logs/data_pipeline.log` (train_big 05:51:38Z to 06:21:21Z) |
| 4 | **Copy the small canonical files from home, with their sha256, instead of regenerating them.** | A box-regenerated `token_bytes.i32` differed at ids 127,998 and 127,999 (8 and 6 bytes against 27 and 17) and moved the anchor from 1.596861 to 1.596985. The canonical file (sha `69f96025...`) was restored. | `JIMOTHY_VAST_SETTLE_2026-10-01.md` 06:05Z; `runs_vast/vast_unigram_s0_20M.first_wrong_token_bytes.result.json` |
| 5 | **Clamp threads from the cgroup quota, and know when it matters.** Run the census on boarding; `nproc --all` and the cgroup, never `nproc`. | 5090 box: `nproc --all` 128, quota 61.44 cores, memory.max 367 GB against a board figure of 252 GB; clamp 15 threads for each of 4 workers. The clamp is worth 2.19x on one worker and 14.5x on four for a CPU-bound GEMM, and 1.0008x on a GPU-bound cell. Our training step is GPU-bound; the on-box tokenizing pass is the CPU-bound phase. | `JIMOTHY_VAST_SETTLE_2026-10-01.md` 05:32Z; `experiments/task-runner/system_study_vast_fleet/MEASURED_2026-07-30_thread-clamp-on-the-real-workload.md`; `track0_harness/cores.py`, `experiments/track0_thread_clamp.sh` |
| 6 | **Arm a stop that does not depend on anyone watching, and make it ship before it destroys.** `poweroff` does not stop billing; only `destroy` does, and a stopped box still bills storage. The box can destroy itself with the injected `CONTAINER_API_KEY`, which reaches only its own instance. | Three 4-GPU hosts sat idle 8.6 to 10.6 h after nobody was left to stop them. On the 5090 box the operator was absent from about 09:45Z to 11:31Z and again 12:10Z to 13:20Z; the box kept training because its chains were box-side. | `~/.claude/skills/jimothy/docs/VAST_AI_OPERATIONS.md` §④ NEGLECT-STOP; `experiments/task-runner/track0_idle_reaper.py --emit-boxside`; the 11:32Z and 13:31Z log entries |
| 7 | **Pull before destroy, and check completeness, not only the clock.** Verify every weights file at home by size and sha256 against the box's own list before the destroy. | The finish script computed its deadline with macOS `date -j -f`, which reads the time as local (AEST), so the deadline was 9 h in the past and the destroy fired on its first loop. Lost: VC base (31 of 310 MB pulled), VBCC4 (35 of 227 MB), VS (51 of 146 MB). | `JIMOTHY_VAST_SETTLE_2026-10-01.md` 16:34:29Z; `runs_vast/box_logs/bf16_sha256.txt` |
| 8 | **Compute every deadline in UTC epoch seconds** (`date -u +%s`, or `date -u -d` on Linux) and refuse a deadline that is already in the past. | The same incident. `SETTLE/tools/spark24-job-queue/end_of_window.sh` does this correctly with `date -u -d "$UNTIL" +%s`. | as above |
| 9 | **Stamp the driver once per host, not once per GPU.** | The 5090 box's own result JSON recorded the driver as `"570.181\n570.181\n570.181\n570.181"`; a group-by-driver on that field would split one host four ways. | `runs_vast/box_logs/probe_sdm_d512.out` (start event); `VAST_AI_OPERATIONS.md` "a per-host scalar that repeats once per card" |
| 10 | **Treat the rented box as a different instrument from the Spark.** Different torch (2.8.0+cu128 against 2.14.1+cu130), driver (570 against 580) and card. Gate with anchors, not bit identity. The same checkpoint evaluated on two devices agreed to 1e-6 (1.5970556 against 1.5970566); retraining the same recipe gave 1.596861 (5090), 1.59706 (M5) and 1.597459 (Spark), inside the 0.004 rerun band. | anchors from SPARK24 R1/R2 and the 06:05Z gate | `SETTLE_CAMPAIGN_2026-09-30.md` (SPARK24 P7); `JIMOTHY_VAST_SETTLE_2026-10-01.md`; `~/.claude/skills/jimothy/SKILL.md` field lesson 6 |
| 11 | **Price the disk with the GPU.** Rank on `dph_total + storage_cost x disk`. Storage varies 6.5 times between hosts ($0.133 to $0.867 a GB a month); a 120 GB request more than doubled one $0.0994 box. | `~/.claude/skills/jimothy/docs/VAST_AI_OPERATIONS.md` §STORAGE | as stated |
| 12 | **Avoid the machines on the bad list.** 34725, 137843, 68203, 57254, 112617, 68196 (never keyed, pull stalls, never boarded, reclaimed while stopped). | `experiments/task-runner/fleet_bad_machines.tsv` | as stated |
| 13 | **A stopped box is not safe custody.** Machine 57254's stopped instance left the board with unpulled evidence after about 18 h stopped. | `fleet_bad_machines.tsv`, RECLAIMED-WHILE-STOPPED row | as stated |
| 14 | **Read limits from the code, not from prose.** On 2026-10-01 the selector enforced $1.75 a GPU while the prose said $0.065. | `JIMOTHY_VAST_SETTLE_2026-10-01.md` 05:31:38Z; `track0_fleet_value_selector.py --limits --json` | as stated |
| 15 | **Name the failure mode of a stuck box.** Docker-pull stall, Hugging Face egress collapse (one box at 6 bytes a second), a stuck layer checksum, never keyed. Two samples of a byte counter, never one status word. | `VAST_AI_OPERATIONS.md` §PROVISIONING FAILURE TAXONOMY | as stated |
| 16 | **Queue the next job on the box before the current one ends.** Box-side chains fired by themselves (VBCC4 at 12:09:38Z after VB). Without a queued item, two GPUs sat idle twice. | `JIMOTHY_VAST_SETTLE_2026-10-01.md` 11:32Z, 13:31Z | as stated |
| 17 | **Keep the box-side scripts in the repository.** The scripts that ran the 2026-10-01 box (`vast_arm.sh`, `export_bf16.py`, `fire_v1.sh`, `fire_v2.sh` and the finish script) are not tracked here and were not found in the repository or its worktrees on 2026-10-03; only `experiments/task-runner/jimothyvast_pull.sh` is. Its OK line checks rsync exit codes, not sha256 against `bf16_sha256.txt`. | search on 2026-10-03 | as stated |

**Pull time is box time.** At the measured 2.32 Mbps aggregate over nine streams, a bf16 file of 309M
parameters (about 0.62 GB) takes about 36 min home, and one of 914M parameters (about 1.83 GB) about 1.75 h.
A full `last.pt` with AdamW state is about 3.7 GB and 11 GB for the same two shapes and does not come home.
UNVERIFIED, arithmetic on `JIMOTHY_VAST_SETTLE_2026-10-01.md` 13:31Z. The box bills at full rate while it
waits on the wire (`PLAYBOOK_what-keeps-the-fleet-hot-and-smooth.md` #24). Ship each bf16 export as it is
written (#14), so the end pull is one file.

## 3. SPEED LEVERS for a head-dominated model

### 3.1 The tied head's loss: MEASURED on the GB10

d 256, `sdm_ngram`, B 32, T 256, random tokens, 10 steps, median of the last 7. MEASURED (ours), Spark job logs
`~/settle24/q/logs/1001010737-0008.log` (load 0.25 at start), `1001011000-0011.log` (load 8.43),
`1001011154-0014.log` (load 1.74). Script: `track4_sdmllm24_bench_loss_paths.py`.

| path | tokens a second | peak GB | first-step loss |
|---|---|---|---|
| plain (fp32 logits, `F.cross_entropy`) | 26,589 · 27,439 | 13.5 | 11.82060 |
| chunked, eager, checkpointed | 18,153 | 4.1 | 11.82060 |
| compiled, full logits | 20,385 | 3.0 | 11.82061 |
| **compiled and chunked** (what we run) | **72,450 · 52,986 · 47,437** | **1.5 to 1.8** | 11.82061 |
| Liger fused linear cross-entropy | 10,408 | 1.7 | 11.82060 |
| our fused CE, eager, chunk 2048 / 4096 / 8192 | 22,137 / 23,157 / 23,480 | 4.8 / 8.5 / 15.8 | 11.82060 |
| our fused CE, compiled, chunk 2048 / 4096 / 8192 | 43,826 / 46,595 / 48,717 | 2.8 / 4.4 / 7.4 | 11.82060 |

```
   the loss path, d 256 on the GB10 (best reading each)

   liger              |@@@@@                                         10,408
   chunked, eager     |@@@@@@@@@                                     18,153
   compiled           |@@@@@@@@@@                                    20,385
   plain              |@@@@@@@@@@@@@@                                27,439
   fused, compiled    |@@@@@@@@@@@@@@@@@@@@@@@@@                     48,717
   compiled+chunked   |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        72,450
```

- The paired smoke at 4M tokens: plain 1.969676 bpb, chunked 1.969798; loss identical at step 450; about
  24k to 27k against about 67k tokens a second. MEASURED (ours), `1001011441-0017.log`, `-0018.log`.
- The compiled-and-chunked reading moved from 72,450 to 47,437 across three runs whose start load went from
  0.25 to 8.43 and back to 1.74. Read one bench as one sample.
- **Untested:** the chunk size for the compiled-and-chunked path (the trainer default is 2048; the fused path
  gained 11% from 2048 to 8192), any loss path on a 5090 other than chunked, and Cut Cross-Entropy itself.

### 3.2 The other levers

| lever | what we hold | label and path | status |
|---|---|---|---|
| table size at equal FLOPs | 40,704 dense against 22,759 and 13,458 with stores | MEASURED (ours), section 0 | the dominant cost of the SDM-only body on the Spark |
| batch size | B 32 to 64 gave +0% to +2% on the plain path at d 256 | MEASURED (ours), `1001010446-0002/-0003.log` | **untested on the SDM-only tables**, where the per-step table traffic should amortise |
| sparse row updates | at B 32, T 256 each hop reads 2k = 64 candidate rows for each of 8,192 positions, 524,288 touches per step; if addresses were uniform that touches 99.97% of a 65,536-row table and 86% of a 262,144-row table | UNVERIFIED, arithmetic; the model logs `distinct_topk_frac` when stats are on (`track4_sdmonly_models.py` line 114) | **untested**; the saving depends on how skewed the addressing is |
| AdamW `fused=True` | the trainer uses the default AdamW | `track4_sdmllm_train_one_arm.py` line 180 | untested |
| torch.compile | the loss lambda compiles `hidden -> loss`; model compile is not used; max-autotune GEMM is unavailable on the GB10 | MEASURED (ours), section 1.3 | max-autotune on a 5090 untested |
| bf16 autocast | in use; bf16 against fp32 top-1 agreement 0.9699 / 0.9638 / 0.9735 on three models | MEASURED (ours), SDMNEXT 03:20Z block, `SETTLE_CAMPAIGN_2026-09-30.md` | standing |
| FP8 | nanochat: 1.17x tokens a second, about 1.05x at matched quality, slower at d12; FP8 without compile 4x slower | MEASURED (elsewhere), via `RESEARCH_BASEDATA_2026-10-04.md` §4 | **untested on ours**; our head is one large GEMM, the case FP8 suits |
| tf32 | `allow_tf32 = True` is set; under bf16 autocast few fp32 matmuls remain | `track4_sdmllm_train_one_arm.py` line 91 | effect untested, expected small |
| memory bandwidth | GB10 230.5 GB/s measured; RTX 5090 1,457 GB/s (Vast's live sample) or 1.79 TB/s (datasheet); RTX PRO 6000 Blackwell 1,792 GB/s; H100 SXM 3.35 TB/s; H200 4.8 TB/s; B200 7.7 TB/s | GB10 MEASURED (ours); others UNVERIFIED, `wikis/WIKI_HW_LANDSCAPE/01-nvidia-gpu-architectures/NV_MEMORY_NVLINK.md`, `experiments/task-runner/vast-card-wiki/cards/` | the two 5090 figures disagree; neither is ours |
| dense bf16 peak | RTX 5090 209.5 TFLOP/s; FP8 838 (FP16 accumulate; about half with FP32) | UNVERIFIED, `vast-card-wiki/cards/rtx-5090-microarch-deep.md` | our 5090 runs reached 98.5 to 102.3 model TFLOP/s (`RESEARCH_BASEDATA_2026-10-04.md` §5) |
| independent arms per GPU | 4 of 4 GPUs at 97 to 99% for 6 h; d 512 200,266, d 768 152,293, d 1024 115,710 tokens a second each | MEASURED (ours), `runs_vast/*.log` | scales perfectly; the right shape for ablations and seeds |
| one model across GPUs (DDP) | dense all-reduce of 309M or 914M bf16 gradients is 0.62 or 1.83 GB a step | UNVERIFIED, arithmetic | `track4_sdmonly_train_ddp.py` syncs only touched table rows; its speed is untested |

## 4. Pre-flight for the long BASE run on a rented box

In order. A step that fails stops the list. Nothing is rented without the navigator's go and budget.

| # | step | command or tool | pass condition |
|---|---|---|---|
| 1 | Step 6 of the plan is true and the navigator has said go with a budget | `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` §3 | every box ticked |
| 2 | Freeze the code: record `git rev-parse HEAD` and the sha256 of every file that will be copied | `sha256sum track4_sdmonly_*.py track4_sdmllm_*.py track4_sdmllm24_*.py` | list saved beside the plan |
| 3 | Measure the shape's cost on the Spark first: the chosen config at the chosen B for 2,000 steps | `track4_sdmonly_spark_arm.sh` through the queue | median tokens a second after step 1,000, stamped |
| 4 | Derive the budget wall: `(credit - floor) / hours`, and the hours from the token target, the rented speed (from step 12) and pull time | arithmetic in the plan's log | the wall covers training, evaluation and the pull |
| 5 | Choose the box with the selector, never from a raw board page; dedupe by `machine_id`; exclude the bad list | `track0_fleet_value_selector.py --plan N` / `--check <offer>` / `--limits --json`; `~/.vast-venv/bin/vastai search offers ... --limit 3000`; `fleet_bad_machines.tsv` | reliability at or above the selector's gate; disk priced in |
| 6 | Read the image tag for the card's compute capability | `track0_fleet_provisioner.py --image-for <cc>` | cu128 tag for cc 1000 and 1200 |
| 7 | Rent with a label and the smallest disk that fits: image (about 4.3 GB) + shards (about 16 GB) + two `last.pt` copies + headroom | `~/.vast-venv/bin/vastai create instance ...` (navigator's go only) | running; ssh keyed |
| 8 | Stamp the host once: driver, torch, CUDA, `torch.cuda.get_arch_list()`, `nproc --all`, cgroup `cpu.max` and `memory.max`, power draw | `nvidia-smi --query-gpu=driver_version,power.draw --format=csv,noheader`; `python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"` | sm_120 present for Blackwell; one driver line per host |
| 9 | Clamp threads before Python starts | `eval "$(bash experiments/track0_thread_clamp.sh --export)"` or `python3 -m track0_harness.cores --workers N --export` | `/proc/<pid>/status Threads:` after a warm-up matches |
| 10 | Install the compiler | `apt install -y gcc g++` | `torch.compile` smoke passes |
| 11 | Arm the neglect stop: ship, then destroy itself with `CONTAINER_API_KEY`, deadline in UTC epoch | `python3 experiments/task-runner/track0_idle_reaper.py --emit-boxside` (read its output before installing) | a deadline in the past refuses to destroy before the ship is verified |
| 12 | Copy the code (sha256 against step 2) and the small canonical files (tokenizer JSON, `token_bytes.i32` sha `69f96025...`) from home | `scp` then `sha256sum -c` | all match |
| 13 | Rebuild the shards on the box from Hugging Face | `track4_sdmllm_prepare_fineweb_edu_token_shards.py`, `track4_sdmllm24_extend_fineweb_edu_shards.py`, `track4_sdmllm24_prepare_smoltalk_chat_shards.py` with the clamp on | every sha256 equals the provenance files (`track4_sdmllm24_train_big_provenance.json`, `track4_sdmllm24_chat_provenance.json`, `track4_sdmllm_data_provenance.json`) |
| 14 | Anchor gate A: evaluate a Spark checkpoint fixture on the box | the trainer's scoring on S0's TEST windows | within 2e-4 bpb of the Spark's number |
| 15 | Anchor gate B: retrain the 20M anchor arm on the box | `track4_sdmonly_train.py --tokens 20000000 --train-name train ...` | within 0.004 of the Spark's run of the same arm |
| 16 | Speed bench on the box: 2,000 steps of the exact BASE config | the same command with a small token target | median after step 1,000 recorded; hours and dollars recomputed |
| 17 | Seal the predictions in the plan before the fire | plan §3 step 2 | committed before the start stamp |
| 18 | Fire detached; chain the next job box-side | `nohup ... &` with `setsid` on the Linux box | process alive at 20 s; step lines advancing; GPU 97% or more |
| 19 | During the run: bf16 export at 0.6B, 1.2B, 2.4B, sha256 on the box, shipped as written; each beat checks the process, not utilisation | `jimothyvast_pull.sh` pattern plus a sha256 check; `track0_fleet_value_selector.py --audit` | single-copy time short; no HOT AND IDLE |
| 20 | End: final evaluation, bf16 export, sha256 list, pull in parallel streams, verify at home, copy to the Keep | `rsync --partial` to a temp name, then `sha256sum -c` against the box's list, then rename | every file verified |
| 21 | Destroy by explicit id, then check the board is empty | `~/.vast-venv/bin/vastai destroy instance <id> -y`; `vastai show instances --raw` | `[]` |

## 5. Proposals: cheap tests on the Spark that de-risk the rented run

Ranked by how much each changes the rented run's shape or cost. Each is a few minutes under the lane rules
(`~/settle24/sdmonly_dev/`, `nice -n 15`, no queue) or a queue job of the coordinator's choosing. None has run.

1. **Batch size against table traffic.** Run `bench_d768_n256_h4` and `bench_d768_n512_h4` at B 32, 64 and
   128 (1.6M tokens each). If the step cost is the tables, tokens a second should rise close to linearly with
   B. This decides the batch, and therefore the learning rate retune, for the rented run.
2. **Rows touched per step, and AdamW `fused=True`.** With store statistics on for 50 steps, read
   `distinct_topk_frac` per hop at B 32 and B 128; then time AdamW default against `fused=True` on
   `n512_h4`, same loss on the first steps. This says whether sparse row updates (lane SPARSEROWS) and
   touched-row sync (lane MULTIGPU) can save anything at these table sizes.
3. **A cross-instrument anchor fixture.** Export one wave-1 checkpoint (`p0_A_c`) as bf16 with its sha256 and
   its TEST bpb, ready for pre-flight step 14.
4. **The box-side scripts, rebuilt and dry-run.** Write the arm runner, the bf16 exporter with its sha256
   list, the chain runner and the finish script, and dry-run them in `sdmonly_dev`. Negative controls: a
   deadline already in the past must refuse to destroy, and a missing or mismatched weights file must cancel
   the destroy.
5. **The cost of the end-of-run scoring.** Time one full TEST pass, the per-window pass and the two ablations
   on a tiny SDM-only checkpoint, and record the MemAvailable drop. The rented deadline must include it.
6. **Chunk size for the compiled chunked loss at d 768.** `--ce-chunk` 2048, 4096 and 8192 on 200 steps;
   identical first-step loss required.
7. **A graph-break and recompile audit.** 100 steps of the SDM-only trainer under
   `TORCH_LOGS=recompiles,graph_breaks`; zero recompile-limit lines required.
8. **A power stamp.** Read `nvidia-smi --query-gpu=power.draw,clocks.sm,utilization.gpu --format=csv` once
   while a wave-1 job runs. Read only, zero cost; it settles whether the Spark's numbers come from a latched
   box.

## Sources

Our record (every path relative to the repository unless it starts with `~`):
- `wikis/WIKI_HW_SPARK_GB10/00-VERIFIED-FACTS.md`, `03-...`, `04-...`, `07-...`, `08-...`, `16-...`, `17-...`
- `wikis/WIKI_HW_M5MAX/00-VERIFIED-FACTS.md`; `wikis/WIKI_HW_LANDSCAPE/00-SYNTHESIS.md`,
  `01-nvidia-gpu-architectures/NV_MEMORY_NVLINK.md`
- `WIKI/theory/177-the-sparkport-day-a-cache-that-was-a-stub-and-a-token-that-is-latency-not-bytes-2026-09-13.md`
- `~/.claude/skills/jimothy/SKILL.md`; its `docs/VAST_AI_OPERATIONS.md`, `docs/THE_PROVISIONER.md`,
  `docs/PLAYBOOK_what-keeps-the-fleet-hot-and-smooth.md`
- `experiments/task-runner/JIMOTHY_VAST_SETTLE_2026-10-01.md`, `fleet_bad_machines.tsv`, `jimothyvast_pull.sh`,
  `track0_idle_reaper.py`, `system_study_vast_fleet/230630-...md`, `.../MEASURED_2026-07-30_thread-clamp-on-the-real-workload.md`,
  `vast-card-wiki/cards/`
- `SETTLE/SETTLE_CAMPAIGN_2026-09-30.md` (the SPARK24, JIMOTHYVAST, SDMCHATS, SDMSCORE and
  SDMNEXT blocks); `SETTLE/tools/spark24-job-queue/` (README, `sdm_arm.sh`, `end_of_window.sh`);
  `SETTLE/runs/spark24/LANE_SPARK24_STATUS.md`; `SETTLE/runs/sdmchats/spark/memguard.py`
- `experiments/track4/sdmllm/`: `track4_sdmllm24_bench_loss_paths.py`, `track4_sdmllm24_fused_ce.py`,
  `track4_sdmllm_train_one_arm.py`, `track4_sdmonly_models.py`, `track4_sdmonly_train.py`,
  `track4_sdmonly_train_ddp.py`, `track4_sdmonly_spark_arm.sh`, `runs_spark24/`, `runs_sdmchats/`,
  `runs_vast/` and `runs_vast/box_logs/`, `RESEARCH_BASEDATA_2026-10-04.md`
- On the Spark, read only on 2026-10-03: `~/settle24/code/README.md`; `~/settle24/q/logs/1001010446-0002.log`,
  `-0003.log`, `1001010737-0008.log`, `1001011000-0011.log`, `1001011154-0014.log`, `1001011441-0017.log`,
  `-0018.log`; `~/settle24/runs/sdmonly/bench.sh`, `bench.out`, `bench_*.log`

Elsewhere, as recorded in our wikis: nanochat discussion 710 (DGX Spark speedrun); NVIDIA Developer Forums
thread 361294 (power draw); the nanochat and modded-nanogpt logs summarised in `RESEARCH_BASEDATA_2026-10-04.md`.
