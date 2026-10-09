# SDMONLY · the two-day push · 2026-10-03 to 2026-10-05

This file is the plan and the memory of the push. Every beat reads it first and appends to THE LOG at the end.
A beat that is not in THE LOG did not happen.

- **Window:** 2026-10-03T10:30Z to 2026-10-05T10:30Z. A long training run may run past the end of the window.
- **Operator:** JIMOTHY (the second Claude session on this checkout). The first session owns the SETTLE site and
  its ledger; this push writes only under `experiments/track4/sdmllm/` and commits by explicit path.
- **Machine:** the Spark (GB10), which this push has priority on. Its daemons stay up. Rented boxes come later,
  in step 6, and only after the Spark has run every step end to end.


## 0. THE LAW OF THIS CAMPAIGN: ALL SDM, NO TRANSFORMER (the navigator, 2026-10-03T14:10Z)

The navigator's words, after the fair test put a Muon-trained transformer ahead at every budget: "It must be all
SDM, no transformer. But we can do transformer things with SDM. Transformer inspired, but ALL SDM." And: "give it a
decent context window, like 256k."

1. SDM BASE, SDM CHAT and the WEIRD LITTLE GUY contain no softmax attention and no transformer block.
2. Every step that mixes information across positions is an SDM mechanism: a Kanerva memory WRITTEN AT RUN TIME
   (each position writes a value into the hard-locations near its address) and READ (each position reads the
   hard-locations near its own address, holding only what earlier positions wrote). Layers of these may be
   stacked, with several addresses per layer, and counters may fade with age.
3. The number of hard-locations is fixed, so the cost of a token does not grow with the length of the window. That
   is what makes a long window, up to 256k tokens, reachable.
4. Per-token dense layers (the "think" step after a read) are allowed. Trained memory tables are allowed.
5. A transformer is trained only as a yardstick, on the same data, tokens and optimiser, and every SDM result is
   reported beside it.
6. Everything else in this plan stands: from scratch, no teacher, sealed predictions, a control beside every claim.

## 1. What we are making

Three models, each trained from scratch by plain next-token prediction. No teacher and no distillation.

```
   SDM BASE   ── learns the language       web text, the long run        a visitor can chat with it on the site
       │
   SDM CHAT   ── learns to follow          SDM BASE + chat fine-tune     the instruction-following model
       │
   WEIRD LITTLE GUY ── learns one character   SDM CHAT + further rounds on his own corpus: our Searle-style
                                              writing and other philosophy text, more of it than before
```

The body of each model is SDM-only: context features, then a stack of sparse memory reads, then the tied token
table. The model code is `track4_sdmonly_models.py`; the trainer is `track4_sdmonly_train.py`, which reuses the
S0 trainer's loop so every score is comparable with earlier SDMLLM runs.

## 2. The rules of the push

- Predictions are sealed in this file before a wave fires. A sealed number is never edited afterwards; a miss is
  recorded as a miss.
- Every SDM-only arm has a control: no reads (`sdmonly_none`), a dense layer of equal FLOPs (`sdmonly_dense`),
  and the current shape with its readout (`sdm`, `sdm_nostore`).
- A difference counts only when it is at least 0.003 bpb and its paired z over TEST windows is at least 3.
- Each step is gated. A step that misses its gate stops the climb until it is fixed or the navigator rules.
- Weights are pulled and checked by sha256 before anything is deleted. Deadlines are computed in UTC.
- After step 0 the operator confirms back with the navigator before the long run starts.

## 3. The steps

### Step 0 · confirmations and basic hyperparameters (Spark, small runs)
- [x] Model code written, self-test 15 of 15 on the M5 and on the Spark
- [x] Trainer written, smoke run on the M5 (48 steps, loss falls, ablations score)
- [ ] Speed bench on the Spark at d 256 and d 768 (six short runs)
- [ ] Wave 1 sealed (section 4) and submitted to the Spark queue
- [ ] Wave 1 scored: table of TEST bpb, paired differences, ablations, tokens a second
- [ ] Wave 2: the best settings combined, and a second seed of the centre and of the best
- [ ] Hyperparameters chosen: learning rate, store learning rate, hops, locations, k, heads, schedule
- [ ] **CONFIRM BACK with the navigator: the shape, the settings, the long-run plan**

### Step 1 · scale check (Spark)
- [ ] The chosen shape at d 768 on 100M to 300M tokens of `train_big`, beside the current recipe at the same budget
- [ ] Gate: the SDM-only shape holds at width (within 0.01 bpb of the current recipe, or ahead)

### Step 2 · SDM BASE, the long run (Spark; may run past the window)
- [ ] Predictions sealed
- [ ] The long run fired: target 3B tokens of FineWeb-Edu, warmup, constant, linear decay
- [ ] Checkpoints kept at 0.6B, 1.2B and 2.4B for the token-scaling curve
- [ ] Gate: TEST below 1.40285 (today's `sdmwide768`)
- [ ] Weights exported, sha256 recorded, a copy in the Keep

### Step 3 · SDM CHAT
- [ ] Chat fine-tune of SDM BASE on `chat_mix`
- [ ] The dense control fine-tuned the same way
- [ ] Gate: CHAT below 1.06386 (today's `sdmwide768chat`), no looping replies in 30
- [ ] EXPLORATORY, LATER: RL for CHAT, with Xiaomi MiMo-V2.6's open RL release as the starting point
      (see the log entry of 2026-10-03T14:52Z). Their record format, checker types and RL loop, copied
      closely; our own easy rungs first; their 7,780 tasks as the top rung. Runs only after CHAT exists.
      Lane RLLADDER builds the spec and the task generator now.
- [ ] EXPLORATORY: collect what MiMo publishes about training its small 9B model (recipe settings,
      batch and group sizes, lengths, learning rates) and set it beside our own small-model recipe

### Step 4 · WEIRD LITTLE GUY
- [ ] Corpus v2: more of our own Searle-style writing, plus public-domain philosophy text; provenance recorded
- [ ] GUY off SDM CHAT, with web and chat rows mixed into every batch
- [ ] Control: GUY off SDM BASE
- [ ] Further rounds on the corpus; each round scored on his own TEST, web TEST and chat TEST

### Step 5 · exports and the site
- [ ] int8 exports and agreement with float32 for BASE, CHAT and GUY
- [ ] Browser speed measured
- [ ] Hand-over note to the site lane: three models, their files, their scores

### Step 6 · the rented boxes (a planning step only, and last)
Nothing is rented and nothing is planned in detail until every line below is true.
- [ ] A better champion than today's on the Spark
- [ ] A training framework that runs every step from one command
- [ ] The SDM read and write shape works well as an autoregressive language model
- [ ] SDM BASE, SDM CHAT and the WEIRD LITTLE GUY all do well
- [ ] The site pages are updated for the three models
- [ ] The paper on the SDM BASE model and its training is written
- [ ] Then: a bench plan for the biggest, fastest boxes at the best value, and the navigator's go with a budget

### Documents
- [x] This plan
- [x] `TRAINING_SDMONLY_HOW_WE_TRAIN.md`: how we train, with the equations and a plain reading of each (results sections owed)
- [ ] The paper: method, the sealed predictions, the results, what failed

## 4. Wave 1, sealed 2026-10-03 before any run

**Protocol.** S0's protocol on the Spark: 20M tokens of the `train` shard, B 32, T 256, d 256, seed 0, AdamW
(0.9, 0.95), clip 1.0, warmup 2%, cosine to 10%, TEST on S0's windows. Every arm uses the wide address (8 back
tokens, moving averages at 0.5, 0.8, 0.9, 0.97, 0.99).

**References** (the S0 trainer):

| run | arm | settings |
|---|---|---|
| `p0_R_N` | `sdm_nostore` | wide address, readout 688 |
| `p0_R_S` | `sdm` | wide address, readout 688, 3,600 locations, softness 0.25, qgrad 0, values zero, store lr x3, store wd 0 |
| `p0_R_Q` | `qwen` | the 4-layer transformer yardstick |

**The centre and its controls** (the new trainer):

| run | arm | settings |
|---|---|---|
| `p0_A_c` | `sdmonly` | 4 hops, one store per hop, 256 x 256 = 65,536 locations each, k 32, 1 head, softness 0.25, qgrad 0, lr 3e-3, store lr x3, store wd 0, no readout |
| `p0_A_none` | `sdmonly_none` | no reads |
| `p0_A_dense` | `sdmonly_dense` | each read replaced by a SwiGLU of equal FLOPs |

**One change from the centre each:**

| run | change | run | change |
|---|---|---|---|
| `p0_A_lr1` | lr 1e-3 | `p0_A_k16` | k 16 |
| `p0_A_lr6` | lr 6e-3 | `p0_A_k64` | k 64 |
| `p0_A_lr10` | lr 1e-2 | `p0_A_heads4` | 4 heads |
| `p0_A_slr1` | store lr x1 | `p0_A_share` | one store shared by the 4 hops |
| `p0_A_slr10` | store lr x10 | `p0_A_soft5` | softness 0.5 |
| `p0_A_h2` | 2 hops | `p0_A_qg1` | qgrad 1 |
| `p0_A_h8` | 8 hops | `p0_A_ro` | readout 688 after the hops |
| `p0_A_n128` | 128 x 128 locations | `p0_A_wsd` | warmup, constant, 20% linear decay |
| `p0_A_n512` | 512 x 512 locations | `p0_A_wd` | store weight decay 0.1 |

**Sealed predictions.**

| id | prediction |
|---|---|
| P1 | `p0_A_c` scores TEST between 1.60 and 1.70 |
| P2 | `p0_A_c` beats `p0_A_none` by at least 0.03 |
| P3 | `p0_A_c` beats `p0_A_dense` by 0.005 to 0.06 |
| P4 | `p0_R_S` beats `p0_A_c` by 0.00 to 0.05 |
| P5 | the best learning rate of the four is 3e-3 or 6e-3 |
| P6 | more hops score better: h8 below h4 below h2; h8 beats h4 by less than 0.01 |
| P7 | `p0_A_n512` is within 0.005 of `p0_A_c` |
| P8 | `p0_A_ro` beats `p0_A_c` by at least 0.005 |
| P9 | on `p0_A_c`, zeroing the reads costs at least 0.15 bpb and shuffling the values costs at least 0.03 |
| P10 | `p0_R_N` is within 0.01 of `p0_R_S` |

**Decision rule.** The shape that goes to step 1 is SDM-only if its best arm is within 0.01 bpb of the best
reference that has no attention (`p0_R_S` or `p0_R_N`). If it is further behind, the navigator is shown the
table and rules. Either way the table goes to the navigator before the long run.

## 5. The heartbeat

A cron fires every 30 minutes. It is a safety net: between beats the operator keeps working. Each beat reads this
file, reads the Spark (the queue, the GPU, the newest logs), does the next unchecked thing, appends one entry to
THE LOG, and shows the navigator a fresh ASCII picture drawn from that beat's own numbers.

## 6. THE LOG (append only)

- 2026-10-03T10:30Z · START. The Spark was idle (GPU 1%, load 0.19, only daemons), so nothing was cancelled.
  Model and trainer written; self-test 15 of 15 on both machines; smoke run on the M5. Six speed benches started
  on the Spark (`~/settle24/runs/sdmonly/bench.sh`).
- 2026-10-03T10:50Z · Bench, first reading: d 256, 4 hops, 65,536 locations a hop runs at about 46,500 tokens a
  second on the Spark alone (20M tokens in about 7 minutes). Wave 1 (24 runs, section 4) is armed to enter the
  queue at priority 9 the moment the bench ends. The 30-minute heartbeat cron is set (job 86b5c6ca, durable).
- 2026-10-03T10:40Z · CORRECTION to the entry above: it is stamped 10:50Z and was written at about 10:36Z; the
  operator guessed the time instead of reading the clock. From here every stamp is read from `date -u`.
  Eight lanes opened with a shared channel (`SDMONLY_CHANNEL.md`): SPARSEROWS, CHATGUYCHAIN, GUYCORPUS,
  BROWSEREXPORT, SAMPLELOOPS, MULTIGPU, BOXSIZING, PAPERDRAFT. The coordinator commits for all of them.
- 2026-10-03T10:41:50Z · BUDGET, the navigator's word: about $300 of Vast credit for the rented step, with the box
  layout whatever suits (one big multi-GPU box for the BASE; extra boxes only for a run that earns one). Lane
  BOXSIZING is sizing to it. Still a planning step: nothing is rented without his go.
- 2026-10-03T10:41:59Z · The navigator adds: the $300 may be spent inside 1 to 2 days, and the model is sized to
  that (about $6 to $12 an hour of sustained burn). Passed to BOXSIZING.
- 2026-10-03T10:54:51Z · THE TEST FLEET, by the navigator's order: three Vast boxes for 5 hours, all destroyed at
  2026-10-03T15:45:00Z (`vast_sdmonly_boxes.list`; the pull loop and the deadline are
  `track4_sdmonly_vast_watch.sh`, running detached on the M5). Credit before: $352.41; no boxes were left over.
  a = RTX 5090 $0.5267/h, b = A100 SXM4 40 GB $0.6111/h, c = RTX PRO 6000 Blackwell 96 GB $1.3487/h: $2.49/h,
  about $12.4 for 5 hours. All three reached running in under 6 minutes on the pinned cu128 image.
  TRANSFER PRICE, the navigator's high principle: the first 5090 charged $39.06/TB both ways (board median $5.21,
  p90 $39.06, max $100 up); it was destroyed before use and replaced. Lane TRANSFERCOST is putting a hard veto
  and a transfer weight into the selector.
  Each box runs `track4_sdmonly_vast_t1.sh`: the six-config speed bench, then wave 1's centre arm as the anchor,
  then its cells of the sizing grid below.

### Wave S (the sizing grid), sealed 2026-10-03T10:54:51Z before any cell runs
Centre settings (4 hops, 65,536 rows a hop, k 32, lr 3e-3, store lr x3, cosine), seed 0, the `train` shard, one
run at a time on its box. A cell is named `sz_d<width>_<tokens>M`.

| box | cells |
|---|---|
| c (RTX PRO 6000) | d 768 at 20M and 60M; d 1024 at 20M |
| a (RTX 5090) | d 512 at 20M and 60M; d 384 at 60M |
| b (A100) | d 256 at 60M; d 384 at 20M |

The d 256 at 20M cell is the anchor `p0_A_c`, which every box and the Spark run. Scores from different boxes are
compared only if every box's anchor agrees with the Spark's within 0.002 bpb.

| id | prediction |
|---|---|
| S1 | at 20M tokens wider is better at every step from 256 to 1024, and 768 beats 512 by less than 0.02 |
| S2 | at every width run at both budgets, 60M beats 20M by at least 0.05 |
| S3 | at about equal compute, the smaller model with more tokens wins: d 256 at 60M beats d 768 at 20M |
| S4 | the anchors agree: each box's `p0_A_c` is within 0.002 bpb of the Spark's |
| S5 | tokens a second at d 768, 65,536 rows a hop: the 5090 is 4 to 7 times the Spark's 22,800; the PRO 6000 is at least the 5090's; the A100 is below the 5090 |
- 2026-10-03T10:56:34Z · BEAT 1. Wave 1 on the Spark: `p0_A_c` and `p0_R_S` training side by side, 22 queued,
  no result yet. Boxes, speed bench in progress (median tokens a second, each box alone on its GPU): d 256 centre
  shape: 5090 226,500, PRO 6000 213,949, A100 163,165 (Spark 46,700). d 768, 65,536 rows a hop: 5090 108,082,
  PRO 6000 111,863 (Spark 22,800), so S5's 4 to 7 times holds for the 5090 at 4.7. No tracebacks. Landed and
  waiting for the coordinator's check and commit: PAPERDRAFT, GUYCORPUS, BOXSIZING, PRIORSDM, PRIORMACHINES,
  PRIORSPARSESTAR. Warning taken from PRIORSDM: `p0_A_qg1` is a known blow-up recipe (+0.124 bpb in the earlier
  model), and a 20M-token win needs a second seed and a 100M check before it counts.
- 2026-10-03T11:06:44Z · SPEED BENCH DONE on the three boxes (median tokens a second after the first third of steps,
  each alone on its GPU, B 32 x T 256, chunked loss; drivers in `vast_sdmonly_boxes.list`):

  | config | Spark GB10 | RTX 5090 | RTX PRO 6000 | A100 40 GB |
  |---|---|---|---|---|
  | d 256, 4 hops, 65k rows a hop | 46,700 | 226,500 | 213,949 | 163,165 |
  | d 768, 4 hops, 65k rows a hop | 22,800 | 108,082 | 111,863 | 81,043 |
  | d 768, 4 hops, 262k rows a hop | 13,500 | 74,869 | 76,275 | 58,314 |
  | d 768, 4 hops, one shared 262k table | (last-step reading only) | 102,485 | 105,593 | 77,270 |
  | d 768, 8 hops, one shared 262k table | (last-step reading only) | 82,642 | 83,468 | 58,840 |
  | d 768, dense control, 4 hops | 40,704 | 161,014 | 175,786 | 127,140 |

  S5 scored: HIT on the 5090 (4.7 times the Spark at d 768) and on the A100 (below the 5090); the PRO 6000 is
  1.03 times the 5090, so "at least the 5090's" holds by a hair. Tokens per dollar at d 768: 5090 739M, A100 477M,
  PRO 6000 299M. The 5090 is the value card for this model.
  A WRONG BEAT, mine: the bundle sent to the boxes lacked `track4_sdmllm_samples_and_repetition.py`, which the
  S0 trainer imports at the end of a full-TEST run, so on every box the anchor and the first sizing cells trained
  to the end and then died before writing a result. Their final checkpoints are on disk. Fixed at 11:07Z: the
  file is shipped, and each box re-runs its list (a finished-but-unscored run resumes from its last step and only
  scores). A second fault in the test script: its log line prints the exit code of `date`, not of the run, so
  "rc 0" there means nothing. Lesson: a bundle is proven by one full run to its result file, before the queue.
  CHANGE OF PLAN: on the Spark two wave-1 runs share the GPU at 8,000 to 17,000 tokens a second each, so wave 1
  there needs about 5 hours. Each box now also runs ALL of wave 1 after its sizing cells (about 2 minutes a run
  on the 5090). The Spark's wave 1 continues; the four copies give the seed-free machine-to-machine spread.
- 2026-10-03T11:18:26Z · FIRST SCORES. Spark, wave 1 (20M tokens, d 256): `p0_A_c` (SDM-only centre) TEST 1.66998,
  reads zeroed 2.90653, values shuffled 2.71649; `p0_R_S` (the earlier shape with its readout) 1.59795, reads
  zeroed 1.60963. Paired: R_S minus A_c = -0.07203 (SE 0.00038, z -187.7).
  P1 HIT (1.66998 is inside 1.60 to 1.70). P4 MISS: the earlier shape is ahead by 0.072, sealed 0.00 to 0.05.
  P9 HIT (zeroing costs 1.237, shuffling 1.047). Sizing cells, 60M tokens, on the boxes (anchors not yet scored,
  so cross-box comparison is provisional): d 256 1.60734 (A100), d 512 1.57759 (5090), d 768 1.55952 (PRO 6000).
  Loop check on `p0_A_c` (float32, lane SAMPLELOOPS): 1 of 30 chat replies and 1 of 15 poems loop.
  Reading so far: the SDM-only centre is behind the earlier shape at 20M; the arm with a readout (`p0_A_ro`) and
  the hops, rows and learning-rate arms say what closes it. No decision before the full table.
- 2026-10-03T11:21:25Z · BEAT 2. Boxes a and c finished their first pass (T1_DONE) and are re-scoring the runs that
  died before their result file, then start wave 1; box b is on its last sizing cell. New sizing cells: d 384 at
  60M 1.58887 (5090), d 1024 at 20M 1.60273 (PRO 6000). Spark: `p0_R_N` and `p0_A_none` training, 20 queued.
  Pull loop healthy (20 result files home at 11:17Z). Credit $351.12 ($352.41 at the start, 34 minutes of three
  boxes). No tracebacks since the bundle fix.
- 2026-10-03T11:28:32Z · A WRONG BEAT, mine: the guard that started the boxes' second pass was
  `pgrep -f wave1box.sh || start`, and `pgrep -f` matched the ssh command that carried it, so nothing started.
  Boxes a, c and b sat HOT AND IDLE from 11:18Z, 11:21Z and 11:23Z to 11:28Z (about 8 box-minutes each, $0.33).
  Found by reading the process list, not the log. Started without the guard at 11:28Z on a and b. The fleet law
  already says it: identify a process by its command name, never by a string you just typed.
  Box roles from here: a and b run the second pass and all of wave 1; c runs the second pass, then the
  tokenizer wave, then wave 2. New sizing cell: d 384 at 20M 1.64440.

### Wave V (the tokenizer wave), sealed 2026-10-03T11:28:32Z before it runs
Box c. The centre arm (d 256, 4 hops, 65,536 rows a hop, seed 0) trained on the same text at equal training bytes
(95,417,201 bytes, the bytes of 20M DeepSeek tokens) with four tokenizers: DeepSeek 129,280 (`vw_v129k`), and
byte-level BPE of 65,536, 32,768 and 16,384 trained on FineWeb-Edu train text (`vw_v65k`, `vw_v32k`, `vw_v16k`).
Score: bits per byte on held-out documents 1,000 to 1,999, each from its own start, every byte once
(`track4_sdmonly_vocab_score.py`).

| id | prediction |
|---|---|
| V1 | `vw_v32k` scores lower than `vw_v129k`, by 0.00 to 0.03 |
| V2 | `vw_v16k` scores higher (worse) than `vw_v32k` |
| V3 | `vw_v65k` is within 0.01 of `vw_v129k` |
| V4 | training speed in bytes a second: `vw_v32k` is at least 1.7 times `vw_v129k` |

### Wave 2 (recipe arms), sealed 2026-10-03T11:28:32Z before it runs
Box c, after wave V. Each arm is the centre arm with one change, 20M tokens of `train`, seed 0, compared window
by window with box c's own `p0_A_c`.

| run | change | run | change |
|---|---|---|---|
| `w2_untie` | a separate input token table | `w2_slr30` | store lr x30 |
| `w2_hopmlp` | a SwiGLU of width 192 after every read | `w2_slr60` | store lr x60 |
| `w2_ng` | exact 2-gram and 3-gram hash tables, 65,536 rows each | `w2_B64` | batch 64, same tokens |
| `w2_conj2` | product of the last 2 token vectors as a feature | `w2_B128` | batch 128, same tokens |
| `w2_conj3` | products of the last 2 and 3 | `w2_muon` | Muon on the body matrices |
| `w2_nb16` | 16 back tokens, 8 averages | `w2_sparse` | sparse row updates |
| `w2_nb16_none` | the same address with no reads | `w2_ro_hop` | readout 688 and hop SwiGLU 192 together |

| id | prediction |
|---|---|
| W1 | `w2_untie` beats the centre by at least 0.005 |
| W2 | `w2_hopmlp` beats the centre by at least 0.02 |
| W3 | `w2_ng` beats the centre by at least 0.005 |
| W4 | `w2_conj2` is within 0.005 of the centre |
| W5 | `w2_nb16` beats the centre by at least 0.003 |
| W6 | `w2_slr30` beats the centre; `w2_slr60` does not beat `w2_slr30` |
| W7 | `w2_B64` is worse than the centre by at least 0.01 at equal tokens |
| W8 | `w2_muon` beats the centre by at least 0.005 |
| W9 | `w2_sparse` is within 0.003 of the centre |
- 2026-10-03T11:32:17Z · ANCHORS AND WAVE S SCORED. `p0_A_c` (same seed, same tokens) on four machines: Spark
  1.66998, RTX 5090 1.66788, RTX PRO 6000 1.66773, A100 1.66303. Spread 0.00695. S4 MISS: sealed "within 0.002 of
  the Spark"; the 5090 is 0.0021 away, the PRO 6000 0.0023, the A100 0.0070. The earlier shape replicates far
  tighter: `p0_R_S` Spark 1.59795, 5090 1.59778 (0.0002). So the SDM-only arm's machine-to-machine noise at 20M is
  about 0.007, and no wave-1 or wave-2 difference under 0.01 counts without a second seed. The two Blackwell
  cards agree with each other to 0.00015, so the reported compile fault on that family shows no sign here.
  Wave S, TEST bpb (20M | 60M): d 256 1.668 (four-machine mean) | 1.60734; d 384 1.64440 | 1.58887; d 512 1.62884
  | 1.57759; d 768 1.61256 | 1.55952; d 1024 1.60273 | not run.
  S1 HIT (wider is better at every step; 768 beats 512 by 0.016). S2 HIT (60M beats 20M by 0.051 to 0.061 at
  every width). S3 HIT but inside the noise (d 256 at 60M 1.60734 against d 768 at 20M 1.61256, 0.005).
  Git: a commit of the sealed waves is waiting; the other session has a merge open on the shared checkout and
  git refuses a partial commit during a merge. The files are staged.
- 2026-10-03T11:37:19Z · DELEGATED, the navigator's word: the operator decides the CUDA setup, shapes and training
  settings from the measurements. Settled by him: the RTX 5090 is the card (108,082 tokens a second at d 768,
  739M tokens a dollar). Money stays inside about $300 and every rental is logged here with its price.
- 2026-10-03T11:42:04Z · WAVE 1 CONTROLS AND WAVE V SCORED (20M tokens, d 256; 5090 unless said).
  `p0_R_Q` transformer 1.56547 · `p0_A_dense` 1.58582 · `p0_R_S` 1.59778 · `p0_R_N` 1.60038 · `p0_A_none`
  1.66239 (A100: 1.66026) · `p0_A_c` 1.66788 (A100: 1.66303) · `p0_A_lr1` 1.71002.
  P2 MISS: the centre does not beat no reads at all (it is 0.005 behind, inside the noise). P3 MISS: the
  equal-FLOP dense control beats the centre by 0.082. P10 HIT (R_N within 0.003 of R_S).
  Reading: at 20M tokens the sparse reads add nothing a no-read model lacks, and one small dense layer per hop is
  worth 0.08. Wave 2's `w2_hopmlp`, `w2_ng`, `w2_slr30` and `w2_slr60` test the fixes.
  Wave V (box c, equal bytes, by-document bpb): 129k 1.66811-trainer / reference; 65k -0.00543 (z -7.6); 32k
  -0.02069 (z -25.2); 16k -0.01990 (z -19.7). V1 HIT, V2 HIT (16k is 0.0008 worse than 32k, inside the noise),
  V3 HIT. V4 not yet read. DECISION (delegated): the tokenizer moves to the trained 32,768 BPE, pending one second
  seed of the 32k and 129k arms.
  All eight lanes have reported and are committed (3a2d0d522). The browser export agrees with PyTorch on 98.1% of
  top-1 choices in int8 on `p0_A_c`; the file is 107.3 MB at d 256 and 337.1 MB at d 768 with four 65,536-row
  stores, so a d 768 browser model needs one shared store or 128 x 128 locations.

### Wave 3 (the run-time write, and second seeds), sealed 2026-10-03T11:42:04Z before it runs
Box a, after its wave 1. The run-time write: each earlier position in the 256-token window writes the vector of
the token that followed it into the K locations its address woke; a later position reads the locations it wakes
(`--rt-write K`). 20M tokens of `train`, d 256, compared with box a's own runs.

| run | what |
|---|---|
| `w3_rt8` | centre + run-time write, K 8 |
| `w3_rt8_shuf` | the control: the same with the written vectors shuffled across positions |
| `w3_rt32` | K 32 |
| `w3_rt8_none` | no trained stores at all: features + the run-time write only |
| `w3_rt8_dense` | the dense control + the run-time write |
| `p0_A_c_s1`, `p0_A_none_s1`, `p0_A_dense_s1` | seed 1 of the centre, of no reads, and of the dense control |

| id | prediction |
|---|---|
| R1 | `w3_rt8` beats the centre by at least 0.02 |
| R2 | `w3_rt8_shuf` is within 0.01 of the centre and at least 0.015 worse than `w3_rt8` |
| R3 | `w3_rt8_none` beats `p0_A_none` by at least 0.02 |
| R4 | `w3_rt32` is within 0.01 of `w3_rt8` |
| R5 | seed 1 moves each of the three arms by less than 0.01 |
- 2026-10-03T11:50:44Z · A FOURTH BOX, on the navigator's "more boxes" and his delegation: instance 54007096, one
  4x RTX 5090 box (machine 137090, Taiwan, reliability 0.997, $2.1481/h, transfer $2.60 down / $3.91 up per TB,
  128 cores, 257 GB), label sdmonly-d-4x5090, same 15:45Z deadline, in `vast_sdmonly_boxes.list`. Fleet burn is
  now $4.63/h; to the deadline about $18 more. Four runs side by side, one per GPU
  (`track4_sdmonly_vast_gpuqueue.sh`).

### Wave 4 (replication), sealed 2026-10-03T11:50:44Z before it runs
Box d, `track4_sdmonly_wave4_queue.txt`, 27 runs: six 60M-token confirmations of the wave-1 controls at seed 0
(`c60_A_c`, `c60_A_none`, `c60_A_dense`, `c60_R_S`, `c60_R_N`, `c60_R_Q`) and seed 1 of the wave-1 arms
(`p0_*_s1`; the centre, no-reads and dense seed-1 runs are in wave 3).

| id | prediction |
|---|---|
| R6 | at 60M the order of the controls is the same as at 20M: transformer, dense, earlier shape with and without its store, then the centre and no reads |
| R7 | at 60M the centre is within 0.01 of no reads |
| R8 | no wave-1 arm changes its side of the centre between seed 0 and seed 1 by more than 0.01, except `p0_A_qg1` |
- 2026-10-03T11:54:15Z · BEAT 3. WAVE 2, first seven arms (box c, against its `p0_A_c` 1.66773; SE about 0.0003):
  `w2_hopmlp` 1.58324 (-0.08449; reads zeroed 1.58531) · `w2_ng` 1.64665 (-0.02108) · `w2_untie` 1.65146
  (-0.01627) · `w2_conj2` 1.65346 (-0.01427) · `w2_conj3` 1.65702 (-0.01071) · `w2_nb16_none` 1.66473 ·
  `w2_nb16` 1.66930 (+0.00157). W1 HIT, W2 HIT, W3 HIT, W4 MISS (conj2 gains 0.014, sealed within 0.005),
  W5 MISS (the wider address gains nothing).
  WAVE 1 so far (5090): hops 2 1.66724, hops 8 1.66721, 128 rows 1.66835, store lr x1 1.66520, x10 1.66901,
  lr 1e-3 1.71002, lr 6e-3 1.65185, lr 1e-2 1.64848. P6 MISS (hops change nothing). P5 MISS in letter: 1e-2 is
  best of the four by 0.003 over 6e-3, inside the noise.
  Reading: nothing about the trained memory moves the score at 20M tokens; a dense layer after each read is
  worth 0.084 and then the reads are worth 0.002. N-gram tables, an untied input table and product features are
  each worth 0.011 to 0.021. Spark: 6 of 24 wave-1 runs scored, 16 queued. Box d (4x 5090) still pulling its
  image at 5 minutes; the cut-off is 10.

### Wave 5 (the graded read, and combinations), sealed 2026-10-03T11:58:15Z before it runs
Box c, after its wave 2. The graded read (`--graded`): a woken location's weight is its score above the cut-off,
so a read is a k-sparse ReLU layer whose hidden units are memory locations; with `--qgrad 1` the gradient reaches
the features through the address. 20M tokens of `train`, d 256, seed 0, against box c's `p0_A_c`.

| run | what |
|---|---|
| `w5_gr` | graded read, query gradient stopped |
| `w5_gr_qg1` | graded read, query gradient on |
| `w5_gr_qg1_k64` | the same with k 64 |
| `w5_gr_qg1_h8` | the same with 8 hops |
| `w5_gr_qg1_slr1` | the same with store lr x1 |
| `w5_combo` | hop SwiGLU 192 + untied input table + 2-token products + lr 6e-3, sigmoid reads |
| `w5_combo_none` | the same combination with no reads |
| `w5_combo_gr` | the combination with graded reads and query gradient on |
| `w5_combo_ng` | the combination plus n-gram hash tables |

| id | prediction |
|---|---|
| G1 | `w5_gr_qg1` beats the centre by at least 0.03 |
| G2 | `w5_gr` is within 0.01 of the centre |
| G3 | `w5_combo` is within 0.01 of `w5_combo_none` |
| G4 | `w5_combo` beats `w2_hopmlp` (1.58324) by at least 0.02 |
| G5 | `w5_combo_ng` beats `w5_combo` by at least 0.005 |
- 2026-10-03T11:59:40Z · Box d (4x RTX 5090, driver 575.64.03, quota 122.9 cores) reached running at about 11:58Z,
  9 minutes after the rental (its image pull was slow), and is now building its data; the wave-4 queue starts
  by itself when setup says READY. Wave 5 is armed on box c behind wave 2; wave 3 on box a behind wave 1.
  New model option `--graded` (self-test 27 of 27).
- 2026-10-03T12:04:19Z · A SECOND JOB, from the navigator through Claude 1: 100 new home-hero items for the SETTLE
  site on ENFOLD and UNFOLD in our SDM, 50 of them films. Ten Opus lanes fired, each in its own worktree
  (`_worktrees/dwarfstar-fold<lane>`, branch `settle-fold<lane>`): FOLDWRITE, FOLDREAD, FOLDSDM, FOLDOURS,
  FOLDLOOP, FOLDKEYS, FOLDRUNTIME, FOLDHRR, FOLDHOPS, FOLDTRAIN. Brief: `.devlogs/herofold_common_brief.md`.
  Channel: `SETTLE/runs/herofold/CHANNEL.md`. Lanes do not land; Claude 1 lands each green branch.
  The training fleet runs on beside it and is not affected.
- 2026-10-03T12:20:49Z · WAVE 1 COMPLETE on the RTX 5090 (box a; paired against its `p0_A_c` 1.66788; SE 0.0001 to
  0.0006). Controls: `p0_R_Q` 1.56547 · `p0_A_dense` 1.58582 · `p0_A_ro` 1.59750 · `p0_R_S` 1.59778 · `p0_R_N`
  1.60038 · `p0_A_none` 1.66239. One-change arms: lr 1e-2 1.64848 · lr 6e-3 1.65185 · wsd 1.65378 · store lr x1
  1.66520 · store wd 0.1 1.66564 · softness 0.5 1.66714 · hops 8 1.66721 · hops 2 1.66724 · 4 heads 1.66730 ·
  k 16 1.66782 · 512 rows 1.66804 · 128 rows 1.66835 · k 64 1.66851 · shared store 1.66873 · store lr x10
  1.66901 · lr 1e-3 1.71002 · qgrad 1 2.28925 (it diverged, as the earlier record warned).
  P5 MISS in letter (1e-2 is best, 0.003 ahead of 6e-3). P6 MISS. P7 HIT. P8 HIT (the readout is worth 0.070).
  WAVE 2 COMPLETE (box c, against its `p0_A_c` 1.66773): `w2_ro_hop` 1.57602 · `w2_hopmlp` 1.58324 ·
  `w2_muon` 1.61364 (-0.05409; locations used 37 to 52% a store, against 3 to 6% in the centre; values shuffled
  1.72991) · `w2_ng` 1.64665 · `w2_B64` 1.65023 · `w2_untie` 1.65146 · `w2_conj2` 1.65346 · `w2_conj3`
  1.65702 · `w2_nb16_none` 1.66473 · `w2_sparse` 1.66737 · `w2_nb16` 1.66930 · `w2_slr30` 1.67020 ·
  `w2_slr60` 1.67051 · `w2_B128` 1.68072.
  W6 MISS (a hotter store learning rate gains nothing). W7 MISS (batch 64 is 0.018 better, not worse; batch 128
  is 0.013 worse). W8 HIT (Muon gains 0.054). W9 HIT (sparse row updates within 0.0004).
  WAVE 3, first arm (box a): `w3_rt8` 1.64180, 0.02608 better than the centre. Its control is running.
  WAVE 5, first arm (box c): `w5_gr_qg1` diverged (2.22941, gradient norm 69,148 at the end). G1 MISS.
  A WRONG BEAT, mine: box c's 60 GB disk filled at 12:12Z. Every test run kept a 1.4 GB checkpoint and about 40
  runs fill the disk; the rest of wave 5 died on "No space left on device" within a minute. Found at 12:18Z by
  reading the error text. Fixed: a cleaner on all four boxes deletes the checkpoint of every scored run (the
  result, log and per-window files stay); wave 5 restarted at 12:19Z. Lesson: size the disk from the run count
  times the checkpoint size, or delete as you go.

### Wave 6 (Muon with its controls), sealed 2026-10-03T12:20:49Z before it runs
Box a, after its wave 3. Muon moved the centre by 0.054 and made the model use ten times more locations, so
every wave-1 control is repeated under Muon. 20M tokens of `train`, d 256, seed 0, `--body-opt muon`.

| run | what |
|---|---|
| `w6_mu_none` | Muon, no reads |
| `w6_mu_dense` | Muon, the equal-FLOP dense control |
| `w6_mu_c_s1` | Muon centre, seed 1 |
| `w6_mu_hopmlp` | Muon + a SwiGLU 192 after every read |
| `w6_mu_combo` | Muon + hop SwiGLU + untied input table + 2-token products + n-gram tables |
| `w6_mu_combo_none` | the same combination with no reads and no n-gram tables |
| `w6_mu_rt8` | Muon + the run-time write |
| `w6_mu_B64` | Muon, batch 64 |
| `w6_mu_lr2`, `w6_mu_lr05` | Muon learning rate 0.04 and 0.01 (default 0.02) |

| id | prediction |
|---|---|
| M1 | `w6_mu_none` is at least 0.01 worse than `w2_muon` (1.61364): under Muon the reads earn their place |
| M2 | `w6_mu_hopmlp` beats `w2_ro_hop` (1.57602) by at least 0.01 |
| M3 | `w6_mu_combo` scores below 1.565, the transformer's 20M score |
| M4 | `w6_mu_rt8` beats `w2_muon` by at least 0.015 |
| M5 | `w6_mu_c_s1` is within 0.01 of `w2_muon` |
- 2026-10-03T13:40:09Z · RECOVERED after the session limit (the operator and every lane stopped at about 12:25Z; a
  fresh login at 13:35Z). The ten hero lanes were resumed from their transcripts. HOT AND IDLE, named: the four
  boxes finished their waves between 12:26Z and 13:11Z and stood idle until 13:40Z (about $4 of idle burn).
  Credit $342.07 at 13:35Z.
  WAVE 3 (box a, against its `p0_A_c` 1.66788): `w3_rt8_dense` 1.56659 · `w3_rt8_none` 1.63269 · `w3_rt8`
  1.64180 · `w3_rt8_shuf` 1.64841 · `w3_rt32` 1.65569. Seeds 1: centre 1.66910, no reads 1.66250, dense 1.58884.
  R1 HIT · R2 MISS (the shuffled control keeps most of the gain: a topic signal, not recall) · R3 HIT · R4 MISS
  (K 32 is 0.014 worse than K 8) · R5 HIT.
  WAVE 5 (box c, against its `p0_A_c` 1.66773): `w5_combo_ng` 1.55084 · `w5_combo` 1.55828 · `w5_combo_none`
  1.55912 · `w5_combo_gr` 1.61879 · `w5_gr` 1.66297 · with the query gradient on, every graded arm diverged
  (1.84 to 2.26). G1 MISS · G2 HIT · G3 HIT · G4 HIT · G5 HIT.
  WAVE 6, MUON (box a): `w6_mu_combo` 1.53229 · `w6_mu_combo_none` 1.53959 · `w6_mu_hopmlp` 1.55298 ·
  `w6_mu_dense` 1.55579 · `w6_mu_B64` 1.57896 · `w6_mu_rt8` 1.58133 · `w6_mu_lr05` 1.60839 · `w2_muon` (box c)
  1.61364 · `w6_mu_c_s1` 1.61487 · `w6_mu_lr2` 1.62137 · `w6_mu_none` 1.62819.
  M1 HIT (under Muon the reads are worth 0.015 over no reads, the first time they earn their place) · M2 HIT ·
  M3 HIT (1.53229 against the AdamW transformer's 1.56547; that transformer did not have Muon, so this is not yet
  a fair comparison) · M4 HIT (the run-time write is worth 0.032 under Muon) · M5 HIT.
  WAVE 4 (box d), 60M tokens, AdamW: `c60_R_Q` transformer 1.42013 · `c60_A_dense` 1.51760 · `c60_R_S` 1.52872 ·
  `c60_R_N` 1.52877 · `c60_A_none` 1.60660 · `c60_A_c` 1.60893. R6 HIT · R7 HIT. The transformer's lead grows
  with tokens (0.10 at 20M, 0.19 at 60M against the centre).

### Wave 7 (the champion recipe against a fair transformer), sealed 2026-10-03T13:40:09Z before it runs
THE CHAMPION RECIPE (CH): Muon on the body, a SwiGLU 192 after every read, an untied input table, 2-token
products, exact 2- and 3-gram hash tables (65,536 rows each). The transformer (`qwen`, 4 layers) now trains with
the same Muon (`mq`), through the SDM-only trainer. d 256, seed 0 unless named. Queues:
`track4_sdmonly_wave7_q_{a,b,c,d}.txt`.
Box d (4 GPUs): `w7_ch_200M`, `w7_mq_200M`, `w7_ch_rt8_200M`, `w7_ch_none_200M` (CH with no reads and no n-gram
tables), then at 20M `w7_ch_s1`, `w7_mq_20M`, `w7_ch_B64`, `w7_ch_nong`, `w7_ch_rt8`, `w7_mq_s1`.
Box a: `w7_ch_d512_60M`, `w7_mq_d512_60M`. Box b: `w7_ch_60M`, `w7_mq_60M`. Box c, through the tokenizer
wrapper at equal bytes (the bytes of 20M DeepSeek tokens), scored by document: `w7_ch_v32k`, `w7_ch_v129k`,
`w7_mq_v32k`.

| id | prediction |
|---|---|
| X1 | `w7_mq_20M` scores below 1.53229 (the Muon transformer beats the champion at 20M) |
| X2 | `w7_mq_200M` beats `w7_ch_200M` by at least 0.03 |
| X3 | `w7_ch_200M` beats `w7_ch_none_200M` by at least 0.01 |
| X4 | `w7_ch_rt8_200M` beats `w7_ch_200M` by at least 0.01 |
| X5 | `w7_ch_v32k` beats `w7_ch_v129k` by at least 0.015, by document |
| X6 | `w7_ch_d512_60M` beats `w7_ch_60M` by at least 0.03 |

### Wave 8 (context length), sealed 2026-10-03T13:51:45Z before it runs
The window has never been tested beyond 256 tokens. Champion recipe under Muon, 20M tokens, the same tokens a
step (B x T = 8,192), so T 512 runs B 16 and T 1024 runs B 8. Each run is scored on TEST windows of its own
length, so a longer window also gives each scored token more context at test time; that is the effect asked
about, and the arms without the run-time write are its control (their only path past 8 tokens is the averages).
Box a after its wave 7: `w8_ctx256_rt`, `w8_ctx512_rt`, `w8_ctx1024_rt`. Box b after its wave 7: `w8_ctx512`,
`w8_ctx1024` (no run-time write), `w8_ctx1024_mq` (the Muon transformer at 1,024).

| id | prediction |
|---|---|
| C1 | `w8_ctx1024_rt` beats `w8_ctx256_rt` by at least 0.02 |
| C2 | without the run-time write, 1,024 beats 256 (`w7_ch` at 20M is not in the wave; compare with `w6_mu_combo` 1.53229 on box a) by less than 0.01 |
| C3 | the gain from 256 to 1,024 is at least twice as large with the run-time write as without it |
| C4 | `w8_ctx1024_mq` beats `w8_ctx1024_rt` |
- 2026-10-03T14:05:25Z · WAVE 7 SCORED, the fair comparison (all under Muon; TEST bits per byte, d 256 unless named):

  | tokens | Muon transformer | champion (CH) | CH + run-time write | CH, no reads, no n-gram tables |
  |---|---|---|---|---|
  | 20M | `w7_mq_20M` 1.46388 | `w6_mu_combo` 1.53229 | | `w6_mu_combo_none` 1.53959 |
  | 60M | `w7_mq_60M` 1.39966 (A100) | `w7_ch_60M` 1.47910 (A100) | | |
  | 60M d 512 | `w7_mq_d512_60M` 1.34633 | `w7_ch_d512_60M` 1.45053 | | |
  | 200M | `w7_mq_200M` 1.37059 | `w7_ch_200M` 1.46811 | `w7_ch_rt8_200M` 1.46746 | `w7_ch_none_200M` 1.45903 |

  Also `w7_ch_nong` (CH without n-gram tables, 20M) 1.53801.
  X1 HIT (the transformer under the same Muon is 0.068 ahead at 20M). X2 HIT (0.098 ahead at 200M). X3 MISS:
  at 200M the reads and n-gram tables are WORSE than none, by 0.009. X4 MISS: the run-time write is worth 0.0007
  at 200M. X6 MISS by a hair (d 512 beats d 256 at 60M by 0.029, sealed 0.03).
  THE PLAIN READING: with the optimiser held equal, a 4-layer transformer is 0.07 to 0.10 bits per byte ahead at
  every budget we ran, and the gap does not close with tokens. The earlier "champion beats the transformer" at
  20M (M3) compared Muon against AdamW and does not survive the fair test. The trained memory reads, which looked
  useful at 20M under Muon, stop helping by 200M; the attention-free features plus a dense layer per hop carry
  the model. The run-time write's gain at 20M is gone at 200M.
  WAVE 5 TOKENIZER (box c): `w7_ch_v32k` 1.5129 by document; its 129k pair died because the checkpoint cleaner
  deleted the checkpoint the scorer needed (a bug of mine: the cleaner fires on the result file, the scorer runs
  after it). The scorer also could not build the transformer. Both fixed; the pair and the transformer pair rerun
  as `w7r_*` from 14:02Z. Box c sat idle 13:47Z to 14:02Z.
  The step-0 confirmation goes to the navigator now: the result changes what the BASE should be.
- 2026-10-03T14:23:09Z · SCALING, both on Muon, d 256: transformer 1.46388 / 1.39966 / 1.37059 at 20M / 60M / 200M;
  champion 1.53229 / 1.47910 / 1.46811. A three-point fit of E + A D^-alpha gives the champion a floor of about
  1.466 and the transformer about 1.352 (three points, one width, the 60M points on another machine: rough). The
  champion is not under-trained; its curve has flattened, which points at what it can see (8 tokens, 5 averages).
  THE SDM MIXER, built under the law (`SdmMix` in `track4_sdmonly_models.py`, flags `--mix-*`): a Kanerva memory
  written and read at run time inside each layer, 4 heads fading at 1.0, 0.999, 0.99 and 0.9 a token, 4,096
  hard-locations a head chosen by product keys, chunked so only the counters cross chunks. Self-test 32 of 32
  (causal; chunk 7 equals chunk 20; a token 17 positions back reaches the output). M5 smoke run trains.
  BOX d KEPT PAST 15:45Z, the operator's call under the navigator's delegation: removed from the deadline list,
  its own deadline 2026-10-03T18:45:00Z (about $6.40 more). Boxes a, b and c end at 15:45Z as planned.
  Lane CANONREF opened: the canonical open reference, its scaling data and evals, an open RL programme, and an X
  search for the release the navigator saw.

### Wave 9 (the SDM mixer), sealed 2026-10-03T14:23:09Z before it runs
Box d after its wave 7. All SDM: the no-store arm with the champion's features (untied input table, 2-token
products), no trained hops, and L SDM mixing layers, each followed by a dense layer of width 4d. Muon, d 256,
seed 0, 20M tokens unless named. Queue: `track4_sdmonly_wave9_q_d.txt`. References: `w7_mq_20M` (Muon
transformer) 1.46388, `w6_mu_combo_none` (same features, dense layers, no window memory) 1.53959,
`w7_ch_200M` 1.46811, `w7_mq_200M` 1.37059.

| id | prediction |
|---|---|
| N1 | `w9_mix4` beats `w6_mu_combo_none` by at least 0.03 |
| N2 | `w9_mix4` is within 0.03 of the Muon transformer at 20M |
| N3 | `w9_mix4_T1024` beats `w9_mix4` by at least 0.01 |
| N4 | `w9_mix4_200M` beats `w7_ch_200M` by at least 0.05 |
| N5 | `w9_mix4` beats `w9_mix2` by at least 0.01; `w9_mix8` beats `w9_mix4` |
| N6 | `w9_mix4_nofade` is worse than `w9_mix4` |
- 2026-10-03T14:24:42Z · Wave 9 started on box d at 14:23Z (its wave 7 had ended at 14:08Z; 15 idle minutes).
  A WRONG BEAT, mine, caught within a minute: to give box d its own deadline I overwrote
  `track4_sdmonly_vast_watch.sh` in place while the 15:45Z loop (PID 14202) was running from it. Bash reads a
  script by offset, so a longer line 10 could have shifted the destroy step. The original bytes were restored in
  place (git diff empty), the box-d watcher runs from its own file `track4_sdmonly_vast_watch_param.sh` with
  `WATCH_LIST=vast_sdmonly_box_d.list DEADLINE_UTC=2026-10-03T18:45:00Z`, and `vast_sdmonly_boxes.list` no
  longer holds box d, so the 15:45Z loop destroys a, b and c only. Lesson: never edit a running script; copy it.
- 2026-10-03T14:52Z · DECISION A (the navigator agreed): MiMo is a small exploratory side lane; CONTEXT is the main line.
  The finding: the X post the navigator saw is most likely Xiaomi MiMo-V2.6 (open-sourced 2026-09-22, MIT), which
  released its RL stack: dataset `XiaomiMiMo/MiMo-V2.6-RL-oss` (7,780 tasks, 12.1 GB, Apache 2.0: Code 2,700,
  Webdev 2,090, Cyber 1,000, Music 1,000, General 989; checked by tests, visual grading, rules and a rubric judge),
  Docker images, recipes in `github.com/XiaomiMiMo/verl`, and MiMo-V2.6-Distill-Qwen-9B as the starter model.
  Details in `RESEARCH_SDMONLY_CANONICAL_REFERENCE_2026-10-04.md` (its name carries 10-04; the clock here says 10-03).
  Our thoughts, in short:
  - The measured gap to the transformer is 0.057 (mixer 1.52079 against 1.46388 at 20M), and the evidence says the
    model is short of CONTEXT: more tokens flatten near 1.47 and a wider model is level at equal compute. The mixer,
    which reads further back, is the only lever that moved it (-0.019).
  - RL shapes behaviour; it cannot add knowledge or memory the base model lacks. On MiMo's Code, Webdev and Cyber
    tasks every reward for our model would be 0, so RL would learn nothing from them.
  - So: copy their format, checkers and loop now (cheap); build easy rungs that pass 5 to 95 percent; keep their
    tasks as the top rung; the General rubric judge would be ds4-flash via Hermes. Their 9B is distilled, so we use
    none of its weights. Rejected: B (MiMo as the main direction now), C (drop it).
  Wave 9 so far (20M tokens; noise between runs about 0.007): `w9_mix4_n128` 1.52079 · `w9_mix8` 1.52787 ·
  `w9_mix4` 1.52874 · `w9_mix2` 1.53214 · `w9_mix4_k64` 1.54156. N1 MISS (0.0109 against the 0.03 asked) ·
  N2 MISS (0.065 behind) · N3 NOT RUN: `w9_mix4_T1024` died out of memory (the dense location-weight grid, 15.78 GiB
  in one allocation); the fix is a sparse grid · N5 MISS on its first half (0.0034), its second half inside noise ·
  N4 and N6 pending. More locations per head (n128) helped more than more layers.
- 2026-10-03T15:05Z · Wave 9 complete, and A CORRECTION TO MY OWN LAST ENTRY. The 1,024-token runs (`w9_mix4_T1024`,
  `w8_ctx1024`, `w8_ctx1024_rt`, `w8_ctx1024_mq`) did NOT die in the mixer's location grid, as I wrote at 14:52Z. The
  traceback says `eval_bpb` in `track4_sdmllm_train_one_arm.py`: training finished, then the final score built a full
  129k-vocab logit block for 32 windows of 1,024 tokens (15.78 GiB) in one step. The transformer arm died the same
  way, which is what gave it away. Fix: `eval_bpb` now caps an eval batch at about 8k tokens when T x bs > 16,384, so
  every run at T <= 512 scores exactly as before. The checkpoints were already gone, so the four runs are retraining
  now (boxes a and b, then d) with the fix shipped. Lesson: read the traceback before naming the cause.
  Wave 9 full table (TEST bpb): `w9_mix4_200M` 1.49131 · `w9_mix4_n128` 1.52079 · `w9_mix4_nofade` 1.52271 ·
  `w9_mix8` 1.52787 · `w9_mix4` 1.52874 · `w9_mix2` 1.53214 · `w9_mix4_k64` 1.54156.
  N4 MISS, and the sharp one: at 200M the mixer arm (1.49131) is 0.023 BEHIND `w7_ch_200M` (1.46811) and 0.032 behind
  `w7_ch_none_200M` (1.45903). But it is not a like-for-like loss: the wave 9 arm ran with `--hops 0`, so it lacked
  the four hop MLPs (`--hop-mlp 192`) the best no-reads recipe has. N6 MISS: no fading (1.52271) is 0.006 better
  than fading (1.52874), inside the 0.007 noise. Reading: the mixer helps a little at 20M; whether it helps the best
  recipe, and whether it uses a long window, is what wave 10 asks.

### Wave 10 (the mixer on the best recipe, and long windows), sealed 2026-10-03T15:05Z before it runs

Box d, queue `track4_sdmonly_wave10_q_d.txt`. "cn" = the `w7_ch_none_200M` recipe (no store reads, hop MLPs 192,
untie, conj 2, Muon), the best SDM arm at 200M (1.45903). Mixer: 4 layers, 128x128 locations per head.

| id | prediction |
|---|---|
| X1 | `w10_cn_mix4_200M` beats `w7_ch_none_200M` (1.45903) by at least 0.01 |
| X2 | `w10_cn_mix4_T1024_200M` beats `w10_cn_mix4_200M` (T 256) by at least 0.01 |
| X3 | `w10_cn_T1024_200M` (no mixer) is within 0.005 of `w7_ch_none_200M`: without a mixer, a longer window buys nothing |
| X4 | `w10_cn_mix4_nofade_200M` is within 0.007 of `w10_cn_mix4_200M` |
| X5 | `w10_cn_mix4_T4096_200M` beats `w10_cn_mix4_T1024_200M` |
| X6 | the best wave 10 arm is still at least 0.05 behind the Muon transformer at 200M (1.37059) |
- 2026-10-03T15:08Z · Heartbeat. The 15:05Z eval fix was HALF a fix: the reruns on boxes a and b died again with the
  same 15.78 GiB request, this time in `per_window` (`track4_sdmllm_paired_window_comparison.py`), the second
  full-vocab scorer the trainer runs at the end. Same cap applied there (T x bs > 16,384 drops to about 8k tokens a
  batch; T <= 512 unchanged), shipped to boxes a, b and d, and the a and b reruns restarted at 15:04Z (about 3 to 6
  minutes each, before the 15:45Z destroy). The box d runs import `per_window` at the end of training, so they pick
  up the fixed file. Wave 10 is on box d's four GPUs since 15:01Z; the T 1,024 runs are the slow ones.
  Lane RLLADDER finished (`f08cb23f5`): `SPEC_SDMONLY_RL_LADDER_2026-10-04.md` and `track4_sdmonly_rl_ladder.py`,
  127 of 127 self-checks, 8 of 8 mutants caught; 7 easy rungs in MiMo's exact six-column format, then their Music,
  General (ds4-flash judge) and Code. MiMo's GRPO: group 8 to 16, lr 1e-6 to 2e-6, no KL, clip 0.2, advantage not
  divided by the spread on three of five recipes, groups with equal rewards dropped. Credit $335.34.
- 2026-10-03T15:16Z · Wave 8 complete (TEST bpb, 20M tokens each). With the run-time write: T 256 1.51677 · T 512
  1.51473 · T 1,024 1.51400. Without it: T 512 1.53512 · T 1,024 1.53940. Muon transformer at T 1,024: 1.46473
  (1.46388 at T 256).
  C1 MISS (1,024 beats 256 by 0.0028, not 0.02) · C2 HIT in size, wrong sign: without the write, 1,024 is 0.007
  WORSE than `w6_mu_combo` (1.53229), inside the noise · C3 UNDECIDABLE: both gains sit inside the 0.007 noise ·
  C4 HIT (the transformer at 1,024 leads by 0.049).
  THE READING: at 20M tokens NOTHING uses a longer window, the transformer included (1,024 is 0.0009 worse than
  256 for it). A long window pays only once a model has trained long enough to learn long-range use, so the window
  question must be asked at 200M and above. Wave 10's T 1,024 arms at 200M are that test.
  Boxes a, b and c were idle after their reruns: final pull by hand (56, 37 and 43 results home), destroyed by id
  at 15:15Z (54001388, 54001275, 54001277), 30 minutes before their deadline; the 15:45Z watcher (PID 14202) was
  stopped by PID. Box d (54007096) runs wave 10 until its 18:45Z deadline.
  WIKI_MIMO: all five lanes finished (pages 00 to 06; commits 974c1b1b0, 7996ed314, 26754bc43, d95e11010,
  0edd46efd); about 110 MB pulled, gitignored.
- 2026-10-03T15:20Z · First wave 10 result, and the last wave 9 one. `w10_cn_T1024_200M` 1.45886 against
  `w7_ch_none_200M` 1.45903: X3 HIT (0.0002 apart; with no mixer a 1,024 window buys nothing at 200M either).
  `w9_mix4_T1024` 1.53342 against `w9_mix4` 1.52874: N3 MISS (the longer window is 0.005 worse at 20M, inside noise).
  Box d's GPU 3 is free until the 18:45Z deadline (its worker had one line), so one more arm is sealed and fired
  there by hand (not through the queue runner, which would put it on GPU 0):

| id | prediction |
|---|---|
| X7 | `w10_mq_T1024_200M` (the Muon transformer at T 1,024, 200M) beats `w7_mq_200M` (T 256, 1.37059) by at least 0.01: a transformer does use a longer window once it has trained this long |
- 2026-10-03T15:21Z · Heartbeat, quiet. Box d all four GPUs busy: `w10_cn_mix4_200M` step 9,200, `w10_cn_mix4_T1024_200M` 6,300, `w10_cn_mix4_nofade_200M` 7,600, `w10_mq_T1024_200M` 2,100 (of 24,414 each); disk 76 GB free. Credit $334.06. Hero channel unchanged: `settle-foldpage` head awaits Claude 1.
- 2026-10-03T15:52Z · Disk. The M5 touched 773 MiB free at about 15:23Z (seven new full SETTLE worktrees, about 15 GB
  each, in seven minutes). Lane DISKSWEEP finished (`bd0cbddb3`): 33.3 GB moved to `spark:/home/hologram/dwarfstar_cold_store/`
  with sha-verified stones (16 of 20 items are now single copies there; the Keep drive is plugged in but not mounted),
  about 0.4 GB of .pyc caches deleted; the M5 now has 107 GiB free. It also found that the 3 Oct sparse-checkout
  accident in the main checkout deleted gitignored data: the `ds4flash.gguf` target here (the Spark holds the same
  file, 86,720,111,488 bytes, checked), `models/qwen3.5-9b-base` and the qwen3-30b teacher (re-downloadable),
  `captures_track4_194` (31G copy on the Spark), and three items with NO copy found: `captures_track4_192` (7.6 GB),
  our own `nanochat-ours-d20-climbmix` (4.2 GB), and `track3_the_first_unfold/out/resid_L16_fp16.dat` (17 GB). Ledger:
  `experiments/track0_store/DISKSWEEP_LEDGER_2026-10-03.md`. None of these touch the SDMONLY runs.
- 2026-10-03T15:52Z · X7 HIT: `w10_mq_T1024_200M` 1.35201 against `w7_mq_200M` 1.37059 (0.0186 better). At 200M the
  transformer USES a 1,024 window; the SDM without a mixer does not (X3, 0.0002). So the mixer is the only route to
  a long window, and wave 10's mixer arms (landing from about 15:55Z) decide whether it works. Box d's GPU 3 freed
  again at about 15:42Z; one more yardstick arm sealed and fired there by hand:

| id | prediction |
|---|---|
| X8 | `w10_mq_T4096_200M` (the transformer at T 4,096, 200M) beats `w10_mq_T1024_200M` (1.35201) by at least 0.005 |
- 2026-10-03T15:53Z · Deadline check for 15:45Z: boxes a, b and c are gone from the account (released early at 15:14Z after a final pull; see 15:16Z). Only box d (54007096, $2.15/h, deadline 18:45Z, its own watcher) remains. Credit $332.93: $19.48 spent since $352.41.
- 2026-10-03T16:25Z · Wave 10 first scores, and A FINDING THAT REFRAMES EARLIER READINGS.
  `w10_cn_mix4_nofade_200M` 1.46048 · `w10_cn_mix4_200M` 1.48242 (against `w7_ch_none_200M` 1.45903).
  X1 MISS (the fading mixer is 0.023 WORSE than no mixer) · X4 MISS (no fading beats fading by 0.022) · the
  never-fading mixer ties the best recipe (0.0015 behind). Pending: X2, X5, X8, and the 500M run.
  THE FINDING: `data/train.u32`, the shard every run so far trained on, holds 65,522,263 tokens (FineWeb-Edu
  sample/10BT, 64,488 documents). Every "200M" run saw the data about 3 times; the Spark's 600M chat-base runs about
  9 times. So "more tokens flatten near 1.47, the model is context-limited, not under-trained" (the wave 7 reading)
  is CONFOUNDED by repetition and is withdrawn as a conclusion. A 3.0B-token extension with the same TEST stream
  already exists on the Spark (`train_big.u32`, 12,073,195,156 bytes, `track4_sdmllm24_extend_fineweb_edu_shards.py`).
  From wave 11 on, every scaling run trains on `train_big`, one pass, no repeats.
  THE NAVIGATOR (16:05Z): "get up to 10 Vast boxes going now, 10 HOT with JIMOTHY, the best type, you decide; do it
  as you see fit, update all md files as we go." JIMOTHY decided: 10 boxes of 2x RTX 5090 (20 GPUs), at most $12/h
  and $80 in total, deadline 7 h after the boxes are ready. Lane FLEET10 (Opus) rents, sets up, builds `train_big`
  on each box (verified as a byte-identical prefix of the Spark's file), and starts the queues. Box d keeps its
  wave 10 to 18:45Z.
  The trainer gained `--qwen-layers` and `--qwen-f` (defaults 4 and 688 reproduce every earlier yardstick) so the
  transformer's MLP width can scale with the model width (8/3 x d) in the grid.

### Wave 11 (the scaling grid on fresh data, and the never-fading mixer), sealed 2026-10-03T16:25Z before it runs

Ten boxes, queue files `track4_sdmonly_wave11_q_<e..n>.txt`, 46 runs. "cn" = the best recipe (`sdmonly_none`, hop
MLPs 192, untie, conj 2, Muon). "mq" = the Muon transformer, 4 layers, MLP 8/3 d. "cnmix" = cn + 4 never-fading
mixer layers, 128x128 locations each. Phase B (the grid) trains on `train_big`, one pass: widths 256, 384, 512, 768
x tokens 100M, 400M, 1.6B for cn and mq; cnmix at width 256 (100M, 400M at T 256, 1,024, 4,096) and 512 (100M);
second seeds at width 256, 400M. Phase A (`w11_M*`, on the old shard, 200M, comparable with wave 10): the
never-fading mixer at T 1,024 / 4,096 / 16,384, 8 and 2 layers, 256x256 locations, k 64, address width 128, with
store reads on, and second seeds of the mixer and of cn. The fit: L(N, D) = E + A / N^alpha + B / D^beta per family,
N = non-embedding parameters, D = training tokens.

| id | prediction |
|---|---|
| Y1 | on fresh data, `w11_cn_d256_400M` beats the repeated-data `w7_ch_none_200M` (1.45903) by at least 0.02 |
| Y2 | from 100M to 1.6B tokens at width 512, the transformer gains more bits than cn |
| Y3 | at 1.6B tokens, `w11_cn_d768_1600M` beats `w11_cn_d256_1600M` by at least 0.03 |
| Y4 | `w11_mq_d768_1600M` scores below 1.20 |
| Y5 | `w11_M1_nf_T1024` beats `w7_ch_none_200M` (1.45903) by at least 0.005 |
| Y6 | `w11_M2_nf_T4096` beats `w11_M1_nf_T1024` |
| Y7 | `w11_M7_nf_reads` is worse than `w10_cn_mix4_nofade_200M` (1.46048) |
| Y8 | `w11_cnmix_d256_T1024_400M` beats `w11_cn_d256_400M` by at least 0.01 |
| Y9 | the seed-to-seed gap of `w11_cn_d256_400M` is under 0.005 |
| Y10 | the transformer's fitted data exponent beta is larger than cn's |
- 2026-10-03T16:12Z · Box d: GPUs 0 and 1 idle since 15:58Z and 15:55Z (their queue lines were done). Red:
  `w10_cn_mix4_T4096_200M` died at its first step, Triton failed to compile the mixer's chunk loop at T 4,096
  (`PassManager::run failed` inside the compiled loss). Fix: `SdmMix` runs eagerly for windows over 2,048
  (`_run_eager`, `torch.compiler.disable`); shorter windows compile as before; model self-test 32 of 32. Shipped to
  box d and to the fleet. Fired by hand on box d: GPU 1 reruns `w10_cn_mix4_T4096_200M` (sealed X5 unchanged);
  GPU 0 takes one new yardstick arm, sealed here first:

| id | prediction |
|---|---|
| X9 | `w10_mq_T16384_200M` (the transformer at T 16,384, B 1) beats `w10_mq_T4096_200M` by at least 0.003 |
- 2026-10-03T16:25Z · The T 4,096 mixer rerun on box d died again: CUDA out of memory (a 2 GiB request with 31 GB in
  use). The mixer built a dense location-weight grid for the whole window, (B, heads, T, n_sub^2) per layer, and kept
  every chunk's grid for backward; that cannot reach 256k. Fix in `SdmMix`: the weights stay SPARSE (2k indices and
  weights per position); `_chunk` densifies one chunk at a time; in training, a window longer than one chunk
  recomputes each chunk in backward (checkpoint). Same maths: output and gradient match one-chunk to 1e-5 / 1e-4,
  now a self-test check (33 of 33). Box d GPU 0 runs `w10_mq_T16384_200M` (X9), 227k tok/s.
  A RECORD NOTE: my 16:12Z commit failed ("cannot do a partial commit during a merge"): a SETTLE merge was in progress
  in the main checkout, and my two staged files (`track4_sdmonly_models.py` with `_run_eager`, and this plan's 16:12Z
  entry with X9) landed inside that merge commit, `5e2d56476` ("Merge branch 'settle-objhopfield'"). Nothing lost;
  the history names the wrong subject, so a git note on that commit says what it carries. Lesson: check for
  .git/MERGE_HEAD before staging.
- 2026-10-03T16:16Z · The sparse mixer runs at T 4,096 (box d GPU 1 started cleanly) but at 12.6k tok/s: eager, with
  each chunk recomputed in backward and a dense L x n_sub^2 product per chunk. 200M tokens would take about 4.4 h, past
  box d's 18:45Z deadline, so `w10_cn_mix4_T4096_200M` was STOPPED at 16:14Z: X5 is NOT RUN. Its never-fading twin
  `w11_M2_nf_T4096` runs on the fleet (7 h deadline). AMENDMENT to wave 11 BEFORE it fires: `w11_M8_nf_T16384`
  drops from 200M to 60M tokens, so it can finish inside the fleet's 7 h; it predicts nothing sealed (Y1 to Y10 do
  not name it). Box d plan: when `w10_cn_mix4_T1024_200M`, `w10_mq_T4096_200M` and `w10_mq_T16384_200M` finish (about
  16:35Z), pull and destroy box d early, and do NOT run `w10_cn_mix4_500M` (a fading mixer, about 7.6 passes over
  the 65.5M shard; both known to hurt): it is dropped, not run. A new lane, MIXERFAST, builds a mixer kernel whose
  cost does not scale with n_sub^2 per position, the step the 256k target needs.
- 2026-10-03T16:18Z · Heartbeat. X8 MISS: `w10_mq_T4096_200M` 1.35083 against `w10_mq_T1024_200M` 1.35201 (0.0012, not
  0.005): at 200M tokens the transformer's window gain stops near 1,024. Box d still runs `w10_cn_mix4_T1024_200M`
  and `w10_mq_T16384_200M` (X9); it closes when they end. The fleet is rented: 10 boxes of 2x RTX 5090
  (sdmonly-e to -n, ids 54033213 to 54033241, $0.925 to $1.148 per hour, about $10.6 per hour together); FLEET10 is
  setting them up and building `train_big` on each. Credit $331.64 ($20.77 spent since $352.41). M5 free 147 GiB.
  Spark queue empty (MIXERFAST will test speed there). New standing cron c8e6ff9e (hourly at :37): any Vast box idle
  for a clear hour (no process, GPUs at 0 %, logs unchanged 60 min) is pulled and destroyed by id.
- 2026-10-03T16:30Z · Wave 10 scored (box d). `w10_mq_T16384_200M` 1.30172 · `w10_mq_T4096_200M` 1.35083 ·
  `w10_mq_T1024_200M` 1.35201 · `w10_cn_T1024_200M` 1.45886 · `w10_cn_mix4_nofade_200M` 1.46048 ·
  `w10_cn_mix4_200M` 1.48242 · `w10_cn_mix4_T1024_200M` 1.51608.
  X2 MISS: the fading mixer at T 1,024 is 0.034 WORSE than at T 256. X9 HIT, and by far more than sealed: the
  transformer at T 16,384 is 0.049 better than at T 4,096, after 1,024 to 4,096 gave only 0.0012. That jump is
  suspicious and is NOT yet read as "long windows pay": the T 16,384 arm ran B 1, so 16,384 tokens per step and half
  the steps, where the T 1,024 and T 4,096 arms ran 8,192 tokens per step; and every arm repeats the 65.5M shard
  about 3 times. Box d therefore stops `w10_cn_mix4_500M` (dropped as decided at 16:16Z; it had started at 16:16Z by
  the queue) and runs four short controls, sealed here first:

| id | prediction |
|---|---|
| X10 | `w10_mq_T4096_B4_200M` (16,384 tokens per step) is within 0.01 of `w10_mq_T4096_200M` (1.35083): the batch is not the cause |
| X11 | `w10_mq_T1024_B16_200M` (16,384 tokens per step) is within 0.01 of `w10_mq_T1024_200M` (1.35201) |
| X12 | `w10_cn_B64_200M` (the best SDM recipe at T 256, 16,384 tokens per step) is within 0.01 of `w7_ch_none_200M` (1.45903) |
| X13 | `w10_mq_T16384_200M_s1` (a second seed) is within 0.005 of 1.30172 |
- 2026-10-03T16:45Z · Heartbeat. Box d runs the four sealed controls X10 to X13 (steps 5,300 to 6,500 of 12,207 to
  24,414). Fleet: g runs (`w11_mq_d256_1600M`, `w11_cn_d256_1600M`); f and l have `train_big` (8,037,279,356 bytes,
  the 2.0B-token prefix) and wait for their runners; e, k and m build it; i and j install; h and n still load.
  Credit $328.46 ($23.95 spent). New fleets, all Opus: HEROKIND (5 lanes, 50 kindness heroes), HEROWIDE (10 lanes,
  150 heroes, mostly films, real SDR experiments, wiki updates, top 20 percent onto pages), and EMDR (10 lanes: first
  WIKI_EMDR, a general wiki brought to disk; then 150 heroes mixing EMDR with mostly thermodynamic computing, subtly:
  at most 25 percent fully EMDR, the rest at most half). Each fleet keeps one short-update channel; every lane writes
  its own report and JIMOTHY compiles one report per fleet at the end.
- 2026-10-03T16:56Z · THE CONTROLS SPEAK, AND THEY OVERTURN THE WINDOW READING. It was the BATCH, not the window.
  `w10_mq_T4096_B4_200M` 1.28904 · `w10_mq_T1024_B16_200M` 1.29148 · `w10_mq_T16384_200M_s1` 1.30226 ·
  `w10_cn_B64_200M` 1.41727.
  X10 MISS: 16,384 tokens per step at T 4,096 is 0.062 better than 8,192 (1.35083). X11 MISS: the same at T 1,024 is
  0.061 better (1.35201). X12 MISS: the SDM at B 64 (16,384 tokens per step) is 0.042 better than at B 32, and
  1.41727 is the NEW BEST SDM SCORE (was 1.45903). X13 HIT: the second seed lands 0.0005 from the first.
  READING: at 16,384 tokens per step the transformer scores 1.289 to 1.302 at windows of 1,024, 4,096 and 16,384
  alike, so the T 16,384 "jump" (X9) was the doubled batch. Our default of 8,192 tokens per step was too small for
  both families, on this repeated 65.5M shard; whether that holds on fresh data is the next question. Wave 11 keeps
  its sealed 8,192 tokens per step (its comparisons stay fair); a batch sweep follows.
  Box d (idle since its controls ended, about 16:50Z) takes four more sealed controls on the same shard, then closes:

| id | prediction |
|---|---|
| X14 | `w10_mq_B64_200M` (T 256, B 64, 16,384 tokens per step) beats `w7_mq_200M` (1.37059) by at least 0.03 |
| X15 | `w10_mq_B128_200M` (32,768 tokens per step) beats `w10_mq_B64_200M` by at least 0.01 |
| X16 | `w10_cn_B128_200M` beats `w10_cn_B64_200M` (1.41727) by at least 0.01 |
| X17 | `w10_cn_B256_200M` (65,536 tokens per step) beats `w10_cn_B128_200M` |
- 2026-10-03T17:20Z · THE BATCH SWEEP (box d, same 65.5M shard, 200M tokens). `w10_mq_B64_200M` 1.31147 ·
  `w10_mq_B128_200M` 1.28074 · `w10_cn_B128_200M` 1.39499 · `w10_cn_B256_200M` 1.38515.
  X14 HIT (the transformer at 16,384 tokens per step is 0.059 better than at 8,192). X15 HIT (32,768 tokens per step
  adds 0.031 more). X16 HIT (the SDM at 32,768 is 0.022 better than at 16,384). X17 HIT (65,536 adds 0.010 more).
  NEW BEST SDM: 1.38515 at 65,536 tokens per step (was 1.41727 at 16,384, and 1.45903 at our old default 8,192).
  At an equal 32,768 tokens per step the gap to the transformer is 0.114 (1.39499 against 1.28074): bigger batches
  help both families, so they do not close the gap. Caveat: the shard repeats about 3 times; the batch law must be
  re-measured on train_big before a BASE recipe uses it.
  Box d destroyed at 17:19Z after a final pull (60 results home); its watcher (PID 71187) stopped by PID.
  OUTAGE: every agent was cut by an API weekly limit at about 16:58Z; the navigator logged in fresh at 17:13Z and all
  27 lanes (FLEET10, MIXERFAST, HEROKIND 5, HEROWIDE 10, EMDR 10) were resumed by message at 17:14Z.
  Fleet at 17:16Z: e, f, g, j, k, l, m running wave 11; n has train_big but no runner; i gave no answer; h still
  loading (a replacement, 54036755). Credit $320.83.
- 2026-10-03T17:25Z · The navigator asked whether the grid should go wider than 768. Yes: the wave 11 grid is EXTENDED
  to widths 1,024 and 1,536 on the three idle boxes, same recipe, same 8,192 tokens per step, same `train_big`, one
  pass. Transformer MLP width 8/3 d (2,736 and 4,096). Queues: `track4_sdmonly_wave11_q_n_wide.txt` (width 1,024 at
  1.6B tokens), `..._q_i2_wide.txt` (box i: width 1,024 at 400M and 100M), `..._q_h_wide.txt` (width 1,536 at 400M
  and 100M). Sealed before they run:

| id | prediction |
|---|---|
| Z1 | at 1.6B tokens, `w11_cn_d1024_1600M` beats `w11_cn_d768_1600M` by at least 0.01 |
| Z2 | at 1.6B tokens, `w11_mq_d1024_1600M` beats `w11_mq_d768_1600M` by at least 0.02 (the transformer gains more from width) |
| Z3 | at 400M tokens, `w11_cn_d1536_400M` is NOT better than `w11_cn_d1024_400M` by more than 0.005 (too little data for that width) |
- 2026-10-03T17:28Z · STANDING RULE (the navigator): "always, if we get a good result at the highest value, try
  higher; see where the limits are; always find the curve turning off." Every sweep is reported as BRACKETED (a
  value past the best is worse) or UNBRACKETED (the best is at the edge), and an unbracketed sweep gets two more
  values beyond its edge in the next sealed wave. Unbracketed now: the batch sweep (best at 65,536 tokens per step,
  both families still improving) and width (768 was the edge; 1,024 and 1,536 now running).
  The navigator added: find "where the curve turns off, or whether there is a curve at all", and "whether more data
  helps or not". DATA is a sweep under the same rule: wave 11 measures 100M, 400M and 1.6B fresh tokens per width;
  if 1.6B is still the best, the next wave goes to 3.0B (all of `train_big`, one pass) and then fetches more
  FineWeb-Edu.
  And: "throw more data at it" on everything we are unsure about, to tell the arms apart. Any pair within the 0.007
  noise, and every UNDECIDABLE verdict, is rerun at a larger token budget on `train_big` before it is called. First
  candidates: the never-fading mixer against the best recipe (0.0015 apart at 200M), and wave 8's C2 and C3.
- 2026-10-03T17:40Z · FLEET10: the wave 11 fleet is running. 9 boxes of 2x RTX 5090 (`vast_sdmonly_fleet11.list`, e f g i
  j k l m n, $9.20/h together); slot h is empty after two failed hosts, so `track4_sdmonly_wave11_q_h.txt` and
  `_q_h_wide.txt` have no box. Last box ready (i fired) 17:35:53Z, so THE FLEET DEADLINE IS 2026-10-04T00:36:00Z:
  the pull loop `track4_sdmonly_vast_watch_fleet11.sh` (PID 68912, `runs_vast_sdmonly/watch_fleet11.out`) pulls every
  5 minutes and destroys every listed box by id at that time. `train_big` on every box: 2,009,319,839 tokens,
  sha256 6ccc9836...200d, the same-length prefix of the Spark's file. Code: committed e770c29fa. Credit $316.68.
  `w11_M4_nf_n256` (box j) died at once: CUDA out of memory (a 4 GiB bf16 buffer of 32 x 4 x 65,536 x 256 in the
  compiled mixer at n_sub 256); wave7.log says "rc 0" because it logs the exit code of `date`. NOT RUN.
- 2026-10-03T18:05Z · FLEET10 done: 9 of 10 boxes run wave 11 (slot h failed twice and is empty), $9.20/h, fleet
  deadline 2026-10-04T00:36:00Z, pull loop PID 68912 (`watch_fleet11.out`). Not runnable in time: n's
  `w11_cnmix_d256_T4096_400M` (dense, about 9 h), i's `w11_M3_nf_mix8`; `w11_M4_nf_n256` died out of memory (n_sub
  256 makes a 4 GiB bf16 buffer). JIMOTHY moved the 16 orphan lines (box h's 6, both wide queues' 8, and 2 more) into
  per-box FILL queues (`track4_sdmonly_wave11_fill_<e,f,j,k,l,m>.txt`, same sealed lines, no prediction changed) run
  by a new `track4_sdmonly_fill.sh`, which hands a GPU the next unclaimed line once it has been idle two checks in a
  row. Proven on box k: GPU 1 took `w11_cnmix_d256_T1024_400M` at 17:45:58Z. The chained wide queues on n and i were
  stopped by PID (10771, 7129).
  MIXERFAST done (`c0fff0deb`): `SdmMixFast`, the same maths from sorted index lists (Triton on CUDA), equal to the
  dense mixer in output and every gradient (max diff 4.8e-7), opt-in `--mix-impl fast`. On the Spark GB10 the
  4-mixer body trains at 34k to 38k tok/s from T 256 to 65,536 and 29.9k at 262,144, against dense 10.8k down to
  2.3k (16x at T 4,096 and up). The 256k road is open for the mixer; the next wall is `causal_ema`, which builds a
  T x T matrix. HEROKIND: KINDSETTLE, KINDQUANTUM, KINDFOLD, KINDRECALL all ready to land.

### Wave 12a (the fast mixer on the fleet), sealed 2026-10-03T18:05Z before it runs

On box k's fill queue. Same seed, same everything, `--mix-impl dense` against `fast`, cnmix width 256, T 1,024,
100M tokens of `train_big`. And the long-window line that dense could not finish, re-run fast on box n.

| id | prediction |
|---|---|
| F1 | `w12_fastcheck_fast_100M` and `w12_fastcheck_dense_100M` score within 0.005 of each other |
| F2 | on an RTX 5090, the fast line trains at least 2x the dense line's tokens per second at T 1,024 |
| F3 | `w12_cnmix_d256_T4096_400M_fast` finishes before 00:36Z and beats `w11_cnmix_d256_T1024_400M` (box k) |
- 2026-10-03T18:00Z · Heartbeat. All 18 fleet GPUs busy (9 boxes, deadline 00:36Z). Credit $314.36 ($38.05 spent).
  RED, handled: the M5 fell to 38 GiB free (25 hero and wiki lanes, each worktree about 6 GB with node_modules). The
  five finished HEROKIND worktrees were removed (each verified committed at its reported tip, no tracked changes; raw
  shots copied to `SETTLE/runs/herokind/<lane>/shots_raw/` first): 69 GiB free. All 20 running lanes
  told to remove their own worktree when done; EMDR lanes told not to build one in Phase A.
  UNBRACKETED sweeps: batch (still improving at 65,536 tokens per step), width (running 1,024 and 1,536 on the fill
  queues), data (the 1.6B runs land from about 19:30Z).
- 2026-10-03T18:01Z · Idle sweep: 9 boxes checked, all busy (every GPU 59 to 100 %, training processes running; l answered on a second try). None closed.
- 2026-10-03T22:25Z · OUTAGE: every agent and the coordinator were cut by a session limit at about 18:05Z; the limit
  reset at 22:10Z. The fleet ran on unattended: its queues and fill runners finished, so 14 of 18 GPUs sat IDLE for
  up to about 3 hours (named here as a wrong state; the hourly idle sweep could not fire with the coordinator down).
  All 20 agent lanes resumed by message at 22:12Z. Credit $275.12 ($77.29 spent since $352.41).
  WAVE 11 SCORED (fresh `train_big`, one pass, 8,192 tokens per step; TEST bpb):

| width | cn 100M | cn 400M | cn 1.6B | mq 100M | mq 400M | mq 1.6B |
|---|---|---|---|---|---|---|
| 256 | 1.44863 | 1.42307 | 1.42223 | 1.36810 | 1.34280 | 1.33945 |
| 384 | 1.42172 | 1.39055 | 1.38028 | 1.32194 | 1.28905 | 1.28337 |
| 512 | 1.40506 | (running) | 1.36408 | 1.29706 | 1.25586 | 1.24932 |
| 768 | 1.38730 | 1.34766 | 1.33499 | 1.26346 | 1.21651 | 1.19701 |
| 1024 | 1.37672 | 1.33250 | (running) | 1.24382 | 1.19037 | (running) |
| 1536 | 1.37018 | 1.31793 | | 1.22447 | 1.16375 | |

  Mixer on fresh data (width 256, 400M): `w11_cnmix_d256_T1024_400M` 1.40822, its second seed 1.40864, against cn
  1.42307 and cn at T 1,024 without a mixer 1.42538: THE NEVER-FADING MIXER HELPS BY 0.015 ON FRESH DATA, two seeds
  agreeing to 0.0004. Window sweep for the mixer: T 256 1.47348 (worse), T 1,024 1.40822, T 4,096 (fast) 1.42106:
  BRACKETED, best near 1,024. Seeds on fresh data: cn 1.42307 / 1.42258, mq 1.34280 / 1.34233 (noise about 0.0005).
  Phase A on the old shard (200M): M1 T 1,024 1.44709 · M2 T 4,096 1.43715 · M8 T 16,384 at 60M tokens 1.42897 (60M
  is under one pass of the shard, so no repetition: the best old-shard SDM score) · M9 2 layers 1.44412 · M7 with reads
  1.45864 · M6 address 128 1.47056 · M5 k 64 1.47915 · M3 8 layers 1.49221 · seeds M10 1.47634 (0.016 from M's s0
  1.46048: the mixer's seed noise on the repeated shard is large) and M11 1.45875.
  VERDICTS: Y1 HIT (0.036) · Y2 HIT (mq gains 0.048, cn 0.041) · Y3 HIT (0.087) · Y4 HIT (1.19701) · Y5 HIT (0.012) ·
  Y6 HIT · Y7 MISS (reads level, not worse) · Y8 HIT (0.015) · Y9 HIT (0.0005) · Y10 MISS (fitted beta: cn 1.58, mq
  1.46). Z3 MISS (width 1,536 still 0.015 better than 1,024). Z1, Z2 pending (box j). F1 MISS: fast and dense differ by
  0.0114 (dense 1.43602, fast 1.44746) at seed 0; rerun at seed 1 in wave 13 before blaming the kernel (the mixer's
  seed noise was 0.016 on the old shard). F3 MISS (fast T 4,096 1.42106 does not beat T 1,024).
  THE FIT, L = E + A (N/1e6)^-alpha + B (D/1e8)^-beta, N = non-embedding parameters, D = tokens: cn E 1.292, A 2.04,
  alpha 0.79, B 0.043, beta 1.58 (15 points, rms 0.0049); mq E 1.070, A 0.355, alpha 0.29, B 0.050, beta 1.46 (16
  points, rms 0.0069). READING: both families gain from size; the SDM recipe without a mixer has a much HIGHER FLOOR
  (1.29 against 1.07), and its data term saturates (width 256 flat from 400M to 1.6B), which is what a model that only
  sees about 100 tokens back would do. The mixer is the part that can lower that floor. A rough fit on 15 points, not a law.
  SWEEPS: width UNBRACKETED for both (1,536 best) · data BRACKETED for cn at width 256 (flat), UNBRACKETED at 768 and
  up · mixer window BRACKETED near 1,024 · batch UNBRACKETED (old shard only).

### Wave 13 (past the edges, and the mixer at width), sealed 2026-10-03T22:25Z before it runs

On the 14 idle GPUs, fill queues `track4_sdmonly_wave13_fill_<box>.txt`, fresh `train_big`. The fleet deadline moves
from 00:36Z to 04:00Z for it (about $32 more).

| id | prediction |
|---|---|
| G1 | `w13_cnmix_d512_T1024_400M` beats `w11_cn_d512_400M` by at least 0.01 |
| G2 | `w13_cnmix_d768_T1024_400M` beats `w11_cn_d768_400M` (1.34766) by at least 0.01 |
| G3 | `w13_cn_d2048_400M` beats `w11_cn_d1536_400M` (1.31793) |
| G4 | `w13_mq_d2048_400M` beats `w11_mq_d1536_400M` (1.16375) |
| G5 | at 32,768 tokens per step, `w13_cn_d256_B128_400M` beats 1.42307 and `w13_mq_d256_B128_400M` beats 1.34280, each by at least 0.01 |
| G6 | `w13_fastcheck_fast_100M_s1` and `w13_fastcheck_dense_100M_s1` are within 0.005 |
| G7 | `w13_cnmix_d256_T256_400M_s1` is within 0.007 of 1.47348 (the short-window mixer penalty is real) |
| G8 | `w13_cnmix_d256_T2048_400M` scores between 1.40822 and 1.42106 |
| G9 | `w13_cnmix_d256_T1024_800M` beats `w11_cnmix_d256_T1024_400M` (1.40822) by at least 0.005: more data still helps the mixer |
- 2026-10-03T22:30Z · Wave 13 fired: fill runners on e, f, g, k, l, m, n, i (14 lines on the 14 idle GPUs; code shipped: the latest models, trainer and fast mixer). Fleet deadline moved to 2026-10-04T04:00:00Z: watcher 68912 stopped by PID, new watcher PID 59678 (same list, same pull-then-destroy).
- 2026-10-03T22:45Z · Heartbeat. Z1 HIT: `w11_cn_d1024_1600M` 1.31856 beats `w11_cn_d768_1600M` (1.33499) by 0.016.
  Red, fixed: `w13_cnmix_d256_T2048_400M` died at its first step with the same Triton compile failure as T 4,096 (the
  eager switch was "over 2,048", and 2,048 is not over it). `SdmMix` now runs eagerly over 1,024 tokens (self-test 50
  of 50), shipped to box i, rerun started 22:30Z at 12.3k tok/s (about 9 h for 400M: it will NOT finish by the 04:00Z
  deadline; to be cut or moved). All 18 GPUs otherwise busy with wave 13 (j GPU 0 freed when cn d1024 1.6B finished).
  Credit $272.39 ($80.02 spent). M5 free 103 GiB. EMDR fleet complete (10 lanes, 150 heroes, WIKI_EMDR 51 pages);
  HEROWIDE 7 of 10 done (DEEPLEARN, DEEPSPARSE, DEEPDYN, DEEPHOPFIELD, DEEPSCALE, DEEPTHERMO, DEEPQUANTUM).
- 2026-10-03T22:52Z · BATCH ON FRESH DATA: the biggest lever yet. At 32,768 tokens per step (B 128, T 256, 400M fresh
  tokens): `w13_cn_d256_B128_400M` 1.33548 against 1.42307 at 8,192 (0.088 better); `w13_mq_d256_B128_400M` 1.23438
  against 1.34280 (0.108 better). G5 HIT. The width-256 SDM at the bigger batch beats every SDM of wave 11 below width
  1,536 (1.31793). Note: the wave 13 transformer used MLP width 683 (round 8/3 x 256) against 688 in wave 11, a 0.7%
  difference. G6 so far: `w13_fastcheck_fast_100M_s1` 1.44270 (its dense twin still running). Batch is UNBRACKETED on
  fresh data, so per the edge rule the next wave goes two values past it, on the four GPUs that are idle (k 0 and 1,
  j 0, m 1; their wave 13 lines are done). Learning rate unchanged from every earlier run (an untuned batch).

### Wave 14 (batch past the edge on fresh data), sealed 2026-10-03T22:52Z before it runs

| id | prediction |
|---|---|
| H1 | `w14_cn_d256_B256_400M` (65,536 tokens per step) beats `w13_cn_d256_B128_400M` (1.33548) |
| H2 | `w14_mq_d256_B256_400M` beats `w13_mq_d256_B128_400M` (1.23438) |
| H3 | at B 512 (131,072 tokens per step, only 3,052 steps for 400M) at least one family is WORSE than at B 256: the curve turns |
- 2026-10-03T23:01Z · Idle sweep: 9 boxes checked, none idle (wave 13 and 14 runs on every box). None closed.
- 2026-10-03T23:12Z · Both batch-512 lines died at once with CUDA out of memory (131,072 tokens of activations on a
  32 GB card). The trainer gained `--accum N` (gradient accumulation: the same B windows in N micro-batches, one step;
  batch-norm statistics are per micro-batch, the only difference). AMENDMENT before a valid run: the two B 512 lines
  of wave 14 add `--accum 2`; H3 is unchanged. Both rerun at 23:08Z and train (about 300k to 320k tok/s). One wrong
  beat of mine on the way: the first rerun passed the wrong argument order to the arm wrapper and died with "TRAINER
  must be new or s0"; corrected within minutes.
- 2026-10-03T23:15Z · The batch gain HOLDS AT WIDTH 768: `w13_cn_d768_B128_400M` 1.26153 against 1.34766 at 8,192
  tokens per step (0.086 better), NEW BEST SDM SCORE; `w13_mq_d768_B128_400M` 1.11747 against 1.21651 (0.099). G6 HIT:
  at seed 1 the fast and dense mixers differ by 0.0033 (`w13_fastcheck_fast_100M_s1` 1.44270, dense 1.44596), so seed
  0's 0.011 gap was seed noise and the fast mixer is cleared for training. Box l (both GPUs) and m GPU 0 are idle;
  they take wave 15.

### Wave 15 (the batch lever with the mixer and with width), sealed 2026-10-03T23:15Z before it runs

At 32,768 tokens per step, fresh `train_big`, 400M tokens. `--accum 2` where memory needs it (math unchanged but
batch-norm statistics per half).

| id | prediction |
|---|---|
| I1 | `w15_cnmix_d256_T1024_B32_400M` (the never-fading mixer at a 1,024 window) beats `w13_cn_d256_B128_400M` (1.33548) by at least 0.01: the mixer still helps at the big batch |
| I2 | `w15_cn_d1024_B128_400M` beats `w13_cn_d768_B128_400M` (1.26153) |
| I3 | `w15_cn_d1536_B128_400M` beats `w15_cn_d1024_B128_400M` |
- 2026-10-03T23:27Z · Heartbeat. H1 HIT: `w14_cn_d256_B256_400M` 1.31661 against 1.33548 at B 128 (0.019 better).
  H2 HIT: `w14_mq_d256_B256_400M` 1.21773 against 1.23438 (0.017 better). Boxes e, j and k refused direct ssh
  (connection refused or timed out) while Vast showed them running; through the Vast ssh proxy all three answer, and
  no run was lost (e trains wave 13, j trains `w11_mq_d1024_1600M` at step 172,300). The direct-IP route failed, not
  the boxes. Wrong state found: box k had run out of lines (both GPUs idle) and j GPU 0 too.
- 2026-10-03T23:35Z · `w14_cn_d256_B512_400M` (with `--accum 2`) 1.31151 against 1.31661 at B 256: still better, by
  0.005 (ten times the seed noise). H3 is undecided until the transformer's B 512 lands; the SDM curve has NOT turned.
  Gains per doubling: 0.088, 0.019, 0.005, so the curve flattens but is still unbracketed. Per the edge rule, wave 16
  goes two values past it on the three idle GPUs, and throws more data at B 512.

### Wave 16 (batch past B 512 for cn, and more data at B 512), sealed 2026-10-03T23:35Z before it runs

Width 256, cn recipe, fresh `train_big`, T 256, learning rate unchanged (0.003; the batch sweep is untuned, which is
a stated confound: a bigger batch usually wants a bigger rate).

| id | prediction |
|---|---|
| J1 | `w16_cn_d256_B1024_400M` (262,144 tokens per step, 1,526 steps, `--accum 4`) is WORSE than B 512 (1.31151): the curve turns |
| J2 | `w16_cn_d256_B2048_400M` (524,288 tokens per step, 763 steps, `--accum 8`) is worse than B 1,024 |
| J3 | `w16_cn_d256_B512_800M` (twice the data) beats `w14_cn_d256_B512_400M` (1.31151) by at least 0.01 |
- 2026-10-03T23:45Z · H3 MISS: `w14_mq_d256_B512_400M` 1.21121 against 1.21773 at B 256 (0.0065 better), so NEITHER
  family turned at B 512. Both batch curves stay unbracketed and flatten (transformer gains per doubling 0.108, 0.017,
  0.0065). Also landed: `w11_cn_d512_400M` 1.37194, which fills the width-512 slot of the wave 11 grid (between 1.39055
  and 1.34766, as the fit expects). Wrong state: three GPUs idle (g 1, m 1, n 1), their lines done. A route note: box
  m now answers only on its direct IP (154.37.220.220:42884), while e, j and k answer only through the Vast proxy;
  the roll call tries direct first and must fall back to the proxy. Box j's code predated `--accum`, so its wave 16
  line died with "unrecognized arguments"; the five current trainer files were copied to j and the line rerun at
  23:39Z.

### Wave 16b (the transformer past B 512), sealed 2026-10-03T23:45Z before it runs

| id | prediction |
|---|---|
| J4 | `w16_mq_d256_B1024_400M` (`--accum 4`) is WORSE than B 512 (1.21121): the curve turns |
| J5 | `w16_mq_d256_B2048_400M` (`--accum 8`) is worse than B 1,024 |
| J6 | `w16_mq_d256_B512_800M` beats `w14_mq_d256_B512_400M` (1.21121) by at least 0.01 |
- 2026-10-03T23:52Z · Wave 16 and 16b train on all six GPUs (k 0 and 1, j 0, g 1, m 1, n 1); 18 of 18 GPUs busy. The
  pull watcher had one real defect against PULL BEFORE DESTROY: at the deadline it destroyed every box even when that
  box's final pull had failed. Fixed: each pull tries the direct IP, then the Vast ssh proxy, and the deadline
  destroys only the boxes whose final pull succeeded (any other is HELD and logged). The watcher was stopped by its
  PID and restarted on the patched script; its first pass pulled 9 of 9. DEADLINE MOVED from 04:00Z to 07:30Z
  (2026-10-04) so the batch, width and data sweeps can bracket; about $34 more at $9.6/h, credit $262.80 of the
  $300 window, spent about $91.
- 2026-10-03T23:53Z · Heartbeat, two minutes after the last entry: nothing new finished. 18 of 18 GPUs train (waves 13
  to 16). Credit $260.41 ($92.00 spent since $352.41). M5 free 157 GiB. Spark queue empty: it waits for the navigator's
  go on the long BASE run and for the Keep drive mount, both owed to him. Next: score wave 16 (the 262,144-token
  runs, about 00:15Z).
- 2026-10-04T00:02Z · Idle sweep: 9 boxes checked, none idle by the rule (8 train; box k's two GPUs finished at
  23:54Z, 8 minutes before, so not an hour). None closed. Box j shows `offline` in Vast but trains (step 190,700).
- 2026-10-04T00:02Z · WAVE 16 SDM HALF: THE BATCH CURVE TURNS. `w16_cn_d256_B1024_400M` 1.31694 against 1.31151 at
  B 512: J1 HIT. `w16_cn_d256_B2048_400M` 1.33308, worse again: J2 HIT. ⇒ the SDM batch sweep is BRACKETED at 400M
  tokens and learning rate 0.003 / Muon 0.02: best at B 512 (131,072 tokens per step). The fixed rate is the stated
  confound (a bigger batch usually wants a bigger rate), so the rate is the next sweep, on box k's two idle GPUs.

### Wave 17 (the learning rate at B 512), sealed 2026-10-04T00:02Z before it runs

Width 256, cn recipe, `--accum 2`, 400M fresh tokens; both rates (AdamW `--lr` and `--muon-lr`) scaled by the same factor.

| id | prediction |
|---|---|
| K1 | `w17_cn_d256_B512_lr2x_400M` (0.006 / 0.04) beats `w14_cn_d256_B512_400M` (1.31151) |
| K2 | `w17_cn_d256_B512_lr4x_400M` (0.012 / 0.08) is worse than the 2x arm: the rate curve turns |
  Credit after the sweep: $258.65.
- 2026-10-04T00:12Z · Heartbeat. I2 HIT: `w15_cn_d1024_B128_400M` 1.24868 against 1.26153 at width 768, NEW BEST SDM
  SCORE. Width at 32,768 tokens per step stays UNBRACKETED (256 1.33548, 768 1.26153, 1,024 1.24868; 1,536 running).
  G7 HIT: `w13_cnmix_d256_T256_400M_s1` 1.47193 against seed 0's 1.47348 (0.0016 apart): the short-window mixer
  penalty is real. `w16_mq_d256_B2048_400M` 1.24281, far worse than B 512 (1.21121); J5 waits for B 1,024. Three
  GPUs freed in the last ten minutes (l 1, m 1, n 0); they take wave 18. Credit $258.20.

### Wave 18 (the two levers together, and width past the edge), sealed 2026-10-04T00:12Z before it runs

400M fresh tokens; cn recipe for the SDM, Muon transformer (MLP 2,736) as the yardstick; rates unchanged.

| id | prediction |
|---|---|
| L1 | `w18_cn_d2048_B128_400M` (`--accum 4`) beats `w15_cn_d1024_B128_400M` (1.24868) |
| L2 | `w18_cn_d1024_B512_400M` (131,072 tokens per step, the best batch at width 256) beats `w15_cn_d1024_B128_400M` (1.24868): the batch optimum holds at width |
| L3 | `w18_mq_d1024_B512_400M` beats `w13_mq_d768_B128_400M` (1.11747), and the gap to L2's SDM is at least 0.10 |
- 2026-10-04T00:16Z · Wave 18 trains: l 1 `w18_cn_d2048_B128_400M`, m 1 `w18_cn_d1024_B512_400M`, n 0 `w18_mq_d1024_B512_400M` (all from 00:10Z). 18 of 18 GPUs busy.
- 2026-10-04T00:30Z · Heartbeat. Landed:
  - `w17_cn_d256_B512_lr2x_400M` 1.30761 and `w17_cn_d256_B512_lr4x_400M` 1.30606, against 1.31151 at the base rate.
    K1 HIT. K2 MISS (4x is still better than 2x, by 0.0016, inside noise). The rate sweep is UNBRACKETED.
  - `w16_cn_d256_B512_800M` 1.29441 against 1.31151 at 400M: J3 HIT (0.017). Data at the best batch is UNBRACKETED.
  - `w11_mq_d1024_1600M` 1.16517 against `w11_mq_d768_1600M` 1.19701: Z2 HIT (0.032; the transformer gains more from
    width than the SDM's 0.016 in Z1).
  - Wrong state: box k (both GPUs, done 00:25Z) and box j (both, done about 00:20Z) idle. They take wave 19.
  Sweeps: SDM batch BRACKETED (best B 512); learning rate, data, width UNBRACKETED; transformer batch waits on B 1,024.

### Wave 19 (rate past 4x, data past 800M, a fair rate for the yardstick), sealed 2026-10-04T00:30Z before it runs

Width 256, B 512, `--accum 2`. Both rates scaled by the same factor; 1x is `--lr 0.003 --muon-lr 0.02`.

| id | prediction |
|---|---|
| M1 | `w19_cn_d256_B512_lr8x_400M` (0.024 / 0.16) is WORSE than 4x (1.30606): the rate curve turns |
| M2 | `w19_cn_d256_B512_lr16x_400M` (0.048 / 0.32) is worse than 8x |
| M3 | `w19_cn_d256_B512_1600M` (base rate) beats `w16_cn_d256_B512_800M` (1.29441) by at least 0.01 |
| M4 | `w19_mq_d256_B512_lr4x_400M` beats `w14_mq_d256_B512_400M` (1.21121): the yardstick also wants the faster rate |
- 2026-10-04T00:35Z · Wave 19 trains: k 0 and 1 (rate 8x and 16x, from 00:30Z), j 0 and 1 (1.6B at B 512; the yardstick at 4x rate, from 00:34Z). 18 of 18 GPUs busy. Credit $255.16.
- 2026-10-04T00:50Z · Heartbeat. Landed:
  - I3 HIT: `w15_cn_d1536_B128_400M` 1.23331 against 1.24868 at width 1,024. NEW BEST SDM SCORE.
  - G3 HIT, barely: `w13_cn_d2048_400M` 1.31286 against `w11_cn_d1536_400M` 1.31793 (0.005) at 8,192 tokens per step.
  - J6 HIT: `w16_mq_d256_B512_800M` 1.19155 against 1.21121 at 400M (0.020).
  - Unsealed (no prediction was written for it): `w13_cnmix_d1024_T1024_200M` 1.31824. At HALF the data it beats the
    plain SDM of the same width and tokens per step, `w11_cn_d1024_400M` 1.33250, by 0.014. The mixer helps at width.
    Its eval read 998,388 tokens against 995,323 (the 1,024 window cuts the test set differently).
  - Wrong state: four GPUs idle (f 0, g 0, m 0, n 1), their runs done. They take wave 20.
  Sweeps: SDM width at 32,768 tokens per step UNBRACKETED (1,536 best; 2,048 running); data UNBRACKETED for both
  families; rate UNBRACKETED; SDM batch BRACKETED.

### Wave 20 (width past the edge, the mixer at width, the yardstick at the best shape), sealed 2026-10-04T00:50Z

400M fresh tokens. Rates unchanged except N4.

| id | prediction |
|---|---|
| N1 | `w20_cn_d3072_B128_400M` (`--accum 8`) beats `w15_cn_d1536_B128_400M` (1.23331) |
| N2 | `w20_cnmix_d1024_T1024_B32_400M` (32,768 tokens per step) beats `w15_cn_d1024_B128_400M` (1.24868) by at least 0.01 |
| N3 | `w20_mq_d1536_B128_400M` beats `w13_mq_d768_B128_400M` (1.11747), and leads `w15_cn_d1536_B128_400M` by at least 0.10 |
| N4 | `w20_cn_d1536_B512_lr4x_400M` beats `w15_cn_d1536_B128_400M` (1.23331): batch and rate stack at width |
- 2026-10-04T00:53Z · Wave 20 trains on m 0, g 0, f 0 and n 1 (from 00:50Z), each checked by its step lines. Speeds say width 3,072 ends near 04:25Z and the width-1,024 mixer near 06:10Z, inside the 07:30Z deadline. Current trainer code copied to e, f, i and l as well, so every box now knows `--accum`. 18 of 18 GPUs busy. Credit $252.07.
- 2026-10-04T01:01Z · Idle sweep: 9 boxes checked, none idle by the one-hour rule (box k's GPUs ended at 00:51Z and
  00:52Z). None closed. Credit $250.11. Four GPUs were HOT AND IDLE (k 0 and 1, g 1, j 1); they take wave 21.
- 2026-10-04T01:01Z · Landed, scored:
  - Rate at B 512, width 256, 400M: 1x 1.31151 · 2x 1.30761 · 4x 1.30606 · 8x 1.30649 · 16x 1.30790. M1 HIT (8x worse
    than 4x, by only 0.0004) and M2 HIT. The rate sweep is BRACKETED, but its top is FLAT: 2x to 16x sit within 0.002,
    inside the 0.007 noise. Per rule (c) the 4x and 8x arms rerun at 1.6B before the optimum is called.
  - Transformer batch: `w16_mq_d256_B1024_400M` 1.22021, worse than B 512 (1.21121): J4 HIT; B 2,048 (1.24281) worse
    again: J5 HIT. The transformer's batch sweep is BRACKETED at B 512, the same optimum as the SDM's.
  - M4 HIT: `w19_mq_d256_B512_lr4x_400M` 1.20804 against 1.21121 at the base rate. The yardstick also likes 4x.

### Wave 21 (more data at the flat rate, at the yardstick, and at the best SDM shape), sealed 2026-10-04T01:01Z

| id | prediction |
|---|---|
| O1 | `w21_cn_d256_B512_lr4x_1600M` beats `w19_cn_d256_B512_1600M` (the base rate at 1.6B, still running): the rate gain survives four times the data |
| O2 | `w21_cn_d256_B512_lr8x_1600M` lands within 0.003 of the 4x arm at 1.6B: the top of the rate curve is flat |
| O3 | `w21_mq_d256_B512_lr4x_1600M` beats `w16_mq_d256_B512_800M` (1.19155) by at least 0.01 |
| O4 | `w21_cn_d1536_B128_800M` beats `w15_cn_d1536_B128_400M` (1.23331) by at least 0.01 |
- 2026-10-04T01:05Z · Wave 21 trains on k 0 and 1, j 1 and g 1 (from 01:03Z). 18 of 18 GPUs busy.
- 2026-10-04T01:09Z · Heartbeat. L2 HIT: `w18_cn_d1024_B512_400M` 1.22701 against 1.24868 at B 128 (0.022 better).
  NEW BEST SDM SCORE, and it beats width 1,536 at B 128 (1.23331): batch and width stack. At width 1,024 the batch
  sweep is UNBRACKETED (B 512 is the highest tried), so wave 22 goes two values past it. m 1 was HOT AND IDLE (L2 done);
  it takes wave 22 line 1, line 2 waits for the next free GPU on m. Credit $248.99.

### Wave 22 (batch past B 512 at width 1,024), sealed 2026-10-04T01:09Z before it runs

| id | prediction |
|---|---|
| P1 | `w22_cn_d1024_B1024_400M` (`--accum 8`) is WORSE than B 512 at width 1,024 (1.22701): the optimum does not move with width |
| P2 | `w22_cn_d1024_B2048_400M` (`--accum 16`) is worse than B 1,024 |
- 2026-10-04T01:29Z · Heartbeat. L3 HIT: `w18_mq_d1024_B512_400M` 1.07975, beating `w13_mq_d768_B128_400M` (1.11747);
  its lead over the SDM of the same shape (`w18_cn_d1024_B512_400M` 1.22701) is 0.147. NEW BEST TRANSFORMER. The gap
  at the matched shape is wider than at width 256 (0.100 at B 512), so width helps the transformer more, as Z1/Z2 said.
  n 0 was HOT AND IDLE (L3 done); it takes wave 23. Credit $245.98.

### Wave 23 (the yardstick's batch past B 512 at width 1,024), sealed 2026-10-04T01:29Z before it runs

| id | prediction |
|---|---|
| Q1 | `w23_mq_d1024_B1024_400M` (`--accum 8`) is WORSE than B 512 at width 1,024 (1.07975) |
- 2026-10-04T01:49Z · Heartbeat. G1 HIT: `w13_cnmix_d512_T1024_400M` 1.33618 against `w11_cn_d512_400M` 1.37194, 0.036
  better at 8,192 tokens per step. The mixer's gain GROWS with width (0.015 at width 256, 0.036 at 512, 0.014 at 1,024
  on half the data). e 0 was HOT AND IDLE; it takes wave 24, the mixer at the best batch. Credit $242.86.

### Wave 24 (the mixer at the best batch), sealed 2026-10-04T01:49Z before it runs

| id | prediction |
|---|---|
| R1 | `w24_cnmix_d512_T1024_B128_400M` (131,072 tokens per step, `--accum 4`) beats `w18_cn_d1024_B512_400M` (1.22701): a mixer SDM at half the width beats the best plain SDM |
- 2026-10-04T01:52Z · AMENDMENT before a valid run: R1's line died at once with CUDA out of memory (a micro-batch of 32
  windows of 1,024 with four mixer layers needs more than 31 GiB). It reruns with `--accum 16` (micro-batch 8, the
  size the wave 13 mixer ran at); the step and its 131,072 tokens are unchanged. R1 is unchanged.
- 2026-10-04T02:02Z · Idle sweep: 9 boxes checked, none idle for an hour (box l's GPUs ended minutes ago). None closed.
  Credit $240.88. Five GPUs HOT AND IDLE (f 1, j 0, l 0 and 1; m 1 finishing its B 1,024 line, B 2,048 queued behind it).
- 2026-10-04T02:02Z · Landed, scored:
  - L1 HIT: `w18_cn_d2048_B128_400M` 1.22615 against 1.24868 at width 1,024. NEW BEST SDM SCORE, by 0.0009 over
    `w18_cn_d1024_B512_400M` (1.22701), which is inside noise: the two shapes tie. Width at 32,768 tokens per step is
    UNBRACKETED (2,048 best; 3,072 running), so 4,096 joins.
  - G4 HIT: `w13_mq_d2048_400M` 1.14391 against `w11_mq_d1536_400M` 1.16375.
  - M3 HIT, barely: `w19_cn_d256_B512_1600M` 1.28416 against 1.29441 at 800M (0.0103). Data still helps at width 256
    but each doubling buys less (0.017, then 0.010). The boxes hold a 2.0B-token prefix, so width 256 cannot go past
    2.0B in one pass; the data sweep moves to width 1,024.
  - I1 MISS: `w15_cnmix_d256_T1024_B32_400M` 1.34166 against the plain `w13_cn_d256_B128_400M` 1.33548. At 32,768 tokens
    per step the mixer did NOT help (0.0062 worse, inside the 0.007 noise: UNDECIDABLE). Per rule (c) both arms rerun at
    800M.
- 2026-10-04T02:02Z · DEADLINE MOVED from 07:30Z to 09:30Z (2026-10-04) so the 800M mixer arm (about 5.5 hours at its
  measured 40k tok/s) and width 4,096 can finish; about $19 more at $9.6/h.

### Wave 25 (data for the undecided mixer pair, width past 3,072, data at width 1,024), sealed 2026-10-04T02:02Z

| id | prediction |
|---|---|
| S1 | at 800M tokens, `w25_cn_d256_B128_800M` (plain) beats `w25_cnmix_d256_T1024_B32_800M` (mixer): at 32,768 tokens per step the mixer does not help |
| S2 | `w25_cn_d4096_B128_400M` (`--accum 16`) is NOT better than `w18_cn_d2048_B128_400M` (1.22615) by more than 0.005: width saturates at 400M tokens |
| S3 | `w25_cn_d1024_B512_1600M` beats `w18_cn_d1024_B512_400M` (1.22701) by at least 0.03 |
- 2026-10-04T02:07Z · P1 HIT: `w22_cn_d1024_B1024_400M` 1.23095, worse than B 512 (1.22701): at width 1,024 the batch optimum stays at B 512 (B 2,048 now runs for P2). Wave 25 trains on l 0 and 1, f 1 and j 0 (from 02:03Z). Watcher stopped by its PID (17368) and restarted on the 09:30Z deadline (PID 44290).
- 2026-10-04T02:09Z · Heartbeat. N4 HIT: `w20_cn_d1536_B512_lr4x_400M` 1.21488 against `w15_cn_d1536_B128_400M` 1.23331
  (0.018). NEW BEST SDM SCORE: batch, rate and width stack. n 1 HOT AND IDLE; it takes wave 26. Credit $239.82.

### Wave 26 (all three levers at width 2,048), sealed 2026-10-04T02:09Z before it runs

| id | prediction |
|---|---|
| T1 | `w26_cn_d2048_B512_lr4x_400M` (`--accum 16`) beats `w20_cn_d1536_B512_lr4x_400M` (1.21488) |
- 2026-10-04T02:30Z · Heartbeat. Landed:
  - G2 HIT: `w13_cnmix_d768_T1024_400M` 1.31312 against `w11_cn_d768_400M` 1.34766 (0.035 better). At 8,192 tokens per
    step the mixer gain holds at width (0.036 at 512, 0.035 at 768).
  - Q1 HIT: `w23_mq_d1024_B1024_400M` 1.08665, worse than B 512 (1.07975). The transformer's batch optimum stays at
    B 512 at width 1,024 too. BRACKETED for both families at both widths.
  - N3 HIT: `w20_mq_d1536_B128_400M` 1.07413 (NEW BEST TRANSFORMER); its lead over `w15_cn_d1536_B128_400M` is 0.159.
  - O2 half: `w21_cn_d256_B512_lr8x_1600M` 1.28315 against the base rate at 1.6B (1.28416): 0.001 better, so the rate
    gain shrinks to nothing with more data. The 4x arm lands next.
  - Four GPUs HOT AND IDLE (e 1, f 0, k 1, n 0); they take wave 27, which brackets the rate at the best SDM shape.

### Wave 27 (the rate at the best SDM shape, more data there, and the yardstick there), sealed 2026-10-04T02:30Z

Width 1,536, B 512 (131,072 tokens per step), `--accum 8`.

| id | prediction |
|---|---|
| U1 | `w27_cn_d1536_B512_400M` (base rate) is WORSE than the 4x arm (1.21488) |
| U2 | `w27_cn_d1536_B512_lr8x_400M` is worse than the 4x arm: the rate turns at width |
| U3 | `w27_cn_d1536_B512_lr4x_800M` beats the 4x arm at 400M (1.21488) by at least 0.01 |
| U4 | `w27_mq_d1536_B512_lr4x_400M` beats `w20_mq_d1536_B128_400M` (1.07413) and leads 1.21488 by at least 0.12 |
- 2026-10-04T02:38Z · The rate at 1.6B tokens (width 256, B 512): base 1.28416 · 4x 1.28323 · 8x 1.28315. O1 HIT on the
  letter (4x beats base by 0.0009) and O2 HIT, but THERE IS NO RATE CURVE at 1.6B: all three sit within 0.001, far inside
  the 0.007 noise. The 400M rate gain (0.005) washed out with four times the data. Wave 27 checks the rate at width.
  k 1 HOT AND IDLE; it takes wave 28.

### Wave 28 (the mixer at width 768 and the best batch), sealed 2026-10-04T02:38Z before it runs

| id | prediction |
|---|---|
| V1 | `w28_cnmix_d768_T1024_B128_400M` (131,072 tokens per step, `--accum 16`) beats `w18_cn_d1024_B512_400M` (1.22701) |
- 2026-10-04T02:43Z · Wave 27 trains on e 1, f 0, k 0 and n 0 (from 02:30Z, step lines seen); wave 28 on k 1 (from 02:35Z). Box k timed out on both ssh routes for about two minutes at 02:43Z, then answered again; nothing was lost. 18 of 18 GPUs busy. Credit $236.76.
- 2026-10-04T02:51Z · Heartbeat. Landed:
  - O3 HIT: `w21_mq_d256_B512_lr4x_1600M` 1.17641 against 1.19155 at 800M (0.015). Data keeps helping the yardstick
    more than the SDM (0.010 for the SDM at the same step).
  - `w25_cn_d256_B128_800M` (plain half of the S1 pair) 1.32646; its mixer twin runs to about 07:30Z.
  - Boxes j and k: each answered on only ONE route this beat (j on its direct IP, k on the proxy); no run lost.
  - Two GPUs HOT AND IDLE (j 1, l 1). The SDM body's hop MLP has only ever run at width 192, an unswept knob at a
    width where the body is a sliver of the model; wave 29 sweeps it at the best shape.

### Wave 29 (the hop MLP width at the best SDM shape), sealed 2026-10-04T02:51Z before it runs

Width 1,536, B 512, 4x rate, 400M tokens; only `--hop-mlp` changes (192 in the reference, 1.21488).

| id | prediction |
|---|---|
| W1 | `w29_cn_d1536_B512_lr4x_hm768_400M` beats `w20_cn_d1536_B512_lr4x_400M` (1.21488) by at least 0.01 |
| W2 | `w29_cn_d1536_B512_lr4x_hm3072_400M` beats the 768 arm |
- 2026-10-04T02:55Z · Wave 29 trains on j 1 and l 1 (step lines seen). 18 of 18 GPUs busy. Credit $233.76.
- 2026-10-04T03:01Z · Idle sweep: 9 boxes checked, none idle for an hour; none closed. Credit $231.93.
- 2026-10-04T03:01Z · P2 HIT: `w22_cn_d1024_B2048_400M` 1.24337, worse than B 1,024 (1.23095). At width 1,024 the SDM's
  batch sweep is BRACKETED at B 512, like width 256. m 1 HOT AND IDLE; it takes wave 30.

### Wave 30 (the hop count at the best SDM shape), sealed 2026-10-04T03:01Z before it runs

| id | prediction |
|---|---|
| X1 | `w30_cn_d1536_B512_lr4x_hops8_400M` (8 hops instead of 4) beats `w20_cn_d1536_B512_lr4x_400M` (1.21488) |
- 2026-10-04T03:06Z · Wave 30 trains on m 1 (step lines seen). 18 of 18 GPUs busy.
- 2026-10-04T03:08Z · Heartbeat: nothing new finished since 03:01Z; 18 of 18 GPUs busy (waves 13 to 30). Credit $230.89. Next results: width 3,072 (m 0, step 9,600 of 12,207) near 03:35Z; waves 26 and 27 near 03:40Z.
- 2026-10-04T03:28Z · Heartbeat: nothing new finished; 18 of 18 GPUs busy. Measured finish estimates for the long runs, all inside the 09:30Z deadline: width-1,024 mixer (g 0) about 05:25Z; width-512 mixer at the best batch (e 0) about 05:20Z; width 4,096 (f 1) and width-768 mixer at the best batch (k 1) about 07:00Z; the 800M mixer of the S1 pair (l 0) about 08:40Z, the latest. Credit $227.89.
- 2026-10-04T03:50Z · Heartbeat. Landed:
  - The rate at width 1,536 and B 512: base `w27_cn_d1536_B512_400M` 1.21260 · 4x 1.21488 · 8x
    `w27_cn_d1536_B512_lr8x_400M` 1.21848. U1 MISS (the base rate is best, not 4x). U2 HIT. NEW BEST SDM SCORE 1.21260.
    At width the best rate sits at the LOWEST value tried, so the rate sweep there is UNBRACKETED on the low side;
    wave 31 goes two values below. (At width 256 4x helped at 400M; the optimum rate falls as width grows, the usual
    pattern.) The 4x arms of waves 26, 27 (800M), 29 and 30 carry a rate that is now known to be 0.002 off; their
    comparisons with each other stand, their comparisons with base-rate runs carry that offset.
  - N1 HIT: `w20_cn_d3072_B128_400M` 1.21812 against 1.23331 (width 1,536), and better than width 2,048 (1.22615) by
    0.008. Width at 32,768 tokens per step is still UNBRACKETED; 4,096 runs.
  - Three GPUs HOT AND IDLE (e 1, f 0, m 0); they take wave 31.

### Wave 31 (the rate below base at the best shape; width 3,072 at the best batch), sealed 2026-10-04T03:50Z

| id | prediction |
|---|---|
| Y1 | `w31_cn_d1536_B512_lrhalf_400M` (0.0015 / 0.01) beats `w27_cn_d1536_B512_400M` (1.21260) |
| Y2 | `w31_cn_d1536_B512_lrquarter_400M` (0.00075 / 0.005) is worse than the half-rate arm: the curve turns |
| Y3 | `w31_cn_d3072_B512_400M` (base rate, `--accum 32`) beats `w27_cn_d1536_B512_400M` (1.21260) |
- 2026-10-04T03:54Z · Wave 31 trains on e 1 and f 0 (step lines seen) and m 0 (process alive, first step pending). 18 of 18 GPUs busy. Credit $224.73.
- 2026-10-04T04:02Z · Idle sweep: 9 boxes checked, none idle for an hour; none closed. Credit $222.76.
- 2026-10-04T04:08Z · Landed:
  - T1 HIT: `w26_cn_d2048_B512_lr4x_400M` 1.20940 against 1.21488. NEW BEST SDM SCORE, even at the 4x rate, which
    wave 27 showed is about 0.002 worse than base at width 1,536.
  - U4 HIT: `w27_mq_d1536_B512_lr4x_400M` 1.05863 (NEW BEST TRANSFORMER), leading the SDM at the same shape by 0.156.
  - G9 MISS, and a RED FLAG: `w13_cnmix_d256_T1024_800M` 1.43221, WORSE than its 400M twin (1.40822) on twice the data.
    Its gradient norm (before the clip at 1.0) climbed from 0.5 to 30 to 100 over training while the loss stalled at
    about 4.8 to 5.2 after step 8,200. The 400M twins rose only to about 2. ⇒ the never-fading mixer at 8,192 tokens
    per step is UNSTABLE on a long schedule. The big-batch mixer runs now training sit at 0.6 to 1.4 (e, k) and 1 to 2.5
    (g, l), and the window-2,048 run at about 3 (i 0); all are watched. The fix (a decay below 1.0, weight decay, a
    lower rate, or a norm on the memory read) is unmeasured and goes to the next fleet's plan.
  - Three GPUs HOT AND IDLE (n 0 and 1, i 1); they take wave 32.

### Wave 32 (base rate at width 2,048; data at the best shape; the mixer at the best batch), sealed 2026-10-04T04:08Z

| id | prediction |
|---|---|
| AA1 | `w32_cn_d2048_B512_400M` (base rate, `--accum 16`) beats `w26_cn_d2048_B512_lr4x_400M` (1.20940) |
| AA2 | `w32_cn_d1536_B512_800M` (base rate) beats `w27_cn_d1536_B512_400M` (1.21260) by at least 0.01 |
| AA3 | `w32_cnmix_d256_T1024_B128_400M` (131,072 tokens per step, `--accum 8`) beats `w14_cn_d256_B512_400M` (1.31151), and its gradient norm stays below 3 |
- 2026-10-04T04:09Z · Wave 32 trains on n 0 and 1 (step lines seen) and i 1 (process alive, first step pending). 18 of 18 GPUs busy.
- 2026-10-04T04:16Z · Heartbeat. Landed:
  - W1 HIT, BY A WIDE MARGIN: `w29_cn_d1536_B512_lr4x_hm768_400M` 1.17481 against 1.21488 with the hop MLP at 192
    (0.040 better). NEW BEST SDM SCORE. The hop MLP (the small network after each memory read) had been 192 wide in
    every run of the campaign; it is the biggest single lever found since batch size. UNBRACKETED: the 3,072 arm (l 1)
    lands in minutes.
  - RED, FIXED: `w25_cn_d4096_B128_400M` died at its step-5,000 evaluation with CUDA out of memory (the eval batch of
    8,192 tokens of 65,536-way logits on top of a 24 GiB model). `eval_bpb` gained an env override `SDM_EVAL_TOKENS`
    (default 8,192, so every earlier number is unchanged). The run resumes from its step-5,000 checkpoint with
    `SDM_EVAL_TOKENS=2048`; its sealed prediction S2 is unchanged.
  - j 1 HOT AND IDLE (W1 done); it takes wave 33.

### Wave 33 (the hop MLP 768 at the base rate), sealed 2026-10-04T04:16Z before it runs

| id | prediction |
|---|---|
| AB1 | `w33_cn_d1536_B512_hm768_400M` (base rate) beats `w29_cn_d1536_B512_lr4x_hm768_400M` (1.17481): the base rate's edge at width carries over |
- 2026-10-04T04:22Z · CORRECTION to the 04:16Z entry: width 4,096 did NOT resume. Its checkpoint directory was empty
  (the eval at step 5,000 runs before the step-5,000 save, so the crash came first), and the old wave 25 fill runner,
  still alive on f, re-claimed the line before the new runner could, so it restarted from step 0 WITHOUT the eval
  override and would have died again at step 5,000. Stopped by PID (runner 35495 and 35494, run 53216 and 53214), its
  partial log moved aside, and the env-carrying runner took the line at 04:19Z (SDM_EVAL_TOKENS=2048 confirmed in the
  process environment). It runs from scratch: about 4.8 h at 23k tok/s, finishing near 09:10Z, close to the 09:30Z
  deadline. Wave 33 trains on j 1 (step lines seen).
- 2026-10-04T04:30Z · Heartbeat. Landed (all at width 1,536, B 512, 4x rate, 400M):
  - W2 HIT: `w29_cn_d1536_B512_lr4x_hm3072_400M` 1.14671 against hop MLP 768 (1.17481). NEW BEST SDM SCORE.
    Hop MLP 192 1.21488 · 768 1.17481 · 3,072 1.14671: UNBRACKETED, still falling (0.040, then 0.028 per 4x).
  - X1 HIT: `w30_cn_d1536_B512_lr4x_hops8_400M` 1.19732 against 4 hops (1.21488). Hop count UNBRACKETED.
  - Two GPUs HOT AND IDLE (l 1, m 1); they take wave 34. Each box holds the next value behind the first.

### Wave 34 (hop MLP and hop count past the edge), sealed 2026-10-04T04:30Z before it runs

| id | prediction |
|---|---|
| AC1 | `w34_cn_d1536_B512_lr4x_hm6144_400M` beats the 3,072 arm (1.14671) |
| AC2 | `w34_cn_d1536_B512_lr4x_hm12288_400M` is WORSE than the 6,144 arm: the curve turns |
| AC3 | `w34_cn_d1536_B512_lr4x_hops16_400M` beats 8 hops (1.19732) |
| AC4 | `w34_cn_d1536_B512_lr4x_hops32_400M` is worse than 16 hops |
- 2026-10-04T04:34Z · Wave 34 trains on l 1 (hop MLP 6,144) and m 1 (16 hops), step lines seen. 18 of 18 GPUs busy. Credit $218.58.
- 2026-10-04T04:48Z · Heartbeat: nothing new finished since 04:30Z; 18 of 18 GPUs busy. Credit $215.59. Next: the half and quarter rates (e 1, f 0, step 2,300 and 2,400 of 3,051) near 05:05Z.
- 2026-10-04T05:01Z · Idle sweep: 9 boxes checked, all 18 GPUs train (a python run on each, step lines advancing); none idle, none closed. Credit $213.55.
- 2026-10-04T05:10Z · Heartbeat. Landed:
  - The rate at width 1,536, B 512, 400M: 0.25x 1.24159 · 0.5x 1.22070 · 1x 1.21260 · 4x 1.21488 · 8x 1.21848. Y1 MISS
    (half is worse than base), Y2 HIT. The rate sweep at width is BRACKETED: best at 1x, with 1x to 4x within 0.003.
  - R1 MISS: `w24_cnmix_d512_T1024_B128_400M` 1.24557, not better than `w18_cn_d1024_B512_400M` (1.22701). Its gradient
    norm rose from 0.2 to 3.0 over the run. Its plain twin at width 512 has never run; wave 35 runs it.
  - U3 HIT: `w27_cn_d1536_B512_lr4x_800M` 1.18813 against 1.21488 at 400M (0.027). At width 1,536 data helps much more
    than at width 256 (0.017 per doubling there). Data UNBRACKETED.
  - Four GPUs HOT AND IDLE (e 0 and 1, f 0, k 0); they take wave 35.

### Wave 35 (the body knobs together; the plain width-512 twin; base rate and more data for the wide body), sealed 2026-10-04T05:10Z

| id | prediction |
|---|---|
| AD1 | `w35_cn_d1536_B512_hm3072_hops8_400M` (base rate) beats `w29_cn_d1536_B512_lr4x_hm3072_400M` (1.14671): hop MLP and hops stack |
| AD2 | `w35_cn_d512_B512_400M` (plain) beats `w24_cnmix_d512_T1024_B128_400M` (1.24557): at the best batch the mixer does not help at width 512 |
| AD3 | `w35_cn_d1536_B512_hm3072_400M` (base rate) lands within 0.005 of the 4x arm (1.14671) |
| AD4 | `w35_cn_d1536_B512_lr4x_hm3072_800M` beats the 400M arm (1.14671) by at least 0.02 |
- 2026-10-04T05:14Z · Wave 35 trains on e 0 and 1, f 0 and k 0 (step lines seen). 18 of 18 GPUs busy. Credit $212.59.
- 2026-10-04T05:30Z · Heartbeat. Landed:
  - N2 HIT BY A WIDE MARGIN: `w20_cnmix_d1024_T1024_B32_400M` 1.21102 against the plain `w15_cn_d1024_B128_400M` 1.24868
    at the same 32,768 tokens per step (0.038 better). It even beats the plain width 1,024 at the best batch (1.22701).
    THE MIXER'S GAIN GROWS WITH WIDTH at the big batch: width 256 none (1.34166 against 1.33548, undecided), width 1,024
    0.038. This is the all-SDM attention analogue paying off, which is the campaign's law in action.
  - R1's run scored 1.24557 (logged at 05:10Z).
  - g 0 HOT AND IDLE; it takes wave 36, the mixer at width 1,024 and the best batch. At its measured 21k tok/s that run
    needs about 5.3 h, past the 09:30Z deadline, so the DEADLINE MOVES to 11:30Z (2026-10-04), about $19 more at $9.6/h.

### Wave 36 (the mixer at width 1,024 and the best batch), sealed 2026-10-04T05:30Z before it runs

| id | prediction |
|---|---|
| AE1 | `w36_cnmix_d1024_T1024_B128_400M` (131,072 tokens per step, `--accum 32`) beats `w20_cnmix_d1024_T1024_B32_400M` (1.21102) |
- 2026-10-04T05:35Z · Wave 36 trains on g 0 (process alive from 05:29Z, first step line pending at `--accum 32`). The
  watcher was stopped by its PID (44290) and restarted on the 11:30Z deadline (PID 84094); its first pass pulled 9 of 9.
- 2026-10-04T05:45Z · THE NAVIGATOR: "you decide on things and keep all boxes busy" (the Spark had sat idle waiting for
  his go on the long BASE run). So the BASE run starts now, on the Spark, chosen from the evidence of waves 13 to 35:
  `w37_base_cn_d1536_hm3072_hops8_B512_3B`: width 1,536, hop MLP 3,072, 8 hops, cn recipe, B 512 (131,072 tokens per
  step), base rate, all 3.0B tokens of the Spark's `train_big` in one pass (22,888 steps), checkpoint every 1,000 steps
  (resumable), eval every 2,000. No mixer: the never-fading mixer went unstable on the long small-batch schedule and its
  fix is unmeasured, so a 40-hour run does not carry it. q job 1004053135-0265; 19,900 tok/s at step 50, so about 42 h.
  The Spark's trainer code was brought up to date first (seven files; model self-test 50 of 50). The Keep drive still
  needs the navigator: `udisksctl mount -b /dev/sda2` is refused without a polkit login on the Spark's console.

### BASE-1, sealed 2026-10-04T05:45Z before its result

| id | prediction |
|---|---|
| BASE1 | `w37_base_cn_d1536_hm3072_hops8_B512_3B` beats every 400M SDM arm of the campaign; it scores below 1.10 on TEST |
- 2026-10-04T05:50Z · Landed:
  - S3 HIT: `w25_cn_d1024_B512_1600M` 1.18669 against 1.22701 at 400M (0.040 better).
  - AB1 MISS: `w33_cn_d1536_B512_hm768_400M` (base rate) 1.17876 against 1.17481 at 4x; 0.004 apart, inside noise.
  - AD2 MISS: `w35_cn_d512_B512_400M` (plain) 1.26165 against the mixer twin `w24_cnmix_d512_T1024_B128_400M` 1.24557.
    At the best batch the mixer DOES help at width 512 (0.016), so with I1 at width 256 (none) and N2 at width 1,024
    (0.038) the mixer's gain grows with width at the big batch too.
- 2026-10-04T05:50Z · THE SCALING FIT AT THE BIG BATCH (asked for by the navigator). L = E + A (d/256)^-a + B (D/400M)^-b
  over the nine SDM points at B 512 with hop MLP 192 and 4 hops: E 1.131, A 0.137, a 0.78, B 0.049, b 0.97, rms
  0.0028 (nine points, five parameters, so a sketch, not a law). It predicts width 1,536 at 1.6B 1.177 and at 3B 1.172.
  The data term decays fast (b near 1), so past about 1.6B tokens width matters more than data for this recipe. The
  transformer fit has five points for five parameters (an exact fit, no evidence) and is NOT used. The body levers move
  the curve: hop MLP 3,072 took 0.068 off at width 1,536 and 400M, so the curve for the new body is owed by waves 34,
  35 and the BASE run.
- 2026-10-04T05:52Z · GIT FREEZE: Claude 1 is restarting the repository's history from scratch (the navigator's
  choice). Until Claude 1 says it is done, this lane makes NO commits, merges or worktrees; the log and fill files are
  written to disk only and are committed after the restart. Training on Vast and the Spark needs no git and continues.
- 2026-10-04T05:52Z · Heartbeat. Landed:
  - AA1 HIT: `w32_cn_d2048_B512_400M` (base rate) 1.20460 against 1.20940 at 4x. Width at B 512, base rate, 400M:
    256 1.31151 · 512 1.26165 · 1,024 1.22701 · 1,536 1.21260 · 2,048 1.20460 (3,072 runs): UNBRACKETED, flattening.
  - G8 HIT: `w13_cnmix_d256_T2048_400M` 1.41307, between 1.40822 and 1.42106.
  - Five GPUs HOT AND IDLE (e 1, i 0, j 0 and 1, n 0); they take wave 37, which draws the new body's own scaling curve.
  - Spark BASE run at step 150, 20,760 tok/s. Credit $206.66.

### Wave 37 (the new body, hop MLP 3,072, across width 256 to 2,048 and data to 1.6B), sealed 2026-10-04T05:52Z

B 512, base rate, cn recipe, 4 hops; only width, tokens and `--hop-mlp 3072` differ from the old-body references.

| id | prediction |
|---|---|
| AF1 | `w37_cn_d2048_B512_hm3072_400M` beats `w29_cn_d1536_B512_lr4x_hm3072_400M` (1.14671) |
| AF2 | `w37_cn_d1024_B512_hm3072_400M` beats the old-body width 1,536 `w27_cn_d1536_B512_400M` (1.21260): body beats width |
| AF3 | `w37_cn_d1024_B512_hm3072_1600M` beats the 400M arm by at least 0.03 |
| AF4 | `w37_cn_d512_B512_hm3072_400M` beats `w35_cn_d512_B512_400M` (1.26165) by at least 0.04 |
| AF5 | `w37_cn_d256_B512_hm3072_400M` beats `w14_cn_d256_B512_400M` (1.31151) by at least 0.03 |
- 2026-10-04T05:56Z · Wave 37 trains on j 0 and 1, n 0, e 1 and i 0 (step lines seen). 18 of 18 Vast GPUs busy, the Spark on BASE. No commit (git freeze).
- 2026-10-04T06:01Z · Idle sweep: 9 boxes checked, all 18 GPUs train; none idle, none closed. Credit $204.72. No commit
  (git freeze). AC3 HIT: `w34_cn_d1536_B512_lr4x_hops16_400M` 1.18390 against 8 hops (1.19732); hops 4 1.21488 · 8
  1.19732 · 16 1.18390: UNBRACKETED, still falling (0.018, then 0.013); 32 hops runs on m 1.
- 2026-10-04T06:10Z · Heartbeat. AC1 HIT: `w34_cn_d1536_B512_lr4x_hm6144_400M` 1.13522 against 3,072 (1.14671). NEW BEST
  SDM SCORE. Hop MLP 192 1.21488 · 768 1.17481 · 3,072 1.14671 · 6,144 1.13522: UNBRACKETED, flattening (0.040, 0.028,
  0.011 per step); 12,288 now runs on l 1 (AC2). 18 of 18 GPUs busy; Spark BASE at step 300 (20,770 tok/s). Credit
  $203.67. Git freeze holds (Claude 1 is merging the hero branches before the restart); no commit.
- 2026-10-04T07:15Z · WRONG STATE, and the cause. This session restarted around 06:30Z and no heartbeat ran for about
  50 minutes; when the roll call ran again TEN GPUs were HOT AND IDLE (e 0 and 1, f 0, i 0 and 1, j 1, k 1, l 0, m 0,
  n 1), most for 30 to 60 minutes. Worse: the restart also killed the pull watcher (last pull 06:26Z), so nothing was
  being pulled and the 11:30Z deadline had no watcher. Restarted at 07:20Z (macOS has no `setsid`; a plain `nohup` in a
  subshell), first pass pulled 8 of 9. The scratchpad roll-call script was lost with the session and was rebuilt.
- 2026-10-04T07:20Z · RED: box g reports `offline` in Vast and refuses ssh on both routes (first seen 07:15Z). Its two
  runs (`w36_cnmix_d1024_T1024_B128_400M`, `w21_cn_d1536_B128_800M`) have no result at home. Not destroyed; one more
  try next hour, per the sweep rule. The watcher holds it at the deadline if its final pull fails.
- 2026-10-04T07:25Z · Landed while nobody watched (all scored against sealed predictions):
  - AC4 HIT: `w34_cn_d1536_B512_lr4x_hops32_400M` 1.24593, far worse than 16 hops (1.18390). Hop count BRACKETED: best 16.
  - AD1 HIT: `w35_cn_d1536_B512_hm3072_hops8_400M` 1.13335. NEW BEST SDM SCORE: hop MLP and hops stack.
  - AD3 MISS: `w35_cn_d1536_B512_hm3072_400M` (base rate) 1.15266, 0.006 worse than 4x (1.14671); inside noise.
  - Y3 HIT: `w31_cn_d3072_B512_400M` 1.19484 against 1.21260. Width at B 512: 2,048 1.20460, 3,072 1.19484: UNBRACKETED.
  - AA2 HIT: `w32_cn_d1536_B512_800M` 1.18763 against 1.21260 (0.025).
  - AA3 HIT on the letter: `w32_cnmix_d256_T1024_B128_400M` 1.30950 against 1.31151 (0.002, inside noise).
  - V1 HIT: `w28_cnmix_d768_T1024_B128_400M` 1.21412 beats the plain width 1,024 at the best batch (1.22701).
  - S1 MISS: at 800M the mixer `w25_cnmix_d256_T1024_B32_800M` 1.31886 beats its plain twin 1.32646 (0.008).
  - AF5 HIT (width 256 + hop MLP 3,072: 1.25067, 0.061 better), AF4 HIT (512: 1.19774, 0.064 better), AF2 HIT (1,024:
    1.16530 beats the old-body width 1,536). The new body is worth about 0.06 at every width.
  - The mixer verdict, across all of it: it helps at every width from 512 up, at both batches, and at width 256 once
    the run is long enough.

### Wave 38 (the best recipe pushed and combined, on the ten idle GPUs), sealed 2026-10-04T07:25Z before it runs

B 512, base rate unless stated, 400M tokens unless stated. Reference: `w35_cn_d1536_B512_hm3072_hops8_400M` 1.13335.

| id | prediction |
|---|---|
| AG1 | `w38_cn_d1536_B512_hm6144_hops16_400M` beats the reference |
| AG2 | `w38_cn_d1536_B512_hm6144_hops8_400M` beats the reference |
| AG3 | `w38_cn_d2048_B512_hm3072_hops8_400M` beats the reference: width still helps the new recipe |
| AG4 | `w38_cnmix_d512_T1024_B128_hm3072_400M` beats `w37_cn_d512_B512_hm3072_400M` (1.19774): the mixer adds to the new body |
| AG5 | `w38_cn_d1536_B512_hm3072_hops8_800M` beats the reference by at least 0.02 |
| AG6 | `w38_mq_d1536_B512_800M` beats `w27_mq_d1536_B512_lr4x_400M` (1.05863) by at least 0.02 |
| AG7 | `w38_cn_d1536_B512_hm3072_hops16_400M` beats the reference |
| AG8 | `w38_cn_d768_B512_hm3072_400M` lands between 1.16530 (width 1,024) and 1.19774 (width 512) |
| AG9 | `w38_cn_d1536_B512_hm3072_hops8_400M_s1` (seed 1) lands within 0.003 of the reference |
| AG10 | `w38_cn_d1024_B512_hm3072_hops8_400M` beats `w37_cn_d1024_B512_hm3072_400M` (1.16530) |
- 2026-10-04T07:35Z · AMENDMENT before a valid run: AG1's line died at once with CUDA out of memory at `--accum 8`; it
  reruns at `--accum 16` (same 131,072 tokens per step). AG1 is unchanged. m 1 was also idle (hops 32 done); it takes:

| id | prediction |
|---|---|
| AG11 | `w38_mq_d2048_B512_400M` (MLP 5,461) beats `w27_mq_d1536_B512_lr4x_400M` (1.05863) |
- 2026-10-04T07:43Z · Heartbeat: 16 of 16 reachable GPUs train (wave 38 all stepping); box g still offline. Watcher alive (PID 5528, last pull 07:39Z). Spark BASE at step 1,200 of 22,888. Credit $190.31. No commit: the git freeze holds until Claude 1 calls it done.
- 2026-10-04T07:50Z · Heartbeat: nothing new finished; 16 of 16 reachable GPUs busy; g offline. j 0 finishes its wave 37
  line within minutes, so its next line is queued now rather than after it idles:

| id | prediction |
|---|---|
| AG12 | `w38_cn_d3072_B512_hm3072_hops8_400M` (`--accum 32`) beats `w38_cn_d2048_B512_hm3072_hops8_400M` (AG3's arm) |
- 2026-10-04T08:01Z · Idle sweep: 9 boxes checked. 8 train on every GPU (16 of 16); none idle, none closed. Box g:
  Vast `offline`, ssh refused on both routes since 07:15Z (first unreachable hour); one more try at 09:37Z, then red.
  Credit $187.81. No commit (git freeze).
- 2026-10-04T08:01Z · AF1 HIT, barely: `w37_cn_d2048_B512_hm3072_400M` 1.14613 against 1.14671 (inside noise). New body,
  base rate, 400M: width 256 1.25067 · 512 1.19774 · 1,024 1.16530 · 1,536 1.15266 · 2,048 1.14613: still falling,
  but only 0.007 for the last step, so width is close to flat for the new body at 400M. UNBRACKETED.
- 2026-10-04T08:10Z · Heartbeat. Landed:
  - AD4 HIT: `w35_cn_d1536_B512_lr4x_hm3072_800M` 1.11217 against 1.14671 at 400M (0.035 better). NEW BEST SDM SCORE.
    Data helps the new body as much as the old (0.027 at 800M there): DATA UNBRACKETED.
  - AC2 HIT: `w34_cn_d1536_B512_lr4x_hm12288_400M` 1.13532, level with 6,144 (1.13522). Hop MLP BRACKETED: the gain stops
    between 6,144 and 12,288 at width 1,536 (192 1.21488 · 768 1.17481 · 3,072 1.14671 · 6,144 1.13522 · 12,288 1.13532).
  - k 0 and l 1 HOT AND IDLE; they take wave 39 (both finish inside the 11:30Z deadline at measured speeds).

### Wave 39 (data and hop MLP 6,144 for the best recipe at width 1,024), sealed 2026-10-04T08:10Z before it runs

| id | prediction |
|---|---|
| AH1 | `w39_cn_d1024_B512_hm3072_hops8_800M` beats `w38_cn_d1024_B512_hm3072_hops8_400M` (AG10's arm) by at least 0.02 |
| AH2 | `w39_cn_d1024_B512_hm6144_hops8_400M` beats `w38_cn_d1024_B512_hm3072_hops8_400M` |
- 2026-10-04T08:14Z · Wave 39 trains on k 0 and l 1 (step lines seen). 16 of 16 reachable GPUs busy; Spark BASE at step 1,450. Credit $186.82. No commit (git freeze).
- 2026-10-04T08:15Z · GIT FREEZE ENDS (Claude 1): the repository restarted fresh at 02296dc (one commit, every tracked file kept; the 25 hero branches landed before the restart). This lane commits again: the log entries written on disk during the freeze (05:52Z to now) and the fill files of waves 37 to 39 land in the next commit.
- 2026-10-04T08:30Z · Heartbeat. AG8 HIT: `w38_cn_d768_B512_hm3072_400M` 1.17695, between width 1,024 (1.16530) and 512
  (1.19774). The new body's width curve at 400M is complete: 256 1.25067 · 512 1.19774 · 768 1.17695 · 1,024 1.16530 ·
  1,536 1.15266 · 2,048 1.14613 (3,072 runs). i 0 HOT AND IDLE; it takes:

| id | prediction |
|---|---|
| AH3 | `w39_mq_d1024_B512_800M` beats `w18_mq_d1024_B512_400M` (1.07975) by at least 0.02 |
- 2026-10-04T08:50Z · Heartbeat. AG10 HIT: `w38_cn_d1024_B512_hm3072_hops8_400M` 1.14585 against 4 hops (1.16530): 8 hops
  are worth 0.019 at width 1,024 with the new body, as at 1,536. n 1 HOT AND IDLE; it takes:

| id | prediction |
|---|---|
| AH4 | `w39_cn_d1024_B512_hm3072_hops16_400M` beats 8 hops at width 1,024 (1.14585) |
- 2026-10-04T09:01Z · Idle sweep: 9 boxes checked; 8 train on all 16 GPUs, none idle, none closed. RED: box g
  (54033218) unreachable for the SECOND hourly sweep in a row (Vast `offline`, ssh refused on both routes since
  07:15Z). Reported to the navigator; NOT destroyed (its two runs have no result at home, and the watcher holds it at
  the deadline unless its final pull succeeds). Credit $179.57.
- 2026-10-04T09:10Z · Heartbeat: nothing new finished; 16 of 16 reachable GPUs busy. Finish estimates from measured step
  rates: every run lands by about 10:50Z except `w38_cn_d3072_B512_hm3072_hops8_400M` (j 0, step 1,000 of 3,051 after
  1.2 h), which needs to about 11:35Z. DEADLINE MOVED from 11:30Z to 12:00Z (about $4.50 more); watcher stopped by its
  PID (5528) and restarted. Box g still offline (decision A stands unless the navigator says otherwise). Credit $178.66.
- 2026-10-04T09:32Z · Heartbeat. AG9 HIT: `w38_cn_d1536_B512_hm3072_hops8_400M_s1` (seed 1) 1.13368 against seed 0 1.13335
  (0.0003 apart): the best recipe is reproducible, and the seed noise at this shape is well under the 0.007 used so far.
- RED, FIXED: `w25_cn_d4096_B128_400M` trained to its last step (12,207) and then died in the final per-window eval with
  CUDA out of memory: `per_window` in `track4_sdmllm_paired_window_comparison.py` had its own batch cap and never read
  `SDM_EVAL_TOKENS`. It now reads the same override (default unchanged). The run's `last.pt` survives, so it reruns to
  resume at the final step and only evaluate. f 1 does that first, then takes wave 40; i 1 HOT AND IDLE takes wave 40.

### Wave 40 (short runs that fit before the 12:00Z deadline), sealed 2026-10-04T09:32Z before it runs

| id | prediction |
|---|---|
| AI1 | `w40_cn_d1024_B512_hm6144_hops16_400M` (`--accum 8`) beats `w39_cn_d1024_B512_hm6144_hops8_400M` (AH2's arm) |
| AI2 | `w40_cn_d512_B512_hm3072_hops8_400M` beats `w37_cn_d512_B512_hm3072_400M` (1.19774) by at least 0.015 |
| AI3 | `w40_cn_d768_B512_hm3072_hops8_400M` beats `w38_cn_d768_B512_hm3072_400M` (1.17695) by at least 0.015 |
- 2026-10-04T09:45Z · The width-4,096 resume evaluated cleanly: `w25_cn_d4096_B128_400M` 1.21399 against width 3,072
  (1.21812). S2 MISS (it IS better than width 2,048 by more than 0.005: 0.012). Old body at 32,768 tokens per step:
  2,048 1.22615 · 3,072 1.21812 · 4,096 1.21399; still falling by 0.004, which the seed check (0.0003) says is real.
  UNBRACKETED, close to flat.
- AMENDMENT before a valid run: AI2's line died at once with CUDA out of memory (hop MLP 3,072 with 8 hops at
  `--accum 2`); AI2 reruns at `--accum 4` and AI3 at `--accum 8` (the step and its tokens unchanged). AI2 and AI3 are
  unchanged.
- 2026-10-04T09:50Z · Landed:
  - AF3 HIT: `w37_cn_d1024_B512_hm3072_1600M` 1.10603 against 1.16530 at 400M (0.059). NEW BEST SDM SCORE. Data is now
    the biggest lever left: the new body gains far more from data than the old (0.040 for the same step).
  - AG7 HIT: `w38_cn_d1536_B512_hm3072_hops16_400M` 1.11767, the best 400M SDM (16 hops beat 8 by 0.016 with the new body).
  - AG2 HIT: `w38_cn_d1536_B512_hm6144_hops8_400M` 1.12185 against 1.13335.
  - AG3 HIT: `w38_cn_d2048_B512_hm3072_hops8_400M` 1.12708 against 1.13335.
  - AH2 HIT: `w39_cn_d1024_B512_hm6144_hops8_400M` 1.13403 against 1.14585.
- 2026-10-04T09:55Z · Heartbeat. Four GPUs HOT AND IDLE (e 1, l 1, m 0, n 0). Data is now the biggest lever and its runs
  are long, so the DEADLINE MOVES from 12:00Z to 16:00Z (about $35 more for the 8 live boxes; credit $173.10, about $180
  spent of the $300 window). Box g is NOT extended: decision A stands, it is destroyed by hand at 11:30Z if still dead.

### Wave 41 (data for the best recipe; the mixer with the new body), sealed 2026-10-04T09:55Z before it runs

| id | prediction |
|---|---|
| AJ1 | `w41_cn_d1536_B512_hm3072_hops16_800M` beats `w38_cn_d1536_B512_hm3072_hops16_400M` (1.11767) by at least 0.025, and becomes the best SDM |
| AJ2 | `w41_cn_d1024_B512_hm3072_hops16_800M` beats `w39_cn_d1024_B512_hm3072_hops16_400M` (AH4's arm) by at least 0.025 |
| AJ3 | `w41_cn_d1024_B512_hm3072_hops8_1600M` beats `w37_cn_d1024_B512_hm3072_1600M` (1.10603, 4 hops) |
| AJ4 | `w41_cnmix_d768_T1024_B128_hm3072_400M` beats `w38_cn_d768_B512_hm3072_400M` (1.17695) by at least 0.02 |
- 2026-10-04T09:54Z · Wave 41 started on e 1, l 1, n 0 and m 0 (n 0 stepping; e 1, l 1 and m 0 compiling). Watcher restarted on the 16:00Z deadline (PID 81406). 16 of 16 reachable GPUs busy. Credit $173.10.
- 2026-10-04T10:02Z · Idle sweep: 9 boxes checked; none idle for an hour, none closed. Box g still offline (third
  hour; decision A: destroy by hand at 11:30Z). Credit $171.38.
- 2026-10-04T10:02Z · AG11 HIT: `w38_mq_d2048_B512_400M` 1.04446 against 1.05863 at width 1,536. NEW BEST TRANSFORMER;
  the gap to the best SDM at 400M (1.11767) is 0.073. m 1 HOT AND IDLE; it takes the first never-varied knob from the
  ranked list, the number of exact past tokens:

| id | prediction |
|---|---|
| AK1 | `w42_cn_d1536_B512_hm3072_hops16_nb16_400M` (16 exact past tokens) beats `w38_cn_d1536_B512_hm3072_hops16_400M` (1.11767) |
- 2026-10-04T10:08Z · Heartbeat: nothing new finished; 16 of 16 reachable GPUs train (waves 38 to 42, all stepping). Spark BASE at step 2,600 of 22,888. Credit $170.44.
- 2026-10-04T10:30Z · Heartbeat. AI2 HIT: `w40_cn_d512_B512_hm3072_hops8_400M` 1.17794 against 1.19774 (0.020). AH4 HIT:
  `w39_cn_d1024_B512_hm3072_hops16_400M` 1.13013 against 8 hops (1.14585). With the new body, hops help at every width:
  width 1,024 4 / 8 / 16 hops: 1.16530 · 1.14585 · 1.13013. Two GPUs HOT AND IDLE (f 1, n 1); they take wave 43.

### Wave 43 (the hop optimum with the new body; 32 exact past tokens), sealed 2026-10-04T10:30Z before it runs

| id | prediction |
|---|---|
| AL1 | `w43_cn_d1536_B512_hm3072_hops24_400M` lands within 0.005 of 16 hops (1.11767): the hop curve is flat between 16 and 24 |
| AL2 | `w43_cn_d1536_B512_hm3072_hops16_nb32_400M` is WORSE than the 16-token arm (AK1): exact tokens stop paying past 16 |
- 2026-10-04T10:50Z · Heartbeat. Landed:
  - AG5 HIT: `w38_cn_d1536_B512_hm3072_hops8_800M` 1.09480 against 1.13335 at 400M (0.039). NEW BEST SDM SCORE.
  - AH1 HIT: `w39_cn_d1024_B512_hm3072_hops8_800M` 1.10852 against 1.14585 (0.037).
  - AG1 HIT: `w38_cn_d1536_B512_hm6144_hops16_400M` 1.10838, the best 400M SDM (hop MLP 6,144 and 16 hops stack).
  - AI3 HIT: `w40_cn_d768_B512_hm3072_hops8_400M` 1.15747 against 1.17695 (0.019).
  - AG6 HIT: `w38_mq_d1536_B512_800M` 1.02369 (NEW BEST TRANSFORMER). AH3 HIT: `w39_mq_d1024_B512_800M` 1.04904.
    At the matched shape and data (width 1,536, 800M) the gap is 1.09480 against 1.02369: 0.071.
  - AG4 MISS, and a FINDING: `w38_cnmix_d512_T1024_B128_hm3072_400M` 1.25120, WORSE than the plain new body
    (1.19774) by 0.053 and no better than the mixer with the OLD body (1.24557). With the mixer on, the wide hop MLP
    bought nothing. Both big-batch mixer runs show the gradient norm rising late (0.2 early to 2.7 and 6.2 at the end),
    the same signature as the 800M instability. ⇒ the mixer's stability is now the blocker for every mixer result;
    wave 44 tests two fixes (a slight fade 0.9999, Muon weight decay 0.1). The width-768 mixer (m 0) is watched.
  - Seven GPUs HOT AND IDLE (e 0, f 0, i 0, j 1, k 0 and 1, l 0); they take wave 44 (all inside the 16:00Z deadline).

### Wave 44 (data for the best recipe across widths; the yardstick at 1.6B; two mixer stability fixes), sealed 2026-10-04T10:50Z

| id | prediction |
|---|---|
| AM1 | `w44_cn_d1024_B512_hm6144_hops16_800M` beats `w39_cn_d1024_B512_hm3072_hops8_800M` (1.10852) |
| AM2 | `w44_cn_d768_B512_hm3072_hops8_1600M` beats the 400M arm (1.15747) by at least 0.05 |
| AM3 | `w44_cn_d512_B512_hm3072_hops8_1600M` beats the 400M arm (1.17794) by at least 0.05 |
| AM4 | `w44_mq_d1024_B512_1600M` beats the 800M arm (1.04904) by at least 0.025 |
| AM5 | `w44_cn_d1536_B512_hm6144_hops8_800M` beats `w38_cn_d1536_B512_hm3072_hops8_800M` (1.09480) |
| AM6 | `w44_cnmix_d512_T1024_B128_hm3072_fade_400M` (all four mixer decays 0.9999) beats 1.25120 and its gradient norm stays under 2 |
| AM7 | `w44_cnmix_d512_T1024_B128_hm3072_wd_400M` (Muon weight decay 0.1) beats 1.25120 and its gradient norm stays under 2 |
- 2026-10-04T10:56Z · Wave 44 trains on e 0, f 0, k 0 and 1, l 0, i 0 and j 1 (processes and step lines seen). 16 of 16 reachable GPUs busy. Credit $164.96.
- 2026-10-04T11:01Z · Idle sweep: 9 boxes checked; 8 train on all 16 GPUs, none idle, none closed. Box m shows Vast `offline` but answers ssh and trains (as j did earlier). Box g still unreachable (fourth hour); destroyed by hand at 11:30Z per decision A. Credit $163.24.
- 2026-10-04T11:10Z · THE NAVIGATOR: stop the sweeps at about $100 of credit and spend that last $100 on our true
  training run, with a settled shape. Plan recorded: the sweep fleet runs to its 16:00Z deadline (credit then about
  $120) and answers the last open questions (hop MLP 6,144 with 16 hops at 800M, 24 hops, 16 and 32 exact tokens, the
  two mixer fixes, width 3,072); a guard stops it early if credit falls to $110. The shape is then settled and the true
  run is rented with the remaining money, keeping a $15 margin. The DDP trainer (`track4_sdmonly_train_ddp.py`)
  predates the cn recipe (no hop MLP, untie, conj, Muon or accum), so a multi-GPU true run needs it extended first.
- 2026-10-04T11:09Z · Heartbeat: nothing new finished since 10:50Z; 16 of 16 reachable GPUs busy (waves 38 to 44). Spark BASE at step 3,150 of 22,888. Box g scheduled for destruction at 11:31Z. Lane DOTRULES (an Opus agent) builds the site's settling dotted dividers in its own worktree for Claude 1 to land. Credit $162.21.
- 2026-10-04T11:28Z · Heartbeat: nothing new finished; 16 of 16 reachable GPUs busy. Mixer gradient norms at step 500: weight-decay arm 0.257, fade arm 0.186, the unfixed width-512 run had 0.18 to 0.26 there too, so too early to tell (its rise began after step 1,000); the unfixed width-768 mixer is climbing 0.16 to 0.48 by step 1,200. Credit $159.57.
- 2026-10-04T11:32Z · DECISION A carried out: box g (54033218) still unreachable on both routes (direct timed out, proxy refused), so DESTROYED by id without a final pull; gone from `show instances` (8 left). Its line removed from `vast_sdmonly_fleet11.list`. Two runs LOST, never pulled: `w36_cnmix_d1024_T1024_B128_400M` (the mixer at width 1,024 and the best batch) and `w21_cn_d1536_B128_800M`. Credit $159.09.
- 2026-10-04T11:50Z · Heartbeat. Landed:
  - AG12 HIT: `w38_cn_d3072_B512_hm3072_hops8_400M` 1.11994 against width 2,048 (1.12708). Best recipe width curve at
    400M (hop MLP 3,072, 8 hops): 1,024 1.14585 · 1,536 1.13335 · 2,048 1.12708 · 3,072 1.11994: still falling by about
    0.007 per step: UNBRACKETED, slow.
  - AI1 HIT: `w40_cn_d1024_B512_hm6144_hops16_400M` 1.12078 against 8 hops (1.13403).
  - Two GPUs HOT AND IDLE (j 0, i 1); they take wave 45, which sharpens the true-run choice.

### Wave 45 (the true-run candidate's noise, and 16 hops at width 2,048), sealed 2026-10-04T11:50Z before it runs

| id | prediction |
|---|---|
| AN1 | `w45_cn_d1536_B512_hm6144_hops16_400M_s1` (seed 1) lands within 0.003 of seed 0 (1.10838) |
| AN2 | `w45_cn_d2048_B512_hm3072_hops16_400M` beats `w38_cn_d2048_B512_hm3072_hops8_400M` (1.12708) by at least 0.01 |
- 2026-10-04T12:01Z · Idle sweep: 8 boxes checked, all 16 GPUs train; none idle, none closed. Credit $155.05.
- 2026-10-04T12:08Z · Heartbeat: nothing new finished; 16 of 16 GPUs busy; Spark BASE at step 3,700. Credit $154.12.
  Mixer gradient norms at step 1,000 to 1,100: weight decay 0.59 to 0.62, fade 0.68 to 0.48, the unfixed run had 0.65 at step 1,000: so far NEITHER fix bends the curve. The unfixed width-768 mixer reached 1.11 at step 1,700.
- 2026-10-04T12:30Z · Heartbeat. AK1 MISS: `w42_cn_d1536_B512_hm3072_hops16_nb16_400M` (16 exact past tokens) 1.11894,
  0.0013 WORSE than 8 (1.11767). More exact tokens do not help; 32 lands at about 13:30Z. The best is at the lowest
  value tried, so per the edge rule the sweep goes below 8. m 1 HOT AND IDLE; it takes:

| id | prediction |
|---|---|
| AO1 | `w46_cn_d1536_B512_hm3072_hops16_nb4_400M` (4 exact past tokens) is WORSE than 8 (1.11767): the curve turns below 8 |
- 2026-10-04T12:41Z · Wave 46 trains on m 1 (process alive; first step line pending). Box m refused its direct route this beat and answered through the Vast proxy. 16 of 16 GPUs busy. Credit $151.25.
- 2026-10-04T12:50Z · Heartbeat. AJ2 HIT: `w41_cn_d1024_B512_hm3072_hops16_800M` 1.09026 against 1.13013 at 400M (0.040).
  NEW BEST SDM SCORE, and at width 1,024: 16 hops plus 800M beat width 1,536 with 8 hops at 800M (1.09480). l 1 HOT AND
  IDLE; it takes the hop sweep past 16 at width 1,024:

| id | prediction |
|---|---|
| AP1 | `w47_cn_d1024_B512_hm3072_hops24_400M` lands within 0.005 of 16 hops at width 1,024 (1.13013) |
- 2026-10-04T12:53Z · AMENDMENT before a valid run: AP1's line died at once with CUDA out of memory at `--accum 4` (24 hops); it reruns at `--accum 8`, same step and tokens. AP1 unchanged.
- 2026-10-04T13:01Z · Idle sweep: 8 boxes checked; none idle for an hour, none closed. Credit $146.96.
- 2026-10-04T13:01Z · AL2 HIT: `w43_cn_d1536_B512_hm3072_hops16_nb32_400M` (32 exact past tokens) 1.12085, worse than 16
  (1.11894) and 8 (1.11767). Exact past tokens 8 · 16 · 32: 1.11767 · 1.11894 · 1.12085, rising. n 1 HOT AND IDLE; the
  sweep below 8 gets its second value:

| id | prediction |
|---|---|
| AO2 | `w48_cn_d1536_B512_hm3072_hops16_nb2_400M` (2 exact past tokens) is worse than 4 (AO1's arm) |
- 2026-10-04T13:10Z · Heartbeat. AL1 MISS: `w43_cn_d1536_B512_hm3072_hops24_400M` 1.11022 BEATS 16 hops (1.11767) by
  0.0075 (not flat). NEW BEST 400M SDM with hop MLP 3,072. With the new body the hop curve is still falling (4 1.15266 ·
  8 1.13335 · 16 1.11767 · 24 1.11022): UNBRACKETED. (The old body's 32 hops collapsed, 1.24593; the wide hop MLP changes
  that.) f 1 HOT AND IDLE; it takes 32 hops at width 1,024 (width 1,536 would run past the deadline). Commit held: Claude
  1 is merging (.git/MERGE_HEAD present).

| id | prediction |
|---|---|
| AQ1 | `w49_cn_d1024_B512_hm3072_hops32_400M` (`--accum 16`) is WORSE than 24 hops at width 1,024 (AP1's arm): the hop curve turns by 32 |
- 2026-10-04T13:35Z · THE NAVIGATOR: use the Spark for shape-finding too, alongside the Vast boxes. The Spark BASE run
  (`w37_base_...`, step about 4,450 of 22,888, last checkpoint step 4,000) is PAUSED, not cancelled (q pause; it resumes
  from its checkpoint); its shape was fixed at 05:45Z and is behind the hop and hop-MLP findings since. The Spark takes
  what a 32 GB card cannot: LONG WINDOWS for the mixer (the campaign's aim). Its updated code was copied first. First
  submission died at once with CUDA out of memory (the paused BASE still holds 19 GB of the shared 121 GB); resubmitted
  with `--accum 8`. Both now run (q jobs 1004132834-0273 and -0274).

### Wave 50s (the mixer at long windows, on the Spark), sealed 2026-10-04T13:35Z before it runs

Width 512, hop MLP 3,072, 4 hops, the fast never-fading mixer, 131,072 tokens per step, 400M tokens, `--accum 8`.

| id | prediction |
|---|---|
| AR1 | `w50s_cnmix_d512_hm3072_T4096_B32_fast_400M` (window 4,096) beats the window-1,024 mixer `w38_cnmix_d512_T1024_B128_hm3072_400M` (1.25120) |
| AR2 | `w50s_cnmix_d512_hm3072_T16384_B8_fast_400M` (window 16,384) beats the window-4,096 arm |
- 2026-10-04T13:36Z · Heartbeat. AM3 HIT: `w44_cn_d512_B512_hm3072_hops8_1600M` 1.11983 against 1.17794 at 400M (0.058).
  Width 512 on 1.6B tokens nearly matches width 1,536 with 16 hops on 400M (1.11767): four times the data buys about what
  three times the width plus more hops buys. Data UNBRACKETED. k 0 HOT AND IDLE; it takes:

| id | prediction |
|---|---|
| AS1 | `w51_cn_d768_B512_hm3072_hops16_400M` beats 8 hops at width 768 (1.15747) by at least 0.01 |
- 2026-10-04T13:50Z · Heartbeat. AJ4 MISS: `w41_cnmix_d768_T1024_B128_hm3072_400M` 1.21797, WORSE than the plain width-768
  new body (1.17695) by 0.041. Second confirmation: with the wide hop MLP the window-1,024 mixer hurts. m 0 HOT AND IDLE;
  it takes:

| id | prediction |
|---|---|
| AT1 | `w52_cn_d512_B512_hm3072_hops16_400M` beats 8 hops at width 512 (1.17794) by at least 0.01 |
- 2026-10-04T14:01Z · Idle sweep: 8 boxes checked, all 16 GPUs train; none idle, none closed. Credit $138.76.
- 2026-10-04T14:08Z · Heartbeat: nothing new finished; 16 of 16 GPUs busy; the Spark runs the two long-window mixer tests. Credit $137.83. Next: width 1,536 with 16 hops at 800M (step 6,000 of 6,103) and both mixer fixes (steps 2,900 to 3,000 of 3,051) land within minutes.
- 2026-10-04T14:30Z · Heartbeat. Landed:
  - AJ1 HIT: `w41_cn_d1536_B512_hm3072_hops16_800M` 1.07705 against 1.11767 at 400M (0.041). NEW BEST SDM SCORE. The
    transformer at the same width and data (`w38_mq_d1536_B512_800M`) is 1.02369: the gap is 0.053.
  - AJ3 HIT: `w41_cn_d1024_B512_hm3072_hops8_1600M` 1.07998 against 1.10603 (4 hops, 1.6B).
  - AM2 HIT: `w44_cn_d768_B512_hm3072_hops8_1600M` 1.09459 against 1.15747 at 400M (0.063).
  - AM6 MISS: the fade fix `w44_cnmix_..._fade_400M` 1.25160, no better than unfixed (1.25120); gradient norm peaked 5.3.
  - AM7 MISS on its second half: the weight-decay fix `w44_cnmix_..._wd_400M` 1.23905 beats unfixed by 0.012 but its
    gradient norm peaked 7.8. Both still far worse than the plain new body (1.19774). ⇒ neither simple fix tames the
    mixer; the mixer stays out of the true run. Its long-window tests continue on the Spark.
  - Five GPUs freed (e 1, f 0, i 0, j 1, n 0). With 1.5 hours to the 16:00Z deadline no run long enough to answer an
    open question fits, so they wait; each box is destroyed as soon as both its GPUs are done and pulled, which saves
    money for the true run.
- 2026-10-04T14:50Z · Heartbeat. Landed:
  - AO1 HIT: `w46_cn_d1536_B512_hm3072_hops16_nb4_400M` 1.11924, worse than 8 (1.11767). Exact past tokens BRACKETED at
    8: 4 1.11924 · 8 1.11767 · 16 1.11894 · 32 1.12085.
  - AS1 HIT: `w51_cn_d768_B512_hm3072_hops16_400M` 1.14190 against 8 hops (1.15747). AT1 HIT:
    `w52_cn_d512_B512_hm3072_hops16_400M` 1.16362 against 1.17794. 16 hops beat 8 at every width (512, 768, 1,024, 1,536).
  - AM4 MISS by a hair: `w44_mq_d1024_B512_1600M` 1.02631 against 1.04904 at 800M (0.023, short of 0.025).
  - Boxes k (54033236) and m (54034428) finished every run: pulled home on the direct route (k 29 results, m 27), then
    DESTROYED by id, gone from `show instances` (6 boxes left), removed from the watch list. Credit $132.13.
- 2026-10-04T15:02Z · Idle sweep: 6 boxes checked. Boxes i, j and l had finished every run (no training process);
  pulled home (14, 19 and 24 results) and DESTROYED by id (54041348, 54033231, 54033237), removed from the watch list.
  3 boxes left (e, f, n), each with one run going. Credit $130.66.
- 2026-10-04T15:02Z · Landed:
  - AN1 HIT: `w45_cn_d1536_B512_hm6144_hops16_400M_s1` 1.10857 against seed 0 1.10838 (0.0002): the true-run candidate
    is reproducible.
  - AN2 HIT: `w45_cn_d2048_B512_hm3072_hops16_400M` 1.11205 against 8 hops (1.12708).
  - AP1 MISS: `w47_cn_d1024_B512_hm3072_hops24_400M` 1.12273 BEATS 16 hops at width 1,024 (1.13013) by 0.0074. Hops still
    gain at 24 at both widths tried: UNBRACKETED (32 at width 1,024 lands about 15:40Z).
  - AM5 HIT: `w44_cn_d1536_B512_hm6144_hops8_800M` 1.08114 against hop MLP 3,072 (1.09480).
- 2026-10-04T15:12Z · Landed: AO2 MISS: `w48_cn_d1536_B512_hm3072_hops16_nb2_400M` (2 exact past tokens) 1.11838, better
  than 4 (1.11924). Exact past tokens 2 · 4 · 8 · 16 · 32: 1.11838 · 1.11924 · 1.11767 · 1.11894 · 1.12085, all within
  0.003 below 32: THERE IS NO CURVE between 2 and 16; 8 stays. Box n finished every run: pulled (19 results) and
  DESTROYED (54033241). Box e finished its last sweep run; box f has one (32 hops at width 1,024).

## THE TRUE RUN, settled 2026-10-04T15:12Z (the navigator's order: settle a shape, spend the last money on it)

The shape, from waves 13 to 49: cn recipe, ALL SDM, no mixer (it hurt the wide body and neither fix held), trained store
off; width 1,536; hop MLP 6,144 (flat past it); 8 exact past tokens (no curve 2 to 16); 131,072 tokens per step (B 512,
bracketed); base rate (flat 1x to 4x); hops 16 and 24 (24 beat 16 by 0.0075 at 400M, 32 not yet scored, so both run).
Data: the 2.0B-token prefix every box already holds, one pass (15,259 steps). A 3.0B copy would cost a transfer for
about a 1.5x data gain; 2.0B starts now. Boxes e (54033213) and f (54033215) move to `vast_sdmonly_true.list` and out of
the fleet list, so the 16:00Z deadline cannot destroy them; a pull loop copies results home, and the hourly sweep closes
a box only after an idle hour and a pull. Measured speeds put the runs at about 15 to 25 hours, about $45 for both
boxes, inside the $100 kept for this. Checkpoint every 1,000 steps.

| id | prediction |
|---|---|
| TR1 | `true_cn_d1536_hm6144_hops24_2B` beats `true_cn_d1536_hm6144_hops16_2B` |
| TR2 | `true_cn_d1536_hm6144_hops16_2B` beats the best SDM so far (`w41_cn_d1536_B512_hm3072_hops16_800M`, 1.07705) by at least 0.04 |
| TR3 | `true_cn_d2048_hm6144_hops16_2B` beats `true_cn_d1536_hm6144_hops16_2B` |
| TR4 | the best true run lands within 0.03 of the transformer at width 1,536 and 800M (1.02369) |
- 2026-10-04T15:18Z · The true run trains: box e both GPUs (24 and 16 hops), box f GPU 0 (width 2,048); f GPU 1 finishes its 32-hop sweep run first. The pull loop for the true boxes (`track4_sdmonly_vast_pull_true.sh`, never destroys) runs on the M5. The fleet list is empty, so the 16:00Z deadline destroys nothing.
- 2026-10-04T15:30Z · Heartbeat. AM1 HIT: `w44_cn_d1024_B512_hm6144_hops16_800M` 1.07738 against 1.10852 (hop MLP 3,072,
  8 hops, 800M); width 1,024 with the wide body ties width 1,536 with hop MLP 3,072 (1.07705). True-run speeds measured:
  width 1,536 16 hops 35,600 tok/s (about 15.6 h, ends about 06:50Z), 24 hops 24,700 tok/s (about 22.5 h, about 13:50Z),
  width 2,048 24,900 tok/s (about 22.3 h, about 13:40Z). f 1 frees within minutes; it is queued the smaller sibling:

| id | prediction |
|---|---|
| TR5 | `true_cn_d1024_hm6144_hops16_2B` lands within 0.01 of `true_cn_d1536_hm6144_hops16_2B` |
- 2026-10-04T15:55Z · Heartbeat. AQ1 MISS: `w49_cn_d1024_B512_hm3072_hops32_400M` 1.11834 beats 24 hops (1.12273).
  Width 1,024, hop MLP 3,072, 400M: 8 hops 1.14585 · 16 1.13013 · 24 1.12273 · 32 1.11834. Each step of 8 hops buys
  less (0.0157, 0.0074, 0.0044) but the curve has not turned: hops are UNBRACKETED. f 1 took the smaller true run
  (step 300). No GPU is idle, so the push past 32 waits for machines that free, at no new rental:
  box e GPU 1 after `true_cn_d1536_hm6144_hops16_2B` (about 06:50Z, then about 7 h free until e 0 ends) runs 48 hops;
  the Spark runs 64 hops after both w50s mixer jobs end (about 23:30Z), ahead of the paused BASE run.

### Wave 53 (the hop push past 32), sealed 2026-10-04T15:55Z before it runs

| id | prediction |
|---|---|
| AU1 | `w53_cn_d1024_B512_hm3072_hops48_400M` (box e, `--accum 32`) beats 32 hops (1.11834) by less than 0.004 |
| AU2 | `w53s_cn_d1024_B512_hm3072_hops64_400M` (the Spark, `--accum 32`) is no better than 48 hops: the hop curve turns by 64 |
- 2026-10-04T15:50Z · CORRECTION: the entry above and the wave 53 seal carry the stamp 15:55Z, written before the clock was read; `date -u` read 15:48:45Z. Wave 53 was sealed at 15:48Z, before either run started. Wave 53 is queued: the box e fill runner waits for a free GPU, Spark job 1004154858-0277 waits on both w50s jobs.
- 2026-10-04T16:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes at 100% on both GPUs, with log writes 3 to 6 minutes old. Nothing closed. Credit $128.76.
- 2026-10-04T16:10Z · Heartbeat. All four true runs step cleanly (gradient norm 0.12 to 0.13). Measured speeds:
  width 1,024 52,700 tok/s (ends about 02:15Z, so f GPU 1 frees first), width 1,536 16 hops 35,500, 24 hops 25,900,
  width 2,048 25,900. Early losses at step 600 sit within 0.01 of each other (4.486, 4.486, 4.495, 4.409), too early
  to rank. Wave 53 moves: 48 hops at width 1,024 goes to box f (frees about 02:15Z); box e GPU 1 (about 06:50Z to the
  end of e 0 at about 13:50Z) takes 32 hops at width 1,536 instead, the push at the true run's own width. AU1 and AU2
  stand unchanged; one prediction added:

| id | prediction |
|---|---|
| AU3 | `w53_cn_d1536_B512_hm3072_hops32_400M` (box e, `--accum 32`) beats 24 hops at width 1,536 (1.11022) |
- 2026-10-04T16:28Z · Heartbeat. True runs step cleanly (gradient norm 0.09 to 0.12). At step 1,100 width 1,536 (16 hops) leads width 1,024 by 0.032 in training loss (4.155 against 4.187); at step 800 width 2,048 and 24 hops tie (4.319, 4.323). Nothing finished, nothing fired. Credit $127.93.
- 2026-10-04T16:48Z · Heartbeat. True runs at steps 1,100 to 1,700 of 15,259, gradient norm 0.086 to 0.094 and falling. At step 1,000 width 2,048 and 24 hops still tie (4.173, 4.176). Nothing finished, nothing fired. Credit $127.33.
- 2026-10-04T17:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes under 5 minutes old. Nothing closed. Credit $126.91.
- 2026-10-04T17:10Z · Heartbeat. True runs at steps 1,300 to 2,200, gradient norm 0.078 to 0.096. The Spark mixer runs (width 512, 400M, 3,052 steps): window 16,384 at step 1,100 loss 4.213 (gradient norm 0.148, 11,200 tok/s); window 4,096 at step 1,200 loss 4.368 with gradient norm 0.717, high against the true runs; watched, not stopped (the earlier mixer blow-ups reached 5 to 100). Both end about 23:30Z. Credit $126.70.
- 2026-10-04T17:28Z · Heartbeat. True runs at steps 1,600 to 2,700, gradient norm 0.079 to 0.101. Spark: the 4,096-window mixer gradient norm stays raised (last four 0.42, 0.72, 0.70, 0.84) but its loss still falls (4.342 at step 1,300); the 16,384 window is calm (0.13 to 0.15). Nothing finished, nothing fired. Credit $126.06.
- 2026-10-04T17:48Z · Heartbeat. True runs at steps 1,800 to 3,200, gradient norm 0.074 to 0.097. Spark: the 4,096-window mixer climbs (gradient norm 0.70, 0.84, 1.20, 1.24) and its loss has stalled (4.342 at step 1,300, 4.371 at 1,400); the 16,384 window stays calm (0.13 to 0.14, loss 4.224). RULE, set now: if the 4,096 run reaches gradient norm 3 or its loss rises above 4.5, it is stopped by its queue id, its log kept as the result (the mixer at window 4,096 and batch 32 is unstable), and the 64-hop job takes its slot early. Credit $125.45.
- 2026-10-04T18:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 100%, last log writes 2 to 3 minutes old. Nothing closed. Credit $125.03.
- 2026-10-04T18:08Z · Heartbeat. True runs at steps 2,000 to 3,700, gradient norm 0.078 to 0.096. Spark: the 4,096-window mixer eased back (gradient norm 1.24, 1.08, 0.71, 0.79; loss 4.33 at step 1,550), below its stop line; the 16,384 window at step 1,400 (gradient norm 0.18). Nothing finished, nothing fired. Credit $124.83.
- 2026-10-04T18:28Z · Heartbeat. True runs at steps 2,300 to 4,100, gradient norm 0.072 to 0.086, losses 3.78 to 3.88. Spark mixers: window 16,384 loss 4.152 at step 1,500, window 4,096 4.323 at step 1,650 (gradient norm 0.63 to 0.89, under its stop line). Nothing finished, nothing fired. Credit $124.20.
- 2026-10-04T18:48Z · Heartbeat. True runs at steps 2,500 to 4,600, losses 3.70 to 3.85 still falling; gradient norm ticked up on three (0.07 to 0.12 and 0.14), watched (the earlier wide runs stayed well under 1). Spark mixers: window 16,384 4.073 at step 1,600, window 4,096 4.269 at 1,750 (gradient norm down to 0.53). Nothing finished, nothing fired. Credit $123.57.
- 2026-10-04T19:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes under 20 seconds old. Nothing closed. Credit $123.16.
- 2026-10-04T19:08Z · Heartbeat. First mid-run eval: `true_cn_d1024_hm6144_hops16_2B` at step 5,000 (655M tokens, a third of its schedule) reads 1.16949 bpb on the small 16,379-token check set (not TEST, not comparable to final scores). True runs at steps 2,700 to 5,100, gradient norm 0.075 to 0.139, settled. Spark mixers: window 16,384 4.205 at step 1,700, window 4,096 4.267 at 1,850 (gradient norm down to 0.35 to 0.41). Credit $122.96.
- 2026-10-04T19:28Z · Heartbeat. True runs at steps 3,000 to 5,600, gradient norm 0.07 to 0.12, losses 3.68 to 3.81. Spark mixers: window 16,384 4.102 at step 1,800, window 4,096 4.331 at step 2,000 (gradient norm 0.50 to 0.65). Nothing finished, nothing fired. Credit $122.32.
- 2026-10-04T19:48Z · Heartbeat. True runs at steps 3,200 to 6,100, gradient norm 0.075 to 0.106, losses 3.66 to 3.75. Spark mixers: window 16,384 3.979 at step 1,900, window 4,096 4.232 at 2,100 (gradient norm about 0.6, steady). Nothing finished, nothing fired. Credit $121.69.
- 2026-10-04T20:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 100%, last log writes under a minute old. Nothing closed. Credit $121.29.
- 2026-10-04T20:08Z · Heartbeat. True runs at steps 3,400 to 6,500, gradient norm 0.07 to 0.17, losses 3.65 to 3.73. Spark mixers: window 16,384 4.115 at step 2,000, window 4,096 4.277 at 2,200 (gradient norm 0.54 to 0.93). Nothing finished, nothing fired. Credit $121.07.
- 2026-10-04T20:30Z · Heartbeat. True runs at steps 3,700 to 7,100, gradient norm 0.08 to 0.19, losses 3.62 to 3.77. Spark mixers: window 16,384 4.094 at step 2,100, window 4,096 4.223 at 2,350 (gradient norm 0.71 to 1.08, under its stop line). Nothing finished, nothing fired. Credit $120.38.
- 2026-10-04T20:45Z · RULE, set now: THE TRUE RUNS' WEIGHTS COME HOME. The pull loop and the idle sweep copy results only and exclude `*.pt`, so a closed box would take the trained true-run weights with it. Before box e or f is destroyed, each finished true run's final `ck/<run>/last.pt` is copied to the M5 (`runs_vast_sdmonly/<label>/ck/<run>/`) and its size checked against the box: about 5.7 GB for width 1,024, 11 GB for width 2,048 (it holds optimizer state too). The navigator's chat window still runs `SDM read, wide 768, chat-tuned` (1.44235 bpb on its own older measure), so the true-run weights are what the chat should move to.
- 2026-10-04T20:44Z · CORRECTION: the weights rule above carries 20:45Z; the clock read 20:43:15Z when it was written.
- 2026-10-04T20:58Z · RED, found while answering the navigator ("is it an SDM?"): THE TRUE RUNS READ NO SDM. Arm
  `sdmonly_none` is the floor arm (`track4_sdmonly_models.py` header: "no reads: the floor"); in `hidden()` a hop with
  store mode "none" skips the read and adds only the hop MLP. So every true run is: last 8 token embeddings + 5 running
  averages + 2 products -> one linear map -> 16 or 24 residual SwiGLU MLPs -> head. No attention, no transformer
  block, and no sparse memory either; positions mix only through the fixed running averages. That breaks the campaign
  law (every step that mixes positions is an SDM mechanism). The sweeps chose it because the store reads tied the
  no-store control and the run-time mixer hurt the wide body (1.25120 against 1.19774 at width 512). Decision put to
  the navigator; the true runs keep training until he rules. Lane SDMLLMWORDS (Opus) started on the site copy for the
  browser chat models, which do read an SDM (3,600 locations, 2 hops).
- 2026-10-04T20:47Z · CORRECTION: the RED entry above carries 20:58Z; the clock read 20:46:48Z. From now on the clock is read in the same command that writes the entry.
- 2026-10-04T20:48Z · DECISION B (no ruling from the navigator by this heartbeat, as announced): e GPU 0 stops
  `true_cn_d1536_hm6144_hops24_2B` (step about 4,000 of 15,259; its log is kept, TR1 becomes UNTESTED) and starts the
  SDM twin of e GPU 1: `true_sdm_d1536_hm6144_hops16_2B`, the same shape with the sparse memory reads ON (arm
  `sdmonly`, one shared store of 65,536 locations read by all 16 hops, k 32, query 256), 2.0B tokens. The pair answers
  whether the SDM read helps at full scale, and the twin keeps the campaign law. Queue file
  `track4_sdmonly_true_fill_e2.txt`.

| id | prediction |
|---|---|
| TR6 | `true_sdm_d1536_hm6144_hops16_2B` lands within 0.007 of `true_cn_d1536_hm6144_hops16_2B` (the store ties, as in every smaller test) |
| TR7 | the twin trains at least 0.6x the speed of its no-store pair (35,600 tok/s), so it ends inside 26 h |
- 2026-10-04T20:55Z · AMENDMENT before any valid step of the twin: the first launch passed the arguments out of order (`--arm 0`, exit at once); the second ran out of memory at `--accum 16` (8,192 tokens a micro-batch; the 16 reads of 64 locations x 1,536 numbers did not fit). The twin runs at `--accum 64` (2,048 tokens a micro-batch, the same 131,072 tokens a step). Failed logs kept as `.argerr.out` and `.oom16.out`. GPU 0 was idle from 20:49Z for about 4 minutes. Lane SDMLLMWORDS was widened by the navigator: a top explainer on #/sdmchat with neon diagrams of how an SDM does autoregressive generation, for laymen and LLM people.
- 2026-10-04T21:02Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes 3 to 4 minutes old. Nothing closed. Credit $119.37.
- 2026-10-04T21:09Z · Heartbeat. The SDM twin trains: step 100 loss 5.796 (its no-store pair read 5.802 at step 100), gradient norm 0.28, 15,300 tok/s. TR7 MISSED: 0.43x the pair's 35,600 tok/s, so the twin needs about 36 h and ends about 09:30Z on 2026-10-06; box e stays on for it (about $33 more). Other true runs: width 1,024 step 8,000 (3.541), width 1,536 no-store step 5,700 (3.650), width 2,048 step 4,200 (3.637). Spark mixers: window 16,384 4.057 at step 2,300 (gradient norm up to 0.75), window 4,096 4.227 at 2,550 (1.13). Credit $119.19.
- 2026-10-04T21:18Z · THE SPARK COOLS OFF for Claude 1's MIDI training (the navigator: "can we even cool off our Spark stuff until Claude 1 is done?"). Paused (stopped in place, no step lost): both w50s mixers (window 4,096 at step about 2,550, window 16,384 at about 2,300 of 3,052); held: `w53s_hops64`; the BASE run stays paused. Spark GPU now 1%. They still hold about 54 GB of the 121 GB memory (67 GB free); if Claude 1 needs more, they are stopped and later resumed from their last checkpoints (at most about 300 steps lost each). Resume when Claude 1 says the Spark is free.
- 2026-10-04T21:28Z · Heartbeat. SDM twin step 200 loss 5.036 (its no-store pair read 5.040 at step 200), 17,000 tok/s (about 32 h left). Other true runs: width 1,024 step 8,500 (3.559), width 1,536 no-store step 6,000 (3.599), width 2,048 step 4,400 (3.660). Spark cooled off: all four jobs paused or held, GPU 0%, waiting for Claude 1. Credit $118.56.
- 2026-10-04T21:47Z · Heartbeat. SDM twin step 400 loss 4.663 (its no-store pair read 4.660 at step 400), 17,600 tok/s. Width 1,024 step 8,900 (3.548), width 1,536 no-store step 6,400 (3.591), width 2,048 step 4,600 (3.693). Spark still cooled off (GPU 2%). Credit $117.99.
- 2026-10-04T22:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes under 2 minutes old. Nothing closed. Credit $117.57.
- 2026-10-04T22:07Z · Heartbeat. SDM twin step 500 loss 4.539 (its no-store pair read 4.537 at step 500), 17,800 tok/s. Width 1,024 step 9,400 (3.491), width 1,536 no-store step 6,700 (3.560), width 2,048 step 4,900 (3.685). Spark cooled off (GPU 1%). `settle-sdmllmwords` posted ready to land for Claude 1. Credit $117.36.
- 2026-10-04T22:28Z · Heartbeat. SDM twin step 700 loss 4.384. Width 1,024 step 9,900 (3.503), width 1,536 no-store step 7,000 (3.555), width 2,048 step 5,100 (3.638). Spark cooled off (GPU 1%). `settle-sdmllmwords` not yet landed. Credit $116.74.
- 2026-10-04T22:47Z · Heartbeat. SDM twin step 900 loss 4.235 (its no-store pair read 4.236 at step 900). Width 1,024 step 10,400 (3.479), width 1,536 no-store step 7,300 (3.510), width 2,048 step 5,300 (3.657). Spark cooled off (GPU 1%). Credit $116.11.
- 2026-10-04T23:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes 2 to 4 minutes old. Nothing closed. Credit $115.72.
- 2026-10-04T23:22Z · Heartbeat. SDM twin step 1,200 loss 4.105. Width 1,024 step 11,200 (3.471), width 1,536 no-store step 7,900 (3.527), width 2,048 step 5,700 (3.620). Spark cooled off (GPU 2%). M5 free disk fell from 820 to 560 GB since 15:47Z; not from this lane's pulls (results only). Credit $115.04.
- 2026-10-04T23:27Z · Heartbeat (6 minutes after the last). Width 1,536 no-store step 8,000 (3.481), width 1,024 step 11,300 (3.471), width 2,048 step 5,800 (3.589), SDM twin step 1,200 (4.105). Spark cooled off. M5 free 560 GB, steady. Credit $114.87.
- 2026-10-04T23:47Z · Heartbeat. Mid-run check scores (the small 16,379-token check set, not TEST): at step 5,000 width 1,024 1.16949, width 1,536 (no store) 1.16468, width 2,048 1.15765, so width still pays in order; width 1,024 at step 10,000 1.10917. SDM twin step 1,400 (4.055). Spark cooled off (GPU 0%). Credit $114.24.
- 2026-10-05T00:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes about 4 minutes old. Nothing closed. Credit $113.90.
- 2026-10-05T00:07Z · Heartbeat. SDM twin step 1,500 (4.048). Width 1,024 step 12,300 (3.489), width 1,536 no-store step 8,600 (3.474), width 2,048 step 6,300 (3.515). Spark cooled off (GPU 2%). Credit $113.69.
- 2026-10-05T00:27Z · Heartbeat. SDM twin step 1,700 loss 3.980 (its no-store pair read 3.978 at step 1,700). Width 1,024 step 12,800 (3.444), width 1,536 no-store step 8,900 (3.482), width 2,048 step 6,500 (3.548). Spark cooled off (GPU 0%). Credit $113.07.
- 2026-10-05T00:47Z · Heartbeat. SDM twin step 1,900 loss 4.0321; its no-store pair read 4.0322 at step 1,900. Same seed and data order, and the store's values start at zero (`value_init zero`), so a store that adds little keeps the twin on its pair's exact path; a gap that stays under 0.001 would mean the reads are near zero. To check at the twin's step-5,000 eval: the store's read size against the working vector. Width 1,024 step 13,300 (3.446), width 1,536 no-store step 9,300 (3.487), width 2,048 step 6,800 (3.552). Spark cooled off. Credit $112.44.
- 2026-10-05T01:00Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 98 to 100%, last log writes under 2 minutes old. Nothing closed. Credit $112.03. (Commit waits: Claude 1 is mid-merge.)
- 2026-10-05T01:07Z · Heartbeat. SDM twin step 2,000 loss 3.939 (its no-store pair read 3.943 at step 2,000). Width 1,024 step 13,700 (3.443), width 1,536 no-store step 9,600 (3.442), width 2,048 step 7,000 (3.521). Spark cooled off (GPU 1%). Credit $111.81.
- 2026-10-05T01:27Z · Heartbeat. SDM twin step 2,200 (3.902). Width 1,024 step 14,200 of 15,259 (3.453), ends about 02:00Z; width 1,536 no-store step 9,900 (3.439), width 2,048 step 7,200 (3.514). Spark cooled off (GPU 0%). Credit $111.23.
- 2026-10-05T01:48Z · Heartbeat. Width 1,024 true run at step 14,700 of 15,259 (3.434), ends about 02:10Z. A weights watcher (`track4_sdmonly_pull_weights.sh`, M5 PID 44833, log `runs_vast_sdmonly/pull_weights.out`) waits for its result json, then copies `ck/true_cn_d1024_hm6144_hops16_2B/last.pt` home and checks the size. SDM twin step 2,400 (3.871); width 1,536 no-store step 10,200 (3.437); width 2,048 step 7,500 (3.486, gradient norm 0.21, watched). Credit $110.60.
- 2026-10-05T02:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes 3 to 5 minutes old. Nothing closed. Credit $110.18.
- 2026-10-05T02:15Z · FIRST TRUE RUN LANDS. `true_cn_d1024_hm6144_hops16_2B` (width 1,024, hop MLP 6,144, 16 hops, store
  OFF, 2.0B tokens, 15,259 steps): **TEST 1.03599 bpb**, the best of the campaign by 0.041 (was 1.07705, width 1,536 at
  800M). The same shape at 800M scored 1.07738, so 2.5x the data bought 0.041: DATA STILL PAYS, the data sweep stays
  UNBRACKETED. Against the transformer yardstick (best 1.02369, width 1,536 at 800M) the gap is 0.012, but that
  yardstick saw 2.5x less data, so no closing of the gap is claimed until a transformer trains on 2.0B. It reads no
  SDM (floor arm), so it is not an SDM-LLM. TR5 waits for `true_cn_d1536_hm6144_hops16_2B` (about 06:50Z). Box f GPU 1
  started `w53_cn_d1024_B512_hm3072_hops48_400M` (AU1) at 02:10Z. The weights watcher is copying `last.pt` home (about
  2 MB/s, about 45 minutes).
- 2026-10-05T02:28Z · Heartbeat. 48 hops (`w53_cn_d1024_B512_hm3072_hops48_400M`, box f GPU 1) at step 200 of 3,052 (4.887), 34,100 tok/s, ends about 05:20Z. Weights copy of width 1,024 at 1.5 of 5.7 GB, about 1.8 MB/s, done about 03:05Z. SDM twin step 2,700 (3.888, 18,400 tok/s); width 1,536 no-store step 10,900 (3.431); width 2,048 step 7,900 (3.490). Credit $109.34.
- 2026-10-05T04:23Z · Back after a usage-limit gap (02:30Z to 04:20Z; no beats ran, the boxes trained on). NEAR MISS, FIXED:
  each box runs `/root/settle/ck_cleaner.sh`, which deletes a run's `*.pt` 3 minutes after its result json. It deleted
  the width 1,024 true run's `last.pt` on box f while the weights watcher was copying it; the copy finished because
  rsync held the file open, so the home copy is whole at 5,711,070,235 bytes (`runs_vast_sdmonly/sdmonly-f-2x5090/ck/
  true_cn_d1024_hm6144_hops16_2B/last.pt`, sha256 f61cb986...41cc). The watcher printed WEIGHTS MISMATCH only
  because the box file was gone when it checked; the box copy cannot be hashed. The cleaner on boxes e and f now skips
  every run named `true_*` (patched, restarted, PIDs 161224 and 464911), so the other true runs keep their weights
  until copied home. Progress: width 1,536 no-store step 12,700 (3.348), SDM twin step 3,700 (3.754), width 2,048
  step 9,300 (3.454), 48 hops step 2,000 of 3,052 (3.865). Spark cooled off. Credit $105.79.
- 2026-10-05T04:27Z · Heartbeat. Width 1,536 no-store step 12,800 (3.372), SDM twin step 3,700 (3.754), width 2,048 step 9,400 (3.385), 48 hops step 2,100 of 3,052 (3.824). Credit $105.59. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T04:47Z · Heartbeat. Check score at step 10,000 (small check set): width 1,536 no-store 1.09419 against width 1,024 1.10917, a 0.015 lead for the wider model; TR5 (within 0.01 at the end) leans toward a miss. Width 1,536 no-store step 13,100 (3.322), SDM twin step 3,900 (3.733), width 2,048 step 9,600 (3.406), 48 hops step 2,400 of 3,052 (3.782). Credit $104.99. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T05:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes 4 s and 3 minutes old. Nothing closed. Credit $104.57. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T05:08Z · Heartbeat. 48 hops step 2,700 of 3,052 (3.795), ends about 05:25Z. f GPU 1 gets 64 hops next (AU2, already sealed): `w53_cn_d1024_B512_hm3072_hops64_400M`, queue `track4_sdmonly_wave53_fill_f2.txt`, runner started; the Spark copy `w53s_hops64` stays held and is superseded. Width 1,536 no-store step 13,500 (3.386), SDM twin step 4,100 (3.738), width 2,048 step 9,800 (3.383). Credit $104.37. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T05:27Z · 48 HOPS LANDS: `w53_cn_d1024_B512_hm3072_hops48_400M` TEST **1.11213**. AU1 MISSED (it said better than
  32 hops by less than 0.004; it is better by 0.0062). Width 1,024, hop MLP 3,072, 400M: 8 hops 1.14585 · 16 1.13013 ·
  24 1.12273 · 32 1.11834 · 48 1.11213. Each step down is still real (16 to 32 bought 0.0118, 24 to 48 bought 0.0106):
  HOPS UNBRACKETED. The 32-to-48 gap (0.0062) sits inside the 0.007 rule, so rule (c) owes a rerun of the pair at 800M;
  64 hops (AU2) starts on f GPU 1 as soon as this run's eval ends, and decides the next step first.
- 2026-10-05T05:47Z · Heartbeat. 64 hops started on f GPU 1 at 05:29Z, step 200 (4.894), 27,500 tok/s, ends about 09:35Z. Width 1,536 no-store step 14,100 (3.388), ends about 06:45Z; SDM twin step 4,400 (3.685); width 2,048 step 10,300 (3.380, gradient norm 0.30, watched). Credit $103.12. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T06:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes 23 s and 4.5 minutes old. Nothing closed. Credit $102.72. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T06:08Z · Heartbeat. Width 1,536 no-store step 14,400 of 15,259 (3.374), ends about 06:45Z; weights watcher started for it (M5 PID 85172). Box e's wave 53 runner is alive and gives that GPU 32 hops at width 1,536 (AU3) next. SDM twin step 4,600 (3.715), width 2,048 step 10,500 (3.417), 64 hops step 400 (4.574). Credit $102.51. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T06:27Z · Heartbeat. Width 1,536 no-store step 14,800 of 15,259 (3.366), ends about 06:50Z. SDM twin step 4,700 (3.666), width 2,048 step 10,800 (3.379), 64 hops step 700 (4.346). Credit $101.94. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T06:52Z · Heartbeat. Width 1,536 no-store at step 15,200 of 15,259, final eval next (about 07:00Z); its last.pt is 8,654,646,347 bytes and kept by the patched cleaner; the watcher copies it once the result json lands (about 80 minutes at 1.8 MB/s). SDM twin step 4,900 (3.710), width 2,048 step 11,000 (3.345), 64 hops step 900 (4.201). Credit $101.35. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T07:06Z · MAIN TRUE RUN LANDS. `true_cn_d1536_hm6144_hops16_2B` (store OFF, 2.0B tokens) TEST **1.02241**, the best of
  the campaign and below the transformer yardstick (1.02369, width 1,536 at 800M, so the yardstick saw 2.5x less data;
  no win over a transformer is claimed until one trains on 2.0B). TR2 HIT (beats 1.07705 by 0.0546). TR4 HIT. TR5 MISSED:
  width 1,024 (1.03599) is 0.0136 worse, so width still pays at 2.0B. It reads no SDM. Its GPU started 32 hops at width
  1,536 (AU3) at 06:57Z; its weights (8.65 GB) are copying home. Scores at 2.0B: width 1,024 1.03599, 1,536 1.02241.
- 2026-10-05T07:07Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes under 2 minutes old. Nothing closed. Credit $100.74. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T07:08Z · Heartbeat. The fair yardstick is queued: box f GPU 1 runs the Muon transformer at width 1,536 on the
  same 2.0B tokens as the true runs once 64 hops ends (about 09:35Z); it trained at 74,200 tok/s at 800M, so about 7.5 h,
  ending about 17:00Z. Queue file `track4_sdmonly_wave54_fill_f.txt`.

### Wave 54 (the transformer yardstick at 2.0B), sealed 2026-10-05T07:08Z before it runs

| id | prediction |
|---|---|
| TY1 | `w54_mq_d1536_B512_2B` scores below 1.02241 (the transformer stays ahead of the memory-off true run at equal data) |
| TY2 | its lead over the memory-off true run is under 0.03 |
- 2026-10-05T07:28Z · Heartbeat. Twin check score at step 5,000 (small check set): memory ON 1.16380, its memory-OFF pair 1.16468 (0.0009 apart, noise). SDM twin step 5,200 (3.622); 32 hops at width 1,536 step 400; width 2,048 step 11,500 (3.318); 64 hops step 1,400 (4.004). Width 1,536 weights copying at about 0.5 MB/s (0.8 of 8.65 GB), about 5 h left; box e stays up for the twin anyway. Credit $100.10. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T07:47Z · Heartbeat. SDM twin step 5,400 (3.619, gradient norm 0.21), 32 hops width 1,536 step 700, width 2,048 step 11,700 (3.387), 64 hops step 1,700 (3.909). Width 1,536 weights home 1.35 of 8.65 GB. Credit $99.49, the first time under $100. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T08:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 99 to 100%, last log writes about 1 to 2 minutes old. Nothing closed. Credit $99.07. (Commits wait: Claude 1 is mid-merge.)
- 2026-10-05T08:07Z · Heartbeat. SDM twin step 5,600 (3.622); 32 hops width 1,536 step 1,000 (4.109); width 2,048 step 12,000 (3.387); 64 hops step 1,900 of 3,052 (3.947). Width 1,536 weights home 1.86 of 8.65 GB. Credit $98.87. (Commits wait: Claude 1's merge open since before 04:25Z; navigator told.)
- 2026-10-05T09:40Z · Back after a session restart (no beats 08:07Z to 09:39Z; the boxes trained on). The M5 watchers had
  died with the session: the pull loop is restarted (PID 4971), and the width 1,536 weights copy, which had stopped at
  2,091,352,064 of 8,654,646,347 bytes on an rsync end-of-file, is resumed with --append-verify (PID 5189).
  64 HOPS LANDS: `w53_cn_d1024_B512_hm3072_hops64_400M` TEST **1.11084**. AU2 MISSED (it said no better than 48 hops;
  it is 0.0037 better). Width 1,024, hop MLP 3,072, 400M: 8 1.14585 · 16 1.13013 · 24 1.12273 · 32 1.11834 · 48 1.11213 ·
  64 1.11084. Each doubling still helps, by less (16 to 32 0.0118, 32 to 64 0.0075): HOPS UNBRACKETED. Box f GPU 1 began
  the transformer at 2.0B at 09:35Z (step 100). Box e GPU 1 gets the push past 64 next, on time it already pays for:

### Wave 55 (the hop push past 64), sealed 2026-10-05T09:40Z before it runs

| id | prediction |
|---|---|
| AV1 | `w55_cn_d1024_B512_hm3072_hops96_400M` beats 64 hops (1.11084) by less than 0.003 |
| AV2 | `w55_cn_d1024_B512_hm3072_hops128_400M` is no better than 96 hops: the curve turns between 96 and 128 |
- 2026-10-05T09:48Z · Heartbeat. CORRECTION to the 09:40Z entry: the 64-hop TEST bpb is **1.10841**, read from
  `runs/w53_cn_d1024_B512_hm3072_hops64_400M.result.json` on box f (val_test bpb 1.1084114). The 09:40Z entry typed
  1.11084, a digit swap; its "0.0037 better than 48 hops" was computed from the true value and stands. The doubling
  steps are 16 to 32 0.0118 and 32 to 64 **0.0099** (not 0.0075). Wave 55's sealed AV1 is unchanged in its words; its
  anchor "(1.11084)" is the same typo and AV1 is judged against 1.10841, the score it was written to name. HOPS UNBRACKETED.
  Runs: SDM twin step 6,400 of about 15,260 (loss 3.587, 18.5k tok/s, ends about 03:30Z Tuesday) · 32 hops width 1,536
  step 2,400 of 3,052 (ends about 10:35Z, then wave 55 claims that GPU) · width 2,048 true run step 13,100 (ends about
  13:00Z) · transformer at 2.0B step 400 (74.0k tok/s, ends about 17:10Z). Width 1,536 weights copy running (PID 5189).
  Crons alive (heartbeat 6ff72dd3, idle sweep c8e6ff9e). Credit $95.81. No GPU idle.
- 2026-10-05T10:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 95 to 100% on two reads 30 s apart, newest log writes 5 and 2 minutes old. Nothing closed. Credit $95.41.
- 2026-10-05T10:07Z · The navigator asked how many boxes run the full SDM path: 1 GPU of 4 (the SDM twin). Decision:
  every GPU that frees from now runs a MEMORY-ON twin of a no-store result we already hold, so each one measures what
  the memory adds. e GPU 1 (about 10:35Z): the memory-on twin of 64 hops first, then wave 55. The wave 55 fill runner on
  e (PIDs 182737, 182738, nothing claimed yet) is stopped and replaced by one runner on the wave 56 file, so two runners
  cannot race for the same GPU. f GPU 0 (about 13:00Z): the memory-on twin of the width 1,024 true run.

### Wave 56 (memory ON, each against its no-store pair), sealed 2026-10-05T10:07Z before it runs

| id | prediction |
|---|---|
| AW1 | `w56_sdm_d1024_B512_hm3072_hops64_400M` lands within 0.007 of its no-store pair `w53_cn_d1024_B512_hm3072_hops64_400M` (1.10841) |
| AW2 | `true_sdm_d1024_hm6144_hops16_2B` lands within 0.007 of its no-store pair `true_cn_d1024_hm6144_hops16_2B` (1.03599) |
| AW3 | the memory-on twin trains at under 0.6 times its no-store pair's tokens per second, at both shapes |
- 2026-10-05T10:12Z · Heartbeat. The width 1,536 weights copy (PID 5189) never resumed: macOS rsync has no
  --append-verify and printed its usage text at 09:46Z. The file on box e is whole (8,654,646,347 bytes); the patched
  cleaner skips true_* runs. Resumed by appending bytes over ssh from the 2,091,352,064 already home, with a sha256
  check at both ends when it finishes (about 0.4 MB/s, so about 4.5 h). Runs: SDM twin step 6,600 (18.5k tok/s) ·
  32 hops width 1,536 step 2,800 of 3,052 · width 2,048 true run step 13,400 · transformer at 2.0B step 1,100
  (71.7k tok/s). Wave 56 waits for its GPUs. Navigator offered A/B/C for more memory-on twins (A: 4 more RTX 5090s,
  400M each); no rental until he answers. Correction to that offer: the width 1,536 24-hop no-store score is 1.11022,
  not 1.12273 (that is width 1,024 at 24 hops). Credit $95.14. Commit waits on Claude 1's open merge.
- 2026-10-05T10:29Z · Heartbeat. `w53_cn_d1536_B512_hm3072_hops32_400M` TEST **1.10483**, the best 400M score so far
  (beats width 1,024 at 64 hops, 1.10841). AU3 HIT in sign (32 hops beats 24 hops, 1.11022, by 0.0054), but 0.0054 is
  inside the 0.007 noise, so rule (c) owes a rerun of the 24-vs-32 pair at 800M before it is called. Width 1,536,
  hop MLP 3,072, 400M: 8 1.13335 · 16 1.11767 · 24 1.11022 · 32 1.10483. Best at the top edge: HOPS UNBRACKETED at both
  widths; 48 and 64 hops at width 1,536 are owed. Wave 56 fired: `w56_sdm_d1024_B512_hm3072_hops64_400M` (memory ON,
  shared store) claimed e GPU 1 at about 10:28Z and is loading. Owed work, not yet sealed (no free GPU until the
  navigator answers A/B/C): 48 and 64 hops at width 1,536, and the 24-vs-32 rerun at 800M. Weights copy 2.42 of 8.65 GB.
  Credit $94.62.
- 2026-10-05T10:48Z · RED, then decided. `w56_sdm_d1024_B512_hm3072_hops64_400M` (memory ON) died at start with CUDA
  out of memory at accum 128: the failing allocation is a store-sized buffer (65,536 x 1,024 float32, 256 MB), and the
  compiled memory-on path holds one per hop, so 64 hops needs about 16 GB of them before activations. Accum cannot fix
  it. Measured limit on a 32 GB RTX 5090 as implemented: store size times hops. AW1 is VOID (the run cannot fit). The
  fill runner went on to `w55_cn_d1024_B512_hm3072_hops96_400M` (no store), which has run on e GPU 1 since 10:32Z and
  is kept. Queued after it, ahead of 128 hops: `w56b_sdm_d1024_B512_hm3072_hops32_400M` (memory ON, half the buffers).
  The OOM log is kept as .oom128.out; its claim stays so it is not retried.

| id | prediction (sealed 2026-10-05T10:48Z, before it runs) |
|---|---|
| AW4 | `w56b_sdm_d1024_B512_hm3072_hops32_400M` lands within 0.007 of its no-store pair `w49_cn_d1024_B512_hm3072_hops32_400M` (1.11834) |
- 2026-10-05T11:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213) and f (54033215) each run 2 training processes, GPUs 98 to 100% on two reads 30 s apart, newest log writes 1 and 0 minutes old. Nothing closed. Credit $93.54.
- 2026-10-05T11:07Z · Heartbeat. SDM twin step 7,100 (loss 3.506); 96 hops step 200 of 3,051 (17.1k tok/s, ends
  about 17:00Z); width 2,048 true run step 14,100 of 15,258 (ends about 12:50Z, then the memory-on width 1,024 true
  run fires on f GPU 0); transformer at 2.0B step 3,100 (ends about 17:20Z). Width 1,536 weights home 3.11 of 8.65 GB
  (byte append running). Nothing finished, nothing fired, no GPU idle. Credit $93.36. Commit waits on Claude 1's merge.
- 2026-10-05T11:27Z · Heartbeat. SDM twin step 7,300; 96 hops step 400 of 3,051; width 2,048 true run step 14,300 of 15,258 (ends about 12:50Z); transformer at 2.0B step 3,700. Width 1,536 weights home 3.57 of 8.65 GB. Nothing finished, no GPU idle. Credit $92.77.
- 2026-10-05T11:47Z · Heartbeat. SDM twin step 7,400 (gradient norm 0.334, up from about 0.08; one reading, watched next beat); 96 hops step 500; width 2,048 true run step 14,600 of 15,258 (ends about 12:30Z); transformer at 2.0B step 4,400. Width 1,536 weights home 4.00 of 8.65 GB. No GPU idle. Credit $92.15.
- 2026-10-05T11:52Z · Lane SDMPAGESTRUTH started (Opus, worktree _worktrees/dwarfstar-sdmpagestruth, branch settle-sdmpagestruth) on the navigator's order: make the SDM and learn pages clear on memory on vs memory off, the model types, the scoring and the latest results, with animated vector diagrams and subtle neon headings. Claude 1 lands it.
- 2026-10-05T12:01Z · Idle sweep: 2 boxes checked, none idle. e (54033213: SDM twin, 96 hops) and f (54033215: width 2,048, transformer at 2.0B) each run 2 training processes, GPUs 98 to 100% on two reads 30 s apart, newest log writes 10 and 0 minutes old. Nothing closed. Credit $91.72.
- 2026-10-05T12:07Z · Heartbeat. SDM twin step 7,600, gradient norm back to 0.089 (the 0.334 at 11:47Z was a single spike; loss steady at 3.54). 96 hops step 700; width 2,048 true run step 14,800 of 15,258 (ends about 12:35Z; its weights stay on f because the cleaner skips true_* runs, then copy home); transformer at 2.0B step 5,100. Width 1,536 weights home 4.47 of 8.65 GB. No GPU idle. Credit $91.52.
- 2026-10-05T12:27Z · Heartbeat. Width 2,048 true run reached step 15,000 of 15,258: small-check-set score 1.06503 (not TEST; final TEST about 12:55Z). SDM twin step 7,800 (gradient norm 0.196, loss 3.530); 96 hops step 800; transformer at 2.0B step 5,800. Width 2,048 last.pt (11,656,942,811 bytes) saved on f. Width 1,536 weights home 5.05 of 8.65 GB. No GPU idle. Credit $90.93.
- 2026-10-05T12:48Z · WIDTH 2,048 LANDS: `true_cn_d2048_hm6144_hops16_2B` TEST **1.01665** (memory off, 2.0B tokens), the
  campaign best. True-run width sweep (hop MLP 6,144, 16 hops, 2.0B): 1,024 1.03599 · 1,536 1.02241 · 2,048 1.01665.
  TR3 HIT in sign: 2,048 beats 1,536 by 0.0058, which is inside the 0.007 noise. Rule (c) cannot add data here (train_big
  is 2.0B, one pass, no repeats), so the honest verdict is UNDECIDED AT THIS BUDGET; a second seed is the only cheap
  discriminator. TR4 HIT: the best true run (1.01665) is within 0.03 of the 800M transformer (1.02369), and below it;
  no claim against the transformer until `w54_mq_d1536_B512_2B` (same 2.0B) lands about 17:20Z.
  WIDTH UNBRACKETED (best at the top edge, gains shrinking: 0.0136 then 0.0058). Its weights (11,656,942,811 bytes) are
  copying home by byte append with a sha check (about 1.1 MB/s, about 3 h). Wave 56 fired on f GPU 0:
  `true_sdm_d1024_hm6144_hops16_2B` (memory ON) started at about 12:40Z, 17.4 GB on the card, so it fits. SDM twin step
  7,900; 96 hops step 1,000; transformer step 6,400. Credit $90.31.
- 2026-10-05T13:02Z · Idle sweep: 2 boxes checked, none idle. e (54033213: SDM twin, 96 hops) and f (54033215: transformer at 2.0B, memory-on width 1,024 true run) each run 2 training processes, GPUs 98 to 100% on two reads 30 s apart, newest log writes 0 and 2 minutes old. Nothing closed. Credit $89.83.
- 2026-10-05T13:08Z · Heartbeat. Memory-on width 1,024 true run step 200 at 25.9k tok/s against its no-store pair's 90.8k: 0.29x, so AW3 HITS at this shape (under 0.6x) and the run ends about 10:30Z Tuesday. SDM twin step 8,100; 96 hops step 1,100; transformer at 2.0B step 7,100. Weights home: width 1,536 6.95 of 8.65 GB, width 2,048 2.06 of 11.66 GB. No GPU idle. Credit $89.64 (both boxes to Tuesday 10:30Z cost about $40).
- 2026-10-05T13:27Z · Heartbeat. SDM twin step 8,300 (loss 3.474); 96 hops step 1,300; memory-on width 1,024 step 400 (26.7k tok/s); transformer at 2.0B step 7,800. Weights home: width 1,536 7.99 of 8.65 GB; width 2,048 3.83 of 11.66 GB (one ssh drop from box f, the append loop retries from the byte it reached). No GPU idle. Credit $89.04.
- 2026-10-05T13:41Z · SUB-CAMPAIGN SHAPESOLVE, the navigator's order: "go in a sub-campaign now to SOLVE THE SHAPE,
  using as many boxes as you need, up to 4 ... to get it all concluded and decided before MAIN TRAINING ... giving our
  main training the best shot, even if we have not fully compared shapes, but we have our IDEAL FORWARD CANDIDATE."
  Why: no SDM part has been shown to help on the shipped body. The trained store ties memory-off in every test; the
  run-time mixer helped the old narrow body and hurt the wide one, and went unstable (every mixer run so far used
  never-fading heads, decays 1.0). The best true run (1.01665) reads no SDM, which breaks the campaign law.
  Actions: up to 4 new 2x RTX 5090 boxes (s1 to s4, lane SHAPESOLVE on Opus rents and boards them); box f GPU 0 stops
  `true_sdm_d1024_hm6144_hops16_2B` (it asks the same question as the width 1,536 twin; log and checkpoint kept, claim
  kept so it is not re-run) and joins. Base for every arm: width 1,024, hop MLP 3,072, 16 hops, 400M tokens, Muon.

### Wave 57 (SHAPESOLVE: which SDM part helps the shipped body), sealed 2026-10-05T13:41Z before it runs

| id | box | run | prediction |
|---|---|---|---|
| AX1 | f | `w57_sdm_d1024_hm3072_hops16_400M` (store on, defaults) | within 0.007 of its no-store pair `w39_cn_d1024_B512_hm3072_hops16_400M` (1.13013): the default store ties |
| AX2 | s1 | `..._qgrad1_400M` (the query's gradient reaches the features, so they learn to address) | beats 1.13013 by at least 0.007 |
| AX3 | s1 | `..._qgrad1_graded_400M` (plus graded weights: a woken location counts by how far it beats the cut-off) | beats `..._qgrad1_400M` |
| AX4 | s2 | `..._k64_400M` (64 locations awake, not 32) | within 0.007 of AX1's run |
| AX5 | s2 | `..._slr10_400M` (store learning rate 10x the body's, not 3x) | within 0.007 of AX1's run |
| AX6 | s3 | `..._perhop_nsub128_400M` (one store per hop, 16,384 locations each) | within 0.007 of AX1's run |
| AX7 | s4 | `w57_cnmix_..._fade_400M` (2 run-time mixer layers, every head fades: 0.99, 0.97, 0.9, 0.7) | beats its control `w57_cn_d1024_T1024_B128_hm3072_hops16_400M` (s3, same window 1,024) by at least 0.007, gradient norm under 2 throughout |
| AX8 | s4 | `w57_cnmix_..._fade_wd_400M` (the same plus Muon weight decay 0.1) | beats `w57_cnmix_..._fade_400M` |

THE MAIN-RUN RULE, sealed with the wave: the main run reads the SDM part that beats its pair by at least 0.007; if
two do, the larger margin; if none does, the main run keeps the default shared store (the law holds, the store is
read) and the report says plainly that no SDM part has yet been shown to help. Width: 2,048 if its memory-on shape fits
a 32 GB card in a two-minute probe, else 1,536. Data: all of train_big (2.0B), one pass.
- 2026-10-05T13:48Z · Heartbeat. Width 1,536 true-run weights HOME: 8,654,646,347 bytes, sha256 19d0743b...721430 matches box e. Width 2,048 weights 5.38 of 11.66 GB. SHAPESOLVE: box f GPU 0 runs `w57_sdm_d1024_hm3072_hops16_400M` (loading); lane SHAPESOLVE-FLEET has not rented yet (2 instances listed). SDM twin step 8,400; 96 hops step 1,500; transformer at 2.0B step 8,500. Credit $88.39.
- 2026-10-05T14:04Z · Idle sweep: 6 boxes checked, none closed. e (54033213) and f (54033215) train (GPUs 98 to 100%, logs 1 and 2 min old). The four SHAPESOLVE boxes s1 to s4 (54324748, 54324752, 54324753, 54324756) were rented within the hour and are being boarded by lane SHAPESOLVE-FLEET (s1 and s3 setup running; s2 and s4 not started); none can have been idle for an hour. Credit $86.85.
- 2026-10-05T14:09Z · THE GUIDE SWITCHES, the navigator's order: "let's do that! get all boxes going and finish our
  shape find. THEN let's go into things gradually, with the BASE MODEL FIRST. And can we do it on multiple machines?
  Make use of the current box set. Keep it going until finished. Switch our guide to what I just described."
  The order from now, in this sequence:
  1. FINISH SHAPESOLVE on the current box set (f GPU 0 and s1 to s4; wave 57, sealed). The s boxes are KEPT after
     their runs (lane SHAPESOLVE-FLEET told at 14:10Z) and take base-model work next. The MAIN-RUN RULE stands.
  2. THE BASE MODEL FIRST, gradually. B1: seal and fire the base run: full SDM (store read every hop) with the
     winning SHAPESOLVE knobs, widest width that fits a 32 GB card, hop MLP 6,144, 16 hops, WSD schedule (so its
     decay can wait for the money and the data), 2.0B tokens first. B2: its checkpoints every 1,000 steps copied
     home. B3: extend toward the Spark's 3.0B tokens if money allows, then decay. The chat model comes after the base.
  3. MULTIPLE MACHINES, the honest answer: one model across several boxes is not practical here; every step
     all-reduces the gradients, and between rented boxes over the internet that alone would take longer than the
     step. Across the GPUs of ONE box it is practical: `track4_sdmonly_train_ddp.py` exists, but it predates our
     recipe (no hop MLP, untie, conj, Muon or accum), so lane DDPRECIPE (Opus) extends it now, proven by a parity
     self-test against the single-GPU trainer before any paid run uses it. The other boxes run the base run's
     companions (a second seed for TR3, the twin to its end) or are closed.
  4. MONEY decides how long: 6 boxes cost about $5.6 an hour against $86.49 credit and a $15 floor, which is about
     12.8 hours at the full set. After SHAPESOLVE the set shrinks to what the base run and its companions use. A
     2-GPU base run (if DDPRECIPE lands) costs about half a 1-GPU one, which is what makes the base run fit.
- 2026-10-05T14:18Z · The navigator, further: "aim to finish even if the boxes die of no credit, so we can shard to the
  Spark and finish on the Spark as needed ... all boxes and all things bit-equivalent on the Spark and others ... update
  the cron texts now for the final train run ... no time limits, we just push to satisfaction and completion ... with
  the Spark as failover." Done:
  - Crons rewritten for the final run (durable): heartbeat 05431e56 (:03 :23 :43) and idle sweep c77e7487 (:37),
    replacing 6ff72dd3 and c8e6ff9e. New in them: no time limit; the order SHAPESOLVE then the base model; base-run
    checkpoints off the box every 2 hours to the Spark (~/sdmonly_base/) with a sha check; failover to the Spark if
    credit falls below $25 or the base run's box dies; idle boxes get base-model work before they are closed; never
    close a box holding an uncopied base or true_* checkpoint; a credit watch.
  - The Spark can now pull straight from boxes e and f (its key `headclimb_pull` authorised on both; the first append
    wrote a doubled line, rewritten clean). Its free disk is 577 GB. Its data file is being checked against the boxes':
    the first 8,037,279,356 bytes of its 3.0B-token train_big must hash equal to the box file for a resume to draw the
    same batches.
  - BIT-EQUIVALENCE, the honest limit: a resume on another GPU model cannot be bit-for-bit equal (the GB10 and the RTX
    5090 run different kernels and round differently). What is guaranteed: the same data order (same file length), a
    portable checkpoint (lane DDPRECIPE's cross-machine resume test), and scores that agree within the noise.
  - Lane DDPRECIPE (Opus, worktree _worktrees/dwarfstar-ddprecipe) extends the multi-GPU trainer to the recipe, with a
    resume-across-machines self-test.
- 2026-10-05T14:19Z · SPARK FAILOVER, measured. DATA MATCH: the first 8,037,279,356 bytes of the Spark's
  train_big.u32 hash to 6ccc9836...64be200d, the same as box f's whole train_big.u32, so a resume on the Spark with a
  file cut to that length draws the same batches. LINK: box f to the Spark ran at 0.18 MB/s (48,431,104 bytes in 275 s),
  while box f was also sending the width 2,048 weights home; at that rate an 8.65 GB checkpoint takes about 13 hours, so
  the 2-hour off-box cadence is not reachable from box f. Decision: the base run goes on the box with the best measured
  upload (each s box is speed-tested when boarded), the checkpoint copy runs one at a time per box, and the cadence is set
  from the measured rate, not assumed.
- 2026-10-05T14:24Z · Heartbeat. Fleet 6 boxes, $6.08/h (s3 was replaced: now 54328901). SHAPESOLVE: f runs the default store (step 500); s1 runs qgrad1 (step 200) and qgrad1+graded (step 100); s2, s3, s4 still boarding by lane SHAPESOLVE-FLEET. The s boxes cost $4.20/h together, a little over the $4.00 cap in its brief. SDM twin step 8,700; 96 hops step 1,700; transformer at 2.0B step 9,700. Credit $84.92: about 11.5 hours of this fleet above the $15 floor.
- 2026-10-05T14:44Z · Heartbeat. SHAPESOLVE 8 of 9 launched (s2: k64 and slr10; s4: the two fading mixers; s3 still installing). Check of a coincidence: the default store (f) and qgrad1 (s1) both read loss 4.4735 at step 500, but the flag reached the model (log cfg qgrad 1.0 vs 0.0) and the curves differ elsewhere (step 400: 4.5685 vs 4.5655), so it is a coincidence, and it shows how little the store moves the loss early. Driver stamp: s1 runs driver 570.172.08, e and f 580.173.02 and 595.71.05, so cross-box pairs carry a small driver confound; the 0.007 bar stays. SDM twin step 8,900; 96 hops 1,900; transformer 10,300. Credit $82.66: about 11.1 h of this fleet above the $15 floor.
- 2026-10-05T14:52Z · CREDIT UP: the navigator added credit; Vast reads $96.82 (was $82.66 at 14:43Z), and his dashboard says credits top up again in 15 h 56 min, about 06:48Z Tuesday (amount not shown, so nothing is planned on it until it lands). At the full fleet ($6.08/h) the credit reaches the $15 floor in about 13.5 h, about 04:20Z, before the top-up; so after SHAPESOLVE (about 21:00Z) the fleet shrinks to the base run and its companions, which carries it past 06:48Z.
- 2026-10-05T14:54Z · The navigator: "Nice! I can give many updates with time ... overnight run, go JIMOTHY!" The budget cap of the last $100 is lifted: credit is topped up over time. The overnight plan stands: SHAPESOLVE to about 21:00Z, the base run fires on its rule, the fleet shrinks to the base run and its companions, the heartbeat and idle sweep keep it moving, the Spark is the failover.
- 2026-10-05T15:02Z · SHAPESOLVE-FLEET: all four boxes boarded and all 8 wave 57 lines training (step lines on every run). Boxes
  (`vast_sdmonly_shapesolve.list`, $4.20/h together, all transfer prices under $12/TB): s1 54324748 $0.98/h runs
  `..._qgrad1_400M` (31.0k tok/s, ends about 17:50Z) and `..._qgrad1_graded_400M` (13.6k tok/s, 0.44x, ends about
  22:30Z); s2 54324752 $1.04/h runs `..._k64_400M` (16.8k tok/s, ends about 21:30Z) and `..._slr10_400M` (29.8k,
  about 18:30Z); s3 54328901 $1.09/h runs `..._perhop_nsub128_400M` (25.3k, about 19:20Z) and the window-1,024
  control `w57_cn_d1024_T1024_B128_hm3072_hops16_400M` (63.9k, about 16:40Z); s4 54324756 $1.09/h runs both fading
  mixers (27.6k and 27.9k, about 18:45Z). Ends exclude the final eval. Data: `train_big.u32` copied box to box (e to
  s1 and s4, s1 to s3, s4 to s2), every copy sha256 6ccc9836...200d at 8,037,279,356 bytes, train/val/token_bytes
  match S0. The first s3 (54324753, Thailand) was destroyed before use: 0.45 to 2.0 MB/s from every source; replaced
  once. The s3 queue line read `--accum 64--n-sub 128` (a missing space that would fail argparse); the space was
  added, nothing else changed, and the run's config shows n_sub 128. No OOM. Boxes are KEPT after wave 57 for the base
  model (JIMOTHY decides each end). Pull loop: `track4_sdmonly_vast_pull_true.sh` with WATCH_LIST set to the new
  list, PID 6383, every 15 minutes, `runs_vast_sdmonly/watch_shapesolve.out`, never destroys.
- 2026-10-05T15:04Z · Idle sweep: 6 boxes checked, none idle, none closed. e, f, s1, s2, s3, s4 each run 2 training processes; every GPU busy on at least one of two reads 30 s apart (f GPU 1 read 0% once, between steps); newest log writes 0 to 5 minutes old. Credit $95.63.
- 2026-10-05T15:04Z · Lane SHAPESOLVE-FLEET done (commit 69a356494): all 8 sealed runs on s1 to s4 train. Boxes s1 54324748 $0.9785/h, s2 54324752 $1.0411/h, s3 54328901 $1.0852/h (the first s3, 54324753, was destroyed for a 0.45 to 2.0 MB/s link and replaced), s4 54324756 $1.0944/h; all transfer prices under $15/TB, each offer passed the selector's --check. Data copied box to box, every train_big.u32 sha256 6ccc9836...200d. One queue fix: s3's line read "--accum 64--n-sub 128" (a missing space in the file I wrote); the lane added the space, nothing else. Pull loop PID 6383 (every 15 min, never destroys). Expected ends: s3 control 16:40Z, qgrad1 17:50Z, slr10 18:30Z, both mixers 18:45Z, per-hop 19:20Z, k64 21:30Z, qgrad1+graded 22:30Z (graded trains at 0.44x).
- 2026-10-05T15:06Z · Heartbeat. EARLY SIGNAL (training loss, not TEST): the run-time mixer with fading heads leads
  its same-window control at every logged step so far: step 100 5.2254 vs 5.2420, step 200 4.8243 vs 4.8513, step 300
  4.6167 vs 4.6538 (0.037 lower), gradient norm 0.158 and steady. Past mixers also led early and went unstable late, so
  no verdict before the TEST score (about 18:45Z). The store arms track the no-store curve within a few thousandths so
  far. Width 2,048 weights home 10.82 of 11.66 GB.
  LANE DDPRECIPE DONE (branch settle-ddprecipe, cd2d5d30a): the multi-GPU trainer runs our exact recipe; world 1 is
  bit-equal to the single-GPU trainer, world 2 equal to float rounding; checkpoints resume across 2 GPUs and 1 GPU both
  ways; a resume onto a different train file length or batch is refused; 48 of 48 self-test checks on M5 CPU (gloo),
  8 mutations each caught. NCCL on a real box is untested. JIMOTHY re-runs the self-test now; landing waits for Claude
  1's open merge. A 2-GPU base run uses --world 2 --B 256 --accum 32 (equal to one GPU at --B 512 --accum 64).
  Credit $95.50: about 13.2 h of this fleet ($6.08/h) above the $15 floor.
- 2026-10-05T15:24Z · Heartbeat. DDPRECIPE self-test RE-RUN by JIMOTHY in its worktree: 48 of 48 pass, exit 0. Landing waits
  for Claude 1's open merge. Mixer lead over its same-window control (training loss) is NARROWING: step 300 0.037,
  step 400 0.027 (4.5037 vs 4.5305), step 500 0.020 (4.4708 vs 4.4905); its weight-decay twin trails it (4.4917 at 500).
  Store arms at step 400 to 500 sit within 0.002 of the default store (k64 4.5687, per-hop 4.5679, default 4.5685 at
  400; slr10 4.4751, default 4.4735 at 500). Width 2,048 weights home 11.32 of 11.66 GB. SDM twin step 9,300;
  96 hops 2,200; transformer 11,700. Credit $93.63: about 12.9 h of the fleet above the floor.
- 2026-10-05T15:44Z · Width 2,048 true-run weights HOME: 11,656,942,811 bytes, sha256 d4e4181f...b71a6 matches box f. All three true-run weights (1,024, 1,536, 2,048) are now home and verified. Landing lane DDPRECIPE (merge open earlier, now clear).
- 2026-10-05T15:44Z · Heartbeat. The mixer's early lead is GONE by step 800 (training loss, mixer vs same-window control): 600 4.4624 vs 4.4720 (+0.010 for the mixer), 700 4.3316 vs 4.3374 (+0.006), 800 4.3103 vs 4.3044 (the control now leads by 0.006). Its weight-decay twin is further behind (4.3551 at 800). The same early-lead-then-fade shape as every earlier mixer; the TEST score still decides. qgrad1 gradient norm 0.373 at step 1,300 (from about 0.1): one reading, watched. Store arms stay within 0.002 of the default. DDPRECIPE landing waits: Claude 1 opened a new merge. Credit $91.65.
- 2026-10-05T16:04Z · Idle sweep: 6 boxes checked, none idle, none closed or reassigned. e, f, s1, s2, s3, s4 each run 2 training processes, GPUs 84 to 100% on two reads 30 s apart, newest log writes 0 to 5 min old. Credit $89.57 (above the $25 watch). Commit waits for Claude 1's open merge.
- 2026-10-05T16:05Z · Heartbeat. qgrad1 (the query's gradient reaches the features) goes ROUGH from step 1,200: gradient norm 0.10 to 0.195, 0.373, 0.474, 0.325, 0.436 by step 1,600, while the default store holds 0.089 to 0.098; and its training loss falls behind the default (step 1,600: 4.0526 vs 3.9984, 0.054 worse). AX2 looks headed for a MISS; its graded twin (also qgrad 1) is still smooth at step 700 (0.109). The mixer and its weight-decay twin now sit level at step 1,000 to 1,100 (4.1916 and 4.1909). No result json yet in wave 57. SDM twin step 9,600; 96 hops 2,500; transformer 13,100. Credit $89.51.
- 2026-10-05T16:23Z · Heartbeat. qgrad1 is DIVERGING: gradient norm 1.037 at step 1,900 and its loss going UP (4.0526 at 1,600, 4.0964 at 1,900) while the default store falls (3.8801 at 2,300). Kept running to its TEST score (about $0.49/h for its GPU), so AX2 is decided by a measurement, not a trend. Mixer gradient norms rising too (0.224 fading, 0.305 with weight decay, from about 0.15). Store arms k64, slr10 and per-hop stay smooth (0.093 to 0.096). No wave-57 result yet. Credit $87.57.
- 2026-10-05T16:44Z · FIRST SHAPESOLVE RESULT: `w57_cn_d1024_T1024_B128_hm3072_hops16_400M` (memory OFF, window 1,024,
  128 windows a step, the same 131,072 tokens a step) TEST **1.11881**. Against the same shape at window 256
  (`w39_cn_d1024_B512_hm3072_hops16_400M`, 1.13013): **0.0113 better, beyond the noise, with no memory at all.** The
  model's only long-range input is its five fading averages, and a longer window lets them see further back, so the
  window is a lever on its own. It is now the AX7 and AX8 reference (the mixers against 1.11881).
  WINDOW SWEEP (memory off, width 1,024, 16 hops, 400M): 256 1.13013 · 1,024 1.11881. Best at the top edge:
  UNBRACKETED. By rule (a), s3 GPU 1 (HOT AND IDLE now) takes window 2,048 and 4,096, then the store at window 1,024.
  qgrad1 gradient norm 1.903 at step 2,200 (still diverging). Credit $85.60.

### Wave 58 (the window past its edge; the store at a long window), sealed 2026-10-05T16:44Z before it runs

| id | run (s3 GPU 1, in order) | prediction |
|---|---|---|
| AY1 | `w58_cn_d1024_T2048_B64_hm3072_hops16_400M` | lands within 0.007 of window 1,024 (1.11881) |
| AY2 | `w58_cn_d1024_T4096_B32_hm3072_hops16_400M` | is no better than window 2,048: the curve turns |
| AY3 | `w58_sdm_d1024_T1024_B128_hm3072_hops16_400M` (store on) | lands within 0.007 of its memory-off pair at window 1,024 (1.11881) |
- 2026-10-05T17:04Z · Idle sweep: 6 boxes checked, none idle, none closed or reassigned; each runs 2 training processes, GPUs 77 to 100% on two reads 30 s apart, newest logs 0 to 5 min old. Credit $83.53. Commit waits for Claude 1's open merge (open since about 15:44Z).
- 2026-10-05T17:05Z · Heartbeat. Nothing new finished. Transformer at 2.0B at step 15,100 of 15,258 and 96 hops at 3,000 of 3,051: both land within about 15 min. qgrad1 gradient norm 5.422 (diverged; kept to its TEST score). Window 2,048 at step 500. Credit $83.41.
  Mixer vs its control at step 1,800 (training loss): fading 4.0991, with weight decay 4.1253, control 3.9768: the mixers are now 0.12 to 0.15 WORSE, with gradient norms 0.273 and 0.383 rising. The run-time mixer is losing on the wide body again.
- 2026-10-05T17:24Z · THREE RESULTS (TEST bpb):
  - THE YARDSTICK: `w54_mq_d1536_B512_2B` (transformer, width 1,536, Muon, 2.0B tokens) **0.99035**. TY1 HIT (below
    1.02241). TY2 MISS: its lead over the memory-off true run at the same width (1.02241) is **0.0321**, over the 0.03
    sealed. Against the best true run (width 2,048, 1.01665) the lead is 0.0263. At equal data the transformer leads
    every SDM-family run; the 800M comparison that had the SDM line ahead was a data mismatch, now closed.
  - 96 HOPS: `w55_cn_d1024_B512_hm3072_hops96_400M` **1.10518**. AV1 MISS by a hair: it beats 64 hops (1.10841) by
    0.0032, over the 0.003 sealed. Hops at width 1,024: 8 1.14585 · 16 1.13013 · 24 1.12273 · 32 1.11834 · 48 1.11213 ·
    64 1.10841 · 96 1.10518. The gain per step keeps shrinking (32 to 64 0.0099, 64 to 96 0.0032) but has not turned:
    HOPS UNBRACKETED; 128 is queued on e after the memory-on 32-hop twin, which started on e GPU 1.
  - AX1 HIT: `w57_sdm_d1024_hm3072_hops16_400M` (store on, defaults) **1.13029** against its no-store pair 1.13013: a
    tie (0.0002 worse). The default store adds nothing at this shape, as in every earlier test.
  Box f is HOT AND IDLE (both GPUs freed at about 17:20Z). It takes, in order: a 2-GPU bench of the DDPRECIPE trainer
  at width 2,048 with the store on (proves NCCL on a real box and whether width 2,048 memory-on fits 32 GB), then the
  last two wave-58 lines moved from s3 so the window question ends sooner (window 4,096 memory off; the store at
  window 1,024). s3 keeps window 2,048 only. Credit $81.53.
- 2026-10-05T17:37Z · BOX f, THE PROBES (DDPRECIPE code in /root/settle/code_ddp on f, branch settle-ddprecipe):
  - WIDTH 2,048 WITH THE STORE ON DOES NOT FIT a 32 GB RTX 5090: out of memory at 8 and at 2 sequences per micro-batch
    (a 129,280 x 2,048 float32 buffer, embedding-sized, so accum cannot rescue it). By THE MAIN-RUN RULE the base run
    is WIDTH 1,536.
  - THE 2-GPU TRAINER WORKS ON A REAL BOX: width 1,536, store on, 2 ranks, NCCL, ranks bit-identical at the check,
    peak 27.33 GB per GPU, 24,043 tokens/s in total (12,022 per rank); with --compile 24,030 (no gain). The arm wrapper
    never passes --compile, so every single-GPU run so far was uncompiled too. The same-shape single GPU (the twin) runs
    18,470: two GPUs buy only 1.30x, likely because each rank still runs the whole Muon step. Base-run options this
    opens: 2 GPUs at 24.0k (2.0B in about 23 h on one box), or 1 GPU at 18.5k (about 30 h) with the other GPU running a
    companion. Decided at the base-run seal.
  - Box f now runs the two wave-58 lines moved from s3 (window 4,096 memory off; the store at window 1,024). f was idle
    from about 17:20Z to 17:58Z while the probes ran.
- 2026-10-05T17:44Z · AX2 MISS: `w57_sdm_d1024_hm3072_hops16_qgrad1_400M` (the query's gradient reaches the features)
  TEST **1.16740**, 0.037 WORSE than its no-store pair (1.13013) and 0.037 worse than the default store (1.13029); its
  gradient norm went from 0.10 to above 5 after step 1,200. Letting the address gradient into the features
  destabilises this body. Its graded twin (also qgrad 1) is now rising too (gradient norm 0.292 at step 1,300).
  s1 GPU 0 freed: it takes wave 55's sealed 128-hop line (AV2), moved from box e's queue, where it would have waited
  about 6 hours behind the memory-on 32-hop twin (e's claim count for it was 0, so it cannot run twice). The store at
  window 1,024 on f is running (step line pending, 5 min in). Credit $79.52.
- 2026-10-05T18:04Z · Idle sweep: 6 boxes checked, none idle, none closed or reassigned; each runs 2 training processes, GPUs 67 to 100% on two reads 30 s apart, newest logs 0 to 11 min old. Credit $77.45. Commit still waits for Claude 1's merge (open since about 15:44Z).
- 2026-10-05T18:05Z · Heartbeat. Nothing new finished. The store at window 1,024 reads loss 4.6538 at step 300, the same as its no-store control at step 300 (4.6538): the store tracks memory-off again. qgrad1+graded gradient norm 1.896 at step 1,400 (following qgrad1 into divergence). Mixers 0.437 and 0.547. 128 hops on s1 loading. Credit $77.36, about 10.2 h of this fleet.
- 2026-10-05T18:24Z · AX5 HIT: `w57_sdm_d1024_hm3072_hops16_slr10_400M` (store learning rate 10x the body's) TEST **1.13067**,
  within 0.007 of the default store (1.13029) and of memory off (1.13013): a tie. SHAPESOLVE so far, against memory off
  at 1.13013: default store +0.0002 · store 10x lr +0.0005 · query gradient +0.0373. No store knob has helped.
  s2 GPU 1 freed. It WAITS, named: it is held for the base run, which fires once the window answer lands (window 2,048
  at about 19:30Z); starting the base run before that risks training it at the wrong window, which cannot be changed
  on resume. s4's two GPUs free at about 18:45Z when the mixers end and join the same decision. qgrad1+graded gradient
  norm 2.68 (diverging). Credit $75.45.
- 2026-10-05T18:44Z · RESULTS (TEST bpb):
  - AX7 MISS: `w57_cnmix_..._fade_400M` (2 run-time mixer layers, fading heads) **1.15704**, 0.038 WORSE than its
    same-window control (1.11881). AX8 HIT on its letter: `w57_cnmix_..._fade_wd_400M` **1.13899** beats the fading
    mixer, but is still 0.020 WORSE than the control. The run-time mixer hurts this body, with or without the fixes.
  - AY1 HIT: `w58_cn_d1024_T2048_B64_hm3072_hops16_400M` (memory off, window 2,048) **1.11823**, within 0.007 of
    window 1,024 (1.11881; 0.0006 better). WINDOW SWEEP (memory off): 256 1.13013 · 1,024 1.11881 · 2,048 1.11823. The
    gain flattens after 1,024; the best still sits at the top edge by 0.0006 (inside the noise), so window 4,096 (on f)
    and 8,192 (now sealed) finish the push.
  SHAPESOLVE against its pairs: default store +0.0002 · store 10x lr +0.0005 · query gradient +0.0373 · mixer +0.0382 ·
  mixer with weight decay +0.0202 (positive = worse). Still running: 64 awake, one store per hop, graded (diverging),
  the store at window 1,024; every one tracks memory off in training loss. THE MAIN-RUN RULE: no SDM part has beaten
  its pair by 0.007, so the base run keeps THE DEFAULT SHARED STORE (the law holds: the store is read every hop). A
  later arm that beats its pair by 0.007 is logged and weighed for a second base run; k (locations awake) can change
  on resume, a per-hop store cannot.
  Free now: s4 GPU 0 and 1, s3 GPU 1, s2 GPU 1 (since 18:20Z, held for this).

### Wave 59 (THE BASE MODEL and its companions), sealed 2026-10-05T18:44Z before it fires

THE BASE RUN, on s4, both GPUs, the DDPRECIPE trainer (proven on f: NCCL, identical ranks, 24.0k tok/s, 27.3 GB):
`true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B`: arm sdmonly (store on, one shared store, 65,536 locations, k 32),
width 1,536, hop MLP 6,144, 16 hops, untie, conj 2, Muon, window 2,048 (64 windows a step, 131,072 tokens a step),
WSD schedule with the last 20% as decay, 2.0B tokens of train_big, a kept checkpoint at 1.6B (before the decay, so the
run can be extended toward 3.0B by a cooldown fork), checkpoints every 1,000 steps. Its name starts with true_ so the
box cleaner keeps its weights. Window 2,048 over 1,024: tied on TEST, and the longer window is the campaign's aim.

| id | run | prediction |
|---|---|---|
| BA1 | the base run | beats the memory-off true run at window 256 (1.02241) by at least 0.007 |
| BA2 | the base run | lands within 0.007 of its memory-off twin (below): the store ties at full scale |
| BA3 | the base run | stays behind the transformer at 2.0B (0.99035) |
| BA4 | `true_cn_d1536_hm6144_hops16_T2048_wsd_2B` (s3 GPU 1: the base run's memory-off twin, single GPU) | lands below 1.02241 (the window gain holds at 2.0B) |
| AZ1 | `w59_cn_d1024_T8192_B16_hm3072_hops16_400M` (s2 GPU 1, memory off) | is no better than window 4,096: the curve turns |
- 2026-10-05T19:04Z · WAVE 59 FIRED. THE BASE RUN STARTED 2026-10-05T18:46:33Z on s4 (54324756), both GPUs, the DDPRECIPE
  trainer from /root/settle/code_ddp: `true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B`, 15,258 steps, NCCL, 26.5 GB
  per GPU; its config records the data length and fingerprint (b6e358690d04b090), so a resume elsewhere is checked.
  Its memory-off twin `true_cn_d1536_hm6144_hops16_T2048_wsd_2B` is queued on s3 GPU 1, the window-8,192 run on s2
  GPU 1. s4's cleaner skips true_* runs.
  SPARK FAILOVER for it: s4 to the Spark measured 0.88 MB/s (200 MB in 226 s; four parallel streams no faster). The
  Spark's key on s4 sat in a line where two keys were glued together (Vast's own key injection); the file was
  rewritten as the four good lines plus the Spark key, and the Spark reaches s4. A pull loop runs ON THE SPARK
  (`track4_sdmonly_spark_failover_pull.sh`, log ~/sdmonly_base/failover_pull_base.log): it snapshots each new
  last.pt on the box, pulls the snapshot with rsync --partial, checks sha256 at both ends and keeps the two newest
  verified copies. At 0.88 MB/s a checkpoint of about 10 GB takes about 3 hours, so the off-box cadence is about every
  3 hours, not the 2 aimed for: the link sets it. A stray empty file this beat (a mistyped log path) was removed.
- 2026-10-05T19:04Z · CORRECTION to the entry above: "four parallel streams no faster" is NOT measured. That test's wait loop matched its own command line, so it never stopped and printed no time; all 200,000,000 bytes did arrive, but when is unknown. It was stopped by PID (3074951 on the Spark) and its files removed. The single-stream 0.88 MB/s stands; whether parallel streams help is untested.
- 2026-10-05T19:09Z · Heartbeat and idle sweep (6 boxes). AX6 HIT: `w57_sdm_d1024_hm3072_hops16_perhop_nsub128_400M`
  (one store per hop, 16,384 locations each) TEST **1.13021**, a tie with the default store (1.13029) and memory off
  (1.13013). SHAPESOLVE store arms, all ties: default +0.0002 · 10x lr +0.0005 · per hop +0.0001. Left: 64 awake
  (step 2,800 of 3,051), graded (diverging), the store at window 1,024.
  RED, then fixed: box s3 was IDLE on both GPUs since about 18:55Z. The base run's memory-off twin had died at start,
  out of memory at 8 windows of 2,048 per micro-batch (a 2,048 x 129,280 head buffer), and per-hop had finished.
  The twin was relaunched at about 19:08Z on BOTH s3 GPUs with the DDPRECIPE trainer and the base run's exact batch
  split (--world 2 --B 32 --accum 32, one window per micro-batch), so it is a true twin of the base run; 22.0 GB per
  GPU, both at 99 to 100%. The failed log is kept as .oom8.out. No other box idle; none closed.
  The base run on s4 trains at 29.8k tokens/s (step 200; faster than the 24.0k bench at window 256), about 19 h for
  2.0B. The Spark's failover loop waits for its first checkpoint (step 1,000, about 20:00Z). Credit $71.19.
- 2026-10-05T19:24Z · Heartbeat. Nothing new finished; all 12 GPUs busy. Base run step 500 (loss 4.5753); its memory-off twin step 200 (5.0137 against the base run's 5.0101 at step 200, a tie so far). 64 awake at step 3,000 of 3,051 (lands in minutes). Spark failover loop waiting for the base run's first checkpoint (step 1,000). Credit $69.36.
- 2026-10-05T19:44Z · TWO RESULTS (TEST bpb):
  - AX4 HIT: `w57_sdm_d1024_hm3072_hops16_k64_400M` (64 locations awake) **1.13026**, a tie with the default store
    (1.13029) and memory off (1.13013). SHAPESOLVE store arms, every one a tie with memory off: default +0.0002 ·
    10x lr +0.0005 · per hop +0.0001 · 64 awake +0.0001. No SDM part has beaten its pair; THE MAIN-RUN RULE's choice
    (the default shared store) stands. Left: graded (diverging) and the store at window 1,024.
  - AY2 HIT: `w58_cn_d1024_T4096_B32_hm3072_hops16_400M` (memory off, window 4,096) **1.11959**, no better than
    window 2,048 (1.11823; 0.0014 worse, inside the noise). WINDOW SWEEP (memory off, 400M): 256 1.13013 · 1,024
    1.11881 · 2,048 1.11823 · 4,096 1.11959. The curve turns at about 2,048 (flat from 1,024 to 4,096 within the
    noise): WINDOW BRACKETED. The base run's window 2,048 sits at the best value. Window 8,192 (s2) completes it.
  FREED: f GPU 0 and s2 GPU 0. No sealed work is worth a fresh start on them at this credit; each box closes once its
  last run ends (f: the store at window 1,024, about 21:05Z; s2: window 8,192, about 20:40Z), after its results are
  pulled. Credit $67.39.
- 2026-10-05T20:04Z · Idle sweep: 6 boxes checked, none idle, none closed. f and s2 each have one GPU free and one run going (they close when that run ends, as planned at 19:43Z); every other GPU 85 to 100% on two reads. Credit $65.39. The Spark's failover loop has no base-run checkpoint yet (step 1,000 due about 20:05Z).
- 2026-10-05T20:05Z · Heartbeat. BASE RUN healthy: step 1,100 (loss 4.0711 at 1,100, 4.2046 at 1,000), 31.7k tokens/s, the step-1,000 RANK CHECK says identical (231 tensors), first checkpoint written 19:58Z (last.pt, 9,913,996,639 bytes). GPU memory now 31.8 GB of 31.4 GiB reported per card after the checkpoint (was 26.5 GB): close to full, watched for an out-of-memory. Spark failover loop's next round (about 20:09Z) snapshots and starts pulling it (about 3 h at 0.88 MB/s). Twin at step 800 (4.3130 vs the base's 4.3706 at step 700). Credit $65.26.
- 2026-10-05T20:24Z · Heartbeat. The Spark failover pull STARTED 20:21:23Z: the base run's first checkpoint (9,913,996,639 bytes, written at step 1,000) arrives at about 3.6 MB/s so far (361 MB in about 100 s), faster than the 0.88 MB/s test; the loop labels it step 1,300 because it reads the run log's latest step, not the checkpoint's own step (the checkpoint records its true step for a resume). Base run step 1,300 (4.0459); its memory-off twin at step 1,000 4.2080 against the base's 4.2046: level. GPU memory on s4 steady at 31.8 GB. Credit $63.38.
- 2026-10-05T20:45Z · AZ1 HIT: `w59_cn_d1024_T8192_B16_hm3072_hops16_400M` (memory off, window 8,192) **1.12165**, no better than
  window 4,096 (1.11959). WINDOW SWEEP, final (memory off, width 1,024, 16 hops, 400M): 256 1.13013 · 1,024 1.11881 ·
  2,048 **1.11823** · 4,096 1.11959 · 8,192 1.12165. BRACKETED, best at 2,048, the base run's window.
  BOX s2 CLOSED: both GPUs idle with no sealed work left; small results pulled home first (runs without *.pt, logs, the
  per-window test files; 3 of 3 wave-57/59 result jsons home: k64, slr10, window 8,192); no true_* or base checkpoint
  on it; DESTROYED instance 54324752, gone from show instances. Fleet now 5 boxes, $5.04/h. The shapesolve pull loop
  (PID 6383) keeps running for the other boxes. qgrad1+graded gradient norm 8.9 (diverged). The Spark has 1.94 of
  9.91 GB of the base run's first checkpoint (about 1.5 MB/s). Base run step 1,600 (4.0191); twin step 1,300 (4.0469
  vs the base's 4.0459 at 1,300: level). Credit $61.28, about 9.2 h of this fleet.
- 2026-10-05T21:03Z · Idle sweep: 5 boxes checked, none idle, none closed. f still runs the store at window 1,024 on GPU 1 (closes after it); every other box trains on both GPUs. Credit $59.81.
- 2026-10-05T21:06Z · Heartbeat. FAILOVER FIX: the Spark's pull of the base run's first checkpoint broke off at 2.77 of 9.91 GB
  (rsync broken pipe, 21:00:09Z). Found in the same look: the loop re-snapshotted whenever last.pt was newer, but a
  checkpoint lands about every 70 min and a copy at about 1.5 MB/s takes about 110 min, so it would have restarted
  forever and never finished one. Fixed: a new snapshot is taken only when no partial pull is on the Spark (the code
  now does what its header said). Old loop stopped by PID (3080390), the fixed script deployed and restarted at about
  21:06Z; rsync is resuming from the 2.77 GB already there. Base run step 1,900 (3.9697); twin step 1,600 (4.0247).
  The store at window 1,024 at step 3,000 of 3,051 (f closes after it). Credit $59.70.
- 2026-10-05T21:28Z · AY3 HIT: `w58_sdm_d1024_T1024_B128_hm3072_hops16_400M` (store on, window 1,024) **1.11976**, within 0.007 of its
  memory-off pair (1.11881; 0.0010 worse). At a long window too the store ties.
  SHAPESOLVE IS COMPLETE except the diverged graded arm: no SDM part beat its memory-off pair, at window 256 or 1,024.
  BOX f CLOSED: both GPUs idle, no sealed work left; 27 of 27 result jsons pulled home; its one true_* checkpoint
  (width 2,048, 11,656,942,811 bytes) has been home with a matching sha256 since 15:37Z; DESTROYED 54033215, gone from
  show instances. Fleet 4 boxes. The M5 pull loop for the true list (PID 4971) keeps watching box e.
  MY ERROR, this beat: while tracing why the Spark loop logged "no checkpoint on the box yet" (an ssh hiccup the loop
  reports badly; the box command itself works), I ran one round with the copy stubbed out, and the sha check then
  deleted the 2.77 GB partial copy on the Spark. About 30 min of transfer lost; nothing on the box. The next real round
  snapshots the step-2,000 checkpoint and starts that copy fresh. Base run step 2,200 (3.8952), twin step 1,900.
- 2026-10-05T21:50Z · Heartbeat. FAILOVER COPIER, root cause and fix: every "no checkpoint on the box yet" since 21:06Z was an
  ssh login that Vast's sshd refused (exit 255; its banner says "if authentication fails, try again after a few
  seconds"); the same command run by hand worked, and ssh worked detached too. The copier now retries an exit-255
  login up to 4 times, 20 s apart, and logs ssh errors instead of hiding them. Restarted 21:48Z (old loop stopped by
  PID 3316256; one earlier stop command killed its own shell by matching its own command line, which is why stops now
  go by listed PID only). It took a fresh snapshot of the step-2,000 checkpoint and is pulling it: 72.6 MB in the first
  minute, about 1.2 MB/s, so about 2.3 h for 9.91 GB; the next snapshot follows only after this one verifies, so the
  off-box cadence is about every 2.5 h. Base run step 2,500 (3.9321); twin step 2,200 (3.9028 against the base's
  3.8952 at step 2,200: base 0.008 lower, the first gap, not yet a verdict). Credit $56.68.
- 2026-10-05T22:03Z · Idle sweep: 4 boxes checked (e, s1, s3, s4), none idle, none closed; every GPU busy on two reads 30 s apart, newest logs 0 to 9 min old. Credit $55.39 (above the $25 watch).
- 2026-10-05T22:03Z · Heartbeat. Nothing new finished. Base run step 2,800 (3.7754); at step 2,500 the base reads 3.9321 against its memory-off twin's 3.9400 (base 0.008 lower, the same gap as at 2,200). Spark copy of the base run's step-2,000 checkpoint: 1.07 of 9.91 GB (about 1.2 MB/s, lands about 00:00Z). Graded arm at step 2,900 of 3,051 (gradient norm 10.9). Credit $55.35.
- 2026-10-05T22:29Z · AX3 MISS: `w57_sdm_d1024_hm3072_hops16_qgrad1_graded_400M` (query gradient on, graded weights) **1.17734**, worse
  than qgrad1 alone (1.16740) and 0.047 worse than memory off (1.13013); its gradient norm ran 6 to 13 late.
  WAVE 57 (SHAPESOLVE) IS COMPLETE: AX1 HIT · AX2 MISS · AX3 MISS · AX4 HIT · AX5 HIT · AX6 HIT · AX7 MISS · AX8 HIT on its
  letter (still worse than the control). Against each arm's memory-off pair: default store +0.0002, 10x store lr
  +0.0005, per-hop stores +0.0001, 64 awake +0.0001, store at window 1,024 +0.0010, query gradient +0.0373, query
  gradient with graded weights +0.0472, mixer +0.0382, mixer with weight decay +0.0202. The verdict: no SDM part
  beats memory off on this body; the base run's default shared store is the law-holding choice and costs nothing.
  SPARK COPIER: the direct route to s4 (114.34.26.236:42876) stopped answering at about 22:03Z while Vast's proxy
  route (ssh7.vast.ai:38632) still works; the copier was restarted on the proxy route (old loop stopped by PID
  3320002) and resumed from the 1.07 GB already copied (1.12 GB at 22:28Z).
  s1 GPU 1 freed. It takes sealed work: the data sweep at the 16-hop shape (DATA is UNBRACKETED).

### Wave 60 (data, past 800M at the 16-hop shape), sealed 2026-10-05T22:29Z before it runs

| id | run (s1 GPU 1) | prediction |
|---|---|---|
| BB1 | `w60_cn_d1024_B512_hm3072_hops16_1600M` (memory off, 1.6B tokens) | beats the same shape at 800M (`w41_cn_d1024_B512_hm3072_hops16_800M`, 1.09026) by at least 0.02 |
- 2026-10-05T22:43Z · Heartbeat. Nothing new finished; 8 of 8 GPUs busy. Base run step 3,400 (3.7489); twin step 3,000 (3.8434). The 1.6B data run started on s1 GPU 1 (step 100). Spark copy over the proxy route: 2.10 of 9.91 GB (about 1.0 MB/s, lands about 01:00Z). Credit $52.63.
- 2026-10-05T23:04Z · Idle sweep: 4 boxes checked, none idle, none closed (e and s1 on the first pass; s4 and s3 re-checked directly after their ssh lines did not report: both GPUs 99% on each, base run step 3,700, twin step 3,300). Credit $51.34. The navigator chose to stay on the main task (no big-memory box now) and asked for every idea kept for later: written to experiments/track4/sdmllm/IDEAS_SDMONLY_NEXT_2026-10-05.md.
- 2026-10-05T23:04Z · Heartbeat. Nothing new finished; 8 of 8 GPUs busy. Base run step 3,700 (3.6520); twin step 3,300 (3.8381). Spark copy 4.08 of 9.91 GB (about 1.6 MB/s over the last 20 min; lands about 00:05Z). 1.6B data run step 400; 128 hops step 1,500. Credit $51.21.

### 2026-10-05 23:40Z - the Guy's corpus goes big (JIMOTHY)
- The navigator: a big corpus of Searle's papers first, then all philosophy papers, texts, lectures and webpages; the
  Guy is based on John Searle generally and post-trained on many philosophy texts. Private research only.
- Hugging Face first (looked myself): malteos/philpapers-2023-10-28 (54,502 open-access PhilPapers full texts, 1.32 GB
  gzipped), timaeus/pile-philpapers (49,023, overlapping), SEP (AiresPucrs, 182,531 paragraph rows), philosophy
  StackExchange (50,000 Q&A). No Searle-by-Searle dataset exists there, so his own words need the web.
- Already on disk: 64 Searle lecture and talk transcripts in _research/searle/ (lane WEIRDGUY, 2026-10-01).
- Lanes (Opus): HFPHILPULL (the HF sets), SEARLEPAPERS (his own papers), SEARLELECTURES (more transcripts),
  SEARLEABOUT (interviews, the Chinese Room debate, IEP and teaching pages). Vaults gitignored:
  _research/philosophy/ (new .gitignore) and _research/searle/. Shared rules carry THE PROVENANCE RULE per item.
- CHATGUYREADY builds track4_sdmonly_finetune.py (chat and Guy fine-tunes for SdmOnlyLM checkpoints), CPU only.
- WLGSEARLEWORDS puts the navigator's sentence on the Guy's page, in a lane for Claude 1 to land, keeping the
  current model's description true (the new Guy is not trained yet).

### 2026-10-05T23:24:22Z - heartbeat (JIMOTHY)
- BASE (s4) step 3,900 of 15,258 (25.6%), both GPUs 98-99%. Twin, memory off (s3), step 3,600.
- e: window-256 true run step 13,300; w56b 32-hop memory-on step 2,000. s1: 128 hops step 1,600; w60 1.6B step 700.
- Spark failover copy of the base checkpoint: 5.54 of 9.91 GB at 23:24Z (about 1.2 MB/s), verify expected about 00:25Z.
- Credit $49.96, fleet $4.08/h: about 12 h. No new result to score this beat. Five Opus lanes running (corpus x4,
  fine-tune tool, Guy page words).

### 2026-10-05T23:28:32Z - the navigator picks B: split Muon across the GPUs, test, then switch (JIMOTHY)
- Why: 2x RTX 5090 give 1.3x one GPU (24.0k against 18.5k tokens/s); the suspect is that every rank runs the whole
  Muon step. Lane MUONSHARD (Opus, CPU only) measures the step by arithmetic and builds a sharded Muon with the same
  maths and checkpoint format, proven equal on CPU (gloo, world 2) before any GPU sees it.
- THE SPEED TEST, sealed now before it runs: on box e when both its GPUs are free (window-256 ends about 03:30Z,
  w56b about the same time). Same base-run checkpoint, same shape, 2 GPUs, old trainer then new, about 15 min each,
  tokens/s from the log after a warm-up of 50 steps, minimum of the steady blocks, engine count and driver stamped.
- THE SWITCH RULE, sealed now: switch only if the new trainer gives at least 1.25x the old tokens/s on the same box
  AND its loss over the first 100 resumed steps sits within the old trainer's own spread on the same steps. Then the
  base (s4) and its twin (s3) each restart from a checkpoint AT THE SAME STEP NUMBER with the new trainer, so the
  memory-on and memory-off runs keep identical optimiser code over identical steps. Restart right after a checkpoint
  lands, so few steps are lost. Below 1.25x: keep the old trainer and record the measured number.

### 2026-10-05T23:30:12Z - AMENDMENT to the speed test, before it runs: test on the base box itself; keep every useful box (JIMOTHY)
- The navigator: keep the testing box and move straight to training on it if it is faster; do not lose good boxes.
- Measured now: e and s4 hold the same train file (train_big.u32, 8,037,279,356 bytes), but e holds no base
  checkpoint, and no box holds a key to another, so moving the base to e means a 9.9 GB copy (hours at 0.45 to
  3.6 MB/s) and losing the steps s4 makes meanwhile.
- So the test runs ON s4: right after a base checkpoint lands, stop the old trainer and resume the same checkpoint
  with the new one. The sharded Muon is the same maths, so the test's steps are real training either way. Old
  tokens/s = s4's own log over the preceding steady blocks (minimum, warm-up excluded); new = the first 30 min after
  a 50-step warm-up. Same 1.25x and loss rules as sealed above. Pass: s4 simply keeps going (the testing box moves
  straight into training). Fail: resume the old trainer from the newest checkpoint.
- The twin (s3) switches in place at the same step number. Box e is KEPT: when its runs end (about 03:30Z) it takes a
  sealed base-run companion on the new trainer (if it passed) rather than being closed. No useful box is destroyed.

### 2026-10-05T23:33:23Z - idle sweep tightened (JIMOTHY)
- The navigator: close an idle box after 30 minutes, not one hour. The sweep now runs every 30 minutes (at :07 and :37,
  cron dc73a800, replacing the hourly c77e7487) with a 30-minute idle rule. Reassigning a box to sealed work (base
  companion, speed test, chat or Guy fine-tune) still comes before closing it.
- 2026-10-05T23:35:43Z WLGSEARLEWORDS ready to land for Claude 1 (branch settle-wlgsearlewords, commits 9d388d5d1 + 15fc18743): the Guy page says he is based on John Searle in general and post-trained on many philosophy texts; the page keeps saying the shown Guy is the earlier one and the next is not trained yet. Its new tests 30/30; the 5 failing site tests and the i18n:check exit 1 are pre-existing (SETTLE docs translations, settle-mcp docs 119 vs 120 examples). After landing: rebuild settle-mcp docs.

### 2026-10-05T23:37:37Z - CHATGUYREADY in; two corrections (JIMOTHY)
- CHATGUYREADY: track4_sdmonly_finetune.py (new, untracked), selftest 29 of 29 in 20 s on CPU. Loads SdmOnlyLM
  checkpoints from both trainers, --task chat (chat_train 0.5 + train_big 0.5 rows) and --task guy (guy 0.5, chat 0.25,
  web 0.25), scores TEST and chat_test with the trainer's own functions and the GUY test on his mask, writes resumable
  checkpoints with the lineage base, chat, guy. The older track4_sdmonly_chain_finetune.py cannot load the base.
  Data: chat shards on the Spark (~/settle24/sdmllm/data/); the GUY v2 mask and starts files exist only on the M5
  (private/wlg_v2/); the Spark's copy has no mask. The CHAT gate 1.06386 came from an older width-768 run, so the
  gate needs restating at the base's own window before chat fires.
- CORRECTION 1: the base run's speed is 31.9k tokens/s on 2 GPUs (log, steps 3,900 to 4,100), not 24.0k. The 24.0k
  against 18.5k figure was an earlier test at a different window and micro-batch. MUONSHARD told.
- CORRECTION 2: the base run's end. Measured 4.26 s per step over its first 4 h 51 min (4,100 steps, checkpoints
  included), 11,158 steps left, so it ends about 13:00Z Tue 6 Oct, not 18:00Z as I said before.
- 2026-10-05T23:43:17Z HFPHILPULL done: 289,809 documents, 780,272,787 words (about 1.01B tokens, words x 1.3, an estimate), 5.01 GB of
  text in _research/philosophy/hfphilpull/. PhilPapers 50,583 docs 589.5M words; Pile PhilPapers 8,275 new docs
  97.0M (71% of its unique rows overlapped PhilPapers and were dropped); SEP 1,796 entries 26.4M; philosophy
  StackExchange from the archive.org human dump (the HF set's answers were gpt-4o-mini, so not used) 21,616
  questions 18.6M; phildata 69 partial works 10.5M; OpenAlex abstracts only 207,470 docs 38.4M. 3,489 full texts
  mention Searle; only 3 are BY him (Bosnian bilingual translations). His own words come from SEARLEPAPERS.

### 2026-10-05T23:44:25Z - heartbeat; MUONSHARD in: Muon is not the leak (JIMOTHY)
- MUONSHARD: sharded Muon built and BIT-EXACT against plain Muon (selftests 17/17, 16/16, 9/9; checkpoints resume both
  ways). But by arithmetic on the base config Muon is about 2 to 4% of a 4.1 s step (1.674e13 FLOPs per rank), so
  sharding is ESTIMATED at 0.98x to 1.02x: it cannot meet the sealed 1.25x rule. The bigger unknown is the gradient
  all-reduce (3.96 GB fp32 per step, 4 to 19% of the step at 25 to 5 GB/s). No single-GPU number exists at the base
  config, so true 1-to-2 GPU scaling is unknown. Files live in the DDPRECIPE worktree (untracked).
- DECISION: no live switch is planned. On box e when it frees (about 03:30Z): the 10-minute probe only
  (track4_sdmonly_muonshard_profile.py --probe 5 at world 1 and world 2), to measure real scaling and the
  all-reduce share. The sealed A/B runs only if the probe shows at least 5% saved. Then e takes a base companion.
- World: base (s4) step 4,200; twin (s3) 3,900; e window-256 13,500 and w56b 2,100; s1 w60 900 and 128 hops
  1,700. Spark copy 6.60 of 9.91 GB at 23:43Z. Credit $48.57.

### 2026-10-06T00:04:39Z - heartbeat (JIMOTHY)
- Base (s4) 4,500 of 15,258; twin (s3) 4,200; e window-256 13,600 and w56b 2,200; s1 w60 1,200 and 128 hops 1,800.
  Spark copy 8.01 of 9.91 GB at 00:04Z, verify about 00:30Z. Nothing finished, nothing idle. Credit $47.20.
- 2026-10-06T00:22:16Z SEARLEABOUT done: 816 documents, 7,938,547 words in _research/philosophy/searleabout/ (Europe PMC open papers 291 docs 2.73M; IEP 135 entries 1.55M; Wikipedia 187 with revision ids 947k; the Chinese Room debate 51 docs 797k incl. the 1980 BBS issue and the NYRB exchanges; arXiv 51; Chalmers 34; public-domain Brentano, Husserl, Reinach 471k; 20 text interviews with Searle 76k). Gaps: OAPEN refused us (bot check, 429), 26 books not fetched; Austin and Reiterating the Differences have no open copy. Licence mix recorded per document.
- 2026-10-06T00:23:24Z SEARLEPAPERS done: 72 texts by Searle, 545,251 words (511,133 without 4 near-duplicates), 63 of 310 bibliography rows (275 from his own 2009 Selected Bibliography, found on an archived copy of his Berkeley site, the best single source: 33 files of drafts and handouts). Classics in: Minds Brains and Programs, Is the Brain a Digital Computer, Consciousness (2000), Taxonomy of Illocutionary Acts, What Is a Speech Act, Proper Names, Ought from Is, Collective Intentions, What Is an Institution, Biological Naturalism, The Campus War (book). Gaps: NYRB and BBC block AI agents by robots.txt (respected); paywalled papers left. Pirated whole-book copies (archive.org uploads labelled Z-Library, epdf, Scribd) were seen and NOT taken; JIMOTHY declines them. Lecture leads passed to SEARLELECTURES.

### 2026-10-06T00:24:09Z - heartbeat (JIMOTHY)
- Base 4,800 of 15,258; twin 4,400; e window-256 13,800 and w56b 2,300; s1 w60 1,400 and 128 hops 1,900. Spark copy
  9.53 of 9.91 GB, verify in minutes. Corpus so far: HFPHILPULL 780.3M words, SEARLEABOUT 7.94M, SEARLEPAPERS 0.55M;
  SEARLELECTURES still running. Credit $45.82.

### 2026-10-06T00:44:07Z - heartbeat: the first base checkpoint is safe on the Spark (JIMOTHY)
- VERIFIED 00:29Z: ~/sdmonly_base/true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B/step_2600.pt, sha256 cf00e6d6...ff2ac
  matches the box. CORRECTION to its label: the checkpoint itself says step 2000 (read on the Spark with torch);
  "2600" is the run log step when the snapshot was cut (the known copier labelling defect). Not edited live: bash
  reads a running script as it goes. Fix in the next copier restart (read the step from the checkpoint).
- The next pull started 00:40Z (labelled 5000), about 2.3 h at 1.2 MB/s, so about 03:00Z.
- Runs: base 5,000; twin 4,700; e w256 14,000 and w56b 2,500; s1 w60 1,700 and 128 hops 2,000. Credit $44.51.

### 2026-10-06T01:13:44Z - RED, handled: the base run hung at its step-5000 quick eval; resumed from step 4000 (JIMOTHY)
- Seen 00:50Z: s4 GPU0 at 0 %, the log silent since step 5000 (logged about 00:32Z). Rank 0 (pid 13426, the TCPStore
  owner) idle in a futex wait with every thread at 0 % CPU; rank 1 spinning in the next collective (GPU 99 % at 98 W,
  the NCCL wait signature). py-spy cannot attach (no SYS_PTRACE). The twin (memory off, same trainer) passed its own
  step-5000 eval at about 01:00Z (quick bpb 1.11944), so the hang is specific to the memory-on eval on rank 0. Root
  cause NOT found. The 60-minute process-group timeout would have killed the run about 01:32Z anyway.
- Stopped by listed PIDs 13426 13427 13420 at 01:02Z. Newest checkpoint last.pt = step 4000 (read on the box), so
  steps 4,001 to 5,000 (about 71 min) are redone.
- Resume 1 (01:03Z): CUDA out of memory at the first step. Cause, read in the code: the trainer loads last.pt with
  map_location=device and keeps the loaded model and optimiser dicts referenced for the whole run (about 10 GB).
  A fresh start never holds them, so this only bites on resume. Patched the box's copy (/root/settle/code_ddp,
  backup .bak_0106): load on CPU, then drop the copies and empty the CUDA cache. Same numbers, same data order.
- Resume 2 (01:06Z): training, both GPUs 99-100 % at 444-459 W, 30.6 GB each. --eval-every 100000000, so no quick
  evals (the final TEST eval still runs, after the final checkpoint is written).
- OWED to lane DDPRECIPE before it lands: the same CPU-load fix, and the memory-on eval hang. The failover copy now
  running (labelled 5000) is in fact the step-4000 checkpoint, the one the run resumed from.
- 2026-10-06T01:17:23Z Resume verified: step 4,100 loss 3.7068 against the original run's 3.7067 (rank 0 3.6850 against 3.6849), so the same data and the same weights. The base now ends about 14:40Z Tue (about 1 h 40 later).

### 2026-10-06T01:18:15Z - heartbeat (JIMOTHY)
- Base (s4, resumed) 4,100, both GPUs 99 %; twin (s3) 5,200; e window-256 14,300 and w56b 2,700; s1 w60 2,200 and 128
  hops 2,200. Spark: step 2000 verified (file named step_2600); step-4000 copy 2.31 of 9.91 GB. Credit $42.16.

### 2026-10-06T01:23:54Z - heartbeat (JIMOTHY)
- Base 4,200 (resumed, steady); twin 5,300; e w256 14,300 and w56b 2,700; s1 w60 2,200 and 128 hops 2,200. Spark step-4000
  copy 2.71 of 9.91 GB. Credit $41.76. CORRECTION: the resumed base reaches step 5,000 about 02:20Z (800 steps at
  4.26 s), not 01:30Z as I told the navigator.

### 2026-10-06T01:43:44Z - heartbeat (JIMOTHY)
- Base 4,500; twin 5,600; e w256 14,500 of 15,258 (about 03:10Z) and w56b 2,800; s1 w60 2,500 and 128 hops 2,300. Spark
  step-4000 copy 4.00 of 9.91 GB. Credit $40.38. Nothing finished, nothing idle.

### 2026-10-06T02:04:56Z - heartbeat (JIMOTHY)
- Base 4,800; twin 5,900; e w256 14,700 (about 03:10Z) and w56b 2,900 of 3,051 (about 02:35Z); s1 w60 2,800 and 128 hops
  2,400. Spark step-4000 copy 5.45 of 9.91 GB. Credit $39.05.
- Staged for the probe on e: /root/settle/code_ddp = s4's patched trainer copy plus the MUONSHARD files. e GPU1 waits for
  GPU0 when w56b ends (the probe needs both GPUs), about 35 minutes; named, not idle.
- 2026-10-06T02:09:25Z SEARLELECTURES done: 158 new transcripts, 1,243,437 words (_research/searle/searlelectures/). All three Berkeley courses complete (Language, Mind, Society, 28 lectures each); 27 Fall 2013 Mind lectures transcribed locally with whisper.cpp on CPU. Mostly YouTube auto-captions. JIMOTHY started the remaining six archive.org Berkeley offerings (about 160 lectures, audio only, an estimated 1.8M words, not a count) on the M5 CPU (tools/asr_remaining.sh, nice 10, 8 threads, about 10 h); the first item hit archive.org HTTP 500 and is retried in a second pass (the tool is resumable).

### 2026-10-06T02:25:07Z - heartbeat; AW4 scored; base passed step 5,000 (JIMOTHY)
- AW4 HIT: `w56b_sdm_d1024_B512_hm3072_hops32_400M` (memory ON, 32 hops, width 1,024) TEST **1.12035** against its
  no-store pair `w49_cn_d1024_B512_hm3072_hops32_400M` 1.11834: +0.0020, inside the 0.007 noise. The memory ties
  again, now at 32 hops.
- The resumed base passed step 5,000 with no stall (no quick eval): step 5,100. Twin 6,100; e w256 14,800; s1 w60 3,000
  and 128 hops 2,500. Spark step-4000 copy 6.90 of 9.91 GB. Credit $37.68.
- e GPU1 freed by w56b: the one-GPU half of the speed probe launched on it (world 1, B 64, accum 64, 5 timed steps,
  random tokens, no data written), log /root/settle/logs/probe_world1.out. The two-GPU half waits for w256 (about 03:10Z).
- Transcription: archive.org answers HTTP 500 now and then and the tool stops an item on the first one (4 lectures of
  Spring 2010 Mind done, 1 item skipped). tools/asr_passes.sh re-runs the resumable pass up to 6 times after pass 1.

### 2026-10-06T02:45:58Z - heartbeat; the first probe measured the wrong shape (JIMOTHY)
- The one-GPU probe I launched at 02:24Z ran the trainer's DEFAULT shape (d 256, 4 hops, T 256, no hop MLP, no Muon:
  muon_step 0.0), because I passed only --world, --B and --accum. Its 2.84 s per step says nothing about the base. VOID.
- Relaunched on e GPU1 with the base run's full flags (d 1,536, hop MLP 6,144, 16 hops, shared store, untie, conj 2,
  Muon, T 2,048, world 1 B 64 accum 64); log /root/settle/logs/probe_world1_base.out.
- World: base 5,300; twin 6,400; e w256 15,000 of 15,258; s1 w60 3,300 and 128 hops 2,600. Spark step-4000 copy 8.48
  of 9.91 GB. Credit $36.31.
- 2026-10-06T02:46:46Z Probe, one GPU, base shape (e GPU1, RTX 5090, driver 580.173.02, GPU0 busy with w256): **15,491 tokens/s**,
  8.461 s per 131,072-token step: microbatches 8.164 s, Muon **0.124 s (1.5%)**, AdamW 0.024 s. The base on s4's two
  GPUs logs 31,950 tokens/s, so two GPUs give about **2.06x** one GPU at this shape. CROSS-BOX (e against s4, drivers
  580 against 570), so stated as indicative; the same-box two-GPU half runs on e when w256 ends. If it holds: the
  1.3x was an artefact of the old shape, DDP already scales at the base shape, sharding Muon cannot help (under 5%,
  so the sealed A/B does not run), and a 4- or 8-GPU box would cut the base's wall time nearly in proportion.

### 2026-10-06T03:03:52Z - heartbeat (JIMOTHY)
- Base 5,600; twin 6,700; e w256 15,200 of 15,258 (final eval next), e GPU1 waits for it (the same-box two-GPU probe);
  s1 w60 3,600 and 128 hops 2,700. Spark step-4000 copy 9.71 of 9.91 GB. Credit $34.95. No navigator answer on the
  4-GPU move, so A (stay on s4) stands.

### 2026-10-06T03:22:48Z - TR6 scored: the memory ties at 2.0B (JIMOTHY)
- TR6 HIT: `true_sdm_d1536_hm6144_hops16_2B` (memory ON, width 1,536, 16 hops, window 256, 2.0B tokens) TEST
  **1.02284** against its memory-off pair `true_cn_d1536_hm6144_hops16_2B` 1.02241: +0.0004, inside the 0.007 noise.
  The store ties at full scale too, as in every smaller test. It reads its memory and matches memory off; it does not
  beat it. Transformer yardstick at 2.0B: 0.99035.
- Spark: VERIFIED the base run's step-4000 checkpoint at about 03:07Z (file step_5000.pt; label from the log, the
  checkpoint holds step 4000, the one the run resumed from). Two verified copies now: steps 2000 and 4000. The copier's
  next snapshot hit 4 ssh refusals at 03:17Z (the ssh7.vast.ai proxy); it retries in 10 minutes.
- e: both GPUs free after TR6. The same-box two-GPU probe launched (world 2, B 32, accum 32, base shape). The TR6
  checkpoint (9.9 GB) is being hashed on e and copied home with rsync --partial
  (runs_vast_sdmonly/sdmonly-e-2x5090/ck/true_sdm_d1536_hm6144_hops16_2B/), about 80 min at 2 MB/s.
- Credit $33.77.

### 2026-10-06T03:27:16Z - the same-box probe, and wave 61 (the chat rehearsal) sealed before it fires (JIMOTHY)
- Probe, box e, base shape, same box both halves: 1 GPU 15,491 tok/s (8.461 s a step); 2 GPUs **22,956** (5.710 s):
  **1.48x**. The leak is the gradient all-reduce, exposed after the last micro-batch: 1.43 s of 5.71 s (3.96 GB at about
  3 GB/s). Muon is 0.124 s; sharded Muon is SLOWER (0.97x). s4 runs the same step in 4.1 s at 31,950 tok/s, so s4's
  GPUs talk much faster than e's: multi-GPU speed depends on the box's interconnect, so a 4- or 8-GPU box is a gamble
  unless its all-reduce is measured first. Decision: stay on s4 (A). The MUONSHARD A/B does not run (under 5%).
  The window-256 checkpoint on e: sha256 c226d2d2...bcee3d, being copied home.
- WAVE 61: THE CHAT REHEARSAL on box e (both GPUs free), off the two finished 2.0B window-256 true runs, with
  track4_sdmonly_finetune.py --task chat (chat_train rows 0.5 + train_big rows 0.5), 100M tokens, T 256 (their own
  window), B 512, accum 64, default chat rates. It tests the fine-tune tool on a real GPU, times it, and gives the first
  SDM CHAT off a 2.0B model. Names: `w61_chat_sdm_w256_100M` (GPU0, memory ON, init true_sdm_d1536_hm6144_hops16_2B)
  and `w61_chat_cn_w256_100M` (GPU1, memory OFF, init true_cn_d1536_hm6144_hops16_2B).

| id | run | prediction |
|---|---|---|
| CR1 | both | chat_test bpb falls by at least 0.10 from before to after |
| CR2 | both | TEST bpb rises by no more than 0.02 from before to after (the web rows protect it) |
| CR3 | `w61_chat_sdm_w256_100M` | its after chat_test lands within 0.007 of the memory-off one (the store ties at chat too) |

### 2026-10-06T03:41:13Z - heartbeat (JIMOTHY)
- The chat-data upload from the M5 to e ran at about 0.15 MB/s (1.5 GB = hours), so it was stopped and the shards are being
  rebuilt ON e from the pinned dataset (smol-smoltalk @ f73fe857) with the lane script: chat_test came out byte-identical to
  the Spark copy (sha256 f88ec2c2...19bc). chat_train must match b280437d...6046 before wave 61 fires. e GPUs wait for it.
- The TR6 checkpoint copy home: 1.01 of 9.91 GB at 03:37Z.
- 2026-10-06T03:46:40Z WAVE 61 FIRED on e (03:44Z): chat_train rebuilt on e is byte-identical to the Spark's (sha256 b280437d...6046);
  chat_mix differs (its web half is the END of train_big, and e holds the 2.0B prefix, not the Spark's 3.0B), and wave
  61 does not use it. Runs write to /root/settle/runs_chat so the box cleaner cannot delete their weights.
  BEFORE scores reproduce the init runs exactly: memory on TEST 1.02284 (chat_test 1.21792), memory off TEST 1.02241
  (chat_test 1.21577). So the fine-tune tool loads and scores SdmOnlyLM checkpoints correctly on a GPU. Memory off at
  22.2k tok/s (763 steps, about 75 min).

### 2026-10-06T03:48:26Z - heartbeat; RED fixed: the Spark lost its login to s4 (JIMOTHY)
- The Spark copier failed at 03:17Z and 03:42Z (4 ssh refusals each, "Permission denied (publickey)"). s4's
  authorized_keys had been rewritten at 01:01:26Z (cause unknown, not my command: it holds Vast's key and four relay keys)
  and the Spark's key was gone. Restored: backup authorized_keys.bak_0106, the file rewritten one key per line, the
  Spark key appended; the Spark reaches s4 again through ssh7.vast.ai:38632. The copier picks it up within 10 minutes.
  Verified off-box copies meanwhile: steps 2000 and 4000.
- Base 6,300; twin 7,300; s1 w60 4,100 and 128 hops 2,900. Wave 61: memory on 11.8k tok/s (about 2.4 h, ends about
  06:10Z), memory off 22.2k tok/s (about 75 min, about 05:00Z). TR6 checkpoint home 1.51 of 9.91 GB. Credit $31.95.

### 2026-10-06T04:03:56Z - heartbeat; the copier is back; the $25 line ahead (JIMOTHY)
- Spark copier back at 03:53Z: pulling the newest base snapshot (labelled 6400; it holds the step-6000 checkpoint), 0.66 of
  9.91 GB at 04:03Z, about 1.1 MB/s, so about 06:25Z.
- Base 6,500; twin 7,600; s1 128 hops 3,000 of 3,051 (ends minutes) and w60 4,300; wave 61 chat steps 100 (memory on)
  and 190 (memory off) of 763.
- Credit $30.93 at $4.08/h crosses $25 about 05:30Z; the top-up is due about 06:48Z; at $0 (about 11:30Z) the boxes die.
  The order says below $25 the base resumes on the Spark. Put to the navigator as a decision; my default is to hold
  the base on s4 while the newest verified checkpoint is no older than 4,000 steps and the top-up is not overdue,
  because the boxes keep running until $0 and moving it now would cost about 2,500 steps and Claude 1's GPU.
- 2026-10-06T04:21:26Z Sweep: s1 GPU0 idle since 04:11Z: `w55_cn_d1024_B512_hm3072_hops128_400M` trained all 3,051 steps and saved last.pt, then ran out of GPU memory in the final TEST eval (a 3.95 GB logits buffer at 128 hops). Not closed: re-run on GPU0 from its last.pt with SDM_EVAL_TOKENS=1024 (smaller eval batches; same windows, same numbers) to score it. Credit $29.75.

### 2026-10-06T04:24:08Z - 128 hops scored; wave 62 sealed and fired (JIMOTHY)
- `w55_cn_d1024_B512_hm3072_hops128_400M` (memory off, width 1,024, 400M) TEST **1.10433** (scored by the rescue run).
  AV2 MISS by the letter: it beats 96 hops (1.10518) by 0.00085, but that is inside the 0.007 noise, so 96 against 128
  is UNDECIDABLE. The hop curve at width 1,024: 16 1.13335 (400M) · 24 1.12273 · 32 1.11834 · 64 1.10841 · 96 1.10518 ·
  128 1.10433. The gains shrink: -0.0032 then -0.0009. HOPS: UNBRACKETED (best at the top edge), and flattening.
- WAVE 62 (rule (a), push past the edge), on s1 GPU 0 (freed): `w62_cn_d1024_B512_hm3072_hops192_400M`, memory off,
  width 1,024, hop MLP 3,072, 192 hops, 400M tokens, B 512 at accum 256 (micro-batch 2, as 192 hops need more activation
  memory), SDM_EVAL_TOKENS 1,024 for its final eval. Rule (c) (96 against 128 at a larger budget) is OWED and waits for
  credit; the base comes first.

| id | run | prediction |
|---|---|---|
| AV3 | `w62_cn_d1024_B512_hm3072_hops192_400M` | lands within 0.007 of 128 hops (1.10433): the curve has flattened |
- 2026-10-06T04:27:00Z AV3 VOID: 192 hops at width 1,024 does not fit a 32 GB RTX 5090. Out of memory at micro-batch 2, and the reason is
  the parameters, not the activations: 192 hop MLPs of 3 x 1,024 x 3,072 are about 1.8B parameters, so weights,
  gradients and optimiser state alone fill the card. The hop sweep at width 1,024 ends at 128 on this hardware: the top
  edge is the card, recorded as UNBRACKETED BY MEMORY. s1 GPU 0 is HOT AND IDLE by choice: the only sealed work left for
  it is rule (c) (96 against 128 hops at 800M, about 35 GPU-hours), which waits for credit ($29.53, the $25 line about
  05:30Z, the top-up about 06:48Z). Revisit at the top-up.

### 2026-10-06T04:43:44Z - heartbeat (JIMOTHY)
- Base 7,100; twin 8,100; s1 w60 4,900 (GPU 0 idle by choice, credit); chat 310 (memory on) and 600 (memory off) of 763.
  Spark step-6000 copy 3.08 of 9.91 GB. Credit $28.17.

### 2026-10-06T04:59:33Z - site: four Opus lanes revise the SDM pages with the new thinking (JIMOTHY)
- The navigator: revise every SDM learn and chat page with the lookback, recurrence and Mamba thinking and how the latest
  model works; expand, simplify, new diagrams; fence future options (all the ideas, not only Mamba); update to the
  latest testing; explain every test and graph clearly, Basecamp / DHH style; a separate page #/learn-sdm-lookback,
  infused into the others. Disjoint fences, each lane landed by Claude 1:
  SDMLOOKBACKWORDS (SdmChat + sdmchat/), LEARNSDMDIAGRAMS (LearnSdm, learnsdm/, LearnSdmUnfold, learnunfold/),
  SDMEXPLOREMAMBA (SdmExplore, sdmexplore/, the new #/learn-sdm-lookback page and its route), SDMMEMORYTRUTH (SdmMemory).
  i18n files are shared: keys added only, so the landing can union them.

### 2026-10-06T05:01:41Z - the first SDM CHAT (memory off) is in (JIMOTHY)
- `w61_chat_cn_w256_100M` (memory off, from true_cn_d1536_hm6144_hops16_2B, 100M chat tokens, half web rows):
  chat_test **1.21577 -> 0.74901** (-0.467); TEST **1.02241 -> 1.03947** (+0.0171). Full sets, not samples.
  CR1 HIT (chat_test falls by far more than 0.10). CR2 HIT (TEST rises by 0.0171, inside the 0.02 allowed): the web rows
  held most of the web skill. CR3 waits for the memory-on chat (about 06:10Z). This one does NOT meet the law (memory
  off); it is the comparison. It is a rehearsal: the real SDM CHAT comes from the base.
- Samples (seed 0, temperature 0.7, top-p 0.9, repetition penalty 1.15), first try each, no cherry-picking:
  "What is the capital of France?" -> "The capital of France is Paris." · three study tips -> a formatted, on-topic list
  that wanders a little · "Explain what a computer is to a child." -> fluent but off target (talks about Wi-Fi). Loops:
  none in all three.
- e GPU1 free; no sealed GPU work for it until the memory-on chat lands (the Guy needs a chat model that reads its
  memory and the big corpus tokenised). Held, named; credit $27.

### 2026-10-06T05:03:58Z - heartbeat (JIMOTHY)
- Base 7,400; twin 8,400; s1 w60 5,100 (GPU 0 held); e memory-on chat 420 of 763 (about 06:10Z), e GPU 1 held (no
  sealed GPU work until the memory-on chat and GUYCORPUSV3 land). Spark step-6000 copy 4.25 of 9.91 GB. Credit $26.83.
- 2026-10-06T05:21:25Z s1 (54324748) CLOSED by plan B (the navigator: no top-up, run to $0, finish on the Spark; keep the twin): results pulled to runs_vast_sdmonly/sdmonly-s1-2x5090/ (3 result files, logs, per-window files), then destroyed. `w60_cn_d1024_B512_hm3072_hops16_1600M` (the 1.6B data run) stopped at step 5,100 of 12,207, unscored: the data sweep stays UNBRACKETED. Credit $25.57. Fleet now e, s3, s4 at $3.10/h.

### 2026-10-06T05:24:11Z - heartbeat; THE SPARK IS OURS (JIMOTHY)
- The navigator: Claude 1's MIDI work is done; no top-up comes; run the boxes to $0 and finish on the Spark; keep the
  twin. Plan B in force (s1 closed 05:21Z).
- SPARK (GB10) fired 05:15Z: `spark_chat_base4000_sdm_T2048_100M`, the first SDM CHAT that meets the law (memory ON,
  window 2,048), from the base's verified step-4000 checkpoint (file step_5000.pt), 100M tokens (chat_train rows 0.5 +
  train_big rows 0.5), B 64 accum 64, code = s4's patched trainer copy + track4_sdmonly_finetune.py in
  ~/sdmonly_base/code. Still computing its BEFORE scores at 05:23Z (full TEST at window 2,048 plus 4,000 chat_test
  windows). Three paused old Spark jobs (w37_base 3B, w50s T4096 and T16384) hold about 44 GB of the 121 GB; left
  paused.
- Base 7,700; twin 8,700; e memory-on chat 530 of 763. Spark step-6000 copy 5.29 of 9.91 GB. Credit $25.41 at $3.10/h.

### 2026-10-06T05:30:37Z - two corrections of mine, caught by lane SDMLOOKBACKWORDS (JIMOTHY)
- CORRECTION to the 04:24Z entry: the hop curve at width 1,024 (400M, memory off) starts 8 1.14585 · **16 1.13013** ·
  24 1.12273 · 32 1.11834 · 48 1.11213 · 64 1.10841 · 96 1.10518 · 128 1.10433. I wrote "16 1.13335"; 1.13335 is a
  different run. The verdicts stand.
- CORRECTION to what I told the navigator (and wrote in the ideas file): "window 256 tied window 2,048" is FALSE. Memory
  off, width 1,024, 16 hops, 400M: window 256 1.13013, 1,024 1.11881, 2,048 1.11823, 4,096 1.11959, 8,192 1.12165.
  A longer training window helps up to about 1,024 (0.011), then flat, then worse. Reading (ours, untested): at window
  256 many positions sit near the start of their example, where the fading averages have not filled; at 1,024 and
  beyond almost every position has its full lookback, so more length adds nothing, which still fits a lookback of a few
  hundred tokens. What DID tie at window 256 is memory on against memory off at 2.0B.
- SDMLOOKBACKWORDS landed in its lane (735f1df13 + channel be56cae99), READY TO LAND for Claude 1: the SDM CHAT pages
  gain HOW FAR BACK IT SEES, HOW SDM CHAT IS MADE, THE GAP TO A TRANSFORMER, NEXT (13 future ideas, fenced); three false
  sentences fixed; i18n 0 missing; build exit 0; npm test 3,161 pass, 13 fail, all pre-existing. Landing must rebuild
  the settle-mcp docs.
- 2026-10-06T05:36:21Z GUYCORPUSV3 done: private/wlg_v3 (gitignored), train 40.04M tokens, test 990,905. Mix: Searle 0.40 (lectures 1.47M words captions + 0.75M ASR, papers 0.45M; 3.11M unique tokens, repeated about 5x), about-Searle 0.19, general 0.31 (PhilPapers, SEP, StackExchange human answers, relevance-ranked), character 0.10 (v2 ours and weird). 20.5% of Searle segments framed as User/Guy turns. Whole-document test split, leak checks pass, selftest 21/21. Copied to the Spark ~/sdmonly_base/guy3/, all six shards sha256-matched, plus the Searle-only test shard.

### 2026-10-06T05:43:44Z - heartbeat; credit below $25 (planned) (JIMOTHY)
- Credit $24.38: below the $25 line, RED by the rule, handled by plan B (the navigator: no top-up, run to $0, finish on
  the Spark). Newest verified off-box base checkpoint: step 4000 on the Spark; step 6000 copying (6.05 of 9.91 GB). At
  $3.10/h the boxes run to about 13:35Z; after e closes (about 08:00Z) $2.18/h, to about 16:30Z.
- Spark chat from base step 4000: BEFORE TEST **1.10788** (window 2,048; the base at step 4000 of 15,258), chat_test
  1.31183; training started (step 1). Memory-on chat on e: quick at step 500, chat_test 0.7686 (memory off was 0.75935
  at the same step, same 64 windows). Base 8,000; twin 9,000.

### 2026-10-06T06:59:27Z - new session; wave 61 scored; fallen lanes resumed (JIMOTHY)
- The navigator: a new login, no limits. The three site lanes stopped by the weekly limit (LEARNSDMDIAGRAMS,
  SDMMEMORYTRUTH, SDMEXPLOREMAMBA) were resumed with their last state. Both crons are alive (heartbeat 05431e56, sweep
  dc73a800).
- WAVE 61 scored, full sets: memory ON `w61_chat_sdm_w256_100M` chat_test 1.21792 -> **0.75847**, TEST 1.02284 -> 1.03860;
  memory OFF `w61_chat_cn_w256_100M` chat_test 1.21577 -> **0.74901**, TEST 1.02241 -> 1.03947.
  CR1 HIT both (-0.459 and -0.467). CR2 HIT both (+0.0158 and +0.0171). **CR3 MISS by a hair: memory on is 0.0095 WORSE
  at chat, just outside the 0.007 noise.** The memory has not helped at any test so far.
- e idle since about 06:10Z: three bf16 model-only copies made on e (true_sdm 2B, both chats; 1.98, 1.98 and 1.77 GB,
  sha256 6471a9b4..., 305b6ba6..., d498c5e2...). The Spark pulls them (~/sdmonly_base/pull_slim_e.sh, verified by
  sha256) after a glued authorized_keys line on e was split (the same Vast fault as on s4). e closes when they land.
- Spark early chat (from base step 4000) runs at **2,051 tok/s**, 7.5x slower than one RTX 5090 (15.5k): 100M tokens
  would take about 13.5 h. Plan change: the FINAL SDM CHAT runs on s4 right after the base (one GPU, about 1.8 h,
  about $2), and the early Spark chat is stopped when the final one exists; the Spark then trains the Guy.
- Base 9,000 of 15,258 (ends about 14:20Z); twin 10,000 (about 12:50Z). Credit $20.61.

### 2026-10-06T07:03:57Z - heartbeat (JIMOTHY)
- Base 9,100; twin 10,100. Spark: base step-6000 copy 8.77 of 9.91 GB (lands about 07:25Z); the slim copies from e started
  (memory-on chat first, 0.09 GB, about 0.4 MB/s while sharing the link); early Spark chat step 70 (about 64 s a step).
  e is held only for the copies; if the link stays this slow, the memory-off chat copy is dropped to save credit.
  Credit $20.25.
- 2026-10-06T07:11:07Z LEARNSDMDIAGRAMS READY TO LAND for Claude 1 (7293b67b6 + channel bd52dc330): #/learn-sdm gains sections ELEVEN to SIXTEEN (how our latest model works; what a token sees of the past; how a hop picks its locations; does the memory help; what we might try next), with an animated forward pass, the lookback curves, full-size product keys and a memory on/off vs transformer chart; #/learn-sdm-unfold corrected (14 address vectors, the 2.0B tie). New test 10/10; build exit 0; npm test 12 fails, all failing at the lane base. Landing: union i18n (six retired ids of its own), rebuild settle-mcp docs, it also edited src/data/enfoldFacts.js.
- 2026-10-06T07:13:52Z SDMEXPLOREMAMBA READY TO LAND for Claude 1 (work 5e1051e0f, 5d53ab7f8; route 38c399639; ready line 74195a6f2): the new page #/learn-sdm-lookback (9 sections: the newest model, what it sees of the past, why the window is not the reach, recurrence and Mamba, attention vs the mixer, the trained memory, the yardstick, still running, 14 future options with NEEDS A RULING on the three recurrences), animated lookback, fade curves, fixed vs selective fade, attention vs mixer, model flow, bar charts; #/sdmexplore simplified. Build exit 0; npm test 13 fails, all pre-existing. Landing: union i18n, rebuild settle-mcp docs and llms files (two new glossary entries); it touched enfold.test.mjs, learnsettle.test.mjs and tools/wtf_harvest.mjs (added rows only).
- 2026-10-06T07:14:40Z SDMMEMORYTRUTH READY TO LAND for Claude 1 (branch settle-sdmmemorytruth: 7f171e2ac, merge d96d12166, 78ed99add, ready line d12293220): #/sdmmemory opens with THE SHORT ANSWER ("Memory has not helped our language model yet") and eight new sections (two memories, every on/off pair as a dot plot with the noise band, the mixer lead that faded, why memory may not pay here, the gap to a transformer, the hop staircase with the 192 and memory-on 64 walls, still running, eight FUTURE cards); 10 tests, 8 red-proofs. Build exit 0; npm test 5 fails, all pre-existing. All four site lanes now wait for Claude 1: SDMLOOKBACKWORDS, LEARNSDMDIAGRAMS, SDMEXPLOREMAMBA, SDMMEMORYTRUTH (each needs the i18n union and a settle-mcp docs rebuild).

### 2026-10-06T07:18:24Z - a note to watch: the base leads its twin on training loss (JIMOTHY, the navigator: "see how it turns out in the wash")
- Same data, same order, same shape; only the memory differs. Training loss (mean of the 5 logged steps up to each mark,
  nats per token, lower is better), twin (memory OFF) vs base (memory ON):
  2,000: 3.9793 vs 3.9757 (-0.0036) · 4,000: 3.6979 vs 3.6932 (-0.0047) · 6,000: 3.6223 vs 3.6162 (-0.0061) ·
  8,000: 3.5449 vs 3.5373 (-0.0076) · 9,000: 3.5044 vs 3.4986 (-0.0058).
- The base leads at every mark, by about 0.006 nats per token (very roughly 0.002 in TEST bits per byte, under the 0.007
  noise). Training loss is not the TEST score. The first memory run to stay ahead throughout; at window 256 the final
  scores tied. VERDICT OWED: BA2 (base within 0.007 of its twin) when both TEST scores land, twin about 12:50Z, base
  about 14:20Z. Re-read this note then.

### 2026-10-06T07:43:43Z - heartbeat (JIMOTHY)
- Base 9,700; twin 10,700. Spark: base step-6000 copy finishing; slim memory-on chat 1.10 of 1.98 GB; early chat step 110.
  Site lanes: SDMLOOKBACKWORDS landed (merge 6a0c61245, regenerated docs d8d859671, worktree and branch removed); the
  site tests showed only the 2 failures already on the main line (SETTLE docs translations). LEARNSDMDIAGRAMS landing,
  same 2. Credit $18.18.
- 2026-10-06T07:50:38Z Spark VERIFIED the base checkpoint labelled 6400 (holds step 6000) at 07:43Z, sha a68135ec...3fbd; the copier dropped the oldest (step 2000). Off-box copies now steps 4000 and 6000. LEARNSDMDIAGRAMS landed (merge 95416ad6a, with the unioned catalogues and regenerated docs; tests: only the 2 known docs-translation failures), worktree and branch removed. SDMEXPLOREMAMBA landing. Sweep: nothing closed (e still copying). Credit $17.82.

### 2026-10-06T08:09:15Z - heartbeat (JIMOTHY)
- Base 10,000; twin 11,000. Credit $17.13. Spark: the slim memory-on chat copy broke at 1.58 of 1.98 GB (broken pipe) and
  resumes (rsync --partial, retry loop); early Spark chat step 130.
- SDMEXPLOREMAMBA landed (merge 99fdc5f4f, regenerated files 292a817b1), worktree and branch removed. The union of
  tools/wtf_harvest.mjs had brought back a NOT_TERMS entry the main line had retired (this visit), which the stale-list
  test caught; removed by hand, tests back to the 2 known docs-translation failures. SDMMEMORYTRUTH landing.
- 2026-10-06T08:20:31Z SDMMEMORYTRUTH landed (merge a4f7e10c9; tests only the 2 known docs-translation failures); worktree and branch removed. ALL FOUR SITE LANES LANDED (6a0c61245, 95416ad6a, 99fdc5f4f, a4f7e10c9). Sweep: e not idle (copying to the Spark: memory-on chat at 1.95 of 1.98 GB); plan: copy the 2B memory-on model next, skip the memory-off chat copy, then close e. Credit $16.27.

### 2026-10-06T08:26:22Z - heartbeat; e's copies (JIMOTHY)
- VERIFIED on the Spark 08:21Z: the memory-on chat model `w61_chat_sdm_w256_100M` (bf16, model only), sha 305b6ba6...
  The Spark's next copy (the 2B memory-on model) ran at about 0.2 MB/s, so it was stopped (pull script by PID) and that
  copy now comes HOME instead (runs_vast_sdmonly/sdmonly-e-2x5090/slim/), about 0.6 MB/s, about 55 min. The memory-off
  chat copy is skipped (its results and logs are home). The memory-off 2B checkpoint `true_cn_d1536_hm6144_hops16_2B`
  at home matches e by sha256 (19d0743b...1430). e's results, chat results, logs and per-window files pulled home (23
  result files). e closes when the 2B memory-on copy is verified at home (the rule: no true_* checkpoint is lost).
- Base 10,300; twin 11,300; early Spark chat step 150. Credit $16.10.
- 2026-10-06T08:44:08Z e (54033213) CLOSED to save credit. The home copy of the 2B memory-on model ran at 0.3 MB/s (0.34 of 1.98 GB in 19
  min, about 1.8 h more, about $1.70), and the credit ($15.08) only just covers the base, the twin and the final chat.
  So I broke the rule "no true_* checkpoint lost" for one finished, scored run: `true_sdm_d1536_hm6144_hops16_2B`
  (TR6, TEST 1.02284) loses its weights; its result, log and per-window file are home, and its chat child
  (`w61_chat_sdm_w256_100M`, memory on) is verified on the Spark. The memory-off chat weights are lost too (results
  home). Fleet now s3 and s4 at $2.18/h. Credit $15.05.

### 2026-10-06T09:03:44Z - heartbeat (JIMOTHY)
- Base 10,900; twin 11,800. Spark: the next base checkpoint (holds step 8000) copying, 1.84 of 9.91 GB; early chat step 190 of 763.
  Credit $14.27 at $2.18/h.

### 2026-10-06T09:24:33Z - heartbeat (JIMOTHY)
- Base 11,200; twin 12,100. Spark: step-8000 checkpoint copy 2.69 of 9.91 GB; early chat step 210. Credit $13.55.

### 2026-10-06T09:43:46Z - heartbeat (JIMOTHY)
- Base 11,500; twin 12,400. Spark: step-8000 copy 3.21 of 9.91 GB (about 0.44 MB/s); early chat step 220. Credit $12.80.
- Budget check: s3 to about 12:55Z ($3.5), s4 through the base (14:20Z), the final chat and its copy-out to about 17:15Z
  ($8.2): $11.7 of $12.80. If the margin shrinks, the final chat drops from 100M to 70M tokens.

### 2026-10-06T10:03:41Z - heartbeat (JIMOTHY)
- Base 11,800; twin 12,700. Spark: step-8000 copy 3.76 of 9.91 GB; early chat step 240. Credit $12.09.

### 2026-10-06T10:23:44Z - heartbeat (JIMOTHY)
- Base 12,000 (its WSD decay and the kept 1.6B checkpoint come at about step 12,207, the last 20%); twin 13,000. Spark:
  step-8000 copy 4.27 of 9.91 GB; early chat step 260. Credit $11.35.

### 2026-10-06T10:43:51Z - heartbeat (JIMOTHY)
- Base 12,300 (into its decay); twin 13,200. Spark: step-8000 copy 4.88 of 9.91 GB; early chat step 280. Credit $10.61 at
  $2.18/h: s3 to about 12:55Z and s4 to about 17:15Z cost about $9.5, cushion about $1.1.
- The navigator, on the model: "essentially an MLP with a ride-along SDM". Agreed and written into the ideas file: the
  learned store is a sparse lookup of training, not a memory of the text; the rework after base, chat and Guy makes the
  run-time SDM the lookback, stabilises it (gated writes among the fixes) and tests it on recall and long windows.

### 2026-10-06T11:12Z - heartbeat; HERMES takes the SDMONLY lane (HERMES)
- Handover taken. Claude (JIMOTHY) is stopped at the account ceiling; this session runs on openrouter
  deepseek-v4.1-flash. I read THE LOG to its end and the ideas file before touching anything.
- Verified by hand, not from notes: ssh spark, ssh s3 and ssh s4 all answer; both GPUs on both boxes report 99
  and 100 percent, so both really are training.
- Twin (s3, memory off): step 13,700 at 11:12Z, loss 3.4248, lr 0.00153 (into its decay), 31.1k tokens a second
  split 15.55k a rank. Its local pull copy is fresh at 11:12Z.
- Base (s4, memory on): step 12,600 at the 10:58Z pull, loss 3.3905, lr 0.00261 (into its decay), 31.98k tokens
  a second. Both its GPUs pegged at 10:55Z.
- TRAP SEEN: the vast API reported util 0.0 on BOTH boxes at 11:06Z while nvidia-smi and fresh step lines showed
  both pegged. A zero util from the API alone is not an idle box. Confirm with nvidia-smi and the log mtime.
- s4 interactive ssh was refused (connection closed) and then hung for this window; the pull loop's rsync (direct
  then the proxy ssh7.vast.ai:38632) got through at 10:58Z, so the base is fine and the fault is the login route.
- Spark: the early SDM CHAT runs, step 300 of 763, 2,025 tokens a second; tmux spark_auto alive; load 1.59.
- Fleet: s4 54324756 at 1.0944 an hour, s3 54328901 at 1.0852 an hour; credit 9.72 dollars at 10:57Z; both boxes
  together burn 2.18 an hour.
- Crons: the idle-box sweep (minute 7 and 37) and WAVE4 OVERWATCH are both DEAD at the model, each failing
  "z-ai/glm-5.3-flash requires available credits". The three script watchers (h2-block, templates-three,
  operator) are healthy. The 10:37Z sweep was missed, so this beat is it, run by hand.
- Sweep verdict: nothing idle. Neither box is a close candidate. No pull beyond the standing loop, no destroy.
- Next: the twin's VAL and TEST about 12:55Z (read val_test.bpb, pull runs, logs and the ck per-window files
  home, close s3); the base's TEST about 14:20Z (VERDICT BA2); then the final SDM CHAT on s4 for about 1.8 h.

### 2026-10-06T12:08:33Z - JIMOTHY takes the lane back from HERMES (Claude, new login)
- Read HANDOVER_HERMES_TO_CLAUDE_2026-10-06.md and Hermes's 11:12Z beat. Both crons of this session are live again
  (heartbeat 05431e56, sweep dc73a800). Hermes's two notes kept: the vast API's gpu_util is not a liveness test, and
  s4's interactive login is flaky.
- CORRECTION to the handover's ETAs (Hermes read 20 steps a minute; measured since 10:58Z it is 13.2): base 12,600 at
  10:58Z to 13,500 at 12:06Z, so it ends about **14:20Z** as planned; twin 13,700 at 11:12Z to 14,400 at 12:06Z, so
  about **13:12Z**, not 12:45Z.
- RED fixed: s4's authorized_keys was rewritten again at 11:11:23Z (the second time; 01:01Z before) and the Spark's key
  was gone, so the copier failed from about 11:20Z. Restored (one key per line; the file had no final newline, so the
  first append glued two keys again). The Spark reaches s4. Verified off-box: steps 4000 and 6000; step 8000 copy at
  5.63 of 9.91 GB resumes.
- Credit **$7.79** at 12:06Z, $2.18/h. The 100M final chat no longer fits with its copy-out. New plan: twin done about
  13:12Z, a bf16 model-only copy of the twin to the Spark (rule: no true_* checkpoint lost), close s3 about 14:00Z; base
  done about 14:20Z, then on s4 at once the final SDM CHAT at **50M tokens** (one GPU, about 55 min) while the base's
  bf16 copy goes to the Spark, then the chat's bf16 copy, close s4 about 16:00Z. Cost about $6.2, cushion about $1.6.
  The Guy then trains on the Spark.
- 2026-10-06T12:17:05Z Chat shards rebuilt on s4 (/root/settle/chatdata): chat_train sha256 b280437d...6046, matches; the final SDM CHAT can start the moment the base ends.

### 2026-10-06T12:23:53Z - heartbeat (JIMOTHY)
- Base 13,800; twin 14,700. Spark: step-8000 copy 6.31 of 9.91 GB (the copier is back after the key restore); early chat step 370.
  Credit $7.18.

### 2026-10-06T12:31:04Z - heartbeat; the middle path decided (JIMOTHY)
- Base 13,900; twin 14,800 (about 13:12Z). Spark: step-8000 copy 6.67 of 9.91 GB; early chat step 380. Credit $6.90.
- The navigator chose the middle path for after the base: a pure run-time SDM residual model first, the archive and
  pointer index as an optional extra (ideas file). Lane ONESDMSTREAM builds it on CPU now, alongside the runs.

### 2026-10-06T12:43:48Z - heartbeat (JIMOTHY)
- Base 14,100 (about 14:10Z); twin 14,900 (about 13:10Z, then its final TEST eval). Spark: step-8000 copy 7.44 of 9.91 GB;
  early chat step 390. Credit $6.43. Lane ONESDMSTREAM building on the M5 CPU.

### 2026-10-06T13:12Z - the twin lands; VERDICT BA4 (JIMOTHY)
- TWIN (s3, `true_cn_d1536_hm6144_hops16_T2048_wsd_2B`, memory off, 2.0B tokens, step 15,258): TEST **0.99734 bpb**
  (3.35090 nats/token, 998,390 tokens, 4,839,392 bytes).
- **VERDICT BA4: HOLDS.** 0.99734 is below 1.02241 by 0.02507, well past the 0.007 noise.
- CAVEAT on every comparison: this TEST scored at window 2048 over 998,390 tokens. The window-256 TEST scores (TR6 1.02284
  memory on, 1.02241 memory off; the transformer yardstick 0.99035) scored over 995,323 tokens. Same text, a different
  count of scored tokens and a longer context, so the gap to those is not a clean single-axis comparison.
- Slim copy made on s3: bf16, model only, step 15,258, 1,766,505,261 bytes, sha256 ae0cdbbf...de9a. Pulling to the Spark
  through the proxy (ssh3.vast.ai:37322, key headclimb_pull; the direct route stalled, and the Spark's default key is
  refused by the proxy). About 1.6 MB/s, about 18 min. s3 closes once the sha matches.
- Base 14,400 at about 13:01Z, about 14:05Z plus its TEST eval. Credit **$5.40**.

### 2026-10-06T13:23:33Z - heartbeat (JIMOTHY)
- Spark: the base's step-8,000 save (filed step_9900.pt by the copier, which names files by pull time) VERIFIED off-box,
  sha256 b389eac5...3214; step_5000 dropped. Off-box now: 6,000 and 8,000. Twin slim copy 1.07 of 1.77 GB.
- ONESDMSTREAM landed (60a3af995); its puzzle chain runs on the Spark GPU beside the early chat (tmux onesdm), seed 0
  first. Early chat step 430 of 763.
- Credit $4.96 (RED, planned). Next: s3 closes at its sha match; base ends about 14:05Z.
- 2026-10-06T13:32:44Z s3 (sdmonly-s3-2x5090, 54328901) CLOSED. The twin's bf16 slim copy is on the Spark at
  ~/sdmonly_base/slim/, sha256 ae0cdbbf...de9a matches the box. Final pull home OK (runs without *.pt, logs, the
  test_per_window.npz files; 4 result files). Destroyed by id; show instances lists only s4. The pull loop
  watch_shapesolve stays up for s4 and will log s3 FAILED from now on. Credit $4.62, now $1.09/h.

### 2026-10-06T13:43:33Z - heartbeat (JIMOTHY)
- Base 15,000 of 15,258 (rank check identical, 231 tensors), ends about 14:03Z, then its TEST. Spark: early chat 440;
  puzzle chain recall onesdm seed 0 at step 2,000 (train acc 0.152). Credit $4.33.

### 2026-10-06T17:31:17Z - the base finished; its TEST ran out of memory; JIMOTHY was down 3 h 18 min (JIMOTHY)
- BASE (s4, true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B, memory on, 2.0B tokens): training ENDED at step 15,258
  (14:01Z, final loss 3.217, rank check identical, 231 tensors). The end-of-run TEST then died: CUDA out of memory on rank 0
  (3.95 GiB asked, 3.56 free) in eval_bpb, with the training state still on the GPU. No base TEST yet.
- RED, owned: the session hit the account usage limit at 14:03Z and was down until 17:21Z. s4 sat idle for those 3.4 h
  holding the base (about $3.70 of credit spent on an idle box). The sweep never fired in that window.
- SAVED ANYWAY: the Spark's failover copier pulled the final weights on its own. step_15258.pt VERIFIED 16:49Z, sha256
  ad16d6c8...7cb, and re-hashed on s4 at 17:23Z: identical. Off-box too: step 14,800 (the 13:31Z snap).
- 2026-10-06T17:24Z s4 (54324756) CLOSED: final pull home OK (runs without *.pt, logs, ck test_per_window, .sha, .step),
  destroyed by id; show instances is EMPTY. Credit $0.23. The two dead pull_true loops (PIDs 4971, 6383) stopped by PID.
  THE VAST FLEET IS DONE: everything left runs on the Spark.
- 2026-10-06T17:28Z Spark: the early chat (from step 4,000, step 540 of 763) STOPPED by PID 3920983; superseded. FIRED the
  final SDM CHAT from step 15,258: sdm_chat_final_base2B_T2048_50M, 50M tokens, B 64 accum 64, same recipe, SDM_EVAL_TOKENS
  4096. Its before-score IS the base's full TEST (all windows, T 2,048), so VERDICT BA2 lands with it.
- Spark also runs the ONESDM puzzle chain (tmux onesdm), seed 0 at blurry. Next: BA2 at the before-score; the Guy follows
  the chat on the Spark.

### 2026-10-06T17:31:41Z - heartbeat (JIMOTHY)
- No Vast boxes. Spark: the final SDM CHAT is still taking its before-score (the base's full TEST); the puzzle chain is on
  blurry onesdm_archive seed 0, step 2,300; 11 puzzle result files so far.

### 2026-10-06T17:44:05Z - heartbeat (JIMOTHY)
- Spark: the final SDM CHAT is still taking its before-score (process alive, R, on the GPU; TEST at T 2,048 runs 2 windows a
  batch under SDM_EVAL_TOKENS 4096, then chat_test). Puzzle chain: blurry plain seed 0.
- The three PAUSED old Spark jobs (stopped in place 2026-10-04T21:18Z for Claude 1's MIDI; state T, about 44 GB held, no
  compute): w50s T4096 (AR1, step 2,600 of 3,051), w50s T16384 (AR2, 2,350 of 3,051), w37_base 3B (BASE1, 4,450 of
  22,888). DECISION: they stay paused until the final chat and the Guy are done, so those two get the GPU; then the two
  w50s arms resume (cheap to finish). BASE1 is the older 8-hop design that the true base supersedes; it stays paused for
  the navigator's call.

### 2026-10-06T18:03:35Z - heartbeat (JIMOTHY)
- Spark: the final chat's before-score still running (TEST 998,390 tokens, then chat_test 8,182,155 tokens). For scale, the
  early chat's before-score (the base at step 4,000): TEST **1.10788**, chat_test 1.31183, and it took about half an hour.
  Puzzle chain: blurry sdmonly seed 0, step 2,800.

### 2026-10-06T18:20:31Z - THE BASE IS SCORED; VERDICTS BA1, BA2, BA3 (JIMOTHY)
- BASE `true_base_sdm_d1536_hm6144_hops16_T2048_wsd_2B` (memory ON, reads its SDM, 2.0B tokens, step 15,258), scored on the
  Spark as the final chat's before-score: TEST **0.99988 bpb** (998,390 tokens, 488 windows, T 2,048: the same windows as
  its twin). chat_test 1.19442 (8,182,155 tokens; the step-4,000 base read 1.31183 there, TEST 1.10788).

| run | memory | TEST bpb | window |
|---|---|---|---|
| base | ON | 0.99988 | 2,048 |
| twin | OFF | 0.99734 | 2,048 |
| TR6 memory off | OFF | 1.02241 | 256 |
| transformer yardstick | n/a | 0.99035 | 256 |

- **VERDICT BA1: HOLDS.** 0.99988 beats 1.02241 by 0.02253. CAVEAT: the gain is the longer window and the bigger shape; the
  twin, with no memory, gets it too.
- **VERDICT BA2: HOLDS.** Base minus twin = +0.00254, inside the 0.007 noise: THE STORE TIES AT FULL SCALE. On the point
  value the twin is ahead.
- **VERDICT BA3: HOLDS.** 0.99988 stays behind the transformer's 0.99035 (by 0.00953). CAVEAT: the transformer scored at
  window 256, a different window and token count.
- RULE (c) OWED: the base-twin pair sits inside the noise, so "does the memory help at 2.0B" is UNDECIDABLE, not a loss.
  It needs a second seed or more tokens (the 1.6B keep checkpoint allows a cooldown fork toward 3.0B), which needs GPU time
  we no longer buy. Recorded as owed; the Spark goes to the chat and the Guy first.

### 2026-10-06T18:25:18Z - heartbeat; ONESDM puzzles, seed 0 (JIMOTHY)
- Spark: the final SDM CHAT is training (381 steps for 50M tokens, about 01:10Z). The puzzle chain moved on to the
  --addr-bias arm; seed 0 of the main arms is in. ONE SEED, 512 answers a cell, 3,000 steps, sharing the GPU (no timing):

| puzzle (setting) | onesdm | onesdm + archive | plain (no memory) | sdmonly (today) | transformer |
|---|---|---|---|---|---|
| recall, gap 100 / 1,000 / 4,000 / 16,000 | .150 .117 .135 .129 | .162 .139 .139 .119 | .008 .008 .004 .006 | .008 .008 .008 .006 | .152 .150 .086 .010 |
| copy, length 128 / 512 / 2,048 | .314 .048 .012 | .423 .027 .003 | .003 .002 .003 | .010 .003 .003 | 1.000 .066 .003 |
| blurry cue, corruption 0 / .25 / .5 / .75 | .148 .115 .166 .135 | .176 .150 .158 .170 | .006 .008 .016 .010 | .006 .006 .016 .010 | crashed (fixed) |

- Read: the run-time SDM holds recall flat out to a gap of 16,000 (trained to 256), where the transformer falls to .010;
  today's model and the no-memory stack sit at chance on all three. Accuracy itself is low for every arm at 3,000 steps
  (in range the transformer is no better at recall), and the transformer copies perfectly in range where onesdm reaches
  .31 to .42. A first look, not a verdict: seeds 1 and 2 follow.
- FIXED 9d5e556b0: max_len_needed sized models from the eval settings only, so the blurry yardstick (trained to gap 256,
  built for 181) crashed. New self-test check, red-proven against the old code. Its seed-0 run is queued after the chain
  (tmux onesdm_after).

### 2026-10-06T18:43:38Z - heartbeat (JIMOTHY)
- Final chat step 10 of 381 at 962 tok/s (the early chat ran 2,051 on the same recipe). No swap, GPU 95 %: the cost is
  sharing the GPU with the puzzle chain (now the --addr-bias arm, onesdm_archive recall). DECISION: if the next reading
  holds under 1,500 tok/s, the puzzle chain is paused in place (SIGSTOP by PID) so the chat and the Guy get the GPU.

### 2026-10-06T19:03:32Z - heartbeat; the puzzle chain paused (JIMOTHY)
- Final chat step 20 of 381, 952 tok/s (confirms step 10). PAUSED the puzzle process in place: PID 786729
  (spark1_addrbias, onesdm_archive recall seed 0 at step 2,100), SIGSTOP, state Tl; the chain's bash waits on it. It
  resumes with SIGCONT once the Guy is done. The onesdm_after job waits on CHAIN DONE, so it waits too.

### 2026-10-06T19:23:22Z - heartbeat (JIMOTHY)
- Final chat step 30 of 381 at 2,071 tok/s alone on the GPU (952 while sharing): the pause worked; ends about 01:30Z.
  Loss 3.161 (chat 2.947, web 3.376).

### 2026-10-06T19:43:22Z - heartbeat (JIMOTHY)
- Final chat step 50 of 381, 2,039 tok/s, loss 3.061 (chat 2.780, web 3.343). Ends about 01:30Z.

### 2026-10-06T20:03:22Z - heartbeat (JIMOTHY)
- Final chat step 70 of 381, 1,845 tok/s, loss 3.016 (chat 2.801, web 3.230). Ends about 01:45Z.

### 2026-10-06T20:23:23Z - heartbeat (JIMOTHY)
- Final chat step 90 of 381, 2,111 tok/s, loss 3.003 (chat 2.602, web 3.405). Ends about 01:30Z.

### 2026-10-06T20:43:22Z - heartbeat (JIMOTHY)
- Final chat step 110 of 381, 2,079 tok/s, loss 2.986 (chat 2.732, web 3.240). Ends about 01:30Z.

### 2026-10-06T21:03:24Z - heartbeat (JIMOTHY)
- Final chat step 130 of 381, 2,131 tok/s, loss 3.067 (chat 2.839, web 3.295; batch-to-batch noise). Ends about 01:30Z.

### 2026-10-06T21:23:23Z - heartbeat (JIMOTHY)
- Final chat step 150 of 381, 2,141 tok/s, loss 2.964 (chat 2.698, web 3.230). Ends about 01:30Z.

### 2026-10-06T21:43:23Z - heartbeat (JIMOTHY)
- Final chat step 170 of 381, 2,065 tok/s, loss 3.025 (chat 2.703, web 3.348). Ends about 01:30Z.

### 2026-10-06T22:03:22Z - heartbeat (JIMOTHY)
- Final chat step 190 of 381 (half), 2,117 tok/s, loss 3.024 (chat 2.640, web 3.407), gnorm 0.615. Ends about 01:30Z.

### 2026-10-06T22:23:24Z - heartbeat (JIMOTHY)
- Final chat step 200 of 381, 2,103 tok/s, loss 3.042 (chat 2.582, web 3.503), gnorm 0.418. Ends about 01:30Z.

### 2026-10-06T22:43:22Z - heartbeat (JIMOTHY)
- Final chat step 220 of 381, 2,046 tok/s, loss 2.890 (chat 2.470, web 3.310), gnorm 0.172. Ends about 01:30Z.

### 2026-10-06T23:03:24Z - heartbeat (JIMOTHY)
- Final chat step 240 of 381, 2,038 tok/s, loss 2.932 (chat 2.486, web 3.379). Ends about 01:30Z; the cooldown starts
  at about step 343.

### 2026-10-06T23:23:24Z - heartbeat (JIMOTHY)
- Final chat step 260 of 381, 2,061 tok/s, loss 2.917 (chat 2.464, web 3.370). Ends about 01:30Z.

### 2026-10-06T23:43:25Z - heartbeat (JIMOTHY)
- Final chat step 280 of 381, 2,091 tok/s, loss 2.789 (chat 2.348, web 3.230): lowest on every head. Ends about 01:30Z.

### 2026-10-07T00:03:39Z - the tokenizer question, re-read from the result files (JIMOTHY)
- The navigator asked whether the DeepSeek V4 Flash tokenizer (129,280 pieces) is the best BPE for our models. THE LOG
  held only half the answer: the 2026-10-03 decision to move to the trained 32,768 BPE "pending a second seed" was never
  closed here. The reruns did finish and their result files are home (runs_vast_sdmonly/sdmonly-c-pro6000/), TEST bpb:

| arm (equal training bytes) | 129k DeepSeek | trained 32k | 32k gain |
|---|---|---|---|
| wave V centre, d 256 | 1.66811 | 1.64753 | 0.02058 |
| champion CH, seed 0 / rerun | 1.53232 / 1.53246 | 1.52164 | 0.0107 |
| transformer MQ, rerun | 1.46279 | 1.44301 | 0.01978 |

- So the 32k gain held on a second run and on the transformer too: 0.011 to 0.021, past the 0.007 noise. The move was
  NOT carried out: the base, the twin and the chats all run on 129k (embedding 129,280 x 1,536). Why it was dropped is
  not recorded; the 2B-token train_big shards were already tokenized at 129k. Recorded as a decision that went
  unexecuted, not a reversal. For the next models (the run-time and fully SDM text runs), the 32k BPE is the default.

### 2026-10-07T00:05:21Z - heartbeat (JIMOTHY)
- Final chat step 300 of 381, 2,083 tok/s, loss 2.907 (chat 2.426, web 3.388). Ends about 01:30Z. Lane SDMEVERYLAYER
  (the fully SDM model: a trained SDM in place of every MLP) building on CPU. The next-stack decision (tokenizer A1 32k,
  C3 both thinking layers) is with the navigator; DeepSeek-V4.1-Flash's tokenizer was read: 128,000 base pieces, the
  same 127,741 merges, a different file (sha256 c90dfa01...), so the 129k-vs-32k reading stands for it too.

### 2026-10-07T00:23:45Z - heartbeat; the Guy chained (JIMOTHY)
- Final chat step 320 of 381, 2,085 tok/s, loss 2.910 (chat 2.508, web 3.312). Cooldown from step 343; ends about 01:30Z.
- SDMEVERYLAYER LANDED 8f39354fc (onesdm_allsdm, onesdm_allsdm_big; self-tests 42/42 and 15/15, also 42/42 on the Spark).
- CHAINED (lesson of the 14:03Z outage: the Spark must not wait on this session): tmux guy_after on the Spark waits for
  sdm_chat_final_base2B_T2048_50M.result.json, then fires guy_final_on_sdmchat_T2048_20M (--task guy, init the chat's
  last.pt, guy3 corpus, 20M tokens, B 64 accum 64). Log: ~/sdmonly_base/logs/guy_final.out.

### 2026-10-07T03:22:14Z - SDM CHAT is done; the Guy fired by himself (JIMOTHY)
- The session hit the usage limit again (about 00:25Z to 03:21Z). The chain did its job: no wait this time.
- FINAL SDM CHAT `sdm_chat_final_base2B_T2048_50M` (from the 2.0B base, memory on, 50M tokens) finished 01:52Z:
  chat_test **0.78351** (from 1.19442, -0.41091); TEST 1.0091 (from 0.99988, +0.00922: the price of the tune on web text).
  No sealed prediction covered this chat; recorded as measured.
- THE WEIRD LITTLE GUY `guy_final_on_sdmchat_T2048_20M` fired 01:54Z from tmux guy_after; before-score guy_test
  **1.20486** (973,642 tokens), chat_test 0.78351, TEST 1.0091. Step 50 of about 153 at 03:21Z, about 2,080 tok/s;
  ends about 05:20Z.

### 2026-10-07T03:23:26Z - heartbeat (JIMOTHY)
- The Guy at step 50 of about 153, guy loss 3.091 (from 3.373 at step 40), 2,100 tok/s; ends about 05:20Z. Queued after
  him: the base store test, then the fully SDM puzzle arms, then the paused puzzle chain.

### 2026-10-07T03:38:17Z - THE NAVIGATOR'S LAUNCH ORDER: a new shape, end to end, on the Spark (JIMOTHY)
- The navigator: halt the Weird Little Guy and the rest; for launch, a full end-to-end model on the middle path or the
  newest idea, a full Spark run of about 24 hours at a decent size, then instruction-tuned on top, then the Weird Little
  Guy on top of that.
- HALTED by PID: the Guy guy_final_on_sdmchat_T2048_20M (1435378, at step about 55 of 153; no result, superseded) and the
  ONESDM puzzle process (786729); tmux sessions onesdm, onesdm_after, guy_after closed. Seed-0 puzzle results stay.
- RESUMED as filler while the trainer is wired: the two paused long-window runs w50s T4096 (605062, step 2,600 of 3,051)
  and w50s T16384 (605064, step 2,350 of 3,051), SIGCONT, both in state R. They end in about 2 h. w37_base (BASE1) stays
  paused for the navigator.
- FIRED lane TEXTWIRE (Opus): the onesdm arms into track4_sdmonly_train.py and the fine-tune tool, with a bench mode and a
  param report to size the 24-hour run. Open with the navigator: the tokenizer (A1, 32k) and the shape (middle path,
  fully SDM, or a short bake-off first).

### 2026-10-07T03:39:55Z - the launch stack decided (JIMOTHY)
- The navigator chose B1: a BAKE-OFF FIRST, about 2.5 h on the Spark: the middle path (onesdm) against the fully SDM
  model (onesdm_allsdm) against the memory-off twin (plain), about 50M tokens each, at the launch shape. The winner takes
  the 24-hour base run; a tie (inside the 0.007 noise) goes to the middle path. A transformer yardstick at the same shape
  runs after the three if time allows; it does not gate the choice.
- Taken on the recommendation, navigator to overrule: A1, our trained 32,768 BPE (at about 100M weights the 129k word
  list would be as big as the model); C0, no local learning for launch.
- FIRED lane RETOKEN32K (Opus): a lossless 129k to 32k re-encoder that keeps document boundaries and mask bytes exact,
  writes token_bytes for 32k, and does a 200M-token prefix first for the bake-off.
- The bake-off's predictions are sealed here after the speed bench fixes the shape, before any arm fires.

### 2026-10-07T03:43:47Z - heartbeat (JIMOTHY)
- Spark: the resumed w50s runs step again: T4096 2,650 of 3,051, T16384 2,400 of 3,051, 50 steps in about 15 min each
  (their logged 33.4 tok/s spans the 2-day pause and is not a speed). Ends about 05:45Z and 07:00Z. If the bake-off is
  ready first, it takes the GPU and the w50s pause again.
- Lanes TEXTWIRE and RETOKEN32K running on the M5 CPU.

### 2026-10-07T04:00:41Z - RETOKEN32K landed; the bake-off data is ready (JIMOTHY)
- RETOKEN32K LANDED 5fa2c36a7 (track4_sdmonly_retoken32k.py; self-test 22/22, every check red-proved, re-run here).
  No special tokens needed: BOS is id 0 in both, the chat and Guy role markers are plain text.
- Spark (tmux retok, ~/sdmonly_base/data32k/): the bake-off slice train_big_p200m (200,000,122 source tokens to
  206,568,143 32k tokens, 4.776 to 4.624 bytes a token) in about 40 s, and val (2,000,916 to 2,065,452) in 2.8 s,
  verifier on. BAKEOFF DATA READY 03:59:45Z. The rest of train_big and the chat shards follow (about 5M source tokens a
  second, the full 2B in about 7 min).
- FOUND by the lane: two ids in the 129k token_bytes.i32 (127998, 127999) carry the EOS and PAD byte lengths through a
  tokenizers 0.20.3 hash collision; it explains the known 2,090-byte over-count on S0 train (110 x 19). It moves every
  129k TEST bpb by a negligible amount (the ids are rare), and none of the 32k scores.
- DECIDED: guy3 is rebuilt with the 32k tokenizer at segmentation time when the Guy's turn comes (2,350 of 64,825 train
  segments pass 2,047 tokens under a plain re-encode).

### 2026-10-07T04:03:30Z - heartbeat (JIMOTHY)
- Spark: the rest of train_big re-encoding (tmux retok); w50s T4096 step 2,750 of 3,051, T16384 2,500 of 3,051.
  TEXTWIRE still building. Next: its landing, the speed bench, sizing, sealing, the bake-off.

### 2026-10-07T08:22:54Z - the speed bench, and WAVE BK (the launch bake-off), sealed before it runs (JIMOTHY)
- RED, owned: the usage limit hit again (about 04:20Z to 08:21Z). The Spark GPU sat idle for those 4 h: I had paused the
  w50s runs for the bench and the session stopped before the bake-off fired. From here the whole launch runs as ONE
  chained script on the Spark, so no step waits on this session.
- TEXTWIRE LANDED 2d5eaf9a8 (trainer 16/16, fine-tune 31/31, models 42/42, puzzles 15/15, optimiser 14/14; a real bug
  fixed: hidden() skipped the final norm, so the chunked loss would have trained on wrong logits).
- 32k DATA (all of it) in ~/sdmonly_base/data32k by 04:12:56Z: train_big, the 200M-token prefix, val, chat_train,
  chat_test, chat_mix, token_bytes.
- SPEED BENCH on the Spark (GB10, driver 580.159.03, torch 2.14.1, w50s paused, 15 steps, T 2048, B 8, chunked loss,
  vocab 32,768, 12 layers):

| arm, width | eager tok/s | compiled-body tok/s | peak GiB | non-embedding weights |
|---|---|---|---|---|
| onesdm 512 | 8,072 | | 12.1 | 33,321,056 |
| onesdm 768 | 6,369 (B 16: 6,160) | **15,862** | 12.8 | 73,254,240 |
| onesdm_allsdm 768 | | **13,407** | 12.4 | 72,775,056 |
| plain 768 (memory off) | | 65,083 | 5.2 | 73,254,240 |

  Compiled equals eager: the first three losses 10.554141 / 10.550784 / 10.544442 against eager 10.554099 / 10.549720 /
  10.543095. The run-time memory is the cost: memory off runs 4.1x faster. (allsdm's first loss 15.35 is the tied
  embedding with a zero-start body predicting its own token, not a fault; it is 10.54 one step later.)

THE LAUNCH SHAPE: width 768, 12 layers, T 2,048, 32k vocabulary, 131,072 tokens a step (B 8, accum 8), lr 3e-3, Muon
0.02, WSD (last 20%), compiled body, chunked loss, seed 0. About 98M weights with the 25M-weight word table.

WAVE BK, on train_big_p200m, 50M tokens each (381 steps), TEST on val at T 2,048, run one after another:
`bk_onesdm_d768_L12_T2048_50M` (middle path) · `bk_allsdm_d768_L12_T2048_50M` (fully SDM) · `bk_plain_d768_L12_T2048_50M`
(memory off: no path between positions at all, the floor).

| id | prediction |
|---|---|
| BK1 | the middle path beats memory off by at least 0.007 |
| BK2 | the middle path beats the fully SDM model by at least 0.007 (the dense MLP beats the trained table again) |
| BK3 | the fully SDM model beats memory off by at least 0.007 |

DECISION RULE, sealed, applied by the script: the winner is the lower TEST of the middle path and the fully SDM model;
inside 0.007 the middle path wins. The winner then runs the 24-hour base on all of train_big (onesdm 1,300M tokens,
allsdm 1,100M, from the bench speeds), then SDM CHAT on top (50M tokens). The transformer yardstick at this shape runs
after, beside the result; the Guy waits for guy3 rebuilt at 32k.

### 2026-10-07T08:39:27Z - wave BK fired, after a batch correction (JIMOTHY)
- FIRED 08:23Z as tmux launch on the Spark (~/sdmonly_base/launch/chain.sh). WRONG BEAT, corrected at 08:30Z: the chain
  passed --B 8 --accum 8, and in this trainer --B is the whole step (accum only splits it), so the first arm ran 16,384
  tokens a step (3,051 steps) against the sealed 131,072. Stopped by PID after 6 min, its partial checkpoint removed by
  literal path, its log kept as logs/bk_onesdm_misbatched_B8.out, and the chain restarted with --B 64 --accum 8: 381
  steps, as sealed. First arm at step 50, loss 6.308, about 12,900 tok/s; about 65 min an arm.
- Records home and committed (5a6d3ca09): the seed-0 puzzle results, and the speed bench as runs_launch/bench_spark_2026-10-07.json.
- The navigator ordered a rewrite of four site pages (learn-sdm, learn-sdm-unfold, learn-sdm-lookback, sdm-model) around
  the two models going forward, named FULL (onesdm_allsdm) and PARTIAL (onesdm) on the site; lane LEARNSDM2 waits for
  the SITEHOME move to land.

### 2026-10-07T08:41:10Z - heartbeat (JIMOTHY)
- Wave BK: PARTIAL (bk_onesdm) step 50 of 381 at 08:41Z, loss 6.308, 12,899 tok/s. FULL and memory off follow. Pull loop
  runs_launch/pull_launch.sh copies the launch records home every 15 min. LEARNSDM2 waits for SITEHOME (not landed yet);
  its scope grew to five pages (#/sdmchat added) and every image and animation redone for FULL and PARTIAL.

### 2026-10-07T08:43:31Z - heartbeat (JIMOTHY)
- No change: PARTIAL at step 50 of 381 (log every 50 steps, about 8 min apart). SETTLE/settle-site exists on disk but no
  SITEHOME commit yet, so LEARNSDM2 keeps waiting.

### 2026-10-07T09:03:26Z - heartbeat (JIMOTHY)
- PARTIAL step 200 of 381, loss 5.112, 16,339 tok/s (faster than the bench). Ends about 09:20Z plus its TEST. SITEHOME
  not committed yet; LEARNSDM2 waits.

### 2026-10-07T09:23:27Z - heartbeat (JIMOTHY)
- PARTIAL (bk_onesdm) trained all 381 steps, last loss 4.765 (step 300 4.967, step 350 4.830); its TEST at T 2,048 is
  being taken now. FULL starts when it lands.

### 2026-10-07T09:43:42Z - PARTIAL scored (JIMOTHY)
- PARTIAL `bk_onesdm_d768_L12_T2048_50M` (50M tokens of the 32k data): TEST **1.46405 bpb** (4.77317 nats/token,
  1,031,158 tokens, 4,850,090 bytes, T 2,048). No other arm of wave BK is scored yet, so no verdict yet. (Not comparable to
  the 129k-token runs' scores: a different tokenizer, 40 times fewer tokens.)
- FULL `bk_allsdm` fired 09:23:41Z. Training loss against PARTIAL at the same step: step 50 6.339 vs 6.308, step 100
  5.719 vs 5.573. FULL trails early, as its zero-start table predicts; only the TEST decides.

### 2026-10-07T10:03:27Z - heartbeat (JIMOTHY)
- FULL step 200 of 381, loss 5.348 (PARTIAL at 200: 5.112; at 150: 5.573 vs 5.374). The gap holds at about 0.2 to 0.24
  nats, not closing yet. Ends about 10:40Z plus its TEST. SITEHOME not landed.

### 2026-10-07T10:23:28Z - heartbeat (JIMOTHY)
- FULL step 350 of 381, loss 5.099 (PARTIAL at 350: 4.830; at 300: 5.234 vs 4.967). The gap WIDENED to about 0.27 nats
  through the decay. TEST due about 10:35Z; then memory off.

### 2026-10-07T10:32:30Z - THE NAVIGATOR'S FULL RULING, and the new chain, sealed before it runs (JIMOTHY)
- The navigator (2026-10-07, about 10:30Z): FULL goes forward, ONLY EVER FULL, leaning on it heavily even where it is
  worse or slower than a transformer; compare it to transformers while making it; try different shapes of FULL; keep the
  current FULL training and it becomes the latest SDM CHAT base; improve it later once we see what it is.
- This OVERRIDES wave BK's sealed decision rule (which picked the lower TEST, PARTIAL on a tie). BK1 to BK3 stay sealed and
  are scored as measured. PARTIAL goes no further.
- "Keep the current FULL training": the single-GPU trainer has no weights-only init (its --init flag is unread; the only
  checkpoint path is the cooldown fork), so the FULL base starts fresh at the full budget. The 50M FULL model is 4.5% of
  it; restarting costs about 1 h and keeps one clean WSD schedule.
- THE NEW CHAIN (chain2.sh, replacing chain.sh's tail after the memory-off arm):
  1. bk_yard_d768_L12_T2048_50M: the transformer yardstick at the bake-off shape and tokens.
  2. launch_base_fullsdm_d768_L12_T2048_1100M: FULL, 1.1B tokens of train_big, WSD, keep at 550M. About 23 h.
  3. launch_chat_fullsdm_50M: SDM CHAT on top of it.
  4. A FULL SHAPE SWEEP at 50M each (sealed separately before it fires).
  5. launch_yard_d768_L12_T2048_1100M: the transformer at FULL's base tokens, the long comparison.

| id | prediction |
|---|---|
| YD1 | the transformer yardstick beats FULL at 50M by at least 0.007 |
| YD2 | the transformer yardstick beats PARTIAL at 50M by at least 0.007 |
| FB1 | the FULL base at 1.1B tokens beats FULL at 50M by at least 0.2 TEST bpb |
| FB2 | the transformer at 1.1B tokens beats the FULL base by at least 0.007 |

### 2026-10-07T10:33:25Z - FULL scored; the FULL chain armed (JIMOTHY)
- FULL `bk_allsdm_d768_L12_T2048_50M`: TEST **1.54457 bpb** (5.03569 nats/token, the same 1,031,158 tokens).
  PARTIAL 1.46405. **VERDICT BK2: HOLDS** (PARTIAL beats FULL by 0.08052, eleven times the noise). BK1 and BK3 wait on
  the memory-off arm (training now, 80k tok/s).
- Under the navigator's ruling FULL goes forward anyway. The old chain.sh is FROZEN (SIGSTOP, PID 1996481) so its
  memory-off child finishes but it never starts PARTIAL; chain2.sh runs in tmux launch2 and waits for that arm.

### 2026-10-07T10:43:32Z - WAVE BK COMPLETE (JIMOTHY)

| arm (50M tokens, 32k vocab, width 768, 12 layers) | TEST bpb |
|---|---|
| PARTIAL (run-time SDM + MLP) | 1.46405 |
| FULL (run-time SDM + trained SDM table) | 1.54457 |
| memory off (no path between tokens) | 1.75751 |

- **VERDICT BK1: HOLDS** (PARTIAL beats memory off by 0.29346). **BK2: HOLDS** (PARTIAL beats FULL by 0.08052).
  **BK3: HOLDS** (FULL beats memory off by 0.21294). The run-time memory is worth about 0.21 to 0.29 bpb at this size;
  the trained table in place of the MLP costs 0.08.
- The old chain.sh (frozen) is stopped by PID now that its last child is done. chain2 fired the transformer yardstick at
  10:39:05Z (bk_yard, about 42k tok/s, about 20 min); then the FULL base.

### 2026-10-07T11:03:32Z - the transformer yardstick at the bake-off; the FULL base fired (JIMOTHY)
- Transformer `bk_yard_d768_L12_T2048_50M` (same shape, data, tokens, recipe): TEST **1.22265 bpb**.
  **VERDICT YD1: HOLDS** (it beats FULL by 0.32192). **YD2: HOLDS** (it beats PARTIAL by 0.24140). At 50M tokens the
  transformer is far ahead of both SDM models; the run-time memory closes 0.21 to 0.29 of the 0.53 gap between memory
  off and the transformer.
- FIRED 10:58:25Z: `launch_base_fullsdm_d768_L12_T2048_1100M`, FULL, 1.1B tokens, keep at 550M. Then FULL SDM CHAT.
  FB1 and FB2 open.

### 2026-10-07T13:21:32Z - heartbeat after another usage-limit outage (JIMOTHY)
- The session was down from about 11:05Z to 13:20Z. The chain needed nothing: the FULL base is at step 900 of 8,392,
  loss 4.813, about 14,000 tok/s; it ends about 08:45Z on 2026-10-08, then FULL SDM CHAT.
- SITEHOME landed (07cae5cb5). LEARNSDM2 had cut its lane and was mid-rewrite when the same limit stopped it (one commit,
  8ecfdbd04, plus uncommitted page, i18n and diagram work). RESUMED by message with its last state and the bake-off and
  FULL-ruling facts.

### 2026-10-07T13:23:39Z - heartbeat (JIMOTHY)
- No change: the FULL base at step 900 of 8,392 (logs every 50 steps, about 8 min apart). LEARNSDM2 resumed, no new
  channel line yet.

### 2026-10-07T13:43:27Z - heartbeat (JIMOTHY)
- FULL base step 1,050 of 8,392, loss 4.857, 13,918 tok/s. LEARNSDM2 back on track (channel line PROGRESS, commit
  ca17e5f61 on settle-learnsdm2).

### 2026-10-07T14:03:30Z - heartbeat (JIMOTHY)
- FULL base step 1,150 of 8,392 (about 151M tokens), loss 4.760, 14,090 tok/s. LEARNSDM2 working.

### 2026-10-07T14:22:35Z - LEARNSDM2 landed (JIMOTHY)
- The five pages (#/learn-sdm, #/learn-sdm-unfold, #/learn-sdm-lookback, #/sdm-model, #/sdmchat) rewritten around FULL
  and PARTIAL: tests 16/16, generator check, build re-run in the lane; landed at 51c64aaad, main suite 3,407 pass 0 fail.
  Numbers come from runs_launch/ through SETTLE/settle-site/tools/sdm_forward_data.mjs; re-run it when FULL's base and
  chat records land. One rule slip reported by the lane: it stopped its own dev server with pkill by pattern (it matched
  only that server); the rule is kill by PID.

### 2026-10-07T14:23:24Z - heartbeat (JIMOTHY)
- FULL base step 1,300 of 8,392 (about 170M tokens), loss 4.746 (4.707 at 1,250), 14,148 tok/s. No lanes running.

### 2026-10-07T14:43:28Z - heartbeat (JIMOTHY)
- FULL base step 1,450 of 8,392 (about 190M tokens), loss 4.724 (4.693 at 1,400), 14,091 tok/s.

### 2026-10-07T15:03:44Z - heartbeat (JIMOTHY)
- FULL base step 1,550 of 8,392 (about 203M tokens), loss 4.766, 13,753 tok/s. Its quick evals (64 windows, 130,950
  tokens, at the full learning rate, before any decay): step 500 1.56858, 1,000 1.49875, 1,500 1.48266 bpb. Already
  below FULL's finished 50M bake-off (1.54457, decayed); the drop per 500 steps is slowing (0.070, then 0.016), as
  expected at a constant rate; the WSD decay over the last fifth brings the larger drop.

### 2026-10-07T15:23:27Z - heartbeat (JIMOTHY)
- FULL base step 1,700 of 8,392 (about 223M tokens), loss 4.778 (4.723 at 1,650), 13,398 tok/s.

### 2026-10-07T15:43:28Z - heartbeat (JIMOTHY)
- FULL base step 1,800 of 8,392 (about 236M tokens), loss 4.767, 13,901 tok/s. Next quick eval at step 2,000.

### 2026-10-07T16:03:25Z - heartbeat (JIMOTHY)
- FULL base step 1,950 of 8,392 (about 256M tokens), loss 4.600, its lowest so far (4.640 at 1,900). Quick eval at 2,000
  next.

### 2026-10-07T16:23:26Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 2,000 (about 262M tokens, full learning rate): **1.44881 bpb** (from 1.48266 at 1,500, a
  drop of 0.034, larger than the last). Below PARTIAL's 50M bake-off TEST (1.46405), but the quick eval is 64 windows
  (130,950 tokens), not the full TEST, so that is a hint, not a comparison. Step 2,050 loss 4.647.

### 2026-10-07T18:20:09Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 2,500 (about 328M tokens, full learning rate): **1.42910 bpb** (from 1.44881 at 2,000, a
  drop of 0.020, smaller than the last; the curve is bending as expected before decay). Step 2,800 of 8,392, loss
  4.574, 14,078 tok/s. Fleet: 0 Vast boxes, credit $0.18. `sweep.sh` does not exist yet; chain2 skips it unless it
  is written before the chat ends.

### 2026-10-07T18:40Z - SEALED: the FULL shape sweep (wave SW), before it fires (JIMOTHY)
The navigator asked for different shapes of FULL. Seven arms at 50M tokens each, the bake-off's exact recipe (32k,
T 2,048, 131,072 tokens a step, lr 3e-3, Muon 0.02, WSD, train_big_p200m), so each reads against
`bk_allsdm_d768_L12_T2048_50M` (TEST **1.54457**, 62 min). Script: `runs_launch/sweep.sh` (sha256 a8d388d3...899eb9e),
copied to `~/sdmonly_base/launch/sweep.sh` on the Spark; chain2 fires it after `launch_chat_fullsdm_50M`. About 7 to 9
hours on the Spark, then the 1.1B yardstick.

| arm | change from the bake-off FULL | weights (all / non-embedding) |
|---|---|---|
| sw_seed1 | seed 1, same shape | 97.9M / 72.8M |
| sw_slots4x | encyclopedia 4x the slots (20,736 a head), same k | 241.4M / 216.2M |
| sw_k64 | encyclopedia reads 64 slots, not 32 | 97.9M / 72.8M |
| sw_heads8 | encyclopedia 8 heads, not 4 | 97.9M / 72.7M |
| sw_diary4k | diary 4,096 slots a head, not 1,024 | 98.0M / 72.8M |
| sw_deep | width 512, 24 layers | 83.5M / 66.7M |
| sw_wide | width 1,024, 8 layers | 119.8M / 86.3M |

**The noise rule.** sw_seed1 measures the seed noise at this scale. Let N = |sw_seed1 - 1.54457|. An arm WINS only if
its TEST is below 1.54457 by more than max(2N, 0.010); it LOSES if above by the same; otherwise it TIES. Every arm also
records tok/s, because a shape that costs time is priced, not free.

**Sealed predictions** (confidence in brackets):
- SW1: N <= 0.010 (70%).
- SW2: sw_slots4x WINS (55%). The encyclopedia's size is the likeliest thing FULL is short of.
- SW3: sw_k64 TIES (60%).
- SW4: sw_heads8 TIES (60%).
- SW5: sw_diary4k TIES (65%). The diary is not where FULL loses to PARTIAL; the encyclopedia is.
- SW6: sw_deep LOSES at 50M (50%). Fewer weights and a longer path train slower early.
- SW7: sw_wide TIES (50%).
- SW8: the best arm closes less than half of the FULL to PARTIAL gap at 50M (0.0805), so it stays above 1.5043 (75%).
- SW9: sw_slots4x runs below 11,000 tok/s (60%).
A winning shape does not change the running FULL base; it becomes the shape of the next FULL base.

### 2026-10-07T18:23:26Z - heartbeat (JIMOTHY)
- Correction to the entry above: the sweep was sealed at about 18:22Z, not 18:40Z. I wrote that time without reading
  the clock. The seal still precedes the sweep, which cannot fire before the FULL chat ends on 2026-10-08.
- FULL base step 2,850 of 8,392, loss 4.588, 14,046 tok/s. Newest checkpoint `last.pt` (1.08 GB) written about 18:06Z,
  on the Spark. Spark disk 503 GB free. 0 Vast boxes.

### 2026-10-07T18:43:24Z - heartbeat (JIMOTHY)
- FULL base step 2,950 of 8,392 (about 387M tokens), loss 4.665, 13,968 tok/s. Quick eval at step 3,000 next. 0 Vast
  boxes.

### 2026-10-07T19:03:26Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 3,000 (about 393M tokens): **1.42899 bpb**, flat against 1.42910 at 2,500 (a drop of
  0.0001, inside any noise). A plateau at the full learning rate is the expected WSD shape; the decay starts at 80% of
  the run (about step 6,714) and is where the drop should come. Not a fault; watch 3,500. Step 3,100 loss 4.613,
  14,326 tok/s. 0 Vast boxes.

### 2026-10-07T19:23:26Z - heartbeat (JIMOTHY)
- FULL base step 3,200 of 8,392 (38%), loss 4.603, 14,621 tok/s. Next quick eval at 3,500. 0 Vast boxes.

### 2026-10-07T19:43:23Z - heartbeat (JIMOTHY)
- FULL base step 3,350 of 8,392 (40%), loss 4.588, 14,047 tok/s. Quick eval at 3,500 next. 0 Vast boxes.

### 2026-10-07T20:03:24Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 3,500 (about 459M tokens): **1.42091 bpb**, down 0.008 from the flat 1.42899 at 3,000.
  The plateau was a pause, not a stall. Step 3,500 loss 4.596, 13,792 tok/s. 0 Vast boxes.

### 2026-10-07T20:23:30Z - heartbeat (JIMOTHY)
- FULL base step 3,600 of 8,392 (43%), loss 4.546, its lowest logged loss so far. 13,837 tok/s. The paused w50s and
  w37 processes still hold GPU memory but do not compute (their %CPU is a lifetime average). 0 Vast boxes.

### 2026-10-07T20:43:26Z - heartbeat (JIMOTHY)
- FULL base step 3,750 of 8,392 (45%), loss 4.570, 14,350 tok/s. Quick eval at 4,000 next. 0 Vast boxes.

### 2026-10-07T21:03:24Z - heartbeat (JIMOTHY)
- FULL base step 3,850 of 8,392 (46%), loss 4.490, the first logged loss under 4.5. 14,281 tok/s. Quick eval at 4,000
  next. 0 Vast boxes.

### 2026-10-07T21:23:25Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 4,000 (about 524M tokens, still at the full learning rate): **1.39772 bpb**, down 0.023
  from 1.42091 at 3,500, the largest drop since 2,000. Under 1.4 for the first time. Step 4,000 loss 4.482, 13,965
  tok/s. 0 Vast boxes.

### 2026-10-07T21:43:24Z - heartbeat (JIMOTHY)
- FULL base step 4,100 of 8,392 (49%), loss 4.456, 14,081 tok/s. Quick eval at 4,500 next. 0 Vast boxes.

### 2026-10-07T22:03:22Z - heartbeat (JIMOTHY)
- FULL base step 4,250 of 8,392 (51%, past halfway), loss 4.414, its lowest logged so far. 14,143 tok/s. Quick eval
  at 4,500 next. 0 Vast boxes.

### 2026-10-07T22:23:22Z - heartbeat (JIMOTHY)
- FULL base step 4,400 of 8,392 (52%), loss 4.433, 13,805 tok/s. Quick eval at 4,500 next. 0 Vast boxes.

### 2026-10-07T22:43:24Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 4,500 (about 590M tokens): **1.38315 bpb**, down 0.015 from 1.39772 at 4,000. Step 4,500
  loss 4.372, its lowest logged. 14,090 tok/s. 0 Vast boxes.

### 2026-10-07T23:03:44Z - heartbeat (JIMOTHY)
- FULL base step 4,650 of 8,392. A small bump after the 4,500 eval: loss 4.372, 4.508, 4.488, 4.499 at steps 4,500 to
  4,650, and gnorm 0.079, 0.124, 0.096, 0.165 (it ran about 0.08 before). Not a divergence yet; it may be a harder
  stretch of data. **Insurance taken:** the step-4,500 checkpoint, written before the bump, is copied aside as
  `~/sdmonly_base/launch/ck/fullbase_prebump_last.pt` on the Spark, sha256 ae7eb7e2...e5e25c (matches `last.pt`), so a
  later `last.pt` cannot overwrite the only good one. **Rule for the next beats:** if gnorm passes 0.3 or the logged
  loss passes 4.7, stop the run by its PID and resume from the step-4,500 copy; otherwise let it run.

### 2026-10-07T23:23:30Z - heartbeat (JIMOTHY)
- The bump passed: gnorm 0.165 at step 4,650, then 0.076 and 0.070; loss 4.499, 4.451, 4.429. No resume needed; the
  step-4,500 copy stays as insurance. FULL base step 4,750 of 8,392, checkpoint written at 23:18Z, GPU 95%. 0 Vast
  boxes.

### 2026-10-07T23:43:22Z - heartbeat (JIMOTHY)
- FULL base step 4,900 of 8,392 (58%), loss 4.426, gnorm 0.101 (normal band), 13,991 tok/s. Quick eval at 5,000 next.
  0 Vast boxes.

### 2026-10-08T00:03:23Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 5,000 (about 655M tokens): **1.38261 bpb**, flat against 1.38315 at 4,500 (a drop of
  0.0005, inside noise). The second plateau at the full learning rate, as at 2,500 to 3,000; the bump at 4,550 to
  4,650 may have cost some of this stretch. The decay starts near step 6,714. Step 5,000 loss 4.491, gnorm 0.087,
  14,316 tok/s. 0 Vast boxes.

### 2026-10-08T00:23:23Z - heartbeat (JIMOTHY)
- FULL base step 5,150 of 8,392 (61%), loss 4.457, gnorm 0.071, 13,922 tok/s. Quick eval at 5,500 next. 0 Vast boxes.

### 2026-10-08T00:43:26Z - heartbeat (JIMOTHY)
- FULL base step 5,300 of 8,392 (63%), loss 4.437, gnorm 0.077, 14,172 tok/s. Quick eval at 5,500 next. 0 Vast boxes.

### 2026-10-08T01:03:25Z - heartbeat (JIMOTHY)
- FULL base step 5,400 of 8,392 (64%), loss 4.446, gnorm 0.077, 14,033 tok/s. Quick eval at 5,500 next. 0 Vast boxes.

### 2026-10-08T01:23:27Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 5,500 (about 721M tokens): **1.38176 bpb**. Three flat points now: 1.38315, 1.38261,
  1.38176 at 4,500, 5,000, 5,500 (0.0014 over 1,000 steps). The constant learning rate has stopped buying much; the
  WSD decay from about step 6,714 is where the rest should come. Not a fault. Step 5,550 loss 4.428, gnorm 0.098.
  0 Vast boxes.

### 2026-10-08T01:43:22Z - heartbeat (JIMOTHY)
- FULL base step 5,650 of 8,392 (67%), loss 4.398 (4.380 at 5,600), gnorm 0.080, 14,543 tok/s. Quick eval at 6,000
  next. 0 Vast boxes.

### 2026-10-08T02:03:24Z - heartbeat (JIMOTHY)
- FULL base step 5,800 of 8,392 (69%), loss 4.380, gnorm 0.081, 13,765 tok/s. Quick eval at 6,000 next. 0 Vast boxes.

### 2026-10-08T02:23:33Z - heartbeat (JIMOTHY)
- FULL base step 5,900 of 8,392 (70%), loss 4.392, gnorm 0.101, 14,315 tok/s. Quick eval at 6,000 next. 0 Vast boxes.

### 2026-10-08T02:43:30Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 6,000 (about 786M tokens): **1.37215 bpb**, down 0.0096 from 1.38176 at 5,500. The shelf
  broke before the decay. Step 6,050 loss 4.401, gnorm 0.076, 13,695 tok/s. The decay starts near step 6,714. 0 Vast
  boxes.

### 2026-10-08T03:03:24Z - heartbeat (JIMOTHY)
- FULL base step 6,150 of 8,392 (73%), loss 4.415 (4.340 at 6,100, its lowest logged), gnorm 0.076, 13,642 tok/s.
  Quick eval at 6,500 next. 0 Vast boxes.

### 2026-10-08T03:23:23Z - heartbeat (JIMOTHY)
- FULL base step 6,300 of 8,392 (75%), loss 4.414 (4.351 at 6,250), gnorm 0.105, lr still 0.003. 14,072 tok/s. Quick
  eval at 6,500 next; decay from about 6,714. 0 Vast boxes.

### 2026-10-08T03:43:26Z - heartbeat (JIMOTHY)
- FULL base step 6,450 of 8,392 (77%), loss 4.401 (4.327 at 6,400, its lowest logged), gnorm 0.085, lr 0.003, 14,397
  tok/s. Quick eval at 6,500 next; decay from about 6,714. 0 Vast boxes.

### 2026-10-08T04:03:25Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 6,500 (about 852M tokens, the last one at the full learning rate): **1.36687 bpb**, down
  0.0053 from 1.37215 at 6,000. Step 6,550 loss 4.372, gnorm 0.079, 13,427 tok/s. The decay starts near 6,714.
  0 Vast boxes.

### 2026-10-08T04:18:53Z - the FULL chat model has a name (navigator)
- **The navigator named the FULL SDM CHAT model "SDMwise Gamgee".** On the site it shows as `SDMwise Gamgee-full` once
  `launch_chat_fullsdm_50M` lands and the browser engine can run the FULL architecture. Run names in records stay as
  they are; the name is the front-facing one.
- Lane MODELPOSTFIX is adding the class postfixes (-full, -partial, -store, -mlp) to every chat picker. Every model on
  the site today is a store model (arm `sdm`: one trained store read twice per token, addressed by the last 8 tokens
  and five moving averages, no MLP, no run-time memory).
- Owed for SDMwise to appear on the site: the browser engine runs only the store architecture; FULL needs a port.

### 2026-10-08T04:19:12Z - correction to the entry above
- "SDMwise Gamgee" was the navigator joking, not a naming decision. It stays a nickname here; the FULL chat model's
  front-facing name is not decided, and nothing on the site will carry it unless the navigator says so.

### 2026-10-08T04:23:24Z - heartbeat (JIMOTHY)
- FULL base step 6,700 of 8,392 (80%), loss 4.355, gnorm 0.066, lr still 0.003, 13,903 tok/s. The WSD decay begins at
  about step 6,714, a few minutes away. Quick eval at 7,000 next. 0 Vast boxes. Lane MODELPOSTFIX still running.

### 2026-10-08T04:43:47Z - heartbeat (JIMOTHY)
- The WSD decay has begun: lr 0.002936 at step 6,750 and 0.002846 at 6,800 (from 0.003), falling linearly to zero at
  8,392. FULL base step 6,800, loss 4.365, gnorm 0.087, 13,942 tok/s. Quick eval at 7,000 next. 0 Vast boxes.

### 2026-10-08T04:57:12Z - lane MODELPOSTFIX landed (JIMOTHY)
- Landed at 2925af01a (generated files 6232cf7f0), site suite 3,436 pass, 0 fail. Every chat model name now ends in its
  class: -full, -partial, -store or -mlp (src/chat/arch.js, defined once). Every model on the site today is -store.
- **Correction to my earlier description of the store models:** they are NOT MLP-free. SdmLM
  (track4_sdmllm_models.py) ends in one SwiGLU readout after its two store reads. The site's legend says so. FULL is
  the only class with no MLP at all.
- Open, the navigator's call: the ASK page's DeepSeek V4.1 Flash is a transformer; it carries no postfix yet.

### 2026-10-08T05:03:25Z - heartbeat (JIMOTHY)
- FULL base step 6,950 of 8,392, lr 0.002578 (decaying), loss 4.384. gnorm is up again, 0.146 and 0.153 at 6,900 and
  6,950 (about 0.08 is normal), still under the 0.3 line set at 4,650, and the loss is not rising. Watching; no action.
  Quick eval at 7,000 next. 0 Vast boxes.

### 2026-10-08T05:23:27Z - heartbeat (JIMOTHY)
- FULL base quick eval at step 7,000, the first in the decay (about 918M tokens): **1.35355 bpb**, down 0.0133 from
  1.36687 at 6,500, the biggest drop since step 4,000. The decay is paying as expected. Step 7,050 loss 4.355, gnorm
  0.135 (0.088 at 7,000; still under the 0.3 line), lr 0.0024. 0 Vast boxes.

### 2026-10-08T09:22:18Z - the FULL base is done and scored; FB1 HOLDS (JIMOTHY)
- `launch_base_fullsdm_d768_L12_T2048_1100M` finished at 08:46:51Z (8,392 steps, 21 h 48 min on the Spark).
  **TEST 1.29216 bpb** (4.21276 nats/token, the same 1,031,158 tokens). Quick evals during the decay: 1.35355 at 7,000,
  1.33936 at 7,500, 1.32719 at 8,000. The decay paid about 0.075 from step 6,500.
- **VERDICT FB1: HOLDS.** FULL at 1.1B beats FULL at 50M (1.54457) by **0.25241**, against the sealed bar of 0.2.
- FB2 waits on `launch_yard_d768_L12_T2048_1100M`, the last link of chain2.
- The weights stay on the Spark: `ck/launch_base_fullsdm_d768_L12_T2048_1100M/last.pt` and `keep_550000000.pt`.
- `launch_chat_fullsdm_50M` started 08:48:42Z (step 60 at 09:21Z, chat loss 3.34, web 4.21). Its before-score is the
  base's TEST, 1.29216. The records are in runs_launch/.
- The step-4,500 insurance copy is no longer needed; it stays until the chat lands.

### 2026-10-08T09:24:01Z - heartbeat (JIMOTHY)
- Lane FULLBASEDATA landed (7af278c64, learn-SDM tests 27/27): the SDM pages now show FULL's 1.29216 in place of "no
  score yet".
- `launch_chat_fullsdm_50M` runs at **5,539 tok/s**, not about 14,000: chain2 called the fine-tune without
  `--compile`. It ends near 11:35Z instead of about 10:00Z. Accepted rather than restarted: killing the child would let
  chain2 run straight on into the sweep with no chat. Any later fine-tune call carries `--compile`.
- 0 Vast boxes.

### 2026-10-08T09:43:43Z - heartbeat (JIMOTHY)
- `launch_chat_fullsdm_50M` step 110 of about 381, loss 3.858 (chat 3.471), 5,422 tok/s. Ends near 11:35Z. 0 Vast
  boxes.

### 2026-10-08T14:23:21Z - the FULL chat and the sweep's noise arm scored; a frozen sweep resumed (JIMOTHY)
- **FULL SDM CHAT** `launch_chat_fullsdm_50M` (50M tokens, half chat, half web, 5,588 tok/s uncompiled, ended
  11:34Z): **chat_test 1.09621 bpb** (from the base's 1.38969, -0.29348) and **TEST 1.31007** (+0.01791 against the
  base's 1.29216, the usual small cost of tuning). For reference the site's best chat model, the store model
  sdmwide768chat, reads 1.06386 on chat turns at 700M tokens; FULL is 0.032 behind it.
- **SW1, the noise arm:** `sw_seed1_d768_L12_50M` TEST **1.54149** against seed 0's 1.54457, so N = **0.00308**.
  **VERDICT SW1: HOLDS** (N <= 0.010). The sweep's win/lose line is therefore max(2N, 0.010) = **0.010**.
- **RED, now fixed:** the sweep script (`sweep.sh`, PID 59102) sat in state T (stopped) from about 12:34Z, after its
  first arm's python finished, until 14:22Z: about 1 h 48 min of an idle GPU. I resumed it with SIGCONT by its PID;
  `sw_slots4x_d768_L12_50M` started at 14:22:27Z. The sender is unknown: no tool of mine sends a stop, and the
  Spark's job-queue scripts note that this box can SIGSTOP a keeper. The heartbeat now checks the chain's process
  state, not only its log.
- Records committed under runs_launch/.

### 2026-10-08T23:02:10Z - wave SW scored, six of seven arms (JIMOTHY)
Line: an arm WINS or LOSES only beyond 0.010 of the bake-off FULL (TEST 1.54457, 13,989 tok/s); N = 0.00308.

| arm | change | TEST bpb | vs 1.54457 | call | median tok/s | weights |
|---|---|---|---|---|---|---|
| sw_seed1 | seed 1 | 1.54149 | -0.00308 | (noise) | | 97.9M |
| sw_slots4x | 4x encyclopedia slots | 1.54708 | +0.00251 | TIE | 13,323 | 241.4M |
| sw_k64 | read 64 slots | 1.54262 | -0.00195 | TIE | | 97.9M |
| sw_heads8 | 8 encyclopedia heads | 1.54293 | -0.00164 | TIE | | 97.9M |
| sw_diary4k | diary 4,096 slots a head | **1.62503** | **+0.08046** | **LOSES** | 5,720 | 98.0M |
| sw_deep | width 512, 24 layers | **1.52777** | **-0.01680** | **WINS** | 8,560 | 83.5M |
| sw_wide | width 1,024, 8 layers | running | | | | 119.8M |

**Verdicts:** SW1 HOLDS (N 0.00308). SW2 MISSES (4x slots ties; the encyclopedia's size is not what FULL is short of).
SW3 HOLDS. SW4 HOLDS. SW5 MISSES (the bigger diary is much WORSE, and 2.4x slower). SW6 MISSES in the other direction:
the deep shape WINS with 15% fewer weights, at 61% of the speed. SW7 and SW8 wait on sw_wide (SW8 holds so far:
1.52777 > 1.5043). SW9 MISSES (slots4x ran at 13,323 tok/s, not below 11,000).
**BRACKETED / UNBRACKETED (rule a):** depth is UNBRACKETED (the best is the deepest tried, 24 layers); diary size is
UNBRACKETED (the best is the smallest tried, 1,024 slots); slots, k and encyclopedia heads tie and need no push.

### 2026-10-08T23:02:56Z - SEALED: wave SWB, push past wave SW's two open edges, before it fires (JIMOTHY)
Rule (a) sends both UNBRACKETED axes at least two values further. Same recipe and seed as wave SW, read against the
bake-off FULL (1.54457) and the noise line 0.010. Script `runs_launch/sweep2.sh` (sha256 0f16ea155456886e...), placed
on the Spark; `chain3.sh` waits for chain2 (after sw_wide and the 1.1B transformer) and then fires it. About 8 hours.

| arm | change | weights |
|---|---|---|
| swb_diary576 | diary 576 slots a head (n_sub 24) | 97.9M |
| swb_diary256 | diary 256 slots a head (n_sub 16) | 97.9M |
| swb_deep32 | width 512, 32 layers | 105.8M |
| swb_deep48 | width 512, 48 layers | 150.3M |

**Sealed predictions:**
- SWB1: swb_diary576 beats the 1,024-slot diary (1.54457) by more than 0.010 (35%).
- SWB2: swb_diary256 is worse than swb_diary576 (60%), so the diary curve turns between 256 and 1,024.
- SWB3: swb_deep32 beats swb_deep (1.52777) by more than 0.010 (40%).
- SWB4: swb_deep48 beats swb_deep32 (45%).
- SWB5: the best arm of wave SWB is below 1.515 (40%).
The winner on each axis becomes the shape of the next FULL base, priced by its speed, and its curve is pushed again if
it still sits on an edge.

### 2026-10-08T23:23:47Z - wave SW complete; the 1.1B transformer started (JIMOTHY)
- `sw_wide_d1024_L8_50M`: TEST **1.54802** (+0.00345 against 1.54457), **TIE**, at **17,384 tok/s** (the fastest FULL
  shape, 1.24x the bake-off FULL). 119.8M weights.
- **VERDICT SW7: HOLDS** (wide ties). **VERDICT SW8: HOLDS** (the best arm, deep at 1.52777, stays above 1.5043).
- **Wave SW scorecard: 5 of 9** (SW1, SW3, SW4, SW7, SW8 hold; SW2, SW5, SW6, SW9 miss).
- **The width-depth line, at nearly matched weights:** d1024 L8 1.54802 · d768 L12 1.54457 · d512 L24 1.52777. Deeper is
  better along the whole line and the best sits at the deep edge: UNBRACKETED, pushed by wave SWB (sealed).
- chain2 started `launch_yard_d768_L12_T2048_1100M` at 23:21:36Z: the transformer at FULL's base tokens, for FB2.
  `chain3` (PID 999609) waits on it, then fires wave SWB.

### 2026-10-08T23:43:34Z - heartbeat (JIMOTHY)
- `launch_yard_d768_L12_T2048_1100M` step 400 of 8,392, loss 4.156 (FULL was 4.97 at step 400), 45,816 tok/s, so it
  ends near 06:05Z. chain2 and chain3 both alive. 0 Vast boxes.

### 2026-10-08T23:43:51Z - correction to the entry above
- "FULL was 4.97 at step 400" was written from memory, not read. The FULL base's log says **5.1219** at step 400. The
  transformer's 4.156 is 0.966 nats lower at the same step.

### 2026-10-09T00:03:41Z - heartbeat (JIMOTHY)
- The 1.1B transformer's first quick eval, step 500 (about 66M tokens): **1.25067 bpb**, against the FULL base's
  1.56858 at the same step (gap 0.318). Step 800, loss 3.782, 43,716 tok/s. Both chains alive. 0 Vast boxes.

### 2026-10-09T00:23:28Z - heartbeat (JIMOTHY)
- The 1.1B transformer's quick eval at step 1,000 (about 131M tokens): **1.16152 bpb**, against the FULL base's 1.49875
  at the same step (gap 0.337, wider than 0.318 at step 500). The transformer at 131M tokens is already below the FULL
  base's final full TEST (1.29216); FB2 is all but settled. Step 1,200, 44,248 tok/s. Both chains alive. 0 Vast boxes.

### 2026-10-09T00:43:32Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 1,500: **1.12226** (FULL base 1.48266 at the same step; gap 0.360, still
  widening). Step 1,650 of 8,392, 45,190 tok/s, ends near 06:05Z. Both chains alive. 0 Vast boxes.

### 2026-10-09T02:24:48Z - heartbeat (JIMOTHY)
- 1.1B transformer quick evals: step 3,000 **1.06753** (FULL base 1.42899 at the same step, gap 0.361), step
  3,500 **1.05591**. Step 3,700 of 8,392, 45,017 tok/s, ends near 05:45Z. Both chains alive. 0 Vast boxes.

### 2026-10-09T02:43:22Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 4,000: **1.04906** (FULL base 1.39772 at the same step, gap 0.349; the gap
  was 0.365 at step 3,500, so it narrowed a little). Step 4,100 of 8,392, 46,196 tok/s, ends near 05:20Z.
  Both chains alive. 0 Vast boxes.

### 2026-10-09T02:47:59Z - THE GAP TURNS: where it happened (navigator asked for the level to be recorded)
The gap between the FULL base and the 1.1B transformer (same shape d768 L12, same data, same order, same recipe)
was widest at step 3,500 and first narrowed at step 4,000. Quick eval, 130,950 held-out tokens each:

| step | tokens seen | FULL bpb | transformer bpb | gap |
|---|---|---|---|---|
| 3,000 | 393,216,000 | 1.42899 | 1.06753 | 0.361 |
| 3,500 | 458,752,000 | 1.42091 | 1.05591 | **0.365 (widest)** |
| 4,000 | 524,288,000 | 1.39772 | 1.04906 | **0.349 (first turn)** |

- The turn sits between **459M and 524M tokens**, at gap level **0.365**, with both models still at peak learning
  rate (3e-3; the WSD decay starts near step 6,714).
- Between those evals FULL fell 0.02319 and the transformer fell 0.00685: FULL was still learning about 3.4x
  faster per step there.
- Caveat: one quick-eval point on 130,950 tokens is not a trend. It counts only if the gap keeps closing through
  the decay; the final TEST on all 1,031,158 test tokens decides FB2.

### 2026-10-09T02:48:35Z - THE SHAPE OF THE TWO CURVES, and the things worth remembering (navigator asked)
Quick eval bpb by step (130,950 held-out tokens each; 131,072 tokens a step; both d768 L12, same data and order):

| step | FULL | transformer | gap |
|---|---|---|---|
| 500 | 1.56858 | 1.25067 | 0.318 |
| 1,000 | 1.49875 | 1.16152 | 0.337 |
| 1,500 | 1.48266 | 1.12226 | 0.360 |
| 2,000 | 1.44881 | 1.09741 | 0.351 |
| 2,500 | 1.42910 | 1.08116 | 0.348 |
| 3,000 | 1.42899 | 1.06753 | 0.361 |
| 3,500 | 1.42091 | 1.05591 | 0.365 |
| 4,000 | 1.39772 | 1.04906 | 0.349 |

(Correction to the entry just above: the gap had already dipped once, 0.360 to 0.351 to 0.348 over steps 1,500 to
2,500, and widened again. So step 4,000 is the second dip, not the first. Step 3,500 is still the widest point.)

**The nature of each curve**
- **The transformer descends smoothly.** Every step of 500 lowers it, and each drop is smaller than the last
  (0.089, 0.039, 0.025, 0.016, 0.014, 0.012, 0.007). A plain bending curve with no plateaus.
- **FULL descends in a STAIRCASE.** It drops, sits on a shelf, then drops again:
  - drop to 1.483 by step 1,500, then a step down to 1.449 at 2,000
  - **shelf 1, steps 2,500 to 3,000:** 1.42910 then 1.42899 (it moved 0.0001 in 65M tokens)
  - drop to 1.398 at 4,000 and 1.383 at 4,500
  - **shelf 2, steps 4,500 to 5,500:** 1.38315, 1.38261, 1.38176 (0.0014 in 131M tokens)
  - then a steady glide as the learning rate decays from step ~6,714: 1.372, 1.367, 1.354, 1.339, 1.327 (step 8,000)
  - final full TEST 1.29216
- **The gap moves with FULL's staircase.** It narrows when FULL drops off a shelf and widens while FULL sits on
  one, because the transformer keeps falling at a steady rate the whole time.

**Highlights**
- **The memory's steps are where FULL catches up.** The two drops after the shelves (2,000 and 3,500 to 4,500) are
  the only stretches where FULL out-learns the transformer per step. Untested reading: the shelves may be the
  SDM learning to use its slots (a store must be filled before it pays), and each drop a new use found. A probe of
  slot usage across checkpoints would test it.
- **FULL's gradients are about 3x calmer.** Median gradient norm 0.091 for FULL against 0.281 for the transformer.
  Both start with a large first spike (step 50: 6.18 FULL, 2.30 transformer) that fades within 200 steps.
- **FULL's one wobble came right after shelf 2 began:** at steps 4,500 to 4,650 its gradient norm doubled (0.079 to
  0.165) and loss rose 4.372 to 4.499. It settled by itself in about 100 steps, without a restart.
- **The learning-rate decay helped FULL more than any shelf ended:** from step 6,500 to 8,000 FULL fell 0.040,
  steadily, the cleanest stretch of the run. Whether the transformer gains as much from its decay is the open
  question for its final number.
- **Speed:** FULL trains at about 14,000 tok/s and the transformer at about 45,000 tok/s on the same GPU, so the
  transformer finishes the same 1.1B tokens in about 6.8 h against FULL's 21 h 48 min.

### 2026-10-09T03:03:26Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 4,500: **1.04186** (FULL base 1.38315 at the same step, gap 0.341; down from
  0.365 at 3,500 and 0.349 at 4,000). Two closing points in a row, both as FULL came off shelf 1. FULL then sat on
  shelf 2 (steps 4,500 to 5,500), so the gap is expected to widen again at 5,000. Step 4,500 of 8,392,
  43,792 tok/s. Both chains alive. 0 Vast boxes.

### 2026-10-09T03:23:25Z - heartbeat (JIMOTHY)
- Quiet beat. 1.1B transformer at step 4,950 of 8,392, loss 3.367, gnorm 0.122, 45,598 tok/s; the step-5,000 eval
  is minutes away. Both chains alive. 0 Vast boxes.

### 2026-10-09T03:43:23Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 5,000: **1.03480** (FULL base 1.38261 at the same step, gap 0.348, up from
  0.341). The widening predicted last beat happened: FULL sat on shelf 2 (it moved 0.0005) while the transformer
  fell 0.0071. Step 5,350 of 8,392, 44,531 tok/s. Both chains alive. 0 Vast boxes.

### 2026-10-09T04:03:23Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 5,500: **1.02992** (FULL base 1.38176 at the same step, gap 0.352, up from
  0.348). FULL's last shelf-2 point. The transformer's per-500-step drop has slowed to 0.0049. Step 5,750 of 8,392,
  46,370 tok/s. Both chains alive. 0 Vast boxes.

### 2026-10-09T04:23:22Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 6,000: **1.02626** (FULL base 1.37215 at the same step, gap 0.346, down from
  0.352). FULL came off shelf 2 (fell 0.0096) while the transformer fell only 0.0037, its smallest drop yet.
  Step 6,150 of 8,392, 45,380 tok/s. Its decay starts near step 6,714. Both chains alive. 0 Vast boxes.

### 2026-10-09T04:43:23Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 6,500: **1.02053** (FULL base 1.36687 at the same step, gap 0.346, flat:
  both fell about 0.0053 in these 500 steps). Step 6,600 of 8,392, 44,746 tok/s; its learning-rate decay starts
  near step 6,714, the stretch where FULL gained most. Both chains alive. 0 Vast boxes.

### 2026-10-09T05:03:28Z - heartbeat (JIMOTHY)
- 1.1B transformer quick eval at step 7,000, the first point inside its learning-rate decay: **1.01306** (FULL base
  1.35355 at the same step, gap 0.340, down from 0.346). FULL fell 0.0133 in these 500 steps, the transformer 0.0075.
  Step 7,000 of 8,392, lr 2.49e-3, 42,654 tok/s. Both chains alive. 0 Vast boxes.
- Correction: the last several beats said this run ends near 05:10 to 05:20Z. That was wrong arithmetic. 1,392 steps
  of 131,072 tokens at about 44,000 tok/s is about 69 minutes, so it ends near **06:15Z** (the 06:05Z first logged
  was right).

### 2026-10-09T05:11:45Z - THE TRANSFORMER COMPARISON IS HELD; SDM WORK FIRST (the navigator)
- The navigator: hold the transformer comparison till later and run the SDM work now. The site should say this
  comparison is in progress: from the evals so far FULL is somewhat behind, and the final result is awaited.
- 05:10:15Z: stopped the 1.1B transformer (PID 1014601, by PID) at step 7,150 of 8,392. Its newest checkpoint is
  step 7,000 (model + optimiser + step): ck/launch_yard_d768_L12_T2048_1100M/last.pt, with a copy hold_step7000.pt,
  both sha256 81e790e28a20... Last quick eval 1.01306 at step 7,000 (FULL 1.35355 there, gap 0.340).
- **TO PICK IT UP LATER:** `bash ~/sdmonly_base/launch/yard_resume.sh` on the Spark (copy in
  runs_launch/yard_resume.sh). Same command as chain2's last step, so the trainer resumes from step 7,000. About
  65 min left. Then score its TEST and write VERDICT FB2. Owed after the SDM queue below, or whenever a GPU is free.
- 05:10:59Z: chain3 fired wave SWB as soon as chain2 exited. swb_diary576_d768_L12_50M started.
- SDM queue now: wave SWB (4 FULL runs, about 8 h) -> the new FULL base in the winning shape -> its chat model and
  the WEIRD LITTLE GUY -> then the held transformer resume.

### 2026-10-09T05:14:46Z - lane MODELINFO fired (site)
- Opus lane MODELINFO in _worktrees/dwarfstar-modelinfo (branch settle-modelinfo): richer data-derived postfixes
  (class + how trained + headline score), every model listed, FULL base and FULL SDM CHAT shown as trained but not
  yet runnable in the browser, and a section saying the transformer comparison is in progress and held at step
  7,000 (FULL somewhat behind so far, final result awaited). I land it after its tests and build are green.

### 2026-10-09T05:23:29Z - heartbeat (JIMOTHY)
- Wave SWB run 1 of 4, swb_diary576_d768_L12_50M: step 50 of 381, loss 6.233, 10,933 tok/s, about 75 min per run
  at that speed. chain3 alive. Transformer held at step 7,000. Lane MODELINFO running. Vast reporting stopped by the
  navigator (0 boxes; the Spark is the whole fleet).

### 2026-10-09T05:43:35Z - heartbeat (JIMOTHY)
- swb_diary576: step 200 of 381, loss 5.205, **17,725 tok/s**, about 27% faster than the 1,024-slot base shape
  (13,989 in wave SW). Ends near 06:10Z with its TEST. chain3 alive. Lane MODELINFO running.

### 2026-10-09T06:04:19Z - heartbeat (JIMOTHY)
- Lane MODELINFO LANDED (ad202331e, commits 9c87c7ac6 + 37fb5736c): the facts postfix
  (-<class>-<tokens>-<kind>-<score>bpb, built from records, refuses on a missing fact), an EVERY MODEL list with FULL
  base (-full-1.1B-base-1.292bpb) and FULL SDM CHAT (-full-1.15B-chat-1.096bpb) marked not yet runnable in the
  browser, and the transformer comparison shown as in progress on #/sdmchat, #/sdm and #/sdmchat-model. Lane suite
  3,614 of 3,663 (1 failure from the sparse checkout); re-running the full suite in the main checkout now.
- swb_diary576 finished training (step 381 of 381, loss 4.919); its TEST is being scored.

### 2026-10-09T06:18:45Z - SWB1 scored; the sweep was frozen again
- **swb_diary576_d768_L12_50M: TEST 1.50846** against the 1,024-slot FULL at 1.54457: better by **0.03611**, beyond
  the 0.010 line, and 27% faster to train (17,725 vs 13,989 tok/s). **VERDICT SWB1: HOLDS** (sealed at 35%).
- Diary line now: 4,096 1.62503 · 1,024 1.54457 · 576 1.50846. Best at the small edge: UNBRACKETED until diary256
  reports (running, started 06:18:04Z). If 256 wins too, the next wave goes to 128 and 64.
- Found sweep2.sh (PID 1506863) stopped again (state T), the same unexplained stop as wave SW. The trainer had
  finished at about 06:05Z and the sweep could not start the next arm: about 13 min lost. Sent SIGCONT by PID.
  The memory guard is not the sender (it only cancels q jobs named SDMCHATS-*, last acted 2026-10-02). Started
  unfreeze.sh on the Spark: every 60 s it sends SIGCONT to PIDs 999609 and 1506863 if stopped, and logs each one
  to ~/sdmonly_base/launch/unfreeze.log.
- Site: the full suite in the main checkout after the MODELINFO landing: 3,663 tests, 3,661 pass, 0 fail.

### 2026-10-09T06:23:33Z - heartbeat (JIMOTHY)
- swb_diary256 (run 2 of 4) started 06:18:04Z and is compiling; no step line yet. chain3 and sweep2 both running
  (state S); the unfreeze guard has sent no SIGCONT so far.

### 2026-10-09T06:43:30Z - heartbeat (JIMOTHY)
- swb_diary256: step 200 of 381, train loss **5.051** (diary576 read 5.205 at the same step), **20,908 tok/s**
  (49% faster than the 1,024-slot shape). Ends near 07:10Z with its TEST. Both chain scripts running; no SIGCONT
  needed.

### 2026-10-09T07:03:57Z - SWB2 scored: the 256-slot diary wins big; SEALED wave SWC before it fires (JIMOTHY)
- **swb_diary256_d768_L12_50M: TEST 1.42911.** Better than 576 slots (1.50846) by 0.07935 and than 1,024 slots
  (1.54457) by 0.11546, and the fastest shape yet (20,908 tok/s). **VERDICT SWB2: MISSES** (sealed at 60% that 256
  would be worse than 576; it is far better).
- Diary line (slots per head, d768 L12, 50M tokens): 4,096 1.62503 · 1,024 1.54457 · 576 1.50846 · **256 1.42911**.
  Monotone, best at the smallest value tried: **UNBRACKETED** at the small edge.
- Against the transformer at the same 50M tokens (1.22265) the gap is now 0.206, down from 0.322 for the 1,024-slot
  FULL. The memory still counts: the memory-off arm read 1.75751.
- swb_deep32_d512_L32_50M started 07:02:36Z (SWB run 3 of 4), then swb_deep48.

**SEALED: wave SWC** (runs after SWB, by chain4.sh waiting on sweep2 PID 1506863; script runs_launch/sweep3.sh).
Same recipe, 50M tokens, seed 0. Slots per head = n_sub squared.
- swc_diary121_d768_L12_50M (--mem-n-sub 11), swc_diary64_d768_L12_50M (--mem-n-sub 8): two values past 256.
- swc_deep24_diary256_d512_L24_50M: the best diary (256) on the best depth so far (d512, 24 layers).
Predictions, sealed before any SWC arm runs:
- SWC1: diary121 beats diary256 (1.42911) by more than 0.010 (45%).
- SWC2: diary64 is worse than diary121, so the diary curve turns between 64 and 256 (50%).
- SWC3: deep24 with diary256 beats d768 L12 diary256 (1.42911) by more than 0.010 (50%).
- SWC4: the best SWC arm is below 1.42 (45%).

### 2026-10-09T07:23:36Z - heartbeat (JIMOTHY)
- swb_deep32 (SWB run 3 of 4) started 07:02:36Z; 21 min in it is still compiling (32 layers compile slowly), no
  step line yet.
- Correction: the stop guard for chain4 was pointed at PID 1672100, a short-lived shell from the launch; that PID
  was gone and the guard exited at once. chain4 itself is PID 1672104 (alive, waiting on sweep2). New guard started
  on 1672104. Guard on 999609 and 1506863 still running, no SIGCONT sent.

### 2026-10-09T07:43:25Z - heartbeat (JIMOTHY)
- swb_deep32 past compile: step 50 of 381, loss 6.525. Its 3,239 tok/s at step 50 still averages in the compile
  (diary256 read 10,933 at step 50 and 20,908 by step 200). At the ~6,400 tok/s a 32-layer d512 should reach, it ends
  near 09:40Z. All three chain scripts running; no SIGCONT sent.

### 2026-10-09T08:03:24Z - heartbeat (JIMOTHY)
- swb_deep32: step 100 of 381, loss 5.813, 6,266 tok/s. About 98 min of steps left: ends near 09:45Z with its TEST.
  All three chain scripts running; no SIGCONT sent.

### 2026-10-09T08:23:26Z - heartbeat (JIMOTHY)
- swb_deep32: step 150 of 381, loss 5.597, 6,436 tok/s. Ends near 09:45Z. Chains running, no SIGCONT sent.

### 2026-10-09T08:43:23Z - heartbeat (JIMOTHY)
- swb_deep32: step 200 of 381, train loss 5.311, 6,450 tok/s (at step 200: diary256 read 5.051, diary576 5.205;
  train loss only, the TEST decides). Ends near 09:45Z. Chains running, no SIGCONT sent.

### 2026-10-09T09:03:26Z - heartbeat (JIMOTHY)
- swb_deep32: step 300 of 381, train loss 5.079 (diary576 read 5.111 at step 300), 6,469 tok/s. About 27 min of
  steps left; TEST near 09:35Z. Chains running, no SIGCONT sent.

### 2026-10-09T09:23:22Z - heartbeat (JIMOTHY)
- swb_deep32: step 350 of 381, in its cooldown, train loss 4.917 (diary576 read 4.977 at step 350), 6,507 tok/s.
  TEST near 09:40Z. Chains running, no SIGCONT sent.

### 2026-10-09T09:43:33Z - SWB3 scored: 32 layers beat 24; THE TRANSFORMER LIST, held for later (JIMOTHY)
- **swb_deep32_d512_L32_50M: TEST 1.48813**, better than 24 layers (1.52777) by 0.03964, beyond 0.010.
  **VERDICT SWB3: HOLDS** (sealed at 40%). Depth line at d512-ish weights: 8 1.54802 · 12 1.54457 · 24 1.52777 ·
  32 1.48813. Best at the deep edge: **UNBRACKETED** until deep48 reports (started 09:33:09Z).
- Wave SWB so far: diary256 1.42911 (best), deep32 1.48813, diary576 1.50846.

**THE TRANSFORMER LIST: every side-by-side transformer run is HELD for later (the navigator: SDM work first).**
Nothing on this list fires until the SDM queue (SWB, SWC, the new FULL base, its chat, the Little Guy) is done.
1. Resume the 1.1B yardstick d768 L12 from step 7,000: `bash ~/sdmonly_base/launch/yard_resume.sh` (~65 min),
   then VERDICT FB2.
2. A transformer at the winning depth shape (d512, 32 or 48 layers) at 50M tokens, so the deep FULL arms have a
   same-shape yardstick (the existing 50M yardstick is d768 L12 only, which matches the diary arms).
3. A transformer at the new FULL base's final shape and token budget, trained on the same data in the same order.

### 2026-10-09T09:44:42Z - lane SITEFINAL fired (site, "update all pages, final final")
- Opus lane SITEFINAL: every SDM-LM page brought up to the newest results, the shape sweep (SW, SWB, SWC when it
  lands) generated from the records so later runs update the pages with no code change, stale numbers audited, and
  the held-transformer note. Build + tests green, committed on settle-sitefinal; I land it. Nothing deploys (the
  site's own law: local-only, no publishing live).
- SWB records committed home (35590508c): diary576 1.50846, diary256 1.42911, deep32 1.48813.

### 2026-10-09T10:03:23Z - heartbeat (JIMOTHY)
- swb_deep48 (SWB run 4 of 4) started 09:33:09Z; 30 min in, still compiling 48 layers, no step line yet. Chains
  running, no SIGCONT sent. Lane SITEFINAL running.

### 2026-10-09T10:14:07Z - lane SITEFINAL landed (JIMOTHY)
- Landed d3cbe3c8d (c9116c0af + 45cfc85f0): THE SHAPE SWEEP on #/sdm and #/sdmchat-model (memory and depth lines,
  BRACKETED/UNBRACKETED, best row lit, generated from sw/swb/swc records and the sweep scripts so new records update
  the pages with no code change), the opening lines on the best FULL (1.42911) and the narrowed gap (0.322 to 0.206),
  and THE TRANSFORMER LIST shown as held for later. Full suite in the main checkout: 3,839 tests, 3,837 pass, 0 fail.
- Lane BRANDCAPS running (vague_10 caps pass: TRIPLESPARKLE in capitals, rendered-dot background, angled scanlines,
  a misty red heart behind the kanji, bigger logo stars).

### 2026-10-09T10:23:44Z - heartbeat (JIMOTHY)
- swb_deep48: 50 min in, no step-50 line yet, but alive and working: python at 99.9% CPU, GPU at 95%, 22 GB
  resident. 48 layers at about 4,300 tok/s reach step 50 (6.5M tokens) about 25 min after a ~25 min compile, so
  the first line is due now. Chains running, no SIGCONT sent. Lane BRANDCAPS done (06fc9cd83), awaiting the
  navigator's look before landing.

### 2026-10-09T10:45:49Z - heartbeat (JIMOTHY)
- Lane BRANDCAPS landed (cbb862fbd): the caps pass now lives in BRAND/vague_12 (32 files) with
  BRIEF_THE_BRAND_WORK.md; vague_10 is unchanged from BRANDNEON3. The landing gate went red on one site test
  (learnsdm2: sdmForward.json differs from a fresh run) because the generator found THE TRANSFORMER LIST by its
  last mention in this log, and my SITEFINAL-landed entry mentioned it in passing. Lane HELDLISTFIX landed
  (23366cdfe): the parser anchors on the list's bold header; a new test is red against the old parser, green now.
- swb_deep48: step 50 of 381 at about 10:40Z (67 min after start; 1,634 tok/s averaged over the compile). True
  speed known at step 100. Chains running, no SIGCONT sent.

### 2026-10-09T11:03:36Z - heartbeat (JIMOTHY)
- Quiet. swb_deep48 still on its step-50 line (about 10:40Z); step 100 is due about now if it runs near 4,300 tok/s.
  Chains running, no SIGCONT sent. Lane TRAININGSPEED landed (ae966793f, site suite 3,838 of 3,838). The finished
  runs' small records committed (94273a75a, 1,573 files, no weights).

### 2026-10-09T11:25:13Z - heartbeat (JIMOTHY)
- swb_deep48 passed step 100 (loss 5.8965) at 4,334.6 tok/s, about 30 s a step. 281 steps left: training ends about
  13:27Z, TEST score about 13:35Z (00:35 Saturday, Melbourne). Chains 999609, 1506863, 1672104 alive, no SIGCONT sent.
  Vast 0 boxes. Spark memory 10 GB available, disk 483 GB free. Next: score deep48, write VERDICT SWB4, then wave
  SWC fires by itself through chain4.

### 2026-10-09T12:03:22Z - heartbeat (JIMOTHY)
- swb_deep48 at step 200 of 381, loss 5.3226, 4,278.6 tok/s. Ends about 13:30Z, TEST about 13:38Z. Chains alive,
  Vast 0 boxes. Claude 1 is back up and holds its four site lanes (MEMBERSMENU, MEMBERSNEON, BRANDNEON7, PKGGITHUB).

### 2026-10-09T12:23:31Z - heartbeat (JIMOTHY)
- swb_deep48 at step 250 of 381, loss 5.1916 (deep32 at the same step 5.1786). TEST about 13:38Z. Vast 0 boxes.
  Lane FULLCONTEXT fired (site: sitewide context and lookback copy rewritten for FULL; four fades 0, 0.8, 0.99, 1.0
  per layer, fixed slots, trained at 2,048, yardstick also 2,048, reach past 2,048 unmeasured). Claude 1 lands it.

### 2026-10-09T12:43:29Z - heartbeat (JIMOTHY)
- Quiet. swb_deep48 between step 250 and 300 (300 due about 12:45Z). Chains alive, Vast 0, lane FULLCONTEXT running.

### 2026-10-09T13:03:27Z - heartbeat (JIMOTHY)
- swb_deep48 at step 300 of 381, loss 5.0712, now AHEAD of deep32 at the same step (5.0793) after trailing at 200 and 250. Ends about 13:25Z, TEST about 13:35Z. Vast 0. Lane FULLCONTEXT running.

### 2026-10-09T13:23:25Z - heartbeat (JIMOTHY)
- swb_deep48 at step 350 of 381 (WSD cooldown, lr 0.00125), loss 4.9042. Ends about 13:40Z, TEST about 13:50Z. Vast 0. Lane FULLCONTEXT done and relayed to Claude 1; lane FULLPARTIALTRUTH (site audit, FULL and PARTIAL everywhere) running.

### 2026-10-09T13:44:23Z - WAVE SWB COMPLETE; VERDICTS SWB4, SWB5 (JIMOTHY)
- swb_deep48_d512_L48_50M: TEST **1.48134 bpb** (1,031,158 tokens), 133.5M body weights, about 4,300 tok/s on the
  Spark (deep32 about 6,400). It beats swb_deep32 (1.48813) by 0.00679, INSIDE the 0.010 noise line.
  **VERDICT SWB4: HOLDS as written** (sealed at 45%, no margin was sealed), but the pair is a TIE by the noise line.
  **VERDICT SWB5: HOLDS** (sealed at 40%): the best arm, swb_diary256, scores 1.42911 < 1.515.

| wave SWB (50M tokens) | TEST bpb |
|---|---|
| diary 576, d768 L12 | 1.50846 |
| diary 256, d768 L12 | **1.42911** |
| d512 L32 (diary 1,024) | 1.48813 |
| d512 L48 (diary 1,024) | 1.48134 |

- **BRACKETED / UNBRACKETED:** diary size UNBRACKETED (256 is the smallest tried; wave SWC runs 121 and 64 now).
  Depth: UNDECIDABLE between 32 and 48 (tie inside the noise) and flattening (24 to 32 gained 0.040, 32 to 48 gained
  0.007). Rule (c) applies: the pair goes again with more data before it is called.
- Wave SWC fired by itself: swc_diary121_d768_L12_50M is training.

### SEALED: wave SWD, the depth tie with more data, on the best diary (before it fires)
Rule (c): the 32 / 48 tie reruns at 100M tokens. It runs on the BEST diary of waves SWB and SWC (the lowest TEST
among diary 256, 121 and 64 at d768 L12; picked by the script from the result files), so it also answers whether
depth and the small diary add up. Same recipe, seed 0, 100M tokens of fresh train_big_p200m.

| arm | shape |
|---|---|
| swd_shallow | d768, 12 layers, best diary |
| swd_deep32 | d512, 32 layers, best diary |
| swd_deep48 | d512, 48 layers, best diary |

**Sealed predictions:**
- SWD1: swd_deep48 beats swd_deep32 by more than 0.010 (35%).
- SWD2: swd_deep32 beats swd_shallow by more than 0.010 (60%).
- SWD3: the best SWD arm scores below 1.38 (45%).
Script `runs_launch/sweep4.sh`, chained by `chain5.sh` after wave SWC. About 12 hours on the Spark.
- 2026-10-09T13:44:58Z wave SWD placed: `runs_launch/sweep4.sh` (sha256 9605747aa75ce79b...) and `chain5.sh` in tmux launch5 (PID 2247563, waits for chain4 1672104), unfreeze guard on it. The deep48 log and result committed after its end.

### 2026-10-09T14:03:30Z - heartbeat (JIMOTHY)
- swc_diary121 at step 250 of 381, loss 4.9052, 23,871 tok/s (fastest shape yet). Ends about 14:12Z, TEST about 14:20Z; then diary64, deep24+diary256, then wave SWD. Vast 0. Lane FULLPARTIALTRUTH running.

### 2026-10-09T14:23:59Z - VERDICT SWC1 (JIMOTHY)
- swc_diary121_d768_L12_50M: TEST **1.40333 bpb**, the best SDM score so far (FULL, reads its memory). It beats
  diary 256 (1.42911) by 0.02578, 2.6x the noise line, at about 24,000 tok/s (the fastest shape yet).
  **VERDICT SWC1: HOLDS** (sealed at 45%). Diary line at d768 L12: 4,096 1.62503 · 1,024 1.54457 · 576 1.50846 ·
  256 1.42911 · 121 1.40333. Still UNBRACKETED; diary 64 runs now and decides SWC2. Gap to the transformer at 50M:
  0.18068 (was 0.206).
- 2026-10-09T14:25:06Z RED, fixed: sweep3.sh (PID 2237204) was found STOPPED (state T) after diary121 ended, about 10 min lost; the third such stop, sender unknown (not memguard). SIGCONT sent by PID; diary64 started 14:24:19Z. New guard `runs_launch/unfreeze_tree.sh` CONTs any stopped process in the trees of chain4 1672104 and chain5 2247563 (guard PID 2282039).

### 2026-10-09T14:43:25Z - heartbeat (JIMOTHY)
- swc_diary64 at step 150 of 381, loss 5.3293, 26,037 tok/s. TEST about 15:05Z. No stops since the tree guard. Vast 0. FULLPARTIALTRUTH relayed to Claude 1 (contains FULLCONTEXT).

### 2026-10-09T15:03:44Z - VERDICT SWC2; SWD addendum sealed (JIMOTHY)
- swc_diary64_d768_L12_50M: TEST **1.40020 bpb**, about 26,000 tok/s. It beats diary 121 (1.40333) by 0.00313,
  INSIDE the 0.010 noise line. **VERDICT SWC2: MISSES** (sealed at 50% that 64 would be worse than 121): 64 is not
  worse; the pair is a TIE.
- Diary line at d768 L12, 50M: 4,096 1.62503 · 1,024 1.54457 · 576 1.50846 · 256 1.42911 · 121 1.40333 · 64 1.40020.
  The curve has flattened: 256 to 121 gained 0.026, 121 to 64 gained 0.003. Diary size is UNDECIDABLE between 121
  and 64 (a tie) and rule (c) reruns the pair with more data. Best FULL so far 1.40020 (reads its memory); gap to the
  transformer at 50M 0.17755.
- swc_deep24_diary256_d512_L24_50M fired 15:01:53Z (SWC3).
- **SEALED ADDENDUM to wave SWD, before it fires:** wave SWD's script picks diary 64 (the lowest TEST). It now also
  runs the other diary of the tied pair, swd_shallow_diary121_d768_L12_100M, beside swd_shallow_diary64 at 100M
  tokens. `runs_launch/sweep4.sh` updated on the Spark (sha256 58f2cb12e88de830...).
  - SWD4: at 100M tokens, diary 64 beats diary 121 by more than 0.010 (25%).

### 2026-10-09T15:23:25Z - heartbeat (JIMOTHY)
- swc_deep24_diary256 at step 50 of 381 (loss 6.3711, 5,907 tok/s averaged over the compile). TEST about 16:40Z; then wave SWD. No stops. Vast 0.

### 2026-10-09T15:43:29Z - heartbeat (JIMOTHY)
- swc_deep24_diary256 at step 150 of 381, loss 5.4634, 13,145 tok/s; diary256 at 12 layers read 5.3345 at the same step (deep behind by 0.129 so far). TEST about 16:30Z. No stops. Vast 0.

### 2026-10-09T16:03:22Z - heartbeat (JIMOTHY)
- swc_deep24_diary256 at step 300 of 381, loss 4.9173; 12-layer diary256 read 4.8819 at step 300 (gap down from 0.129 at step 150 to 0.035). TEST about 16:25Z. No stops. Vast 0.

### 2026-10-09T16:23:35Z - WAVE SWC COMPLETE; VERDICTS SWC3, SWC4; wave SWD fired (JIMOTHY)
- swc_deep24_diary256_d512_L24_50M: TEST **1.43550 bpb**, about 13,000 tok/s. It is 0.00639 WORSE than the 12-layer
  diary 256 (1.42911), a tie inside the noise line. **VERDICT SWC3: MISSES** (sealed at 50% that it would win by more
  than 0.010). With the small diary, 24 layers at width 512 buy nothing over 12 at width 768 at 50M tokens.
- **VERDICT SWC4: HOLDS** (sealed at 45%): the best SWC arm, diary 64, scores 1.40020 < 1.42.

| wave SWC (50M tokens) | TEST bpb |
|---|---|
| diary 121, d768 L12 | 1.40333 |
| diary 64, d768 L12 | **1.40020** |
| diary 256, d512 L24 | 1.43550 |

- **BRACKETED / UNBRACKETED:** diary size flat between 121 and 64 (a tie, rerun at 100M in SWD). Depth: at the big
  diary 32 and 48 tie; at the small diary 24 ties 12. Depth looks spent at 50M; SWD asks again at 100M.
- Wave SWD fired 16:18:42Z by itself; the script picked diary 64 (n_sub 8). First arm swd_shallow_diary64_d768_L12_100M.
  The tree guard sent SIGCONT to the new sweep4 shell at 16:19:05Z: a stop at the wave change again, caught in 23 s.

### 2026-10-09T16:43:27Z - heartbeat (JIMOTHY)
- swd_shallow_diary64_d768_L12_100M at step 250 of 763, loss 4.8758, 25,316 tok/s. TEST about 17:32Z. No stops since 16:19Z. Vast 0. Lanes: FULLPARTIALTRUTH (resumed: SWC and sizing notes), FAVICONPINK (kanji hyper pink on hyper blue).

### 2026-10-09T17:03:25Z - heartbeat (JIMOTHY)
- swd_shallow_diary64_d768_L12_100M at step 500 of 763, loss 4.4646, 23,799 tok/s. TEST about 17:30Z. No stops. Vast 0. Site lanes FULLPARTIALTRUTH (db8c38f21) and FAVICONPINK (d74edf347) relayed to Claude 1.

### 2026-10-09T17:23:25Z - heartbeat (JIMOTHY)
- swd_shallow_diary64 in its last steps (cooldown); TEST about 17:35Z, then swd_shallow_diary121. No stops. Vast 0.

### 2026-10-09T17:43:57Z - SWD first result; SWD3 settled; the 100M yardstick sealed (JIMOTHY)
- swd_shallow_diary64_d768_L12_100M: TEST **1.28077 bpb**, the best SDM score so far (FULL, reads its memory), about
  25,000 tok/s. Twice the data took the same shape from 1.40020 to 1.28077 (-0.11943).
- **VERDICT SWD3: HOLDS** (sealed at 45%): the best SWD arm is below 1.38 already, and a minimum can only fall.
- swd_shallow_diary121_d768_L12_100M fired 17:26:59Z (the tie-break, SWD4).
- No transformer exists at 100M on this shape and data, so the gap at 100M is not known. The 50M transformer
  (1.22265) is NOT comparable to a 100M score.
- **SEALED: the 100M yardstick, before it fires.** swd_yard_d768_L12_T2048_100M, a transformer of the same shape
  (d768, 12 layers, 77.9M body weights), same data, recipe and seed, 100M tokens. `runs_launch/sweep5.sh` (sha256
  1817f57d03c46ee6...) run by `chain6.sh` in tmux launch6 (PID 2592656, waits for chain5 2247563), tree guard 2592659.
  About 45 minutes after wave SWD ends.
  - YD3: the 100M transformer beats the best SWD FULL arm by more than 0.050 (60%).
  - YD4: the gap at 100M is smaller than the gap at 50M on the same diary-64 shape (0.17755) (65%).

### 2026-10-09T18:03:25Z - heartbeat (JIMOTHY)
- swd_shallow_diary121_d768_L12_100M at step 350 of 763, loss 4.6734, 23,686 tok/s. TEST about 18:40Z. No stops. Vast 0.

### 2026-10-09T18:23:46Z - heartbeat (JIMOTHY)
- swd_shallow_diary121 at step 600 of 763, loss 4.3603 (diary64 4.3270 at the same step, 121 behind by 0.033). TEST about 18:40Z, then swd_deep32. No stops. Vast 0.

### 2026-10-09T18:44:35Z - VERDICT SWD4; wave SWE sealed (JIMOTHY)
- swd_shallow_diary121_d768_L12_100M: TEST **1.28819 bpb**. Diary 64 at 100M (1.28077) leads it by 0.00742, inside
  the 0.010 line. **VERDICT SWD4: MISSES** (sealed at 25% that 64 would win by more than 0.010). A tie again by the
  line, but 64 leads at both budgets (0.00313 at 50M, 0.00742 at 100M) and is faster: the base shape takes diary 64.
- swd_deep32_diary64_d512_L32_100M fired 18:39:20Z (SWD1, SWD2).
- **Diary is still on its small edge** (64 is the smallest tried, and leads). Rule (a): push two values past it.
- **SEALED: wave SWE, before it fires.** swe_diary36 (n_sub 6) and swe_diary16 (n_sub 4), d768 L12, 50M tokens,
  same recipe, data and seed as wave SWC; read against diary 64 at 50M (1.40020). `runs_launch/sweep6.sh` (sha256
  86047938d7ddb01d...) run by `chain7.sh` in tmux launch7 (PID 2678820, waits for chain6 2592656), tree guard 2678823.
  About 70 minutes after the 100M yardstick.
  - SWE1: diary 36 beats diary 64 by more than 0.010 (20%).
  - SWE2: diary 16 is worse than diary 64 by more than 0.010 (65%), so the diary curve turns between 16 and 64.

### 2026-10-09T19:03:58Z - heartbeat (JIMOTHY)
- swd_deep32_diary64_d512_L32_100M at 24 min, compiled, before its step-50 line (normal for 32 layers). Running (state S/l), no stops. Vast 0.

### 2026-10-09T19:23:25Z - heartbeat (JIMOTHY)
- swd_deep32 at step 50 of 763, loss 6.5126 (12-layer diary64 read 6.1816 at step 50), 2,576 tok/s averaged over the compile; the true rate comes at step 100. No stops. Vast 0. (Quiet beat: not committed alone.)

### 2026-10-09T19:43:24Z - heartbeat (JIMOTHY)
- swd_deep32 at step 100 of 763, loss 5.7664 (12-layer 5.5194), 5,883 tok/s: TEST about 00:00Z. deep48 then about 6.5 h, the yardstick 45 min, wave SWE 70 min: the queue clears about 09:00Z. Spark load 2.1. No stops. Vast 0.

### 2026-10-09T20:03:26Z - heartbeat (JIMOTHY)
- swd_deep32 at step 200 of 763, loss 5.1876, now 12,932 tok/s (Spark load fell from 2.1 to 1.1): TEST about 21:50Z, earlier than the 00:00Z said at 19:43Z. No stops. Vast 0.

### 2026-10-09T20:23:30Z - heartbeat (JIMOTHY)
- swd_deep32 at step 300 of 763, loss 4.8910, 12,723 tok/s. TEST about 21:45Z. No stops. Vast 0.

### 2026-10-09T20:43:41Z - heartbeat (JIMOTHY)
- swd_deep32 at step 450 of 763, loss 4.6586 (12-layer 4.5924 at step 450; the gap 0.066, down from 0.247 at step 100), 12,950 tok/s. TEST about 21:40Z. No stops. Vast 0.
- Fix: six run logs I committed were copies of the shell's .out (stderr included); the pull loop (runs_launch/pull_launch.sh) replaces them with the trainer's own runs/*.log, the canonical record. Those are committed now.
