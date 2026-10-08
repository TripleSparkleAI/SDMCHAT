# SDMLLMSTORE · does a properly trained store beat the no-store control? · 2026-09-30

Lane SDMLLMSTORE (wave four, lane 4 "SDM to LLM"). M5 only. Branch `settle-sdmllmstore`.
Code and results: `experiments/track4/sdmllm/` (files prefixed `track4_sdmllmstore_`, plus extensions to
`track4_sdmllm_models.py` and `track4_sdmllm_train_one_arm.py`). Sealed predictions and scores:
`SETTLE/SETTLE_CAMPAIGN_2026-09-30.md` (SDMLLMSTORE lines).

## 1. The question

S0 (`REPORT_SDMLLM_S0.md`) found that the learned SDM store cost 0.017 to 0.025 bpb against the same model
with the store removed, in all three seeds. The owed list asked for three things. First, a store optimiser:
its own learning rate, no weight decay on keys and values, and value initialisation. Second, a softness sweep
from 0.25 to 8. Third, two more transformer seeds. Optionally, an n-gram hash address arm. The question is
whether a store that trains properly beats the no-store control.

## 2. Protocol

S0's protocol is kept unchanged: DeepSeek V4 tokenizer (129,280 ids), FineWeb-Edu shards, 20M training tokens
(2,441 steps, batch 32, T 256), AdamW (0.9, 0.95), peak lr 3e-3, warmup 2%, cosine to 10%, grad clip 1.0,
bf16 autocast on MPS, and the same batches per seed. Every arm is scored on the same 3,873 TEST windows.
Paired differences:

    delta = (sum_w nats_A(w) - sum_w nats_B(w)) / (ln 2 * sum_w bytes(w))

Plain reading: how many bits per byte arm A is worse than arm B on exactly the same text. The SE comes from
2,000 bootstrap resamples of windows (the same draw as S0).

Instrument check: recomputing S0's pairs from the S0 checkpoints reproduces S0 to 5 decimals (sdm s0 minus
nostore s0 = +0.02484). Deltas below are arm minus `sdm_nostore` of the same seed unless stated; negative
means the store wins.

## 3. Pilot, disclosed before the seal

Per-parameter gradient norms on the S0 checkpoints (steps 2400 to 2440):

| model | total grad norm | embedding | wx | store keys and values |
|---|---|---|---|---|
| sdm seed 0 | 2.49 | 2.15 | 1.21 | not in the top 8 |
| sdm seed 1 | 2.78 | 2.38 | 1.38 | not in the top 8 |
| sdm softness 2.0 | 0.43 | 0.33 | 0.16 | small |
| sdm_nostore | 0.42 | 0.33 | 0.15 | none |

S0's sdm runs clipped (gradient norm above 1.0) on 16 to 20% of logged steps, all late in training. Nostore
and softness 2.0 never clipped. The large gradient does not sit in the store's parameters. It flows from the
hard cut-off back through the query into the shared features. The global clip then shrinks every parameter's
update, the embedding's included. This pilot shaped the arms below.

## 4. New mechanisms (defaults reproduce S0 bit-for-bit in forward)

- **Store parameter group.** Keys and values get their own AdamW group with an lr multiplier and their own
  weight decay. In S0 they sat in the decayed group at 0.1 because they are 2-D.
- **Value initialisation.** `s0` is randn x 0.02 sqrt(k). `zero` makes the store start as a no-op.
- **Query-gradient scale** alpha on the stream entering the query:

      g(x) = stopgrad(x) + alpha * (x - stopgrad(x))

  Plain reading: the forward value is x, but only a fraction alpha of the gradient flows back into x. With
  alpha 0 the address features are trained only by the readout path, while W_q, the keys and the values
  still train.
- **N-gram hash address arm (`sdm_ngram`).** The similarity address is replaced by fixed hashes:

      x <- x + sum over n in orders of T_n[ h_n(y_{t-n+1}, ..., y_t) mod R ]
      h_n = (sum_i a_{n,i} * y_{t-i}) mod (2^31 - 1)

  Plain reading: hash the last n token ids with fixed random odd multipliers into a row of a learned table,
  and add that row to the working vector. Orders (2, 3) use 2 tables of 3,856 rows. Orders (1,) use 1 table
  of 7,712 rows. Both total 2,898,432 non-embedding parameters, exactly the sdm arm's count. Rows start at
  zero and use the store recipe (lr x3, weight decay 0). Eval ablations: `shuffle_keys` permutes the
  bucket-to-row map, `zero_read` drops the read.

## 5. Results (TEST bpb)

### Block 1, seed 0, softness 0.5

| arm | bpb | delta vs nostore | SE | clipped steps | max gnorm | zero_read | shuffle_keys |
|---|---|---|---|---|---|---|---|
| sdm (S0) | 1.63020 | +0.02484 | 0.00026 | 20% | 2.7 | 1.6566 | 1.6534 |
| storeopt (lr x3, wd 0, zero init) | 1.72924 | +0.12388 | 0.00053 | 63% | 28.1 | 1.8042 | 1.7723 |
| qstop (S0 store settings, alpha 0) | 1.60666 | +0.00130 | 0.00022 | 0% | 0.48 | 1.6114 | 1.6099 |
| storeopt_qstop | 1.60541 | +0.00006 | 0.00022 | 0% | 0.47 | 1.6498 | 1.6262 |
| storeopt_qscale01 (alpha 0.1) | 1.60582 | +0.00046 | 0.00022 | 0% | 0.49 | 1.6263 | 1.6154 |
| sdm_nostore (S0) | 1.60536 | 0 | | 0% | 0.47 | | |

- storeopt's value rows grew to mean norm 4.2 and max 26.6 (S0: 2.0 and 10.0). Its embedding gradient
  reached 15.4 at the end of training.
- Factorial reading. Stopping the query gradient recovers 0.0235 bpb against S0 sdm. The optimiser knobs are
  worth another 0.0012 once alpha is 0, and cost 0.099 while alpha is 1.

### Block 2, softness sweep on storeopt_qstop, seed 0

| softness | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| bpb | 1.60506 | 1.60541 | 1.60557 | 1.60572 | 1.60603 | 1.60569 |
| delta vs nostore | -0.00030 | +0.00006 | +0.00021 | +0.00036 | +0.00067 | +0.00033 |
| zero_read | 1.6497 | 1.6498 | 1.6611 | 1.6729 | 1.6988 | 1.6848 |
| locations used | 0.458 | 0.428 | 0.416 | 0.382 | 0.579 | 0.671 |

Every delta SE is 0.00021 to 0.00022. The sweep spans 0.0010 bpb, within the fixed-seed rerun noise (section 6).

### Block 3, the sealed winner (storeopt_qstop, softness 0.25) over seeds

| seed | bpb | nostore | delta | SE |
|---|---|---|---|---|
| 0 | 1.60506 | 1.60536 | -0.00030 | 0.00021 |
| 1 | 1.60740 | 1.60613 | +0.00126 | 0.00023 |
| 2 | 1.60233 | 1.60501 | -0.00268 | 0.00021 |

Mean -0.0006. Not a consistent win.

### N-gram hash arms (store recipe, 3 seeds each)

| arm | seed 0 | seed 1 | seed 2 | delta vs nostore (s0 / s1 / s2) | zero_read s0 | shuffle s0 |
|---|---|---|---|---|---|---|
| sdm_ngram, orders (2, 3) | 1.60195 | 1.60309 | 1.60177 | -0.00341 / -0.00304 / -0.00324 | 1.6079 | 1.6146 |
| sdm_ngram, orders (1,) (the control) | 1.59706 | 1.59798 | 1.59751 | -0.00830 / -0.00816 / -0.00749 | 1.6138 | 1.6228 |

Delta SEs are 0.00016 to 0.00018.
- (2, 3) minus (1,): +0.00490 / +0.00511 / +0.00425 (SE 0.00017 to 0.00018). The context-free table beats
  the n-gram table in every seed.
- Against the dense-MLP control at matched parameters (S0 `sdm_mlp` seed 0, 1.60933): (2, 3) -0.00738,
  (1,) -0.01227. Extra parameters as such do not explain the gain.
- The (1,) table's eval ablation costs +0.017 (zero_read) and +0.026 (shuffle) on seed 0. The model relies
  on the per-token vector.

### Transformer yardstick

| seed | 0 | 1 | 2 | 3 | 4 | mean | spread |
|---|---|---|---|---|---|---|---|
| qwen bpb | 1.61161 | 1.55801 | 1.59720 | 1.57797 | 1.56804 | 1.58257 | 0.0536 |

qwen seed 3 minus the orders (1,) arm, both seed 0: -0.01909 (SE 0.00049).

## 6. Controls and noise

- **Reproduction control:** S0's sdm seed 0 rerun on the new code gives 1.62759 against 1.63020, a paired
  delta of -0.0026 (SE 0.00017). Forward passes are bit-identical to S0's code (checked on sdm, nostore,
  qwen and mlp), so the gap is MPS run-to-run nondeterminism amplified by that arm's clipping.
  **The bootstrap SE (about 0.0002) understates rerun noise by about ten times.** The seed spreads are the
  honest yardstick: nostore 0.0011, the orders (1,) arm 0.0009, the block-3 winner 0.0051.
- **Negative controls:** eval-time zero_read and shuffle_keys on every store arm. The orders (1,) table is
  itself the control for the n-gram address, and it inverted the expected result.
- **Positive control for the instrument:** S0's published deltas reproduce exactly from the checkpoints.

## 7. Verdicts

- **Stopping the query gradient removes S0's whole store cost.** The store then ties the no-store control:
  +0.00006 at softness 0.5 and a mean of -0.0006 over three seeds at softness 0.25. It is heavily used,
  since zero_read costs +0.045 to +0.05, but it adds nothing net. The readout and features learn the same
  thing without it.
- **The owed optimiser knobs are harmful while the query gradient flows.** A higher lr with no weight decay
  lets value rows grow, gradients explode and the clip throttles training: +0.124 bpb. Once alpha is 0 they
  are worth 0.0012.
- **Softness stops mattering once the gradient path is fixed** (0.0010 across 0.25 to 8). S0's softness
  effect (0.022) was the gradient pathology, not the read.
- **No context-addressed store beats no-store across seeds.** This covers similarity address, soft or hard,
  and bigram plus trigram hash. The (2, 3) hash arm wins by 0.003 in all seeds, but the context-free unigram
  table beats it by 0.0045 at equal parameters.
- **The only consistent winner, -0.008 bpb in all three seeds (3x the rerun noise), is a hashed table keyed
  on the current token.** It works as an untied input embedding: at this vocabulary and budget the tied
  129,280-row table is also the output layer, and a free per-token input vector helps. It is a finding about
  the tied embedding, not about memory.
- **The transformer still leads** the best SDM-family arm by 0.015 bpb on the 5-seed mean. Its seed spread
  stays wide (0.054).

## 8. Stamps

Every run: M5 Max, macOS 25F71, torch 2.8.0, powermode 2 at start and end, load average 12 to 165 on 18
cores (stamped per run in `runs/<run>.result.json` and per queue step in `runs_store/queue_A.log`). No timing
is a claim. Two training processes ran until about 20:05Z, when queue B was stopped at the navigator's
request. The interrupted arm (softness 0.25 seed 2, step about 1,250) was deleted and restarted from scratch,
not resumed. From then on only one process ran. No arm was dropped.

## 9. Sealed predictions, scored

| id | sealed | result | verdict |
|---|---|---|---|
| P1 | storeopt still loses, +0.015 (band +0.005 to +0.025) | +0.124 | miss |
| P2 | qstop -0.004 (band -0.010 to +0.001) | +0.0013 | miss |
| P3 | storeopt_qstop best block-1 arm, -0.006 | best, but +0.0001 | split |
| P4 | qscale01 between storeopt and storeopt_qstop | 1.60582, between | hit |
| P5 | qstop arms never clip; storeopt clips on at least 10% of steps | 0% and 63% | hit |
| P6 | softness 0.25 worst; best at 1, 2 or 4 | 0.25 best, 4 worst | miss |
| P7 | sweep spread under 0.015 on a qstop recipe | 0.0010 | hit |
| P8 | winner beats nostore in all 3 seeds, mean -0.005 | -0.0003 / +0.0013 / -0.0027 | miss |
| P9 | zero_read and shuffle each at least +0.010 on the winner | +0.045 / +0.017 | hit |
| P10 | qwen s3, s4 in 1.55 to 1.62; 5-seed mean 1.590 +-0.015, below nostore | 1.578, 1.568; 1.5826 | hit |
| P11 | repro within 0.003 of 1.6302 | 0.0026 | hit |
| P12 | sdm_ngram -0.008 (band -0.025 to -0.002); zero_read at least +0.02 | -0.0034; +0.006 | split |
| P13 | sdm_ngram beats the best block-1 similarity arm, seed 0 | 1.60195 vs 1.60541 | hit |
| P14 | unigram control within 0.002 of nostore; ngram beats it by 0.002 | -0.0083; ngram worse by 0.0049 | miss |
| P15 | unigram table beats nostore in seeds 1 and 2 by at least 0.004 | -0.0082, -0.0075 | hit |

8 hits, 2 splits, 5 misses. P12 to P15 are follow-ups sealed after earlier results; see their ledger lines.

## 10. Owed

1. An untied input embedding at full vocabulary against the hashed unigram table. This asks whether the
   hash's collisions help or hurt, and gives the clean name for the one win.
2. A store whose gain cannot be absorbed by the readout: a much larger store at matched FLOPs (S0's
   sdm_big was on the broken gradient path), or a longer token budget where fixed features saturate.
3. Engram-style multi-head hashing with a gate, for n = 2 and 3, to reduce collisions before the n-gram
   address is written off.
4. Any win claimed at 0.001 to 0.003 needs 3 seeds, because fixed-seed reruns move 0.0026 on this machine.
5. S1 on the Spark after HEADCLIMB, or a priced rental, carries the store question only if items 1 and 2
   show a gap.

## 11. Files

- `track4_sdmllm_models.py` (grad_scale, value_init, NgramStore, ngram_orders), `track4_sdmllm_train_one_arm.py`
  (store group, flags, per-window npz)
- `track4_sdmllmstore_paired.py`, `track4_sdmllmstore_summary.py`, `track4_sdmllmstore_queue.sh`,
  `track4_sdmllmstore_spec.txt`
- `track4_sdmllmstore_paired_result.json`, `track4_sdmllmstore_result.json`, `runs/*.log`, `runs/*.result.json`,
  `runs_store/` (queue logs, block-1 and block-2 paired jsons, the S0 reproduction)
- Checkpoints (8.0 GB, gitignored) live in the worktree's `checkpoints/`. Best: `sdm_ngram_unigram_s0_20M/last.pt`
  (1.59706 bpb, base model, no chat tuning). Best with a real SDM read: `sdm_storeopt_qstop_soft0p25_s2_20M/last.pt`
  (1.60233).
