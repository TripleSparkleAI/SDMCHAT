# How we train the SDM-only language models

This document describes the model in `track4_sdmonly_models.py` and the way `track4_sdmonly_train.py` trains it.
It states what is built. Results are in the plan file `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` and, once they
exist, in the run folder `runs_sdmonly/`. A number that is not in one of those places has not been measured.

**The law (2026-10-03).** Every model is all SDM. There is no softmax attention and no transformer block. Every
step that mixes information across positions is a Kanerva memory written and read at run time (§9). A transformer
is trained only as a yardstick, on the same data, tokens and optimiser. Every model is trained from scratch, with
no teacher and no distillation. The full statement is section 0 of the plan file. Sections 8 to 14 below record
what changed on 2026-10-03; where they disagree with sections 1 to 7, they win.

## 1. The three models

| model | made from | trained on | scored on |
|---|---|---|---|
| SDM BASE | nothing (from scratch) | FineWeb-Edu web text | TEST bits per byte on held-out FineWeb-Edu |
| SDM CHAT | SDM BASE | chat turns, half mixed with web text | CHAT bits per byte, looping replies |
| WEIRD LITTLE GUY | SDM CHAT | his own corpus, with web and chat rows mixed in | his own TEST, and how much web and chat he forgot |

Every model is trained by next-token prediction. There is no teacher model and no distillation.

## 2. The model

```
   tokens  y_t, y_t-1, ... y_t-7                       all earlier tokens
        │  embedding E (V x d, tied with the output)        │
        ▼                                                   ▼
   8 back-token vectors                     5 moving averages (0.5 0.8 0.9 0.97 0.99)
        └──────── each RMS-normalised, joined, one linear map W_x ────────┘
                                   │
                                   ▼   x  (the working vector, d wide)
        ┌─ hop h = 1 .. H ─────────────────────────────────────────────────
        │  address   q = BatchNorm( W_q,h · norm(x) )        d -> d_a
        │  locate    the k best of n_sub x n_sub locations   (product keys)
        │  read      r = (1/k) · sum over woken locations of  w_m · v_m
        │  add       x <- x + r
        └──────────────────────────────────────────────────────────────────
                                   │
                                   ▼   logits = norm(x) · E^T      (the cleanup against the token table)
                              next token, which joins the context
```

### 2.1 The context features

For position t the features are the embeddings of the 8 most recent tokens and 5 causal moving averages of all
earlier embeddings:

    a_t(β) = Σ_{s ≤ t} β^(t−s) e_s  /  Σ_{s ≤ t} β^(t−s)

**In words:** each average is a blend of every earlier token, weighted so that recent tokens count more; β sets
how far back it reaches (0.5 reaches about 2 tokens, 0.99 about 100). The 13 vectors are RMS-normalised, joined
end to end and mapped to the working vector x by one matrix W_x.

### 2.2 The address and the product keys

The address q is a linear map of x followed by batch normalisation with no learned scale. It is split into two
halves q¹ and q². Each half is scored against n_sub unit-length sub-keys:

    s¹_i = q¹ · c¹_i        s²_j = q² · c²_j        score(i, j) = (s¹_i + s²_j) / √2

**In words:** a location is a pair (i, j), so n_sub sub-keys in each half give n_sub² locations. A location's
score is the sum of its two half scores. The best 2k locations must lie among the best 2k sub-keys of each half,
so the search scores 2·n_sub sub-keys and a 2k by 2k grid instead of all n_sub² locations, and it is exact. The
reason: if a cell (i, j) is among the best 2k, then fewer than 2k cells beat it, and every cell (i, j') with a better
second half and every (i', j) with a better first half beats it; so i is among the best 2k first halves and j among
the best 2k second halves. Sorted by score, the best cells always fill a corner staircase of the grid. The self-test
compares the search with a brute-force search.

### 2.3 The read

    θ = the k-th best score            w_m = sigmoid( (score_m − θ) / softness )
    r = (1 / k) · Σ_{m in the best 2k} w_m · v_m

**In words:** the 2k best locations are weighed. A location well above the cut-off counts almost fully, one just
below it counts a little, so the cut is soft and the read changes smoothly as the address moves. With the trained
softness the cut is very soft: on the trained centre arm the 33rd candidate weighs 0.407 on average and the 64th
0.314, so all 64 candidates contribute (measured on `p0_A_c`, 4,096 positions, first hop). The read is the average
of the woken locations' value rows. Values start at zero, so at the first step the model is exactly the model
with no reads; the self-test checks this.

### 2.4 What is not in the model

There is no attention. There is no dense layer between the reads by default (`--readout-f 0`). The only step
that is not linear in the features is the choice of which locations wake.

## 3. What comes from our earlier measurements

| choice | where it was measured |
|---|---|
| the wide address: 8 back tokens and 5 averages | SPARK24: the first change that moved the family (W over U by 0.0083 bpb) |
| the soft cut-off and the 1/k gain | SOFTSDM, then `SdmStore` |
| the address gradient into x stopped (`qgrad 0`) | SDMLLMSTORE: the hard cut-off's gradient through the query cost 0.017 to 0.025 bpb |
| values start at zero; store learning rate x3; no weight decay on the store | SDMLLMSTORE |
| a control with no store beside every store claim | `track4_245` |

Product keys are from Lample et al. 2019 ("Large Memory Layers with Product Keys") and Berges et al. 2024
("Memory Layers at Scale"). They let the store hold tens of thousands to millions of locations at the cost of a
few hundred scores.

## 4. The controls

| arm | what it is | what it answers |
|---|---|---|
| `sdmonly_none` | the same model with no reads | how much the reads add |
| `sdmonly_dense` | each read replaced by a SwiGLU layer of the same FLOPs | does a sparse read beat a dense layer at equal compute |
| `sdm`, `sdm_nostore` | the earlier shape: 3,600 locations, 2 hops, a dense readout | is SDM-only as good as the shape it replaces |
| `qwen` | a 4-layer transformer | how far the family is from attention |

After training, two ablations are scored on the SDM-only arm: every read returns zero, and every value table is
shuffled so the addressing is kept and the content is scrambled.

## 5. The training recipe

| part | setting |
|---|---|
| loss | next-token cross-entropy over the 129,280-token DeepSeek V4 vocabulary |
| batch | 32 windows of 256 tokens, offsets drawn from a seeded generator, so every arm with one seed sees the same tokens in the same order |
| optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 on 2-D non-embedding weights, gradient clip 1.0; since wave 6, Muon on the body matrices (`--body-opt muon`, §10) |
| store group | its own learning-rate multiplier and weight decay |
| schedule | 2% warmup, then cosine to 10% of peak; or `--sched wsd`: 2% warmup, constant, linear decay to zero over the last 20% |
| precision | bf16 autocast for matmuls, fp32 weights, fp32 cross-entropy |
| score | bits per byte on S0's held-out FineWeb-Edu windows; a target that is the BOS token is not scored |

The trainer file adds no loop of its own. It swaps the model builder and the schedule inside the S0 trainer and
calls that trainer's `train()`, so batches, checkpoints, resume, stamps and scoring are identical to every
earlier SDMLLM run.

The warmup-constant-decay schedule is the one Hägele et al. 2024 show matches cosine while letting one run serve
several token budgets: a checkpoint from the constant part can be decayed at any point.

## 6. How a run is launched

On the Spark, through the queue:

    python3 ~/settle24/code/q.py --root ~/settle24/q submit -p 9 --threads 2 --gpu 1 --mem 14 --name SDMONLY-<run> \
      -- nice -n 10 ~/settle24/code/sdmonly_arm.sh new <run> sdmonly 0 20000000 train --hops 4 --n-sub 256

`sdmonly_arm.sh` is the installed copy of `track4_sdmonly_spark_arm.sh`. Logs and result files go to
`~/settle24/runs/sdmonly/`, checkpoints to `~/settle24/ck/sdmonly/<run>/`.

On the M5, for a smoke run:

    python3 track4_sdmonly_models.py --selftest
    python3 track4_sdmonly_train.py --arm sdmonly --tokens 400000 --n-sub 64 --hops 2 --val-windows 8 --run smoke --fresh

## 7. What this model is and is not

It is a language model whose body is sparse reads from trained memory tables. It keeps the shape of Kanerva's
memory: many locations, a small set that wakes for an address, a summed read, and reads repeated from the result.

The trained stores do not write memories while they run. Their keys and values are learned by gradient during
training and are fixed afterwards. It has no binding and no unbinding. (2026-10-03: the SDM mixer layer of §9 does
write at run time. It is now the model's only route past the last 8 tokens and the 5 averages.)

## 8. The data, and the repetition we found (2026-10-03)

| shard | tokens | what it is |
|---|---|---|
| `train` (`data/train.u32`) | 65,522,263 | the S0 train shard: the first 64,488 documents of FineWeb-Edu `sample/10BT` |
| `train_big` (`train_big.u32`) | 3,018,298,789 (12,073,195,156 bytes) | `train` plus later documents; 34 exact duplicates of `val` documents skipped |
| `val` | 2,000,916 | held out; its second half is TEST, the same for every run |

Every run up to wave 10 trained on `train`. So a "200M-token" run saw that shard about 3 times, and the Spark's
600M chat-base runs saw it about 9 times. The wave 7 reading "more tokens flatten near 1.47, so the model is short
of context, not under-trained" is confounded by that repetition and is withdrawn as a conclusion.

**From wave 11 on, every scaling run trains on `train_big`, one pass, no repeats.** The TEST stream does not
change, so every new score is comparable with every old one. Each rented box builds `train_big` with
`track4_sdmllm24_extend_fineweb_edu_shards.py` and checks that it is a byte-identical prefix of the Spark's file.
The wave 11 arms named `w11_M*` stay on `train` at 200M on purpose, so they compare with wave 10.

## 9. The SDM mixer layer (2026-10-03)

The trained stores of §2 can only see the last 8 tokens and the 5 averages. The SDM mixer layer is the model's way
to read further back, and it obeys the law: it is a Kanerva memory written and read while the model runs.

```
   position t                        the layer's memory: M hard-locations per head (fixed, e.g. 128 x 128)
     │                               ┌───────────────────────────────────────────────
     ├─ address  q_t = BN(W_a x_t)   │  ·  ·  ●  ·  ·  ●  ·  ·  ·  ●  ·   ← the 2k locations t picks
     │   product keys pick 2k of M   │  (each with a soft weight, the cut-off of §2.3)
     │                               │
     ├─ WRITE  v_t = W_v x_t ───────▶│  add w · v_t into each picked location's counter
     │                               │
     └─ READ ◀───────────────────────│  the same picked locations, holding only what
          r_t = mean of the counters │  EARLIER positions s < t wrote
          x_t <- x_t + W_o r_t       └───────────────────────────────────────────────
          then a dense layer of width 4d
```

In plain words:

- **Fixed locations.** Each head has a fixed number of hard-locations, n_sub squared (`--mix-n-sub`; 64 gives
  4,096 and 128 gives 16,384). Product keys pick about k = 32 of them for each position, with the same soft
  cut-off as the trained stores. The address map and the sub-keys are learned in training and then fixed.
- **Written at run time.** Each position writes a learned vector of itself into the locations it picked. Nothing
  in the layer's content is a trained table: it starts empty for every text.
- **Read from the past only.** Each position reads the locations its own address picks, and those hold only what
  earlier positions wrote. The read is divided by the total weight it gathered, so it is a mean of counters.
- **Fading heads.** Each head can fade old writes by a factor per token (`--mix-decays`). The first mixer used
  1.0, 0.999, 0.99 and 0.9. A never-fading head uses 1.0. Wave 10 found never-fading better (1.46048 against
  1.48242 at 200M), so wave 11 runs every mixer as 4 never-fading heads (`--mix-decays 1.0,1.0,1.0,1.0`).
- **Cost does not grow with the window.** The window is processed in chunks (`--mix-chunk`, 256). Inside a chunk
  the write-then-read is one masked product; between chunks only the counters are carried. The number of locations
  is fixed, so a token costs the same at T 256 and at T 16,384. That is what makes a long window reachable.

The code is `SdmMix` in `track4_sdmonly_models.py` (flags `--mix-*` in `track4_sdmonly_train.py`). Its self-test
checks causality, that chunk 7 equals chunk 20, and that a token 17 positions back reaches the output.

## 10. Muon on the body (2026-10-03)

Muon replaces AdamW for the 2-D body matrices (`--body-opt muon`, `track4_sdmonly_optim.py`). Each step takes the
momentum (Nesterov), replaces it by the nearest semi-orthogonal matrix (5 quintic Newton-Schulz iterations), and
steps. The embedding, the output head and every norm stay on AdamW. The default Muon learning rate is 0.02.

Wave 6 measured it: under Muon the centre arm gained 0.054 bpb and used 37 to 52% of each store's locations,
against 3 to 6% under AdamW. Every later wave trains the SDM arms and the transformer yardstick with Muon, so the
yardstick comparison is fair.

## 11. Long windows: the eval batch cap (2026-10-03)

Scoring a window of 1,024 tokens at batch 32 builds a block of 129k-vocab logits of 15.78 GiB in one step, and the
runs died after training finished. Both scorers, `eval_bpb` in `track4_sdmllm_train_one_arm.py` and `per_window`
in `track4_sdmllm_paired_window_comparison.py`, now cap a batch when T x bs > 16,384: the batch drops to about
8,192 tokens (bs = 8,192 // T, at least 1). Every run at T 512 or less scores exactly as before.

Training at a longer window keeps the tokens per step fixed at B x T = 8,192: T 1,024 runs B 8, T 4,096 runs B 2,
and T 16,384 runs B 1.

## 12. The tokenizer (2026-10-03)

Wave V trained the centre arm at equal training bytes (95,417,201) with four tokenizers and scored by document.
Against the DeepSeek 129,280 vocabulary: a trained 65,536 BPE gained 0.00543 (z -7.6), 32,768 gained 0.02069
(z -25.2), and 16,384 gained 0.01990 (z -19.7). The delegated decision: the tokenizer moves to the trained 32,768
BPE, pending a second seed of the 32k and 129k arms. The champion recipe at 32k scored 1.5129 by document; its
129k pair is rerunning as `w7r_*` and is not yet scored in THE LOG. Wave 11 still trains on the DeepSeek tokens of
`train_big`, so it compares with every earlier run.

## 13. The scaling grid of wave 11 (2026-10-03)

Wave 11 asks how each family scales on fresh data. It runs 46 runs on 10 rented boxes of 2x RTX 5090 (queues
`track4_sdmonly_wave11_q_<e..n>.txt`).

| family | what it is |
|---|---|
| cn | the best SDM recipe: no trained store reads (`sdmonly_none`), a SwiGLU of width 192 after each hop, an untied input table, 2-token products, Muon |
| mq | the Muon transformer yardstick: 4 layers, MLP width 8/3 d (`--qwen-f`; 2,048 at d 768) |
| cnmix | cn plus 4 never-fading mixer layers of 128 x 128 locations each |

- **The grid (Phase B, `train_big`, one pass):** widths 256, 384, 512 and 768 by tokens 100M, 400M and 1.6B, for
  cn and mq. cnmix at width 256 (100M; 400M at T 256, 1,024 and 4,096) and at width 512 (100M). Second seeds at
  width 256 and 400M.
- **The mixer tests (Phase A, `train`, 200M, comparable with wave 10):** the never-fading mixer at T 1,024, 4,096
  and 16,384; 8 and 2 layers; 256 x 256 locations; k 64; address width 128; with store reads on; second seeds of
  the mixer and of cn.
- **The fit, per family:**

      L(N, D) = E + A / N^alpha + B / D^beta

  **In words:** the loss falls toward a floor E as the model grows and as it sees more text. N is the number of
  non-embedding parameters, D the number of training tokens. alpha says how fast size pays, beta how fast data pays.
- The trainer gained `--qwen-layers` and `--qwen-f`. Their defaults, 4 and 688, reproduce every earlier yardstick.
- The ten sealed predictions are Y1 to Y10 in the plan.

## 14. After the BASE: CHAT, then RL (2026-10-03)

- **CHAT** is SFT of the BASE on open chat conversations (`chat_mix`: smol-smoltalk joined with an equal number
  of FineWeb-Edu tokens; §6.1 of the paper draft), by plain next-token prediction, with no teacher model in the loop.
- **RL is exploratory and later.** Decision A (2026-10-03T14:52Z): context is the main line, and RL runs only
  after CHAT exists. RL shapes behaviour a model already shows sometimes; it cannot add context or knowledge.
- **The format is MiMo's.** We copy the six-column record, the checker types and the loop from Xiaomi's
  MiMo-V2.6-RL-oss release, and use none of its weights. Our tasks form a ladder of easy rungs (copy a line, count,
  close brackets, format, one fact, arithmetic, tiny code), then MiMo's Music, General and Code on top. We train
  only where a rung's pass rate sits between 5 and 95 percent. The loop is nanochat's `chat_rl.py`: a group of 8
  replies, advantage = reward minus the group mean, no KL, no division by the spread. The judge for rubric rungs
  is ds4-flash through the M5 Hermes proxy.
- Spec: `SPEC_SDMONLY_RL_LADDER_2026-10-04.md` and `track4_sdmonly_rl_ladder.py`. Study:
  `wikis/WIKI_MIMO/05-what-we-copy-for-sdm.md`. The forward path: `FORWARD_PATH_SDMONLY_2026-10-03.md`.

## Sources

- Lample, Sablayrolles, Ranzato, Denoyer, Jégou. Large Memory Layers with Product Keys. NeurIPS 2019.
- Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh. Memory Layers at Scale. arXiv 2412.09764, 2024.
- Hägele, Bakouch, Kosson, Ben Allal, Von Werra, Jaggi. Scaling Laws and Compute-Optimal Training Beyond Fixed
  Training Durations. arXiv 2405.18392, 2024.
- Kanerva. Sparse Distributed Memory. MIT Press, 1988.
- Local: `REPORT_SDMLLM_S0.md`, `REPORT_SDMLLMSTORE.md`, `RESEARCH_BASEDATA_2026-10-04.md`,
  `../HISTORY_SDM_AND_SPARSESTAR_2026-10-04.md`.
