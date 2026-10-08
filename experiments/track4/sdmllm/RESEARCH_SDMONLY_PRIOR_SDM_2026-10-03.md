# RESEARCH · SDM-only · what our earlier SDM measurements say about the new model · 2026-10-03

This document collects every SDM measurement in this repo that bears on the SDM-only language model
(`track4_sdmonly_models.py`, trained by `track4_sdmonly_train.py`, described in
`TRAINING_SDMONLY_HOW_WE_TRAIN.md`). It reads the reports, the ledger
`SETTLE/SETTLE_CAMPAIGN_2026-09-30.md`, the `*.result.json` files and the training logs
directly. It runs nothing.

## 0. How to read the numbers

| label | meaning |
|---|---|
| MEASURED | ours, read from the named file in this repo |
| ARITHMETIC | computed here from measured or configured numbers; the formula is shown |
| PAPER CLAIM | an outside paper's number, as recorded in our own wiki or harvest file; not reproduced by us |
| UNVERIFIED | a source that could not be opened, or a number with no file behind it |
| PROPOSAL | a test or a design change; nothing in it has been run |

TEST bpb is bits per byte on the held-out FineWeb-Edu windows of S0 (995,323 scored tokens, 4,824,566
bytes). Lower is better. The S0 report counts 3,873 windows and the later ledger lines count 3,892; both
name the same TEST token and byte totals, so this document quotes each number with the window count its
source states.

### The model this document is about

```
   x_0 = W_x [ n(e_t) .. n(e_{t-7}) , n(ema_0.5) .. n(ema_0.99) ]
   for h = 1..H:   q_h = BN( W_q,h  n( g(x) ) )          g = the query-gradient scale (qgrad)
                   (scores, ids) = product-key top 2k of n_sub^2 locations
                   x <- x + (1/k) sum_m sigmoid((s_m - theta)/softness) v_m
   logits = n(x) E^T
```

Plain reading: the context enters only through 8 back-token vectors and 5 moving averages. Each hop
builds an address from the working vector, wakes about k locations of its own table, and adds their
value rows. There is no attention and no dense layer by default.

Centre of wave 1 (from `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`): d 256, 4 hops, one table per hop,
256 x 256 = 65,536 locations each, k 32, 1 head, softness 0.25, qgrad 0, lr 3e-3, store lr x3, store
weight decay 0, values start at zero, 20M tokens.

ARITHMETIC: store parameters at the centre are 65,536 x 256 values + 2 x 256 x 128 sub-keys + 256 x 256
query map = 16,908,288 per hop, 67,633,152 over 4 hops. The tied token table is 129,280 x 256 =
33,095,680. The stores are twice the size of the token table. Every model in our record had a store of
at most 16,384 rows.

---

## 1. The five findings that most change how we build and train it

```
   ✦ 1  THE CONTEXT IS THE CEILING, NOT THE STORE
        N  no store, 4 back + 2 averages   1.50670  ████████████████████████████████
        W  8 back + 5 averages             1.49853  ███████████████████████████████▌   -0.0083
        Q  4-layer transformer             1.35223  ████████████████████              -0.1545
        all three d 256, 300M tokens, Spark; hops add depth, never context

   ✦ 2  QUERY GRADIENT + STORE RECIPE = BLOW-UP
        storeopt  qgrad 1, store lr x3, wd 0, zero init   +0.124 bpb   63% steps clipped   gnorm max 28.1
        storeopt_qstop   same, qgrad 0                    +0.00006     0% clipped          gnorm max 0.47

   ✦ 3  LOCATIONS STOP BEING USED AS TRAINING GOES ON
        before query BN (smoke)   2.7% ▏
        20M  d256                 40 to 48% ▏████████████
        300M d256                 9.3% ▏███
        300M d512 / 600M d768     17 to 25% ▏██████

   ✦ 4  THE NOISE FLOOR IS TEN TIMES THE WINDOW SE, AND 20M RANKINGS FLIP AT 300M
        window bootstrap SE ~0.0002 · a fixed-seed rerun moved 0.0026 · seed spreads 0.0011 to 0.0051
        per-token table: -0.008 at 20M (3 seeds) -> -0.0006 at 300M (2 seeds) -> -0.0112 at 3B (1 seed)

   ✦ 5  AT LONG BUDGETS THE GRADIENT CLIP IS ON MOST OF THE TIME, IN EVERY ARM
        20M qgrad 0     0% of logged steps over 1.0
        300M d256 S     89% in the second half
        3B d512 VA      99.8% in the second half, median pre-clip norm 5.31
```

1. **The fixed context window is the limit of the family, and SDM-only keeps that limit.** At 300M
   tokens a 4-layer transformer beats the no-store model by 0.15447 (SE 0.00097). The only change that
   moved the SDM family was a wider address (8 back tokens, 5 averages): -0.00827 (s0) and -0.00654 (s1)
   at d 256, and at d 512 the wide address on 300M tokens beat the narrow one on 1B by 0.02629. Hops
   cannot add context, so the next address change is worth more than the next store change.
2. **Never let the query gradient flow while the store has its own fast recipe.** Wave-1 arm `p0_A_qg1`
   is that combination (qgrad 1 with store lr x3, wd 0, zero init). Watch its gradient norm and clip rate.
3. **Usage collapses with training even with batch-normalised queries.** The fraction of locations ever
   picked over 64 TEST windows fell from about 40-48% at 20M to 9.3% at 300M (d 256, 3,600 locations).
   A 65,536-row table that collapses the same way wastes most of its rows. Instrument usage per hop
   during training, not only at the end.
4. **Decide nothing below 0.005 on one seed at 20M, and confirm every 20M winner at 100M to 300M.**
   The plan's rule (0.003 and a window-bootstrap z of 3) can pass on seed noise.
5. **At long budgets the clip at 1.0 is active on most logged steps for every arm in the family,** the
   no-store arm included. The long run's effective learning rate is set by the clip, not by the
   schedule, unless the clip is raised or the cause is found.

---

## 2. The table of findings

Column "for SDM-only": KEEP (already in the recipe, and the evidence supports it), CHANGE (the evidence
says the recipe should move), TEST (the evidence is mixed or absent; a run decides it).

### 2a. The SDM language models (SdmLM, `track4_sdmllm_models.py`)

| # | finding | number (MEASURED) | source | for SDM-only |
|---|---|---|---|---|
| 1 | Hard cut-off with query gradient: the learned store loses to no store in every seed | sdm minus nostore +0.0248 / +0.0232 / +0.0170, SE under 0.0003 (20M, d 256) | `REPORT_SDMLLM_S0.md` §5 | KEEP qgrad 0 |
| 2 | The cost is a gradient path through the query, not the read | qgrad 0 recovers 0.0235; sdm clipped on 16-20% of steps, nostore never | `REPORT_SDMLLMSTORE.md` §3, §5 block 1 | KEEP qgrad 0 |
| 3 | Store optimiser knobs with the query gradient on are destructive | storeopt (lr x3, wd 0, zero init, qgrad 1) +0.12388; 63% clipped; max gnorm 28.1; value row norm mean 4.2, max 26.6 | `REPORT_SDMLLMSTORE.md` §5 | CHANGE: treat `p0_A_qg1` as a risk arm |
| 4 | With qgrad 0 the store knobs are worth little | storeopt_qstop +0.00006 vs qstop +0.00130: 0.0012 | `REPORT_SDMLLMSTORE.md` §5 | KEEP; small lever |
| 5 | qgrad 0.1 sits between | +0.00046 | same | KEEP 0 |
| 6 | Softness stops mattering once the gradient path is fixed | 0.25 to 8 spans 0.0010 (1.60506 to 1.60603) | same, block 2 | KEEP 0.25; low priority |
| 7 | Before the fix, softness looked like a lever | softness 2.0 better than 0.5 by 0.0220 | `REPORT_SDMLLM_S0.md` §5 | do not re-read it as a lever |
| 8 | The read ties no store at 20M after the fix | 3 seeds -0.00030 / +0.00126 / -0.00268, mean -0.0006 | `REPORT_SDMLLMSTORE.md` §5 block 3 | TEST: SDM-only needs `sdmonly_none` beside it (wave 1 has it) |
| 9 | A dense MLP of equal parameters beat the broken store | sdm minus sdm_mlp +0.0209 | `REPORT_SDMLLM_S0.md` §5 | KEEP the dense control |
| 10 | Frozen random values are ignored by training | sdm_frozen 1.6074 vs nostore 1.6054; its zero_read changes 0.0002 | same | do not freeze values |
| 11 | 16,384 locations (on the broken path) lost more | sdm_big minus nostore +0.0196; usage 9.4% | same | TEST: big tables at qgrad 0 never run before wave 1 |
| 12 | Hashed n-gram (2,3) table instead of a similarity address helps a little | -0.00341 / -0.00304 / -0.00324 vs nostore | `REPORT_SDMLLMSTORE.md` §5 | TEST as an extra head |
| 13 | A context-free per-token table beats the n-gram table | (1,) minus nostore -0.0083 / -0.0082 / -0.0075; (2,3) minus (1,) +0.0049 / +0.0051 / +0.0043 | same | TEST: an untied input table |
| 14 | At 300M, d 256, the store read, the per-token table and no store tie | S 1.50641, U 1.50680, N 1.50670; S minus N -0.00029 (SE 0.00026) | ledger 2026-10-01 11:45Z; `runs_spark24/*.result.json` | the read has never beaten no store |
| 15 | The per-token table's 20M gain is gone at 300M, d 256 | U minus N +0.0001 (s0), -0.00137 (s1) | ledger 06:25Z, 21:51Z | 20M results do not carry |
| 16 | The per-token table helps at 3B, d 512, narrow address | VA minus VD -0.01124 (SE 0.00042) | ledger 2026-10-01 11:50Z | TEST at the long run |
| 17 | The SDM read loses to no store at 3B, d 512, narrow address | VS 1.48547 vs VD 1.48046: +0.0050 | ledger 2026-10-01 16:45Z (VS result file lost with the box) | caution |
| 18 | Attention is worth 0.154 at 300M | Q minus N -0.15447 (SE 0.00097) | ledger 11:45Z | the context window is the ceiling |
| 19 | The wide address (8 back, 5 averages) moves the family | W minus U -0.00827 (SE 0.00026); W s1 minus U s1 -0.00654 (SE 0.00027) | ledger 16:44Z, 21:34Z | KEEP; TEST wider |
| 20 | Wide address at 300M beats narrow at 1B (d 512) | W2 minus C -0.02629 (SE 0.00045) | ledger 21:34Z | address beats tokens |
| 21 | Width pays | W2 minus W -0.06875 (d 512 vs 256); NWL minus NW -0.02423 (d 768/600M vs d 512/300M) | ledger 21:34Z; 2026-10-03 06:30Z | scale width in step 1 |
| 22 | Wide address d 512 300M: read, table, none tie three ways | SW 1.42995, NW 1.43012, W2 1.42977; two-seed mean SW minus NW +0.00014 (SE 0.00021) | ledger 2026-10-02 07:00Z, 13:12Z | the read adds nothing at 512 |
| 23 | At d 768 the read is ahead by a small amount, one seed | SWL minus NWL -0.00304 (SE 0.00031, z -9.8); CHAT -0.02127 | ledger 2026-10-03 06:30Z | the first sign the read helps with width |
| 24 | The trained model leans on its read even when no store ties | zero_read: S +0.170, SW +0.185 / +0.200, SWL +0.139, SWLC +0.182 | result files | zero_read cost is not a gain measure |
| 25 | Locations used fall with training | see §3d figure | result files | CHANGE: instrument it |
| 26 | Chat: half web rows keep TEST; chat-only buys chat and loses web | WCO minus WCC: CHAT -0.0376, TEST +0.186 | ledger 2026-10-01 23:28Z | KEEP the web half in SDM CHAT |
| 27 | Chat: four times the chat tokens pays | VACC4 (400M) 1.10385 vs VCC (100M) 1.13365: -0.0298 | `runs_vast/*.result.json` | step 3 budget |
| 28 | Best chat number in the record reads no SDM | VCCC4 (d 1024, per-token table, 400M chat) CHAT 1.02350; weights lost | ledger 16:45Z | a second bar beside 1.06386 |
| 29 | Guy fine-tune without web rows forgets the web | FineWeb TEST 1.7136 -> 3.5029 (WG0); half web rows -> 1.8313 (WG50), guy TEST 0.2748 -> 0.2303 | ledger 2026-10-01 11:58Z | KEEP row mixing in step 4 |
| 30 | Bootstrap SE understates rerun noise about ten times | fixed-seed rerun moved -0.0026 (SE 0.00017) | `REPORT_SDMLLMSTORE.md` §6 | CHANGE the decision rule (§6) |
| 31 | The transformer yardstick at 20M is loose | 5-seed spread 0.0536 | same §5 | do not use 20M qwen as a bar |
| 32 | int8 export of the token table needs column scales | per-row only: cost 0.019 bpb (one column max 15 vs median RMS 0.3); with column scales 0.0017 | ledger 2026-10-01 06:25Z | export note |
| 33 | int8 export costs about 3 points of top-1 agreement over bf16 | int8 0.9365; bf16 vs fp32 0.9699 (sdmwide768chat) | ledger 2026-10-03 03:20Z | export note |

### 2b. Stores read with the teacher's state (nanochat base-d20; every number reads the teacher's own `x_final`)

| # | finding | number (MEASURED) | source | for SDM-only |
|---|---|---|---|---|
| 34 | Store content carries the gain | values frozen to random: +0.810 bpb, z +52.5 | `sparsestar-proper/MEASURED_track4_234_*` | values must be trained |
| 35 | Soft values move the best k down to 32 | soft k 32 1.319804; k 4,096 1.325162; soft@32 beats soft@4096 by 0.005357; rises from k about 128 | `MEASURED_track4_237_*` §2 | TEST k 16/64 (wave 1) |
| 36 | A teacher-free address loses most of the gain | token mash address 2.035848 vs teacher address 1.319804: +0.716044, z 70.27 | `MEASURED_track4_239_*` §2 | the address is the model |
| 37 | Unweighted bundling of more tokens hurts | mash n 4 beat n 8 on CALIB by 0.101973 | same §5 | weight recent tokens; do not bundle |
| 38 | An untrained cleanup hop makes it worse every time | hop 1 +0.054381 (z 20.67), hop 2 +0.035917 (z 29.73); recovery -7.6%, -12.6% | same §3 | do not iterate an untrained read |
| 39 | The hop converges, to an attractor that is not the answer | cos(q_t, q_t-1) 0.9999 by t 5; one hop raises cos to truth +0.098 and costs +0.145961 bpb | `MEASURED_track4_240_*` §0 | risk for a shared table across hops |
| 40 | Zeroing 16 rogue dimensions before addressing | -0.1839 bpb at k 4,096 (z -25.8) | `MEASURED_track4_238_*` §0 | KEEP query BN (per-dimension scale) |
| 41 | The selection effect of a better address is positive, the score frame can be negative | selection +0.0013 to +0.0065 for 5 of 5 operators; score frame negative for 4 of 5 | `MEASURED_track4_224_*` | keep scoring and selecting in one learned space |
| 42 | A trained no-store readout 132.5x smaller beats the store | r 128: 1.116828 vs store 1.188416; hard-target readouts never beat the store at any rank (best 1.262508) | `MEASURED_track4_245_*` §1 | every claim needs a no-store control |
| 43 | Hashed 3-gram address is starved | 1.18 facts per exact 3-gram key; a shuffled-member control 0.02715 vs 0.02660 | `WIKI/theory/173` §4 | exact long n-gram keys are empty at our scale |
| 44 | Shorter keys and a backoff address win | n 1 0.10560 vs n 3 0.05613; BACKOFF5_min8 0.11563, x6.72 the incumbent's margin, 30.5% fewer cells | `WIKI/theory/173` §5 | TEST a multi-order exact head |
| 45 | The best context length grows with data | backoff mean length = -0.4834 + 0.3315 log10(facts), r² 0.992, 98,016 to 6,273,024 facts | `MEASURED_ADDRESSSCALE_*` §4 | supports multi-order |
| 46 | n at least 4 falls below the no-address floor | n 4 0.04827 vs floor 0.04852 at 392,064 facts | same §2 | exact keys longer than 3 are a loss at this scale |
| 47 | A backoff rule that helps a hashed store hurts an exact table | min-support 2: -7.5%; 8: -24.3% on a drafter | `track3-semiotic-codebook/MEASURED_MINSUPTRANSFER_*` | evidence against a naive backoff head |
| 48 | Linear value maps buy nothing; non-linear ones lose | 21 of 21 invertible maps give identical 0.1411; non-linear -1.87 to -5.05 points | `WIKI/theory/170`, `171` | do not add value post-processing for its own sake |
| 49 | No geometry measure survives behaviour-preserving maps | effective rank, recall@1, DCI, MIG all move under rotation or scaling | `WIKI/theory/174` | judge by bpb and ablations only |

### 2c. Tables injected into a frozen model (Track 3, nanochat base-d20)

| # | finding | number (MEASURED) | source | for SDM-only |
|---|---|---|---|---|
| 50 | A train-free injected memory hurts, and real equals random | present minus absent +2.439e-4 (z +7.14); random vs real z -0.35 | `track3-semiotic-codebook/MEASURED_ENGRAMCONTENT_2026-08-31.md` | values must be learned |
| 51 | Pooled values collapse into one cone; more members per row makes it worse | mean pairwise cosine 0.9392 at 6 members per row, 0.9990 at 383 | `MEASURED_ENGRAMCONTENT_PHASE2_*` §1, §2 | value rows trained by gradient avoid pooling; watch value-row cosine |
| 52 | A frozen random table is worth nothing; a learned one is worth a little | frozen random |z| at most 1.47 over 512x sizes; learned-then-frozen -0.0786 (z -3.57); best learned -0.1086 = 4.6% of what the base buys | `MEASURED_FROZENTABLE_*` §2-4 | KEEP trained values |
| 53 | A learned table helps only where populated, and the curve is an inverted U | helps from 1,875 down to 14.7 positions per slot; not distinguishable from zero at 7.41 (-0.00375, z -0.74); hurts at 3.96 (+0.0108) and 1.38 (+0.0141) | `MEASURED_OCCUPANCYINVERSION_*` §3 | see §3c arithmetic |
| 54 | Half of the far-end damage is in the co-fitted readout | with the table switched off for 7,981 of 8,000 probes the arm still costs +0.00745 (z +4.41) | same §6 | a huge table can distort the rest of the model |
| 55 | Store edits are not local | suppressing one token took 262,144 of 819,200 rows (32.0%); reach equals vote width | `MEASURED_NARROWREADREMATCH_*` §2 (reproducing T3-84) | no editability claim |

### 2d. Kanerva's memory rebuilt (SETTLE, binary addresses, counters)

| # | finding | number (MEASURED) | source | for SDM-only |
|---|---|---|---|---|
| 56 | Capacity grows with locations only while the activation fraction shrinks | 10% noise capacity about M^0.96 under the SNR radius; a fixed 2.6% activation saturates from 10,000 locations | `thermosim/runs/sdmscale/REPORT_SDMSCALE.md` §3 | KEEP k fixed as M grows (product keys do this) |
| 57 | Reads fail by races between whole patterns, not by independent noise | every cue whose rival shares more locations fails (55 of 55) | `runs/sdmradius/REPORT_SDMRADIUS.md` §4 | a soft top-k read blends rivals; expect mixtures |
| 58 | The soft cut-off is attention at a low temperature | softness 1: 99.2% agreement with softmax attention, 0% recall | `runs/softsdm/REPORT_SOFTSDM.md` §6 | softness is not a free capacity knob |
| 59 | Iterated reads from far cues stall on a mixture | at load pT 2.8 about 1,400 patterns share the vote; 16 of 200 cues land | `runs/sdmtrack/REPORT_SDMTRACK.md` §3 | same warning as 39 |

### 2e. Memory layers in the literature, as recorded in our wiki (PAPER CLAIM)

| # | claim | number | source | for SDM-only |
|---|---|---|---|---|
| 60 | Query BatchNorm fixes dead slots at 1M slots, not needed at 16k-65k | usage 25.8% -> 80.3% at 1M; 16k perplexity 22.8 -> 23.0 with BN | `wikis/WIKI_SDR/wiki/143-*` §2 | KEEP BN; it was needed in our runs (2.7% before BN) |
| 61 | Flat keys leave most slots unused; product keys use all | 10% of keys used at 147k flat; 100% product | `WIKI_SDR/wiki/142-*` §2 | KEEP product keys |
| 62 | Activation density matters more than slot count; field standard about 128 per token | k 32 x 4 heads | `WIKI/research/HARVEST_sizing_memory-layers_2026-07-19.md` | the centre has 4 hops x 32 = 128 |
| 63 | Key width half the value width | d_a = d/2 | same, knob 3 | TEST d_a 128 |
| 64 | Every memory layer in the literature sits beside a compute-heavy backbone | 1:10 to 48:1 stored ratios | same, knob 5 | SDM-only has no backbone; untested territory |
| 65 | Large k early, then shrink, prevents dead units | e.g. K 4096 -> 50 over the first 10% of steps | `WIKI_SDR/wiki/142-*` §4 | TEST k anneal |
| 66 | Aux-loss-free selection bias balances use without gradient interference | bias on selection scores only, updated by ±γ from load | same §5 | TEST if usage collapses |
| 67 | Memory Layers at Scale uses qk-norm for small base models and shares one pool across 3 layers | stated, no key/value init, no store lr, no store wd given | same §6 | our store recipe is our own measurement |
| 68 | A top-k blended read is listwise; sharpening it toward argmax costs capacity | about 16x at d 256 | `WIKI_SDR/wiki/100-*` §3 | do not anneal softness to zero |

---

## 3. THE ADDRESS

### 3a. What the address is in SDM-only, and what it is not

```
   context ──▶ features (8 back + 5 averages) ──▶ x ──▶ q_1 = BN(W_q,1 n(x)) ──▶ wake 32 of 65,536
                     ▲                                  q_2 = BN(W_q,2 n(x + r_1)) ...
                     └── the ONLY place new context enters
```

Every hop's address is a linear map of the same features plus earlier reads. A hop can recombine what
the features hold. It cannot fetch a token the features dropped. So the measured context limit of the
family (row 18, 0.154 to attention) is also the limit of SDM-only.

### 3b. What we measured about addresses

**Wider windows help; nothing else has moved the family.**

```
   T H E   C O N T E X T   L A D D E R   (TEST bpb, MEASURED)

   d 256, 300M tokens, Spark
     4 back + 2 averages, no store (N)       1.50670  ─────────────────────────╮
     4 back + 2 averages, SDM read (S)       1.50641  ─────────────────────────┤ tie
     8 back + 5 averages, table (W)          1.49853  ───────────────────────  -0.0083
     4-layer transformer (Q)                 1.35223  ─────────                 -0.1545

   d 512
     4 back + 2 averages, table, 1B (C)      1.45606
     8 back + 5 averages, table, 300M (W2)   1.42977   -0.0263 on a third of the tokens
```

- The 239 cell measured how much the address matters when the reader is fixed: replacing the
  teacher's state with a free token mash cost 0.716044 bpb, 72.7% of the store's whole gain. With
  an unweighted bundle, 4 tokens beat 8 by 0.101973 on CALIB: each extra bound token added crosstalk.
  SDM-only weights its tokens through W_x and the moving averages, which is the right family.
- 16 back tokens and 8 averages (W3) were sealed as P23 (gain 0.002 to 0.010 over W) and cancelled
  before running (ledger 2026-10-01 21:32Z). It is the cheapest untested address lever we hold.

**Exact n-gram keys are starved; short keys and backoff win.** On a 391,680-fact store, an exact
3-gram key holds 1.18 facts, and slot members are strangers (a shuffled-membership control matches
the real address, 0.02715 vs 0.02660). Realised leave-one-out ability: n 1 0.10560, n 2 0.08425, n 3
0.05613; a min-support backoff 0.11563 (`WIKI/theory/173` §4-5). Across 64x of data the backoff's mean
context length follows

    L(facts) = -0.4834 + 0.3315 · log10(facts)          r² = 0.992 over 98,016 to 6,273,024 facts

Plain reading: the best exact context grows by about a third of a token for every tenfold more data.
EXTRAPOLATION, not measured: at 2e7 facts L is 1.94 tokens; at 3e9 it is 2.66. Those facts were
(context, teacher argmax) pairs, not next tokens, so the curve's level does not transfer; its shape
(short exact keys, longer only where data supports it) is the transferable part.

**Standardising the address space is a large lever.** Zeroing 16 rogue dimensions before addressing
bought 0.1839 bpb (238). Per-dimension scale was 89% of the closure in 170. In the language models a
smoke run without query BN picked 2.7% of locations (S0 §4). SDM-only already uses BN without affine
terms on every hop. KEEP.

### 3c. Usage, and the collapse that training produces

The record measures "locations used" as the fraction of locations picked at least once in the top k
over 64 TEST windows (64 x 256 positions x 32 picks = 524,288 picks). For a 3,600-row store a uniform
picker would use about 100%.

```
   L O C A T I O N S   U S E D   (last hop, 64 TEST windows, MEASURED)

   smoke, no query BN, 20M          2.7%   ▏█
   S0 sdm s0 / s1 / s2, 20M         39.6 / 43.7 / 47.5%  ▏████████████
   store winner s0 / s1 / s2, 20M   45.8 / 45.9 / 47.3%  ▏████████████
   S d 256, 300M                     9.3%  ▏███
   SW d 512, 300M, s0 / s1          22.9 / 17.4%  ▏██████
   SWL d 768, 600M                  25.3%  ▏███████
   SWLC (chat fine-tune)            23.8%  ▏██████
   sdm_big 16,384 rows, 20M          9.4%  ▏███
```

Sources: `runs/*.result.json`, `runs_spark24/spark_S_s0_300M.result.json`, `runs_sdmchats/*.result.json`
(field `locations_used_fraction_last_hop`), `REPORT_SDMLLM_S0.md` §4.

What this means for SDM-only:
- Usage falls with training, with query BN on. A 65,536-row table per hop could end up mostly unused
  at the long-run budget.
- ARITHMETIC: the census saturates differently by table size. With 524,288 picks a uniform picker uses
  1 - exp(-524,288 / M) of the rows: 99.97% at 65,536 rows and 86.5% at 262,144 rows. Compare `p0_A_n512`
  with the centre only after dividing by that ceiling.
- The census counts a row picked once the same as a row picked a thousand times. PKM's access-KL
  (PAPER CLAIM, `WIKI_SDR/wiki/143` §1) catches concentration:

      z_i = (sum over positions of w_i) / (sum over positions and rows of w)
      access_KL = log M + sum_i z_i log z_i

  Plain reading: 0 when every row carries equal weight; log M when one row carries everything.

ARITHMETIC: at the centre, 20M tokens x 32 reads / 65,536 rows = 9,766 reads per row per hop if picks
were uniform. SWL's shared 3,600-row store saw 600M x 2 hops x 32 / 3,600 = 10.7 million per row. Each
SDM-only row is updated about a thousand times less often. Track 3 found a learned table helps only
where rows are populated (row 53; the sign changes near 7.41 positions per slot in one fit, and the
same study refutes occupancy as the axis on its own, so this is not a universal constant).

### 3d. Proposals for the SDM-only address (PROPOSAL; none run)

| id | proposal | evidence for | evidence against | cost |
|---|---|---|---|---|
| A1 | Wider window: 16 back tokens, 8 averages (0.3 0.5 0.7 0.8 0.9 0.95 0.97 0.99) | the only lever that moved the family (rows 19, 20); P23 sealed and never run | 239: more unweighted tokens hurt; diminishing returns predicted | flags only (`--n-back`, `--decays`) |
| A2 | An exact n-gram hash head beside the product keys: hashed 2-gram (and 1-gram) rows added to x before hop 1, Engram-style | (2,3) hash table -0.0034 vs no store in 3 seeds (row 12); n 1 and n 2 beat n 3 (row 44); exact keys are cheap to compute | lost to the per-token table by 0.0049 (row 13); exact-table backoff hurt a drafter (row 47); tables tie no store at 300M (row 14) | reuse `NgramStore` from `track4_sdmllm_models.py` |
| A3 | Multi-order address per hop: hop 1 reads only the last 1-2 tokens, later hops the full window | backoff wins by matching context length to evidence (rows 44, 45) | no measurement of per-hop context split in any model | small model change |
| A4 | Usage balancing: an aux-loss-free bias on selection scores, or k annealed from 128 to 32 over the first 10% | usage collapse measured (§3c); both methods recorded in `WIKI_SDR/wiki/142` | PAPER CLAIM only; the wiki warns the small-M literature may not transfer; the read tied no store at 20M with 40-48% usage | code: a per-row bias buffer, or a k schedule |
| A5 | Address orthogonality term on sub-keys | recorded result that cleanup pays only on a trained quasi-orthogonal address (CLAUDE.md, citing `2602.21467`) | `WIKI_SDR/wiki/142` §1: mechanism changes beat loss regularisers in the one direct test; PAPER CLAIM on a 10x10 grid world | a loss term; lowest priority |
| A6 | Key width half the value width: d_a 128 | Memory Layers at Scale and PKM (row 63); halves the score cost | at d_a 128 each half is 64 wide; no local measurement | flag only (`--d-a 128`) |
| A7 | An untied input table beside the tied head | per-token table -0.008 (20M, 3 seeds), -0.0112 (3B, d 512) | tie at 300M d 256 (2 seeds) and at 300M d 512 wide | reuse the `sdm_ngram` orders (1,) table, or an untied embedding |

---

## 4. THE VALUES AND THE READ

### 4a. The read and the gradient switch

    w_m = sigmoid( (s_m - θ) / softness ),   θ = the k-th best score (detached)
    r = (1/k) Σ_{m in top 2k} w_m v_m

Plain reading: about k rows wake; rows near the cut-off count partly; the read is the average of their
value rows.

    g(x) = stopgrad(x) + α · (x - stopgrad(x))

Plain reading: the forward value is x; only a fraction α of the gradient flows back into x through the
address. α = qgrad.

### 4b. What we measured

| knob | measured | source | SDM-only setting | verdict |
|---|---|---|---|---|
| qgrad | 0 recovers 0.0235 over 1; 0.1 gives +0.00046; at qgrad 1 with the store recipe +0.124 | `REPORT_SDMLLMSTORE.md` §5 | 0 | KEEP; `p0_A_qg1` is a risk arm |
| softness | 0.25 to 8 within 0.0010 at qgrad 0; 0.25 numerically best | same, block 2 | 0.25 | KEEP; do not sharpen to 0 (row 68) |
| k | never swept in the language models; in the teacher store soft values peak at k 32 and k above about 128 costs | `MEASURED_track4_237_*` §2 | 32 | TEST (wave 1 has 16, 64) |
| value init | zero vs randn·0.02√k part of a 0.0012 package at qgrad 0 | `REPORT_SDMLLMSTORE.md` §5 | zero | KEEP |
| store lr | x3 part of the same 0.0012 package; destructive with qgrad 1 | same | x3 | KEEP; `p0_A_slr10` may reproduce value-norm growth (row 3) |
| store weight decay | 0 part of the same package; S0 used 0.1 | same | 0 | KEEP |
| hops | always 2 through one shared store in every earlier model; never swept | result files | 4, one table per hop | new territory; wave 1 tests 2 and 8 |
| one table shared by hops | the untrained re-read converges to a shared attractor and costs bits (rows 38, 39); trained shared hops never compared with separate tables | `MEASURED_track4_239_*`, `240` | separate | KEEP separate; `p0_A_share` tests it |
| frozen or random contents | ignored or inert (rows 10, 50, 52); learned contents matter (rows 34, 52) | as cited | trained | KEEP |
| readout after the hops | every earlier model had one; it absorbed what the store learned | `REPORT_SDMLLMSTORE.md` §7 | none | TEST (`p0_A_ro`) |

### 4c. Cleanup iterations, and why trained hops differ

```
   T H E   U N T R A I N E D   H O P   (239, store-only TEST bpb, k 32)

   teacher address        1.321097  ◆
   token mash address     2.149085  ◇───────────────────╮
   mash + 1 hop           2.366652  ◇──────────────────────╮   worse
   mash + 2 hops          2.484944  ◇────────────────────────╮ worse again
   unigram base           2.304655  (for scale)
```

The untrained hop re-queries with the similarity-weighted centroid of the rows it found. From a rough
cue those rows are an arbitrary sample, and their centroid drifts toward the store's dense core (239
§3). 240 then showed the iteration does converge, from three starts, to endpoints at pairwise cosine
0.956 to 0.986, and that landing there costs +0.145961 bpb even when it moves the cue 0.098 closer in
cosine.

SDM-only's hops are trained: each has its own query map and its own table, and the read is added to x
rather than replacing the query. So 239 and 240 do not predict that hops hurt. They predict two things
to watch: a shared table across hops (`p0_A_share`) is closer to the iterated read and should lose; and
any per-hop usage that collapses onto the same few rows on every hop is the attractor in another form.

### 4d. The gradient clip at long budgets (MEASURED from the training logs)

Pre-clip global gradient norm, logged every 200 steps, clip at 1.0 (`track4_sdmllm_train_one_arm.py`
line 221). Share of logged steps above 1.0 in the second half of training:

```
   20M  store winner (qgrad 0)     0%   ▏
   20M  nostore                     0%   ▏
   20M  storeopt (qgrad 1)        100%   ▏██████████  max 28.15
   300M N  (no store, d 256)       72%   ▏███████
   300M S  (SDM read, d 256)       89%   ▏█████████
   300M SW (SDM read, d 512)       55%   ▏██████
   300M NW (no store, d 512)       30%   ▏███
   600M SWL (SDM read, d 768)      54%   ▏█████
   600M NWL (no store, d 768)      74%   ▏███████
   3B   VD (no store, d 512)       97%   ▏██████████  median 2.81, max 26.0
   3B   VA (token table, d 512)  99.8%   ▏██████████  median 5.31, max 30.1
```

Logs: `runs/*.log`, `runs_spark24/*.log`, `runs_sdmchats/*.log`, `runs_vast/*.log` (VS's log is partial).

The no-store arms clip as often as the store arms, so this is a property of the family at long budgets,
not of the read. Where the norm sits was measured only at 20M on the broken path (the embedding
dominated; `REPORT_SDMLLMSTORE.md` §3). For SDM-only's long run the effective step size in the second
half is set by the clip. That may be harmless or may be a lost lever; nothing in the record decides it.

---

## 5. WHAT FAILED AND MUST NOT BE REPEATED

| what | the measurement that killed it | source |
|---|---|---|
| A hard cut-off with the query gradient flowing into shared features | +0.017 to +0.025 bpb in 3 seeds; 16-20% of steps clipped | `REPORT_SDMLLM_S0.md` §5; `REPORT_SDMLLMSTORE.md` §3 |
| Fast store optimiser settings while the query gradient flows | +0.12388 bpb; 63% clipped; value rows to max norm 26.6 | `REPORT_SDMLLMSTORE.md` §5 |
| Frozen random value tables | ignored (sdm_frozen 1.6074 vs nostore 1.6054); inert at every size (|z| at most 1.47) | S0 §5; `MEASURED_FROZENTABLE_*` §2 |
| Train-free injected memory (values mined, never trained) | hurts (+2.439e-4, z +7.14) and real equals random | `MEASURED_ENGRAMCONTENT_2026-08-31.md` |
| Untrained cleanup hops from a rough address | +0.054 then +0.036 bpb; converges to a shared attractor that is not the answer | `MEASURED_track4_239_*`, `240` |
| Unweighted bundling of more context tokens into one address | n 8 worse than n 4 by 0.102 | `MEASURED_track4_239_*` §5 |
| Exact n-gram keys of 4 or more tokens at our data sizes | below the no-address floor at every size measured | `MEASURED_ADDRESSSCALE_*` §2 |
| "Fewer slots for better values" when values are pooled | pooling 6 -> 383 members raised entry cosine 0.9392 -> 0.9990 | `MEASURED_ENGRAMCONTENT_PHASE2_*` §2 |
| A wide aperture with soft values | k 4,096 worse than k 32 by 0.005357 (z -4.65) | `MEASURED_track4_237_*` §2 |
| Non-linear value maps to "clean" a store | -1.87 to -5.05 points against doing nothing | `WIKI/theory/171` |
| Geometry measures as success criteria (effective rank, recall@1, DCI, MIG) | noise scores 838.9 to 937.0 effective rank vs 67.4 for the best real arm; all moved by behaviour-preserving maps | `WIKI/theory/169`, `174` |
| Claiming a store gain without a no-store control | a trained readout 132.5x smaller beat the store | `MEASURED_track4_245_*` |
| Reading zero_read cost as the store's contribution | zero_read +0.139 on SWL while no store sits 0.003 behind | ledger 2026-10-03 06:30Z |
| Chat fine-tune on chat alone | CHAT -0.0376 bought TEST +0.186 | ledger 2026-10-01 23:28Z |
| Character fine-tune without web rows | FineWeb TEST 1.71 -> 3.50 | ledger 2026-10-01 11:58Z |
| Trusting a 20M winner at scale | per-token table -0.008 at 20M, tie at 300M | `REPORT_SDMLLMSTORE.md` §5; ledger 21:51Z |
| Trusting a window-bootstrap SE as the noise floor | rerun noise about ten times the SE | `REPORT_SDMLLMSTORE.md` §6 |

---

## 6. Measurement rules this record supports

- **Seeds.** At 20M on one machine, a difference under about 0.005 needs a second seed: measured seed
  spreads are 0.0011 (nostore), 0.0009 (per-token table) and 0.0051 (the store winner), and a fixed-seed
  rerun moved 0.0026. At 300M on the Spark the spreads were 0.00002 (W), 0.0017 (U), 0.0002 (N),
  0.0025 (SW) and 0.0018 (NW). The plan's rule (0.003 and window-bootstrap z of 3) does not exclude seed
  noise on its own.
- **Scale.** Confirm a 20M winner at 100M or more before it shapes the long run (row 15).
- **Usage.** Report usage per hop divided by its uniform ceiling, and access-KL beside it.
- **Shuffle ablations mean different things in the two stores.** In `SdmStore` `shuffle_keys` permutes
  the key rows (the address changes). In `ProductKeyStore` it permutes the value rows (the address is
  kept, the content is scrambled). Do not compare the two numbers.
- **The usage field in the S0 trainer is the last hop,** although a code comment says hop 0
  (`track4_sdmllm_train_one_arm.py` line 270 against line 277).

---

## 7. Cheap tests for the Spark, ranked (PROPOSAL)

Minutes are ARITHMETIC at about 46,700 tokens a second for d 256 (the coordinator's measured rate):
20M tokens is about 7.1 minutes of training, 100M about 36. Scoring and ablations add time that was not
measured here. None of these duplicates a wave-1 arm.

| rank | test | the one change | compared with | tokens | minutes | why |
|---|---|---|---|---|---|---|
| 1 | W3 address | `--n-back 16 --decays 0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99` | `p0_A_c`; and the same change on `p0_A_none` to separate address from read | 20M x 2 | about 15 | the only family lever, untested at the next width (rows 18-20) |
| 2 | Centre and no-read at 100M with usage logged per hop every eval | log usage and access-KL per hop during training (trainer change) | `p0_A_c` vs `p0_A_none` at 100M; usage curve vs §3c | 100M x 2 | about 72 | 20M rankings flip; usage collapse appears with training (rows 15, 25) |
| 3 | Exact 2-gram hash head | an `NgramStore` (orders 2, hashed, zero init, store recipe) added to x before hop 1 | `p0_A_c` | 20M | about 8 | the exact address family won at short keys (rows 12, 44) |
| 4 | Untied input table | an `NgramStore` orders (1,) added to x0 | `p0_A_c` | 20M, then 100M if it wins | 8 / 36 | -0.008 at 20M and -0.011 at 3B, tie at 300M (rows 13, 15, 16) |
| 5 | Key width half | `--d-a 128` | `p0_A_c` | 20M | about 7 | literature default, cheaper read (row 63) |
| 6 | Clip at the long budget | clip 2.0 instead of 1.0 (trainer change) | the centre at 100M from test 2 | 100M | about 36 | most logged steps clip at long budgets in every arm (§4d) |
| 7 | k anneal | k 128 to 32 over the first 10% of steps (model change) | `p0_A_c` | 20M | about 8 | usage collapse; PAPER CLAIM fix (row 65) |
| 8 | Selection bias | per-row bias on selection scores, updated ±γ from use, outside backprop (model change) | test 2's centre | 100M | about 36 | only if test 2 shows usage falling below about a quarter of its ceiling |

Order of firing: 1 and 5 need only flags. 3 and 4 reuse code that already exists in
`track4_sdmllm_models.py`. 2, 6, 7 and 8 need small changes in files owned by others, so they go to the
channel as proposals first.

---

## 8. Sources

Reports and ledgers
- `experiments/track4/HISTORY_SDM_AND_SPARSESTAR_2026-10-04.md`
- `experiments/track4/sdmllm/REPORT_SDMLLM_S0.md`, `REPORT_SDMLLMSTORE.md`
- `SETTLE/SETTLE_CAMPAIGN_2026-09-30.md`: SDMLLM, SDMLLMSTORE, SPARK24, JIMOTHYVAST,
  SDMCHATS, SDMSCORE, SDMNEXT lines
- `experiments/track4/sdmllm/runs/`, `runs_spark24/`, `runs_sdmchats/`, `runs_vast/`: `*.result.json`
  and `*.log`
- `experiments/track4/sdmllm/track4_sdmonly_models.py`, `track4_sdmonly_train.py`,
  `track4_sdmllm_models.py`, `track4_sdmllm_train_one_arm.py`

The store series and the theory pages
- `experiments/track4/sparsestar-proper/MEASURED_track4_224_*`, `234`, `237`, `238`, `239`, `240`,
  `241`, `245`, `MEASURED_ADDRESSSCALE_*`
- `WIKI/theory/110`, `167`, `168`, `170`, `171`, `172`, `173`, `174`

Track 3
- `experiments/track3-semiotic-codebook/MEASURED_ENGRAMCONTENT_2026-08-31.md`,
  `MEASURED_ENGRAMCONTENT_PHASE2_2026-09-01.md`, `MEASURED_FROZENTABLE_*`, `MEASURED_OCCUPANCYINVERSION_*`,
  `MEASURED_NARROWREADREMATCH_*`, `MEASURED_MINSUPTRANSFER_*`

Kanerva's memory
- `SETTLE/runs/softsdm/REPORT_SOFTSDM.md`, `sdmkeys/REPORT_SDMKEYS.md`,
  `sdmscale/REPORT_SDMSCALE.md`, `sdmradius/REPORT_SDMRADIUS.md`, `sdmrefuse/REPORT_SDMREFUSE.md`,
  `sdmtrack/REPORT_SDMTRACK.md`, `sdmcoded/REPORT_SDMCODED.md`
- `SETTLE/kanerva/README.md`, `KANERVA_TERMS.md`

Wiki and harvests (PAPER CLAIM rows)
- `wikis/WIKI_SDR/wiki/100-sizing-a-trainable-topk-memory-under-the-listwise-capacity-law.md`,
  `142-training-dynamics-of-a-small-slot-memory-collapse-revival-and-schedules.md`,
  `143-training-a-slot-memory-from-scratch-bypass-collapse-and-diagnostics.md`
- `WIKI/research/HARVEST_sizing_memory-layers_2026-07-19.md`,
  `HARVEST_2026-07-29_knnlm-lineage-who-built-the-naive-table.md`,
  `HARVEST_2026-07-29_memory-frontier-grafting-srttt-engramv2.md`

Not opened, and so not used: the `track3_T3-84` and `T3-85` documents themselves (their figures are
quoted through `MEASURED_NARROWREADREMATCH_*`, which re-read them from the result files), the VS result
file (lost with its box; its score is the ledger's), and `TRACK3_PLAN.md` beyond its title.
