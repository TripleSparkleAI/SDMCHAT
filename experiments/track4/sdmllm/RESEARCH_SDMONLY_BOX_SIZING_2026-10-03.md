# BOXSIZING · what SDM BASE about $300 of Vast.ai trains in 24 to 48 hours · 2026-10-03

Lane BOXSIZING, a research lane. Nothing was trained, nothing was rented and nothing was spent. The board below
comes from read-only `vastai search offers` calls. Parameter and FLOP counts come from `track4_sdmonly_models.py`
itself (the model built on PyTorch's `meta` device, then `flops_per_token`), not from hand arithmetic.

**The question.** Can the SDM-only BASE train "this way" on four of the newest, biggest Vast boxes in about two
days, and what model size does that buy? The budget is about **$300 of Vast credit for the whole rented step**
(bench, the BASE run, its controls and a safety floor), spent within 24 to 48 hours, which is $6 to $12 an hour of
sustained burn. The account holds $352.41 (hand-over note; not queried here). One big multi-GPU box trains the
BASE; extra boxes are used only where a separate run earns its place (dense control, second seed).

Labels, as in `RESEARCH_BASEDATA_2026-10-04.md`:
- **MEASURED (ours)**: read from our own run logs or this lane's own board read.
- **MEASURED (elsewhere)**: a number another group published, read from its page.
- **SCALED**: our arithmetic on a measured number.
- **ESTIMATE**: a judgement. It must be measured before money is spent against it.
- **UNVERIFIED**: seen only second-hand.

## 0. The answer in one picture

```
   $300 of Vast, and what it buys for the SDM-only model (ESTIMATE speeds, MEASURED prices)

   four 8x B200 boxes, 48 h ......... $12,010   only 3 such machines exist on the board
   four 4x B200 boxes, 48 h ......... $5,910
   four 1x H100 SXM, 48 h ........... $403
   ─────────────────────────────────────────────── $352.41 in the account · $300 ruled
   PROPOSAL: one 4x H100 SXM, 24 h .. $168 cap    d 2048, 19.4B tokens, 3x compute-optimal
           + one 4x RTX 5090, 24 h ... $44 cap    four control and seed arms
           + bench and width ladder .. $15
                                       expected $116 to $205 · cap $227 · reserve at least $73

   => the model is small in compute (GPT-2-small class per token). Four of the biggest
      boxes would finish its compute-optimal run in a few hours and then sit idle.
      What binds is DATA, the BROWSER, and the fact that every speed off the Spark
      is still an estimate.
```

## 1. The board now

**Search time:** 2026-10-03T10:39:24Z (verified board) and 10:39:45Z (all hosts). **Queries, exact:**

```bash
~/.vast-venv/bin/vastai search offers 'rentable=true gpu_ram>=24' --limit 3000 --raw
~/.vast-venv/bin/vastai search offers 'rentable=true verified=any gpu_ram>=24' --limit 3000 --raw
```

The first query carries the CLI's default `verified=true` and returned **881 offers**; the second returned
**1,492** (873 verified, 324 deverified, 295 unverified). `gpu_ram` is in GB in the query (it is MB in the output).
**All-in price** = `dph_total + storage_cost x 200 GB / 730 h`, so a 200 GB disk is priced in. Only offers with
`disk_space >= 200` count. Offers are deduplicated by `machine_id` (cheapest offer per machine) before counting.
Reliability is the offer's `reliability` field. MEASURED (ours).

### The newest large cards, verified board

| card | GPUs | machines | min $/h all-in | median $/h | $/GPU-h at min | VRAM/GPU | median rel | NVLink field |
|---|---|---|---|---|---|---|---|---|
| **B300** (sm_103) | 1 | 1 | 9.99 | 9.99 | 9.99 | 268.6 GB | 0.995 | n/a |
| B300 | 2 | 2 | 19.93 | 21.47 | 9.96 | 268.6 GB | 0.996 | |
| B300 | 4 | 1 | 45.95 | 45.95 | 11.49 | 268.6 GB | 0.998 | 956 |
| B300 | 8 | 1 | 120.14 | 120.14 | 15.02 | 268.6 GB | 0.997 | 956 |
| **B200** (sm_100) | 1 | 3 | 7.76 | 8.86 | 7.76 | 179.1 GB | 0.994 | |
| B200 | 4 | 8 | 30.26 | 33.00 | 7.57 | 179.1 GB | 0.992 | 0, 900, 956 |
| B200 | 8 | 3 | 62.55 | 62.55 | 7.82 | 179.1 GB | 0.996 | 956 |
| H200 (SXM) | 1 | 3 | 5.11 | 5.19 | 5.11 | 140.4 GB | 0.998 | |
| H200 (SXM) | 8 | 2 | 56.07 | 57.38 | 7.01 | 140.4 GB | 0.999 | 478 |
| H200 NVL | 1 | 8 | 4.19 | 4.65 | 4.19 | 140.4 GB | 0.997 | |
| H200 NVL | 4 | 2 | 18.72 | 18.72 | 4.68 | 140.4 GB | 0.946 | 0, 478 |
| **H100 SXM** (sm_90) | 1 | 8 | 1.79 | 2.20 | 1.79 | 79.6 GB | 0.994 | |
| H100 SXM | 2 | 1 | 4.30 | 4.30 | 2.15 | 79.6 GB | 0.999 | 478 |
| H100 SXM | 4 | 3 | 6.99 | 8.57 | 1.75 | 79.6 GB | 0.999 | 478 |
| H100 SXM | 8 | **0** | | | | | | |
| **RTX PRO 6000 S** (sm_120) | 1 | 15 | 1.15 | 1.63 | 1.15 | 95.6 GB | 0.998 | none |
| RTX PRO 6000 S | 4 | 9 | 5.41 | 6.02 | 1.35 | 95.6 GB | 0.998 | none |
| RTX PRO 6000 S | 8 | 4 | 10.71 | 12.34 | 1.34 | 95.6 GB | 0.996 | none |
| RTX PRO 6000 WS | 1 / 4 / 8 | 13 / 4 / 2 | 1.38 / 5.63 / 11.23 | 1.48 / 6.49 / 11.56 | 1.38 / 1.41 / 1.40 | 95.6 GB | 0.99 | none |
| **RTX 5090** (sm_120) | 1 | 55 | 0.52 | 0.72 | 0.52 | 31.8 GB | 0.996 | none |
| RTX 5090 | 4 | 38 | 1.85 | 2.49 | 0.46 | 31.8 GB | 0.993 | none |
| RTX 5090 | 8 | 30 | 3.56 | 5.67 | 0.45 | 31.8 GB | 0.994 | none |
| A100 SXM4 40 GB (sm_80) | 1 | 22 | 0.51 | 0.92 | 0.51 | 40.0 GB | 0.994 | |
| A100 SXM4 40 GB | 8 | 4 | 3.65 | 5.84 | 0.46 | 40.0 GB | 0.986 | 0, 300 |

The all-hosts board adds a few: 8x H100 SXM (3 machines, min $16.38), 8x H200 (3, min $40.26), 8x B300 (2,
min $65.05), 1x RTX 5090 from $0.27. Those hosts are not verified, so they are not used below.

- **Nothing newer than B300 is listed.** No GB200 or GB300 rows exist on either board. MEASURED (ours).
- **The NVLink field is unreliable.** The cheapest 8x A100 SXM4 (machine 151759, $3.65) and 4x B200 rows report
  0, though SXM boards carry NVLink. Treat a 0 as "not reported", and check `nvidia-smi topo -m` on the box.

### Do four such boxes exist at once?

Distinct machines with at least that many GPUs of the card, reliability at least 0.97, 200 GB disk, and the sum of
the four cheapest real offers (never 4 x the cheapest):

| four boxes of | machines | yes? | 4 cheapest, $/h | 48 h cost | against $300 |
|---|---|---|---|---|---|
| 8x B300 | 1 | **no** | | | |
| 8x B200 | 3 | **no** | | | at $62.55 each, $12,010 even if a fourth appeared |
| 4x B200 | 8 | yes | 123.13 | **$5,910** | 20x over |
| 8x H200 | 2 | **no** | | | |
| 4x H100 SXM | 3 | **no** | | | |
| 1x H100 SXM | 7 | yes | 8.39 | $403 | 1.3x over |
| 4x RTX PRO 6000 S | 7 | yes | 23.25 | $1,116 | 3.7x over |
| 8x RTX 5090 | 29 | yes | 15.24 | $732 | 2.4x over |
| 4x RTX 5090 | 32 | yes | 7.80 | $374 | 1.2x over |
| 8x A100 SXM4 | 4 | yes | 28.73 | $1,379 | 4.6x over |

MEASURED (ours), prices at 10:39Z. **Four of the newest, biggest boxes for two days costs 20 to 40 times the
budget, and on the verified board four 8x B200 or 8x B300 machines do not exist at once.**

## 2. What "four boxes" can mean for one model

```
   (a) ONE box, data-parallel              (b) FOUR boxes, four different runs
   ┌ GPU0 ┐ ┌ GPU1 ┐ ┌ GPU2 ┐ ┌ GPU3 ┐      box A  BASE           box C  dense control
   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      box B  BASE seed 1    box D  second size
      └── all-reduce every step ──┘          no traffic between them at all
      NVLink: 478 to 956 GB/s               each run is its own model
```

- **Gradients cannot cross the internet every step.** One step of the d 2048 candidate carries 859M fp32
  gradients, **3.4 GB**. At an advertised 1 Gbps between hosts that is about **27 s a step** against roughly
  0.1 s of compute. SCALED from the parameter count and the board's `inet` fields. Our own links home measured
  0.34 to 0.9 Mbps (`JIMOTHY_VAST_SETTLE_2026-10-01.md`). Methods that sync rarely (local SGD, DiLoCo) exist;
  none is built or tested here, so they are out of scope.
- **So one model trains on one box.** The BASE needs (a): the biggest single multi-GPU box the budget affords.
- **Controls and seeds need (b)**, and they scale perfectly: one independent arm per GPU, as on 2026-10-01
  (four arms on a 4x RTX 5090, 97 to 99% busy each; MEASURED, ours).
- **Our need is one (a) box for the BASE plus one cheap (b) box for its controls.** This matches the navigator's
  ruling: one big box for the BASE, extra boxes only where a separate run earns its place.

## 3. Model sizing

### 3.1 Candidates, from the code

Every row is `SdmOnlyLM` with the trainer's defaults (`d_a 256`, `k 32`, 1 head, `n_back 8`, 5 averages, one
store per hop, no readout), V = 129,280, T = 256. Columns: the tied token table (embedding and head), the value
tables, everything else; forward FLOPs per token split into head and body. MEASURED (ours) from the code.

| d | hops | rows/hop | token table | value tables | rest | total | non-emb | fwd head | fwd body | head share |
|---|---|---|---|---|---|---|---|---|---|---|
| 768 | 4 | 65,536 | 99.3M | 201.3M | 8.73M | **309.3M** | 210.1M | 198.6M | 18.8M | 91.3% |
| 768 | 4 | 262,144 | 99.3M | 805.3M | 8.99M | 913.6M | 814.3M | 198.6M | 19.3M | 91.1% |
| 768 | 8 | 65,536 | 99.3M | 402.7M | 9.78M | 511.7M | 412.4M | 198.6M | 21.3M | 90.3% |
| 768 | 8 | 262,144 | 99.3M | 1,610.6M | 10.31M | 1,720.2M | 1,620.9M | 198.6M | 22.4M | 89.9% |
| 1024 | 4 | 65,536 | 132.4M | 268.4M | 14.96M | **415.8M** | 283.4M | 264.8M | 31.7M | 89.3% |
| 1024 | 4 | 262,144 | 132.4M | 1,073.7M | 15.22M | 1,221.3M | 1,089.0M | 264.8M | 32.2M | 89.1% |
| 1024 | 8 | 65,536 | 132.4M | 536.9M | 16.28M | 685.5M | 553.1M | 264.8M | 34.9M | 88.4% |
| 1024 | 8 | 262,144 | 132.4M | 2,147.5M | 16.80M | 2,296.7M | 2,164.3M | 264.8M | 35.9M | 88.1% |
| 1536 | 4 | 65,536 | 198.6M | 402.7M | 32.53M | **633.8M** | 435.2M | 397.1M | 67.8M | 85.4% |
| 1536 | 4 | 262,144 | 198.6M | 1,610.6M | 32.80M | 1,842.0M | 1,643.4M | 397.1M | 68.3M | 85.3% |
| 1536 | 8 | 65,536 | 198.6M | 805.3M | 34.37M | 1,038.3M | 839.7M | 397.1M | 72.2M | 84.6% |
| 1536 | 8 | 262,144 | 198.6M | 3,221.2M | 34.90M | 3,454.7M | 3,256.1M | 397.1M | 73.3M | 84.4% |
| 2048 | 4 | 65,536 | 264.8M | 536.9M | 56.92M | **858.6M** | 593.8M | 529.5M | 117.4M | 81.8% |
| 2048 | 4 | 262,144 | 264.8M | 2,147.5M | 57.18M | 2,469.4M | 2,204.7M | 529.5M | 118.0M | 81.8% |
| 2048 | 8 | 65,536 | 264.8M | 1,073.7M | 59.29M | 1,397.8M | 1,133.0M | 529.5M | 123.2M | 81.1% |
| 2048 | 8 | 262,144 | 264.8M | 4,295.0M | 59.81M | 4,619.5M | 4,354.8M | 529.5M | 124.3M | 81.0% |
| 3072 | 4 | 65,536 | 397.1M | 805.3M | 126.2M | 1,328.6M | 931.5M | 794.3M | 257.7M | 75.5% |
| 4096 | 4 | 65,536 | 529.5M | 1,073.7M | 222.7M | 1,825.9M | 1,296.4M | 1,059.0M | 452.5M | 70.1% |

**The shape that decides everything:** rows per hop barely change the FLOPs (d 768: 217.4M forward FLOPs per token
at 65,536 rows, 217.9M at 262,144) but quadruple the value tables. A token reads 2k = 64 candidate rows a hop. The
value tables are a large, cheaply read memory; the compute is the head.

### 3.2 Tokens per parameter, and which "parameter" counts

The training compute of one token, and a parameter count that has the same compute:

    C = 3 · F_fwd · D            N_c = F_fwd / 2

**In words:** training costs about three forward passes per token (forward plus a backward twice as dear), over D
tokens. N_c is the size of a dense transformer that does the same work per token (a dense model spends 2 FLOPs per
parameter per token). For this model N_c is far smaller than the parameter count, because the value tables are
read 64 rows at a time, not multiplied in full.

The three published rules from `RESEARCH_BASEDATA_2026-10-04.md` §3, applied literally, give very different budgets.
Billions of tokens; the last column is ours, Chinchilla's 20 per parameter counted on N_c:

| model | Chinchilla 20 x total | nanochat 10.5 x total | MiniCPM 192 x non-emb | MiniCPM 192 x (non-emb minus value tables) | **20 x N_c** |
|---|---|---|---|---|---|
| d 768, h4, 65k | 6.2 | 3.2 | 40 | 1.7 | **2.2** |
| d 768, h4, 262k | 18.3 | 9.6 | 156 | 1.7 | 2.2 |
| d 1024, h4, 65k | 8.3 | 4.4 | 54 | 2.9 | **3.0** |
| d 1536, h4, 65k | 12.7 | 6.7 | 84 | 6.2 | **4.6** |
| d 2048, h4, 65k | 17.2 | 9.0 | 114 | 10.9 | **6.5** |
| d 2048, h8, 65k | 28.0 | 14.7 | 218 | 11.4 | 6.5 |
| d 2048, h4, 262k | 49.4 | 25.9 | 423 | 11.0 | 6.5 |
| d 3072, h4, 65k | 26.6 | 14.0 | 179 | 24.2 | **10.5** |
| d 4096, h4, 65k | 36.5 | 19.2 | 249 | 42.7 | **15.1** |

SCALED from the code's counts. **No published rule was fitted to a model whose parameters are mostly a sparsely
read table.** Memory-layer work (Berges et al. 2024, "Memory Layers at Scale") compares models at equal FLOPs and
treats memory parameters as cheap, which is what 20 x N_c does. ESTIMATE: this document uses **20 x N_c as
"compute-optimal"** and reports the literal rules beside it. Under that rule, "under-trained" is below 0.7x,
"compute-optimal" 0.7x to 1.5x, "over-trained" above 1.5x.

### 3.3 The browser

Today's chat model `sdmwide768` has 115,802,880 parameters and its export is **165.9 MB** (MEASURED, ours,
`SETTLE_CAMPAIGN_2026-09-30.md`): **1.43 bytes a parameter**, not 1.0, so the export carries more than raw int8.
Its browser speed was 11.87 tokens a second on the M5. Each candidate's export, from 1.0 byte a parameter (pure
int8, a floor) to 1.43 (today's ratio, SCALED), and browser speed SCALED from 11.87 by forward FLOPs (ESTIMATE):

```
   export size (MB), pure int8 to today's ratio                       browser tok/s (ESTIMATE)
   today sdmwide768        116 ─ 166  ██                               11.9 (MEASURED)
   d 768  h4  65k shared   158 ─ 227  ██▌                              ~12
   d 768  h4  65k          309 ─ 443  █████       heavy                ~12
   d 1024 h4  65k          416 ─ 596  ███████     heavy                ~9
   d 768  h4 262k          914 ─ 1,309 ███████████████  too large      ~12
   d 1536 h4  65k          634 ─ 908  ██████████  too large            ~6
   d 2048 h4  65k          859 ─ 1,230 ██████████████  too large       ~4
   d 2048 h4 262k        2,469 ─ 3,538 ████████████████████████████████████  server only
```

- **The thresholds are ESTIMATES**: up to about 250 MB is today's experience; 250 MB to 1 GB is a heavy download
  that works on a desktop; above about 1 GB is too large for a page (a 32-bit WebAssembly heap tops out at 4 GB, and
  some browsers cap one ArrayBuffer near 2 GB; UNVERIFIED for our engine).
- **A browser BASE is d 768, 4 hops, 65,536 rows, with one shared store** (`--share-store`: 158.2M parameters, the
  same 217.4M FLOPs), or without it at 2 to 2.7 times today's download.
- **If the browser limit is lifted for it, the "big nice" server-side BASE is d 2048, 4 hops, 65,536 rows:**
  858.6M parameters, 0.86 to 1.23 GB exported, about 4 tokens a second in a browser and fast on any server GPU.

## 4. The two-day budget, box by box

### 4.1 How the speeds are estimated

Only the Spark has run the SDM-only model. Three measured points, alone on the GPU, B 32 x T 256, bf16, chunked CE,
dense AdamW (MEASURED, ours): d 256 / 4 hops / 65,536 rows **46,700 tok/s**; d 768 / 4 / 65,536 **22,800**;
d 768 / 4 / 262,144 **13,500**. A three-term fit through those points:

    t_token = c + 3 · F_fwd / R + N_all / (B_tok · u)

**In words:** each token costs a fixed overhead c, its compute at an effective rate R, and its share of the
optimiser's pass over every parameter N_all, which happens once per step of B_tok tokens at u parameters a second.
On the Spark: c = 10.7 µs, R = 36.8 TFLOP/s, u = 2.44G parameters a second. SCALED (an exact fit to three points,
so it has no test of its own). It explains the 262,144-row slowdown: the optimiser term grows from 15.4 µs to
45.6 µs a token while the compute term stays at 17.7 µs.

For a rented card the same three terms are set from the card's datasheet and our one rented measurement:

    t_step = 0.005 s + 3 · F_fwd · B_tok / (MFU · P_peak) + 36 bytes · N_all / (0.8 · BW)
    t_comm = 2 (n - 1) / n · 4 bytes · N_all / L            box rate = n · B_tok / (t_step + t_comm)

**In words:** a step pays 5 ms of launch overhead, its matmuls at a fraction MFU of the card's dense BF16 peak, and
a dense AdamW pass over every parameter at 80% of memory bandwidth. A data-parallel box also all-reduces the fp32
gradient of the whole model each step over a link of L bytes a second. Every input is labelled here:

| card | dense BF16 peak, TFLOP/s | memory BW, TB/s | MFU used | tokens/step/GPU | all-reduce link L |
|---|---|---|---|---|---|
| RTX 5090 | 209.5 (UNVERIFIED, BASEDATA §5) | 1.79 | 0.50 to 0.58 | 8,192 | 12 GB/s PCIe, no P2P (ESTIMATE) |
| RTX PRO 6000 S | 252 if FP32-accumulate is half rate, 480 if full (both UNVERIFIED) | 1.60 | 0.50 to 0.55 | 16,384 | 20 GB/s PCIe (ESTIMATE) |
| A100 SXM4 40 GB | 312 (vendor) | 1.56 | 0.40 to 0.55 | 8,192 | 100 GB/s NVLink (ESTIMATE) |
| H100 SXM | 989 (vendor) | 3.35 | 0.25 to 0.45 | 16,384 | 150 GB/s NVLink (ESTIMATE) |
| H200 NVL | 835 (vendor) | 4.8 | 0.25 to 0.45 | 16,384 | 60 GB/s bridges (ESTIMATE) |
| B200, B300 | 2,250 (MEASURED elsewhere: SemiAnalysis InferenceX, flopper.io; B300 keeps B200's BF16) | 8.0 | 0.20 to 0.40 | 16,384 | 300 GB/s NVLink (ESTIMATE) |

- **The anchor for MFU is our one rented run:** the older SdmLM shape at d 768 on one RTX 5090 ran 152,293 tok/s,
  98.5 TFLOP/s of model FLOPs, about 47 to 50% of the card's peak (MEASURED, ours, `runs_vast/`). With the 5 ms and
  optimiser terms above, its matmuls ran at about 0.58. The HBM cards get a lower MFU because this model's body is
  small matmuls and top-k, which stay launch-bound on a big GPU (ESTIMATE).
- **Cross-check:** the formula gives the d 768 / 4 / 65,536 shape 128k to 144k tok/s on one 5090, which is 5.6 to
  6.3 times the Spark's 22,800. Our one measured 5090-to-Spark ratio, on the older shape, was 4.85. So the 5090 row
  may be up to 20% optimistic. SCALED.

### 4.2 Tokens in 48 hours

`lo` uses the low MFU and the half-rate PRO 6000 case; `hi` the high ones. "Eff" is the data-parallel efficiency
from the formula's own comm term with plain DDP; "accum 4" is four micro-batches per optimiser step. ESTIMATE
throughout, priced at the verified board's minimum all-in rates (MEASURED, ours).

| box | $/h | 48 h $ | d 768 h4 65k, tok/s (eff) | d 2048 h4 65k, tok/s (eff) | eff at accum 4 (d 2048) | largest d to 1x CO in 48 h | to 3x |
|---|---|---|---|---|---|---|---|
| 1x RTX 5090 | 0.52 | 25 | 128k to 144k | 46k to 52k | | 2048 | 1024 |
| 4x RTX 5090 | 1.85 | 89 | 150k to 155k (0.29) | 54k to 56k (0.29) | 0.60 | 2048 | 1024 |
| 8x RTX 5090 | 3.56 | 171 | 268k to 276k (0.26) | 96k to 100k (0.26) | 0.56 | 3072 | 1536 |
| 1x RTX PRO 6000 S | 1.15 | 55 | 166k to 302k | 58k to 110k | | 2048 | 1024 |
| 4x RTX PRO 6000 S | 5.41 | 260 | 343k to 446k (0.51) | 122k to 161k (0.52) | 0.80 | 3072 | 2048 |
| 8x RTX PRO 6000 S | 10.71 | 514 | 634k to 807k (0.48) | 225k to 291k (0.48) | 0.78 | 4096 | 2048 |
| 1x A100 SXM4 | 0.51 | 25 | 144k to 182k | 52k to 67k | | 2048 | 1024 |
| 8x A100 SXM4 | 3.65 | 175 | 836k to 982k (0.72) | 302k to 359k (0.72) | 0.90 | 4096 | 3072 |
| **1x H100 SXM** | 1.97 | **94** | 313k to 494k | 113k to 186k | | 3072 | 2048 |
| **4x H100 SXM** | 6.99 | 336 | 1.01M to 1.44M (0.81) | 365k to 536k (0.81) | 0.94 | 4096 | 3072 |
| 1x H200 NVL | 4.19 | 201 | 277k to 451k | 99k to 168k | | 3072 | 1536 |
| 4x H200 NVL | 18.72 | 899 | 728k to 974k (0.66) | 261k to 357k (0.66) | 0.88 | 4096 | 2048 |
| 1x B200 | 7.76 | 373 | 537k to 880k | 204k to 363k | | 4096 | 2048 |
| 4x B200 | 30.26 | 1,453 | 1.79M to 2.64M (0.83) | 671k to 1.05M (0.82) | 0.95 | 4096 | 4096 |
| 8x B200 | 62.55 | 3,003 | 3.48M to 5.07M (0.81) | 1.30M to 2.01M (0.80) | 0.94 | 4096 | 4096 |
| 1x B300 | 9.99 | 480 | same as 1x B200 | | | 4096 | 2048 |
| 8x B300 | 120.14 | 5,767 | same as 8x B200 | | | 4096 | 4096 |

"Largest d" is the widest of d 768, 1024, 1536, 2048, 3072, 4096 (4 hops, 65,536 rows) whose 48-hour `lo` token
count reaches 20 x N_c (1x CO) or 60 x N_c (3x).

**Three things this table says, under its stated assumptions:**

1. **The asked-for 70 to 90% data-parallel scaling holds only on NVLink boxes.** On a PCIe RTX 5090 box the formula
   gives 0.26 to 0.29: the whole model's gradient, value tables included, crosses PCIe every step. Four micro-batches
   per step lift it to about 0.6, and `track4_sdmonly_train_ddp.py --table-sync rows` sends only the touched rows
   (lane MULTIGPU; not yet timed). ESTIMATE. **An 8x RTX 5090 is the cheapest-looking box and probably the worst
   one for a single run of this model.**
2. **Every box in the budget reaches compute-optimal for d 2048 or larger in 48 h.** Compute is not what binds.
3. **Tokens per dollar on d 2048 (`lo`), SCALED:** 1x A100 SXM4 366M, 1x RTX 5090 322M, 8x A100 SXM4 298M,
   1x H100 SXM 207M, 4x H100 SXM 188M, 1x RTX PRO 6000 S 182M, 1x B200 95M. The two cheapest are single small
   cards, which need about 4 to 5 days for d 2048 at 3x. **Among boxes that finish d 2048 at 3x inside 48 hours,
   the 8x A100 SXM4 is cheapest per token on paper, but only if it really has NVLink** (its field reads 0) and its
   MFU holds; the 1x H100 SXM is next, with no all-reduce at all, so it is the lowest-risk box.

## 5. THE TRADE-OFF AT A FIXED $300

### 5.1 The data we can choose from

| set | tokens (DeepSeek tokenizer) | on disk now? | label |
|---|---|---|---|
| `train_big` (FineWeb-Edu `sample/10BT`, part of 5 files) | **3.018B** | yes, sha256 in its provenance | MEASURED (ours) |
| FineWeb-Edu `sample/10BT`, whole | about **9.4B** (10B GPT-2 tokens x 0.938) | no | SCALED |
| FineWeb-Edu, full | about **1.22T** | no | SCALED (OPENDATA) |
| ClimbMix | about **335B** (non-commercial licence) | no | SCALED (BASEDATA) |
| HF Smol-Data 100B mix | about **94B** (100B GPT-2 x 0.938) | no | ESTIMATE (ratio taken from FineWeb-Edu) |

**More than 3.018B unique tokens needs a download and tokenizing pass on the rented box.** Our home links (0.34 to
0.9 Mbps) cannot carry shards. MEASURED (ours): `train_big` took 1,937 s to tokenize on the Spark (1.55M tok/s,
20 cores), and the 4x RTX 5090 box rebuilt all its shards (about 4.2B tokens with the chat sets) in about 41 min,
pulling Hugging Face at 4.5 MB/s. FineWeb-Edu parquet holds about 2.97 GB per billion of our tokens. SCALED: about
**10.5 minutes of box time per extra billion tokens**, set by the download. The training GPUs idle during it unless
it runs during the bench.

**Repetition rule.** Muennighoff et al. (NeurIPS 2023, abstract read 2026-10-03): *"with constrained data for a
fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to
having unique data."* MEASURED (elsewhere). So below, unique tokens are set to the run's tokens divided by 4, and
never fewer than `train_big`'s 3.018B.

### 5.2 The trade, side by side

The same money cases for every size. `lo` speeds; `hi` is about 2 times more.

```
   24 h on 4x H100 SXM, $168, lo speed. One character = 1B tokens, to scale.
   |  = 1x compute-optimal (20 N_c)    :  = 3x (60 N_c)    ●  = tokens the box trains

   d 768    |   :                                      ▶ 87.5B  40x over    small, many tokens
   d 1024    |     :                                   ▶ 65.7B  22x over
   d 1536      |        :                            ● 43.1B  9.3x over
   d 2048       |            :            ● 31.5B  4.9x over   ◀ the biggest nice model
   d 3072           |         ●           : 19.9B  1.9x
   d 4096               ●|                             : 14.0B  0.9x         big, nothing spare
          +---------+---------+---------+---------+-----
          0B        10B       20B       30B       40B
          ├─┤ train_big 3.0B
          ├────────┤ sample/10BT 9.4B, and 4 epochs of it reach 37.6B
```

**(a) a smaller model on more tokens, (b) a larger model on fewer tokens, (c) only part of the set.** Every row is
(c) against the full corpora; the columns show how much.

24 hours on 4x H100 SXM ($168), `lo`:

| model | tokens | epochs of train_big | of sample/10BT | of FineWeb-Edu | of ClimbMix | of Smol-Data | 20 x N_c | verdict | tok per param (total / non-emb) | prep for 4-epoch cap |
|---|---|---|---|---|---|---|---|---|---|---|
| d 768 h4 65k | 87.5B | 29.0 | 9.3 epochs | 7.2% | 26% | 93% | 40x | over | 283 / 416 | 198 min, $23 |
| d 1024 h4 65k | 65.7B | 21.8 | 7.0 epochs | 5.4% | 20% | 70% | 22x | over | 158 / 232 | 141 min, $16 |
| d 1536 h4 65k | 43.1B | 14.3 | 4.6 epochs | 3.5% | 13% | 46% | 9.3x | over | 68 / 99 | 82 min, $10 |
| **d 2048 h4 65k** | 31.5B | 10.5 | 3.4 epochs | 2.6% | 9.4% | 34% | 4.9x | over | 37 / 53 | 51 min, $6 |
| d 3072 h4 65k | 19.9B | 6.6 | 2.1 epochs | 1.6% | 5.9% | 21% | 1.9x | over | 15 / 21 | 20 min, $2 |
| d 4096 h4 65k | 14.0B | 4.7 | 1.5 epochs | 1.2% | 4.2% | 15% | 0.9x | optimal | 8 / 11 | 5 min, $1 |

48 hours on 1x H100 SXM ($94), `lo`:

| model | tokens | epochs of train_big | of sample/10BT | 20 x N_c | verdict | prep |
|---|---|---|---|---|---|---|
| d 768 h4 65k | 54.1B | 17.9 | 5.8 epochs | 25x | over | 110 min, $4 |
| d 1024 h4 65k | 40.7B | 13.5 | 4.3 epochs | 14x | over | 75 min, $3 |
| d 1536 h4 65k | 26.7B | 8.9 | 2.8 epochs | 5.7x | over | 38 min, $1 |
| **d 2048 h4 65k** | 19.5B | 6.5 | 2.1 epochs | **3.0x** | over | 20 min, $1 |
| d 3072 h4 65k | 12.2B | 4.0 | 1.3 epochs | 1.2x | optimal | none |
| d 4096 h4 65k | 8.6B | 2.8 | 0.9 epochs | 0.6x | under | none |

48 hours on 2x H100 SXM ($207, one machine on the verified board): 33.7B at d 2048 (5.2x), 21.2B at d 3072
(2.0x), 15.0B at d 4096 (1.0x). SCALED from the same formula.

### 5.3 Which side the published evidence favours

- **For a model judged at inference, the small-model evidence favours over-training.** SmolLM2-135M trained on 2T
  tokens, about 14,800 per parameter, on purpose: inference cost dominates. MiniCPM measured 192 tokens per
  non-embedding parameter as its average optimum, ten times Chinchilla's 20. SmolLM2 and codelion found one good
  data mix, single stage, works best at small sizes. MEASURED (elsewhere), all from BASEDATA §2 and §3.
- **That pushes toward side (a): a model somewhat smaller than the widest that fits, trained 3x or more past
  compute-optimal.** Side (b), the widest model at 1x (d 4096), spends the whole budget to reach the cheapest point
  of a curve we have never measured for this family, and is the slowest to run.
- **Against going small:** d 768 at 87.5B is 29 epochs of `train_big` and 9.3 epochs of `sample/10BT`, well past
  the 4-epoch rule. It needs about 22B unique tokens and more than 3 hours of download.
- **"The biggest nice model we can do" is d 2048, 4 hops, 65,536 rows, at 19.4B tokens (3x compute-optimal).**
  It is 6.4 epochs of `train_big` or 2.1 of `sample/10BT`, so it fits the 4-epoch rule with `sample/10BT`, after
  about 51 minutes of download. It is the largest row that reaches 3x on the cheapest single box in 48 h, and it
  leaves half of the 4x H100's day spare. Its MiniCPM-style count (192 x non-embedding, value tables excluded) is
  10.9B, which 19.4B also clears.
- **The stretch is d 3072 at 2x** (21.2B on 2x H100 in 48 h, or about 20B on 4x H100 in 24 h). It costs the same
  money but has no spare, and no width above 1024 has been measured for this family.

## 6. What must be measured first, and its price

Every speed above the Spark is an ESTIMATE. The DDP trainer's own bench mode runs on random tokens, so it needs no
data shards and takes minutes. On each box, with the repo's `experiments/track4/sdmllm/` copied over:

```bash
# set up (about 5 minutes). Image by compute capability, from the board's compute_cap and cuda_max_good:
#   sm_120 (RTX 5090, RTX PRO 6000):  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime  (cu128 is required;
#                                     MEASURED ours on driver 570.181)
#   sm_100 (B200):                    a cu128 or newer image (torch 2.7 or later)
#   sm_103 (B300):                    CUDA 12.9 or newer (UNVERIFIED); a cu129 or cu130 image
#   sm_90  (H100, H200):              cu124 to cu128; machine 59522 reports cuda_max_good 12.4, so it needs a
#                                     cu124 image or a different host
apt-get install -y gcc g++        # torch.compile needs a compiler (MEASURED ours, 2026-10-01)
cd sdmllm && python3 track4_sdmonly_models.py --selftest && python3 track4_sdmonly_train_ddp.py --selftest

# one GPU, the shapes that matter (about 2 minutes each, compile included)
for D in 768 1536 2048 3072; do
  python3 track4_sdmonly_train_ddp.py --world 1 --bench 300 --arm sdmonly --d $D --hops 4 --n-sub 256 \
    --B 32 --loss chunked --compile
done
python3 track4_sdmonly_train_ddp.py --world 1 --bench 300 --arm sdmonly --d 2048 --hops 4 --n-sub 256 --B 64 \
  --loss chunked --compile
python3 track4_sdmonly_train_ddp.py --world 1 --bench 300 --arm sdmonly --d 2048 --hops 4 --n-sub 512 --B 32 \
  --loss chunked --compile
# all GPUs of a multi-GPU box, dense and row-only table sync
for S in dense rows; do
  python3 track4_sdmonly_train_ddp.py --bench 300 --arm sdmonly --d 2048 --hops 4 --n-sub 256 --B 32 \
    --loss chunked --compile --table-sync $S
done
nvidia-smi topo -m; nvidia-smi --query-gpu=driver_version --format=csv,noheader   # the driver stamp
```

| card | box | about 30 min | why |
|---|---|---|---|
| H100 SXM | 4x, machine 59522 | **$3.50** | the proposed BASE box: per-GPU rate and real DP efficiency |
| H100 SXM | 1x | $1.00 | the cheapest-token box, no all-reduce |
| RTX PRO 6000 S | 1x | $0.60 | settles 252 against 480 TFLOP/s (FP32 accumulate rate) |
| B200 | 1x | $3.90 | how far the biggest card falls below its peak on this model |
| RTX 5090 | 4x | $0.90 | the control box; ties the formula to the measured 152k |
| A100 SXM4 | 1x | $0.30 | BASEDATA's flagged value candidate |
| **total** | | **about $10** | |

Add a short width ladder on the 4x H100 before the long run: d 768, 1536 and 2048 at 300M tokens each on
`train_big` (WSD), 10 to 20 minutes each, about **$5**. It answers whether width still pays above 1024 for the
SDM-only shape before $100 is spent on d 2048. **Bench plus ladder: about $15.**

## 7. Recommendation

**A PROPOSAL for the navigator to rule on.** Every speed is an ESTIMATE until section 6 runs; prices are the
2026-10-03T10:39Z board.

```
   24 h, the proposal                          48 h, the alternatives
   BOX A  4x H100 SXM  machine 59522           1x H100 SXM, $1.97/h
          $6.99/h · NVLink · rel 0.999           d 2048 h4 65k at 19.5B (3.0x) in 48 h, $94
          d 2048 h4 65k · 19.4B tokens (3x)      no spare hours at lo speed
          + 3 cooldown forks, 4.3B tokens      2x H100 SXM, one machine, $4.30/h
          then the browser BASE d 768 · 6.5B     d 3072 h4 65k at 21.2B (2.0x), $207
          wall 11 to 21 h · $75 to $146          leaves $34 of reserve after box B and bench
   BOX B  4x RTX 5090 · $1.85/h
          four control and seed arms
          about 14 h · $26 to $44
   bench + width ladder ............ $15
   ─────────────────────────────────────
   total $116 to $205, capped at $227 · reserve at least $73 of $300
```

- **Box A, the BASE.** 4x H100 SXM, machine 59522 (Washington, 128 cores, 504 GB RAM, rel 0.999, NVLink field 478),
  $6.99/h all-in with 200 GB. Fallbacks: machine 20897 ($8.57) and 68467 ($13.77). One DDP run, WSD, global batch
  4 x 64 x 256 = 65,536 tokens with `--lr-scale sqrt` (the D^0.383 rule from BASEDATA §4 suggests about 31,000 at
  this token count, so the bench should compare B 32 against B 64 per GPU).
- **The model.** d 2048, 4 hops, 65,536 rows a hop, k 32: 858.6M parameters (264.8M token table, 536.9M value
  tables, 56.9M rest), 647.0M forward FLOPs per token, N_c 323M. Hops and rows follow wave 1's verdict (P6, P7);
  8 hops costs about 13% more time, 262,144 rows about 30%.
- **The tokens.** 19.4B (3 x 20 x N_c) of FineWeb-Edu `sample/10BT`, about 9.4B unique, 2.1 epochs. Prep about
  51 minutes on the box, during the bench. Cooldown forks at 2B, 6.5B and 13B (20% of each, 4.3B tokens in all)
  give the token-scaling curve the plan wants. Then the browser BASE (d 768, 4 hops, 65,536 rows, shared store if wave 1 allows) at 6.5B, 1 to 2 hours.
- **Box B, the controls.** 4x RTX 5090 (32 machines qualify, from $1.85/h), one arm per GPU, each compared with a
  cooldown fork of the BASE at the same tokens: d 2048 dense control at 2B (about 12 h), d 2048 seed 1 at 2B, d 768
  dense control at 6.5B (about 14 h), d 768 seed 1 at 6.5B. A 2B fork of d 2048 is 0.3x compute-optimal, so that
  comparison sits early on the curve; that is the price of fitting the controls in budget.
- **Expected:** wall 11 to 21 hours for box A (prep 1 h, BASE 7 to 15 h, forks 2 to 3 h, browser BASE 1 to 2 h),
  14 hours for box B; total $116 to $205, with a hard cap of $227 if both boxes run the full 24 hours; at least
  $73 of the $300 left. SCALED from ESTIMATE speeds.
- **The value alternative for box A:** 8x A100 SXM4 (machine 151759, $3.65/h) trains more tokens per dollar on
  paper (section 4.2), but its NVLink field reads 0 and its rate rests on an A100 MFU nobody has measured for us.
  Choose it only if the bench shows NVLink in `nvidia-smi topo -m` and at least 0.7 data-parallel efficiency.

**The two biggest risks:**
1. **The speeds are estimates and could be half of `lo`.** At half, d 2048 at 19.4B takes about 30 hours on box A.
   The fallback is fixed in advance: start the final decay when the 22nd hour arrives (the WSD run allows it),
   which lands about 14B tokens, 2.2x compute-optimal, at the same $168; drop the forks and move the browser BASE to
   box B. The bench removes this risk for $10.
2. **Width above d 1024 has never been measured for this model family.** VC (d 1024) beat VB (d 768) by 0.021 bpb
   on the older shape at 2.5B tokens (MEASURED, ours); nothing wider exists. If d 2048 does not beat d 1536 at
   matched tokens in the width ladder, the right BASE is d 1536 at 13.9B (3x), which is cheaper and smaller in the
   browser.

## 8. What could not be verified

- No SDM-only speed off the Spark exists. Every rented-card rate is an ESTIMATE from the formula in 4.1.
- RTX 5090 and RTX PRO 6000 dense BF16 peaks with FP32 accumulate: second-hand pages disagree by 2x for the PRO 6000.
- Which board offers really have NVLink, where the field reads 0.
- The B300's CUDA requirement (sm_103) for a PyTorch image.
- The browser thresholds (download size and heap limits) for our engine.
- That one host can sustain Hugging Face downloads faster than the 4.5 MB/s measured on one box.
- Whether `--table-sync rows` and the row optimiser (lane SPARSEROWS, refused by the DDP trainer for now) change the
  data-parallel efficiency; neither has been timed on a multi-GPU box.

## Sources

Read 2026-10-03 UTC.
- `vastai search offers` (read-only, two queries above), 2026-10-03T10:39Z.
- B300 and B200 BF16 and memory figures: [SemiAnalysis InferenceX, B300](https://inferencex.semianalysis.com/chips/b300) ·
  [flopper.io, B300 SXM](https://flopper.io/gpu/nvidia-b300-sxm-288gb) (via search).
- RTX PRO 6000 Blackwell: [NVIDIA, Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) ·
  [flopper.io, Workstation Edition](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition) ·
  [NVIDIA forum on FP32-accumulate rate](https://forums.developer.nvidia.com/t/rtx-pro-4000-blackwell-fp16-fp8-fp6-fp4-throughput-with-fp32-accumulation/382903) (via search; UNVERIFIED).
- Muennighoff et al., Scaling Data-Constrained Language Models: [arXiv 2305.16264](https://arxiv.org/abs/2305.16264) (abstract, via search).
- Berges et al., Memory Layers at Scale, arXiv 2412.09764 (cited from `TRAINING_SDMONLY_HOW_WE_TRAIN.md`).
- Local: `track4_sdmonly_models.py` (counts and FLOPs), `track4_sdmonly_train_ddp.py` (bench mode, table sync),
  `RESEARCH_BASEDATA_2026-10-04.md` (rules, 5090 measurement, datasets),
  `../../task-runner/JIMOTHY_VAST_SETTLE_2026-10-01.md` (box prep time, links, image, time-zone lesson),
  `../../../SETTLE/SETTLE_CAMPAIGN_2026-09-30.md` (165.9 MB export, browser speed),
  `track4_sdmllm24_train_big_provenance.json` (3,018,298,789 tokens, 1,937 s), `SDMONLY_CHANNEL.md` (Spark speeds).
