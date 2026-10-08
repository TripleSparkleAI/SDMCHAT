# MIXERFAST: the SDM mixer from sorted index lists (2026-10-03)

The SDM mixer (`SdmMix` in `track4_sdmonly_models.py`) is the all-SDM answer to attention. Each position writes a
value into the 2k hard-locations its address picks, and reads the 2k locations its own address picks, holding only
what earlier positions wrote, faded per head. `SdmMixFast` computes the same layer with the same parameters, at a
per-token cost that does not depend on n_sub^2 and stays flat from a 256-token window to a 65,536-token window.

- Code: `track4_sdmonly_mixfast.py` (the scan, the Triton kernels, the autograd functions),
  `SdmMixFast` in `track4_sdmonly_models.py`, `--mix-impl fast` in `track4_sdmonly_train.py`,
  `track4_sdmonly_mixfast_bench.py` (the speed table).
- The default stays `mix_impl="dense"`, so no past run and no past number changes.
- The mixer is still a Kanerva write and read. Nothing here is a transformer part.

## 1. The method

### The maths both versions compute

For one head of one batch row, with 2k picked locations per position (index `si[t, c]`, weight `w[t, c]`):

```
A[t, m]  = sum_c w[t, c] [si[t, c] = m]                      (2k non-zero entries in a row of n_sub^2)
num[t]   = sum_{s<t} decay^(t-s) (sum_m A[t, m] A[s, m]) V[s]
den[t]   = sum_{s<t} decay^(t-s)  sum_m A[t, m] A[s, m]
read[t]  = num[t] / (den[t] + 1e-6)
```

The dense version builds `A` for one chunk of L = 256 positions as a (L, n_sub^2) array and multiplies. That costs
about L x n_sub^2 per position and head (256 x 16,384 at n_sub 128).

### The fast version

Two positions meet only through a location that both picked. So the fast version never builds `A`.

```
  every position t lists its 2k triples (location m, position t, weight w)

  (b,h,m)  t:  3     17     40     41     90
            ───●─────●──────●──────●──────●──▶ time      one SEGMENT = the history of one location
               write write  write  write  write
                     read   read   read   read           each triple first READS what the segment holds
                     ▲ holds w3 V3, faded by decay^(17-3)   then WRITES w V of its own position
```

1. Sort all B x heads x T x 2k triples by (batch row, head, location), with a STABLE sort, so positions stay in
   time order inside each location. A run of equal keys is a segment: the history of one location.
2. Walk each segment in time. The state of the segment before triple i is exactly what location m holds when
   position p_i reads it: `R_i = sum over earlier j of decay^(p_i - p_j) w_j [V[p_j], 1]`. A position picks a
   location at most once, so "earlier in the segment" means "earlier in time".
3. Add `w_i R_i` into the reading position's row. The 2k triples of one position meet there. The extra lane of
   `[V, 1]` gives the denominator.

The cost per position and head is about 2k x (dh + 1) multiply-adds for the read and the same for the write, plus
a sort of 2k keys. Nothing scales with n_sub^2. Nothing scales with the window except the sort's log factor.

### Two implementations of the walk

- **Torch (CPU, float64, and the reference):** `seg_scan_excl` works in tiles of 32 consecutive triples. Inside a
  tile, `P[i, j] = [same segment] [j < i] exp(lg |p_i - p_j|)` and `E = P @ x` is one batched matmul. Across tiles,
  a segment that continues carries its state through `c_k = A_k c_(k-1) + b_k`, solved by a Hillis-Steele scan
  that stops when every chain reaches its start. Every factor is `exp(lg x |gap|) <= 1`, so nothing overflows,
  for never-fading heads (lg 0) and strongly fading ones alike.
- **Triton (CUDA float32, used in training):** one program per segment walks it in time (`_seg_fwd_kernel`) and
  adds `w R` into the output row with an atomic add. The backward kernel (`_seg_bwd_kernel`) walks forward for the
  reader part of dw and backward for Q, what later readers will pull from this location:
  `dL/dw[t, c] = R . g[t] + Q . [V, 1][t]` and `dL/dV[s] = sum_c w[s, c] Q_(s, c)`.
  The sort from the forward pass is kept for the backward pass (about 28 bytes a triple).

### Other parts of SdmMixFast

- `locate` gives the same exact top-2k product-key search as `ProductKeyStore.locate`, with less work. The two
  half-score lists are sorted, so a rank pair (a, b) can be in the top c only if (a + 1)(b + 1) <= c. That
  staircase is 303 pairs for c = 64, against 4,096 in the full grid. This change cut the address step from 52 ms
  to 13 ms per layer at 16k tokens.
- Rows (batch row x head) never share a segment. So forward and backward run over slices of at most 2^23 triples,
  and the product-key search runs over slices of 16,384 positions. A 262,144-token window then fits in memory.
- The layer runs eagerly (a sort and a data-dependent loop) inside a compiled model, and outside autocast
  (float32). Under training autocast, the dense layer ran its products in bf16.

## 2. Equivalence evidence

All checks compare the fast layer with the dense layer after copying the dense layer's state dict into it.

| check | where | result |
|---|---|---|
| torch scan = brute-force double loop, tiles 1, 4, 7, 32, 512, reversed scan | M5 CPU, float64 | equal to 1e-9 |
| `torch.autograd.gradcheck` of the scan function, one and several row slices | M5 CPU, float64 | pass |
| Triton kernels = torch scan, output and dw, dV, 4 decays, row slices, segments ~120 long | Spark GB10, float32 | pass |
| fast = dense output, decays (1, 0.99, 0.9, 0.5) and (1, 1, 1, 1), dense chunks 5, 16, 37, B 3, T 37, scan tile 3, several row and search slices | M5 CPU | pass (6 of 6) |
| fast = dense gradients of x and of every parameter, the same 6 cases | M5 CPU | pass (6 of 6) |
| the staircase search = the full product-key search (scores and locations) | M5 CPU | pass |
| the whole model with fast mixers = the dense-mixer model; causal; a change at position 2 reaches position 19 | M5 CPU | pass |
| realistic size (d 256, 4 heads, n_sub 128, k 32): B 4 x T 1,024 and B 1 x T 4,096, both decay sets | Spark GB10, float32, TF32 off | output max diff 4.8e-7 (scale 1.57); grad x max diff 2.2e-6 (scale 3.6); worst parameter-gradient relative diff 6.4e-7 |

Self-tests: `python3 track4_sdmonly_models.py --selftest` gives 50 of 50 (it was 33; every earlier check still
passes). `python3 track4_sdmonly_mixfast.py --selftest` gives 8 of 8 on the M5 and 10 of 10 on the Spark, where
the two Triton checks run.

## 3. Speed

A training step (forward, backward, AdamW) of the wave-11 cnmix body: `sdmonly_none`, d 256, hop MLPs 192, untie,
conj 2, 4 mixer layers, n_sub 128 (16,384 locations a head), k 32, 4 never-fading heads, d_a 64. Vocabulary
129,280, random tokens, the trainer's loss path (compiled chunked tied-head cross-entropy), bf16 autocast.
B = max(1, 16,384 / T), so from T 16,384 up a step holds T tokens. The "none" column is the same body with no
mixer.

**Stamps for every row:** NVIDIA GB10 (the Spark), driver 580.159.03, CUDA 13.0, torch 2.14.1+cu130, Triton 3.8.0.
Spark queue empty and GPU at 0 to 19% busy before each point (only an idle 548 MiB process of another project on
the GPU). Load average 0.0 to 2.6 on 20 cores. Each point runs in a fresh process: 2 warmup steps, then 6 timed
steps (dense at T 262,144: 1 and 2). Times are 2026-10-03, 16:46Z to 17:45Z.

### Without the moving-average features (`--no-ema`)

The five causal moving averages are built in `track4_sdmllm_models.causal_ema` as a T x T matrix per decay. At
T 65,536 that is 17 GB per decay, so the long-window rows drop them.

| T | B | none tok/s | dense tok/s | fast tok/s | fast / dense | mixer us/token, dense | mixer us/token, fast | peak GiB dense | peak GiB fast | peak GiB none |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 66,650 | 10,788 | 34,054 | 3.2x | 77.7 | 14.4 | 23.0 | 5.6 | 3.5 |
| 1,024 | 16 | 66,597 | 6,921 | 35,696 | 5.2x | 129.5 | 13.0 | 14.6 | 5.5 | 3.5 |
| 4,096 | 4 | 69,256 | 2,328 | 37,185 | 16.0x | 415.1 | 12.5 | 9.0 | 5.5 | 3.5 |
| 16,384 | 1 | 69,310 | 2,325 | 37,603 | 16.2x | 415.8 | 12.2 | 9.0 | 5.5 | 3.5 |
| 65,536 | 1 | 75,744 | 2,307 | 36,880 | 16.0x | 420.3 | 13.9 | 38.5 | 24.3 | 16.0 |
| 262,144 | 1 | 75,846 | not measured | 29,929 | - | - | 20.2 | - | 99.6 | 70.1 |

"Mixer us/token" is 1e6 / (tok/s with the mixer) - 1e6 / (tok/s of none): the time the four mixer layers add per
token. Fast stays at 12 to 14 us from T 256 to 65,536. At 262,144 it rises to 20 us. Dense rises from 78 to 420 us,
because it compiles only up to 2,048 and runs eagerly with a recompute in backward beyond.

Dense at T 262,144 is NOT MEASURED. Its worker process died twice during the first step (17:01Z and 17:32Z) and
wrote no error line. A likely cause is memory: the dense layer runs the product-key search over the whole window
at once, and that builds a (T, heads, 4,096) candidate grid per layer, about 17 GB in float32 at 262,144 tokens.
That cause was not confirmed. At its measured 2,307 tok/s a 262,144-token step would take about 114 s, so the
missing row would not change the comparison.

### With the moving averages (the cnmix model as trained in wave 11)

| T | B | none tok/s | dense tok/s | fast tok/s | fast / dense | peak GiB dense / fast / none |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 64 | 71,705 | 11,795 | 37,273 | 3.2x | 23.1 / 5.7 / 3.6 |
| 1,024 | 16 | 71,541 | 7,671 | 36,950 | 4.8x | 14.7 / 5.6 / 3.6 |
| 4,096 | 4 | 70,795 | 2,615 | 38,736 | 14.8x | 9.3 / 5.8 / 3.7 |
| 16,384 | 1 | 60,938 | 2,575 | 35,781 | 13.9x | 11.6 / 8.1 / 6.1 |

These GB10 numbers are about 6x below the RTX 5090 numbers in the plan's log (65k, 45k and 12.6k tok/s for dense).
Compare ratios across the two GPUs, never absolute tok/s.

## 4. Limits that remain

- **The moving averages are the next wall, outside the mixer.** `causal_ema` is O(T^2) in memory and time. A 256k
  window needs it replaced by a recurrent form (one decayed sum per decay, which is exact). That is in
  `track4_sdmllm_models.py` and was not in this lane's scope.
- **Long segments.** At T 262,144, each location's history is about 1,024 triples long. Each Triton program walks
  its segment one triple at a time, so the walk becomes latency-bound with fewer, longer programs. That is why the
  mixer cost goes from about 13 to 20 us/token. Splitting long segments into blocks with a carried state (the tile
  scheme of the torch path) would remove this.
- **Memory at 262k.** The kept sort costs about 28 bytes a triple: 1.9 GB a layer at 262,144 tokens a step.
  Peak memory with fast mixers was 99.6 GiB at B 1 x T 262,144, against 70.1 GiB without mixers. That fits the
  Spark's 121 GB. It does not fit a 32 GB RTX 5090 at that step size, so a 5090 would need a smaller T or
  recomputing the sort in backward (one extra sort per layer, about 5 ms at 16k tokens).
- **Precision differs from training's dense path.** Fast runs in float32 (the Triton walk has no matmul). The dense
  layer ran its products in bf16 under autocast. Loss curves of a fast run and a dense run should agree closely,
  but not bit for bit. No training run has used `--mix-impl fast` yet.
- **Determinism.** The atomic adds make the last bits of the output depend on scheduling. All equivalence checks
  pass at the tolerances above.
- **The rest of the step.** At T 16,384 the mixers take about 45% of the step (12.2 of 26.6 us per token). The
  next largest costs per mixer layer are the backward walk (11 ms at 16k tokens), the sort (5 ms) and the top-k of
  the address step (3.5 ms).

## 5. How to use it

```bash
python3 track4_sdmonly_train.py --arm sdmonly_none ... --mix-layers 4 --mix-n-sub 128 --mix-impl fast
python3 track4_sdmonly_models.py --selftest          # 50 checks, fast = dense included
python3 track4_sdmonly_mixfast.py --selftest         # scan, gradcheck, and on a CUDA box the Triton kernels
python3 track4_sdmonly_mixfast_bench.py --Ts 1024,16384 --impls none,dense,fast --no-ema
```

Raw records (one JSON line per point, with start and end stamps): `mixerfast_bench_2026-10-03_noema.jsonl`,
`mixerfast_bench_2026-10-03_ema.jsonl`.
