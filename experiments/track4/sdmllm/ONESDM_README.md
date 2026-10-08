# ONESDM: the pure run-time SDM model and the memory puzzles

This folder holds a language model in which every step that mixes positions is a Kanerva memory written and
read at run time. There is no attention and no transformer block. The files are:

- `track4_onesdm_models.py`: the model (`OneSdmLM`), its run-time memory (`RunTimeSdm`), the trained table that
  replaces the MLP in the all-SDM arms (`TrainedSdm`), the optional archive (`Archive`), and the baselines.
- `track4_onesdm_puzzles.py`: three synthetic memory puzzles, a cost bench, a trainer and an evaluator.

The design follows "DECIDED: the middle path" in `IDEAS_SDMONLY_NEXT_2026-10-05.md`.

## What the model is

```
   token ── ENFOLD (token table) ──▶ x            THE COMPLICATE = the residual stream x
                                     │
   every layer:                      ├─▶ x + Mem(x)        the run-time Kanerva memory
                                     ├─▶ x + Archive(x)    only when the archive is on
                                     └─▶ x + MLP(norm(x))
                                     │
   next-token scores ◀── UNFOLD (tied table) ── norm(x)
```

### The run-time memory, per layer and per head

- **Fixed size.** Each head holds `M = n_sub^2` hard locations. Each location keeps a value sum (`dh` floats) and
  a written weight. The default is `n_sub 32`, so M is 1,024 locations per head.
- **Addresses.** The model projects the normalised residual into a write key, a read query and a value. The key
  and the query each pick the best `k` locations (default 16) by product keys. Each address is split into two
  halves, each half is L2-normalised and scored by cosine against `n_sub` sub-keys, and a location scores the
  mean of its two half scores. The weights over the `k` locations are `softmax(scale * score)`, with `scale`
  learned per head.
- **Why product keys and not sign-of-projection addressing:**
  - The top-k search is exact and costs `2 n_sub` scores per token, not `n_sub^2`.
  - The soft weights over the `k` locations give the address a gradient. A sign hash gives none.
  - Every earlier SDMONLY store and mixer used product keys, so results stay comparable.
- **Write gate.** `g = sigmoid(W_g h + b)` lies in [0, 1] per token and per head. It scales every weight of the
  write, so filler can learn to write weakly.
- **Write and read.** At token t the model first reads, then writes. The read sees only what earlier tokens
  wrote. A write made at s reaches a read at t faded by `lam^(t-1-s)`:
  - `mem <- lam * mem + g * a_t v_t^T`
  - `cnt <- lam * cnt + g * a_t`
- **Fades.** Each head has its own fade `lam`. The default is `0.0, 0.8, 0.99, 1.0`. A fade of 0 makes a
  previous-token head (the self-test checks this exactly). A fade of 1 never forgets.
- **Read.** In `mean` mode the read is `num / (den + 1e-3)`. That is a weighted mean of the values written where
  the read looks, and it is zero where nothing was written. In `sum` mode the read is `num`.
- **Output.** An optional per-head RMS norm, with an RMS floor of 0.1, then an output map that starts at zero.
  The read is added to the residual. It never overwrites it.
- **Causality and chunks.** Training runs in chunks of `chunk` tokens (default 128). Inside a chunk the
  write-then-read is one masked product. Only the memory state crosses chunk boundaries. Each token therefore
  costs about `chunk * M` per head, whatever the length of the text. `forward_sequential` runs the same model
  token by token with sparse gathers and scatters. It is the reference, and it is also the decode step.

### Stability measures

The earlier mixer led early, then fell behind as its gradients climbed. These measures address that, and each
one is a flag:

| measure | flag | default |
|---|---|---|
| output map starts at zero, so the model starts as a plain residual MLP stack (checked exactly) | always on | on |
| L2-normalised queries and keys; no batch norm, because batch statistics mix future positions into the past | always on | on |
| per-head RMS norm on the read, RMS floor 0.1 | `--no-out-norm` | on |
| separate learning-rate multiplier for the memory and archive parameters | `--mem-lr-mult` | 1.0 |
| a gradient clip on the memory parameters alone, applied before the global clip | `--mem-clip` | off |
| gradient scale from the memory back into the residual stream | `--mem-in-grad` | 1.0 |
| learned fades | `--learn-decay` | fixed |
| write gate | `--no-gate` turns it off | on |
| read mode | `--read-mode mean or sum` | mean |
| shared address bias per head, added to both key and query, initialised N(0, S^2); a large S starts every head as a fading average (a fade-0 head as an exact previous-token head) | `--addr-bias S` | 0 (the bias is still learned) |

Every log line records the gradient norm of the memory parameters and of the rest of the model, both measured
before clipping.

### The archive (off by default)

- Each token keeps a compressed key and a compressed exact vector, `d_c` floats each (default 64). This buffer
  grows with the text.
- The token's position is written as a pointer into a pointer SDM. For each of the `k` slots its key wakes, the
  SDM keeps a ring of the last `P` positions written there.
- A read wakes `k` slots by its query and takes their `k * P` pointers. It adds the last `W` positions, which is
  the exact recent window. It drops duplicates, scores each candidate as `q . k_s / sqrt(d_c)`, and keeps the top
  `n_fetch`. It returns their exact vectors, soft-weighted (`soft`) or hard with a soft gradient (`st`), through
  an output map that starts at zero.
- The archive is exactly causal, and the parallel form equals the ring-buffer recurrence (both are self-tested).
- **Not trained well yet.** The choice of pointers is discrete, so the slot choice gets no gradient. The pointer
  addresses use the same compressed key and query as the scores, so they move only because the scoring trains
  those maps.
- Scoring a small fixed set of `k * P + W` candidates with a softmax is a soft selection over a fixed-size set.
  The brief allows it. Its cost per token is fixed.

### The arms

| arm | what it is |
|---|---|
| `onesdm` | the pure run-time SDM model |
| `onesdm_archive` | the same model with the archive on (every layer but the first, unless `--arc-layers`) |
| `plain` | the same stack with the memory off: the no-memory floor; nothing mixes positions |
| `sdmonly` | the current base model (8 back tokens, 5 fading averages, product-key store, no run-time memory) at about matched size |
| `yardstick_transformer` | the 4-layer QwenLM from `track4_sdmllm_models.py`; a YARDSTICK only, never part of the model |
| `onesdm_allsdm` | the fully SDM model: the run-time memory exactly as `onesdm`, and a trained SDM (`TrainedSdm`) in place of every MLP, matched in weights |
| `onesdm_allsdm_big` | the same with a table 16 times larger (n_sub x 4) and the same k: the same values touched per token, about 10 times the weights |

The two all-SDM arms run only when named in `--arms`. The default campaign is still the first five.

Non-embedding parameters at the defaults (width 256, 4 layers, MLP 688), measured by the self-test on 2026-10-07:

| arm | non-embedding parameters |
|---|---|
| onesdm | 2,924,320 |
| onesdm_archive | 3,121,696 |
| plain | 2,924,320 |
| sdmonly | 3,552,768 |
| yardstick_transformer | 2,902,784 |
| onesdm_allsdm | 2,946,352 (onesdm + 0.75%) |
| onesdm_allsdm_big | 28,829,488 |

(This table read 2,923,808 for onesdm and plain and 3,121,184 for onesdm_archive. The difference is 512, the learned
address bias `addr_b`, which was added after the table was written.)

### The all-SDM arms: a trained SDM in place of the MLP

```
   every layer of onesdm_allsdm:
     x ── x + Mem(x)            the run-time Kanerva memory (written and read while reading), as in onesdm
       └─ x + TrainedSdm(x)     a trained table read once per token, in place of the SwiGLU MLP

   TrainedSdm, per token and per head:
     h = RMSNorm(x)
     q = W_q h                         heads x d_a wide, never wider than d
     pick the top k of n_sub^2 slots   exact product-key search (the same addressing as the memory)
     w = softmax(scale * score)        over the k picked slots; scale learned per head
     r = sum of w_i * value_i          each value row is d / heads wide
     out = W_o (heads concatenated)    back to width d
```

- **No attention, no MLP, no dense expansion.** The self-test walks every module of both arms: no SwiGLU, no attention
  module, and no Linear in the layer body wider than d. The same check fails on `onesdm`.
- **Zero start.** Every value starts at zero, so at init each TrainedSdm adds exactly nothing. The self-test checks
  the output against the same model with every TrainedSdm removed, exactly.
- **Gradient.** The top-k pick is hard. Gradient reaches the picked value rows, and the sub-keys and the query map
  through the scores of the picked slots. The self-test checks all three receive gradient once the values are nonzero.
- **Sizes at the defaults.** `onesdm_allsdm`: n_sub 41, so 1,681 slots per head, 4 heads, rows 64 wide (the same
  weights as 1,681 slots 256 wide), k 32 per head, d_a 32. `onesdm_allsdm_big`: n_sub 164, 26,896 slots per head,
  the same k. Flags: `--sdm-n-sub` (0 = matched), `--sdm-heads`, `--sdm-d-a`, `--sdm-k`, `--sdm-scale`,
  `--sdm-big-mult`.
- **What they test.** `onesdm` against `onesdm_allsdm` asks whether a trained SDM can do the MLP's job at the same
  weights. `onesdm_allsdm` against `onesdm_allsdm_big` asks whether more slots help when the compute per token stays
  the same, which is the reason to use a sparse table at all.

## The puzzles

All three puzzles use a puzzle vocabulary, not the LM tokenizer: PAD, SEP, QUERY and BOS, then 128 key symbols,
128 value symbols and 128 noise symbols (V 388). The loss counts only the scored positions. **Higher accuracy is
better.** Chance is about 0.008.

| puzzle | sequence | scored | trained at | tested at |
|---|---|---|---|---|
| A recall at distance (`recall`) | BOS, 8 (key value) pairs, `gap` noise tokens, 4 (QUERY key value) triples | the value after each queried key | gap uniform in [0, 256] | gaps 100, 1,000, 4,000, 16,000 (the long gaps are never seen in training) |
| B copy the repeat (`copy`) | BOS, L random symbols, SEP, the same L symbols | every token of the second copy | L uniform in [8, 128] | L 128, 512, 2,048 |
| C blurry cue (`blurry`) | as A, keys are 4 symbols; at query time a fraction of them are replaced by other key symbols | the value after each queried key | corruption 0, gap uniform in [0, 256] | corruption 0, 0.25, 0.5, 0.75 at gap 100 |

The **cost bench** runs a forward pass at T 1,024, 4,096 and 16,384 with batch size 1. It reports seconds per
token (best of the repeats, after one warm run), CUDA peak memory on a GPU, and the bytes a model must keep to
continue decoding. **Flat is better.**

Each run writes one JSON to `runs_onesdm/` (or `$ONESDM_RUNS`). The JSON holds the full model and training
config, parameter counts, the training curve with memory and body gradient norms, every evaluated setting, and
stamps: host, device, torch, CUDA and driver, load average and git sha.

## How to run

### Self-tests (CPU, about 10 seconds together)

```bash
cd experiments/track4/sdmllm
python3 track4_onesdm_models.py --selftest     # 42 checks
python3 track4_onesdm_puzzles.py --selftest    # 15 checks, including the positive controls
```

### The full campaign on one GPU

Run one command per puzzle, so a failure costs one puzzle and not the whole set.

```bash
cd experiments/track4/sdmllm
export ONESDM_RUNS=runs_onesdm
python3 track4_onesdm_puzzles.py --device cuda --label gpu1 --puzzles recall --seeds 0,1,2
python3 track4_onesdm_puzzles.py --device cuda --label gpu1 --puzzles copy   --seeds 0,1,2
python3 track4_onesdm_puzzles.py --device cuda --label gpu1 --puzzles blurry --seeds 0,1,2
python3 track4_onesdm_puzzles.py --device cuda --label gpu1 --cost --cost-reps 5
```

The defaults are 3,000 steps, B 64, learning rate 1e-3, AdamW, eval_n 128 sequences per setting, and all five
arms. The stability arms are separate runs with a different `--label`:

```bash
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_addrbias --puzzles recall,copy,blurry --arms onesdm,onesdm_archive --addr-bias 1.0 --seeds 0,1,2
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_memclip --puzzles recall --arms onesdm --mem-clip 0.5 --mem-lr-mult 0.3
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_learndecay --puzzles recall --arms onesdm --learn-decay
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_sum --puzzles recall --arms onesdm --read-mode sum
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_muon --puzzles recall --arms onesdm,yardstick_transformer --opt muon
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_st --puzzles recall --arms onesdm_archive --arc-weighting st
```

The all-SDM arms, one command per puzzle, with `onesdm` beside them so every comparison shares a label:

```bash
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_allsdm --puzzles recall --arms onesdm,onesdm_allsdm,onesdm_allsdm_big --seeds 0,1,2
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_allsdm --puzzles copy   --arms onesdm,onesdm_allsdm,onesdm_allsdm_big --seeds 0,1,2
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_allsdm --puzzles blurry --arms onesdm,onesdm_allsdm,onesdm_allsdm_big --seeds 0,1,2
python3 track4_onesdm_puzzles.py --device cuda --label gpu1_allsdm --cost --arms onesdm,onesdm_allsdm,onesdm_allsdm_big --cost-reps 5
```

No GPU time is measured for these arms. The big arm holds about 10 times the weights, so its optimiser state is
larger, but each token reads the same number of value rows.

**Time estimates. These are NOT measured on a GPU.** They come from the CPU pipeline check scaled by hand, so
treat them as guesses until one run is timed:

- About 3 to 6 minutes per (puzzle, arm, seed) on a 4090 or A100 class card. The archive arm is probably about
  1.5 times slower (a sort and gathers).
- The 16,000-gap evaluation is about 2 million tokens per arm and seed.
- The full campaign is 3 puzzles x 5 arms x 3 seeds = 45 runs, so about 2 to 5 hours on one card, or under one
  hour across 6 cards (split by `--puzzles` and `--seeds`).
- The cost bench takes a few minutes.

### Reading the results

- **A:** compare `onesdm` against `plain` at every gap. The gap between them is what the memory buys. Then compare
  both against `yardstick_transformer` at gaps beyond the training range: does the SDM hold where the
  transformer's length generalisation breaks?
- **B:** accuracy on the second copy, by length. Copying needs a previous-token step and a content match, which
  is the induction pattern done with SDM reads.
- **C:** accuracy should fall slowly as corruption rises. A sharp cliff at 0.25 means the addressing is exact
  match, not fuzzy.
- In every curve, watch `grad_mem` against `grad_rest`. A climbing `grad_mem` is the earlier mixer's failure.

## What is untested

- **Nothing has trained on a GPU.** The only runs are the self-tests and small CPU pipeline checks. Those checks
  run few steps, at small batch, on a loaded machine. They prove the pipeline and nothing about the model.
- **Capacity at long gaps is unknown.** With the fade-1 head, every noise token still writes (weakly, once the
  gate learns). Whether 16,000 noise writes blur 8 pairs past use is exactly what puzzle A measures.
- **The archive's pointer choice is untrained.** Only the scoring of fetched candidates learns. Whether useful
  pointers come back at long gaps is open.
- **The disjoint alphabets make the gate's job easy.** Noise can be recognised by token identity alone. An
  overlapping-alphabet variant is not built.
- **The single-GPU text trainer is wired (lane TEXTWIRE, 2026-10-07).** `track4_sdmonly_train.py --arm onesdm`
  (and `onesdm_allsdm`, `onesdm_allsdm_big`, `plain`, `yardstick_transformer`) trains these models on text, with
  `--bench-steps`, `--param-report` and `--rt-reset-at-bos`; the fine-tune tool runs chat and guy on their
  checkpoints. The multi-GPU trainer (`track4_sdmonly_train_ddp.py`) still refuses these arms. `torch.compile` of
  the onesdm body is untested on CUDA (`--compile-body`).
- **The cost comparison is a prefill comparison.** For the yardstick, the decode-step cost and its key-value cache
  are reported analytically (`decode_state_bytes`), not timed. `forward_sequential` is the SDM decode step, but it
  is not timed in the bench.
- **The sdmonly baseline is only causal in eval mode.** Its batch norm uses batch statistics while training, which
  mixes positions slightly. It is causal in eval mode, where every puzzle score is taken.
- **The all-SDM arms have not trained on a GPU.** On the CPU the self-test's positive control (one-pair recall,
  d 64, 2 layers, 300 steps, one seed) reaches 1.000 for `onesdm_allsdm`. **That control does not show the trained
  table does any work.** With the table's values frozen at zero, which leaves only the run-time memory, the same
  control also reaches 1.000. A single-seed CPU probe at 2 and 4 pairs (same budget) gave trained 0.572 and 0.381,
  frozen 0.555 and 0.420, `onesdm` 0.496 and 0.426, chance 0.125. At this size and budget the probe does not separate
  them in either direction. The GPU puzzles are where the table's value is measured.
- **The table's learning rate is the body's.** Product-key memory papers often train the value table faster. The
  values here use the body learning rate and no weight decay; a separate rate for them is not built.
- **The previous-token mechanism must be learned.** A fade-0 head reads the last write only where the read and
  write addresses overlap. At init they overlap by chance (with M 1,024 and k 16, about 0.25 shared locations per
  pair). `--addr-bias` makes the overlap total at init. It is the most promising untested default: in one
  single-seed CPU check on a loaded machine it looked better than the default, which is not evidence. The GPU
  campaign runs both.
