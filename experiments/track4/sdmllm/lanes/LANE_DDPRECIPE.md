# LANE DDPRECIPE

Goal: `track4_sdmonly_train_ddp.py` trains the SDMONLY recipe on N GPUs of one box. It is proven equal to the
single-GPU trainer `track4_sdmonly_train.py`, and its checkpoints move between the two trainers in both
directions.

## Status

Checkpoint 2 (2026-10-05). Done, pending JIMOTHY's landing.

- DONE: the DDP loop runs the recipe: Muon on the body (`--body-opt muon`), `--accum` with `no_sync`,
  `--sched wsd`, the compiled chunked loss, `--keep-at`, the cooldown fork (`--init-ckpt`,
  `--cooldown-tokens`) and resume.
- DONE: these are refused by name:
  - `--arm qwen`
  - `--store-opt sparse`
  - `--rt-shuffle`
  - an `--accum` that does not divide `--B`
  - any later trainer flag the loop does not know
- DONE: after each step the batch-norm running statistics are combined to what one device would hold.
- DONE: rank check. At every checkpoint the ranks gather a bit-exact digest of every parameter, buffer and
  Muon buffer. The run stops if the ranks differ.
- DONE: the checkpoint layout equals the single trainer's, plus a `ddp` record. `cfg["data"]` records the
  train length, a fingerprint, the global batch and T. Both trainers refuse a resume that would read other
  batches. `--data-n` (both trainers) reads the first N tokens of a longer file.
- DONE: torch.compile's DDP graph split is off by default; `--ddp-graph-split` turns it on. With it on,
  world 1 is not bit-equal to the single trainer: the embedding differed by 8.8e-5 after 6 steps.
- DONE: `SDMONLY_DUMP_GRADS=DIR@steps` (environment variable, both trainers, tests only) saves the pre-clip
  gradient by name.
- DONE: `track4_sdmonly_vast_arm.sh` accepts the trainer `ddp`.
- DONE: the self-test passes 48 of 48 checks on CPU with gloo: about 3.5 minutes on a quiet M5, about 8 minutes at load 130.

## The equivalence

World W at `--B B --accum A` equals the single trainer at `--B W*B --accum W*A`. The 2 x 5090 form of the
run line `--B 512 --accum 64` is `--world 2 --B 256 --accum 32`.

The self-test measured:

- world 1: bit-equal to the single trainer (every loss, TEST bpb, every tensor).
- world 2, gradient from the same state: equal to 3e-7 relative.
- over 12 steps: losses equal to the logged 1e-4, TEST bpb equal, parameters 1.6e-5 relative in L2.
- Adam's first update of an element is about ±0.44 lr for any non-zero gradient. So a gradient component
  within float noise of zero can take either sign. One such flip in a 12-step probe moved one tensor by 2%
  in L2. For this reason parameters over many steps get a 5e-2 L2 bound, and the strict check is the
  gradient from an identical state.

## Mutations (each one restored; sha256 checked after each)

| mutation | caught by |
|---|---|
| M1 drop `no_sync` | accum: all-reduce calls per step 3,4,4 against 1,2,2 |
| M2 Muon matrices out of DDP's reducer | the in-run rank check stops the run at step 6; ranks tests red |
| M3a checkpoint key `model` renamed | the run cannot score its own checkpoint (`KeyError: 'model'`); parity1 and resume red |
| M3b checkpoint `opt` without Muon's buffers | parity1 optimiser-layout check; resume 2 -> 1 refused by the single trainer |
| M4 drop the batch-norm combine | the in-run rank check stops the run |
| M4b drop the combine and the rank check | parity2: batch-norm counts differ |
| M5 drop `train_n` from the resume check | length: both refusal checks red (the fingerprint still refuses, without naming the length) |
| M6 every rank draws rank 0's rows | batch tests and every parity2 check; the B-4 control becomes equal |
| M7 drop the 1/accum loss scale | parity2: gradient 1.0 relative, losses 12 |

## Not equivalent, by design or by limit

- At world 2 and above the float sums run in another order, so a run is not bit-equal; see the bounds above.
- The combined batch-norm running statistics are computed in float64, so the buffers differ from one device's
  sequential float32 updates at the last bit.
- `--sync-bn` equals one device only at `--accum 1`.
- NCCL was not exercised: the self-test runs on CPU with gloo. The first GPU run should be a short `--bench`;
  check that it prints `ranks_identical: true`.

## Remains

- JIMOTHY lands the branch, then runs the bench on the 2 x 5090 box.
