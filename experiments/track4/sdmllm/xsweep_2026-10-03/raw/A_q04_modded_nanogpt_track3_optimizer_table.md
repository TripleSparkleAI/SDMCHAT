# A_q04 modded-nanogpt optimization track (track 3) table + optimizer PRs

- Query: GET https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/records/track_3_optimization/README.md
- Tool: python3 urllib
- Fetch: 2026-10-03T10:47:44.104692Z 200
- Table rows: 48 (rows 1-46 plus header)

## Benchmark definition, verbatim

> Runs must not modify the dataset, batch size, or architecture used by the baseline. Runs also must not perform more than one forward-backward pass per step.

## Verbatim rows 1, 2, 36, 44, 45, 46

```
| 1 | 3600(!) | 3.2777 (n=1)Ⓧ | [Muon](https://kellerjordan.github.io/posts/muon/) with aux Adam, lr=.02 wd=.01 | 2026/04/26 | [log](results/7b8270c5-a9cd-4a73-b7d8-5d86a2d1e428.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 2 | 5625 | 3.2790 (n=1)Ⓧ | [Adam](https://arxiv.org/abs/1412.6980) lr=0.0015 betas=(0.9, 0.95) warmup_steps=250 (note: this is most likely undertuned) | 2026/04/26 | [log](results/a63a68d1-24aa-4a22-af9a-224e43209ea4.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 8 | 3250 | 3.2778 (n=10)✓ | [NorMuon](https://arxiv.org/abs/2510.05491)[H](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd) (Muon NS direction + Adafactor-style row/col variance preconditioning, then hyperball constraint on hidden matrices) with per-module init std (attn.proj std=.026, mlp.proj std=.031, mlp.fc std=.031, qkv default), lr=.018 mu=0.95 beta2=0.95 h_cooldown_frac=1.0 aux_cooldown_frac=.4, end 25 steps early | 2026/04/30 | [log](results/20260430_normuonh/f45b5dcf-16bb-4e83-b5c7-4ef4981f0e9f.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/273) by [@kaiyue-wen](https://github.com/kaiyue-wen) |
| 36 | 3250 | 3.2787 (n=10)✓ | Tuned baseline Muon + aux AdamW hyperparameters: Adam embed/proj/1D lr=.7/.004/.015 wd=0.001, Muon lr=.025 wd=.05 | 2026/06/11 | [log](results/20260610_tuned_baseline_3250/263ea3c4-2b13-4adf-8a71-0410386b20e1.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/323) by [@konstmish](https://github.com/konstmish) |
| 44 | 2750(!) | 3.2789 (n=20)✓ | Setup from #41, plus SOAP-Muon on all hidden matrices w/ precondition_frequency=1 (prev. had SOAP on MLP + attn.proj w/ precond_freq=1), tune auxiliary β2's, double mu cooldown, set rademacher init CGI α=.125, and remove neutral geometry modules including (Circuit,Contra)Muon and Aurora | 2026/06/10 | [log](results/20260609_soap_f1_auxb2_clean/H100_ff29b392-e7b7-453e-b9d5-a7dfe0605dd0.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/321) by [@ypwang61](https://github.com/ypwang61) and [@nooraovo](https://github.com/nooraovo) |
| 45 | 2720(!) | 3.2786 (n=10)✓ | Setup from #44, plus at the final step blend the weights towards EMA(horizon = 150 steps) | 2026/06/12 | [log](results/20260611_tailema_2720_submission/8878c81f-5f73-461f-a41e-c0887e15c1ca.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/325) by [@jn2clark](https://github.com/jn2clark) |
| 46 | 2690(!) | 3.2783 (n=8)✓ | Setup from #45, plus RowUpdateFloor per-output-row u/w-floor, and Cautious Weight Decay `CWD=0.025` | 2026/06/19 | [log](results/20260619_cwd_rowfloor_tailema/A40_seed0_5c87fa44-7ca7-4d54-971d-d952f9b15792.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/328) by [@ypwang61](https://github.com/ypwang61) |
```

## Reading

- The merged table ends at row 46, dated 2026/06/19 (2690 steps). No merged track 3 row inside 2026-08-01..2026-10-03.
- Open track 3 PRs in the window (A_q03): 377 (2580 steps, n=16, anchor gradient + fp32 master embedding), 369 (3060), 368 (3010), 370 IsoMuon (3190 on the tuned Muon baseline), 362 Muon-NSR (3250), 359 (3065), 357 (3160), 353 AdamW baseline (4950), 351 (3125).

### PR 353: Tune AdamW baseline to 4950 steps (created 2026-08-14T15:05:52Z)

- Fetch: 2026-10-03T10:48:07.349975Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/353
- State: open

```
## Changes

- `train_steps`: 3250 -> 4950
- Replace Muon on the 2D block-matrix parameters with AdamW
- AdamW learning rates:
  - embedding: 0.7
  - projection: 0.004
  - 1D parameters: 0.015
  - 2D block matrices: 0.001
- AdamW weight decay:
  - embedding, projection, and 1D parameters: 0.001 -> 0.002
  - 2D block matrices: 1.0
- 2D block-matrix AdamW betas: `(0.9, 0.9)`
- Add a 100-step linear warmup from `1e-7`

Validation logs are included in:

`records/track_3_optimization/results/20260701_tuned_adamw_steps_4950_wd_1_wd2_0.002_matrix_b2_0.9/`

## 10-run result at 4950 steps

- mean val loss: 3.277604
- median val loss: 3.277095
- std: 0.002516
- min: 3.274590
- max: 3.282550

## README significance criterion

```text
(3.28 - mean) * sqrt(10) = 0.007577
```

This passes the required `>= 0.004`.

```

### PR 364: Endgame EMA weight blending (port of Track 3 Tail-EMA readout) (created 2026-09-05T12:39:46Z)

- Fetch: 2026-10-03T10:48:08.148302Z 200
- URL: https://github.com/KellerJordan/modded-nanogpt/pull/364
- State: closed

```
Blends an EMA of the tail-of-training weights into the final weights at the last step:

`w <- (1 - gamma) * w_final + gamma * w_ema`

### Prior art

This is the Tail-EMA eval readout from the Track 3 optimization benchmark (PR #325, reused in #328) — same formulation, including excluding the embedding table. The main track currently has **no weight averaging of any kind**, so this is a port: re-tuned for this trainer and reimplemented against its sharded optimizer.

### Result (8xH100, 1285 steps, FP8 and compile on)

| gamma | 0.00 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 |
|---|---|---|---|---|---|---|
| val_loss | 3.2778 | 3.2755 | 3.2754 | **3.2754** | 3.2756 | 3.2760 |

**-0.0024 at gamma = 0.25-0.30.** Broad optimum: horizon 130-170 x gamma 0.25-0.35 all land within 0.0002, and update stride 1/4/16 are indistinguishable — so the setting is not sensitive.

**Cost: 0.070 ms/step** steady state (~0.12% of a 58 ms step), timed with CUDA events. At the measured end-of-run slope (0.000477 loss/step) the gain is worth ~319 ms against ~103 ms of cost.

### On the measurement design

gamma is swept **within a single run**, against the same final weights, the same validation data, and the same forward pass — the only thing that varies is gamma. This is not a cost-saving compromise; it is a strictly better instrument for this quantity:

- `gamma=0` reproduces that run's own printed `val_loss` **exactly**, which doubles as a correctness assertion on the blend-and-restore path.
- There is no sampling variation to average away. A cross-run A/B estimates the difference *through* init and floating-point noise; this design removes that variance source rather than averaging it down.
- It resolves differences of **0.0001**. Matching that precision cross-run, at the sigma of this repo's own record logs (0.0017, n=26), would take roughly **2300 runs per arm**. That precision is what makes the horizon/gamma plateau and the stride-invariance measurable at all — a cross-run design could not establish those at any realistic n.

The cost is measured the same way: directly, with CUDA events, rather than inferred by differencing noisy end-to-end wallclocks. On my hardware end-to-end sigma is ~940 ms, which cannot resolve a ~300 ms effect; the event timing sidesteps that entirely.

### Caveat

What I have **not** done is demonstrate the wallclock gain end-to-end. At the sigma visible in this repo's record logs (549 ms, n=26) that needs roughly **30 runs per arm** — the scale this track already operates at, but more 8xH100 time than I have. The loss gain and the per-step cost are bot
```

