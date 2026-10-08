# A_q18 AutoTrust 24.9 s claim, Ember optimizer, Token Geometry paper

## (a) AutoTrust README

- Query: GET https://raw.githubusercontent.com/AutoTrustAI/nanogpt-speedrun-sota-by-guru/master/README.md
- Tool: python3 urllib (api.github.com was rate-limited at 10:52 UTC: 'API rate limit exceeded', HTTP 403)
- Fetch: 2026-10-03T10:52:33.869675Z 200

```
**AUTOTRUST AI  ·  SCIENCEGURU  ·  RESEARCH**

# New Record：ScienceGuru Cuts the NanoGPT Speedrun to 24.9 Seconds, 3X Faster than Recursive's June record of 75.4s

Running Guru Turbo 1.2, AutoTrust’s research platform trained GPT-2 Small to the 3.28 validation-loss target in 24.90 seconds on eight H100s: 2.71× faster than the current official record.

September 25, 2026  ·  ScienceGuru  ·  Guru Turbo 1.2  ·  NanoGPT Speedrun

Today we are releasing ScienceGuru’s result on the NanoGPT Speedrun, the open benchmark that asks how quickly a GPT-2-sized model can be trained to a fixed quality bar. Running Guru Turbo 1.2, ScienceGuru trained the model to a mean validation loss of 3.2750 in a mean of 24.90 seconds on eight H100 GPUs, across five preregistered seeds with every run kept.

That is 2.71× faster than the current official record, Canonical Token Masking (#91, about 67.56 seconds), and 1.60× faster than ANVIL2 (39.91 seconds), the fastest open submission we checked. The code, logs, source hashes and a verification script are open at [github.com/AutoTrustAI/nanogpt-speedrun-sota-by-guru](https://github.com/AutoTrustAI/nanogpt-speedrun-sota-by-guru).

![ScienceGuru NanoGPT Speedrun scorecard](assets/blog-nanogpt-scorecard.png)

*Five-seed mean on 8×H100. The result is self-reported and is not yet an accepted leaderboard record.*

## Why the NanoGPT Speed run

The speedrun fixes the data, the hardware and the target, a FineWeb validation loss of 3.28 or lower on one 8×H100 node, and measures only training time. Any gain has to hold up on a codebase the community has optimized for more than two years: 91 official records have taken training from 45 minutes to about 67.6 seconds, through changes to optimizers, architecture, numerical precision, kernels, communication and data loading.

That makes it a demanding test for automated research. In June, [Recursive reported](https://www.recursive.com/articles/first-steps-toward-automated-ai-research) a 77.5-second solution from its automated research system, and its faster ReLU² kernel became official record #87.
```

Reading: self-reported, dated September 25, 2026, five seeds, mean val loss 3.2750, mean 24.90 s on 8xH100. The README itself says: 'The result is self-reported and is not yet an accepted leaderboard record.' It is not in the modded-nanogpt README table (A_q01). State: POST CLAIM ONLY (the repo is the claimant's own).

## (b) Ember README

- Query: GET https://raw.githubusercontent.com/katop1234/ember/master/README.md
- Fetch: 2026-10-03T10:52:32.876832Z 200

```
<p align="center">
  <img src="ember-logo.png" alt="Ember" width="440">
</p>

<p align="center">
  📄 <b>Paper:</b> <a href="https://arxiv.org/abs/2607.01455">Token Geometry (arXiv:2607.01455)</a> — accepted at the <b>Sci-FM</b> and <b>MOSS</b> workshops @ <b>COLM 2026</b>
</p>

> 🏁 **Submitted a NanoGPT speedrun record PR** — <a href="https://github.com/KellerJordan/modded-nanogpt/pull/346">PR&nbsp;#346</a>, in review (July 23, 2026): 1384 steps, −6 vs. prior record. n=30 replications on 2×8×H100 (Prime Intellect + Vast), mean final val loss 3.2788 — one-sided t-test vs. the 3.28 target: p&nbsp;=&nbsp;0.000125.

The embedding table and LM-head are a language model's read/write interface between discrete
tokens and continuous computation. Their gradient geometry is different from dense hidden
weights — Ember exploits that: a drop-in optimizer for token tables using **O(V + D)** state
instead of Adam's **O(2VD)**, at matched quality. ~**1500× less** optimizer memory at 50K
vocab × 768 dim, growing with vocabulary.

Row × column factored second moment, no first moment. One knob: `beta2` (default `0.999`).

**Why it helps:**
- **Memory:** state is ~1 MB → replicate it, never shard it. Token tables drop out of
  ZeRO/FSDP optimizer-state sharding.
- **Distributed:** row-sharded tables sync with one ~D-float all-reduce per step
  (put it on a dedicated communicator — see `ember.py`).
- **Deterministic:** contiguous reductions only, no atomics → bitwise reproducible at
  fixed world size.
- **Cheap step:** touches only gradient + weights (no m/v buffers) — ~3× less memory
  traffic than Adam's step.

## Install

```bash
pip install -e .
```
Or copy [`ember.py`](ember.py) — the whole repo is one file. PyTorch ≥ 2.1.

## Quickstart

```python
from ember import Ember
# was: optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Ember(model, lr=1e-3)
```

Given the model, Ember routes automatically: factored update on token tables (every
`nn.Embedding` + the LM head, tied weights de-duped), internal **AdamW** on everything else.
Standard `step()` / `zero_grad()` / `state_dict()`. `betas`/`eps`/`weight_decay` apply to the
AdamW side; `beta2=` overrides Ember's; `body_lr=` overrides the non-table lr.

## Different body optimizer (e.g. Muon)

```python
from ember import Ember, split_embedding_params

emb, other = split_embedding_params(model)
opt_emb   = Ember(emb, lr=1e-3)
opt_other = Muon(other, lr=2e-2)   # or torch.optim.AdamW(other, ...)
```

## Measured (July 2026)

- **Parity with tuned Adam on Adam's home turf**: on the
  [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt), swapping
  Adam→Ember on the token tables at each optimizer's own optimum differs by ~0.001 val
  loss — within seed noise (n=10 vs n=15 at 8×H100).
- **Memory:** 2 GB → ~400 KB optimizer state at Pythia-2.8B; −3 GB peak VRAM at speedrun scale.
- **Stability:** Adam's dense second moment goes stale on rare rows and throws 10²–10⁴×
  oversized steps late in training; Ember's row statistic stays 1–3× throughout.
- **One config for the whole token interface** — input table + LM-head, no per-table tuning.

## Integration notes

- **HF Trainer schedulers can zero your lr** — verify `param_groups[i]['lr']` ≠ 0 after the first scheduler step.
- **We recommend keeping the state at fp32** — we noticed bf16 underflows on rare rows.

```

## (c) Token Geometry, arXiv 2607.01455 (WebFetch of https://arxiv.org/abs/2607.01455, 2026-10-03 ~10:53 UTC)

- Authors: Kathan Shah. v1 2026-07-01, v3 2026-07-15 (inside the widened window from 2026-06).
- Abstract verbatim: "We introduce Ember, a lightweight optimizer for embedding and LM-head matrices that utilizes O(V + D) VRAM, instead of Adam's O(2VD), and forgoes the need to shard both token table optimizer states. We provide empirical evidence that Ember scales effectively across batch size and parameter count. We show that the optimization trajectory of tokens can be well described by a simple 1D ray..."
- The abstract carries no loss number. The parity claim (about 0.001 val loss, within seed noise) and the 10^2-10^4x oversized-step claim are stated in the README only, not in the abstract.

## (d) modded-nanogpt PR 346 (Ember inside a speedrun submission): see A_q03. Open, not merged.
