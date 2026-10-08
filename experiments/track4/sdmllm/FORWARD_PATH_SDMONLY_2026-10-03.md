# SDMONLY · the forward path · 2026-10-03

This is the path from today's measurements to the three models: SDM BASE, SDM CHAT and the WEIRD LITTLE GUY. It
was decided by JIMOTHY (the operator) with the navigator on 2026-10-03. Every model is all SDM, trained from
scratch, with no teacher and no distillation (the law: section 0 of the plan).

## Where we stand (2026-10-03)

- The Muon transformer yardstick leads the best SDM recipe at every budget: 0.068 bpb at 20M tokens and 0.098 at
  200M (wave 7).
- At 200M the trained store reads score worse than no reads, by 0.009. The fading SDM mixer hurts by 0.023; the
  never-fading mixer ties the best recipe, 0.0015 behind (wave 10).
- At 200M the transformer gains 0.0186 from a 1,024 window; the SDM without a mixer gains 0.0002 (wave 10).
- Every run so far trained on a 65,522,263-token shard, so 200M-token runs repeated it about 3 times. The scaling
  readings built on them are withdrawn. From wave 11 every scaling run trains on `train_big`, one pass.

## The path

```
  1  WAVE 11 ──── 10 boxes of 2x RTX 5090 (20 GPUs), at most $12/h, $80, 7 h
     │            scaling grid on fresh data: cn, mq, cnmix; long-window mixer tests
     │            fit L(N, D) = E + A / N^alpha + B / D^beta per family
     ▼
  2  PICK THE BASE SHAPE ── width and tokens from the fitted curves, inside the budget
     │                       ** confirm with the navigator before step 3 **
     ▼
  3  BASE ─────── the long run on train_big, one pass
     ▼
  4  CHAT ─────── SFT on open chat conversations
     │            then RL, exploratory, on the MiMo-format ladder
     ▼
  5  WEIRD LITTLE GUY ── further rounds on Searle and philosophy
     ▼
  6  SITE AND PAPER
```

### 1. Wave 11, the scaling law on fresh data

- **Fleet.** 10 boxes of 2x RTX 5090 (20 GPUs), at most $12 an hour and $80 in total, ending 7 hours after the
  boxes are ready. The RTX 5090 is the value card for this model (108,082 tokens a second at d 768, about 739M
  tokens a dollar, measured 2026-10-03). Ruling: the navigator asked for up to 10 boxes at 16:05Z on 2026-10-03;
  JIMOTHY chose the type and the limits.
- **What it runs.** 46 runs (queues `track4_sdmonly_wave11_q_<e..n>.txt`):
  - the grid: cn (the best SDM recipe) and mq (the Muon transformer, MLP width 8/3 d) at widths 256, 384, 512 and
    768 by 100M, 400M and 1.6B tokens of `train_big`, one pass;
  - cnmix (cn plus 4 never-fading mixer layers) at width 256 and 512, and at T 256, 1,024 and 4,096;
  - long-window mixer tests on the old shard at 200M, comparable with wave 10: T 1,024, 4,096 and 16,384, more and
    fewer layers, more locations, larger k, a wider address, store reads on;
  - second seeds of cn, mq and the mixer.
- **The fit.** L(N, D) = E + A / N^alpha + B / D^beta per family. N is non-embedding parameters, D training tokens.
- **The predictions.** Y1 to Y10, sealed in the plan before the wave runs.

### 2. Pick the BASE shape

Read the fitted curves for cn and cnmix. Choose the width and token count that give the lowest predicted TEST bpb
inside the BASE budget. **The choice is confirmed with the navigator before the long run starts.** The
transformer's curve is reported beside it as the yardstick.

### 3. BASE, the long run

The chosen shape on `train_big`, one pass, warmup-stable-decay schedule, with checkpoints kept for a token curve.
Its own predictions are sealed in the plan before it starts.

### 4. CHAT, then RL

- **SFT** of the BASE on open chat conversations (`chat_mix`), by next-token prediction. Gate: CHAT bpb below
  1.06386 and no looping reply in 30.
- **RL, exploratory.** Decision A (2026-10-03T14:52Z): context is the main line; RL comes later and only after
  CHAT exists. RL can shape behaviour the model already shows sometimes; it cannot add context or knowledge.
- **The ladder.** Tasks in MiMo-V2.6-RL-oss's six-column format: easy rungs first (copy a line, count, close
  brackets, format, one fact, arithmetic, tiny code), then MiMo's Music, General and Code. Train only where a
  rung's pass rate sits between 5 and 95 percent. The loop is nanochat's `chat_rl.py`. Rubric rungs are judged by
  ds4-flash through the M5 Hermes proxy. No MiMo weights are used.

### 5. The WEIRD LITTLE GUY

CHAT given further rounds on his own corpus: Searle-style writing and public-domain philosophy, with web and chat
rows mixed into every batch to limit forgetting.

### 6. Site and paper

Export BASE, CHAT and the GUY to the browser (int8), put them on the site, and fill the paper draft's results.

## Open questions

1. **Does any SDM mixing layer use a long window?** At 20M nothing used one. At 200M the transformer gains 0.0186
   from T 1,024 and the SDM without a mixer gains 0.0002. The mixer is the only all-SDM route; wave 10's X2 and X5
   and wave 11's Y5, Y6 and Y8 decide whether it works.
2. **Do the trained store reads pay at scale?** At 200M on repeated data they cost 0.009. Wave 11's Y7 tests store
   reads with the never-fading mixer; the grid on fresh data says whether the cost holds.
3. **How do the two families scale?** Y2 and Y10 ask whether the transformer gains more from data than cn.
4. **The tokenizer.** The trained 32,768 BPE beat the 129k vocabulary by 0.02069 bpb at equal bytes on the centre
   arm. The champion pair (`w7r_*`) is not yet scored. The BASE uses whichever wins, decided before step 3.

## Links

- The plan, the law and THE LOG: [`PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`](PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md)
- How we train: [`TRAINING_SDMONLY_HOW_WE_TRAIN.md`](TRAINING_SDMONLY_HOW_WE_TRAIN.md)
- The paper draft: [`PAPER_SDMONLY_DRAFT.md`](PAPER_SDMONLY_DRAFT.md)
- The RL ladder spec: [`SPEC_SDMONLY_RL_LADDER_2026-10-04.md`](SPEC_SDMONLY_RL_LADDER_2026-10-04.md)
- The open reference for scaling, evals and RL: [`RESEARCH_SDMONLY_CANONICAL_REFERENCE_2026-10-04.md`](RESEARCH_SDMONLY_CANONICAL_REFERENCE_2026-10-04.md)
- The MiMo study: [`wikis/WIKI_MIMO/00-MASTER-INDEX.md`](../../../wikis/WIKI_MIMO/00-MASTER-INDEX.md) and
  [`05-what-we-copy-for-sdm.md`](../../../wikis/WIKI_MIMO/05-what-we-copy-for-sdm.md)
- The coordination channel: [`SDMONLY_CHANNEL.md`](SDMONLY_CHANNEL.md)
