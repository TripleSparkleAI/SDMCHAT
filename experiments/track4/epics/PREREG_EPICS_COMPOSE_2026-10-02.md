# PREREG · EPICS COMPOSE · sealed 2026-10-02 before any job fired

## What runs

All three runs start from the same checkpoint: SDMCHATS's `SW` (the SDM read, d 512, 8 back plus five moving
averages, 300M FineWeb-Edu tokens), its final `~/settle24/ck/main/sdmchats_SW_d512_s0_300M/last.pt`. The recipe is
SW's own (`--d 512 --f 1376 --n-back 8 --decays 0.5,0.8,0.9,0.97,0.99 --store-lr-mult 3 --store-wd 0 --value-init
zero --qgrad 0 --softness 0.25`), with the continuation learning rate the chat continuations used (`--lr 1e-3`, warmup
2%, cosine to 10%), seed 1, B 32, T 256.

| run | data | tokens |
|---|---|---|
| `epics_BASE` | none (evaluation only) | 0 |
| `epics_EP12` | `epics_train.u32` (6,023,067 tokens, about two passes) | 12,000,000 |
| `epics_FW12` (the control) | `train_big.u32` (FineWeb-Edu) | 12,000,000 |

Every run reports bits per byte on the held-out epic lines (`epics_test.u32`, 315,730 tokens, 1,233,487 bytes:
every 20th block of 64 lines of every work, capped at 150,000 characters per epic) and on FineWeb-Edu TEST (S0's
3,873 windows). The shards and their split rule: `track4_epics_make_token_shards.py`, `epics_shards_provenance.json`.

## Predictions (sealed)

1. `epics_BASE` on the epic test lines: between 1.75 and 1.95 bpb (old verse and archaic spelling read worse than
   the web text SW trained on; SW's own FineWeb TEST is near 1.62).
2. `epics_EP12` lowers the epic test bpb by 0.15 to 0.35 against BASE.
3. `epics_FW12` moves the epic test bpb by less than 0.03 against BASE.
4. THE CLAIM: EP12's epic test bpb is lower than FW12's by at least 0.10. If the gap is under 0.10 the claim fails and
   the report says so.
5. Forgetting: EP12's FineWeb TEST rises by 0.02 to 0.08 against BASE; FW12's FineWeb TEST moves by less than 0.02.

Samples from EP12 and BASE are drawn after the numbers, from the same three cues, and shown beside the numbers. They
carry no score.
