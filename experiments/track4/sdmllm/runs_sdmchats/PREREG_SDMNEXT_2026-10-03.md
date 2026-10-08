# PREREG SDMNEXT, 2026-10-03

Sealed at 00:40Z on 2026-10-03, before either run below has started. The commit that adds this file is the seal;
both jobs are submitted to the Spark queue after it.

Stamps at the seal. Spark GB10 (spark-f9cd): load 0.36 / 0.34 / 0.25, GPU 1%, MemAvailable 114 GB, NVIDIA driver
580.159.03, torch 2.14.1+cu130, CUDA 13.0, powermode unknown (Linux). The M5 is at power mode 1 (LOW POWER) and load
31 of 18 cores, so no M5 timing is taken now; any browser speed run waits for high power and a quiet machine.

Trainer, data and TEST windows are the ones every earlier row used: `track4_sdmllm_train_one_arm.py` (sha256 prefix
6d6c990dd379fc96, identical on the M5 and the Spark), train_big, the 3,892 non-overlapping T=256 TEST windows
(4,824,566 bytes), chat_test scored on up to 4,000 windows (1,022,736 tokens, 4,819,015 bytes). Queue:
`~/settle24/code/q` through `~/settle24/code/sdm_arm.sh`, named SDMCHATS-* so `~/settle24/sdmchats/memguard.py`
covers them.

## Run 1: NWL, the no-store control at width 768

`sdmchats_NWL_d768_s0_600M`: arm `sdm_nostore`, seed 0, 600,000,000 tokens of train_big, d 768, readout 2,064,
8 tokens back, decays 0.5, 0.8, 0.9, 0.97, 0.99. It is SWL's recipe with the store off, exactly as NW s0 was to
SW s0 at width 512 (NW's command carried no store flags, and neither does this one).

Command: `nice -n 10 ~/settle24/code/sdm_arm.sh sdmchats_NWL_d768_s0_600M sdm_nostore 0 600000000 --d 768 --f 2064
--n-back 8 --decays 0.5,0.8,0.9,0.97,0.99 --extra-val chat_test --ckpt-every 1000`

Controls already measured: SWL (the SDM read, same recipe) TEST 1.40285, CHAT 1.63517, zero_read 1.54155.
At width 512, two seeds: SW minus NW +0.00014 (window SE 0.00021; seed deltas -0.00017 and +0.00046), and the
seed-to-seed move of one arm 0.0018 to 0.0025. At width 512 and 3B tokens (JIMOTHYVAST) the read lost to no store by
0.0050.

- NL1: NWL's TEST bpb lies in 1.395 to 1.412 (point 1.403).
- NL2 (the question this run exists for): SWL minus NWL lies in -0.004 to +0.004 (point 0.000). The SDM read does
  not beat no store at width 768 either.
- NL3: NWL's CHAT bpb lies within 0.015 of SWL's 1.63517.
- NL4: NWL minus NW s0 (1.43012) lies in -0.035 to -0.020: width and tokens buy the same gain with no store.
- NL5: SWL with its read zeroed (1.54155) is worse than NWL by at least 0.10. The read model leans on its read for
  work the no-store model does with its other weights.

Decision rule, sealed with the predictions. One seed per arm, so the window SE understates the real noise. "The read
helps at 768" is claimed only if SWL minus NWL is below -0.005 (twice the largest seed-to-seed move measured at 512)
AND resolved on the windows (|z| > 2). "The read hurts" needs above +0.005 by the same test. Anything between is a tie.

## Run 2: SWLC, the chat fine-tune of SWL

`sdmchats_SWLC_d768_chat_100M`: SWL's last checkpoint fine-tuned on 100,000,000 tokens of chat_mix (half
smol-smoltalk in the page template, half FineWeb-Edu), the recipe of `sdmchats_SWC_d512_chat_100M`: fresh AdamW,
peak lr 1e-3, warmup 2%, cosine to 10%, store lr x3, no store weight decay, zero value init, query gradient stopped,
softness 0.25, seed 0.

Command: `nice -n 10 ~/settle24/code/sdm_arm.sh sdmchats_SWLC_d768_chat_100M sdm 0 100000000 --d 768 --f 2064
--n-back 8 --decays 0.5,0.8,0.9,0.97,0.99 --store-lr-mult 3 --store-wd 0 --value-init zero --qgrad 0 --softness 0.25
--train-name chat_mix --lr 1e-3 --init /home/hologram/settle24/ck/main/sdmchats_SWL_d768_s0_600M/last.pt
--extra-val chat_test --ckpt-every 1000`

Controls: sdmwide512chat (SWC) CHAT 1.11209, TEST 1.47952 (+0.0496 over SW), zero_read +0.132; SPARK24's WCC
(per-token table) CHAT 1.11825; WCO (chat only) 1.08065. SWL's own CHAT before the fine-tune 1.63517.

- LC1: SWLC's CHAT bpb lies in 1.070 to 1.105 (point 1.090), below sdmwide512chat's 1.11209.
- LC2: SWLC's TEST bpb is worse than SWL's 1.40285 by 0.030 to 0.070 (point +0.047).
- LC3: zero_read on SWLC costs +0.08 to +0.20 TEST bpb.
- LC4: the int8 export agrees with float32 PyTorch on the top-1 token on at least 0.90 of 48 TEST windows (SWL gave
  0.913).
- LC5: in the loop check, at most 1 of 30 chat replies loops (3 seeds, the SDMSCORE `loopcheck.mjs` setup).
- LC6: in the browser, SWLC runs at no less than 0.55 of sdmwide512chat's tokens per second (SWL gave 0.649 of SW).

Promotion rule, sealed. SWLC becomes the chat champion (SDMCHAT opens on it) only if its CHAT bpb is below
sdmwide512chat's on the same chat_test windows with the paired difference resolved (|z| > 2 over 2,000 window
resamples, numpy default_rng(0)), and LC4 holds at 0.90 or better. Otherwise sdmwide512chat stays the default.

## Scoring

`track4_sdmnext_score.py` (this folder): TEST pairs from each run's `test_per_window.npz`; chat pairs from per-window
chat_test nats computed once per checkpoint on the Spark GPU and cached beside it as `chat_per_window.npz`; the same
bootstrap as SDMSCORE. Every prediction above is judged HIT or MISS in writing in the ledger.
