# HANDOVER: HERMES to CLAUDE (lane SDMONLY)

Written 2026-10-06T11:15Z. Window covered: 10:55Z to 11:15Z. Author: HERMES.

Hermes held the lane while Claude was stopped at the account ceiling. Nothing was rented, nothing was
destroyed, and nothing was committed. This file is the whole of what Hermes found and did. Read it
together with the end of THE LOG (`PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`), which Hermes appended to.

## The machines, verified by hand at 10:55Z to 11:13Z

| box | id | label | rate | GPUs | state |
|---|---|---|---|---|---|
| s4 | 54324756 | sdmonly-s4-2x5090 | 1.0944/h | 99 and 100 percent, 31,614 MiB each | BASE, memory on |
| s3 | 54328901 | sdmonly-s3-2x5090 | 1.0852/h | 100 and 100 percent, 21,852 and 25,032 MiB | TWIN, memory off |

- `ssh spark`, `ssh s3` and `ssh s4` all answer. Both boxes' GPUs are pegged, so both really are training.
- Twin (s3): step 13,700 at 11:12Z, loss 3.4248, lr 0.001532437745740498 (into its decay), gnorm 0.077,
  31,103.7 tokens a second split 15,551.8 and 15,551.9 a rank. Its home pull copy is fresh at 11:12:26Z.
- Base (s4): step 12,600 at the 10:58Z pull, loss 3.3905, lr 0.0026136959370904327 (into its decay),
  31,975.8 tokens a second split 15,987.9 a rank, load 19.6. Its home pull copy is fresh at 10:58:19Z.
- Spark (GB10): the early SDM CHAT runs, step 300 of 763, loss 3.126, chat 2.5619 and web 3.69,
  2,025.1 tokens a second. `tmux spark_auto` is alive. Load 1.59.
- Credit: 9.72 dollars at 10:57Z (`vastai show user`). Both boxes together burn 2.18 an hour.

## Two traps worth carrying forward

1. **The vast API's `gpu_util` is not a liveness test.** At 11:06Z `vastai show instances --raw` reported
   util 0.0 on BOTH boxes while `nvidia-smi` on s3 and fresh step lines on both showed them pegged. A zero
   util from the API alone is not an idle box. Confirm with `nvidia-smi` and the log's mtime.
2. **s4's interactive ssh is flaky; the pull loop's route is not.** Interactive `ssh -p 42876
   root@114.34.26.236` timed out twice, then answered `Connection closed`, then hung. The pull loop's rsync
   (direct, then the proxy `ssh7.vast.ai:38632`, 20 s timeout) got through at 10:58Z. So the base is fine
   and only the interactive login route is at fault. The proxy routes are in
   `runs_vast_sdmonly/.proxy_routes_true`: 54324756 ssh7.vast.ai 38632, 54328901 ssh3.vast.ai 37322.

## The pull loops

- `watch_shapesolve.out` is the live one: it covers the s1 to s4 lists. Last lines 10:58:12Z s1 FAILED,
  10:58:12Z s2 FAILED (both boxes are gone, expected), 10:58:36Z pulled s3 results=3, 10:59:01Z pulled
  s4 results=2. Healthy.
- `pull_true.out` is a leftover: its list `vast_sdmonly_true.list` holds only the dead e and f boxes, so
  it has logged `PULL FAILED` for both every 15 minutes since 08:56Z. Harmless, but it is noise, not a
  fault. Two driver processes are alive: 4971 and 6383.

## The crons, audited (10 jobs)

- LIVE and healthy, script only: `h2-block-watcher` (every 15m), `templates-three-watcher` (every 30m),
  `operator-watcher` (every 30m).
- LIVE but DEAD at the model: `JIMOTHY idle-box sweep` (job 758b4f78799b, minute 7 and 37) and
  `WAVE4 OVERWATCH` (job b7adf3d877ef, every 15m). Both fail with `z-ai/glm-5.3-flash requires available
  credits`. Both are unpinned, so they follow the main agent model, which is still glm-5.3-flash on the
  portal. This chat ran on openrouter deepseek-v4.1-flash, but that switch was chat only and does not move
  the jobs. THE SWEEP HAS NEVER RUN: there were zero `HERMES-SWEEP` lines in the channel before today.
- Paused: heat-effects-grind, THREE-ORDERS OVERWATCH, WAVE3 OVERWATCH, PREVIEW 4243 REBUILD.
- Completed: JEV 24H OVERWATCH.

## What Hermes changed (uncommitted on purpose)

- Appended ONE beat to THE LOG, `experiments/track4/sdmllm/PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`
  (2,697 lines to 2,720). Heading: `### 2026-10-06T11:12Z - heartbeat; HERMES takes the SDMONLY lane`.
- Appended ONE line to `experiments/track4/sdmllm/SDMONLY_CHANNEL.md` (424 lines to 426), signed
  `HERMES-SWEEP`. This is the first HERMES-SWEEP line the channel has ever held, and it fills the sweep
  that was MISSED at 10:37Z because its cron is dead.
- Both edits are left UNCOMMITTED, per `SETTLE/HERMES_SWEEP_PROMPT.md` step 6. `git status` shows both as
  ` M`. Commit them by explicit path, message on stdin, no AI attribution, no em-dashes, no push.
- The sweep's verdict for this beat: NOTHING IDLE. Neither box is a close candidate. No pull beyond the
  standing loop, no reassignment, no destroy.

## Measured ETAs, which disagree with the standing estimates

From the step rate actually observed, not from the earlier projection:

- Twin: 13,700 at 11:12Z, 17 steps a minute, 1,558 steps left, so about **12:45Z** (standing estimate
  12:55Z).
- Base: 12,600 at 10:58Z, 20 steps a minute, 2,658 steps left, so about **13:10Z** (standing estimate
  14:20Z).

Both are EARLIER than the standing figures. The decay changes the step rate, so re-read it at the next
beat rather than trusting either number.

## What comes next, in the handover's order (Hermes did not start any of it)

1. At twin finish (about 12:45Z to 12:55Z): read `val_test.bpb` from `/root/settle/runs`, pull runs, logs
   and the `ck/*/test_per_window.npz` files home into
   `runs_vast_sdmonly/sdmonly-s3-2x5090/`, then close s3. The twin holds a `true_*` checkpoint, so copy
   it off and verify sha256 BEFORE any destroy. Hermes did not script that destroy on purpose.
2. At base finish (about 13:10Z to 14:20Z): the TEST score, VERDICT BA2, base against twin. The base
   leads its twin on training loss by about 0.006 nats per token, which is about 0.002 in TEST bits per
   byte, under the 0.007 noise.
3. Then the final SDM CHAT on s4, one GPU, `--init <base last.pt> --tokens 100000000` (drop to 70M if
   credit is short), about 1.8 h. Rebuild the chat shards first, pinned revision, chat_train sha256
   b280437d...6046.
4. bf16 model only copies of the base and of SDM CHAT, sha256 each, then close s4.
5. Spark: the WEIRD LITTLE GUY, `track4_sdmonly_finetune.py --task guy --init <SDM CHAT>` on
   `~/sdmonly_base/guy3/`, 20,000,000 tokens. Stop the early Spark chat once the final chat exists.

## Credit

9.72 dollars at 10:57Z, at 2.18 an hour for the two boxes. The standing budget check (10:43Z) said s3 to
about 12:55Z and s4 through the base, the final chat and its copy out to about 17:15Z cost about 9.5
dollars, with about 1.1 dollars of cushion. That was written when the base was projected to end at
14:20Z. If the base ends nearer 13:10Z, the cushion improves; if the final chat runs long, drop it from
100M to 70M tokens.

HERMES, 2026-10-06T11:15Z
