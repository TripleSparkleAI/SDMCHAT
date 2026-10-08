# SDMONLY · the RL ladder · spec · 2026-10-04

This spec copies the record format and the RL settings of Xiaomi's open RL release, MiMo-V2.6-RL-oss, and defines a
ladder of tasks easy enough for the SDM CHAT model. The tool is `track4_sdmonly_rl_ladder.py` in this folder.

The law of the campaign stands (`PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` section 0): every model is all SDM, trained
from scratch, with no teacher and no distillation. Nothing in this ladder trains on text a stronger model wrote.
Every reward is a rule check against a gold that code computes, or (rung 9 only) a rubric judge that grades a reply
and writes nothing the model learns from.

Why a ladder: our best SDM model scores 1.52 bits per byte and cannot fix a repository. On MiMo's Code, Webdev and
Cyber tasks every reward would be 0, every GRPO group would have zero variance, and the trainer would learn nothing.
GRPO needs groups where some samples pass and some fail.

```
   ✦ rung 10  MiMo Code      2,698 repo tasks, executable tests        reward 0 today, MASKED here
   · rung  9  MiMo General     989 knowledge-work tasks, rubric judge   needs tools + judge, MASKED here
   · rung  8  MiMo Music     1,000 ABC compositions                     8a format gate here, 8b their score
   │ rung  7  code_tiny      a tiny function from its doc string, executable tests
   │ rung  6  arith          small-number arithmetic, number only
   │ rung  5  fact_qa        one fact from facts in the prompt
   │ rung  4  format         put a given word into a stated format
   │ rung  3  brackets       close open brackets, innermost first
   │ rung  2  count          continue a count
   ▼ rung  1  copy_line      copy a line exactly
      each rung: 5 levels · train only where pass rate sits in 5 to 95 percent
```

## 1. Their spec, as found

### 1.1 The dataset

`huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss`, Apache 2.0, one train split. Read on 2026-10-04 through the
datasets-server API (`/info`, `/rows`, `/statistics`, `/parquet`), the repository tree API, and the five parquet files
(all five checked in full; they are small).

| subset | rows | parquet bytes | data_source | ability | verifier (their README) |
|---|---|---|---|---|---|
| code | 2,698 | 13,314,620 | `opensource-code` | `swe` | executable tests |
| webdev | 2,093 | 4,120,378 | `blackbox/webdev` | `webdev` | visual grading |
| cyber | 1,000 | 227,632 | `arvo` | (column absent) | rule checks |
| music | 1,000 | 135,467 | `music` | `music_generation` | rule checks |
| general | 989 | 3,122,244 | `mimoagent/general_agent` 925, `mimoagent/terminal_bench` 64 | `agent` | rubric judging |
| total | 7,780 | | | | |

The 12.1 GB size of the repository is the `general/envs/` tree: one directory per General task with its workspace
files, sqlite state for each simulated system, tool servers and verifier files. The parquets are about 21 MB in all.
The earlier addendum's 2,700 and 2,090 were rounded; the exact counts are 2,698 and 2,093.

**Columns.** Code, webdev and general carry six columns. Cyber lacks `ability`. Music lacks `agent_name`.

| column | type | value |
|---|---|---|
| `data_source` | string | the subset tag above; in verl it groups metrics, and the launcher names the scorer explicitly |
| `ability` | string | as above |
| `agent_name` | string | `mimo_swe_agent` in every row that has the column |
| `prompt` | list of {role, content} | exactly one turn, role `user`, in all 7,780 rows |
| `reward_model` | {style, ground_truth} | `style` is `rule` and `ground_truth` is the empty string in all 7,780 rows |
| `extra_info` | struct | see below |

**extra_info.** Code, cyber, general and webdev carry `{index: int64, instance_id: string, dataset_type: string,
instance_json: string}`, where `instance_json` is a JSON document. Music carries `{src_id, lang, tag, length,
nvoice_want, bpm, meter}` and no `index`.

| subset | keys inside `instance_json` (one key set per subset unless noted) |
|---|---|
| code | cwd, dataset_type, docker_image, instance_id, problem_statement, test_command, test_patch, verifier_timeout_sec |
| cyber | cwd, dataset_type, description, docker_image, instance_id, problem_statement |
| webdev | category, cwd, dataset_type, docker_image, instance_id, problem_statement, task_id |
| general (925, `general_agent`) | cwd, dataset_type, docker_image, env_task_dir, instance_id, num_turns, problem_statement |
| general (64, `terminal_bench`) | agent_timeout_sec, allow_internet, category, cpus, cwd, dataset_type, difficulty, display_description, display_title, docker_image, ext_id, instance_id, memory_mb, problem_statement, schema_version, storage_mb, subcategory, tags, tests_files, verifier_timeout_sec |

So the gold never sits in `ground_truth`. Code carries its tests in `test_patch` plus `test_command`; terminal_bench
carries base64 test files in `tests_files`; general_agent carries its rubric in `general/envs/<id>/verifier_meta.json`.

**Music.** All 1,000 rows: `lang` zh 504 and en 496; `length` short 504 and long 496; `meter` 4/4 746, 3/4 147,
2/4 57, 6/8 38, 7/8 12; `nvoice_want` 1 to 6 (2 is the mode, 475 rows); 79 tags; prompts 141 to 825 characters.
Each prompt asks for a complete piece in ABC notation with a key, tempo, meter, bar count, instrumentation and voice
count. Two small differences from the builder in the fork (`recipes/design/music/build_parquet.py`): the builder
writes `ability` "music" and an `extra_info.index`; the published file has `music_generation` and no index.

### 1.2 The verifiers

- **Music** (`recipes/design/music/scorer/`): `extract_abc` takes the first fenced block holding an `X:` header, else
  the text from the last `X:` header. `abc2midi` renders it; a missing binary raises rather than scoring 0. A sample
  is rejected (reward 0) when abc2midi reports any error, ten or more bar warnings, a blank line inside the tune, or a
  MIDI channel with conflicting programs. Otherwise the reward is the "human-likeness" total over 100: 85 percent from
  18 MIDI features scored against human percentile bands in 6 groups (rhythm 0.22, texture 0.20, acoustic 0.14,
  tonal 0.16, register 0.16, structure 0.12), and 15 percent from Jensen-Shannon agreement of pitch-class, interval
  and duration histograms with a human corpus. The builder's docstring states that nothing reads `extra_info` at
  scoring time, so the score does not check the asked meter or voice count.
- **General** (one task read in full, `s3k_0453_consulting_bizops_analytics_en_t4_rl_002`): `verifier_meta.json` lists
  items `{id, tier (critical or important), method (llm or rule), question, pass_anchor, files}`. `verify.py` sends the
  judge one item at a time with a fixed prompt (in Chinese) that asks for 0 or 1, no partial credit, judged only on
  the evidence text. Rule gates run first; a failed gate makes the task 0. The score is the weighted mean of item
  scores (weight 1 unless set). The judge comes from environment variables `GA_JUDGE_URL`, `GA_JUDGE_KEY`,
  `GA_JUDGE_MODEL` (default `gpt-4o-mini`) and `GA_JUDGE_API` (`chat` in `general.sh`). If no judge answers, the
  rollout is MASKED (`reward_error`), never scored 0. With `REWARD_BINARIZE=True` and threshold 1.0, the trainer
  turns any score below 1.0 into 0.
- **Code**: the agent works in a Docker image; `test_command` runs the tests after `test_patch` is applied. The
  fork's `recipes/code/reward.py` is only a fallback that reads a reward the runner already computed.

### 1.3 The GRPO settings in the fork

Read from `github.com/XiaomiMiMo/verl` at commit `a2ad9f6160b03ff2d47e59832bfb6b289f37c917` (branch `mimo-oss`,
2026-09-26), cloned at `wikis/WIKI_MIMO/code/mimo-verl`. Values are the launch-script defaults, which override the
yaml where both set a value. The env examples state that their values are "the reference run's" values.

| setting | code | general | music | webdev | cyber |
|---|---|---|---|---|---|
| launch script | `scripts/code/train.sh` | `scripts/general/general.sh` | `scripts/design/music.sh` | `scripts/design/webdev.sh` | `scripts/arvo/arvo.sh` |
| advantage | GRPO | GRPO | GRPO | GRPO | GRPO |
| divide advantage by group std | no | no | yes (not set, so verl's default `true` applies) | yes (same) | no |
| group size N | 16 | 8 | 8 | 8 | 16 |
| prompts per step | 64 | 64 | 32 | 32 | 64 |
| PPO mini batch | 64 | 64 | 32 | 32 | 64 |
| learning rate | 1e-6, wd 0.01 | 2e-6 | 1e-6 | 1e-6 | 1e-6 |
| KL loss / KL in reward | off / off | off / off | off / off | off | off |
| clip ratio | 0.2 (low 0.2, high 0.2, dual-clip c 3.0) | 0.2 | 0.2 | 0.2 | 0.2 |
| loss aggregation | prompt-mean | prompt-mean | not set | prompt-mean | prompt-mean |
| drop zero-variance groups (filter_groups) | on | on | off | off | on |
| entropy bonus | 0 | 0 | 0 | 0 | 0 |
| sampling | T 1.0, top-p 0.95, top-k 20 | T 1.0, top-p 0.95, top-k -1 | verl defaults (stated deliberately) | T 0.6, top-p 0.95, top-k 20 | T 1.0, top-p 0.95, top-k -1 |
| prompt / total context, tokens | 16,384 / 262,144 | 49,152 / 262,144 | 16,384 / 116,384 | 16,384 / 262,144 | 16,384 / 262,144 |
| length | 200 steps, 10 epochs | 5 epochs | 4 epochs | 1 epoch | 5 epochs |
| GPUs (nodes x per node) | 8 x 8 | 4 x 8 | 8 x 8 | 8 x 8 | 8 x 8 |
| other | 4 agent harnesses mixed by step hash, seed 20260911 | length penalty on (max 0.1); infra failures score -999 and are masked | abc2midi required | | |

`prompt-mean` (fork's `core_algos.compute_prompt_loss_weights`) weights each token by 1 / (number of prompts in the
batch x action tokens of that prompt's group), so every prompt counts equally whatever its length.

### 1.4 What MiMo publishes about training the small 9B model

| item | value | source |
|---|---|---|
| model | MiMo-V2.6-Distill-Qwen-9B, the policy every recipe names | fork README; `general.env.example` sets `MODEL_PATH=Qwen/Qwen3.5-9B` as its example |
| how it was made | supervised fine-tuning of Qwen3.5-9B on MiMo-generated data | report section 7.1; model card |
| SFT data | 77.4B tokens, 27.2B loss-bearing: code 23.2B, cyber 11.0B, general 22.0B, visual 21.2B | report table 4 |
| RL | GRPO run separately per domain from the same SFT checkpoint; a separate multi-harness coding run on 4 harnesses | report section 7.2 |
| RL hyperparameters | not stated in the report; the fork's recipes (table 1.3) are the only stated values | UNKNOWN whether they equal the runs behind table 6 |
| RL steps actually run | UNKNOWN | |
| GPU hours | UNKNOWN | |
| results (SFT to RL) | SWE-bench Verified 61.1 to 66.2; Terminal Bench 2.1 37.1 to 52.8; MiMo Cyber (mini) 31.3 to 47.0; MiMo Visual Coding (mini) 64.0 to 72.4; internal music 45.7 to 52.5 | report table 6 and section 7.2 |
| measured pathology | on Qwen3.5-9B the `qwen3_coder` tool parser read one batched turn as 47 calls, 44 of them empty; 76.6 percent of tool calls were to tools that do not exist and 89.1 percent of trajectories ended on budget exhaustion | docstring in `recipes/*/agent_loop.py` |

The 9B starts from a distilled checkpoint that already passes many tasks. That is why GRPO works for it. Our model
starts from scratch and passes almost nothing, so we copy the format and the loop and replace the tasks.

## 2. Our ladder spec

### 2.1 The record

Every ladder row has MiMo's six columns with their nested keys.

```
data_source   "sdmonly-ladder/<rung>"            e.g. "sdmonly-ladder/arith"
ability       "<rung>"
agent_name    "sdmonly_single_turn"
prompt        [{"role": "user", "content": "<task text>"}]
reward_model  {"style": "rule", "ground_truth": "<gold as a JSON string>"}
extra_info    {"index": int, "instance_id": "<rung>-L<level>-<split>-s<seed>-<index>",
               "dataset_type": "sdmonly_ladder",
               "instance_json": "{rung, level, split, seed, index, verifier, params}" (a JSON string)}
```

One deliberate difference: our gold sits in `ground_truth`, where verl's reward signature
`compute_score(data_source, solution_str, ground_truth, extra_info)` passes it. MiMo leaves it empty because their gold
lives in containers and env files. MiMo's own rows (rungs 8 to 10) are used unchanged.

The SDM CHAT model sees a row as its training template: BOS, then `User: <content>\nAssistant:`, and the reply stops at
EOS, an added token, or `\nUser:` (as in `track4_sdmonly_generate.py`).

Train and eval never share a task. The eval split adds 1,000,000 to the seed, and every row's `instance_id` names its
rung, level, split, seed and index, so an overlap check is one set intersection.

### 2.2 The shared stop rule

Every gold carries `max_chars`. A reply longer than that scores 0, whatever it contains. Our chat models loop; a
verifier that read only the head of a reply would pay the model for looping.

### 2.3 The rungs

Levels are guesses until calibrated. The pass band is measured, never assumed.

| # | rung | task | rule check (1 or 0) | levels 1 to 5 |
|---|---|---|---|---|
| 1 | copy_line | "Copy this line exactly:" then a line | first line of the reply, trimmed, equals the line, case and spaces included | 2, 4, 8, 16 common words; 5 random letter strings |
| 2 | count | "Continue the count with the next k numbers: 3, 4, 5," | the integers in the reply are exactly the next k, no more | +1 next 1; +1 next 3; start 10 to 90; step 2, 5 or 10; counting down |
| 3 | brackets | "Close every open bracket, innermost first: ( [" | the bracket characters in the reply are exactly the closers, in order | depth 1, 2, 3, 5, 8 over `()[]{}<>` |
| 4 | format | put a given word into a stated format | exact match after trimming; level 4 parses JSON; level 5 compares lines | capitals; square brackets; `ANSWER: <word>`; JSON `{"word": ...}`; numbered list of 3 words |
| 5 | fact_qa | facts "Mia has a red bike." then one question | the gold word appears and no other candidate from the prompt does | 1, 2, 3, 5 facts asking a colour; 5 facts asking who |
| 6 | arith | "What is 7 + 5? Answer with the number only." | exactly one integer in the reply, equal to the answer | a+b under 10; two-digit sums; differences; products to 12x12; a+b-c |
| 7 | code_tiny | a Python signature and doc string | the reply's function plus 5 generated asserts run green in a child python (5 s timeout) | one-line arithmetic; one condition; strings; loops over lists; small algorithms |
| 8a | MiMo Music, format gate | their 496 English rows (`lang` en) first | an ABC tune is found, has K: and M:, M: equals the asked meter, the voice count equals `nvoice_want`, no blank line inside | none (their rows) |
| 8b | MiMo Music, their score | the same rows | their scorer through abc2midi, reward = total / 100 | none |
| 9 | MiMo General | their 989 rows | their rubric, judged by ds4-flash (section 2.5) | none |
| 10 | MiMo Code | their 2,698 rows | their tests in their Docker images | none |

Rung 8a is ours: it checks the asked meter and voice count, which their score ignores, and it gives a non-zero start
long before a reply survives abc2midi. Rungs 8b, 9 and 10 cannot run in this tool. `compute_score` returns None for
rungs 9 and 10, and None means MASKED: the trainer drops the rollout, as MiMo does when a verifier cannot run.

Rung 7 executes model text. The child python runs with `-I`, an empty environment, a temporary working directory, a
5 second timeout and (POSIX) CPU and memory limits. That guards against accidents; it is not a security sandbox. A
real RL run executes rung 7 inside the box's container. The reference solutions live only in the tool (`REF`), compute
the test expectations and serve the selftest, and never enter a row.

### 2.4 The band, calibration and promotion

1. **Calibrate.** Before RL, sample G = 8 replies per task for 256 eval tasks per (rung, level) from the current
   checkpoint, at temperature 1.0 (MiMo's rollout temperature) with the site's sampler. Record the pass rate p and the
   share of mixed groups (groups with at least one pass and one fail). `--verify` prints p and the band label.
2. **Train in band.** A (rung, level) is trainable when 0.05 <= p <= 0.95. Train each rung at its easiest trainable
   level. At p = 0.05 and G = 8, about 34 percent of groups are mixed (1 - 0.95^8); below that GRPO sees almost no
   signal. Drop zero-variance groups, as MiMo does on code, cyber and general.
3. **Promote a level** when the rolling pass rate over the last 1,024 rollouts at that level reaches 0.80, and the
   held-out eval at that level agrees within 0.05. Then the next level becomes the training level.
4. **Demote** when a level's rolling pass rate falls below 0.05 for 1,024 rollouts: go back one level.
5. **Pass a rung** when its level 5 reaches 0.80 on 256 eval tasks. A higher rung opens when its level 1 calibrates at
   p >= 0.05; rungs may train together, mixed in proportion to their mixed-group share.
6. **The top rungs.** Rung 8a opens when it calibrates at p >= 0.05. Rung 8b opens when a quarter of rung 8a passes
   also survive abc2midi with no reject. Rungs 9 and 10 open only when a model passes any of their tasks at all; until
   then code is measured by bits per byte and solution log-probability, as in `RESEARCH_SDMONLY_CANONICAL_REFERENCE`
   section 5.3.

No pass rate here is predicted. The current checkpoint's pass rates are UNMEASURED.

### 2.5 The judge for rung 9

The judge is ds4-flash through the M5 Hermes proxy, the project's prime judge. It is never a frontier judge. MiMo's
`verify.py` already reads its judge from the environment, so no code changes:

```
GA_JUDGE_URL   = http://127.0.0.1:8789/v1        (the proxy; a remote box reaches the M5 over Tailscale)
GA_JUDGE_API   = chat
GA_JUDGE_MODEL = deepseek/deepseek-v4-flash-0731
GA_JUDGE_KEY   = read at run time from ~/.ds4/hermes_proxy_fleet_key (never printed, never committed)
```

ds4-flash is a thinking model; `verify.py` asks for up to 4,000 output tokens, which is enough. Every judged reward
records the serving endpoint and its quant, because one rung id names several providers. A judge that cannot answer
masks the rollout. The judge grades; it never writes text the model trains on.

Rung 9 also needs its environment: the workspace files and the MCP tool servers in `general/envs/<id>/`. A
single-turn SDM model has no tools, so rung 9 waits for a tool-using SDM model.

### 2.6 Our GRPO settings, proposed

| setting | value | why |
|---|---|---|
| loop | nanochat `scripts/chat_rl.py`, ported to our model and our sampler | the canonical reference's choice; every other framework needs vLLM or SGLang |
| group size | 8 | MiMo music and general |
| advantage | reward minus group mean, no division by std | MiMo code, general and cyber, and nanochat (MiMo music and webdev keep verl's division) |
| KL | none | MiMo and nanochat |
| clipping | none while on-policy with one update per batch | nanochat; MiMo's clip 0.2 matters only off-policy |
| loss aggregation | prompt-mean | MiMo |
| zero-variance groups | dropped | MiMo code, cyber, general |
| learning rate | UNKNOWN; sweep | MiMo's 1e-6 to 2e-6 is for AdamW on a 9B; our optimizer and size differ |
| reply budget | 64 tokens for rungs 1 to 6, 256 for rung 7, 2,048 for rung 8 | the tasks' gold lengths |

## 3. The tool

```
python3 track4_sdmonly_rl_ladder.py --list
python3 track4_sdmonly_rl_ladder.py --rung all --level 1 --n 256 --out /tmp/ladder_L1.parquet
python3 track4_sdmonly_rl_ladder.py --verify /tmp/ladder_L1.parquet --replies replies.jsonl
python3 track4_sdmonly_rl_ladder.py --mimo wikis/WIKI_MIMO/data/music.parquet --lang en --show
python3 track4_sdmonly_rl_ladder.py --selftest
```

`--selftest` runs 127 checks in about 7 s on the M5: right and wrong replies for every verifier (42 cases plus 8 music
cases), 8 planted mutants that the cases must catch (case-blind copy, a count that allows extra numbers, order-blind
brackets, a fact check with no distractor rule, an arithmetic check that reads the first integer, a code check that
only compiles, copy without the stop rule, a music check that skips the meter), gold-passes at every rung and level,
shifted gold failing, schema, determinism, train and eval disjoint, masking of MiMo's general and code rows, and a
jsonl and parquet round trip. Planting the same kinds of defect into the real verifiers of arith, brackets, code_tiny,
fact_qa and music turned the selftest red each time.

## 4. Still UNKNOWN

- The RL hyperparameters behind MiMo's table 6 results, beyond the fork's recipe defaults; steps run; GPU hours.
- The pass rates of our checkpoint on every rung.
- Whether a judge as cheap as ds4-flash agrees with MiMo's own judge on their rubric items.

## Sources

Read 2026-10-04 (UTC).

- Dataset: `https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss` (README); datasets-server `/info`, `/rows`,
  `/statistics`, `/parquet`; Hub tree API for `general/envs/s3k_0453_consulting_bizops_analytics_en_t4_rl_002/`
  (`verifier_meta.json`, `verify.py`, `run_verify.py`, `manifest.json`). The five parquets as copied to
  `wikis/WIKI_MIMO/data/`; music parquet sha256 `8fa2c95fb008df032eae184f3304dd3a73e3d25727e3aac0a715484859fa4856`.
- Fork: `https://github.com/XiaomiMiMo/verl` at `a2ad9f6160b03ff2d47e59832bfb6b289f37c917` (also
  `wikis/WIKI_MIMO/code/mimo-verl`): `README.md`; `scripts/code/train.sh`, `env.example`; `scripts/general/general.sh`,
  `general.env.example`; `scripts/design/music.sh`, `music.env.example`, `webdev.sh`; `scripts/arvo/arvo.sh`;
  `recipes/code/config/train.yaml`, `run_train.sh`, `reward.py`; `recipes/general/config/general.yaml`,
  `env_actor.py`; `recipes/design/config/music.yaml`, `webdev.yaml`; `recipes/arvo/config/arvo.yaml`;
  `recipes/design/music/build_parquet.py`, `scorer/__init__.py`, `score.py`, `pipeline.py`;
  `recipes/design/agent_loop.py`; `verl/trainer/ppo/core_algos.py`.
- 9B: `https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (model card); MiMo-V2.6 technical report section
  7 (`wikis/WIKI_MIMO/papers/MiMo_V2_6_technical_report.pdf`, pages 33 to 35).
- This project: `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` section 0; `RESEARCH_SDMONLY_CANONICAL_REFERENCE_2026-10-04.md`
  sections 5.3, 6 and the MiMo addenda; `track4_sdmonly_generate.py` (chat template and stop rules).
