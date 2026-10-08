# CANONREF · one canonical open reference for scaling, data, evaluation and RL · 2026-10-04

A research note for the SDM-only language model (an attention-free model whose context mixing is a Kanerva
memory written and read at run time, trained from scratch on FineWeb-Edu and scored in bits per byte on held-out
FineWeb-Edu text). It answers four questions: which open release to compare our scaling curve against, which
training set keeps that comparison fair, which evaluation suite fits a model of 10M to 300M parameters, and which
open RL programme to follow. It also records the search for a recent open RL release for a small model.

Nothing was trained, rented or committed. Small downloads went to `/private/tmp/canonref/`. Every source was read
on 2026-10-03 (UTC).

**Labels.** MEASURED (elsewhere): the number or statement was read at the primary source named beside it
(repository file, model card, dataset file, paper page). SECONDARY: read only in a news article, digest or search
summary. UNVERIFIED: not found at a primary source, or not opened. ARITHMETIC: computed in this note from the
numbers given, with the inputs shown. SCALED: an extrapolation across a change of budget; it is a guess and must
not enter a plan unlabelled.

## 0. The X search, and the release most likely meant

**X was not reachable.** One call to `GET /2/tweets/search/recent` with the query
`("open RL" OR "fully open" OR RLVR OR GRPO) (8B OR 32B) -is:retweet` returned HTTP 402 `credits depleted`, with
the rate limit at 449 of 450 remaining. That is the credit pool, not a rate limit. The raw response is
`/private/tmp/canonref/x_q01_open_rl.json`. The search then ran on the web: web search, Hugging Face Hub API,
GitHub API and raw repository files. No post on x.com was fetched, so no post's wording or engagement is known.

**The most likely match is K2 Horizon from the Institute of Foundation Models (IFM, MBZUAI), launched on
2026-09-03.** It is the only release in the last two months that is fully open, has both a ~8B-class model
(7B) and a 32B model, and ships an RL codebase for coding. It is a best match, not a confirmed one: the post the
navigator saw was not found.

| candidate | date | sizes | licence | what is open | status of this row |
|---|---|---|---|---|---|
| **K2 Horizon** (IFM / MBZUAI) | HF repos created 2026-09-01; RL repo created 2026-09-03 | 0.9B, 3.7B, **7B**, **32B**, 36B-A4B, 375B-A23B | Apache-2.0 | 7B: weights, every intermediate checkpoint incl. each RL expert, W&B logs, pretraining code, pretraining data; RL code (RL360). Technical report "In Progress" | MEASURED (cards, API, README) |
| Nemotron 3.5 Lightning 30B-A3B (NVIDIA) | HF created 2026-08-01, modified 2026-08-24 | 30B total, 3B active | card says `other` (OpenMDW-1.1 per secondary) | weights; GRPO across seven environment types with NeMo RL and NeMo Gym; a released subset of the RL data | card MEASURED; recipe page SECONDARY |
| Apertus 1.5 (Swiss AI) | HF created 2026-07-24 | 8B, 70B | Apache-2.0 | "fully open" (weights, data, recipes); adds a thinking mode | date MEASURED; RL details UNVERIFIED |
| OpenThinkerAgent-8B-RL (OpenThoughts-Agent) | HF created 2026-06-09 | 8B (Qwen3-8B) | Apache-2.0 | data, code, weights; RLOO on 5,000 RL tasks | date MEASURED; rest SECONDARY |
| Olmo 3 (Ai2) | 2025-11-20 | **7B**, **32B** (Base, Think), 7B Instruct, 7B RL-Zero | open | weights, Dolma 3 data, Dolci post-training data, OLMo-core, open-instruct, OLMES | MEASURED (Ai2 blog); older than two months |
| MiMo-V2.6 RL environments (Xiaomi) | 2026-09-21 | large model | not read | 7,780 RL tasks in Docker containers with verifiers | SECONDARY |

### K2 Horizon, read at its primary sources

MEASURED (HF model cards `IFM/K2-Horizon-7B` and `IFM/K2-Horizon-32B`, raw README; GitHub API; RL360 README;
read 2026-10-03):

- The 7B card's artifact index lists training logs (`wandb.ai/llm360/K2-Horizon-7B`), checkpoints, code
  (`github.com/ifm-ai/xllm`, created 2026-09-02, Apache-2.0), and data (`IFM/TxT360-v2`) as "Available". The
  technical report is "In Progress", expected "End of September 2026". The 32B card was last updated 2026-09-28
  with the same index.
- The 7B was pretrained on 22.9T tokens at 8K, then four midtraining stages (1.1T, 498B, 110B, 199B tokens) up to
  a 512K context. RL then branches into experts: math (2,399 steps, 29.1B tokens), code stage 1 (601 steps,
  6.2B), code stage 2 (1,499 steps, 12.3B), search (59 steps, 8.4B), tool use (39 steps, 1.4B). The experts are
  merged, then SFT runs in two phases. Every RL expert checkpoint is a released branch (`rl_math`, `rl_code1`,
  `rl_code2`, `rl_search`, `rl_tool-use`, `rl_merged`).
- **The 32B has no RL stage on its card.** Its stages are pretraining, four midtraining stages and two SFT phases.
- The 7B has a vocabulary of 250,624, 36 layers and hidden size 4,096 (`config.json`).
- After post-training, the 3.7B and 7B get a short anti-repetition stage, which the card names Final Token
  Preference Optimization (FTPO): a LoRA preference update on pairs mined from the model's own looping outputs.
  This is the same failure our looping-reply measure counts.
- **RL360** (`github.com/ifm-ai/RL360`, the repository first published as `horizon-post-train`; created
  2026-09-03, last push 2026-09-25, Apache-2.0): "Reinforcement Learning codebase used in the post-training
  pipeline for the K2 Horizon series. It trains language models on tasks that involve writing code, using tools,
  and working in a sandbox." Components: Miles (rollouts and training), Megatron-LM (policy update), SGLang
  (inference), Harbor (sandboxes and verifiers). GRPO is the default. The demo overfits the 7B on 32 coding tasks,
  eight attempts each; mean reward rose from 0.373 to 0.901 over 99 rollout iterations (public W&B workspace
  `mbzuai-llm/rl360-public`). The default allocation is 33 nodes of eight H100 or H200 GPUs.
- UNVERIFIED: whether the RL task data for the 7B's experts is released. It is not in the artifact index. The
  launch-post wording and the "@IFM_AI, 2026-09-03" attribution come from a secondary page.

**What it means for us.** K2 Horizon is a frontier-scale RL programme that happens to be fully open. Its RL code
needs a Megatron model, an SGLang server and a multi-node cluster. It is the right thing to read for how coding RL
with sandboxes is organised, and the wrong thing to run on a 100M-parameter attention-free model (section 6).

## 1. The answer on one screen

```
   ┌─ THE CANONICAL REFERENCE ─────────────────────────────────────────────
   │  nanochat (karpathy/nanochat)
   │    unit      bits per byte on FineWeb-Edu  = our unit, our data
   │    scaling   miniseries v1: 11 sizes, 91M-477M params, val_bpb each,
   │              compute-optimal tokens:params = 8 (2026-01-07)
   │    recipe    Muon + AdamW, plain PyTorch, one file per stage
   │    evals     CORE (22 DCLM tasks, centred) + ChatCORE
   │    RL        chat_rl.py: REINFORCE-style "GRPO" on GSM8K, plain PyTorch
   │
   ├─ THE SECOND, for the gap below 91M params ─────────────────────────────
   │  DataDecide (Ai2): FineWeb-Edu ladder 4M .. 1B, 3 seeds,
   │    intermediate checkpoints, OLMES evals. We score them on OUR TEST.
   │
   ├─ DATA ── keep FineWeb-Edu. Both comparators have a FineWeb-Edu arm.
   │
   ├─ EVALS ─ 1 bpb on TEST A (primary)   2 CORE via nanochat core_eval
   │          3 continuous scores (log-prob of the right answer), because
   │            accuracy sits at chance below ~100M params
   │          code: bpb on held-out Python, not pass@1
   │
   ├─ RL ──── follow nanochat chat_rl.py; port it, do not adopt a framework
   │          coding RL at our size has no reward signal: pass rate ~0
   │          read K2 Horizon RL360 and Olmo 3 open-instruct for later
   │
   └─ WHERE WE STAND (arithmetic and one extrapolation, see section 3)
        our largest run: ~4e16 FLOPs. nanochat's smallest: 4.0e17.
        Delphi's smallest: 3e18. We are 10x to 75x below every published ladder.
```

## 2. The canonical reference: candidates side by side

The question was whether one recent, fully open, end-to-end small-LM release publishes all five of: (a) data,
(b) recipe, (c) a family of sizes with losses, (d) an evaluation suite and harness, (e) SFT and RL with code.

| release | data | recipe | sizes with losses, and their unit | evals and harness | SFT and RL | fits us because | does not fit because |
|---|---|---|---|---|---|---|---|
| **nanochat** | FineWeb-Edu until 2026-03-04, then ClimbMix (CC-BY-NC) | full, plain PyTorch, Muon | miniseries v1 d10 to d20 with **val_bpb on FineWeb-Edu**; `scaling_laws.sh` at 1e18 to 1e19 FLOPs | CORE (DCLM), bpb, ChatCORE | SFT; RL on GSM8K (`chat_rl.py`) | same unit, same data (v1), same optimiser family, code small enough to port | sizes start at 91M and 4e17 FLOPs; RL is GSM8K only; no commit after 2026-07-03 |
| modded-nanogpt | FineWeb (not Edu), GPT-2 tokenizer | record-by-record speedrun | one size (124M), target val loss 3.28 nats per token | val loss only | none | the optimiser and trick record | one size, nats per GPT-2 token on FineWeb, no evals |
| Olmo 2 / Olmo 3 (Ai2) | Dolma 3, Dolmino, Longmino, Dolci | full (OLMo-core, open-instruct) | 7B and 32B; ladder papers (UNVERIFIED here) | OLMES, OlmoBaseEval | SFT, DPO, RLVR (math, code, IF, chat), RL-Zero 7B | the most complete open RLVR programme | 7B is the smallest; nothing near our size |
| SmolLM2 / SmolLM3 (HF) | FineWeb-Edu, DCLM, The Stack, smoltalk | published mixtures and playbook | 135M, 360M, 1.7B (SmolLM2); 3B (SmolLM3); no loss ladder found | lighteval | SFT, APO (SmolLM3); no RL found | a 135M model trained on FineWeb-Edu with published zero-shot scores | trained on 2T tokens, so not a point on our curve |
| Pythia (EleutherAI) | The Pile | full | 70M to 12B, many checkpoints (UNVERIFIED here) | lm-eval-harness | none | the classic scaling suite | Pile, not FineWeb-Edu; nats per GPT-NeoX token |
| DCLM baselines | DCLM-Baseline | full | 412M to 7B (abstract) | CORE (defined there) | none | origin of CORE | sizes start at 412M |
| **DataDecide** (Ai2, ICML 2025) | 25 corpora incl. **FineWeb-Edu**, DCLM, Dolma | OLMo, fixed per size | **14 sizes, 4M to 1B**, 3 seeds, intermediate checkpoints; perplexity on 11 sets (nats, GPT-NeoX tokenizer) | OLMES 10 tasks, continuous metrics | none | the only open ladder at our sizes on our data | trained at 100 tokens per param; we must score them ourselves for bpb |
| **Marin Delphi** (2026-05-11) | Nemotron-CC + StarCoderData + ProofPile 2 | full, AdamH, WSD | 88 models, 3e18 to 1e23 FLOPs, 157M to 25B params, losses on W&B | soft metrics to hard metrics | none | a pre-registered scaling law that held 300x past its fit | not FineWeb-Edu; nats per Llama 3 token; eval set not identified in what was read |
| Apertus 1.5 | open | full | 8B, 70B | suite released | thinking mode | fully open | far above our size |
| K2 Horizon | TxT360-v2 | full | 0.9B to 375B | resources released | SFT, RL experts (7B), RL360 | the newest fully open RL code | far above our size |

**RECOMMENDATION: nanochat is the canonical comparison and recipe.** It is the one release where all five parts
exist, are small, and are written in plain PyTorch, and where the published scaling family is scored in our unit
(bits per byte) on our training distribution (FineWeb-Edu, miniseries v1). We already use its optimiser family
(Muon on the body), our repository already carries a full study of it (`wikis/WIKI_NANOCHAT/`, 15 pages), and we
already hold one of its checkpoints. Its RL stage is 326 lines of plain PyTorch (`scripts/chat_rl.py` at master, MEASURED 2026-10-03) that only needs a model that
can sample and return log-probabilities, which ours can.

**SECOND: DataDecide, for one gap only.** nanochat's ladder starts at 91M scaling parameters and 4e17 FLOPs. Our
runs are 10 to 100 times smaller. DataDecide trains on FineWeb-Edu at 4M, 6M, 8M, 10M, 14M, 16M, 20M, 60M, 90M,
150M and 300M parameters, three seeds each, with intermediate checkpoints on the Hub. Scoring those checkpoints on
our TEST A gives a transformer curve on our data, at our sizes, in our unit, with seed spread, at zero training
cost.

## 3. A scaling comparison that is actually fair

### 3.1 Which curves share our axes

Our axis is **bits per byte on held-out FineWeb-Edu text** against **training compute** (FLOPs) or **training
text** (bytes). Bits per byte does not depend on the tokenizer, so curves from different tokenizers can share it.

| source | unit as published | same axes as ours? | how |
|---|---|---|---|
| nanochat miniseries v1 | val_bpb, FineWeb-Edu last shard, context 2,048 | **yes, directly**, with two caveats | different held-out documents; context 2,048 against our 256 (longer context lowers bpb) |
| nanochat after 2026-03-04 | val_bpb on ClimbMix | **no** | different distribution; the repo itself says the numbers are not comparable across the switch |
| DataDecide FineWeb-Edu ladder | perplexity on 11 sets, GPT-NeoX tokenizer; OLMES scores | **yes, after we score the checkpoints on TEST A** | open weights, HF `OLMoForCausalLM`; compute bpb from summed log-prob over TEST A bytes |
| SmolLM2-135M, nanochat-students base-d20, Pythia | various | **one point each, after we score them** | same method; Pythia is a Pile model, so its point says what a Pile model scores on our text |
| modded-nanogpt | val loss 3.28 nats per token | **only with a conversion** | bpb = loss / ln 2 / bytes-per-token. Needs GPT-2's bytes per token on the FineWeb val set, which we have not measured |
| Marin Delphi | loss in nats per Llama 3 token (eval set not identified in what was read) | **no**, shape only | the compute-optimal ratio and the curvature can be compared; the level cannot |
| Pythia, OLMo, DCLM published losses | nats per token on their own sets | **no** | different data and tokenizer; score the checkpoints instead |

**The one conversion everyone needs:** bits per byte = (nats per token / ln 2) × (tokens / bytes) on the same
text. It is only valid when the bytes-per-token ratio is measured on that exact text with that exact tokenizer.

### 3.2 nanochat miniseries v1, extracted

MEASURED (GitHub discussion `karpathy/nanochat#420`, 2026-01-07, raw page fetched; table verbatim). Data
FineWeb-Edu, tokens:params ratio 8, context 2,048. "params_M" is nanochat's `num_scaling_params` (all parameters
except the input embedding, per the same page). FLOPs is ARITHMETIC: 6 × params × tokens.

| depth | width | params | tokens | val_bpb | CORE | FLOPs (6ND) |
|---|---|---|---|---|---|---|
| 10 | 640 | 91M | 0.73B | 1.0312 | 0.071 | 4.0e17 |
| 11 | 704 | 112M | 0.89B | 1.0096 | 0.0918 | 6.0e17 |
| 12 | 768 | 135M | 1.08B | 0.9825 | 0.1059 | 8.7e17 |
| 13 | 832 | 163M | 1.30B | 0.9644 | 0.1015 | 1.3e18 |
| 14 | 896 | 194M | 1.55B | 0.9437 | 0.1185 | 1.8e18 |
| 15 | 960 | 229M | 1.83B | 0.9257 | 0.1158 | 2.5e18 |
| 16 | 1024 | 268M | 2.15B | 0.9101 | 0.1332 | 3.5e18 |
| 17 | 1088 | 313M | 2.50B | 0.8952 | 0.1518 | 4.7e18 |
| 18 | 1152 | 362M | 2.90B | 0.8817 | 0.1611 | 6.3e18 |
| 19 | 1216 | 417M | 3.33B | 0.8687 | 0.1659 | 8.3e18 |
| 20 | 1280 | 477M | 3.82B | 0.8572 | 0.1708 | 1.1e19 |

Also MEASURED there: the compute-optimal tokens:params ratio is about 8, with parameters and tokens each scaling as
C^0.5. Karpathy also writes that GPT-2 and GPT-3 cannot be placed on this val_bpb line, "because they were trained
on an entirely different data distribution". That is the same reason the Pile and Delphi curves are shape-only
for us.

### 3.3 Our own points on the same axes

MEASURED (ours, THE LOG of the plan, 2026-10-03, TEST A, both on Muon, d 256, context 256). Training FLOPs is
ARITHMETIC: 3 × forward FLOPs per token × tokens, with forward FLOPs per token 69.4M (head 66.19M + body 3.21M at
V 129,280, from `flops_per_token` in VOCABSIDE's table; the champion's per-hop SwiGLU adds a little to the body, so
this is a slight undercount). Bytes: VOCABWAVE measured 20M DeepSeek tokens = 95,417,201 bytes of training text.

| run | tokens | training text | FLOPs | TEST A bpb |
|---|---|---|---|---|
| transformer yardstick (`w7_mq_*`) | 20M / 60M / 200M | 95 MB / 286 MB / 954 MB (60M and 200M SCALED from the 20M ratio) | 4.2e15 / 1.2e16 / 4.2e16 | 1.46388 / 1.39966 / 1.37059 |
| SDM champion (`w6_mu_combo`, `w7_ch_*`) | 20M / 60M / 200M | same | same | 1.53229 / 1.47910 / 1.46811 |

nanochat d10, the smallest point in 3.2, used 0.73B tokens (about 3.4 GB of text at 4.6 to 4.9 bytes per token,
SCALED from our measured bytes per token for 32k and 65k tokenizers on FineWeb-Edu) and 4.0e17 FLOPs. **Our
largest run sits about 10 times below the smallest published nanochat point in compute and about 3.5 times below
it in text.** No two points share a budget, so no direct pair exists yet.

**One extrapolation, SCALED and not a measurement.** A pure power law fitted to the eleven miniseries points gives
bpb = 10.04 × C^-0.0562. At 4.2e16 FLOPs it reads 1.17; at 4.2e15 it reads 1.33. That is a factor-of-100
extrapolation below the fitted range, with a context of 2,048 against our 256 and a compute-optimal shape against
our fixed d 256, so it is only a reason to run the real comparison, not a number to plan on. If it held even
roughly, a compute-optimal transformer at our budget would sit 0.2 bpb below our yardstick, which is itself 0.1
below the champion.

### 3.4 Marin Delphi, extracted for its shape

MEASURED (`marin-community/delphi-blog-data`, config `delphi-ladder`, rows of kind `optimum`, downloaded
2026-10-03). Loss is nats per Llama 3 token on a validation set this lane did not identify. Do not convert.

| compute C | optimal params N* | optimal tokens D* | D*/N* (ARITHMETIC) | loss at optimum |
|---|---|---|---|---|
| 3e18 | 381M | 1.48B | 3.9 | 3.586 |
| 9e18 | 558M | 2.90B | 5.2 | 3.386 |
| 1.8e19 | 754M | 4.20B | 5.6 | 3.271 |
| 3e19 | 910M | 5.70B | 6.3 | 3.195 |
| 9e19 | 1.48B | 10.2B | 6.9 | 3.041 |
| 1.8e20 | 1.94B | 15.4B | 8.0 | 2.946 |
| 3e20 | 2.35B | 20.9B | 8.9 | 2.883 |

Held-out runs in the same file: 1e21 FLOPs 2.758, 1e22 2.531, 1e23 (25.0B params, 628B tokens) 2.355. The blog
states the 1e23 loss was predicted within 0.2% from fits at 3e20 and below (MEASURED, openathena.ai/blog/delphi,
read 2026-10-03). The useful lesson for us is method, not level: pre-register the fit before the large run, and fit
on compute-optimal points, not on a fixed width.

### 3.5 What to run to make the comparison real (a proposal, not run)

1. **Score open checkpoints on TEST A**, by document and by window at T 256 and T 2,048: the DataDecide FineWeb-Edu
   ladder (4M to 300M, final checkpoints plus the intermediate ones nearest 20M, 60M and 200M tokens where they
   exist), SmolLM2-135M, and nanochat-students base-d20. This places our points against transformers trained on
   our data, at our sizes, with seed spread, at zero training cost. DataDecide's 20M model uses batch 64 × 2,048,
   so its checkpoints are spaced in steps of 1,250 (about 164M tokens) (ARITHMETIC from its card).
2. **Run our yardstick on nanochat's own shape and ratio** at two budgets (for example 1e17 and 4e17 FLOPs, the
   second equal to nanochat d10) on FineWeb-Edu with a 32k tokenizer at context 2,048, and check the 4e17 run
   lands near 1.03 bpb. That calibrates our trainer against the published point before any SDM claim is made.
3. **Fit SDM and transformer on compute-optimal points**, not on one width, the way nanochat and Delphi do.

## 4. The training set

**Keep FineWeb-Edu as the base for the scaling comparison.**

- **Comparability.** Both comparators have a FineWeb-Edu arm: nanochat miniseries v1 (val_bpb on FineWeb-Edu) and
  the DataDecide FineWeb-Edu ladder. TEST A is FineWeb-Edu. Switching the training data breaks all three links at
  once, which is exactly what happened to nanochat's own numbers on 2026-03-04.
- **Quality.** Better sets exist. nanochat's switch to ClimbMix cut its time to GPT-2 CORE by 27% (MEASURED in
  BASEDATA from nanochat `dev/LOG.md`). LATESTDATA proposes a mix with Ultra-FineWeb L1-HQ and FinePhrase.
- **Licence.** FineWeb-Edu is ODC-By. ClimbMix is CC-BY-NC-4.0 (MEASURED in BASEDATA), which rules it out for
  anything we might ship.
- **The trade, stated plainly.** For the paper and the scaling question, comparability wins: FineWeb-Edu. For the
  final BASE that a visitor chats with, quality may win: that is LATESTDATA's data wave, a separate decision. If a
  mix is adopted, keep one FineWeb-Edu-only arm per scaling point, so the comparison survives the switch.
- **Tokenizer.** Bits per byte makes the tokenizer free on the comparison axis. VOCABWAVE's 32,768 BPE fitted on
  FineWeb-Edu matches nanochat's own default vocabulary size (32,768, `tok_train.py`, from `WIKI_NANOCHAT/01`).

## 5. Evaluations for a model of 10M to 300M parameters

### 5.1 The suite, in order of how much signal it gives at our size

| rank | measure | harness | why at our size |
|---|---|---|---|
| 1 | **bpb on TEST A** (and TEST B when it exists) | ours | smooth, low noise, tokenizer-free. nanochat prefers it to CORE for steering: "has less noise than CORE" (MEASURED, `WIKI_NANOCHAT/04`) |
| 2 | **CORE** (22 DCLM tasks, each accuracy centred so chance = 0 and perfect = 1, then averaged) | `nanochat/core_eval.py` + the downloaded eval bundle | the standard small-model ensemble; needs only log-probs of continuations, so it runs on any model. The task list sits in the bundle, not in the repository (UNVERIFIED list) |
| 3 | **continuous task scores**: the log-probability (or probability) of the correct answer on ARC, HellaSwag, PIQA, MMLU, BoolQ, CSQA, SocialIQA, OBQA, WinoGrande | OLMES style; easiest to add as a mode of our own CORE port | below ~100M params accuracy sits near chance and moves in steps; the log-prob of the right answer still moves smoothly |
| 4 | ChatCORE (chat model): ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval and a spelling task | `nanochat/scripts/chat_eval.py` | for SDM CHAT; expect chance-level accuracy except on formats taught in SFT |
| 5 | our looping-reply measure | `track4_sdmonly_loops.py` | the failure a small chat model shows first |

**MEASURED (DataDecide abstract, arXiv 2504.11393):** "using continuous likelihood metrics as proxies in small
experiments makes benchmarks including MMLU, ARC, HellaSwag, MBPP, and HumanEval >80% predictable at the target 1B
scale with just 0.01% of the compute." That is the reason for rank 3, and for the code rule below.

**Harness choice.** Port nanochat's `core_eval.py`: it is plain PyTorch, it needs only a function that returns
per-token log-probabilities, and it already defines the centring. OLMES and lighteval assume a Hugging Face model
class; wrapping our model for them is possible but buys nothing CORE does not already give. **Context limit:**
our model trains at T 256; CORE's few-shot prompts may be longer. Report the number of truncated prompts beside
every CORE score.

### 5.2 What small models are known to score

| model | params | training tokens | scores | source |
|---|---|---|---|---|
| nanochat d10 | 91M | 0.73B, FineWeb-Edu | CORE 0.071, bpb 1.0312 | MEASURED, `nanochat#420` |
| nanochat d20 (miniseries) | 477M | 3.82B | CORE 0.1708, bpb 0.8572 | MEASURED, same |
| GPT-2 small (calculated by nanochat) | 124M | WebText | CORE 0.114 | MEASURED as quoted in `WIKI_NANOCHAT/04` from #420 |
| GPT-2 XL | 1.5B | WebText | CORE 0.256525 | MEASURED, `WIKI_NANOCHAT/05` |
| SmolLM2-135M | 135M | 2T | HellaSwag 42.1, ARC (avg) 43.9, PIQA 68.4, MMLU cloze 31.5, CSQA 33.9, WinoGrande 51.3, OBQA 34.6, TriviaQA 4.1, GSM8K 5-shot 1.4 (zero-shot, lighteval) | MEASURED, HF card |
| nanochat d32 chat | 1.88B | 37.6B | SFT: GSM8K 0.1274, HumanEval 0.1280, ARC-E 0.6797; after RL: GSM8K 0.1994 | MEASURED, `WIKI_NANOCHAT/04` from discussion #8 |

**What to expect from ours (an ESTIMATE, not a measurement):** a model scoring 1.37 to 1.47 bpb sits well above
nanochat d10's 1.03, so CORE near 0 to 0.05 and multiple-choice accuracy near chance. That is why the continuous
scores and bpb carry the signal at our size.

### 5.3 Code at tiny scale

HumanEval and MBPP pass@1 will be zero or within noise of zero for a 10M to 300M general model: a 1.88B nanochat
reaches 0.128 HumanEval only after SFT, and SmolLM2-135M scores 1.4 on GSM8K after 2T tokens. The meaningful code
measures at our size are:

- **bpb on held-out code** (a fixed Python slice, never trained on), reported beside TEST A;
- **the log-probability of the canonical solution** for each HumanEval and MBPP task (DataDecide's continuous
  metric), which moves long before any sample passes a test.

## 6. RL for coding, at our size

### 6.1 What an open RL recipe needs

1. Prompts with a **verifier**: an exact answer (GSM8K), unit tests (code), or a format check.
2. A **sandbox** that runs model-written code safely (Harbor in RL360, NeMo Gym, the `verifiers` environments of
   Prime Intellect, Docker in each).
3. A **rollout engine** that samples many completions fast (vLLM or SGLang in every framework above).
4. A **policy-gradient trainer**: GRPO and its variants (DAPO, Dr. GRPO, GSPO, CISPO, RLOO).
5. **Logs** of reward, length and pass rate per step.

### 6.2 Which programme to follow

| programme | trainer and stack | what it assumes | for us |
|---|---|---|---|
| **nanochat `scripts/chat_rl.py`** | REINFORCE-like "GRPO" on GSM8K: no KL to a reference, on-policy (no PPO clip), DAPO-style token-level normalisation, advantage r minus mean (no division by sigma) (MEASURED, docstring quoted in `WIKI_NANOCHAT/01`) | a model that samples and returns log-probs, in plain PyTorch | **follow this one**: port it to our model and our generator |
| K2 Horizon RL360 | Miles + Megatron-LM + SGLang + Harbor, GRPO | a Megatron transformer, 33 nodes by default | read for the coding-sandbox design |
| Olmo 3 open-instruct (OlmoRL) | RLVR on math, code, IF, chat; RL-Zero checkpoints | HF transformer, vLLM | the most complete open RLVR programme with data; read for task mix and reward design |
| NeMo RL + NeMo Gym | GRPO, async | Megatron or HF transformer | read |
| TRL `GRPOTrainer`, veRL, OpenRLHF, prime-rl | GRPO family | HF transformer, vLLM | not usable without an inference adapter for our model |

Every framework in the table except nanochat's assumes a transformer that vLLM, SGLang or Megatron can serve. Ours
is neither, and `track4_sdmonly_generate.py` is our own sampler. That alone decides it: port the 326-line loop.

### 6.3 Is RL meaningful for a model of our size?

**Not for coding, and not yet for GSM8K.** GRPO-style training learns only from groups where some samples succeed
and some fail; when every sample fails, every advantage is zero and nothing moves. The published small-model
results all start from a model that already passes sometimes:

- nanochat d32 (1.88B): GSM8K 0.1274 after SFT to 0.1994 after RL. Karpathy calls RL "sucking supervision bits
  through a straw" and says it "creates a GSM-specific model, not a general chat model" (MEASURED, as quoted in
  `WIKI_NANOCHAT/04`).
- TinyZero (countdown task, veRL): "For Qwen2.5-0.5B base, we know it fails to learn reasoning", while 3B and up
  learns it (MEASURED, GitHub README, read 2026-10-03).
- Open-R1: GRPO on Qwen2.5-0.5B base reaches about 51% on GSM8K (MEASURED, HF blog `open-r1/update-2`). That base
  was pretrained on trillions of tokens with math data, so it is not evidence for a 100M model trained on 200M
  tokens of web text.
- A 2026 HF blog reports GRPO on a 350M model (LFM2.5-350M) for structured-output compliance in 100 steps
  (SECONDARY, search summary). Format rewards are the one place a small model has a non-zero start.

**What is meaningful at our size:**

1. **Format and behaviour rewards for SDM CHAT**: ends the turn, answers in the asked format, does not loop. Our
   looping-reply measure is a ready verifier. K2 Horizon solves the same looping failure with a preference stage
   (FTPO), which is cheaper than RL and worth an arm.
2. **Synthetic verifiable tasks chosen so the base pass rate is between about 5% and 95%**: small-number
   arithmetic, copying and reversing, closing brackets, a spelling task like nanochat's. These test the RL
   pipeline end to end and teach a skill the model can almost do.
3. **Coding RL waits** until a model scores above zero on short tests. Until then, measure code by bpb and
   solution log-probability (section 5.3).

## 7. What stayed UNVERIFIED

- The X post the navigator saw. X returned 402; K2 Horizon is a best match, not a confirmed one.
- Whether K2 Horizon's RL task data is released; the launch post's wording; the technical report (not out when
  read).
- Nemotron 3.5 Lightning's recipe details and RL data subset (search summary only; the card was read).
- Apertus 1.5's RL stage; OpenThoughts-Agent's task counts; MiMo-V2.6's environment release.
- The validation set and exact tokenizer used for nanochat miniseries v1 (the repository default today is a
  32,768 vocabulary; the released d20 checkpoint uses 65,536).
- The validation set behind Delphi's loss column.
- The 22-task CORE list (it lives in the downloaded bundle).
- Pythia's and the OLMo ladder's published numbers (not opened in this lane).
- nanowhale-100m's "held-out perplexity 13.62" (DeepSeek tokenizer, FineWeb-Edu, 110M params): the card does not
  name the held-out set, so it is not converted to bpb here.
- Every bytes-per-token figure used for text sizes other than our own measured ones (marked SCALED).

## Sources

All read 2026-10-03 (UTC).

- X API call: `/private/tmp/canonref/x_q01_open_rl.json` (HTTP 402).
- K2 Horizon: https://huggingface.co/IFM/K2-Horizon-7B (raw README, `config.json`) ·
  https://huggingface.co/IFM/K2-Horizon-32B (raw README) · https://github.com/ifm-ai/RL360 (raw README; GitHub
  API for dates and licence) · https://github.com/ifm-ai/xllm (GitHub API) · Hugging Face API model list for
  author `IFM` · secondary: https://www.marktechpost.com/2026/09/06/ifm-releases-k2-horizon-six-apache-2-0-models-from-0-9b-to-375b/
- Nemotron 3.5 Lightning: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (card, API) ·
  secondary: https://docs.nvidia.com/nemotron/latest/nemotron/lightning35/rl.html
- Apertus 1.5: https://huggingface.co/swiss-ai/Apertus-v1.5-8B (API) · OpenThinkerAgent-8B-RL:
  https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-RL (API)
- Olmo 3: https://allenai.org/blog/olmo3
- nanochat: https://github.com/karpathy/nanochat/discussions/420 (raw page) · `wikis/WIKI_NANOCHAT/01`, `04`, `05`
  (this repository, from the nanochat repo at commit `92d63d4`)
- Marin Delphi: https://openathena.ai/blog/delphi/ · https://huggingface.co/datasets/marin-community/delphi-blog-data
  (`delphi-ladder` parquet) · https://huggingface.co/marin-community/delphi-1e22-9.7Bparams-160Btokens-seed62746
  · Hugging Face API model list for `marin-community` (88 Delphi models)
- DataDecide: https://arxiv.org/abs/2504.11393 · https://huggingface.co/datasets/allenai/DataDecide-eval-results
  (README) · https://huggingface.co/datasets/allenai/DataDecide-ppl-results (README) ·
  https://huggingface.co/allenai/DataDecide-fineweb-edu-20M (README, `config.json`, branch list)
- SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-135M · nanowhale: https://huggingface.co/HuggingFaceTB/nanowhale-100m
- DCLM: https://arxiv.org/abs/2406.11794
- TinyZero: https://github.com/Jiayi-Pan/TinyZero · Open-R1: https://huggingface.co/blog/open-r1/update-2
- modded-nanogpt target 3.28: `xsweep_2026-10-03/raw/A_q03_modded_nanogpt_prs.md` (this folder)
- Our numbers: `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` THE LOG; `RESEARCH_SDMONLY_VOCAB_AND_SIDE_BY_SIDE_2026-10-03.md`
  §1.1; VOCABWAVE's block in `SDMONLY_CHANNEL.md`; `RESEARCH_BASEDATA_2026-10-04.md` §1.

## Addendum, 2026-10-04: the X post is most likely Xiaomi MiMo-V2.6

A second search, run on x.com through the web index after the X API returned HTTP 402, found the thread the navigator most likely saw.

- Elie Bakouch (x.com/eliebakouch/status/2102136275708879078): "they will release ~7k RL training data and the framework leading to this top 6 model on AA". Thomas Wolf's release list (x.com/Thom_Wolf/status/2102123398230954053) sits beside it.
- The release is Xiaomi MiMo-V2.6, open-sourced 2026-09-22 under MIT. The family is MiMo-V2.6-Pro-RL (about 1T total, 42B active), MiMo-V2.6-Flash-RL, and MiMo-V2.6-Distill-Qwen-9B.
- The open RL part: more than 7,000 RL environments (software engineering, vulnerability reproduction, knowledge tasks, web development) and an end-to-end framework built on verl (environment interaction, trajectory collection, reward evaluation, policy optimisation). The 9B model is offered as the starting point for community RL runs.
- Reported RL cost: about $2.62M for Pro and $850k for Flash, 30 RL steps over about 750,000 trajectories in under six days. Not independently replicated.
- Sources: eweek.com/news/xiaomi-mimo-v26-open-source-rl-reproduction/, pandaily.com/xiaomi-mimo-v2-6-pro-flash-open-weights-rl-environments.

What this changes for us: nothing about the base recipe. The 9B is distilled, and our own models stay from scratch. The 7,000 environments and the verl framework are the useful part, as a task source for the CHAT stage, filtered to the 5 to 95% pass band at our size. K2 Horizon (RL360) stays the second candidate, and nanochat stays the canonical reference.

### The dataset itself, read 2026-10-04

huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss (Apache 2.0, one train split, parquet, 12.1 GB):

| subset | rows | task | verifier |
|---|---|---|---|
| Code | 2,700 | software engineering in a repo | executable tests |
| Webdev | 2,090 | web development | visual grading |
| Cyber | 1,000 | vulnerability reproduction | rule checks |
| Music | 1,000 | symbolic composition | rule checks |
| General | 989 | knowledge work | rubric judging |
| total | 7,780 | | |

Each row carries `prompt`, `reward_model` and `extra_info` (docker image, problem statement, working directory). Docker images: `xiaomimimo/mimo-v2.6-rl-oss`. Recipes in github.com/XiaomiMiMo/verl: `scripts/code/train.sh`, `scripts/arvo/arvo.sh`, `scripts/general/general.sh`, `scripts/design/webdev.sh`, `scripts/design/music.sh`. The policy model in the recipes is MiMo-V2.6-Distill-Qwen-9B.

Not stated on either page: GPU counts, and which judge model the rubric and visual graders need. Each domain uses its own agent framework, joined in `recipes/`.

What it is: an RL task set for a capable agent, complete as an RL release. What it is not: pretraining data or SFT conversations. At our size every Code, Webdev and Cyber reward would be 0, so GRPO would get no signal. We copy the record format, the verifier types and the GRPO loop, build easy tasks in the same format, and keep the 7,780 tasks as the top of the ladder.
