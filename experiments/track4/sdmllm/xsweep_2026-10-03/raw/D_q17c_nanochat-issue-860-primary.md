# D_q17c primary: karpathy/nanochat issue #860 and comments (verbatim)
- url: https://api.github.com/repos/karpathy/nanochat/issues/860 (+ /comments)
- tool: python3 urllib, GitHub REST, unauthenticated
- fetched: 2026-10-03 10:51:34 UTC 

## hgt312 at 2026-09-26T03:33:06Z

`950a1dc` removed `tasks/customjson.py` and `tasks/spellingbee.py`, which also shrinks the SFT mixture in
`scripts/chat_sft.py` from 1,071,759 to 789,759 rows. On 8xH100 that costs most of the model's GSM8K ability.

One shared d24 base checkpoint (`--depth=24 --target-param-data-ratio=8 --fp8`, min val bpb 0.717730), two SFT
runs, identical `chat_eval` settings; the arms differ only in the mixture lines:

| task | master `92d63d4` | + identity/spelling |
|---|---:|---:|
| ARC-Easy | 62.92 | 63.59 |
| ARC-Challenge | 49.74 | 50.77 |
| MMLU | 37.32 | 36.57 |
| GSM8K | **2.35** | **10.31** |
| HumanEval | 12.80 | 9.15 |
| ChatCORE (5 shared tasks) | 0.2303 | 0.2414 |

GSM8K goes 31 → 136 of 1319 problems. Nothing else moves beyond noise. GSM8K training exposure is identical in
both arms (same rows, same 4 epochs), so the 282k short synthetic rows appear to be what teaches the model to
emit a parseable final answer — master's 2.35% looks like a formatting failure, not a reasoning one.
(SpellingBee is excluded: master deletes the task, so only one arm can be scored on it.)

Repro: `git revert 950a1dc`, then the standard `base_train` / `chat_sft` / `chat_eval` sequence above.

Was this eval cost known and accepted? If not, would you prefer restoring the two tasks or replacing the
answer-format signal some other way?


## mira687 at 2026-09-26T06:57:07Z

The "formatting failure, not a reasoning one" reading is testable on the two SFT checkpoints you already have, without a third training run — and the scorer's contract is narrow enough to be worth spelling out first.

`tasks/gsm8k.py` extracts with `GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")` (line 21) and then scores `is_correct = int(pred_num == ref_num)` (line 106) — a **string** comparison after comma-stripping, not a numeric one. So the only thing that counts is a `#### ` marker, space included, followed immediately by a digit string matching the reference character for character.

Running that function verbatim against a reference of `10`:

| completion | extracted | scored |
|---|---|---|
| `she earned $10.` + `\n#### 10` | `10` | 1 |
| `The answer is 10.` | `None` | 0 |
| `#### $10` | `None` | 0 |
| `#### 10.0` | `10.0` | **0** |
| `#### 10.` | `10.` | **0** |
| `####10` | `None` | 0 |
| `\boxed{10}` | `None` | 0 |
| `Let me check: #### 5` … `correcting: #### 10` | `5` | 0 — `search` takes the first marker, same as the official code |
| `**#### 10**` | `10` | 1 |

And on the canonical test split (1,319 problems, `openai/grade-school-math` `test.jsonl`): every reference parses, 14 contain commas, **none contain a decimal point**, and 2 are negative. So a model that answers `10.0` where the reference is `10` is scored wrong by construction, however good the arithmetic was.

That gives a cheap way to split format from reasoning, inference only:

1. count how often `extract_answer(assistant_response) is None` per arm — the "never emitted a marker" rate;
2. re-score the non-`None` ones numerically (`float(pred) == float(ref)`) to catch `10.0`-style near misses;
3. re-score the `None` ones with a fallback extractor — last number in the completion, say.

If master's 2.35 moves toward 10.31 under (3), the 282k short synthetic rows were teaching the answer format, which is your hypothesis with a number on it. If it barely moves, they were teaching something else and the GSM8K delta is real capability.

One practical note: `run_generative_eval` in `scripts/chat_eval.py` (line 28) computes accuracy inline and doesn't keep the generations, so this needs an eval re-run rather than a re-score of saved output — but that is 1,319 prompts at temperature 0, not another SFT.

Happy to do the scoring side if you dump a few hundred master-arm completions somewhere.


