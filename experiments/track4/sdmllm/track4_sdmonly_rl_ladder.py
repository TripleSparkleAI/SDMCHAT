"""SDMONLY RL ladder: verifiable tasks for RL on the SDM CHAT model, in the record format of Xiaomi's
MiMo-V2.6-RL-oss release, from trivial rungs up to MiMo's own Music, General and Code subsets.

<claudes_code_comments>
** Function List **
WORDS / NAMES / COLOURS / THINGS - small fixed vocabularies the generators draw from
_rng(rung, level, seed, i) - one deterministic random generator per task
_row(rung, level, seed, i, split, prompt, gold, params) - one row in MiMo's six-column layout
gen_copy_line(rng, level) - rung 1: copy a line exactly
gen_count(rng, level) - rung 2: continue a count
gen_brackets(rng, level) - rung 3: close open brackets, innermost first
gen_format(rng, level) - rung 4: put a given word into a stated output format
gen_fact_qa(rng, level) - rung 5: answer a one-fact question from facts in the prompt
gen_arith(rng, level) - rung 6: small-number arithmetic, number only
gen_code_tiny(rng, level) - rung 7: a tiny Python function from its doc string, with executable tests
_too_long(reply, gold) - the shared stop rule: a reply longer than its cap scores 0
verify_copy_line / verify_count / verify_brackets / verify_format / verify_fact_qa / verify_arith - rule checks
extract_code(reply) - the Python source in a reply (fenced block, else from the first "def ")
verify_code_tiny(reply, gold) - runs the reply plus the tests in a child python with a timeout
extract_abc(text) - MiMo's ABC extraction, copied from recipes/design/music/scorer/pipeline.py (Apache-2.0)
verify_music_format(reply, extra_info) - rung 8a: ABC header, meter, voice count and MiMo's blank-line reject
compute_score(data_source, solution_str, ground_truth, extra_info) - verl's reward signature; dispatches by
    data_source; returns a float in [0, 1], or None (MASKED) where the verifier cannot run here
generate(rung, level, n, seed, split) - n rows of one rung and level
render_chat(row) - the SDM CHAT prompt for a row ("User: <content>\nAssistant:")
write_rows(rows, path) / read_rows(path) - jsonl or parquet
schema_check(rows) - every row has MiMo's columns and nested keys
selftest() - verifiers pass gold and fail wrong answers, mutation checks, determinism, schema, round trip
main() - CLI

** Technical Review **
- Record layout. MiMo's code, cyber, general and webdev subsets share six columns: data_source (str), ability (str),
  agent_name (str), prompt (list of {role, content}, exactly one user turn), reward_model {style, ground_truth},
  extra_info {index, instance_id, dataset_type, instance_json (a JSON string)}. This tool writes exactly that layout.
  One deliberate difference: MiMo leaves reward_model.ground_truth empty in all 7,780 rows (their gold sits in
  instance_json or in per-task env files); ours puts the gold in ground_truth as a JSON string, which is where verl's
  reward function signature compute_score(data_source, solution_str, ground_truth, extra_info) expects it.
- Rungs 1 to 7 are ours and generated here. Rung 8 is MiMo's Music subset (their rows, unchanged): this tool checks the
  ABC format (8a); their full score (8b) needs the abc2midi binary and their scorer, not run here. Rungs 9 (General,
  rubric judge) and 10 (Code, executable tests in Docker) are MiMo's rows; compute_score returns None for them, which
  is MASKED, following MiMo's own rule that a verifier that cannot run never reports 0.
- Every rung has 5 levels; the level is the knob that moves a rung's pass rate into the 5 to 95 percent band at the
  current checkpoint. The levels are guesses until calibrated; the spec gives the calibration and promotion rules.
- Shared stop rule: each gold carries max_chars; a longer reply scores 0. The SDM chat models loop, and a verifier that
  ignored the tail would reward looping.
- Train and eval never share a task: split "eval" adds 1,000,000 to the seed, and instance_id carries rung, level,
  split, seed and index.
- code_tiny runs model text in a child python with -I, an empty environment, a temp cwd, a 5 s timeout and (POSIX)
  CPU and memory limits. That is a guard against accidents, not a security sandbox: a real RL run executes it inside
  the box's container. Reference implementations live only in this module (REF), are used to compute test
  expectations and in --selftest, and never enter a row.
- No teacher and no distillation anywhere: every reward is a rule check against a gold computed by code, or (rung 9)
  a rubric judge that grades and never writes text the model is trained on.
Docs: SPEC_SDMONLY_RL_LADDER_2026-10-04.md · RESEARCH_SDMONLY_CANONICAL_REFERENCE_2026-10-04.md (MiMo addenda) ·
wikis/WIKI_MIMO/ (the release) · PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md section 0 (the law)
</claudes_code_comments>
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile

LADDER_DATASET_TYPE = "sdmonly_ladder"
AGENT_NAME = "sdmonly_single_turn"
EVAL_SEED_OFFSET = 1_000_000
LEVELS = (1, 2, 3, 4, 5)

# The six columns of MiMo's code/cyber/general/webdev subsets, and their nested keys (read 2026-10-04).
MIMO_COLUMNS = ("data_source", "ability", "agent_name", "prompt", "reward_model", "extra_info")
MIMO_REWARD_KEYS = ("style", "ground_truth")
MIMO_EXTRA_KEYS = ("index", "instance_id", "dataset_type", "instance_json")

WORDS = ("apple river stone light house green paper music table window bread cloud field horse letter "
         "garden orange silver winter summer candle forest bridge island market planet rocket shadow "
         "teacher thunder valley yellow basket button camera dragon flower hammer jacket kitten ladder "
         "mirror needle pencil rabbit spider tomato violin wallet zebra anchor bottle carpet doctor engine "
         "feather guitar helmet jungle kettle lemon magnet napkin ocean pillow quilt ribbon saddle tunnel").split()
NAMES = "Mia Tom Ana Ben Lucy Omar Ivy Sam Nora Leo Zoe Max Ella Finn Ruth".split()
COLOURS = "red blue green yellow black white orange purple brown pink".split()
THINGS = "bike hat car cup kite coat ball boat bag lamp".split()

RUNGS = {
    1: "copy_line", 2: "count", 3: "brackets", 4: "format", 5: "fact_qa", 6: "arith", 7: "code_tiny",
    8: "mimo_music", 9: "mimo_general", 10: "mimo_code",
}
GENERATED = ("copy_line", "count", "brackets", "format", "fact_qa", "arith", "code_tiny")


def _rng(rung, level, seed, i):
    return random.Random(f"{rung}|{level}|{seed}|{i}")


def _row(rung, level, seed, i, split, prompt, gold, params):
    instance_id = f"{rung}-L{level}-{split}-s{seed}-{i:06d}"
    inst = {"rung": rung, "level": level, "split": split, "seed": seed, "index": i,
            "verifier": f"verify_{rung}", "params": params}
    return {
        "data_source": f"sdmonly-ladder/{rung}",
        "ability": rung,
        "agent_name": AGENT_NAME,
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gold, ensure_ascii=False, sort_keys=True)},
        "extra_info": {"index": i, "instance_id": instance_id, "dataset_type": LADDER_DATASET_TYPE,
                       "instance_json": json.dumps(inst, ensure_ascii=False, sort_keys=True)},
    }


# ---------------------------------------------------------------------------------------------------- generators

def gen_copy_line(rng, level):
    if level <= 4:
        n = {1: 2, 2: 4, 3: 8, 4: 16}[level]
        line = " ".join(rng.choice(WORDS) for _ in range(n))
    else:
        line = " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 7)))
                        for _ in range(5))
    prompt = f"Copy this line exactly:\n{line}"
    return prompt, {"expected": line, "max_chars": 2 * len(line) + 20}, {"words": len(line.split())}


def gen_count(rng, level):
    if level == 1:
        start, step, k = rng.randint(1, 9), 1, 1
    elif level == 2:
        start, step, k = rng.randint(1, 9), 1, 3
    elif level == 3:
        start, step, k = rng.randint(10, 90), 1, 3
    elif level == 4:
        start, step, k = rng.randint(1, 20), rng.choice((2, 5, 10)), 3
    else:
        start, step, k = rng.randint(30, 99), -rng.randint(1, 3), 3
    shown = [start + step * j for j in range(3)]
    expected = [start + step * (3 + j) for j in range(k)]
    word = "number" if k == 1 else f"{k} numbers"
    prompt = f"Continue the count with the next {word}: " + ", ".join(map(str, shown)) + ","
    return prompt, {"expected": expected, "max_chars": 12 * k + 20}, {"start": start, "step": step, "k": k}


OPEN, CLOSE = "([{<", ")]}>"


def gen_brackets(rng, level):
    depth = {1: 1, 2: 2, 3: 3, 4: 5, 5: 8}[level]
    opens = [rng.choice(OPEN) for _ in range(depth)]
    expected = " ".join(CLOSE[OPEN.index(c)] for c in reversed(opens))
    prompt = "Close every open bracket, innermost first: " + " ".join(opens)
    return prompt, {"expected": expected, "max_chars": 4 * depth + 20}, {"depth": depth}


def gen_format(rng, level):
    w = rng.choice(WORDS)
    if level == 1:
        prompt, exp = f'Write the word "{w}" in capital letters and nothing else.', w.upper()
    elif level == 2:
        prompt, exp = f'Write the word "{w}" inside square brackets and nothing else.', f"[{w}]"
    elif level == 3:
        prompt, exp = f'Reply in the form ANSWER: <word>, using the word "{w}".', f"ANSWER: {w}"
    elif level == 4:
        prompt, exp = f'Reply with JSON only, of the form {{"word": "..."}}, using the word "{w}".', {"word": w}
    else:
        ws = rng.sample(WORDS, 3)
        prompt = "Write these words as a numbered list, one per line: " + ", ".join(ws)
        exp = "\n".join(f"{j + 1}. {x}" for j, x in enumerate(ws))
    max_chars = 3 * len(json.dumps(exp)) + 20
    return prompt, {"template": f"L{level}", "expected": exp, "max_chars": max_chars}, {"word": w}


def gen_fact_qa(rng, level):
    n = {1: 1, 2: 2, 3: 3, 4: 5, 5: 5}[level]
    people = rng.sample(NAMES, n)
    colours = rng.sample(COLOURS, n)
    things = [rng.choice(THINGS) for _ in range(n)]
    facts = [f"{p} has a {c} {t}." for p, c, t in zip(people, colours, things)]
    j = rng.randrange(n)
    if level < 5:
        q = f"What colour is {people[j]}'s {things[j]}? Answer with one word."
        gold, cands = colours[j], colours
    else:
        q = f"Who has the {colours[j]} {things[j]}? Answer with one name."
        gold, cands = people[j], people
    prompt = " ".join(facts) + "\n" + q
    distractors = [c for c in cands if c != gold]
    return prompt, {"expected": gold, "distractors": distractors, "max_chars": 60}, {"facts": n}


def gen_arith(rng, level):
    if level == 1:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        expr, val = f"{a} + {b}", a + b
    elif level == 2:
        a, b = rng.randint(10, 49), rng.randint(10, 49)
        expr, val = f"{a} + {b}", a + b
    elif level == 3:
        a = rng.randint(10, 99)
        b = rng.randint(0, a)
        expr, val = f"{a} - {b}", a - b
    elif level == 4:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        expr, val = f"{a} * {b}", a * b
    else:
        a, b = rng.randint(10, 60), rng.randint(10, 39)
        c = rng.randint(0, a + b)
        expr, val = f"{a} + {b} - {c}", a + b - c
    prompt = f"What is {expr}? Answer with the number only."
    return prompt, {"expected": val, "max_chars": 24}, {"expr": expr}


# Rung 7. REF holds the reference implementations; they compute test expectations and serve --selftest. They are
# never written into a row (no solution text reaches the model by any path).
CODE_TASKS = {
    1: [("add_one", "x", "Return x plus 1.", "int"), ("double", "x", "Return x times 2.", "int"),
        ("negate", "x", "Return minus x.", "int"), ("square", "x", "Return x times x.", "int")],
    2: [("is_even", "n", "Return True if n is even, else False.", "int"),
        ("abs_value", "x", "Return the absolute value of x.", "int"),
        ("max2", "a, b", "Return the larger of a and b.", "int2")],
    3: [("reverse", "s", "Return the string s reversed.", "str"),
        ("first_char", "s", "Return the first character of the non-empty string s.", "str"),
        ("shout", "s", "Return s in upper case followed by an exclamation mark.", "str")],
    4: [("sum_list", "xs", "Return the sum of the numbers in the list xs.", "list"),
        ("count_char", "s, c", "Return how many times the character c occurs in the string s.", "strchar"),
        ("max_list", "xs", "Return the largest number in the non-empty list xs.", "list")],
    5: [("factorial", "n", "Return n factorial for n >= 0 (factorial of 0 is 1).", "small"),
        ("is_palindrome", "s", "Return True if s reads the same forwards and backwards, else False.", "pal"),
        ("fizzbuzz_word", "n", 'Return "FizzBuzz" if n is divisible by 15, "Fizz" if by 3, "Buzz" if by 5, '
                                'else the number as a string.', "int")],
}
REF = {
    "add_one": "def add_one(x):\n    return x + 1\n",
    "double": "def double(x):\n    return x * 2\n",
    "negate": "def negate(x):\n    return -x\n",
    "square": "def square(x):\n    return x * x\n",
    "is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "abs_value": "def abs_value(x):\n    return x if x >= 0 else -x\n",
    "max2": "def max2(a, b):\n    return a if a >= b else b\n",
    "reverse": "def reverse(s):\n    return s[::-1]\n",
    "first_char": "def first_char(s):\n    return s[0]\n",
    "shout": "def shout(s):\n    return s.upper() + '!'\n",
    "sum_list": "def sum_list(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n",
    "count_char": "def count_char(s, c):\n    return sum(1 for ch in s if ch == c)\n",
    "max_list": "def max_list(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m\n",
    "factorial": "def factorial(n):\n    r = 1\n    for k in range(2, n + 1):\n        r *= k\n    return r\n",
    "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]\n",
    "fizzbuzz_word": ("def fizzbuzz_word(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n"
                      "        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n"),
}


def _ref_fn(name):
    ns = {}
    exec(REF[name], ns)  # module-owned reference text, never model output
    return ns[name]


def _code_args(rng, kind):
    if kind == "int":
        return (rng.randint(-50, 50),)
    if kind == "int2":
        return (rng.randint(-50, 50), rng.randint(-50, 50))
    if kind == "str":
        return ("".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 8))),)
    if kind == "list":
        return ([rng.randint(-20, 20) for _ in range(rng.randint(1, 6))],)
    if kind == "strchar":
        s = "".join(rng.choice("abcab") for _ in range(rng.randint(1, 10)))
        return (s, rng.choice("abc"))
    if kind == "small":
        return (rng.randint(0, 8),)
    if kind == "pal":
        half = "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
        return (half + half[::-1],) if rng.random() < 0.5 else (half + "x" + half,)
    raise ValueError(kind)


def gen_code_tiny(rng, level):
    name, args, doc, kind = rng.choice(CODE_TASKS[level])
    fn = _ref_fn(name)
    tests = []
    for _ in range(5):
        a = _code_args(rng, kind)
        tests.append(f"assert {name}({', '.join(repr(x) for x in a)}) == {fn(*a)!r}")
    prompt = (f'Write this Python function.\n\ndef {name}({args}):\n    """{doc}"""\n\n'
              "Reply with the complete function and nothing else.")
    return prompt, {"entry": name, "tests": tests, "max_chars": 1200}, {"name": name}


GENERATORS = {"copy_line": gen_copy_line, "count": gen_count, "brackets": gen_brackets, "format": gen_format,
              "fact_qa": gen_fact_qa, "arith": gen_arith, "code_tiny": gen_code_tiny}


# ----------------------------------------------------------------------------------------------------- verifiers

def _too_long(reply, gold):
    return len(reply) > int(gold.get("max_chars", 10_000))


def verify_copy_line(reply, gold):
    if _too_long(reply, gold):
        return 0.0
    lines = [ln for ln in reply.strip().split("\n")]
    return 1.0 if lines and lines[0].strip() == gold["expected"] else 0.0


_INT_RE = re.compile(r"-?\d+")


def verify_count(reply, gold):
    if _too_long(reply, gold):
        return 0.0
    got = [int(x) for x in _INT_RE.findall(reply)]
    exp = list(gold["expected"])
    return 1.0 if got == exp else 0.0


def verify_brackets(reply, gold):
    if _too_long(reply, gold):
        return 0.0
    got = [ch for ch in reply if ch in OPEN + CLOSE]
    exp = [ch for ch in gold["expected"] if ch in CLOSE]
    return 1.0 if got == exp else 0.0


def verify_format(reply, gold):
    if _too_long(reply, gold):
        return 0.0
    exp = gold["expected"]
    text = reply.strip()
    if gold["template"] == "L4":
        try:
            return 1.0 if json.loads(text) == exp else 0.0
        except (ValueError, TypeError):
            return 0.0
    if gold["template"] == "L5":
        got = "\n".join(ln.strip() for ln in text.split("\n") if ln.strip())
        return 1.0 if got == exp else 0.0
    return 1.0 if text == exp else 0.0


def _words(text):
    return re.findall(r"[A-Za-z]+", text.lower())


def verify_fact_qa(reply, gold):
    if _too_long(reply, gold):
        return 0.0
    ws = set(_words(reply))
    if gold["expected"].lower() not in ws:
        return 0.0
    if any(d.lower() in ws for d in gold["distractors"]):
        return 0.0
    return 1.0


def verify_arith(reply, gold):
    if _too_long(reply, gold):
        return 0.0
    got = _INT_RE.findall(reply)
    return 1.0 if len(got) == 1 and int(got[0]) == int(gold["expected"]) else 0.0


_FENCE_PY = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(reply):
    m = _FENCE_PY.search(reply)
    if m:
        return m.group(1)
    k = reply.find("def ")
    return reply[k:] if k >= 0 else ""


def _limits():  # runs in the child before exec (POSIX only)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 2 ** 20, 512 * 2 ** 20))
    except (ImportError, ValueError, OSError):
        pass


def verify_code_tiny(reply, gold, timeout=5.0):
    if _too_long(reply, gold):
        return 0.0
    code = extract_code(reply)
    if f"def {gold['entry']}" not in code:
        return 0.0
    src = code + "\n\n" + "\n".join(gold["tests"]) + "\nprint('LADDER_TESTS_OK')\n"
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", src], cwd=td, env={}, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if p.returncode == 0 and p.stdout.strip().endswith("LADDER_TESTS_OK") else 0.0


# MiMo's ABC extraction, copied verbatim from recipes/design/music/scorer/pipeline.py (XiaomiMiMo/verl a2ad9f6,
# Apache-2.0, Copyright 2026 Bytedance Ltd. and/or its affiliates).
_FENCE_RE = re.compile(r"```(?:abc|ABC)?\s*\n(.*?)```", re.DOTALL)
_XA_RE = re.compile(r"X\s*:\s*\d+")


def extract_abc(text):
    if not text:
        return None
    for m in _FENCE_RE.finditer(text):
        body = m.group(1).strip()
        if _XA_RE.search(body):
            return body
    matches = list(_XA_RE.finditer(text))
    if matches:
        return text[matches[-1].start():].strip()
    return None


def verify_music_format(reply, extra_info):
    """Rung 8a: the format gate in front of MiMo's music score. 1 if the reply holds an ABC tune with a K: line,
    the asked meter, the asked number of voices, and no blank line inside (MiMo's own 'blank' reject)."""
    abc = extract_abc(reply)
    if not abc:
        return 0.0
    lines = abc.split("\n")
    if any(not ln.strip() for ln in lines[:-1]):  # pipeline.py: blank = any blank line except the last
        return 0.0
    head = {}
    voices = set()
    for ln in lines:
        m = re.match(r"^([A-Za-z]):\s*(.*)$", ln.strip())
        if m:
            head.setdefault(m.group(1), m.group(2).strip())
            if m.group(1) == "V":
                voices.add(m.group(2).split()[0] if m.group(2).split() else "")
        for vm in re.finditer(r"\[V:\s*([^\]\s]+)", ln):
            voices.add(vm.group(1))
    if "K" not in head or "M" not in head:
        return 0.0
    if head["M"].replace(" ", "") != str(extra_info.get("meter", "")).replace(" ", ""):
        return 0.0
    want = int(extra_info.get("nvoice_want") or 1)
    n = max(1, len(voices))
    return 1.0 if n == want else 0.0


MASKED_SOURCES = ("mimoagent/", "opensource-code", "arvo", "blackbox/webdev")


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    """verl's custom reward signature. Returns a float in [0, 1], or None when the verifier cannot run here (MASKED:
    the trainer must drop the rollout, never score it 0)."""
    extra_info = extra_info or {}
    if data_source.startswith("sdmonly-ladder/"):
        rung = data_source.split("/", 1)[1]
        gold = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        return VERIFIERS[rung](solution_str or "", gold)
    if data_source == "music":
        return verify_music_format(solution_str or "", extra_info)
    if data_source.startswith(MASKED_SOURCES):
        return None
    raise ValueError(f"no verifier for data_source {data_source!r}")


VERIFIERS = {"copy_line": verify_copy_line, "count": verify_count, "brackets": verify_brackets,
             "format": verify_format, "fact_qa": verify_fact_qa, "arith": verify_arith,
             "code_tiny": verify_code_tiny}


# ----------------------------------------------------------------------------------------------------------- I/O

def generate(rung, level, n, seed=0, split="train"):
    if rung not in GENERATORS:
        raise ValueError(f"rung {rung!r} is not generated here; MiMo's rows are read with --mimo")
    eff = seed + (EVAL_SEED_OFFSET if split == "eval" else 0)
    rows = []
    for i in range(n):
        prompt, gold, params = GENERATORS[rung](_rng(rung, level, eff, i), level)
        rows.append(_row(rung, level, eff, i, split, prompt, gold, params))
    return rows


def render_chat(row):
    """The SDM CHAT prompt: the chat shards' training template, one user turn, then the cue. BOS is the caller's."""
    return f"User: {row['prompt'][0]['content']}\nAssistant:"


def write_rows(rows, path):
    if path.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_rows(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def schema_check(rows):
    errs = []
    for k, r in enumerate(rows):
        if tuple(r.keys()) != MIMO_COLUMNS and set(r.keys()) != set(MIMO_COLUMNS):
            errs.append(f"row {k}: columns {sorted(r.keys())}")
        if set(r["reward_model"].keys()) != set(MIMO_REWARD_KEYS):
            errs.append(f"row {k}: reward_model keys")
        if set(r["extra_info"].keys()) != set(MIMO_EXTRA_KEYS):
            errs.append(f"row {k}: extra_info keys")
        if not (isinstance(r["prompt"], list) and len(r["prompt"]) == 1 and r["prompt"][0]["role"] == "user"):
            errs.append(f"row {k}: prompt is not one user turn")
        if not isinstance(r["extra_info"]["instance_json"], str):
            errs.append(f"row {k}: instance_json is not a string")
    return errs


# ------------------------------------------------------------------------------------------------------ selftest

def _cases():
    """(rung, gold, reply, expected score). Each rung has right and wrong replies, including the wrongs the mutants
    below would let through."""
    g = lambda **kw: kw  # noqa: E731
    return [
        ("copy_line", g(expected="Hello world", max_chars=42), "Hello world", 1.0),
        ("copy_line", g(expected="Hello world", max_chars=42), "  Hello world\n", 1.0),
        ("copy_line", g(expected="Hello world", max_chars=42), "hello world", 0.0),
        ("copy_line", g(expected="Hello world", max_chars=42), "Hello", 0.0),
        ("copy_line", g(expected="Hello world", max_chars=42), "Hello world\n" * 9, 0.0),
        ("count", g(expected=[6, 7, 8], max_chars=56), " 6, 7, 8", 1.0),
        ("count", g(expected=[6, 7, 8], max_chars=56), "6, 7, 9", 0.0),
        ("count", g(expected=[6, 7, 8], max_chars=56), "6, 7, 8, 9, 10", 0.0),
        ("count", g(expected=[97, 95, 93], max_chars=56), "97, 95, 93", 1.0),
        ("count", g(expected=[97, 95, 93], max_chars=56), "97, 95, -93", 0.0),
        ("brackets", g(expected="] )", max_chars=28), " ] )", 1.0),
        ("brackets", g(expected="] )", max_chars=28), "])", 1.0),
        ("brackets", g(expected="] )", max_chars=28), ") ]", 0.0),
        ("brackets", g(expected="] )", max_chars=28), "] ) )", 0.0),
        ("brackets", g(expected="] )", max_chars=28), "( ] )", 0.0),
        ("format", g(template="L1", expected="APPLE", max_chars=41), "APPLE", 1.0),
        ("format", g(template="L1", expected="APPLE", max_chars=41), "apple", 0.0),
        ("format", g(template="L2", expected="[apple]", max_chars=47), "[apple]", 1.0),
        ("format", g(template="L3", expected="ANSWER: apple", max_chars=65), "ANSWER: apple", 1.0),
        ("format", g(template="L3", expected="ANSWER: apple", max_chars=65), "apple", 0.0),
        ("format", g(template="L4", expected={"word": "apple"}, max_chars=80), '{"word": "apple"}', 1.0),
        ("format", g(template="L4", expected={"word": "apple"}, max_chars=80), '{"word": "pear"}', 0.0),
        ("format", g(template="L4", expected={"word": "apple"}, max_chars=80), "word: apple", 0.0),
        ("format", g(template="L5", expected="1. a\n2. b\n3. c", max_chars=80), "1. a\n2. b\n3. c", 1.0),
        ("format", g(template="L5", expected="1. a\n2. b\n3. c", max_chars=80), "1. a\n2. c\n3. b", 0.0),
        ("fact_qa", g(expected="red", distractors=["blue", "green"], max_chars=60), "Red.", 1.0),
        ("fact_qa", g(expected="red", distractors=["blue", "green"], max_chars=60), "It is red", 1.0),
        ("fact_qa", g(expected="red", distractors=["blue", "green"], max_chars=60), "blue", 0.0),
        ("fact_qa", g(expected="red", distractors=["blue", "green"], max_chars=60), "red blue green", 0.0),
        ("fact_qa", g(expected="red", distractors=["blue", "green"], max_chars=60), "redish", 0.0),
        ("arith", g(expected=12, max_chars=24), "12", 1.0),
        ("arith", g(expected=12, max_chars=24), " 12.", 1.0),
        ("arith", g(expected=12, max_chars=24), "13", 0.0),
        ("arith", g(expected=12, max_chars=24), "12 12 12", 0.0),
        ("arith", g(expected=-3, max_chars=24), "-3", 1.0),
        ("arith", g(expected=12, max_chars=24), "7 + 5 = 12", 0.0),
        ("code_tiny", g(entry="add_one", tests=["assert add_one(1) == 2", "assert add_one(-1) == 0"],
                        max_chars=1200), "def add_one(x):\n    return x + 1\n", 1.0),
        ("code_tiny", g(entry="add_one", tests=["assert add_one(1) == 2", "assert add_one(-1) == 0"],
                        max_chars=1200), "```python\ndef add_one(x):\n    return x + 1\n```", 1.0),
        ("code_tiny", g(entry="add_one", tests=["assert add_one(1) == 2", "assert add_one(-1) == 0"],
                        max_chars=1200), "def add_one(x):\n    return x + 2\n", 0.0),
        ("code_tiny", g(entry="add_one", tests=["assert add_one(1) == 2"], max_chars=1200),
         "def add_one(x):\n    while True:\n        pass\n", 0.0),
        ("code_tiny", g(entry="add_one", tests=["assert add_one(1) == 2"], max_chars=1200),
         "def add_one(x)\n    return x + 1\n", 0.0),
        ("code_tiny", g(entry="add_one", tests=["assert add_one(1) == 2"], max_chars=1200), "x + 1", 0.0),
    ]


ABC_OK = "X:1\nT:t\nM:4/4\nL:1/8\nK:C\nV:1\nCDEF GABc|\nV:2\nC,D,E,F, G,A,B,C|"
MUSIC_CASES = [
    ({"meter": "4/4", "nvoice_want": 2}, ABC_OK, 1.0),
    ({"meter": "4/4", "nvoice_want": 2}, "Here it is.\n```abc\n" + ABC_OK + "\n```\n", 1.0),
    ({"meter": "3/4", "nvoice_want": 2}, ABC_OK, 0.0),
    ({"meter": "4/4", "nvoice_want": 3}, ABC_OK, 0.0),
    ({"meter": "4/4", "nvoice_want": 2}, ABC_OK.replace("V:2\n", "\nV:2\n"), 0.0),
    ({"meter": "4/4", "nvoice_want": 2}, ABC_OK.replace("X:1\n", ""), 0.0),
    ({"meter": "4/4", "nvoice_want": 2}, ABC_OK.replace("K:C\n", ""), 0.0),
    ({"meter": "4/4", "nvoice_want": 1}, "X:1\nM:4/4\nL:1/8\nK:C\nCDEF GABc|", 1.0),
]


def _mutants():
    """Planted defects. Each must be caught by at least one case (a case whose score changes)."""
    def m_copy(reply, gold):  # case-insensitive compare
        return 1.0 if reply.strip().split("\n")[0].strip().lower() == gold["expected"].lower() else 0.0

    def m_count(reply, gold):  # drops the "no extra numbers" rule
        got = [int(x) for x in _INT_RE.findall(reply)]
        return 1.0 if got[:len(gold["expected"])] == list(gold["expected"]) else 0.0

    def m_brackets(reply, gold):  # order-blind
        got = sorted(ch for ch in reply if ch in OPEN + CLOSE)
        return 1.0 if got == sorted(ch for ch in gold["expected"] if ch in CLOSE) else 0.0

    def m_fact(reply, gold):  # no distractor check
        return 1.0 if gold["expected"].lower() in set(_words(reply)) else 0.0

    def m_arith(reply, gold):  # first integer only
        got = _INT_RE.findall(reply)
        return 1.0 if got and int(got[0]) == int(gold["expected"]) else 0.0

    def m_code(reply, gold):  # compiles, tests never run
        try:
            compile(extract_code(reply), "<reply>", "exec")
            return 1.0 if f"def {gold['entry']}" in extract_code(reply) else 0.0
        except SyntaxError:
            return 0.0

    def m_stop(reply, gold):  # copy_line without the shared stop rule
        lines = reply.strip().split("\n")
        return 1.0 if lines and lines[0].strip() == gold["expected"] else 0.0

    return {"copy_line": [m_copy, m_stop], "count": [m_count], "brackets": [m_brackets], "fact_qa": [m_fact],
            "arith": [m_arith], "code_tiny": [m_code]}


def _music_mutant(reply, extra_info):  # skips the meter check
    return verify_music_format(reply, {**extra_info, "meter": (extract_abc(reply) or "M:?").split("M:")[-1]
                                       .split("\n")[0].strip()})


def selftest(verbose=False):
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        if verbose or not ok:
            print(("PASS " if ok else "FAIL ") + name)

    cases = _cases()
    for k, (rung, gold, reply, want) in enumerate(cases):
        got = VERIFIERS[rung](reply, gold)
        check(f"case {k:02d} {rung} {reply[:24]!r} -> {want}", got == want)
    for k, (ei, reply, want) in enumerate(MUSIC_CASES):
        check(f"music case {k} -> {want}", verify_music_format(reply, ei) == want)

    for rung, ms in _mutants().items():
        for m in ms:
            caught = any(m(reply, gold) != want for (r, gold, reply, want) in cases if r == rung)
            check(f"mutant {rung}.{m.__name__} caught", caught)
    caught = any(_music_mutant(reply, ei) != want for ei, reply, want in MUSIC_CASES)
    check("mutant music.skip_meter caught", caught)

    # Every generated task verifies its own gold as 1, at every level; a gold from another task mostly fails.
    for rung in GENERATED:
        n = 6 if rung == "code_tiny" else 40
        for level in LEVELS:
            rows = generate(rung, level, n, seed=7)
            ok = True
            for r in rows:
                gold = json.loads(r["reward_model"]["ground_truth"])
                if rung == "code_tiny":
                    reply = REF[gold["entry"]]
                elif rung == "format" and gold["template"] == "L4":
                    reply = json.dumps(gold["expected"])
                elif rung == "count":
                    reply = ", ".join(map(str, gold["expected"]))
                else:
                    reply = str(gold["expected"])
                ok &= compute_score(r["data_source"], reply, r["reward_model"]["ground_truth"],
                                    r["extra_info"]) == 1.0
            check(f"gold passes {rung} L{level} n={n}", ok)
        rows = generate(rung, 3, 40, seed=11)
        golds = [json.loads(r["reward_model"]["ground_truth"]) for r in rows]
        if rung != "code_tiny":
            wrong = sum(VERIFIERS[rung](str(golds[(i + 1) % 40]["expected"]) if rung != "count" else
                                        ", ".join(map(str, golds[(i + 1) % 40]["expected"])), golds[i]) == 0.0
                        for i in range(40))
            check(f"shifted gold fails {rung} L3 ({wrong}/40 >= 30)", wrong >= 30)
        check(f"schema {rung}", not schema_check(rows))
        check(f"deterministic {rung}", generate(rung, 3, 5, seed=11) == rows[:5])
        ev = generate(rung, 3, 5, seed=11, split="eval")
        check(f"eval disjoint {rung}", not {r["extra_info"]["instance_id"] for r in ev}
              & {r["extra_info"]["instance_id"] for r in rows})

    check("masked general row -> None", compute_score("mimoagent/general_agent", "anything", "", {}) is None)
    check("masked code row -> None", compute_score("opensource-code", "anything", "", {}) is None)
    check("music row dispatch", compute_score("music", ABC_OK, "", {"meter": "4/4", "nvoice_want": 2}) == 1.0)
    try:
        compute_score("unknown/source", "x", "", {})
        check("unknown source raises", False)
    except ValueError:
        check("unknown source raises", True)
    row = generate("arith", 1, 1, seed=0)[0]
    check("render_chat template", render_chat(row).startswith("User: What is ") and render_chat(row)
          .endswith("\nAssistant:"))

    with tempfile.TemporaryDirectory() as td:
        rows = generate("fact_qa", 2, 8, seed=3)
        for ext in (".jsonl", ".parquet"):
            p = os.path.join(td, "r" + ext)
            try:
                write_rows(rows, p)
                check(f"round trip {ext}", read_rows(p) == rows)
            except ImportError:
                print(f"SKIP round trip {ext} (pyarrow missing)")

    n_ok = sum(ok for _, ok in checks)
    print(f"selftest: {n_ok}/{len(checks)} passed")
    return n_ok == len(checks)


# ---------------------------------------------------------------------------------------------------------- CLI

HELP_EPILOG = """examples:
  python3 track4_sdmonly_rl_ladder.py --list
  python3 track4_sdmonly_rl_ladder.py --rung arith --level 2 --n 3 --show
  python3 track4_sdmonly_rl_ladder.py --rung all --level 1 --n 256 --out /tmp/ladder_L1.parquet
  python3 track4_sdmonly_rl_ladder.py --rung code_tiny --level 1 --n 64 --split eval --out /tmp/code_eval.jsonl
  python3 track4_sdmonly_rl_ladder.py --verify /tmp/ladder_L1.parquet --replies replies.jsonl
  python3 track4_sdmonly_rl_ladder.py --mimo wikis/WIKI_MIMO/data/music.parquet --lang en --show
  python3 track4_sdmonly_rl_ladder.py --selftest

replies.jsonl holds one {"instance_id": ..., "reply": ...} per line; --verify prints the pass rate per rung and level.
Spec: SPEC_SDMONLY_RL_LADDER_2026-10-04.md
"""


def main():
    ap = argparse.ArgumentParser(description="SDMONLY RL ladder: tasks and rule-check verifiers in MiMo-V2.6-RL-oss "
                                 "record format.", epilog=HELP_EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("generate")
    g.add_argument("--rung", help="one of " + ", ".join(GENERATED) + ", or all")
    g.add_argument("--level", type=int, default=1, choices=LEVELS)
    g.add_argument("--n", type=int, default=16, help="tasks per rung")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--split", choices=("train", "eval"), default="train")
    g.add_argument("--out", help=".jsonl or .parquet")
    g.add_argument("--show", action="store_true", help="print rows (rendered chat prompt and gold)")
    v = ap.add_argument_group("verify and inspect")
    v.add_argument("--verify", metavar="ROWS", help="rows file to score against --replies")
    v.add_argument("--replies", metavar="JSONL")
    v.add_argument("--mimo", metavar="PARQUET", help="read a MiMo-V2.6-RL-oss parquet and check its layout")
    v.add_argument("--lang", help="with --mimo on music: keep rows of this extra_info.lang")
    ap.add_argument("--list", action="store_true", help="list rungs and levels")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        ok = selftest(a.verbose)
        print("NEXT -> python3 track4_sdmonly_rl_ladder.py --rung all --level 1 --n 256 --out /tmp/ladder_L1.parquet")
        sys.exit(0 if ok else 1)
    if a.list:
        for k, name in RUNGS.items():
            src = "generated here, levels 1-5" if name in GENERATED else "MiMo's rows" + (
                " (format gate here; full score needs abc2midi)" if name == "mimo_music" else " (MASKED here)")
            print(f"  rung {k:2d}  {name:12s} {src}")
        print("NEXT -> python3 track4_sdmonly_rl_ladder.py --rung arith --level 1 --n 3 --show")
        return
    if a.mimo:
        rows = read_rows(a.mimo)
        if a.lang:
            rows = [r for r in rows if r["extra_info"].get("lang") == a.lang]
        cols = sorted(rows[0].keys()) if rows else []
        print(f"{a.mimo}: {len(rows)} rows, columns {cols}")
        print(f"  reward_model.ground_truth non-empty: {sum(1 for r in rows if r['reward_model']['ground_truth'])}")
        print(f"  extra_info keys: {sorted(rows[0]['extra_info'].keys()) if rows else []}")
        if a.show:
            for r in rows[:3]:
                print("---\n" + render_chat(r))
        print("NEXT -> score replies to these rows with compute_score(data_source, reply, '', extra_info)")
        return
    if a.verify:
        rows = {r["extra_info"]["instance_id"]: r for r in read_rows(a.verify)}
        tally = {}
        for rep in read_rows(a.replies):
            r = rows[rep["instance_id"]]
            s = compute_score(r["data_source"], rep["reply"], r["reward_model"]["ground_truth"], r["extra_info"])
            inst = json.loads(r["extra_info"]["instance_json"]) if r["data_source"].startswith("sdmonly") else {}
            key = (r["data_source"], inst.get("level"))
            t = tally.setdefault(key, [0, 0, 0])
            if s is None:
                t[2] += 1
            else:
                t[0] += s
                t[1] += 1
        for (ds, lv), (s, n, masked) in sorted(tally.items(), key=lambda kv: str(kv[0])):
            rate = s / n if n else float("nan")
            band = "IN BAND" if 0.05 <= rate <= 0.95 else ("TOO HARD" if rate < 0.05 else "TOO EASY")
            print(f"  {ds:28s} L{lv}  pass {rate:.3f} over {n}  masked {masked}  {band}")
        print("NEXT -> train only on levels IN BAND; see the spec's promotion rule")
        return
    if a.rung:
        rungs = GENERATED if a.rung == "all" else (a.rung,)
        rows = [r for rung in rungs for r in generate(rung, a.level, a.n, a.seed, a.split)]
        errs = schema_check(rows)
        if errs:
            print("schema errors:", errs[:5])
            sys.exit(1)
        if a.show:
            for r in rows:
                print(f"--- {r['extra_info']['instance_id']}\n{render_chat(r)}\n[gold] {r['reward_model']['ground_truth']}")
        if a.out:
            write_rows(rows, a.out)
            print(f"wrote {len(rows)} rows to {a.out}")
        print(f"NEXT -> sample replies from the SDM CHAT checkpoint for these prompts, then --verify ROWS --replies JSONL")
        return
    ap.print_help()
    print("\nNEXT -> python3 track4_sdmonly_rl_ladder.py --selftest")


if __name__ == "__main__":
    main()
