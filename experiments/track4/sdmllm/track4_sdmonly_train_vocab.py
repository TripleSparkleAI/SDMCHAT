"""SDMONLY vocabulary wave: train one SDM-only arm on a re-tokenized shard set, at an equal budget in BYTES.

<claudes_code_comments>
** Function List **
read_meta(data) - meta.json of a shard set (written by track4_sdmonly_vocab_shards.py), with sanity checks
tokens_for_bytes(meta, n_bytes) - the token budget that covers n_bytes of training text at this set's bytes per token
split_args(argv) - pull this wrapper's own flags (--data, --bytes) out of the argument list
selftest() - budget arithmetic, flag parsing, the V/token_bytes refusal, and that the S0 module really takes the V
main() - set SDMLLM_DATA, set the S0 trainer's vocabulary to the set's V, run track4_sdmonly_train.main(), stamp the result

** Technical Review **
- This file adds no training code. The S0 trainer (track4_sdmllm_train_one_arm) keeps the vocabulary size in a module
  global V and reads its shards from DATA, which it computes from SDMLLM_DATA at import. So this wrapper sets
  SDMLLM_DATA BEFORE importing anything, imports track4_sdmonly_train (which imports S0), sets S0.V to meta.json's V,
  and calls track4_sdmonly_train.main() with the remaining flags. Every place that reads V (the model build, eval_bpb,
  the ablation scoring) reads the module global at call time. The paired per-window scorer copies V at its own
  import, so it is imported here and its copy set too. token_bytes.i32 must have exactly V entries.
- Budget rule: arms are compared at EQUAL TRAINING BYTES. --bytes N becomes --tokens round(N / bytes_per_token),
  where bytes_per_token = train text bytes / train tokens INCLUDING each document's BOS (the trainer draws windows
  over the whole shard, BOS included). The result json gets a "vocab_wave" block with V, the bytes per token used,
  the byte budget, the token budget and the bytes the steps actually cover (steps are whole batches of B x T).
- --tokens without --bytes passes through unchanged (and is stamped the same way). --tokens with --bytes is refused.
- The wave's skip test looks for the "vocab_wave" key, which is written last, after the trainer and its ablations.
Docs: RESEARCH_SDMONLY_VOCAB_AND_SIDE_BY_SIDE_2026-10-03.md §6.
</claudes_code_comments>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
USAGE = """usage: track4_sdmonly_train_vocab.py [--data DIR] (--bytes N | --tokens N) [track4_sdmonly_train.py flags...]

Train one SDM-only arm on the shard set in DIR (or $SDMLLM_DATA), with the vocabulary its meta.json names.
  --data DIR    a data_<name>/ folder made by track4_sdmonly_vocab_shards.py (sets SDMLLM_DATA)
  --bytes N     training budget in bytes of text; converted to tokens with the set's measured bytes per token
  --selftest    check the budget arithmetic and that the trainer takes the set's V
Every other flag goes to track4_sdmonly_train.py (see its --help).

Example (the wave's 32k arm):
  SDMLLM_DATA=data_v32k python track4_sdmonly_train_vocab.py --bytes 96500000 --arm sdmonly --d 256 --hops 4 \\
      --n-sub 256 --seed 0 --run vw_v32k --loss chunked
"""


def read_meta(data):
    path = os.path.join(data, "meta.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: build the set with track4_sdmonly_vocab_shards.py first")
    meta = json.load(open(path))
    tb = os.path.getsize(os.path.join(data, "token_bytes.i32")) // 4
    if tb != meta["V"]:
        raise SystemExit(f"{data}: token_bytes.i32 has {tb} entries but meta.json says V = {meta['V']}")
    return meta


def tokens_for_bytes(meta, n_bytes):
    bpt = meta["splits"]["train"]["bytes_per_token"]
    return int(round(n_bytes / bpt)), bpt


def split_args(argv):
    data, n_bytes, rest, i = None, None, [], 0
    while i < len(argv):
        x = argv[i]
        if x in ("--data", "--bytes"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{x} needs a value")
            if x == "--data":
                data = argv[i + 1]
            else:
                n_bytes = int(float(argv[i + 1]))
            i += 2
            continue
        if x.startswith("--data=") or x.startswith("--bytes="):
            k, v = x.split("=", 1)
            if k == "--data":
                data = v
            else:
                n_bytes = int(float(v))
            i += 1
            continue
        rest.append(x)
        i += 1
    return data, n_bytes, rest


def selftest():
    import tempfile
    import numpy as np
    ok = []

    def check(name, cond, info=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)

    tmp = tempfile.mkdtemp(prefix="vw_selftest_", dir="/private/tmp" if os.path.isdir("/private/tmp") else None)
    meta = {"name": "st", "V": 7, "splits": {"train": {"bytes_per_token": 4.0}}}
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"))
    np.zeros(7, dtype=np.int32).tofile(os.path.join(tmp, "token_bytes.i32"))
    m = read_meta(tmp)
    check("tokens_for_bytes: 400 bytes at 4.0 bytes/token is 100 tokens", tokens_for_bytes(m, 400) == (100, 4.0))
    d, b, rest = split_args(["--data", tmp, "--bytes", "1e6", "--arm", "sdmonly", "--run=x"])
    check("split_args keeps the trainer's flags and takes --data/--bytes", d == tmp and b == 1000000 and rest == ["--arm", "sdmonly", "--run=x"])
    np.zeros(8, dtype=np.int32).tofile(os.path.join(tmp, "token_bytes.i32"))
    try:
        read_meta(tmp)
        refused = False
    except SystemExit:
        refused = True
    check("NEGATIVE: a token_bytes table of the wrong length is refused", refused)
    os.environ["SDMLLM_DATA"] = tmp
    sys.path.insert(0, HERE)
    import track4_sdmonly_train as T
    import track4_sdmllm_paired_window_comparison as PW
    check("the S0 module read SDMLLM_DATA at import", os.path.abspath(T.S0.DATA) == tmp)
    T.S0.V = 7
    PW.V = 7
    import torch
    model = T.M.build_sdmonly("sdmonly", T.S0.V, {"d": 16, "d_a": 8, "n_sub": 4, "k": 2, "hops": 1, "n_back": 2, "decays": (0.8,)})
    model.eval()
    tb = torch.ones(7, dtype=torch.int64)
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 3] * 4, dtype=np.uint32)
    ev = T.S0.eval_bpb(model, arr, np.array([0, 9]), 8, tb, torch.device("cpu"))
    check("S0.eval_bpb runs at the patched V (7)", ev["tokens"] > 0 and ev["bpb"] > 0)
    T.S0.V = 129280
    try:
        T.S0.eval_bpb(model, arr, np.array([0, 9]), 8, tb, torch.device("cpu"))
        broke = False
    except Exception:
        broke = True
    check("NEGATIVE: the unpatched V (129,280) does not fit a V 7 model", broke)
    n_ok = sum(ok)
    print(f"selftest {n_ok} of {len(ok)}", flush=True)
    return n_ok == len(ok)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    if argv[0] == "--selftest":
        good = selftest()
        print("NEXT -> python3 track4_sdmonly_vocab_shards.py --selftest")
        sys.exit(0 if good else 1)
    data, n_bytes, rest = split_args(argv)
    data = os.path.abspath(data or os.environ.get("SDMLLM_DATA", ""))
    if not data or not os.path.isdir(data):
        raise SystemExit("give --data DIR or set SDMLLM_DATA to a data_<name>/ folder\n\n" + USAGE)
    os.environ["SDMLLM_DATA"] = data
    meta = read_meta(data)
    has_tokens = any(x == "--tokens" or x.startswith("--tokens=") for x in rest)
    if n_bytes is not None and has_tokens:
        raise SystemExit("give --bytes or --tokens, not both")
    if n_bytes is not None:
        n_tok, bpt = tokens_for_bytes(meta, n_bytes)
        rest = rest + ["--tokens", str(n_tok)]
    elif not has_tokens:
        raise SystemExit("give --bytes N (the wave's rule) or --tokens N")
    sys.path.insert(0, HERE)
    import track4_sdmonly_train as T  # noqa: E402  (imports the S0 trainer, which reads SDMLLM_DATA now)
    import track4_sdmllm_paired_window_comparison as PW  # noqa: E402
    S0 = T.S0
    if os.path.abspath(S0.DATA) != data:
        raise SystemExit(f"S0.DATA is {S0.DATA}, expected {data}")
    S0.V = int(meta["V"])
    PW.V = int(meta["V"])
    a = T.parse(rest)
    bpt = meta["splits"]["train"]["bytes_per_token"]
    steps = a.tokens // (a.B * a.T)
    block = {"name": meta["name"], "V": int(meta["V"]), "data": data, "meta_sha256": None,
             "bytes_per_token_train": bpt, "bytes_target": n_bytes, "tokens_target": a.tokens,
             "steps": steps, "tokens_trained": steps * a.B * a.T, "bytes_trained": steps * a.B * a.T * bpt,
             "rule": "equal training bytes; tokens = round(bytes / bytes_per_token), bytes_per_token = train text "
                     "bytes / train tokens including BOS",
             "tokenizer_sha256": meta.get("tokenizer_sha256")}
    import hashlib
    block["meta_sha256"] = hashlib.sha256(open(os.path.join(data, "meta.json"), "rb").read()).hexdigest()
    print(json.dumps({"event": "vocab_wave", **block}), flush=True)
    sys.argv = [os.path.join(HERE, "track4_sdmonly_train.py")] + rest
    T.main()
    res_path = os.path.join(os.environ["SDMLLM_RUNS"], f"{a.run}.result.json")
    res = json.load(open(res_path))
    res["vocab_wave"] = block
    json.dump(res, open(res_path + ".tmp", "w"), indent=1)
    os.replace(res_path + ".tmp", res_path)
    print("VOCAB_WAVE", a.run, json.dumps({"V": block["V"], "bytes_per_token": round(bpt, 4),
                                           "tokens": a.tokens, "test_bpb": res["val_test"]["bpb"]}), flush=True)
    print(f"NEXT -> python3 track4_sdmonly_vocab_score.py --run {a.run} --data {data}")


if __name__ == "__main__":
    main()
