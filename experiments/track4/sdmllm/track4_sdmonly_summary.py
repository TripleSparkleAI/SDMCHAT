"""SDMONLY summary: one table of every scored run, with paired differences against a named run.

<claudes_code_comments>
** Function List **
load_runs(runs_dir, prefix) - read every <run>.result.json whose name starts with the prefix
per_window(ck_dir, run) - the run's per-window TEST nats and bytes, or None when the file is absent
paired(a, b, draws, seed) - bpb(a) - bpb(b) over the same windows, with a bootstrap standard error and z
table(runs, ck_dir, against) - rows sorted by TEST bpb: score, ablations, speed, paired difference
selftest() - the paired difference on made-up windows: sign, zero for identical arms, SE shrinks with windows
main() - CLI

** Technical Review **
- TEST bpb is read from each result json (`val_test.bpb`), the S0 trainer's own number. Nothing is recomputed.
- The paired difference uses checkpoints/<run>/test_per_window.npz (nats and bytes per TEST window, written by
  the S0 trainer at the end of a run). bpb over a set of windows is sum(nats) / (ln 2 * sum(bytes)); the
  bootstrap resamples windows with replacement, the same windows for both arms, 2,000 draws by default.
- A run with no per-window file shows its score and a blank difference; it is never estimated.
- Median tokens a second is taken over the logged steps after the first 10% (compile and warmup excluded).
- Run it on the machine that holds the files. On the Spark:
  python3 track4_sdmonly_summary.py --runs ~/settle24/runs/sdmonly --ck ~/settle24/ck/sdmonly --prefix p0_ --against p0_A_c
</claudes_code_comments>
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_runs(runs_dir, prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, f"{prefix}*.result.json"))):
        r = json.load(open(f))
        out[r["run"]] = r
    return out


def per_window(ck_dir, run):
    f = os.path.join(ck_dir, run, "test_per_window.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return z["nats"].astype(np.float64), z["bytes"].astype(np.float64)


def paired(a, b, draws=2000, seed=0):
    (na, ba), (nb, bb) = a, b
    if len(na) != len(nb) or not np.array_equal(ba, bb):
        return None
    ln2 = math.log(2)
    diff = (na.sum() - nb.sum()) / (ln2 * ba.sum())
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(na), size=(draws, len(na)))
    d = (na[idx].sum(1) - nb[idx].sum(1)) / (ln2 * ba[idx].sum(1))
    se = float(d.std(ddof=1))
    return {"diff": float(diff), "se": se, "z": float(diff / se) if se > 0 else 0.0,
            "a_lower_frac": float((na < nb).mean())}


def median_tps(runs_dir, run):
    f = os.path.join(runs_dir, f"{run}.log")
    if not os.path.exists(f):
        return None
    tps = [json.loads(ln)["tok_per_s"] for ln in open(f) if '"event": "step"' in ln]
    tps = tps[len(tps) // 10:]
    return float(np.median(tps)) if tps else None


def table(runs, runs_dir, ck_dir, against):
    base = per_window(ck_dir, against) if against else None
    rows = []
    for name, r in runs.items():
        row = {"run": name, "arm": r["arm"], "test": r["val_test"]["bpb"],
               "non_emb": r["params"]["non_embedding_trainable"], "body_flops": r["flops_per_token_fwd"]["body"],
               "zero": r.get("val_test_ablate_zero_read", {}).get("bpb"),
               "shuffle": r.get("val_test_ablate_shuffle_keys", {}).get("bpb"), "tps": median_tps(runs_dir, name),
               "load": (r.get("stamp_start", {}).get("load") or [None])[0], "driver": r.get("stamp_start", {}).get("driver")}
        pw = per_window(ck_dir, name)
        row["paired"] = paired(pw, base) if (pw is not None and base is not None and name != against) else None
        rows.append(row)
    return sorted(rows, key=lambda x: x["test"])


def show(rows, against):
    print(f"{'run':<16}{'arm':<15}{'TEST bpb':>10}{'vs ' + (against or '-'):>14}{'SE':>9}{'z':>8}"
          f"{'zero_read':>11}{'shuffle':>10}{'non-emb M':>11}{'tok/s':>9}{'load':>6}")
    for r in rows:
        p = r["paired"]

        def f(v, n=5):
            return "" if v is None else f"{v:.{n}f}"

        d = "%+.5f" % p["diff"] if p else ""
        se = f(p["se"]) if p else ""
        z = "%+.1f" % p["z"] if p else ""
        tps = "%.0f" % r["tps"] if r["tps"] else ""
        print(f"{r['run']:<16}{r['arm']:<15}{r['test']:>10.5f}{d:>14}{se:>9}{z:>8}{f(r['zero']):>11}{f(r['shuffle']):>10}"
              f"{r['non_emb'] / 1e6:>11.2f}{tps:>9}{f(r['load'], 1):>6}")


def selftest():
    g = np.random.default_rng(1)
    by = g.integers(900, 1400, size=800).astype(np.float64)
    a = (g.normal(1.1, 0.05, size=800) * by, by)
    b = (a[0] + 0.01 * by, by)
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    p = paired(a, b)
    check("a lower than b gives a negative difference", p["diff"] < 0)
    check("the difference is 0.01 nats per byte in bits", abs(p["diff"] + 0.01 / math.log(2)) < 1e-9)
    check("identical arms give exactly zero", paired(a, a)["diff"] == 0.0 and paired(a, a)["z"] == 0.0)
    noisy = (a[0] + g.normal(0, 5, size=800), by)
    small = paired((a[0][:100], by[:100]), (noisy[0][:100], by[:100]))
    big = paired(a, noisy)
    check("the standard error shrinks with more windows", big["se"] < small["se"])
    check("different windows are refused", paired(a, (b[0], by + 1)) is None)
    print(f"{sum(ok)} of {len(ok)} checks pass")
    return all(ok)


def main():
    p = argparse.ArgumentParser(description="SDMONLY summary table. Example: python3 %(prog)s --prefix p0_ --against p0_A_c")
    p.add_argument("--runs", default=os.path.join(HERE, "runs_sdmonly"))
    p.add_argument("--ck", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--prefix", default="p0_")
    p.add_argument("--against", default=None, help="run name every other run is compared with, window by window")
    p.add_argument("--json", default=None, help="also write the rows here")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    runs = load_runs(os.path.expanduser(a.runs), a.prefix)
    if not runs:
        print(f"no result files under {a.runs} with prefix {a.prefix}. Runs still training have a .log and no .result.json.")
        return
    rows = table(runs, os.path.expanduser(a.runs), os.path.expanduser(a.ck), a.against)
    show(rows, a.against)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    print("NEXT -> record the table in PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md and score the sealed predictions")


if __name__ == "__main__":
    main()
