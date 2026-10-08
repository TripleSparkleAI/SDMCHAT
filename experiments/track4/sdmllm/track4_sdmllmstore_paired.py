"""SDMLLMSTORE paired comparison: per-window TEST nats for any set of runs (this lane's and S0's), paired bpb deltas
with a window-bootstrap SE. Example:
    SDMLLM_DATA=/abs/main/sdmllm/data SDMLLM_S0_CKPT=/abs/main/sdmllm/checkpoints \
    python3 track4_sdmllmstore_paired.py --out result.json sdm_opt_s0_20M:sdm_nostore_s0_20M qwen_s3_20M:sdm_nostore_s0_20M

<claudes_code_comments>
** Function List **
find_run(run) - locate a run's checkpoint dir: this lane's checkpoints first, then the S0 checkpoints (read-only)
per_window_cached(run, ...) - per-window nats/bytes; read test_per_window.npz, else score last.pt and cache it here
main() - score every run named in the pairs, write bpb per run and every pair's delta, SE, z

** Technical Review **
- Same windows (3,873 non-overlapping T=256 TEST windows), same scoring function and the same bootstrap draw
  (numpy default_rng(0), 2,000 resamples of windows) as S0's track4_sdmllm_paired_window_comparison.py, so the
  deltas are comparable with S0's table. A pair's delta = (sum_w nats_A - sum_w nats_B) / (ln 2 * sum_w bytes).
- S0 checkpoints are only read; a cache computed from one is written into THIS lane's gitignored checkpoints dir.
- A pair is 'resolved' when |delta| > 2 SE; seed spread is the second yardstick and is reported by the report.
- Reproduction control: recomputing S0's sdm_s0 - sdm_nostore_s0 must give S0's +0.0248.
</claudes_code_comments>
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track4_sdmllm_train_one_arm import load_split, val_windows, stamp, V, DATA  # noqa: E402
from track4_sdmllm_models import build  # noqa: E402
from track4_sdmllm_paired_window_comparison import per_window  # noqa: E402

LOCAL = os.path.join(HERE, "checkpoints")
S0 = os.environ.get("SDMLLM_S0_CKPT", LOCAL)


def find_run(run):
    for base in (LOCAL, S0):
        if os.path.exists(os.path.join(base, run, "last.pt")) or os.path.exists(os.path.join(base, run, "test_per_window.npz")):
            return os.path.join(base, run)
    raise FileNotFoundError(run)


def per_window_cached(run, val, starts, tb, device):
    d = find_run(run)
    for base in (os.path.join(LOCAL, run), d):
        f = os.path.join(base, "test_per_window.npz")
        if os.path.exists(f):
            z = np.load(f)
            return z["nats"], z["bytes"]
    ck = torch.load(os.path.join(d, "last.pt"), map_location=device, weights_only=False)
    m = build(ck["arm"], V, ck["cfg"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    n, b = per_window(m, val, starts, 256, tb, device)
    os.makedirs(os.path.join(LOCAL, run), exist_ok=True)
    np.savez(os.path.join(LOCAL, run, "test_per_window.npz"), nats=n, bytes=b)
    return n, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs", nargs="+", help="A:B pairs (delta = A minus B)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    val = load_split("val")
    tb = torch.from_numpy(np.fromfile(os.path.join(DATA, "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
    starts = val_windows(val, 256, "test")
    pairs = [x.split(":") for x in a.pairs]
    runs = sorted({r for pr in pairs for r in pr})
    per, byts = {}, None
    for r in runs:
        n, b = per_window_cached(r, val, starts, tb, device)
        if byts is not None:
            assert np.array_equal(b, byts), f"window bytes differ for {r}"
        per[r], byts = n, b
        print(r, round(float(n.sum() / (math.log(2) * b.sum())), 5), flush=True)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(starts), size=(2000, len(starts)))
    out = {"windows": int(len(starts)), "stamp": stamp(),
           "bpb": {r: float(per[r].sum() / (math.log(2) * byts.sum())) for r in runs}, "pairs": {}}
    for A, B in pairs:
        dlt = per[A] - per[B]
        delta = dlt.sum() / (math.log(2) * byts.sum())
        boot = dlt[idx].sum(1) / (math.log(2) * byts[idx].sum(1))
        se = float(boot.std())
        out["pairs"][f"{A} - {B}"] = {"delta_bpb": float(delta), "se": se, "z": float(delta / se),
                                       "resolved": bool(abs(delta) > 2 * se)}
        print(A, "-", B, round(float(delta), 5), "se", round(se, 5), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
