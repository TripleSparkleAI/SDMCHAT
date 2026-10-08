"""SDMSCORE paired comparison from cached per-window TEST nats (no model load). Example, on the Spark:
    python3 track4_sdmscore_paired_from_npz.py --ck ~/settle24/ck/main --out result.json \
        sdmchats_SW_d512_s1_300M:sdmchats_NW_d512_s1_300M sdmchats_SW_d512_s0_300M:sdmchats_NW_d512_s0_300M \
        --mean sdmchats_SW_d512_s0_300M:sdmchats_NW_d512_s0_300M,sdmchats_SW_d512_s1_300M:sdmchats_NW_d512_s1_300M

<claudes_code_comments>
** Function List **
load(ck, run) - read a run's test_per_window.npz (nats and bytes per window)
pair(per, byts, idx, A, B) - paired bpb delta A minus B with its window-bootstrap SE
main() - bpb per run, every pair, and the mean of several pairs (a seed average) with its bootstrap SE

** Technical Review **
- The trainer (track4_sdmllm_train_one_arm.py) writes test_per_window.npz beside last.pt at the end of a run, scored
  on the same non-overlapping T=256 TEST windows as S0's track4_sdmllm_paired_window_comparison.py. This script only
  reads those files, so a pair costs no GPU time.
- Delta = (sum_w nats_A - sum_w nats_B) / (ln 2 * sum_w bytes). SE = std of the same delta over 2,000 window
  resamples drawn with numpy default_rng(0), exactly as track4_sdmllmstore_paired.py draws them, so numbers agree with
  the earlier tables. Every run must carry byte-identical window byte counts or the script refuses.
- --mean A:B,C:D averages the listed pairs and bootstraps that average with the same resample indices (windows only).
  Seed spread is a separate yardstick: the script reports half the absolute difference of the pair deltas.
- Control: sdmchats_SW_d512_s0_300M minus sdmchats_NW_d512_s0_300M must give SDMCHATS's -0.00017 (SE 0.00033).
</claudes_code_comments>
"""
import argparse
import json
import math
import os
import socket
import time

import numpy as np


def load(ck, run):
    z = np.load(os.path.join(os.path.expanduser(ck), run, "test_per_window.npz"))
    return z["nats"].astype(np.float64), z["bytes"].astype(np.float64)


def pair(per, byts, idx, A, B):
    dlt = per[A] - per[B]
    delta = dlt.sum() / (math.log(2) * byts.sum())
    boot = dlt[idx].sum(1) / (math.log(2) * byts[idx].sum(1))
    return float(delta), boot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs", nargs="*", help="A:B pairs (delta = A minus B)")
    p.add_argument("--ck", required=True, help="checkpoint root holding RUN/test_per_window.npz")
    p.add_argument("--mean", action="append", default=[], help="comma list of A:B pairs to average")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    pairs = [x.split(":") for x in a.pairs]
    means = [[x.split(":") for x in m.split(",")] for m in a.mean]
    runs = sorted({r for pr in pairs for r in pr} | {r for m in means for pr in m for r in pr})
    per, byts = {}, None
    for r in runs:
        n, b = load(a.ck, r)
        if byts is not None and not np.array_equal(b, byts):
            raise SystemExit(f"window bytes differ for {r}")
        per[r], byts = n, b
    nw = len(byts)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, nw, size=(2000, nw))
    out = {"windows": nw, "bytes": int(byts.sum()), "bootstrap": "2000 window resamples, numpy default_rng(0)",
           "stamp": {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
                     "load": list(os.getloadavg())},
           "bpb": {r: float(per[r].sum() / (math.log(2) * byts.sum())) for r in runs}, "pairs": {}, "means": {}}
    for r in runs:
        print(r, round(out["bpb"][r], 5))
    for A, B in pairs:
        delta, boot = pair(per, byts, idx, A, B)
        se = float(boot.std())
        out["pairs"][f"{A} - {B}"] = {"delta_bpb": delta, "se": se, "z": delta / se, "resolved": abs(delta) > 2 * se}
        print(A, "-", B, round(delta, 5), "se", round(se, 5))
    for m in means:
        ds, bs = zip(*[pair(per, byts, idx, A, B) for A, B in m])
        mean = float(np.mean(ds))
        se = float(np.mean(bs, axis=0).std())
        key = " + ".join(f"({A} - {B})" for A, B in m) + f" / {len(m)}"
        out["means"][key] = {"mean_delta_bpb": mean, "window_se": se, "deltas": [float(d) for d in ds],
                             "seed_half_spread": float((max(ds) - min(ds)) / 2), "resolved": abs(mean) > 2 * se}
        print("MEAN", round(mean, 5), "window se", round(se, 5), "deltas", [round(float(d), 5) for d in ds])
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
