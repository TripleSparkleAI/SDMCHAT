"""SDMLLMSTORE summary: collect every run's TEST bpb, eval-time ablations, clip fraction and location use into one json.
Example: python3 track4_sdmllmstore_summary.py > runs_store/summary.txt

<claudes_code_comments>
** Function List **
row(run) - one run's bpb, ablations, fraction of logged steps with gnorm above 1, max gnorm, location use
main() - rows for this lane's runs and the S0 references, written to track4_sdmllmstore_result.json

** Technical Review **
- Reads runs/<run>.result.json and runs/<run>.log from this worktree; S0 runs are read from the same tracked
  paths (S0's logs and result jsons are committed), so no checkpoint is touched.
- clip_frac = share of logged steps (every 50) whose pre-clip gradient norm exceeded the clip threshold 1.0, the
  quantity the pilot tied to the store's cost.
</claudes_code_comments>
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S0_REF = ["sdm_s0_20M", "sdm_s1_20M", "sdm_s2_20M", "sdm_nostore_s0_20M", "sdm_nostore_s1_20M", "sdm_nostore_s2_20M",
          "sdm_softness2_s0_20M", "sdm_mlp_s0_20M", "sdm_frozen_s0_20M", "qwen_s0_20M", "qwen_s1_20M", "qwen_s2_20M"]


def row(run):
    r = json.load(open(os.path.join(HERE, "runs", f"{run}.result.json")))
    steps = [json.loads(ln) for ln in open(os.path.join(HERE, "runs", f"{run}.log")) if '"event": "step"' in ln]
    out = {"bpb": round(r["val_test"]["bpb"], 5), "cfg": r["cfg"], "store_lr_mult": r.get("store_lr_mult", 1.0),
           "store_wd": r.get("store_wd", 0.1),
           "clip_frac": round(sum(s["gnorm"] > 1 for s in steps) / len(steps), 3),
           "gnorm_max": max(s["gnorm"] for s in steps),
           "stamp_start": r["stamp_start"], "stamp_end": r["stamp_end"]}
    for ab in ("shuffle_keys", "zero_read"):
        if f"val_test_ablate_{ab}" in r:
            out[f"ablate_{ab}"] = round(r[f"val_test_ablate_{ab}"]["bpb"], 5)
    if "locations_used_fraction_last_hop" in r:
        out["locations_used"] = round(r["locations_used_fraction_last_hop"], 3)
    return out


def main():
    mine = sorted(os.path.basename(p)[:-12] for p in glob.glob(os.path.join(HERE, "runs", "*.result.json")))
    rows = {r: row(r) for r in mine}
    json.dump({"runs": rows, "s0_reference": S0_REF}, open(os.path.join(HERE, "track4_sdmllmstore_result.json"), "w"), indent=1)
    for r, v in rows.items():
        print(f"{r:42s} {v['bpb']:.5f} clip {v['clip_frac']:.2f} gmax {v['gnorm_max']:7.3f}",
              {k: v[k] for k in v if k.startswith("ablate") or k == "locations_used"})


if __name__ == "__main__":
    main()
