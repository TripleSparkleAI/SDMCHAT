"""SDMLLM S0 second yardstick: score the July SparseStar students (ds4-flash vocab, distilled from ds4-flash
text) on the SDMLLM FineWeb-Edu TEST windows, in bits per byte.

<claudes_code_comments>
** Function List **
load_student(path) - import the SparseStar student module by path and load a checkpoint
main() - score each listed checkpoint on the first N TEST windows; write json

** Technical Review **
- The students (PenttiFusion local attention plus a sparse global memory, 37.6M and 53.4M params, tied
  129,280 vocab, max_pos 256) were trained on 15.8k to about 100k tokens of ds4-flash generated text. Their
  numbers here are an out-of-domain reading: FineWeb-Edu is not their training distribution, so this is a
  yardstick for "what the earlier SDM-flavoured students know about web text", not a fair comparison.
- Same windows, same scoring rule (BOS targets excluded, bpb over token_bytes) as the S0 trainer, restricted
  to the first --windows TEST windows for time; the S0 arms are re-scored on the same subset for a paired
  comparison when their checkpoints exist.
</claudes_code_comments>
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track4_sdmllm_train_one_arm import eval_bpb, load_split, val_windows, V  # noqa: E402

SS = "/Users/happyrobot/Code/TripleSparkle/_flows/dwarfstar/experiments/track4/sparsestar-distilled"
CKPTS = [
    "corpus_data/checkpoints/student_best.pt",
    "latest-findings-round/lane5/ablation_2x2_out/A2_small37M_corpusFULL/seed0/best_on_S.pt",
    "latest-findings-round/lane5/ablation_2x2_out/A4_big53M_corpusFULL/seed0/best_on_S.pt",
]


def load_student(path, device):
    p = os.path.join(SS, "track4_student-penttifusion-local-attn-plus-sparse-global-memory-torch.py")
    spec = importlib.util.spec_from_file_location("track4_student_mod", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    ck = torch.load(os.path.join(SS, path), map_location="cpu", weights_only=False)
    cfg = mod.StudentConfig(**{k: v for k, v in ck["config"].items() if k in mod.StudentConfig.__dataclass_fields__})
    m = mod.SparseStarStudent(cfg)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--also", nargs="*", default=["qwen_s0_20M", "sdm_s0_20M"])
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    val = load_split("val")
    tb = torch.from_numpy(np.fromfile(os.path.join(HERE, "data", "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
    starts = val_windows(val, 255, "test", a.windows)  # max_pos 256 -> 255 inputs + 1 target
    out = {"windows": a.windows, "T": 255, "students": {}, "s0_arms_same_windows": {}}
    for p in CKPTS:
        m, ck = load_student(p, device)
        r = eval_bpb(m, val, starts, 255, tb, device, bs=16)
        out["students"][p] = {"n_params": ck.get("n_params"), "config": ck.get("config"), **r}
        print(p, r, flush=True)
    from track4_sdmllm_samples_and_repetition import load_model
    for run in a.also:
        if os.path.exists(os.path.join(HERE, "checkpoints", run, "last.pt")):
            m = load_model(run, device)
            r = eval_bpb(m, val, starts, 255, tb, device, bs=16)
            out["s0_arms_same_windows"][run] = r
            print(run, r, flush=True)
    json.dump(out, open(os.path.join(HERE, "track4_sdmllm_prior_students_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
