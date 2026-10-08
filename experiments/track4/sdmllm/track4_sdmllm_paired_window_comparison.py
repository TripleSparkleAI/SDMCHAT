"""SDMLLM S0 paired comparison: per-window TEST nats for several trained arms, and the paired bpb difference
of every pair with a standard error across windows.

<claudes_code_comments>
** Function List **
per_window(model, ...) - nats and bytes for every TEST window (BOS targets excluded)
main() - score the named runs on identical windows, write per-pair deltas with window-bootstrap SE

** Technical Review **
- Each arm is scored on the same 3,873 non-overlapping TEST windows (T=256) the trainer uses.
- Pair difference: delta = (sum_w nats_A - sum_w nats_B) / (ln 2 * sum_w bytes); its SE is a bootstrap over
  windows (2,000 resamples, seed 0), so the correlation between the two arms on the same text is kept.
- A difference is called resolved when |delta| > 2 SE. Seed-to-seed spread is reported separately by the
  report, and is the second yardstick a difference must clear.
</claudes_code_comments>
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track4_sdmllm_train_one_arm import load_split, val_windows, V  # noqa: E402
from track4_sdmllm_samples_and_repetition import load_model  # noqa: E402


@torch.no_grad()
def per_window(model, arr, starts, T, tb, device, bs=32):
    nats, byts = [], []
    cap = int(os.environ.get("SDM_EVAL_TOKENS", "8192"))  # the same override as eval_bpb, for a very wide model
    if T * bs > 2 * cap:  # long windows: keep about `cap` tokens of full-vocab logits per batch (default: T <= 512 unchanged)
        bs = max(1, cap // T)
    for i in range(0, len(starts), bs):
        x = np.stack([arr[a:a + T + 1] for a in starts[i:i + bs]]).astype(np.int64)
        xb = torch.from_numpy(x).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
            lg = model(xb[:, :-1])
        tgt = xb[:, 1:]
        nll = F.cross_entropy(lg.float().reshape(-1, V), tgt.reshape(-1), reduction="none").view(tgt.shape)
        m = tgt != 0
        nats.append((nll * m).sum(1).cpu().numpy())
        byts.append((tb[tgt] * m).sum(1).cpu().numpy())
    return np.concatenate(nats), np.concatenate(byts)


def main():
    runs = sys.argv[1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    val = load_split("val")
    tb = torch.from_numpy(np.fromfile(os.path.join(HERE, "data", "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
    starts = val_windows(val, 256, "test")
    per = {}
    for r in runs:
        m = load_model(r, device)
        n, b = per_window(m, val, starts, 256, tb, device)
        per[r] = n
        byts = b
        print(r, round(n.sum() / (math.log(2) * b.sum()), 5), flush=True)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(starts), size=(2000, len(starts)))
    out = {"windows": int(len(starts)), "bpb": {r: float(per[r].sum() / (math.log(2) * byts.sum())) for r in runs}, "pairs": {}}
    for i, a in enumerate(runs):
        for bname in runs[i + 1:]:
            d = per[a] - per[bname]
            delta = d.sum() / (math.log(2) * byts.sum())
            boot = d[idx].sum(1) / (math.log(2) * byts[idx].sum(1))
            se = float(boot.std())
            out["pairs"][f"{a} - {bname}"] = {"delta_bpb": float(delta), "se": se, "z": float(delta / se),
                                               "resolved": bool(abs(delta) > 2 * se)}
            print(a, "-", bname, round(float(delta), 5), "se", round(se, 5), flush=True)
    json.dump(out, open(os.path.join(HERE, "track4_sdmllm_paired_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
