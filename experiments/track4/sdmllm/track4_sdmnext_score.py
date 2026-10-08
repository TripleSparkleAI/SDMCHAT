"""SDMNEXT scoring: the no-store control at width 768 (NWL) and the chat fine-tune of SWL (SWLC), judged against the
predictions sealed in runs_sdmchats/PREREG_SDMNEXT_2026-10-03.md. Example, on the Spark:
    SDMLLM_DATA=~/settle24/sdmllm/data ~/settle24/venv/bin/python track4_sdmnext_score.py \
        --ck ~/settle24/ck/main --runs ~/settle24/runs/main --out sdmnext_score_result.json
    python3 track4_sdmnext_score.py --selftest        # no GPU, no data

<claudes_code_comments>
** Function List **
load_test(ck, run)                 - a run's cached per-window TEST nats and bytes (written by the trainer)
chat_per_window(ck, run, ...)      - per-window chat_test nats and bytes, computed once on the GPU and cached
pair(per, byts, idx, A, B)         - paired bpb delta A minus B and its window-bootstrap distribution
pairs_table(per, byts, pairs)      - every pair as delta, SE, z, resolved, with one shared resample index
result(runs_dir, run)              - a run's result json (TEST, CHAT, zero_read), or None if it has not landed
judge(r)                           - every sealed prediction as HIT, MISS or WAITING, with the number it was judged on
selftest()                         - the pair arithmetic and the judge on synthetic numbers
main()                             - score whatever has landed and write one JSON

** Technical Review **
- TEST: the trainer writes test_per_window.npz beside last.pt at the end of a run (3,892 non-overlapping T=256
  windows, 4,824,566 bytes). The bootstrap is SDMSCORE's: 2,000 window resamples from numpy default_rng(0), delta =
  (sum nats_A - sum nats_B) / (ln 2 * sum bytes), resolved when |delta| > 2 SE. Window bytes must match exactly.
- CHAT: the trainer writes only the chat_test total, so chat pairs need per-window nats. They are computed with the
  trainer's own per_window() on val_windows(chat_test, 256, "all", 4000), the windows the trainer's --extra-val used,
  and cached as chat_per_window.npz beside last.pt. The cached total must reproduce the result json's chat bpb to
  1e-4 or the script refuses (a check that the windows are the trainer's).
- The predictions are copied from the sealed PREREG verbatim as numbers (NL1 to NL5, LC1 to LC3, the decision rule and
  the promotion rule). LC4 to LC6 are browser-side measurements and are judged in the ledger, not here.
- A run that has not landed is WAITING, never guessed. Nothing here trains or changes a checkpoint.
</claudes_code_comments>
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SWL = "sdmchats_SWL_d768_s0_600M"
NWL = "sdmchats_NWL_d768_s0_600M"
SWLC = "sdmchats_SWLC_d768_chat_100M"
SWC = "sdmchats_SWC_d512_chat_100M"
NW0 = "sdmchats_NW_d512_s0_300M"
SW0 = "sdmchats_SW_d512_s0_300M"
TEST_PAIRS = [(SWL, NWL), (NWL, NW0), (SWL, SW0), (SWLC, SWL), (SWLC, SWC)]
CHAT_RUNS = [SWLC, SWC, SWL, NWL]
CHAT_PAIRS = [(SWLC, SWC), (SWL, NWL), (SWLC, SWL)]
SEED_MOVE = 0.005          # the decision rule: twice the largest seed-to-seed move measured at width 512


def load_test(ck, run):
    p = os.path.join(os.path.expanduser(ck), run, "test_per_window.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["nats"].astype(np.float64), z["bytes"].astype(np.float64)


def chat_per_window(ck, run, data, device, want_bpb=None):
    p = os.path.join(os.path.expanduser(ck), run, "chat_per_window.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["nats"].astype(np.float64), z["bytes"].astype(np.float64)
    last = os.path.join(os.path.expanduser(ck), run, "last.pt")
    if not os.path.exists(last):
        return None
    import torch
    sys.path.insert(0, HERE)
    os.environ.setdefault("SDMLLM_DATA", data)
    from track4_sdmllm_models import build
    from track4_sdmllm_train_one_arm import V, load_split, val_windows
    from track4_sdmllm_paired_window_comparison import per_window
    ckd = torch.load(last, map_location=device, weights_only=False)
    m = build(ckd["arm"], V, ckd["cfg"]).to(device)
    m.load_state_dict(ckd["model"])
    m.eval()
    tb = torch.from_numpy(np.fromfile(os.path.join(data, "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
    arr = load_split("chat_test")
    starts = val_windows(arr, 256, "all", 4000)
    with torch.no_grad():
        n, b = per_window(m, arr, starts, 256, tb, device)
    n, b = n.astype(np.float64), b.astype(np.float64)
    got = n.sum() / (math.log(2) * b.sum())
    if want_bpb is not None and abs(got - want_bpb) > 1e-4:
        raise SystemExit(f"{run}: per-window chat bpb {got:.6f} does not reproduce the result json's {want_bpb:.6f}")
    np.savez(p, nats=n, bytes=b)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return n, b


def pair(per, byts, idx, A, B):
    dlt = per[A] - per[B]
    delta = dlt.sum() / (math.log(2) * byts.sum())
    boot = dlt[idx].sum(1) / (math.log(2) * byts[idx].sum(1))
    return float(delta), boot


def pairs_table(per, byts, pairs):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(byts), size=(2000, len(byts)))
    out = {}
    for A, B in pairs:
        if A not in per or B not in per:
            continue
        d, boot = pair(per, byts, idx, A, B)
        se = float(boot.std())
        out[f"{A} - {B}"] = {"delta_bpb": d, "se": se, "z": d / se if se else float("inf"), "resolved": abs(d) > 2 * se}
    return out


def result(runs_dir, run):
    p = os.path.join(os.path.expanduser(runs_dir), f"{run}.result.json")
    if not os.path.exists(p):
        p = os.path.join(HERE, "runs_sdmchats", f"{run}.result.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    if "val_test" not in r:
        return None
    return {"test": r["val_test"]["bpb"], "chat": r.get("val_extra", {}).get("chat_test", {}).get("bpb"),
            "zero_read": r.get("val_test_ablate_zero_read", {}).get("bpb"),
            "shuffle_keys": r.get("val_test_ablate_shuffle_keys", {}).get("bpb"),
            "stamp_start": r.get("stamp_start"), "stamp_end": r.get("stamp_end")}


def _in(x, lo, hi):
    return "HIT" if lo <= x <= hi else "MISS"


def judge(r):
    """r: {"res": {run: result}, "test_pairs": {...}, "chat_pairs": {...}} -> {id: {verdict, value, sealed}}"""
    res, tp, cp = r["res"], r["test_pairs"], r["chat_pairs"]
    out = {}

    def put(k, sealed, value, verdict):
        out[k] = {"sealed": sealed, "value": value, "verdict": verdict}

    nwl, swl, swlc, nw0 = res.get(NWL), res.get(SWL), res.get(SWLC), res.get(NW0)
    if nwl:
        put("NL1", "NWL TEST in 1.395 to 1.412", nwl["test"], _in(nwl["test"], 1.395, 1.412))
        d = tp[f"{SWL} - {NWL}"]["delta_bpb"]
        put("NL2", "SWL minus NWL in -0.004 to +0.004", d, _in(d, -0.004, 0.004))
        put("NL3", "NWL CHAT within 0.015 of SWL's 1.63517", nwl["chat"], _in(nwl["chat"] - swl["chat"], -0.015, 0.015))
        d4 = tp[f"{NWL} - {NW0}"]["delta_bpb"]
        put("NL4", "NWL minus NW s0 in -0.035 to -0.020", d4, _in(d4, -0.035, -0.020))
        gap = swl["zero_read"] - nwl["test"]
        put("NL5", "SWL zero_read worse than NWL by at least 0.10", gap, "HIT" if gap >= 0.10 else "MISS")
        p = tp[f"{SWL} - {NWL}"]
        rule = ("the read helps" if p["delta_bpb"] < -SEED_MOVE and p["resolved"] else
                "the read hurts" if p["delta_bpb"] > SEED_MOVE and p["resolved"] else "tie")
        out["DECISION"] = {"rule": "helps if SWL-NWL < -0.005 and |z|>2; hurts if > +0.005 and |z|>2; else tie",
                           "value": p, "verdict": rule}
    else:
        for k in ("NL1", "NL2", "NL3", "NL4", "NL5"):
            put(k, "see PREREG", None, "WAITING")
    if swlc:
        put("LC1", "SWLC CHAT in 1.070 to 1.105", swlc["chat"], _in(swlc["chat"], 1.070, 1.105))
        d = swlc["test"] - swl["test"]
        put("LC2", "SWLC TEST worse than SWL by 0.030 to 0.070", d, _in(d, 0.030, 0.070))
        z = swlc["zero_read"] - swlc["test"]
        put("LC3", "SWLC zero_read cost +0.08 to +0.20", z, _in(z, 0.08, 0.20))
        cpair = cp.get(f"{SWLC} - {SWC}")
        if cpair:
            win = cpair["delta_bpb"] < 0 and cpair["resolved"]
            out["PROMOTION_CHAT_HALF"] = {"rule": "SWLC CHAT below SWC's on the same windows, |z| > 2 (LC4 >= 0.90 also needed)",
                                          "value": cpair, "verdict": "PASS" if win else "FAIL"}
    else:
        for k in ("LC1", "LC2", "LC3"):
            put(k, "see PREREG", None, "WAITING")
    return out


def selftest():
    ok = True
    rng = np.random.default_rng(1)
    byts = rng.integers(900, 1300, size=500).astype(np.float64)
    base = rng.gamma(5, 60, size=500)
    per = {"A": base - 0.01 * math.log(2) * byts, "B": base.copy(), "C": base + rng.normal(0, 1, 500)}
    t = pairs_table(per, byts, [("A", "B"), ("C", "B")])
    d = t["A - B"]["delta_bpb"]
    ok &= abs(d + 0.01) < 1e-12                         # a planted -0.01 bpb shift comes back exactly
    ok &= t["A - B"]["resolved"] and not t["C - B"]["resolved"]
    print("pair arithmetic", "PASS" if ok else "FAIL")
    res = {SWL: {"test": 1.40285, "chat": 1.63517, "zero_read": 1.54155},
           NWL: {"test": 1.4031, "chat": 1.64, "zero_read": None}, NW0: {"test": 1.43012},
           SWLC: {"test": 1.448, "chat": 1.09, "zero_read": 1.58}}
    tp = {f"{SWL} - {NWL}": {"delta_bpb": -0.00025, "se": 0.0004, "resolved": False},
          f"{NWL} - {NW0}": {"delta_bpb": -0.027, "se": 0.0004, "resolved": True}}
    cp = {f"{SWLC} - {SWC}": {"delta_bpb": -0.022, "se": 0.0005, "resolved": True}}
    j = judge({"res": res, "test_pairs": tp, "chat_pairs": cp})
    want = {"NL1": "HIT", "NL2": "HIT", "NL3": "HIT", "NL4": "HIT", "NL5": "HIT", "LC1": "HIT", "LC2": "HIT", "LC3": "HIT"}
    g1 = all(j[k]["verdict"] == v for k, v in want.items()) and j["DECISION"]["verdict"] == "tie" \
        and j["PROMOTION_CHAT_HALF"]["verdict"] == "PASS"
    print("judge, all inside the bands", "PASS" if g1 else "FAIL")
    # each band is aimed at by name: move one number out and only that prediction turns
    res2 = json.loads(json.dumps(res))
    res2[NWL]["test"] = 1.42
    tp2 = json.loads(json.dumps(tp))
    tp2[f"{SWL} - {NWL}"] = {"delta_bpb": -0.0172, "se": 0.0004, "resolved": True}
    j2 = judge({"res": res2, "test_pairs": tp2, "chat_pairs": cp})
    g2 = j2["NL1"]["verdict"] == "MISS" and j2["NL2"]["verdict"] == "MISS" and j2["DECISION"]["verdict"] == "the read helps" \
        and j2["NL3"]["verdict"] == "HIT"
    print("judge, NL1 and NL2 moved out", "PASS" if g2 else "FAIL")
    j3 = judge({"res": {SWL: res[SWL]}, "test_pairs": {}, "chat_pairs": {}})
    g3 = j3["NL1"]["verdict"] == "WAITING" and j3["LC1"]["verdict"] == "WAITING"
    print("judge, nothing landed is WAITING", "PASS" if g3 else "FAIL")
    allok = bool(ok and g1 and g2 and g3)
    print("SELFTEST", "PASS" if allok else "FAIL")
    return allok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("<claudes")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ck", default="~/settle24/ck/main")
    p.add_argument("--runs", default="~/settle24/runs/main")
    p.add_argument("--data", default=os.environ.get("SDMLLM_DATA", os.path.expanduser("~/settle24/sdmllm/data")))
    p.add_argument("--out", default="sdmnext_score_result.json")
    p.add_argument("--no-chat", action="store_true", help="skip the GPU chat per-window pass")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    res = {r: result(a.runs, r) for r in [SWL, NWL, SWLC, SWC, NW0, SW0]}
    res = {k: v for k, v in res.items() if v}
    per, byts = {}, None
    for r in res:
        t = load_test(a.ck, r)
        if t is None:
            continue
        if byts is not None and not np.array_equal(t[1], byts):
            raise SystemExit(f"TEST window bytes differ for {r}")
        per[r], byts = t
    test_pairs = pairs_table(per, byts, TEST_PAIRS) if byts is not None else {}
    chat_pairs = {}
    if not a.no_chat:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        cper, cbyts = {}, None
        for r in CHAT_RUNS:
            if r not in res:
                continue
            t = chat_per_window(a.ck, r, a.data, device, res[r]["chat"])
            if t is None:
                continue
            if cbyts is not None and not np.array_equal(t[1], cbyts):
                raise SystemExit(f"chat window bytes differ for {r}")
            cper[r], cbyts = t
            print("chat", r, round(float(t[0].sum() / (math.log(2) * t[1].sum())), 5), flush=True)
        chat_pairs = pairs_table(cper, cbyts, CHAT_PAIRS) if cbyts is not None else {}
    j = judge({"res": res, "test_pairs": test_pairs, "chat_pairs": chat_pairs})
    out = {"lane": "SDMNEXT", "prereg": "experiments/track4/sdmllm/runs_sdmchats/PREREG_SDMNEXT_2026-10-03.md",
           "stamp": {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
                     "load": [round(x, 2) for x in os.getloadavg()]},
           "bootstrap": "2000 window resamples, numpy default_rng(0)", "results": res,
           "test_pairs": test_pairs, "chat_pairs": chat_pairs, "judged": j}
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in test_pairs.items():
        print("TEST", k, f"{v['delta_bpb']:+.5f} se {v['se']:.5f}")
    for k, v in chat_pairs.items():
        print("CHAT", k, f"{v['delta_bpb']:+.5f} se {v['se']:.5f}")
    for k, v in j.items():
        print(k, v["verdict"], v.get("value") if not isinstance(v.get("value"), dict) else "")
    print("NEXT -> judge LC4 to LC6 after the export (tools/export_sdm_chat.py, browserspeed.py, loopcheck.mjs)")


if __name__ == "__main__":
    main()
