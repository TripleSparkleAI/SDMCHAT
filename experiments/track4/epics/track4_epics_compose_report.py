"""EPICS compose report: the three runs' result json files and the samples json -> the sealed prediction's verdicts and
the page's module SETTLE/settle-site/src/epics/composeResults.js (lane EPICS).

<claudes_code_comments>
** Function List **
load(run) - a run's result json from compose/ (copied from the Spark's ~/settle24/runs/main/)
verdicts(b, e, f) - the five sealed predictions of PREREG_EPICS_COMPOSE_2026-10-02.md, each held or missed
main() - write compose/SUMMARY.json and the page module; print the table

** Technical Review **
- Epic bpb is the trainer's --extra-val epics_test (every window of the held-out shard); FineWeb bpb is val_test (S0's
  3,873 TEST windows). The prediction's bands are applied exactly as sealed, with no rounding in their favour.
</claudes_code_comments>
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "..", "..", "SETTLE", "settle-site", "src", "epics", "composeResults.js")


def load(run):
    return json.load(open(os.path.join(HERE, "compose", f"{run}.result.json")))


def verdicts(b, e, f):
    eb, ee, ef = b["val_extra"]["epics_test"]["bpb"], e["val_extra"]["epics_test"]["bpb"], f["val_extra"]["epics_test"]["bpb"]
    wb, we, wf = b["val_test"]["bpb"], e["val_test"]["bpb"], f["val_test"]["bpb"]
    # the words are the page's (EpicCompose.jsx, through t()); this module keeps the numbers and the verdicts
    return [
        {"id": 1, "pass": 1.75 <= eb <= 1.95, "measured": eb},
        {"id": 2, "pass": 0.15 <= eb - ee <= 0.35, "measured": eb - ee},
        {"id": 3, "pass": abs(ef - eb) < 0.03, "measured": ef - eb},
        {"id": 4, "pass": ef - ee >= 0.10, "measured": ef - ee},
        {"id": 5, "pass": 0.02 <= we - wb <= 0.08 and abs(wf - wb) < 0.02, "measured": we - wb, "measured2": wf - wb},
    ]


def main():
    b, e, f = load("epics_BASE"), load("epics_EP12"), load("epics_FW12")
    v = verdicts(b, e, f)
    runs = [
        {"key": "BASE", "data": "nothing (SW, 300M FineWeb-Edu tokens)", "r": b, "src": "experiments/track4/epics/compose/epics_BASE.result.json"},
        {"key": "EP12", "data": "12M tokens of the epics", "r": e, "src": "experiments/track4/epics/compose/epics_EP12.result.json"},
        {"key": "FW12", "data": "12M tokens of FineWeb-Edu (control)", "r": f, "src": "experiments/track4/epics/compose/epics_FW12.result.json"},
    ]
    out = {"runs": [{"key": x["key"], "epicsBpb": x["r"]["val_extra"]["epics_test"]["bpb"], "fwBpb": x["r"]["val_test"]["bpb"],
                     "epicsBytes": x["r"]["val_extra"]["epics_test"]["bytes"], "src": x["src"]} for x in runs],
           "model": {"params": e["params"]["total_trainable"], "trainTokens": 300_000_000 + e["tokens_target"]},
           "verdicts": v, "testTokens": b["val_extra"]["epics_test"]["tokens"], "samples": []}
    sp = os.path.join(HERE, "compose", "epics_compose_samples.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        name = {"BASE (SW)": "BASE", "EP12 (epics)": "EP12", "FW12 (control)": "FW12"}
        for i, c in enumerate(s["cues"]):
            out["samples"].append({"work": c["work"], "cue": c["cue"],
                                   "by": {name[k]: {"greedy": s["samples"][k][i]["greedy"], "sampled": s["samples"][k][i]["sampled"]} for k in s["samples"]}})
    json.dump(out, open(os.path.join(HERE, "compose", "SUMMARY.json"), "w"), indent=1, ensure_ascii=False)
    js = ("// COMPOSE RESULTS - written by experiments/track4/epics/track4_epics_compose_report.py from the Spark runs' result json\n"
          "// files (experiments/track4/epics/compose/); do not edit by hand.\n"
          f"export const COMPOSE = {json.dumps(out, indent=1, ensure_ascii=False)};\n")
    open(SITE, "w").write(js)
    for x in out["runs"]:
        print(f"| {x['key']} | {x['epicsBpb']:.4f} | {x['fwBpb']:.4f} |")
    for x in v:
        print(("HELD  " if x["pass"] else "MISSED"), x["id"], round(x["measured"], 4), round(x.get("measured2", 0), 4))


if __name__ == "__main__":
    main()
