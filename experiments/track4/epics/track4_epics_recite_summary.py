"""EPICS recite summary: the recitation curve's frontier per (hard locations, counter width) from the cell jsons
(lane EPICS). Reads recite_curve/hash64_tuned and hash64_refine (Beowulf, plus Paradise Lost past 147,247 characters),
writes recite_curve/SUMMARY.json and prints a markdown table.

<claudes_code_comments>
** Function List **
cells() - every (M, N, bits) result from the 64-bit, tuned-radius cells, keyed by text
main() - per (M, bits): the largest N recited word for word and the smallest N that was not, with bytes per character

** Technical Review **
- WORD-PERFECT = the opening recital (cue: the first 48 characters) reproduces all N characters with no mistake.
- Errors = positions read wrong from the true context (teacher forced), (1 - accuracy) x (N - 1), rounded.
- Bytes = M x 256 x bits / 8 (the shipped counters); gzip is gzip -9 of the packed counters, as measured in the cell.
</claudes_code_comments>
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def cells():
    out = []
    for d in ("hash64_tuned", "hash64_refine"):
        for f in glob.glob(os.path.join(HERE, "recite_curve", d, "*.json")):
            j = json.load(open(f))
            for c in j["cells"]:
                out.append({"text": j["file"], "M": j["M"], "radius": j["radius"], "N": j["N"], "bits": c["bits"],
                            "wordPerfect": c["wordPerfect"], "errors": round((1 - c["teacherForcedAcc"]) * (j["N"] - 1)),
                            "openingFirstError": c["openingFirstError"], "cueMedian": c["cueMedianFirstError"],
                            "bytes": c["bytes"], "gzip": c["gzipBytes"], "src": os.path.relpath(f, HERE)})
    return out


def main():
    cs = cells()
    rows = []
    for M in sorted({c["M"] for c in cs}):
        for b in (16, 8, 4, 2, 1):
            g = [c for c in cs if c["M"] == M and c["bits"] == b]
            if not g:
                continue
            ok = [c for c in g if c["wordPerfect"]]
            best = max(ok, key=lambda c: c["N"]) if ok else None
            bad = [c for c in g if not c["wordPerfect"] and (not best or c["N"] > best["N"])]
            fail = min(bad, key=lambda c: c["N"]) if bad else None
            rows.append({"M": M, "bits": b, "radius": g[0]["radius"], "bytes": g[0]["bytes"],
                         "wp_max_N": best["N"] if best else 0, "wp_text": best["text"] if best else None,
                         "first_fail_N": fail["N"] if fail else None, "first_fail_errors": fail["errors"] if fail else None,
                         "bytes_per_char": round(g[0]["bytes"] / best["N"], 2) if best else None,
                         "gzip_per_char": round(best["gzip"] / best["N"], 2) if best else None})
    json.dump({"rows": rows, "cells": cs}, open(os.path.join(HERE, "recite_curve", "SUMMARY.json"), "w"), indent=1)
    print("| hard locations | bits | radius | memory bytes | word-perfect up to N | first N that failed (errors) | bytes per char | gzip bytes per char |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["bits"] == 16 or r["bits"] == 2:
            continue
        ff = f"{r['first_fail_N']:,} ({r['first_fail_errors']:,})" if r["first_fail_N"] else "not reached (the text ended)"
        print(f"| {r['M']:,} | {r['bits']} | {r['radius']} | {r['bytes']:,} | {r['wp_max_N']:,} | {ff} | {r['bytes_per_char'] or '-'} | {r['gzip_per_char'] or '-'} |")


if __name__ == "__main__":
    main()
