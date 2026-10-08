"""EPICS overlap check: how much of the held-out epic text also appears, verbatim, in the training text (lane EPICS).

<claudes_code_comments>
** Function List **
shingles(s, k) - every k-character substring of s, as a set of hashes
main() - rebuild the exact train and test documents of track4_epics_make_token_shards.py (same split, same caps),
         and report the fraction of test 64-character shingles found anywhere in the train text, overall and per epic

** Technical Review **
- A shingle of 64 characters is about 12 words; finding one in the train text means the test line is a verbatim repeat
  (a refrain, a formula, a quotation, a note repeated by the edition), which a model can score from memory.
- The control: the same fraction for the FineWeb-Edu-trained control is not needed; the number reports how much of
  the held-out set is repeats, so a reader can see what part of the epic-test gain could be memory.
</claudes_code_comments>
"""
import json
import os
import zlib

from track4_epics_make_token_shards import split_work, take, group_of, CAP_TRAIN, CAP_TEST

HERE = os.path.dirname(os.path.abspath(__file__))
K = 64


def shingles(s, k=K):
    return {zlib.crc32(s[i:i + k].encode()) ^ (len(s[i:i + k]) << 24) for i in range(0, max(0, len(s) - k + 1))}


def main():
    man = json.load(open(os.path.join(HERE, "EPICS_MANIFEST.json")))
    groups = {}
    for w in man["works"]:
        tr, te = split_work(open(os.path.join(HERE, w["clean"]["file"]), encoding="utf-8").read())
        g = groups.setdefault(group_of(w["slug"]), {"train": [], "test": []})
        g["train"] += tr
        g["test"] += te
    train_sh = set()
    tests = {}
    for name, g in groups.items():
        tr, _ = take(g["train"], CAP_TRAIN)
        te, _ = take(g["test"], CAP_TEST)
        for d in tr:
            train_sh |= shingles(d)
        tests[name] = te
    out = {"k": K, "per_epic": {}}
    hit_all = tot_all = 0
    for name, te in tests.items():
        hit = tot = 0
        for d in te:
            s = [d[i:i + K] for i in range(0, len(d) - K + 1)]
            tot += len(s)
            hit += sum(1 for x in s if (zlib.crc32(x.encode()) ^ (K << 24)) in train_sh)
        out["per_epic"][name] = round(hit / max(1, tot), 4)
        hit_all += hit
        tot_all += tot
    out["overall"] = round(hit_all / tot_all, 4)
    json.dump(out, open(os.path.join(HERE, "compose", "overlap_check.json"), "w"), indent=1)
    print("overall fraction of held-out 64-char shingles found verbatim in train:", out["overall"])
    for k, v in sorted(out["per_epic"].items(), key=lambda x: -x[1])[:8]:
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
