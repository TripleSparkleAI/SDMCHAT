"""EPICS compose data: the epics corpus as DeepSeek V4 Flash token shards for the SDMLLM trainer (lane EPICS).

<claudes_code_comments>
** Function List **
group_of(slug) - the epic a work belongs to (volumes and halves share one group: mahabharata-ganguli-v1 -> mahabharata-ganguli)
chunks(text, n) - the text cut into blocks of n lines (the last block may be short)
split_work(text) - (train blocks, test blocks): every block whose index % 20 == 7 is held out
main() - tokenize, cap each group, write epics_train.u32, epics_test.u32 and epics_shards_provenance.json

** Technical Review **
- HELD-OUT LINES. Each clean text is cut into blocks of 64 lines; block i is TEST when i % 20 == 7, so about 5% of every
  work is held out, spread through the poem rather than taken from its end (an edition's end is often notes or an
  index). Each maximal run of consecutive TRAIN blocks is one document; each TEST block is its own document.
- CAPS. One epic must not drown the rest: a group's train text stops at 3,000,000 characters and its test text at
  150,000 characters, taken in block order (first blocks first). The four Mahabharata volumes are one group, so the
  Mahabharata gives 3,000,000 training characters of its 14,947,263.
- FORMAT: the SDMLLM shard format exactly (track4_sdmllm_prepare_fineweb_edu_token_shards.py): each document is
  [BOS id 0] + tokenizer ids of the text with no special tokens, little-endian uint32. bpb uses the existing
  token_bytes.i32; the provenance json checks sum(token_bytes) over non-BOS ids against the utf-8 bytes per split.
</claudes_code_comments>
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data"))
BLOCK, EVERY, PICK = 64, 20, 7
CAP_TRAIN, CAP_TEST = 3_000_000, 150_000


def group_of(slug):
    return re.sub(r"-(v)?\d+$", "", slug)


def chunks(text, n=BLOCK):
    lines = text.split("\n")
    return ["\n".join(lines[i:i + n]) + "\n" for i in range(0, len(lines), n)]


def split_work(text):
    tr, te = [], []
    run = []
    for i, c in enumerate(chunks(text)):
        if i % EVERY == PICK:
            if run:
                tr.append("".join(run))
                run = []
            te.append(c)
        else:
            run.append(c)
    if run:
        tr.append("".join(run))
    return tr, te


def take(docs, cap):
    out, n = [], 0
    for d in docs:
        if n >= cap:
            break
        d = d[: cap - n]
        out.append(d)
        n += len(d)
    return out, n


def main():
    from tokenizers import Tokenizer
    man = json.load(open(os.path.join(HERE, "EPICS_MANIFEST.json")))
    tok = Tokenizer.from_file(os.path.join(DATA, "ds4_v4flash_tokenizer.json"))
    tb = np.fromfile(os.path.join(DATA, "token_bytes.i32"), dtype=np.int32)
    groups = {}
    for w in man["works"]:
        text = open(os.path.join(HERE, w["clean"]["file"]), encoding="utf-8").read()
        assert hashlib.sha256(text.encode()).hexdigest() == w["clean"]["sha256"], w["slug"]
        tr, te = split_work(text)
        g = groups.setdefault(group_of(w["slug"]), {"train": [], "test": [], "works": []})
        g["train"] += tr
        g["test"] += te
        g["works"].append(w["slug"])
    prov = {"manifest_built_utc": man["built_utc"], "block_lines": BLOCK, "test_rule": f"block index % {EVERY} == {PICK}",
            "cap_train_chars_per_group": CAP_TRAIN, "cap_test_chars_per_group": CAP_TEST, "groups": {}, "split": {}}
    out = {"train": [], "test": []}
    for name, g in groups.items():
        tr, ntr = take(g["train"], CAP_TRAIN)
        te, nte = take(g["test"], CAP_TEST)
        out["train"] += tr
        out["test"] += te
        prov["groups"][name] = {"works": g["works"], "train_chars": ntr, "test_chars": nte,
                                "train_chars_available": sum(map(len, g["train"])), "test_docs": len(te)}
    for split, docs in out.items():
        enc = tok.encode_batch(docs, add_special_tokens=False)
        ids = []
        for e in enc:
            ids.append(0)
            ids.extend(e.ids)
        arr = np.asarray(ids, dtype=np.uint32)
        path = os.path.join(DATA, f"epics_{split}.u32")
        arr.tofile(path)
        nb = int(tb[arr[arr != 0]].sum())
        utf8 = sum(len(d.encode("utf-8")) for d in docs)
        prov["split"][split] = {"docs": len(docs), "tokens": int(len(arr)), "utf8_bytes": utf8, "token_bytes_sum_non_bos": nb,
                                "token_bytes_minus_utf8_bytes": nb - utf8,
                                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "file": f"epics_{split}.u32"}
        print(split, prov["split"][split])
    prov["tokenizer_sha256"] = hashlib.sha256(open(os.path.join(DATA, "ds4_v4flash_tokenizer.json"), "rb").read()).hexdigest()
    json.dump(prov, open(os.path.join(HERE, "epics_shards_provenance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
