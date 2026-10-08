"""SDMLLM S0 data: a FineWeb-Edu sample tokenized with the DeepSeek V4 Flash tokenizer into uint32 shards.

<claudes_code_comments>
** Function List **
stream_docs(target_gpt2_tokens) - read FineWeb-Edu sample/10BT rows remotely, row group by row group
token_byte_lengths(tok) - utf-8 byte length of every vocab id (0 for special tokens), for bits per byte
main() - tokenize, split train/val by document, write shards + provenance json

** Technical Review **
- Source: HuggingFaceFW/fineweb-edu, config sample/10BT, file 000_00000.parquet, read through
  HfFileSystem so only the needed row groups cross the network. Revision is pinned and recorded.
- Documents are taken in file order until the dataset's own GPT-2 token_count column sums to the
  target. Each document is encoded as [BOS=0] + ids(text). Train and validation are split BY DOCUMENT:
  the LAST documents (a fixed count) are held out, so no document straddles the split.
- Bits per byte uses token_bytes[id] = number of utf-8 bytes the id decodes to (special tokens 0).
  bpb = sum over scored positions of CE (nats) / (ln 2 * sum of token_bytes of the targets). Positions
  whose target is BOS are not scored (a document boundary is not text).
- Added tokens matched literally in text count their own utf-8 bytes; the byte-count identity is reported
  per split (token_bytes_minus_utf8_bytes, 0 = exact), not assumed.
- Output: data/train.u32, data/val.u32 (raw little-endian uint32), data/token_bytes.i32, and
  track4_sdmllm_data_provenance.json (repo, revision, file, licence, row range, sha256 of shards).
</claudes_code_comments>
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPO = "HuggingFaceFW/fineweb-edu"
FILE = "sample/10BT/000_00000.parquet"
TARGET_GPT2_TOKENS = int(float(os.environ.get("SDMLLM_TARGET_TOKENS", "70e6")))
VAL_DOCS = int(os.environ.get("SDMLLM_VAL_DOCS", "2000"))


def token_byte_lengths(tok):
    # byte-level BPE: each vocab string is a sequence of the gpt2 byte-to-unicode chars
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    V = tok.get_vocab_size()
    out = np.zeros(V, dtype=np.int32)
    special = set()
    for a in json.load(open(os.path.join(DATA, "ds4_v4flash_tokenizer.json")))["added_tokens"]:
        special.add(a["id"])
    added = {a["id"]: a["content"] for a in json.load(open(os.path.join(DATA, "ds4_v4flash_tokenizer.json")))["added_tokens"]}
    for s, i in tok.get_vocab().items():
        if i in added:
            # an added token matched literally in text stands for its own utf-8 bytes
            out[i] = len(added[i].encode("utf-8"))
        elif all(ch in u2b for ch in s):
            out[i] = len(s)
        else:
            out[i] = len(s.encode("utf-8"))
    return out, special


def main():
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    os.makedirs(DATA, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPO)
    rev = info.sha
    lic = info.card_data.get("license") if info.card_data else None
    fs = HfFileSystem()
    path = f"datasets/{REPO}@{rev}/{FILE}"
    t0 = time.time()
    pf = pq.ParquetFile(fs.open(path, "rb", block_size=16 * 2**20))
    print("row groups", pf.num_row_groups, "rows", pf.metadata.num_rows, "cols", pf.schema_arrow.names)
    texts, ids_, gcount = [], [], 0
    rg = 0
    while gcount < TARGET_GPT2_TOKENS and rg < pf.num_row_groups:
        t = pf.read_row_group(rg, columns=["text", "id", "token_count"])
        for text, did, tc in zip(t["text"].to_pylist(), t["id"].to_pylist(), t["token_count"].to_pylist()):
            texts.append(text)
            ids_.append(did)
            gcount += tc
            if gcount >= TARGET_GPT2_TOKENS:
                break
        rg += 1
        print(f"  row group {rg}: docs {len(texts)} gpt2 tokens {gcount/1e6:.1f}M  {time.time()-t0:.0f}s", flush=True)
    print(f"read {len(texts)} docs in {time.time()-t0:.0f}s")

    tok = Tokenizer.from_file(os.path.join(DATA, "ds4_v4flash_tokenizer.json"))
    tb, special = token_byte_lengths(tok)
    np.asarray(tb, dtype=np.int32).tofile(os.path.join(DATA, "token_bytes.i32"))
    t1 = time.time()
    enc = tok.encode_batch(texts, add_special_tokens=False)
    print(f"tokenized in {time.time()-t1:.0f}s")
    split = len(texts) - VAL_DOCS
    out = {}
    for name, rng in (("train", range(0, split)), ("val", range(split, len(texts)))):
        parts = []
        nbytes = 0
        for i in rng:
            parts.append(np.array([0] + enc[i].ids, dtype=np.uint32))
            nbytes += len(texts[i].encode("utf-8"))
        arr = np.concatenate(parts)
        p = os.path.join(DATA, f"{name}.u32")
        arr.tofile(p)
        tb_sum = int(tb[arr[arr != 0].astype(np.int64)].sum())
        out[name] = {"docs": len(rng), "tokens": int(arr.size), "utf8_bytes": nbytes,
                     "token_bytes_sum_non_bos": tb_sum,
                     "doc_index_range": [rng.start, rng.stop], "first_id": ids_[rng.start], "last_id": ids_[rng.stop - 1],
                     "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        print(name, out[name])
    # the token_bytes table must reproduce the true utf-8 byte count of the text exactly
    for name in out:
        out[name]["token_bytes_minus_utf8_bytes"] = out[name]["token_bytes_sum_non_bos"] - out[name]["utf8_bytes"]
        print(name, "token_bytes - utf8 bytes =", out[name]["token_bytes_minus_utf8_bytes"])
    prov = {"repo": REPO, "revision": rev, "licence": lic, "file": FILE, "config": "sample/10BT",
            "selection": "documents in file order until the dataset's gpt2 token_count column sums to target",
            "target_gpt2_tokens": TARGET_GPT2_TOKENS, "row_groups_read": rg, "val_docs": VAL_DOCS,
            "doc_format": "[BOS id 0] + tokenizer ids of text, no EOS", "split": out,
            "tokenizer_sha256": hashlib.sha256(open(os.path.join(DATA, "ds4_v4flash_tokenizer.json"), "rb").read()).hexdigest(),
            "token_bytes_check": "sum of token_bytes over non-BOS ids minus utf-8 bytes of the source text, per split (0 = exact)",
            "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(prov, open(os.path.join(HERE, "track4_sdmllm_data_provenance.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
