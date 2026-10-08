"""SDMLLM on the Spark (lane SPARK24): extend the FineWeb-Edu training shard from 65.5M to billions of tokens,
with the SAME held-out TEST stream, proven identical by sha256.

<claudes_code_comments>
** Function List **
download(fn) - fetch one sample/10BT parquet file at the pinned revision (HF cache), return path and LFS sha256
docs_in(path, start) - iterate (row index, doc id, text) of one parquet file from row `start`, row group by row group
encode(tok, texts) - [BOS 0] + ids per text, as one uint32 array
main() - verify the existing train/val shards re-tokenize byte-identically, then append new documents to train_big.u32

** Technical Review **
- Source: HuggingFaceFW/fineweb-edu, sample/10BT, revision 87f09149 (the one SDMLLM S0 used), licence odc-by.
  S0 took file 000 rows 0..64,488 as train and rows 64,488..66,488 as val (2,000 docs; TEST is the second half
  of that stream). This script re-reads rows 0..66,488 from the downloaded file and re-tokenizes them with the
  same tokenizer JSON: the val shard must reproduce sha256 89fbba2c... exactly and the train shard c2f7bc05...,
  or the run stops. That proves the tokenizer, the document order and the doc format are unchanged on this box.
- New training text: file 000 rows 66,488 to the end, then files 001, 002, ... in order, until the cumulative
  DeepSeek token count reaches SDMLLM24_TARGET_TOKENS (default 3e9). Every new document whose exact text equals a
  val document's text (sha1) is skipped and counted. Doc format unchanged: [BOS id 0] + ids, no EOS.
- Output: data/train_big.u32 = train.u32 followed by the new documents (so the first 65.5M tokens are S0's
  train shard exactly), data/train_big_provenance.json (files, LFS sha256s, row ranges, token and byte counts,
  skipped duplicates, shard sha256). Batches in the trainer are drawn uniformly over the whole shard.
- Tokenization uses tokenizers' encode_batch (Rayon threads) over 20k-document chunks.
</claudes_code_comments>
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data"))
REPO = "HuggingFaceFW/fineweb-edu"
REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]
TARGET = int(float(os.environ.get("SDMLLM24_TARGET_TOKENS", "3e9")))
S0_TRAIN_DOCS, S0_VAL_DOCS = 64488, 2000
S0_SHA = {"train": "c2f7bc059eb399cad90a1ab38b2ecba0a679adc179d284aa0d34697dbdb9e5c7",
          "val": "89fbba2c788c0f15c4835cdc9197c035c1a9ad45365e83d3692c737653aa974f"}
CHUNK = 20000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download(fn):
    from huggingface_hub import hf_hub_download, HfApi
    p = hf_hub_download(REPO, fn, repo_type="dataset", revision=REV,
                        cache_dir=os.path.expanduser(os.environ.get("SDMLLM24_HF_CACHE", "~/settle24/hf")))
    info = HfApi().dataset_info(REPO, revision=REV, files_metadata=True)
    sib = next(s for s in info.siblings if s.rfilename == fn)
    lfs = getattr(sib, "lfs", None)
    return p, (lfs.sha256 if lfs else None), sib.size


def docs_in(path, start=0):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    row = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if row + n <= start:
            row += n
            continue
        t = pf.read_row_group(rg, columns=["text", "id"])
        for text, did in zip(t["text"].to_pylist(), t["id"].to_pylist()):
            if row >= start:
                yield row, did, text
            row += 1


def encode(tok, texts):
    enc = tok.encode_batch(texts, add_special_tokens=False)
    return np.concatenate([np.array([0] + e.ids, dtype=np.uint32) for e in enc]) if enc else np.zeros(0, np.uint32)


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, "ds4_v4flash_tokenizer.json"))
    tb = np.fromfile(os.path.join(DATA, "token_bytes.i32"), dtype=np.int32)
    t0 = time.time()
    prov = {"repo": REPO, "revision": REV, "licence": "odc-by", "config": "sample/10BT", "files": [],
            "target_tokens": TARGET, "made_by": "track4_sdmllm24_extend_fineweb_edu_shards.py (lane SPARK24)"}

    # 1. re-tokenize S0's rows and check both shards byte for byte
    p0, sha0, size0 = download(FILES[0])
    log("downloaded", FILES[0], size0, "bytes in", round(time.time() - t0), "s")
    texts, val_texts = [], []
    for row, did, text in docs_in(p0):
        if row >= S0_TRAIN_DOCS + S0_VAL_DOCS:
            break
        (texts if row < S0_TRAIN_DOCS else val_texts).append(text)
    checks = {}
    for name, tx in (("val", val_texts), ("train", texts)):
        h = hashlib.sha256()
        for i in range(0, len(tx), CHUNK):
            h.update(encode(tok, tx[i:i + CHUNK]).tobytes())
        checks[name] = h.hexdigest()
        ok = checks[name] == S0_SHA[name]
        log(f"re-tokenized S0 {name}: sha256 {checks[name][:16]} {'MATCHES' if ok else 'DIFFERS FROM'} S0 {S0_SHA[name][:16]}")
        if not ok:
            sys.exit(f"STOP: S0 {name} shard does not reproduce; the tokenizer or the document order differs")
    prov["s0_reproduced"] = checks
    val_hashes = {hashlib.sha1(t.encode("utf-8")).hexdigest() for t in val_texts}
    del texts, val_texts

    # 2. append new documents
    base = np.fromfile(os.path.join(DATA, "train.u32"), dtype=np.uint32)
    out_p = os.path.join(DATA, "train_big.u32.tmp")
    out = open(out_p, "wb")
    out.write(base.tobytes())
    total = int(base.size)
    shard_h = hashlib.sha256(base.tobytes())
    new_docs = skipped = new_bytes = 0
    for fi, fn in enumerate(FILES):
        if total >= TARGET:
            break
        p, sha, size = (p0, sha0, size0) if fi == 0 else download(fn)
        start = S0_TRAIN_DOCS + S0_VAL_DOCS if fi == 0 else 0
        rec = {"file": fn, "lfs_sha256": sha, "bytes": size, "first_row": start}
        buf, last_row = [], start
        for row, did, text in docs_in(p, start):
            if hashlib.sha1(text.encode("utf-8")).hexdigest() in val_hashes:
                skipped += 1
                continue
            buf.append(text)
            new_bytes += len(text.encode("utf-8"))
            last_row = row
            if len(buf) == CHUNK:
                a = encode(tok, buf)
                out.write(a.tobytes())
                shard_h.update(a.tobytes())
                total += int(a.size)
                new_docs += len(buf)
                buf = []
                log(f"{fn} row {row}: shard {total / 1e9:.3f}B tokens, {new_docs} new docs, {(time.time() - t0) / 60:.1f} min")
                if total >= TARGET:
                    break
        if buf and total < TARGET:
            a = encode(tok, buf)
            out.write(a.tobytes())
            shard_h.update(a.tobytes())
            total += int(a.size)
            new_docs += len(buf)
        rec["last_row"] = last_row
        prov["files"].append(rec)
    out.close()
    os.replace(out_p, os.path.join(DATA, "train_big.u32"))
    arr = np.memmap(os.path.join(DATA, "train_big.u32"), dtype=np.uint32, mode="r")
    nb = arr[base.size:]
    prov["split"] = {"train_big": {"tokens": int(arr.size), "s0_train_tokens_prefix": int(base.size),
                                   "new_docs": new_docs, "new_tokens": int(nb.size), "new_utf8_bytes": new_bytes,
                                   "new_token_bytes_sum_non_bos": int(tb[nb[nb != 0].astype(np.int64)].sum()),
                                   "skipped_exact_duplicates_of_val_docs": skipped, "sha256": shard_h.hexdigest()}}
    prov["made_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prov["seconds"] = round(time.time() - t0)
    json.dump(prov, open(os.path.join(DATA, "train_big_provenance.json"), "w"), indent=1)
    log("DONE", json.dumps(prov["split"]))


if __name__ == "__main__":
    main()
