"""SDMONLY vocabulary wave: re-tokenize the S0 FineWeb-Edu shards with another tokenizer, document by document.

<claudes_code_comments>
** Function List **
byte_level_map() - the GPT-2 byte-to-unicode table, inverted (vocab char -> byte)
source_byte_table(tok_json) - true bytes of every DeepSeek id (added tokens: their utf-8 content; BOS: nothing)
split_docs(arr) - start and end of each document in a [BOS 0] + ids shard
decode_split(arr, table, max_docs) - documents of a shard back to text, plus their utf-8 byte counts
encoder_without_added(path) - the target tokenizer with its added tokens removed, for encoding text only
target_byte_table(tok_json) - utf-8 bytes of every target id (0 for BOS), refusing any non-byte-level entry
bos_unreachable(tok_json, bos) - structural proof that no merge and no alphabet byte yields the BOS id
encode_split(enc, texts, bos) - [BOS] + ids per document, with a count of any BOS produced from text
sha(x) - sha256 of bytes or of a file
write_meta(...) - meta.json for one shard set: V, BOS, tokens, bytes, bytes per token, sha256 of every file
build_target(name, path, ...) - write data_<name>/ from decoded text with one tokenizer
build_as_is(name, ...) - data_<name>/ for the source tokenizer itself: links to the source shards, plus meta.json
selftest() - round trips, byte identities, BOS proof, and two negative controls, on a few real val documents
main() - CLI

** Technical Review **
- Source: the S0 shards (`train.u32`, `val.u32`, [BOS id 0] + DeepSeek ids per document, no EOS) and the DeepSeek
  tokenizer json beside them. Each document is decoded by joining the TRUE bytes of its ids: byte-level vocab strings
  map through the GPT-2 byte table; added tokens stand for their own utf-8 content. The joined bytes must decode as
  strict utf-8. The total is checked against `track4_sdmllm_data_provenance.json`'s `utf8_bytes` (the text the
  shards were made from) and the difference is reported, not assumed. (The canonical `token_bytes.i32` over-counts
  by 2,090 bytes on train and 57 on val; this table does not. That difference is reported for the as-is set too.)
- Target tokenizers are byte-level BPE files (`tokenizers` library json) with `<|bos|>` at id 0 as their only added
  token. BOS GUARANTEE, three parts: (1) text is encoded by a copy of the tokenizer with its added tokens removed, so a
  literal "<|bos|>" in text is encoded as ordinary bytes and never matched as the special token; (2) id 0's vocab
  string is not a single alphabet byte and no merge rule produces it (checked on the file), so plain BPE cannot emit
  it; (3) every encoded document is checked, and any id 0 produced from text aborts the build.
- Every target id's byte count comes from its own vocab string (all characters must be in the byte alphabet, else
  the build refuses). Per document, the sum over its ids must equal the utf-8 length of its text, exactly; the build
  refuses otherwise. So for a target set, sum of token_bytes over non-BOS ids == text bytes, to the byte.
- Output data_<name>/: train.u32, val.u32 (uint32, the S0 format, read by the S0 trainer unchanged),
  token_bytes.i32, tokenizer.json (the full file, added token kept, for decoding), val_doc_bytes.i64 (true utf-8
  bytes of each of the val documents, the same file for every set; the scorer's denominator), and meta.json.
- bytes per token = text bytes / tokens INCLUDING the BOS of each document, because the trainer draws windows over
  the whole shard; this is the number the wrapper uses to turn a byte budget into a token budget.
- `--max-docs N` caps TRAIN documents (for an M5 smoke); val is always complete, since the scorer needs docs
  1,000 to 1,999. A capped set says so in meta.json and the provenance byte check is reported as not applicable.
Docs: RESEARCH_SDMONLY_VOCAB_AND_SIDE_BY_SIDE_2026-10-03.md §3.1 and §6.
</claudes_code_comments>
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROV = os.path.join(HERE, "track4_sdmllm_data_provenance.json")
SRC_V = 129280


def byte_level_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def source_byte_table(tok_json):
    """True bytes of every source id. BOS (0) decodes to nothing; added tokens to their own content."""
    j = json.load(open(tok_json))
    u2b = byte_level_map()
    vocab = j["model"]["vocab"]
    V = max(max(vocab.values()), max(a["id"] for a in j["added_tokens"])) + 1
    table = [b""] * V
    for s, i in vocab.items():
        if all(ch in u2b for ch in s):
            table[i] = bytes(u2b[ch] for ch in s)
        else:
            table[i] = s.encode("utf-8")
    for a in j["added_tokens"]:
        table[a["id"]] = a["content"].encode("utf-8")
    table[0] = b""
    return table


def split_docs(arr):
    starts = np.flatnonzero(arr == 0)
    if len(starts) == 0 or starts[0] != 0:
        raise SystemExit("shard does not start with BOS (id 0)")
    ends = np.append(starts[1:], len(arr))
    return starts, ends


def decode_split(arr, table, max_docs=None):
    starts, ends = split_docs(arr)
    n = len(starts) if max_docs is None else min(max_docs, len(starts))
    texts, nbytes, bad = [], np.zeros(n, dtype=np.int64), []
    for d in range(n):
        ids = arr[starts[d] + 1:ends[d]].tolist()
        bb = b"".join([table[i] for i in ids])
        nbytes[d] = len(bb)
        try:
            texts.append(bb.decode("utf-8"))
        except UnicodeDecodeError:
            bad.append(d)
            texts.append(None)
    if bad:
        raise SystemExit(f"{len(bad)} documents do not decode as strict utf-8 (first: {bad[:5]}); refusing")
    return texts, nbytes, int(ends[n - 1])


def encoder_without_added(path):
    from tokenizers import Tokenizer
    j = json.load(open(path))
    j["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(j))


def target_byte_table(tok_json, bos=0):
    j = json.load(open(tok_json))
    u2b = byte_level_map()
    vocab = j["model"]["vocab"]
    V = max(vocab.values()) + 1
    if sorted(vocab.values()) != list(range(V)):
        raise SystemExit(f"{tok_json}: vocab ids are not 0..{V - 1} without gaps")
    out = np.zeros(V, dtype=np.int32)
    for s, i in vocab.items():
        if i == bos:
            continue
        if not all(ch in u2b for ch in s):
            raise SystemExit(f"{tok_json}: id {i} {s!r} is not byte-level; refusing (token_bytes would be wrong)")
        out[i] = len(s)
    return out, V


def bos_unreachable(tok_json, bos=0):
    j = json.load(open(tok_json))
    vocab = j["model"]["vocab"]
    inv = {i: s for s, i in vocab.items()}
    bos_s = inv[bos]
    specials = [a for a in j["added_tokens"]]
    merges = j["model"]["merges"]
    pairs = [m.split(" ", 1) if isinstance(m, str) else m for m in merges]
    produced = sum(1 for a, b in pairs if a + b == bos_s)
    return {"bos_id": bos, "bos_string": bos_s, "single_alphabet_char": len(bos_s) == 1,
            "merges_producing_bos": produced, "added_tokens": [(a["id"], a["content"]) for a in specials],
            "ok": len(bos_s) > 1 and produced == 0 and all(a["id"] == bos for a in specials)}


def encode_split(enc, texts, bos=0, batch=4096):
    parts, produced = [], 0
    for i in range(0, len(texts), batch):
        for e in enc.encode_batch(texts[i:i + batch], add_special_tokens=False):
            ids = np.asarray(e.ids, dtype=np.uint32)
            produced += int((ids == bos).sum())
            parts.append(np.concatenate([np.array([bos], dtype=np.uint32), ids]))
    return parts, produced


def sha(x):
    if isinstance(x, (bytes, bytearray)):
        return hashlib.sha256(x).hexdigest()
    h = hashlib.sha256()
    with open(x, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def text_sha(texts):
    h = hashlib.sha256()
    for t in texts:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def write_meta(out, name, V, tok_path, splits, extra):
    meta = {"name": name, "V": int(V), "bos_id": 0, "doc_format": "[BOS id 0] + ids of text, no EOS",
            "tokenizer_file": os.path.basename(tok_path), "tokenizer_sha256": sha(tok_path), "splits": splits,
            "files_sha256": {f: sha(os.path.join(out, f)) for f in ("train.u32", "val.u32", "token_bytes.i32",
                                                                    "val_doc_bytes.i64")},
            "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "made_by": "track4_sdmonly_vocab_shards.py",
            **extra}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return meta


def split_record(parts_or_arr, nbytes_docs, texts, tb):
    arr = parts_or_arr if isinstance(parts_or_arr, np.ndarray) else np.concatenate(parts_or_arr)
    nb = int(nbytes_docs.sum())
    tb_sum = int(tb[arr[arr != 0].astype(np.int64)].sum())
    return arr, {"docs": int(len(nbytes_docs)), "tokens": int(arr.size), "tokens_non_bos": int((arr != 0).sum()),
                 "text_bytes": nb, "token_bytes_sum_non_bos": tb_sum, "token_bytes_minus_text_bytes": tb_sum - nb,
                 "bytes_per_token": nb / arr.size, "bytes_per_text_token": nb / max(1, int((arr != 0).sum())),
                 "text_sha256": text_sha(texts)}


def decode_source(src, src_tok, max_docs):
    table = source_byte_table(src_tok)
    prov = json.load(open(PROV)) if os.path.exists(PROV) else None
    dec, check = {}, {}
    for split in ("train", "val"):
        arr = np.fromfile(os.path.join(src, f"{split}.u32"), dtype=np.uint32)
        cap = max_docs if split == "train" else None
        t0 = time.time()
        texts, nbytes, used = decode_split(arr, table, cap)
        dec[split] = (texts, nbytes, arr[:used])
        full = cap is None or cap >= len(split_docs(arr)[0])
        want = prov["split"][split]["utf8_bytes"] if prov else None
        check[split] = {"decoded_docs": len(texts), "decoded_bytes": int(nbytes.sum()),
                        "provenance_utf8_bytes": want,
                        "decoded_minus_provenance": (int(nbytes.sum()) - want) if (want is not None and full) else None,
                        "capped": not full, "source_sha256": sha(arr.tobytes()), "secs": round(time.time() - t0, 1)}
        print(json.dumps({"event": "decoded", "split": split, **check[split]}), flush=True)
    return dec, check


def build_target(name, path, dec, check, out_root, max_docs):
    from tokenizers import __version__ as tv
    out = os.path.join(out_root, f"data_{name}")
    os.makedirs(out, exist_ok=True)
    proof = bos_unreachable(path)
    if not proof["ok"]:
        raise SystemExit(f"{name}: BOS can be produced from text or is not the only added token: {proof}")
    tb, V = target_byte_table(path)
    enc = encoder_without_added(path)
    splits = {}
    for split in ("train", "val"):
        texts, nbytes, _ = dec[split]
        t0 = time.time()
        parts, produced = encode_split(enc, texts)
        if produced:
            raise SystemExit(f"{name} {split}: {produced} BOS ids produced from text; refusing")
        per_doc = np.array([int(tb[p[1:].astype(np.int64)].sum()) for p in parts], dtype=np.int64)
        mism = np.flatnonzero(per_doc != nbytes)
        if len(mism):
            raise SystemExit(f"{name} {split}: token bytes differ from text bytes in {len(mism)} docs (first {mism[:5]})")
        arr, rec = split_record(parts, nbytes, texts, tb)
        rec["literal_bos_string_docs"] = sum(1 for t in texts if "<|bos|>" in t)
        rec["encode_secs"] = round(time.time() - t0, 1)
        arr.tofile(os.path.join(out, f"{split}.u32.tmp"))
        os.replace(os.path.join(out, f"{split}.u32.tmp"), os.path.join(out, f"{split}.u32"))
        splits[split] = rec
        print(json.dumps({"event": "encoded", "name": name, "split": split, **{k: rec[k] for k in
              ("docs", "tokens", "text_bytes", "bytes_per_token", "token_bytes_minus_text_bytes", "encode_secs")}}), flush=True)
    tb.tofile(os.path.join(out, "token_bytes.i32"))
    dec["val"][1].astype(np.int64).tofile(os.path.join(out, "val_doc_bytes.i64"))
    j = json.load(open(path))
    with open(os.path.join(out, "tokenizer.json"), "w") as f:
        json.dump(j, f, ensure_ascii=False)
    meta = write_meta(out, name, V, path, splits, {
        "tokenizer_source": os.path.abspath(path), "tokenizers_version": tv, "bos_guarantee": {
            "how": "encode with added tokens removed; no merge or alphabet byte yields id 0 (checked); every doc checked",
            **proof}, "decode_check": check, "max_train_docs": max_docs})
    print(json.dumps({"event": "built", "name": name, "out": out, "V": V,
                      "bytes_per_token_train": meta["splits"]["train"]["bytes_per_token"]}), flush=True)
    return meta


def build_as_is(name, src, src_tok, dec, check, out_root):
    out = os.path.join(out_root, f"data_{name}")
    os.makedirs(out, exist_ok=True)
    for f, target in (("train.u32", "train.u32"), ("val.u32", "val.u32"), ("token_bytes.i32", "token_bytes.i32"),
                      ("tokenizer.json", os.path.basename(src_tok))):
        link = os.path.join(out, f)
        rel = os.path.relpath(os.path.join(src, target), out)
        if os.path.islink(link) or os.path.exists(link):
            if os.path.realpath(link) == os.path.realpath(os.path.join(src, target)):
                continue
            raise SystemExit(f"{link} exists and points elsewhere; refusing to replace it")
        os.symlink(rel, link)
    tb = np.fromfile(os.path.join(src, "token_bytes.i32"), dtype=np.int32)
    splits = {}
    for split in ("train", "val"):
        texts, nbytes, arr = dec[split]
        if split == "train" and check["train"]["capped"]:
            raise SystemExit("--deepseek-as-is needs the whole train shard (no --max-docs)")
        _, rec = split_record(arr, nbytes, texts, tb.astype(np.int64))
        splits[split] = rec
    dec["val"][1].astype(np.int64).tofile(os.path.join(out, "val_doc_bytes.i64"))
    meta = write_meta(out, name, SRC_V, src_tok, splits, {
        "as_is": True, "bos_guarantee": {"how": "source shards: BOS count equals document count (checked)",
                                         "bos_count_train": int((dec["train"][2] == 0).sum()),
                                         "docs_train": splits["train"]["docs"]},
        "note": "token_bytes.i32 is the canonical S0 table; its sum differs from the text bytes as reported per split",
        "decode_check": check})
    if meta["bos_guarantee"]["bos_count_train"] != splits["train"]["docs"]:
        raise SystemExit("source train shard: BOS count differs from document count")
    print(json.dumps({"event": "built", "name": name, "out": out, "V": SRC_V,
                      "bytes_per_token_train": splits["train"]["bytes_per_token"]}), flush=True)
    return meta


def selftest():
    import tempfile
    ok = []

    def check(name, cond, info=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)

    src = os.path.join(HERE, "data")
    src_tok = os.path.join(src, "ds4_v4flash_tokenizer.json")
    tokdir = os.path.join(HERE, "tokenizers")
    tpath = os.path.join(tokdir, "trained_bpe_32768.json")
    from tokenizers import Tokenizer
    val = np.fromfile(os.path.join(src, "val.u32"), dtype=np.uint32)
    s, e = split_docs(val)
    small = val[:e[19]]
    table = source_byte_table(src_tok)
    texts, nbytes, _ = decode_split(small, table)
    full = Tokenizer.from_file(src_tok)
    re_ids = [full.encode(t, add_special_tokens=False).ids for t in texts]
    orig = [small[s[d] + 1:e[d]].tolist() for d in range(20)]
    check("DeepSeek decode then encode gives the original ids (20 val docs)", re_ids == orig)
    check("decoded bytes are the utf-8 length of the text", all(len(t.encode()) == n for t, n in zip(texts, nbytes)))
    proof = bos_unreachable(tpath)
    check("trained 32k: BOS is unreachable from text (no merge, not a byte, only added token)", proof["ok"], str(proof["merges_producing_bos"]))
    enc = encoder_without_added(tpath)
    lit = enc.encode("hello <|bos|> world", add_special_tokens=False).ids
    check("a literal '<|bos|>' in text never becomes id 0", 0 not in lit and len(lit) > 3)
    fulltok = Tokenizer.from_file(tpath)
    check("the full file WOULD match it (why the encoder drops added tokens)",
          0 in fulltok.encode("hello <|bos|> world", add_special_tokens=False).ids)
    tb, V = target_byte_table(tpath)
    parts, produced = encode_split(enc, texts)
    per_doc = np.array([int(tb[p[1:].astype(np.int64)].sum()) for p in parts])
    check("trained 32k: token bytes equal text bytes in every doc", (per_doc == nbytes).all() and produced == 0)
    dec_back = [fulltok.decode(p[1:].tolist()) for p in parts]
    check("trained 32k: decode(encode(doc)) == doc", dec_back == texts)
    # negative control 1: a byte table shifted by one id no longer matches the text bytes
    tb_bad = np.roll(tb, 1)
    per_bad = np.array([int(tb_bad[p[1:].astype(np.int64)].sum()) for p in parts])
    check("NEGATIVE: a shifted byte table breaks the per-doc byte identity", (per_bad != nbytes).any())
    # negative control 2: dropping one byte token from the source table breaks the strict decode or the byte count
    table_bad = list(table)
    sp = full.encode(" the", add_special_tokens=False).ids[0]
    table_bad[sp] = b""
    t2, n2, _ = decode_split(small, table_bad)
    check("NEGATIVE: a broken source table changes the decoded bytes", int(n2.sum()) != int(nbytes.sum()))
    # end-to-end on a temp root with 30 train docs
    with tempfile.TemporaryDirectory() as tmp:
        dec, chk = {}, {}
        tr = np.fromfile(os.path.join(src, "train.u32"), dtype=np.uint32, count=200000)
        ttexts, tn, used = decode_split(tr, table, 30)
        dec["train"] = (ttexts, tn, tr[:used])
        dec["val"] = (texts, nbytes, small)
        chk = {"train": {"capped": True}, "val": {"capped": False}}
        meta = build_target("st32", tpath, dec, chk, tmp, 30)
        back = np.fromfile(os.path.join(tmp, "data_st32", "train.u32"), dtype=np.uint32)
        check("built shard: BOS count equals document count", int((back == 0).sum()) == 30)
        check("built shard: meta bytes per token is text bytes / tokens incl BOS",
              abs(meta["splits"]["train"]["bytes_per_token"] - tn.sum() / back.size) < 1e-12)
        check("built shard: token_bytes sum equals text bytes", meta["splits"]["train"]["token_bytes_minus_text_bytes"] == 0)
    n_ok = sum(ok)
    print(f"selftest {n_ok} of {len(ok)}", flush=True)
    return n_ok == len(ok)


def main():
    p = argparse.ArgumentParser(
        description="Re-tokenize the S0 FineWeb-Edu shards with other tokenizers, document by document, into "
                    "data_<name>/ (train.u32, val.u32, token_bytes.i32, val_doc_bytes.i64, tokenizer.json, meta.json).",
        epilog="Examples:\n"
               "  python3 %(prog)s --selftest\n"
               "  python3 %(prog)s --src data --deepseek-as-is v129k --tok v32k=tokenizers/trained_bpe_32768.json\n"
               "  python3 %(prog)s --src data --tok v32k=tokenizers/trained_bpe_32768.json --max-docs 3000 "
               "--out-root /private/tmp/vw   (an M5 smoke)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data")),
                   help="folder with the DeepSeek train.u32, val.u32, token_bytes.i32 (default $SDMLLM_DATA or ./data)")
    p.add_argument("--src-tokenizer", default=None, help="default <src>/ds4_v4flash_tokenizer.json")
    p.add_argument("--tok", action="append", default=[], metavar="NAME=PATH", help="a target tokenizer json (repeatable)")
    p.add_argument("--deepseek-as-is", default=None, metavar="NAME", help="also write data_NAME/ for the source tokenizer")
    p.add_argument("--out-root", default=None, help="where data_<name>/ go (default: the parent of --src)")
    p.add_argument("--max-docs", type=int, default=None, help="cap TRAIN documents (smoke only; val is always complete)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        good = selftest()
        print("NEXT -> python3 track4_sdmonly_vocab_shards.py --src data --tok v32k=tokenizers/trained_bpe_32768.json")
        sys.exit(0 if good else 1)
    if not a.tok and not a.deepseek_as_is:
        p.print_help()
        print("\nNothing to build: give --tok NAME=PATH and/or --deepseek-as-is NAME.")
        sys.exit(2)
    src = os.path.abspath(a.src)
    src_tok = a.src_tokenizer or os.path.join(src, "ds4_v4flash_tokenizer.json")
    out_root = os.path.abspath(a.out_root or os.path.dirname(src))
    todo = []
    for t in a.tok:
        name, path = t.split("=", 1)
        if os.path.exists(os.path.join(out_root, f"data_{name}", "meta.json")):
            print(json.dumps({"event": "skip", "name": name, "why": "meta.json exists"}), flush=True)
            continue
        todo.append((name, path))
    as_is = a.deepseek_as_is if (a.deepseek_as_is and not os.path.exists(
        os.path.join(out_root, f"data_{a.deepseek_as_is}", "meta.json"))) else None
    if a.deepseek_as_is and not as_is:
        print(json.dumps({"event": "skip", "name": a.deepseek_as_is, "why": "meta.json exists"}), flush=True)
    if not todo and not as_is:
        print("all sets already built")
        return
    dec, check = decode_source(src, src_tok, a.max_docs)
    if as_is:
        build_as_is(as_is, src, src_tok, dec, check, out_root)
    for name, path in todo:
        build_target(name, path, dec, check, out_root, a.max_docs)
    print("NEXT -> SDMLLM_DATA=<out-root>/data_<name> python3 track4_sdmonly_train_vocab.py --bytes N --arm sdmonly ...")


if __name__ == "__main__":
    main()
