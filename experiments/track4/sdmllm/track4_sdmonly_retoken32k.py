"""SDMONLY RETOKEN32K: re-encode a DeepSeek V4 Flash token shard (129,280 ids) into our trained 32,768-piece byte-level
BPE, losslessly, keeping document boundaries, loss masks, segment starts and the segment index.

<claudes_code_comments>
** Function List **
ByteTable(entries) - flat byte buffer of a vocabulary; gather(ids) turns an id array into its bytes, vectorised
source_table(path) - true bytes of every DeepSeek id (from the json vocab; BOS 0 bytes; added tokens their content)
bpe_entries(path) - true bytes of every id of a byte-level BPE file (BOS 0 bytes), refusing a non-byte-level entry
token_bytes_rule(path) - the token_bytes.i32 rule: true bytes per id, added tokens their content, BOS/EOS/PAD 0
split_valid(bb) - a byte run as valid utf-8 strings and stray bytes, in order
first_bos_at_or_after(mm, n) - the first BOS index >= n in a shard (len if none)
plan_chunks(mm, lo, hi, chunk) - chunk edges inside [lo, hi), every inner edge on a BOS
process_chunk(job) - decode one chunk to bytes, re-encode run by run, verify, return ids, mask, BOS positions, stats
run(...) - the whole re-encode of one shard: chunks over workers, ordered write, starts, segments, provenance
selftest() - real samples (val docs, guy3 test segments) plus synthetic docs; 22 checks, each independent
main() - CLI

** Technical Review **
- Source format: uint32 little-endian, each document [BOS id 0] + ids, no EOS (S0 train/val/train_big, chat shards,
  guy corpora). Beside a masked shard: <stem>_mask.u8 (1 = a training target, one byte per token, the BOS too),
  <stem>_starts.i64 (positions of segment BOS tokens) and <stem>_segments.jsonl (one record per segment with
  "start" and "tokens"). All present siblings are carried across.
- Decoding uses the json vocab directly (ByteTable.gather), NOT tokenizers' decode: tokenizers 0.20.3 maps the
  DeepSeek added tokens 1 and 2 onto ids 127998 and 127999 in get_vocab()/id_to_token, so its decode of id 127998
  (" bottoms") is wrong. The json vocab is the truth; the S0 vocab wave proved decode then encode returns the
  original ids with this table.
- Runs: a chunk is cut into runs at every BOS and at every change of the mask bit. A BOS run emits id 0 with its
  own mask bit. A text run is its bytes, split into valid utf-8 strings (encoded by the 32k BPE with its added
  tokens removed, so a literal "<|bos|>" in text is ordinary bytes) and stray bytes (a run edge inside a utf-8
  character: each stray byte becomes its single-byte token). Each output token takes its run's mask bit, so no
  token can straddle a mask boundary and the masked byte set is unchanged. An unmasked shard has one run per
  document, which is exactly the vocab wave's per-document encode.
- Every chunk is verified in its worker (on by default): output bytes == source bytes; BOS count and BOS byte
  positions equal; byte mask equal, BOS mask bits equal, every mask change on an output token start; id 0 only at
  BOS. A failure aborts the run. --no-verify exists for the self-test only.
- Workers: multiprocessing spawn pool, imap (ordered), chunks cut on BOS (~--chunk-tokens each). Each worker
  encodes single-threaded (TOKENIZERS_PARALLELISM=false), so --workers N uses N cores. Output order is source order.
- Range: --from-token A starts at the first BOS >= A (0 stays 0); --prefix-tokens N ends at the first BOS >= N
  (the end of the shard if none). Both are absolute source indices, so a prefix run and a --from-token run with the
  same number concatenate (cat) to the full run exactly, since documents are encoded independently.
- token_bytes.i32 rule, learned from data/token_bytes.i32 and its maker (the S0 prepare script): the utf-8 bytes
  each id stands for (byte-level strings through the GPT-2 byte table); an added token counts its content's utf-8
  bytes; the control tokens BOS, EOS and PAD count 0. On the DeepSeek file this reproduces 129,278 of 129,280
  values. The two others (127998 " bottoms", 127999 a CJK piece) hold 27 and 17 in the canonical file: the prepare
  script iterated tokenizers' get_vocab(), which in 0.20.3 places the EOS and PAD strings on those ids (a hash-map
  collision whose winner changes from run to run), so the canonical file took the EOS/PAD string lengths there.
  True lengths are 8 and 6 (110 x 19 = the 2,090-byte over-count on S0 train). This rule reads only the json, so it
  is deterministic; on the 32k file it equals the true byte length of every id, with BOS (id 0) = 0.
- Output: <out>.u32 (+ _mask.u8, _starts.i64, _segments.jsonl when the source has them), and
  <out>_retoken32k.json (provenance). The out folder gets token_bytes.i32 and tokenizer_32k.json once.
Docs: CLAUDE.md lane brief RETOKEN32K (2026-10-07); reuses track4_sdmonly_vocab_shards.py helpers.
</claudes_code_comments>
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track4_sdmonly_vocab_shards as VS  # noqa: E402

SRC_TOK = os.path.join(HERE, "data", "ds4_v4flash_tokenizer.json")
TGT_TOK = os.path.join(HERE, "tokenizers", "trained_bpe_32768.json")
BOS = 0
CHUNK_TOKENS = 1_000_000


# ---------------------------------------------------------------- byte tables
class ByteTable:
    def __init__(self, entries):
        self.len = np.array([len(b) for b in entries], dtype=np.int64)
        self.off = np.zeros(len(entries) + 1, dtype=np.int64)
        np.cumsum(self.len, out=self.off[1:])
        self.buf = np.frombuffer(b"".join(entries), dtype=np.uint8)
        self.V = len(entries)

    def gather(self, ids):
        ids = np.asarray(ids).astype(np.int64)
        L = self.len[ids]
        toff = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(L, out=toff[1:])
        tot = int(toff[-1])
        if tot == 0:
            return np.zeros(0, dtype=np.uint8), toff
        idx = np.repeat(self.off[ids] - toff[:-1], L) + np.arange(tot, dtype=np.int64)
        return self.buf[idx], toff


def source_table(path):
    return VS.source_byte_table(path)


def bpe_entries(path, bos=BOS):
    j = json.load(open(path))
    u2b = VS.byte_level_map()
    vocab = j["model"]["vocab"]
    V = max(vocab.values()) + 1
    if sorted(vocab.values()) != list(range(V)):
        raise SystemExit(f"{path}: vocab ids are not 0..{V - 1} without gaps")
    out = [b""] * V
    for s, i in vocab.items():
        if i == bos:
            continue
        if not all(ch in u2b for ch in s):
            raise SystemExit(f"{path}: id {i} {s!r} is not byte-level")
        out[i] = bytes(u2b[ch] for ch in s)
    return out


CONTROL = {"<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>", "<|bos|>", "<|eos|>", "<|pad|>"}


def token_bytes_rule(path):
    """token_bytes.i32 rule: the utf-8 bytes each id stands for, read from the json vocab (byte-level strings
    through the byte table); an added token counts its content's utf-8 bytes; the control tokens (BOS, EOS, PAD)
    count 0. Deterministic: it never reads tokenizers' get_vocab(), whose collisions vary run to run."""
    j = json.load(open(path))
    u2b = VS.byte_level_map()
    vocab = j["model"]["vocab"]
    V = max([*vocab.values(), *(a["id"] for a in j["added_tokens"])]) + 1
    out = np.zeros(V, dtype=np.int32)
    for s, i in vocab.items():
        out[i] = len(s) if all(ch in u2b for ch in s) else len(s.encode("utf-8"))
    for a in j["added_tokens"]:
        out[a["id"]] = 0 if a["content"] in CONTROL else len(a["content"].encode("utf-8"))
    return out


def split_valid(bb):
    """[(True, str) | (False, bytes)] covering bb in order; False parts are bytes that are not valid utf-8 here."""
    parts = []
    while bb:
        try:
            parts.append((True, bb.decode("utf-8")))
            break
        except UnicodeDecodeError as e:
            if e.start:
                parts.append((True, bb[:e.start].decode("utf-8")))
            parts.append((False, bb[e.start:e.end]))
            bb = bb[e.end:]
    return parts


def sha_file(p):
    return VS.sha(p)


def sha_range(mm, lo, hi, block=1 << 24):
    h = hashlib.sha256()
    for a in range(lo, hi, block):
        h.update(np.ascontiguousarray(mm[a:min(hi, a + block)]).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- chunking
def first_bos_at_or_after(mm, n, block=1 << 22):
    n = max(0, int(n))
    while n < len(mm):
        blk = np.asarray(mm[n:n + block])
        z = np.flatnonzero(blk == BOS)
        if len(z):
            return n + int(z[0])
        n += len(blk)
    return len(mm)


def plan_chunks(mm, lo, hi, chunk):
    edges = [lo]
    t = lo + chunk
    while t < hi:
        c = first_bos_at_or_after(mm, t)
        if c >= hi:
            break
        if c > edges[-1]:
            edges.append(c)
        t = c + chunk
    edges.append(hi)
    return list(zip(edges[:-1], edges[1:]))


# ---------------------------------------------------------------- worker
_W = {}


def _init(src_tok, tgt_tok):
    from tokenizers import Tokenizer
    _W["src"] = ByteTable(source_table(src_tok))
    tent = bpe_entries(tgt_tok)
    _W["tgt"] = ByteTable(tent)
    _W["enc"] = VS.encoder_without_added(tgt_tok)
    u2b = VS.byte_level_map()
    vocab = json.load(open(tgt_tok))["model"]["vocab"]
    _W["byte_id"] = {b: vocab[ch] for ch, b in u2b.items()}
    _W["key"] = (src_tok, tgt_tok)
    _W["added_src"] = np.array(sorted(a["id"] for a in json.load(open(src_tok))["added_tokens"] if a["id"] != BOS),
                               dtype=np.int64)


def process_chunk(job):
    src_path, mask_path, s, e, verify, src_tok, tgt_tok = job
    if _W.get("key") != (src_tok, tgt_tok):
        _init(src_tok, tgt_tok)
    arr = np.array(np.memmap(src_path, dtype=np.uint32, mode="r")[s:e])
    m = np.array(np.memmap(mask_path, dtype=np.uint8, mode="r")[s:e]) if mask_path else None
    flat, toff = _W["src"].gather(arr)
    n = len(arr)
    isb = arr == BOS
    key = m if m is not None else np.zeros(n, dtype=np.uint8)
    brk = np.zeros(n, dtype=bool)
    if n:
        brk[0] = True
        brk[1:] = isb[1:] | isb[:-1] | (key[1:] != key[:-1])
    rs = np.flatnonzero(brk)
    re_ = np.append(rs[1:], n)
    plan, strs, stray = [], [], 0
    for a, b in zip(rs.tolist(), re_.tolist()):
        if isb[a]:
            plan.append(("bos", int(key[a]), None))
            continue
        parts = []
        for ok, x in split_valid(flat[toff[a]:toff[b]].tobytes()):
            if ok:
                parts.append((True, len(strs)))
                strs.append(x)
            else:
                parts.append((False, x))
                stray += len(x)
        plan.append(("text", int(key[a]), parts))
    encs = _W["enc"].encode_batch(strs, add_special_tokens=False) if strs else []
    ids_l, msk_l, bos_out, pos = [], [], [], 0
    bid = _W["byte_id"]
    for kind, bit, parts in plan:
        if kind == "bos":
            bos_out.append(pos)
            ids_l.append(np.array([BOS], dtype=np.uint32))
            msk_l.append(np.array([bit], dtype=np.uint8))
            pos += 1
            continue
        for ok, x in parts:
            t = np.asarray(encs[x].ids, dtype=np.uint32) if ok else np.array([bid[c] for c in x], dtype=np.uint32)
            ids_l.append(t)
            msk_l.append(np.full(len(t), bit, dtype=np.uint8))
            pos += len(t)
    out = np.concatenate(ids_l) if ids_l else np.zeros(0, dtype=np.uint32)
    omask = np.concatenate(msk_l) if msk_l else np.zeros(0, dtype=np.uint8)
    bos_src = np.flatnonzero(isb)
    bos_out = np.asarray(bos_out, dtype=np.int64)
    ver = {}
    if verify:
        oflat, ooff = _W["tgt"].gather(out)
        ver["bytes_equal"] = bool(oflat.size == flat.size and np.array_equal(oflat, flat))
        ver["bos_count_equal"] = bool(len(bos_out) == len(bos_src))
        ver["bos_byte_positions_equal"] = bool(ver["bos_count_equal"] and np.array_equal(toff[bos_src], ooff[bos_out]))
        ver["zero_only_at_bos"] = bool(int((out == BOS).sum()) == len(bos_out))
        if m is not None:
            bsrc = np.repeat(m, _W["src"].len[arr.astype(np.int64)])
            bout = np.repeat(omask, _W["tgt"].len[out.astype(np.int64)])
            ver["masked_bytes_equal"] = bool(np.array_equal(bsrc, bout))
            ver["bos_mask_bits_equal"] = bool(ver["bos_count_equal"] and np.array_equal(m[bos_src], omask[bos_out]))
            chg = np.flatnonzero(bsrc[1:] != bsrc[:-1]) + 1
            starts = ooff[:-1][_W["tgt"].len[out.astype(np.int64)] > 0]
            ver["no_straddle"] = bool(np.isin(chg, starts).all())
        bad = [k for k, v in ver.items() if not v]
        if bad:
            raise RuntimeError(f"chunk [{s}, {e}): verification failed: {bad}")
    added = _W["added_src"]
    nadd = int(np.isin(arr.astype(np.int64), added).sum()) if n else 0
    stats = {"src_tokens": n, "out_tokens": int(len(out)), "text_bytes": int(flat.size), "runs": len(plan),
             "stray_bytes": stray, "src_added_ids": nadd, "bos": int(len(bos_src))}
    return {"s": s, "e": e, "ids": out.tobytes(), "mask": omask.tobytes() if m is not None else None,
            "bos_src": (bos_src + s), "bos_out": bos_out, "stats": stats, "verify": ver}


# ---------------------------------------------------------------- one shard
def siblings(stem):
    return {k: f"{stem}{suf}" for k, suf in (("mask", "_mask.u8"), ("starts", "_starts.i64"),
                                             ("segments", "_segments.jsonl"))}


def write_vocab_files(out_dir, tgt_tok):
    tb_path = os.path.join(out_dir, "token_bytes.i32")
    tb = token_bytes_rule(tgt_tok)
    if os.path.exists(tb_path):
        if not np.array_equal(np.fromfile(tb_path, dtype=np.int32), tb):
            raise SystemExit(f"{tb_path} exists and differs from the 32k table; refusing")
    else:
        tb.tofile(tb_path)
    tk = os.path.join(out_dir, "tokenizer_32k.json")
    if not os.path.exists(tk):
        shutil.copyfile(tgt_tok, tk)
    elif sha_file(tk) != sha_file(tgt_tok):
        raise SystemExit(f"{tk} exists and differs from {tgt_tok}; refusing")
    return tb_path


def run(src, out, workers=1, prefix_tokens=None, from_token=0, chunk_tokens=CHUNK_TOKENS, verify=True,
        src_tok=SRC_TOK, tgt_tok=TGT_TOK, force=False, quiet=False, source_sha=True):
    t0 = time.time()
    if not src.endswith(".u32") or not out.endswith(".u32"):
        raise SystemExit("--src and --out must end in .u32")
    if os.path.abspath(src) == os.path.abspath(out):
        raise SystemExit("--out must differ from --src")
    sstem, ostem = src[:-4], out[:-4]
    sib_s, sib_o = siblings(sstem), siblings(ostem)
    have = {k: os.path.exists(p) for k, p in sib_s.items()}
    targets = [out, f"{ostem}_retoken32k.json"] + [sib_o[k] for k in have if have[k]]
    if not force and any(os.path.exists(p) for p in targets):
        raise SystemExit(f"{out} (or a sibling) exists; pass --force to replace")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    mm = np.memmap(src, dtype=np.uint32, mode="r")
    if have["mask"] and len(np.memmap(sib_s["mask"], dtype=np.uint8, mode="r")) != len(mm):
        raise SystemExit(f"{sib_s['mask']} length differs from {src}")
    lo = 0 if not from_token else first_bos_at_or_after(mm, from_token)
    hi = len(mm) if prefix_tokens is None else first_bos_at_or_after(mm, prefix_tokens)
    if hi < lo:
        raise SystemExit(f"empty range: start {lo} > end {hi}")
    chunks = plan_chunks(mm, lo, hi, chunk_tokens)
    jobs = [(os.path.abspath(src), os.path.abspath(sib_s["mask"]) if have["mask"] else None, a, b, verify,
             os.path.abspath(src_tok), os.path.abspath(tgt_tok)) for a, b in chunks]
    tmp_u, tmp_m = out + ".tmp", (sib_o["mask"] + ".tmp") if have["mask"] else None
    fu = open(tmp_u, "wb")
    fm = open(tmp_m, "wb") if tmp_m else None
    tot = {"src_tokens": 0, "out_tokens": 0, "text_bytes": 0, "runs": 0, "stray_bytes": 0, "src_added_ids": 0, "bos": 0}
    bos_src_all, bos_out_all, pos, nverify = [], [], 0, 0
    if workers <= 1:
        _init(os.path.abspath(src_tok), os.path.abspath(tgt_tok))
        it = map(process_chunk, jobs)
        pool = None
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers, initializer=_init,
                                            initargs=(os.path.abspath(src_tok), os.path.abspath(tgt_tok)))
        it = pool.imap(process_chunk, jobs, chunksize=1)
    try:
        last = time.time()
        for k, r in enumerate(it):
            fu.write(r["ids"])
            if fm:
                fm.write(r["mask"])
            bos_src_all.append(r["bos_src"])
            bos_out_all.append(r["bos_out"] + pos)
            pos += r["stats"]["out_tokens"]
            for kk in tot:
                tot[kk] += r["stats"][kk]
            nverify += bool(r["verify"])
            if not quiet and (time.time() - last > 30 or k == len(jobs) - 1):
                last = time.time()
                el = time.time() - t0
                print(json.dumps({"event": "progress", "chunks": f"{k + 1}/{len(jobs)}", "src_tokens": tot["src_tokens"],
                                  "out_tokens": tot["out_tokens"], "src_tok_per_s": round(tot["src_tokens"] / el)}),
                      flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    fu.close()
    if fm:
        fm.close()
    secs = time.time() - t0
    bos_src_all = np.concatenate(bos_src_all) if bos_src_all else np.zeros(0, np.int64)
    bos_out_all = np.concatenate(bos_out_all) if bos_out_all else np.zeros(0, np.int64)
    os.replace(tmp_u, out)
    if tmp_m:
        os.replace(tmp_m, sib_o["mask"])
    seg_info = None
    if have["starts"]:
        st = np.fromfile(sib_s["starts"], dtype=np.int64)
        st = st[(st >= lo) & (st < hi)]
        idx = np.searchsorted(bos_src_all, st)
        if len(st) and (idx.max() >= len(bos_src_all) or not np.array_equal(bos_src_all[idx], st)):
            raise SystemExit("a _starts.i64 entry is not a BOS position of the source")
        bos_out_all[idx].astype(np.int64).tofile(sib_o["starts"] + ".tmp")
        os.replace(sib_o["starts"] + ".tmp", sib_o["starts"])
    if have["segments"]:
        nxt = np.append(bos_out_all[1:], pos)
        n_seg, over = 0, 0
        maxlen = 0
        with open(sib_s["segments"]) as fi, open(sib_o["segments"] + ".tmp", "w") as fo:
            for line in fi:
                r = json.loads(line)
                if not (lo <= r["start"] < hi):
                    continue
                i = int(np.searchsorted(bos_src_all, r["start"]))
                if i >= len(bos_src_all) or bos_src_all[i] != r["start"]:
                    raise SystemExit(f"segment start {r['start']} is not a BOS position of the source")
                r["tokens_129k"] = r["tokens"]
                r["start_129k"] = r["start"]
                r["start"] = int(bos_out_all[i])
                r["tokens"] = int(nxt[i] - bos_out_all[i] - 1)
                maxlen = max(maxlen, r["tokens"])
                over += r["tokens"] > 2047
                n_seg += 1
                fo.write(json.dumps(r) + "\n")
        os.replace(sib_o["segments"] + ".tmp", sib_o["segments"])
        seg_info = {"segments": n_seg, "max_segment_tokens_32k": maxlen, "segments_over_2047_tokens_32k": over}
    tb_path = write_vocab_files(os.path.dirname(os.path.abspath(out)), tgt_tok)
    nb = tot["text_bytes"]
    prov = {
        "what": "a DeepSeek V4 Flash (129,280) token shard re-encoded into the trained 32,768 byte-level BPE",
        "made_by": "track4_sdmonly_retoken32k.py", "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_path": os.path.abspath(src), "source_sha256": sha_file(src) if source_sha else None,
        "source_file_tokens": int(len(mm)), "source_range": [int(lo), int(hi)],
        "source_range_sha256": sha_range(mm, lo, hi) if source_sha else None,
        "prefix_tokens_requested": prefix_tokens, "from_token_requested": from_token,
        "source_tokens": tot["src_tokens"], "output_path": os.path.abspath(out), "output_tokens": tot["out_tokens"],
        "output_sha256": sha_file(out), "output_bytes_text": nb, "documents_bos": tot["bos"],
        "bytes_per_token_129k": nb / max(1, tot["src_tokens"]), "bytes_per_token_32k": nb / max(1, tot["out_tokens"]),
        "bytes_per_text_token_129k": nb / max(1, tot["src_tokens"] - tot["bos"]),
        "bytes_per_text_token_32k": nb / max(1, tot["out_tokens"] - tot["bos"]),
        "bytes_per_token_rule": "text bytes / tokens, BOS included (the trainer draws windows over the whole shard)",
        "source_tokenizer": os.path.abspath(src_tok), "source_tokenizer_sha256": sha_file(src_tok),
        "tokenizer_file": os.path.abspath(tgt_tok), "tokenizer_sha256": sha_file(tgt_tok),
        "token_bytes_path": tb_path, "token_bytes_sha256": sha_file(tb_path),
        "siblings": {k: (sib_o[k] if have[k] else None) for k in have},
        "siblings_sha256": {k: sha_file(sib_o[k]) for k in have if have[k]},
        "segments": seg_info, "runs": tot["runs"], "stray_bytes_as_single_byte_tokens": tot["stray_bytes"],
        "source_added_token_ids_seen": tot["src_added_ids"],
        "verify": {"on": verify, "chunks_verified": nverify, "chunks": len(jobs),
                   "checks": "bytes equal, BOS count and byte positions, id 0 only at BOS; masked: byte mask, BOS mask "
                             "bits, every mask change on a token start"},
        "workers": workers, "chunk_tokens": chunk_tokens, "seconds": round(secs, 2),
        "src_tokens_per_second": round(tot["src_tokens"] / max(secs, 1e-9))}
    json.dump(prov, open(f"{ostem}_retoken32k.json", "w"), indent=1)
    if not quiet:
        print(json.dumps({"event": "done", "out": out, **{k: prov[k] for k in (
            "source_tokens", "output_tokens", "output_bytes_text", "bytes_per_token_129k", "bytes_per_token_32k",
            "seconds", "src_tokens_per_second")}}), flush=True)
    return prov


# ---------------------------------------------------------------- self-test
def selftest(workers_mp=3):
    """Every check is evaluated on its own: a stage that raises fails its checks and the rest still run."""
    import tempfile
    from tokenizers import Tokenizer
    res = []
    Z = {}

    def check(name, fn, info=None):
        try:
            cond = bool(fn())
            note = info() if (info and cond is not None) else ""
        except (Exception, SystemExit) as e:  # a missing output or a crashed stage is a failed check, not a crashed self-test
            cond, note = False, f"raised {type(e).__name__}: {str(e)[:90]}"
        res.append(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  {note}" if note else ""), flush=True)

    def safe(key, fn):
        try:
            Z[key] = fn()
        except (Exception, SystemExit) as e:
            print(f"  (stage {key} raised {type(e).__name__}: {str(e)[:110]})", flush=True)
            Z[key] = None

    def u32(p):
        return np.fromfile(p, dtype=np.uint32)

    src_tok_full = Tokenizer.from_file(SRC_TOK)
    S_ENT = source_table(SRC_TOK)
    S = ByteTable(S_ENT)
    T = ByteTable(bpe_entries(TGT_TOK))
    hf32 = Tokenizer.from_file(TGT_TOK)
    enc32 = VS.encoder_without_added(TGT_TOK)

    # token_bytes rule
    canon = np.fromfile(os.path.join(HERE, "data", "token_bytes.i32"), dtype=np.int32)
    safe("rule129", lambda: token_bytes_rule(SRC_TOK))
    safe("rule32", lambda: token_bytes_rule(TGT_TOK))
    art = {127998: 27, 127999: 17}  # " bottoms" and a CJK piece; 27 = len(EOS string), 17 = len(PAD string)
    check("token_bytes rule on the 129k tokenizer reproduces data/token_bytes.i32 on every id but the two "
          "get_vocab() collision artefacts, which hold the EOS and PAD string lengths",
          lambda: set(np.flatnonzero(Z["rule129"] != canon).tolist()) == set(art) and
          all(int(canon[i]) == v for i, v in art.items()) and all(int(Z["rule129"][i]) == len(S_ENT[i]) for i in art),
          lambda: f"differ at {np.flatnonzero(Z['rule129'] != canon).tolist()}")
    check("token_bytes rule on the 32k tokenizer equals the true byte length of every id",
          lambda: np.array_equal(Z["rule32"], T.len))

    synth = ["def f(x):\n    return {'k': [x ** 2 for _ in range(3)]}  # code\n\tif x != 0: pass\n",
             "鬼滅の刃 は 日本の漫画です。 Ünïcödé, emoji 🦡🎸 and 𝔘𝔫𝔦 math ∑∫ and ελληνικά.",
             "A literal <|bos|> and <｜begin▁of▁sentence｜> inside text; bottoms up.\r\n  trailing   spaces   "]
    with tempfile.TemporaryDirectory() as tmp:
        # unmasked sample: 300 real val docs + synthetic docs encoded with the 129k tokenizer
        val = np.fromfile(os.path.join(HERE, "data", "val.u32"), dtype=np.uint32)
        vs = np.flatnonzero(val == BOS)
        real = val[:vs[300]]
        syn = [np.array([BOS] + src_tok_full.encode(t, add_special_tokens=False).ids, dtype=np.uint32) for t in synth]
        sample = np.concatenate([syn[0], real[:vs[150]], syn[1], real[vs[150]:], syn[2]])
        p_src = os.path.join(tmp, "u", "web.u32")
        os.makedirs(os.path.dirname(p_src))
        sample.tofile(p_src)
        sb, soff = S.gather(sample)
        sbos = np.flatnonzero(sample == BOS)
        o1 = os.path.join(tmp, "o1", "web.u32")
        safe("o1", lambda: run(p_src, o1, workers=1, chunk_tokens=20000, verify=False, quiet=True))
        check("round trip (unmasked, 303 docs, multi-byte utf-8, code): 32k bytes == 129k bytes",
              lambda: np.array_equal(T.gather(u32(o1))[0], sb), lambda: f"{sb.size} bytes")

        def per_doc():
            out = u32(o1)
            ob = np.flatnonzero(out == BOS)
            od = [hf32.decode(out[x + 1:y].tolist()) for x, y in zip(ob, np.append(ob[1:], len(out)))]
            sd = [sb[soff[x]:soff[y]].tobytes().decode("utf-8") for x, y in zip(sbos, np.append(sbos[1:], len(sample)))]
            return od == sd
        check("round trip by an independent decoder: tokenizers' 32k decode of each doc == source text", per_doc)

        def bounds(src_ids, src_off, out_path):
            out = u32(out_path)
            ooff = T.gather(out)[1]
            sbx, obx = np.flatnonzero(src_ids == BOS), np.flatnonzero(out == BOS)
            return len(sbx) == len(obx) and np.array_equal(src_off[sbx], ooff[obx])
        check("document boundaries (unmasked): BOS count and BOS byte positions identical", lambda: bounds(sample, soff, o1))
        check("a literal '<|bos|>' in text never becomes id 0, and id 0 sits only at document starts",
              lambda: int((u32(o1) == BOS).sum()) == len(sbos))

        # masked sample: the first 200 segments of the real guy3 test shard + two synthetic docs, one with a mask
        # boundary inside a utf-8 character, one with a boundary inside a word the 32k BPE would merge across
        g = os.path.join(HERE, "private", "wlg_v3", "guy3_test")
        gu = np.fromfile(g + ".u32", dtype=np.uint32)
        gm = np.fromfile(g + "_mask.u8", dtype=np.uint8)
        gs = np.fromfile(g + "_starts.i64", dtype=np.int64)
        segs = [json.loads(x) for x in open(g + "_segments.jsonl")][:200]
        cut = int(gs[200])
        ids_x = src_tok_full.encode("User: show me 🦡 now\nGuy: 🦡 here.", add_special_tokens=False).ids
        kx = ids_x.index(102)  # the middle piece of the emoji: the boundary splits the character
        ids_y = src_tok_full.encode("User: that is unbelievably good\nGuy: yes, truly.", add_special_tokens=False).ids
        ky = ids_y.index(src_tok_full.token_to_id("Ġun")) + 1  # boundary after " un", inside " unbelievably"
        docs = [(ids_x, kx), (ids_y, ky)]
        mu = [gu[:cut]]
        mmk = [gm[:cut]]
        extra = []
        pos = cut
        for ids, k in docs:
            mu.append(np.array([BOS] + ids, dtype=np.uint32))
            mmk.append(np.array([1] + [0] * k + [1] * (len(ids) - k), dtype=np.uint8))
            extra.append({"start": pos, "doc": f"synthetic:{len(extra)}", "seg": 0, "of": 1, "tokens": len(ids),
                          "framed": False, "bos_in_mask": 1})
            pos += 1 + len(ids)
        mu, mm_ = np.concatenate(mu), np.concatenate(mmk)
        mdir = os.path.join(tmp, "m")
        os.makedirs(mdir)
        msrc = os.path.join(mdir, "guy.u32")
        mu.tofile(msrc)
        mm_.tofile(os.path.join(mdir, "guy_mask.u8"))
        np.array([*gs[:200], *(r["start"] for r in extra)], dtype=np.int64).tofile(os.path.join(mdir, "guy_starts.i64"))
        with open(os.path.join(mdir, "guy_segments.jsonl"), "w") as f:
            for r in segs + extra:
                f.write(json.dumps(r) + "\n")
        nseg = 200 + len(extra)
        check("precondition: a mask boundary inside a utf-8 character is in the sample",
              lambda: not _valid_utf8(S.gather(np.array(ids_x[:kx], dtype=np.uint32))[0].tobytes()))

        def merged_would_straddle():
            text = S.gather(np.array(ids_y, dtype=np.uint32))[0].tobytes()
            bnd = len(S.gather(np.array(ids_y[:ky], dtype=np.uint32))[0])
            e = enc32.encode(text.decode("utf-8"), add_special_tokens=False).ids
            starts = set(T.gather(np.array(e, dtype=np.uint32))[1].tolist())
            return bnd not in starts
        check("precondition: one boundary falls inside a token the 32k BPE makes from the whole text", merged_would_straddle)
        msb, msoff = S.gather(mu)
        m1 = os.path.join(tmp, "m1", "guy.u32")
        safe("m1", lambda: run(msrc, m1, workers=1, chunk_tokens=50000, verify=False, quiet=True))
        check(f"round trip (masked, {nseg} segments): 32k bytes == 129k bytes",
              lambda: np.array_equal(T.gather(u32(m1))[0], msb), lambda: f"{msb.size} bytes")
        check("document boundaries (masked): BOS count and BOS byte positions identical", lambda: bounds(mu, msoff, m1))
        bsrc = np.repeat(mm_, S.len[mu.astype(np.int64)])

        def mask_bytes():
            ou, om = u32(m1), np.fromfile(m1[:-4] + "_mask.u8", dtype=np.uint8)
            bout = np.repeat(om, T.len[ou.astype(np.int64)])
            return len(om) == len(ou) and np.array_equal(bsrc, bout) and np.array_equal(mm_[mu == BOS], om[ou == BOS])
        check("masks: the set of masked BYTES is identical, and BOS mask bits identical in order", mask_bytes,
              lambda: f"{int(bsrc.sum())} masked bytes")

        def no_straddle():
            ou = u32(m1)
            ooff = T.gather(ou)[1]
            chg = np.flatnonzero(bsrc[1:] != bsrc[:-1]) + 1
            return bool(np.isin(chg, ooff[:-1][T.len[ou.astype(np.int64)] > 0]).all())
        check("masks: no output token straddles a mask boundary", no_straddle,
              lambda: f"{int((bsrc[1:] != bsrc[:-1]).sum())} boundaries")

        def starts_ok():
            ou, nst = u32(m1), np.fromfile(m1[:-4] + "_starts.i64", dtype=np.int64)
            return len(nst) == nseg and bool((ou[nst] == BOS).all()) and int((ou == BOS).sum()) == len(nst)
        check("_starts.i64 carried: same count, every entry is an output BOS, all BOS listed", starts_ok)

        def segs_ok():
            ou, om = u32(m1), np.fromfile(m1[:-4] + "_mask.u8", dtype=np.uint8)
            nst = np.fromfile(m1[:-4] + "_starts.i64", dtype=np.int64)
            nsg = [json.loads(x) for x in open(m1[:-4] + "_segments.jsonl")]
            nxt = np.append(nst[1:], len(ou))
            return len(nsg) == nseg and all(r["start"] == int(x) and r["tokens"] == int(y - x - 1) and
                                            r["bos_in_mask"] == int(om[x]) for r, x, y in zip(nsg, nst, nxt))
        check("_segments.jsonl carried: start == starts, tokens == new segment length, bos_in_mask == mask bit", segs_ok)

        # multiprocess == single process, many chunks
        o3 = os.path.join(tmp, "o3", "web.u32")
        m3 = os.path.join(tmp, "m3", "guy.u32")
        safe("o3", lambda: run(p_src, o3, workers=workers_mp, chunk_tokens=20000, verify=False, quiet=True))
        safe("m3", lambda: run(msrc, m3, workers=workers_mp, chunk_tokens=8000, verify=False, quiet=True))
        check("precondition: the multi-worker runs used several chunks", lambda: Z["o3"]["verify"]["chunks"] >= 10
              and Z["m3"]["verify"]["chunks"] >= 5, lambda: f"{Z['o3']['verify']['chunks']} and {Z['m3']['verify']['chunks']} chunks")
        check(f"multiprocess ({workers_mp} workers) output == single-process output, byte for byte (all files)",
              lambda: all(_cmp(x, y) for x, y in [(o1, o3), (m1, m3)] +
                          [(m1[:-4] + sx, m3[:-4] + sx) for sx in ("_mask.u8", "_starts.i64", "_segments.jsonl")]))

        # prefix and from-token
        N = int(vs[100]) + 37  # mid-document
        op = os.path.join(tmp, "op", "web.u32")
        orr = os.path.join(tmp, "or", "web.u32")
        safe("op", lambda: run(p_src, op, workers=1, prefix_tokens=N, chunk_tokens=20000, verify=False, quiet=True))
        safe("or", lambda: run(p_src, orr, workers=2, from_token=N, chunk_tokens=20000, verify=False, quiet=True))

        def prefix_cut():
            c = Z["op"]["source_range"][1]
            return c > N and sample[c] == BOS and not (sample[N:c] == BOS).any()
        check("prefix stops at the first document boundary at or after N, never mid-document", prefix_cut,
              lambda: f"N={N} cut={Z['op']['source_range'][1]}")

        def prefix_equal():
            out, outp = u32(o1), u32(op)
            k = int((sample[:Z["op"]["source_range"][1]] == BOS).sum())
            return np.array_equal(outp, out[:int(np.flatnonzero(out == BOS)[k])])
        check("prefix output == the full output up to the same document", prefix_equal)
        check("prefix output + from-token output (same N) == full output",
              lambda: np.array_equal(np.concatenate([u32(op), u32(orr)]), u32(o1)))

        # the built-in verifier refuses a corrupted encode
        def refuses():
            _init(os.path.abspath(SRC_TOK), os.path.abspath(TGT_TOK))
            real_enc = _W["enc"]

            class Drop:
                def encode_batch(self, xs, add_special_tokens=False):
                    return [type("E", (), {"ids": e.ids[:-1]})() for e in real_enc.encode_batch(xs, add_special_tokens)]
            _W["enc"] = Drop()
            try:
                process_chunk((p_src, None, 0, int(vs[5]), True, os.path.abspath(SRC_TOK), os.path.abspath(TGT_TOK)))
                return False
            except RuntimeError as e:
                return "bytes_equal" in str(e)
            finally:
                _W.clear()
        check("NEGATIVE: the in-run verifier refuses an encoder that drops a token", refuses)

        def verified_run():
            ov = os.path.join(tmp, "ov", "guy.u32")
            pv = run(msrc, ov, workers=2, chunk_tokens=50000, verify=True, quiet=True)
            return pv["verify"]["chunks_verified"] == pv["verify"]["chunks"] and _cmp(ov, m1)
        check("the in-run verifier passes a correct run, every chunk verified", verified_run)
        check("provenance json: source sha, token counts match the files, bytes per token both sides",
              lambda: Z["m1"]["source_sha256"] == sha_file(msrc) and Z["m1"]["source_tokens"] == len(mu)
              and Z["m1"]["output_tokens"] == len(u32(m1)) and Z["m1"]["output_bytes_text"] == msb.size
              and Z["m1"]["tokenizer_sha256"] == sha_file(TGT_TOK)
              and abs(Z["m1"]["bytes_per_token_32k"] - msb.size / len(u32(m1))) < 1e-12)
        try:
            print(f"sample bytes per token (BOS included): 129k {sb.size / len(sample):.4f}  "
                  f"32k {sb.size / len(u32(o1)):.4f}  ({len(sbos)} docs, {sb.size} bytes)")
        except Exception:
            pass
    n = sum(res)
    print(f"selftest {n} of {len(res)}", flush=True)
    return n == len(res)


def _valid_utf8(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _cmp(a, b):
    return os.path.exists(a) and os.path.exists(b) and open(a, "rb").read() == open(b, "rb").read()


def main():
    p = argparse.ArgumentParser(
        description="Re-encode a DeepSeek V4 Flash (129,280) token shard into the trained 32,768 byte-level BPE, "
                    "losslessly: same bytes, same document boundaries, same masked bytes. Carries _mask.u8, "
                    "_starts.i64 and _segments.jsonl when they sit beside the source.",
        epilog="Examples:\n"
               "  python3 %(prog)s --selftest\n"
               "  python3 %(prog)s --src data/val.u32 --out data32k/val.u32 --workers 8\n"
               "  python3 %(prog)s --src data/train_big.u32 --out data32k/train_big_p200m.u32 "
               "--prefix-tokens 200000000 --workers 16\n"
               "  python3 %(prog)s --src data/train_big.u32 --out data32k/train_big_rest.u32 "
               "--from-token 200000000 --workers 16\n"
               "  cat data32k/train_big_p200m.u32 data32k/train_big_rest.u32 > data32k/train_big.u32",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", help="source .u32 shard (129k ids)")
    p.add_argument("--out", help="output .u32 path; siblings, token_bytes.i32 and tokenizer_32k.json go beside it")
    p.add_argument("--workers", type=int, default=1, help="processes; each encodes single-threaded (default 1)")
    p.add_argument("--prefix-tokens", type=int, default=None, help="end at the first BOS at or after this source index")
    p.add_argument("--from-token", type=int, default=0, help="start at the first BOS at or after this source index")
    p.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS, help="source tokens per work chunk (cut on BOS)")
    p.add_argument("--tokenizer", default=TGT_TOK, help="target BPE json (default tokenizers/trained_bpe_32768.json)")
    p.add_argument("--src-tokenizer", default=SRC_TOK, help="source tokenizer json (default data/ds4_v4flash_tokenizer.json)")
    p.add_argument("--no-verify", action="store_true", help="skip the per-chunk verification (not recommended)")
    p.add_argument("--no-source-sha", action="store_true", help="skip hashing the source (saves a read of a big file)")
    p.add_argument("--force", action="store_true", help="replace existing outputs")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        good = selftest()
        print("NEXT -> python3 track4_sdmonly_retoken32k.py --src data/val.u32 --out data32k/val.u32 --workers 8")
        sys.exit(0 if good else 1)
    if not a.src or not a.out:
        p.print_help()
        print("\nNothing to do: give --src and --out (or --selftest).")
        sys.exit(2)
    run(a.src, a.out, workers=a.workers, prefix_tokens=a.prefix_tokens, from_token=a.from_token,
        chunk_tokens=a.chunk_tokens, verify=not a.no_verify, src_tok=a.src_tokenizer, tgt_tok=a.tokenizer,
        force=a.force, source_sha=not a.no_source_sha)
    print("NEXT -> SDMLLM_DATA=<out folder> python3 track4_sdmllm_train_one_arm.py ...  (token_bytes.i32 is there)")


if __name__ == "__main__":
    main()
