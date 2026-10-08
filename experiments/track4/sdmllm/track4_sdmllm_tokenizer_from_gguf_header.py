"""SDMLLM S0 tokenizer: export the DeepSeek V4 Flash BPE tokenizer from the GGUF header only.

<claudes_code_comments>
** Function List **
read_gguf_kv(path, want) - parse GGUF metadata key/values, stop before tensor info
build_tokenizer(kv, hf_json) - HF tokenizers BPE from GGUF tokens+merges, pre-tokenizer from HF json
compare_with_hf(tok, hf_tok, texts) - vocab/merges identity and encode equality on varied texts
roundtrip(tok, texts) - decode(encode(x)) == x on varied text
main() - export data/ds4_v4flash_tokenizer.json + provenance json, print verdicts

** Technical Review **
- The GGUF file is 86.7 GB; this script reads only the header metadata section (a few MB) by
  streaming the KV table and stopping at the first tensor-info record. The weights are never touched.
- GGUF v3 layout: magic 'GGUF', u32 version, u64 n_tensors, u64 n_kv, then n_kv x (string key,
  u32 type, value). Arrays are (u32 elem_type, u64 n, elems). Strings are (u64 len, bytes).
- GGUF stores tokens, merges, token types and the pre-tokenizer NAME ("joyai-llm"), not its regex.
  The regex lives in llama.cpp's source keyed by that name, and in the publisher's tokenizer.json.
  So the pre-tokenizer, normalizer and decoder are taken from the publisher's tokenizer.json
  (downloaded, revision recorded), and the vocab + merges come from the GGUF. The script then proves
  the two vocabularies and merge lists are identical, and that encodings agree on varied text.
- Round trip: decode(encode(x)) == x on ASCII, code, CJK, emoji, whitespace runs and mixed scripts.
</claudes_code_comments>
"""
import hashlib
import json
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GGUF = "/Users/happyrobot/Code/TripleSparkle/_flows/dwarfstar/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
HF_REPO = "deepseek-ai/DeepSeek-V4-Flash"

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SCALAR = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f",
          T_BOOL: "<?", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}


def read_gguf_kv(path):
    kv = {}
    with open(path, "rb") as f:
        rd = f.read
        assert rd(4) == b"GGUF", "not a GGUF file"
        version = struct.unpack("<I", rd(4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", rd(16))

        def rstr():
            n = struct.unpack("<Q", rd(8))[0]
            return rd(n).decode("utf-8", errors="surrogateescape")

        def rval(t):
            if t == T_STR:
                return rstr()
            if t == T_ARR:
                et = struct.unpack("<I", rd(4))[0]
                n = struct.unpack("<Q", rd(8))[0]
                if et == T_STR:
                    return [rstr() for _ in range(n)]
                fmt = SCALAR[et]
                sz = struct.calcsize(fmt)
                raw = rd(sz * n)
                return list(struct.unpack("<" + fmt[1:] * n, raw))
            fmt = SCALAR[t]
            return struct.unpack(fmt, rd(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            k = rstr()
            t = struct.unpack("<I", rd(4))[0]
            kv[k] = rval(t)
        header_bytes = f.tell()
    return version, n_tensors, n_kv, header_bytes, kv


def sha(b):
    return hashlib.sha256(b).hexdigest()


TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1905, Einstein published four papers; E = mc^2 was one result.",
    "def f(x):\n    return x ** 2  # square it\n\n\tprint(f(3))",
    "    leading spaces, trailing spaces   \n\n\n",
    "日本語のテキストと中文混合，还有한국어도 있습니다。",
    "Emoji: 🙂🚀🔥 and accents: café naïve Zürich São Paulo.",
    "URL https://example.com/a?b=c&d=e#frag and email test@example.com",
    "Numbers 3.14159 2,718,281 0x1F 1e-9 12345678901234567890",
    "Привет, мир! Ελληνικά. العربية. हिन्दी।",
    "<|begin▁of▁sentence|>literal special-looking text",
    "a" * 50 + " " + "ab" * 30,
    "Mixed\r\nline\rendings nbsp​zwsp",
]


def main():
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    version, n_tensors, n_kv, header_bytes, kv = read_gguf_kv(GGUF)
    tok_keys = {k: v for k, v in kv.items() if k.startswith("tokenizer.")}
    tokens = kv["tokenizer.ggml.tokens"]
    merges = kv["tokenizer.ggml.merges"]
    ttypes = kv.get("tokenizer.ggml.token_type")
    print(f"GGUF v{version} tensors {n_tensors} kv {n_kv} header_bytes {header_bytes} read {time.time()-t0:.1f}s")
    print("model:", tok_keys.get("tokenizer.ggml.model"), "pre:", tok_keys.get("tokenizer.ggml.pre"),
          "tokens", len(tokens), "merges", len(merges))
    for k, v in tok_keys.items():
        if not isinstance(v, list):
            print("  ", k, "=", (v[:120] + f"... [{len(v)} chars]") if isinstance(v, str) and len(v) > 120 else v)

    from huggingface_hub import hf_hub_download, HfApi
    rev = HfApi().model_info(HF_REPO).sha
    hf_json_path = hf_hub_download(HF_REPO, "tokenizer.json", revision=rev, local_dir=os.path.join(DATA, "hf_v4flash"))
    hf_cfg_path = hf_hub_download(HF_REPO, "tokenizer_config.json", revision=rev, local_dir=os.path.join(DATA, "hf_v4flash"))
    hf = json.load(open(hf_json_path))
    hf_vocab = hf["model"]["vocab"]
    hf_merges = hf["model"]["merges"]
    hf_merges = [" ".join(m) if isinstance(m, list) else m for m in hf_merges]
    added = hf.get("added_tokens", [])

    # identity checks: GGUF vocab/merges against the publisher's tokenizer.json
    inv = [None] * len(tokens)
    for s, i in hf_vocab.items():
        if i < len(inv):
            inv[i] = s
    for a in added:
        if a["id"] < len(inv) and inv[a["id"]] is None:
            inv[a["id"]] = a["content"]
    n_vocab_eq = sum(1 for i, s in enumerate(tokens) if inv[i] == s)
    n_merge_eq = sum(1 for a, b in zip(merges, hf_merges) if a == b)
    print(f"vocab identical ids: {n_vocab_eq}/{len(tokens)} (hf model vocab {len(hf_vocab)}, added {len(added)})")
    print(f"merges identical in order: {n_merge_eq}/{len(merges)} (hf {len(hf_merges)})")

    # build tokenizer from GGUF content, pre-tokenizer/normalizer/decoder from the publisher json
    out = dict(hf)
    out["model"] = dict(hf["model"])
    gg_vocab = {}
    added_ids = {a["id"] for a in added}
    for i, s in enumerate(tokens):
        if i not in added_ids:
            gg_vocab[s] = i
    out["model"]["vocab"] = gg_vocab
    out["model"]["merges"] = merges
    tok_path = os.path.join(DATA, "ds4_v4flash_tokenizer.json")
    json.dump(out, open(tok_path, "w"), ensure_ascii=False)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    hf_tok = Tokenizer.from_file(hf_json_path)
    rt_ok, enc_eq = 0, 0
    for t in TEXTS:
        ids = tok.encode(t, add_special_tokens=False).ids
        back = tok.decode(ids, skip_special_tokens=False)
        rt_ok += back == t
        enc_eq += ids == hf_tok.encode(t, add_special_tokens=False).ids
        if back != t:
            print("ROUNDTRIP FAIL:", repr(t[:60]), "->", repr(back[:60]))
    # a larger text: this script's own source
    src = open(__file__).read()
    ids = tok.encode(src, add_special_tokens=False).ids
    rt_src = tok.decode(ids) == src
    enc_src = ids == hf_tok.encode(src, add_special_tokens=False).ids
    print(f"roundtrip {rt_ok}/{len(TEXTS)} varied texts; source file {rt_src}; encode==publisher {enc_eq}/{len(TEXTS)}, src {enc_src}")
    print(f"source file: {len(src.encode())} bytes -> {len(ids)} tokens ({len(src.encode())/len(ids):.2f} bytes/token)")
    # negative control: a corrupted merge list must change encodings (the comparison can fail)
    bad = dict(out)
    bad["model"] = dict(out["model"])
    bad["model"]["merges"] = merges[1000:]
    bad_path = os.path.join(DATA, "_negcontrol_tokenizer.json")
    json.dump(bad, open(bad_path, "w"), ensure_ascii=False)
    badtok = Tokenizer.from_file(bad_path)
    neg_diff = sum(badtok.encode(t, add_special_tokens=False).ids != hf_tok.encode(t, add_special_tokens=False).ids for t in TEXTS)
    os.remove(bad_path)
    print(f"negative control (first 1000 merges dropped): encodings differ on {neg_diff}/{len(TEXTS)} texts")

    prov = {
        "gguf_path": GGUF,
        "gguf_version": version, "gguf_n_tensors": n_tensors, "gguf_n_kv": n_kv,
        "gguf_header_bytes_read": header_bytes,
        "weights_loaded": False,
        "tokenizer_scalar_keys": {k: (f"<{len(v)} chars, sha256 {sha(v.encode())[:16]}>" if isinstance(v, str) and len(v) > 200 else v)
                                  for k, v in tok_keys.items() if not isinstance(v, list)},
        "n_tokens": len(tokens), "n_merges": len(merges),
        "sha256_tokens_joined": sha("\n".join(tokens).encode("utf-8", "surrogateescape")),
        "sha256_merges_joined": sha("\n".join(merges).encode("utf-8", "surrogateescape")),
        "publisher_repo": HF_REPO, "publisher_revision": rev,
        "publisher_tokenizer_json_sha256": sha(open(hf_json_path, "rb").read()),
        "pre_tokenizer_source": "publisher tokenizer.json (GGUF stores only the name 'joyai-llm')",
        "vocab_identical_ids": n_vocab_eq, "merges_identical_in_order": n_merge_eq,
        "roundtrip_varied": [rt_ok, len(TEXTS)], "roundtrip_source": rt_src,
        "encode_equal_publisher": [enc_eq, len(TEXTS)], "encode_equal_publisher_source": enc_src,
        "negative_control_encodings_differ": [neg_diff, len(TEXTS)],
        "exported_tokenizer_sha256": sha(open(tok_path, "rb").read()),
        "bos_id": kv.get("tokenizer.ggml.bos_token_id"), "eos_id": kv.get("tokenizer.ggml.eos_token_id"),
        "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(prov, open(os.path.join(HERE, "track4_sdmllm_tokenizer_provenance.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote", tok_path)


if __name__ == "__main__":
    main()
