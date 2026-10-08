"""SDMLLM chat data (lane SPARK24): smol-smoltalk conversations in the CHATSDM page's own prompt template, tokenized with
the DeepSeek V4 tokenizer, plus a mixed shard that keeps FineWeb-Edu text beside them.

<claudes_code_comments>
** Function List **
fmt(messages) - one conversation as the site's template: "User: ...\nAssistant: ...\n" per turn (system dropped)
encode_rows(tok, rows) - [BOS 0] + ids per conversation, one uint32 array
main() - download train and test parquet at a pinned revision, write chat_train.u32, chat_test.u32, chat_mix.u32

** Technical Review **
- Source: HuggingFaceTB/smol-smoltalk at revision f73fe857 (licence apache-2.0), 4 train parquet files and 1 test
  file. The template matches SETTLE/settle-site/src/sdmchat/Chat.jsx chatPrompt(): each user turn "User: <text>\n",
  each assistant turn "Assistant: <text>\n"; the page then cues "Assistant:" and stops at "\nUser:". A system
  message, when present, is dropped (the page has none).
- chat_test.u32 is the dataset's own test split (never trained on): the chat held-out stream, scored in bits per
  byte with the same windows rule as TEST.
- chat_mix.u32 = all chat_train conversations followed by the same number of FineWeb-Edu tokens taken from the END of
  train_big.u32 (the part a uniform sampler visits as often as any other), so a window drawn uniformly over the mix
  is about half chat and half web text. Provenance json records counts, bytes and sha256s.
</claudes_code_comments>
"""
import hashlib
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data"))
REPO, REV = "HuggingFaceTB/smol-smoltalk", "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
TRAIN = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
TEST = ["data/test-00000-of-00001.parquet"]


def fmt(messages):
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append(f"User: {m['content']}\n")
        elif m["role"] == "assistant":
            out.append(f"Assistant: {m['content'].strip()}\n")
    return "".join(out)


def encode_rows(tok, texts):
    enc = tok.encode_batch(texts, add_special_tokens=False)
    return np.concatenate([np.array([0] + e.ids, dtype=np.uint32) for e in enc])


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, "ds4_v4flash_tokenizer.json"))
    cache = os.path.expanduser("~/settle24/hf")
    prov = {"repo": REPO, "revision": REV, "licence": "apache-2.0",
            "template": "User: <text>\\n / Assistant: <text>\\n per turn, system messages dropped, [BOS 0] per conversation",
            "made_by": "track4_sdmllm24_prepare_smoltalk_chat_shards.py (lane SPARK24)"}
    for name, files in (("chat_test", TEST), ("chat_train", TRAIN)):
        texts = []
        for f in files:
            p = hf_hub_download(REPO, f, repo_type="dataset", revision=REV, cache_dir=cache)
            t = pq.read_table(p, columns=["messages"])
            texts += [fmt(m) for m in t["messages"].to_pylist()]
        parts = [encode_rows(tok, texts[i:i + 20000]) for i in range(0, len(texts), 20000)]
        arr = np.concatenate(parts)
        arr.tofile(os.path.join(DATA, f"{name}.u32"))
        prov[name] = {"conversations": len(texts), "tokens": int(arr.size),
                      "utf8_bytes": int(sum(len(x.encode("utf-8")) for x in texts)),
                      "sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "files": files}
        print(name, prov[name], flush=True)
    chat = np.fromfile(os.path.join(DATA, "chat_train.u32"), dtype=np.uint32)
    web = np.memmap(os.path.join(DATA, "train_big.u32"), dtype=np.uint32, mode="r")
    tail = np.asarray(web[-chat.size:])
    mix = np.concatenate([chat, tail])
    mix.tofile(os.path.join(DATA, "chat_mix.u32"))
    prov["chat_mix"] = {"tokens": int(mix.size), "chat_tokens": int(chat.size),
                        "web_tokens_from_end_of_train_big": int(tail.size), "sha256": hashlib.sha256(mix.tobytes()).hexdigest()}
    prov["made_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(prov, open(os.path.join(DATA, "chat_provenance.json"), "w"), indent=1)
    print("DONE", json.dumps(prov["chat_mix"]))


if __name__ == "__main__":
    main()
