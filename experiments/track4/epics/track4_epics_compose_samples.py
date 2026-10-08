"""EPICS compose samples: the same cues continued by SW (the base), EP12 (continued on the epics) and FW12 (the
control, continued on FineWeb-Edu), after the numbers are in (lane EPICS).

<claudes_code_comments>
** Function List **
load(path, device) - rebuild an SDMLLM arm from a checkpoint (its own arm and cfg)
generate(model, ids, n, temp, topk, seed, device) - autoregressive sampling over the last 256 tokens
held_out_cues(k) - the first two lines of k held-out epic blocks (never trained on), chosen by a fixed stride
main() - three cues plus one invented opening, x {greedy, T 0.8 top-50 seed 0}, 120 tokens, into a json

** Technical Review **
- The cues come from the held-out TEST blocks of track4_epics_make_token_shards.py (block index % 20 == 7), so the
  continued model has not read them. The invented opening ("Sing, goddess, of the starship ...") tests style
  without a memorised continuation.
- Samples carry no score; the numbers are the bits per byte in the trainer's result json.
</claudes_code_comments>
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SDM = os.path.expanduser("~/settle24/sdmllm")
sys.path.insert(0, SDM)
sys.path.insert(0, HERE)
from track4_sdmllm_models import build  # noqa: E402
from track4_epics_make_token_shards import split_work  # noqa: E402

V = 129280
CK = os.path.expanduser("~/settle24/ck/main")
RUNS = {"BASE (SW)": f"{CK}/sdmchats_SW_d512_s0_300M/last.pt", "EP12 (epics)": f"{CK}/epics_EP12/last.pt",
        "FW12 (control)": f"{CK}/epics_FW12/last.pt"}
INVENTED = "Sing, goddess, of the starship and the crew that sailed beyond the moon,\n"


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["arm"], V, ck["cfg"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def generate(model, ids, n, temp, topk, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = list(ids)
    out = []
    for _ in range(n):
        x = torch.tensor([ids[-256:]], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)[0, -1].float().cpu()
        if temp == 0:
            nxt = int(logits.argmax())
        else:
            v, i = (logits / temp).topk(topk)
            nxt = int(i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)])
        ids.append(nxt)
        out.append(nxt)
    return out


def held_out_cues():
    man = json.load(open(os.path.join(HERE, "EPICS_MANIFEST.json")))
    want = ["paradise-lost", "kalevala-crawford", "song-of-roland"]
    cues = []
    for w in man["works"]:
        if w["slug"] not in want:
            continue
        _tr, te = split_work(open(os.path.join(HERE, w["clean"]["file"]), encoding="utf-8").read())
        block = te[len(te) // 2]
        lines = [ln for ln in block.split("\n") if ln.strip()]
        cues.append({"work": w["slug"], "cue": "\n".join(lines[:2]) + "\n"})
    return cues + [{"work": "invented", "cue": INVENTED}]


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(SDM, "data", "ds4_v4flash_tokenizer.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cues = held_out_cues()
    out = {"cues": cues, "tokens": 120, "settings": ["greedy", "T 0.8 top-50 seed 0"], "samples": {}}
    for name, path in RUNS.items():
        m = load(path, device)
        out["samples"][name] = []
        for c in cues:
            ids = [0] + tok.encode(c["cue"], add_special_tokens=False).ids
            g = tok.decode(generate(m, ids, 120, 0, 0, 0, device))
            s = tok.decode(generate(m, ids, 120, 0.8, 50, 0, device))
            out["samples"][name].append({"work": c["work"], "greedy": g, "sampled": s})
        del m
        torch.cuda.empty_cache() if device.type == "cuda" else None
    json.dump(out, open(os.path.join(HERE, "epics_compose_samples.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote epics_compose_samples.json")


if __name__ == "__main__":
    main()
