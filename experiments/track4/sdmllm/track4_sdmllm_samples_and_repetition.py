"""SDMLLM S0 free-running generation: the SDM loop run open-ended at the 129,280-token vocabulary, beside the
matched transformer and the order-3 count store, with repetition statistics.

<claudes_code_comments>
** Function List **
load_model(run) - rebuild an arm from its checkpoint
generate(model, prompt_ids, n, temp, topk, seed) - autoregressive sampling; the emitted token is rolled back in
exact3_sampler() - order-3 Dirichlet chain as a sampler (the count-store floor, same prompts)
rep_stats(ids) - distinct-1/2/3, repeated-4-gram fraction, longest repeated span
main() - fixed prompts x {greedy, T=0.8 top-50} x seed 0, samples json + summary

** Technical Review **
- Prompts are fixed strings encoded with the exported DeepSeek V4 tokenizer, prefixed by BOS (id 0) as in
  training. Generation keeps at most the last 256 tokens (the training window).
- Sampling: greedy (temperature 0), and temperature 0.8 with top-50 truncation, torch.Generator seeded 0.
- Repetition statistics are computed on the generated ids only: distinct-n = unique n-grams / n-grams;
  rep4 = fraction of 4-grams that occurred earlier in the same continuation; longest_repeat = the longest
  span that already appeared earlier in the continuation.
- The count-store sampler uses the same masses the floor script tuned on CALIB (read from its json).
</claudes_code_comments>
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from track4_sdmllm_models import build  # noqa: E402

V = 129280
PROMPTS = [
    "The water cycle describes how water moves",
    "In 1776, the thirteen American colonies",
    "Photosynthesis is the process by which plants",
    "To solve a quadratic equation, students first",
    "The most important thing about learning a new language is",
    "Once upon a time, in a small village near the mountains,",
]


def load_model(run, device):
    ck = torch.load(os.path.join(HERE, "checkpoints", run, "last.pt"), map_location=device, weights_only=False)
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
        logits = model(x)[0, -1].float().cpu()
        if temp == 0:
            nxt = int(logits.argmax())
        else:
            v, i = (logits / temp).topk(topk)
            nxt = int(i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)])
        ids.append(nxt)
        out.append(nxt)
    return out


def rep_stats(ids):
    r = {}
    for n in (1, 2, 3):
        grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
        r[f"distinct_{n}"] = round(len(set(grams)) / max(1, len(grams)), 4)
    seen, rep = set(), 0
    grams4 = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    for gm in grams4:
        rep += gm in seen
        seen.add(gm)
    r["rep4"] = round(rep / max(1, len(grams4)), 4)
    longest = 0
    s = ids
    for i in range(len(s)):
        for j in range(i):
            L = 0
            while i + L < len(s) and j + L < i and s[i + L] == s[j + L]:
                L += 1
            longest = max(longest, L)
    r["longest_repeat"] = longest
    return r


class Exact3Sampler:
    def __init__(self, masses):
        from track4_sdmllm_count_store_floors import hkey, jkey, table  # noqa: F401
        self.hkey = hkey
        train = np.fromfile(os.path.join(DATA, "train.u32"), dtype=np.uint32).astype(np.int64)
        self.uni = (np.bincount(train, minlength=V).astype(np.float64) + 1.0) / (len(train) + V)
        self.masses = masses
        # per order: dict context-key -> (next ids, counts), built lazily from sorted arrays
        self.tabs = {}
        for k in range(1, len(masses) + 1):
            pos = np.arange(k, len(train))
            keys = hkey(train, pos, k)
            order = np.argsort(keys, kind="stable")
            self.tabs[k] = (keys[order], train[pos][order])

    def dist(self, ctx):
        p = self.uni.copy()
        stream = np.array(ctx, dtype=np.int64)
        for k, m in enumerate(self.masses, start=1):
            if len(stream) < k:
                break
            q = self.hkey(stream, np.array([len(stream)]), k)[0]
            keys, nxt = self.tabs[k]
            lo, hi = np.searchsorted(keys, q, "left"), np.searchsorted(keys, q, "right")
            c = np.bincount(nxt[lo:hi], minlength=V).astype(np.float64)
            p = (c + m * p) / (c.sum() + m)
        return p

    def generate(self, ids, n, temp, topk, seed):
        rng = np.random.default_rng(seed)
        ids = list(ids)
        out = []
        for _ in range(n):
            p = self.dist(ids)
            if temp == 0:
                nxt = int(p.argmax())
            else:
                lg = np.log(np.maximum(p, 1e-300)) / temp
                top = np.argpartition(-lg, topk)[:topk]
                w = np.exp(lg[top] - lg[top].max())
                nxt = int(top[rng.choice(topk, p=w / w.sum())])
            ids.append(nxt)
            out.append(nxt)
        return out


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, "ds4_v4flash_tokenizer.json"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runs = sys.argv[1:] or ["qwen_s0_20M", "sdm_s0_20M", "sdm_nostore_s0_20M", "sdm_mlp_s0_20M"]
    n_new = 128
    modes = [("greedy", 0.0, 1), ("t0.8_top50", 0.8, 50)]
    samples = {}
    summary = {}
    gens = {}
    for run in runs:
        if not os.path.exists(os.path.join(HERE, "checkpoints", run, "last.pt")):
            print("missing", run)
            continue
        m = load_model(run, device)
        gens[run] = lambda ids, n, t, k, s, m=m: generate(m, ids, n, t, k, s, device)
    floors = os.path.join(HERE, "track4_sdmllm_count_store_floors_result.json")
    if os.path.exists(floors):
        masses = json.load(open(floors))["arms"]["EXACT3"]["masses"]
        ex = Exact3Sampler(masses)
        gens["EXACT3_count_store"] = ex.generate
    for name, fn in gens.items():
        samples[name] = {}
        agg = {}
        for mode, t, k in modes:
            rows = []
            for pr in PROMPTS:
                pid = [0] + tok.encode(pr, add_special_tokens=False).ids
                out = fn(pid, n_new, t, k, 0)
                st = rep_stats(out)
                rows.append({"prompt": pr, "continuation": tok.decode(out), "stats": st})
                for kk, vv in st.items():
                    agg.setdefault(mode, {}).setdefault(kk, []).append(vv)
            samples[name][mode] = rows
        summary[name] = {mode: {kk: round(float(np.mean(vv)), 4) for kk, vv in d.items()} for mode, d in agg.items()}
        print(name, json.dumps(summary[name]), flush=True)
    json.dump({"n_new": n_new, "modes": modes, "prompts": PROMPTS, "summary": summary, "samples": samples},
              open(os.path.join(HERE, "track4_sdmllm_samples_result.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
