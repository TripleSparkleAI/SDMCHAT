"""SDMONLY generation: load any SDMLLM checkpoint (the SDM-only shape or the old shape) and generate text, one prompt,
one chat reply, or a chat in the terminal, with the site's own sampler ported to Python.

<claudes_code_comments>
** Function List **
Tok(path) - the DeepSeek V4 tokenizer: encode, decode, token_text (the UTF-8 one id stands for), bos/eos/added ids
mulberry32(seed) - the site's seeded random generator, bit for bit (sdmchat.js mulberry32)
sample_next(logits, params, rng, recent, words) - the site's sampleNext: repetition penalty, word penalties,
    no-repeat n-gram, temperature, top-k, top-p, one draw
loop_stats(tok, ids) - the site's loopStats: a token 3-gram 3+ times, or one word 3 times in a row; distinct-2
export_surfaces(out) - run node once over the site files and loopcheck.mjs; write prompts, templates, SAMPLING,
    the word index and parity fixtures to one json (the Spark has no node and no site)
load_surfaces(path) - that json, from --surfaces-json, $SDMONLY_SURFACES, a file beside this script, or node
render_chat(turns, text) - the chat prompt: the training template (fmt of the chat shard builder) + "User: q\nAssistant:"
load_checkpoint(path, device, result_json) - rebuild an SDM-only arm (shapes read from the state dict, extra cfg from
    the result json) or an old-shape arm (track4_sdmllm_models.build), load the weights, eval mode
Stepper(model, T, device) - logits for the last position of the last T ids (the models keep no state between steps);
    .batch(many) does equal-length windows in one forward
_Reply - one reply's state; advance(logits) is loopcheck.mjs gen()'s loop body: nudge, minTokens, stop strings, end
generate(step, tok, prompt, params, ...) - one reply
generate_many(step, tok, prompt, params_list, ...) - several seeds of one prompt in lockstep, one batched forward
resolve_params(a, surfaces) - a preset from SAMPLING, then any flag the caller gave on top
repl(...) - a terminal chat: /reset, /seed N, /temp X, /quit
selftest() - generator parity with node, sampler rules, sampler parity with the site, determinism, template round trip
main() - CLI

** Technical Review **
- Why a port and not a call into the site: the site runs int8 exports in JavaScript; this tool runs any PyTorch
  checkpoint in float32, so a checkpoint can be read and chatted with before it is exported. The sampler, the stop
  rules and the loop measure are line-for-line ports of SETTLE/settle-site/src/engine/sdmchat.js (sampleNext,
  loopStats, mulberry32) and SETTLE/runs/sdmchats/loopcheck.mjs (gen). The selftest checks the port
  against fixtures node computed with the site's own functions (same logits, same seed -> same draws).
- Arithmetic mirrors the JS: the work vector is float32 (a Float32Array), each penalty is computed in float64 and
  stored back as float32, top-k orders by value then by lower index (the JS insertion list), the softmax over the
  candidates is float64. The recent-token context is ids from the LAST PROMPT TOKEN onward (loopcheck's
  ids.slice(n0 - 1)).
- Prompts are never typed here: they come from samples.js and wlg.js through node (export_surfaces), with the poem
  nudge and the OLD settings read out of loopcheck.mjs's source text. The word index (buildWordIndex, with its
  function-word list) is exported as an array so Python uses the exact same word groups.
- The chat template is the chat shards' training template: track4_sdmllm24_prepare_smoltalk_chat_shards.fmt per turn
  ("User: <text>\n", "Assistant: <text stripped>\n"), then "User: <q>\nAssistant:". BOS (id 0) is prepended. A reply
  stops on EOS or any added token (BOS starts the next conversation in the training stream) or on a stop string.
- Checkpoints: {"model","opt","step","cfg","arm"}. For SDM-only arms the store shape (heads, n_sub, d_a, number of
  stores), the dense control width and the readout width are read from the state dict, so a run whose result json
  has no "sdmonly" block (the dense and none arms) still loads. V comes from the embedding.
- Not bitwise the site: the logits differ (float32 PyTorch here, int8 there), so a sample here is the same algorithm
  on the float model, not the same text the page would show.
Docs: PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md (step 3 gate) · SETTLE/runs/sdmchats/loopcheck.mjs
</claudes_code_comments>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SITE = os.environ.get("SDMONLY_SITE", os.path.join(REPO, "SETTLE", "settle-site"))
LOOPCHECK = os.environ.get("SDMONLY_LOOPCHECK", os.path.join(REPO, "SETTLE", "runs", "sdmchats", "loopcheck.mjs"))
DATA = os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data"))
SURFACES_BESIDE = os.path.join(HERE, "track4_sdmonly_surfaces.json")
SDMONLY_ARMS = ("sdmonly", "sdmonly_dense", "sdmonly_none")


# ---------------------------------------------------------------- tokenizer

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class Tok:
    """The DeepSeek V4 tokenizer as the site uses it: bos 0, eos 1, 1,283 added tokens, byte-level BPE."""

    def __init__(self, path=None):
        from tokenizers import Tokenizer
        self.path = path or os.path.join(DATA, "ds4_v4flash_tokenizer.json")
        self.hf = Tokenizer.from_file(self.path)
        # ids come from the json itself: the tokenizers library renumbers the added tokens (BOS becomes 127,997,
        # an id the vocab also uses), while training and the site use the file's ids (BOS 0, EOS 1)
        j = json.load(open(self.path))
        self.added = {int(t["id"]): t["content"] for t in j["added_tokens"]}
        self.vocab = {int(i): s for s, i in j["model"]["vocab"].items()}
        self.V = max(max(self.vocab), max(self.added)) + 1
        self.bos, self.eos = 0, 1
        self.added_ids = set(self.added)
        self._u2b = {u: b for b, u in _bytes_to_unicode().items()}
        self._bytes = {}

    def encode(self, text):
        return self.hf.encode(text, add_special_tokens=False).ids

    def token_bytes(self, i):
        b = self._bytes.get(i)
        if b is None:
            if i in self.added:
                b = self.added[i].encode("utf-8")
            else:
                s = self.vocab.get(int(i), "")
                b = bytes(self._u2b[c] for c in s if c in self._u2b)
            self._bytes[i] = b
        return b

    def token_text(self, i):
        return self.token_bytes(i).decode("utf-8", errors="replace")

    def decode(self, ids):
        return b"".join(self.token_bytes(int(i)) for i in ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- the site's sampler, ported

def mulberry32(seed):
    st = [seed & 0xFFFFFFFF]

    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = st[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return nxt


class Words:
    """buildWordIndex as exported: of[id] = word group or -1; groups[w] = the ids that spell word w."""

    def __init__(self, word_of):
        self.of = np.asarray(word_of, dtype=np.int64)
        order = np.argsort(self.of, kind="stable")
        of_sorted = self.of[order]
        keep = of_sorted >= 0
        order, of_sorted = order[keep], of_sorted[keep]
        n = int(of_sorted.max()) + 1 if len(of_sorted) else 0
        cuts = np.searchsorted(of_sorted, np.arange(n + 1))
        self.groups = [order[cuts[w]:cuts[w + 1]] for w in range(n)]


def _top_k_order(work, k):
    """Indices of the k largest values, largest first, ties to the lower index (the site's topK insertion list)."""
    V = len(work)
    k = min(k, V)
    if k >= V:
        idx = np.arange(V)
    else:
        thr = np.partition(work, V - k)[V - k]
        above = np.nonzero(work > thr)[0]
        eq = np.nonzero(work == thr)[0][: k - len(above)]
        idx = np.concatenate([above, eq])
    return idx[np.lexsort((idx, -work[idx].astype(np.float64)))]


def sample_next(logits, params, rng, recent=(), words=None):
    """sdmchat.js sampleNext. logits: float32 array of V. Returns (id, p_sampled)."""
    work = np.array(logits, dtype=np.float32, copy=True)
    V = len(work)
    recent = list(recent)
    pen = params.get("repetitionPenalty", 1) or 1
    win = params.get("repeatWindow", 64)
    if pen != 1:
        seen = np.array(sorted(set(recent[-win:] if win else recent)), dtype=np.int64)
        if len(seen):
            w = work[seen].astype(np.float64)
            work[seen] = np.where(w > 0, w / pen, w * pen).astype(np.float32)
    wp = params.get("wordPenalty", 0) or 0
    if words is not None and (wp > 0 or params.get("noAdjacentWord")):
        counts = {}
        for i in (recent[-win:] if win else recent):
            g = int(words.of[i])
            if g >= 0:
                counts[g] = counts.get(g, 0) + 1
        if wp > 0:
            for g, c in counts.items():
                ids = words.groups[g]
                work[ids] = (work[ids].astype(np.float64) - wp * c).astype(np.float32)
        last = int(words.of[recent[-1]]) if recent else -1
        if params.get("noAdjacentWord") and last >= 0:
            work[words.groups[last]] = -np.inf
    ng = params.get("noRepeatNgram", 0) or 0
    if ng > 1 and len(recent) >= ng:
        r = recent[-256:] if len(recent) > 256 else recent
        L = len(r)
        tail = r[L - ng + 1:]
        for i in range(0, L - ng + 1):
            if r[i:i + ng - 1] == tail:
                work[r[i + ng - 1]] = -np.inf
    temp = params.get("temperature", 1)
    if temp is None:
        temp = 1
    if temp <= 0:
        return int(np.argmax(work)), 1.0
    k = min(int(params.get("topK", 0) or 0), V)
    top_p = params.get("topP", 1)
    if top_p is None:
        top_p = 1
    if k <= 0:
        mx = float(work.max())
        e = np.exp((work.astype(np.float64) - mx) / temp)
        probs = (e.astype(np.float32) / e.sum()).astype(np.float32)
        if top_p < 1:
            order = _top_k_order(probs, V)
            keep, m = [], 0.0
            for c in order:
                keep.append(c)
                m += float(probs[c])
                if m >= top_p:
                    break
            r_ = rng() * m
            acc = 0.0
            for c in keep:
                acc += float(probs[c])
                if acc >= r_:
                    return int(c), float(probs[c]) / m
            return int(keep[-1]), float(probs[keep[-1]]) / m
        r_ = rng()
        cs = np.cumsum(probs.astype(np.float64))
        i = int(np.searchsorted(cs, r_, side="left"))
        i = min(i, V - 1)
        return i, float(probs[i])
    cand = _top_k_order(work, k)
    v = work[cand].astype(np.float64)
    p = np.exp((v - v.max()) / temp)
    p = p / p.sum()
    keep_n = len(p)
    if top_p < 1:
        m = 0.0
        for i in range(len(p)):
            m += p[i]
            if m >= top_p:
                keep_n = i + 1
                break
    mass = float(sum(p[:keep_n]))
    r_ = rng() * mass
    acc = 0.0
    for i in range(keep_n):
        acc += p[i]
        if acc >= r_:
            return int(cand[i]), float(p[i]) / mass
    return int(cand[keep_n - 1]), float(p[keep_n - 1]) / mass


def _is_word_char(c):
    return c == "'" or unicodedata.category(c)[0] in ("L", "N")


def loop_stats(tok, ids):
    """sdmchat.js loopStats: loops if a token 3-gram occurs 3+ times or one word occurs 3 times in a row."""
    ids = [int(i) for i in ids]
    grams, worst, worst_gram = {}, 0, None
    for i in range(len(ids) - 2):
        g = (ids[i], ids[i + 1], ids[i + 2])
        c = grams.get(g, 0) + 1
        grams[g] = c
        if c > worst:
            worst, worst_gram = c, list(g)
    words, cur = [], []
    for ch in tok.decode(ids).lower():
        if _is_word_char(ch):
            cur.append(ch)
        elif cur:
            words.append("".join(cur))
            cur = []
    if cur:
        words.append("".join(cur))
    run, max_run, run_word = 1, (1 if words else 0), (words[0] if words else None)
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run > max_run:
            max_run, run_word = run, words[i]
    bi = {(ids[i], ids[i + 1]) for i in range(len(ids) - 1)}
    distinct2 = len(bi) / (len(ids) - 1) if len(ids) > 1 else 1
    why = []
    if worst >= 3:
        why.append(f"3-gram {json.dumps(tok.decode(worst_gram), ensure_ascii=False)} x{worst}")
    if max_run >= 3:
        why.append(f'"{run_word}" x{max_run} in a row')
    return {"loops": bool(why), "why": "; ".join(why), "maxGram": worst, "maxRun": max_run, "distinct2": distinct2}


# ---------------------------------------------------------------- the site's prompts and settings, through node

_EXPORT_JS = r"""
import fs from 'node:fs';
import path from 'node:path';
const SITE = process.env.SL_SITE, LC = process.env.SL_LOOPCHECK;
const E = await import(path.join(SITE, 'src/engine/sdmchat.js'));
const S = await import(path.join(SITE, 'src/sdmchat/samples.js'));
const W = await import(path.join(SITE, 'src/wlg/wlg.js'));
const T = new E.Tokenizer(JSON.parse(fs.readFileSync(path.join(SITE, 'public/data/sdmchat/tokenizer.json'), 'utf8')));
const WI = E.buildWordIndex(T);
const src = fs.readFileSync(LC, 'utf8');
const grab = (re, what) => { const m = src.match(re); if (!m) throw new Error('loopcheck.mjs: cannot find ' + what); return new Function('return (' + m[1] + ')')(); };
const OLD = grab(/const OLD = (\{[\s\S]*?\n\});/, 'OLD');
const nudge = grab(/poem:\s*\{[^\n]*nudge:\s*(\{[^}]*\})/, 'the poem nudge');
const oldChatStop = grab(/which === 'old' && sf === 'chat' \? \{ \.\.\.SURFACES\.chat, stop: (\[[^\]]*\]) \}/, 'the old chat stop');
const surfacesSrc = (src.match(/const SURFACES = \{[\s\S]*?\n\};/) || [''])[0];
const turns = [{ role: 'user', text: 'Hello there' }, { role: 'assistant', text: '  Hi! How can I help?  ' }];
// parity fixtures: logits from mulberry32, then the site's own sampleNext and loopStats on them
const V = T.tokens.length;
const g = E.mulberry32(7);
const logits = new Float32Array(V);
for (let i = 0; i < V; i++) logits[i] = (g() - 0.5) * 12;
const recent = [201, 6756, 28, 7492, 344, 270, 12709, 8295, 344, 270];
const draws = {};
for (const [name, params] of Object.entries(E.SAMPLING)) {
  const rng = E.mulberry32(3);
  const out = [];
  const ctx = recent.slice();
  for (let n = 0; n < 12; n++) { const { id } = E.sampleNext(logits, params, rng, ctx, new Float32Array(V), WI); out.push(id); ctx.push(id); }
  draws[name] = out;
}
const loopCases = ['poem poem poem poem about the sea', 'The sky is blue because air scatters short waves of light more than long ones.',
  'the cat sat on the mat and the cat sat on the mat and the cat sat on the mat'];
const loops = loopCases.map((s) => { const ids = T.encode(s); return { text: s, ids, stats: E.loopStats(T, ids) }; });
const r1 = E.mulberry32(1);
const textIds = [0, 1, 2, 201, 271, 127997, 127998, 127999, 128000, 129279];
for (let i = 0; i < 300; i++) textIds.push((i * 431 + 17) % V);
const tokentext = textIds.map((i) => [i, T.tokenText(i)]);
const out = {
  made_by: 'track4_sdmonly_generate.py --export-surfaces (node over the site files)',
  utc: new Date().toISOString(),
  source: { site: SITE, loopcheck: LC, files: ['src/engine/sdmchat.js', 'src/sdmchat/samples.js', 'src/wlg/wlg.js', 'public/data/sdmchat/tokenizer.json'] },
  tokenizer: { bos: T.bos, eos: T.eos, V, added: T.addedIds.size, nl: T.encode('\n'), nn: T.encode('\n\n') },
  sampling: E.SAMPLING,
  old: OLD,
  surfaces: {
    chat: { prompts: S.CHAT_SAMPLES, rendered: S.CHAT_SAMPLES.map((q) => S.chatPrompt([], q)), stop: S.CHAT_STOP, oldStop: oldChatStop },
    poem: { prompts: S.POEM_TITLES, rendered: S.POEM_TITLES.map((q) => S.poemPrompt('title', q, true)), nudge },
    wlg: { prompts: W.SAMPLES, rendered: W.SAMPLES.map((q) => W.wlgPrompt([], q)), stop: W.STOP_STRINGS },
  },
  loopcheck_surfaces_src: surfacesSrc,
  template_check: { turns, text: 'Next question', rendered: S.chatPrompt(turns, 'Next question') },
  word_of: Array.from(WI.of),
  fixtures: { mulberry32_seed1: [r1(), r1(), r1(), r1(), r1()], logits_seed: 7, logits_scale: 12, recent, draws, loops, tokentext },
};
process.stdout.write(JSON.stringify(out));
"""


def export_surfaces(out_path):
    node = shutil.which("node")
    if not node:
        raise SystemExit("export needs node (the site's JS); run it on the M5 and copy the json to the Spark")
    for p in (SITE, LOOPCHECK):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    r = subprocess.run([node, "--input-type=module", "-e", _EXPORT_JS], capture_output=True, text=True,
                       env={**os.environ, "SL_SITE": SITE, "SL_LOOPCHECK": LOOPCHECK}, cwd=SITE)
    if r.returncode != 0:
        raise SystemExit("node export failed:\n" + r.stderr)
    data = json.loads(r.stdout)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


_SURF_CACHE = {}


def load_surfaces(path=None, allow_node=True):
    path = path or os.environ.get("SDMONLY_SURFACES")
    if not path and os.path.exists(SURFACES_BESIDE):
        path = SURFACES_BESIDE
    if path:
        if path not in _SURF_CACHE:
            _SURF_CACHE[path] = json.load(open(path))
        return _SURF_CACHE[path]
    if allow_node and shutil.which("node") and os.path.exists(SITE) and os.path.exists(LOOPCHECK):
        if "node" not in _SURF_CACHE:
            tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"track4_sdmonly_surfaces_{os.getpid()}.json")
            _SURF_CACHE["node"] = export_surfaces(tmp)
        return _SURF_CACHE["node"]
    return None


def render_chat(turns, text):
    """turns: [{"role": "user"|"assistant", "content": str}] -> the chat prompt ending in "Assistant:"."""
    from track4_sdmllm24_prepare_smoltalk_chat_shards import fmt
    return f"{fmt(turns)}User: {text}\nAssistant:"


# ---------------------------------------------------------------- checkpoints and the model step

def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _result_json_for(ck_path, result_json=None, runs_dir=None):
    if result_json:
        return result_json
    run = os.path.basename(os.path.dirname(os.path.abspath(ck_path)))
    for d in [runs_dir, os.environ.get("SDMLLM_RUNS"), os.path.join(HERE, "runs_sdmonly")]:
        if d and os.path.exists(os.path.join(d, f"{run}.result.json")):
            return os.path.join(d, f"{run}.result.json")
    return None


def load_checkpoint(path, device, result_json=None, runs_dir=None):
    """-> (model, info). Works for the SDM-only arms and the old SdmLM / QwenLM arms."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, arm, cfg = ck["model"], ck["arm"], dict(ck["cfg"])
    V = sd["emb.weight"].shape[0]
    info = {"ckpt": os.path.abspath(path), "arm": arm, "step": ck.get("step"), "V": V}
    if arm in SDMONLY_ARMS:
        import track4_sdmonly_models as M
        rj = _result_json_for(path, result_json, runs_dir)
        if rj:
            cfg.update(json.load(open(rj)).get("sdmonly", {}))
            info["result_json"] = os.path.abspath(rj)
        cfg["readout_f"] = sd["readout.up.weight"].shape[0] if "readout.up.weight" in sd else 0
        if "store.banks.0.keys1" in sd:
            heads, n_sub, half = sd["store.banks.0.keys1"].shape
            cfg.update(heads=heads, n_sub=n_sub, d_a=2 * half)
            nb = len({k.split(".")[2] for k in sd if k.startswith("store.banks.")})
            cfg["share_store"] = nb == 1 and cfg.get("hops", 4) > 1
        if "mlps.0.up.weight" in sd:
            cfg["dense_f"] = sd["mlps.0.up.weight"].shape[0]
        model = M.build_sdmonly(arm, V, cfg)
    else:
        from track4_sdmllm_models import build
        if "wq.0.weight" in sd:
            cfg["d_a"] = sd["wq.0.weight"].shape[0]
        model = build(arm, V, cfg)
    model.load_state_dict(sd)
    model.to(device).eval()
    st = getattr(model, "store", None)
    if st is not None and hasattr(st, "collect_stats"):
        st.collect_stats = False
    info["cfg"] = {k: v for k, v in cfg.items() if not isinstance(v, (list, tuple)) or len(v) < 16}
    info["T"] = int(cfg.get("T", 256))
    return model, info


class Stepper:
    """logits of the next token after the last T ids (the SDMLLM models recompute from the window each step)."""

    def __init__(self, model, T, device):
        self.model, self.T, self.device = model, T, device

    @torch.no_grad()
    def __call__(self, ids):
        return self.batch([ids])[0]

    @torch.no_grad()
    def batch(self, many):
        """many: equal-length id lists (one prompt, several seeds, in lockstep) -> (B, V) float32 logits."""
        x = torch.tensor([list(ids[-self.T:]) for ids in many], dtype=torch.long, device=self.device)
        h = self.model.hidden(x)[:, -1]
        return (h @ self.model.emb.weight.t()).float().cpu().numpy()


class _Reply:
    """One reply's state in loopcheck.mjs gen(): ids, its own seeded generator, line counters, text, stop reason."""

    def __init__(self, tok, prompt_ids, params, words, stop_strings, nudge, on_token):
        self.tok, self.params, self.words, self.stops, self.nudge, self.on_token = tok, params, words, stop_strings, nudge, on_token
        self.ids = list(prompt_ids)
        self.n0 = len(self.ids)
        self.rng = mulberry32(int(params.get("seed", 1)) & 0xFFFFFFFF)
        self.nl_id, self.nn_id = tok.encode("\n")[0], tok.encode("\n\n")[0]
        self.added = np.array(sorted(tok.added_ids), dtype=np.int64)
        self.line_tok, self.lines, self.text, self.reason, self.n = 0, 0, "", "max tokens", 0
        self.done = int(params.get("maxTokens", 160)) <= 0

    def advance(self, logits):
        p, tok, n = self.params, self.tok, self.n
        src = logits
        nd = self.nudge
        if nd and nd.get("strength", 0) > 0 and self.line_tok >= nd["lineTokens"]:
            src = logits.copy()
            se = nd.get("stanzaEvery", 0)
            target = self.nn_id if se > 0 and self.lines > 0 and (self.lines + 1) % se == 0 else self.nl_id
            src[target] = np.float32(float(src[target]) + nd["strength"] + 0.5 * (self.line_tok - nd["lineTokens"]))
        if p.get("minTokens") and n < p["minTokens"]:
            if src is logits:
                src = logits.copy()
            src[tok.eos] = -1e9
            src[self.added] = -1e9
        nid, _ = sample_next(src, p, self.rng, self.ids[self.n0 - 1:], self.words)
        self.n += 1
        if nid == tok.eos or nid in tok.added_ids:
            self.reason, self.done = "end", True
            return
        self.ids.append(nid)
        tt = tok.token_text(nid)
        self.text += tt
        if self.on_token:
            self.on_token(tt)
        if "\n" in tt:
            self.lines += tt.count("\n")
            self.line_tok = 0
        else:
            self.line_tok += 1
        hit = next((s for s in self.stops if s in self.text), None)
        if hit:
            self.text = self.text[: self.text.index(hit)]
            self.reason, self.done = "stop", True
        elif self.n >= int(p.get("maxTokens", 160)):
            self.done = True

    def result(self):
        return {"text": self.text, "ids": self.ids[self.n0:], "reason": self.reason}


def generate_many(step, tok, prompt, params_list, words=None, stop_strings=(), nudge=None):
    """Several replies to ONE prompt (one per params, e.g. one per seed) in lockstep: every live reply has the same
    length at every step, so one batched forward serves them all; a finished reply leaves the batch."""
    pid = [tok.bos] + tok.encode(prompt)
    reps = [_Reply(tok, pid, pr, words, stop_strings, nudge, None) for pr in params_list]
    live = [r for r in reps if not r.done]
    while live:
        lg = step.batch([r.ids for r in live])
        for r, row in zip(live, lg):
            r.advance(row)
        live = [r for r in live if not r.done]
    return [r.result() for r in reps]


def generate(step, tok, prompt, params, words=None, stop_strings=(), nudge=None, on_token=None):
    """loopcheck.mjs gen(): BOS + prompt, then sample until EOS / an added token / a stop string / maxTokens."""
    r = _Reply(tok, [tok.bos] + tok.encode(prompt), params, words, stop_strings, nudge, on_token)
    while not r.done:
        r.advance(step.batch([r.ids])[0])
    return r.result()


# ---------------------------------------------------------------- CLI

PARAM_FLAGS = {"temperature": "temperature", "top_k": "topK", "top_p": "topP", "repetition_penalty": "repetitionPenalty",
               "repeat_window": "repeatWindow", "no_repeat_ngram": "noRepeatNgram", "word_penalty": "wordPenalty",
               "max_tokens": "maxTokens", "min_tokens": "minTokens"}


def resolve_params(a, surfaces):
    preset = a.preset or ("chat" if (a.chat or a.repl) else "none")
    base = {}
    if preset != "none":
        if not surfaces:
            raise SystemExit(f"--preset {preset} needs the site's SAMPLING: pass --surfaces-json (made by --export-surfaces)")
        base = dict(surfaces["sampling"][preset])
    else:
        base = {"temperature": 0.8, "topK": 50, "topP": 0.95, "repetitionPenalty": 1.0, "maxTokens": 160}
    for flag, key in PARAM_FLAGS.items():
        v = getattr(a, flag)
        if v is not None:
            base[key] = v
    if a.no_adjacent_word is not None:
        base["noAdjacentWord"] = a.no_adjacent_word
    return preset, base


def stops_for(preset, a, surfaces):
    if a.stop:
        return a.stop
    if (a.chat or a.repl) and surfaces:
        return surfaces["surfaces"]["chat"]["stop"]
    if preset == "wlg" and surfaces:
        return surfaces["surfaces"]["wlg"]["stop"]
    return []


def repl(step, tok, params, words, stops):
    turns = []
    seed = int(params.get("seed", 1))
    print("chat with the checkpoint. /reset clears the turns, /seed N, /temp X, /quit. Empty line quits.")
    while True:
        try:
            q = input("User: ").strip()
        except EOFError:
            break
        if not q or q == "/quit":
            break
        if q == "/reset":
            turns = []
            print("(turns cleared)")
            continue
        if q.startswith("/seed "):
            seed = int(q.split()[1])
            continue
        if q.startswith("/temp "):
            params["temperature"] = float(q.split()[1])
            continue
        print("Assistant:", end="", flush=True)
        t0 = time.time()
        r = generate(step, tok, render_chat(turns, q), {**params, "seed": seed}, words, stops,
                     on_token=lambda s: print(s, end="", flush=True))
        print(f"\n  ({len(r['ids'])} tokens, {r['reason']}, {time.time() - t0:.1f} s)")
        turns += [{"role": "user", "content": q}, {"role": "assistant", "content": r["text"]}]
        seed += 1


def selftest():
    ok, skipped = [], []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    # 1. mulberry32 against values node printed with the site's own function (seed 1 and seed 4,000,000,000)
    r = mulberry32(1)
    check("mulberry32 seed 1 equals the site's", [r() for _ in range(3)] == [0.6270739405881613, 0.002735721180215478, 0.5274470399599522])
    r = mulberry32(4000000000)
    check("mulberry32 seed 4e9 equals the site's", [r(), r()] == [0.6919068221468478, 0.5511944761965424])
    # 2. sampler rules on small vectors
    lg = np.array([0.0, 3.0, 1.0, 2.0, 2.5], dtype=np.float32)
    check("temperature 0 is greedy", sample_next(lg, {"temperature": 0}, mulberry32(1))[0] == 1)
    draws = {sample_next(lg, {"temperature": 1.0, "topK": 2}, mulberry32(s))[0] for s in range(40)}
    check("top-k 2 draws only the two largest", draws <= {1, 4} and len(draws) == 2)
    check("no-repeat 3-gram bans the token that would repeat a 3-gram",
          sample_next(lg, {"temperature": 0, "noRepeatNgram": 3}, mulberry32(1), [1, 2, 1, 7, 1, 2])[0] != 1)
    check("repetition penalty moves greedy off a recent token",
          sample_next(lg, {"temperature": 0, "repetitionPenalty": 2.0}, mulberry32(1), [1])[0] == 4)
    check("ties go to the lower index", list(_top_k_order(np.array([1, 5, 5, 2, 5], dtype=np.float32), 2)) == [1, 2])
    # 3. determinism: a tiny random SDM-only model, same seed twice, then another seed
    import track4_sdmonly_models as M

    class FakeTok:
        bos, eos, added_ids = 0, 1, {0, 1}

        def encode(self, s):
            return [10 + (ord(c) % 400) for c in s]

        def token_text(self, i):
            return "\n" if i == 11 else f"<{i}>"
    torch.manual_seed(0)
    m = M.build_sdmonly("sdmonly", 500, {"d": 32, "d_a": 16, "n_sub": 12, "k": 5, "hops": 2, "n_back": 4, "decays": (0.8, 0.97)})
    for b in m.store.banks:
        torch.nn.init.normal_(b.values, std=0.5)
    m.eval()
    m.store.collect_stats = False
    stp = Stepper(m, 64, torch.device("cpu"))
    pr = {"temperature": 0.9, "topK": 40, "topP": 0.95, "repetitionPenalty": 1.2, "noRepeatNgram": 3, "maxTokens": 30}
    a1 = generate(stp, FakeTok(), "hello", {**pr, "seed": 5})["ids"]
    a2 = generate(stp, FakeTok(), "hello", {**pr, "seed": 5})["ids"]
    a3 = generate(stp, FakeTok(), "hello", {**pr, "seed": 6})["ids"]
    check("generation is deterministic for a seed", a1 == a2 and len(a1) > 0)
    check("another seed gives another sample", a1 != a3)
    many = generate_many(stp, FakeTok(), "hello", [{**pr, "seed": 5}, {**pr, "seed": 6}])
    check("lockstep batch of seeds equals one-at-a-time (CPU)", [m_["ids"] for m_ in many] == [a1, a3])
    # 4. the chat template round trip, against the training template
    turns = [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "  Hi! How can I help?  "}]
    from track4_sdmllm24_prepare_smoltalk_chat_shards import fmt
    p = render_chat(turns, "Next question")
    full = p + " A reply.\n"
    check("prompt + reply + newline equals the training template of the whole conversation",
          full == fmt(turns + [{"role": "user", "content": "Next question"}, {"role": "assistant", "content": "A reply."}])
          and p.endswith("User: Next question\nAssistant:"))
    tok_path = os.path.join(DATA, "ds4_v4flash_tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tok(tok_path)
        check("chat prompt encodes and decodes back to itself", tok.decode(tok.encode(p)) == p)
        check("tokenizer: bos 0 and eos 1 are added tokens, 1,283 added, 129,280 ids",
              "begin" in tok.added.get(0, "") and "end" in tok.added.get(1, "") and len(tok.added_ids) == 1283 and tok.V == 129280)
        check("token text of a few ids", tok.token_text(5) == "#" and tok.token_text(1000) == " в")
    else:
        skipped.append("tokenizer checks (no tokenizer at " + tok_path + ")")
    # 5. parity with the site: template, sampler draws on the same logits, loopStats
    surf = load_surfaces()
    if surf and os.path.exists(tok_path):
        tc = surf["template_check"]
        py = render_chat([{"role": t["role"], "content": t["text"]} for t in tc["turns"]], tc["text"])
        check("chat template equals the site's chatPrompt", py == tc["rendered"])
        check("chat sample prompts equal the site's render",
              all(render_chat([], q) == r_ for q, r_ in zip(surf["surfaces"]["chat"]["prompts"], surf["surfaces"]["chat"]["rendered"])))
        fx = surf["fixtures"]
        check("mulberry32 equals the exported fixture", (lambda g: [g() for _ in range(5)])(mulberry32(1)) == fx["mulberry32_seed1"])
        g = mulberry32(fx["logits_seed"])
        V = surf["tokenizer"]["V"]
        logits = np.array([(g() - 0.5) * fx["logits_scale"] for _ in range(V)], dtype=np.float32)
        words = Words(surf["word_of"])
        for name, params in surf["sampling"].items():
            rng, ctx, out = mulberry32(3), list(fx["recent"]), []
            for _ in range(12):
                i, _p = sample_next(logits, params, rng, ctx, words)
                out.append(i)
                ctx.append(i)
            check(f"sampler draws equal the site's sampleNext ({name}, 12 draws)", out == fx["draws"][name])
        check("token text equals the site's on 310 ids (incl. BOS, EOS, 127,997 to 127,999)",
              all(tok.token_text(i) == t for i, t in fx["tokentext"]))
        for case in fx["loops"]:
            st = loop_stats(tok, case["ids"])
            js = case["stats"]
            check(f"loopStats equals the site's on {case['text'][:28]!r}",
                  st["loops"] == js["loops"] and st["why"] == js["why"] and st["maxGram"] == js["maxGram"]
                  and st["maxRun"] == js["maxRun"] and abs(st["distinct2"] - js["distinct2"]) < 1e-12)
    else:
        skipped.append("site parity checks (no surfaces json and no node+site, or no tokenizer)")
    for s in skipped:
        print("SKIP " + s)
    print(f"{sum(ok)} of {len(ok)} checks pass" + (f", {len(skipped)} groups skipped" if skipped else ""))
    return all(ok)


def parse(argv=None):
    p = argparse.ArgumentParser(
        description="Generate from an SDMLLM checkpoint (SDM-only or old shape) with the site's sampler.",
        epilog="examples:\n"
               "  python3 %(prog)s --ckpt ~/settle24/ck/sdmonly/p0_A_c/last.pt --chat 'Why is the sky blue?' --n 3\n"
               "  python3 %(prog)s --ckpt checkpoints/sdm_s0_20M/last.pt --prompt 'The water cycle' --temperature 0\n"
               "  python3 %(prog)s --ckpt CK --repl\n"
               "  python3 %(prog)s --export-surfaces track4_sdmonly_surfaces.json   (on the M5: node reads the site)\n"
               "  python3 %(prog)s --selftest",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", help="path to <run>/last.pt")
    p.add_argument("--result-json", help="the run's result json (default: SDMLLM_RUNS/<run>.result.json if found)")
    p.add_argument("--runs-dir", help="where <run>.result.json lives")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--prompt", help="raw text after BOS")
    g.add_argument("--chat", help="one user message in the chat template")
    g.add_argument("--repl", action="store_true", help="chat in the terminal")
    p.add_argument("--preset", choices=["chat", "poem", "wlg", "none"], help="SAMPLING from the site (default chat for --chat/--repl)")
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float)
    p.add_argument("--repetition-penalty", type=float)
    p.add_argument("--repeat-window", type=int)
    p.add_argument("--no-repeat-ngram", type=int)
    p.add_argument("--word-penalty", type=float)
    na = p.add_mutually_exclusive_group()
    na.add_argument("--no-adjacent-word", dest="no_adjacent_word", action="store_true", default=None)
    na.add_argument("--adjacent-word-ok", dest="no_adjacent_word", action="store_false")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--min-tokens", type=int)
    p.add_argument("--stop", action="append", help="a stop string (repeatable); default: the chat surface's")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=1, help="samples, seeds seed .. seed+n-1")
    p.add_argument("--device")
    p.add_argument("--tokenizer", help="tokenizer json (default SDMLLM_DATA/ds4_v4flash_tokenizer.json)")
    p.add_argument("--surfaces-json", help="the export of --export-surfaces (prompts, SAMPLING, word index)")
    p.add_argument("--export-surfaces", metavar="OUT", help="write the site export json and stop")
    p.add_argument("--json", action="store_true", help="print one json line per sample")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


def main():
    a = parse()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.export_surfaces:
        d = export_surfaces(a.export_surfaces)
        print(f"wrote {a.export_surfaces}: {len(d['surfaces']['chat']['prompts'])} chat, {len(d['surfaces']['poem']['prompts'])} poem, "
              f"{len(d['surfaces']['wlg']['prompts'])} wlg prompts; SAMPLING {list(d['sampling'])}")
        print("NEXT -> python3 track4_sdmonly_generate.py --selftest --surfaces-json " + a.export_surfaces)
        return
    if not a.ckpt or not (a.prompt is not None or a.chat or a.repl):
        print("need --ckpt and one of --prompt / --chat / --repl (see --help)")
        print("NEXT -> python3 track4_sdmonly_generate.py --help")
        sys.exit(2)
    surfaces = load_surfaces(a.surfaces_json)
    preset, params = resolve_params(a, surfaces)
    words = Words(surfaces["word_of"]) if surfaces else None
    if words is None and (params.get("wordPenalty") or params.get("noAdjacentWord")):
        print("warning: no word index (no surfaces json), so word penalties are off", file=sys.stderr)
    stops = stops_for(preset, a, surfaces)
    nudge = surfaces["surfaces"]["poem"]["nudge"] if (preset == "poem" and surfaces) else None
    device = pick_device(a.device)
    tok = Tok(a.tokenizer)
    model, info = load_checkpoint(a.ckpt, device, a.result_json, a.runs_dir)
    step = Stepper(model, info["T"], device)
    print(f"# {info['arm']} step {info['step']} from {info['ckpt']} on {device}; preset {preset}; params {json.dumps(params)}",
          file=sys.stderr)
    if a.repl:
        repl(step, tok, {**params, "seed": a.seed}, words, stops)
        print("NEXT -> python3 track4_sdmonly_loops.py --ckpt " + a.ckpt)
        return
    prompt = render_chat([], a.chat) if a.chat else a.prompt
    for i in range(a.n):
        t0 = time.time()
        r = generate(step, tok, prompt, {**params, "seed": a.seed + i}, words, stops, nudge)
        st = loop_stats(tok, r["ids"])
        if a.json:
            print(json.dumps({"seed": a.seed + i, "prompt": prompt, **r, "loops": st["loops"], "why": st["why"],
                              "distinct2": st["distinct2"]}, ensure_ascii=False))
        else:
            print(f"--- seed {a.seed + i} · {len(r['ids'])} tokens · {r['reason']} · loops {st['loops']} {st['why']} · "
                  f"distinct-2 {st['distinct2']:.3f} · {time.time() - t0:.1f} s")
            print(prompt + r["text"])
    print("NEXT -> python3 track4_sdmonly_loops.py --ckpt " + a.ckpt, file=sys.stderr)


if __name__ == "__main__":
    main()
