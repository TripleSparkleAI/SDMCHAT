"""SDMONLY vocabulary wave: the score that stays comparable across tokenizers (bits per byte, by document).

<claudes_code_comments>
** Function List **
pick_device(cpu) - cuda, then mps, then cpu
doc_bounds(arr) - start and end of each [BOS 0] + ids document in a shard
doc_windows(arr, lo, hi, T) - non-overlapping windows of T inputs tiled over each document, each doc from its own BOS
score_docs(model, arr, docs, T, tb, device, bs, autocast) - nats and table bytes per document, every target once
load_run(run, data, device) - rebuild a finished SDM-only run from its checkpoint and result json
score_run(a, run) - score one run, write per-doc arrays and a "vocab_score" block into its result json
paired(a, b) - paired difference of two runs in bits per byte over the same documents, bootstrap SE over documents
table(runs) - the wave table: V, bytes/token, tokens trained, by-doc bpb, trainer TEST bpb, deltas against the first
selftest() - by-document score == a direct per-window loop on a tiny model (DeepSeek tokenizer); negative controls
main() - CLI

** Technical Review **
- TEST is fixed BY DOCUMENT: val documents 1,000 to 1,999 (the second half by document) of the run's own shard set.
  Every set holds the same 2,000 val documents, re-tokenized, so every model scores exactly the same text.
- Each document is [BOS] + ids. Window j holds inputs d[jT .. jT+T-1] and targets d[jT+1 .. jT+T]; the last window
  is shorter and padded with id 0 (pad targets are masked, and the model is causal, so padding after a position does
  not change it; the selftest checks this against unpadded windows). A target equal to BOS is never scored. So every
  text token of every document is a target exactly once, and the model's context restarts at each window start, as in
  the trainer's own TEST.
- bpb = total nats / (ln 2 x total TEXT bytes), where text bytes are the utf-8 bytes of the documents
  (val_doc_bytes.i64, written by the shard builder from the decoded text, identical across sets). bpb_table uses the
  set's token_bytes.i32 instead (the trainer's definition). For a byte-level BPE set the two denominators are equal to
  the byte (checked; --strict refuses a mismatch). For the DeepSeek set as is, the canonical table differs from the
  text by a few bytes, and the difference is reported.
- The trainer's own window TEST (second half of the val shard by token position, T+1 windows) is copied from the
  result json for reference; it is NOT comparable across tokenizers.
- Per-document nats and bytes go to <ckpt>/<run>/vocab_score_per_doc.npz so runs pair by document (paired()).
- Precision: bf16 autocast on cuda/mps for the model's matmuls (as eval_bpb), fp32 cross-entropy, float64 sums.
Docs: RESEARCH_SDMONLY_VOCAB_AND_SIDE_BY_SIDE_2026-10-03.md §3.1.
</claudes_code_comments>
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track4_sdmonly_models as M  # noqa: E402

LN2 = math.log(2)


def pick_device(cpu=False):
    if cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def doc_bounds(arr):
    starts = np.flatnonzero(arr == 0)
    if len(starts) == 0 or starts[0] != 0:
        raise SystemExit("shard does not start with BOS (id 0)")
    return starts, np.append(starts[1:], len(arr))


def doc_windows(arr, starts, ends, docs, T):
    """List of (doc_index_within_docs, inp ndarray, tgt ndarray) with len(tgt) <= T."""
    out = []
    for k, d in enumerate(docs):
        doc = np.asarray(arr[starts[d]:ends[d]], dtype=np.int64)
        for j in range(0, len(doc) - 1, T):
            out.append((k, doc[j:j + T], doc[j + 1:j + T + 1]))
    return out


@torch.no_grad()
def score_docs(model, arr, docs, T, tb, device, bs=32, autocast=True):
    starts, ends = doc_bounds(arr)
    wins = doc_windows(arr, starts, ends, docs, T)
    nats = np.zeros(len(docs), dtype=np.float64)
    byts = np.zeros(len(docs), dtype=np.int64)
    ntok = np.zeros(len(docs), dtype=np.int64)
    model.eval()
    for i in range(0, len(wins), bs):
        chunk = wins[i:i + bs]
        x = np.zeros((len(chunk), T), dtype=np.int64)
        y = np.zeros((len(chunk), T), dtype=np.int64)
        for r, (_, inp, tgt) in enumerate(chunk):
            x[r, :len(inp)] = inp
            y[r, :len(tgt)] = tgt
        xb, yb = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=autocast and device.type in ("mps", "cuda")):
            lg = model(xb)
        nll = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), yb.reshape(-1), reduction="none").view(yb.shape)
        m = yb != 0
        rn = (nll * m).sum(1).cpu().double().numpy()
        rb = (tb[yb] * m).sum(1).cpu().numpy()
        rt = m.sum(1).cpu().numpy()
        for r, (k, _, _) in enumerate(chunk):
            nats[k] += rn[r]
            byts[k] += rb[r]
            ntok[k] += rt[r]
    return nats, byts, ntok


def ckpt_root():
    return os.environ.get("SDMLLM_CKPT", os.path.join(HERE, "checkpoints"))


def runs_root():
    return os.environ.get("SDMLLM_RUNS", os.path.join(HERE, "runs_sdmonly"))


def load_run(run, data, device):
    res_path = os.path.join(runs_root(), f"{run}.result.json")
    res = json.load(open(res_path))
    ck = torch.load(os.path.join(ckpt_root(), run, "last.pt"), map_location=device)
    meta = json.load(open(os.path.join(data, "meta.json")))
    V = int(meta["V"])
    extra = {k: v for k, v in res.get("sdmonly", {}).items() if k != "store_params"}
    cfg = {**ck["cfg"], **extra}
    arm = ck.get("arm", res.get("arm", "sdmonly"))
    if arm == "qwen":  # the transformer yardstick, built by the S0 model file
        from track4_sdmllm_models import build as build_s0
        model = build_s0("qwen", V, cfg).to(device)
    else:
        model = M.build_sdmonly(arm, V, cfg).to(device)
    if model.emb.weight.shape[0] != ck["model"]["emb.weight"].shape[0]:
        raise SystemExit(f"{run}: checkpoint vocabulary {ck['model']['emb.weight'].shape[0]} != meta V {V} ({data})")
    model.load_state_dict(ck["model"])
    model.eval()
    return model, res, res_path, meta


def score_run(a, run):
    device = pick_device(a.cpu)
    data = os.path.abspath(a.data)
    model, res, res_path, meta = load_run(run, data, device)
    T = a.T or res.get("T", 256)
    tb = torch.from_numpy(np.fromfile(os.path.join(data, "token_bytes.i32"), dtype=np.int32).astype(np.int64)).to(device)
    val = np.fromfile(os.path.join(data, "val.u32"), dtype=np.uint32)
    lo, hi = a.docs
    docs = list(range(lo, hi))
    text_bytes_all = np.fromfile(os.path.join(data, "val_doc_bytes.i64"), dtype=np.int64)
    nats, byts, ntok = score_docs(model, val, docs, T, tb, device, a.bs)
    text_bytes = text_bytes_all[lo:hi]
    diff = int(byts.sum() - text_bytes.sum())
    if a.strict and diff != 0 and not meta.get("as_is"):
        raise SystemExit(f"{run}: table bytes differ from text bytes by {diff}; refusing (--no-strict to score anyway)")
    out = {"docs": [lo, hi], "T": T, "V": int(meta["V"]), "set": meta["name"], "tokens_scored": int(ntok.sum()),
           "text_bytes": int(text_bytes.sum()), "table_bytes": int(byts.sum()), "table_minus_text_bytes": diff,
           "nats": float(nats.sum()), "bpb": float(nats.sum() / (LN2 * text_bytes.sum())),
           "bpb_table": float(nats.sum() / (LN2 * byts.sum())),
           "bytes_per_scored_token": float(text_bytes.sum() / ntok.sum()),
           "trainer_window_test_bpb": res.get("val_test", {}).get("bpb"), "device": str(device),
           "rule": "val docs [lo, hi) by document, each from its own BOS, windows of T tiled, every text byte once"}
    np.savez(os.path.join(ckpt_root(), run, "vocab_score_per_doc.npz"), nats=nats, text_bytes=text_bytes,
             table_bytes=byts, tokens=ntok, docs=np.arange(lo, hi))
    res["vocab_score"] = out
    json.dump(res, open(res_path + ".tmp", "w"), indent=1)
    os.replace(res_path + ".tmp", res_path)
    print("VOCAB_SCORE", run, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()
                                          if k != "rule"}), flush=True)
    return out


def load_per_doc(run):
    z = np.load(os.path.join(ckpt_root(), run, "vocab_score_per_doc.npz"))
    return z["nats"], z["text_bytes"], z["docs"]


def paired(ra, rb, n_boot=2000, seed=0):
    na, ba, da = load_per_doc(ra)
    nb, bb, db = load_per_doc(rb)
    if not (np.array_equal(da, db) and np.array_equal(ba, bb)):
        raise SystemExit(f"{ra} and {rb} were not scored on the same documents and bytes")
    delta = (na.sum() - nb.sum()) / (LN2 * ba.sum())
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(na), size=(n_boot, len(na)))
    boots = (na[idx].sum(1) - nb[idx].sum(1)) / (LN2 * ba[idx].sum(1))
    se = float(boots.std(ddof=1))
    return float(delta), se


def table(runs):
    rows = []
    for r in runs:
        res = json.load(open(os.path.join(runs_root(), f"{r}.result.json")))
        vs, vw = res.get("vocab_score"), res.get("vocab_wave", {})
        if vs is None:
            rows.append(f"{r:<12} NOT SCORED")
            continue
        line = (f"{r:<12} V {vs['V']:>7,}  B/tok train {vw.get('bytes_per_token_train', float('nan')):.4f}  "
                f"tokens {vw.get('tokens_trained', 0):>11,}  bytes {vw.get('bytes_trained', 0):>14,.0f}  "
                f"bpb(doc) {vs['bpb']:.5f}  bpb(table) {vs['bpb_table']:.5f}  trainerTEST {vs['trainer_window_test_bpb']:.5f}")
        if r != runs[0] and runs[0] in [x for x in runs]:
            try:
                d, se = paired(r, runs[0])
                line += f"  vs {runs[0]}: {d:+.5f} (SE {se:.5f}, z {d / se if se else float('nan'):+.1f})"
            except (SystemExit, FileNotFoundError) as e:
                line += f"  vs {runs[0]}: n/a ({e})"
        rows.append(line)
    head = ("THE TOKENIZER WAVE · sdmonly centre (d 256, 4 hops, n_sub 256, seed 0) · equal training bytes\n"
            "bpb(doc) = nats / (ln2 x text bytes) over val docs 1,000-1,999 by document (comparable across tokenizers)\n"
            "trainerTEST = the trainer's window TEST by token offset (NOT comparable across tokenizers)\n"
            "delta = paired over the same documents (run minus first), SE by bootstrap over documents (2,000)\n")
    return head + "\n".join(rows)


def selftest():
    ok = []

    def check(name, cond, info=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)

    import track4_sdmonly_vocab_shards as VS
    torch.manual_seed(0)
    data = os.path.join(HERE, "data")
    src_tok = os.path.join(data, "ds4_v4flash_tokenizer.json")
    V = 129280
    val = np.fromfile(os.path.join(data, "val.u32"), dtype=np.uint32)
    s, e = doc_bounds(val)
    small = val[:e[11]]
    docs = list(range(4, 12))
    model = M.SdmOnlyLM(V, d=16, d_a=16, n_sub=8, k=4, hops=2, n_back=4, decays=(0.8, 0.97))
    # give the batch norms non-trivial running stats, then freeze them
    model.train()
    with torch.no_grad():
        for _ in range(3):
            model(torch.from_numpy(small[:4 * 64].astype(np.int64).reshape(4, 64)))
    model.eval()
    tb_np = np.fromfile(os.path.join(data, "token_bytes.i32"), dtype=np.int32).astype(np.int64)
    tb = torch.from_numpy(tb_np)
    dev = torch.device("cpu")
    T = 64
    nats, byts, ntok = score_docs(model, small, docs, T, tb, dev, bs=5, autocast=False)
    # direct computation: one unpadded window at a time, a plain python sum
    dn, db = 0.0, 0
    with torch.no_grad():
        for d in docs:
            doc = small[s[d]:e[d]].astype(np.int64)
            for j in range(0, len(doc) - 1, T):
                inp = torch.from_numpy(doc[j:j + T])[None]
                tgt = doc[j + 1:j + T + 1]
                lg = model(inp)[0, :len(tgt)]
                lp = torch.log_softmax(lg.float(), -1)
                for p, t in enumerate(tgt):
                    if t != 0:
                        dn -= float(lp[p, t])
                        db += int(tb_np[t])
    bpb_b = nats.sum() / (LN2 * byts.sum())
    bpb_d = dn / (LN2 * db)
    check("by-document score == direct per-window computation (padded batch vs unpadded loop)",
          abs(bpb_b - bpb_d) < 1e-6 * bpb_d, f"{bpb_b:.8f} vs {bpb_d:.8f}")
    check("every text token is scored exactly once", int(ntok.sum()) == sum(int(e[d] - s[d] - 1) for d in docs))
    table = VS.source_byte_table(src_tok)
    texts, nb, _ = VS.decode_split(small, table)
    check("denominator == utf-8 bytes of the documents' text (DeepSeek, these docs)",
          int(np.asarray(nb)[docs].sum()) == int(byts.sum()), f"{int(byts.sum())} vs {int(np.asarray(nb)[docs].sum())}")
    tb_bad = torch.roll(tb, 1)
    nats2, byts2, _ = score_docs(model, small, docs, T, tb_bad, dev, bs=5, autocast=False)
    check("NEGATIVE: a shifted byte table changes the score", abs(nats2.sum() / (LN2 * byts2.sum()) - bpb_b) > 1e-4,
          f"{nats2.sum() / (LN2 * byts2.sum()):.5f} vs {bpb_b:.5f}")
    nats3, _, _ = score_docs(model, small, [d + 1 for d in docs[:-1]] + [3], T, tb, dev, bs=5, autocast=False)
    check("NEGATIVE: scoring other documents changes the nats", abs(nats3.sum() - nats.sum()) > 1e-3)
    nats4, _, _ = score_docs(model, small, docs, 32, tb, dev, bs=5, autocast=False)
    check("a shorter window loses context (score moves), so T is part of the protocol",
          abs(nats4.sum() - nats.sum()) > 1e-6)
    n_ok = sum(ok)
    print(f"selftest {n_ok} of {len(ok)}", flush=True)
    return n_ok == len(ok)


def main():
    p = argparse.ArgumentParser(
        description="Score finished SDM-only runs in bits per byte BY DOCUMENT (val docs 1,000-1,999), comparable "
                    "across tokenizers; or print the wave table.",
        epilog="Examples:\n"
               "  python3 %(prog)s --selftest\n"
               "  SDMLLM_RUNS=/root/settle/runs SDMLLM_CKPT=/root/settle/ck python %(prog)s --run vw_v32k --data data_v32k\n"
               "  python %(prog)s --table vw_v129k,vw_v65k,vw_v32k,vw_v16k",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], help="a finished run name (repeatable)")
    p.add_argument("--data", default=os.environ.get("SDMLLM_DATA"), help="the run's data_<name>/ folder (meta.json, val.u32)")
    p.add_argument("--docs", default="1000:2000", help="val documents [lo:hi) (default 1000:2000)")
    p.add_argument("--T", type=int, default=None, help="window length (default: the run's T)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="score even if table bytes differ from text bytes on a re-tokenized set")
    p.add_argument("--table", default=None, help="comma list of scored runs: print the wave table (first = reference)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    a.docs = tuple(int(x) for x in a.docs.split(":"))
    if a.selftest:
        good = selftest()
        print("NEXT -> python3 track4_sdmonly_vocab_score.py --run <run> --data data_<name>")
        sys.exit(0 if good else 1)
    if a.table:
        print(table(a.table.split(",")))
        return
    if not a.run:
        p.print_help()
        print("\nNothing to do: give --run NAME (with --data) or --table A,B,...")
        sys.exit(2)
    if not a.data:
        raise SystemExit("give --data data_<name> (or set SDMLLM_DATA)")
    for r in a.run:
        score_run(a, r)
    print("NEXT -> python3 track4_sdmonly_vocab_score.py --table " + ",".join(a.run))


if __name__ == "__main__":
    main()
