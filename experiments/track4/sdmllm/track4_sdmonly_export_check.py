"""SDMONLY export check: PyTorch float32 against the JavaScript engine reading the exported file.

<claudes_code_comments>
** Function List **
torch_reference(model, val, tb, starts, T, out_dir, n_samples) - PyTorch fp32 logits per position -> reference files
perturb_copy(src, dst, tensor) - a copy of an export with one tensor negated (the negative control)
run_node(ref_json, model_file, node, limit) - run the engine's --parity and return its JSON line
check_pipeline(model, val, tb, out_dir, ...) - export fp32 and int8, reference, parity for each, the control
selftest() - the whole pipeline on a tiny random model and random tokens, end to end through node
main() - CLI on a real checkpoint and the held-out TEST windows of val.u32

** Technical Review **
- WHAT IS COMPARED. PyTorch runs the checkpoint in float32 on the CPU (no autocast) over N TEST windows of
  val.u32 (S0's val_windows: the second half, non-overlapping 257-token slices) and writes, for every position:
  the argmax, the log-sum-exp, the target's negative log-likelihood, the logits at S fixed vocabulary ids (seeded,
  without replacement), and the target's byte count. The JS engine (track4_sdmonly_engine.mjs --parity) steps each
  window token by token from a fresh rolling state and computes the full logits; it reports top-1 agreement, the
  mean absolute logit difference over the S sampled ids (an unbiased estimate of the mean over all V), the largest
  sampled difference, and bits per byte both ways. Positions whose target is BOS (id 0) are skipped, as in eval_bpb.
- THE THREE FILES. fp32 (every tensor float32: must agree almost exactly, so a difference is an engine bug), int8
  (the browser file: the gap is quantisation error), and the int8 file with one tensor negated (default the hop-0
  value table): its agreement must drop, which shows the comparison can fail and that the engine reads that tensor.
- The verdict thresholds: fp32 top-1 at least 0.995 and mean abs logit difference under 1e-3; the control's top-1
  at least 0.01 below the int8 file's, or its mean difference at least 3 times the int8 file's.
- NEAR-TIES. A location whose score sits within about 1e-5 of the k-th or the 2k-th score can swap in or out under
  float rounding, and the change carries through the later hops. Measured on p0_A_c (fp32 file, 4 windows, numpy
  reference against PyTorch): median per-position max difference 2.5e-5; 17 of 1,024 positions over 1e-2, and the
  largest of them all sat at a boundary gap of 6e-6 or less. So the fp32 test is top-1 and the mean, not the maximum.
- Writes only into --out (default a scratch folder under the system temp dir).
Docs: track4_sdmonly_export.py (the format) · track4_sdmonly_engine.mjs (the engine)
</claudes_code_comments>
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track4_sdmonly_export as X  # noqa: E402

ENGINE = os.path.join(HERE, "track4_sdmonly_engine.mjs")


def val_windows(n, T, max_windows):
    starts = np.arange(n // 2, n - (T + 1), T + 1)
    return starts[:max_windows]


def torch_reference(model, val, tb, starts, T, out_dir, n_samples=4096, seed=0):
    import torch
    import torch.nn.functional as F
    V = model.V
    S = min(n_samples, V)
    sample_ids = np.sort(np.random.default_rng(seed).choice(V, S, replace=False)).astype(np.int64)
    N = len(starts)
    toks = np.stack([np.asarray(val[s:s + T + 1]) for s in starts]).astype(np.uint32)
    top1 = np.zeros((N, T), np.int32)
    lse = np.zeros((N, T), np.float32)
    nll = np.zeros((N, T), np.float32)
    samp = np.zeros((N, T, S), np.float32)
    nb = tb[toks[:, 1:].astype(np.int64)].astype(np.int32)
    model.eval()
    with torch.no_grad():
        for w in range(N):
            x = torch.from_numpy(toks[w:w + 1, :T].astype(np.int64))
            lg = model(x)[0].float()
            tgt = torch.from_numpy(toks[w, 1:].astype(np.int64))
            ls = torch.logsumexp(lg, -1)
            top1[w] = lg.argmax(-1).numpy()
            lse[w] = ls.numpy()
            nll[w] = (ls - lg.gather(1, tgt[:, None])[:, 0]).numpy()
            samp[w] = lg[:, torch.from_numpy(sample_ids)].numpy()
    files = {"tokens": "ref_tokens.u32", "top1": "ref_top1.i32", "lse": "ref_lse.f32", "nll": "ref_nll.f32",
             "sampled": "ref_sampled.f32", "bytes": "ref_bytes.i32"}
    for key, arr in zip(files, (toks, top1, lse, nll, samp, nb)):
        arr.tofile(os.path.join(out_dir, files[key]))
    mask = toks[:, 1:] != 0
    ref = {"n_windows": N, "T": T, "V": V, "sample_ids": sample_ids.tolist(), "files": files,
           "starts": [int(s) for s in starts], "bpb_pytorch_fp32": float(nll[mask].sum() / (math.log(2) * nb[mask].sum())),
           "positions": int(mask.sum())}
    path = os.path.join(out_dir, "ref.json")
    json.dump(ref, open(path, "w"))
    return path, ref


def perturb_copy(src, dst, tensor="store.0.values"):
    """Copy an export with `tensor` negated (q values for a quantised tensor, floats for f32)."""
    header, _ = X.read_sdmo(src, dequant=False)
    raw = bytearray(open(src, "rb").read())
    base = header["blob_offset"]
    hit = [e for e in header["tensors"] if e["name"] in (tensor, tensor + ".q")]
    if not hit:
        raise SystemExit(f"{src}: no tensor {tensor}")
    e = hit[0]
    lo, hi = base + e["offset"], base + e["offset"] + e["bytes"]
    if e["dtype"] == "f32":
        raw[lo:hi] = (-np.frombuffer(bytes(raw[lo:hi]), np.float32)).astype(np.float32).tobytes()
    elif e["dtype"] == "i8":
        raw[lo:hi] = (-np.frombuffer(bytes(raw[lo:hi]), np.int8).astype(np.int16)).astype(np.int8).tobytes()
    else:  # i4: nibble n holds q = n - 8; -q is nibble 16 - n (n is 1..15)
        b = np.frombuffer(bytes(raw[lo:hi]), np.uint8)
        lo4, hi4 = 16 - (b & 15).astype(np.int16), 16 - (b >> 4).astype(np.int16)
        raw[lo:hi] = ((lo4 & 15) | ((hi4 & 15) << 4)).astype(np.uint8).tobytes()
    open(dst, "wb").write(raw)
    return e["name"]


def run_node(ref_json, model_file, node="node", limit=0):
    r = subprocess.run([node, ENGINE, "--parity", ref_json, model_file, str(limit)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"node parity failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def check_pipeline(model, val, tb, out_dir, windows=8, T=256, n_samples=4096, node="node", values_bits=8,
                   matrices="f32", perturb="store.0.values", meta_extra=None, log=print, modes=("fp32", "int8")):
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for mode in modes:
        p = os.path.join(out_dir, f"model.{mode}.sdmo")
        t0 = time.time()
        X.write_sdmo(model, p, mode=mode, values_bits=values_bits, matrices=matrices, T=T, meta_extra=meta_extra)
        files[mode] = p
        log(f"export {mode}: {os.path.getsize(p) / 1e6:.2f} MB ({time.time() - t0:.1f} s)")
    starts = val_windows(len(val), T, windows)
    t0 = time.time()
    ref_json, ref = torch_reference(model, val, tb, starts, T, out_dir, n_samples)
    log(f"PyTorch fp32 reference: {ref['n_windows']} windows, {ref['positions']} scored positions, "
        f"bpb {ref['bpb_pytorch_fp32']:.5f} ({time.time() - t0:.1f} s)")
    out = {"windows": ref["n_windows"], "positions": ref["positions"], "bpb_pytorch_fp32": ref["bpb_pytorch_fp32"],
           "file_bytes": {m: os.path.getsize(p) for m, p in files.items()}}
    for mode, p in files.items():
        out[mode] = run_node(ref_json, p, node)
        log(f"JS {mode}: " + json.dumps({k: out[mode][k] for k in ("top1_agreement", "mean_abs_logit_diff",
                                                                    "max_abs_logit_diff_sampled", "bpb_js", "js_tokens_per_s")}))
    pert = os.path.join(out_dir, "model.int8.perturbed.sdmo")
    name = perturb_copy(files["int8"], pert, perturb)
    out["control"] = run_node(ref_json, pert, node)
    out["control"]["perturbed_tensor"] = name + " (negated)"
    log(f"JS control ({name} negated): " + json.dumps({k: out["control"][k] for k in ("top1_agreement", "mean_abs_logit_diff", "bpb_js")}))
    os.remove(pert)
    f, i, c = out.get("fp32"), out["int8"], out["control"]
    out["verdict"] = {
        "fp32_exact": None if f is None else (f["top1_agreement"] >= 0.995 and f["mean_abs_logit_diff"] < 1e-3),
        "control_drops": (c["top1_agreement"] <= i["top1_agreement"] - 0.01) or (c["mean_abs_logit_diff"] >= 3 * i["mean_abs_logit_diff"]),
    }
    json.dump(out, open(os.path.join(out_dir, "export_check.json"), "w"), indent=1)
    return out


def selftest():
    import torch
    import track4_sdmonly_models as M
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))

    torch.manual_seed(0)
    V, T = 400, 32
    m = M.SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=2, n_back=4, decays=(0.6, 0.9, 0.99), readout_f=24)
    for b in m.store.banks:
        torch.nn.init.normal_(b.values, std=0.5)
    torch.nn.init.normal_(m.emb.weight, std=0.5)
    m.train()
    with torch.no_grad():
        for _ in range(10):
            m(torch.randint(1, V, (4, T)))
    m.eval()
    rng = np.random.default_rng(1)
    val = rng.integers(0, V, 40 * (T + 1)).astype(np.uint32)
    tb = rng.integers(1, 6, V).astype(np.int32)
    out_dir = tempfile.mkdtemp(prefix="sdmo_check_selftest_")
    r = check_pipeline(m, val, tb, out_dir, windows=6, T=T, n_samples=128, matrices="i8", log=lambda s: None)
    check("fp32 file: JS top-1 equals PyTorch almost everywhere", r["fp32"]["top1_agreement"] >= 0.995,
          f"{r['fp32']['top1_agreement']:.4f}")
    check("fp32 file: mean abs logit difference under 1e-4", r["fp32"]["mean_abs_logit_diff"] < 1e-4,
          f"{r['fp32']['mean_abs_logit_diff']:.2e}")
    check("fp32 file: JS bpb equals the PyTorch bpb", abs(r["fp32"]["bpb_js"] - r["bpb_pytorch_fp32"]) < 1e-4,
          f"{r['fp32']['bpb_js']:.6f} vs {r['bpb_pytorch_fp32']:.6f}")
    check("int8 file: agreement is measured and below fp32's or equal", r["int8"]["top1_agreement"] <= r["fp32"]["top1_agreement"] + 1e-9,
          f"{r['int8']['top1_agreement']:.4f}")
    check("negative control: negating a value table drops agreement", r["verdict"]["control_drops"],
          f"top-1 {r['control']['top1_agreement']:.4f}, mean abs {r['control']['mean_abs_logit_diff']:.3f}")
    check("the verdict calls the fp32 file exact", r["verdict"]["fp32_exact"])
    print(f"{sum(ok)} of {len(ok)} checks pass")
    return all(ok)


def main():
    p = argparse.ArgumentParser(
        description="Parity of an SDM-only export: PyTorch fp32 against the JS engine on the fp32 and int8 files, "
                    "with a perturbed-tensor negative control. Example:\n"
                    "  python3 %(prog)s --ckpt /path/ck/p0_A_c/last.pt --windows 8 --out /tmp/p0_A_c_check\n"
                    "  python3 %(prog)s --selftest",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt")
    p.add_argument("--result", default=None)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--samples", type=int, default=4096, help="vocabulary ids sampled for the mean abs logit difference")
    p.add_argument("--out", default=None)
    p.add_argument("--node", default="node")
    p.add_argument("--values-bits", type=int, default=8, choices=[8, 4])
    p.add_argument("--matrices", default="f32", choices=["f32", "i8"])
    p.add_argument("--perturb", default="store.0.values")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--modes", default="fp32,int8", help="fp32,int8 or int8 (the control always uses the int8 file)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.ckpt:
        p.print_help()
        print("\nNEXT -> python3 track4_sdmonly_export_check.py --selftest")
        return
    import torch
    torch.set_num_threads(a.threads)
    data = os.environ.get("SDMLLM_DATA", os.path.join(HERE, "data"))
    val = np.memmap(os.path.join(data, "val.u32"), dtype=np.uint32, mode="r")
    tb = np.fromfile(os.path.join(data, "token_bytes.i32"), dtype=np.int32)
    model, cfg, info = X.load_checkpoint(a.ckpt, a.result)
    out_dir = a.out or tempfile.mkdtemp(prefix="sdmo_check_")
    extra = {"checkpoint": os.path.abspath(a.ckpt), "checkpoint_sha256": X.sha256_file(a.ckpt), "cfg": cfg}
    r = check_pipeline(model, val, tb, out_dir, windows=a.windows, T=cfg.get("T", 256), n_samples=a.samples,
                       node=a.node, values_bits=a.values_bits, matrices=a.matrices, perturb=a.perturb, meta_extra=extra,
                       modes=tuple(a.modes.split(",")))
    print(json.dumps(r["verdict"]), "->", os.path.join(out_dir, "export_check.json"))
    print(f"NEXT -> node track4_sdmonly_engine.mjs --bench {os.path.join(out_dir, 'model.int8.sdmo')}")


if __name__ == "__main__":
    main()
