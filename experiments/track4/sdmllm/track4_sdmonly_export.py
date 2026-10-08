"""SDMONLY export: one SDM-only checkpoint -> one browser model file (.sdmo), plus a numpy reference forward.

<claudes_code_comments>
** Function List **
quantise_rows(w, bits, colscale) - symmetric per-row int8 or int4 quantisation, optional per-column pre-scale
pack_i4(q) / unpack_i4(b, shape) - two signed 4-bit values per byte (low nibble first, stored as q + 8)
load_checkpoint(path, result) - torch checkpoint -> (SdmOnlyLM in eval, cfg, info); infers shapes from the weights
fold_tensors(model) - the inference tensors: norms and batch norm folded into the maps, sub-keys made unit length
write_sdmo(model, path, mode, ...) - write the one-file export and return its header
read_sdmo(path) - parse a .sdmo file -> (header, {name: numpy array, dequantised where asked})
estimate_bytes(d, V, hops, n_sub, heads, d_a, ...) - the int8 file size as arithmetic, for planning
numpy_forward(parsed, ids, ema_window) - incremental forward from the file's tensors, mirroring the JS engine
selftest() - fold exactness, sliding-window averages, int8 and int4 round trips, shared store, heads, header
main() - CLI: export a checkpoint, estimate sizes, or run the self-test

** Technical Review **
- THE FILE (.sdmo, format version 1). One file so the browser fetches one thing; the tensor table and the blob are
  the site's export_sdm_chat.py layout unchanged, only wrapped:
    bytes 0..3      magic b"SDMO"
    bytes 4..7      uint32 LE, format version (1)
    bytes 8..11     uint32 LE, H = length of the JSON header in bytes
    bytes 12..12+H  the JSON header (UTF-8)
    zero padding to the next multiple of 16; header["blob_offset"] says where the blob starts
    the blob: every tensor C-contiguous little-endian at a 16-byte aligned offset RELATIVE TO THE BLOB START,
              listed in header["tensors"] as {name, dtype, shape, offset, bytes} (the site's tensor table).
  --pair writes the site's own two-file form instead: <key>.json (the same header) and <key>.bin (the blob).
- DTYPES: "f32"; "i8"; "i4" (two values a byte, low nibble first, nibble = q + 8, q in -7..7). A quantised matrix X
  is stored as X.q (i8 or i4, shape [rows, cols]) and X.scale (f32, [rows]); a reader looks for X first (f32) and
  falls back to X.q * X.scale[row] (* X.colscale[col] for the embedding). The embedding keeps the site's names
  (emb.q, emb.scale, emb.colscale) and scheme: divide each column by its max abs, then one scale per row.
- TENSORS (header["sdmonly"] holds the numbers that are not tensors: hops, heads, d_a, n_sub, M, k, softness,
  n_store, store_of_hop, readout_f, body, ema_window):
    emb                 V x d           tied input table and output head
    feat.wx             d x (F d)       W_x with each feature block's RMSNorm weight folded into its columns
    hop.<h>.wq          (heads d_a) x d diag(1/sqrt(var+eps)) W_q diag(qnorm weight): batch norm and query norm folded
    hop.<h>.qb          heads d_a       -running_mean / sqrt(var+eps), the folded batch-norm shift
    store.<s>.keys1/2   heads x n_sub x d_a/2   sub-keys already unit length (f32, small)
    store.<s>.values    M x d           the value table (i8 by default, i4 with --values-bits 4)
    readout.gate/up     f x d           (only when readout_f > 0) with the readout's RMSNorm weight folded in
    readout.down        d x f
    nf.w                d               the final RMSNorm weight (kept, d multiplies a token)
- THE FORWARD, one position t (what numpy_forward and the JS engine compute):
    u_j = rms(e_{t-j}) for j < n_back (zero before the window start); a_i = rms(EMA_i), EMA over the last
    min(t+1, ema_window) tokens; x = feat.wx [u; a]; per hop: q = hop.wq rms(x) + hop.qb, split per head, product-key
    top-2k exact search, theta = k-th score, w = sigmoid((s - theta)/softness), x += sum w v / (k heads);
    optional readout x += down(silu(gate rms(x)) * up rms(x)); logits = (rms(x) * nf.w) . E^T.
  rms(z) = z / sqrt(mean(z^2) + 1e-6). A half score is a dot with a unit sub-key; a location scores (s1+s2)/sqrt 2.
- EMA WINDOW. Training windows are 256 tokens and every average starts at the window start. ema_window = T (256)
  gives the same semantics at every position of an unbounded chat: the average covers the last 256 tokens, kept
  incrementally (add the new term, subtract the term that falls out). ema_window = 0 keeps an unbounded average.
- SIZE. estimate_bytes() is arithmetic only; main() prints the measured size of every file it writes.
Docs: PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md step 5 · track4_sdmonly_engine.mjs · track4_sdmonly_export_check.py
</claudes_code_comments>
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAGIC = b"SDMO"
VERSION = 1
V_DEFAULT = 129280
DTYPES = {"f32": np.float32, "i8": np.int8}


# ---------------------------------------------------------------- quantisation

def quantise_rows(w, bits=8, colscale=False):
    """Symmetric per-row quantisation. Returns (q int8 in [-qmax, qmax], row scales f32, column scales f32 or None)."""
    w = np.asarray(w, dtype=np.float32)
    c = None
    if colscale:
        c = np.abs(w).max(0).astype(np.float32)
        c[c == 0] = 1.0
        w = w / c[None, :]
    qmax = 127 if bits == 8 else 7
    s = (np.abs(w).max(1) / qmax).astype(np.float32)
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s[:, None]), -qmax, qmax).astype(np.int8)
    return q, s, c


def pack_i4(q):
    """int8 array with values in -7..7 (last dim even) -> uint8 bytes, two values a byte, low nibble first."""
    u = (q.astype(np.int16) + 8).astype(np.uint8).reshape(-1)
    if u.size % 2:
        u = np.concatenate([u, np.array([8], np.uint8)])
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8)


def unpack_i4(b, shape):
    b = np.asarray(b, dtype=np.uint8)
    lo = (b & 15).astype(np.int16) - 8
    hi = (b >> 4).astype(np.int16) - 8
    out = np.empty(b.size * 2, np.int16)
    out[0::2], out[1::2] = lo, hi
    n = int(np.prod(shape))
    return out[:n].reshape(shape).astype(np.int8)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- checkpoint -> folded tensors

def load_checkpoint(path, result=None):
    """Build the SdmOnlyLM a checkpoint holds. Shapes the cfg does not carry (d_a, n_sub, heads, readout_f,
    share_store) are read from the weights; a result json's "sdmonly" block, if given, is checked against them."""
    import torch
    import track4_sdmonly_models as M
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, cfg, arm = ck["model"], dict(ck["cfg"]), ck.get("arm", "sdmonly")
    if arm not in ("sdmonly", "sdmonly_none"):
        raise SystemExit(f"arm {arm!r}: only sdmonly and sdmonly_none export (the dense control has no SDM read)")
    if arm == "sdmonly":
        k1 = sd["store.banks.0.keys1"]
        cfg["heads"], cfg["n_sub"], cfg["d_a"] = int(k1.shape[0]), int(k1.shape[1]), int(k1.shape[2]) * 2
        n_banks = len({k.split(".")[2] for k in sd if k.startswith("store.banks.")})
        cfg["share_store"] = n_banks == 1 and cfg["hops"] > 1
    cfg["readout_f"] = int(sd["readout.gate.weight"].shape[0]) if "readout.gate.weight" in sd else 0
    cfg["d"] = int(sd["emb.weight"].shape[1])
    V = int(sd["emb.weight"].shape[0])
    checked = {}
    if result and os.path.exists(result):
        blk = json.load(open(result)).get("sdmonly", {})
        for key in ("d_a", "n_sub", "heads", "readout_f", "share_store"):
            if key in blk and key in cfg:
                checked[key] = blk[key] == cfg[key]
                if not checked[key]:
                    print(f"WARN result json says {key}={blk[key]}, the weights say {cfg[key]}; the weights win")
    model = M.build_sdmonly(arm, V, cfg)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg, {"arm": arm, "step": ck.get("step"), "result_checked": checked}


def fold_tensors(model):
    """The inference tensors as float32 numpy arrays (names as in the file, before quantisation)."""
    import torch
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    d = model.d
    nfeat = model.n_back + len(model.decays)
    out = {"emb": sd["emb.weight"]}
    wx = sd["wx.weight"].copy()
    for j in range(nfeat):
        wx[:, j * d:(j + 1) * d] *= sd[f"fnorms.{j}.w"][None, :]
    out["feat.wx"] = wx
    if model.store_mode == "sdm":
        for h in range(model.hops):
            bn = model.qbn[h]
            inv = 1.0 / np.sqrt(sd[f"qbn.{h}.running_var"] + bn.eps)
            out[f"hop.{h}.wq"] = (inv[:, None] * sd[f"wq.{h}.weight"]) * sd[f"qnorms.{h}.w"][None, :]
            out[f"hop.{h}.qb"] = (-sd[f"qbn.{h}.running_mean"] * inv).astype(np.float32)
        for s, bank in enumerate(model.store.banks):
            with torch.no_grad():
                out[f"store.{s}.keys1"] = torch.nn.functional.normalize(bank.keys1.float(), dim=-1).numpy()
                out[f"store.{s}.keys2"] = torch.nn.functional.normalize(bank.keys2.float(), dim=-1).numpy()
            out[f"store.{s}.values"] = sd[f"store.banks.{s}.values"]
    if model.f:
        rw = sd["rnorm.w"][None, :]
        out["readout.gate"] = sd["readout.gate.weight"] * rw
        out["readout.up"] = sd["readout.up.weight"] * rw
        out["readout.down"] = sd["readout.down.weight"]
    out["nf.w"] = sd["nf.w"]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}


def model_numbers(model):
    sm = {"body": model.store_mode, "hops": model.hops if model.store_mode == "sdm" else 0, "heads": model.heads,
          "d_a": model.d_a, "n_sub": model.n_sub, "M": model.M, "k": model.k, "readout_f": model.f}
    if model.store_mode == "sdm":
        b0 = model.store.banks[0]
        sm["softness"] = float(b0.softness)
        sm["n_store"] = len(model.store.banks)
        sm["store_of_hop"] = [h if len(model.store.banks) > 1 else 0 for h in range(model.hops)]
    return sm


# ---------------------------------------------------------------- write / read

def write_sdmo(model, path, mode="int8", values_bits=8, matrices="f32", meta_extra=None, pair=False, T=256,
               ema_window=None):
    """Write the export. mode int8: embedding and values quantised (matrices too with matrices='i8'); fp32: none."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tens = fold_tensors(model)
    blobs, table, off = [], [], 0
    zero_rows = {}

    def add(name, arr, dtype):
        nonlocal off
        if dtype == "i4":
            b, shape = pack_i4(arr).tobytes(), list(arr.shape)
        else:
            a = np.ascontiguousarray(arr.astype(DTYPES[dtype]))
            b, shape = a.tobytes(), list(a.shape)
        pad = (-off) % 16
        if pad:
            blobs.append(b"\0" * pad)
            off += pad
        table.append({"name": name, "dtype": dtype, "shape": shape, "offset": off, "bytes": len(b)})
        blobs.append(b)
        off += len(b)

    def add_matrix(name, w, how):
        if how == "f32":
            add(name, w, "f32")
            return
        bits = 4 if how == "i4" else 8
        q, s, c = quantise_rows(w, bits=bits, colscale=(name == "emb"))
        add(name + ".q", q, how)
        add(name + ".scale", s, "f32")
        if c is not None:
            add(name + ".colscale", c, "f32")

    vals_how = "f32" if mode == "fp32" else ("i4" if values_bits == 4 else "i8")
    mats_how = "f32" if mode == "fp32" else matrices
    for name, w in tens.items():
        if name == "emb":
            add_matrix(name, w, "f32" if mode == "fp32" else "i8")
        elif name.endswith(".values"):
            zero_rows[name] = int((np.abs(w).max(1) == 0).sum())
            add_matrix(name, w, vals_how)
        elif name == "feat.wx" or name.endswith(".wq") or name.startswith("readout."):
            add_matrix(name, w, mats_how)
        else:
            add(name, w, "f32")
    header = {
        "format": "sdmo", "format_version": VERSION, "kind": "sdmonly",
        "d": model.d, "V": model.V, "T": T, "n_back": model.n_back, "decays": [float(x) for x in model.decays],
        "f": model.f, "sdmonly": {**model_numbers(model), "ema_window": T if ema_window is None else ema_window},
        "rms_eps": 1e-6,
        "quantisation": {"mode": mode, "embedding": "f32" if mode == "fp32" else "i8 row+column scales",
                         "values": vals_how + ("" if vals_how == "f32" else " per-row scale"), "matrices": mats_how,
                         "zero_value_rows": zero_rows},
        "folding": "feature RMSNorm weights folded into feat.wx columns; per hop W_q' = diag(1/sqrt(var+eps)) W_q "
                   "diag(qnorm.w), qb = -mean/sqrt(var+eps); sub-keys unit length; readout RMSNorm folded into gate/up",
        "tensors": table, "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(meta_extra or {}),
    }
    blob = b"".join(blobs)
    if pair:
        base = path[:-5] if path.endswith(".sdmo") else path
        header["bin"] = os.path.basename(base) + ".bin"
        header["bin_bytes"] = len(blob)
        with open(base + ".bin", "wb") as f:
            f.write(blob)
        with open(base + ".json", "w") as f:
            json.dump(header, f, indent=1)
        return header
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    start = 12 + len(hjson)
    header["blob_offset"] = start + ((-start) % 16)
    # the offset itself lengthens the JSON; iterate until it is stable
    for _ in range(4):
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        start = 12 + len(hjson)
        bo = start + ((-start) % 16)
        if bo == header["blob_offset"]:
            break
        header["blob_offset"] = bo
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(hjson)) + hjson)
        f.write(b"\0" * (header["blob_offset"] - start))
        f.write(blob)
    return header


def read_sdmo(path, dequant=True):
    """Parse a .sdmo file (or a site-style <key>.json beside <key>.bin). Returns (header, tensors).
    With dequant, quantised matrices come back as float32 under their base name (X from X.q, X.scale, X.colscale)."""
    if path.endswith(".json"):
        header = json.load(open(path))
        blob = open(os.path.join(os.path.dirname(path), header["bin"]), "rb").read()
        base = 0
    else:
        raw = open(path, "rb").read()
        if raw[:4] != MAGIC:
            raise ValueError(f"{path}: not an SDMO file")
        ver, hl = struct.unpack("<II", raw[4:12])
        if ver != VERSION:
            raise ValueError(f"{path}: format version {ver}, this reader knows {VERSION}")
        header = json.loads(raw[12:12 + hl].decode("utf-8"))
        blob, base = raw, header["blob_offset"]
    t = {}
    for e in header["tensors"]:
        b = blob[base + e["offset"]: base + e["offset"] + e["bytes"]]
        if e["dtype"] == "i4":
            t[e["name"]] = unpack_i4(np.frombuffer(b, np.uint8), e["shape"])
        else:
            t[e["name"]] = np.frombuffer(b, DTYPES[e["dtype"]]).reshape(e["shape"])
    if dequant:
        for name in [n[:-2] for n in list(t) if n.endswith(".q")]:
            w = t[name + ".q"].astype(np.float32) * t[name + ".scale"][:, None]
            if name + ".colscale" in t:
                w = w * t[name + ".colscale"][None, :]
            t[name] = w
    return header, t


def estimate_bytes(d, V=V_DEFAULT, hops=4, n_sub=256, heads=1, d_a=256, n_back=8, n_decays=5, readout_f=0,
                   share_store=False, values_bits=8, matrices="f32"):
    """The int8 file size, arithmetic only (no header, no padding)."""
    S = 1 if share_store else hops
    Mloc = n_sub * n_sub
    F = n_back + n_decays
    mb = 4 if matrices == "f32" else 1
    parts = {
        "embedding": V * d + 4 * V + 4 * d,
        "values": S * Mloc * d * values_bits // 8 + 4 * S * Mloc,
        "sub_keys": 4 * S * heads * n_sub * d_a,
        "query_maps": hops * (mb * heads * d_a * d + 4 * heads * d_a + (4 * heads * d_a if mb == 1 else 0)),
        "wx": mb * F * d * d + (4 * d if mb == 1 else 0),
        "readout": mb * 3 * d * readout_f,
        "norms": 4 * d,
    }
    parts["total"] = sum(parts.values())
    return parts


# ---------------------------------------------------------------- numpy reference forward (mirrors the JS engine)

def _rms(z, eps=1e-6):
    return z / np.sqrt(np.mean(z * z) + eps)


def numpy_forward(parsed, ids, ema_window=None, positions=None):
    """Logits at every position (or at `positions`) of `ids`, computed incrementally from the file's tensors."""
    with np.errstate(all="ignore"):  # numpy 2 + Accelerate raise spurious matmul warnings on finite inputs
        return _numpy_forward(parsed, ids, ema_window, positions)


def _numpy_forward(parsed, ids, ema_window, positions):
    header, t = parsed
    d, V, nb = header["d"], header["V"], header["n_back"]
    decays = np.array(header["decays"], np.float64)
    sm = header["sdmonly"]
    W = sm["ema_window"] if ema_window is None else ema_window
    emb = t["emb"]
    num = np.zeros((len(decays), d), np.float64)
    den = np.zeros(len(decays), np.float64)
    back = [np.zeros(d, np.float32) for _ in range(nb)]
    out = {}
    for pos, tok in enumerate(ids):
        e = emb[tok].astype(np.float64)
        num = decays[:, None] * num + e[None, :]
        den = decays * den + 1.0
        if W and pos >= W:
            old = emb[ids[pos - W]].astype(np.float64)
            dw = decays ** W
            num -= dw[:, None] * old[None, :]
            den -= dw
        back = [_rms(emb[tok])] + back[:-1]
        if positions is not None and pos not in positions:
            continue
        feats = back + [_rms((num[i] / den[i]).astype(np.float32)) for i in range(len(decays))]
        x = t["feat.wx"] @ np.concatenate(feats).astype(np.float32)
        for h in range(sm["hops"]):
            q = t[f"hop.{h}.wq"] @ _rms(x) + t[f"hop.{h}.qb"]
            s = sm["store_of_hop"][h]
            k1, k2, vals = t[f"store.{s}.keys1"], t[f"store.{s}.keys2"], t[f"store.{s}.values"]
            k, da, heads, n_sub = sm["k"], sm["d_a"], sm["heads"], sm["n_sub"]
            read = np.zeros(d, np.float64)
            for hd in range(heads):
                qh = q[hd * da:(hd + 1) * da]
                s1 = k1[hd] @ qh[: da // 2]
                s2 = k2[hd] @ qh[da // 2:]
                c1 = min(2 * k, n_sub)
                i1 = np.argsort(-s1, kind="stable")[:c1]
                i2 = np.argsort(-s2, kind="stable")[:c1]
                comb = ((s1[i1][:, None] + s2[i2][None, :]) / math.sqrt(2.0)).reshape(-1)
                c = min(2 * k, c1 * c1)
                order = np.argsort(-comb, kind="stable")[:c]
                sv = comb[order]
                rows = i1[order // c1] * n_sub + i2[order % c1]
                kk = min(k, c)
                w = 1.0 / (1.0 + np.exp(-(sv - sv[kk - 1]) / sm["softness"]))
                read += (w[:, None] * vals[rows]).sum(0)
            x = x + (read / (k * heads)).astype(np.float32)
        if header["f"]:
            n = _rms(x)
            g = t["readout.gate"] @ n
            u = t["readout.up"] @ n
            x = x + t["readout.down"] @ ((g / (1.0 + np.exp(-g))) * u)
        hfin = _rms(x) * t["nf.w"]
        out[pos] = emb @ hfin
    return out


# ---------------------------------------------------------------- self-test

def selftest():
    import torch
    import track4_sdmonly_models as M
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))

    torch.manual_seed(0)
    V, T = 300, 16
    tmp = tempfile.mkdtemp(prefix="sdmo_selftest_")

    def tiny(**kw):
        m = M.SdmOnlyLM(V, d=32, d_a=16, n_sub=12, k=5, hops=3, n_back=4, decays=(0.6, 0.8, 0.97), **kw)
        for b in m.store.banks:
            torch.nn.init.normal_(b.values, std=0.3)
        m.train()
        with torch.no_grad():  # give batch norm non-trivial running statistics
            for _ in range(20):
                m(torch.randint(1, V, (4, T)))
        for nrm in list(m.fnorms) + list(m.qnorms) + [m.nf]:
            torch.nn.init.uniform_(nrm.w, 0.5, 1.5)
        m.eval()
        return m

    def torch_logits_at(m, ids, pos, window):
        lo = max(0, pos - window + 1) if window else 0
        with torch.no_grad():
            return m(torch.tensor([ids[lo:pos + 1]]))[0, -1].numpy()

    ids = [int(x) for x in torch.randint(1, V, (40,))]
    for label, kw in [("1 head, readout, one store a hop", {"heads": 1, "readout_f": 24}),
                      ("2 heads, no readout, shared store", {"heads": 2, "readout_f": 0, "share_store": True})]:
        m = tiny(**kw)
        p = os.path.join(tmp, "m32.sdmo")
        hdr = write_sdmo(m, p, mode="fp32", T=T)
        parsed = read_sdmo(p)
        pos = [0, 3, 15, 16, 25, 39]
        nf = numpy_forward(parsed, ids, positions=set(pos))
        err = max(float(np.abs(nf[q] - torch_logits_at(m, ids, q, T)).max()) for q in pos)
        check(f"fp32 file forward equals the module, sliding 16-token window [{label}]", err < 2e-4, f"max abs {err:.2e}")
        check(f"header round trip [{label}]", parsed[0]["sdmonly"]["hops"] == 3 and parsed[0]["d"] == 32
              and hdr["sdmonly"]["n_store"] == (1 if kw.get("share_store") else 3))
    # unbounded averages match a module run over the whole sequence
    m = tiny(heads=1, readout_f=0)
    p = os.path.join(tmp, "m32b.sdmo")
    write_sdmo(m, p, mode="fp32", T=T, ema_window=0)
    nf = numpy_forward(read_sdmo(p), ids, positions={39})
    err = float(np.abs(nf[39] - torch_logits_at(m, ids, 39, 0)).max())
    check("ema_window 0: unbounded averages equal the module on the whole sequence", err < 2e-4, f"max abs {err:.2e}")
    # int8 and int4 files: dequantised tensors close to float32, logits close
    m = tiny(heads=1, readout_f=24)
    tens = fold_tensors(m)
    for bits, tol in [(8, 0.006), (4, 0.08)]:
        p = os.path.join(tmp, f"m{bits}.sdmo")
        write_sdmo(m, p, mode="int8", values_bits=bits, matrices="i8", T=T)
        h8, t8 = read_sdmo(p)
        rel = float(np.abs(t8["store.0.values"] - tens["store.0.values"]).max() / np.abs(tens["store.0.values"]).max())
        check(f"int{bits} values round trip within {tol} of the max", rel <= tol, f"rel {rel:.4f}")
        lg8 = numpy_forward((h8, t8), ids, positions={39})[39]
        lg32 = torch_logits_at(m, ids, 39, T)
        check(f"int{bits} file logits track float32 (correlation > 0.99)", np.corrcoef(lg8, lg32)[0, 1] > 0.99,
              f"corr {np.corrcoef(lg8, lg32)[0, 1]:.4f}")
    q = np.random.default_rng(0).integers(-7, 8, size=(5, 7)).astype(np.int8)
    check("int4 pack and unpack are inverse (odd sizes too)", bool((unpack_i4(pack_i4(q), q.shape) == q).all()))
    # the negative control: a perturbed tensor moves the logits
    h, t = read_sdmo(os.path.join(tmp, "m8.sdmo"))
    t2 = dict(t)
    t2["store.0.values"] = -t["store.0.values"]
    a = numpy_forward((h, t), ids, positions={39})[39]
    b = numpy_forward((h, t2), ids, positions={39})[39]
    check("negative control: negating one value table moves the logits", float(np.abs(a - b).max()) > 1e-2,
          f"max abs {float(np.abs(a - b).max()):.3f}")
    # the pair form carries the same tensors
    p = os.path.join(tmp, "pair.sdmo")
    write_sdmo(m, p, mode="int8", T=T, pair=True)
    hp, tp = read_sdmo(os.path.join(tmp, "pair.json"))
    check("site-style pair (.json + .bin) reads back the same tensors", np.allclose(tp["emb"], t["emb"]))
    est = estimate_bytes(32, V=V, hops=3, n_sub=12, heads=1, d_a=16, n_back=4, n_decays=3, readout_f=24, matrices="i8")
    blob = sum(e["bytes"] for e in h["tensors"])
    check("estimate_bytes within 2% of the measured blob", abs(est["total"] - blob) / blob < 0.02, f"{est['total']} vs {blob}")
    print(f"{sum(ok)} of {len(ok)} checks pass")
    return all(ok)


# ---------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(
        description="Export an SDM-only checkpoint to one browser file (.sdmo). Example:\n"
                    "  python3 %(prog)s --ckpt ck/p0_A_c/last.pt --result runs/p0_A_c.result.json "
                    "--out /tmp/p0_A_c.int8.sdmo\n"
                    "  python3 %(prog)s --estimate --d 768 --hops 4 --n-sub 256\n"
                    "  python3 %(prog)s --selftest",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", help="checkpoint .pt (<ck>/<run>/last.pt)")
    p.add_argument("--result", help="the run's result json (optional; its sdmonly block is cross-checked)")
    p.add_argument("--out", help="output path (.sdmo, or with --pair a base path: <base>.json + <base>.bin)")
    p.add_argument("--mode", default="int8", choices=["int8", "fp32"])
    p.add_argument("--values-bits", type=int, default=8, choices=[8, 4])
    p.add_argument("--matrices", default="f32", choices=["f32", "i8"], help="wx, query maps, readout")
    p.add_argument("--ema-window", type=int, default=None, help="default T (256); 0 = unbounded averages")
    p.add_argument("--pair", action="store_true", help="write the site's two-file form instead of one .sdmo")
    p.add_argument("--key", default=None, help="model key written into the header")
    p.add_argument("--estimate", action="store_true", help="print the int8 size arithmetic and exit")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--d-a", type=int, default=256)
    p.add_argument("--hops", type=int, default=4)
    p.add_argument("--n-sub", type=int, default=256)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--readout-f", type=int, default=0)
    p.add_argument("--share-store", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.estimate:
        e = estimate_bytes(a.d, hops=a.hops, n_sub=a.n_sub, heads=a.heads, d_a=a.d_a, readout_f=a.readout_f,
                           share_store=a.share_store, values_bits=a.values_bits, matrices=a.matrices)
        print(json.dumps({k: round(v / 1e6, 2) for k, v in e.items()}, indent=1), "(MB, arithmetic)")
        print("NEXT -> python3 track4_sdmonly_export.py --ckpt <ck>/<run>/last.pt --out <file>.sdmo")
        return
    if not (a.ckpt and a.out):
        p.print_help()
        print("\nNEXT -> python3 track4_sdmonly_export.py --selftest")
        return
    model, cfg, info = load_checkpoint(a.ckpt, a.result)
    res = json.load(open(a.result)) if a.result and os.path.exists(a.result) else {}
    extra = {"key": a.key or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt))), "arm": info["arm"],
             "cfg": cfg, "step": info["step"], "checkpoint": os.path.abspath(a.ckpt),
             "checkpoint_sha256": sha256_file(a.ckpt), "checkpoint_bytes": os.path.getsize(a.ckpt),
             "test_bpb": res.get("val_test", {}).get("bpb"), "train_tokens": res.get("tokens_target"),
             "result_checked": info["result_checked"]}
    t0 = time.time()
    hdr = write_sdmo(model, a.out, mode=a.mode, values_bits=a.values_bits, matrices=a.matrices, meta_extra=extra,
                     pair=a.pair, T=cfg.get("T", 256), ema_window=a.ema_window)
    path = a.out if not a.pair else (a.out[:-5] if a.out.endswith(".sdmo") else a.out) + ".bin"
    size = os.path.getsize(path)
    est = estimate_bytes(model.d, V=model.V, hops=hdr["sdmonly"]["hops"], n_sub=model.n_sub, heads=model.heads,
                         d_a=model.d_a, n_back=model.n_back, n_decays=len(model.decays), readout_f=model.f,
                         share_store=hdr["sdmonly"].get("n_store", 1) == 1 and model.hops > 1,
                         values_bits=a.values_bits, matrices=a.matrices)
    print(json.dumps({"wrote": path, "bytes": size, "MB": round(size / 1e6, 2), "mode": a.mode,
                      "estimate_MB": round(est["total"] / 1e6, 2) if a.mode == "int8" else None,
                      "zero_value_rows": hdr["quantisation"]["zero_value_rows"], "seconds": round(time.time() - t0, 1)}))
    print(f"NEXT -> python3 track4_sdmonly_export_check.py --ckpt {a.ckpt} --windows 8")


if __name__ == "__main__":
    main()
