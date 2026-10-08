// THE SDM-ONLY ENGINE - the forward pass of an SDM-only language model, one token at a time, in plain JavaScript.
//
// <claudes_code_comments>
// ** Function List **
// parseSdmo(buffer)                    - read the .sdmo container: magic, version, JSON header, blob offset
// packSdmo(header, tensors)            - write a .sdmo container from typed arrays (tests and tools)
// SdmOnlyModel(meta, buffer, base)     - typed-array views over the exported tensors
// SdmOnlyModel.fromSdmo(buffer)        - the one-file form;  .fromPair(meta, bin) - the site's json + bin form
// SdmOnlyModel.embed(id, out)          - one dequantised embedding row
// SdmOnlyModel.newState()              - an empty rolling context: back-token ring, moving averages, id ring
// SdmOnlyModel.push(state, id)         - add one token to the context (O(d) per average; no hops, no head)
// SdmOnlyModel.forward(state, opts)    - the next-token logits from the context, and the internals
// SdmOnlyModel.stepState(state, id)    - push, then forward
// SdmOnlyModel.step(ids, opts)         - the site engine's call shape: the last position of a token list, with a
//                                        cached rolling state so an extended list costs one push
// SdmOnlyModel.locate(q, head, store)  - exact product-key top-2k search for one head
// topK(values, k) / softmaxInto / mulberry32 / sampleNext - the sampler, the site engine's shapes
// generate(model, promptIds, params)   - prompt, then sample until maxTokens or the stop id
// selftest()                           - brute-force forward, product-key search, int8/int4, cache, container, sampler
// parityMain / benchMain / main        - node CLI: --selftest, --parity, --bench, --info
//
// ** Technical Review **
// - No imports, no DOM, no fetch in the library part: it runs in a Web Worker, a page or node. The CLI at the end
//   loads node:fs and node:path dynamically, only when this file is run directly by node.
// - THE FILE is written by track4_sdmonly_export.py (format and folding are documented there): an embedding E (int8
//   with a per-row and a per-column scale), feat.wx with the feature RMSNorm weights folded in, per hop a query map
//   with batch norm folded (hop.h.wq, hop.h.qb), per store two unit sub-key tables and a value table (int8 or int4
//   per-row), an optional SwiGLU readout, nf.w. A tensor named X is float32; otherwise X.q (i8 or i4) + X.scale.
// - THE CONTEXT IS STATE, NOT A WINDOW OF IDS. The model sees the past only through n_back back tokens and a few
//   moving averages, so a state holds: a ring of the last n_back RMS-normalised embeddings, and per decay b the
//   running sums num = sum b^(t-s) e_s and den = sum b^(t-s) (float64). push() is num = b num + e, den = b den + 1;
//   with ema_window W (default 256, the training window) it also subtracts b^W e_{t-W}, so the average covers the
//   last W tokens exactly as a training window did, with no recomputation over the context. A prompt costs one push
//   a token (no hops, no head): the hidden state of a position never feeds a later position.
// - ONE POSITION: f = [n_back ring rows; rms(num_i / den_i)]; x = feat.wx f; per hop: q = wq rms(x) + qb; per head
//   the half queries are scored against n_sub unit sub-keys each, the top c1 = min(2k, n_sub) of each half are kept
//   (sorted), and the c = min(2k, c1^2) best sums (s1 + s2)/sqrt 2 are taken from the c1 x c1 grid with an early
//   stop (both lists are sorted, so a row or a column can be abandoned once it cannot beat the current worst). That
//   is the exact top-2k of all n_sub^2 locations. theta = the k-th score, w = sigmoid((s - theta)/softness),
//   x += sum w v_row / (k heads). Then the optional readout, then logits_v = sum_i E_vi rms(x)_i nf_i: with the
//   int8 embedding, scale_v * sum_i q_vi (h_i c_i), the column scale folded into h once a token. The head is
//   V x d multiply-adds a token, about 90% of the work at d 256.
// - SAMPLER: the site engine's sampleNext without its word-penalty and n-gram extras (repetition penalty,
//   temperature, top-k, top-p, one seeded draw). The logits are a Float32Array(V), the same as the site's, so the
//   site may keep its own sampler.
// - CLI (node): --selftest; --parity <ref.json> <model.sdmo> compares with PyTorch references written by
//   track4_sdmonly_export_check.py; --bench <model.sdmo> times prompt pushes and full steps; --info prints the header.
// Docs: track4_sdmonly_export.py (the format) · PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md step 5
// </claudes_code_comments>

export const SDMO_MAGIC = 'SDMO';
export const SDMO_VERSION = 1;

// ---------------------------------------------------------------- the container

export function parseSdmo(buffer) {
  const u8 = new Uint8Array(buffer);
  const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (magic !== SDMO_MAGIC) throw new Error(`not an SDMO file (magic ${JSON.stringify(magic)})`);
  const dv = new DataView(buffer);
  const version = dv.getUint32(4, true);
  if (version !== SDMO_VERSION) throw new Error(`SDMO format version ${version}; this engine reads ${SDMO_VERSION}`);
  const hl = dv.getUint32(8, true);
  const meta = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + hl)));
  return { meta, base: meta.blob_offset };
}

export function packSdmo(header, tensors) {
  // tensors: [{ name, dtype: 'f32'|'i8'|'i4', shape, data: TypedArray (i4: Uint8Array of packed bytes) }]
  const table = [];
  let off = 0;
  for (const t of tensors) {
    off += (16 - (off % 16)) % 16;
    table.push({ name: t.name, dtype: t.dtype, shape: t.shape, offset: off, bytes: t.data.byteLength });
    off += t.data.byteLength;
  }
  const h = { ...header, tensors: table, blob_offset: 0 };
  let enc = new TextEncoder().encode(JSON.stringify(h));
  for (let i = 0; i < 4; i++) {
    const start = 12 + enc.length;
    const bo = start + ((16 - (start % 16)) % 16);
    if (bo === h.blob_offset) break;
    h.blob_offset = bo;
    enc = new TextEncoder().encode(JSON.stringify(h));
  }
  const buf = new ArrayBuffer(h.blob_offset + off);
  const u8 = new Uint8Array(buf);
  u8.set([83, 68, 77, 79], 0);
  const dv = new DataView(buf);
  dv.setUint32(4, SDMO_VERSION, true);
  dv.setUint32(8, enc.length, true);
  u8.set(enc, 12);
  tensors.forEach((t, i) => u8.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), h.blob_offset + table[i].offset));
  return buf;
}

function view(buffer, base, t) {
  const o = base + t.offset;
  if (t.dtype === 'f32') return new Float32Array(buffer, o, t.bytes / 4);
  if (t.dtype === 'i8') return new Int8Array(buffer, o, t.bytes);
  if (t.dtype === 'i4') return new Uint8Array(buffer, o, t.bytes);
  throw new Error(`unknown dtype ${t.dtype}`);
}

// ---------------------------------------------------------------- small maths

function rmsInto(x, n, out, w = null) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const r = 1 / Math.sqrt(ss / n + 1e-6);
  if (w) for (let i = 0; i < n; i++) out[i] = x[i] * r * w[i];
  else for (let i = 0; i < n; i++) out[i] = x[i] * r;
  return out;
}

function norm(a, n = a.length) {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

// y = M x for a matrix accessor M (f32, or i8/i4 with per-row scale)
function matvec(M, x, y) {
  const { rows, cols } = M;
  if (M.kind === 'f32') {
    const W = M.w;
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const o = r * cols;
      for (let c = 0; c < cols; c++) s += W[o + c] * x[c];
      y[r] = s;
    }
  } else if (M.kind === 'i8') {
    const Q = M.q;
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const o = r * cols;
      for (let c = 0; c < cols; c++) s += Q[o + c] * x[c];
      y[r] = s * M.s[r];
    }
  } else {
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const o = r * cols;
      for (let c = 0; c < cols; c++) {
        const k = o + c;
        const b = M.q[k >> 1];
        s += ((k & 1 ? b >> 4 : b & 15) - 8) * x[c];
      }
      y[r] = s * M.s[r];
    }
  }
  return y;
}

// out += a * row(M, r)
function axpyRow(M, r, a, out) {
  const d = M.cols;
  const o = r * d;
  if (M.kind === 'f32') for (let i = 0; i < d; i++) out[i] += a * M.w[o + i];
  else if (M.kind === 'i8') {
    const as = a * M.s[r];
    for (let i = 0; i < d; i++) out[i] += as * M.q[o + i];
  } else {
    const as = a * M.s[r];
    for (let i = 0; i < d; i++) {
      const k = o + i;
      const b = M.q[k >> 1];
      out[i] += as * ((k & 1 ? b >> 4 : b & 15) - 8);
    }
  }
}

// insert v (with payload p) into a descending list of capacity cap; returns the new count
function insertDesc(vals, pay, cnt, cap, v, p) {
  let i = cnt < cap ? cnt : cap - 1;
  if (cnt >= cap && v <= vals[cap - 1]) return cnt;
  while (i > 0 && vals[i - 1] < v) {
    vals[i] = vals[i - 1];
    pay[i] = pay[i - 1];
    i--;
  }
  vals[i] = v;
  pay[i] = p;
  return cnt < cap ? cnt + 1 : cnt;
}

// ---------------------------------------------------------------- the model

export class SdmOnlyModel {
  constructor(meta, buffer, base = 0) {
    if (meta.kind !== 'sdmonly') throw new Error(`model kind ${meta.kind}: this engine runs sdmonly exports`);
    this.meta = meta;
    this.d = meta.d;
    this.V = meta.V;
    this.T = meta.T;
    this.nBack = meta.n_back;
    this.decays = meta.decays;
    this.f = meta.f;
    const sm = meta.sdmonly;
    this.sm = sm;
    this.hops = sm.hops;
    this.heads = sm.heads;
    this.dA = sm.d_a;
    this.nSub = sm.n_sub;
    this.k = sm.k;
    this.softness = sm.softness;
    this.window = sm.ema_window ?? meta.T;
    const t = {};
    for (const x of meta.tensors) t[x.name] = { v: view(buffer, base, x), shape: x.shape, dtype: x.dtype };
    this.t = t;
    this.mat = (name) => {
      if (t[name]) return { kind: 'f32', w: t[name].v, rows: t[name].shape[0], cols: t[name].shape[1] };
      const q = t[name + '.q'];
      if (!q) throw new Error(`tensor ${name} is missing`);
      return { kind: q.dtype, q: q.v, s: t[name + '.scale'].v, c: t[name + '.colscale']?.v || null, rows: q.shape[0], cols: q.shape[1] };
    };
    this.vec = (name) => t[name].v;
    this.E = this.mat('emb');
    if (this.E.kind === 'i4') throw new Error('an int4 embedding is not supported');
    this.wx = this.mat('feat.wx');
    this.wq = [];
    this.qb = [];
    for (let h = 0; h < this.hops; h++) {
      this.wq.push(this.mat(`hop.${h}.wq`));
      this.qb.push(this.vec(`hop.${h}.qb`));
    }
    const nStore = this.hops ? sm.n_store : 0;
    this.stores = [];
    for (let s = 0; s < nStore; s++) {
      this.stores.push({ k1: this.vec(`store.${s}.keys1`), k2: this.vec(`store.${s}.keys2`), values: this.mat(`store.${s}.values`) });
    }
    if (this.f) {
      this.gate = this.mat('readout.gate');
      this.up = this.mat('readout.up');
      this.down = this.mat('readout.down');
    }
    this.nfw = this.vec('nf.w');
    const d = this.d;
    this.nD = this.decays.length;
    this.F = this.nBack + this.nD;
    this.decW = this.decays.map((b) => (this.window ? Math.pow(b, this.window) : 0));
    const c1 = Math.min(2 * this.k, this.nSub);
    const c = Math.min(2 * this.k, c1 * c1);
    this.c1 = c1;
    this.c = c;
    this.buf = {
      e: new Float32Array(d), old: new Float32Array(d), f: new Float32Array(this.F * d), x: new Float32Array(d),
      nx: new Float32Array(d), q: new Float32Array(this.heads * this.dA), read: new Float64Array(d), tmp: new Float32Array(d),
      g: new Float32Array(this.f || 1), u: new Float32Array(this.f || 1), h: new Float32Array(d),
      s1: new Float64Array(this.nSub), s2: new Float64Array(this.nSub),
      v1: new Float64Array(c1), i1: new Int32Array(c1), v2: new Float64Array(c1), i2: new Int32Array(c1),
      gv: new Float64Array(c), gp: new Int32Array(c), logits: new Float32Array(this.V),
    };
    this._cache = null;
  }

  static fromSdmo(buffer) {
    const { meta, base } = parseSdmo(buffer);
    return new SdmOnlyModel(meta, buffer, base);
  }

  static fromPair(meta, bin) {
    return new SdmOnlyModel(meta, bin, 0);
  }

  embed(id, out) {
    const E = this.E;
    const d = this.d;
    const o = id * d;
    if (E.kind === 'f32') for (let i = 0; i < d; i++) out[i] = E.w[o + i];
    else {
      const s = E.s[id];
      if (E.c) for (let i = 0; i < d; i++) out[i] = E.q[o + i] * s * E.c[i];
      else for (let i = 0; i < d; i++) out[i] = E.q[o + i] * s;
    }
    return out;
  }

  newState() {
    const d = this.d;
    return {
      pos: 0,
      back: new Float32Array(this.nBack * d),
      backIds: new Int32Array(this.nBack).fill(-1),
      num: new Float64Array(this.nD * d),
      den: new Float64Array(this.nD),
      ring: new Int32Array(this.window || 1),
    };
  }

  push(state, id) {
    const d = this.d;
    const B = this.buf;
    this.embed(id, B.e);
    const W = this.window;
    const drop = W && state.pos >= W;
    if (drop) this.embed(state.ring[state.pos % W], B.old);
    for (let i = 0; i < this.nD; i++) {
      const b = this.decays[i];
      const o = i * d;
      const num = state.num;
      if (drop) {
        const bw = this.decW[i];
        for (let k = 0; k < d; k++) num[o + k] = b * num[o + k] + B.e[k] - bw * B.old[k];
        state.den[i] = b * state.den[i] + 1 - bw;
      } else {
        for (let k = 0; k < d; k++) num[o + k] = b * num[o + k] + B.e[k];
        state.den[i] = b * state.den[i] + 1;
      }
    }
    if (W) state.ring[state.pos % W] = id;
    // the back ring: slot (pos mod n_back) holds the newest row
    const slot = state.pos % this.nBack;
    rmsInto(B.e, d, state.back.subarray(slot * d, (slot + 1) * d));
    state.backIds[slot] = id;
    state.pos++;
    return state;
  }

  // the exact top-2k locations for one head of one hop: fills B.gv (scores, descending, already / sqrt 2) and B.gp
  // (location ids); returns the count
  locate(q, head, store) {
    const B = this.buf;
    const half = this.dA >> 1;
    const n = this.nSub;
    const qo = head * this.dA;
    const ko = head * n * half;
    for (let j = 0; j < n; j++) {
      let a = 0;
      let b = 0;
      const o = ko + j * half;
      for (let i = 0; i < half; i++) {
        a += q[qo + i] * store.k1[o + i];
        b += q[qo + half + i] * store.k2[o + i];
      }
      B.s1[j] = a;
      B.s2[j] = b;
    }
    const c1 = this.c1;
    let n1 = 0;
    let n2 = 0;
    for (let j = 0; j < n; j++) {
      n1 = insertDesc(B.v1, B.i1, n1, c1, B.s1[j], j);
      n2 = insertDesc(B.v2, B.i2, n2, c1, B.s2[j], j);
    }
    const c = this.c;
    let cnt = 0;
    for (let a = 0; a < c1; a++) {
      if (cnt === c && B.v1[a] + B.v2[0] <= B.gv[c - 1]) break;
      for (let b = 0; b < c1; b++) {
        const s = B.v1[a] + B.v2[b];
        if (cnt === c && s <= B.gv[c - 1]) break;
        cnt = insertDesc(B.gv, B.gp, cnt, c, s, a * c1 + b);
      }
    }
    const r2 = Math.SQRT1_2;
    for (let m = 0; m < cnt; m++) {
      const ab = B.gp[m];
      B.gp[m] = B.i1[(ab / c1) | 0] * n + B.i2[ab % c1];
      B.gv[m] *= r2;
    }
    return cnt;
  }

  forward(state, opts = {}) {
    const d = this.d;
    const B = this.buf;
    const info = opts.info ? { pos: state.pos, hops: [] } : null;
    // features: the back ring (newest first), then the moving averages
    for (let j = 0; j < this.nBack; j++) {
      const seg = B.f.subarray(j * d, (j + 1) * d);
      if (state.pos - 1 - j >= 0) {
        const slot = (state.pos - 1 - j) % this.nBack;
        seg.set(state.back.subarray(slot * d, (slot + 1) * d));
      } else seg.fill(0);
    }
    for (let i = 0; i < this.nD; i++) {
      const den = state.den[i] || 1;
      for (let k = 0; k < d; k++) B.tmp[k] = state.num[i * d + k] / den;
      rmsInto(B.tmp, d, B.f.subarray((this.nBack + i) * d, (this.nBack + i + 1) * d));
    }
    matvec(this.wx, B.f, B.x);
    if (info) {
      // the site engine's science record: window, back ids (newest first), each feature block's share of x, the
      // averages' weights on the most recent tokens, and |x| before the hops
      const W = this.window;
      info.window = W ? Math.min(state.pos, W) : state.pos;
      info.back = [];
      for (let j = 0; j < this.nBack; j++) info.back.push(state.pos - 1 - j >= 0 ? state.backIds[(state.pos - 1 - j) % this.nBack] : null);
      info.blocks = [];
      const Fd = this.F * d;
      for (let b = 0; b < this.F; b++) {
        let ss = 0;
        for (let r = 0; r < d; r++) {
          let s = 0;
          if (this.wx.kind === 'f32') for (let c = 0; c < d; c++) s += this.wx.w[r * Fd + b * d + c] * B.f[b * d + c];
          else {
            for (let c = 0; c < d; c++) s += this.wx.q[r * Fd + b * d + c] * B.f[b * d + c];
            s *= this.wx.s[r];
          }
          ss += s * s;
        }
        info.blocks.push(Math.sqrt(ss));
      }
      info.ema = this.decays.map((b, i) => ({
        decay: b,
        top: info.back.filter((id) => id !== null).map((id, j) => ({ pos: -1 - j, id, w: Math.pow(b, j) / (state.den[i] || 1) })),
      }));
      info.xNorm0 = norm(B.x);
      info.readoutNorm = 0;
    }
    // the hops
    const k = this.k;
    for (let h = 0; h < this.hops; h++) {
      rmsInto(B.x, d, B.nx);
      matvec(this.wq[h], B.nx, B.q);
      const qb = this.qb[h];
      for (let i = 0; i < B.q.length; i++) B.q[i] += qb[i];
      const store = this.stores[this.sm.store_of_hop[h]];
      B.read.fill(0);
      const hop = info ? { idx: [], w: [], scores: [], thetas: [], head: [], fireMass: 0 } : null;
      for (let hd = 0; hd < this.heads; hd++) {
        const cnt = this.locate(B.q, hd, store);
        const kk = Math.min(k, cnt);
        const theta = B.gv[kk - 1];
        for (let m = 0; m < cnt; m++) {
          const w = 1 / (1 + Math.exp(-(B.gv[m] - theta) / this.softness));
          axpyRow(store.values, B.gp[m], w, B.read);
          if (hop) {
            hop.idx.push(B.gp[m]);
            hop.w.push(w);
            hop.scores.push(B.gv[m]);
            hop.head.push(hd);
            hop.fireMass += w;
          }
        }
        if (hop) hop.thetas.push(theta);
      }
      const g = 1 / (k * this.heads);
      if (hop) {
        hop.theta = hop.thetas[0];
        hop.xNorm = norm(B.x);
        hop.readNorm = norm(B.read) * g;
        info.hops.push(hop);
      }
      for (let i = 0; i < d; i++) B.x[i] += B.read[i] * g;
    }
    if (this.f) {
      rmsInto(B.x, d, B.nx);
      matvec(this.gate, B.nx, B.g);
      matvec(this.up, B.nx, B.u);
      for (let i = 0; i < this.f; i++) {
        const z = B.g[i];
        B.g[i] = (z / (1 + Math.exp(-z))) * B.u[i];
      }
      matvec(this.down, B.g, B.tmp);
      if (info) info.readoutNorm = norm(B.tmp);
      for (let i = 0; i < d; i++) B.x[i] += B.tmp[i];
    }
    rmsInto(B.x, d, B.h, this.nfw);
    if (info) info.hNorm = norm(B.h);
    return this.head(B.h, opts, info);
  }

  head(hIn, opts, info) {
    const d = this.d;
    const E = this.E;
    let h = hIn;
    if (E.c) {
      h = this.buf.nx;
      for (let i = 0; i < d; i++) h[i] = hIn[i] * E.c[i];
    }
    if (opts.only) {
      const out = new Float32Array(opts.only.length);
      opts.only.forEach((v, j) => {
        let s = 0;
        const o = v * d;
        if (E.kind === 'f32') for (let i = 0; i < d; i++) s += E.w[o + i] * h[i];
        else {
          for (let i = 0; i < d; i++) s += E.q[o + i] * h[i];
          s *= E.s[v];
        }
        out[j] = s;
      });
      return { logits: null, only: out, info };
    }
    const lg = opts.out || this.buf.logits;
    const V = this.V;
    const W = E.kind === 'f32' ? E.w : E.q;
    const unroll = d % 4 === 0;
    for (let v = 0; v < V; v++) {
      const o = v * d;
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      let s3 = 0;
      if (unroll) {
        for (let i = 0; i < d; i += 4) {
          s0 += W[o + i] * h[i];
          s1 += W[o + i + 1] * h[i + 1];
          s2 += W[o + i + 2] * h[i + 2];
          s3 += W[o + i + 3] * h[i + 3];
        }
      } else for (let i = 0; i < d; i++) s0 += W[o + i] * h[i];
      const s = s0 + s1 + s2 + s3;
      lg[v] = E.kind === 'f32' ? s : s * E.s[v];
    }
    return { logits: lg, info };
  }

  stepState(state, id, opts = {}) {
    this.push(state, id);
    return this.forward(state, opts);
  }

  // The site engine's call: forward the LAST position of `ids`. A list that extends the previous call's list costs one
  // push a new token; any other list rebuilds the state (pushes only: cheap).
  step(ids, opts = {}) {
    const c = this._cache;
    let st;
    if (c && ids.length >= c.ids.length && c.ids.every((v, i) => ids[i] === v)) {
      st = c.state;
      for (let i = c.ids.length; i < ids.length; i++) this.push(st, ids[i]);
    } else {
      st = this.newState();
      const from = this.window ? Math.max(0, ids.length - this.window) : 0;
      for (let i = from; i < ids.length; i++) this.push(st, ids[i]);
    }
    this._cache = { ids: Array.from(ids), state: st };
    return this.forward(st, { info: true, ...opts });
  }
}

// ---------------------------------------------------------------- sampling (the site engine's shapes)

export function topK(values, k) {
  const vals = new Float64Array(k);
  const idx = new Int32Array(k);
  let n = 0;
  for (let i = 0; i < values.length; i++) n = insertDesc(vals, idx, n, k, values[i], i);
  return Array.from(idx.subarray(0, n));
}

export function softmaxInto(logits, temp, out) {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  let s = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp((logits[i] - mx) / temp);
    out[i] = e;
    s += e;
  }
  for (let i = 0; i < logits.length; i++) out[i] /= s;
  return out;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// params: { temperature, topK, topP, repetitionPenalty, repeatWindow }. Returns { id, pSampled }.
export function sampleNext(logits, params, rng, recent = [], scratch = null) {
  const V = logits.length;
  const work = scratch && scratch.length === V ? scratch : new Float32Array(V);
  work.set(logits);
  const pen = params.repetitionPenalty ?? 1;
  if (pen !== 1) {
    for (const id of new Set(recent.slice(-(params.repeatWindow ?? 64)))) work[id] = work[id] > 0 ? work[id] / pen : work[id] * pen;
  }
  const temp = params.temperature ?? 1;
  if (temp <= 0) {
    let best = 0;
    for (let i = 1; i < V; i++) if (work[i] > work[best]) best = i;
    return { id: best, pSampled: 1 };
  }
  const k = params.topK > 0 ? Math.min(params.topK, V) : V;
  const cand = k < V ? topK(work, k) : Array.from({ length: V }, (_, i) => i);
  let mx = -Infinity;
  for (const c of cand) if (work[c] > mx) mx = work[c];
  const p = cand.map((c) => Math.exp((work[c] - mx) / temp));
  const tot = p.reduce((a, b) => a + b, 0);
  let order = cand.map((c, i) => i);
  if ((params.topP ?? 1) < 1) order.sort((a, b) => p[b] - p[a]);
  let keep = order.length;
  if ((params.topP ?? 1) < 1) {
    let m = 0;
    for (let i = 0; i < order.length; i++) {
      m += p[order[i]] / tot;
      if (m >= params.topP) {
        keep = i + 1;
        break;
      }
    }
  }
  let mass = 0;
  for (let i = 0; i < keep; i++) mass += p[order[i]];
  const r = rng() * mass;
  let acc = 0;
  for (let i = 0; i < keep; i++) {
    acc += p[order[i]];
    if (acc >= r) return { id: cand[order[i]], pSampled: p[order[i]] / mass };
  }
  return { id: cand[order[keep - 1]], pSampled: p[order[keep - 1]] / mass };
}

export function generate(model, promptIds, params = {}, { seed = 1, stopId = 1, onToken = null } = {}) {
  const st = model.newState();
  for (const id of promptIds) model.push(st, id);
  const rng = mulberry32(seed);
  const out = [];
  const recent = Array.from(promptIds);
  const scratch = new Float32Array(model.V);
  for (let n = 0; n < (params.maxTokens ?? 64); n++) {
    const { logits } = model.forward(st);
    const { id } = sampleNext(logits, params, rng, recent, scratch);
    if (id === stopId) break;
    out.push(id);
    recent.push(id);
    if (onToken) onToken(id);
    model.push(st, id);
  }
  return out;
}

// ---------------------------------------------------------------- self-test (a synthetic model, no files)

function synthetic({ V = 97, d = 16, dA = 8, nSub = 7, k = 3, hops = 2, heads = 2, nBack = 3, decays = [0.6, 0.9], f = 12, share = false, T = 9, valuesDtype = 'f32', embDtype = 'f32', seed = 3 } = {}) {
  const rng = mulberry32(seed);
  const randn = () => {
    const u = rng() || 1e-9;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  };
  const F = nBack + decays.length;
  const tensors = [];
  const f32 = (name, shape, scale = 1, map = null) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = randn() * scale;
    if (map) map(a, shape);
    tensors.push({ name, dtype: 'f32', shape, data: a });
    return a;
  };
  const quant = (name, shape, scale, dtype) => {
    const [rows, cols] = shape;
    const s = new Float32Array(rows);
    const qmax = dtype === 'i8' ? 127 : 7;
    const q = new Int8Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      s[r] = (scale * (0.5 + rng())) / qmax;
      for (let c = 0; c < cols; c++) q[r * cols + c] = Math.round((rng() * 2 - 1) * qmax);
    }
    let data = q;
    if (dtype === 'i4') {
      data = new Uint8Array(Math.ceil(q.length / 2));
      for (let i = 0; i < q.length; i++) data[i >> 1] |= (q[i] + 8) << (i & 1 ? 4 : 0);
    }
    tensors.push({ name: name + '.q', dtype, shape, data });
    tensors.push({ name: name + '.scale', dtype: 'f32', shape: [rows], data: s });
  };
  const unit = (a, shape) => {
    const w = shape[shape.length - 1];
    for (let o = 0; o < a.length; o += w) {
      let s = 0;
      for (let i = 0; i < w; i++) s += a[o + i] * a[o + i];
      s = Math.sqrt(s);
      for (let i = 0; i < w; i++) a[o + i] /= s;
    }
  };
  if (embDtype === 'f32') f32('emb', [V, d], 0.5);
  else quant('emb', [V, d], 0.5, embDtype);
  f32('feat.wx', [d, F * d], 1 / Math.sqrt(F * d));
  for (let h = 0; h < hops; h++) {
    f32(`hop.${h}.wq`, [heads * dA, d], 1 / Math.sqrt(d));
    f32(`hop.${h}.qb`, [heads * dA], 0.1);
  }
  const nStore = share ? 1 : hops;
  for (let s = 0; s < nStore; s++) {
    f32(`store.${s}.keys1`, [heads, nSub, dA / 2], 1, unit);
    f32(`store.${s}.keys2`, [heads, nSub, dA / 2], 1, unit);
    if (valuesDtype === 'f32') f32(`store.${s}.values`, [nSub * nSub, d], 0.8);
    else quant(`store.${s}.values`, [nSub * nSub, d], 0.8, valuesDtype);
  }
  if (f) {
    f32('readout.gate', [f, d], 0.3);
    f32('readout.up', [f, d], 0.3);
    f32('readout.down', [d, f], 0.3);
  }
  f32('nf.w', [d], 0.2, (a) => a.forEach((_, i) => (a[i] = 1 + 0.2 * a[i])));
  const header = {
    format: 'sdmo', format_version: 1, kind: 'sdmonly', d, V, T, n_back: nBack, decays, f,
    sdmonly: { body: 'sdm', hops, heads, d_a: dA, n_sub: nSub, M: nSub * nSub, k, softness: 0.25, n_store: nStore,
      store_of_hop: Array.from({ length: hops }, (_, h) => (share ? 0 : h)), readout_f: f, ema_window: T },
  };
  return packSdmo(header, tensors);
}

// The definition, computed the slow way: averages recomputed over the window, every location scored.
function bruteForward(model, ids, pos) {
  const d = model.d;
  const W = model.window;
  const lo = W ? Math.max(0, pos - W + 1) : 0;
  const e = new Float32Array(d);
  const f = new Float32Array(model.F * d);
  for (let j = 0; j < model.nBack; j++) {
    if (pos - j >= lo) {
      model.embed(ids[pos - j], e);
      rmsInto(e, d, f.subarray(j * d, (j + 1) * d));
    }
  }
  model.decays.forEach((b, i) => {
    const acc = new Float64Array(d);
    let ws = 0;
    for (let s = lo; s <= pos; s++) {
      const w = Math.pow(b, pos - s);
      model.embed(ids[s], e);
      for (let k = 0; k < d; k++) acc[k] += w * e[k];
      ws += w;
    }
    const a = new Float32Array(d);
    for (let k = 0; k < d; k++) a[k] = acc[k] / ws;
    rmsInto(a, d, f.subarray((model.nBack + i) * d, (model.nBack + i + 1) * d));
  });
  const x = matvec(model.wx, f, new Float32Array(d));
  const nx = new Float32Array(d);
  const half = model.dA / 2;
  for (let h = 0; h < model.hops; h++) {
    rmsInto(x, d, nx);
    const q = matvec(model.wq[h], nx, new Float32Array(model.heads * model.dA));
    for (let i = 0; i < q.length; i++) q[i] += model.qb[h][i];
    const st = model.stores[model.sm.store_of_hop[h]];
    const read = new Float64Array(d);
    for (let hd = 0; hd < model.heads; hd++) {
      const all = [];
      for (let a = 0; a < model.nSub; a++) {
        for (let b = 0; b < model.nSub; b++) {
          let s = 0;
          for (let i = 0; i < half; i++) {
            s += q[hd * model.dA + i] * st.k1[(hd * model.nSub + a) * half + i];
            s += q[hd * model.dA + half + i] * st.k2[(hd * model.nSub + b) * half + i];
          }
          all.push([s / Math.SQRT2, a * model.nSub + b]);
        }
      }
      all.sort((u, v) => v[0] - u[0]);
      const top = all.slice(0, Math.min(2 * model.k, all.length));
      const theta = top[Math.min(model.k, top.length) - 1][0];
      for (const [s, row] of top) axpyRow(st.values, row, 1 / (1 + Math.exp(-(s - theta) / model.softness)), read);
    }
    for (let i = 0; i < d; i++) x[i] += read[i] / (model.k * model.heads);
  }
  if (model.f) {
    rmsInto(x, d, nx);
    const g = matvec(model.gate, nx, new Float32Array(model.f));
    const u = matvec(model.up, nx, new Float32Array(model.f));
    for (let i = 0; i < model.f; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
    const o = matvec(model.down, g, new Float32Array(d));
    for (let i = 0; i < d; i++) x[i] += o[i];
  }
  const h = rmsInto(x, d, new Float32Array(d), model.nfw);
  const lg = new Float32Array(model.V);
  const er = new Float32Array(d);
  for (let v = 0; v < model.V; v++) {
    model.embed(v, er);
    let s = 0;
    for (let i = 0; i < d; i++) s += er[i] * h[i];
    lg[v] = s;
  }
  return lg;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

export function selftest() {
  const results = [];
  const check = (name, cond, detail = '') => {
    results.push(!!cond);
    console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${detail ? `  (${detail})` : ''}`);
  };
  const ids = [];
  const r = mulberry32(11);
  for (let i = 0; i < 30; i++) ids.push(Math.floor(r() * 97));
  const variants = [
    ['f32, 2 heads, readout, a store a hop', {}],
    ['i8 values and i8 embedding, shared store, no readout', { valuesDtype: 'i8', embDtype: 'i8', share: true, f: 0 }],
    ['i4 values, 1 head, odd d', { valuesDtype: 'i4', heads: 1, d: 15, f: 0 }],
    ['unbounded averages (window 0)', { T: 0 }],
  ];
  for (const [label, opt] of variants) {
    const m = SdmOnlyModel.fromSdmo(synthetic(opt));
    const st = m.newState();
    let worst = 0;
    for (let p = 0; p < ids.length; p++) {
      const { logits } = m.stepState(st, ids[p]);
      if ([0, 2, 8, 9, 17, 29].includes(p)) worst = Math.max(worst, maxAbsDiff(logits, bruteForward(m, ids, p)));
    }
    check(`incremental context equals the brute-force definition [${label}]`, worst < 1e-4, `max abs ${worst.toExponential(2)}`);
  }
  // the product-key search returns exactly the brute-force top-2k set
  const m = SdmOnlyModel.fromSdmo(synthetic({ nSub: 11, k: 4, dA: 10 }));
  const rq = mulberry32(5);
  let same = true;
  for (let trial = 0; trial < 50; trial++) {
    const q = new Float32Array(m.heads * m.dA).map(() => rq() * 2 - 1);
    for (let hd = 0; hd < m.heads; hd++) {
      const cnt = m.locate(q, hd, m.stores[0]);
      const got = Array.from(m.buf.gp.subarray(0, cnt)).sort((a, b) => a - b);
      const half = m.dA / 2;
      const all = [];
      for (let a = 0; a < m.nSub; a++)
        for (let b = 0; b < m.nSub; b++) {
          let s = 0;
          for (let i = 0; i < half; i++) s += q[hd * m.dA + i] * m.stores[0].k1[(hd * m.nSub + a) * half + i] + q[hd * m.dA + half + i] * m.stores[0].k2[(hd * m.nSub + b) * half + i];
          all.push([s, a * m.nSub + b]);
        }
      all.sort((u, v) => v[0] - u[0]);
      const want = all.slice(0, cnt).map((x) => x[1]).sort((a, b) => a - b);
      if (cnt !== 2 * m.k || want.join() !== got.join()) same = false;
    }
  }
  check('product-key search returns the brute-force top 2k locations (50 queries x 2 heads)', same);
  // step(ids) with the cache equals a fresh state; a non-prefix list rebuilds
  const m2 = SdmOnlyModel.fromSdmo(synthetic({ T: 9 }));
  const a1 = Float32Array.from(m2.step(ids.slice(0, 20)).logits);
  const a2 = Float32Array.from(m2.step(ids.slice(0, 21)).logits);
  const fresh = SdmOnlyModel.fromSdmo(synthetic({ T: 9 }));
  const b2 = Float32Array.from(fresh.step(ids.slice(0, 21)).logits);
  const other = Float32Array.from(m2.step([5, ...ids.slice(1, 21)]).logits);
  const b3 = Float32Array.from(SdmOnlyModel.fromSdmo(synthetic({ T: 9 })).step([5, ...ids.slice(1, 21)]).logits);
  check('step(ids): an extended list (cached, one push) equals a fresh model', maxAbsDiff(a2, b2) < 1e-6);
  check('step(ids): a changed list rebuilds the state', maxAbsDiff(other, b3) < 1e-6 && maxAbsDiff(a1, a2) > 0);
  const inf = fresh.step(ids.slice(0, 21)).info;
  check("step(ids).info carries the site engine's science fields",
    ['window', 'back', 'blocks', 'ema', 'xNorm0', 'hops', 'readoutNorm'].every((key) => key in inf) && inf.window === 9 &&
    inf.back[0] === ids[20] && inf.blocks.length === fresh.F && inf.hops.length === 2 &&
    inf.hops[0].idx.length === 2 * fresh.k * fresh.heads && typeof inf.hops[0].theta === 'number' && inf.hops[0].scores.length > 0);
  // the container
  const buf = synthetic({});
  const { meta } = parseSdmo(buf);
  check('container: header parses, blob 16-aligned, every tensor in range', meta.blob_offset % 16 === 0 &&
    meta.tensors.every((t) => t.offset % 16 === 0 && meta.blob_offset + t.offset + t.bytes <= buf.byteLength));
  let threw = false;
  try {
    parseSdmo(new ArrayBuffer(16));
  } catch {
    threw = true;
  }
  check('container: a non-SDMO buffer is refused', threw);
  // negative control: a perturbed value table changes the logits
  const pm = SdmOnlyModel.fromSdmo(synthetic({}));
  const base = Float32Array.from(pm.step(ids).logits);
  pm.stores[0].values.w.forEach((v, i, a) => (a[i] = -v));
  pm._cache = null;
  const pert = Float32Array.from(pm.step(ids).logits);
  check('negative control: negating one value table moves the logits', maxAbsDiff(base, pert) > 1e-3, `max abs ${maxAbsDiff(base, pert).toFixed(3)}`);
  // the sampler
  const lg = Float32Array.from({ length: 50 }, (_, i) => Math.sin(i * 1.7) * 3);
  let am = 0;
  for (let i = 1; i < lg.length; i++) if (lg[i] > lg[am]) am = i;
  check('temperature 0 is the argmax', sampleNext(lg, { temperature: 0 }, mulberry32(1)).id === am);
  const draws = (s) => Array.from({ length: 20 }, ((rng) => () => sampleNext(lg, { temperature: 1, topK: 5 }, rng).id)(mulberry32(s)));
  const t5 = new Set(topK(lg, 5));
  const d1 = draws(7);
  check('seeded draws repeat, and top-k keeps the draws inside the top k', d1.join() === draws(7).join() && d1.every((x) => t5.has(x)));
  const gen = generate(SdmOnlyModel.fromSdmo(synthetic({})), [1, 2, 3], { temperature: 0, maxTokens: 8 }, { stopId: -1 });
  check('generate: greedy continuation has the asked length', gen.length === 8);
  const ok = results.filter(Boolean).length;
  console.log(`${ok} of ${results.length} checks pass`);
  return ok === results.length;
}

// ---------------------------------------------------------------- node CLI

async function loadModel(path) {
  const fs = await import('node:fs');
  const nodePath = await import('node:path');
  if (path.endsWith('.json')) {
    const meta = JSON.parse(fs.readFileSync(path, 'utf8'));
    const b = fs.readFileSync(nodePath.join(nodePath.dirname(path), meta.bin));
    return SdmOnlyModel.fromPair(meta, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  }
  const b = fs.readFileSync(path);
  return SdmOnlyModel.fromSdmo(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

async function parityMain(refPath, modelPath, limit) {
  const fs = await import('node:fs');
  const nodePath = await import('node:path');
  const ref = JSON.parse(fs.readFileSync(refPath, 'utf8'));
  const dir = nodePath.dirname(refPath);
  const rd = (name, Ctor) => {
    const b = fs.readFileSync(nodePath.join(dir, ref.files[name]));
    return new Ctor(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  };
  const toks = rd('tokens', Uint32Array);
  const top1 = rd('top1', Int32Array);
  const lse = rd('lse', Float32Array);
  const nll = rd('nll', Float32Array);
  const samp = rd('sampled', Float32Array);
  const tbytes = rd('bytes', Int32Array);
  const sampleIds = ref.sample_ids;
  const S = sampleIds.length;
  const T = ref.T;
  const N = Math.min(ref.n_windows, limit || ref.n_windows);
  const model = await loadModel(modelPath);
  let pos = 0;
  let agree = 0;
  let sumAbs = 0;
  let nAbs = 0;
  let maxAbs = 0;
  let natsRef = 0;
  let natsJs = 0;
  let bytes = 0;
  let lseAbs = 0;
  const t0 = Date.now();
  for (let w = 0; w < N; w++) {
    const st = model.newState();
    for (let p = 0; p < T; p++) {
      const { logits } = model.stepState(st, toks[w * (T + 1) + p]);
      const tgt = toks[w * (T + 1) + p + 1];
      if (tgt === 0) continue;
      const r = w * T + p;
      let am = 0;
      let mx = logits[0];
      for (let v = 1; v < logits.length; v++) if (logits[v] > mx) {
        mx = logits[v];
        am = v;
      }
      let se = 0;
      for (let v = 0; v < logits.length; v++) se += Math.exp(logits[v] - mx);
      const lseJs = mx + Math.log(se);
      if (am === top1[r]) agree++;
      for (let j = 0; j < S; j++) {
        const dlt = Math.abs(logits[sampleIds[j]] - samp[r * S + j]);
        sumAbs += dlt;
        if (dlt > maxAbs) maxAbs = dlt;
      }
      nAbs += S;
      lseAbs += Math.abs(lseJs - lse[r]);
      natsJs += lseJs - logits[tgt];
      natsRef += nll[r];
      bytes += tbytes[r];
      pos++;
    }
  }
  const secs = (Date.now() - t0) / 1000;
  const out = {
    model: modelPath, windows: N, positions: pos, top1_agreement: agree / pos, mean_abs_logit_diff: sumAbs / nAbs,
    max_abs_logit_diff_sampled: maxAbs, sampled_ids_per_position: S, mean_abs_lse_diff: lseAbs / pos,
    bpb_pytorch_fp32: natsRef / (Math.LN2 * bytes), bpb_js: natsJs / (Math.LN2 * bytes),
    js_seconds: secs, js_tokens_per_s: (N * T) / secs,
  };
  console.log(JSON.stringify(out));
  return out;
}

async function benchMain(modelPath, nPrompt = 64, nGen = 32) {
  const model = await loadModel(modelPath);
  const st = model.newState();
  const ids = Array.from({ length: nPrompt }, (_, i) => (i * 7919 + 11) % model.V);
  let t0 = performance.now();
  for (const id of ids) model.push(st, id);
  const pushMs = (performance.now() - t0) / nPrompt;
  model.forward(st); // warm
  t0 = performance.now();
  for (let i = 0; i < 8; i++) model.forward(st, { only: [0] });
  const bodyMs = (performance.now() - t0) / 8;
  t0 = performance.now();
  let id = 0;
  for (let i = 0; i < nGen; i++) {
    const { logits } = model.stepState(st, id);
    id = sampleNext(logits, { temperature: 0 }, Math.random).id;
  }
  const stepMs = (performance.now() - t0) / nGen;
  const out = {
    model: modelPath, d: model.d, V: model.V, hops: model.hops, n_sub: model.nSub, k: model.k,
    prompt_push_ms_per_token: +pushMs.toFixed(3), body_ms_per_token: +bodyMs.toFixed(3),
    full_step_ms_per_token: +stepMs.toFixed(2), tokens_per_s: +(1000 / stepMs).toFixed(1), node: process.version,
  };
  console.log(JSON.stringify(out));
  return out;
}

const HELP = `track4_sdmonly_engine.mjs - the SDM-only forward pass in plain JavaScript (node and browser).

  node track4_sdmonly_engine.mjs --selftest
  node track4_sdmonly_engine.mjs --info  <model.sdmo>
  node track4_sdmonly_engine.mjs --bench <model.sdmo> [prompt_tokens] [generated_tokens]
  node track4_sdmonly_engine.mjs --parity <ref.json> <model.sdmo> [max_windows]

In code (a Worker or a page):
  import { SdmOnlyModel, sampleNext, mulberry32 } from './track4_sdmonly_engine.mjs';
  const model = SdmOnlyModel.fromSdmo(await (await fetch('model.sdmo')).arrayBuffer());
  const st = model.newState();  for (const id of promptIds) model.push(st, id);
  const { logits } = model.forward(st);  // Float32Array(V), the next token's logits

NEXT -> python3 track4_sdmonly_export_check.py --help`;

async function main(argv) {
  const [cmd, ...rest] = argv;
  if (cmd === '--selftest') process.exit(selftest() ? 0 : 1);
  else if (cmd === '--info') {
    const fs = await import('node:fs');
    const b = fs.readFileSync(rest[0]);
    const { meta } = parseSdmo(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    console.log(JSON.stringify({ ...meta, tensors: meta.tensors.map((t) => `${t.name} ${t.dtype} [${t.shape}]`) }, null, 1));
  } else if (cmd === '--bench') await benchMain(rest[0], +(rest[1] || 64), +(rest[2] || 32));
  else if (cmd === '--parity') await parityMain(rest[0], rest[1], +(rest[2] || 0));
  else console.log(HELP);
}

if (typeof process !== 'undefined' && process.argv && process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  main(process.argv.slice(2)).catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
