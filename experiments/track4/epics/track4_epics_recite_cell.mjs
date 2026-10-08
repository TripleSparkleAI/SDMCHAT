// EPICS RECITE - one cell of the recitation curve: the poem lab's SDM (SETTLE/settle-site/src/poemlab/poemsdm.js)
// learns the first N characters of an epic, then is quantised and asked to recite. Prints one JSON line.
//
// <claudes_code_comments>
// ** Function List **
// arg(name, dflt)              - a --name value from the command line
// measure(sdm, enc, alpha, t)  - teacher-forced accuracy, the recital from the opening, and recitals from random cues
// main                         - learn, then for each counter width: quantise a copy, measure, report bytes
//
// ** Technical Review **
// - THE MEMORY: PoemSdm with binding (D 256, the address's own bits as the key), error-driven writes ('onError') for up
//   to --passes passes, radius 111 (about 2% of the hard locations answer each address). The hard locations' addresses
//   come from the seed, so what a page must download is the counters alone: M x 256 x bits / 8 bytes. Write counts
//   are not read when the store binds (read() uses the bound counter sums only), so they are not shipped.
// - THE RECITAL: greedy, reading its own output. From the opening: the cue is the first 48 characters and the memory
//   writes the rest; firstError is how many characters come out right before the first mistake. From random cues: K
//   positions drawn with a seeded generator, each cued with the 48 characters before it, reciting up to 400.
// - WORD-PERFECT means the opening recital reaches the end of the N characters with no mistake.
// </claudes_code_comments>

import { learn, recite, teacherForced, wordsMatched, mulberry32, PoemSdm, ContextAddress } from '../../../SETTLE/settle-site/src/poemlab/poemsdm.js';
import { readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

const argv = process.argv.slice(2);
const arg = (n, d) => {
  const i = argv.indexOf('--' + n);
  return i < 0 ? d : argv[i + 1];
};
const file = arg('file');
const N = +arg('n', 8000);
const M = +arg('M', 4096);
const radius = +arg('radius', 111);
const scales = arg('scales', '48').split(',').map(Number);
const passes = +arg('passes', 4);
const bitsList = arg('bits', '16,8,4,2,1').split(',').map(Number);
const K = +arg('cues', 20);
const hashBits = +arg('hash', 32);
const CUE = 48;

const full = readFileSync(file, 'utf8');
const text = full.slice(0, N);

function measure(sdm, enc, alpha) {
  const tf = teacherForced(sdm, enc, alpha, text, 1, text.length);
  const open = recite(sdm, enc, alpha, text.slice(0, CUE), text.length - CUE, text, { stopAtError: true });
  const openFirst = open.firstErr < 0 ? text.length - CUE : open.firstErr;
  const rng = mulberry32(99);
  const cue = [];
  for (let k = 0; k < K; k++) {
    const p = CUE + Math.floor(rng() * Math.max(1, text.length - CUE - 1));
    const L = Math.min(400, text.length - p);
    const r = recite(sdm, enc, alpha, text.slice(p - CUE, p), L, text.slice(p - CUE, p + L), { stopAtError: true });
    cue.push(r.firstErr < 0 ? L : r.firstErr);
  }
  cue.sort((a, b) => a - b);
  return {
    teacherForcedAcc: +tf.acc.toFixed(6),
    openingFirstError: openFirst,
    openingWords: wordsMatched(open.text, text, CUE),
    wordPerfect: openFirst >= text.length - CUE,
    cueMedianFirstError: cue[Math.floor(K / 2)],
    cueFullFraction: cue.filter((x, i) => x >= Math.min(400, text.length - CUE)).length / K,
    cueFirstErrors: cue,
  };
}

const t0 = Date.now();
const r = learn(text, { M, radius, scales, writeMode: 'onError', passes, seed: 1, enc: new ContextAddress({ scales, seed: 7, hashBits }) });
const learnMs = Date.now() - t0;
const base = r.sdm;
const out = { file: file.split('/').pop(), N, M, radius, scales, passes, hashBits, D: base.D, alphabet: r.alpha.size, writes: r.writes,
  meanActive: +r.meanActive.toFixed(1), learnSeconds: +(learnMs / 1000).toFixed(1), cells: [] };
for (const bits of bitsList) {
  const sdm = Object.assign(Object.create(PoemSdm.prototype), base);
  sdm.counters = base.counters.slice();
  sdm.act = new Int32Array(base.M);
  sdm.quantise(bits);
  const bytes = (sdm.M * sdm.D * bits) / 8;
  // the shipped form: counters packed to `bits` per counter, then gzip
  const levels = bits >= 16 ? null : bits === 1 ? 1 : (1 << (bits - 1)) - 1;
  let packed;
  if (bits === 16) packed = Buffer.from(sdm.counters.buffer.slice(0));
  else {
    let mx = 1;
    for (const v of sdm.counters) if (Math.abs(v) > mx) mx = Math.abs(v);
    const step = bits === 1 ? 1 : mx / levels;
    packed = Buffer.alloc(Math.ceil((sdm.counters.length * bits) / 8));
    let bitpos = 0;
    for (const v of sdm.counters) {
      const q = bits === 1 ? (v > 0 ? 1 : 0) : Math.round(v / step) + levels;
      for (let b = 0; b < bits; b++, bitpos++) if ((q >> b) & 1) packed[bitpos >> 3] |= 1 << (bitpos & 7);
    }
  }
  const t1 = Date.now();
  const m = measure(sdm, r.enc, r.alpha);
  out.cells.push({ bits, bytes, gzipBytes: gzipSync(packed, { level: 9 }).length, bytesPerChar: +(bytes / N).toFixed(2), measureSeconds: +((Date.now() - t1) / 1000).toFixed(1), ...m });
}
console.log(JSON.stringify(out));
