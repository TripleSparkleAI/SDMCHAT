// loopcheck.mjs - lane SDMCHATS: does each chat surface's model loop? Runs the site's own engine (src/engine/sdmchat.js)
// on each surface's prompt template and sampling settings, several prompts and seeds, and counts looping replies.
//
// <claudes_code_comments>
// ** Function List **
// load(dir, key)            - one exported model + the tokenizer, from public/data/sdmchat or private/wlg
// gen(model, ...)           - the worker's generation loop without the page: nudge, minTokens, stop strings, end of text
// SURFACES                  - each page's prompt template, sample prompts and sampling settings, read from the page code
// main()                    - every surface x model x prompt x seed; loop rate, distinct-2, mean length; JSON + samples
//
// ** Technical Review **
// - Loop metrics come from src/engine/sdmchat.js loopStats (the same function the site tests pin), so the site and
//   this script agree on what a loop is: a token 3-gram repeated 3 or more times, or one word 3 times in a row.
// - Usage: node loopcheck.mjs [--models sdmread,wide512] [--surfaces chat,poem,wlg] [--seeds 3] [--out file.json]
//   [--params old|new] (old = the settings the pages had before lane SDMCHATS, new = SAMPLING in sdmchat.js)
// </claudes_code_comments>
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.resolve(HERE, '../../../SETTLE/settle-site');
const E = await import(path.join(SITE, 'src/engine/sdmchat.js'));
const S = await import(path.join(SITE, 'src/sdmchat/samples.js'));
const W = await import(path.join(SITE, 'src/wlg/wlg.js'));

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? process.argv[i + 1] : d;
};

let TOK = null;
let WORDS = null;
function load(dir, key) {
  if (!TOK) {
    TOK = new E.Tokenizer(JSON.parse(fs.readFileSync(path.join(SITE, 'public/data/sdmchat/tokenizer.json'), 'utf8')));
    WORDS = E.buildWordIndex(TOK);
  }
  const meta = JSON.parse(fs.readFileSync(path.join(dir, `${key}.json`), 'utf8'));
  const b = fs.readFileSync(path.join(dir, meta.bin));
  return new E.SdmModel(meta, b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
}

function gen(model, prompt, params, { stopStrings = [], nudge = null } = {}) {
  const tok = TOK;
  const ids = [tok.bos, ...tok.encode(prompt)];
  const n0 = ids.length;
  const rng = E.mulberry32(params.seed >>> 0);
  const scratch = new Float32Array(model.V);
  const work = new Float32Array(model.V);
  const nlId = tok.encode('\n')[0];
  const nnId = tok.encode('\n\n')[0];
  let lineTok = 0;
  let lines = 0;
  let text = '';
  let reason = 'max tokens';
  for (let n = 0; n < params.maxTokens; n++) {
    const { logits } = model.step(ids);
    let src = logits;
    if (nudge && nudge.strength > 0 && lineTok >= nudge.lineTokens) {
      work.set(logits);
      const target = nudge.stanzaEvery > 0 && lines > 0 && (lines + 1) % nudge.stanzaEvery === 0 ? nnId : nlId;
      work[target] += nudge.strength + 0.5 * (lineTok - nudge.lineTokens);
      src = work;
    }
    if (params.minTokens && n < params.minTokens) {
      if (src === logits) {
        work.set(logits);
        src = work;
      }
      src[tok.eos] = -1e9;
      for (const sid of tok.addedIds) src[sid] = -1e9;
    }
    const { id } = E.sampleNext(src, params, rng, ids.slice(n0 - 1), scratch, WORDS);
    if (id === tok.eos || tok.addedIds.has(id)) {
      reason = 'end';
      break;
    }
    ids.push(id);
    const tt = tok.tokenText(id);
    text += tt;
    if (tt.includes('\n')) {
      lines += (tt.match(/\n/g) || []).length;
      lineTok = 0;
    } else lineTok++;
    const hit = stopStrings.find((s) => text.includes(s));
    if (hit) {
      text = text.slice(0, text.indexOf(hit));
      reason = 'stop';
      break;
    }
  }
  return { text, ids: ids.slice(n0), reason };
}

const NEW = E.SAMPLING;
const OLD = {
  // the settings the three pages had before lane SDMCHATS (and their old chat stop strings, below)
  chat: { temperature: 0.8, topK: 50, topP: 0.95, repetitionPenalty: 1.1, maxTokens: 160 },
  poem: { temperature: 0.9, topK: 80, topP: 0.95, repetitionPenalty: 1.15, maxTokens: 120, minTokens: 48 },
  wlg: { temperature: 0.8, topK: 50, topP: 0.95, repetitionPenalty: 1.1, maxTokens: 160 },
};
const SURFACES = {
  chat: { prompts: S.CHAT_SAMPLES, prompt: (q) => S.chatPrompt([], q), stop: S.CHAT_STOP },
  poem: { prompts: S.POEM_TITLES, prompt: (q) => S.poemPrompt('title', q, true), nudge: { strength: 3, lineTokens: 8, stanzaEvery: 4 } },
  wlg: { prompts: W.SAMPLES, prompt: (q) => W.wlgPrompt([], q), stop: W.STOP_STRINGS },
};

const models = arg('models', 'sdmread').split(',');
const surfaces = arg('surfaces', 'chat,poem').split(',');
const nSeeds = Number(arg('seeds', '3'));
const which = arg('params', 'new');
const out = { which, override: arg('override', '{}'), rows: [], samples: [] };
for (const key of models) {
  const dir = key === 'weirdguy' ? path.join(SITE, 'private/wlg') : path.join(SITE, 'public/data/sdmchat');
  if (!fs.existsSync(path.join(dir, `${key}.json`))) {
    console.log(`${key}: not exported, skipped`);
    continue;
  }
  const model = load(dir, key);
  for (const sf of surfaces) {
    const spec = which === 'old' && sf === 'chat' ? { ...SURFACES.chat, stop: ['\nUser:', '\nAssistant:'] } : SURFACES[sf];
    const base = { ...(which === 'old' ? OLD[sf] : NEW[sf]), ...JSON.parse(arg('override', '{}')) };
    let loops = 0, n = 0, d2 = 0, len = 0;
    const t0 = Date.now();
    for (const p of spec.prompts) {
      for (let s = 1; s <= nSeeds; s++) {
        const r = gen(model, spec.prompt(p), { ...base, seed: s }, { stopStrings: spec.stop, nudge: spec.nudge });
        const st = E.loopStats(TOK, r.ids);
        loops += st.loops ? 1 : 0;
        d2 += st.distinct2;
        len += r.ids.length;
        n++;
        if (s === 1 || st.loops) out.samples.push({ model: key, surface: sf, prompt: p, seed: s, loops: st.loops, why: st.why, text: r.text });
      }
    }
    const row = { model: key, surface: sf, replies: n, looping: loops, loopRate: +(loops / n).toFixed(3), distinct2: +(d2 / n).toFixed(3), meanTokens: +(len / n).toFixed(1), secs: (Date.now() - t0) / 1000 };
    out.rows.push(row);
    console.log(JSON.stringify(row));
  }
}
const of = arg('out', null);
if (of) fs.writeFileSync(of, JSON.stringify(out, null, 1));
