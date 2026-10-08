#!/usr/bin/env python3
"""browserspeed.py - lane SDMSCORE: tokens per second of exported SDM chat models in a real browser, through the
site's own worker (src/engine/sdmchat.worker.js), interleaved so both models see the same machine state.

<claudes_code_comments>
** Function List **
stamp()            - UTC, load average and macOS power mode (the measurement laws)
run_once(page, key, seed) - one generate message to the worker; returns the worker's own n and tokens per second
main()             - load every model once, then alternate them for --rounds rounds; median and minimum time per token

** Technical Review **
- Usage: python3 browserspeed.py --base http://localhost:5240 --keys sdmwide512,sdmwide768 [--rounds 6] [--out f.json]
- The page opens the dev server's root, then creates one module Worker from /src/engine/sdmchat.worker.js and talks
  to it with the same messages the chat page sends ('load', then 'generate' with SAMPLING.chat and a fixed seed).
  The tokens per second is the worker's own `done.tps` (n / seconds of the generate loop), so the figure is the one
  the chat page shows in its footer.
- The arms alternate (A B A B ...) so a load change on the machine lands on both. The report gives each model's
  median and best tokens per second and the ratio of medians; the minimum time per token is the least noisy figure.
- Sealed SL3 (SETTLE_CAMPAIGN_2026-09-30.md, 05:26Z): SWL's export runs at no less than a third of SW's tokens per
  second in the browser.
</claudes_code_comments>
"""
import argparse
import json
import os
import statistics
import subprocess
import time

from playwright.sync_api import sync_playwright

PROMPTS = ["User: Write a poem about the sea.\nAssistant:", "The water cycle describes how water moves",
           "User: What is a black hole?\nAssistant:"]
CHAT = {"temperature": 0.8, "topK": 50, "topP": 0.95, "repetitionPenalty": 1.25, "repeatWindow": 128,
        "noRepeatNgram": 3, "wordPenalty": 0.4, "noAdjacentWord": True, "maxTokens": 96}


def stamp():
    pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    mode = next((ln.split()[-1] for ln in pm.splitlines() if "powermode" in ln), "unknown")
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load": list(os.getloadavg()),
            "ncpu": os.cpu_count(), "powermode": mode}


JS_SETUP = """async (keys) => {
  window.__w = new Worker('/src/engine/sdmchat.worker.js', { type: 'module' });
  window.__msgs = [];
  window.__w.onmessage = (e) => { if (e.data.type !== 'token' && e.data.type !== 'progress') window.__msgs.push(e.data); };
  const out = {};
  for (const key of keys) {
    const t0 = performance.now();
    window.__w.postMessage({ type: 'load', key, base: '/data/sdmchat' });
    await new Promise((res, rej) => { const iv = setInterval(() => {
      const m = window.__msgs.find((x) => (x.type === 'ready' && x.key === key) || x.type === 'error');
      if (m) { clearInterval(iv); m.type === 'error' ? rej(new Error(m.message)) : res(); } }, 50); });
    out[key] = { load_ms: performance.now() - t0 };
  }
  return out;
}"""

JS_GEN = """async ([key, prompt, params, runId]) => {
  window.__w.postMessage({ type: 'generate', runId, key, base: '/data/sdmchat', prompt, params, stopStrings: [] });
  return await new Promise((res) => { const iv = setInterval(() => {
    const m = window.__msgs.find((x) => x.runId === runId && (x.type === 'done' || x.type === 'error'));
    if (m) { clearInterval(iv); res(m); } }, 20); });
}"""


def run_once(page, key, prompt, seed, rid):
    m = page.evaluate(JS_GEN, [key, prompt, {**CHAT, "seed": seed}, rid])
    if m["type"] == "error":
        raise RuntimeError(m["message"])
    return {"n": m["n"], "tps": m["tps"], "reason": m["reason"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--keys", required=True)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "browserspeed_result.json"))
    a = p.parse_args()
    keys = a.keys.split(",")
    res = {"base": a.base, "keys": keys, "stamp_start": stamp(), "params": CHAT, "runs": {k: [] for k in keys}}
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        page = br.new_page()
        page.goto(a.base + "/", wait_until="domcontentloaded")
        res["load"] = page.evaluate(JS_SETUP, keys)
        rid = 0
        for k in keys:          # warm-up, discarded
            rid += 1
            run_once(page, k, PROMPTS[0], 99, f"w{rid}")
        for r in range(a.rounds):
            order = keys if r % 2 == 0 else keys[::-1]
            for k in order:
                rid += 1
                x = run_once(page, k, PROMPTS[r % len(PROMPTS)], r + 1, f"r{rid}")
                res["runs"][k].append(x)
                print(r, k, x, flush=True)
        br.close()
    res["stamp_end"] = stamp()
    res["summary"] = {k: {"median_tps": statistics.median(x["tps"] for x in v), "best_tps": max(x["tps"] for x in v),
                          "tokens": sum(x["n"] for x in v)} for k, v in res["runs"].items()}
    if len(keys) == 2:
        a0, a1 = (res["summary"][k] for k in keys)
        res["summary"]["ratio_median"] = a1["median_tps"] / a0["median_tps"]
        res["summary"]["ratio_best"] = a1["best_tps"] / a0["best_tps"]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
