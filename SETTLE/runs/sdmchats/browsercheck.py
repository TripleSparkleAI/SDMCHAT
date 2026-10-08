#!/usr/bin/env python3
"""browsercheck.py - lane SDMCHATS: open each SDM chat surface in a real browser, let the page run its own model, and
check that every reply finishes without looping. Screenshots at 1440 and 390.

<claudes_code_comments>
** Function List **
loops(text)              - the same loop rule as src/engine/sdmchat.js loopStats, on words: a word 3 times in a row,
                           or a word 3-gram 3 or more times
chat_page(page, route, prompts) - wait for the autostarted reply, then send each prompt through the composer
magic8(page)             - wait for the teacher model to load, ask the sample questions, read the answers
main()                   - every surface; samples and verdicts to browsercheck_result.json; shots to shots/

** Technical Review **
- Usage: python3 browsercheck.py --base http://localhost:5233 [--surfaces llm,poem,wlg,magic8] [--shots]
- The page decides the model and the sampling; this script only types, waits and reads, so a pass here is a pass of
  the shipped settings. A reply is read from the DOM once its footer appears (the turn is no longer live).
</claudes_code_comments>
"""
import argparse
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def loops(text):
    w = re.findall(r"[a-z0-9']+", text.lower())
    why = []
    run = 1
    for i in range(1, len(w)):
        run = run + 1 if w[i] == w[i - 1] else 1
        if run >= 3:
            why.append(f'"{w[i]}" x{run} in a row')
            break
    grams = {}
    for i in range(len(w) - 2):
        g = " ".join(w[i:i + 3])
        grams[g] = grams.get(g, 0) + 1
    worst = max(grams.items(), key=lambda kv: kv[1], default=(None, 0))
    if worst[1] >= 3:
        why.append(f'"{worst[0]}" x{worst[1]}')
    return why


def last_reply(page):
    return page.evaluate("""() => { const m = [...document.querySelectorAll('.ck-msg--model')].pop();
      if (!m) return null; const r = m.querySelector('.ck-reply'); return { live: m.classList.contains('ck-msg--live'),
      text: r ? r.innerText : '', foot: (m.querySelector('.ck-msg__line') || {}).innerText || '' }; }""")


def wait_reply(page, n_before, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        n = page.evaluate("() => document.querySelectorAll('.ck-msg--model').length")
        r = last_reply(page)
        if n > n_before and r and not r["live"] and r["foot"]:
            return r
        time.sleep(1)
    return {"text": (last_reply(page) or {}).get("text", ""), "foot": "TIMEOUT"}


def chat_page(page, base, route, prompts, shots, tag, autostart=True):
    out = []
    page.goto(f"{base}/#/{route}")
    if autostart:
        r = wait_reply(page, 0)
        first_user = page.evaluate("() => { const u = [...document.querySelectorAll('.ck-msg--user .ck-bubble')].pop(); return u ? u.innerText : null; }")
        out.append({"prompt": first_user, "autostart": True, **r, "loops": loops(r["text"])})
    for p in prompts:
        n = page.evaluate("() => document.querySelectorAll('.ck-msg--model').length")
        t0 = time.time()
        # the send is ignored until the model has loaded: retry until the user turn appears
        while time.time() - t0 < 300:
            page.fill("textarea", p)
            page.keyboard.press("Enter")
            time.sleep(1)
            if page.evaluate("() => document.querySelectorAll('.ck-msg--model').length") > n:
                break
            time.sleep(2)
        r = wait_reply(page, n)
        out.append({"prompt": p, **r, "loops": loops(r["text"])})
    status = page.evaluate("() => (document.querySelector('.ck-head__status') || {}).innerText || ''")
    if shots:
        page.screenshot(path=str(HERE / "shots" / f"{tag}_1440.png"))
    return {"status": status, "replies": out}


def magic8(page, base, shots):
    page.goto(f"{base}/#/magic8")
    t0 = time.time()
    while time.time() - t0 < 300:
        if page.locator(".m8-ask button[type=submit]").first.is_enabled():
            break
        time.sleep(1)
    res = []
    page.locator(".m8-check input").check()  # the slow teacher too, so the page names it and times it
    btns = page.locator(".m8-samples button")
    for i in range(min(4, btns.count())):
        q = btns.nth(i).inner_text()
        btns.nth(i).click()
        time.sleep(6)
        res.append({"q": q, "readout": page.evaluate("() => (document.querySelector('.m8-readout') || {}).innerText || ''")[:300],
                    "status": page.evaluate("() => (document.querySelector('.m8-status') || {}).innerText || ''")[:200]})
        print("magic8 |", q, "|", res[-1]["readout"].replace("\n", " / ")[:200], flush=True)
    if shots:
        page.screenshot(path=str(HERE / "shots" / "magic8_1440.png"))
    return {"answers": res}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--base", default="http://localhost:5233")
    a.add_argument("--surfaces", default="llm,poem,wlg,magic8")
    a.add_argument("--shots", action="store_true")
    a.add_argument("--out", default=str(HERE / "browsercheck_result.json"))
    args = a.parse_args()
    (HERE / "shots").mkdir(exist_ok=True)
    result = {"base": args.base, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "surfaces": {}}
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for s in args.surfaces.split(","):
            page = br.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(20000)
            if s == "llm":
                r = chat_page(page, args.base, "llm", ["Write a poem about the sea.", "What is a black hole?"], args.shots, "llm")
            elif s == "poem":
                r = chat_page(page, args.base, "poem", ["The Old Clock", "Winter Morning"], args.shots, "poem")
            elif s == "wlg":
                r = chat_page(page, args.base, "wlg", ["is curiosity just chemicals?", "draw me a little animal", "show me your face"], args.shots, "wlg", autostart=False)
            else:
                r = magic8(page, args.base, args.shots)
            if args.shots and s != "magic8":
                page.set_viewport_size({"width": 390, "height": 900})
                time.sleep(1)
                page.screenshot(path=str(HERE / "shots" / f"{s}_390.png"))
            result["surfaces"][s] = r
            for rep in r.get("replies", []):
                print(s, "|", rep["prompt"], "|", "LOOP " + "; ".join(rep["loops"]) if rep["loops"] else "ok", "|", rep["foot"], "|", rep["text"][:160].replace("\n", " / "))
            page.close()
        br.close()
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
