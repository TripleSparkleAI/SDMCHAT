#!/usr/bin/env python3
"""shots.py - lane SDMSCORE: open #/sdmchat, pick sdmwide768 in the model dropdown, ask one question, and save shots
at 1440 and 390. Usage: python3 shots.py --base http://localhost:5240
The reply is read from the DOM once its footer appears; the run reports the reply, its footer and whether it loops
(src/engine/sdmchat.js loopStats is the site's rule; here a word 3 times in a row or a word 3-gram 3 times)."""
import argparse
import json
import os
import re
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def loops(text):
    w = re.findall(r"[a-z0-9']+", text.lower())
    grams = {}
    for i in range(len(w) - 2):
        g = " ".join(w[i:i + 3])
        grams[g] = grams.get(g, 0) + 1
    rep = any(w[i] == w[i - 1] == w[i - 2] for i in range(2, len(w)))
    return rep or max(grams.values(), default=0) >= 3


def run(page, base, width, height, out):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(base + "/#/sdmchat", wait_until="domcontentloaded")
    sel = page.locator("select").filter(has=page.locator("option[value='sdmwide768']")).first
    sel.wait_for(timeout=60000)
    t0 = time.time()
    while sel.is_disabled() and time.time() - t0 < 300:
        time.sleep(1)
    sel.select_option("sdmwide768")
    ta = page.locator("textarea").first
    t0 = time.time()
    while ta.is_disabled() and time.time() - t0 < 300:
        time.sleep(1)
    # the page answers a sample question on its own when a model loads; wait for that reply to finish first
    t0 = time.time()
    while time.time() - t0 < 300 and page.evaluate("() => !!document.querySelector('.ck-msg--live')"):
        time.sleep(1)
    n0 = page.evaluate("() => document.querySelectorAll('.ck-msg--model').length")
    ta.fill("Why is the sky blue?")
    page.locator("button", has_text=re.compile("^send", re.I)).first.click()
    t0 = time.time()
    reply = None
    while time.time() - t0 < 300:
        r = page.evaluate("""() => { const m = [...document.querySelectorAll('.ck-msg--model')].pop(); if (!m) return null;
          const r = m.querySelector('.ck-reply'); return { n: document.querySelectorAll('.ck-msg--model').length,
          live: m.classList.contains('ck-msg--live'), text: r ? r.innerText : '',
          foot: (m.querySelector('.ck-msg__line') || {}).innerText || '' }; }""")
        if r and r["n"] > n0 and not r["live"] and r["foot"]:
            reply = r
            break
        time.sleep(1)
    page.locator(".ck").first.scroll_into_view_if_needed()
    page.screenshot(path=out)
    return {"width": width, "reply": reply and reply["text"], "foot": reply and reply["foot"],
            "loops": bool(reply and loops(reply["text"])),
            "selected": sel.input_value()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "shots"), exist_ok=True)
    res = []
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for w, h in ((1440, 900), (390, 844)):
            page = br.new_page()
            res.append(run(page, a.base, w, h, os.path.join(HERE, "shots", f"sdmchat_sdmwide768_{w}.png")))
            page.close()
        br.close()
    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": res},
              open(os.path.join(HERE, "shots_result.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
