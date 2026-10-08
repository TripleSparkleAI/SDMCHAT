"""EPICS corpus: public-domain epic poems from Project Gutenberg, downloaded, stripped of the Gutenberg wrapper,
and recorded with full provenance (lane EPICS, 2026-10-02).

<claudes_code_comments>
** Function List **
catalog_rows() - the Project Gutenberg catalog CSV (raw/pg_catalog.csv), keyed by ebook number
death_years(authors) - every death year named in a catalog Authors field
fetch(work) - download raw/<slug>.pg<id>.txt once (kept bytes are what the sha256 covers)
strip_gutenberg(text) - the text between the START and END markers, BOM removed, CRLF -> LF
extract_section(text, sec) - an optional [start, end) slice by heading (the Shah Nameh inside a collection)
licence(work, cat) - the licence basis: US (published before 1931) and life + 70 (every named person dead before 1956)
main() - build clean/<slug>.txt, the manifest EPICS_MANIFEST.json and samples/<slug>.txt (first 1,500 chars)

** Technical Review **
- Source of every bibliographic field: track4_epics_works.json (the title, the translator, the year of first
  publication of the translation) plus the Gutenberg catalog row (the catalog's own Authors string, Issued date and
  language, copied verbatim). Death years for the life + 70 test are parsed from the catalog Authors string, never typed.
- The raw download is kept byte for byte in raw/ (gitignored) and its sha256 is recorded with the download time. The
  clean text is a pure function of the raw bytes: utf-8 decode, BOM dropped, CRLF and lone CR to LF, the slice between
  the "*** START OF" and "*** END OF" marker lines, leading and trailing blank lines dropped. Nothing inside is edited:
  the translators' introductions, notes and footnotes stay, so the clean text is the edition, not only the poem.
- A re-run never re-downloads a file already in raw/ (pass --refresh to fetch again); a refresh with a different sha256
  is reported, not hidden.
</claudes_code_comments>
"""
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
CLEAN = os.path.join(HERE, "clean")
SAMPLES = os.path.join(HERE, "samples")
CLEANING = ("utf-8 decode; BOM removed; CRLF and CR to LF; kept the lines strictly between the '*** START OF' and "
            "'*** END OF' Project Gutenberg marker lines; leading and trailing blank lines removed; nothing inside edited")


def catalog_rows():
    path = os.path.join(RAW, "pg_catalog.csv")
    with open(path, encoding="utf-8") as f:
        return {r["Text#"]: r for r in csv.DictReader(f)}


def death_years(authors):
    return [int(y) for y in re.findall(r"-(\d{4})\b", authors)]


def fetch(w, refresh=False):
    os.makedirs(RAW, exist_ok=True)
    url = f"https://www.gutenberg.org/cache/epub/{w['pg']}/pg{w['pg']}.txt"
    path = os.path.join(RAW, f"{w['slug']}.pg{w['pg']}.txt")
    meta_path = path + ".download.json"
    if refresh or not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "settle-epics-corpus/1 (research; one file per work)"})
        data = urllib.request.urlopen(req, timeout=120).read()
        old = hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None
        open(path, "wb").write(data)
        meta = {"url": url, "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if old and old != meta["sha256"]:
            meta["changed_from_sha256"] = old
            print("CHANGED", w["slug"], old, "->", meta["sha256"])
        json.dump(meta, open(meta_path, "w"), indent=1)
        time.sleep(1.0)
    meta = json.load(open(meta_path))
    data = open(path, "rb").read()
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise SystemExit(f"{path}: bytes on disk do not match the recorded sha256")
    return data, meta


def strip_gutenberg(text):
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    s = next(i for i, ln in enumerate(lines) if ln.startswith("*** START OF"))
    e = next(i for i, ln in enumerate(lines) if ln.startswith("*** END OF"))
    body = lines[s + 1:e]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body) + "\n"


def extract_section(text, sec):
    starts = [m.start() for m in re.finditer(r"^" + re.escape(sec["start"]) + r"\s*$", text, re.M)]
    if not starts:
        raise SystemExit(f"section start {sec['start']!r} not found")
    a = starts[sec.get("start_occurrence", 1) - 1]
    m = re.search(r"^" + re.escape(sec["end"]) + r"\s*$", text[a + 1:], re.M)
    if not m:
        raise SystemExit(f"section end {sec['end']!r} not found")
    return text[a:a + 1 + m.start()].rstrip("\n") + "\n"


def licence(w, cat):
    deaths = death_years(cat["Authors"])
    last = max(deaths) if deaths else None
    us = w["pub_year"] < 1931
    life70 = last is not None and last < 1956
    return {
        "us_public_domain": us,
        "us_basis": f"published {w['pub_year']}, before 1931 (US works published before 1931 are in the public domain as of 2026)" if us else "NOT ESTABLISHED",
        "life_plus_70_public_domain": life70,
        "life_plus_70_basis": (f"every person named in the Gutenberg catalog entry died by {last}" if life70 else
                               f"a person named in the Gutenberg catalog entry died in {last}; not public domain under life + 70 until {last + 71}" if last else "no death year in the catalog entry"),
        "gutenberg_licence": "the Project Gutenberg wrapper and trademark text are removed; the remaining text is public domain in the US per Project Gutenberg",
    }


def main():
    refresh = "--refresh" in sys.argv
    works = json.load(open(os.path.join(HERE, "track4_epics_works.json")))
    cat = catalog_rows()
    cat_meta = {"file": "raw/pg_catalog.csv", "url": "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
                "sha256": hashlib.sha256(open(os.path.join(RAW, "pg_catalog.csv"), "rb").read()).hexdigest()}
    os.makedirs(CLEAN, exist_ok=True)
    os.makedirs(SAMPLES, exist_ok=True)
    out = []
    for w in works:
        row = cat[str(w["pg"])]
        data, meta = fetch(w, refresh)
        text = strip_gutenberg(data.decode("utf-8"))
        if w.get("section"):
            text = extract_section(text, w["section"])
        open(os.path.join(CLEAN, f"{w['slug']}.txt"), "w", encoding="utf-8").write(text)
        open(os.path.join(SAMPLES, f"{w['slug']}.txt"), "w", encoding="utf-8").write(text[:1500])
        rec = {
            "slug": w["slug"], "title": w["title"], "original": w["original"], "translator": w.get("translator"),
            "editor": w.get("editor"), "translation_first_published": w["pub_year"], "form": w["form"],
            "alternate_of": w.get("alternate_of"),
            "source": {"site": "Project Gutenberg", "ebook": w["pg"], "url": meta["url"],
                       "landing": f"https://www.gutenberg.org/ebooks/{w['pg']}", "catalog_title": row["Title"],
                       "catalog_authors": row["Authors"], "catalog_issued": row["Issued"], "catalog_language": row["Language"]},
            "raw": {"file": f"raw/{w['slug']}.pg{w['pg']}.txt", "sha256": meta["sha256"], "bytes": meta["bytes"],
                    "downloaded_utc": meta["downloaded_utc"]},
            "clean": {"file": f"clean/{w['slug']}.txt", "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                      "chars": len(text), "utf8_bytes": len(text.encode("utf-8")), "lines": text.count("\n"),
                      "cleaning": CLEANING + (f"; then the section from the heading {w['section']['start']!r} up to "
                                               f"the heading {w['section']['end']!r}" if w.get("section") else "")},
            "licence": licence(w, row),
        }
        out.append(rec)
        print(f"{w['slug']:30s} {rec['clean']['chars']:>10,d} chars  US={rec['licence']['us_public_domain']} L70={rec['licence']['life_plus_70_public_domain']}")
    man = {"name": "EPICS corpus", "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "builder": "experiments/track4/epics/track4_epics_build_corpus.py", "catalog": cat_meta, "works": out,
           "total_chars": sum(r["clean"]["chars"] for r in out), "total_utf8_bytes": sum(r["clean"]["utf8_bytes"] for r in out)}
    json.dump(man, open(os.path.join(HERE, "EPICS_MANIFEST.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("total", f"{man['total_chars']:,d}", "chars,", len(out), "works")


if __name__ == "__main__":
    main()
