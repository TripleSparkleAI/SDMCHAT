"""SDMONLY loop check: do a checkpoint's chat replies and poems loop? The Python twin of loopcheck.mjs, for PyTorch
checkpoints (the SDM-only shape and the old shape), writing the same json schema as loops_sdmwide768chat.json.

<claudes_code_comments>
** Function List **
stamp() - utc, load average, host, torch, device, GPU and driver when there is one
run_check(models, surfaces, ...) - every model x surface x prompt x seed: loop rate, distinct-2, mean length, samples
selftest() - the loop detector fires on looping strings and not on normal ones; one end-to-end pass on a tiny model
main() - CLI

** Technical Review **
- The definition is the earlier one, unchanged (SETTLE/runs/sdmchats/loopcheck.mjs, lane SDMCHATS):
  each surface's prompts (chat: the 10 CHAT_SAMPLES of SETTLE/settle-site/src/sdmchat/samples.js; poem: the 5
  POEM_TITLES after the Tyger primer, with the line nudge), seeds 1..3, the surface's SAMPLING from
  src/engine/sdmchat.js (or the OLD settings with --params old), the chat stop strings, BOS + prompt, and a reply
  LOOPS when a token 3-gram occurs 3 or more times or one word occurs 3 times in a row (loopStats). So the default run
  is 30 chat replies and 15 poems. distinct-2 = distinct token bigrams / bigrams, averaged over replies; meanTokens =
  mean reply length in tokens (the stop token included, as loopcheck counts it).
- Prompts, SAMPLING, the stop strings, the nudge and the word index come from the site files through
  track4_sdmonly_generate.export_surfaces (node, on the M5), carried to the Spark as a json; nothing is retyped.
- Output keys match loopcheck.mjs: {which, override, rows: [{model, surface, replies, looping, loopRate, distinct2,
  meanTokens, secs}], samples: [{model, surface, prompt, seed, loops, why, text}]} (a sample is kept for seed 1 and
  for every looping reply). Added keys: engine, checkpoints, surfaces_json, stamp_start, stamp_end.
- The difference from the earlier numbers: loopcheck ran the site's int8 export in JavaScript; this runs the float32
  PyTorch checkpoint. Same sampler and same detector (the generate selftest checks both against the site's own
  functions), different logits. To compare with loops_sdmwide768chat.json exactly, run the exported model through
  loopcheck.mjs too.
- Speed: the seeds of one prompt run in lockstep (same prompt, so every live reply has the same length) through one
  batched forward; --one-at-a-time runs them singly. On CPU both give the same replies (selftest); on a GPU a batch
  of 3 and a batch of 1 can round differently, so the json records seeds_batched.
- The step-3 gate in the plan reads the chat row: looping 0 of 30.
Docs: PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md · track4_sdmonly_generate.py
</claudes_code_comments>
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track4_sdmonly_generate as G  # noqa: E402


def stamp(device):
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(), "torch": torch.__version__,
           "device": str(device)}
    try:
        out["load"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        out["load"] = None
    if device.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
        try:
            out["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            out["driver"] = "unknown"
    return out


def run_check(models, surf, surfaces=("chat", "poem"), n_seeds=3, which="new", override="{}", tok=None,
              max_prompts=None, log=print, batch_seeds=True):
    """models: list of (name, step_fn). Returns the loopcheck-schema dict."""
    words = G.Words(surf["word_of"])
    ov = json.loads(override)
    out = {"which": which, "override": override, "rows": [], "samples": []}
    for name, step in models:
        for sf in surfaces:
            spec = surf["surfaces"][sf]
            stop = spec.get("stop", [])
            if which == "old" and sf == "chat":
                stop = spec["oldStop"]
            base = {**(surf["old"][sf] if which == "old" else surf["sampling"][sf]), **ov}
            nudge = spec.get("nudge")
            loops = n = 0
            d2 = 0.0
            length = 0
            t0 = time.time()
            pairs = list(zip(spec["prompts"], spec["rendered"]))[:max_prompts]
            for p, rendered in pairs:
                seeds = list(range(1, n_seeds + 1))
                if batch_seeds:
                    reps = G.generate_many(step, tok, rendered, [{**base, "seed": s} for s in seeds], words, stop, nudge)
                else:
                    reps = [G.generate(step, tok, rendered, {**base, "seed": s}, words, stop, nudge) for s in seeds]
                for s, r in zip(seeds, reps):
                    st = G.loop_stats(tok, r["ids"])
                    loops += 1 if st["loops"] else 0
                    d2 += st["distinct2"]
                    length += len(r["ids"])
                    n += 1
                    if s == 1 or st["loops"]:
                        out["samples"].append({"model": name, "surface": sf, "prompt": p, "seed": s, "loops": st["loops"],
                                               "why": st["why"], "text": r["text"]})
            row = {"model": name, "surface": sf, "replies": n, "looping": loops, "loopRate": round(loops / n, 3),
                   "distinct2": round(d2 / n, 3), "meanTokens": round(length / n, 1), "secs": round(time.time() - t0, 2)}
            out["rows"].append(row)
            log(json.dumps(row))
    return out


def selftest():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    tok_path = os.path.join(G.DATA, "ds4_v4flash_tokenizer.json")
    if not os.path.exists(tok_path):
        print("SKIP: no tokenizer at " + tok_path + " (set SDMLLM_DATA)")
        return False
    tok = G.Tok(tok_path)
    loop_cases = {
        "a word three times in a row": "Write a poem about the sea. poem poem poem and the waves",
        "a 3-gram three times": "so the cat sat on the mat, so the cat sat by the door, so the cat sat down",
        "a long greedy loop": "a new way to be a new way to be a new way to be a few years",
    }
    for name, s in loop_cases.items():
        st = G.loop_stats(tok, tok.encode(s))
        check(f"fires on {name} ({st['why']})", st["loops"])
    normal = ("The sky looks blue because sunlight is scattered by the air, and shorter blue waves scatter more than "
              "the longer red ones, so blue light reaches our eyes from every part of the sky.")
    st = G.loop_stats(tok, tok.encode(normal))
    check("does not fire on a normal sentence", not st["loops"])
    check("does not fire on two repeats only", not G.loop_stats(tok, tok.encode("poem poem, the sea the sea")).get("loops"))
    ids = [5, 6, 5, 6, 7]
    check("distinct-2 counts distinct bigrams over bigrams (3 of 4)", abs(G.loop_stats(tok, ids)["distinct2"] - 0.75) < 1e-12)
    check("the empty reply does not loop and has distinct-2 1", G.loop_stats(tok, [])["loops"] is False and G.loop_stats(tok, [])["distinct2"] == 1)
    surf = G.load_surfaces()
    if surf is None:
        print("SKIP end-to-end: no surfaces json (run track4_sdmonly_generate.py --export-surfaces on the M5)")
        return all(ok)
    check("the prompt set is 10 chat and 5 poem prompts, as loopcheck ran", len(surf["surfaces"]["chat"]["prompts"]) == 10
          and len(surf["surfaces"]["poem"]["prompts"]) == 5)
    import track4_sdmonly_models as M
    torch.manual_seed(0)
    m = M.build_sdmonly("sdmonly", tok.V, {"d": 32, "d_a": 16, "n_sub": 12, "k": 5, "hops": 2, "n_back": 4, "decays": (0.8, 0.97)})
    m.eval()
    m.store.collect_stats = False
    step = G.Stepper(m, 64, torch.device("cpu"))
    r1 = run_check([("tiny", step)], surf, ("chat", "poem"), 2, "new", '{"maxTokens": 12, "minTokens": 4}', tok, 2, log=lambda s: None)
    r2 = run_check([("tiny", step)], surf, ("chat", "poem"), 2, "new", '{"maxTokens": 12, "minTokens": 4}', tok, 2, log=lambda s: None)
    keys = {"model", "surface", "replies", "looping", "loopRate", "distinct2", "meanTokens", "secs"}
    check("rows carry loopcheck's keys", all(set(r) == keys for r in r1["rows"]) and len(r1["rows"]) == 2)
    check("samples carry loopcheck's keys", all(set(s) == {"model", "surface", "prompt", "seed", "loops", "why", "text"} for s in r1["samples"]))
    check("2 prompts x 2 seeds = 4 replies per surface", all(r["replies"] == 4 for r in r1["rows"]))
    check("the check is deterministic", [s["text"] for s in r1["samples"]] == [s["text"] for s in r2["samples"]])
    r3 = run_check([("tiny", step)], surf, ("chat", "poem"), 2, "new", '{"maxTokens": 12, "minTokens": 4}', tok, 2,
                   log=lambda s: None, batch_seeds=False)
    check("lockstep seeds give the same replies as one at a time (CPU)", [s["text"] for s in r1["samples"]] == [s["text"] for s in r3["samples"]])
    print(f"{sum(ok)} of {len(ok)} checks pass")
    return all(ok)


def parse(argv=None):
    p = argparse.ArgumentParser(
        description="Loop check for PyTorch SDMLLM checkpoints: 30 chat replies and 15 poems by default, the definition "
                    "of loopcheck.mjs (a token 3-gram 3+ times, or one word 3 times in a row).",
        epilog="examples:\n"
               "  python3 %(prog)s --ckpt ~/settle24/ck/sdmonly/p0_A_c/last.pt --surfaces-json track4_sdmonly_surfaces.json "
               "--out loops_p0_A_c.json\n"
               "  python3 %(prog)s --ckpt A/last.pt --name A --ckpt B/last.pt --name B --surfaces chat\n"
               "  python3 %(prog)s --selftest",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", default=[], help="<run>/last.pt (repeatable)")
    p.add_argument("--name", action="append", default=[], help="row label per --ckpt (default: the run folder name)")
    p.add_argument("--result-json", action="append", default=[], help="result json per --ckpt (optional)")
    p.add_argument("--runs-dir", help="where <run>.result.json files live")
    p.add_argument("--surfaces", default="chat,poem", help="chat,poem,wlg (default chat,poem as loopcheck)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--params", default="new", choices=["new", "old"], help="new = SAMPLING, old = loopcheck's OLD")
    p.add_argument("--override", default="{}", help='json merged into the sampling, e.g. \'{"temperature": 0}\'')
    p.add_argument("--max-prompts", type=int, help="first N prompts per surface (tests only)")
    p.add_argument("--one-at-a-time", action="store_true",
                   help="generate each seed alone (default: the seeds of one prompt in lockstep, one batched forward)")
    p.add_argument("--surfaces-json", help="the export of track4_sdmonly_generate.py --export-surfaces")
    p.add_argument("--tokenizer")
    p.add_argument("--device")
    p.add_argument("--out", help="json path (loopcheck schema)")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


def main():
    a = parse()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.ckpt:
        print("need --ckpt (see --help)")
        print("NEXT -> python3 track4_sdmonly_loops.py --selftest")
        sys.exit(2)
    surf = G.load_surfaces(a.surfaces_json)
    if surf is None:
        raise SystemExit("no surfaces json: run track4_sdmonly_generate.py --export-surfaces OUT on the M5, pass --surfaces-json OUT")
    device = G.pick_device(a.device)
    tok = G.Tok(a.tokenizer)
    models, infos = [], {}
    st0 = stamp(device)
    for i, ck in enumerate(a.ckpt):
        name = a.name[i] if i < len(a.name) else os.path.basename(os.path.dirname(os.path.abspath(ck)))
        rj = a.result_json[i] if i < len(a.result_json) else None
        model, info = G.load_checkpoint(ck, device, rj, a.runs_dir)
        infos[name] = info
        models.append((name, G.Stepper(model, info["T"], device)))
        print(f"# {name}: {info['arm']} step {info['step']} on {device}", flush=True)
    out = run_check(models, surf, a.surfaces.split(","), a.seeds, a.params, a.override, tok, a.max_prompts,
                    batch_seeds=not a.one_at_a_time)
    out.update({"engine": "pytorch float32 checkpoint, sampler and loopStats ported from sdmchat.js (track4_sdmonly_generate.py)",
                "seeds_batched": not a.one_at_a_time,
                "checkpoints": infos, "surfaces_json": {"path": a.surfaces_json or os.environ.get("SDMONLY_SURFACES") or "node",
                                                       "utc": surf.get("utc"), "source": surf.get("source")},
                "stamp_start": st0, "stamp_end": stamp(device)})
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print("wrote", a.out)
    print("NEXT -> python3 track4_sdmonly_generate.py --ckpt " + a.ckpt[0] + " --repl")


if __name__ == "__main__":
    main()
