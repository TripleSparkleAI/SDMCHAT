#!/usr/bin/env python3
"""memguard.py - lane SDMCHATS: keep MemAvailable above 20 GB beside HEADCLIMB, and bring our jobs back when it recovers.

Every 10 s. Two low readings (< LOW GB) in a row cancel the newest running SDMCHATS-* q job (it resumes from its last
checkpoint) and put it at the FRONT of the resume list. When MemAvailable has stayed >= HIGH GB for RECOVER readings
and no SDMCHATS job started in the last GAP s, the first entry is resubmitted with its original command and priority.
Each reading also logs our jobs' GPU memory (nvidia-smi) once a minute to memguard_mem.tsv.
Touches only q jobs named SDMCHATS-*. Log: ~/settle24/sdmchats/memguard.log; list: resume.json.
"""
import glob, json, os, subprocess, time
H = os.path.expanduser("~")
Q = f"{H}/settle24/code/q"; SPOOL = f"{H}/settle24/q"; D = f"{H}/settle24/sdmchats"
LOG = f"{D}/memguard.log"; LST = f"{D}/resume.json"; TSV = f"{D}/memguard_mem.tsv"
LOW, HIGH, RECOVER, GAP = 20, 29, 12, 600


def log(m):
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + m + "\n")


def mem():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def ours(state):
    out = []
    for f in glob.glob(f"{SPOOL}/{state}/*.json"):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if j.get("name", "").startswith("SDMCHATS-"):
            out.append(j)
    return out


def lst():
    try:
        return json.load(open(LST))
    except Exception:
        return []


def gpu_rows():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return r.stdout.strip().replace("\n", ";")


log(f"armed pid {os.getpid()} (v3)")
low = high = 0
last_start = 0.0
last_tsv = 0.0
while True:
    try:
        m = mem()
        low = low + 1 if m < LOW else 0
        high = high + 1 if m >= HIGH else 0
        run = sorted(ours("running") + ours("queue"), key=lambda j: j.get("submitted_t", 0))
        if time.time() - last_tsv > 60:
            with open(TSV, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S', time.gmtime())}\t{m:.1f}\t{','.join(j['name'] for j in run)}\t{gpu_rows()}\n")
            last_tsv = time.time()
        if low >= 2 and run:
            j = run[-1]
            log(f"MemAvailable {m:.1f} GB: cancel {j['id']} {j['name']}")
            subprocess.run([Q, "cancel", j["id"]], capture_output=True)
            cmd = j["cmd"] if "--ckpt-every 1000" in j["cmd"] else j["cmd"] + " --ckpt-every 1000"
            L = lst()
            L.insert(0, {"name": j["name"], "cmd": cmd, "priority": j["priority"], "threads": j["threads"], "gpu": j["gpu"], "mem": j["mem_gb"]})
            json.dump(L, open(LST, "w"), indent=1)
            low = 0
            time.sleep(60)
            continue
        L = lst()
        if L and high >= RECOVER and time.time() - last_start > GAP:
            e = L.pop(0)
            r = subprocess.run([Q, "submit", "-p", str(e["priority"]), "--gpu", str(e["gpu"]), "--threads", str(e["threads"]),
                                "--mem", str(e["mem"]), "--name", e["name"], "--", e["cmd"]], capture_output=True, text=True)
            log(f"MemAvailable {m:.1f} GB held: resubmit {e['name']} -> {r.stdout.strip()[:80]} {r.stderr.strip()[:200]}")
            json.dump(L, open(LST, "w"), indent=1)
            last_start = time.time()
            high = 0
    except Exception as ex:  # never die: a dead guard is worse than a noisy one
        log(f"guard error {ex!r}")
    time.sleep(10)
