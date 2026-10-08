"""SDMLLM S0 floors: add-one unigram and the Dirichlet-backoff n-gram count stores (EXACT1..EXACT5) on the same
tokens, the same TEST targets and the same within-window contexts the neural arms see.

<claudes_code_comments>
** Function List **
hkey(stream, pos, k) - 64-bit hash of the k tokens before each position
table(keys, ys) - sorted unique context keys with counts, and sorted unique (context, next) joints with counts
lookup(tab, qkeys, ys) - context and joint counts for query positions (0 where unseen)
chain_p(...) - p(y | context) through the Dirichlet backoff chain up to a given order
targets(arr, half, T) - TEST (or CALIB) target positions, their within-window index, BOS targets removed
sample_exact(...) - free-running sampling from the order-3 chain, for the repetition comparison
main() - build tables on train, tune masses on CALIB, score TEST, write the json

** Technical Review **
- The chain: p_0(y) = (c(y) + 1) / (N + V); for k = 1..K, p_k(y|ctx) = (c(ctx, y) + m_k p_{k-1}(y)) /
  (c(ctx) + m_k). A context of order k is used only if the target sits at least k positions into its
  window, because a neural arm sees no tokens before its window start; so the floors are scored on exactly
  the targets and the information the neural arms get.
- Keys are 64-bit polynomial hashes of the context ids; joints hash (context key, y). A collision needs two
  different contexts to share 64 bits, expected count about n^2 / 2^65, far below one for 65M rows.
- SDMLLM_FLOOR_TRAIN_TOKENS=20000000 builds the tables on a 20M-token prefix: a data-SIZE-matched floor for
  the neural arms, which see 20M tokens drawn from the whole shard. The default uses all 65.5M train tokens.
- Masses m_k are tuned order by order on a seeded 250k-target subsample of CALIB (first half of the held-out docs) over a geometric grid,
  lower orders fixed, then frozen; TEST (second half) is scored once.
- bpb = sum of -log p over scored targets / (ln 2 * sum of token_bytes of those targets).
</claudes_code_comments>
"""
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
V = 129280
T = 256
P1 = np.uint64(0x9E3779B97F4A7C15)
P2 = np.uint64(0xC2B2AE3D27D4EB4F)
MASS_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]


def hkey(stream, pos, k):
    h = np.full(len(pos), np.uint64(k) * P2, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(1, k + 1):
            h = (h * P1) ^ (stream[pos - j].astype(np.uint64) + np.uint64(1))
            h = h ^ (h >> np.uint64(29))
    return h


def jkey(ctx, ys):
    with np.errstate(over="ignore"):
        h = (ctx ^ (ys.astype(np.uint64) * P2)) * P1
        return h ^ (h >> np.uint64(31))


def table(keys, ys):
    uk, kc = np.unique(keys, return_counts=True)
    uj, jc = np.unique(jkey(keys, ys), return_counts=True)
    return uk, kc.astype(np.float64), uj, jc.astype(np.float64)


def _find(sorted_arr, q):
    i = np.clip(np.searchsorted(sorted_arr, q), 0, len(sorted_arr) - 1)
    return i, sorted_arr[i] == q


def lookup(tab, qkeys, ys):
    uk, kc, uj, jc = tab
    i, hit = _find(uk, qkeys)
    c_ctx = np.where(hit, kc[i], 0.0)
    j, jhit = _find(uj, jkey(qkeys, ys))
    c_joint = np.where(jhit & hit, jc[j], 0.0)
    return c_ctx, c_joint


def targets(arr, half):
    n = len(arr)
    lo, hi = (n // 2, n) if half == "test" else (0, n // 2)
    starts = np.arange(lo, hi - (T + 1), T + 1)
    pos = (starts[:, None] + np.arange(1, T + 1)[None, :]).reshape(-1)
    within = np.tile(np.arange(1, T + 1), len(starts))
    keep = arr[pos] != 0
    return pos[keep], within[keep]


def chain_p(stream, pos, within, ys, tabs, uni, masses):
    p = uni[ys]
    for k, m in enumerate(masses, start=1):
        ok = within >= k
        q = hkey(stream, pos, k)
        c_ctx, c_joint = lookup(tabs[k], q, ys)
        pk = (c_joint + m * p) / (c_ctx + m)
        p = np.where(ok, pk, p)
    return p


def bpb(p, ys, tb):
    return float(-np.log(np.maximum(p, 1e-300)).sum() / (math.log(2) * tb[ys].sum()))


def main():
    t0 = time.time()
    train = np.fromfile(os.path.join(DATA, "train.u32"), dtype=np.uint32).astype(np.int64)
    limit = int(float(os.environ.get("SDMLLM_FLOOR_TRAIN_TOKENS", "0")))
    if limit:
        # a data-matched floor: the neural arms see 20M tokens drawn from the whole shard, so the matched count
        # store is built on a 20M-token prefix (same size, not the same draws)
        train = train[:limit]
    val = np.fromfile(os.path.join(DATA, "val.u32"), dtype=np.uint32).astype(np.int64)
    tb = np.fromfile(os.path.join(DATA, "token_bytes.i32"), dtype=np.int32).astype(np.int64)
    N = len(train)
    uni = (np.bincount(train, minlength=V).astype(np.float64) + 1.0) / (N + V)
    K = 5
    tabs = {}
    for k in range(1, K + 1):
        pos = np.arange(k, N)
        tabs[k] = table(hkey(train, pos, k), train[pos])
        print(f"order {k}: {len(tabs[k][0]):,} contexts, {len(tabs[k][2]):,} joints  {time.time()-t0:.0f}s", flush=True)
    cpos, cwin = targets(val, "calib")
    # tune on a fixed 250k-target subsample of CALIB (seeded) for time; TEST is always scored in full
    sub = np.sort(np.random.default_rng(0).choice(len(cpos), size=min(250_000, len(cpos)), replace=False))
    cpos, cwin = cpos[sub], cwin[sub]
    tpos, twin = targets(val, "test")
    cy, ty = val[cpos], val[tpos]
    res = {"train_tokens": N, "calib_targets": int(len(cpos)), "test_targets": int(len(tpos)),
           "test_bytes": int(tb[ty].sum()), "mass_grid": MASS_GRID, "arms": {}}
    res["arms"]["UNIGRAM"] = {"test_bpb": bpb(uni[ty], ty, tb), "calib_bpb": bpb(uni[cy], cy, tb)}
    print("UNIGRAM", res["arms"]["UNIGRAM"], flush=True)
    masses = []
    for k in range(1, K + 1):
        best = None
        for m in MASS_GRID:
            b = bpb(chain_p(val, cpos, cwin, cy, tabs, uni, masses + [m]), cy, tb)
            if best is None or b < best[1]:
                best = (m, b)
        masses.append(best[0])
        tb_ = bpb(chain_p(val, tpos, twin, ty, tabs, uni, masses), ty, tb)
        res["arms"][f"EXACT{k}"] = {"masses": list(masses), "calib_bpb": best[1], "test_bpb": tb_}
        print(f"EXACT{k}", res["arms"][f"EXACT{k}"], flush=True)
    # negative control: shuffle the order-3 table's joint counts across joints (breaks context-next pairing)
    rng = np.random.default_rng(0)
    uk, kc, uj, jc = tabs[3]
    shuf = dict(tabs)
    shuf[3] = (uk, kc, uj, jc[rng.permutation(len(jc))])
    m3 = res["arms"]["EXACT3"]["masses"]
    res["arms"]["SHUF3_control"] = {"test_bpb": bpb(chain_p(val, tpos, twin, ty, shuf, uni, m3), ty, tb)}
    print("SHUF3_control", res["arms"]["SHUF3_control"], flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["train_tokens_limit"] = limit
    name = "track4_sdmllm_count_store_floors_result.json" if not limit else f"track4_sdmllm_count_store_floors_{limit // 10**6}M_result.json"
    json.dump(res, open(os.path.join(HERE, name), "w"), indent=1)
    print("NEXT -> python3 track4_sdmllm_samples_and_repetition.py")


if __name__ == "__main__":
    main()
