# C_q19 primary check: how record #92 builds its sampled softmax (blog + code at PR #360 head sha 4f5270e5)

## Source 1: Hyperstition blog, Training NanoGPT in 39.9 Seconds
- url: https://hyperstition.cc/training-nanogpt-in-39-9-seconds
- tool: WebFetch; fetched UTC 2026-10-03 ~10:53; post date given on page: September 27, 2026
- verbatim quotes returned by the fetch reader:
  - "keeping every target token in a local batch and adding a deduplicated set of non-target tokens"
  - "The final schedule uses 10,240 → 14,336 → 24,576 candidates, followed by all 50,304 entries for the final 87 steps"
  - "For omitted elements of the vocabulary, we keep the gradient at zero in the sampled softmax variant"
  - "sampled softmax with prefix cross-entropy and a fused loss kernel contributes 8.41 seconds"
  - "this is a greater improvement than the previous 45 world records combined"
- no logQ / sampling-probability correction is mentioned on the page.

## Source 2: train_gpt.py at 4f5270e5 (raw.githubusercontent.com), fetched 2026-10-03T10:53:02.188161Z 200
```
# ---- sampled softmax: the shared candidate set -------------------------------
# Shared-negative sampled softmax for the early stages; the mechanism doc is triton_kernels.py.
SNS_CANDIDATES_PER_STAGE = [10240, 10240, 10240, 32768, 32768]
# Stage 2 ramps rather than stepping: the count has to reach full-softmax resolution before the batch taper.
SNS_CANDIDATE_RAMP = {2: [14336, 14336, 24576]}
_SNS_STRIDE = 20011          # coprime with the vocab, so the negative sweep is a permutation
_SNS_PIN_RING = 4            # deeper than the host run-ahead, so no slot is rewritten while its copy is still pending

def _sns_p_at(step: int) -> int:
    """Candidate count for `step` -- a multiple of the CE kernel's vector width; 0 is full softmax."""
    if step >= _SNS_UNTIL_RESOLVED:
        return 0
    _i, (_a, _b) = next((i, bd) for i, bd in enumerate(training_schedule.boundaries) if step < bd[1])
    _r = SNS_CANDIDATE_RAMP.get(_i)
    return _r[(step - _a) * len(_r) // (_b - _a)] if _r else SNS_CANDIDATES_PER_STAGE[_i]
...
    def _sns_negatives(self, need: int, mark, V: int):
        """`need` classes not already marked and distinct from each other: the next window of the precomputed stride permutation (k*_SNS_STRIDE mod V, coprime to V) -- uniform and provably duplicate-free."""
        draw = min(V, int(need * 1.7) + 256)
        idx, self._sns_off = self._sns_perm[self._sns_off:self._sns_off + draw], (self._sns_off + draw) % V
        idx = idx[~mark[idx]]
        if idx.size >= need:
            return idx[:need]
        free = np.flatnonzero(~mark[:V])
        assert free.size >= need, f"[sampled-softmax] only {free.size} free classes for {need}"
        return free[::max(1, free.size // need)][:need]

    def sns_stage(self, step: int, targets_np):
        """HOST half of the candidate build for one microbatch: fills a pinned ring slot, no device work and no D2H sync -- the targets are the loader's own
        CPU tensor. C is every target plus enough negatives to reach P, ASCENDING; prefix targets are not forced in -- the CE kernel no-ops a prefix at -1."""
        P = _sns_p_at(step)
        if not P:
            return None
        V, mark, pos, T = self._sns_vocab, self._sns_mark, self._sns_pos, int(targets_np.shape[0])
        mark.fill(False)
        mark[targets_np] = True
        U = int(np.count_nonzero(mark[:V]))
        assert U <= P, f"[sampled-softmax] candidate overflow at step {step}: {U} always-included > P={P}"
        if U != P:
            self._sns_marku[self._sns_negatives(P - U, mark, V)] = 1
```

## Source 3: triton_kernels.py at 4f5270e5, fetched 2026-10-03T10:53:11.275813Z 200
```
# SNS -- Shared-Negative Sampled softmax for the EARLY stages, where the full vocabulary's 2.5-5 GB of logit traffic per
# step buys resolution not yet worth paying for.  Until the cutover the three GEMMs run at width P << 50304 against a
# per-step SHARED candidate set C, with the CE CUDA kernel reused UNCHANGED at VOCAB_SIZE = P.
class SampledSoftcappedCrossEntropy(torch.autograd.Function):
    """Sampled-candidate variant of FusedSoftcappedCrossEntropy: the same arguments plus a trailing `sns_p`, this stage's padded candidate
    count (a python int, so a traced constant). The three GEMMs run at width P against the row-gather sns_gather leaves in _SNS_WC/_SNS_WCT.
    `targets` / `prefix_targets` are taken for signature parity and shape checking only; the kernel is fed the host-built POSITION vectors."""
```

## Observations (from the code, not from any claim)
- Candidate set C = every target token of the microbatch plus negatives from a fixed stride permutation (k*20011 mod V), uniform over classes, duplicate-free. The CE kernel is the full-softmax kernel recompiled at VOCAB_SIZE = P. No logQ term appears in the candidate build or in the SNS kernel doc.
- The search-tool summary in C_q17 said the full softmax is used for the first 100 steps; the code returns a positive P from step 0 (SNS_CANDIDATES_PER_STAGE[0] = 10240). The summary claim is not supported by the code read here.
- GitHub REST API returned HTTP 403 (unauthenticated rate limit) on a git/trees call at 2026-10-03T10:52:53Z; raw.githubusercontent.com reads continued to work.
