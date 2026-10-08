# E q17 - WebGPU top-k / argmax over a large vocabulary

- query: `WebGPU top-k sampling argmax large vocabulary GPU logits readback`
- tool: WebSearch
- fetched: 2026-10-03 10:57 UTC

## result list
1. GitHub - stfurkan/bitgpu: Fast WebGPU runtime for 1-bit (binary-weight) LLMs in the browser - https://github.com/stfurkan/bitgpu
2. Sampling on the GPU ... Gumbel-max with a counter-based RNG (#23) by jiazhihao · PR #74 · jiazhihao/mpk-apple - https://github.com/jiazhihao/mpk-apple/pull/74
3. Sample from the top k logits instead of sorting the whole vocabulary · PR #61 · bisand/kvad - https://github.com/bisand/kvad/pull/61
4. perf(vulkan): keep top-k top-p temperature and penalties on the GPU · Issue #12402 · anthony-chaudhary/fak - https://github.com/anthony-chaudhary/fak/issues/12402
5. v1: GPU sampling (+ experiments that passed) and the 1.0.0 changelog · PR #98 · Nehanth/pooled - https://github.com/Nehanth/pooled/pull/98
6. Attention Once Is All You Need: Efficient Streaming Inference with Stateful Transformers (arXiv 2605.13784) - https://arxiv.org/pdf/2605.13784
7. Add sampling penalties, min_p and exact top-k/top-p over the whole vocabulary · PR #243 · incoai/splash - https://github.com/incoai/splash/pull/243
8. perf(metal): slice-parallel GPU sampler, 4030µs → 229µs at 262k vocab · PR #207 · AI-native-Systems-Research/scratchy - https://github.com/AI-native-Systems-Research/scratchy/pull/207
9. RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection (arXiv 2501.14336) - https://arxiv.org/pdf/2501.14336

## tool summary claims (unverified unless checked)
- full-logits readback round trip "costs 2-4 ms per token on Apple silicon" (source not identified in the summary)
- kvad PR: partial selection over 151,936 logits "dropped from 2.0-2.1 ms | 0.16 ms"
- scratchy: Metal sampler 4030 us -> 229 us at 262k vocab

## primary check: Nehanth/pooled PR #98 (WebFetch 10:58 UTC)
- PR date September 27, 2026
- "argmax or top-k runs in the same submit, so a greedy token reads back 16 B instead of the 1 MB logits vector"
- "multi-workgroup argmax and top-k (k <= 64) over the columns of a logits matrix"
- "value desc / index asc (the host greedy's tie rule), NaN and -Inf never picked"
- measured (as returned, not a verbatim sentence): Chrome solo spec K=3 74.0 -> 84.2 tok/s (+13.5%); +5-7% for plain decoding. Hardware not returned.
- state: CONFIRMED as the PR author's own measurement; hardware unknown.
