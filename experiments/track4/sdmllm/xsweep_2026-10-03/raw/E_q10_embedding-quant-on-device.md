# E q10 - embedding table quantization int4 on-device LLM

- query: `embedding table quantization int4 on-device LLM vocabulary embedding memory 2026`
- tool: WebSearch
- fetched: 2026-10-03 10:53 UTC

## result list
1. CARVQ: Corrective Adaptor with Group Residual Vector Quantization for LLM Embedding Compression - https://arxiv.org/pdf/2510.12721
2. FlexQ: Efficient Post-training INT6 Quantization for LLM Serving - https://arxiv.org/pdf/2508.04405
3. Hummingbird: A Smaller and Faster LLM Accelerator on Embedded FPGA - https://arxiv.org/pdf/2507.03308
4. Low-Precision Hardware Architectures Meet Recommendation Model Inference at Scale - https://arxiv.org/pdf/2105.12676
5. 2026-06-19 Gemma 4 Technical Report - https://arxiv.org/pdf/2607.02770
6. Post-Training 4-bit Quantization on Embedding Tables - https://arxiv.org/html/1911.02079
7. Bekko Embedding - https://arxiv.org/pdf/2607.25180
8. Gemma 4 on a Tesla T4, Part 3: Int4 Embeddings Serve E2B in 2.86 GiB at 2.30x bf16 - DEV Community - https://dev.to/gde/gemma-4-on-a-tesla-t4-part-3-int4-embeddings-serve-e2b-in-286-gib-at-230x-bf16-3kch
9. What Survives When You Compress a Recursive Reasoner for the Edge? - https://arxiv.org/pdf/2606.26488
10. On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding - https://arxiv.org/pdf/2510.15620

## tool summary claims (unverified unless checked below)
- Hummingbird: under quantization "the embedding table is often retained in FP16 to preserve model accuracy"
- Gemma 4 report: "the vocabulary we use has 262k entries"
- Bekko: int8 table plus fp32 per-row scales cut the static embedding table "by about 74.7% versus fp32"

## primary check: DEV article (WebFetch 10:54 UTC)
- Author xbill (Google Developer Experts), dated September 30 (2026).
- "Every sampled group of 32 values in both tables sits on the same 4-bit grid as the linear layers"
- sizes: `embed_tokens_per_layer` 4.375 GiB -> 1.230 GiB; `embed_tokens` 0.750 GiB -> 0.211 GiB; checkpoint 6.11 GiB -> 2.64 GiB
- single 512-token request 85.28 tok/s vs bf16 37.04 (2.30x); eight concurrent 239.64 vs 164.62 (1.46x), on a Tesla T4
- "The output layer also runs once per generated token over the whole 262,144-token vocabulary, so at bf16 it is one of the largest reads in every decode step."
- "All eight greedy test prompts produce the same tokens as the build with bf16 embeddings."
- No perplexity measured (per the fetch).
- state: CONFIRMED as the author's own measurement (one blog author, T4 server GPU, not browser).
