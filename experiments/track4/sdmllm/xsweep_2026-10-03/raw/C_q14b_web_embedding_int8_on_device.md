# C_q14b int4 / int8 quantisation of embedding tables on device (web)

- query: "int4 int8 quantization embedding table on-device small language model vocabulary 2026 browser"
- tool: WebSearch
- fetched UTC: 2026-10-03 ~10:52
- note: snippets are the search tool's summary.

## Result list
1. Bekko Embedding: Parameter-Efficient Multilingual Retrieval with Ultra-Compact Encoders - https://arxiv.org/pdf/2607.25180
2. Low-Precision Hardware Architectures Meet Recommendation Model Inference at Scale - https://arxiv.org/pdf/2105.12676
3. Closing the Semantic-Edge Gap: Tiny Language Models for 6G - https://arxiv.org/pdf/2609.03747
4. Hummingbird: embedded FPGA LLM accelerator - https://arxiv.org/pdf/2507.03308
5. Small Language Models guide 2026 (cogitx) - https://cogitx.ai/blog/small-language-models-slms-comprehensive-guide-2026
6. Model quantization concepts (zeroentropy) - https://zeroentropy.dev/concepts/model-quantization/
7. SLMs on edge devices 2026 (renard-digital) - https://renard-digital.fr/blog/en/small-language-models-edge-devices-2026/
8. PinFM - https://arxiv.org/pdf/2507.12704
9. ewin-reg/WeMM-Embedding-2B-INT8 - https://huggingface.co/ewin-reg/WeMM-Embedding-2B-INT8

## Search-tool summary points
- Bekko: row-wise symmetric int8 table plus fp32 row scales cuts the static embedding table by about 74.7% versus fp32; graph "Gather(int8 table) -> Cast(float) -> Gather(row_scale) -> Mul".
- PinFM: relative L2 deviation 0.45% at int8, 7.8% at int4 for its embeddings.
- Meta recsys (2105.12676): int4 errors smaller on large tables than small ones; int4 on the larger half plus int8 on the rest cuts size 40-50% vs all-int8.
- None about browser runtimes specifically.
