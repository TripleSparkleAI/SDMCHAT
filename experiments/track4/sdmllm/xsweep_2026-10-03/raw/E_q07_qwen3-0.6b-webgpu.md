# E q07 - Qwen3 0.6B WebGPU browser tokens per second q4f16

- query: `Qwen3 0.6B WebGPU browser tokens per second q4f16`
- tool: WebSearch
- fetched: 2026-10-03 10:50 UTC

## result list
1. Roadmap: NLP Models in the Browser (WebGPU or CPU) · Issue #5 · bthek1/model_playground - https://github.com/bthek1/model_playground/issues/5
2. Run AI Models in the Browser with WebGPU & WASM - https://maddevs.io/writeups/running-ai-models-locally-in-the-browser/
3. Qwen3 in the browser, zero keys - WebLLM 0.2.83 hands-on - DEV Community - https://dev.to/creeta/qwen3-in-the-browser-zero-keys-webllm-0283-hands-on-3ai2
4. [Performance] Qwen3.5-4B (q4f16) is ~3x slower decode + ~20x slower TTFT vs Qwen3-4B on WebGPU (Transformers.js 4.0.0-next.7) · Issue #1599 - https://github.com/huggingface/transformers.js/issues/1599
5. Run an LLM Inside a Browser Tab (pinggy) - https://pinggy.io/blog/run_llm_in_browser_webgpu/
6. Speed Benchmark - Qwen - https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html
7. WebGPU Browser AI Inference: Cut Client-Side LLM Costs in 2026 - https://www.buildmvpfast.com/blog/webgpu-browser-ai-inference-cost-savings-2026
8. Qwen (Wikipedia) - https://en.wikipedia.org/wiki/Qwen
9. 1,400 Tokens Per Second in Your Browser (essamamdani) - https://essamamdani.com/blog/lfm2-5-230m-webgpu-browser-inference-1400-tokens-per-second

## tool summary claims (unverified)
- roadmap issue: "a 0.6B model at q4f16 on WebGPU produces maybe 20 to 40 tokens per second on a laptop" (an estimate, not a measurement)
- WebLLM q4f16_1 Qwen3 0.6B "335MB download needing ~1.4GB of VRAM"; "q4f16_1 ... needs the shader-f16 WebGPU feature; q4f32_1 is the fp32 fallback"
- no measured Qwen3-0.6B browser tok/s was found by the tool.

## primary check: HF Hub API (E_q22_hf_api_onnx_sizes.txt and E_q22b)
- onnx-community/Qwen3-0.6B-ONNX: model_q4f16.onnx 569.8 MB; model_q4.onnx 919.1 MB; model_int8.onnx 617.7 MB; model_fp16.onnx 1202.8 MB; onnxruntime/webgpu/webgpu-int4-kld-block-32/model.onnx 543.3 MB. CONFIRMED.
- mlc-ai/Qwen3-0.6B-q4f16_1-MLC: 9 shards, total 351.5 MB; params_shard_0.bin holds only `model.embed_tokens.q_weight [151936, 128] uint32` (77,791,232 bytes); `model.embed_tokens.q_scale [151936, 32] float16` (9,723,904 bytes) sits in shard 1. CONFIRMED from tensor-cache.json.
- Qwen/Qwen3-0.6B config: vocab_size 151936, hidden_size 1024, tie_word_embeddings true. CONFIRMED.
