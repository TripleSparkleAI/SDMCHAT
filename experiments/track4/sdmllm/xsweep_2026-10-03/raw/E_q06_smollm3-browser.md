# E q06 - SmolLM3 on-device browser WebGPU 2026

- query: `SmolLM3 on-device browser WebGPU 2026`
- tool: WebSearch
- fetched: 2026-10-03 10:53 UTC

## result list
1. Run an LLM Inside a Browser Tab: WebGPU and Local Inference in 2026 (pinggy) - https://pinggy.io/blog/run_llm_in_browser_webgpu/
2. Llamas on the Web (arXiv 2605.20706) - https://arxiv.org/html/2605.20706v1
3. Xenova on X, status 1813258097185448377 (SmolLM v1 announcement, 2024) - via search, X not fetched, engagement unknown
4. The Power Of SmolLM With WebGPU (aicompetence) - https://aicompetence.org/smollm-with-webgpu-efficient-ai-in-browser/
5. WebAR (Wikipedia) - https://en.wikipedia.org/wiki/WebAR
6. transformers.js-examples/smollm-webgpu - https://github.com/huggingface/transformers.js-examples/tree/main/smollm-webgpu
7. SmolLM3 by Hugging Face (medium) - https://medium.com/data-science-in-your-pocket/smollm3-by-hugging-face-redefining-efficiency-in-ai-with-a-compact-3b-parameter-powerhouse-0d90825580cf
8. Run an LLM in Your Browser: WebGPU, No Install, No Server (localaimaster) - https://localaimaster.com/blog/run-llm-in-browser
9. WebLLM: A High-Performance In-Browser LLM (arXiv 2412.15803) - https://arxiv.org/html/2412.15803v1

## tool summary claims (unverified)
- HF Space HuggingFaceTB/SmolLM3-3B-Instruct-WebGPU exists.
- LlamaWeb test table includes SmolLM3-3B q4_k_m 1.92 GB; LlamaWeb "streams weights using only four 1 MB buffers ... directly from OPFS into WebGPU buffers"; Safari "especially strict memory usage limits".
- Browser support line: Chrome/Edge 113+, Firefox 141+ on Windows, 145+ on Apple Silicon Macs, Safari on macOS Tahoe 26 / iOS 26.
- "WebGPU only guarantees a 256MB maxBufferSize and a 128MB maxStorageBufferBindingSize" by default.

## primary check
- the default limits (256 MiB maxBufferSize, 128 MiB maxStorageBufferBindingSize) are checked in q16b against the WebGPU spec.
- the OPFS four-buffer streaming claim was not in the q01b fetch output; UNVERIFIED.
