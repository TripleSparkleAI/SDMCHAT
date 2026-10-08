# Band E search matrix (written before any query ran)

Band: E - browser and on-device small LMs.
Written: 2026-10-03 10:47 UTC.
Window: 2026-08-01 to 2026-10-03, widened to 2026-06-01 where the narrow window is empty.
X API: unavailable (HTTP 402 credits depleted). Web-mirror fallback only.

| id | axis | query | tool |
|---|---|---|---|
| q01 | engine | WebLLM WebGPU tokens per second 2026 | WebSearch |
| q02 | engine | transformers.js v4 WebGPU release | WebSearch |
| q03 | engine | ONNX Runtime Web WebGPU LLM 2026 | WebSearch |
| q04 | engine | wllama llama.cpp wasm 2026 | WebSearch |
| q05 | model size | Gemma 3 270M browser WebGPU download size | WebSearch |
| q06 | model size | SmolLM3 browser on-device | WebSearch |
| q07 | model size | Qwen3 0.6B WebGPU tokens per second | WebSearch |
| q08 | model size | LFM2-350M browser / LFM2 on-device | WebSearch |
| q09 | export | q4f16 ONNX export web LLM | WebSearch |
| q10 | export | embedding table quantization int4 on-device LLM (vocab embedding) | WebSearch |
| q11 | export | tied lm_head quantization large vocabulary on-device | WebSearch |
| q12 | platform | WebGPU subgroups Chrome shipped | WebSearch |
| q13 | platform | Safari 26 WebGPU shader-f16 | WebSearch |
| q14 | platform | Firefox WebGPU 2026 release | WebSearch |
| q15 | platform | WebAssembly relaxed SIMD matrix multiply | WebSearch |
| q16 | platform | wasm memory64 browser support 2026 | WebSearch |
| q17 | kernel | top-k argmax large vocabulary GPU WebGPU | WebSearch |
| q18 | delivery | OPFS Cache API model weights browser caching | WebSearch |
| q19 | delivery | Hugging Face file size limit browser chunked weights / WebLLM shards | WebSearch |
| q20 | papers | arXiv Atom: WebGPU AND language model, newest | python3 urllib |
| q21 | releases | GitHub REST: latest releases of transformers.js, web-llm, onnxruntime, wllama, llama.cpp | python3 urllib |
| q22 | models | HF Hub API: onnx-community and mlc-ai models, most downloaded / recently modified | python3 urllib |

Primary-source checks are recorded inside each raw file under "primary check".
