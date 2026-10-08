# E q02b - primary check: Hugging Face blog "Transformers.js v4"

- url: https://huggingface.co/blog/transformersjs-v4
- tool: WebFetch
- fetched: 2026-10-03 10:49 UTC

## returned
- Publication date: February 9, 2026. Authors: Joshua (Xenova) and Nico Martin.
- quote: "~60 tokens per second on an M4 Pro Max" running GPT-OSS 20B (q4f16)
- dtypes mentioned: fp32, fp16, q4, q4f16
- `env.useWasmCache` caches WASM runtime files; `ModelRegistry` with `clear_pipeline_cache()` and `is_pipeline_cached()`.
- "support for larger models exceeding 8B parameters"

## state
CONFIRMED (blog, primary). Note "M4 Pro Max" is the blog's wording; Apple sells M4 Pro and M4 Max, so the hardware name is ambiguous as written.
