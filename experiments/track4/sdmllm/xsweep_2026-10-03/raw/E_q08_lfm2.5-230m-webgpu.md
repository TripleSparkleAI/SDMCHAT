# E q08 - LFM2.5 230M WebGPU browser tokens per second (plus q08b, the Space and the post)

- queries: `LFM2.5 230M WebGPU browser tokens per second`; `webml-community LFM2.5 WebGPU kernels Hugging Face Space 1400 tok/s Victor Mustar`
- tool: WebSearch (two calls)
- fetched: 2026-10-03 10:50 UTC

## result list, call 1
1. Digg - https://digg.com/tech/hq3bj9fg
2. LFM2.5-230M: Built to Run Anywhere (Liquid AI blog) - https://www.liquid.ai/blog/lfm2-5-230m
3. Running an LLM agent entirely in your browser - DEV Community - https://dev.to/lajosbencz/running-an-llm-agent-entirely-in-your-browser-5foe
4. 1,400 Tokens Per Second in Your Browser (essamamdani) - https://essamamdani.com/blog/lfm2-5-230m-webgpu-browser-inference-1400-tokens-per-second
5. Browser Inference Breakthrough: LFM2.5 230M Hits 1,400 tok/s via Custom WebGPU Kernels (baguaai) - https://baguaai.com/browser-inference-breakthrough-lfm2-5-230m-hits-1400-tok-s-via-custom-webgpu-kernels/
6. Liquid AI Ships LFM2.5-230M ... (MarkTechPost, 2026-06-27) - https://www.marktechpost.com/2026/06/27/liquid-ai-ships-lfm2-5-230m-with-llama-cpp-mlx-vllm-sglang-and-onnx-support-for-on-device-inference/
7. LFM2.5-230M: Liquid AI Edge Agent Model - 213 tok/s on Phone CPU (explainx) - https://www.explainx.ai/blog/liquid-ai-lfm2-5-230m-edge-agent-model-2026
8. WebGPU: Your Browser Just Got Superpowers (substack) - https://productpower.substack.com/p/webgpu-your-browser-just-got-superpowers
9. github.com/josephrocca/WebGPT - https://github.com/josephrocca/WebGPT

## result list, call 2
1. LFM2.5 WebGPU Summarizer - webml-community Space - https://huggingface.co/spaces/webml-community/lfm2.5-webgpu-summarizer
2. LFM2 WebGPU Kernels - webml-community Space - https://huggingface.co/spaces/webml-community/lfm2-webgpu-kernels
3. LFM2.5-VL-3B WebGPU - LiquidAI Space - https://huggingface.co/spaces/LiquidAI/LFM2.5-VL-3B-WebGPU
4. LFM2.5 Edge Research Agent - LiquidAI Space - https://huggingface.co/spaces/LiquidAI/LFM2.5-2.6B-WebGPU
5. MONARCH - LFM2.5 WebGPU Kernels - inductiveML Space - https://huggingface.co/spaces/inductiveML/monarch-webgpu
6. LiquidAI/LFM2.5-350M - https://huggingface.co/LiquidAI/LFM2.5-350M
7. webml-community - https://huggingface.co/webml-community
8. Real-time video captioning with LFM2.5-VL-1.6B and WebGPU - https://paulabartabajo.substack.com/p/real-time-video-captioning-with-lfm25
9. a Hugging Face Space by webml-community (daily.dev) - https://daily.dev/posts/a-hugging-face-space-by-webml-community-7csqkjyen
10. Xenova on X (x.com/xenovacom/status/2070210622239707568) - via search, X not fetched, engagement unknown. Title text as returned by the search engine: "While we eagerly await Fable 5's return, our agentic WebGPU kernel optimization framework kept running. Opus 4.8 picked up where Fable left off, pushing Liquid AI's new LFM2.5 230M to an unbelievable 1,400 tok/s... running locally in your browser. Don't blink or you'll miss it."
11. Liquid AI on X (x.com/liquidai/status/2070148140368318884) - via search, X not fetched.
12. LiquidAI/LFM2.5-230M · Hugging Face - https://huggingface.co/LiquidAI/LFM2.5-230M
13. LiquidAI/LFM2.5-Encoder-230M - https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M
14. Raspberry Pi 5 for Local AI in 2026: LFM2.5-230M at 42 tok/s (runaihome) - https://runaihome.com/blog/raspberry-pi-5-local-ai-lfm25-2026/
15. WebGPU Benchmark Results (15.52x speedup) - Xenova/webgpu-embedding-benchmark discussion 69 - https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/69

## primary check: Liquid AI blog (WebFetch 10:51 UTC)
- Published June 25, 2026. 230M parameters. "Raspberry Pi 5: 42 tok/s decode, 523 tok/s prefill"; "Samsung Galaxy S25 Ultra (Snapdragon Gen4): 213 tok/s decode, 1,158 tok/s prefill". These are native CPU numbers, not browser.
- The blog does not state the 1,400 tok/s figure, vocab size, or file sizes.
- state of the CPU numbers: CONFIRMED as vendor claims (vendor blog).

## primary check: config and files (HF API, see E_q22)
- LiquidAI/LFM2.5-230M config: vocab_size 65536, hidden_size 1024, tie_word_embeddings true, 14 layers. CONFIRMED.
- LiquidAI/LFM2.5-230M-ONNX: model_q4.onnx_data 211.1 MB; model_q8.onnx_data 489.3 MB; model_fp16.onnx_data 475.8 MB; model_q4f32.onnx_data 403.0 MB. CONFIRMED.
- the 1,400 tok/s: only in a search-engine rendering of an X post title and in secondary blogs; hardware "M4 Max" only in secondary blogs. State POST CLAIM ONLY.
