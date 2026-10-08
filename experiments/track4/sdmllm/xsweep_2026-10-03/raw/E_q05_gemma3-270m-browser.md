# E q05 - Gemma 3 270M browser WebGPU transformers.js download size

- query: `Gemma 3 270M browser WebGPU transformers.js download size`
- tool: WebSearch
- fetched: 2026-10-03 10:50 UTC

## result list
1. GitHub - svenflow/webgpu-gemma: Run Gemma 3 1B locally in the browser via WebGPU. Q8_0 quantized, multi-turn chat, zero dependencies. - https://github.com/svenflow/webgpu-gemma
2. google/gemma-3-270m-it · The model is now added to WebAI.js - https://huggingface.co/google/gemma-3-270m-it/discussions/11
3. Add Gemma 3 270M as an on-device model option by akspad · Pull Request #54 · akspad/dedoomify - https://github.com/akspad/dedoomify/pull/54
4. Introducing Gemma 3 270M (Simon Willison, 2025-08-14) - https://simonwillison.net/2025/Aug/14/gemma-3-270m/
5. Google unveils ultra-small ... Gemma 3 270M (VentureBeat) - https://venturebeat.com/ai/google-unveils-ultra-small-and-efficient-open-source-ai-model-gemma-3-270m-that-can-run-on-smartphones
6. GitHub - lagna360/gemma3-270m-browser-demo - https://github.com/lagna360/gemma3-270m-browser-demo
7. google/gemma-3-270m · Hugging Face - https://huggingface.co/google/gemma-3-270m
8. Google rolls out Gemma 3 270M (testingcatalog) - https://www.testingcatalog.com/google-rolls-out-gemma-3-270m-multimodal-model-for-phones-and-edge-inference/
9. Gemma 3 270M: Run AI in 125MB (localaimaster) - https://localaimaster.com/models/gemma-3-270m

## tool summary claims (unverified)
- PR dated Oct 2, 2026 labels "On-device AI: Gemma 3 270M (private, ~280 MB)" and says download untested.
- "170 million embedding parameters ... 256k vocabulary ... 100 million transformer block parameters" (Google wording, via tool)
- a q4 transformers.js run "41.7 t/s. TTFT: 0.51s" (hardware not given in summary)

## primary check: svenflow/webgpu-gemma README (WebFetch, 10:50 UTC)
- Mac Mini M4 Pro (Chrome 134): Gemma 3 1B "59.8 t/s", TTFT "0.28s"; Gemma 3 270M "136.8 t/s", TTFT "0.11s"; claims 3.3x over transformers.js on 270M.
- iPhone 17 Pro Max (Safari, iOS 26): 1B "34.4 t/s"; 270M "101.1 t/s", TTFT "0.14s".
- 270M: "~300MB download", "~500MB GPU memory"; library "12KB gzipped"; "Q8_0 quantization"; "Range request streaming"; "Peak JS memory is ~50MB instead of ~1GB" via layer-by-layer GPU uploads; "18 custom WGSL shaders".
- state: POST CLAIM ONLY (author's own README benchmark; no independent reproduction). Repo date not returned.

## primary check: HF Hub file sizes (see E_q22_hf_api_onnx_sizes.txt)
onnx-community/gemma-3-270m-it-ONNX: model_q4f16.onnx_data 272.6 MB; model_q4.onnx_data 322.9 MB; model_fp16.onnx_data 569.9 MB; model.onnx_data (fp32) 1139.5 MB. CONFIRMED (HF API).
