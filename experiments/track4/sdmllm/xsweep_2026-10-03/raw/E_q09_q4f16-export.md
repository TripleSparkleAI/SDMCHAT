# E q09 - q4f16 ONNX export practice for WebGPU

- query: `q4f16 ONNX export WebGPU MatMulNBits block size 32 transformers.js quantization`
- tool: WebSearch
- fetched: 2026-10-03 10:57 UTC

## result list
1. onnx-community/kev-4b-ONNX - https://huggingface.co/onnx-community/kev-4b-ONNX
2. occ-ai/OCC-RAG-0.6B-ONNX - https://huggingface.co/occ-ai/OCC-RAG-0.6B-ONNX
3. geeek/medgemma-1.5-4b-it-ONNX - https://huggingface.co/geeek/medgemma-1.5-4b-it-ONNX
4. geeek/medgemma-4b-it-ONNX - https://huggingface.co/geeek/medgemma-4b-it-ONNX
5. m1rhan/laya-typed-decisions-ONNX - https://huggingface.co/m1rhan/laya-typed-decisions-ONNX
6. BananaMind/BananaMind-CodeQ-1.3-2B-ONNX - https://huggingface.co/BananaMind/BananaMind-CodeQ-1.3-2B-ONNX
7. onnx-webgpu-converter (lobehub skill) - https://lobehub.com/skills/jakerains-agentskills-onnx-webgpu-converter
8. Transformers.js (promptfoo docs) - https://www.promptfoo.dev/docs/providers/transformers/
9. Skills (skills.lc) - onnx-webgpu-converter
10. Add/update the quantized ONNX model files ... Xenova/tiny-random-WhisperForConditionalGeneration discussion 1

## tool summary claims (unverified)
- model card: "4-bit MatMulNBits weights, block 32, embedding as GatherBlockQuantized ... q4f16 runs the rest of the graph in fp16 for WebGPU"
- "the 4-bit (q4) files use fp32 math, so they run on GPUs without shader-f16 support"
- tool: `onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer`

## primary check
- The op names GatherBlockQuantized and MatMulNBits were confirmed present in three published graphs (E_q22b). Block size 32 for those three graphs was not read (string count only). State of the card sentence: POST CLAIM ONLY (community model cards).
