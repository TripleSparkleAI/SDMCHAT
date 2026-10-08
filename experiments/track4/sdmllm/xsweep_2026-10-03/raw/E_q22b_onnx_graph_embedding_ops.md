# E q22b - HF Hub: what the small-model ONNX exports do with the embedding and the head

- tool: python3 urllib, downloaded three small ONNX graph files (weights are external .onnx_data, not downloaded) and counted byte strings. The `onnx` python package is not installed, so this is a string count, not a parsed graph.
- fetched: 2026-10-03 10:52 UTC

## raw output
```
== onnx-community/gemma-3-270m-it-ONNX onnx/model_q4f16.onnx 330717 bytes
   GatherBlockQuantized 1
   MatMulNBits 127
   GroupQueryAttention 90
   Gather 43
   embed_tokens 20
   lm_head 2
   names: ['lm_head/MatMul_Quant', 'lm_head/Transpose/output_0', 'model.embed_tokens.weight']
== LiquidAI/LFM2.5-230M-ONNX onnx/model_q4.onnx 154010 bytes
   GatherBlockQuantized 6
   MatMulNBits 83
   GroupQueryAttention 30
   Gather 60
   embed_tokens 19
   lm_head 0
   names: []
== onnx-community/LFM2-350M-ONNX onnx/model_q4f16.onnx 182827 bytes
   GatherBlockQuantized 1
   MatMulNBits 93
   GroupQueryAttention 30
   Gather 33
   embed_tokens 16
   lm_head 14
   names: ['lm_head/MatMul_Quant', 'lm_head/Transpose/output_0', 'lm_head/num_logits_to_keep/Neg', 'lm_head/num_logits_to_keep/Neg/output_0', 'lm_head/num_logits_to_keep/Slice', 'lm_head/num_logits_to_keep/Slice/output_0', 'lm_head/num_logits_to_keep/Unsqueeze', 'lm_head/num_logits_to_keep/Unsqueeze/output_0', 'model.embed_tokens.weight']
```

## reading (string evidence only)
- All three graphs contain the op name `GatherBlockQuantized`, the ONNX Runtime contrib op for a block-quantized embedding gather, and `MatMulNBits` for the 4-bit matmuls.
- Two graphs contain node names `lm_head/MatMul_Quant` and `lm_head/Transpose`, which is consistent with the tied head being run as a separate quantized MatMulNBits over the transposed embedding. Whether the export stores the 4-bit weights once or twice is NOT established by a string count.
- LFM2-350M carries `lm_head/num_logits_to_keep/Slice`: the export slices to the last position before the head, so prefill does not pay the full vocab matmul per prompt token.
- state: CONFIRMED that the op names occur in the published graphs; the storage layout is UNVERIFIED.
