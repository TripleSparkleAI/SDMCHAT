# E q23 - HF Hub: onnx-community/gemma-4-E2B-it-ONNX file sizes

- tool: python3 urllib, huggingface.co/api/models/<repo>/tree/main/onnx
```
fetched 2026-10-03T10:55:34.041295Z
== onnx-community/gemma-4-E2B-it-ONNX
         0.3 MB  onnx/audio_encoder_q4f16.onnx
       171.3 MB  onnx/audio_encoder_q4f16.onnx_data
         0.7 MB  onnx/decoder_model_merged_q4f16.onnx
      1519.7 MB  onnx/decoder_model_merged_q4f16.onnx_data
         0.0 MB  onnx/embed_tokens.onnx
      1610.6 MB  onnx/embed_tokens.onnx_data
      9395.2 MB  onnx/embed_tokens.onnx_data_1
         0.0 MB  onnx/embed_tokens_fp16.onnx
       805.3 MB  onnx/embed_tokens_fp16.onnx_data
      4697.6 MB  onnx/embed_tokens_fp16.onnx_data_1
         0.0 MB  onnx/embed_tokens_q4.onnx
      1762.7 MB  onnx/embed_tokens_q4.onnx_data
         0.0 MB  onnx/embed_tokens_q4f16.onnx
      1590.7 MB  onnx/embed_tokens_q4f16.onnx_data
         0.0 MB  onnx/embed_tokens_quantized.onnx
       465.6 MB  onnx/embed_tokens_quantized.onnx_data
      2348.8 MB  onnx/embed_tokens_quantized.onnx_data_1
       367.0 MB  onnx/embed_tokens_quantized.onnx_data_2
         0.2 MB  onnx/vision_encoder_q4f16.onnx
        99.2 MB  onnx/vision_encoder_q4f16.onnx_data
```

- reading: the q4f16 token-embedding graph (embed_tokens_q4f16.onnx_data, 1590.7 MB) is larger than the q4f16 decoder (1519.7 MB). The embeddings ship as a separate ONNX graph from the decoder. Gemma 4 E2B carries per-layer embeddings (the DEV article in q10 names `embed_tokens_per_layer` 4.375 GiB bf16), so this file is not only the 262k x d input table.
- state: CONFIRMED (HF API sizes).
