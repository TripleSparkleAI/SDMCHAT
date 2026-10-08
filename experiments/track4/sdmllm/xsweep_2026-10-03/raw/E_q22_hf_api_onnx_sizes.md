# E q22 - HF Hub API: file sizes of small browser-model exports

- tool: python3 urllib, huggingface.co/api/models/<repo>/tree/main?recursive=true; files over 1 MB with model extensions printed
- two repos returned HTTP 401 (gated or absent): onnx-community/LFM2.5-230M-ONNX, onnx-community/SmolLM3-3B-ONNX

```
fetched 2026-10-03T10:49:03.925538Z
== onnx-community/Qwen3-0.6B-ONNX
       315.1 MB  onnx/model.onnx
      2090.3 MB  onnx/model.onnx_data
       891.6 MB  onnx/model_bnb4.onnx
      1202.8 MB  onnx/model_fp16.onnx
       617.7 MB  onnx/model_int8.onnx
       919.1 MB  onnx/model_q4.onnx
       569.8 MB  onnx/model_q4f16.onnx
       617.7 MB  onnx/model_quantized.onnx
       617.7 MB  onnx/model_uint8.onnx
       524.6 MB  onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128/model.onnx
        11.4 MB  onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128/tokenizer.json
       504.7 MB  onnxruntime/cuda/cuda-int4-kld-block-128/model.onnx
        11.4 MB  onnxruntime/cuda/cuda-int4-kld-block-128/tokenizer.json
       543.3 MB  onnxruntime/webgpu/webgpu-int4-kld-block-32/model.onnx
        11.4 MB  onnxruntime/webgpu/webgpu-int4-kld-block-32/tokenizer.json
         9.1 MB  tokenizer.json
         2.8 MB  vocab.json
  total 9462.2 MB
== onnx-community/gemma-3-270m-it-ONNX
      1139.5 MB  onnx/model.onnx_data
       569.9 MB  onnx/model_fp16.onnx_data
       322.9 MB  onnx/model_q4.onnx_data
       272.6 MB  onnx/model_q4f16.onnx_data
       545.0 MB  onnx/model_quantized.onnx_data
        20.3 MB  tokenizer.json
  total 2876.3 MB
== onnx-community/LFM2.5-230M-ONNX ERROR <HTTPError 401: 'Unauthorized'>
== LiquidAI/LFM2.5-230M-ONNX
       951.5 MB  onnx/model.onnx_data
       475.8 MB  onnx/model_fp16.onnx_data
       211.1 MB  onnx/model_q4.onnx_data
       403.0 MB  onnx/model_q4f32.onnx_data
       489.3 MB  onnx/model_q8.onnx_data
         4.7 MB  tokenizer.json
  total 2536.3 MB
== onnx-community/SmolLM3-3B-ONNX ERROR <HTTPError 401: 'Unauthorized'>
== HuggingFaceTB/SmolLM2-135M-Instruct
       269.1 MB  model.safetensors
       540.3 MB  onnx/model.onnx
       175.4 MB  onnx/model_bnb4.onnx
       270.3 MB  onnx/model_fp16.onnx
       137.1 MB  onnx/model_int8.onnx
       182.1 MB  onnx/model_q4.onnx
       117.7 MB  onnx/model_q4f16.onnx
       137.1 MB  onnx/model_quantized.onnx
       137.1 MB  onnx/model_uint8.onnx
         2.1 MB  tokenizer.json
  total 1969.8 MB
== onnx-community/LFM2-350M-ONNX
      1450.7 MB  onnx/model.onnx_data
       725.4 MB  onnx/model_fp16.onnx_data
       293.6 MB  onnx/model_q4.onnx_data
       255.0 MB  onnx/model_q4f16.onnx_data
       509.9 MB  onnx/model_quantized.onnx_data
         3.3 MB  tokenizer.json
  total 3238.8 MB
== mlc-ai/Qwen3-0.6B-q4f16_1-MLC
        77.8 MB  params_shard_0.bin
        32.7 MB  params_shard_1.bin
        31.9 MB  params_shard_2.bin
        33.5 MB  params_shard_3.bin
        32.0 MB  params_shard_4.bin
        31.9 MB  params_shard_5.bin
        33.5 MB  params_shard_6.bin
        32.0 MB  params_shard_7.bin
        30.1 MB  params_shard_8.bin
        11.4 MB  tokenizer.json
         2.8 MB  vocab.json
  total 351.5 MB
```

## q22 follow-up: MLC tensor-cache.json and configs (python3 urllib, 10:51 UTC)
```
params_shard_0.bin 77791232
    model.embed_tokens.q_weight [151936, 128] uint32 f32-to-bf16 77791232
params_shard_1.bin 32740608
    model.embed_tokens.q_scale [151936, 32] float16 f32-to-bf16 9723904
    model.layers.0.input_layernorm.weight [1024] float16 f32-to-bf16 2048
    model.layers.0.mlp.down_proj.q_weight [1024, 384] uint32 f32-to-bf16 1572864
    ...
n shards 9
Qwen/Qwen3-0.6B {'vocab_size': 151936, 'hidden_size': 1024, 'tie_word_embeddings': True, 'num_hidden_layers': 28}
google/gemma-3-270m <HTTPError 401: 'Unauthorized'>
LiquidAI/LFM2.5-230M {'vocab_size': 65536, 'hidden_size': 1024, 'tie_word_embeddings': True, 'num_hidden_layers': 14}
LiquidAI/LFM2.5-350M {'vocab_size': 65536, 'hidden_size': 1024, 'tie_word_embeddings': None, 'num_hidden_layers': 16}
```
- reading: MLC's q4f16_1 build of Qwen3-0.6B stores the tied 151,936 x 1024 table as packed 4-bit (128 uint32 per row = 1024 nibbles) with one fp16 scale per 32 values (32 scales per row). 77.8 MB + 9.7 MB = 87.5 MB for the table, 24.9% of the 351.5 MB download.
- transformers.js q4f16 of the same model is 569.8 MB, 1.62x the MLC build. The cause is not established here (graph not parsed).
- state: CONFIRMED (HF API and the published manifest).
