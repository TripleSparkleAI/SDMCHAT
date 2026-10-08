# D_q22b primary: PleIAs/Monad config.json and model metadata
- urls: https://huggingface.co/PleIAs/Monad/raw/main/config.json , https://huggingface.co/api/models/PleIAs/Monad
- tool: python3 urllib
- fetched: 2026-10-03 10:53:52 UTC

config.json (verbatim): {"architectures": ["LlamaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 1, "eos_token_id": 2, "head_dim": 64, "hidden_act": "silu", "hidden_size": 256, "initializer_range": 0.02, "intermediate_size": 768, "max_position_embeddings": 2048, "mlp_bias": false, "model_type": "llama", "num_attention_heads": 4, "num_hidden_layers": 64, "num_key_value_heads": 4, "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 10000, "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.51.3", "use_cache": true, "vocab_size": 8192}

lastModified 2025-12-14T19:31:25.000Z createdAt 2025-11-10T13:32:17.000Z licence apache-2.0

README lines mentioning tokens / MMLU / layers / tokenizer (verbatim):
> **Monad** is a 56 million parameters generalist Small Reasoning Model, trained on 200 billions tokens from <a href="https://huggingface.co/PleIAs/Baguettotron">SYNTH</a>, a fully open generalist dataset.
> As of 2025, Monad is the best contender for the smallest viable language models. Despite being less than half of gpt-2, Monad not only answers in consistent English but performs significanly beyond chance on MMLU and other major industry benchmarks.
> Monad is strictly monolingual in English. We trained a new custom tokenizer (likely one of the smallest tokenizer to date, less than 8,000 individual tokens), exclusively trained on SYNTH so that we maintain a relatively good compression ratio.
> Monad is a 56M parameters decoders with a standard Qwen/Llama-like design, except for its extremely compact size and overall opiniated architecture for depth (with 64 layers)
> Monad was trained on 16 h100 from Jean Zay (compute plan n°A0191016886). Full pre-training took a bit less than 6 hours.
> Monad attains performance on MMLU significantly beyond chance with close to 30% of positive rate. We also find non-random results on gsm8k (8%) and HotPotQA (8%)