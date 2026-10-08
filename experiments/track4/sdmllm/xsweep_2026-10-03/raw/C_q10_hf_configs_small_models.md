# C_q10 HF Hub config.json read: tie_word_embeddings, vocab_size, hidden_size for small models

- query: GET https://huggingface.co/<repo>/raw/main/config.json for each repo below
- tool: python3 urllib, unauthenticated
- note: gated repos return 401; that is recorded as the exact error

- HuggingFaceTB/SmolLM3-3B | fetched 2026-10-03T10:49:53.546376Z | tie_word_embeddings=True vocab_size=128256 hidden_size=2048 num_hidden_layers=36 model_type=smollm3
- HuggingFaceTB/SmolLM2-135M | fetched 2026-10-03T10:49:53.944155Z | tie_word_embeddings=True vocab_size=49152 hidden_size=576 num_hidden_layers=30 model_type=llama
- HuggingFaceTB/SmolLM2-360M | fetched 2026-10-03T10:49:54.333847Z | tie_word_embeddings=True vocab_size=49152 hidden_size=960 num_hidden_layers=32 model_type=llama
- google/gemma-3-270m | fetched 2026-10-03T10:49:54.754847Z | ERROR <HTTPError 401: 'Unauthorized'>
- Qwen/Qwen3-0.6B | fetched 2026-10-03T10:49:55.133646Z | tie_word_embeddings=True vocab_size=151936 hidden_size=1024 num_hidden_layers=28 model_type=qwen3
- Qwen/Qwen3-1.7B | fetched 2026-10-03T10:49:55.554897Z | tie_word_embeddings=True vocab_size=151936 hidden_size=2048 num_hidden_layers=28 model_type=qwen3
- LiquidAI/LFM2-350M | fetched 2026-10-03T10:49:55.935019Z | tie_word_embeddings=None vocab_size=65536 hidden_size=1024 num_hidden_layers=16 model_type=lfm2
- google/gemma-3-1b-pt | fetched 2026-10-03T10:49:56.311940Z | ERROR <HTTPError 401: 'Unauthorized'>
- meta-llama/Llama-3.2-1B | fetched 2026-10-03T10:49:56.756738Z | ERROR <HTTPError 401: 'Unauthorized'>
- microsoft/Phi-4-mini-instruct | fetched 2026-10-03T10:49:57.127925Z | tie_word_embeddings=True vocab_size=200064 hidden_size=3072 num_hidden_layers=32 model_type=phi3
- PleIAs/Baguettotron | fetched 2026-10-03T10:49:57.518185Z | tie_word_embeddings=True vocab_size=65536 hidden_size=576 num_hidden_layers=80 model_type=llama
- Qwen/Qwen3.5-0.8B | fetched 2026-10-03T10:49:57.898319Z | tie_word_embeddings=True vocab_size=248320 hidden_size=1024 num_hidden_layers=24 model_type=qwen3_5
- google/gemma-4-E2B | fetched 2026-10-03T10:49:58.312291Z | tie_word_embeddings=True vocab_size=262144 hidden_size=1536 num_hidden_layers=35 model_type=gemma4
- ibm-granite/granite-4.0-350m | fetched 2026-10-03T10:49:58.691570Z | tie_word_embeddings=True vocab_size=100352 hidden_size=1024 num_hidden_layers=28 model_type=granitemoehybrid