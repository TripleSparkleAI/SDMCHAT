# C_q11 nanochat untied lm_head and embedding learning rates (web)

- query: "nanochat untied lm_head embedding learning rate Adam vocab 32768"
- tool: WebSearch
- fetched UTC: 2026-10-03 ~10:50
- note: snippets are the search tool's summary; primary checked in C_q11b (GitHub source).

## Result list
1. Training a 3.8B LLM to 0.384 CORE for $998 - https://hugovergnes.github.io/little-lm-3-8b/
2. HarleyCooper/nanochat561 - https://huggingface.co/HarleyCooper/nanochat561
3. Inside nanochat Part 3: Understanding Optimizers - https://medium.com/@akdemir_bahadir/inside-nanochat-part-3-understanding-optimizers-7f989d407153
4. nanochat-analysis.md gist - https://gist.github.com/JustinAngel2/acc3a9da5369456be19cab5d3ea9ef07
5. Nonsmooth Optimization via Orthogonalized Momentum - https://arxiv.org/pdf/2609.13677
6. Can Muon Fine-tune Adam-Pretrained Models? - https://arxiv.org/pdf/2605.10468
7. Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size - https://arxiv.org/pdf/2506.15025
8. Quick Reference (nanochat wiki) - https://thegovind.github.io/nanochat-wiki/getting-started/quick-reference
9. How Much Is an AI Token Worth? Scaling Laws for Wild AI-Generated Web Text - https://arxiv.org/html/2609.40295v1
10. Lost in Backpropagation: The LM Head is a Gradient Bottleneck - https://arxiv.org/pdf/2603.10145

## Search-tool summary
- older nanochat defaults: "unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02", AdamW lrs scaled by "(model_dim / 768) ** -0.5", AdamW "betas=(0.8, 0.95) and eps=1e-10".
- a third-party wiki lists embedding lr "Default: 0.3".
- nanochat tokenizer default vocab 32768; embeddings and lm_head untied.
