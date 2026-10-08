# Band C search matrix (written before any query ran)

Written 2026-10-03 10:47 UTC. Band C: large-vocabulary output-layer tricks.
Window 2026-08-01 to 2026-10-03, widened to 2026-01 where thin. Older strong primaries allowed when clearly dated.
The X API returned HTTP 402 (credits depleted), so every query below uses the web-mirror fallback.

| id | target | tool |
|---|---|---|
| q01 | modded-nanogpt PR #360 (body, numbers, files) | GitHub REST API |
| q02 | modded-nanogpt PRs after #360 that touch the head, embedding or sampled softmax | GitHub REST API |
| q03 | sampled softmax in LM pretraining 2026 | arXiv Atom API |
| q04 | sampled softmax / negative sampling / candidate sampling pretraining 2026 | WebSearch |
| q05 | low-rank / factorised / bottleneck LM output layer 2026 | arXiv Atom API |
| q06 | adaptive softmax / hierarchical softmax revival 2026 | arXiv Atom API + WebSearch |
| q07 | tied vs untied embeddings small models 2026 | arXiv Atom API |
| q08 | SmolLM3 tie_word_embeddings and reason | HF Hub config + WebSearch |
| q09 | Gemma 3 270M embedding params, training and quantisation | WebSearch + Google primary |
| q10 | Qwen3-0.6B tie_word_embeddings | HF Hub config |
| q11 | nanochat untied head / lm_head setup | GitHub REST API |
| q12 | vocabulary size for small models 2026 (over-tokenized, vocab trimming, pruning) | arXiv Atom API |
| q13 | tokenizer transplant / vocabulary trimming for small models | WebSearch |
| q14 | int8/int4 quantisation of large embedding tables on device | arXiv Atom API + WebSearch |
| q15 | z-loss / logit cap for large vocab | arXiv Atom API |
| q16 | sparse embedding gradients with tied heads in PyTorch | WebSearch |
| q17 | x.com mirror: sampled softmax / lm_head / embedding speedrun posts | WebSearch |
| q18 | output embedding / lm head gradient (rare tokens, anisotropy) 2026 | arXiv Atom API |
| q19 | HF Hub: small LM configs Aug-Oct 2026 (tie_word_embeddings, vocab) | HF API |
| q20 | follow-up on any strong lead | as needed |

Every query result lands in raw/C_qNN_slug.md before it is judged.
