# Band A search matrix (written before any query ran)

Written 2026-10-03 10:47 UTC. Window 2026-08-01 to 2026-10-03, widen to 2026-06 if thin.
The X API returned HTTP 402 (credits depleted), so every query below uses the web-mirror fallback.

| id | target | tool |
|---|---|---|
| q01 | modded-nanogpt README record table | GitHub API (raw README) |
| q02 | modded-nanogpt commits since 2026-08-01 | GitHub API commits |
| q03 | modded-nanogpt pull requests (recent, all states) | GitHub API pulls |
| q04 | nanochat commits since 2026-08-01 | GitHub API commits |
| q05 | nanochat dev/LOG.md or equivalent changes | GitHub API contents/commits by path |
| q06 | "speedrun" GPT records 2026 | WebSearch |
| q07 | Muon variants (NorMuon, AdaMuon, weight decay, PolarGrad, Dion) | arXiv Atom API |
| q08 | Muon for embeddings / output head | WebSearch + arXiv |
| q09 | schedule-free and new optimisers 2026 | arXiv Atom API |
| q10 | SOAP / Shampoo / Adam-mini / MARS / Lion follow-ups 2026 | arXiv Atom API |
| q11 | embedding initialisation / tied embedding small LM | arXiv Atom API |
| q12 | value embeddings / logit soft-capping / z-loss | WebSearch + arXiv |
| q13 | norm placement in small LMs 2026 | arXiv Atom API |
| q14 | u-muP / muP for small models 2026 | arXiv Atom API |
| q15 | learning-rate and batch-size scaling small models 2026 | arXiv Atom API |
| q16 | tokens-per-parameter overtraining 2026 | arXiv Atom API + WebSearch |
| q17 | large vocabulary output head training cost (sampled softmax, sparse embedding update) | arXiv Atom API |
| q18 | product-key memory / memory layer training recipes 2026 | arXiv Atom API |
| q19 | x.com mirror: modded-nanogpt / NanoGPT speedrun record via search | WebSearch |
| q20 | Hugging Face Hub: small LM pretraining releases Aug-Oct 2026 | HF API |

Every query result lands in raw/A_qNN_slug.md before it is judged.
