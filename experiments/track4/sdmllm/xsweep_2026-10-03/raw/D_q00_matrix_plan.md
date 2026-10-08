# Band D (data) - search matrix, written before any query

Written 2026-10-03 10:47 UTC. Window 2026-08-01 to 2026-10-03, widened to 2026-06 where empty.
X API unavailable (HTTP 402, credits depleted). Fallback: WebSearch, WebFetch, python3 urllib against the
arXiv Atom API, the GitHub REST API (unauthenticated) and the HF Hub API.

| # | row | query intent | tool |
|---|---|---|---|
| D_q01 | small-scale ablations | small-scale pretraining data ablation 2026, <500M params, 100M-1B tokens | WebSearch |
| D_q02 | small-scale ablations | arXiv: "data ablation" small language model pretraining 2026 | arXiv API |
| D_q03 | FineWeb family | FineWeb-Edu new version / FineWeb 2026 release | WebSearch + HF API |
| D_q04 | FinePDFs | finepdfs-edu results small models | WebSearch + HF card |
| D_q05 | DCLM successors | DCLM successor 2026 dataset | WebSearch |
| D_q06 | synthetic | Cosmopedia v3 / synthetic textbooks pretraining 2026 | WebSearch + HF API |
| D_q07 | synthetic | BeyondWeb / rephrased web results | WebSearch + arXiv |
| D_q08 | synthetic | Nemotron synthetic pretraining data small models | WebSearch |
| D_q09 | curricula | TinyStories-like curriculum 2026 | WebSearch + arXiv |
| D_q10 | chat SFT data | SmolTalk2 / smol-smoltalk card and mix | HF API + card |
| D_q11 | chat SFT data | Gemma 3 270M / LFM2 / Qwen3-0.6B instruct data | WebSearch |
| D_q12 | chat SFT data | chat SFT under 200M params results 2026 | WebSearch + arXiv |
| D_q13 | chat SFT data | Tulu / magpie small / OpenHermes successors 2026 | WebSearch + HF API |
| D_q14 | dedup / quality | dedup and quality classifier at small scale 2026 | WebSearch + arXiv |
| D_q15 | books | Common Corpus / Institutional Books / Common Pile licence and status | WebSearch + HF API |
| D_q16 | books | philosophy public-domain text dataset | HF API |
| D_q17 | community | nanochat / modded-nanogpt data swaps reported on GitHub 2026 | GitHub API |
| D_q18 | community | x.com posts on small model data ablations (via search, X not fetched) | WebSearch |
| D_q19 | HF recent | HF Hub datasets sorted by lastModified, search "pretrain" / "edu" | HF API |
| D_q20 | arXiv recent | arXiv: synthetic pretraining data 2026-08..10 | arXiv API |

Every query gets raw/D_qNN_<slug>.md written before judging: exact query, tool, UTC fetch time, verbatim
result list, or EMPTY, or the exact error.
