# X API status, 2026-10-03

Tool: `x_probe.py` (this folder), python3 urllib, GET https://api.x.com/2/tweets/search/recent, Bearer token
read from the environment by name and never printed.

| call | query | fetched (UTC) | HTTP | x-rate-limit-remaining | body |
|---|---|---|---|---|---|
| 1 | `"modded-nanogpt" -is:retweet` | see `x_q01_modded_nanogpt.json` | 402 | 449 of 450 | `{"detail":"credits depleted","status":402,"title":"Payment Required","type":"https://api.x.com/2/problems/credits-depleted"}` |
| 2 (the one permitted retry) | same | see `x_q01_modded_nanogpt_retry.json` | 402 | 448 of 450 | identical |

This is a 402 (credit pool empty), not a 429 (rate limit): the rate-limit budget was nearly untouched. Per the
method, the sweep stopped calling X after the single retry and fell back to a web-mirror sweep (web search,
arXiv Atom API, GitHub REST API, Hugging Face Hub API). Posts on x.com that a search engine surfaces are cited
as "via search, X not fetched", and their engagement is recorded as unknown, never estimated.
