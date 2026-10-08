# E q18 - OPFS / Cache API for model weights

- query: `OPFS Cache API storing LLM model weights browser quota 2026`
- tool: WebSearch
- fetched: 2026-10-03 10:57 UTC

## result list
1. Cache converted weights in the browser · Issue #77 · Nehanth/swarmllm - https://github.com/Nehanth/swarmllm/issues/77
2. WebLLM weights: use the OPFS cache backend so the 303 MB and 485 MB shards can be stored · Issue #70 · OlehZhyhinas/LatexGen - https://github.com/OlehZhyhinas/LatexGen/issues/70
3. Store WebLLM weights through the OPFS backend ... · PR #72 · OlehZhyhinas/LatexGen - https://github.com/OlehZhyhinas/LatexGen/pull/72
4. Prefetch WebLLM weights with a balanced 8-way pool into the OPFS store · PR #73 · OlehZhyhinas/LatexGen - https://github.com/OlehZhyhinas/LatexGen/pull/73
5. How to Store Files on a User's Device Using OPFS (Telerik) - https://www.telerik.com/blogs/how-store-files-user-device-opfs
6. GitHub - P0u4a/opfs-cache - https://github.com/P0u4a/opfs-cache
7. Storage quotas and eviction criteria - MDN - https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria
8. Qwen (Wikipedia)
9. WebLLM のモデル保存方式を OPFS 既定にし、Quota exceeded を診断できるようにする · PR #3 · chihirohashimotoac-coder/01ArrangementSupport-beta - https://github.com/chihirohashimotoac-coder/01ArrangementSupport-beta/pull/3

## tool summary claims (unverified unless checked)
- "WebLLM 0.2.84 ships an OPFS artifact cache ... selected by appConfig.cacheBackend = \"opfs\""
- PR #72/#73: Chromium build fails cache.add above roughly 250 MB; 4 connections ~40 MB/s, 8 connections ~74 MB/s.
- MDN: storage is best-effort by default and can be evicted.

## primary check: LatexGen issue #70 (WebFetch 10:58 UTC)
- issue date September 14, 2026
- error: "UnknownError: Unexpected internal error" on `cache.add` of large shards
- shard sizes: Qwen 3.5 4B 303 MB, Qwen 3.5 9B 485 MB, MiniCPM5 2B 128 MB
- "at least one Chromium build"; an earlier attempt on September 5, 2026 "hit the same wall at 328 MB"
- OPFS write 10.4 s, read 0.1 s for the 303 MB shard
- the "roughly 250 MB" threshold is in PR #72 per the tool; not opened, UNVERIFIED.
- state: CONFIRMED as one developer's report on one unnamed Chromium build. Not a browser vendor statement.
