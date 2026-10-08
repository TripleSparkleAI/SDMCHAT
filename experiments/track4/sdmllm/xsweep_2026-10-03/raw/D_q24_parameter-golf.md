# D_q24 Parameter Golf (FineWeb bpb at tiny scale)
- query: `OpenAI parameter golf FineWeb bpb 16MB leaderboard data tricks`
- tool: WebSearch (two sub-searches)
- fetched: 2026-10-03 ~10:58 UTC

Results (verbatim title, url):
1. OpenAI's Parameter Golf: Train a 16MB Language Model - https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod
2. incrediblecrab/2026-parameter-golf - https://github.com/incrediblecrab/2026-parameter-golf
3. mellowyellow71/parameter-golf - https://github.com/mellowyellow71/parameter-golf
4. FAQ - Parameter Golf - https://www.mintlify.com/openai/parameter-golf/reference/faq
5. What Parameter Golf taught us - https://openai.com/index/parameter-golf/ (and /index/what-parameter-golf-taught-us/)
6. Leaderboard - https://openai-parameter-golf.mintlify.app/leaderboard
7. Scoring & Evaluation - https://openai-parameter-golf.mintlify.app/concepts/scoring
8. aitoolsclub article ; deepwiki openai/parameter-golf
9. Parameter Golf: What Really Works? - https://arxiv.org/pdf/2607.01517
10. Record: CaseOps Tokenizer + Tapered WD - val_bpb 1.0678 (3-seed mean) by romeerp - https://github.com/openai/parameter-golf/pull/1729
11. edawite/openai-golf ; openai/parameter-golf ; Divyesh-Thirukonda/parameter-golf ; codesota leaderboard page

Primary abstract (arXiv API, 2607.01517, published 2026-07-01, verbatim): "The verified leaderboard score dropped from 1.2244 to 1.058 BPB across three phases, a 13.6% reduction, despite individual techniques rarely improving BPB by more than 1%. We show that most techniques' gains shrink when re-measured among competitive submissions, isolating the few methods that help regardless of the surrounding stack."
Tool summary fragments (unverified): contest ran 2026-03-18 to 2026-04-30; "tokenization techniques (SP8192, CaseOps) spread sharply through Phase 2, reaching over 70% by Phase 3".
In window: 2607.01517 (widened). Contest itself is outside window.
