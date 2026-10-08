# A_q22 web search: optimiser state for sparsely touched table rows (product-key memory values)

- Query: "product-key memory layer value table optimizer sparse rows Adam second moment stale 2026"
- Tool: WebSearch
- Fetch UTC: 2026-10-03 ~10:54
- Hits: 10

## Verbatim result list

1. 2026-2-24 Fast-weight Product Key Memory, Tianyu Zhao and Llion Jones - https://arxiv.org/pdf/2601.00671 (Jan 2026, outside window)
2. GitHub - KellerJordan/modded-nanogpt - https://github.com/kellerjordan/modded-nanogpt
3. Blog: Survey of Optimizers - https://arxiv.org/html/2608.28557 (2026-08-28)
4. Blog: Survey of Optimizers (pdf) - https://arxiv.org/pdf/2608.28557
5. Weight-sparse transformers have interpretable circuits - https://arxiv.org/pdf/2511.13653
6. [2601.00671] Fast-weight Product Key Memory - https://arxiv.org/abs/2601.00671
7. No Subspace to Track: Non-Identifiability and Optimizer State in Low-Rank Training - https://arxiv.org/pdf/2607.05872 (2026-07)
8. Memory-Efficient LLM Training with Dynamic Sparsity: From Stability to Practical Scaling - https://arxiv.org/pdf/2606.00888 (2026-06)
9. AdaLomo: Low-memory Optimization with Adaptive Learning Rate - https://arxiv.org/pdf/2310.10195
10. Fast-weight Product Key Memory (html) - https://arxiv.org/html/2601.00671

## Primary checks

- modded-nanogpt README (A_q01 fetch, line 46), verbatim: "Sparse row-wise embedding updates: only touched rows are exchanged and updated (beta1 = 0, one second-moment value per row), every 4 steps". CONFIRMED in the README. This is part of record 92 (already known in outline); the detail beta1 = 0 and one second moment per row is recorded here because it matches the nanochat PR 854 recipe and Ember (A_q18).
- nanochat PR 854 body (A_q05), verbatim: "Row-wise optimiser on the GPU: state is one scalar per row and Engram layer, kept in VRAM." and "Rows are zero at init and are trained by a row-wise RMSprop update". CONFIRMED in the PR body (open PR, not merged).
- The search summary attributed to 2607.05872 a measured lag table versus beta2 on Pythia-160M; the primary was not opened, so that is UNVERIFIED and not used.
- No 2026 source found that studies stale second moments for a product-key value table directly. Result: EMPTY for the exact question.
