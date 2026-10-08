# C_q08b primary check: The Smol Training Playbook, Embedding Sharing section

- url: https://huggingfacetb-smol-training-playbook.hf.space/
- tool: WebFetch
- fetched UTC: 2026-10-03 ~10:51
- date: the playbook page carries no date in the extracted text; a reading note dated 2025-11-29 cites it (https://jianyuh.github.io/training/2025/11/29/Smol-Train.html), so it predates that.

## Verbatim quotes returned
- configurations: "our baseline 1.2B model with tied embeddings (16 layers)" vs "1.46B untied" vs "1.2B model with untied embeddings and reduced layers"
- "our baseline 1.2B model with tied embeddings achieves comparable performance to the 1.46B untied equivalent on all the benchmarks except for WinoGrande, despite having 18% fewer parameters."
- "increasing model depth provides greater benefits than untying embeddings at equivalent parameter budgets"
- "This makes embedding sharing (reusing input embeddings in the output) a natural optimization for small models"

## Note
- The search-tool summary in C_q08 said the tied baseline "beat" the untied model; the primary says "comparable performance". The primary wording is used.
