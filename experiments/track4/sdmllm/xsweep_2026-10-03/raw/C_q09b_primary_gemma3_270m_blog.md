# C_q09b primary check: Google Developers Blog, Introducing Gemma 3 270M

- url: https://developers.googleblog.com/en/introducing-gemma-3-270m/
- tool: WebFetch (page converted to markdown, quotes extracted by the fetch tool's reader)
- fetched UTC: 2026-10-03 ~10:51
- publication date as stated on the page: AUG. 14, 2025

## Verbatim quotes returned
- "Our new model has a total of 270 million parameters: 170 million embedding parameters due to a large vocabulary size and 100 million for our transformer blocks."
- "Thanks to the large vocabulary of 256k tokens, the model can handle specific and rare tokens, making it a strong base model to be further fine-tuned in specific domains and languages."
- "Quantization-Aware Trained (QAT) checkpoints are available, enabling you to run the models at INT4 precision with minimal performance degradation, which is essential for deploying on resource-constrained devices."

## Not found on the page
- No statement on how the embedding table itself was trained, untied or tied, or whether the embedding was quantized differently from the blocks.
- HF config.json for google/gemma-3-270m returned HTTP 401 (gated), see C_q10.
