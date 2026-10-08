# C_q09 Gemma 3 270M: embedding parameters, vocabulary, QAT

- query: "Gemma 3 270M 170 million embedding parameters 256k vocabulary quantization aware training"
- tool: WebSearch (web-mirror fallback; X API returned HTTP 402)
- fetched UTC: 2026-10-03 ~10:50
- note: the "snippet" text below is the search tool's own summary of the pages, not verbatim page text. Primary checked separately (see C_q09b).

## Result list (title, url)
1. Google Gemma3 270M : The best Smallest LLM for everything - https://medium.com/data-science-in-your-pocket/google-gemma3-270m-the-best-smallest-llm-for-everything-efcf927a74be
2. Introducing Gemma 3 270M: The compact model for hyper-efficient AI - Google Developers Blog - https://developers.googleblog.com/en/introducing-gemma-3-270m/
3. Google Open - Sources Gemma 3 with Only 0.27B Parameters ... - https://eu.36kr.com/en/p/3423760912649602
4. Google rolls out Gemma 3 270M multimodal model for phones - https://www.testingcatalog.com/google-rolls-out-gemma-3-270m-multimodal-model-for-phones-and-edge-inference/
5. Gemma 3 Technical Report - https://arxiv.org/html/2503.19786v1
6. Gemma 4 Technical Report - https://arxiv.org/pdf/2607.02770
7. Gemma 3 270M: Run AI in 125MB - https://localaimaster.com/models/gemma-3-270m
8. Gemma 3 - https://aiwiki.ai/wiki/gemma_3
9. Google introduces Gemma 3 270M for hyper-efficient on-device AI - https://www.allaboutai.com/ai-news/google-introduces-gemma-3-270m/

## Search-tool summary snippets (not verbatim page text)
- "a total of 270 million parameters: 170 million embedding parameters due to a large vocabulary size and 100 million for our transformer blocks"
- "large vocabulary of 256k tokens, the model can handle specific and rare tokens"
- Gemma 3 report: quantized versions "obtained by finetuning each model for a small number of steps, typically 5,000, using Quantization Aware Training (QAT)" using "probabilities from the non-quantized checkpoint as targets"; formats "per-channel int4, per-block int4, and switched fp8".
- "INT4 quantized model only consumes 0.75% of the battery power in 25 conversations"
- third-party: "32K token context window, about 125MB of memory at INT4, and 6 trillion training tokens" (not confirmed by Google announcement per the summary)
