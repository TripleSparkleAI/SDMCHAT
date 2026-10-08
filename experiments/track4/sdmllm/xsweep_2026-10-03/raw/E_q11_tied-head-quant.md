# E q11 - tied lm_head quantization large vocabulary

- query: `tied lm_head quantization large vocabulary output layer small model on-device int4 accuracy`
- tool: WebSearch
- fetched: 2026-10-03 10:54 UTC

## result list
1. Rethinking Small VLM Quantization (arXiv 2607.08029) - https://arxiv.org/html/2607.08029
2. A Method for Layer Bit-Width Allocation in LLM Quantization ... (arXiv 2608.28003) - https://arxiv.org/pdf/2608.28003
3. ARCHead: Activation-Metric Residual Correction for Large Language Model Output Heads (arXiv 2608.02703) - https://arxiv.org/pdf/2608.02703
4. Compress and Forget: bitsandbytes Quantization Amplifies Proactive Interference in LLMs (arXiv 2608.18578) - https://arxiv.org/pdf/2608.18578
5. Understanding INT4 Quantization for Language Models (arXiv 2301.12017) - https://arxiv.org/pdf/2301.12017
6. Understanding Post-Training Quantization with LLM Compressor (HF blog) - https://huggingface.co/blog/rishiraj/llm-compressor
7. Deep Dive into Quantization of LLMs (substack) - https://bhavishyapandit9.substack.com/p/deep-dive-into-quantization-of-llms
8. 4-bit Quantization of LSTM-based Speech Recognition Models - https://arxiv.org/pdf/2108.12074
9. MobileLLM (arXiv 2402.14905) - https://arxiv.org/pdf/2402.14905

## tool summary claims (unverified unless checked)
- ARCHead "row INT8 added only about 1.7% to perplexity" (tool summary; not in the abstract; UNVERIFIED)
- bitsandbytes study: quantizing lm_head produced "no statistically detectable change in accuracy" in three 4B-7B models (UNVERIFIED, abstract not fetched)

## primary check: arXiv Atom API, 2608.02703 (fetched 10:54 UTC)
- published 2026-08-03T14:40:59Z
- abstract verbatim excerpts: "practical backends often retain the final language-modeling head (LM-head) in BF16 or FP16. Quantizing this projection naively can strongly perturb the vocabulary-logit distribution." ... "On Qwen3-8B-Base, it uses 25.6% of BF16 head storage while attaining 1.007 relative perplexity; storage-matched naive INT4 yields 1.14-1.16."
- state: CONFIRMED (abstract). The model is 8B with hidden 4096; transfer to a d 256-768 head is not shown in the abstract.
