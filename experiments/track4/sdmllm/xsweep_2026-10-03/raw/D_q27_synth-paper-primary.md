# D_q27 primary: arXiv html 2609.37891v1 (It's All Training / SYNTH) via WebFetch
- url: https://arxiv.org/html/2609.37891v1
- tool: WebFetch (extraction prompt asked for verbatim quotes)
- fetched: 2026-10-03 ~10:52 UTC

Quotes returned by the fetch tool:
- Section 4.2 Data ablations: FineWiki "~9B tokens, ~17 epochs"; FinePDFs-Edu "~130B tokens, ~1.2 epochs"
- Section 4.2: "Synth leads by 16-17 points compared to both post-trained web models on multiple-choice and 11-14 points on open-ended tasks"
- Section 4.2: "We therefore post-train both on SmolTalk...plus the MMLU auxiliary training split...~100M tokens over 3 epochs"
- Monad: "~180B tokens", "custom 8k-vocabulary tokenizer", "56M parameters, 64 layers, dmodel=384" (the HF config.json says hidden_size 256, see D_q22b), "outperforming Gemma-3-270M and SmolLM2-360M on average" (multiple-choice)
- Abstract: "At iso-compute, Synth outperforms filtered web data, and our models remain competitive with similarly-sized open-weight baselines."
- Section 3: "almost 80B tokens of synthetic text in 8 languages"; "About 20% of Synth is non-English"
Caveats: the extraction is a tool reading; the 600M ablation compares a model trained on SYNTH with web models that needed a separate post-training step, and the benchmarks are task accuracy, not held-out web bpb.
