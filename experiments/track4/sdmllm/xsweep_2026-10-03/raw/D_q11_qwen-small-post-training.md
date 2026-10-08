# D_q11 Qwen3-0.6B / Qwen3.5-0.8B instruct data
- query: `Qwen3-0.6B post-training data OR "Qwen3.5-0.8B" SFT data small model distillation 2026`
- tool: WebSearch (three sub-searches)
- fetched: 2026-10-03 ~10:56 UTC

Results (verbatim title, url):
1. Qwen3.5 Small Models: 0.8B-9B Specs & Setup - https://www.therundown.ai/tools/qwen3-5-small
2. Are the Qwen3 base models post-distillation? - https://github.com/QwenLM/Qwen3/discussions/1367
3. Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities - 2602.16093
4. Qwen3 Technical Report - https://arxiv.org/pdf/2505.09388 (+ html, alphaxiv, themoonlight, jsdelivr copy)
5. Qwen/Qwen3.5-0.8B - https://huggingface.co/Qwen/Qwen3.5-0.8B
6. LLM Post-Training deep dive 2502.21321 ; Neural Thickets 2603.12228 ; WinDOM 2606.25964 ; kili-technology Qwen3 data story ; Small Models Struggle to Learn from Strong Reasoners 2502.12143
7. Eliciting Weak-to-Strong Generalization with On-Policy Reverse Distillation - 2609.08798 ; Uni-OPD 2605.03677 ; Weak-to-Strong On-Policy Distillation 2607.26246
8. medium posts on Qwen 3.5 small ; Effective Learning for Small Reasoning Models 0.5B 2506.13404 ; Qwen2 report ; Reliability-Aware Distillation 2607.19956 ; Making Small LMs Efficient Reasoners 2505.07961 ; EasyDistill 2505.20888

In window: 2609.08798 (distillation method, Qwen3-0.6B student), WinDOM, 2607.26246. Data not released for Qwen3-0.6B or Qwen3.5-0.8B (tool summary). Relevance to a from-scratch 16-60M model: low (logit distillation needs a same-tokenizer teacher).
