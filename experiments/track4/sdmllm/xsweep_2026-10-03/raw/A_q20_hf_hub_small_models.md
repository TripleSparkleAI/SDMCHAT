# A_q20 Hugging Face Hub: recent small-model pretraining releases

- Tool: python3 urllib, Hugging Face Hub API

## search=SmolLM

- Query: GET https://huggingface.co/api/models?search=SmolLM&sort=createdAt&direction=-1&limit=15
- Fetch: 2026-10-03T10:53:08.556379Z 200

```
2026-10-03 EB-Sky/python-smollm2-v1 downloads= 0 likes= 0
2026-10-03 mrbhazied/tuned-SmolLM2-135M-Instruct-test-s downloads= 48 likes= 0
2026-10-02 mradermacher/reqlint-smollm3-3b-GGUF downloads= 248 likes= 0
2026-10-02 styal/SmolLM2-135M-outlier-reg-only-nref-500steps downloads= 11 likes= 0
2026-10-01 styal/SmolLM2-135M-outlier-reg-only-lr2e3-500steps downloads= 18 likes= 0
2026-10-01 jgalego/reqlint-smollm3-3b downloads= 146 likes= 0
2026-10-01 notpaulmartin/SmolLM3-3B-flax downloads= 195 likes= 0
2026-10-01 amd/SmolLM2_rai_1.8.0_medusa_1.0_npu downloads= 0 likes= 0
2026-10-01 amd/SmolLM2-135M-Instruct_rai_1.8.0_medusa_1.0_npu downloads= 0 likes= 0
2026-10-01 amd/SmolLM-Instruct_rai_1.8.0_medusa_1.0_npu downloads= 0 likes= 0
2026-10-01 SLM-Archive/SmolLM2-135M-Instruct downloads= 205 likes= 0
2026-10-01 SLM-Archive/SmolLM2-135M downloads= 211 likes= 0
2026-09-30 styal/SmolLM2-135M-outlier-reg-500steps downloads= 22 likes= 0
2026-09-30 styal/SmolLM2-135M-outlier-mask-500steps downloads= 12 likes= 0
2026-09-30 mradermacher/SmolLM3-MathInstruct-SFT-GGUF downloads= 552 likes= 1
```

## search=nanochat

- Query: GET https://huggingface.co/api/models?search=nanochat&sort=createdAt&direction=-1&limit=15
- Fetch: 2026-10-03T10:53:09.017733Z 200

```
2026-09-28 chicoballer/nanochat-d8-assignment1 downloads= 0 likes= 1
2026-09-28 chicoballer/nanochat_assignment downloads= 0 likes= 0
2026-09-28 Jessek43/nanochat-d2 downloads= 0 likes= 0
2026-09-28 tohoku-nlp/nanochat-jp-v3-d8 downloads= 0 likes= 0
2026-09-28 tohoku-nlp/nanochat-jp-v4-d22 downloads= 0 likes= 0
2026-09-28 tohoku-nlp/nanochat-jp-v3-d22 downloads= 0 likes= 0
2026-09-27 tohoku-nlp/nanochat-jp-v3-sft-hf downloads= 12 likes= 0
2026-09-27 tohoku-nlp/nanochat-jp-v4-sft-hf downloads= 21 likes= 0
2026-09-25 Batman55072/NanoChat-31M-V2 downloads= 252 likes= 0
2026-09-24 niuk77/nanochat-d12 downloads= 0 likes= 0
2026-09-24 burtenshaw/nanochat downloads= 314 likes= 0
2026-09-16 krakiun/ro-nanochat-d20-chat downloads= 0 likes= 0
2026-09-14 yhshin1020/nanochat_d18_pretrain downloads= 0 likes= 0
2026-09-08 Marcolini/nanochat-d24-base-r24-s615173 downloads= 507 likes= 0
2026-09-08 Marcolini/nanochat-d24-chat-champion downloads= 502 likes= 0
```

## search=modded-nanogpt

- Query: GET https://huggingface.co/api/models?search=modded-nanogpt&sort=createdAt&direction=-1&limit=15
- Fetch: 2026-10-03T10:53:09.460242Z 200

```
2026-06-28 ldmberman/modded-nanogpt-upscaling downloads= 0 likes= 0
2024-11-10 Fizzarolli/modded-nanogpt-logs downloads= 0 likes= 0
```

## Reading

- No new SmolLM-family base release and no recipe-bearing model card surfaced in the window. Hits are fine-tunes, quantisations, course assignments and community nanochat checkpoints (for example tohoku-nlp/nanochat-jp-v4-d22, 2026-09-28). Low signal for band A; no model card was opened.
