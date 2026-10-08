# C_q08 SmolLM3 tied embeddings and the stated reason (web)

- query: "SmolLM3 tied embeddings blog \"tie\" embeddings small model reason"
- tool: WebSearch
- fetched UTC: 2026-10-03 ~10:50
- note: snippets are the search tool's summary; primary checked in C_q08b (playbook page).

## Result list (two batches)
1. Papers Explained 176: Smol LM - https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc
2. Weight tying in language models: when and why LLMs share embeddings - https://medium.com/@vishal09vns/...
3. SmolLM-135M discussion 15 - https://huggingface.co/HuggingFaceTB/SmolLM-135M/discussions/15
4. SmolLM3: The Complete Blueprint - https://learnopencv.com/smollm3-explained/
5. SmolLM3 by Hugging Face (medium) - https://medium.com/data-science-in-your-pocket/...
6. SmolLM3 transformers docs - https://huggingface.co/docs/transformers/en/model_doc/smollm3
7. notpaulmartin/SmolLM3-3B-flax - https://huggingface.co/notpaulmartin/SmolLM3-3B-flax
8. Weight Tying Biases Token Embeddings Towards the Output Space - https://arxiv.org/pdf/2603.26663
9. VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models - https://arxiv.org/pdf/2508.15229
10. SmolLM3 3B specs - https://apxml.com/models/smollm3-3b
11. smol-training-playbook crawl gist - https://gist.github.com/unclecode/e5da5fb6a1d37022b089e243e0d9e00e
12. The Smol Training Playbook - https://huggingfacetb-smol-training-playbook.hf.space/
13. smol training playbook gist - https://gist.github.com/jph00/3c97a2c6c5075c4e7b98faae634b033a
14. PebbleGPT - https://github.com/sanjayriram44/PebbleGPT
15. Smol Training Playbook Reading Note - https://jianyuh.github.io/training/2025/11/29/Smol-Train.html
16. Kingy AI review - https://kingy.ai/...
17. SmolLM2 paper - https://arxiv.org/pdf/2502.02737
18. theneuron explainer - https://www.theneuron.ai/...
19. emergentmind SmolLM2 family - https://www.emergentmind.com/topics/smollm2-family

## Search-tool summary
- playbook: tied baseline "beat a bigger untied model on every benchmark except WinoGrande, despite having 18% fewer parameters"; "increasing model depth provides greater benefits than untying embeddings at equivalent parameter budgets"; ablation used "a 1B Llama3.2-style baseline trained on 45B tokens".
- Qwen3 small models (0.6B, 1.7B, 4B) tied, 8B and above untied (claim from summary; config check in C_q10 confirms 0.6B and 1.7B tied).
