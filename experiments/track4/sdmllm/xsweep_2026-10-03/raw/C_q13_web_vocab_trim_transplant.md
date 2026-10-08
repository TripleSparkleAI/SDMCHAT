# C_q13 vocabulary trimming / pruning / tokenizer transplant for small models (web)

- query: "vocabulary trimming pruning small language model tokenizer transplant 2026"
- tool: WebSearch
- fetched UTC: 2026-10-03 ~10:52
- note: snippets are the search tool's summary, not verbatim page text. Not primary-checked except where stated in band_C_head.md.

## Result list (two batches)
1. Efficient Tokenizer Adaptation for Pre-trained Models (EACL 2026 Findings) - https://aclanthology.org/2026.findings-eacl.341.pdf
2. VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models - https://arxiv.org/pdf/2508.15229
3. Efficient Vocabulary Reduction for Small Language Models (COLING 2025 Industry) - https://aclanthology.org/2025.coling-industry.64.pdf
4. Ambient @ EgoLongQA 2026: Distilling Long-Video perception into a Sub-2B Model - https://arxiv.org/pdf/2609.07154
5. VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding in LLMs - https://arxiv.org/pdf/2506.22694
6. PruneSLU (Interspeech 2025) - https://www.isca-archive.org/interspeech_2025/do25_interspeech.pdf
7. On Multilingual Encoder Language Model Compression for Low-Resource Languages - https://arxiv.org/pdf/2505.16956
8. Efficient Tokenizer Adaptation for Pre-trained Models - https://www.arxiv.org/pdf/2512.03989
9. Teaching Old Tokenizers New Words (ACL Anthology page) - https://aclanthology.org/2026.findings-eacl.341/
10. Deep Learning and Machine Learning -- NLP - https://arxiv.org/pdf/2411.05026
11. Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit - https://arxiv.org/pdf/2506.06607
12. Tucano 2 Cool - https://arxiv.org/pdf/2603.03543
13. Tokenizer-Aware Cross-Lingual Adaptation of Decoder- ... - https://aclanthology.org/2026.eacl-long.357.pdf
14. Model-Aware Tokenizer Transfer - https://arxiv.org/abs/2510.21954
15. MoirfEolas and CriochScore - https://arxiv.org/pdf/2609.05022
16. Achieving Tokenizer Flexibility in Language Models through Heuristic ... - https://arxiv.org/pdf/2505.09738
17. Bolmo: Byteifying the Next Generation of Language Models - https://arxiv.org/pdf/2512.15586
18. Tokenizer-Free Architectures (emergentmind) - https://www.emergentmind.com/topics/tokenizer-free-architectures

## Search-tool summary points
- EgoLongQA 2026 team: "keep token ids [0,143000) unchanged, append a small set of explicitly retained rare tokens, and relocate the 33 added and special tokens" (vocabulary 248,320 rows).
- multilingual encoders: "Restricting the vocabulary to the top 40K most frequent tokens ... introduces no measurable performance loss".
- These are all post-hoc adaptations of a pretrained model; none is a from-scratch training trick.
