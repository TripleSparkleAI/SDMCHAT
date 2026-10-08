# q15 arxiv-ultramem

- query: `abs:UltraMem OR ti:UltraMem`
- tool: python3 urllib, arXiv Atom API
- url: http://export.arxiv.org/api/query?search_query=abs%3AUltraMem+OR+ti%3AUltraMem&sortBy=submittedDate&sortOrder=descending&max_results=40
- fetched_utc: 2026-10-03T10:48:06Z
- filter: published >= 2026-06-01 (results older than that are listed but marked OLD)
- entries returned: 2; in window: 0

## [OLD] UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning
- url: http://arxiv.org/abs/2508.18756v1
- published: 2025-08-26
- abstract (verbatim): While Mixture of Experts (MoE) models achieve remarkable efficiency by activating only subsets of parameters, they suffer from high memory access costs during inference. Memory-layer architectures offer an appealing alternative with very few memory access, but previous attempts like UltraMem have only matched the performance of 2-expert MoE models, falling significantly short of state-of-the-art 8-expert configurations. We present UltraMemV2, a redesigned memory-layer architecture that closes this performance gap. Our approach introduces five key improvements: integrating memory layers into every transformer block, simplifying value expansion with single linear projections, adopting FFN-based value processing from PEER, implementing principled parameter initialization, and rebalancing memory-to-FFN computation ratios. Through extensive evaluation, we demonstrate that UltraMemV2 achieves performance parity with 8-expert MoE models under same computation and parameters but significantly low memory access. Notably, UltraMemV2 shows superior performance on memory-intensive tasks, with improvements of +1.6 points on long-context memorization, +6.2 points on multi-round memorization, and +7.9 points on in-context learning. We validate our approach at scale with models up to 2.5B activated parameters from 120B total parameters, and establish that activation density has greater impact on performance than total sparse parameter count. Our work brings memory-layer architectures to performance parity with state-of-the-art MoE models, presenting a compelling alternative for efficient sparse computation.

## [OLD] Ultra-Sparse Memory Network
- url: http://arxiv.org/abs/2411.12364v2
- published: 2024-11-19
- abstract (verbatim): It is widely acknowledged that the performance of Transformer models is logarithmically related to their number of parameters and computational complexity. While approaches like Mixture of Experts (MoE) decouple parameter count from computational complexity, they still face challenges in inference due to high memory access costs. This work introduces UltraMem, incorporating large-scale, ultra-sparse memory layer to address these limitations. Our approach significantly reduces inference latency while maintaining model performance. We also investigate the scaling laws of this new architecture, demonstrating that it not only exhibits favorable scaling properties but outperforms MoE. In experiments, the largest UltraMem we train has 20 million memory slots. The results show that our method achieves state-of-the-art inference speed and model performance within a given computational budget, paving the way for billions of slots or experts.

