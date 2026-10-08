# C_q20_arxiv_coupled_adam_embedding_lr Coupled Adam / embedding learning rate vs vocabulary size / OEC (follow-up on rare-token lead)

- query: ti:"Coupled Adam" OR ti:"Optimal Embedding Learning Rate" OR ti:"Output Embedding Centering" OR ti:"Better Embeddings"
- url: http://export.arxiv.org/api/query?search_query=ti%3A%22Coupled+Adam%22+OR+ti%3A%22Optimal+Embedding+Learning+Rate%22+OR+ti%3A%22Output+Embedding+Centering%22+OR+ti%3A%22Better+Embeddings%22&sortBy=submittedDate&sortOrder=descending&max_results=20
- tool: python3 urllib, arXiv Atom API, sortBy=submittedDate desc
- fetched UTC: 2026-10-03T10:54:25.613687Z
- filter: entries published >= 2024-06-01 listed; older ones counted
- returned 5 entries; 3 in window; 2 older

## Output Embedding Centering for Stable LLM Pretraining
- url: http://arxiv.org/abs/2601.02031v3
- published: 2026-01-05
- abstract (verbatim): Pretraining of large language models is not only expensive but also prone to certain training instabilities. A specific instability that often occurs at the end of training is output logit divergence. The most widely used mitigation strategies, z-loss and logit soft-capping, merely address the symptoms rather than the underlying cause of the problem. In this paper, we analyze the instability from the perspective of the output embeddings' geometry and identify anisotropic embeddings as its source. Based on this, we propose output embedding centering (OEC) as a new mitigation strategy, and demonstrate that it suppresses output logit divergence. OEC can be implemented in two different ways: as a deterministic operation called $μ$-centering, or a regularization method called $μ$-loss. Our experiments show that both variants outperform z-loss in terms of training stability, while being on par with logit soft-capping. This holds true both in the presence and the absence of weight tying. As a secondary result, we find that $μ$-loss is significantly less sensitive to regularization hyperparameter tuning than z-loss.

## Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size
- url: http://arxiv.org/abs/2506.15025v1
- published: 2025-06-17
- abstract (verbatim): Pretraining large language models is a costly process. To make this process more efficient, several methods have been proposed to optimize model architecture/parametrization and hardware use. On the parametrization side, $μP$ (Maximal Update Parametrization) parametrizes model weights and learning rate (LR) in a way that makes hyperparameters (HPs) transferable with width (embedding dimension): HPs can be tuned for a small model and used for larger models without additional tuning. While $μ$P showed impressive results in practice, recent empirical studies have reported conflicting observations when applied to LLMs. One limitation of the theory behind $μ$P is the fact that input dimension (vocabulary size in LLMs) is considered fixed when taking the width to infinity. This is unrealistic since vocabulary size is generally much larger than width in practice. In this work, we provide a theoretical analysis of the effect of vocabulary size on training dynamics, and subsequently show that as vocabulary size increases, the training dynamics \emph{interpolate between the $μ$P regime and another regime that we call Large Vocab (LV) Regime}, where optimal scaling rules are different from those predicted by $μ$P. Our analysis reveals that in the LV regime, the optimal embedding LR to hidden LR ratio should roughly scale as $Θ(\sqrt{width})$, surprisingly close to the empirical findings previously reported in the literature, and different from the $Θ(width)$ ratio predicted by $μ$P. We conduct several experiments to validate our theory, and pretrain a 1B model from scratch to show the benefit of our suggested scaling rule for the embedding LR.

## Better Embeddings with Coupled Adam
- url: http://arxiv.org/abs/2502.08441v3
- published: 2025-02-12
- abstract (verbatim): Despite their remarkable capabilities, LLMs learn word representations that exhibit the undesirable yet poorly understood feature of anisotropy. In this paper, we argue that the second moment in Adam is a cause of anisotropic embeddings, and suggest a modified optimizer called Coupled Adam to mitigate the problem. Our experiments demonstrate that Coupled Adam significantly improves the quality of embeddings, while also leading to better upstream and downstream performance on large enough datasets.
