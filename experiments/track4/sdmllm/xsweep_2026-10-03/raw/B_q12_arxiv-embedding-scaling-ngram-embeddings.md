# q12 arxiv-embedding-scaling-ngram-embeddings

- query: `abs:"embedding layer" AND abs:scaling AND abs:"n-gram"`
- tool: python3 urllib, arXiv Atom API
- url: http://export.arxiv.org/api/query?search_query=abs%3A%22embedding+layer%22+AND+abs%3Ascaling+AND+abs%3A%22n-gram%22&sortBy=submittedDate&sortOrder=descending&max_results=40
- fetched_utc: 2026-10-03T10:47:34Z
- filter: published >= 2026-06-01 (results older than that are listed but marked OLD)
- entries returned: 1; in window: 0

## [OLD] Scaling Embedding Layers in Language Models
- url: http://arxiv.org/abs/2502.01637v3
- published: 2025-02-03
- abstract (verbatim): We propose $SCONE$ ($S$calable, $C$ontextualized, $O$ffloaded, $N$-gram $E$mbedding), a new method for extending input embedding layers to enhance language model performance. To avoid increased decoding costs, $SCONE$ retains the original vocabulary while introducing embeddings for a set of frequent n-grams. These embeddings provide contextualized representation for each input token and are learned with a separate model during training. After training, embeddings are precomputed and stored in off-accelerator memory; during inference, querying them has minimal impact on latency due to the low complexity of embedding lookups. $SCONE$ enables two new scaling strategies: increasing the number of n-gram embeddings and scaling the model used to learn them, both while maintaining fixed accelerator usage during inference (in terms of FLOPS and memory). We show that scaling both aspects enables a model with 1B accelerator-resident parameters to outperform a 1.9B-parameter baseline across diverse corpora, while using only about half the FLOPS and accelerator memory during inference.

