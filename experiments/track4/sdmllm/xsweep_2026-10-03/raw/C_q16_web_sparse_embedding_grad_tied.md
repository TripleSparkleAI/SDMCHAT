# C_q16 sparse embedding gradients with a tied head in PyTorch (web)

- query: "PyTorch sparse embedding gradient tied lm_head weight sparse=True Adam row-wise update large vocabulary"
- tool: WebSearch
- fetched UTC: 2026-10-03 ~10:52
- note: snippets are the search tool's summary. The tool itself states its section on tying is "my own reasoning, not something the sources confirm".

## Result list (three batches)
1. Sparse Embedding failing with Adam (PyTorch Forums) - https://discuss.pytorch.org/t/sparse-embedding-failing-with-adam-torch-cuda-sparse-floattensor-has-no-attribute-addcmul/5589
2. torch/optim/sparse_adam.py - https://github.com/pytorch/pytorch/blob/main/torch/optim/sparse_adam.py
3. Training Larger and Faster Recommender Systems with PyTorch Sparse Embeddings (NVIDIA Merlin) - https://medium.com/nvidia-merlin/...
4. Optimize a transformed Embedding with sparse gradients (forums) - https://discuss.pytorch.org/t/optimize-a-transformed-embedding-with-sparse-gradients/139982
5. SparseAdam fails when a parameter has an empty sparse gradient tensor, issue #82486 - https://github.com/pytorch/pytorch/issues/82486
6. bug with nn.Embedding sparse gradients, issue #17919 - https://github.com/pytorch/pytorch/issues/17919
7. Embedding - PyTorch main documentation - https://docs.pytorch.org/docs/main/generated/torch.nn.modules.sparse.Embedding.html
8. Hitchhiker's Guide to Agentic AI - https://arxiv.org/pdf/2606.24937
9. Phantom Transitions in LM Fine-Tuning - https://arxiv.org/pdf/2606.07559
10. What's the behavior of optimizer.step() when embedding layer is sparse? (forums) - https://discuss.pytorch.org/t/resolved-whats-the-behavior-of-optimizer-step-when-embedding-layer-is-sparse/4347
11. torch.nn.modules.sparse (0.3.1 docs) - https://docs.pytorch.org/docs/0.3.1/_modules/torch/nn/modules/sparse.html
12. torchrec issue #2394 - https://github.com/meta-pytorch/torchrec/issues/2394
13. Two-dimensional Sparse Parallelism for Large Scale DLRM Training - https://arxiv.org/pdf/2508.03854
14. HEAT - https://arxiv.org/pdf/2304.07334
15. issue #31651 - https://github.com/pytorch/pytorch/issues/31651
16. issue #4666 - https://github.com/pytorch/pytorch/issues/4666
17. Sparse All-Reduce in PyTorch (Speechmatics) - https://blog.speechmatics.com/Sparse-All-Reduce-Part-1
18. Bug of nn.Embedding when sparse=True and padding_idx is set - https://discuss.pytorch.org/t/...
19. Embedding returns an error (forums) - https://discuss.pytorch.org/t/embedding-returns-an-error-that-i-dont-understand/1072
20. When should I use sparse embedding (forums) - https://discuss.pytorch.org/t/when-should-i-use-sparse-embedding-instead-of-dense-embedding/29084
21. Embedding weights tied to projection out logits (flax discussion 2186) - https://github.com/google/flax/discussions/2186
22. Z-Loss Backward Geometry in Dense Output Heads and Sparse Routers - https://arxiv.org/pdf/2609.16179
23. Weight Tying Biases Token Embeddings Towards the Output Space - https://arxiv.org/pdf/2603.26663
24. Padding in PyTorch and TensorFlow embedding layers - https://minibatchai.com/2021/06/22/Embedding.html

## Search-tool summary points
- SparseAdam "only updates the entries with non-zero gradients"; supported sparse-grad optimizers: SGD, SparseAdam, Adagrad (CPU).
- With a full softmax on a tied table, the output side gives a gradient to every row, so the tied gradient is dense (tool's reasoning, unverified).
- Row-wise AdaGrad in DLRM keeps one state value per row.
