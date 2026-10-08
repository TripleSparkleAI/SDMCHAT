# F_q13 fp8-training-rtx5090-torchao (matrix row F_q14)
- Query: `torchao FP8 training RTX 5090 sm_120 float8 rowwise support consumer Blackwell`
- Tool: WebSearch. Fetched ~2026-10-03 10:55 UTC.
- Hits: 18.

Result list (title | url):
1. Fall back from FBGEMM for rowwise FP8 on GPUs it has no kernel for (RTX PRO 6000 / 5090) - PR #12098 unslothai/unsloth | https://github.com/unslothai/unsloth/pull/12098
2. fix(fp8): both FP8 training paths ask one gate - PR #1044 MakazhanAlpamys/Soup | https://github.com/MakazhanAlpamys/Soup/pull/1044
3. Quantized Training - torchao 0.17 docs | https://docs.pytorch.org/ao/stable/workflows/training.html
4. pytorch/ao | https://github.com/pytorch/ao
5. TorchAO paper arXiv 2507.16099 | https://arxiv.org/pdf/2507.16099
6. [feature request][mxfp8] Orientation-flexible MXFP8 GEMM for sm_120: cuBLAS is TN-only on Blackwell GeForce - Issue #4932 pytorch/ao | https://github.com/pytorch/ao/issues/4932
7. Quantized Inference - torchao 0.17 docs | https://docs.pytorch.org/ao/stable/workflows/inference.html
8. pytorch/ao PR #3160 diff | https://github.com/pytorch/ao/pull/3160.diff
9. pytorch/ao PR #3207 files | https://github.com/pytorch/ao/pull/3207/files
10. pytorch #167244 | https://github.com/pytorch/pytorch/issues/167244
11. pytorch #150733 | https://github.com/pytorch/pytorch/issues/150733
12. [scaled_mm][mxfp8] sm_120: only TN operand layout works; NT/NN rejected by cuBLAS heuristic - Issue #198126 pytorch | https://github.com/pytorch/pytorch/issues/198126
13. Fooocus #3862 | https://github.com/lllyasviel/Fooocus/issues/3862
14. Support for NVIDIA RTX 5090 (CUDA SM_120) - PyTorch Forums | https://discuss.pytorch.org/t/support-for-nvidia-rtx-5090-cuda-sm-120/223683
15. Is there a PyTorch build that supports RTX 5090 - PyTorch Forums | https://discuss.pytorch.org/t/is-there-a-pytorch-build-that-supports-nvidia-rtx-5090-compute-capability-12-0-sm-120/223536
16. NVIDIA forums Rtx 5090 | https://forums.developer.nvidia.com/t/rtx-5090/331369
17. RTX 5080 CUDA compatibility - NVIDIA forums | https://forums.developer.nvidia.com/t/im-facing-a-cuda-compatibility-issue-with-my-new-rtx-5080-in-deep-learning/347350
18. HF simple-chat commit | https://huggingface.co/spaces/alex4cip/simple-chat/commit/2c96300933744eb331049e7d61c5c1166470a436

Engine summary (NOT a source; open primaries): unsloth #12098 "FBGEMM ships no sm120 kernels for f8f8bf16_rowwise"; ao #4932 / pytorch #198126 "~11 days old": cuBLAS block-scaled FP8 on cc 12.x is TN-only, "TN MXFP8 runs at 3.2x bf16 on a 5060 Ti".
