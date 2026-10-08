# F_q12 scatter-add-index-add-blackwell
- Query: `index_add scatter_add embedding backward slow Blackwell atomics PyTorch issue sm_100 sm_120`
- Tool: WebSearch. Fetched ~2026-10-03 10:52 UTC.
- Hits: 28.

Result list (title | url):
1. scatter_add index size mismatch - Issue #27614 | https://github.com/pytorch/pytorch/issues/27614
2. Scatter add much slower when compiled - PyTorch Forums | https://discuss.pytorch.org/t/scatter-add-much-slower-when-compiled/213844
3. scatter_add is slow with fp16 inputs when the index tensor is concentrated - Issue #74487 | https://github.com/pytorch/pytorch/issues/74487
4. [MPS] Migrate index_add and index_select to Metal kernels - PR #187109 | https://github.com/pytorch/pytorch/pull/187109
5. torch-mojo-backend PR #404 | https://github.com/gabrieldemarmiesse/torch-mojo-backend/pull/404
6. nn.Embedding backwards slow under high row contention - Issue #20655 | https://github.com/pytorch/pytorch/issues/20655
7. IPEX #426 scatter_add 5x slower | https://github.com/intel/intel-extension-for-pytorch/issues/426
8. MIDUS: Memory-Infused Depth Up-Scaling (arXiv 2512.13751) | https://arxiv.org/pdf/2512.13751
9. NSL PR #769 embedding backward kernels (atomic + deterministic) | https://github.com/bwiemz/NSL/pull/769
10. Use atomicAdd for bfloat16 in Ampere and above - PR #84981 | https://github.com/pytorch/pytorch/pull/84981
11. [Inductor] atomic_add does not support bf16 - Issue #97016 | https://github.com/pytorch/pytorch/issues/97016
12. Vector-Matrix Multiplication is slower in Blackwell (B200) than Hopper (H200) - Issue #161134 | https://github.com/pytorch/pytorch/issues/161134
13. [Inductor][ROCm] Fall back for BF16 atomic accumulation - PR #198199 | https://github.com/pytorch/pytorch/pull/198199
14. tl.atomic_add for bfloat16 - triton Issue #1387 | https://github.com/triton-lang/triton/issues/1387
15. V100 inductor bf16 sm_80 - Issue #103993 | https://github.com/pytorch/pytorch/issues/103993
16. Recommended FP8 recipe for Blackwell (B200)? - NVIDIA/Model-Optimizer #2015 | https://github.com/NVIDIA/Model-Optimizer/issues/2015
17. FP16 and BF16 way slower than FP32 - PyTorch Forums | https://discuss.pytorch.org/t/fp16-and-bf16-way-slower-than-fp32-and-tf32/162778
18. github.com pytorch PR #167380 | https://github.com/pytorch/pytorch/pull/167380
19. backward on 5090(sm_120) much slower than on 4090 - Issue #160838 | https://github.com/pytorch/pytorch/issues/160838
20. (dup #20655)
21. Slow Embedding backward - PyTorch Forums | https://discuss.pytorch.org/t/slow-embedding-backward/81341
22. Segfault in embedding_dense_backward - Issue #142837 | https://github.com/pytorch/pytorch/issues/142837
23. embedding_dense_backward OOB CPU - Issue #145267 | https://github.com/pytorch/pytorch/issues/145267
24. Dense embedding double backward - Issue #6469 | https://github.com/pytorch/pytorch/issues/6469
25. Sorting in embedding_dense_backward_cuda takes very long time - Issue #30711 | https://github.com/pytorch/pytorch/issues/30711
26. PR #9078 | https://github.com/pytorch/pytorch/pull/9078
27. [Intel GPU] Fallback embedding_dense_backward on XPU - PR #146888 | https://github.com/pytorch/pytorch/pull/146888
28. ComfyUI #14157 RTX 5090 nvfp4 bypasses allocator | https://github.com/Comfy-Org/ComfyUI/issues/14157

Engine summary (NOT a source; open primaries): #160838 "4090 takes 0.20s forward and 0.21s backward ... 5090 takes 0.21s forward and 5.34s backward"; #74487 fp16 scatter_add slow with concentrated index; #161134 B200 slower than H200 on vector-matrix.
