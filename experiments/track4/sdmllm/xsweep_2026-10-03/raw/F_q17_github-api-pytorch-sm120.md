# F_q17_github-api-pytorch-sm120
- Tool: python3 urllib, GitHub REST API /search/issues (unauthenticated)
- Fetched: 2026-10-03T10:52:26Z

## Query: `repo:pytorch/pytorch sm_120 is:issue created:>2026-05-01`
- total_count: 18
- [open] created 2026-09-20 updated 2026-10-01 comments 10 | activation_memory_budget: physical peak allocated memory is non-monotonic across budgets | https://github.com/pytorch/pytorch/issues/197838
- [closed] created 2026-09-23 updated 2026-09-28 comments 4 | Since 2.13, inductor's 4x random path corrupts CUDA dropout masks | https://github.com/pytorch/pytorch/issues/198333
- [open] created 2026-09-22 updated 2026-09-23 comments 3 | [scaled_mm][mxfp8] sm_120 (Blackwell GeForce): only TN operand layout works; NT/NN rejected by cuBLAS heuristic | https://github.com/pytorch/pytorch/issues/198126
- [open] created 2026-05-14 updated 2026-09-22 comments 3 | `F.conv2d` CPU performance regression (~4-16×) for bfloat16 inputs introduced in 2.13.0. | https://github.com/pytorch/pytorch/issues/183815
- [open] created 2026-05-14 updated 2026-09-22 comments 3 | `torch.scatter_reduce` CPU performance regression (~15-58×) introduced in 2.13.0. | https://github.com/pytorch/pytorch/issues/183812
- [open] created 2026-08-18 updated 2026-09-16 comments 10 | SDPA cuDNN backend silently corrupts greedy generation on B200 (sm_100) for models with num_key_value_heads < 8 | https://github.com/pytorch/pytorch/issues/193893
- [open] created 2026-07-29 updated 2026-08-28 comments 3 | [Inductor][Triton][CUDA][sm_120] BF16 autocast fusion of embeddings, Linear, and RMSNorm silently produces wrong results | https://github.com/pytorch/pytorch/issues/191433
- [open] created 2026-08-17 updated 2026-08-26 comments 0 | Multi-GPU bf16 training on RTX PRO 6000 Blackwell (sm_120): resume training random lottery | https://github.com/pytorch/pytorch/issues/193752
- [closed] created 2026-08-10 updated 2026-08-10 comments 2 | _scaled_mm / cuBLASLt: per-row FP8 scaling returns CUBLAS_STATUS_NOT_SUPPORTED on Blackwell (sm_120); per-tensor works | https://github.com/pytorch/pytorch/issues/192707
- [closed] created 2026-07-22 updated 2026-07-27 comments 2 | [inductor] Mix-order reduction silently corrupts conv-bias grads (bf16, sm_120) in 2.10.0 - fixed in 2.11 without regression test? | https://github.com/pytorch/pytorch/issues/190796
- [open] created 2026-05-04 updated 2026-07-11 comments 2 | [Bug] CUDA caching allocator unable to satisfy 3-18 GiB contiguous allocation despite 50+ GiB free on RTX PRO 6000 Blackwell + WSL2 (16 GiB hidden CUDA overhead) | https://github.com/pytorch/pytorch/issues/182286
- [closed] created 2026-06-04 updated 2026-07-06 comments 1 | torch._inductor "duplicate template name" / "duplicate extern kernel" AssertionError on 2.11.0+cu130 (Blackwell sm_120): templates double-registered in a single process | https://github.com/pytorch/pytorch/issues/186220
- [closed] created 2026-06-29 updated 2026-07-02 comments 2 | `int32` overflow in `embedding_bag(mode="max")` backward pass | https://github.com/pytorch/pytorch/issues/188467
- [open] created 2026-06-11 updated 2026-06-14 comments 3 | torch._int_mm dispatches Ampere-era cutlass_80 kernels on sm_103 (GB300): 6.4-13.4x SLOWER than the fp32 tf32 matmul of the same shape | https://github.com/pytorch/pytorch/issues/187094
- [closed] created 2026-06-05 updated 2026-06-09 comments 0 | AOTInductor on Windows: models with >2GB constants fail to load - 32-bit `lseek` overflow (`weights_offset must be aligned to 16K boundary`) | https://github.com/pytorch/pytorch/issues/186357

## Query: `repo:pytorch/pytorch 5090 is:issue created:>2026-05-01`
- total_count: 49
- [open] created 2026-10-02 updated 2026-10-02 comments 0 | Inductor-on-CUDA tests fail with TritonMissing instead of skipping when Triton is not installed | https://github.com/pytorch/pytorch/issues/199563
- [open] created 2026-09-24 updated 2026-10-01 comments 3 | torch.topk returns different values on CPU and CUDA for a 0-D input with k=0 | https://github.com/pytorch/pytorch/issues/198507
- [open] created 2026-09-18 updated 2026-09-30 comments 3 | Inductor split-scan workspace sizing can be overrun by an undersized R0_BLOCK, causing silent memory corruption | https://github.com/pytorch/pytorch/issues/197493
- [open] created 2026-08-15 updated 2026-09-25 comments 1 | torchaudio is missing from the cu132 wheel index (only stale ≤2.2.0 wheels are listed) | https://github.com/pytorch/pytorch/issues/193651
- [open] created 2026-09-09 updated 2026-09-24 comments 3 | Random SIGSEGV (exit code 139), CUDA errors, and occasional complete system freeze during YOLOv8 OBB training on RTX 5090 | https://github.com/pytorch/pytorch/issues/196393
- [closed] created 2026-09-04 updated 2026-09-23 comments 1 | `torch.histc` with `out=` applies different argument checks on CPU and CUDA. | https://github.com/pytorch/pytorch/issues/196008
- [closed] created 2026-08-17 updated 2026-09-23 comments 1 | torch.nn.LSTMCell segfaults when given malformed hidden state input | https://github.com/pytorch/pytorch/issues/193823
- [open] created 2026-06-29 updated 2026-09-22 comments 6 | torch.special.logit with eps > 0.5 has tiny vectorized-vs-scalar exact parity mismatch on non-contiguous float32 input | https://github.com/pytorch/pytorch/issues/188382
- [open] created 2026-05-14 updated 2026-09-22 comments 3 | `F.conv2d` CPU performance regression (~4-16×) for bfloat16 inputs introduced in 2.13.0. | https://github.com/pytorch/pytorch/issues/183815
- [open] created 2026-05-14 updated 2026-09-22 comments 3 | `torch.scatter_reduce` CPU performance regression (~15-58×) introduced in 2.13.0. | https://github.com/pytorch/pytorch/issues/183812
- [open] created 2026-09-18 updated 2026-09-18 comments 0 | Windows JIT CUDA extensions fail on CUDA 13.2+ (CCCL requires /Zc:preprocessor) | https://github.com/pytorch/pytorch/issues/197574
- [closed] created 2026-07-08 updated 2026-09-16 comments 1 | AttributeError: Can't get local object 'WeakValueDictionary.__init__.<locals>.remove' during AOTAutograd graph caching on Python 3.13 | https://github.com/pytorch/pytorch/issues/189293
- [open] created 2026-06-24 updated 2026-09-15 comments 2 | expandable_segments: partial mapAndSetAccess failure → bulk cuMemUnmap of unmapped range → terminate in destructor | https://github.com/pytorch/pytorch/issues/188008
- [closed] created 2026-09-02 updated 2026-09-14 comments 0 | torch.batch_norm_backward_reduce does not use FP32 precision for reduction under torch.amp low-precision operations as expected | https://github.com/pytorch/pytorch/issues/195701
- [open] created 2026-09-03 updated 2026-09-14 comments 7 | `torch.cumsum` drops the sign of IEEE `-0.0` on CPU (CUDA and NumPy preserve it) | https://github.com/pytorch/pytorch/issues/195851

## Query: `repo:pytorch/pytorch B200 is:issue created:>2026-05-01`
- total_count: 66
- [open] created 2026-08-19 updated 2026-10-03 comments 3 | Any `sdpa_kernel` block inside a compiled region bypasses AOTAutogradCache | https://github.com/pytorch/pytorch/issues/194007
- [open] created 2026-10-02 updated 2026-10-02 comments 0 | [CI] B200 smoke tests CUDA host-memory stats tests fail when earlier tests leave active pinned allocations | https://github.com/pytorch/pytorch/issues/199580
- [open] created 2026-10-02 updated 2026-10-02 comments 4 | [vllm] [2.14 regression][B200][AsyncTP] NVFP4 + async-TP (fuse_gemm_comms + SP) under torch.compile: 2nd sequence in batch decodes to "!!!!!" | https://github.com/pytorch/pytorch/issues/199505
- [open] created 2026-08-04 updated 2026-10-02 comments 0 | [torch 2.14] umbrella issue - vLLM CI failures | https://github.com/pytorch/pytorch/issues/192057
- [open] created 2026-09-28 updated 2026-09-28 comments 0 | F.mse_loss with reduction='mean'/'sum' returns a 0-d tensor that holds the full elementwise buffer's storage | https://github.com/pytorch/pytorch/issues/198951
- [closed] created 2026-08-31 updated 2026-09-25 comments 5 | [Inductor] functorch_dp_cifar10 training accuracy has failed 100% on H100 since 2026-06-13 (all configs) | https://github.com/pytorch/pytorch/issues/195483
- [closed] created 2026-09-01 updated 2026-09-23 comments 0 | Context Parallel block mask incompatible with FlexAttentions's FlashAttention4 backend on Blackwell | https://github.com/pytorch/pytorch/issues/195563
- [closed] created 2026-09-22 updated 2026-09-22 comments 5 | linux.dgx.b200 runner queuing exceeds 4 hours and dgxb200-05 "CUDA not available" | https://github.com/pytorch/pytorch/issues/198076
- [open] created 2026-09-22 updated 2026-09-22 comments 1 | AOTInductor drops the storage offset of a mutable view argument under dynamic shapes (missing lowering for aten.sym_storage_offset) | https://github.com/pytorch/pytorch/issues/198068
- [open] created 2026-08-13 updated 2026-09-22 comments 1 | [Inductor][AOTI][B200] TorchBench compilation latency regressed 57% since Aug 11 | https://github.com/pytorch/pytorch/issues/193460
- [open] created 2026-06-29 updated 2026-09-22 comments 7 | [CI][B200] smoke_b200 test_nv_universal_gemm fails: cutlass.cute.arch has no attribute 'ProxyKind' (unpinned cutlass_api vs pinned cutlass-dsl) | https://github.com/pytorch/pytorch/issues/188477
- [open] created 2026-07-26 updated 2026-09-22 comments 0 | [Inductor] NVGemm broken: upstream renamed cutlass.operators -> cutlass_api | https://github.com/pytorch/pytorch/issues/191130
- [open] created 2026-08-18 updated 2026-09-16 comments 10 | SDPA cuDNN backend silently corrupts greedy generation on B200 (sm_100) for models with num_key_value_heads < 8 | https://github.com/pytorch/pytorch/issues/193893
- [closed] created 2026-09-15 updated 2026-09-15 comments 2 | B200 dgxb200-03 in bad state | https://github.com/pytorch/pytorch/issues/197073
- [closed] created 2026-06-29 updated 2026-09-14 comments 8 | RTX 5090 cudaErrorLaunchTimeout | https://github.com/pytorch/pytorch/issues/188411


## Follow-up primary fetch attempt
- 2026-10-03T10:52:49Z python3 urllib GET https://api.github.com/repos/pytorch/pytorch/issues/191433 -> ERROR: urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
- rate_limit: core 0 of 60 remaining (shared IP with other bands), reset in 1578 s; search 10 of 10.
- Fallback: WebFetch on the github.com HTML issue page.

Note: em and en dashes in quoted titles and text were normalised to '-' for the house style; no other character was changed.
