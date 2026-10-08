# Band F - search matrix (written before any query ran)

Started: 2026-10-03 10:47:21 UTC. Window 2026-08-01 to 2026-10-03, widened to 2026-05 where empty.
Tools: WebSearch (US engine), WebFetch, python3 urllib against the GitHub REST API (unauthenticated) and arXiv.
X API unavailable (HTTP 402 credits depleted). x.com and reddit hits are cited "via search, not fetched".

| # | slug | question |
|---|---|---|
| F_q01 | rtx5090-pytorch-sm120-wheels | Which torch / CUDA wheels support sm_120 (5090), and which do not |
| F_q02 | sm120-triton-compile-issues | torch.compile / Triton failures on sm_120 (GitHub issues) |
| F_q03 | rtx5090-training-throughput | Reported training tokens/sec on one RTX 5090 |
| F_q04 | rtx-pro-6000-training | RTX PRO 6000 Blackwell training benchmarks and problems |
| F_q05 | pro6000-p2p-nccl | RTX PRO 6000 P2P / NCCL over PCIe / multi-GPU scaling |
| F_q06 | rtx5090-p2p-blocked | Is P2P disabled on RTX 5090, DDP impact |
| F_q07 | h200-vs-h100-vs-b200-small-model | Small-model training comparisons H100 / H200 / B200 |
| F_q08 | b200-driver-cuda-nccl | B200 driver / CUDA 12.8+ / NCCL version problems |
| F_q09 | vastai-known-issues-2026 | Vast.ai known issues 2026 (driver mismatch, images, disk, host reliability) |
| F_q10 | vastai-no-kernel-image-blackwell | "no kernel image is available" on Vast Blackwell hosts |
| F_q11 | pytorch-docker-tags-blackwell | pytorch docker image tags that work on Blackwell |
| F_q12 | scatter-add-index-add-blackwell | scatter_add / index_add / embedding backward performance, atomics |
| F_q13 | embedding-backward-deterministic | embedding_dense_backward / EmbeddingBag backward perf issues 2026 |
| F_q14 | fp8-training-rtx5090-torchao | FP8 / MXFP8 training on 5090 via torchao, sm_120 support |
| F_q15 | mxfp8-b200-torchao-small | MXFP8 on B200 for small models |
| F_q16 | modded-nanogpt-5090 | modded-nanogpt / nanoGPT speedrun on single 5090 |
| F_q17 | nanochat-single-gpu | nanochat on single 5090 / PRO 6000 / H200 / B200 |
| F_q18 | github-api-pytorch-sm120 | GitHub REST search: pytorch/pytorch issues sm_120 2026 |
| F_q19 | github-api-nccl-p2p | GitHub REST search: NVIDIA/nccl issues P2P 5090 / PRO 6000 |
| F_q20 | github-api-vastai | GitHub REST search: vast-ai repos, issues 2026 |
| F_q21 | cuda13-torch-2-9 | torch 2.9 / 2.10 CUDA 13.0 wheels, driver floor |
| F_q22 | large-vocab-ce-chunked | chunked / fused cross-entropy for large vocab heads on Blackwell (Liger, cut-cross-entropy) |

Each query gets `raw/F_qNN_<slug>.md` with the exact query, tool, UTC time and the verbatim result list, or EMPTY, or the exact error.
