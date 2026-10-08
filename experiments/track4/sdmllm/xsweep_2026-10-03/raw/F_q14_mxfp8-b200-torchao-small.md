# F_q14 mxfp8-b200-torchao-small (matrix row F_q15)
- Query: `torchao MXFP8 training B200 small model speedup 2026`
- Tool: WebSearch. Fetched ~2026-10-03 10:55 UTC.
- Hits: 9.

Result list (title | url):
1. pytorch/ao | https://github.com/pytorch/ao
2. Quantized Training - torchao 0.17 docs | https://docs.pytorch.org/ao/stable/workflows/training.html
3. Enabling Up to 41% Faster Pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200 with TorchTitan - pytorch.org blog | https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/
4. Accelerating 2K scale pre-training up to 1.28x with TorchAO, MXFP8 on Crusoe B200 - pytorch.org blog | https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/
5. MXFP8 Training for MoEs 1.3x on GB200 - pytorch.org blog | https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/
6. MXFP8 Expert Parallel Training - torchao 0.17 | https://docs.pytorch.org/ao/0.17/eager_tutorials/mxfp8_expert_parallel_training.html
7. Releases - pytorch/ao | https://github.com/pytorch/ao/releases
8. Quantized Inference - torchao 0.17 | https://docs.pytorch.org/ao/stable/workflows/inference.html
9. Faster Diffusion on Blackwell: MXFP8 and NVFP4 with Diffusers and TorchAO | https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/

Engine summary (NOT a source; verify on torchao training docs): 8x B200 Llama3-8B mxfp8_cublas 9,969 vs bf16 8,307.5 tok/s (+20.0%), float8 tensorwise +25.4%; inference roofline 0.93x at 1024^3, 1.20 at 2048.
