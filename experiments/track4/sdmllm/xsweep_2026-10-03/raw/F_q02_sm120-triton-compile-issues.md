# F_q02 sm120-triton-compile-issues
- Query: `torch.compile Triton sm_120 RTX 5090 error github issue 2026`
- Tool: WebSearch. Fetched ~2026-10-03 10:48 UTC.
- Hits: 18.

Result list (title | url):
1. torch.compile on a simulated GPU ... PR #191 pantheongpu/pantheonsim | https://github.com/pantheongpu/pantheonsim/pull/191
2. Add CUDA sm_120 support for RTX 5090 in official wheels - Issue #167244 pytorch | https://github.com/pytorch/pytorch/issues/167244
3. Fooocus Issue #3862 sm_120 not compatible | https://github.com/lllyasviel/Fooocus/issues/3862
4. pytorch Issue #150733 | https://github.com/pytorch/pytorch/issues/150733
5. Add official support for CUDA sm_120 - Issue #159207 | https://github.com/pytorch/pytorch/issues/159207
6. ebook2audiobook Issue #761 | https://github.com/DrewThomasson/ebook2audiobook/issues/761
7. PyTorch Forums 216518 | https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090-with-cuda-capability-sm-120-is-not-compatible-with-the-current-pytorch-installation/216518
8. ragflow Issue #10418 GPU build for SM-120 uses old PyTorch 2.6 | https://github.com/infiniflow/ragflow/issues/10418
9. NVIDIA Developer Forums Rtx 5090 | https://forums.developer.nvidia.com/t/rtx-5090/331369
10. RTX 5070 Ti + PyTorch Nightly + Triton: "sm_120 is not defined for option 'gpu-name'" - PyTorch Forums | https://discuss.pytorch.org/t/rtx-5070-ti-blackwell-pytorch-nightly-triton-still-getting-sm-120-is-not-defined-for-option-gpu-name-error/220460
11. Docs: bring-up recipe for Triton on NVIDIA Blackwell sm_121 (GB10 / DGX Spark) - Issue #10331 triton-lang/triton | https://github.com/triton-lang/triton/issues/10331
12. RLinf Issue #1638 BEHAVIOR install does not produce an sm_120 torch | https://github.com/RLinf/RLinf/issues/1638
13. Pytorch support for sm120 - Page 3 - PyTorch Forums | https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099?page=3
14. pytorch Issue #164342 | https://github.com/pytorch/pytorch/issues/164342
15. ragflow #10418 (dup)
16. PTX JIT broken on RTX 5080 - missing libnvptxcompiler.so in CUDA 12.8 / 12.9 - HF forum | https://discuss.huggingface.co/t/ptx-jit-broken-on-rtx-5080-blackwell-sm-120-missing-libnvptxcompiler-so-in-cuda-12-8-12-9/161827
17. Running Llama-3.1-8B-FP4 triton error sm_121a - NVIDIA forums | https://forums.developer.nvidia.com/t/running-llama-3-1-8b-fp4-get-triton-error-value-sm-121a-is-not-defined-for-option-gpu-name/348808

Engine summary (NOT a source): RLinf #1638 "2 days ago" says the PyPI torch 2.7.1 wheel is +cu126 with no sm_120 kernels; Triton #10331 says set TRITON_PTXAS_PATH to the system CUDA 13 ptxas, and that TRITON_OVERRIDE_ARCH=sm90 makes things worse.
