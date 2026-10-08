# F_q08 b200-driver-cuda-nccl
- Query: `B200 driver CUDA 12.8 NCCL version error PyTorch training container problem`
- Tool: WebSearch. Fetched ~2026-10-03 10:50 UTC.
- Hits: 19.

Result list (title | url):
1. NVLS error, when running on aarch64 b200 - Issue #1769 NVIDIA/nccl | https://github.com/NVIDIA/nccl/issues/1769
2. Azure container apps #1682 driver 12080 incompatible with CUDA 12.8 containers | https://github.com/microsoft/azure-container-apps/issues/1682
3. NCCL+Torch Distributed Error - PyTorch Forums | https://discuss.pytorch.org/t/nccl-torch-distributed-error/220496
4. Pytorch version incompatible with cuda - PyTorch Forums | https://discuss.pytorch.org/t/pytorch-version-incompatible-with-cuda/154491
5. 2.7 test docker image has NCCL with older CUDA - Issue #150049 pytorch | https://github.com/pytorch/pytorch/issues/150049
6. PyTorch Release 25.01 - NVIDIA Docs | https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-01.html
7. Error with B200 cuda setup with torch.cuda cannot load - NVIDIA forums | https://forums.developer.nvidia.com/t/error-with-b200-cuda-setup-with-torch-cuda-cannot-load/336708
8. vLLM Server Launch Freezes at Using NCCL on B200 - Issue #20862 | https://github.com/vllm-project/vllm/issues/20862
9. PyTorch Release 17.06 | https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_17.06.html
10. ncclUnhandledCudaError ... Cuda failure 1 'invalid argument' for cuda12.8 - NVIDIA forums | https://forums.developer.nvidia.com/t/ncclunhandledcudaerror-call-to-cuda-function-failed-cuda-failure-1-invalid-argument-for-cuda12-8/324455
11. Troubleshooting CUDA Driver Initialization Failures - NVIDIA NIM | https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/cuda-driver.html
12. CSDN Error 802 | https://blog.csdn.net/zwhszdx/article/details/139861534
13. Error 802 system not yet initialized - PyTorch Forums | https://discuss.pytorch.org/t/runtimeerror-unexpected-error-from-cudagetdevicecount-error-802-system-not-yet-initialized/219320
14. CSDN 802 | https://blog.csdn.net/weixin_38060850/article/details/122439796
15. CSDN fabricmanager | https://blog.csdn.net/gary101818/article/details/132687029
16. B200 CUDA Error 802: A Strange Issue Solved by Enabling MIG - realdraw (2025-10-31) | https://tech.realdraw.ai/en/blog/2025_10_31_realdraw_b200/
17. Error 802 (wordpress, 2025-11-19) | https://marcellomorettoni.wordpress.com/2025/11/19/error-802-system-not-yet-initialized/
18. CUDA initialization failure Error 802 - NVIDIA forums | https://forums.developer.nvidia.com/t/cuda-initialization-failure-with-error-error-802-system-not-yet-initialized/337819
19. Error 802 CUDA 11.3 - NVIDIA forums | https://forums.developer.nvidia.com/t/error-802-system-not-yet-initialized-cuda-11-3/234955

Engine summary (NOT a source): Error 802 on B200 = fabric manager not running / version mismatch; PyTorch loads its own NCCL from pip (nvidia-nccl-cu12), not the system libnccl.
