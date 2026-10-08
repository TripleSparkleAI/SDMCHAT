# F_q06 rtx5090-p2p-blocked
- Query: `RTX 5090 P2P disabled NCCL_P2P_DISABLE DDP multi-GPU training slow`
- Tool: WebSearch. Fetched ~2026-10-03 10:50 UTC.
- Hits: 20.

Result list (title | url):
1. DDP training on RTX 4090 (ADA, cu118) - PyTorch Forums | https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366
2. Multi-GPU training hangs at startup on first NCCL collective (P2P/CUMEM deadlock, CUDA 13.2) - Issue #11 janickb/MaskDINO | https://github.com/janickb/MaskDINO/issues/11
3. DDP training on RTX 4090 #4 | https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366/4
4. P2P issue using two RTX 5090 GPUs - NVIDIA forums | https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776
5. Question about nccl p2p disable - Issue #631 NVIDIA/nccl | https://github.com/NVIDIA/nccl/issues/631
6. NCCL Tuning for Multi-GPU LLM Training (2026) - Spheron | https://www.spheron.network/blog/nccl-tuning-multi-gpu-llm-training-2026/
7. [Bug]: Multi GPU inference using two RTX 5090s (TP=2) - Issue #14628 vllm | https://github.com/vllm-project/vllm/issues/14628
8. What means there is no P2P support - vLLM Forums | https://discuss.vllm.ai/t/what-means-there-is-no-p2p-support/1928
9. NCCL P2P issue using two RTX 5090 - Issue #1637 NVIDIA/nccl | https://github.com/NVIDIA/nccl/issues/1637
10. Standard nVidia CUDA tests fail with dual RTX 4090 Linux box | https://forums.developer.nvidia.com/t/standard-nvidia-cuda-tests-fail-with-dual-rtx-4090-linux-box/233202/12
11. LingzheZhao/open-gpu-kernel-modules (P2P) | https://github.com/LingzheZhao/open-gpu-kernel-modules
12. Support RTX 5090 - Issue #29 tinygrad/open-gpu-kernel-modules | https://github.com/tinygrad/open-gpu-kernel-modules/issues/29
13. aikitoria/open-gpu-kernel-modules | https://github.com/aikitoria/open-gpu-kernel-modules
14. Duanyll/open-gpu-kernel-modules | https://github.com/Duanyll/open-gpu-kernel-modules
15. p2p disabled using 8x5090 after installation - Issue #42 tinygrad/open-gpu-kernel-modules | https://github.com/tinygrad/open-gpu-kernel-modules/issues/42
16. tinygrad-p2p-patched-driver.sh gist | https://gist.github.com/morgangiraud/4f58a62316fac7b4a32b81f0a66893fc
17. Multi-GPU Tinygrad Patch (Medium) | https://morgangiraud.medium.com/multi-gpu-tinygrad-patch-4904a75f8e16
18. RTX 5090 Support - Unraid | https://forums.unraid.net/topic/188921-rtx-5090-support/
19. Patching NVIDIA's driver and vLLM to enable P2P on consumer GPUs - smcleod.net (2026-02) | https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/
20. ASUS RTX 5090 OC not supported by 570.124.04 driver - NVIDIA forums | https://forums.developer.nvidia.com/t/asus-rtx-5090-oc-gpu-at-pci0-0-is-not-supported-by-the-570-124-04-driver-rhel-9-5/327260

Engine summary (NOT a source): NVIDIA forum reply says GeForce RTX 50 series does not support GPU-GPU P2P; NCCL #1637 dual-5090 hang fixed by a later NCCL release; patched drivers need rented-host root, unusable on Vast.
