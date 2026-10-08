# F_q20 vastai-filters-p2p (matrix row F_q20 repurposed: GitHub API on vast-ai deferred by rate limit)
- Query: `vast.ai search offers cuda_max_good driver_version filter Blackwell rented multi-GPU P2P "nvidia-smi topo"`
- Tool: WebSearch. Fetched ~2026-10-03 10:58 UTC.
- Hits: 18.

Result list (title | url):
1. Vast.ai CLI 101: Finding a GPU Offer You Can Actually Use - DEV Community | https://dev.to/big_mazzy_06d057cc24398c5/vastai-cli-101-finding-a-gpu-offer-you-can-actually-use-10op
2. search offers - Vast.ai docs | https://docs.vast.ai/api-reference/search/search-offers
3. Vast.ai | https://cloud.vast.ai/
4. search volumes - Vast.ai docs | https://docs.vast.ai/api-reference/volumes/search-volumes
5. vastai search instances - Vast.ai docs | https://docs.vast.ai/cli/reference/search-instances
6. Finding & Renting Instances - Vast.ai docs | https://docs.vast.ai/documentation/instances/choosing/find-and-rent
7. CLI Hello World - Vast.ai docs | https://docs.vast.ai/cli/get-started
8. vast-ai/vast-cli | https://github.com/vast-ai/vast-cli
9. gpunex Vast review 2026 | https://www.gpunex.com/blog/vast-ai-review-2026/
10. Multi-vGPU and P2P - NVIDIA AI Enterprise | https://docs.nvidia.com/ai-enterprise/release-7/latest/infra-software/vgpu/features/multi-vgpu-p2p.html
11. MIG on RTX Pro 6000 (Medium) | https://medium.com/@sangjinn/maximize-your-gpu-efficiency-configuring-4-nvidia-mig-instances-on-the-rtx-pro-6000-blackwell-1c9b3714af61
12. Enabling GPUDirect P2P in OpenStack VMs - menlo | https://menlo.ai/blog/gpudirect-p2p-openstack
13. Multi-gpu P2P capabilities and debugging tips (Medium, morgangiraud) | https://morgangiraud.medium.com/multi-gpu-nvidia-p2p-capabilities-and-debugging-tips-fb7597b4e2b5
14. eduardopessin/consumer-multigpu-inference (PCIe BAR1 P2P on consumer Blackwell, driver patch) | https://github.com/eduardopessin/consumer-multigpu-inference
15. Why 4 GPUs Can Be Slower Than 1 on Budget Clouds - cortwave | https://cortwave.github.io/posts/multi-gpu/
16. NVLink and GPU Interconnect - sushruta | https://sushruta.github.io/p/nvlink-multiple-gpus/
17. smcleod.net patching P2P 2026-02 | https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/
18. 4 tests fail (P2P related, 2 x Blackwell GPUs) - Issue #390 NVIDIA/cuda-samples | https://github.com/NVIDIA/cuda-samples/issues/390

Engine summary (NOT a source): Vast CLI fields cuda_max_good, driver_version, compute_cap, bw_nvlink, gpu_lanes; cortwave blog claims budget clouds disable P2P; consumer-multigpu-inference claims GeForce drivers stub out the BAR1 P2P route.
