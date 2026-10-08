# F_q11 pytorch-docker-tags-blackwell
- Query: `pytorch docker image tag Blackwell 5090 B200 "cuda12.8" OR "cuda13.0" pytorch/pytorch 2.9 2.10 runtime devel`
- Tool: WebSearch. Fetched ~2026-10-03 10:52 UTC.
- Hits: 20.

Result list (title | url):
1. pytorch/pytorch - Docker Image tags | https://hub.docker.com/r/pytorch/pytorch/tags
2. Docker Setup for PyTorch, CUDA 12.8, Python 3.11 - Runpod | https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11
3. Hardened Images catalog dhi/pytorch | https://hub.docker.com/hardened-images/catalog/dhi/pytorch/guides
4. pytorch/manylinuxaarch64-builder tags | https://hub.docker.com/r/pytorch/manylinuxaarch64-builder/tags
5. layer 2.8.0-cuda12.8-cudnn9-devel | https://hub.docker.com/layers/pytorch/pytorch/2.8.0-cuda12.8-cudnn9-devel/images/sha256-a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
6. layer 2.10.0-cuda13.0-cudnn9-runtime | https://hub.docker.com/layers/pytorch/pytorch/2.10.0-cuda13.0-cudnn9-runtime/images/sha256-1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b
7. layer 2.7.0-cuda12.8-cudnn9-devel | https://hub.docker.com/layers/pytorch/pytorch/2.7.0-cuda12.8-cudnn9-devel/images/sha256-e97058f7b9b583517643477cc9a0433e54594038efd85ec4abd7836233d626f3
8. layer 2.7.0-cuda12.8-cudnn9-runtime | https://hub.docker.com/layers/pytorch/pytorch/2.7.0-cuda12.8-cudnn9-runtime/images/sha256-7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80
9. cnstark/pytorch-docker | https://github.com/cnstark/pytorch-docker
10. ghcr pytorch package | https://github.com/orgs/pytorch/packages/container/package/pytorch
11. layer 2.8.0-cuda12.8-cudnn9-runtime | https://hub.docker.com/layers/pytorch/pytorch/2.8.0-cuda12.8-cudnn9-runtime/images/sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
12. layer 2.9.1-cuda12.8-cudnn9-runtime | https://hub.docker.com/layers/pytorch/pytorch/2.9.1-cuda12.8-cudnn9-runtime/images/sha256-7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385
13-14. (dup layers)
15. Runpod guide (dup)
16. Pytorch support for sm120 page 3 - PyTorch Forums | https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099?page=3
17. HF blog pytorch install guide | https://huggingface.co/blog/daya-shankar/pytorch-install-guide
18. gpu-mart pytorch 2.4 docker | https://www.gpu-mart.com/blog/install-pytorch-with-gpu-support-on-docker
19. [Bug]: 5090 RTX seems to be broken - Issue #30493 vllm | https://github.com/vllm-project/vllm/issues/30493
20. (dup cnstark)

Engine summary (NOT a source): vLLM #30493 user: ghcr.io/pytorch/pytorch:2.9.1-cuda12.9-cudnn9-devel gives "Unexpected error from cudaGetDeviceCount()" on a 5090; Docker Hub lists 2.10.0-cuda13.0-cudnn9-runtime.
