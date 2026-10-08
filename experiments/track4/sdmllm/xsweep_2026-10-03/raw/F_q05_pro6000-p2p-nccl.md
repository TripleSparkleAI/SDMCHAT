# F_q05 pro6000-p2p-nccl
- Query: `RTX PRO 6000 Blackwell P2P NCCL PCIe peer-to-peer disabled multi-GPU`
- Tool: WebSearch. Fetched ~2026-10-03 10:50 UTC.
- Hits: 9.

Result list (title | url):
1. rtx6kpro/hardware/pcie-bandwidth.md - local-inference-lab/rtx6kpro | https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/pcie-bandwidth.md
2. rtx6kpro/optimization/nccl-tuning.md - voipmonitor/rtx6kpro | https://github.com/voipmonitor/rtx6kpro/blob/master/optimization/nccl-tuning.md
3. Dual RTX PRO 6000 Blackwell Max-Q - how to make P2P NCCL work - Level1Techs | https://forum.level1techs.com/t/dual-rtx-pro-6000-blackwell-max-q-how-to-make-p2p-nccl-work/242403
4. NCCL P2P hang on dual RTX PRO 6000 Blackwell Workstation Edition (WRX90E-SAGE SE) - NVIDIA forums | https://forums.developer.nvidia.com/t/nccl-p2p-hang-on-dual-rtx-pro-6000-blackwell-workstation-edition-wrx90e-sage-se/365048
5. rtx6kpro/troubleshooting/common-issues.md | https://github.com/local-inference-lab/rtx6kpro/blob/master/troubleshooting/common-issues.md
6. [Issue]: NCCL hangs during AllReduce with P2P enabled on dual RTX Pro 6000's - Issue #1999 NVIDIA/nccl | https://github.com/nvidia/nccl/issues/1999
7. Cross-device collectives hang on 2x RTX PRO 6000 Blackwell - Issue #41182 jax-ml/jax | https://github.com/jax-ml/jax/issues/41182
8. [Bug] P2P: tensor parallelism issue with two RTX PRO 6000 MAXQ - Issue #15181 sgl-project/sglang | https://github.com/sgl-project/sglang/issues/15181
9. NVIDIA forums: can 2 Blackwell workstation cards 6000 pro peer-to-peer in linux | https://forums.developer.nvidia.com/t/can-2-blackwell-workstation-cards-6000-pro-peer-to-peer-in-linux/338905/3

Engine summary (NOT a source; to verify at primaries): NCCL #1999 tested NCCL 2.26.5, 2.27.7, 2.29.2, hangs with P2P, only NCCL_P2P_DISABLE=1 helps; JAX #41182 "1 day ago" 4x PRO 6000 hang, P2P disabled gives wrong result; rtx6kpro guide: P2P on ~0.36-0.45 us vs ~14 us off; fixes IOMMU/ACS off, uvm_disable_hmm=1, ForceP2P registry dwords.
