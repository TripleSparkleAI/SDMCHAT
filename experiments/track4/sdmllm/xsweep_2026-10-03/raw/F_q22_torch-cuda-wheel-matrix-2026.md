# F_q22 torch-cuda-wheel-matrix-2026 (matrix row F_q21)
- Query: `PyTorch 2.11 OR 2.12 OR 2.13 release CUDA 12.8 wheels dropped cu130 default Blackwell driver 580`
- Tool: WebSearch. Fetched ~2026-10-03 10:58 UTC.
- Hits: 29.

Result list (title | url):
1. Introducing CUDA 13.2 and Deprecating CUDA 12.8 (Release 2.12) - dev-discuss | https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337
2. Addition of CUDA 13.0 nightly builds and deprecation CUDA 12.9 builds - dev-discuss | https://dev-discuss.pytorch.org/t/addition-of-cuda-13-0-nightly-builds-and-deprecation-cuda-12-9-builds/3209
3. Releases - pytorch/pytorch | https://github.com/pytorch/pytorch/releases
4. Transitioning PyPI CUDA wheels to CUDA 13.0 as the stable (Release 2.11) - dev-discuss | https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325
5. Proposal to bring back 12.9 wheels - Issue #165165 | https://github.com/pytorch/pytorch/issues/165165
6. Issue #167817 nightly Windows RTX 5080 | https://github.com/pytorch/pytorch/issues/167817
7. RELEASE.md | https://github.com/pytorch/pytorch/blob/main/RELEASE.md
8. rapidsai build-planning #255 | https://github.com/rapidsai/build-planning/issues/255
9. Previous PyTorch Versions | https://pytorch.org/get-started/previous-versions/
10. raidenworks install | https://raidenworks.com/2024/01/20/torch-cuda-is_available-false-%F0%9F%99%84/
11. PyTorch 2.12 Release Blog | https://pytorch.org/blog/pytorch-2-12-release-blog/
12. PyTorch cannot detect RTX 5080 - Forums | https://discuss.pytorch.org/t/pytorch-cannot-detect-rtx-5080-requires-cuda-12-8-nightly-build-fails-to-install-correctly-on-windows/223957
13-15. pytorch/xla PR files, NVIDIA 21.08 notes
16. Removing CUDA 13.0 builds from PyTorch CI/CD, starting the week of Sept 28 - dev-discuss | https://dev-discuss.pytorch.org/t/removing-cuda-13-0-builds-from-pytorch-ci-cd-starting-the-week-of-sept-28/3447
17. [CD] Remove CUDA 13.0 from the binary build matrix - PR #198913 | https://github.com/pytorch/pytorch/pull/198913
18. vLLM PR #55387 Move default CUDA build from 13.0 to 13.2 | https://github.com/vllm-project/vllm/pull/55387
19. [RFC] CUDA support matrix for Release 2.14 - promote CUDA 13.2 - Issue #190355 | https://github.com/pytorch/pytorch/issues/190355
20. test-infra PR #8932 | https://github.com/pytorch/test-infra/pull/8932
21. PyTorch 2.13 Release Blog | https://pytorch.org/blog/pytorch-2-13-release-blog/
22. freenode PyTorch 2.14 RFC article | https://freenode.net/article/pytorch-2-14-rfc-would-make-cuda-13-2-the-default-pypi-build

Engine summary (NOT a source; verify at dev-discuss 3447 and 3337): "CUDA Toolkit 13.2.2 fixes a compiler bug that has been present since CUDA 12.8, one that could cause silently incorrect results"; 2.12 Blackwell cu130 needs driver 580.65.06 (Linux); 2.13 "ptxas is no longer bundled in the cu13 binary".
