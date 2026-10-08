# Band F - rented-GPU practice for single-box training on Blackwell and Hopper

Swept 2026-10-03, 10:47 to 11:05 UTC. Window 2026-08-01 to 2026-10-03, widened to 2026-05 and earlier where a primary source needed it.
The X API was unavailable (HTTP 402, credits depleted). This band used WebSearch, WebFetch and the unauthenticated GitHub REST search API.
The GitHub core API hit its rate limit during the sweep (403, 0 of 60 left on a shared IP), so issue pages were read with WebFetch instead.
Raw logs: `raw/F_q00_matrix_plan.md` to `raw/F_q25_primary_fetch_log.md`.

Reading rules for this file:
- WebSearch returns a search-engine summary beside the link list. That summary is never cited as a source here.
- A quote is marked verbatim only when the primary page returned it in quotation marks. WebFetch passes pages through a summariser, so a quote is exact text from the page as returned, not a screenshot.
- Dashes inside quoted text are normalised to '-'.

## 1. Search matrix

| raw file | query (short) | tool | hits | relevant primary found |
|---|---|---|---|---|
| F_q01 | RTX 5090 PyTorch sm_120 wheel CUDA 12.8 torch 2.9 | WebSearch | 18 | yes, wheel matrix (see F_q22) |
| F_q02 | torch.compile Triton sm_120 5090 github issue 2026 | WebSearch | 17 distinct | yes, Triton #10331 |
| F_q03 | RTX 5090 LLM training tokens/sec nanoGPT | WebSearch | 19 | yes, nanogpt-from-scratch, Little LM |
| F_q04 | RTX PRO 6000 Blackwell training benchmark multi-GPU | WebSearch | 10 | LoRA-only DDP numbers, not opened |
| F_q05 | RTX PRO 6000 P2P NCCL PCIe | WebSearch | 9 | yes, NCCL #1999, JAX #41182 |
| F_q06 | RTX 5090 P2P disabled NCCL DDP slow | WebSearch | 20 | partial, NCCL #1637 (fix version not visible) |
| F_q07 | B200 vs H200 vs H100 small-model training | WebSearch | 37 | no small-model training primary; mostly 8B FP8 and inference |
| F_q08 | B200 driver CUDA 12.8 NCCL container problems | WebSearch | 19 | not opened; Error 802 / fabric-manager anecdotes |
| F_q09 | Vast 5090 "no kernel image" 2026 | WebSearch | 20 | EMPTY for a Vast-specific 2026 thread; Vast RTX 5 docs opened |
| F_q10 | Vast known issues 2026 | WebSearch | 18 | reviews and competitor blogs only; none opened |
| F_q11 | pytorch docker tags Blackwell cuda12.8 / cuda13.0 | WebSearch | 20 | Docker Hub layer pages listed, not opened |
| F_q12 | index_add / scatter_add / embedding backward Blackwell atomics | WebSearch | 28 | EMPTY for a Blackwell-specific regression; #160838 opened and found to be FP64 |
| F_q13 | torchao FP8 training 5090 sm_120 rowwise | WebSearch | 18 | yes, pytorch #192707, ao #4932 |
| F_q14 | torchao MXFP8 B200 small model | WebSearch | 9 | torchao docs listed, not opened |
| F_q15 | nanochat single GPU 5090 / PRO 6000 / H200 / B200 | WebSearch | 29 | yes, nanochat #288; EMPTY for single H200 or B200 |
| F_q16 | modded-nanogpt 5090 / B200 / PRO 6000 single GPU | WebSearch | 29 | EMPTY for a single-GPU record on these cards |
| F_q17 | GitHub API: pytorch issues sm_120, 5090, B200 since 2026-05 | GitHub REST search | 18 / 49 / 66 | yes, #191433, #193752, #190796, #192707 |
| F_q18 | GitHub API: NCCL issues 5090, RTX PRO 6000, Blackwell P2P | GitHub REST search | 6 / 3 / 7 | yes, NCCL #2418 |
| F_q19 | NCCL version bundled in torch 2.9 to 2.11 | WebSearch | 58 (dupes) | not opened; torch 2.10 pin claim UNVERIFIED |
| F_q20 | Vast offer filters, P2P on rented boxes, nvidia-smi topo | WebSearch | 18 | cuda-samples #390 opened |
| F_q21 | fused / chunked large-vocab CE on sm_120 | WebSearch | 20 | EMPTY for an sm_120 benchmark; torch 2.13 blog opened |
| F_q22 | torch 2.11 / 2.12 / 2.13 CUDA wheel matrix, driver | WebSearch | 29 | yes, dev-discuss 3337, 3447, 2.13 blog |
| F_q23 | GitHub API: Vast-related issues 2026 | GitHub REST search | 3 / 302 (noise) / 15 / 19 | modded-nanogpt #365 opened |
| F_q24 | Vast B200 / PRO 6000 / H200 prices Sept 2026 | WebSearch | 29 | third-party trackers only, none opened |
| F_q25 | primary fetch log (22 pages, 1 API error) | WebFetch, urllib | - | - |

25 query rows. Explicit EMPTY results: 6 (F_q09 Vast thread, F_q12 Blackwell atomics, F_q15 single H200/B200 nanochat, F_q16 single-GPU speedrun, F_q21 sm_120 CE benchmark, and the F_q17 follow-up API call, which errored).

## 2. Ranked signals

### S1. NCCL 2.26.2 crashes on two RTX PRO 6000 for collectives of 512 KiB and above; 2.31.2 does not
- **State: CONFIRMED** as a reporter's measured issue. There is no maintainer explanation.
- Who and when: NVIDIA/nccl #2418, opened 2026-09-17, closed 2026-09-21. https://github.com/NVIDIA/nccl/issues/2418
- Environment: 2x RTX PRO 6000 Workstation Edition, driver 580.173.02, PyTorch 2.7.0+cu128, bare metal, no NVLink.
- Verbatim: "The test succeeds for smaller message sizes but reproducibly fails when reaching 524288 bytes" and "an illegal memory access was encountered". Failing NCCL "2.26.2+cuda12.2"; working "2.31.2+cuda12.9".
- Why it matters to us: DDP gradient buckets are far larger than 512 KiB. A box whose image ships torch 2.7.0 and its NCCL 2.26.2 would be expected to fail on the first all-reduce. The Vast RTX 5 page still names "PyTorch 2.7 or greater" as the floor.

### S2. P2P on multi-GPU RTX PRO 6000 hosts hangs or corrupts data, and the P2P capability flag does not predict it
- **State: CONFIRMED** as three independent reports. Root cause not established by NVIDIA in any of them.
- NVIDIA/nccl #1999, 2026-01-26, 2x PRO 6000, driver 580.119.02, NCCL 2.26.5 / 2.27.7 / 2.29.2. Verbatim: "the only one that makes a difference is this: NCCL_P2P_DISABLE=1" and "This is 100% reproducible, 100% failure rate." Closed the next day. The closing reason was not visible. https://github.com/nvidia/nccl/issues/1999
- jax-ml/jax #41182, 2026-10-01, 4x PRO 6000 Max-Q, driver 595.91.07, CUDA 13.2, NCCL 2.32.3, bare metal. With P2P on, collectives hang. Verbatim: "NCCL_P2P_DISABLE=1 (2 GPUs): completes, but prints correct: False". https://github.com/jax-ml/jax/issues/41182
- NVIDIA/cuda-samples #390, 2025-10-16, 2x PRO 6000, driver 580.95.05. Peer access reported "Yes" and "cudaMemcpyPeer / cudaMemcpy between GPU0 and GPU1: 103.88GB/s", yet simpleP2P and three other tests failed verification. https://github.com/NVIDIA/cuda-samples/issues/390
- The newest report runs NCCL 2.32.3, so S2 is not closed by the NCCL upgrade in S1.
- Hedge: all three are workstation or self-built hosts. None is a Vast host. Whether a given Vast host shows this is unknown until tested.

### S3. Inductor on sm_120 produces silently wrong results for bf16 autocast fusions close to our model's pattern
- **State: CONFIRMED** as reporter issues. #191433 is open and labelled "module: correctness (silent)" and "triaged".
- pytorch #191433, 2026-07-29, PyTorch 2.13.0+cu130, RTX 5080. Title: "[Inductor][Triton][CUDA][sm_120] BF16 autocast fusion of embeddings, Linear, and RMSNorm silently produces wrong results". Verbatim: "the compiled result is finite but shows a maximum absolute error of ≈ 2.2". https://github.com/pytorch/pytorch/issues/191433
- pytorch #190796, 2026-07-22, PyTorch 2.10.0+cu128, RTX 5090. bf16 autocast plus compile corrupted conv-bias gradients through a "MixOrderReductionGrid" kernel. Workaround `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`. The reporter states it is resolved in 2.11.0 (Triton 3.7.x) through 2.13.0. https://github.com/pytorch/pytorch/issues/190796
- Our job uses bf16 autocast, torch.compile, embedding-style gathers and a tied linear head. That is the pattern in #191433. We have not reproduced it.

### S4. The PyTorch CUDA wheel matrix moved off CUDA 12.8; images pinned to cu128 stop at torch 2.11
- **State: CONFIRMED** from PyTorch's own posts.
- dev-discuss 3337, atalman, 2026-04-01: "CUDA 12.8 is being deprecated and removed from CI/CD pipelines and binary build matrices." Release 2.12 matrix: 12.6.3 legacy, 13.0.x stable, 13.2.x experimental. https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337
- PyTorch 2.13 blog, 2026-07-08: "CUDA 13.0 remains the default build; small wheels are now always built for CUDA Linux and the CUDA 12.8/12.9 builds were removed; ptxas is no longer bundled in the cu13 binary." Also "Triton: pin advanced to 3.7.1." https://pytorch.org/blog/pytorch-2-13-release-blog/
- dev-discuss 3447, atalman, 2026-09-23: "CUDA Toolkit 13.2.2 fixes a compiler bug that has been present since CUDA 12.8." It produces "silently incorrect results - no crash, no error, just wrong numbers." It "only triggers in kernels with two or more nested levels of thread divergence". The same post says "PyTorch 2.14 and earlier are unaffected." The summary did not make clear whether "unaffected" refers to the bug or to the build-matrix change. https://dev-discuss.pytorch.org/t/removing-cuda-13-0-builds-from-pytorch-ci-cd-starting-the-week-of-sept-28/3447
- The driver floor for cu130 wheels (a figure of 580.65.06 appeared in a search summary) was NOT found on the 3337 page. **UNVERIFIED.**
- Consequence: a Vast host whose driver reports a maximum CUDA version of 12.8 cannot run the current default wheels. The host filter must follow the wheel.

### S5. Multi-GPU bf16 training on 4x RTX PRO 6000 gave different eval losses on identical resume launches
- **State: POST CLAIM ONLY.** One reporter. No maintainer diagnosis.
- pytorch #193752, 2026-08-17, "4x NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB, sm_120), GCE g4 instance", nvidia-nccl-cu13 2.30.7, driver 610.43.02, torch 2.11 and a 2.15 nightly. Six identical resume launches printed "eval_on_start = 6.501 | 0.1723 | 0.1884 | 6.501 | 0.2461 | 3.325". Single-GPU runs were consistent. https://github.com/pytorch/pytorch/issues/193752
- Relevance: this is a cloud host, not a workstation. It supports checking collective results by value on a rented box.

### S6. Single-card training throughput anchors for small GPT-style models
- **State: CONFIRMED** that each source states the number. None is our model.
- Padraigobrien08/nanogpt-from-scratch (GPT-2 124M): "1 GPU: 185,928 tokens/sec | 2 GPU: 366,469 | 4 GPU: 721,904 | 8 GPU: 1,414,340" and "95.1% efficiency at 8 GPUs" on "8× RTX 5090, no NVLink". H100 "298,199 tokens/sec" baseline, "377,315" with torch.compile. No date, torch version or provider on the page as returned. https://github.com/Padraigobrien08/nanogpt-from-scratch
- Little LM blog, 2026-09-04, FP8, 150M model: "RTX 5090 at 184,662 tok/s, B200 at 477,440 tok/s. 2.59× from hardware alone". Provider not named. https://hugovergnes.github.io/little-lm-3-8b/
- nanochat discussion #288, 2025-11-13, single RTX PRO 6000 Server Edition, torch 2.9 with CUDA 13, d20 model: "around 85k tokens/s or 142 TPS/W" at 600 W and "around 58k TPS or 193 TPS/W" at 300 W. https://github.com/karpathy/nanochat/discussions/288
- The 8x 5090 result shows that DDP over PCIe without NVLink scaled at 95.1% for a 124M model. It does not say whether P2P was on.
- EMPTY: no single-H200 or single-B200 small-model training number was found.

### S7. Rowwise FP8 GEMM does not run on sm_120 through cuBLAS; tensorwise does; MXFP8 is TN-only on cc 12.x
- **State: CONFIRMED** for #192707 (reporter measurement, triaged as an enhancement). **POST CLAIM ONLY** for the ao #4932 speed figure.
- pytorch #192707, 2026-08-10, RTX PRO 5000 Blackwell (sm_120), PyTorch 2.10.0+cu130, cuBLAS 13.1.0.3. Verbatim: "Per-row _scaled_mm fails at the HEURISTIC stage: cublasLtMatmulAlgoGetHeuristic returns 0 algorithms" and "PerTensor with the same a/b/out_dtype succeeds." https://github.com/pytorch/pytorch/issues/192707
- pytorch/ao #4932, 2026-09-22, RTX 5060 Ti, torch 2.14.0+cu130. Quotes cuBLAS documentation that block-scaled FP8 requires TN on "Blackwell GeForce (compute capability 12.x) GPUs". TN MXFP8 measured "542us vs 1674us at 4096x3072x3072" against bf16. No maintainer reply. https://github.com/pytorch/ao/issues/4932
- Relevance: our 129,280 x d output head is the GEMM most likely to gain from FP8. On a 5090 or PRO 6000, a rowwise recipe is expected to fail at the first call. A tensorwise recipe is expected to run. Speed on our shapes is unmeasured.

### S8. PyTorch 2.13 ships a fused, chunked linear plus cross-entropy module
- **State: CONFIRMED** from the release blog. Marked "API Unstable".
- PyTorch 2.13 blog, 2026-07-08: nn.LinearCrossEntropyLoss "processes the vocabulary dimension in chunks, never materializing the full logits matrix", "reduces peak memory by up to ~4x for large-vocabulary workloads", and "supports label smoothing, weight tying, and z-loss regularization out of the box". https://pytorch.org/blog/pytorch-2-13-release-blog/
- Relevance: it is a candidate replacement for our hand-chunked head. Its speed against ours is unmeasured, and the "4x" is a memory claim, not a speed claim.

### S9. Vast specifics: the RTX 5 floor is stated as CUDA 12.8 and PyTorch 2.7, and 8x H100 nodes are reported as unavailable
- **State: CONFIRMED** for the Vast docs page text. **POST CLAIM ONLY** for availability.
- docs.vast.ai/rtx-5-series (no date on page): Blackwell needs "CUDA 12.8 and PyTorch 2.7 or greater"; "If you manually change the Docker image, ensure it's compiled for CUDA 12.8 or else you may lose compatibility with these GPUs." https://docs.vast.ai/rtx-5-series
- That floor is now the version family that carries NCCL 2.26.2 (S1).
- modded-nanogpt #365, ELanning, 2026-09-06: "Currently on Runpod and Vast.ai it is not even possible to rent a 8xH100 node." https://github.com/KellerJordan/modded-nanogpt/issues/365

### S10. Late-September price snapshots put a single B200 at about the whole budget rate
- **State: UNVERIFIED.** Third-party tracker figures seen only in search results. No tracker page was opened.
- Reported Vast ranges per GPU-hour: B200 about $4.85 to $10.41 (a typical figure of $6.25 for 2026-09-28); H200 about $2.63 to $5.12; RTX PRO 6000 about $1.10 to $1.99. Sources listed in `raw/F_q24_vastai-pricing-sept2026.md`.
- Our budget is about $300 for about 48 hours, which is $6.25 per hour for the whole box. If these snapshots hold, one B200 uses the full budget, and four RTX PRO 6000 cost about the same.

### S11. Our DGX Spark (GB10, sm_121): Triton needs the system ptxas
- **State: CONFIRMED** as a reporter's recipe. Not an NVIDIA statement.
- triton-lang/triton #10331, eniktab, 2026-05-18, GB10, PyTorch 2.9.0+cu130, Triton 3.5. Verbatim: "The PyTorch-bundled ptxas predates sm_121." Fix: "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas". It warns against "TRITON_OVERRIDE_ARCH=sm90". https://github.com/triton-lang/triton/issues/10331
- This concerns our own Spark, not a rented box.

### S12. Negative findings, recorded so nobody chases them
- "backward on 5090 much slower than on 4090" (pytorch #160838, 2025-08-17) is an FP64 GEMM case. The profiler line is "cutlass_80_tensorop_d884gemm" at "97.93%" of backward time. It is not evidence about bf16 training or about atomics. https://github.com/pytorch/pytorch/issues/160838
- No Blackwell-specific regression in scatter_add, index_add or embedding backward was found (F_q12 EMPTY). The known contention issues (#20655 row contention, #74487 fp16 with concentrated indices) predate Blackwell. Whether our 65,536-row scatter-add hits contention depends on our index distribution, which is ours to measure.
- No Vast-specific 2026 thread on "no kernel image" was found (F_q09 EMPTY).
- The claim that GeForce RTX 50 cards do not support P2P appeared only in a search summary of NVIDIA forum replies. The forum page was not opened. **UNVERIFIED.**

## 3. Proposed changes to how we rent and set up boxes

At most four. Each has a pre-flight test that runs in under a minute on a fresh box.

### P1. Gate the software stack on the box before anything else
- Require torch 2.11 or newer with a cu130 build. Refuse NCCL 2.26.x. Filter Vast offers so the host's maximum CUDA version is at least the wheel's CUDA version.
- Sources: S1 (NCCL #2418), S4 (dev-discuss 3337, 2.13 blog), S3 (#190796 fixed in 2.11).
- Test: `python -c "import torch as t; v=t.cuda.nccl.version(); print(t.__version__, t.version.cuda, v, t.cuda.get_device_capability(), t.cuda.get_arch_list()); assert v[:2]!=(2,26), 'NCCL 2.26.x'; assert any(a in t.cuda.get_arch_list() for a in ('sm_100','sm_120','sm_90')); x=t.randn(4096,4096,device='cuda',dtype=t.bfloat16); print((x@x).float().abs().mean().item())"`
- `torch.cuda.nccl.version()` reports the NCCL torch was built against. A pip-installed NCCL can differ. The `NCCL_DEBUG=INFO` line in P2 shows the one that actually loads.

### P2. On any multi-GPU box, prove collectives by value before renting time to the run
- Run an all-reduce at 256 KiB, 1 MiB and 32 MiB under a 60-second timeout. Check that the result equals world_size times the input. Run it with P2P at its default and again with `NCCL_P2P_DISABLE=1`.
- If default fails and disabled passes, run training with P2P disabled. If both fail, release the box. Log `nvidia-smi topo -m` and `nvidia-smi topo -p2p r`, but do not trust them alone.
- Sources: S1 (NCCL #2418 fails at exactly 524288 bytes), S2 (NCCL #1999 hang; JAX #41182 "correct: False" with P2P off; cuda-samples #390 peer access "Yes" with failed verification), S5 (#193752 inconsistent multi-GPU results on a cloud host).
- Test: `NCCL_DEBUG=INFO timeout 60 torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) allreduce_check.py` where the script all-reduces `torch.ones(n, device='cuda')` for each size and asserts every element equals the world size. Repeat with `NCCL_P2P_DISABLE=1`.

### P3. Check compiled against eager on the box, on one real batch, before the long run
- Run one fixed batch eager and compiled, both under bf16 autocast. Compare the loss and the per-parameter gradient max absolute difference against a tolerance we set from a known-good box.
- If it fails, retry with `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`. If it still fails, run eager or narrow the compiled region, and record which.
- Sources: S3 (#191433 open on 2.13 for embeddings plus Linear plus RMSNorm under bf16 autocast on sm_120; #190796 gradient corruption on 2.10).
- Test: one forward and backward per mode on a seeded batch at our B 32 x T 256. The compile time is the cost, so cap the test at a small d if one minute is tight.

### P4. Probe the FP8 recipe on the box before choosing it
- On sm_120 (5090, PRO 6000), call per-row and per-tensor `torch._scaled_mm` once on a head-shaped GEMM. Expect per-row to fail. Use a tensorwise recipe if FP8 is wanted at all. On sm_100 (B200), run the same call to confirm both work. Time FP8 against bf16 on our 129,280 x d head before adopting it.
- Sources: S7 (pytorch #192707 per-row heuristic returns 0 algorithms on sm_120; ao #4932 TN-only MXFP8 on cc 12.x).
- Test: `python -c "import torch as t; a=t.randn(8192,1024,device='cuda').to(t.float8_e4m3fn); b=t.randn(4096,1024,device='cuda').to(t.float8_e4m3fn).t(); s=t.ones((),device='cuda'); print('tensorwise', t._scaled_mm(a,b,scale_a=s,scale_b=s,out_dtype=t.bfloat16).shape); ra=t.ones(8192,1,device='cuda'); rb=t.ones(1,4096,device='cuda'); print('rowwise', t._scaled_mm(a,b,scale_a=ra,scale_b=rb,out_dtype=t.bfloat16).shape)"` (the rowwise line is expected to raise on sm_120).
- These snippets are written for this file and have not been run by this sweep.
