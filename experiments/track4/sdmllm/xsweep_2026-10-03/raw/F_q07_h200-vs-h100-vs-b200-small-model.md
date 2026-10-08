# F_q07 h200-vs-h100-vs-b200-small-model
- Query: `B200 vs H200 vs H100 small model training throughput tokens/sec 2026`
- Tool: WebSearch. Fetched ~2026-10-03 10:50 UTC.
- Hits: 37 (four link blocks). Mostly vendor and rental-blog pages, many inference-only.

Result list (selected verbatim title | url; full set below):
1. H200 achieves nearly 12,000 tokens/sec on Llama2-13B with TensorRT LLM | https://nvidia.github.io/TensorRT-LLM/blogs/H200launch.html
2. NVIDIA B200: 192GB Specs, Benchmarks, vs H100 - Spheron | https://www.spheron.network/blog/nvidia-b200-complete-guide/
3. NVIDIA H200 vs. B200 - vast.ai article | https://vast.ai/article/nvidia-h200-vs-b200-comparing-datacenter-grade-accelerators
4. NVIDIA H100 vs H200 - cudocompute | https://www.cudocompute.com/blog/nvidia-h100-vs-h200-how-will-they-compare
5. NVIDIA B200 vs H200 vs H100 - deploybase | https://deploybase.ai/articles/b200-vs-h200-vs-h100
6. B200 vs H100 - deploybase | https://deploybase.ai/articles/b200-vs-h100
7. NVIDIA B200 GPU (2026) - inworld | https://inworld.ai/resources/nvidia-b200-gpu-cloud
8. H100 vs H200 in 2026 - neysa | https://neysa.ai/blog/nvidia-h100-vs-h200/
9. B200 vs H200 - gpuvendor | https://gpuvendor.com/blog/b200-vs-h200
10. TensorRT-LLM perf overview | https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html
11-19. (inference: ori.co, MS techcommunity x2, arXiv 2409.03992, arXiv 2512.15674, HPCwire MLPerf Inference v5.1, simplismart, VALDI, koyeb)
20. NeMo Framework 25.02 performance summary | https://docs.nvidia.com/nemo-framework/user-guide/25.07/performance/archive/25.02-performance-summary.html
21. NeMo Automodel Performance Summary | https://docs.nvidia.com/nemo/automodel/performance/performance-summary
22. Megatron Bridge 0.5.1 Performance Archive | https://docs.nvidia.com/nemo/megatron-bridge/0.5.1/performance-summary-archive.html
23. Megatron Bridge 0.1.0 performance summary | https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/performance-summary.html
24. cioinfluence NeMo H200 | https://cioinfluence.com/it-and-devops/enhance-training-performance-with-latest-nvidias-nemo-framework-features-and-nvidia-h200/
25. Multi-GPU Benchmark B200 vs H200 vs H100 vs MI300X - aimultiple | https://research.aimultiple.com/multi-gpu/
26-28. NeMo 25.04 / 25.04 archive / 25.07 archive summaries (docs.nvidia.com)
29. premai GPU buying guide 2026 | https://www.premai.io/blog/gpu-buying-guide-for-llms-rtx-5090-vs-h100-vs-h200-complete-comparison-2026/
30. AI Application Benchmarking: Power-Aware Performance Analysis (arXiv 2603.16164) | https://arxiv.org/pdf/2603.16164
31. jarvislabs H100 vs H200 | https://jarvislabs.ai/ai-faqs/nvidia-h100-vs-h200-gpu-comparison
32. Spheron H100 vs H200 | https://www.spheron.network/blog/nvidia-h100-vs-h200/
33. whitefiber H100 vs H200 vs B200 vs B300 | https://www.whitefiber.com/blog/choosing-gpu-infrastructure
34. thundercompute best GPU 2026 | https://www.thundercompute.com/blog/best-gpu-for-llm
35. acecloud H200 | https://acecloud.ai/blog/train-llms-faster-nvidia-h200/
36. inferenceengineering GPU inference | https://inferenceengineering.tech/learn/gpu-inference/
37. TokenDyno throughput | https://tokendyno.com/blog/llm-throughput-benchmark/

Engine summary (NOT a source; to verify): NeMo 25.11 Llama3 8B FP8 B200 30,624 tok/s/GPU vs H100 14,451; arXiv 2603.16164 "at 700 W the H200 achieves ... 34 107 tokens/s, compared with 30 271 tokens/s for the H100". Llama-8B FP8 numbers are far from our ~small model regime.
