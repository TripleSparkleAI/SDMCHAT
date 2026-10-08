# E q20 - arXiv Atom API, newest papers (window filter 2026-06-01 onward)

- tool: python3 urllib, export.arxiv.org/api/query, sortBy submittedDate descending, max_results 15
- three search_query strings, verbatim in the output below; entries older than 2026-06-01 were dropped from the printout

```
fetched 2026-10-03T10:54:35.572132Z
== query: all:WebGPU AND all:"language model" | returned 4
  2026-08-09 http://arxiv.org/abs/2608.08730v2 | Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference
== query: all:browser AND all:"language model" AND all:inference | returned 15
  2026-09-27 http://arxiv.org/abs/2609.33731v1 | HTN Planning as a Coordination Layer for Multi-Server MCP Tool Orchestration
  2026-09-02 http://arxiv.org/abs/2609.02088v1 | Rendering-in-the-Loop: An Execution-Driven Agent for Interactive Web Development
  2026-08-28 http://arxiv.org/abs/2608.28950v2 | The Web-CLI: Verifiable Privacy for Tools, Models, and Inference Engines in the Browser
  2026-08-09 http://arxiv.org/abs/2608.08730v2 | Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference
  2026-06-12 http://arxiv.org/abs/2606.14654v1 | Abstracting Cross-Domain Action Sequences into Interpretable Workflows
== query: abs:"on-device" AND abs:"embedding" AND abs:quantization AND abs:vocabulary | returned 4
  2026-07-14 http://arxiv.org/abs/2607.13093v4 | Efficient and Privacy Aware Edge Cloud Collaborative Inference for Large Language Models
  2026-07-10 http://arxiv.org/abs/2607.09957v2 | Workload-Driven Optimization for On-Device Real-Time Subtitle Translation
  2026-06-26 http://arxiv.org/abs/2606.27871v1 | LocalNav: Distilling Frontier VLMs and Embodied RL for On-Device Object Goal Navigation
```

## follow-up abstract reads (arXiv API, 10:55 UTC)
- 2608.08730 "Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference", published 2026-08-09T14:21:55Z. Abstract verbatim excerpt: "We show that the dispatch overhead, not kernel quality, is the bottleneck at batch size 1, and isolate the dispatch count as the cause. Therefore, we conclude that at batch size 1, the effective approach to LLM inference optimization in WebGPU is reducing dispatch count."
- HTML body (WebFetch of https://arxiv.org/html/2608.08730v2, 10:56 UTC), as returned: Vulkan "24-36 µs" per dispatch, Metal "32-71 µs" per dispatch; naive single-op measurement overestimates by "∼20x"; "A reduction of dispatches from 876 to 564, while keeping computationally the same WGSL shaders and with negligible memory savings, improves throughput by 53%" (Qwen2.5-0.5B, NVIDIA RTX 5090, Dawn/Vulkan; TTFT 71.4 ms to 41.6 ms); "Command buffer submission dominates the dispatch" at "12.9 µs per dispatch".
- 2608.28950 "The Web-CLI", published 2026-08-28. Architecture paper; no small-LM throughput numbers in the abstract. Low relevance.
- state: CONFIRMED for the abstract; the body numbers came through the fetch tool's extraction and are recorded as returned.
