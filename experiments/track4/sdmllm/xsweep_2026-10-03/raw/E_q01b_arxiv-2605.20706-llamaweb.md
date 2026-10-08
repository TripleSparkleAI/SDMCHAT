# E q01b - primary check: arXiv 2605.20706 (LlamaWeb)

- url: https://arxiv.org/html/2605.20706v1
- tool: WebFetch (page converted, a small model extracts quotes; quotes below are as returned)
- fetched: 2026-10-03 10:49 UTC

## returned verbatim quotes
- dateline: "arXiv:2605.20706v1 [cs.DC] 20 May 2026"
- abstract opening: "Running language models in the browser presents a unique opportunity to build efficient, private, and portable AI applications, but requires contending with constrained memory availability and heterogeneous hardware targets."
- "on an Apple M3 the llama model with q4_k_m weights runs at ~1 tok/s on Firefox, but ~52 tok/s on Chrome"
- Llama 3.2 1B q4_k_m: decode ~100 tok/s on a high-end GPU, 4-17 tok/s on low-power mobile (returned as a summary line, not as a quote)
- "WebGPU does not yet support a way to directly pass small amounts of data to kernels at runtime"
- "allocates a single buffer at startup with enough slots to fit the parameters for a configurable number of kernels"
- "When subgroups are available, the kernel is specialized to perform a subgroup reduction, otherwise falling back to a generic shared memory reduction"
- "using f16 accumulation for the register-tiling kernel caused some models...to generate incoherent output on Apple M-series GPUs, so we currently use f32 accumulation"
- Table 1 (as returned): LlamaWeb 177,691 available models, WebLLM 400, Transformers.js 41,632.

## state
CONFIRMED for the quoted sentences (read from the arXiv HTML through the fetch tool). The "~100 tok/s" and "4-17 tok/s" figures were returned without a verbatim quote and stay UNVERIFIED.
