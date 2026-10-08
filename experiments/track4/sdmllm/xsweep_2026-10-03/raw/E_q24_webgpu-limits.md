# E q24 - WebGPU default buffer limits

- query: `WebGPU maxStorageBufferBindingSize maxBufferSize default limit 128MB 256MB adapter limits LLM weights`
- tool: WebSearch (two internal searches)
- fetched: 2026-10-03 10:56 UTC

## result list
1. Run an LLM Inside a Browser Tab (pinggy) - https://pinggy.io/blog/run_llm_in_browser_webgpu/
2. GitHub - nff747/webgpu-vram-pager - https://github.com/nff747/webgpu-vram-pager
3. [webkit-changes] [WebGPU] default limit for maxBufferSize is incorrect - https://www.mail-archive.com/webkit-changes@lists.webkit.org/msg209948.html
4. Babylon.js forum: maximum maxStorageBufferBindingSize - https://forum.babylonjs.com/t/question-about-the-maximum-maxstoragebufferbindingsize-of-webgpu-storagebuffer-array/50678
5. WebGPU Memory Limits: maxStorageBufferBindingSize (ayoob.ai) - https://ayoob.ai/blog/webgpu-maxstoragebufferbindingsize-limits-enterprise
6. WebGPU bugs are holding back the browser AI revolution (medium) - https://medium.com/@marcelo.emmerich/webgpu-bugs-are-holding-back-the-browser-ai-revolution-27d5f8c1dfca
7. Limits in wgpu - Rust - https://wgpu.rs/doc/wgpu/struct.Limits.html
8. GPUSupportedLimits - webdocs.dev
9. docs.rs wgpu Limits
10. (irrelevant) WebHttpBindingElement.MaxBufferSize - learn.microsoft.com
11. Aura PR #31 - https://github.com/barakplasma/Aura/pull/31
12. Coarse "available memory" signal on GPUAdapter ... Issue #6957 · gpuweb/gpuweb - https://github.com/gpuweb/gpuweb/issues/6957
13. Deep Dive: Bypassing Browser Memory Caps ... DEV
14. GPUSupportedLimits - MDN - https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits
15. GPUAdapter: limits property - MDN
16. GPUDevice: limits property - MDN

## tool summary claims (unverified unless checked)
- WebLLM requests 1 GB for both limits and retries at 256 MB and 128 MB.
- "Chrome 153 on an M3 Pro Mac reports a 4,294,967,292-byte ceiling for both limits"
- adapter-limit ranges per vendor from one survey (Adreno 256-512 MB, Mali 128-256 MB) - UNVERIFIED.

## primary check: MDN GPUSupportedLimits (WebFetch 10:57 UTC)
- maxBufferSize default 268435456 bytes; maxStorageBufferBindingSize default 134217728 bytes; maxComputeWorkgroupStorageSize 16384; maxComputeInvocationsPerWorkgroup 256.
- higher limits are requested through `requiredLimits` in `GPUAdapter.requestDevice()`.
- state: CONFIRMED.
