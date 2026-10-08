# E q19 - WebLLM params_shard size and why shards

- query: `WebLLM params_shard size why shards Cache API IndexedDB model weights`
- tool: WebSearch (two internal searches)
- fetched: 2026-10-03 10:57 UTC

## result list
1. LatexGen issue #70 - https://github.com/OlehZhyhinas/LatexGen/issues/70
2. LatexGen PR #72 - https://github.com/OlehZhyhinas/LatexGen/pull/72
3. GitHub - NrenV09/WebLLM-AI - https://github.com/NrenV09/WebLLM-AI
4. How to Run a Small LLM in Your Browser with WebLLM (tinyweights.dev) - https://tinyweights.dev/posts/run-llm-in-browser-webllm/
5. webSLM ... DEV Community - https://dev.to/vishalmysore/webslm-fine-tuning-compiling-and-running-domain-specific-small-language-models-entirely-in-the-1i5i
6. WebGPU Inference: LLMs That Run in Your Browser (medium)
7. GitHub - future-ai-org/tiny-models-inference - https://github.com/future-ai-org/tiny-models-inference
8. WebLLM Javascript SDK - mlc-llm docs - https://llm.mlc.ai/docs/deploy/webllm.html
9. WebLLM: Run LLMs in the Browser with JavaScript - W3Tweaks
10. (tvm) [Feat][Web] Support per-parameter tensor cache encoding (#20136) - http://www.mail-archive.com/commits@tvm.apache.org/msg118832.html
11. (tvm) [Contrib] Support NDArray cache taking generator (#16693) - https://www.mail-archive.com/commits@tvm.apache.org/msg106512.html
12. [Bug] · Issue #2730 · mlc-ai/mlc-llm - https://github.com/mlc-ai/mlc-llm/issues/2730
13. Files (vmray report)
14. Re: [PR] [Contrib] Support NDArray cache taking generator [tvm]
15. Compiling for WebGPU on OSX ... Issue #452 · mlc-ai/mlc-llm
16. webSLM (medium)
17. TVM RPC will fail when allocating large arrays on an Android phone - TVM Discuss

## primary check: apache/tvm main, python/tvm/contrib/tvmjs.py (python3 urllib, 10:58 UTC)
- `def dump_tensor_cache( params: ..., cache_dir: str, encode_format: str | Mapping[str, str] = "f32-to-bf16", meta_data=None, shard_cap_mb=32, show_progress: bool = True, update_if_exists: bool = False,`
- code lines: `if self.pending_nbytes + len(data) >= self.shard_cap_nbytes:` and `if len(data) * 2 >= self.shard_cap_nbytes:` (a tensor larger than half the cap gets its own shard; it is not split).
- state: CONFIRMED. MLC's reasons for 32 MB are not stated in the code read.

## primary check: mlc-ai/Qwen3-0.6B-q4f16_1-MLC tensor-cache.json (see E_q07): 9 shards, the 4-bit embedding alone is shard 0 at 77.8 MB, the other shards are 30-34 MB. CONFIRMED.
