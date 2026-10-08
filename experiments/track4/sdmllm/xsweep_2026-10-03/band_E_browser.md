# Band E - browser and on-device small LMs

Sweep run 2026-10-03, 10:47 to 10:58 UTC. Window 2026-08-01 to 2026-10-03, widened to 2026-06-01 where needed. Older platform facts are kept when they are still the standing state (marked with their own date).

The X API returned HTTP 402 (credits depleted), so this band uses the web-mirror fallback: WebSearch, WebFetch, and python3 urllib against the arXiv Atom API, the HF Hub API, the npm registry and raw GitHub files. The GitHub REST API was also unavailable from this host (HTTP 403 "rate limit exceeded" on every unauthenticated call). Two X posts appear only as search-engine titles; they are marked "via search, X not fetched" and their engagement is unknown.

Every query has a raw record in `raw/E_qNN_*.md`, written before judging. WebSearch returns a title and url list plus a tool-written summary with no per-result dates or snippets. The raw files record that summary as unverified and keep primary checks separate.

## 1. Search matrix

| id | axis | query (verbatim) | tool | hits | primary checks |
|---|---|---|---|---|---|
| q01 | engine | WebLLM WebGPU tokens per second 2026 | WebSearch | 9 | arXiv 2605.20706 (q01b) |
| q02 | engine | transformers.js v4 WebGPU release | WebSearch | 9 | HF blog (q02b) |
| q03 | engine | ONNX Runtime Web WebGPU LLM 2026 | WebSearch | 9 | none (secondary only) |
| q04 | engine | wllama llama.cpp wasm 2026 | WebSearch | 9 | npm registry (q21) |
| q05 | model size | Gemma 3 270M browser WebGPU transformers.js download size | WebSearch | 9 | webgpu-gemma README, HF API |
| q06 | model size | SmolLM3 on-device browser WebGPU 2026 | WebSearch | 9 | none new |
| q07 | model size | Qwen3 0.6B WebGPU browser tokens per second q4f16 | WebSearch | 9 | HF API, MLC tensor-cache.json |
| q08 | model size | LFM2.5 230M WebGPU browser tokens per second | WebSearch | 9 | Liquid AI blog, HF config |
| q08b | model size | webml-community LFM2.5 WebGPU kernels Hugging Face Space 1400 tok/s Victor Mustar | WebSearch | 15 (2 are X posts, not fetched) | none for the 1,400 figure |
| q09 | export | q4f16 ONNX export WebGPU MatMulNBits block size 32 transformers.js quantization | WebSearch | 10 | graph op names (q22b) |
| q10 | export | embedding table quantization int4 on-device LLM vocabulary embedding memory 2026 | WebSearch | 10 | DEV article (Gemma 4 int4 embeddings) |
| q11 | export | tied lm_head quantization large vocabulary output layer small model on-device int4 accuracy | WebSearch | 9 | arXiv 2608.02703 abstract |
| q12 | platform | WebGPU subgroups shipped Chrome stable release | WebSearch | 9 | Chrome 134 blog |
| q13 | platform | Safari 26 WebGPU shader-f16 subgroups support | WebSearch | 26 | Safari TP 249 notes |
| q14 | platform | Firefox WebGPU 2026 release macOS Linux shader-f16 | WebSearch | 15 | gpuweb Implementation Status wiki |
| q15 | platform | WebAssembly relaxed SIMD matrix multiply dot product int8 browser support 2026 | WebSearch | 9 | caniuse |
| q16 | platform | wasm memory64 shipped Chrome Firefox Safari 4GB limit 2026 | WebSearch | 16 | none (not decision-relevant) |
| q17 | kernel | WebGPU top-k sampling argmax large vocabulary GPU logits readback | WebSearch | 9 | pooled PR #98 |
| q18 | delivery | OPFS Cache API storing LLM model weights browser quota 2026 | WebSearch | 9 | LatexGen issue #70 |
| q19 | delivery | WebLLM params_shard size why shards Cache API IndexedDB model weights | WebSearch | 17 | apache/tvm tvmjs.py source |
| q20 | papers | arXiv `all:WebGPU AND all:"language model"` / `all:browser AND all:"language model" AND all:inference` / `abs:"on-device" AND abs:"embedding" AND abs:quantization AND abs:vocabulary`, newest first | arXiv Atom API | in window: 1 / 5 / 3 (returned 4 / 15 / 4) | 2608.08730 abstract and HTML |
| q21 | releases | GitHub REST /releases for transformers.js, web-llm, onnxruntime, wllama, llama.cpp | python3 urllib | EMPTY (HTTP 403 rate limit exceeded, all 5) | - |
| q21b | releases | npm registry for @huggingface/transformers, @mlc-ai/web-llm, onnxruntime-web, @wllama/wllama | python3 urllib | 4 of 4 | primary |
| q22 | models | HF Hub tree API for 8 small-model repos | python3 urllib | 6 of 8 (2 returned HTTP 401) | primary |
| q22b | models | 3 small ONNX graph files, op-name string counts | python3 urllib | 3 of 3 | primary (string count, not a parsed graph) |
| q23 | models | HF Hub tree API, onnx-community/gemma-4-E2B-it-ONNX | python3 urllib | 1 of 1 | primary |
| q24 | platform | WebGPU maxStorageBufferBindingSize maxBufferSize default limit 128MB 256MB adapter limits LLM weights | WebSearch | 16 | MDN GPUSupportedLimits |
| q25 | model size | Gemma 4 E2B browser WebGPU transformers.js size tokens per second | WebSearch | 17 | file sizes only (q23) |

27 query lines. Empties: q21 (GitHub REST, rate limit). Partial: q22 (two 401s).

## 2. Ranked signal list

Ranked by how much each one should change what we build. States: CONFIRMED (primary source read and the number is there), POST CLAIM ONLY (only the author's own post or README, no independent check), UNVERIFIED (not traced to a primary).

### S1. At batch size 1, WebGPU LLM speed is bound by dispatch count, not kernel quality
- who: authors of arXiv 2608.08730, "Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference"
- date: 2026-08-09 (v1 published); v2 read
- link: https://arxiv.org/abs/2608.08730
- primary checked: abstract via arXiv API; body via WebFetch of the HTML
- state: CONFIRMED (abstract); body numbers as extracted by the fetch tool
- verbatim (abstract): "We show that the dispatch overhead, not kernel quality, is the bottleneck at batch size 1, and isolate the dispatch count as the cause. Therefore, we conclude that at batch size 1, the effective approach to LLM inference optimization in WebGPU is reducing dispatch count."
- body, as returned: per-dispatch cost "24-36 µs" on Vulkan and "32-71 µs" on Metal; "A reduction of dispatches from 876 to 564, while keeping computationally the same WGSL shaders and with negligible memory savings, improves throughput by 53%" (Qwen2.5-0.5B, RTX 5090, Dawn/Vulkan).
- why it matters to us: our per-token graph has no attention and no per-layer stack, so its dispatch count can be far lower than a transformer's. That is a structural advantage on WebGPU only if the port keeps the dispatch count small.

### S2. Production browser builds quantise the tied vocabulary table to 4 bits with group-32 scales
- who: MLC (mlc-ai), Qwen3-0.6B-q4f16_1-MLC on the HF Hub
- date: read 2026-10-03 (repo files; upload date not read)
- link: https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC
- primary checked: tensor-cache.json and HF tree API
- state: CONFIRMED
- verbatim (manifest): `model.embed_tokens.q_weight [151936, 128] uint32` (77,791,232 bytes, alone in params_shard_0.bin) and `model.embed_tokens.q_scale [151936, 32] float16` (9,723,904 bytes). Qwen3-0.6B config: `vocab_size 151936, hidden_size 1024, tie_word_embeddings true`.
- reading: the tied 151,936 x 1024 table costs 87.5 MB in this build, 24.9% of the 351.5 MB download. The ONNX q4f16 export of the same model is 569.8 MB. The three ONNX graphs inspected (gemma-3-270m-it q4f16, LFM2.5-230M q4, LFM2-350M q4f16) contain the op `GatherBlockQuantized` for the embedding and nodes named `lm_head/MatMul_Quant` (string count, CONFIRMED that the names occur; storage layout UNVERIFIED).

### S3. Naive INT4 on a large output head costs real perplexity; INT4 storage of QAT-trained tables can be exact
- who (a): arXiv 2608.02703, ARCHead. date 2026-08-03. link https://arxiv.org/abs/2608.02703. primary: abstract. state CONFIRMED.
- verbatim (a): "practical backends often retain the final language-modeling head (LM-head) in BF16 or FP16. Quantizing this projection naively can strongly perturb the vocabulary-logit distribution." and "On Qwen3-8B-Base, it uses 25.6% of BF16 head storage while attaining 1.007 relative perplexity; storage-matched naive INT4 yields 1.14-1.16."
- who (b): xbill (Google Developer Experts), DEV Community, "Gemma 4 on a Tesla T4, Part 3". date September 30, 2026. link https://dev.to/gde/gemma-4-on-a-tesla-t4-part-3-int4-embeddings-serve-e2b-in-286-gib-at-230x-bf16-3kch. primary: the article. state CONFIRMED as the author's own measurement (server GPU, eight prompts, no perplexity).
- verbatim (b): "Every sampled group of 32 values in both tables sits on the same 4-bit grid as the linear layers" ; "The output layer also runs once per generated token over the whole 262,144-token vocabulary, so at bf16 it is one of the largest reads in every decode step." ; "All eight greedy test prompts produce the same tokens as the build with bf16 embeddings." Sizes: `embed_tokens` 0.750 GiB to 0.211 GiB; single 512-token request 85.28 tok/s against bf16 37.04.
- reading: ARCHead's evidence is an 8B model with a 4096-wide head; it does not show what happens at d 256-768. (b) works because the tables were trained on the grid. Together they say: int4 on our head is worth testing, and it is not safe to assume.

### S4. Cache API writes of single large weight files failed in one Chromium build; OPFS stored the same file
- who: OlehZhyhinas, LatexGen issue #70
- date: September 14, 2026
- link: https://github.com/OlehZhyhinas/LatexGen/issues/70
- primary checked: the issue
- state: CONFIRMED as one developer's report on "at least one Chromium build" (build not named in what was read). Not a browser vendor statement.
- verbatim: "UnknownError: Unexpected internal error" on `cache.add`; a prior attempt on September 5, 2026 "hit the same wall at 328 MB"; writing the 303 MB shard to OPFS took 10.4 s and reading it back 0.1 s.
- related, CONFIRMED from source: TVM's `dump_tensor_cache(..., shard_cap_mb=32, ...)` in apache/tvm `python/tvm/contrib/tvmjs.py`, which is why WebLLM weights arrive as about 32 MB `params_shard_*.bin` files. The tool summary of PR #72 says the failure starts "above roughly 250 MB"; that PR was not opened (UNVERIFIED).
- why it matters to us: our export is one file of about 107 MB (d 256) to 337 MB (d 768). The d 768 file sits in the size range where this report saw failures.

### S5. GPU-side argmax / top-k reads back 16 bytes instead of the logits vector
- who: Nehanth, pooled PR #98
- date: September 27, 2026
- link: https://github.com/Nehanth/pooled/pull/98
- primary checked: the PR
- state: CONFIRMED as the PR author's measurement; hardware not stated in what was read
- verbatim: "argmax or top-k runs in the same submit, so a greedy token reads back 16 B instead of the 1 MB logits vector" ; "multi-workgroup argmax and top-k (k <= 64) over the columns of a logits matrix" ; "value desc / index asc (the host greedy's tie rule), NaN and -Inf never picked". As returned (not a verbatim sentence): Chrome solo spec K=3 74.0 to 84.2 tok/s (+13.5%), plain decoding +5-7%.
- our arithmetic: 129,280 float32 logits are 517 KB per token.

### S6. WebGPU default limits are 128 MiB per storage binding and 256 MiB per buffer; more must be requested
- who: MDN, GPUSupportedLimits
- link: https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits
- state: CONFIRMED
- verbatim values: maxBufferSize 268435456 bytes; maxStorageBufferBindingSize 134217728 bytes; maxComputeWorkgroupStorageSize 16384; maxComputeInvocationsPerWorkgroup 256. Higher values are requested through `requiredLimits` in `GPUAdapter.requestDevice()`.
- our arithmetic: an int8 129,280 x d table is 99.3 MB at d 768 and 132.4 MB at d 1024, both under 134,217,728 bytes. An fp16 table at d 768 (198.6 MB) is over it.

### S7. Platform feature map for the kernels we would write
- subgroups: Chrome stable since Chrome 134. Chrome blog, February 26, 2025: "After a year of development and trials, the subgroups WebGPU feature enabling SIMD-level parallelism is now available." CONFIRMED.
- subgroups in Safari: Safari Technology Preview 249, Jul 29, 2026: "Added support for subgroups in WebGPU. (317145@main) (154874391)". CONFIRMED for Technology Preview only; no stable Safari page read confirms it.
- WebGPU availability: gpuweb Implementation Status wiki (last edited October 2, 2026, as returned): Firefox "✅ 141" on Windows, "✅ 147 on all macOS versions" on Apple Silicon, Linux "👷 Nightly", Android behind a flag; Safari "✅ 26" on macOS, iOS, iPadOS, visionOS. CONFIRMED.
- relaxed SIMD (int8 dot product in wasm): caniuse lists Chrome from 114, Firefox from 146, and Safari not supported through 27.1-TP, Safari on iOS not supported through 27.2. CONFIRMED.
- f16 accumulation caution: LlamaWeb (arXiv 2605.20706, 20 May 2026): "using f16 accumulation for the register-tiling kernel caused some models...to generate incoherent output on Apple M-series GPUs, so we currently use f32 accumulation". CONFIRMED (as returned by the fetch tool).
- browser gap: LlamaWeb: "on an Apple M3 the llama model with q4_k_m weights runs at ~1 tok/s on Firefox, but ~52 tok/s on Chrome". CONFIRMED, for their engine.

### S8. What small models ship at, measured on named hardware
- svenflow/webgpu-gemma README (custom WGSL engine, Q8_0): Gemma 3 270M "136.8 t/s" on a Mac Mini M4 Pro in Chrome 134 and "101.1 t/s" on an iPhone 17 Pro Max in Safari iOS 26; "~300MB download", "~500MB GPU memory"; claims 3.3x over transformers.js on 270M. POST CLAIM ONLY (author's README).
- Hugging Face blog, February 9, 2026: GPT-OSS 20B q4f16 "~60 tokens per second on an M4 Pro Max" with transformers.js v4. CONFIRMED that the blog says it; the hardware name is ambiguous as written.
- LFM2.5-230M at 1,400 tok/s in the browser: the claim appears in a search-engine rendering of an X post by Xenova (status 2070210622239707568, via search, X not fetched, engagement unknown) and in secondary blogs, which name an M4 Max. The Liquid AI blog (June 25, 2026) does not state it; it states native CPU decode "213 tok/s" on a Galaxy S25 Ultra and "42 tok/s" on a Raspberry Pi 5. POST CLAIM ONLY for the browser figure. LFM2.5-230M config: vocab 65,536, hidden 1024, tied (CONFIRMED).
- download sizes, HF API, CONFIRMED: gemma-3-270m-it q4f16 272.6 MB (fp32 1139.5 MB); LFM2.5-230M q4 211.1 MB, q8 489.3 MB; LFM2-350M q4f16 255.0 MB; SmolLM2-135M-Instruct q4f16 117.7 MB, int8 137.1 MB; Qwen3-0.6B q4f16 ONNX 569.8 MB, MLC q4f16_1 351.5 MB; gemma-4-E2B-it token-embedding graph q4f16 1590.7 MB against a 1519.7 MB q4f16 decoder.
- reading: the shipped sub-500M range is about 120 to 300 MB at 4 bits. Our 107 MB (d 256 int8) to 337 MB (d 768 int8) sits inside that range, and at large vocabularies the token table is the largest single object in these files.

### S9. Engine release state (npm registry, CONFIRMED)
- @huggingface/transformers 4.3.0, published 2026-09-16
- @mlc-ai/web-llm 0.2.85, published 2026-09-08
- onnxruntime-web 1.30.0, published 2026-09-14
- @wllama/wllama 3.8.1, published 2026-10-02
- no throughput claim is attached to these; they date the field.

### Lower-confidence items recorded and not ranked
- "WebGPU reached Baseline in January 2026" (q01 tool summary). UNVERIFIED.
- "WebLLM hitting 180 tok/s (Qwen 3.5 0.8B)" (q03, secondary blog, no hardware in the summary). UNVERIFIED.
- "0.6B model at q4f16 on WebGPU produces maybe 20 to 40 tokens per second on a laptop" (q07, a roadmap issue estimate, not a measurement). UNVERIFIED.
- wllama64 raising wasm memory to 16 GiB via Memory64 (q04). UNVERIFIED and not decision-relevant at our size.
- full-logits readback "costs 2-4 ms per token on Apple silicon" (q17 tool summary, source not identified). UNVERIFIED.

## 3. Proposed changes (at most 4)

### P1. Ship the export as about 32 MB shards plus a manifest, cached in OPFS
- source: S4 (LatexGen #70, Cache API failure at 303 MB and 328 MB on one Chromium build; OPFS write 10.4 s, read 0.1 s), and TVM's `shard_cap_mb=32` default (S4).
- change: split the one file into fixed shards with a JSON manifest (tensor name, shard, offset, nbytes, sha256), fetch shards in parallel, store them in OPFS, fall back to the Cache API only when OPFS is missing.
- test: in Chrome and Safari on the M5, load the d 768 export (337 MB) three ways (one file via Cache API, shards via Cache API, shards via OPFS); record success, cold load time and warm load time for each.

### P2. Try a 4-bit group-32 tied table, against the current int8 per-row table, on our own loss
- source: S2 (MLC stores the tied 151,936 x 1024 table as 4-bit with one fp16 scale per 32 values), S3 (ARCHead: naive INT4 head 1.14-1.16 relative perplexity on Qwen3-8B; the Gemma 4 article: int4 storage was exact because QAT put the tables on the grid).
- our arithmetic: at d 768, int8 per-row is 99.3 MB and int4 group-32 with fp16 scales is 55.8 MB; at d 256, 33.1 MB and 18.6 MB.
- test: export d 768 both ways, score held-out bits per byte and greedy-token agreement against the fp32 model on the same windows; adopt int4 only if the loss gap is inside the run-to-run noise of the training seeds; if it is not, try int4 for the input lookup with the int8 copy kept for the head, and report the file-size cost of the second copy.

### P3. When the WebGPU path is built, keep the whole token step in a few dispatches and finish with GPU argmax / top-k
- source: S1 (dispatch count dominates at batch 1; 876 to 564 dispatches gave +53%), S5 (GPU argmax reads back 16 B instead of the logits vector), S6 (128 MiB default binding; request higher limits).
- change: one dispatch for the back-token gather and moving-average update plus the linear map, one per memory read (or one for all reads), one for the head matvec, and a two-stage argmax / top-k (k up to 64) over the 129,280 logits; request `maxStorageBufferBindingSize` and `maxBufferSize` from the adapter, and keep each table at or under 134,217,728 bytes so the default still works. Keep f32 accumulation in the head (S7, LlamaWeb's f16-accumulation report on Apple M-series).
- test: on the M5 in Chrome, count dispatches per token and measure tok/s with GPU argmax against reading back all 517 KB of logits and doing argmax in JS; require identical greedy tokens to the CPU JS engine over a fixed 256-token prompt set.

### P4. Move the CPU head matvec from plain JS to WebAssembly SIMD, with relaxed SIMD where present
- source: S7 (relaxed SIMD int8 dot product shipped in Chrome 114+ and Firefox 146+, not in Safari through 27.1-TP and iOS 27.2, per caniuse); llama.cpp PR #19590 uses `wasm_i32x4_relaxed_dot_i8x16_i7x16_add` (title read only, UNVERIFIED beyond that).
- note: the relaxed int8 dot multiplies an 8-bit operand by a 7-bit one, so the activation side must be quantised to 0..127 or split; the plain SIMD128 path needs no such change and runs in Safari.
- test: microbenchmark one 129,280 x 768 int8 matvec (our head) in plain JS, wasm SIMD128 and wasm relaxed SIMD, in Chrome, Firefox and Safari on the M5; report ms per call and the max abs logit difference against the fp32 reference.
