# E q15 - WebAssembly relaxed SIMD int8 dot product

- query: `WebAssembly relaxed SIMD matrix multiply dot product int8 browser support 2026`
- tool: WebSearch
- fetched: 2026-10-03 10:55 UTC

## result list
1. WASM Relaxed SIMD Enhancement by JeremyCEY · Pull Request #19590 · ggml-org/llama.cpp - https://github.com/ggml-org/llama.cpp/pull/19590
2. Relaxed Integer Dot Product instructions · Issue #52 · WebAssembly/relaxed-simd - https://github.com/WebAssembly/relaxed-simd/issues/52
3. Intent to Ship: WebAssembly Relaxed SIMD (blink-dev) - https://groups.google.com/a/chromium.org/g/blink-dev/c/HzLlEGLSx7E
4. Relaxed-width SIMD (WebAssembly) - caniuse - https://caniuse.com/wf-wasm-simd-relaxed
5. Using SIMD with WebAssembly - Emscripten docs - https://emscripten.org/docs/porting/simd.html
6. The State of WebAssembly in 2026: Beyond the Browser - DEV - https://dev.to/3ni8ma/the-state-of-webassembly-in-2026-beyond-the-browser-1j2p
7. GitHub - WebAssembly/relaxed-simd - https://github.com/WebAssembly/relaxed-simd
8. WebAssembly in 2026: SIMD, Threads, Wasm 3.0 ... (alldevtoolshub) - https://www.alldevtoolshub.com/blog/webassembly-browser-tools-2026-simd-threads-wasm-3/
9. WebAssembly and SIMD: A match made in the browser (medium) - https://robaboukhalil.medium.com/webassembly-and-simd-a-match-made-in-the-browser-7a7daa4f2ecd

## tool summary claims (unverified unless checked)
- `i32x4.dot_i8x16_i7x16_add_s`: second operand is 7-bit; the main building block for int8 matmul.
- llama.cpp PR #19590 (February 2026) uses `wasm_i32x4_relaxed_dot_i8x16_i7x16_add`.

## primary check: caniuse wf-wasm-simd-relaxed (WebFetch 10:56 UTC)
- Chrome supported from 114 (through 157 listed); Edge from 114; Firefox from 146 (through 160 listed); Safari "❌ Versions 3.1-26.6", "❌ Version 27", "❌ Versions 27.1-TP"; Safari on iOS not supported through 27.2. Global usage 77.59%.
- state: CONFIRMED (caniuse data). The PR #19590 content was not opened; UNVERIFIED beyond its title.
