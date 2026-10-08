# E q16 - wasm memory64 and the 4 GB limit

- query: `wasm memory64 shipped Chrome Firefox Safari 4GB limit 2026`
- tool: WebSearch (two internal searches)
- fetched: 2026-10-03 10:55 UTC

## result list
1. WASM 64 for WebGL - Unity Discussions - https://discussions.unity.com/t/wasm-64-for-webgl/1615192
2. Memory64: Unlocking WebAssembly's True Potential with 16GB of In-Browser Memory - SciChart - https://www.scichart.com/blog/memory64-unlocking-webassemblys-true-potential-with-16gb-of-in-browser-memory/
3. Bugzilla 1660420 - https://bugzilla.mozilla.org/show_bug.cgi?id=1660420
4. Wasm's Identity Crisis (RedMonk, 2025-10-17) - https://redmonk.com/kholterhoff/2025/10/17/wasms-identity-crisis/
5. The State of WebAssembly (platform.uno 2024-2025) - https://platform.uno/blog/state-of-webassembly-2024-2025/
6. Chrome, wasm32, 4GB (2GB?) limits, workarounds - Rust forum - https://users.rust-lang.org/t/chrome-wasm32-4gb-2gb-limits-workarounds/78161
7. State of WebAssembly 2026 (devnewsletter) - https://devnewsletter.com/p/state-of-webassembly-2026/
8. Up to 4GB of memory in WebAssembly · V8 - https://v8.dev/blog/4gb-wasm-memory
9. Intent to prototype: WebAssembly 64-bit memory model (mozilla dev-platform)
10. Memory64 (WebAssembly) - caniuse - https://caniuse.com/wf-wasm-memory64
11. WebKit bug 300538 - https://bugs.webkit.org/show_bug.cgi?id=300538
12. Release Notes for Safari Technology Preview 251 - https://webkit.org/blog/18194/release-notes-for-safari-technology-preview-251/
13. Release Notes for Safari Technology Preview 252 (daily.dev) - https://daily.dev/posts/release-notes-for-safari-technology-preview-252-n58mpd5gm
14. WebAssembly: Browser Support (testmuai)
15. wasm-bulk-memory Safari (lambdatest)
16. InfoQ Web Development

## tool summary claims (unverified)
- caniuse: Memory64 supported from Chrome 133; Safari not supported through 27.x stable.
- Safari Technology Preview 251 added WebAssembly Memory64.
- browser JS API caps Memory64 at 16 GB; Memory64 can cost speed due to bounds checks.

## primary check
Not opened individually; our export (107-337 MB) is far under the 4 GB wasm32 limit, so this signal does not drive a change. State UNVERIFIED for version numbers; recorded for completeness.
