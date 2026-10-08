# E q14 - Firefox WebGPU 2026 release

- query: `Firefox WebGPU 2026 release macOS Linux shader-f16`
- tool: WebSearch (three internal searches)
- fetched: 2026-10-03 10:55 UTC

## result list
1. Firefox (Wikipedia)
2. Firefox 147 Released: WebGPU Support ... (webpronews) - https://www.webpronews.com/firefox-147-released-webgpu-support-enhanced-security-and-more/
3. Floorp / Zen Browser / LibreWolf / Basilisk (Wikipedia)
4. Implementation Status (gpuweb wiki) - https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
5. WebGPU Hits Critical Mass: All Major Browsers Now Ship It - https://www.webgpu.com/news/webgpu-hits-critical-mass-all-major-browsers/
6. shader-f16 WebGPU support by browser and device (web3dsurvey) - https://web3dsurvey.com/webgpu/features/shader-f16
7. Review request for 'Intent to Ship: WebGPU f16 support' - https://groups.google.com/a/chromium.org/g/spec-mentors/c/uKGifWxE_zU
8. "shader-f16" requirements exclude all Qualcomm devices · Issue #5006 · gpuweb/gpuweb - https://github.com/gpuweb/gpuweb/issues/5006
9. Intent to Ship: WebGPU f16 support (blink-dev) - https://groups.google.com/a/chromium.org/g/blink-dev/c/AsKn-UwMYAE
10. WebGPU Inference: LLMs That Run in Your Browser (medium)
11. Half precision float (16 bit) is not enabled in Linux version for webgpu. [338730587] - Chromium - https://issues.chromium.org/issues/338730587
12. 9 WebGPU Tricks for Faster In-Browser ML (medium)
13. shader-f16 - WebGPU feature availability (95%) - webgpu.report - https://webgpu.report/features/shader-f16
14. Firefox To Get WebGPU Support Enabled By Default (winaero) - https://winaero.com/firefox-webgpu-support/
15. Firefox enables WebGPU ... Linux and Windows (ubunlog)

## tool summary claims (unverified unless checked)
- latest stable Firefox 156.0.1 (September 22, 2026).
- gpuweb issue 5006: on Vulkan, shader-f16 requires uniformAndStorageBuffer16BitAccess, which excluded every Qualcomm device in that author's list (issue is about 22 months old).

## primary check: gpuweb Implementation Status wiki (WebFetch 10:56 UTC)
- "Last edited: October 2, 2026" (as returned)
- Firefox Windows: "✅ 141"; macOS Apple Silicon: "✅ 145 on macOS 26+" and "✅ 147 on all macOS versions"; macOS other: Nightly; Linux: "👷 Nightly" (Mozilla expects to ship on Linux in 2026); Android: behind a flag.
- Safari: "✅ 26" on macOS Tahoe 26, iOS 26, iPadOS 26, visionOS 26.
- Chrome Android: "✅ ARM/Qualcomm/Intel, Android 12+: 121"; Chrome Linux: "✅ Intel Gen12+: 144", "✅ NVIDIA (driver 535.183.01+) Wayland: 147".
- state: CONFIRMED (the wiki is the working group's own tracker).
