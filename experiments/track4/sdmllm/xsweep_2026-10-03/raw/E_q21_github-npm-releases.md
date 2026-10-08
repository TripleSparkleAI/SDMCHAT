# E q21 - GitHub REST releases (FAILED) and q21b npm registry (fallback)

## q21 GitHub REST API, unauthenticated, /repos/{repo}/releases?per_page=6
- result: every call returned the error below (the unauthenticated hourly quota was already spent on this host). EMPTY.
```
fetched 2026-10-03T10:54:47.005512Z
== huggingface/transformers.js ERROR <HTTPError 403: 'rate limit exceeded'>
== mlc-ai/web-llm ERROR <HTTPError 403: 'rate limit exceeded'>
== microsoft/onnxruntime ERROR <HTTPError 403: 'rate limit exceeded'>
== ngxson/wllama ERROR <HTTPError 403: 'rate limit exceeded'>
== ggml-org/llama.cpp ERROR <HTTPError 403: 'rate limit exceeded'>
```

## q21b npm registry API (registry.npmjs.org), python3 urllib
```
fetched 2026-10-03T10:54:56.792420Z
== @huggingface/transformers dist-tags: {'next': ('4.0.0-next.11', '2026-03-30'), 'latest': ('4.3.0', '2026-09-16')}
   versions since 2026-06-01: 1 ['4.3.0@2026-09-16']
== @mlc-ai/web-llm dist-tags: {'latest': ('0.2.85', '2026-09-08')}
   versions since 2026-06-01: 1 ['0.2.85@2026-09-08']
== onnxruntime-web dist-tags: {'extensions': ('1.9.0-extensions', '2021-09-15'), 'latest': ('1.30.0', '2026-09-14'), 'dev': ('1.31.0-dev.20260918-bc8e7ed75', '2026-09-18')}
   versions since 2026-06-01: 25 ['1.30.0-dev.20260910-f2c39fe2f@2026-09-10', '1.31.0-dev.20260911-2a43ec07e@2026-09-11', '1.31.0-dev.20260914-8d85527a0@2026-09-14', '1.30.0@2026-09-14', '1.31.0-dev.20260915-fed7f01a0@2026-09-15', '1.31.0-dev.20260917-bff4fcf7d@2026-09-17', '1.31.0-dev.20260916-2a720f925@2026-09-17', '1.31.0-dev.20260918-bc8e7ed75@2026-09-18']
== @wllama/wllama dist-tags: {'latest': ('3.8.1', '2026-10-02')}
   versions since 2026-06-01: 6 ['3.5.0@2026-06-15', '3.5.1@2026-06-15', '3.6.0@2026-08-16', '3.6.1@2026-08-27', '3.7.0@2026-10-02', '3.8.1@2026-10-02']
```

- state: CONFIRMED (npm registry publish times are primary). transformers.js latest 4.3.0 published 2026-09-16; web-llm 0.2.85 published 2026-09-08; onnxruntime-web 1.30.0 published 2026-09-14; wllama 3.8.1 published 2026-10-02.
