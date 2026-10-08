# q20 arxiv-sparse-parametric-memory

- query: `abs:"parametric memory" AND (abs:sparse OR abs:lookup) AND abs:pretraining`
- tool: python3 urllib, arXiv Atom API
- url: http://export.arxiv.org/api/query?search_query=abs%3A%22parametric+memory%22+AND+%28abs%3Asparse+OR+abs%3Alookup%29+AND+abs%3Apretraining&sortBy=submittedDate&sortOrder=descending&max_results=40
- fetched_utc: 2026-10-03T10:48:27Z
- filter: published >= 2026-06-01 (results older than that are listed but marked OLD)
- entries returned: 1; in window: 0

## [OLD] Neural FOXP2 -- Language Specific Neuron Steering for Targeted Language Improvement in LLMs
- url: http://arxiv.org/abs/2602.00945v2
- published: 2026-02-01
- abstract (verbatim): LLMs are multilingual by training, yet their lingua franca is often English, reflecting English language dominance in pretraining. Other languages remain in parametric memory but are systematically suppressed. We argue that language defaultness is governed by a sparse, low-rank control circuit, language neurons, that can be mechanistically isolated and safely steered. We introduce Neural FOXP2, that makes a chosen language (Hindi or Spanish) primary in a model by steering language-specific neurons. Neural FOXP2 proceeds in three stages: (i) Localize: We train per-layer SAEs so each activation decomposes into a small set of active feature components. For every feature, we quantify English vs. Hindi/Spanish selectivity overall logit-mass lift toward the target-language token set. Tracing the top-ranked features back to their strongest contributing units yields a compact language-neuron set. (ii) Steering directions: We localize controllable language-shift geometry via a spectral low-rank analysis. For each layer, we build English to target activation-difference matrices and perform layerwise SVD to extract the dominant singular directions governing language change. The eigengap and effective-rank spectra identify a compact steering subspace and an empirically chosen intervention window (where these directions are strongest and most stable). (iii) Steer: We apply a signed, sparse activation shift targeted to the language neurons. Concretely, within low to mid layers we add a positive steering along the target-language dominant directions and a compensating negative shift toward the null space for the English neurons, yielding controllable target-language defaultness.

