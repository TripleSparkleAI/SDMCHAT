# LANE SDMPAGESTRUTH

Branch `settle-sdmpagestruth`, worktree `_worktrees/dwarfstar-sdmpagestruth`. Started 2026-10-05T11:52Z.

## The order (the navigator, corrected from fast typing)

"Can we make this cleaner on the SDM-relevant pages and the learn pages? Explain memory versus no memory like that,
but more simply. Cover the types of model and the things we try, and the scoring, and update all the results to the
latest. There might be old work there: update everything that needs it as you go. Use these ASCII-style diagrams, but
make them proper vector diagrams, beautiful and wonderful ones, and some animated. And good use of neon colour too, in
the headings: subtle, not heavy."

## Sources for every number

- `experiments/track4/sdmllm/PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` (THE LOG)
- `runs_vast_sdmonly/*/*.result.json`, field `val_test.bpb` (gitignored, read in the main checkout; every number below
  was checked against its json on 2026-10-05 before it was typed)

## Inventory: what is stale (checkpoint 1, 2026-10-05T12:00Z)

| page | file | what is stale |
|---|---|---|
| SDMCHAT `#/sdmchat` | `src/pages/SdmChat.jsx` | nothing at all on the big SDM-only runs: no memory on vs off, no model types, no scoring, no latest results. PLANNED · NOT RUN says "none has been run" with no pointer to the larger runs that did happen (a different model). |
| LEARN SDM `#/learn-sdm` | `src/pages/LearnSdm.jsx` | no bridge from Kanerva's memory to our language model, no memory on vs off |
| LEARN SDM UNFOLD `#/learn-sdm-unfold` | `src/pages/LearnSdmUnfold.jsx` | hops section: "Our model makes 4 such reads" is the 20M-token shape; the current shape adds a SwiGLU after every read, runs 16 to 64 hops, and the best scores have the reads taken out. WHAT IS NOT CLAIMED says the transformer "leads at every budget we ran": the transformer at 2.0B tokens has not landed |
| SDM EXPLORE `#/sdmexplore` | `src/sdmexplore/SparseCodes.jsx` | "at 20M tokens the trained reads added nothing": true at 20M, says nothing of the 2.0B test |
| SDMMEMORY `#/sdmmemory` | `src/pages/SdmMemory.jsx` | RECORDED · SDM LANGUAGE MODEL names only the chat model |
| RESULTS `#/results` | `src/pages/Results.jsx` | no pointer to the SDM-only results |

Checked and NOT stale: `src/chat/modelcards.js`, `src/sdmchat/useSdmEngine.js` (the browser chat family, SdmLM,
3,600 locations, default wide 768 chat-tuned 1.44235 on web text), `src/sdmexplore/HowSdmChat.jsx` and `recorded.js`
(the SDMLLM and SDMLLMSTORE reports, a different model), `src/pages/SdmGuy.jsx`, `src/pages/SdmStudio.jsx`,
`src/pages/Kanerva.jsx`, `src/science.js` (chart helpers, no claims).

## Plan

1. `src/sdmtruth/`: one shared kit, mounted on SDMCHAT (full) and LEARN SDM (the on/off explainer and a short
   results line): MEMORY ON, MEMORY OFF (two animated vector stacks), THE KINDS OF MODEL WE TRY, HOW WE SCORE,
   THE LATEST RESULTS. Every number in `src/sdmtruth/facts.js` with its run and log stamp; a test finds each in the log.
2. Fix the stale copy listed above.
3. Build, tests, screenshots, READY TO LAND.

## Status (checkpoint 2, 2026-10-05T12:45Z): DONE, ready to land

- [x] kit `sites/settle-site/src/sdmtruth/` (facts.js, stackMath.js, Stacks.jsx, Figures.jsx, Sections.jsx, sdmtruth.css)
- [x] SDMCHAT: four new sections after HOW IT WORKS; ABOUT links to two of them; PLANNED points at the latest results
- [x] LEARN SDM: TEN · FROM THIS MEMORY TO A LANGUAGE MODEL, the on/off explainer, the short results
- [x] stale copy: LEARN SDM UNFOLD (hops p2, new p4, claims p1, a top action), SDM EXPLORE (sparse l4), SDMMEMORY
      (the LLM lede), RESULTS (note2)
- [x] tests: `tests/sdmtruth.test.mjs` (11); npm test 3 failures, the same 3 as the clean base (sparse checkout:
      docs translations and a wiki path); vite build exit 0
- [x] i18n: 140 messages hand-translated into ja, zh, nl and hi (the Hermes route answered HTTP 404 upstream for
      every call, 2026-10-05T12:24Z); messages 0 missing in every locale
- [x] screenshots and frame sheets: `SETTLE/runs/sdmpagestruth/` (`shots.py --base http://localhost:<port>`)

## Numbers on the pages, all checked against the result jsons and the log

2.0B true runs, memory off: width 1,536 1.02241 · width 1,024 1.03599. Best 400M: 1.10483. Transformer at 800M:
1.02369. Twin check at step 5,000 (check set): on 1.16380, off 1.16468. Hops at width 1,024: 1.14585, 1.13013,
1.12273, 1.11834, 1.11213, 1.10841. Hops at width 1,536: 1.13335, 1.11767, 1.11022, 1.10483. Test set 995,323
tokens, 4,824,566 bytes. As of the log entry 2026-10-05T11:47Z.

## Owed after landing (not this lane's)

- When the memory-on twin, the transformer at 2.0B, 96 hops and the width 2,048 true run land, update the facts in
  `src/sdmtruth/facts.js` (values, runs, stamps, PENDING) and AS_OF; the test checks every value against the log.
- The memory-off true runs are called "true runs" in the log; the pages never call them SDM results.
