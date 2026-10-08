# Ideas for after the base run (SDM-only language model)

Ideas raised while the SDM-only base run trained, kept for later. None of these is running. Each line says what
it is, why it might matter, and what we already know. Numbers are TEST bits per byte on the 995,323-token test set,
lower is better.

## Bigger models

- **A card with more memory (H200, B200 or similar).** On a 32 GB RTX 5090 the store-on model does not fit at width
  2,048 (out of memory even at 2 sequences per micro-batch), and store-on runs above about 32 hops at width 1,024 do
  not fit (each hop holds a store-sized buffer, 65,536 x width floats). A 141 GB card would allow width 2,048 with the
  store on and 64 or more hops. Memory-off width 2,048 scored 1.01665 against 1.02241 at width 1,536 (2.0B tokens).
- **More hops.** Hops are UNBRACKETED at width 1,024 (memory off, 400M tokens): 64 hops 1.10841, 96 hops 1.10518.
  A 128-hop run is in progress. The gain per step is shrinking.
- **More data.** The data sweep is UNBRACKETED. The Spark holds 3.0B tokens of train_big; the rented boxes hold the
  first 2.0B. A 1.6B-token run at width 1,024 is in progress against 1.09026 at 800M.
- **Extend the base run.** The base run keeps a checkpoint at 1.6B tokens, before its decay. A cooldown fork from
  there could continue on the Spark's 3.0B tokens and decay at the end.
- **A second seed of width 2,048 (memory off).** Width 2,048 beat width 1,536 by 0.0058, inside the 0.007 noise, and
  no more data exists at that budget, so a second seed is the way to settle it.

## Making the memory earn its place

Every store variant tied memory off: default, 64 locations awake, 10x store learning rate, one store per hop, and the
store at window 1,024. Letting the query's gradient reach the features made training diverge (+0.037, +0.047).

- **A stable run-time memory (the mixer).** This is the law's own mechanism, the one that could give a long window.
  On the wide body it led early, then fell behind (+0.038 with fading heads, +0.020 with Muon weight decay) and its
  gradient norm climbed. Untried fixes: gradient clipping on the mixer alone, a lower mixer learning rate, a norm on
  the mixer's output, fewer mixer layers, or the mixer only in the last hops.
- **A learned address without blowing up.** The query gradient diverges when it flows into the features. Untried:
  a separate small loss for the address, normalised queries, or a gradient scale well below 1.
- **Tests where memory should pay.** Next-token loss on web text may simply not need the store. A recall-heavy
  evaluation (facts or names seen once, needed later in the window) could show a difference that bits per byte hides.
- **The window on its own.** A longer training window helps with memory off (256: 1.13013, 2,048: 1.11823), because
  the five fading averages see further back. Bracketed at 2,048 (4,096: 1.11959, 8,192: 1.12165).

## Speed and tooling

- **Multi-GPU scaling is only 1.3x.** Two RTX 5090s give 24.0k tokens/s against 18.5k on one (width 1,536, store on).
  A likely cause is that each rank runs the whole Muon step; a Muon sharded across ranks could help.
- **torch.compile is never used by the box arm wrapper.** A compiled single-GPU run has not been measured.
- **The failover copy is slow.** Box-to-Spark links measured 0.9 to 3.6 MB/s, so a 9.9 GB checkpoint takes 1 to 3 hours.
  A bf16 copy of the weights, or a model-only copy, would move faster; a resume without optimiser state loses the
  Muon momentum.
- **The Spark copier's labels.** It names a copy by the run log's latest step, not the checkpoint's own step, and an
  ssh refusal used to read as "no checkpoint". Refusals are now retried and logged.

## After the base model

- **The chat model.** Chat tuning comes after the base run, on the base run's weights.
- **The site.** The SDM pages show results up to the 2.0B no-memory runs; the base run and its twin need adding when
  they finish.

## Does the SDM catch up later? (the navigator, 2026-10-06)

- **The idea:** the SDM may learn more slowly early and catch up later, a difference in when it "groks" rather than in
  what it can reach. A transformer looks straight back at any earlier token; the SDM sees the past through the last 8
  tokens, 5 fading averages and its memory reads, a summary that may need more data before it is used well.
- **What we already know** (width 1,536, TEST bits per byte, lower is better): the gap to the transformer was 0.156 in
  the early waves at 400M, 0.046 at 400M after the shape hunt, 0.053 at 800M and 0.032 at 2.0B. The points are not one
  controlled curve: the hop MLP grew from 3,072 to 6,144 between 800M and 2.0B.
- **How to test it:** one fixed shape, memory on and off, and the transformer, each at 400M, 800M, 2.0B and 3.0B, so the
  gap is read along a clean data axis. A gap that keeps shrinking supports a late catch-up; a constant gap does not.
  The cheapest first step is the base run's kept 1.6B checkpoint extended toward the Spark's 3.0B tokens, beside a
  transformer at 3.0B.

## Our lookback, and ideas from Mamba (the navigator, 2026-10-06)

- **What the model sees of the past today:** the last 8 tokens exactly, plus 5 fading averages with fixed decays 0.5,
  0.8, 0.9, 0.97 and 0.99 (half-lives of about 1, 3, 7, 23 and 69 tokens). Beyond a few hundred tokens nothing reaches
  the current position. The training window (256 or 2,048) sets the example length, not the reach. Measured (memory off, width 1,024, 400M): window 256 1.13013,
  1,024 1.11881, 2,048 1.11823, 4,096 1.11959. Longer helps up to about 1,024, then flat: at 256 many positions sit
  near the start of their example, before the fading averages fill. The learned SDM store holds facts from training; the text being read is never written to it.
- **The transformer yardstick is short too:** `w54_mq_d1536_B512_2B` trains at window 256 with 4 layers. A long-window
  transformer has never been in the comparison.
- **The match with Mamba:** a fading average is a recurrence (each step's summary is the last one faded, plus the new
  token), the core of Mamba and other state-space models. Ours has fixed decays; Mamba's forgetting depends on the
  input. So today's model is a state-space model without selectivity, plus a learned memory of facts.
- **Ideas, in rough order of cost:**
  1. Learned decays, many of them: let each channel learn its own fade rate, spread over many timescales, instead of
     five fixed ones.
  2. Selective fade (Mamba's key step): the fade rate at each token is computed from the token, so the state can keep
     an important word and drop filler. Trains with a parallel scan.
  3. Selective writing into the run-time Kanerva memory (the mixer): a learned gate decides how strongly each token
     writes and how fast each memory slot decays. This is Mamba's selectivity on an SDM state, and a candidate fix for
     the mixer's unstable gradients.
  4. A long-state query: build the SDM read address from the selective state, so memory reads see beyond 100 tokens.
  5. A reset at document boundaries, so one document's state does not leak into the next.
- **The law question for the navigator:** the law says every step that mixes positions must be an SDM mechanism. The
  fixed fading averages are already in the base. Ideas 1, 2 and 5 are recurrences rather than Kanerva memories, so
  they need his ruling; ideas 3 and 4 keep the mixing inside the SDM.
- **The fair long-context test:** our model against a transformer at window 2,048 (and longer), on text that needs
  long memory.

- **Timing (the navigator):** none of this touches the running base. These are for future runs.

## The honest shape of the model, and where the SDM should go next (the navigator, 2026-10-06)

- **What the base is today:** an MLP language model over a short lookback (8 exact tokens plus 5 fading averages, a
  fixed recurrence), with a learned sparse memory read at every hop. The navigator's words: "essentially an MLP with a
  ride-along SDM".
- **What the memory is today:** a store written only by training (65,536 locations, product-key addressing, top 32
  read). It holds what training put there; the text being read is never written into it. That makes it a sparse
  key-value layer (close to the product-key memory layers of the literature) more than a memory of the text. Every test
  so far has it tied or a hair either side of memory off, so its 100M parameters add little on this test.
- **What the law asks for and the base does not do yet:** "a Kanerva memory written at run time and read", with a fixed
  number of locations, so a token's cost does not grow with the window. That is the run-time memory (the mixer), the
  only part that could replace the fading averages as the model's sense of the past and reach a long window. It led
  early, then fell behind with growing gradients, so it is not in the base.
- **The rework, as the next campaign after the base, chat and Guy are done:**
  1. Make the run-time SDM the lookback: every token writes into a fixed Kanerva memory, every token reads it; the
     fading averages become a fallback or go.
  2. Stabilise it with the untried fixes: gradient clipping on the memory alone, a lower learning rate for it, a norm on
     its output, fewer memory layers, and selective (gated) writing, which is Mamba's idea applied to an SDM state.
  3. Test it where memory should pay: a recall test (a fact seen once, needed hundreds or thousands of tokens later)
     and a long-window comparison against a transformer at window 2,048 and longer.
  4. Keep the learned store only if it earns its place in those tests.

## New definitions, and the one-SDM shape (the navigator, 2026-10-06)

- **THE COMPLICATE** is now simply the model's vector space: its hidden state, where meaning lives. **ENFOLD** is a token
  going into the hidden state (token to vector); **UNFOLD** is the hidden state choosing the next token (vector to
  token). These replace the earlier meanings; they have nothing to do with the learned knowledge store.
- **The shape the navigator wants:** the whole hidden state is ONE SDM. Every token is enfolded by writing it into that
  one fixed-size Kanerva memory; each layer is a pass that reads the memory, thinks and writes back; the next token is
  unfolded by reading it. No separate lookback, no growing per-token cache, a fixed cost per token at any length. The
  open question is capacity: how much text one SDM holds before overlapping writes blur, which is what the recall
  puzzles measure.
- **Closest past attempts:** the mixer (a run-time Kanerva memory written while reading, beside the fading averages: it
  led early, then fell behind with climbing gradients); and the earlier one-shot VSA generator, where the readout from
  the folded memory was the weak point.

## DECIDED: the middle path (the navigator, 2026-10-06)

- Build the PURE run-time SDM model first: a standard residual stack where, in every layer, the residual stream writes a
  learned value at a learned key into a fixed-size Kanerva memory and reads at a learned query, the read added back
  (add, never overwrite). Soft addressing, values started at zero, normalised queries, a small learning rate and gated
  writes for the memory, chunked training with the memory carried across chunks. Fully differentiable, fixed memory,
  fixed cost per token.
- Add, as a separate component switched OFF by default, the DeepSeek-style ARCHIVE: each token's compressed exact
  vector kept in plain RAM, its position written into an SDM as a pointer, a few exact fetches per token, plus a
  recent window read exactly. Sharp far recall for a store that grows about 100 times slower than a transformer's cache;
  the open problem is training the pointer choice without an attention teacher.
- Measure both on the recall puzzles (recall at distance, copy the repeat, blurry cue) and on cost per token, against
  today's model and a tiny transformer yardstick, before any full-size run.
