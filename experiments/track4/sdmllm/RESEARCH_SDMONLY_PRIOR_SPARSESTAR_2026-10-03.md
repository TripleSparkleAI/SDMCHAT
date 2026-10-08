# What the SPARSESTAR record says about the SDM-only model's read and write

This document reads the repo's own SPARSESTAR record (OAKENFOLD, THE COMPLICATE, the 4.x quadrants, the char-level
store series, the 4.10 BDH study, the SPARSESTAR distilled track, the SDR band, WIKI_SDR and WIKI_PICBREEDER) and
asks one question: what would make the read and the write ("unfold" and "enfold") of the SDM-only language model in
`track4_sdmonly_models.py` as good as they can be, for an autoregressive model trained from scratch by next-token
prediction, with no teacher and no distillation.

**Labels.** Every number carries one of these:
- **MEASURED** - a number from our own record, with its path.
- **UNVERIFIED** - anything else: arithmetic from the code that nobody has measured, an extrapolation, a paper's
  claim we have not reproduced, or a reasoned expectation.

Distilled results appear below as evidence about mechanisms only. None of them is a recipe for this model, because
the standing ruling for this work is no teacher and no distillation.

---

## 0. The short answer

Three additions earn one arm each:

1. **A run-time write into the woken rows (the missing ENFOLD).** At each position, add the next token's embedding
   into the few locations the address woke, inside the current window; later positions whose address wakes the same
   locations read it back. It is the one candidate that lets the model see past its fixed window of 8 back tokens
   and 5 averages, and our record says that window is the limit.
2. **Bound (conjunction) features in the address.** Add `e_t ⊙ e_t-1` and `e_t ⊙ e_t-1 ⊙ e_t-2` as two more context
   features. The address is the one place with measured wins, and the body has no multiplicative term at all.
3. **An orthogonality pressure on the representation.** Penalise the off-diagonal Gram energy of normalised hidden
   vectors. It is the only factorisation pressure in our record that was trained from scratch on next-character
   loss and improved that loss.

The rest of this document is the evidence for and against each, the steps of the cycle that stay absent, and what
may not be claimed.

---

## 1. The enfold / unfold / fold-back cycle beside the SDM-only model

The cycle as defined in `experiments/track4/sparsestar-proper/THE_COMPLICATE_our-enfolded-order.md`:

    C = Σ_i bind(addr_i, val_i)
    ENFOLD:     C += bind(addr, val)
    UNFOLD:     addr -> unbind -> SDM cleanup -> argmax  (the EXPLICATE, the emitted token)
    FOLD-BACK:  the review verdict re-enfolds into C (confidence update)

**In words:** facts are bound to their addresses and summed into one state; to generate, the state is unbound with
an address, the noisy result is snapped to the nearest codebook item, and the item is emitted; a verdict on the
emission is written back.

OAKENFOLD adds "a TREE of named SDMs" (`OAKENFOLD_the-generative-structure.md`, "What OAKENFOLD names").

The SDM-only model, read from `track4_sdmonly_models.py` (`features`, `hidden`, `forward`, `ProductKeyStore.forward`):

```
   STEP OF THE CYCLE            DEFINED AS                       SDM-ONLY MODEL                      STATE
   ──────────────────────────── ──────────────────────────────── ─────────────────────────────────── ───────
   address from context         addr from the context            x = W_x [8 back embeddings ;        PRESENT
                                                                  5 moving averages], q = BN(W_q x)
   binding (role ⊛ filler)      bind(addr, val), invertible      none. Position enters by            ABSENT
                                                                  concatenation (a direct sum), no ⊛
   superposed state C           one vector holds every fact      x is a sum of hop reads; the 5      PARTIAL
                                                                  averages are decaying unbound
                                                                  bundles of past embeddings
   write at run time (ENFOLD)   C += bind(addr, val)             none. Keys and values are learned   ABSENT
                                                                  by gradient and frozen after
   unbind                       addr -> unbind                   none                                ABSENT
   locate / read                SDM read from the address        product-key top-k, soft cut,        PRESENT
                                                                  (1/k) Σ w_m v_m, added to x
   cleanup against a codebook   nearest codebook atom            only at the end: logits =           END ONLY
     between steps                                                norm(x) E^T against the token table
   iterate                      read again from the result       4 hops, a separate store per hop    PRESENT
   emit and roll back in        token joins the context          next token joins the back tokens    PRESENT
   fold-back of a verdict       verdict re-enfolds               none                                ABSENT
   tree of named nodes          a tree of separate named SDMs    flat: one store per hop, unnamed    ABSENT
```

Two of these readings are structural and need stating plainly:

- **The model already has a bundle and a cleanup.** Each hop adds `(1/k) Σ w_m v_m` into x, so x is a weighted sum of
  value rows: a bundle with the addresses left implicit. The tied head scores `norm(x)` against every token
  embedding, which is the codebook cleanup that `wikis/WIKI_SDR/wiki/160-cleanup-as-the-readout-generative-decoding-and-knowing-it-failed.md`
  calls the readout ("cleanup IS the readout"). What the model lacks is binding, unbinding, a run-time write and
  fold-back.
- **The moving averages are a superposed run-time state.** Each `a_t(β)` is a decaying sum of past embeddings
  (`TRAINING_SDMONLY_HOW_WE_TRAIN.md` §2.1). It is a superposition with no roles, read linearly through `W_x`.

`experiments/track4/HISTORY_SDM_AND_SPARSESTAR_2026-10-04.md` §13 already gives the verdict for the earlier `SdmLM`:
"It never binds and never unbinds. Its outer loop has the unfold's shape, and that is all it shares." The same holds
for the SDM-only model.

---

## 2. The absent steps, one at a time

### 2.1 The write at run time (ENFOLD)

**What we measured before.**

| finding | number | label and path |
|---|---|---|
| the SDM family's limit is its context, not its store | three context-feature arms tie within 0.0004 bpb at 300M tokens, d 256; a 4-layer transformer on the same tokens scores 1.35223 against 1.50641 to 1.50680 | MEASURED, ledger via `experiments/track4/HISTORY_SDM_AND_SPARSESTAR_2026-10-04.md` §10b |
| the wider context (8 back, 5 averages) was the first change that moved the family | W 1.49853 against U 1.50680 | MEASURED, same §10b |
| the gap that remains to attention at d 256, 300M | 1.49853 − 1.35223 = 0.14630 bpb | UNVERIFIED (difference of two MEASURED numbers from runs not paired) |
| at 20M tokens the transformer's whole lead over no store is small | transformer 1.58257 (5 seeds), no store 1.6055 (3 seeds) | MEASURED, `REPORT_SDMLLMSTORE.md` §5 and `REPORT_SDMLLM_S0.md`, via HISTORY §10a |
| a BDH-style Hebbian working memory (causal outer-product state over sparse-positive addresses) trains from random init and reaches the behaviour band 10 of 10 seeds | quotient C_1 +0.300, C_2 +0.116, causal locality +0.058, all 10 of 10 seeds, basis moved not at all (0 of 10 beat their rotation null) | MEASURED, `experiments/track4/sparsestar-proper/4.10-pathway-bdh-cq/E5_MEASURED_the-hebbian-arm-wins-the-quotient-and-moves-no-basis.md` §④ |
| a residual (delta-rule) write beats an additive write in a crosstalk-bound char SDM | −0.189 bpc (3.4578 to 3.3557) | MEASURED, `THINKING_r6-delta-rule-is-deltanet_2026-07-21.md` (commit b216dbb53 cited there) |
| the same, three configurations | −0.569, −0.342, −0.385 bpc; top-1 recall +0.12 to +0.18 | MEASURED, `track4_61_delta_rule_writes_FINDINGS.md` |
| the opposite sign in a well-resourced store at matched bytes | delta +0.118 bpc worse | MEASURED, node `DeltaRuleWriteNull`, quoted in `wikis/WIKI_SDR/wiki/153-sdm-write-rules-and-readout-confidence-track6-measured.md` §1 |
| a flat (no-decay) superposed run-time state wins below capacity; decay wins above it | crossover M* in (128, 256] items at D 1024, λ 0.98; decay ahead in 10 of 10 seeds at M 512 to 2048 | MEASURED, `4.10-pathway-bdh-cq/E3_MEASURED_decay-vs-tail-law_the-crossover-at-M-between-128-and-256.md` §① |
| surprise-gated retention on a saturated SDM | surprise-only 0.856 against naive replay 0.440; saturation cliff pushed about 5× (280 to 1,400 writes) | MEASURED (toy SDM, seed D72), `experiments/track4/sparsestar-distilled/SPARSESTAR_PLAN.md` "THE HARD FENCES ON ONLINE LEARNING" |

The working rule the record settled on for write rules (`wikis/WIKI_SDR/wiki/153-*.md` §1): "additive when
under-loaded, delta when crosstalk-bound."

The 4.10 study already names the link between a sparse write and attention. E5's state is
`scores = tril(a @ a.T, -1)`, "linear attention, Q ≡ K, strictly causal", over a sparse-positive address `a`
(`E5_PREREG_hebbian-fourth-arm.md` §②). The SDM-only model already computes a sparse code at every position: the
soft weights `w_m` over the woken locations. So a Hebbian write into the woken rows **is** that linear attention,
with the location code as both key and query.

**The arm, stated as equations.** Use one hop (the last), and keep only the 8 strongest woken locations as the
write set `W8(s)`. For a target vector `u_s` and positions `s < t` in the same window:

    ΔV_m(t) = η · Σ_{s<t, m in W8(s)} λ^(t−1−s) · w_m(s) · u_s
    r_fast(t) = (g / k) · Σ_{m in W(t)} w_m(t) · ΔV_m(t)
    x <- x + r_fast(t)

**In words:** every earlier position in the window adds its target into the few locations it woke, slightly
faded by distance; the current position reads those locations with its own wake weights and adds the result to
its working vector. `g` starts at zero, so at step 0 the arm is exactly the centre model.

The same read in its kernel form, which is how it trains in one batched pass:

    r_fast(t) = (g η / k) · Σ_{s<t} λ^(t−1−s) · K(t, s) · u_s      with  K(t, s) = Σ_m w_m(t) · w_m(s)

**In words:** position t attends to an earlier position s exactly as much as their woken location sets overlap.
Two positions that wake no common location do not see each other at all.

Two target choices, one arm each if a slot is free:
- **`u_s = E[y_s+1]`, the next token's embedding.** At t the store returns "what followed this address earlier in
  the window", and the tied head gives that token mass. This is a copy (induction) mechanism. It is causal: `s < t`
  means `y_s+1` is at or before position t.
- **`u_s = x_s`, the hidden vector.** This is E5's choice (the value is the dense residual).

**Is a from-scratch gradient-trained version plausible?** Yes, on the record. E5 trained the outer-product
version from random init in all 10 seeds. The write parameters (`g`, `η`, `λ`) and the target embedding all
receive gradient; the location choice stays the only non-linear step.

**The cheapest honest test.** One arm against the centre `p0_A_c`, with a content control that keeps the write's
norm and destroys its meaning: the targets `u_s` permuted at random across positions within each window. A gain
that survives the permutation is a norm or regularisation effect and not a memory.

**Evidence for.**
- The record's own diagnosis says the family is context-limited (HISTORY §10b), and this is the only candidate in
  this document that reaches tokens outside the fixed window.
- E5: the sparse outer-product state trains from scratch at tiny scale.
- The write is per window and sparse: 256 positions × 8 locations into 65,536 locations per hop. That is far below
  any capacity limit the record measured (UNVERIFIED arithmetic), so the additive rule is the one the record's
  working rule picks.
- For the browser: per token it costs 8 row updates and 32 row reads per hop on a sparse table that lives for one
  conversation (UNVERIFIED arithmetic). No attention matrix is needed at inference.

**Evidence against.**
- E5 matched behaviour on purpose, so it has **no loss number**. Its gain was on quotient geometry, at 2.45× the
  parameters of its control, on a synthetic RHM task, under a distillation objective (E5 §⑥(a)). It shows the
  mechanism trains; it does not show it lowers bpb.
- At 20M tokens the whole lead of a real attention model over no store is about 0.023 bpb (S0 numbers above). A
  write that recovers part of that may sit below the push's bar of 0.003 bpb at z ≥ 3. A positive at 20M needs a
  100M confirm; a null at 20M is UNDERPOWERED, not negative.
- The kernel `K(t, s)` exists only where addresses repeat inside a window. If addresses rarely repeat, the arm
  reads nothing. The location audit in §5 (test 7) measures this before anyone over-reads a null.
- 245's lesson, in `THE_POINT_*_2026-08-03.md` §④: "The readout was the ballgame"; a trained dense path may absorb what the
  fast read offers.
- In the 4.10 framing this is a **transient** working-state mechanism (`THE_POINT_*_2026-08-03.md`, the
  2026-08-14 addendum). A win here says nothing about the durable store.

### 2.2 Binding

**What we measured before.**

| finding | number | label and path |
|---|---|---|
| a free token mash address (`Σ R[tok_i] ⊙ P_i`, bipolar codebooks, Hadamard bind with position roles) costs most of the store's gain when it replaces a learned address | 1.319804 to 2.035848 bpb; 72.7% of the gain lost | MEASURED, `MEASURED_track4_239_the-teacher-free-address-costs-0.716-bpb-*.md` §0, §2 |
| in that bundle, more bound context was worse | n 4 beat n 8 on CALIB, 2.081495 against 2.183468 | MEASURED, same §5 |
| bind against permute as the order code in the 6.1 VSA generator | 3.671 against 3.686 bpc, "a wash" | MEASURED, `track4_cross-verify-nanogpt-vs-bohm-penti_FINDINGS.md` |
| position binding localised damage under 2-char context corruption | top-1 0.32 against permutation's 0.17 | MEASURED (starter), `track4_core_bohm-penti-substrate_THINKING.md` §5 |
| chained unbind depth | d* = log2(D) − 5 | MEASURED for D 1k to 16k, `STEP1_MASTERY_QUEUE.md` BU-1 |
| superposition capacity, one unbind and cleanup | M* ≈ D/32 | MEASURED for D ≥ 1k, `STEP1_MASTERY_QUEUE.md` BU-2 |
| a parallelogram (bind-shaped) substitution offset over named fields | hit@1 0.681 against 0.0466 for a random offset | MEASURED, `STEP1_MASTERY_QUEUE.md` W3c-D1 |
| the address is starved: a 3-gram key holds 1.18 facts; a min-support backoff address gives 6.72× the incumbent's margin over the floor | as stated | MEASURED, `WIKI/theory/173-the-address-is-starved-not-misconceived-2026-09-01.md` |
| Engram-style multi-head hashing for n = 2 and 3 is owed | owed item 3 | `REPORT_SDMLLMSTORE.md` §10 |

**The exact point about token-and-position binding.** The model concatenates 8 back-token vectors and maps them by
one matrix `W_x`. Concatenation is binding with one-hot roles: each token sits in its own block. A compressed bind
`Σ_j P_j ⊙ e_t−j` with fixed ±1 roles is a linear function of that concatenation, so a learned `W_x` can already
express it (UNVERIFIED reasoning, from the shapes). **A token-and-position bound address therefore cannot add
expressive power to this model.** 239 is consistent with that: its bound address lost to a learned one by 0.716.

**What binding can add here is a product of two tokens.** The body is linear in its features except for the
choice of locations (`TRAINING_SDMONLY_HOW_WE_TRAIN.md` §2.4). A Hadamard bind of two learned embeddings,
`e_t ⊙ e_t−1`, is a bigram conjunction that no linear map of the concatenation can form.

**The arm.** Two more features, each RMS-normalised like the others:

    b2_t = norm( e_t ⊙ e_t−1 )        b3_t = norm( e_t ⊙ e_t−1 ⊙ e_t−2 )
    x = W_x [ e_t ; ... ; e_t−7 ; a_t(0.5) ; ... ; a_t(0.99) ; b2_t ; b3_t ]

**In words:** the address now also sees "this token next to that token" as one feature, built by multiplying the
two embeddings element by element.

**Plausible from scratch?** Yes. It is two element-wise products and a wider `W_x`; the extra cost is
2 · 2 · d · d = 262,144 FLOPs a token at d 256, against a total of about 69.4M (UNVERIFIED arithmetic from
`flops_per_token`). It exports to the browser as two multiplies per feature.

**The control.** `n_back 10`: the same feature count (15 blocks) and the same `W_x` width, with two more plain back
tokens instead of the two products. This separates "more features" from "multiplicative features".

**Evidence for.** The address is where every measured win sits (173; W over U in HISTORY §10b). Engram's own
address is a hashed n-gram, which is a discrete conjunction (`REPORT_SDMLLMSTORE.md` §10 item 3 asks for it).

**Evidence against.** 239: n 4 beat n 8 in an unweighted bundle, which is crosstalk arriving by itself. The S0
EXACT5 count store, a discrete n-gram memory, scored 1.6969 against no store's 1.6055 at 20M (MEASURED, HISTORY
§10a). The `BU-1` and `BU-2` laws were measured at D ≥ 1k; at d 256 they give d* = 3 and M* ≈ 8 (UNVERIFIED
extrapolation), so any binding scheme that superposes many pairs into one d 256 vector is capacity-starved. The
proposed arm superposes nothing; it adds two features.

### 2.3 A superposed state that persists and can be unbound

**What exists.** The 5 moving averages persist across positions and superpose past embeddings without roles.
`STEP1_MASTERY_QUEUE.md` A5 measured a "remembering state" against a window as an address and found it did not
help (MEASURED: best window+EMA blend 2.6939 against the window's 2.6785). SPARK24 then measured the wider address
that includes 5 averages as the first change that moved the family (MEASURED, HISTORY §10b). The two are different
models and different measurands; they are cited side by side and not reconciled here.

**What is absent.** A persisting state that can be unbound with the current token. The VSA form is
`C_t = λ C_t−1 + bind(e_t−1, e_t)` read by `unbind(C_t, e_t)`. The outer-product form is a decaying fast-weight
matrix:

    M_t = λ · M_t−1 + φ(e_t−1) ⊗ e_t        read:  r_t = M_t^T φ(e_t)

**In words:** the state remembers which token followed which, faded by λ, and the current token asks it "what
followed me before".

**Evidence against the VSA form at d 256.** BU-2 gives M* ≈ D/32; at d 256 that is about 8 pairs (UNVERIFIED
extrapolation below the measured range). E3 found decay necessary above about 128 to 256 items at D 1024
(MEASURED). A 256-token window writes 255 pairs. The outer-product form holds about d pairs before interference
grows (UNVERIFIED, standard capacity of a d × d associative matrix, not measured here).

**Relation to 2.1.** The sparse write in 2.1 is the same idea with the location code as the key, and its capacity
is the number of locations, not d. The outer-product form is listed as a variant in §3 (rank 6) and is not one of
the three recommended arms: it adds a dense linear-attention state, and the sparse form is the one that keeps the
model's identity as a stack of sparse memory reads.

### 2.4 Cleanup against a codebook between hops

**What we measured before. This is the most consistent negative in the record.**

| finding | number | label and path |
|---|---|---|
| a cleanup hop (re-query with the weighted centroid of what came back) after a teacher-free address | worse every hop: 2.035848 to 2.090229 to 2.126147 bpb; recovery −7.6% then −12.6% | MEASURED, `MEASURED_track4_239_*` §2, §3 |
| the same hop from a corrupted good address, 17 rungs | harmful in 16 of 17 rungs; from a perfect cue it costs +0.143706 bpb | MEASURED, `MEASURED_track4_240_NO-BASIN-*` §2 |
| the hop converges, to a shared point that is not the answer | cos(q_t, q_t−1) 0.9999 by t = 5; sim@1 rises 0.9704 to 0.9938 while bpb rises | MEASURED, same §0, §4 |
| no store, at any orthogonality, gets a basin from that hop | best lift +0.0129 against a +0.05 criterion, over 9 stores × 6 τ_h × 10 seeds; a perfectly orthonormal store lifts exactly 0.0000 | MEASURED, `4.10-pathway-bdh-cq/E4_MEASURED_the-hop-not-the-address-VOID-twice-*.md` §1 |
| in the 6.1 VSA generator, vector recovery with codebook cleanup against a counts readout | 4.708 against 3.647 bpc | MEASURED, `track4_cross-verify-nanogpt-vs-bohm-penti_FINDINGS.md` |
| softmax sharpening on top of the counts readout | every β worse than linear (best 4.131 against 3.6317) | MEASURED, `track4_1_beta-sweep-readout-rescore_FINDINGS.md` |

E4 relocates the defect to the hop rule: "The hop is either the identity or it is damage" (§1). 240 calls it "a
working cleanup pointed at the wrong target".

**How the SDM-only model differs.** Its hops are not a fixed-point iteration on one store. Each hop has its own
trained store and adds a residual read. Wave 1's `p0_A_h2` and `p0_A_h8` test the hop count. A cleanup step in
OAKENFOLD's sense would snap x towards a token embedding between hops.

**The arm, if it is run at all.** A soft snap against a small codebook, the 4,096 most frequent tokens `E_S`,
after hop 2 of 4, with a zero-initialised gain:

    c = softmax( norm(x) E_S^T / τ ) · E_S        x <- x + g · c

**In words:** guess the next token from the half-built vector, and pull the vector part of the way towards the
embedding of that guess.

Against the full vocabulary this would cost 4 · V · d = 132,382,720 FLOPs a token, about twice the head; against
4,096 tokens it costs 4,194,304 (UNVERIFIED arithmetic). **It is ranked last in §3**, and its role is to retire the
cleanup step for this model at the cost of one run, not to be expected to win.

### 2.5 Fold-back of a verdict

In an autoregressive model with no judge, the verdict on a prediction at position s is the token that actually
arrives at s+1. Writing that verdict back is the run-time write of 2.1. The residual form writes the error:

    u_s = E[y_s+1] − r_fast(s)

**In words:** write only what the fast store failed to predict at s. This is DeltaNet's erase-before-write
(`THINKING_r6-delta-rule-is-deltanet_2026-07-21.md`, "THE REALIZATION").

So fold-back is not a separate mechanism here. It is the choice between the additive and the residual target in
2.1. The record's working rule picks additive for an under-loaded store (`153` §1), and a per-window sparse store
is under-loaded. The residual variant is test 2 in §5, run only after test 1.

Evidence against surprise as the gate: surprise-weighted writes lost on a frequency-estimation task, −0.081 bits
over 3 seeds (MEASURED, `track4_core_bohm-penti-substrate_THINKING.md` §4: "on a frequency-estimation task the
count MASS IS the estimate").

### 2.6 The tree of named nodes

OAKENFOLD's tree of named SDMs has no counterpart, and nothing in the record supports building one now. Its
measurable form was the 4.3 and 4.4 breeding loop, whose fitness function is retired (`MEASURED_OAKENFOLDSTATE_*`
§②: "`4.3` cannot be run as written"). Its named leg exists in Track 3 and is "UNTESTED AS AN ADVANTAGE" (same §③).
It is not proposed here.

---

## 3. Architecture candidates under the no-teacher ruling and the browser

The families come from `experiments/track4/sparsestar-distilled/SPARSESTAR_ARCHITECTURE_CORPUS.md`. A candidate is
kept only if it trains from scratch on next-token loss and runs in the browser engine without attention over the
context. Ranked by the strength of our own evidence and the size of the change.

| rank | candidate | corpus family | change in `track4_sdmonly_models.py` |
|---|---|---|---|
| 1 | sparse run-time write into the woken rows (2.1) | E2 "online-grown model"; the "online learner" of `SPARSESTAR_PLAN.md` | `ProductKeyStore.forward` returns the top-8 ids and weights; new `fast_write(si, w, u, lam, eta)` builds `K(t,s)` from id equality and adds `g·r_fast` in `hidden` after the last hop; flags `--fast-write {none,next,hidden}` |
| 2 | bound conjunction features (2.2) | C1 Engram skeleton (hashed n-gram address) | `features` appends `norm(e_t⊙e_t−1)` and `norm(e_t⊙e_t−1⊙e_t−2)`; `fnorms` and `W_x` widen by 2 blocks; flag `--conj 2` |
| 3 | orthogonality pressure on the representation (§5 test 4) | none in the corpus; the 4.10 E4 cell | `hidden` keeps `norm(x)` (and optionally each hop's normalised q); trainer adds `λ·L_ortho`; flag `--ortho-lambda` |
| 4 | residual (fold-back) target for the write (2.5) | the DeltaNet reading of R6 | `fast_write` target `u_s = E[y_s+1] − r_fast(s)`; flag `--fast-write delta` |
| 5 | width before hops, no shared store | none in the corpus; the 4.10 E2 cell | none: it is a setting (`d`, `hops`, `share_store`) already in the code; wave 1's `p0_A_share` and `p0_A_h8` read it |
| 6 | decaying outer-product state (2.3) | B5 holographic attention, in its matrix form | new module beside `features`: `M_t = λM_t−1 + φ(e_t−1)⊗e_t` in chunked causal form; flag `--tpr` |
| 7 | growth curriculum for the location set | E2 online-grown; WIKI_PICBREEDER page 05's growth curricula | trainer splits each sub-key into two at a named token count (`n_sub` 128 to 256), children = parent ± ε, values copied |
| 8 | soft cleanup between hops (2.4) | D3 holographic / the OAKENFOLD unfold | `hidden` inserts `x += g·softmax(norm(x)E_S^T/τ)E_S` after hop 2 |

Notes on the ranking:

- **Rank 5 carries real evidence and needs no code.** E2 of the 4.10 study (MEASURED,
  `4.10-pathway-bdh-cq/E2_MEASURED_weight-shared-recurrence-*.md` §①): at equal compute a weight-shared block is
  taxed, +0.0775, +0.1721, +0.2280 bpc at 2, 4, 8 applications, all 30 paired seed differences positive; and
  "the biggest lever measured here is neither: it is WIDTH", one wide block beating an eight-deep narrow stack by
  0.1020 bpc at equal parameters. That predicts `p0_A_share` loses to `p0_A_c`. It is a char-level transformer
  result and transfers to this model only as a direction (UNVERIFIED).
- **Rank 7 rests on paper claims, not on ours.** WIKI_PICBREEDER page 05 records that "Complexification is a
  curriculum, importable to SGD", citing arXiv 2406.06262 and a PNAS paper that needed "error-driven + Hebbian +
  blocked curriculum before vanilla SGD would factorize" (UNVERIFIED, papers not reproduced). Splitting a sub-key
  does not preserve the function exactly, because the duplicates shift the k-th score θ (UNVERIFIED reasoning from
  `locate`).
- **Excluded, with the reason.** Family A (MoE experts): a different body. B1 PenttiFusion and B3: attention over
  the context. C2 CRT-SDM and C5 cerebellar, which use fixed random addresses: a frozen random table is worth
  nothing in our record (FROZENTABLE, |z| ≤ 1.47; MEASURED via HISTORY §7), and E4's untrained store was the chance
  floor. Family D metaphors: no mechanism beyond what is listed. Family F shaders: an inference implementation for
  the export work, not a read or write. Resonators and triadic memory: on the corpus's dead list (§4 there).
- **One note for the optimiser work, outside this document's scope.** The tied head is the final cleanup, and at
  d 256 with V 129,280 its D/V is 0.00198 (UNVERIFIED arithmetic). In the SpamLang replication, AdamW at a fixed
  step budget failed every cell at D/V ≤ 0.0078 while SGD with momentum learned the same cells "to near-zero loss"
  (MEASURED, `4.11-lm-head-gradient-bottleneck/MEASURED_T3_spamlang_replication_2026-08-15.md` §⑨.1). That is a
  synthetic task with a deterministic map, so transfer to language modelling is UNVERIFIED. It bears on the head's
  optimiser, which is the subject of the sparse-rows and Muon work, and not on the read.

---

## 4. What un-fracturing asks that bits per byte does not measure

**The goal.** `CLAUDE.md` § THE POINT: "take a capability that exists inside a FRACTURED ENTANGLED representation
and re-express it in a UNIFIED FACTORED one", and to keep the capability while doing so. The 2026-08-14 addendum to
`THE_POINT_*_2026-08-03.md` narrows it: "KEEP THE TRANSIENT STATE'S POWER, UN-FRACTURE THE DURABLE SUBSTRATE."

Bits per byte measures the ability. It is blind to the organisation by construction: two models with the same
next-token distribution have the same bpb, whatever their insides.

**What the September closures retired.** All were measured on nanochat-d20's `x_final`, never on a model like this
one.

| page | what it retires |
|---|---|
| `WIKI/theory/169` | effective rank: norm-matched noise scores 838.9 to 937.0 against the best real arm's 67.4 (MEASURED, via HISTORY §8) |
| `WIKI/theory/170` | a linear value map is a no-op: 21 of 21 invertible maps return the identical next-token score 0.1411 (MEASURED) |
| `WIKI/theory/171` | every non-linear value map tried is worse than none, the 12.1M-parameter MLP by 1.96 points (MEASURED) |
| `WIKI/theory/172` | the index reading of a store cannot beat no store, by construction |
| `WIKI/theory/174` | DCI, MIG, sweep-locality, effective rank and recall@1: "Anything the free group can move is a measure of the basis. Anything it cannot move is blind to the basis" |

**What survives, and in what form.**

1. **A declared decoder.** `174`'s correction (record `MEASURED_DECLAREDDECODER_*_2026-09-02.md`, cited in
   174): un-fracturing is "well-posed exactly MODULO" the stabiliser of a declared decoder class and its fitting
   procedure. The class that escapes is `D_axis_k`, "a readout allowed only `k` of the `d` coordinates", whose
   stabiliser is the monomial group. Measured on teacher states, its coordinate quantity moves exactly 0.0 under
   permutation and 2.44% to 5.65% under rotation (MEASURED, 174).
2. **The SDM-only model's store has a privileged basis by construction.** Its read touches about k of M locations,
   each a discrete atom with its own key and value row. Relabelling locations (keys and values together) leaves
   behaviour unchanged; a rotation of the location set does not exist. So the read is an axis-restricted decoder
   over the location set, and questions asked per location are posed modulo relabelling only (UNVERIFIED reasoning
   from the code; it follows 174's argument and has not been measured on this model). The working vector x does
   not have this property: a rotation of x can be absorbed into `W_x`, the values, `W_q` and `E`, except where the
   per-axis RMSNorm gains and the per-axis query BatchNorm break it (UNVERIFIED reasoning).
3. **Measurands that live on the locations.** Usage and access-KL per hop (PKM's two diagnostics,
   `wikis/WIKI_SDR/wiki/142-training-a-slot-memory-from-scratch-bypass-collapse-and-diagnostics.md` §1); per-location
   leave-one-out purity of the next token (173's member-mode agreement, which 173 says is "a diagnostic, never a
   bound"); per-location causal effect on TEST loss when one row is zeroed. 174 states why slot purity survives the
   free group: "it never reads a value at all". It describes the address partition, and that is all it describes.
4. **Behaviour-grounded ablations already in the trainer.** `zero_read` and `shuffle_keys` measure how much the
   trained model routes through the store. HISTORY §11d states their limit: zeroing the read "does not say the store
   adds anything a no-store network lacks".

**What cannot be claimed, from this model or any arm in §5.**

- That anything is un-fractured, factored or less entangled than anything else. `CLAUDE.md` § THE POINT: "THIS IS A
  GOAL, NEVER A CLAIM", and the measurement has not been run.
- Any comparison of organisation between `sdmonly` and `sdmonly_dense` on location measures. The dense control has
  no locations, so the comparison has no common measurand.
- Any DCI, MIG, sweep-locality, effective-rank or recall@1 reading of x offered as evidence of factorisation (174).
- That sparsity buys factorisation. E5: sparse-positive activation "bought **no** basis movement whatsoever (0/10 vs
  rotation null)"; E1: 5% top-k sparsity cost +0.438792 bpb of which positivity was 8.0%
  (`E1_MEASURED_positive-sparsity-readout_*.md` §①, §②).
- That the store is editable as an advantage. T3-84 and T3-85 left editability "SATURATED, NOT UNDER-POWERED" and the
  fence stays up (`CLAUDE.md` § THE FOUR TRACKS, Track 3).
- That a gain from the run-time write (§2.1) is a gain in the durable representation. It is a transient mechanism,
  and the 4.10 addendum keeps the two halves apart.
- That any number in `170` to `174` transfers to this model without being measured on it.
- "Lossless", and any speed or rivalry claim.

---

## 5. Cheap Spark tests (proposals)

Protocol for all: wave 1's (20M tokens of `train`, B 32, T 256, d 256, seed 0, the centre `p0_A_c` settings). The
bench measured 46,700 tokens a second at this shape (MEASURED, `SDMONLY_CHANNEL.md`, block stamped 2026-10-03T11:20Z), so
a run is about 7 minutes; arms that add compute say so. The push's bar applies: 0.003 bpb and paired z ≥ 3 over TEST
windows. `REPORT_SDMLLMSTORE.md` §10 item 4 records that fixed-seed reruns moved 0.0026 on the M5, so a win under
0.003 needs three seeds.

| # | test | what changes | control | what it decides |
|---|---|---|---|---|
| 1 | RUNWRITE | sparse run-time write, last hop, top-8 locations, target `E[y_s+1]`, λ 0.98, `g` and `η` learned, `g` zero-init | (a) `p0_A_c`; (b) the same arm with targets permuted across positions in each window; (c) the trained arm scored with the write switched off | does an in-window associative write lower bpb beyond its norm |
| 2 | RUNWRITE-DELTA | test 1 with target `E[y_s+1] − r_fast(s)` | test 1 | additive or residual write in an under-loaded store (`153`'s rule) |
| 3 | CONJ | two conjunction features, `norm(e_t⊙e_t−1)` and `norm(e_t⊙e_t−1⊙e_t−2)` | `n_back 10` (same 15 feature blocks, same `W_x` width) and `p0_A_c` | do multiplicative features help the address beyond more features |
| 4 | ORTHO | `L_ortho` on `norm(x)` before the head, sampled over 1,024 positions a batch, λ 0.1 and 1.0 (two runs) | `p0_A_c`; report usage and access-KL per hop beside bpb | does E4's from-scratch gain transfer |
| 5 | LOCAUDIT | no training: on the `p0_A_c` checkpoint, per hop usage, access-KL, leave-one-out next-token purity per location, branch-norm ratio, sub-key migration cos(init, trained), and the share of positions whose top-8 locations recur earlier in the same window | the same measures on the step-0 model and on the `shuffle_keys` ablation | how much repeat there is for test 1 to use; whether keys move at all |
| 6 | TPR | decaying outer-product state `M_t = λM_t−1 + φ(e_t−1)⊗e_t`, read with `φ(e_t)`, added before the hops, gain zero-init | test 1 and `p0_A_c` | sparse location key against a dense d × d key |
| 7 | CLEANUP | soft snap after hop 2 against the 4,096 most frequent tokens, τ learned, `g` zero-init | `p0_A_c` and `p0_A_h8` | retires or keeps the between-hop cleanup |
| 8 | GROW | `n_sub` 128 for the first 10M tokens, then each sub-key split into two (± ε), values copied | `n_sub` 128 throughout (`p0_A_n128` if wave 1 runs it) and `n_sub` 256 throughout (`p0_A_c`) | does a growth schedule beat starting at full size |

**The equation for test 4:**

    z_i = norm(x_i) / ||norm(x_i)||        L_ortho = mean over i ≠ j of (z_i · z_j)²        loss = CE + λ · L_ortho

**In words:** penalise pairs of positions whose final vectors point the same way, averaged over a sample.

The evidence behind it: in E4 a one-layer char model trained from scratch on next-character loss, with the penalty
on its L2-normalised address (which was also the input to its head), scored CE 2.7106 ± 0.0113 at λ 0 and
2.5902 ± 0.0082 at λ 1.0 bits a char over 10 seeds, monotone across four λ values, and its one-shot recovery at the
hardest corruption rose from 0.054 to 0.136 (MEASURED, `E4_MEASURED_*` §2). E4's own sealed prediction was that λ 1.0
would pay the worst CE; it paid the best.

The evidence against: E4's addresses were ReLU outputs with mean pairwise cos 0.5045, and positive vectors are
correlated by construction; the SDM-only model's `norm(x)` is signed. Positions in one window share context and may
rightly share direction, and the penalty pushes them apart. Neither point has been measured.

**Costs, UNVERIFIED arithmetic from `flops_per_token` at d 256, V 129,280:** the centre is about 69.4M FLOPs a token,
95.4% of it the head. Test 1's pair comparison is T² · 8² = 4,194,304 per window per hop, about 134M per batch of
32 windows. Test 3 adds 262,144 FLOPs a token. Test 6 adds about 262,144. Test 7 adds 4,194,304. Test 8 has the
centre's cost after the split. None of these is a measured wall time.

**Order.** Test 5 first, because it costs no training and says whether test 1 has anything to read. Then 1, 3 and 4,
which are the three recommended arms. Test 2 only if test 1 passes. Tests 6, 7 and 8 last. A positive at 20M on
test 1 earns a 100M confirm with a second seed before anyone builds on it.

**Predictions are not sealed here.** Sealing belongs in the plan file before a wave fires.

---

## 6. Sources

Our record, read for this document:
- `experiments/track4/HISTORY_SDM_AND_SPARSESTAR_2026-10-04.md` §§3 to 14
- `experiments/track4/sparsestar-proper/`: `OAKENFOLD_the-generative-structure.md`,
  `THE_COMPLICATE_our-enfolded-order.md`, `BOHMIAN_ANALOGIES_oakenfold.md`,
  `THE_POINT_un-fracturing-representations-while-retaining-ability_2026-08-03.md` (§④, §⑤ and the 2026-08-14
  addendum), `TRACK4_PLAN.md`, `track4_OMNIBUS_02_addendum-deep-code-all-systems-stubbed.md` §§0 to 2,
  `MEASURED_OAKENFOLDSTATE_*_2026-09-02.md`, `track4_cross-verify-nanogpt-vs-bohm-penti_FINDINGS.md`,
  `track4_1_beta-sweep-readout-rescore_FINDINGS.md`, `track4_61_delta_rule_writes_FINDINGS.md`,
  `track4_core_bohm-penti-substrate_THINKING.md`, `THINKING_r6-delta-rule-is-deltanet_2026-07-21.md`,
  `STEP1_MASTERY_QUEUE.md` (A1 to A5, W2-1 to W3c, BU-1 to BU-3), `MEASURED_track4_239_*`, `MEASURED_track4_240_*`,
  `4.1-from-scratch/` to `4.4-distilled-bred/` (counted in OAKENFOLDSTATE)
- `experiments/track4/sparsestar-proper/4.10-pathway-bdh-cq/`: `E1_MEASURED_*`, `E2_MEASURED_*`, `E3_MEASURED_*`,
  `E4_PREREG_ortho-cleanup-basin.md`, `E4_MEASURED_*`, `E5_PREREG_hebbian-fourth-arm.md`, `E5_MEASURED_*`
- `experiments/track4/sparsestar-proper/4.11-lm-head-gradient-bottleneck/MEASURED_T3_spamlang_replication_2026-08-15.md`
- `experiments/track4/sparsestar-distilled/`: `SPARSESTAR_PLAN.md`, `SPARSESTAR_ARCHITECTURE_CORPUS.md`,
  `SPARSESTAR_CURRICULUM.md` §5, `TRACK4_M1_STATUS.md`
- `experiments/track4/sdr-spaghetti/TRACK4_SDR_SPAGHETTI_RETRO.md` (50 toy selftests; their numbers are toy
  measurements and none is cited above as evidence for an arm)
- `experiments/sdr-series/SDR_DEEP_00` to `05` (serving-focused; SDR_DEEP_03 holds that random-code nulls and
  bounds survive learned geometry while capacities do not)
- `wikis/WIKI_SDR/wiki/`: `142-*`, `152-*`, `153-*`, `160-*`, `161-*`
- `wikis/WIKI_PICBREEDER/`: `05-FER-vs-UFR.md` (the objective-versus-search verdict), `06-THE-BRIDGE-to-track6.md`,
  `08-metric-pathologies-and-descriptor-gaming.md`
- `WIKI/theory/169` to `174`
- `experiments/track4/sdmllm/`: `track4_sdmonly_models.py`, `TRAINING_SDMONLY_HOW_WE_TRAIN.md`,
  `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`, `REPORT_SDMLLMSTORE.md` §10, `SDMONLY_CHANNEL.md`
- `CLAUDE.md` (the dwarfstar project file): § THE POINT, § THE TRACKS, § THE FOUR TRACKS
