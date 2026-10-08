# A Language Model Made of Sparse Memory Reads

### The SDM-only base model, and how it is trained from scratch

**Status of this draft.** The frame, the model, the training method and the experimental design are written.
Every measured result of the new model is left as a marked placeholder of the form `[RESULT OWED: ...]`. A
method detail that is not yet fixed is marked `[SETTING OWED: ...]`. No number in this document describes the
new model's performance. Every number it does quote comes from a named file in this repository or from a named
primary source, read on the date given in the references.

**Status update, 2026-10-03.** Three things changed since the draft above was written.

1. **The law.** The model must be all SDM: no softmax attention and no transformer block. Every step that mixes
   positions is a Kanerva memory written and read at run time (the SDM mixer layer, §8.9). A transformer is
   trained only as a yardstick, on the same data, tokens and optimiser. The SDM body is no longer reads only: it
   also has a dense layer after each hop and after each mixer.
2. **The optimiser.** From wave 6 on, every SDM arm and the transformer yardstick train the body with Muon. The
   AdamW-only recipe of §6.4 describes waves 1 to 5.
3. **Results.** §8.7 to §8.12 hold the measured results of waves 7 to 10, the withdrawn reading, and the planned
   scaling-law comparison of wave 11. Every number there is copied from THE LOG of
   `PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`. A result wave 11 will answer is marked PENDING.

---

## Abstract

We ask whether the body of a language model can consist of sparse memory reads and nothing else, when the
model is trained the ordinary modern way: from scratch, by next-token cross-entropy, with AdamW, on public
web text. The model has no attention and no stacked feed-forward layers. A fixed set of context features
forms a working vector. A stack of product-key memories then reads into that vector: at each read a learned
address wakes about 32 of 65,536 locations, and their value rows are summed into the vector. The token
embedding, shared with the output layer, turns the final vector into a distribution over 129,280 tokens. The
memory has the shape of Kanerva's sparse distributed memory and none of its write rule: keys and values are
learned by gradient during training and are fixed afterwards. We train three models in a chain, a base model
on FineWeb-Edu, a chat model fine-tuned from it, and a character model fine-tuned from the chat model, and we
score each in bits per byte on held-out text. Before any run, we sealed ten predictions and a decision rule,
and we set every arm beside a no-read control, a dense control of equal compute, an earlier shape with a dense
readout, and a small transformer.
`[RESULT OWED: wave-1 headline, the centre arm's TEST bpb and its paired difference against no reads and against the equal-FLOP dense control]`
`[RESULT OWED: how many of the ten sealed predictions hit]`
`[RESULT OWED: the long-run SDM BASE TEST bpb at 3B tokens, against 1.40285]`
`[RESULT OWED: SDM CHAT CHAT bpb and looping replies; WEIRD LITTLE GUY scores]`
`[RESULT OWED: the gap to the transformer at matched tokens]`

---

## 1. Introduction

### 1.1 The question

A transformer moves information between positions with attention, and it transforms each position with a
dense feed-forward layer. Both steps are dense: every weight of a layer touches every token. A sparse memory
works differently. It holds many locations, and an address wakes only a few of them. Kanerva proposed such a
memory as a model of human long-term memory [K93]. Product-key memory layers later made the idea trainable
inside neural networks at large scale [L19, B24]. In those systems the memory is an addition to a transformer.
The transformer still does the work of reading the context.

This paper asks a narrower and stranger question. Can the whole body of a language model be sparse memory
reads? We keep a small, fixed way of looking at the context, and we let a stack of memory reads do all the
learned transformation between that context and the next token. We then train the model the way a modern
small language model is trained, with no teacher and no special tricks for the memory beyond those our own
earlier measurements forced on us.

### 1.2 Why it matters

Two reasons motivate the question.

The first is practical. A model with no attention needs no key-value cache that grows with the length of the
text, and its per-token cost is fixed. Such a model can run in a web browser on a laptop. An earlier model of
this family, which still carried a dense readout layer, already runs that way at 12.72 tokens a second (median,
in the browser on an Apple M5, `experiments/track4/HISTORY_SDM_AND_SPARSESTAR_2026-10-04.md` §11d).

The second is about memory itself. In a dense layer, knowledge is spread across every weight. In a memory of
separate rows, a piece of knowledge has an address. A model whose body is such a memory could, in later work,
have rows written while it runs, without a gradient step. This paper does not build that write. It asks first
whether a memory-only body can learn language at all when it is trained normally.

### 1.3 What is claimed and what is not

We claim a model definition, a training method, and a sealed experimental design. The definitions in §5 and §6
are taken from the code, and each names the function it comes from.

We do not claim that the model beats a transformer. Our own earlier measurements say a small transformer is
ahead of this family (§4.6). We do not claim that the memory carries knowledge that a dense network of the same
compute lacks; the controls in §7 exist to test exactly that. We do not claim a run-time write, binding, or any
property of Kanerva's memory beyond its shape. §9 lists the limits in full.

---

## 2. Notation

| symbol | meaning |
|---|---|
| V | vocabulary size, 129,280 |
| d | width of the working vector (256 in wave 1) |
| E | the token embedding, a V x d matrix, shared with the output layer |
| y_t | the token at position t; e_t = E[y_t] its embedding |
| x | the working vector, d wide |
| H | number of hops (memory reads in sequence) |
| n | sub-keys per half; a store has M = n² locations |
| d_a | address width, split into two halves of d_a / 2 |
| k | the number of locations a read is centred on (32) |
| τ | softness of the cut-off (0.25) |
| h | number of address heads per read (1 by default) |
| sg(·) | stop-gradient: the value passes forward, no gradient passes back |
| ρ(·) | RMS normalisation with a learned scale |
| σ(·) | the logistic sigmoid |

---

## 3. Background

### 3.1 Kanerva's sparse distributed memory

Kanerva's memory begins from one observation. If addresses are long binary strings, the space of possible
addresses is far too large to build a location for each. So the memory builds a small random sample of them,
the hard locations, and lets a cue wake every hard location near it. In his words, "the mth location is
activated by x ... if the Hamming distance between x and the location's address Am is below or equal to a
threshold value H", and "the threshold is chosen so that but a small fraction of the hard locations are
activated by any given x" [K93, §3].

A write adds the stored word, recoded as plus and minus ones, into up-down counters at every woken location. A
read pools the counters of the woken locations and thresholds them: "their contents are accumulated (vector
addition) into a vector of U sums, s, and the sums are compared to a threshold value 0" [K93]. The read works
when the cue is only near the stored address, because most of the same locations wake.

Four features of this design matter here. The addresses are fixed and random. A location wakes by distance
within a radius. The contents are counters written in one shot. The read is a sum over the woken set.

### 3.2 Product-key memory layers

Lample et al. made a large key-value memory trainable inside a transformer [L19]. A query selects the k keys
with the largest inner product, the selected scores pass through a softmax, and the output is the weighted sum
of the selected value rows [L19, eqs. 1-3]. Searching all keys is too slow for a large memory, so the keys are
built as products. Each key is the concatenation of two sub-keys, one from each of two codebooks C and C′, so
|K| = |C| x |C′|. The query is split into two halves, each half picks its k best sub-keys, and "we are
guaranteed that the k most similar keys in K are of the form {(c_i, c′_j) | i ∈ I_C, j ∈ I_C′}" [L19, §3.1].
The search costs O(√|K| x d_q) for the halves plus O(k² x d_q) for the grid [L19, §3.2]. Two further details
of that paper carry into ours: a batch normalisation on the query, which raised key usage of a one-million-slot
memory "from 25.8% to 80.3%" [L19, §4.5], and multiple heads that each have their own query and sub-keys and
share one value table [L19, §3.3]. The values there were trained with a higher learning rate, "since the
memory values are learned with sparse updates" [L19, §4.3].

Berges et al. scaled the same layer up [B24]. In their abstract, memory layers "use a trainable key-value
lookup mechanism to add extra parameters to a model without increasing FLOPs", and their models with up to 128B
memory parameters, pretrained on one trillion tokens, outperform "dense models with more than twice the
computation budget, as well as mixture-of-expert models when matched for both compute and parameters". Their
Memory+ design places three memory layers in a transformer that share one memory pool, adds a SiLU gate on the
memory output, and uses query-key normalisation where training is unstable [B24, §3].

### 3.3 Conditional memory and its challenge

DeepSeek's Engram adds a memory beside a mixture-of-experts transformer as "a complementary sparsity axis"
[C26]. Engram is addressed by deterministic n-gram hashing, which allows lookup in constant time and prefetching
from host memory. The authors report a U-shaped law for splitting parameters between computation and memory,
and gains such as MMLU +3.4 and BBH +5.0 at 27B parameters [C26].

A later study questions what that memory does [W26]. In autoregressive image generation, Engram behaves "not as
a content-addressed retriever but as a gated architectural side-pathway". Swapping its hash inputs for random
same-class exemplars "produces statistically indistinguishable next-token distributions", and training with
the table frozen to random noise "costs only ΔFID=0.10" [W26]. That study is in the image domain only. It sets
the right test for any memory claim in language: the content of the table must be shown to matter.

### 3.4 Does a memory add knowledge, or a second view?

Xu et al. asked why nearest-neighbour language models (kNN-LM) beat their base model [X23]. They replaced the
datastore with a learned matrix of vocabulary size, with no retrieval at all, and found that "we can recover
about 55% of the performance gain achieved by kNN-LM" [X23, §4.1]. On their setup the base model scored a
perplexity of 21.750, kNN-LM 19.095, and the learned matrix 20.353 [X23, Table 2]. Their reading is that much of
a retrieval gain comes from ensembling two views and from the softmax temperature, rather than from stored
knowledge. The lesson we take is a rule of method: every memory claim needs a control that has no memory and
the same budget.

### 3.5 How small language models are trained today

The training recipe here follows current small-model practice.

- **The schedule.** Hägele et al. show that a constant learning rate followed by a cooldown "scales predictably
  and reliably similar to cosine", and that one run can then serve several training lengths [H24]. SmolLM2 trains
  its 135M and 360M models "using the WSD scheduler with 20% decay and a learning rate of 3.0×10⁻³", and finds
  that "these smaller models benefited from a single-stage training approach with consistently high-quality
  data" [A25].
- **Data and scoring.** nanochat, "the simplest experimental harness for training LLMs", scores pretraining in
  validation bits per byte and trains with Muon and AdamW together [N26]. We use its unit.
- **Optimisers.** The modded-nanogpt speedrun trains a model to a FineWeb validation loss of 3.28 on eight H100
  GPUs as fast as possible. It applies the Muon optimiser to the hidden weight matrices, while "embeddings and
  the language modeling head use standard Adam" [M26]. Our current recipe uses AdamW for every parameter; Muon
  for the body is a candidate, not part of the recipe in §6.

### 3.6 Where this model sits

Our model is a trained product-key memory stack. It keeps Kanerva's shape: many locations, a small set that
wakes for an address, a summed read, and further reads from the result. It drops his substance: the addresses
are trained real vectors and not fixed random binary strings, a location wakes by rank and not by a radius,
the contents are trained value rows and not counters, and nothing is written after training.

It differs from the product-key layers of [L19, B24] in three ways. The memory is the whole body, not an
insertion into a transformer. The read uses a soft cut-off centred on the k-th score instead of a softmax over
the top k (§5.4). And the gradient from the address into the working vector is stopped (§5.3), a choice our own
measurements forced (§4.4).

---

## 4. Earlier measurements that fixed the design

Each choice below was set by a measurement in this repository. Unless stated, every score is TEST bits per
byte on the same held-out FineWeb-Edu text: 3,892 windows of 257 tokens, 995,323 scored targets, 4,824,566
bytes. Lower is better.

> **A correction to an earlier record.** `REPORT_SDMLLM_S0.md` §3 and the header of
> `track4_sdmllm_paired_window_comparison.py` give the TEST window count as 3,873. The code
> (`val_windows` in `track4_sdmllm_train_one_arm.py`) produces 3,892 windows over the 2,000,916-token
> validation shard, and every stored `test_per_window.npz` holds 3,892 rows. The token and byte totals agree
> in every source.

### 4.1 Every memory claim gets a no-store control

A store read in an earlier line of work used a frozen teacher's final hidden state as its query. A trained
readout with no store, of 8,552,448 parameters and 132.5 times smaller than the 1,133,281,280-parameter store,
reached 1.116828 bpb against the store's 1.188416
(`experiments/track4/sparsestar-proper/MEASURED_track4_245_the-matched-budget-no-store-control_*_2026-07-29.md`;
`track4_245_result.json`). That setting differs from ours: it had a teacher in the loop. Its lesson is the rule
of §3.4, and every arm in this paper follows it.

### 4.2 The first SDM language model lost to its own control

The first model of this family (2,898,432 non-embedding parameters, one shared store of 3,600 locations, two
hops, a SwiGLU readout) was trained for 20M tokens on an M5. With the store, it scored worse than the same
model with the store removed in all three seeds: by +0.0248, +0.0232 and +0.0170 bpb, each with a bootstrap
standard error under 0.0003 (`experiments/track4/sdmllm/REPORT_SDMLLM_S0.md` §5). A dense SwiGLU of the store's
parameter count was also better than the store, by 0.0209.

### 4.3 The cause was the address gradient

A pilot found that the trained store's gradient norm was large (2.49 and 2.78, against 0.42 with no store), and
that it did not sit in the store's own parameters. It flowed from the hard cut-off back through the query into
the shared features. The global gradient clip then shrank every update (`REPORT_SDMLLMSTORE.md` §3).

Stopping the gradient from the query into the working vector removed the whole cost. The store then tied the
no-store model: +0.00006 at softness 0.5, and a mean of -0.0006 over three seeds at softness 0.25
(`REPORT_SDMLLMSTORE.md` §5, §7). With the gradient still flowing, a higher store learning rate and no weight
decay on the store cost +0.124 bpb; with it stopped, the same settings were worth 0.0012. This paper uses the
stopped gradient, a zero start for the values, a store learning rate of three times the base rate, and no
weight decay on the store.

### 4.4 The soft cut-off

The read's cut-off is soft: a location just below the k-th score still counts a little. This came from a study
of Kanerva's memory built from probabilistic bits, where a location fires with probability
`sigmoid((t - d)/w)` and the gain is one over the expected count (`SETTLE/runs/softsdm/REPORT_SOFTSDM.md`).
In the first language model, a softer cut-off recovered 0.022 bpb (`REPORT_SDMLLM_S0.md` §6). Once the address
gradient was stopped, softness from 0.25 to 8 moved the score by only 0.0010 (`REPORT_SDMLLMSTORE.md` §5, block
2). So the softness effect was a symptom of the gradient path. We keep softness 0.25, the best value of that
sweep, and test 0.5 in wave 1.

### 4.5 The wide address

The context features first used the last 4 tokens and 2 moving averages. Widening them to the last 8 tokens and
5 moving averages was the first change that moved the family: 1.49853 against 1.50680, a paired difference of
-0.00827 (SE 0.00026) at width 256 and 300M tokens (`SETTLE/SETTLE_CAMPAIGN_2026-09-30.md`, the
SPARK24 line of 2026-10-01 16:44Z; `runs_spark24/spark_W_s0_300M.result.json`). Every arm in this paper uses the
wide address.

### 4.6 At width 768 the read ties no store, and a transformer is ahead

At width 768 and 600M tokens, the model with a store read scored 1.40285 and the same model with no store
1.40589: a paired difference of -0.00304 (SE 0.00031, z -9.8) on one seed per arm
(`runs_sdmchats/sdmchats_SWL_d768_s0_600M.result.json`, `runs_sdmchats/sdmchats_NWL_d768_s0_600M.result.json`;
ledger line of 2026-10-03). The sealed rule for that comparison needed 0.005, twice the largest seed-to-seed
move seen at width 512, so the decision was a tie. At width 512 two seeds gave +0.00014 (SE 0.00021). On
held-out chat text the read was ahead by 0.0213.

A 4-layer transformer of width 256 (2,902,784 non-embedding parameters), trained on 300M tokens, scored
1.35223 (`runs_spark24/spark_Q_s0_300M.result.json`). That is 0.0506 better than the width-768 store model,
which had about 5.7 times the non-embedding parameters and twice the tokens. The comparison is unpaired but on
the same TEST text.

### 4.7 A caution on noise

The window bootstrap measures how a difference varies over text. It does not measure how a run varies when it
is repeated. A repeat of one arm with the same seed moved 0.0026 bpb on the M5, about ten times the bootstrap
standard error (`REPORT_SDMLLMSTORE.md` §6). A difference of 0.001 to 0.003 therefore needs several seeds.

### 4.8 What the earlier shape had that this one does not

Every model above carries a dense SwiGLU readout after its reads, and its store has 3,600 locations. The model
in this paper removes the readout by default and uses product keys to reach 65,536 locations per hop. §7
measures both changes against the earlier shape directly.

---

## 5. The model

The model is `SdmOnlyLM` in `track4_sdmonly_models.py`. Its parts are defined below in the order the forward
pass uses them.

```
   tokens   y_t  y_t-1  ...  y_t-7                       every earlier token
     │       embedding E (V x d), shared with the output        │
     ▼                                                          ▼
   8 back-token vectors                          5 moving averages, β = .5 .8 .9 .97 .99
     │ each normalised by its own ρ                             │ each normalised by its own ρ
     └──────────── joined (13 d wide) ──── W_x ──── x ──────────┘
                                                    │
     ┌── hop 1 ─────────────────────────────────────▼──────────
     │   address   q = BN( W_q ρ( sg(x) ) )         d -> d_a
     │   halves    q¹ · 256 sub-keys     q² · 256 sub-keys
     │   grid      best 64 of each half -> 64 x 64 -> best 64 locations of 65,536
     │   weigh     w = σ( (score - θ_k) / τ )
     │   read      r = (1/k) Σ w · v           x <- x + r
     ├── hop 2, hop 3, hop 4: the same, each with its own store
     └─────────────────────────────────────────────────────────
                                                    │
                          logits = ρ(x) · Eᵀ        ▼        next token, which joins the context
```

### 5.1 The context features

For each position t, and each decay β in {0.5, 0.8, 0.9, 0.97, 0.99}, the model forms a causal moving average of
all earlier embeddings:

$$a_t(\beta) = \frac{\sum_{s \le t} \beta^{\,t-s}\, e_s}{\sum_{s \le t} \beta^{\,t-s}} \tag{1}$$

**In words.** Each average blends every token so far, and recent tokens count more. A decay of 0.5 reaches back
about two tokens; a decay of 0.99 about a hundred. (`causal_ema` in `track4_sdmllm_models.py`; the window is the
training context of 256 tokens.)

The features are the 8 most recent embeddings, the current one included, and the 5 averages, each passed
through its own RMS normalisation, joined end to end, and mapped by one matrix:

$$f_t = \big[\rho_0(e_t), \rho_1(e_{t-1}), \dots, \rho_7(e_{t-7}), \rho_8(a_t(0.5)), \dots, \rho_{12}(a_t(0.99))\big], \qquad x_t = W_x f_t \tag{2}$$

**In words.** The model sees the last eight tokens exactly and everything earlier as five blurred summaries.
One linear map turns those thirteen vectors into the working vector x. This is the only place where context
enters. (`SdmOnlyLM.features`, `SdmOnlyLM.hidden`.) Positions before the start of a window read as zero
vectors.

The RMS normalisation, used here and below, is

$$\rho(v) = g \odot \frac{v}{\sqrt{\tfrac{1}{d}\sum_i v_i^2 + \epsilon}}, \qquad \epsilon = 10^{-6} \tag{3}$$

with a learned scale g (`RMSNorm`).

### 5.2 The address

Hop h forms an address from the working vector:

$$u = W_{q,h}\; \rho_h\big(\mathrm{sg}_\alpha(x)\big), \qquad \mathrm{sg}_\alpha(x) = \mathrm{sg}(x) + \alpha\,\big(x - \mathrm{sg}(x)\big) \tag{4}$$

$$q = \mathrm{BN}(u), \qquad \mathrm{BN}(u)_i = \frac{u_i - \mu_i}{\sqrt{\nu_i + \epsilon}} \tag{5}$$

**In words.** The address is a linear map of the normalised working vector, then a batch normalisation with no
learned scale or shift. In training, μ and ν are the mean and variance of each address coordinate over the
B x T positions of the batch; in evaluation they are running averages. The normalisation spreads addresses over
the locations, as it does in [L19]. With α = 0, the default, the forward value is unchanged but no gradient
passes from the address back into x. The map W_q, the norm scale, the keys and the values still train.
(`grad_scale`; `SdmOnlyLM.hidden`; `nn.BatchNorm1d(affine=False)`.)

With h > 1 heads, u has h x d_a coordinates and each head takes its own slice.

### 5.3 The product keys, and why the search is exact

A store has two sets of n sub-keys, c¹_1 ... c¹_n and c²_1 ... c²_n, each of width d_a / 2 and normalised to
unit length when used. A location is a pair (i, j), so the store has M = n² locations. The address is split
into halves q¹ and q², and

$$s^1_i = q^1 \cdot \hat c^{\,1}_i, \qquad s^2_j = q^2 \cdot \hat c^{\,2}_j, \qquad S(i,j) = \frac{s^1_i + s^2_j}{\sqrt 2} \tag{6}$$

**In words.** A location's score is the sum of its two half scores. After the batch normalisation each address
coordinate has unit variance, and each sub-key has unit length, so a half score is roughly a standard normal
number. Dividing the sum by √2 keeps the total on the same scale, so that the softness τ has the same meaning
it had in the earlier store. (`ProductKeyStore.locate`.)

The search keeps the best 2k sub-keys of each half, I¹ and I², scores the 2k x 2k grid I¹ x I², and keeps its
best 2k entries:

$$\mathcal{C} = \operatorname{top}_{2k}\big\{S(i,j) : i \in I^1, j \in I^2\big\}, \quad I^1 = \operatorname{top}_{2k}(s^1),\; I^2 = \operatorname{top}_{2k}(s^2) \tag{7}$$

**Why this is exact.** Suppose a location (i, j) is among the best 2k of all n² locations, but i is not among
the best 2k sub-keys of the first half. Then at least 2k indices i′ have s¹_i′ > s¹_i. Each pair (i′, j) then
scores above (i, j), which gives 2k locations above (i, j). That contradicts its place in the best 2k. The same
holds for j. So the best 2k locations of the whole store lie inside the grid, and (7) finds them, ties aside.
This is the argument of [L19, §3.1] with 2k in place of k. A built-in check compares the search with a
brute-force scan of every location (`selftest`, checks 1 and 2).

```
                 sub-keys, half 2  (n = 256)
               j: · · · · ◆ · ◆ · · · ◆ · · · · ·
   sub-keys    ·
   half 1      ◆   ■     ■       ■        ◆ = one of the best 2k = 64 sub-keys of a half
   (n = 256)   ·                          ■ = a grid cell that is scored
   i:          ◆   ■     ■       ■        only 64 x 64 = 4,096 cells are scored,
               ·                          not 256 x 256 = 65,536
               ◆   ■     ■       ■        the best 64 cells are the best 64 locations
```

The cost of one search is 2n half scores of width d_a / 2 plus a 2k x 2k grid. At n = 256 and k = 32 that is 512
half scores and 4,096 sums, against 65,536 full scores for a flat store.

### 5.4 The soft read

Let θ be the k-th best score in 𝒞, treated as a constant for the gradient. Each candidate location m in 𝒞 gets
a weight, and the read sums the weighted value rows v_m (each d wide):

$$w_m = \sigma\!\Big(\frac{S_m - \mathrm{sg}(\theta)}{\tau}\Big), \qquad r = \frac{1}{k\,h}\sum_{\text{heads}}\;\sum_{m \in \mathcal{C}} w_m\, v_m, \qquad x \leftarrow x + r \tag{8}$$

**In words.** About k locations wake. The k-th location has weight exactly one half. A location well above the
cut counts almost fully; one well below counts almost nothing; one near the cut counts in between, so the read
changes smoothly as the address moves. The sum is divided by k, the expected number of woken locations, as in
the soft version of Kanerva's memory (§4.4), and by the number of heads. With several heads, each head has its
own sub-keys and all heads share one value table, as in [L19]. (`ProductKeyStore.forward`.)

The weights pass gradient to the sub-keys through the scores, and to the value rows of the 2k candidates only.
So each training step changes at most 2k rows of a value table per head and position. A built-in check
confirms that only the woken rows receive gradient (`selftest`, check "sparse write").

```
   weight w
   1.0 ┤                       ▄▄▄▄▄▄▄▄▄▄
       │                   ▄▀▀
   0.5 ┤ · · · · · · · · ·●· · · · · · ·    ● the k-th best score, θ
       │              ▄▀▀
   0.0 ┤▄▄▄▄▄▄▄▄▄▄▄▄▀▀
       └──────────────┼──────────────── score
                      θ        width of the bend ≈ τ = 0.25
       only the best 2k = 64 scores are on this curve; the rest weigh zero
```

The value tables start at zero. At the first step every read is therefore zero, and the model is exactly the
model with no reads. A built-in check confirms this equality.

Hops repeat (4), (5), (7) and (8). By default each hop has its own store. One wave-1 arm shares a single store
across all hops, as the Memory+ design shares one pool across layers [B24].

### 5.5 The output: the shared table as the cleanup

After the last hop,

$$z = \rho_f(x)\, E^{\top}, \qquad p(y_{t+1} = v \mid y_{\le t}) = \frac{\exp z_v}{\sum_{u} \exp z_u} \tag{9}$$

**In words.** The final vector is compared by inner product with every token's embedding, and the softmax turns
the comparisons into probabilities. The same table E that turns tokens into vectors at the input turns the
vector back into a token at the output. It plays the role a cleanup memory plays after a noisy read: it snaps a
vector to the nearest stored item. The sampled token becomes y_{t+1} and joins the context for the next step.
(`SdmOnlyLM.forward`.)

### 5.6 What is not in the model

There is no attention, no positional encoding beyond the fixed back-token slots and the decays, and, by
default, no dense layer anywhere in the body. The only operation that is not linear in the features, apart
from the normalisations, is the choice of which locations wake and how much. One wave-1 arm adds back a SwiGLU
readout of width 688 after the hops, to measure what that layer is worth.

### 5.7 Sizes and compute

The sizes below are computed from the code by `count_params` and `flops_per_token` at the wave-1 settings
(d = 256, d_a = 256, n = 256, k = 32, T = 256, V = 129,280). They are properties of the definitions, not
measurements.

| arm | non-embedding parameters | of which store | forward FLOPs per token, body | forward FLOPs per token, head |
|---|---|---|---|---|
| `sdmonly`, the centre (4 hops) | 68,489,728 | 67,633,152 | 3,211,264 | 66,191,360 |
| `sdmonly_none` (no reads) | 856,576 | 0 | 2,031,616 | 66,191,360 |
| `sdmonly_dense` (equal-FLOP SwiGLU, width 192) | 1,446,400 | 0 | 3,211,264 | 66,191,360 |
| `sdmonly`, 8 hops | 136,123,904 | 135,266,304 | 4,390,912 | 66,191,360 |
| `sdmonly`, 512 x 512 locations | 270,078,464 | 269,221,888 | 3,735,552 | 66,191,360 |
| `sdmonly`, one shared store | 17,961,472 | 17,104,896 | 3,211,264 | 66,191,360 |
| `sdm`, the earlier shape (3,600 locations, 2 hops, readout 688) | 3,358,976 | | 7,102,464 | 66,191,360 |
| `sdm_nostore` | 1,384,704 | 0 | 3,088,384 | 66,191,360 |
| `qwen`, 4-layer transformer | 2,902,784 | 0 | 6,324,224 | 66,191,360 |

The embedding adds 33,095,680 parameters to every arm. Three facts follow from the table.

- **The output layer dominates compute.** In the centre arm the head is 95.4% of the forward FLOPs. Any change
  to the body moves a small share of the cost.
- **The memory is cheap in compute and large in parameters.** One read costs 294,912 FLOPs per token: 131,072
  for the address map, 131,072 for the half scores and 32,768 for the weighted sum over 2k rows
  (`read_flops`). The centre arm's 67.1M value parameters cost 1,179,648 FLOPs per token across four hops.
- **The dense control matches compute, not parameters.** A SwiGLU of width 192 costs exactly the FLOPs of one
  read (`matched_dense_f`), and it has about 47 times fewer non-embedding parameters than the centre arm. The
  FLOP count covers matrix products only. It omits the top-k selection, the sigmoid and the batch
  normalisation, so the sparse arm's true cost is higher than the table shows.

---

## 6. Training

### 6.1 The data and the tokenizer

**Tokenizer.** The DeepSeek V4 tokenizer, 129,280 tokens and 127,741 merges, read from the header of a GGUF
file of that model without loading its weights, and checked token for token and merge for merge against the
publisher's `tokenizer.json` (`REPORT_SDMLLM_S0.md` §2; `track4_sdmllm_tokenizer_provenance.json`).

**Web text.** FineWeb-Edu, configuration `sample/10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`,
licence ODC-By. Each document is a BOS token followed by its ids, with no EOS.

| split | contents | tokens | source |
|---|---|---|---|
| `train` | 64,488 documents, the first in file order | 65,522,263 | `track4_sdmllm_data_provenance.json` |
| `val` | the next 2,000 documents, held out | 2,000,916 | same |
| `train_big` | `train` plus later documents of five files; 34 exact duplicates of `val` documents skipped | 3,018,298,789 | `track4_sdmllm24_train_big_provenance.json` |

The first half of `val` (CALIB) is reserved for tuning count-based floors. The second half is TEST: 3,892
non-overlapping windows of 257 tokens.

**Chat text.** `HuggingFaceTB/smol-smoltalk`, revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, licence
Apache-2.0, in a `User: ... / Assistant: ...` template with system messages dropped and a BOS per conversation.
Its train split holds 378,859,013 tokens and its test split 19,809,072. The mix `chat_mix` joins the chat
train split with an equal number of FineWeb-Edu tokens from the end of `train_big`: 757,718,026 tokens
(`track4_sdmllm24_chat_provenance.json`).

**The character's corpus.** `[SETTING OWED: corpus v2 of the WEIRD LITTLE GUY: its sources, its size in tokens, its licence and provenance record, its train and test split]`

### 6.2 The batch

A step uses B = 32 windows of T + 1 = 257 tokens. The window offsets for step s are drawn from a generator
seeded by the pair (seed, s). So the batch at a step depends only on the seed and the step, a resumed run sees
exactly the batches a straight run would see, and every arm with one seed sees the same tokens in the same
order (`batch_for` in `track4_sdmllm_train_one_arm.py`). A step holds 8,192 tokens; 20M tokens is 2,441 steps.
Windows may cross document boundaries.

### 6.3 The loss

$$\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \ln p\big(y_{b,t+1} \mid y_{b,\le t}\big) \tag{10}$$

**In words.** The model is scored on how much probability it gives each next token, averaged over every
position of the batch. Nothing else enters the loss: no teacher, no auxiliary term, no balancing loss for the
memory. On CUDA the loss is computed in compiled, checkpointed chunks so that the full B x T x V logits are never
held at once; the arithmetic is the same (bf16 products, fp32 log-softmax; `ce_chunked` in
`track4_sdmllm24_fused_ce.py`).

### 6.4 The optimiser and its groups

AdamW with betas (0.9, 0.95) and ε = 10⁻⁸, gradient clipping at a global norm of 1.0. Parameters fall into three
groups (`train` in `track4_sdmllm_train_one_arm.py`):

| group | members | weight decay | learning rate |
|---|---|---|---|
| decayed | 2-D matrices other than the embedding and the stores: W_x, every W_q, any SwiGLU | 0.1 | 1 x the schedule |
| not decayed | the embedding E, every RMS scale | 0 | 1 x the schedule |
| store | every parameter whose name starts with `store.`: the sub-keys and the value tables | 0 by default | 3 x the schedule by default |

The address maps W_q sit in the decayed group, not the store group. Initialisation: E and every 2-D matrix
outside the stores from N(0, 0.02²); sub-keys from N(0, 1), normalised in use; value tables at zero.

### 6.5 The schedules

The learning rate at step s of S, with peak η and warmup W = ⌊0.02 S⌋ steps, is

$$\eta_{\cos}(s) = \begin{cases} \eta\,\frac{s+1}{W} & s < W \\[2pt] \eta\,\big(0.1 + 0.9\cdot\tfrac12(1 + \cos \pi p)\big),\; p = \frac{s-W}{S-W} & s \ge W \end{cases} \tag{11}$$

$$\eta_{\mathrm{wsd}}(s) = \begin{cases} \eta\,\frac{s+1}{W} & s < W \\[2pt] \eta & W \le s < (1-f)S \\[2pt] \eta\,\frac{S-s}{fS} & s \ge (1-f)S \end{cases} \qquad f = 0.2 \tag{12}$$

**In words.** Both schedules warm up over the first 2% of steps. The cosine schedule (11) then falls smoothly to
a tenth of the peak. The warmup-stable-decay schedule (12) holds the peak and then falls in a straight line to
zero over the last 20% of steps. A checkpoint from the stable part of (12) can be cooled down at any point, so
one long run yields a score at several token budgets [H24]. (`lr_at`; `wsd_lr` in `track4_sdmonly_train.py`.)
Wave 1 uses (11) except for one arm; the long run in §6.8 uses (12).

### 6.6 Precision

Matrix products run in bf16 under autocast; weights and optimiser state are fp32; the cross-entropy and the
batch normalisation run in fp32. On CUDA, TF32 is allowed for fp32 products.

### 6.7 The score: bits per byte

$$\mathrm{bpb} = \frac{\sum_{t \in \mathcal{S}} -\ln p(y_{t})}{\ln 2 \cdot \sum_{t \in \mathcal{S}} \mathrm{bytes}(y_{t})} \tag{13}$$

**In words.** The total surprise of the model over the scored tokens, in bits, divided by the number of UTF-8
bytes those tokens spell. The unit does not depend on the tokenizer, so any two models can be compared on the
same text. 𝒮 is every target in the TEST windows except a BOS token. (`eval_bpb`.) After training, the trainer
also writes each TEST window's summed nats and bytes, the input of the paired comparison in §7.4.

### 6.8 The three-model chain

```
   SDM BASE ───────────▶ SDM CHAT ───────────▶ WEIRD LITTLE GUY
   from scratch           from BASE             from CHAT
   train_big, WSD         chat_mix              his corpus, with web and
   target 3B tokens                             chat rows in every batch
   │                      │                     │
   scored on              scored on             scored on
   TEST bpb (web)         CHAT bpb              his own TEST bpb
                          looping replies       web TEST and CHAT bpb
                          TEST bpb (forgetting) (how much he forgot)
```

**SDM BASE.** The shape and settings chosen after waves 1 and 2, trained from scratch on `train_big` for a
target of 3B tokens with schedule (12), keeping checkpoints at 0.6B, 1.2B and 2.4B tokens for a token-scaling
curve. Its gate is a TEST score below 1.40285, the best earlier model of the family (§4.6).
`[SETTING OWED: the chosen shape and hyperparameters, after wave 2]`

**SDM CHAT.** SDM BASE fine-tuned on `chat_mix`. It is scored on CHAT bpb, the bits per byte on the first 4,000
windows of the chat test split, and on looping: the share of 30 generated replies that fall into repetition.
TEST bpb on web text is scored again to measure forgetting. Its gate is CHAT below 1.06386, the earlier chat
model of the family, with no looping reply in 30. A dense control is fine-tuned the same way.
`[SETTING OWED: chat fine-tune learning rate, token budget and schedule; the definition of a looping reply]`

**WEIRD LITTLE GUY.** SDM CHAT fine-tuned on the character's corpus, with web and chat rows mixed into every
batch to limit forgetting, over several rounds. Each round is scored on his own TEST, on web TEST and on CHAT
bpb. A control starts from SDM BASE instead of SDM CHAT. An earlier version of this fine-tune put only the
character's own reply tokens in the loss (`track4_weirdguy_finetune.py`).
`[SETTING OWED: the loss mask, the row mix, the number of rounds and the learning rate of the guy fine-tune]`

---

## 7. Experimental design

### 7.1 Wave 1: the protocol

Every wave-1 run follows one protocol: 20M tokens of the `train` split, B 32, T 256, d 256, seed 0, the
optimiser of §6.4, schedule (11), and TEST on the 3,892 windows of §6.1. Every arm uses the wide address of
§4.5. The runs are on the Spark (an NVIDIA GB10).

### 7.2 Wave 1: the arms

**References**, trained by the earlier trainer:

| run | arm | settings |
|---|---|---|
| `p0_R_N` | `sdm_nostore` | wide address, readout 688 |
| `p0_R_S` | `sdm` | wide address, readout 688, 3,600 locations, softness 0.25, α 0, values zero, store lr x3, store wd 0 |
| `p0_R_Q` | `qwen` | the 4-layer transformer |

**The centre and its controls:**

| run | arm | settings |
|---|---|---|
| `p0_A_c` | `sdmonly` | 4 hops, one store per hop, 256 x 256 = 65,536 locations each, k 32, 1 head, softness 0.25, α 0, lr 3e-3, store lr x3, store wd 0, no readout |
| `p0_A_none` | `sdmonly_none` | no reads |
| `p0_A_dense` | `sdmonly_dense` | each read replaced by a SwiGLU of equal FLOPs |

**One change from the centre each:**

| run | change | run | change |
|---|---|---|---|
| `p0_A_lr1` | lr 1e-3 | `p0_A_k16` | k 16 |
| `p0_A_lr6` | lr 6e-3 | `p0_A_k64` | k 64 |
| `p0_A_lr10` | lr 1e-2 | `p0_A_heads4` | 4 heads |
| `p0_A_slr1` | store lr x1 | `p0_A_share` | one store shared by the 4 hops |
| `p0_A_slr10` | store lr x10 | `p0_A_soft5` | softness 0.5 |
| `p0_A_h2` | 2 hops | `p0_A_qg1` | α 1 (the address gradient flows) |
| `p0_A_h8` | 8 hops | `p0_A_ro` | readout 688 after the hops |
| `p0_A_n128` | 128 x 128 locations | `p0_A_wsd` | schedule (12) |
| `p0_A_n512` | 512 x 512 locations | `p0_A_wd` | store weight decay 0.1 |

### 7.3 What each control answers

| control | question it answers |
|---|---|
| `p0_A_none` | How much do the reads add to the fixed features and the shared table? |
| `p0_A_dense` | Does a sparse read beat a dense layer that costs the same compute? |
| `p0_R_S`, `p0_R_N` | Is the memory-only shape as good as the earlier shape with a readout, and does the earlier shape's store tie its own no-store control at this budget? |
| `p0_R_Q` | How far is the family from a transformer at the same tokens? |
| `p0_A_ro` | What is the dense readout worth on top of the memory-only shape? |
| `p0_A_qg1` | Does the stopped address gradient still matter in the new shape (§4.3)? |

### 7.4 Ablations after training

On the trained centre arm and every `sdmonly` arm, two ablations are scored on the same TEST windows
(`score_ablations` in `track4_sdmonly_train.py`):

- **zero read**: every read returns zero. This says how much the trained network routes through its memory.
- **shuffled values**: each value table is permuted by a fixed permutation, so every address still wakes the same
  locations but reads another location's content. This says whether the content of a location matters, the
  test §3.3 asks of any memory.

The trainer also counts, per store, the fraction of locations woken by at least one TEST token in 64 windows.

A large zero-read cost does not show that the memory adds knowledge. A network trained with its reads will lean
on them; the no-read and dense controls answer whether a network without them does as well. §4.6 holds a case:
the width-768 store model with its read zeroed scored 1.54155, 0.136 worse than the no-store model, while the
intact store model was only 0.003 better than the no-store model (`runs_sdmchats/sdmchats_SWL_d768_s0_600M.result.json`).

### 7.5 The paired comparison

Two arms scored on the same windows w are compared by

$$\Delta = \frac{\sum_w n_A(w) - \sum_w n_B(w)}{\ln 2 \cdot \sum_w b(w)} \tag{14}$$

where n_A(w) is arm A's summed nats on window w and b(w) the window's bytes.

**In words.** The bits per byte by which arm A is worse than arm B on exactly the same text. A negative Δ means A
is better.

The standard error comes from 2,000 bootstrap resamples of the windows, drawn with replacement with a fixed
seed, the same resample for both arms, and z = Δ / SE (`paired` in `track4_sdmonly_summary.py`). Resampling
windows keeps the correlation between two arms on the same text, which makes the comparison much sharper than a
difference of two separate scores.

**The counting rule.** A difference counts only when |Δ| is at least 0.003 bpb and |z| is at least 3. The rule
asks for size as well as resolution, because §4.7 shows that the bootstrap understates the run-to-run noise.

### 7.6 The sealed predictions

The following were written down before any wave-1 run started, and are not edited after.

| id | prediction |
|---|---|
| P1 | `p0_A_c` scores TEST between 1.60 and 1.70 |
| P2 | `p0_A_c` beats `p0_A_none` by at least 0.03 |
| P3 | `p0_A_c` beats `p0_A_dense` by 0.005 to 0.06 |
| P4 | `p0_R_S` beats `p0_A_c` by 0.00 to 0.05 |
| P5 | the best learning rate of the four is 3e-3 or 6e-3 |
| P6 | more hops score better: h8 below h4 below h2; h8 beats h4 by less than 0.01 |
| P7 | `p0_A_n512` is within 0.005 of `p0_A_c` |
| P8 | `p0_A_ro` beats `p0_A_c` by at least 0.005 |
| P9 | on `p0_A_c`, zeroing the reads costs at least 0.15 bpb and shuffling the values costs at least 0.03 |
| P10 | `p0_R_N` is within 0.01 of `p0_R_S` |

### 7.7 The decision rule

The shape that goes on to the scale check is memory-only if its best arm is within 0.01 bpb of the best
reference that has no attention (`p0_R_S` or `p0_R_N`). If it is further behind, the table goes to the person
who rules on the project, who decides. In either case the table is shown before the long run starts.

### 7.8 After wave 1

- **Wave 2** combines the best wave-1 settings and adds a second seed of the centre and of the best arm.
- **The scale check** trains the chosen shape at d 768 on 100M to 300M tokens of `train_big`, beside the earlier
  recipe at the same budget. Its gate: the memory-only shape is within 0.01 bpb of the earlier recipe, or ahead.
- **The long run** is SDM BASE (§6.8), with its own predictions sealed before it starts.

`[SETTING OWED: the wave-2 arms and their sealed predictions]`
`[SETTING OWED: the scale-check arms and their sealed predictions]`
`[SETTING OWED: the long run's sealed predictions]`

---

## 8. Results

### 8.1 Wave 1

**Table 3. Wave 1, TEST bpb and paired differences against the centre `p0_A_c`.** Rows will be sorted by TEST
bpb once filled. Δ, SE and z follow (14); "used" is the fraction of locations woken in 64 TEST windows (mean over
stores); tok/s is the median after the first 10% of steps, with the box's load at the start.

| run | arm | change | TEST bpb | Δ vs `p0_A_c` | SE | z | zero read | shuffled values | used | tok/s | load |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `p0_R_N` | `sdm_nostore` | reference | | | | | | | | | |
| `p0_R_S` | `sdm` | reference | | | | | | | | | |
| `p0_R_Q` | `qwen` | reference | | | | | | | | | |
| `p0_A_c` | `sdmonly` | the centre | | 0 | | | | | | | |
| `p0_A_none` | `sdmonly_none` | no reads | | | | | | | | | |
| `p0_A_dense` | `sdmonly_dense` | equal-FLOP SwiGLU | | | | | | | | | |
| `p0_A_lr1` | `sdmonly` | lr 1e-3 | | | | | | | | | |
| `p0_A_lr6` | `sdmonly` | lr 6e-3 | | | | | | | | | |
| `p0_A_lr10` | `sdmonly` | lr 1e-2 | | | | | | | | | |
| `p0_A_slr1` | `sdmonly` | store lr x1 | | | | | | | | | |
| `p0_A_slr10` | `sdmonly` | store lr x10 | | | | | | | | | |
| `p0_A_h2` | `sdmonly` | 2 hops | | | | | | | | | |
| `p0_A_h8` | `sdmonly` | 8 hops | | | | | | | | | |
| `p0_A_n128` | `sdmonly` | 128 x 128 locations | | | | | | | | | |
| `p0_A_n512` | `sdmonly` | 512 x 512 locations | | | | | | | | | |
| `p0_A_k16` | `sdmonly` | k 16 | | | | | | | | | |
| `p0_A_k64` | `sdmonly` | k 64 | | | | | | | | | |
| `p0_A_heads4` | `sdmonly` | 4 heads | | | | | | | | | |
| `p0_A_share` | `sdmonly` | one shared store | | | | | | | | | |
| `p0_A_soft5` | `sdmonly` | softness 0.5 | | | | | | | | | |
| `p0_A_qg1` | `sdmonly` | α 1 | | | | | | | | | |
| `p0_A_ro` | `sdmonly` | readout 688 | | | | | | | | | |
| `p0_A_wsd` | `sdmonly` | schedule (12) | | | | | | | | | |
| `p0_A_wd` | `sdmonly` | store wd 0.1 | | | | | | | | | |

`[RESULT OWED: Table 3, all 24 rows, from track4_sdmonly_summary.py --prefix p0_ --against p0_A_c, with the driver and GPU stamps of each run]`

**Table 4. The sealed predictions, scored.**

| id | sealed | measured | verdict |
|---|---|---|---|
| P1 | `p0_A_c` in 1.60 to 1.70 | | |
| P2 | c beats none by ≥ 0.03 | | |
| P3 | c beats dense by 0.005 to 0.06 | | |
| P4 | R_S beats c by 0.00 to 0.05 | | |
| P5 | best lr is 3e-3 or 6e-3 | | |
| P6 | h8 < h4 < h2; h8 - h4 under 0.01 | | |
| P7 | n512 within 0.005 of c | | |
| P8 | ro beats c by ≥ 0.005 | | |
| P9 | zero read ≥ +0.15, shuffle ≥ +0.03 on c | | |
| P10 | R_N within 0.01 of R_S | | |

`[RESULT OWED: Table 4, measured value and HIT or MISS for P1 to P10; a miss is recorded as a miss]`
`[RESULT OWED: the decision of §7.7, with the gap between the best memory-only arm and the best attention-free reference]`

### 8.2 Wave 2 and the chosen settings

`[RESULT OWED: wave-2 table; second-seed spread of the centre and the best arm; the chosen learning rate, store learning rate, hops, locations, k, heads and schedule]`

### 8.3 The scale check

`[RESULT OWED: the chosen shape at d 768 on 100M to 300M tokens, beside the earlier recipe; gate verdict]`

### 8.4 SDM BASE

| checkpoint | tokens | TEST bpb | Δ vs 1.40285 | CHAT bpb (not trained on chat) |
|---|---|---|---|---|
| decayed from the stable phase | 0.6B | | | |
| decayed from the stable phase | 1.2B | | | |
| decayed from the stable phase | 2.4B | | | |
| final | 3B | | | |

`[RESULT OWED: Table 5, the SDM BASE token-scaling curve, and the gate verdict against 1.40285]`
`[RESULT OWED: zero-read and shuffled-value ablations of SDM BASE, and locations used per store]`

### 8.5 SDM CHAT and the WEIRD LITTLE GUY

`[RESULT OWED: SDM CHAT CHAT bpb against 1.06386, looping replies out of 30, web TEST after the fine-tune; the dense control fine-tuned the same way]`
`[RESULT OWED: WEIRD LITTLE GUY per round: his own TEST bpb, web TEST bpb, CHAT bpb; the control started from SDM BASE]`

### 8.6 The browser

`[RESULT OWED: int8 export agreement with float32 (top-1 on TEST windows, bpb cost) and browser tokens a second, for BASE, CHAT and GUY]`

### 8.7 Wave 7: the best SDM recipe against a fair transformer (2026-10-03)

The champion recipe (CH) is Muon on the body, a SwiGLU of width 192 after every read, an untied input table,
2-token products, and exact 2- and 3-gram hash tables of 65,536 rows each. The transformer yardstick is `qwen`, 4
layers, trained with the same Muon through the same trainer. Width 256 unless named. TEST bpb, lower is better.
At this point every run trained on the `train` shard (§8.11).

| tokens | Muon transformer | champion (CH) | CH + run-time write | CH, no store reads, no n-gram tables |
|---|---|---|---|---|
| 20M | `w7_mq_20M` 1.46388 | `w6_mu_combo` 1.53229 | | `w6_mu_combo_none` 1.53959 |
| 60M | `w7_mq_60M` 1.39966 (A100) | `w7_ch_60M` 1.47910 (A100) | | |
| 60M, d 512 | `w7_mq_d512_60M` 1.34633 | `w7_ch_d512_60M` 1.45053 | | |
| 200M | `w7_mq_200M` 1.37059 | `w7_ch_200M` 1.46811 | `w7_ch_rt8_200M` 1.46746 | `w7_ch_none_200M` 1.45903 |

CH without the n-gram tables at 20M (`w7_ch_nong`) scored 1.53801.

- **The transformer is ahead at every budget.** 0.068 at 20M (X1 HIT) and 0.098 at 200M (X2 HIT). The earlier
  result that the champion beat a transformer at 20M (prediction M3) compared Muon against AdamW. It does not
  survive the fair test.
- **At 200M the trained store reads do not pay.** CH with its reads and n-gram tables (1.46811) is 0.009 WORSE
  than the same recipe with neither (1.45903). X3 MISS.
- **The run-time write of wave 3 fades out.** At 200M it is worth 0.0007 (1.46746 against 1.46811). X4 MISS.
- **Width.** At 60M, d 512 beats d 256 by 0.029; sealed 0.03. X6 MISS by a hair.
- **Tokenizer.** CH at the trained 32,768 BPE scored 1.5129 by document at equal bytes. Its 129k pair (`w7r_*`)
  is not yet scored in THE LOG. X5 PENDING.

### 8.8 Wave 8: context length at 20M tokens (2026-10-03)

Champion recipe under Muon, 20M tokens, the same 8,192 tokens a step at every window. Each run is scored on TEST
windows of its own length.

| window T | CH + run-time write | CH, no run-time write | Muon transformer |
|---|---|---|---|
| 256 | 1.51677 | `w6_mu_combo` 1.53229 | 1.46388 |
| 512 | 1.51473 | 1.53512 | |
| 1,024 | 1.51400 | 1.53940 | 1.46473 |

- C1 MISS: with the write, 1,024 beats 256 by 0.0028, not 0.02.
- C2 HIT in size, wrong sign: without the write, 1,024 is 0.007 worse than 256, inside the 0.007 noise.
- C3 UNDECIDABLE: both gains sit inside the noise.
- C4 HIT: the transformer at 1,024 leads the SDM at 1,024 by 0.049.

**Reading.** At 20M tokens nothing uses a longer window. The transformer's 1,024 window is 0.0009 worse than its
256 window. A long window can only pay once a model has trained long enough to learn long-range use.

### 8.9 Wave 9: the SDM mixer at 20M tokens (2026-10-03)

The SDM mixer layer is a Kanerva memory written and read while the model runs. Each position picks about 32 of a
fixed set of hard-locations by product keys, writes a learned vector of itself into them, and reads the locations
its own address picks, which hold only what earlier positions wrote. Heads fade old writes per token (1.0, 0.999,
0.99 and 0.9 in wave 9; 1.0 is never). The number of locations is fixed, so a token costs the same at any window.
Wave 9 used the no-store arm with the champion's features, no trained hops, and L mixer layers, each followed by a
dense layer of width 4d. Muon, d 256, 20M tokens unless named. Run-to-run noise at this size is about 0.007.

| run | what | TEST bpb |
|---|---|---|
| `w9_mix4_200M` | 4 mixer layers, 200M tokens | 1.49131 |
| `w9_mix4_n128` | 4 layers, 128 x 128 locations a head | 1.52079 |
| `w9_mix4_nofade` | 4 layers, no fading | 1.52271 |
| `w9_mix8` | 8 layers | 1.52787 |
| `w9_mix4` | 4 layers, 64 x 64 locations a head | 1.52874 |
| `w9_mix2` | 2 layers | 1.53214 |
| `w9_mix4_T1024` | 4 layers, T 1,024 | 1.53342 |
| `w9_mix4_k64` | 4 layers, k 64 | 1.54156 |

- N1 MISS: `w9_mix4` beats the same features with dense layers and no window memory (1.53959) by 0.0109; sealed
  0.03.
- N2 MISS: `w9_mix4` is 0.065 behind the Muon transformer at 20M.
- N3 MISS: the 1,024 window is 0.005 worse at 20M, inside noise.
- N4 MISS: at 200M the mixer arm (1.49131) is 0.023 behind `w7_ch_200M` and 0.032 behind `w7_ch_none_200M`. It is
  not like-for-like: the wave 9 arm had no hop SwiGLUs.
- N5 MISS on its first half (2 to 4 layers gains 0.0034); its second half is inside noise.
- N6 MISS: no fading (1.52271) is 0.006 better than fading (1.52874), inside noise.

The best all-SDM model at 20M, `w9_mix4_n128` (1.52079), is 0.057 behind the transformer (1.46388). More locations
per head helped more than more layers.

### 8.10 Wave 10: the mixer on the best recipe, and long windows at 200M (2026-10-03)

"cn" is the `w7_ch_none_200M` recipe: no store reads, hop SwiGLUs of width 192, untied input table, 2-token
products, Muon (1.45903 at 200M). The mixer has 4 layers of 128 x 128 locations a head. 200M tokens of `train`.

| run | what | TEST bpb |
|---|---|---|
| `w10_mq_T1024_200M` | Muon transformer, T 1,024 | 1.35201 |
| `w7_mq_200M` | Muon transformer, T 256 (wave 7) | 1.37059 |
| `w10_cn_T1024_200M` | cn, no mixer, T 1,024 | 1.45886 |
| `w7_ch_none_200M` | cn, no mixer, T 256 (wave 7) | 1.45903 |
| `w10_cn_mix4_nofade_200M` | cn + 4 never-fading mixer layers | 1.46048 |
| `w10_cn_mix4_200M` | cn + 4 fading mixer layers | 1.48242 |

- **The transformer uses a longer window at 200M; the SDM without a mixer does not.** The transformer gains
  0.0186 from T 1,024 (X7 HIT). cn without a mixer gains 0.0002, which is nothing (X3 HIT). So the mixer is the
  only route to a long window in an all-SDM model.
- **The fading mixer hurts.** With fading heads the mixer is 0.023 worse than no mixer (X1 MISS). Without fading it
  ties the best recipe, 0.0015 behind. Never-fading beats fading by 0.022 (X4 MISS).
- PENDING in wave 10: X2 (the mixer at T 1,024 against T 256), X5 (the mixer at T 4,096 against T 1,024), X8 (the
  transformer at T 4,096 against T 1,024), and a 500M-token mixer run.

### 8.11 A withdrawn reading: the training data repeated (2026-10-03)

The `train` shard every run used through wave 10 holds 65,522,263 tokens (FineWeb-Edu `sample/10BT`, 64,488
documents). Every "200M" run therefore saw the data about 3 times, and the Spark's 600M chat-base runs about 9
times.

Two readings rested on curves drawn through those repeated runs, and both are withdrawn as conclusions:

- "More tokens flatten near 1.47, so the SDM is not under-trained; it is short of context" (wave 7).
- The three-point fit of E + A D^-alpha that gave the champion a floor of about 1.466 and the transformer about
  1.352. It had three points, one width, and the 60M points on another machine; it also rested on repeated data.

The measured numbers stand as measurements of runs on a 65.5M-token shard. What they say about scaling is open
again, and wave 11 asks it on fresh data.

### 8.12 The planned scaling-law comparison, wave 11 (2026-10-03)

Wave 11 trains 46 runs on 10 rented boxes of 2x RTX 5090. Phase B, the grid, trains on `train_big`
(3,018,298,789 tokens) in one pass, scored on the same TEST stream as every earlier run.

- **Families.** cn (the best SDM recipe); mq (the Muon transformer, 4 layers, MLP width 8/3 d); cnmix (cn plus 4
  never-fading mixer layers of 128 x 128 locations).
- **Grid.** Widths 256, 384, 512 and 768 by 100M, 400M and 1.6B tokens, for cn and mq. cnmix at width 256 (100M;
  400M at T 256, 1,024 and 4,096) and 512 (100M). Second seeds at width 256, 400M.
- **Phase A** (on `train`, 200M, comparable with wave 10): the never-fading mixer at T 1,024, 4,096 and 16,384; 8
  and 2 layers; 256 x 256 locations; k 64; address width 128; with store reads on; second seeds.
- **The fit, per family:** L(N, D) = E + A / N^alpha + B / D^beta, with N the non-embedding parameters and D the
  training tokens.

| id | sealed prediction | result |
|---|---|---|
| Y1 | on fresh data, `w11_cn_d256_400M` beats the repeated-data `w7_ch_none_200M` (1.45903) by at least 0.02 | PENDING |
| Y2 | from 100M to 1.6B tokens at width 512, the transformer gains more bits than cn | PENDING |
| Y3 | at 1.6B tokens, `w11_cn_d768_1600M` beats `w11_cn_d256_1600M` by at least 0.03 | PENDING |
| Y4 | `w11_mq_d768_1600M` scores below 1.20 | PENDING |
| Y5 | `w11_M1_nf_T1024` beats `w7_ch_none_200M` (1.45903) by at least 0.005 | PENDING |
| Y6 | `w11_M2_nf_T4096` beats `w11_M1_nf_T1024` | PENDING |
| Y7 | `w11_M7_nf_reads` is worse than `w10_cn_mix4_nofade_200M` (1.46048) | PENDING |
| Y8 | `w11_cnmix_d256_T1024_400M` beats `w11_cn_d256_400M` by at least 0.01 | PENDING |
| Y9 | the seed-to-seed gap of `w11_cn_d256_400M` is under 0.005 | PENDING |
| Y10 | the transformer's fitted data exponent beta is larger than cn's | PENDING |

**The open-reference comparison.** nanochat's miniseries v1 reports val_bpb on FineWeb-Edu, our unit and our data,
for 91M to 477M parameters at a tokens-to-parameters ratio of 8 (d10: 91M parameters, 0.73B tokens, 1.0312 bpb,
about 4.0e17 FLOPs). Our largest run before wave 11 was about 4e16 FLOPs, about 10 times below that point
(`RESEARCH_SDMONLY_CANONICAL_REFERENCE_2026-10-04.md` §3). nanochat trains at context 2,048 and scores on other
held-out documents, so its level is not yet directly comparable with ours. The note proposes scoring open
FineWeb-Edu checkpoints (DataDecide, SmolLM2-135M) on our TEST and calibrating our yardstick at nanochat's shape.
That is a proposal, not run.

---

## 9. Limits, and what is not claimed

- **No write at run time.** Keys and values are learned by gradient during training and are fixed afterwards.
  The model never stores a new memory while it runs. That is the step that would make it a memory in Kanerva's
  full sense, and it is not built here. (2026-10-03: the SDM mixer layer of §8.9 writes at run time. The trained
  stores still do not. Whether the mixer uses a long window at scale is open; see §8.10 and §8.12.)
- **No binding.** The model has no bind or unbind operator and no superposed state. It stores separate rows and
  selects among them.
- **Trained by gradient.** The addresses are trained real vectors, not fixed random ones, and a location wakes by
  rank, not by a distance radius. Every result is a result about a trained product-key memory stack.
- **One corpus, one tokenizer.** Every web score is on FineWeb-Edu with the DeepSeek V4 tokenizer. The output
  layer over 129,280 tokens holds most of the compute (§5.7), so a smaller vocabulary might change the balance
  between body and head. The results may not carry to other data or another vocabulary.
- **One seed in wave 1.** Each wave-1 arm runs once. The bootstrap standard error understates the run-to-run
  noise by about ten times (§4.7). Differences below a few thousandths of a bit per byte are not settled until
  wave 2 adds seeds.
- **Small models, short budgets.** Wave 1 trains on 20M tokens at width 256. The relative order of arms may change
  at larger width and longer training, which is why the scale check exists.
- **Matched compute is matrix products only.** The equal-FLOP control omits the cost of the top-k selection, the
  sigmoid and the normalisation. The sparse arm's wall-clock cost per token is reported beside its FLOPs and may
  be higher.
- **A small transformer may stay ahead.** On the record so far a 4-layer transformer of width 256 beats every
  model of this family at matched or larger budgets (§4.6). This work does not aim to overturn that. It asks
  what a memory-only body can do, and it reports the gap. (2026-10-03: with Muon on both sides the gap was 0.068
  at 20M and 0.098 at 200M, §8.7.)
- **Repeated data before wave 11 (2026-10-03).** Every run through wave 10 trained on a 65,522,263-token shard,
  so runs past about 65M tokens repeated it. Scaling readings from those runs are withdrawn (§8.11).
- **No claim about structure.** The model has no named dimensions and no measured geometry. Nothing here says its
  representations are cleaner or more factored than a dense network's.

---

## 10. References

Each source was opened at the address given, on 2026-10-03 (UTC). A quotation in the text is from the page or
PDF named here.

- **[K93]** Kanerva, P. Sparse Distributed Memory and Related Models. In M. H. Hassoun (ed.), *Associative Neural
  Memories: Theory and Implementation*, pp. 50-76. Oxford University Press, 1993.
  https://redwood.berkeley.edu/wp-content/uploads/2020/08/KanervaP_SDMrelated_models1993.pdf (PDF read, §3).
- **[K88]** Kanerva, P. *Sparse Distributed Memory.* MIT Press, 1988. Cited through [K93]; the book itself was not
  opened. UNVERIFIED beyond its citation in [K93].
- **[L19]** Lample, G., Sablayrolles, A., Ranzato, M., Denoyer, L., Jégou, H. Large Memory Layers with Product
  Keys. NeurIPS 2019. arXiv:1907.05242. https://arxiv.org/abs/1907.05242 (abstract and PDF read, §3.1 to §3.3, §4.3, §4.5).
- **[B24]** Berges, V.-P., Oğuz, B., Haziza, D., Yih, W., Zettlemoyer, L., Ghosh, G. Memory Layers at Scale.
  arXiv:2412.09764, 2024. https://arxiv.org/abs/2412.09764 and https://arxiv.org/html/2412.09764v2 (abstract and
  §3 read).
- **[C26]** Cheng, X., Tian, R., Zeng, W., Dai, D., Chen, Q., et al. Conditional Memory via Scalable Lookup: A New
  Axis of Sparsity for Large Language Models. arXiv:2601.07372, 2026 (revised 2026-07-12). The module is named
  Engram. https://arxiv.org/abs/2601.07372 (abstract read).
- **[W26]** Wang, J., He, Q., Gu, C., Heng, P.-A. Does Engram Do Memory Retrieval in Autoregressive Image
  Generation? arXiv:2605.13179, 2026. https://arxiv.org/abs/2605.13179 (abstract read).
- **[X23]** Xu, F. F., Alon, U., Neubig, G. Why do Nearest Neighbor Language Models Work? arXiv:2301.02828, 2023.
  https://arxiv.org/abs/2301.02828 (abstract and PDF read, §1 and §4.1).
- **[H24]** Hägele, A., Bakouch, E., Kosson, A., Ben Allal, L., Von Werra, L., Jaggi, M. Scaling Laws and
  Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS 2024. arXiv:2405.18392.
  https://arxiv.org/abs/2405.18392 (abstract read).
- **[A25]** Ben Allal, L., Lozhkov, A., Bakouch, E., et al. SmolLM2: When Smol Goes Big - Data-Centric Training of a
  Small Language Model. arXiv:2502.02737, 2025. https://arxiv.org/abs/2502.02737 and
  https://arxiv.org/html/2502.02737v1 (abstract and training sections read).
- **[N26]** Karpathy, A. nanochat. https://github.com/karpathy/nanochat (README read).
- **[M26]** Jordan, K., et al. modded-nanogpt. https://github.com/KellerJordan/modded-nanogpt (README read).

**Repository sources.** All paths are under `experiments/` unless they begin with `runs_`, in which case they are
under `experiments/track4/sdmllm/`.

- `track4/sdmllm/track4_sdmonly_models.py`: the model (§5), sizes and FLOPs (§5.7)
- `track4/sdmllm/track4_sdmonly_train.py`: the WSD schedule, the ablations (§6.5, §7.4)
- `track4/sdmllm/track4_sdmllm_train_one_arm.py`: batches, optimiser groups, the cosine schedule, scoring (§6)
- `track4/sdmllm/track4_sdmllm_models.py`: `RMSNorm`, `SwiGLU`, `causal_ema`, `grad_scale`, the earlier shape
- `track4/sdmllm/track4_sdmonly_summary.py`: the paired comparison (§7.5)
- `track4/sdmllm/track4_sdmonly_wave1_submit.sh`: the exact wave-1 commands
- `track4/sdmllm/PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md` §4: the sealed arms, predictions and decision rule
- `track4/sdmllm/REPORT_SDMLLM_S0.md`, `track4/sdmllm/REPORT_SDMLLMSTORE.md`: §4.2 to §4.4 and §4.7
- `track4/HISTORY_SDM_AND_SPARSESTAR_2026-10-04.md`: the record behind §4
- `SETTLE/SETTLE_CAMPAIGN_2026-09-30.md`: the SPARK24 and SDMNEXT scores in §4.5 and §4.6
- `track4/sparsestar-proper/MEASURED_track4_245_*_2026-07-29.md`, `track4_245_result.json`: §4.1
- `SETTLE/runs/softsdm/REPORT_SOFTSDM.md`: the soft cut-off (§4.4)
- `track4/sdmllm/track4_sdmllm_data_provenance.json`, `track4_sdmllm24_train_big_provenance.json`,
  `track4_sdmllm24_chat_provenance.json`, `track4_sdmllm_tokenizer_provenance.json`: data and tokenizer (§6.1)
