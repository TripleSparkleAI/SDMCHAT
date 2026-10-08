# C_q01 modded-nanogpt PR #360 (record #92, ANVIL2)

- query: GET https://api.github.com/repos/KellerJordan/modded-nanogpt/pulls/360 , /pulls/360/files , /issues/360/comments , raw record README records/track_1_short/2026-08-30_ANVIL2/README.md at sha 4f5270e5
- tool: python3 urllib, GitHub REST API unauthenticated
- fetched (UTC, HTTP status): 2026-10-03T10:46:57.502825Z 200
- title: New Record: 0.665 minutes (39.9 seconds): ANVIL2, Sampled-softmax, New Embedding Table, Full-stack fp8 (-34.0s, -46% same hardware)
- author: devenpzak; created_at 2026-08-31T21:42:16Z; closed_at 2026-09-28T21:19:54Z; merged_at 2026-09-28T21:19:54Z

## PR body (verbatim)

Created by: Deven Pietrzak (MIT Math. X: [DevenPzak](https://x.com/DevenPzak). Lab: [Hyperstition (fka Social Physics Lab)](https://x.com/hyperstition_cc))

After a few weeks since my last record without touching the PR, I finally got access to some more H100s. The following record was built over 5 days. **Independently reviewed and reproduced by four separate sources over the past two days**

This PR will replace #349.

| | runs | wall (training section) | final val CE |
|---|---|---|---|
| **this PR** | **18** | **39.914 +/- 0.120 s** | **3.27731 +/- 0.00099** |
| record #89, same machine, in between record runs | 9 | 73.889 +/- 0.137 s | 3.27828 +/- 0.00205 |
| **delta** | | **-33.98 s, -46.0 %, 1.85x** | |

Vs 3.28: p = 9.4e-10**. Cold cache for all. 

Note: Lost one leg's raw logs when the pod I was renting expired. Recorded the result and it's included in stats and footnoted in statistics.md

## Ablation table

Mostly n=2. 

| group | net seconds |
|---|---:|
| **Sampled-softmax training loss** | **+8.40** |
| **Embeddings**, hashed n-gram table | **+7.05** |
| **ANVIL**, optimizer | **+5.08** |
| &nbsp;&nbsp;├─ update rule + anneal/shipping | +4.16 |
| &nbsp;&nbsp;└─ optimizer-update graph capture | +0.92 |
| **Full-stack FP8** | **+4.96** |
| **Mixed-width attention** | **+3.82** |
| **Training-step graph capture** | **+2.99** |
| **Sparse grad comms & data loader** | **+1.20** |
| **MUDDformer residual mix** | **+0.48** |
| **Total** | **+33.98** |

ANVIL is an improvement of 21 millinats at roughly equal wall clock, compared to record 89's shipper optimizer.

**For this table, I estimated the val-to-wall-clock conversion using the exchange rates in ms/millinat that made the changes add up to 33.98s, which ended up as 164 ms/millinat. This was within the empirical measured cost of 120-250 ms/millinat.

## Reproducing it

```bash
python data/cached_fineweb10B.py 9
bash run.sh
```

Notes for reproducing it in README, but make sure to follow the following especially, as other ways could throw off the reproduction: use the pinned torch build, Driver >= 580, bare metal needs `libcudart.so.13`, make sure to get the kernel properly.


Note:

A lot of people seem to take issue with the bigram table. Wanted to address a couple misconceptions about it:

1. It's only 7 seconds of 34. I've also had people tell me my ablation was wrong, so I went back, removed the embedding table increases, and added steps until the model was back under 3.28. On net, it cost 7 seconds, validating the ablation table. 

2. People think nobody had done it because it was too many parameters / shouldn't be allowed. In fact, it’s the opposite. The table was already 290M parameters (over twice the size of the model), and the reason it hadn’t grown was because it would have made the time worse. 

The old code kept copies of the table on every GPU, wrote and zeroed a table-sized gradient every single step, and wrote the table back to every GPU after every step. This made the cost linear in table size but the gains roughly logarithmic. 290M was the crossing point, which is why it was set to that size. 

I sharded the table across 8 GPUs, pulled only the rows the current batch’s tokens actually hashed to, sent gradients back as a segment-sum rather than table-sized tensor, and ran Adam only on the touched rows. Then the optimizer cost scaled with rows used per step, making it almost invariant to table size (1/4 size embedding table, 16B instead of 65B, only saved 0.17 seconds). This let me push to 224x and add a trigram channel. 

Even if I didn’t expand it, just from the engineering described in the last paragraph, I would have saved over one second from this change alone. The real reason no one had expanded the embedding table was not because of the table itself but rather because it would have been too slow. The table was already significantly larger than the model, so the bigram table was already far too large to ever scale to frontier training. 

## Comments (verbatim, first 1500 chars each)

### bai634140-ui 2026-09-01T03:27:08Z

wow，really?

### xTimeCrystal 2026-09-01T05:24:34Z

65B embedding parameters! I wonder if it is legal to train an [infini-gram](https://arxiv.org/abs/2401.17377) model on CPU first, then achieve the loss req on step 1?

### ShizukaKuze 2026-09-01T05:39:55Z

> 65B embedding parameters! I wonder if it is legal to train an [infini-gram](https://arxiv.org/abs/2401.17377) model on CPU first, then achieve the loss req on step 1?

There's no way this follows the rules. Firstly, the repository maintainer already said, "This is the most tricky; I would say that hardcoding it ok only when there's a coherent theory for why it makes sense, i.e., when we could expect the hardcoded value to generalize to other settings in some sense. Which isn't the case for the lambdas; since I have no idea how to determine them a priori without just doing a run."

I honestly have not looked through the code, but I'm just willing to go on a tangent and say that if there really are 65 billion embedding parameters, it'sunctionallylookup table.

> Disproportionately degrades the readability of the codebase. A 200 line kernel to drop 300ms is considered worthwhile. 500 lines that convolute the optimizer layout for a 50ms gain will likely be rejected.

This is ridiculous. It adds nearly 200,000 lines of code. (I was actually wrong here. Still, 65 billion embedding table kind of defeats the purpose. At that point, how much of the model is actually GPT?

### devenpzak 2026-09-01T08:01:59Z

@xTimeCrystal @ShizukaKuze 

I sincerely appreciate y'all bringing these up, and that's why I included the bigram/trigram count, so this could be discussed. I expanded the already-existing table from 377,280 to 84,602,880 rows. I knew people would ask about it, but I firmly believe that it's legal since it's (a) the same accepted mechanism widened and given a new trigram channel, (b) entirely learned on clock, with substantial collisions preventing memorization, and (c) seems to be clearly allowed by the rules. **It's also only ~20% of the PR's record**. Details below. Sorry for the delay, this took a while to reply to.

**Regarding "nearly 200,000 lines of code"**

> > Disproportionately degrades the readability of the codebase. A 200 line kernel to drop 300ms is considered worthwhile. 500 lines that convolute the optimizer layout for a 50ms gain will likely be rejected.
> 
> This is ridiculous. It adds nearly 200,000 lines of code. (I was actually wrong here. Still, 65 billion embedding table kind of defeats the purpose. At that point, how much of the model is actually GPT?

Of the +194,329 in the diff, **189,260 are the 30 files in the record folder**, which is the attached evidence, solely consisting of 26 run logs, their statistics, and the README. Those are required attachments, not code. If you take a look at past PRs, you'll notice they had similar structure. For example, the most recent two accepted PRs, #342 and #337, had diffs of 125k and 211k lines, resp

### trianxy 2026-09-01T08:49:26Z

@devenpzak, wow, this is a big one :)

Personal opinion: It would be of great help if you can create a new sub-PR (or just carve out a first commit) which implements only one part of this PR (eg the ANVIL2 optimizer), with as few code line changes to current master as possible.

Once we understand and confirm the latter diff, I think we can go faster through the rest (and resolve discussions like the new embedding table).

Btw: you cite #89 a lot, but that seems to be a typo

### ShizukaKuze 2026-09-02T03:51:45Z

> h



> @xTimeCrystal @ShizukaKuze
> 
> I sincerely appreciate y'all bringing these up, and that's why I included the bigram/trigram count, so this could be discussed. I expanded the already-existing table from 377,280 to 84,602,880 rows. I knew people would ask about it, but I firmly believe that it's legal since it's (a) the same accepted mechanism widened and given a new trigram channel, (b) entirely learned on clock, with substantial collisions preventing memorization, and (c) seems to be clearly allowed by the rules. **It's also only ~20% of the PR's record**. Details below. Sorry for the delay, this took a while to reply to.
> 
> **Regarding "nearly 200,000 lines of code"**
> 
> > > Disproportionately degrades the readability of the codebase. A 200 line kernel to drop 300ms is considered worthwhile. 500 lines that convolute the optimizer layout for a 50ms gain will likely be rejected.
> > 
> > 
> > This is ridiculous. It adds nearly 200,000 lines of code. (I was actually wrong here. Still, 65 billion embedding table kind of defeats the purpose. At that point, how much of the model is actually GPT?
> 
> Of the +194,329 in the diff, **189,260 are the 30 files in the record folder**, which is the attached evidence, solely consisting of 26 run logs, their statistics, and the README. Those are required attachments, not code. If you take a look at past PRs, you'll notice they had similar structure. For example, the most recent two accepted PRs, #342 and #337, 

### devenpzak 2026-09-04T05:42:15Z

> @devenpzak, wow, this is a big one :)
> 
> Personal opinion: It would be of great help if you can create a new sub-PR (or just carve out a first commit) which implements only one part of this PR (eg the ANVIL2 optimizer), with as few code line changes to current master as possible.
> 
> Once we understand and confirm the latter diff, I think we can go faster through the rest (and resolve discussions like the new embedding table).
> 
> Btw: you cite #89 a lot, but that seems to be a typo

Thanks for the message! I will consider working on a sub-PR in the near future, but at the moment I am working on a couple other projects (and currently don't have H100s). Whenever I get a chance I will do this, or hopefully it will already be reviewed by then.

I fixed the PR description - PR 89 was a typo, was supposed to be record 89. That was a stand-in description because I was trying to push it out as soon as possible; now I fixed it and made it a lot more concise.

### devenpzak 2026-09-04T06:11:30Z

> In my opinion, if everything holds to be true, this is a very impressive pull request. However, the embedding table made me unironically spit out my drink. If it's truly 65 billion parameters, and if it get merged, I don't think this repository would have very much real world implication anymore. It's obviously up to the project maintainer, and it's just my opinion. But I do strongly believe that if the 65 billion parameter embedding table was merged into this repository, it would seriously degrade this repository's real-world implications. Besides that, I think it's very commendable what you have accomplished!

That's a fair point, and I appreciate your compliments! 

However, I do feel like this repository's real-world applications don't exclusively come from the entire configuration being deployable in real LLM training. The challenge is more about speed optimization. Of course, that often leads to innovations like Muon (and hopefully now ANVIL) which come out and into frontier usage. But at the same time, things like the specific batch / schedule tuning, disabling garbage collection, the untimed warmup pass, and even the former ~290M parameter bigram embedding table are optimizations that have played a role in dozens of records but have little applications to anything else. The challenge combines both of these into one question of pure optimization, so even if a massive bigram/trigram embedding table doesn't have as many real-world applications, it doesn't mean that

### BurnyCoder 2026-09-07T02:15:57Z

https://x.com/vvvincent_c/status/2096275917316522245 
> the most suspicious change is using a 65B-parameter lookup table for training the 124M param model.

Yeah. No.

### KellerJordan 2026-09-07T06:17:54Z

Personally I would consider all of the changes described here to be 100% legitimate. Both @ClassicLarry and I have the power to accept new records— for me to accept I’d need to first reproduce this result myself, & I haven’t had time to do so yet.

### ClassicLarry 2026-09-07T07:28:55Z

I haven't gone through all the specifics yet, but at a high level I agree it looks great. Hoping to find a block of time soon to look at more closely and refactor the code into something easier to follow, now that we are jumping to 5k+ lines. I'd be curious to understand where the intuition came from on Sampled-softmax training loss and Embeddings, hashed n-gram table. And if any of the results are a joint effort across Social Physics Lab or Autoresearch harness on Codex (in the past groups have set records then after the fact revealed a larger initiative behind it, which is totally fine too). In any case, very impressive!



### overrule 2026-09-15T21:35:26Z

Hi @ClassicLarry, I'm the founder of [Hyperstition](https://x.com/hyperstition_cc/status/2098471087415992455) (formerly Social Physics Lab). The nanoGPT speedrun was a testing ground for ANVIL I, the first iteration of our optimizer. Since then, we have released ANVIL III, which beats muon by 20-28 millinats, regime dependent, [tested up to 1.2B scale](https://hyperstition.cc/cutting-pretraining-costs-by-62-percent). Just today, we released an API for our model, [Feather](https://x.com/hyperstition_cc/status/2099972576005214469), which beats Qwen3-1.7b using 180x fewer total training tokens.  

Would love to chat in private, and share some of our findings on broader pretraining research. We'll be further be releasing our results on frontier RL over the coming weeks. For anyone interested in learning more about our work, or simply chatting, drop in a text at aj@socialphysics.inc. 

### ClassicLarry 2026-09-18T07:27:46Z

Ok this one is up next. I'll try to merge it in with the latest main, going piece by piece to validate the impact. Hopefully later this weekend. and I'll message you to chat!

### mtae 2026-09-29T13:24:36Z

@devenpzak Congrats on the record, and thanks for the detailed writeup. Two questions from someone who doesn't have 8xH100 access to reproduce:

1. Held-out generalization: have you evaluated the trained model on a different slice of FineWeb, one that is disjoint from both the training shards and the standard validation set? If so, is the loss similar to the 3.277 on the standard validation set, and how does the gap compare to record #89's model on the same slice? I'm curious mainly because the large hashed n-gram table could in principle benefit more from any near-duplicate text between train and val.

2. Checkpoint: is it possible to release a checkpoint (or even just the dense weights plus a way to regenerate the n-gram table), or the eval script and a small-scale version of it? That would let people re-check the reported validation loss, run it on other data, and confirm that validation uses the full softmax and full model, without needing to redo the training run.

No worries if the table size makes a full checkpoint impractical. A held-out-slice number alone would already help. Thanks!

### OhadRubin 2026-09-29T13:52:56Z

For what it's worth, I'll toss my own opinion here:
Having read only most of the thread, there seems to be a disagreement about if this record deserves to be accepted. It was obviously merged. But I think the spirit of the previous, sub 65B param catagory was testing changes that aren't so damn memory hungry.
Maybe an additional catagory is needed? Where total parameters are also limited?

## Record README excerpts (verbatim, lines 120-165, 218-226, 274-287, 362-380)

```
ablation run is counted, the same rule the record pool itself is held to.

| group / component | net@r* |
|---|---|
| **ANVIL** | **+5.08** |
| &nbsp;&nbsp;├─ ANVIL algorithm (update rule + anneal/shipping) | +4.16 |
| &nbsp;&nbsp;├─ optimizer-update graph capture (tgr) | +0.92 |
| **TRAINING-STEP GRAPH CAPTURE** | **+2.99** |
| **SAMPLED-SOFTMAX TRAINING LOSS (val pipeline untouched, full-vocab)** | **+8.40** |
| **EMBEDDINGS (hashed n-gram table)** | **+7.05** |
| **FULL-STACK FP8** | **+4.96** |
| **MIXED-WIDTH ATTENTION** | **+3.82** |
| **SPARSE GRAD COMMS & DATA LOADER** | **+1.20** |
| &nbsp;&nbsp;├─ value-embedding sparse gradient exchange | +1.13 |
| &nbsp;&nbsp;├─ parallel shard loader (openpack) | +0.07 |
| **MUDDFORMER RESIDUAL MIX (dynamic dense connections)** | **+0.48** |
| **TOTAL** | **+33.98** |

Per-mechanism measurements behind the groups (removal deltas vs interleaved anchors):

| mechanism | wallΔ | valΔ (millinats) | legs | net@r* |
|---|---|---|---|---|
| CE stack (sampled softmax + prefix-CE + fused kernel) | +8.80 | -2.4 | 2 | **+8.40** |
| n-gram table (84.6M hashed, sparse) | +1.19 | +35.8 | 2 | **+7.05** |
| fp8 extension (MLP+lm_head cache) | +5.02 | -0.4 | 2 | **+4.96** |
| ships / anneal endgame (EMA+tavg+blend+decon) | +0.52 | +20.2 | 2 | **+3.82** |
| attention pkg (mixed-width, dv64, bankc2, fp8 QKV) | +4.95 | -6.9 | 2 | **+3.82** |
| fwd/bwd CUDA-graph runners | +2.99 | +0.0 | 2 | **+2.99** |
| VE sparse path | +1.05 | +0.5 | 2 | **+1.13** |
| optimizer tail graphs (tgr) | +0.84 | +0.5 | 3 | **+0.92** |
| MUDD 10-coef mix | -0.51 | +6.1 | 2 | **+0.48** |
| ANVIL update rule vs 89-Muon | -0.47 | +4.9 | 2 | **+0.33** |
| openpack loader | +0.07 | +0.0 | 2 | **+0.07** |
| **TOTAL** | +24.45 | +58.2 | | **+33.98** |

*anchors: wall n=7, val mean 3.2775 | Σwall=24.45s Σval=58.2 millinats | implied rate r*=164 ms/millinat (measured band ~120-250)*

Notes: (1) ablations are not strictly additive: interactions are absorbed by the implied
rate, which is why it is stated as a normalization convention; (2) the sampled softmax is
training-only: validation is always the full 50,304-way softmax, unchanged from record #89;
(3) the optimizer-update graph row's three legs read +2,090, +55 and +365 ms against their
interleaved anchors; the first ran immediately before another leg that is also high against its
own replicate, so the two of them look like a localised artifact over two adjacent runs. All
three are counted anyway, under the all-runs-count rule; (4) swapping the whole optimizer process for record #89's optimizer exactly as
#89 ships it (no weight shipping, eager tail) costs +21.1 millinats of val at roughly equal
wall. That is the whole-mechanism check quoted above; the two ANVIL rows are measured from their
...
decay ×1.5; exact-fp32 commits on bf16 storage via a uint16 mantissa sidecar; and a bank
tail-blend ship step (fp32 EMA over the last ~300 steps, blended into the shipped banks).
lm_head/embed on sharded Adam; embedding-table LR multipliers 70.

**FP8 everywhere it pays**: full fp8 MLP forward **and** backward (dx/dW1/dW2) with statically
clip-free scales; lagged comm-overlapped weight-cache quantize; fp8 lm_head cache feeding the
fused CE in both layouts, refreshed by a tiled-transpose kernel. Record #89 has fp8 on the MLP
up- and down-projection forward only.

...

**Sampled (shared-negative) softcapped cross-entropy for the early stages.** Over roughly the
first 93 % of the run the training CE is computed against a candidate set (every target in
the batch, and every prefix target, plus a per-step duplicate-free stride sweep of negatives)
instead of the full 50,304-way softmax: `P = 10,240` from step 0, 14,336 from 681, 24,576
from 961, off at 1101, with logged headroom of 34-61 % on the candidate budget. The batch's
own targets are always in the candidate set. **This biases the training gradient. Validation
is always the full 50,304-way softmax**, so the graded number is unaffected.

**A fixed-max LSE in the training CE kernel.** The softcap bounds the logit at `z ∈ (0, 23]`,
so the online block max in the CE kernel is replaced by that static bound and the sigmoid is
cached in fp16. Training-only; validation takes the eager full-softmax branch.

**Depth reduction and mixed-width attention.** Layer 7 is removed whole and layers 4 and 9
...
**The hashed n-gram table is 84,602,880 rows** (bigram + trigram channels), against record
#89's 377,280.

**Parameter count.** The transformer is the track's GPT-2-scale (124M-class) one. Beside it the
model carries 84,602,880 × 768 = **6.50e10 parameters** of hashed n-gram table plus 201,216 ×
768 = 1.55e8 of value embeddings, so the honest total is **≈65.3 billion parameters**, of which
>99 % is the table, sharded one eighth per rank and read one row-set per step.

**Table-size dose grid** (development tree, only the row count changed; 2 unseeded runs each, same 8×H100 box; supporting data for the EMBEDDINGS row of the ablation study above):

| rows | wall (s) | final val CE |
|---|---|---|
| 84,602,880 (this PR) | 39.52 (n=2, knobbed dev tree) | 3.2791 |
| 21,150,720 (¼) | 39.35 / 39.36 | 3.2825 / 3.2863 |
| 377,280 (record #89's size, this PR's two-channel hashing) | 39.00 / 38.98 | 3.3149 / 3.3152 |

The large table costs ~0.17 s of wall against the ¼-size one and is worth ~5 millinats of val there, and ~36 millinats against record #89's row count. The table is a lookup: no FLOPs, sparse row updates, never on the wire as a whole.

**Host-side GC and logging.** Carried over from ANVIL (part A): automatic garbage collection is
...
```