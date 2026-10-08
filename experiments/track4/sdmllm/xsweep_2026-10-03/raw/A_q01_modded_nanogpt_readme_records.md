# A_q01 modded-nanogpt README record table

- Query: GET https://api.github.com/repos/KellerJordan/modded-nanogpt/readme (Accept: raw)
- Tool: python3 urllib, GitHub REST API unauthenticated
- Fetch: STATUS 200 UTC 2026-10-03T10:46:35.392717Z

## Verbatim track 1 rows 86-92 (last rows of the table)

```
87 | 1.256 minutes | [Faster Implementation of Relu^2 Kernel](https://x.com/classiclarryd/status/2083739041338630372) | 06/11/26 | [log](records/track_1_short/2026-06-11_RecursiveFromBest/this_pr/00088a48-30a3-4ebd-9768-6061011337f4.txt),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/322) | @cong_ml and AI System [Recursive](https://www.recursive.com/)
88 | 1.243 minutes | [Prefix token prediction auxiliary loss](https://x.com/classiclarryd/status/2083961001930834419) | 07/13/26 | [log](records/track_1_short/2026-07-13_PrefixTokenPrediction/prefix-1375/1b20ccf2-cb2f-4b6b-bc8a-2d9cd146f549.txt),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/337) | @jvarho
89 | 1.23 minutes | [MLP down projection in FP8 with efficient delayed scaling metric](https://x.com/classiclarryd/status/2086582390135406713) | 07/17/26 | [log](records/track_1_short/2026-07-17_FP8DownProjection/this_pr/11cb620c-daaf-4e85-83fc-258a5eb7ba09.txt),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/342)  | @Mister-dev-oss, @CerovazS, @MarioPaerle, @GabrieleCirillo, @crisostomi
90 | 1.13 minutes | 128 -> 96 dim QK, Fuse QK Norm, RoPE, and KeyOffset into Triton, Move MLP bwk to FP8, Move QKV fwd and bwk to FP8.  | 08/03/26 | [log](records/track_1_short/2026-08-03_FP8MLPBackwardPackedQKV),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/344)  | @theonlyglitch_
91 | 1.126 minutes | Mask logits for infeasible token continuations during validation.  | 08/06/26 | [log](records/track_1_short/2026-08-06-CanonicalMasking),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/350)  | @jvarho
92 | 0.665 minutes | Sampled softmax over the early stages,  full FP8 MLP fwd+bwd, 84.6M-row hashed n-gram table, ANVIL optimizer stack, depth reduction + mixed-width QK attention, full CUDA-graph capture of the training step | 08/30/26 | [log](records/track_1_short/2026-08-30_ANVIL2),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/360) | @DevenPzak
```

## Verbatim track 2 last row

```
18 | 17.35 minutes | Bulk transfer short track features | 12/31/25 | [log](records/track_2_medium/2025-12-31_BulkSmallTrackTransfer/354be270-7d41-44b7-8064-f040923f024f.txt),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/188) | -
```

## Reading

- The table ends at record 92 (0.665 minutes, 08/30/26, PR 360). No record after 92 is listed as of the fetch time.
- Track 2 (GPT-2 Medium) ends at record 18 (17.35 minutes, 12/31/25). No change in window.
