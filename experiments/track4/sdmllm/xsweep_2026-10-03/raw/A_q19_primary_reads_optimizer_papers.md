# A_q19 primary-source reads of the optimizer and head papers found in A_q06 to A_q09

- Tool: WebFetch of arxiv.org abstract pages (and one HTML body page)
- Fetch UTC: 2026-10-03 between 10:51 and 10:53

## 2610.01395 AF-Muon: An AdamW-Free Muon Optimizer for Tied-Embedding Models

- https://arxiv.org/abs/2610.01395, submitted 2026-10-01. Authors: Arash Lagzian, Paniz Halvachi, Junming Zhang, Zhouhan Lin, Dianbo Liu.
- Verbatim: "AF-Muon therefore trains every parameter class with a single first-moment buffer and no second-moment state, saving around 20% optimizer-state memory relative to Hybrid Muon in our benchmark. Across nine tied-token settings - decoder-only language models from 124M to 1B parameters, a fully shared T5-style encoder-decoder, and ImageGPT-style image-token, protein, and sparse-MoE variants, spanning text, image, and protein-sequence data - AF-Muon improves mean validation loss and perplexity over both Hybrid Muon and a SCION-style Sign endpoint."
- The abstract gives no size of the loss improvement. The fetch summary also reported "about 1% step-time overhead in matched training"; that phrase was returned by the fetch tool and was not seen in the abstract text quoted above, so it is not used.
- Code: https://github.com/arashlagzian/afmoun (surfaced by A_q17; not opened).

## 2609.37535 Why Adaptive Optimizers Underestimate Rare Tokens

- https://arxiv.org/abs/2609.37535, submitted 2026-09-29. Author: Sangsidhya Kar.
- Verbatim: "For RMSProp with periodic arrivals, we can solve the fixed point in closed form: if a token is absent for at least two consecutive minibatches, its equilibrium probability is strictly below its data frequency for every learning rate, and the ratio tends to κ/(2(e^κ/2−1)). Here κ is the mean number of steps between occurrences divided by the second-moment time constant 1/(1−β₂). In the same model, SGD and AMSGrad retain the unbiased fixed point."
- Verbatim: "We test these predictions both in a unigram model and in a small language model trained from a known generating distribution. With random arrivals, the bias is larger than the periodic formula predicts; in the language model, the optimizers with the biased fixed point also fit the generating distribution less well."
- Verbatim: "every method whose update is linear in past gradients does, as do Kronecker-factored and orthogonalized methods such as Shampoo and Muon. Adam, Adafactor, Lion, and sign descent do not"

## 2609.04577 Optimizer Memory Schedules for Outscaling the Overtraining Axis

- https://arxiv.org/abs/2609.04577, submitted 2026-09-04. Authors: Katie Everett, Shikai Qiu.
- Verbatim (abstract, from A_q06 Atom text): "We compare these four optimizers across models from 51M to 253M parameters and overtraining (OT) factors from 1x to 256x, sweeping the base learning rate at every setting. The preferred learning rate schedule can reverse across the overtraining axis, the best weight decay coefficient scales approximately as sqrt(OT), and longer horizons generally favor longer fixed memory."
- Verbatim (from fetch): "Muon and SOAP instead provide roughly constant token-efficiency advantages over AdamW across most of the measured range"

## 2607.23777 Scale Weight Decay and Train Better

- https://arxiv.org/abs/2607.23777, submitted 2026-07-26. Author: Anuj Apte.
- Verbatim: "we propose to scale weight decay by the fraction of the peak learning rate η/η_max."
- Verbatim: "When applied to the training of mixture-of-experts models, Muon with scaled weight decay (Muon-SW) consistently outpaces Muon with identical hyperparameters, reaching the same validation loss 30% faster at our largest scale across models from 72 - 930 million parameters trained at ~600 tokens per active parameter."
- Body (https://arxiv.org/html/2607.23777v1), verbatim equations as returned: baseline "W_{t+1}=(1−η_t λ)W_t−η_t U_t"; proposal "W_{t+1}=(1−η_t²λ/η_max)W_t−η_t U_t". So the baseline already couples decay to η_t as torch AdamW does, and the proposal adds one more factor η_t/η_max.
- Independent echo: modded-nanogpt PR 369 (A_q03), verbatim: "On Muon I also use Defazio-style weight decay: the weight-decay coefficient follows the learning-rate schedule, so decay anneals with the step size instead of staying fixed."

## 2609.16179 Z-Loss Backward Geometry in Dense Output Heads and Sparse Routers

- https://arxiv.org/abs/2609.16179, submitted 2026-07-31. Author: Bum Jun Kim.
- Verbatim: "These factors include common-shift coordinates, tied-embedding pathways, output-to-hidden gain, fused-loss source consistency, optimizer-facing updates, and top-k router reduction scale."
- Verbatim: "Across evaluations of models in the GPT-2 and Pythia family on WikiText-103 and FineWeb-Edu, architecture-aware variants reduce backward-geometry tails while maintaining comparable validation perplexity in low-coefficient regimes."
- No loss improvement claimed; the claim is diagnostic.

## 2608.22253 Toward a First-Principles Update Geometry for the Language-Model Head

- https://arxiv.org/abs/2608.22253, v1 2026-08-23, v2 2026-09-09. Authors: Aditya Somasundaram, Charles Guille-Escuret, Alexander Moreno, Zhengzhong Liu, Eric Xing.
- Verbatim: "With Muon on the backbone, experiments across three seeds at 190M, 380M, and 640M parameters show that RowNorm reduces mean final step diameters and empirical Hilbert RMS perturbations by factors of $45$--$60$ and $12$--$15$, respectively, with only a $0.0057$--$0.0153$ increase in mean final validation loss."
- Reading: the proposed head geometry costs validation loss in their runs; it is a stability tool, not a loss win.
