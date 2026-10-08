# C_q21 primary checks: arXiv HTML pages of three papers found in q05/q07/q18

- tool: WebFetch on arxiv.org html / abs pages; quotes below are what the fetch reader returned as verbatim
- fetched UTC: 2026-10-03 ~10:54-10:56

## 1. Why Adaptive Optimizers Underestimate Rare Tokens - https://arxiv.org/html/2609.37535v1
- author and date returned: Sangsidhya Kar (Presidency University, Kolkata), v1 September 29, 2026
- setup: "The model uses an embedding of width 64, a residual MLP block with LayerNorm and hidden width 256, a final LayerNorm, and an untied output layer with bias"; data "a first-order Markov chain on 2048 tokens"; V=4096 (half unused); "B=256"; "2×10⁴ steps with a 200-step warmup, and then a constant learning rate"; adaptive methods "learning rate 3×10⁻³, β₁=0.9 (0 for RMSProp)".
- rare tokens (<0.05 expected occurrences per batch), β₂=0.95, mean log(p̄/f): Adam -2.72, RMSProp -2.65, AMSGrad -0.14, SGD -0.22.
- test cross-entropy: Adam 3.603, RMSProp 3.594, AMSGrad 3.500, SGD 3.467.
- mitigations: "Several ways to reduce the bias: increasing β₂, increasing the batch size, using a running maximum of the second moment as in AMSGrad, sharing the second moment across the vocabulary as in Coupled Adam, or using SGD for the output bias."
- tied: "With tied embeddings, the common shift from Section 3 now affects the loss because it also shifts the input embeddings."

## 2. Leviathan - https://arxiv.org/abs/2601.22040 and https://arxiv.org/html/2601.22040v2
- authors: Reza T. Batley and Sourav Saha; v1 29 Jan 2026, v2 7 May 2026
- LEV: vocabulary "factorized into k components using a base-b decomposition, producing indices (i₁,…,iₖ)"; "shared codebooks C₁,…,Cₖ∈ℝ^(b×d_seed) to yield a seed representation z(i)=∑ᵣ₌₁ᵏCᵣ[iᵣ]"; then per-head projection, LayerNorm, sigmoid, "a univariate quadratic B-spline basis expansion with κ knots", and "Eᵢ=∑ℓ₌₁ʰWℓℳℓ(z̃ℓ)".
- params at 1.2B: standard embedding "V⋅D=410.4M"; LEV "2.25M parameters"; overhead "0.2%" relative to tied baseline.
- vocab 200,376 (o200k_base); training tokens 4.2B (200M), 8.4B (400M), 24.1B (1.2B).
- perplexity improvements returned: 200M 2.1%, 400M 5.6%, 1.2B 9.2%.
- abstract: gains "concentrated in rare tokens, where continuous parameterization reduces perplexity by 81%, falling to near zero for the most frequent."

## 3. Weight Tying Biases Token Embeddings Towards the Output Space - https://arxiv.org/html/2603.26663v1
- authors and date: Lopardo, Harish, Arnett, Gupta; March 27, 2026
- models: OLMo-1B tied and untied; GPT-Neo-2.7B vs Pythia-2.8B; Qwen3 4B tied vs 8B untied.
- method: "we train OLMo-1B with tied embeddings from scratch...applying gradient hooks to multiply input-layer gradients by a constant factor"; factors 2x, 5x, 10x.
- result at step 10,000 (20B tokens), 5x: cosine vs untied input 0.216 -> 0.222; vs untied output 0.384 -> 0.369; WikiText-2 perplexity tied 35.71 vs 36.64 with scaling ("no consistent performance gains").
