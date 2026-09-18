<!-- scope: Shumailov et al. (Nature 2024) — recursive training on model-generated data loses the tails of the original distribution (discrete/Gaussian theory; GMM, VAE, OPT-125m experiments)
     deps: [[self-instruct]]
     see-also: [[strong-model-collapse]], [[faithful-synth-eval]], [[synthetic-data-scaling-laws]], [[rephrasing-the-web]]
-->

# AI models collapse when trained on recursively generated data
- **Core Insight:** When each model generation is trained on data sampled from the previous generation, low-probability events disappear first and the fitted distribution loses variance; for recursive Gaussian fitting with a fixed sample size the fitted covariance converges to 0 almost surely (Theorem 3.1), and OPT-125m fine-tuned on its predecessors' wikitext2 outputs degrades in perplexity (Fig. 1b,c).
- **Guideline:** When a training pipeline reuses model-generated text across generations and the tails of the original distribution matter, keep access to original human-produced data, because in the paper's OPT-125m runs preserving 10% of the original data led to "only minor degradation" compared with retaining none (Fig. 1c vs 1b). Otherwise, expect loss of low-probability content even without any shift in the true distribution.
- **Authors:** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal
- **Year:** 2024 (Nature 631, 755–759; published 24 July 2024; received 20 October 2023). Preprint: arXiv:2305.17493 (v1 2023-05), titled "The Curse of Recursion: Training on Generated Data Makes Models Forget". Author Correction: Nature 640, E6 (21 March 2025).
- **URL:** https://www.nature.com/articles/s41586-024-07566-y
- **Source type:** paper
- **Relevant topics:** model collapse, recursive training on generated data, tail loss, synthetic-data risk, data provenance

## Abstract
The paper asks what happens to later language models once LLM-generated text makes up much of the text online. It finds that indiscriminate use of model-generated content in training causes irreversible defects in the resulting models, in which the tails of the original content distribution disappear. The authors call this effect "model collapse" and show that it can occur in LLMs, variational autoencoders (VAEs), and Gaussian mixture models (GMMs). They give theoretical intuition for the effect and argue that it applies to learned generative models in general. They conclude that data about genuine human interactions with systems will become more valuable as LLM-generated content enters crawled data.

## Key Contributions
- Definition 2.1: model collapse as a degenerative process across generations, with two cases. In early model collapse the model loses information about the tails; in late model collapse it converges to a distribution with little resemblance to the original, often with much reduced variance ("What is model collapse?").
- Three error sources that compound over generations: statistical approximation error (finite samples; the primary type), functional expressivity error, and functional approximation error ("What is model collapse?").
- Theory: a discrete distribution with exact approximation, modelled as a Markov chain whose only absorbing states are delta functions, and Theorem 3.1 for recursive Gaussian fitting ("Theoretical intuition"). The Supplementary Information (SI) adds a 1D Gaussian analysis, a multidimensional lower bound (SI §3.1), and regularized density estimation in a reproducing kernel Hilbert space, RKHS (SI §2, §3.2).
- Experiments: GMM and VAE trained from scratch (SI §4) and OPT-125m fine-tuned recursively on wikitext2 ("Model collapse in language models").

## Key Figures/Tables to Study
- **Fig. 1a:** schematic of the generational feedback loop. **Fig. 1b,c:** per-sequence perplexity histograms, scored by the generation-0 model, and mean perplexity per generation, with no original data retained (b) and with 10% retained (c).
- **Example 1:** OPT-125m outputs at generations 0, 1, 5, and 9 for the same input.
- **SI Fig. 3–5:** recursive 1D Gaussian fitting at several sample sizes (Fig. 3) and two data-pooling variants (Fig. 4, 5). **SI Fig. 6, 8:** GMM collapse over 2000 iterations. **SI Fig. 7:** VAE samples at generations 5, 10, and 20.

## Technical Details
**Process ("Theoretical intuition").** Dataset D_i at generation i has M_i i.i.d. samples from p_i. The model fit is `p_{θ_{i+1}} = F_θ(p_i)`. The next data distribution is `p_{i+1} = α_i p_{θ_{i+1}} + β_i p_i + γ_i p_0`, where α_i, β_i, γ_i ≥ 0 sum to 1 and are the shares of data from the new model, the previous generation, and the original distribution. The mathematical models use β_i = γ_i = 0. The numerical experiments use other values. The March 2025 Author Correction changed a typo that had printed "α_i = γ_i = 0" here.

**Discrete distribution, exact fit.** With sample size M, a state with probability q ≤ 1/M has fewer than 1 expected sample per generation, so information about it is lost. The chain converges with probability 1 to a delta function. The probability of ending at a given state equals that state's probability under the original distribution ("Discrete distributions with exact approximation").

**Gaussian (Theorem 3.1; SI §3.1.3).** For original data D_0 (not necessarily Gaussian) with non-zero sample variance, refit recursively with unbiased sample mean and variance and a fixed sample size: `E[W_2²(N(μ_n, Σ_n), D_0)] → ∞` and `Σ_n → 0` almost surely as n → ∞. W_2 is the Wasserstein-2 distance; μ_n, Σ_n are the fitted mean and covariance at generation n.

**1D Gaussian expansions (SI §3.1.1).**
`Var(X_j^n) = σ² (1 + n/M) + O(2)`   and   `E[R_W2^{n+1}] = (3/2) σ² (1/M_0 + 1/M_1 + … + 1/M_n) + O(2)` (SI eq. 6)
X_j^n is a sample at generation n; σ² is the variance of the original N(μ, σ²); M is a constant sample size; M_i is the sample size at generation i; O(2) is second order in 1/M_i; R_W2^{n+1} = W_2²(N(μ, σ²), N(μ_{n+1}, σ²_{n+1})).
Worked example: σ² = 1, M = 100, n = 10 gives Var(X_j^10) = 1 × (1 + 10/100) = 1.1. With M_0 = … = M_10 = 100, E[R_W2^11] = 1.5 × (11 × 1/100) = 0.165. The marginal variance grows because the fitted parameters follow a random walk; the authors state that the distance stays finite only if M_i grows superlinearly (SI §3.1.1).

**OPT-125m experiments ("Model collapse in language models").** The recipe ledger is below. Reported results:
- Fine-tuning on real wikitext2 gives 34 mean perplexity, from a zero-shot baseline of 115.
- With 5 epochs and no original data, training on generated data "allows us to adapt to the underlying task, losing some performance, from 20 to 28 perplexity points" (Fig. 1b).
- With 10 epochs and 10% original data, performance shows "only minor degradation" (Fig. 1c).
- Over generations, the generation-0 model assigns low perplexity to more of the generated sequences, and a longer high-perplexity tail appears. The authors read the tail as sequences the original model would never produce (Fig. 1 caption).
- A repetition penalty of 2.0 "causes the perplexity to double compared with the original", and models "remain as susceptible to model collapse, if not more" ("Ablation: Repetitions").

**GMM and VAE (SI §4.1).** A GMM separating two artificial Gaussians, with no cross-generational data and sampled 1000 times, misperceives the distribution within 50 iterations and reaches a low-variance point estimate by iteration 2000 (SI Fig. 6). A VAE trained on MNIST, with no original data in later generations, produces samples that mix digits by generation 20 (SI Fig. 7).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OPT-125m (Meta, Hugging Face) | 125M | SFT (causal-LM fine-tune) | data and generation | wikitext2; five-way beam search; 64-token blocks, next 64 tokens predicted per block; generated set same size as original training set | Nature 631:755 "Model collapse in language models" | verified 2026-09-14 | no ablation reported |
| OPT-125m | 125M | SFT (causal-LM fine-tune) | setting 1 | 5 epochs; no original data retained | same section, Fig. 1b | verified 2026-09-14 | compared with setting 2 (Fig. 1b vs 1c) |
| OPT-125m | 125M | SFT (causal-LM fine-tune) | setting 2 | 10 epochs; random 10% of original data points sampled each generation | same section, Fig. 1c | verified 2026-09-14 | compared with setting 1 (Fig. 1b vs 1c) |
| OPT-125m | 125M | eval-gate | base checkpoint for later generations | best model on the original task by wikitext2 validation set | same section | verified 2026-09-14 | no ablation reported |
| OPT-125m | 125M | SFT (causal-LM fine-tune) | seeds | 5 runs per experiment | same section | verified 2026-09-14 | not applicable |
| OPT-125m | 125M | SFT (causal-LM fine-tune) | learning rate, batch size, optimizer, total number of generations | not reported | checked: main text, Fig. 1 caption, SI §4 and Fig. 9–10 captions | not reported | not applicable |

## Findings relevant to generality and distillation
- The first information lost is about low-probability events, and later generations converge toward a point estimate with small variance ("Main" introduction; Definition 2.1). The authors note that low-probability events are often relevant to marginalized groups ("Discussion").
- The LLM setting trains each generation on text generated by the previous fine-tuned model. Fine-tuning "does not curb the effects of model collapse" ("Ablation: Repetitions").
- Self-distillation: the RKHS model is chosen because of "the similarity of model collapse to self-distillation"; the authors show that, in expectation, the density estimator collapses onto the topmost directions in the data (SI §2).
- Accumulation in the 1D Gaussian model: when all data from generations 1…n are pooled to fit model n + 1, variance in the estimates falls as the pool grows linearly (SI Fig. 5). When a fixed-size sample is drawn from the pool, the parameters "jump around quite significantly" (SI Fig. 4). The SI text states that the changes are "inevitable, although attenuated" in these variants (SI §3.1.1).
- Measurement: the paper reports mean perplexity per generation on the original wikitext2 test set and per-sequence perplexity histograms scored by the generation-0 model. The histograms show both a shift toward more probable sequences and a longer tail (Fig. 1 caption).

## Connections
- [[strong-model-collapse]] — later theoretical work on synthetic data within scaling laws; see that card for its claims.
- [[faithful-synth-eval]], [[synthetic-data-scaling-laws]] — later work on measuring and mixing synthetic data; not evaluated in this paper.
- [[rephrasing-the-web]], [[self-instruct]] — synthetic-data generation methods whose settings differ from the recursive retraining studied here.
- Code for all experiments: Zenodo https://doi.org/10.5281/zenodo.10866595 (ref. 13).

## Verification
- Checked on 2026-09-14 against: https://www.nature.com/articles/s41586-024-07566-y (HTML, updated version), its Supplementary Information PDF, and the Author Correction https://doi.org/10.1038/s41586-025-08905-3.
- Corrections to the previous card version:
  - "converges to a mode-collapsed near-Gaussian regardless of architecture" → the paper describes convergence to a delta function or a point estimate with very small variance ("Discrete distributions…"; Theorem 3.1). It does not say "near-Gaussian".
  - "prove — analytically for Gaussian mixtures / linear regression" → the theory covers discrete distributions, Gaussians, and RKHS density estimation (main text; SI §2–3). GMMs and VAEs are experiments only (SI §4). Linear regression is not analyzed; SI §2 cites Mobahi et al.'s RKHS regularized-regression analysis only as related prior work.
  - "Var[μ_k^{(n)}] ≈ n · σ²/N + O(model error)" → `Var(X_j^n) = σ²(1 + n/M) + O(2)` for 1D Gaussian refitting (SI §3.1.1).
  - "statistical sampling error" → "statistical approximation error".
  - "defines the curse of recursion" → that phrase is the arXiv preprint title, not a term defined in the Nature article.
  - URL field mixed the Nature article with arXiv:2305.17493 → the card describes the Nature version, and the preprint is listed separately under Year.
  - "by generation 9 the model emits nonsense while new-sample perplexity appears to improve" → Example 1 shows text with repeating phrases at generation 9. Fig. 1 shows more low-perplexity samples together with a longer high-perplexity tail.
  - "optionally mix with a fraction of real … up to 10 generations" → the mixed setting retains a random 10% of original data each generation. The total number of generations is not stated in the text.
- Removed as unsupported by the source: "First formal definition"; "Nature-stamped anchor reference"; "Within ~5 generations rare-token perplexity spikes while average perplexity looks fine"; "the paper deliberately studies the unfiltered case to isolate the statistical mechanism"; "accumulation + filtering avoids the worst case"; "a constant real-data fraction bounds error"; "verification with an external judge breaks the loop"; all "2025 follow-ups" claims (Dohmatob 1% contamination, Gerstgrasser et al., Zhu et al., He et al., Garg et al., Zhang et al. arXiv:2510.16657), which are not in this paper; the Connections claims about [[nemotron-4-synthetic]] and [[tulu-3-sft-mix]] anchors and the [[prismatic-synthesis]] "gradient manifold" clause.
- Not reported by the source: learning rate, batch size, optimizer, number of LLM generations, experiments on models larger than 125M parameters.
