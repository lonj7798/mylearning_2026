---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/perplexity-correlations-data-selection.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2409.05816
created_at: "2026-09-15"
---

# Excerpt: Improving Pretraining Data Using Perplexity Correlations

**Authors:** Tristan Thrush, Christopher Potts, Tatsunori Hashimoto (Stanford University)
**Version read:** arXiv:2409.05816v2 (10 Mar 2025), ICLR 2025; v1 September 2024 (v1 contains the preregistration).
**Status:** no library card existed for this slug on 2026-09-15; values read in the v2 PDF text.

## Model and estimator
- Single-index model (Eq. 1): y_i = f(⟨θ*, x_i⟩ + ε_i), where y_i is benchmark error of model i, x_i ∈ R^D the per-domain bits-per-byte losses, f monotone increasing, θ* weights over D domains (§3).
- Proposition 1: for non-negative θ*, minimizing pretraining loss under sampling distribution θ* minimizes expected downstream error under the model.
- Estimator (§4.1): γ_j = Σ_{k≠l} sign(y_k − y_l)(rank_j(x_{k,j}) − rank_j(x_{l,j})). Eq. 6 writes the same U-statistic with the empirical CDF Φ.
- Projection (Theorem 2): select domains from largest to smallest estimate, taking each domain's full token count τ_i m, until the budget m is filled; the boundary domain receives the remainder.
- A fastText classifier (bigram features from DCLM) trained on selected vs unselected domains then filters at page level (§4.1).

## Setup and results
- 90 Open LLM Leaderboard models (≈33M to ≈9B parameters, App. K Fig. 9); RedPajama-V2 sample, 9,841 domains with ≥ 25 pages (§5).
- Initial experiments (§5.1): Pythia-160M configuration, 3.2B tokens; targets LAMBADA (EN, IT, FR, DE, ES), ARC Easy, PIQA, SciQ. Table 1 average rank over 8 evals: none 3.750, language filter 4.000, DSIR 4.500, DCLM fastText + EN filter 3.750, without language filter 3.250, + manual language filter 1.375, perplexity correlations 1.750.
- App. G Table 2 (160M): per-device batch 128 on 4 A100, LR 5e-3, warmup ratio 0.1, Adam (0.9, 0.95, 1e-8), WD 0.1, cosine, grad-norm 1.0.
- Preregistered experiments (§5.2, App. N, Fig. 3): DCLM 1B-1x track (1.4B model, 28.8B tokens from a 10% sample of the 1.64T pool). Target "DCLM Core". Raw pool: large benefit growing with scale. Pre-filtered pool: no benefit; coefficients ranged .23 to .33 with most above .29 "so we could have predicted no or small gains before pretraining"; raw pool coefficients ranged −.07 to .32.
- §5.3 / Fig. 4: models trained on unusual data (for example Phi, multilingual, code, GPT-4 outputs) are predicted less accurately.
- App. O: top-correlated domains for ARC Easy include api-bridge.azurewebsites.net, www.aaeoptometry.com, www.akronchildrens.org; for DCLM Core (raw pool) nrich.maths.org, serc.carleton.edu, www.metoffice.gov.uk, au.finance.yahoo.com.
- App. N, Fig. 11: 410M, 8.2B tokens: perplexity correlations duplicate the original 3.2B selected tokens while uniform sampling uses 8.2B unique tokens.

## How ch-10a uses it
§5.3 (benchmark-correlated selection, worked example of γ), §6 (target-shaped objectives), §7 (pre-filtered pool yields no gain).
