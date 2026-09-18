<!-- scope: theory of model collapse inside the scaling-laws regime; a fixed synthetic fraction removes the benefit of more data
     deps: [[model-collapse]]
     see-also: [[prismatic-synthesis]], [[rephrasing-the-web]], [[synthetic-data-scaling-laws]]
-->

# Strong Model Collapse
- **Core Insight:** In a regression setting analysed in the proportionate scaling limit, the test error on the real distribution carries an extra term `ζ` whose leading part scales as `p₂²`, the squared fraction of synthetic data; unless `p₂ → 0` the error plateaus above the real-data-only baseline, so larger training sets stop helping even when synthetic data is as little as 1% of the pool (Abstract; §3.1 Thm. 1; §3.2 Thm. 2).
- **Guideline:** When a fraction of the pretraining pool is model-generated and cannot be identified and removed, do not expect a fixed mixing weight to recover the clean scaling curve: in the paper's isotropic under-parameterized analysis the optimal mixing coefficient `α*` goes to 0, that is, discard the synthetic data (§5.1 Cor. 2, Eq. 18; Fig. 9).
- **Authors:** Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, Julia Kempe (FAIR at Meta; NYU Center for Data Science; NYU Courant; UCLA)
- **Year:** 2024 (arXiv v1 2024-10; v2 2024-10-08)
- **URL:** https://arxiv.org/abs/2410.04840
- **Source type:** paper
- **Relevant topics:** model collapse, neural scaling laws, random-projection models, operator-valued free probability, synthetic-data fraction

## Abstract
The paper studies supervised regression on a training set mixing real data (fraction `p₁`) and synthetic
data (fraction `p₂`), evaluated on the real distribution. For the classical linear model and then for a
random-projections model whose output dimension `m` stands in for model size, the authors derive the test
error as `E + ζ`, where `E` is the usual bias-variance expression for clean data and `ζ` is an extra term
caused by the mismatch `∆` between the real and synthetic regression targets. They call the result strong
model collapse: any non-vanishing `p₂` leaves `ζ` bounded away from zero, so larger training sets do not
improve test error. They further show the effect of model size is non-monotone: below the interpolation
threshold `m = n` larger models amplify collapse, past it larger models mitigate but do not remove it.
Predictions are checked on random projections with Gaussian data, two-layer networks on MNIST, and GPT-2
models trained on BabiStories.

## Key Contributions
- Theorem 1: test error of the classical linear model on a real/synthetic mixture, `E_test ≃ E + ζ`, with
  `ζ = p₂²·(1 + p₁u)·tr ∆Σ³(Σ + κI)⁻² + p₂·u·tr ∆Σ(p₁Σ + κI)²(Σ + κI)⁻²` (§3.1, Eq. 9-10).
- Theorem 2: the same decomposition for the random-projections model `f(x) = v'Sx`, proved with
  operator-valued free probability theory (§3.2, Eq. 11; Appendix D).
- The strong-collapse statement: in the scaling-laws regime `ϕ = d/n → 0⁺`, `ζ ≃ p₂²·tr ∆`, so collapse is
  not removed by naively mixing synthetic with real data (§3.1).
- A double-descent picture for collapse in model size, with the sign of the model-size effect flipping at
  the interpolation threshold `m = n` (§3.2, Figs. 1, 2, 4).
- Analysis of two mitigation schemes: weighted single-step mixing (§5.1) and iterative mixing (§5.2).

## Key Figures/Tables to Study
- Figure 1: Pareto diagram of mixture test error against real-data-only, coloured by parameterization
  rate `ψ = m/n`, for four synthetic-quality levels `c² ∈ {0, 0.1, 0.5, 1}`.
- Figure 4: test error against network width `m` for mixing ratios `p₂` and quality levels `c²`.
- Figure 8: BabiStories GPT-2 results — loss against tokens for `p₂` from 0.0 to 1.0 (left), and for
  12/18/24 layers at `p₂ = 1` (right).
- Figure 9: failure of weighted mixing. Figure 10: iterative against single-step mixing.

## Technical Details
- Setting: real samples from `P₁` with target `w₁*` and noise `σ₁²`, synthetic from `P₂` with `w₂*` and
  `σ₂²`; `∆ := cov(w₁* − w₂*)` is the displacement matrix; `p₁, p₂` are the fractions of the `n` training
  samples; the limit is proportionate scaling with `ϕ = d/n` (§2).
- When `∆ = 0` the extra term vanishes and Theorem 1 reduces to the standard ridge decomposition (§3.1).
- Strong collapse in the scaling regime: for `ϕ → 0⁺`, `ζ ≃ p₂² tr ∆`; if `tr ∆` stays bounded away from
  zero so does `ζ`, unless all synthetic data is discarded (§3.1).
- Model size: the first term of `ζ` in Theorem 2 is lower-bounded by `p₂² tr ∆Σ³(Σ + θI)⁻²`, a factor that
  depends only on design choices through `θ`, which is why different model sizes give different collapse
  profiles (§3.3). Bias, variance, and `ζ` all diverge at `m = n` in the unregularized limit `λ → 0⁺`
  (§3.3, Fig. 2).
- Figure 1 setup: `d = 600`, isotropic `Σ = I`, `n = 500` total training samples, `∆ = (c²/d)Σ⁻¹`; the
  leftmost panel uses `n₂ = 50` synthetic against `n₁ = 450` real samples (§3.3).
- Language-model experiment: BabiStories, a TinyStories reproduction generated with Mixtral-8x7B; a
  GPT-2-small of 124M parameters trained on it is used as the generator of the synthetic split, which is
  then mixed back with the original data; perplexity is evaluated on a validation split of the original
  BabiStories (§4.2). Figure 8 sweeps `p₂ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0}`
  at 12 layers, over roughly 6×10⁸ to 4×10⁹ tokens.
- Model-size experiment: 12 layers (124M), 18 layers (166M), 24 layers (204M), embedding dimension and
  head count held constant, trained on synthetic data only (`p₂ = 1`) with a synthetic set 10× the size of
  the original. Larger models keep lower loss below about 1×10¹⁰ tokens; beyond about 3×10¹⁰ tokens the
  smaller models reach lower loss (§4.2, Fig. 8 right).
- Weighted single-step mixing (§5.1): with weights `(1 − α)/p₁` on real and `α/p₂` on synthetic samples,
  `E_test ≃ p₂²α²c² + ((1 − α)p₁σ₁² + αp₂σ₂²)ϕ + O(ϕ²)` (Cor. 2, Eq. 17), a U-shaped function of `α`
  minimized at `α* = clip_[0,1](1 − ((p₁σ₁² − p₂σ₂²)/(2c²))ϕ)` (Eq. 18). For `tr ∆ = Ω(1)` and bounded
  noise, `α* → 0`; any fixed `α ∈ (0,1]` gives `E_test ≥ p₂²α²c²`.
- Iterative mixing (§5.2, after Ferbach et al. 2024): with `t` rounds of regenerating synthetic data from
  the previous model, `E_test ≃ E/(1 − p₂²) + Θ(p₂^{2t})` (Cor. 3, Eq. 19). Taking `t` of order
  `log(n/d)` restores a scaling law proportional to `E`, but in Figure 10 (5 runs, `p₁ = p₂ = 0.5`)
  iterative mixing is consistently worse than training on the real portion alone, which the authors read
  as the procedure neutralizing rather than exploiting the synthetic data (Interpretation).
- Iterative mixing with vanishing real data: as `p₂ → 1`, for any `t ≥ 1`, `E_test ≃ c₀² + tE`, which grows
  with `t` (§5.2).

## Findings relevant to generality
- The conclusion the authors draw for practice is to preserve and label real data, by curating it or by
  avoiding unintended synthetic data in training (§6).
- The analysis is a regression setting with a fixed displacement `∆`, and the language-model evidence is
  GPT-2 at 124M-204M parameters on a synthetic children's-story corpus. No result is reported for filtered
  or verifier-checked synthetic data, for instruction tuning, or for current pretraining scale.

## Connections
- [[model-collapse]] — the recursive-training result this paper extends into the scaling-laws regime.
- [[synthetic-data-scaling-laws]] — empirical scaling measurements on synthetic mixtures.
- [[prismatic-synthesis]], [[rephrasing-the-web]] — synthetic-data methods this theory is not evaluated on.
- [[faithful-synth-eval]], [[nathan-lambert-synthetic-data]] — cards citing this one for contamination.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2410.04840 (arXiv v2, 8 Oct 2024, full PDF text)
  and the arXiv abstract page.
- Corrections to the previous card version:
  - "Aligned with independent 2025 work (Gerstgrasser, Zhu, He, Garg) that accumulation + filtering is
    the escape" → the paper does not cite Gerstgrasser et al. at all. The accumulation paper is
    Gerstgrasser, Schaeffer, Dey, Rafailov, et al., arXiv:2404.01413, 2024. This paper's own mitigation
    analysis covers weighted single-step mixing and iterative mixing, and reports both as insufficient
    (§5.1-5.2).
  - "Strong Model Collapse (Dohmatob et al., ICLR 2025 Spotlight)" as the title → the artifact title is
    "Strong Model Collapse"; the arXiv abstract page carries no venue comment, so the venue is not stated
    here.
  - "Authors: ... (Meta / NYU)" → affiliations are FAIR at Meta, NYU Center for Data Science, NYU Courant,
    and UCLA; Feng and Subramonian did the work while interning at Meta.
  - "`E[R_test] ≈ f(N) + c(p)·σ_synth²`" → the paper's form is `E_test ≃ E + ζ` with `ζ` given in Eq. 10
    (linear model) and Eq. 11 (random projections); the leading dependence is `p₂² tr ∆`, which is a
    target displacement, not a synthetic-noise variance.
  - "GPT-2-scale LM training with 1% synthetic injection" → Figure 8 sweeps eleven values of `p₂`, of
    which 0.01 is one; the model-size panel uses `p₂ = 1`.
- Removed as unsupported by the source: "ICLR 2025 Spotlight"; "load-bearing reference for the theoretical
  side of the 2025 synthetic-data-risk debate"; "1% threshold is regime-dependent — for very small
  training sets or weak models, tolerance is higher"; "high-quality filtered synthetic (e.g., via an
  external verifier) behaves like real in the limit"; "Provides the theoretical basis for 2025-era
  'real-data anchor' / 'verifier-in-the-loop' recipes"; the [[prismatic-synthesis]] and
  [[rephrasing-the-web]] interpretations attributed to this paper.
- Not reported by the source: optimizer, learning rate, batch size, and token budget of the GPT-2 runs in
  the main text (deferred to Appendix A.3); any experiment on instruction tuning, post-training, or
  models above 204M parameters; any measurement of a filtering or provenance-detection procedure.
