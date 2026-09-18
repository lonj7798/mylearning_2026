<!-- scope: ΨPO, a preference-optimization family that contains RLHF and DPO; IPO is the identity-link member with a bounded squared loss
     deps: [[dpo]]
     see-also: [[simpo]], [[kto]], [[orpo]], [[bradley-terry-rm]], [[hf-dpo-zoo]], [[likelihood-displacement]]
-->

# A General Theoretical Paradigm to Understand Learning from Human Preferences
- **Core Insight:** When the empirical preference probability between two actions reaches 0 or 1, the DPO/RLHF objective drives the optimal policy to put probability 0 on the dispreferred action for every value of the KL coefficient τ, so the KL regularization stops binding; setting Ψ to the identity gives IPO, whose optimal policy stays at σ(0.5 τ⁻¹) in the two-action case and therefore returns to π_ref as τ grows (§4.2, §5.3.1).
- **Guideline:** When preference labels are close to deterministic or the action space is large relative to the dataset so some actions never win a comparison, use the IPO squared loss rather than the DPO log-sigmoid loss, because IPO regresses the log-ratio gap to a finite target τ⁻¹/2 while DPO's target is unbounded (§5.2, Eq. 17; §5.4). This paper tests the claim only in a three-action bandit, so the language-model case is not established here.
- **Authors:** Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos (Google DeepMind)
- **Year:** 2023 (arXiv v1 2023-10; v2 2023-11-22)
- **URL:** https://arxiv.org/abs/2310.12036
- **Source type:** paper
- **Relevant topics:** preference learning theory, ΨPO, DPO overfitting, KL regularization, offline contextual bandit

## Abstract
RLHF as commonly deployed rests on two approximations: that pairwise preferences can be replaced by pointwise rewards, and that a reward model fitted on those rewards generalizes to out-of-distribution data sampled by the policy. DPO removes the second approximation but keeps the first. The paper introduces ΨPO, an objective expressed directly in terms of pairwise preferences that avoids both, analyzes RLHF and DPO as special cases, and identifies their failure mode under deterministic preferences. It then takes Ψ to be the identity, deriving IPO, for which it gives an efficient offline optimization procedure, a uniqueness guarantee for the optimum, and illustrative bandit experiments where IPO does not exhibit DPO's degenerate solutions.

## Key Contributions
- Defines ΨPO for a non-decreasing Ψ : [0,1] → ℝ and shows RLHF and DPO are the special case Ψ(q) = log(q/(1−q)) when the Bradley-Terry model holds (§4.1, Eq. 6, Proposition 1).
- Identifies the weak-regularization failure: as preferences become deterministic, the KL term's effect vanishes and DPO's optimal policy assigns probability 0 to the dispreferred action for any τ (§4.2).
- Derives IPO by re-expressing the Ψ = identity objective as a root-finding problem, then as a single squared-error objective L(π) (§5.1, Eq. 13).
- Proves uniqueness: under Supp(µ) = Supp(π_ref), L has a unique local and global minimum in the set of policies with support equal to Supp(µ), namely π* (Theorem 2).
- Gives the sampled IPO loss and Algorithm 1, using both orderings of each pair to reduce variance (§5.2, Eq. 17).
- Runs three bandit examples showing DPO ignoring π_ref and IPO tracking it as τ changes (§5.3, §5.4, Figures 1–2).

## Key Figures/Tables to Study
- **Figure 1** — action-probability learning curves for IPO and DPO on dataset D₁ = {(y_a,y_b), (y_b,y_c), (y_a,y_c)} (a total ordering over three actions), across values of τ.
- **Figure 2** — the same comparison on D₃ = {(y_a,y_b), (y_b,y_a)}, where the pair (y_a, y_c) is never observed.
- **Proposition 1 and §4.2** — the derivation that DPO's regularization weakens as preferences approach {0,1}.
- **Eq. 17 and Algorithm 1** — the implementable loss.

## Technical Details

### ΨPO objective (§4, Eq. 6)
`max_π  E_{x~ρ, y~π(·|x), y'~µ(·|x)} [ Ψ( p*(y ≻ y' | x) ) ] − τ · D_KL(π || π_ref)`
- `Ψ` — a non-decreasing function [0,1] → ℝ.
- `p*(y ≻ y'|x)` — the true human preference probability, the expectation over raters of the indicator that y is preferred to y'.
- `µ` — the behavior policy that generated both candidate actions.
- `τ ∈ ℝ⁺` — the KL regularization coefficient; larger τ keeps π closer to π_ref.
- The expectation is over actions drawn from π and µ, not over a fixed preference dataset.
- Under the Bradley-Terry assumption, Ψ(q) = log(q/(1−q)) makes the optimizers of ΨPO, of the RLHF objective, and of DPO identical (Proposition 1, §4.1).

### Weak-regularization result (§4.2)
- For two actions with p*(y ≻ y') = 1, satisfying the optimality condition requires the reward gap r(y) − r(y') → +∞, so π*(y') = 0 for any τ.
- The paper notes this also arises in the finite-data regime: if the true preference is 0.8 but few samples give an empirical p̂ = 1, the empirical optimum sets π(y') = 0 regardless of τ. It states this is expected to matter most when context and action spaces are large, as for language models (§4.2, Interpretation by the authors).
- The paper's explanation for RLHF's relative robustness is that the reward model is underfit when preference probabilities are in {0,1}, and that underfitting supplies the regularization DPO loses by not training a reward model (§4.2).

### IPO loss (§5.1–5.2)
`h_π(y, y') = log[ π(y) π_ref(y') / ( π(y') π_ref(y) ) ]`
`L(π) = E_{y,y'~µ} [ ( h_π(y, y') − ( p*(y ≻ µ) − p*(y' ≻ µ) ) / τ )² ]`  (Eq. 13)
Sampled form over a dataset D of (y_w, y_l) pairs, after exploiting the symmetry of each pair:
`E_{(y_w, y_l)~D} [ ( h_π(y_w, y_l) − τ⁻¹/2 )² ]`  (Eq. 17)
- `y_w`, `y_l` — the preferred and dispreferred actions of a pair.
- `τ⁻¹/2` — the fixed regression target; it is finite for every τ > 0.
- The paper's reading: IPO regresses the gap between `log(π(y_w)/π(y_l))` and `log(π_ref(y_w)/π_ref(y_l))` to τ⁻¹/2, so weaker regularization means a larger target log-ratio (§5.2).

### Two-action asymptotic comparison (§5.3.1)
- Setting: two actions, p*(y₁ ≻ y₂) = 1, uniform π_ref and µ.
- DPO converges to π*(y₁) = 1, π*(y₂) = 0 for every τ.
- IPO: p*(y₁ ≻ µ) = 3/4 and p*(y₂ ≻ µ) = 1/4, so π*(y₁) = σ(0.5 τ⁻¹) and π*(y₂) = σ(−0.5 τ⁻¹). As τ → ∞, π* → π_ref; as τ → 0, π* → the deterministic policy.
- Check by hand at τ = 1: π*(y₁) = σ(0.5) ≈ 0.622, not 1.

### Sampled-preference examples (§5.4)
- Action space Y = {y_a, y_b, y_c}; policies parameterized as π_θ(y_i) = softmax(θ)_i with θ ∈ ℝ³.
- Two failure patterns for DPO: an action that wins against all others is pushed to probability 1 regardless of τ; an action that never wins is pushed to 0 regardless of τ. IPO instead moves toward π_ref as τ grows.
- The authors note the never-wins case is the more common one, because in a large action space with a small dataset some actions are sampled rarely or once (§5.4, Interpretation).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Tabular softmax bandit policy (no language model) | 3 parameters (θ ∈ ℝ³) | preference | optimizer / LR / mini-batch / steps | Adam / 0.01 / 9 / 18,000 | arXiv:2310.12036 §5.4 | verified 2026-09-18 | no ablation reported |
| same | — | preference | seeds and reporting | 10 seeds per hyper-parameter set; mean and 95% confidence intervals | arXiv:2310.12036 §5.4 | verified 2026-09-18 | — |
| same | — | preference | implementation / hardware | flax + optax; cloud VM with 4 cores and 32 GB RAM | arXiv:2310.12036 §5.4 | verified 2026-09-18 | — |
| — | — | preference | τ values used in Figures 1–2 | not reported in the text (figures show "varying values of τ") | checked §5.3, §5.4, figure captions | not reported | — |
| — | — | preference | language-model training settings (LR, batch, epochs, π_ref) | not reported; the paper runs no language-model experiment | checked §1–§6 | not reported | §6: "Future works should scale those experiments to more complex settings such as training language models on human preferences data" |

## Findings relevant to negative feedback
- The mechanism the paper isolates is negative-as-gradient: the dispreferred action's probability is driven to 0 under DPO whenever it never wins in the dataset, including when it was simply never compared enough (§5.4). This is a degenerate outcome, not a measured downstream regression — the paper reports no task metric.
- IPO's control is a bounded regression target rather than a clipped or weighted push-down: the log-ratio gap is regressed to τ⁻¹/2, which limits how far the rejected side can be suppressed for a given τ (§5.2).

## Connections
- [[dpo]] — the logit-link special case of ΨPO (Proposition 1).
- [[bradley-terry-rm]] — the pointwise-reward assumption ΨPO is built to avoid (§1, §4.1).
- [[simpo]], [[orpo]], [[kto]] — later preference objectives that change the loss shape or drop π_ref.
- [[likelihood-displacement]] — later analysis of what happens to chosen and rejected log-probabilities under these losses.
- [[hf-dpo-zoo]] — practitioner comparison that reports language-model results for IPO, which this paper does not.
- [[on-off-policy-rlhf]], [[policy-coverage-loss]], [[trl-online-dpo]], [[openrlhf-dpo]] — work and code that instantiate the IPO loss.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2310.12036 (arXiv v2, 2023-11-22)
- Corrections to the previous card version:
  - ΨPO written as `E_{x,y_w,y_l}[Ψ(p*)]` over a preference dataset → Eq. 6 takes the expectation over `x~ρ, y~π(·|x), y'~µ(·|x)`.
  - "Figure 2: 2-action toy showing DPO policy drifting to assign probability 1 to y_w" → Figure 2 is the three-action dataset D₃ with an unobserved pair; the two-action case is the closed-form argument in §5.3.1 and has no figure. The three-action total-ordering run is Figure 1.
  - "Proves that when the preference dataset has P(y_w ≻ y_l) → 1, DPO's loss becomes unbounded" → the paper's statement is that the optimal reward gap diverges and the optimal policy becomes deterministic for any τ, so the KL term stops constraining (§4.2).
  - "yields a bounded mean-squared error loss" → the loss is a squared error whose regression target τ⁻¹/2 is finite; the paper does not claim the loss itself is bounded.
  - Card gave no source type and no loci; both added.
- Removed as unsupported by the source:
  - The hyperparameter table ("τ 0.01–0.5 (paper: 0.005, 0.01, 0.1)", "Learning rate 5e-7 (same as DPO)", "π_ref SFT, frozen", "Batch size 32–64 pairs", "Epochs 1–3"). The paper's only reported optimization settings are the bandit run in §5.4 (Adam, LR 0.01, mini-batch 9, 18,000 steps); no τ values are printed.
  - "Less sensitive to β/τ sweep than DPO."
  - "Better preserves chosen-side log-probabilities (DPO drops both sides, IPO mostly drops rejected)." The paper reports action probabilities in a three-action bandit, not chosen/rejected log-probabilities of a language model.
  - "Slightly under-performs DPO on noisy preferences, outperforms on clean / distilled."
  - "Switch from DPO to IPO when the preference dataset has near-deterministic labels (distilled or BoN-gated); tune τ rather than β, and expect smaller log-prob drift on the chosen side" — the distilled/BoN framing and the log-prob-drift expectation are not in the paper; the guideline was rewritten to the tested claim.
- Not reported by the source: any language-model or text experiment; any benchmark score; the τ values plotted in Figures 1–2; guidance on converting τ to the β used in DPO implementations.
