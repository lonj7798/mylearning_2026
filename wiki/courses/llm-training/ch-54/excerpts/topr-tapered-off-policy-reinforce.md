<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# Tapered Off-Policy REINFORCE: Stable and efficient reinforcement learning for LLMs
- **Artifact:** Nicolas Le Roux, Marc G. Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alex Fréchette, et al. (Mila; Reliant AI). arXiv:2503.14286 (v1 2025-03; text read from v2, 2025-03-19). Source type: paper.
- **Core Insight:** Applying importance sampling only to negative-reward trajectories, with the ratio truncated to [0, 1], and leaving positive trajectories at weight 1 (an SFT-style update) keeps off-policy REINFORCE stable without KL regularization; on GSM8K with Llama 3 8B, naive REINFORCE degrades and PPO stops improving as training becomes more off-policy, while TOPR keeps improving (§3, Eq. 8; Fig. 1 left).
- **Guideline:** When training on trajectories sampled from an older or different policy and negatives are kept, downweight each negative by `min(π(τ)/µ(τ), 1)` and leave positives unweighted, because removing negatives entirely wastes inference and keeps π close to µ, while un-truncated importance sampling has variance proportional to the squared ratio (§2.2, §2.3, §3).

## Technical details (with loci)
- **General family (§3, Eq. 7).** The gradient is written with two truncation windows, `[a₊, b₊]` for positive trajectories and `[a₋, b₋]` for negative ones, applied to the ratio `π(τ)/µ(τ)`; `µ` is the behavior policy, `π` the policy being trained, `R(τ)` the trajectory reward. Different choices of the four limits recover existing methods (Table 1).
- **Canonical TOPR (§3, Eq. 8).** `a₋ = 0` and every other limit 1, giving
  `∇J_topr(π) = Σ_{τ∈T⁺} µ(τ)R(τ)∇log π(τ) + Σ_{τ∈T⁻} µ(τ)·clip(π(τ)/µ(τ), 0, 1)·R(τ)∇log π(τ)`,
  described in the paper as "SFT update for positive examples" plus "TIS update for negative examples".
- **Pseudo-code (Fig. 1 right, Algorithm 1).** `α = clip(π(y|x)/µ(y|x), 0, 1) if R(x, y) < 0 else 1`; `ℓ = stop-grad(α)·r(x, y)·log π(y|x)`.
- **Why `a₋ = 0` (§3).** Any `a₋ > 0` places a floor under the weight of negative trajectories, which the paper argues must eventually lead to model degeneracy, as with naive REINFORCE (§2.1). `a₊ > 0` guarantees a minimum learning rate on positives and accelerates them when they are unlikely under π.
- **Setting (§4.1).** Llama 3 instruction-tuned models, 8B unless stated; GSM8K (7,473 train / 1,319 test) and MATH; n = 16 candidate solutions per training question with 8-shot chain-of-thought prompting.
- **Baseline parameter (Abstract, §1, §4).** The paper reports that REINFORCE's baseline acts in the off-policy regime as a control on the positive/negative composition of the dataset rather than only as a variance-reduction term, and that choosing it well requires more than the mean return.
- **Stated scope (§4).** Results are single-iteration and fully offline, aiming to characterize stability, plus multi-iteration experiments in which 8B models match 70B-model benchmark performance (Abstract).

## Not reported
No asynchronous-system experiment; no evaluation of pass@k or out-of-domain retention.
