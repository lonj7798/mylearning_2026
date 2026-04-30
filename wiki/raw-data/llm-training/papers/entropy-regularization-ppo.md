<!-- scope: entropy bonus in A2C/PPO; carry-through into LLM RL
     deps: [[maximum-entropy-rl]]
     see-also: [[entropy-mechanism-llm-rl]], [[kl-control-rlhf]]
-->

# Entropy Regularization in A2C/PPO (Mnih 2016 → Schulman 2017)
- **Core Insight:** Adding a small entropy bonus `+ β · H(π)` to the actor's loss prevents premature convergence to a deterministic policy; it is the minimal exploration knob in on-policy RL and survives essentially unchanged in modern LLM RL loops.
- **Guideline:** Keep an entropy-bonus term in the PPO objective with a small coefficient (0.0 to 1e-2 depending on scale); monitor `H(π)` every step; if entropy collapses despite the bonus, escalate to target-entropy tuning or covariance-targeted fixes (Clip-Cov / KL-Cov).
- **Authors:** Volodymyr Mnih et al. ("A3C / A2C", 2016); John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov ("PPO", 2017)
- **Year:** 2016 / 2017
- **URL:** https://arxiv.org/abs/1602.01783 ; https://arxiv.org/abs/1707.06347
- **Relevant topics:** entropy bonus, PPO actor loss, A2C, exploration coefficient, GRPO inheritance

## Abstract
A3C (Mnih 2016) introduces a policy-gradient actor-critic with a simple exploration regularizer: `L_actor = −E[log π(a|s) · A] − β · H(π(·|s))`. PPO (Schulman 2017) inherits the term verbatim and adds the clipped surrogate objective. The total PPO loss is
  `L_PPO = L_CLIP − c_v · L_VF + c_H · H(π)`
with `c_H ∈ [0, 0.01]` typical on Atari and continuous control. The entropy bonus is widely observed to improve early-training stability and final score on hard-exploration tasks; it is preserved in every modern LLM-RL framework (TRL, OpenRLHF, verl, Reinforce++), though often with coefficient zero or with LLM-specific modifications.

## Key Contributions
- **Loss form (A3C/A2C):** `L = −E[A · log π] − β · E[H(π(·|s))] + (value-function term)`.
- **PPO form (Schulman 2017):**
  `L_PPO = E_t[min(r_t(θ) A_t, clip(r_t(θ), 1−ε, 1+ε) A_t) − c_v (V_θ − V_target)^2 + c_H H(π)]`
  with `ε = 0.2`, `c_v ≈ 0.5`, `c_H ≈ 0.01` as Atari defaults.
- **Empirical effect:** on Atari and MuJoCo, `c_H > 0` consistently helps exploration and reduces seed variance; too-large `c_H` hurts asymptotic performance.
- **LLM carry-through:** TRL `PPOTrainer` exposes `entropy_coef`; OpenRLHF and verl expose equivalent config. GRPO (DeepSeekMath) by default omits the bonus (relies on KL-to-reference), which is one of the reasons entropy collapse has become a named problem in LLM-RL (**[[entropy-mechanism-llm-rl]]**).
- **Link to max-ent (Haarnoja):** the bonus is the small-α limit of SAC's max-ent term — entropy-regularization in on-policy settings is a "poor man's" soft RL.

## Key Figures/Tables to Study
- **A3C Fig. 4 / ablation table** — entropy bonus vs without, across Atari games.
- **PPO Fig. 3** — learning curves showing the clipped surrogate; appendix lists the default entropy coefficient per environment.
- **"What Matters in On-Policy RL" (Andrychowicz 2020, arXiv:2006.05990)** — large-scale hparam sweep showing `c_H` is a second-tier hyperparameter: moderate effect, strong interaction with learning rate and advantage normalization.

## Technical Details
- **Entropy computed per step, per policy head:** `H(π(·|s)) = −Σ_a π(a|s) log π(a|s)`; for LLMs this is per-token over the vocabulary distribution.
- **Batching note:** entropy is averaged over tokens in a rollout batch; coefficient typical range in LLM RL: 0.0 (GRPO default) to 1e-3.
- **Interaction with advantage normalization:** if advantages are normalized, the effective entropy coefficient is rescaled; make sure tuning happens in the same regime.
- **Failure modes specific to LLMs:** because the vocabulary is huge, most tokens contribute near-uniform entropy; the collapse happens in the tail — the tokens that truly matter become sharply peaked while average entropy stays moderate. That is why a blanket entropy bonus under-corrects (documented in **[[entropy-mechanism-llm-rl]]**).

## Connections
- Direct ancestor of every entropy-related knob in LLM RL — GRPO, PPO-LLM, Reinforce++.
- Motivates target-entropy and covariance-aware alternatives in **[[entropy-mechanism-llm-rl]]**.
- Pairs with KL-to-reference in **[[kl-control-rlhf]]** — different regularizers, different failure modes.
- Theoretical cousin of max-ent RL (**[[maximum-entropy-rl]]**).
