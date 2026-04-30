<!-- scope: batching multiple prompts across rollouts in LLM RL
     deps: [[ppo]], [[grpo]]
     see-also: [[rloo]], [[async-rollout]], [[verl-grpo]]
-->

# Minibatch Sharing Across Prompts in LLM RL — Framework Synthesis
- **Core Insight:** For critic-free algorithms (GRPO, RLOO, REINFORCE++), the variance of the advantage estimator depends much more on the number of rollouts *per prompt* than on the number of prompts per minibatch — but throughput depends on both; the optimal recipe packs `B` prompts × `n` rollouts into every batch (`B·n` = effective batch size) with typical `n ∈ {4, 8, 16}`.
- **Guideline:** When tuning a GRPO/RLOO run, trade off `B` (prompt diversity) vs `n` (advantage-variance reduction) under fixed `B·n` compute budget. 8 rollouts per prompt is the sweet spot; dropping below 4 makes group baselines noisy.
- **Sources:** Synthesized from (i) Ahmadian et al. 2024 RLOO (§4 hyperparameter ablations); (ii) DeepSeekMath §4 GRPO ablations; (iii) verl/TRL code ("group size" configs); (iv) Lambert / Interconnects GRPO notes.
- **Year:** 2024–2025 convention
- **URLs:**
  - RLOO: https://arxiv.org/abs/2402.14740
  - DeepSeekMath: https://arxiv.org/abs/2402.03300
  - verl config: https://github.com/verl-project/verl (see `actor_rollout_ref.rollout.n`)
- **Relevant topics:** batch composition, rollout count, group baseline variance, prompt sharing

## Abstract (synthesized)
There is no single primary paper for "minibatch sharing" in LLM RL; the pattern emerged from ablations in RLOO and DeepSeekMath GRPO and is now a framework configuration knob. This page collects the operative findings. Group-relative estimators (RLOO, GRPO) require sampling multiple completions per prompt; those multiple completions must share a minibatch for the per-prompt baseline to be computable. Throughput then optimizes over (`B` prompts, `n` rollouts each) with total budget `B·n`. Ablations converge on `n ≥ 4` with `n=8` as the usual default.

## Key Contributions / Findings
- **RLOO §4 ablation:** leave-one-out baseline variance scales as `1/(n−1)`; moving from n=2 to n=4 halves variance; n=8 vs n=4 reduces variance further but with diminishing returns.
- **DeepSeekMath GRPO:** uses n=64 per prompt; ablations in appendix show gains plateau past n=16 for math tasks, but 16 is unstable on harder OOD problems.
- **Framework defaults:** verl `rollout.n=8`; TRL `num_generations=8`; OpenRLHF PPO `n_samples_per_prompt=4` (critic available, so less reliant on group baseline).
- **Throughput asymmetry:** higher `n` with fixed `B·n` increases GPU utilization because the vLLM engine batches same-prompt requests together (shared KV cache for the prompt prefix).

## Key Figures/Tables to Study
- **RLOO Figure 2:** advantage estimator variance vs n — the 1/(n−1) curve.
- **DeepSeekMath Figure 7 (appendix):** GRPO pass@1 vs n — knee at n=16 for MATH, keeps rising for AIME.
- **Interconnects GRPO notes (Lambert 2024):** empirical sweep `n ∈ {4, 8, 16}` — 8 wins on most tasks.

## Technical Details — Batch Composition
- **Typical config (verl / TRL / OpenRLHF):**
  ```
  B (prompts per step) = 128
  n (rollouts per prompt) = 8
  Effective batch (sequences) = 1024
  Micro-batch per device = 4–16 (FSDP-sharded)
  PPO epochs = 1 or 2
  ```
- **Prefix KV sharing:** same-prompt rollouts share the prompt prefix forward pass in vLLM — effectively free beyond the first rollout. This makes higher `n` cheaper than linear in compute.
- **Group baseline math (GRPO):**
  ```
  A_{i,k} = (R_{i,k} − μ_i) / (σ_i + ε),   μ_i = mean_k R_{i,k}
  ```
  Well-defined only if `n ≥ 2`; numerically stable at `n ≥ 4`.
- **RLOO baseline (leave-one-out):**
  ```
  A_{i,k} = R_{i,k} − (1/(n−1)) · Σ_{j≠k} R_{i,j}
  ```
  Unbiased; lower variance than GRPO's σ-normalized form under small-n.
- **Length-matched packing:** OpenRLHF and verl pack same-prompt rollouts together so the padding mask overhead is amortized.

## Connections
- Underlies every GRPO/RLOO implementation — see [[verl-grpo]], [[trl-grpo]], [[rloo]].
- Complements [[async-rollout]]: async rollouts keep the trainer saturated while the next `B·n` batch is being generated; batch composition determines what "next" means.
- The `n=1` special case is REINFORCE (no baseline) or requires a learned critic (PPO) — and the critic is what critic-free algorithms are trying to avoid.
- Related to [[reinforce-plus-plus]] variance-reduction variants that attempt to reduce `n` requirement via learned group statistics.
