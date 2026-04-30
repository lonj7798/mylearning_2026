<!-- scope: GRPO group-relative advantage estimator in verl
     deps: [[grpo]], [[verl-ppo-loss]]
     see-also: [[trl-grpo]], [[openrlhf-ppo]], [[entropy-logging-patterns]]
-->

# verl — GRPO Advantage & Trainer
- **Framework:** verl (volcengine / Bytedance Seed)
- **Repo URL:** https://github.com/verl-project/verl
- **Version/commit:** `main` branch (fetched 2026-04-21)
- **Relevant file(s):**
  - `verl/trainer/ppo/core_algos.py` ≈ lines 290–365 (`compute_grpo_outcome_advantage`, `compute_grpo_vectorized_outcome_advantage`)
  - `verl/trainer/ppo/core_algos.py` ≈ lines 498–550 (`compute_grpo_passk_outcome_advantage`)
- **Core pattern:** GRPO replaces the PPO critic with a *group-baseline*: for every prompt, sample `n` rollouts, compute outcome reward, then z-score each rollout's reward against the group mean/std. The same clipped policy loss (`compute_policy_loss_vanilla`) is reused; only the advantage estimator changes.
- **Why it matters:** GRPO is the loss DeepSeek used for R1-Zero / R1 / DeepSeekMath; verl's implementation is the canonical open-source reference and is what most R1 reproductions actually run.

## Context
GRPO (Shao et al. 2024, DeepSeekMath) eliminates the value-network half of PPO by using *group-relative* baselines: for each prompt index `i`, sample `n` completions, compute the scalar reward `r_{i,k}` per completion, normalize by the per-prompt group mean and std, and broadcast that group-relative score back to every response token as the advantage. verl exposes this as a registry-pluggable `@register_adv_est(AdvantageEstimator.GRPO)` function — the rest of the PPO machinery (clipped surrogate, KL, optimizer) is unchanged.

## Code excerpt
```python
# verl/trainer/ppo/core_algos.py, ~lines 290–335
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (B, T) — reward placed on last response token
    response_mask: torch.Tensor,         # (B, T)
    index: np.ndarray,                   # (B,)  — prompt id per rollout
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)  # (B,) outcome reward per rollout
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                tens = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(tens)
                id2std[idx] = torch.std(tens)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]   # Dr.GRPO mode
        scores = scores.unsqueeze(-1) * response_mask        # broadcast to all response tokens
    return scores, scores  # (advantages, returns) — same tensor; no critic
```

## What to notice
- **No critic model:** `(advantages, returns)` are the *same* tensor; the value-loss term in the PPO objective is dropped (`vf_coef = 0`).
- **Per-token broadcast:** the per-rollout scalar advantage is `unsqueeze(-1) * response_mask` — every token in a completion shares the same advantage. This is the source of GRPO's well-known length bias.
- **`norm_adv_by_std_in_grpo`** toggles between standard GRPO (z-score, std denominator) and *Dr.GRPO* (Liu 2025) which removes the std normalization to fix variance bias on prompts where all rollouts succeed/fail.
- **Singleton groups** (n=1) get mean=0, std=1 — zero advantage, the prompt contributes no gradient. In practice `n ≥ 4`.
- **Outcome-only rewards:** `token_level_rewards.sum(dim=-1)` — GRPO assumes a single outcome scalar per rollout (typically a verifier output 0/1). Process rewards require the GRPO-Pass@k or process variants.
- **The same `compute_policy_loss_vanilla` is reused** — see [[verl-ppo-loss]]. GRPO is just an advantage estimator change plus a configuration flip (`use_critic = false`).

## Comparison to paper / to other frameworks
- **vs DeepSeekMath paper:** matches Eq. 18 of Shao et al. 2024 exactly; verl adds the optional `norm_adv_by_std_in_grpo=False` Dr.GRPO toggle.
- **vs TRL `GRPOTrainer` (`trl/trainer/grpo_trainer.py`):** TRL bundles advantage normalization, KL term, and clipped objective into a single `_compute_loss` (see [[trl-grpo]]); verl keeps them separate via the registry.
- **vs OpenRLHF GRPO:** OpenRLHF reuses its `PolicyLoss` PPO module and computes group baselines in the experience-buffer pre-processing — same algebra, different code home.
- **GRPO Pass@k variant** in verl (`compute_grpo_passk_outcome_advantage`) credits only the *best* response per group with `(r_max − r_second_max)/σ` — useful when you only care about pass@k metrics.
