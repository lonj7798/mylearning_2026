---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-grpo.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: verl GRPO — `compute_grpo_outcome_advantage`

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-grpo.md`
**Artifact:** `@register_adv_est(AdvantageEstimator.GRPO)` at `verl/trainer/ppo/core_algos.py` ~lines 290–335. The group-baseline advantage. Reuses `compute_policy_loss_vanilla` unchanged.

---

## Why this excerpt exists in ch-55

Ch-55 §3 is the line-by-line walk of this function. GRPO in verl is *only* an advantage estimator — the rest of the PPO machinery is unchanged, the critic is simply disabled. Understanding this function + `compute_policy_loss_vanilla` = understanding verl's GRPO end-to-end.

---

## The full quoted body

```python
# verl/trainer/ppo/core_algos.py, ~lines 290-335
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (B, T) — reward placed on last response token
    response_mask: torch.Tensor,         # (B, T)
    index: np.ndarray,                   # (B,)  — prompt id per rollout
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)    # (B,) outcome reward per rollout
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx]  = torch.tensor(1.0)
            else:
                tens = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(tens)
                id2std[idx]  = torch.std(tens)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]   # Dr.GRPO mode
        scores = scores.unsqueeze(-1) * response_mask        # broadcast to all response tokens
    return scores, scores   # (advantages, returns) — same tensor; no critic
```

---

## Six things from the source's "What to notice"

1. **No critic model.** `(advantages, returns)` are the *same* tensor. `vf_coef = 0`. The critic worker is instantiated but its loss is skipped — this is where GRPO's memory-halving vs PPO comes from.
2. **Per-token broadcast.** The per-rollout scalar advantage is `unsqueeze(-1) * response_mask` — every token in a completion shares the same advantage. Origin of GRPO's length bias ([[dr-grpo]]).
3. **`norm_adv_by_std_in_grpo` toggle.** `True` → standard GRPO z-score. `False` → Dr.GRPO (Liu 2025), which removes the std denominator to fix the variance bias on all-correct / all-wrong groups.
4. **Singleton groups (n=1).** Get mean=0, std=1 → zero advantage, no gradient contribution. In practice `n ≥ 4`.
5. **Outcome-only.** `token_level_rewards.sum(dim=-1)` collapses to a scalar per rollout — GRPO as coded here assumes a single outcome reward per rollout (typically a verifier 0/1). Process rewards use the GRPO-Pass@k or process variants.
6. **Reuses `compute_policy_loss_vanilla`.** GRPO is `adv_estimator = GRPO` + `use_critic = False` + the same policy loss. One registry flip, not a separate trainer.

---

## Where KL enters

**Not here.** Paper [[grpo]] Eq. 3 has `β·D_KL` *inside the loss*. verl intentionally moves it out — β·KL is subtracted from the per-token reward *before* this function runs, via `kl_penalty(logp, ref_logp, mode)` with `mode ∈ {k1, k2, k3}` (default k3). This matches the reward-shaping convention of [[kl-control-rlhf]] and differs from TRL-GRPO, which adds the K3 estimator to the loss term directly.

Why the choice matters: reward-shaping means the advantage estimator sees a KL-adjusted reward signal, so large-KL tokens are penalized through the same GRPO group-baseline mechanism as the RM reward. Adding KL to the loss bypasses the baseline and applies the penalty uniformly — different optimizer dynamics on long-tailed KL distributions.

---

## GRPO-Pass@k variant

`compute_grpo_passk_outcome_advantage` (lines 498–550): credits only the *best* response per group with `(r_max − r_second_max)/σ`; other rollouts in the group contribute zero advantage. Useful when the downstream metric is pass@k rather than pass@1.

---

## Hparam anchor — DeepSeekMath paper → verl config

| Knob                 | [[grpo]] paper | verl default / field                              |
|----------------------|----------------|---------------------------------------------------|
| Group size G         | 64             | `n` rollouts per prompt (4–64)                    |
| Clip ε               | 0.2            | `clip_ratio_low`, `clip_ratio_high`               |
| KL coefficient β     | 0.04           | `beta_kl` in AlgoConfig                           |
| KL estimator         | k3             | `kl_penalty_mode = "k3"` (default)                |
| Normalize by std     | yes            | `norm_adv_by_std_in_grpo = True` (False → Dr.GRPO)|
| Epochs per rollout μ | 1              | `ppo_mini_batch_epochs = 1`                       |

---

## Comparison to other frameworks

- **TRL `GRPOTrainer._compute_loss`** — bundles advantage normalization, KL term (as a loss addition via K3), and clipped objective into one function. Easier to swap `loss_type` but harder to isolate.
- **OpenRLHF GRPO** — reuses its `PolicyLoss` PPO module and computes group baselines in the experience-buffer pre-processing. Same algebra, different file layout.
