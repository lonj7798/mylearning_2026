---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-grpo.md
source_url: https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py
created_at: "2026-04-23"
---

# Excerpt: verl GRPO advantage — the registry-hook implementation

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-grpo.md`
**Artifact:** `verl/trainer/ppo/core_algos.py` ~L290–335, the `compute_grpo_outcome_advantage` function registered as `@register_adv_est(AdvantageEstimator.GRPO)`.

---

## Why this source anchors ch-40 §7

verl is the Bytedance/volcengine RL framework that most R1 reproductions actually run on. Its split design — advantage estimator in one function, policy loss in another — makes the GRPO-vs-Dr.GRPO diff especially clean: it is one boolean flag on one function. Ch-40 §7 uses this to show the reader that all the theory from §4–§5 reduces to dict-based groupby logic.

---

## The code ch-40 §7 quotes verbatim

Source lines 22–52:

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (B, T) — reward on last response token
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
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores  # (advantages, returns) — same tensor; no critic
```

---

## What ch-40 §7 points out

1. **`(advantages, returns) = (scores, scores)`** — no critic. The PPO-family value loss term (`vf_coef * (V(s) − R)²`) is disabled entirely; `use_critic = false` in the top-level config. Ch-40 §1's "kill PPO's critic" story is implemented here as a two-element tuple where both elements are the same tensor.

2. **The Dr.GRPO toggle is one line.** The `norm_adv_by_std_in_grpo=True` default is vanilla GRPO; flipping it to `False` is Dr.GRPO — the advantage becomes `r_i − mean(r)` with no std denominator. Ch-40 §5's "delete the divisor" is a single boolean flip in config.

3. **Singleton groups degrade gracefully.** `if len(id2score[idx]) == 1: id2mean=0, id2std=1` → advantage = r_i (zero if r_i is zero). In practice verl requires n ≥ 4 rollouts per prompt; the singleton branch exists only for edge cases.

4. **Per-token broadcast: `scores.unsqueeze(-1) * response_mask`.** The per-rollout scalar advantage becomes a (B, T) tensor where every response token in rollout i shares the same value. This is identical to TRL's `.unsqueeze(1)` broadcast — same mechanism, different syntax. This is where the length-bias interaction lives: combined with the policy-loss aggregator, per-token gradient scales as `|A|/|o_i|` for GRPO aggregation, `|A|/L_max` for Dr.GRPO aggregation.

5. **Outcome-only assumption.** `token_level_rewards.sum(dim=-1)` assumes the reward is placed on a single token (the last one). Process rewards — per-step feedback from a PRM — require the separate `compute_grpo_passk_outcome_advantage` variant at L498–550.

---

## The split vs fused design (ch-40 §7 contrast)

- verl splits: `compute_grpo_outcome_advantage` produces `(advantages, returns)`; then `compute_policy_loss_vanilla` (shared with PPO) produces the clipped surrogate. Two separate registry hooks.
- TRL fuses: `_compute_loss` does advantages broadcast + surrogate + KL + aggregation in one method. See [[trl-grpo]].

Algebraic equivalence for `loss_type="grpo"` and `AdvantageEstimator.GRPO + use_critic=false`. Practical difference: adding a new advantage estimator in verl means registering a new function; in TRL it means adding a branch to `_compute_loss`.

---

## Side variant: GRPO Pass@k (source lines 67)

`compute_grpo_passk_outcome_advantage` at L498–550 credits only the *best* rollout per group with `(r_max − r_second_max)/σ`. Useful when the downstream metric is pass@k rather than pass@1 — matches the training objective to the deployment objective.

---

## Connections to the rest of the track

- [[grpo]] — the paper this code implements (Eq. 3 of Shao 2024).
- [[dr-grpo]] — enabled by `norm_adv_by_std_in_grpo=False`.
- [[trl-grpo]] — the TRL equivalent with the fused `_compute_loss`.
- [[verl-ppo-loss]] — the shared policy loss used downstream of this advantage estimator.
- [[deepseek-r1]] — R1 reproductions are the primary user of this code path.
