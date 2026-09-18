---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md
source_url: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/trainer/ppo/core_algos.py#L1285-L1376
created_at: "2026-04-23"
updated_at: "2026-09-17"
---

# Excerpt: verl PPO loss — `compute_policy_loss_vanilla`

**Canonical extract:** `wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md` (verified 2026-09-14 against commit `753aed3`). This file was rewritten in the 2026-09 revision to match that card; where the two differ, the card is correct.

---

## The body the chapter quotes (core_algos.py L1336–1361)

```python
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

## Defaults (config, not a model recipe)

| Key | Default | Locus |
|---|---|---|
| `clip_ratio`; `clip_ratio_low`; `clip_ratio_high` | 0.2; 0.2; 0.2 | actor.yaml L36, L39, L42 |
| `clip_ratio_c` | 3.0, asserted > 1.0 | actor.yaml L83; core_algos.py L1323–1334 |
| log-ratio clamp | [−20, 20] | core_algos.py L1338 |
| `loss_agg_mode`; `loss_scale_factor` | `token-mean`; null | actor.yaml L86, L90 |
| `entropy_coeff`; `ppo_epochs` | 0; 1 | actor.yaml L93, L119 |
| `use_kl_loss`; `kl_loss_coef`; `kl_loss_type` | false; 0.001; `low_var_kl` (k3) | actor.yaml L103, L113, L116 |
| `use_kl_in_reward`; `kl_penalty` | False; `kl` (k1) | ppo_trainer.yaml L98, L101 |
| `policy_loss.loss_mode` | `vanilla` (12 registered names) | actor.yaml L61; core_algos.py L1379–2414 |

## Behaviour on a negative-advantage token (A = −1, ε = 0.2, c = 3)

| r | loss | gradient |
|---|---|---|
| 0.5 | max(0.5, 0.8) = 0.8 | 0 — the lower clip is active |
| 1.0 | 1.0 | 1.0 |
| 2.0 | max(2.0, 1.2) = 2.0, below the cap | 2.0 |
| 5.0 | max(5.0, 1.2) = 5.0, capped to 3.0 | 0 — the dual clip is active |

Ye et al. (arXiv:1912.09729v3, Eq. 5) give the reason for the cap: with off-policy data and Â_t < 0 a large ratio "will introduce a big and unbounded variance". Their ε = 0.2 and c = 3 are verl's defaults.

## Where entropy and KL enter

Not in this function. `verl/workers/utils/losses.py::ppo_loss` subtracts `entropy_coeff * entropy_loss` (L122–129) and adds `kl_loss * kl_loss_coef` when `use_kl_loss` (L131–142). The alternative in-reward KL lives in `ray_trainer.py` L78–117. `actor/ppo_kl` (L1340) is a k1 estimate between the old and current policy that is logged and never added to the loss.

## Aggregation modes (`agg_loss`, L1140–1206)

`token-mean` (masked sum over the global valid-token count, L1170–1175), `token-sum`, `seq-mean-token-sum`, `seq-mean-token-sum-norm` (divided by `loss_scale_factor`, default the response-mask width), `seq-mean-token-mean`. The docstring states that aggregation over the global batch makes the loss invariant to the FSDP or Megatron layout (L1150). The verl GRPO docs map Dr. GRPO to `seq-mean-token-sum-norm` (grpo.md L59).

## Corrections to the previous excerpt version

1. "~lines 1080–1140" → L1285–1376 at commit 753aed3.
2. "Dual-clip … floors the loss" → it caps the per-token loss at `c·|A|`, which zeroes the gradient for r > c.
3. "KL-to-ref is reward-shaped via `kl_penalty(...)`" → verl supports both placements and both default to off; the GRPO docs recommend the loss form.
4. "`seq-mean-token-sum` = Dr.GRPO" → the documented Dr. GRPO mode is `seq-mean-token-sum-norm`.
5. "verl logs entropy but doesn't add an entropy term (optional via `entropy_loss` registry)" → there is no such registry hook; `ppo_loss` subtracts `entropy_coeff · entropy_loss`, with the coefficient defaulting to 0.
6. "TRL `PPOTrainer` … symmetric `cliprange` only" → TRL removed that trainer in commit 700b845 (2026-09-04, PR #7020); the comparison was not re-checked at an older commit.
7. "rollout IS weights … per-token" → token-level or sequence-level, off by default (`rollout_is: null`).

## Connections

- [[verl-grpo]] — the advantages this loss consumes.
- [[ppo]] — the symmetric-ε original.
- [[dr-grpo]] — the length-bias argument behind the aggregation mode.
- [[entropy-logging-patterns]] — the metric names and KL estimators.
