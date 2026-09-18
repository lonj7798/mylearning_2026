---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-grpo.md
source_url: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/trainer/ppo/core_algos.py#L266-L530
created_at: "2026-04-23"
updated_at: "2026-09-17"
---

# Excerpt: verl GRPO — `compute_grpo_outcome_advantage` and relatives

**Canonical extract:** `wiki/raw-data/llm-training/frameworks/verl-grpo.md` (verified 2026-09-14 against commit `753aed3`). This file was rewritten in the 2026-09 revision to match that card.

---

## The body the chapter quotes (core_algos.py L311–329)

```python
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
```

`scores` starts as `token_level_rewards.sum(dim=-1)` (L304). `μ_g`, `σ_g` are the mean and `torch.std` (divisor G−1) over responses sharing a prompt `uid`; `ε = 1e-6` (L272). The same tensor is returned as both advantages and returns (L331).

## Worked example (binary reward, G = 4)

| Case | μ | σ | advantage of a correct response | advantage of a wrong response |
|---|---|---|---|---|
| 1 of 4 correct | 0.25 | 0.5 | +1.5 | −0.5 |
| 3 of 4 correct | 0.75 | 0.5 | +0.5 | −1.5 |
| 0 or 4 correct | — | 0 | 0 | 0 |
| 1 of 4, no std scaling | 0.25 | — | +0.75 | −0.25 |

A group of size 1 substitutes mean 0 and std 1, so its advantage is the raw reward divided by 1 + 1e-6 (L315–317). `rollout.n` defaults to 1 (`rollout.yaml` L127), so GRPO needs it raised explicitly (grpo.md L29).

## Dr. GRPO is three settings (docs/algo/grpo.md L57–62)

`loss_agg_mode: seq-mean-token-sum-norm`; `loss_scale_factor` set to a constant such as the maximum response length; `use_kl_loss: False`; `algorithm.norm_adv_by_std_in_grpo: False`. The std term lives in this estimator; the length term lives in `agg_loss`.

## Pass@k estimator (`grpo_passk`, L471–530)

Implements arXiv:2503.19595. Errors on groups smaller than 2 (L516–519); gives only the highest-reward response the advantage `r_max − r_second_max`, divided by the group std when `norm_adv_by_std_in_grpo` is true (L520–527); every other response gets 0. No response receives a negative advantage. With rewards [1, 0, 0, 0] the correct response gets 1/0.5 = 2.0.

## Multi-objective variant

`gdpo` (L361–468, arXiv:2601.05242) normalizes per reward dimension, takes a weighted sum, then whitens over the batch (L466).

## Corrections to the previous excerpt version

1. "Singleton groups get mean=0, std=1 — zero advantage" → the advantage is the raw reward divided by 1 + 1e-6; only all-equal groups of size ≥ 2 get zero.
2. "≈ lines 290–335" / "498–550" → L266–358 and L471–530 at 753aed3.
3. "GRPO memory footprint is ~half PPO's at equal model size" — removed; no source, and the saving depends on critic size, offloading, and whether a reference model is loaded.
4. "verl intentionally does not put KL in the loss; it subtracts β·KL from the reward" → both placements exist and both default to off; the GRPO docs recommend the loss form (`use_kl_loss: True`).
5. "`vf_coef` is 0 and `compute_value_loss` never runs" → no `vf_coef` key exists; `use_critic` comes from `need_critic` (ray_trainer.py L348), which is False when `adv_estimator != gae` and `critic.enable` is unset.
6. "Dr.GRPO is one boolean" → it is the four-key combination above.

## Connections

- [[grpo]] — DeepSeekMath §4.1.2, the advantage this implements.
- [[dr-grpo]] — the two normalization biases.
- [[verl-ppo-loss]] — the loss that consumes these advantages.
- [[rlvr-beyond-base-model]] — why a Pass@k objective is of interest.
