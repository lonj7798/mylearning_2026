<!-- scope: how verl / OpenRLHF / TRL log entropy and KL during RL training
     deps: [[entropy-mechanism-llm-rl]], [[kl-control-rlhf]]
     see-also: [[verl-ppo-loss]], [[openrlhf-ppo]], [[trl-ppo]], [[trl-grpo]]
-->

# Entropy & KL Logging — Cross-Framework Comparison
- **Frameworks compared:** verl, OpenRLHF, HuggingFace TRL
- **Repo URLs:**
  - verl: https://github.com/verl-project/verl
  - OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
  - TRL: https://github.com/huggingface/trl
- **Relevant file(s):**
  - verl: `verl/trainer/ppo/core_algos.py` (`ppo_kl`, `kl_penalty` with k1/k2/k3 switches)
  - OpenRLHF: `openrlhf/models/loss.py` (`PolicyLoss.forward` returns `ppo_kl` and `vllm_kl`) + `AdaptiveKLController` wiring in `openrlhf/trainer/ppo_trainer.py`
  - TRL: `trl/experimental/ppo/ppo_trainer.py` (inner loop ≈ 850–870 and metric block 883–907); `trl/trainer/grpo_trainer.py` (`_compute_loss` ≈ 2578–2620)
- **Core pattern:** All three frameworks treat entropy as a *metric* (never a loss term in default configs) and treat KL as *either* a reward shaper (verl/OpenRLHF) *or* a loss-side regularizer (TRL GRPO via β·per_token_kl). The three canonical KL estimators — K1, K2, K3 — appear across the codebases with predictable tradeoffs.
- **Why it matters:** Entropy collapse (policy becomes near-deterministic, KL shoots up, pass@1 wins but pass@k drops) is the single most-common RL-for-LLM failure mode; knowing which metric each framework actually reports is prerequisite to debugging it.

## Context
"Entropy" in LLM-RL reporting almost never means the true categorical entropy of the policy — it usually means one of two proxies: the mean negative log-prob on the rollout (which conflates entropy with logprob), or the true `logsumexp(logits) − Σ p·logp` computed from the logits in a no-grad block. KL is even messier: raw `Δlogp = logπ − logπ_old` is K1 (biased, noisy), `0.5·Δlogp²` is K2 (Schulman low-variance), `exp(−Δlogp) + Δlogp − 1` is K3 (Schulman unbiased, non-negative). Frameworks differ in which they compute, where they apply it (reward shaping vs loss penalty), and what they name it in dashboards.

## Code excerpt — verl
```python
# verl/trainer/ppo/core_algos.py (PPO loss, ppo_kl monitor)
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)   # K1, logged as actor/ppo_kl

# verl/trainer/ppo/core_algos.py  (kl_penalty switcher; applied in reward shaping)
def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "k1":  return logprob - ref_logprob
    if kl_penalty == "k2":  return 0.5 * (logprob - ref_logprob) ** 2
    if kl_penalty == "k3":  # Schulman unbiased
        diff = ref_logprob - logprob
        return torch.exp(diff) - diff - 1
```
*No entropy term in the loss.* Entropy is logged separately as `actor/entropy` via `verl_F.entropy_from_logits` on the logits of the rollout; see `verl/workers/actor/*`.

## Code excerpt — OpenRLHF
```python
# openrlhf/models/loss.py  (PolicyLoss.forward returns both KLs)
clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
ppo_kl  = masked_mean(-log_ratio.detach(), action_mask, dim=None)   # K1-like monitor
vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None) \
          if enable_vllm_is_correction else None
return loss, clip_ratio, ppo_kl, vllm_kl
```
```python
# openrlhf/trainer/ppo_trainer.py  (KL-to-reference applied as reward shaping)
self.kl_ctl = (AdaptiveKLController(init_coef, target, horizon)
               if adaptive else FixedKLController(init_coef))
# ... each step, per-token reward is shaped as:  reward_t -= kl_ctl.value * kl_t
self.kl_ctl.update(status["kl"], rollout.batch_size * n_samples_per_prompt)
```

## Code excerpt — TRL
```python
# trl/experimental/ppo/ppo_trainer.py  (no-grad entropy from logits)
prob_dist = torch.nn.functional.softmax(logits, dim=-1)
entropy   = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
approxkl  = 0.5 * (logprobs_diff ** 2).mean()     # K2
entropy_stats[ppo_epoch, minibatch, grad_accum] = entropy.mean()
# dashboard fields:
metrics["objective/entropy"]  = gather(mean_entropy).mean().item()   # −Σ logp (biased)
metrics["policy/entropy_avg"] = gather(entropy_stats).mean().item()  # true H(π)
metrics["policy/approxkl_avg"] = gather(approxkl_stats).mean().item()
metrics["objective/kl"]       = gather(mean_kl).mean().item()        # K1
```
```python
# trl/trainer/grpo_trainer.py  (GRPO: KL enters the loss directly via K3)
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) \
             - (ref_per_token_logps - per_token_logps) - 1          # K3
per_token_loss = per_token_loss + self.beta * per_token_kl
# entropy is always logged:
mean_entropy = masked_batch_mean(entropies)
self._metrics[mode]["entropy"].append(gather(mean_entropy).nanmean().item())
```

## What to notice
- **Two "entropies" in TRL PPO:** `objective/entropy` is the cheap `(−logprob).sum(1).mean()` proxy; `policy/entropy_avg` is the true categorical entropy computed from logits. They disagree when the policy is high-variance.
- **Three KL estimators in circulation:**
  - **K1** (`logπ − logπ_ref`): simple, used by verl reward shaping and TRL `objective/kl`. Biased (not non-negative).
  - **K2** (`0.5·Δlogp²`): variance-reduced, always non-negative. TRL `approxkl_stats`.
  - **K3** (`exp(−Δlogp) + Δlogp − 1`): unbiased, always ≥ 0. verl `kl_penalty="k3"` and TRL GRPO loss term — this is the modern default.
- **Where KL is applied differs:**
  - verl / OpenRLHF: subtract β·KL from the per-token reward *before* advantage computation.
  - TRL PPO: same (KL in reward as `non_score_reward`) but uses an `AdaptiveKLController`.
  - TRL GRPO: β·per_token_kl added *to the loss* directly — no reward shaping, no controller.
- **OpenRLHF adds `vllm_kl`:** mean logprob difference between the rollout engine and the training forward pass. If this diverges from zero, training-inference mismatch is active and PPO will destabilize unless IS correction (TIS/iCEPO) is enabled.
- **Entropy bonuses are not default anywhere.** Cui 2025 ("Entropy Mechanism of RL for LLMs") and DAPO argue for token-level entropy masking or entropy annealing; TRL exposes this via `top_entropy_quantile`.
- **Entropy collapse signature is identical across frameworks:** `entropy` falls ≥30% in <100 steps, `ppo_kl` spikes ≥0.1, `clipfrac` pegs to 1. Use whichever framework metric maps to these three.

## Comparison table
| Concern | verl | OpenRLHF | TRL PPO | TRL GRPO |
|---|---|---|---|---|
| Entropy logged | `actor/entropy` (true H) | per-step mean `−logp` | `objective/entropy` + `policy/entropy_avg` | `_metrics["entropy"]` (true H) |
| Entropy in loss | no (optional registry hook) | no | no | no (β·KL only) |
| KL-to-ref where | reward shaping | reward shaping (`kl_ctl`) | reward shaping (`kl_ctl`) | loss term `β·K3` |
| KL estimator | k1/k2/k3 switch | K1 | K1 + K2 approx | K3 |
| Rollout-vs-train KL | via IS weights | `vllm_kl` metric | n/a | `importance_sampling_ratio` |
| Entropy masking | optional registry | no | no | `top_entropy_quantile` |

## Connections
- Entropy dynamics theory: [[entropy-mechanism-llm-rl]].
- KL estimator tradeoffs: [[john-schulman-kl-tricks]] blog.
- vLLM train-vs-rollout mismatch: see [[openrlhf-ppo]] IS-correction branches.
