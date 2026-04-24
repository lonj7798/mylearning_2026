---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md
source_url: https://github.com/verl-project/verl ; https://github.com/OpenRLHF/OpenRLHF ; https://github.com/huggingface/trl
created_at: "2026-04-23"
---

# Excerpt: verl / OpenRLHF / TRL — entropy and KL logging patterns

**Source library:** `wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md`
**Frameworks:** verl (ByteDance), OpenRLHF (Jian Hu et al.), HuggingFace TRL
**Year:** 2024–2025

---

## Why this source anchors ch-43

The read chapter's §3 and §4 derive k1 / k2 / k3 mathematically and place KL into either the reward stream or the loss. This excerpt shows **how each framework actually implements those choices in code**, with direct quotations from the canonical source files. Useful for ch-55 (verl internals) and ch-56 (OpenRLHF internals), and essential for ch-46's RL lab where you will read these exact lines.

---

## The three KL estimators, as shipped

Source lines 23–37 — verl switch:

```python
# verl/trainer/ppo/core_algos.py  (kl_penalty switcher; applied in reward shaping)
def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "k1":  return logprob - ref_logprob
    if kl_penalty == "k2":  return 0.5 * (logprob - ref_logprob) ** 2
    if kl_penalty == "k3":  # Schulman unbiased
        diff = ref_logprob - logprob
        return torch.exp(diff) - diff - 1
```

All three estimators from [[excerpts/john-schulman-kl-tricks]] appear as a config switch in one file — verl is the cleanest place to read them.

Note the sign conventions: verl's k1 returns `log(π_new / π_ref)` (positive when π_new > π_ref for the sample); the read chapter writes k1 as `−log r` with `r = p/q = π_ref/π_new`. The two conventions differ only by a sign flip in the argument; the framework's downstream code uses this sign consistently.

verl also clamps the log-ratio before exponentiation:

```python
# verl/trainer/ppo/core_algos.py (PPO loss, ppo_kl monitor)
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
```

This is the fix for the "k3 exploded" caveat in [[excerpts/john-schulman-kl-tricks]]: clamping keeps `r` bounded so one tail sample cannot overwhelm a batch.

---

## Where KL is applied — two families

**Reward-shaping family (verl, OpenRLHF, TRL PPO).** Subtract `β · KL` from the per-token reward before advantage computation.

Source lines 50–55 — OpenRLHF:

```python
# openrlhf/trainer/ppo_trainer.py  (KL-to-reference applied as reward shaping)
self.kl_ctl = (AdaptiveKLController(init_coef, target, horizon)
               if adaptive else FixedKLController(init_coef))
# ... each step, per-token reward is shaped as:  reward_t -= kl_ctl.value * kl_t
self.kl_ctl.update(status["kl"], rollout.batch_size * n_samples_per_prompt)
```

The `AdaptiveKLController` targets a desired per-batch KL; see [[excerpts/kl-control-rlhf]] for the InstructGPT ancestor of this pattern.

**Loss-term family (TRL GRPO).** Add `β · k3` directly to the per-token loss.

Source lines 71–77 — TRL GRPO:

```python
# trl/trainer/grpo_trainer.py  (GRPO: KL enters the loss directly via K3)
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) \
             - (ref_per_token_logps - per_token_logps) - 1          # K3
per_token_loss = per_token_loss + self.beta * per_token_kl
# entropy is always logged:
mean_entropy = masked_batch_mean(entropies)
self._metrics[mode]["entropy"].append(gather(mean_entropy).nanmean().item())
```

No reward shaping, no AdaptiveKLController — just `+ β · k3` in the per-token loss, with entropy logged as a separate scalar metric. This is the DeepSeekMath GRPO convention.

---

## Two "entropies" in TRL PPO

Source lines 59–69:

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

TRL PPO logs two different "entropies":

- `objective/entropy` = mean negative log-prob on the rollout. This is a **proxy** that conflates entropy with logprob. Cheap, but misleading when the policy has high variance across states.
- `policy/entropy_avg` = the **true categorical entropy** `logsumexp(logits) − Σ p · logits`, computed from logits in a no-grad block. Expensive but correct.

When these two disagree, the policy's state-marginal distribution is non-uniform (some states are much sharper than others). The collapse threshold `H < 0.1` applies to the *true* entropy `policy/entropy_avg`, not the proxy.

---

## Collapse signature across frameworks

Source line 92:

> Entropy collapse signature is identical across frameworks: `entropy` falls ≥30% in <100 steps, `ppo_kl` spikes ≥0.1, `clipfrac` pegs to 1. Use whichever framework metric maps to these three.

This is the diagnostic triple. All three frameworks expose something named `entropy` (true or proxy), something named `ppo_kl` or `kl` (the K1 approximation used for monitoring), and `clipfrac` or `clip_ratio` (fraction of tokens where the PPO clip activated). When all three move together in that pattern, you are collapsing, not converging.

---

## Comparison table

Source lines 95–102:

| Concern | verl | OpenRLHF | TRL PPO | TRL GRPO |
|---|---|---|---|---|
| Entropy logged | `actor/entropy` (true H) | per-step mean `−logp` | `objective/entropy` + `policy/entropy_avg` | `_metrics["entropy"]` (true H) |
| Entropy in loss | no (optional registry hook) | no | no | no (β·KL only) |
| KL-to-ref where | reward shaping | reward shaping (`kl_ctl`) | reward shaping (`kl_ctl`) | loss term `β·K3` |
| KL estimator | k1/k2/k3 switch | K1 | K1 + K2 approx | K3 |
| Rollout-vs-train KL | via IS weights | `vllm_kl` metric | n/a | `importance_sampling_ratio` |
| Entropy masking | optional registry | no | no | `top_entropy_quantile` |

Two practitioner takeaways:

1. **Entropy is never a loss term in any default config.** All four framework defaults ship `c_H = 0` or equivalent. If you want the entropy bonus from [[entropy-regularization-ppo]], you have to turn it on explicitly.
2. **K3 is the modern default.** verl exposes it as a switch (default k3 in many recipes); TRL GRPO hard-codes it. TRL PPO and OpenRLHF still compute K1 for the reward-shaping path, but the K3 convention has won in GRPO-family methods.

---

## `vllm_kl` — the inference-vs-training mismatch monitor

Source lines 52–55 and 90:

> OpenRLHF adds `vllm_kl`: mean logprob difference between the rollout engine and the training forward pass. If this diverges from zero, training-inference mismatch is active and PPO will destabilize unless IS correction (TIS/iCEPO) is enabled.

This is a metric without a direct counterpart in verl/TRL. Because rollouts are generated by a separate engine (vLLM / SGLang) with its own kernel implementations and numerical precision, there is an implicit distribution shift between "what the sampler thinks the policy is" and "what the trainer computes". `vllm_kl` is the K1-style mean log-prob difference between the two. Zero means perfect agreement; anything else means importance-sampling correction or kernel alignment is required.

Relevant to ch-43 because: collapse triage (§2 of the read chapter) should check `vllm_kl` before suspecting β. A spurious collapse often traces back to a kernel mismatch rather than a real exploration problem.

---

## Connections

- Read-chapter §3 gives the math; this excerpt gives the shipped implementations.
- Read-chapter §4 (KL-to-reward vs KL-as-loss) maps 1:1 to this excerpt's reward-shaping vs loss-term families.
- [[excerpts/john-schulman-kl-tricks]] — the upstream derivation of k1/k2/k3.
- [[excerpts/openrlhf-entropy-debugging]] — the issue-tracker-level triage for the failures this excerpt's metrics surface.
- Downstream: ch-55 (verl internals), ch-56 (OpenRLHF internals), ch-57 (TRL internals) read these exact files in depth.
