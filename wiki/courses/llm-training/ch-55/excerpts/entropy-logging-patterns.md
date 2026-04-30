---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: Entropy & KL logging — verl's row in the cross-framework table

**Source library:** `wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md`
**Artifact:** what verl logs by default + the `kl_penalty` switcher + the comparison table row for verl.

---

## Why this excerpt exists in ch-55

Ch-55 §5 enumerates verl's default entropy / KL metrics and proposes five additions. This file gives the verbatim source for what verl ships — and the cross-framework comparison table that justifies each addition.

---

## What verl logs by default

From the source's "Code excerpt — verl":

```python
# verl/trainer/ppo/core_algos.py (PPO loss, ppo_kl monitor)
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio  = torch.exp(negative_approx_kl)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)   # K1, logged as actor/ppo_kl
```

Source note: *"No entropy term in the loss. Entropy is logged separately as `actor/entropy` via `verl_F.entropy_from_logits` on the logits of the rollout; see `verl/workers/actor/*`."*

So the default metric set, verbatim:

- `actor/ppo_kl` — K1 monitor `mean(logπ − logπ_old)` inside the loss.
- `actor/entropy` — true categorical entropy `logsumexp(logits) − Σ p·logp` via `verl_F.entropy_from_logits`.
- `actor/pg_clipfrac` — fraction of tokens where the clipped ratio branch won.
- `actor/pg_clipfrac_lower` — fraction where dual-clip fired (requires advantage < 0).

---

## The `kl_penalty` switcher — where KL-to-ref lives

```python
# verl/trainer/ppo/core_algos.py  (kl_penalty switcher; applied in reward shaping)
def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "k1":  return logprob - ref_logprob
    if kl_penalty == "k2":  return 0.5 * (logprob - ref_logprob) ** 2
    if kl_penalty == "k3":                                             # Schulman unbiased
        diff = ref_logprob - logprob
        return torch.exp(diff) - diff - 1
```

verl subtracts `β · kl_penalty(...)` from the per-token reward *before* the advantage estimator runs. Default mode is `k3` (always ≥ 0, unbiased). This is the *reward-shaping* pattern of [[kl-control-rlhf]], distinct from TRL-GRPO which adds K3 to the loss term directly.

---

## Three KL estimators, when to pick which

From the source's "What to notice":

- **K1** (`logπ − logπ_ref`): simple, used for verl reward shaping and TRL `objective/kl`. Biased (not non-negative).
- **K2** (`0.5·Δlogp²`): variance-reduced, always non-negative. Used by TRL for `approxkl_stats`.
- **K3** (`exp(−Δlogp) + Δlogp − 1`): unbiased AND always ≥ 0. verl `kl_penalty="k3"` and TRL GRPO loss term — the modern default.

Failure signal: if `actor/kl_loss` ever goes negative under `k3`, your ref forward pass is misaligned (wrong device / fp32 vs bf16 mismatch / stale checkpoint).

---

## Cross-framework comparison — verl's column

| Concern              | verl                      | OpenRLHF              | TRL PPO                   | TRL GRPO                 |
|----------------------|---------------------------|-----------------------|---------------------------|--------------------------|
| Entropy logged       | `actor/entropy` (true H)  | per-step `−logp` mean | `objective/entropy` + `policy/entropy_avg` | `_metrics["entropy"]` |
| Entropy in loss      | no (optional registry)    | no                    | no                        | no (β·KL only)           |
| KL-to-ref where      | reward shaping            | reward shaping        | reward shaping            | loss term β·K3           |
| KL estimator         | k1/k2/k3 switch           | K1                    | K1 + K2 approx            | K3                       |
| Rollout-vs-train KL  | via IS weights            | `vllm_kl` metric      | n/a                       | `importance_sampling_ratio` |
| Entropy masking      | optional registry         | no                    | no                        | `top_entropy_quantile`   |

---

## What ch-55 recommends adding

From ch-55 §5's list of five, each maps to a gap in the verl column above:

1. **`rollout_kl`** → fills the "rollout-vs-train KL" row (verl gates it only through IS weights; OpenRLHF ships `vllm_kl` as a standalone metric).
2. **Per-bucket entropy** → addresses the blind spot where global `actor/entropy` stays healthy while difficulty-stratified entropy collapses.
3. **`clipfrac_positive_only`** → disentangles ε_high vs ε_low tuning decisions; verl already splits the arithmetic internally but emits only aggregate `pg_clipfrac` / `pg_clipfrac_lower`.
4. **Reward over ref reward on held-out set** → the earliest detector of reward hacking (connects [[reward-hacking-taxonomy]] from ch-42).
5. **Sequence-level IS histogram** → informs the seq-mask-tis clip threshold choice.

---

## Connections

- Entropy dynamics: `entropy-mechanism-llm-rl.md` (Cui 2025 `R = -a·exp(H) + b`).
- KL estimator derivation: Schulman's "Approximating KL Divergence" blog.
- [[kl-control-rlhf]] — why reward-shaping β·KL is algebraically equivalent to a tilted-posterior target and why it's the right place to put KL for PPO.
- [[verl-ppo-loss]] — where `ppo_kl` is computed; the K1 monitor.
