---
chapter: ch-58
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: entropy/KL logging patterns — the §3 crib sheet's origin

**Source library:** `wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md`
**Artifact:** cross-framework comparison table of entropy/KL reporting in verl, OpenRLHF, TRL.

---

## Why this source defines ch-58 §3 verbatim

Ch-58 §3 "Logging-pattern crib sheet" is not an independent synthesis — it is a re-cut of this source's comparison table, filtered to the three signals (KL-to-old, KL-to-ref, rollout-vs-train KL, entropy, clipfrac) that the next chapter's capstone will monitor.

## The three-KL taxonomy

Source:

> Three KL estimators in circulation:
>   - K1 (`logπ − logπ_ref`): simple, used by verl reward shaping and TRL `objective/kl`. Biased (not non-negative).
>   - K2 (`0.5·Δlogp²`): variance-reduced, always non-negative. TRL `approxkl_stats`.
>   - K3 (`exp(−Δlogp) + Δlogp − 1`): unbiased, always ≥ 0. verl `kl_penalty="k3"` and TRL GRPO loss term — this is the modern default.

Ch-58 §1 bullet "Three canonical KL estimators (K1 / K2 / K3)" is this taxonomy compressed. Every estimator citation in §3's first column is attested by this source.

## The "where KL is applied" divergence

Source:

> Where KL is applied differs:
>   - verl / OpenRLHF: subtract β·KL from the per-token reward *before* advantage computation.
>   - TRL PPO: same (KL in reward as `non_score_reward`) but uses an `AdaptiveKLController`.
>   - TRL GRPO: β·per_token_kl added *to the loss* directly — no reward shaping, no controller.

Ch-58 matrix row 8 cells are this divergence enumerated. The TRL-GRPO-only loss-term path is the attested exception.

## The `vllm_kl` diagnostic — OpenRLHF-only

Source:

> OpenRLHF adds `vllm_kl`: mean logprob difference between the rollout engine and the training forward pass. If this diverges from zero, training-inference mismatch is active and PPO will destabilize unless IS correction (TIS/iCEPO) is enabled.

Ch-58 §3 "rollout-vs-train KL" row cites this as OpenRLHF-exclusive. The §6 graduation criterion "you hit the vLLM-IS-correction wall" is operationalized through this metric.

## The "two entropies in TRL PPO" subtlety

Source:

> Two "entropies" in TRL PPO: `objective/entropy` is the cheap `(−logprob).sum(1).mean()` proxy; `policy/entropy_avg` is the true categorical entropy computed from logits. They disagree when the policy is high-variance.

Ch-58 matrix row 9 TRL cell says "`objective/entropy` biased, `policy/entropy_avg` true H". The attestation is here. A learner who builds a dashboard panel on `objective/entropy` and concludes the policy hasn't collapsed — when the true categorical entropy has crashed — will reach exactly the failure this source warns against.

## The collapse signature — framework-independent

Source:

> Entropy collapse signature is identical across frameworks: `entropy` falls ≥30% in <100 steps, `ppo_kl` spikes ≥0.1, `clipfrac` pegs to 1. Use whichever framework metric maps to these three.

Ch-58 §3 final line reproduces this verbatim. The "framework-independent" claim is what makes the crib sheet useful — a learner diagnosing a run on OpenRLHF can use the same thresholds as a learner on verl.

## The source's own comparison table — ch-58 §3's template

Source:

> | Concern | verl | OpenRLHF | TRL PPO | TRL GRPO |
> |---|---|---|---|---|
> | Entropy logged | `actor/entropy` (true H) | per-step mean `−logp` | `objective/entropy` + `policy/entropy_avg` | `_metrics["entropy"]` (true H) |
> | Entropy in loss | no (optional registry hook) | no | no | no (β·KL only) |
> | KL-to-ref where | reward shaping | reward shaping (`kl_ctl`) | reward shaping (`kl_ctl`) | loss term `β·K3` |
> | KL estimator | k1/k2/k3 switch | K1 | K1 + K2 approx | K3 |
> | Rollout-vs-train KL | via IS weights | `vllm_kl` metric | n/a | `importance_sampling_ratio` |
> | Entropy masking | optional registry | no | no | `top_entropy_quantile` |

Ch-58 §3 collapses this into a 5-row signal-focused version. The fidelity check: every ch-58 §3 cell is a re-projection of one of these six source rows.

## What ch-58 inherits verbatim

- K1/K2/K3 formulas and their framework mappings.
- "TRL GRPO adds β·K3 to loss" as the single cross-framework exception.
- `vllm_kl` as OpenRLHF-exclusive rollout-vs-train diagnostic.
- `objective/entropy` (biased) vs `policy/entropy_avg` (true H) terminology.
- Collapse thresholds: entropy −30% in <100 steps, ppo_kl > 0.1, clipfrac → 1.

## Connections

- **[[verl-ppo-loss]]** — where K1/K2/K3 switch is defined (`kl_penalty()`).
- **[[openrlhf-ppo]]** — where `vllm_kl` is defined (`PolicyLoss.forward` return tuple).
- **[[trl-ppo]]** — where the two-entropy-fields discipline is visible.
- **[[trl-grpo]]** — where β·K3 in loss is the active path.
- **[[openrlhf-entropy-debugging]]** — the community-triage order this crib sheet enables.
