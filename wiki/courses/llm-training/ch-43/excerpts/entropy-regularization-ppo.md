---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/entropy-regularization-ppo.md
source_url: https://arxiv.org/abs/1707.06347
created_at: "2026-04-23"
---

# Excerpt: Entropy regularization from A3C to PPO — the bonus LLM-RL inherited

**Source library:** `wiki/raw-data/llm-training/papers/entropy-regularization-ppo.md`
**Authors:** Volodymyr Mnih et al. (A3C, 2016); John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (PPO, 2017)
**Year:** 2016 / 2017

---

## Why this source anchors ch-43

A3C introduced the entropy-bonus term `+ β · H(π)` as the minimal exploration knob for policy-gradient RL. PPO inherited it verbatim. Every modern LLM-RL framework (TRL, OpenRLHF, verl) exposes an `entropy_coef` knob whose lineage traces back to this one-line addition. The read chapter's §5 uses this to ground "what LM-RL inherited": the *loss form* of the bonus is 2016's, the *coefficient* is often zero, and the *failure mode* (symmetric bonus under-corrects on large vocabularies) motivates the covariance-targeted interventions of §1.

---

## The loss form

Source lines 16, 21–22:

> A3C loss: `L_actor = −E[log π(a|s) · A] − β · H(π(·|s))`.
> PPO form: `L_PPO = E_t[min(r_t(θ) A_t, clip(r_t(θ), 1−ε, 1+ε) A_t) − c_v (V_θ − V_target)^2 + c_H H(π)]` with `ε = 0.2`, `c_v ≈ 0.5`, `c_H ≈ 0.01` as Atari defaults.

Three terms:

- `L_CLIP = E[min(r A, clip(r, 1−ε, 1+ε) A)]` — clipped surrogate policy-gradient.
- `c_v · L_VF = c_v · (V − V_target)²` — value-function regression.
- `c_H · H(π)` — the entropy bonus, with sign flipped so that **maximizing** entropy enters the objective **negatively** in the loss (some codebases write `−c_H · H`; sign conventions vary).

The Atari defaults — `ε = 0.2`, `c_v ≈ 0.5`, `c_H ≈ 0.01` — are the ones many ML engineers have memorized. They are not LLM-appropriate; see below.

---

## The LLM-RL carry-through

Source line 25:

> LLM carry-through: TRL `PPOTrainer` exposes `entropy_coef`; OpenRLHF and verl expose equivalent config. GRPO (DeepSeekMath) by default omits the bonus (relies on KL-to-reference), which is one of the reasons entropy collapse has become a named problem in LLM-RL.

The framework defaults flipped to `c_H = 0` during the transition from pre-LLM PPO to LLM-RL. Why:

1. **Large vocabulary, small mass.** On Atari's 18-action discrete or MuJoCo's 6–20-d continuous, the entropy-bonus gradient is well-scaled against the policy-gradient term. On a 128k-vocab softmax, the entropy of a fully-uniform policy is `log(128000) ≈ 11.76` nats. Even if the actual distribution is peaked, the per-token entropy is in `1–3` nats — orders of magnitude larger than Atari's `log(18) ≈ 2.89` as the maximum. A `c_H = 0.01` bonus on a quantity that is already large gradient-wise would swamp the policy-gradient term.
2. **KL-to-reference takes over the regularizer role.** RLHF (ch-38) adds a KL-to-SFT penalty in the reward stream (β ≈ 0.02 per [[kl-control-rlhf]]). This is a *directional* regularizer (pull toward `π_ref`) rather than a *symmetric* one (flatten the distribution). Practitioners found the KL term sufficient; the entropy bonus was dropped by convention.
3. **GRPO (DeepSeekMath) explicitly omits `c_H`.** Advantages are group-relative z-scores; KL-to-reference is applied as a loss term ([[excerpts/entropy-logging-patterns]]). No entropy bonus by default.

The cost: when collapse happens, the flat bonus can't rescue it because — as the read chapter's §1 and [[excerpts/entropy-mechanism-llm-rl]] explain — the collapse is driven by a thin tail of high-covariance tokens, not by the bulk of the distribution. Cui 2025's Clip-Cov / KL-Cov are the surgical rescue; `c_H` is the blunt tool.

---

## The A2C-to-SAC limit

Source line 27:

> Link to max-ent (Haarnoja): the bonus is the small-α limit of SAC's max-ent term — entropy-regularization in on-policy settings is a "poor man's" soft RL.

If you Taylor-expand the soft-Bellman value function `V(s) = α · log ∫ exp(Q(s,a)/α) da` around small α, the first-order correction to the standard Bellman V is exactly an entropy term weighted by α. The A3C/PPO `+ c_H · H(π)` actor-loss regularizer is what you get in the "small α, on-policy" limit, i.e. when you commit to a policy that is approximately greedy and only add a tiny bit of entropy pressure.

This is why [[excerpts/maximum-entropy-rl]] calls PPO's entropy bonus a "poor man's soft RL": the full soft-Bellman recursion (log-sum-exp replacing max, critic learning soft Q) is expensive and off-policy; the bonus is the cheap first-order approximation you can drop into on-policy REINFORCE/PPO without changing anything else.

---

## Atari-scale vs LLM-scale coefficient calibration

Source lines 23, 36:

> `ε = 0.2`, `c_v ≈ 0.5`, `c_H ≈ 0.01` as Atari defaults.
> Batching note: entropy is averaged over tokens in a rollout batch; coefficient typical range in LLM RL: 0.0 (GRPO default) to 1e-3.

Two orders of magnitude of scale difference between Atari's `c_H = 0.01` and LLM-RL's typical `c_H = 0` to `1e-3`. Not because the bonus "matters less" on LLMs — the opposite, in a sense, because entropy is a more delicate quantity when vocabularies are huge — but because the **per-token gradient from the bonus is proportional to ∂H/∂z**, and in a large-vocab softmax that gradient is spread so thinly that `c_H · ∂H/∂z` needs to be small to not swamp the per-token policy-gradient signal.

---

## The failure-mode argument

Source lines 37–39:

> Failure modes specific to LLMs: because the vocabulary is huge, most tokens contribute near-uniform entropy; the collapse happens in the tail — the tokens that truly matter become sharply peaked while average entropy stays moderate. That is why a blanket entropy bonus under-corrects (documented in [[entropy-mechanism-llm-rl]]).

This is the key sentence for connecting this excerpt to the read chapter's §1. The LLM failure mode is "sharp peak on a handful of tokens + uniform bulk elsewhere" — a state that has *moderate* average entropy (because most of the vocab is still uniform-ish) but is *effectively collapsed* (the decisions that matter are deterministic). A flat `+ c_H · H(π)` bonus pushes on the average, not on the peak, so the bonus cannot arrest the collapse without being so large it hurts the task objective.

Covariance-targeted interventions (Cui 2025 Clip-Cov, KL-Cov) fix this by operating on the peak tokens specifically. See [[excerpts/entropy-mechanism-llm-rl]] for the derivation of why `Cov(log π, A)` pinpoints the collapse-driving tokens.

---

## Practitioner table (coefficient regimes)

| Regime | Typical `c_H` | Why |
|--------|---------------|-----|
| Atari A3C / PPO | 0.01 | small action space, tunable |
| MuJoCo PPO | 0 – 5e-3 | continuous actions, per-[[entropy-collapse-ppo]] |
| LLM RLHF-PPO | 0 | KL-to-reference takes over |
| LLM GRPO (DeepSeekMath) | 0 | group-baseline + KL-in-loss |
| LLM with entropy masking (DAPO, `top_entropy_quantile`) | 0 (mask-based) | targets tail, not symmetric bonus |
| LLM with Clip-Cov / KL-Cov (Cui 2025) | 0 (covariance-based) | same logic, different selector |

The trend is clear: LLM-RL has moved *away* from symmetric entropy regularization and *toward* asymmetric, token-selective interventions.

---

## Connections

- Read-chapter §1 explains why the flat bonus under-corrects; this excerpt gives the historical reason it exists at all.
- Read-chapter §5 ("Max-ent RL ancestry") uses this as the on-policy small-α limit of SAC.
- [[excerpts/maximum-entropy-rl]] — the SAC ancestor.
- [[excerpts/entropy-mechanism-llm-rl]] — the modern covariance-targeted replacement.
- [[excerpts/entropy-logging-patterns]] — the framework-level `entropy_coef` config knob.
- ch-37 (Policy-Gradient Foundations) — where A3C/PPO's score-function estimator was introduced.
