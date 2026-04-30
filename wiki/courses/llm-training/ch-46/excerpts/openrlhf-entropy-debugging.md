---
chapter: ch-46
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md
source_url: https://github.com/OpenRLHF/OpenRLHF
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF/verl/TRL entropy-debugging — ch-46's instrumentation spec

**Source library:** `wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md`
**Artifact:** Standard logged-metric set, community-standard triage order, framework defaults, common failure patterns.

---

## Why this source defines ch-46 §3 Instrumentation

Ch-46 §3 ("Instrumentation hooks — the four signals") does not invent a logging format; it copies the one that has converged across the three major open RLHF stacks. This source documents the convergence, and ch-46 lifts the signal set and triage order directly.

---

## The four signals, from the source

Source §Key Points:

> **Standard logged metrics:** per-token entropy, per-batch KL(π‖π_ref), PPO ratio mean/std, reward mean/std, clipped-fraction, response length histogram.

Ch-46 §3 compresses this into four signals (entropy, KL, reward_mean/reward_std, pass_rate/win_rate) plus `len_mean`, because: the four compress to "exploration, drift, signal, outcome" respectively, and five is what the MetricsSink callback can emit per step without blowing out the log file. Clipped-fraction and PPO ratio are on-demand — ch-46 reads them via TRL's default logging when a cell goes off the rails.

---

## The collapse definition ch-46 encodes as an assertion

Source §Key Points:

> If a run drops per-token entropy below ~0.1 nats and reward hasn't already saturated, it's diagnostic of collapse, not of convergence.

This is ch-46 §7 Acceptance's implicit gate: a cell whose `entropy` signal dives below 0.1 before `reward_mean` saturates must be tagged "collapse" in the memo §2 row. The distinction between "collapse" and "convergence" is exactly the reward-saturation check.

---

## The triage order — ch-46 §5 diagnostic flow

Source §Guideline:

> (1) confirm KL-to-reference term is on and finite, (2) bump rollout temperature by 0.1–0.2, (3) raise entropy coefficient an order of magnitude, (4) check advantage normalization is per-batch zero-mean unit-var, (5) only then suspect the reward signal.

This is the ch-46 memo §5 fix-order. Suspect-the-reward is step 5 out of 5. A learner who writes "the reward was wrong" in the first paragraph of their post-mortem has skipped four diagnostic steps. The memo rubric should penalize this.

---

## Framework defaults — what ch-46's configs reflect

Source §Key Points:

> **Default entropy coefficient `c_H`:** OpenRLHF and TRL default to `0.0` for LLM-RL (counter to pre-LLM PPO practice).
> **Default KL coefficient `β`:** around 0.01–0.1 of the reward scale.
> **Rollout sampler:** all three frameworks default to `T = 1.0`, `top_p = 1.0` for training rollouts.
> **Advantage normalization:** ON by default in OpenRLHF and verl, OFF by default in TRL (a recurring footgun).

Key ch-46 configuration choices, each justified by this section:
- `c_H = 0.0` (no entropy bonus) — TRL / OpenRLHF default; the lab *measures* entropy rather than regularizing it, because the post-mortem is the deliverable.
- β_KL sweep {0.01, 0.05, 0.1} — bracketing the 0.01–0.1 community default range.
- `T=1.0`, `top_p=1.0` in `GRPOConfig` — standard training rollout sampler.
- `scale_rewards=False` (Option B) — Dr.GRPO requires advantage mean-only; std-norm is off.

The "recurring footgun" line is why ch-46 §7 Acceptance criterion #3 exists: if advantage normalization is mis-configured in TRL (which has it OFF by default), the implicit-reward margin can look flat even when the loss is training — a silent failure attested in TRL issues.

---

## Common failure patterns — ch-46 §5 is populated from here

Source §Key Points:

> **Common failure patterns from issue trackers:**
> - Entropy crash within 100 steps → KL term accidentally off or β too small.
> - Reward-but-no-entropy-change after ~1000 steps → advantage normalization misconfigured.
> - Sudden length explosion in rollouts → entropy healthy but reward + length are confounded (reward hacking).
> - `NaN` in PPO ratio → very aggressive update; lower LR and clip range.

Each line maps to a ch-46 diagnostic:

| Pattern | Ch-46 §5 failure mode |
|---|---|
| Entropy crash < 100 steps | §5(b) entropy collapse |
| No entropy change | adv-norm misconfig — §7 Acceptance #3 checks this |
| Length explosion while entropy healthy | §5(a) reward hacking (Option A length hack) or §5(c) length bias (Option B) |
| NaN in ratio | not a scheduled post-mortem; ch-46 halts and debugs |

---

## GRPO-specific community defaults ch-46 inherits

Source §Key Points:

> **GRPO specifics (all three):** `group_size = 8` is typical small; `group_size = 16–32` is common for reasoning; no critic; advantages are group-relative z-scores.

Ch-46 uses G=8 (the "typical small" figure) because the lab is resource-bounded and the signal pattern that produces each failure mode does not need G=32 to appear. For a production RL training run, the source's own §Technical Details suggest G=16–32 for reasoning; ch-46 documents this as "what you'd scale up to after the post-mortem."

---

## KL estimator — why the `kl` signal is positive by construction

Source §Technical Details:

> **KL estimator:** all three default to k3 (`(π_ref/π) − 1 − log(π_ref/π)`) — see **[[kl-control-rlhf]]**.

The k3 estimator is unbiased and always ≥ 0. Ch-46's §3 logging reads this from TRL's `_metrics[mode]["kl"]` directly. A negative or NaN KL log is an immediate halt signal (see §7 Acceptance criterion #3's implicit check).

---

## Connections to the rest of the track

- **ch-43 (entropy/KL control)** — full-read chapter; the theory this blog operationalizes.
- **[[entropy-mechanism-llm-rl]]** — mechanistic backing for the triage order's step-3 "raise entropy coefficient" advice.
- **[[trl-grpo]]** / **[[openrlhf-ppo]]** / **[[verl-grpo]]** — the three frameworks this source compares; ch-46 uses TRL but the signal set is framework-independent.
- **[[kl-control-rlhf]]** — the KL-estimator reference.
- **[[sampling-temperature-schedule]]** — the triage order's step-2 "bump rollout T" reference.
