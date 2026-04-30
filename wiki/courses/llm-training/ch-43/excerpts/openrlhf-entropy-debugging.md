---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md
source_url: https://github.com/OpenRLHF/OpenRLHF ; https://github.com/volcengine/verl ; https://github.com/huggingface/trl
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF / verl / TRL — the entropy-collapse triage protocol

**Source library:** `wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md`
**Sources:** OpenRLHF maintainers (Jian Hu et al.); verl authors at ByteDance; HuggingFace TRL team. Synthesis of framework READMEs, GitHub issues tagged entropy / KL / collapse, community Discord digests.
**Year:** 2023–2025 (living document)

---

## Why this source anchors ch-43

The read chapter's §2 ("Collapse threshold and triage tree") distills this source into five ordered steps. The source itself is the practitioner-level consensus that has emerged from watching hundreds of RL runs crash in the same ways. Where [[excerpts/entropy-mechanism-llm-rl]] gives the theorem and [[excerpts/nathan-lambert-entropy-rl]] gives the framing, this excerpt gives the **ordered debugging protocol** — which knob to turn first, and why.

---

## Standard logged metrics

Source line 21:

> Standard logged metrics: per-token entropy, per-batch KL(π‖π_ref), PPO ratio mean/std, reward mean/std, clipped-fraction, response length histogram.

This is the minimum-viable dashboard. Missing any of these and you cannot triage a collapse — they are the observables the triage tree branches on. Most production stacks log:

- `entropy` (mean, and 10/50/90 percentile over rollout tokens)
- `kl_ref` (K1 mean; sometimes K3 for the non-negative version)
- `ratio_mean`, `ratio_std`
- `clipfrac` (fraction of tokens where PPO clip fired)
- `reward_mean`, `reward_std`, `reward_hist`
- `response_len_hist`

Response-length histogram is the under-appreciated one: reward hacking often shows up first as length explosion while entropy looks healthy.

---

## The collapse diagnostic

Source line 22:

> If a run drops per-token entropy below ~0.1 nats and reward hasn't already saturated, it's diagnostic of collapse, not of convergence.

The critical qualifier is "and reward hasn't already saturated". If reward hit the ceiling 200 steps ago and entropy has since drifted down to 0.08 nats, that is *convergence* (you got what you were going to get; policy has committed). If reward is still climbing when entropy hits 0.1, that is *collapse* (exploration is dying while the policy still has room to improve). Same entropy number, opposite diagnosis.

The operational rule: **always overlay the reward curve on the entropy curve**. The diagonal you trace in `(H, R)` space is the `R = −a · exp(H) + b` curve from [[excerpts/entropy-mechanism-llm-rl]].

---

## Default coefficient values

Source lines 22–23:

> Default entropy coefficient `c_H`: OpenRLHF and TRL default to `0.0` for LLM-RL (counter to pre-LLM PPO practice). verl exposes it with default `1e-3` on some presets. Community norm for when to raise: entropy drops faster than reward rises within the first 200 updates.
> Default KL coefficient `β`: around 0.01–0.1 of the reward scale; adaptive-KL is supported in OpenRLHF (adjust `β` to hit a target KL per batch) and is a safer default than fixed-β for new reward functions.

Two anchors for ch-46's RL lab:

- `c_H = 0` is the modern default. Raising it is a *corrective move*, not a design choice.
- `β ∈ [0.01, 0.1]` as a fraction of the reward scale. If RM rewards live in `[−1, 1]`, that means β on the order of 0.01–0.1 absolute. If RM rewards live in `[0, 100]`, scale up β accordingly.

Adaptive-KL (OpenRLHF's `AdaptiveKLController`) targets a per-batch KL and adjusts β multiplicatively. Safer than fixed β when the reward distribution is unknown.

---

## Rollout sampler defaults

Source line 24:

> Rollout sampler: all three frameworks default to `T = 1.0`, `top_p = 1.0` for training rollouts. Framework-level integration with vLLM / SGLang handles KV-cache reuse; temperature is set on the sampler, not the policy.

`T = 1.0, top_p = 1.0` is on-policy with full support, which is what PPO's importance-sampling math assumes. Departing from this (raising T, truncating top_p) introduces off-policy bias that PPO's clip does not absorb — see [[sampling-temperature-schedule]] for the full discussion.

For eval, the conventions diverge: R1-style deployment uses `T = 0.6, top_p = 0.95`; strict greedy uses `T = 0`. The "evaluate at greedy, train at T=1" split is load-bearing: you never want a dashboard to show "training reward" at low T because the reward curve will look artificially sharp.

---

## Advantage normalization — the footgun

Source line 25:

> Advantage normalization: all three offer per-batch zero-mean unit-var normalization; it is ON by default in OpenRLHF and verl, OFF by default in TRL (a recurring footgun).

This is the #2 cause of spurious collapse diagnoses (after accidentally-off β). TRL's `PPOTrainer` has `normalize_advantages=False` as default; OpenRLHF and verl default to True. If you migrate a recipe from TRL to verl (or vice versa) without changing this flag, the effective learning rate on the policy-gradient term shifts by `~std(A)`, which can be order-of-magnitude depending on reward scale.

Symptom: entropy healthy but reward moves too fast (TRL user moving to verl) or too slow (verl user moving to TRL). Fix: explicitly set `normalize_advantages` in your config regardless of framework default.

---

## The triage tree (the source's ordered recipe)

Source lines 8 and 27–31 give the diagnostic protocol the read chapter quotes:

> When entropy collapses in an open-source RL run, follow the community-standard triage:
> 1. Confirm KL-to-reference term is on and finite.
> 2. Bump rollout temperature by 0.1–0.2.
> 3. Raise entropy coefficient an order of magnitude.
> 4. Check advantage normalization is per-batch zero-mean unit-var.
> 5. Only then suspect the reward signal.

The order is not arbitrary. Each step:

1. **Is *cheapest to verify*.** Logging KL once per step tells you if it silently went to zero or NaN. One-line fix.
2. **Is *fastest to try*.** Temperature is a config change; no retraining needed. If it jolts the curve, diagnosis confirmed.
3. **Is *longer-cycle*.** Raising `c_H` needs a new run (or at least a hot-reload of the optimizer config in frameworks that support it). Do it after T has been ruled out.
4. **Is *systemic*.** Advantage normalization is a whole-recipe decision; fixing it usually means re-running from the last good checkpoint.
5. **Is *most expensive*.** Retraining the RM or re-curating prompts is days-to-weeks. Do not go here without ruling out the above.

---

## Common failure patterns

Source lines 27–31:

> Common failure patterns from issue trackers:
> - Entropy crash within 100 steps → KL term accidentally off or β too small.
> - Reward-but-no-entropy-change after ~1000 steps → advantage normalization misconfigured.
> - Sudden length explosion in rollouts → entropy healthy but reward + length are confounded (reward hacking).
> - NaN in PPO ratio → very aggressive update; lower LR and clip range.

Each failure has a characteristic signature. The "entropy healthy, rollouts exploding in length" pattern is the important one — it says *not all failures are entropy failures*. Reward hacking (ch-42) can leave entropy intact while length and format collapse. Monitor both.

---

## GRPO specifics

Source line 26:

> GRPO specifics (all three): `group_size = 8` is typical small; `group_size = 16–32` is common for reasoning; no critic; advantages are group-relative z-scores.

Two things to note for ch-43:

- Group size trades off compute for baseline quality. Smaller groups have noisier baselines, which hurts entropy monitoring (you can't tell group-noise from policy drift). 8 is the floor; 16–32 is the reasoning-RL default; larger is unusual.
- "No critic" means entropy dynamics are the primary diagnostic signal. In PPO, critic drift can confound collapse diagnosis; in GRPO it cannot.

---

## KL estimator convention

Source line 41:

> KL estimator: all three default to k3 (`(π_ref/π) − 1 − log(π_ref/π)`) — see [[kl-control-rlhf]].

This is the convergence point. Regardless of framework, the modern KL estimator is k3. Legacy K1 persists in some reward-shaping paths ([[excerpts/entropy-logging-patterns]] shows both side-by-side), but anywhere a new estimator decision is being made in 2025, it's k3.

---

## Connections

- Read-chapter §2 uses this excerpt's five-step triage tree.
- [[excerpts/entropy-logging-patterns]] — shows exactly where in the code each knob lives.
- [[excerpts/nathan-lambert-entropy-rl]] — the framing this protocol formalizes.
- [[excerpts/entropy-mechanism-llm-rl]] — once the protocol exhausts the "turn a knob" fixes, Clip-Cov / KL-Cov are the surgical next step.
- ch-46 (RL lab) — the lab's instrumentation checklist is a superset of this triage tree; the lab's failure post-mortem deliverable uses this ordering.
