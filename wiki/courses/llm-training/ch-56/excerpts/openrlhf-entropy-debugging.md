---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md
source_url: https://github.com/OpenRLHF/OpenRLHF (issues tagged "entropy" / "KL" / "collapse")
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF practitioner defaults + community-standard triage

**Source library:** `wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md`
**Authors:** OpenRLHF maintainers (Jian Hu et al.), verl / TRL teams, community issue trackers

---

## Why this source anchors ch-56

Framework internals make no sense without their *defaults* and
*triage order*. This source is the practitioner-literature synthesis
that explains why OpenRLHF defaults `entropy_coef` to 0.0, why
advantage normalization is ON, and what the canonical fix sequence is
for a collapsing entropy curve. Ch-56 §5 and §7 are direct
digests of it.

---

## The community-standard triage, attested

Source §Core Insight and §Guideline:

> When entropy collapses in an open-source RL run, follow the
> community-standard triage:
> (1) confirm KL-to-reference term is on and finite,
> (2) bump rollout temperature by 0.1–0.2,
> (3) raise entropy coefficient an order of magnitude,
> (4) check advantage normalization is per-batch zero-mean unit-var,
> (5) only then suspect the reward signal.

Step 1 is the highest-yield: a misconfigured `kl_ctl` (β too small, or
KL estimator negative) accounts for most first-month failures. Step 4
is the OpenRLHF-specific gotcha because advantage normalization is
only ON by default in OpenRLHF and verl — not in TRL.

---

## Default hyperparameters

Source §Key Points and §Technical Details:

| Knob | OpenRLHF | verl | TRL |
|---|---|---|---|
| `entropy_coef` | 0.0 | 0.0 (some presets 1e-3) | 0.0 |
| KL mode | adaptive (safer default for new rewards) | fixed or adaptive | fixed + controller |
| `β` (KL-to-ref) | 0.01–0.1 of reward scale | same | same |
| Rollout `T` | 1.0 | 1.0 | 1.0 |
| `top_p` | 1.0 | 1.0 | 1.0 |
| Advantage normalization | ON | ON | OFF (footgun) |

The source is explicit:

> adaptive-KL is supported in OpenRLHF (adjust `β` to hit a target KL
> per batch) and is a safer default than fixed-β for new reward
> functions.

This is why ch-56 §5 marks AdaptiveKLController as the OpenRLHF
default — the community converged on it.

---

## Four attested failure patterns

Source §Key Points:

> Common failure patterns from issue trackers:
> - Entropy crash within 100 steps → KL term accidentally off or β too small.
> - Reward-but-no-entropy-change after ~1000 steps → advantage normalization misconfigured.
> - Sudden length explosion in rollouts → entropy healthy but reward + length are confounded (reward hacking).
> - NaN in PPO ratio → very aggressive update; lower LR and clip range.

Ch-56 §7's failure-mode map is the same four patterns, mapped to
OpenRLHF-specific fixes.

---

## GRPO defaults, attested

Source §Key Points:

> GRPO specifics (all three): `group_size = 8` is typical small;
> `group_size = 16–32` is common for reasoning; no critic;
> advantages are group-relative z-scores.

This matters for ch-56 because OpenRLHF's PPO trainer is the same
PolicyLoss with GAE swapped for group z-score — GRPO is not a separate
trainer, it is a different advantage computation feeding the same
`PolicyLoss.forward`.

---

## Eval-vs-training sampler

Source §Technical Details:

> Evaluation vs training sampler: eval at `T = 0.0` or `T = 0.6`
> with `top_p = 0.95`; training always at higher `T` / full support.

If you eval at training-`T`, your pass-rate numbers are noisy and do
not reflect deployment behavior. If you train at eval-`T`, entropy
collapses immediately. This separation is a silent requirement of
every modern RL run.

---

## Connections

- [[excerpts/openrlhf-ppo]] — where these defaults are actually
  wired in.
- [[excerpts/entropy-logging-patterns]] — the metrics this triage
  sequence reads.
- [[excerpts/kl-control-rlhf]] — why adaptive-β is safer than fixed.
- Host chapter: [[ch-56]] §5 + §7.
- Forward to [[ch-57]] (TRL) — TRL's advantage-norm OFF default is
  the exception that this source flags as a footgun.
