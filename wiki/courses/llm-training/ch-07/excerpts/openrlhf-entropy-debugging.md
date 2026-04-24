---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md
source_url: https://github.com/OpenRLHF/OpenRLHF
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF / verl / TRL — the entropy-collapse triage ladder

**Source library:** `wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md`
**Sources:** Maintainers of OpenRLHF (Jian Hu et al.), verl (ByteDance), and HuggingFace TRL; living documents (framework READMEs, GitHub issues tagged "entropy" / "KL" / "collapse").

---

## Why this source anchors ch-07

Ch-07 §6's RL-specific failure section is built on this source's triage ladder. The value of the source is *convergent practice*: three independent open-source RL frameworks — each maintained by a different team, each with different design choices — have landed on the same five-step diagnostic procedure. When three independent implementations converge on a rule, the rule is a law of the problem rather than an opinion.

The source states the triage literally:

> *"When entropy collapses in an open-source RL run, follow the community-standard triage: (1) confirm KL-to-reference term is on and finite, (2) bump rollout temperature by 0.1–0.2, (3) raise entropy coefficient an order of magnitude, (4) check advantage normalization is per-batch zero-mean unit-var, (5) only then suspect the reward signal."*

The ordering is not random: it is ordered by cost to test. Each step is cheaper to verify than the last, and the first four cover ~90% of real incidents. The fifth — suspecting the reward signal — requires retraining the RM or rewriting the verifier, which is the most expensive intervention. Ch-07's diagnostic-tree philosophy is this ordering applied to RL.

---

## The standard logged metrics — what every RL dashboard must have

From the source:

> *"Standard logged metrics: per-token entropy, per-batch KL(π‖π_ref), PPO ratio mean/std, reward mean/std, clipped-fraction, response length histogram. If a run drops per-token entropy below ~0.1 nats and reward hasn't already saturated, it's diagnostic of collapse, not of convergence."*

The 0.1-nats threshold is the operational detection rule. Under bf16 with a standard LM head, a post-softmax distribution with entropy 0.1 nats is concentrated on essentially one token per position — the policy has become nearly deterministic. If reward is still climbing, fine; if reward has plateaued, the policy is stuck exploring a vanishingly small neighborhood and cannot recover without intervention.

Ch-07 §6's entropy-dashboard table maps directly onto the metrics list. Notice the six-item composition: *entropy, KL, ratio, reward, clipped-fraction, length*. Each is a separate invariant with a distinct failure mode:

| Metric | Failure mode it catches |
|---|---|
| per-token entropy | collapse (§6) |
| KL(π‖π_ref) | unbounded drift from reference |
| PPO ratio | NaN from π_old underflow (§6) |
| reward mean/std | reward signal broken or collapsed advantage (§1c) |
| clipped-fraction | clip threshold mis-set (§2) |
| response length | reward hacking / length exploitation (§6) |

A production RL trainer logs all six. Missing any one leaves a diagnostic hole.

---

## The KL estimator — why k3 is canonical

From the source:

> *"KL estimator: all three default to k3 (`(π_ref/π) − 1 − log(π_ref/π)`) — see [[kl-control-rlhf]]."*

The k3 estimator is an unbiased estimator of `KL(π_ref ‖ π)` with lower variance than the naive `log(π_ref/π)` form. But it contains a `log(π_ref/π)` term — ch-07 §1b's `log(0)` NaN surface. If `π_ref` has assigned zero probability to a token that the current policy is sampling (unlikely in the policy direction), or `π` has collapsed to zero on a token the reference thought probable (the RL-collapse direction), the estimator NaNs.

Defensive implementation: clamp both probabilities at `exp(-50)` before division. The source doesn't give this code, but every framework ships it in their KL-computation utility; a hand-rolled k3 that skips the clamp NaNs as soon as the policy starts collapsing — which is the exact moment you most need the KL signal to be reliable.

---

## Community-standard defaults — the convergent baseline

From the source (Technical Details):

> *"Entropy computation: exact categorical entropy at each position, averaged over valid (non-pad) tokens.*
> *Clip range `ε`: 0.2 (PPO); GRPO often 0.2 on the ratio.*
> *Learning rate: 1e-6 to 5e-6 for 7B-class policies with bf16, halved for 70B.*
> *Rollout length: 1k–4k for standard RLHF; 8k–32k for reasoning RL (DeepSeek-R1-style).*
> *KL estimator: all three default to k3.*
> *Evaluation vs training sampler: eval at T = 0.0 or T = 0.6 with top_p = 0.95; training always at higher T / full support."*

Notice the eval-vs-training sampler distinction. Training uses `T = 1.0, top_p = 1.0` (full support) because PPO needs gradient on the tails of the distribution; eval uses lower temperature to get a reproducible answer. A common §6 bug: someone sets the training sampler to `T = 0.6, top_p = 0.95` thinking "this is what inference will see" — which collapses the exploration distribution and kills the policy gradient signal within a few hundred updates. The fix is obvious once named; the bug is common because the reasoning "train how you infer" sounds correct.

---

## The advantage-normalization footgun — OpenRLHF vs TRL default divergence

The source is explicit about the single most common silent-failure bug across the three frameworks:

> *"Advantage normalization: all three offer per-batch zero-mean unit-var normalization; it is ON by default in OpenRLHF and verl, OFF by default in TRL (a recurring footgun)."*

A practitioner moving a PPO config from OpenRLHF to TRL, or vice versa, inherits different normalization behavior silently. Ch-07 §6's triage step (4) exists because this bug is observed repeatedly in framework-swap workflows. OLMo 3's "Moving SFT from Open Instruct to Olmo Core reportedly improved throughput by 8×" ([[excerpts/olmo-3]]) is an SFT-side swap; the RL-side equivalent (swap from OpenRLHF to TRL for legal/license reasons, for example) carries this exact silent change.

The debug is one config line: `advantage_normalization=True`. The difficulty is *knowing* that this is the knob to check. The source exists to make that knob famous.

---

## GRPO-specific considerations

From the source:

> *"GRPO specifics (all three): group_size = 8 is typical small; group_size = 16–32 is common for reasoning; no critic; advantages are group-relative z-scores."*

GRPO (Group Relative Policy Optimization) computes advantages as z-scores within a group of rollouts for the same prompt, eliminating the critic network. This is the RL algorithm for DeepSeek-R1-style reasoning training. The §1c /0 mode applies specifically here: if all K rollouts in a group earn the same reward (all correct, or all wrong), the group std is 0 and every advantage is NaN. For binary reward on easy prompts this is *common* — which is why GRPO implementations special-case "all-same-reward" groups by either skipping them or falling back to a tiny perturbation.

The ch-07 §1c canonical fix — `std.clamp_min(1e-6)` — applies, but in GRPO's case the semantically correct move is to *drop* the group entirely from the update, not to compute a tiny non-zero advantage. Log the drop rate: if > 10% of groups are dropped, the training distribution is too easy or too hard and the group-size needs tuning.

---

## Common failure patterns — the incident-log shape

The source provides the clearest available summary of RL failures mapped to root cause:

> *"Common failure patterns from issue trackers:*
> *- Entropy crash within 100 steps → KL term accidentally off or β too small.*
> *- Reward-but-no-entropy-change after ~1000 steps → advantage normalization misconfigured.*
> *- Sudden length explosion in rollouts → entropy healthy but reward + length are confounded (reward hacking).*
> *- NaN in PPO ratio → very aggressive update; lower LR and clip range."*

Each pattern maps to a ch-07 §6 triage step. Notice the timescales: entropy crash within 100 steps indicates a missing or mis-scaled regularization term (fast); a 1000-step plateau with healthy entropy indicates an adv-normalization or reward issue (slow, subtle). The timescale is part of the diagnostic — "when" matters as much as "what."

Reward hacking (length explosion) is a different failure mode in ch-07's ordering; it belongs to the broader "reward overoptimization" territory covered in later chapters. Ch-07 §6 touches it only as a correlate of healthy entropy (i.e. entropy is fine, but the response distribution has been gamed to earn reward).

---

## What to take from OpenRLHF / verl / TRL convergent practice for ch-07

1. **Three independent frameworks converged on the same five-step triage.** Use it in order; don't skip.
2. **Six-metric RL dashboard is the minimum.** Entropy, KL, ratio, reward, clipped-fraction, length.
3. **Advantage normalization is the #1 silent-failure footgun at framework swaps.** OpenRLHF/verl default ON, TRL defaults OFF.
4. **The 0.1-nats threshold is the operational collapse detector.** Below that, investigate; don't wait for reward to plateau.
5. **GRPO's all-same-reward group is the /0 mode's RL face.** Drop the group; don't compute a tiny advantage.

---

## Connections

- [[excerpts/gradient-clipping]] — PPO-ratio NaN mitigation is the same unscale-clamp discipline.
- [[excerpts/mixed-precision]] — log(0) in k3 KL is the §1b NaN source under any precision.
- [[excerpts/adam]] — RL's LR (1e-6 to 5e-6) is an order below SFT's; the underlying optimizer is identical.
- [[excerpts/olmo-2]] / [[excerpts/llama-3]] — production post-training pipelines that exercise this triage in practice.
- [[ch-07]] — §1c (/0 in advantage), §6 (entropy-collapse triage), §7 (the advantage.std < 1e-4 assertion).
