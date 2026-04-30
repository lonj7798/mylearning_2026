---
chapter: ch-46
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/entropy-mechanism-llm-rl.md
source_url: https://arxiv.org/abs/2505.22617
created_at: "2026-04-23"
---

# Excerpt: Entropy Mechanism — the §5(b) post-mortem

**Source library:** `wiki/raw-data/llm-training/papers/entropy-mechanism-llm-rl.md`
**Artifact:** R = -a·exp(H) + b law; covariance theorem; Clip-Cov / KL-Cov interventions; H<0.1 nats collapse signal.

---

## Why this source is the §5(b) failure mode

Ch-46's second canonical failure mode is entropy collapse. Cui et al. 2025 gives both the *symptom* (a quantitative predictive law on R given H) and the *mechanism* (the covariance of log-prob with advantage burns entropy). The ch-46 signal pattern ("entropy drops below 0.1 nats within the first ~200 updates") is this source's collapse threshold verbatim.

---

## The empirical law ch-46 fits on the first 10% of steps

Source §Key Contributions:

> **Empirical law:** Across >20 models and settings, `R = -a·exp(H) + b` — once H is small, further entropy loss yields diminishing reward; the "performance ceiling" is reached as `H → 0`.

Ch-46 §5(b) uses this as a *predictive* tool: fit `a, b` on the first ~10% of training steps, extrapolate the ceiling. If the extrapolated ceiling is below the target (e.g. pass-rate 0.5 vs target 0.65), raise β_KL or add entropy regularization *before* spending the rest of the compute budget on a run that will saturate.

---

## The collapse threshold — what ch-46's `entropy < 0.1` alarm fires on

Source §Technical Details:

> **Collapse signal (practical):** token-level `H < 0.1` nats sustained for multiple updates.

H in nats at 0.1 means the next-token distribution is near-deterministic — top-1 probability > 0.9 on average. Ch-46 §3 Instrumentation's `entropy` signal is this `H`. Once it drops below 0.1, the reward trajectory is almost entirely explained by the already-baked-in policy; further RL does essentially nothing except amplify existing preferences.

---

## The mechanism — why high-advantage common tokens are the culprit

Source §Key Contributions:

> **Mechanistic theorem:** For softmax policies under policy-gradient updates, the expected change in token entropy is proportional to `-Cov_{a~π}(log π(a|s), A(s,a))` — large advantages on already-high-probability tokens are what burn entropy.

This is the mechanism the ch-46 §5(b) post-mortem cites. Tokens where `π(a|s)` is already high *and* the advantage `A(s,a)` is positive have large covariance, and each such token drives a disproportionate entropy drop. RLVR with binary rewards hits this regime hard: once the policy starts solving easy prompts, the positive-advantage tokens are the already-likely "1+1=2" style tokens, and entropy dies.

---

## Why vanilla entropy bonus fails at LLM scale

Source §Technical Details:

> **Vanilla entropy bonus** (A2C-style) — adds `+ β · H(π)` to the loss, where the paper empirically found β in `{1e-4, 1e-3, 1e-2}` either under-corrects or over-corrects; treating all tokens symmetrically hurts high-quality trajectories.

This is the subtle point ch-46's §5(b) memo must address: a reader's instinct is to "just turn on entropy bonus". The paper's ablation says no — symmetric entropy bonus either doesn't help or destroys quality. The asymmetric fixes (Clip-Cov, KL-Cov) targeting only the top-covariance tokens are the attested remedy.

---

## Clip-Cov and KL-Cov — the two proposed fixes

Source §Technical Details:

> **Clip-Cov:** rank tokens by `p_t · A_t` per batch, set gradient of the top fraction (e.g. 2%) to zero.
> **KL-Cov:** for those same top-covariance tokens apply `β_KL · KL(π_new‖π_old)` (forward, token-level, k3 approximation).

Practical ch-46 integration: TRL's `top_entropy_quantile` parameter implements the Clip-Cov family — it keeps gradient on high-entropy tokens only (dual of masking out high-covariance tokens). The ch-46 §5(b) fix cites this parameter as the one-line remediation; the KL-Cov variant is cited as a future direction for the memo's §5 "next instrumentation" slot.

---

## Algorithm-agnosticism — why Option B has this as a universal risk

Source §Key Figures/Tables to Study:

> **Fig. 1** (entropy vs step across algorithms): same exponential-decay shape regardless of PPO/GRPO/RLOO — collapse is algorithm-agnostic.

The ch-46 HTML `rl-sweep.html` `B-entropy` view illustrates this shape for GRPO specifically, but the paper's point is that PPO, GRPO, and RLOO all exhibit the *same exponential decay* curve. Picking a different RL algorithm does not avoid the problem; only structural fixes (KL-to-ref budget, covariance-targeted intervention, rollout temperature, top-entropy masking) do.

---

## Why ch-46 Option A can collapse too

Source notes apply in principle to policy-gradient RL. Though Option A (DPO) is not a rollout-based method, low β in DPO *also* drives log-ratio runaway, which is analogous: the policy's sharp updates on preference violations push the distribution to a narrow mode. The `A-entropy` curve in the HTML companion reflects this — β=0.05 in DPO shows a collapse-like shape even without an explicit rollout loop.

---

## Connections to the rest of the track

- **ch-43 (entropy and KL control)** — the full-read chapter; ch-46 lab operationalizes it.
- **[[grpo]]** — the base RL loss the collapse manifests in.
- **[[openrlhf-entropy-debugging]]** — practitioner-level triage that Uses Cui 2025 as its mechanistic backing.
- **[[kl-control-rlhf]]** — KL-to-reference budget is the *symmetric* fix; Cui's Clip-Cov/KL-Cov are the *asymmetric* (targeted) fix.
- **[[deepseek-r1]]** — the R1 recipe's empirical stability is partly explained by this paper's analysis.
