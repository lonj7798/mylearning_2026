---
chapter: ch-16
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/on-off-policy-rlhf.md
source_url: https://arxiv.org/abs/2404.14367
created_at: "2026-04-23"
---

# Excerpt: On-vs-off-policy RLHF — distribution shift as the root pathology

**Source library:** `wiki/raw-data/llm-training/papers/on-off-policy-rlhf.md`
**Paper:** Tang, Guo, Zheng, Calandriello, Cao, Tarassov, Munos, Valko, Cheng, Dabney (DeepMind), "Understanding the Performance Gap Between On-Policy and Off-Policy RLHF" (2024).

---

## Why this source anchors ch-16

This paper provides the theoretical spine for ch-16 §3. Where [[excerpts/replay-buffer-rlhf]] gives the operational "don't replay trajectories" rule, this paper gives the *reason*: the pathology that kills offline DPO — distribution shift — is the same pathology that kills trajectory replay. Once you see them as the same mechanism, the no-trajectory-replay rule becomes obvious.

Ch-16 cites this source at §3.2 (the claim that ≈80% of PPO's advantage over offline DPO is distribution shift) and uses it as the theoretical anchor for §4's coverage discussion.

---

## The core claim

From the source (Abstract, Key Contributions):

> We rigorously characterize the performance differences across a range of RLHF tasks and find consistent evidence that (1) the primary cause of the gap is distribution shift — DPO trains on samples from a distribution different from the policy's own; (2) iterative (on-policy) DPO largely closes this gap; (3) PPO's advantage over DPO vanishes when DPO is made on-policy.

The key reframing: **"PPO vs DPO" is not the right comparison.** The right comparison is "on-policy vs off-policy preference optimization," and once you control for that, PPO and iterative DPO are roughly equivalent. This is a 2024 finding that shaped how 2025 systems (Tülu 3, OLMo 2, Llama 3) made their DPO-vs-PPO choices.

---

## The ≈80% decomposition

From the source (Key Contributions):

> Provides a decomposition of the performance gap into (i) distribution-shift contribution (≈80%) and (ii) variance-reduction contribution (≈20%).

Ch-16 §3.2 quotes the 80% number as evidence that distribution shift is the dominant pathology, not a second-order effect. This matters for the chapter's argument because trajectory replay is distribution shift *inside a PPO loop* — you're using stored trajectories from `π_old` as if they came from `π_θ`. The same 80%-dominant mechanism should hit trajectory replay too, and empirically it does ([[excerpts/replay-buffer-rlhf]] and DeepSeek-R1 §3.2 both confirm).

The 20% variance-reduction contribution is what PPO's clip and advantage-normalization buy. This is real but secondary. A system without trajectory replay but with a naive advantage estimator will underperform a system with proper variance reduction — but the gap between "no replay + clean estimator" and "replay + clean estimator" is much larger than the gap between "clean estimator" and "noisy estimator."

---

## The iterative-DPO recipe and what it tells us about curriculum

From the source (Technical Details):

> **Iterative DPO recipe:**
>   - Each step: sample 2 responses per prompt from current π_t.
>   - Label with a frozen RM → chosen/rejected.
>   - DPO update with β=0.1, 1 grad step per pair, reference = π_0 (fixed).

Notice the "sample 2 responses from current `π_t`" — this is the on-policy bit. The frozen RM is a fixed labeler. The reference policy for the DPO objective is frozen at `π_0`. This is *not* trajectory replay; each round generates fresh responses at the current policy and throws them away after the gradient.

Ch-16's §3.4 and §6 implement the same pattern for RLVR. The prompt-level replay buffer stores *prompts*, not responses. Every time a prompt is sampled, fresh responses are generated from `π_θ`. The IS correction that would be needed for true trajectory replay is skipped because no stored response ever enters the gradient.

---

## The coverage framing

From the source (Key Contributions):

> Formalizes a coverage argument: DPO on off-policy pairs is biased because the implicit reward's normalization constant depends on the sampling distribution.

This is the theoretical root of ch-16 §4's policy-coverage principle (the (d) point in the curriculum-principles list). When the source-data distribution does not cover the target policy's behavior, the implicit reward signal is biased. For trajectory replay in PPO, the same argument applies: the stored `y ~ π_old` may not cover the region `π_θ` now wants to explore, so the gradient estimated from those stored samples is biased toward `π_old`'s bias.

[[excerpts/policy-coverage-loss]] extends this to a formal transfer bound. Ch-16 uses the coverage framing at a looser operational level: "a prompt whose current pass-rate is zero is out of coverage."

---

## Figure 6 — the distribution-shift control experiment

From the source (Key Figures/Tables to Study):

> **Figure 6 (synthetic distribution-shift control):** as you increase the mismatch between DPO training data and policy, the gap grows predictably.

This is the cleanest experimental evidence for the 80% claim. The paper constructs synthetic setups where they control the distribution-shift magnitude between DPO training pairs and the current policy; the performance gap tracks the shift predictably. Ch-16 doesn't reproduce the figure but uses its implication: distribution shift is a *continuous* pathology, not a binary on/off one. Trajectory replay at `K = 5` steps of lag has less shift than at `K = 50` steps — consistent with ch-16 §3.2's `K = 20, T = 1000 → E[ρ] ≈ 2.7` escalation table.

---

## What this excerpt unlocks

- **ch-16 §3.2** — the IS-ratio derivation is the per-token form of the distribution-shift argument.
- **ch-16 §4(d)** — the coverage principle is directly from this paper.
- **ch-17 (lab)** — writing an ablation memo; the "iterative DPO ≈ PPO" reframing is a good example of a result that looks like a paradigm shift until you add the on-policy control, then becomes a careful-comparison lesson.

## Connections

- [[excerpts/replay-buffer-rlhf]] — operational companion; this source is the theory, that one is the code.
- [[excerpts/policy-coverage-loss]] — formal transfer bound that extends the coverage argument.
- [[ch-16]] — §3.2 (IS-ratio derivation as distribution-shift instance), §4 (coverage principle).
