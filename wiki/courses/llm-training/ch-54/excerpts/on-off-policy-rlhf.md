---
chapter: ch-54
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/on-off-policy-rlhf.md
source_url: https://arxiv.org/abs/2404.14367
created_at: "2026-04-23"
---

# Excerpt: Tang 2024 — the 80/20 decomposition ch-54 §3 rests on

**Source library:** `wiki/raw-data/llm-training/papers/on-off-policy-rlhf.md`
**Artifact:** the isolation of distribution shift (80%) vs variance reduction (20%) as explanations for PPO-over-offline-DPO.

---

## Why this source anchors ch-54

§3 of ch-54 makes a strong claim: the on-policy-vs-off-policy gap in LM-RL is *almost entirely a distribution problem*, not an algorithm problem. Tang 2024 is the load-bearing evidence. Before this paper, the community split roughly into "DPO is simpler, so it should win" and "PPO has a critic, so it should win." Both framings were wrong — both algorithms converge to similar performance *provided their training data is drawn from the current policy*. The question moved from "which loss?" to "how fresh?"

---

## The decomposition §3 requires you to memorize

From the source (line 15, slightly paraphrased in-page):

> We find consistent evidence that (1) the primary cause of the gap is distribution shift — DPO trains on samples from a distribution different from the policy's own; (2) iterative (on-policy) DPO largely closes this gap; (3) PPO's advantage over DPO vanishes when DPO is made on-policy.

And (line 21):

> Provides a decomposition of the performance gap into (i) distribution-shift contribution (≈80%) and (ii) variance-reduction contribution (≈20%).

ch-54 §3 turns this into:

```
gap(algorithm) ≈ 0.8 · gap(distribution_shift) + 0.2 · gap(variance) + ~0 · gap(algebra)
```

The "algebra" term is what most algorithm-preferring debates argue about; the empirical answer is that it contributes roughly zero at this scale.

---

## The iterative-DPO recipe §3 quotes

From the source (line 32):

> **Iterative DPO recipe:**
>   - Each step: sample 2 responses per prompt from current π_t.
>   - Label with a frozen RM → chosen/rejected.
>   - DPO update with β=0.1, 1 grad step per pair, reference = π_0 (fixed).

Three things to notice: (a) the reference π_0 is **frozen** across the iteration — the implicit-reward anchor stays fixed while the policy moves, matching the DPO derivation; (b) the preference labels come from a **frozen RM** (not human), so this is synthetic in the [[west-of-n]] sense; (c) one gradient step per pair — not full batches — which keeps the "distribution of training data" tightly tied to π_t.

---

## Figure 3 — the KL-vs-reward Pareto

From the source (line 26):

> **Figure 3 (KL-vs-reward Pareto frontier):** iterative DPO dominates offline DPO at every KL budget.

"At every KL budget" is the important phrase. Not "at the right KL" or "given tuning" — uniformly dominates. This is why §3 of ch-54 treats on-policy as the default starting point rather than a tuning-dependent option.

---

## Why [[policy-coverage-loss]] is the follow-on

Tang measures shift as a scalar gap. [[policy-coverage-loss]] sharpens it into a shape: a source policy is useful iff it covers the target-optimal policy's support. Offline DPO on Anthropic-HH helps helpfulness tasks because HH pairs cover helpful response regions; it helps math tasks less because HH covers the wrong actions. ch-54 §3 uses both: Tang for "how much is lost" and coverage for "why this particular source."

---

## The staleness-bounded corollary §5 inherits

At the async-training end of ch-54, this paper's framing becomes: the off-policyness in async RL is parameterized by `k = queue_depth + partial_rollout_depth`. Tang's gap collapses monotonically in `k` — at k=1 the gap is negligible, at k=5 you are approaching offline-DPO territory. That is why ch-54 §5 caps queue depth at 1–2 and watches `vllm_kl`.

---

## Connections

- **ch-54 §3** — gap decomposition is the source of the quoted 80/20 split.
- **ch-54 §5** — staleness `k` is the async version of Tang's distribution-shift axis.
- **ch-37 / ch-38** — the [[dpo]] chapter whose offline baseline this paper dissects.
- **[[trl-online-dpo]]** — the implementation of iterative DPO Tang motivates.
- **[[policy-coverage-loss]]** — the coverability lens that sharpens "distribution shift."
