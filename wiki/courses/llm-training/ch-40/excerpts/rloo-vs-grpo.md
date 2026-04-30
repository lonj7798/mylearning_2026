---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rloo-vs-grpo.md
source_url: (synthesized reference — see component papers)
created_at: "2026-04-23"
---

# Excerpt: RLOO vs GRPO — the equivalence-in-the-limit argument

**Source library:** `wiki/raw-data/llm-training/papers/rloo-vs-grpo.md`
**Artifact:** Synthesized comparative reference drawing from Ahmadian 2024, Shao 2024, Liu 2025, Hu 2025, plus practitioner context from HF / OpenRLHF / verl docs.

---

## Why this source anchors ch-40 §6

Ch-40 §6's closing claim — "RLOO (large k) ≈ GRPO without /std without clip ≈ Dr.GRPO, they are the same estimator" — is the load-bearing theoretical statement of the chapter. Without it, the four variants look like four separate algorithms; with it, they are four points in one parameter space (keep/drop std-norm, keep/drop clip, keep/drop 1/|o_i|, choose baseline scope). This source is the only place the argument is written out in one place.

---

## The side-by-side table ch-40 §6 reproduces

Source lines 17–27:

| Axis | RLOO (Ahmadian 2024) | GRPO (DeepSeekMath 2024) |
|------|----------------------|--------------------------|
| Samples per prompt | k ∈ {2, 4} typical | G ∈ {8, 16, 32, 64} |
| Baseline | `b_i = (1/(k−1)) Σ_{j≠i} R_j` | `mean(R_1..R_G)` |
| Normalization | none (raw reward − baseline) | / std(R_1..R_G) |
| Importance ratio clip | no | yes, ε = 0.2 |
| KL placement | shaped per-token reward | inside the loss, k3 estimator |
| Epochs per rollout | 1 | 1 |
| Value network | no | no |

This is the table ch-40 §6 extends by adding REINFORCE++ and Dr.GRPO rows.

---

## The equivalence chain (ch-40 §6 relies on this)

Source lines 38–42:

1. **RLOO baseline → mean at large k**:
   `b_i = (1/(k−1)) Σ_{j≠i} R_j = (kR̄ − R_i) / (k − 1) = R̄ − (R_i − R̄) / (k − 1)`.
   As k → ∞, the correction term `(R_i − R̄)/(k−1) → 0`, so `b_i → R̄ = mean(R)`.
   Therefore RLOO's advantage `R_i − b_i → R_i − mean(R)` — *exactly* Dr.GRPO's advantage.

2. **Std-normalization is optional**: GRPO adds `/std(R)` on top. Dr.GRPO removes it. If we set the std-norm aside, GRPO's advantage = Dr.GRPO's advantage = RLOO's large-k advantage.

3. **PPO-clip rarely binds at 1 epoch**: with μ=1 (single step per rollout), π_θ has barely drifted from π_θ_old, so ρ_{i,t} ≈ 1 and `clip(ρ, 1±ε)` almost never differs from ρ. The clip is mostly dormant.

**Conclusion**: RLOO (k→∞) ≡ GRPO − (/std) − (clip) ≡ Dr.GRPO. The four variants collapse to one estimator plus three optional decorations (std-norm, clip, 1/|o_i|), each of which introduces a known bias or variance tradeoff.

---

## When each wins empirically (ch-40 §8 guideline comes from this)

Source lines 45–53:

| Scenario | Winner | Why |
|----------|--------|-----|
| Continuous RM score, small k | GRPO | std-norm stabilizes gradient magnitude across prompts of varying spread |
| Verifiable 0/1 reward, large G | Dr.GRPO > GRPO ≈ RLOO | std(R) degenerates when all-right or all-wrong |
| Small batch / tight memory | RLOO, k=2 | least overhead |
| Reasoning RL with long chains | Dr.GRPO or REINFORCE++ | avoid length inflation |
| Very large global batch, single rollout | REINFORCE++ | global advantage normalization |

Ch-40 §8's recommendation — default to Dr.GRPO for reasoning RL with verifiable rewards; RLOO k=2–4 for plain RLHF with a scalar RM — is exactly this table compressed.

---

## Practitioner stack ranking (source lines 54–59)

1. **Dr.GRPO** as default for reasoning RL.
2. **RLOO k=2–4** for plain RLHF with a scalar RM (no process reward).
3. **PPO** only when you already have a trained value network or want explicit entropy control via the value loss.
4. **REINFORCE++** when k ≥ 2 rollouts per prompt is too expensive.

---

## What this source adds beyond the primary papers

Each primary paper (RLOO, GRPO, Dr.GRPO, REINFORCE++) argues its own case. This reference is the first place the four are compared as siblings in one family. The equivalence-chain argument is what lets ch-40 §6 claim "the critic is settled; the remaining debate is *which* normalization" — a narrative point no single primary paper makes.

---

## Connections to the rest of the track

- [[rloo]], [[grpo]], [[dr-grpo]], [[reinforce-plus-plus]] — the four primaries.
- [[vanilla-pg]] — the common ancestor.
- [[ppo]] — the baseline the family subtracts from.
- [[nathan-lambert-grpo]] — practitioner parallel to this comparative view.
- [[trl-grpo]], [[verl-grpo]] — where the choices are encoded in code.
