---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/r1-zero-analysis.md
source_url: https://arxiv.org/abs/2503.20783
created_at: "2026-04-23"
---

# Excerpt: R1-Zero analyses — what the 2025 reproductions settled

**Source library:** `wiki/raw-data/llm-training/papers/r1-zero-analysis.md`
**Authors:** Liu et al. (Dr.GRPO); Hu et al. (Open-Reasoner-Zero); Pan et al. (TinyZero)
**Year:** 2025
**URLs:** arXiv 2503.20783; github.com/Open-Reasoner-Zero; github.com/Jiayi-Pan/TinyZero

---

## Why this source anchors ch-45

DeepSeek-R1-Zero's "pure-RL elicits reasoning" claim generated a 2025 wave of
reproductions that **dissected the claim piece by piece**. For ch-45 this is
the source that separates what pure-RL self-training actually does (elicit
latent capability, improve sampling efficiency) from what the field initially
*thought* it did (create new reasoning from a weak base). Every quote below
is directly cited in ch-45's §8 because the distinction matters for picking
your self-improvement loop.

---

## Finding 1 — GRPO has two exploitable biases (Dr.GRPO, Liu 2025)

Source lines 21-22:

> Dr.GRPO (Liu 2025): identifies two biases in the standard GRPO advantage:
> (i) length bias from per-token mean aggregation that rewards longer correct
>     responses and longer wrong responses asymmetrically;
> (ii) difficulty bias from per-prompt std normalization that inflates
>      gradients on easy prompts.
> Proposes removing the std normalization and switching aggregation to batch-mean
> divided by (B * max_completion_length).

Mechanism. Per-token mean aggregation `(1/L) Σ_t ...` favors longer sequences
when the reward is positive and penalizes longer sequences less severely when
the reward is negative — an asymmetry that silently selects for length. The
per-prompt std normalization `A_i = (R_i − μ) / σ` inflates gradients on prompts
where σ is small (easy prompts everyone gets right), which is the opposite of
what you want.

The corrected loss (source line 34):

```
L_DrGRPO = -(1 / (B * L_max))
           * sum_{i,t} mask_{i,t}
           * min( r_{i,t} * A_i,  clip(r_{i,t}) * A_i )
with A_i = (R_i - mu_group)     -- no std normalization
```

This matters for ch-45 because **length inflation** in Self-Rewarding (Meta-Rewarding's
fix: length-bias control term) is the same structural problem as length inflation in
GRPO (Dr.GRPO's fix: corrected aggregation). Different algorithms, same failure mode.

---

## Finding 2 — the reasoning prior in the base model is load-bearing (ORZ, 2025)

Source line 22:

> Open-Reasoner-Zero (ORZ, 2025): reproduces R1-Zero emergence on Qwen2.5-7B-Base
> with GRPO + rule-based verifier + 32K context. Confirms:
>   (1) emergence happens;
>   (2) it disappears if the base is not reasoning-pretrained;
>   (3) asymmetric DAPO-style clipping eps_high=0.28 is stabilizing.

The money sentence is **(2)**. Translated: R1-Zero does *not* create reasoning
from scratch. It elicits a reasoning capability that pretraining already installed
via math/code-heavy data. If you run the exact same GRPO recipe on a base model
that was pretrained on chat data rather than reasoning data, you get neither the
length growth nor the aha moment.

This reframes the "emergent reasoning" claim from "RL creates reasoning" to
"RL crosses a detection threshold on reasoning capability that is already present
but not expressed." For ch-45 this is critical: self-improvement loops are
*eliciting* loops, not *creating* loops, and they inherit the base model's ceiling.

Source line 42 drives this home:

> Reproduction caveats: none of the papers reproduced R1-Zero from a *non-math-pretrained*
> base; the reasoning prior is necessary.

---

## Finding 3 — no PRM needed, outcome reward dominates

Source line 24:

> Shared finding: no PRM needed; outcome-only verifier is sufficient and in fact dominates.

All three reproductions (Dr.GRPO, ORZ, TinyZero) tested PRM-augmented variants and
reported no gain or a loss. The mechanism: PRMs are learned reward models, and
learned reward models are reward-hackable. The rule-based verifier is a
deterministic closed-form function with no such failure mode. [[ch-44]] discussed
the process-vs-outcome tradeoff; R1-Zero's reproductions are the strongest
empirical case for outcome-only when the outcome is cleanly verifiable.

---

## Finding 4 — RL may be sharpening, not expanding (Yue 2025)

See [[excerpts/rlvr-beyond-base-model]] for the pass@k critique in depth. The
short version: if the base model matches or exceeds the RL model at large k,
the RL step improved **sampling efficiency**, not **capability boundary**.

This is the strongest caution on R1-Zero-style training circulating in 2025.
It does not kill the paradigm — pass@1 gains are real and practically useful —
but it changes how you interpret the numbers.

---

## Finding 5 — emergence coincides with an entropy plateau, not collapse

Source line 47:

> Related to entropy-mechanism analyses ([[entropy-mechanism-llm-rl]]): emergence
> coincides with an entropy plateau, not entropy collapse.

This is a 2025 refinement on the 2024 entropy-collapse literature. In R1-Zero
training, policy entropy does *not* crash — it plateaus. The aha moment appears
during this plateau, not after it. Entropy regularization in R1-Zero (beta-KL to
reference, small) is the only thing preventing collapse, and the plateau is the
regime where the policy is still exploring but already concentrated enough to
consistently earn verifier reward.

---

## Connections

- Corrects the GRPO loss of [[grpo]] and [[verl-grpo]] — Dr.GRPO is now a supported
  `loss_type` in TRL.
- Reinforces the [[let-verify]] vs [[deepseek-r1]] tradeoff: outcome-only rewards
  beat PRM when the outcome is verifiable.
- Contradicts [[excerpts/self-rewarding-lm]] saturation: verifiable RL scales further
  than judge-based RL because the reward source isn't drifting.
- Complemented by [[excerpts/rlvr-beyond-base-model]] on the pass@k reframing.
- Host chapter: [[ch-45]] §8.
