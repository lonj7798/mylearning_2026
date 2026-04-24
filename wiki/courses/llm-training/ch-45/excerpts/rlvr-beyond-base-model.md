---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlvr-beyond-base-model.md
source_url: https://arxiv.org/abs/2504.13837
created_at: "2026-04-23"
---

# Excerpt: RLVR Beyond Base Model — the pass@k critique

**Source library:** `wiki/raw-data/llm-training/papers/rlvr-beyond-base-model.md`
**Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, Gao Huang
**Year:** 2025 (arXiv 2504.13837)

---

## Why this source anchors ch-45

This is the 2025 paper that forces a reframing of *every* method in ch-45.
If its central claim holds, self-improvement loops do not expand the base model's
reasoning boundary; they sharpen it. That is a very different story from the one
told by the 2024 DeepSeek-R1 paper and everything that immediately followed.
Ch-45 readers who will run a self-improvement loop in [[ch-46]] need to see this
critique *before* they run their experiment, because it changes the evaluation
protocol (pass@1 is not sufficient; you need pass@large-k).

---

## The central claim, attested verbatim

Source line 7:

> Under large-pass@k evaluation, many RLVR gains look like improved sampling
> efficiency rather than genuine expansion of the base model's reasoning boundary.

Source line 15:

> The paper argues that RLVR does not, in its current form, reliably create
> fundamentally new reasoning patterns. Across math, coding, and visual reasoning
> tasks, RL-trained models beat base models at small k, but at large k the base
> models often match or exceed them. The authors interpret this as evidence that
> RLVR mostly redistributes probability mass toward already-existing successful
> paths, while narrowing exploration and reducing the broader coverage of solvable
> problems.

Parse this carefully. The paper is not claiming RLVR is useless. It is claiming
RLVR does a **different** thing than the field has been attributing to it.
RL improves the probability mass on already-existing good paths; it narrows
the distribution; at k=1 this is a win; at k=128 it can be a loss because
some problems the base model could occasionally solve at high k are no longer
reachable from the RL-trained distribution.

---

## The evaluation move

Source lines 19-22:

> Introduces large-k pass@k as a probe of reasoning-capacity boundary rather
> than average-case performance.
> Shows a recurring pattern: RL improves pass@1, base model wins at high k.
> Frames RLVR as sampling-efficiency improvement rather than capability-boundary expansion.
> Contrasts RL with distillation, arguing that distillation can truly introduce
> knowledge beyond the base model.

The reframing is the contribution. "Is this model better?" becomes an ambiguous
question. Better on pass@1 or better on pass@128? If you do not specify, you can
get opposite answers from the same two checkpoints.

---

## Why this matters for every ch-45 method

Every self-improvement loop in ch-45 uses an outer-loop that *narrows the
policy distribution*:

- Self-Rewarding → DPO margin → narrow toward judge-argmax.
- SPIN → DPO margin → narrow toward data distribution.
- ReST-EM → SFT on correct-only → narrow toward survivor set.
- R1-Zero → GRPO advantage → narrow toward high-reward rollouts.

If Yue et al.'s interpretation is right, **all four** trade exploration breadth
for pass@1 sharpness. The exploit that saved you on problem X at k=64 in the
base model may not be reachable at k=64 from your RL-trained checkpoint.

The practical implication for [[ch-46]]'s lab: instrument `pass@k` across
difficulty buckets, not just aggregate pass@1. If your RL gain is concentrated
in the "easy" bucket and your "hard" bucket regresses at large k, you have
evidence of exactly the pattern Yue et al. describe.

---

## The distillation contrast

Source line 22 (quoted above) flags distillation as the "cleaner route for
importing new capabilities." The mechanism: distillation copies a **distribution**
from a stronger teacher. RLVR moves probability mass within the student's
**own** distribution. If the student's base distribution has zero mass on a
solution path, RLVR cannot find it (no on-policy gradient); distillation can
introduce it directly via teacher tokens.

This is why DeepSeek-R1 (the full paper) includes a distillation stage after
the R1-Zero-style pure RL: distilling R1 traces into Qwen-7B / Llama-8B produces
models that can solve problems neither the student base nor the RL-trained
intermediate could solve alone. The distillation is *not* redundant with the RL;
they do different things.

---

## What the critique does not say

The paper does *not* claim:
- RLVR is useless at pass@1 (it isn't — the pass@1 gains are real).
- Self-improvement loops do not work (they do — ReST-EM's MATH 34 → 50 is real).
- The base model is always enough (it isn't — most users care about pass@1).

It claims that **pass@1 gain is not evidence of new reasoning capability**, and
that the 2024 field over-attributed "emergence" to the RL stage when a significant
fraction of the gain is distribution sharpening the base could already do under
enough sampling.

This is a **methodology refinement**, not a refutation. Ch-45 presents it that way.

---

## Connections

- Directly responds to [[deepseek-r1]] and [[excerpts/r1-zero-analysis]].
- Consistent with the "emergence requires a reasoning-pretrained base" finding
  from ORZ (see [[excerpts/r1-zero-analysis]] Finding 2).
- Cross-referenced by [[spurious-rewards-rlvr]] and [[echo-chamber-rl-post-training]]
  as part of the 2025 "RL as prior sharpening" thesis.
- Partially disputed by [[prorl]] which claims prolonged RL *can* reach new
  solution regions.
- Host chapter: [[ch-45]] §8.
- Forward to [[ch-46]] lab evaluation design: instrument pass@k not just pass@1.
