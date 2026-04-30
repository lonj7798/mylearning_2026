---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/interplay-pretraining-midtraining-rl.md
source_url: https://arxiv.org/abs/2512.07783
created_at: "2026-04-23"
---

# Excerpt: Interplay of Pretraining, Mid-Training, and RL - the controlled framework ch-32 stands on

**Source library:** `wiki/raw-data/llm-training/papers/interplay-pretraining-midtraining-rl.md`
**Artifact:** controlled synthetic-reasoning framework isolating causal roles of pretrain / mid-train / RL

---

## Why this source anchors ch-32

This paper is the single strongest controlled evidence that **mid-training is a distinct, causally-useful stage**. Every prior claim about mid-training was either a lab's own model report (potentially confounded by proprietary data) or a small ablation inside a larger paper. This work builds a clean synthetic-reasoning testbed and isolates three causal claims:

1. RL yields true pass@128 gains only when the base model has headroom.
2. RL data must sit near the model's competence boundary.
3. Under fixed compute, **mid-training outperforms RL-only post-training**.

Ch-32's "mid-training is a real stage" argument in Section 1, its "RL is multiplicative with mid-training quality" stage-dependency note in Section 6, and its "better data shifts where each stage's contribution peaks but does not eliminate any stage" open-question framing are all downstream of this paper.

---

## The controlled framework ch-32 references

From the source (lines 32-37):

- Synthetic tasks with explicit atomic operations and parseable reasoning traces.
- Measures two forms of generalization:
  - **Extrapolative**: composing operations into harder problems.
  - **Contextual**: reusing the same reasoning under different surface contexts.

Ch-32 uses this distinction to clarify what mid-training actually teaches. Extrapolative gains require raw capability (harder compositions); contextual gains require reusable structure (same reasoning, different surface). Mid-training, per the paper, disproportionately helps contextual generalization - which is why a small high-quality mix can have outsized downstream impact. The prior learned in mid-training transfers across surface contexts that SFT and RL alone would not generalize to.

---

## The edge-of-competence result ch-32 quotes

From the source (lines 39-42):

- RL creates true capability gains only when pre-training has left enough unused capacity.
- RL data must sit near the model's competence boundary; too-easy tasks give no signal, too-hard tasks give nothing to exploit.

Ch-32's claim that RL is "multiplicative with mid-training quality" is a restatement: mid-training moves the competence boundary to a more-useful location, so the same RL run now has a better edge to push. This is why ch-32 argues that allocating compute to RL at the expense of mid-training is generally a mistake - without the right boundary, RL spins.

---

## The mid-training vs RL-only result ch-32 builds on

From the source (lines 44-47):

- Mid-training is a distinct and important stage, not just a naming variation on SFT.
- Under the paper's controlled setting, mid-training gives better results than using the same compute budget for RL-only post-training.
- Interpretation: mid-training helps install reusable priors that later RL can exploit.

Ch-32 transcribes this finding directly. The "same compute budget" qualifier matters: the comparison is not "mid-training plus RL vs RL alone" (which would trivially favor the former); it is "mid-training budget reallocated to RL vs kept as mid-training." That reallocation loses.

---

## The process-supervision finding ch-32 uses

From the source (lines 49-52):

- Adding process-level verification to outcome rewards reduces reward hacking.
- Denser signal than final-answer correctness alone; better alignment of reward with valid reasoning.
- Result: better structural fidelity, not just better top-line accuracy.

Ch-32 uses this to add a fourth "reason mid-training matters": process rewards require parsable intermediate steps, and mid-training on structured data (math with steps, code with tests) installs that parsability more reliably than raw-web pretrain. Without mid-training, process rewards have nothing to reward.

---

## What ch-32 does not claim from the source

- The paper's experiments are at a controlled scale, not frontier-scale. Ch-32 is careful to flag this: "controlled-but-small-scale" evidence.
- The paper does not compare mid-training to specific production recipes (OLMo 3 Dolmino vs Llama 3 annealing); it compares mid-training-as-concept to RL-only-as-concept.
- Specific mid-training hyperparameters, data-mix proportions, and token budgets are study-specific, not generally prescriptive.

Ch-32 uses this source as the controlled-experiment backbone and supplements it with the OLMo 3 model-flow disclosure (the production existence proof).

---

## Connections

- **[[front-loading-reasoning]]** - complementary on the pretraining side; asymmetric allocation (diversity in pretrain, quality in SFT).
- **[[olmo-3]]** - production-scale model-flow release that makes the mid-training stage explicit.
- **[[prorl]]** - policy-intervention debate this paper reframes causally.
- **[[rlvr-beyond-base-model]]** - the "does RL add new capability or reshape existing mass" question; this paper's answer is "conditional on headroom."
- **[[math-shepherd]]**, **[[lets-verify]]** - process-reward lineage this paper validates as anti-hacking.
