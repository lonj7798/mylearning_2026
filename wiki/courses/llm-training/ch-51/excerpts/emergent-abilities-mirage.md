---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: "primary source arXiv:2304.15004 (v2, 2023-05-22), NeurIPS 2023 (no library card for this slug on 2026-09-15)"
source_url: https://arxiv.org/abs/2304.15004
created_at: "2026-09-15"
note: "Read from the cached v2 text on 2026-09-15. Used in ch-51 only for the metric-choice and resolution argument."
---

# Excerpt: Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer, Miranda, Koyejo 2023)

**Authors:** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo (Stanford University). Source type: paper.

## Claim

"For a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due [to]
the researcher's choice of metric rather than due to fundamental changes in model behavior with scale.
Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous
metrics produce smooth, continuous, predictable changes in model performance" (abstract).

## The two mechanisms (§2, App. A.1)

1. **Metric shape.** A metric that is nonlinear in the per-token error rate (Accuracy on a multi-token target) or
   discontinuous (Multiple Choice Grade, Exact String Match) turns a smooth improvement in per-token error into an
   apparently sharp jump. Under a linear metric on the same fixed outputs (Token Edit Distance) or a continuous
   one (Brier Score) the same family improves smoothly (§3, Figs. 3 and 6).
2. **Resolution.** "having insufficient resolution to estimate model performance in the smaller parameter regime,
   with resolution set by 1/test dataset size" (§2). App. A.1 defines resolution as "the smallest interval
   measurable by a scientific instrument" and works the coin-flip analogue: with F flips, the resolution-limited
   probability is a multiple of 1/F.

## Evidence

- InstructGPT/GPT-3 family on 2-shot 2-digit × 2-digit multiplication and 2-shot 4-digit addition: with Accuracy
  the family shows an apparent emergent jump at 4 or 5 target digits; with Token Edit Distance on the *same fixed
  outputs*, performance improves smoothly and predictably with scale (§3, Fig. 3).
- Generating additional test data raises the resolution and shows that all models in the family are above chance
  even on Accuracy, improving smoothly (§3, Fig. 4).
- Meta-analysis of BIG-Bench: claimed emergent abilities concentrate on a few metrics; "> 92% of emergent
  abilities" appear under one of two metrics, Multiple Choice Grade (discontinuous) and Exact String Match
  (nonlinear) (§1, §4, Fig. 5C). Changing the LaMDA family's metric from Multiple Choice Grade to the continuous
  Brier Score removes the apparent emergence (§4, Fig. 6).
- The same recipe produces "never-before-seen seemingly emergent abilities" in vision models by choosing a
  discontinuous metric (§5).

## Limits

The paper analyzes fixed model outputs and metric choice. It does not claim that no capability changes with scale;
its claim is about the shape of the measured curve for the tasks and families analyzed.

## Used in

ch-51 §7 (metric choice, resolution limit of a finite eval set, and why a checkpoint curve can look like a step).
