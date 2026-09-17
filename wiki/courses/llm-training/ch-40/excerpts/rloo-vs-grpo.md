---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: library card [[rloo-vs-grpo]] (synthesized comparative reference; no primary artifact, no Verification section)
source_url: (none — the card lists its component papers [[rloo]], [[grpo]], [[dr-grpo]], [[reinforce-plus-plus]])
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; scope reduced to the axis list, with two of the card's claims corrected)"
---

# Excerpt: the comparison axes, and two claims this chapter does not carry over

Used by [[read]] §2, §4, and the Sources list. This card has no primary artifact behind it, so it is used only for the list of axes on which the family differs. Every number in the chapter comes from a paper, not from here.

## Axes the card is used for
| Axis | RLOO | GRPO |
|---|---|---|
| Samples per prompt | k ∈ {2, 4} | G ∈ {8 … 64} |
| Baseline | mean of the other k−1 rewards | group mean |
| Spread normalization | none | divide by group std |
| Importance-ratio clip | not used in the paper's runs | yes |
| KL placement | shaped per-token reward | inside the loss |
| Value network | none | none |

## Claim 1 — "equivalence in the limit"
The card says RLOO's baseline is "off by one sample" from the group mean and that the two coincide as `k` grows. The relation is exact at every `k` and is a constant factor, not a vanishing correction:
```
R_i − (1/(k−1)) Σ_{j≠i} R_j = (k/(k−1)) (R_i − R̄)
```
[[dr-grpo]] App. A states the same identity with `G/(G−1)`. At `G = 8` the factor is 1.14, at `G = 64` it is 1.016.

## Claim 2 — "the clip rarely binds"
The card generalizes [[rloo]] §3.2 ("clipped on average < 5% of the time per batch") to the family. That measurement belongs to a setting with two gradient steps per rollout batch and short generations. DeepSeek-R1-Zero takes 16 updates per rollout batch ([[deepseek-r1-recipe]] v2 §2.1), DAPO 16 (§4.1), GSPO 4 (§5.1); in those settings the clip bound is active, which is why [[dapo]] measures an accuracy change from decoupling the bounds and [[gspo]] measures clipped-token fractions of 0.15 against 0.0013.

## Claim 3 — "std normalization helps with continuous rewards"
Stated without evidence in the card. The chapter keeps the conditional form and gives the mechanism in §5: group-level division makes advantages comparable across prompts, and with near-tie reward-model scores it converts reward noise of a few hundredths into full-magnitude advantages. [[reinforce-plus-plus]] recommends batch-level normalization instead; [[gigpo-verl-agent]] §5.2 finds the choice task-dependent.
