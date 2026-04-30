---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/judge-llm-bias.md
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
---

# Excerpt: Zheng 2023 — MT-Bench, Chatbot Arena, and the canonical bias inventory

**Source library:** `wiki/raw-data/llm-training/papers/judge-llm-bias.md`
**Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica
**Year:** 2023 (NeurIPS Datasets & Benchmarks)

---

## Why this source is the anchor of ch-49

Every number in §3 of the `read.md` bias table cites this paper. The reason it functions as canon is not just priority — Zheng 2023 is one of the few works that ran a human-expert pass large enough (3K votes) to *calibrate* the bias numbers, not just surface them. Later papers (Con-J, J1, Self-Taught Evaluators) assume Zheng 2023's numbers when framing their contributions. The chapter cannot be read without these numbers in hand.

---

## The parity claim, verbatim

Source §Key Contributions:

> "Agreement numbers: GPT-4 vs human-expert agreement is 85%+ on MT-Bench and ~80% on Chatbot Arena; the same rate two humans agree with each other."

This is the origin of "GPT-4 is good enough as a judge." Every subsequent claim in the field qualifies this line. Ch-49 §3 takes the parity seriously *on aggregate* and then shows it is misleading on every axis needed for go/no-go.

---

## The bias quartet

Source §Key Contributions:

> "Position bias: A vs B ordering changes the winner in 20-30% of cases; mitigated by swap-and-average or 'two-game' scoring."
> "Verbosity bias: longer responses win more often than a length-controlled baseline."
> "Self-enhancement bias: GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer; Claude shows the same toward Claude."
> "Limited reasoning in pair judging: on math/coding pairs, LLM judges can confirm a wrong answer if it is presented confidently."

Ch-49 §3 puts each of these into a bias-x-method-x-correction row. The quartet drives the chapter's framing.

---

## The numeric handles

Source §Key Figures/Tables:

> "Fig. 2 (position bias sweep) — swap A/B, see how often the judge flips; GPT-4 flips ~22%, GPT-3.5 ~40%."
> "Fig. 4 (verbosity vs win rate) — clear upward slope."
> "Fig. 5 (self-enhancement heatmap) — judge x candidate self-preference matrix."

These three figures are the numbers panel 1 of `figures/judge-bias.html` seeds. The interactive's GPT-4/GPT-3.5/Claude selector is directly the Fig. 2 swap-rate axis; the verbosity scenario is Fig. 4's slope; the self-enhancement scenario reads off the Fig. 5 diagonal.

---

## The mitigations — and their limits

Source §Technical Details:

> "Position-bias mitigation: evaluate both orders, take a win only if the judge is consistent; otherwise declare tie."
> "Verbosity-bias mitigation: length-controlled evaluation pairs where responses differ only in length; compute length-residualized win rate."
> "Self-enhancement mitigation: never use the candidate as its own judge; for preference-label generation, use a stronger independent model; for RM training data, pool multiple judges."
> "Chain-of-thought judging: asking the judge to reason before giving a verdict improves agreement but does not eliminate the biases."
> "Reference-guided grading: attach a gold reference solution to the prompt; raises agreement on objective tasks (math, coding), less effect on writing tasks."

Note the last two sentences. CoT helps but does not fix. Reference-guided helps on math/coding and *not* on writing. These limits are why ch-49 does not treat calibration as a single knob — it has to be per-category, per-rubric, per-family.

---

## What the paper releases

Source §Key Contributions:

> "Dataset release: 3K expert MT-Bench votes and 30K Chatbot Arena conversations, widely reused by later preference-dataset work."

The 3K expert MT-Bench votes are the historical anchor set. Every subsequent synthetic-judge paper (Con-J, STE, J1) evaluates against this set or a descendant. When ch-49 §8 says "maintain a human-label anchor set of 500-2000 pairs," this is the archetype.

---

## Connections into ch-49

- §3 bias table: direct quotes of all four biases + mitigations.
- `figures/judge-bias.html` Panel 1 severity axis: seeded from Fig. 2 (22% vs 40%), Fig. 4 slope, Fig. 5 diagonal.
- §5 "why GPT-4-as-judge is being replaced": self-enhancement bias at ecosystem level.
- §8 anchor set: the 3K MT-Bench expert votes are the prototype.
- §4 calibration: CoT-before-verdict helps but does not eliminate — direct quote.
