---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/gsm-symbolic.md, wiki/raw-data/llm-training/papers/gsm1k.md
source_url: https://arxiv.org/abs/2410.05229 ; https://arxiv.org/abs/2405.00332
primary_text_checked: arXiv:2410.05229v2 and arXiv:2405.00332, 2026-09-17
---

# Excerpt: two ways to build a perturbation set

Used by ch-53 §4. GSM-Symbolic keeps the reasoning and changes the surface; GSM1k keeps the
specification and writes new items. The two measure different parts of benchmark-specific inflation.

## A. GSM-Symbolic (Mirzadeh et al., Apple; arXiv v1 2024-10, ICLR 2025)

**Construction (§3.1).** A GSM8K question is rewritten as a template with typed variables and
sampling domains, for example:

```
When {name} watches her {family}, she gets out a variety of toys for him. The bag of building
blocks has {x} blocks in it. ... bringing her total number of toys ... up to {total}.
How many bouncy balls came in the tube?

#variables:
- name   = sample(names)
- family = sample(["nephew", "cousin", "brother"])
- x      = range(5, 100)
```

Conditions attached to the template keep the answer a positive integer.

**Protocol (§3.2).** 100 templates × 50 instantiations = 5,000 items, organized as 50 datasets of 100
items each, so that every dataset is one complete instantiation of the same 100 questions. Scoring is
8-shot chain of thought with greedy decoding. The paper reports about 500 evaluations over more than
20 open models from 2B to 27B plus GPT-4o-mini, GPT-4o, o1-mini, and o1-preview.

**Results used by ch-53.**
- Across the 50 sets, worst-to-best gap exceeds 12% for Gemma2-9B and is around 15% for
  Phi-3.5-mini (§4.1).
- Accuracy on the 100 original GSM8K items is frequently more than one standard deviation above the
  centre of the model's GSM-Symbolic distribution; this holds for 21 of 25 models (§4.1). The authors
  name data contamination as one possible explanation (Interpretation).
- Changing only proper names produces a distribution closer to the original score; changing numbers
  moves the mean down and the variance up; changing both is the largest shift (§4.2, Fig. 4).
- Adding or removing clauses moves the mean down and the variance up as difficulty rises (§4.3).
- GSM-NoOp adds one clause that looks relevant but changes no operation. Drops (Fig. 8a): Phi-3-mini
  -128k-instruct −65.7, Phi-3-small-128k-instruct −64.0, Gemma2-9b and Gemma2-9b-it −63.0,
  Phi-3.5-mini-instruct −62.5, Phi-3-medium-128k-instruct −57.8. Supplying eight shots of the same
  question from GSM-Symbolic does not recover performance beyond one standard deviation (§4.4).

**Limits.** Grade-school arithmetic only. The pattern-matching explanation is the authors'
interpretation of the accuracy drops, not a separate measurement.

## B. GSM1k (Zhang et al., Scale AI; arXiv v1 2024-05)

**Construction (§3.1).** 1,205 new grade-school problems written by human annotators who were shown
three GSM8K examples and asked for problems of similar difficulty. Solutions are positive integers and
require only the four basic operations. No language model was used at any point of the construction.

**Difficulty matching (§3.2).**
- Human distinguishability: 19 annotators picked the single GSM1k item out of five (four GSM8K) in
  21.83% of 1,205 attempts, against 20% for chance.
- Human solve rate under 15 minutes of time pressure, 14 annotators: 4.07 ± 0.93 problems on GSM8K and
  4.36 ± 1.11 on GSM1k, so GSM1k is at least as easy.
- Models released before GSM8K (GPT-2, GPT-NeoX-20B) show minimal difference between the two sets.

**Results used by ch-53.**
- The worst models score 8% lower on GSM1k than on GSM8K; frontier models show minimal difference
  (§1, §5.2).
- Memorization probe (§5.4): the per-character log-likelihood of generating the GSM8K test set,
  `(1/c) Σ_i log p(x_i | x_<i)` with c the character count, correlates with the GSM8K−GSM1k gap at
  Spearman 0.36 (p = 0.03); Pearson r² = 0.26 and Kendall τ = 0.29 are also reported. Every percentage
  point of gap is associated with an increase of 1.2 × 10⁻² in the per-character log-likelihood.
- The authors state that contamination is not the full explanation: outlier models have high
  log-likelihood on the GSM8K test set with minimal overfit (§5.4).
- Even the most overfit models still solve new grade-school problems (§5.3).

**Limits.** GSM1k is not public (the paper states release conditions), so ch-53 builds a
GSM-Symbolic-style set instead. Evaluation used a fork of EleutherAI's LM Evaluation Harness with the
generation limit raised from 256 to 1,000 tokens (§4).

Related: [[gsm-symbolic]], [[gsm1k]], [[swe-bench-illusion]], [[read]].
