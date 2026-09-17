---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "A Careful Examination of Large Language Model Performance on Grade School Arithmetic (Zhang et al., Scale AI)"
source_url: https://arxiv.org/abs/2405.00332
created_at: "2026-09-15"
---

# Excerpt: GSM1k — a fresh, human-written counterpart to GSM8k

This excerpt stands in for the library card `gsm1k`, which did not exist when ch-34 was written.
Every value below was read in arXiv:2405.00332v4 (2024-11-22; NeurIPS 2024 Datasets and Benchmarks Track).

- **Authors:** Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, et al. (Scale AI)
- **Source type:** paper (benchmark)

## Design (§1, §3)
- 1,205 grade-school math problems written by human annotators "without assistance from any LLM or other synthetic
  data source" (§1). Problems are matched to GSM8k on human solve rate, number of solution steps, and answer
  magnitude (Abstract, §3).
- The dataset is withheld from public release to avoid contamination, with a pre-committed later release (§1, §3).
- All models are evaluated with the LM Evaluation Harness 5-shot format, so GSM8k scores can differ from those in
  model reports (App. F).

## Findings (Abstract, §5)
- Accuracy drops of up to 8% from GSM8k to GSM1k (Abstract).
- "Several families of models, including the Phi and Mistral families of models, show systematic tendencies to
  perform stronger on GSM8k compared to GSM1k for almost every release and scale of models" (§5.1).
- Frontier or close-to-frontier models, including Mistral Large, show minimal gaps (§5.2).
- Overfit models still reason: Phi-2 drops 6% but solves over half of GSM1k (§5.3).
- Spearman rank correlation 0.36 (p = 0.03) between a model's per-character log-likelihood of the GSM8k test set and
  its GSM8k − GSM1k gap; Pearson r² = 0.26, Kendall τ = 0.29 (§5.4). The Abstract writes "Spearman's r² = 0.36".
  The authors conclude that partial memorization explains part of the gap but "data contamination is likely not the
  full story" (§5.4).

## Selected rows (App. F, standard prompt; Diff = GSM8k − GSM1k; Z and p from a two-proportion Z-test)
| Model | Diff | GSM8k | GSM1k | Z | p |
|---|---|---|---|---|---|
| phi-2 | 0.063 | 0.566 | 0.504 | 3.167 | 0.001 |
| Phi-3-medium-128k-instruct | 0.044 | 0.869 | 0.825 | 3.103 | 0.001 |
| Phi-3-mini-4k-instruct | 0.040 | 0.788 | 0.748 | 2.385 | 0.009 |
| Phi-3-mini-128k-instruct | 0.011 | 0.757 | 0.746 | 0.645 | 0.260 |
| Meta-Llama-3-70B-Instruct | 0.014 | 0.914 | 0.900 | 1.251 | 0.105 |
| gpt-4 | −0.012 | 0.911 | 0.923 | −1.161 | 0.877 |
| claude-2.1 | −0.004 | 0.887 | 0.891 | −0.336 | 0.632 |

- App. F calls the test "two-tailed", but the printed p-values equal the upper-tail probability P(Z > z)
  (for example z = −0.336 gives 0.632). This is a reading of the table, not a statement by the authors.
- Under the alternative prompt, Phi-3-mini-128k-instruct shows Diff 0.035, Z 2.211, p 0.014, and Phi-3-mini-4k-instruct
  shows Diff 0.007, GSM8k 0.807, GSM1k 0.800, Z 0.474, p 0.318 (App. F), so the verdict for both checkpoints depends on
  the prompt format, in opposite directions.

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2405.00332v4: Abstract, §1, §3, §5.1-5.4, App. F tables.
