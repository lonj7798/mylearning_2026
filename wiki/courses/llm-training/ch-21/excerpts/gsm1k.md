---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: "A Careful Examination of Large Language Model Performance on Grade School Arithmetic (Zhang et al., Scale AI)"
source_url: https://arxiv.org/abs/2405.00332
created_at: "2026-09-15"
---

# Excerpt: GSM1k — a fresh counterpart to GSM8k

This excerpt stands in for the library card `gsm1k`, which did not exist when ch-21 was revised. Every value was read
in arXiv:2405.00332v4 (2024-11-22; NeurIPS 2024 Datasets and Benchmarks Track). arXiv v1 is dated 2024-05.

- **Authors:** Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, et al. (Scale AI)

## Construction (§1, §3)
- 1,205 problems written by human annotators; "No language models were used to construct this dataset" (§3).
- Annotators saw 3 GSM8k examples of estimated difficulty N and wrote problems with N resolution steps; the step
  distribution matched GSM8k; answer magnitudes were matched by discarding candidates (§3.2, Fig. 3).
- Quality: of 2,108 initial problems, 1,419 passed an independent second solve and 1,375 a quality audit (§3.1).
- Human checks: annotators picked the GSM1k item out of five 21.83% of the time (chance 20%) (§3.2.1); annotators
  solved 4.07 ± 0.93 GSM8k and 4.36 ± 1.11 GSM1k problems in 15 minutes (§3.2.2).
- Evaluation: LM Evaluation Harness 5-shot format for all models, so GSM8k scores can differ from model reports (App. F).

## Findings (Abstract, §5)
- Accuracy drops of up to 8% (Abstract).
- "Several families of models, including the Phi and Mistral families of models, show systematic tendencies to
  perform stronger on GSM8k compared to GSM1k for almost every release and scale of models" (§5.1).
- Frontier or close-to-frontier models, including Mistral Large, show minimal gaps (§5.2).
- "while Phi-2 has a 6% drop in accuracy between GSM8k and GSM1k, we find that it is still able to correctly solve over
  half of GSM1k problems" (§5.3).
- Spearman rank correlation 0.36 (p = 0.03) between per-character log-likelihood of the GSM8k test set and the
  GSM8k − GSM1k gap; Pearson r² = 0.26; Kendall τ = 0.29 (§5.4). The authors conclude that "data contamination is
  likely not the full story" and name collecting benchmark-like training data or selecting checkpoints on benchmarks
  as other possible causes (§5.4).

## Selected rows (App. F, standard prompt; Diff = GSM8k − GSM1k)
| Model | Diff | GSM8k | GSM1k | Z-score | p-value |
|---|---|---|---|---|---|
| phi-2 | 0.063 | 0.566 | 0.504 | 3.167 | 0.001 |
| phi-1.5 | 0.051 | 0.324 | 0.274 | 2.814 | 0.002 |
| Phi-3-medium-128k-instruct | 0.044 | 0.869 | 0.825 | 3.103 | 0.001 |
| Phi-3-mini-4k-instruct | 0.040 | 0.788 | 0.748 | 2.385 | 0.009 |
| Mistral-7B-v0.1 | 0.027 | 0.391 | 0.364 | 1.421 | 0.078 |
| Phi-3-mini-128k-instruct | 0.011 | 0.757 | 0.746 | 0.645 | 0.260 |
| mistral-large-latest | −0.001 | 0.853 | 0.854 | −0.049 | 0.519 |
| gpt-4 | −0.012 | 0.911 | 0.923 | −1.161 | 0.877 |

- Under the alternative prompt (App. E, F), Phi-3-mini-4k-instruct has Diff 0.007 (p 0.318) and Phi-3-mini-128k-instruct
  Diff 0.035 (p 0.014), so the verdict for these two checkpoints depends on the prompt format.
- App. F calls the test "two-tailed", but the printed p-values equal the upper-tail probability (for example
  z = −1.161 gives 0.877). This is a reading of the table, not a statement by the authors.

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2405.00332v4: Abstract, §1, §3, §5.1–5.4, App. F.
