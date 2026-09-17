---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md and llama-3-recipe.md (sections on annealing data, contamination analysis, post-training decontamination)
source_url: https://arxiv.org/abs/2407.21783
primary_version: arXiv:2407.21783v3 (2024-11-23; v1 2024-07)
created_at: "2026-09-15"
revised: 2026-09 (generality revision; replaces the 2026-04 excerpt, whose n-gram filtering procedure, thresholds, and dropped-token share do not appear in the report)
---

# Excerpt: Contamination handling in The Llama 3 Herd of Models

Verbatim quotations used by ch-14 `read.md`, with loci. Author: Llama Team, AI @ Meta. Checked against the v3 PDF on 2026-09-15. Source type: official technical report.

What the report does and does not describe:
- Pre-training data: a post-hoc 8-gram **contamination analysis** that estimates score effects (§5.1.4). The report does not describe removing documents from the pre-training corpus by n-gram overlap, and it prints no share of removed tokens.
- Post-training data: **exact-match decontamination** against benchmark prompts (§5.2).
- Annealing data: benchmark training sets excluded (§3.1.3).
- Organization: a separate pre-training data team (§10).

## Annealing data (§3.1.3)
> "We do not include any training sets from commonly used benchmarks in our annealing data. This enables us to assess the true few-shot learning capabilities and out-of-domain generalization of Llama 3."
> "Following OpenAI (2023a), we evaluate the efficacy of annealing on the GSM8k (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) training sets in annealing."
> "We find that annealing improved the performance of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively. However, the improvements on the 405B model are negligible"

## Contamination analysis (§5.1.4)
> "Any of these methods can suffer from false positives and negatives, and how to best run contamination analyses is currently still an open field of research. Here, we largely follow the suggestions of Singh et al. (2024)."
> "For all our evaluation datasets, we score examples based on 8-gram overlap ... We consider an example of a dataset D to be contaminated if a ratio T_D of its tokens are part of an 8-gram occurring at least once in the pre-training corpus. We select T_D separately for each dataset, based on which value shows the maximal significant estimated performance gain across the three model sizes."

Selected rows of Table 15 (percent of the evaluation set considered contaminated; estimated performance gain for 8B / 70B / 405B):

| Benchmark | Contam. % | 8B | 70B | 405B |
|---|---|---|---|---|
| AGIEval | 98 | 8.5 | 19.9 | 16.3 |
| BIG-Bench Hard | 95 | 26.0 | 36.0 | 41.0 |
| GSM8K | 41 | 0.0 | 0.1 | 1.3 |
| HellaSwag | 85 | 14.8 | 14.8 | 14.3 |
| MATH | 1 | 0.0 | -0.1 | -0.2 |
| NaturalQuestions | 52 | 1.6 | 0.9 | 0.8 |
| PiQA | 55 | 8.5 | 7.9 | 8.1 |
| QuaC | 99 | 2.4 | 11.0 | 6.4 |
| SQuAD | 0 | 0.0 | 0.0 | 0.0 |
| MMLU, MMLU-Pro, HumanEval, MBPP | – | – | – | – |

> "For Natural Questions, on the other hand, the estimated 52% contamination seems to have virtually no effect on the performance. For SQuAD and MATH, low thresholds yield high levels of contamination, but no performance gains."
> "for MBPP, HumanEval, MMLU and MMLU-Pro, other contamination detection methods may be needed: even with higher thresholds, 8-gram overlap gives such high contamination scores that it is impossible to get a good performance gain estimate."

## Post-training data (§5.2)
> "We apply decontamination of the post-training data by running exact match with the prompts from each benchmark."

## Organization of data and evaluation (§5.3, §10)
> "Modeling teams did not have access to our human-evaluation prompts to prevent accidental contamination or overfitting on the test set." (§5.3)
> "to ensure Llama 3 is not accidentally overfitted on commonly used benchmarks, our pre-training data was procured and processed by a separate team that was strongly incentivized to prevent contamination of that pre-training data with external benchmarks." (§10)
> "we ensure that our human evaluations remain trustworthy by allowing only a small set of researchers who do not contribute to model development to perform and access these evaluations." (§10)
