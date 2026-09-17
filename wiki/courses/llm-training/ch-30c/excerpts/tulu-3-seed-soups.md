---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "Tülu 3: Pushing Frontiers in Open Language Model Post-Training, §4.3 and §4.3.1 (Random Seeds and Model Soups)"
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-09-15"
---

# Excerpt: Tülu 3 SFT seed soups

The library card [[tulu-3]] predates the 2026-09 verification pass and does not record this experiment.
This excerpt records the passage as printed in arXiv:2411.15124v5 (2025-04-14).

## Setting (§4.3)
- Base models Llama 3.1 8B and 70B; effective batch size 128; maximum sequence length 4,096; 2 epochs; LR 5e-6 (8B)
  and 2e-6 (70B), "found after a hyperparameter search" (§4.3, Table 11).
- "For merging experiments we used mergekit (Goddard et al., 2024), using linear weighted averaging" (§4.3).
- §4 intro: the team "explored model merging techniques to develop an SFT training procedure that well balances
  performance across the core skills". Only the seed-soup result is reported with numbers.

## Result (§4.3.1, Table 14; "Average performance" as printed)
| Model | Seed | Average |
|---|---|---|
| Tülu 3 8B SFT | 42 (default) / 123 / 456 / 789 / 1011 | 59.9 / 60.1 / 59.8 / 59.8 / 59.8 |
| Tülu 3 8B SFT | best soup (42 & 123) | 60.2 |
| Tülu 3 70B SFT | 42 (default) / 123 / 456 | 71.8 / 70.0 / 72.6 |
| Tülu 3 70B SFT | best soup (123 & 456) | 72.5 |

- Authors' reading: "SFT performance noticeably varies based on the seed, highlighting the importance of multiple
  training runs, and ... the best model soup does not always outperform the best single training run. Because of
  this, we use the best single SFT training run for each model size as our final SFT models."

## Derived
- Seed range: 0.3 points at 8B (59.8-60.1) and 2.6 points at 70B (70.0-72.6).
- Best soup minus best single seed: +0.1 at 8B, −0.1 at 70B.

## Not reported
- Soup weights (the report names "linear weighted averaging" but does not give the weights), which other soups were
  tried, and a per-benchmark breakdown of Table 14.

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2411.15124v5 (PDF), §4 intro, §4.3, §4.3.1, Table 14.
