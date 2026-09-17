---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/catastrophic-forgetting-continual-finetuning.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2308.08747
created_at: "2026-09-15"
---

# Excerpt: An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning

**Authors:** Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang (Westlake University; WeChat AI, Tencent)
**Version read:** arXiv:2308.08747v5 (5 Jan 2025); v1 August 2023.
**Status:** the library card for this slug did not exist when ch-30a was written. Every quote below was checked against the v5 PDF text on 2026-09-15.

## Setup (§3.1, §4.1-4.2)
- Task sequence: "Simp → Emdg → InqQG → Exp → HGen" (five generation tasks from Scialom et al. 2022), "100,000 data samples are used for training" (§3.1).
- Models: BLOOMZ 1.1B, 1.7B, 3B, 7.1B; mT0 1.2B, 3.7B; LLaMA-7B; Alpaca-7B; BLOOM-7.1B (§4.1, Table 6).
- Training: "the batch size is 4 on each device, the learning rate is 2e-5, and the scheduler is set constant for BLOOMZ and mT0"; LLaMA and Alpaca use a cosine scheduler at 2e-5; "The max sequence length of the inputs is 512. We train our model 3 epochs" (§4.2). 8 A100 40G GPUs, 4 GPUs for 1B-scale models (§4.2).
- Evaluation: MMLU 5-shot (domain knowledge); BoolQ, PIQA, Winogrande, Hellaswag, MathQA, Mutual zero-shot (reasoning); RACE-high and RACE-middle (reading comprehension); CrowS-Pairs (bias), via lm-evaluation-harness (§3.2-3.3).

## Forgetting metric (§3.3, Eq. 1)
FG_i = (1/|E_i|) Σ_{e∈E_i} (1/N) Σ_{m=1..N} (R_o^e − R_m^e) / R_o^e × 100%,
where R_o^e is the initial model's result on element e and R_m^e the result after the m-th task. The average runs over all N intermediate checkpoints, not only the final one.

## Table 4 (main results; R_o → R_{-1}, FG)
| Model | Domain knowledge | Reasoning | Reading comprehension |
|---|---|---|---|
| BLOOMZ-1.1b | 27.19 → 23.84, FG 9.54 | 47.37 → 41.97, FG 6.73 | 36.77 → 27.28, FG 18.04 |
| BLOOMZ-7.1b | 33.08 → 25.61, FG 18.37 | 59.15 → 49.24, FG 13.62 | 48.79 → 33.05, FG 26.75 |
| mT0-3.7b | 30.99 → 20.14, FG 20.15 | 48.61 → 38.39, FG 16.73 | 41.10 → 30.45, FG 28.42 |

## Table 6 (effect of general instruction tuning)
| Model | Domain knowledge FG | Reasoning FG | Reading comprehension FG |
|---|---|---|---|
| LLAMA-7b | 34.57 | 31.33 | 31.72 |
| ALPACA-7b | 18.14 | 7.56 | 10.31 |

## Quotes used in ch-30a
- Scale (§5.2): "the FG values in domain knowledge are 9.54%, 10.72%, 14.63%, and 18.37% in BLOOMZ-1.1b, 1.7b, 3b, and 7.1b, respectively." The authors attribute this to initial performance: "the initial performance Rso is boosted by the increasing model scale, but the final performance is relatively similar across different scales."
- Replay of general data (§5.4): "we mix 10,000 general instruction data samples from ALPACA ... the performance of MMLU-human in the initial LLAMA-7b model is 34.72%, but it decreases to 26.8% when trained solely on the instruction data. However, when trained on the mixed data, the performance becomes 30%."
- Limitations (§7): experiments stop at 7B and use one task order.

## How ch-30a uses it
§2 (evidence of forgetting and the FG worked example), §5.1 (general-data replay), Recipe row.
