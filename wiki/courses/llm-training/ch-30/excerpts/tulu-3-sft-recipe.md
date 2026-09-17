---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: arXiv:2411.15124v5 (Tülu 3 report)
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the paper)"
---

# Excerpt: Tülu 3 SFT settings and analyses as printed in the report

Used by [[read]] §3, §6, §9, and the Recipe table. The earlier version quoted the card [[allenai-tulu-sft-recipe]], whose NEFTune, packing, optimizer, FSDP, mixture-share, and ablation-delta rows do not appear in the report; none of them is repeated here. Card for the report: [[tulu-3]].

## Table 11 and §4.3 (SFT training settings)

| Hyperparameter | 8B | 70B |
|---|---|---|
| Learning Rate | 5 × 10⁻⁶ | 2 × 10⁻⁶ |
| Learning Rate Schedule | Linear | Linear |
| Batch Size (effective) | 128 | 128 |
| Max Token Length | 4,096 | 4,096 |
| Warm up ratio | 0.03 | 0.03 |
| Number of Epochs | 2 | 2 |

§4.3 (verbatim): "we used between 4 and 16 8xH100 nodes … The final 8B model is trained on 32 GPUs for 6 hours and the 70B model was trained on 64 GPUs for 50 hours. … We trained for two epochs using a learning rate of 5e-6 for our 8B models, and 2e-6 for our 70B models, which we found after a hyperparameter search." The SFT mix contains 939,344 prompts (Table 7).

## §4.3.2 Batch aggregation

"Averaging the loss across padding tokens without taking into account gradient accumulation or distributed training setups." For two samples with n₁, n₂ non-padding tokens, one forward pass gives L = (l_{n1} + l_{n2})/(n₁ + n₂) (Eq. 1); gradient accumulation gives L = (l_{n1}/n₁ + l_{n2}/n₂)/2 (Eq. 2). "That is, in the second case we weight each example equally, while in the first we weight each token equally." "To fix this issue, we opted generally to use a sum loss instead of averaging ('mean loss') when training. This removes the issue by simply removing the denominator from the above equations and requires an adjustment to learning rates." Fine-tuning Llama 3.0 on the Tülu 2 SFT mixture: "using a sum loss with a learning rate of 5.00E-06 worked best. Surprisingly, we additionally found that training for longer did not yield further improvements, and so used 2 epochs for training" (Figs. 5-6). The figures print curves without numeric labels.

## §4.3.1 Seeds and soups (Table 14)

| Model | Seed | Average |
|---|---|---|
| Tülu 3 8B SFT | 42 (default) / 123 / 456 / 789 / 1011 | 59.9 / 60.1 / 59.8 / 59.8 / 59.8 |
| Best 8B soup | 42 & 123 | 60.2 |
| Tülu 3 70B SFT | 42 (default) / 123 / 456 | 71.8 / 70.0 / 72.6 |
| Best 70B soup | 123 & 456 | 72.5 |

"We see that SFT performance noticeably varies based on the seed, highlighting the importance of multiple training runs, and that the best model soup does not always outperform the best single training run. Because of this, we use the best single SFT training run for each model size as our final SFT models."

## §2.2, Table 3 and §7.4.1 (development and unseen suites)

"Crucially, we did not examine scores on our unseen set when developing our models, allowing us to observe how much we may have overfit to particular evaluations in our decisions around data mixtures, algorithms, and hyperparameters." Development and unseen pairs include MMLU / MMLU-Pro, BigBenchHard / AGIEval English, MATH / Deepmind Mathematics, HumanEval / BigcodeBench, IFEval / IFEval-OOD. No unseen safety evaluation exists.

Table 32 (8B SFT, selected columns: Dev avg, Unseen avg, MATH, DMM, IFEval, IFEval-OOD):

| Model | Dev. Avg | Uns. Avg | MATH | DMM | IFE | IFEO |
|---|---|---|---|---|---|---|
| Tülu 3 8B SFT | 64.1 | 29.9 | 31.5 | 32.3 | 72.8 | 17.6 |
| w/o WildChat | 62.8 | 28.8 | 31.8 | 31.2 | 70.1 | 20.8 |
| w/o Safety | 63.7 | 29.7 | 32.6 | 32.6 | 71.0 | 17.6 |
| w/o Persona Data | 59.8 | 29.4 | 30.1 | 31.8 | 53.6 | 18.0 |
| w/o Math Data | 62.2 | 27.4 | 23.5 | 23.3 | 70.6 | 18.3 |

§7.4.1 (verbatim): "the data choices generalize on average … In individual skills, we see that our choices overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning."

## What the report does not state

NEFTune, packing, optimizer betas, weight decay, and the distributed-training strategy for SFT were not found in §4.3, Table 11, or App. B. The open-instruct reproduction commands and their loss-reduction drift are in [[open-instruct-allenai-recipes]].
