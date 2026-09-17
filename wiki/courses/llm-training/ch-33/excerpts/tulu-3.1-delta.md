---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.1.md
source_url: https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the model card and open-instruct docs/tulu3.md@098424c)"
---

# Excerpt: Tülu 3 → Tülu 3.1 (8B) — what the model card says changed

**Primary sources:** Llama-3.1-Tulu-3.1-8B model card (Hugging Face README: "Version 3.1 update", "Performance", "Hyperparamters", "Learning curves", "Reproduction command"); arXiv:2411.15124v5 Table 21; open-instruct `docs/tulu3.md` at commit 098424c.
**Library cards:** [[tulu-3.1]], [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]]. The previous version of this excerpt called Tülu 3.1 a "controlled public ablation" of PPO versus GRPO with unpublished hyperparameters. The card publishes the hyperparameters, and several of them differ from the Tülu 3 run, so the comparison is not controlled.

## The card's statement

"The new version of our Tülu model is from an improvement only in the final RL stage of training. We switched from PPO to GRPO (no reward model) and did further hyperparameter tuning to achieve substantial performance improvements across the board over the original Tülu 3 8B model."

- Fine-tuned from `allenai/Llama-3.1-Tulu-3-8B-DPO`; RL dataset `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` (the same dataset name as the Tülu 3 PPO command, t3 L313–322).
- Reward model row: "None with GRPO".

## Settings that changed between the two final RL runs

| Setting | Tülu 3 8B (PPO) | Tülu 3.1 8B (GRPO) |
|---|---|---|
| Algorithm; value model | PPO; initialized from an RM | GRPO; none |
| Learning rate; schedule | 3 × 10⁻⁷; linear | 5 × 10⁻⁷; constant |
| KL β | 0.05 | 0.01 (estimator `kl3` in the command) |
| Samples per prompt; unique prompts per iteration | not stated as such; effective batch 224 | 16; 48 (effective batch 768) |
| Mini-batches; update iterations K | 1; 4 | 2; 1 |
| No-EOS penalty | −10.0 | 0.0 |
| Episodes | Table 21: 100,000; released checkpoint "earlier than final" | planned 10,000,000; released at step 1920 = episode 1,474,560 |
| Unchanged | response length 2,048; temperature 1.0; clip ε 0.2; γ 1.0 | same |

Sources: arXiv:2411.15124v5 Table 21 and §6.2–6.4; model card "Hyperparamters"; open-instruct `docs/tulu3.md` L313–346 and L468–547.

## Reported results (model card "Performance" table, 8B)

| Benchmark | Tülu 3 SFT | Tülu 3 DPO | Tülu 3 | Tülu 3.1 |
|---|---:|---:|---:|---:|
| Avg. | 60.4 | 64.4 | 64.8 | 66.3 |
| MMLU (0-shot CoT) | 65.9 | 68.7 | 68.2 | 69.5 |
| TruthfulQA | 46.8 | 56.1 | 55.0 | 59.9 |
| BBH | 67.9 | 65.8 | 66.0 | 68.9 |
| MATH | 31.5 | 42.0 | 43.7 | 47.8 |
| GSM8K | 76.2 | 84.3 | 87.6 | 90.0 |
| HumanEval (pass@10) | 86.2 | 83.9 | 83.9 | 84.8 |
| IFEval | 72.8 | 81.1 | 82.4 | 83.9 |
| AlpacaEval 2 | 12.4 | 33.5 | 34.5 | 34.9 |
| Safety (6-task avg.) | 93.1 | 87.2 | 85.5 | 81.2 |

The DPO and Tülu 3 columns match arXiv:2411.15124v5 Table 23. The card notes that the paper was later updated with fixed evaluations for some other models.

## What can and cannot be concluded

- Result (single run): the Tülu 3.1 checkpoint scores 1.5 points higher on the average and 4.3 points lower on safety than Tülu 3 8B.
- Not supported: attributing either change to GRPO. Algorithm, value model, β, learning rate and schedule, samples per prompt, EOS penalty, and training length all differ. An attribution would need runs that change one of these at a time from the same DPO checkpoint.
- The library card [[tulu-3-1]] describes "Tülu 3.1" as a Nov 2024 refresh of the recipe on Llama 3.1 and OLMo 2 bases. The model card and the Tülu 3 technical blog do not describe such a release.

## Used in

ch-33 §4.4, Common mistakes table, Recipe.
