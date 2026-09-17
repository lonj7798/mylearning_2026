---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "OLMo 2: 2 OLMo 2 Furious (arXiv:2501.00656v3)"
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the primary text; the earlier version had a 32K context extension, code verifiers, GPU-hour figures, and Tülu 3 RLVR values that the report does not contain)"
---

# Excerpt: OLMo 2 — Tülu 3 post-training on OLMo bases, with its sweeps and stage table

The library card [[olmo-2]] has not been verified. Every value below was read in arXiv:2501.00656v3.
Released launch values are in [[open-instruct-allenai-recipes-recipe]].

## Base model (§2.3, Table 3; §4.5, Tables 13-14)
- Sequence length 4,096 for 7B, 13B, and 32B (Table 3). No long-context stage is described.
- Peak LR 3.0e-4 / 9.0e-4 / 6.0e-4; batch 1024 / 2048 / 2048 instances; cosine over 5T / 5T / 6.5T tokens, truncated
  after 4T (7B) and 6T (32B) (Table 3).
- Mid-training soups: 7B averages three 50B-token anneals on different data orders; 13B and 32B average three 100B runs
  and one 300B run (§2.3, §4.5). Table 14 (six mixes, 7B, 50B tokens): the text says souping "consistently equals or
  outperform[s]" the best single checkpoint, but for mix E GSM* is 60.5 for the best single run and 43.0 for the soup.

## SFT (§5)
- `tulu-3-sft-olmo-2-mixture` (7B, 13B): 939,104 prompts. Removing the Aya split and multilingual WildChat samples lowered
  the average by about 0.5 points, so they were kept.
- `tulu-3-sft-olmo-2-mixture-0225` (1B, 32B): 866,138 prompts; removes synthetic instructions mentioning a date cutoff
  and keeps Persona MATH and Grade School Math items only where a majority vote over 5 completions is reached.
- Table 17 (7B SFT configurations, same data, effective batch 128, linear schedule; caption: "warmup up ratio of 0.3"):
  2 epochs, 1e-5, sum loss → 49.97; 3 epochs, 4e-6, sum → 49.76; 3 epochs, 4e-6, mean → 48.25; 2 epochs, 2e-6, mean →
  48.18 (other rows 49.74, 49.59 with 2 epochs, 1e-5, sum).
- Hyperparameter selection: "At each stage we experiment with 1 random seed initially to arrive on a configuration and
  up to 4 with final hyperparameters." SFT LR sweep 1e-5, 2e-5 (♥), 3e-5 for 7B; 1e-6, 4e-6, 5e-6 (♥), 7.5e-6, 8e-6 for
  13B; 32B swept 1e-6 to 5e-6, best 4e-6. "OLMo 2 required significantly higher learning rates compared to the Llama 3.1
  training recipe."

## DPO and RLVR (§5)
- On-policy preference data from development OLMo 2 SFT models; 366.7k prompts (7B) and 377.7k (13B); responses from 20
  models; GPT-4o-2024-08-06 rates helpfulness, truthfulness, honesty, instruction following; highest average rating is
  chosen, a random lower-rated completion is rejected.
- DPO LR sweep 5e-7, 6e-7, 7e-7, 8e-7 (♥ 13B), 1e-6 (♥ 7B); 32B best 2e-6.
- RLVR for 7B and 13B: PPO with value function initialized from a reward model; data GSM8K, MATH training sets, and
  prompts with constraints. 13B ran two more RLVR passes (GSM8K, then MATH) after GSM8K and MATH were lower than a
  development model (Fig. 13). Sweeps: β 0.03, 0.05, 0.07 (♥ 7B), 0.1 (♥ 13B); LR 3e-7 (♥ 13B), 4e-7 (♥ 7B).
- 1B and 32B RLVR: GRPO; 32B LR 5e-7, KL β 0.1.

## Stage table (Table 16; AVG as printed; Safety is the Tülu 3 safety average; the final Instruct model is the RLVR output)
| Model | AVG | AE2 | BBH | DROP | GSM8K | IFE | MATH | MMLU | Safety | PQA | TQA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7B SFT | 51.4 | 10.2 | 49.6 | 59.6 | 74.6 | 66.9 | 25.3 | 61.1 | 94.6 | 23.6 | 48.6 |
| 7B DPO | 55.9 | 27.9 | 51.1 | 60.2 | 82.6 | 73.0 | 30.3 | 60.8 | 93.7 | 23.5 | 56.0 |
| 7B Instruct | 56.5 | 29.1 | 51.4 | 60.5 | 85.1 | 72.3 | 32.5 | 61.3 | 93.3 | 23.2 | 56.5 |
| 13B SFT | 56.6 | 11.5 | 59.9 | 71.3 | 76.3 | 68.6 | 29.5 | 68.0 | 94.3 | 29.4 | 57.1 |
| 13B DPO | 62.0 | 38.3 | 61.4 | 71.5 | 82.3 | 80.2 | 35.2 | 67.9 | 90.3 | 29.0 | 63.9 |
| 13B Instruct | 63.4 | 39.5 | 63.0 | 71.5 | 87.4 | 82.6 | 39.2 | 68.5 | 89.7 | 28.8 | 64.3 |
| 32B SFT | 61.7 | 16.9 | 69.7 | 77.2 | 78.4 | 72.4 | 35.9 | 76.1 | 93.8 | 35.4 | 61.3 |
| 32B DPO | 68.8 | 44.1 | 70.2 | 77.5 | 85.7 | 83.8 | 46.8 | 78.0 | 91.9 | 36.4 | 73.5 |
| 32B Instruct | 68.8 | 42.8 | 70.6 | 78.0 | 87.6 | 85.6 | 49.7 | 77.3 | 85.9 | 37.5 | 73.2 |

AE2 = AlpacaEval 2 (LC win rate), IFE = IFEval, PQA = PopQA, TQA = TruthfulQA. Evaluation regime: Table 15.

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2501.00656v3: §2.3, §4.5, §5, Tables 3, 13-17.
