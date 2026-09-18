---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "2 OLMo 2 Furious (arXiv:2501.00656v3): §2.3, §2.5, §4.1, §4.5, §5, Tables 3, 9, 13, 16, 17"
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-09-17"
note: "The library card model-reports/olmo-2.md has no Verification section and states a 32K context extension, ~460K H100 GPU-hours for the 7B, and a 50B cooldown for every size; none of these is in the report. ch-58a uses the values below, whose loci were read in arXiv:2501.00656v3 on 2026-09-15 for the ch-14a and ch-34 excerpts of the same passages."
---

# Excerpt: OLMo 2 — the full stage chain from pre-training to RLVR

Source type: official technical report (Allen Institute for AI), arXiv v3. Released checkpoints: OLMo-2-1124-7B,
OLMo-2-1124-13B, OLMo-2-0325-32B and their SFT, DPO and Instruct variants.

## Stage chain
pre-training (stage 1) → mid-training anneal with souping (stage 2) → SFT → DPO → RLVR. No long-context stage is
described; Table 3 gives sequence length 4,096 for all three sizes.

## Stage 1 and stage 2 (§2.3, Tables 3, 13; §4.1, §4.5)
- FLOP split: stage 1 uses "⩾ 90% training FLOPs" (also stated as "90–95%"); stage 2 uses "5–10% of training FLOPs",
  during which "we linearly decay the learning rate to zero".
- Tokens: "OLMo 2 7B is trained on 4.05 trillion tokens (3.90 trillion for pretraining stage), OLMo 2 13B is trained on
  5.6 trillion tokens (5 trillion for pretraining stage), and OLMo 2 32B is trained on 6.6 trillion tokens (6.06 trillion
  for pretraining stage)" (§2.3).
- Table 3: batch × sequence 1024 × 4096 (7B), 2048 × 4096 (13B, 32B); peak LR 3.0e-4 / 9.0e-4 / 6.0e-4; warmup 2000 steps;
  cosine horizon 5T / 5T / 6.5T tokens; truncation after 4T (7B) and 6T (32B).
- Souping: the 7B averages three 50B-token anneals with different data orders; the 13B and 32B average three 100B runs and
  one 300B run, and "The final model is the average of all four models" (§2.3).
- Table 13 mix shares (50B / 100B / 300B): Filtered DCLM 47.2 / 50.2 / 51.9; Decontaminated FLAN 16.6 / 16.7 / 11.3;
  peS2o 5.85 / 9.52 / 19.4; Dolmino Math 20.8 / 17.5 / 10.8. §4.5: the 7B uses the 50B mix and the 13B the 100B mix,
  "ensuring the same number of steps during learning rate anneal".

## Post-training (§5)
- SFT data: `tulu-3-sft-olmo-2-mixture`, 939,104 prompts (7B, 13B); `tulu-3-sft-olmo-2-mixture-0225`, 866,138 prompts
  (1B, 32B).
- SFT LR sweeps: 1e-5, 2e-5 (selected), 3e-5 for 7B; 1e-6 … 8e-6 with 5e-6 selected for 13B; 1e-6 … 5e-6 with 4e-6
  selected for 32B. "OLMo 2 required significantly higher learning rates compared to the Llama 3.1 training recipe."
- Table 17 (7B SFT, same data, effective batch 128, linear schedule, warmup ratio 0.3): 2 epochs at 1e-5 with sum loss
  → 49.97; 3 epochs at 4e-6 sum → 49.76; 3 epochs at 4e-6 mean → 48.25; 2 epochs at 2e-6 mean → 48.18.
- Preference data: on-policy generations from development SFT models; 366.7k prompts (7B) and 377.7k (13B); responses from
  20 models; GPT-4o-2024-08-06 rates helpfulness, truthfulness, honesty and instruction following; the highest-rated
  completion is chosen and a random lower-rated one is rejected.
- DPO LR sweep 5e-7 … 1e-6 (1e-6 selected for 7B, 8e-7 for 13B); 32B best 2e-6.
- RLVR: PPO for 7B and 13B with the value function initialized from a reward model; data is GSM8K and MATH training sets
  plus constraint prompts. The 13B ran two further RLVR passes (GSM8K, then MATH) after those scores fell below a
  development model (Fig. 13). Sweeps: β 0.03, 0.05, 0.07 (7B), 0.1 (13B); LR 4e-7 (7B), 3e-7 (13B). The 1B and 32B used
  GRPO; 32B LR 5e-7, β 0.1.
- "At each stage we experiment with 1 random seed initially to arrive on a configuration and up to 4 with final
  hyperparameters."

## Stage deltas (Table 16, AVG as printed)
| Model | AVG | AE2 | GSM8K | IFEval | MATH | MMLU | Safety |
|---|---|---|---|---|---|---|---|
| 7B SFT | 51.4 | 10.2 | 74.6 | 66.9 | 25.3 | 61.1 | 94.6 |
| 7B DPO | 55.9 | 27.9 | 82.6 | 73.0 | 30.3 | 60.8 | 93.7 |
| 7B Instruct (RLVR) | 56.5 | 29.1 | 85.1 | 72.3 | 32.5 | 61.3 | 93.3 |
| 32B SFT | 61.7 | 16.9 | 78.4 | 72.4 | 35.9 | 76.1 | 93.8 |
| 32B DPO | 68.8 | 44.1 | 85.7 | 83.8 | 46.8 | 78.0 | 91.9 |
| 32B Instruct (RLVR) | 68.8 | 42.8 | 87.6 | 85.6 | 49.7 | 77.3 | 85.9 |

## Evaluation regime (§2.5, footnote 6, Table 6 caption, Table 9)
- "we maintained a held-out suite of tasks which were not used for model development decisions" (§2.5); GSM8K "was only
  partially held-out" because 200 of 1319 examples were used as a development set.
- Table 6 caption: "OLMo 2 models were not evaluated on held-out datasets prior to release."
- Table 9 (held-out columns), 7B pre-training → pre-training and mid-training: Avg 53.0 → 62.9; GSM8K 24.1 → 67.5;
  MMLU PRO 27.4 → 31.0; AGIEval 44.6 → 50.4; TQA 74.6 → 78.0.

## Verification
- Loci read in arXiv:2501.00656v3 on 2026-09-15 (ch-14a and ch-34 excerpts of the same passages); reassembled for ch-58a
  on 2026-09-17 without changing any value or locus.
- Not reported: pre-training compute in GPU hours for any size; parameter counts of 7B and 13B in the body; seeds or
  variance for Table 8; any long-context stage.
