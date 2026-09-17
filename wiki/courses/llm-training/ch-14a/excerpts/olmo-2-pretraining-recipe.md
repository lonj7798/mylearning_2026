---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "2 OLMo 2 Furious (arXiv:2501.00656v3), §2.3–§2.5, Tables 3–6, 8–9, 13, §4.1, §4.5, App. A.1"
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-09-15"
note: "The library card model-reports/olmo-2.md has no Verification section as of 2026-09-15. It states context 'extended to 32K in cooldown', 7B pre-training compute '~460K H100 GPU-hours', and a ~50B cooldown for all sizes; none of these is supported by the passages below (Table 3 sequence length 4096; 13B and 32B use 100B and 300B anneals). Values in ch-14a come from this excerpt."
---

# Excerpt: OLMo 2 — pre-training budget, schedule, and stage-2 mixture

Source type: official technical report (Allen Institute for AI). PDF text of arXiv v3 read on 2026-09-15.

## Two stages (§2.3)
- Stage 1 pre-training uses "⩾ 90% training FLOPs" (also "90–95%"); stage 2 mid-training uses "5–10% of training FLOPs", where "we linearly decay the learning rate to zero".
- Schedule: warmup "from 0 to the peak learning rate over 2000 steps, followed by a cosine decay calibrated to reach 10% of the peak learning rate after a specified max tokens"; initialization truncated normal, std 0.02.
- Souping: "For OLMo 2 7B, we anneal three separate times for 50B tokens each, with different randomized data orders; we average the resulting models". 13B and 32B: three 100B runs "(same number of update steps as the 7B)" and one 300B run; "The final model is the average of all four models."
- "In total, OLMo 2 7B is trained on 4.05 trillion tokens (3.90 trillion for pretraining stage), OLMo 2 13B is trained on 5.6 trillion tokens (5 trillion for pretraining stage), and OLMo 2 32B is trained on 6.6 trillion tokens (6.06 trillion for pretraining stage)."

## Table 3 (hyperparameters)
| | 7B | 13B | 32B |
|---|---|---|---|
| Layers / d_model | 32 / 4096 | 40 / 5120 | 64 / 5120 |
| Attention (Q/KV) | 32/32 MHA | 40/40 MHA | 40/8 GQA |
| Batch size × sequence length | 1024 × 4096 | 2048 × 4096 | 2048 × 4096 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Peak LR | 3.0 × 10^−4 | 9.0 × 10^−4 | 6.0 × 10^−4 |
| LR warmup | 2000 steps | 2000 steps | 2000 steps |
| LR schedule (cosine) | 5T tokens | 5T tokens | 6.5T tokens |
| Truncation | (after 4T) | n/a | after 6T |

## Learning-rate experiments (§4.1, Figure 11, Table 8; 7B setting)
- "for the 7B variant, we stop the schedule at 4T tokens", based on "Previous experience with OLMo-0424" that the last part of a cosine "can be cut off and replaced by a linear decay to zero with little loss of performance". "The 13B ran with a higher peak learning rate from the start, so we decided to run it to 5T tokens before moving to the mid-training stage."
- Peak LRs 6, 9, 12, and 30 × 10^−4 were run besides 3 × 10^−4; 30 × 10^−4 had loss spikes during warmup and was abandoned; "higher learning rates universally perform better early on ... but eventually the lower learning rate setting overtakes the others"; for 3 versus 6 × 10^−4 "the cross-over point is well past 200B tokens".
- Checkpoints at 300B tokens decayed to zero: "a higher learning rate does make mid-training more effective, but it does so by exactly the amount that the pretraining is worse" (Figure 11 caption: equivalent training loss after 50B or 100B decays).
- Table 8, OLMES 9 MC tasks (validation, cloze): 300B + 50B decay: 62.5 (3e-4), 63.9 (6e-4), 64.1 (9e-4), 63.6 (12e-4); 300B + 100B decay: 64.6 (6e-4), 64.5 (9e-4), 64.2 (12e-4); 2T + 100B high-quality decay: 73.8 (3e-4), 73.9 (6e-4). Caption: "Average scores vary by less than two points across all variants". On GSM8K the 6e-4 setting is "2.8 points better"; "More study is needed".

## Stage-1 data (Table 4)
OLMo 2 Mix 1124, 3.90T tokens: DCLM-Baseline 3.71T; StarCoder (filtered) 83.0B; peS2o 58.6B; arXiv 20.8B; OpenWebMath 12.2B; Algebraic Stack 11.8B; Wikipedia & Wikibooks 3.7B. "over 95% derived from web data" (§2.4.1).

## Stage-2 data (Table 5, Table 13, §4.5)
- Dolmino high-quality subset 832.6B tokens; Dolmino Math Mix 10.7B tokens, which includes "GSM8K Train split" (2.74M tokens) (Table 5).
- Table 13, Mix % of the 50B / 100B / 300B mixes: Filtered DCLM 47.2 / 50.2 / 51.9; Decontam. FLAN 16.6 / 16.7 / 11.3; StackExchange Q&A 2.45 / 2.47 / 1.68; peS2o 5.85 / 9.52 / 19.4; Wikipedia/Wikibooks 7.11 / 3.57 / 4.86; Dolmino Math 20.8 / 17.5 / 10.8.
- §4.5: math and Stack Exchange data are repeated twice for 100B and four times for 300B because "keeping mixing proportion roughly constant across sources is beneficial"; the 7B uses the 50B mix and the 13B the 100B mix, "ensuring the same number of steps during learning rate anneal".

## Evaluation split (§2.5, footnote 6, Table 6 caption, Table 9)
- "we maintained a held-out suite of tasks which were not used for model development decisions" (§2.5). GSM8K "was only partially held-out": 200 of 1319 examples were used as a development set, GSM* (footnote 6).
- Table 6 caption: "OLMo 2 models were not evaluated on held-out datasets prior to release"; Qwen 2.5 models are trained on a "maximum of 18 trillion tokens"; "developers have declined to disclose exact token counts for each model size".
- Table 9 (held-out columns AGIEval, GSM8K, MMLU PRO, TQA), 7B pre-training → pre-training & mid-training: Avg 53.0 → 62.9; AGIEval 44.6 → 50.4; GSM8K 24.1 → 67.5; MMLU PRO 27.4 → 31.0; TQA 74.6 → 78.0. 13B: AGIEval 48.2 → 54.2; GSM8K 37.3 → 75.1; MMLU PRO 31.2 → 35.1; TQA 80.3 → 81.9.
- Table 9 caption: "Pretrain checkpoints have been trained on 4 trillion (1B, 7B), 5 trillion (13B) and 7 trillion (32B) tokens" (compare §2.3: 6.06T for 32B).

## Verification
- Read on 2026-09-15 against arXiv:2501.00656v3 PDF text.
- Not reported: parameter counts of the 7B and 13B models in the report body; pre-training compute in GPU hours for 7B; seeds or variance for Table 8.
