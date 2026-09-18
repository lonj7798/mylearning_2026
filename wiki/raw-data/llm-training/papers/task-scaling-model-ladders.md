<!-- scope: two-step task scaling laws ((N, D) -> bits-per-byte task loss -> ranked-classification accuracy) fit on 16 small OLMo 2 ladder models to predict 7B-4T and 13B-5T accuracy on 8 OLMES tasks; design-choice comparison; checkpoint-variance analysis of task predictability
     deps: [[chinchilla-compute-optimal]]
     see-also: [[overtraining-downstream-scaling]], [[olmes]], [[olmo-2]]
-->

# Establishing Task Scaling Laws via Compute-Efficient Model Ladders
- **Core Insight:** A power law from model size N and tokens D to a bits-per-byte task loss, chained with a sigmoid from task loss to ranked-classification accuracy, fit on 16 ladder models (190M to 1.3B, 1x to 10x Chinchilla tokens; 5.2e21 FLOPs, under 1.0% of the two targets' combined compute) predicts OLMo 2 7B-4T and 13B-5T accuracy on 8 tasks with average absolute errors of 3.8 and 4.2 points, and with 0.3 to 2.1 points of error on MMLU, HellaSwag, PIQA, and Social IQa (§2.1, §3.3, Fig. 2).
- **Guideline:** When forecasting per-task accuracy of an over-trained model from small runs, first compute the standard deviation of task loss over the final 10 checkpoints of the largest ladder model and treat high-variance tasks as poorly predictable, because loss SD10 correlated with step-2 accuracy error (Pearson r = 0.821 for 7B-4T, 0.855 for 13B-5T) and the high-variance tasks ARC-Challenge, ARC-Easy, and OpenBookQA had 7.8% to 17.5% relative chained error (§5, Table 4, Fig. 2); with a fixed ladder budget, add larger model sizes rather than longer training (App. B.3).
- **Authors:** Akshita Bhagia, Jiacheng Liu, Alexander Wettig, David Heineman, Oyvind Tafjord, Ananya Harsh Jha, et al. (Allen Institute for AI; University of Washington; Princeton University)
- **Year:** 2024 (arXiv v1 2024-12; COLM 2025)
- **URL:** https://arxiv.org/abs/2412.04403
- **Source type:** paper
- **Relevant topics:** downstream task scaling laws, over-trained regime, model ladders, bits-per-byte task loss, ranked classification vs multiple choice, evaluation noise, OLMo 2

## Abstract
The paper develops task scaling laws and model ladders to predict the accuracy of pretrained LMs on individual tasks in the over-trained setting. Standard power laws for language modeling loss do not model task performance accurately, so prediction uses two steps: model and data size predict an intermediate loss, and that loss predicts task accuracy. A set of small "ladder" models supplies the data points for both fitted functions, and predictions are made for a 7B model trained to 4T tokens and a 13B model trained to 5T tokens. The ladder costs 1% of the targets' compute. On four multiple-choice tasks in ranked-classification format, the accuracy of both targets is predicted within 2 points of absolute error. Tasks with higher prediction error also have higher metric variance across checkpoints. The paper compares design choices and gives recommendations for new models and tasks.

## Key Contributions
- A fixed ladder of 4 sizes × 4 token multipliers = 16 models with hyperparameters extrapolated from the 7B target, instead of a search for compute-optimal models (§2.1, §8).
- Two-step prediction: a power law for task loss (Eq. 1) and a sigmoid for accuracy (Eq. 2), chained end to end (§3).
- Comparison of intermediate features (task loss, TaskCE, C4 loss), compute-FLOPs as input, and a single-step fit (Table 3, App. C).
- Checkpoint-variance analysis (SD over the final n checkpoints) as an indicator of which tasks can be forecast (§5, Table 4).
- Extensions to multiple-choice format using target-model checkpoints (App. B.2) and to a 32B-6T target (App. D).

## Key Figures/Tables to Study
- **Table 1:** ladder and target configurations (parameters, tokens, batch, peak LR, warmup, shape).
- **Fig. 2:** chained predictions and per-task errors for both targets.
- **Table 3:** step-1 and chained errors for each design choice.
- **Table 4:** SD10 of the 1B-10xC ladder model against 7B-4T prediction errors.
- **Fig. 8-10 (App. B.3):** 7B-4T prediction error vs ladder FLOPs, largest ladder size, and largest token multiplier.
- **Fig. 7 (App. B.2):** three-phase MC accuracy curves of the targets.

## Technical Details
- **Ladder (§2.1):** N ∈ {190M, 370M, 760M, 1.3B} non-embedding parameters; D ∈ {1xC, 2xC, 5xC, 10xC} with 1xC = 20·N; checkpoints every 200 steps (1xC, 2xC), 500 (5xC), and 1000 (10xC). Same architecture and data mixture as the targets.
- **Targets (§2):** OLMo 2 models after stage 1 pretraining and before stage 2 annealing. The 7B stage-1 run reached 3.9T tokens of a cosine schedule set for 5T; the authors added 50B tokens with LR linearly decayed to 10% of peak, for 3.95T in total (§2.1). 7B-4T is about 28xC (App. B.3, footnote 4).
- **Compute (§2.1):** C ≈ 6ND. Ladder total 5.2e21 FLOPs = 3.2% of 7B-4T (1.6e23), 1.3% of 13B-5T (3.9e23), under 1.0% of both combined.
- **Task loss and formats (§2.2):** task loss is the negative log-likelihood of the correct answer divided by its length in bytes (bits per byte), chosen over per-token normalization to reduce tokenizer effects. In RC format the predicted answer is the option with minimum task loss; in MC format the options are in the prompt and the answer letter with smallest loss is chosen. Relative error = |prediction − actual| / actual × 100%.
- **Tasks (§2.3):** MMLU, HellaSwag, ARC-Challenge, ARC-Easy, PIQA, CommonsenseQA, Social IQa, OpenBookQA from OLMES, 5-shot; test split where available, else validation. BoolQ and Winogrande are excluded because their task loss and accuracy are noisier.
- **Step 1 (§3.1):** L(N, D) = A/N^α + B/D^β + E. A, B, α, β, E are fit per task on (N, D, L) from the n = 16 ladder models, with L the mean over each model's last 5 checkpoints. Objective: mean Huber loss (δ = 1e-3) between log predicted and log actual loss; L-BFGS-B over (log A, log B, α, β, E) with A, B, α, β, E ≥ 0. Average relative fitting error 0.2% to 1.2%; average relative prediction error over all tasks 5.2% (7B-4T) and 6.6% (13B-5T) (Fig. 3).
- **Step 2 (§3.2):** Acc(L) = a/(1 + e^{−k(L − L₀)}) + b. a, b, k, L₀ are fit per task by non-linear least squares (scipy curve_fit) on m ≈ 1400 (L, Acc) points from final and intermediate ladder checkpoints, after a moving average of window 5, dropping the first 10% of each run's checkpoints, and adding the point (L = 0.0, Acc = 1.0). Fitting error 0.4% to 2.6%; average relative prediction error 3.7% (7B-4T) and 3.5% (13B-5T) (Fig. 4).
- **Chained results (§3.3, Fig. 2):** average absolute error 3.8 points (7B-4T) and 4.2 points (13B-5T); average relative error 5.9% and 6.1%. MMLU predicted 48.4 vs actual 49.0 (7B-4T) and 51.3 vs 51.6 (13B-5T). ARC-Challenge predicted 51.5 vs 61.9 and 52.7 vs 63.8; step 1 overestimates its task loss and step 2 underestimates accuracy (§4).
- **Design choices (Table 3, App. C):** compute-FLOPs as input, L(C) = A/C^α + E, had higher fitting error on all tasks and cannot distinguish compute-optimal from over-trained models of equal FLOPs (App. C.1). TaskCE (cross-entropy with answer-option losses as logits) had lower step-2 error (2.75% vs 3.6%, averaged over both targets) but higher chained error on MMLU (18.3%, 7B-4T) and HellaSwag (7.2%); its values fall in a narrow range, e.g. PIQA accuracy is random at TaskCE 0.69 and 83% at 0.65 (App. C.2). C4-en loss gave higher chained error on 5 of 8 tasks and lower error on ARC-Challenge and ARC-Easy (App. C.3). A single-step fit had higher error on 4 of 8 tasks and a degenerate fit on CommonsenseQA (App. C.5).
- **Internal inconsistency:** §3.3 states that the single-step approach has average prediction errors "> 30 points", while Table 3 lists single-step relative errors of 0.5% to 22.6%.
- **Variance analysis (§5):** Relative SD_n = SD(final n checkpoints) / Mean(final n checkpoints) × 100% (Eq. 3), computed for 1B-10xC with n = 10. Loss SD10: HellaSwag 0.0007, MMLU 0.0026, OpenBookQA 0.0047, Winogrande 0.0115 (Table 4). Loss SD10 correlates with step-2 accuracy error: r = 0.821, p = 0.004 (7B-4T); r = 0.855, p = 0.002 (13B-5T).
- **Ladder budget (App. B.3):** on MMLU, the full ladder (3.2% of 7B-4T compute) gives 1.3% error; ladder FLOPs of 0.1% of the target give about 12% (Fig. 8). Adding larger sizes lowers error for most tasks (Fig. 9); adding longer-trained models gives a smaller reduction (Fig. 10).

## Recipe ledger
All rows verified 2026-09-14 against arXiv:2412.04403v2, Table 1 unless another locus is given. Evidence for every ladder value: extrapolated from the 7B-4T configuration with the method of Porian et al. (2024) (§2.1); no ablation reported.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Ladder 190M | 190,354,176 | pretrain-stable | 1xC tokens; batch (sequences / tokens); steps at 1xC; peak LR; warmup | 3,807,083,520; 128 / 524,288; 7,272; 9.7e-4; 363 | Table 1 | verified | Porian et al. extrapolation |
| Ladder 370M | 371,262,464 | pretrain-stable | same fields | 7,425,249,280; 192 / 786,432; 9,452; 7.8e-4; 472 | Table 1 | verified | Porian et al. extrapolation |
| Ladder 760M | 758,220,288 | pretrain-stable | same fields | 15,164,405,760; 320 / 1,310,720; 11,580; 6.1e-4; 578 | Table 1 | verified | Porian et al. extrapolation |
| Ladder 1.3B | 1,279,395,840 | pretrain-stable | same fields | 25,587,916,800; 384 / 1,572,864; 16,279; 5.2e-4; 813 | Table 1 | verified | Porian et al. extrapolation |
| Ladder (all sizes) | 190M-1.3B | pretrain-stable | dimension / heads / layers / MLP ratio | 190M 768/12/12/8; 370M 1,024/16/16/8; 760M 1,536/16/16/8; 1.3B 2,048/16/16/8 | Table 1 | verified | no ablation reported |
| Ladder (all sizes) | 190M-1.3B | pretrain-stable | token multipliers; LR schedule | 1xC, 2xC, 5xC, 10xC; linear warmup, then cosine decay to 10% of peak, with the decay horizon matched to each run's data size | §2.1 | verified | matches the targets' schedule (§2.1) |
| Ladder (all sizes) | 190M-1.3B | pretrain-stable | sequence length | 4,096 tokens | batch tokens / batch sequences, e.g. 524,288 / 128 (Table 1) | derived | not applicable |
| Ladder (16 models) | 190M-1.3B | pretrain-stable | total compute | 5.2e21 FLOPs (C ≈ 6ND) | §2.1 | verified | not applicable |
| OLMo 2 7B-4T (stage-1 checkpoint) | 6,887,575,552 | pretrain-stable | batch (sequences / tokens); peak LR; warmup; dimension / heads / layers / MLP ratio | 1,024 / 4,194,304; 3.0e-4; 2000; 4,096/32/32/5.375 | Table 1 | verified | no ablation reported |
| OLMo 2 7B-4T | 6,887,575,552 | pretrain-stable | tokens; schedule | 3.9T of a cosine schedule to 10% of peak over a 5T horizon | §2.1 | verified | not applicable |
| 7B-4T authors' decay run (not the released OLMo 2 stage 2) | 6,887,575,552 | pretrain-decay/anneal | tokens; LR decay | 50B tokens, linear decay to 10% of peak; 3.95T total | §2.1 | verified | linear decay chosen to use fewer tokens than cosine (§2.1) |
| OLMo 2 13B-5T (stage-1 checkpoint) | 13,202,396,160 | pretrain-stable | tokens; batch (sequences / tokens); peak LR; warmup; dimension / heads / layers / MLP ratio | 5T; 2,048 / 8,388,608; 9.0e-4; 1000; 5,120/40/40/4 | §2; Table 1 | verified | no ablation reported |
| Targets | 7B, 13B | eval-gate | evaluation setting | OLMES, 5-shot, RC format, 8 tasks | §2.3 | verified | BoolQ, Winogrande excluded as noisier (§2.3) |

## Findings relevant to generality
- **Which tasks are predictable:** low-variance tasks with many instances (MMLU 14,042 test; HellaSwag 10,042 validation) had low error; OpenBookQA (500 instances) had high error for most design choices, and its 2.35% fitting error with C4 loss indicates randomness (§6, Fig. 2, App. C.3). The authors recommend validating a design choice on tasks with many instances (§6).
- **Choice of intermediate feature:** C4 loss predicted several tasks here, but the authors note this may not hold for downstream tasks in other domains; a task-specific loss "may work across a wider range of downstream tasks" (§6).
- **Format and emergence:** RC measures progress across a wide range of scales, while MC ability does not emerge until several billion parameters (§2.2, footnote 2). MC accuracy of the targets rises rapidly around 70k steps (7B-4T) and 20k steps (13B-5T), at the same point for all tasks (App. B.2). Fitting on the third phase gave MMLU MC end-to-end absolute errors of 0.3 (7B-4T) and 0.4 points (13B-5T), but required target checkpoints covering about 50% and 25% of training compute (App. B.2).
- **Larger target (App. D, Fig. 18):** OLMo 2 32B-6T (1.15e24 FLOPs; ladder 0.45%): average absolute error 5.81 points (9.57% relative) on 7 tasks; HellaSwag 2.7, PIQA 0.7, Social IQa 1.2 points; ARC-Challenge 13.2 and ARC-Easy 11.0 points.
- **Scope limits:** only base-model checkpoints before annealing and post-training; only multiple-choice tasks; validation on 70B and larger is future work (§2, §8).

## Connections
- [[olmo-2]]: the 7B and 13B stage-1 checkpoints are the targets; App. D adds OLMo 2 32B.
- [[olmo-2-official-configs]]: released OLMo 2 7B stage-1 and stage-2 configs to compare with the Table 1 target rows.
- [[olmes]]: source of the 8 tasks, the RC and MC formats, and the 5-shot setting.
- [[chinchilla-compute-optimal]]: origin of the Eq. 1 form, the Huber fit in log space, and 1xC = 20·N.
- [[kaplan-scaling-laws]]: C ≈ 6ND compute estimate.
- [[resolving-scaling-discrepancies]]: Porian et al. (2024), used to extrapolate ladder peak LR, batch size, and warmup.
- [[overtraining-downstream-scaling]]: Gadre et al. (2024), which predicts average top-1 error over 17 LLM-foundry tasks from C4 loss; this paper predicts individual tasks.
- [[llama-3]]: Dubey et al. (2024), a two-step prediction for ARC-Challenge that relies on compute-optimal models (§1, §7).
- [[predicting-downstream-elusive]]: Schaeffer et al. (2024), cited for accuracy depending on the losses of incorrect options (App. C.2).
- [[signal-and-noise-eval]], [[benchmark-variance-quantified]]: evaluation-noise studies related to the SD10 analysis.
- [[datadecide]]: AI2 work on predicting pretraining-data decisions from small experiments.
- [[small-scale-proxies-instabilities]]: small-model proxies for training stability rather than task accuracy.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2412.04403 (v2, 2025-08-22, COLM 2025; v1 2024-12-05). The v1 text was not compared.
- Audit claims not found in the source: none. Precision notes: "1% of target compute" is the share of both targets combined (3.2% of 7B-4T, 1.3% of 13B-5T; §2.1); "within 2 points" is stated in the abstract and §4, but Fig. 2 lists a 2.1-point error for HellaSwag on 13B-5T; the "7B / 4T-token" target was trained on 3.95T tokens (§2.1).
- Not reported by the source: data mixture composition beyond "same data mixture" as OLMo 2; sequence length (derived above); predictions for annealed, post-trained, or generative-task settings.
