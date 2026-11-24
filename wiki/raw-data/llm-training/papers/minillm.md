<!-- scope: MiniLLM (Gu, Dong, Wei, Huang; ICLR 2024) — white-box knowledge distillation of generative LMs by minimizing reverse KL with on-policy policy-gradient training; student samples scored by teacher log-probabilities
     deps: [[ppo]], [[rlhf-instructgpt]]
     see-also: [[distilling-step-by-step]], [[distillation-source-matters]], [[orca]], [[on-off-policy-rlhf]], [[kl-control-rlhf]], [[phi-4-mini-reasoning]]
-->

# MiniLLM: On-Policy Distillation of Large Language Models
- **Core Insight:** Replacing forward KL (word-level KD, or SFT on teacher samples) with reverse KL KL[q_θ‖p], optimized on student-sampled responses with a policy-gradient estimator, gives the best student GPT-4-feedback score in 20 of 21 student/test-set cells (the exception is GPT-2 340M on VicunaEval, 42.6 vs 43.0 for SeqKD) across GPT-2 (120M–760M), OPT (1.3B–6.7B) and LLaMA-7B students, e.g. LLaMA-7B DollyEval 76.4 vs 73.6 for sequence-level KD (§2.1–2.2, Table 1).
- **Guideline:** When a white-box teacher is available and the student is too small to cover all teacher modes, train the student on its own samples scored by teacher log-probabilities, and add three stabilizers — single-step decomposition, teacher-mixed sampling (α = 0.2), and length normalization — because without the last two the student learns short or repeated strings that the teacher scores highly (§2.2, §3.4, Table 4).
- **Authors:** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang (Tsinghua CoAI Group; Microsoft Research)
- **Year:** 2023 (arXiv v1 2023-06-14; v6 2026-01-31); ICLR 2024 per arXiv comments field
- **URL:** https://arxiv.org/abs/2306.08543 (code: https://github.com/microsoft/LMOps/tree/main/minillm)
- **Source type:** paper
- **Relevant topics:** knowledge distillation, reverse KL, on-policy distillation, exposure bias, mode seeking, policy gradient, reward hacking in distillation

## Abstract
Standard KD for generative LMs approximately minimizes forward KL between teacher p and student q_θ, which pushes q_θ to cover all modes of p and to place probability on regions where p is near zero. MiniLLM minimizes reverse KL, which is mode-seeking, and derives an on-policy optimization procedure for it. On instruction following, MiniLLM students give more precise responses, higher overall quality, lower exposure bias, better calibration and better long-response performance than baselines, for students from 120M to 13B parameters in the abstract (the main table uses students up to 7B with 13B teachers).

## Key Contributions
- Reverse-KL objective for LLM distillation with a toy Gaussian-mixture illustration of mode-seeking vs mode-covering (§2.1, Fig. 2).
- Policy-gradient form of the gradient, with per-step reward r_t = log p(y_t|·)/q_θ(y_t|·) (§2.2, Eq. 2).
- Three stabilizers: single-step decomposition (variance), teacher-mixed sampling (reward hacking), length normalization (length bias) (§2.2, Eqs. 3–7).
- Results on 5 instruction-following test sets with Rouge-L, GPT-4 feedback and human evaluation (§3.2, Table 1, Fig. 4).
- Analyses: teacher-size scaling, exposure bias (ExAccErr), calibration (ECE), response-length subsets, diversity (§3.3).

## Key Figures/Tables to Study
- Fig. 2: forward vs reverse KL toy fit. Fig. 3: sequence-level KD vs MiniLLM data flow.
- Table 1: main results. Table 4 and Fig. 8: ablation of the three stabilizers. Table 2: calibration. Fig. 6: exposure bias.

## Technical Details
**Objective (Eq. 1).** θ = argmin KL[q_θ‖p] = argmin −E_{x∼p_x, y∼q_θ} log [p(y|x)/q_θ(y|x)].

**Gradient (Eq. 2).** ∇L = −E Σ_t (R_t − 1) ∇ log q_θ(y_t|y_<t, x), with R_t = Σ_{t'≥t} log[p(y_t'|·)/q_θ(y_t'|·)].

**Stabilizers (§2.2).**
1. Single-step decomposition (Eq. 3): split ∇L into (∇L)_Single, the gradient of E_{y_t∼q_θ(t)}[r_t] computed exactly by summing over the vocabulary, and (∇L)_Long, which uses R_{t+1}.
2. Teacher-mixed sampling (Eq. 4): sample from p̃ = α·p + (1 − α)·q_θ at each step; importance weight approximated per token as w_t ≈ q_θ(y_t|·)/p̃(y_t|·) to avoid the variance of the product over steps (Eq. 5).
3. Length normalization (Eq. 6): R^Norm_{t+1} = (1/(T − t − 1)) Σ_{t'>t} log[p/q_θ], because unnormalized R_{t+1} favours short outputs.
- A PPO-style ratio clip is applied to the Long part, and a pretraining LM loss L_PT on D_PT is added (Algorithm 1).

**Training procedure (§2.3, App. B.1).** Phase 1: SFT the student on D for 3 epochs, pick the lowest-validation-loss checkpoint. Phase 2: Algorithm 1 for 5,000 steps; final checkpoint chosen by validation Rouge-L.

**Setup (§3.1).** Teachers: GPT-2-1.5B, OPT-13B, LLaMA-13B, each fine-tuned on D. Students: GPT-2 120M/340M/760M; OPT 1.3B/2.7B/6.7B; LLaMA-7B. D = databricks-dolly-15k with 1K validation and 0.5K test held out, about 12.5K training examples. D_PT = OpenWebText (GPT-2) or the RoBERTa corpus (others). Test sets: DollyEval (500), SelfInst (252), VicunaEval (80), S-NI and UnNI (ground-truth length ≥ 11 subsets). Responses sampled at temperature 1, averaged over 5 seeds.

**Results (Table 1, GPT-4 feedback on DollyEval).** GPT-2 120M: SFT 38.6, KD 40.3, SeqKD 41.2, MiniLLM 44.7 (teacher 58.4). OPT-1.3B: 52.6 / 52.7 / 51.0 / 60.7 (teacher 70.3). LLaMA-7B: 73.0 / 73.7 / 73.6 / 76.4 (teacher 79.0).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MiniLLM students (all) | 120M–7B | distillation | teacher-mix α | 0.2 | arXiv:2306.08543v6 §3.1; App. C.4 | verified 2026-11-24 | Fig. 16: α = 0.2 suitable across families; larger students less sensitive |
| MiniLLM students | 120M–7B | SFT (phase 1) | epochs; LR / batch | 3; best of the SFT baseline sweep | App. B.1 | verified 2026-11-24 | checkpoint by lowest validation loss |
| MiniLLM students | 120M–7B | distillation (phase 2) | LR; mini-batch; rollout buffer; inner epochs | 5e-6; 64; 256 sentences; 4 | App. B.1 | verified 2026-11-24 | no ablation reported |
| MiniLLM students | 120M–7B | distillation (phase 2) | clip ε; max length; sampling T; steps | 0.2; 512; 1.0; 5,000 | App. B.1 | verified 2026-11-24 | no ablation reported |
| MiniLLM students | OPT-1.3B, LLaMA-7B | distillation | pretraining loss L_PT | on | Algorithm 1; App. C.4, Table 7 | verified 2026-11-24 | Table 7: w/o L_PT, CLS 70.2 → 65.7 (1.3B), 78.8 → 74.3 (7B); Inst. 52.8 → 53.2, 71.2 → 71.1 |
| LLaMA-13B → LLaMA-7B | 7B | distillation | compute | < 10 hours on 16 V100 32G | App. B.1 | verified 2026-11-24 | not applicable |
| Baselines (SFT, KD, SeqKD) | <1.3B / ≥1.3B | SFT | LR grid; batch; epochs | [5e-4, 1e-4, 5e-5] / [5e-5, 1e-5, 5e-6]; 32 or 64; 20 / 10 | App. B.1 | verified 2026-11-24 | KD mixes distillation and LM loss at rate 0.5 |

## Findings relevant to generality and negative feedback
- **Reward hacking in distillation.** Without teacher-mixed sampling or length normalization, reverse KL still decreases but the student produces repeated, short or meaningless strings that have high teacher probability (§3.4, Fig. 8, App. D). Validation / Dolly Rouge-L: full 27.4 / 24.6; w/o length norm 17.4 / 14.7; w/o teacher-mixed 22.3 / 20.4; w/o single-step 27.0 / 23.7 (Table 4, GPT-2-125M student).
- **Exposure bias.** ExAccErr of SFT, KD and SeqKD grows with generation length, while MiniLLM's stops growing after about 150 tokens (§3.3, Fig. 6).
- **Calibration and retention.** On SST2 / BoolQ (LLaMA-7B), ECE: teacher 0.025 / 0.356, KD 0.191 / 0.682, SeqKD 0.243 / 0.681, MiniLLM 0.099 / 0.502 (Table 2). Removing L_PT lowers zero-shot classification accuracy (Table 7), i.e. the LM loss is the forgetting control.
- **Diversity.** Distinct-4 and test LM loss: MiniLLM 99.0 / 3.95 vs SFT 99.5 / 3.89 on DollyEval (Table 3); the authors argue one correct response is sufficient for many applications (§3.3).
- **Where it does not help.** On S-NI prompts with ground-truth length ≤ 5 tokens, all methods score similarly; the advantage appears for ≥ 6-token responses (§3.3, Fig. 7).
- **Out of distribution.** Gains are reported as larger on test sets other than Dolly (§3.2).

## Connections
- [[ppo]], [[rlhf-instructgpt]]: the two-phase SFT-then-policy-optimization pipeline and the L_PT term follow InstructGPT (§2.3).
- [[distilling-step-by-step]], [[orca]]: black-box distillation by SFT on teacher outputs, the forward-KL family this paper contrasts with.
- [[on-off-policy-rlhf]]: related on-policy vs off-policy data question in preference RL.
- [[phi-4-mini-reasoning]]: a later small-model recipe that uses black-box (sequence-level) distillation, for contrast.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2306.08543 (v6 PDF, full text, dated 31 Jan 2026) and the arXiv abstract page (comments: "Published as a conference paper in ICLR 2024"). The repository path https://github.com/microsoft/LMOps/tree/main/minillm was confirmed to exist (README returns 200); code was not inspected.
- The "20 of 21" count was made by comparing student GPT-4 columns (DollyEval, SelfInst, VicunaEval × 7 students) in Table 1; the paper itself says "outperforms the baselines in almost all cases".
- Not reported: exact number of GPUs for smaller students; wall-clock for OPT runs; any LLaMA-2 or later model results in v6 main table.
