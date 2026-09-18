<!-- scope: ReST-EM (Singh et al., arXiv:2312.06585, TMLR 2024) — expectation-maximization self-training of PaLM 2 on binary-reward-filtered samples for MATH and APPS; iterations, transfer, pass@K, distillation
     deps: [[star]]
     see-also: [[v-star]], [[rejection-sampling-finetuning]], [[rlvr-tulu3]], [[rlvr-beyond-base-model]]
-->

# Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
- **Core Insight:** Fine-tuning PaLM 2 models on their own samples filtered by a binary correctness reward outperforms fine-tuning on human-written solutions on Hendrycks MATH and APPS (Introductory) (§5.1, Figs. 2-3); with 64-sample majority voting, PaLM 2-L after ReST-EM reaches 48.82% MATH test accuracy against 44.02% for the base model (§5.2).
- **Guideline:** When a task has an automatic binary correctness check and a fixed set of a few thousand training problems, run ReST-EM iterations that restart fine-tuning from the base model and evaluate the held-out test set after every iteration, because training accuracy kept rising while MATH test gains were small after the first iteration and APPS (2,342 problems) regressed at the second iteration (§5.1, Fig. 4). Otherwise, for a problem set as small as APPS, stop after one iteration.
- **Authors:** Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, et al. (Google DeepMind; R. Agarwal also Mila)
- **Year:** 2023 (arXiv v1 2023-12-12; published in Transactions on Machine Learning Research, 04/2024)
- **URL:** https://arxiv.org/abs/2312.06585
- **Source type:** paper
- **Relevant topics:** self-training, expectation-maximization for RL, rejection-sampled SFT, binary verifiable reward, MATH, APPS, transfer, pass@K, distillation

## Abstract
Fine-tuning on human-written data is limited by the quantity and diversity of that data. The paper studies whether scalar feedback, such as checking a math answer, can replace it. ReST-EM (1) samples outputs from the model and filters them with a binary reward, (2) fine-tunes the model on the kept samples, and (3) repeats a few times. On MATH and APPS with PaLM 2 models, ReST-EM scales favorably with model size and surpasses fine-tuning on human data (Abstract).

## Key Contributions
- Derives ReST-EM as EM for RL: a binary optimality variable O with p(O=1|x,y) ∝ f(r(x,y)); the E-step reweights samples by reward, the M-step maximizes reward-weighted log-likelihood (§3, Eqs. 2-3).
- Simplifies ReST (Gulcehre et al., 2023): no human data in the Generate step, and every Improve step fine-tunes the base model rather than the previous iterate (§3; Table 1).
- Shows larger gains from self-generated data than from human data for PaLM 2-S, PaLM 2-S*, and PaLM 2-L (Figs. 2-3).
- Ablations on iterations, human vs model data at equal question count, distillation to a smaller model, dataset size, and difficulty (§5.3).
- Transfer evaluation on GSM8K, HumanEval, the 2023 Hungarian HS finals exam, and Big-Bench Hard (§5, §5.4).

## Key Figures/Tables to Study
- Table 1: ReST-EM vs ReST vs STaR vs RFT (base-model restart, rationalization, temperature sampling, held-out evaluation).
- Figs. 2-3: MATH/GSM8K and APPS/HumanEval test accuracy vs iteration, with human-data SFT baselines.
- Fig. 4: train vs test accuracy vs iteration (MATH with PaLM 2-L; APPS with PaLM 2-S*).
- Fig. 5: pass@K for PaLM 2-L base vs ReST-EM on HumanEval, APPS, MATH.
- Fig. 6: SFT(7K), SFT(5K), ReST*(5K), ReST-EM(5K); distillation from PaLM 2-L into PaLM 2-S.
- Fig. 7: ReST-EM vs ReST on APPS and HumanEval (PaLM 2-S*). Fig. 8: dataset size; difficulty levels. Fig. 9: BBH.

## Technical Details
**Objective.** M-step: J(θ) = E_(x,y)~D_i [ r(x,y) log p_θ(y|x) ] (Algorithm 1). x is the input context, y a sampled output, r ∈ {0,1} the binary reward, D_i the iteration-i sample set. Because the experiments use r ∈ {0, 1} (§3 Remark), an incorrect sample has zero weight and is not trained on (derived from Algorithm 1).
**EM vs online RL.** EM-based RL samples from the fixed policy of the previous iteration, which decouples data collection from optimization; the authors state this makes scaling to large policies easier (§3).
**Data.** MATH has 7,500 and APPS (Introductory) 2,342 training problems; MATH is checked against the ground-truth answer, APPS by test cases (§5).
**Generate step.** 32 solutions per problem for MATH and 64 for APPS; top-K sampling with K = 40, temperature 0.7; at most 10 solutions per problem kept, to limit the over-representation of easy problems (§5 Implementation Details).
**Scale trend.** MATH test accuracy improvement with ReST-EM is 5.94% for PaLM 2-S and 6.34% for PaLM 2-L; APPS improvement is 5.6% for PaLM 2-S* and 6.4% for PaLM 2-L (§5.1; the baseline for these differences is not stated at the locus).
**Iterations.** On MATH, one iteration with 3× the samples per problem gives 40.3% pass@1, below 41% at iteration 2 and 41.9% at iteration 3 (PaLM 2-L; §5.3). On APPS most of the gain comes from iteration 1; further iterations regress on APPS and HumanEval (§5.1).
**Equal-question comparison.** On about 5K MATH questions with at least one correct model solution, fine-tuning on one random model solution per question (ReST*) beats one human solution per question (SFT 5K) (§5.3, Fig. 6 left).
**Dataset size.** One iteration on 1,000 MATH questions gives significant gains; 4,000 questions scored slightly below 2,000, which the authors attribute to fine-tuning variance with a single run (§5.3, Fig. 8 left).
**Difficulty.** Questions binned by base-model success at T = 1.0 (easy 75-100%, medium 50-75%, hard 25-50%, very hard <25%) all improve; medium and hard gain most (§5.3, Fig. 8 right).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| PaLM 2-S (Bison), PaLM 2-S* (Codey), PaLM 2-L (Unicorn) | not reported | SFT | training problems (unique prompts) | MATH 7,500; APPS (Introductory) 2,342 | arXiv:2312.06585v4 §5 Training Datasets | verified 2026-09-14 | Fig. 8 left: 1K-7K MATH questions, PaLM 2-L, 1 iteration, 1 run |
| same | not reported | SFT | samples per problem, Generate step | MATH 32; APPS 64 | §5 Implementation Details | verified 2026-09-14 | §5.3: 3× samples in 1 iteration (40.3%) < iterations 2-3 (41%, 41.9%), PaLM 2-L MATH |
| same | not reported | SFT | sampling for Generate step | top-K, K = 40; temperature 0.7 | §5 Implementation Details | verified 2026-09-14 | no ablation reported |
| same | not reported | SFT | reward / filter | binary: ground-truth answer match (MATH); test cases (APPS) | §5 Training Datasets | verified 2026-09-14 | n/a |
| same | not reported | SFT | max kept solutions per problem | 10 (MATH and APPS) | §5 Implementation Details | verified 2026-09-14 | no ablation reported |
| same | not reported | SFT | loss and input | next-token loss on model solutions only; input = few-shot prompt + question | §5 Implementation Details, Eq. 1 | verified 2026-09-14 | no ablation reported |
| same | not reported | SFT | initialization per iteration | base pretrained model | §3 Improve step; Table 1 | verified 2026-09-14 | Fig. 7 (PaLM 2-S*): similar APPS, better HumanEval transfer than continuing from last iterate |
| same | not reported | SFT | M-step stopping rule | train while reward improves on D_val | Algorithm 1 | verified 2026-09-14 | no ablation reported |
| same | not reported | SFT | iterations run | MATH 3; APPS 2 | Figs. 2-3 x-axes; Fig. 6 caption | verified 2026-09-14 | Fig. 4 train-test gap |
| same | not reported | SFT | learning rate, batch, epochs, optimizer, schedule, tokens, compute | not reported | checked §3, §5, Algorithm 1, all figure captions; v4 has no appendix | not reported | n/a |
| same | not reported | eval-gate | decoding | greedy (Figs. 2-3); pass@K T = 1.0, nucleus p = 0.95 (Fig. 5); majority voting 64 samples (§5.2) | Fig. 2 caption; Fig. 5 caption; §5.2 | verified 2026-09-14 | n/a |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality.** On BBH, the MATH-trained and APPS-trained PaLM 2-L models show no major degradation; the MATH model beats the base model with chain-of-thought prompting, and all three are similar with direct prompting (§5.4, Fig. 9). PaLM 2-L after ReST-EM on MATH scores above all compared models except GPT-4 on the Hungarian exam (1-shot, temperature 0.1, manual grading; §5.4, Fig. 10). Restarting from the base model each iteration is the design choice the authors link to transfer (§3, Fig. 7).
- **Coverage (pass@K).** ReST-EM is stronger than the base model for all K in Fig. 5, with the largest gap at K = 1 (§5.2). The authors state ReST-EM may not close the gap to pass@K at large K (§6).
- **Negative feedback.** Incorrect samples are discarded (reward 0, zero weight); the paper does not train on them in any of the four senses of style guide §6.1. In preliminary experiments, STaR-style rationalization (conditioning on the correct answer) increased false positives: correct final answer with incorrect reasoning (§4).
- **Distillation.** PaLM 2-S fine-tuned on one PaLM 2-L solution per question (Distill*) beats PaLM 2-S fine-tuned on human solutions; multiple PaLM 2-L solutions (Distill) beat PaLM 2-S's own ReST-EM data, which the authors attribute to more questions having solutions (§5.3, Fig. 6 right).

## Connections
- [[star]]: greedy decoding with one solution per problem plus rationalization; ReST-EM uses temperature sampling and no rationalization (§4, Table 1).
- [[v-star]]: uses the incorrect self-generated solutions that ReST-EM discards to train a DPO verifier (arXiv:2402.06457 Abstract).
- [[rejection-sampling-finetuning]]: RFT (Yuan et al., 2023) corresponds to one Generate and one Improve step of ReST-EM (§4).
- [[rlvr-tulu3]]: online RL on binary verifiable rewards, the coupled sampling-and-update setting that §3 contrasts with EM.
- [[rlvr-beyond-base-model]]: large-k pass@k comparison of RL-trained and base models; relates to the §6 pass@K limitation.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2312.06585 (v4, 18 Apr 2024, full text); numbers also searched in v1 (12 Dec 2023).
- Corrections: "PaLM-2-L MATH 34.1% → 50.6%" and "APPS 16.4% → 31.2%" → not in v1 or v4; reported values are the §5.1-§5.3 numbers above. "K=32 at T=1.0, top-p=0.95" → 32 (MATH) / 64 (APPS), top-K 40, T 0.7 (§5); T 1.0 / p 0.95 is the pass@K evaluation (Fig. 5). "At most 4 distinct correct solutions" → 10 (§5). "Iterations: 2 for MATH; saturation at 2-3 then overfitting" → MATH run to 3 iterations with small gains after iteration 1; APPS regresses at iteration 2 (§5.1, Fig. 4). "EM on a latent rationale variable" → EM over a binary optimality variable (§3); the latent-rationale view is TRICE (§4). "Ablated on -S and -XS" → PaLM 2-S, PaLM 2-S*, PaLM 2-L (§5). "Figure 4 = BBH; Figure 6 = diversity; Table 2" → Fig. 4 is train-test gap, BBH is Fig. 9, Fig. 6 is human-data and distillation comparison, there is no Table 2. "Google DeepMind / Brain" → Google DeepMind and Mila. [[v-star]] "value function over partial rationales" → DPO verifier on correct and incorrect solutions.
- Removed as unsupported: "≈340B active params"; "1 epoch; lr=1e-5; batch 128"; "~100 H100-hrs per iter, N=10K problems"; "Figure 2 iter-1 +8%, iter-2 +6%"; "iter-3 regresses unless diversity filtering is added"; "iterations reduce solution-path diversity"; precursor claims about [[rlvr-tulu3]] and DeepSeek-R1; Self-Rewarding LM saturation comparison.
- Not reported by the source: PaLM 2 parameter counts, optimizer and learning-rate settings, compute, per-iteration APPS/MATH accuracy tables.
