<!-- scope: SCoRe — two-stage multi-turn online RL that trains a single model to self-correct
     deps: [[ppo]], [[grpo]]
     see-also: [[star]], [[rest-em]], [[deepseek-r1]], [[let-verify]]
-->

# Training Language Models to Self-Correct via Reinforcement Learning
- **Core Insight:** Supervised fine-tuning on model-generated correction traces produces a Δ(t1,t2) of at most 1.8% on MATH (Table 1), while two-stage on-policy multi-turn RL (SCoRe) raises Δ(t1,t2) by 15.6% over the base Gemini 1.5 Flash model on MATH and by 9.1% over base Gemini 1.0 Pro on HumanEval (Abstract; §6.1).
- **Guideline:** When training intrinsic self-correction from a model's own data, use on-policy multi-turn RL with (a) a first stage that constrains the first attempt to the base model with a KL penalty while optimizing the second attempt, and (b) a second stage that adds a shaped bonus α·(r(y2) − r(y1)) with α > 1, because removing either component reduces Δ(t1,t2) from 4.4% to 2.2% or 2.6% (§6.3, Table 4).
- **Authors:** Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, JD Co-Reyes, Avi Singh, et al. (Google DeepMind)
- **Year:** 2024 (arXiv v1 2024-09; v2 2024-10-04)
- **URL:** https://arxiv.org/abs/2409.12917
- **Source type:** paper
- **Relevant topics:** self-correction, multi-turn RL, reward shaping, on-policy training, math and code reasoning

## Abstract
Self-correction is a desirable capability of large language models but has been found largely ineffective in current models. Existing training methods for self-correction depend on multiple models, a more advanced model, or additional supervision. The paper develops SCoRe, a multi-turn online RL approach that improves an LLM's self-correction ability using entirely self-generated data. The authors first show that variants of supervised fine-tuning on offline model-generated correction traces are insufficient: SFT falls prey either to a distribution mismatch between the mistakes of the data-collection policy and the model's own responses, or to behavior collapse, in which training implicitly prefers one mode of correction behavior that is ineffective at test time. SCoRe trains under the model's own distribution of self-generated correction traces and applies regularization that steers learning toward a self-correction behavior effective at test time rather than fitting high-reward responses for a given prompt. The regularization consists of an initial phase of multi-turn RL that produces a policy initialization less susceptible to collapse, followed by a reward bonus that amplifies self-correction. With Gemini 1.0 Pro and 1.5 Flash, SCoRe improves the base models' self-correction by 15.6% on MATH and 9.1% on HumanEval.

## Key Contributions
- Diagnoses two failure modes of SFT on self-correction traces: distribution shift between the data-collection policy's first attempts and the trained model's own first attempts, and behavior collapse in which the model makes minor or no edits at the second attempt (§4).
- Measures those failure modes: STaR on 𝒟STaR gives Δ(t1,t2) = −14.2%; Pair-SFT on 𝒟SFT gives +1.8%; adding correct-to-correct pairs raises STaR to +0.4% and drives Pair-SFT to exactly 0% edits (Table 1).
- Introduces SCoRe, a two-stage on-policy recipe built on REINFORCE with a KL penalty to a fixed reference (§5, Eq. 2-4).
- Reports Δ(t1,t2) = 4.4% on MATH500 with Accuracy@t2 = 64.4%, versus −11.2% and 41.4% for base Gemini 1.5 Flash (Table 2), and Δ(t1,t2) = 12.2% with Accuracy@t2 = 64.6% on HumanEval versus 3.0% and 56.7% for base Gemini 1.0 Pro (Table 3).
- Shows that self-correction can be combined with parallel sampling: at a budget of 32 samples per problem, parallel-only self-consistency gains 7.4% accuracy while K parallel samples plus one self-correction round gains 10.5% (§6.2, Figure 1 right).

## Key Figures/Tables to Study
- **Table 1** — self-correction metrics after training on 𝒟STaR and 𝒟SFT; both give negative or near-zero Δ(t1,t2).
- **Figure 4** — histograms of edit-distance ratios between the first and second attempts for SFT, STaR, Pair-SFT and SCoRe; SFT-trained models make conservative edits.
- **Figure 6** — behavior collapse in standard multi-turn RL, and Δ(t1,t2) over training for Stage I versus Stage II.
- **Figure 7** — overview of the two stages.
- **Tables 2 and 3** — MATH and HumanEval results against Self-Refine, STaR and Pair-SFT.
- **Table 4** — ablations: without multi-turn training, without Stage I, without reward shaping, and with STaR replacing REINFORCE in Stage II.

## Technical Details
- **Turn structure:** two attempts (l = 2). Turn 1 prompt is the problem; turn 2 prompt appends the turn-1 response plus a self-correction instruction that does not reveal whether the first answer was right (§6, Appendix C).
- **MATH self-correction instruction (verbatim, Appendix C):** "There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution."
- **Base RL algorithm:** REINFORCE policy gradient with a KL penalty against a fixed reference policy, following Ahmadian et al. (2024) (§3, Eq. 2).
- **Stage I objective (Eq. 3):** maximize r̂(y2, y*) − β2·KL(π_θ(·|x1) ‖ π_ref(·|x1)); the strict KL penalty applies to the first attempt only, so the first-attempt distribution stays near the base model while the second attempt is optimized. The default KL term of Eq. 2 is retained with a small weight.
- **Stage II objective (Eq. 4):** maximize Σ_{i=1,2} r̂(y_i, y*) − β1·KL(π_θ(·|x_i) ‖ π_ref(·|x_i)), initialized from Stage I.
- **Reward shaping (§5.2):** the second-attempt reward is augmented with b̂(y2 | y1, y*) = α·(r̂(y2,y*) − r̂(y1,y*)), α a positive constant "ideally larger than 1.0"; it rewards transitions that flip incorrect to correct and penalizes correct-to-incorrect transitions.
- **Reward:** binary, from an answer match against ground truth (MATH) or from passing all test cases (code) (§6).
- **Data splits:** MATH training set augmented with 4500 problems from the test set, evaluated on the remaining 500 (MATH500), following Lightman et al. (2023); code models train on MBPP and are evaluated on HumanEval, which does not expose test cases (§6).
- **Decoding:** greedy (temperature 0) for all evaluations, except temperature 0.7 for the inference-compute scaling study (§6).
- **Checkpoint selection:** the checkpoint with the highest training reward (§6).
- **Multi-attempt behavior:** trained on two attempts only, SCoRe improves slightly past turn 2 over 10 attempts and then plateaus; the base model never exceeds its first-attempt accuracy and Pair-SFT does not improve past the second attempt (Appendix A.1, Figure 8).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | base model | Gemini 1.5 Flash | arXiv:2409.12917v2 App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | optimizer | Adam | App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | learning rate | 5e-6 | App. B Table 5 | verified 2026-09-18 | fixed across runs (§6) |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | training steps | 3000 | App. B Table 5 | verified 2026-09-18 | fixed sample and update budget (§6) |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | batch size | 512 | App. B Table 5 | verified 2026-09-18 | fixed across runs (§6) |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | sampling temperature | 1.0 | App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | reward-shaping α | 10 | App. B Table 5 | verified 2026-09-18 | §6.3 Table 4: removing shaping drops Δ(t1,t2) 4.4% → 2.6% |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | β1 (KL to reference, both turns) | 0.01 | App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.5 Flash (SCoRe, MATH) | not reported | RL | β2 (KL on first attempt, Stage I) | 0.1 | App. B Table 5 | verified 2026-09-18 | §6.3 Table 4: removing Stage I drops Δ(t1,t2) 4.4% → 2.2% |
| Gemini 1.0 Pro (SCoRe, MBPP) | not reported | RL | base model | Gemini 1.0 Pro | App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.0 Pro (SCoRe, MBPP) | not reported | RL | learning rate | 1e-5 | App. B Table 5 | verified 2026-09-18 | fixed across runs (§6) |
| Gemini 1.0 Pro (SCoRe, MBPP) | not reported | RL | training steps | 1500 | App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.0 Pro (SCoRe, MBPP) | not reported | RL | batch size | 128 | App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.0 Pro (SCoRe, MBPP) | not reported | RL | reward-shaping α | 10 | App. B Table 5 | verified 2026-09-18 | no ablation reported |
| Gemini 1.0 Pro (SCoRe, MBPP) | not reported | RL | β1 / β2 | 0.01 / 0.25 | App. B Table 5 | verified 2026-09-18 | no ablation reported |

## Findings relevant to negative feedback
- The shaped bonus α·(r(y2) − r(y1)) is a negative-as-gradient signal on one transition type: a correct first attempt turned incorrect receives a negative reward of magnitude α (§5.2). With α = 10 the fraction of correct answers changed to incorrect falls from 15.8% (base) to 1.4% (SCoRe) on MATH (§6.1, Table 2).
- Adding correct-to-correct pairs to the SFT datasets, an offline attempt at the same control, removes the behavior entirely for Pair-SFT: Δ(t1,t2), Δ^{i→c} and Δ^{c→i} all become 0% (Table 1). The model stops editing rather than learning when to edit.

## Findings relevant to generality
- The code models train on MBPP and are evaluated on HumanEval; SCoRe reaches Δ(t1,t2) = 12.2% on HumanEval, 9.1 points above the base model, which the authors describe as generalization from the training task (§6.1).
- On the offline repair task MBPP-R the base model scores 47.3% and SCoRe 60.6% (Table 3).
- Replacing on-policy REINFORCE with STaR in Stage II lowers Accuracy@t2 to 58.4% and Δ(t1,t2) to 2.2%, which the authors contrast with Havrilla et al. (2024a) reporting similar convergence for STaR and on-policy RL in the single-turn setting (§6.3).

## Connections
- [[star]] and [[rest-em]] bootstrap successful solutions; SCoRe evaluates STaR as a baseline for correction traces and reports Δ(t1,t2) = −14.2% for it on MATH (Table 1).
- [[deepseek-r1]] uses a single-turn verifiable-reward setup; SCoRe keeps the same binary reward but spans two turns with a shaped inter-turn bonus.
- [[let-verify]] supplies the MATH500 split used here (Lightman et al., 2023, §6).
- [[ppo]] and [[grpo]] are alternative policy-gradient estimators; SCoRe uses REINFORCE with a KL penalty (§3).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2409.12917 (arXiv v2, 2024-10-04)
- Corrections to the previous card version:
  - "9.1 pts on MBPP" → the 9.1% gain is on HumanEval; MBPP is the training set for the code runs (Abstract; §6.1).
  - "Base model: Gemini 1.0 Pro; also reproduced on Gemma-2-9B" → MATH runs use Gemini 1.5 Flash and code runs use Gemini 1.0 Pro (App. B Table 5; §6).
  - "Optimizer: AdamW, lr 1e-6, batch 256" → Adam; lr 5e-6 and batch 512 (MATH), lr 1e-5 and batch 128 (MBPP) (App. B Table 5).
  - "with α=2.0" → α = 10 in both configurations; the text states only that α should be larger than 1.0 (§5.2; App. B Table 5).
  - "Achieves 15.6 pts ... the first method to cross zero" → SCoRe's own Δ(t1,t2) is 4.4% on MATH and the 15.6% is the improvement over the base model's −11.2%; STaR and Pair-SFT baselines already have small positive deltas (Table 1, Table 2).
  - "Figure 2 (self-correction Δ over training)" → Figure 2 shows two example correction traces; the training-curve comparison is Figure 6.
  - "Figure 5 (ablation of Stage I vs direct Stage II)" → the Stage I ablation is Table 4; Figure 5 shows self-correction performance on different sets of first attempts.
  - "Figure 4 (reward-shaping bonus coefficient)" → Figure 4 shows edit-distance histograms; no figure sweeps α.
  - "Table 2 (MATH, MBPP, HumanEval): SCoRe gains vs Self-Refine / reflexion / STaR" → Table 2 is MATH only and Table 3 is HumanEval and MBPP-R; Reflexion is cited as similar to Self-Refine but is not a baseline row.
  - "Stage II loss: ... with α=2.0" trajectory-level formula → Stage II optimizes the summed two-turn reward of Eq. 4 with the bonus applied to the second attempt only (§5.2).
- Removed as unsupported by the source: reproduction on Gemma-2-9B; "training set: MATH training split + self-generated traces" stated without the 4500-problem test-set augmentation; the claim that SCoRe is the first method with a positive delta; the claim that SCoRe uses "the same clipped REINFORCE / GRPO style update as DeepSeek-R1"; the analogy between Stage I and IPO's regularization.
- Not reported by the source: parameter counts of Gemini 1.0 Pro and 1.5 Flash; number of prompts or samples per prompt per RL step; total compute; the value of the small default KL weight used alongside β2 in Stage I.
