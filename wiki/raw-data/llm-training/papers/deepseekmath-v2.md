<!-- scope: DeepSeekMath-V2 (arXiv:2511.22570): an RL-trained natural-language proof verifier, a meta-verifier that checks the verifier's cited issues, a proof generator rewarded by the verifier and by the accuracy of its own self-analysis, and automated labeling by scaled verification
     deps: [[grpo]], [[deepseekmath]], [[lets-verify]]
     see-also: [[deepseek-v3.1]], [[deepseek-v3-2]], [[deepseek-prover-v2]], [[generative-reward-models]], [[self-correct-rl]], [[training-verifiers-to-solve-math-word-problems]], [[deepseek-r1]]
-->

# DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning
- **Core Insight:** A proof verifier trained with RL to match expert scores (0, 0.5, 1) can be made faithful by multiplying its reward by a meta-verifier score that checks whether the cited issues exist; used as the reward model for a proof generator that must also score its own proof, it yields a model that, with scaled test-time verification and refinement, scores 118/120 on Putnam 2024 and solves 5 of 6 IMO 2025 problems (§2.1.2, §2.2.2, §3.3.3, Table 1).
- **Guideline:** When final-answer rewards do not apply (proofs), train a generative verifier on expert scores, add a meta-verification reward so the verifier cannot earn full reward while citing non-existent issues, and reward the generator for an accurate self-assessment as well as for proof quality (α = 0.76, β = 0.24), so that admitting an error scores higher than a false claim of correctness (§2.1.2, Eq. 5–6).
- **Authors:** Zhihong Shao, Yuxiang Luo, Chengda Lu, Z.Z. Ren (core contributors), Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, Xiaokang Zhang (DeepSeek-AI)
- **Year:** 2025 (arXiv v1 2025-11-27; only version)
- **URL:** https://arxiv.org/abs/2511.22570 (code, outputs, weights: https://github.com/deepseek-ai/DeepSeek-Math-V2)
- **Source type:** paper
- **Relevant topics:** generative verifiers, meta-verification, proof reward, self-verification, generation–verification gap, automated labeling, sequential refinement, test-time compute, GRPO

## Abstract
RL with final-answer rewards has saturated AIME and HMMT, but a correct answer does not guarantee correct reasoning, and theorem proving has no final answer to check. The paper trains an LLM verifier that scores natural-language proofs without reference solutions, then trains a proof generator with the verifier as reward model and rewards it for finding and fixing issues in its own proofs before finalizing. To keep the verifier ahead of the improving generator, verification compute is scaled to label new hard-to-verify proofs automatically, and these labels retrain the verifier. With scaled test-time compute the model reaches gold-level scores on IMO 2025 and CMO 2024 and 118/120 on Putnam 2024.

## Key Contributions
- Verifier trained by GRPO with reward R_format · R_score, where R_score = 1 − |s′ − s| against expert scores s ∈ {0, 0.5, 1} (§2.1.1, Eq. 1–2).
- Meta-verifier that scores the verifier's analysis; the verifier's reward becomes R_format · R_score · R_meta (§2.1.2, Eq. 3).
- Generator reward that combines proof score and self-analysis accuracy weighted by the meta-verification score (§2.2.2, Eq. 5–6).
- Automated labeling: n verifier analyses per proof, m meta-verifications per issue-reporting analysis, label = lowest score confirmed by at least k valid analyses (§2.3). It replaced human annotation in the last two training iterations (§2.3).
- Evidence that self-verification guides refinement: Pass@1 on ISL 2024 rises from 0.15 to 0.27 as max sequential iterations go from 1 to 8 (Fig. 2).

## Key Figures/Tables to Study
- Eq. 2, 3, 5, 6: the three reward definitions (verifier, meta-verified verifier, generator with self-analysis).
- Fig. 2: Pass@1 and self-selected Best@32 on ISL 2024 vs max sequential iterations.
- Fig. 3: IMO-ProofBench Basic/Advanced against Gemini Deep Think (IMO Gold) and other baselines.
- Table 1: IMO 2025, CMO 2024, Putnam 2024 problems solved under high-compute search.
- App. A.1–A.4: the generation, verification, meta-verification, and refinement prompts.

## Technical Details
**Data.** D_p: 17,503 proof problems crawled from AoPS contest collections, prioritizing olympiads, team selection tests, and post-2010 problems that require proofs (§2.1.1). Candidate proofs come from a variant of DeepSeek-V3.2-Exp-Thinking prompted to refine over multiple rounds; experts score sampled proofs to form D_v = {(X, Y, s)} (§2.1.1). Sizes of D_v and D_mv: not reported.

**Verifier (§2.1.1).** Initialized from a DeepSeek-V3.2-Exp-SFT version fine-tuned on math and code reasoning data. Output format: a summary of identified issues, then a score in \boxed{} after a fixed key phrase; R_format checks the phrases "Here is my evaluation of the solution:" and "Based on my evaluation, the final overall score should be:".

**Meta-verification (§2.1.2).** Failure mode addressed: on flawed proofs (s < 1) the verifier can predict the correct score while hallucinating issues and still get full reward. Experts score verifier analyses with rubric I_mv to form D_mv = {(X, Y, V, ms)}, ms ∈ {0, 0.5, 1}; a meta-verifier is RL-trained with the same format and score rewards. The enhanced verifier is trained on D_v and D_mv and performs both tasks. On a validation split of D_v, the meta-verifier-rated quality of verifier analyses rose from 0.85 to 0.96 with unchanged score-prediction accuracy.

**Generator (§2.2).** Base objective: maximize E[R_Y], R_Y = verifier score of proof Y (Eq. 4). Observed failure: a generator asked to generate and self-analyze in one shot claims correctness when the external verifier finds flaws. Fix: the generator outputs proof Y plus self-analysis Z in the verifier format; R = R_format(Y, Z) · (α·R_Y + β·R_Z), R_Z = R_score(s′, s) · R_meta(Z), α = 0.76, β = 0.24 (Eq. 5–6).

**Training loop (§3.1).** GRPO. Each iteration first optimizes verification; the generator is initialized from the verifier checkpoint and optimized for generation. From iteration 2, the verifier is initialized from a checkpoint that merges verification and generation abilities from the previous iteration by rejection fine-tuning. Iteration count, learning rate, group size, KL, batch size: not reported.

**Automated labeling (§2.3).** (1) n independent analyses per proof; (2) for analyses with score 0 or 0.5, m meta-verifications, analysis valid by majority; (3) if at least k lowest-score analyses are valid, label = that score; if no valid issue in any attempt, label = 1; else discard or send to experts. Values of n, m, k: not reported. "Quality checks confirmed" agreement with experts; no agreement figure is given.

**Test-time search (§3.3.3).** Pool initialized with 64 proofs × 64 analyses each. Per iteration: take the 64 proofs with highest mean verification score, pair each with 8 analyses (prioritizing scores 0 or 0.5), generate one refined proof per pair; stop after 16 iterations or when a proof passes all 64 verifications. One model performs both generation and verification.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeekMath-V2 | not reported (base: DeepSeek-V3.2-Exp-Base) | RL | proof problem set D_p | 17,503 AoPS proof problems | arXiv:2511.22570v1 §2.1.1 | verified 2026-11-24 | no ablation reported |
| Verifier | same | reward-model | score levels; R_score | {0, 0.5, 1}; 1 − abs(s′ − s) | §2.1.1, Eq. 1 | verified 2026-11-24 | no ablation reported |
| Verifier | same | reward-model | reward composition | R_format · R_score · R_meta | §2.1.2, Eq. 3 | verified 2026-11-24 | meta-rated analysis quality 0.85 → 0.96, score accuracy unchanged (§2.1.2) |
| Generator | same | RL | α, β in Eq. 5 | 0.76, 0.24 | §2.2.2 | verified 2026-11-24 | no ablation reported |
| Verifier + generator | same | RL | algorithm | GRPO, alternating verifier/generator per iteration | §3.1 | verified 2026-11-24 | no ablation reported |
| Verifier + generator | same | RL | LR, batch, group size, KL, iterations | not reported (checked §2–3, App. A, GitHub README) | — | not reported | — |
| Auto-labeling | same | reward-model | n, m, k | not reported | §2.3 | not reported | — |
| Test-time search | same | eval-gate | pool; pairs; max iterations; pass rule | 64 proofs × 64 analyses; 8 analyses per proof; 16; pass all 64 | §3.3.3 | verified 2026-11-24 | — |
| One-shot eval | same | eval-gate | samples; verifier votes | 8 proofs per problem; majority of 8 analyses | §3.3.1 | verified 2026-11-24 | — |

## Findings relevant to generality and negative feedback
- **Negative feedback.** Expert-labeled flawed proofs (s = 0 or 0.5) are training targets for the verifier; the generator is explicitly rewarded for reporting its own errors accurately (R_Z), so an honest negative self-assessment of a flawed proof earns reward that a false claim of correctness does not (§2.2.2).
- **Reward hacking.** The meta-verifier exists because score-only reward lets the verifier cite non-existent issues while predicting the correct score (§2.1.2).
- **Generality.** One-shot evaluation on 91 in-house CNML-level problems covers five categories (algebra 13, geometry 24, number theory 19, combinatorics 24, inequality 11); DeepSeekMath-V2 is above GPT-5-Thinking-High and Gemini 2.5-Pro in every category, as judged by the authors' own final verifier (§3.2, §3.3.1, Fig. 1). The authors state the hardest IMO-level problems remain difficult (§3.3.3).
- **Results.** IMO-ProofBench Basic 99.0 vs 89.0 for Gemini Deep Think (IMO Gold); Advanced 61.9 vs 65.7 (Fig. 3; DeepSeekMath-V2 graded by the authors' experts, baselines from Luong et al.). IMO 2025: P1–P5 solved, 83.3%; CMO 2024: 73.8%; Putnam 2024: 98.3% (118/120) (Table 1).

## Connections
- [[lets-verify]], [[training-verifiers-to-solve-math-word-problems]]: earlier verifier/reward-model work on final-answer math; this paper scores whole proofs with a generative verifier.
- [[deepseek-v3.1]] (card for the DeepSeek-V3.2 report) and [[deepseek-v3-2]]: V3.2-Speciale uses this paper's dataset and reward method (arXiv:2512.02556 §3).
- [[deepseek-prover-v2]]: formal (Lean) counterpart; the related-work section cites it (§4).
- [[generative-reward-models]]: verifier as a generative reward model.

## Verification
- Checked on 2026-11-24 against: arXiv:2511.22570v1 PDF (body, App. A) and https://github.com/deepseek-ai/DeepSeek-Math-V2 README. Fig. 2 and Fig. 3 values were read from the rendered figures.
- Not reported by the source: parameter count of the final model (README: "built on top of DeepSeek-V3.2-Exp-Base"); sizes of D_v and D_mv; number of expert annotators; RL hyperparameters; number of training iterations; n, m, k; expert–auto-label agreement rate.
