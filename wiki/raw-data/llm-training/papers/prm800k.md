<!-- scope: OpenAI comparison of process vs outcome supervision for math reward models, and the released PRM800K step-label dataset
     deps: [[training-verifiers-to-solve-math-word-problems]]
     see-also: [[math-shepherd]], [[omegaprm]], [[reward-model-overoptimization]], [[deepseek-r1]], [[let-verify]], [[lets-verify]]
-->

# Let's Verify Step by Step
- **Core Insight:** A process-supervised reward model (PRM) fine-tuned from GPT-4 on about 800K human step labels selects a correct solution for 78.2% of 500 held-out MATH test problems at best-of-1860, compared with 72.4% for an outcome-supervised reward model (ORM) and 69.6% for majority voting (§3, Fig. 3).
- **Guideline:** When a reward model ranks many sampled solutions to multi-step math problems and human step labels are affordable, train a PRM on labels collected up to the first incorrect step and prioritize labeling convincing wrong-answer solutions, because the PRM beat the ORM at every tested N (§3) and this selection was about 2.6× more data efficient than uniform labeling in the small-scale runs (§4.2, Fig. 4a).
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, et al. (OpenAI; corresponding author Karl Cobbe)
- **Year:** 2023 (arXiv v1 2023-05; no later arXiv version)
- **URL:** https://arxiv.org/abs/2305.20050 (dataset and scored samples: https://github.com/openai/prm800k)
- **Source type:** paper
- **Relevant topics:** process reward models, outcome reward models, step-level human labels, active learning, best-of-N search, MATH

## Abstract
The paper compares two ways to supervise a reward model for multi-step math reasoning. Outcome supervision gives one label for the final result; process supervision gives a label for each intermediate step. At large scale, all models are fine-tuned from base GPT-4. At small scale, models pretrained with about 200 times less compute are supervised by the large PRM instead of by humans, which allows controlled ablations. The process-supervised model solves 78% of a representative subset of the MATH test set with best-of-N search. Active learning improves the data efficiency of process supervision. The authors release PRM800K, the 800,000 step-level human labels used to train their best reward model.

## Key Contributions
- Large scale: the PRM outperforms the ORM and majority voting for all N, and the gap widens as N increases (§3, Fig. 3).
- Small scale, identical solution sets: process supervision from PRM_large beats outcome supervision from PRM_large and from final-answer checking at every data size (§4.1, Fig. 4a).
- Active learning that surfaces convincing wrong-answer solutions is estimated to be about 2.6× more data efficient than uniform labeling (§4.2).
- Held-out STEM check (AP Calculus, AP Chemistry, AP Physics, AMC10/12; best-of-100): PRM 72.9%, ORM 63.8%, majority vote 61.3% (§5, Table 1).
- Release of PRM800K, the labeling instructions, and the large-scale scored samples (App. B; GitHub README).

## Key Figures/Tables to Study
- Fig. 3: best-of-N curves for PRM, ORM, and majority voting, N up to 1860.
- Fig. 4a/4b: small-scale process vs outcome supervision, with and without active learning.
- Table 1: out-of-distribution STEM results. Table 3: label balance by collection phase.
- Table 4: four PRM scoring strategies. Fig. 6 (App. G): PRM vs ORM by difficulty quintile.

## Technical Details
- **Generator.** The base model is fine-tuned for 1 epoch on few-shot-generated MATH training solutions that reach the correct answer, only to teach a newline-delimited step format (§2.3). The generator is not trained with RL (§2.1).
- **MathMix.** About 1.5B math tokens used as an extra pretraining step (§2.2; App. A, Table 2).
- **Labels.** Each step is labeled positive, negative, or neutral (§2.4). The released files encode ratings as +1, −1, 0 (GitHub README). In phase 2, labeling of a solution stops at the first negative step (App. D).
- **Dataset size.** 800K step labels across 75K solutions to 12K problems (§2.4). The unfiltered collection has 1,085,590 labels over 101,599 solutions; quality-control labels and incomplete tasks are removed for training (App. B). Phase 1 is about 5% of PRM800K, about 40,000 labels (App. B).
- **Label balance.** Combined over both phases, 14.2% of labeled solutions end in a correct answer and 73.1% of labeled steps are correct (App. B, Table 3).
- **Split.** 4.5K MATH test problems are added to training; evaluation uses the remaining 500 problems, selected uniformly at random (§2.4; App. C).
- **PRM objective.** The PRM predicts one token after the last token of each step and is trained by maximizing the log-likelihood of those label tokens (§2.6). It predicts probabilities for positive, negative, and neutral (App. F.1).
- **Solution score.** Product over steps of P(positive), with neutral counted as positive (§2.6; App. F.2). The product has a slight bias against solutions with more steps (App. F.2).
- **ORM.** Trained on uniformly sampled solutions to predict final-answer correctness at every token; the final-token prediction is the solution score (§2.5; App. E).
- **Data collection loop.** Phase 2 has 10 generations. In each, the current best PRM ranks N solutions per problem, the highest-scoring wrong-answer solutions go to labelers, and the PRM is retrained on all data so far (App. B).
- **Labeler quality control.** Screening on 30 questions with at least 75% agreement with gold labels; 10–20 quality-control problems per generation (App. B).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Large-scale generator (GPT-4 base + MathMix) | not reported | SFT | format data; epochs | few-shot MATH train solutions filtered to correct answer; 1 epoch | arXiv:2305.20050v1 §2.3 | verified 2026-09-14 | no ablation reported |
| Large-scale models (GPT-4 base) | not reported | mid-train | MathMix tokens seen | roughly 3B tokens (2 epochs) of the ~1.5B-token mix | arXiv:2305.20050v1 App. A | verified 2026-09-14 | no ablation reported |
| Small-scale models (~200× less pretraining compute) | not reported | mid-train | MathMix variant; epochs | 1B tokens without critiques data; 6 epochs (roughly 6.6B tokens) | arXiv:2305.20050v1 App. A | verified 2026-09-14 | no ablation reported |
| Large-scale PRM | not reported | reward-model | training data | PRM800K: 800K step labels, 75K solutions, 12K problems | arXiv:2305.20050v1 §2.4, App. B | verified 2026-09-14 | §3 Fig. 3: PRM > ORM at all N |
| All PRMs | n/a | reward-model | epochs | 2 | arXiv:2305.20050v1 App. F.1 | verified 2026-09-14 | App. F.1: 2 epochs > 1 epoch on smaller datasets; benefit diminishes on larger ones (no table) |
| All PRMs | n/a | reward-model | learning rate; batch size; optimizer | not reported (App. F.1 says low learning rates were important; sweep on first ~10% of PRM800K) | checked §2.6, App. F.1 | not reported | none |
| All PRMs | n/a | reward-model | supervision extent | up to the first incorrect step | arXiv:2305.20050v1 §2.6 | verified 2026-09-14 | no ablation reported |
| Large-scale PRM | not reported | eval-gate | solution score | product of step P(positive); neutral = positive | arXiv:2305.20050v1 App. F.2, Table 4 | verified 2026-09-14 | Table 4 (best-of-1860): 78.2% vs 77.6% (min), 77.4% (product, neutral = negative), 77.8% (min, neutral = negative) |
| Large-scale ORM | not reported | reward-model | training samples | 100 uniform samples per problem; temperature 1.0; no rebalancing | arXiv:2305.20050v1 §3, App. E | verified 2026-09-14 | §3: mixing uniform samples into PRM800K solutions did not improve the ORM (no table) |
| All ORMs | n/a | reward-model | epochs; dropout; LM objective | 1 epoch; no dropout; no joint LM objective | arXiv:2305.20050v1 App. E | verified 2026-09-14 | App. E: not sensitive to most other hyperparameters (no table) |
| Small-scale PRMs | not reported | reward-model | active-learning selection | PRM_selector (1 sample/problem) scores 1000 samples/problem; pick 80% most convincing wrong-answer + 20% most convincing remaining | arXiv:2305.20050v1 §4.2 | verified 2026-09-14 | Fig. 4a: ~2.6× data efficiency vs uniform |
| Small-scale RMs | not reported | reward-model | synthetic step label | step incorrect if PRM_large P(negative) > 20% | arXiv:2305.20050v1 App. H | verified 2026-09-14 | no ablation reported |
| Large-scale PRM, ORM | not reported | eval-gate | best-of-N protocol | up to 1860 samples per problem on 500 MATH test problems; samples without an answer within 1024 tokens discarded | §3 Fig. 3; App. C; github.com/openai/prm800k@7ecc794 README | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality, negative feedback, distillation
- **Generality.** The PRM advantage holds on recent AP and AMC exam problems released after the pretraining dataset was compiled (§5, Table 1); §5 says 224 questions, while the Table 1 problem counts sum to 234. The authors state it is unknown how far the results generalize beyond math (§6.2). MATH contamination cannot be ruled out, but the authors expect it to affect all methods similarly (§6.3).
- **Negative feedback.** Final-answer grading gives positive labels to solutions with wrong reasoning, which is a false-positive source for ORM targets (§2.5, §4). On hard problems most solutions contain an error, so an outcome-level negative label carries little information about where the error is (§6.1). On the easiest quintile, ORM accuracy decreases slightly as N grows, while the PRM does not (App. G).
- **Distillation.** PRM_large labels replace human labels for small models; outcome labels from PRM_large train a better ORM than final-answer checking (§4.1, Fig. 4b).

## Connections
- [[training-verifiers-to-solve-math-word-problems]]: ORMs follow that paper's token-level verifier setup (§2.5; App. E).
- [[reward-model-overoptimization]]: the large-RM-supervises-small-RM setup is compared to Gao et al. (2022) (§7.2).
- [[math-shepherd]], [[omegaprm]]: replace human step labels with automatically estimated labels; Math-Shepherd scores solutions by the minimum step score, while this paper's default is the product (App. F.2).
- [[deepseek-r1]]: lists PRMs among unsuccessful attempts for large-scale RL and cites this paper (arXiv:2501.12948v1 §4.2).
- [[let-verify]], [[lets-verify]]: other library cards on the same paper.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2305.20050 (v1, the only version) and github.com/openai/prm800k README at commit 7ecc794.
- Corrections to the previous card version:
  - "aggregate step scores with `min`" (Guideline) → the paper uses the product; min is 0.6 points lower with neutral = positive (App. F.2, Table 4).
  - "active learning on solutions where the PRM is uncertain or disagrees with an ORM" → selection of convincing wrong-answer solutions (§2.4, §4.2).
  - "PRM Pareto-dominates the ORM with matched compute and matched number of samples" → the large-scale training sets are not comparable (ORM: 100 uniform samples per problem; PRM: PRM800K); PRM is higher at all N (§3).
  - "78.2% on MATH-500" → 500 held-out MATH test problems; the paper does not use the name MATH-500 (§2.4, App. C).
  - "Fig. 1 = PRM vs ORM curves; Fig. 3 = label distribution; Fig. 6 = active learning; Table 1 = final MATH numbers" → Fig. 1 is the labeling interface; Fig. 3 is best-of-N curves; Fig. 4a is active learning; Fig. 6 is the difficulty breakdown; Table 1 is OOD STEM; label balance is Table 3.
  - "Training loss is on non-neutral steps only" → the PRM predicts positive, negative, and neutral (App. F.1).
  - "PRM is a smaller model fine-tuned from a public GPT-4-family checkpoint" → large-scale PRM is fine-tuned from base GPT-4 (not public) after MathMix; small-scale models use ~200× less pretraining compute (§2.2).
  - "Step separator: newline or literal 'Step k:' token" → newline-delimited steps; prediction after the last token of each step (§2.3, §2.6).
  - "dep: [[math-shepherd]]" → Math-Shepherd is later work; moved to see-also.
- Removed as unsupported by the source: "softmax-avg scoring tested"; "step labeling ~10× more expensive than outcome labeling"; "process reward for RL later realized in Math-Shepherd"; "PRM label set used to train Qwen-Math and DeepSeekMath"; "step-level signal reduces the proxy-vs-gold gap"; step-level RL link to Tülu 3 RLVR and DeepSeek-R1 training stages.
- Not reported by the source: PRM and ORM learning rates, batch sizes, optimizer settings; GPT-4 parameter count; labeling cost.
