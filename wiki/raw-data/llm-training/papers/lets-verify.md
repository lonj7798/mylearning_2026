<!-- scope: process-supervised reward models for MATH and the released PRM800K step-label dataset
     deps: [[training-verifiers-to-solve-math-word-problems]]
     see-also: [[prm800k]], [[let-verify]], [[math-shepherd]], [[omegaprm]], [[deepseek-r1]], [[reward-model-overoptimization]]
-->

# Let's Verify Step by Step
- **Core Insight:** A process-supervised reward model (PRM) fine-tuned from GPT-4 on about 800K human step labels selects a correct solution for 78.2% of the 500 held-out MATH test problems at best-of-1860, against 72.4% for an outcome-supervised reward model (ORM) and 69.6% for majority voting (§3, Fig. 3).
- **Guideline:** When a reward model must rank many sampled solutions to multi-step math problems and human step labels are affordable, label only up to the first incorrect step and surface convincing wrong-answer solutions to labelers, because the PRM is above the ORM at every tested N (§3, Fig. 3) and this selection was estimated at about 2.6× the data efficiency of uniform labeling in the small-scale runs (§4.2). When step labels are not affordable, outcome supervision by a stronger PRM is a better baseline than final-answer checking (§4.1, Fig. 4b).
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, et al. (OpenAI; ten authors, last-listed Karl Cobbe)
- **Year:** 2023 (arXiv v1 2023-05-31; ICLR 2024)
- **URL:** https://arxiv.org/abs/2305.20050 (dataset, labeling instructions and scored samples: https://github.com/openai/prm800k)
- **Source type:** paper
- **Relevant topics:** process supervision, outcome supervision, PRM800K, verifier training, active learning, best-of-N search, MATH

## Abstract
The paper compares outcome supervision, which labels only the final result, with process supervision, which labels each intermediate reasoning step, for training reward models on the MATH dataset. At large scale all models are fine-tuned from the base GPT-4 model; at small scale the base models are pretrained with roughly 200 times less compute and are supervised by the large PRM instead of by humans, which makes controlled ablations affordable. The generator is fixed and is not trained with reinforcement learning, so the comparison isolates verifier quality, measured by best-of-N search. The process-supervised model solves 78.2% of a representative subset of the MATH test set. Active learning improves the data efficiency of process supervision. The authors release PRM800K, the 800,000 step-level human labels used to train the best reward model.

## Key Contributions
- Large-scale result: the PRM is above the ORM and above majority voting at every N tested, and the gap widens as N grows (§3, Fig. 3).
- Small-scale controlled comparison on identical solution sets: process supervision from PRM_large beats both outcome supervision from PRM_large and outcome supervision from final-answer checking at every data-collection scale, evaluated by best-of-500 (§4.1, Fig. 4a).
- Active learning that selects convincing wrong-answer solutions, estimated at about 2.6× the data efficiency of uniform labeling (§4.2).
- Out-of-distribution check on recent AP and AMC exams at best-of-100: PRM 72.9%, ORM 63.8%, majority vote 61.3% (§5, Table 1).
- Release of PRM800K, the labeler instructions, and the held-out 500-problem MATH test split (App. B, App. C).

## Key Figures/Tables to Study
- Fig. 1: the labeling interface, which defines the unit of human feedback.
- Fig. 2: large-scale PRM step scores on a correct and an incorrect solution to the same problem.
- Fig. 3: best-of-N curves for PRM, ORM and majority voting, N up to 1860.
- Fig. 4a/4b: small-scale process vs outcome supervision, with and without active learning.
- Table 1: out-of-distribution STEM results. Table 3: label balance by collection phase. Table 4: the four PRM scoring strategies.

## Technical Details
- **Scope.** One fixed generator per model scale; the generator is not improved with RL, and "outcome" and "process" refer only to the supervision given to the reward model (§2.1).
- **Base models.** All large-scale models are fine-tuned from the base GPT-4 model, which was pretrained only for next-token prediction and not with RLHF. Small-scale base models are similar in design but pretrained with roughly 200× less compute (§2.2).
- **MathMix.** All models receive an additional fine-tuning step on roughly 1.5B math-relevant tokens called MathMix; a 1B-token variant excluding critiques data is used for some experiments (§2.2; App. A).
- **Generator format.** The generator is fine-tuned for one epoch on few-shot-generated MATH training solutions filtered to the correct final answer, so that it emits newline-delimited step-by-step solutions. The paper states this step is meant to fix the format, not to teach new skills (§2.3).
- **Label schema.** Each step is labeled positive, negative or neutral. Neutral means appropriate, reasonable and easily verifiable but not progressing toward the solution; positive means neutral and progressing; all other steps are negative. Labelers see the ground-truth final answer but no reference solution (§2.4; App. D).
- **PRM800K size.** 1,085,590 raw step-level labels over 101,599 solution samples were collected; after discarding quality-control labels and incomplete tasks the training set is about 800,000 step-level labels over 75,000 solutions to 12,000 problems (§2.4; App. B).
- **Test split.** 4,500 MATH test-split problems were moved into the PRM800K training set to avoid overfitting the 7,500 training problems, leaving 500 held-out test problems selected uniformly at random (§2.4; App. C).
- **Active learning at collection time.** Phase 2 is split into 10 generations; each generation samples N solutions per problem, ranks them with the current best PRM, and surfaces the highest-scoring wrong-answer solutions. The PRM is retrained between generations (App. B). Phase 1 is about 5% of PRM800K, roughly 40,000 labels (App. B).
- **Label balance.** Phase 1: 85.1% of labeled solutions end correct, 58.6% of steps correct. Phase 2: 13.2% and 74.1%. Combined: 14.2% and 73.1% (App. B, Table 3).
- **PRM training and scoring.** The PRM predicts the label of each step at the last token of that step, as a single token whose log-likelihood is maximized; one forward pass over the whole solution yields all step predictions. The solution score is the product of per-step probabilities that the step is positive, with neutral counted as positive. Process supervision is given only up to the first incorrect step (§2.6; App. F.2).
- **ORM training.** ORMs follow the token-level verifier setup of Cobbe et al. (2021): uniform samples from the generator at temperature 1.0 with no rebalancing, one epoch, no dropout, no joint language-modeling objective; the final-token score is the solution score (§2.5; App. E).
- **Large-scale training sets are not comparable.** The ORM is trained on 100 uniform samples per problem, an order of magnitude more data than PRM800K and with no overlap; the authors state each is their best attempt at that form of supervision rather than a matched comparison (§3).
- **Small-scale active-learning procedure.** A PRM_selector trained on one sample per problem scores 1000 samples per problem; for each larger reward model, N samples per problem are selected such that 80% are the most convincing wrong-answer samples and 20% are the most convincing remaining samples (§4.2).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Large-scale PRM (GPT-4 base) | not reported | mid-train | MathMix tokens | roughly 1.5B math-relevant tokens | arXiv:2305.20050v1 §2.2, App. A | verified 2026-09-18 | no ablation reported; a 1B variant without critiques data is used for some runs (App. A) |
| Large-scale generator | not reported | SFT | epochs on filtered correct MATH solutions | 1 | §2.3 | verified 2026-09-18 | no ablation reported |
| All PRMs | not reported | reward-model | epochs | 2 | App. F.1 | verified 2026-09-18 | App. F.1: 2 epochs beats 1 on the smaller phase-1 and early phase-2 datasets; benefit diminishes on larger sets |
| All PRMs | not reported | reward-model | learning rate | not reported (App. F.1 says low learning rates were important and that hyperparameters were swept on the first ~10% of PRM800K; no value given) | App. F.1 | not reported (checked body, App. E, App. F, GitHub README) | not applicable |
| All PRMs | not reported | reward-model | training targets | per-step probability of positive/negative/neutral at the step's last token | §2.6, App. F.1 | verified 2026-09-18 | not applicable |
| Large-scale ORM | not reported | reward-model | epochs; sampling temperature; per-problem samples | 1 epoch; temperature 1.0; 100 uniform samples per problem | App. E; §3 | verified 2026-09-18 | App. E: performance not sensitive to most other hyperparameters within a reasonable range |
| PRM / ORM | not reported | eval-gate | solution score reduction | product of per-step positive probabilities, neutral counted as positive | App. F.2, Table 4 | verified 2026-09-18 | App. F.2, Table 4: product with neutral-as-positive is best of four strategies, but differences are described as minor |
| PRM / ORM | not reported | eval-gate | best-of-N budget | up to N = 1860 samples per problem (MATH); N = 100 (OOD STEM) | §3, Fig. 3; §5, Table 1 | verified 2026-09-18 | not applicable |

## Findings relevant to generality and negative feedback
- **Generality.** On 224 held-out STEM questions from recent AP Physics, AP Calculus, AP Chemistry, AMC10 and AMC12 exams, the PRM remains above the ORM and majority voting (§5, Table 1; the Table 1 per-subject counts sum to 234, which conflicts with the 224 stated in §5). The authors state it is unknown how far the results generalize beyond math (§6.2). MATH test contamination cannot be ruled out; string-matching decontamination was applied to MathMix (§6.3).
- **Negative feedback.** Final-answer grading assigns positive labels to solutions that reach the right answer through incorrect reasoning, which is a false-positive source for ORM targets (§2.5, §4). On hard problems most generated solutions contain an error, so an outcome-level negative label carries little information about where the error is; the authors call this the credit-assignment advantage of process supervision (§6.1). Supervising only up to the first mistake is a deliberate choice; the authors state that supervising beyond it would give process supervision an even greater information advantage (§2.6).
- **Distillation.** PRM_large replaces human labelers for the small-scale models, and outcome labels from PRM_large train a better ORM than final-answer checking does (§4.1, Fig. 4b).

## Connections
- [[training-verifiers-to-solve-math-word-problems]]: the ORM baseline follows that paper's token-level verifier setup (§2.5, App. E).
- [[prm800k]]: the other full library card on this same paper; use it for the figure-by-figure breakdown.
- [[let-verify]]: redirect card pointing here.
- [[math-shepherd]], [[omegaprm]]: replace human step labels with automatically estimated ones.
- [[reward-model-overoptimization]]: the large-RM-supervises-small-RM setup is compared with Gao et al. (2022) (§7.2).
- [[deepseek-r1]]: lists PRMs among unsuccessful attempts at large-scale RL and cites this paper.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2305.20050 (v1, 31 May 2023, the only arXiv version).
- Corrections to the previous card version:
  - "**Year:** 2024" with URL https://openreview.net/forum?id=v8L0pN6EOi → arXiv v1 2023-05-31; the canonical URL is the arXiv abs page. The OpenReview id could not be confirmed (the forum page returns a bot-verification interstitial), so it is not carried on the card.
  - "reaches 78.2% on a representative subset of the MATH test set" without the baselines → 78.2% PRM against 72.4% ORM and 69.6% majority voting at best-of-1860 on the 500 held-out MATH test problems (§3, Fig. 3).
  - "process supervision beats outcome supervision even when both are judged by final-answer accuracy" stated as a matched comparison → at large scale the two training sets are not comparable (ORM: 100 uniform samples per problem, an order of magnitude more data); the matched comparison is the small-scale one in §4.1.
  - "Shows that active learning materially improves label efficiency" → estimated at approximately 2.6× more data efficient than uniform labeling, from the slopes of the lines of best fit in the small-scale runs (§4.2).
  - "The training set contains 800K step-level labels across 75K solutions to 12K problems" left the raw totals out → 1,085,590 raw labels over 101,599 samples before filtering (App. B).
  - "Human labelers mark each step as positive, negative, or neutral" without definitions → the three labels are defined in App. D, and labelers see the final answer but no reference solution.
  - "**Main result table:** process-supervised PRM versus outcome-supervised ORM on MATH" → the large-scale comparison is Fig. 3 and the inline table above it; Table 1 is the out-of-distribution STEM result.
- Removed as unsupported by the source: "[[yejin-choi-group]] connects here conceptually because STaR-like self-improvement also depends on reliable intermediate-signal filtering" (no such link in the paper, and the card asserted a conceptual connection with no source); "This paper is best read as the bridge from early verifier work to later PRM/RLVR systems" and "if you only supervise outcomes, you will reward many wrong internal computations that accidentally land on the right answer" as a stated conclusion (the paper's own claim is narrower: final-answer grading produces false positives, §2.5); "[[tulu-3]]" as a connection (not cited by the paper).
- Not reported by the source: parameter counts or exact sizes of the large- and small-scale models; PRM and ORM learning rates and batch sizes; wall-clock or GPU cost; the number of labelers.
