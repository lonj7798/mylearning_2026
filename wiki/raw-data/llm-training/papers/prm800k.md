<!-- scope: process reward model from step-level human labels on MATH solutions
     deps: [[math-shepherd]]
     see-also: [[deepseek-r1]], [[rlvr-tulu3]]
-->

# Let's Verify Step by Step (PRM800K)
- **Core Insight:** Step-level (process) supervision dominates outcome-only supervision for multi-step reasoning — even when the outcome labels are free from the grader, paying for per-step correctness labels unlocks a much higher Best-of-N ceiling.
- **Guideline:** If you can afford it, label correctness at every reasoning step; train the PRM as a step-level classifier over a special "step separator" token; aggregate step scores with `min` (the solution is only as good as its worst step) when selecting a final answer.
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.20050
- **Relevant topics:** process reward models (PRMs), outcome reward models (ORMs), step-level supervision, active learning, MATH benchmark, Best-of-N

## Abstract
Large language models often get the final answer right through an incorrect reasoning chain, or vice versa, so outcome-only reward signals leave performance on the table. The authors collect PRM800K — ~800,000 step-level human annotations over 75,000 GPT-4 solutions to MATH problems — and train a Process Reward Model (PRM) that scores the correctness of each step. Using the PRM as a verifier in a Best-of-N pipeline, the process-supervised model reaches 78.2% on a representative subset of MATH, substantially beating outcome-supervised baselines. Active learning on the hardest / most disagreed-on solutions is shown to be much more data-efficient than uniform labeling.

## Key Contributions
- **PRM800K dataset:** ~800K step-level labels (`correct / incorrect / neutral`) on ~75K GPT-4-generated solutions to the MATH competition dataset.
- **PRM training:** for every step token-separator position, the model predicts a three-way label; loss is cross-entropy only at those positions; solution-level score is obtained by aggregating per-step predictions.
- **Aggregation:** the paper uses the **product** of per-step probabilities of "correct" (equivalently `exp(sum log p_correct)`); `min` over steps is a close competitor and is the form adopted by later work (Math-Shepherd).
- **Outcome vs process head-to-head:** with matched compute and matched number of samples, the PRM Pareto-dominates the ORM on MATH across N from 1 to 1860.
- **78.2% on MATH-500 (representative subset)** via Best-of-1860 with the PRM selector, vs 72.4% for the ORM-selector baseline and 69.6% for majority voting.
- **Active learning:** prioritizing labeling on solutions where the current PRM is uncertain or disagrees with an ORM gives a ~2.6× data efficiency multiplier.

## Key Figures/Tables to Study
- **Fig. 1** (PRM vs ORM Best-of-N curves) — clearest demonstration that PRM advantage grows with N.
- **Fig. 3** (per-step label distribution) — how often steps are labeled "neutral" — shows the labeling schema.
- **Fig. 6** (active learning efficiency) — effective sample complexity reduction.
- **Table 1** (final Best-of-N numbers on MATH) — the headline numbers.

## Technical Details
- **Step separator:** a newline or a literal "Step k:" token; the PRM emits a score at that position using its hidden state.
- **Labels per step:** `+1` correct, `−1` incorrect, `0` neutral (ambiguous or filler). Training loss is on non-neutral steps only.
- **Scoring functions tested:**
  - `prod`: `∏_t p_correct(step_t)` (used in the paper).
  - `min`: `min_t p_correct(step_t)` (used by Math-Shepherd; equivalently dominated by the worst step).
  - `softmax-avg`: softmax-weighted average, smoother.
- **Base model:** GPT-4 generator; PRM is a smaller model fine-tuned from a public GPT-4-family checkpoint.
- **Cost:** step-level labeling is ~10× more expensive per example than outcome labeling — the paper explicitly notes active learning is needed to make PRMs practical.
- **Use modes:** (a) Best-of-N re-ranker, (b) process reward for RL (not done in this paper; later realized in Math-Shepherd).

## Connections
- Direct precursor to **[[math-shepherd]]** (automatic PRM labels via rollout-MC) and to step-level RL in **[[rlvr-tulu3]]** and **[[deepseek-r1]]** training stages.
- The PRM label set is one of the training ingredients for many open reasoning models (Qwen-Math, DeepSeekMath) — dataset is publicly released.
- Complements **[[reward-model-overoptimization]]**: step-level signal reduces the proxy-vs-gold gap because a wrong step is a much harder thing to fake.
