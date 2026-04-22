<!-- scope: process reward models with step-by-step verification (PRM800K)
     deps: [[prm800k]]
     see-also: [[math-shepherd]], [[rlvr-tulu3]], [[deepseek-r1]]
-->

# Let's Verify Step by Step
- **Core Insight:** On math problems, a **Process Reward Model** (PRM) that verifies each reasoning step outperforms an outcome-only Reward Model (ORM) at ranking solutions — and the gap widens with the number of candidates.
- **Guideline:** For verifiable-reasoning tasks, collect step-level labels (not just final-answer correctness) and train a PRM; use it as a ranker or as the reward signal for RL. On MATH, PRM + best-of-N gives a ≥10-pt absolute lift over ORM at N=64.
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- **Year:** 2023 (OpenAI)
- **URL:** https://arxiv.org/abs/2305.20050
- **Relevant topics:** process reward models, verifiable rewards, MATH benchmark, chain-of-thought, best-of-N

## Abstract
In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78.2% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.

## Key Contributions
- Trained and released **PRM800K**: 800K step-level labels (positive / negative / neutral) across ~75K MATH solutions — the canonical process-supervision dataset.
- Showed the process RM outperforms outcome RM at best-of-N selection: 78.2% vs 72.4% on a MATH test subset at N=1860.
- Introduced **active learning** for PRM: route uncertain steps to labelers, yielding ~2.6× label efficiency.
- Provided the "credit assignment = reasoning step" operational definition that now underlies all process-reward work (Math-Shepherd, RLVR, DeepSeek-R1).

## Key Figures/Tables to Study
- **Figure 1 (MATH test-set accuracy vs N):** PRM curve dominates ORM at every N; gap grows as N grows.
- **Figure 3 (calibration):** PRM is better calibrated per-step than ORM on full-solution.
- **Figure 6 (active learning):** active-PRM reaches the same quality as random with 38% of the labels.
- **Table 1 (MATH subset):** 78.2% PRM vs 72.4% ORM vs 69.6% majority-vote — key headline.

## Technical Details
- **Base generator:** GPT-4 (prompted) and a fine-tuned variant; base PRM and ORM are small-scale fine-tunes.
- **Labeling protocol:** labelers see one step at a time, mark ∈{positive, negative, neutral}; first negative step is the failure point.
- **PRM training:** binary classifier per step on (prefix, step) → ∈{good, bad}, cross-entropy.
- **Scoring a full solution:** multiply per-step "good" probabilities → solution score.
- **Dataset stats:** 800K labels, ~75K solutions, 12K problems from MATH training set.
- **Active learning:** rank unlabeled steps by model uncertainty (entropy on the good/bad head); label top quantile.
- **Inference cost:** PRM scores every step (1 forward per step); ORM scores final answer (1 forward per solution) — PRM is ~L× more expensive at inference where L is step count.

## Connections
- Direct precursor to [[math-shepherd]] (automatic PRM labeling via Monte-Carlo rollouts) and [[rlvr-tulu3]] (verifiable rewards replacing PRM on problems with programmatic checkers).
- The "ORM vs PRM at high N" result motivates **Best-of-N selection** as a deployment strategy — see [[best-of-n]].
- DeepSeek-R1 ([[deepseek-r1]]) abandoned PRMs in favor of pure outcome RL + group-relative advantages (GRPO); Let-Verify is the paper they ablate against.
- The PRM800K dataset remains the reference benchmark — any new process-reward method reports numbers on it.
- Links to chain-of-thought reasoning — step-level labels presume a step-structured CoT; the paper's label schema is what makes reasoning auditable.
