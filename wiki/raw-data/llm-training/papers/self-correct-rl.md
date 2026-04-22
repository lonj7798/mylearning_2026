<!-- scope: SCoRe — multi-turn RL to teach LMs to self-correct
     deps: [[grpo]], [[ppo]]
     see-also: [[deepseek-r1]], [[let-verify]]
-->

# SCoRe — Training Language Models to Self-Correct via Reinforcement Learning
- **Core Insight:** LLMs fail at self-correction not because they can't identify mistakes but because standard SFT on "correction trajectories" collapses to producing correct answers immediately and ignoring the self-correction step; a *two-turn* RL formulation — where the reward rewards improvement *between* turns — fixes this.
- **Guideline:** Don't SFT on self-correction traces. Instead use an explicit two-turn RL setup where turn 1 is the first answer, turn 2 conditions on turn 1 + a "try again" prompt, and the reward rewards `r(turn_2) − r(turn_1)` (the improvement delta) plus a regularizer that prevents mode-collapse onto "same answer repeated".
- **Authors:** Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, JD Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M. Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, Aleksandra Faust
- **Year:** 2024 (Google DeepMind)
- **URL:** https://arxiv.org/abs/2409.12917
- **Relevant topics:** self-correction, multi-turn RL, reward shaping, reasoning

## Abstract
Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs. Existing approaches for training self-correction either require multiple models or rely on a more capable model or other forms of supervision. To this end, we develop a multi-turn online reinforcement learning (RL) approach, SCoRe, that significantly improves an LLM's self-correction ability using entirely self-generated data. To build SCoRe, we first show that variants of supervised fine-tuning (SFT) on offline model-generated correction traces are often insufficient for instilling self-correction behavior. In particular, we observe that training via SFT falls prey to either a distribution mismatch between the training data and the model's own responses or implicit preferences for certain modes of self-correction behavior. SCoRe addresses these challenges by training under the model's own distribution of self-generated correction traces and using appropriate regularization to steer the learning process into learning a self-correction strategy that is effective at test time as opposed to simply fitting high-reward responses for a given prompt.

## Key Contributions
- Diagnosed two failure modes of SFT-on-self-correction: (1) distribution shift — SFT data drawn from a stronger teacher; (2) mode collapse — the model learns to produce the correct answer in turn 1 and no-op in turn 2.
- Introduced **SCoRe**, a two-stage online RL recipe:
  - **Stage I:** RL on turn-2 only, with a heavy KL regularization to the base model on turn-1 (keeps turn-1 behavior fixed while learning to edit).
  - **Stage II:** joint RL over both turns with a reward-shaping bonus on the improvement delta `r(y_2) − r(y_1)`.
- Achieves 15.6 pts of self-correction accuracy gain on MATH with Gemini 1.0 Pro and 9.1 pts on MBPP — the first method to cross zero (models historically *got worse* on self-correction).
- Establishes the on-policy requirement: off-policy / offline methods systematically fail the self-correction task.

## Key Figures/Tables to Study
- **Figure 2 (self-correction Δ over training):** shows SFT flatlines while SCoRe monotonically climbs.
- **Figure 5 (ablation of Stage I vs direct Stage II):** Stage I is essential — skipping it causes mode collapse.
- **Table 2 (MATH, MBPP, HumanEval):** SCoRe gains vs Self-Refine / reflexion / STaR.
- **Figure 4 (reward-shaping bonus coefficient):** shows the `Δr` term's sweet spot.

## Technical Details
- **Base model:** Gemini 1.0 Pro; also reproduced on Gemma-2-9B.
- **Turn structure:**
  - Turn 1 prompt = question.
  - Turn 2 prompt = question + turn-1 response + "There might be an error in the above. Please revise."
- **Reward:** correctness (0/1 from ground-truth checker).
- **Stage I loss:** RL on turn 2 only, `∇ log π(y_2 | x, y_1) · r(y_2)` with `KL(π || π_ref)` only on turn 1 (freezes initial attempt).
- **Stage II loss:** joint REINFORCE over both turns, `[r(y_1) + α·(r(y_2) − r(y_1))] · Σ ∇ log π(y_t | ...)` with α=2.0.
- **Optimizer:** AdamW, lr 1e-6, batch 256.
- **Training set:** MATH training split + self-generated traces.

## Connections
- Related to [[star]] / [[rest-em]] (bootstrapping rationales) — those bootstrap *any* correct solution; SCoRe bootstraps *correction* specifically.
- Uses the same clipped REINFORCE / GRPO style update as [[deepseek-r1]] — but with a multi-turn action space.
- The Stage-I distinction (freeze turn 1, only update turn 2) is analogous to [[ipo]]'s "regularize toward the chosen" idea.
- Complements [[let-verify]]: Let-Verify produces a step-level verifier; SCoRe trains the policy to *use* such a verifier's signal to edit its own answer.
