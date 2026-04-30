<!-- scope: process-supervised reward models and the PRM800K dataset
     deps: [[training-verifiers-to-solve-math-word-problems]]
     see-also: [[math-shepherd]], [[deepseek-r1]], [[tulu-3]]
-->

# Let's Verify Step by Step
- **Core Insight:** Step-level human feedback produces a much more reliable verifier of reasoning than outcome-only labels; for hard math, process supervision beats outcome supervision even when both are judged by final-answer accuracy.
- **Guideline:** When you care about faithful reasoning and not just the final answer, collect step labels on the most convincing wrong solutions, train a process reward model, and use it first for best-of-N search before worrying about full RL.
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- **Year:** 2024
- **URL:** https://openreview.net/forum?id=v8L0pN6EOi
- **Relevant topics:** process supervision, PRM800K, verifier training, active learning, best-of-N reasoning

## Abstract
The paper compares outcome supervision and process supervision for reasoning models on the MATH dataset. Instead of only scoring whether a final answer is correct, the authors collect human labels for individual reasoning steps and train a process reward model (PRM). The resulting PRM substantially outperforms outcome-supervised reward models for selecting correct solutions, reaches 78.2% on a representative subset of the MATH test set, and comes with the released PRM800K dataset of 800K step-level labels.

## Key Contributions
- Establishes **process supervision** as a stronger verifier-training signal than outcome supervision on hard reasoning tasks.
- Releases **PRM800K**, a large public dataset of step-level labels for mathematical reasoning.
- Shows that **active learning** materially improves label efficiency.
- Demonstrates that reward-model reliability can be studied separately from RL by using best-of-N search over a fixed generator.

## Key Figures/Tables to Study
- **Figure 1:** the labeling interface; it makes the human-feedback unit concrete.
- **Figure 2:** PRM highlighting the exact bad step in an incorrect solution.
- **Main result table:** process-supervised PRM versus outcome-supervised ORM on MATH.
- **Ablation section on active learning:** important for real-world data-collection budgets.

## Technical Details

### Setup
- The paper intentionally fixes the generator and does **not** run RL; it studies verifier quality in isolation.
- Large-scale models are finetuned from a base GPT-4 model, while smaller models are used for cleaner ablations.
- The generator is taught to produce **newline-delimited step-by-step solutions** so each step can be labeled cleanly.

### PRM800K data
- The training set contains **800K step-level labels across 75K solutions to 12K problems**.
- Human labelers mark each step as **positive, negative, or neutral**.
- Data collection preferentially surfaces **convincing wrong-answer solutions**: responses the current PRM rates highly even though the final answer is wrong.

### Model and evaluation
- The PRM predicts the correctness of each step after the last token of the step.
- At test time, the PRM scores full solutions and is used for **best-of-N search** over sampled solutions.
- The main metric is how often the selected solution is actually correct after automatic grading.

### Practical message
- This paper is best read as the bridge from early verifier work to later PRM/RLVR systems.
- It argues that if you only supervise outcomes, you will reward many wrong internal computations that accidentally land on the right answer.

## Connections
- [[training-verifiers-to-solve-math-word-problems]] is the precursor: verifier ranking with outcome labels only.
- [[deepseek-r1]] and [[tulu-3]] push toward RL with verifiable rewards; Let's Verify strengthens the verifier side.
- [[reward-model-overoptimization]] explains why a better reward signal is valuable but still not the whole story.
- [[yejin-choi-group]] connects here conceptually because STaR-like self-improvement also depends on reliable intermediate-signal filtering.
