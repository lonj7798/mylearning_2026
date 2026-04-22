<!-- scope: outcome-based verifier ranking and the GSM8K dataset
     deps: [[ppo]]
     see-also: [[lets-verify]], [[reward-model-overoptimization]], [[best-of-n]]
-->

# Training Verifiers to Solve Math Word Problems
- **Core Insight:** For verifiable reasoning tasks, generation quality is not enough; sampling many solutions and ranking them with a separate verifier scales better than plain finetuning.
- **Guideline:** Train a separate verifier whenever answers are checkable: keep the generator high-coverage, sample many candidates at higher temperature, and use verifier ranking as cheap test-time compute before moving to RL.
- **Authors:** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
- **Year:** 2021
- **URL:** https://arxiv.org/abs/2110.14168
- **Relevant topics:** verifiers, best-of-N, GSM8K, sample-and-rank, outcome supervision

## Abstract
The paper introduces GSM8K, a diverse 8.5K-problem dataset for grade-school mathematical reasoning, and studies verification as an alternative to direct finetuning. A separate verifier judges whether sampled candidate solutions are correct, and the system outputs the highest-ranked candidate. The paper shows that this approach significantly improves performance and appears to scale better with data than plain finetuning.

## Key Contributions
- Releases **GSM8K**, which became the standard open benchmark for early LLM math reasoning.
- Establishes the **sample-and-rank** recipe for reasoning: generator + verifier instead of generator-only.
- Shows verification can deliver a gain comparable to a very large model-size increase.
- Finds that **dropout** is a surprisingly strong regularizer for both finetuning and verification.

## Key Figures/Tables to Study
- **Figure 4:** the verification training pipeline; still the clearest picture of the recipe.
- **Scaling plots for finetuning vs verification:** this is the main practical reason the paper mattered.
- **Coverage discussion around test@100:** explains why the generator must not become too overconfident.

## Technical Details

### Dataset
- GSM8K contains **8.5K** high-quality grade-school math problems.
- Problems are designed for **high linguistic diversity** and **moderate difficulty**, with natural-language solutions rather than just equations.

### Verification pipeline
1. Finetune a generator on the training set for **2 epochs**.
2. Sample **100 completions per training problem**.
3. Label each completion as correct or incorrect using the final answer.
4. Train a verifier for **1 epoch** on these labeled solutions.
5. At test time, sample multiple candidates and return the one with the highest verifier score.

### Important implementation detail
- The paper stresses **coverage**: a generator that is too overfit gives poor sample diversity and hurts verification.
- They also train the model to use **calculator annotations**, reducing arithmetic mistakes inside sampled solutions.

### Why this paper still matters
- It is the clean precursor to PRMs, best-of-N reasoning, rejection-sampling finetuning, and RLVR.
- It showed early that **extra test-time samples plus a good scorer** can beat more brute-force generator scaling.

## Connections
- [[lets-verify]] upgrades verifier training from outcome labels to step-level process supervision.
- [[reward-model-overoptimization]] is the cautionary follow-up: once you optimize against a scorer, its failures matter.
- [[deepseek-r1]] and [[tulu-3]] inherit the same "verifiable tasks + extra compute + learned selection signal" philosophy.
- [[best-of-n]] is the direct downstream systems pattern this paper helped normalize.
