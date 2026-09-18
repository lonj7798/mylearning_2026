<!-- scope: inference-time reasoning — MCTS over a five-action reasoning space plus a second SLM used as a mutual-consistency discriminator
     deps: [[lets-verify]]
     see-also: [[rstar-math]], [[math-shepherd]], [[omegaprm]]
-->

# Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers
- **Core Insight:** A 7B-class model can raise its own GSM8K accuracy from 12.51% to 63.91% (LLaMA2-7B) at inference time by running 32 MCTS rollouts over a five-action reasoning space and keeping only trajectories that a second small model, given a random prefix of the trajectory, completes to the same final answer (Abstract; §4.2 Table 2).
- **Guideline:** When no stronger teacher model and no trained reward model are available, use MCTS with several distinct reasoning actions for generation and a second small model's prefix-completion agreement for answer selection, because in this paper substituting either component alone (RAP's single action type; self-verification) gives lower accuracy (§4.3 Tables 4 and 5). This is an inference-time procedure; the paper does not fine-tune on the produced trajectories.
- **Authors:** Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang (Microsoft Research Asia; Qi and Ma at Harvard University, work done during an MSRA internship)
- **Year:** 2024 (arXiv v1 2024-08)
- **URL:** https://arxiv.org/abs/2408.06195
- **Source type:** paper
- **Relevant topics:** reasoning, MCTS, small-model reasoning, unsupervised verification, test-time scaling

## Abstract
rStar is a self-play mutual reasoning procedure that improves the reasoning accuracy of small language models without fine-tuning and without a superior model. Reasoning is split into generation and discrimination. A target small model augments MCTS with a set of five human-like reasoning actions to construct candidate reasoning trajectories. A second small model of similar capability then acts as a discriminator: it is given the earlier part of a candidate trajectory and asked to complete the remaining steps; trajectories whose completed answer matches the original are called mutually consistent and are treated as more likely correct. Experiments cover five small models and five reasoning tasks (GSM8K, GSM-Hard, MATH, SVAMP, StrategyQA).

## Key Contributions
- A five-action reasoning space for MCTS node expansion, replacing the single action type used by RAP and ToT (§3.2).
- Mutual reasoning consistency: unsupervised trajectory verification by a second small model completing a randomly truncated prefix, in place of a trained reward model or self-verification (§3.3).
- A reward function for small models that avoids self-rewarding of intermediate nodes and uses back-propagated terminal correctness signal instead (§3.2).
- Accuracy gains on five small models across five reasoning tasks without any fine-tuning (§4.2 Tables 2 and 3).
- Ablations isolating the contribution of the action space, the generator, and the discriminator (§4.3 Tables 1, 4, 5).

## Key Figures/Tables to Study
- **Figure 2** — the generation/discrimination loop, showing where the second model enters.
- **Table 1** — action-space ablation on 200 LLaMA3-8B examples.
- **Table 2** — main results on GSM8K, GSM-Hard, SVAMP, StrategyQA for five models.
- **Table 3** — MATH-500 results, reported only for LLaMA3-8B-Instruct and Phi3-mini-4k.
- **Table 5** — discriminator ablation, including the effect of swapping the discriminator model.
- **Table 7** — inference cost per GSM8K question.

## Technical Details
- **Action space (five actions)** (§3.2): A1 propose a one-step thought; A2 propose the remaining thought steps (standard CoT, "fast thinking"); A3 propose the next sub-question together with its answer; A4 answer the sub-question again using few-shot CoT; A5 rephrase the question or sub-question. Ordering constraints: A4 can occur only after A3, and A5 only after the root question (§3.2).
- **Action-space ablation** on 200 LLaMA3-8B examples (§3.2, Table 1): A3 alone (equivalent to RAP) 70.5; A3+A5 72.5; A3+A4+A5 73.5; A2+A3+A4+A5 74.0; all five 75.0.
- **Reward** (§3.2): unexplored nodes start at Q(s_i, a_i) = 0. On reaching a terminal node, Q(s_d, a_d) is computed from the self-consistency majority-voting confidence of the final answer and back-propagated by Q(s_i, a_i) ← Q(s_i, a_i) + Q(s_d, a_d). Node selection uses UCT with an exploration constant c (§3.2, Eq. 1).
- **Discrimination** (§3.3): for a trajectory t = x ⊕ s_1 ⊕ … ⊕ s_d, steps from a randomly sampled index i are masked; the prefix x ⊕ s_1 ⊕ … ⊕ s_{i−1} is given to the second model, which completes the rest. The trajectory is kept if the completed answer matches the original answer. In the implementation, the split point is sampled between 20% and 80% of the steps (§4.1).
- **Final selection** (§3.3): among validated trajectories, the score is the trajectory reward multiplied by the terminal node's rollout confidence; the highest-scoring trajectory is returned.
- **Models evaluated** (§4.1): Phi3-mini (3.8B), LLaMA2-7B, Mistral-7B, LLaMA3-8B, LLaMA3-8B-Instruct.
- **Search settings** (§4.1): 32 MCTS rollouts; tree depth d = 8 on MATH and d = 5 elsewhere; A1 and A3 expand at most 5 nodes per depth, other actions 1. Phi3-mini-4k (3.8B) is the discriminator for all targets; when Phi3 is the target it self-discriminates.
- **Main results, GSM8K** (§4.2, Table 2), few-shot CoT → rStar: LLaMA2-7B 12.51 → 63.91; Mistral-7B 36.46 → 81.88; LLaMA3-8B 47.23 → 85.52; LLaMA3-8B-Instruct 74.53 → 91.13; Phi3-mini-4k 83.45 → 90.67.
- **Other tasks** (§4.2, Table 2): GSM-Hard LLaMA2-7B 3.71 → 18.57, Mistral-7B 13.57 → 37.91; SVAMP LLaMA2-7B 48.10 → 74.90; StrategyQA LLaMA2-7B 58.82 → 67.25.
- **MATH-500** (§4.2, Table 3): LLaMA3-8B-Instruct 17.80 → 42.94; Phi3-mini-4k 32.20 → 48.60. The paper evaluates MATH-500 only on these two instruction-tuned models, citing LaTeX-heavy instruction following.
- **Discriminator choice** (§4.3, Table 5 right), LLaMA3-8B-Instruct generator on GSM8K: majority voting 88.70; LLaMA3-8B-Instruct as its own discriminator 88.78; LLaMA3.1-8B-Instruct 89.52; Phi3-Mini-Instruct (the default) 91.13; GPT-4 (2024-05-01) 92.57. The paper states the choice "generally does not affect" the result and that GPT-4 "only slightly improves performance (91.13% to 92.57%)".
- **Discriminator vs. alternatives** (§4.3, Table 5 left), GSM8K: with the rStar generator on LLaMA3-8B, majority voting 74.38, self-verification 75.52, mutual consistency 85.52; on LLaMA3-8B-Instruct, 88.70 / 86.63 / 91.13.
- **Inference cost** (App. A.1, Table 7): per GSM8K question at 32 rollouts, 166.81 model calls and 367.1k generated tokens for LLaMA2-7B; 148.90 calls and 348.6k tokens for Mistral. Completing 32 rollouts over the whole GSM8K test set takes about 4.5 days on a single A100 per model.
- **Baseline for the exploration claim** (§1): after 32 rounds of RAP self-exploration with LLaMA2-7B on GSM8K, 24% of generated trajectories are correct.

## Findings relevant to generality
- The action-space ablation is the only generality evidence in the paper: each added action raises accuracy on the 200-example LLaMA3-8B subset, with the full set best at 75.0 (§3.2, Table 1). The action set is hand-designed for question-answering-style reasoning; the paper's five tasks are four math tasks plus StrategyQA (§4.1).
- The discriminator's effectiveness does not depend on using a stronger model: a 3.8B discriminator recovers most of the benefit, and GPT-4 adds 1.44 points on LLaMA3-8B-Instruct (§4.3).
- The paper reports that self-evaluation and self-verification by the generator itself are ineffective for small models, and that naive reward-based self-exploration guidance performs no better than random guesses (§4.3; App. A.1).

## Connections
- [[rstar-math]] — same group; replaces mutual consistency with a trained process preference model and adds training rounds.
- [[math-shepherd]], [[omegaprm]], [[lets-verify]] — process reward models, the trained-verifier alternative rStar avoids.
- [[star]], [[quiet-star]] — self-generated reasoning traces used for training rather than inference-time search.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2408.06195 (arXiv v1, 12 Aug 2024).
- Corrections to the previous card version:
  - Title "rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers" → the published title is "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers" (title page).
  - "GSM8K from 12% → 63% … 37% → 82%" → 12.51% → 63.91% and 36.46% → 81.88% (Abstract; Table 2).
  - "MATH: Mistral-7B 10.2 → 25.4" → not reported; MATH-500 is evaluated only on LLaMA3-8B-Instruct (17.80 → 42.94) and Phi3-mini-4k (32.20 → 48.60) (Table 3).
  - Benchmark list "GSM8K, MATH, SVAMP, ASDiv, MultiArith" → GSM8K, GSM-Hard, MATH, SVAMP, StrategyQA (§4.1).
  - Action definitions A1–A5 were mismatched ("one-step CoT, subquestion decomposition, rephrasing, direct-answer with verification, propose-a-new-subquestion") → corrected to the paper's A1–A5 (§3.2).
  - "mask the second half of the reasoning" and "same family but different prompting" → the mask point is sampled between 20% and 80% of steps, and the discriminator is Phi3-mini-4k for all target models (§3.3; §4.1).
  - "reward 0/1 based on self-consistency of its final answer" → the terminal reward is the self-consistency majority-voting confidence, back-propagated additively (§3.2).
  - "~1 minute on 1 A100" per problem → 166.81 calls and 367.1k tokens per GSM8K question; about 4.5 days on one A100 for the full GSM8K test set (Table 7).
  - "~30× the cost of greedy CoT" → not reported as a ratio; the absolute costs above are what the paper gives.
- Removed as unsupported by the source: the framing of rStar as a synthetic-training-data generator ("generating high-quality synthetic traces"; "the verified best trajectory is the synthetic trace"); "Length 300–2000 tokens depending on action mix"; "tree-structured, decomposition-heavy" trace-style claim; "Beats prior prompting methods (CoT-SC, ToT, Plan-and-Solve)" (Plan-and-Solve is not a baseline); "two models can share a systematic bias and consistently agree on a wrong answer" (not analyzed in the paper); "generalization to non-math domains may require redesigning A1–A5" (course inference, not a paper claim).
- Not reported by the source: any fine-tuning of a model on rStar trajectories; false-positive rate of mutual consistency; per-task token budgets outside GSM8K.
