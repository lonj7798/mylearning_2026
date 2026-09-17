---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rstar-math.md
source_url: https://arxiv.org/abs/2501.04519
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: rStar-Math — code-augmented search data, Q-value annotation, and the process preference model

**Checked on 2026-09-15 against arXiv:2501.04519v1 (full PDF including App. A).** The library card `papers/rstar-math.md` had no verification section at that date and contains the errors listed at the end; this excerpt is the checked extract used by ch-24 §5 and the Recipe.

## Data generation (§3.2, App. A.1)

- Each step is a one-step natural-language CoT written as a Python comment plus code; a candidate is kept only if its code, concatenated with all previous steps, executes.
- Selection: `UCT(s) = Q(s) + c·sqrt(ln N_parent(s) / N(s))`, `Q(s) = q(s)/N(s)` (Eq. 1); c = 2; tree depth 16; 8 candidate nodes per step; 16 rollouts per problem.
- Terminal-guided annotation (rounds 1–2): `q(s_i)^k = q(s_i)^{k−1} + q(s_d)^k` (Eq. 2); terminal score +1 if the answer is correct, −1 otherwise; `q(s_i)^0 = 0`.
- From round 3 the PPM predicts the initial q value of each step (Eq. 3); terminal nodes are still scored against ground truth.

## Training data and models

- Problems: 747K word problems with final answers, mainly NuminaMath (competition level only) and MetaMath, plus GPT-4 problems seeded from 7.5K MATH train and 3.6K AMC-AIME problems, kept if at least 3 of 10 GPT-4 solutions agree (§3.4.1).
- Policy SFT: top-2 correct trajectories per problem by average Q; 2 epochs; sequence length 4,096; batch 128; AdamW; linear schedule, LR 7e-6 for Qwen; fine-tuned from the base model in each round; synthetic problems with trajectory accuracy under 50% are removed (§3.4.1, App. A.1).
- PPM: initialized from the policy with a scalar tanh head; per step, the two highest-Q candidates leading to correct answers are positives and the two lowest-Q candidates leading to wrong answers are negatives (the final step uses whole trajectories); loss `−(1/(2×2)) E[log σ(r_θ(x, y_pos) − r_θ(x, y_neg))]` (Eq. 4); 1 epoch, batch 512, LR 7e-6. Reason given: Q-values are too imprecise to use directly as reward labels (§1, §3.3).
- Rounds (§3.4.2, Table 2): round 1 DeepSeek-Coder-V2-Instruct (236B), 8 rollouts, 5 candidates, about two weeks on 10 nodes of 8×H100; rounds 2–4 the 7B policy (and PPM from round 3); round 4 adds 64 and up to 128 rollouts for unsolved problems. Coverage 60.17% → 66.60% → 77.86% → 90.25%. Of 20 sampled unsolved problems, 19 had wrong reference answers.

## Results

- Policy alone, MATH (Table 3): base 58.8; rounds 1–4 69.6, 73.6, 75.8, 78.4.
- With PPM-guided search over 8 trajectories (Table 6): 75.2, 86.6, 87.0, 89.4.
- Table 5, Qwen2.5-Math-7B: rStar-Math with 64 trajectories 90.0 MATH, 53.3 AIME 2024, 65.6 OlympiadBench; Qwen2.5-Math-7B-Instruct 82.6 / 6.0 / 41.6.
- Greedy policy (Table 10): 78.4 MATH, 26.7 AIME 2024.
- SFT-data ablation, Qwen2.5-Math-7B (Table 7, MATH): MetaMath 55.2; NuminaMath-CoT 69.6; random self-samples 72.4; rejection sampling with ORM 73.4; step-by-step verified 78.4.
- Reward model (Table 8, MATH): ORM best-of-N 82.6; PQM (MSE on Q-values) 88.2; PPM 89.4.
- Backtracking appears in search outputs without self-reflection data or prompts (§5, Fig. 4, App. A.2).
- Stated scope: word problems; code or general reasoning would need test cases, human labels, or mutual verification (§5).

## Errors in the earlier excerpt and library card

- Round 0 "Qwen2.5-Math-7B-Instruct"; "Q-gap > δ sibling pairs"; "pairwise avoids Goodhart issues of scalar PRMs"; "MATH 58 → 78 → 85 → 88 → 90"; "58.5 Olympiad"; "PUCT with prior P"; "~100K GPU-hours"; "replacing PPM with a scalar PRM loses 6 MATH points".
