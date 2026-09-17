---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/lema-learning-from-mistakes.md on 2026-09-15)
source_url: https://arxiv.org/abs/2310.20689
source_version: arXiv v4 (2024-03-29); v1 2023-10-31
created_at: "2026-09-15"
---

# Excerpt: Learning From Mistakes Makes LLM Better Reasoner (An, Ma, Lin, Zheng, Lou, Chen; XJTU, Microsoft, PKU)

Facts used by [[read]], read in the arXiv v4 PDF on 2026-09-15.

## Data construction (§2.1-2.2, §3.2)
- Inaccurate reasoning paths r̃ are sampled from a reasoning model and kept when Ans(r̃) ≠ a (Eq. 1). GPT-4 is the corrector; a correction is kept only if its final answer is correct (Eq. 2).
- A correction has three parts: the incorrect step, an explanation of the mistake, and a corrected solution from that step (§2.1, Fig. 3).
- Human check of 50 GPT-4 corrections of LLaMA-2-70B paths on GSM8K: 35 excellent, 11 good, 4 poor (§2.1). With GPT-3.5-Turbo as corrector, "nearly half" of 20 corrections were poor (App. D.6).
- Sources of wrong paths: LLaMA-2, WizardLM, WizardMath, Text-Davinci-003, GPT-3.5-Turbo, GPT-4. Pair counts: 12,523 (GSM8K), 6,306 (MATH), 7,241 (CSQA) (§3.2).
- GPT-4 correcting its own MATH errors: 217 correct corrections out of 2,696 paths (8.0%); correcting LLaMA-2-70B: 37.5%; GPT-3.5-Turbo: 26.9% (App. D.6).

## Loss mask (App. B.2, Fig. 6)
- Correction example input: the problem, "Original Solution: {…}", and "Incorrect Step:". Output: the step label, explanation, and corrected solution.
- "Note that during the fine-tuning process, the input part serves as a prompt and only the loss in the output part participates in the back-propagation." The wrong solution is therefore never a target.

## Training (§3.3)
- QLoRA by default: rank 64, dropout 0.05; LR 1e−4 for models ≥ 34B and 2e−4 below 34B; batch size 96; 2,000 steps; checkpoints every 100 steps. All saved checkpoints are evaluated and "the accuracy of the best checkpoint" is reported; greedy decoding, max length 2,048.

## Results (Table 1-3, Fig. 4)
- LLaMA-2-70B, CoT fine-tuning → + LEMA: GSM8K 81.4 → 83.5; MATH 23.6 → 25.0; SVAMP 80.3 → 81.6; ASDiv 80.7 → 82.2; CSQA 84.2 → 85.3 (Table 1). LLaMA-2-7B: GSM8K 52.6 → 54.1; MATH 8.7 → 9.4.
- SVAMP and ASDiv use GSM8K training data; the authors treat them as out-of-distribution tasks (§4.1).
- Controlled data size (Fig. 4): LEMA-32K vs CoT-32K and LEMA-45K vs CoT-45K; "LEMA can still bring gains for four out of five backbone LLMs under the same data size … The only exception is for LLaMA-2-7B."
- Controlled training tokens (Table 2): CoT-45K has 5.4M tokens and LEMA-45K 5.8M; at 5.8M tokens LLaMA-2-70B CoT 82.1 vs LEMA 83.5; LLaMA-2-13B 64.2 vs 65.7. Expanding CoT to 6.8M tokens gives 82.2 for LLaMA-2-70B (§4.1).
- Specialized models (Table 3; baselines are the numbers reported in the original papers): WizardMath-70B 81.6 → 84.2; MetaMath-70B 82.3 → 85.4 on GSM8K.
- Correction-centric evolution on MATH with LLaMA-2-70B raises LEMA from 25.0% to 29.3% (§1).
