---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/step-dpo.md
source_url: https://arxiv.org/abs/2406.18629
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Step-DPO — preference pairs at the first wrong step, with self-generated chosen steps

**Checked on 2026-09-15 against arXiv:2406.18629v1.** The library card `papers/step-dpo.md` had no verification section at that date and contains the errors listed at the end; this excerpt is the checked extract used by ch-24 §6, the negative-samples section, and the Recipe.

## Loss (§3.1, Eq. 2)

`L(θ) = −E[log σ(β log(π_θ(s_win | x; s_{1∼k−1}) / π_ref(s_win | x; s_{1∼k−1})) − β log(π_θ(s_lose | x; s_{1∼k−1}) / π_ref(s_lose | x; s_{1∼k−1})))]`

Stated motivation: in long-chain math answers "the first error often appears midway", and "rejecting an entire undesirable answer in DPO may also discard preceding correct reasoning steps, introducing significant noise" (§3.1).

## Data pipeline (§3.2)

1. Error collection: prompt the reference model with "Let's think step by step. Step 1:"; keep answers whose final answer differs from ground truth.
2. Step localization: verify steps in order until the first error, "manually or using GPT-4"; that step is `s_lose`.
3. Rectification: sample continuations from `π_ref` given the correct prefix; keep those whose final answer matches; the first step of a kept continuation is `s_win`. Samples whose `s_win` is still wrong may be filtered manually or by GPT-4.
- Humans or GPT-4 "only locate errors and rank answers"; they do not write corrections (§3.2).
- Out-of-distribution chosen steps (GPT-4 corrections) have low reference log probability; the authors cite gradient decay (§3.2).

## Settings (§4.1)

- SFT data: 374K DeepSeekMath responses to MetaMath and MMIQC problems with correct answers; 299K used for SFT; SFT 3 epochs (7B) or 2 (>30B), batch 256, LR 5e-6, linear decay, warmup ratio 0.03.
- Step-DPO: about 10K pairs (remaining SFT data plus an AQuA subset); 8 epochs (7B) or 4 (>30B); batch 128; LR 5e-7; β 0.4 (0.5 for 72B); cosine schedule, warmup 0.1.

## Results

- Table 3 (5K pairs, MATH): Qwen2-7B-SFT 54.8 → DPO 55.0 → Step-DPO 55.8; Qwen2-72B-SFT 61.7 → 62.5 → 64.1.
- Table 4 (MATH): Qwen2-7B-SFT 54.8; Step-DPO with GPT-4-corrected steps 55.1; with self-generated steps 55.8.
- Table 1 (10K pairs, MATH / GSM8K): Qwen2-7B-SFT 54.8 → 55.8 / 88.2 → 88.5; Llama-3-70B-SFT 56.9 → 59.5; Qwen2-72B-SFT 61.7 → 64.7; Qwen2-72B-Instruct 69.4 → 70.8 / 92.4 → 94.0 (reproduced with the authors' prompt); DeepSeekMath-RL 51.7 → 53.2.
- Fig. 2: DPO's validation reward margin is limited and plateaus; Step-DPO's is larger.

## Errors in the earlier excerpt and library card

- "Teacher (GPT-4 / Qwen2-72B) generates the corrected step" (the reference model generates it).
- "Qwen2-7B-Instruct 53.0 → 58.6 with 10K pairs; full-trajectory DPO on 100K pairs 54.3" (not in the paper).
- "KL-constrained gradient dilution" explanation; "steps of 30–120 tokens"; "$5K–$10K GPT-4 cost".
