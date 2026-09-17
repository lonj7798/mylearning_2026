---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/dust-into-gold-negative-distillation.md on 2026-09-15)
source_url: https://arxiv.org/abs/2312.12832
source_version: arXiv v1 (2023-12-20); AAAI 2024
created_at: "2026-09-15"
---

# Excerpt: Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data (Li, Yuan, Feng, Pan, Sun, Wang, Wang, Li; BIT, Xiaohongshu)

Facts used by [[read]], read in the arXiv v1 PDF on 2026-09-15.

## Setting
- Teacher: gpt-3.5-turbo or gpt-4 generate chain-of-thought for MATH; 8 responses per question; responses split into D_pos and D_neg by final-answer correctness. Student: LLaMA-7B with LoRA (Methodology; Experimental Setup).
- MATH test set: 5,000 problems across seven subjects (App. Table 6). Out-of-distribution tests: GSM8K, ASDiv, MultiArith, SVAMP.

## Complementarity (Table 1)
- Correct MATH test answers: model trained on positives 253, model trained on negatives 166, intersection 29, IoU 0.074.

## Negative Assistant Training (NAT; Eq. 3-5)
- Step 1: train a Neg-LoRA with ordinary likelihood on D_neg (Eq. 3), then freeze it.
- Step 2: train a Pos-LoRA on D_pos; per layer, a corrected attention merges the two outputs: α = W_Q(h_input)·W_K([h_pos; h_neg])^T + [0.5; −0.5] (Eq. 4); h_output = α·W_V([h_pos; h_neg]) (Eq. 5). The paper states that the correction term constrains the weights for h_neg to [−0.5, 0.5], so negative knowledge can be added or subtracted.

## Results (Table 2, greedy decoding; average accuracy % over MATH subjects)
- GPT-3.5-Turbo traces: Fine-tune (original data) 3.88; CoT KD (positives only) 5.29; MIX (MLE on positives + negatives) 3.03 (−21.9% relative to fine-tune); CL 5.16; NT 4.48; UL 4.96; NAT 6.81 (+75.5%).
- GPT-4 traces: CoT KD 5.59; MIX 4.49; CL 5.24; NT 5.14; UL 6.03; NAT 6.83 (+76.0%).
- Baseline definitions (Appendix): NT maximizes E_pos log P − λ2·E_neg log P with λ2 = 0.05 (Eq. 15); UL uses a sequence-level unlikelihood term on negatives with λ3 = 0.05 (Eq. 16).
- The text reads MIX's result as "directly training the negative samples will make model toxic" and states that using negatives "only in the negative direction is not sufficient" (Main Results).
- Other parts: Negative Calibrated Enhancement (KL between Neg and NAT models weights self-distillation samples, Eq. 6-8) and Adaptive Self-Consistency (a ranker trained on positive and negative rationales, Eq. 9-11; ranker accuracy about 60%).
- After NAT, IoU between the negative model and the NAT model is 0.110 (App. Table 8).

## Limits visible in the source
- Absolute MATH accuracies are 3-7% for a 7B student; differences between methods are a few points on a 5,000-problem test set. Seeds and variance: not reported.
