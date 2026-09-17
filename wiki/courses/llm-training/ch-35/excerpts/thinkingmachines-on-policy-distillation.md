---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: Thinking Machines Lab blog post (no library card at the time of writing; chapter-local verified extract)
source_url: https://thinkingmachines.ai/blog/on-policy-distillation/
created_at: "2026-09-15"
---

# Excerpt: On-Policy Distillation

- **Authors:** Kevin Lu in collaboration with others at Thinking Machines Lab
- **Year:** 2025 (published 2025-10-27; page includes a June 2026 update note)
- **Source type:** practitioner evidence (blog post with experiments, numbers, and code; not a model release report)
- **Used in:** ch-35 §1, §6.9, Negative samples and negative feedback, Generalization lens

## Objective ("Loss function: reverse KL", "Pseudocode")
- "We choose the per-token reverse KL — the divergence between the student's (π_θ) and teacher's (π_teacher) distribution for each token conditioned on the same prior trajectory."
- Discount factor zero: "the student only optimizes the immediate next token".
- Implementation: sample from the student, query teacher log-probabilities on the sampled tokens, `reverse_kl = sampled_logprobs - teacher_logprobs`, `advantages = -reverse_kl`, then an importance-sampling policy-gradient loss. The post uses sampled tokens only: "we do not consider logit (top-k) distillation in any of our experiments".
- Example on SimpleBench (student Qwen3-4B-Instruct-2507, teacher Qwen3-235B-A22B-Instruct-2507): the teacher penalizes tokens "that start phrases which lead the student astray"; "the final answer, though wrong, isn't penalized — it is entirely predictable conditional on the whole preceding sequence."
- "SFT, using forwards KL, adds support for new tokens. Reverse-KL methods can then perform mode seeking within the initialization's support."

## Reasoning experiment (Qwen3-8B-Base student)
- Off-policy distillation on OpenThoughts-3 (QwQ-32B responses): 400k prompts reach 60% on AIME'24; extrapolation of the log-linear trend puts 70% at about 2M prompts.
- On-policy distillation from the 400k checkpoint reaches 70% in about 150 steps (about 77K prompts, 4 samples per prompt). The on-policy teacher was Qwen3-8B "as it performs slightly better"; FLOPs are counted for a 32B teacher.
- FLOPs table: SFT-400K 60% (teacher 8.5×10²⁰, student 3.8×10²⁰); SFT-2M extrapolated ~70% (3.4×10²¹, 1.5×10²¹); RL 68%; on-policy distillation 70% (8.4×10¹⁹, 8.2×10¹⁹), "9-30×" compute efficiency versus SFT-2M. GPU-hour reduction "closer to 18x" when the SFT dataset is given.
- LoRA rank 32 trails full fine-tuning by 13% after SFT and by 6% after on-policy distillation.

## Personalization experiment (Qwen3-8B, internal documents)
| Model | Internal QA (knowledge) | IF-eval (chat) |
|---|---|---|
| Qwen3-8B | 18% | 85% |
| + midtrain (100% documents) | 43% | 45% |
| + midtrain (70% documents, 30% chat) | 36% | 79% |
| + midtrain (70%) + on-policy distill | 41% | 83% |
The distillation phase uses the earlier Qwen3-8B as teacher on Tulu 3 prompts and has "no relation to the internal document data". "Although mixing in at least 30% of chat data helps preserve most instruction-following ability, there is no weighting which maintains the original performance on IF-eval."

## Verification
- Checked on 2026-09-15 against https://thinkingmachines.ai/blog/on-policy-distillation/ (page text fetched 2026-09-15).
- Not reported by the source: learning rates; number of seeds; confidence intervals for the AIME'24 and IF-eval numbers.
