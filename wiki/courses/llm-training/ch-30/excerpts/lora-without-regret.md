---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: John Schulman and Thinking Machines Lab, "LoRA Without Regret" (2025-09-29)
source_url: https://thinkingmachines.ai/blog/lora/
created_at: "2026-09-15"
source_type: practitioner evidence (lab blog with experiments; not peer reviewed)
---

# Excerpt: LoRA Without Regret

No library card for this source exists yet (planned slug `lora-without-regret`).

## Setup (verbatim, section "Methods and results")

"We varied the LoRA rank over three orders of magnitude, with rank between 1 and 512, and compared these to full fine-tuning. To eliminate potential confounds from using a suboptimal learning rate, we swept the LR for each experimental condition. We used constant learning rate schedule (no warmup or cooldown). Our experiments used Llama 3 series models and Qwen3 models, including a mixture of experts (MoE) model. The main supervised learning experiments used the Tulu3 and OpenThoughts3 datasets". "In supervised learning, we measured log loss rather than employing sampling-based evals". Single epoch on Tulu3 and a subset of OpenThoughts3 for the rank sweep.

## Findings (verbatim bullets from "What matters for LoRA")

- "For supervised fine-tuning on small-to-medium-sized instruction-tuning and reasoning datasets, LoRA performs the same as full fine-tuning."
- "For datasets that exceed LoRA capacity, LoRA underperforms FullFT."
- "In some scenarios, LoRA is less tolerant of large batch sizes than full fine-tuning … This penalty is not mitigated by increasing the LoRA rank".
- "Even in small data settings, LoRA performs better when applied to all weight matrices, especially MLP and MoE layers. Attention-only LoRA underperforms even when we match the number of trainable parameters".
- "LoRA performs equivalently to FullFT for reinforcement learning even with small ranks."

## Learning rate ("LoRA rank")

"We find that the optimal learning rate for FullFT is lower by a factor of 10 than for high-rank LoRAs." The post notes that Biderman et al. (2024), Figure S1, found "a similar 10x ratio" with sampling evaluations. "The optimal LR changes by a factor of less than 2 between rank=4 and rank=512." Parametrization used: W′ = W + (α/r)·BA with α = 32.

## Limits for this chapter

The post measures training log loss and RL reward, not forgetting of abilities outside the training data, and it does not report seeds.

## Connections

- [[read]] §8; [[lora-learns-less-forgets-less]] for forgetting measurements.
