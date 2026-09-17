---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/lora-without-regret.md (planned card; not present on 2026-09-15)
source_url: https://thinkingmachines.ai/blog/lora/
created_at: "2026-09-15"
---

# Excerpt: LoRA Without Regret

**Author:** John Schulman in collaboration with others at Thinking Machines Lab. Published 2025-09-29.
**Source type:** research blog with experiments (reliability: practitioner-evidence; not peer reviewed; the post trains no released model).
**Status:** no library card existed for this slug on 2026-09-15; statements checked against the fetched page text.

## Findings stated in the post
- Supervised fine-tuning: "For supervised fine-tuning on small-to-medium-sized instruction-tuning and reasoning datasets, LoRA performs the same as full fine-tuning." Experiments: one epoch on Tulu3 and a subset of OpenThoughts3, sweeping rank and LR with a constant LR schedule; the metric is log loss, not sampling evaluations.
- Capacity: "LoRA works well when not capacity constrained, i.e., the number of trainable parameters exceeds the amount of information to be learned."
- Layers: attention-only LoRA underperforms; "attention-only with rank 256 underperforms MLP-only with rank 128, despite them having approximately the same number of parameters."
- Reinforcement learning: "LoRA fully matches the learning performance of FullFT when running policy gradient algorithms for reinforcement learning, even with ranks as low as 1" (GSM and MATH experiments; DeepMath with Qwen3-8b-base).
- Information argument (labeled by the author as an argument, not a measurement): supervised learning provides O(number of tokens) bits per episode; policy gradient provides O(1) bits per episode through the advantage. For ~10,000 MATH problems × 32 samples, about 320,000 bits, while rank-1 LoRA on Llama-3.1-8B has about 3M parameters.
- Learning rate: "the optimal LR for LoRA is consistently 10x the one used for FullFT in the same application, for both supervised learning and reinforcement learning." A fit over 14 Llama and Qwen models on Tulu3 gave a multiplier of 9.8. For short runs (under about 100 steps, anecdotal) a multiplier around 15× is suggested.

## Not addressed
The post does not measure forgetting or held-out capability retention; it cites Biderman et al. (2024) for the setting where LoRA underperforms ("settings that resemble pre-training").

## Verification
- Checked on 2026-09-15 against https://thinkingmachines.ai/blog/lora/ (page text fetched 2026-09-15).
