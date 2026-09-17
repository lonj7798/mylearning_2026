---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "SmolLM3: smol, multilingual, long-context reasoner (Hugging Face blog, July 8, 2025), pre-training sections"
source_url: https://huggingface.co/blog/smollm3
created_at: "2026-09-15"
---

# Excerpt: SmolLM3 blog — pre-training configuration and stage mixtures

Source type: official blog (the organization that trained SmolLM3). The library card [[smollm-3]] covers post-training, has no Verification section, and does not give the pre-training values below.

## Budget and hardware
- Headline: "3B model trained on 11T tokens" (feature list).
- "Training Configuration: We use a global batch size of 2.36M tokens with 4096 sequence length, a learning rate of 2e-4, and the AdamW optimizer (beta1: 0.9, beta2: 0.95) with weight decay of 0.1 and gradient clipping of 1. We use the WSD (Warmup-Stable-Decay) scheduler, with 2000 warmup steps, and a linear decay to 0 in the final 10% training steps." "The model was trained on 384 H100 GPUs for 24 days." (section "Training Configuration").
- Architecture statements used in ch-14a: GQA with 4 groups; NoPE on every 4th layer; no weight decay on embedding layers "Following OLMo 2" (section "Model architecture").

## Stages ("Data mixture and training stages")
- "we train SmolLM3 on 11.2T tokens using a three-stage training strategy"; mixtures were chosen from "ablations on 3B models trained on 50B to 100B tokens".
- Stage 1, stable phase (0T → 8T): Web 85% (12% multilingual); Code 12%; Math 3%.
- Stage 2, stable phase (8T → 10T): Web 75% (12% multilingual); Code 15% (adds Stack-Edu); Math 10% (FineMath4+, InfiWebMath4+, MegaMath).
- Stage 3, decay phase (10T → 11.1T): Web 63% (12% multilingual); Code 24%; Math 13% ("introducing instruction and reasoning datasets such as OpenMathReasoning").

## Later stages (pointers only)
- Long-context extension: 100B tokens in two 50B stages, 4k → 32k with RoPE theta 1.5M and 32k → 64k with theta 5M. The released config for 4k → 32k sets `rope_theta: 2000000.0`; ch-32e covers this conflict.
- Reasoning mid-training: 35B tokens for 4 epochs (~140B tokens).

## Verification
- Read on 2026-09-15 against https://huggingface.co/blog/smollm3 (page text cached 2026-09-14; "Published July 8, 2025").
- Internal inconsistencies recorded in ch-14a: total tokens "11T" (headline), "11.2T" (stages paragraph), and stage 3 ending at "11.1T".
- Not reported by the blog: tokens per step as a product of parallelism settings (see [[smollm3-training-configs]]); per-stage evaluation numbers.
