---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/naturalthoughts.md on 2026-09-15)
source_url: https://arxiv.org/abs/2507.01921
source_version: arXiv v1 (2025-07-02)
created_at: "2026-09-15"
---

# Excerpt: NaturalThoughts: Selecting and Distilling Reasoning Traces for General Reasoning Tasks (Li, Emad, Padthe, Lanchantin, Yuan, Nguyen, et al.; FAIR at Meta)

Facts used by [[read]], read in the arXiv v1 PDF text on 2026-09-15.

## Data and annotation (§3.1)
- Questions sampled from NaturalReasoning, "2.8 million questions that span multiple domains"; teacher DeepSeek-R1. The paper does not describe a correctness filter on the teacher responses.
- Annotations: question domain and topic (13 top-level domains); meta-reasoning strategies per trace and a 0-10 verbosity score, both by Llama-3.1-70B-Instruct.

## Selection strategies (§3.2)
- Diversity: uniform over topic domains; uniform over clusters of Llama-3.1-8B-Instruct question embeddings; reasoning-strategy count between R_min = 4 and R_max = 8 with low-density traces downsampled.
- Difficulty: "Long" samples each example with probability p = (l/C)^τ, l = reasoning length in tokens, C = 5000, τ = 2.5; verbosity subsets; "Models Disagree" when DeepSeek-R1 and Llama-3.3-70B answers disagree as judged by Llama-3.1-8B-Instruct.

## Training (§4)
- Students: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct (Llama-3.3-70B-Instruct in Table 3).
- Loss masked on the question; maximum response length 16,384 tokens with complete responses only; dynamic batches of about 400k tokens; 10 epochs for 1k examples, 6 for 10k, 8 for 100k and 500k; AdamW, weight decay 0.1, constant LR 2e-5; 32 H200 GPUs.
- Evaluation: temperature 0.6, top-p 0.95 (Llama) and 0.7, 1.0 (Qwen); maximum 16,384 generated tokens; 24 seeds for GPQA-Diamond, 16 for MATH-500, 1 for MMLU-Pro and SuperGPQA.

## Results (Tables 1-4)
- Llama-3.1-8B-Instruct, GPQA-D / MATH-500 / MMLU-Pro / SuperGPQA: base 29.0 / 49.1 / 47.7 / 21.9; LIMO (817) 34.0 / 56.5 / 55.1 / 23.9; S1K (1k) 36.9 / 59.4 / 56.2 / 27.2; NT Random 1k 37.1 / 57.8 / 56.6 / 27.3; NT Random 10k 37.6 / 61.3 / 57.8 / 28.8; NT Models Disagree 10k 39.7 / 61.9 / 59.1 / 29.7; NT Random 100k 42.5 / 67.5 / 59.8 / 31.2; NT Models Disagree 100k 45.2 / 70.2 / 59.8 / 32.2; NT Random 500k 48.3 / 72.3 / 61.9 / 31.3; NT Models Disagree 500k 45.2 / 70.8 / 59.8 / 30.7; OpenThoughts3 100k 37.8 / 82.2 / 59.0 / 30.0; DeepSeek-R1-Distill-Llama-8B (800k) 46.3 / 89.1 / 56.2 / 29.3.
- Topic-uniform sampling at 10k (32.7 GPQA-D) scored below random 10k (37.6); the authors attribute this to a topic distribution "too concentrated among a small set of topics" (§5.1).
- Qwen2.5-7B-Instruct (Table 2): NT Random 500k 48.6 / 83.1 / 62.3 / 35.2 vs OpenThoughts3 1.2M 46.9 / 91.2 / 59.1 / 33.5.
- Llama-3.3-70B-Instruct (Table 3): NT Random 100k 67.6 / 88.5 / 78.9 / 50.6 vs DeepSeek-R1-Distill-Llama-70B 65.2 / 94.5 / 78.5 / 49.4.
- Mixed System-1/System-2 distillation on GPQA-D (Table 4, Think mode): System-2 37.6 at 8,740.6 tokens; difficulty-based mixing 38.9 at 7,562.4 tokens with 36% System-2 examples; System-1 only 34.0 at 321.3 tokens (No-Think).

## Limits
Selection gains measured at 10k shrink or reverse at 500k for "Models Disagree"; no safety, instruction-following, or chat evaluation.
