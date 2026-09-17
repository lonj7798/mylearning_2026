---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/hammer.md
source_url: https://arxiv.org/abs/2410.04587
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against arXiv:2410.04587 and the MadeAgents/Hammer repository; the library card has not been verified and its masking ratio, irrelevance share, and score claims do not match these sources)"
---

# Excerpt: Hammer — function masking and irrelevance augmentation for cross-benchmark robustness

**Source library:** `wiki/raw-data/llm-training/papers/hammer.md` (not yet verified; values below were read on 2026-09-15)
**Paper:** Lin, Wen, Peng, Nie, Liao, Wang et al., "Hammer: Robust Function-Calling for On-Device Language Models via Function Masking", arXiv:2410.04587 (2024-10). Code: github.com/MadeAgents/Hammer.

## Problem statement (§1, §3, Table 1)

Function-calling models score inconsistently across benchmarks. Table 1: xLAM-7B-fc averages 69.05 over BFCL, API-Bank, Seal-Tools, Tool-Alpaca, and Nexus Raven (Nexus Raven 57.5), below Gorilla-OpenFunctions-v2 (70.48) and Granite-20B-FunctionCalling (74.19). The authors attribute this to reliance on function and parameter names. When names in the Seal-Tools test set are replaced with random strings, xLAM-1B-fc drops more than Hammer-1.5B (Fig. 2, values shown only in the figure).

## Method (§4)

- Function masking (§4.1): function names and parameter names in the candidate list are replaced with random strings; default parameter values are randomized and appended to descriptions; labels are updated with the same mapping.
- Irrelevance augmentation (§4.2): 7,500 examples sampled from xlam-function-calling-60k with the correct function removed from the candidate list and the label replaced by an empty list.
- Released processing script (`train/data_processing.py`, main branch, read 2026-09-14): the 60,000 + 7,500 examples are copied three times and shuffled; each copy is masked with probability 2/3 (`if random.random()>1/3: # Masking ratio p=1-1/3=0.67`); function names become 5–15 random characters and parameter names 4–10 characters from `[A-Za-z0-9_.]`.
- Released training script (`scripts/train.sh`): LLaMA-Factory, LoRA rank 32 on all modules, LR 5e-5, cosine, warmup ratio 0.00833, 1 epoch, cutoff length 2048, per-device batch 4, gradient accumulation 2, weight decay 0.01. The paper does not state that this configuration produced the released checkpoints.

## Results

- BFCL (09/20/2024, Table 2): Hammer-7B overall 83.92, irrelevance 72.87, relevance 92.68; xLAM-7B-fc 79.41, irrelevance 79.76; GPT-4-0125-Preview (Prompt) 85.79.
- Other benchmarks, F1 function name + arguments averaged over API-Bank L-1, L-2, Tool-Alpaca, Seal-Tools, Nexus Raven (Table 3): Hammer-7B 76.21; Granite-20B-FC 72.56; xLAM-7B-fc 67.65; Qwen2-7B-Instruct 59.84; GPT-4-0613 78.79.
- Same method on DeepSeek-Coder (Table 5): Deepseek-Coder-7B-Hammer 74.94 vs xLAM-7B-fc 67.65 (same base family, xLAM data only).
- Masking-ratio ablation (§5.5, Fig. 5; Qwen2-1.5B, Seal-Tools training, 1 epoch): a larger ratio slows learning on the same task (Seal-Tools) and improves the cross-task result (API-Bank). No optimum is stated.
- Irrelevance-share ablation (§5.6, Fig. 6; Qwen2-1.5B-Instruct, 10,000 samples): irrelevance detection rises and function-calling accuracy falls as the share grows; overall is best at about 10%, which sets the 7.5K size. The authors state the proportion "may require adjustment depending on the underlying model and training dataset".

## Not reported

Numeric values for Figs. 2, 5, 6; seeds and variance; general-capability benchmarks.
