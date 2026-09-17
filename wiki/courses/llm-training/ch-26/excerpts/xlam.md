---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/xlam.md
source_url: https://arxiv.org/abs/2409.03215
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against arXiv:2409.03215; the library card has not been verified and its base-model, score, and DPO-recipe claims do not match the paper)"
---

# Excerpt: xLAM — data unification, format augmentation, and the function-calling models

**Source library:** `wiki/raw-data/llm-training/papers/xlam.md` (not yet verified; values below were read in the paper on 2026-09-15)
**Paper:** Zhang, Lan, Zhu, Liu, Hoang, Kokane et al., "xLAM: A Family of Large Action Models to Empower AI Agent Systems", arXiv:2409.03215 (2024-09).

## Models (Table 1, §4.2)

| Model | Base | Context | Purpose |
|---|---|---|---|
| xLAM-1b-fc-r | DeepSeek-Coder-1.3B-instruct | 16k | function calling |
| xLAM-7b-fc-r | DeepSeek-Coder-7B-instruct-v1.5 | 4k | function calling |
| xLAM-7b-r | Mistral-7b | 32k | general |
| xLAM-8x7b-r | Mixtral-8x7b | 32k | general |
| xLAM-8x22b-r | Mixtral-8x22b | 64k | general |

## Data pipeline (§3)

- Unification into one function-calling-style format: task instruction, available tools, format instruction, few-shot examples, query, steps (§3.1).
- Prompt-format augmentation: shuffle the tool list, the order of name/description/parameters, and input sections; vary concatenation tokens (§3.2).
- Format instruction-following augmentation: "To avoid the model overfitting on JSON format … we prepare 15 different output formats along with their corresponding format instructions and format converters. The output formats include JSON, XML, YAML, plain text, etc." (§3.2).
- Quality checks: undefined functions or arguments, wrong argument types, argument hallucination (LLM judge step by step), low-quality reasoning (§3.3).
- Synthetic function-calling data: APIGen, 3,673 APIs, 60,000 samples from DeepSeek-V2-Chat and Mixtral-8x22B-Inst (§3.4).
- Mixture: general instruction data is 20% to 30% of the general models' training set; xLAM-7b-fc-r and xLAM-1b-fc-r draw 50% of training data from the synthetic function-calling set and 50% from other tasks (§3.5).
- DPO: rejected samples are responses from less powerful models, with a human-verified subset (§3.5). LoRA is used for DPO in all xLAM models; SFT uses a cosine schedule with 100 warmup steps (§4.1). β is not reported.

## Results

- BFCL v2 (cutoff 2024-09-03, Table 5): xLAM-8x22b-r overall 87.31 (rank 1); xLAM-8x7b-r 83.38 (rank 6); xLAM-7b-r 80.33 (rank 14); xLAM-7b-fc-r 80.18 (rank 17), irrelevance 79.54, relevance 80.49. All models were trained before the v2 live data was released (§5.1).
- ToolQuery-Unified (Table 3, §5.2.1): when the system prompt is given in the unified format with required structured output, GPT-4o's success drops by 42% relative to ToolQuery, while xLAM-8x22b-r is described as comparable. The authors attribute the stability to training on the unified format.
- ToolBench pass rate (Table 4): xLAM-7b-r 0.5308 (unseen instructions), 0.5300 (unseen tools, seen category), 0.5850 (unseen tools, unseen category).

## Not reported

Epoch counts, learning rates, batch sizes, DPO β, total SFT tokens, and general-capability benchmarks (MMLU, IFEval) for the function-calling models.
