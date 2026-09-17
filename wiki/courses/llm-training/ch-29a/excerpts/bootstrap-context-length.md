---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2412.18860v2 (Bootstrap Your Own Context Length), §3-5, Tables 1, 3, 4 (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2412.18860
created_at: "2026-09-15"
---

# Excerpt: SelfLong — agent-workflow synthesis of long-input instruction data

- **Authors:** Liang Wang, Nan Yang, Xingxing Zhang, Xiaolong Huang, Furu Wei (Microsoft)
- **Year:** 2024 (arXiv v1 2024-12; v2 2025-03 used here)
- **Source type:** paper
- **Used in:** ch-29a §3.5, §7, Recipe

## Workflow (§3.1)
1. An LLM generates instructions that require integrating multiple documents; "a random text chunk is prepended to each prompt during every LLM call" for diversity.
2. E5-mistral-7b retrieves documents from about 10M Fineweb-Edu documents.
3. Retrieved documents are split into chunks of at most 4K tokens; query-focused summarization agents summarize each chunk for the
   instruction, "filtering out information that is irrelevant", recursively until the summaries fit the LLM.
4. The LLM writes the response from the summaries.
- "During model training, the synthetic instruction and retrieved documents are concatenated to form the input, while the generated
  response constitutes the target output. The intermediate summaries are not utilized during training."
- Long-output data: documents of 2K–32K tokens receive a back-translated writing instruction.

## Training (§3.2, §4.1)
- Data: 69K long-input samples (4.6B tokens), 10K long-output samples (77M tokens), Tulu-v2, deduplicated Infinity-Instruct, and ProLong
  continued-pretraining subsets; about 8.3B tokens total. Generator GPT-4o.
- Stages at 256K, 512K, 1M from 128K Llama-3 models; each stage doubles length and quadruples RoPE base.
- Loss: averaged over all input and output tokens for long samples; output tokens only for short samples.

## Results
- RULER at 1M (Table 1): SelfLong-8B-1M 69.6; Llama-3-8B-1M 64.3; LWM-Text-Chat-1M 60.1.
- Table 3 ablation (RULER, SelfLong-8B):
| Variant | 32k | 128k | 256k | 1M |
|---|---|---|---|---|
| Full | 89.5 | 82.0 | 79.7 | 69.6 |
| adjusted RoPE θ only (no training) | 71.0 | 58.8 | 56.1 | 48.1 |
| short data only (≤4K) | 83.0 | 61.0 | 55.0 | 46.6 |
| w/o synthetic data | 84.9 | 78.2 | 77.3 | 63.5 |
| Llama-3.1-8B-Instruct as generator | 83.7 | 80.0 | 77.9 | 66.1 |
- §5.1: with short data only, generalization to longer contexts is "even poorer ... compared to 'w/ adjusted RoPE θ only'".
- Table 4, Llama-3.1-8B-Instruct → SelfLong-8B-1M: BBH 50.1 → 48.7; GPQA 26.8 → 30.4; IFEval 77.4 → 65.9; MMLU Pro 37.5 → 35.3; MUSR 36.9 → 41.5.
  3B: IFEval 73.9 → 57.0.
