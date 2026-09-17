---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: Hugging Face blog "SmolLM3: smol, multilingual, long-context reasoner" (the library card smollm-3 has no Verification section; chapter-local verified extract)
source_url: https://huggingface.co/blog/smollm3
created_at: "2026-09-15"
---

# Excerpt: SmolLM3 — reasoning mid-training on distilled traces

- **Authors:** Hugging Face (SmolLM team)
- **Year:** 2025 (published 2025-07-08)
- **Source type:** official blog (the organization that trained the model)
- **Used in:** ch-35 §2, Negative samples and negative feedback, Generalization lens

## Reasoning mid-training ("Mid-training" → "Reasoning Mid-training")
- "Our mid-training dataset contained 35B tokens sourced from Open Thought's OpenThoughts3-1.2M and a subset from NVIDIA's Llama-Nemotron-Post-Training-Dataset-v1.1 with reasoning traces from R1."
- "We used the ChatML chat template and wrapped packing to avoid providing too much structure to the model. We trained the model for 4 (~140B tokens) epochs."
- The stage "targeted a general capability without yet focusing on a specific domain".

## SFT ("Supervised Finetuning")
- 1.8B tokens: 1B non-reasoning (12 datasets) and 0.8B reasoning (10 datasets); 4 epochs (~8B tokens).
- Missing reasoning traces for some domains were generated "by prompting Qwen3-32B in reasoning mode with prompts from existing non-reasoning datasets".

## Preference stage ("Off-policy model alignment with Anchored Preference Optimization")
"We selected generations from Qwen3-32B as 'chosen' and responses from Qwen3-0.6B as 'rejected' for alignment with Anchored Preference Optimization." APO was more stable and gave "higher downstream performance in our internal ablations" (no numbers printed).

## Regression and repair ("Model Merging")
- "We observed performance degradation on long context benchmarks like RULER. We traced this degradation back to the reasoning mid-training stage."
- A linear merge of the APO model soup (0.9) with a mid-training checkpoint that has strong long-context performance (0.1) "recover[ed] the base model's RULER score on contexts up to 128k tokens."

## Verification
- Checked on 2026-09-15 against https://huggingface.co/blog/smollm3 (published 2025-07-08).
- Not reported by the source: RULER numbers before and after merging in the prose; mid-training learning rate.
