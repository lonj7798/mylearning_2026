---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: Hugging Face blog "SmolLM3: smol, multilingual, long-context reasoner" (2025-07), sections "Long Context extension" and "Model Merging" (chapter-local verified extract; the smollm-3 card has not been re-verified)
source_url: https://huggingface.co/blog/smollm3
created_at: "2026-09-15"
---

# Excerpt: SmolLM3 — long-context extension and merge-based recovery of RULER

- **Authors:** Hugging Face (SmolLM team)
- **Year:** 2025 (blog, July 2025)
- **Source type:** official blog
- **Used in:** ch-29a §7, Recipe, Generalization lens

## Long-context extension
- "we trained SmolLM3 on an additional 100B tokens to extend its context length. We sequentially extended the context window in two stages
  for 50B tokens each: first transitioning from 4k to 32k context with RoPE theta increased to 1.5M, then from 32k to 64k context with RoPE
  theta increased to 5M."
- "we found that upsampling specific long context data such as code repositories, books, and long web pages (beyond the naturally long
  samples in our mixture) didn't further boost performance on RULER and HELMET benchmarks." No numbers are printed.
- Intra-document masking is used during training ("tokens from different documents in the same training sequence don't attend to each other").
- YaRN is used at inference to reach 128k.

## Post-training regression and merge
- "we observed performance degradation on long context benchmarks like RULER. We traced this degradation back to the reasoning mid-training
  stage ... Additionally, the APO training data was limited to 24k tokens".
- Merge recipe: "1. Take each APO checkpoint and create a model 'soup'. 2. Combine the model soup with a mid-training checkpoint that has
  strong long-content performance. A linear merge with weights of 0.9 and 0.1 for the APO model soup and mid-training checkpoint,
  respectively, achieved the best performance. We were able to recover the base model's RULER score on contexts up to 128k tokens."
- RULER values before and after the merge are not printed in the blog.
