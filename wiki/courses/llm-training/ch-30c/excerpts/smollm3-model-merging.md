---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "SmolLM3: smol, multilingual, long-context reasoner (Hugging Face blog), sections 'Off-policy model alignment with APO' and 'Model Merging'"
source_url: https://huggingface.co/blog/smollm3
created_at: "2026-09-15"
---

# Excerpt: SmolLM3 merge to recover long-context ability

The library card [[smollm-3]] predates the 2026-09 verification pass and does not describe the merge step.
This excerpt records the passage as published. Source type: official blog (the organization that trained the model).

## What the blog states
- After APO (Anchored Preference Optimization, a DPO variant), "downstream evaluations showed improvements across
  mathematics, science, instruction following, coding, chat, and multilingual tasks", but the team "observed
  performance degradation on long context benchmarks like RULER".
- The team traced the degradation to the reasoning mid-training stage, and notes that APO training data was limited
  to 24k tokens because most of the reasoning dataset was below that length.
- Merge recipe, performed with MergeKit:
  1. "Take each APO checkpoint and create a model 'soup'."
  2. "Combine the model soup with a mid-training checkpoint that has strong long-content performance. A linear merge
     with weights of 0.9 and 0.1 for the APO model soup and mid-training checkpoint, respectively, achieved the best
     performance. We were able to recover the base model's RULER score on contexts up to 128k tokens."
- "The resulting model is the checkpoint we are releasing today."

## Not reported in the blog
- RULER scores before and after the merge; the number of APO checkpoints in the soup; which mid-training checkpoint
  was used; other merge weights that were tried; the effect of the merge on non-long-context benchmarks.
- The Smol Training Playbook (same organization) names model merging as a post-training step but gives no merge
  numbers in the sections checked (searched for "merge", "soup", "RULER").

## Verification
- Read on 2026-09-15 in the cached Markdown of https://huggingface.co/blog/smollm3 (sections "Post-training" through
  "Model Merging") and the cached Smol Training Playbook text.
