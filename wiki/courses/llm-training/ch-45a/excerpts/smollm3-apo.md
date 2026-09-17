---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "the SmolLM3 release blog's alignment and merging sections; the library card [[smollm-3]] covers the same release but states no numbers for this stage"
source_url: https://huggingface.co/blog/smollm3
primary_version: "huggingface.co/blog/smollm3 (markdown source, fetched 2026-09-14)"
created_at: "2026-09-15"
---

# Excerpt: SmolLM3 — off-policy APO alignment and the merge that repaired long context

Hugging Face SmolLM team, 2025. ch-45a uses this source for the smallest preference-stage row and for the only
ledger row where the preference stage's context cap is the stated cause of a measured regression.

## Pair construction (verbatim)
> "After the SFT step, we performed a round of model alignment using a combination of the Tulu3 preference
> dataset for non-reasoning mode and new synthetic preference pairs for reasoning mode, that we generated from
> Qwen3-32B and Qwen3-0.6B... We selected generations from Qwen3-32B as 'chosen' and responses from Qwen3-0.6B
> as 'rejected' for alignment with Anchored Preference Optimization."

Both sides are off-policy: the chosen response comes from a larger model and the rejected response from a
smaller one of the same family, so the pair encodes a capability gap rather than a judgement about the policy's
own outputs.

## Objective
> "Anchored Preference Optimization (APO) is a variant of Direct Preference Optimization (DPO) that provides a
> more stable optimization objective... The APO objective has been shown to be more stable, and we also observed
> higher downstream performance in our internal ablations."

The blog cites arXiv:2408.06266 for APO. No β, learning rate, batch size, or epoch count is printed, and no
numbers from the internal ablation are given.

## The long-context regression and the merge
> "While downstream evaluations showed improvements across mathematics, science, instruction following, coding,
> chat, and multilingual tasks, we observed performance degradation on long context benchmarks like RULER. We
> traced this degradation back to the reasoning mid-training stage, where the focus on reasoning capabilities
> impacted long context performance. Additionally, the APO training data was limited to 24k tokens since the
> vast majority of our reasoning dataset fell below this length."

> "1. Take each APO checkpoint and create a model 'soup'. 2. Combine the model soup with a mid-training
> checkpoint that has strong long-content performance. A linear merge with weights of 0.9 and 0.1 for the APO
> model soup and mid-training checkpoint, respectively, achieved the best performance. We were able to recover
> the base model's RULER score on contexts up to 128k tokens."

Surrounding stage sizes: mid-training 35B tokens of reasoning data for 4 epochs (~140B tokens); SFT 1.8B tokens
(1B non-reasoning across 12 datasets, 0.8B reasoning across 10 datasets) for 4 epochs with best-fit-decreasing
packing and loss masked on user turns and tool results.

## Verification
- Read on 2026-09-15 from the cached markdown of the SmolLM3 blog post (scratchpad `sources/smollm3-blog-md.txt`),
  the mid-training, SFT, APO and model-merging sections.
- Not reported (blog checked): APO β, learning rate, batch size, epochs, number of preference pairs, and the
  RULER numbers before and after the merge.
