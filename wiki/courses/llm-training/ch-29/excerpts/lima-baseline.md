---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2305.11206 (primary text; the library card [[lima]] had no Verification section on 2026-09-15)
source_url: https://arxiv.org/abs/2305.11206
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; re-read against the arXiv PDF)"
---

# Excerpt: what LIMA's 1,000 examples are, and why they cannot be reproduced as a filter

**Artifact:** Zhou et al., "LIMA: Less Is More for Alignment" (arXiv v1 2023-05; NeurIPS 2023). Loci are section and table numbers of the arXiv PDF.

## Composition (§2, Table 1)

| Source | Examples | Mean input length | Mean output length |
|---|---|---|---|
| Stack Exchange (STEM) | 200 | 117 | 523 |
| Stack Exchange (Other) | 200 | 119 | 530 |
| wikiHow | 200 | 12 | 1,811 |
| Pushshift r/WritingPrompts | 150 | 34 | 274 |
| Natural Instructions | 50 | 236 | 92 |
| Paper Authors (Group A) | 200 | 40 | 334 |

- Total: exactly 1,000 sequences, roughly 750,000 tokens (Table 1 caption).
- 750 questions and answers were selected from community forums "sampling for quality and diversity"; 250 prompts and responses were written by hand "while optimizing for task diversity and emphasizing a uniform response style" (§1).
- Stack Exchange sampling (§2.1): 200 questions from 75 STEM exchanges and 200 from 99 other exchanges, sampled with temperature τ = 3 across exchanges; the highest-scoring self-contained questions; the top answer with score at least 10; automatic removal of answers shorter than 1,200 or longer than 4,096 characters, written in the first person, or referencing other answers. wikiHow: 200 articles, a category first (of 19) and then an article (§2.1). r/WritingPrompts examples were selected manually (§2.1).
- Test set: 300 prompts (70 r/AskReddit, 230 Group B); dev set: 50 prompts (Table 1).

The selection includes manual choices and hand-written responses. There is no published algorithm that turns a synthetic pool into "a LIMA-style subset", so a "LIMA-recipe filter" baseline is not defined by this paper.

## Training (§3)

- Base: LLaMA 65B; a special end-of-turn token separates speakers.
- 15 epochs; AdamW β1 = 0.9, β2 = 0.95, weight decay 0.1; no warmup; LR 1e-5 decaying linearly to 1e-6; batch 32 examples (64 for smaller models); texts over 2048 tokens trimmed.
- Residual dropout rising linearly from 0.0 at the bottom layer to 0.3 at the top (0.2 for smaller models).
- Checkpoints are selected manually between the 5th and 10th epochs on the 50-example dev set, because perplexity did not correlate with generation quality.

## Ablations (§5; 7B LLaMA, ChatGPT grades helpfulness on a 1–6 scale, 5 samples per prompt, 95% confidence intervals)

- Diversity: 2,000 Stack Exchange examples (heterogeneous prompts) score higher than 2,000 wikiHow examples (homogeneous prompts) (Figure 5).
- Quality: 2,000 unfiltered Stack Exchange examples score 0.5 points below 2,000 filtered ones (Figure 5).
- Quantity: exponentially increasing Stack Exchange training sets; "doubling the training set does not improve response quality" (Figure 6).
- The 7B model needed at least 2,000 examples for stable training (footnote 5).

## What other studies measured when LIMA's 1K was used as a baseline

- [[deita]] Table 5, LLaMA-1-13B: LIMA 1K scores 4.29 MT-Bench and 41.98 AlpacaEval, against 5.84 / 73.91 for a random 6K subset of the DEITA pool. The LIMA set was not matched in size or source.

## Use in ch-29

- The lab does not use a "LIMA-matched" arm. It uses a random subset of the same filtered pool and a matched-size random subset of a general mixture ([[tulu-3-sft-and-eval]]) as baselines.
- LIMA's §5 result is the reason every arm is matched in size: quantity alone did not change quality in that setting, so size differences between arms would confound the comparison.
- LIMA's 15 epochs belong to a 1,000-example set with manual checkpoint selection and are not transferred to the lab.

## Connections

- [[lima]] — library card for the same paper (not verified at the time of this revision).
- [[deita]] — automated selection compared against LIMA 1K.
