---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/livecodebench.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2403.07974
primary_version: arXiv:2403.07974v2 (2024-06-06; v1 2024-03)
created_at: "2026-09-15"
---

# Excerpt: LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code

Verbatim quotations used by ch-14 `read.md`, with loci. Authors: Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, et al. (UC Berkeley, MIT, Cornell). Checked against the v2 PDF on 2026-09-15. Source type: paper (benchmark). The full treatment of time-segmented evaluation belongs to ch-47a and ch-48; ch-14 uses only the evidence that benchmark-like content published before a model's data cutoff can reach its training data.

## Design (Abstract, §1)
> "LiveCodeBench hosts over five hundred coding problems that were published between May 2023 and May 2024."
> "we collect problems from weekly contests on competition platforms and tag them with a release date. Next, for newer models, we only consider problems released after the model’s cutoff date to ensure that the model has not encountered the exact problem in the training dataset."

## Contamination evidence (§5.1)
> "DeepSeek models were released in Sep 2023 and might have already been trained on some of the problems in our benchmark."
> "We notice a stark drop in the performance of DS-Ins-33B model after Aug. 2023 (right before its release date), which suggests that the earlier problems might indeed be contaminated."
> "performance of the GPT-4-O model drops on problems released since November (its official cutoff date)."
> "we find that this drop in performance primarily occurs for the LeetCode problems only and that the model performance is relatively smooth across the months for problems from other platforms."
> "we find that even the DS-Base-33B model also suffers from contamination dropping from Pass@1 ∼ 60 in May problems to Pass@1 ∼ 0 in September LeetCode problems. This also suggests the likely inclusion of competition problems in the pretraining of the DeepSeek models, thereby affecting all instruction models trained from it."
> "GPT-4-Turbo, Gemini-Pro, Mistral-L, and Claude-3s ... Irrespective of the release or cutoff dates, we do not find any drastic performance variations across the months"
