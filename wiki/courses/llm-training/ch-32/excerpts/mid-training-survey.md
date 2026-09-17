---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: primary source arXiv:2510.06826v1 (planned library card papers/mid-training-survey.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2510.06826
created_at: "2026-09-15"
---

# Excerpt: Mid-Training of Large Language Models: A Survey

**Authors:** Kaixiang Mo, Yuxin Shi, Weiwei Weng, Zhiqiang Zhou, Shuman Liu, Haibo Zhang, et al. (Shopee; Nanyang Technological University). arXiv v1 2025-10-08. Source type: paper (survey; no new experiments).

## What the survey contributes

- A taxonomy of mid-training with three parts: data distribution, learning-rate scheduling, and long-context extension (Abstract; §I contribution list).
- A consolidated table of models that disclose mid-training data (Table I), a stage-by-stage summary of mid-training data distributions (Table II), an overview of LR schedulers in representative LLMs (Table III), and a frequency-scaling view of long-context methods (Table IV).
- The survey describes mid-training as "multiple annealing-style phases that refine data quality, adapts optimization schedules, and extend context length" (Abstract).

## Explanations the survey offers (Interpretation, not tested by the survey)

- Gradient noise scale, the information bottleneck, and curriculum learning are proposed as explanations of why mid-training helps (Abstract; §I). The survey cites these as theoretical perspectives; it runs no controlled experiment on them.

## Used in

ch-32 §1.1 (taxonomy only).
