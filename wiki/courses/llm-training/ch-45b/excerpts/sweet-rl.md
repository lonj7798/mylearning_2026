---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2503.15478 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2503.15478
created_at: "2026-09-15"
---

# Excerpt: SWEET-RL — Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks

- **Authors:** Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar, Xian Li
  (Meta FAIR; UC Berkeley)
- **Year:** 2025 (arXiv v1 2025-03-19)
- **Source type:** paper
- **Used in:** [[read]] §4, Recipe, Connections to [[ch-29d]]

## Problem statement (Abstract, verbatim)
"Large language model (LLM) agents need to perform multi-turn interactions in real-world tasks. However,
existing multi-turn RL algorithms for optimizing LLM agents fail to perform effective credit assignment over
multiple turns while leveraging the generalization capabilities of LLMs and it remains unclear how to develop
such algorithms."

## Method
SWEET-RL is expanded by the authors as "RL with Step-WisE Evaluation from Training-time information". A critic
model is trained on information that is available only at training time (for example the reference solution) and
emits a per-turn signal used to train the policy. The policy itself never sees the training-time information,
so the asymmetry is between critic and actor rather than between training and deployment inputs of the policy.

## Benchmark
ColBench: an LLM agent collaborates with a simulated human partner over multiple turns on backend programming
and frontend design tasks.

## Reported result (Abstract)
"a 6% absolute improvement in success and win rates on ColBench compared to other state-of-the-art multi-turn
RL algorithms", which the authors state enables Llama-3.1-8B to match or exceed GPT-4o on these collaborative
content-creation tasks.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2503.15478 (abstract page).
- Not extracted here: the critic architecture and loss, the exact baselines behind the "6% absolute" comparison,
  ColBench task counts, and any ablation. Read the paper before citing numbers beyond the abstract figure above.
