---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2403.04132
created_at: "2026-09-15"
---

# Excerpt: Chatbot Arena — An Open Platform for Evaluating LLMs by Human Preference

**Authors:** Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios N. Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, Ion Stoica
**Year:** 2024 (arXiv v1 2024-03-07)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v1 PDF text.

## Why ch-49 uses it

It is the human-vote reference that automatic judge benchmarks are validated against, it defines the Bradley–Terry ranking those benchmarks reuse, and it gives the cost and agreement numbers that explain why judges are used at all.

## Platform and scale (§1, §4)

- Pairwise battles between anonymous models, voted by users; over 240K votes from about 90K users at the time of writing.
- Ranking is a Bradley–Terry fit, not an Elo update; the paper reports intervals both with and without multiplicity correction, because the ranking is a function of all scores simultaneously (Fig. 5).
- Model pairs are chosen by an active sampling rule rather than uniformly (§4).
- One ranking experiment replays T = 213,576 historical votes (§7.1).

## Prompt diversity (§6.1)

Topic modeling with BERTopic (text-embedding-3-small → UMAP to 5 dimensions → HDBSCAN, minimum cluster size 32) identifies 600 clusters. The largest cluster accounts for 1% of prompts and the rest fall below 0.5%.

## Do the prompts separate models? (§6.2)

Sampling 30 prompts from each of seven topic clusters and comparing Llama-2-70b-chat with GPT-4 under an LLM judge: GPT-4's win rate reaches 97% in clusters requiring coding and reasoning and falls below 60% in clusters with less problem-solving (Table 2).

## Vote quality (§6.3)

160 battles (GPT-4-Turbo vs Llama-2-13B and GPT-4-Turbo vs GPT-3.5-Turbo-0613) were relabeled by expert graduate students, blind, with external fact-checking. Manual labeling of one comparison took 3 to 5 minutes.

Pairwise agreement rates (Table 3):

| Setting | Crowd vs Expert 1 | Crowd vs Expert 2 | Crowd vs GPT-4 | Expert 1 vs Expert 2 | Expert 1 vs GPT-4 | Expert 2 vs GPT-4 |
|---|---|---|---|---|---|---|
| vs Llama-2-13b | 72.8% | 77.8% | 75.6% | 89.8% | 81.0% | 78.5% |
| vs GPT-3.5-Turbo | 73.8% | 83.1% | 75.6% | 79.4% | 76.3% | 79.3% |

The paper summarizes this as 72–83% crowd–expert agreement, with expert–expert agreement at similar levels (79.4% and 89.8%).

GPT-4-Turbo win rates across judge types (Table 4): vs Llama-2-13b — Arena users 81.2%, Expert 1 89.4%, Expert 2 86.9%, GPT-4 judge 78.8%; vs GPT-3.5-Turbo — 76.3% / 82.5% / 89.4% / 79.4%.

## Limits relevant to ch-49

Votes come from self-selected users on prompts they choose, so the prompt distribution is the platform's, not a fixed suite. [[leaderboard-illusion]] audits what follows from that: sampling-rate asymmetry, private variant testing, and deprecation.

## Connections

[[judge-llm-bias]] (same group; MT-Bench and the LLM-judge study use Arena votes), [[arena-hard-benchbuilder]] (curates Arena prompts into an automatic benchmark), [[lmsys-style-control]] (adds style features to this Bradley–Terry fit), [[ppe-reward-model-eval]] (deploys post-DPO models to this platform to get downstream ground truth).
