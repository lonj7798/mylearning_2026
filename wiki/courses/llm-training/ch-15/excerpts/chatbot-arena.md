---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/chatbot-arena.md (planned card; not present on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2403.04132
primary_version: arXiv:2403.04132v1 (2024-03-07); ICML 2024
created_at: "2026-09-15"
---

# Excerpt: Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

Authors: Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios N. Angelopoulos, Tianle Li, Dacheng Li, et al. (UC Berkeley and collaborators). Source type: paper. Read in the v1 PDF text on 2026-09-15 for ch-15 §2, §4, §6.

## Data (§3)
- Collection began April 2023; "As of Jan 2024, we have received around 240K votes from over 90K users"; more than 50 models; flagged unsafe requests 3% of requests (§3.2).
- Users see two anonymous models side by side and vote; users accept terms of use that allow public data release (§3).

## Ranking model and pair selection (§4)
- Bradley-Terry coefficients estimated from pairwise votes, with sandwich variance for confidence intervals (§4).
- "Active sampling rule": choose model pair a with probability proportional to the reduction in confidence-interval size, P_t(a) ∝ sqrt(Σ̂_t,a,a / |{t : A_t = a}|) − sqrt(Σ̂_t,a,a / (|{t : A_t = a}| + 1)) (Eq. 9).

## Prompt diversity (§6.1)
- BERTopic over user prompts: "The pipeline identifies 600 clusters covering a wide range of topics"; "the largest cluster only accounts for 1% of the entire set and the rest quickly drop to <0.5%".

## Vote quality (§6.3, Table 3)
160 randomly selected battles each for GPT-4-Turbo vs Llama-2-13b-chat and GPT-4-Turbo vs GPT-3.5-Turbo-0613; experts were UC Berkeley graduate students who labeled blind and fact-checked with external resources.

| Llama-2-13b pairs | Expert 1 | Expert 2 | GPT-4 |
|---|---|---|---|
| Crowd | 72.8% | 77.8% | 75.6% |
| Expert 1 | – | 89.8% | 81.0% |
| Expert 2 | – | – | 78.5% |

| GPT-3.5-Turbo pairs | Expert 1 | Expert 2 | GPT-4 |
|---|---|---|---|
| Crowd | 73.8% | 83.1% | 75.6% |
| Expert 1 | – | 79.4% | 76.3% |
| Expert 2 | – | – | 79.3% |

- The authors attribute the 10%-20% expert-expert disagreement mostly to prompts without a ground-truth answer, and the 5%-10% gap between crowd-expert and expert-expert agreement mostly to crowd users "making mistakes or overlooking factual errors in model's response" (§6.3).
