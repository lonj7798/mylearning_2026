---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2503.09516 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2503.09516
created_at: "2026-09-15"
---

# Excerpt: Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

- **Authors:** Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, Jiawei Han
- **Year:** 2025 (arXiv v1 2025-03-12; later versions through v5 2025-08)
- **Source type:** paper
- **Used in:** [[read]] §2, Negative samples, Recipe, Generalization lens

## Abstract claim
"Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved
token masking for stable RL training and a simple outcome-based reward function. Experiments on seven
question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20%
(Qwen2.5-3B) over various RAG baselines under the same setting."

## Retrieved-token masking (§3.1)
- The PPO and GRPO objectives carry a token indicator: "I(y_t) is the token loss masking operation such that
  I(y_t)=1 if y_t is a LLM generated token" (§3.1, Eq. 2). Retrieved passages receive I(y_t)=0.
- The same mask is applied inside the GRPO KL term (§3.1, "GRPO with Search Engine").

## Rollout template (§3.3)
- The system instruction reads in part: "You must conduct reasoning inside `<think>` and `</think>` first every
  time you get new information ... call a search engine by `<search>` query `</search>`".
- Retrieved passages are inserted between `<information>` and `</information>`; the final answer is emitted
  between `<answer>` and `</answer>`.

## Reward (§3.4, Eq. 4)
- `r_φ(x, y) = EM(a_pred, a_gold)`, where `a_pred` is the answer extracted from response `y`. The paper uses no
  learned reward model and no format reward.

## Retrieved-token-masking ablation (Table 4; discussed §5.4)
Qwen2.5-7B-base trained with PPO; exact-match score per dataset and the seven-dataset average.

| Setting | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | Avg |
|---|---|---|---|---|---|---|---|---|
| With retrieved-token mask | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | **0.431** |
| Without mask | 0.388 | 0.567 | 0.391 | 0.325 | 0.321 | 0.108 | 0.304 | **0.343** |

The masked run is higher on every one of the seven datasets.

## Main results (Table 2, seven-dataset average exact match)
Qwen2.5-7B-Base 0.431; Qwen2.5-7B-Instruct 0.385; Qwen2.5-3B-Base 0.303; Qwen2.5-3B-Instruct 0.325.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2503.09516 (abstract page) and the arXiv HTML rendering
  (v4) for §3.1, §3.3, §3.4, Table 2, Table 4 and §5.4.
- Not extracted here: the PPO-vs-GRPO comparison, response-length dynamics, retriever configuration, and the
  training-data composition. Verify at the source before citing those.
