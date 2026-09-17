---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: arXiv:2509.13313v3 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2509.13313
created_at: "2026-09-15"
---

# Excerpt: ReSum — Unlocking Long-Horizon Search Intelligence via Context Summarization

**Authors:** Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, et al. (CUHK; Tongyi Lab, Alibaba Group; HKUST; Penn State). arXiv v1 2025-09; version read v3 2026-03-26. Source type: paper.
**Used in:** [[read]] §5, Negative feedback, Recipe.

## Paradigm (§3)
- ReSum "periodically invoke[s] an external tool to condense interaction histories into compact summaries. By resuming exploration from these compressed states, agents maintain awareness of prior discoveries while freeing up context space."
- It is positioned against architectural memory methods: "Current solutions, such as MEM1 and MemAgent, often rely on architectural modifications, e.g., generating internal memory tokens. While effective, these approaches break compatibility with pre-existing agents and necessitate end-to-end retraining."
- Trigger: "summarization is consistently triggered as the conversation history approaches the context limit". The WebSailor agents used have a 32k context window; the summary tool is invoked on approaching it (App. D: 4k for the query prompt and 28k for responses).
- **ReSumTool-30B**: a summary model fine-tuned for this role, trained on ⟨Conversation, Summary⟩ pairs collected by running ReSum rollouts with WebSailor-30B.

## ReSum-GRPO: advantage broadcasting (§3)
- A rollout becomes `n_g` segments. One reward per rollout: "we extract the final answer `a_{g,T}` from its last segment and compute a trajectory-level reward `R_g ∈ {0, 1}`."
- `Â_g = (R_g − mean({R₁,…,R_G})) / std({R₁,…,R_G})`, "which is broadcast to all segments within rollout g as `Â_g^{(i)} = Â_g` for i ∈ {1, …, n_g}."
- Stated purpose: "(1) encouraging agents to reason successfully from compressed states, and (2) ensuring early exploration steps receive appropriate bonus when they contribute to the final success."
- "ReSum-GRPO only modifies long trajectories by utilizing segmented rollouts, while short trajectories are processed identically to standard GRPO."

## Training-free results (Table 1, Pass@1 %)
Backbones are WebSailor-3B / -7B / -30B; all context windows 32k; tool-call budget 60; judge Qwen2.5-72B-Instruct.

| Backbone | Paradigm / summary tool | GAIA | BrowseComp-zh | BrowseComp |
|---|---|---|---|---|
| WebSailor-3B | ReAct | 25.6 | 8.2 | 3.3 |
| WebSailor-3B | Recent History (truncate to latest 22k) | 27.2 | 13.2 | 3.8 |
| WebSailor-3B | ReSum + ReSumTool-30B | 35.3 | 13.7 | 6.8 |
| WebSailor-7B | ReAct | 31.7 | 13.2 | 5.7 |
| WebSailor-7B | ReSum + ReSumTool-30B | 40.5 | 17.2 | 9.0 |
| WebSailor-30B | ReAct | 45.0 | 23.9 | 12.8 |
| WebSailor-30B | Recent History | 40.1 | 24.1 | 10.3 |
| WebSailor-30B | MEM1 | 33.3 | 25.0 | 12.7 |
| WebSailor-30B | ReSum + ReSumTool-30B | 47.3 | 24.1 | 16.0 |
| WebSailor-30B | ReSum + GPT-OSS-120B | 51.5 | 27.3 | 18.8 |

Abstract-level statement: "ReSum achieves a 4.5% improvement over ReAct in training-free settings, with ReSum-GRPO yielding a further 8.2% gain" (averages over the reported settings).

## RL results (Table 2, Pass@1 %; 1K training samples, 4 epochs)

| Backbone | RL | Paradigm | GAIA | BrowseComp-zh | BrowseComp |
|---|---|---|---|---|---|
| WebSailor-3B | — | ReAct | 25.6 | 8.2 | 3.3 |
| WebSailor-3B | GRPO | ReAct | 28.5 | 11.8 | 4.2 |
| WebSailor-3B | GRPO | ReSum | 38.5 | 17.3 | 8.5 |
| WebSailor-3B | ReSum-GRPO | ReSum | 37.9 | 20.5 | 9.2 |
| WebSailor-30B | GRPO | ReAct | 48.2 | 23.3 | 14.3 |
| WebSailor-30B | GRPO | ReSum | 48.5 | 29.3 | 15.0 |
| WebSailor-30B | MEM1-GRPO | MEM1 | 35.7 | 29.1 | 19.5 |
| WebSailor-30B | ReSum-GRPO | ReSum | 48.5 | 33.3 | 18.3 |

- "WebSailor-3B improves Pass@1 from 8.2% to 20.5% on BrowseComp-zh. In contrast, GRPO fails to enable agents to master summary-conditioned reasoning."
- Cost note: "MEM1-GRPO consumes nearly 3× more tokens than ReSum for a mere 1.2% improvement" (App. F.3).
- Behaviour after training: "the agent flexibly switches behaviors: it directly solves simpler queries without summarization while correctly leveraging summaries for complex, long-horizon tasks."

## Context-budget sweep (Table 3; Tongyi-DeepResearch-30B-A3B, up to 128k)

| Context limit | Tool calls | ReAct BrowseComp-zh | ReSum BrowseComp-zh | ReAct BrowseComp | ReSum BrowseComp |
|---|---|---|---|---|---|
| 32k | 40 | 41.2 | 43.8 | 27.7 | 34.5 |
| 48k | 60 | 42.5 | 46.7 | 32.8 | 38.2 |
| 64k | 80 | 43.6 | 48.6 | 36.3 | 40.3 |
| 96k | 100 | 46.0 | 47.9 | 39.8 | 41.0 |
| 128k | 120 | 45.7 | 46.6 | 42.2 | 44.5 |

"The gains are particularly substantial under stricter context constraints ... Notably, even with a massive 128k context, ReSum yields improvements."

## Not reported by the source
Per-segment token counts, the number of summarization events per rollout, learning rate and batch size in the main text (App. D.2 holds hyper-parameters), and non-search general-capability evaluations of the trained agents.

## Verification
- Read on 2026-09-15 against the cached PDF text of arXiv:2509.13313v3 (Abstract, §1, §3, §4.1–4.4, Tables 1–3, App. D–F).
