---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: arXiv:2602.15763v2, GLM-5 technical report (no library card as of 2026-09-15; chapter-local verified extract limited to context management)
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
---

# Excerpt: GLM-5 — context management for search agents

**Authors:** GLM-5 Team (Zhipu AI / Tsinghua University byline). arXiv v2 2026-02-24. Source type: official technical report.
**Used in:** [[read]] §4, §6, Recipe.

## Keep-recent-k (§4.2.4)
- Motivation stated: "We further observe that model accuracy degrades substantially under extremely long contexts (e.g., beyond 100k tokens)."
- Trajectory notation used by the report: `(q, r₁, a₁, o₁, r₂, a₂, o₂, …, r_n, a_n, o_n)`, where `q` is the question, `rᵢ` the reasoning at round i, `aᵢ` the action (tools: `search`, `open`, `find`, `python`), and `oᵢ` the tool observation.
- Rule: "We fold only observations earlier than the most recent k rounds: `oᵢ ← Tool result is omitted to save tokens.` i = 1, …, n − k." Reasoning `rᵢ` and actions `aᵢ` are kept.
- Setting and effect: "In our experiments, we set k = 5, which yields a stable improvement and improves GLM-5 from 55.3%(w/o keep-recent-k) to 62.0%(w/ keep-recent-k)."
- Sensitivity: "We also find that using different values of keep recent k or alternatively triggering keep-recent once the context length reaches a predefined token threshold, leads to the same results." No sweep values are printed.

## Hierarchical Context Management (§4.2.4)
- "During inference with keep-recent, if the total context length exceeds a threshold T, we discard the entire tool-call history and restart with a fresh context, while continuing to apply the keep-recent strategy. We select T = 32k via parameter search."
- Result: "Compared to using Discard-all alone, combining with keep-recent-k achieves consistent gains across all budgets, reaching a final score of 75.9, outperforming all open-source models equipped with context-management." Figure 8 plots BrowseComp accuracy against steps for GLM-4.7 Discard-all, GLM-4.7 Fewest-step, GLM-5 HCM, GLM-5 Fewest-step and GLM-5 Pass@K; per-point values are not printed.

## Evaluation protocol (§6.1.3, App. B)
- "We use a discard-all strategy as context management for BrowseComp, which is the same as DeepSeek-V3.2, and Kimi K2.5."
- App. B: "BrowseComp: Without context management, we retain details from the most recent 5 turns. With context management, we use the same discard-all strategy as DeepSeek-V3.2 and Kimi K2.5."
- Judge: "we standardize all judge-based components using the official OpenAI evaluation prompt and the proprietary model o3-mini as the judge", after observing that "performance on BrowseComp is sensitive to both the judge prompt and the judge model, and open-source judges can introduce systematic bias" (§4.2.4).

## Table 7 rows used by the chapter (BrowseComp, %)
Column order as printed: GLM-5, GLM-4.7, DeepSeek-V3.2, Kimi K2.5, Claude Opus 4.5, Gemini 3 Pro, GPT-5.2 (xhigh).

| Row | GLM-5 | GLM-4.7 | DeepSeek-V3.2 | Kimi K2.5 | Claude Opus 4.5 | Gemini 3 Pro | GPT-5.2 (xhigh) |
|---|---|---|---|---|---|---|---|
| BrowseComp | 62.0 | 52.0 | 51.4 | 60.6 | 37.0 | 37.8 | — |
| BrowseComp (w/ Context Manage) | 75.9 | 67.5 | 67.6 | 74.9 | 57.8 | 59.2 | 65.8 |

The 51.4 / 67.6 pair matches the numbers the DeepSeek report gives for itself ([[deepseek-v3.1]] §4.1, §4.4), and 60.6 / 74.9 matches the Kimi K2.5 report ([[kimi-k2-5]] §5).

## Not reported by the source
Per-point values behind Figure 8, the keep-recent-k sweep, the token threshold variant that "leads to the same results", the step budget used for the Table 7 rows, and the number of runs per cell.

## Verification
- Read on 2026-09-15 against the cached text of arXiv:2602.15763v2 (§4.2.4, §6.1.3, Table 7, App. B.3 evaluation settings).
