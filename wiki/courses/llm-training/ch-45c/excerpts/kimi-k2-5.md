---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: Kimi K2.5 technical report (no library card as of 2026-09-15; chapter-local verified extract limited to context management)
source_url: https://github.com/MoonshotAI/Kimi-K2.5
created_at: "2026-09-15"
---

# Excerpt: Kimi K2.5 — context management in agentic evaluation

**Authors:** Kimi Team, Moonshot AI. Source type: official technical report.
**Used in:** [[read]] §2, §4, §7, Recipe.

## Reported agentic scores (§5, "Agentic Capabilities")
- "On BrowseComp, K2.5 achieves 60.6% without context management techniques, 74.9% with Discard-all context management [14] — substantially outperforming GPT-5.2's reported 65.8%, Claude Opus 4.5 (37.0%) and Gemini 3 Pro (37.8%)." Reference [14] is the DeepSeek-V3.2 report.

## Protocol (App., "Context Management Strategies")
- Default: "Unless otherwise specified below, no context management is applied to agentic evaluations; tasks exceeding the model's supported context window are directly counted as failures rather than truncated."
- HLE with tools: "we employ a Hide-Tool-Result Context Management strategy: when the context length exceeds predefined thresholds, only the most recent round of tool messages (observations and return values) is retained, while the reasoning chain and thinking processes from all previous steps are preserved in full."
- BrowseComp: "Under the context management setting, we adopt the same discard-all strategy proposed by DeepSeek, where all history is truncated once token thresholds are exceeded." The thresholds themselves are not printed.

## Agent Swarm as a context-management architecture (§5)
- Table 6 (Agent Swarm vs single agent vs proprietary baselines):

| Benchmark | K2.5 Agent Swarm | Kimi K2.5 | Claude Opus 4.5 | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|---|
| BrowseComp | 78.4 | 60.6 | 37.0 | 65.8 | 77.9 |
| WideSearch | 79.0 | 72.7 | 76.2 | — | — |
| In-house Swarm Bench | 58.3 | 41.6 | 45.8 | — | — |

- "On BrowseComp, Agent Swarm achieves 78.4%, representing a 17.8% absolute gain over the single-agent K2.5 (60.6%) and surpassing even GPT-5.2 Pro (77.9%)."
- Framing: "an agent swarm is a kind of proactive and intelligent context management ... This approach differs from test-time context truncation strategies such as Hide-Tool-Result [2], Summary [71], or Discard-all [14], which react to context overflow by compressing or discarding accumulated histories. While effective at reducing token usage, these methods are inherently reactive and often sacrifice structural information or intermediate reasoning."
- Mechanism: sub-agents "maintain independent working memories and perform local reasoning without directly mutating or contaminating the global context of the central orchestrator. Only task-relevant outputs—rather than full interaction traces—are selectively routed back to the orchestrator."
- Wall-clock: "On the WideSearch benchmark, it reduces the execution time required to reach target performance by 3× ∼ 4.5× compared to a single-agent baseline" as target Item-F1 rises from 30% to 70% (Figure 8).
- "As shown in Figure 7, this proactive strategy outperforms Discard-all in both efficiency and accuracy on BrowseComp." Figure 7 has no printed per-point values.

## Not reported by the source
The token thresholds that trigger Discard-all and Hide-Tool-Result, the number of runs per cell, the sub-agent context budget, and an ablation that separates the swarm's parallel compute from its context isolation.

## Verification
- Read on 2026-09-15 against the cached report text (§5 agentic capabilities and Table 6; appendix evaluation-settings section "Context Management Strategies").
- Cross-check: the 60.6 / 74.9 pair is reproduced in the GLM-5 report's Table 7 ([[glm-5]]).
