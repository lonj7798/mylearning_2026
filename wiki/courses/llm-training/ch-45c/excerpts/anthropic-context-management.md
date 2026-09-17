---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: Anthropic product announcement, "Managing context on the Claude Developer Platform" (no library card as of 2026-09-15)
source_url: https://www.anthropic.com/news/context-management
created_at: "2026-09-15"
---

# Excerpt: Managing context on the Claude Developer Platform

**Publisher:** Anthropic. Published 2025-09-29. Source type: official product announcement (vendor-reported internal evaluation; the evaluation set is not released and the baseline is not described).

## The two features
- **Context editing** — "automatically clears stale tool calls and results from within the context window when approaching token limits." The removal happens inside the running context window; the conversation structure is kept.
- **Memory tool** — "enables Claude to store and consult information outside the context window through a file-based system. Claude can create, read, update, and delete files in a dedicated memory directory stored in your infrastructure that persists across conversations." It "operates entirely client-side through tool calls. Developers manage the storage backend."
- Claude Sonnet 4.5 is described as having "built-in context awareness—tracking available tokens throughout conversations".

## Reported numbers (section "Performance improvements with context management")
- Setting: "an internal evaluation set for agentic search", "complex, multi-step tasks". No task count, no per-task scores, no error bars.
- "combining the memory tool with context editing improved performance by 39% over baseline."
- "Context editing alone delivered a 29% improvement."
- Second setting: "In a 100-turn web search evaluation, context editing enabled agents to complete workflows that would otherwise fail due to context exhaustion—while reducing token consumption by 84%."
- The post does not state whether 39% and 29% are relative or absolute changes, and does not name the baseline agent or the metric.

## Use cases named
- Coding: "Context editing clears old file reads and test results while memory preserves debugging insights and architectural decisions".
- Research: "Memory stores key findings while context editing removes old search results".
- Data processing: "Agents store intermediate results in memory while context editing clears raw data".

## Not reported by the source
Evaluation set size and composition, baseline configuration, metric definition, number of runs, model versions other than "Claude Sonnet 4.5", the trigger threshold used for context editing, and whether the 84% token reduction was measured at equal accuracy.

## Verification
- Read on 2026-09-15 from https://www.anthropic.com/news/context-management (page date September 29, 2025; canonical link given on the page as https://claude.com/blog/context-management).
- This is the "new context tool (Anthropic, 2025a)" that [[deepseek-v3.1]] §4.4 names as the analogue of its Discard-all strategy.
- Companion engineering post with the mechanism descriptions: [[anthropic-context-engineering]].
