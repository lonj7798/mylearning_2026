---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: MiniMax news post, "Interleaved Thinking Unlocks Reliable MiniMax-M2 Agentic Capability" (no library card as of 2026-09-15)
source_url: https://www.minimax.io/news/interleaved-thinking-unlocks-reliable-minimax-m2-agentic-capability
created_at: "2026-09-15"
---

# Excerpt: Interleaved Thinking Unlocks Reliable MiniMax-M2 Agentic Capability

**Publisher:** MiniMax. Published 2025-11-03. Source type: official blog by the organization that trained MiniMax-M2.

## Definition given
- "Interleaved thinking ... means alternating between explicit reasoning and tool use, while carrying that reasoning forward between steps."
- The failure the post addresses is an API-layer one: "the widely-used OpenAI Chat Completion API does not support passing reasoning content back in subsequent requests", so applications drop the previous turns' thinking. The Anthropic-compatible API supports returning `thinking_blocks` in the message history; MiniMax's OpenAI-compatible API returns a separate `reasoning_details` field that the caller must pass back.

## Reported numbers (retained vs discarded prior-round thinking, same model)
Quoted sentence: "Retaining prior‑round thinking state improves performance significantly compared to discarding it, as evident across benchmarks: SWE‑Bench Verified 69.4 vs. 67.2 (Δ=+2.2; +3.3%), Tau^2 87 vs. 64 (Δ=+23; +35.9%), BrowseComp 44.0 vs. 31.4 (Δ=+12.6; +40.1%), GAIA 75.7 vs. 67.9 (Δ=+7.8; +11.5%), and xBench 72.0 vs. 66.0 (Δ=+6.0; +9.1%)."

| Benchmark | Thinking retained | Thinking discarded | Absolute difference |
|---|---|---|---|
| SWE-Bench Verified | 69.4 | 67.2 | +2.2 |
| Tau² | 87 | 64 | +23 |
| BrowseComp | 44.0 | 31.4 | +12.6 |
| GAIA | 75.7 | 67.9 | +7.8 |
| xBench | 72.0 | 66.0 | +6.0 |

## Qualitative claims (no measurement attached)
- "When prior state is dropped, cumulative understanding breaks down, state drift increases, self‑correction weakens, and planning degrades — especially on long‑horizon toolchains and run‑and‑fix loops."
- Recommended usage: "interleave thinking with tool feedback rather than front‑loading it, and persist the chain of thought so it compounds across turns."

## Not reported by the source
Model size and variant used for the table (only "M2"), the harness, the number of runs, the tool budget, the context window used, the judge for BrowseComp and GAIA, and whether the discarded-thinking condition also dropped tool results. The post gives no table caption and no appendix.

## Verification
- Read on 2026-09-15 from the page text at https://www.minimax.io/news/interleaved-thinking-unlocks-reliable-minimax-m2-agentic-capability (page dated Nov. 3, 2025).
- A retention rule with the same direction, stated as a template rule rather than a benchmark, appears in [[deepseek-v3.1]] §3.2.1.
