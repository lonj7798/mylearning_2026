---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: "Interleaved Thinking Unlocks Reliable MiniMax-M2 Agentic Capability (MiniMax-AI, 2025-11-03)"
source_url: https://www.minimax.io/news/interleaved-thinking-unlocks-reliable-minimax-m2-agentic-capability
created_at: "2026-09-15"
---

# Excerpt: Interleaved Thinking Unlocks Reliable MiniMax-M2 Agentic Capability

- **Organization:** MiniMax-AI
- **Year:** 2025 (post dated 2025-11-03)
- **Source type:** official blog
- **Used in:** [[read]] §2 (template fidelity across turns), Common mistakes

## The measured comparison
The post compares MiniMax-M2 evaluated with prior-round reasoning preserved and fed back across turns against
the same model with that reasoning discarded (verbatim): "Retaining prior-round thinking state improves
performance significantly compared to discarding it, as evident across benchmarks: SWE-Bench Verified 69.4 vs.
67.2 (Δ=+2.2; +3.3%), Tau^2 87 vs. 64 (Δ=+23; +35.9%), BrowseComp 44.0 vs. 31.4 (Δ=+12.6; +40.1%), GAIA 75.7 vs.
67.9 (Δ=+7.8; +11.5%), and xBench 72.0 vs. 66.0 (Δ=+6.0; +9.1%)."

| Benchmark | Thinking retained | Thinking discarded | Δ |
|---|---|---|---|
| SWE-Bench Verified | 69.4 | 67.2 | +2.2 |
| τ² | 87 | 64 | +23 |
| BrowseComp | 44.0 | 31.4 | +12.6 |
| GAIA | 75.7 | 67.9 | +7.8 |
| xBench | 72.0 | 66.0 | +6.0 |

This is an inference-time comparison of one model under two history-construction policies. It is not an RL
ablation, and no seeds or variance are reported.

## Cause named by the post
"The root cause is that the widely-used OpenAI Chat Completion API does not support passing reasoning content
back in subsequent requests." The post adds a `reasoning_details` field to its OpenAI-compatible API so the chain
of thought can be passed back, and notes that the Anthropic-compatible API carries `thinking_blocks` natively.

## Design statement
"Interleaved thinking is essential for LLM agents: it means alternating between explicit reasoning and tool use,
while carrying that reasoning forward between steps." The recommendation given is to "interleave thinking with
tool feedback rather than front-loading it, and persist the chain of thought so it compounds across turns."

## Verification
- Checked on 2026-09-15 against the cached text of the MiniMax news post dated 2025-11-03.
- Not reported by the post: evaluation scaffolds, sample counts, temperature, or whether the two settings share a
  single evaluation harness beyond the API difference described.
