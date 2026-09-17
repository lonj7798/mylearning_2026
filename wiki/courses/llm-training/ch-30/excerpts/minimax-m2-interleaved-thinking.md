---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: MiniMax, "Interleaved Thinking Unlocks Reliable MiniMax-M2 Agentic Capability" (MiniMax News, 2025-11-03)
source_url: https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2
created_at: "2026-09-15"
source_type: official blog (the organization that trained MiniMax-M2)
---

# Excerpt: MiniMax-M2 and retained thinking across turns

No library card for this source exists yet (planned slug `minimax-m2-interleaved-thinking`).

## Definition (verbatim)

"Interleaved thinking is essential for LLM agents: it means alternating between explicit reasoning and tool use, while carrying that reasoning forward between steps."

## Observed failure and stated cause (verbatim)

"From community feedback, we've often observed failures to preserve prior-round thinking state across multi-turn interactions with M2. The root cause is that the widely-used OpenAI Chat Completion API does not support passing reasoning content back in subsequent requests. Although the Anthropic API natively supports this capability, the community has provided less support for models beyond Claude, and many applications still omit passing back the previous turns' thinking in their Anthropic API implementations."

## Reported numbers (verbatim)

"Retaining prior‑round thinking state improves performance significantly compared to discarding it, as evident across benchmarks: SWE‑Bench Verified 69.4 vs. 67.2 (Δ=+2.2; 3.3%), Tau^2 87 vs. 64 (Δ=+23; 35.9%), BrowseComp 44.0 vs. 31.4 (Δ=+12.6; 40.1%), GAIA 75.7 vs. 67.9 (Δ=+7.8; 11.5%), and xBench 72.0 vs. 66.0 (Δ=+6.0; 9.1%)."

## Not reported

Evaluation settings (scaffold, number of runs, sampling parameters), whether the discarded-thinking condition used the same client otherwise, and how M2's training data rendered earlier thinking.

## Connections

- [[read]] §7; [[chat-template-matrix]] for the Qwen3 and DeepSeek-V3.2 history rules; [[glm-5]] for GLM-5 Preserved Thinking.
