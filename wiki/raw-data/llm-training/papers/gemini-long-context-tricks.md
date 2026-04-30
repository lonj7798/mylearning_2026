<!-- scope: official Gemini long-context guidance as a practical reference for million-token usage
     see-also: [[ruler]], [[babilong]]
-->

# Gemini Long-Context Tricks
- **Core Insight:** Very large context windows are useful only if you change prompting and pipeline design; naive “dump everything in” usage wastes both quality and cost.
- **Guideline:** For million-token context, structure the prompt, cache stable context, and design retrieval/placement strategy instead of assuming the model will reason uniformly over the whole sequence.
- **Author/Org:** Google
- **Year:** 2024-2026 docs and product guidance
- **URL:** https://ai.google.dev/gemini-api/docs/long-context
- **Relevant topics:** long context, context caching, million-token prompting, Gemini

## Summary
Google’s official long-context guidance for Gemini is valuable as a practical engineering note rather than a formal paper. The key lesson is that ultra-long context changes application design: prompt structure, latency, caching, and relevance placement all become part of the training/eval story.

## Key Points
- Long context supports new workflows, but cost and latency scale materially.
- Stable context should be cached when possible.
- Prompt organization still matters; more context does not imply uniform attention quality.
- Million-token usage should be paired with task design and evaluation, not assumed to be self-solving.

## Connections
- Practical product-side complement to synthetic long-context evaluations such as [[ruler]] and [[babilong]].
- Useful when interpreting public claims about Gemini’s long-window capabilities.

