---
chapter: ch-32c
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/interconnects-llama-4-long-context.md (planned card; not present on 2026-09-15; the library card-issue map proposes folding this post into model-reports/llama-4.md)
source_url: https://www.interconnects.ai/p/llama-4
created_at: "2026-09-15"
---

# Excerpt: Llama 4: Did Meta just push the panic button?

**Author:** Nathan Lambert (Interconnects), published 2025-04-07.
**Version read:** web page text cached 2026-09-14.
**Source type:** **anecdotal** for long-context claims (commentary without controlled measurements). ch-32c does not use it as support for any quantitative claim.

## Statements relevant to long context
- "One of the flagship features is the 10M (on Scout, Maverick is 1M) token context window on the smallest model, but even that didn't have any released evaluations beyond Needle in a Haystack (NIAH), which is seen as a necessary condition, but not one that is sufficient to say it is a good long-context model. Some more modern long-context evaluations include RULER or NoLiMa."
- "Other independent evaluation results range from medium to bad and confusing — I suspect very weird results are hosting issues with the very long context models." (The author gives no measurement for this suspicion.)
- "The very-long-context base models will be extremely useful for research."

## Statements about the release (for context only)
- Parameter and token counts added in brackets by the author: Scout 109B total parameters, ~40T training tokens; Maverick 400B total, ~22T training tokens (these match the model card; see [[llama-4]]).
- The LMArena ELO 1417 was reported for "an experimental chat version" of Maverick, which the author states is a different model from the released one.

## What the official sources state (from [[llama-4]], verified 2026-09-14)
- Scout supports 10M tokens and was "both pre-trained and post-trained with a 256K context length"; the post's long-context evidence is NIAH retrieval and cumulative NLL over 10M code tokens.
