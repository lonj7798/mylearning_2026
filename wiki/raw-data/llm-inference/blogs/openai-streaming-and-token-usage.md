<!-- scope: OpenAI API streaming and token usage accounting for inference
     deps: [[prefill-vs-decode]], [[kv-cache-memory-formula]]
     see-also: [[gpt-3-language-models-are-few-shot-learners]], [[batching-for-inference]]
-->

# OpenAI Streaming and Token Usage
- **Core Insight:** Streaming exposes generated output incrementally, while usage accounting separates input, output, and total tokens after or near completion depending on endpoint behavior.
- **Guideline:** Treat streaming output as partial until the final event/chunk; for billing and limits, record authoritative `usage` when provided and handle interrupted streams.
- **Authors:** OpenAI API documentation
- **Year:** 2026
- **URL:** https://developers.openai.com/api/reference/responses/overview ; https://developers.openai.com/api/docs/guides/migrate-to-responses ; https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
- **Relevant topics:** streaming, SSE, Responses API, Chat Completions, token usage, input tokens, output tokens

## Abstract
OpenAI's current documentation recommends the Responses API for new projects while Chat Completions remains supported. Streaming returns incremental events/chunks so applications can display tokens before the full response is complete. Token usage is endpoint-specific metadata and may arrive only at the end of a streamed response.

## Key Contributions
- Responses API is the recommended newer primitive for model responses and agentic tool use.
- Streaming reduces perceived latency by delivering decode output incrementally.
- Chat Completions streaming can include final usage with `stream_options: {"include_usage": true}`.
- OpenAI docs note that interrupted streams may not deliver the final usage chunk.
- Usage accounting distinguishes prompt/input tokens and generated/output tokens.

## Key Figures/Tables to Study
- **Responses Overview:** Inputs, outputs, tools, and stateful response objects.
- **Migrate to Responses:** Differences between Chat Completions messages and Responses items.
- **Chat streaming event reference:** Final chunk behavior when `include_usage` is enabled.
- **Usage fields in API objects:** Input/prompt tokens, output/completion tokens, total tokens.

## Technical Details
Streaming maps naturally to the inference timeline:

```text
request accepted
prefill runs over input tokens
decode begins
events/chunks stream as output tokens or items become available
final event/chunk carries completion metadata when available
```

For Chat Completions streams, the documented `stream_options: {"include_usage": true}` behavior is that most chunks carry null usage and the final chunk carries token usage for the full request. If the stream is interrupted or cancelled, the final usage chunk may be missing.

Applications should therefore persist partial text separately from final accounting, attach request IDs for reconciliation, and avoid treating streamed deltas as a completed answer until the terminal event has arrived.

## Connections
- [[prefill-vs-decode]]: streaming begins after enough prefill work has completed to produce the first token.
- [[batching-for-inference]]: hosted APIs typically stream from continuously batched decode loops.
- [[gpt-3-language-models-are-few-shot-learners]]: few-shot prompts increase input-token usage and TTFT.
- [[kv-cache-memory-formula]]: provider-side serving must hold KV state while streamed generation proceeds.
