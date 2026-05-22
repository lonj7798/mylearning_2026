---
chapter: ch-01
course: llm-inference
phase: read
excerpt_of: "OpenAI Streaming and Token Usage docs (Responses + Chat Completions APIs)"
source_url: https://developers.openai.com/api/reference/responses/overview
created_at: "2026-05-21"
---

# Excerpt: Streaming and token accounting at the API boundary

**Authors:** OpenAI API documentation (2026 snapshot)
**Year:** 2023–2026
**Venue:** OpenAI developer docs
**URLs:** https://developers.openai.com/api/reference/responses/overview ; https://developers.openai.com/api/docs/guides/migrate-to-responses ; https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
**Raw-data source:** [[raw-data/openai-streaming-and-token-usage]]

---

## The SSE event timeline

For a streaming chat request, the engine emits Server-Sent Events as decode produces tokens:

```
event: response.created               {"id": "resp_…", "status": "in_progress", …}
event: response.output_item.added     {"item": {"type": "message", "role": "assistant", …}}
event: response.output_text.delta     {"delta": "Hello"}
event: response.output_text.delta     {"delta": " world"}
event: response.output_text.delta     {"delta": "!"}
event: response.output_text.done      {"text": "Hello world!"}
event: response.completed             {"usage": {"input_tokens": 12,
                                                   "output_tokens": 3,
                                                   "total_tokens": 15},
                                         "finish_reason": "stop"}
```

The `delta` events arrive one per decoded token (modulo per-chunk batching by the server). The `completed` event is terminal and carries the authoritative token count.

For the older Chat Completions endpoint, set `stream_options: {"include_usage": true}` and the final chunk (the one whose `choices` array is empty) will carry the `usage` object. Without this flag, you have to count tokens client-side.

---

## The authoritative timeline mapped to engine phases

```
client                                       server                  engine
------                                       ------                  ------
POST /v1/responses (stream=true)  →
                                             route, tokenize
                                             admit to scheduler   →   add to next-step batch
                                                                       PREFILL forward pass
                                                                       write KV cache
                                             ← first logit ready ←
                                             sample, emit delta   →   DECODE step 2
                          ← SSE delta ←                                ...
                                             sample, emit delta   →   DECODE step N
                                             EOS / max_tokens     →   leave batch
                          ← SSE completed ←  emit usage
```

The TTFT (time-to-first-token) metric is measured at the client's first `delta` arrival. TPOT (time-per-output-token) is the median inter-`delta` interval. Both are decode-loop properties from ch-03 and ch-04.

---

## Three `finish_reason` cases

| value | meaning | what your client should do |
|---|---|---|
| `"stop"` | Natural EOS token or stop-string matched | Accept the response as complete |
| `"length"` | Hit `max_tokens` cap | Show truncated; offer "continue" |
| `"content_filter"` | Safety classifier intercepted | Display rejection message |
| `"tool_calls"` | Model emitted a structured tool/function call | Parse `tool_calls`, dispatch tools, send results back |

A 500-token streamed response with `finish_reason="length"` is *not* the same outcome as one with `finish_reason="stop"`. Production apps that ignore this end up showing mid-sentence truncations to users.

---

## Interrupted streams

The docs explicitly warn: if the connection drops mid-stream, the final `usage` chunk may never arrive. The server-side request continues until cancellation propagates; tokens already decoded are lost from the client's perspective.

Defensive client patterns:

```python
async def call_with_partial_persistence(req):
    partial = []
    usage = None
    try:
        async for event in client.responses.stream(req):
            if event.type == "response.output_text.delta":
                partial.append(event.delta)
            elif event.type == "response.completed":
                usage = event.usage
                break
    except Exception as e:
        log.warning("stream interrupted at %d chars; req_id=%s",
                    sum(len(p) for p in partial), req.id)
    return "".join(partial), usage  # usage may be None
```

For billing/reconciliation, persist `request_id` separately and fetch the canonical usage out-of-band via the `/v1/responses/{id}` GET when available.

---

## Token counting is tokenizer-dependent

The same 200-character user prompt produces wildly different token counts depending on the model's tokenizer:

| Text | tiktoken cl100k_base | Llama-3 tokenizer | Qwen-3 tokenizer |
|---|---:|---:|---:|
| "Hello world" | 2 | 2 | 2 |
| "The quick brown fox jumps over the lazy dog" | 9 | 10 | 9 |
| Korean: "안녕하세요 세계" | 7 | 6 | 5 |
| JSON: `{"name":"Alice","age":30}` | 11 | 12 | 11 |

Always count with the model's own tokenizer. Estimating English at "~4 chars / token" works as a rough heuristic but underestimates non-Latin scripts by 2–3×.

---

## Common pitfalls

- **Assuming `usage` always arrives**: it does not on interrupted streams. Persist partial tokens separately.
- **Treating a `delta` as a token boundary**: servers may batch multiple decoded tokens into one SSE chunk for efficiency. Use the final `output_text` event for canonical text.
- **Counting chars instead of tokens for rate limits**: rate limits are tokens-per-minute, not chars-per-minute. Use `tiktoken` (OpenAI) or `tokenizers` (HF) for exact counts.
- **Ignoring `finish_reason="length"`**: silent truncation; user sees mid-sentence cutoffs.

---

## Connections

- [[excerpts/autoregressive-loop]] — SSE deltas correspond one-to-one to decode iterations; streaming is the decode loop exposed to the client.
- [[raw-data/batching-for-inference]] — server-side, the loop is continuously batched; SSE per-stream demuxes from a shared batch.
- [[ch-04]] — continuous batching is the matched primitive that makes streaming viable at production QPS.
- [[ch-19]] — TTFT / TPOT / goodput metrics are measured at exactly the SSE boundary described above.
