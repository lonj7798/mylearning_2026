---
chapter: ch-17
course: llm-inference
phase: read
excerpt_of: "SGLang structured output — grammar backends and constrained decoding"
source_url: https://docs.sglang.ai/advanced_features/structured_outputs.html
created_at: "2026-05-21"
---

# Excerpt: Structured Output in SGLang — XGrammar + Outlines + llguidance

**Authors:** SGLang project + XGrammar (MLC), Outlines (Outlines.dev), llguidance (Microsoft)
**Year:** 2024–present
**URLs:** https://docs.sglang.ai/advanced_features/structured_outputs.html
**Raw-data source:** [[raw-data/sglang-structured-output]]

---

## The three backends

| Backend | Default? | Strength | Weakness |
|---------|---------|----------|----------|
| **XGrammar** | yes | Fastest mask computation; precomputed per-grammar token-mask tables | Less feature-rich for very complex EBNF |
| **Outlines** | no | Broadest grammar surface (Lark, JSON Schema with refs, complex regex) | Per-step mask computation is slower |
| **llguidance** | no | Microsoft backend; tight chat-template integration; tool-call structural-tag native | Smaller user base; less battle-tested at extreme load |

Select via `--grammar-backend xgrammar|outlines|llguidance`.

---

## Constraint families

Four constraint types accepted by both OpenAI-compat and native APIs:

```python
sampling_params = {
    "json_schema": {...},        # JSON conforming to a schema
    "regex": r"\d{3}-\d{4}",     # regex match
    "ebnf": "<grammar>",         # arbitrary EBNF grammar
    "structural_tag": {          # tool-call / function-call shape
        "begin": "<tool_call>",
        "end": "</tool_call>",
        "schema": {...},
    },
}
```

**Exactly one** constraint per request; mixing is undefined behavior.

---

## How constraints reach the runtime

OpenAI-compatible surface:

```python
client.chat.completions.create(
    model="llama3",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "person", "schema": person_schema},
    },
)
```

Native surface (full constraint vocabulary):

```python
import requests
r = requests.post("http://server:30000/generate", json={
    "text": prompt,
    "sampling_params": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "json_schema": person_schema,
    },
})
```

---

## Per-step behavior

For each decoded token:

```
1. Sample logits from model forward.
2. Backend computes the allowed-token mask for the current grammar state.
3. Mask is applied to logits (set disallowed → -inf).
4. Standard top-k/top-p/temperature sampling on masked logits.
5. Backend advances grammar state to accept the chosen token.
```

XGrammar's edge: step 2 is a table lookup per (state, token) — precomputed at schema-compile time. Outlines computes the FSA transition on the fly.

---

## Interaction with RadixAttention

Grammar **state** is per-request — it cannot be shared across requests.

Grammar **definitions** are often shared — many requests use the same JSON schema (e.g. a tool-call API). The XGrammar backend caches the compiled mask table per schema; subsequent requests with the same schema skip compilation.

Combined with RadixAttention prefix cache: an agent that issues identical tool-call shaped requests pays:
- Grammar compile: once per schema (cached across requests)
- Prefill: cached prefix matched, suffix only
- Decode: per-step mask + sample

Net speedup vs naive stateless serving: typically **3–6×** for tool-heavy agents.

---

## The decode-speed penalty

A complex grammar costs decode speed. Reported TPOT overhead by grammar complexity:

| Grammar type | TPOT overhead vs unconstrained |
|--------------|-------------------------------:|
| Simple JSON schema (no refs, flat) | < 5 % |
| Nested JSON with optional fields | 5–15 % |
| Complex EBNF (e.g. SQL grammar) | 20–40 % |
| Regex with backtracking-prone patterns | 30–60 % |

The trade-off: strict grammar guarantees vs faster generation + post-hoc validation. For latency-critical paths, post-hoc validation often wins; for correctness-critical paths (regex match for IDs, SQL syntax), constraint-during-decode is non-negotiable.

---

## Common pitfalls

- **`response_format={"type": "json"}` without a schema.** That's "JSON mode" without grammar guarantees — model usually emits valid JSON but no syntactic enforcement. For real guarantees, use `json_schema`.
- **Sending a 10k-line schema.** XGrammar must compile a token-mask table; very large schemas have many states and large compile cost. Cache + reuse schemas across requests.
- **Switching grammar mid-generation.** Backends don't support this; finish the request, start a new one.
- **Expecting semantic validity from syntactic constraints.** A JSON-schema-conforming output can still be factually wrong; constrained decoding only enforces structure.

---

## Connections

- [[excerpts/sglang-radixattention]] — the prefix cache that combines with grammar-mask caching.
- [[excerpts/sglang-scheduler]] — where grammar context is attached to requests.
- [[excerpts/vllm-structured-output]] (ch-16) — vLLM's parallel implementation (xgrammar + outlines).
- [[ch-01]] — constrained decoding foundations.
