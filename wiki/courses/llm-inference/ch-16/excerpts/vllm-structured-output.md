---
chapter: ch-16
course: llm-inference
phase: read
excerpt_of: "vLLM V1 Structured Output: vllm/v1/structured_output/"
source_url: https://github.com/vllm-project/vllm/tree/main/vllm/v1/structured_output
created_at: "2026-05-21"
---

# Excerpt: vLLM V1 Structured Output

**Source files:**
- `vllm/v1/structured_output/__init__.py` (manager + backend dispatch)
- `vllm/v1/structured_output/backend_xgrammar.py` (xgrammar integration)
- `vllm/v1/structured_output/backend_outlines.py` (outlines integration)
- `vllm/v1/structured_output/request.py` (per-request grammar state)

**Raw-data source:** [[raw-data/vllm-structured-output]]

---

## The constraint surface

vLLM exposes five constraint types via OpenAI-compatible request fields:

```python
# Choices (output must be one of these strings)
{"guided_choice": ["yes", "no", "maybe"]}

# Regex (output must match this regex)
{"guided_regex": r"\d{4}-\d{2}-\d{2}"}    # e.g. a date

# JSON schema (output must satisfy this schema)
{"guided_json": {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}}

# Context-free grammar (output must satisfy this EBNF)
{"guided_grammar": "?start: NUMBER \"+\" NUMBER \"=\" NUMBER\nNUMBER: /\\d+/"}

# Structural tag
{"structural_tag": "<answer>...</answer>"}
```

All are passed as extra fields to OpenAI Chat Completions / Completions requests via `extra_body`.

---

## The bitmask pipeline

For each constrained request, at each decode step:

```
1. Grammar state is advanced from previous state given last committed token.
2. Grammar emits the set of allowed next tokens.
3. Construct vocab-sized boolean mask: True for allowed, False for forbidden.
4. Apply mask to logits BEFORE sampling: logits[~mask] = -inf
5. Sample normally (top-p, top-k, temperature, etc.)
```

The bitmask is the **only mechanism**. No grammar-aware sampling; no rejection. Just zero out the forbidden columns of the logits.

---

## The xgrammar backend (the default in 2025+)

xgrammar (Ye et al. 2025) is the fast grammar-bitmask compiler:

```python
import xgrammar as xgr

# At request admission time:
compiler = xgr.GrammarCompiler(tokenizer_info)
grammar = compiler.compile_json_schema(schema_json)
# or compiler.compile_regex(regex), compiler.compile_grammar(ebnf)

# Per request, persistent state:
matcher = xgr.GrammarMatcher(grammar)

# Per decode step:
bitmask = matcher.compute_bitmask()  # shape: (vocab_size,) bool
logits[~bitmask] = float("-inf")
sampled_token = sample(logits)
matcher.accept_token(sampled_token)  # advance state
```

Per-step bitmask compute: ~5-15 µs. Per-request grammar compile: ~50-500 ms (cached by schema hash, so the same schema across many requests pays compile once).

### Why xgrammar is fast

xgrammar precomputes a **per-state token-allowed bitmask** at grammar-compile time. The runtime cost is just: look up the current state, return the precomputed bitmask. No per-step grammar walk.

For very complex grammars (1000+ productions), the precomputation cost grows but the per-step cost stays roughly constant.

---

## The outlines backend (legacy)

outlines (Willard & Louf 2023) was the original backend. Slower per-step (~50-200 µs) because it walks the FSM dynamically, but supports a wider grammar surface and is more permissive about JSON schema edge cases.

```bash
vllm serve <model> --guided-decoding-backend outlines
```

Still useful for grammars that xgrammar doesn't support. As of 2026, xgrammar covers ~95% of real-world JSON schemas; outlines handles the rest.

---

## Manager + scheduler integration

The `StructuredOutputManager` is owned by the engine, not the scheduler:

```python
class StructuredOutputManager:
    def __init__(self, vllm_config):
        self.backend = self._select_backend(vllm_config.guided_decoding_backend)
        self.compiled_grammar_cache = {}   # schema_hash → compiled grammar

    def init_request(self, request, structured_output_request):
        schema_hash = hash_schema(structured_output_request)
        if schema_hash not in self.compiled_grammar_cache:
            self.compiled_grammar_cache[schema_hash] = self.backend.compile(structured_output_request)
        grammar = self.compiled_grammar_cache[schema_hash]
        request.grammar_matcher = self.backend.make_matcher(grammar)

    def compute_bitmasks(self, scheduled_reqs) -> torch.Tensor:
        # Stack per-request bitmasks into (num_reqs, vocab_size) tensor
        bitmasks = [req.grammar_matcher.compute_bitmask() for req in scheduled_reqs if req.uses_grammar]
        return torch.stack(bitmasks)

    def accept_token(self, request, token_id):
        request.grammar_matcher.accept_token(token_id)
```

The scheduler calls `compute_bitmasks(scheduled_reqs)` during its `schedule()` call and embeds the bitmask in `SchedulerOutput.grammar_bitmask`. The model runner applies the mask in the sampler.

---

## JSON schema worked example

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="...")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    messages=[{
        "role": "user",
        "content": "Extract structured data: John Smith is 35 years old."
    }],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
            "required": ["name", "age"]
        }
    }
)
print(response.choices[0].message.content)
# {"name": "John Smith", "age": 35}    # guaranteed schema-valid
```

The grammar guarantees *syntactic* validity (output is parseable JSON satisfying the schema). It does **not** guarantee semantic correctness — the model could produce `{"name": "Jane Doe", "age": 35}` and that would still be schema-valid.

---

## Performance impact

Per-step overhead for xgrammar bitmask:

| Schema complexity | Per-step overhead |
|--------------------|-------------------|
| Simple choice (≤5 options) | ~3 µs |
| Date regex | ~5 µs |
| JSON schema (~10 properties) | ~10-20 µs |
| Complex nested JSON (~100 properties) | ~30-50 µs |
| Big EBNF (Python AST) | ~50-100 µs |

On a 30 ms decode step, even the worst case adds ~0.3% overhead. Compile time dominates only for first-of-schema requests (~100-500 ms).

---

## Pitfalls

- **Schema cache must hit.** First request with a new schema pays compile. If clients send distinct schemas every request (e.g. randomly generated), compile dominates. Stabilize schemas; pre-warm via a dummy request.
- **Bitmask is computed on CPU, transferred to GPU per step.** For tiny vocab (<32k), this is fast (~10 µs). For large vocab (>200k), bitmask transfer can be a bottleneck — xgrammar 2025+ has CUDA-side bitmask computation for this case.
- **Grammar state must be reset on regenerate.** If a request is rejected and resubmitted, the grammar matcher must be reinitialized. vLLM does this automatically.
- **Streaming with grammars**: the bitmask applies per-token; streaming yields each token after grammar advance. This means streaming output is always grammar-valid prefix-wise.
- **Speculative decoding + structured output**: spec-dec proposals must all satisfy the grammar at each verification position. Drafter ignores the grammar (it can't easily mask), so most spec-dec proposals get rejected for grammar reasons → spec-dec barely helps on constrained outputs. Disable spec-dec when structured-output is on.

---

## Connections

- [[excerpts/vllm-scheduler]] — calls `compute_bitmasks()` each step.
- [[excerpts/vllm-production-knobs]] — `--guided-decoding-backend xgrammar` (default) vs outlines.
- [[sglang-structured-output]] — SGLang's analogous integration (ch-17).
- [[ch-16]] — parent chapter.
