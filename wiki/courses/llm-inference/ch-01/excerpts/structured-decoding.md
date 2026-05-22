---
chapter: ch-01
course: llm-inference
phase: read
excerpt_of: "Structured Generation and Constrained Decoding (XGrammar / Outlines / Guidance / LMQL synthesis)"
source_url: https://arxiv.org/abs/2411.15100
created_at: "2026-05-21"
---

# Excerpt: Constrained decoding — schema as a logit mask

**Authors:** XGrammar / MLC team, Outlines maintainers, Guidance authors, LMQL authors
**Year:** 2022–2024 (XGrammar 2024 is the consolidating reference)
**Venue:** arXiv + open-source projects
**URLs:** https://arxiv.org/abs/2411.15100 ; https://github.com/mlc-ai/xgrammar ; https://github.com/dottxt-ai/outlines ; https://github.com/guidance-ai/guidance
**Raw-data source:** [[raw-data/structured-generation-constrained-decoding]]

---

## The runtime mechanism in one box

At each decode step `t`:

```python
def constrained_decode_step(logits, grammar_state, grammar):
    # 1. Grammar tells us which token ids are valid in the current state.
    valid_ids = grammar.valid_next_tokens(grammar_state)     # set[int]
    # 2. Build a mask: 0 for valid, -inf for invalid.
    mask = torch.full_like(logits, -float("inf"))
    mask[list(valid_ids)] = 0.0
    # 3. Apply mask BEFORE the sampler.
    masked_logits = logits + mask
    next_id = sample(masked_logits, **sampling_params)
    # 4. Advance the grammar state with the chosen token.
    grammar_state = grammar.advance(grammar_state, next_id)
    return next_id, grammar_state
```

The model never produces an out-of-grammar token because every alternative has probability zero after the mask. This is the whole trick. The grammar engine is responsible for steps 1 and 4; everything else is the standard decoder loop.

---

## Grammar representations

Different constraint types compile to different state machines:

| Constraint | Automaton | Per-step cost |
|---|---|---|
| Regex | DFA (deterministic FSA) | O(1) state lookup |
| JSON schema | DFA (preprocessed) | O(1) state lookup |
| Enum / oneOf | DFA | O(1) state lookup |
| EBNF / CFG | Pushdown automaton | O(stack depth) |
| Context-sensitive | (not supported) | n/a |

JSON schema compiles to a DFA because well-formed JSON is *almost* regular — keys, types, and structure can be enumerated. Free-form values (strings) are handled with sub-grammars. EBNF requires stack state for nested matched brackets.

---

## Why a naive prompt instruction fails

Asking the model "Please output JSON" works most of the time but fails in ~1–5% of cases with:

- Trailing commas in arrays: `[1, 2, 3,]`
- Single quotes for strings: `{'name': 'Alice'}`
- Markdown fences wrapping JSON: ` ```json\n{...}\n``` `
- Prose preface: `"Sure! Here's the JSON: {...}"`
- Premature truncation when the model hits `max_tokens` mid-object.

A constrained decoder eliminates all five — the grammar literally does not allow them as token transitions.

---

## The tokenizer-alignment problem

Grammars naturally operate at the character level. Logits operate at the token level (BPE / SentencePiece). The naive bridge:

```
for token_id in vocab:
    decode token_id to chars
    check if chars are accepted by grammar starting at current state
```

…is `O(|vocab|)` per step (50k–256k tokens). At 50 tokens/sec decode this dominates kernel time. Mature backends precompute a per-state allowed-token bitmask:

```
state_mask[s] = bitmask of token ids that the grammar accepts in state s
```

XGrammar's headline result: per-step mask lookup drops from ~10 ms (naive) to <100 µs (precompiled). For a typical JSON schema with ~50 states the total mask table is a few MB.

---

## Compilation overhead

For a fresh grammar:

| Engine | Compile time (medium JSON schema) | Per-step overhead |
|---|---:|---:|
| Outlines (FSA) | 0.5–2 s | ~50 µs |
| XGrammar | 0.1–0.5 s | ~10 µs |
| llama.cpp grammars | <0.1 s | ~5 µs |

The compile cost is amortized across all requests that use the same schema, so for hot APIs (e.g. tool-calling endpoints) it's negligible. For one-off schemas it can dominate TTFT.

---

## Production integration points

- **vLLM**: `guided_json=schema`, `guided_regex=pattern`, `guided_grammar=ebnf_str`, `guided_choice=enums`. Backend selectable via `guided_decoding_backend=xgrammar|outlines`. See [[raw-data/vllm-structured-output]].
- **SGLang**: `regex=...` constraints inside its frontend DSL; JSON via `json_schema=`. See [[raw-data/sglang-structured-output]].
- **OpenAI Responses API**: `response_format: {"type": "json_schema", "json_schema": {...}}` enforces structure server-side using their proprietary grammar backend.
- **Anthropic Messages API**: `tools` parameter; tool-input schemas are enforced via structured decoding internally.

---

## Limitations

1. **Quality degradation on adversarial grammars.** Forcing the model into low-probability token paths can collapse semantic quality. A grammar that constrains every field of a 50-field schema may produce technically-valid-but-semantically-empty JSON. Validate semantics separately.
2. **Streaming + JSON**: partial JSON mid-stream is not parseable. Either buffer until completion or use a streaming JSON parser (e.g. `ijson`).
3. **Tokenizer mismatch across models**: a grammar pre-compiled for Llama-3's tokenizer is not portable to Qwen-3. Compile per (grammar, tokenizer) pair.
4. **CFG limitations**: grammars are context-free; semantic constraints (date must be valid, ID must exist in DB) need post-validation.

---

## Connections

- [[excerpts/sampling-strategies]] — the mask is applied to logits *before* the sampler runs; otherwise standard sampling pipeline.
- [[raw-data/vllm-structured-output]] — production wiring inside vLLM.
- [[raw-data/sglang-structured-output]] — SGLang's DSL-integrated approach.
- [[ch-16]] — vLLM scheduler interactions with structured-output bitmasks.
- [[raw-data/flashinfer]] — kernels must coexist with dynamic per-request grammar masks.
