<!-- scope: SGLang structured output support through OpenAI and native APIs
     deps: [[sglang]]
     see-also: [[vllm-structured-output]]
-->

# SGLang Structured Output
- **Core Insight:** SGLang constrains decoding through grammar backends that support JSON schema, regex, EBNF, and structural tags.
- **Guideline:** Prefer XGrammar for general use unless a specific backend feature requires Outlines or llguidance.
- **Authors:** SGLang project
- **Year:** 2024-present
- **URL:** https://docs.sglang.ai/advanced_features/structured_outputs.html
- **Relevant topics:** constrained decoding, JSON schema, regex, EBNF, structural tags, tool calling

## Abstract
SGLang exposes structured output constraints through OpenAI-compatible APIs and the native runtime API. A request can specify one constraint family, such as JSON schema, regex, EBNF, or structural tags. The runtime uses grammar backends to restrict token choices during decoding so generated text follows the requested syntax.

## Key Contributions
- Supports JSON schema, regular expression, and EBNF constraints.
- Supports structural tags for tool/function-call shaped outputs.
- Provides multiple grammar backends: XGrammar by default, plus Outlines and llguidance.
- Works through both OpenAI-compatible `response_format` and native `/generate` sampling parameters.
- Documents examples for plain extraction and tool-call-like formats.

## Key Figures/Tables to Study
- Structured outputs docs: backend list and examples.
- OpenAI-compatible examples: `response_format` for JSON schema and structural tags.
- Native API examples: `sampling_params` carrying `json_schema`, `regex`, `ebnf`, or `structural_tag`.
- Runtime grammar integration in SGLang source under `python/sglang/srt`.

## Technical Details
Public API:
- Launch with default XGrammar or choose another backend using `--grammar-backend`.
- OpenAI path: call Chat Completions with `response_format`.
- Native path: POST to `/generate` with structured constraints inside `sampling_params`.

Runtime approach:
- Parse the user's constraint into a backend grammar or automaton.
- Track grammar state for each request.
- Mask invalid next tokens before sampling.
- Advance grammar state as tokens are accepted.
- Return normal generated text plus metadata such as finish reason, prompt tokens, completion tokens, cached tokens, and latency.

Relevant code/docs:
- Feature docs: https://docs.sglang.ai/advanced_features/structured_outputs.html
- OpenAI API docs: https://docs.sglang.ai/basic_usage/openai_api.html
- Sampling parameters/native API: https://docs.sglang.ai/basic_usage/sampling_params.html
- Source root: https://github.com/sgl-project/sglang/tree/main/python/sglang

Strengths:
- Broad grammar surface, especially EBNF and structural tags.
- Same feature is available through high-level OpenAI clients and low-level runtime calls.
- Good fit with SGLang's programmatic frontend and tool-use workflows.

Limitations:
- Syntax validity does not ensure factual or semantically valid fields.
- Complex constraints can reduce decode speed or interact poorly with some tokenizers.
- Backend-specific grammar syntax and feature support should be checked before relying on portability.

## Connections
- Complements [[sglang-radixattention]] for agent and tool workflows that reuse prompts and require machine-readable outputs.
- Compare with [[vllm-structured-output]] for API shape and backend differences.
- Connects to generation topics: grammar masks, logits processors, and stop conditions.
