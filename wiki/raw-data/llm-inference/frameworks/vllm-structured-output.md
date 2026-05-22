<!-- scope: vLLM structured output and guided decoding source page
     deps: [[vllm]]
     see-also: [[sglang-structured-output]]
-->

# vLLM Structured Output
- **Core Insight:** vLLM implements constrained decoding by converting schemas, regexes, choices, or grammars into token masks during generation.
- **Guideline:** Use structured outputs for machine-readability, but benchmark latency because grammar compilation and per-step masking add overhead.
- **Authors:** vLLM project
- **Year:** 2024-present
- **URL:** https://docs.vllm.ai/en/latest/features/structured_outputs.html and https://github.com/vllm-project/vllm/tree/main/vllm/v1/structured_output
- **Relevant topics:** constrained decoding, JSON schema, regex, grammar, OpenAI-compatible API, xgrammar

## Abstract
vLLM supports structured outputs through its OpenAI-compatible online server and offline API. Requests can constrain generation to choices, regexes, JSON schemas, context-free grammars, or structural tags. Internally, structured-output managers maintain per-request grammar state and provide token bitmasks that are applied during sampling.

## Key Contributions
- Exposes constraints through OpenAI-compatible request parameters and vLLM-specific extra parameters.
- Supports multiple constraint types: `guided_choice`, `guided_regex`, `guided_json`, `guided_grammar`, and structural tags.
- Uses grammar backends such as xgrammar or guidance depending on configuration/version.
- Integrates constraints into the scheduler/model-output path through grammar bitmasks.
- Works with streaming and normal generation when the backend can update grammar state incrementally.

## Key Figures/Tables to Study
- Structured outputs docs: examples for OpenAI Chat and Completions APIs.
- `vllm/v1/structured_output`: grammar state, backend integration, and manager code.
- Scheduler calls to structured-output manager: where bitmasks become part of a scheduled step.
- OpenAI protocol definitions: request fields accepted by the server.

## Technical Details
Public API:
- Online: call `/v1/chat/completions` or `/v1/completions` and pass guided decoding fields in request extras.
- Offline: use sampling parameters that carry guided decoding constraints.
- Server launch can select the guided-decoding backend where supported.

Runtime approach:
- Constraint input is parsed into a grammar representation.
- Each active request keeps grammar state that advances with generated tokens.
- Before sampling, vLLM computes an allowed-token bitmask for each constrained request.
- The sampler suppresses invalid next tokens, forcing output to remain in-language for the constraint.

Relevant code/docs:
- Feature docs: https://docs.vllm.ai/en/latest/features/structured_outputs.html
- OpenAI server docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Source directory: https://github.com/vllm-project/vllm/tree/main/vllm/v1/structured_output
- Protocol/source search root: https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai

Strengths:
- Practical API for JSON-schema and regex-constrained serving without changing client architecture.
- Strong fit for extraction, tool-call formatting, classification-by-choice, and agent protocols.
- Backend selection lets vLLM track improvements in external grammar engines.

Limitations:
- Constraints guarantee syntax, not semantic correctness.
- Large or complex schemas can increase compile time and per-token overhead.
- Backend feature parity changes across releases, so exact supported fields should be checked against current docs.

## Connections
- Builds on [[vllm]] serving and [[vllm-scheduler]] step outputs.
- Compare to [[sglang-structured-output]], which also emphasizes native runtime and OpenAI-compatible paths.
- Related to generation fundamentals: logits processors, masks, stop conditions, and tool calling.
