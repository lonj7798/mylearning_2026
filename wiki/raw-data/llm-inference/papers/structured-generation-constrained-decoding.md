<!-- scope: structured generation via constrained decoding, JSON schema, regex, and context-free grammar token masking
     deps: [[hf-generation-strategies]]
     see-also: [[vllm-structured-output]], [[sglang-structured-output]], [[flashinfer]]
-->

# Structured Generation and Constrained Decoding
- **Core Insight:** Structured output becomes reliable when the decoder masks invalid next tokens according to a schema, regex, finite-state machine, or context-free grammar instead of asking the model to "please output JSON".
- **Guideline:** Use constrained decoding for syntax guarantees, then still validate semantics; grammar masks can force format correctness but cannot prove the answer is true or useful.
- **Authors:** XGrammar / MLC team, LMQL authors, Guidance authors, Outlines maintainers, and structured-generation community
- **Year:** 2022–2026
- **URL:** https://arxiv.org/abs/2411.15100 ; https://github.com/mlc-ai/xgrammar ; https://lmql.ai/ ; https://github.com/guidance-ai/guidance
- **Relevant topics:** structured output, constrained decoding, JSON schema, regex, CFG, token masks, tool calling, agents

## Abstract
This page is a synthesis card for constrained decoding rather than a single paper. The core runtime idea is to compile a structure specification, such as JSON schema, regex, EBNF, or a context-free grammar, into a state machine or grammar engine. At each decode step, the runtime computes the valid next-token set and masks invalid tokens before sampling. Modern engines such as XGrammar, Outlines, Guidance, LMQL, vLLM guided decoding, and SGLang structured output use variants of this idea to make structured outputs parseable in production.

## Key Contributions
- Replaces prompt-only formatting with a hard inference-time constraint.
- Converts output specs into token masks applied before sampling.
- Supports common production schemas: JSON, enum choices, regex, EBNF, function/tool calls, and agent action formats.
- Makes grammar backends part of the serving engine, so batching and streaming can still work.
- Exposes the key caveat: syntax validity is not semantic correctness.

## Key Figures/Tables to Study
- XGrammar architecture: precompiled grammar/token-mask cache for low-overhead structured generation.
- LMQL language examples: constraints embedded into LLM programs.
- Guidance examples: constrained generation blocks and tool/function shapes.
- vLLM/SGLang structured-output docs: how grammar backends appear in production servers.

## Technical Details

### Runtime loop
At generation step `t`:
1. Model returns logits over the vocabulary.
2. Constraint backend updates its parser/state from previous tokens.
3. Backend returns valid token ids for the next state.
4. Runtime sets invalid-token logits to `-inf`.
5. Sampling or greedy selection proceeds over the valid set only.

### Practical trade-offs
- Large schemas can have compile-time overhead.
- Tokenization matters: a string-level grammar must be mapped to tokenizer pieces.
- Some constraints are finite-state; full CFG support needs stack-like parser state.
- Strong constraints can lower answer quality if they force unnatural token paths.
- Validation and retry are still needed for semantic constraints such as dates, IDs, and business rules.

## Connections
- [[vllm-structured-output]] — vLLM production integration.
- [[sglang-structured-output]] — SGLang grammar and structured-output stack.
- [[hf-generation-strategies]] — unconstrained sampling baseline.
- [[flashinfer]] — serving kernels must coexist with dynamic structured-generation constraints.
