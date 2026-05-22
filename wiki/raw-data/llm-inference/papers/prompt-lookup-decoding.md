<!-- scope: prompt lookup / n-gram assisted decoding in Hugging Face Transformers
     deps: fast-inference-from-transformers-via-speculative-decoding
     see-also: hf-assisted-generation, lookahead-decoding
-->

# Prompt Lookup Decoding
- **Core Insight:** When the output is likely to copy from the prompt, n-gram matches in the prompt can serve as a zero-model drafter.
- **Guideline:** Enable prompt lookup decoding for summarization, editing, retrieval-heavy, and code tasks with repeated spans; do not expect broad gains on open-ended generation.
- **Authors:** Hugging Face Transformers implementation and documentation
- **Year:** 2024
- **URL:** https://huggingface.co/docs/transformers/assisted_decoding ; https://github.com/huggingface/transformers
- **Relevant topics:** prompt lookup, n-gram speculation, assisted decoding, zero-drafter speculation, copying workloads

## Abstract
This is a source page for an official implementation/docs artifact rather than a standalone paper. Prompt lookup decoding is an assisted-decoding mode in Hugging Face Transformers that proposes candidate tokens by finding matching n-grams in the existing prompt. The target model verifies those candidates like speculative decoding, but no assistant model is loaded.

## Key Contributions
- Adds a lightweight speculation path controlled by `prompt_lookup_num_tokens`.
- Uses prompt n-gram matches as the draft source.
- Avoids assistant-model memory, tokenizer compatibility, and loading costs.
- Fits workloads where generation copies or lightly edits prompt content.
- Exposes the method through the same `generate()` family as assisted decoding.

## Key Figures/Tables to Study
- Hugging Face assisted decoding docs: prompt lookup example with `prompt_lookup_num_tokens`.
- Transformers generation configuration code: see how assisted decoding is selected.
- Pull requests/issues around prompt lookup: useful for understanding constraints and intended workloads.
- Benchmarks should be workload-specific; the docs emphasize method behavior more than universal speedup.

## Technical Details
Prompt lookup searches the prompt for n-grams matching the current suffix and proposes the following tokens from the matched prompt span. The target model then scores the drafted continuation in a parallel forward pass and accepts the valid prefix.

The method is cheap because the proposer is just a string/token lookup. It can fail to help when generations are novel, when prompt repetitions are rare, or when the verification overhead exceeds accepted-token savings.

## Connections
- [[hf-assisted-generation]] is the broader Hugging Face interface for assistant-model and prompt-lookup modes.
- [[lookahead-decoding]] also avoids a separate draft model but uses a model-internal lookahead algorithm.
- [[fast-inference-from-transformers-via-speculative-decoding]] supplies the verification logic that prompt lookup reuses.
- [[speculative-decoding]] is the umbrella draft/verify method family.
- [[self-speculative-decoding]] is another no-extra-model option with different compute tradeoffs.
