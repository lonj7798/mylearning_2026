<!-- scope: Hugging Face assisted generation blog and Transformers assisted-decoding API
     deps: fast-inference-from-transformers-via-speculative-decoding
     see-also: prompt-lookup-decoding, speculative-decoding
-->

# Assisted Generation: A New Direction Toward Low-Latency Text Generation
- **Core Insight:** A small assistant model can cheaply propose candidate tokens that the larger model verifies in parallel, reducing end-user latency with the normal `generate()` workflow.
- **Guideline:** Use Hugging Face assisted generation when main and assistant models are tokenizer-compatible, the assistant is much faster, and batch size is small enough for speculative decoding support.
- **Authors:** Joao Gante
- **Year:** 2023
- **URL:** https://huggingface.co/blog/assisted-generation ; https://huggingface.co/docs/transformers/assisted_decoding
- **Relevant topics:** assisted generation, speculative decoding, Transformers `generate`, assistant model, prompt lookup, latency

## Abstract
The Hugging Face blog explains assisted generation as a practical speculative-decoding interface for Transformers. A smaller assistant model proposes several tokens, then the main model verifies them with a single forward pass. The post motivates the method from the latency of autoregressive generation and reports up to 10x latency reduction on commodity hardware for favorable cases. The current documentation also covers prompt lookup decoding and universal assisted decoding variants.

## Key Contributions
- Connects speculative decoding theory to the common `model.generate()` API.
- Explains why a forward pass over a sequence can verify many proposed tokens.
- Provides greedy and sampling assisted-generation examples.
- Documents practical constraints such as tokenizer compatibility and lack of batched-input support for speculative decoding.
- Extends the user-facing family to prompt lookup and universal assisted decoding in official docs.

## Key Figures/Tables to Study
- "Language decoder forward pass, revisited": explains verification using shifted logits.
- Greedy assisted generation section: clearest implementation flow.
- Sampling assisted generation section: connects to probability-corrected speculation.
- Assisted decoding docs: current API examples for `assistant_model` and `prompt_lookup_num_tokens`.

## Technical Details
In Transformers, assisted decoding is selected by passing an `assistant_model` to `generate()`. The assistant drafts candidate tokens. The main model processes the prompt plus candidates and verifies which drafted tokens agree with its own predictions or sampling correction. Accepted tokens are appended, and the loop continues from the first rejection.

The docs note that speculative decoding works best when the assistant is significantly smaller and uses the same tokenizer. Prompt lookup replaces the assistant model with n-gram matches from the prompt, which is useful for copying-heavy workloads.

## Connections
- [[fast-inference-from-transformers-via-speculative-decoding]] is the key lossless sampling paper behind the method.
- [[prompt-lookup-decoding]] is the no-assistant mode exposed in the same documentation.
- [[medusa]], [[eagle]], and [[self-speculative-decoding]] are alternative proposer designs for the same draft/verify bottleneck.
- [[speculative-decoding]] is the earlier draft/verify paper lineage behind the terminology.
- [[multi-token-prediction-inference]] connects assisted generation to model-native future-token heads.
