<!-- scope: synthesis source card for continuous batching / iteration-level scheduling
     deps: [[orca]]
     see-also: [[pagedattention]], [[sarathi-serve]], [[vtc]]
-->

# Continuous Batching
- **Core Insight:** Rebuild the active batch at token-iteration granularity so finished requests release capacity and waiting requests can enter without waiting for a static batch to drain.
- **Guideline:** Treat continuous batching as a scheduler pattern, not a single paper; combine it with KV-block accounting, token budgets, and prefill/decode policy.
- **Authors:** Synthesis around Orca, vLLM, Hugging Face TGI/Transformers serving docs, and related systems
- **Year:** 2022-present
- **URL:** https://www.usenix.org/conference/osdi22/presentation/yu ; https://github.com/huggingface/text-generation-inference ; https://huggingface.co/docs/transformers/main//continuous_batching
- **Relevant topics:** dynamic batching, iteration-level scheduling, in-flight batching, KV cache, token budgets, prefill/decode scheduling

## Abstract
There is no single canonical "continuous batching" paper. Orca introduced the core research term "iteration-level scheduling"; production systems and docs often call the same family of techniques continuous batching or in-flight batching. The scheduler repeatedly forms a batch from currently running decode requests plus newly admitted prompts, constrained by maximum sequences, maximum batched tokens, and KV-cache memory. The goal is high GPU utilization without forcing all requests in a batch to finish together.

## Key Contributions
- Converts batching from a request-level decision to an every-iteration decision.
- Reduces padding and head-of-line blocking for variable output lengths.
- Allows streaming responses to finish independently.
- Makes KV-cache allocation a scheduler constraint because active membership changes continuously.
- Supports per-request sampling parameters if logits processing is handled outside shared model compute.
- Creates the foundation for fairness, chunked prefill, and SLO-aware policies.

## Key Figures/Tables to Study
- Orca request-level versus iteration-level scheduling timeline.
- vLLM scheduler docs/source: waiting, running, swapped, and token-budget decisions.
- Hugging Face continuous batching docs: manager loop and `ContinuousBatchingConfig`.
- TGI README/features: continuous batching as a production-serving feature.

## Technical Details
In a typical loop, the engine runs prefill for newly admitted requests, then decode steps for running requests. Each step checks for EOS, max-token limits, cancellations, and available KV blocks. Completed requests free KV cache and sequence slots. Waiting requests enter when token budget and cache budget allow.

The scheduler usually has multiple caps: `max_num_seqs`, `max_num_batched_tokens`, maximum model length, and available KV blocks. These caps are more important than nominal request count because a few long prompts can consume the same memory as many short chats.

Continuous batching alone does not decide how to mix prefill and decode. Systems differ: some prioritize prefills to reduce TTFT, some prioritize decodes to protect TPOT, and chunked-prefill systems split long prompts so decode traffic is not stalled.

## Connections
- [[orca]] is the foundational research source.
- [[pagedattention]] gives continuous batching practical KV memory management.
- [[sarathi-serve]] refines the prefill/decode interaction with chunked prefill.
- [[vtc]] adds fair client selection to the same scheduler family.
