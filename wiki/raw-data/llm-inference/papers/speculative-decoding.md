<!-- scope: original SpecDec paper for draft-and-verify seq2seq generation
     deps: transformer-inference-loop
     see-also: fast-inference-from-transformers-via-speculative-decoding, medusa, eagle
-->

# Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation
- **Core Insight:** A fast drafter can propose multiple tokens and the target model can verify them in parallel, reducing sequential autoregressive steps.
- **Guideline:** Use speculative decoding when a cheap, high-agreement draft mechanism exists and target-model verification over multiple tokens is faster than decoding those tokens one by one.
- **Authors:** Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2203.16487
- **Relevant topics:** speculative execution, draft model, verification, seq2seq generation, greedy decoding, NMT

## Abstract
This paper introduces SpecDec, an early formal application of speculative execution to autoregressive sequence generation. A non-autoregressive or faster drafter proposes future tokens, then the original autoregressive model verifies the draft in parallel. The target model remains the authority, so accepted tokens match what it would have produced under the supported decoding setting.

## Key Contributions
- Frames speculative decoding as draft generation followed by parallel target verification.
- Introduces a specialized Spec-Drafter and Spec-Verification procedure.
- Shows speedups for sequence-to-sequence generation without retraining the target model.
- Analyzes how verification accepts a prefix of drafted tokens and resumes after rejection.
- Establishes terminology later reused by LLM speculative decoding systems.

## Key Figures/Tables to Study
- Draft-then-verify algorithm: the central control flow for speculative execution.
- Verification diagrams: show how multiple proposed tokens are checked in one target pass.
- Latency tables: compare speedups against standard autoregressive decoding.
- Error/quality comparisons: confirm when the accelerated path preserves output behavior.

## Technical Details
The method separates the generation loop into a proposal phase and a verification phase. The drafter predicts several future tokens cheaply. The target autoregressive model then evaluates those positions in parallel, because a transformer forward pass over a draft sequence produces next-token predictions at every position.

If the draft prefix is accepted, the system commits multiple tokens for one target-model call. If a token is rejected, the target model's token is used and a new drafting round begins. The speedup depends on the drafter cost, the average accepted prefix length, and the target model's ability to verify many tokens efficiently.

## Connections
- [[fast-inference-from-transformers-via-speculative-decoding]] develops the lossless sampling formulation now commonly cited for LLMs.
- [[hf-assisted-generation]] operationalizes the idea in Hugging Face `generate()`.
- [[medusa]], [[eagle]], and [[self-speculative-decoding]] replace the separate draft model with heads, feature predictors, or skipped layers.
- [[prompt-lookup-decoding]] is a heuristic drafter variant for copying-heavy prompts.
- [[lookahead-decoding]] is a no-drafter parallel decoding alternative.
