<!-- scope: Google speculative decoding paper with lossless sampling correction
     deps: speculative-decoding
     see-also: hf-assisted-generation, eagle, medusa
-->

# Fast Inference from Transformers via Speculative Decoding
- **Core Insight:** Speculative decoding can preserve the exact target-model distribution by accepting draft tokens with a probability correction and sampling a residual distribution on rejection.
- **Guideline:** Use this formulation when accelerating sampling, not just greedy decoding, because it gives the correctness test for distribution-preserving speculation.
- **Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias
- **Year:** 2023
- **URL:** https://proceedings.mlr.press/v202/leviathan23a.html
- **Relevant topics:** speculative decoding, assisted generation, rejection sampling, lossless acceleration, draft model

## Abstract
The paper presents speculative decoding for transformer inference with an exact sampling guarantee. A smaller approximation model drafts several tokens. The large target model evaluates those tokens in one forward pass. Draft tokens are accepted using a probability-ratio test, and if a token is rejected, the next token is sampled from the positive part of the difference between the target and draft distributions. This preserves the target model's output distribution while reducing the number of expensive target forward passes.

## Key Contributions
- Provides a distribution-preserving speculative decoding algorithm for sampling.
- Separates drafter quality, speculation length, and target verification cost as speedup factors.
- Demonstrates wall-clock acceleration on large transformer models without modifying target weights.
- Gives the correction rule that prevents biased rejection behavior.
- Popularizes the small-assistant-model pattern used in many LLM libraries.

## Key Figures/Tables to Study
- Algorithm 1: exact speculative decoding loop and acceptance test.
- Proof of distribution preservation: key for explaining "lossless" sampling.
- Speedup plots: show dependence on model size ratio and acceptance rate.
- Appendix details: useful for implementation of residual distribution sampling.

## Technical Details
Let `q` be the draft distribution and `p` the target distribution. For each proposed token, accept with probability `min(1, p(x) / q(x))`. If the token is rejected, sample from the normalized positive residual `max(0, p - q)`. If all drafted tokens are accepted, the target pass can also supply one extra token.

The target model still evaluates all drafted positions, but it does so in a single parallel forward pass. This is valuable because autoregressive serving is often memory-bandwidth and launch-latency dominated per token; verifying several tokens amortizes the cost of loading target weights.

## Connections
- [[speculative-decoding]] is the earlier SpecDec draft/verify formulation.
- [[hf-assisted-generation]] implements this family as assisted generation in Transformers.
- [[eagle]] and [[eagle-2]] improve drafter quality by drafting in feature space.
- [[medusa]] uses extra heads on the target model rather than a separate assistant model.
- [[prompt-lookup-decoding]] replaces the learned assistant with prompt n-gram lookup.
