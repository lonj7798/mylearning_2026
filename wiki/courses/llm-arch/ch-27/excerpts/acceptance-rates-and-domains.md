# Excerpt: Acceptance Rates — When Speculative Decoding Helps

<!-- sources: [[speculative-decoding|paper]], [[deepseek-v3|report]], [[raschka-reasoning-llms|blog]], [[weng-why-we-think|blog]] -->

## The Acceptance Rate is Everything

The speedup from speculative decoding is determined almost entirely by the acceptance rate $\alpha$ — the fraction of draft tokens that match what the target model would have generated.

With acceptance rate $\alpha$ and speculation length $K$:

$$E[\text{tokens per iteration}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

At $\alpha = 0.5$ with $K = 5$: ~2.0 tokens/iter (barely faster than standard decoding after accounting for draft cost). At $\alpha = 0.85$: ~4.7 tokens/iter (roughly 3x speedup). The relationship is highly nonlinear — the difference between 0.7 and 0.9 acceptance rate can be the difference between 1.5x and 3.5x speedup.

## What Determines the Acceptance Rate

The acceptance rate at position $i$ is:

$$\alpha_i = \sum_{x} \min(p(x | x_{<i}), q(x | x_{<i}))$$

This is the total variation overlap between the target and draft distributions. When both models agree (assign high probability to the same tokens), $\alpha$ is high. When they disagree, $\alpha$ drops.

Agreement depends on two factors:

1. **Text entropy.** Low-entropy tokens (deterministic completions, boilerplate, syntactic structure) have high agreement because both models are confident about the same token. High-entropy tokens (creative choices, reasoning steps, rare facts) have low agreement because the models may be confident about different tokens.

2. **Draft-target alignment.** Same-family models (Llama 3 70B + Llama 3 8B) agree more than independently trained models, because they share training data and learned similar distributions.

## Domain-Dependent Speedups

| Domain | Typical Alpha | Speedup (K=5) | Why |
|--------|--------------|---------------|-----|
| Code completion | 0.80-0.90 | 2.5-3.0x | Syntax is highly constrained; variable names repeat; boilerplate patterns dominate |
| Translation | 0.75-0.85 | 2.0-2.5x | Grammar constrains word order; most translation choices are deterministic |
| Summarization | 0.70-0.80 | 1.8-2.2x | Content is anchored by source text, reducing uncertainty |
| Structured output (JSON) | 0.85-0.95 | 3.0-4.0x | Schema constrains most tokens; only values are uncertain |
| Creative writing | 0.50-0.65 | 1.2-1.6x | Many semantically valid continuations; the target model's specific choice is unpredictable |
| Mathematical reasoning | 0.50-0.70 | 1.3-1.8x | Reasoning steps require capabilities the draft model may lack |
| Chain-of-thought | 0.55-0.70 | 1.3-1.8x | Thinking tokens are exploratory; scaffolding tokens are predictable |

## The Reasoning Model Tension

Modern reasoning models (DeepSeek-R1, OpenAI o1, Qwen QwQ) generate extremely long chain-of-thought traces. These traces contain two types of tokens:

**Structural scaffolding** (predictable): "Let me think about this...", "Wait, I need to reconsider...", reasoning format markers. A draft model can predict these accurately because they follow common patterns. Acceptance rates for these tokens are typically 0.8+.

**Substantive reasoning** (unpredictable): The actual logical steps, calculations, and conclusions. These require the target model's full capabilities and are precisely the tokens where a smaller draft model diverges. Acceptance rates drop to 0.3-0.5 for novel reasoning steps.

The blended acceptance rate across a full reasoning trace (mix of scaffolding and substance) typically lands at 0.55-0.70. This still provides a meaningful speedup (1.3-1.8x), but less than the 2.5-3x seen in code completion. Given that reasoning traces can be thousands of tokens long, even a 1.5x speedup translates to significant latency savings.

## Batch Size Interaction

Speculative decoding's benefit scales inversely with batch size:

- **Batch = 1:** Maximum benefit. Single-token decoding is maximally bandwidth-bound; all extra compute from verification is free.
- **Batch = 8-16:** Moderate benefit. Standard decoding starts amortizing weight-loading across batch members, but is still bandwidth-bound.
- **Batch = 32+:** Diminishing benefit. Standard decoding approaches compute-bound territory. The draft model's overhead may exceed the speedup.

This makes speculative decoding primarily a **latency optimization for interactive settings**: chat, code completion, real-time applications where batch size is 1 and users are waiting. For high-throughput batch processing (offline evaluation, data generation), other optimizations (continuous batching, larger batch sizes) provide more benefit.

## Adaptive Speculation Length

The optimal speculation length K varies within a single generation:

- When generating boilerplate or predictable text, longer K (8-10) is optimal because most tokens will be accepted
- When the model enters a high-entropy region (novel reasoning, creative transition), shorter K (2-3) reduces wasted draft compute

Production systems like vLLM implement **adaptive K**: they track recent acceptance rates and dynamically adjust speculation length. When the last few iterations had high acceptance, K increases. When rejections spike, K decreases. This provides 10-20% additional speedup over fixed K.

## Temperature and Sampling Interact with Acceptance

The sampling temperature affects acceptance rates in a non-obvious way:

- **Temperature = 0 (greedy):** Both models pick their argmax. If they agree on the top token (common for predictable text), alpha = 1.0. If they disagree, alpha = 0.0. The rate becomes binary — either perfect acceptance or immediate rejection.

- **Low temperature (0.1-0.5):** Distributions are peaked. Agreement is high when models share the same top token, moderate otherwise. Acceptance rates are typically 0.7-0.9.

- **High temperature (0.8-1.2):** Distributions are spread out. Even when models disagree on the top token, there's significant distributional overlap. Acceptance rates stabilize at 0.5-0.7.

- **Very high temperature (>1.5):** Both distributions approach uniform. Overlap increases, but the draft model's proposals are essentially random — accepted tokens carry little information. Speedup is modest.

The practical sweet spot for speculative decoding is the temperature range where models are confident enough to agree on likely tokens (low entropy) but not so greedy that a single disagreement causes immediate rejection.

## The Bottom Line

Speculative decoding is not a universal accelerator. It is a targeted optimization for the specific regime where autoregressive decoding is most wasteful: low-batch, bandwidth-bound, predictable-text generation. Understanding when your workload falls in this regime — and what acceptance rates to expect — is the difference between a meaningful 2-3x speedup and wasted engineering effort.
