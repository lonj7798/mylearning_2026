# Excerpt: Llama 4's MoE + iRoPE Architecture

<!-- source: [[llama-4|report]], Architecture Summary and Key Innovations -->

## The Dual Innovation

Llama 4 introduces two architectural innovations simultaneously: Mixture-of-Experts (MoE) for compute efficiency and interleaved RoPE (iRoPE) for extreme context length. These are not independent choices — they enable each other.

MoE reduces active parameters per token (17B active out of 109-400B total), which makes long-context inference economically feasible. iRoPE enables length generalization from 256K training context to 10M inference context, which is only viable because MoE keeps per-token cost low enough to serve 10M tokens.

## MoE Configuration: Scout vs Maverick

From the report, the two released models differ primarily in expert count:

**Scout (109B total / 17B active):**
- 16 expert FFN modules per layer
- Routing: 1 shared expert (always active) + 1 routed expert (selected per token)
- Active ratio: 17B / 109B = 15.6%
- Fits on a single H100 with int4 quantization

**Maverick (400B total / 17B active):**
- 128 expert FFN modules per layer
- Same routing: 1 shared + 1 routed
- Active ratio: 17B / 400B = 4.25%
- Fits on a single H100 DGX host

The **shared expert** is a critical design element. By always activating one expert regardless of routing decisions, Llama 4 guarantees a capability floor. Every token gets at least one full FFN computation. The routed expert adds specialization on top of this baseline. This differs from pure top-k routing (e.g., Mixtral's top-2 from 8) where a token's entire FFN processing depends on routing quality.

## iRoPE: Position Encoding for Extreme Context

Standard RoPE applies rotary position embeddings to queries and keys in every attention layer. Position is baked into every attention computation, which limits generalization beyond training context length.

iRoPE interleaves two types of attention layers:

1. **Position-aware layers** — standard RoPE applied to Q and K. These layers know where tokens are in the sequence.
2. **Position-free layers** — no positional encoding. Attention is purely content-based, agnostic to token position.

The position-free layers have no inherent context length limit. They compute the same attention pattern whether the sequence is 256K or 10M tokens long. The position-aware layers provide the necessary positional grounding, while position-free layers provide unbounded content matching.

**Inference-time temperature scaling:** When extrapolating beyond training length (256K to 10M), iRoPE applies temperature scaling to attention logits in position-aware layers. This prevents the attention distribution from becoming too peaked or too flat at positions the model never saw during training.

## The Extrapolation Math

- Training context: 256K tokens
- Inference context: 10M tokens
- Extrapolation ratio: 10M / 256K = 39x

This 39x extrapolation is qualitatively different from LLaMA 3's approach:
- LLaMA 3 used progressive training (8K -> 128K) = 16x extension, but this required multi-stage training at increasing context lengths
- Llama 4's extrapolation is achieved at inference time with no additional training, through architectural properties of iRoPE

## Connection to Course Chapters

- [[ch-14]] covers MoE routing strategies, load balancing, and the shared expert concept
- [[ch-06]] covers RoPE mathematics and the challenge of length generalization
- [[ch-16]] covers context extension techniques including iRoPE theory
- [[ch-07]] covers GQA, which Llama 4 retains unchanged from LLaMA 3
