# Excerpt: Why LLaMA 3 Chose Dense Over MoE

<!-- source: [[llama-3|report]], Section: Architecture Summary and Design Decisions -->

## The Decision

LLaMA 3 405B is the largest dense transformer ever openly released. Meta explicitly chose not to use Mixture-of-Experts, stating that dense models are "simpler to train, scale, and serve." This was already a debatable claim when published in 2024, given that Mixtral (2023) and DeepSeek-V2 (2024) had demonstrated MoE at scale.

## From the Report

The Llama 3 technical report frames the architecture choice:

> We use a standard dense Transformer model architecture... We opted for a model design that is most compatible with existing training and inference infrastructure.

Key decisions that follow from choosing dense:

- **Scaling law experiments** used smaller dense models to predict 405B performance. These predictions are more reliable for dense architectures because the parameter-to-performance relationship is simpler (no routing dynamics, no expert collapse risk).

- **All 405B parameters active** for every token. No routing overhead, no load balancing, no expert parallelism communication. The serving stack is pure tensor parallelism.

- **128K vocabulary + GQA-8 + RoPE(500K)** — every component was already validated at smaller scale (Llama 2, Mistral, etc.). Zero novel architectural risk.

## The Tradeoff Ledger

**What dense at 405B buys:**

| Benefit | Mechanism |
|---------|-----------|
| Training predictability | Well-characterized scaling laws |
| Serving simplicity | No routing, no expert parallelism |
| Ecosystem adoption | Existing tooling works unchanged |
| Reproducibility | No novel components to reverse-engineer |

**What it costs:**

| Cost | Magnitude |
|------|-----------|
| Inference compute per token | 405B FLOPs (vs 17B for Llama 4 Maverick at similar total params) |
| Memory for weights | ~810 GB at FP16 (requires multi-node serving) |
| Context ceiling | 128K (progressive RoPE scaling is expensive to extend further) |
| Knowledge efficiency | Every parameter must be useful for every token (no specialization) |

## Why This Matters for Architecture Understanding

The dense choice reveals Meta's 2024 priority ordering:

1. **Training success probability** > inference efficiency
2. **Ecosystem compatibility** > architectural innovation
3. **Predictability** > optimality

By 2025, with LLaMA 3 training complete and the dense ceiling documented, the priority ordering reversed — and Llama 4 was the result.

## Connection to Course Chapters

- [[ch-14]] covers MoE fundamentals — the alternative LLaMA 3 rejected
- [[ch-07]] covers GQA — the attention variant LLaMA 3 standardized across all sizes
- [[ch-09]] covers RMSNorm — unchanged since LLaMA 1
- [[ch-08]] covers SwiGLU — the activation function that became universal
