# Excerpt: The Expert Granularity Spectrum

**Sources:** [[mixtral|report]], [[dbrx|report]], [[deepseek-v2|report]], [[deepseek-v3|report]], [[llama-4|report]], [[qwen-3|report]]

---

## The Design Axis: Few Large vs Many Small Experts

Expert granularity is a first-order architectural decision. It determines the combinatorial richness of routing, the per-expert capacity, the communication pattern, and the practical complexity of load balancing. The MoE landscape spans three orders of magnitude:

- **Coarse-grained:** 8 experts, each a full-sized FFN (Mixtral)
- **Medium-grained:** 16 experts, each ~half-sized (DBRX, Llama 4 Scout)
- **Fine-grained:** 128-256 experts, each a narrow FFN (DeepSeek, Qwen3, Llama 4 Maverick)

## Coarse: Mixtral 8x7B

Each of Mixtral's 8 experts is a complete Mistral 7B FFN: SwiGLU with intermediate dimension 14,336. This means each expert is a powerful, self-sufficient network on its own.

With top-2 routing, $\binom{8}{2} = 28$ combinations per layer.

**Strengths:**
- Implementation simplicity — 8 experts fit comfortably in standard parallelism schemes
- Each expert has high individual capacity — can capture complex, broad patterns
- Straightforward load balancing with only 8 targets

**Weaknesses:**
- Only 28 routing combinations limits specialization fineness
- Each expert must be somewhat general (it handles 25% of all tokens)
- Total parameter count is limited by practical expert size

Mixtral matched Llama 2 70B quality at 46.7B total params / 12.9B active — a 6x inference speedup. This demonstrated MoE's viability and set the benchmark that subsequent architectures aimed to surpass.

## Medium: DBRX (16 Experts, Top-4)

DBRX doubles the expert count and doubles the active count: 16 experts with top-4 routing.

$$\binom{16}{4} = 1{,}820 \text{ combinations — 65x more than Mixtral}$$

Databricks ran extensive ablations comparing expert configurations and concluded this tradeoff "clearly favored more experts." The 65x increase in combinations allows the router to assemble much more specialized expert teams per token.

**The per-expert capacity tradeoff:** With the same total parameter budget distributed across 16 experts instead of 8, each expert is roughly half the size. But each token activates 4 experts (vs 2), so the total active capacity per token is comparable. The net gain is routing diversity at no active-parameter cost.

**Performance:** DBRX (132B total / 36B active) surpassed Mixtral and GPT-3.5, demonstrating that fine-grained routing buys real quality.

## Fine-Grained: DeepSeek-V2/V3

DeepSeek pushes expert count to 160 (V2) and 256 (V3), with each expert having an intermediate dimension of only 1,536 — roughly 1/9th of a Mixtral expert.

**V2 (160 experts, top-6):** $\binom{160}{6} \approx 2.1 \times 10^{11}$ combinations
**V3 (256 experts, top-8):** $\binom{256}{8} \approx 4.4 \times 10^{13}$ combinations

At this scale, the combinatorial space is effectively infinite — the router never needs to reuse the same expert team. This enables extreme specialization: each expert can capture a very narrow feature or pattern, and the router assembles bespoke teams for each token.

**Why this works despite tiny experts:** The key insight is that individual expert capacity matters less when you activate 6-8 of them. The aggregate capacity of 8 small experts can equal or exceed 2 large experts, while the routing diversity is astronomically higher. The model learns to compose narrow specialists rather than relying on broad generalists.

**Communication challenge:** 256 experts across many devices means tokens may need to travel far. DeepSeek handles this with device-limited (V2) and node-limited (V3) routing, plus DualPipe communication overlap.

## Ultra-Fine with Few Active: Llama 4 Maverick

Llama 4 Maverick takes a unique position: 128 routed experts with only top-1 routing, plus 1 shared expert. Each token activates just 2 experts total (1 shared + 1 routed).

This is the most aggressive sparsity among production models:
- **400B total parameters, 17B active (4.3% activation ratio)**
- Only 128 routing combinations (vs DeepSeek-V3's trillions), but each token's compute is minimal
- Competes with DeepSeek-V3 on reasoning benchmarks at less than half the active parameters

The design philosophy differs from DeepSeek: instead of assembling a team of specialists (top-8), Maverick bets that the shared expert provides sufficient baseline competence and only one specialist is needed per token. The constraint is that the shared expert must be excellent — it handles 100% of tokens and carries the quality floor.

## No Shared Experts: Qwen3

Qwen3's MoE models use 128 experts with top-8 routing but **no shared experts** — a deliberate departure from the DeepSeek and Llama 4 design.

The Qwen team's reasoning: with 8 active experts per token, the router can naturally assign some experts to common-knowledge roles. The routing mechanism itself handles the shared-vs-specialized split dynamically rather than architecturally.

This is an interesting bet. With global-batch load balancing (rather than per-sequence), the system ensures all experts are utilized, but there is no architectural guarantee that any expert will always be available for common patterns. Whether this matters in practice depends on how robust the routing is — if the same 2-3 experts consistently handle common patterns across all inputs, they function as de facto shared experts.

## The Combinatorial Argument

The mathematical core of the fine-grained argument is the binomial coefficient:

| N experts | k active | Combinations |
|-----------|----------|-------------|
| 8 | 2 | 28 |
| 16 | 4 | 1,820 |
| 64 | 8 | 4.4 x 10^9 |
| 128 | 8 | 2.3 x 10^10 |
| 256 | 8 | 4.4 x 10^13 |

Going from 8 to 16 experts (2x more) with doubled active count gives 65x more combinations. Going from 16 to 256 experts (16x more) at the same active count gives ~24 billion times more combinations. This exponential growth in routing diversity is the fundamental reason the field has moved toward fine-grained MoE.

The diminishing returns curve tempers this: quality gains from adding experts follow a logarithmic pattern. The practical sweet spot appears to be 128-256 experts, where combinatorial richness is high but per-expert capacity has not degraded to the point of being useless.
