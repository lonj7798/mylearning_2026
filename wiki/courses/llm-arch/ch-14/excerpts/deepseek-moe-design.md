# Excerpt: DeepSeek MoE Design — From V2 to V3

**Sources:** [[deepseek-v2|DeepSeek AI, "DeepSeek-V2" (2024) (report)]], [[deepseek-v3|DeepSeek AI, "DeepSeek-V3" (2024) (report)]]

---

## DeepSeekMoE: Fine-Grained Experts with Shared Expert Isolation

DeepSeek-V2 introduced a fine-grained MoE design that departs sharply from the Mixtral/DBRX pattern. Instead of 8-16 large experts, DeepSeek uses 160 small experts (V2) or 256 (V3), each with an intermediate dimension of 1,536 — roughly 1/9th the size of a Mixtral expert (14,336 dim).

### The Shared Expert Architecture

The core design innovation is **shared experts**: 1-2 expert FFNs that process every token unconditionally, outside the routing mechanism. The MoE output becomes:

$$y = \sum_{s=1}^{N_s} E_{\text{shared}}^{(s)}(x) + \sum_{i \in \text{TopK}(g(x), k)} g(x)_i \cdot E_i^{\text{routed}}(x)$$

DeepSeek-V2 uses $N_s = 2$ shared experts; V3 uses $N_s = 1$. The shared experts are always active and do not participate in routing. This serves two functions:

1. **Common knowledge isolation.** Shared experts learn patterns that apply to all tokens (basic syntax, frequent patterns, general reasoning steps). This frees routed experts from redundantly learning these patterns.
2. **Baseline quality guarantee.** Even if routing makes a poor decision, the token still receives processing from the shared expert. This provides a quality floor that pure routed-only MoE lacks.

### Device-Limited Routing

With 160+ experts distributed across multiple devices, unrestricted routing could send a token to experts on every device, causing massive all-to-all communication. DeepSeek-V2 constrains each token to experts on at most $M = 3$ devices (out of 8 in their setup).

The implementation: the router first selects the top-$M$ devices by aggregate affinity score, then selects the top-$k$ experts within those devices. This two-stage routing bounds communication while preserving intra-device routing diversity.

### Token Dropping in V2

DeepSeek-V2 uses token dropping as a load-balancing mechanism: when a device's expert pool exceeds its compute budget, the tokens with the lowest affinity scores are dropped (processed only through the residual connection). To mitigate quality loss, ~10% of training sequences are exempted from dropping, ensuring some fraction of data always receives full expert processing.

## DeepSeek-V3: Auxiliary-Loss-Free Balancing

V3's primary MoE innovation eliminates auxiliary loss as the main balancing mechanism.

### The Problem with Auxiliary Losses

Traditional load-balancing losses ($\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{balance}}$) have a fundamental conflict: the balancing loss gradient pulls the router toward uniform distribution, while the task loss gradient pulls it toward optimal (potentially non-uniform) routing. These gradients compete during backpropagation, and any nonzero $\alpha$ degrades the task loss landscape.

DeepSeek-V2 used three separate auxiliary losses:
- Expert-level balancing ($\alpha = 0.003$)
- Device-level balancing ($\alpha = 0.05$)
- Communication-level balancing ($\alpha = 0.02$)

Even with small coefficients, these losses measurably degraded model quality compared to the theoretical optimum.

### Bias-Term Mechanism

V3 replaces the primary auxiliary loss with per-expert bias terms $b_i$ added to the router logits:

$$g'(x)_i = (x \cdot W_g)_i + b_i$$

The bias terms are updated by a simple control rule: if expert $i$ is overloaded (receives more tokens than $1/N$), decrease $b_i$; if underloaded, increase $b_i$. The update magnitude is controlled by a speed parameter $\gamma$.

**Critical design point:** The bias terms are *not* trained by gradient descent. They are adjusted by a non-gradient heuristic based on observed load statistics. This means they introduce zero interference with the task loss gradient — the router's learned weights $W_g$ are optimized purely for task performance, while the bias terms handle load balancing as a separate control loop.

### No Token Dropping

Because the bias terms actively rebalance load in real time, V3 eliminates token dropping entirely. Every token receives full expert processing at every layer. This is a meaningful quality improvement: in V2, dropped tokens at overloaded experts received no MoE processing for that layer.

### Training Stability Results

The proof of concept: V3 trained 671B parameters on 14.8T tokens with **zero irrecoverable loss spikes and zero rollbacks**. This is remarkable for a model of this scale — MoE training instability (loss spikes requiring rollbacks, routing collapse requiring restarts) was previously a routine operational challenge.

A small complementary sequence-wise auxiliary loss is retained, but with an "extremely small coefficient" — it serves as a gentle regularizer rather than the primary balancing mechanism.

## V3 Architecture Summary

| Component | DeepSeek-V2 | DeepSeek-V3 |
|-----------|-------------|-------------|
| Routed experts | 160 | 256 |
| Shared experts | 2 | 1 |
| Active routed | 6 | 8 |
| Expert dim | 1,536 | ~1,536 |
| Balancing | 3 auxiliary losses | Bias terms + tiny aux loss |
| Token dropping | Yes | No |
| Total params | 236B | 671B |
| Active params | 21B | 37B |
| Training tokens | 8.1T | 14.8T |
| Loss spikes | Not reported | Zero |

## Node-Limited Routing in V3

V3 extends device-limited routing to node-limited routing: each token may access experts on at most 4 nodes (where each node contains 8 GPUs with multiple experts). The DualPipe algorithm then overlaps computation and all-to-all communication within this constraint, achieving near-zero communication overhead.

The combination of more experts (256 vs 160), fewer shared experts (1 vs 2), no token dropping, and auxiliary-loss-free balancing represents a maturation of the DeepSeekMoE design — each change simplifies the system while improving quality.
