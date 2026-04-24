# Excerpt: Switch Transformer — Top-1 Routing and Load Balancing

**Source:** [[switch-transformer|Fedus, Zoph, Shazeer, "Switch Transformers" (2022) (paper)]]

---

## The Switch Routing Simplification

Prior MoE work (Shazeer et al., 2017) used top-2 routing: each token is sent to two experts, computed twice, and the results are combined by gating weight. Switch Transformers make the counterintuitive decision to route each token to exactly one expert.

**Why top-1 works:**

1. **Communication cost halved.** With top-2, each token must travel to two experts (potentially on different devices), be processed twice, and have results returned and combined. Top-1 requires one dispatch, one computation, one return.

2. **Expert batch size doubled.** With fixed total tokens per batch and top-1 routing, each expert receives twice as many tokens as with top-2 (on average). Larger batch sizes mean better GPU utilization — the matrix multiplications inside each expert operate closer to peak throughput.

3. **Quality preserved at scale.** The paper demonstrates that with enough experts (8 to 2048), the marginal quality contribution of a second expert per token is negligible. The model compensates by having a deeper stack of single-expert layers.

## Load Balancing: The Auxiliary Loss

The balancing loss is defined over a batch of $T$ tokens with $N$ experts:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i = \frac{1}{T} \sum_{x \in \text{batch}} \mathbf{1}[\text{argmax}(g(x)) = i]$ is the fraction of tokens routed to expert $i$
- $P_i = \frac{1}{T} \sum_{x \in \text{batch}} g(x)_i$ is the mean routing probability for expert $i$

The product $f_i \cdot P_i$ is minimized when both are uniform at $1/N$. Because $f_i$ involves a hard argmax (non-differentiable), the gradient flows only through $P_i$ — the router learns to adjust its probability distribution to equalize load.

**Why $\alpha = 0.01$:** The paper tested values from 0.001 to 0.1. At $\alpha = 0.1$, the auxiliary loss dominates and forces perfectly uniform routing, destroying specialization. At $\alpha = 0.001$, routing collapse returns. The sweet spot at 0.01 provides enough pressure to prevent collapse while allowing meaningful expert specialization.

## Capacity Factor and Token Dropping

Each expert has a fixed buffer:

$$\text{Expert Capacity} = \left\lceil \frac{T}{N} \cdot C_f \right\rceil$$

where $C_f$ is the capacity factor (1.0 to 1.5). Tokens beyond this capacity are dropped — they skip the expert FFN and pass through the residual connection only.

At $C_f = 1.0$: the buffer exactly fits uniform distribution. Any imbalance causes drops.
At $C_f = 1.5$: 50% headroom, fewer drops but more wasted memory.

The paper reports capacity factors of 1.0--1.25 work well in practice. Token dropping has a dual nature: it prevents stragglers from slowing down training (a practical benefit), but dropped tokens receive degraded processing (a quality cost).

## Precision Management

A critical practical finding: **the router must operate in float32** even when the rest of the model uses bfloat16. The softmax over expert logits is numerically sensitive, and bfloat16's limited mantissa causes routing errors that destabilize training. The expert FFN computations can safely use bfloat16.

This selective precision approach — float32 router, bfloat16 experts — became standard practice and is used by all subsequent MoE architectures including DeepSeek-V2/V3 and Mixtral.

## Scale Results

| Configuration | Params | Speedup vs T5-XXL |
|--------------|--------|-------------------|
| Switch-Base (128 experts) | — | 7x pre-training speed |
| Switch-Large (128 experts) | — | Similar gains |
| Switch-XXL | 395B | 4x over T5-XXL |
| Switch-C | 1.571T | First trillion-param model trained |

The speedup is measured at equal FLOPs — the Switch model achieves the same loss in 1/7th the training steps. The gains come from the fact that each token accesses a larger effective parameter space (more total experts) without proportional compute increase.

## Distillation: Compressing Sparse into Dense

A practical contribution often overlooked: the paper demonstrates distilling large sparse MoE models into small dense models. A 1.6T Switch-C model can be distilled into a dense model that retains 30-40% of the quality gap between the original dense baseline and the sparse teacher.

The distillation procedure uses standard knowledge distillation (soft targets from the teacher's output distribution). The key finding is that distillation quality depends heavily on the task — for knowledge-heavy tasks (where the sparse model's advantage comes from more stored parameters), distillation preserves less of the gain. For reasoning-heavy tasks, more of the quality transfers because the reasoning patterns are less tied to raw parameter count.

This established a practical deployment pattern: train a large sparse MoE for maximum quality, then distill to a smaller dense model for serving if the MoE's routing overhead is prohibitive in your deployment setting.

## Architecture Placement: Every Layer vs Alternating

Switch Transformers explored two placement strategies:

1. **Every-other-layer:** Replace the FFN in alternating transformer layers with Switch MoE layers. The other layers retain a standard dense FFN. This provides a balance between routing overhead and capacity.

2. **Every-layer:** Replace all FFNs with Switch MoE layers. This maximizes capacity but doubles the routing overhead.

The paper found that every-other-layer placement provided the best quality-per-FLOP at smaller scales, while every-layer placement became favorable at very large scale (hundreds of billions of parameters) where the per-token routing overhead is amortized across more experts.

Later architectures (DeepSeek-V2/V3, Mixtral) apply MoE at every layer, reflecting the shift toward larger models where the routing overhead is negligible.

## Key Limitation

Switch Transformers provide no speedup during training in terms of per-step latency — the all-to-all communication for routing adds overhead. The 7x speedup is in **sample efficiency**: fewer steps to reach the same loss. At inference, the speedup is real — only one expert FFN is computed per token, giving true compute savings proportional to 1/N.

## Legacy and Influence

Switch Transformers established three principles that remain foundational:

1. **Top-1 routing is viable** — subsequent work (Llama 4) uses top-1 with shared experts
2. **Auxiliary loss + capacity factor** is the baseline balancing recipe (superseded by DeepSeek-V3's bias terms, but still widely used)
3. **Float32 router precision** is non-negotiable — every subsequent MoE system maintains this

The paper's greatest contribution may be psychological: it demonstrated that trillion-parameter models are trainable with simple engineering, removing the aura of impossibility from extreme-scale sparse models.
