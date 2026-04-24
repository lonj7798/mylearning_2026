# Chapter 14: Mixture of Experts

<!-- scope: MoE fundamentals, routing strategies, load balancing, auxiliary-loss-free balancing, fine-grained vs coarse-grained experts, shared experts
     deps: [[ch-08]], [[ch-13]]
     see-also: [[ch-19]], [[ch-21]], [[ch-23]]
-->

## Overview

Every dense transformer uses all of its parameters for every token. A 70B model does 70B parameters worth of work whether the input is a comma or a complex reasoning step. Mixture of Experts (MoE) breaks this coupling: the model has many more total parameters than it activates per token, and a learned gating network decides which subset — the "experts" — processes each token. The result is a model with the knowledge capacity of a much larger network but the per-token compute cost of a much smaller one.

This is not a minor efficiency trick. DeepSeek-V3 ([[deepseek-v3|report]]) has 671B total parameters but activates only 37B per token — a 5.5% activation ratio. Mixtral 8x7B ([[mixtral|report]]) matches Llama 2 70B quality while using only 12.9B active parameters, running 6x faster at inference. The insight is architectural: **knowledge capacity and per-token compute are separable design axes**, and MoE is the mechanism that separates them.

But MoE introduces a problem that dense models do not have: routing. Which experts should process which tokens? How do you prevent all tokens from collapsing onto a few "popular" experts while the rest sit idle? How do you balance load across devices in a distributed setting? The history of MoE is largely a history of routing strategies and load-balancing mechanisms — from the auxiliary losses of Switch Transformers ([[switch-transformer|paper]]) to the auxiliary-loss-free bias terms of DeepSeek-V3.

This chapter covers MoE from first principles through the 2025 design frontier. We start with the gating mechanism and sparsity, move through routing strategies (top-k, top-1, expert choice), confront the routing collapse problem, and then examine how modern architectures solve it. We close with the expert granularity spectrum — from Mixtral's 8 coarse experts to DeepSeek's 256 fine-grained experts — and the shared-expert design that has become standard.

---

## 1. MoE Fundamentals: Gating, Experts, and Sparsity

In a standard transformer, each layer has one FFN block that processes every token identically (same weights, different activations). An MoE layer replaces this single FFN with $N$ expert FFN blocks and a gating network (router) that selects which experts process each token.

### The Gating Network

The router is a simple linear layer followed by softmax. Given a token's hidden state $x \in \mathbb{R}^{d}$:

$$g(x) = \text{Softmax}(x \cdot W_g) \qquad W_g \in \mathbb{R}^{d \times N}$$

This produces a probability distribution over $N$ experts. The top-$k$ experts are selected, and the MoE layer output is the weighted sum of their outputs:

$$y = \sum_{i \in \text{TopK}(g(x), k)} g(x)_i \cdot E_i(x)$$

where $E_i(x)$ is the output of expert $i$ applied to input $x$, and $g(x)_i$ is the gating weight (routing probability) for expert $i$. Experts not in the top-$k$ contribute zero — this is the sparsity.

### Why Sparsity Enables Scale

The key property: **total parameters scale with $N$ (number of experts), but per-token FLOPs scale with $k$ (number of active experts)**. If each expert is an FFN with $d_{\text{ff}}$ intermediate dimension:

- Total parameters in the MoE layer: $N \times 2 \times d \times d_{\text{ff}}$ (for a standard up/down projection pair)
- Active parameters per token: $k \times 2 \times d \times d_{\text{ff}}$
- Activation ratio: $k / N$

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">MoE Parameter Efficiency: Total vs Active Parameters</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Model</th>
<th style="text-align:right; padding:8px;">Total Params</th>
<th style="text-align:right; padding:8px;">Active Params</th>
<th style="text-align:right; padding:8px;">Activation Ratio</th>
<th style="text-align:right; padding:8px;">Experts (N / k)</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Mixtral 8x7B</td>
<td style="text-align:right; padding:8px;">46.7B</td>
<td style="text-align:right; padding:8px;">12.9B</td>
<td style="text-align:right; padding:8px;">27.6%</td>
<td style="text-align:right; padding:8px;">8 / 2</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">DBRX</td>
<td style="text-align:right; padding:8px;">132B</td>
<td style="text-align:right; padding:8px;">36B</td>
<td style="text-align:right; padding:8px;">27.3%</td>
<td style="text-align:right; padding:8px;">16 / 4</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">DeepSeek-V2</td>
<td style="text-align:right; padding:8px;">236B</td>
<td style="text-align:right; padding:8px;">21B</td>
<td style="text-align:right; padding:8px;">8.9%</td>
<td style="text-align:right; padding:8px;">160 / 6</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ff6b81; font-weight:bold;">DeepSeek-V3</td>
<td style="text-align:right; padding:8px;">671B</td>
<td style="text-align:right; padding:8px;">37B</td>
<td style="text-align:right; padding:8px;">5.5%</td>
<td style="text-align:right; padding:8px;">256 / 8</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#a29bfe; font-weight:bold;">Qwen3-235B-A22B</td>
<td style="text-align:right; padding:8px;">235B</td>
<td style="text-align:right; padding:8px;">22B</td>
<td style="text-align:right; padding:8px;">9.4%</td>
<td style="text-align:right; padding:8px;">128 / 8</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#55efc4; font-weight:bold;">Llama 4 Maverick</td>
<td style="text-align:right; padding:8px;">400B</td>
<td style="text-align:right; padding:8px;">17B</td>
<td style="text-align:right; padding:8px;">4.3%</td>
<td style="text-align:right; padding:8px;">128 / 1 (+1 shared)</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Active parameters include shared layers (attention, embeddings). The activation ratio for the MoE FFN layers alone is k/N, which is even lower.
</div>
</div>

This table reveals an important trend: as MoE architectures have matured, activation ratios have dropped from ~28% (Mixtral) to under 6% (DeepSeek-V3). Models are storing more knowledge in more experts while activating fewer of them per token.

### Where MoE Layers Live in the Transformer

MoE replaces only the FFN sublayer. The attention sublayer remains dense — every attention head processes every token. A transformer layer with MoE looks like:

```
Input -> LayerNorm -> Multi-Head Attention -> Residual Add
      -> LayerNorm -> MoE(Router + N Expert FFNs) -> Residual Add -> Output
```

Some architectures apply MoE to every layer (DeepSeek-V2/V3, DBRX), while others interleave MoE and dense FFN layers (Switch Transformer replaces FFN in every other layer). The router itself is lightweight — a single linear projection — so the overhead of routing is negligible compared to the expert computation.

---

## 2. Routing Strategies

The router decides which experts process which tokens. This decision is the core design challenge of MoE, and the three main strategies represent different points on a simplicity/quality tradeoff.

### Top-K Routing (Shazeer et al., 2017)

The original large-scale MoE work routed each token to the top-$k$ experts by gating score, with $k = 2$ as the default. Noisy top-k gating adds Gaussian noise before selection to encourage exploration:

$$H(x)_i = (x \cdot W_g)_i + \epsilon \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i), \qquad \epsilon \sim \mathcal{N}(0, 1)$$

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

The noise serves two purposes: it prevents the router from deterministically always choosing the same experts (breaking symmetry during training), and it provides a form of load balancing by occasionally routing tokens to less-popular experts.

Mixtral 8x7B uses top-2 routing with 8 experts. Each token at each layer activates exactly 2 of 8 expert FFNs, meaning it uses 2/8 = 25% of the MoE layer's parameters per token. The two expert outputs are weighted by their softmax gating scores and summed.

### Top-1 / Switch Routing ([[switch-transformer|paper]])

Fedus, Zoph, and Shazeer (2022) made a counterintuitive simplification: **route each token to exactly one expert**. This "switch" routing halves the computation and communication cost compared to top-2, with no quality degradation at scale.

The argument is pragmatic. With top-2, each token must be sent to two different experts (potentially on different devices), computed twice, and the results combined. With top-1, each token goes to one expert on one device. The communication savings are substantial in distributed training where experts live on different GPUs.

Switch Transformers scaled to 1.6 trillion parameters with top-1 routing, achieving a 7x pre-training speedup over the dense T5-XXL baseline at the same FLOP budget. The paper's core insight: **simplicity wins at scale**. The extra expressiveness of top-2 routing is less important than the communication efficiency of top-1 when you have hundreds of experts.

### Expert Choice Routing

Standard routing is token-choice: each token picks its experts. Expert choice routing inverts this — **each expert picks its tokens**. Each expert selects the top-$k'$ tokens with the highest affinity scores from the batch, where $k'$ is a fixed capacity per expert.

This guarantees perfect load balance by construction: every expert processes exactly $k'$ tokens. But it introduces a different problem — some tokens may be selected by many experts (over-processed) while others are selected by none (dropped). In practice, expert choice routing requires a mechanism to handle unselected tokens, typically passing them through the residual connection unchanged.

Expert choice routing has seen limited adoption in production models. The token-choice paradigm (top-k or top-1) dominates, with load balancing handled by auxiliary losses or bias terms rather than by restructuring the routing direction.

---

## 3. The Routing Collapse Problem and Load Balancing

Left to its own devices, MoE routing collapses. A few experts become "popular" early in training, receive more tokens, train faster on more diverse data, become even better, and attract even more tokens. This positive feedback loop leads to **routing collapse**: most experts are unused, and the model degenerates into a small dense model with dead parameters.

[See the interactive routing visualization: [figures/routing-dynamics.html](figures/routing-dynamics.html)]

### Auxiliary Load-Balancing Loss

The standard solution is an auxiliary loss that penalizes uneven expert utilization. Switch Transformers define it as:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i$ = fraction of tokens actually routed to expert $i$ (the empirical load)
- $P_i$ = mean routing probability assigned to expert $i$ (the router's confidence)
- $N$ = number of experts
- $\alpha$ = balancing coefficient (typically 0.01)

This loss is minimized when both $f_i$ and $P_i$ are uniform ($1/N$ each). The product $f_i \cdot P_i$ is differentiable with respect to the router weights (through $P_i$), so the router learns to spread tokens more evenly.

**Why $\alpha$ must be small.** The auxiliary loss competes with the main language modeling loss. Too large an $\alpha$ forces perfectly uniform routing at the cost of model quality — the router cannot learn meaningful specialization. Too small and routing collapse returns. Switch Transformers use $\alpha = 0.01$; DeepSeek-V2 uses multiple auxiliary losses with $\alpha$ values between 0.003 and 0.05 for expert-level, device-level, and communication-level balancing.

### Capacity Factor and Token Dropping

Each expert has a fixed buffer size during training:

$$\text{Expert Capacity} = \frac{\text{tokens\_per\_batch}}{\text{num\_experts}} \times \text{capacity\_factor}$$

The capacity factor (typically 1.0--1.5) provides headroom above perfectly uniform distribution. Tokens that exceed an expert's capacity are **dropped** — they skip the MoE layer entirely and pass through the residual connection unchanged. This is a hard load-balancing mechanism that prevents any single expert from becoming a bottleneck.

Token dropping has a real cost: dropped tokens receive no expert processing for that layer, potentially degrading quality. DeepSeek-V2 ([[deepseek-v2|report]]) mitigates this by exempting ~10% of training sequences from dropping, ensuring some sequences always see full expert computation.

### Router Z-Loss (ST-MoE)

A subtler instability: the router's logits can grow very large during training, causing numerical overflow in softmax. The router Z-loss penalizes large logits:

$$\mathcal{L}_{z} = \frac{1}{B} \sum_{x} \left(\log \sum_{i=1}^{N} e^{g_i(x)}\right)^2$$

This keeps logits in a stable range without affecting the relative ordering of experts. It is added as a third term: $\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha_1 \mathcal{L}_{\text{balance}} + \alpha_2 \mathcal{L}_{z}$.

The HuggingFace MoE survey ([[hf-mixture-of-experts|blog]]) notes that the Z-loss "significantly improved training stability with no quality degradation," making it essentially free to include.

---

## 4. Auxiliary-Loss-Free Balancing: DeepSeek-V3's Approach

DeepSeek-V3 ([[deepseek-v3|report]]) identified a fundamental tension in auxiliary-loss-based balancing: **any loss term that competes with the main training objective degrades model quality**. Even with small $\alpha$, the auxiliary loss pulls the router toward uniform distribution when the optimal distribution might be non-uniform. Some tokens genuinely need certain experts more than others.

### Bias Terms Instead of Loss Penalties

DeepSeek-V3's solution replaces the auxiliary loss with **adaptive bias terms** added directly to the router logits:

$$g'(x)_i = g(x)_i + b_i$$

where $b_i$ is a per-expert bias that is updated outside the gradient computation:
- If expert $i$ is overloaded: decrease $b_i$ (makes the expert less likely to be selected)
- If expert $i$ is underloaded: increase $b_i$ (makes the expert more likely to be selected)

The bias update is controlled by a speed parameter $\gamma$ that determines how aggressively the system rebalances. Critically, **the bias terms are not trained by gradient descent** — they are adjusted by a simple heuristic based on observed load. This means they do not introduce any gradient signal that competes with the language modeling loss.

### Why This Matters

The results are stark. DeepSeek-V3 trained 671B parameters on 14.8T tokens with **zero irrecoverable loss spikes and zero rollbacks**. For context, training instability is the primary operational challenge of large MoE models — Switch Transformers and earlier systems frequently required rollbacks when routing collapsed or logits exploded. The auxiliary-loss-free approach eliminates an entire category of training failure.

DeepSeek-V3 also eliminates token dropping entirely. Because the bias terms actively rebalance load in real time, no expert exceeds its capacity, and every token receives full expert processing. This is a meaningful quality improvement over systems that drop 5-15% of tokens during training.

A small complementary sequence-wise auxiliary loss with an extremely small coefficient is retained, but it plays a minor stabilization role rather than being the primary balancing mechanism.

---

## 5. Fine-Grained vs Coarse-Grained Experts

The number and size of experts is a critical design choice with direct implications for routing expressiveness, implementation complexity, and quality.

### Coarse-Grained: Mixtral (8 experts, top-2)

Mixtral 8x7B ([[mixtral|report]]) uses 8 expert FFN blocks per layer, each identical to a full Mistral 7B FFN (intermediate dimension 14,336 with SwiGLU). With top-2 routing, each token selects 2 of 8 experts, yielding $\binom{8}{2} = 28$ possible expert combinations per layer.

This design prioritizes simplicity. Each expert is large and capable on its own, the routing decision is coarse (only 8 options), and the implementation is straightforward. Mixtral was the model that brought MoE to mainstream attention — its quality matched Llama 2 70B at a fraction of the inference cost.

### Fine-Grained: DBRX (16 experts, top-4)

DBRX ([[dbrx|report]]) doubles the expert count and doubles the active experts: 16 experts with top-4 routing. Each expert is smaller than Mixtral's (since the total parameter budget is distributed across more experts), but the number of possible combinations explodes: $\binom{16}{4} = 1{,}820$ — that is **65x more combinations than Mixtral**.

More combinations means finer-grained specialization. Instead of choosing between 28 possible "expert teams," the router can assemble from 1,820 different teams, allowing more precise matching of expert capabilities to token requirements. Databricks' ablations showed this tradeoff "clearly favored more experts" in quality.

### Ultra-Fine-Grained: DeepSeek (160-256 experts, top-6/8)

DeepSeek pushes fine-grained MoE to the extreme. DeepSeek-V2 uses 160 routed experts with top-6 routing per layer, and DeepSeek-V3 scales to 256 routed experts with top-8 routing. Each individual expert is much smaller (intermediate dimension 1,536 in V2 vs. Mixtral's 14,336), but the combinatorial space is vast.

[See the expert-layout comparison: [figures/expert-layout-comparison.html](figures/expert-layout-comparison.html)]

The tradeoff at this extreme:
- **Routing diversity:** $\binom{256}{8} \approx 4.4 \times 10^{13}$ combinations — effectively infinite specialization
- **Per-expert capacity:** Each expert is small, so it can only capture a narrow feature or pattern
- **Communication overhead:** Tokens may need to travel to experts on many different devices; DeepSeek-V3 mitigates this with node-limited routing (max 4 nodes per token)
- **Load balancing complexity:** With 256 experts, maintaining uniform utilization is harder, motivating the auxiliary-loss-free approach

### The Granularity Spectrum (2025 Landscape)

| Model | Experts | Active | Combinations per Layer | Expert Size |
|-------|---------|--------|----------------------|-------------|
| Mixtral 8x7B | 8 | 2 | 28 | Large (14,336 dim) |
| DBRX | 16 | 4 | 1,820 | Medium |
| Llama 4 Scout | 16 | 1 (+1 shared) | 16 | Medium |
| Qwen3-235B | 128 | 8 | ~2.3 x 10^10 | Small |
| Llama 4 Maverick | 128 | 1 (+1 shared) | 128 | Small |
| DeepSeek-V2 | 160 | 6 | ~2.1 x 10^11 | Small (1,536 dim) |
| DeepSeek-V3 | 256 | 8 | ~4.4 x 10^13 | Small |

The industry has clearly moved toward more, smaller experts. The dominant configuration in 2025 is 128 routed experts with top-8 routing (Qwen3, Llama 4 Maverick) or 256 with top-8 (DeepSeek-V3). The combinatorial argument is compelling: more expert combinations give the router more room to specialize, and the auxiliary-loss-free balancing approach makes large expert counts practical.

---

## 6. Shared Experts

DeepSeek-V2 ([[deepseek-v2|report]]) introduced a design element that has become influential: **shared experts** that process every token, regardless of routing.

### The Problem Shared Experts Solve

In a pure routed-only MoE, every expert must learn both common-knowledge patterns (syntax, frequent collocations, basic reasoning) and specialized patterns. This is wasteful — common knowledge is duplicated across many experts. It also means that a token routed to an unusual expert combination might miss out on processing that handles basic linguistic regularities.

Shared experts solve this by dedicating one or more expert FFNs to always-on processing:

$$y = E_{\text{shared}}(x) + \sum_{i \in \text{TopK}(g(x), k)} g(x)_i \cdot E_i(x)$$

The shared expert captures common patterns that every token needs, freeing the routed experts to specialize on rarer, more token-specific features.

### Adoption Across Architectures

| Model | Shared Experts | Routed Experts | Routing |
|-------|---------------|----------------|---------|
| DeepSeek-V2 | 2 | 160 | Top-6 |
| DeepSeek-V3 | 1 | 256 | Top-8 |
| Llama 4 Scout | 1 | 16 | Top-1 |
| Llama 4 Maverick | 1 | 128 | Top-1 |
| Qwen3 MoE | 0 | 128 | Top-8 |

The Llama 4 family ([[llama-4|report]]) adopted the shared expert design directly: each token is processed by 1 shared expert plus 1 routed expert (Maverick). This is the most conservative MoE routing — only a single routing decision per token per layer — but the shared expert guarantees every token gets a baseline level of processing.

Qwen3 ([[qwen-3|report]]) is the notable exception: it dropped shared experts entirely (Qwen2.5-MoE had them), relying on 128 routed experts with top-8 to cover both common and specialized patterns. The Qwen team's bet is that with enough active experts (8 per token), the routing will naturally assign at least some experts to common-knowledge roles. Whether this holds at all task types remains an open question.

### Shared Experts and Device Placement

A practical advantage: the shared expert lives on every device (since every token needs it), so it requires no inter-device communication. The routing communication overhead is confined to the routed experts. For Llama 4 Maverick with 1 shared + 1 routed, only the single routed expert requires cross-device token movement, making the communication pattern simple despite having 128 total routed experts.

---

## 7. Expert Parallelism and Communication

MoE models introduce a unique distributed computing challenge. In a dense model, every device processes the same layers on different data (data parallelism) or different layers on the same data (pipeline parallelism). In MoE, different tokens within the same batch must be routed to different devices depending on which experts they need.

### All-to-All Communication

The standard MoE communication pattern is **all-to-all**: after the router makes its decisions, tokens are dispatched from their current device to whichever device holds their selected expert(s), processed, and then returned. This all-to-all dispatch is the primary communication bottleneck in MoE training and inference.

The cost scales with:
- **Number of active experts $k$**: top-2 routing sends each token to 2 devices (worst case); top-1 sends to 1
- **Number of devices**: more devices means more potential communication paths
- **Expert placement**: if frequently co-selected experts live on the same device, communication drops

Switch Transformers' top-1 routing was motivated partly by this: halving $k$ from 2 to 1 halves the worst-case communication.

### Node-Limited Routing

DeepSeek-V3 introduces **node-limited routing**: each token may only be routed to experts on at most 4 nodes (out of 8 total nodes in their cluster). This caps the communication fan-out while still allowing routing to 256 experts spread across those nodes. The constraint is enforced by masking out experts on disallowed nodes before the top-k selection.

### DualPipe: Hiding Communication Behind Computation

DeepSeek-V3's DualPipe algorithm overlaps expert computation with all-to-all communication, achieving near-zero communication overhead. While experts on the current device are computing, tokens for remote experts are being transmitted in the background. This requires careful scheduling of the pipeline stages but makes the communication cost nearly invisible in practice.

---

## 8. What Do Experts Learn?

A natural question: do experts specialize by topic, language, syntax, or something else?

The findings are surprising. The HuggingFace survey ([[hf-mixture-of-experts|blog]]) reports:

- **Encoder models (e.g., BERT-MoE):** Experts show clear token-level specialization — one expert handles punctuation, another proper nouns, another common verbs, etc.
- **Decoder models (e.g., Mixtral, Switch Transformers):** Specialization is much less clean. Experts handle mixed token types, and it is difficult to assign simple semantic labels to each expert.
- **Multilingual models:** No language specialization. Despite intuitions that experts might partition by language, the auxiliary load-balancing loss prevents this clustering. All experts process tokens from all languages.

The lack of clean specialization in decoder models is important. It means the router is learning something more subtle than "this expert handles math" or "this expert handles code." The specialization likely operates at the level of low-level features (particular activation patterns in the hidden state) rather than human-interpretable categories.

---

## 9. Scaling Behavior and Diminishing Returns

Adding more experts increases model capacity at constant FLOPs, but with **diminishing returns** ([[hf-mixture-of-experts|blog]]):

| Expert Count | Relative Quality Gain |
|-------------|----------------------|
| 2 | Baseline |
| 8 | +25--30% |
| 128 | +45--50% |
| 512 | ~+55% (plateau begins) |
| 2048 | Negligible gain over 512 |

This logarithmic scaling means there is a practical ceiling on useful expert count. Going from 8 to 128 experts yields substantial gains, but going from 128 to 2048 yields almost nothing. The 128-256 expert range that DeepSeek and Qwen have converged on appears to be near the practical optimum — enough experts for rich combinatorial routing, not so many that the marginal value of each expert is negligible.

---

## 10. MoE for Fine-Tuning: Overfitting and Expert Freezing

MoE models overfit faster during fine-tuning than dense models of equivalent quality. The HuggingFace survey ([[hf-mixture-of-experts|blog]]) identifies the core issue: sparsity means each expert sees only a fraction of the fine-tuning data, so experts memorize their subset quickly.

**Practical mitigations:**
- Higher dropout in expert layers (0.3--0.5 vs. 0.1 global)
- Smaller batch sizes
- Higher learning rates (paradoxically — faster learning with stronger regularization)

A counterintuitive finding: **freezing expert weights and updating only shared layers** (attention, embeddings, LayerNorm) retains ~95% of fine-tuning quality with 30-50% faster training and 40% less VRAM. This works because shared layers affect every token, while each expert only affects tokens routed to it. Updating the shared layers gives broad coverage; updating experts gives narrow, easily overfit coverage.

Most strikingly, MoE models benefit *more* from instruction tuning than dense models — a 1.8x multiplier in quality improvement. The hypothesis is that MoE's diverse expert combinations are particularly well-suited for the diverse task distribution of instruction-tuning data.

---

## Core Insights from the Literature

### Insight 1: Simplicity wins at scale — top-1 routing works
**Paper:** Fedus, Zoph, Shazeer, "Switch Transformers" ([[switch-transformer|paper]])

The MoE community assumed top-2 or higher $k$ was necessary for quality. Switch Transformers showed that top-1 routing — the simplest possible choice — works at scale, achieving 7x pre-training speedup over dense baselines. The key realization is that at 128+ experts, the marginal value of routing to a second expert is small compared to the communication savings of top-1. This insight has influenced all subsequent MoE work: Llama 4 Maverick uses top-1 routed + 1 shared, achieving competitive quality with minimal routing complexity. **Guideline:** Start with top-1 routing and only increase $k$ if ablations show quality gains that justify the communication cost.

### Insight 2: Auxiliary losses degrade quality — decouple balancing from training
**Paper:** DeepSeek AI, "DeepSeek-V3" ([[deepseek-v3|report]])

Every auxiliary loss term competes with the main training objective. DeepSeek-V3 demonstrated that replacing auxiliary losses with non-gradient bias terms achieves better load balance *and* better model quality simultaneously. The proof is in the training stability: zero loss spikes, zero rollbacks across 14.8T tokens of training. This is a paradigm shift in MoE training — the routing problem is reframed from an optimization objective to a control-systems problem. **Guideline:** When routing is unstable, add adaptive bias rather than gradient penalties. Reserve auxiliary losses for secondary stabilization only.

### Insight 3: More smaller experts beat fewer larger experts
**Paper:** Databricks, "DBRX" ([[dbrx|report]])

DBRX's fine-grained MoE (16 experts, top-4) provides $\binom{16}{4} = 1{,}820$ expert combinations per layer — 65x more than Mixtral's $\binom{8}{2} = 28$. Databricks' ablations "clearly favored more experts." The combinatorial argument is the key: even if each individual expert is weaker (smaller), the routing network has vastly more options for assembling the right team for each token. This insight has driven the industry from 8-expert coarse MoE toward 128-256-expert fine-grained MoE. **Guideline:** Prefer more, smaller experts when the infrastructure supports it. The combinatorial gains in routing diversity outweigh the per-expert capacity reduction.

### Insight 4: Shared experts eliminate redundant learning
**Paper:** DeepSeek AI, "DeepSeek-V2" ([[deepseek-v2|report]])

Dedicating 1-2 experts to always-on processing frees routed experts from learning common patterns. The shared expert handles baseline linguistic competence while routed experts specialize. This is architecturally simple — just remove the gating for certain experts — but it changes the training dynamics: routed experts can specialize more aggressively because they do not need to be self-sufficient. Llama 4 adopted this directly. **Guideline:** Include at least one shared expert in fine-grained MoE architectures. The capacity cost is small (one expert's worth of always-on compute) but the specialization benefit is substantial.

### Insight 5: MoE decouples knowledge capacity from per-token compute
**Paper:** Meta AI, "Llama 4" ([[llama-4|report]])

Llama 4 Maverick has 400B total parameters but only 17B active — a 4.3% activation ratio. It competes with DeepSeek-V3 (37B active) on reasoning and coding despite activating less than half the parameters per token, because it has more total knowledge stored in its 128 experts. The design implication is that MoE transforms the scaling question from "how big is my model" to "how much knowledge does my model store, and how much compute can I afford per token at serving time." **Guideline:** Size the expert count (total parameters) for knowledge capacity and the active expert count for serving cost. These are independent design axes.

---

## Key Takeaways

1. **MoE replaces the FFN with N expert FFNs and a router.** Only $k$ experts activate per token, making total parameters and per-token compute independent design choices. The gating function is a simple softmax linear layer.

2. **Routing collapse is the central failure mode.** Without intervention, popular experts get more traffic, train faster, and attract even more traffic. Every MoE system must address this with auxiliary losses, bias terms, or capacity constraints.

3. **Auxiliary-loss-free balancing is the current frontier.** DeepSeek-V3's bias-term approach decouples load balancing from the training objective, yielding better quality and zero training instability. This eliminates token dropping entirely.

4. **Fine-grained MoE has won.** The industry has moved from 8 coarse experts (Mixtral) to 128-256 fine-grained experts (DeepSeek-V3, Qwen3, Llama 4). The combinatorial explosion in routing options dominates the per-expert capacity reduction.

5. **Shared experts handle common knowledge, freeing routed experts to specialize.** DeepSeek and Llama 4 use 1-2 always-on shared experts alongside the routed pool. Qwen3 is the notable exception, betting that top-8 routing naturally covers common patterns.

6. **Top-1 routing is sufficient when paired with shared experts.** Llama 4 Maverick routes to just 1 routed expert + 1 shared expert and achieves competitive quality. The communication savings of top-1 routing compound at scale.

7. **Expert specialization in decoders is not human-interpretable.** Experts do not cleanly map to topics or languages. The routing operates on low-level feature patterns, not semantic categories.

8. **Scaling expert count has diminishing returns past ~128-256.** The quality gain from 8 to 128 experts is substantial; from 128 to 2048, it plateaus. The current consensus of 128-256 experts appears near-optimal.

---

## References

- [[switch-transformer|Fedus, Zoph, Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2022) (paper)]] — top-1 routing, auxiliary loss, capacity factors
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]] — auxiliary-loss-free balancing, 256 experts, shared experts, DualPipe
- [[deepseek-v2|DeepSeek AI, "DeepSeek-V2" (2024) (report)]] — DeepSeekMoE, fine-grained experts, shared experts, device-limited routing
- [[mixtral|Mistral AI, "Mixtral of Experts" (2024) (report)]] — coarse-grained MoE (8 experts, top-2), GQA + MoE integration
- [[dbrx|Databricks, "DBRX" (2024) (report)]] — fine-grained MoE (16 experts, top-4), combinatorial routing argument
- [[llama-4|Meta AI, "Llama 4" (2025) (report)]] — shared + routed experts, top-1 routing at 128-expert scale
- [[qwen-3|Qwen Team, "Qwen3 Technical Report" (2025) (report)]] — 128 experts without shared experts, global-batch load balancing
- [[hf-mixture-of-experts|Sanseviero et al., "Mixture of Experts Explained" (Hugging Face blog)]] — MoE survey, expert specialization, fine-tuning, scaling behavior
