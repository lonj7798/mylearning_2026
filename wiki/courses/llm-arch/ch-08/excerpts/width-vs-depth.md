<!-- scope: Width vs depth experimental evidence, parent: [[ch-08]] -->

# Width vs. Depth: Experimental Evidence

Given a fixed parameter budget, should you make the FFN wider or add more Transformer layers? This excerpt collects the experimental evidence across scales and tasks, and extracts the practical decision rules.

---

## The Theoretical Tradeoff

**Width** (larger $d_{ff}$): Each additional neuron in the FFN adds one key-value memory slot (under the FFN-as-memory hypothesis). Wider layers store more knowledge per compositional step.

**Depth** (more layers $L$): Each additional layer adds one attention + FFN block. More layers enable more steps of composition -- multi-hop reasoning, complex syntactic processing, iterative refinement.

The total parameters scale as:

$$P_{\text{total}} \approx L \times (P_{\text{attn}} + P_{\text{FFN}}) = L \times (4d^2 + 3d \cdot d_{ff})$$

(For SwiGLU with GQA. The attention count simplifies with GQA but $4d^2$ is a reasonable approximation.)

So adding one layer costs $\sim 4d^2 + 3d \cdot d_{ff}$ parameters, while doubling $d_{ff}$ in every layer costs $L \times 3d \times d_{ff}$ additional parameters. The tradeoff is not symmetric: widening is proportionally more expensive at fixed depth.

---

## Evidence: Very Shallow Models Fail at Reasoning

### The Depth Floor

Multiple independent studies converge on a minimum depth for competent language modeling:

- **GPT-2 family:** The jump from 12 layers (117M) to 24 layers (345M) brought disproportionate reasoning improvements -- more than the 3x parameter increase would predict from scaling laws alone.
- **Scaling law experiments (Hoffmann et al., 2022):** At fixed compute budget, the optimal depth-to-width ratio shifts toward deeper models as total compute increases.
- **Practical observation:** No competitive LLM uses fewer than 24 layers. Most frontier models use 32-126 layers.

Below ~12 layers, multi-hop reasoning degrades sharply. The model can memorize individual facts but cannot chain them. This is the "depth floor" -- there's a minimum number of compositional steps required for coherent language generation.

### Why Depth Helps Reasoning

Each layer is one round of:
1. **Attend** -- gather information from other positions
2. **Process** -- transform the gathered information through the FFN

Multi-hop reasoning (e.g., "What country is the birthplace of the inventor of the telephone?") requires:
- Step 1: Identify "inventor of the telephone" = Alexander Graham Bell
- Step 2: Identify "birthplace of Alexander Graham Bell" = Edinburgh
- Step 3: Identify "country containing Edinburgh" = Scotland/UK

Each step requires at least one attention operation (to route the right information) and one FFN operation (to retrieve the fact). A 3-hop question needs at least 3 effective layers, plus overhead for input processing and output formatting. Deeper models have more capacity for longer reasoning chains.

---

## Evidence: Narrow Models Fail at Recall

### The Width Floor

When $d_{ff}$ is too small relative to the knowledge the model needs to store:

- **Small models on knowledge-intensive benchmarks:** Models with narrow FFNs (e.g., $d_{ff} < 2048$) consistently underperform on trivia, factual QA, and open-domain tasks, even with many layers.
- **ROME editing experiments:** Models with narrow FFNs have fewer "editable" facts -- there are fewer dedicated key-value slots for individual associations, so facts are more distributed and harder to localize.
- **MoE motivation:** The entire Mixture-of-Experts paradigm exists because dense FFN width is too expensive to scale. Rather than making the FFN $256 \times$ wider, you create 256 expert FFNs and route to a subset.

### The Scaling-Law Perspective

Kaplan et al. (2020) found that model quality scales as a power law in parameters, with no strong preference between depth and width *at moderate scales*. But at the frontier:

- **Llama 3's scaling experiments** showed diminishing returns from width at fixed depth. The 405B model (126 layers, $d_{ff} = 53{,}248$, ratio = 3.25x) has a *lower* width ratio than the 8B model (32 layers, $d_{ff} = 14{,}336$, ratio = 3.5x). This suggests Meta found more value in depth at very large scale.

- **DeepSeek-V3** ([[deepseek-v3|report]]) sidesteps the tradeoff with MoE: 256 experts give $256 \times d_{expert}$ total FFN capacity (extreme effective width) but only $8 \times d_{expert}$ active per token (modest compute). The base $d_{model} = 7168$ with 61 layers is moderately deep and moderately wide.

---

## Architecture Comparison: Width-First vs. Depth-First

### Width-First: gpt-oss (2025)

gpt-oss made a deliberate bet on width over depth. Its 20B model uses:
- $d_{model} = 2880$
- Layers = 24
- Wide FFN relative to depth

The design philosophy: fewer layers of composition, but each layer has high-capacity memory. This optimizes for tasks where factual recall matters more than multi-step reasoning.

### Depth-First: Llama 3 405B

Meta's largest dense model:
- $d_{model} = 16{,}384$
- Layers = 126
- $d_{ff} = 53{,}248$ (ratio = 3.25x)

126 layers is extremely deep by historical standards (GPT-3 had 96 layers at 175B). The Llama 3 team chose depth, suggesting that at frontier scale, additional compositional steps have higher returns than additional memory per step.

### Width-via-MoE: DeepSeek-V3

- $d_{model} = 7{,}168$
- Layers = 61
- 256 routed experts + 1 shared per layer
- Active per token: 37B of 671B total

MoE decouples width from compute. The *effective* width (total expert capacity) is enormous, but the *active* width (compute per token) is moderate. This gives the knowledge capacity of an impossibly wide model at practical compute cost.

---

## Empirical Rules of Thumb

Based on the collected evidence:

1. **Minimum depth for language modeling: ~24 layers.** Below this, multi-hop reasoning degrades regardless of width.

2. **Minimum width for knowledge tasks: $d_{ff} \ge 4 \times d_{model}$ (SwiGLU: $\ge 2.67d$).** Below this, factual recall suffers.

3. **At moderate scale (1-30B), width and depth are roughly interchangeable per parameter.** The Kaplan scaling laws don't show a strong preference.

4. **At frontier scale (100B+), depth has higher marginal returns.** Llama 3's 126-layer design reflects this. Each layer adds a compositional step; each width increment adds only memory.

5. **MoE makes width nearly free.** If you need more knowledge capacity without proportional compute cost, MoE is the standard solution ([[ch-14]]).

6. **The failure modes are asymmetric:** A too-shallow model fails at reasoning (unfixable without adding layers). A too-narrow model fails at recall (fixable with retrieval augmentation or MoE).

---

## The "Optimal Shape" Question

Is there an optimal $L : d_{model} : d_{ff}$ ratio for a given parameter budget?

The answer depends on the task distribution. For a general-purpose LLM targeting both reasoning and knowledge:

$$L \approx 2\sqrt{P / (4d^2 + 3d \cdot d_{ff})}, \qquad d_{ff} \approx 3.5d$$

This is a very rough heuristic. The actual optimal shape depends on:
- Training data distribution (reasoning-heavy vs. knowledge-heavy)
- Target deployment (latency-sensitive favors width; throughput-sensitive favors depth)
- Hardware constraints (pipeline parallelism favors more layers; tensor parallelism favors wider layers)

---

## References

- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- [[llama-3|Meta AI, "The Llama 3 Herd of Models" (2024) (report)]]
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]]
- [[glu-variants|Shazeer, "GLU Variants Improve Transformer" (2020) (paper)]]
