# Excerpt: Qwen 3 MoE — Extreme Sparsity and Fine-Grained Routing

<!-- source: [[qwen-3|report]] -->

## The 10:1 Sparsity Ratio in Context

The Qwen3-30B-A3B model activates 3B of 30B total parameters per token — a 10:1 ratio. To appreciate how aggressive this is, compare across the MoE landscape:

| Model | Year | Total | Active | Ratio | Experts (Total/Active) | Shared Experts |
|-------|------|-------|--------|-------|----------------------|---------------|
| Mixtral 8x7B | 2023 | 47B | 13B | 3.6:1 | 8 / 2 | No |
| Qwen 2.5-MoE | 2024 | 57B | 14B | 4.1:1 | 64 / 8 | Yes |
| DeepSeek-V2 | 2024 | 236B | 21B | 11.2:1 | 160 / 6 | Yes (2) |
| Qwen3-30B-A3B | 2025 | 30B | 3B | 10:1 | 128 / 8 | No |
| Qwen3-235B-A22B | 2025 | 235B | 22B | 10.7:1 | 128 / 8 | No |
| DeepSeek-V3 | 2024 | 671B | 37B | 18.1:1 | 256 / 8 | Yes (1) |

The trend is clear: MoE sparsity ratios have been increasing steadily. Early MoE models (Mixtral) used modest 2-4:1 ratios. Current state-of-the-art pushes 10:1 or higher. The architectural enabler is fine-grained expert segmentation — more, smaller experts provide the router with enough options to maintain quality even when activating a small fraction.

## 128 Experts: Why Fine-Grained?

Both Qwen 3 MoE models use exactly 128 routed experts. The expert *size* differs (the 235B model has ~8x larger experts than the 30B model), but the *count* is identical. This is a deliberate design choice with several implications:

**Combinatorial routing richness.** With 128 experts and top-8 routing, the number of possible expert combinations per token is:

C(128, 8) = 2.31 x 10^10

Compare to Mixtral's C(8, 2) = 28 combinations. The Qwen 3 router can express over 10 billion distinct "expert programs" per token. This combinatorial richness allows the model to represent highly specialized token-type distinctions — different expert subsets for mathematical notation, legal text, Python code, Chinese characters, etc.

**Load balancing challenge.** More experts means more ways for the router to collapse onto a small subset of favorites. If the router consistently prefers 16 out of 128 experts, the remaining 112 experts are effectively dead parameters. The global-batch load balancing loss addresses this, but the risk is real.

**Standardized routing architecture.** Using 128 experts for both the 30B and 235B models means the routing mechanism (gating network, top-k selection, load balancing) is identical. Only the expert FFN dimensions change. This simplifies engineering — one router implementation, one set of deployment tools, one load balancing configuration.

## No Shared Experts: A Simplicity Bet

DeepSeek-V2 introduced the concept of shared experts — a subset of experts that are always active for every token, providing common linguistic knowledge. DeepSeek-V3 continued this pattern. Qwen 2.5-MoE also used shared experts.

Qwen 3 drops them entirely. The arguments:

**For shared experts (DeepSeek's position):**
- Common patterns (syntax, function words, basic semantics) are needed by every token
- Without shared experts, multiple routed experts must independently learn these patterns
- Shared experts provide a stable "backbone" that routed experts specialize on top of
- Prevents catastrophic quality drops when the router makes unusual selections

**Against shared experts (Qwen 3's position):**
- Shared experts consume a fixed compute budget on every token, regardless of difficulty
- They reduce the "dynamic" compute budget available for specialized routing
- With 128 fine-grained experts, frequently-selected routed experts can serve the same function as shared experts
- Simpler architecture, simpler training, simpler deployment

The Qwen 3 team's implicit argument: if your router is good enough and your expert count is high enough, the most commonly-needed experts will be naturally selected most often, serving as *de facto* shared experts without the architectural overhead of a dedicated shared pathway.

## Global-Batch Load Balancing

Standard MoE load balancing adds an auxiliary loss that penalizes uneven expert utilization within each sequence. Qwen 3 replaces this with global-batch balancing: the auxiliary loss operates across the entire training batch.

**Why this matters:**

1. **Better statistics.** A single sequence might legitimately need mathematical experts disproportionately (e.g., a math problem). Per-sequence balancing would penalize this legitimate skew. Global-batch balancing expects that across thousands of sequences in a batch, the skew averages out.

2. **Smoother gradients.** The balancing loss gradient is computed over more tokens, reducing variance and providing a more stable training signal.

3. **Inference caveat.** At inference time, especially for single-sequence serving (chatbot scenario), there is no "batch" to balance over. Expert utilization may be highly uneven for individual requests. Whether this matters depends on hardware utilization — if some expert GPUs are idle while others are overloaded, throughput suffers.

## KV Head Reduction in MoE Models

A subtle but important detail: the MoE models use 4 KV heads (vs. 8 in all dense models). This is architectural co-optimization:

- MoE models already consume significant memory for expert parameters (128 expert FFNs per layer)
- Reducing KV heads from 8 to 4 halves the KV cache memory
- The 235B model with 64 query heads and 4 KV heads has a 16:1 GQA ratio — extremely aggressive
- This keeps attention memory subordinate to expert memory in the overall memory budget

The practical consequence: the MoE models' memory footprint is dominated by expert storage, not KV cache. For the 30B-A3B model, the 128 expert FFNs contain the vast majority of the 30B parameters, while the attention layers (with only 4 KV heads) are comparatively lightweight.

## Implications for Inference Deployment

The 10:1 sparsity ratio has direct deployment consequences. At inference time, only 8 of 128 experts are activated per token. If experts are distributed across GPUs (expert parallelism), only 8/128 = 6.25% of expert-holding GPUs are active per token. This creates a tension:

- **Memory:** All 30B (or 235B) parameters must be stored across the GPU cluster, even though only 3B (or 22B) are used per token.
- **Compute:** The active FLOPs per token are equivalent to a dense 3B (or 22B) model — dramatically cheaper than the total parameter count suggests.
- **Communication:** The router's top-8 selection must be communicated to the relevant expert-holding GPUs, and results aggregated. With 128 experts across many GPUs, the all-to-all communication pattern is more complex than for models with fewer, larger experts.

The net result: a Qwen3-30B-A3B deployment requires storage for 30B parameters but computes like a 3B model. For latency-sensitive applications, this is highly favorable — the per-token cost is 3B-class, but the quality benefits from 30B parameters of stored knowledge.
