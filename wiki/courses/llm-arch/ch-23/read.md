# Chapter 23: Case Study — Qwen 3, 3.5, and 3.6

<!-- scope: Qwen 3/3.5/3.6 architecture evolution — dual-mode thinking, dense/MoE lineup, aggressive MoE sparsity, training pipeline, DeltaNet hybrid backbone, multi-token prediction, thinking preservation
     deps: [[ch-12]], [[ch-14]]
     see-also: [[ch-19]], [[ch-18]], [[ch-07]]
-->

## Overview

Qwen 3 (Alibaba, May 2025) is a family of eight models — six dense (0.6B to 32B) and two Mixture-of-Experts (30B-A3B, 235B-A22B) — that share a single architectural innovation: **dual-mode inference**, where a single checkpoint operates in either *thinking mode* (extended chain-of-thought reasoning) or *non-thinking mode* (direct response), controlled by a runtime flag with no weight changes. Its successors — Qwen 3.5 (February 2026) and Qwen 3.6 (April 2026) — push the series in two orthogonal directions: 3.5 overhauls the architecture with a hybrid DeltaNet backbone and 512-expert MoE, while 3.6 demonstrates that training innovations alone (multi-token prediction, thinking preservation) can let a 27B dense model outperform a 397B MoE predecessor.

This chapter treats the Qwen 3 family as a case study in five architectural ideas:

1. **Inference-time compute allocation as an architecture decision.** The model's training pipeline — not its weights — determines whether it "thinks." This is a fundamentally different framing from test-time scaling approaches like beam search or majority voting, which are *external* to the model. Qwen 3 internalizes the thinking/non-thinking distinction into the model itself.

2. **Extreme MoE sparsity.** The Qwen3-30B-A3B model has a 10:1 ratio of total-to-active parameters — 128 experts, 8 active per token, no shared experts. This is substantially more aggressive than DeepSeek-V2's ~4:1 ratio or Mixtral's ~2:1, and raises concrete questions about expert utilization, load balancing, and the architectural tradeoff between parameter count and active compute.

3. **Dense-to-MoE design space.** The dense lineup spans three orders of magnitude (0.6B to 32B), making systematic architectural choices visible: where do you add depth vs. width? When do you tie embeddings? How does the GQA configuration scale?

4. **Hybrid linear-attention as production backbone.** Qwen 3.5 replaces 75% of attention layers with Gated DeltaNet, a linear-attention variant with delta-rule memory correction, enabling 262K native context at ~19x throughput improvement — the first open-weight model family to deploy sub-quadratic attention at scale.

5. **Training innovations outweighing architectural scale.** Qwen 3.6 demonstrates that multi-token prediction and thinking preservation on an unchanged backbone can make a dense 27B model beat a 397B MoE on agentic coding tasks, proving that architecture and training are separable, high-leverage design axes.

This chapter assumes familiarity with Mixture-of-Experts routing and gating from [[ch-12]] and the general structure of LLM training pipelines from [[ch-14]]. The dual-mode training pipeline connects to test-time compute scaling ideas from [[weng-why-we-think|blog]] and the reasoning RL literature surveyed in [[raschka-reasoning-llms|blog]].

---

## 1. The Dense Model Lineup: Scaling Decisions Under the Microscope

The six dense models share a common architectural template — Transformer decoder blocks with SwiGLU activation, RoPE positional encoding, RMSNorm pre-normalization, and grouped-query attention (GQA) — but differ in how they allocate capacity across depth, width, and head structure.

### Architecture Table

| Model | Params | Layers | d_model | Q Heads | KV Heads | d_head | Context | Embedding Tie |
|-------|--------|--------|---------|---------|----------|--------|---------|---------------|
| Qwen3-0.6B | 0.6B | 28 | 1024 | 16 | 8 | 64 | 32K | Yes |
| Qwen3-1.7B | 1.7B | 28 | 2048 | 16 | 8 | 128 | 32K | Yes |
| Qwen3-4B | 4B | 36 | 2560 | 32 | 8 | 80 | 128K | Yes |
| Qwen3-8B | 8B | 36 | 4096 | 32 | 8 | 128 | 128K | No |
| Qwen3-14B | 14B | 40 | 5120 | 40 | 8 | 128 | 128K | No |
| Qwen3-32B | 32B | 64 | 5120 | 64 | 8 | 80 | 128K | No |

Several patterns are worth extracting:

**KV heads are fixed at 8 across the entire lineup.** This is a deliberate GQA design choice (see [[ch-07]] for the GQA mechanism). Every model uses 8 KV heads regardless of the number of query heads, giving GQA ratios from 2:1 (0.6B, with 16 query heads) to 8:1 (32B, with 64 query heads). The implication: at larger scales, each KV head serves more query heads, compressing the KV cache more aggressively. The 32B model's 8:1 ratio matches the industry standard established by Llama 2 70B.

**Embedding tying disappears above 4B.** The three smallest models (0.6B, 1.7B, 4B) tie input and output embeddings — the same weight matrix serves as both the token embedding layer and the final output projection. This saves parameters proportional to `vocab_size * d_model` (for Qwen 3's 151,669-token vocabulary at d_model=2560, that's ~388M parameters — nearly 10% of the 4B model). Above 4B, the parameter savings become relatively insignificant compared to total model size, and untied embeddings allow the output projection to specialize for next-token prediction without constraining the input representation.

**Depth scales non-uniformly.** The jump from 0.6B to 1.7B adds zero layers (both 28L), relying entirely on width (1024 to 2048). From 1.7B to 4B adds 8 layers. From 14B to 32B adds 24 layers with *identical* width (both 5120). This reveals the scaling strategy: at small scale, width is cheap and effective; at large scale, depth becomes the primary capacity lever. The 32B model is essentially a deeper version of the 14B with more query heads.

**Context length has a discrete threshold.** The two smallest models support 32K context; everything from 4B onward supports 128K. This likely reflects training cost: extending context length requires long-context training data and longer training sequences, costs that are easier to justify for larger models expected to handle complex, multi-document tasks.

**QK-Norm is applied across all models.** Every Qwen 3 model normalizes the query and key vectors before computing attention scores. This prevents attention logit growth at long sequence lengths — a problem that becomes acute at 128K context — without requiring careful learning rate tuning. QK-Norm has become standard practice for long-context models (also used in Gemma 2, Llama 3.1+).

### Vocabulary: Byte-Level BPE at 151,669 Tokens

The tokenizer uses byte-level BPE with a 151,669-token vocabulary, covering 119 languages and dialects (up from 29 in Qwen 2.5). The byte-level fallback ensures that any arbitrary byte sequence can be tokenized without unknown tokens — critical for code, structured data, and low-resource languages. The vocabulary expansion for multilingual coverage is a significant investment: every additional token adds a row to the embedding matrix and a column to the output projection (unless tied), increasing memory and compute proportionally.

### d_head Variation: Not Always 128

A detail often overlooked: the per-head dimension ($d_k = d_\text{model} / H_Q$) is *not* constant across the lineup. The 0.6B model uses $d_k = 64$ (1024 / 16), the 4B and 32B models use $d_k = 80$ (2560/32 and 5120/64 respectively), while the 1.7B, 8B, and 14B models use $d_k = 128$. This means RoPE frequency bands and attention score magnitudes differ across model sizes — a consequence of prioritizing round numbers for $d_\text{model}$ and $H_Q$ over a fixed head dimension. QK-Norm mitigates the downstream effects: by normalizing Q and K before the dot product, the attention logit scale is decoupled from the head dimension, making the architecture more robust to this variation.

### SwiGLU Activation

All Qwen 3 models use SwiGLU in the FFN (and in each expert FFN for MoE models). SwiGLU combines the Swish activation with a gating mechanism:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

where $W_1$ and $W_2$ are separate linear projections and $\odot$ is element-wise multiplication. The gating mechanism ($xW_2$) allows the network to learn which features to suppress, providing smoother gradients than ReLU-based alternatives. SwiGLU has become the default FFN activation in modern LLMs (Llama 2+, Mistral, Gemma, PaLM 2), replacing the original Transformer's ReLU FFN. The cost: SwiGLU requires 3 weight matrices per FFN instead of 2 (the third being the gate projection), increasing FFN parameter count by 50% for the same hidden dimension. In practice, the hidden dimension is reduced to compensate, keeping the total parameter count comparable.

---

## 2. MoE Architecture: Extreme Sparsity Without Shared Experts

The two MoE models take the dense architecture template and replace the feed-forward network (FFN) in each Transformer block with a mixture-of-experts layer.

### MoE Architecture Table

| Model | Total | Active | Layers | Q/KV Heads | Experts | Active | Shared | d_expert |
|-------|-------|--------|--------|------------|---------|--------|--------|----------|
| Qwen3-30B-A3B | 30B | 3B | 48 | 32 / 4 | 128 | 8 | 0 | ~small |
| Qwen3-235B-A22B | 235B | 22B | 94 | 64 / 4 | 128 | 8 | 0 | ~large |

### The 10:1 Sparsity Ratio

The Qwen3-30B-A3B model activates only 3B of its 30B total parameters per token — a 10:1 ratio. For comparison:

| Model | Total | Active | Ratio | Experts (Total/Active) |
|-------|-------|--------|-------|----------------------|
| Mixtral 8x7B | 47B | 13B | 3.6:1 | 8 / 2 |
| DeepSeek-V2 | 236B | 21B | 11.2:1 | 160 / 6 (+2 shared) |
| Qwen3-30B-A3B | 30B | 3B | 10:1 | 128 / 8 |
| Qwen3-235B-A22B | 235B | 22B | 10.7:1 | 128 / 8 |

The high ratio means the model stores 10x more "knowledge" in its parameters than it accesses for any single token. The bet is that different tokens need different subsets of expertise, and the router learns to select the right 8 experts per token. The risk: with 128 experts and only 8 active, 93.75% of experts are idle per token. If the router fails to distribute load evenly, some experts become over-trained while others are rarely used, wasting parameters.

### No Shared Experts — A Deliberate Departure

DeepSeek-V2 and V3 use **shared experts** — a subset of experts that are always active for every token, providing a "common knowledge" backbone that routed experts can specialize on top of. Qwen 2.5-MoE also used shared experts. Qwen 3 drops them entirely.

The architectural argument for shared experts is compelling: common linguistic patterns (articles, prepositions, basic syntax) are needed regardless of which specialized experts are active. Without shared experts, multiple routed experts must independently learn these patterns, leading to parameter redundancy.

The argument *against* shared experts is simplicity and routing efficiency. Shared experts consume a fixed compute budget on every token regardless of difficulty. Eliminating them forces the router to be more selective and allows the full compute budget to be dynamically allocated. The Qwen 3 team's decision suggests they found the routing mechanism (global-batch load balancing) sufficient to handle common patterns through frequently-selected experts, without the architectural overhead of a dedicated shared pathway.

### Fine-Grained Expert Segmentation

Both MoE models use 128 experts — the same count at 30B and 235B. The difference is expert *size*: each expert in the 235B model is roughly 8x larger than in the 30B model. This "fine-grained" segmentation (many small experts rather than few large ones) has two consequences:

1. **Finer routing granularity.** With 128 experts and top-8 routing, the router can compose 128-choose-8 = ~2.3 x 10^10 distinct expert combinations. This is a vastly richer space than Mixtral's 8-choose-2 = 28 combinations. The model can express more specialized token-type distinctions.

2. **Better load balancing potential.** More experts provide more options for distributing load. But more experts also make load balancing harder — the router must learn to avoid collapsing onto a small subset of favorites.

### Global-Batch Load Balancing

Qwen 3 replaces per-sequence load balancing (used in most prior MoE models) with **global-batch load balancing**. Instead of enforcing that experts are evenly used within each sequence, the loss penalty operates across the entire training batch.

The advantage: statistical balancing improves with batch size. In a large batch, natural variation in expert demand across sequences averages out, so the balancing loss provides a smoother, more reliable gradient signal. The disadvantage: individual sequences may have highly uneven expert utilization. A sequence about mathematics might route overwhelmingly to math-specialized experts, and global balancing will not prevent this. Whether this matters depends on the inference regime — for offline batch processing, global balancing is fine; for single-sequence interactive use, within-sequence balance might matter more.

### KV Head Reduction in MoE Models

Notice that the MoE models use only **4 KV heads** (vs. 8 in all dense models). With 32 query heads in the 30B-A3B model, this gives an 8:1 GQA ratio — the same effective ratio as the dense 32B model but achieved with fewer absolute KV heads. The 235B-A22B model with 64 query heads and 4 KV heads gives a 16:1 ratio, which is extremely aggressive.

The motivation is clear: MoE models are already memory-constrained by the expert parameters. Reducing KV heads minimizes the additional memory pressure from the attention KV cache, keeping the inference memory budget dominated by expert storage rather than attention state. This is a concrete example of architectural co-optimization — the attention configuration is adjusted to accommodate the MoE memory profile.

[See the interactive dense-vs-MoE comparison: [figures/dense-vs-moe.html](figures/dense-vs-moe.html)]

---

## 3. Dual-Mode Architecture: Thinking and Non-Thinking in One Model

The most architecturally significant feature of Qwen 3 is not a structural innovation (the Transformer blocks are standard) but a **behavioral** one: the same weights produce fundamentally different output patterns depending on a runtime mode flag.

### What the Modes Do

- **Thinking mode:** The model generates an extended chain-of-thought reasoning trace inside `<think>...</think>` tokens before producing the final answer. This trace can span hundreds or thousands of tokens of intermediate reasoning — decomposing problems, considering alternatives, checking work.

- **Non-thinking mode:** The model produces a direct answer with no reasoning trace. Response latency is lower, token cost is lower, and the output is stylistically identical to a standard chat model.

### How Mode Switching Works

The critical insight: **no weights change between modes**. The model is architecturally identical in both modes. The difference is entirely in the system prompt and the presence or absence of special tokens (`<think>`, `</think>`) that frame the reasoning trace. The model has been trained to respond to these tokens as mode-switching signals.

This is a form of **instruction-conditioned behavior** — the same phenomenon that lets a single model respond differently to "explain like I'm five" vs. "give me the formal proof," but taken to an extreme. The model has learned two fundamentally different generation strategies (deliberative reasoning vs. direct response) and selects between them based on a prefix signal.

The architectural implication is that the model's capacity must be partitioned (at least conceptually) between the two modes. Thinking mode requires the model to have learned how to generate coherent multi-step reasoning, maintain logical consistency across long reasoning chains, and arrive at answers that benefit from the extended computation. Non-thinking mode requires the model to have learned when deliberation is unnecessary and how to produce concise, accurate direct responses. A single set of weights must support both.

### The Thinking Budget Mechanism

Qwen 3 adds a **budget control** for thinking mode: users can set a maximum token count for the reasoning trace. If the model's thinking exceeds this budget, it "gracefully transitions to generating a response with incomplete reasoning." This is inference-time compute allocation made explicit — the user directly controls how much computation the model invests in a problem.

This connects to the scaling law findings in [[weng-why-we-think|blog]]: test-time compute is not a free lunch. Easy problems benefit from brief or no thinking; hard problems benefit from extended thinking; and there is a point of diminishing returns. The thinking budget gives users a knob to navigate this tradeoff based on their latency and cost constraints.

### Why This Matters Architecturally

Previous reasoning models (DeepSeek-R1, OpenAI o1) required **separate model deployments** for reasoning and non-reasoning use cases. You would serve R1 for math/code tasks and DeepSeek-V3 for general chat. This doubles serving infrastructure and forces routing decisions at the application layer.

Qwen 3's unified approach eliminates this. A single model, single set of weights, single deployment handles both modes. The "routing" happens at the prompt level, which is trivially cheap. For production deployments, this halves the GPU allocation compared to maintaining separate reasoning and chat models.

The tradeoff: a unified model may be slightly worse at both modes than two dedicated models would be. The thinking mode quality is constrained by the need to also support fast non-thinking responses, and vice versa. The Qwen 3 team's benchmark results suggest this penalty is small — but the training pipeline required to achieve it is complex.

### Contrast with DeepSeek-R1's Training Approach

DeepSeek-R1 ([[raschka-reasoning-llms|blog]]) used a six-stage pipeline to build a *dedicated* reasoning model:

1. Pure RL on base model (R1-Zero) to discover reasoning
2. Use R1-Zero outputs to generate cold-start SFT data
3. Instruction fine-tune on SFT data
4. RL with accuracy, format, and consistency rewards
5. Additional SFT on 800K examples (600K CoT + 200K knowledge)
6. Final RL with rule-based + human preference rewards

Qwen 3's pipeline is shorter (four post-training stages) but tackles a harder problem: it must produce a model that reasons *and* responds directly, whereas DeepSeek-R1 only needed to produce a model that reasons. The mode fusion stage (Qwen 3's stage 3) has no analogue in the DeepSeek pipeline — it is the additional cost of unification.

The DeepSeek approach also revealed a key finding that Qwen 3 exploits: reasoning emerges from pure RL at large scale (the "aha moment" in R1-Zero), but smaller models learn reasoning more efficiently through distillation. Qwen 3's distillation path for sub-10B models directly applies this lesson.

[See the interactive dual-mode architecture diagram: [figures/dual-mode-architecture.html](figures/dual-mode-architecture.html)]

---

## 4. The Training Pipeline: How Dual-Mode Behavior Is Created

The dual-mode capability does not emerge from architecture — it is manufactured by a carefully staged training pipeline. This is the core technical contribution of Qwen 3: **inference-time compute allocation is a training decision, not an architecture decision.**

### Pre-Training: Three Stages, 36 Trillion Tokens

| Stage | Tokens | Seq Length | Purpose |
|-------|--------|------------|---------|
| General | ~30T | 4,096 | Broad knowledge and language modeling |
| Reasoning | ~5T | 4,096 | High-quality STEM, code, and reasoning data |
| Long Context | ~100sB | 32,768 | Context extension to 128K |

The three-stage curriculum is not novel in structure (many LLMs use staged pre-training), but the scale is notable: 36 trillion tokens total, with 5 trillion specifically curated for reasoning quality. The reasoning stage acts as a domain-specific pre-training phase that lays the foundation for the post-training RL to build on.

The long-context stage trains at 32K sequence length but the models support 128K context via position interpolation at inference time. Training at shorter lengths saves compute (self-attention cost scales quadratically) while position encoding extrapolation (via RoPE) extends the effective context.

### Post-Training: Four Stages Building Dual-Mode

The post-training pipeline is where the dual-mode behavior is constructed. Each stage builds on the previous one, and the order is not interchangeable.

**Stage 1: Long-CoT Cold Start.** The model is fine-tuned on curated examples of long chain-of-thought reasoning. This teaches the model the *format* of thinking — how to structure multi-step reasoning, use intermediate conclusions, and arrive at well-supported answers. This is supervised fine-tuning (SFT), not RL.

**Stage 2: Reasoning RL with GRPO.** The model undergoes reinforcement learning using Group Relative Policy Optimization (GRPO) on **only 3,995 query-verifier pairs**. This is remarkably data-efficient: fewer than 4,000 training examples produce measurable reasoning improvement.

GRPO works by generating multiple responses to each query, scoring them with a verifier (compiler for code, deterministic checker for math), and updating the policy based on relative performance within the group. The "relative" part is key — the model learns from the *ranking* of its own outputs, not from absolute reward values. Concretely, for each query $q$:

1. Sample $K$ candidate responses $\{r_1, \ldots, r_K\}$ from the current policy
2. Score each response with the verifier: $s_i \in \{0, 1\}$ (correct/incorrect)
3. Compute advantage as deviation from group mean: $A_i = s_i - \bar{s}$
4. Update policy to increase log-probability of high-advantage responses

Unlike PPO, GRPO does not require a separate value network — the group mean serves as the baseline, reducing training infrastructure complexity. This is particularly important at Qwen 3's scale: maintaining a value network for a 235B-parameter model would require substantial additional GPU memory.

The extreme data efficiency (3,995 examples) suggests that reasoning RL is not teaching the model new knowledge — the pre-training stages already provided the knowledge. RL is teaching the model to *use* that knowledge more effectively by rewarding successful reasoning chains and penalizing failed ones. As [[raschka-reasoning-llms|blog]] notes, "high-quality verified queries are more important than quantity for reasoning RL."

Compare the data volumes across the pipeline to see what each stage contributes:

| Stage | Data Volume | What It Teaches |
|-------|-------------|-----------------|
| General pre-training | ~30T tokens | Language, world knowledge, basic patterns |
| Reasoning pre-training | ~5T tokens | STEM knowledge, code patterns, analytical structure |
| Long-CoT SFT | Thousands of examples | Format of multi-step reasoning |
| Reasoning RL (GRPO) | 3,995 query-verifier pairs | Strategy for deploying reasoning effectively |
| Mode Fusion | Both-mode examples | Behavioral switching on token signal |
| General RL | Broad reward signals | Recovery of non-reasoning capabilities |

The 9-orders-of-magnitude gap between pre-training data (30T tokens) and RL data (3,995 examples) is the clearest evidence that these stages serve fundamentally different purposes. Pre-training builds capability; RL refines policy.

**Stage 3: Thinking Mode Fusion.** This is the critical stage for dual-mode behavior. The model learns to operate in both thinking and non-thinking modes, with the mode selected by the presence or absence of the `<think>` token prefix. The training data includes examples of both modes, and the model learns to switch behavior based on the mode signal.

This stage must solve the **mode confusion problem**: the model must not generate reasoning traces when in non-thinking mode, and must not skip reasoning when in thinking mode. Achieving this cleanly requires careful data curation — the non-thinking examples must be high-quality direct responses (not truncated thinking), and the thinking examples must demonstrate genuine deliberation (not padding).

**Stage 4: General Domain RL.** A final RL pass improves performance across general tasks (not just reasoning). This stage ensures that the reasoning-focused stages 1-3 did not degrade the model's performance on tasks where thinking is unnecessary — summarization, translation, creative writing, factual Q&A.

### Distillation: Strong-to-Weak Knowledge Transfer

The smaller dense models (0.6B through 8B) are trained via **distillation from the flagship models**, requiring only 1/10 of the GPU hours of the full four-stage pipeline. This is not classical knowledge distillation (matching teacher logits) — it is SFT on teacher-generated outputs, similar to the DeepSeek-R1 distillation approach described in [[raschka-reasoning-llms|blog]].

The distillation result is striking: distilled models achieve "superior pass@1 and pass@64 results" compared to models trained through the full pipeline at equivalent sizes. This aligns with the scale-dependent finding from DeepSeek-R1 research — at smaller model sizes (< 10B), pure SFT from high-quality teacher outputs consistently outperforms pure RL. The model capacity at small scale is insufficient to discover effective reasoning strategies through RL exploration; it is more efficient to simply show the model what good reasoning looks like.

| Training Approach | GPU Cost | Quality | Best At |
|------------------|----------|---------|---------|
| Full 4-stage pipeline | 1x | Highest at large scale | 14B+ models |
| Distillation from flagship | 0.1x | Superior at small scale | 0.6B-8B models |

This is a concrete architectural insight: **the optimal training pipeline depends on model size.** There is no universal recipe. Large models can discover reasoning through RL; small models learn it faster through imitation.

---

## 5. Inference-Time Compute Allocation as Architecture Decision

Qwen 3 crystallizes a trend that has been building since DeepSeek-R1 and OpenAI o1: **how much computation a model spends per query is itself an architectural choice**, not merely an operational one.

### The Compute Spectrum

Consider the full range of inference-time compute allocation in Qwen 3:

1. **Non-thinking mode, smallest model (0.6B):** Minimal compute per query. ~1.2B FLOPs per token, direct response. Suitable for classification, simple Q&A, low-latency applications.

2. **Non-thinking mode, largest model (235B-A22B):** 22B active parameters, ~44B FLOPs per token, but still direct response. High-quality answers without deliberation.

3. **Thinking mode, smallest model (0.6B):** The model attempts chain-of-thought reasoning with only 0.6B parameters. The reasoning quality is limited by model capacity, but the model can still decompose problems and check work.

4. **Thinking mode, largest model (235B-A22B), maximum budget:** Extended reasoning with 22B active parameters, potentially thousands of tokens of deliberation. The highest-quality, highest-cost configuration.

The thinking budget mechanism adds continuous control within thinking mode. The user effectively selects a point on a cost-quality Pareto frontier. This is not a new idea in computation (adaptive computation time has been explored since Graves 2016), but Qwen 3 operationalizes it in a production-ready way.

### Connection to Scaling Laws

[[weng-why-we-think|blog]] reports a key asymmetry in test-time compute scaling: "easier questions benefit from purely sequential test-time compute, whereas harder questions often perform best with an optimal ratio of sequential to parallel compute." The thinking budget mechanism maps directly onto this finding — users can allocate more thinking budget to harder questions.

But Weng also notes the limits: "test-time and pretraining compute are NOT 1:1 exchangeable." You cannot compensate for a weak base model by simply thinking longer. The pre-training stages (30T general + 5T reasoning) establish the model's fundamental capability; thinking mode allows more effective *use* of that capability, not expansion beyond it.

This has practical implications for model selection. The Qwen3-30B-A3B in thinking mode is not equivalent to the Qwen3-235B-A22B in non-thinking mode, even if both spend similar total FLOPs per query. The larger model has fundamentally more knowledge encoded in its 235B parameters; the smaller model's extended thinking cannot compensate for knowledge it does not have.

### Benchmark Evidence

The Qwen3-235B-A22B flagship demonstrates competitive results across reasoning and general benchmarks:

| Benchmark | Category | Qwen3-235B-A22B |
|-----------|----------|-----------------|
| AIME'24 | Math competition | 85.7% |
| AIME'25 | Math competition | 81.5% |
| LiveCodeBench v5 | Code generation | 70.7% |
| Codeforces | Competitive programming | 2,056 rating |
| MMLU | General knowledge | 87.8% |
| MMLU-Pro | Hard knowledge | 68.2% |
| BBH | Reasoning | 88.9% |
| GSM8K | Math word problems | 94.4% |

The AIME results are particularly informative. AIME is a math competition where problems require multi-step reasoning — exactly the domain where thinking mode should provide maximum benefit. The 85.7% on AIME'24 and 81.5% on AIME'25 are state-of-the-art, achieved by a model with only 22B active parameters per token. Compare: DeepSeek-R1 (671B total, ~37B active) and OpenAI o1 achieve similar-range results but with substantially more active compute per token. The MoE sparsity is paying off — the 235B model stores expert knowledge in its 235B parameters but deploys only the relevant 22B per token.

The Qwen3-30B-A3B is equally notable for what it achieves at 3B active parameters. While the report does not provide head-to-head comparisons against dense 3B models on all benchmarks, the architectural claim is clear: a 30B-parameter MoE model running at 3B-active compute should substantially outperform a dense 3B model trained on the same data, because it has 10x more stored knowledge accessible via routing.

---

## 6. Design Decisions and Tradeoffs

### Tradeoff 1: Unified Model vs. Separate Reasoning/Chat Models

| Dimension | Unified (Qwen 3) | Separate (DeepSeek R1 + V3) |
|-----------|------------------|----------------------------|
| Deployment cost | 1 model, 1 GPU allocation | 2 models, 2x GPU allocation |
| Routing complexity | Prompt-level (trivial) | Application-level (requires classifier) |
| Per-mode quality | Slightly compromised | Fully optimized |
| Training complexity | Higher (4-stage post-training) | Lower per model |
| Operational simplicity | Higher | Lower |

The Qwen 3 team bet on operational simplicity. For most deployment scenarios, the slight per-mode quality penalty is worth the halved infrastructure cost and eliminated routing complexity.

### Tradeoff 2: No Shared Experts vs. Shared Expert Baseline

Dropping shared experts simplifies the architecture but forces routed experts to redundantly learn common patterns. The Qwen 3 team's empirical finding was evidently that global-batch load balancing made shared experts unnecessary — but this may not generalize to different data distributions or smaller batch sizes during inference.

### Tradeoff 3: 128 Fine-Grained Experts vs. Fewer Coarse Experts

More experts means finer specialization but harder load balancing and higher router overhead. The choice of 128 experts with top-8 routing gives the richest combinatorial space for expert selection, but the router must learn a more complex assignment function. The standardization of 128 experts across both MoE sizes (30B and 235B) simplifies the routing architecture but means the 30B model's experts are individually quite small.

### Tradeoff 4: Global-Batch vs. Per-Sequence Load Balancing

Global-batch balancing provides smoother gradient signals and better statistical properties at the cost of per-sequence imbalance. This is well-suited to batch training and batch inference, but may cause uneven expert utilization in online, single-sequence serving scenarios.

---

## 7. Qwen 3.5: Hybrid DeltaNet Backbone

Qwen 3.5 (Alibaba, February 2026) is not an incremental update — it is an architectural overhaul. The central change: **75% of attention layers are replaced by Gated DeltaNet (GDN)**, a linear-attention variant that eliminates the quadratic KV-cache cost of standard attention for three out of every four layers. This is the first major open-weight model family to adopt DeltaNet at scale.

### The 3:1 Hybrid Layout

The backbone uses a fixed alternating pattern: three consecutive Gated DeltaNet layers followed by one Gated Attention layer. For the 397B flagship (60 layers):

```
15 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))
```

The rationale is a precision-vs-efficiency tradeoff. Full quadratic attention captures arbitrary pairwise token dependencies — critical for tasks requiring precise token-level interactions (entity coreference, long-range syntactic agreement). But most layers do not need this full expressiveness. GDN's linear-attention mechanism handles the bulk of sequence processing: it maintains a recurrent state matrix $S_t$ updated by a delta rule, where the correction term $(v_t - S_{t-1}^T k_t)$ is the prediction error — the difference between the incoming value and what the current state predicts for this key. Instead of naively accumulating key-value history (as in standard linear attention), DeltaNet *corrects* its memory, yielding strong in-context retrieval without quadratic cost.

The architectural bet: retaining full attention every 4th layer is enough to preserve the fine-grained token-pair interactions the model needs, while GDN handles the "coarse" contextual processing cheaply.

[See the DeltaNet hybrid deep-dive: [excerpts/qwen-3-5-deltanet.md](excerpts/qwen-3-5-deltanet.md)]

### 512 Experts with Shared Experts Restored

Qwen 3.5 doubles the expert count from 128 to 512 in the flagship model. More notably, it **reintroduces shared experts** — 1 shared expert per MoE layer, always active alongside 10 routed experts. This reverses Qwen 3's decision to drop shared experts entirely.

The reversal likely reflects empirical evidence: at 512 experts with only 10 routed active per token (~2% activation), the routing space is so vast that a single shared expert provides a stable "common knowledge" anchor without meaningfully constraining the dynamic compute budget. The cost is minimal — 1 shared expert adds ~0.2% to per-token compute — but the benefit is routing stability, especially for common linguistic patterns that every token needs regardless of specialization.

| Dimension | Qwen 3 MoE | Qwen 3.5 MoE (Flagship) |
|-----------|-----------|------------------------|
| Total experts | 128 | 512 |
| Routed active | 8 | 10 |
| Shared experts | 0 | 1 |
| Activation fraction | 6.25% | ~2.1% |
| Total params (flagship) | 235B | 397B |
| Active params (flagship) | 22B | 17B |

The active parameter count actually *decreases* from 22B to 17B despite the larger total. Qwen 3.5 pushes sparsity even further: a 23:1 total-to-active ratio, up from Qwen 3's 10:1. The bet is that 512 fine-grained experts with extreme sparsity provide better quality-per-FLOP than 128 experts with more generous activation.

### 262K Native Context and Million-Token Extension

Qwen 3.5 doubles the native context from 128K to 262K, extensible to approximately 1.01M tokens via YaRN RoPE scaling. The GDN backbone makes this practical: for 75% of layers, there is no per-token KV-cache storage — the recurrent state $S_t$ has a fixed memory footprint regardless of sequence length. Only the 25% of full-attention layers maintain a growing KV cache.

The throughput implication is dramatic. The report claims approximately **19x decoding throughput improvement** over an equivalent full-attention architecture at long contexts. This is the engineering payoff of the hybrid design: at 262K+ sequence lengths, the full-attention KV cache becomes the dominant memory and bandwidth bottleneck. Removing it from 75% of layers directly translates to higher batch sizes and lower latency.

### Native Multimodal Early Fusion

Unlike Qwen 3, which used a separate Qwen3-VL model for vision-language tasks, Qwen 3.5 performs **early fusion** — text, image, and video tokens are processed through the same backbone during pre-training. This is architecturally significant: the model learns cross-modal representations from the ground up rather than bolting on a vision encoder post-hoc.

The Qwen3.5-Omni variant extends this further with ARIA (Adaptive Rate Interleave Alignment) for streaming speech synthesis and a Thinker-Talker architecture for real-time audio generation. The key point for LLM architecture study: multimodal fusion is migrating from the application layer (separate vision/audio models) into the base architecture.

### Vocabulary Expansion: 151K to 248K

The vocabulary expands from 151,669 to 248,320 tokens, a 64% increase that supports 201 languages (up from 119). This is a direct memory-vs-coverage tradeoff: the embedding table grows by ~64%, but the model achieves better tokenization efficiency for low-resource languages (fewer tokens per word, preserving context window capacity). For a model targeting global deployment, the vocabulary expansion is a necessary investment.

---

## 8. Qwen 3.6: Training Innovation Over Architecture

Qwen 3.6 (Alibaba, April 2026) shares the exact same hybrid DeltaNet backbone as Qwen 3.5 — not a single architectural change to the layer layout, attention mechanism, or MoE configuration. Every gain comes from **training innovations**: multi-token prediction, thinking preservation across conversation turns, and refined inference defaults. The headline result is striking: **Qwen3.6-27B, a dense 27B model, outperforms the 397B-A17B MoE predecessor on agentic coding benchmarks.**

### Multi-Token Prediction (MTP)

Standard language model training predicts one token at a time: given prefix $x_1, \ldots, x_t$, predict $x_{t+1}$. MTP trains the model to predict multiple future tokens simultaneously — $x_{t+1}, x_{t+2}, \ldots, x_{t+k}$ — using additional prediction heads attached to the final layers.

The immediate application is **speculative decoding**: at inference time, the model drafts several candidate next tokens in parallel and verifies them against the main prediction head, accelerating generation when the drafts are accepted. But the deeper effect is on representation quality. MTP training forces the model to encode richer forward-looking features in its hidden states — the representation at position $t$ must carry enough information to predict not just the immediate next token but several tokens ahead. This implicitly encourages the model to capture longer-range dependencies and plan ahead during generation.

The benchmark evidence suggests MTP's representation effect is substantial, not merely a speculative-decoding speedup. Qwen3.6-27B achieves 94.1% on AIME'26 (vs. Qwen3.5-27B's 92.7%) and 77.2% on SWE-bench (vs. Qwen3.5-27B's lower score) — improvements that cannot be explained by faster decoding alone.

[See the MTP and thinking preservation deep-dive: [excerpts/qwen-3-6-mtp.md](excerpts/qwen-3-6-mtp.md)]

### Thinking Preservation Across Turns

Previous thinking models (including Qwen 3 and 3.5) discarded or compressed the `<think>...</think>` reasoning traces between conversation turns. In a multi-turn agentic workflow — where the model issues tool calls, receives results, and reasons about next steps — this means the model re-derives its scratch work at every turn. Each tool-call round forces the model to reconstruct context that was already computed but thrown away.

Qwen 3.6 introduces **thinking preservation**: the chain-of-thought reasoning from earlier turns is retained in the conversation history, visible to the model in subsequent turns. This has two effects:

1. **Reduced redundant computation.** The model does not re-derive reasoning it has already performed, saving tokens and latency in multi-turn interactions.

2. **Improved KV-cache efficiency.** In agentic workflows with many tool-call rounds, the preserved thinking traces are part of the cached prefix. The model can attend to its earlier reasoning without regenerating it, keeping total token count lower.

For agentic coding — the primary target of Qwen 3.6 — thinking preservation is transformative. A typical SWE-bench problem involves multiple rounds: read file, analyze, propose fix, verify. Without preservation, the model wastes tokens re-establishing context at each step. With preservation, the reasoning compounds across turns.

### Dense 27B Beating Sparse 397B

The most architecturally provocative result in Qwen 3.6:

| Benchmark | Qwen3.6-27B (dense) | Qwen3.5-397B-A17B (MoE) |
|-----------|---------------------|--------------------------|
| SWE-bench Pro | 53.5% | 50.9% |
| SkillsBench | 48.2% | 30.0% |
| Terminal-Bench 2.0 | 59.3% | — |
| AIME'26 | 94.1% | 91.3% |

A 27B dense model with all parameters active outperforms a 397B MoE model with 17B active parameters. The parameter-count advantage is 14.7x in favor of the MoE model, yet the dense model wins on agentic tasks.

The explanation is not that dense architectures are inherently superior to MoE. It is that **training innovations can outweigh architectural scale advantages**, at least for certain task domains. MTP and thinking preservation are particularly effective for agentic coding because these tasks require:

- **Sequential multi-step reasoning** — where MTP's richer forward-looking representations help the model plan ahead
- **Multi-turn coherence** — where thinking preservation eliminates the re-derivation tax
- **Deterministic execution** — where the tighter inference defaults (temperature 0.2, top_p 0.9 vs. 0.6/0.95) reduce the circular reasoning loops that plagued Qwen 3.5

The implication for architecture research: the architecture-vs-training dichotomy is real. A fixed backbone (Qwen 3.5's DeltaNet hybrid) can yield large gains from training-side innovations alone. The next frontier is not always a new layer type — sometimes it is a better training objective.

### Inference Defaults as an Architectural Decision

Qwen 3.6 ships with notably tighter inference defaults: temperature 0.2 and top_p 0.9, compared to Qwen 3.5's temperature 0.6 and top_p 0.95. This is not merely a configuration change — the model was trained with these defaults in mind. The RL post-training used tighter sampling to reduce circular reasoning loops where the model would generate repetitive, unproductive thinking chains.

This connects to a broader point: **inference parameters are part of the effective architecture.** A model trained with one set of sampling parameters and deployed with another may underperform. Qwen 3.6 makes this explicit by co-designing the training regime and the inference defaults.

---

## Core Insights from the Literature

### Insight 1: Inference-time compute allocation is an architecture decision, not a serving decision
**Source:** [[qwen-3|report]]

The Qwen 3 training pipeline — cold-start SFT, reasoning RL, mode fusion, general RL — manufactures dual-mode behavior that could not be achieved by prompting alone. The model's ability to switch between deliberative reasoning and direct response is baked into its weights through four stages of post-training, each building on the previous. This reframes inference-time compute: it is not something you bolt on at serving time (beam search, majority voting) but something you design into the model through training. **Guideline:** When evaluating models for deployment, treat thinking/non-thinking mode support as an architectural feature comparable to context length or parameter count — it determines the range of cost-quality tradeoffs available at inference time.

### Insight 2: The optimal training pipeline depends on model scale
**Source:** [[qwen-3|report]], [[raschka-reasoning-llms|blog]]

Qwen 3's distilled small models outperform equivalently-sized models trained through the full RL pipeline, requiring only 1/10 the GPU cost. This confirms the DeepSeek-R1 finding: pure RL requires large model capacity to discover reasoning strategies; smaller models learn reasoning more efficiently through imitation of larger teachers. The scale boundary appears to be around 10-14B — below this, distillation dominates; above it, RL becomes competitive. **Guideline:** For models under 10B parameters, invest in high-quality teacher outputs rather than RL infrastructure. RL's exploration-based learning is compute-wasteful at small scale.

### Insight 3: Extreme MoE sparsity (10:1) is viable with fine-grained routing
**Source:** [[qwen-3|report]]

The 30B-A3B model demonstrates that a 10:1 total-to-active parameter ratio is practical with 128 fine-grained experts and global-batch load balancing. This ratio was previously seen only at much larger scales (DeepSeek-V2 at 236B). The architectural enablers are: (a) many small experts that provide combinatorial routing richness, (b) global-batch balancing that leverages batch-level statistics, and (c) elimination of shared experts that simplifies the compute path. **Guideline:** When designing MoE models for memory-constrained deployment, consider 128+ fine-grained experts with high sparsity (8-10:1 ratio). The routing overhead is manageable, and the parameter efficiency is substantial — a 30B-parameter model that runs like a 3B model.

### Insight 4: Reasoning RL is extraordinarily data-efficient
**Source:** [[qwen-3|report]], [[raschka-reasoning-llms|blog]]

GRPO training on only 3,995 query-verifier pairs produces measurable reasoning improvement in a model that was pre-trained on 36 trillion tokens. This 9-orders-of-magnitude difference between pre-training data volume and RL data volume confirms that reasoning RL is not teaching knowledge — it is teaching *strategy*. The RL signal reshapes how the model deploys its existing knowledge, rewarding reasoning chains that arrive at verified correct answers. **Guideline:** For reasoning RL, invest in verifier quality (deterministic checkers, compilers) rather than dataset scale. A few thousand high-quality query-verifier pairs outperform millions of unverified examples.

### Insight 5: Hybrid linear-attention backbones unlock practical million-token context
**Source:** [[qwen-3-5|report]]

Qwen 3.5's 3:1 Gated DeltaNet-to-attention ratio eliminates per-token KV-cache storage for 75% of layers, achieving ~19x decoding throughput improvement at long contexts while preserving full-attention expressiveness every 4th layer. The delta-rule memory correction — where the update is $(v_t - S_{t-1}^T k_t)$, the prediction error rather than the raw value — gives GDN stronger in-context retrieval than naive linear attention. This is the concrete design point where sub-quadratic attention becomes production-viable: not by replacing all attention (which degrades quality) but by hybridizing at a specific ratio. **Guideline:** When designing long-context models, the 3:1 linear-to-quadratic ratio is a proven starting point. Full replacement of attention with linear variants sacrifices too much precision; full retention of quadratic attention sacrifices too much throughput. The hybrid is the engineering sweet spot.

### Insight 6: Training innovations can outweigh architectural scale advantages
**Source:** [[qwen-3-6|report]]

Qwen3.6-27B (dense, all parameters active) outperforms Qwen3.5-397B-A17B (17B active of 397B) on agentic coding benchmarks — a 14.7x parameter disadvantage overcome by multi-token prediction, thinking preservation, and refined RL. The architecture is identical (same DeltaNet hybrid backbone); the gains come entirely from training-side changes. MTP forces richer forward-looking representations; thinking preservation eliminates re-derivation waste in multi-turn workflows; tighter inference defaults reduce circular reasoning. **Guideline:** Do not assume that the next performance frontier requires a new architecture. On a fixed backbone, training objectives (MTP), conversation-level optimizations (thinking preservation), and inference co-design (sampling defaults trained into the model) are high-leverage, lower-risk interventions. Architecture and training are separable levers — explore both before concluding that one is exhausted.

---

## Key Takeaways

1. **Dual-mode inference is a training-pipeline achievement, not an architectural one.** The Transformer blocks in Qwen 3 are standard. The thinking/non-thinking capability is manufactured through a four-stage post-training pipeline (cold-start SFT, reasoning RL, mode fusion, general RL). The architecture is the *pipeline*.

2. **The dense lineup reveals scaling patterns.** KV heads fixed at 8; embedding tying below 4B; depth as the primary scaling lever above 14B; context length jumping at 4B. These are pragmatic engineering choices, not theoretical optimizations.

3. **Qwen 3's MoE models push extreme sparsity.** 128 experts, 8 active, no shared experts, 10:1 total-to-active ratio. This bets on fine-grained routing granularity over shared-expert safety nets, and global-batch balancing over per-sequence balancing.

4. **Distillation beats RL at small scale.** Models under ~10B learn reasoning more efficiently from teacher outputs (1/10 the cost) than from RL exploration. RL requires large capacity to discover reasoning strategies; small models should imitate rather than explore.

5. **The thinking budget is user-controlled compute allocation.** Qwen 3 exposes a knob that directly trades latency and cost for quality, within a single model deployment. This operationalizes the test-time compute scaling literature.

6. **Reasoning RL is a strategy optimizer, not a knowledge injector.** 3,995 verified examples reshape how 36T tokens of pre-trained knowledge are deployed. The ratio tells you what RL is doing: refining policy, not teaching facts.

7. **Unified models halve deployment cost.** A single Qwen 3 checkpoint replaces separate reasoning and chat models, eliminating application-level routing and reducing GPU allocation by 2x compared to dual-model deployments.

8. **Hybrid DeltaNet is the first production-viable sub-quadratic backbone.** Qwen 3.5's 3:1 GDN-to-attention ratio removes KV-cache cost from 75% of layers, enabling 262K native context (1M extended) with ~19x throughput improvement. The delta-rule correction gives GDN better retrieval than naive linear attention. This is not a research prototype — it is deployed at 397B-parameter scale.

9. **Shared experts return when sparsity gets extreme enough.** Qwen 3 dropped shared experts at 128/8 routing (6.25% activation). Qwen 3.5 restores them at 512/10 routing (~2% activation). At ultra-high sparsity, one shared expert provides routing stability at negligible cost.

10. **MTP reshapes representations, not just decoding speed.** Multi-token prediction training in Qwen 3.6 forces richer forward-looking hidden states. The 27B dense model's gains over the 397B MoE on sequential reasoning tasks cannot be explained by speculative decoding alone — MTP changes what the model learns, not just how fast it generates.

11. **Thinking preservation is the missing piece for agentic workflows.** Retaining chain-of-thought traces across conversation turns eliminates re-derivation waste and enables compounding reasoning. Qwen 3.6-27B's 77% relative improvement over Qwen 3.5-397B on SkillsBench is largely attributable to this single training innovation.

12. **Architecture and training are separable levers.** Qwen 3.5 changed the architecture (DeltaNet hybrid). Qwen 3.6 changed only the training (MTP, thinking preservation, refined RL) on the same architecture. Both produced large gains. The lesson: exhaust neither lever before exploring the other.

---

## References

- [[qwen-3|Qwen Team, "Qwen3 Technical Report" (2025) (report)]] — Primary source for Qwen 3 architecture, training pipeline, and benchmarks
- [[qwen-3-5|Qwen Team, "Qwen3.5 Technical Report" (2026) (report)]] — Hybrid DeltaNet backbone, 512-expert MoE, 262K native context, multimodal early fusion
- [[qwen-3-6|Qwen Team, "Qwen3.6 Technical Report" (2026) (report)]] — Multi-token prediction, thinking preservation, dense 27B beating sparse 397B on agentic tasks
- [[raschka-reasoning-llms|Raschka, "Understanding Reasoning LLMs" (2025) (blog)]] — Four approaches to reasoning: inference scaling, pure RL, SFT+RL, distillation; scale-dependent strategy selection
- [[weng-why-we-think|Weng, "Why We Think" (2025) (blog)]] — Test-time compute scaling laws, dual-process theory, faithfulness of reasoning, continuous-space thinking
- [[ch-07]] — GQA mechanism and KV cache reduction strategies
- [[ch-12]] — MoE fundamentals: gating, routing, load balancing, expert specialization
- [[ch-14]] — LLM training pipeline structure: pre-training stages, SFT, RLHF
- [[ch-19]] — Related case study
- [[ch-18]] — Related case study
