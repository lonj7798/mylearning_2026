# Chapter 19: Case Study — DeepSeek V3 and V4

<!-- scope: DeepSeek-V3 and V4 architecture deep dive — MLA, DeepSeekMoE, auxiliary-loss-free balancing, MTP, FP8, hybrid sparse attention (CSA+HCA), Engram conditional memory, manifold-constrained hyper-connections, FP4 experts, Muon optimizer, 1M-token context
     deps: [[ch-07]], [[ch-14]]
     see-also: [[ch-18]], [[ch-23]]
-->

## Overview

DeepSeek-V3 is a 671B-parameter Mixture-of-Experts model that activates only 37B parameters per token — roughly 5.5% of total capacity. It was trained on 14.8 trillion tokens for a total cost of 2.788 million H800 GPU-hours, or approximately **$5.6 million** at $2/GPU-hour. That figure is an order of magnitude below what comparable frontier models cost. The entire training run completed without a single irrecoverable loss spike or rollback — an extraordinary stability achievement at this scale.

This chapter dissects the architectural decisions that made DeepSeek-V3 both cheap and competitive, then traces how its successor — DeepSeek-V4 ([[deepseek-v4|report]]) — pushes every axis further. Each decision connects to foundational concepts from earlier chapters: Multi-head Latent Attention ([[ch-07]]) compresses the KV cache; DeepSeekMoE ([[ch-14]]) routes tokens through fine-grained experts; auxiliary-loss-free balancing eliminates a persistent quality penalty from MoE training; multi-token prediction densifies the training signal; and FP8 mixed-precision training halves compute cost at the hardware level. None of these ideas is individually unprecedented — DeepSeek's contribution is the *integration* of all five into a single coherent system, with each choice reinforcing the others.

The result: a model that matches GPT-4o and Claude 3.5 Sonnet on standard benchmarks ([[deepseek-v3|report]]) while costing a fraction to train. V4 then extends this foundation with hybrid sparse attention, a deterministic memory module, manifold-constrained residual connections, FP4 expert precision, and the Muon optimizer — scaling to 1.6T parameters and 1M-token context while *reducing* per-token inference cost. Understanding *how* requires tracing each architectural choice through the lens of cost, quality, and systems design.

---

## 1. Architecture at a Glance

| Component | Value |
|-----------|-------|
| Total Parameters | 671B |
| Active Parameters per Token | 37B (~5.5%) |
| Transformer Layers | 61 |
| Hidden Dimension | 7,168 |
| Attention Heads (queries) | 128 |
| KV Compression Dim ($d_c$) | 512 |
| Decoupled RoPE Dim ($d_h^R$) | 64 |
| Routed Experts per Layer | 256 |
| Shared Experts per Layer | 1 |
| Active Routed Experts (top-K) | 8 |
| Context Length | 128K tokens |
| Vocabulary Size | 129,280 tokens |
| Training Tokens | 14.8 trillion |
| Total Training Compute | 2.788M H800 GPU-hours |

Three facts jump out. First, the activation ratio (37B/671B = 5.5%) is remarkably low — Mixtral activates ~22% of parameters (12.9B/46.7B) and Llama 4 Scout activates ~10.5% (17B/162B). DeepSeek achieves this through extreme expert granularity: 256 small experts rather than 8 or 16 large ones. Second, the KV cache stores only 576 dimensions per token per layer (512 latent + 64 RoPE) instead of the 32,768 that standard MHA would require. Third, the entire model trained on 14.8T tokens — well beyond Chinchilla-optimal for 37B active parameters — because the MoE structure lets additional parameters come cheaply while FP8 halves the FLOP cost per token.

> **See:** [MLA Compression Diagram](figures/mla-compression.html) for a visual walkthrough of the down-projection, caching, and up-projection pipeline.

---

## 2. Multi-head Latent Attention (MLA)

MLA was introduced in DeepSeek-V2 ([[deepseek-v2|report]]) and carried forward without modification to V3. The core idea, covered in [[ch-07]], is to compress keys and values into a shared low-rank latent vector rather than storing full per-head KV vectors.

### 2.1 The Compression Pipeline

Standard MHA stores keys and values for each of $H$ heads at each position. For DeepSeek-V3 with $H = 128$ and $d_k = 128$, that means caching $2 \times 128 \times 128 = 32{,}768$ dimensions per token per layer. MLA replaces this with three steps:

**Step 1 — Down-project to latent:**

$$c_t = W_{\text{DKV}} \, x_t \qquad W_{\text{DKV}} \in \mathbb{R}^{d_c \times d_\text{model}}, \quad d_c = 512$$

The hidden state $x_t \in \mathbb{R}^{7168}$ is projected down to a 512-dimensional latent $c_t$. This is the **only** vector cached per token per layer.

**Step 2 — Up-project to reconstruct K, V per head:**

$$k_t^{(h)} = W_{UK}^{(h)} \, c_t, \qquad v_t^{(h)} = W_{UV}^{(h)} \, c_t$$

Each head reconstructs its own keys and values from the shared latent. The up-projection matrices $W_{UK}^{(h)}, W_{UV}^{(h)} \in \mathbb{R}^{d_k \times d_c}$ are per-head parameters (not cached — they are static model weights).

**Step 3 — Decoupled RoPE component:**

Rotary position embeddings are position-dependent and must be applied to keys *before* caching. But the compressed latent $c_t$ is position-agnostic — the up-projection cannot recover where token $t$ appeared in the sequence. DeepSeek solves this by maintaining a small separate key vector $k_t^R \in \mathbb{R}^{64}$ that carries only the RoPE-encoded positional signal. This is cached alongside $c_t$.

Total cache per token per layer: $d_c + d_h^R = 512 + 64 = 576$ dimensions.

### 2.2 Why This Works: The Low-Rank Hypothesis

The compression from 32,768 to 576 dimensions (a **93.3% reduction**) seems aggressive. Why doesn't it destroy capacity? Two reasons:

1. **KV representations are highly redundant across heads.** In standard MHA, many heads learn near-identical key/value patterns. The low-rank latent forces the model to factor out this shared structure, learning a compact common basis. The up-projection matrices then reconstruct head-specific variations from that shared basis.

2. **The bottleneck regularizes.** DeepSeek-V2 ablations showed MLA *outperforming* MHA on MMLU (78.5% vs 71.3% for the 67B predecessor). The compressed latent acts as an information bottleneck that forces more structured, less redundant representations — similar to how autoencoders learn useful features via dimensionality reduction.

### 2.3 Queries Are Also Compressed

An easily overlooked detail: MLA compresses queries too, not just keys and values. The query compression dimension is $d_c' = 1{,}536$ — larger than the KV compression dimension (512) because queries need higher capacity to express diverse attention patterns. During inference this doesn't matter (queries aren't cached), but during training it reduces activation memory.

### 2.4 MLA vs GQA: The Quantitative Case

For a model at DeepSeek-V3's scale ($H = 128$, $d_k = 128$, 61 layers):

| Attention Variant | Cache per Token per Layer | Cache @ 128K Context (FP16) |
|------------------|--------------------------|----------------------------|
| MHA (128 KV heads) | 32,768 dims | ~490 GB |
| GQA-8 (8 KV groups) | 2,048 dims | ~30.7 GB |
| GQA-1 (MQA) | 256 dims | ~3.8 GB |
| **MLA** | **576 dims** | **~8.6 GB** |

MLA's 576-dim cache sits between MQA and GQA-8, but unlike MQA, it preserves per-head diversity through the up-projection. Each head can learn its own reconstruction from the shared latent, whereas MQA forces all heads to use literally identical K/V vectors.

---

## 3. DeepSeekMoE: Fine-Grained Expert Routing

### 3.1 Architecture: 256 Routed + 1 Shared

Each MoE layer in DeepSeek-V3 contains 257 expert FFNs: 256 *routed* experts selected by a gating network, plus 1 *shared* expert that processes every token unconditionally. Each token activates its top-8 routed experts, so the per-token compute is:

$$\text{FLOPs per token} = \text{shared expert} + 8 \times \text{routed expert} = 9 \text{ expert FFNs}$$

This is a fundamentally different design point from Mixtral (8 experts, top-2) or Llama 4 Scout (16 experts, top-1). DeepSeek chooses **many small experts** over **few large experts**.

### 3.2 Why Fine-Grained: Combinatorial Flexibility

With 256 experts and top-8 selection, the number of possible expert combinations per token is $\binom{256}{8} \approx 4.4 \times 10^{13}$. Compare this to Mixtral's $\binom{8}{2} = 28$ combinations.

This astronomical combinatorial space means the model can assign nearly unique expert subsets to different input types. A token appearing in a mathematical proof can be routed to an entirely different set of 8 experts than the same token in a Python docstring — and both can differ from the routing when that token appears in Chinese prose. Fine-grained routing converts the expert selection itself into a *representational mechanism*, not just a compute-allocation mechanism.

The cost of fine-granularity is routing complexity. With 256 experts distributed across multiple nodes, the all-to-all communication pattern during training becomes a distributed systems challenge. DeepSeek mitigates this with **node-limited routing**: each token can access experts on at most 4 nodes (out of 8 in their cluster), bounding communication while preserving most routing flexibility.

### 3.3 The Shared Expert: Always-On Common Knowledge

The single shared expert processes every token regardless of routing decisions. Its purpose is to capture **high-frequency, domain-agnostic patterns** — syntactic structure, common token transitions, formatting conventions — that would otherwise be redundantly learned across many routed experts.

This separation is important. Without a shared expert, the routing network faces a dilemma: assign common patterns to many experts (wasting capacity on redundancy) or concentrate them in a few (creating load imbalance). The shared expert resolves this by providing a guaranteed channel for common knowledge, freeing routed experts to specialize.

DeepSeek-V2 used 2 shared experts; V3 reduced this to 1. The report does not elaborate on why, but the likely reason is that a single larger shared expert with the saved parameter budget allocated to more routed experts yields better specialization.

> **See:** [DeepSeekMoE Routing Visualization](figures/deepseek-moe-routing.html) for an interactive diagram of token-to-expert assignment with the shared expert highlighted.

---

## 4. Auxiliary-Loss-Free Load Balancing

### 4.1 The Problem with Auxiliary Losses

MoE models suffer from **routing collapse**: without intervention, the gating network converges to sending most tokens to a small subset of experts, leaving others underutilized. The standard remedy is an auxiliary loss that penalizes imbalanced expert loads — a term added to the training objective that pushes the router toward uniform distribution.

The problem: auxiliary losses **directly degrade model quality**. They inject a gradient signal that has nothing to do with language modeling — the model is being told to sacrifice prediction accuracy in order to balance load. DeepSeek-V2 used three separate auxiliary losses (expert-level, device-level, and communication-level) with carefully tuned coefficients ($\alpha = 0.003, 0.05, 0.02$). Getting these coefficients wrong either causes routing collapse (too weak) or measurable quality degradation (too strong).

### 4.2 The Bias-Based Solution

DeepSeek-V3 replaces gradient-based auxiliary losses with a **non-gradient bias mechanism**:

1. Each expert $e$ maintains a bias term $b_e$ that is added to its gating score during routing (but **not** during loss computation).
2. After each training step, if expert $e$ is overloaded (receives more tokens than its fair share), $b_e$ is decreased by a small amount $\gamma$. If underloaded, $b_e$ is increased by $\gamma$.
3. The bias adjustment is a simple additive update — it does **not** flow gradients back through the gating network.

$$\text{routing score}_e = g_e(x) + b_e$$

$$b_e \leftarrow b_e + \gamma \cdot \text{sign}(\text{target\_load} - \text{actual\_load}_e)$$

The speed parameter $\gamma$ controls how aggressively biases adjust. This is a control-theoretic approach: the biases act as an integral controller that nudges routing toward balance without corrupting the training objective.

### 4.3 Why This Is Better

The key insight: **load balancing is a systems constraint, not a learning objective.** The auxiliary loss approach conflates two fundamentally different goals — learning good representations and distributing compute evenly. The bias approach cleanly separates them:

- The gating network's gradients come *exclusively* from the language modeling loss. It learns to route tokens to the experts that produce the best predictions.
- The bias terms shift routing decisions at the margin. A token that the gating network scores similarly for experts $e_1$ and $e_2$ will be nudged toward the underloaded one.

This separation produces two benefits. First, **no quality penalty** — the training objective is pure language modeling. Second, **better training stability** — DeepSeek-V3 completed 14.8T tokens with zero loss spikes and zero rollbacks. The auxiliary-loss-free approach eliminates a known source of training instability in MoE models: sudden changes in auxiliary loss coefficients or expert load distributions that cascade into loss spikes.

A small sequence-wise auxiliary loss with an extremely small coefficient is still used as a complement, but its coefficient is so small that it effectively serves as a safety net rather than the primary balancing mechanism.

### 4.4 No Token Dropping

A direct consequence of stable load balancing: DeepSeek-V3 **never drops tokens** during training or inference. DeepSeek-V2 dropped tokens with the lowest affinity scores when devices exceeded their compute budget, with ~10% of training sequences exempt from dropping to preserve information. V3's bias mechanism keeps loads sufficiently balanced that dropping is unnecessary — every token sees its preferred experts.

---

## 5. Multi-Token Prediction (MTP)

### 5.1 Beyond Next-Token Prediction

Standard language model training predicts one token per position: given tokens $t_1, \ldots, t_i$, predict $t_{i+1}$. Multi-token prediction extends this: predict $t_{i+1}, t_{i+2}, \ldots, t_{i+D}$ simultaneously, where $D$ is the prediction depth.

DeepSeek-V3 implements this through $D$ sequential **MTP modules**, each predicting one additional future token. The combined training loss is:

$$\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda \sum_{d=1}^{D} \mathcal{L}_{\text{MTP}}^{(d)}$$

where $\lambda$ controls the weight of MTP losses relative to the standard next-token prediction loss.

### 5.2 Maintaining the Causal Chain

A critical design choice: each MTP module $d$ receives the output of module $d-1$ and the embedding of the token that module $d-1$ was predicting. This maintains a **complete causal chain** — module $d$ has access to the identity of all $d-1$ intermediate predicted tokens, not just the original context. Without this causal dependency, later modules would be predicting future tokens without knowing what the model expects the intervening tokens to be, which would provide a much weaker training signal.

The MTP modules share the main model's embedding and output head — they do not introduce a separate vocabulary projection. This saves parameters and ensures consistent token representations across prediction depths.

### 5.3 Why MTP Helps Training

MTP provides **denser training signal** per forward pass. Standard next-token prediction wastes the rich intermediate representations: every hidden state is used to predict only one token, but contains information relevant to predicting many future tokens. MTP extracts this additional signal without proportionally increasing compute — the MTP modules are lightweight relative to the main Transformer. The foundational argument for multi-token prediction was developed by Gloeckle et al. (2024) ([[multi-token-prediction|paper]]), who showed that training with 4 prediction heads improves sample efficiency and enables self-speculative decoding (3x speedup) without a separate draft model.

The DeepSeek-V3 report frames MTP as improving "pre-training performance," and the ablations confirm quality gains on downstream benchmarks. The mechanism likely works because predicting multiple tokens forces the model to build representations that are less myopic — capturing not just what comes next, but the broader trajectory of the text.

### 5.4 MTP for Speculative Decoding

An elegant bonus: the MTP modules trained during pre-training can be repurposed as **draft models for speculative decoding** during inference. Instead of training a separate small model to produce candidate tokens, DeepSeek-V3 uses its own MTP heads. The report claims a **1.8x inference speedup** with this approach. The MTP modules are well-calibrated draft models because they were trained on the same data, with the same representations, to predict exactly the tokens that the main model will verify.

---

## 6. FP8 Mixed-Precision Training

### 6.1 The Precision Hierarchy

DeepSeek-V3 is one of the first frontier models to use FP8 (8-bit floating point) for the bulk of training computation. The precision allocation is carefully stratified:

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Linear layer forward/backward (GEMMs) | **FP8** | Bulk compute — 2x throughput |
| Activations (stored for backward) | **FP8** | Memory savings |
| Embeddings, output head | FP32 | Vocabulary-scale precision matters |
| Gating / routing | FP32 | Routing decisions are discrete and sensitive |
| Normalization (RMSNorm) | FP32 | Running statistics need precision |
| Attention scores | FP16 | Softmax stability |
| Master weights, optimizer states | FP32 | Accumulation accuracy |

The core insight: most FLOPs in a Transformer are in the linear projections (attention QKV projections, attention output, FFN up/down/gate projections). These are the GEMMs (General Matrix Multiplications) that FP8 accelerates. Everything else — normalization, attention softmax, gating — remains in higher precision because these operations are more numerically sensitive but consume a tiny fraction of total FLOPs.

### 6.2 Fine-Grained Quantization

Naive FP8 quantization would use a single scaling factor per tensor. DeepSeek-V3 uses **fine-grained quantization**:

- **Activations:** tile-wise quantization with tiles of size $1 \times 128$. Each tile of 128 elements gets its own scaling factor.
- **Weights:** block-wise quantization with blocks of size $128 \times 128$. Each block of 16,384 elements gets its own scaling factor.

The scaling factors are computed along the inner dimension of the GEMM, ensuring that elements that are multiplied together share compatible dynamic ranges. This dramatically reduces quantization error compared to per-tensor scaling.

### 6.3 High-Precision Accumulation

FP8 multiplication produces FP16 intermediate products, which are accumulated in FP32. But hardware constraints mean the accumulator may not be FP32 for every multiply-add. DeepSeek-V3 enforces **accumulation into FP32 every 128 elements** — the inner product is computed in chunks of 128, each chunk accumulated in FP32, then the partial sums are combined. This prevents the numerical drift that would occur if thousands of FP8 products were accumulated in lower precision.

### 6.4 Validation: Is Quality Preserved?

The report claims **less than 0.25% relative loss error** compared to BF16 training. To put this in perspective: at a loss of ~2.0, a 0.25% relative error is 0.005 nats — well within the noise of different random seeds. The fine-grained quantization and high-precision accumulation make FP8 training practically lossless.

The compute savings are substantial: FP8 GEMMs run at approximately **2x the throughput** of BF16 on H800 GPUs (which have native FP8 tensor cores). Since GEMMs dominate training time, this translates to a roughly 40-50% reduction in wall-clock training time at equivalent quality.

---

## 7. DeepSeek-V4: The Next Leap

DeepSeek-V4-Pro is a 1.6T-parameter MoE model that activates 49B parameters per token — roughly 3% of total capacity. Where V3 was the proof that integrated MoE engineering could match frontier dense models at a fraction of the cost, V4 asks a harder question: can you scale to 1M-token context and 2.4x total parameters while *decreasing* per-token inference cost? The answer requires five new architectural components, each targeting a different bottleneck that V3 could not solve by scaling alone.

> **See:** [V3 vs V4 Architectural Evolution](figures/v3-vs-v4-evolution.html) for a visual comparison of how each component evolved.

### 7.1 Hybrid Sparse Attention: CSA + HCA

V3's MLA stores a compressed 576-dim latent per token per layer. At 128K context, this is manageable. At 1M tokens, even 576 dims per token per layer becomes prohibitive — and the attention computation itself is O(L^2), which at L = 1M is 7,800x the cost of L = 128K.

V4 replaces uniform MLA with a **hybrid of two complementary sparse attention mechanisms**, collectively called DSA2:

**Compressed Sparse Attention (CSA)** uses a *lightning indexer* — a lightweight FP8 batched matrix multiply — to score all keys in a sequence, then applies a GPU partial-sort to select only the top-k most relevant tokens per query. This converts attention from O(L^2) to O(Lk), where k is a fixed budget of selected tokens per query. The indexer is cheap because it operates on compressed representations, not full keys.

**Heavily Compressed Attention (HCA)** applies extreme KV-cache compression for heads that need broad but coarse context awareness — think "what general topic is being discussed 500K tokens back" rather than "what exact word appeared at position 312,847." HCA heads store far fewer dimensions per token than even MLA, trading fine-grained recall for massive context coverage.

The critical design choice: **per-head pathway assignment**. Each attention head in a layer is assigned to either the CSA or HCA pathway. CSA heads get high-resolution, sparse access to specific tokens. HCA heads get low-resolution, dense access to the entire context. The hybrid exploits the empirical observation that different attention heads serve fundamentally different roles — some are "lookup" heads (CSA) and some are "summary" heads (HCA).

The result at 1M tokens: V4-Pro requires only **27% of single-token inference FLOPs** and **10% of KV cache** compared to V3.2 (a V3 variant). That 10x KV-cache reduction *on top of* MLA's existing 57x reduction over standard MHA makes million-token inference practically viable.

### 7.2 Manifold-Constrained Hyper-Connections (mHC)

Standard residual connections add the layer output directly to the residual stream: $x_{l+1} = x_l + f_l(x_l)$. This works well for moderate depth, but as models grow deeper, two failure modes emerge: gradient explosion (residual magnitudes grow without bound) and representation collapse (the residual stream overwhelms the layer contribution, making deeper layers ineffective).

V4 replaces standard residuals with **manifold-constrained hyper-connections (mHC)**, based on arXiv:2512.24880. Instead of a simple addition, mHC mixes the residual and layer output through a **doubly stochastic mixing matrix** constrained to the Birkhoff Polytope:

$$x_{l+1} = M_l \begin{pmatrix} x_l \\ f_l(x_l) \end{pmatrix}$$

where $M_l$ is a mixing matrix projected onto the Birkhoff Polytope using the **Sinkhorn-Knopp algorithm** — iterative row and column normalization that converges to a doubly stochastic matrix.

Why doubly stochastic? A doubly stochastic matrix preserves the L1 norm of its input. This is a **manifold constraint**: regardless of depth, the signal magnitude neither explodes nor collapses. The identity matrix is doubly stochastic (recovering standard residuals as a special case), but the model can learn richer cross-channel mixing — signals from earlier layers can be selectively amplified or attenuated without breaking the norm-preservation guarantee.

V4 uses a **4x wider residual stream** with mHC, adding only 6.7% training time overhead. The wider stream provides more channels for the mixing matrix to operate on, enabling richer cross-layer information flow while the Birkhoff constraint keeps everything stable.

### 7.3 Engram Conditional Memory

Every attention mechanism — MHA, MLA, CSA, HCA — answers queries by *computing* over stored representations. But many queries are factual lookups: "What is the capital of France?" doesn't require flexible attention over context; it requires retrieving a memorized fact. Using attention for pure recall wastes both compute and context bandwidth.

Engram ([[deepseek-v4|report]], arXiv:2601.07372) introduces a **deterministic memory axis** orthogonal to both attention and MoE computation:

1. **Multi-head hashing** maps suffix N-grams of the current position (up to trigrams in V4) into prime-sized embedding buckets. Each N-gram hash produces a lookup key into a large embedding table.
2. A small **depthwise convolution** over the N-gram context produces a **context-aware gating scalar** $g \in [0, 1]$ that controls how much of the retrieved embedding is injected into the hidden state.
3. The injection is additive: $x_l' = x_l + g \cdot \text{Engram}(x_l)$.

The lookup is **O(1)** in sequence length — it depends only on the local N-gram window, not on the full context. This is the core insight: *don't calculate when you can look up*. By offloading factual recall to Engram, attention heads are freed to focus on relational reasoning that genuinely requires flexible computation.

V4 places Engram modules at **layers 2 and 15** with 8 heads and dimension 1280. The placement is strategic: early layers (layer 2) handle surface-level lexical patterns; a mid-depth layer (15) handles entity-level factual recall. The Engram paper establishes a **Sparsity Allocation Law**: under a fixed sparse parameter budget, the optimal allocation is approximately 20-25% memory (Engram) and 75-80% computation (MoE).

### 7.4 Expanded MoE: 384 Experts in FP4

V4 scales the MoE axis from V3's 256 routed experts to **384 routed experts** per layer, but *reduces* the number of active experts from 8 to **6 per token**. The net effect: even more expert combinations ($\binom{384}{6} \approx 3.3 \times 10^{13}$) with a lower activation ratio (~3% vs ~5.5%).

The bigger change is precision. V3 trained expert parameters in FP8; V4 drops to **FP4 — 4-bit floating point** for all MoE expert parameters, with most other parameters in FP8. FP4 halves expert memory and compute cost relative to V3's already-aggressive FP8 regime. The fine-grained quantization techniques from V3 (tile-wise activations, block-wise weights, periodic FP32 accumulation) presumably carry forward at the FP4 level, though the full technical report with ablations has not yet been published.

V4 retains V3's auxiliary-loss-free load balancing with bias-based routing and no token dropping.

### 7.5 Muon Optimizer

V4 replaces AdamW with the **Muon optimizer** for pre-training. Muon delivers faster convergence and greater training stability than AdamW, though detailed comparisons specific to V4's scale have not been published. The optimizer change is particularly significant given V4's training scale: >32T tokens (more than 2x V3's 14.8T).

### 7.6 Training and Post-Training

V4's pre-training uses a 32K context window, extended to 1M tokens during post-training stages. The post-training pipeline is a notable departure from V3's simpler SFT-then-RL approach:

1. **Domain-specific expert cultivation:** Independent SFT and RL (GRPO) per domain — reasoning, coding, math, etc. Each domain builds specialized expert capabilities in isolation.
2. **Unified model consolidation:** On-policy distillation merges domain-specific expert proficiencies into a single model.

This two-stage paradigm enables deeper specialization before unification. V3's single-pass SFT/RL could not push individual domains as far because different domains' gradient signals competed during joint training.

V4 also introduces the **DeepSeek-V4-Flash** variant (284B total, 13B active) — a smaller model sharing the same architectural innovations, confirming that the hybrid attention and Engram designs scale down effectively.

> **See:** [DeepSeek-V4 Innovations Deep Dive](excerpts/deepseek-v4-innovations.md) for a detailed comparison of each V4 component against its V3 predecessor.

---

## 8. Systems Design: DualPipe and Communication Overlap

### 8.1 The Communication Challenge

MoE training requires **all-to-all communication** at every MoE layer: tokens must be dispatched to their assigned experts (which may reside on different nodes) and the expert outputs must be gathered back. With 256 experts across 8 nodes and 64-way expert parallelism, this communication is substantial.

DeepSeek-V3 eliminates tensor parallelism entirely — a significant simplification. The parallelism strategy is:

- **16-way Pipeline Parallelism:** the 61 Transformer layers are distributed across 16 pipeline stages
- **64-way Expert Parallelism:** experts are distributed across 64 GPUs (8 nodes x 8 GPUs)
- **ZeRO-1 Data Parallelism:** optimizer states are sharded

### 8.2 DualPipe: Hiding Communication Behind Compute

The DualPipe algorithm overlaps all-to-all expert communication with computation from *other* micro-batches. While one micro-batch is waiting for expert dispatch/gather, the GPU is computing attention or other operations for a different micro-batch. This achieves **near-zero communication overhead** — the all-to-all transfers happen in the background while the GPU stays busy.

This is why eliminating tensor parallelism matters: tensor parallelism requires synchronous intra-layer communication that *cannot* be overlapped with compute from the same layer. By using only pipeline and expert parallelism, DeepSeek-V3 ensures that all communication can be pipelined behind useful work.

---

## 9. Training Recipe and Stability

### 9.1 Pre-Training

- **Data:** 14.8 trillion tokens of diverse, high-quality text
- **Optimizer:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $= 0.1$)
- **Sequence length:** 4K initially, extended to 32K then 128K post-training
- **Batch size ramping:** gradual increase over training
- **Zero loss spikes:** the combination of auxiliary-loss-free balancing, FP8 with fine-grained quantization, and careful learning rate scheduling produced a training run with zero irrecoverable loss spikes

### 9.2 Post-Training and R1 Distillation

DeepSeek-V3's post-training incorporates **distillation from DeepSeek-R1** ([[deepseek-r1|report]]), transferring chain-of-thought reasoning capabilities. R1 was trained with pure RL (GRPO) on the V3 base model, and its reasoning patterns are distilled back into V3 via supervised fine-tuning on R1-generated data. This creates a feedback loop: V3 is the base for R1, and R1's reasoning is distilled back to improve V3.

The post-training cost is minimal: approximately 5K H800 GPU-hours for SFT + RL, compared to 2.664M hours for pre-training. Post-training is less than 0.2% of total training compute.

---

## 10. Cost Analysis: $5.6M for a Frontier Model

### 10.1 Where the Savings Come From

| Decision | Savings Mechanism | Estimated Impact |
|----------|-------------------|------------------|
| MoE (5.5% activation) | 671B capacity at 37B compute cost | ~18x parameter efficiency |
| FP8 training | 2x GEMM throughput | ~40-50% wall-clock reduction |
| No tensor parallelism | Simpler communication, DualPipe overlap | ~10-15% efficiency gain |
| Auxiliary-loss-free balancing | Zero rollbacks, no wasted compute | Unknown, but zero rollbacks is significant |
| MLA | Smaller activations during training | 42.5% training cost reduction (V2 over predecessor) |

### 10.2 The Compound Effect

These savings compound. MoE reduces the per-token FLOP count. FP8 halves the cost of each FLOP. DualPipe eliminates communication idle time. Auxiliary-loss-free balancing prevents training restarts. MLA reduces activation memory, enabling larger batch sizes.

The total training budget of 2.788M H800 GPU-hours at $2/GPU-hour gives $5.576M. Compare this to estimates of $100M+ for GPT-4 and similar figures for Claude 3. Even accounting for the fact that DeepSeek's GPU cost is likely lower than US market rates, the architectural efficiency gains are real and substantial.

### 10.3 What This Does NOT Mean

The $5.6M figure does not include:
- The research and engineering effort to develop MLA, DeepSeekMoE, and the FP8 framework
- The cost of training DeepSeek-V2 (the predecessor that validated these architectural choices)
- Data curation and cleaning costs
- Post-training costs (though these are small: ~5K GPU-hours)

The cost figure represents *compute* for the final training run, not total R&D investment. But even with this caveat, the architectural innovations demonstrably reduced training compute by an order of magnitude compared to dense models of similar capability.

---

## Core Insights from the Literature

### Insight 1: Auxiliary losses are a quality tax on MoE training
**Source:** [[deepseek-v3|DeepSeek-V3 Technical Report (report)]]

Traditional MoE load balancing injects a gradient signal into the training objective that has nothing to do with language modeling. The bias-based approach cleanly separates the *learning* problem (predict the next token well) from the *systems* problem (distribute compute evenly). This separation is not just an engineering convenience — it produces measurably better models because the gating network optimizes purely for prediction quality. **Guideline:** When routing instability appears in MoE training, reach for adaptive bias terms before auxiliary losses. If auxiliary losses are necessary, keep coefficients extremely small and treat them as safety nets, not primary mechanisms.

### Insight 2: Fine-grained MoE routing is a representational choice, not just a compute choice
**Source:** [[deepseek-v3|DeepSeek-V3 Technical Report (report)]], [[deepseek-v2|DeepSeek-V2 Technical Report (report)]]

The difference between 8 experts (top-2) and 256 experts (top-8) is not just about parameter efficiency — it fundamentally changes what the routing mechanism can express. With $\binom{256}{8}$ possible combinations, the expert selection itself becomes a high-dimensional discrete feature of each token. This converts routing from a static resource allocation mechanism into a dynamic representation mechanism that adapts per-token, per-layer. **Guideline:** When designing MoE architectures, treat the number of experts and top-K as representational hyperparameters, not just efficiency knobs. More, smaller experts with higher top-K provide exponentially more routing diversity.

### Insight 3: Training precision is an architecture decision, not a post-hoc optimization
**Source:** [[deepseek-v3|DeepSeek-V3 Technical Report (report)]]

DeepSeek-V3 designed for FP8 from the start — tile-wise activation quantization, block-wise weight quantization, periodic FP32 accumulation, and careful precision allocation per component. This is qualitatively different from post-training quantization (which compresses an already-trained model) or naive mixed-precision (which just toggles between FP16 and FP32). Fine-grained training-time quantization treats precision as a per-component architectural hyperparameter. **Guideline:** For future large-scale training, choose precision per component based on numerical sensitivity, not by applying a blanket precision level. GEMMs tolerate FP8; attention softmax and normalization do not.

### Insight 4: Separate memory from computation — don't calculate when you can look up
**Source:** [[deepseek-v4|DeepSeek-V4 Technical Report (report)]], Engram (arXiv:2601.07372)

DeepSeek-V4's Engram module reveals that a large fraction of what attention does in practice is factual recall — retrieving memorized associations rather than performing relational reasoning. By introducing a deterministic O(1) memory axis (hashed N-gram embeddings with context-aware gating), V4 offloads pure-recall workload from attention entirely. The Sparsity Allocation Law formalizes this: under a fixed sparse parameter budget, ~20-25% should go to memory and ~75-80% to computation. This separation compounds with hybrid sparse attention (CSA+HCA): attention heads freed from factual recall can focus on relational reasoning, and the hybrid further splits "precise lookup" (CSA) from "broad summary" (HCA) heads. The result is a 10x KV-cache reduction and 73% FLOP reduction at 1M context versus V3. **Guideline:** When scaling to long contexts, audit what fraction of attention heads perform factual recall versus relational reasoning. Factual recall heads are candidates for replacement by deterministic memory modules — they waste quadratic attention compute on what is fundamentally a hash-table lookup.

### Insight 5: The KV bottleneck is a feature, not a bug
**Source:** [[deepseek-v2|DeepSeek-V2 Technical Report (report)]]

MLA's 93.3% KV cache compression *improves* quality over standard MHA. The low-rank latent forces the model to learn structured, non-redundant representations — it regularizes attention. This inverts the conventional wisdom that compression is a necessary evil traded against quality. When the baseline representations contain massive redundancy (as MHA heads do), forced compression eliminates noise and improves generalization. **Guideline:** Don't assume that larger KV caches produce better models. At 100B+ scale, the redundancy in full MHA KV representations likely hurts more than it helps.

---

## Key Takeaways

1. **MLA compresses KV cache by 93.3% without quality loss.** By storing a 512-dim latent + 64-dim RoPE component instead of 32,768 dims of full MHA, DeepSeek-V3 achieves inference memory efficiency between MQA and GQA-8 while preserving per-head representational diversity through learned up-projections.

2. **Fine-grained MoE (256 experts, top-8) provides exponentially more routing diversity than coarse MoE.** The $\binom{256}{8} \approx 4.4 \times 10^{13}$ expert combinations convert routing into a representational mechanism. The shared expert captures common patterns, freeing routed experts to specialize.

3. **Auxiliary-loss-free load balancing separates learning from systems constraints.** Bias terms adjust routing without corrupting the training objective. This separation yields better model quality *and* better training stability — zero loss spikes across 14.8T tokens.

4. **Multi-token prediction densifies the training signal.** Predicting multiple future tokens per position forces less myopic representations and provides free speculative decoding at inference (1.8x speedup).

5. **FP8 mixed-precision training is practically lossless with fine-grained quantization.** Tile-wise activations (1x128) and block-wise weights (128x128) with periodic FP32 accumulation keep relative loss error below 0.25% while doubling GEMM throughput.

6. **The $5.6M training cost is a compound effect.** MoE sparsity, FP8 throughput, DualPipe communication overlap, auxiliary-loss-free stability, and MLA activation savings all multiply together. No single innovation explains the cost — the integration does.

7. **DeepSeek-V3's architecture is a systems design, not a collection of tricks.** Each choice enables the others: MoE sparsity makes FP8's 2x throughput matter more (fewer FLOPs to begin with), auxiliary-loss-free balancing makes MoE training stable enough to run without restarts, MLA reduces activation memory enough to fit larger batches, and MTP repurposes training infrastructure for inference speedup.

8. **V4's hybrid sparse attention (CSA+HCA) solves the million-token wall.** Per-head pathway assignment splits "precise lookup" heads (CSA, O(Lk)) from "broad summary" heads (HCA, extreme compression), reducing KV cache to 10% and inference FLOPs to 27% of V3.2 at 1M tokens.

9. **Engram separates memory from computation.** Deterministic O(1) N-gram hashing offloads factual recall from attention, freeing heads for relational reasoning. The Sparsity Allocation Law (20-25% memory, 75-80% compute) formalizes the budget split.

10. **Manifold-constrained hyper-connections (mHC) stabilize ultra-deep residual streams.** Doubly stochastic mixing matrices on the Birkhoff Polytope preserve signal magnitude across depth, preventing both explosion and collapse — enabling a 4x wider residual stream at only 6.7% training overhead.

11. **FP4 expert precision compounds with FP8 training.** V4 drops MoE experts from FP8 to FP4, halving expert memory and compute cost again. Combined with the activation ratio drop from 5.5% to 3%, V4 activates fewer parameters more cheaply per token despite having 2.4x total parameters.

---

## References

- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]] — primary source for V3
- [[deepseek-v4|DeepSeek AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" (2026) (report)]] — V4 architecture, hybrid attention, Engram, mHC
- [[deepseek-v2|DeepSeek AI, "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (2024) (report)]] — MLA and DeepSeekMoE origins
- [[deepseek-r1|DeepSeek AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL" (2025) (report)]] — R1 distillation into V3 post-training
- [[ch-07|Chapter 7: Attention Variants]] — MHA, MQA, GQA, MLA foundations
- [[ch-14|Chapter 14: Mixture of Experts]] — MoE fundamentals, routing, load balancing
- [[multi-token-prediction|Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction" (2024) (paper)]] — foundational MTP work adopted by DeepSeek-V3
