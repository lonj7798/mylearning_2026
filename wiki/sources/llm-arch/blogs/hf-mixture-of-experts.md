<!-- scope: MoE architecture, routing, and training
     deps: [[ch-04]]
     see-also: [[raschka-llm-architecture-comparison]], [[openmythos]]
-->

# Mixture of Experts Explained

- **Core Insight:** MoE routing is the critical design choice — load balancing determines whether expert specialization occurs.
- **Guideline:** Focus on routing stability when designing or evaluating MoE architectures.

- **Author:** Omar Sanseviero, Lewis Tunstall, Philipp Schmid, Sourab Mangrulkar, Younes Belkada, Pedro Cuenca (Hugging Face)
- **URL:** https://huggingface.co/blog/moe
- **Relevant chapters:** MoE architecture, scaling, routing, training, inference optimization

## Summary
A comprehensive guide to Mixture of Experts in LLMs, covering the two main components (sparse MoE layers and gate/router networks), historical development from 1991 to Mixtral 8x7B, gating mechanisms with formulas, load balancing, training stabilization via router Z-loss, what experts learn, scaling behavior, fine-tuning challenges, and inference optimization including distillation, expert merging, and parallelism strategies.

## Key Content

### What Is a Mixture of Experts?

MoE replaces dense feed-forward (FFN) layers in transformers with sparse MoE layers containing multiple "experts" (each an FFN) and a gate network (router) that routes tokens to appropriate experts.

**Key numbers for Mixtral 8x7B:**
- 8 experts x 7B parameters = 56B total, but shared layers (attention) mean actual memory = ~47B
- Only 2 experts active per token -> active parameters ~12B
- Inference FLOPs: similar to a 12B model, not 56B

**Pretraining:** 4x faster than dense models (Switch Transformers vs T5-XXL)

### Two Main Components

**1. Sparse MoE Layers:** Replace traditional dense FFN layers. Contain multiple expert networks.

**2. Gate Network (Router):** Learned network that routes tokens to experts. Routes per-token (can send one token to multiple experts). Composed of trainable parameters optimized during pretraining.

### Historical Development

| Year | Contribution |
|------|-------------|
| 1991 | Adaptive Mixture of Local Experts (foundation) |
| 2014 | MoEs as components in deeper networks |
| 2017 | Shazeer et al. scale to 137B LSTM; introduce sparsity |
| 2020 | GShard scales transformers to 600B+ |
| 2021 | Switch Transformers: 1.6T parameters, single-expert routing |
| 2022 | ST-MoE: Router Z-loss for stability |
| 2023 | Mixtral 8x7B: high-quality open model |

### Gating Mechanisms

**Basic softmax routing:**
G_sigma(x) = Softmax(x * W_g)

Problem: No sparsity — all experts computed.

**Noisy Top-K Gating (Shazeer et al.):**

Step 1 — Add noise for exploration:
H(x)_i = (x * W_g)_i + StandardNormal() * Softplus((x * W_noise)_i)

Step 2 — Keep top-k experts:
KeepTopK(v, k)_i = v_i if in top k, else negative infinity

Step 3 — Apply softmax:
G(x) = Softmax(KeepTopK(H(x), k))

Noise enables exploration and load balancing. Top-k creates sparsity (only k experts computed).

### Load Balancing

**The problem:** Popular experts train faster -> selected more -> reinforced -> most experts unused.

**Solution:** Auxiliary loss encouraging uniform expert usage:

Expert Capacity = (tokens_per_batch / num_experts) * capacity_factor

Example: (1024 / 8) * 1.25 = 160 tokens per expert

If tokens exceed capacity: skip expert, send via residual connection. Token drops can act as regularization.

### Router Z-Loss (ST-MoE, 2022)

Penalizes large logits before softmax to prevent numerical instability:

z_loss = (log(sum(exp(logits))))^2
total_loss = task_loss + alpha_1 * aux_loss + alpha_2 * z_loss

Effect: Keeps logits in reasonable range, reduces roundoff errors in softmax. Significantly improved training stability with no quality degradation.

### What Do Experts Learn?

**Encoder:** Token-level specialization (punctuation expert, proper noun expert, common word expert, verb expert, number expert).

**Decoder:** Less specialization; mixed token types across experts.

**Multilingual models — surprising finding:** No language specialization. Expected one expert per language; actual result is mixed tokens from all languages. Auxiliary loss prevents language segregation.

### Scaling the Number of Experts

Diminishing returns:
- 2 experts: baseline
- 8 experts: +25-30%
- 128 experts: +45-50%
- 512+ experts: plateau begins (~+55%)
- 2048 experts: negligible gains over 512

### Fine-Tuning MoEs

**Core challenge:** Overfitting. Sparse models memorize faster than dense models.

**Solutions:**
- Higher dropout in experts (0.3-0.5 vs 0.1 global)
- Smaller batch sizes (8-16 vs 32-64)
- Higher learning rates (5e-4 to 1e-3)
- Keep auxiliary loss for instruction-tuned models

**Counter-intuitive finding:** Freezing expert parameters and updating only shared layers (attention, embeddings) retains ~95% of quality with 30-50% faster fine-tuning and 40% less VRAM. This works because MoE layers are sparse/isolated while shared layers affect all tokens.

**Breakthrough — Instruction tuning (2023):**
- Dense T5 vs T5: +25% improvement
- MoE vs MoE with instruction tuning: +45% improvement (1.8x more benefit!)
- MoEs benefit MORE from instruction tuning than dense models

### Fine-Tuning by Task Type

| Task Type | Sparse vs Dense |
|-----------|----------------|
| Knowledge-heavy (TriviaQA, MMLU) | Sparse +20-30% better |
| Reasoning-heavy (SuperGLUE) | Dense -10-15% better |
| Small datasets | Dense clearly better |
| Large datasets | Sparse competitive or better |

### Switch Transformers: Single-Expert Routing

Key innovation: Route each token to exactly ONE expert (not top-2).

**Benefits:**
1. Simpler gating function
2. Doubled expert batch size (better GPU utilization)
3. Reduced communication (one device instead of two)
4. Preserved quality despite simplification

**Precision management:** Router in float32 (numerical stability), expert FFNs in bfloat16 (speed). No quality loss + 20% speedup.

### Inference Optimization

**Distillation to dense models:**
- Train small dense student from large MoE teacher
- Retains 30-40% of sparsity gains
- 5-10x smaller, much faster inference

**Expert merging:** Average weights of similar experts to reduce parameter count. 56B -> 28B with 95-97% quality retained.

**Expert parallelism:** Place different experts on different devices. Non-MoE layers use data parallelism; tokens route to expert-containing workers.

### MoE Output Formula

y = sum_{i=1}^{n} G(x)_i * E_i(x)

Where G(x) = gate network output (routing probabilities), E_i(x) = i-th expert output.

## Notable Insights
- The parameter count comparison is misleading: Mixtral 8x7B's active parameters (~12B) make it comparable to a 12B dense model in compute, not a 47B model. Always compare active parameters for fair benchmarking.
- The lack of language specialization in multilingual MoEs is surprising and suggests that the auxiliary loss for load balancing is powerful enough to override what might seem like natural clustering.
- Expert freezing during fine-tuning works because experts are sparse and isolated — each token only sees 2 of 8 experts per layer, so updating the shared layers (which ALL tokens see) has broader impact.
- The 1.8x multiplier for instruction tuning benefit in MoEs vs dense models suggests MoE architectures are particularly well-suited for diverse multi-task training.
- MegaBlocks' block-sparse matmul is a practical breakthrough: traditional batched matmul wastes compute by padding to uniform expert sizes; block-sparse handles variable expert loads without waste.
