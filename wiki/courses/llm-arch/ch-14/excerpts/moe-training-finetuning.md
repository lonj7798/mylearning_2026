# Excerpt: MoE Training Dynamics and Fine-Tuning

**Sources:** [[hf-mixture-of-experts|blog]], [[switch-transformer|paper]], [[deepseek-v3|report]]

---

## Training: Faster Convergence, Not Faster Steps

MoE models do not run faster per training step than dense models of equivalent active parameters. In fact, they are slightly slower due to routing overhead (the router computation) and all-to-all communication (dispatching tokens to their expert devices). The speedup is in **sample efficiency**: MoE models reach the same loss in fewer steps because each step accesses a larger effective parameter space.

Switch Transformers measured this precisely: a 7x pre-training speedup over T5-XXL means reaching the same validation loss in 1/7th the training steps, not that each step is 7x faster. The wall-clock speedup depends on how efficiently the routing communication is overlapped with computation.

**Precision requirements for training stability:**

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Router logits/softmax | float32 | Softmax is numerically sensitive; bfloat16 causes routing errors |
| Expert FFN computations | bfloat16 | Safe for matrix multiplications |
| Attention layers | bfloat16/float32 | Standard mixed-precision rules apply |
| DeepSeek-V3 experts | FP8 | Fine-grained quantization with 128-element accumulation |

DeepSeek-V3 pushed this further with FP8 expert computation (tile-wise quantization for activations, block-wise for weights), doubling throughput over BF16 with less than 0.25% relative loss error. The key innovation: high-precision accumulation every 128 elements prevents numerical drift in the low-precision arithmetic.

## Router Z-Loss for Stability

The ST-MoE paper (2022) identified a subtle instability: router logits can grow unboundedly during training, causing softmax overflow. The router Z-loss penalizes large logits:

$$\mathcal{L}_{z} = \frac{1}{B} \sum_{x} \left(\log \sum_{i=1}^{N} e^{g_i(x)}\right)^2$$

This is the squared log-sum-exp of the router logits — it is minimized when logits are small and uniform in scale. Adding this term with a small coefficient ($\alpha_2 \approx 0.001$) "significantly improved training stability with no quality degradation." It has become standard practice in MoE training.

## What Experts Learn

Analysis of trained MoE models reveals:

**Encoder models (e.g., BERT-MoE):** Clear token-level specialization emerges. Individual experts consistently handle specific token types — one for punctuation, one for proper nouns, one for verbs, etc. This specialization is interpretable and aligns with linguistic categories.

**Decoder models (e.g., Mixtral, GPT-MoE):** Specialization is much less clean. Each expert handles a mix of token types, and it is difficult to assign human-readable labels. The routing appears to operate on sub-token-level activation patterns rather than semantic categories.

**Multilingual models:** The most surprising finding — no language-specific experts emerge. Despite intuitions that experts would partition by language (an "Arabic expert," a "Chinese expert"), the auxiliary load-balancing loss prevents this clustering. All experts process tokens from all languages. The balancing loss is stronger than the natural clustering tendency.

This suggests that MoE routing in decoder models is doing something more subtle than human-interpretable specialization. The "expertise" likely lies in combinations of low-level features (specific activation patterns, particular numerical ranges in the hidden state) rather than high-level semantic categories.

## Scaling Experts: Diminishing Returns

Adding experts increases model capacity at constant per-token FLOPs, but with strongly diminishing returns:

- 2 to 8 experts: **+25-30%** quality gain
- 8 to 128 experts: **+15-20%** additional quality gain
- 128 to 512 experts: **+5-10%** additional quality gain
- 512 to 2048 experts: **negligible** additional gain

This logarithmic curve means the practical ceiling is around 128-256 experts. Beyond that, each additional expert contributes so little that the communication overhead and load-balancing complexity dominate.

The current consensus (DeepSeek-V3 at 256, Qwen3 at 128, Llama 4 Maverick at 128) appears to be at or near this practical optimum.

## Fine-Tuning MoE: The Overfitting Problem

MoE models overfit faster during fine-tuning than dense models of equivalent quality. The mechanism is straightforward: sparsity means each expert sees only a fraction of the fine-tuning data.

With 8 experts and top-2 routing, each expert processes ~25% of tokens. With 128 experts and top-8, each expert processes ~6.25% of tokens. The effective dataset size per expert is much smaller than the actual fine-tuning set, accelerating memorization.

**Practical mitigations:**

| Technique | Dense Default | MoE Recommendation |
|-----------|--------------|-------------------|
| Expert dropout | 0.1 | 0.3-0.5 |
| Batch size | 32-64 | 8-16 |
| Learning rate | Standard | Higher (5e-4 to 1e-3) |
| Auxiliary loss | N/A | Keep during fine-tuning |

The higher learning rate + higher dropout combination may seem contradictory, but it works because the dropout provides strong regularization that the faster learning rate can overcome for genuinely useful updates.

## Expert Freezing: A Counterintuitive Strategy

The most surprising fine-tuning finding: **freezing all expert parameters and updating only shared layers** (attention, embeddings, LayerNorm) retains approximately 95% of full fine-tuning quality while being 30-50% faster and requiring 40% less VRAM.

Why this works:
- Shared layers (attention, embeddings) affect **every token** — updating them has broad impact across the entire input distribution
- Expert layers are sparse — each expert affects only the tokens routed to it (6-25% depending on architecture)
- Updating experts on small datasets leads to rapid overfitting because each expert's effective dataset is tiny
- The pre-trained expert weights already encode the model's core knowledge; fine-tuning mainly needs to adjust how that knowledge is composed and attended to

## Instruction Tuning: MoE Shines

A striking result from the HuggingFace survey: MoE models benefit **1.8x more from instruction tuning** than dense models. Instruction-tuning a dense T5 improves performance by ~25%, while the same treatment on an MoE version improves performance by ~45%.

The hypothesis: instruction-tuning data is inherently diverse (many different tasks, formats, domains). MoE architectures, with their diverse expert combinations, are particularly well-suited to capture this diversity. Different expert teams can specialize for different instruction types, while a dense model must represent all instructions in the same parameter set.

**Task-type performance comparison:**

| Task Type | MoE vs Dense |
|-----------|-------------|
| Knowledge-heavy (TriviaQA, MMLU) | MoE +20-30% better |
| Reasoning-heavy (SuperGLUE) | Dense 10-15% better |
| Small fine-tuning datasets | Dense clearly better |
| Large fine-tuning datasets | MoE competitive or better |

MoE excels on knowledge retrieval (more parameters = more stored knowledge) but can underperform on pure reasoning with limited data (where dense models' uniform parameter usage provides more consistent gradients).

## Inference Optimization: Distillation and Merging

Two techniques for reducing MoE inference cost:

**Distillation to dense:** Train a small dense student from a large MoE teacher. Retains 30-40% of the quality gap between the baseline dense model and the MoE teacher. The resulting model is 5-10x smaller and much faster. Switch Transformers demonstrated this works but noted the retained quality depends heavily on the task.

**Expert merging:** Average the weights of similar experts to reduce total parameter count. For example, merging Mixtral's 8 experts into 4 (averaging pairs with highest weight similarity) yields a 28B-parameter model that retains 95-97% of the original quality. This is a post-hoc optimization that requires no retraining.
