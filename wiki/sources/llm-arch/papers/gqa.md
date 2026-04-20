<!-- scope: GQA — grouped KV heads balance quality and inference speed
     deps: [[mqa]], [[attention-is-all-you-need]]
     see-also: [[flash-attention]], [[paged-attention]]
-->

# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Core Insight:** Grouped key/value heads are the sweet spot between MHA quality and MQA inference efficiency.
- **Guideline:** Default to GQA with G = H/8 groups for new models; for existing MHA checkpoints, uptrain to GQA with 5% of original compute.
- **Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.13245
- **Relevant chapters:** attention mechanisms, inference optimization, KV cache, model architecture

## Abstract
Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.

## Key Contributions
- Introduces grouped-query attention (GQA), a generalization that interpolates between multi-head attention (MHA) and multi-query attention (MQA) by using G key-value head groups
- Proposes an uptraining recipe to convert existing MHA checkpoints to GQA/MQA using only 5% of original pre-training compute, avoiding the need to retrain from scratch
- Demonstrates that GQA achieves quality close to MHA while matching MQA's inference speed
- Shows the uptraining approach works across model scales, making it practical for large existing models
- GQA has become the standard attention configuration in most modern LLMs (LLaMA 2/3, Mistral, Gemma, etc.)

## Architecture Details
- **MHA vs MQA vs GQA:** In MHA with H query heads, there are also H key and H value heads. MQA collapses to 1 key head and 1 value head shared across all queries. GQA uses G groups (1 < G < H), where each group of H/G query heads shares one key and one value head
- **KV cache reduction:** GQA with G groups reduces the KV cache size by a factor of H/G compared to MHA. For example, with H=32 query heads and G=8 groups, KV cache is 4x smaller
- **Uptraining procedure:** Start from a pretrained MHA checkpoint. For each GQA group, initialize the shared key/value head by mean-pooling the key/value heads from the original MHA heads in that group. Then continue pretraining for a small fraction of original compute
- **Mean-pooling initialization:** Averaging the original heads (rather than selecting one or random init) provides a better starting point because it preserves the most information from the original model
- **Inference speedup:** The primary benefit is during autoregressive decoding, where smaller KV cache means less memory bandwidth consumption per token generated. The speedup scales with the ratio H/G
- **Memory-bandwidth bound:** During decoding, the bottleneck is loading KV cache from HBM. GQA reduces this proportionally to the number of KV heads, directly translating to faster generation
- **Common configurations:** LLaMA 2 70B uses GQA with 8 KV heads and 64 query heads (8x reduction). This has become the de facto standard

## Tradeoffs Discussed
- MQA (G=1) offers maximum inference speedup but can degrade quality, especially on tasks requiring fine-grained multi-head attention patterns
- GQA trades some inference speed (relative to MQA) for better quality by retaining multiple distinct KV representations
- The uptraining approach requires additional compute (5% of pretraining) — nontrivial for very large models but much cheaper than retraining
- The optimal number of GQA groups G is a hyperparameter that must be tuned; too few groups hurts quality, too many reduces the inference benefit
- GQA does not help during training (only inference), since training is parallelized across sequence length and not memory-bandwidth-bound in the same way
