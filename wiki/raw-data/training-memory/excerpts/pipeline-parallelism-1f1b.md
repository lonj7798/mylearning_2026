# Pipeline Parallelism: GPipe and the 1F1B Schedule
<!-- slug: pipeline-parallelism-1f1b · type: paper · source: https://arxiv.org/abs/1811.06965 + Narayanan 2021 SC Megatron -->

**Core Insight.** Pipeline parallelism stages layers across devices and uses microbatch interleaving to hide the startup bubble. GPipe (all-forward-then-all-backward) and 1F1B (interleave per microbatch) share the same bubble fraction `(p-1)/m`, but 1F1B caps activation memory at p microbatches in flight vs. m for GPipe — the critical memory difference at large m.

**Guideline.** Choose pp (pipeline stages) so the bubble `(p-1)/m` < 5%; that requires m ≥ 20×(p-1). Use 1F1B rather than GPipe: same bubble, fraction of the activation memory. For tighter bubble at the cost of extra all-reduce communication per stage: use the interleaved (virtual stages) 1F1B from Megatron, which cuts bubble by a factor equal to the number of virtual chunks per rank.

## Technical Details

- **Bubble fraction formula:** `(p-1) / (m + p-1)` where p = pipeline stages, m = microbatches. Simplified to `(p-1)/m` when m >> p. Both GPipe and 1F1B satisfy this same ratio; neither schedule is strictly better on idle time.
- **GPipe activation peak:** Each stage holds activations for all m microbatches simultaneously during the all-forward pass → memory scales as O(m). Gradient accumulation across microbatches is mandatory before any weight update.
- **1F1B activation peak:** Each stage holds activations for at most p microbatches in flight (the pipeline depth) → memory scales O(p), independent of m. Enabled by starting the backward pass for microbatch k as soon as its forward completes, freeing its activations before the next microbatch enters.
- **GPipe result:** 557M AmoebaNet ImageNet 84.4% top-1; 6B 128-layer multilingual Transformer trained on 8 accelerators — "25× larger network vs. single accelerator."
- **Re-materialization (GPipe):** Activations are checkpointed at partition boundaries and recomputed during backward, reducing peak memory per device to O(1/p) of the full model's activations at the cost of one extra forward pass.
- **Interleaved 1F1B (Narayanan 2021 / Megatron):** Assigns each device `v` non-contiguous model chunks (virtual stages) instead of one; reduces bubble by factor v, from `(p-1)/m` to `(p-1)/(v·m)`. Cost: v additional peer-to-peer sends/receives per microbatch step and slightly higher peak activation memory (~v× vs. standard 1F1B within a stage).
- **Training-memory angle:** PP is the *only* parallelism that physically places different layers on different GPUs — it doesn't shard a single layer's weights (that's TP). Each rank holds only its pp-partition's weights and activations, so pp-degree directly divides weight memory. The bubble inefficiency and micro-batch gradient-accumulation requirement are the memory-neutral trade that PP demands.

## Citation
Huang, Y., Cheng, Y., Chen, D., Lee, H., Ngiam, J., Le, Q. V., & Chen, Z. (2019). GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. NeurIPS 2019. https://arxiv.org/abs/1811.06965

Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., et al. (2021). Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. SC '21. https://arxiv.org/abs/2104.04473
