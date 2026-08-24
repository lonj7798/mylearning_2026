# DeepSpeed-MoE and GShard: Expert Parallelism
<!-- slug: deepspeed-moe-ep · type: paper · source: https://arxiv.org/abs/2006.16668 + https://proceedings.mlr.press/v162/rajbhandari22a -->

**Core Insight.** Expert parallelism (EP) shards the expert weight matrices across EP_size GPUs so each rank holds only E/EP_size experts; tokens are routed to the correct rank via a pair of all-to-all collectives (dispatch + combine). This keeps expert weight memory bounded per GPU while enabling sparse conditional computation.

**Guideline.** Combine EP with DP and TP: EP handles sparse expert layers (all-to-all routing), TP handles dense layers within a rank (AllReduce), DP handles data replication across EP groups. Set the capacity factor ≥1.0 (experts accept at least tokens/num_experts tokens per batch); values below 1.0 cause token dropping and training instability.

## Technical Details

- **Expert parallelism memory model:** With E total experts and EP_size ranks, each GPU holds `E/EP_size` expert weight matrices. Non-expert layers (attention, embedding) are replicated across EP ranks under DP style. Net expert weight memory per GPU = `W_expert / EP_size`.
- **All-to-all routing (two collectives per MoE layer):**
  1. **Dispatch:** Each GPU sends its assigned tokens to the EP rank that holds the target expert.
  2. **Combine:** Each GPU receives the expert outputs from all EP ranks and aggregates them (weighted by routing score).
- **Capacity factor:** `capacity = capacity_factor × (tokens_per_batch / num_experts)`. Determines max tokens an expert can process; overflow tokens are dropped (capacity_factor < 1) or padded (capacity_factor > 1). GShard uses top-2 routing with random second expert to improve load balance.
- **GShard at scale (Lepikhin 2020):** 600B parameter multilingual translation model across 2048 TPU v3 cores, top-2 gating. "Scaling multilingual neural machine translation Transformer models with Sparsely-Gated Mixture-of-Experts beyond 600 billion parameters."
- **DeepSpeed-MoE hybrid parallelism (Rajbhandari 2022):** Combines EP + DP for sparse experts, TP for dense layers, and ZeRO for optimizer states — the first framework to train MoE at trillion-parameter scale.
- **All-to-all communication cost:** Two all-to-all calls per MoE layer, each of size `tokens × d_model` bytes per rank. Unlike all-reduce (volume fixed), all-to-all volume scales with sequence-length × batch-size and can dominate at long contexts.
- **Training-memory angle:** EP cuts expert-weight memory by EP_size — critical for MoE models where expert parameters are 4–16× the dense parameter count. But the all-to-all buffers (dispatch + combine tensors of size `tokens × d_model × 2`) are transient peaks that must be budgeted separately from steady-state weight memory.

## Citation
Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z. (2021). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. ICLR 2021. https://arxiv.org/abs/2006.16668

Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y., Awan, A. A., Rasley, J., & He, Y. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale. ICML 2022. https://proceedings.mlr.press/v162/rajbhandari22a
