<!-- scope: DeepSeek-V4 technical report (preview)
     deps: [[ch-01]], [[ch-02]]
     see-also: [[deepseek-v3]], [[deepseek-v2]], [[deepseek-r1]]
-->

# DeepSeek-V4 Technical Report (Preview)
- **Core Insight:** Hybrid sparse attention (CSA + HCA) combined with Engram conditional memory and manifold-constrained residual connections enables million-token context at a fraction of prior KV-cache and FLOP cost.
- **Guideline:** When scaling context length, replace uniform full attention with a hybrid of compressed sparse and heavily compressed heads — pair with a deterministic memory module (Engram) to offload factual recall from attention entirely.

- **Organization:** DeepSeek AI
- **Year:** 2026
- **Title:** DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- **URL (HF model card):** https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- **Foundational papers:** Engram (arXiv:2601.07372), mHC (arXiv:2512.24880), DeepSeek Sparse Attention (DSA)
- **Relevant chapters:** MoE scaling, sparse attention, long-context efficiency, residual connection design, conditional memory, FP4/FP8 mixed precision

## Abstract
DeepSeek-V4 series presents two strong Mixture-of-Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T total parameters (49B activated) and DeepSeek-V4-Flash with 284B total parameters (13B activated) — both supporting a context length of one million tokens. The series incorporates a hybrid attention mechanism combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to dramatically improve long-context efficiency. In the 1M-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2. Manifold-Constrained Hyper-Connections (mHC) strengthen conventional residual connections, enhancing stability of signal propagation across layers while preserving model expressivity. The Engram module modernizes classic hashed N-gram embedding to provide deterministic knowledge lookup with approximate O(1) time complexity. Both models are pre-trained on more than 32T diverse and high-quality tokens using the Muon optimizer, followed by a comprehensive post-training pipeline.

## Architecture Summary

### DeepSeek-V4-Pro

| Component | Value |
|-----------|-------|
| Total Parameters | 1.6T |
| Active Parameters per Token | 49B (~3% activation) |
| Routed Experts | 384 per layer |
| Active Routed Experts (top-K) | 6 per token |
| Head Dimension | 512 |
| Attention Mechanism | Hybrid CSA + HCA (with Sparse MQA and SWA) |
| Context Length | 1M tokens (pre-trained at 32K, extended to 1M) |
| Residual Connections | Manifold-Constrained Hyper-Connections (mHC) |
| Memory Module | Engram (conditional memory via hashed N-gram lookup) |
| MoE Expert Precision | FP4 |
| Other Parameters Precision | FP8 |

### DeepSeek-V4-Flash

| Component | Value |
|-----------|-------|
| Total Parameters | 284B |
| Active Parameters per Token | 13B |
| Context Length | 1M tokens |
| Attention Mechanism | Hybrid CSA + HCA (shared design with Pro) |

## Key Architectural Innovations

### 1. Hybrid Attention: CSA + HCA (DSA2)

DeepSeek-V4 replaces the uniform MLA attention of V3 with a hybrid of two complementary sparse attention mechanisms, collectively called DSA2 (integrating DeepSeek Sparse Attention and Native Sparse Attention):

- **Compressed Sparse Attention (CSA):** Uses a lightning indexer (FP8 batched matrix multiplies) and fine-grained token selection via GPU partial-sort algorithms to reduce attention complexity from quadratic O(L^2) to linear O(Lk), where k is the number of selected tokens per query.
- **Heavily Compressed Attention (HCA):** Applies extreme KV-cache compression for heads that need broad but coarse context awareness, slashing KV-cache to 10% of V3.2 levels at 1M context.

The hybrid assigns each attention head to either the CSA or HCA pathway, achieving fine-grained per-head sparsity. Custom CUDA kernels manage variable per-query sparsity for the gather operations and main attention.

**Impact:** At 1M tokens, V4-Pro needs only 27% of the inference FLOPs and 10% of the KV cache versus V3.2 for a single generated token.

### 2. Manifold-Constrained Hyper-Connections (mHC)

mHC replaces standard residual connections with a richer mixing framework while preserving training stability:

- Projects the residual connection space onto the **Birkhoff Polytope** — the set of doubly stochastic matrices — using the Sinkhorn-Knopp algorithm.
- Doubly stochastic mixing matrices preserve signal magnitude across depth, preventing both explosion and collapse of residual streams no matter how deep the model grows.
- A 4x wider residual stream adds only 6.7% training time overhead.
- Restores the identity-mapping stability property of vanilla residuals while enabling richer cross-layer signal mixing.

**Paper:** arXiv:2512.24880 (co-authored by DeepSeek founder Liang Wenfeng, uploaded January 1, 2026).

### 3. Engram Conditional Memory

Engram introduces a deterministic memory axis complementary to the MoE computation axis:

- **Mechanism:** Multi-head hashing maps suffix N-grams of the current position into prime-sized embedding buckets. A small depthwise convolution over the N-gram context and a context-aware gating scalar (range [0, 1]) control how much of the retrieved embedding is injected into each hidden-state branch.
- **Sparsity Allocation Law:** Under a fixed sparse parameter budget, optimal allocation is approximately 20-25% memory (Engram) and 75-80% computation (MoE).
- **Instantiation in V4:** Engram modules placed at layers 2 and 15 with max N-gram size 3, 8 heads, dimension 1280.
- **Effect:** Provides O(1) factual/knowledge lookup without consuming attention bandwidth — "don't calculate when you can look up."

**Paper:** arXiv:2601.07372 (DeepSeek + Peking University, January 12, 2026).

### 4. Expanded MoE with FP4 Expert Precision

- Scales from 256 routed experts (V3) to **384 routed experts** per layer, but reduces active experts from 8 (V3) to **6 per token**, achieving finer-grained specialization with a lower activation ratio (~3% vs ~5.5%).
- MoE expert parameters stored and computed in **FP4 precision** (down from FP8 in V3), with most other parameters in FP8. This further halves expert memory and compute cost relative to V3's FP8 regime.
- Retains auxiliary-loss-free load balancing (bias-based routing) introduced in V3, with no token dropping.

### 5. Muon Optimizer

Replaces AdamW with the **Muon optimizer** for pre-training, delivering faster convergence and greater training stability compared to V3's optimizer setup.

## Training Details

- **Pre-training data:** >32T diverse, high-quality tokens (more than double V3's 14.8T)
- **Pre-training context:** 32K tokens, extended to 1M in post-training stages
- **Optimizer:** Muon (replaces AdamW used in V3)
- **Hardware:** Reported to run on Huawei Ascend 910B/950PR chips (in addition to or instead of NVIDIA H800s used for V3)
- **Estimated training cost:** ~$5.2M (unverified; V3 was ~$5.6M)

**Post-training pipeline (two-stage):**
1. **Domain-specific expert cultivation:** Independent SFT and RL (GRPO) per domain, building specialized expert capabilities in reasoning, coding, math, etc.
2. **Unified model consolidation:** On-policy distillation merges domain-specific expert proficiencies into a single model, integrating distinct capabilities across domains.

This two-stage paradigm is a departure from V3's simpler SFT-then-RL pipeline, enabling deeper specialization before unification.

## What Changed from DeepSeek-V3 to V4

| Dimension | DeepSeek-V3 | DeepSeek-V4-Pro |
|-----------|-------------|-----------------|
| Total Parameters | 671B | 1.6T (~2.4x) |
| Active Parameters | 37B | 49B (~1.3x) |
| Routed Experts | 256 | 384 |
| Active Experts (top-K) | 8 | 6 |
| Attention | Multi-head Latent Attention (MLA) | Hybrid CSA + HCA (DSA2) |
| Context Length | 128K | 1M (~8x) |
| KV Cache (at max context) | Full MLA cache | 10% of V3.2 equivalent |
| Residual Connections | Standard | mHC (Birkhoff Polytope constrained) |
| Memory Module | None | Engram (conditional N-gram memory) |
| Expert Precision | FP8 | FP4 |
| Other Precision | BF16/FP8 | FP8 |
| Pre-training Tokens | 14.8T | >32T (~2.2x) |
| Optimizer | AdamW | Muon |
| Post-training | SFT + RL (GRPO) | Two-stage: domain expert SFT/RL then on-policy distillation |
| Training Objective | MTP (multi-token prediction) | MTP (retained; details not yet published) |
| Model Variants | Single model | V4-Pro (1.6T) + V4-Flash (284B/13B active) |

**Key themes of the V3-to-V4 transition:**
1. **Context length as a first-class target** — 8x context expansion (128K to 1M) driven by entirely new attention architecture rather than just positional encoding changes.
2. **Efficiency at every level** — FP4 experts, sparse attention, Engram O(1) lookup all compound to reduce per-token cost despite larger total parameters.
3. **Separation of memory from computation** — Engram decouples factual recall (static memory) from reasoning (MoE computation), following the Sparsity Allocation Law.
4. **Training stability through architecture** — mHC addresses residual stream degradation at depth, replacing ad-hoc fixes with a principled manifold constraint.
5. **Hardware diversification** — first major DeepSeek model reportedly trained on Huawei Ascend chips, reducing dependence on NVIDIA hardware.

## Release Status (as of April 2026)

DeepSeek-V4 preview weights are available on Hugging Face (deepseek-ai/DeepSeek-V4-Pro and deepseek-ai/DeepSeek-V4-Flash). A full technical report with complete architectural specifications (exact layer count, hidden dimensions, detailed ablations) has not yet been published. The information above is synthesized from the Hugging Face model cards, the three foundational research papers (Engram, mHC, DSA), code leaks, and credible third-party analyses. Some architectural details (e.g., exact layer count, vocabulary size) remain unconfirmed pending the official technical report.
