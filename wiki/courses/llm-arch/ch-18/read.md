# Chapter 18: Case Study — LLaMA 3 and Llama 4

<!-- scope: LLaMA family evolution from dense conservative baseline (LLaMA 3) to sparse long-context MoE (Llama 4)
     deps: [[ch-09]], [[ch-14]], [[ch-16]]
     see-also: [[ch-19]], [[ch-20]]
-->

## Overview

The Llama family is the best lens through which to study how LLM architecture philosophy evolves within a single organization. In two generations — LLaMA 3 (2024) to Llama 4 (2025) — Meta moved from a dense, conservative baseline built on proven components to a sparse, ambitious mixture-of-experts architecture targeting 10 million token contexts. This is not incremental improvement. It is a fundamental rethinking of what a foundation model should optimize for.

LLaMA 3 is the culmination of the "simple and scale" philosophy. Every component — GQA, RoPE, RMSNorm, SwiGLU — was already proven elsewhere. Meta's contribution was combining them at 405B scale with 15 trillion training tokens, demonstrating that meticulous execution of known techniques beats architectural novelty. The result became the industry's reference architecture: when a paper says "standard Transformer," they mean something very close to LLaMA 3.

Llama 4 broke from this philosophy entirely. MoE replaces dense layers. iRoPE replaces standard RoPE. Early fusion replaces bolted-on multimodality. The 10M token context window requires architectural innovations that have no precedent in the Llama lineage. Where LLaMA 3 was a statement about the sufficiency of existing techniques, Llama 4 is a bet that the next frontier requires structural change.

This chapter dissects both architectures in detail, traces the design decisions that connect them to the components studied in earlier chapters ([[ch-07]], [[ch-08]], [[ch-09]], [[ch-14]], [[ch-16]]), and analyzes the tradeoffs that make each generation's philosophy coherent on its own terms.

---

## 1. The Llama Lineage: Four Generations in Two Years

Before diving into architecture, the trajectory matters. Each generation had a thesis:

| Generation | Year | Thesis | Key Bet |
|-----------|------|--------|---------|
| LLaMA 1 ([[llama-1\|report]]) | 2023 | Public data + proven components can match proprietary models | Data quality > architecture novelty |
| Llama 2 ([[llama-2\|report]]) | 2023 | Scale context (2K->4K), add GQA for inference, add RLHF | Inference cost is a first-class design constraint |
| LLaMA 3 ([[llama-3\|report]]) | 2024 | Dense architecture + massive data (15T tokens) scales to GPT-4 level | Chinchilla was right: more data, not more tricks |
| Llama 4 ([[llama-4\|report]]) | 2025 | MoE + iRoPE + early fusion opens new capability frontiers | Decoupling knowledge capacity from per-token compute |

The naming shift itself is telling. "LLaMA" (uppercase) for the original research artifacts became "Llama" (sentence case) as the family became a product line. We follow Meta's convention: LLaMA for generations 1 and 3, Llama for 2 and 4.

[See the evolution timeline: [figures/llama-evolution.html](figures/llama-evolution.html)]

---

## 2. LLaMA 3: Anatomy of the Modern Baseline

### 2.1 Architecture Table

| Component | 8B | 70B | 405B |
|-----------|-----|------|--------|
| Layers | 32 | 80 | 126 |
| Model Dimension | 4,096 | 8,192 | 16,384 |
| FFN Dimension | 14,336 | 28,672 | 53,248 |
| Attention Heads | 32 | 64 | 128 |
| KV Heads (GQA) | 8 | 8 | 8 |
| Context Length | 128K | 128K | 128K |
| Vocab Size | 128,256 | 128,256 | 128,256 |
| Training Tokens | ~15T | ~15T | ~15T |

The FFN-to-model dimension ratio is 3.5x across all sizes (14336/4096 = 3.5). This is the SwiGLU-adjusted ratio: the standard 4x expansion becomes approximately (8/3) * 4 / 3 ≈ 3.5x because SwiGLU introduces a third projection matrix (gate, up, down) while the intermediate dimension is reduced to keep total parameter count comparable. See [[ch-08]] for the derivation.

### 2.2 Component-by-Component Analysis

**Grouped-Query Attention (GQA) with 8 KV heads universally.** LLaMA 3 uses 8 KV heads at every model size — including 8B. This is a departure from Llama 2, which only applied GQA to the 34B and 70B variants (keeping full MHA for 7B and 13B). The decision reflects a philosophical shift: standardize the architecture across all sizes rather than optimizing per-size. The cost of GQA at 8B is minimal (slight quality reduction from sharing KV heads across 4 query heads per group), but the benefit is enormous — identical serving infrastructure at every scale. As covered in [[ch-07]], the KV cache reduction is proportional to the ratio of query heads to KV heads: 32/8 = 4x for 8B, 64/8 = 8x for 70B, 128/8 = 16x for 405B.

**RoPE with theta = 500,000.** The base frequency theta controls how quickly the rotation "wraps around" at different head dimensions. LLaMA 1 used theta = 10,000 with a 2K context. Scaling to 128K requires either interpolation (which degrades quality) or increasing theta so that the highest-frequency components still have periods longer than the context length. LLaMA 3 chose theta = 500,000 combined with a progressive context extension schedule: train at 8K first, then gradually extend to 128K in later stages. This avoids the cost of training on 128K sequences from the start — the vast majority of training runs at 8K, and the model learns to generalize to longer contexts during the extension phase. See [[ch-06]] and [[ch-16]] for the theory behind RoPE frequency scaling.

**RMSNorm (pre-normalization).** Applied before each attention and FFN sub-layer, consistent across all sizes. No QK-norm (unlike OLMo 2), no post-norm experimentation. This is the configuration that LLaMA 1 established and every subsequent generation has retained. As [[ch-09]] covers, pre-norm RMSNorm trades a theoretical representational cost for training stability — and at 405B parameters with 15T tokens, training stability is non-negotiable.

**SwiGLU activation.** The gated FFN uses three projection matrices:

$$\text{SwiGLU}(x) = (\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$$

where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}$ and $W_{\text{down}} \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}$. The third matrix (gate) adds parameters, but the reduced $d_\text{ff}$ compensates. As [[ch-08]] details, SwiGLU consistently outperforms GELU across model sizes, and LLaMA 3 follows this finding without modification.

**128K vocabulary.** A 4x expansion from Llama 2's 32K vocab. Trained on multilingual data with BPE via SentencePiece. The larger vocabulary improves encoding efficiency (fewer tokens per input, especially for non-English languages) at the cost of a larger embedding table. For the 405B model, the embedding table is 128,256 * 16,384 = ~2.1B parameters — a small fraction of total model parameters, but a non-trivial memory cost during inference since the full embedding table must be resident.

### 2.3 The "Dense and Scale" Philosophy

LLaMA 3 405B is a deliberately dense model. Meta explicitly chose not to use MoE despite its inference efficiency advantages, arguing that dense models are "simpler to train, scale, and serve." This deserves scrutiny because it was already a controversial claim in 2024, and Llama 4's reversal makes it more so.

The case for dense at 405B:

1. **Training stability.** MoE introduces routing dynamics that can cause load imbalance, expert collapse, and training instability. At 405B parameters on 15T tokens, a single training instability event costs millions of dollars in wasted compute. Dense training is predictable.

2. **Scaling law predictability.** Meta invested heavily in scaling law experiments — training smaller models to predict optimal hyperparameters for the 405B run before committing compute. These scaling laws are better understood for dense models than for MoE, where the relationship between total parameters, active parameters, and loss is more complex.

3. **Serving simplicity.** A dense 405B model requires tensor parallelism across multiple GPUs but no expert routing, no all-to-all communication, no load balancing at inference time. Every request uses exactly the same compute path.

The case against (which Llama 4 implicitly makes):

1. **Inference cost.** Every token activates all 405B parameters. DeepSeek-V3 ([[deepseek-v2\|report]]) achieves comparable quality with 671B total but only 37B active parameters — roughly 10x cheaper per token.

2. **Diminishing returns.** Scaling laws suggest that increasing model size has diminishing returns in loss reduction. MoE lets you increase knowledge capacity (total parameters) without proportionally increasing per-token compute.

3. **Context length ceiling.** Dense models have a fixed compute-per-token cost that makes very long context (>128K) economically prohibitive at inference time. MoE's lower active parameter count makes 10M token contexts at least theoretically servable.

### 2.4 Training Scale

LLaMA 3 405B was trained on 16,384 H100 GPUs for approximately 54 days on ~15 trillion tokens. For context:

- **Data scale:** 15T tokens is roughly 10x Chinchilla-optimal for a 405B model (Chinchilla suggests ~8T tokens for 400B parameters). LLaMA 3 significantly over-trains relative to Chinchilla scaling laws, following the LLaMA 1 insight that smaller models benefit disproportionately from more data — and extending this principle to the largest model.

- **Post-training:** Replaced Llama 2's PPO-based RLHF with Direct Preference Optimization (DPO). Also added rejection sampling and tool-use training. The shift from PPO to DPO reflects the broader industry trend: DPO is simpler, more stable, and requires no reward model.

- **Context extension:** Trained at 8K context initially, then extended to 128K through a 6-stage progressive schedule with increasing RoPE theta. Each stage doubles the context and adjusts the frequency base.

---

## 3. Llama 4: The Sparse and Long Revolution

### 3.1 Architecture Table

| Component | Scout | Maverick | Behemoth |
|-----------|-------|----------|----------|
| Total Parameters | 109B | 400B | ~2T |
| Active Parameters | 17B | 17B | 288B |
| Experts | 16 | 128 | 16 |
| Active Experts | 1 shared + 1 routed | 1 shared + 1 routed | Not specified |
| Context Length | 10M tokens | Not specified | Not specified |
| Positional Encoding | iRoPE | iRoPE | iRoPE |
| Multimodality | Early fusion | Early fusion | Early fusion |

The first number to absorb: Scout has 109B total parameters but only 17B active. That is a 6.4x ratio. Maverick has 400B total but the same 17B active — a 23.5x ratio. This means Maverick has 23.5x more "knowledge capacity" than its per-token compute cost suggests. The entire MoE thesis, covered in [[ch-14]], is that you can decouple these two quantities.

### 3.2 MoE Architecture: Scout vs Maverick

**Scout (16 experts).** Each transformer layer contains 16 expert FFN modules. For each token, the router selects 1 shared expert (always active) + 1 routed expert. The shared expert provides a base capability floor — every token gets at least one full FFN computation — while the routed expert provides specialization. With 16 experts and top-1 routing, each token sees 2/16 = 12.5% of the total expert capacity.

**Maverick (128 experts).** Same routing strategy (1 shared + 1 routed), but the expert pool is 8x larger. Each token still sees only 2 expert computations, but the routing network now selects from 128 options instead of 16. This means:

- **More specialization.** With 128 experts, the router can learn finer-grained token-to-expert mappings. A token about organic chemistry can route to a different expert than a token about constitutional law, even if both would have hit the same expert in a 16-expert model.

- **More total parameters at same active cost.** Maverick has ~3.7x more total parameters than Scout (400B vs 109B) while maintaining identical active parameter count (17B). The additional parameters are entirely in the expert FFN weights.

- **Harder load balancing.** With 128 experts, the probability of routing imbalance increases. If some experts are systematically under-utilized, their parameters are wasted. DeepSeek-V3 addressed this with auxiliary-loss-free balancing (see [[ch-14]]); Meta's approach for Llama 4 is not fully detailed but likely involves similar techniques.

[See the component comparison: [figures/component-comparison.html](figures/component-comparison.html)]

### 3.3 iRoPE: Interleaved Rotary Position Embeddings

Standard RoPE, as used in LLaMA 3, applies rotary position embeddings to every attention layer. This grounds every layer's attention computation in absolute position, which is essential for distinguishing token order — but it also limits length generalization, because RoPE's rotation frequencies were calibrated for the training context length.

iRoPE takes a different approach: **interleave attention layers with and without positional embeddings**. Some layers apply standard RoPE (position-aware), while others compute attention with no positional encoding at all (position-free). The position-free layers attend purely based on content similarity, regardless of where tokens appear in the sequence.

The critical insight is that position-free attention layers have **no inherent context length limit**. They compute the same attention pattern whether the context is 256K tokens or 10M tokens, because they never encode position. The position-aware layers provide positional grounding, while the position-free layers provide unbounded-length content matching.

At inference time, iRoPE uses **temperature scaling** to manage the transition from training context (256K) to inference context (10M). The attention logits in position-aware layers are scaled to prevent the distribution from becoming too sharp or too flat when extrapolating beyond the training length.

This is a fundamentally different strategy from LLaMA 3's progressive context extension. LLaMA 3 trained at 8K and gradually increased to 128K — an expensive multi-stage process that still caps context at the maximum training length. Llama 4 trains at 256K and uses architectural properties (iRoPE + temperature scaling) to generalize to 10M at inference. The generalization is architectural, not data-driven.

For full treatment of iRoPE's theory and connection to NoPE (no positional encoding), see [[ch-06]] and [[ch-16]].

### 3.4 Early Fusion Multimodality

LLaMA 3 added vision capability post-hoc: a separate image encoder (adapter-based) was bolted onto a frozen or lightly fine-tuned language backbone. This is **late fusion** — the language model's core weights never learn to process visual information natively.

Llama 4 uses **early fusion**: text and vision tokens are processed jointly in a unified transformer backbone from the beginning of pre-training. The vision encoder (based on MetaCLIP) produces visual tokens that enter the same residual stream as text tokens, and both modalities participate in every attention layer.

The architectural implications:

1. **Cross-modal attention is native.** In early fusion, every attention layer can attend across text and vision tokens. In late fusion, cross-modal interaction is limited to the adapter layers. This enables richer multimodal reasoning.

2. **Pre-training data must include multimodal examples.** You cannot early-fuse a model that was pre-trained on text only. Llama 4's pre-training corpus includes diverse text, image, and video data — a fundamentally different data pipeline from LLaMA 3's text-first approach.

3. **The vision encoder is frozen during backbone training.** MetaCLIP is trained separately, then its outputs are adapted to the Llama backbone. The backbone learns to process visual tokens, but the visual feature extraction itself is not end-to-end trained with the language model. This is a pragmatic compromise: end-to-end multimodal training is more expensive and less stable.

### 3.5 Training and Post-Training Innovation

**Scale:** Llama 4 Scout was trained on ~40 trillion tokens, Maverick on ~22 trillion tokens. Both exceed LLaMA 3's 15T. The training corpus covers 200 languages (vs. LLaMA 3's 8+), with 100+ languages having 1B+ tokens each — a 10x increase in multilingual coverage.

**FP8 pre-training.** Llama 4 trains in FP8 precision without quality sacrifice. This is an infrastructure-level innovation that halves memory consumption and increases compute throughput compared to FP16/BF16 training. Behemoth achieves 390 TFLOPs/GPU on 32K GPUs — near the theoretical maximum for H100s at FP8.

**MetaP hyperparameter tuning.** A technique for setting per-layer learning rates and initialization scales that transfer across batch sizes, model width, depth, and training duration. This reduces the cost of hyperparameter search for large models — a critical concern when a single training run costs millions of dollars.

**Lightweight post-training.** Llama 4's post-training pipeline is deliberately lighter than LLaMA 3's:

1. Lightweight SFT — removed 50%+ of "easy" SFT data, keeping only hard examples
2. Online RL — continuous filtering of medium-to-hard prompts
3. Lightweight DPO — final alignment stage

The insight: aggressive SFT over-constrains the model, limiting the benefit of subsequent RL. By reducing SFT data and increasing RL, Meta gives the model more room to discover effective behaviors through exploration rather than imitation. For Behemoth, this is taken further: 95% of SFT data removed, followed by large-scale RL with curriculum learning. The principle is that larger models need even less supervised data and more reinforcement learning.

---

## 4. Design Philosophy Evolution: Simple-and-Scale vs Sparse-and-Long

### 4.1 What Changed and Why

The shift from LLaMA 3 to Llama 4 is not just component substitution. It reflects a change in what Meta believes the binding constraint on LLM capability is:

**LLaMA 3's thesis: the binding constraint is data and compute, not architecture.** Given enough tokens and enough GPUs, a dense transformer with known-good components (GQA, RoPE, RMSNorm, SwiGLU) will reach any capability frontier. Architecture innovation is a distraction from the real work: data curation, training infrastructure, and scaling law optimization.

**Llama 4's thesis: the binding constraint is now the dense architecture itself.** Dense models hit two ceilings simultaneously:

1. **Inference cost ceiling.** A dense 405B model costs the same compute per token whether it's answering "what's 2+2" or synthesizing a novel protein structure. There's no way to spend less compute on easy tokens and more on hard ones.

2. **Context length ceiling.** Dense attention at 405B scale with 128K context is already at the edge of economic viability. Extending to 1M+ tokens requires reducing per-token compute (MoE) and decoupling position from attention (iRoPE).

### 4.2 The Conservatism-Ambition Tradeoff

LLaMA 3's conservatism bought three things:

1. **Ecosystem adoption.** Because every component was well-understood, inference tooling (vLLM, TensorRT-LLM, llama.cpp) supported LLaMA 3 immediately. As Raschka ([[raschka-llm-architecture-comparison\|blog]]) observes, GQA's persistence as the dominant attention variant is partly because inference stacks are deeply optimized for it. LLaMA 3's architecture is the easiest frontier model to serve.

2. **Reproducibility.** No novel components means any team can rebuild the architecture from public papers. This drove the enormous open-source ecosystem: thousands of fine-tuned variants, quantized versions, and derivative models.

3. **Training predictability.** Scaling law experiments on smaller models reliably predicted 405B performance. Dense model scaling laws are better characterized than MoE scaling laws, reducing the risk of a failed training run.

Llama 4's ambition costs all three — but buys capabilities that conservative design cannot reach:

1. **Serving complexity.** MoE requires expert parallelism, load balancing, and routing-aware scheduling. Scout fits on a single H100 with int4 quantization (a serving win), but Maverick's 128 experts require sophisticated serving infrastructure. The tooling is catching up but is not as mature as dense-model serving.

2. **Training risk.** MoE training introduces new failure modes: routing collapse (all tokens go to the same experts), load imbalance (some experts are over/under-utilized), and expert-parallelism communication overhead. MetaP and FP8 training mitigate some of these, but the risk profile is fundamentally different from dense training.

3. **10M token context.** No other production model offers 10M token context. If iRoPE's length generalization works as claimed, this opens application categories (full-codebase analysis, book-length document understanding, multi-day conversation) that 128K models cannot serve. The architectural bet is that context length is a capability multiplier, not just a convenience.

---

## 5. Component-by-Component: LLaMA 3 vs Llama 4

[See the full comparison table: [figures/component-comparison.html](figures/component-comparison.html)]

### Attention

| Aspect | LLaMA 3 | Llama 4 |
|--------|---------|---------|
| Mechanism | GQA (8 KV heads) | GQA (8 KV heads) |
| Position encoding | RoPE (theta=500K) | iRoPE (interleaved) |
| Max context | 128K (trained) | 10M (generalized from 256K training) |
| Context strategy | Progressive 8K->128K training | Train at 256K, generalize via iRoPE |

GQA persists across both generations. The change is in how positional information is injected, not in how KV heads are shared. This confirms that GQA has reached a stable equilibrium in the design space — the quality/efficiency tradeoff is well-understood and well-served by existing tooling.

### Feed-Forward Network

| Aspect | LLaMA 3 | Llama 4 |
|--------|---------|---------|
| Type | Dense SwiGLU | MoE SwiGLU |
| FFN per layer | 1 (always active) | 16-128 experts (2 active) |
| Active FFN params per token | 100% | ~12.5% (Scout), ~1.6% (Maverick) |
| Total FFN params | Fixed | 6.4x (Scout) to 23.5x (Maverick) more |

This is where the architectural philosophies diverge most sharply. LLaMA 3 spends every FFN parameter on every token. Llama 4 maintains a massive library of specialized FFN modules and selects a tiny fraction per token. The hypothesis — grounded in the MoE theory from [[ch-14]] — is that most of a model's knowledge is only relevant to a small fraction of inputs, so conditional computation is more parameter-efficient than unconditional computation.

### Normalization

Both use pre-norm RMSNorm. Llama 4 does not experiment with post-norm, QK-norm, or hybrid normalization. This is notable: normalization is the one component that has remained completely unchanged across four Llama generations. The implication is that pre-norm RMSNorm is at or near a global optimum for this axis of the design space — there is simply not much to gain from normalization innovation.

### Multimodality

| Aspect | LLaMA 3 | Llama 4 |
|--------|---------|---------|
| Approach | Late fusion (adapters) | Early fusion (joint backbone) |
| Vision encoder | Separate, bolted on | MetaCLIP, integrated |
| Training | Text-first, vision added later | Multimodal from pre-training |
| Cross-modal depth | Adapter layers only | Every attention layer |

Early fusion is architecturally more expensive (requires multimodal pre-training data) but produces models with genuinely integrated multimodal understanding. Late fusion is cheaper and more modular but limits cross-modal interaction to the adapter boundary.

---

## 6. The Numbers That Matter

### 6.1 Inference Cost: Dense vs MoE

The most consequential difference between LLaMA 3 and Llama 4 is per-token inference cost. For the flagship models:

- **LLaMA 3 405B:** Every token activates 405B parameters. At FP16, this requires ~810 GB of memory just for weights, spread across multiple GPUs with tensor parallelism.

- **Llama 4 Maverick (400B total / 17B active):** Every token activates 17B parameters — **23.8x fewer** than LLaMA 3 at similar total parameter count. The weight memory is ~800 GB total, but only ~34 GB is active per token. Maverick fits on a single H100 DGX host.

- **Llama 4 Scout (109B total / 17B active):** Fits on a single H100 with int4 quantization. This is a deployment-feasibility milestone: a frontier-class model on one GPU.

### 6.2 Performance Comparison

**LLaMA 3 405B:**

| Benchmark | Score |
|-----------|-------|
| MMLU (5-shot) | 87.3% |
| HumanEval (0-shot) | 89.0% |
| GSM8K (8-shot) | 96.8% |
| GPQA (0-shot, CoT) | 51.1% |

**Llama 4 Maverick (17B active):**

- Beats GPT-4o and Gemini 2.0 Flash across broad benchmarks
- Competitive with DeepSeek-V3 on reasoning and coding at less than half the active parameters
- LMArena ELO: 1417 (experimental chat version)

The direct comparison is difficult because Llama 4 was evaluated on a different benchmark suite, but the headline is clear: Maverick achieves competitive-to-superior quality with 17B active parameters where LLaMA 3 needed 405B. The MoE thesis — that you can decouple knowledge capacity from per-token compute — is empirically validated.

### 6.3 Scaling: 8B Dense vs 17B Active MoE

LLaMA 3 8B vs Llama 4 Scout (17B active, 109B total) is the more revealing comparison at the "deployable on one GPU" scale:

- LLaMA 3 8B: 8B parameters, all active. Decent quality, good inference speed.
- Llama 4 Scout: 17B active (2.1x more per-token compute than LLaMA 3 8B), 109B total. Significantly better quality. Fits on one H100 with int4 quantization.

Scout spends ~2x more compute per token than LLaMA 3 8B, but has access to 109B parameters of knowledge — 13.6x more than LLaMA 3 8B. The MoE architecture makes the marginal knowledge essentially free at inference time (the inactive expert weights sit in memory but cost no compute).

---

## 7. What the Llama Family Teaches About Architecture Design

### 7.1 The Baseline Convergence

Every major LLM released in 2024-2025 shares LLaMA 3's core recipe: decoder-only transformer, GQA, RoPE, RMSNorm, SwiGLU. Raschka ([[raschka-llm-architecture-comparison\|blog]]) documents this convergence across DeepSeek V3, OLMo, Gemma 3, Qwen 3, and others. The components that LLaMA 1 assembled in 2023 have become the unquestioned foundation.

This convergence is informative. It means the "components" axis of the design space is largely settled. The remaining axes of variation are:

1. **Dense vs sparse** (MoE expert count and routing)
2. **Context strategy** (RoPE scaling vs iRoPE vs hybrid attention)
3. **Multimodal integration** (early vs late fusion)
4. **Post-training methodology** (SFT/RLHF/DPO ratios, RL exploration)

Llama 4 makes bets on all four.

### 7.2 When to Be Conservative

LLaMA 3's success demonstrates that architectural conservatism is a legitimate strategy — not timidity, but a calculated decision to invest engineering effort in data, infrastructure, and training rather than novelty. The payoff is:

- Lower training risk (no novel failure modes)
- Faster ecosystem adoption (tooling already exists)
- Better scaling law predictability (well-characterized relationships)
- Easier reproducibility (any team can rebuild it)

The cost is an inference efficiency ceiling. Dense models have a fixed compute-per-token floor that MoE models do not.

### 7.3 When to Be Ambitious

Llama 4 demonstrates that ambition is required when the conservative design hits a ceiling. Three signals that Meta was hitting ceilings:

1. **The 405B model was already near Chinchilla-efficient data.** Adding more data yields diminishing returns.
2. **128K context was a hard limit.** Progressive RoPE extension cannot reach 1M+.
3. **Inference cost made 405B impractical for high-volume deployment.** The cost-per-token was too high for consumer-facing applications.

MoE, iRoPE, and early fusion address all three ceilings simultaneously. The risk is real (new failure modes, immature tooling, complex serving), but the alternative — continuing to scale dense models — has its own diminishing returns.

---

## Core Insights from the Literature

### Insight 1: Architectural conservatism is a training-risk management strategy
**Report:** Meta AI, "The Llama 3 Herd of Models" ([[llama-3\|report]])

LLaMA 3's decision to use a dense transformer with no novel components was not a failure of imagination — it was a deliberate optimization for training predictability at unprecedented scale. When a single 405B training run costs tens of millions of dollars and weeks of H100 time, the value of predictability exceeds the value of novelty. Scaling law experiments on smaller models can reliably predict dense model performance; MoE scaling laws are less characterized. **Guideline:** Default to proven architecture components for your first training run at a new scale. Introduce novelty only when the conservative design demonstrably hits a ceiling.

### Insight 2: MoE decouples knowledge capacity from inference cost
**Report:** Meta AI, "Llama 4" ([[llama-4\|report]])

Llama 4 Maverick has 400B total parameters but 17B active — the same active count as Scout with 109B total. The 291B additional parameters in Maverick are "free" at inference time: they sit in GPU memory but cost no additional compute per token. This decoupling is the fundamental insight of MoE architecture. Knowledge capacity scales with total parameters, but inference cost scales with active parameters. **Guideline:** When inference cost is the binding constraint, MoE lets you increase model knowledge without increasing per-token cost. When training stability is the constraint, dense is safer.

### Insight 3: iRoPE turns context length into an architectural property
**Report:** Meta AI, "Llama 4" ([[llama-4\|report]])

Standard RoPE encodes position into every attention layer, tying the model's effective context to its training context length. iRoPE interleaves position-aware and position-free layers, making context generalization an architectural property rather than a training artifact. Scout trains at 256K context but generalizes to 10M via iRoPE + temperature scaling — a 39x extrapolation. This is qualitatively different from LLaMA 3's progressive RoPE extension, which could at most double the context per stage. **Guideline:** For extreme context lengths (1M+), architectural approaches to length generalization (iRoPE, hybrid attention) are more promising than training-based extension (progressive RoPE scaling).

### Insight 4: The Llama family demonstrates that component convergence precedes structural divergence
**Reports:** LLaMA 1-4 ([[llama-1\|report]], [[llama-2\|report]], [[llama-3\|report]], [[llama-4\|report]])

Across four generations, the sub-block components (GQA, RMSNorm, SwiGLU) have remained stable since LLaMA 1. The variation has moved to higher-order structural decisions: dense vs sparse, context strategy, multimodal integration. This pattern — convergence at the component level enabling divergence at the structural level — is a recurring theme in engineering. Once the building blocks are standardized, innovation moves to how they're composed. **Guideline:** When evaluating a new architecture, focus on structural decisions (MoE, context strategy, fusion method) rather than component choices (attention variant, normalization, activation). The component space is largely explored.

### Insight 5: Lighter post-training unlocks RL exploration
**Report:** Meta AI, "Llama 4" ([[llama-4\|report]])

Llama 4's post-training pipeline removes 50%+ of SFT data (95% for Behemoth) and increases RL. The insight is that heavy SFT over-constrains the model, limiting the behaviors RL can discover. By reducing imitation learning and increasing exploration, the model can find effective strategies that no supervised example demonstrated. This is consistent with the DeepSeek-R1 finding that RL alone can produce chain-of-thought reasoning without any SFT examples of reasoning. **Guideline:** For large models, treat SFT as a lightweight initialization for RL, not as the primary alignment mechanism. The larger the model, the less SFT data it needs.

---

## Key Takeaways

1. **LLaMA 3 is the modern baseline architecture.** GQA + RoPE + RMSNorm + SwiGLU at dense scale. When papers say "standard Transformer," they mean this configuration. Understanding LLaMA 3 is understanding the shared foundation of all 2024-2025 LLMs.

2. **Llama 4 is a structural bet, not a component substitution.** The changes (dense->MoE, RoPE->iRoPE, late fusion->early fusion) are not incremental improvements to individual components. They are a redesign of how components are composed, driven by hitting the ceilings of the dense paradigm.

3. **MoE's core value is decoupling knowledge from compute.** Maverick achieves 400B parameters of knowledge capacity at 17B parameters of per-token compute. This ratio (23.5x) is the quantitative measure of MoE's structural advantage.

4. **Context length is becoming an architectural property.** iRoPE demonstrates that extreme context lengths (10M tokens) can be achieved through architecture (interleaved position-free layers) rather than brute-force training at long context lengths.

5. **Component convergence is complete.** GQA, RMSNorm, and SwiGLU have been unchanged across four Llama generations. Future LLM innovation will occur at the structural level (MoE routing, context strategy, fusion method), not the component level.

6. **The conservatism-ambition tradeoff is real and context-dependent.** LLaMA 3's conservatism was correct for a first training run at 405B scale. Llama 4's ambition is correct for breaking through the ceilings that conservatism revealed. Neither philosophy is universally better.

7. **Post-training is shifting from imitation to exploration.** Llama 4's lighter SFT + heavier RL pipeline suggests that large models benefit more from reinforcement learning exploration than from supervised imitation. The trend accelerates with model size.

---

## References

- [[llama-1|Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023) (report)]] — LLaMA 1 architecture and training
- [[llama-2|Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) (report)]] — GQA introduction, RLHF pipeline
- [[llama-3|Meta AI, "The Llama 3 Herd of Models" (2024) (report)]] — Dense 405B architecture, scaling law-driven design
- [[llama-4|Meta AI, "Llama 4: The Beginning of a New Era of Natively Multimodal AI" (2025) (report)]] — MoE, iRoPE, early fusion, extreme context
- [[raschka-llm-architecture-comparison|Raschka, "The Big LLM Architecture Comparison" (2026) (blog)]] — Cross-model architecture taxonomy and convergence analysis
