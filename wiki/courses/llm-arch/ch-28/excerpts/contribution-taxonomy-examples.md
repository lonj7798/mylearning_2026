# Contribution Taxonomy: Classifying Architecture Papers

<!-- scope: extended examples of the four contribution categories
     parent: [[ch-28]]
-->

## Purpose

Section 3 of [[ch-28]] introduces four contribution categories: bottleneck identification, novel structural solution, engineering optimization, and empirical validation at scale. This excerpt provides additional classified examples to build pattern recognition.

---

## Category 1: New Bottleneck Identification

The rarest and most impactful category. The paper reframes the community's understanding of *where* the problem lies.

### Example: "Attention Is All You Need" (Vaswani et al., 2017)

**Claimed bottleneck:** Sequential computation in RNNs prevents parallelization across sequence positions during training.

**Why this bottleneck identification mattered:** The NLP community had been incrementally improving RNN architectures (LSTM, GRU, attention over RNN states). Vaswani et al. argued that the *sequential nature of recurrence* was the fundamental constraint, not the capacity of individual cells. By removing recurrence entirely and replacing it with parallel self-attention, they unlocked massive training speedups on GPU hardware.

**In retrospect:** The bottleneck identification was correct but incomplete. The Transformer solved the training parallelism problem but introduced a new bottleneck: $O(N^2)$ memory and compute in sequence length. This new bottleneck spawned the entire subfield of efficient attention — including Flash Attention, which identified a *further* bottleneck (memory IO, not FLOPs) within the attention computation itself.

**Pattern:** Great bottleneck identifications often solve one problem while creating another. The new problem becomes the next research frontier.

### Example: Chinchilla (Hoffmann et al., 2022)

**Claimed bottleneck:** LLMs are significantly undertrained — the community over-invested in parameter count and under-invested in training data.

**Why this bottleneck identification mattered:** Before Chinchilla, the prevailing wisdom (from Kaplan et al.'s scaling laws) was that model size was the dominant factor in performance. Chinchilla showed that training data should scale proportionally with parameters. A 70B model trained on 1.4T tokens outperformed a 175B model trained on 300B tokens.

**Downstream impact:** LLaMA was the direct response — Meta trained smaller models (7B-65B) on significantly more data (1.4T tokens). The entire open-source LLM ecosystem follows Chinchilla's prescription.

---

## Category 2: Novel Structural Solution

The paper proposes a new architectural component that addresses a known problem in a fundamentally different way.

### Example: Rotary Position Embedding (RoPE, Su et al., 2021)

**Known problem:** Transformers need positional information. Absolute positional embeddings (sinusoidal or learned) struggle with length generalization — the model degrades on sequences longer than those seen in training.

**Prior solutions:** ALiBi added linear biases to attention scores. Relative positional encodings (T5 style) added learned biases per head. Both work but have limitations.

**RoPE's structural novelty:** Instead of adding positional information to embeddings or attention scores, RoPE *rotates* query and key vectors by position-dependent angles. The dot product $q_m \cdot k_n$ then naturally encodes the relative position $m - n$ through the rotation angle difference. This is a fundamentally different mathematical mechanism from addition-based approaches.

**Why it is Category 2, not Category 3:** RoPE is not a better implementation of relative position encoding. It is a different mathematical framework (rotation vs. addition) with different properties (exact relative encoding, natural frequency decomposition, compatibility with linear interpolation for length extension). The subsequent PI, NTK-aware, and YaRN extensions all build on RoPE's rotational structure.

### Example: Mixture of Experts Routing (Shazeer et al., 2017)

**Known problem:** Scaling model parameters increases compute proportionally. Larger models are better but more expensive to run.

**Structural novelty:** Only activate a subset of parameters for each input token via a learned routing function. Total parameters scale independently from per-token compute.

**Why it is Category 2:** The idea of conditional computation existed before, but applying it at the granularity of Transformer feed-forward layers with a differentiable routing function was new. The specific interaction between routing, load balancing, and training stability created a novel set of research challenges that continue today (DeepSeek-V3's auxiliary-loss-free balancing, shared experts, device-limited routing).

---

## Category 3: Engineering Optimization

The paper improves an existing method through better implementation without changing the mathematical formulation.

### Example: Flash Attention 2 ([[flash-attention-2|paper]])

**What changed from Flash Attention 1:**
- Better warp partitioning within thread blocks
- Reduced non-matmul FLOPs by restructuring the softmax computation
- Added sequence-level parallelism for better GPU occupancy

**What did NOT change:** The tiling algorithm, the online softmax, the recomputation strategy, and the exact mathematical result.

**Why this is Category 3, not Category 2:** FA-2 computes the same result as FA-1 using the same high-level algorithm. The contribution is engineering: making the existing algorithm run 2x faster by using the GPU more efficiently. This is valuable — FA-2's 72% model FLOPs utilization is remarkable — but it is optimization, not invention.

**How to tell the difference:** If you could swap FA-2 for FA-1 and get the same model quality (just slower training), it is engineering. If swapping would change the model's learned representations, it would be architectural.

### Example: Megatron-LM Tensor Parallelism ([[megatron-lm|paper]])

**What it does:** Splits individual Transformer layers across GPUs by partitioning weight matrices along specific dimensions, with carefully chosen split points that minimize inter-GPU communication.

**Why it is engineering:** The Transformer architecture is unchanged. Tensor parallelism is a distributed systems optimization that enables training larger models on multi-GPU hardware. The model produced is mathematically identical to one trained on a single (impossibly large) GPU.

---

## Category 4: Empirical Validation at Scale

The paper demonstrates that known techniques work at unprecedented scale, providing confidence for the community.

### Example: GPT-3 (Brown et al., 2020)

**Architectural novelty:** Essentially none. GPT-3 is a scaled-up GPT-2 with minor hyperparameter changes (alternating dense/sparse attention layers, learned positional embeddings, different layer widths).

**Contribution:** Demonstrating that scaling a Transformer decoder to 175B parameters unlocks in-context learning — the ability to perform new tasks from a few examples in the prompt without gradient updates. This was an emergent capability that could not have been predicted from smaller models.

**Why this matters despite lacking architectural novelty:** GPT-3 showed that scale itself is an architectural decision. The paper's contribution was not "here is a new architecture" but "here is evidence that this architecture does surprising things at this scale."

### Example: Llama 3 ([[llama-3|report]])

**Architectural novelty:** Minor. Llama 3 uses the same GQA + RoPE + RMSNorm + SwiGLU stack as Llama 2, with incremental improvements (larger vocabulary, grouped query attention at all sizes).

**Contribution:** Validated the Llama 2 architecture at 405B parameters with 15T training tokens. Demonstrated that the "standard" architecture, when trained with sufficient data at sufficient scale, matches or exceeds more exotic architectures. This validation gave the community confidence to adopt the GQA + RoPE + RMSNorm + SwiGLU stack as the default baseline.

---

## Using the Taxonomy in Practice

When you finish reading a paper's ablation tables (Pass 2 of the three-pass method), classify the contribution:

| Category | Frequency | Time Investment | Signal |
|----------|-----------|-----------------|--------|
| 1. Bottleneck ID | ~5% of papers | Full Pass 3, deep study | Potential field shift |
| 2. Structural | ~15% of papers | Pass 3, implementation study | New design option |
| 3. Engineering | ~30% of papers | Skim Pass 3, check benchmarks | Practical speedup |
| 4. Empirical | ~50% of papers | Pass 2 sufficient | Scale data point |

**Mixed categories are common.** DeepSeek-V2 is Category 2 (MLA) + Category 3 (device-limited routing) + Category 4 (validation at 236B). Mamba is Category 1 (content-based reasoning bottleneck) + Category 2 (selective SSM). The highest-impact papers often combine Category 1 with Category 2 — they identify a new bottleneck and propose a structural solution in the same work.

**A paper's category does NOT determine its quality.** A superb Category 3 paper (Flash Attention 2) is more valuable than a weak Category 2 paper (a novel but poorly-validated architecture). The taxonomy helps you calibrate expectations and reading depth, not quality.
