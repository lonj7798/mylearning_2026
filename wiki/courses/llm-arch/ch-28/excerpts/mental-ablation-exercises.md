# Mental Ablation Exercises

<!-- scope: practice exercises for mental ablation on architecture design choices
     parent: [[ch-28]]
-->

## Purpose

Mental ablation — asking "what would happen without this?" for each design choice — is a skill that improves with practice. These exercises use components from papers covered in earlier chapters. For each one, attempt the mental ablation before reading the analysis.

---

## Exercise 1: Flash Attention's Recomputation in the Backward Pass

**Component:** Flash Attention ([[flash-attention|paper]]) recomputes the attention matrix from Q, K, V during backpropagation instead of storing it from the forward pass.

**The question:** What would happen if you stored the attention matrix instead of recomputing it?

### Analysis

**Without recomputation (store the matrix):**
- You must store the full $N \times N$ attention matrix $P$ for each head, each layer
- At seq length 4K: 4096 x 4096 x 2 bytes (FP16) = 32 MB per head per layer
- For a 32-layer, 32-head model: 32 x 32 x 32 MB = 32 GB just for attention matrices
- This dominates GPU memory and limits sequence length
- The backward pass would be faster in FLOPs (no recomputation), but the memory constraint would force shorter sequences or smaller batches

**With recomputation (Flash Attention's choice):**
- Memory usage drops from $O(N^2)$ to $O(N)$ per layer
- Extra FLOPs are incurred during the backward pass
- But the recomputation happens in fast SRAM (tiled, just like the forward pass)
- Net effect: **more FLOPs, less memory, faster wall-clock** because the recomputation is SRAM-bound (fast) while storing/loading the matrix is HBM-bound (slow)

**The lesson:** Recomputation is not "wasting compute." When the alternative is HBM-bound storage and retrieval, recomputing in SRAM is genuinely faster. This is counterintuitive only if you think of FLOPs as the bottleneck — the whole point of Flash Attention is that they are not.

---

## Exercise 2: Mamba's 1D Convolution Before the Selective SSM

**Component:** Each Mamba block ([[mamba|paper]]) applies a short 1D convolution (kernel size 4) to the input before feeding it to the selective SSM.

**The question:** What would happen without the convolution?

### Analysis

**Without the convolution:**
- The selective SSM receives each token independently at the projection level
- The SSM recurrence provides sequential context, but only through the state — the input to the SSM has no local context
- Short-range patterns (bigrams, trigrams) that a convolution captures cheaply would need to be learned entirely through the SSM state dynamics
- The SSM state dimension (typically N=16) would need to encode both local n-gram patterns and long-range dependencies

**With the convolution:**
- Local context (4 tokens) is mixed before the SSM sees the input
- The SSM can focus its limited state capacity on medium- and long-range dependencies
- This is a division of labor: convolution handles local, SSM handles global
- The convolution is computationally trivial (kernel size 4) relative to the SSM

**Connection to Transformers:** This mirrors how attention handles local patterns through nearby tokens while the FFN processes each position. The division of labor between local mixing and global processing appears repeatedly across architectures.

**The lesson:** Small, cheap components can have outsized importance when they offload work from a more expensive component. The convolution costs almost nothing but frees the SSM state for higher-level patterns.

---

## Exercise 3: GQA's Choice of G = H/8 Groups

**Component:** The standard GQA configuration ([[gqa|paper]]) uses $G = H/8$ groups (e.g., 8 KV heads for 64 query heads in Llama 2 70B).

**The question:** What would happen with G = H/4 (more groups, less compression)?

### Analysis

**With G = H/4 (e.g., 16 KV heads for 64 query heads):**
- KV cache is 4x smaller than MHA (vs 8x with G = H/8)
- Each KV head is shared by 4 query heads (vs 8 with G = H/8)
- More diverse KV representations — each group can specialize more
- But 2x more KV cache memory than the standard G = H/8

**With G = H/8 (standard):**
- KV cache is 8x smaller than MHA
- Each KV head is shared by 8 query heads
- Less representational diversity per KV head
- But the quality-speed tradeoff hits a sweet spot: most of MQA's speed with most of MHA's quality

**Why H/8 won:**
- The quality loss from G = H/4 to G = H/8 is ~0.1-0.3% on standard benchmarks — barely measurable
- The inference speedup from 4x to 8x KV reduction is significant: halving the bytes loaded per decoding step
- The marginal return on representational diversity diminishes rapidly — 8 distinct KV heads are already enough for most attention patterns

**The lesson:** The "right" number of groups is determined by the diminishing-returns curve of quality vs. compression. The GQA paper's ablation of different G values is a textbook example of finding a knee in this curve.

---

## Exercise 4: DeepSeek-V2's Shared Experts

**Component:** DeepSeek-V2 ([[deepseek-v2|report]]) uses 2 shared experts that are always active alongside 6 of 160 routed experts per token.

**The question:** What would happen without shared experts?

### Analysis

**Without shared experts:**
- All expert knowledge is distributed across routed experts only
- Common, frequently-needed computations (language modeling basics, syntactic processing) must be redundantly learned by many experts
- Token routing becomes more critical: if a token's assigned experts lack basic capabilities, quality degrades
- Load balancing becomes harder because some "utility" experts would be in constant demand

**With shared experts:**
- The 2 shared experts handle common computations that every token needs
- Routed experts can specialize in less frequent, more domain-specific computations
- Routing failure is less catastrophic because the shared experts provide a quality floor
- The total compute cost increases (2 extra experts per token), but this is modest relative to the 6 routed experts

**The tradeoff:** Shared experts add ~33% to per-token expert compute (8 active experts total vs 6 routed-only). But they reduce the variance in quality across different routing assignments, making the model more robust. DeepSeek-V3 kept shared experts (reduced to 1), confirming their value.

**The lesson:** In MoE architectures, the distinction between "always-on" and "routed" components matters. Shared experts are a form of architectural insurance — they guarantee a minimum quality floor regardless of routing decisions. This is an example of brute-force engineering (always compute more) solving a hard optimization problem (perfect routing).

---

## General Framework for Mental Ablation

When you encounter any architectural component, run through these steps:

1. **State the component clearly:** What exactly does it do? (Not what the paper *says* it does — what it *actually* does, mathematically or computationally.)

2. **Remove it:** What fails, degrades, or changes? Be specific about the failure mode.

3. **Find the simplest replacement:** Is there a simpler component that prevents the same failure? If yes, the component may be over-engineered. If no, the component is earning its complexity.

4. **Check the ablation table:** Does the paper actually ablate this component? If not, you have identified a gap in the evidence. If yes, does the measured effect match your predicted failure mode?

5. **Consider scale dependence:** Would the component matter more or less at a different model size, sequence length, or batch size? Components that matter only at specific scales are less fundamental than components that matter universally.
