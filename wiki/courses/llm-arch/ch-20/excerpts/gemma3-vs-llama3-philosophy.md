# Excerpt: Gemma 3 vs LLaMA 3 --- Precision vs Scale

<!-- source: [[gemma-3|report]], [[ch-18]], [[raschka-llm-architecture-comparison|blog]] -->

## Two Legitimate Design Philosophies

The open-weight LLM landscape bifurcates into two camps:

1. **Scale-first (LLaMA 3):** Use simple, well-understood components. Push parameters and data. Minimize architectural risk.
2. **Precision-first (Gemma 3):** Use every architectural tool available. Maximize capability per parameter. Accept complexity for efficiency.

Neither is wrong. They optimize for different deployment constraints.

## The Architectural Gap

**LLaMA 3 (405B):**
- All-global GQA attention at every layer
- Pre-norm RMSNorm only
- SwiGLU activation
- No distillation (self-supervised)
- No built-in quantization strategy
- Vision added post-hoc as a separate model

**Gemma 3 (27B):**
- 5:1 local/global interleaved attention with dual RoPE
- Pre-norm + post-norm + QK-norm RMSNorm
- GeGLU activation
- Distillation from larger teacher at all sizes
- QAT with INT4/FP8 support built in
- Native SigLIP vision with Pan-and-Scan

LLaMA 3 uses ~4 architectural techniques. Gemma 3 uses ~10. Each additional technique adds implementation complexity, testing burden, and potential failure modes. But each also contributes to capability-per-FLOP efficiency.

## Performance Per Memory

The comparison that matters for deployment:

| Model | Params | Memory (INT4) | Arena Elo | Elo/GB |
|-------|--------|---------------|-----------|--------|
| Gemma 3 27B | 27B | ~14.1 GB | 1338 | 94.9 |
| LLaMA 3.1 70B | 70B | ~36 GB | ~1280 | ~35.6 |
| LLaMA 3.1 405B | 405B | ~210 GB | 1269 | ~6.0 |

Gemma 3 delivers roughly 16x more capability per GB of memory than LLaMA 3.1 405B. This ratio is the product of every architectural decision stacking multiplicatively:
- 5:1 interleaving reduces KV cache by ~6x
- QAT enables aggressive INT4 without quality loss
- Distillation compresses teacher knowledge into smaller parameters
- GQA reduces per-token cache

## When Each Philosophy Wins

**LLaMA 3's approach wins when:**
- You have abundant GPU memory (data center, multi-GPU)
- Reproducibility matters (simpler architecture = easier to replicate)
- Raw benchmark scores matter more than serving cost
- You want a foundation for fine-tuning (simpler architecture is easier to modify)
- Community adoption matters (LLaMA's architecture is the best-understood)

**Gemma 3's approach wins when:**
- Memory is the binding constraint (edge, single-GPU, mobile)
- Serving cost dominates training cost (high-volume deployment)
- You need multimodal capability without a separate vision pipeline
- Quantization quality matters (QAT > PTQ)
- You deploy at the 4B or 12B scale where distillation provides outsized gains

## The Convergence Question

Will these philosophies converge? Signs point to yes:

- **Llama 4** adopted MoE (architectural complexity), moving toward Gemma's philosophy
- **Gemma 3** still uses GQA rather than MLA, staying conservative on attention
- Both use RoPE, both use RMSNorm, both use gated activations

The shared substrate (RoPE + GQA + RMSNorm + gated FFN) is converging. The divergence is in *how aggressively* each team applies additional optimizations on top of that substrate.

## Implications for Architecture Research

If you are designing a new model:

1. **Start with the LLaMA 3 baseline** --- it is the most battle-tested modern architecture
2. **Add Gemma 3's innovations incrementally** --- each one should be validated with ablation
3. **QK-norm is free** --- add it unconditionally; OLMo 2 + Gemma 3 convergence is strong evidence
4. **5:1 interleaving is validated** --- if KV cache matters for your deployment, adopt it
5. **Distillation requires a teacher** --- only viable if you have access to a larger model in the same family
6. **QAT is strictly better than PTQ** --- budget 5K fine-tuning steps for quantization if you plan to serve quantized

The meta-lesson: architectural innovation has not plateaued. Models that combine multiple synergistic techniques (interleaving + dual RoPE + QK-norm + distillation + QAT) extract significantly more capability per parameter than architecturally conservative models scaled up.
