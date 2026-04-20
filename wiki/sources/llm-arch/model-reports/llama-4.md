<!-- scope: Llama 4 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[llama-1]], [[llama-2]], [[llama-3]], [[mixtral]]
-->

# Llama 4: The Beginning of a New Era of Natively Multimodal AI — Technical Report
- **Core Insight:** MoE + interleaved RoPE (iRoPE) enables 10M token context at serving-feasible active parameter count.
- **Guideline:** Decouple knowledge capacity (total params) from per-token compute (active params) via MoE for extreme context.

- **Organization:** Meta AI
- **Year:** 2025
- **URL:** https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **Relevant chapters:** Mixture-of-Experts, native multimodality, extreme context length, early fusion, iRoPE

## Abstract
Llama 4 introduces Meta's first natively multimodal, mixture-of-experts (MoE) foundation models. The family includes Scout (109B total / 17B active, 16 experts), Maverick (400B total / 17B active, 128 experts), and Behemoth (~2T total / 288B active, 16 experts, still in training). Scout supports an industry-leading 10 million token context window. The models use early fusion to jointly process text and vision tokens, and the MoE architecture enables training with less compute than Llama 3 despite having more total parameters.

## Architecture Summary

| Component | Scout | Maverick | Behemoth |
|-----------|-------|----------|----------|
| Total Parameters | 109B | 400B | ~2T |
| Active Parameters | 17B | 17B | 288B |
| Number of Experts | 16 | 128 | 16 |
| Context Length | 10M tokens | Not specified | Not specified |

- **Architecture type:** Mixture-of-Experts (MoE) — first in the Llama family
- **Routing:** Each token sent to 1 shared expert + 1 routed expert (Maverick)
- **Positional encoding:** iRoPE (interleaved Rotary Position Embeddings) — interleaves attention layers with and without positional embeddings, enabling length generalization to extremely long contexts
- **Multimodality:** Early fusion — text and vision tokens processed jointly in a unified backbone
- **Vision encoder:** Based on MetaCLIP, trained separately then adapted with frozen Llama backbone

## Key Architectural Innovations

1. **Mixture-of-Experts architecture** — first Llama model to use MoE, enabling massive total parameter counts while keeping active parameters manageable. Scout fits on a single H100 GPU with int4 quantization despite 109B total parameters.
2. **iRoPE (interleaved RoPE)** — interleaves attention layers without positional embeddings with RoPE layers, using inference-time temperature scaling for length generalization. Enables Scout to generalize from 256K training context to 10M token inference context.
3. **Early fusion multimodality** — integrates text and vision tokens into a unified backbone from the start, rather than adding adapters post-hoc. Enables joint pre-training with unlabeled multimodal data.
4. **Extreme expert scaling** — Maverick uses 128 routed experts (vs. typical 8-16), providing 65x+ more expert combinations and better routing diversity.
5. **MetaP hyperparameter tuning** — technique for reliably setting per-layer learning rates and initialization scales that transfer across batch sizes, model width, depth, and training token counts.
6. **FP8 pre-training** — trained in FP8 precision without quality sacrifice, significantly improving compute efficiency.

## Design Decisions and Tradeoffs

- **MoE over dense:** A major departure from Llama 3's dense architecture. MoE reduces compute per token but introduces routing complexity, load balancing challenges, and communication overhead across devices.
- **iRoPE for extreme context:** Rather than training at long context lengths (expensive), train at 256K and use iRoPE's length generalization properties to scale to 10M at inference time.
- **Early fusion vs. late fusion:** Early fusion enables richer cross-modal interaction but requires multimodal data during pre-training; late fusion (Llama 3's approach) is simpler but limits cross-modal understanding.
- **Shared + routed expert design:** Maverick sends every token to one shared expert for base capability, plus one routed expert for specialization. This ensures no token gets zero expert attention while still enabling specialization.
- **Lightweight post-training:** Revamped pipeline to "lightweight SFT -> online RL -> lightweight DPO." Removed 50%+ of easy SFT data, focusing on harder examples. This prevents over-constraining the model's exploration capability.
- **Behemoth pruning:** For Behemoth, 95% of SFT data removed (vs 50% for smaller models), followed by large-scale RL with curriculum learning, indicating that larger models need even less supervised data and more RL.

## Training Details

- **Training data:** Over 30 trillion tokens (2x+ Llama 3's 15T)
- **Languages:** 200 languages, including 100+ with 1B+ tokens each (10x more multilingual tokens than Llama 3)
- **Data types:** Diverse text, image, and video datasets
- **Compute:** FP8 precision; Behemoth trained on 32K GPUs achieving 390 TFLOPs/GPU
- **Scout pretraining:** ~40 trillion tokens
- **Maverick pretraining:** ~22 trillion tokens

**Post-training pipeline:**
1. Lightweight Supervised Fine-Tuning (SFT) with hard example selection
2. Online Reinforcement Learning with continuous filtering of medium-to-hard prompts
3. Lightweight Direct Preference Optimization (DPO)

## Performance Highlights

**Scout (17B active):**
- Best multimodal model in its class
- Outperforms Gemma 3, Gemini 2.0 Flash-Lite, and Mistral 3.1
- Industry-leading 10M token context window
- Fits on a single H100 GPU (int4 quantized)

**Maverick (17B active):**
- Beats GPT-4o and Gemini 2.0 Flash across broad benchmarks
- Competitive with DeepSeek-V3 on reasoning and coding at less than half the active parameters
- LMArena ELO: 1417 (experimental chat version)
- Fits on single H100 DGX host

**Behemoth (288B active, still training):**
- Outperforms GPT-4.5, Claude Sonnet 3.7, and Gemini 2.0 Pro on STEM benchmarks
- MATH-500 and GPQA Diamond: among the world's top-performing LLMs
