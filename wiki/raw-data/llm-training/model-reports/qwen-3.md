<!-- scope: Qwen3 technical report with unified thinking/non-thinking post-training
     deps: [[deepseek-r1]], [[llama-3]]
     see-also: [[deepseek-v3]], [[tulu-3]], [[self-instruct]]
-->

# Qwen3 Technical Report
- **Core Insight:** Qwen3 treats "reasoning mode" as a trainable capability inside one unified model family rather than as a separate model class, combining reasoning-stage pretraining, long-CoT cold-start finetuning, RL, and strong-to-weak distillation.
- **Guideline:** If you want controllable test-time compute, train a single model to support both fast and deep modes, then expose a thinking budget instead of maintaining separate chat and reasoning stacks.
- **Authors:** Qwen Team
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2505.09388
- **Relevant topics:** thinking budget, reasoning mode, multilingual pretraining, distillation, long-CoT cold start, RL

## Abstract
Qwen3 is a family of dense and MoE open-weight LLMs designed to unify rapid-response chat behavior and slower multi-step reasoning inside the same model. The report describes a 36T-token multilingual pretraining run across 119 languages and dialects, a three-stage pretraining curriculum, and a multi-stage post-training process that first builds reasoning ability with long-CoT SFT and RL, then fuses reasoning and non-reasoning data into a single deployment-friendly model family.

## Key Contributions
- A **unified thinking / non-thinking framework** rather than separate chat and reasoning models.
- A **thinking budget** interface so reasoning effort can be scaled at inference time.
- Large multilingual pretraining: **36T** tokens across **119** languages and dialects.
- A three-stage pretraining curriculum with a dedicated **reasoning stage** and a **long-context stage**.
- Strong-to-weak distillation for smaller models, with the report explicitly noting that distillation outperforms RL in both quality and efficiency for those models.

## Key Figures/Tables to Study
- **Introduction / opening summary:** the report states the whole recipe compactly.
- **Tables 1 and 2:** dense and MoE architecture tables.
- **Section 3.1 and 3.2:** pretraining data construction and the three-stage pretraining schedule.
- **Post-training overview:** this is where the model-family design philosophy becomes clear.

## Technical Details

### Model family
- Dense models from **0.6B to 32B**.
- MoE models including **Qwen3-30B-A3B** and **Qwen3-235B-A22B**.
- Dense models use GQA, SwiGLU, RoPE, RMSNorm, and add **QK-Norm** for training stability.

### Pretraining data
- Total of **36T tokens** across **119 languages and dialects**.
- Data expansion includes:
  - OCR-style text extraction from large PDF corpora using **Qwen2.5-VL**
  - synthetic math data from **Qwen2.5-Math**
  - synthetic code/data variants from **Qwen2.5-Coder** and related models
- The report says the data is annotated at large scale for educational value, domain, and safety, then mixed at the **instance level** using proxy-model ablations.

### Three-stage pretraining
1. **General stage:** over **30T** tokens at sequence length **4096**.
2. **Reasoning stage:** about **5T** higher-quality tokens with more STEM, coding, reasoning, and synthetic data, still at **4096**.
3. **Long-context stage:** hundreds of billions of tokens at **32768** sequence length, later supporting longer inference contexts.

### Long-context and deployment
- Uses **YARN** and **Dual Chunk Attention** to increase usable context during inference.
- Dense models expose contexts up to **128K** in the released family tables.

### Post-training
- Stage 1-2: build reasoning ability with **long-CoT cold-start finetuning** and **RL** focused on math and coding.
- Stage 3-4: merge data with and without reasoning paths, then run **general-domain RL**.
- Smaller models are improved with **off-policy and on-policy strong-to-weak distillation** from larger teacher models.
- The report explicitly says **distillation from advanced teacher models significantly outperforms RL** for smaller models in both performance and efficiency.

## Connections
- [[deepseek-v3]] is the best comparison on open 2024-2025 reasoning-centric model reports.
- [[deepseek-r1]] is the pure-RL reasoning extreme; Qwen3 is a more integrated multi-mode recipe.
- [[llama-3]] is the synthetic-data + DPO loop analogue from Meta.
- [[self-instruct]] matters here conceptually because Qwen3 also leans on synthetic data generation as a scaling tool rather than relying only on raw human data.
