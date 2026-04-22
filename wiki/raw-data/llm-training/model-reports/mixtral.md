<!-- scope: Mixtral of Experts — Mistral AI's open 8x7B and 8x22B MoE
     deps: [[README]]
     see-also: [[deepseek-v3]]
-->

# Mixtral of Experts
- **Core Insight:** A sparse MoE (8 experts, top-2 routing, 12.9B active / 46.7B total) trained with standard recipes competes with 70B dense models at ~3x inference speedup.
- **Guideline:** For latency-sensitive open deployments, top-2 MoE at ~12B active parameters is the sweet spot; post-training fits on standard SFT + DPO.
- **Authors:** Albert Q. Jiang et al. (Mistral AI)
- **Year:** 2024 (Jan)
- **URL:** https://arxiv.org/abs/2401.04088
- **Relevant topics:** Top-2 MoE routing, SFT + DPO post-training, Mistral 7B base, 32K context, Mixtral-Instruct

## Abstract
Mixtral 8x7B is a sparse Mixture-of-Experts model where each layer has 8 feedforward experts and a router selects top-2 experts per token. Built on the Mistral 7B architecture (SwiGLU, GQA, RoPE, sliding-window attention). Total parameters 46.7B; active 12.9B per token. Pretrained on multilingual web data. Mixtral 8x7B — Instruct is fine-tuned via SFT + DPO on curated instruction/preference data. Outperforms Llama 2 70B on most benchmarks at a fraction of the inference cost.

## Key Contributions
- Open release of a competitive 8-expert MoE at 46.7B total / 12.9B active.
- Demonstration that standard SFT + DPO post-training (no iterative RLHF, no dual reward models) suffices on a strong base.
- Top-2 routing with auxiliary load-balancing loss.
- 32K context window.
- Mixtral 8x22B follow-up in 2024 scales the same recipe.

## Key Figures/Tables to Study
- **Router analysis figure:** token-by-token expert utilization heatmap.
- **Performance vs Llama 2 70B table:** especially multilingual, code, math.
- **Expert-per-domain distribution:** whether experts specialize (paper finds: not strongly by topic, but by token-level syntax).

## Technical Details — Post-Training Pipeline

### Architecture context
- **Layers:** 32 transformer blocks, each with 8 FFN experts.
- **Routing:** top-2, linear router; softmax over expert logits; auxiliary load-balancing loss.
- **Total params:** 46.7B; **active per token:** 12.9B.
- **Context:** 32K (sliding-window attention 4K + rotary).
- **Base:** Mistral 7B-style attention (GQA, SwiGLU).

### SFT (Mixtral-Instruct)
- **Data:** curated instruction datasets (specific sources and volume not itemized in the arXiv report).
- Standard completion-masked loss.
- Single-stage SFT prior to DPO.

### DPO
- **Purpose:** preference-optimize Mixtral-Instruct using paired human-feedback data.
- **Hyperparameters (beta, LR, batch size):** not disclosed in the public Mixtral paper.
- Single epoch of DPO on top of SFT.
- Community replications converged on beta ~ 0.1, LR ~ 5e-7 for 8x7B scale, matching Llama 2 / Llama 3 defaults.

### What Mixtral is NOT
- No iterative RLHF rounds (unlike Llama 2/3).
- No RLVR (unlike Tulu 3).
- No dedicated reward model released.
- No disclosed synthetic data pipeline.
Mixtral's post-training is deliberately minimalist; the paper emphasizes the base model and MoE routing as the primary contribution.

### Scale
- **Pretraining tokens:** not disclosed (estimated multi-T).
- **Training compute:** not disclosed.
- **Inference:** ~6x faster than a 46.7B dense equivalent at equivalent active-params count.

### Benchmark highlights
- **MMLU:** 70.6% (outperforms Llama 2 70B's 69.9%).
- **GSM8K:** 58.4% (vs Llama 2 70B 56.8%).
- **HumanEval:** 40.2%.
- **MT-Bench:** Mixtral 8x7B-Instruct 8.3 vs GPT-3.5 Turbo 8.32.

### Mixtral 8x22B (2024 follow-up)
- 141B total / 39B active. Same top-2 routing. 64K context. Same minimalist post-training (SFT + DPO).

## Connections
- [[deepseek-v3]] — much larger MoE (671B total / 37B active) with auxiliary-loss-free routing; contrast in routing design.
- [[llama-2]] / [[llama-3]] — dense competitors; Mixtral outperforms Llama 2 70B at lower inference cost.
- [[dpo]] — DPO as the single preference-optimization stage.
- [[qwen-2.5]] — Qwen MoE releases are a parallel open-MoE line.
