<!-- scope: inference-relevant facts from DeepSeek-R1 report and model cards
     see-also: deepseek-v3-inference, qwen-3-inference
-->

# DeepSeek-R1 Inference
- **Core Insight:** Reasoning models change inference cost because they intentionally spend more output tokens on deliberation.
- **Guideline:** Benchmark DeepSeek-R1 with explicit max reasoning/output length, sampling settings, and stop behavior; otherwise latency comparisons are misleading.
- **Authors:** DeepSeek-AI
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2501.12948
- **Relevant topics:** reasoning model, MoE, MLA, long output, sampling, distillation

## Abstract
DeepSeek-R1 is a reasoning-oriented model released with large MoE checkpoints and distilled dense variants. The main inference impact is not only architecture inherited from DeepSeek-V3-style models, but also behavior: reasoning prompts can produce substantially longer completions before final answers.

## Key Contributions
- Demonstrates open reasoning-model behavior with long chain-of-thought-style generation.
- Releases distilled variants for smaller dense serving targets.
- Inherits MoE/MLA considerations for the largest model.
- Documents prompting and generation behavior that affect latency and cost.

## Key Figures/Tables to Study
- DeepSeek-R1 paper tables: model variants and distillation targets.
- Model card inference examples: recommended prompt format and sampling notes.
- Runtime docs from serving frameworks: special handling for long reasoning outputs.

## Technical Details
For serving, output-token budget is the main capacity driver. Reasoning mode can increase decode time, KV-cache retention time, and queue occupancy. Sampling settings disclosed in model cards should be preserved for quality reproduction, while production serving may lower maximum tokens or expose reasoning-effort controls to manage latency.

## Connections
- [[deepseek-v3-inference]] covers the base architecture implications.
- [[ttft-tpot-itl]] explains why long decode changes TPOT and request latency.
