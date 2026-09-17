---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2502.02737v1 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2502.02737
created_at: "2026-09-15"
---

# Excerpt: SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model

**Report:** Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. (Hugging Face). arXiv v1 2025-02-04. Source type: official technical report. This excerpt covers the pre-training budget, schedule, batch, and stage mixtures used in ch-14a.

## Budget and rationale (§1, §4)
- SmolLM2 1.7B is trained on 11 trillion tokens, "approximately two epochs on our collected datasets" (§4).
- Rationale, quoted: longer training "deviates from the Chinchilla-optimal guidelines", but "the resulting performance gains and reduced inference costs make extended training a worthwhile trade-off" (citing de Vries, 2023) (§4).
- Secondary statements about other models, as printed by this report: Qwen2-1.5B on 7T tokens, Qwen2.5-1.5B on 18T, and Llama3.2-1B "derived from a pruned 8B model, was trained using distillation on 9 trillion tokens" (§4).
- Cost: "around 1e23 FLOPs, or $250,000 USD worth of GPU compute for SmolLM2" (§1); the cost motivated an "online" approach of adjusting mixtures during one run instead of several from-scratch runs (§4).
- Design principles (§4): (1) "Performance-driven interventions" on key benchmarks; (2) upsampling high-quality math and code "during the annealing phase"; (3) introducing medium-sized datasets mid-training "to avoid dilution by larger datasets early on"; (4) staying close to "the recommended 4–5 epoch threshold for most datasets", citing Muennighoff et al. (2023).

## Training setup (§4.1, App. A Table 6, Fig. 3)
- 256 H100 GPUs, nanotron; AdamW (β1, β2) = (0.9, 0.95) (§4.1).
- Warmup Stable Decay (WSD) schedule "to avoid setting a fixed training duration"; 2,000 warmup steps; peak LR 5.0 × 10^−4; decay to zero "over 10% of the total training steps" (§4.1; Fig. 3 caption). Tokenizer vocabulary 49,152 tokens (§4.1).
- Table 6: 24 layers, model dimension 2,048, FFN dimension 8,192, 32 heads, sequence length 2,048 (before context extension), "Token per batch 2M", tied embeddings, RoPE θ = 10,000.

## Stage mixtures (§4.2–§4.5, Fig. 2, Table 3)
| Stage | Tokens | English web | Code | Math | Other |
|---|---|---|---|---|---|
| 1 (stable) | 0–6T | 90% (FineWeb-Edu 60 / DCLM 40) | 10% StarCoderData | 0% | — |
| 2 (stable) | 6–8T | 75% | 20% | 5% (OWM) | — |
| 3 (stable) | 8–10T | 74% (FineWeb-Edu 40 / DCLM 60) | 16% (Stack-Edu) | ≈10% | — |
| 4 (decay) | 10–11T | 58% | 24% (Stack-Edu, more languages) | 14% | 4% Cosmopedia v2 |

- Text-stated values: stage 1 code limited to 10% "to ensure approximately 4 epochs over 11T tokens", no math "due to our math datasets' relatively small size" (§4.2); stage 2 "75% English web data ... 20% code data, and 5% math data" (§4.3); stage 3 math "approximately 10%" (§4.4); stage 4 math 14%, Stack-Edu 24%, web 58%, Cosmopedia v2 4%, including 0.08% OWM and 0.02% AugGSM8K, "an augmented version of the GSM8K benchmark's training set" (§4.5). Stage-1 web 90% and stage-3 web 74% and code 16% are read from Fig. 2.
- Stage 2 finding: "OWM integration had no significant impact on math performance" (§4.3).
- Table 3, average score after each stage (stages 1 → 4): Knowledge/Reasoning 55.50, 56.76, 57.47, 60.24; Math 3.21, 3.7, 7.27, 22.07; Code 8.87, 10.56, 16.75, 23.21; Generative Tasks 31.54, 31.30, 34.70, 36.12.
- Stage 3 had a loss spike that "remained even after rewinding training and skipping data associated with the spike"; cause undetermined (§4.4).
- Released base model: context extension 2k → 8k starts "by taking an intermediate checkpoint from stage 4 (before the final 75 billion tokens of training)", with RoPE 130k and 40% long-context documents; "After this step, we obtain the final SmolLM2 base model" (§4.6).

## Smaller sizes (§6)
- SmolLM2-360M on 4T tokens and SmolLM2-135M on 2T tokens; data ablations re-run at the target training length; single-stage training; "WSD scheduler with 20% decay and a learning rate of 3.0 × 10^−3" (§6).

## Verification
- Read on 2026-09-15 against arXiv:2502.02737v1 PDF text: §1, §2, §4–§4.6, §6, App. A Table 6, Fig. 2–3, Table 3.
- Not reported by the source: batch size and sequence length for 360M and 135M; tokens of the context-extension stage; an ablation that separates the stage-4 data change from the LR decay; seeds or variance for Table 3.
