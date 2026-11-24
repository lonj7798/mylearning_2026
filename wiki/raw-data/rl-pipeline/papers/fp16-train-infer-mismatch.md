<!-- scope: Qi, Liu, Zhou et al., "Defeating the Training-Inference Mismatch via FP16" (arXiv 2510.26788, Oct 2025): BF16 rounding as root cause of RL train–inference mismatch, FP16 + loss scaling, offline mismatch analysis, perfectible-dataset sanity test, comparison to TIS / MIS / GSPO, precision ablation, MoE / LoRA / 14B / OctoThinker generalization
     deps: [[grpo]], [[truncated-importance-sampling]]
     see-also: [[thinking-machines-nondeterminism]], [[nemo-rl-tp-accuracy]], [[mai-rl-infrastructure]], [[mai-thinking-1]], [[verl-rollout]]
-->

# Defeating the Training-Inference Mismatch via FP16
- **Core Insight:** The train–inference mismatch in LLM RL is mainly a precision problem: BF16 has 7 mantissa bits, so small implementation differences between engines round differently and compound over long sequences. FP16 (10 mantissa bits) keeps the two engines' outputs much closer; switching both training and inference to FP16 removes most of the mismatch without algorithm changes (Abstract; Sec. 3.4).
- **Guideline:** For RL fine-tuning of a pre-trained model, use FP16 in both sampler and learner with dynamic loss scaling; the dynamic range is already fixed by pre-training, so BF16's range buys little. Under FP16 the plain unbiased importance-weighted policy gradient works, and the choice among GRPO / TIS / MIS / GSPO matters little. A growing mismatch is an early warning of collapse.
- **Authors:** Penghui Qi*†, Zichen Liu*, Xiangxin Zhou*, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin (Sea AI Lab; National University of Singapore). *core contributors, †project lead.
- **Year:** 2025 (arXiv v1, 30 Oct 2025).
- **URL:** https://arxiv.org/abs/2510.26788 (code: https://github.com/sail-sg/Precision-RL)
- **Source type:** paper (preprint, 16 pages)
- **Relevant topics:** BF16 vs FP16, loss scaling, importance sampling (IS / TIS / MIS), deployment gap, sanity test for RL algorithms, MoE RL, LoRA RL

## Summary
The paper names two consequences of sampling from inference policy µ while training policy π: a **biased gradient** (Eq. 3) and a **deployment gap**, since θ optimized under π is not optimal under µ (Eq. 4). Algorithmic fixes address only the first, and the GRPO-based implementations of [[truncated-importance-sampling]] and sequence-level MIS need an extra forward pass, "approximately 25%" more training cost assuming backward = 2× forward (Sec. 1, 2.1.1). Engineering fixes (FP32 `lm_head`, manual alignment, deterministic inference) are described as insufficient or costly (Sec. 2.2). The proposal is to train and sample in FP16.

## Key Contributions
- Diagnosis: BF16 rounding is the root cause; FP16 offers "8 times more precision (2^10 values vs. 2^7 values)" (Sec. 3.4).
- Offline measurement: FP16 sequence-level log-ratio mismatch "approximately 24× smaller" than BF16 (Fig. 2, Sec. 3.5).
- A "perfectible dataset" sanity test: a reliable algorithm should approach 100% training accuracy (Sec. 4).
- Results across VeRL and Oat, algorithms, model families, LoRA, MoE and a 14B dense model (Fig. 1, 3–6).

## Key Figures/Tables to Study
**Table 1 (Sec. 3.1).** FP16: 5 exponent / 10 mantissa bits, smallest normal ≈ 6.1×10⁻⁵, max ≈ 6.6×10⁴, next value above 1 = 1 + 2⁻¹⁰ ≈ 1.000977. BF16: 8 / 7 bits, ≈ 1.2×10⁻³⁸, ≈ 3.4×10³⁸, 1 + 2⁻⁷ ≈ 1.007812.

**Table 2 (Sec. 3.5)** DeepSeek-R1-Distill-Qwen-1.5B inference scores, no training:

| dtype | AMC23 (8K) | AIME24 (8K) | AMC23 (32K) | AIME24 (32K) |
|---|---|---|---|---|
| BF16 | 50.38 | 22.60 | 62.35 | 29.90 |
| FP16 | 50.60 | 20.10 | 63.10 | 30.94 |
| FP32 | 51.54 | 22.30 | 62.42 | 28.44 |

Higher inference precision alone does not improve scores. **Fig. 2**: token-probability scatter and sequence log-ratio vs length; panel annotations show "Slope = -1.01, KL = 7.64" for BF16 vs "Slope = -0.07, KL = 0.32" for FP16 (KL between µ and π; argument order not legible in the text layer).

## Technical Details
**Estimators (Sec. 2.1).** PG-IS: E_{y∼µ(θ′)}[π(y|x, θ)/µ(y|x, θ′) · ∇log π · A] (Eq. 5). TIS: min(ratio, C) (Eq. 6). MIS: ratio · 𝟙{ratio ≤ C} (Eq. 7). GRPO-Token-TIS (Eq. 9) multiplies each token's clipped GRPO term by min(ρ_t, C), ρ_t = π(y_t|θ′)/µ(y_t|θ′). GRPO-Seq-MIS (Eq. 10) uses one sequence ratio ρ with mask ρ ≤ C. The GRPO baseline is Dr.GRPO.

**Sanity test (Sec. 4, 4.1).** MATH problems where the initial model's accuracy over 40 rollouts is 20–80% → 1,460 questions for DeepSeek-R1-Distill-Qwen-1.5B; pass criterion e.g. >95% training accuracy. Setup: 8 × A100 80G, context 8,000, 64 questions × 8 rollouts per policy iteration, 4 gradient steps, clip_higher 0.28, C = 3 for IS methods.

**BF16 results (Sec. 4.2, Fig. 3).** Vanilla GRPO peaks at 73% (VeRL) / 84% (Oat) then collapses. GRPO-Token-TIS collapses after 82% / 88%. GSPO stays stable longer than token-TIS "despite not using the inference policy µ at all" (its VeRL gradient norm became NaN after 1,200 steps). Only GRPO-Seq-MIS does not collapse, but converges slowly and peaks at 95% training accuracy and 34% AIME 2024, vs 99% and 39% for FP16. Collapsing runs show growing mismatch beforehand, with π − µ approaching extreme values (one policy near 1, the other near 0).

**FP16 results (Sec. 4.3).** With FP16 the algorithms become "almost indistinguishable"; the problem becomes "nearly on-policy". Original GRPO is slightly lower on AIME 2024 and slightly higher on AIME 2025.

**Precision ablation (Sec. 4.4, Fig. 5; VeRL, vLLM + FSDP).** With BF16 training, raising inference precision prolongs stability; FP32 inference + BF16 training is fully stable but FP32 inference is "nearly three times slower". FP16 for both gives the lowest mismatch, near 100% training accuracy, and no speed loss.

**Generalization (Sec. 5).** MoE: Qwen3-30B-A3B-Base with GRPO-Seq-MIS, GRPO-Token-TIS, PG-Seq-TIS — FP16 higher training and validation accuracy for all three. LoRA: Qwen2.5-Math-1.5B, rank 32, α = 64, LR 4×10⁻⁵ — BF16 collapses after about 600 steps, FP16 does not. Dense: Qwen3-14B-Base with DAPO — FP16 rewards rise faster and AIME 2024 is higher. OctoThinker-3B (from Llama3.2-3B) GRPO — BF16 destabilizes after about 150 steps.

**Caveats (Sec. 6).** FP16 is not claimed to be universally optimal; very large models may face overflow; efficiency pressure pushes toward FP8. Under BF16 there is a bias–variance trade-off: low-variance biased methods (GRPO, token-TIS, GSPO) converge fast then collapse; less biased ones (PG-Seq-IS, GRPO-Seq-MIS) are stable but slow.

**Contrast with production.** [[mai-thinking-1]] (Sec. 3.6.4) reports using BF16 for learner and inference plus routing and top-p mask replay; [[mai-rl-infrastructure]] removes the gap by bitwise parity rather than by precision change.

## Verification
Read the 16-page arXiv PDF (v1) via `pdftotext` on 2026-11-24: Abstract, Sec. 1–6. All numbers are from the text, Tables 1–2 and figure text layers (Fig. 2 annotations). Appendix A (Table 3 hyperparameters) and Fig. 6 were not read in detail.
