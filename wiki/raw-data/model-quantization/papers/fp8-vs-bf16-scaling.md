<!-- scope: scaling-law studies comparing FP8/FP4 to BF16 training
     deps: [[chinchilla]], [[deepseek-v3-fp8]]
     see-also: [[fp8-formats-paper]], [[nvfp4-training]], [[mxfp4-pretraining]]
-->

# Scaling Laws for Precision (Kumar et al. 2024)
- **Core Insight:** Training in lower precision reduces a model's "effective parameter count," and post-training quantization degradation grows monotonically with the number of pretraining tokens — meaning more pretraining can actively hurt a model that will be served at low precision.
- **Guideline:** Choose the serving precision before deciding the pretraining token budget — for any fixed deployment bit-width there is a token count past which additional pretraining is wasted (or harmful) under post-training quantization.
- **Authors:** Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan
- **Year:** 2024 (submitted Nov 7, 2024; rev. Nov 30, 2024)
- **URL:** https://arxiv.org/abs/2411.04330
- **Relevant topics:** scaling laws, low-precision training, post-training quantization degradation, effective parameter count

## Abstract
Standard scaling laws (Chinchilla et al.) are precision-agnostic. This paper extends them by treating the bit-width as a first-class scaling axis, deriving precision-aware scaling laws fit on 465+ controlled pretraining runs of models up to 1.7B parameters trained on up to 26B tokens. Two main findings: (1) training in lower precision is equivalent to training a model with a smaller "effective parameter count," giving a closed-form for how much extra parameters compensate a precision drop; (2) the degradation from post-training quantization (PTQ) grows with pretraining data — past some token threshold, extra pretraining actually makes the deployed (quantized) model worse, because the model's weights become sharper / more outlier-driven and tolerate quantization less.

## Key Contributions
- A precision-aware scaling law: validation loss L(N, D, P) where P is the training precision in bits.
- "Effective parameter" formulation: N_eff = N · f(P), with f(P) → 1 as P → ∞ and a closed-form lower bound at FP4 / FP8.
- Empirical demonstration on > 465 pretraining runs (1.7B params × 26B tokens) that the law holds across W4, W8, BF16, FP32 training regimes.
- The "inference-pretraining paradox": for a fixed serving precision (e.g. INT4 PTQ), there is a token budget D* beyond which loss(D, P_serve) increases with more D.
- Mixed-precision predictor: lets you slice the model into precision tiers (e.g. attention BF16, FFN FP8) and predict total loss as a weighted sum of per-tier laws.

## Key Figures/Tables to Study
- The "inference-pretraining paradox" plot — eval loss as a function of pretraining tokens, with one curve per serving precision; the FP4-serve curve U-shapes upward past ~10× Chinchilla-optimal.
- The N_eff scaling figure — effective parameter multiplier vs precision, fitted across runs.
- The ablation showing the law extrapolates from 1.7B / 26B-token sweep up to bigger published runs (Llama, OPT, Pythia) within ~5% loss.

## Technical Details

### Functional form (loss)
- Adds a precision penalty to the Chinchilla form: L(N, D, P) ≈ A · N^{-α} + B · D^{-β} + E_PTQ(N, D, P_serve) + E_train(P_train).
- E_train decays with training precision: lower training-bit-width raises the asymptote.
- E_PTQ *grows* with D: more pretraining data → larger PTQ penalty for the same serving bit-width.

### Effective parameter count
- N_eff(P) = N · (1 − γ / 2^P) approximately, with γ ≈ 0.4 fit empirically across W4 / W8 / BF16 ablations.
- Implication: dropping from BF16 (P=16) to W4 effectively shaves ~ 25 % of the model's capacity — must add parameters to recover.

### Mixed-precision extension
- Loss is approximately a precision-tier-weighted average; lets you predict the loss of e.g. "attention in BF16, FFN in NVFP4" before running the experiment.

### Validation
- 465 + Llama-style pretraining runs at small scale (≤ 1.7B params, ≤ 26B tokens) — fit there, then extrapolated to published BF16/FP8 results.

## Connections
- [[chinchilla]] — the base scaling law this extends; same N^-α + D^-β structure, with a precision penalty added.
- [[deepseek-v3-fp8]] — the law's prediction for FP8 training agrees with DSV3's < 0.25 % loss gap empirically.
- [[nvfp4-training]] — the FP4 training endpoint of the law; NVFP4 paper validates that with right ingredients FP4 sits near the law's lower envelope.
- [[fp8-formats-paper]] — the bit-format that defines the P-axis at P=8.
- [[low-bit-favors-undertrained]] — companion finding (Ouyang et al.) that low-bit quantization is *kinder* to undertrained models, the same coin from the other side.
