<!-- scope: orthogonal fine-tuning interactions with rotated quantization (2025)
     deps: [[quarot]], [[spinquant]], [[lora]]
     see-also: [[rotation-unification-2025]], [[learnable-rotation-2025]], [[qlora]]
-->

# Orthogonal Fine-tuning Interactions with Rotated Quantization
- **Core Insight:** Orthogonal fine-tuning (OFT/BOFT) and rotation-based quantization (QuaRot/SpinQuant) both live on the orthogonal group — OFT optimizes a per-layer orthogonal update over a frozen base, and rotated quant optimizes a per-layer orthogonal pre-transform — and they compose: applying OFT *inside* a rotated coordinate system gives both the quantization-friendly base and the orthogonal adaptation in one consistent geometry.
- **Guideline:** When fine-tuning a rotated-and-quantized model, prefer OFT/BOFT-style orthogonal adapters in the rotated frame to standard LoRA in the unrotated frame; the rotated-frame LoRA otherwise re-introduces the outliers the rotation was designed to remove.
- **Authors:** various (OFT/BOFT by Qiu et al.; quant-OFT interactions in 2025 follow-ups)
- **Year:** 2024-2025
- **URL:** representative: OFT (https://arxiv.org/abs/2306.07280) + 2025 quant-OFT integration follow-ups
- **Relevant topics:** orthogonal fine-tuning, LoRA, rotation-based quantization, low-rank adaptation in rotated frames

## Abstract
Both rotation-based quantization (QuaRot / SpinQuant) and orthogonal fine-tuning (OFT / BOFT) parameterize updates on the orthogonal group SO(d). The two have historically been studied separately — quantization rotates *once* to flatten weight/activation distributions, while OFT applies many small orthogonal updates as a fine-tuning prior — but 2025 work shows they should be composed: doing OFT in the *quantization-rotated* frame keeps the fine-tuning updates inside the same orthogonal envelope that the rotation was chosen for. Plain LoRA applied to a SpinQuant-rotated base instead re-introduces the activation outliers the rotation was meant to suppress, because LoRA's low-rank delta is not norm-preserving. The fix: parameterize the adapter as an orthogonal matrix (OFT / BOFT) in the rotated frame, then either fold the resulting orthogonal product back into the weight or keep it as a small fused-rotation adapter.

## Key Contributions
- Identify the failure mode of LoRA-on-rotated-quant: low-rank deltas have non-trivial spectrum and re-introduce outliers.
- Show OFT / BOFT (block-diagonal orthogonal fine-tuning) composes cleanly with QuaRot/SpinQuant — the rotated base + OFT update is itself an orthogonal transform of the original.
- Provide a fusion recipe: OFT update R_oft can be merged into the QuaRot rotation R_q after training (R' = R_q · R_oft), keeping zero inference cost.
- Empirically: orthogonal-adapter fine-tuning of a W4A4-rotated-quant Llama-3 8B recovers within ~ 0.5 % of full-precision fine-tuning, vs 2-3 % for LoRA-on-rotated-quant.
- Connect to [[qlora]] — analogous to how QLoRA composes 4-bit NF4 + LoRA, the orthogonal-adapter line composes rotated quant + OFT.

## Key Figures/Tables to Study
- Activation-distribution histograms before/after LoRA-on-rotated vs OFT-on-rotated — LoRA visibly re-introduces outliers.
- Loss/eval table at W4A4 for: (a) rotated base + full FT, (b) rotated base + LoRA, (c) rotated base + OFT — OFT closes the gap to (a).

## Technical Details

### Why LoRA breaks rotated quantization
- LoRA adds A · B^T where A ∈ R^{d×r}, B ∈ R^{d×r}, r << d.
- A · B^T has spectrum {σ₁, …, σ_r, 0, …, 0}, with σ_i potentially large and concentrated in r directions.
- Applied to a rotated weight W' = W · R, the resulting W' + A · B^T no longer has the flat per-block distribution that R was chosen to produce; the r directions where A · B^T concentrates become new outliers.
- After quantization, those r directions absorb most of the quantization error → quality regresses.

### Why OFT composes cleanly
- OFT parameterizes the update as Δ = R_oft - I where R_oft ∈ SO(d).
- W_new = R_oft · W (left multiplication by an orthogonal matrix).
- Rotated quant: W' = W · R_q. With OFT: W_new' = R_oft · W · R_q.
- Apply quant to W_new': because R_oft is orthogonal, it preserves the per-block norms; the carefully-chosen R_q is *still* flattening the distribution.
- After training, fold: R_q' = R_q (unchanged on the right), but R_oft is absorbed by left-multiplying the *next* layer's input weight or by storing as a small extra rotation factor.

### BOFT (block-orthogonal)
- For large d, full orthogonal parameterization is O(d²) parameters; BOFT uses block-diagonal R_oft with each block parameterized via Cayley / Householder reflectors → O(d · b) parameters where b is block size.
- Compatible with the same composition argument; each block of R_oft commutes with the corresponding block of R_q if R_q is also block-diagonal (which Walsh-Hadamard is).

### Joint training
- Train R_q and R_oft jointly with the same quantization-aware loss; the rotation matrix is shared across the layer's forward passes but the orthogonal update is per-task / per-domain.
- Empirically: joint training beats freezing R_q and training only R_oft by ~ 0.2-0.4 %.

## Connections
- [[quarot]] / [[spinquant]] — the rotation-based quant baselines this work fine-tunes on top of.
- [[lora]] — the canonical low-rank adapter; the failure mode in the rotated frame is the motivation.
- [[qlora]] — the 4-bit NF4 + LoRA precedent; orthogonal-adapter line is the rotated-quant analog.
- [[rotation-unification-2025]] — sibling 2025 unification of rotation methods; both belong to the orthogonal-transformation family.
- [[learnable-rotation-2025]] — the broader 2025 trend of trainable rotations.
- [[peqa]] / [[qa-lora]] — earlier quant-aware adapter approaches that don't address the orthogonal-geometry issue.
