<!-- scope: Microsoft Research's BitNet team — 1-bit / 1.58-bit LLM training from scratch + bitnet.cpp inference
     deps: [[bitnet]], [[bitnet-b158]], [[bitnet-a48]], [[bitnet-models]]
     see-also: [[microscaling-formats]]
-->

# Microsoft Research BitNet Team — 1-Bit / 1.58-Bit LLM Pretraining
- **Core Insight:** BitNet shows that sub-2-bit LLMs become viable when quantization is part of pretraining from scratch rather than imposed after training.
- **Guideline:** Read this lab track when studying ternary weights, trained-from-scratch low-bit transformers, and specialized inference kernels for 1.58-bit models.
- **Authors:** Microsoft Research BitNet team
- **Year:** 2023–2026
- **URL:** https://github.com/microsoft/BitNet ; https://huggingface.co/microsoft/BitNet-b1.58-2B-4T
- **Relevant topics:** BitNet, BitNet b1.58, ternary weights, bitnet.cpp, sub-2-bit training

## Summary
Microsoft Research's BitNet team — led by Furu Wei with first authors Hongyu Wang and Shuming Ma — established the **sub-2-bit-LLM-from-scratch** research line. Starting with [[bitnet]] (2023, pure 1-bit weights), they generalized to ternary weights in [[bitnet-b158]] (2024) under the "Era of 1-bit LLMs" framing, then extended to 4-bit activations in [[bitnet-a48]] (2024). The team also released the canonical 2B-scale trained-from-scratch ternary checkpoint ([[bitnet-models]] — `microsoft/BitNet-b1.58-2B-4T`) plus the `bitnet.cpp` inference framework with hardware-specialized kernels for the ternary regime.

## Notable Works
- [[bitnet]] (Wang 2023) — original 1-bit transformer; BitLinear layer; trained from scratch.
- [[bitnet-b158]] (Ma 2024) — ternary {-1, 0, +1} weights; the "Era of 1-bit LLMs" claim; matches FP16 perplexity at ≥3B scale.
- [[bitnet-a48]] (Ma 2024) — 4-bit activations on top of 1.58-bit weights; absmean activation scaling.
- BitNet-b1.58-2B-4T (Microsoft 2024) — official 2B-parameter, 4T-token release.
- bitnet.cpp framework (2024-2025) — ternary-specialized CPU/GPU/NPU inference.
- Microscaling formats work (Rouhani 2023) — adjacent Microsoft contribution (different group): the OCP MX format spec.

## Recurring themes
- **Training from scratch is the unlock for sub-2-bit**: PTQ can't reach 1-bit because the model never learned to be robust to it; pretraining with quant-in-the-loop solves this.
- **Ternary, not binary**: pure binary doesn't include the "zero / no-op" state that the empirical zero-clustering of LLM weights requires; ternary {-1, 0, +1} at log2(3) ≈ 1.58 bits/weight is the right point.
- **Hardware co-design**: the kernels (bitnet.cpp) are part of the contribution, not an afterthought; ternary multiplies reduce to add/subtract/no-op.

## Open Resources
- BitNet repo: https://github.com/microsoft/BitNet (apache-2.0)
- 2B-4T model: https://huggingface.co/microsoft/BitNet-b1.58-2B-4T
- 2B-4T gguf: https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf
- Papers: search Furu Wei + Hongyu Wang + Shuming Ma on Microsoft Research publications page.

## Connections
- [[dettmers-group]] / [[han-song-mit]] / [[frantar-alistarh-ist-austria]] — orthogonal direction; those labs do PTQ for existing FP16 models, BitNet trains-from-scratch for the regime PTQ can't reach.
- [[nvidia-quantization]] — partial overlap on MX formats but different teams; BitNet's stochastic-rounding for ternary work informs the broader sub-4-bit training literature.
- [[bitnet-models]] — the release-track companion file.
