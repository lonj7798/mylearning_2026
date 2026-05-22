<!-- scope: Qualcomm AI Research — Markus Nagel et al.; pre-LLM-era PTQ playbook (AdaRound, BRECQ, DFQ), production NPU stack
     deps: [[adaround]], [[brecq]], [[data-free-quantization]]
     see-also: [[qualcomm-quantization-whitepaper]]
-->

# Qualcomm AI Research — Pre-LLM PTQ Playbook + NPU Production
- **Core Insight:** Qualcomm's pre-LLM PTQ work supplied the calibration, rounding, equalization, and reconstruction playbook that modern LLM PTQ methods still inherit.
- **Guideline:** Read this lab track before GPTQ/AWQ if the learner needs the AdaRound, BRECQ, DFQ, and AIMET lineage behind today's LLM quantization.
- **Authors:** Qualcomm AI Research, Markus Nagel, Tijmen Blankevoort, Mart van Baalen, Yelysei Bondarenko, and collaborators
- **Year:** 2019–2026
- **URL:** https://www.qualcomm.com/research/artificial-intelligence/ai-research ; https://github.com/quic/aimet
- **Relevant topics:** AdaRound, BRECQ, data-free quantization, PTQ taxonomy, AIMET, Snapdragon NPU

## Summary
Qualcomm AI Research, anchored by Markus Nagel, Tijmen Blankevoort, Mart van Baalen, and Yelysei Bondarenko, **defined the pre-LLM-era PTQ playbook** that every modern LLM quant method inherits. The canonical "Qualcomm whitepaper" ([[qualcomm-quantization-whitepaper]]) and the [[adaround]] / [[brecq]] / [[data-free-quantization]] papers established the per-channel / per-tensor / symmetric / asymmetric taxonomy, the block-wise reconstruction objective, the cross-layer equalization trick, and the bias-correction recipe — all before the GPT-3 era. In the LLM era the same group has continued with **GPTVQ** (vector PTQ with GPTQ-style updates) and ongoing work on edge-device quant for Snapdragon NPUs.

## Notable Works
- [[data-free-quantization]] (Nagel 2019 DFQ) — weight equalization + bias correction without data; the foundation of "you don't always need calibration."
- [[adaround]] (Nagel 2020) — per-weight learned rounding direction via rectified sigmoid; closed-form via the Hessian objective; the direct ancestor of GPTQ.
- [[brecq]] (Li 2021) — block-wise reconstruction PTQ; the cross-layer dependency framework.
- [[qualcomm-quantization-whitepaper]] — the canonical practitioner PTQ guide.
- GPTVQ (van Baalen 2024) — vector PTQ with GPTQ-style updates; carries the Qualcomm PTQ lineage into the LLM era.
- Outlier mitigation papers (Bondarenko et al.) — early formal study of why transformers develop outliers; informs SmoothQuant / AWQ.

## Recurring themes
- **PTQ-first, calibration-light**: the Qualcomm DNA is "minimize PTQ data dependence"; this comes from edge / on-device deployment where calibration corpora are awkward.
- **Mathematical formulation, then production playbook**: Nagel papers consistently provide both a formal optimization problem and a practitioner checklist (the whitepaper is the canonical example).
- **Edge / NPU as the real target**: even when the papers benchmark on ImageNet / GLUE, the deployment target is always low-power inference on Snapdragon.

## Open Resources
- Qualcomm AI Research publications: https://www.qualcomm.com/research/artificial-intelligence/ai-research
- Qualcomm whitepaper (linked from many community PTQ guides; arxiv version at https://arxiv.org/abs/2106.08295 "A White Paper on Neural Network Quantization")
- AIMET (Qualcomm's open-source quant toolkit): https://github.com/quic/aimet

## Connections
- [[frantar-alistarh-ist-austria]] — direct successor; GPTQ is AdaRound-at-LLM-scale. Frantar acknowledges the lineage explicitly in the GPTQ paper.
- [[dettmers-group]] / [[han-song-mit]] — adjacent LLM-era PTQ labs that build on the Qualcomm taxonomy.
- [[nvidia-quantization]] — different runtime target (GPU vs NPU) but shared format vocabulary (per-tensor / per-channel / sym / asym).
- [[intel-quantization]] — adjacent on edge-device PTQ; competing toolkits (Intel Neural Compressor vs Qualcomm AIMET).
