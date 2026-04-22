<!-- scope: equivalent reference note for weight initialization in the Lil'Log / educational-survey style
     see-also: [[weight-init]], [[batch-vs-layer-norm]]
-->

# Lilian Weng Equivalent Note: Weight Initialization
- **Core Insight:** Initialization matters because it sets early signal propagation, gradient scale, and optimization stability before normalization layers can rescue the training run.
- **Guideline:** Use variance-preserving initialization that matches the nonlinearity and architecture, then validate activation/gradient scales empirically during the first few hundred steps.
- **Author/Org:** Equivalent reference note in the style of Lilian Weng’s educational surveys
- **Year:** 2026 note based on standard initialization literature
- **URL:** https://lilianweng.github.io/
- **Relevant topics:** Xavier init, Kaiming init, residual scaling, transformer stability

## Summary
This repository originally planned for a dedicated Lilian Weng weight-initialization post, but Lil'Log does not appear to have a single canonical article devoted to that topic. This note fills the slot as an equivalent educational reference: the practical initialization ideas that matter for LLM training are variance preservation, residual-stream stability, and compatibility with normalization and learning-rate schedules.

## Key Points
- Xavier/Glorot is the default variance-preserving baseline for roughly symmetric activations.
- Kaiming/He init is better matched to ReLU-family nonlinearities.
- In transformers, residual scaling and normalization interact with init; “good enough” init can still fail if residual branches are too large.
- Modern stacks rely less on exotic initialization than on the combination of sane init, normalization, warmup, and clipping.

## Practical Details
- Check activation and gradient norms at startup.
- Match init to activation family and architectural depth.
- For transformers, pay attention to residual-path magnitude and output-head scale.
- If the model diverges in the first steps, inspect init before over-tuning the optimizer.

## Connections
- Complements [[weight-init]] as the classical technical reference.
- Relevant to [[gradient-clipping]], [[lr-schedules]], and [[batch-vs-layer-norm]] because these mechanisms stabilize the same early-training dynamics.

