---
chapter: ch-07
course: model-quantization
phase: read
excerpt_of: "LLM.int8() and Emergent Features (Dettmers blog, Aug 2022)"
source_url: https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/
created_at: "2026-05-21"
---

# Excerpt: Dettmers blog — outliers as emergent feature selection

**Author:** Tim Dettmers
**Year:** August 2022
**Raw-data source:** [[raw-data/blogs/dettmers-llm-int8-blog]]

---

## The reframing

The paper ([[llm-int8]]) reports outliers as an empirical fact. The blog reframes them: outliers are **not noise** but a coordinated, structurally-sparse mechanism the model uses for feature selection.

> "A gradual change in a property that suddenly undergoes a phase shift and then changes the quality of its substrate."

Below 6.7B: outliers are probabilistic, layer-uncoordinated, INT8 still works. Above 6.7B: outliers are deterministic, layer-coordinated, INT8 collapses.

---

## The three quantitative findings (as a practitioner)

1. **Gradual onset**: even 125M-class models have occasional outlier features, but they are sparse and inconsistent across layers.
2. **Critical threshold ≈ 6.7B**: all layers suddenly agree on the same outlier dimensions; magnitudes jump.
3. **Rapid growth**: outlier peak magnitude grows ~15 (6B) → ~60 (13B) → ~95 (66B).

---

## The dual-stream interpretation

The blog's load-bearing causal hypothesis:

> Transformers run two parallel computations:
> - One stream learns explanatory features (the bulk of dimensions, normal magnitudes).
> - A second stream uses large-magnitude outlier dimensions to *remove* noisy/context-irrelevant features via subtractive interaction.

This explains:

- **Why clipping outliers destroys the model.** Without the subtraction signal, the model has un-filtered features. Empirically, clipping at 6.0 on a 13B model drops zero-shot accuracy by ~10%.
- **Why outliers are sign-coherent across layers.** The downstream layer "knows where" to apply the subtraction.
- **Why outliers concentrate in ~6 specific dims.** It's a small fixed-cost mechanism; the model only needs a handful of feature-suppression channels.

---

## Emergence tracks perplexity, not parameter count

> Models with different architectures emerge at different parameter scales but at similar perplexity.

Two implications:

1. **Emergence is a property of the learned function, not size per se.**
2. **Better-trained smaller models can pre-emerge.** A heavily-trained 3B can exhibit outlier phase transition.

→ Parameter count is a heuristic, not a guarantee. **Measure per-channel activation amplitudes on your specific model** before deciding which PTQ recipe to use.

---

## Practical consequence (the field's directive)

Any production INT8 deployment of ≥7B models needs one of:

| Strategy | Reference |
|---|---|
| Isolate outliers in FP16 | [[llm-int8]] (ch-07) |
| Migrate outliers into weights | [[smoothquant]], [[awq]] (ch-09) |
| Preserve outlier weights in FP16 | [[spqr]], [[owq]], [[squeezellm]] (ch-11) |
| Rotate outliers away | [[quip]] (ch-13), [[quarot]] (ch-14) |

Research papers benchmarking only on sub-6.7B models **do not predict 7B+ behavior**. This is the most important practical lesson.

---

## The "phase shift" framing as research signal

Dettmers' blog reads as a manifesto for the entire 2022-2024 LLM quantization research line. Every subsequent method either accepts the phase-shift framing (and tries a different fix for the same problem) or implicitly rejects it (e.g. by assuming smoothness of activation distributions). The first camp won — every successful method post-2022 explicitly addresses the outlier structure.

---

## What's NOT in the blog (but should be in your mental model)

- **The 6-dimensions number is for OPT-6.7B specifically.** Llama-2-7B has a slightly different distribution (~12 dims); Llama-2-70B has ~80 dims. Always measure your model.
- **Activation outliers ↔ weight outliers correspondence.** Activation outliers are concentrated in specific *input* channels; the corresponding weight rows are not unusually large. SmoothQuant exploits this asymmetry.
- **The post-Llama-3 era.** Llama-3 models trained with deliberate per-channel activation clipping at train time show *much smaller* outlier magnitudes — possibly because the training team read this blog and instrumented for it.

---

## Connections

- [[excerpts/llm-int8]] — the paper this blog accompanies.
- [[ch-07]] — parent synthesis.
- [[ch-09]] — [[smoothquant]] and [[awq]] are the migration-based responses.
- [[ch-13]] — [[quip]] is the rotation-based response.
