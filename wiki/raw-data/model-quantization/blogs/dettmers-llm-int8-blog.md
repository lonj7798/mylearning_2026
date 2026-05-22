<!-- scope: Tim Dettmers' explanatory blog post on LLM.int8() and emergent outlier features
     deps: [[llm-int8]]
     see-also: [[smoothquant]], [[awq]], [[spqr]]
-->

# LLM.int8() and Emergent Features (Dettmers blog)
- **Core Insight:** Outlier features in transformers are not random noise but a coordinated, structurally-sparse mechanism the model uses for feature selection — they live in ≈6 hidden dimensions at the 6.7B emergence threshold, share consistent signs across layers, and grow in magnitude with perplexity (not pure parameter count).
- **Guideline:** Don't extrapolate findings between sub-6.7B and post-6.7B transformers; they're qualitatively different systems. When deploying INT8, always check whether the model crossed the phase shift — if yes, mixed-precision decomposition is mandatory.
- **Authors:** Tim Dettmers
- **Year:** 2022
- **URL:** https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/
- **Relevant topics:** outlier features, emergence, scaling phase transitions, quantization intuition

## Abstract
The blog companion to the LLM.int8() paper ([[llm-int8]]), written for practitioners. It explains *why* INT8 PTQ catastrophically fails above 6.7B parameters without the outlier-decomposition trick, by reframing the failure as an emergent phenomenon: a small number of feature dimensions develop very large activations, those dimensions coordinate across layers, and a single absmax INT8 scale per row gets dominated by them, leaving the other 99.9% of values represented by 1–2 quantization levels. The post also lays out the dual-stream interpretation — transformers learn one stream of explanatory features and another stream of *suppressive* features carried by large-magnitude outlier dimensions.

## Key Points

### What "emergent" means
"A gradual change in a property that suddenly undergoes a phase shift and then changes the quality of its substrate." Below 6.7B: outliers are probabilistic, layer-uncoordinated, and INT8 still works. Above 6.7B: outliers are deterministic, layer-coordinated, and INT8 collapses.

### The three quantitative findings
1. **Gradual onset**: even 125M-class models have occasional outlier features, but they are sparse and inconsistent across layers.
2. **Critical threshold ≈ 6.7B**: all layers suddenly agree on the same outlier dimensions; magnitudes jump.
3. **Rapid growth**: outlier peak magnitude grows ~15 (6B) → ~60 (13B) → ~95 (66B).

### Structural sparsity of outliers
At 6.7B, a sequence with ~150k outlier values uses only **6 distinct feature dimensions** of width ~12k. This is what makes the mixed-precision split practical: you don't need 1000 FP16 columns, you need 6.

### Sign coherence across layers
Outlier dimensions maintain consistent positive/negative signs across layers, suggesting downstream layers "know where" to apply the feature-suppression operation. Before the phase shift, signs disagree between layers; after, they align.

### Dual-stream interpretation
The blog proposes transformers run two parallel computations:
- One stream learns explanatory features (the bulk of dimensions).
- A second stream uses large-magnitude outlier dimensions to *remove* noisy/context-irrelevant features via subtractive interaction.
This is why outliers can't simply be clipped — they encode a routing decision that downstream layers depend on.

### Emergence tracks perplexity, not parameter count
Models with different architectures emerge at different parameter scales but at similar perplexity. Two implications: (i) emergence is a property of the learned function, not size per se, and (ii) better-trained smaller models can pre-emerge.

### Practical consequence
Quantization research papers benchmarking only on sub-6.7B models do not predict 7B+ behavior. Any production INT8 deployment of ≥7B models needs either:
- Mixed-precision decomposition ([[llm-int8]]).
- Migration into weights ([[smoothquant]], [[awq]]).
- Outlier-preserving sparse storage ([[spqr]], [[owq]], [[squeezellm]]).
- Rotation-based outlier removal ([[quip]], [[quarot]]).

## Connections
- Companion paper: [[llm-int8]].
- The downstream lineage that takes "outlier exists, isolate it" → "outlier exists, migrate it": [[smoothquant]], [[awq]].
- The downstream lineage that says "outlier exists, rotate it away": [[quip]], [[quip-sharp]], [[quarot]], [[spinquant]].
- Dettmers' next paper applying this to W4: [[qlora]].
