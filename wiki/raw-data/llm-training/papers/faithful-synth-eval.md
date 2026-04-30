<!-- scope: detecting distribution preservation/corruption in synthetic data (2024/2025 line)
     deps: [[model-collapse]]
     see-also: [[strong-model-collapse]], [[prismatic-synthesis]], [[genie]]
-->

# Faithful Synthetic-Data Evaluation: Detecting Distribution Preservation vs Corruption
- **Core Insight:** Averaged loss / perplexity *hide* tail degradation in synthetic corpora — to detect whether synthetic data preserves the target distribution you need tail-mass measurements (rare-event recall, per-cluster density, gradient-space coverage) and external-verifier pipelines; several 2024/25 papers make this explicit.
- **Guideline:** When auditing a synthetic corpus before training, measure: (1) rare-token / rare-ngram recall vs real reference, (2) embedding-cluster occupancy, (3) (optionally) gradient-space coverage (G-Vendi), (4) external-verifier flag rate; reject corpora that drop in any of these even if average PPL looks fine.
- **Authors / papers:**
  - "Escaping Model Collapse via Synthetic Data Verification" (Zhang et al. 2025, arxiv 2510.16657) — external verifier for convergence guarantees.
  - "A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective" (2025, arxiv 2509.16499).
  - "Collapse or Thrive?" (2025 openreview Xr5iINA3zU) — empirical tail behavior.
  - Related: [[prismatic-synthesis]] (gradient-space diversity as an upstream quality metric).
- **Year:** 2024–2025
- **URL:** https://arxiv.org/abs/2510.16657 ; https://arxiv.org/html/2509.16499v1
- **Relevant topics:** synthetic data audit, distribution preservation, tail recall, external verification

## Abstract (aggregate)
This cluster of 2024–25 papers formalizes the problem of **distinguishing "safe" from "corrupting" synthetic data** before training on it. Core move: don't trust aggregate perplexity / loss; measure *tail preservation* directly. Several papers converge on three complementary strategies:
1. **Tail recall metrics** — fraction of rare real-ngrams / rare concepts reproducible from the synthetic corpus.
2. **External verification** — a stronger model or rule-based verifier filters out low-quality synthetic (e.g., answer-match for math, NLI for RAG).
3. **Gradient / embedding coverage** — G-Vendi (see [[prismatic-synthesis]]), embedding-cluster occupancy, or kNN diversity.

"Escaping Model Collapse via Synthetic Data Verification" (2025) gives both analytical convergence guarantees and empirical evidence that external verification breaks the collapse loop even under repeated training. "Closer Look at Model Collapse" (2025) reframes collapse in terms of **generalization → memorization drift** as synthetic fraction grows.

## Key evaluation axes (consolidated)

### 1. Tail-mass measurements
- **Rare-token recall:** does the synthetic corpus produce real-reference's rare tokens at comparable frequency?
- **Rare n-gram overlap:** same for multi-token patterns.
- **Rare-concept recall:** LLM-tagged categorical entities (named entities, terminologies) — are long-tail ones preserved?

### 2. External verification
- **Task-specific verifier:** math → answer matcher; code → unit tests; NLI → entailment classifier (à la [[genie]]); factual → retrieval-grounded checker.
- **Strong-judge filter:** a model substantially stronger than the synthesizer audits outputs.
- **Provides the convergence-guaranteeing filter** Zhang et al. (2025) prove keeps iterated training bounded.

### 3. Coverage / diversity metrics
- **G-Vendi** (see [[prismatic-synthesis]]): entropy of gradient-density matrix.
- **Embedding-cluster occupancy:** count distinct embedding clusters inhabited.
- **kNN diversity:** average kNN distance in embedding space.

### 4. Drift-over-iteration signals (if iterative training)
- Monitor delta on (1–3) across training rounds — early warning of collapse.

## What the 2025 papers add specifically
- **Convergence guarantees under verification:** Zhang et al. show that with a reliable external verifier, iterated synthetic training converges (no collapse) in analytical regression settings, and empirically in LLM text generation.
- **Memorization↔generalization tradeoff:** "Closer Look at Model Collapse" identifies that increasing synthetic fractions shift models toward memorization-heavy regimes, which surface-metrics don't catch.
- **Mixture-ratio optima:** derive analytic optimal real:synthetic ratios (He et al. 2025, Garg et al. 2025) as a function of data quality.

## Practitioner takeaways
- **Never audit a synthetic corpus by average PPL alone.** Always include tail + diversity metrics.
- **Build an external verifier into the pipeline** before training, not after.
- **Accumulate, don't replace** real data — the single most robust mitigation.
- **Audit per-cluster** — mode collapse often appears in specific topic clusters before showing up globally.

## Risks + gotchas
- **Verifier quality ceiling:** the verifier itself may have bias or blind spots; compound-verifier (multi-axis) reduces this.
- **Tail metrics are statistics of rare events** — noisy for small corpora; requires large samples.
- **G-Vendi is proxy-dependent** — changing proxy model changes rankings.
- **Research is moving fast** — 2026 may introduce better primitives.

## Connections
- Theoretical backbone: [[model-collapse]] and [[strong-model-collapse]].
- Upstream gradient-coverage approach: [[prismatic-synthesis]].
- Content-grounded faithfulness filter: [[genie]].
- Key auditing primitive for every other §2b target — implicitly or explicitly.
