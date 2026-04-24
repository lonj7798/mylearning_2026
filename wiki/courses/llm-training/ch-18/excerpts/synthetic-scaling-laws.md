---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "Synthetic-data scaling laws — SynthLLM (Microsoft 2025), Demystifying Synthetic Data (EMNLP 2025), BeyondWeb (2025)"
source_url: https://arxiv.org/abs/2503.19551
created_at: "2026-04-23"
---

# Excerpt: Scaling laws for synthetic data — the stage-6 empirics Lambert's framework predicts

**Authors / papers:** SynthLLM (Microsoft, arXiv 2503.19551); Demystifying Synthetic Data (EMNLP 2025, arXiv 2510.01631); BeyondWeb (arXiv 2508.10975)
**Year:** 2025
**Raw-data source:** [[raw-data/synthetic-data-scaling-laws]]

---

## Why this cluster of papers matters for ch-18

Stages 1-5 of the loop decide *what goes into the dataset*. Stage 6 — **mix** — decides how the dataset composes with real data at training time. For most of the synthetic-data literature, stage 6 was a guess: "use 50% synthetic, or 20%, or whatever seems to work." The 2025 scaling-laws cluster turns that guess into measured curves.

The raw-data file's core insight:

> "Synthetic pretraining data *can* follow rectified scaling laws — but the behavior depends sharply on how it's made: **rephrased** synthetic (WRAP-style) scales cleanly with no collapse signal up to observed scales, while **textbook-style pure-generated** synthetic shows model-collapse-predicted degradation; the optimal real:synth mix converges empirically to ~2:1 (≈30% rephrased synthetic) for best speedup at large data budgets."

This is the empirical ground-truth for stage-6 decisions, at least for pretraining. For ch-18, it is the first concrete number to attach to the stage-6 box.

---

## Key numbers and their operational meaning

From the raw-data file:

> "- **SynthLLM:** synthetic data (rephrase-style) obeys a rectified scaling law; returns plateau near 300B synthetic tokens at 8B model scale; 3B peaks around 4T tokens; 8B peaks around 1T tokens of synthetic.
> - **Demystifying (EMNLP 2025):** optimal real:synthetic ratio empirically ≈ 2:1 for rephrased synthetic, yielding **5-10x speedup** at fixed validation loss at large data budgets. Pure-generated textbook-style synthetic shows model-collapse patterns as predicted by [[model-collapse]]; rephrased does not.
> - **BeyondWeb:** detailed ablations at trillion-token scale — prompt-design axes, generator-model choice, and source-data composition each matter and combine non-linearly."

Unpack:

- **~30% rephrased synthetic is the sweet spot** at pretraining scale. Not 50%, not 10% — measured empirically.
- **5-10x speedup** at fixed validation loss, meaning the real + 30%-synthetic corpus matches a pure-real corpus using 5-10x less compute at the relevant scales.
- **Plateaus depend on model size:** 3B saturates at ~4T synth tokens, 8B at ~1T. **Smaller models absorb more synthetic before saturating** — counter to the intuition that bigger models benefit more.

Notice: the plateau-per-model-size result ties back to OMI-2's stage-1 saturation (~5M solutions for 8B at frontier teacher). Both are expressions of the same "the student absorbs what the teacher can teach" bound. The scaling-law papers make it quantitative; the stage-1 ablation of OMI-2 makes it modality-specific.

---

## Rephrased vs pure-generated — the stage-1-type axis

From the raw-data file:

> "**Not all synthetic is equal.** Rephrased synthetic ≠ pure-generated synthetic in scaling behavior.
> **Pure-generated textbook-style collapses** at sufficient scale — consistent with [[model-collapse]] predictions."

For ch-18, this is a crucial refinement of Raschka's stage-1 taxonomy. Rephrased (rewrite a real document) ≠ pure-generated (imagine a new textbook from scratch). They are both stage-1 operations, but they have different stage-6 scaling behaviour.

- **Rephrased** preserves the real-data manifold. Scales cleanly.
- **Pure-generated** lives on whatever manifold the teacher produces. Collapses at high fractions.

The operational consequence: if your stage 1 is rephrase-style, you can push the mix ratio up and get measured speedups. If your stage 1 is pure-generated textbook-style, high mix ratios reproduce model-collapse signatures and hurt you.

Notice: this is a stage-1-choice-determines-stage-6-ceiling claim. It is exactly the kind of porting observation the design-pattern lens is meant to highlight. Choosing stage 1 as rephrase vs full-generate is not a stylistic decision — it puts a hard ceiling on how much you can mix.

---

## The ceiling shape: rectified scaling

From the raw-data file:

> "Rectified scaling form (schematic): For rephrased synthetic + real mixture, validation loss L(N) ≈ `L(N) ≈ L_∞ + A · N^{-α}` up to a plateau `L_∞ + c(synth fraction)`, with `c(·)` increasing past ~50% synthetic for textbook-style, stable for rephrase-style."

Two terms: the usual power-law decay `A · N^{-α}`, plus a synth-fraction-dependent floor `c(synth fraction)`. The floor is the key addition. For rephrased synthetic, `c(·)` is roughly flat out to ~50% synthetic share; for textbook-style, it rises (the pipeline hurts you past a mixing threshold).

Notice: the floor is a stage-6 penalty, not a stage-1 penalty. Adding more synthetic *beyond* the optimal mix does not improve things; it lifts the asymptotic loss. This is what the scaling-laws papers quantify that earlier work ([[model-collapse]] theory) only asserted.

---

## Prompt design as a first-class axis

From the raw-data file:

> "**Prompt design is a first-class axis:** the same generator with different prompts yields measurably different scaling curves (BeyondWeb)."

Two pipelines with the same teacher and the same data volume can have different scaling behaviour because their stage-1 prompts are different. This is an empirical justification for treating prompt design as engineering, not art — and for ch-19, which is the dedicated chapter on generation methods.

For ch-18's purposes, the point is: when we say "stage 1 is where the money goes," we mean more than the inference cost. The *design* choices at stage 1 (rephrase vs generate, which prompt, which teacher, which decoding parameters) set the ceilings that stages 4, 5, and 6 operate under. Prompt design is the stage-1 hyperparameter most likely to be under-specified in papers.

---

## Practitioner takeaways, mapped to the loop

From the raw-data file:

> "- **Default to rephrase-style** if synthetic is your mass driver.
> - **30% synthetic fraction** is a sensible starting ratio; tune up or down based on scale.
> - **Measure, don't extrapolate** past the explored scale range.
> - **Don't mix pure-generated textbook synthetic at high fractions** without aggressive verification.
> - For mid-training (post-pretraining, pre-SFT), higher synthetic fractions may be safer because the pretraining base is already anchored."

Translate to loop-speak:

- **Stage 1 choice (rephrase vs generate) bounds stage 6 headroom.** Pick stage 1 knowing what stage 6 will allow.
- **Stage 6 ratio default: 30% for pretraining rephrase.** Not a universal rule — a starting point.
- **If stage 1 is pure-generated, invest more in stage 4 to earn higher stage-6 ratios.** This is the APIGen analogue for pretraining: aggressive verification buys higher synthetic share.
- **The base model pre-trained on real data is itself the anchor for mid-training / SFT / RL** — lower collapse risk downstream because upstream anchoring is already in place.

---

## Why this matters before the modality chapters

ch-22 covers synthetic pretraining (WRAP, Cosmopedia, textbooks) where these scaling laws apply directly. ch-27 covers mix ratios and scaling for synthetic data at SFT and preference stages. Both chapters will assume you understand that stage 1 and stage 6 are coupled — you cannot optimise stage 6 ratio without committing to a stage 1 method.

For ch-18: this coupling is the strongest single example of "the loop is the right unit of analysis, not the paper." Rephrased-vs-generated is a stage-1 distinction that predicts a stage-6 behaviour. If you read WRAP in isolation, you see a stage-1 paper; if you read the scaling-laws papers in isolation, you see a stage-6 paper; only the loop-lens shows they are the same object.

---

## Open problems Lambert and the scaling-laws cluster agree on

From the raw-data file:

> "- **Benchmarks used in scaling studies** may not reflect downstream task performance; measure task-specific curves.
> - **'Rephrased' is a method cluster,** not a single recipe — exact prompts / generator / chunk size all matter.
> - **Generator-model collapse** can still happen over time as generators are themselves trained on synthetic web text (contamination drift).
> - Results are still evolving; 2026 follow-ups may refine or contradict these."

Three caveats worth naming:

- **Benchmark ≠ downstream.** Validation loss and downstream task accuracy can diverge, especially for synthetic-trained models that may fit the benchmark distribution.
- **"Rephrased" is a cluster.** There is no single WRAP recipe; each specific implementation has its own curve.
- **Generator collapse over time.** The rephrase teacher itself is a model trained on the web; if the web becomes increasingly synthetic, the teacher's distribution drifts. This is the longest-running version of the stage-4 quality-control problem.

For ch-18, these caveats are why the loop-lens beats any specific recipe: the recipe will change, but the stages and their couplings will not.

---

## Generator-model choice: an under-appreciated axis

From the raw-data file:

> "Generator model choice matters more at higher synth fractions — weaker generators saturate faster."

Three implications for ch-18:

- **At low synth fraction (<10%), generator choice matters less.** The base-training signal from real data dominates, and synthetic is a minor contributor. You can get away with a weaker (cheaper) generator.
- **At moderate synth fraction (~30%, the sweet spot), generator choice starts to show up.** A weaker generator's distribution is amplified to a meaningful share of the training mix, and its limits become the trained model's limits.
- **At high synth fraction (>50%), generator choice is the dominant axis.** The trained model's ceiling is the generator's ceiling.

Connect this to the OMI-2 teacher-strength result: "Llama-3.1-405B at 1M samples beats Mixtral at 10M." Same empirical pattern: when you rely heavily on synthetic, the teacher defines the ceiling. OMI-2 sees this at the reasoning-trace level; the pretraining scaling-laws cluster sees it at the web-rephrase level.

For the ch-18 loop: **stage-1 teacher choice is a stage-6 determinant**. You cannot independently pick the teacher and the mix ratio; a weaker teacher forces you to use a lower synthetic share to avoid their distribution's bias dominating. These are coupled decisions.

## Connections

- [[excerpts/nathan-lambert-synth]] — Lambert's "accumulation + verification" rule is the qualitative counterpart to these quantitative scaling laws.
- [[excerpts/openmathinstruct-2]] — OMI-2's ~5M solution saturation is a stage-1 scaling datapoint; the pretraining cluster here adds the stage-6 story.
- [[excerpts/self-instruct]] — Self-Instruct is bootstrap, not rephrase; these scaling laws suggest bootstrap-style pretraining would face the pure-generated ceiling, and that matches the empirical Phi-textbook caveats.
- [[ch-18]] — parent. The scaling-law numbers are ch-18's stage-6 anchors and the empirical ground for the "rephrased vs generated" nuance.
