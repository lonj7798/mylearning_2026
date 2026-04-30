---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/synthetic-data-scaling-laws.md
source_url: https://arxiv.org/abs/2503.19551 ; https://arxiv.org/abs/2510.01631 ; https://arxiv.org/abs/2508.10975
created_at: "2026-04-23"
---

# Excerpt: SynthLLM + Demystifying Synthetic Data + BeyondWeb — rephrased ≠ pure-generated under scaling

**Source library:** `wiki/raw-data/llm-training/papers/synthetic-data-scaling-laws.md`
**Papers:** SynthLLM (arXiv 2503.19551, Microsoft 2025); Demystifying Synthetic Data (EMNLP 2025, arXiv 2510.01631); BeyondWeb (arXiv 2508.10975, 2025)

---

## Why this source anchors ch-23

Ch-23 §3 lists "rephrased synthetic survives, pure-generated does not" as one of the structural mitigability boundaries. This excerpt is the empirical backbone for that claim. The 2025 scaling-law cluster splits synthetic into two behaviorally distinct classes and measures them against real-data baselines at trillion-token scale. The key number — **30% synthetic share as the rephrased-optimum** — appears in ch-23 §5's comparison table and §6's design guidance.

---

## The split: rephrased vs pure-generated

```
# synthetic-data-scaling-laws.md, lines 7-8, 20
Synthetic pretraining data can follow rectified scaling laws — but
behavior depends sharply on how it's made: rephrased synthetic
(WRAP-style) scales cleanly with no collapse signal up to observed
scales, while textbook-style pure-generated synthetic shows
model-collapse-predicted degradation.
```

The paper cluster is explicit: **"synthetic" is not a monolith**. Two regimes behave very differently under scaling:

- **Rephrased synthetic.** Each output is anchored to a specific real document (or passage) and rewritten — paraphrased, restyled, expanded with explanations, etc. The real anchor constrains the distribution; the output is a *transformation* of real, not a free generation from the model.
- **Pure-generated / textbook-style synthetic.** Each output is a free generation from the model given a topic prompt. No anchor; the distribution is whatever the generator emits.

Shumailov's collapse mechanism ([[excerpts/model-collapse]]) predicts that pure-generated collapses because the iteration is recursive sample-and-refit. It does *not* predict rephrased collapse because rephrased is a single-pass transformation — the distribution is pinned to real documents, not to a model's previous generation.

Demystifying Synthetic Data (EMNLP 2025) is the empirical confirmation. Rephrased synthetic obeys clean rectified scaling up to the measured scales (~300B tokens at 8B model size); pure-generated textbook synthetic departs from the scaling law and shows the Shumailov-predicted degradation pattern at sufficient fraction.

---

## The 30% share — empirical optimum for rephrased

```
# synthetic-data-scaling-laws.md, lines 7-8, 19
The optimal real:synth mix converges empirically to ~2:1
(≈30% rephrased synthetic) for best speedup at large data budgets.
```

Two ways to read this number:

- **As a recipe:** if you're building a pretraining mix with rephrased synthetic, 30% synthetic share is the first-order starting point. Tune up if your corpus is rich in rephrasable documents; tune down if real data is more abundant than compute.
- **As theory confirmation:** the optimum being *interior* (not 0%, not 100%) is what He et al. 2025 and Garg et al. 2025 predict analytically. Rephrased synthetic adds diversity on top of real without introducing [[excerpts/strong-model-collapse]]'s bias floor (because the anchor constraint keeps `σ_synth²` small). The optimum balances the diversity win against the (small) bias cost.

Empirically reported gains at 30% rephrased:

- 5–10× speedup at fixed validation loss at large data budgets (>100B tokens).
- At 8B model scale, synthetic contribution saturates near 1T synthetic tokens.
- At 3B scale, saturation near 4T synthetic tokens (smaller models can absorb more synthetic before saturating — likely because their bias floor is higher anyway and synthetic noise is relatively smaller).

The 5–10× speedup is the single biggest practical result in the 2025 synthetic-pretraining literature. It is why the field is willing to tolerate synthetic despite the collapse literature — because *rephrased* synthetic pays.

---

## Rectified scaling form

```
# synthetic-data-scaling-laws.md, lines 31-34
For rephrased synthetic + real mixture, validation loss L(N) ≈
  L(N) ≈ L_∞ + A · N^{-α}
up to a plateau L_∞ + c(synth fraction),
with c(·) increasing past ~50% synthetic for textbook-style,
stable for rephrase-style.
```

The formula is the 2025 empirical version of a rectified scaling law. Key pieces:

- `L_∞` — the asymptotic best-case loss under the fixed compute budget.
- `A · N^{-α}` — the standard power-law decay with data.
- `c(synth fraction)` — the offset-from-asymptote induced by synthetic contamination.

The critical shape: `c(·)` is **flat** for rephrased-style synthetic up to ~30–50% share, then starts rising. For textbook-style, `c(·)` rises monotonically even at small fractions. This is the functional form of the split — the rephrased optimum is "stay on the flat region"; the textbook-style non-optimum is "there is no flat region."

Connects to [[excerpts/strong-model-collapse]]'s `c(p) · σ²`: rephrased synthetic has small `σ_synth²` (anchor constraint), so `c(p) · σ²` is small even at moderate `p`; textbook-style has larger `σ_synth²` and the product is meaningfully positive even at small `p`.

---

## BeyondWeb — the prompt-design result

```
# synthetic-data-scaling-laws.md, line 29
Prompt design is a first-class axis: the same generator with different
prompts yields measurably different scaling curves (BeyondWeb).
```

BeyondWeb's contribution is that **how you prompt the rephraser matters at scaling-law level**. Three axes the paper ablates:

1. **Prompt specificity.** Generic "rephrase this text" vs targeted "rephrase this text as a pedagogical explanation." Targeted prompts give steeper scaling curves.
2. **Generator model choice.** Stronger generators help at higher synthetic fractions (weaker generators saturate faster).
3. **Source-data composition.** Rephrasing a web-text-rich corpus vs a textbook-rich corpus yields different downstream curves.

These axes combine non-linearly. The operational implication for ch-23: the "gate" concept in §6 needs to extend to prompt design. A rephraser with a weak prompt is a weak verifier — the anchor constraint holds but the distribution transformation is uncontrolled.

---

## Why rephrased is implicitly "verified"

Ch-23 §3 lists rephrased synthetic under mitigable regimes because the anchor constraint functions as a gate:

- Each output must be a paraphrase of a specific real passage.
- Paraphrase relationship can be *checked* — NLI entailment from rephrased to original, or bidirectional entailment.
- Unsupported claims (hallucinations added by the rephraser) can be detected and filtered.

This makes rephrased synthetic an instance of ch-23 §4 Axis 2 (external verification): the real anchor is the "ground truth" against which the paraphrase is checked. The gate is cheap (NLI or embedding similarity) and 100% external. This is why rephrased survives scaling while pure-generated does not.

The corollary: pure-generated synthetic can be *converted* to mitigable by adding an external check. APIGen-60K ([[excerpts/apigen]]) does this for function-calling: pure-generated candidates, but every one passes a 3-layer verifier before acceptance. The resulting corpus behaves like rephrased in the scaling-law sense — no collapse, clean scaling — because the verifier has collapsed the `σ_synth²` term.

---

## The generator-contamination drift (open problem)

```
# synthetic-data-scaling-laws.md, line 51
Generator-model collapse can still happen over time as generators are
themselves trained on synthetic web text (contamination drift).
```

The unspoken problem: as the open web accumulates LLM-generated content, the *generators* used to make rephrased synthetic are themselves trained on contaminated pretraining. This means `σ_synth²` slowly drifts even for rephrased synthetic. The 2025 numbers assume generators are clean; by 2027 they may not be.

Mitigations (not in the source; extrapolated):

- Use generators trained before the web contaminated heavily (pre-2023 checkpoints).
- Verify generators against high-quality human-labeled evals before deploying them as rephrasers.
- Monitor `c(p) · σ²` over time; if it starts rising at constant `p`, your generator has drifted.

This is a 2026-forward concern. Ch-23 flags it but does not resolve it.

---

## Practitioner takeaways (direct lift)

```
# synthetic-data-scaling-laws.md, lines 41-46
Default to rephrase-style if synthetic is your mass driver.
30% synthetic fraction is a sensible starting ratio.
Measure, don't extrapolate past the explored scale range.
Don't mix pure-generated textbook synthetic at high fractions
without aggressive verification.
For mid-training, higher synthetic fractions may be safer.
```

Each of these maps to a ch-23 principle:

- "Default to rephrase-style" → §3 mitigable list; §6 gate template.
- "30% synthetic fraction" → §5 comparison table rephrased row.
- "Measure, don't extrapolate" → §7 dashboard (all metrics require measurement, not extrapolation).
- "Don't mix pure-generated textbook at high fractions without verification" → §2's `c(p)·σ²` argument.
- "Mid-training is safer" → §6 template discussion; the pretraining base anchors subsequent stages.

---

## Connections

- [[excerpts/model-collapse]] — the collapse mechanism this paper cluster confirms for pure-generated, disconfirms for rephrased.
- [[excerpts/strong-model-collapse]] — the theoretical backbone explaining why the split exists (`σ_synth²` depends on method).
- [[excerpts/faithful-synth-eval]] — the verification toolkit that turns pure-generated into rephrased-equivalent.
- [[excerpts/apigen]] — a pure-generated pipeline made scaling-safe via 3-layer verification.
- [[ch-23]] — §3 mitigability, §5 comparison, §6 gate designs all reference this cluster.
