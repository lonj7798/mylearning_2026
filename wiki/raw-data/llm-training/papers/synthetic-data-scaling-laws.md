<!-- scope: 2025 synthetic-data scaling laws — SynthLLM + Demystifying Synthetic Data + BeyondWeb
     deps: [[rephrasing-the-web]]
     see-also: [[model-collapse]], [[strong-model-collapse]], [[hf-cosmopedia]]
-->

# Synthetic Data Scaling Laws (2025 lines — SynthLLM, Demystifying Synthetic Data, BeyondWeb)
- **Core Insight:** Synthetic pretraining data *can* follow rectified scaling laws — but the behavior depends sharply on how it's made: **rephrased** synthetic (WRAP-style) scales cleanly with no collapse signal up to observed scales, while **textbook-style pure-generated** synthetic shows model-collapse-predicted degradation; the optimal real:synth mix converges empirically to ~2:1 (≈30% rephrased synthetic) for best speedup at large data budgets.
- **Guideline:** For pretraining with synthetic data, prefer rephrased over pure-generated; target ~30% synthetic share; expect 5-10× speedup at comparable validation loss; don't extrapolate scaling past ~1T tokens for 8B models without measuring directly.
- **Authors / papers (representative 2025):**
  - **SynthLLM** — "Scaling Laws of Synthetic Data for Language Models" (Microsoft, arxiv 2503.19551).
  - **Demystifying Synthetic Data** — EMNLP 2025 (arxiv 2510.01631).
  - **BeyondWeb** — "Lessons from Scaling Synthetic Data for Trillion-scale Pretraining" (arxiv 2508.10975).
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2503.19551 ; https://arxiv.org/abs/2510.01631 ; https://arxiv.org/abs/2508.10975
- **Relevant topics:** synthetic data scaling, rephrased vs pure-generated, real:synth ratio, rectified scaling

## Abstract (aggregate)
A 2025 cluster of papers systematically measures how synthetic pretraining data behaves under scaling. Key findings:
- **SynthLLM:** synthetic data (rephrase-style) obeys a rectified scaling law; returns plateau near 300B synthetic tokens at 8B model scale; 3B peaks around 4T tokens; 8B peaks around 1T tokens of synthetic.
- **Demystifying (EMNLP 2025):** optimal real:synthetic ratio empirically ≈ 2:1 for rephrased synthetic, yielding **5–10× speedup** at fixed validation loss at large data budgets. Pure-generated textbook-style synthetic shows model-collapse patterns as predicted by [[model-collapse]]; rephrased does not.
- **BeyondWeb:** detailed ablations at trillion-token scale — prompt-design axes, generator-model choice, and source-data composition each matter and combine non-linearly.

## Key findings (aggregated)
1. **Not all synthetic is equal.** Rephrased synthetic ≠ pure-generated synthetic in scaling behavior.
2. **Optimal mix ratio converges empirically** to ~30% rephrased synthetic for rephrase-style corpora.
3. **Clean rectified scaling law** holds for rephrased synthetic up to measured scales; plateau heights depend on model size.
4. **Pure-generated textbook-style collapses** at sufficient scale — consistent with [[model-collapse]] predictions.
5. **Generator model choice matters more at higher synth fractions** — weaker generators saturate faster.
6. **Prompt design is a first-class axis:** the same generator with different prompts yields measurably different scaling curves (BeyondWeb).

## Rectified scaling form (schematic)
For rephrased synthetic `+` real mixture, validation loss `L(N)` ≈
`L(N) ≈ L_∞ + A · N^{-α}` up to a plateau `L_∞ + c(synth fraction)`,
with `c(·)` increasing past ~50% synthetic for textbook-style, stable for rephrase-style.

## Training outcomes (from papers)
- Real + 30% rephrased synthetic: 5-10× speedup at large (>100B) data budgets.
- 8B model on rephrased synthetic: plateau near 1T synthetic tokens.
- 3B model: plateau near 4T synthetic tokens (smaller models can absorb more synthetic before saturating).

## Practitioner takeaways
- **Default to rephrase-style** if synthetic is your mass driver.
- **30% synthetic fraction** is a sensible starting ratio; tune up or down based on scale.
- **Measure, don't extrapolate** past the explored scale range.
- **Don't mix pure-generated textbook synthetic at high fractions** without aggressive verification.
- For mid-training (post-pretraining, pre-SFT), higher synthetic fractions may be safer because the pretraining base is already anchored.

## Risks + gotchas
- **Benchmarks used in scaling studies** may not reflect downstream task performance; measure task-specific curves.
- **"Rephrased" is a method cluster,** not a single recipe — exact prompts / generator / chunk size all matter.
- **Generator-model collapse** can still happen over time as generators are themselves trained on synthetic web text (contamination drift).
- Results are still evolving; 2026 follow-ups may refine or contradict these.

## Connections
- Empirical complement to [[model-collapse]] / [[strong-model-collapse]] theory.
- Quantifies the scaling behavior promised by [[rephrasing-the-web]] (WRAP).
- Supports [[hf-cosmopedia]] / [[phi-textbooks]]-style recipes but differentiates them sharply.
- 2025 key references for any synthetic-pretraining chapter.
