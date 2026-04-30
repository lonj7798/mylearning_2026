---
chapter: ch-17
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md
source_url: https://karpathy.github.io/2019/04/25/recipe/
created_at: "2026-04-23"
---

# Excerpt: Karpathy's Recipe — the ablation discipline ch-17's memo structure enforces

**Source library:** `wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md`
**Blog post:** Andrej Karpathy 2019, "A Recipe for Training Neural Networks."

---

## Why Karpathy's recipe governs the memo format

Ch-17 could have been a "run these four stages and report a table" lab. It is not. The grade-bearing deliverable is `ablation-memo.md` with a predicted delta before each observed delta and a "one thing I was wrong about" paragraph. That structure is Karpathy's methodology applied to data curation.

From the source (lines 17–22):

> The framing: **"neural net training is a leaky abstraction"** — unlike standard SWE where libraries compose, ML components silently corrupt each other's assumptions.
>
> The six-step recipe below — arguably the most cited workflow document in practical ML.
>
> The concrete heuristic **"Adam 3e-4 is safe"** — still a correct default for small-model prototyping in 2025.
>
> **"Overfit a single batch"** as the mandatory pipeline sanity check before any serious experiment.

Two bullets drive the ch-17 memo:

**"Leaky abstraction."** A filter pipeline is worse than a training loop for silent failures. If your lang-ID model is loading the wrong version, every downstream score is subtly off. If your perplexity KenLM was trained on a weird Wikipedia dump, `head` bucket documents are not actually Wikipedia-like. These bugs do not throw. The only defence is the predict-before-measure habit the memo enforces: when observation disagrees with prediction by more than a calibrated amount, stop and look for the leak.

**"Overfit a single batch."** The ch-17 analogue is the Wikipedia-held-out perplexity sanity check in §2: score 100 held-out Wikipedia pages, median PPL should be 50–120. If it is 500, your KenLM is broken; do not run the full pipeline and pretend the filter is working. Ch-17 is exactly specifying Karpathy's overfit-one-batch rule for a non-training stage.

---

## The "be a scientist" discipline, literally

From the source (lines 32–36):

> ### 1. Become one with the data
> Spend *hours* looking at raw examples. Sort by every attribute you can think of. Find the duplicates, the corrupt records, the label-noise patterns. Most production-level ML wins come from data fixes, not model changes. Quote: "I look at thousands of examples, understand their distribution, and look for patterns."

The ch-17 decontamination step is an instance. Hand-labelling 100 dropped documents to measure n-gram FP rate is not a ceremony — it is Karpathy's "spend hours looking at raw examples" applied to a specific numerical question. If the student writes `FP_rate = 12%` without having looked at any of the 100 docs, the memo has failed regardless of what the number says.

From the source (lines 82–85):

> | "Init well." | Loss too high at step 0; bad convergence. |
> | "Visualize just before the net." | Mis-normalized inputs; label mismatches. |
> | "Generalize a special case." | Hard-to-debug loops; always write the `N=1` case first. |

"Visualize just before the net" maps directly to: inspect the first 50 surviving documents *after* your four-stage filter, *before* training. If they are all forum spam, your filter is broken and no eval will fix that. The memo's filter-by-filter paragraph structure forces you to log what each stage removed, which is the "visualize just before the net" rule applied to data curation rather than to training batches.

---

## The prediction discipline that earns partial credit

From the source (lines 63–66):

> ### 5. Tune
> - **Random search > grid search** in high-dim HP spaces.
> - Coarse to fine: wide-range random sweep → local refinement.

The deeper lesson behind random-over-grid is not about search strategy — it is about the epistemic cost of running experiments. Grid search optimistically assumes you can enumerate the thing worth varying. Random search admits you cannot. The ch-17 memo's "predicted delta / observed delta" structure admits the same: you cannot know the answer before running, but you can *commit* to a prediction and then learn from where you were wrong.

A paragraph that reads "I predicted lang-ID would gain +2 on HellaSwag; it gained +0.5; here is my hypothesis about what I misjudged" is a Karpathy-compliant paragraph. A paragraph that reads "lang-ID gained +0.5 on HellaSwag" is not. The difference is not stylistic — it is the difference between doing an experiment and reporting a number.

The lab's "one thing I was wrong about" paragraph is Karpathy's "review the 10 worst validation examples — they reveal systematic errors" (source line 72) applied to your own predictions. The student reviews the prediction they most misforecast and names a systematic mental-model error. If no paragraph names such an error, the student was not predicting honestly — which is itself an observation, and the memo should say so.
