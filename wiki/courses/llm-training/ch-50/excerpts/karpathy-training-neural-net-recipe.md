---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md
source_url: https://karpathy.github.io/2019/04/25/recipe/
created_at: "2026-04-23"
---

# Excerpt: Karpathy's Recipe — "review the 10 worst" as the root of failure bucketing

**Source library:** `wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md`
**Artifact:** Operational methodology for training neural networks; prose essay with six-step recipe and enumerated silent-failure patterns.

---

## Why this source grounds ch-50 methodologically

Every slice-report discipline this chapter teaches has its operational root in Karpathy's recipe. "Neural net training fails silently" is the reason per-slice evaluation exists at all — aggregate loss can drop while a specific capability silently regresses. "Review the 10 worst validation examples" is the literal procedure ch-50 §3 names as "cluster-by-reason." "Predict outcome before run" is how ch-50 §4's signed-delta thresholds become honest. The chapter is, in a real sense, Karpathy applied to post-training eval.

---

## "Neural net training fails silently" — the reason per-slice exists

Source §Abstract:

> Training fails silently (a bug shows up as "the model doesn't work," not as an error message), so practitioners must adopt an incremental, hypothesis-driven workflow.

Silent failure is the direct justification for slicing. If the training loop exploded, you would know; since it does not, the only evidence that a capability broke is a per-slice metric moving in the wrong direction. Aggregate loss / aggregate eval score is the coarsest possible observation; it corresponds to the coarsest possible debugging. Ch-50's guideline "always compute per-slice first and aggregate second" is simply the silent-failure doctrine applied at eval time.

---

## "Review the 10 worst" — the pipeline in §3

Source §Technical Details — 6. Squeeze out the juice:

> Review the 10 worst validation examples — they reveal systematic errors.

This is one sentence, and it is the entire origin of failure bucketing. The procedure:

1. Sort validation examples by score (ascending).
2. Read the bottom 10 (or 30, or 50).
3. Tag each with the reason it failed.
4. Count tags.
5. The most common tag is the biggest fixable bucket.

Ch-50 §3's `bucket_failures()` pseudocode is an automated version of this. The human-in-the-loop step that makes it honest is Karpathy's: you *read* the failures, you do not hope an aggregate number told you what is wrong. Any team that skips the reading step ends up with a bucket ontology that matches the grader's biases, not the model's failures.

---

## "Predict outcome before run" — where signed-delta thresholds come from

Source §Famous Heuristics and Pitfalls:

> "Generalize a special case." — Hard-to-debug loops; always write the `N=1` case first.
> "Monitor and clip gradient norms." — Silent divergence.

And from the six-step recipe's tuning stage:

> Tune LR schedule last — after architecture is fixed.

The broader principle Karpathy applies (and ch-46 already inherits) is *predict-before-run*: write down your expected per-slice direction before evaluating. Ch-50 §4's effect-size threshold is the formalization — an un-predicted gain is not automatically a win; it must pass the threshold your predictions pre-registered.

If you predict `MMLU-STEM +2 pp` and measure `MMLU-STEM +2.6 pp`, that is a confirmed directional win, worth more than an unpredicted `MMLU-STEM +2.6 pp`. The latter might be noise lining up by chance; the former is evidence your understanding of the training intervention was correct.

---

## "Be paranoid about `model.train()` vs `model.eval()`" — a slice invariant

Source §Famous Heuristics and Pitfalls:

> "Be paranoid about `model.train()` vs `model.eval()`." — BN/dropout active at eval.

The equivalent post-training failure is *rollout tokenizer vs train tokenizer drift* (OpenRLHF § vllm_kl; Llama 3 post-training § format drift). The symptom is a slice-specific regression that makes no sense given the training data — e.g., format-following drops while content quality rises. Without per-slice eval, the drift is invisible. With per-slice eval, the drift shows as a `format-violation` bucket growing while all other buckets shrink; the debug path is "check tokenization between rollout and train," not "train more on format."

Ch-50's ledger is the institutional memory that surfaces this — `format-violation` growing while aggregate is flat is the pattern to pattern-match on in the next run.

---

## "Adam 3e-4 is safe" — the default baseline before per-slice chasing

Source §Key Contributions:

> The concrete heuristic **"Adam 3e-4 is safe"** — still a correct default for small-model prototyping in 2025.

Per-slice reports tempt a team into over-tuning the training config to maximize a specific slice. Karpathy's "safe default" is the counterweight: before tuning for a per-slice win, verify the recipe is above the default baseline on aggregate. Ch-50 §6's three-line-chart view has this guardrail role — it is the "did the overall recipe still work?" check that prevents a 50-slice report from approving a micro-tuned, over-fit checkpoint.

---

## "Don't be a hero" — against bespoke per-slice fixes

Source §Key Contributions:

> The principle **"don't be a hero"** — use the simplest known-good architecture; inventing novelty before baseline is a near-universal time-waster.

The eval-side analogue: use the simplest known-good slicing first. [[longbench]]'s 6 categories, [[ruler]]'s 13 tasks, [[bfcl]]'s 7 categories, MMLU-by-4-domain are all off-the-shelf. Invent a novel slicing only after the off-the-shelf sets show you need one. A team that invents a slicing before measuring the standard ones typically ends up with a slice set that flatters their model — exactly Karpathy's warning against hero novelty.

---

## Six-step recipe applied to eval

Source §Technical Details — The Six-Step Recipe:

> 1. Become one with the data.
> 2. Set up end-to-end skeleton + dumb baselines.
> 3. Overfit.
> 4. Regularize.
> 5. Tune.
> 6. Squeeze out the juice.

For eval, the transliteration is:

1. **Become one with the eval data.** Read prompts and gold answers; notice contamination and task-quality variance ([[longbench]] §gotchas).
2. **End-to-end skeleton + dumb baselines.** Random-choice baseline per MC task; "always call first tool" baseline on BFCL. Verify score at baseline matches expected chance level.
3. **Overfit.** Can your grader assign full score to a *known-perfect* response? If not, the grader is broken — fix before measuring checkpoints.
4. **Regularize.** Add item-level CIs (paired bootstrap); add effect-size thresholds; require two seeds.
5. **Tune.** Add slices. Tune the slicing axis (category vs task vs domain) to the decision you are making.
6. **Squeeze.** Read the 10 worst per slice; build the bucket ontology; publish the ledger.

Ch-50 is step 5-6 of this recipe applied to post-training eval.

---

## RL-fine-tuning gets the same silent-failure list

Source §Connections:

> **Post-training relevance (2025)**: the SFT / DPO / PPO workflow failures in 2024–2025 reports (Tülu 3, OLMo 2, Llama 3 postmortems) are all variants of Karpathy's silent-failure list — wrong tokenizer, wrong prompt template, forgotten masking, mis-normalized reward.

Ch-50 §5's canonical-buckets list reads as Karpathy's silent-failure list translated into post-training failure names. `format-violation` = wrong prompt template. `verifier-loophole` = mis-normalized reward. `language-drift` = forgotten masking. `stale-knowledge` = wrong data. Each bucket is a Karpathy-style silent failure that the per-slice ledger surfaces before it silently drifts further.

---

## Connections to ch-50

- **§1 aggregate-hides-story** — "training fails silently" is the same claim at the training-loss level.
- **§3 cluster-by-reason** — "review the 10 worst" is the literal origin of the pipeline.
- **§4 when-is-regression-real** — "predict outcome before run" produces the signed-delta thresholds.
- **§5 failure-ledger** — Karpathy-style silent-failure enumeration, institutionalized across runs.
- **§6 three-line-vs-50-slice** — "Adam 3e-4" (safe baseline) vs custom-tune is the same trade-off at the hyperparameter layer.
- **ch-51** — "two-run minimum" extends "monitor and clip gradient norms" into the variance-quantification regime.
