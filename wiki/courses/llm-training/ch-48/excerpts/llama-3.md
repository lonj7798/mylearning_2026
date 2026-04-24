---
chapter: ch-48
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — downstream contamination through iterative SFT → RS → DPO

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Artifact:** Iterative-rounds pipeline that creates the train → RM pref → eval leakage pathway; rejection-sampling data flow between rounds.

---

## Why this source motivates ch-48 §5

Ch-48 §5 "Downstream contamination: train → RM pref → eval" is not a theoretical risk. It is the exact shape of Llama 3's post-training pipeline. The source's Figure 7 (post-training flow diagram) traces six rounds of SFT → rejection sampling → DPO, each round re-mining the previous checkpoint's outputs. That re-mining is where contamination compounds.

---

## The pipeline shape that creates the risk

Source §Technical Details — Post-Training Pipeline / Overall structure:

> Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations.

Read this flow with contamination in mind. Each box is a re-entry point for eval leakage:

1. **RM training** consumes human preferences. If the prompts overlap eval, the RM learns to reward "the answer that matches the memorized solution."
2. **Rejection Sampling** generates K=10–30 completions per prompt and keeps the top by RM score. If the RM rewards memorized answers, the top-K is enriched in memorized content.
3. **SFT** on that top-K trains the policy to produce memorized answers.
4. **DPO** preference pairs sampled with the latest RM amplify the same signal.

A single contamination event in round 1's SFT mix becomes a compounded contamination event by round 6. This is why ch-48 §5 requires re-hashing at every stage boundary, not just at pretraining.

---

## The rejection-sampling detail that turns one leak into many

Source §Technical Details / SFT / Rejection sampling:

> for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score.

If prompt_i is a paraphrased eval instance, K=30 samples * 6 rounds = 180 opportunities for the policy to regurgitate the memorized gold answer and have the RM reward it. The ch-48 decontamination check at "SFT → RM preference" boundary is designed to catch this: flag any RS output whose content matches an eval instance's gold answer, not only outputs whose prompt matches an eval prompt.

---

## Why the filter classifier is not a decontamination substitute

Source §Technical Details / SFT / Filtering:

> topic classifier + quality classifier (both distilled from Llama 3) remove low-quality rejection-sampled text before SFT.

These classifiers filter for quality, not for contamination. A memorized correct answer to a benchmark problem is *maximally high-quality* by the classifier's metric. This is a classic trap: contamination looks like signal, and a quality filter amplifies contamination while removing noise. Ch-48's memo template §7 explicitly requires an anti-contamination pass orthogonal to the quality classifier.

---

## The synthetic-data disclosure that reveals exposure

Source §Technical Details / SFT / Data sources:

> filtered synthetic data for code/math/reasoning/multilingual/long-context/tool use.

Each of these synthetic pipelines has a different contamination surface:
- **Code** — public LeetCode / APPS problems appear in eval sets; direct overlap risk.
- **Math** — GSM8K / MATH / AIME are public; contamination risk parallels [[bespoke-stratos]].
- **Long-context / tool use** — custom synthesis, lower public-benchmark overlap.

The memo should tag each synthesis pipeline with its contamination exposure tier.

---

## What ch-48 borrows from this pipeline

| Llama 3 design decision | Ch-48 contamination implication |
|---|---|
| 6 rounds of SFT → RS → DPO | Each boundary is a mandatory decontamination checkpoint |
| RM gates rejection-sampling | RM itself must be decontamination-audited |
| Most-recent-batch DPO data only | Good for format drift; neutral for contamination |
| Synthetic data dominates SFT pool | Synthetic pipelines inherit teacher memorization |

---

## Connections

- **[[bespoke-stratos]]** — minimal reproduction of the teacher-memorization pathway.
- **[[olmo-3]]** — the open-flow counterpart; explicit per-stage decontamination tooling.
- **[[deduplicating-training-data]]** — the primitives Llama 3 internally uses (undisclosed but methodologically equivalent).
- **[[anthropic-sleeper-agents-data]]** — adversarial variant of the same downstream-amplification mechanism.
