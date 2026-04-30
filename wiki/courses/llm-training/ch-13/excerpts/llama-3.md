---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — what the herd report discloses about mix, and what it hides

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Paper:** Grattafiori et al. 2024, "The Llama 3 Herd of Models" (Meta).

---

## Why this source anchors ch-13

Ch-13 §5 compares how four labs report their mixing decisions. Llama 3 sits on the "bucket-names-disclosed, weights-withheld" end of the spectrum. This excerpt walks through the specific disclosures and the specific omissions — both matter for the "how to read a mix disclosure" skill ch-13 is building.

---

## What Llama 3 discloses

From the source (lines 58-62):

> ### Scale
> - **Pretraining:** 15.6T tokens, 8K native context, 8-way sequence parallel for long-context extension.
> - **405B compute:** 3.8e25 FLOPs.
> - **Post-training compute:** not disclosed as a standalone number; post-training is "a small fraction" of pretraining compute.

15.6T total pretraining tokens is disclosed. This is a ceiling on the mix — every per-domain weight must sum (in expectation, after resampling) to 15.6T. But the per-domain decomposition is given only as bucket names, not as fractions.

From the source (line 20):

> - Heavy synthetic-data generation for coding, math, multilingual, reasoning, long-context, tool use, and factuality — each capability gets a dedicated synthetic pipeline.

Seven capability buckets are named: coding, math, multilingual, reasoning, long-context, tool use, factuality. Each has its own synthetic pipeline. Implicitly this is a 7-dimensional α vector (plus the general-web backbone). The values of that α are not published.

---

## What Llama 3 hides

The specific missing pieces:

1. **Pretraining α.** No per-domain token counts, no percentages. The reader can infer only ordering ("web dominates, code is substantial") from the paper's prose.
2. **Per-round SFT mix evolution.** From the source (line 35):
   > Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations.

   Each round has a fresh SFT mix. Whether round-3's mix upweights code relative to round-2's mix, or downweights it, is not disclosed. The six rounds imply six α vectors; six are hidden.
3. **Filter thresholds.** From the source (line 40):
   > topic classifier + quality classifier (both distilled from Llama 3) remove low-quality rejection-sampled text before SFT

   The thresholds that define "low-quality" are the actual quality lever; the paper names the classifiers but not the thresholds.

For ch-13 §5's teaching payload, the pattern is: **frontier labs disclose capability categories but withhold the weight vector**. A reader who only sees "web + code + math + multilingual" has been told the *support* of α, not α itself.

---

## The iterative structure as implicit re-weighting

From the source (lines 35):

> Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations. Each round uses a fresh batch of ~human preference annotations plus synthetic data resampled from the round-N-1 best model.

The six-round structure is itself a form of mix re-weighting that ch-13 §4 underscores as stage-specific. Each round:

1. Uses the round-(N-1) best model to generate K=10-30 completions per prompt (source line 39).
2. Filters by RM score — the RM score *is* a per-sample weight.
3. Applies the topic + quality classifier — another per-sample filter.
4. The surviving samples form round-N's SFT mix.

This is not a fixed α; it is an α that evolves with the model. The "mix disclosure" for Llama 3 post-training is necessarily a *procedure* rather than a weight vector — and the paper is clear about the procedure while opaque about the numbers it produces.

---

## The post-training mix as capability-bucketed

From the source (lines 55-58):

> ### Data mix (per round)
> - ~50–80% synthetic rejection-sampled data
> - Remainder: human SFT demonstrations, preference data, capability-specific synthetic (code-exec-filtered code, math with verifier, multi-turn tool use traces, long-context QA).

50–80% synthetic is disclosed — the single most informative mix fact in the report. This fraction is high compared to Llama 2's era (where human SFT demonstrations dominated) and reflects the maturation of rejection sampling as the dominant SFT data source.

The "capability-specific synthetic" tail has a per-capability quality gate:
- Code: execution filter (pass/fail on hidden tests).
- Math: verifier (symbolic or LLM-as-judge).
- Tool use: trajectory synthesis with tool-execution validation.

Each capability's data is quality-gated by its own domain-specific verifier before entering the mix. The α is implicit: whatever fraction of the seed prompts × acceptance rate × per-capability pipeline throughput ends up in the final pool.

---

## What a DoReMi for Llama 3 would look like

Speculation, but instructive: if Meta had run DoReMi at Llama 3 pretraining scale, the proxy would have been ~15B parameters (30× scale-down from 405B). Training a 15B proxy on ~300B tokens (Chinchilla) is nontrivial — roughly a small lab's flagship run. This is possibly why the frontier-lab pattern is to *not* run DoReMi but to run a sweep of ~15B-scale ablations with hand-tuned α candidates, which is still cheaper than a 405B sweep but loses DoReMi's optimality guarantee.

Ch-13 §1's observation that "hand-tuning is sample-inefficient" applies symmetrically: DoReMi is cheaper than hand-tuning at trunk scale, but a frontier lab doing trunk-scale ablations is still paying less than the production-scale run, so the DoReMi vs hand-tune choice depends on how many candidate mixes you have to compare.

---

## Connections

- `[[llama-3]]` — raw source.
- `[[ch-13]]` — §5.1 summarizes what's disclosed, §4 places post-training mix in the stage-specific table.
- `[[olmo-2]]`, `[[olmo-3]]` — much more forthcoming counterparts on the same axis.
- `[[deepseek-v3]]` — even more closed; 14.8T tokens with no decomposition at all.
- `[[doremi]]` — the method Llama 3 did not run (as far as we know).
