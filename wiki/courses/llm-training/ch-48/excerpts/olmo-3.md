---
chapter: ch-48
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3 — per-stage decontamination as public tooling

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Artifact:** Model-flow-as-release philosophy; Dolma 3 → Dolma 3 Mix with "stronger decontamination"; decontamination + dedup utilities released alongside the model.

---

## Why OLMo 3 is the ch-48 reference implementation

Ch-48 insists that decontamination must run at every stage boundary and be a reproducible artefact, not a one-shot script. OLMo 3 is the cleanest public example of that discipline. Because the entire *model flow* is released — not only the final checkpoint — every stage's decontamination run is a public, inspectable artefact.

---

## The multi-stage flow that drives the per-stage rule

Source §Technical Details / Base-model training stages:

> 1. Initial large-scale pretraining for broad text, code, and math coverage.
> 2. Mid-training on harder data distributions to sharpen programming, quantitative reasoning, and reading comprehension.
> 3. Long-context extension on very long documents.

Source §Technical Details / Post-training:

> Each main branch follows SFT -> DPO -> RLVR.

Six stages, six decontamination opportunities. Ch-48 §5 "Downstream contamination" treats each of these as a potential leakage re-entry; OLMo 3 is the public proof that the discipline is tractable at 7B/32B scale.

---

## Dolma 3 → Dolma 3 Mix: the explicit decontamination delta

Source §Technical Details / Data curriculum:

> - **Dolma 3:** about **9.3T** source tokens spanning web pages, science PDFs processed with `olmOCR`, code, math problems/solutions, and encyclopedic text.
> - **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens with stronger math/code emphasis and stronger decontamination.

The 9.3T → 6T reduction reflects quality filters, topic reweighting, *and* decontamination. OLMo 3 keeps these transforms auditable. Ch-48 memo §7 "what was removed" table requires exactly this granularity: document counts per stage, delta in token count, per-eval hit distribution.

---

## The cooldown is the highest-risk stage

Source §Technical Details / Data curriculum:

> - **Dolma 3 Dolmino:** **100B** mid-training tokens sampled from a ~2.2T high-quality pool for math, science, code, instruction following, and reading comprehension.

The same pattern [[olmo-2]] introduced (Dolmino cooldown) is amplified in OLMo 3: a small, high-quality, late-stage pool. This is structurally the highest-risk contamination stage because:
- High-quality pools are sampled from domain-aligned sources (math textbooks, coding repos) that share distribution with eval sets.
- Late-stage exposure has outsized effect on model behaviour per token — contamination here shows up strongly in benchmarks.
- Small pools (100B) make global dedup cheap; no excuse for skipping per-stage decontamination.

Ch-48 §5's recommendation to check pretraining → mid-training → cooldown boundaries separately is directly motivated by this structure.

---

## Public tooling as a memo requirement

Source §Key Contributions:

> Couples model release with tooling: **Olmo-core**, **Open Instruct**, **OLMES**, **OlmoTrace**, decontamination, and dedup utilities.

The explicit naming of "decontamination utility" as a first-class released artefact is the operational pattern ch-48 §7 memo template asks for under "Reproducibility": code commit SHA, hash-pinning artefact, run-time and compute. OLMo 3 is the canonical model for what this deliverable looks like.

---

## The "RL Zero" branch as a contamination control

Source §Technical Details / Family structure:

> **RL Zero** path for direct RL experimentation from the base model.

RL Zero is a contamination control: it starts from the audited base model, without SFT or DPO, so researchers can measure whether capability gains attributed to RL are real or are instead downstream contamination flowing through SFT → DPO. Ch-48 §5 recommends this exact pattern: if an eval number jumps dramatically during post-training, ask whether a contamination pathway could account for it before celebrating.

---

## What ch-48 takes from OLMo 3

| OLMo 3 practice | Ch-48 adoption |
|---|---|
| Release decontamination utility alongside model | Memo §7 reproducibility bullet |
| Per-stage data mixes with explicit decontam deltas | §5 per-stage check requirement |
| Cooldown as a separately-audited stage | Highest-risk stage callout in §5 |
| RL Zero branch as a capability-source control | Downstream-contamination A/B protocol |

---

## Connections

- **[[olmo-2]]** — earlier Dolmino cooldown introduction; same per-stage pattern at smaller scale.
- **[[dolma]]** — Dolma 1's filter cascade that Dolma 3 Mix inherits.
- **[[tulu-3]]** — post-training recipe OLMo 3 builds on; SFT/DPO/RLVR decontamination boundaries.
- **[[llama-3]]** — closed counterpart; same pipeline risks without the public audit trail.
