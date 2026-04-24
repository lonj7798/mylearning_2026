---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3's OLMES + OlmoTrace — a lab-grade harness you can reverse-engineer

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Anchor report:** Team Olmo 2025 — "OLMo 3: Fully Open Model Flow"

---

## Why this source anchors §1 of ch-53

The ch-53 harness is an educational miniature of what real labs ship. OLMo 3 is the best currently-public example of a full eval stack because the model flow is released as the artifact, not just the weights. Reading the OLMo 3 report is the fastest way to see where the ch-53 skeleton sits in the real hierarchy.

From `olmo-3.md` §Key Contributions:

> Couples model release with tooling: **Olmo-core**, **Open Instruct**, **OLMES**, **OlmoTrace**, decontamination, and dedup utilities.

OLMES is the eval harness, OlmoTrace is the contamination / provenance tracer, and the decontamination utility is the Lee 2021 gate in production form. All three concepts show up in the ch-53 deliverables.

---

## The model-flow framing — why the harness is a library, not a script

From `olmo-3.md` §Abstract:

> The key claim is not only that the models are strong, but that every stage of their construction is public: data, checkpoints, code, and dependencies across pretraining, mid-training, long-context extension, and post-training.

And §Family structure:

> Base models at **7B** and **32B**.
> Reasoning-focused **Think** models at **7B** and **32B**.
> Chat/tool-use **Instruct** path.
> **RL Zero** path for direct RL experimentation from the base model.

OLMo 3 ships four branches that must be comparable. That comparability is what forces the harness into library form: each branch's CI calls the same scoring code, and the slice tables have to line up field-for-field so regressions across branches are legible. The ch-53 `TaskSpec` / `RunResult` shape is engineered for exactly this: `run_a = harness.run(task, branch_a)`, `run_b = harness.run(task, branch_b)`, `compare(run_a, run_b)` with no ad-hoc glue.

---

## The post-training staging — why ch-53 compares SFT and RL specifically

From `olmo-3.md` §Post-training:

> Each main branch follows **SFT -> DPO -> RLVR**.

The ch-53 lab compares `ch-34` (SFT) with `ch-44` (RLVR). OLMo 3's pipeline makes the same comparison as a matter of routine — each stage is versioned, each stage is scored on the same eval surface, and the report calls out where each capability was added. The regression rule in §6 of the read.md ("did the RL stage regress on any slice?") is the ch-53 version of the OLMo 3 internal question "did RLVR cost us anything we care about?"

---

## The efficiency note — calibrating resource-constrained expectations

From `olmo-3.md` §Efficiency and infrastructure:

> Pretraining used up to **1,024 H100 GPUs**.
> Mid-training used **128 H100 GPUs**.
> Post-training used **256 H100 GPUs**.
> Moving SFT from **Open Instruct** to **Olmo Core** reportedly improved throughput by **8x**.
> In-flight weight updates, continuous batching, and threading work made RL training about **4x** more efficient.

The OLMo 3 stack optimizes eval as aggressively as training — the "In-flight weight updates, continuous batching" note is exactly what lets them evaluate intermediate checkpoints without multiplying wall-clock cost. The ch-53 resource-constrained path is the cheap cousin: offline rollouts + cached generations + CPU bootstrap. The architectural lesson — eval must be as cheap to run as any single training step or it gets skipped — holds at both scales.

---

## Why OLMo 3 is the right reference and not lm-eval-harness

lm-eval-harness is the right API shape. OLMES is the right operational shape. From `olmo-3.md` §Why OLMo 3 matters:

> For a learner, it is unusually valuable because you can study where a capability was added: base, mid-training, long-context, DPO, or RLVR.

The ch-53 harness inherits two design choices from this framing:

1. **Every run writes a structured artifact, not just a score.** The memo template in §7 is the artifact; the slice table is the dataframe the memo reads from. Both are cross-run comparable.
2. **Contamination and provenance are first-class outputs.** OlmoTrace lets you trace a prediction back to training data; the ch-53 contamination gate is the minimum viable version — it does not trace, but it refuses to score if the training set is too close.

---

## The three-stage data curriculum — why contamination is non-trivial

From `olmo-3.md` §Data curriculum:

> **Dolma 3:** about **9.3T** source tokens.
> **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens.
> **Dolma 3 Dolmino:** **100B** mid-training tokens.
> **Dolma 3 Longmino:** about **50B** long-context tokens.
> **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

Five distinct training stages means five distinct dumps to check the eval against. The ch-53 `TaskSpec.contamination_scope` field exists precisely because a MATH problem can be clean against Dolci-RLVR and contaminated against Dolma 3 Mix — the gate must run against the right dump per task.

---

## What this source does not tell you

OLMo 3 does not publish the OLMES source in detail in the paper; the code is the authoritative reference. The ch-53 skeleton is intentionally smaller than OLMES (single-threaded runner, no distributed generation) — treat OLMES as the eventual target, not the starting point. Read the OLMo 3 paper first, then the OLMES repo, then extend the ch-53 harness one task family at a time.
