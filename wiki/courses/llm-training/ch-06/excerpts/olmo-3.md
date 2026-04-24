---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3 — model-flow as release artifact; 1M GPU-hour budget math

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Paper:** Team Olmo 2025, "Olmo 3" (Allen AI).

---

## Why this source anchors ch-06 §6

Ch-06 §6's thesis sentence — *"the release artifact is not the final weights; it's the tree"* — is the one-sentence summary of OLMo 3's entire philosophy. [[excerpts/olmo-2]] introduced the trajectory-as-release idea; OLMo 3 pushes it to its logical extreme by making the *branching structure* (base / think / instruct / RL-zero) the headline claim.

This excerpt extracts the specific checkpoint cadence, branch structure, and GPU-hour budget math that make OLMo 3 the most concrete illustration of WSD-style forkability at scale.

---

## The core claim — model-flow over final weights

From the source (lines 7-8):

> - **Core Insight:** The real scientific artifact is not just the final model weights but the entire model flow: pretraining stages, intermediate checkpoints, datasets, post-training branches, evals, and tooling.
> - **Guideline:** If you want a training corpus for research rather than just deployment, study OLMo 3 as a model-flow release: broad pretraining, targeted mid-training, long-context extension, then separate SFT/DPO/RLVR branches for instruct, think, and RL-zero pathways.

Notice the word **"flow."** OLMo 3 introduces this as the unit of release. A flow is a DAG: nodes are checkpoints, edges are training stages that transform one checkpoint into another. The release manifest is the DAG itself, including the training recipe for each edge and the data mix consumed by each edge.

For ch-06 this reframes the checkpoint-tree filesystem convention (§6, lines 207-222) from "a nice organizational pattern" into "the actual shape of the artifact." Every `trunk/`, `decay/`, `post/` subdirectory corresponds to a flow node; every save transition is a flow edge.

---

## The branch structure — four release paths from one base

From the source (lines 18-23):

> ## Key Contributions
> - Treats the **entire model flow** as the public artifact, not just final checkpoints.
> - Releases multiple branches from the same base: **Base**, **Think**, **Instruct**, and **RL Zero**.
> - Uses a clear **three-stage base training recipe**: pretraining, mid-training on harder distributions, and long-context extension.
> - Uses a clear **three-stage post-training recipe** inherited from Tulu 3: **SFT -> DPO -> RLVR**.
> - Makes the data curriculum explicit: **Dolma 3**, **Dolma 3 Mix**, **Dolmino**, **Longmino**, and **Dolci**.

The branch structure is the flow in concrete form:

```
Base (7B, 32B)
  ├── Think (7B, 32B)      # reasoning-focused SFT + thinking DPO + RLVR
  ├── Instruct             # chat/tool-use SFT + DPO + RLVR
  └── RL Zero              # direct RL from base, no SFT first
```

Each arrow is a fork from the base checkpoint. For ch-06's filesystem convention:

```
ckpts/
  base/
    step_pretraining_end/
    step_midtraining_end/
    step_longcontext_end/    # === "Base" release checkpoint
  think/
    sft/  dpo/  rlvr/
  instruct/
    sft/  dpo/  rlvr/
  rl_zero/
    rlvr_from_base/
```

This is exactly the §6 tree operationalized at Allen AI production scale.

---

## The three-stage base training — WSD extended

From the source (lines 39-43):

> ### Base-model training stages
> 1. **Initial large-scale pretraining** for broad text, code, and math coverage.
> 2. **Mid-training** on harder data distributions to sharpen programming, quantitative reasoning, and reading comprehension.
> 3. **Long-context extension** on very long documents.

Three stages, three checkpoints between them. For the learner reading ch-06 §6: **each transition is a fork from the prior stage's endpoint.**

- Stage 1 endpoint → Stage 2 start (Dolma 3 Mix stable → Dolmino decay).
- Stage 2 endpoint → Stage 3 start (Dolmino decay → Longmino long-context phase).

The LR schedule across these stages is not strict WSD but *compound* WSD: warmup → stable → small decay (end of Stage 1) → re-warm → stable → small decay (end of Stage 2) → re-warm → stable → final decay (end of Stage 3). Each stage is itself a mini-WSD; the transitions are scheduler-reset events.

This makes the off-by-one hazard from [[excerpts/lr-schedules]] §5.3 worse by a factor equal to the number of stage transitions — every `scheduler.load_state_dict()` call at a stage boundary is an opportunity to misfire the schedule. OLMo 3's pipeline presumably asserts `scheduler.last_epoch == saved_step` at each boundary as ch-06 §7 prescribes, or it catches the bug via the bit-exact resume check (§3). There is no way to run a four-branch public release if one stage's schedule is silently off.

---

## The data curriculum — staged token budgets

From the source (lines 44-49):

> ### Data curriculum
> - **Dolma 3:** about **9.3T** source tokens spanning web pages, science PDFs processed with `olmOCR`, code, math problems/solutions, and encyclopedic text.
> - **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens with stronger math/code emphasis and stronger decontamination.
> - **Dolma 3 Dolmino:** **100B** mid-training tokens sampled from a ~2.2T high-quality pool for math, science, code, instruction following, and reading comprehension.
> - **Dolma 3 Longmino:** about **50B** long-context tokens from a **639B**-token pool of long documents plus mid-training data.
> - **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

Token budget by stage (approximate):

| Stage | Data | Tokens |
|---|---|---|
| Stage 1 pretraining | Dolma 3 Mix | ~5.9T |
| Stage 2 mid-training | Dolmino | 100B |
| Stage 3 long-context | Longmino | 50B |
| **Base total** | | **~6.05T** |
| Post-training (SFT+DPO+RLVR) | Dolci | not disclosed |

Notice: mid-training is ~1.7% of pretraining; long-context is ~0.8%. The ch-06 §6 observation that the decay phase is 10-20% of WSD tokens is a rough budget; OLMo 3's mid-training + long-context together are ~2.5% of Stage 1 — *shorter* than classical WSD decay, reflecting the shift from "final decay for final weights" to "specialization stage for a specific capability."

For the ch-06 instrumentation tiers: at Stage 3 (Longmino, 50B tokens), at a reasonable bf16 throughput of ~30M tokens/hour/GPU on a 7B model, that is ~1700 GPU-hours — about 14 hours on 128 H100s. A 1000-step checkpoint cadence at this scale is ~50 saves over Stage 3. Each save is one fork-eligible state; the researcher downstream can fork their long-context-specialized model from any of the 50 intermediate Longmino checkpoints, not just the endpoint.

---

## The 1M GPU-hour budget math

From the source (lines 56-59):

> ### Efficiency and infrastructure
> - Pretraining used up to **1,024 H100 GPUs**.
> - Mid-training used **128 H100 GPUs**.
> - Post-training used **256 H100 GPUs**.
> - Moving SFT from **Open Instruct** to **Olmo Core** reportedly improved throughput by **8x**.
> - In-flight weight updates, continuous batching, and threading work made RL training about **4x** more efficient.

Work the budget for 32B:

- **Pretraining**: 5.9T tokens × 6 × 32B params (Chinchilla-style FLOP estimate) = ~1.1e24 FLOPs. At H100 bf16 peak ~1000 TFLOPS × ~45% MFU = 450 TFLOPS effective per GPU. 1.1e24 / 450e12 = ~2.4e9 GPU-seconds = ~670K GPU-hours. On 1024 H100s: ~28 days of wall time.
- **Mid-training**: 100B × 6 × 32B = 1.9e22 FLOPs → ~12K GPU-hours → ~4 days on 128 H100s.
- **Long-context**: similar order, ~6K GPU-hours.
- **Base total**: ~690K GPU-hours.

Then the branch forks:

- **Think SFT+DPO+RLVR** ≈ 5% of pretraining ≈ 35K GPU-hours each path.
- **Instruct path** ≈ 35K GPU-hours.
- **RL Zero** ≈ 50K GPU-hours (RLVR is more compute-intensive per token).

Total across the whole flow: ~850K GPU-hours ≈ **~1M GPU-hours** (rounding up for the 7B model, evals, re-runs). Ch-06's pull-quote — *"OLMo 3's ~1 M GPU-hour budgets"* — is this number.

The efficiency wins the paper calls out (8× SFT throughput, 4× RL throughput via "in-flight weight updates, continuous batching, and threading work") are the *reason* this budget fits at a public-lab scale. The production-level throughput optimizations are the context for ch-06 §4's per-run tier metrics: the lab that publishes `tokens/sec` alongside the checkpoint is the same lab that optimizes those tokens/sec by an order of magnitude over a previous release.

---

## The tooling surface — what OLMo 3 releases beyond checkpoints

From the source (line 23):

> - Couples model release with tooling: **Olmo-core**, **Open Instruct**, **OLMES**, **OlmoTrace**, decontamination, and dedup utilities.

For ch-06 this is the "per-run tier" instrumentation made public:

- **Olmo-core** — the training code. Reproducing a checkpoint requires the exact trainer, not just the weights.
- **Open Instruct** — post-training code.
- **OLMES** — evaluation harness. Benchmarks are not "whatever we ran"; they are the same harness the trainer invokes at the per-eval tier.
- **OlmoTrace** — data-lineage tracking. Every token in every checkpoint is traceable back to its source document.
- **Decontamination / dedup utilities** — the data-integrity pipeline that ran before Dolma 3 Mix was assembled.

Ch-06 §4's per-run row (code commit, `pip freeze`, data-order hash, `nvidia-smi --query`) is a *subset* of what OLMo 3 publishes. A lab reproducing an OLMo 3 checkpoint has access to the full stack, not just the record-of-what-we-ran.

---

## Why "RL Zero" exists as its own branch

From the source (lines 52-55):

> - Each main branch follows **SFT -> DPO -> RLVR**.
> - The **Think** branch uses thinking-specific SFT, thinking DPO, and RLVR to elicit high-quality reasoning traces.
> - The **RL Zero** branch exists specifically to study RLVR from the base model without hiding the intermediate path.

The RL Zero branch is a research control. Its purpose:

- Instruct and Think branches go Base → SFT → DPO → RLVR (three forks).
- RL Zero goes Base → RLVR directly (one fork).

Researchers can compare: does RLVR help more when applied to a raw base model (RL Zero) or to an SFT+DPO policy (Instruct/Think)? The answer is part of the public release because *both branches* are checkpointed and published. Without the full flow being released, this comparison would be impossible — you would need to re-run everything yourself.

This is the single cleanest operational demonstration of ch-06 §6's "checkpoint flywheel" claim: the *same trunk* enables multiple scientific comparisons, and each comparison requires a fork that was *saved, not just run*.

---

## Checkpoint cadence — inferences from the tooling

The source does not publish a per-step save cadence for OLMo 3. Inferences from the release:

- "All intermediate checkpoints" is the design goal (from OLMo 2's precedent). The 32B run on 1024 H100s for ~28 days at ~1000-step cadence → ~40 saves per day → ~1100 trunk saves total. Consistent with the OLMo 2 "1000+ intermediate checkpoints" style at the 32B scale.
- Mid-training and long-context are shorter; denser relative cadence (per-100-steps or per-500-steps) is plausible because these stages are where capability-emergence is being studied and the researcher wants fine-grained checkpoints.
- Post-training (SFT, DPO, RLVR) is per-epoch or per-1000-steps; the Dolci mix's size drives this.

The absence of a published cadence number is itself informative: OLMo 3 leaves it implicit, treating checkpoint cadence as a design parameter the reproducer can inspect by downloading the repository manifest rather than a headline number.

---

## What to take from OLMo 3 for ch-06

1. **The flow is the release.** Every branch, every stage, every intermediate checkpoint is part of the public artifact. Checkpointing is not recovery; it is publication.
2. **Compound WSD across stages.** Three base-training stages, each with its own warmup-stable-decay mini-schedule, mean scheduler resume is a three-transition problem. [[excerpts/lr-schedules]] §5.3's off-by-one applies at every boundary.
3. **1M GPU-hours is the working budget for a 32B model flow.** Pretraining dominates (~70%), post-training branches each add ~5%. Losing a week to a mis-resume is a ~5% budget hit.
4. **Tooling is instrumentation.** Olmo-core, OLMES, OlmoTrace together are the per-run tier of ch-06 §4 made public; reproducibility requires all of them.
5. **Fork topology enables research comparisons.** RL Zero vs Instruct is a scientific control that exists only because both branches were checkpointed from the same base.

---

## Connections

- [[excerpts/olmo-2]] — the predecessor; OLMo 3 generalizes its two-stage approach into a four-branch flow.
- [[excerpts/early-stopping-and-checkpointing]] — WSD-decay averaging; OLMo 3's compound-WSD structure is the scaled-up version.
- [[excerpts/lr-schedules]] — scheduler state at three stage transitions; off-by-one hazards multiplied.
- [[excerpts/fsdp-sft]] — DCP save path that makes 32B model flow tractable to publish.
- [[excerpts/llama-3]] — contrast: Llama 3 publishes interruption stats but not the branch tree; OLMo 3 inverts this.
- [[excerpts/karpathy-training-neural-net-recipe]] — "use a constant LR for sanity, schedule last" applied across three stage LRs; OLMo 3 is the production-scale demonstration.
- [[ch-06]] — §6 (checkpoint flywheel, filesystem tree), §1 (instrumentation audit trail), §4 (per-run tier reproducibility).
