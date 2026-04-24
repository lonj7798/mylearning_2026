---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMES and the model-flow view of evaluation

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Artifact:** OLMES as a fully-open eval suite co-released with OLMo 3; the design move that *eval is part of the model flow*, not a separate artifact.

---

## Why this source opens ch-47

Ch-47 argues that a benchmark number is the output of a 6-tuple `(task_shape, prompt_template, matcher, inference_config, subset_slice, harness_version)`. OLMo 3 is the clearest public example of a release that treats the harness as a first-class artefact of the model flow, not a convenience script. The source's framing — "the real scientific artifact is not just the final model weights but the entire model flow" — generalises directly from pretraining data to eval infrastructure. That is the thesis of ch-47 §2 and §6.

---

## The quote that anchors §2 of ch-47

Source §Core Insight:

> The real scientific artifact is not just the final model weights but the entire model flow: pretraining stages, intermediate checkpoints, datasets, post-training branches, evals, and tooling.

Notice: "evals and tooling" are enumerated alongside weights. OLMo 3 does not ship "a model plus some scripts"; it ships a closed-loop artefact whose eval recipe is reconstructible from the release. That is why ch-47 insists you never quote a number without the harness-version coordinate — the entire OLMo 3 release philosophy collapses if you let the eval recipe float.

---

## Tooling list — the dial on the harness

Source §Key Contributions:

> Couples model release with tooling: **Olmo-core**, **Open Instruct**, **OLMES**, **OlmoTrace**, decontamination, and dedup utilities.

OLMES is on the short list of first-class release tools. Notice: `decontamination` and `dedup utilities` are listed alongside the harness. That is the pairing ch-47 §6 demands: versioning an eval is useless if the eval set is contaminated by the training data; the harness-version coordinate and the data-decontamination coordinate are co-dependent. You cannot version one without the other.

---

## Why OLMES design is release-scale, not academic-convenience

Source §Family structure + §Post-training:

> Base models at **7B** and **32B**. Reasoning-focused **Think** models at **7B** and **32B**. Chat/tool-use **Instruct** path. **RL Zero** path for direct RL experimentation from the base model. [...] Each main branch follows **SFT -> DPO -> RLVR**.

Four branches × two sizes = eight checkpoints, each post-trained through three distinct stages. The release would be unreadable without a shared eval frame; OLMES is what makes those 8 × N numbers comparable. Ch-47 §2's harness table attributes to OLMES the axis **"prompt-format control: explicit prompt format version string in task id"** precisely because that is what lets the Think-7B number and the Instruct-7B number be read off the same axis.

---

## Data curriculum as a precedent for eval curriculum

Source §Data curriculum:

> **Dolma 3:** about **9.3T** source tokens [...] **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens [...] **Dolma 3 Dolmino:** **100B** mid-training tokens [...] **Dolma 3 Longmino:** about **50B** long-context tokens [...] **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

Five named, sized, purpose-specific data pools — each with its own identity and disclosure. Ch-47 §5 (slicing) and §6 (versioning) lift this structure onto eval. A release does not have "an eval suite"; it has a *curriculum* of evals, each named and versioned separately (MMLU for knowledge, GSM8K for math, RULER for long-context, HarmBench for safety), each with its own inference-config profile. The Dolma/Dolmino/Longmino pattern is the upstream template for how §5 recommends you report numbers.

---

## The reproducibility claim that forces harness-pinning

Source §Why OLMo 3 matters:

> It is one of the clearest public examples that **openness can apply to training trajectories**, not only to final checkpoints.
> For a learner, it is unusually valuable because you can study where a capability was added: base, mid-training, long-context, DPO, or RLVR.

Attribute-where-a-capability-was-added is exactly the question a versioned harness answers for evaluation. If MMLU moves 2 points between ch-36's SFT output and ch-45's RLVR output, OLMES lets you ask whether the delta is from the weights, from the prompt-format bump, from a stop-string change, or from a re-bucketed subset. Without a release-pinned harness id, the delta is unattributable.

---

## Efficiency as evidence for harness discipline

Source §Efficiency and infrastructure:

> Pretraining used up to **1,024 H100 GPUs**.
> Mid-training used **128 H100 GPUs**.
> Post-training used **256 H100 GPUs**.
> Moving SFT from **Open Instruct** to **Olmo Core** reportedly improved throughput by **8x**.
> In-flight weight updates, continuous batching, and threading work made RL training about **4x** more efficient.

Notice: none of these efficiency gains would be observable — or trusted — without eval-side discipline. If OLMES produced a noisy number, an 8× throughput improvement that nudged a score by 0.3 points would be indistinguishable from noise. The release demonstrates that the eval side of the model flow has to be at least as reproducible as the training side, or you cannot claim the training speedup preserved quality. Ch-47 §6 treats this symmetrically: the release contract is the harness id, the checkpoint id, *and* the decontamination audit.

---

## Mid-training / long-context extension — §1 shape matching

Source §Base-model training stages:

> 1. **Initial large-scale pretraining** for broad text, code, and math coverage.
> 2. **Mid-training** on harder data distributions to sharpen programming, quantitative reasoning, and reading comprehension.
> 3. **Long-context extension** on very long documents.

Three distinct stages, three distinct capability targets. Ch-47 §1 insists `task_shape` must match capability; OLMo 3's base-training structure is the upstream analog — each stage has its own evaluation intent. A base-pretraining checkpoint should be measured on broad knowledge (MMLU MCQ); a mid-training checkpoint on programming and quant reasoning (HumanEval generation, GSM8K); a long-context checkpoint on [[ruler]]. Reporting the mid-training checkpoint on a long-context benchmark measures nothing — the capability was not added yet.

---

## The "Olmo-core + OLMES" pairing as infra lesson

Source §Tooling list (quoted above):

> Olmo-core, Open Instruct, OLMES, OlmoTrace, decontamination, and dedup utilities.

`Olmo-core` and `OLMES` are paired: one is the training runtime, the other is the eval runtime. The release ships both at the same version id, because a training run and its evaluation are operationally one artefact. Ch-47's release-discipline guidance ("pin the harness commit SHA alongside the weights") is the external-facing projection of this internal pairing; OLMo 3 does it because the team cannot internally attribute capability changes without it, and the release-public version is just the same discipline exposed.

`OlmoTrace` is the third leg: trace-level inspection of the training run. Ch-47 does not cover tracing directly, but the parallel to eval-time slicing is exact — trace aggregates hide training hacks the same way eval aggregates hide capability hacks.

---

## What ch-47 keeps, changes, drops from OLMo-3's release posture

| OLMo-3 release pattern | Ch-47 normative claim | Reason |
|---|---|---|
| Ship tools alongside weights | A number without a harness version is half a sentence | Same principle, stated in the eval frame |
| Task-id embeds format version | `olmes:mmlu:v1.1` is a contract, not a suggestion | §6 versioning |
| Data curriculum as separate named pools | Eval curriculum as separate named tasks with own inference config | §1 + §5 |
| Four branches from one base | Numbers across branches must share the same harness tuple | §2 cross-harness comparison caveat |
| "Eval" listed in the model-flow enumeration | Harness is model-flow infrastructure; not eng-only | Opening §Why this chapter exists |

---

## Connections

- **[[ch-47]]** — this excerpt anchors §Why this chapter exists, §2 (harness table), §6 (versioning).
- **[[excerpts/judge-llm-bias]]** — OLMES formalises *prompt-format* versioning; judge-llm-bias covers the *matcher* versioning for open-ended tasks. Same spirit, different axis.
- **[[excerpts/bfcl]]** — BFCL's V1→V4 version generations are the tool-calling analog of OLMES's task-id discipline.
- **[[llama-3]]** — Llama 3 eval suite is release-scale too; OLMES-style pinning is what would make its numbers reproducible by a third party.
- **Track 7 (Eval) / ch-48..ch-53** — every downstream chapter in this track inherits the 6-tuple vocabulary that this excerpt grounds.
