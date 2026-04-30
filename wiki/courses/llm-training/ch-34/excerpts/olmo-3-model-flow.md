---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3 — the model-flow as the public artifact

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Report:** Team Olmo (Allen AI), 2025, "OLMo 3".

---

## Why this source anchors ch-34 §4

OLMo 3 is the public origin of **model-flow-as-artifact**: the claim that the scientific object a lab should release is not the final weights but the *entire path* — pretraining stages, intermediate checkpoints, post-training branches, data curriculum, and tooling. For ch-34, this makes OLMo 3 the "branches-per-checkpoint" champion on the modes axis. Qwen 3 carries two modes in one model; OLMo 3 carries four *branches* from one base.

---

## The core stance

From the source (lines 7-8):

> - **Core Insight:** The real scientific artifact is not just the final model weights but the entire model flow: pretraining stages, intermediate checkpoints, datasets, post-training branches, evals, and tooling.
> - **Guideline:** If you want a training corpus for research rather than just deployment, study OLMo 3 as a model-flow release.

This stance is why OLMo 3's post-training is less algorithmically novel than Qwen 3's or Phi-4-reasoning's — the novelty is in what's *published*, not in the recipe. The recipe is Tülu 3's SFT → DPO → RLVR, applied separately per branch.

---

## The four branches

From the source (lines 32-37):

> ### Family structure
> - Base models at **7B** and **32B**.
> - Reasoning-focused **Think** models at **7B** and **32B**.
> - Chat/tool-use **Instruct** path.
> - **RL Zero** path for direct RL experimentation from the base model.

Mapped to ch-34's design stances:

| Branch | SFT? | DPO? | RLVR? | Purpose |
|---|---|---|---|---|
| Base | — | — | — | Pretrained checkpoint (start of every other branch) |
| Think | thinking-specific SFT | thinking DPO | yes | Reasoning traces + math/code |
| Instruct | general SFT | general DPO | yes | Chat + tool-use |
| RL Zero | — | — | **yes from Base** | RL without SFT priming, for research |

The **RL Zero branch is ch-34's most interesting item**. It is a deliberate ablation: what happens if you run RLVR directly on the base model without SFT or DPO first? That setup cannot ship as a product — it is going to behave poorly as a chat model — but it is exactly the configuration needed to study RL-only reasoning emergence (the question DeepSeek-R1-Zero raised). By releasing this branch, OLMo 3 gives the research community the checkpoints to reproduce and extend R1-Zero-style claims on an open base.

---

## The three base-model stages

From the source (lines 39-43):

> ### Base-model training stages
> 1. **Initial large-scale pretraining** for broad text, code, and math coverage.
> 2. **Mid-training** on harder data distributions to sharpen programming, quantitative reasoning, and reading comprehension.
> 3. **Long-context extension** on very long documents.

Mapped to attested data volumes (lines 44-49):

> - **Dolma 3:** about **9.3T** source tokens ...
> - **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens with stronger math/code emphasis and stronger decontamination.
> - **Dolma 3 Dolmino:** **100B** mid-training tokens sampled from a ~2.2T high-quality pool for math, science, code, instruction following, and reading comprehension.
> - **Dolma 3 Longmino:** about **50B** long-context tokens from a **639B**-token pool of long documents plus mid-training data.
> - **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

This is the first fully-named staged curriculum in the public LLM literature. Every earlier lab (Qwen, Phi, Llama) has a staged curriculum but does not label the stages separately with per-stage datasets. OLMo 3's naming convention — **Dolma / Dolmino / Longmino / Dolci** — gives each stage a citeable artifact.

---

## Post-training: Tülu recipe per branch

From the source (lines 50-54):

> ### Post-training
> - Each main branch follows **SFT -> DPO -> RLVR**.
> - The **Think** branch uses thinking-specific SFT, thinking DPO, and RLVR to elicit high-quality reasoning traces.
> - The **RL Zero** branch exists specifically to study RLVR from the base model without hiding the intermediate path.

Dolci is the post-training analogue of Dolma: a *suite* of mixes, not a single dataset. The SFT mix, DPO mix, and RLVR prompts are published as separate named components. Within the Think branch, the SFT mix contains thinking-specific long-CoT traces (the Qwen 3 Stage-1 analogue), and the DPO mix contains preference pairs over reasoning responses.

---

## Infrastructure — the 8× / 4× claim

From the source (lines 56-61):

> ### Efficiency and infrastructure
> - Pretraining used up to **1,024 H100 GPUs**.
> - Mid-training used **128 H100 GPUs**.
> - Post-training used **256 H100 GPUs**.
> - Moving SFT from **Open Instruct** to **Olmo Core** reportedly improved throughput by **8x**.
> - In-flight weight updates, continuous batching, and threading work made RL training about **4x** more efficient.

These numbers do not affect the algorithm but they affect who can reproduce. An 8× SFT speedup means the Dolci SFT mix runs on a fraction of the GPU-days the Tülu 3 SFT took, and a 4× RL speedup means the RLVR stage is no longer the cost bottleneck.

For ch-34's stance table, this puts OLMo 3's stabilizer column at **infrastructure**: not a new optimizer, not a new architecture trick, not a new reward shape — a faster trainer and better batching.

---

## Why the model-flow stance matters for learning

From the source (lines 62-65):

> ### Why OLMo 3 matters
> - It is one of the clearest public examples that **openness can apply to training trajectories**, not only to final checkpoints.
> - For a learner, it is unusually valuable because you can study where a capability was added: base, mid-training, long-context, DPO, or RLVR.

This is the attribution claim. If OLMo 3-Think 32B beats OLMo 3-Instruct 32B on MATH by X pp, and all else is held fixed except the Think-branch SFT mix, then the MATH delta is *causally attributable* to the Think-branch mix. That is the kind of stage-level attribution ch-34's lab comparison table would like to do for every lab, but only OLMo 3 makes it directly feasible.

---

## What OLMo 3 does not disclose

- Dolci per-mix sizes (SFT prompts, DPO pairs, RLVR prompts).
- Per-branch SFT LR and DPO β (assumed Tülu / OLMo 2 defaults).
- GRPO vs PPO choice at the RLVR stage (raw-data calls it RLVR but does not re-commit to PPO).
- Branch-diff numbers on the benchmarks — the report frames the branch structure, not the full delta table.

---

## Connections

- `[[olmo-3]]` — raw source.
- `[[ch-34]]` — §4 uses this for the model-flow stance.
- `[[olmo-2]]` — predecessor; OLMo 3 extends openness from data + recipe to data + recipe + stages + branches.
- `[[tulu-3]]` — the per-branch post-training recipe.
- `[[dolma]]` — Dolma 3 is the successor pretraining corpus.
- `[[allen-ai]]` — lab-level worldview.
- `[[deepseek-r1]]` — RL-Zero-style research that OLMo 3's RL Zero branch enables.
