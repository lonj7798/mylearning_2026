---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3 — Dolma 3, Dolmino, Longmino: the stage-curriculum maximum-disclosure release

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Paper:** Team Olmo 2025, "Olmo 3" (Allen AI).

---

## Why this source anchors ch-09 §4 and §6

OLMo 3 is the 2025 open-data maximum-disclosure reference. Where [[llama-3]] hides and [[qwen-3]] partially discloses, OLMo 3 publishes the full *curriculum* — four named datasets, three stage budgets, data-lineage tooling (OlmoTrace), and every stage's purpose. For ch-09's four-axis framework, OLMo 3 is the "stage-level granularity, fully disclosed" row; it's the counter-movement to the frontier-lab silence.

This excerpt walks through the Dolma 3 / Dolma 3 Mix / Dolmino / Longmino curriculum and explains why each stage exists as a separately-budgeted dataset.

---

## The "model flow is the release" thesis

From the source (lines 7-8):

> - **Core Insight:** The real scientific artifact is not just the final model weights but the entire model flow: pretraining stages, intermediate checkpoints, datasets, post-training branches, evals, and tooling.
> - **Guideline:** If you want a training corpus for research rather than just deployment, study OLMo 3 as a model-flow release: broad pretraining, targeted mid-training, long-context extension, then separate SFT/DPO/RLVR branches for instruct, think, and RL-zero pathways.

OLMo 3's thesis makes the **pretraining data a first-class release artifact**, not a side-note. "The full curriculum is the release" means: if you want to reproduce the base model, you need all four datasets (Dolma 3, Dolma 3 Mix, Dolmino, Longmino), in their correct stage sequence, with the correct transition points. The release bundle includes all of this.

For ch-09 §6's four-axis framework, OLMo 3 is a vertical column of *everything you want to know*: token counts disclosed, per-source composition disclosed, stage budgets disclosed, licence manifest published, tooling released.

---

## The four-dataset curriculum

From the source (lines 44-49):

> ### Data curriculum
> - **Dolma 3:** about **9.3T** source tokens spanning web pages, science PDFs processed with `olmOCR`, code, math problems/solutions, and encyclopedic text.
> - **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens with stronger math/code emphasis and stronger decontamination.
> - **Dolma 3 Dolmino:** **100B** mid-training tokens sampled from a ~2.2T high-quality pool for math, science, code, instruction following, and reading comprehension.
> - **Dolma 3 Longmino:** about **50B** long-context tokens from a **639B**-token pool of long documents plus mid-training data.
> - **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

Each name maps to a stage:

| Dataset | Size | Stage | Role |
|---|---|---|---|
| **Dolma 3** | 9.3 T source | upstream pool | The unfiltered/lightly-filtered source pool; successor to Dolma 1.7. Research reference. |
| **Dolma 3 Mix** | 5.9 T | **Stage 1 pretraining** | Filtered pretraining mix with math/code emphasis and stronger decontamination. |
| **Dolmino** | 100 B | **Stage 2 mid-training** | Sampled from a 2.2 T high-quality pool for math/science/code/instructions/reading-comprehension. |
| **Longmino** | 50 B | **Stage 3 long-context** | Long documents + mid-training data for context-length extension. |
| **Dolci** | (post-training) | SFT, DPO, RLVR | Separate mixes for each post-training stage. |

Each of these is a **separately-released, separately-documented** dataset. Contrast with [[llama-3]] (one undisclosed 15.6T blob) and [[qwen-3]] (30T + 5T + LC stage budgets without named datasets).

---

## The Dolma 3 → Dolma 3 Mix compression

Dolma 3 has 9.3T source tokens; Dolma 3 Mix has 5.9T pretraining tokens. The 3.4T delta is what gets filtered out of Dolma 3 in producing the Mix. This is explicit disclosure of the *filter strength*: ~36% of the source pool is removed in producing the pretraining corpus.

Compare to [[fineweb]]-Edu: 15T FineWeb → 1.3T FineWeb-Edu (~92% removed). OLMo 3's Dolma 3 Mix is a much less aggressive filter than FineWeb-Edu — presumably because OLMo 3's Mix is the *bulk* slice (not a curriculum-end slice), and the aggressive classifier filtering is reserved for the Dolmino mid-training stage.

The Dolma 3 → Dolma 3 Mix step includes:

- Stronger decontamination vs Dolma 1.7 (explicit claim; threshold not quantified).
- Stronger math/code emphasis (meaning either up-weighting of math/code slices or adding more math/code sources).
- Standard Dolma six-stage filter cascade (inherited from [[dolma]] v1.7).

---

## Dolmino — the 100B mid-training budget

Dolmino is **100 B tokens sampled from a ~2.2 T high-quality pool**. This is the curriculum-end analogue of FineWeb-Edu: a much-smaller high-quality slice used for a *mid-training* stage, not for Stage 1.

Why 100 B and not more? This is the same question as "why is FineWeb-Edu's useful training share 1.3 T?" — the high-quality pool *is* small, because high-quality data is rare. The 2.2 T pool is Allen AI's internal reference for "text we are very confident is educational/useful," and 100 B is a Chinchilla-adjacent budget for a mid-training stage on a 7B-32B model.

Dolmino's composition target (from the source):

- Math
- Science
- Code
- Instruction following
- Reading comprehension

These are the *capability* categories, not source categories. Dolmino is organized around what you want the model to *learn* at this stage, not where the tokens came from. For ch-09's reader: this is the 2025-era mid-training design pattern — capability-keyed mix selection, not source-keyed.

---

## Longmino — the 50B long-context stage

Longmino is **50 B tokens drawn from a 639 B pool of long documents plus mid-training data**. Stage 3's entire budget is ~0.8% of Stage 1's budget.

Why so small? Because long-context training is about *context length*, not total tokens. Training at 32K context on 50B tokens means ~1.5M long-document examples, which is plenty to teach a model the attention pattern for long context. The 639B pool is the eligibility set (all long documents that *could* be used); 50B is the actual training budget. This disclosure — pool size vs training size — is the kind of detail Llama 3 and Qwen3 do not publish.

For ch-09 §1's "what CC omits" discussion: long-document content is systematically under-represented in web corpora (CC truncates large documents, and Wikipedia articles are mostly short). The 639B long-document pool requires *active sourcing* — academic papers, books, legal documents — because the web substrate is short-document-biased.

---

## The OlmoTrace tooling — data lineage disclosed

From the source (line 23):

> - Couples model release with tooling: **Olmo-core**, **Open Instruct**, **OLMES**, **OlmoTrace**, decontamination, and dedup utilities.

OlmoTrace is the data-lineage component. From the Allen AI public description: every token in a released OLMo 3 checkpoint can be traced back to its source document through the training-step logs. This is the operational form of ch-11's lineage-tracking topic, and it is the single disclosure capability that Meta, Alibaba, and DeepSeek explicitly do *not* publish.

For ch-09 §5's licence-governance discussion: OlmoTrace is what an opt-out register interoperates with. A creator's request "remove my content from model X" is only actionable if the lab can *identify* whether the content was included. OLMo 3 can answer that question publicly; closed labs cannot (or choose not to).

---

## The 1M-GPU-hour budget and what it buys

From the source (lines 56-59):

> ### Efficiency and infrastructure
> - Pretraining used up to **1,024 H100 GPUs**.
> - Mid-training used **128 H100 GPUs**.
> - Post-training used **256 H100 GPUs**.
> - Moving SFT from **Open Instruct** to **Olmo Core** reportedly improved throughput by **8x**.
> - In-flight weight updates, continuous batching, and threading work made RL training about **4x** more efficient.

Rough total: ~1M H100-hours for the full flow (derivation in [[excerpts/olmo-3]] of ch-06 for the detailed math). For ch-09's point about open-vs-closed data disclosure: **1M GPU-hours is ~5% of Meta's Llama 3 405B pretraining budget**. An open lab can do a full-flow, fully-disclosed release at ~5% of a frontier closed-lab's pretraining compute.

This matters for the disclosure argument: the open community can afford to be transparent because the model-flow-as-artifact release is affordable at academic-lab budgets. OLMo 3 at 7B-32B scale is a choice about where to spend compute (on disclosure infrastructure and multi-stage curricula) rather than a capability of unlimited resource.

---

## The four-axis framework, OLMo 3 row

1. **Token count**: Dolma 3 (9.3 T source), Dolma 3 Mix (5.9 T pretraining), Dolmino (100 B), Longmino (50 B). All disclosed.
2. **Composition**: stronger math/code emphasis in Mix; capability-keyed mix in Dolmino; long-document selection in Longmino. Source-level composition also available through the Dolma 3 dataset documentation.
3. **Licence regime**: per-source manifest published. Permissive-licensed code; Gutenberg for books; CC for web; peS2o for academic. OlmoTrace makes removals actionable.
4. **Disclosure granularity**: stage-by-stage budgets + per-stage capability purpose + tooling release. This is the ceiling.

Compare to the other rows in ch-09 §2's table — the OLMo 3 row is the row everyone else is measured against.

---

## What to take from OLMo 3 for ch-09

1. **The full curriculum is the release.** Four separately-documented datasets (Dolma 3, Dolma 3 Mix, Dolmino, Longmino) for one base model.
2. **Stage budgets disclosed at all scales.** 5.9 T + 100 B + 50 B. The 40× and 20× drops between stages are visible.
3. **Capability-keyed mix at mid-training.** Dolmino targets math/science/code/instructions/reading-comprehension, not source categories.
4. **OlmoTrace enables opt-out actionability.** Lineage tracking is what makes governance credible.
5. **Open disclosure is affordable at ~5% frontier-lab budgets.** The 1M-GPU-hour budget funds the tooling and the transparency.

---

## Connections

- [[excerpts/the-pile]] — the 2020 hand-curated baseline; OLMo 3 generalizes the "explicit mixture, documented" discipline to a staged curriculum.
- [[excerpts/dolma]] — the predecessor; Dolma 1.7 is the ancestor of Dolma 3, and the six-stage cascade is inherited.
- [[excerpts/fineweb]] — the complementary classifier-filtered web corpus; OLMo 3's Dolmino uses FineWeb-Edu-style classifier filtering at a similar granularity.
- [[excerpts/llama-3]] — the polar opposite; Llama 3 discloses token total only, OLMo 3 discloses the full curriculum.
- [[excerpts/qwen-3]] — the partial-disclosure mid-point; stage budgets disclosed but within-stage composition hidden.
- [[ch-09]] — §2 (Dolma 3 Mix comparison-table row), §4 (disclosure counter-movement), §6 (four-axis maximum-disclosure), §7 (staged curriculum as the modern shape of a pretraining corpus).
- [[ch-10]] — the Dolma 3 six-stage cascade details are the spine of the next chapter.
