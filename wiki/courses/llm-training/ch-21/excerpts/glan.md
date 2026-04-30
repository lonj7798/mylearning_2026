---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/glan.md
source_url: https://arxiv.org/abs/2402.13064
created_at: "2026-04-23"
---

# Excerpt: GLAN — the taxonomy-as-seed paradigm

**Source library:** `wiki/raw-data/llm-training/papers/glan.md`
**Authors:** Li, Dong, Tang, Wang, et al. (Microsoft Research) — 2024.

---

## Why this source anchors ch-21 §1–§2, §5

GLAN is the cleanest statement of the top-down paradigm. It is also the only paper among the ch-21 primary sources that is explicitly about instruction tuning (not pretraining), which makes it the right reference point for comparing to Self-Instruct and bottom-up siblings from ch-19.

The three claims ch-21 pulls directly from GLAN:

1. **Taxonomy is a third paradigm.** "Prior instruction-tuning data synthesis (Self-Instruct, Evol-Instruct) is seeded by a small real instruction pool and inherits its biases. GLAN replaces the seed with a **taxonomy**." This is the seed / no-seed / taxonomy trichotomy ch-21 §1 opens with.
2. **Tree depth is the coverage knob.** "Ablations on taxonomy depth confirm deeper trees give flatter capability distributions." The interactive companion makes this knob physical.
3. **Teacher bias concentrates at the top.** "GPT-4's view of what constitutes a field/discipline shapes the entire downstream corpus." Ch-21 §5 quotes this directly for the curator-bias discussion.

---

## The six-level tree — reconstructed

The paper's named hierarchy (with the level-name vocabulary it actually uses):

```
Level 0 — Field          hand-curated, ~36 entries
Level 1 — Subfield       GPT-4 decomposition per field
Level 2 — Discipline     GPT-4 decomposition per subfield
Level 3 — Subject        GPT-4 enumeration per discipline
Level 4 — Session        GPT-4 syllabus per subject (with learning objectives)
Level 5 — Concept        GPT-4 concept-list per session
Level 6 — Instruction    generation at the leaf (with varied difficulty)
```

From the source:

> Each discipline gets an auto-generated subject list; each subject gets an auto-generated syllabus of class sessions; each class session is enumerated as a concept list. Instructions are generated at the concept level, guaranteeing coverage across all branches.

The one-line summary ch-21 uses: *"Field → Subfield → Discipline → Subject → Session → Concept → Instruction."*

---

## Why levels 0–1 are the bias-injection point

From the source (Risks + gotchas):

> **Teacher-bias concentrates at the top:** GPT-4's view of what constitutes a field/discipline shapes the entire downstream corpus.

Concretely: the root list decides what kinds of knowledge exist. If the root list has "Mathematics" but not "Applied Engineering Ethics," no downstream node can generate ethics content. If the root list has "Computer Science" as one item, CS gets one root slice; if the root list had "Computer Science" and also "Computational Linguistics" and "Information Theory" as separate roots, those sub-areas get their own full subtree instead of being sub-branches of CS.

The level-1 Subfield choice is almost as important. Whether "Machine Learning" is placed as a Subfield under Computer Science (inherits CS-style Discipline children: theory, systems, etc.) or under Mathematics (inherits math-style Discipline children: statistics, optimization) changes tens of thousands of leaves.

Ch-21 §5's "hand-curated at the top" claim is this observation generalized across Phi-1.5, Phi-4, Nemotron, and Cosmopedia.

---

## What "generate at the concept level" looks like

The source's step 5:

> **Concept → Instructions:** for each concept, prompt for instruction-response pairs at varied difficulty levels; include verification-friendly answer formats for math/code.

Two details worth highlighting:

- **Varied difficulty per concept.** GLAN asks for multiple instances per concept at different difficulty settings — one easy, one medium, one hard. This is the within-leaf diversity knob; it does not add branches but it does thicken the sampling of each branch.
- **Verification-friendly formats.** For math and code concepts, GLAN asks the teacher to emit answers in a format the authors can later verify automatically (final numeric answer boxed, code that returns a value). This is the "verify" step of the ch-18 loop specialized to leaf generation.

---

## The coverage audit result

From the source:

> Mistral-7B + GLAN outperforms same-base models fine-tuned on Alpaca / WizardLM / CodeAlpaca on MATH, GSM8K, HumanEval, MBPP, BBH, ARC, MMLU.
> No task-specific data used — generalization attributed to coverage.

This is the pay-off of the coverage guarantee. Every listed benchmark probes a named part of the tree. GLAN spans the union of those parts and Alpaca / WizardLM / CodeAlpaca do not. The null-hypothesis explanation — that Mistral-7B simply trained on more data — is ruled out by same-token-count ablations the paper runs.

The finding ch-21 extracts: **coverage is worth more than volume at fixed compute**, provided the coverage is constructed and not just statistical.

---

## The "add a capability = add a subtree" claim

From the source:

> Demonstrated fine-grained coverage control: adding a taxonomy node adds a capability.

This is the clearest operational benefit of top-down synthesis. If you want your model to handle, say, organic chemistry retrosynthesis, you do not need to collect user logs or seed examples. You:

1. Add "Organic Chemistry Retrosynthesis" as a Subject under Chemistry.
2. Ask the teacher to write a syllabus of sessions for that subject.
3. Ask for concept lists per session.
4. Generate leaves.

The cost scales linearly with the size of the added subtree, not with the curation of new examples. This property is why taxonomy-driven synthesis is attractive for labs that want to add capabilities incrementally without rebuilding a data pipeline each time — see Phi-4's "50 synthetic categories" as a flat-taxonomy variant of the same idea.

---

## Connections

- [[excerpts/phi-1-5]] — the 20K-topic list is a flat (level-1) precursor of GLAN's tree.
- [[excerpts/nemotron-4-synthetic]] — different shape (task families × RM filter) but same top-down move.
- [[excerpts/cosmopedia]] — the HF reproduction uses a mixed-source taxonomy (curated + web-cluster-derived) as an explicit audit against single-curator bias.
- [[excerpts/mathscale]] — the seed-derived concept graph is the complement — taxonomy extracted from data rather than pre-curated.
- [[ch-18]] — the ch-18 loop's "generate" step is specialized by GLAN to "traverse-tree-and-generate-at-leaf."
- [[ch-21]] §2 and §5.
