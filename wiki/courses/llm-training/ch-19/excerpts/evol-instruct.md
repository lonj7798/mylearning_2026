---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/evol-instruct.md
source_url: https://arxiv.org/abs/2304.12244
created_at: "2026-04-23"
---

# Excerpt: Evol-Instruct — five In-Depth operators plus one In-Breadth operator, verbatim

**Source library:** `wiki/raw-data/llm-training/papers/evol-instruct.md`
**Heritage:** Xu et al. 2023 (Microsoft) — ICLR 2024. Sits between [[excerpts/self-instruct]] (which flattens difficulty) and [[excerpts/wizardcoder]] / [[excerpts/wizardmath]] (which domain-specialize the operators).

---

## Why this source anchors ch-19

Ch-19 §3 introduces the *complexity axis* — the fact that training-data difficulty, not just volume or topic diversity, is a first-class knob. Evol-Instruct is the first paper to name that axis, build an operator set that targets it, and show empirically that the complexity histogram shape (long tail vs flat) predicts downstream ability. Every subsequent synthetic SFT paper either applies Evol-Instruct directly, adapts its operators (WizardCoder, WizardMath), or explicitly argues against it (LIMA's curated-simplicity counter-bet). You cannot skip it.

---

## The six operators — verbatim with the paper's exact phrasing

From the source file:

> *In-Depth Evolving* (make the instruction harder):
> 1. **Add constraints** — impose an extra condition that the response must satisfy.
> 2. **Deepening** — increase the depth and breadth of a question.
> 3. **Concretizing** — replace general concepts with more specific ones.
> 4. **Increased reasoning steps** — explicitly request more reasoning steps to solve the task.
> 5. **Complicate input** — add complexity to the input itself (e.g., code, table, nested structure).
>
> *In-Breadth Evolving* (make the instruction more diverse):
> 6. **Mutation to a new instruction** in a rarer domain or long-tail topic.

Five-plus-one is deliberate. The five In-Depth operators move along one axis (difficulty); the single In-Breadth operator moves along an orthogonal axis (topic). Mixing them in the training pool requires both — pure In-Depth collapses onto a narrow topic bin, pure In-Breadth leaves the difficulty profile flat.

---

## What each operator actually does to a seed — one worked example

Take the Alpaca seed *"Write a short poem about autumn."* The five In-Depth operators produce:

1. **Add constraints** → "Write a short poem about autumn. The poem must be exactly 14 lines, use iambic pentameter, and include at least one reference to migrating birds."
2. **Deepening** → "Write a short poem about autumn that explores both the sensory experience (smell of leaves, crispness of air) and the psychological one (end-of-year introspection, anticipatory melancholy). Contrast at least three of these dimensions."
3. **Concretizing** → "Write a short poem about autumn in the Catskills mountains of New York, specifically during the second week of October 1987, from the perspective of a retired librarian walking her terrier."
4. **Increased reasoning steps** → "Write a short poem about autumn. First, list five images associated with the season. Second, categorize each as sensory, emotional, or philosophical. Third, select three images with maximum dimensional spread. Fourth, compose a stanza for each. Fifth, weave the stanzas into a unified poem."
5. **Complicate input** → "Given the following data structure, write a short poem about autumn:\n```json\n{'season':'autumn', 'location':'Vermont', 'mood':'melancholy', 'constraints':{'lines':12,'rhyme':'ABAB'}}\n```"

The In-Breadth mutation would produce a *new* instruction like *"Write a short technical explanation of why deciduous trees change color in autumn, suitable for a 10-year-old."* — same topic domain, different task type.

Each operator lands the output in a *different* region of instruction space. Running all six on one seed and then sampling produces the characteristic long-tail histogram in the paper's Figure 3.

---

## The elimination step — the four rejection rules

The source is precise:

> **Elimination step**: drop evolutions that (a) fail the LLM's own "same-or-similar" check against the input, (b) contain "sorry" / refusal markers indicating the LLM couldn't evolve, (c) have punctuation-only outputs, or (d) copy the input verbatim.

Three details worth noting.

**Rule (a) uses the teacher as its own judge.** Evol-Instruct prompts the same LLM that did the evolution to check whether the output is "same or similar" to the input. This is the first widespread use of LLM-as-judge inside a data pipeline, predating the LLM-as-judge evaluation literature by a year. The rationale: the teacher has the best-calibrated judgment about its own outputs; a cheap external classifier would miss semantic-equivalence cases that an LLM flags trivially.

**Rule (b) flags teacher refusal.** If the teacher can't evolve — because the evolved instruction would be harmful, or because it genuinely exceeds the teacher's competence — it produces a "sorry, I cannot" reply. Dropping these prevents the training set from inheriting refusal patterns as if they were responses. Later pipelines (Magpie, WizardCoder) all import this filter.

**Rule (d) catches the "no-op operator" failure.** When concretizing is applied to an already-concrete instruction, the teacher sometimes copies the input verbatim. The verbatim-copy filter exists because empirically ~5–10% of evolutions are no-ops per round.

---

## The four-round pipeline and ~250K yield

The source:

> 1. Seed with the 52K Alpaca instructions.
> 2. Apply one randomly chosen operator to each instruction via an LLM prompt; collect the evolved instruction.
> 3. Generate a response with the same LLM.
> 4. **Elimination step**: [four rules above].
> 5. Iterate 4 rounds → ~250K evolved instructions (after filtering).

Four rounds is load-bearing. After round 1 the complexity distribution is already more tail-heavy than Alpaca. After round 4 the distribution saturates — operator applications start failing the same-or-similar check because the instruction is already at the teacher's depth ceiling. Round 5 would show ~40% elimination rate with minimal complexity gain.

The 52K → 250K expansion is ~5× over four rounds. Not every seed survives every round; the surviving pool is a *different* set at each round, weighted toward seeds whose topic admits deep evolution (reasoning, technical writing) and against seeds whose topic saturates quickly (simple Q&A, greetings).

---

## Where Evol-Instruct's operators fail — and what domain-specialization fixes

The source flags:

> **Connections** — WizardMath / WizardCoder extend the recipe to math and code.

The reason the generic operators underperform on math/code is specific. "Increased reasoning steps" applied to a math problem pushes past the teacher's solve ceiling — the evolved instruction is harder than GPT-4 can verify, so the eliminator rule (a) drops it *correctly* but the training set loses the hardest slice. WizardMath's bidirectional evolution (downward + upward) sidesteps this: downward-evolved problems are always below the ceiling. "Complicate input" applied to code produces unparseable inputs that the teacher refuses; WizardCoder replaces it with "require specific library / language," which stays parseable.

The generic operator set is a *starting* point. In practice every serious pipeline customizes at least one operator per target domain.

---

## Connections

- [[excerpts/self-instruct]] — Evol-Instruct's 52K seed pool is Self-Instruct's output.
- [[excerpts/wizardcoder]] — code-domain operator specialization.
- [[excerpts/wizardmath]] — math-domain bidirectional specialization.
- [[excerpts/persona-hub]] — orthogonal diversity axis; the "who asks" complement to "how hard is it."
- [[ch-19]] — this excerpt is the foundation of §3 and §4 (specialization).
