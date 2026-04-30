---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-instruct.md
source_url: https://arxiv.org/abs/2212.10560
created_at: "2026-04-23"
---

# Excerpt: Self-Instruct — the breadth-generator loop ch-29 inherits

**Source library:** `wiki/raw-data/llm-training/papers/self-instruct.md`
**Artifact:** 4-step pipeline (seed, instruction gen, instance gen, filter)

---

## Why this source anchors ch-29

The synthetic track's capstone needs two generators that together cover both *breadth* (novel tasks) and *depth* (complexity). Self-Instruct is the breadth generator. Without the 175-seed / 8-ICL / ROUGE-L-0.7 recipe, Evol-Instruct's depth operators evolve the same small set of topics endlessly and the pool collapses. The lab's 70/30 Self-Instruct/Evol-Instruct mix is exactly the ratio that preserves the complexity tail [[evol-instruct]] documents without forfeiting breadth.

---

## The attested 4-step pipeline — what ch-29 implements verbatim

From the source (lines 31–39):

1. **Instruction generation** — prompt the LM with 8 in-context examples (6 from seed, 2 from prior accepted) and ask for a new task instruction.
2. **Classification-vs-non-classification branching** — ask the LM whether the instruction is classification; this changes the instance-generation prompt template (input-first for classification to avoid label bias, output-first otherwise).
3. **Instance generation** — for each accepted instruction, prompt the LM to produce `(input, output)`.
4. **Filtering** — drop instructions with ROUGE-L > 0.7 to any existing instruction (diversity filter); drop instances where `input == output`; drop if instruction contains "image/graph/file"; drop ill-formatted generations.

The prompt template (source lines 44–51) is quoted verbatim in ch-29's `build_instruction_prompt`:

```
Come up with a series of tasks:
Task 1: <seed 1>
Task 2: <seed 2>
...
Task 8: <seed 8>
Task 9:
```

Ch-29's `stop=["\nTask"]` is the same stop sequence implied by this template.

---

## What ch-29 keeps, changes, drops from Self-Instruct

| Self-Instruct default | Ch-29 choice | Reason |
|-----------------------|--------------|--------|
| 175 seed tasks | 175 seeds (unchanged) | the count is attested; reducing the seed pool collapses breadth |
| 6 from seed + 2 from recent | same | this ratio is the attested recipe; the 2 recent keep the generator coupled to the evolving pool |
| ROUGE-L > 0.7 diversity filter | Replaced by MinHash-LSH at J=0.8 | MinHash is the standard for near-duplicate detection at > 10K scale; ROUGE-L is O(N²) |
| `(input, output)` per instruction | same | the two-field output shape makes SFT loss masking trivial |
| 52K/82K instructions/instances at the end | 5K/5K target | ch-29 is a lab, not a dataset-release paper |
| No complexity axis | Mixed 70/30 with Evol-Instruct | Self-Instruct's flat complexity distribution is the weakness [[evol-instruct]] Figure 1 attacks |

---

## The one explicit failure mode the source documents

From the source (line 52):

> Failure modes observed: hallucinated "impossible" tasks, output-bias in classification, repetition — addressed via the diversity filter.

Ch-29 instantiates all three mitigations:
- "Hallucinated impossible tasks" → the `image/graph/file` keyword filter in `format_valid`.
- "Output-bias in classification" → the input-first template branch, preserved in `generate_instance`.
- "Repetition" → MinHash dedup (replacing ROUGE-L at scale).

---

## Why 175 seeds specifically

The paper does not claim 175 is optimal; the authors chose it to cover classification, generation, extraction, and open-ended within a budget a team could hand-write in one sitting. Ch-29 reuses the public seed file rather than rewriting: the win-condition is not novel seeds, it is the cascade's behaviour.

---

## Connections to the rest of the track

- **ch-19** — the full-read chapter on [[self-instruct]]; read that before this lab.
- **ch-20** — [[evol-instruct]] extends this pipeline along the complexity axis; ch-29 uses them together.
- **ch-23** — [[cherry-llm]] / [[ifd]] is the filter that specifically addresses "hallucinated instruction-response mismatch" — the residual failure mode after Self-Instruct's own diversity filter.
- **ch-25** — MinHash is the scalable replacement for ROUGE-L 0.7 at ch-29's pool sizes.
