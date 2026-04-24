---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/needle-in-haystack-data.md
source_url: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
created_at: "2026-04-23"
---

# Excerpt: NIAH — the minimum viable harness, and what it gets wrong

**Source library:** `wiki/raw-data/llm-training/papers/needle-in-haystack-data.md`
**Artifact:** Greg Kamradt's Nov 2023 Needle-in-a-Haystack test. Simplest useful long-context eval: hide one sentence in essay filler, vary depth × length, visualize a 2D heatmap. Community-adopted, saturated for 2025 frontier models, now the ancestor of RULER and BABILong.

---

## Why this source grounds §4 (matchers) and §5 (slicing) of ch-47

Ch-47 §4 opens with the simplest matcher family — exact / substring — and names NIAH as the canonical example. Ch-47 §5 argues slicing is mandatory; NIAH's depth × length heatmap is the minimum viable slicing surface. The source is valuable to this chapter because it is both the **template** for long-context harness design and the **cautionary tale** of a harness whose aggregate number overstates capability.

---

## The one-line harness specification

Source §Synthesis pipeline / Original NIAH:

> **Filler:** Paul Graham essays (publicly available, mostly absent from post-2021 training cutoffs).
> **Needle:** a single sentence stating a specific fact (the "best thing in SF" sentence, or user-configurable).
> **Injection:** inserted at programmatic depths (e.g., 0%, 10%, 20%, …, 100% of filler).
> **Context lengths tested:** 1K, 4K, 8K, 16K, 32K, 64K, 128K, … up to the model's claimed max.
> **Evaluation prompt:** "What is the best thing to do in San Francisco?" — scored against gold answer.

Notice: the harness fits in six lines. This is its power — zero dependencies, trivial reproducibility, direct visual output. Ch-47 §2 attributes to NIAH the property that a harness can be "a generator plus a matcher"; NIAH is the proof that the floor is low.

But the six lines also expose every decision ch-47 §3 and §4 surface: `depth` grid is a slicing coordinate; `context length` is a slicing coordinate; `filler=PG essays` is a generator knob with attested contamination risk (source §Risks); `evaluation prompt` is the inference_config; matcher is exact-substring by default or LLM-judge by toggle.

---

## Exact-substring vs LLM-judge matcher — §4 matcher choice

Source §Modality-specific technical details:

> **Evaluation metric:** exact-substring match (simple) or LLM-judge (lenient).

Notice the parenthetical labels — "simple" vs "lenient." Ch-47 §4 expands this into a tradeoff: exact-substring is cheap and brittle (paraphrase fails it); LLM-judge is forgiving but brings the position/verbosity/self-enhancement biases [[judge-llm-bias]] quantifies. There is no free lunch — the source makes this explicit by listing both matchers without recommending one.

In practice, the community moved from "exact" to "LLM-judge with a strict rubric" between 2023 and 2025, and reported scores rose 3–8 points. That is a matcher-drift version bump in everything but name; ch-47 §6 insists it must be named.

---

## The heatmap — §5 slicing as the default report

Source §Key Contributions:

> **Heatmap visualization** (context length × needle depth × accuracy).

Ch-47 §5 argues aggregate numbers hide hacks; NIAH's native report is a **heatmap**, not a mean. That is the slicing discipline ch-47 wants — every cell visible, failure patches findable. A model may hit 99% average but fail systematically at `depth ≈ 40% and length ≈ 64K`; the heatmap surfaces it immediately, the mean hides it.

Notice: the community still usually reports NIAH as a single percentage. This is a loss of information the original test did *not* impose. The harness gives you the heatmap; the culture compresses it.

---

## Over-reporting — the canonical §5 cautionary tale

Source §Quality / diversity evaluation:

> Single NIAH: saturated for 2025 models — most score >95% at claimed context.
> Multi-needle (8-needle): strong discriminator in 2025; Claude-3.5 ~90%, Llama-3.1-70B ~70%, weaker open models <50%.
> NIAH is **not** a proxy for real long-context reasoning — models can pass NIAH at 128K while failing multi-hop reasoning at 32K.

Notice the gap: 95% single-NIAH and <50% multi-needle on the same model. The single number is misleading; the slice set is honest. Ch-47 §5 attributes to this observation the rule that **slicing is mandatory**. A harness that reports only the mean is an artefact; a harness that reports the grid is a measurement.

The "1M context" marketing claim that ch-47 §6 warns against is anchored in this source's §Risks: "'1M context' claims based only on NIAH are misleading."

---

## Filler contamination — the §6 versioning gotcha

Source §Risks + gotchas:

> **Paul Graham contamination:** many models have PG essays in training — canary filler choice matters.

Ch-47 §6 lists three drift sources; filler-change falls under "Data refresh / contamination fix." Swapping filler from PG essays to a 2024+ blog corpus is not a cosmetic change — it can drop scores 5–10 points on models whose retrieval leaned on having-seen-it-before. The harness version must name the filler.

---

## Fair-comparison clause — §6 release discipline

Source §Risks + gotchas:

> **Fair comparison requires identical filler + prompts** — different NIAH implementations give different numbers.

This is the clearest attestation in ch-47's reading list for the §6 rule "never cross-cite numbers between harnesses without naming the harness." The source names even *same-named* NIAH implementations as non-comparable. Ch-47 sharpens this into "harness_version coordinate is part of the tuple, always."

---

## What ch-47 keeps, changes, drops from NIAH

| NIAH design choice | Ch-47 normative claim | Reason |
|---|---|---|
| 6-line harness spec | A harness is generator + matcher | §2 minimum definition |
| Exact-substring default | Substring is cheap, brittle, honest about what it misses | §4 matcher family |
| LLM-judge toggle | Judge-matchers have known biases; pin rubric | §4 + [[judge-llm-bias]] |
| Depth × length heatmap | Slicing is mandatory, not optional | §5 |
| PG-essay filler | Filler is a version coordinate | §6 |
| Same-named-NIAH incomparable | Cross-harness comparison needs full version tuple | §6 |
| Saturated at frontier | Harness must evolve; see BFCL V1→V4 | §6 + [[excerpts/bfcl]] |

---

## Connections

- **[[ch-47]]** — this excerpt grounds §2 (minimum harness), §4 (exact matcher), §5 (heatmap as slice), §6 (filler as version coordinate).
- **[[excerpts/ruler]]** — RULER expands NIAH into 13 tasks with orthogonal knobs; the direct successor.
- **[[babilong]] (raw-data)** — BABILong hybridises NIAH with symbolic reasoning; retrieval + reasoning slices.
- **[[longbench]] (raw-data)** — natural-task complement; different matcher zoo.
- **[[ch-50]]** (downstream) — long-context harness deep-dive; NIAH is the entry point.
