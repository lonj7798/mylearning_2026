<!-- scope: reasoning-trace synthesis — concept-graph-driven math problem and solution generation
     deps: [[metamath]]
     see-also: [[mammoth-2]], [[openmathinstruct-2]]
-->

# MathScale: Scaling Instruction Tuning for Mathematical Reasoning
- **Core Insight:** Extract math **concepts** and **topics** from a seed corpus, build a knowledge graph over (concept × topic) pairs, and sample pairs to prompt a teacher into generating unbounded fresh problems — decouples scale from the finite pool of seed questions.
- **Guideline:** When scaling math SFT past the seed-problem ceiling, mine a concept/topic taxonomy and generate new problems by sampling edges of the graph; this produces genuine novelty, not rephrasing.
- **Authors:** Zhengyang Tang, Xingxing Zhang, Benyou Wang, Furu Wei (Microsoft Research / CUHK-SZ)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.02884
- **Relevant topics:** math reasoning, synthetic problem generation, concept taxonomy, SFT scaling

## Abstract
MathScale builds MathScaleQA, a 2M-example math SFT corpus created by first extracting math *topics* and *knowledge points* (concepts) from existing math-instruction datasets via GPT-3.5, constructing a concept graph where edges weight co-occurrence frequency, then sampling (topic, concept) pairs to prompt GPT-3.5 for fresh problems and their CoT solutions. MathScale-7B reaches 66.3% on the MwpBench it introduces, outperforming open baselines of similar size.

## Key Contributions
- **Concept-graph method** — a principled way to scale math data past the seed-problem pool.
- **MathScaleQA** (2M samples) publicly released.
- **MwpBench** — a new multi-source math word problem eval.
- Demonstrated genuinely novel problems: minimal overlap with GSM8K / MATH, measured by MinHash.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** existing math instruction corpora (GSM8K, MATH, MetaMathQA).
- **Concept extraction:** prompt GPT-3.5 for each seed problem: "List the math topics and knowledge points required." Aggregate into a vocabulary of ~2K topics × ~5K knowledge points.
- **Graph construction:** nodes = topics and concepts; edges weighted by co-occurrence across seed problems.
- **Sampling:** draw (topic, concept-set) pairs from the graph; pass to GPT-3.5 as a **problem-generation prompt** with instructions to author a fresh problem using the given topic and requiring the listed concepts.
- **Solution generation:** GPT-3.5 solves its own generated problem via CoT.
- **Filtering:**
  - Drop problems whose solution is invalid (solver fails parse, no answer).
  - Drop near-duplicates of seed problems via MinHash similarity threshold (Jaccard > 0.7 rejected).
  - Retain all solutions (no gold-answer check — the generator is both author and solver).
- **Output shape:** 2M (problem, CoT) pairs; trace length 200–800 tokens. No tool-integrated code.
- **Teacher:** GPT-3.5-turbo.
- **Cost / compute:** not precisely disclosed; order ~$20K in API.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** short CoT (~400 tokens avg).
- **Trace style:** standard CoT.
- **Correctness verifier:** NONE in the usual sense — the teacher generates both the problem and the answer, so "correctness" is self-consistency rather than gold-match. This is the chief weakness of the method.
- **Concept-graph sampling:** authors prioritize rare edges (less co-occurring concept pairs) to push distributional coverage.
- **Novelty audit:** ~5% of MathScaleQA overlaps (MinHash > 0.5) with GSM8K/MATH; 95% is genuinely new word problems.

## Quality / diversity evaluation
- MathScale-7B: +5-point gain on GSM8K and +3 on MATH over the same base trained on MetaMathQA alone.
- MwpBench eval shows balanced coverage across algebra, geometry, probability, number theory.
- Authors note diminishing returns past 1.5M — concept graph eventually saturates.

## Risks + gotchas
- **Unverified solutions:** teacher generates both problem and answer → systematic errors propagate; some fraction of "gold" answers are wrong.
- **Concept-graph bias:** topics underrepresented in seeds stay underrepresented downstream (geometry with diagrams is poorly served).
- Difficulty distribution skews easy-to-medium — teacher rarely invents olympiad-level problems unprompted.

## Connections
- Conceptual ancestor: [[self-instruct]] (task taxonomy → new task generation).
- Sibling augmentation methods: [[metamath]] (operator-based), [[mammoth-2]] (web-scale extraction), [[openmathinstruct-2]] (question-augmentation + strong teacher).
- Used downstream in [[wizardmath]]-style recipes and GLAN.
