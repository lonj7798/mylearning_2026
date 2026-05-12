<!-- chapter: ch-19 — discuss transcript
     deps: [[summary]], [[qa]], [[qa-deep]], [[read]]
     scope: discuss-phase exchange, condensed for verdict record
-->

# Ch-19 — Discuss Transcript (Micro-Discuss / Option A)

Single application probe at learner request. Probe: design a synthetic-data pipeline for code generation given 500 anchor (problem, solution) pairs, GPT-4 API budget, open-weight aligned model + GPU, Python sandbox.

---

## Stage 1 — Generate (composition)

Learner composed 3 methods with explicit role assignments:

1. **Humpback** — provide code (from anchor) → extract instruction. Used GPT-4 + open-weight model simultaneously for instruction-inference diversity (per Q14's k=3 fix).
2. **Evol-Instruct** — listed all 5 In-Depth operators (Add constraints / Deepening / Concretizing / Increased reasoning steps / Complicate input); GPT-4 or open-weight model solves the modified problems.
3. **Bootstrap** — expand instructions from seeds; LLMs solve.

Teacher pushback: used *generic* Evol-Instruct operators rather than WizardCoder's 5 code-native operators. Chapter line 129 specifically warns generic Evol wastes mass on operators that don't touch code-specific failure modes.

Not used: Persona, Magpie, WRAP-style rephrase. Learner correctly flagged the consequence later (see Failure mode).

## Stage 4 — Verify

Learner identified two ceiling-breaking verifiers from Q6:
- **Sandbox execution**: run code, compare output to expected. APIGen canonical pattern.
- **LLM judge with gold/seed reference**: comparison against ground truth (Q6 row 6 = ceiling-breaking).

Both connect to ground truth. Independent of generator. ✅ Correct application of Q6's principle.

## Stage 3 — Dedup

Learner's answer: ROUGE-L on instructions (correct, matches Q5); decided code dedup matters less than instruction dedup.

Teacher nudge: missing semantic code dedup axis (same algorithm, different variable names → surface dedup misses). AST-based or execution-trace dedup catches semantic duplicates. Production-quality omission but not load-bearing for the probe.

## Stage 5 — Select

Learner: 2-dimensional selection — classify by difficulty + topic diversity, distribute equally across both axes. Maps cleanly to Evol's In-Depth × In-Breadth framing (Q4) plus WizardMath's bidirectional spectrum-smoothing insight (line 115). ✅ Strong, framework-correct answer.

## Failure mode (the single biggest risk)

> *"securing the diversity... the strategy what I pick is expanding the data from the seed, so that the generated synthetic dataset may limited into certain type of questions."*

✅ **Correctly diagnosed.** The composed pipeline (Humpback + Evolve + Bootstrap) is entirely seed-amplification. None adds new topic coverage. If 500 seeds don't cover topic X, synthetic data won't either.

Implicit mitigation: add Persona (Q9 amplifier) or Magpie (extract beyond seed prior). Learner is aware of this from earlier qa work.

## Bonus — ceiling question

Learner named two ceiling-breakers (gold-reference + cross-model voting), then **independently invented a third pattern**:

> *"how about generate the instruction with solid answer. and then generate the code until get that answer with instruction. this may like pass@k. also we can use this pass@k as a difficulty metrics."*

The pass@k pattern combines three orthogonal signals in one mechanism:
- Per-attempt pass/fail = Stage 4 ground-truth verifier (ceiling-breaking)
- K-attempt aggregate = Q6 multi-sample-style noise reduction
- pass@k value = Stage 5 difficulty selector

This **independently reinvents rollout-pass-rate filtering** — a 2024-2025 frontier technique used in OmegaPRM (ch-22), RLVR/Math-Shepherd (ch-44), and Kimi-K2 / Tülu-3 RL. The framework from ch-18/19 produced the insight without seeing the chapters yet.

One minor slip: *"for bootstrap method, we can try to add llm as a judger to validate the data"* — same-LLM judge doesn't break ceiling (Q6 row 1 = correlated failures). Need cross-model or LLM+gold-reference. Caught by teacher; learner's other answers used the correct patterns.

---

## Verdict reasoning (pre-evaluator-skip)

Three criteria from probe spec:

1. ✅ **Application**: composed 3 methods with role assignments for a novel domain
2. ✅ **Diagnosis**: correctly identified diversity-ceiling failure mode of seed-bound composition
3. ✅ **Synthesis**: named two ceiling-breaking verifiers from Q6 + invented a third (pass@k as combined verifier/difficulty/selector)

Stage 3 (dedup) had a minor production-quality gap (semantic dedup) but framework-correct.

The pass@k insight is **above the bar** — it extends the framework to a problem the chapter didn't fully cover. Signature of internalized framework, not memorized facts.

Recommended verdict: **Mastery**.

Per learner direction: commit + push to course/llm-training branch only; **do not merge to main** this cycle. Move to ch-20.
