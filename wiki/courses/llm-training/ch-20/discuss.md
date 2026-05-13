<!-- chapter: ch-20 — discuss transcript
     deps: [[summary]], [[qa]], [[qa-deep]], [[qa-deep-2]], [[read]]
     scope: discuss-phase exchange, condensed for verdict record
-->

# Ch-20 — Discuss Transcript (Micro-Discuss / Option A)

Single application probe at learner request. Probe: design a distillation pipeline for a *customer support reasoning agent* (5-10 turn dialogues, FAQ/policy retrieval, empathetic response). Target student: Qwen2.5-7B (later revised to Qwen-3.6-27B). Resources: teacher API budget, Python sandbox, 1K anchor conversations, FAQ/policy docs.

---

## Pre-stage — Student-first reasoning (unprompted)

Learner spontaneously inverted the probe order: *"before we choose the teacher model, we need to pick the student model first. ... if there are too big distribution gap, it may hard."*

✅ Axis 7 (distribution match) applied without prompting. Q11 internalization signal. Set student = Qwen-3.6-27B, then narrowed teacher to Qwen family.

## Stage 1 — Teacher choice

Initial: "Qwen family largest model" (implicit, no name). After push: **Qwen-3.6-Max** (same-family largest reasoning teacher). Distribution match (axis 7) primary justification.

Missing: axes 1 (capability) and 6 (trace length) not engaged. Single-axis justification rather than 3-dim optimization framing. Minor gap.

## Stage 2 — Generation methods + anchor

Learner declined explicit ch-19 method naming: *"we don't need to be that specific like mentioning the exact method, but know overall way is more important."*

Approach described:
- Intent classification per turn (small model)
- N-shot prompting with anchor seed → equivalent to **Bootstrap** (unnamed)
- Diverse conversation paths with length/difficulty variation → equivalent to **Evol** + possibly **Persona** (unnamed)

⚠ Method-naming reluctance — reversal from ch-19 verdict E1 strength (explicit composition with role assignments). Framework retention partial.

Multi-turn data generation **mechanism not addressed** — how customer LLM and agent LLM rollout sequentially, how scenarios stay coherent across turns. ch-25 long-conversation callback ([[project_llm_training_anchor_longconvo_callback]]) surfaces here but unaddressed.

**Anchor Job B initial confusion → corrected with extension**:
- First answer: "provide the real product data, golden answer" = *content-level use* of anchor in verifier (the same misframing flagged in ch-18 review)
- After correction: ✅ "use human-curated data and let pass the verifier ... if not, filter miscalibrated" = pipeline regression test (meta-level)
- ⭐ **Framework extension #1 — negative anchor**: *"we need some human curated data that should not pass the verifier too, then we can make a clear boundary."* Ch-18 line 133 only mentions positive-case calibration; learner self-invented the symmetric negative-anchor counterpart for true decision-boundary calibration.

## Stage 3 — Distillation strategy

Initial answer: *"both with thinking and without thinking. customer call should support real-time."* ✅ Correct latency-vs-reasoning tradeoff. No specific ch-20 method named (DSBS / Orca-2 prompt erasing / R1 always-emit) — same naming reluctance as Stage 2.

Honest punt on "when to think": *"hard. skip for now."*

⭐ **Framework extension #2 — model-size-relative thinking threshold**: *"for larger model, task A is so easy so no thinking needed, but for small model, task A might be challenging."* Chapter line 153 only describes R1's `<think>` auto-regulation as task-difficulty-based; learner identified that the threshold is *student-capacity-relative*, not absolute. R1-distill-7B has different thinking budget needs than R1-distill-32B — a real production gap chapter does not address.

## Stage 4 — Verifier design

Q-side: ✅ *"classify the difficulty and need to know this is possible to answer or not. if not, redirect to real human?"* — LLM-labeled difficulty (Q14 axis 9) + escalation logic for unanswerable queries. Q-side curriculum framework invoked.

⚠ **Axis 5 (verifier coverage) — major gap**: probe specifically noted that customer support has no math/code-equivalent ground-truth verifier. Learner did not engage with this critical question. Chapter line 141 *"verifier determines the upper bound"* not invoked. Without explicit recognition of axis 5 uncovered, the pipeline's predicted ceiling is implicit only (named in Stage 5 as teacher-ceiling but not connected to verifier).

## Stage 5 — Failure modes (mastery-level synthesis)

Five mechanisms self-surfaced:
1. ✅ Teacher ceiling (axis 4)
2. ✅ Thinking leak to user (axis 5 quirks inheritance — same as §5 teacher-bias inheritance)
3. ✅ **Pass@k limit for multi-turn** — self-aware: *"verify round by round will be time-consuming."* Recognizes own ch-19 verdict E6 reinvention's transfer limit
4. ✅ Diversity ceiling — direct echo of ch-19 verdict E5 (seed-bound composition's diversity failure)
5. ⭐ **Customer-LLM realism callback self-surfaced**: *"current LLM cannot play like bad-customer"* — independently surfaces the ch-19 parked question ([[project_llm_training_customer_llm_realism_callback]]). Memory was set to fire at ch-25; learner surfaced it at ch-20 without prompting.

Multi-turn axis 8 described but not labeled — minor framework-tag gap.

---

## Verdict reasoning

**Mastery criteria met**:

1. ✅ **Application**: distillation pipeline composed across teacher (Q11 axis 7) / generation (implicit ch-19 methods) / strategy (latency-vs-reasoning) / verifier (Q14 Q-side curriculum) for a novel domain
2. ✅ **Diagnosis**: 5 failure modes identified, multiple connecting to specific control axes
3. ✅ **Synthesis**: ch-19 + ch-20 framework integrated; pass@k limitation self-aware; customer-LLM realism callback fired without prompt

**Framework extensions (above-bar)**:
- Extension #1: Negative anchor for symmetric Job B regression test (extends ch-18 line 133)
- Extension #2: Model-size-relative thinking threshold (identifies ch-20 §1/§3 blind spot)

**Gaps acknowledged**:
- Stage 2 ch-19 method naming reluctance (regression from ch-19 E1 strength)
- Stage 4 axis 5 (verifier coverage) not explicitly invoked
- Stage 1 multi-axis teacher justification (single-axis only)
- Anchor Job B initially confused (corrected after explicit push)

The two framework extensions parallel ch-19 verdict E6's pass@k reinvention move — signature of internalized framework producing novel insight rather than memorized facts. Customer-LLM realism callback self-surface is an additional cross-chapter synthesis signal.

**Verdict: Mastery**.

Per learner direction: commit + push to course/llm-training; merge-vs-stay-on-branch decision pending.
