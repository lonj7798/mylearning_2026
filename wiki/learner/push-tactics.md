# Push Tactics
<!-- scope: tactic modes the teaching agent uses to push the learner during discuss phase
     deps: [[session-log]]
     see-also: [[learning-style]], [[strengths]], [[weaknesses]]
-->

Five tactic modes are available. The profiler agent selects the best blend
based on data in [[session-log]].

## Tactic Modes

| Tactic | Description | Use When |
|--------|-------------|----------|
| **interrogator** | Rapid-fire questions; learner must defend every claim | Learner makes confident but shallow statements |
| **debater** | Teacher takes the opposing view; learner must rebut | Learner needs to strengthen reasoning and argumentation |
| **examiner** | Structured quiz; closed questions with correct/incorrect feedback | Recall and pattern recognition need drilling |
| **coach** | Supportive guidance; hints before answers; praise progress | Learner is stuck, frustrated, or losing confidence |
| **blend** | Mix of the above tactics within a single session | Default when no single mode fits the learner's current state |

## Selection Rubric

The profiler agent picks a tactic (or a blend) based on:

1. Verdict history — repeated `incomplete` verdicts signal need for `examiner` or `interrogator`
2. Session affect — frustration signals shift to `coach`
3. Confidence calibration — overconfidence signals `debater` or `interrogator`
4. Default — `blend` when fewer than 3 sessions have been recorded

## Tactic Outcome History

| Session | Chapter | Tactic | Outcome Signal | Notes |
|---------|---------|--------|----------------|-------|
| #2 | llm-training/ch-18 | blend | engagement | Direct corrections with ✓⚠✗ severity tags produced non-defensive integration and self-initiated domain transfer. Learner explicitly requested real-time flagging mid-session; the tactic's directness was welcomed, not resisted. |

## Tactic Guidance Notes (updated after ch-18)

**blend with direct correction works well for this learner.** Key properties that drove engagement:
- Severity tags (✓ ⚠ ✗) on each probe answer — learner knows exactly where they stand without ambiguity
- No hedging on corrections — "factually wrong AND chapter-claim wrong" landed without defensiveness
- Allowing learner-initiated pivots within the discuss structure (sales-call self-extension) — treating them as a positive signal rather than redirecting back to the probe
- Career-frame connections — mapping chapter insights to Anthropic research goal increases retention

**Examiner mode** is worth testing in a future session for chapters with dense comparison tables (like the 4×6 pipeline comparison in ch-18). The Alpaca factual error suggests that cold recall of specific cells benefits from structured quiz format before the main discuss probe.

See [[session-log]] for session-by-session tactic selections.
