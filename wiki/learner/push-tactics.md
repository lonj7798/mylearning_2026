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

See [[session-log]] for session-by-session tactic selections.
