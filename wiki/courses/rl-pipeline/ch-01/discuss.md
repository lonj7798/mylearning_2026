<!-- chapter: ch-01 — discuss transcript (rl-pipeline)
     deps: [[read]], [[qa]]
     scope: discuss-phase exchange, condensed for verdict record
-->

# ch-01 — Discuss Transcript

Teacher-inline, live Socratic probes. Context: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[olmo-3-rl-pipeline]].

## Probe 1 — 300-step distilled SWE run (rollout median 93 min, Fig. 2)
Premise fixed on request: open-framework default = colocated sync.
- **Initial:** "distillation regime" (classified by starting checkpoint). Sync → no staleness → restart from checkpoint is fine. Async → long rollout spans steps → off-policy ✓.
- **Push:** regime = cost of restart = wall clock, not checkpoint name; staleness (numerics) and restart cost (lost time) are separate.
- **Re-answer:** 300 × 93 min ≈ 2.7 weeks (teacher: 465 h ≈ 19.4 d, lower bound since sync waits for the max) → weeks regime ✓. Variable = sampler workload → teacher sharpened to rollout duration (Recipe step 1 "longest rollout type"). Learner GPU idle ✓ (teacher: in colocated the same GPUs idle during the long tail; Olmo 3 training = 125/1,000 s).

## Probe 2 — application to the learner's sales voice agent (TMR)
- **Duration:** 2–5 min/round (QC + regenerate) × up to 40 rounds; placed "between IF and tool use" by turn count. Teacher: 80–200 min → web-research/SWE scale.
- **Regime:** "climb from base, no distillation" — name-based again; step count and wall clock not filled.
- **Own factor:** QC + regenerate. Initial claim "each turn fast → little staleness" contradicts the per-token definition (Q1) under episode-level training.
- **Learner counter-design ⭐:** train per round. Rewards are turn-local (IF, naturalness, tool-calling, banned-words); final outcome is *not* a reward. → No need to wait for round 40; staleness bounded by one round. Teacher accepted: coherent with this reward design. Rocket segments do not achieve this because advantages need group grading of complete rollouts.
- **Baseline:** asked "what is advantage?" → teacher gave GRPO A_i = (r_i − mean)/std, worked G=4 example (+1.41, −1.41, 0, 0). Learner design: best-of-G answer becomes the next round's shared prefix → same-prefix group, GRPO baseline valid ✓.

## Probe 3 — best-of-G prefix vs deployment
1. **Prefix mismatch:** customer side follows scenario/stage instructions, TMR driven mainly by stage instruction → mismatch small. Teacher: plausible, but TMR's own prior turns are best-of-G; testable (best-prefix vs single-sample rollout eval, compare per-round reward).
2. **Error-containing prefixes:** teacher's question was ambiguous; teacher then wrongly said it had asked about "recovery" (learner's "복구" = infra failure recovery, the ch-01 sense). Learner's valid correction: a best answer can still contain the error, so such prefixes do reach round 6; and RL lowers the error rate. Teacher added: training frequency of error prefixes ≈ P(best still errs) vs deployment ≈ P(single sample errs) — gap shrinks as the error rate falls.
3. **Turn-level blind spot:** learner rejected teacher's contradiction example — a judge conditioned on full history catches it ✓ (valid if the rubric checks cross-turn consistency). **Learner-derived ⭐:** turn-level reward is greedy; multi-turn build-up persuasion (turns that pay off later) is under-rewarded.

## Parked
- Grader for delayed-payoff turns without outcome reward (stage-level reward?) → ch-11/12, ch-19, capstone ch-22.
- Measure best-prefix vs single-sample rollout gap → ch-10.
