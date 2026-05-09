<!-- chapter: ch-18 — discuss transcript
     deps: [[summary]], [[qa]], [[read]]
     scope: discuss-phase exchange, condensed for evaluator
-->

# Ch-18 — Discuss Transcript (Micro-Discuss / Option A)

Single-probe lite discuss per learner request after a long summarize-phase iteration. Probe 1 only; learner self-extended to a domain application.

---

## Probe 1 — Alpaca through the 6-stage checklist (cold)

**Teacher prompt**: walk Alpaca (Stanford 2023, ~$500, 52K samples, GPT-3.5 + Self-Instruct seeds) through the 6-stage checklist. Bonus: name Raschka sub-type and predict Stage 4 from it.

### Learner answers (initial)

| Stage | Answer |
|---|---|
| 1 Generate | "generate the data based on the seed anchor" |
| 2 Filter | "not sure, but I think it may hard to filter out. because instruction itself may hard to filterout" |
| 3 Dedup | "yes" |
| 4 Verify | "yes with LLM as a judge" |
| 5 Select | "not sure" |
| 6 Mix | "not sure" |

Bonus question: not attempted.

### Teacher correction

- Stage 1 ✅ correct shape
- Stage 2 ⚠ wrong intuition — Alpaca DID have surface filters (length cap, banned-word list, schema validity)
- Stage 3 ⚠ right but no specifics (ROUGE-L > 0.7 inherited from Self-Instruct)
- Stage 4 ❌ **factually wrong AND chapter-claim wrong** — Alpaca had NO verifier. This is the canonical "Self-Instruct-era weakness." Quoting line 97 of read.md: *"Self-Instruct: Minimal — no gold-answer checker, no executor. Just format + ROUGE. This is the paper's acknowledged weakness."* Alpaca inherited this empty cell wholesale.
- Stage 5 ⚠ honest gap — empty (all surviving samples kept)
- Stage 6 ⚠ honest gap — used standalone, no real-data mix at SFT

The Stage 4 misconception was load-bearing: bootstrap → "stage 4 has no ground truth unless task is verifiable; this is where Self-Instruct-era pipelines are weakest." Learner had this claim in their own summary §5 but did not apply it deductively to Alpaca.

### Learner self-correction (immediate, post-feedback)

> "Oh I see. from alpaca, filter out with banned-word and length cap! this is insightful to me. oh I see for 4, no verifier from aplaca. 5. empty. 6. mix stand alone sft dataset."

Integrated all four corrections without defensiveness.

---

## Self-extended Probe — Sales-Call Conversation Dataset (learner-initiated)

After receiving Alpaca corrections, learner pivoted to apply the loop to a domain they care about. **This pivot is significant** — it moved from "factual recall test" to "framework application to a novel domain," which is the highest level of transfer the chapter is designed to teach.

### Learner's sales-call pipeline

| Stage | Learner's design |
|---|---|
| 1 Generate | "generate using LLM" |
| 2 Filter | "clear rules when show up product. if not includes those word, or if include banned words, filter out" |
| 3 Dedup | "challenging for conversation, but maybe round-level dedup" |
| 4 Verify | "LLM as a judge to filter out" |
| 5 Select | "carefully select for diversity to handle diverse customer scenario (diverse objection handling)" |
| 6 Mix | "stand-alone lora finetune or combine with replay dataset" |

### Teacher assessment

| Stage | Verdict |
|---|---|
| 1 | ⚠ Underspecified — didn't name the Raschka sub-type (which predicts Stage 4 cost) |
| 2 | ✅ Strong — exactly stage-2 work for compliance-bound domains |
| 3 | ✅ Right intuition that multi-turn dedup is harder; ch-25 covers this |
| 4 | ✅ Right call (LLM judge) but triggers the unverifiable-task warning |
| 5 | ✅ Strong — coverage-based selection is mature for sales objection handling |
| 6 | ✅ Replay-as-mix is the right instinct (line 137: "accumulation over replacement") |

Two refinements offered:
1. Naming the stage-1 sub-type (bootstrap / full-generate / rewrite / backtranslate) determines stage-4 cost. Rewrite-from-real-transcripts collapses stage 4 to faithfulness check; full-generate-from-scratch requires the expensive judge.
2. Sales-call quality has no cheap ground truth → falls in "unverifiable tasks don't compound" zone (corollary #1). Practical consequences: judge must be calibrated against real outcomes, anchor must be refreshed, judge-swap stress test needed.

The career-advice frame (line 173) maps directly to the learner's domain: "the moat in sales-AI isn't the generator — it's the judge calibration. Whoever owns 'what does good sales call mean operationally + how do we audit the judge' is load-bearing."

---

## Verdict reasoning (pre-evaluator)

Initial Alpaca probe alone: Partial-quality (one critical factual error on the chapter's central claim).

Sales-call self-extension: strong evidence of framework installation. Learner:
- Correctly identified compliance filters for stage 2
- Recognized multi-turn dedup as a harder open problem
- Reached for LLM-judge at stage 4 for the right reasons (no symbolic verifier exists)
- Reasoned about diversity selection for objection coverage
- Recognized replay-as-anchor mixing for the right reason

Trajectory matters: factual gap on Alpaca, immediate integration of correction, self-initiated novel-domain application. The framework is installed even if specific facts about prior pipelines required teacher correction.

Recommended verdict: **Mastery**, with note that the Alpaca-fact error suggests the learner should re-check the 4×6 table in line 96-97 of read.md when revisiting.
