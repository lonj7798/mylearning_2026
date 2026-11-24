# 23d — Matched Scenario Matrices and Failure Localization

<!-- scope: proposed implementation records, controlled comparisons, metrics, and failure repair for speech layers, interruptions, action commitments, and matched acoustic diagnostics
     deps: [[read]]
     see-also: [[read]], [[methods]], [[long-tail-papers]]
     sources: [[audio-multichallenge]], [[sdiareward]], [[deepdialogue]]
-->

## 1. Define the semantic case before recording

Proposed case fields include family ID, known facts and corrections, active policy, tool schema and actual state, speaker/recording provenance, permitted next actions, required values, and timing events. Keep intended generation conditions separate from observed recording properties. A blueprint can request a hesitant correction, but reviewers should check whether it actually occurred and remained intelligible.

For the worked callback, specify that the correction precedes commitment and confirmation is still missing. A separate family can cover committed booking recovery with a supported tool result. Do not combine those cases under one gold call. All versions need an allowed action set, including asking for repetition when uncertainty prevents a valid action.

## 2. Build a matched diagnostic matrix

| Condition | Fixed information | Changed input | Diagnostic purpose |
|---|---|---|---|
| Clean transcript | Scenario and policy | Human-checked words | Context/action reasoning reference |
| Actual ASR transcript | Same recording meaning | Recognition output | Downstream effect of ASR loss |
| Controlled TTS | Same text and policy | Voice/rate/noise condition | Repeatable acoustic probe |
| Human improvisation | Same goal/constraints | Natural wording and timing | Less scripted interaction |
| Live interruption | Same corrected facts | Event order/commit state | Controller and state integration |

For an ASR pipeline, compare the same LLM on clean and actual transcripts. If the clean version succeeds and actual transcript fails, inspect recognition differences before changing LLM targets. If both fail, context or policy reasoning remains a candidate cause. For native audio models, clean-text and audio results involve different input conditions, so the comparison is diagnostic rather than an automatically fair modality ranking.

## 3. Metrics with explicit scopes

Word error rate is (substitutions+deletions+insertions)/reference-word count. A low WER can hide one critical wrong date; a high WER can consist of harmless fillers. Therefore measure task-critical value recovery separately, using supported canonicalization and human adjudication for ambiguous audio. A perfect transcript still does not verify instruction scope.

For actions, measure tool selection, valid no-call behavior, required argument values, provenance support, and precondition compliance. For delivery, assess intelligibility, pronunciation of important values, appropriate pacing, and requested tone. For timing, instrument response onset, time to stop after a recognized interruption, and whether action commitment respected the declared boundary. Choose tolerances from product requirements rather than inventing a universal latency threshold.

An interruption event needs a reference timestamp and a definition of detection/stop. If the system never detected the interruption, do not report only post-detection stop time and hide missed events. Record tool requests and tool confirmations separately: an emitted request is not proof of a completed action.

## 4. Repeat critical values at the right moment

A corrected readback can confirm ‘Thursday the twelfth at three’ before a booking, according to the current policy. It should not keep repeating an obsolete date or treat silence as confirmation unless the policy explicitly defines and supports that behavior. Repetition is a verification opportunity, not a substitute for listening. If overlap leaves a value uncertain, ask for it rather than announcing a confident interpretation.

Evaluate both the final value and the sequence of commitments. A transcript of the last response can look correct even if the system already booked the old date. Keep event logs and tool state in the evaluation artifact, with appropriate privacy handling. The error belongs to the full pipeline even when no individual last sentence is wrong.

## 5. Training comparisons and repair decisions

Compare a baseline with targeted text-data, acoustic-data, or controller changes separately where possible. Keep model and inference settings fixed for each relevant comparison. Hold out speaker and scenario families; keep close text/audio variants in one split. Include clean common cases, rare corrections, and controlled stress conditions, with their collection origin labeled.

If recognition is the failure, a text-only SFT change may not help. If state extraction fails even with clean text, verified correction families are plausible training targets. If controller commit timing fails, correct orchestration and test the boundary. If delivery fails while task behavior is valid, tune or replace the speech component under matched words. A joint intervention can still be useful, but report its combined effect rather than attributing everything to data selection.

[Main lesson](read.md) · [Methods](methods.md) · [Next chapter](../ch-24/read.md) · [한국어](practice_kor.md).

