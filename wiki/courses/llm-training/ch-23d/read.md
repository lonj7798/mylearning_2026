---
chapter: ch-23d
course: llm-training
phase: read
---

# Chapter 23d — Speech, Turn Taking, and Layered Evaluation

<!-- scope: speech layers, interruptions, action commitments, and matched acoustic diagnostics
     deps: [[../ch-23c/read]]
     see-also: [[methods]], [[practice]], [[long-tail-papers]]
     sources: [[audio-multichallenge]], [[sdiareward]], [[deepdialogue]]
-->

## Overview

**Core insight.** A transcript removes information about timing, pitch, interruption, and overlap. Spoken task failures can originate in recognition, context reasoning, policy interpretation, speech delivery, or the controller that commits actions. Adding more text dialogue cannot identify or repair all of these causes.

**Guideline.** Locate the audio boundary in your system and compare matched scenarios through its layers. Evaluate task validity separately from naturalness, and preserve action preconditions across interruptions. This chapter proposes diagnostics, not a universal training recipe.

Read the main lesson → [Methods and research limits](methods.md) → [Practice: matrices and metrics](practice.md). [한국어](read_kor.md). Dates and scheduling policies in the worked case are fictional teaching examples.

## Key Concepts

### 1. Three systems that should not be conflated

In an ASR→text LLM→TTS pipeline, ASR converts input audio into text, the LLM reasons over that text and instructions, and TTS speaks the output. A controller decides when to listen, speak, interrupt, or commit a tool action. A native audio model can consume acoustic features directly, but it still operates in a system with timing and action boundaries. A text-only chatbot sees neither raw prosody nor actual overlap unless they are represented in its input.

A text SFT improvement can help interpretation of a correctly transcribed self-correction. It cannot demonstrate robustness to an accent the ASR fails to recognize or prevent a controller from committing a tool call after the user interrupted. Diagnose the failing layer before collecting more argument-filling data.

### 2. What the transcript preserves and loses

The words ‘Thursday at three’ can be spoken calmly, hesitantly, quickly, or over another speaker. A clean transcript may preserve the intended values but omit uncertainty, overlap, and the fact that a correction started before the assistant finished. It can also silently normalize fragments or choose a recognition hypothesis that was never clearly spoken.

For evaluation, retain the recording, actual ASR output, and a human-checked transcript where appropriate. The clean transcript is a diagnostic reference, not proof that the production system heard it correctly. If a transcript includes uncertainty tags, their meaning must be defined and available consistently; do not assume a model understands invented annotations automatically.

### 3. Worked interruption case: facts and commitments

The customer requested Tuesday, November 10, 2026, at 15:00 in a known timezone. The active policy requires confirmation before scheduling. The assistant begins, ‘I will confirm Tuesday at three—’. The customer interrupts, ‘No, Thursday… Thursday the twelfth.’ The intended final date is November 12. There is still no explicit confirmation of the corrected booking.

If no action has committed, the supported next behavior is to stop the outdated readback, update the date from the correction, and ask for confirmation of Thursday at 15:00. No tool call is authorized yet. If the controller continues speaking Tuesday, speech timing failed. If the model hears Thursday but stores Tuesday, state update failed. If it stores Thursday and books immediately, policy timing failed.

Now consider that a prior action already committed. The system must inspect actual tool state and follow the cancellation/change policy; it cannot pretend the booking never occurred. Pending actions, committed actions, and spoken promises are different. A generation example that assumes a successful rollback without tool evidence teaches an unsafe state fiction. Include the actual tool result and the permitted recovery action in any training target.

For an ambiguous overlap where the day cannot be recovered reliably, asking the customer to repeat the date is a valid response. A familiar Tuesday value is not a justified fallback. The model should not confuse the most likely recognition hypothesis with a confirmed scheduling fact.

### 4. Input, output, and controller tails

Input variation includes accent, rate, room noise, pauses, disfluency, corrections, and overlapping speech. Output variation includes pronunciation, prosody, pacing, and emotional delivery. Controller variation includes response onset, interruption detection, stopping speech, resumption, and tool-commit timing. These interact but require separate labels and instrumentation.

An unusual pronunciation may be rare in your logs and common for a different population. Use supported metadata or observed acoustic features; do not infer identity from accent. A synthetic voice style is a controlled condition, not a certificate of population coverage. A generated emotional label is also not a verified acoustic expression.

### 5. Why naturalness and task success diverge

A pleasant voice can speak the wrong date. A factually correct answer can be delivered while talking over a customer's correction. A short, slightly hesitant clarification can be more appropriate than a fluent unsupported call. Evaluate independently: did it preserve the required facts, follow policy, communicate intelligibly, and coordinate turns?

Episode-level audio preferences can capture qualities a text judge misses. They can also favor a voice, pacing, or conversational style without checking tool truth. Conversely, a deterministic schema validator checks arguments but not whether the customer could understand the spoken response. Each evaluator has a defined scope; none replaces the others.

### 6. Controlled speech generation versus real interaction

TTS can keep the words fixed while changing rate, voice, or supported delivery parameters, which helps isolate an acoustic condition. But a script read aloud may lack spontaneous self-repair or interruption. Human improvisation from a scenario blueprint can supply those phenomena, while requiring careful task/rubric validation and consented recording.

Both methods can be useful in a matched test. Treat TTS variants as controlled probes and human recordings as sampled behavior with collection limits. Neither alone establishes every relevant accent, emotional trajectory, or long-tail customer case. Noise injection can test a mechanism without reproducing the frequency of real noisy calls.

## Recipe

Prepare a semantic scenario and gold constraints, obtain controlled and human audio where appropriate, run the actual recognition/reasoning/controller pipeline, and preserve layer outputs. Repair the failing layer under matched conditions before claiming a full-system improvement. If you fine-tune only text, say which text behavior improved and which audio/controller claims remain untested.

## Generalization lens

Hold out speakers, recording conditions, scenario families, and instruction combinations separately. Keep all audio/text variants of a family together. Report common cases and clearly labeled acoustic stress cases, with denominators and uncertainty. Test that better audio preferences do not hide worse fact or tool correctness.

## Questions

1. Which failure can a clean-transcript test hide?
2. Why do pending and committed actions need different interruption targets?
3. What does a high audio preference score fail to verify?

[Methods](methods.md) → [Practice](practice.md) → [24](../ch-24/read.md). [Catalog](../../../raw-data/llm-training/long-tail-papers.md).


## Source cards

- [[audio-multichallenge]] — [card](../../../raw-data/llm-training/papers/audio-multichallenge.md)
- [[sdiareward]] — [card](../../../raw-data/llm-training/papers/sdiareward.md)
- [[deepdialogue]] — [card](../../../raw-data/llm-training/papers/deepdialogue.md)
