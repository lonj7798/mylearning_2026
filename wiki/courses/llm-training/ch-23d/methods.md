# 23d — Human Audio Benchmarks and Spoken Reward Models

<!-- scope: primary-source setups, measured outcomes, and transfer limits for speech layers, interruptions, action commitments, and matched acoustic diagnostics
     deps: [[read]]
     see-also: [[read]], [[practice]], [[long-tail-papers]]
     sources: [[audio-multichallenge]], [[sdiareward]], [[deepdialogue]]
-->

## 1. Audio MultiChallenge: improvisation guided by goals

**Result (single study).** [[audio-multichallenge]] evaluates 452 dialogues from 47 speakers with 1,712 rubrics. Scenario blueprints and goals guide human improvisation rather than requiring verbatim synthetic scripts (curation section and Table 1). Recordings preserve disfluency and ambient variability. The benchmark combines memory, instruction-following, and coherence with spoken phenomena such as mid-utterance correction and backtracking.

This is fixed-context evaluation, not a closed-loop benchmark of live interruption detection or controller timing. Its curated challenging conversations are not a random census of customer calls. A weak score identifies failures under those test conditions, not their production incidence. The useful design is a coherent task plus human improvisation plus explicit grading criteria. Applying that design to sales requires your own product truth, active policy, and tool-precondition rubrics.

## 2. SDiaReward: full-episode preferences

**Result (single study).** [[sdiareward]] models and benchmarks spoken dialogue reward using modality and colloquialness. §3.1 separates modality-based pairing and colloquialness pairing, and the evaluation includes full episodes rather than only isolated written replies. Prosody, emotion, and spoken language can matter to preferences.

§6 discusses domain-dependent absolute score offsets. Thus an absolute reward number should not be treated as a universally calibrated quality scale across domains. Relative comparisons under matched conditions are more meaningful, while factual and tool checks remain separate. The paper does not show that a reward model automatically knows deployment rarity or verifies an appointment database.

**Interpretation.** Use a spoken preference instrument to audit delivery alongside explicit task rubrics. If it is used during optimization, retain an independent evaluation because the model can learn the instrument's preferences. A higher reward with worse corrected-date accuracy is a divergence, not an overall win.

## 3. Expressive TTS and its limits

**Result (single study).** [[deepdialogue]] connects domain/emotion-conditioned text to speech and tests speech emotion recognition, including transfer to RAVDESS (§4.2). Its20 text-emotion labels map to eight audio classes in Appendix B.3. That design provides controlled emotional speech variation, not evidence of twenty naturally distinct spoken affect states.

TTS gives repeatable conditions and can preserve a script, which is useful for matched diagnostics. It does not establish spontaneous overlap, natural customer accents, or live action timing. The study's emotion-recognition gains should not be relabeled sales success. Longer dialogues are also more likely to be filtered in the text construction, so examine the retained interactions before synthesizing audio.

## 4. What to borrow from each study

| Source | Borrowable evaluation idea | Additional sales evidence needed |
|---|---|---|
| Audio MultiChallenge | Blueprint-driven human audio and rubrics | Product/policy truth and frequency estimates |
| SDiaReward | Episode-level spoken preference comparison | Independent facts/tools/timing checks |
| DeepDialogue | Controlled emotion-conditioned TTS | Natural repairs, overlap, and speaker coverage |

**Open question.** Does a particular targeted training intervention improve your complete spoken system on new speakers and policy conditions? A text improvement, a reward score, or a TTS emotion label addresses only part of that question. Evaluate the layers and their integration under the actual deployment configuration.

[Main lesson](read.md) · [Practice](practice.md) · [한국어](methods_kor.md).

