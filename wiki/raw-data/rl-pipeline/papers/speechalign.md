<!-- scope: SpeechAlign (Fudan): preference optimization for a codec-language-model TTS (SpeechGPT AR + SoundStorm NAR, SpeechTokenizer RVQ); preference pairs = real-recording ("golden") AR tokens preferred over the model's own synthetic AR tokens; CoH, DPO, RLHF-PPO with BT reward model, Best-of-N; iterative DPO; continued-SFT control; human naturalness A/B tests
     deps: [[dpo]], [[ppo]], [[bradley-terry-rm]]
     see-also: [[speechjudge]], [[spin]], [[sdiareward]], [[subjective-reward-guide]]
-->

# SpeechAlign: Aligning Speech Generation to Human Preferences
- **Core Insight:** For a codec-LM TTS, preference pairs can be built without human labels by preferring tokens extracted from real recordings over the model's own generated tokens for the same text; humans preferred the real-token speech in 71% of 100 verified pairs (21% tie, 8% lose; Table 2), and three rounds of DPO on such pairs lowered LibriSpeech WER from 7.2 to 6.0 and raised speaker similarity from 0.87 to 0.90 (Table 3), while continued SFT on the same real tokens did not help (8.0 / 0.88).
- **Guideline:** When human-recorded speech in the target style exists, use "real recording beats own generation" as the preference signal and verify a random sample with human A/B tests before training, and compare against continued SFT on the same data, because in this study the gains came from the preference objective rather than from extra exposure to real tokens (§5.1, Table 3); stop adding pairs when gains flatten (50K → 250K pairs gave no larger gain, §5.1).
- **Authors:** Dong Zhang, Zhaowei Li, Shimin Li, Xin Zhang, Pengyu Wang, Yaqian Zhou, Xipeng Qiu (Fudan University)
- **Year:** 2024 (arXiv v1 2024-04-08)
- **URL:** https://arxiv.org/abs/2404.05600 (code: https://github.com/0nutation/SpeechGPT; demo: https://0nutation.github.io/SpeechAlign.github.io/)
- **Source type:** paper
- **Relevant topics:** speech preference optimization, reference-based preference pairs (real vs generated), iterative DPO / self-improvement, BT reward model for speech tokens, PPO vs DPO vs Best-of-N for TTS, train-inference distribution gap

## Summary
The baseline (SpeechAlign-sft) is a SpeechGPT-based AR model that maps text to first-layer SpeechTokenizer codes and a SoundStorm-based NAR model that generates the other 7 RVQ layers (§2.1). In training the NAR model sees "golden" AR tokens from real speech, but at inference it sees synthetic AR tokens; t-SNE shows the two form separate clusters (§2.2, Fig. 2a), and reconstruction from synthetic tokens is worse (WER 7.2 vs 5.9, SIM 0.87 vs 0.93; Table 1). SpeechAlign samples 50K LibriSpeech speech–text pairs, takes golden AR tokens as chosen and the AR model's tokens as rejected, optimizes the AR model, and repeats with the updated model (§3.1–3.3, Alg. 1).

## Key Contributions
- Human-label-free preference data for speech, checked by a 100-pair human test (§3.1, Table 2).
- Comparison of four alignment methods on the same pairs: Chain-of-Hindsight, DPO, RLHF-PPO with a BT reward model initialized from the AR model, Best-of-N with that RM (§3.2).
- Iterative DPO (Iter1–3), with synthetic data accumulated from the latest iteration (§4.1).
- Controls: continued SFT on the golden tokens; data-size ablation; a 130M AR model (§5.1–5.2).

## Key Figures/Tables to Study
- Table 1 (distribution-gap effect), Table 2 (human verification), Table 3 (WER/SIM), Fig. 1 (human A/B win rates vs SFT; values only in the figure), Fig. 4 (data size, small model).

## Technical Details
**Table 3 (zero-shot TTS; LibriSpeech WER / SIM, VCTK WER / SIM).** Ground truth 4.0 / –, 1.7 / –. SFT 7.2 / 0.87, 8.8 / 0.79. Continue SFT 8.0 / 0.88, 9.8 / 0.80. CoH 7.3 / 0.89, 10.2 / 0.81. RLHF-PPO 7.1 / 0.89, 8.5 / 0.80. BoN 8.0 / 0.88, 7.5 / 0.79. DPO-Iter1 6.7 / 0.88, 8.5 / 0.82. DPO-Iter2 6.2 / 0.89, 8.0 / 0.83. DPO-Iter3 6.0 / 0.90, 7.9 / 0.83. Each value is the mean of ten evaluations. WER uses Whisper medium-en; SIM uses WavLM-base-plus-sv cosine similarity (§4.2).

**Human evaluation (§4.2).** Evaluators hear the prompt speech, the baseline output, and the model output, and pick which is "more natural and closer to the prompt speech" or a tie; 6 raters per item. The text states BoN, RLHF-PPO, and DPO-Iter1 "significantly outperform" the baseline in win rate and DPO-Iter3 beats DPO-Iter1 (§4.3); numeric win rates appear only in Fig. 1.

**Small model (§5.2).** 130M AR model: first DPO iteration lowers WER 11.4 → 9.3 and raises SIM 0.87 → 0.88; later iterations give no further SIM gain (the text reports "a consistent upward trend in WER", direction not clarified).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| SpeechAlign-DPO | SpeechGPT-based AR (size not stated in §4.1) | pref-opt | batch; lr; steps; GPUs | 128; 5e-7; 2,000; 8×A100 80G | §4.1 | verified 2026-11-24 | Table 3 |
| SpeechAlign reward model | same | RM | batch; lr; steps | 32; 1e-5; 1,000 | §4.1 | verified 2026-11-24 | PPO/BoN rows in Table 3 |
| SpeechAlign-RLHF-PPO | same | RL | batch; lr; steps; KL β | 16; 1e-5; 1,000; β not reported | §4.1, §3.2 | verified 2026-11-24 (β not reported) | Table 3 |
| Preference data | — | data | pairs per iteration | 50K at iteration 0; 100K in iterations 1–3 | §4.1 | verified 2026-11-24 | Fig. 4a (0 / 50K / 250K) |

## Findings relevant to generality and negative feedback
- **Negative signal.** The rejected side is always the current model's own output, so each iteration pushes away from the current distribution toward real recordings, the same structure as [[spin]] (cited in §6).
- **Generality.** Gains held on VCTK speakers not in the preference data (Table 3, §4.3).
- **Scope.** Read speech (LibriSpeech, VCTK); naturalness of conversational or sales speech, disfluency, and turn-taking are not measured. The "human preference" is human-verified only on 100 pairs; training labels come from the real-vs-generated rule.

## Connections
- [[speechjudge]]: learned naturalness judge from 99K human pairs; complementary to SpeechAlign's label-free pairs.
- [[spin]]: same "real data beats own sample" self-play construction in text.
- [[sdiareward]]: spoken-dialogue-level reward including colloquialness, closer to a voice-agent setting.

## Verification
- Read the arXiv v1 PDF (2404.05600v1) in full on 2026-11-24 via pdftotext -layout.
- Selection: chosen as the speech-style RL card because it had the highest Semantic Scholar citation count (78, queried 2026-11-24) among six readable 2024–2025 candidates (Emo-DPO 49, Koel-TTS 42, Align-SLM 41, "Preference Alignment Improves LM-Based TTS" 37, DLPO 10).
- Not reported: AR model parameter count for the main runs, KL coefficient for PPO, DPO β, numeric human win rates (figure only).
