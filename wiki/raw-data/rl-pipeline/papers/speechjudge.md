<!-- scope: SpeechJudge (CUHK-Shenzhen / ByteDance Seed / DataBaker): 99K human-annotated TTS pairs (intelligibility pointwise + naturalness pairwise CMOS); SpeechJudge-Eval (1,000 full-agreement pairs) showing WER/SIM/FAD/MOS predictors/deepfake detectors/AudioLLMs vs humans; SpeechJudge-GRM (Qwen2.5-Omni-7B, SFT on Gemini CoT + GRPO on hard cases) vs BTRM; best-of-100 selection; offline/online DPO for TTS with the GRM as reward
     deps: [[generative-reward-models]], [[bradley-terry-rm]], [[grpo]], [[dpo]]
     see-also: [[speechalign]], [[sdiareward]], [[audio-multichallenge]], [[litbench]], [[subjective-reward-guide]]
-->

# SpeechJudge: Towards Human-Level Judgment for Speech Naturalness
- **Core Insight:** Common automatic speech-quality proxies agree poorly with human naturalness preferences between two modern TTS outputs (UTMOS 53.7%, DNSMOS 57.9%, WER 57.9%, best AudioLLM Gemini-2.5-Flash 69.1%; Table 2), and a generative judge trained on human pairwise labels reaches 77.2% (79.4% with 10-vote majority) versus 72.7% for a Bradley-Terry head on the same backbone and data (Table 3).
- **Guideline:** Do not use a MOS predictor as the naturalness reward for modern TTS or voice agents; train a pairwise judge on human A/B naturalness labels from your own speech domain and validate it on full-agreement pairs, because MOS predictors were near chance on this task (Table 2) and the authors state their own data cover "TTS-style read speech rather than spontaneous conversation" (§6).
- **Authors:** Xueyao Zhang, Chaoren Wang, Huan Liao, Ziniu Li, Yuancheng Wang, Li Wang, Dongya Jia, Yuanzhe Chen, Xiulin Li, Zhuo Chen, Zhizheng Wu (CUHK-Shenzhen; ByteDance Seed; DataBaker Technology)
- **Year:** 2025 (arXiv v1 2025-11-11; v2 2025-11-30)
- **URL:** https://arxiv.org/abs/2511.07931 (data/model/code: https://github.com/AmphionTeam/SpeechJudge)
- **Source type:** paper
- **Relevant topics:** speech naturalness reward, human preference data for audio, MOS predictor validity, AudioLLM-as-judge, GRM vs BTRM, inference-time voting, TTS DPO with learned reward, inter-annotator agreement

## Summary
SpeechJudge-Data has 99K (text, audio1, audio2) samples from six zero-shot TTS models (ARS, CosyVoice2, CosyVoice2-INTP, Ints-INTP, F5-TTS, MaskGCT), regular (Emilia) and expressive (emotional, accented, whisper, game-character) references, and zh/en/code-switched text (§3.1, Fig. 2). Annotators mark intelligibility (binary) per clip and naturalness on a five-point comparative MOS per pair (§3.2). 69 annotators over two months gave 2.49 annotations per pair on average; cost estimated at over 500K RMB (§3.2). SpeechJudge-Eval has 1,000 pairs with full agreement and no ties (§4.1). SpeechJudge-GRM is Qwen2.5-Omni-7B fine-tuned with LoRA: SFT on Gemini-2.5-Flash CoT where Gemini agreed with humans, then GRPO with reward +1/−1 for matching the human label on the cases Gemini got wrong (§5.1).

## Key Contributions
- Agreement analysis: 51.5% full agreement + 17.2% weak agreement overall; expressive speech has lower agreement than regular (44.3% vs 54.5% full agreement) (§3.2, Fig. 3).
- Benchmark of proxy metrics against human naturalness preference (Table 2).
- GRM with CoT and voting outperforms BTRM (Table 3).
- Use as reward: best-of-100 selection and TTS post-training with offline and online DPO (§5.3–5.4).

## Key Figures/Tables to Study
- Fig. 3 (agreement levels), Table 2 (proxy-metric accuracy), Table 3 (GRM vs BTRM), Fig. 5 (best-of-100 human test), Fig. 6 (TTS post-training), §6 (limitations and residual failure cases).

## Technical Details
**Table 2 (accuracy, regular / expressive / total).** WER 59.3 / 57.0 / 57.9; DNSMOS 61.0 / 55.8 / 57.9; UTMOS 54.0 / 53.5 / 53.7; audiobox PQ 61.3 / 54.3 / 57.1; AASIST deepfake detector 40.5 / 50.8 / 46.7; Qwen2.5-Omni-7B 62.0 / 59.7 / 60.6; Gemini-2.5-Flash 73.5 / 66.2 / 69.1; Gemini-2.5-Pro 73.0 / 62.2 / 66.5; GPT-4o Audio 71.5 / 64.7 / 67.4.

**Table 3.** BTRM 77.5 / 69.5 / 72.7; GRM (SFT) 77.8 / 73.7 / 75.3, voting@10 77.6 total; GRM (SFT+RL) 79.0 / 76.0 / 77.2, voting@10 80.5 / 78.7 / 79.4.

**Best-of-100 (Fig. 5).** Qwen2.5-Omni-7B Talker generates 100 speeches per text; humans compare the RM's best against a random sample. Win/tie/lose: GRM 43.0 / 26.8 / 30.2%; BTRM 39.5 / 28.0 / 32.5%.

**TTS post-training (Fig. 6a, Qwen2.5-0.5B-TTS).** Text accuracy / naturalness CMOS vs base: base 84.0% / 0.00; offline DPO on INTP 87.0% / 0.18; offline DPO on SpeechJudge-Data 91.0% / 0.16; offline DPO on GRM-relabeled INTP 91.0% / 0.21; online DPO with GRM reward 90.0% / 0.25 ±0.09.

**Residual failures (§6).** Clean-but-robotic vs slightly-noisy-but-lively speech, prosody vs articulation, very high-F0 emotional speech, whispers; the GRM "can over-weight cleanness, under-weight style-appropriate prosody".

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| SpeechJudge-GRM | 7B | RM SFT → RL | data split; RL reward | SFT on Gemini-agreeing CoT; GRPO on Gemini-disagreeing cases; ±1 for matching human label | §5.1, Fig. 4 | verified 2026-11-24 | Table 3 (+1.9 from RL) |
| SpeechJudge-Data (train) | — | data | label rule | drop full-disagreement; majority vote; drop ties | §5.1 | verified 2026-11-24 | no ablation |
| Qwen2.5-0.5B-TTS online DPO | 0.5B | TTS RL | reward; prompts | SpeechJudge-GRM; INTP prompts only | §5.4 | verified 2026-11-24 | Fig. 6 |
| GRM hyperparameters | 7B | RM | lr, LoRA rank, group size | App. E | not read | not extracted | — |

## Findings relevant to generality and negative feedback
- **Domain limit.** Data are synthetic TTS read speech in zh/en/code-switching, judged by professional raters in China; agreement is higher on Chinese than on English and mixed subsets (§6, App. B.2).
- **Human ceiling.** Only about 70% of pairs reach full or weak agreement (Fig. 3), so a judge accuracy near 80% on full-agreement pairs does not mean 80% on all pairs.
- **Teacher bias.** The CoT style is inherited from Gemini-2.5-Flash; intermediate rationales were not human-verified at scale (§6).

## Connections
- [[speechalign]]: preference optimization for codec TTS using real recordings as the preferred side.
- [[sdiareward]]: episode-level spoken-dialogue rewards including colloquialness; SpeechJudge is utterance-level TTS only.
- [[litbench]]: in creative writing, distilled CoT hurt judge accuracy; in speech, CoT SFT + RL helped (75.3 → 77.2 vs BTRM 72.7).

## Verification
- Read the arXiv v2 PDF (2511.07931v2) on 2026-11-24 via pdftotext (plain and -layout): §1–7; Tables 2–3 values read from -layout text. Appendices A–G not read in detail.
