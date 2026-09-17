---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: primary source arXiv:2411.15124v5 (library card [[tulu-3]] does not cover the evaluation sections)
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-09-15"
---

# Excerpt: Tülu 3 evaluation regime — development suite, unseen suite, decontamination

**Paper:** Lambert et al., "Tülu 3: Pushing Frontiers in Open Language Model Post-Training" (Allen Institute for AI; arXiv v1 2024-11, read at v5, 2025-04-14). Source type: official technical report.

## Core skills and suites (§2.1-2.2, Table 3)
- Core skills: knowledge recall, reasoning, mathematics, coding, instruction following, general chat, safety (§2.1).
- Development / unseen pairs (Table 3): Knowledge MMLU, PopQA, TruthfulQA / MMLU-Pro, GPQA; Reasoning BBH, DROP / AGIEval English; Math MATH, GSM8K / DeepMind Mathematics; Coding HumanEval, HumanEval+ / BigCodeBench; Instruction following IFEval, AlpacaEval 2 / IFEval-OOD, HREF; Safety Tülu 3 Safety / none.
- The authors did not examine unseen-suite scores during development, so the unseen suite shows how much the data, algorithm, and hyperparameter decisions overfit the development evaluations (§2.2).

## Settings (Table 24, §7.2-7.3)
- Development: MMLU 0-shot CoT EM; PopQA 15-shot EM; TruthfulQA 6-shot MC2; BBH 3-shot CoT EM; DROP 3-shot F1; GSM8K 8-shot CoT EM; MATH 4-shot CoT flex EM; HumanEval and HumanEval+ pass@10 at temperature 0.8; IFEval prompt-level loose; AlpacaEval 2 length-controlled win rate, greedy up to 8,192 tokens; Tülu 3 Safety macro average over HarmBench, XSTest, WildGuardTest, JailbreakTrigger, Do-Anything-Now, WildJailbreakTest (§7.2.1).
- MATH "flex" extraction tries the Minerva format, the last `<ans>` tag, and text between the last two `$`; moving from Minerva-only to flex "can sometimes improve reported scores by up to 10 points" (§7.2).
- Unseen suite formulation principles: tasks posed as humans would pose them (no few-shot examples as dialog, no prescribed CoT examples), clear instructions requesting concise reasoning and an answer format, lenient answer extraction (§7.3).
- BigCodeBench: "hard subset" of 148 of 1,140 instances, instruct formulation, calibrated score (§7.3). IFEval-OOD: 52 constraints in six categories (§7.3.1). HREF: 11 task categories, judge Llama 3.1 70B Instruct, baseline Llama 3.1 405B Instruct; composite agreement with humans 69.4% versus 67% between humans (§7.3.2).

## Decontamination (§3.2; App. B.2 Table 37)
- 8-gram matching on prompts (user turns). A test instance overlaps a training instance when more than 50% of its tokens have 8-gram matches with that one instance.
- A training set is contaminated when it overlaps more than 2% of the instances of any evaluation. Sets contaminated with unseen evaluations are removed; for development evaluations, the whole set is removed if performance is not significantly affected, otherwise only matching instances.
- Table 37 (overlap >5%): e.g., Evol CodeAlpaca 70.7% of HumanEval; LMSys Chat 1M 46.5% of AlpacaEval, 90.3% of Do-Anything-Now, 10.3% of MMLU.

## Dev-versus-unseen results (§7.4, Tables 31-33, Fig. 24)
- Table 31 (8B SFT → DPO → final): average dev 64.9 → 68.3 → 68.8, unseen 29.9 → 31.9 → 32.4; Coding HumanEval 86.2 → 83.9 → 83.9, BigCodeBench 11.5 → 9.5 → 7.4; IF IFEval 72.8 → 81.1 → 82.4, IFEval-OOD 17.6 → 23.9 → 24.3. 70B coding BigCodeBench 12.2 → 23.0 → 21.6.
- Persona data = Tülu 3 Persona datasets built to target mathematics, coding, and instruction following (§4.2, Table 10).
- Table 32 (8B SFT ablations, columns MMLU, GPQA, BBH, AGIEval, MATH, DM Math, HumanEval, BigCodeBench, IFEval, IFEval-OOD): final 62.1, 31.9, 67.9, 56.2, 31.5, 32.3, 86.2, 11.5, 72.8, 17.6 (dev avg 64.1, unseen 29.9); w/o WildChat 61.0, 31.5, 65.6, 53.1, 31.8, 31.2, 85.3, 7.4, 70.1, 20.8; w/o Safety 62.0, 31.9, 68.3, 55.6, 32.6, 32.6, 84.5, 10.8, 71.0, 17.6; w/o Persona 62.4, 29.5, 68.3, 56.9, 30.1, 31.8, 84.5, 10.8, 53.6, 18.0; w/o Math 62.2, 32.6, 68.9, 54.1, 23.5, 23.3, 86.0, 8.8, 70.6, 18.3.
- Authors' reading: SFT data choices generalize on average but overfit development evaluations in precise instruction following and to some extent knowledge recall and reasoning (§7.4.1). DPO data scaling overfit MATH; hypothesized cause: LaTeX formatting on DeepMind Mathematics interfered with reasoning and answer extraction (Fig. 24). For the four models with both scores printed, IFEval is 80.6-88.0 and IFEval-OOD 24.3-34.5 (Tülu 3 8B 82.4/24.3, Llama 3.1 8B Instruct 80.6/26.1, Tülu 3 70B 83.2/27.8, Llama 3.1 70B Instruct 88.0/34.5; Tables 5, 6, 33); the authors hypothesize overfitting to IFEval's 25 constraints (§7.4.2).

## Seeds (§4.3.1, Table 14)
- 8B SFT seeds 42, 123, 456, 789, 1011: averages 59.9, 60.1, 59.8, 59.8, 59.8; best soup (42 & 123) 60.2. 70B seeds 42, 123, 456: 71.8, 70.0, 72.6; best soup (123 & 456) 72.5. Best single run used as the final SFT model.

## Inconsistencies inside the source
- Table 31 lists the 8B SFT development average as 64.9 and Knowledge (MMLU) as 65.9; Table 32 lists 64.1 and MMLU 62.1 for the same checkpoint name.
- Table 24 and the 8B/70B results tables print MMLU 0-shot CoT, PopQA 15-shot, BBH 3-shot CoT; Table 4 (405B) labels MMLU "5 shot, CoT", PopQA "3 shot", BBH "0 shot, CoT".
- §7.4.1 prose says post-SFT stages improve harder unseen coding evaluations; Table 31 shows this at 70B, not at 8B.

## Verification
- Read on 2026-09-15 against arXiv:2411.15124v5 PDF text (§2, §3.2, §4.3.1, §7, App. B.2).
