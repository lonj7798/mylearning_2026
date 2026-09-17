---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2512.13607v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2512.13607
created_at: "2026-09-15"
---

# Excerpt: Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models

- **Authors:** Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, Yang Chen, et al. (NVIDIA)
- **Year:** 2025 (arXiv v1 2025-12; v2 2026-03-27)
- **Source type:** official technical report (models and data: huggingface.co/collections/nvidia/nemotron-cascade)
- **Used in:** [[read]] §1, §2.6, §5, Recipe, figure `figures/cascade-stage-tracker.html`

## Pipeline (§1, Fig. 2)
Multi-stage SFT on Qwen3-8B-Base and Qwen3-14B-Base, then sequential domain-wise RL: RLHF → instruction-following RL (IF-RL) → Math RL → Code RL → SWE RL. The authors contrast this with DeepSeek-R1 and Qwen3, "which blend diverse prompt distributions from all (reasoning) domains for joint RL training" (§1). Fig. 1 labels "2232 RL steps in total" for the 14B-Thinking run.

## RL objective (§4.1.2, Eq. 1)
GRPO, strictly on-policy (one gradient update per rollout group, importance ratio exactly 1), KL term removed, token-level loss; advantage Â_{i,t} = (r_i − mean({r_i})) / std({r_i}) for all t. For RLHF r_i is the scalar reward-model output.

## Authors' explanation of low forgetting (§4.1.1; Interpretation)
(i) RL data are policy-generated, so old behaviors keep being sampled if still rewarded; (ii) RL optimizes expected reward rather than token-level targets; (iii) forgetting "may still occur when the reward of a new domain sharply conflicts with that of a previous one ... particularly when prompts from different domains are semantically similar"; (iv) prompt overlap across stages is minimized, math and competitive-programming prompts are removed from RLHF, and stages go from general (RLHF, IF-RL) to specialized (math, code, SWE).

## RLHF stage (§4.2–4.3, §6)
- Reward model: Qwen2.5-72B-Instruct with a linear head, Bradley–Terry loss, 82K preference pairs (HelpSteer2 10K, HelpSteer3 filtered to 36K, plus generated pairs); batch 256, LR 2e-6, 1 epoch (§4.2).
- RLHF prompts are a subset of the RM's preference prompts; "failing to exclude math-related prompts during RLHF resulted in a 2% performance drop on the AIME25 benchmark" (§4.3.1).
- Thinking mode: only the text after the end-of-thinking token goes to the RM; if thinking does not terminate, the whole unfinished response is scored (§4.3.2).
- Code-switch penalty: mixed-language responses to English prompts get "the lowest score in the batch minus 10" (§4.3.2).
- Unified model: equal split of thinking and non-thinking prompts per batch; "Half-Half" gave the highest ArenaHard and better math and code than either single mode (§6.1, Fig. 9).
- RM size (on AceReason-Nemotron-1.0-7B): the 7B RM showed length-driven reward gains; the 72B RM gave about 3% higher AIME25 than the 7B RM (§6.2).
- Settings, body text (§4.3.2): max response 12K, no overlong filtering, batch 128, 8 rollouts, temperature 0.6, top-p 0.95, LR 2e-6, entropy and KL coefficients 0, about 800 steps. Appendix Table 15: batch 256; steps 800 (8B) and 900 (14B).
- Authors' explanation for the IFEval drop after RLHF: prompt overlap with IFEval and RM preferences that "conflict with the strict instruction-following constraints" (§4.3.3).

## IF-RL stage (§4.4)
- Data: 40K filtered Llama-Nemotron IF prompts, 60K LMSYS-Chat-1M prompts with IFEval constraints, and IF-RLVR data of Pyatkin et al.; stage 1 IFEval taxonomy, stage 2 IF-Bench-Train taxonomy (§4.4.1–4.4.2).
- A rule-based verifier alone "degraded human alignment results" (§4.4.2).
- Unified model: RLHF in both modes, then IF-RL only in non-thinking mode; reversing RLHF and IF-RL order gave "much worse results" (no numbers) (§4.4.2).
- Thinking model reward: r_i = R_IF(o_i) + sigmoid(R̂_RM(o_i)) if R_IF(o_i) = 1, else 0; R̂_RM is the group-normalized RM score (§4.4.2).
- Body text: batch 128, 8 samples, temperature 0.6, LR 2e-6, KL 0; thinking-mode stage 1 about 500 steps, stage 2 about 300. Table 16: batch 256; 14B-Thinking 800 and 120 steps; 8B-Thinking 550 and 300.
- After IF-RL: reasoning benchmarks show minor drops "fully recoverable" in later stages except ArenaHard; entropy and reasoning length decrease (§4.4.3, Fig. 8).

## Later stages (§4.5–4.7)
- Math RL: 18K AceReason-Math problems, 9-gram decontamination; "Applying Math RL directly to RLHF checkpoints yielded very similar results" (§4.5). Code RL: a −1 code-switching reward hurt coding, so 0 is used (§4.6.2). Table 18 (14B Math RL): 28K → 40K max length, LR 2.5e-6, temperature 1.2 → 1.1, steps 0–120 and 120–220. Table 19 (14B Code RL): 56K, LR 4e-6, 64 steps, temperature 1.0. Table 20 (14B SWE RL): 16 rollouts, LR 2.5e-6, 120 steps.

## Per-stage results (Tables 2, 4, 5, 6, 7, 8; pass@1)
14B-Thinking, SFT → RLHF → IF-RL → Math RL → Code RL → SWE RL:
| Benchmark | SFT | RLHF | IF-RL | Math | Code | SWE |
|---|---|---|---|---|---|---|
| MMLU-Pro | 76.0 | 77.7 | 76.4 | 76.9 | 77.6 | 77.0 |
| GPQA-Diamond | 68.3 | 71.3 | 70.1 | 67.6 | 70.3 | 69.6 |
| ArenaHard | 76.9 | 93.1 | 90.2 | 89.3 | 89.8 | 89.5 |
| IFEval (strict prompt) | 69.8 | 56.2 | 81.3 | 84.3 | 81.8 | 81.9 |
| IFBench | 24.3 | 25.6 | 40.4 | 41.0 | 41.0 | 41.7 |
| AIME 2025 | 81.1 | 81.8 | 82.3 | 83.3 | 83.5 | 83.3 |
| LiveCodeBench v6 | 63.1 | 72.3 | 72.7 | 72.4 | 74.8 | 74.6 |
| SWE-bench Verified | 34.5 | 38.8 | 38.4 | 39.7 | 39.6 | 43.1 |
The 8B unified and 8B-Thinking columns are in the same tables; the unified model's IFEval is measured in non-thinking mode. The authors attribute most small differences after Math RL and Code RL to "evaluation variance and checkpoint selection" (§4.5.3, §4.6.3). One score per stage; no seeds reported.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2512.13607 (v2, 2026-03-27): Abstract, §1, §2, §4.1–4.7, §6.1–6.2, App. D Tables 15–20.
- Inconsistencies inside the source: Table 4 marks the 14B-Thinking IFEval score after RLHF as ↓12.4, while the printed SFT and RLHF scores (69.8 in Table 2 and 56.2 in Table 4) differ by 13.6; Table 1's ↑12.1 for the final model is consistent with the printed SFT score. Table 4 also marks the 14B-Thinking ArenaHard change as ↑19.2, while the printed scores 76.9 (Table 2) and 93.1 (Table 4) differ by 16.2. The corresponding arrows for the 8B-Thinking and 8B unified columns agree with their printed scores (checked 2026-09-15).
- Conflicts inside the source: RLHF and IF-RL batch size 128 (body) vs 256 (Tables 15–16); IF-RL thinking-mode steps "around 500/300" (body) vs 550/300 (8B) and 800/120 (14B) (Table 16); IF-RL thinking stage-1 overlong filtering "with" (body) vs False (Table 16).
- Not reported: seeds or variance per stage; a mixed-domain RL baseline with the same data and compute.
