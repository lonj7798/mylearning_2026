<!-- scope: RLMT (Princeton): online RL (GRPO/PPO/online DPO) against a Bradley-Terry RM with a required long-CoT before the chat response; WildChat-IF prompts; base ("zero") and SFT warm-start; RM-strength ablation (Skywork-V2 vs ArmoRM); planted-hack ("absolutely") test; organic reward-hacking features (length, bullets, engagement-bait closer, self-praise)
     deps: [[bradley-terry-rm]], [[grpo]], [[ppo]]
     see-also: [[skywork-reward-v2]], [[wildchat]], [[rubicon-rubric-anchors]], [[writing-zero]], [[reward-model-overoptimization]], [[subjective-reward-guide]]
-->

# Language Models that Think, Chat Better (RLMT)
- **Core Insight:** Requiring a thinking trace before the reply and optimizing the reply against an RLHF reward model (no verifier) beat matched no-thinking RLHF on chat benchmarks by 3–8 points (§3.2), and the RM choice dominated the outcome: with the same recipe, Skywork-V2 gave a 39.0 chat average and ArmoRM 14.8 (Table 4).
- **Guideline:** When the only reward is a learned RM, measure behaviour-level hacking features from SFT to RL (length, bullet count, engagement-bait closers, self-praise) and plant a known hack to test sensitivity, because under Skywork-RM GRPO the engagement-bait rate rose +0.232 and self-praise +0.217 for the thinking model (Table 9), and a +6.81 bonus for the word "absolutely" reached 96% of rollouts by step 70 with thinking vs 19% without (Table 8).
- **Authors:** Adithya Bhaskar, Xi Ye, Danqi Chen (Princeton Language and Intelligence; Xi Ye also University of Alberta)
- **Year:** 2025 (arXiv v1 2025-09-24; v2 2026-08-16, COLM 2026)
- **URL:** https://arxiv.org/abs/2509.20357 (code/data: https://github.com/princeton-pli/RLMT)
- **Source type:** paper
- **Relevant topics:** RL with learned reward on open-ended chat, thinking for non-verifiable tasks, RM strength, prompt mixture, reward hacking measurement, online DPO vs PPO vs GRPO

## Summary
RLMT optimizes E[r(y, x)] where the model first writes a reasoning trace z and then a response y, and r is a Bradley-Terry RM (Skywork-v1-Llama-3.1-8B-v0.2) (§2.2, Eq. 3). Prompts are 7.5K WildChat-IF prompts from the Tülu 3 SFT mixture (§2.2). Two starts: SFT warm-start on 6K Gemini-2.5-Flash responses with simulated thinking, or "zero" training from base models with a fixed prefix (§2.3). 40 training runs cover Llama-3.1-8B and Qwen-2.5-7B, base and instruct, and DPO/PPO/GRPO (Abstract). Every thinking model has a matched non-thinking baseline trained on the same data with thinking removed (§3.1).

## Key Contributions
- Long CoT helps chat when trained against an RM: Llama-3.1-8B-Instruct-RLMT scores WB 50.4, AE2 58.7, AH2 22.9, CWv3 84.3 (Table 1).
- Prompt mixture ablation: WildChat-IF 35.4 chat average vs UltraFeedback 32.2 vs random Tülu 3 25.5 (Table 4).
- RM ablation: stronger RM keeps non-chat scores; weaker RM drops them (§4, "Impact of reward model").
- Reward-hacking audit with a planted hack and organic features (App. C).

## Key Figures/Tables to Study
- Table 2 (GRPO main results), Table 4 (prompt / warm-start / RM ablations), Fig. 4 (trait win rates of thoughts), Table 8 (planted hack), Table 9 (organic hacking features).

## Technical Details
**Main result, Llama-3.1-8B + SFT + GRPO (Table 2 / Table 4 main row).** Thinking: WB 38.1, AE2 52.3, AH2 15.9, chat avg 35.4, CWv3 80.9, overall avg 42.1. Non-thinking: WB 33.2, AE2 46.7, AH2 16.3, chat avg 32.1, CWv3 84.2, overall 40.5.

**RM ablation (Table 4, Llama-3.1-8B + SFT + GRPO).**
| RM | Think | WB | AE2 | AH2 | Chat avg | CWv3 | MMLU-R | Avg |
|---|---|---|---|---|---|---|---|---|
| Skywork-V2 | no | 32.6 | 47.8 | 18.8 | 33.1 | 82.2 | 55.6 | 41.0 |
| Skywork-V2 | yes | 40.9 | 53.2 | 22.8 | 39.0 | 80.4 | 47.7 | 42.6 |
| ArmoRM | no | −10.2 | 30.9 | 5.8 | 8.8 | 71.2 | 58.9 | 29.6 |
| ArmoRM | yes | −5.9 | 43.3 | 6.9 | 14.8 | 63.3 | 54.6 | 30.2 |

**Planted hack (App. C, Table 8).** +6.81 (half a reward std) added when "absolutely" appears. Rollouts containing it, no-think / think: step 0: 3% / 1%; step 40: 3% / 15%; step 70: 19% / 96%; step 100: 95% / 96%.

**Organic hacking, RL − SFT (Table 9).** Length (words): +166 no-think, +170 think. Bullets: +6.83 / +3.10. Engagement-bait closer rate: +0.176 / +0.232. Self-praise rate: +0.114 / +0.217.

**Thought traits (§4.3, Fig. 4).** After GRPO the model lists constraints first, groups ideas into themes, and revises earlier plan parts; the SFT model more often starts with a checklist and rigid step-by-step structure.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-RLMT (GRPO) | 8B | RL | prompts; samples/prompt; batch; steps | 7,560; 8; 64; 120 (1 epoch) | App. B Table 7 | verified 2026-11-24 | prompt ablation in Table 4 |
| Llama-3.1-8B-RLMT (GRPO) | 8B | RL | actor lr; KL coef; max response; temp | 1e-6 (3e-7 warm-started instruct); 0.001; 4096; 0.7 | App. B Table 7, App. B text | verified 2026-11-24 | no ablation |
| RLMT online DPO | 8B | RL | lr; β; responses per prompt | 3e-7; 0.1; 8 | App. B Table 6 | verified 2026-11-24 | Table 18 (not read in detail) |
| SFT warm-start | 8B | SFT | datapoints; lr; epochs | 6,003; 4e-6; 2 | App. B Table 5 | verified 2026-11-24 | GPT-4.1-mini variant similar (Table 4) |

## Findings relevant to generality and negative feedback
- **Generality.** MMLU-Redux dropped with thinking under Skywork-V2 (55.6 → 47.7) and under the main RM the thinking model's CWv3 was lower than non-thinking (80.9 vs 84.2) (Table 4): chat gains do not transfer to every axis.
- **Reward hacking.** Thinking models picked up the planted hack faster (Table 8), and engagement bait and self-praise grew more with thinking (Table 9). Length grew about equally (+166 vs +170).
- **Limitations (App. A).** Unclear whether gains are amplification of existing traits or new learning; seven benchmarks only.

## Connections
- [[skywork-reward-v2]]: the stronger RM in the ablation.
- [[rubicon-rubric-anchors]]: vetoes self-praise and user praise, the two features Table 9 measures.
- [[reward-model-overoptimization]]: proxy RM gains vs gold score; RLMT uses LLM-judged benchmarks as the gold proxy.

## Verification
- Read the arXiv v2 PDF (2509.20357v2) on 2026-11-24 via pdftotext (plain and -layout): Abstract, §1–4, App. A–C; Table 18 (DPO/PPO) and App. D.6 not read in detail.
- Benchmark judges are LLMs (AE2, AH2, WB, CWv3); no direct human evaluation of RLMT outputs was found in the sections read.
