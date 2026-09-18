<!-- scope: Demystifying RL in Agentic Reasoning (Oct 2025) — controlled study of SFT data (real vs stitched tool-use trajectories), RL data diversity and model-aware difficulty, GRPO variants (token/sequence loss, clip-higher, overlong shaping), entropy, and tool-call modes with a code interpreter; releases DemyAgent-4B
     deps: [[grpo]], [[dapo]]
     see-also: [[gspo]], [[entropy-mechanism-llm-rl]], [[high-entropy-minority-tokens]], [[pass-at-k-training]], [[simpletir]], [[skywork-or1-rl-data]]
-->

# Demystifying Reinforcement Learning in Agentic Reasoning
- **Core Insight:** With Qwen3-4B-Instruct-2507 and a code-interpreter tool, SFT on 3K real end-to-end tool-use trajectories gives AIME2025 average@32 of 29.79% versus 3.65% for stitched synthetic trajectories (Table 1), and subsequent GRPO with token-level loss, clip-higher, and overlong reward shaping reaches 70.93%/68.13% average@32 on AIME2024/2025 within 450 steps versus a best of 54.7%/40.93% for the GRPO baseline (§4.1).
- **Guideline:** When cold-starting a tool-using model for RL, use SFT trajectories generated end to end by a model that actually calls the tool rather than trajectories where reasoning steps were replaced by tool calls afterward, because the stitched ReTool SFT set left both Qwen2.5-7B and Qwen3-4B below 7% average@32 on AIME2024/2025 (Table 1); tune ε_high per model, because 0.315 sped up training relative to 0.28 but 0.35 gave worse training effectiveness for Qwen3-4B (§4.3, Fig. 6).
- **Authors:** Zhaochen Yu, Ling Yang, Jiaru Zou, Shuicheng Yan, Mengdi Wang (National University of Singapore, UIUC, Princeton University)
- **Year:** 2025 (arXiv v1 2025-10; no later version)
- **URL:** https://arxiv.org/abs/2510.11701 (code and models: https://github.com/Gen-Verse/Open-AgentRL)
- **Source type:** paper
- **Relevant topics:** agentic RL, tool-integrated reasoning, cold-start SFT data, RL data diversity, difficulty filtering, GRPO variants, clip-higher, overlong reward shaping, policy entropy, pass@k vs average@k, long-CoT models and tool use

## Abstract
The paper studies RL for agentic reasoning along three axes: data, algorithm, and reasoning mode. (i) Real end-to-end tool-use trajectories give a stronger SFT initialization than stitched synthetic trajectories, and high-diversity, model-aware RL datasets sustain exploration and improve RL. (ii) Exploration-friendly techniques (clip-higher, overlong reward shaping, adequate policy entropy) improve training efficiency. (iii) A deliberative strategy with fewer tool calls outperforms frequent tool calls or verbose self-reasoning. The authors release a real end-to-end agentic SFT dataset and an RL dataset, and report that a 4B model trained with these practices exceeds 32B models on AIME2024/2025, GPQA-Diamond, and LiveCodeBench-v6.

## Key Contributions
- Real vs synthetic SFT data on two base models, measured by average@32, pass@32, maj@32 (§3.1, Table 1).
- RL data: diverse 30K math/science/code set vs DAPO-Math-17k (§3.2); model-aware difficulty resampling for the weaker model (§3.3).
- Three GRPO recipes: GRPO-T (baseline), GRPO-TCR (Token-level loss, Clip-higher, overlong Reward shaping), GRPO-SCR (Sequence-level loss, same other parts) (§2.3, §4.1); ε_high sweep 0.28/0.315/0.35 (§4.3).
- Reasoning-mode analysis: reactive vs deliberative tool use (§5.1); long-CoT starting models (§5.2-5.3).
- DemyAgent-4B, Qwen3-4B-RA-SFT, Qwen2.5-7B-RA-SFT, and both datasets (§6).

## Key Figures/Tables to Study
- **Table 1:** real vs stitched SFT trajectories. **Fig. 2 / Fig. 3:** diverse and model-aware RL data.
- **Fig. 4:** the three recipes (average@32, pass@32, maj@32). **Figs. 5-6:** entropy and ε_high sweep.
- **Figs. 7-8:** tool calls, response length per round, tool-use success. **Figs. 9-10:** long-CoT models. **Table 2:** final comparison.

## Technical Details
- Composite reward: r = 1 + 0.1n if the answer matches, and "min(−1 + 0.1n)" otherwise, where n = number of tool invocations; the text says the tool bonus is clipped to prevent tool abuse, but the printed min has one argument and no bound (§2.2, Eq. 8).
- Overlong shaping: r_length = 0 for |y| ≤ L_max − L_cache; ((L_max − L_cache) − |y|)/L_cache for L_max − L_cache < |y| ≤ L_max; −1 for |y| > L_max (Eq. 9). L_max and L_cache values are not given. GRPO-TCR and GRPO-SCR use r_out+tool + r_length; GRPO-T uses r_out+tool (§4.1).
- Loss aggregation: token-level averages over all tokens of all G responses (Eq. 4); sequence-level averages per response with a length-normalized sequence importance ratio (Eqs. 5, 7).
- SFT data (App. A.3): teacher Qwen3-Coder-30B-A3B in the Qwen-Agent framework with SandboxFusion as the code interpreter, on 6K problems (s1-1k, 3K self-curated LeetCode, 2K ReTool multi-turn set). LeetCode and ReTool trajectories are scored with ReasonFlux-PRM; the top 1K of each plus s1-1k form the 3K set.
- Synthetic baseline: the ReTool multi-turn SFT set, where long-CoT steps are replaced by tool calls and responses (§3.1).
- Table 1 (Qwen3-4B-Instruct-2507, synthetic → real, AIME2025): average@32 3.65% → 29.79%, pass@32 22.22% → 72.88%, maj@32 0.10% → 45.82%. Qwen2.5-7B-Instruct AIME2025: 5.21% → 18.24%, 25.56% → 48.42%, 12.08% → 29.18%. The §3.1 text prints 29.97% and 45.22% for the Qwen3-4B values.
- RL data (App. A.4): 17K DAPO-Math, 4,902 math and 3,586 code problems from Skywork-OR1, 3K MegaScience science problems, described as 30K (sum 28,488, derived). With the diverse set, average@32 on AIME2025 exceeds 50% within 150 steps vs 220 steps for DAPO-Math-17k (§3.2).
- Model-aware set (§3.3): the SFT model does 8 rollouts per problem; 0% and 100% problems are removed; easy ≥ 0.75, medium, hard ≤ 0.25; Qwen2.5-7B's set is resampled to match Qwen3-4B's difficulty histogram. Motivation: with the full set, Qwen2.5-7B's average reward stayed near zero.
- Loss granularity (§4.1): on Qwen3-4B-RA-SFT, token-level exceeds sequence-level by 3.95 points (AIME24) and 3.86 (AIME25) at equal budget; on Qwen2.5-7B-RA-SFT they are comparable. GRPO-TCR reaches GRPO-T's best within 100 steps, "25% of the training computation".
- ε_high (§4.3, Fig. 6): 0.315 reaches at step 60 the result that 0.28 reaches at step 100 ("40% faster"); 0.35 on Qwen3-4B lifts faster early but trains worse than 0.28.
- Tool use (§5.1, Fig. 8): deliberative-mode runs (fewer calls, longer reasoning per round) exceed 70% tool-call success; reactive-mode runs are lower; the strongest runs are deliberative.
- Table 2 (AIME24, AIME25, GPQA-Diamond, LiveCodeBench-v6): DemyAgent-4B 72.6, 70.0, 58.5, 26.8; rStar2-Agent-14B 80.6, 69.8, 60.9, not reported; ReTool-32B 72.5, 54.3; Qwen3-4B-Instruct-2507 self-contained 63.3, 47.4, 52.0, 35.1; the same model under the agentic prompt without agentic training 17.9, 16.3, 44.3, 23.0.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-RA-SFT, Qwen2.5-7B-RA-SFT | 4B, 7B | SFT | examples; teacher; filter | 3K trajectories; Qwen3-Coder-30B-A3B; ReasonFlux-PRM top-1K selection (LeetCode, ReTool) + s1-1k | arXiv:2510.11701v1 App. A.3 | verified 2026-09-14 | Table 1: real vs ReTool synthetic |
| same | 4B, 7B | SFT | epochs; batch; optimizer; initial LR; max response length | 5; 32; AdamW; 5e-5; 32,768 | App. A.1 | verified 2026-09-14 | no ablation reported |
| DemyAgent-4B and study runs | 4B, 7B | RL | prompts | 17K DAPO-Math + 4,902 math + 3,586 code (Skywork-OR1) + 3K MegaScience ("30k") | App. A.4 | verified 2026-09-14 | §3.2 Fig. 2 vs DAPO-Math-17k |
| same | 4B, 7B | RL | epochs; batch; LR; max prompt length | 3; 64; 1e-6; 2,560 | App. A.1 | verified 2026-09-14 | no ablation reported |
| GRPO-T baseline | 4B, 7B | RL | KL coefficient; clip ε; aggregation; max response length | 0.001; 0.2; token-level; 16,384 | App. A.1, §4.1 | verified 2026-09-14 | Fig. 4 (baseline) |
| GRPO-TCR | 4B, 7B | RL | ε_low; ε_high | 0.20; 0.28 | §4.1 | verified 2026-09-14 | Fig. 4 vs GRPO-T |
| GRPO-SCR | 4B, 7B | RL | ε_low; ε_high | 0.0003; 0.0004 | §4.1 | verified 2026-09-14 | Fig. 4: token-level ahead by 3.95/3.86 on 4B |
| DemyAgent-4B | 4B | RL | algorithm; base; ε_high | GRPO-TCR; Qwen3-4B-RA-SFT; 0.315 | §6 "Training Recipe" | verified 2026-09-14 | §4.3 Fig. 6 sweep 0.28/0.315/0.35 |
| Qwen2.5-7B-RA-SFT (model-aware run) | 7B | RL | difficulty filter | 8 rollouts/problem; drop 0% and 100%; match Qwen3-4B histogram | §3.3 | verified 2026-09-14 | Fig. 3 vs full 30K |
| all RL runs | 4B, 7B | RL | samples per prompt G; rollout temperature; KL for TCR/SCR; L_max, L_cache; max tool calls; total steps | not reported | checked §2-6, App. A-B | not reported | — |
| all runs | 4B, 7B | eval-gate | temperature; top_p; max length; samples | 1.0; 0.6; 16,384; 32 (LiveCodeBench: pass@1, pass@5) | App. A.2 | verified 2026-09-14 | not applicable |
| all runs | 4B, 7B | all | framework; compute | VeRL; 8× A100-80G | App. A.1 | verified 2026-09-14 | not applicable |

## Findings relevant to generality, negative feedback, agentic training, and distillation
- **pass@k during agentic RL.** GRPO-TCR and GRPO-SCR raise pass@k and average@k together ("over 10% gains" on AIME2024/2025); GRPO-T shows the trade-off where pass@k is suppressed (§4.2, Fig. 4). The authors attribute the difference to GRPO-T's low clip bound and KL term (Interpretation).
- **Entropy.** GRPO-T's entropy collapses early; the better runs rise faster and plateau higher (§4.3, Figs. 4-5). The relation is non-monotonic: too high ε_high gives excessive entropy and instability (§4.3).
- **Negative signals (course standard §6.1).** Wrong answers get reward −1 + 0.1n and over-length outputs a penalty down to −1 (Eqs. 8-9); responses whose reward is below the group mean receive negative normalized advantages (Eq. 3), i.e., negative as gradient. All-wrong problems are removed only in the model-aware set (§3.3). No split of gains by advantage sign is reported.
- **Long-CoT starting models.** RL from Qwen3-4B-Thinking-2507 drives tool calls toward zero on reasoning tasks (§5.2, Fig. 9). SFT on the agentic trajectories restores tool use, but these models reach only performance comparable to the instruct-based models (§5.3, Fig. 10).
- **Distillation stage.** Cold-start SFT is distilled from Qwen3-Coder-30B-A3B trajectories with PRM-based selection; teacher sampling parameters are not reported (App. A.3).
- **Scope limits.** Tool: a code interpreter only (§8.3). Models: 4B and 7B only (§9). Evaluation covers AIME, GPQA-Diamond, and LiveCodeBench; no general-chat, instruction-following, or forgetting evaluation. DemyAgent-4B's LiveCodeBench-v6 score (26.8, with tools) is below Qwen3-4B-Instruct-2507's self-contained score (35.1) in the same table (Table 2).

## Connections
- [[dapo]] — source of clip-higher, overlong reward shaping, and the DAPO-Math-17k prompts used here.
- [[gspo]] — cited for sequence-level importance ratios (GRPO-SCR); [[grpo]], [[deepseekmath]], [[dr-grpo]] — GRPO and its analyses cited in §2.2 and §4.
- [[entropy-mechanism-llm-rl]], [[high-entropy-minority-tokens]], [[clip-low-clip-high-entropy]] — entropy analyses of RLVR; the first two are cited in §4.3 and §7.
- [[pass-at-k-training]], [[rlvr-beyond-base-model]] — pass@k under RLVR without tools, which §4.2 contrasts with the agentic case.
- [[skywork-or1-rl-data]], [[s1]], [[limo]] — RL and SFT data sources; LIMO and s1 are cited in §8.1 for small curated SFT sets.
- [[toolformer]] — cited in §1 as stitch-style tool-data synthesis. [[search-r1]], [[simpletir]] — other tool-integrated RL work.
- [[livecodebench]] — code benchmark in Table 2. [[qwen-3]] — base model family.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.11701 (v1, 2025-10-13, the only version).
- Audit claims not found in the source: "4,902 Skywork-OR1 math/code" → 4,902 math and 3,586 code problems (App. A.4). "Filtered by difficulty for the model at hand" → the 30K set is unfiltered; the model-aware subset is built only for Qwen2.5-7B (§3.3). "29.97% average@32 on AIME2025" → printed in §3.1 text; Table 1 prints 29.79% (conflict inside the paper). "eps_high=0.315 converged about 40% faster" → the paper says it reaches equal performance at step 60 vs 100, not convergence (§4.3). "Matching 14B-32B agents" → rStar2-Agent-14B is higher on AIME24 (80.6) and GPQA-D (60.9); DemyAgent-4B is higher only on AIME25 (70.0 vs 69.8) (Table 2).
- Internal inconsistencies: §4.1 gives "initial accuracy of 29.79% and 33.23%" in AIME2024/2025 order, while Table 1 gives 33.23% (AIME2024) and 29.79% (AIME2025). Fig. 2's caption names the ReTool dataset as the comparison; §3.2 text names DAPO-Math-17k.
- Not reported by the source: group size, rollout temperature, KL for TCR/SCR, overlong-shaping lengths, seeds or variance.
