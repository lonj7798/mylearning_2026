<!-- scope: Arena Learning (WizardLM team, Microsoft, 2024) — simulated offline chatbot arena with Llama3-70B-Instruct as judge; battle losses select SFT targets and DPO/PPO pairs over three iterations; WizardArena offline Elo benchmark
     deps: [[wizardlm]], [[evol-instruct]], [[dpo]], [[ppo]]
     see-also: [[judge-llm-bias]], [[self-rewarding-lm]], [[orca-math]], [[instag]], [[ifd]], [[deita]], [[hh-rlhf]]
-->

# Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena
- **Core Insight:** Keeping only the instructions on which the target model loses a judged pairwise battle against stronger models, and training on the winner's response (SFT) or the win/loss pair (DPO, PPO reward model), raised WizardLM-β-7B (Mistral-7B) from 871 to 1274 WizardArena-Mix Elo and MT-Bench from 6.41 to 8.16 over three iterations, while SFT data per iteration shrank from 14.6K to 7.8K (§3.4, Fig. 8; §3.5, Table 9).
- **Guideline:** When selecting post-training data from a large instruction pool, use pairwise judged battles against several stronger models to pick the instructions where the current model is weakest; at equal size (10K), this pair-judge selection beat random, K-Means, length, IFD and INSTAG selection on WizardArena Elo and MT-Bench (§3.5, Table 5).
- **Authors:** Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Qingwei Lin, Jianguang Lou, Shifeng Chen, Yansong Tang, Weizhu Chen (Microsoft Corporation; Tsinghua University; SIAT-UCAS)
- **Year:** 2024 (arXiv v1 2024-07-15; only version)
- **URL:** https://arxiv.org/abs/2407.10627
- **Source type:** paper
- **Relevant topics:** LLM-as-judge, data flywheel, weakness-targeted data selection, iterative SFT/DPO/PPO, offline Elo evaluation, judge-human agreement

## Abstract
Human-annotated Chatbot Arena battles are effective for evaluation but slow and costly. Arena Learning simulates such battles offline with an LLM judge and uses the outcomes both to evaluate models and to generate post-training data for SFT and RL. WizardArena, an offline test set, predicts Elo rankings consistent with the online LMSYS Arena. A data flywheel iteratively updates training data from the target model's battle losses so that it learns from the strengths of several models. The method is applied to WizardLM-β and is stated to underlie WizardLM-2.

## Key Contributions
- Judge protocol: Llama3-70B-Chat/Instruct scores two responses 1–10 with explanations; each pair is judged twice with positions swapped (§2.2, §3.1).
- Iterative flywheel over SFT → DPO → PPO with disjoint data shards per stage (§2.3.2, Table 1).
- WizardArena-Mix: 1,000 diverse (K-Means) + 1,000 hard (GPT-4-rated) multi-turn test instructions (§3.1).
- Consistency of WizardArena with LMSYS Arena rankings: 98.79% average vs 87.88% for Arena-Hard-v0.1 and 42.95% for MT-Bench (§3.2, Table 3).
- Ablations: selection strategy, threshold K, number of battle models, battle modes, judge choice, stronger battle models (§3.5).

## Key Figures/Tables to Study
- Table 1: which model battles on which shard at each stage. Table 3: offline–online ranking consistency.
- Fig. 8: per-stage Elo / MT-Bench / Arena-Hard / AlpacaEval for 7B. Table 5: data-selection comparison.
- Table 9: SFT data count and difficulty per iteration. Table 10: which battle model wins how much data.

## Technical Details
**Data (§3.1).** Initial SFT-I0 on 10K random ShareGPT samples. Pool: open instruction datasets, filtered by (1) illegal/toxic removal, (2) instruction length < 10 removed, (3) duplicate 10-character prefixes removed, (4) MinHashLSH dedup, (5) gte-large embedding decontamination against the top-5 nearest benchmark items (WizardArena, Arena-Hard Auto, MT-Bench, AlpacaEval, OpenLLM Leaderboard), (6) non-English removed → 276K, split into 9 parts. Each iteration I1–I3 uses 90K.

**Flywheel (§2.3.2, Table 1).** SFT-I1: SFT-I0 battles models M on D1; keep instances where the target loses; winner response is the target; train from Mistral-Base on D0 ∪ D1. DPO-I1: SFT-I1 battles M on D2; win/loss → chosen/rejected. PPO-I1: DPO-I1 battles M on D3; pairs train the reward model and PPO. Iteration 2 starts from the best PPO-I1 as competitor and retrains SFT from Mistral-Base on D0 ∪ D1 ∪ D4, and so on.

**Battle models (§3.1).** 7B: {Command R+, Qwen1.5-72B-Chat, OpenChat-3.5}. 8x22B: {GPT-4o, GPT-4-1106-preview, WizardLM-2-8x22B-0415}. Elo by Bradley–Terry with 100 bootstraps, median reported.

**Threshold K (§3.5, Fig. 9, Table 9).** Described as controlling SFT data size and the chosen–rejected gap; the exact definition (judge-score margin) is not written out. Best K in I1: 3.0 for SFT (30K → 14.6K, Elo 1047 → 1117 vs K = 0), 2.0 for DPO (→ 18.1K, Elo 1165 → 1187). K lowered to 1.0 in I2 and I3.

**Results.** 7B per iteration on WizardArena-Mix: +343, +32, +28 Elo (§3.3). 8x22B: +460 Elo, +2.07 MT-Bench (§3.3). Table 8, 7B I0 → I3: Arena-Hard Auto 5.2 → 31.5; AlpacaEval 2.0 LC 8.2 → 34.2; OpenLLM average 57.75 → 68.08. 8x22B-I3: Arena-Hard 64.3, AlpacaEval LC 48.9.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WizardLM-β-7B (Mistral-7B) | 7B | SFT | epochs; LR; batch; seq len | 3; 5e-6; 128; 4096 | arXiv:2407.10627v1 §3.1 | verified 2026-11-24 | no ablation reported |
| WizardLM-β-7B | 7B | preference | DPO LR; epochs; β | 5e-7; 2; 0.3 | §3.1 | verified 2026-11-24 | Fig. 8: DPO adds Elo at each iteration |
| WizardLM-β-7B | 7B | reward-model | base; epochs; LR | Mistral-7B; 1; 1e-6 | §3.1 | verified 2026-11-24 | no ablation reported |
| WizardLM-β-7B | 7B | RL (PPO) | LR; epochs; KL coefficient | 1e-7; 1; 0.4 | §3.1 | verified 2026-11-24 | Fig. 8: PPO adds Elo at each iteration |
| WizardLM-β-7B | 7B | data selection | battle models; threshold K (SFT / DPO, I1) | 3 models; 3.0 / 2.0 | §3.5, Fig. 9–10 | verified 2026-11-24 | Fig. 10: 0→3 battle models, Elo 876 → 1159 |
| WizardLM-β-7B | 7B | SFT | data per iteration (selected) | 14.6K → 11.3K → 7.8K (33.7K total of 90K) | Table 9 | verified 2026-11-24 | Table 9: mean difficulty 4.7 (pool) → 5.8 → 6.5 → 7.4 |
| Judge | 70B | eval-gate | judge; position swap; Elo bootstrap | Llama3-70B-Instruct; 2 games; 100 | §2.2, §3.1 | verified 2026-11-24 | Table 6: 99.26% Spearman vs GPT-4 judge |

## Findings relevant to generality and negative feedback
- **Negative signal source.** Rejected responses are the target model's own losing answers; chosen responses are other models' winning answers, so DPO/PPO pairs are off-policy on the chosen side. The target model's share of chosen responses in DPO rose 1.1K → 1.6K → 2.3K across iterations (Table 10).
- **Data shrinks as the model improves.** Fewer losses per iteration; selection shifts toward Coding, Math and Reasoning and away from Writing and Roleplay (Table 9; Fig. 11).
- **Generality.** OpenLLM Leaderboard average (ARC, HellaSwag, MMLU, TruthfulQA) rises 57.75 → 68.08 for 7B (Table 8); no forgetting slice or safety evaluation is reported.
- **Judge dependence.** The only stated limitation: if the judge fails to imitate human evaluators, rankings and training data are compromised (§5). Training and evaluation use the same judge model family, and battle models include models evaluated on WizardArena, which the paper does not analyse as a circularity risk.
- **Stronger opponents help.** Battling {GPT-4o, GPT-4-1106, WizardLM-2-8x22B} instead of the 7B default set gives 45–55 Elo more at each I1 stage (Table 11).

## Connections
- [[wizardlm]], [[evol-instruct]]: same team; WizardArena subsets are framed after in-breadth / in-depth evolution (§2.4).
- [[instag]], [[ifd]]: selection baselines beaten in Table 5.
- [[judge-llm-bias]]: position bias motivates the two-game swap.
- [[self-rewarding-lm]]: another loop where a model judge creates preference data.
- [[orca-math]]: teacher-agreement labeling of student samples, a single-judge analogue.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2407.10627 (v1 PDF, full text).
- Source inconsistencies: Table 2 lists 3 days inference + 6 days judging (text: 9 days on sixteen 80G GPUs) and "40x" speed-up vs "∼1 year" for LMSYS; the text says 12 months. The judge is named both "Llama3-70B-Chat" and "Llama3-70B-Instruct".
- Not reported: exact definition of threshold K; PPO batch/rollout settings; Mixtral-8x22B hyperparameters; release of WizardArena data or WizardLM-β weights.
