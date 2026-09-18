<!-- scope: AgentTuning (arXiv 2310.12823): the AgentInstruct trajectory dataset (1,866 GPT-4 trajectories, 6 AgentBench tasks) and η-weighted mixing with ShareGPT to train AgentLM-7B/13B/70B
     deps: [[self-instruct]]
     see-also: [[agent-flan]], [[fireact]], [[lumos]]
-->

# AgentTuning: Enabling Generalized Agent Abilities for LLMs
- **Core Insight:** Fine-tuning Llama-2-chat on data sampled 0.2 from 1,866 filtered GPT-4 agent trajectories and 0.8 from ShareGPT raises the held-out agent score of the 70B model from 0.51 to 1.40 (+176%) with general score 0.96 vs 0.95 (Table 4), while agent-only training at 7B drops held-out to 0.09 (Table 5).
- **Guideline:** When adding agent trajectories to SFT, mix them with general instruction data and pick the mixture weight on held-out agent tasks, because at 7B agent-only data scored held-out 0.09 and general 0.22 versus 0.67 and 0.63 for the η = 0.2 mixture (Table 5). At 7B and 13B the mixture's held-out score is close to general-only training (0.67 vs 0.64; 0.78 vs 0.81), so the held-out advantage of mixing over general-only data appears only at 70B (1.40 vs 0.98).
- **Authors:** Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, et al. (Tsinghua University; Zhipu.AI)
- **Year:** 2023 (arXiv v1 2023-10; v2 2023-10-22)
- **URL:** https://arxiv.org/abs/2310.12823 (code and data: https://github.com/THUDM/AgentTuning)
- **Source type:** paper
- **Relevant topics:** agent SFT, trajectory synthesis and filtering, data mixing, held-out agent generalization, general-ability retention, contamination analysis

## Abstract
Open LLMs score below commercial models such as GPT-3.5 and GPT-4 on agent tasks (Fig. 1b, reprinted from AgentBench: average overall score 0.42 for open vs 2.24 for API-based models). AgentTuning aims to improve agent ability without losing general ability. It builds AgentInstruct, a dataset of high-quality interaction trajectories, and uses a hybrid instruction-tuning strategy that mixes AgentInstruct with open general-domain instructions. Applied to the Llama 2 chat series, it produces AgentLM-7B, 13B, and 70B. AgentLM-70B is reported as comparable to GPT-3.5-turbo on unseen agent tasks, and general abilities are maintained. The dataset and models are released.

## Key Contributions
- AgentInstruct: 1,866 filtered trajectories with ReAct thoughts from six AgentBench tasks (§2.1, Table 1).
- A hybrid mixture objective with a scanned agent weight η (§2.2.2, Eq. 1).
- Evaluation on 6 held-in, 6 held-out, and 4 general tasks at three model sizes (§3, Tables 3-4).
- Ablations on trajectory filtering, agent vs general data, and single-task training (Table 2, Table 5, Fig. 3b); error analysis (Fig. 3a); contamination analysis (App. B).

## Key Figures/Tables to Study
- Table 1: instructions, filtered trajectories, turns, and keep ratio per task.
- Table 4: all held-in, held-out, and general scores. Table 5: agent-only vs general-only vs mixed at 7B/13B/70B.
- Fig. 3: error types and per-task transfer heatmap. App. A Table 6: hyper-parameters. App. B Table 7: contamination.

## Technical Details
- **Trajectory.** A conversation (u1, a1, ..., un, an) with final reward r ∈ [0, 1] (§2). Generation used gpt-4-0613 and gpt-3.5-turbo-0613 (§2.1).
- **Instructions.** ALFWorld, WebShop, Mind2Web, Knowledge Graph use their train splits. Operating System uses Self-Instruct: GPT-4 writes a task, reference solution, and evaluation script; a second GPT-4 solves it; trajectories are kept when both answers agree. Database uses Task Derivation from BIRD (reference-SQL trajectories with GPT-4-filled thoughts, fixed at 2 turns; and GPT-4 interaction checked against the reference SQL result) plus Self-Instruct for INSERT/UPDATE/DELETE (§2.1.1).
- **Counts (Table 1).** Instructions → filtered trajectories (avg turns, keep ratio): ALFWorld 954 → 336 (13.52, 35.2%); WebShop 1,485 → 351 (3.68, 23.6%); Mind2Web 23,378 → 122 (1.001, 0.52%; Mind2Web is evaluated step-wise with teacher forcing, Table 1 note); KG 2,501 → 324 (6.04, 13.0%); OS 647 → 195 (3.85, 30.1%); DB Self-Instruct 1,074 → 178 (2.13, 16.6%); DB Task Derivation 5,302 → 360 (2.03, 6.79%); total 35,341 → 1,866 (5.24, 5.29%).
- **Interaction.** GPT-4 acts as the agent with a 1-shot example; Mind2Web partly used gpt-3.5-turbo for budget reasons. An episode ends at the goal or the token limit; three identical consecutive outputs count as a repetition failure; malformed actions are mapped to the closest valid action by BLEU. Every action has a ReAct thought (§2.1.2).
- **Filtering.** Keep r = 1 for all tasks except Mind2Web, which uses r ≥ 2/3. At 7B, filtered data scores held-in 1.96 / held-out 0.65 vs 1.34 / 0.47 unfiltered (§2.1.3, Table 2).
- **General data.** English ShareGPT: 57,096 GPT-3.5 and 3,670 GPT-4 conversations, sampled GPT-4 : GPT-3.5 = 1 : 4 (§2.2.1; App. A gives 0.2 : 0.8).
- **Mixture objective.** J(θ) = η · E_{(x,y)~D_agent}[log π_θ(y|x)] + (1 − η) · E_{(x,y)~D_general}[log π_θ(y|x)] (§2.2.2, Eq. 1). π_θ(y|x) is the fine-tuned model's probability of response y given instruction and history x; D_agent is AgentInstruct; D_general is ShareGPT; η is the agent mixture ratio. The paper says J is minimized while writing log-likelihood terms; App. A calls η the AgentInstruct sampling ratio. η was scanned from 0 to 1 in steps of 0.1 on 7B, and η = 0.2 was best on held-out tasks (§2.2.2).
- **Evaluation (Table 3).** Held-in: ALFWorld, WebShop, Mind2Web, KG, OS, Database. Held-out: SciWorld, MiniWoB++, HotpotQA, WebArena, ReWOO, Digital Card Game. General: MMLU, HumanEval, GSM8K, MT-Bench. Each task score is normalized to an average of 1 across evaluated models before weighting (§3.1). Decoding is mostly greedy; WebArena uses nucleus sampling p = 0.9 (§3.1).
- **Main results (Table 4).** Held-in overall: AgentLM 1.96 / 2.11 / 2.55 (7B/13B/70B) vs Llama-2-chat 0.19 / 0.20 / 0.27; GPT-3.5 1.59, GPT-4 2.75. Held-out overall: AgentLM 0.67 (+76%) / 0.78 (+57%) / 1.40 (+176%) vs 0.38 / 0.49 / 0.51; GPT-3.5 1.49, GPT-4 2.13. General overall: 0.62 (−1%) / 0.69 (−7%) / 0.96 (+1%) vs 0.63 / 0.74 / 0.95.
- **70B general benchmarks (Table 4).** Llama-2-70b-chat → AgentLM-70B: MMLU 62.1 → 59.5, HumanEval 30.8 → 28.7, GSM8K 54.7 → 59.7, MT-Bench 6.85 → 7.26.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| AgentLM-7B / 13B / 70B | all | SFT | Base checkpoint | Llama-2-{7,13,70}b-chat | arXiv:2310.12823v2 §2.2.3 | verified 2026-09-14 | chosen for instruction following (§2.2.3); no ablation reported |
| AgentLM-7B / 13B / 70B | all | SFT | Agent sampling ratio η | 0.2 | §2.2.2; App. A | verified 2026-09-14 | η scanned 0-1 in 0.1 steps on 7B, best held-out (§2.2.2; per-η scores not tabulated); endpoints in Table 5 |
| AgentLM-7B / 13B / 70B | all | SFT | ShareGPT GPT-3.5 : GPT-4 sampling ratio | 0.8 : 0.2 | §2.2.1; App. A | verified 2026-09-14 | motivated by GPT-4 response quality (citing Wang et al. 2023a); no ablation reported |
| AgentLM-7B / 13B / 70B | all | SFT | Agent trajectories; filter | 1,866; r = 1 (Mind2Web r ≥ 2/3) | Table 1; §2.1.3 | verified 2026-09-14 | Table 2 (7B): filtered 1.96 / 0.65 vs unfiltered 1.34 / 0.47 |
| AgentLM-7B | 7B | SFT | Peak LR | 5e-5 | §2.2.3; App. A Table 6 | verified 2026-09-14 | no ablation reported |
| AgentLM-13B | 13B | SFT | Peak LR | 5e-5 | §2.2.3; App. A Table 6 | verified 2026-09-14 | no ablation reported |
| AgentLM-70B | 70B | SFT | Peak LR | 1e-5 | §2.2.3; App. A Table 6 | verified 2026-09-14 | no ablation reported |
| AgentLM-7B / 13B / 70B | all | SFT | Batch size (unit not stated); sequence length | 64; 4,096 | §2.2.3; Table 6 | verified 2026-09-14 | no ablation reported |
| AgentLM-7B / 13B / 70B | all | SFT | Warmup ratio; decay ratio; schedule | 0.02; 0.9; cosine | Table 6 | verified 2026-09-14 | no ablation reported |
| AgentLM-7B / 13B / 70B | all | SFT | Optimizer; β1, β2; ε; weight decay; grad clip | AdamW; 0.9, 0.95; 1e-8; 0.1; 1.0 | §2.2.3; Table 6 | verified 2026-09-14 | no ablation reported |
| AgentLM-7B / 13B / 70B | all | SFT | Hidden dropout; attention dropout | 0.05; 0 | Table 6 | verified 2026-09-14 | no ablation reported |
| AgentLM-7B / 13B / 70B | all | SFT | Loss masking; framework | loss on model output only; Megatron-LM (TP for 7B/13B, TP + PP for 70B) | §2.2.3 | verified 2026-09-14 | no ablation reported |

Not reported (checked body, App. A-D): epochs or steps, total training tokens, number of ShareGPT conversations actually sampled, GPU type and count, compute time, generation cost.

## Findings relevant to generality, negative feedback, agentic training
- **General vs agent data (Table 5).** Held-in / held-out / general. 7B: mixed 1.96 / 0.67 / 0.63; general-only 0.38 / 0.64 / 0.61; agent-only 1.34 / 0.09 / 0.22. 13B: 2.11 / 0.78 / 0.69; 0.43 / 0.81 / 0.63; 1.57 / 0.10 / 0.19. 70B: 2.55 / 1.40 / 0.96; 0.99 / 0.98 / 1.00; 2.47 / 0.87 / 0.83. The authors conclude that general data is needed for agent generalization and speculate that a certain model size is needed for held-out gains beyond general-only training (Interpretation, §3.4).
- **Retention.** General overall changes −1%, −7%, +1% at 7B/13B/70B (Table 4). The 13B drop is the largest.
- **Contamination.** 10-gram token matching with up to 4 mismatched tokens; an example is dirty above 80% contaminated tokens and clean below 20%. Overall rate 15.58%; 7 dirty of 1,021 held-in examples (ALFWorld 6, WebShop 1) (App. B, Table 7).
- **Negative samples.** Failed trajectories (r < 1) are discarded, i.e. treated as negative marginal value; no negative-gradient training is used. Training on unfiltered trajectories lowers both held-in and held-out scores (Table 2).
- **Error types.** On ALFWorld, WebShop, and KG, AgentLM makes fewer elementary errors (invalid actions, repetition, refusal) than Llama 2 (§3.3, Fig. 3a). Single-task training mainly improves that task; Mind2Web transfers least, possibly due to its single-round format (§3.4, Fig. 3b).

## Connections
- [[agent-flan]] — reuses this AgentInstruct dataset, re-implements AgentTuning as a baseline, and states a 1:1 ShareGPT mix.
- [[self-instruct]] — method used to build OS and Database instructions.
- [[fireact]] — concurrent ReAct-trajectory fine-tuning work; not cited in this paper.
- [[agentinstruct]] — different artifact with the same name (Microsoft, 2024).
- [[lumos]], [[swe-gym]], [[apigen-mt]] — later agent-data work; not compared here.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2310.12823 (arXiv v2, 22 Oct 2023; full PDF incl. App. A-B; page 5 rendered to confirm r ≥ 2/3).
- Corrections: "agent : general = 1:10 by examples, empirically optimal" → η = 0.2 agent sampling ratio chosen by a 0-1 scan on 7B held-out tasks (§2.2.2, App. A). "Mixing ablation 1:1 (−5 MT-Bench), 1:10, 1:50 (gains halved)" → the reported ablation is agent-only vs general-only vs mixed (Table 5). "Filtering: task success, >30-step length filter, LLM-judge pass" → r = 1, Mind2Web r ≥ 2/3 (§2.1.3). "AgentBench overall 4.02 vs 1.58" → held-out 1.40 vs 0.51, held-in 2.55 vs 0.27 on the paper's own normalized scores (Table 4). "Teacher GPT-4 (Oct-2023 version)" → gpt-4-0613, with gpt-3.5-turbo-0613 for part of Mind2Web (§2.1). "Ancestor of Agent-FLAN at millions of samples" → Agent-FLAN uses 24,703 agent samples (Agent-FLAN Table 6).
- Removed as unsupported: "environment provides real, not simulated, observations", "actions range from text commands to clicks to SQL", "avg 8-15 turns", "avg 12 turns, max 30", "~22K turn-level pairs", "~5M training tokens", "~$20K GPT-4 cost", "MMLU/MT-Bench/BBH within 1 point" (BBH not evaluated), "first public agent-SFT set at this scale", risk bullets not stated by the paper.
