<!-- scope: ETO (ACL 2024) — an SFT agent explores training tasks, its failed trajectories become the rejected side of trajectory-level DPO pairs against expert trajectories; WebShop, ScienceWorld, ALFWorld
     deps: [[dpo]], [[fireact]]
     see-also: [[agenttuning]], [[lumos]], [[rft-scaling-relationship-math]], [[ipr-step-level-refinement]], [[agent-flan]]
-->

# Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents
- **Core Insight:** Llama-2-7B-Chat trained with ETO (SFT on expert ReAct trajectories, then DPO on expert-success vs own-failure trajectory pairs) reaches average reward 67.4 on WebShop, 73.8/65.0 on ScienceWorld seen/unseen, and 68.6/72.4 on ALFWorld seen/unseen, versus 63.1, 67.4/53.0, and 60.0/67.2 for the SFT agent (Table 2).
- **Guideline:** When expert trajectories exist and the environment returns a final reward, pairing the SFT agent's lower-reward trajectories with the expert trajectory for the same task and training with DPO outperformed adding the agent's successes to SFT data (RFT) on all five test sets (Table 2); limit the number of explore-train iterations, because performance declined after the third iteration on WebShop and ScienceWorld and after the first on ALFWorld (§4.3, Fig. 4).
- **Authors:** Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, Bill Yuchen Lin (Peking University, UCLA, Ohio State University, UIUC, Allen Institute for AI)
- **Year:** 2024 (arXiv v1 2024-03; v2 2024-07; ACL 2024 Main Conference)
- **URL:** https://arxiv.org/abs/2403.02502
- **Source type:** paper
- **Relevant topics:** agent training, behavioral cloning, trajectory-level preference optimization, DPO, learning from failures, iterative offline RL, out-of-distribution evaluation

## Abstract
Open LLM agents are usually built by fine-tuning on successful expert trajectories. ETO instead lets the agent learn from its exploration failures in an iterative loop. In the exploration phase, the agent interacts with the environment on the training tasks and collects failure trajectories, which are paired with successful trajectories into contrastive pairs. In the training phase, the agent updates its policy on these pairs with a contrastive objective such as DPO. The cycle repeats. On three agent tasks, ETO exceeds the baselines, and the paper also analyses task-solving efficiency and a setting without expert trajectories.

## Key Contributions
- A three-step method: behavioral cloning (SFT) on expert ReAct trajectories; exploration by the SFT agent on the training instructions; DPO on failure-success trajectory pairs, repeated for I iterations with πref reset to the current policy (§3, Algorithm 1).
- Comparison against SFT, Best-of-N (N=10), RFT, and PPO baselines, plus GPT-4 and GPT-3.5-Turbo, on WebShop, ScienceWorld, and ALFWorld (§4.1, Table 2).
- Ablations on iteration count (§4.3), trajectory-level vs step-level vs mixed contrastive pairs (§4.4, Table 4), and self-play without expert trajectories (§4.5, Table 5).

## Key Figures/Tables to Study
- **Algorithm 1 / Fig. 2:** the explore-train loop and pair construction.
- **Table 2:** average reward for all methods; **Table 7 (App. B):** success rates.
- **Table 4:** pair granularity; **Table 5:** starting without behavioral cloning; **Fig. 4:** iterations.

## Technical Details
- Task formulation: a POMDP with reward R: S × A → [0, 1]; the final reward r(u, e) ∈ [0, 1], with 1 meaning success (§2). WebShop and ScienceWorld give dense final rewards in [0, 1]; ALFWorld gives binary rewards (§4.1).
- SFT loss is computed only on action tokens; instruction and observation tokens are masked (§3.1, Eq. 5). Each action includes a CoT rationale (ReAct style) (§3.1).
- Pair construction: e_w and e_l are the higher- and lower-reward trajectories, chosen from the expert trajectory e and the agent trajectory ê. Only pairs with different rewards are kept; if both succeed, the pair is discarded (§3.2).
- Objective: the DPO loss −E log σ(β log πθ(e_w|u)/πref(e_w|u) − β log πθ(e_l|u)/πref(e_l|u)) (Eq. 11). u = task instruction; e_w, e_l = preferred and rejected trajectories; πref = policy at the start of the iteration; β = constraint weight; σ = sigmoid.
- Data (Table 1): training instances WebShop 1,938, ScienceWorld 1,483, ALFWorld 3,321; test-seen 200/194/140; test-unseen none/241/134; average expert turns 4.9/14.4/10.1.
- Expert trajectories (App. A): WebShop and ALFWorld provide some human-annotated trajectories; for WebShop, GPT-4 also explores and trajectories with reward > 0.7 are kept; ScienceWorld golden trajectories come from its heuristic search; GPT-4 writes the missing CoT rationales. ScienceWorld tasks 9 and 10 are excluded for trajectory length.
- Table 2 (average reward; WebShop, SciWorld seen, SciWorld unseen, ALFWorld seen, ALFWorld unseen): GPT-4 63.2, 64.8, 64.4, 42.9, 38.1; Llama-2-7B-Chat untuned 17.9, 3.8, 3.1, 0.0, 0.0; SFT 63.1, 67.4, 53.0, 60.0, 67.2; Best-of-N 63.8, 70.2, 57.6, 62.1, 69.4; RFT 63.6, 71.6, 54.3, 62.9, 66.4; PPO 64.2, 59.4, 51.7, 22.1, 29.1; ETO 67.4, 73.8, 65.0, 68.6, 72.4.
- Other base models (Table 3; WebShop, SciWorld seen, unseen): Llama-2-13B SFT 66.3, 68.1, 57.6 → ETO 70.7, 71.4, 68.6; Mistral-7B SFT 60.1, 63.8, 52.2 → ETO 66.2, 68.5, 62.5.
- Success rate (Table 7): WebShop (reward = 1.0) SFT 33.0 → ETO 37.5; ScienceWorld seen 70.6 → 80.3, unseen 73.5 → 78.2.
- Pair granularity on WebShop (Table 4): trajectory-level (lr 1e-6, β 0.1) 67.4; step-level (1e-6, 0.1) 8.3; step-level (1e-7, 0.5) 62.8; mixture (1e-6, 0.1) 64.3; SFT 63.1.
- Without behavioral cloning (Table 5, WebShop): untuned Llama-2-7B-Chat 17.9; RFT 48.4; ETO 12.5; RFT then ETO 51.2. Exploration uses temperature 1.0 and pairs are built from the agent's own trajectories (§4.5).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ETO on Llama-2-7B-Chat | 7B | SFT | optimizer; batch; peak LR; warmup; schedule; epochs | AdamW; 64; 1e-5; 3% warmup; cosine; 3 epochs | arXiv:2403.02502v2 §4.1 | verified 2026-09-14 | no ablation reported |
| ETO on Llama-2-7B-Chat | 7B | SFT | loss masking | loss on agent action tokens only | §3.1 Eq. 5 | verified 2026-09-14 | no ablation reported |
| ETO on Llama-2-7B-Chat | 7B | preference | exploration samples | 1 trajectory per training instance per iteration | §4.1 | verified 2026-09-14 | no ablation reported |
| ETO on Llama-2-7B-Chat | 7B | preference | loss; batch; LR; epochs | DPO; 32; 1e-6; 3 epochs | §4.1 | verified 2026-09-14 | Table 4 varies LR only for step-level pairs |
| ETO on Llama-2-7B-Chat | 7B | preference | DPO β | 0.1 (WebShop, ScienceWorld); 0.5 (ALFWorld) | §4.1 | verified 2026-09-14 | no ablation of β for trajectory-level pairs reported |
| ETO on Llama-2-7B-Chat | 7B | preference | iterations | 2 (WebShop, ScienceWorld); 1 (ALFWorld) | §4.1 | verified 2026-09-14 | §4.3 Fig. 4: gains in iterations 1-2, decline after 3; ALFWorld gains only in iteration 1 |
| ETO on Llama-2-7B-Chat | 7B | preference | reference model | πref = current policy at start of each iteration | Algorithm 1 | verified 2026-09-14 | no ablation reported |
| ETO (all runs) | 7B, 13B | eval-gate | decoding; prompt | temperature 0.0 (Best-of-N excepted); ReAct format with 1-shot example | §4.1 Evaluation | verified 2026-09-14 | not applicable |
| ETO (all runs) | 7B, 13B | all | compute | 8 NVIDIA A100 80G | §4.1 | verified 2026-09-14 | not applicable |
| ETO on Llama-2-13B-Chat, Mistral-7B | 13B, 7B | SFT, preference | hyperparameters | not reported separately | checked §4.1, §4.2, App. A-E | not reported | — |
| ETO | all | preference | number of pairs per iteration; max steps per episode; PPO baseline settings | not reported | checked §3-4, App. A-E | not reported | — |

## Findings relevant to generality, negative feedback, and agentic training
- **Negative type (§6.1 of the course standard): negative as gradient.** The agent's lower-reward trajectory is the rejected term of DPO (Eq. 11). Negatives are labeled by the environment's final reward, not by a judge. Pairs where both trajectories succeed are discarded (§3.2).
- **Evidence for using failures.** RFT (successes added to SFT data) scores 63.6/71.6/54.3/62.9/66.4 against ETO's 67.4/73.8/65.0/68.6/72.4 (Table 2). The authors read this as "the comparison between failure and expert trajectories is essential" (§4.2, Interpretation). The comparison also changes the loss (DPO vs SFT) and the use of expert trajectories as chosen samples, so the share of gain due to the rejected term is not measured.
- **Failure modes.** Step-level pairs built from final rewards dropped WebShop reward to 8.3 at lr 1e-6, β 0.1; the authors attribute this to inaccurate step-quality estimates (§4.4). More iterations reduced performance, attributed to overfitting on a fixed training set with limited pair diversity (§4.3). Without behavioral cloning, ETO alone scored 12.5, below the untuned model's 17.9 (Table 5). Chosen and rejected log-probabilities are not reported.
- **Out-of-distribution results.** ScienceWorld unseen tests use unseen task variations (App. A: boiling water in training, boiling lead in test). ETO gains 12.0 points over SFT there (53.0 → 65.0), which the paper reports as 22% (§1) and 20% (§4.2). On ALFWorld unseen, RFT (66.4) and PPO (29.1) fall below SFT (67.2) while ETO reaches 72.4 (Table 2).
- **Base-model capability vs agent capability.** Mistral-7B scores below Llama-2-7B after both SFT and ETO (Tables 2-3). The authors conclude there is "not a strong correlation" between general LLM capability and agent capability (§4.2, Interpretation, 3 base models).
- **Stated limits.** Each agent is trained for one task; transfer and multi-task training are not tested (Limitations 2). Pairs assume the agent errs from the first step; identifying the first bad action is left to future work (Limitations 1).

## Connections
- [[dpo]] — the training-phase loss (Eq. 11); ETO applies it to whole multi-turn trajectories.
- [[fireact]], [[agenttuning]], [[lumos]] — agent methods that §5 describes as behavioral cloning on trajectories from teacher agents such as GPT-4.
- [[rft-scaling-relationship-math]] — source of the RFT baseline (Yuan et al., 2023, cited in §4.1); [[rejection-sampling-finetuning]] covers the general method; [[best-of-n]] is the N=10 baseline.
- [[ppo]] — the online RL baseline, which drops to 22.1 on ALFWorld seen (Table 2).
- [[step-dpo]] — step-level DPO for math reasoning; compare with ETO's step-level variant (Table 4).
- [[ipr-step-level-refinement]], [[agent-q]] — later agent training methods in this library that use step-level or search-based preference signals, the direction named in ETO's Limitations 1.
- [[agent-flan]] — uses negative samples as SFT content (§6.1 type 2); ETO uses them as gradient (type 4).
- [[lazy-likelihood-displacement-grpo]], [[dpo-positive]] — analyses of what the rejected-sample gradient does to chosen-sample likelihood, which ETO does not measure.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2403.02502 (v2, 2024-07-10; v1 2024-03-04).
- Audit claims not found in the source: "ETO vs SFT/RFT/PPO" column mapping in the batch notes → the notes' second and third comparison values on ScienceWorld and ALFWorld are Best-of-N and RFT, not RFT and PPO; PPO scores 59.4, 51.7, 22.1, 29.1 on those four test sets (Table 2). "About 20% relative over SFT" → the paper prints both 22% (§1) and 20% (§4.2); 65.0/53.0 = 1.226 (derived). "The canonical negative-gradient recipe for agent trajectories" → the paper makes no such claim about its standing; it describes itself as learning from exploration failures via contrastive pairs (abstract, §3.2).
- Not reported by the source: LoRA vs full fine-tuning, max sequence length, pair counts, PPO hyperparameters, seeds or variance.
