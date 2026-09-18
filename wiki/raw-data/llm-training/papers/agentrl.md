<!-- scope: AgentRL (Tsinghua University, Z.AI; arXiv 2510.04206): asynchronous multi-turn, multi-task agentic RL framework; cross-policy sampling; per-task advantage normalization; five AgentBench-FC environments; held-out BFCL-v3 test
     deps: [[grpo]]
     see-also: [[agentrl-recipe]], [[agenttuning]], [[bfcl]], [[async-rollout]], [[areal-async-rl]], [[ragen-starpo]], [[gigpo-verl-agent]]
-->

# AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework
- **Core Insight:** One Qwen2.5-14B-Instruct policy trained with RL on five AgentBench-FC environments at once averages 67.7 task success, against 67.8 for the best of five single-task policies, while each single-task policy drops on tasks it was not trained on (the DB policy scores 0.2 on ALFWorld) (Table 4).
- **Guideline:** When one policy is trained with group-relative RL on several agent environments that differ in difficulty and trajectory length, normalize token-level advantages separately per task, because removing this step lowered the 14B five-task average from 65.0 to 59.4 and the tasks were learned at different rates (Table 6; §4.2).
- **Authors:** Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, et al. (Tsinghua University; Z.AI)
- **Year:** 2025 (arXiv v1 2025-10; preprint marked "under review")
- **URL:** https://arxiv.org/abs/2510.04206 (code: https://github.com/THUDM/AgentRL)
- **Source type:** paper (with released code and configs)
- **Relevant topics:** agentic RL, multi-turn RL, multi-task RL, asynchronous rollout, exploration, advantage normalization, function calling, held-out generalization

## Abstract
RL training of LLM agents in multi-turn, multi-task settings lacks scalable infrastructure and stable algorithms. AgentRL provides a fully asynchronous generation-training pipeline, and for heterogeneous environments a unified function-call API, containerized environment deployment, and a centralized controller. On the algorithm side it adds cross-policy sampling to encourage exploration in multi-turn settings and task advantage normalization to stabilize multi-task training. Trained on open LLMs across five agentic tasks, AgentRL models outperform GPT-5, Claude-Sonnet-4, DeepSeek-R1, and open agent models in the authors' evaluation, and one multi-task model matches the best task-specific models. The framework and algorithm are used in building AutoGLM (Abstract).

## Key Contributions
- Asynchronous rollout-training design with a bounded data queue (§3.1, Figs. 3-4).
- Environment deployment framework: function-call API, one container per task environment, controller managing "thousands of parallel training episodes" (§3.1, Fig. 5; App. E.3).
- Cross-policy sampling: actions inside one trajectory come from more than one model (§3.1, Fig. 6; App. C, Eq. 20).
- Task advantage normalization over all tokens of each task's batch (§3.2, Eq. 1).
- Multi-task vs single-task comparison, held-out BFCL-v3 test, and ablations (Tables 3-6).

## Key Figures/Tables to Study
- Table 3: success rates of API models, open models, agent-trained models, and AgentRL at 3B-32B.
- Table 4: five single-task policies vs one multi-task policy (14B).
- Table 5: BFCL-v3 held-out results (32B). Table 6 and Fig. 7: ablations and training curves.
- Fig. 8: pass@k of cross-policy sampling at inference and in training (WebShop). Table 7: termination states before and after RL.

## Technical Details
- **Problem setup (App. B.1, Eqs. 11-17):** each task is an MDP; the state is a pair (environment state, token context); one action is a full token sequence, so the policy probability factorizes over tokens. Task i has samples x_{i,j}; each sample produces a group of K_{i,j} trajectories used for GRPO group advantages.
- **Asynchronous pipeline (§3.1):** rollout runs on its own resource group; the trainer pulls available data after each update and accepts a batch size that varies within a range. To limit off-policy bias, the data queue has a maximum size and all queued trajectories move to the trainer at each step. Fig. 4 (Qwen2.5-14B, WebShop) labels AgentRL throughput at 17K, 33K, 56K tokens/s on 16, 32, 64 GPUs vs 9K, 16K, 29K for the synchronous baseline (values read from figure labels).
- **Cross-policy sampling (§3.1; App. C):** at each step the action is drawn from a model chosen at random from a pool M. Because mixing architectures inside the training pipeline is hard, training uses the current model and an earlier version: some rollout engines are marked "stale" and update parameters every several steps instead of every step. In the released trainer, each generation call goes to a stale engine with probability `rollout_stale_ratio`, and stale engines receive weights every `stale_step` steps (agentrl_trainer.py `cross_sampler`, sync loop).
- **Task advantage normalization (§3.2, Eq. 1):** Ã_{i,s,g,t,k} = (Â_{i,s,g,t,k} − μ_i) / σ_i.
  - Â: token-level advantage; i task, s sample within the task, g trajectory within the group, t environment step, k token position in action a_t.
  - μ_i, σ_i: mean and standard deviation of all token-level advantages of task i in the current batch.
  - Released code groups items by `data_source`, uses only loss-masked tokens, and divides by (std + 1e-6) (agentrl_trainer.py `adv_norm`).
- **Training data (§4; App. D.2-D.3):** ALFWorld and WebShop use official training sets; OS, KG, and DB tasks were synthesized with Self-Instruct using o3 and claude4-sonnet; DB adds BIRD training samples. The five environments were converted to OpenAI function-call format (KG exposes seven tools). Smaller datasets are replicated so each task appears about as often as the largest one, and datasets are interleaved one element at a time.
- **Reward (App. E.1-E.2):** rewards normalized to [0, 1]; tasks without intrinsic reward get 1 for a correct trajectory and 0 otherwise; abnormal termination, exceeding the maximum interaction rounds, or exceeding the maximum response length gets −0.2.
- **Training setup (§4.1; App. E.2):** built on Verl with an asynchronous rewrite; GRPO baseline; temperature 0.8; 8 samples per rollout; SGLang inference and FSDP training; H800 GPUs, at least 16 GPUs for 14B; "over 1000 steps" of multi-task training. Qwen2.5-Instruct models start RL with no SFT warm-up; GLM4-9B first received a cold-start SFT phase on "a limited set" of SFT data to learn the function-call format. Full ledger: [[agentrl-recipe]].
- **Evaluation (App. E.1-E.2; Table 3 notes):** temperature 0.8, mean of four runs, same settings for API models. API models get a one-shot example on KG; trained models do not. Qwen2.5-7B-Instruct gets a one-shot demonstration on ALFWorld because it otherwise fails to produce valid tool calls.

## Recipe ledger
The paper and the released configs disclose RL settings; the table is in [[agentrl-recipe]] (paper values and config values recorded separately).

## Findings relevant to generality, negative feedback, and agentic training
- **Multi-task vs single-task (Result, single study; Table 4, 14B):** single-task averages are 42.3 (ALFWorld policy), 31.9 (DB), 38.8 (KG), 30.2 (OS), 37.8 (WebShop). ALFWorld success is 0.2 for the DB policy and 0.0 for the WebShop policy. The multi-task policy scores 91.5 / 72.2 / 72.8 / 43.6 / 58.5 (avg 67.7) vs best-of-five 89.7 / 73.9 / 72.2 / 43.1 / 60.3 (avg 67.8).
- **Held-out function calling (Result, single study; Table 5, 32B):** BFCL-v3 overall 59.9 → 61.4, multi-turn 16.2 → 19.2, live 77.4 → 79.3, non-live 86.0 → 85.8. This is one held-out benchmark at one model size.
- **In-distribution results (Table 3):** AgentRL averages 60.0 (3B), 62.0 (7B), 67.7 (14B), 70.4 (32B), 65.0 (GLM-4-9B-0414); GPT-5 52.2, Claude-Sonnet-4 Thinking 58.2, DeepSeek-R1 49.3, Qwen2.5-32B-Instruct 37.2, AgentLM-70B 51.4. API and general open models are prompted in the same function-call environments, the Hephaestus and AgentLM WebShop values are copied from their papers (Table 3 "*"), and AgentRL models are trained on these environments' task distributions.
- **Ablations (Table 6, 14B, avg 65.0):** without cross-policy sampling 60.7; without task advantage normalization 59.4. KG drops most (67.7 → 55.7 and 54.7). The paper does not explain why this baseline (65.0) differs from the Table 3 14B average (67.7).
- **Exploration and pass@k (§4.3, Fig. 8; figure only, no numbers in text):** at inference on WebShop, cross-policy sampling between Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct is slightly below the best single model at small k and above both models and above a 50/50 trajectory mix at large k. In a preliminary training run with non-identical settings, the cross-sampling model keeps an advantage as k grows.
- **Termination behavior (Table 7, Qwen2.5-14B-Instruct vs AgentRL):** "Task Limit Reached" falls from 0.68 to 0.074 (ALFWorld), 0.444 to 0.118 (OS), 0.275 to 0.020 (WebShop).
- **Negative signals:** failed trajectories receive reward 0 and incomplete ones −0.2 (App. E.1-E.2); under GRPO they enter the loss as negative advantages (negative as gradient). The paper describes no other use of failed trajectories.
- **Limitations stated by the authors (App. G.1):** cross-policy sampling can introduce distribution shifts that appear as "mild, transient instabilities"; validation covers controlled environments only.

## Connections
- [[agentrl-recipe]]: RL settings from the paper and released configs.
- [[agenttuning]]: AgentLM-7B/13B/70B baselines in Table 3; Table 1 lists AgentTuning as multi-turn and multi-task but not asynchronous and without interactive environments.
- [[bfcl]]: held-out benchmark in Table 5.
- [[grpo]]: baseline algorithm (App. E.2); [[dapo]]: summarized in App. A as background.
- [[verl-rollout]]: Verl is the base of the AgentRL trainer (App. E.2).
- [[async-rollout]], [[areal-async-rl]]: other asynchronous RL trainers; AReaL appears in Table 1.
- [[ragen-starpo]], [[gigpo-verl-agent]]: multi-turn agent RL methods listed in Table 1 as single-task.
- [[model-collapse]]: cited as motivation for adding policy diversity (§3.1).
- [[self-instruct]]: method used to synthesize OS, KG, and DB training tasks (App. D.2).
- [[rlvr-beyond-base-model]]: related pass@k-at-large-k analysis of RL.
- [[qwen-2.5]], [[glm-4]]: base model families.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.04206 (arXiv v1, PDF). Released code read on the same date at https://github.com/THUDM/AgentRL (main branch, commit not pinned): README.md, examples/training/agentrl_trainer.py, examples/training/configs/qw14b_waodk.yaml, qw3b_ws.yaml.
- Audit claims not found in the source: "trained without SFT warm-up on ... GLM-4-9B" (App. E.2 states GLM4-9B had a cold-start SFT phase; only the Qwen models skip SFT); "there is no special negative-data mechanism" (the paper describes the reward and GRPO only and does not make this statement).
- Not reported by the paper: learning rate, KL coefficient, clip ε, batch size, maximum lengths, total GPU hours, and the amount of GLM4-9B cold-start SFT data. Some of these appear in the released configs (see [[agentrl-recipe]]).
