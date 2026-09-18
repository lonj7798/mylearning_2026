<!-- scope: LOOP (arXiv:2502.01600, Apple) — interactive digital agents in AppWorld formulated as a POMDP; Leave-One-Out PPO (no critic, one LLM copy, per-token importance weights, no std normalization) trained with LoRA on Qwen2.5-32B-Instruct from 72 tasks; comparison with SFT, RFT, EI, DPO variants, PPO with critic, RLOO, GRPO; learned behaviors
     deps: [[ppo]], [[rloo]]
     see-also: [[grpo]], [[rloo-vs-grpo]], [[dr-grpo]], [[rejection-sampling-finetuning]], [[agent-q]], [[gigpo-verl-agent]]
-->

# Reinforcement Learning for Long-Horizon Interactive LLM Agents
- **Core Insight:** LOOP (PPO with a leave-one-out baseline and per-token clipping, no value network) trained Qwen2.5-32B-Instruct with LoRA on 24 AppWorld scenarios (72 tasks) and reached 71.3 task goal completion on Test-Normal, 9 percentage points above OpenAI o1 (61.9) and 81% relative above the base model (39.2) (Abstract, §5.4, Table 1).
- **Guideline:** When training a multi-turn tool-use agent with a scalar task reward and K rollouts per task, use a leave-one-out baseline without dividing by the return standard deviation and apply importance weights per token, because in AppWorld reward normalization cost 9 points and per-turn or per-trajectory weights scored 64.1 and 53.3 versus 71.3 (§5.4, Table 1); these results come from one environment with 32B LoRA training.
- **Authors:** Kevin Chen, Marco Cusumano-Towner, Brody Huval, Aleksei Petrenko, Jackson Hamburger, Vladlen Koltun, Philipp Krähenbühl (Apple)
- **Year:** 2025 (arXiv v1 2025-02; v3 2025-03; preprint)
- **URL:** https://arxiv.org/abs/2502.01600
- **Source type:** paper
- **Relevant topics:** agentic RL, multi-turn tool use, PPO variants, RLOO, GRPO, advantage normalization, importance weighting, rollout diversity

## Abstract
Interactive digital agents (IDAs) use APIs of stateful digital environments to carry out user requests. Instruction-tuned LLM agents had not been trained in their target environments, and prior methods complete less than half of AppWorld tasks. The paper trains IDAs with RL directly in the target environment, formalized as a partially observable Markov decision process (POMDP), and derives LOOP, a data- and memory-efficient PPO variant with no value network and exactly one copy of the LLM in memory. A 32B agent trained with LOOP outperforms the OpenAI o1 agent by 9 percentage points (15% relative). The authors describe it as the first reported application of RL to IDAs in a stateful, multi-domain, multi-app environment via direct API calls. The trained agent consults API documentation, avoids unwarranted assumptions, confabulates less, and recovers from setbacks (Abstract).

## Key Contributions
- POMDP formulation in which environment tokens are appended to the state but only LLM-emitted tokens a(x) enter the likelihood and gradient (§4.1, Eq. 6–8).
- LOOP: K rollouts per task, leave-one-out advantage, several PPO epochs over shuffled mini-batches (§4.2, Algorithm 1). With N_epoch = 1 and no mini-batches it reduces to RLOO; GRPO differs by normalizing with the return standard deviation (§4.2).
- A comparison of 13 fine-tuning variants (SFT, DPO, and RL families) and 4 untuned models on AppWorld (Table 1) with a three-run replication (App. E, Table 2).
- Quantified behavior changes after RL and an analysis of rollout diversity (§5.5–5.6, Fig. 3–4).

## Key Figures/Tables to Study
- Fig. 2 (POMDP and per-token / per-turn / per-trajectory importance weights), Algorithm 1.
- Table 1 (all methods, best run) and Table 2 (mean of three training runs).
- Fig. 3a (behavior metrics, dev), Fig. 4 (100 rollouts, 4 strategy modes), Fig. 7 (training curves).

## Technical Details
- **Advantage (Eq. 3):** A(c, x_k) = K/(K−1) · (R(c, x_k) − (1/K) Σ_i R(c, x_i)). R is the return of rollout k; K is the number of rollouts for the same task context c.
- **Per-token PPO objective (Eq. 5, Eq. 10):** L = E_{x∼p_ψ}[(1/|a(x)|) Σ_{t∈a(x)} min((p_θ(x_t|·)/p_ψ(x_t|·)) A, g_ε(A))], with g_ε(A) = A + ε|A|. p_θ is the policy being updated, p_ψ the sampling policy, a(x) the LLM-emitted token positions, ε the trust-region width. No KL term appears in the LOOP objective.
- **AppWorld (§1, §2, §5.1):** 9 apps, 457 API endpoints (1,470 parameters in total, §5.5), up to 17 parameters per endpoint; tasks need up to 40 interactions and 32K tokens; environment state up to 30M text tokens. 250 scenarios × 3 variants = 750 tasks; splits train 35 scenarios / 105 tasks (90 available), dev 20 / 60 (57 available), Test-N 56 / 168, Test-C 139 / 417. Test-C requires more complex interaction sequences and involves apps not seen in training. Metrics: task goal completion (TGC) and scenario goal completion (SGC, all 3 variants must pass).
- **Reward (§5.2, App. D):** fraction of the task's unit tests passed, in [0, 1]; the tests check requested state changes, absence of extraneous changes, and the final answer (§5.1).
- **Results, Test-N TGC / Test-C TGC (Table 1, best run, 5 evaluations):** Qwen2.5-32B 39.2 / 21.0; GPT-4o 48.8 / 30.2; o1 61.9 / 36.7; SFT-GT 6.2 / 0.8; RFT 47.9 / 26.4; EI 58.3 / 32.8; DPO-MCTS 57.0 / 31.8; DMPO 59.0 / 36.3; PPO with learned critic 50.8 / 26.4; RLOO 57.2 / 36.7; GRPO 58.0 / 39.5; GRPO without KL 59.0 / 42.7; LOOP bandit 53.3 / 27.7; LOOP turn 64.1 / 40.8; LOOP token 71.3 / 45.7 (SGC 53.6 / 26.6); LOOP RwNorm 61.9 / 39.8.
- **Three training runs (App. E, Table 2):** LOOP token 66.4 ± 4.8 / 41.7 ± 3.4; GRPO no KL 60.2 / 39.3; RLOO 56.3 / 31.9; LOOP bandit 48.9 ± 7.6 (unstable, clipped importance weights).
- **Relative gains (§5.4):** +81% (Test-N) and +117% (Test-C) over the base model; +15% and +24% over o1.
- **Behavior changes, base → LOOP on dev (Fig. 3a, §5.5):** turns with multiple code cells 0.080 → 0.013 (∼6x fewer); `show_api_doc` calls per rollout 3.0 → 4.7 (∼1.6x); "assum{e,ed,ing}" per rollout 0.68 → 0.024 (∼30x fewer); "dummy" 0.18 → 0.029 (∼6x fewer); give-up rate after failed API calls 0.23 → 0.076 (∼3x lower); execution errors per turn 0.23 → 0.082. Metric definitions are in App. A.1.
- **Rollout diversity (Fig. 4, §5.6):** for one task after training, 98 of 100 rollouts succeed and 94 of the 98 have distinct API-call sequences, grouped into 4 strategies.
- **Baseline details (App. C):** RFT used 1,613 successful base-model rollouts over a temperature sweep 0.05–1.0 covering 87 of 90 train tasks, epoch 30 of 100 selected on dev; a second RFT iteration did not improve significantly. EI keeps return-1.0 rollouts with the RL sampling settings. PPO's critic is a 3-layer MLP value head on the last hidden state; λ_GAE ∈ {0.95, 0.99, 0.999} diverged and λ_GAE = 1.0 was most stable.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LOOP (token) on Qwen2.5-32B-Instruct | 32B | RL | Adapter | LoRA r = 16, α = 32 on q, k, v, o and MLP | arXiv:2502.01600v3 App. D | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | Learning rate; grad-norm clip | constant 5 × 10⁻⁵; 1 | App. D | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | Tasks per iteration; rollouts per task; temperature | 40; K = 6 (240 rollouts); 1.0 | App. D, §5.2 | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | Rollout early stop | stop when ≥ 4 rollouts per task and 90% of all rollouts are collected | App. D | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | Advantage | leave-one-out, no std normalization; rollouts with abs(Â) < 0.01 dropped before the gradient | Eq. 3, §4.2, App. D | verified 2026-09-14 | Table 1: RwNorm 61.9 vs 71.3 Test-N TGC, best run |
| LOOP (token) | 32B | RL | Importance weight | per token | §5.4, Table 1 | verified 2026-09-14 | Table 1: token 71.3, turn 64.1, trajectory 53.3; Table 2 same order over 3 runs |
| LOOP (token) | 32B | RL | Clip ε; epochs per iteration; mini-batch size; KL | ε, N_epoch, mini-batch size not reported; no KL term in objective | checked body, Eq. 5, Algorithm 1, App. B–E | not reported / verified (no KL) | for GRPO the authors attribute a 2 pp drop to the KL penalty (§5.4) |
| LOOP (token) | 32B | RL | Training data | 24 scenarios × 3 = 72 tasks, difficulty 1–2 only | §5.2, §5.6, App. D.1 | verified 2026-09-14 | App. D: adding difficulty-3 tasks did not help and hurt |
| LOOP (token) | 32B | RL | Interaction and length limits | 40 interactions (train), 50 (eval); 1,500 output tokens per turn; API responses truncated above 3K tokens | §5.2, App. D | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | Log-probabilities | recomputed under the generating policy, not taken from vLLM | App. D | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | Compute | two H100 8-GPU nodes (one for 2 vLLM servers of 4 GPUs each, one for torchtune + FSDP2 training); best run 42 hours | App. D, Fig. 7 | verified 2026-09-14 | — |
| LOOP (token) | 32B | eval-gate | Checkpoint selection | best dev-set checkpoint, evaluated 5 times | Table 1 caption, §5.3 | verified 2026-09-14 | — |

## Findings relevant to generality, negative feedback, agentic training
- **Small data, held-out tasks:** all RL-like methods (EI, DPO-MCTS, DMPO, RLOO, GRPO, LOOP) beat SFT-GT and RFT despite 72 training tasks (§5.6, Table 1). SFT on ground-truth solution code lowered performance on every split except train (App. C.1); EI overfit the training data more than the RL methods (App. C.3). The authors attribute RL's generalization to rollout diversity that persists late in training (Interpretation, §5.6, Fig. 4).
- **Generality measurement:** Test-C includes apps not seen in training (§5.1); LOOP raises Test-C TGC from 21.0 to 45.7 (Table 1). No evaluation outside AppWorld is reported.
- **Negative signals:** rollouts whose return is below the mean of the other K−1 rollouts receive negative leave-one-out advantages, and their tokens' probabilities are lowered through the clipped per-token PPO term (negatives used as gradient); near-zero-advantage rollouts are dropped (Eq. 3, Eq. 5, App. D). RFT and EI discard failures; DPO-MCTS and DMPO use losing rollouts in preference pairs (§5.3, App. C). A split of the gain between positive and negative advantages is not reported.
- **Normalization:** according to the authors, dividing by the return standard deviation favors trajectories with consistent returns (low standard deviation), so the largest training signal comes from scenarios the LLM either fully solves or fails on; in AppWorld it lowered LOOP (token) by 9 points and is the likely reason GRPO scores lower (§4.2, §5.4).
- **Limitations stated (§6):** the best agent succeeds on about 7 of 10 tasks; AppWorld lacks non-determinism, transient failures, unsolvable or ambiguous tasks, adversarial scenarios, user clarification steps, and interactive counterparties.

## Connections
- [[ppo]], [[rloo]] — LOOP is PPO's clipped objective with RLOO's leave-one-out baseline; RLOO is its on-policy special case (§4.2).
- [[grpo]], [[rloo-vs-grpo]], [[dr-grpo]] — GRPO adds std normalization, which hurt here (§5.4); Dr. GRPO also removes it.
- [[rejection-sampling-finetuning]], [[rest-em]] — RFT and expert-iteration baselines (App. C.2–C.3).
- [[agent-q]] — DPO-MCTS is a simplified Agent Q without the LLM critic (App. C.4).
- [[gigpo-verl-agent]], [[ragen-starpo]], [[demystifying-agentic-rl]] — later sources in this library on multi-turn agent RL algorithms and recipes.
- [[sft-memorizes-rl-generalizes]] — compare with the SFT-GT and EI overfitting observations (App. C.1, C.3).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.01600 (arXiv v3, 2025-03-08; v1 2025-02-03).
- Audit claims not found in the source: "the earliest clean evidence that outcome-reward, critic-free multi-turn RL works in stateful tool environments" → the paper claims, to the authors' knowledge, the first reported application of RL to IDAs in a stateful, multi-domain, multi-app environment via direct API calls (Abstract, §1); the reward is the fraction of unit tests passed, not a binary outcome (App. D). "B lists it as missing" is an internal audit note, not a source claim.
- Not reported by the source: PPO clip ε, epochs per iteration, mini-batch size, total gradient steps, GRPO KL coefficient.
