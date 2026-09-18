<!-- chapter: ch-54
     track: infra
     kind: content
     title: Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments
     deps: [ch-53, ch-43a, ch-45b]
     sources: [[best-of-n]], [[on-off-policy-rlhf]], [[async-rlhf-noukhovitch]], [[areal-async-rl]],
              [[async-rollout]], [[verl-rollout]], [[openrlhf-ppo]], [[verl-ppo-loss]],
              [[rollout-training-mismatch-tis]], [[glm-5]], [[deepseek-v3.1]], [[minimax-forge-agent-rl]],
              [[replay-buffer-rlhf]], [[kimi-k1-5]], [[deepswe]], [[r2e-gym]], [[qwen3-coder]],
              [[skyrl-agent]], [[asearcher]], [[agent-lightning]], [[prime-intellect-environments-hub]],
              [[topr-tapered-off-policy-reinforce]], [[asymmetric-reinforce]], [[rls-razor]],
              [[retaining-by-doing]], [[rlvr-beyond-base-model]]
     figures: figures/async-pipeline.html
     revised: 2026-09 (generality revision)
-->

# Chapter 54 — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments

> **Core insight.** In an RL run the policy-gradient formula is a small part of the engineering. Generation dominates wall-clock time — verl's colocated 128-GPU DAPO baseline spends 177.85 s of a 356.30 s step in generation, and moving the same run to a disaggregated 64:64 rollout/train layout cut 400 steps from 1 d 16 h 48 m to 17 h 22 m ([[verl-rollout]], `fully_async_policy/README.md` L333–361). Making generation asynchronous means training on trajectories produced by earlier weight versions, and that staleness is measurable and controllable: in [[areal-async-rl]]'s 1.5B math ablation, naive PPO falls from 42.0 AIME24 at staleness η = 0 to 23.3 at η = 4, while the same run with a decoupled PPO objective holds 42.2 (§7.4, Table 2). A second, independent off-policyness comes from the sampler and the trainer assigning different probabilities to the same tokens under the same weights ([[rollout-training-mismatch-tis]]).
>
> **Guideline.** When generation and training share GPUs and a step is dominated by waiting for the longest response, move to a decoupled layout with an explicit maximum-staleness parameter, because [[areal-async-rl]] reports 2.77× lower end-to-end time at matched or higher final accuracy (§7.2, Table 1). When staleness is above one training step, change the objective before raising the cap: the decoupled PPO objective kept the AIME24 score within 1 point of the synchronous oracle up to η = 8, and the same run without it lost up to 18.7 points (§7.4, Table 2). When rollouts and gradients come from different backends, log the sampler log-probabilities and apply a truncated ratio, because [[rollout-training-mismatch-tis]] measured token-probability differences up to 1.0 on DAPO with Qwen2.5-32B. When trajectories run inside sandboxes, record the failure reason of every sample and exclude environment crashes from the reward, because [[glm-5]] reports that they otherwise enter training as model failures (§4.1.2). Otherwise — a short synchronous run on one node, with generation under 30% of step time — keep the synchronous loop and spend the effort on evaluation instead.

---

## Why this chapter matters for a general-purpose model

The pipeline so far is pre-training → mid-training → SFT → preference optimization → RL → evaluation. This chapter sits inside the RL stage, but it is not about the objective — the policy-gradient, preference, and negative-gradient chapters earlier in the track cover that. It is about where the data that enters the objective comes from, how old it is, which samples are kept, and what physical system produces it.

That matters for breadth for three reasons.

1. **The data source changes how much prior ability survives.** [[rls-razor]] reports an empirical forgetting law: the amount of catastrophic forgetting after fine-tuning is predicted by `E_{x∼τ}[KL(π₀‖π)]`, the KL between the fine-tuned and base policy measured on the **new** task, with a quadratic fit of R² = 0.96 in a controlled MNIST setting and R² = 0.71 in the LLM experiments (§4, Fig. 3, Fig. 11). [[retaining-by-doing]] reports the same ordering across Llama and Qwen models up to 8B on instruction following, general knowledge, and arithmetic: RL forgets less than SFT at comparable target-task performance, and the authors attribute the difference to on-policy data (Abstract, Fig. 2). Rollout infrastructure decides how on-policy your data actually is.
2. **Every throughput mechanism in this chapter is also a filter.** Staleness caps, prompt replay, dynamic filtering, and scheduler windows all decide which prompts and which trajectories reach the gradient. A filter chosen for throughput is a curriculum whether or not it was designed as one, and a curriculum can narrow the training distribution.
3. **Agentic RL puts an execution environment inside the loop.** Once a sandbox can crash, the reward signal contains a term that has nothing to do with the model. Separating environment failures from model failures is a data-quality problem that only exists at this stage.

Read [figures/async-pipeline.html](figures/async-pipeline.html) alongside §4: it lets you set the staleness cap and the rollout/trainer speed ratio and read off the resulting queue depth, generator lead, and the measured [[areal-async-rl]] accuracy at that cap.

---

## §1 The rollout loop and where its time goes

**Definition.** A *rollout* is one generation pass: prompts go to an inference engine, the engine returns token ids and per-token log-probabilities, a reward function or verifier scores the result, and the trainer computes advantages and takes a gradient step. In the *synchronous* arrangement one process alternates the two phases on the same GPUs. In the *decoupled* or *asynchronous* arrangement a generator and a trainer run concurrently on separate resources, connected by a queue.

**The measurable problem.** In the synchronous arrangement the trainer idles during generation, and generation cannot finish until the longest response in the batch finishes. Two independent sources report the size of the effect:

- [[verl-rollout]]: in the colocated 128-H20 baseline (Qwen2.5-Math-7B, DAPO, 28K max response length), a step takes 356.30 s of which `gen` is 177.85 s — 49.9% (`fully_async_policy/README.md`@753aed3 L360). Reconfigured as 64 rollout GPUs and 64 training GPUs with `partial_rollout=True` and `staleness_threshold=0.5`, 400 steps took 17 h 22 m instead of 1 d 16 h 48 m, a factor of 2.35, with AIME-2024 acc/mean@1 of 0.3094 versus 0.2958 at the last checkpoint (L333–361, L369). **Result (single study).**
- [[areal-async-rl]]: on 16 nodes, 250 PPO steps with a 1.5B R1-Distill-Qwen model, AReaL took 14.8 training hours against 41.0 for its own synchronous variant (42.2 vs 42.0 AIME24), and at 7B 25.4 hours against 57.7 (63.1 vs 63.0) — up to 2.77× end to end (§7.2, Table 1). **Result (single study).** Its ablation of interruptible generation alone gives 12% (1.5B) and 17% (7B) higher generation throughput on 4 nodes (§7.5, Fig. 6b).

**Implication for a general-purpose model.** Throughput is not only a cost question. [[asearcher]] reports that the turn limits of 10 or fewer in earlier online-RL search agents were a consequence of long trajectories blocking a synchronous batch, not of the algorithm; decoupling execution from updates allowed a limit of 128 turns per trajectory, and during training the agent reached over 100 tool calls and over 400k generated tokens in one trajectory (§1, Fig. 1). Behaviors that only appear over long horizons cannot be trained by a system that cannot afford long horizons.

---

## §2 Best-of-N: the training-free baseline and its ceiling

**Definition.** Best-of-N (BoN) samples N responses from a fixed policy, scores each with a reward model, and returns the highest-scoring one. It trains nothing.

**Why it belongs in an infrastructure chapter.** It measures how much of the target reward is already reachable by resampling the current policy, and it requires no gradient step, no trainer, and no rollout-training loop, so it can be run before any RL infrastructure exists.

**Formula.** [[best-of-n]] states the analytic divergence of the BoN policy from the sampling policy (Table 10 caption; App. G.3 fn. 15):

```
KL(BoN ‖ base) = log N − (N − 1)/N
```

`N` is the number of samples per prompt; the expression assumes no ties and depends on nothing else — not on the reward model's quality.

**Worked example.** N = 8: `ln 8 = 2.0794`, `7/8 = 0.875`, so KL = 1.204 nats. N = 64: `ln 64 = 4.1589`, `63/64 = 0.9844`, KL = 3.174. N = 512: `6.2383 − 0.998 = 5.240`. The paper's Table 10 prints 1.2, 3.2, and 5.2 for the same N, so the arithmetic checks against the source.

**Evidence and its limits.** In the same appendix the authors write that "at a given average reward, the best-of-N and PPO policies have similar quality as judged by human labelers (not shown)", and that the PPO policies sit farther from the supervised policy in KL: `KL(ppo, sup)` is 18.0 at 1.3B and 14.0 at 6.7B, against 3.2 for BoN-64 (App. G.3; Tables 9–10). **Result (single study), with the per-N human numbers not printed.** The setting is Reddit TL;DR summarization in 2020 with 1.3B and 6.7B models, so the comparison does not transfer to reasoning or agentic tasks without re-measurement.

**The ceiling.** [[best-of-n]] §4.3 (Fig. 5) reports that as optimization against the reward model increases, human preference first rises and then falls until the reward model is anti-correlated with labelers. BoN inherits that ceiling, because argmax over N samples is itself an optimization against the reward model. Any gain from raising N is a gain against the same proxy that RL would over-optimize; see ch-42 for the reward-hacking treatment.

---

## §3 On-policy versus off-policy data: what has actually been measured

**A claim this chapter does not make.** A fixed split of the online-versus-offline gap — for example about 80% attributed to distribution shift — is sometimes credited to Tang et al. (2024). No such split appears in that paper. Its abstract states the opposite: offline data coverage and data quality, each by itself, do not explain the gap ([[on-off-policy-rlhf]], Abstract; §5.1–§5.2).

**What [[on-off-policy-rlhf]] reports.** Tang et al. (2024) hold the loss (IPO), the hyper-parameters, and the source preference data fixed and vary only where the responses come from. On four datasets — OpenAI summarization, Anthropic helpfulness, Chat arena side-by-side, Anthropic harmlessness — with T5X policies at 770M, 3B, and 11B:

- On-policy sampling reaches a higher peak win rate against a fixed golden policy on all four (§2.1, Fig. 1).
- Data coverage does not explain it: training offline on the shuffled stream of the online run's own samples gives little improvement over plain offline, except on Chat arena sxs (§5.1, Fig. 4).
- Data quality does not explain it either: offline training on pairs sampled from the 4k-step online policy and relabeled by the golden model improves only slightly over SFT (§5.2, Fig. 5).
- Offline training makes the policy a **better pairwise classifier and a worse generator**: offline policies peak near 70% classification accuracy on the preference set while online policies stay below 50%, and within offline runs classification accuracy correlates little with win rate (§5.3, Figs. 7–8).
- Log-probabilities of the preferred responses fall below their SFT values during training, most strongly offline (§5.3.3, Fig. 8 bottom).

**Status.** **Result (single study)** for each item; the paper runs no PPO and no DPO experiments, so it is not evidence about PPO versus DPO.

**What [[async-rlhf-noukhovitch]] adds.** Noukhovitch et al. (2024) measure staleness directly. They generate `N` minibatches, then perform `N` minibatch updates, so update 1 is on-policy and update `N` is `N−1` steps stale. On TL;DR with a Pythia 410M policy and a gold reward model: PPO's win rate is highest at N = 1 and falls as N grows, with a logarithmic dropoff such that N = 1 and N = 2 behave almost identically (§3.2, Fig. 3). All values of N lie on the same win-rate-versus-KL Pareto curve, so staleness slowed progress along the frontier rather than moving the frontier. Comparing losses at N ∈ {1, 2, 4, 8, 16}, PPO is best on-policy but degrades quickly, RLOO likewise, and Online DPO is the most robust — the only method still learning at N = 64 (§3.3, Fig. 4).

**Conditions.** 410M policy, one task, one reward model. The 8B chatbot run in the same paper is a throughput result, not a staleness sweep: synchronous and asynchronous Online DPO both reach 57.20% GPT-4o win rate on No Robots, with compute time 230 versus 142 (§4, Table 1).

**Implication for a general-purpose model.** The quantity that degrades with staleness is generation quality, and the quantity that improves is a discrimination-like score. If the only metric you log is reward, a stale run can look fine while its generations get worse.

---

## §4 Asynchrony: staleness as a quantity you set

### 4.1 The staleness cap

[[areal-async-rl]] makes staleness an explicit hyper-parameter η: the maximum number of training steps by which a trajectory's generating policy may lag the current one. With current policy version `i`, cumulative generated trajectories `N_r`, and training batch size `B` prompts, new generation requests are admitted only while (§5.1, Eq. 3):

```
⌊(N_r − 1) / B⌋ ≤ i + η
```

`η = 0` reduces the system to synchronous RL. Older trajectories in the buffer are used first when a batch is assembled.

**Worked example.** With `B = 512` and `η = 8`, suppose the trainer has completed `i = 100` updates. The constraint allows `⌊(N_r − 1)/512⌋ ≤ 108`, so `N_r ≤ 55,808 = 109 × 512`. The generator may therefore be at most `η + 1 = 9` batches ahead of the trainer, and requests beyond that are rejected until the trainer advances.

### 4.2 The objective has to change with the cap

Raising η alone degrades the run. The paper's own reason for the fix is that a small η stalls generation whenever a few very long trajectories are in flight, so throughput pushes toward large η (§5.1). Its answer is a decoupled PPO objective that separates the *behavior* policy `π_behav` (which sampled the tokens) from a *proximal* policy `π_prox` (the trust-region center), §5.2 Eqs. 4–5:

```
J(θ) = E_{q∼D, a_t∼π_behav} Σ_t (π_prox/π_behav) · min( u_t(θ)·Â_t , clip(u_t(θ), 1−ε, 1+ε)·Â_t )
u_t(θ) = π_θ(a_t | s_t) / π_prox(a_t | s_t)
```

`Â_t` is the advantage at token t, `ε` the clip range, and `π_prox` is set to the parameters immediately before the current update, with token probabilities recomputed when the global batch arrives. The paper's argument is that using `π_behav` as the trust-region center pulls the latest policy toward old, lower-quality policies (§5.2). Proposition 1 shows that a sequence whose tokens came from several versions is equivalent to sampling from a single behavior policy, which is what makes interrupted generation valid (§5.2, proof in §D).

**The measurement.** 1.5B model, math, 250 steps, without / with the decoupled objective (§7.4, Table 2):

| Max staleness η | AIME24 w/o | AIME24 with | MATH-500 w/o | MATH-500 with |
|---|---|---|---|---|
| 0 (oracle) | 42.0 | 42.0 | 89.2 | 89.2 |
| 1 | 41.8 | 42.1 | 89.9 | 89.8 |
| 2 | 40.0 | 41.8 | 89.6 | 89.6 |
| 4 | 23.3 | 42.2 | 66.9 | 89.5 |
| 8 | 35.7 | 41.0 | 87.8 | 89.2 |
| 16 | 35.8 | 38.7 | 87.4 | 89.1 |
| ∞ | 34.0 | 36.9 | 87.1 | 88.1 |

Effective training throughput over the same sweep rises from 128.7k tokens/s at η = 0 to 396.8k at η = ∞ (§7.4, Fig. 5c). Production defaults in the same paper are η = 4 for coding and η = 8 for math (§7.1). The η = 4 row without the decoupled objective is non-monotone against η = 8, which the paper does not explain; treat individual rows as one seed each (§B: a fixed seed of 1 across all experiments).

### 4.3 How frameworks implement the cap

- **[[async-rollout]] (OpenRLHF).** `GenerateSamplesActor` and `TrainingActor` run concurrently, joined by `rollout_queue = Queue(maxsize=async_queue_size)` plus a `rollout_slots` queue pre-filled with the same number of tokens, acting as a counting semaphore (ppo_trainer_async.py@64c1cc4 L287–297). The default queue size is 1 (`train_ppo_ray.py` L274–283). The enqueued payload carries **no policy-version field, and neither loop filters samples by age** (L149–151, L227–232). The README states that larger queues are "more off-policy" and that async training "may affect convergence", and the repository reports no throughput or convergence measurement for the mode (README L665–679; verified across arXiv:2405.11143 v1–v6).
- **[[verl-rollout]] (`fully_async_policy`).** A rollouter and a trainer on separate GPUs, with `staleness_threshold` (0.5 in the released experiment, ablated over 0 / 0.1 / 0.3 / 0.5 at README L380–396) and `partial_rollout=True`, which interrupts generation before a parameter sync and resumes the interrupted samples afterwards (L55–58, L222–226, L351–352).
- **[[glm-5]].** Each response records the sequence of model versions `(w₀, …, w_k)` that produced its segments. With current version `w′`, the sample is discarded if `w′ − w₀ > τ` (§4.1.2). This is a per-trajectory filter rather than an admission-control rate limit, which is what multi-turn agent rollouts spanning several weight updates require. The optimizer is reset after each weight push to the inference engine (§4.1.1).
- **[[minimax-forge-agent-rl]].** Forge frames the scheduler itself as the staleness control. Strict FIFO suffers head-of-line blocking from a single slow trajectory; greedy fetch of whatever finishes first produces a training stream "initially dominated by short, easy tasks and later by clustered hard ones" (§1.2). *Windowed FIFO* lets the trainer fetch any completed trajectory inside a sliding window `[T_i, T_{i+W−1}]` (for example `W = 0.3N`) and forbids fetching completed tasks beyond it; the window advances only as head tasks are consumed (§3.1). The mechanism is a distribution control, not only a latency control.

[figures/async-pipeline.html](figures/async-pipeline.html) shows the four mechanisms side by side and reports the measured AReaL accuracy for each staleness setting.

---

## §5 Sampler–trainer mismatch: a second source of off-policyness

Staleness is off-policyness introduced by a configured cap. A second source of off-policyness is present even at η = 0, and it is not configured: the generation backend and the training backend do not assign the same probabilities to the same tokens.

**The measurement.** [[rollout-training-mismatch-tis]] reports that with identical parameters, vLLM as sampler and FSDP as learner assign different probabilities to the same tokens: the maximum per-token difference reaches 1.0 on DAPO with Qwen2.5-32B, meaning tokens where `π_vllm = 1` and `π_fsdp = 0`, and about 0.4 on the smaller GSM8K example (Fig. 1; "Normal RL training"). The gap persists after two vLLM patches (returning the probabilities actually used for sampling, and casting `lm_head` to fp32).

**The correction.** Truncated importance sampling multiplies the policy-gradient term by a capped learner/sampler ratio (formula paragraphs):

```
E_{a∼π_sampler(θ)}[ min(π_learner(a, θ)/π_sampler(a, θ), C) · R(a) · ∇θ log π_learner(a, θ) ]
```

`C` is the truncation cap. **Worked example from the source's variance argument:** for a token with ratio 16, un-truncated importance sampling amplifies gradient noise by 16² = 256×, TIS with `C = 2` by 4×, and TIS with `C = 8` by 64×. The released script uses `C = 8` for the 32B DAPO run and `C = 2` for the 0.5B GSM8K run (recipe ledger, `dapo_qwen32b_bf16.sh` L5, L106; `gsm8k_qwen0_5b_int8.sh` L2, L32).

**Reported effects.** With INT8 rollouts on DAPO-32B, entropy falls below 0.2 and keeps decreasing, responses become abnormally long, and the k1 KL estimator frequently logs negative values; adding TIS reverses all three (Figs. 5–6). Logged reward is `E_{π_sampler}[R]`, so the BF16 run without TIS logs a *higher* reward than the run with TIS while scoring lower on AIME (Fig. 6) — a direct example of the training-reward metric disagreeing with the held-out metric. When the gap is small (DeepSeek-R1-Distill-Qwen-1.5B with BF16 rollouts on both sides), TIS gives no gain (Fig. 3). Factor analysis: differing parallelism between sampler and learner raises the maximum mismatch (learner Ulysses SP8 against a TP2 sampler gives double-digit counts of responses above 0.5, Fig. 7), long uninterrupted responses raise it (20K versus 4K caps, Fig. 9), and no sampler backend is consistently better (Fig. 11).

**Implementations.**

- verl: `imp_ratio = exp(old_log_prob − rollout_log_probs)` clamped at `max=imp_ratio_cap` with no lower bound, multiplied into the per-token loss (`core_algos.py` L586–590 at the blog's pinned commit). In current verl the feature is `rollout_is_weights`, token- or sequence-level, and the YAML default is `rollout_is: null`, that is off ([[verl-ppo-loss]], `config/algorithm/rollout_correction.yaml`; the token and sequence options are defined at `config/algorithm.py` L85–89).
- OpenRLHF: `--algo.advantage.is_correction_enable` with type `tis` (clamp to `[low, high]`), `icepop` (zero out-of-band tokens), or `seq-mask-tis`; default threshold `[0.5, 5.0]`, off by default; the diagnostic `vllm_kl = masked_mean(rollout_log_probs − old_log_probs)` ([[async-rollout]], `loss.py` L150–173).
- [[glm-5]]: in the asynchronous setting the rollout engine may update several times inside one trajectory, so tracking the exact `π_θ_old` would require keeping a checkpoint history. GLM-5 uses the rollout log-probability directly as the behavior proxy, `r_t(θ) = exp(log π_θ(a_t|s_t) − log π_rollout(a_t|s_t))`, and masks tokens whose ratio leaves `[1 − ε_ℓ, 1 + ε_h]` out of the gradient entirely (§4.1.2, Eqs. 3–5).

**MoE makes it worse, and the fix is routing replay.** [[deepseek-v3.1]] (the DeepSeek-V3.2 report) lists *Keep Routing* — reusing the expert routes chosen at inference time during the training pass — and *Keep Sampling Mask* — applying the sampling top-p/top-k truncation mask to `π_θ` — among four GRPO stabilizers, and states that Keep Routing is "crucial for RL training stability of MoE models", in use since DeepSeek-V3-0324 (§3.1). verl exposes the corresponding plumbing: with `enable_rollout_routing_replay` the vLLM server returns routed experts alongside token ids ([[verl-rollout]], `vllm_async_server.py` L716–718).

**Token-in-token-out.** The last mismatch is textual. [[verl-rollout]]'s agent-loop documentation reports that re-applying the chat template to the final message history produces token ids that differ from the concatenation of per-turn prompt and response ids, because tool parsers rewrite assistant content and decode-encode is not always invertible, and that doing so "make[s] PPO training not even converged in single-turn" (`docs/advance/agent_loop.rst` L155–182). [[glm-5]] gives the same design a name, TITO, and a component: a gateway that intercepts generation requests and records token ids and metadata, so re-tokenization never happens (§4.1.2). When rollouts are multi-turn or use tool calls, build the training sequence from the token ids the engine returned rather than from the rendered text, because verl reports non-convergence for the text path even in single-turn training.

---

## §6 Reuse and selection: replay, prioritization, and filtering as curricula

**Two different things are called replay.**

1. *Trajectory replay* reuses stored completions in a later gradient step. The importance ratio between the current policy and the policy that produced those tokens must then be corrected, which is the same machinery as §4 and §5, with a larger gap.
2. *Prompt replay* re-selects a prompt and generates fresh completions under the current policy. No importance correction is needed because no old tokens enter the gradient, but the prompt distribution is no longer the dataset's.

**A caution about the library card.** [[replay-buffer-rlhf]] is a framework synthesis rather than an extract of one primary artifact, and the 2026-09 audit could not confirm two of its specific claims at their stated loci — its description of TRL's experimental replay trainer as prompt-only, and a DeepSeek-R1 "§3.2" negative result on trajectory replay (R1 §3.2 is the distilled-model evaluation; its unsuccessful attempts, §4.2, are PRM and MCTS). Use the card as a pointer to the design space, not as evidence for either claim.

**The attested reuse mechanism is partial rollout.** [[kimi-k1-5]] gives each rollout a fixed output-token budget; an unfinished trajectory is written to a replay buffer and continued in the next iteration, so only the current segment needs on-policy computation, and segments can be excluded from the loss (§2.6.2). Detected repetitions are terminated early and can be penalized. [[verl-rollout]]'s `partial_rollout=True` is the same idea at the systems level: interrupt before a parameter sync, resume afterwards (`fully_async_policy/README.md` L55–58). The mechanism converts a long-tail latency problem into a controlled multi-version trajectory, which [[areal-async-rl]]'s Proposition 1 then justifies treating as a single behavior policy.

**Dynamic filtering.** The DAPO recipe as released in [[rollout-training-mismatch-tis]]'s scripts filters groups by accuracy and regenerates up to 10 generation batches to fill a training batch (512 prompts per step, generation batch 1,536 prompts, 16 samples per prompt; `dapo_qwen32b_bf16.sh` L28–35). Groups where every sample succeeds or every sample fails give a zero group-relative advantage, so they are discarded. [[deepswe]]'s *compact filtering* masks the loss of trajectories that ended by max context, max environment steps, or a 20-minute generation timeout, with the stated reason that an agent can pass all tests by accident and rewarding such trajectories led to collapse in a Qwen3-14B ablation (§2.3, Fig. 6; the effect is shown as a curve, without a number).

**Why all of this is a curriculum.** Accuracy filters keep prompts of intermediate difficulty. Prioritized prompt replay weighted by group-reward variance, the design [[replay-buffer-rlhf]] describes for TRL's experimental GRPO replay trainer, keeps the same prompts for the same reason; the 2026-09 audit could not confirm the card's description of that trainer at its stated locus, so treat the mechanism as a design pattern rather than an attested implementation. Windowed FIFO ([[minimax-forge-agent-rl]] §3.1) keeps whichever tasks the window admits. Each of these narrows the effective training distribution relative to the dataset you assembled, and none of the sources reports what happens to the domains they drop. Two reports state the counter-practice directly: [[deepseek-v3.1]] merges reasoning, agent, and human-alignment training into **one** RL stage rather than sequential per-domain stages (§3), and [[minimax-forge-agent-rl]] mixes reasoning, general QA, and agent tasks simultaneously, stating that sequential training produces negative transfer between domains (§4.1). When a filter or a scheduler window selects which prompts reach the gradient, hold out a per-domain regression set that the filter cannot reach, because no source cited here measures the effect of its own filter on the domains it removes.

---

## §7 Environments in the loop: agentic rollout infrastructure

When the rollout is a multi-turn agent, the generator is an inference server together with a pool of execution environments. The reported pool sizes below range from 512 concurrent Docker containers to 20,000 parallel environments, so environment capacity becomes a system requirement alongside GPU capacity.

**Scale, reported.**

| System | Environment scale | Locus |
|---|---|---|
| [[deepswe]] (Qwen3-32B) | 512 Docker containers per RL iteration (batch 64 × 8 passes); 64 H100 for six days | blog §2.2, intro |
| [[r2e-gym]] | 8,135 executable problems built from commits; 4,578-problem subset with no SWE-Bench repository overlap | §2, Table 1 |
| [[qwen3-coder]] (480B-A35B) | 20,000 independent environments in parallel | blog, "Scaling Long-Horizon RL" |
| [[glm-5]] | over 10K real-world software-engineering environments, plus terminal and multi-hop search domains; over 1k concurrent rollouts | §3.3, §4.1.1 |
| [[asearcher]] (QwQ-32B) | 128 turns per trajectory; over 100 tool calls and 400k generated tokens observed during training | §1, Fig. 1 |

**Orchestration.** [[deepswe]] reports that running thousands of containers crashed the Docker daemon, so scheduling moved to Kubernetes; each worker node has about 200 CPU cores and over 6 TB of local NVMe with preloaded SWE-bench images, the cluster scales past 1000 CPU cores, and the autoscaler removes nodes underutilized for roughly twenty minutes (§2.2). Image caching and node-local storage are what make environment reset cheap enough to run at that rate.

**Long-tail latency.** [[minimax-forge-agent-rl]] states the range plainly: agent rollout completion times run "from seconds to hours" (§1.2). Three published answers:

1. Interrupt and resume — partial rollout ([[kimi-k1-5]] §2.6.2, [[verl-rollout]] `fully_async_policy`), plus AReaL's interruptible rollout workers (§5.2, §7.5).
2. Keep the decode path uninterrupted — [[glm-5]] runs prefill and decode on dedicated resources so a burst of prefills cannot stall long-horizon decodes, which the report names as its fix for tail behavior in multi-turn agentic RL (§3.6.2).
3. Overlap the CPU-bound stages — [[skyrl-agent]] splits each rollout into runtime initialization, agent run, and reward calculation with bounded queues per stage; its *Async Pipeline* strategy overlaps those with GPU generation and gives a 1.55× speedup over naive asynchronous batching (§3.2, Fig. 1b, Fig. 3).

**Failures that are not the model's.** [[glm-5]] records the failure reason of every sample and excludes samples that failed by environment collapse, because such failures "reflect environment instability rather than the model's capability". For group-based methods this leaves an incomplete group, which is padded by repeating valid samples when more than half the group is valid, and dropped otherwise (§4.1.2). [[deepswe]]'s compact filtering does the analogous thing for exhausted budgets. Treat "sandbox died", "tool timed out", "context exhausted", and "model produced a wrong patch" as four distinct labels, because only the last is a training signal.

**Interfaces.** Three designs recur:

- **Gym-style wrapping.** [[skyrl-agent]]'s agent loop acts only through OpenAI-style function calls; a Gym environment is added by wrapping `step()` as a tool, and each dataset binds its own tools and verifiers so one job can mix task types (§3.1, Listing 1).
- **Transition records rather than one masked sequence.** [[skyrl-agent]] records every LLM call with input tokens, output tokens, and log-probabilities, then converts to a backend-agnostic format; transitions sharing a prefix are packed with masks, which reduces to concatenate-and-mask when the context is never modified (§3.3). [[agent-lightning]] takes the same position: one action is one LLM call, each call in an episode receives the episode's final return, and a single-turn RL method is then applied to the grouped transitions (§3.3.2). Its stated reasons against concatenate-and-mask are that masks couple training to agent logic, break the token-continuity assumption behind RoPE, and produce long sequences (§3.3.2, §5.1) — reasons, not measurements: the paper reports training curves and no numeric tables, and no comparison against masked multi-turn training.
- **Agent-side decoupling.** [[agent-lightning]]'s Training-Agent Disaggregation puts a server with an OpenAI-like API next to the RL framework and a client inside the agent (§3.4.1). [[minimax-forge-agent-rl]] pushes the same interface further: agents route requests to a gateway and the framework is agnostic to their internals, so context compression, history rewriting, and multi-agent loops need no training-loop change (§2.3).

**Where environments come from.** [[r2e-gym]]'s SWE-Gen builds executable environments from commits by collecting or generating tests and back-translating code changes into problem statements, so human-written issues are not required (§2). [[prime-intellect-environments-hub]] treats environments as shareable artifacts with their own evaluation reports, natively consumed by a trainer, with sandboxes that plug into the same verifier interface. That post is an announcement with no measurements, so it supports the interface claim and nothing quantitative.

**Efficiency result worth carrying.** [[skyrl-agent]] trained SA-SWE-32B from Qwen3-32B with RL only on the same 4.5K R2E-Gym instances DeepSWE used, reaching 39.4% Pass@1 on SWE-Bench Verified in 4,601 H100 hours, against DeepSWE's 36.4% under the same simple ReAct evaluation and 9,180 H100 hours (§4.3, Table 2). The paper attributes the difference to tool design plus the dispatcher, not to the objective — and, notably for breadth, reports that the SWE-only trained model also improved on Terminal-Bench, BrowseComp-Plus, and WebArena (Abstract, Table 3).

---

## Negative samples and negative feedback

**Which sense of "negative".** This section is about **negative as gradient** (sense 4 of the ch-43a taxonomy): tokens or sequences with `Â < 0`, whose likelihood the update decreases. Environment crashes are not negatives at all; they are corrupted labels, handled in §7.

**Where negative advantages come from here.** A group-relative baseline makes roughly half of a mixed group negative by construction. [[deepswe]]'s leave-one-out advantage with n = 8 rollouts and 2 successes gives a failure `0 − 2/7 ≈ −0.29` and a success `1 − 1/7 ≈ 0.86`; when all 8 fail every advantage is 0 (derived in the card from the stated components; the post prints no formula).

**Why stale negatives are the unstable ones.** In the PPO clipped surrogate, the two advantage signs are clipped asymmetrically. From [[openrlhf-ppo]] (`loss.py` L136–141, derived): for `Â < 0` the objective is `Â · max(r, 1 − ε_low)`, so the gradient is zero when `r < 1 − ε_low` but **the loss grows linearly without bound as r increases**. For `Â > 0` the objective is `Â · min(r, 1 + ε_high)`, bounded above. A stale or mismatched negative sample is exactly the case where `r` can be large, so the unbounded side is the side staleness hits.

**Control 1 — dual clip.** Cap the per-token loss for `Â < 0` at `c·|Â|`, which zeroes the gradient once `r > c` ([[verl-ppo-loss]] `core_algos.py` L1323–1361, citing arXiv:1912.09729). **Worked example** with `ε_low = ε_high = 0.2`, `Â = −1`: at `r = 0.7` the loss is `max(0.7, 0.8) = 0.8` and is flat in `r`; at `r = 1.5` the loss is 1.5 with gradient flowing; at `r = 5.0` with `c = 3` the loss is capped at 3 and the gradient is zero. verl applies `clip_ratio_c = 3.0` by default; OpenRLHF's `dual_clip` defaults to `None`, so the bound is off unless set. Neither project reports an ablation of `c`.

**Control 2 — asymmetric importance sampling.** [[topr-tapered-off-policy-reinforce]] truncates the ratio for negatives to `[0, 1]` and leaves positives at weight 1:

```
α = clip(π(y|x)/µ(y|x), 0, 1)  if R(x, y) < 0  else  1
ℓ = stop-grad(α) · r(x, y) · log π(y|x)
```

`µ` is the behavior policy. The two terms are, in the authors' words, "SFT update for positive examples" plus "TIS update for negative examples" (§3, Eq. 8; Fig. 1 right). Their argument for the lower limit of exactly 0 is that any positive floor under a negative's weight must eventually produce degeneracy, as with naive REINFORCE (§2.1, §3). On GSM8K with Llama 3 8B, as training becomes more off-policy, naive REINFORCE degrades and PPO stops improving while TOPR keeps improving (Fig. 1 left; the figure prints curves, not a table).

**Control 3 — move the baseline instead of the ratio.** [[asymmetric-reinforce]] uses `A = r − V` with a tunable baseline and no importance correction. Theorem 4.2 gives a phase transition: while `V < V^µ` (the behavior policy's expected reward) the limit policy improves on µ and keeps wide support; at `V ≥ V^µ` the support "suddenly shrinks" to generically one element. Empirically, with Llama-3.1-8B-Instruct on MATH, ±1 rewards, G = 8 samples per prompt and the behavior policy refreshed every 250 steps, all 7 runs at `δV = 0` collapsed in train and test accuracy, and the authors report greater stability for the 7 runs at `δV = −0.1` (§5.2, Fig. 5 left). The recommendation is to set the baseline slightly **below** the per-prompt mean reward when data is stale — that is, to weight positives more than negatives.

**Control 4 — drop the sample.** [[deepseek-v3.1]] masks a sequence entirely when it is both negative and far off-policy: `M_{i,t} = 0` when `Â_{i,t} < 0` **and** `(1/|o_i|) Σ_t log(π_old/π_θ) > δ`, where `π_old` is the probability returned by the inference engine (§3.1, Eq. 9). [[glm-5]] masks any token whose rollout ratio leaves `[1 − ε_ℓ, 1 + ε_h]`, both tails, rather than clipping it (§4.1.2, Eq. 5). [[rollout-training-mismatch-tis]] notes that AReaL's decoupled PPO drops a sample entirely when the ratio exceeds a threshold instead of truncating it ("Decoupled PPO" discussion).

**Mechanism, stated once.** For a softmax over the vocabulary, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`: pushing down token `y` moves its probability mass onto the currently most likely alternatives, which need not be better ones. When the sample is stale, the push-down is computed against probabilities that the current policy no longer holds. [[rollout-training-mismatch-tis]] offers a concrete version for the sampler gap: for negative-advantage rollouts with `π_learner/π_sampler < 1`, a decrease in `π_learner` may not appear in `π_sampler`, so updates keep pushing the learner down, which the authors give as their explanation for entropy collapse while stating that the exact mechanism is an open question (**Interpretation**; mechanism paragraph).

**Diagnostics.** Log separately by advantage sign: mean ratio, clip fraction, and the dual-clip activation count — [[openrlhf-ppo]]'s `clip_ratio` metric counts both signs together and does not count dual-clip activations (L180), so it cannot tell you which side is saturating. Log `vllm_kl`, token entropy, pass@1 and pass@k at large k, and the share of samples dropped by each filter, separated into "environment failure", "budget exhausted", and "model failure".

**Honesty about size of effect.** No source cited in this chapter measures how much of an asynchronous run's improvement comes from negative samples. The controls above are stability results, not attribution results. Claims about the positive/negative split belong to ch-43a, where they are measured.

---

## Recipe

Values are quoted at their loci. Framework defaults, released configs, and published model runs are separate facts and are labelled as such.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| AReaL, R1-Distill-Qwen | 1.5B–32B | RL | max staleness η | 4 (coding); 8 (math) | arXiv:2505.24298v5 §7.1 | verified 2026-09-17 | §7.4 Table 2 staleness sweep, 1 seed |
| AReaL, R1-Distill-Qwen 1.5B | 1.5B | RL | prompts per batch; answers per prompt; temperature / top-p | 512; 16; 1.0 / 1.0 | §B.1 Table 3 | verified 2026-09-17 | no ablation reported |
| AReaL, R1-Distill-Qwen 1.5B | 1.5B | RL | clip ε; PPO minibatches; LR (constant); max generation length | 0.2; 4; 2.0×10⁻⁵; 27,648 tokens | §B.1 Table 3 | verified 2026-09-17 | no ablation reported |
| AReaL device split | all | RL | inference : training devices | three quarters of devices allocated to inference | §7.1 | verified 2026-09-17 | stated as a fixed ratio for all runs (§7.1) |
| verl `fully_async_policy`, Qwen2.5-Math-7B | 7B | RL | `staleness_threshold`; `partial_rollout` | 0.5; True | github.com/verl-project/verl@753aed3 `fully_async_policy/README.md` L351–352 | verified 2026-09-14 ([[verl-rollout]]) | staleness ablation 0 / 0.1 / 0.3 / 0.5, L380–396 |
| verl `fully_async_policy`, Qwen2.5-Math-7B | 7B | RL | `rollout.n`; `ppo_mini_batch_size`; max response | 16; 32; 28K tokens | same file L335, L339–340 | verified 2026-09-14 | no ablation reported |
| verl `fully_async_policy`, Qwen2.5-Math-7B | 7B | RL | hardware; steps; train batch | 128 H20 (64 rollout : 64 train); 400; 512 | L333, L344–345, L361 | verified 2026-09-14 | 32 / 64 / 128-GPU comparison L353–361 |
| verl framework default | — | RL | `clip_ratio_c` (dual clip); `rollout_is` | 3.0; `null` (off) | `actor.yaml` L83; `config/algorithm/rollout_correction.yaml` | verified 2026-09-14 ([[verl-ppo-loss]]) | no ablation reported |
| OpenRLHF framework default | — | RL | `async_enable`; `async_queue_size`; `partial_rollout` | False; 1; False | github.com/OpenRLHF/OpenRLHF@64c1cc4 `cli/train_ppo_ray.py` L274–283 | verified 2026-09-14 ([[async-rollout]]) | no ablation reported |
| OpenRLHF framework default | — | RL | IS correction enable; type; thresholds | False; `tis`; [0.5, 5.0] | same file L257–270 | verified 2026-09-14 | no ablation reported |
| DAPO run from Qwen2.5-32B (flash-rl scripts) | 32B | RL | TIS cap C | 8 | github.com/yaof20/verl@fd02f7b `dapo_qwen32b_bf16.sh` L5, L106 | verified 2026-09-14 ([[rollout-training-mismatch-tis]]) | Fig. 1 TIS vs no TIS; no sweep over C |
| DAPO run from Qwen2.5-32B | 32B | RL | prompts per step; generation batch; samples per prompt; dynamic filter | 512; 1,536 prompts; 16; filter on accuracy, up to 10 generation batches | same script L28–35 | verified 2026-09-14 | no ablation reported |
| GSM8K PPO run, Qwen2.5-0.5B-Instruct | 0.5B | RL | TIS cap C | 2 | `gsm8k_qwen0_5b_int8.sh` L2, L32 | verified 2026-09-14 | Fig. 2, Fig. 4 |
| DeepSWE-Preview from Qwen3-32B | 32B | RL | containers per RL iteration; hardware | 512 (batch 64 × 8 passes); 64 H100 × 6 days | together.ai/blog/deepswe §2.2, intro | verified 2026-09-14 ([[deepswe]]) | no ablation reported |
| DeepSWE-Preview | 32B | RL | compact-filtering triggers | max context; max environment steps; 20-minute generation timeout | §2.3 | verified 2026-09-14 | Qwen3-14B curve (Fig. 6), no number |
| Qwen3-Coder-480B-A35B-Instruct | 480B (35B active) | RL | parallel environments | 20,000 | Qwen blog 2025-07-22, "Scaling Long-Horizon RL" | verified 2026-09-17 | no ablation reported |
| Llama-3.1-8B-Instruct (AsymRE) | 8B | RL | baseline offset δV; behavior-policy refresh; samples per prompt | ≈ −0.1; every 250 gradient steps; 8 | arXiv:2506.20520v2 §5.2–5.3 | verified 2026-09-14 ([[asymmetric-reinforce]]) | Fig. 5 left: 7/7 runs at δV = 0 collapse |
| LLaMA 3.1 8B async Online DPO | 8B | RL | GPU split | 1 GPU vLLM generation, 7 GPUs training (of 8) | arXiv:2410.18252 §4 | verified 2026-09-17 | Table 1: 57.20% win rate both, 230 vs 142 compute time |
| GLM-5 | 744B | RL | staleness threshold τ; ε_ℓ, ε_h; sync interval K | not reported | report §4.1.1–§4.1.2 (checked body) | not reported | — |
| Kimi k1.5 | not reported | RL | partial-rollout token budget | not reported | arXiv:2501.12599 §2.6.2 | not reported | — |

**Starting point for a small general-purpose run.** Every number below comes from a `verified` row above, with the conditions under which its source used it. Begin synchronous and measure the generation share of step time; verl's colocated 7B DAPO baseline on 128 H20 spent 49.9% there, which is the level at which decoupling repaid itself in that setting. If you decouple, start at `η = 1`: [[async-rlhf-noukhovitch]] measured N = 1 and N = 2 to be nearly identical at 410M on TL;DR, and [[areal-async-rl]] found η ≤ 2 within about 2 points of the synchronous oracle at 1.5B on math even with naive PPO. Raise η only after switching to a decoupled or ratio-corrected objective, which is what held 42.2 at η = 4 where naive PPO gave 23.3. Keep the sampler IS correction on when rollouts and gradients use different backends or precisions, with `C = 8` at 32B as the released DAPO value and `C = 2` at 0.5B. Leave dual clip at verl's `c = 3.0`. None of these values has a published ablation outside its own setting, so treat each as a starting point to re-measure, not a default.

---

## Generalization lens

**(a) What increases breadth.**

- *On-policy data preserves prior ability.* [[rls-razor]]: forgetting is predicted by KL to the base model measured on the new task (R² = 0.96 on ParityMNIST, R² = 0.71 on LLM runs), and on-policy updates are biased toward low-KL solutions; an oracle SFT distribution constructed to minimize that KL forgot even less than RL, which shows the mechanism is the KL bias rather than the algorithm (§4). [[retaining-by-doing]]: RL forgets less than SFT at comparable target performance across Llama and Qwen up to 8B on three task families (Abstract, Fig. 2). **Replicated** across two independent papers for the direction of the effect.
- *Mixed-domain single-stage RL.* [[deepseek-v3.1]] merges reasoning, agent, and alignment RL into one stage (§3); [[minimax-forge-agent-rl]] mixes reasoning, general QA, and agent tasks simultaneously and states that sequential training produces negative transfer (§4.1). **Result (single study) each, official reports without ablation.**
- *Longer horizons, more transfer.* [[skyrl-agent]] reports that SA-SWE-32B, trained only on SWE tasks, improved on Terminal-Bench, BrowseComp-Plus, and WebArena (Abstract, Table 3). [[asearcher]] reports +15.0 GAIA, +22.4 xBench, +15.6 Frames from RL on synthesized search QA (Fig. 1).

**(b) What causes narrowing or forgetting.**

- *Staleness.* [[areal-async-rl]] Table 2 is the cleanest measurement in this chapter: without a decoupled objective, η = 4 costs 18.7 AIME24 points and 22.3 MATH-500 points. [[async-rlhf-noukhovitch]] Figs. 3–4 show the same direction on TL;DR with a logarithmic dropoff in the number of stale steps.
- *Reward-metric drift.* [[rollout-training-mismatch-tis]] Fig. 6: the BF16 run without TIS logs a higher training reward than the run with TIS while scoring lower on AIME. Training reward is computed under the sampler and can move opposite to the held-out metric.
- *Entropy collapse.* [[rollout-training-mismatch-tis]] Fig. 5: entropy falls below 0.2 and keeps decreasing under an uncorrected INT8-rollout gap. [[asymmetric-reinforce]] Fig. 8: entropy falls faster for larger baselines, with support collapse to a single element predicted at `V ≥ V^µ` (Thm. 4.2).
- *Coverage loss.* [[rlvr-beyond-base-model]] reports that RLVR models have higher pass@1 than their base models but that at large k the base models match or exceed them on pass@k in the math, code, and visual-reasoning benchmarks tested; the authors interpret this as RLVR raising the probability of paths the base model could already sample (**Interpretation**). A run that only logs pass@1 cannot see this.
- *Filters as curricula.* Accuracy-based dynamic filtering, variance-weighted prompt replay (design pattern, unconfirmed implementation; §6), and Windowed FIFO each narrow the effective prompt distribution, and none of the sources reports the effect on the domains that were dropped. **Open question.**

**(c) How to measure it at this stage.**

1. **KL to the base model on a fixed non-task prompt set**, logged every N steps with fixed decoding settings. [[rls-razor]] validates KL-on-the-new-task as the forgetting predictor; adding a fixed set of unrelated prompts gives a second series that is directly interpretable as drift on ability you did not intend to change.
2. **pass@k at large k and token entropy** on a held-out split, not only pass@1 and training reward (§(b) above for why each is needed).
3. **A per-domain held-out regression set** that no filter, replay buffer, or scheduler window can reach, evaluated at fixed temperature and top-p.
4. **Broad periodic evaluation** — MMLU-class knowledge, IFEval-style instruction following, chat quality, and out-of-domain reasoning — on the same cadence as the throughput and staleness metrics, so that a regression is attributable to a training window.
5. **Staleness and system metrics as first-class series**: realized staleness distribution (not only the cap), `vllm_kl`, clip fraction split by advantage sign, and the share of samples dropped per reason.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Raising the staleness cap without changing the objective | Training reward keeps rising while held-out accuracy falls; large clip fractions on negative-advantage tokens | Re-run one short job at η = 0 or queue size 1 and compare held-out accuracy at equal steps, as in [[areal-async-rl]] Table 2 |
| Assuming the queue depth equals the realized staleness | Measured policy-version gaps exceed the configured cap | Log the version of every trajectory segment, as [[glm-5]] does with `(w₀, …, w_k)`; the OpenRLHF async payload carries no version field at all ([[async-rollout]] L149–151) |
| Ignoring the sampler–trainer probability gap | Negative k1 KL estimates; entropy falling below 0.2; responses growing abnormally long | Log `vllm_kl` and the maximum per-token probability difference; [[rollout-training-mismatch-tis]] reports 1.0 on DAPO-32B, 0.4 on the GSM8K example |
| Re-tokenizing the chat history to build the training sequence | PPO fails to converge even in single-turn; token counts differ between engine output and training input | Assert that concatenated per-turn token ids equal the training sequence ([[verl-rollout]] `agent_loop.rst` L155–182); adopt a TITO gateway ([[glm-5]] §4.1.2) |
| Training an MoE policy without routing replay | Instability specific to MoE that does not appear in a dense model of similar size | Enable routing replay ([[deepseek-v3.1]] Keep Routing, §3.1; verl `enable_rollout_routing_replay`) |
| Scoring environment crashes as model failures | Reward variance tracks cluster health; groups with all-zero reward cluster in time | Record a failure reason per sample and exclude environment collapse, padding or dropping the group ([[glm-5]] §4.1.2) |
| Greedy consumption of whichever trajectory finishes first | Average task difficulty in a batch drifts over training; gradient oscillation | Bound the fetch window ([[minimax-forge-agent-rl]] §3.1) and log the difficulty distribution per batch |
| Treating a filter as free | Held-out accuracy improves on the retained domains and degrades elsewhere | Keep a per-domain regression set outside the filter and evaluate it on a fixed cadence |
| Reporting BoN as "matching PPO" | A claim with no per-N human evaluation behind it | [[best-of-n]] App. G.3 states similar quality at equal average reward, "not shown"; the KL values are 3.2 (BoN-64) vs 18.0 / 14.0 (PPO) |

---

## Check your understanding

1. [[areal-async-rl]] reports that the AIME24 score with naive PPO falls from 42.0 at η = 0 to 23.3 at η = 4, yet the same staleness with a decoupled PPO objective gives 42.2. Explain what the decoupled objective changes about the gradient such that the same data becomes usable. Which term in Eqs. 4–5 does the work?
2. The AReaL paper argues that a **small** η is a system problem, not only a conservative choice. Reconstruct that argument from the admission rule `⌊(N_r − 1)/B⌋ ≤ i + η` and the fact that response lengths are long-tailed.
3. [[on-off-policy-rlhf]] reports that offline training makes a policy a better pairwise classifier and a worse generator. Why does this make "reward on the training stream" an unreliable stopping signal, and which additional series would catch the problem?
4. Staleness and the sampler–trainer gap are both described as off-policyness, and both are corrected with a truncated ratio. Give one property that distinguishes them, and explain why [[glm-5]] can reuse rollout log-probabilities as the behavior proxy while [[areal-async-rl]] recomputes probabilities under a proximal policy.
5. For `Â < 0` the PPO objective is unbounded in `r` while for `Â > 0` it is bounded. Derive that from the clipped surrogate, then explain why this asymmetry makes stale data more dangerous for negatives than for positives, and how dual clip, TOPR, and a lowered baseline each cut the same failure at a different place.
6. [[deepswe]] masks trajectories that ran out of context, steps, or time, and [[glm-5]] drops trajectories whose sandbox crashed. Both remove samples from the gradient. Explain why one is a reward-correctness fix and the other is a data-quality fix, and what each would corrupt if it were omitted.
7. Dynamic accuracy filtering, variance-prioritized prompt replay, and Windowed FIFO were each introduced for a different reason. Argue that all three change the same thing about the training distribution, and design the smallest measurement that would reveal the cost.
8. [[rls-razor]]'s forgetting law is measured for SFT and RL, not for stale rollout data at a controlled η. State what you would have to run to test whether the law extends to staleness, and what result would falsify the extension.

---

## Connections

- **Previous:** ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness. The harness built there is what detects the narrowing this chapter's filters and staleness caps can cause.
- **Depends on:** ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. The derivations behind dual clip, TOPR, and asymmetric baselines live there; this chapter applies them to stale data.
- **Depends on:** ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. §7 here is the systems side of the same loop: where the environments run and how their failures are labelled.
- **Related:** ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity, which holds the full treatment of [[rls-razor]] and [[retaining-by-doing]].
- **Related:** ch-42 — Reward Hacking and Judge Design, for the over-optimization ceiling that bounds Best-of-N in §2.
- **Next:** ch-55 — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation, which opens the code paths this chapter quotes by locus.
- **Later in the infra track:** ch-56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls; ch-57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers; ch-58 — Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability. The infra track ends at ch-58.
- **Capstone:** ch-59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate, which reproduces a stage under a generality gate and uses the instrumentation listed here.

---

## Sources

- [[areal-async-rl]] — the staleness parameter η and its admission rule, the decoupled PPO objective, and the Table 2 staleness ablation that anchors §4.
- [[async-rlhf-noukhovitch]] — the controlled measurement of how RLHF degrades with the number of stale minibatch updates, and the loss-robustness comparison.
- [[on-off-policy-rlhf]] — the controlled on-policy versus offline comparison, and the generation-versus-discrimination result used in §3.
- [[best-of-n]] — the analytic BoN KL formula, its printed values, and the over-optimization result that bounds BoN in §2.
- [[async-rollout]] — OpenRLHF's asynchronous trainer: queue, slots, lock, partial rollout, IS-correction modes, and the absence of a policy-version field.
- [[verl-rollout]] — verl's token-in/token-out vLLM server, weight-sync abort and resume, agent-loop masking, and the `fully_async_policy` timing tables quoted in §1.
- [[verl-ppo-loss]] — dual clip on negative advantages, `rollout_is_weights`, and the worked clipping example in the negatives section.
- [[openrlhf-ppo]] — the clipped surrogate's sign asymmetry, dual-clip default, and the limits of the `clip_ratio` diagnostic.
- [[rollout-training-mismatch-tis]] — the measured sampler–learner probability gap, truncated importance sampling and its variance argument, and the entropy and reward-drift side effects.
- [[glm-5]] — TITO, direct double-sided token masking, staleness filtering by model-version gap, environment-failure exclusion, and prefill-decode disaggregation for tail latency.
- [[deepseek-v3.1]] — Keep Routing and Keep Sampling Mask for MoE rollouts, and off-policy sequence masking for negative-advantage sequences.
- [[minimax-forge-agent-rl]] — Windowed FIFO as a distribution control, gateway-based black-box agent training, prefix-tree merging, and mixed-domain RL.
- [[replay-buffer-rlhf]] — a framework synthesis of replay designs, used in §6 as a pointer with its unconfirmed claims flagged.
- [[kimi-k1-5]] — partial rollouts: fixed output budget, continuation from a replay buffer, and segment loss exclusion.
- [[deepswe]] — Kubernetes sandbox orchestration at 512 containers per iteration, compact filtering, and the leave-one-out advantage arithmetic.
- [[r2e-gym]] — procedural construction of executable SWE environments and their counts.
- [[qwen3-coder]] — 20,000 parallel environments as the stated answer to environment scaling.
- [[skyrl-agent]] — the asynchronous pipeline dispatcher, transition-based backend bridge, and the cost and out-of-domain comparison against DeepSWE.
- [[asearcher]] — fully asynchronous agentic RL removing the turn limit, and the long-horizon behavior it enabled.
- [[agent-lightning]] — training-agent disaggregation and per-call transitions as an alternative to concatenate-and-mask.
- [[prime-intellect-environments-hub]] — environments as shared, versioned artifacts with their own evaluation reports.
- [[topr-tapered-off-policy-reinforce]] — the asymmetric tapered importance-sampling rule for negatives used in the negatives section.
- [[asymmetric-reinforce]] — the baseline phase transition and the δV = −0.1 recommendation for stale data.
- [[rls-razor]] — the forgetting law, KL-to-base as the predictor, and the oracle-SFT control.
- [[retaining-by-doing]] — the independent replication that RL forgets less than SFT at comparable target performance.
- [[rlvr-beyond-base-model]] — pass@k coverage loss after RL, the reason large-k pass@k is on the measurement list.
