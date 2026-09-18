<!-- scope: DeepSWE-Preview blog (Agentica + Together AI, 2025-07): RL-only training of Qwen3-32B as a multi-turn SWE agent on R2E-Gym environments, GRPO++ with compact filtering, Kubernetes rollout infrastructure, test-time scaling with hybrid verifiers, negative results (SFT warm start, other datasets, non-thinking mode).
     deps: [[r2e-gym]], [[dapo]], [[dr-grpo]], [[rloo]]
     see-also: [[deepswe-recipe]], [[skyrl-agent]], [[swe-smith]], [[swe-gym]], [[agentica-deepcoder]], [[deepscaler]]
-->

# DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL
- **Core Insight:** DeepSWE-Preview, trained from Qwen3-32B with RL only on 4.5K R2E-Gym problems for six days on 64 H100 GPUs, scores 42.2% Pass@1 (average of 16 runs) and 71.0% Pass@16 on SWE-Bench-Verified, and 59.0% with hybrid test-time scaling at K=16 (intro; §4 table).
- **Guideline:** When a multi-turn agent is trained with a binary outcome reward and trajectories can end by context limit, step limit, or timeout, mask the loss of those trajectories instead of scoring them as failures, because the authors' Qwen3-14B ablation shows that this "compact filtering" prevents or delays reward collapse (§2.3, Figure 6; the effect is shown only as a curve, without a reported number).
- **Authors:** Michael Luo, Naman Jain, Jaskirat Singh, Sijun Tan, Ameen Patel, Qingyang Wu, et al. (Agentica team and Together AI; last authors Koushik Sen, Ion Stoica)
- **Year:** 2025 (blog published 2025-07-02; no arXiv version)
- **URL:** https://www.together.ai/blog/deepswe (the rLLM README cites a Notion copy of the same post)
- **Source type:** official blog; the recipe companion also uses released code (github.com/agentica-project/rllm@709dec43b740)
- **Relevant topics:** agentic RL, SWE agents, executable environments, GRPO variants, trajectory masking, sparse outcome reward, test-time scaling, verifiers, SFT warm start vs RL from base

## Summary
The post introduces DeepSWE-Preview, a coding agent trained from Qwen3-32B with reinforcement learning only, with no SFT stage (intro). Training uses 4.5K problems from R2E-Gym, each mapped to a Docker image, and a reward of 1 when the final patch passes selected tests and 0 otherwise (§2.1-§2.2). The RL algorithm, GRPO++, combines changes from DAPO, Dr. GRPO, and RLOO with two changes of the authors: compact filtering and no entropy loss (§2.3). The post reports 42.2% Pass@1 and 71.0% Pass@16 on SWE-Bench-Verified, and 59% with a hybrid of execution-based and execution-free verifiers (§4). It describes emergent behaviors, and three experiments that did not help: RL after SFT on Claude trajectories, RL on SWE-Smith or SWE-Gym data, and non-thinking mode (§5-§6). Dataset, code, training logs, and evaluation logs are released (intro).

## Key Contributions
- RL-only training of a 32B SWE agent from Qwen3-32B with thinking mode enabled (intro; HF model card "DeepSWE Overview").
- GRPO++: clip high, no KL loss, no reward standard-deviation normalization, length normalization by max context length, leave-one-out advantage, compact filtering, no entropy loss (§2.3).
- Compact filtering: loss masking for trajectories that reach max context, max environment steps, or a 20-minute generation timeout (§2.3).
- Kubernetes support in R2E-Gym to run 512 containers per RL iteration (§2.2).
- A test-time scaling comparison of output-token scaling against rollout scaling with verifiers (§3).
- Reported negative results for SFT warm start, alternative SWE datasets, and non-thinking mode (§6).

## Key Figures/Tables to Study
- Figure 2: SWE-Bench-Verified Pass@1 rising from 23% to 42% over 200 RL steps.
- Figure 5: GRPO++ against GRPO on FrozenLake (training reward).
- Figure 6: Qwen3-14B with and without compact filtering. Figure 7: response length falls while environment steps rise.
- Figure 9: Pass@1 against max output tokens (16K to 128K). Figure 10: TTS strategies against K.
- §4 table (Figure 11): DeepSWE-Preview against open-weight SWE agents with scaffold labels.

## Technical Details
**Task and tools.** An episode is a pull request in a container with a terminal and the repository (§1). R2E-Gym defines four tools: Execute Bash, Search, File Editor (view, create, replace strings, insert, undo), and Finish/Submit, which ends the trajectory (§2.2).

**Reward.** The reward is 1 when the generated patch passes a selected sample of Pass2Pass and Fail2Pass tests within the time limit, and 0 when at least one test fails or the run times out (§2.2). The training time limit is 5 minutes; the official SWE-Bench evaluation limit is 30 minutes (§2.2).

**Data.** 4.5K problems from a subset of R2E-Gym; problems from repositories shared with SWE-Bench-Verified, such as sympy, were removed to avoid contamination (§2.1).

**Infrastructure.** Each RL iteration of the final run spawned 512 Docker containers (BS=64, 8 passes) (§2.2). Running thousands of containers crashed the Docker daemon, so the authors moved scheduling to Kubernetes; each worker node has about 200 CPU cores and over 6 TB of local NVMe SSD with preloaded SWE-bench images (§2.2). The cluster scales beyond 1000 CPU cores; the autoscaler removes nodes underutilized for roughly twenty minutes (§2.2). Training used 64 H100 GPUs for six days (intro). The framework is rLLM (intro; §2), which the rLLM README describes as built on a "heavily modified fork of verl".

**Multi-turn GRPO and GRPO++.** Following prior work (RAGEN, verl, ROLL, ART, Sky-RL), the multi-turn extension masks environment observations (user messages in ChatML format) in each trajectory (§2.3). GRPO++ components, as stated (§2.3):
1. Clip high (DAPO): a larger upper clip bound, stated to encourage exploration and stabilize entropy.
2. No KL loss (DAPO).
3. No reward standard deviation (Dr. GRPO), stated to remove difficulty bias.
4. Length normalization (Dr. GRPO): the surrogate loss is divided by max context length.
5. Leave one out (RLOO): the baseline for each sample excludes that sample.
6. Compact filtering (authors): see next paragraph.
7. No entropy loss (authors): entropy loss led to exponentially increasing entropy and collapse; it is stated to be unnecessary when the base model's token-level entropy is within 0.3-1.

**Compact filtering.** DAPO's overlong filtering masks max-context trajectories; compact filtering also masks trajectories that time out during generation (20 minutes) or reach maximum steps (§2.3). The authors give two reasons: (a) an agent can pass all tests by chance, for example by answering correctly in the first 10 steps and editing random files later, and rewarding such trajectories leads to collapse (Figure 6); (b) average response length per step decreases while environment steps increase (Figure 7) (§2.3).

**Advantage of a failed rollout (derived; the post gives no formula).** With a leave-one-out baseline and no std normalization, A_i = r_i − (1/(n−1)) Σ_{j≠i} r_j, where r_i ∈ {0,1} is rollout i's reward and n the rollouts per problem. With n = 8 and 2 successes, a failure gets 0 − 2/7 ≈ −0.29 and a success gets 1 − 1/7 ≈ 0.86; when all 8 fail, every advantage is 0.

**Test-time scaling and evaluation.** DeepSWE-Verifier (execution-free) is trained for 2 epochs on correct and incorrect patches; the execution-based verifier uses an LLM to generate tests, and the hybrid follows [[r2e-gym]] (§3). Evaluation uses the R2E-Gym codebase at 64k max context and 100 max environment steps, with patches scored by the official SWE-bench repository and Pass@1 averaged over 16 runs (§4).

**Recipe values.** The recipe ledger, including the released training script, which differs from the blog in batch size, step limit, and response length, is in [[deepswe-recipe]].

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Agentic training results.** Pass@1 rose from 23% to 42% in about 200 RL steps (Figure 2). §4 table: DeepSWE-Preview 42.2% (R2E-Gym agent), 57.9% (hybrid Best@8), 59% (hybrid Best@16); Devstral-Small 24B 46.6%, SWE-Agent-LM 32B 40.2%, Skywork-SWE 32B 38.0% (47.0% with execution-free Best@8), R2EGym-Agent 32B 34.4%. Rows use different scaffolds.
- **Negative feedback.** Failed or timed-out patches get reward 0 (§2.2), which is negative as gradient through the leave-one-out advantage (derived above). Trajectories that end by context, step, or generation-time limit are masked, so they carry no gradient (§2.3).
- **Long context.** Scaling max output tokens from 16K to 128K raises Pass@1, but the gain beyond 32K is ≤2%; Pass@1 at 128K is 43.2% (§3, Figure 9). The authors state that scaling output tokens "does not seem to be effective" for SWE tasks (§3).
- **Rollout scaling.** Hybrid scaling reaches 59.0% at K=16, and "a majority" of TTS gains are reached at K=8 (§3). Execution-based or execution-free verifiers alone "can bring 10+%" (Figure 10 caption).
- **Emergent behavior (anecdotes, §5).** The agent checks edge cases and searches for and runs regression tests; it uses about 2K thinking tokens on localization and fix steps and about 100-200 on navigation or search steps.
- **SFT warm start (§6).** RL on four SFT models (Qwen3-32B trained on Claude-Sonnet 3.7/4 thinking and non-thinking trajectories) did not improve after 100 iterations; the SFT models were slightly below SWE-agent-LM-32B.
- **Training data (§6).** RL on SWE-Smith and SWE-Gym gave limited improvement with a high solve-none rate; the authors state R2E-Gym "provided sufficient curriculum learning".
- **Non-thinking mode (§6).** RL in non-thinking mode gave limited improvement; the authors suggest model capacity as a possible cause.
- **Generality.** The post reports no evaluation outside SWE-Bench-Verified (and its SWE-Bench-Hard validation curve, Figure 2).

## Connections
- [[r2e-gym]]: source of the training environments, tools, and the hybrid verifier used for TTS.
- [[deepswe-recipe]]: recipe ledger with blog values and released-script values.
- [[dapo]]: origin of clip high, no KL loss, and overlong filtering, which compact filtering extends.
- [[dr-grpo]]: origin of removing the std term and the length normalization used in GRPO++.
- [[rloo]], [[rloo-vs-grpo]]: leave-one-out baseline used for the advantage.
- [[swe-smith]], [[swe-gym]]: alternative RL datasets that gave limited improvement (§6).
- [[skywork-swe]]: Skywork-SWE 32B, a baseline in the §4 table.
- [[ragen-starpo]], [[loop-appworld]]: RAGEN and LOOP, multi-turn RL work cited in §2.3.
- [[skyrl-agent]]: SkyRL-Agent paper from the same lab; the §4 table lists a "SkyRL-Agent (14B)" row at 21.6% with the OpenHands scaffold.
- [[agentica-deepcoder]], [[deepscaler]]: earlier Agentica RL models trained with rLLM.
- [[openhands-data]]: OpenHands scaffold used by several baselines in the §4 table.

## Verification
- Created on 2026-09-14 from https://www.together.ai/blog/deepswe (post dated 2025-07-02, fetched 2026-09-14); model card https://huggingface.co/agentica-org/DeepSWE-Preview; rLLM README and code at github.com/agentica-project/rllm@709dec43b740.
- Audit claims not found in the source:
  - "Training: max context 64K, max 100 environment steps" → the post gives 64k and 100 steps for evaluation only (§4); the released script sets max_response_length=32768 and agent.max_steps=50 (see [[deepswe-recipe]]).
  - "SkyRL-Agent later puts DeepSWE's cost at 9,180 H100-hours" → not in this post.
  - "SFT on Claude/GPT-4 trajectories gave no gain" → the post names only Claude-Sonnet 3.7/4 and states that RL on top of the SFT models did not improve after 100 iterations (§6).
- Internal inconsistency: hybrid TTS is 59% / 59.0% in §3-§4 and "59.2%" in §8 and the rLLM SWE README.
- Not reported by the post: learning rate, clip values, KL settings, training max context and step limit, verifier base model and training data size, per-problem difficulty filtering.
