---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: arXiv:2602.15763v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
---

# Excerpt: GLM-5: from Vibe Coding to Agentic Engineering

- **Authors:** GLM-5 Team, Zhipu AI & Tsinghua University (byline)
- **Year:** 2026 (arXiv v2 2026-02-24)
- **Source type:** official technical report (models and code: https://github.com/zai-org/GLM-5)
- **Used in:** [[read]] §1, §2, §4, §5, §7, Recipe, Generalization lens

## Scale and stage order (§1, §2, §3)
- 744B total / 40B active, 256 experts, 80 layers; "doubling the total size of GLM-4.5, which utilized 355B total and 32B active parameters" (§2.1). Total training-token budget 28.5T for the base model (§1, §2).
- §1 describes post-training as "a sequential Reinforcement Learning pipeline—starting with Reasoning RL, followed by Agentic RL, and finishing with General RL", with on-policy cross-stage distillation used "to prevent catastrophic forgetting".
- §3 opens with SFT, then "specialized Reinforcement Learning (RL) stages for reasoning and agentic tasks, and concluding with a general RL stage", and places "on-policy cross-stage distillation as the final refinement". §3.5 names the teachers as "Reasoning RL and General RL". The two placements of distillation are not reconciled in the text.

## Mid-training (§2.3)
- Three context stages: "32K (1T tokens), 128K (500B), and 200K (50B)". "Long documents and synthetic agent trajectories are up-sampled at the later stages accordingly."
- Software-engineering data: repo-level files, commit diffs, issues, PRs and relevant source files concatenated into unified sequences; relaxed repository filtering yields "approximately 10 million issue–PR pairs"; "After filtering, the issue–PR portion of the dataset comprises approximately 160B unique tokens."
- Long-context data combines natural sources with synthetic interleaved packing; at 200K a small proportion of MRCR-like data is added. "a subsequent 200K mid-training stage, building upon the initial 128K phase, further bolstered the model's performance even within the 128K context window."

## SFT (§3.1)
- Three categories: General Chat; Reasoning; "Coding & Agent: frontend and backend engineering code, tool calling, coding agents, search agents, and general-purpose agents". Compared with GLM-4.5, GLM-5 "significantly expands the scale of Agent and Coding data during the SFT stage". Sizes are not printed.
- Maximum context in SFT: 202,752 tokens. Three thinking characteristics: interleaved thinking, preserved thinking (all thinking blocks retained across turns in coding-agent settings), turn-level thinking.
- Negative content handling: "Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions."

## Reasoning RL (§3.2)
- GRPO with IcePop; the KL regularization term is removed. Eq. 1 multiplies the clipped PPO term by `pop(ρ_{i,t}, 1/β, β)`, which returns ρ inside [1/β, β] and 0 outside; ρ_{i,t} = π_old^train / π_old^infer.
- Settings printed: β = 2, ε_low = 0.2, ε_high = 0.28, "Training is performed entirely on-policy with a group size of 32 and a batch size of 32."
- DSA indexer: `torch.topk` is used as a deterministic top-k operator; non-deterministic CUDA or TileLang top-k "caused drastic performance degradation during RL after only a few steps, accompanied by a sharp drop in entropy". Indexer parameters are frozen during RL by default.

## Agentic RL (§3.3, §4.1)
- Objective (§4.1): `L(θ) = E_x[(1/K) Σ_i (r(x,y_i) − r̄(x))]` over K traces from π_old. "only model-generated tokens are used for optimization, and the environment feedback is ignored in loss computation."
- Fully asynchronous: training and inference engines on separate GPUs; weights pushed every K gradient updates; "we also reset the optimizer after each weight update of the inference engine".
- Token-in-Token-out gateway records token IDs and metadata, stated as "critical for asynchronous RL training because it preserves exact action-level correspondence".
- Direct double-sided importance sampling: `r_t(θ) = exp(log π_θ(a_t|s_t) − log π_rollout(a_t|s_t))`, with `f(x; ε_ℓ, ε_h) = x` inside (1−ε_ℓ, 1+ε_h) and 0 otherwise. π_old is discarded, which "eliminate[s] the computational overhead of separate old-policy inference".
- Staleness filter: for each response the sequence of rollout model versions (w_0, …, w_k) is logged; a sample is discarded when `w′ − w_0 > τ` for the current version w′.
- Environment-failure filter: samples that fail because of sandbox collapse are excluded; "we pad the group by repeating valid samples if the number of valid samples exceeds half of the group size; otherwise, we drop the entire group."
- Multi-Task Rollout Orchestrator with per-task microservices "supports over 1k concurrent rollouts".

## Environment scaling (§4.2)
- SWE: RepoLaunch-based pipeline builds "over 10k verifiable environments across thousands of repositories spanning 9 programming languages, including Python, Java, Go, C, CPP, JavaScript, TypeScript, PHP, and Ruby", with F2P and P2P test extraction.
- Terminal: seed-based and web-corpus-based synthesis into Harbor-format Dockerized tasks; "thousands of diverse and verifiable terminal-agent environments with Docker construction accuracy exceeding 90%".
- Search: a Web Knowledge Graph over "more than two million high-information web pages"; three-stage difficulty filtering (drop questions a tool-free reasoning model answers in at least one of eight attempts; drop questions an early-stage agent solves in a few steps; bidirectional verification).

## Context management at inference (§4.2.4)
- Keep-recent-k folds tool observations older than the most recent k rounds into a placeholder string. With k = 5, BrowseComp goes from 55.3% (without) to 62.0% (with).
- Hierarchical Context Management combines keep-recent with Discard-all above a threshold T = 32k, selected by parameter search, "reaching a final score of 75.9".
- Judge protocol standardized to the official OpenAI evaluation prompt with o3-mini as judge.

## On-policy cross-stage distillation (§3.5)
- Motivation: "sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities."
- Teachers are the final checkpoints of earlier stages; prompts come from those teachers' RL training sets, "mixed in appropriate proportions" (proportions not printed).
- The GRPO advantage is replaced by `Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t}|x,y_{i,<t}) / π_θ^train(y_{i,t}|x,y_{i,<t}) ) ]` (Eq. 2). Group size 1, batch size 1024.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2602.15763 (v2, 2026-02-24): §1, §2.1, §2.3, §3.1-§3.5, §4.1, §4.2, §6.1.3.
- Not reported by the source: SFT dataset size; agentic-RL learning rate, group size K, ε_ℓ and ε_h values, staleness threshold τ; distillation prompt proportions and step count; any before/after number isolating cross-stage distillation; GPU hours.
