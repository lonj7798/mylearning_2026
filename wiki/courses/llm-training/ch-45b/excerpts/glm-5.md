---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2602.15763v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
---

# Excerpt: GLM-5: from Vibe Coding to Agentic Engineering — agentic-RL sections

- **Authors:** GLM-5 Team, Zhipu AI & Tsinghua University (byline)
- **Year:** 2026 (arXiv v2 2026-02-24)
- **Source type:** official technical report (models and code: https://github.com/zai-org/GLM-5)
- **Used in:** [[read]] §2, §3, §5, §6, Recipe, Generalization lens

## Stage placement (§1, §3.3)
"We implemented a sequential Reinforcement Learning pipeline—starting with Reasoning RL, followed by Agentic RL,
and finishing with General RL" (§1). Agentic RL covers "coding and search agent tasks" (§3.3).

## Objective and observation masking (§4.1)
For each problem `x`, `K` agent traces `{y_1, …, y_K}` are sampled from `π_old`, and the model is optimized with
`L(θ) = E_{x∼D}[ (1/K) Σ_{i=1..K} (r(x, y_i) − r̄(x)) ]`, where `r̄(x) = (1/K) Σ_i r(x, y_i)`. There is no division
by the reward standard deviation. Verbatim: "It is noted that only model-generated tokens are used for
optimization, and the environment feedback is ignored in loss computation."

## Token-in-token-out (§4.1.2)
"token-in-token-out (TITO) means the training pipeline consumes the exact tokenization and decoded-token stream
produced by the inference engine". Text-in-text-out "treats the rollout engine as a black box that returns
finalized text; the trainer then reconstructs the trajectory by re-tokenizing that text". The stated risk:
"re-tokenization can introduce subtle mismatches in token boundaries, whitespace/normalization handling,
truncation, or special-token placement, which in turn can corrupt step alignment between actions and
rewards/advantages". The report implements "a TITO Gateway that intercepts all generation requests from rollout
tasks and records each trajectory's token IDs and metadata".

## Direct double-sided importance sampling (§4.1.2, Eqs. 3-5)
`L(θ) = E_t[ f(r_t(θ), ε_ℓ, ε_h) Â_t log π_θ(a_t | s_t) ]` with `r_t(θ) = exp(log π_θ(a_t|s_t) − log π_rollout(a_t|s_t))`
and `f(x; ε_ℓ, ε_h) = x` if `1 − ε_ℓ < x < 1 + ε_h`, else `0`. Rollout log-probabilities are reused as the behavior
policy, so `π_θ_old` is never recomputed; tokens outside the interval "are entirely masked from gradient
computation". The report relates this to IcePop and states its own version is "simpler by further removing the
π_θold". `ε_ℓ` and `ε_h` values are not printed.

## Dropping off-policy and environment-failure samples (§4.1.2)
- Staleness: for each response the sequence of rollout model versions `(w_0, …, w_k)` is logged; a sample is
  discarded if `w′ − w_0 > τ` for the current version `w′` and a threshold `τ`. `τ` is not printed.
- Environment failures: "coding-agent sandboxes can be inherently unstable and may fail for reasons unrelated to
  the model (e.g., environment crashes). Such failures introduce noisy training signals because they reflect
  environment instability rather than the model's capability." Those samples are excluded by recorded failure
  reason.
- Group repair rule (verbatim): "For group-based sampling methods such as GRPO, removing failed samples can leave
  an incomplete group. In that case, we pad the group by repeating valid samples if the number of valid samples
  exceeds half of the group size; otherwise, we drop the entire group."

## Serving choices that exist because of multi-turn rollouts (§3.6.2, §4.1.1, §4.1.2)
- Asynchronous, decoupled inference and training engines; weights pushed back "every K gradient updates"; the
  optimizer is reset after each weight update of the inference engine.
- A server-based Multi-Task Rollout Orchestrator registers each task's rollout and reward logic as a microservice,
  standardizes "trajectories from all agentic tasks into a unified message-list representation", "supports over 1k
  concurrent rollouts" and controls per-task rollout ratios.
- DP-aware routing pins all requests of one rollout to a fixed data-parallel rank by consistent hashing so the
  shared prefix stays in the KV cache.
- Prefill-decode disaggregation is used because "a heavy prefill can preempt or disrupt ongoing decodes",
  "significantly improving tail behavior in multi-turn agentic RL".
- Heartbeat-driven fault tolerance deregisters unhealthy rollout servers (§3.6.3).

## Environment scaling (§4.2)
- SWE: a RepoLaunch-based pipeline builds "over 10k verifiable environments across thousands of repositories
  spanning 9 programming languages, including Python, Java, Go, C, CPP, JavaScript, TypeScript, PHP, and Ruby",
  with Fail-to-Pass and Pass-to-Pass tests extracted by LLM-generated log parsers.
- Terminal: a three-phase synthesis pipeline (task draft, concrete implementation in the Harbor format, iterative
  refinement) yields "thousands of diverse and verifiable terminal-agent environments with Docker construction
  accuracy exceeding 90%".

## Evaluation note (§6.1.3)
BrowseComp is evaluated with "a discard-all strategy as context management ... which is the same as DeepSeek-V3.2,
and Kimi K2.5". GLM-5 is reported at $4,432 on Vending-Bench 2.

## Verification
- Checked on 2026-09-15 against the cached PDF text of arXiv:2602.15763v2: §1, §3.3, §3.6.2, §3.6.3, §4.1, §4.1.1,
  §4.1.2, §4.2, §4.2.1, §4.2.2, §6.1.3.
- Not reported by the source: `ε_ℓ`, `ε_h`, `τ`, group size `K` for agentic RL, learning rate, number of agentic-RL
  steps, per-environment reward definitions, and any ablation isolating TITO, the staleness filter or the
  environment-failure filter.
