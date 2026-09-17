---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: arXiv:2607.24653v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2607.24653
created_at: "2026-09-15"
---

# Excerpt: Kimi K3: Open Frontier Intelligence

- **Authors:** Kimi Team (Moonshot AI)
- **Year:** 2026 (arXiv v1 2026-07-27; v2 2026-08-07)
- **Source type:** official technical report
- **Used in:** [[read]] §1, §4, §5, §6, Recipe, Distillation, Negative feedback

## Scale and pre-training (Abstract, §3)
- 2.8T total parameters, 104B activated, 16 of 896 routed experts (Stable LatentMoE), Kimi Delta Attention plus Attention Residuals, native vision, 1M-token context. The abstract states "approximately 2.5× improvement in overall scaling efficiency over Kimi K2".
- Pre-training corpus: Web Text, Code, Mathematics, Knowledge plus a vision corpus; pipelines build on Kimi K2 and K2.5, including the K2 rephrasing recipe (§3.1). The report does not state that agentic trajectories are injected into pre-training.
- Context curriculum: 8K → 64K during pre-training, 256K → 1M during cooldown (§3.4). NoPE: positional information comes from KDA gating, so the model "extrapolates directly to 1M-token contexts without any positional-encoding modification".

## Post-training stage order (§4.1)
- Three stages: SFT cold start, then RL that trains "a single expert for each domain at every reasoning effort level", then Multi-Teacher On-Policy Distillation (MOPD) to consolidate.
- RL domains: (i) general tasks; (ii) general agents (long-horizon assistant tasks, deep research, paragraph-level writing); (iii) coding agents (SWE, coding experience, kernel tasks, web development). "Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models."
- SFT data: trajectories synthesized by "domain-specialized models from the prior Kimi series, followed by multi-stage verification and human-in-the-loop annotation", serialized with an XTML-based chat template. Sizes are not printed.
- Quantization-aware training (MXFP4 expert weights, MXFP8 activations) runs from SFT onward, so "rollout and training share the same quantization scheme — eliminating the train–inference mismatch" (§4.1.4).

## Partial rollout under long horizons (§4.1.2)
- K completions are sampled for each of N prompts; "the generation phase pauses as soon as a fraction λ ∈ (0,1) of trajectories completes (i.e., λNK)". Paused rollouts are enqueued and resumed first in the next iteration, supported by resumable sandboxes (§5.3.2).
- A single long-horizon trajectory therefore spans several iterations. "Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization", following the K2.5 algorithm.

## Reasoning-effort RL and negative reward (§4.1.2)
- Each problem x carries an initial token budget b_0(x) estimated from the cold-start model. The task reward is "overridden with −1 for trajectories whose total token budget T(y) exceeds a scaled threshold τ · b_0(x)".
- For general tasks T(y) counts thinking tokens; for agentic tasks it counts "cumulative output tokens, including both reasoning traces and tool-call arguments".
- Curriculum: train a max-budget variant first with large τ, then anneal τ to obtain high- and low-effort experts; τ is set per domain with human guidance.
- Agentic GRM: tournament-style binary comparisons with a mandatory protocol — read the outcome, generate a rubric, score candidates, record scores. Verbosity control: a candidate whose output exceeds σ · ℓ_0 "automatically loses the binary comparison".

## Multi-Teacher On-Policy Distillation (§4.1.3, Eq. 15)
- For domain d and sampled effort e, the per-token reward is
  `r_opd^d(y_t | e, x, y_<t) = clip( sg[ log( π_teacher^(d,e)(y_t|x,y_<t) / π_θ(y_t|e,x,y_<t) ) ], −R_max, R_max )`.
- The dense reward "seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks".
- "While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting."

## Unified white-box RL environment (§4.2.1)
- Stated failure mode: "Training with a single fixed agent harness can cause a model to overfit to a particular tool schema, system prompt, context management mechanism, or interaction protocol."
- The environment represents a harness as configurable modules (tool interfaces, system prompts, context management strategies, skills, memories, subagents) and can instantiate Kimi Code, Claude Code, Codex, OpenClaw and Hermes, "as well as entirely new ones". Harness configurations are varied per task group during RL.

## Environment families (§4.2.2-§4.2.7)
- Knowledge-graph-guided task synthesis: an agent-built directed acyclic graph of concepts drives keyword sampling, web material retrieval, and task synthesis across coding, knowledge and vision task types.
- Kernel optimization: correctness plus performance reward; matching an expert implementation gives 0.5, approaching the hardware roofline moves the reward toward 1; a hacking-detection system penalizes "CUDA graph replay, input caching, and precision reduction".
- Personal assistant: mock Gmail, Notion, Slack and Canvas applications over multiple simulated days; "A single rollout may involve up to thousands of tool calls and millions of context tokens."
- Autonomous Execution Tasks: initial state, constrained goal, tool action space, execution budgets, independent verifier; "Rewards are grounded in the verifier's evaluation of the final environment state rather than the agent's self-reported completion"; public diagnostic verifiers are paired with hidden held-out verifiers.
- Web development: "Every task runs in a containerized sandbox and is rolled out under diverse agent scaffolds rather than a single fixed harness, to promote cross-scaffold generalization." Reward is zeroed "when a project fails to build, runs with errors, or fakes rather than implements the artifact".

## Verification
- Checked on 2026-09-15 against the Kimi K3 technical report (arXiv:2607.24653, v2 2026-08-07): Abstract, §3.1, §3.3, §3.4, §4.1, §4.2, §5.3.
- Not reported by the source: SFT dataset size; RL learning rate, K, N, λ, τ, σ, R_max; number of environments or tasks per family; GPU hours; pre-training token count for the text corpus.
