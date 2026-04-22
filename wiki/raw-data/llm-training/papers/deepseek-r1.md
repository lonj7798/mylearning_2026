<!-- scope: pure-RL reasoning model training from a rule-based verifier
     deps: [[rlvr-tulu3]]
     see-also: [[entropy-mechanism-llm-rl]], [[math-shepherd]]
-->

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL
- **Core Insight:** Reasoning behavior (long chains-of-thought, self-correction, "aha moments") can emerge from pure RL on a base LM with only a rule-based verifier reward — no SFT traces, no PRM, no preference RM needed — if you use GRPO with very long rollouts.
- **Guideline:** For verifiable domains, start with a capable base model, define a rule-based reward = (accuracy via exact-match grader) + (format reward for `<think>…</think><answer>…</answer>`), use GRPO with a large group size and long rollout budgets (≥8k tokens), and let chain-of-thought length grow on its own.
- **Authors:** DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, et al.; 200+ contributors)
- **Year:** 2025 (Nature 645, 633–638)
- **URL:** https://arxiv.org/abs/2501.12948
- **Relevant topics:** GRPO, RLVR, rule-based reward, R1-Zero, emergent long CoT, aha moment, format reward

## Abstract
The DeepSeek-R1 report introduces two models: **R1-Zero**, trained directly from a DeepSeek-V3-Base checkpoint with pure RL and a rule-based reward (no SFT); and **R1**, which adds a small amount of cold-start SFT and a multi-stage RL pipeline to fix readability and language-mixing issues. Both demonstrate that long-horizon reasoning behaviors (reflection, backtracking, working through multiple approaches) emerge from the RL pressure alone. DeepSeek-R1 matches or surpasses OpenAI o1 on AIME 2024 (79.8 vs 79.2 pass@1), MATH-500 (97.3 vs 96.4), and LiveCodeBench, among other benchmarks.

## Key Contributions
- **R1-Zero: pure-RL reasoning from a base LM.** Pure GRPO training from DeepSeek-V3-Base with only two reward components:
  - **Accuracy reward:** rule-based grader (exact-match math answers, unit-test execution for code, regex match for instruction following).
  - **Format reward:** +1 iff output follows `<think>...</think><answer>...</answer>`.
- **Why rule-based rewards sidestep reward hacking:** the reward is a deterministic, closed-form function; there is no learned RM to overoptimize, no Goodhart drift. The only remaining exploit surfaces are bugs in the grader itself.
- **Emergent long chain-of-thought:** average response length grows from ~400 tokens early training to 10k+ tokens late training, without any supervised example of such length.
- **"Aha moment":** mid-training, the model spontaneously develops self-reflection patterns ("Wait, let me reconsider…", "Let me check step 3…") — documented as a phase transition in behavior, not a gradient-trained style.
- **R1 multi-stage pipeline:** (1) small cold-start SFT on clean CoT, (2) reasoning-RL, (3) rejection-sampling SFT, (4) broad-domain RL with mixed reward; fixes R1-Zero's language mixing and unreadable traces.
- **Distillation transfer:** R1 traces distilled into Qwen-7B / Llama-8B produce models competitive with much larger baselines.

## Key Figures/Tables to Study
- **Fig. 2** (R1-Zero AIME accuracy vs training step) — the iconic monotone climb from ~15% to ~70% pass@1 with pure RL.
- **Fig. 3** (average response length vs step) — length grows; correlates with capability gain.
- **Fig. 4** ("aha moment" snippet) — sampled rollout showing self-correction emerging mid-training.
- **Benchmark table** — AIME / MATH-500 / Codeforces numbers vs o1, Claude 3.5, GPT-4o.

## Technical Details
- **Algorithm:** GRPO (no value model; advantage = normalized group reward). Group size G = 16–64 rollouts per prompt. Sequence length up to 32k.
- **Reward:** `r = r_acc + r_format` where `r_acc ∈ {0, 1}` from a rule grader and `r_format ∈ {0, 1}` from a regex check.
- **KL penalty:** KL-to-reference term kept (β small, applied per token on reward) for R1-Zero; arguably the only regularizer preventing entropy collapse in this regime — links directly to **[[entropy-mechanism-llm-rl]]**.
- **No critic / no PRM:** advantages are pure group-relative: `A_i = (r_i − mean(r_{1:G})) / std(r_{1:G})`.
- **Cold-start SFT (R1 only):** ~thousands of hand-cleaned long-CoT examples before RL.
- **Language-mixing fix:** add a "language consistency reward" in later RL stages.
- **Failure modes reported:** R1-Zero responses become hard to read (interleaved English/Chinese/symbols); repetition loops; occasional reward hacking of the format tag via hollow `<think>` blocks — mitigated by careful format checker.

## Connections
- Most prominent industrial validation of **[[rlvr-tulu3]]**'s verifier-grounded reward thesis.
- Entropy dynamics during R1-Zero training are the canonical testbed for **[[entropy-mechanism-llm-rl]]**.
- Distillation pipeline shows how RLVR-trained reasoning transfers — bridges to pre-RL SFT literature.
- Rule-based reward sidesteps all of **[[reward-model-overoptimization]]** on verifiable prompts by construction.
