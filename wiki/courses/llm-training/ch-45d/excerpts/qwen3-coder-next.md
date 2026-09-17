---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: Qwen3-Coder-Next Technical Report, 2026-03-03 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2603.00729
created_at: "2026-09-15"
---

# Excerpt: Qwen3-Coder-Next Technical Report

- **Authors:** Qwen Team (Alibaba)
- **Year:** 2026 (report dated 2026-03-03)
- **Source type:** official technical report
- **Used in:** [[read]] §1, §2, §3, §4, §5, §7, Recipe, Generalization lens

## Scale (§6)
- "a hybrid mixture-of-experts architecture with 80 billion total parameters and only 3 billion active parameters per forward pass", built on the Qwen3-Next base model (§3).

## Task synthesis and infrastructure (§2)
- Two pipelines: mining issue-related GitHub PRs into Docker environments built by "a specialized environment-building agent" with a verification script, and synthesizing new instances inside existing executable datasets (SWE-Smith, SWE-Flow, SWE-Rebench, Multi-SWE-RL).
- Bug injection via "model-driven rewriting, semantic perturbations, and rule-based transformations"; a bug is kept only when it fails existing tests and is resolved by patch reversion. Issue text is generated in natural language and bug-triggering test files are excluded "To mitigate shortcut learning".
- "This process yields approximately 800K verifiable software engineering task instances spanning over nine programming languages."
- Verifier hygiene: "automated detection to identify and filter non-functional verifiers", a trained environment-construction model, and a quality-assurance agent that removes "ambiguous tasks, inconsistent environments, and misaligned tests".
- MegaFlow orchestration on Alibaba Cloud Kubernetes; each task is an Argo workflow with rollout, evaluation and post-processing stages, the agent container co-located with the environment container.

## Mid-training principle and data (§3.1)
- Stated principle: "heavy reliance on synthetic data can significantly improve performance on targeted tasks, but may lead to over-specialization, reduced response diversity, and weaker adaptation to other tasks during fine-tuning. Therefore, our goal is to introduce the minimum amount of synthetic data required for the model to reliably perform common user tasks, while preserving response diversity and maintaining strong general-purpose capabilities."
- Natural data: GitHub language coverage raised "from 92 to 370 languages"; training context expanded "from 32,768 tokens to 262,144 tokens"; "repository-level data is expanded to approximately 600B tokens, representing a major portion of the mid-training recipe and proving more impactful than file-level datasets alone".
- Text-code grounding data rewritten by Qwen3-Coder-480B-A35B-Instruct into clean Markdown. Table 1 (mid-training ablation):
```
Model      Evalplus   MultiplE   CRUX-Eval
Baseline    54.38      36.02      57.13
Reformat    63.09      48.35      58.94
```
- Synthetic multi-turn agentic trajectories generated with Qwen3-Coder-480B-A35B-Instruct across SWE-agent, Mini-SWE-agent, OpenHands, Claude-Code, Qwen-Code and Terminus, then rule-filtered for missing termination signals, task failures and malformed tool calls.
- A small amount of instruction-following data is mixed in "to enable early monitoring of downstream task performance during mid-training".
- Training: "trillions of tokens", context 262,144, next-token prediction plus FIM objectives, best-fit packing, and masking of "highly repetitive segments".

## Cross-scaffold transfer (§3.1.2, Figure 3)
- "within the same scaffold, performance consistently improves with increased mid-training tokens"; "cross-scaffold transfer remains limited. Models trained on trajectories from one scaffold do not transfer strongly to others."
- "OpenHands, which is highly specialized for SWE tasks, transfers poorly to SWE-Agent, while transfer in the opposite direction is moderately successful. This highlights a trade-off between framework generality and specialization." Figure 3 x-axis runs from 1b to 8b training tokens; numeric values are not printed in the text.

## SFT (§4.1)
- Three sources: in-house proprietary corpora, execution-verified agentic trajectories, documentation-grounded open-domain QA.
- Verification: a Mini-SWE-agent instance acts as a user simulator that executes proposed code or commands and judges from compiler output, runtime errors and environment state whether the response advances the task.
- Preference modeling: n candidates per request from in-house models, all (n choose 2) pairs scored by "a dedicated pairwise judging model trained to score responses against a multi-dimensional checklist, including factual accuracy, task usefulness, and conversational style".

## Expert models and distillation (§4.2)
- Four experts, all from the same initial model: Web Development (Playwright-rendered VLM checklist plus dynamic interaction checks), User Experience (multi-scaffold CLI/IDE tool-call format adherence), Single-turn QA, Software Engineering.
- SWE expert: "SFT and RL prompts are fully disjoint" to prevent leakage; instances filtered by pass-rate so RL focuses on "informative failures".
- Reward shaping: trajectory-level reward on final completion, plus an "unfinished trajectory penalty" when turns exceed a maximum, plus a "turn-level tool-format penalty" where "tokens associated with invalid tool calls receive token-level penalties".
- Reward-hacking blocker: removing remotes, branches and tags is not enough; a heuristic blocks "Any tool call containing both a repository link (e.g., github.com/{repo}) and network-access keywords (e.g., git, curl, wget)", returning explicit feedback to the agent.
- §4.2.5: "we perform expert distillation to consolidate capabilities from multiple domain experts into a single unified deployment model", distilling Web Development, User Experience, Single-turn RL and Software Engineering experts into the SFT model, so that "a single model must handle diverse tasks spanning multiple domains without relying on expert routing or multi-model orchestration".

## Evaluation (§5.1, §5.3)
- SWE-Bench Verified, maximum 300 agent turns, all baselines replicated per scaffold with hacking-free protections. Qwen3-Coder-Next (80A3): 70.6 (SWE-Agent), 71.1 (MiniSWE-Agent), 71.3 (OpenHands). SWE-Bench Multilingual: 62.8 / 56.2 / 64.3. SWE-Bench-Pro: 42.7 (SWE-Agent), 38.7 (MiniSWE-Agent).
- General tasks (§5.3): "Qwen3-Coder-Next is competitive with Qwen-Next, slightly improving on MMLU-Redux and GPQA, while remaining close on MMLU, MMLU-Pro, and SuperGPQA." On competitive math it "substantially outperforms the baseline across all benchmarks, with large gains on HMMT25 Feb and AIME25"; the authors read this as code reasoning transferring to math reasoning. Per-benchmark values are in Tables 8 and 9.

## Verification
- Checked on 2026-09-15 against the Qwen3-Coder-Next technical report (2026-03-03): §2.1, §2.2, §3.1, §3.2, §4.1, §4.2.1-§4.2.5, §5.1, §5.3, §6.
- Not reported by the source: mid-training token total beyond "trillions"; mixture percentages; SFT dataset size; RL algorithm name, learning rate, group size, clip values, KL coefficient, step count; numeric values behind Figure 3; GPU hours.
