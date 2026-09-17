---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: GLM-5 Team (Zhipu AI and Tsinghua University), "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763v2 (2026-02-24)
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
source_type: official technical report
note: "No library card exists for this source on 2026-09-15 (planned slug glm-5). This extract covers the agentic environment and RL passages used by ch-29c; ch-32d/excerpts/glm-5.md covers pre-training and mid-training."
---

# Excerpt: GLM-5 verifiable agent environments (§3.1, §3.3, §4.1-4.2, §6.2)

Model size: "a 744B parameter model (40B active" (§2.1); Table 10 lists 744B total and 40B activated parameters.

## Scale statement (§3.3)
> "To scaling agentic environments, we scale verifiable training environments across three domains: over 10K real-world Software Engineering (SWE), terminal tasks, and high-difficulty multi-hop search tasks."

## SWE environments (§4.2.1)
- Issue-PR pairs filtered with rule-based and LLM-based filters, categorized as bug fixing, feature implementation, refactoring, and others.
- Environment setup based on the RepoLaunch framework: analyze installation and dependencies, build the environment, generate test commands, then have an LLM write language-aware log-parsing functions that extract Fail-to-Pass (F2P) and Pass-to-Pass (P2P) tests.
- > "Using this pipeline, we construct over 10k verifiable environments across thousands of repositories spanning 9 programming languages, including Python, Java, Go, C, CPP, JavaScript, TypeScript, PHP, and Ruby."

## Terminal environments (§4.2.2)
- From seed data: LLM-brainstormed task drafts → a construction agent writes Harbor-format tasks (description, Dockerized environment, test scripts) → a refine agent checks them against manual rubrics that Docker images build, tests match the specification, and "the environments are robust against potential exploits or shortcuts". > "the pipeline yields thousands of diverse and verifiable terminal-agent environments with Docker construction accuracy exceeding 90%."
- From web corpus: a quality classifier keeps code-relevant pages; stratified sampling over topic and difficulty; a coding agent writes a task from a page and runs the Harbor validation script on its own output, revising until all automated checks pass.

## Search tasks (§4.2.3)
- A Web Knowledge Graph built from over two million high-information pages collected from an early search agent's trajectories; low- to mid-frequency entities are seed nodes; subgraphs become multi-hop questions.
- Three-stage filter: > "(1) Remove questions that a tool-free reasoning model correctly answers in at least one of eight independent attempts. (2) Filter out questions solvable by an early-stage agent with basic search, browsing, and computation within a few steps. (3) Apply a verification agent for bidirectional validation ... rejecting samples with non-unique answers, inconsistent evidence, or incorrect labels."

## RL objective and sample handling (§4.1, §4.1.2)
- Group objective over K traces with mean-reward baseline r̄(x) = (1/K) Σ r(x, y_i); "only model-generated tokens are used for optimization, and the environment feedback is ignored in loss computation."
- Stale samples: a sample is discarded if w′ − w_0 > τ (current policy version minus the oldest rollout version); τ is not printed.
- Environment failures: > "we record the failure reason for each sample and exclude samples that fail due to environment collapse. For group-based sampling methods such as GRPO, removing failed samples can leave an incomplete group. In that case, we pad the group by repeating valid samples if the number of valid samples exceeds half of the group size; otherwise, we drop the entire group."

## SFT data for agents (§3.1)
> "Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions."

## Evaluation on fresh tasks (§6.2.4, Table 9; Table 7)
- The report evaluates on SWE-rebench because SWE-bench Verified "is a static, public, human-validated test set and released for more than 2 years".
- Table 9 (SWE-rebench, January 2026), resolved % / pass@5 %: Claude Opus 4.6 52.9 / 70.8; GPT-5.2 (xhigh) 51.7 / 58.3; Claude Sonnet 4.5 47.1 / 60.4; Gemini 3 Pro 46.7 / 58.3; Claude Opus 4.5 43.8 / 58.3; GLM-5 42.1 / 50.0; GLM-4.7 41.3 / 56.3; Kimi K2.5 37.9 / 50.0.
- Table 7, SWE-bench Verified: GLM-5 77.8, GLM-4.7 73.8, DeepSeek-V3.2 73.1, Kimi K2.5 76.8, Claude Opus 4.5 80.9, Gemini 3 Pro 76.2, GPT-5.2 (xhigh) 80.0.

## Not reported
Number of terminal and search tasks; stale threshold τ; share of samples dropped for environment collapse; overlap checks between synthesized environments and SWE-bench, Terminal-Bench, or BrowseComp.
