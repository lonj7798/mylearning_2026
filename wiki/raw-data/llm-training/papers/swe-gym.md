<!-- scope: agentic trajectory synthesis — executable SWE environments + trajectory generation
     deps: [[agentinstruct]]
     see-also: [[swe-rl]], [[openhands-data]]
-->

# SWE-Gym: An Open Environment for Training Software Engineering Agents and Verifiers
- **Core Invariant:** Training SWE agents requires **executable** environments (Docker sandboxes with real test suites), not just text-level patch supervision; SWE-Gym provides 2,438 real Python tasks each with a pre-installed environment + unit tests, enabling trajectory-level RL and rejection-sampling SFT with ground-truth execution reward.
- **Guideline:** For SWE agent training, invest in the **environment** side first — build a Docker-based task harness where every training example can execute its tests in seconds; this unlocks rejection-sampling + RL pipelines that pure text data cannot.
- **Authors:** Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang (UC Berkeley + CMU + Apple)
- **Year:** 2024 (Dec arXiv), 2025 workshop / conference
- **URL:** https://arxiv.org/abs/2412.21139
- **Relevant topics:** SWE agents, executable environments, trajectory RL, rejection sampling, SWE-Bench

## Abstract
SWE-Gym releases 2,438 Python software-engineering tasks — real GitHub issues from 11 popular repos — each packaged as a Docker image with the correct dependency set and a hidden unit-test suite. With SWE-Gym, the authors demonstrate: (a) rejection-sampling SFT using agent trajectories lifts Qwen-2.5-Coder-Instruct 7B from 3.0% → 15.3% on SWE-Bench Verified; (b) training a verifier on successful vs failed trajectories and using it for best-of-N at inference lifts to 20.3%; (c) the same pipeline scaled to 32B reaches 32.0% — a new open SOTA at release.

## Key Contributions
- **2,438 executable SWE tasks** across 11 Python repos, each with isolated Docker env + tests.
- **Trajectory rejection-sampling SFT** recipe — concrete pipeline from task to trained agent.
- **Verifier training** — learns to rank trajectories by predicted success from execution-labeled data.
- **Open environment registry** — 491 tasks immediately compatible with SWE-Bench Lite.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Environment construction
- **Source:** 11 popular Python repositories (e.g., astropy, sympy, django, matplotlib, …).
- **Task extraction:** for each repo, mine merged PRs with linked issues that include `test_` additions/modifications. Result: 2,438 tasks.
- **Docker packaging:** each task = image with repo at pre-PR commit + dependencies installed + the PR's `test_*` files applied (so tests exist but code doesn't satisfy them yet). Image also contains: issue text, file index, hidden test command.

### Trajectory collection (rejection-sampling SFT)
- **Step 1 — Agent rollouts:** run the **OpenHands** agent scaffold (successor to SWE-agent) with a teacher model (Qwen-2.5-Coder-32B or Claude) on each SWE-Gym task; capture up to K=10 trajectories per task.
- **Step 2 — Execution labeling:** run hidden tests; label each trajectory success/failure.
- **Step 3 — Filter:** keep only trajectories where all hidden tests pass.
- **Step 4 — SFT:** fine-tune Qwen-2.5-Coder-7B/32B on the filtered successful trajectories.
- **Output shape:** ~20K–60K successful trajectories; each trajectory is a multi-turn (agent, env) conversation with file-read / file-edit / bash / test-run actions. Avg trajectory length ~15K tokens, up to 100K.

### Verifier training + inference-time best-of-N
- Train a separate verifier model on (trajectory, success) pairs from SWE-Gym.
- At eval time, sample K trajectories, score with verifier, pick best.
- **Teacher model(s):** Qwen-2.5-Coder-32B-Instruct (primary), Claude-3.5-Sonnet (secondary ablation).
- **Cost / compute:** rollout cost dominates — ~10K H100-hours for full 32B RS-SFT loop.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** Docker sandboxes with real Python test suites.
- **Action space (OpenHands/SWE-agent scaffold):**
  - `str_replace_editor` (file view/edit).
  - `execute_bash` (run shell commands including pytest).
  - `browse` (file system navigation).
  - `finish` (submit final patch).
- **Trajectory length:** median ~15K tokens; tail to >100K (long debugging sessions).
- **Success criterion:** all hidden tests pass; training filter requires full-pass not partial.
- **Data scale:** 2,438 task instances; each can generate K trajectories → tens of thousands of training trajectories.
- **Key ablation:** SFT with verifier best-of-N gives +5 points over SFT alone.

## Quality / diversity evaluation
- Qwen-2.5-Coder-7B-Instruct: SWE-Bench Verified 3.0% → **15.3% (+12.3)** after rejection-sampling SFT on SWE-Gym trajectories.
- With verifier best-of-N: **20.3%**.
- 32B version: **32.0% SWE-Bench Verified** — open SOTA at release (Dec 2024); matched only by much larger closed models.
- Scaling: trajectory count and verifier-N both show log-linear returns.

## Risks + gotchas
- **Executable-env maintenance:** Docker images rot as dependencies drift; SWE-Gym must be maintained to stay runnable.
- **Test leakage risk:** training on tasks whose test files overlap with SWE-Bench Verified is guarded by repo exclusion.
- **Language-narrow:** Python only.
- **Cost:** running rollouts for 2.4K × K=10 trajectories requires serious compute and orchestration.

## Connections
- Sibling 2025 SWE recipe: [[swe-rl]] (Meta) — does text-level RL without environments; complementary cost/quality tradeoff.
- Agent scaffold: OpenHands (successor to SWE-agent, see [[openhands-data]]).
- Eval: SWE-Bench / SWE-Bench Verified / SWE-Bench Lite.
- Rejection-sampling SFT ancestry: [[rejection-sampling-finetuning]].
