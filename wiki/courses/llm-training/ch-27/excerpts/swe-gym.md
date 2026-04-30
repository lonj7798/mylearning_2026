---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/swe-gym.md
source_url: https://arxiv.org/abs/2412.21139
created_at: "2026-04-23"
---

# Excerpt: SWE-Gym — 2,438 executable tasks and the RS-SFT recipe

**Source library:** `wiki/raw-data/llm-training/papers/swe-gym.md`
**Paper:** Pan et al. 2024 (UC Berkeley + CMU + Apple), "SWE-Gym: An Open Environment for Training Software Engineering Agents and Verifiers."

---

## Why this source anchors ch-27 §2.2

Ch-27 §2.2's thesis — "invest in the environment first" — is SWE-Gym's entire design statement. The paper's core contribution is not an algorithm; it's a **harness**. 2,438 Docker images with hidden tests. Everything else (rejection sampling, verifier best-of-N) is downstream.

## The environment construction

From the source (lines 26-28):

> - **Source:** 11 popular Python repositories (e.g., astropy, sympy, django, matplotlib, …).
> - **Task extraction:** for each repo, mine merged PRs with linked issues that include `test_` additions/modifications. Result: 2,438 tasks.
> - **Docker packaging:** each task = image with repo at pre-PR commit + dependencies installed + the PR's `test_*` files applied (so tests exist but code doesn't satisfy them yet).

The Docker-packaging trick is crucial. By applying the PR's test files on top of the pre-PR commit, you get a broken repo where the *tests* encode the desired behavior and the *code* is missing the fix. An agent that produces the correct patch makes tests pass; anything else fails.

This is a subtle but important design. Naive alternatives either don't apply the tests (so the agent has no specification to work against) or apply both tests and the reference fix (so the "task" is trivial). SWE-Gym's version splits the difference: specification yes, reference fix no.

## The RS-SFT pipeline

From the source (lines 31-34):

> - **Step 1 — Agent rollouts:** run the **OpenHands** agent scaffold with a teacher model (Qwen-2.5-Coder-32B or Claude) on each SWE-Gym task; capture up to K=10 trajectories per task.
> - **Step 2 — Execution labeling:** run hidden tests; label each trajectory success/failure.
> - **Step 3 — Filter:** keep only trajectories where all hidden tests pass.
> - **Step 4 — SFT:** fine-tune Qwen-2.5-Coder-7B/32B on the filtered successful trajectories.

Four steps. Nothing clever algorithmically — it's rejection sampling with execution-grounded labels. The cleverness is in step 1's *harness* making step 2 cheap.

## The headline numbers

From the source (lines 56-58):

> - Qwen-2.5-Coder-7B-Instruct: SWE-Bench Verified 3.0% → **15.3% (+12.3)** after rejection-sampling SFT on SWE-Gym trajectories.
> - With verifier best-of-N: **20.3%**.
> - 32B version: **32.0% SWE-Bench Verified** — open SOTA at release (Dec 2024).

These numbers are the single strongest empirical case for execution-grounded rejection-sampling SFT in the 2024-2025 literature. 5× improvement at 7B from 3.0% to 15.3%, a further +5 points from the verifier. 32B reaches 32.0%, matching or beating much larger closed models at release.

## The verifier

From the source (lines 37-40):

> ### Verifier training + inference-time best-of-N
> - Train a separate verifier model on (trajectory, success) pairs from SWE-Gym.
> - At eval time, sample K trajectories, score with verifier, pick best.

Two models: policy (the SFT agent) and verifier (the ranker). The verifier is trained on both successful and failed trajectories — the failure trajectories are what make it useful, because they teach the ranker to discriminate.

This is the same "verifier gives +5 pts" result that recurs across reasoning-RL papers (PRM / best-of-N / Let's-verify-step-by-step). SWE-Gym's contribution is showing it transfers to agent trajectories with execution-labeled data.

## The action space

From the source (lines 45-49):

> - **Action space (OpenHands/SWE-agent scaffold):**
>   - `str_replace_editor` (file view/edit).
>   - `execute_bash` (run shell commands including pytest).
>   - `browse` (file system navigation).
>   - `finish` (submit final patch).

Four actions. Simpler than WebArena's 10, simpler than Kimi-K2's 20K+. The OpenHands scaffold intentionally keeps the vocab tight — file-edit, shell, browse, finish — and lets composition happen inside bash commands.

## What to take from SWE-Gym for ch-27

1. **Environment first.** 2,438 Docker images is the paper's real contribution; the algorithm is standard RS-SFT.
2. **Tests-on-pre-fix-code is the task definition trick.** Apply the PR's tests but not its code; the agent fills the gap.
3. **Execution-labeled RS-SFT gives 5× improvement at 7B.** 3.0% → 15.3% on SWE-Bench Verified.
4. **Verifier best-of-N adds +5 pts.** Train the ranker on successful + failed trajectories.
5. **Python-only is a real ceiling.** Multi-language SWE data is the 2026 frontier.

## Connections

- [[ch-27]] §2.2 — SWE-Gym is the environment-grounded SWE corpus.
- [[excerpts/swe-rl]] — the complementary no-environment RL recipe; SWE-Gym does multi-turn with execution, SWE-RL does single-turn with similarity.
- [[excerpts/openhands-data]] — the scaffold SWE-Gym uses. Both papers are consumers of the same action-space design.
