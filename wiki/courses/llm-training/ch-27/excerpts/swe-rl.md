---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/swe-rl.md
source_url: https://arxiv.org/abs/2502.18449
created_at: "2026-04-23"
---

# Excerpt: SWE-RL — difflib.ratio() as a scalable agentic reward

**Source library:** `wiki/raw-data/llm-training/papers/swe-rl.md`
**Paper:** Wei et al. 2025 (Meta FAIR + UIUC + CMU), "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution."

---

## Why this source anchors ch-27 §3

Ch-27 §3 argues that **rule-based dense rewards scale; execution-based sparse rewards do not (at training time, for RL).** SWE-RL is the existence proof — 41.0% SWE-Bench Verified with a one-line reward function and no test execution during training.

## The reward function

From the source (line 37):

> - **Reward:** `r = difflib.SequenceMatcher(None, predicted_patch, ground_truth_patch).ratio()` — a scalar in [0, 1].

That's the entire reward. `difflib.SequenceMatcher.ratio()` is a Python-stdlib function that computes a similarity ratio via Ratcliff-Obershelp (matching blocks / total length × 2). It's dense — every generated patch gets a real-valued score, not a binary label.

The design decision behind this is the paper's most important claim: **dense signal beats sparse signal under GRPO even when the dense signal is approximate**. Execution rewards (test suite pass/fail) are binary and sparse — most samples in a rollout group score 0. Similarity rewards are continuous and dense — every sample contributes gradient. With GRPO's group-relative advantage estimator, dense reward is a structural win.

## The data scale

From the source (lines 27-33):

> ### Data collection
> - **Seed input:** GitHub Archive BigQuery export of all issues and PRs with code changes from top-starred repositories.
> - **Pair extraction:** for each merged PR, extract (issue_description, code_context, ground_truth_patch).
> - **Filters:**
>   - PR merged and has linked issue.
>   - Patch modifies ≤ 10 files and ≤ 500 lines (tractable for 7B context window).
>   - Language is Python (primary) or other popular languages.
>   - Dedup near-identical issues with MinHash.
> - **Output:** ~11M (issue, context, patch) triples.

11 million triples, all auto-mined from public GitHub. No teacher, no API, no execution — just scraping and MinHash dedup. This is what makes SWE-RL cheaper than SWE-Gym's Docker-based approach despite running on 10× more samples.

## The RL algorithm

From the source (lines 36-39):

> - **Algorithm:** **GRPO** with group size G=8, KL coefficient β=0.02, LR 1e-6.
> - **No execution:** unlike SWE-Gym / SWE-agent setups, the training loop does not run unit tests — this is a compute-saving design choice.

GRPO with typical hyperparameters (G=8, β=0.02, LR 1e-6). The novelty isn't the algorithm; it's what goes into the reward and what doesn't. Dropping execution means no Docker-orchestration overhead during rollouts — the reward is a pure function call on the generated string.

## The headline numbers

From the source (lines 58-64):

> - **Llama3-SWE-RL-70B: 41.0% SWE-Bench Verified** (beats DeepSeek-Coder-V2-Instruct 18.0%, matches SWE-Gym-32B).
> - **Out-of-domain transfer** — Llama3-SWE-RL-70B vs Llama-3.1-70B-Instruct baseline:
>   - HumanEval+: +6 points.
>   - MATH: +4 points.
>   - BIG-Bench Hard: +3 points.

The SWE-Bench Verified number is the headline, but the out-of-domain transfer is the more interesting result. Training only on SWE tasks pushes MATH by +4 and BBH by +3 — benchmarks with no software-engineering content. The authors hypothesize "long-horizon grounded planning" is the transferable skill.

Ch-27 §3 calls this "provocative." It is. If true at scale, it means agentic RL is a general reasoning booster, not just a domain-specific training technique.

## The gaming risk

From the source (line 67):

> - **Similarity reward can be gamed:** a predicted patch that copies context verbatim gets partial credit without fixing anything. Authors mitigate with format filters (must be diff, must modify code).

A patch that copies the surrounding context verbatim will have high `SequenceMatcher.ratio()` against the ground-truth patch — *if* the ground-truth patch preserves most of the context. Mitigation: require the output be a unified diff (format filter) and that it actually modify code (no-op filter). Without these, the RL loop finds the copy-context shortcut.

This is a general cautionary tale for rule-based rewards: **every similarity metric has a degenerate-optimum, and you must filter it out at the format level before RL starts**.

## The single-turn limitation

From the source (line 69):

> - **Single-turn limitation:** SWE-RL does not train multi-turn agent behavior (file navigation, test running). Paper notes this as future work.

SWE-RL is issue → patch in one shot. No `str_replace_editor.view`, no `execute_bash`, no iterative debugging. This is why ch-27 §3 pairs SWE-RL with SWE-Gym — the former does cheap dense-signal RL, the latter does environment-grounded multi-turn. Different stages of a complete pipeline, not competing recipes.

## What to take from SWE-RL for ch-27

1. **One-line reward can scale.** `difflib.SequenceMatcher.ratio()` is the whole reward.
2. **Dense beats sparse under GRPO.** Every sample contributes gradient; this compounds at 11M-sample scale.
3. **Out-of-domain transfer from SWE is real.** +4 MATH from pure SWE training.
4. **Single-turn is a real limit.** Multi-turn RL on executable envs is a separate problem.
5. **Filter format before RL.** Every similarity metric has a degenerate optimum.

## Connections

- [[ch-27]] §3 — SWE-RL is the §3 existence proof for rule-based-reward scaling.
- [[excerpts/swe-gym]] — the complementary multi-turn recipe.
- [[excerpts/kimi-k2-agentic-data]] — K2 combines rule-based RLVR with self-critique; SWE-RL is pure rule-based.
