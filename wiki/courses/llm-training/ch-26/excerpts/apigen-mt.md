---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/apigen-mt.md
source_url: https://arxiv.org/abs/2504.03601
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against arXiv v4 and the verified card)"
---

# Excerpt: APIGen-MT — validated task blueprints, then simulated conversations

**Source library:** `wiki/raw-data/llm-training/papers/apigen-mt.md` (verified 2026-09-14)
**Paper:** Prabhakar, Liu, Zhu, Zhang, Awalgaonkar, Wang et al., arXiv:2504.03601 (v1 2025-04; v4 2025-07).

## Phase 1: task blueprint (§4.1)

- Context: each τ-bench domain's APIs form a directed dependency graph; tasks start from random walks plus API, policy, domain-data, persona (PersonaHub), and example samplers (§4.1.1).
- The generator outputs a thought, an instruction `q`, ground-truth actions `a_gt`, and expected outputs `o_gt`.
- Validation (§4.1.2):
  1. Action validation: format check; execution in the environment with the state change recorded as a diff patch; policy compliance through domain policies written as Python unit tests.
  2. Alignment validation: a committee of LLM judges scores Correctness, Completeness, Satisfaction, Creativity (0/1 each), aggregated by majority vote.
  3. Final review: tasks above a score threshold are accepted; failures receive a summarized improvement plan and are regenerated. The threshold value is not reported.
- Reflection budget: at most 3 turns (retail) and 5 (airline) (§4.3).
- Reverse task recombination: tasks with the same persona are concatenated, the policy check is rerun, a combined instruction is written, and the result is revalidated from Stage 2 (§4.1.3).

## Phase 2: simulated interplay (§3.2.2, §4.2)

- A persona-guided human LM receives `q` and reveals details over turns. "The simulated user is unaware of the underlying environment and available APIs" (§3.2.2).
- The agent is gpt-4o in function-calling mode. A trajectory is kept only if the final state matches `a_gt` and responses match `o_gt`; up to 3 attempts per task; all unique successes kept.
- Best-of-N (N = 4) sampling with self-critique stabilizes the simulated user (Table 3: gpt-4o retail 62.8 → 67.0, variance 11.1 → 2.6).
- Failed trajectories are discarded; the authors name them as possible "additional contrastive signal" for future work (§6).

## Statistics and results

- Phase 1 success 70% with agentic feedback vs 28% without; Phase 2 trajectory success 67%; average 7 tool calls and 6 user turns per trajectory; 1 to 29 turns (Fig. 4). The text also states that gpt-4o takes an average of 12 turns per task (§4.3).
- APIs: 15 read and 13 write across retail and airline; gpt-4o and DeepSeek V3 for generation, validation, and interplay (§4.3).
- BFCL v3 (leaderboard 04/03/2025, Table 1): xLAM-2-70b-fc-r overall 78.19 (rank 1); xLAM-2-8b-fc-r 72.83 (rank 4), multi-turn 69.25.
- τ-bench pass@1 (Table 2, retail / airline / overall): xLAM-2-70b 67.1 / 45.2 / 56.2; xLAM-2-8b 58.2 / 35.2 / 46.7; gpt-4o-2024-11-20 62.8 / 43.0 / 52.9.
- Fig. 6 plots pass^k curves; on airline xLAM-2-70b has a higher pass^5 than Claude 3.5 Sonnet (new) despite lower pass^1 (§5.3). Per-k values are not printed in the text.

## Training (§5.1)

Filtered behavioral cloning; trajectories split at every assistant response; loss on assistant tokens only; data = APIGen-MT trajectories + APIGen function-calling data + other agentic data (proportions not reported); at most 3 epochs; full fine-tuning with LLaMA-Factory. LR, batch, and sequence length are not reported. No DPO stage is described.

## Generality notes

Evaluation covers BFCL v3 and τ-bench only. The τ-bench evaluation uses the same two domains, APIs, and policies used for data generation; overlap checks against τ-bench test tasks are not reported.

## Connections

[[apigen]], [[xlam]], [[bfcl]], [[tau-bench]].
