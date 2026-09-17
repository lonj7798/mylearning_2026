---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: arXiv:2502.18449v2 (SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution), NeurIPS 2025
source_url: https://arxiv.org/abs/2502.18449
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the primary source because the library card [[swe-rl]] predates the 2026-09 audits and states a base model, data scale, and transfer numbers the paper does not support)"
---

# Excerpt: SWE-RL, read from the paper

Used by [[read]] §6 and §7 and the Recipe table. Checked on 2026-09-15 against arXiv:2502.18449v2 (1 Dec 2025).

## Reward (§2.1, Eq. 1)
```
R(o) = −1                                   if o has the wrong format
     = compare(patch_pred, patch_gt)        otherwise
```
- `compare` is Python's `difflib.SequenceMatcher`, returning a sequence similarity in [0, 1] between the predicted patch and the oracle patch from the merged pull request.
- The policy is prompted to write its reasoning inside `<think>` tags and its search/replace edits inside `<solution>` tags (Fig. 2); issue solving is the only training subtask.
- The optimizer is GRPO with group-normalized advantages `A_i = (r_i − mean(r)) / std(r)` and a KL term against a reference policy (Eq. 2). The paper does not print ε, β, or the learning rate.

## Data (§2, App. A)
- GitHub events from GHArchive, 1 Jan 2015 to 31 Aug 2024, plus 4.6M cloned repositories; repositories used by SWE-bench are excluded.
- Aggregation gives 24M PR instances; filtering (merged PRs, bot removal, empty or oversized changes, CodeLlama-style hunk filters) leaves about **11M unique PR instances**.
- **273k high-quality PR seeds** are selected from those 11M for RL. Each seed carries the issue text, code context (changed files plus files predicted relevant but unchanged by Llama-3.1-70B-Instruct), and the oracle patch.

## Training configuration (§3.1)
- Policy: **Llama-3.3-70B-Instruct**. 1,600 steps, 16k context window, global batch 512 (16 rollouts from each of 32 problems), one optimizer step per global step, Adam, 512 H100 GPUs, about 32 wall-clock hours.
- SFT baseline (Llama3-SWE-SFT-70B): same base model, Magicoder-style synthetic code-editing data seeded from the same PRs, plus Llama 3 coding and general SFT data.

## Results
- SWE-bench Verified with the Agentless Mini scaffold, 500 repair samples per problem at temperature 1.0 and the top 30 reproduction tests: **41.0%** for Llama3-SWE-RL-70B, 36.2% for the SFT baseline (Table 1). Reference rows in Table 1: GPT-4o with Agentless 38.8, SWE-Gym-32B 32.0, SWE-Fixer-72B 32.8.
- Repair-only with oracle files and greedy decoding (Table 2): Llama-3.3-70B-Instruct 5.4 with 12.2% correct format (16.6 with 20-sample majority voting, 44.6% format), SFT 29.6 at 96.2% format, RL 34.8 at 95.6% format.
- Sample scaling (Fig. 4): 33.6 at 20 repair samples, 40.0 at 160, 41.0 at 500; 38.8 with 1 reproduction test, 41.0 at 20, unchanged at 30.
- Reward ablation (Fig. 5): a discrete reward (1 for exact patch match, else 0) gives 29.0 repair-only accuracy against 34.8 for the continuous similarity reward, with similar format accuracy (94.2% against 95.6%); the average discrete reward stays near zero for the whole run because real patches rarely match exactly.
- Out-of-domain (Table 3), zero-shot greedy decoding, base / SFT / RL: HumanEval+ 76.2 / 73.2 / 79.9; BigCodeBench-Hard (I) 28.4 / 25.7 / 28.4; CRUXEval-I 60.5 / 68.4 / 71.6; CRUXEval-O 61.9 / 75.1 / 75.5; MATH strict 63.2 / 54.0 / 73.7 (lenient 70.9 / 71.7 / 73.7); MMLU 86.49 / 85.26 / 86.82. The SFT model is below the base model on average; the RL model is above it.
- Significance statement (§3.5): with Eval Arena, improvements above 0.8 points on MMLU, 3 points on CRUXEval, and above 3 points on full MATH are significant on their own; the authors argue the aggregate across benchmarks reaches significance at the 0.05 level.

## Stated limitations (§5)
- The reward compares sequence similarity, not semantic equivalence, which may discourage functionally equivalent alternative solutions.
- Agentless Mini splits the pipeline into separate inference stages, so the model does not learn from interaction feedback; training is single-turn patch generation only.
