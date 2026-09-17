---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/toolace.md
source_url: https://arxiv.org/abs/2409.00920
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the ICLR 2025 version on arXiv; the library card has not been verified and its difficulty mix, ablation deltas, and 11,300-dialog count are not in the paper)"
---

# Excerpt: ToolACE — synthesized API pool, model-calibrated complexity, dual-layer verification

**Source library:** `wiki/raw-data/llm-training/papers/toolace.md` (not yet verified; values below were read in the paper on 2026-09-15)
**Paper:** Liu, Huang, Zeng, Hao, Yu, Li et al., "ToolACE: Winning the Points of LLM Function Calling", arXiv:2409.00920, ICLR 2025.

## Tool Self-Evolution Synthesis (§2.1)

1. Speciation: a frontier-LLM agent extracts API domains and functionalities from API-related pretraining documents and builds a hierarchical API context tree.
2. Adaptation: each API receives a sampled subtree of functionalities.
3. Evolution: an LLM writes new API definitions from a subtree and an example from a buffer, with diversity indicators (new functionality or parameters, constraints, mutated parameter types, updated returns).

Result: 26,507 APIs across 390 domains, with nested parameter types (Table 1). The paper does not state a seed-API count.

## Self-guided dialog generation (§2.2)

- Three LLM agents (user, assistant, tool). Four dialog types: single, parallel, dependent function calls, and non-tool-use dialogs (§2.2.1).
- The assistant's action is sampled several times and kept only when decisions agree (§2.2.1). Tool responses are simulated by the tool agent.
- Complexity (Eq. 1): `H_M(x, y) = −(1/n_y) Σ_i log p(t_i | x, t_1..t_{i−1})`, computed with the model M that will be trained. Bounds come from a small prior set: the loss of samples M already gets right is a lower bound, and the loss of samples that stay high after fine-tuning is an upper bound (§2.2.2).
- The user agent is told to write a more complex query (more APIs, larger distance from API descriptions) when a sample is too easy, and a simpler one when too hard (§2.2.3).

## Dual-layer verification (§2.3)

- Rule layer: API definition clarity, function-calling executability, dialog correctness, sample consistency (App. B Table 4). Executability is checked without running calls: the name must be in the tool list, required parameters present, and formats matched by regular expressions.
- Model layer: separate LLM agents for hallucination detection (parameter values not in the query or system prompt), consistency validation, and tool-response checks. Results are "overseen by human experts".

## Results

- BFCL-v3 leaderboard of 09/20/2024 (Table 2): ToolACE-8B (FC) overall 59.22 (rank 3), relevance 85.37, irrelevance 83.81, multi-turn 14.37; xLAM-7b-fc-r overall 51.45 (rank 18), multi-turn 0.00.
- Later analyses use BFCL non-live categories (footnote 3). Fig. 3 (overall): no verification 89.59, rule layer only 90.47, both layers 91.41. Irrelevance: 82.92, 90.00, 89.17.
- Complexity subsets of 60,000 each (Fig. 4, overall): easy 90.47, medium 90.71, hard 89.65. Irrelevance: 90.42, 88.75, 86.25.
- Diversity subsets of about 30,000 from 6, 14, 30 API clusters (Fig. 5, overall): 88.18, 88.35, 88.41.
- Same size, 25,000 samples (Table 6): ToolLLM data overall 24.90, irrelevance 4.41; xLAM data 40.51, irrelevance 11.87; ToolACE data 58.19, irrelevance 86.42.
- Removing data types at 25,000 (Table 7): without non-tool-use ("Multi-type") samples, irrelevance 6.99 and multi-turn 1.75 (full: 86.42, 16.50); without parallel samples, non-live AST 74.75 (full 86.96).
- Evaluator choice (Table 8): the learner as its own evaluator 59.22 overall vs Qwen1.5-7B-Chat 57.61 and Qwen1.5-14B-Chat 57.67.
- General capabilities (Fig. 8): compared on MMLU, HumanEval, GSM8K, CommonSenseQA; the text reports "negligible performance degradation on some benchmarks" vs LLaMA-3.1-8B-Instruct; values are shown only in a radar chart.

## Training (§3.1, App. C.2 Table 5)

LLaMA-3.1-8B-Instruct with LoRA rank 16, alpha 32, all modules; LR 1e-4; warmup ratio 0.1; cosine; batch 48; 3 epochs.

## Not reported

Total dataset size as one number, overall rejection rate, generator model names, seeds, and variance.
