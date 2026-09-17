---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2506.14965v1 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2506.14965
created_at: "2026-09-15"
---

# Excerpt: Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective (GURU)

- **Authors:** Zhoujun Cheng, Shibo Hao, Tianyang Liu, Fan Zhou, Yutao Xie, Feng Yao, et al. (UC San Diego, MBZUAI, CMU, Purdue)
- **Year:** 2025 (arXiv v1 2025-06-17)
- **Source type:** paper (data, models, code: https://github.com/LLM360/Reasoning360)
- **Used in:** [[read]] §3, Generalization lens, Recipe

## Data (§2)
92K verifiable examples in six domains: Math, Code, Science, Logic, Simulation, Tabular. Binary rewards with three verification types: rule-based matching (Math, Logic, Simulation, Tabular), execution against test suites (Code, reward 1 only if all tests pass), and a 1.5B model-based verifier for Science (the General-Reasoner verifier). Deduplication removes a sample when its question is a strict substring of another: 27.2% of Math and 7.5% of Code samples. Difficulty filtering uses pass rates of Qwen2.5-7B-Instruct and Qwen3-30B-A8B over N = 16 runs per sample.

## Transfer study (§3.1)
3K samples per domain are drawn, giving six single-domain sets and a mixed 18K set (GURU-18K); Qwen2.5-7B-Base and Qwen2.5-32B-Base are RL-trained on each. Online testing on 13 tasks every 10 steps; the checkpoint with the best average is selected. Findings as stated: "Math, Code, and Science consistently achieve stronger performance gains than other domains from cross-domain RL training"; "domains less represented in pretraining, such as Logic, Simulation, and Tabular, show limited improvement under cross-domain training"; easier tasks (MATH500, AMC, HumanEval, MBPP) transfer more than harder ones (AIME24, LiveCodeBench), and the lowest-baseline tasks (ARC-AGI, CodeI/O) show "marginal to negligible gains" from out-of-domain data. Figure 3 holds the per-cell numbers; the PDF text layer does not carry them.

## Difficulty filtering (§3.3, Table 2; Qwen2.5-7B-Base, 200 RL steps, best validation accuracy)
| Training data | MATH500 | AMC | AIME24 | HumanEval | LiveCodeBench | HiTab | Multihiertt |
|---|---|---|---|---|---|---|---|
| Unfiltered math | 75.8 | 52.1 | 15.8 | 82.3 | 11.1 | 56.5 | 32.0 |
| Difficulty-filtered math | 78.6 | 58.4 | 21.7 | 73.1 | 10.7 | 53.5 | 35.5 |
The authors report accuracy collapse on HumanEval and HiTab after roughly 100 and 150 steps of filtered training, and conclude that aggressive difficulty filtering is reasonable when only in-domain performance matters but "introduces a risk of negative transfer to easier tasks in other domains".

## Main models (§4, Table 3)
GRPO in verl from Qwen2.5-7B/32B-Base: LR 1e-6 (AdamW, 10-step linear warm-up), prompt batch 512 per RL step, 16 responses per prompt at temperature 1.0, mini-batch 64 (8 gradient updates per step), 4K prompt / 8K generation tokens, no KL and no entropy loss, clip ε = 0.2, 20 nodes × 8 Hopper GPUs; 3 epochs (7B) and 2 epochs (32B), about 3 days each.
Average over 17 benchmarks: GURU-7B 43.29 vs Open-Reasoner-Zero-7B 35.42, General-Reasoner-7B 33.76, SimpleRL-7B 33.97; GURU-32B 54.24 vs ORZ-32B 47.53, SimpleRL-32B 46.25. IFEval is an exception at 7B: GURU-7B 35.81 vs General-Reasoner-7B 39.56 and SimpleRL-7B 36.69.
Pass@k (§4.3): on AIME the 7B curves of the base and RL model cross at k = 64, while the 32B curves never cross; on the synthetic Zebra Puzzle both GURU models stay above their base models across the sampled range.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2506.14965 (v1, 2025-06-17): Abstract, §1–§4.3, Tables 1–3.
- Inconsistency inside the source: the Abstract states GURU outperforms the best baselines "by 7.9% and 6.7%", §4.2 states "9.0% and 6.7%"; Table 3 gives differences of 7.87 and 6.71 points.
- Not reported: per-cell numbers of Figure 3 in machine-readable form; seeds.
