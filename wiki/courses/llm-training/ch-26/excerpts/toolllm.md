---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/toolllm.md
source_url: https://arxiv.org/abs/2307.16789
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the arXiv PDF; the earlier Table 3 values and the error-only pseudocode did not match the paper)"
---

# Excerpt: ToolLLM — ToolBench, DFSDT, and unseen-tool evaluation splits

**Source library:** `wiki/raw-data/llm-training/papers/toolllm.md`
**Paper:** Qin, Liang, Ye, Zhu, Yan, Lu et al., arXiv:2307.16789 (2023-07).

## Data (§2, Table 1)

- 16,464 REST APIs in 49 RapidAPI categories; 3,451 tools retained (§2.1).
- Instructions and solution paths generated with `gpt-3.5-turbo-16k`; settings I1 (single tool), I2 (intra-category multi-tool), I3 (intra-collection multi-tool).
- 126,486 (instruction, solution path) pairs with 469,585 real API calls (Table 1, §2.3).
- API responses longer than 1,024 tokens are compressed, then truncated (App. A.2).

## DFSDT (§2.3, App. A.4, App. prompts)

- A depth-first search over reasoning and action steps. The model can call `Finish` with `give_up_and_restart`, which abandons the current node; later child expansions receive a "diversity" prompt that shows the previous children.
- Child sorting is skipped and a pre-order traversal is used because the first generated node is usually ranked highest (App. A.4).
- "If the model does not retract an action (e.g., for the case of simple instructions), then DFSDT degrades to ReACT, which makes it as efficient as ReACT" (App. A.4).

## Annotation yield (Table 3, pass rate with ChatGPT)

| Method | I1 | I2 | I3 | Average |
|---|---|---|---|---|
| ReACT | 37.8 | 40.6 | 27.6 | 35.3 |
| ReACT@N (cost-matched) | 49.4 | 49.4 | 34.6 | 44.5 |
| DFSDT | 58.0 | 70.6 | 62.8 | 63.8 |

Only passed annotations are kept as training data (§3.1).

## Generalization splits (§3.2)

- Inst.: unseen instructions for tools seen in training.
- Tool: unseen tools from a seen category.
- Cat.: unseen tools from an unseen category.
- Evaluated: I1-Inst, I1-Tool, I1-Cat, I2-Inst, I2-Cat, I3-Inst, with oracle APIs unless the retriever is used.

ToolLLaMA-2-7B + DFSDT pass / win (Table 4): I1-Inst 57.0 / 55.0; I1-Tool 61.0 / 55.3; I1-Cat 62.0 / 54.5; I2-Inst 77.0 / 68.5; I2-Cat 77.0 / 58.0; I3-Inst 66.0 / 69.0. ChatGPT + DFSDT averages 64.8 / 64.3 and GPT-4 + DFSDT 71.1 / 70.4. Vicuna and Alpaca pass 0 in every setting.

ToolEval agreement with human annotators: 87.1% (pass rate) and 80.3% (win rate) (§3.1).

## Out-of-distribution check (§3.3, Table 5)

On APIBench (not trained on), ToolLLaMA + its retriever gets AST accuracy 16.77 / 51.16 / 40.59 on HuggingFace / TorchHub / TensorHub, vs Gorilla-ZS + BM25 10.51 / 44.62 / 34.31 and Gorilla-RS + BM25 15.71 / 50.00 / 41.90.

## Training (App. A.3)

LLaMA-2 7B, positional interpolation ratio 2 to 8,192 tokens, LR 5e-5, warmup ratio 0.04, total batch 64, 2 epochs, checkpoint chosen on the development set.
