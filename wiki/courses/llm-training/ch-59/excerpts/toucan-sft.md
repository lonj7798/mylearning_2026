<!-- excerpt for [[ch-59]] — Toucan SFT setup and the per-slice deltas it produces
     source: TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments, arXiv:2510.01179v1
     read 2026-09-17 from the cached primary text. No library card exists for this artifact yet.
-->

# Toucan — tool-agentic SFT setup and per-slice results

## Data (Abstract, §3)

- 1.5 million trajectories synthesized from "nearly 500 real-world Model Context Protocols (MCPs)"; the
  filtered pool is "a refined set of 495 high-quality MCP servers" with "more than 2,000 tools" (§3).
- The SFT subset used for the experiments is selected on judged scores (question quality and scenario realism
  5; response completeness and conciseness at least 4; desired tool-use percentage 1.0) and totals 119.3K
  instances: "28.3K instances from the original pipeline, 40K instances from Ext.1 (Irrelevance), 15.8K
  instances from Ext.2 (Diversify), and 35.2K instances from Ext.3 (Multi-Turn)" (§4).

## Table 5 (App. C.2): supervised fine-tuning hyper-parameters

| Hyper-parameter | Value |
|---|---|
| Tool-call template | Hermes |
| Learning rate | 2 × 10⁻⁵ |
| Number of epochs | 2 |
| Number of devices | 8 or 64 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 8 (8 GPUs) or 1 (64 GPUs) |
| Effective batch size | 64 |
| Optimizer | AdamW with βs = (0.9, 0.999) and ε = 10⁻⁸ |
| DeepSpeed | zero3 |
| Max sequence length | 32768 |

## Table 2 (§4): BFCL V3, base against the same base fine-tuned on Toucan

| Model | Overall | Non-live (AST) | Live (AST) | Multi Turn | Relevance | Irrelevance |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 55.10% | 84.19% | 72.32% | 12.88% | 72.22% | 67.93% |
| + Toucan | 58.26% | 78.52% | 74.50% | 22.62% | 66.67% | 75.18% |
| Qwen2.5-14B-Instruct | 57.69% | 83.38% | 73.70% | 19.75% | 83.33% | 68.46% |
| + Toucan | 65.09% | 85.42% | 76.01% | 35.25% | 72.22% | 75.96% |
| Qwen2.5-32B-Instruct | 61.73% | 85.58% | 76.01% | 26.38% | 72.22% | 72.68% |
| + Toucan | 70.45% | 87.12% | 78.90% | 46.50% | 77.78% | 78.10% |

At 7B the overall score rises 3.16 points while the non-live AST slice falls 5.67 points (84.19 → 78.52) and
the relevance slice falls 5.55 points (72.22 → 66.67). At 14B the relevance slice falls 11.11 points
(83.33 → 72.22) while overall rises 7.40. Only at 32B does every column in this table move upward.

## Table 3 (§4): τ-bench and τ²-bench

| Model | τ-bench Avg. | τ²-bench Avg. | τ² Telecom |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 15.03% | 16.08% | 16.70% |
| + Toucan | 22.48% | 17.77% | 10.50% |
| Qwen2.5-14B-Instruct | 30.85% | 24.46% | 20.18% |
| + Toucan | 35.24% | 30.43% | 20.18% |
| Qwen2.5-32B-Instruct | 38.76% | 29.40% | 21.11% |
| + Toucan | 42.33% | 31.60% | 20.20% |

The τ² Telecom column falls at 7B (16.70% → 10.50%) and at 32B (21.11% → 20.20%) while both averages rise.
All evaluations were run on an 8 × H100 server with the official BFCL-V3 setup and GPT-4o as the user
simulator for τ-bench and τ²-bench (§4).
