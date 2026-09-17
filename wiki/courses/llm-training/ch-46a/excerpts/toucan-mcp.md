---
chapter: ch-46a
course: llm-training
phase: read
excerpt_of: "TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments, arXiv:2510.01179v1 (2025-10-01)"
source_url: https://arxiv.org/abs/2510.01179
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug toucan-mcp). Values read from the v1 PDF on 2026-09-15; ch-26/excerpts/toucan-mcp.md and ch-29c/excerpts/toucan-mcp.md cover the same paper."
---

# Excerpt: TOUCAN — tasks and trajectories from real MCP servers

**Authors:** Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, Rameswar Panda (University of Washington; MIT-IBM Watson AI Lab).

## Scale (Table 1, §1)
- 1,527,259 trajectories, 567,262 multi-turn; tool responses "Real Executed"; "nearly 500 real-world" MCP servers with "more than 2,000 tools".

## Pipeline (§3.1)
1. **Server onboarding.** About 2,800 MCP servers crawled from GitHub and Smithery. Two filters: remote servers reachable over streamable HTTP, and no third-party credentials such as API keys. > "This process reduced the dataset to 30.6% (871 servers)." Test questions per tool then removed servers whose tools returned errors, leaving 495 servers.
2. **Task synthesis.** Five open models (Mistral-Small, DevStral-Small, GPT-OSS, Kimi-K2, Qwen3-32B) generate tasks from one server (1 to N tools), from several servers (2 to N tools), or from 25 manually selected featured servers; N = 3.
3. **Task filtering.** Kimi-K2 rates each task 1-5 on tool selection difficulty, tool selection uniqueness, question quality, scenario realism, verifiability, and stability.
4. **Trajectory generation.** GPT-OSS-120B, Kimi-K2, and Qwen3-32B with the Qwen-agent and OpenAI-agent frameworks, calling the real servers.
5. **Post-filtering.** Rules remove trajectories that fail to connect, contain no tool calls, contain failed tool responses, or contain local file paths; required-tool coverage and order are measured; GPT-OSS-120B rates completeness and conciseness.

## Extensions (§3.2)
- Ext.1 Irrelevance: server metadata shuffled across instances so tasks are unsolvable with the given tools; only trajectories with zero tool calls kept.
- Ext.2 Persona-based diversification; Ext.3 Multi-turn by splitting tasks or adding follow-up queries.

## SFT subset and hyperparameters (§4.1; App. C.2 Table 5)
- Selection: question quality and scenario realism 5, completeness and conciseness ≥ 4, desired tool-use percentage 1.0. 28.3K core + 40K irrelevance + 15.8K diversify + 35.2K multi-turn = 119.3K instances.
- Hermes template; LR 2e-5; 2 epochs; effective batch 64; AdamW (0.9, 0.999), ε 1e-8; DeepSpeed ZeRO-3; max sequence 32,768.

## Results (Table 2, BFCL V3)
| Model | Overall | Non-live AST | Multi-turn | Relevance | Irrelevance |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 55.10 | 84.19 | 12.88 | 72.22 | 67.93 |
| + TOUCAN | 58.26 | 78.52 | 22.62 | 66.67 | 75.18 |
| Qwen2.5-14B-Instruct | 57.69 | 83.38 | 19.75 | 83.33 | 68.46 |
| + TOUCAN | 65.09 | 85.42 | 35.25 | 72.22 | 75.96 |
| Qwen2.5-32B-Instruct | 61.73 | 85.58 | 26.38 | 72.22 | 72.68 |
| + TOUCAN | 70.45 | 87.12 | 46.50 | 77.78 | 78.10 |

## Not reported
Overlap or contamination checks against BFCL, τ-bench, τ²-bench, or MCP-Universe; general-capability benchmarks. Servers that need API keys are excluded by design (§3.1).

## Used in

ch-46a §2 (agent-data slices and the BFCL component scores after Toucan SFT) and the negative-feedback section (irrelevance trajectories as negative content).
