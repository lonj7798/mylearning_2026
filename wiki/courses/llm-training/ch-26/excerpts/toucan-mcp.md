---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2510.01179
created_at: "2026-09-15"
---

# Excerpt: Toucan — 1.5M tool-agentic trajectories from real MCP servers

**Paper:** Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, Rameswar Panda, "TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments", arXiv:2510.01179 v1 (2025-10-01). Dataset: hf.co/datasets/Agent-Ark/Toucan-1.5M. Read 2026-09-15.

## Pipeline (§3.1–§3.2)

1. MCP server onboarding: about 2,800 servers crawled from GitHub and Smithery; kept remote servers reachable over streamable HTTP and removed servers with problematic tools, leaving 495 servers (§3.1, Fig. 1). The abstract says "nearly 500" MCP servers and §1 says the environments have "more than 2,000 tools".
2. Task synthesis with five LLMs from single servers, several servers, and 25 featured servers.
3. Task filtering: Kimi-K2 rates six dimensions on a 1–5 scale (tool-selection difficulty, tool-selection uniqueness, question quality, scenario realism, verifiability, stability).
4. Trajectory generation: GPT-OSS-120B, Kimi-K2, and Qwen3-32B with Qwen-agent and OpenAI-agent frameworks, calling the real servers.
5. Post-filtering: rules remove failed connections, trajectories with no tool calls, failed tool responses, and local file paths; required-tool coverage and order are measured; GPT-OSS-120B rates completeness and conciseness.
- Extensions: Ext.1 irrelevance (server metadata shuffled so tasks are unsolvable with the given tools; only trajectories with zero tool calls kept); Ext.2 persona-based diversification; Ext.3 multi-turn by splitting tasks or adding follow-up queries.

## SFT subset and settings (§4.1, App. C.2 Table 5)

- 119.3K instances: 28.3K core, 40K irrelevance, 15.8K diversify, 35.2K multi-turn; selected with question quality and realism 5, completeness and conciseness ≥ 4, and required-tool coverage 1.0.
- Hermes tool-call template; LR 2e-5; 2 epochs; effective batch 64; AdamW (0.9, 0.999), ε 1e-8; ZeRO-3; max sequence length 32,768.

## Results

- BFCL V3 overall (Table 2): Qwen2.5-7B-Instruct 55.10 → 58.26; 14B 57.69 → 65.09; 32B 61.73 → 70.45. For 7B, non-live AST 84.19 → 78.52 and relevance 72.22 → 66.67 decrease while multi-turn rises 12.88 → 22.62.
- τ-bench / τ²-bench averages (Table 3, GPT-4o user simulator): 7B 15.03 → 22.48 and 16.08 → 17.77; telecom for 7B 16.70 → 10.50.
- Extension ablation on Qwen2.5-14B-Instruct (Fig. 9; BFCL v3, τ-bench airline, retail): base 57.69 / 17.25 / 44.46; + single-turn core 60.16 / 15.50 / 36.95; + irrelevance 64.74 / 16.75 / 41.63; + diversify 64.56 / 17.25 / 43.70; + multi-turn 65.09 / 22.00 / 48.48.
- MCP-Universe: tuned models are above similar-size open models in most domains (Figs. 7–8; values shown only in figures). Most MCP-Universe servers need configuration and were not in the synthesis pipeline (§4.2).

## Not reported

Contamination or overlap checks against BFCL, τ-bench, or MCP-Universe; general-capability benchmarks; seeds or variance. Data were collected in June 2025 and exclude servers that need API keys (§5).
