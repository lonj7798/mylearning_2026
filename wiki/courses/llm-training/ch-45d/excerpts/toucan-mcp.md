---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: arXiv:2510.01179 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2510.01179
created_at: "2026-09-15"
---

# Excerpt: TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments

- **Authors:** Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, Rameswar Panda
- **Year:** 2025 (arXiv v1 2025-10-01)
- **Source type:** paper; dataset released at https://hf.co/datasets/Agent-Ark/Toucan-1.5M (CC BY 4.0)
- **Used in:** [[read]] §3, Recipe

## Server onboarding (§Stage 1)
- "From an initial crawl yielding approximately 2,800 MCP servers", two filters are applied: keep only remote servers reachable over streamable HTTP, and exclude servers requiring third-party credentials. "This process reduced the dataset to 30.6% (871 servers)."
- Test questions are then run against each server's tools, dropping servers with failing tools: "This rigorous curation process resulted in a refined set of 495 high-quality MCP servers", exposing "more than 2,000 tools" (§1).

## Pipeline
- Task synthesis from MCP servers in single-server, multi-server and featured-server modes with five generator models; task filtering by an LLM over six dimensions; trajectory generation by three teacher models inside two agent frameworks with real MCP tool execution; post-filtering by rule checks plus an LLM judge on completeness and conciseness.
- Extensions: irrelevance tasks the toolset cannot solve, persona-based diversification, multi-turn self-simulation.
- Total dataset: 1.5M trajectories.

## SFT subset and results (§4.1, Table 2)
- SFT subset selection keeps question-quality and scenario-realism scores of 5, completeness and conciseness of at least 4, and full desired-tool use. "The resulting SFT dataset comprises 28.3K instances from the original pipeline, 40K instances from Ext.1 (Irrelevance), 15.8K instances from Ext.2 (Diversify), and 35.2K instances from Ext.3 (Multi-Turn), totaling 119.3K instances."
- BFCL V3 overall accuracy after fine-tuning:
```
Qwen2.5-7B-Instruct    55.10%  ->  58.26%  (+3.16)
Qwen2.5-14B-Instruct   57.69%  ->  65.09%  (+7.40)
Qwen2.5-32B-Instruct   61.73%  ->  70.45%  (+8.72)
```
  For reference in the same table: GPT-4.1 68.69%, GPT-4.5-Preview 70.32%, Qwen3-235B-A22B 67.94%.
- τ-Bench and τ²-Bench results are in Table 3; MCP-Universe results are reported as a Pareto-frontier comparison.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2510.01179: Abstract, §1, §3 (Stage 1-3 and extensions), §4.1, Table 2.
- Not reported by the source: training on models above 32B; RL use of the dataset; per-domain task counts in the full 1.5M set.
