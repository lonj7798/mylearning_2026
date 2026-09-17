---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: "Towards General Agentic Intelligence via Environment Scaling (AgentScaler), Tongyi Lab, arXiv:2509.13311v1 (2025-09-16)"
source_url: https://arxiv.org/abs/2509.13311
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug agentscaler). Values read from the v1 PDF on 2026-09-15."
---

# Excerpt: AgentScaler — database-backed simulated tool environments

**Authors:** Runnan Fang, Shihao Cai, Baixuan Li, Jialong Wu, Guangyu Li, Wenbiao Yin, et al. (Tongyi Lab, Alibaba Group).

## Design principle (§2)
> "any function call can be interpreted as a read–write operation over an underlying environmental database D"
- API(func, α) ≡ op(func)(α; D), op(func) ∈ {read, write}; α are the call arguments. Tools in one domain share a database schema S_k, so environment design reduces to partitioning tools into domains and assigning each a schema.

## Environment build (§2.1)
- More than 30,000 APIs collected from ToolBench, API-Gen, and an internal repository; low-quality APIs removed; some descriptions rewritten with explicit input-output specifications.
- Tool graph edges (Eq. 1): E = {(i, j) | sim(φ(P_func_i), φ(P_func_j)) > τ, i ≠ j}, where φ embeds a tool's parameter list and sim is cosine similarity; τ is not printed.
- Louvain community detection gives domains; an LLM then checks pairwise dependencies inside each domain. "In total, we obtained M domains (exceeding 1,000)."
- Each tool is written as Python code that reads and writes the domain database. > "when generating database structures and formalizing code within specific domains of τ-bench, we observe through manual inspection that our outputs exhibit a high degree of consistency with the official implementations provided by τ-bench"

## Task construction (§2.2)
- Initialize a diverse database state; sample a tool sequence by a directed walk on the domain graph until the maximum step count or a node without outgoing edges; generate arguments and execute each call on the database. Verifiability comes from "(i) database-level state consistency and (ii) exact matching of tool sequences."

## Experience collection and filtering (§3.1)
- A simulated user pursues the task intent; the agent uses domain tools until the simulated user deems the task complete.
- Three filters: validity control (well-formed turns, n-gram repetition filter); environment state alignment (final database state must equal the gold state); function-calling exact match (used when the sequence has only read operations).
- > "we do not filter out trajectories in which tool calls return errors ... Retaining them in the training data helps improve the robustness of the model."

## Training (§3.2, §4.1)
- Loss only on tool-call tokens and assistant-response tokens; user turns and tool responses masked (Eq. 2).
- Stage 1 general domains, Stage 2 vertical domains. Backbones: Qwen3-Thinking-4B-2507 (AgentScaler-4B), Qwen3-8B (AgentScaler-8B), Qwen3-Thinking-30B-A3B-2507 (AgentScaler-30B-A3B). Learning rate, epochs, and data size are not reported.

## Results (Tables 1-2)
| Model | τ-bench Retail / Airline | τ²-Bench Retail / Airline / Telecom | ACEBench-en Overall |
|---|---|---|---|
| Qwen3-Thinking-30B-A3B | 67.8 / 48.0 | 58.8 / 58.0 / 26.3 | 67.2 |
| AgentScaler-30B-A3B | 70.4 / 54.0 | 70.2 / 60.0 / 55.3 | 75.7 |
| Qwen3-8B | 45.2 / 25.0 | 41.2 / 30.5 / 23.5 | 65.9 |
| AgentScaler-8B | 50.4 / 42.0 | 58.8 / 44.0 / 45.4 | 67.4 |
| Qwen3-Thinking-4B | 59.1 / 52.5 | 56.1 / 52.0 / 28.7 | 49.5 |
| AgentScaler-4B | 64.3 / 54.0 | 62.3 / 56.0 / 48.2 | 65.9 |
- ACEBench-en Special subset: Qwen3-Thinking-4B 84.7 → AgentScaler-4B 76.7; Qwen3-Thinking-30B-A3B 86.7 → AgentScaler-30B-A3B 82.7 (Table 1).
- ACEBench-zh, described as out-of-distribution (Table 2), Normal / Special / Agent / Overall: 4B 34.7/85.3/6.7/43.9 → 70.8/70.0/38.4/65.6; 8B 80.3/72.7/35.0/71.3 → 75.2/79.3/58.4/73.7; 30B-A3B 73.4/86.7/55.8/74.2 → 85.3/83.3/64.1/81.5.
- Accuracy falls as the number of tool calls per trajectory rises on τ-bench (Fig. 5).

## Limitations (Limitation section)
No RL; validated only up to 30B scale.
