---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents, arXiv:2510.24702v2 (2026-03-04; ICLR 2026)"
source_url: https://arxiv.org/abs/2510.24702
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug agent-data-protocol). Values read from the v2 PDF on 2026-09-15."
---

# Excerpt: Agent Data Protocol (ADP)

**Authors:** Yueqi Song, Ketan Ramaneti, Zaid Sheikh, Ziru Chen, Boyu Gou, Tianbao Xie, et al. (CMU; OSU; HKU; Duke; Fujitsu Research; All Hands AI).

## Claim (§1)
> "we argue that the issue is not a lack of data, but rather a lack of standardization."

## Schema (§3.2)
- `Trajectory(id, content, details)`: content alternates actions and observations; details holds dataset-specific metadata.
- Actions: `APIAction(function, kwargs, description)`, `CodeAction(language, content, description)`, `MessageAction(content)`.
- Observations: `TextObservation(source ∈ {user, environment}, content)`, `WebObservation(html, axtree, url, viewport_size, image_observation)`.
- Implemented as Pydantic schemas.

## Pipeline (§3.3-3.4)
1. Raw → ADP, once per dataset. 2. ADP → SFT, once per agent harness (OpenHands, SWE-Agent, AgentLab), which sets the system prompt, context handling, and action syntax. 3. Automated quality checks: tool-call format, most tool calls paired with an English thought (threshold 80%), proper conversation ending.
- Converter cost (Tables 7-8): 4,892 lines of code for 13 Raw→ADP converters; ADP→SFT average about 77 lines. For A = 100 harnesses: about 100 × 4,892 = 489,200 lines without ADP vs 4,892 + 77 × 100 = 12,592 with ADP (§6.3).

## Corpus (§2.1, §4, App. C)
- 13 datasets, 1.3M trajectories (ADP Dataset V1). Average 10.1 rounds; SWE-smith 26.8 rounds, Synatra 1.0; action mix overall 53% API / 24% code / 23% message (Table 2).
- Sampling multipliers w_d (Table 9): orca agentinstruct 0.001, synatra 0.01, code feedback 0.1, nebius SWE-agent trajectories 0.2, agenttuning subsets 2, swe-gym openhands sampled trajectories 3, others 1.
- Coding/SWE harnesses train on the non-web portion (about 30K samples); AgentLab trains on the web portion (about 20K samples) (App. C.1).

## Results (Tables 3-6, 10)
- SWE-Bench Verified: Qwen-2.5-7B-Coder-Instruct SWE-Agent 0.4 → 20.2, OpenHands 2.8 → 20.4; 14B SWE-Agent 2.0 → 34.4; 32B SWE-Agent 2.2 → 40.3 (SWE-smith alone 40.2), OpenHands 10.6 → 36.8.
- WebArena (AgentLab): 7B 4.5 → 21.0; 14B 5.5 → 22.2; 32B 10.9 → 22.9. AgentBench OS (OpenHands): 7B 3.5 → 27.1; 32B 27.8 → 34.7. GAIA 7B 7.3 → 9.1.
- Mixed vs single-source (Table 6, same harness and model): SWE-Bench, Qwen-2.5-7B-Instruct, OpenHands: SWE-smith only 1.0 vs ADP 10.4; Qwen-3-8B: CodeActInstruct + Code-Feedback 0.2, SWE-smith only 11.0, ADP 16.6. WebArena: Go-Browse only 16.0 vs ADP 20.1. AgentBench OS: AgentInstruct only 21.5 vs ADP 25.7. GAIA: AgentInstruct only 0.6 vs ADP 9.1.
- Equal scale (Table 10): Qwen3-8B, OpenHands, about 30K samples each: SWE-smith up-sampled 11.0 vs ADP 16.6.

## Not reported
Deduplication or decontamination across the 13 source datasets and the four evaluation benchmarks; general (non-agent) benchmarks after training.
