---
chapter: ch-30b
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2510.24702
created_at: "2026-09-15"
---

# Excerpt: Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents

- **Authors:** Yueqi Song, Ketan Ramaneti, Zaid Sheikh, Ziru Chen, Boyu Gou, Tianbao Xie, et al. (Carnegie Mellon University and collaborators)
- **Year:** arXiv v1 2025-10; v2 2026-03-04; ICLR 2026
- **Checked against:** arXiv:2510.24702v2 (PDF, full text including Appendices C-E), 2026-09-15
- **Why this excerpt exists:** ch-30b cites ADP for per-dataset sampling multipliers in an agentic SFT mixture and for the diverse-versus-single-domain comparison. The library card `agent-data-protocol` was planned but not present when the chapter was written.

## What ADP is (§3-§4)
- A common schema in which a trajectory is an alternating sequence of actions (API calls, code, messages) and observations (text, web). Each raw dataset is converted once to ADP, and each agent harness has one converter from ADP to its SFT format.
- 13 datasets are converted (Table 1, counts rounded to 0.1K as printed): AgentInstruct 1.9K, Code-Feedback 66.4K, CodeActInstruct 7.1K, Go-Browse 9.5K, Mind2Web 1.0K, Nebius SWE trajectories 13.4K, NNetNav-live 5.0K, NNetNav-wa 4.2K, openhands-feedback 0.2K, Orca AgentInstruct 1046.1K, SWE-Gym 0.5K, SWE-smith 5.0K, Synatra 99.9K. The text describes the total as "over 1.3M instances" (§5.1); the Table 1 rows sum to 1,260.2K (derived).

## Mixture weights (App. C, Table 9)
- Per-dataset multiplier w_d; for a dataset with n_d raw trajectories, m_d = ⌈w_d · n_d⌉ examples are drawn per epoch, without replacement if w_d < 1 and with replacement if w_d > 1.
- Values: agenttuning alfworld, db, kg, mind2web, os, webshop: 2 each; code feedback 0.1; codeactinstruct 1; go-browse-wa 1; mind2web 1; nebius SWE-agent-trajectories 0.2; nnetnav-live 1; nnetnav-wa 1; openhands 1; orca agentinstruct 0.001; swe-gym openhands sampled trajectories 3; swe-smith 1; synatra 0.01.
- Harness-specific subsets (App. C.1): for OpenHands CodeActAgent and SWE-Agent, only the non-web portion is used ("to avoid potential interference from web-specific interaction patterns"), about 30K samples; for AgentLab (WebArena), only the web portion, about 20K samples.
- The paper reports no ablation of w_d; App. C.1 states that future experiments could explore different multipliers.

## Training (§5.1)
- Base models: Qwen2.5-Coder-Instruct 7B, 14B, 32B, plus Qwen2.5-7B-Instruct and Qwen3-8B in Table 6. SFT pipeline: LLaMA-Factory. Learning rate, epochs, and sequence length are not reported in the sections read (§5.1, App. C-E).

## Results with loci
- Table 6 (cross-task transfer, same harness and model):
  - SWE-bench Verified, OpenHands: Qwen2.5-7B-Instruct SWE-smith only 1.0% vs ADP 10.4%; Qwen3-8B CodeActInstruct + Code-Feedback 0.2%, SWE-smith only 11.0%, ADP 16.6%.
  - WebArena, AgentLab, Qwen2.5-7B-Instruct: Go-Browse only 16.0% vs ADP 20.1%.
  - AgentBench OS, OpenHands, Qwen3-8B: AgentInstruct only 21.5% vs ADP 25.7%.
  - GAIA, OpenHands, Qwen2.5-7B-Instruct: AgentInstruct only 0.6% vs ADP 9.1%.
- The authors state that ADP "avoids the negative transfer that single-domain tuning often induces on other tasks" (§6.2) (Interpretation).
- Equal data scale (App. E.1, Table 10): Qwen3-8B on SWE-smith up-sampled to about 30K samples scores 11.0% on SWE-bench Verified with OpenHands; ADP at about 30K scores 16.6%.
- Table 5: SWE-bench Verified with OpenHands for Qwen2.5-Coder-32B-Instruct: base 10.6%, SWE-Gym 20.6% (row collected from prior work), ADP 36.8%.

## Limits
- Every Table 6 row evaluates on the target task of the single-domain baseline; effects of single-domain tuning on other agent tasks are shown only in the SWE-bench row with CodeActInstruct + Code-Feedback.
- The "ADP data" used for SWE-bench excludes web datasets (App. C.1), so it is a code, SWE, and tool mixture rather than the full corpus.
- Non-agentic benchmarks (knowledge, chat, safety) are not reported; no seeds or confidence intervals are reported.
- The paper does not study failed trajectories as a training signal; openhands-feedback contains recorded trajectories with human feedback, and it is converted like the other datasets.
