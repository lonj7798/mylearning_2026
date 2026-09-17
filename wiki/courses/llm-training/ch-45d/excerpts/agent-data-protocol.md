---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: arXiv:2510.24702v2 (ICLR 2026) (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2510.24702
created_at: "2026-09-15"
---

# Excerpt: Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents

- **Authors:** Yueqi Song, Ketan Ramaneti, Zaid Sheikh, et al. (CMU and collaborators)
- **Year:** 2025 (arXiv v1 2025-10-28; v2 2026-03-04; published at ICLR 2026)
- **Source type:** paper
- **Used in:** [[read]] §3, §7, Generalization lens

## What ADP is (§1, §3)
- A schema that acts as an "interlingua" between raw agent datasets and agent harnesses. A Trajectory is an alternating sequence of Actions (API/tool calls, code actions, messages) and Observations.
- Two conversion stages: Raw → ADP (one converter per dataset) and ADP → SFT (one converter per harness). Conversion cost falls from O(D × A) to O(D + A) for D datasets and A harnesses.
- 13 datasets converted, "over 1.3M instances"; larger datasets are subsampled so no single source dominates (mixture weights in App. C).

## Training and evaluation setup (§5)
- Base models: Qwen2.5-Coder-Instruct family (and Qwen3-8B in §6.2). Harnesses: OpenHands CodeActAgent, SWE-Agent, AgentLab. Benchmarks: SWE-Bench Verified, WebArena, AgentBench OS, GAIA.

## Main results (§6.1)
- Headline: "an average performance gain of ∼20% over corresponding base models" (Abstract).
- "The 32B model reaches 40.3% (+38.1%) with SWE-Agent and 36.8% ... with OpenHands" on SWE-Bench Verified.
- WebArena: 7B 21.0% (+16.5%), 14B 22.2% (+16.7%), 32B 22.9% (+12.0%).
- AgentBench OS: "the 7B model improves from 3.5% to 27.1% (+23.6%)".
- GAIA: "the 7B model improves from 7.3% to 9.1% (+1.8%)".

## Cross-task transfer (§6.2, Table 6)
- Comparison of (i) base, (ii) task-specific fine-tuning, (iii) the mixed ADP corpus, holding agent setup and evaluation fixed.
- "ADP consistently outperforms task-specific tuning on the target task and, critically, avoids the negative transfer that single-domain tuning often induces on other tasks."
- Numbers quoted in the text: SWE-Bench, Qwen2.5-7B-Instruct 10.4% with ADP vs 1.0% with SWE-smith only; Qwen3-8B 16.6% with ADP vs 0.2% with CodeActInstruct + Code-Feedback and 11.0% with SWE-smith only. WebArena, Qwen2.5-7B-Instruct 20.1% with ADP vs 16.0% with Go-Browse only. AgentBench OS, Qwen3-8B 25.7% with ADP vs 21.5% with AgentInstruct only. GAIA: "AgentInstruct Only results in 0.6% accuracy, while ADP improves it to 9.1%."

## Conversion cost (§6.3, Tables 7-8)
- Raw → ADP converters total 4,892 lines of code across 13 datasets; ADP → SFT converters average about 77 lines per harness (OpenHands ~150, SWE-Agent ~50, AgentLab ~30).

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2510.24702 (v2, 2026-03-04): Abstract, §1, §2.1, §3, §5, §6.1, §6.2, §6.3.
- Not reported by the source: results above 32B; RL on ADP data; controlled per-dataset ablation of the mixture weights.
