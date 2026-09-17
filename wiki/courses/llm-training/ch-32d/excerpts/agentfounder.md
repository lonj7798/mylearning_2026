---
chapter: ch-32d
course: llm-training
phase: read
excerpt_of: Su, Zhang, Li, Chen, Wang, Song, et al. (Tongyi Lab, Alibaba Group), "Scaling Agents via Continual Pre-training", arXiv:2509.13310v1 (2025-09-16)
source_url: https://arxiv.org/abs/2509.13310
created_at: "2026-09-15"
source_type: paper
---

# Excerpt: AgentFounder — agentic continual pre-training (Agentic CPT)

No library card for this source exists at the time of writing (planned slug `agentfounder`). Only passages used by [[read]] §3, §9, §10, the negatives section, and the Recipe table are extracted.

## Rationale (Abstract, §1)

- "post-training approaches building upon general-purpose foundation models consistently underperform in agentic tasks, particularly in open-source implementations. We identify the root cause: the absence of robust agentic foundation models forces models during post-training to simultaneously learn diverse agentic behaviors while aligning them to expert demonstrations, thereby creating fundamental optimization tensions."
- Two stated principles (§1): seed data sources "must be broad and not confined to any single domain", and training data "must comprehensively include various types of agentic behaviors, preventing models from imitating and memorizing specific behavioral patterns".
- Footnote 1: "GLM-4.5 incorporates synthetic agent trajectories during mid-training."

## Training pipeline (§2.1)

Starting from Qwen3-30B-A3B-Base:
- "Agentic CPT Stage 1: We process approximately 200B tokens of agent data and knowledge reasoning corpora with 32K context length, following the same next-token prediction paradigm as Eq. 1."
- "Agentic CPT Stage 2: We further refine these capabilities using 100B tokens of carefully curated, high-quality agent data with extended 128K context windows".
- CPT corpus (§3.1.1): "(1) high-quality web-crawled data filtered for factual accuracy, (2) historical tool invocation records, e.g., search results and web page content (3) offline Wikipedia data, and (4) mixed-quality discarded trajectories from previous post-training iterations."

## Data synthesis (§2.2-§2.3, App. B.1)

- First-order Action Synthesis (FAS): entity-anchored knowledge memory → multi-style questions; planning action synthesis generates problem analyses and first-step actions without calling tools; reasoning action synthesis generates an answer A1 from internal knowledge, then refines it to A2 given the required knowledge, with no tool calls in either step. Both use LLM-as-judge rejection sampling.
- App. B.1: FAS planning data is 50% correct / 50% incorrect before filtering; the filter "removes 43.5% of problematic samples, increasing retained trajectory accuracy from 50% to 82%".
- High-order Action Synthesis (HAS): for a trajectory of K steps with a binary judgment J, generate N alternative "thought and invocation" candidates per step without tool execution, shuffle them with the original step, insert "I will choose option n_k" followed by the real response R_k, and append "My decision is {Correct/Incorrect}" (§2.3). This yields "(N + 1) × K potential reasoning-actions".

## Results

- Table 3 (Pass@1; Qwen3-30B-A3B-Base → AgentFounder-30B-Base under the same SFT data): SFT-A BrowseComp-en 26.9 → 31.4, BrowseComp-zh 29.8 → 35.6, GAIA 67.0 → 72.8, HLE 23.5 → 30.4; SFT-B 28.6 → 39.9, 35.6 → 43.3, 71.8 → 72.8, 27.0 → 31.5; SFT-C 24.5 → 38.8, 36.7 → 44.3, 68.9 → 71.8, 27.9 → 28.9. Stated average gains: 5.75%, 6.13%, 6.45%.
- Table 4 (50B tokens, SFT-A; Stage 1 only vs Stage 1 & 2): BrowseComp-en Pass@1 31.4 vs 35.5; BrowseComp-zh 34.3 vs 37.2; GAIA 69.9 vs 72.8. In Stage 1 only, "some HAS data may be truncated due to length constraints."
- Table 5 (≈50B tokens, SFT-A): Non-CPT BrowseComp-en Pass@1 26.9, FAS 31.4, FAS+HAS 31.4; BrowseComp-zh 29.8 / 37.0 / 40.1; GAIA 67.0 / 72.8 / 69.9.
- §3.5.1: average accuracy 20.4% (1B), 32.7% (4B), 48.9% (30B).
- §3.5.2: data volume 0B to 315B tokens; "the most substantial improvements (3.8%) occurring within the initial 15B tokens"; total gain "8.0% (from 54.2% to 62.2%)" in average Pass@3; Stage 2 at 128K gives +1.8% at 65B over 50B and +1.0% at 315B over 210B.
- §3.6.1: same SFT-A data for 1,340 steps; final SFT loss 0.8656 (baseline) vs 0.7953 (AgentFounder-30B, 315B).
- Table 6 (ACEBench overall): Qwen3-30B-A3B 67.2, AgentFounder-30B 70.0.

## Verification

- Checked on 2026-09-15 against https://arxiv.org/abs/2509.13310 (v1), main text and App. B.
- Not reported by the source: CPT learning rate, schedule, batch size, and replay of general pre-training data; mixture percentages within each CPT stage; number of evaluation seeds; any non-agentic benchmark (knowledge, math, code, chat) before and after CPT.
