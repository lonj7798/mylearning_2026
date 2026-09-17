---
chapter: ch-32d
course: llm-training
phase: read
excerpt_of: Jiang, Wang, Liu, Wang, Xu, Yao, et al. (University of Washington; Snowflake; UNC Chapel Hill), "MidTool: Mid-training Data Synthesis for Agentic Tool Use", arXiv:2608.20314v1 (2026-08-20)
source_url: https://arxiv.org/abs/2608.20314
created_at: "2026-09-15"
source_type: paper
---

# Excerpt: MidTool — tool-use mid-training corpus and ablation

No library card for this source exists at the time of writing (planned slug `midtool`). Only passages used by [[read]] §6, §9, §11, the negatives section, and the Recipe table are extracted.

## Corpus (§2, Table 2)

| Source | Tokens (B), source / context-grounded augmentation | Samples | Ratio |
|---|---|---|---|
| Web | 4.4 / 4.1 | 6.86M | 42% |
| PDF | 2.6 / 2.1 | 1.34M | 23% |
| Code | 3.8 / 1.5 | 2.60M | 26% |
| Native Agentic Trajectory | 1.8 | 0.42M | 9% |
| Total | 20.3 | 11.22M | 100% |

- Two synthesis branches: context-grounded trajectory augmentation from web, PDF, and code documents; native agentic trajectory synthesis from real APIs and MCP skills, generated with GPT-5, GPT-5.1, and GPT-5.2, "strictly validated for turn ordering, schema grounding, required arguments, and tool-response consistency; invalid generations are additionally retried with quality-control feedback before discard"; plus rollouts from agentic world model (AWM) environments and filtered Nemotron Agentic traces (§2.3).
- "Following Olmo et al. (2025), we normalize all trajectories into a plain chat-style template without special control tokens such as <im_start>." (§2.4)
- Decontamination audit with DeCon found "no evidence of leakage" (§2.4, App. A.3).

## Setup (§3.1, App. B)

- Qwen3-4B-Base and Qwen3-8B-Base; mid-training (Table 10): 1 epoch, max sequence length 8,192, AdamW, LR 3 × 10⁻⁵, betas (0.9, 0.999), weight decay 0.01, WSD schedule, 50 warmup steps, target global batch 4M tokens with packing.
- SFT (Table 11): 100K TOUCAN tool-use samples, max sequence length 32,768, LR 2 × 10⁻⁵, betas (0.9, 0.95), weight decay 0.01, cosine with warmup ratio 0.001, batch 128.
- RL (Table 12): GRPO, 64 steps, LR 1 × 10⁻⁶ (4B) / 5 × 10⁻⁷ (8B), batch 32, 16 rollouts per prompt, KL coefficient 0.001, entropy coefficient 0.0, clip ratio high 0.28, maximum 20 agent turns; 526 AWM environments.

## Results (Tables 3-6)

- BFCLv3 overall: 4B SFT 39.73% → MidTool-Mix + SFT 50.25% (+RL 39.51% → 54.18%); 8B SFT 47.62% → 51.12% (+RL 45.79% → 55.12%). BFCL hallucination column: 4B 60.46% → 56.95%; 8B 65.03% → 59.82% (SFT rows).
- τ²-Bench overall Pass@1: 4B 8.54% → 12.23%; 8B 10.43% → 14.75%. Telecom, 8B SFT rows: Pass@1 4.39% → 1.54%, Pass@4 13.16% → 3.51%.
- MCP-Universe overall score: 4B 13.20 → 18.66; 8B 15.18 → 17.82; web-search subset 0.00 in all rows.
- Table 6 (4B + SFT, only the mid-training corpus changes; BFCL overall / τ² Pass@1 / MCP-Universe score): no mid-training 39.73% / 8.54% / 13.20; Dolmino-20BT (matched budget) 43.10% / 7.37% / 5.41; processed data without trajectories 42.30% / 7.30% / 12.20; + native agentic trajectories 47.59% / 4.23% / 6.80; + context-grounded trajectories 44.66% / 8.99% / 8.46; MidTool-Mix 50.25% / 12.23% / 18.66.
- App. C.2: RL reward curves of mid-trained and base models "move much closer" at later steps, while the post-RL benchmark gap persists.

## Verification

- Checked on 2026-09-15 against https://arxiv.org/abs/2608.20314 (v1), §1-§3, App. A-D.
- Not reported by the source: number of evaluation seeds; any non-tool-use benchmark (knowledge, math, general chat) before or after mid-training; replay of general data inside MidTool-Mix beyond the filtered web/PDF/code sources.
