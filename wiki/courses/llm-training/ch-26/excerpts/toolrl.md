---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2504.13958
created_at: "2026-09-15"
---

# Excerpt: ToolRL — rule-based rewards for tool calls, GRPO vs SFT on the same data

**Paper:** Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji, "ToolRL: Reward is All Tool Learning Needs", arXiv:2504.13958 v1 (2025-04-16). Read 2026-09-15.

## Reward (§3.3)

```
R_format  ∈ {0, 1}   1 if all required fields (<think>, <tool_call>/<response>) appear in the correct order
r_name    = |N_G ∩ N_P| / |N_G ∪ N_P|                          ∈ [0, 1]
r_param   = Σ_{G_j∈G} |keys(P_G) ∩ keys(P_P)| / |keys(P_G) ∪ keys(P_P)|   ∈ [0, |G|]
r_value   = Σ_{G_j∈G} Σ_{k∈keys(G_j)} 1[P_G[k] = P_P[k]]          ∈ [0, Σ|keys(G_j)|]
r_match   = r_name + r_param + r_value,   S_max = 1 + |G| + Σ_j |keys(G_j)|
R_correct = 6 · R_max / S_max − 3                                 ∈ [−3, 3]
R_final   = R_format + R_correct                                  ∈ [−3, 4]
```

N_G and N_P are the tool-name sets of the ground-truth calls G and predicted calls P; R_max is the total match score under the best matching between P and G.

## Training (§3.4, §4.1–4.2, App. B)

- Data: 4K examples, 2K from ToolACE, 1K from Hammer (masked), 1K from xLAM; multi-step trajectories split into single steps with history in the prompt.
- GRPO without a KL term; batch 512; 4 rollouts per query; 15 epochs; LR 1e-6; temperature 1.0; max prompt 2,048 and response 1,024 tokens; veRL on 2 A100 80GB (Table 8).
- SFT baselines: SFT400 and SFT4k on the same examples, "with thought content distilled from Deepseek-R1 trajectories" (App. B).

## Results

- BFCL V3 overall (Table 1): Qwen2.5-7B-Instruct raw 41.97, SFT4k 36.53, GRPO cold start 58.38; Qwen2.5-3B raw 33.04, SFT4k 41.97, GRPO 52.98; Llama-3.2-3B raw 22.09, SFT4k 44.16, GRPO 44.10. The Qwen2.5-3B and Qwen2.5-7B SFT400 rows print identical overall and irrelevance values (34.08, 8.11).
- Bamboogle multi-hop QA with web search (Table 3): Qwen2.5-7B raw 69.6, SFT400 28.8, SFT4k 30.4, GRPO 72.0.
- SFT-initialized GRPO reaches higher training reward but lower benchmark scores than cold-start GRPO; the authors hypothesize memorization and overfitting from SFT initialization (§4.3, Fig. 5).
- A length reward on the think block lowered BFCL overall for Qwen2.5-1.5B from 46.20 to 33.23 and irrelevance from 56.44 to 4.52 (Table 5).

## Not reported

Seeds and variance; general-capability benchmarks outside tool use and Bamboogle.
