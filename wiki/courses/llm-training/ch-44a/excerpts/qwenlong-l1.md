---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2505.17667v2 (QwenLong-L1), §2.2-§2.4, §3, §4.1-§4.3, Tables 3-4 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2505.17667
created_at: "2026-09-15"
---

# Excerpt: QwenLong-L1 — Towards Long-Context Large Reasoning Models with Reinforcement Learning

- **Authors:** Fanqi Wan, Weizhou Shen, Shengyi Liao, Yingcheng Shi, Chenliang Li, Ziyi Yang, Ji Zhang, Fei Huang, Jingren Zhou, Ming Yan (Tongyi Lab, Alibaba Group)
- **Year:** 2025 (this extract read from the v2 PDF, dated 2025-05-27; the v1 date was not checked)
- **Source type:** paper
- **Used in:** ch-44a §5, §6, Recipe, Generalization lens.

## Problem statement (§1, Figure 2)
Compared with short-context reasoning RL, long-context reasoning RL shows "delayed reward convergence", "marked reduction in
output entropy when processing long-context inputs, which restricts exploratory behavior", and "intermittent spikes in KL
divergence" caused by "longer output length with heterogeneous input length distributions". The objective is extended to
`max_θ E_{x,c∼D, y∼π_θ(·|x,c)}[r_φ(x, c, y)] − β D_KL[π_θ(y|x,c) ‖ π_ref(y|x,c)]` (Eq. 2), where `c` is the long context that
the policy must ground in before reasoning. The KL term is removed in the implementation "to encourage the exploration
capacity of the policy model".

## Progressive context scaling (§2.3)
- **Curriculum-guided phased RL**: RL is split into K phases with target input lengths `L_k`; in phase k only examples with
  `L_{k−1} < |x| + |c| ≤ L_k` are used (Eq. 9).
- **Difficulty-aware retrospective sampling**: instances from earlier phases are re-inserted with importance weights
  `diff(x, c) = 1 / mean({r_i}_{i=1..G})` (Eq. 10), the inverse mean reward of a group from the base model.
- **Warm-up SFT** on teacher demonstrations (DeepSeek-R1) at the first phase's input length, before RL.

## Hybrid reward (§2.4, Eqs. 12-14)
`r_φ(x, y) = max(r_rule(y), r_LLM(x, y))`: exact string match of the extracted answer, or a binary LLM judge
(Qwen2.5-1.5B-Instruct at temperature 0) for semantic equivalence. No format reward is used, because "excessive format
rewards could oversimplify the learning objective".

## Settings (§3.1, §3.2, Table 3)
- Data: DocQA-RL-1.6K (1.6K problems: 600 DocMath, 600 R1-synthesized multiple-choice logic, 200 MultiHopRAG, 200 Musique);
  SFT set 5.3K DeepSeek-R1 triplets. Train statistics: SFT 5,305 examples (avg 13,064 tokens, max 20,003); RL 1,591 examples
  (avg 11,437, max 59,559).
- Base models: R1-Distill-Qwen-14B and R1-Distill-Qwen-32B.
- RL: two phases, 20K input length in phase I and 60K in phase II; "difficulty-aware retrospective sampling to maintain the
  most difficult samples with an average accuracy of zero from phase I to II"; 32×A100-80G; train batch 128, mini batch 32,
  rollout number 8, learning rate 2e-6; sampling temperature 0.7, top-p 0.95, **maximum output length 10K**.
- SFT: 20K input length, 3 epochs, batch 128, learning rate 5e-6.
- Evaluation: maximum input 120K, output 10K, temperature 0.7, top-p 0.95; score is the maximum of exact match and
  DeepSeek-V3 judged accuracy.

## Results (§4.1, Table 4; average over seven DocQA benchmarks)
R1-Distill-Qwen-14B 64.2 → SFT 65.0 (+0.8) → QwenLong-L1-14B-GRPO 68.2 (+4.0) → -DAPO 68.3 (+4.1).
R1-Distill-Qwen-32B 65.6 → SFT 68.7 (+3.2) → QwenLong-L1-32B-GRPO 70.3 (+4.7) → -DAPO 70.7 (+5.1).
Comparison points: Claude-3.7-Sonnet-Thinking 70.7, OpenAI-o3-mini 70.4, DeepSeek-R1 72.1.
Pass@2 of QwenLong-L1-14B is 73.7, above DeepSeek-R1's pass@1 of 72.1 (§4.1, Figure 4).

## Ablations and trade-off (§4.2, §4.3)
- Warm-up SFT improves every configuration and "sustains lower gradient norm during different RL phases" (Figure 5a-b).
- Single-stage RL "exhibits heightened instability, as demonstrated by fluctuating KL divergence and entropy collapse"
  (Figure 5c); the benefit of phased RL "is less pronounced when models are initialized with SFT".
- Long-context SFT (10K distilled triplets) beats the base model by 2.6 points and the short-context SFT model by 2.1, but
  RL on top of it adds only 0.3 points to a final 67.4, against 3.2 points and 68.2 when RL starts from the short-context
  SFT model (§4.3, Figure 6).

## Not reported
Clip range, group-size sensitivity, number of RL steps per phase, truncation rate at the 10K output limit, and
short-context benchmark scores for QwenLong-L1 itself (MMLU comparisons appear in the LoongRL paper instead).
