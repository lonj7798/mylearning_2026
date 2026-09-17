---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: arXiv:2504.11343v2 (no library card for slug raft-reinforce-rej-minimalist on 2026-09-15; chapter-local verified extract; ch-31 holds a longer extract)
source_url: https://arxiv.org/abs/2504.11343
created_at: "2026-09-15"
---

# Excerpt: A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce

- **Authors:** Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, et al. (Salesforce AI Research; UIUC)
- **Year:** 2025 (arXiv v1 2025-04; v2 2025-06-12; preprint)
- **Source type:** paper
- **Used in:** ch-37 §4 (baseline sets the sign), Negative samples and negative feedback, Recipe.

## Objectives (§3)
- Reward r(x, a) ∈ {−1, 1} from Math-Verify.
- RAFT (Eq. 1): sample n responses per prompt, keep the highest-reward responses, maximize Σ log π_θ(a | x).
- RAFT++ (Eq. 6): the clipped REINFORCE loss with the indicator I[r(x, a) = max_i r(x, a_i)] in place of the reward, so only highest-reward responses receive gradient.
- REINFORCE (Eq. 5): clipped token-level loss weighted by r(x, a) ∈ {−1, 1}.
- GRPO: r replaced by (r_i − mean(r)) / std(r) over the n responses.
- The authors' reading (§5): REINFORCE with ±1 reward "can be viewed as fine-tuning on the positive samples and unlearning on the negative samples".

## Setup (§4)
verl; Numina-Math prompts; Qwen2.5-Math-7B-base and LLaMA-3.2-3B-instruct; AdamW, LR 1 × 10⁻⁶; 1,024 prompts per iteration; n = 4 responses per prompt for RAFT and GRPO; mini-batch 512; maximum 4,096 generated tokens. Table 1 caption: batch size, mini-batch size, and actor LR were tuned per algorithm and refers to an appendix for the per-algorithm values; v2 contains no such appendix, so those values are not available. Evaluation: average@16 at temperature 1.0 on MATH500, Minerva Math, OlympiadBench.

## Results (Table 1, average column)
| Algorithm | Qwen2.5-Math-7B-base | LLaMA-3.2-3B-instruct |
|---|---|---|
| Base | 23.6 | 13.1 |
| RAFT | 52.3 | 25.9 |
| RAFT++ | 56.1 | 27.6 |
| Reinforce | 53.9 | 24.2 |
| GRPO | 56.3 | 28.4 |
| PPO | 52.5 | 26.9 |
| Reinforce-Rej | 56.4 | 28.5 |

## Findings (§5)
- RAFT++ "exhibits a much more rapid decline in policy entropy compared to GRPO"; once entropy is low its improvement slows and GRPO overtakes it later in training (§5.1, Fig. 2–3). "These findings suggest that negative samples play a crucial role in maintaining exploration" (§5.1; Interpretation by the authors).
- GRPO ablation on LLaMA-3.2-3B-instruct (Fig. 4): removing prompts whose responses are all wrong gave the largest gain over vanilla REINFORCE; removing all-correct prompts "does not help much"; mean-zero normalization alone raised KL without improving reward; std normalization added little over removing both kinds of prompts.
- Reinforce-Rej filters both all-correct and all-wrong prompts and improves KL efficiency and entropy stability (§5).
- Conclusion (§6): future work "should focus on more principled designs for incorporating negative samples, rather than relying on them indiscriminately" (Abstract).

## Not reported
Seeds or confidence intervals; per-algorithm tuned hyperparameters; non-math or out-of-domain evaluations; pass@k at large k; a split of the gain between positive and negative samples.

## Verification
- Read on 2026-09-15 against arXiv:2504.11343v2 PDF text (Abstract, §3–§6, Table 1, Fig. 2–4 captions).
