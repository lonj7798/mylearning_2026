---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reinforce-plus-plus.md (verified card, 2026-09-14)
source_url: https://arxiv.org/abs/2501.03262
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; previous version replaced)"
---

# Excerpt: REINFORCE++

- **Authors:** Jian Hu, Jason Klein Liu, Haotian Xu, Wei Shen (v9); v1 by Jian Hu
- **Year:** 2025 (arXiv v1 2025-01-04; v9 2025-11-10)
- **Source type:** paper
- **Used in:** ch-37 §2 (bias of group-normalized advantages), §6 (where the KL term is placed). Full algorithm treatment is in ch-40.

## Items used by ch-37, with loci (from the verified card)
- **Bias of the GRPO advantage (App. A.1, Theorem 1).** (r_i − mean)/std over a group of N ≥ 2 is a biased estimator because the local std depends on r_i.
- **KL in the reward (Eq. 4).** A_{q,o_t} = r(o_{1:T}, q) − β Σ_{i=t}^{T} KL(i), KL(t) = log[π_θold(o_t | q, o_<t) / π_ref(o_t | q, o_<t)]: a single-sample log-ratio on the sampled token.
- **Global normalization (Eq. 5).** A^norm = (A − mean over the batch) / (std over the batch + ε).
- **w/ Baseline variant (Eqs. 6–8).** Group-mean subtraction, then global std, plus a separate k2 KL loss.
- **Reward sign (§5.1).** Plain REINFORCE++ "performs best with symmetric rewards, such as −1/1"; group-mean subtraction lets the w/ Baseline variant use 0/1 rewards.
- **Small-set overfitting (v9 Table 2; model not stated).** Trained on 30 AIME-24 questions: GRPO 95.0 train pass@1, AIME-25 pass@1 0.0, pass@16 0.4; REINFORCE++ (k > 1) 71.0, 2.5, 40.0.

## Not used
The previous version of this excerpt claimed a variance figure comparing group and global normalization; no such figure exists in v1 or v9 (card Verification section).
