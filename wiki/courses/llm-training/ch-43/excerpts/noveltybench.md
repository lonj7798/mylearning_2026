---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2504.05228
primary_version: arXiv:2504.05228v4 (9 Aug 2025); COLM 2025
created_at: "2026-09-15"
---

# Excerpt: NoveltyBench — Evaluating Language Models for Human-like Diversity

Zhang, Diddee, Holm, Liu, Liu, Samuel, Wang, Ippolito (Carnegie Mellon University), 2025.

## Metrics (§3)
- Generations of the same prompt are partitioned into equivalence classes by a trained partitioner; two
  generations are in the same class when they "provide distinct value to a user" is false (the annotation
  guideline ignores paraphrase-level differences).
- `distinct_k` = the number of equivalence classes among `k` samples (Eq. 1).
- `utility_k` (Eq. 2) weights the first generation of each new class by its quality score and discounts later
  generations geometrically with a patience parameter `p`, set to 0.8. Quality `u_i` comes from
  Skywork-Reward-Gemma-2-27B-v0.2.
- Sampling for evaluation: 10 independent generations per prompt at temperature 1.0, which the authors
  describe as the best case for diversity because most APIs default to lower temperature or nucleus sampling
  (§4.1).

## Results used in ch-43
- §4.2: the evaluated models "produce on average fewer than 3 distinct responses in 10 queries"; smaller
  models such as Gemma 2-2B and Llama 3.2-1B are the most diverse; Claude 3, Gemini and GPT-4o score below 4
  on `utility_10` out of a maximum of 10.
- §4.4, Fig. 6 (OLMo 2 checkpoints released after each post-training stage, `distinct_10` bar labels):
  1B — SFT 8.83, DPO 8.08, RLVR 7.85; 7B — 7.46, 5.96, 5.72; 13B — 7.47, 5.61, 5.16; 32B — 7.25, 5.22, 5.08.
  The text summarizes this as "each alignment stage progressively reduces model diversity, with significant
  drops occurring during DPO", and notes that utility rises from SFT to DPO because the quality of the
  remaining distinct outputs improves.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2504.05228v4 (scratchpad `sources/noveltybench.txt`).
  The Figure 6 values above are the bar labels as they appear in the extracted text, grouped by model size in
  the printed order SFT, DPO, RLVR.
- Not reported: token-level entropy of the evaluated models; any causal experiment that varies the KL penalty
  or entropy coefficient.
