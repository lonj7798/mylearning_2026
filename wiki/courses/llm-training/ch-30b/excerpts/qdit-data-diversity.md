---
chapter: ch-30b
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2311.14736
created_at: "2026-09-15"
---

# Excerpt: Data Diversity Matters for Robust Instruction Tuning (QDIT)

- **Authors:** Alexander Bukharin, Shiyang Li, Zhengyang Wang, Jingfeng Yang, Bing Yin, Xian Li, et al. (Georgia Institute of Technology; Amazon)
- **Year:** arXiv v1 2023-11; v3 2024-11
- **Checked against:** arXiv:2311.14736v3 (PDF, full text including Appendices A-F), 2026-09-15
- **Why this excerpt exists:** ch-30b uses QDIT as the worked example of a diversity measure for data selection. The library card `qdit-data-diversity` was planned but not present when the chapter was written.

## Method (§3)
- Diversity of a selected subset A of the pool V is the facility location function (Eq. 1):
  `d(A) = Σ_{v∈V} max_{a∈A} sim(a, v)`
  where sim is the cosine similarity of instruction embeddings from the sentence-transformer all-mpnet-base-v2 (App. A).
- Quality q(a) is a per-example score: ChatGPT ratings from AlpaGasus for Alpaca, and a reward model trained on Anthropic HH for the other datasets (§4.2).
- Quality-diversity score for adding a to A (§3.2): `f(a | A, α) = (1 − α) d(a | A) + α q(a)`, α ∈ [0, 1], where d(a | A) is the marginal gain in d.
- Selection is greedy (Algorithm 1). α = 1 reduces to quality-driven selection; α = 0 reduces to the classical greedy facility-location algorithm (§3.2).
- The text read does not state how d and q are put on a common numeric scale.
- Cost: about 9.42 minutes on one A100 40G to select 10,000 examples, versus 48 minutes on 8 A100 40G to train on Ultrachat 10K (App. B).

## Setup (§4.2; App. D)
- Base model LLaMA-1 7B; batch 128, LR 2e-5, 3 epochs, max length 512, weight decay 0 (App. D, Table 4).
- Pools: Alpaca 52K and Dolly 15K (select about 5%), Ultrachat 1.3M, LMSYS-Chat 1M, and "Mixed 270K" (Alpaca + Dolly + OIG-small-chip2) (select 10K).
- Evaluation: pairwise winning score judged by Claude 2 in both orders on five instruction sets, and "HH Score" from the reward model. Worst case = average over the lowest 10% of instructions; best case = top 10% (§4.2).

## Results with loci
- Table 1, Ultrachat 1.3M → 10K (Random / Quality / QDIT): mean HH score 6.219 / 6.961 / 6.993; lowest-10% HH score 2.620 / 3.405 / 3.497; top-10% HH score 9.526 / 10.454 / 10.454.
- Main results text, QDIT versus quality-driven selection: worst-case HH score +5.2%, worst-case winning score versus Alpaca 52K +6.26%, average HH score +1.5% (§4.3).
- Quality-only selection lowers the lowest-10% winning score relative to random selection in two of five settings (§4.3).
- α ∈ {0.5, 0.7, 0.9} typically gives the highest worst-case and average performance; α = 0.1 and α = 1 both drop (§4.4, Fig. 4).
- Threshold-based de-duplication (QDIT:Threshold) and cluster-balanced selection (QDIT:Cluster) achieve "slightly worse average performance compared to QDIT, but they often result in much worse worst-case performance" (§4.4, Fig. 8a; no numbers in the text).
- Benchmarks (Table 3): QDIT has a higher average than quality-driven selection on four of five pools (e.g. Ultrachat 10K: 52.05 vs 50.17).
- Worst-case gains persist with Claude-2.1, GPT-3.5-Turbo, and GPT-4-Turbo as judges and in an author-annotated human study (§4.4, Fig. 8b).

## Limits
- Diversity is measured over the candidate pool, not over a target task distribution; the authors state that it is difficult to attribute instruction-following gains directly to test-set coverage (§4.4, Fig. 9).
- One base model family for the main results (LLaMA-1 7B; LLaMA-2-13B in a secondary figure), short maximum length (512), and judge- or reward-model-based metrics.
- Selection is within one instruction pool; cross-skill mixtures (math, code, agentic, long-context) are not studied.
