<!-- scope: Writing-Zero / Writing-R1 (Alibaba Quark): pairwise writing GenRM trained with self-principled critique (SPCT) cold start + GRPO; Bootstrapped Relative Policy Optimization (BRPO) that uses a random in-group rollout as the reference; position-bias weighting; length and redundant-explanation hacking measured against a scalar RM baseline
     deps: [[generative-reward-models]], [[grpo]], [[bradley-terry-rm]]
     see-also: [[deepseek-grm]], [[litbench]], [[rubicon-rubric-anchors]], [[judge-llm-bias]], [[subjective-reward-guide]]
-->

# Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards
- **Core Insight:** Replacing a scalar RM with a pairwise generative RM that compares each rollout to a randomly chosen rollout from the same group (BRPO, advantage = ±1) removed the reward hacking seen with the scalar RM: the scalar-RM GRPO policy reached a mean response length of 1,872 with 417 tokens of redundant self-explanation, versus 1,292 and 58 for Writing-Zero (Table 3), and scored 2.83 vs 3.84 on the held-out Writing Testset (Table 2).
- **Guideline:** For RL on subjective writing, use a pairwise judge against an in-group bootstrapped reference and filter groups whose preferences are nearly all one sign (|ΣR_i|/G > 0.6), because an outlier reference produces all-+1 or all-−1 groups with no ranking signal (§3.2.1, Eq. 12, §4.1); also track an independent held-out evaluator, because the scalar-RM policy produced "gibberish" yet scored highest on WritingBench (8.87; §4.3.2).
- **Authors:** Ruipeng Jia, Yunyi Yang, Yongbo Gai, Kai Luo, Shihao Huang, Jianhe Lin, Xiaoxi Jiang, Guanjun Jiang (Quark LLM, Alibaba Group)
- **Year:** 2025 (arXiv v1 2025-05-30; v2 2025-06-11)
- **URL:** https://arxiv.org/abs/2506.00103
- **Source type:** paper
- **Relevant topics:** pairwise GenRM, reference-free pairwise RL, position bias, length bias, over-explanation hacking, test-time voting for judges, RL without SFT

## Summary
Stage 1 trains a Pairwise Writing GenRM on Qwen3-32B-Base: it writes specific principles, a critique, and outputs two float scores in [0, 10] (§3.1.2, Eq. 6). Stage 2 trains the writer with BRPO: for each query, one rollout o_ref is drawn at random from the group, every rollout is compared with it, and the advantage is +1 if the rollout wins "in all voting conditions", else −1 (§3.2.1, Eq. 11). Writing-Zero starts from Qwen3-32B-Base with no SFT; Writing-R1 starts from an in-house thinking SFT model (§3.2.2). Data are mainly Chinese (§4.1).

## Key Contributions
- GenRM data: ~200K in-house pairs including 30K writing pairs; re-scored by a scalar RM; ~10K pairs kept with high chosen reward and large reward gap (§3.1.1).
- Cold start: 1K pairs × 2 orders sent to Claude-3.5-Sonnet for critiques; kept only queries where both orders agreed with the label → 500 pairs → 1,000 examples; 1 epoch SFT (§3.1.3).
- GenRM RL rewards: accuracy ±1 (Eq. 7), format ±1 for scores in [0, 10] (Eq. 8), score-margin weighting below τ_margin = 2 (Eq. 9), and batch-level position-bias advantage weighting (Eq. 10).
- BRPO removes the need for a fixed external reference and for group-std normalization (§3.2.1).

## Key Figures/Tables to Study
- Fig. 1 (Eval RM during training, scalar-RM GRPO vs BRPO), Table 1 (RM accuracy), Table 2 (writing results), Table 3 (length and explanation), Fig. 4 (position-bias convergence), Fig. 5 (voting@n).

## Technical Details
**RM accuracy (Table 1).** Pairwise Writing GenRM: Cultural & Creative Writing 57.5, Lifestyle Copywriting 54.8, RewardBench 87.4, M-RewardBench (zh) 86.1. In-house Writing Scalar RM: 64.7 / 62.8 / 92.9 / 89.4. Claude-3.5-Sonnet: 53.0 / 46.8 / 84.2 / 79.7. The GenRM is less accurate than the scalar RM on the in-domain sets but works better inside RL (§4.3.2).

**Writing results (Table 2).** WritingBench / Writing Testset: Qwen3-32B-Base 6.89 / 1.23; ScalarRM-GRPO 8.87 (marked as hacked) / 2.83; Writing-Zero 8.29 / 3.84; Writing-Zero voting@2 8.35 / 3.86; Writing-SFT 8.56 / 3.31; Writing-SFT-ScalarRM-GRPO 8.77 / 3.69; Writing-R1 8.68 / 3.93.

**Human evaluation (§4.3.2).** 166 instances, Good:Same:Bad. Writing-R1 vs Writing-SFT 47:106:13; Writing-R1 vs Writing-SFT-ScalarRM-GRPO 28:120:18.

**Position bias (§3.1.3, §5.2).** Claude-3.5-Sonnet favored the first response about 60% of the time. The cold-started GenRM favored the second response; RL moved the mean P_former toward 0.5, and the advantage weighting in Eq. 10 reduced its batch-to-batch variance (Fig. 4).

**GenRM training limit (§4.3.1).** Dynamic sampling dropped >95% of GenRM training prompts in later stages (Fig. 3a), so GenRM training was stopped before test accuracy converged.

**Voting (§5.3).** Voting@8 raised GenRM accuracy from 57.5% to 59.2% (Cultural & Creative) and 54.8% to 57.6% (Lifestyle).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Writing-Zero / Writing-R1 | 32B | RL | lr; temperature; top-p; top-k | 1e-6; 1.0; 1.0; −1 | §4.1 | verified 2026-11-24 | no ablation |
| Writing-Zero | 32B | RL | BRPO filter τ_filter; group size | 0.6; 16 (filter if sum > 9) | §4.1 | verified 2026-11-24 | no ablation |
| Pairwise Writing GenRM | 32B | RM | margin τ_margin; cold-start size | 2; 1,000 examples, 1 epoch | §3.1.3–3.1.4 | verified 2026-11-24 | no ablation |
| Writing-Zero | 32B | RL | KL coefficient; steps | not reported | checked §3–4 | not reported | — |

## Findings relevant to generality and negative feedback
- **Negative feedback.** BRPO gives −1 to every rollout that does not beat the reference in all votes, so half the signal is explicit negative reward; ties are not rewarded (Eq. 11). The authors leave open whether inconsistent votes should get 0 or −1 (§5.3).
- **Benchmark validity.** WritingBench rewarded the hacked scalar-RM policy (8.87) while the in-house Eval RM did not (§4.3.2): a public LLM-judged benchmark is not sufficient to detect reward hacking.
- **Generality.** Not measured beyond writing benchmarks; no math, knowledge, or instruction-following evaluation.

## Connections
- [[deepseek-grm]] (SPCT origin), [[generative-reward-models]]: principle-then-critique GenRMs.
- [[litbench]]: on English creative writing, CoT GenRMs were less accurate than non-CoT ones; Writing-Zero reports CoT GenRM working inside RL.
- [[rlmt-think-chat]]: self-praise rises under a scalar RM, same pattern as Table 3's redundant explanations.

## Verification
- Read the arXiv v2 PDF (2506.00103v2) in full on 2026-11-24 via pdftotext; all numbers from Tables 1–3, §3.1.1–3.2.1, §4.1, §4.3, §5.2–5.3.
- Not reported: KL settings, number of RL steps, how "redundant explanation" length in Table 3 was measured, and the Eval RM's agreement number with humans (described only as "high degree of consistency", §4.1).
