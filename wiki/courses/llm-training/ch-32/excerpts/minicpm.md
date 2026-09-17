---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: primary source arXiv:2404.06395v3 (planned library card model-reports/minicpm.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2404.06395
created_at: "2026-09-15"
---

# Excerpt: MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies

**Authors:** Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, et al. (Tsinghua University; Modelbest Inc.). arXiv v1 2024-04; read at v3 (PDF text). Source type: paper / model report.

## Warmup-Stable-Decay schedule (§4.2, Eq. 1)

The paper splits training into a warmup stage ending at step W, a stable stage ending at step T, and a decay stage:

```
WSD(T; s) = (s / W) η          for s < W
          = η                  for W < s < T
          = f(s − T) η         for T < s < S,   0 < f(s − T) ≤ 1 decreasing
```

η is the maximum learning rate, s the current step, S the final step (§4.2).

- "Loss Decreases Dramatically in Decay Stage": on 0.036B models the loss drops quickly once the LR decays and reaches the level of a cosine schedule at the same end step; a checkpoint from before decay can be trained further at the high LR and decayed later (§4.2, Figure 5).
- "10% Steps are Enough": decay runs forked from the same stable checkpoints (40N, 60N, 80N data) show that a decay of 10% of total tokens is sufficient, while 2.5% "falls short"; later experiments use about 10% (§4.2).

## Two-stage pre-training strategy (§5, Table 1)

- Proposal: use large coarse-quality data in the stable stage; in the annealing (decay) phase, mix "diverse and high-quality knowledge and ability-oriented SFT data" into the pre-training data (§5).
- Ablation (Table 1): A-1 = 2.4B intermediate stable checkpoint, decay on pre-training data only, then 4B-token SFT; A-2 = decay with high-quality unlabeled data and SFT data mixed in, then the same 4B-token SFT. B-1/B-2 = 1.2B last stable checkpoint, decay on pre-training data only, then 6B or 12B SFT tokens; B-3 = decay with high-quality + SFT data, then 6B SFT tokens.

| Run | C-Eval | CMMLU | MMLU | GSM8K | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|
| A-1 | 40.0 | 41.5 | 44.6 | 27.7 | 5.1 | 27.7 | 24.4 |
| A-2 | 52.6 | 51.1 | 50.9 | 42.3 | 5.4 | 30.4 | 30.3 |
| B-1 | 40.9 | 41.5 | 47.9 | 34.2 | 7.9 | 43.9 | 30.5 |
| B-2 | 41.2 | 42.0 | 47.9 | 34.4 | 7.3 | 43.9 | 29.8 |
| B-3 | 49.1 | 46.8 | 49.6 | 31.8 | 10.5 | 44.5 | 32.8 |

- Authors' conclusion: "the benefits of introducing high-quality data at the beginning of the decay stage are much higher than simply adding it during the SFT phase" and B-2 vs B-3 shows the gap is "not due to the insufficient training tokens in SFT stage" (§5). Note: B-3 GSM8K (31.8) is below B-1 (34.2). No seeds or variance reported.

## MiniCPM training stages (§6.2, §6.3, §6.4)

- Stable stage: around 1T tokens, WSD, batch 3.93 million tokens, max LR 0.01, Adam optimizer (§6.2).
- Decay stage: "a mixture of the pretraining data and high-quality SFT data"; exponential annealing with T "set to be 5000 steps (20B tokens)" (§6.2). The decay-stage mix adds UltraChat, SlimOrca, OssInstruct, EvolInstruct and proprietary "SFT" data (LeetCode, K12 textbooks and questions) (§6.3, Figure 11).
- SFT stage: "still necessary"; SFT data similar to the annealing data excluding pre-training data, about 6 billion tokens, LR aligned with the end of annealing (§6.2).
- The final decay checkpoint is not the one fine-tuned; the fine-tuned checkpoint is marked in Figure 12 (§6.4).
- Compute-optimal data-to-model ratio fit with WSD: about 192 tokens per parameter (§4.5).
- Not reported: checkpoint averaging.

## Used in

ch-32 §1.3 (WSD equation), §2.1 (SFT data in the decay stage), Recipe rows.
