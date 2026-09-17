<!-- excerpt for: ch-32e
     source: Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", arXiv:2404.06395v3 (2024-06-03; v1 2024-04)
     scope: WSD schedule definition, decay-length finding, SFT data in the decay stage, MiniCPM training stages
     library card: none as of 2026-09-15 (proposed slug minicpm)
-->

# MiniCPM decay stage — verified extract

Checked on 2026-09-15 against the arXiv v3 PDF text.

## WSD definition (§4.2, Eq. 1)
WSD(T; s) = (s / W) η for s < W; η for W < s < T; f(s − T) η for T < s < S, "where 0 < f(s − T) ≤ 1 is a decreasing function about s, η is the maximum learning rate". W is the end of warmup, T the end of the stable stage, S the end of training.

## Decay-length finding (§4.2, Figure 5)
- On 0.036B models, loss "experiences a significant rapid decline" once the LR starts to decrease, and reaches or goes below the cosine schedule's loss at T = S.
- "among all three stable training checkpoints in 40N, 60N, and 80N training data, having a decay of 10% of the total tokens is sufficient to achieve the best results, while a decay of 2.5% of total tokens falls short." The authors then use "a decay of about 10%".

## SFT data in the decay stage (§5, Table 1)
- Proposal: pretraining uses only large-scale coarse-quality data; "During the annealing phase, we use diverse and high-quality knowledge and ability-oriented SFT data, mixed into the pre-training data."
- Table 1:

| Run | Setting | C-Eval | CMMLU | MMLU | GSM8K | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|---|
| A-1 | 2.4B, decay on pretraining data only, then 4B-token SFT | 40.0 | 41.5 | 44.6 | 27.7 | 5.1 | 27.7 | 24.4 |
| A-2 | 2.4B, decay with high-quality + SFT data mixed in, then 4B-token SFT | 52.6 | 51.1 | 50.9 | 42.3 | 5.4 | 30.4 | 30.3 |
| B-1 | 1.2B, decay on pretraining data only, then 6B-token SFT | 40.9 | 41.5 | 47.9 | 34.2 | 7.9 | 43.9 | 30.5 |
| B-2 | 1.2B, decay on pretraining data only, then 12B-token SFT | 41.2 | 42.0 | 47.9 | 34.4 | 7.3 | 43.9 | 29.8 |
| B-3 | 1.2B, annealing with high-quality + SFT data, then 6B-token SFT | 49.1 | 46.8 | 49.6 | 31.8 | 10.5 | 44.5 | 32.8 |

- A-runs start from an intermediate stable-stage checkpoint of MiniCPM-2.4B; B-runs from the last stable-stage checkpoint of MiniCPM-1.2B. Seeds and number of runs are not stated.

## MiniCPM training stages (§6.2-6.4, Table 2)
- Stable stage: "around 1T data", WSD, batch 3.93 million, max LR 0.01 (§6.2). Table 2: 1.1T total tokens for both sizes; batch 4M (2.4B) and 2M → 4M (1.2B).
- Decay stage: "a mixture of the pretraining data and high-quality SFT data. For the specific annealing form of the WSD scheduler, we employ exponential annealing, i.e. f(s − T) = 0.5^((s−S)/T), in which T is set to be 5000 steps (20B tokens)." (printed as shown; see note).
- Decay-stage data adds UltraChat, SlimOrca, OssInstruct, EvolInstruct and proprietary "SFT"-suffixed data such as LeetCode and K12 questions (§6.3, Figure 11).
- SFT stage: SFT data similar to the annealing phase without pretraining data, about 6B tokens, starting LR equal to the end-of-annealing LR, WSD with exponential decay (§6.2).
- "since we continue to SFT the model after the decay stage, we do not utilize the final checkpoints" of the decay stage (§6.4).

Note on the printed decay formula: with s < S the exponent (s − S)/T is negative, which gives f > 1 and contradicts 0 < f ≤ 1 in Eq. 1. A reading consistent with Eq. 1 is a half-life of 5000 steps counted from the start of decay (Interpretation, not stated by the authors).
