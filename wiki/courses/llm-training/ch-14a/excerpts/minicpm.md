---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2404.06395v3 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2404.06395
created_at: "2026-09-15"
---

# Excerpt: MiniCPM — WSD schedule, decay length, reuse across budgets, released schedule

**Report:** Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, et al. (Tsinghua University; Modelbest Inc.). arXiv v1 2024-04; v3 2024-06-03 read. Source type: official technical report.

## WSD definition (§4.2, Eq. 1)
```
WSD(T; s) = (s / W)·η          for s < W
          = η                  for W < s < T
          = f(s − T)·η         for T < s < S
```
W is the end step of warmup, T the end step of the stable stage, S the final step, η the maximum learning rate, and "0 < f(s − T) ≤ 1 is a decreasing function about s" (§4.2).

## Findings (§4.3, Fig. 5–6)
- Decay behaviour (0.036B models): "in the decay stage, as the learning rate begins to decrease, the loss experiences a significant rapid decline and quickly decreases to be equal to or lower than the Cosine LRS at step T = S" (§4.3).
- "10% Steps are Enough": from stable checkpoints at 40N, 60N, and 80N tokens (0.036B models), "having a decay of 10% of the total tokens is sufficient to achieve the best results, while a decay of 2.5% of total tokens falls short" (§4.3). N is the model's parameter count.
- Inference-compute rationale: "a 0.036B model can match the performance of a 0.17B model with an acceptable increase (∼ 4 times) in training compute while saving a lot of inference computation (Sardana & Frankle, 2023) (saving ∼ 5 times per inference call)" (§4.3, Fig. 6).
- Scaling-law measurement: with m model sizes and m data sizes, cosine runs cost "approximately O(m²)C"; WSD gives "linear cost (O(mC))" because decays branch from stable checkpoints "without re-training the models from scratch to different amounts of tokens" (§4.5).
- Fitted allocation: "the data size should be 192 times larger than the model size on average, as opposed to 20 times in Hoffmann et al. (2022)" (§4.5).

## Released models (Table 2, §6.2, §6.4)
- Table 2 ("N (B)" is the number of non-embedding parameters; the batch column is headed "Batch size (M)"): MiniCPM-1.2B, 1,247,442,432, batch 2M → 4M, 1.1T tokens; MiniCPM-2.4B, 2,442,057,984, batch 4M, 1.1T tokens.
- Stable stage: "We utilize around 1T data", "WSD LRS, with a batch size of 3.93 million and a max learning rate of 0.01" (§6.2). The model uses the Tensor Program width and depth scaling described in §3 ("Tensor Program ... proposes a framework to stabilize the hyper-parameters for models with different scales"), so its LR is set under that parameterization.
- Decay stage: pre-training data mixed with high-quality SFT data; exponential annealing `f(s − T) = 0.5^((s−S)/T)`, "in which T is set to be 5000 steps (20B tokens)" (§6.2, notation as printed).
- §6.4: "since we continue to SFT the model after the decay stage, we do not utilize the final checkpoints"; "The first drop in MiniCPM-1.2B is the result of enlarging batch size, which might have a similar effect as decreasing learning rate (Smith et al., 2017)."

## Verification
- Read on 2026-09-15 against arXiv:2404.06395v3 PDF text (§4.2–§4.5, §6.1–§6.4, Table 2).
- Not reported: seeds or run-to-run variance for the decay-length comparison; the step of the decay checkpoint that was fine-tuned; the unit of the batch-size column (tokens is not printed).
