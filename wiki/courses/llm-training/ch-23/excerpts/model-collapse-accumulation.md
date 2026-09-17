---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: primary source arXiv:2404.01413v2 (no library card as of 2026-09-15; card planned under slug model-collapse-accumulation)
source_url: https://arxiv.org/abs/2404.01413
created_at: "2026-09-15"
---

# Excerpt: Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data

**Paper:** Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Dhruv Pai, Henry Sleight, et al. (Stanford, Constellation, UMD, MIT). arXiv v1 2024-04; v2 2024-04-29 read. Source type: paper.

## Definitions (§2, Fig. 1)
- **Replace:** model n is trained only on data sampled from model n−1.
- **Accumulate:** model n is trained on the original real data plus all synthetic datasets from iterations 1…n−1.
- "Model collapse" here means test error that worsens notably over iterations; "avoiding collapse" means bounded error (§2). The theory section identifies collapse with test error diverging (§3, footnote 3).

## Language-model experiment (§2.1, App. C)
- GPT-2 (9M) and Llama-2 (12M, 42M, 125M/126M) pretrained from scratch for one epoch on TinyStories (470M tokens, GPT-3.5/4-generated).
- Each iteration samples a new dataset the same size as TinyStories from the previous model, at temperature 1.0 or 0.3; a newly initialized model is trained each iteration.
- Replace raises test cross-entropy for all architectures, sizes, and temperatures; accumulate gives equal or lower test cross-entropy (Fig. 2). Temperature 0.3 raises error faster under replace (App. Fig. 13).
- Table 2 (evaluation cross-entropy): GPT-2 9M t=1 1.82; t=4 accumulate 1.74; t=4 replace 2.39; t=10 replace 2.91; t=4 replace with dataset grown to the accumulate size 2.18. Temperature 0.3: t=4 replace 5.82, t=10 replace 9.85. Llama-2 126M: t=1 1.71; t=4 accumulate 1.59; t=4 replace 2.23.
- Ablation: replacing with a synthetic dataset grown to the accumulated size still degrades, at a lower rate (Table 2, right column).

## Other modalities (§2.2–2.3)
- GeoDiff on 40,000 GEOM-Drugs conformations, 8 iterations: test loss rises under replace and stays roughly constant under accumulate (Fig. 4).
- VAE on CelebA: replace collapses to a single mode; accumulate slows but does not stop the test-error increase, and minor attributes (glasses, accessories) stop appearing (§2.3, Fig. 5–6).

## Linear-regression theory (§3, Theorems 1–2, Eq. 3–4)
- Setting: x ~ N(0, I_d), y = x·w* + ε, ε ~ N(0, σ²); T samples per iteration; each iteration fits OLS and relabels the same X with the fitted model plus fresh noise.
- Replace (from Dohmatob et al. 2024a): E_test = σ²d/(T−d−1) × n.
- Accumulate: E_test = σ²d/(T−d−1) × Σ_{i=1..n} 1/i² ≤ σ²d/(T−d−1) × π²/6.
- Reason given: iteration i contributes a 1/i share of the data, so its noise enters the squared error with weight 1/i² (§3.2).
- "Halfway" (pure synthetic of size T×i): MSE grows as O(log n) (footnote 1, App. E).

## Limits stated by the authors (§4)
- Future work: fresh real data each iteration, different generation schedules, human filtering as in RLHF, deterministic (temperature 0) generation.
- At least four phenomena share the name "model collapse": unbounded test error, collapse to few modes, collapse to uniformity, amplification of artifacts (§4).

## Verification
- Read on 2026-09-15 against arXiv:2404.01413v2 PDF text (§1–4, App. B–C).
