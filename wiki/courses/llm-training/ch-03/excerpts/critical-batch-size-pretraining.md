---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: primary source arXiv:2410.21676v4 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2410.21676
created_at: "2026-09-15"
---

# Excerpt: How Does Critical Batch Size Scale in Pre-training?

**Paper:** Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, Sham Kakade
(Harvard; Kempner Institute; UC Berkeley; HKU; Amazon). arXiv v1 2024-10; v4 2025-04-21 read; ICLR 2025. Source
type: paper.

## Setting (§2.1)
- Autoregressive LMs of 85M, 151M, 302M, 604M, 1.2B parameters, context length 512, trained on C4 with Adam;
  gpt-neox-20b tokenizer (vocabulary 50280). 151M models are used for most ablations.
- Batch size is counted in sequences of 512 tokens unless stated (Fig. 5 caption: "context length is 512 by default").
- Training beyond fixed durations: constant LR plus exponential weight averaging (EWA), ξ_{t+1} = τ·ξ_t + (1 − τ)·θ_t,
  evaluated on ξ; it "can match the efficiency of cosine scheduling and WSD, especially for large batch sizes"
  (§2.2, Fig. 2).

## Definition (§3.1, Definition 1)
- R_opt(N, D) = min over B of the best loss for model size N on D tokens; f_{N,D}(B) = steps to reach R_opt at batch B;
  linear-scaling reference f*(B) = D/B matched at B_opt.
- Critical batch size B*(N, D) = the largest B′ > B_opt with f(B′) ≤ 1.2·f*(B′), i.e. "the batch size that leads to a
  20% overhead compared to linear scaling". "20% can be replaced by any other suitable measure."

## Fits (§3.2–§3.3)
- Chinchilla setting (target loss at a Chinchilla step count for each size): `B* = 93.20 · N^0.47`, N in millions
  (§3.2, Fig. 1 left). The authors state CBS "around 2^9 to 2^11" is helpful for models below 1B on
  Chinchilla-optimal tokens.
- Fixed data size (3.072B tokens, the Chinchilla token count of the 151M model): "almost no increase in CBS when
  enlarging the model size"; `B* = 621.341 · N^0.087` (§3.3, Fig. 1).
- Fixed model size (302M) with targets at 0.28×, 0.5×, 2×, 4× the Chinchilla step: CBS increases with tokens (§3.3,
  Fig. 1, Fig. 6).
- Takeaway: "the increase in CBS in Chinchilla-optimal training is more strongly attributed to extended data size or
  training durations rather than the increase of model size" (§3.3).
- Theory: in infinite-width limits the CBS stays nearly invariant with width; for least-squares with mini-batch SGD,
  B*(D) = Θ(D^c) with 0 < c < 1/2 in the variance-dominated regime (§1.2, §4).

## Ablations (§2.3–§2.4, App. B)
- Context lengths 2^9 to 2^12 give "similar scaling w.r.t. batch size" (§2.3, Fig. 4).
- Scaling a 151M model to 604M by width or by depth gives overlapping efficiency curves (§2.4, Fig. 3).
- β1 = 0.95 "consistently performs well"; β2 = 0.95 helps large-batch short runs, while larger β2 (0.99, 0.999,
  0.9995) helps long-duration training and "substantially improves small batch size training" (App. B takeaways).

## Derived values (not printed as such)
Chinchilla-setting fit: N = 151 gives B* ≈ 985 sequences (≈ 5.0e5 tokens); N = 1200 gives ≈ 2,610 sequences (≈ 1.34e6
tokens). Fixed-data fit: N = 151 gives ≈ 961; N = 1200 gives ≈ 1,151.

## Verification
- Read on 2026-09-15 against arXiv:2410.21676v4 PDF text (Abstract, §1–§4, App. B takeaways).
- Not reported: models above 1.2B parameters; data other than C4; downstream task evaluations.
