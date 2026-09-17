---
chapter: ch-08a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2001.08361v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2001.08361
created_at: "2026-09-15"
---

# Excerpt: Scaling Laws for Neural Language Models

**Paper:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, et al. (OpenAI; Johns Hopkins University). arXiv v1 2020-01 (single version read). Source type: paper.

## Setup (§2, §3)
- Data: WebText2, 2.29 × 10^10 tokens after tokenization, 6.6 × 10^8 tokens reserved as test set; BPE vocabulary n_vocab = 50257; loss is cross-entropy in nats averaged over a 1024-token context (§1.3, §2, §2.3).
- Models: decoder-only Transformers from 768 to 1.5 billion non-embedding parameters; dataset sizes from 22 million to 23 billion tokens; batch 2^19 tokens for most runs (§3).
- Training: Adam (Adafactor for models above 1B) for a fixed 2.5 × 10^5 steps, batch 512 sequences of 1024 tokens, 3000-step linear warmup, cosine decay to zero; results at convergence were "largely independent of learning rate schedule" (§2.2).

## Quantities (§1.3, §2.1)
- N = number of parameters excluding vocabulary and positional embeddings; N ≈ 2·d_model·n_layer·(2·d_attn + d_ff) = 12·n_layer·d_model² with d_attn = d_ff/4 = d_model (Eq. 2.1).
- Forward-pass compute C_forward ≈ 2N + 2·n_layer·n_ctx·d_model; the context term is dropped when d_model ≫ n_ctx/12, giving C ≈ 6N FLOPs per training token (§2.1, Table 1).
- C ≈ 6NBS, B = batch size, S = steps; 1 PF-day = 8.64 × 10^19 FLOPs (§1.3).
- Using total parameters (including embeddings) obscures the trend; non-embedding N gives a single trend across depths (Fig. 6).

## Power laws (§1.2, Eq. 1.1–1.8; §4.2 Table 2)
- L(N) = (N_c/N)^α_N, α_N ≈ 0.076, N_c ≈ 8.8 × 10^13 (trained to convergence on large data) (Eq. 1.1).
- L(D) = (D_c/D)^α_D, α_D ≈ 0.095, D_c ≈ 5.4 × 10^13 tokens (large models, early stopping) (Eq. 1.2).
- L(C_min) = (C_c^min/C_min)^α_C^min, α_C^min ≈ 0.050, C_c^min ≈ 3.1 × 10^8 PF-days (Eq. 1.3). C_min is the compute needed at a batch size far below the critical batch size (§1.3).
- Doubling N multiplies loss by 2^−α_N = 0.95 (§1.2).
- Joint form L(N, D) = [(N_c/N)^(α_N/α_D) + D_c/D]^α_D (Eq. 1.5); the fit of the full form gives α_N = 0.076, α_D = 0.103, N_c = 6.4 × 10^13, D_c = 1.8 × 10^13 (Table 2); the fit fails for datasets reduced to about 2 × 10^7 tokens (§4.2).
- Overfitting depends on N^0.74/D: an 8× larger model needs about 5× more data to avoid a penalty (§1.1).
- Critical batch size B_crit(L) = B*/L^(1/α_B), B* ≈ 2 × 10^8 tokens, α_B ≈ 0.21 (Eq. 1.4).
- The values of N_c, D_c, C_c depend on vocabulary and tokenization "and hence do not have a fundamental meaning" (§1.2).

## Compute allocation (§6.1–6.3, Fig. 14)
- Fixed-batch fit: N = (1.6 × 10^9)·C^0.88; batch-adjusted fit: N = (1.3 × 10^9)·C_min^0.73, with C in PF-days (Fig. 14).
- N ∝ C_min^0.73, B ∝ C_min^0.24, S ∝ C_min^0.03 (Eq. 1.7–1.8, Eq. 6.1–6.2); data processed grows as D ∼ C^0.27 (§1.1).
- Each 10× increase in compute increases the optimal model size by 5× and the number of data examples processed by 2× (Fig. 14 caption).
- Models between 0.6× and 2.2× the optimal size can be trained with a 20% larger compute budget (Fig. 12).
- "Convergence is inefficient": the compute-efficient model is trained far short of convergence (§1.1, Fig. 2).
- The L(C_min) and L(D) trends intersect near C* ∼ 10^4 PF-days, N* ∼ 10^12 parameters, D* ∼ 10^12 tokens, L* ∼ 1.7 nats/token; the values are "highly uncertain, varying by an order or magnitude in either direction"; the laws must break down at or before this point (§6.3, Eq. 6.8).

## Transfer (§1.1, §3.2.2)
- Loss on other text distributions (Books, Common Crawl, Wikipedia, Internet Books) follows the WebText2 trend with a roughly constant offset (§1.1, Fig. 8).

## Verification
- Read on 2026-09-15 against arXiv:2001.08361v1 PDF text (Abstract, §1–§2.3, §3.1–3.2, §4.1–4.2, §6.1–6.3).
- Not reported by the source: downstream task accuracy scaling; allocation results with learning-rate schedules matched to each token budget (all runs share the 2.5 × 10^5-step schedule, §2.2).
