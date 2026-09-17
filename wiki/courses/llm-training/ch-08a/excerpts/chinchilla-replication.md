---
chapter: ch-08a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2404.10102v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2404.10102
created_at: "2026-09-15"
---

# Excerpt: Chinchilla Scaling: A replication attempt

**Paper:** Tamay Besiroglu, Ege Erdil, Matthew Barnett, Josh You (Epoch AI). arXiv v1 2024-04; v2 2024-05-15 read. Source type: paper.

## Data reconstruction (§2)
- The authors extract points from Hoffmann et al. Figure 4 (left) by parsing the SVG: model size and training FLOPs from point positions, loss from point color on a 256-value log color scale from 2.00 to 5.00, giving loss precision of about 0.01 (§2).
- Five points near 10^19 FLOPs with token-to-parameter ratios below 0.4 have losses up to 70% higher than neighbors; the main fit excludes them ("the five outliers") (§2). The main dataset has 240 points; 245 with outliers (Table 1, Table 3).

## Refit of Approach 3 (§3, Eq. 1–4, Table 1)
- Same form L(N, D) = E + A/N^α + B/D^β and same Huber objective (δ = 10^−3) and initialization grid as Hoffmann et al. App. D.2 (Eq. 1–2).
- Refit (240 points): E = 1.8172, A = 482.01, B = 2085.43, α = 0.3478, β = 0.3658; a = β/(α+β) = 0.5126, bootstrap standard error 0.02 (Eq. 3, Table 1).
- Hoffmann et al. values from the TeX source comments: E = 1.6934, A = 406.4, B = 410.7, α = 0.3392, β = 0.2849, a = 0.454 (Eq. 4, Table 1).
- With the rounded values printed in the paper body (E = 1.69, α = 0.34, β = 0.28), residuals are not centered on zero; 98% of the refit's Huber losses are below Hoffmann et al.'s median, and a likelihood-ratio test gives p < 10^−235 (§3.1).

## Two causes (§3.1, §3.2)
1. Rounding: using β = 0.28 instead of 0.2849 adds a bias of about (10^11)^0.0049 − 1 ≈ 13% to the data term at D = 10^11 tokens (§3.1).
2. Optimizer stopping: with unrounded values the Hoffmann et al. fit is still worse by about 40–50 nats of log likelihood (Table 2: 837.78 versus 879.77 without outliers). One of the Hoffmann et al. lead authors (Borgeaud, 2024) confirmed that they averaged Huber losses instead of summing them, and the resulting loss scale made L-BFGS-B stop early, both in the fit and in bootstrapping (§3.1, §3.2).

## Confidence intervals (§3.2)
- Hoffmann et al. report a in 0.454–0.455; the refit's standard error 0.018 gives an 80% interval width of about 0.05, roughly 50-fold wider.
- An interval of width 0.001 would require about 240 × 2116 ≈ 600,000 training runs; Hoffmann et al. report "over 400" (§3.2).

## Implied allocation (§3.3, §4, Fig. 5, App. A.1)
- Hoffmann et al.'s Approach 3 parameters imply about 70 tokens per parameter, inconsistent with the 20 tokens per parameter used for Chinchilla 70B and with Approaches 1 and 2 (§3.3).
- The refit implies about 20 tokens per parameter, consistent with Chinchilla's training and Approaches 1–2 (§3.3). With the five outliers included, point estimates imply 25.6 tokens per parameter (App. A.1, Table 3).
- The refit is consistent with a tokens-to-parameters ratio between 4 and 40 for models trained on 10^26 FLOP or more (§4).
- The authors state that reporting exponents to a few significant figures "does not provide enough precision" to establish the relationship given the magnitudes of N and D (§4).

## Verification
- Read on 2026-09-15 against arXiv:2404.10102v2 PDF text (Abstract, §1–§4, App. A.1–A.2).
- Scope note: the refit uses data digitized from a figure, not the original loss values (§2).
