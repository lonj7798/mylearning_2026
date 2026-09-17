---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/john-schulman-kl-tricks.md
source_url: http://joschu.net/blog/kl-approx.html
revised_at: "2026-09-15"
---

# Excerpt: Approximating KL Divergence (Schulman, 2020-03-07)

Checked against the post on 2026-09-15. Used in read.md §4. The card's statements about practitioner reports on social media are not used.

## Definitions (samples x ∼ q, r = p(x)/q(x), estimating KL[q, p])
- k1 = log(q(x)/p(x)) = −log r: unbiased, "negative for half of the samples", high variance.
- k2 = ½(log r)²: biased, lower variance; its expectation is an f-divergence that matches KL to second order when p ≈ q.
- k3 = (r − 1) − log r: unbiased (control variate r − 1 has zero expectation) and always ≥ 0.

## Gaussian comparison (q = N(0,1))
| p | true KL | k1 bias/true, std/true | k2 bias/true, std/true | k3 bias/true, std/true |
|---|---|---|---|---|
| N(0.1, 1) | 0.005 | 0, 20 | 0.002, 1.42 | 0, 1.42 |
| N(1, 1) | 0.5 | 0, 2 | 0.25, 1.73 | 0, 1.7 |

- "k3 has even lower standard deviation than k2 while being unbiased, so it appears to be a strictly better estimator" (for the second setting).
- In RLHF with samples from the policy, q = π_θ and p = π_ref, so k1 = log π_θ − log π_ref.
