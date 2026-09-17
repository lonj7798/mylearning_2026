---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/john-schulman-kl-tricks.md
source_url: http://joschu.net/blog/kl-approx.html
primary_version: blog post dated 2020-03-07
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the blog text)"
---

# Excerpt: John Schulman, "Approximating KL Divergence" (2020-03-07)

The post estimates `KL[q, p] = E_{x~q}[log(q(x)/p(x))]` from samples `x ~ q` when `p(x)` and `q(x)` can be
evaluated but the sum over `x` cannot. With `r = p(x)/q(x)`:

| Estimator | Formula | Property stated in the post |
|---|---|---|
| `k1` | `−log r` | unbiased; high variance; negative for half of the samples |
| `k2` | `½(log r)²` | biased, low variance; its expectation is an f-divergence that matches KL to second order because `f''(1) = 1` for both |
| `k3` | `(r − 1) − log r` | unbiased (the control variate `r − 1` has zero expectation under `q`) and non-negative (`log x ≤ x − 1`) |

The post derives `k3` as `−log r + λ(r − 1)` with `λ = 1`, chosen so that the estimator is the vertical
distance between `log` and its tangent at `r = 1`, which is a Bregman divergence and therefore non-negative.
The same construction gives an estimator for the other direction: `KL[p, q] ≈ r·log r − (r − 1)`.

## The two numeric comparisons in the post
Samples are drawn from `q`; the code in the post uses `p = N(0, 1)` and `q = N(0.1, 1)` (the prose names the
two the other way round).

| True KL | Estimator | bias / true | stdev / true |
|---|---|---|---|
| 0.005 | k1 | 0 | 20 |
| 0.005 | k2 | 0.002 | 1.42 |
| 0.005 | k3 | 0 | 1.42 |
| 0.5 | k1 | 0 | 2 |
| 0.5 | k2 | 0.25 | 1.73 |
| 0.5 | k3 | 0 | 1.7 |

The post concludes for the second setting: "k3 has even lower standard deviation than k2 while being
unbiased, so it appears to be a strictly better estimator."

## What the post does not say
It makes no claim about RLHF practice, about which estimator any framework uses, about gradients of these
estimators when they are used as a training loss, or about an estimator "exploding" in TRL. Those claims were
in the previous version of this excerpt and of the library card; the gradient question is worked out in
ch-43 §6 from the definitions, and framework defaults come from [[entropy-logging-patterns]].
