---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reward-ensembling.md
source_url: https://arxiv.org/abs/2310.02743
created_at: "2026-04-23"
---

# Excerpt: Coste 2023 — ensembling §3 of ch-41 uses as Goodhart insurance

**Source library:** `wiki/raw-data/llm-training/papers/reward-ensembling.md`
**Artifact:** mean / LCB / min / UWO aggregators over K BT RMs

---

## Why this source anchors ch-41

§3 is the cheap, direct answer to §2's inverted-U. Coste 2023 asks: if the proxy has bounded generalization, average K proxies. The result — peak shifts from `d ≈ 3` to `d ≈ 5–8` — is the single most cost-effective intervention on the overoptimization curve, which is why §6's decision framework has "ensemble it" as a *modifier* applicable on top of every RM choice.

---

## The four aggregators §3 enumerates

From the source (lines 18–22):

> *Mean:* `r(x,y) = (1/K) Σ_k r_k(x,y)`.
> *LCB:* `r(x,y) = mean_k r_k − λ · std_k r_k` (pessimistic under disagreement).
> *Min:* `r(x,y) = min_k r_k(x,y)`.
> *UWO (uncertainty-weighted objective):* reward minus a penalty on std.

Ch-41 §3 reports all four with a single trade-off rule: mean gives the highest peak, LCB and min are safer but cap lower. Pick LCB for safety-critical deployments; pick mean when you can afford to ride closer to the edge of the KL budget.

---

## Variance reduction intuition §3 formalizes

The source does not write this out; ch-41 §3 derives it explicitly: for K iid RMs each with error variance `σ²`, the mean has variance `σ² / K`. The proxy-vs-gold gap at fixed `d` scales with RM error, so halving variance (K = 4) pushes the peak right by a constant on the `d` axis. This matches the empirical peak-shift from `d ≈ 3` to `d ≈ 5–8` for K = 3–5.

The interactive figure [figures/rm-overopt.html](../figures/rm-overopt.html) models this as `β_eff = β₀ / sqrt(K)`. Slide K from 1 to 10 and watch the peak drift right.

---

## The empirical peak shift §3 quotes

From the source (line 23):

> peak moves from `d ≈ 3` (Gao baseline) to `d ≈ 5–8` depending on K.

This is the exact number ch-41 §3 embeds in the chapter body. Diminishing returns past K = 5 — don't bother with K = 10 unless RM-forward compute is free.

---

## The counterexample §3 refuses to skip

From the source (line 25):

> if all RMs are systematically miscalibrated in the same direction (shared label noise, shared blind spot), ensembling does not help — demonstrated on adversarial prompts.

Ch-41 §3 promotes this to an operational rule: *ensemble diversity must come from data shards, not just random seeds*. Seed-only diversity leaves all K RMs exposed to the same dataset biases; only data-shard diversity hedges against shared blind spots. This is why Nemotron-4's HelpSteer2 pipeline ([[nemotron-4-synthetic]]) trains per-attribute heads on different subsets rather than just reshuffling seeds.

---

## Disagreement as OOD signal §3 surfaces

From the source (line 37):

> **Disagreement as signal:** `std_k r_k` correlates with OOD-ness of the response; can be used as an anomaly flag.

Ch-41 §3 lifts this into a diagnostic: surface `std_k r_k` during RL even when you are not using LCB as the reward. High std means the policy has drifted into novel territory — the same territory where §2's proxy-gold gap blows up. This is the quickest in-loop signal that you are approaching the peak.

---

## The overhead tax

From the source (line 38):

> K RMs roughly multiply the reward-forward-pass cost by K during RL; often affordable since the RM is smaller than the policy.

Ch-41 §3 notes this is only true while RMs are small. For 70B-scale RMs (increasingly common with GenRMs — [[generative-reward-models]]), K × forward cost stops being free. This is why §4's GenRM line and §3's ensemble line are not a strict upgrade — they trade off against each other on a compute budget.

---

## Connections to the rest of ch-41

- **§2** — the direct defense against [[reward-model-overoptimization]]'s law.
- **§4** — complementary to GenRM ensembles; `std_k r_k` becomes calibrated uncertainty with generative heads.
- **§5** — HelpSteer2's 5-dim RM is implicitly an ensemble over *attributes*, not over seeds.
- **§6** — "ensemble it" is a modifier on every decision-framework row, not a row itself.
