---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/doremi.md
source_url: https://arxiv.org/abs/2305.10429
created_at: "2026-04-23"
---

# Excerpt: DoReMi — proxy-model group DRO for pretraining mix

**Source library:** `wiki/raw-data/llm-training/papers/doremi.md`
**Paper:** Xie et al. 2023, "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining" (NeurIPS 2023).

---

## Why this source anchors ch-13

DoReMi is the pivot paper in the chapter. The fixed-mix era (GLaM, PaLM) treated α as a hyperparameter to sweep; DoReMi treated α as the solution to a minimax game. Ch-13 uses this source for three load-bearing claims:

1. The **minimax/group-DRO objective** (§2 of read.md) is a direct transcription of DoReMi's equation — excess loss, not raw loss, is the adversary's payoff.
2. The **exponentiated-gradient α update** (§2.1) derives from DoReMi's algorithm — not projection, not Euclidean gradient ascent; entropic mirror descent.
3. The **30× proxy-to-production transfer** (§2.3 and §2.4) is the empirical result that makes DoReMi cheap enough to run at all.

---

## The core result — 30× scale transfer

From the source (lines 49-52):

> **Key empirical findings:**
> - 280M proxy transfers to 8B (30× scale) — scale-robust weights.
> - On The Pile, all 22 per-domain perplexities improved vs baseline — DRO is not just reshuffling loss.
> - Works even better when the reference is itself trained with uniform weights (as opposed to prior iteration of DoReMi).

The 30× gap is the operational justification for ever running DoReMi. Before DoReMi, labs hand-swept α with ~8B-scale ablations and paid for a grid of full pretraining runs. DoReMi replaces the grid with one 280M-scale minimax run. At Chinchilla ratios, a 280M model consumes ~5.6B tokens; an 8B model consumes ~160B tokens for the same flop-optimal target. The ratio is ~30× in compute, matching the parameter-count ratio.

**What transfers.** Not the proxy's weights — those stay with the proxy. Only α transfers. The production model starts from scratch with the proxy-derived α baked into the data sampler.

**What does not transfer.** The proxy's loss landscape, the proxy's optima, the proxy's calibration. The claim is narrower than it sounds: only the *ordering* of per-domain difficulty (and therefore the softmax of accumulated excess loss) is scale-invariant enough to cross the 30× gap.

---

## The excess-loss construction — why reference models matter

From the source (lines 32-35):

> - Reference model: a small LM pretrained with uniform domain weights, used to compute per-domain loss baselines.
> - Proxy model: a 280M-param LM trained with Group DRO against the reference.

The reference is the hidden lever that makes DoReMi work. Group DRO on raw loss would concentrate α on the highest-entropy domain (math, code) because those have the largest absolute loss, and the adversary would starve every other domain. The reference lets the adversary measure *relative* underperformance: "how much worse is the proxy doing on domain k than a fair-share training would achieve?"

This is the insight that distinguishes DoReMi from generic DRO. The formal objective (from the source):

```
min_θ  max_{α ∈ Δ}  Σ_k α_k · ( L_k(θ) − L_k(θ_ref) )
```

`L_k(θ_ref)` is subtracted per-domain. The adversary only sees the *deviation* from the baseline, which normalizes across intrinsic difficulty. Ch-13 §2 walks through why this one substitution is the key to avoiding "DRO gets stuck on math" mode.

**Two common mistakes when reimplementing DoReMi:**
- Using a task-tuned reference. The source notes (line 52) that DoReMi works best with a uniform-weight reference. A task-tuned reference makes the excess-loss signal noisy because the reference already over-learns the downstream-relevant domains; there is no room for the proxy to improve.
- Forgetting to average α. The source (line 47): "take the final α from proxy training (optionally averaged over last k steps)." The adversary oscillates; the time-average is the equilibrium strategy. Last-iterate α is known to be unstable in two-player games — Freund-Schapire's averaging argument applies.

---

## The Pareto result — all domains improve

From the source (line 51):

> - On The Pile, all 22 per-domain perplexities improved vs baseline — DRO is not just reshuffling loss.

This is the single most surprising result in the paper and ch-13 §2.4 dedicates a full section to it. If DoReMi downweights a domain, conventional wisdom says that domain's perplexity should *worsen*. It does not.

Two mechanisms explain the Pareto improvement:

1. **Domain overlap.** The Pile's 22 domains have substantial distributional overlap (CommonCrawl, OpenWebText2, Books3, Project Gutenberg share vocabulary and genre). Downweighting one overlapping domain leaves the model seeing near-equivalent text under another label — per-domain perplexity measured on held-out domain `k` is evaluated on the model, not on the training-time label.
2. **Capacity reallocation.** Upweighting hard domains (arXiv math, GitHub code) produces better general-purpose representations that transfer back to easy domains. The capacity freed from memorizing redundant web text is spent on structural knowledge (syntax of formal languages, mathematical notation) that improves everywhere.

For ch-13's teaching payload, this is the strongest argument against hand-tuning: humans would never propose "downweight Wikipedia to improve Wikipedia perplexity," which is precisely what DoReMi sometimes discovers.

---

## What DoReMi does not tell you

The source is explicit about what lies outside DoReMi's scope:

- **No downstream task knowledge is used.** This is a feature for generalization but a limitation for labs that already have target downstream evals — DoReMi cannot directly optimize for "improve MMLU." The source (line 21) notes it still *matches* task-tuned GLaM weights on GLaM's corpus, but this is a claim about the Pareto frontier, not about beating targeted tuning.
- **Domains must be pre-partitioned.** DoReMi does not discover domains from raw text; it accepts a disjoint `K`-way partition as input. The partition is a modeling choice that controls what "group" means in group DRO. Too-fine partition (1000 domains) produces noisy excess-loss estimates; too-coarse (3 domains) loses resolution.
- **Transfer across stages is untested.** DoReMi was demonstrated on pretraining mixes. Using DoReMi to set SFT or RLVR mixes is outside the paper's scope — and as ch-13 §4 argues, per-example selectors ([[less]]) are the better tool at those stages.

---

## Connections

- `[[doremi]]` — raw source this excerpt extracts from.
- `[[ch-13]]` — §2 derives the EG update; §2.4 unpacks the Pareto result.
- `[[less]]` — the per-example gradient-similarity analogue for SFT.
- `[[scaling-laws-data-quality]]` — why DoReMi's per-domain effective sample size interacts with mix.
- `[[llama-3]]` / `[[olmo-2]]` — labs that built on DoReMi's conceptual frame without always disclosing whether DoReMi was the actual method used.
