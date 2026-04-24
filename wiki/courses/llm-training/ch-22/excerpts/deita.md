---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/deita.md
source_url: https://arxiv.org/abs/2312.15685
created_at: "2026-04-23"
---

# Excerpt: DEITA — complexity × quality × diversity

**Source library:** `wiki/raw-data/llm-training/papers/deita.md`
**Paper:** Liu et al. 2023 (ICLR 2024), "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning" (HKUST).

---

## Why this source anchors ch-22

DEITA is the first paper to argue that "quality" for SFT data is not one scalar. It decomposes selection into three orthogonal axes — complexity, quality, diversity — each scored by a trained component, and combines them with a *lexicographic* selection rule rather than a weighted sum. The result: a 6K-sample subset from a 300K-pool that fine-tunes Mistral-7B to match Zephyr-7B-beta (trained on 200K+) on MT-Bench.

For ch-22 §5 this is the flagship capability-agnostic recipe.

---

## The three axes — scorer construction

From the source (lines 29-37):

**Evol-Complexity scorer.**
- Take each seed instruction.
- Apply [[evol-instruct]]-style upward mutations: add constraints, increase reasoning depth, increase breadth.
- Ask ChatGPT to rank the variants by complexity.
- Distill the pairwise rankings into a 13B LLM with a scoring head.

**Evol-Quality scorer.**
- Same structure, quality-focused mutations (improve clarity, add detail, improve informativeness).
- ChatGPT ranks; distill into a second 13B scorer.

Note the scorers are *trained*, not prompted. DEITA pays a one-time scorer-training cost (ChatGPT labeling for the ranking data, 13B fine-tune for each scorer) and then scores the pool cheaply thereafter. This is why DEITA is more efficient than AlpaGasus at 300K+ pool sizes: AlpaGasus pays the ChatGPT call per sample forever; DEITA amortizes the ChatGPT cost into the scorer.

---

## The diversity constraint — not a scorer

From the source (lines 34-36):

> Score-first diversity-aware greedy selector: iterate top-scored samples, admit only if embedding distance > threshold to already-selected set.

Diversity is enforced as a *set-level constraint* during selection, not as a per-sample score. The selector loops:

```
selected = []
for sample in sorted(pool, key=evol_score, descending=True):
    if min(cos_dist(sample.emb, s.emb) for s in selected) > tau:
        selected.append(sample)
        if len(selected) == target_size: break
```

With `tau ≈ 0.9` and target_size = 6K.

---

## Why lexicographic, not weighted sum

From the source (lines 41-42):

> Final selection objective is a lexicographic combination: highest `complexity × quality` subject to the diversity constraint — *not* a weighted sum, because Liu et al. show pure combined-score without diversity fails.

The failure mode of `α·C + β·Q + γ·D`: you buy a little diversity in exchange for complexity and quality, and end up with a mediocre-everything set. Lexicographic says: never trade diversity for score. If a sample looks great but is too close to something already picked, skip it regardless of its score. This is the same structural choice [[prismatic-synthesis]] makes at the gradient-geometry level: diversity is a *constraint*, not a term.

---

## Axis ablations

From the source (lines 44-46):

- Remove diversity filter → score collapses; near-duplicates dominate the set.
- Remove complexity → reasoning benchmarks weaken.
- Remove quality → format compliance weakens.

All three are necessary. None subsumes the others.

---

## Headline result

From the source (lines 44-48):

- Pool: 300K (ShareGPT + UltraChat + WizardLM).
- Selected: DEITA-6K / DEITA-10K.
- Base model: Mistral-7B.
- Result: matched Zephyr-7B-beta (~200K SFT) on MT-Bench at release date.
- Transfer: same selection fine-tunes Llama / Mistral / Yi comparably.

The 6K-beats-200K headline is the strongest single data point for the "curation-beats-scale" thesis at the time of writing.

---

## What DEITA misses — the Prismatic reading

From the source (lines 49-54):

> Embedding diversity is surface-level: two samples requiring identical reasoning can live far apart in embedding space — [[prismatic-synthesis]] argues gradient-space diversity is a stricter objective.

This is the load-bearing ch-22 critique. DEITA's diversity axis is embedding-based; embedding-distance can disagree with gradient-distance by a lot. Two samples with cosine-distance 0.9 in a text-embedding model can induce near-identical gradient updates (same pattern, same reasoning shape), and DEITA will keep both; conversely DEITA can drop gradient-diverse samples that happen to be near in embedding space.

For general chat alignment, embedding-diversity is close enough. For OOD reasoning, it is not — see Prismatic's 300+ runs.

---

## Operational notes

- **Scorer-training data labeling** is the bulk cost; it is paid once. For a new pool in a new domain, re-label.
- **τ = 0.9** is dataset-specific — sweep it.
- **Target size 6K vs 10K** — DEITA releases both; DEITA-6K is the SOTA claim, DEITA-10K is slightly safer.

---

## Connections

- **[[ch-22]]** §5 — the three-axis slot.
- **[[instag]] / [[instag-diversity]]** — the tag-space predecessor to DEITA's complexity axis.
- **[[cherry-llm]] / [[ifd]]** — cheaper single-axis alternative.
- **[[alpagasus]]** — the single-axis (quality-only) predecessor DEITA subsumes.
- **[[less]]** — alternative gradient-based; different geometry.
- **[[prismatic-synthesis]]** — supersedes DEITA on the diversity axis via gradient entropy.
- **[[tulu-3-sft-mix]]** — consumes DEITA-lineage recipes in the modern open SFT stack.
