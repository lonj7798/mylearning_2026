---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: Team Olmo 2025 — "OLMo 3" (data curriculum section) + Walsh et al. 2025 — "OLMo 2"
source_url: https://arxiv.org/abs/2512.13961 ; https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
---

# Excerpt: OLMo 2 / OLMo 3 Decontamination and Two-Stage Pretraining

**Source:** `wiki/raw-data/llm-training/model-reports/olmo-2.md`, `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Primary papers:**
- Pete Walsh et al., "OLMo 2: 2 OLMo 2 Furious", 2025 (arXiv:2501.00656)
- Team Olmo, "OLMo 3", 2025 (arXiv:2512.13961)

---

## Bibliographic header

OLMo 2 and OLMo 3 are the most transparent open-model pretraining runs of 2025. For ch-14 topics, they are the best-documented examples of:
- Two-stage pretraining (bulk + cooldown) as a data-constrained-scaling adaptation.
- Per-stage decontamination with different thresholds.
- Released decontamination tooling (OlmoTrace).

From the OLMo 3 raw-data notes:

> *"Uses a clear three-stage base training recipe: pretraining, mid-training on harder distributions, and long-context extension. Makes the data curriculum explicit: Dolma 3, Dolma 3 Mix, Dolmino, Longmino, and Dolci."*

Every frontier lab runs something like this recipe; OLMo is unusual in publishing the datasets and the tooling.

---

## The two-stage budget — OLMo 2 and the Dolmino cooldown

OLMo 2 (7B / 13B / 32B):

- **Stage 1 data:** OLMo-Mix-1124 — ~3.9T tokens from DCLM, Dolma 1.7, Starcoder, Proof Pile II.
- **Stage 2 cooldown data:** Dolmino mix — ~50B higher-quality tokens, curated from a pool ~10× larger.
- **Context:** 4K native, extended to 32K in cooldown.

This is the Muennighoff formula (see [[excerpts/data-constrained-scaling]]) applied deliberately: bulk training runs at `R ≈ 1` on the 3.9T corpus; cooldown runs at `R ≈ 5–10` on the 50B high-quality corpus. The cooldown stage is where repetition is engaged, not the bulk stage.

**Why the split.** Cooldown data is more expensive to curate (heavier decontamination, classifier filtering, manual review), so you have less of it. You run more epochs on it because (a) Muennighoff's decay allows it at this scale, and (b) the quality asymptote `E(q)` is better, so per-epoch gains are larger. This is the practical composition of Muennighoff + Subramanyam formalism.

---

## OLMo 3's expanded curriculum

OLMo 3 (7B / 32B) — the data budget grew:

- **Dolma 3 source pool:** ~9.3T tokens (raw)
- **Dolma 3 Mix (pretraining):** ~5.9T tokens after stronger decontamination and filtering
- **Dolmino (mid-training):** 100B tokens from a ~2.2T high-quality pool
- **Longmino (long-context extension):** 50B tokens from a 639B-token pool of long documents
- **Dolci (post-training):** separate SFT / DPO / RLVR mixes

From the raw-data notes:

> *"The real scientific artifact is not just the final model weights but the entire model flow: pretraining stages, intermediate checkpoints, datasets, post-training branches, evals, and tooling."*

**Notice the decontamination pattern:** each stage's corpus is a filtered subset of a larger pool. Dolma 3 Mix is 5.9T out of 9.3T raw — 37% drop rate. Dolmino is 100B out of 2.2T — 95.5% drop rate. The *later* the stage, the more aggressive the filtering; the more aggressive the filtering, the smaller the corpus, the more epochs it sees.

---

## The OlmoTrace tool — open decontamination infrastructure

OLMo 3 releases `OlmoTrace` as part of the tooling package:

From the raw-data notes:

> *"Couples model release with tooling: Olmo-core, Open Instruct, OLMES, OlmoTrace, decontamination, and dedup utilities."*

OlmoTrace is the open counterpart to Meta's closed Llama 3 decontamination pipeline. It is essentially:

1. Eval-set n-gram index (Bloom filter per eval, 13-gram default for bulk, 8-gram for cooldown).
2. Streaming document scanner with per-document overlap-rate computation.
3. Decontamination audit log — every dropped document is recorded with the triggering eval and the overlap fraction. Makes the decontamination reproducible.
4. Provenance tracking: documents carry metadata about source URL, scrape date, filter stage decisions.

**The `13-gram for bulk, 8-gram for cooldown` split is load-bearing.** Bulk data sees each document once at low LR; an occasional 8-gram match with an eval is unlikely to install that sample into the model's memory. Cooldown data sees each document many times at higher effective learning rate (Allen-Zhu retention regime), so even a single leaked 8-gram could install an eval answer. The tighter threshold is justified by the repetition count.

This distinction — **decontamination threshold as a function of stage-specific repetition count** — is one of the subtle pieces of 2025-era practice that is not in the original Muennighoff paper but falls out of combining it with Allen-Zhu retention.

---

## The data-constrained reality at OLMo 3 scale

OLMo 3 has 9.3T source tokens but uses only 5.9T for pretraining. The 37% drop rate breaks down roughly as:

- Exact-duplicate dedup: ~10%
- Near-duplicate / MinHash: ~15%
- Low-quality classifier filtering: ~8–10%
- Eval decontamination: < 1%
- Provenance / policy filtering: ~2–3%

**The decontamination step is *not* the dominant drop cause.** Dedup and quality filtering eat most of the dropped tokens. This is a common misreading — "decontamination is expensive" is false; "dedup + quality filtering is expensive" is true, and decontamination is a small additional step on top.

The practical cost of decontamination is **operational**, not in data yield:
- You must enumerate every eval you care about ahead of time.
- Eval leakage discovered later requires re-running the pipeline.
- Bloom-filter false positives are tolerable; false negatives are catastrophic (the eval is burned).

---

## What this teaches about data-constrained scaling in practice

Triangulating OLMo 2 + OLMo 3 + Llama 3:

| Stage | Tokens seen | Epochs | Decontam strictness | Role |
|---|---|---|---|---|
| Bulk pretraining | 4–15T | ~1 | looser (13-gram, τ=0.5) | load the model with broad distribution |
| Mid-training / cooldown | 50–100B | 2–10 | strict (8-gram, τ=0.1) | sharpen on high-quality |
| Long-context extension | 50B | 1–5 | strict | add positional range |
| SFT | 10–100M prompts | 1–3 | strict (gold-standard eval) | align to instructions |

Every stage sits at a different point in the Muennighoff / Subramanyam / Allen-Zhu formula space. The 2025 recipe does not treat pretraining as a monolithic run — it is a curriculum where each sub-stage has distinct quality, repetition, and contamination parameters.

---

## Connections

- The scaling law that OLMo-style curricula implement: [[excerpts/data-constrained-scaling]], [[excerpts/scaling-laws-data-quality]]
- The closed-lab counterpart: [[excerpts/llama-3-decontamination]]
- Cooldown-stage synthetic injection (next chapter territory): [[rephrasing-the-web]], [[prismatic-synthesis]]
- Chapter synthesis: [[ch-14]]
