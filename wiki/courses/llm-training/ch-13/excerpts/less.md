---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/less.md
source_url: https://arxiv.org/abs/2402.04333
created_at: "2026-04-23"
---

# Excerpt: LESS — per-example gradient-similarity as the SFT-stage counterpart to DoReMi

**Source library:** `wiki/raw-data/llm-training/papers/less.md`
**Paper:** Xia, Malladi, Gururangan, Arora, Chen 2024, "LESS: Selecting Influential Data for Targeted Instruction Tuning" (ICML 2024 Spotlight).

---

## Why this source anchors ch-13

Ch-13's §3 comparison table lists LESS as the SFT-stage counterpart to DoReMi's pretraining-stage method. Both use the *gradient/loss signal of a small proxy* as a selector, but they operate at different stages and different granularities:

| | DoReMi | LESS |
|---|---|---|
| Stage | pretraining | SFT / targeted post-training |
| Granularity | domain (group) | example (individual sample) |
| Objective | worst-case excess loss | gradient similarity to target few-shot set |
| Needs downstream? | no | yes (few-shot exemplars) |

This excerpt unpacks the second column.

---

## The scoring function — LESS is influence-function approximation done right

From the source (lines 28-33):

> ## Scoring function (exact form)
> Let `g_i` be the projected gradient of training sample `x_i` under Adam-adjusted influence; let `g_{val}` be the averaged projected gradient over the few-shot target set.
> - Raw influence ≈ `<g_i, g_{val}> / sqrt(<g_{val}, g_{val}>)`.
> - LESS variant uses **cosine similarity** of L2-normalized Adam-adjusted gradients — numerically more stable.
> - Project to `d ≈ 8K` via fixed Gaussian random projection; preserves inner products (JL lemma).

Two moves make classical influence practical:

1. **Adam-adjustment.** Vanilla influence functions assume SGD. Under Adam, the effective step direction is divided by the square root of the second-moment accumulator. Naive influence over-credits samples whose gradients are large in absolute terms; LESS corrects by folding in Adam's denominator, producing per-sample "scoring gradients" that reflect the *actual* update the optimizer would have applied.
2. **Random projection to 8K dimensions.** Per-sample gradients on a 7B model are 7B-dimensional — storing one per sample is infeasible. LESS applies a fixed Gaussian random projection (Johnson-Lindenstrauss) to 8K dims. Inner products are preserved up to JL error, so cosine-similarity scores in the low-dim space approximate full-dim scores.

The result is a **datastore**: one projected, L2-normalized gradient vector per candidate sample, ~8K × 4 bytes = 32 KB per sample. 400K samples = 12.8 GB — fits on a single host.

---

## The selection pipeline

From the source (lines 34-43):

> ## Synthesis/selection pipeline (REQUIRED — be concrete)
> - **Seed input:** a pool of instruction data (typically 400K+ mixed sources) + a small target few-shot set representing the wanted capability.
>
> - **Selection step(s):**
>   1. **Warmup:** LoRA-train the target base model on the pool for ~4% of the full training budget (reads all data cheaply, stabilizes gradients).
>   2. **Gradient datastore build:** for each pool sample, compute the Adam-adjusted per-sample gradient, project to 8K dims, L2-normalize, store.
>   3. **Query vector:** for the target few-shot set, compute the same projected gradient, average, normalize.
>   4. **Select:** cosine-rank pool samples against query, keep top 5%.
>   5. **Train:** full fine-tune (not LoRA) on the selected 5%.

The datastore is **built once, queried many times**. One expensive warmup + gradient pass produces an artifact that can answer arbitrary "which 5% should I use for target capability X" questions, amortized across many targets.

The *warmup* step is non-negotiable. From the source (line 26): "the gradients only stabilize after ~1% of training; skipping warmup degrades selection." Raw random-init gradients are noise; ~4% of a full training budget stabilizes the gradient geometry enough for similarity to mean something.

---

## The transferability claim

From the source (lines 19-21):

> - Demonstrated **model-size and model-family transferability**: a 7B-model datastore selects useful data for 13B models and cross-family.

This is the SFT-stage analogue of DoReMi's 30× scale transfer. A 7B model's gradient geometry selects data that *also* improves a 13B model, and the selection transfers across families (Llama → Mistral). The mechanism is unclear — gradients are model-family-specific in principle — but the empirical result is clean.

For ch-13 this matters because it says "the cheap-proxy-for-expensive-production pattern works at both ends." DoReMi at pretraining (30× scale) and LESS at SFT (2× scale + cross-family) are both instances of a general principle: *selection signals computed on small models are robust enough to inform training on big ones.*

---

## Where LESS fails — and why DoReMi's failure mode is different

From the source (lines 57-62):

> ## Risks + gotchas
> - **Target-set dependence:** selection is only as good as the few-shot target exemplars.
> - **Adam-adjustment matters:** vanilla influence (SGD assumption) gives worse selections at LLM scale.
> - **Random-projection variance:** averaging multiple seeds stabilizes results.
> - **Quality gating not guaranteed:** LESS optimizes for similarity, not correctness — combine with answer verifiers for synthetic data.
> - **Compute non-trivial** — warmup + gradient datastore is not free; worth it when many selection targets will be served.

"Target-set dependence" is the failure mode that separates LESS from DoReMi. LESS *needs* a target few-shot set to define the capability. If the target set is biased or miscalibrated, LESS propagates that bias into the selected 5%. DoReMi avoids this because it has no downstream target — the worst-case excess loss is a target-free objective.

This is why ch-13 §3 lists LESS as "right tool when you have a targeted capability" and DoReMi as "right tool when you don't." They are not competitors; they answer different questions.

---

## The relationship to DSIR

Ch-13 §3's comparison table lists DSIR (Data Selection via Importance Resampling; Xie et al. 2023) alongside LESS. Both do "distribution-similarity selection," but at different levels:

- **DSIR**: score pool documents by `p_target / p_source` via n-gram hash features. Cheap, no gradients, fast on TB-scale pools. Right tool for "resample the web to match a reference corpus."
- **LESS**: score pool samples by gradient cosine-similarity to target exemplars. Expensive, needs model training, precise. Right tool for "select the 5% of SFT data that will most improve MMLU."

DSIR answers "similar text"; LESS answers "similar gradient direction." The second is closer to what the model actually learns from, which is why LESS wins when compute permits.

---

## Connections

- `[[less]]` — raw source.
- `[[ch-13]]` — §3 comparison table positions LESS as an SFT-stage mixer.
- `[[doremi]]` — pretraining-stage counterpart; same proxy-transfer pattern.
- `[[cherry-llm]]` / `[[ifd]]` — no-gradient per-example selectors; alternative to LESS.
- `[[prismatic-synthesis]]` — selects for gradient *coverage*; LESS selects for *alignment*.
