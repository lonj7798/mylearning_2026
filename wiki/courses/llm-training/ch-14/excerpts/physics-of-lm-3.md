---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: Allen-Zhu & Li 2024 — "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws"
source_url: https://arxiv.org/abs/2404.05405
created_at: "2026-04-23"
---

# Excerpt: Allen-Zhu's 2-Bits-per-Parameter Knowledge Capacity

**Source:** `wiki/raw-data/llm-training/papers/physics-of-lm-3.md`
**Primary paper:** Zeyuan Allen-Zhu, Yuanzhi Li, "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws", 2024
**arXiv:** https://arxiv.org/abs/2404.05405

---

## Bibliographic header

The Physics of Language Models line studies LM behavior in controlled synthetic settings that isolate specific capabilities. Part 3.3 focuses on **knowledge-capacity scaling laws**: how much factual knowledge a model can store per parameter, as distinct from how much its cross-entropy loss drops per parameter.

From the raw-data notes:

> *"Scaling laws should track how much factual knowledge a model can store and retrieve, not only loss or benchmark score. When reasoning about data budgets, ask not only 'how much loss drops' but 'how much distinct knowledge the model can actually absorb.'"*

This is the paper you cite when someone claims loss is a sufficient measure of pretraining progress.

---

## The core result: 2 bits per parameter

The central empirical law, fit across model sizes from millions to billions of parameters and varying training budgets:

```math
K \approx 2 \cdot N \quad \text{bits of factual knowledge at saturation}
```

Informally: a transformer with `N` parameters, trained long enough on a factual corpus, stores about `2N` bits of distinct, retrievable facts. The law is *linear in N* and *independent of data volume past saturation*.

Concrete numbers:

| Model | Parameters `N` | Capacity `2N` bits | In GB |
|---|---|---|---|
| BERT-base | 110M | 220M bits | 27 MB |
| Llama 7B | 7B | 14B bits | 1.75 GB |
| Llama 70B | 70B | 140B bits | 17.5 GB |
| GPT-4 class (est.) | 1.8T | 3.6T bits | 450 GB |

A "fact" in the paper's setup is a structured tuple — e.g., `(entity, attribute, value)` such as `(Marie Curie, nationality, Polish)`. The paper constructs synthetic factual corpora with known ground-truth tuple counts, trains LMs, probes retrievability, and fits the capacity curve.

**Why this is load-bearing.** The law is a hard ceiling on what a model *can know*. If your deployment use-case requires the model to correctly answer 50B distinct factual queries, a 7B model cannot do it regardless of how much data you train it on. You need ≥ 25B parameters just for the storage to exist.

---

## The measurement protocol — why this is not hand-waving

From the raw-data notes:

> *"Uses factual knowledge representations rather than generic next-token loss alone. Measures how much knowledge survives storage and can be flexibly queried."*

The experimental setup:
1. **Generate synthetic fact corpus.** N_facts tuples drawn from a controlled generator with known vocabulary and relational structure.
2. **Train transformer.** Vary model size, training steps, exposure count per fact.
3. **Probe with held-out queries.** For each trained fact, test retrieval under several query formulations (not only the training form).
4. **Count retrievable facts.** The model "stores" a fact if it answers correctly under query perturbation, not only under the verbatim training phrasing.

This last point is what differentiates the "2 bits/parameter" result from a trivial memorisation bound. The facts must be *flexibly retrievable* — accessible through paraphrased queries — which rules out the model having merely memorised surface n-grams. What the model stores is a compressed, indexed representation of the fact, not a lookup table.

---

## Repetition as a factual-recall knob

The paper's second load-bearing result is the **repetition curve for rare facts**:

```math
P(\text{recall} \mid \text{fact seen } k \text{ times}) \approx 1 - \exp(-k / \tau)
```

where `τ` is the fact's "exposure time constant" — empirically 100–1000 for facts that appear in web-scale corpora, much smaller (τ ≈ 10) for facts embedded in obviously-factual context (Wikipedia-style infoboxes).

Two implications:

1. **Rare facts need repeated exposure, not just any exposure.** A fact seen once in 1T tokens is ~0% recoverable if τ = 300. Seeing it 100 times at τ = 300 gives ~28% recall; 1000 times gives ~96% recall.
2. **Rephrasing synthetic data works because it multiplies effective exposure.** [[rephrasing-the-web]] (WRAP) paraphrases documents to increase per-fact exposure without requiring fresh real data. The mechanism is exactly this: each paraphrase of "Marie Curie is Polish" counts as an independent exposure under the recall law.

This is a *separate* argument for repetition from Muennighoff's. Muennighoff says bulk repetition helps loss up to R_T ≈ 4 epochs. Allen-Zhu says specific-fact repetition helps recall up to ~τ exposures per fact. The two curves interact: bulk 2× repetition of a corpus doubles per-fact exposure. Past ~4 bulk epochs, you want *targeted* rephrasing rather than more bulk repetition.

---

## The saturation phase diagram

The paper maps out three regimes as a function of (N, D, τ_corpus):

1. **Under-parameterised (`D / τ > 2N`):** the corpus contains more facts than the model can store. The model is forced to compress, dropping rare or low-salience facts. This is the 2024+ frontier regime — every 7B model discards most of what it is trained on.
2. **Balanced (`D / τ ≈ 2N`):** rough match between training content and capacity. The model stores everything it sees "enough" times, but the stored set fills the model.
3. **Over-parameterised (`D / τ < 2N`):** the model has spare capacity. Loss keeps dropping but retention is already saturated. You are paying for parameters that are not doing retention work (they may be doing reasoning, but not storage).

The practical reading: for deployment models that must answer factual queries broadly, size `N` based on the union of facts you need to support (2 bits each). For reasoning-heavy models, retention matters less and you should size based on reasoning benchmarks, not fact recall.

---

## What this changes about data planning

Combined with Muennighoff (see [[excerpts/data-constrained-scaling]]):

```
# Muennighoff: effective tokens under repetition
D' = U · (1 − exp(−R / R_T)),    R_T ≈ 4

# Allen-Zhu: retention ceiling
K_max = 2 · N    bits

# Allen-Zhu: per-fact recall saturation
P(k) = 1 − exp(−k/τ),     τ ≈ 100–1000 for web-rare facts
```

The combined planning rule:

1. Pick `N` so `2N` covers the fact budget you need retained.
2. Pick `D` so `D' = U (1 − e^{−D/(U·R_T)})` is Chinchilla-optimal for that `N`.
3. Check that rare-fact exposure count `k` exceeds `τ` for the facts you care about; if not, synthesise paraphrases to lift `k`.

This is a meaningful tightening of the pure loss-based scaling law.

---

## Where the law breaks

- **Non-factual content.** Reasoning, style, procedural knowledge (coding, tool use) are not captured by the 2-bit/parameter fit. The law is specifically about stored *facts*.
- **Retrieval-augmented generation.** If the model queries an external KB at inference, the stored-fact count is nearly irrelevant. This is why retrieval-heavy deployments can use much smaller `N` for equivalent user-visible accuracy.
- **Pre-training vs post-training.** The law is measured on pretraining-style corpora. Whether post-training can add or erase stored knowledge at the same rate is an open question (preliminary evidence: SFT adds much less knowledge per token than pretraining).

---

## Connections

- Companion paper on repeat budget: [[excerpts/data-constrained-scaling]]
- Mechanism for multiplying per-fact exposure: [[rephrasing-the-web]]
- The contamination-induced retention degradation: [[excerpts/model-collapse]]
- Chapter synthesis: [[ch-14]]
