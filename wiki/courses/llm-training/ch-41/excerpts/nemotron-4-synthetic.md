---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md
source_url: https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
created_at: "2026-04-23"
---

# Excerpt: Nemotron-4 / HelpSteer2 — §5 of ch-41 uses for multi-attribute RMs

**Source library:** `wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md`
**Artifact:** 5-dimensional reward model (Helpfulness, Correctness, Coherence, Complexity, Verbosity)

---

## Why this source anchors ch-41

§5 exists because "one scalar per preference" is a compression choice, not a physical constant. Nemotron-4's HelpSteer2 reward model is the production-scale demonstration that (a) you can train K parallel heads on the same backbone from ~10K human labels, (b) you can compose them at RL time via `r_compose = Σ w_k · r_k` without retraining, and (c) this composability is what lets the same RM serve as a filter, judge, and RPO scorer across the whole pipeline.

---

## The five dimensions §5 quotes verbatim

From the source's Key Contributions and Synthesis Pipeline sections (lines 15, 20–22 and attested HelpSteer2 labels referenced throughout):

The five HelpSteer2 attributes are:

| Dimension | Labeling question |
|-----------|------------------|
| **Helpfulness** | Does the response address the request? |
| **Correctness** | Is the content factually right? |
| **Coherence** | Is the response well-structured and clear? |
| **Complexity** | Is the response appropriately detailed? |
| **Verbosity** | Is the response too long, too short, or right-sized? |

Each attribute gets its own scalar output head on the same base LM. Labels are 0–4 Likert ratings from human annotators. Ch-41 §5 keeps the names exactly as attested — the ontology matters; these labels *are* the reward specification.

---

## Why separate heads §5 defends as non-obvious

The source does not derive this; ch-41 §5 makes the argument explicit. A scalar BT RM trained on pairwise preferences will absorb *all* trade-offs into one number. Whatever the crowd labelers' implicit weighting was (usually "longer = better" unless told otherwise), that weighting is baked into the weights. Per-attribute heads invert the story: the trade-off weights `w_k` live *outside* the model and can be changed at RL time.

```
r_compose(x, y) = Σ_k w_k · r_k(x, y)
```

Want a safety-first policy? Upweight a safety head. Want concise answers? Downweight Verbosity's penalty — or even flip its sign to reward brevity. Nemotron uses this to disentangle verbosity from helpfulness, which scalar BT RMs systematically confuse.

---

## The pipeline §5 names — three roles for one RM

From the source (lines 18–22 and Synthesis Pipeline):

> Uses a reward model both as a filter and as a judge for preference ranking when ground truth is unavailable.
> Implements preference fine-tuning with DPO followed by RPO, with the reward model used to select higher-quality chosen responses.

Ch-41 §5 reports the three roles:

1. **Filter** on synthetic responses during alignment — drops low-scoring samples before SFT.
2. **Judge** for DPO pair construction — per-attribute scores pick the chosen/rejected.
3. **Scorer inside RPO** — reward-aware preference optimization uses the per-attribute scalars directly.

Each role wants a different weighting of the 5 dimensions. The composability is what makes one-RM-serves-all viable.

---

## The 98% synthetic ratio §5 contextualizes

From the source (line 15):

> over 98% of the training data for alignment is synthetic, while only about 20K human-annotated examples are used overall, split between SFT and HelpSteer2 reward-model data.

Ch-41 §5 uses this number to make a specific point: the 5-dim RM is the *bottleneck resource* that gates the synthetic pipeline. HelpSteer2's ~10K human-labeled pairs are the expensive substrate; everything else is synthetic generations scored by the resulting RM. If the 5 dimensions are wrong, the whole synthetic pipeline propagates the wrong reward spec.

---

## The DPO → RPO staging §5 flags

From the source (line 22):

> Implements preference fine-tuning with DPO followed by RPO, with the reward model used to select higher-quality chosen responses.

And (line 48):

> DPO alone can overfit to reward gaps; the paper adds SFT loss and then RPO to reduce that effect.

Ch-41 §5 notes this only briefly — the full DPO vs RPO discussion belongs to ch-44+. But the staging is part of why multi-attribute RMs earn their keep: different stages use different `w_k` weightings of the same RM.

---

## The open problem §5 does not hide

From the chapter body (echoing the source's emphasis on careful calibration):

> The attribute set is ontological — it encodes a specific theory of "what preference is." HelpSteer2's 5 dimensions are not universal.

Ch-41 §5 treats this as the *honest* caveat: picking the dimensions is the hard part, not training the heads. Other projects use safety, factuality, harmlessness, reasoning quality. The dimensions you pick *are* the reward specification, and there is no principled way to derive them — they come from product choices, not math.

---

## Connections to the rest of ch-41

- **§1** — 5 parallel BT losses with separate heads on a shared backbone.
- **§3** — implicitly ensembles over attributes, not seeds — complementary to Coste-style seed ensembling.
- **§4** — could be combined with GenRM (per-attribute critiques) though Nemotron uses scalar heads.
- **§6** — default for "multiple axes must be composed differently at RL time."
- **ch-23 / ch-29 (synthetic data)** — the RM is what filters and scores the synthetic stack Nemotron's pipeline produces.
- **ch-44+ (DPO / RPO)** — downstream consumers of per-attribute scores.
