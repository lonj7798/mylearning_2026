---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3 — four named mixes for four training stages

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Report:** Team Olmo 2025, "Olmo 3" (Allen AI).

---

## Why this source anchors ch-13

OLMo 3 is ch-13's strongest example of the claim that **mix is stage-specific**. The report names four corpora for four distinct training stages — Dolma 3 Mix, Dolmino, Longmino, Dolci — and publishes the composition of each. This is the cleanest public demonstration that "the mix" is plural, and it is the organizing structure of ch-13 §4's stage-specific mix table.

---

## The four-mix architecture

From the source (lines 44-50):

> ### Data curriculum
> - **Dolma 3:** about **9.3T** source tokens spanning web pages, science PDFs processed with `olmOCR`, code, math problems/solutions, and encyclopedic text.
> - **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens with stronger math/code emphasis and stronger decontamination.
> - **Dolma 3 Dolmino:** **100B** mid-training tokens sampled from a ~2.2T high-quality pool for math, science, code, instruction following, and reading comprehension.
> - **Dolma 3 Longmino:** about **50B** long-context tokens from a **639B**-token pool of long documents plus mid-training data.
> - **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

There are four training-stage mixes listed here (Dolma 3 Mix for pretraining, Dolmino for mid-training, Longmino for long-context extension, Dolci for post-training), plus the 9.3T source pool (Dolma 3) that feeds them. Each stage has a *different objective* and therefore a *different α*.

**Dolma 3 Mix → pretrain.** 5.9T tokens, broad coverage, mild tilt toward math/code. Objective: maximize broad capability with decent worst-case domain performance. This is the stage DoReMi was designed for.

**Dolmino → mid-train.** 100B tokens sampled from a 2.2T high-quality pool. The 2.2T → 100B distillation ratio (1 in 22) is itself a mix signal — Dolmino is an *aggressive* quality filter applied to the stage-2 data, not just a subset. Objective: install reusable priors for math, science, code, IFT, and reading comprehension so the post-training stage has something to build on.

**Longmino → long-context extension.** 50B tokens from a 639B long-document pool. Ratio 1:12.8. Objective: extend context without degrading short-context capability. The long-document subsample is paired with mid-training data, not used alone — which is itself a mix decision (mixing long docs with short mid-training text stabilizes training).

**Dolci → post-training.** Separate sub-mixes for SFT, DPO, and RLVR. This is the single most explicit acknowledgment of the ch-13 §4 claim that post-training stages each need their own mix.

---

## Ratios as mix information

The mix-information-dense reading of OLMo 3:

| Stage | Pool size | Used | Ratio | Implied α shape |
|---|---|---|---|---|
| pretrain (Dolma 3 Mix) | — | 5.9T | — | broad |
| mid-train (Dolmino) | 2.2T | 100B | 1:22 | highly selective |
| long-ctx (Longmino) | 639B | 50B | 1:12.8 | selective, long-doc biased |
| post-train (Dolci) | — | smaller | — | capability-bucketed |

The **selectivity ratio** is itself a mix lever. A 1:22 distillation (Dolmino) signals "only the top 4.5% of the high-quality pool survives" — an aggressive per-domain filter on top of whatever domain weighting is applied. The ratio is as important as the α vector for understanding what the stage's data looks like.

---

## The three-stage pretraining recipe

From the source (lines 39-43):

> ### Base-model training stages
> 1. **Initial large-scale pretraining** for broad text, code, and math coverage.
> 2. **Mid-training** on harder data distributions to sharpen programming, quantitative reasoning, and reading comprehension.
> 3. **Long-context extension** on very long documents.

Stage 1 is where DoReMi-style reweighting makes the most sense — large pool, stable per-domain loss signal, broad coverage objective. Stage 2 and Stage 3 are more naturally thought of as *curated cooldowns* than as group-DRO problems. OLMo 3's release confirms this by publishing hand-curated sub-pools (Dolmino, Longmino) rather than minimax-derived weight vectors.

For ch-13 §4's table, OLMo 3 fills in the "how do real labs do this" column:

- Pretrain mix: broad, publicly disclosed.
- Mid-train mix: aggressive selectivity from a quality pool, publicly disclosed.
- Long-context mix: long-doc subsample with mid-training leavening, publicly disclosed.
- Post-train mix: three sub-mixes (SFT/DPO/RLVR) in Dolci, publicly disclosed.

---

## What OLMo 3 does not tell you

The report is transparent about mix composition but less so about **how the mix was chosen**. The source does not say:

- Was any stage's α DoReMi-derived?
- What ablations produced the specific percentages in Dolma 3 Mix?
- How were the 100B Dolmino tokens sampled from the 2.2T pool — uniform, quality-classifier-reweighted, domain-stratified?

The release pattern is "we show you the mix, we don't show you the sweep." This is still a massive improvement over DeepSeek-V3's "14.8T diverse high-quality tokens" and is what ch-13 §5.3 celebrates — but it is not a full methodology disclosure.

---

## Why this matters for ch-13's learner

OLMo 3 is the artifact a learner should study if they want to *see* stage-specific mixing rather than only read about it. The jump from "we have a mix" to "we have four stage-specific mixes with different selectivity ratios" is the step the fixed-mix era (GLaM, PaLM) never took publicly. OLMo 3 takes it in a single release.

For the reader asking "should I use DoReMi at every stage?": OLMo 3's structure argues *no*. Pretrain → DoReMi or hand-tune. Mid-train → curate a high-quality sub-pool and subsample. Long-context → assemble a long-doc subset paired with short-context data. Post-train → split into SFT/DPO/RLVR mixes each with their own objective. Each stage needs a different tool because each stage has a different objective.

---

## Connections

- `[[olmo-3]]` — raw source.
- `[[ch-13]]` — §4 stage-specific table and §5.3 are built on this source.
- `[[olmo-2]]` — two-stage predecessor; OLMo 3 expands the stage count.
- `[[tulu-3]]` — supplies the Dolci post-training recipe.
- `[[dolma]]` — foundational data transparency worldview.
- `[[doremi]]` — stage-1 method; not explicitly attributed to OLMo 3 but conceptually aligned.
