---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3 — the model-flow worldview as a lineage requirement

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Paper:** Team Olmo (Allen AI) 2025, "OLMo 3".

---

## Why this source anchors ch-11

OLMo 3 is the paper that makes the **lineage-as-release-artifact** thesis explicit. Dolma gave the toolkit; FineWeb gave the scale; OLMo 3 gives the full *model flow*: Dolma 3 → Dolma 3 Mix → Dolmino → Longmino → Dolci, each with its own filter gates and its own tokenizer-consumer assertions. For ch-11's thesis ("curation decides what enters the model; operations decide whether you can ever build the same model again"), OLMo 3 is the cleanest worked example in the public literature.

Ch-11 cites OLMo 3 three times: for the model-flow architecture (§3), for the 1024-GPU / 8× SFT throughput numbers (§2), for Dolmino / Longmino as separately-tracked lineages (§3). This excerpt pulls those specifically and expands the operational architecture they imply.

---

## The data curriculum — five lineages, not one corpus

From the source (lines 44-49):

> ### Data curriculum
> - **Dolma 3:** about **9.3T** source tokens spanning web pages, science PDFs processed with `olmOCR`, code, math problems/solutions, and encyclopedic text.
> - **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens with stronger math/code emphasis and stronger decontamination.
> - **Dolma 3 Dolmino:** **100B** mid-training tokens sampled from a ~2.2T high-quality pool for math, science, code, instruction following, and reading comprehension.
> - **Dolma 3 Longmino:** about **50B** long-context tokens from a **639B**-token pool of long documents plus mid-training data.
> - **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

Five *separate* named lineages, each serving a different stage of training. The operational consequence ch-11 §3 builds on: **you cannot treat the corpus as one artifact**. Each named mix has:

- Its own filter pass (stronger decontamination for Dolma 3 Mix; long-doc filter for Longmino; rubric filter for Dolci-SFT).
- Its own pool (9.3T source → 5.9T pretraining; 2.2T high-quality → 100B Dolmino; 639B long-doc → 50B Longmino).
- Its own consumer (Dolma 3 Mix → base pretrain; Dolmino → mid-training; Longmino → long-context stage; Dolci → post-training).

This is the model-flow worldview ch-11 §3 mirrors. The operational demand: an attribute-file per document must carry `mix_membership: [dolma3, dolma3_mix, dolmino, longmino, dolci_sft, dolci_dpo, dolci_rlvr]` — five-to-seven boolean flags indicating which mixes the document is part of. Alternatively and equivalently, each mix is a *query* over the shared raw pool, and the mix identity is the hash of the query.

The second formulation is what OLMo 3's tooling (Olmo-core, Open Instruct, OLMES) actually implements. From the source (lines 56-60):

> - Pretraining used up to **1,024 H100 GPUs**.
> - Mid-training used **128 H100 GPUs**.
> - Post-training used **256 H100 GPUs**.
> - Moving SFT from **Open Instruct** to **Olmo Core** reportedly improved throughput by **8x**.
> - In-flight weight updates, continuous batching, and threading work made RL training about **4x** more efficient.

The 8× SFT throughput improvement is the operational payoff of the model-flow architecture: because Dolci-SFT is a separate mix with its own sharded layout and its own dataloader state, moving it to Olmo Core's streaming implementation is a one-mix migration, not a full-pipeline rewrite. Ch-11 §2's "MDS shards + deterministic resumable streaming" discussion points directly to this capability.

---

## The four-branch post-training tree

From the source (lines 19-23):

> - Treats the **entire model flow** as the public artifact, not just final checkpoints.
> - Releases multiple branches from the same base: **Base**, **Think**, **Instruct**, and **RL Zero**.
> - Uses a clear **three-stage base training recipe**: pretraining, mid-training on harder distributions, and long-context extension.
> - Uses a clear **three-stage post-training recipe** inherited from Tulu 3: **SFT -> DPO -> RLVR**.

Four branches × three post-training stages = **twelve distinct checkpoints**, each consuming a specific slice of the Dolci mix. Base-SFT uses general Dolci-SFT; Think-SFT uses the reasoning-heavy subset; RL Zero starts from the base model (no SFT) to isolate the RL signal. Each branch has its own mix-query over Dolci.

For ch-11 §3's reproducibility argument this is the decisive case. A lab consuming OLMo 3 that wants to reproduce "OLMo 3-Think 7B" needs to know *which specific Dolci queries* were used. The paper ships these queries as part of the public artifact. A lab that only shipped final weights cannot be reproduced this way — the queries are the missing load-bearing information.

Ch-11 §3's "mix version = hash(query_source + attribute_file_hashes)" formulation is directly realized in OLMo 3's release.

---

## Tooling as a first-class artifact

From the source (line 23):

> - Couples model release with tooling: **Olmo-core**, **Open Instruct**, **OLMES**, **OlmoTrace**, decontamination, and dedup utilities.

Five tools, each solving one part of the operations stack:

| Tool | Responsibility | Ch-11 section |
|---|---|---|
| Olmo-core | Trainer + dataloader (MDS-style streaming, resumable state) | §2 (shard layout) |
| Open Instruct | Post-training harness for SFT/DPO/RLVR | §2 (consumer-side mix routing) |
| OLMES | Eval harness with contamination gating | §3 (decontamination as lineage edge) |
| OlmoTrace | Provenance / lineage tracking | §3 (cross-stage doc-id tracking) |
| dedup + decontam utilities | MinHash + n-gram contamination checks | §4 (dedup, PII-cascade analog) |

The implicit claim: **every modern open pretraining release needs this tool set** (or a proprietary equivalent). Ch-11's chapter scope tracks approximately this set, minus the training-side tools (which belong in ch-05/ch-06). OlmoTrace specifically is the clearest public implementation of the attribute-file lineage graph that ch-11 §3 describes abstractly.

---

## OCR as a pipeline stage — olmOCR

From the source (line 45):

> - **Dolma 3:** about **9.3T** source tokens spanning web pages, science PDFs processed with **`olmOCR`**, code, math problems/solutions, and encyclopedic text.

This is a ch-11-§2 operational detail that only OLMo 3 has surfaced publicly. Science PDFs are a non-trivial text source (arXiv, biorxiv, MSc-thesis mirrors, journal archives) but they require OCR or structured PDF parsing to extract usable training data. `olmOCR` is Allen AI's LLM-based PDF OCR tool.

For the pipeline this adds a stage:

```
raw PDF → olmOCR → extracted text (with its own doc_id = BLAKE3(extracted text))
```

The extracted text becomes the CCNet-style primary key. The original PDF can be discarded from the training pipeline (though kept in archives). The olmOCR stage is expensive (LLM inference on every PDF page, ~1s/page on a dedicated GPU pool) but one-time: the extracted text is cached and reused across mix queries.

Ch-11 §2 doesn't pull out olmOCR explicitly but the "raw shards" stage in the pipeline figure implicitly covers "whatever stage produces the extracted text." OLMo 3's disclosure confirms that extraction is non-trivial for non-web sources and merits its own operational treatment.

---

## Decontamination as a lineage edge

From the source's tooling list:

> - dedup + decontam utilities

Decontamination — removing training documents that match eval benchmarks (MMLU, ARC, HumanEval test sets, etc.) — is a ch-12 + ch-14 topic but has a ch-11 operational implication: it's a lineage edge.

The OLMo 3 / OLMES pattern: every benchmark eval ships with a known n-gram list; the training pipeline runs n-gram overlap against this list at ingest and emits `decontam.overlapped: list[benchmark_id]` as an attribute. A document with non-empty `decontam.overlapped` is dropped from the pretraining mix but kept in the raw pool (so it can still be used for, e.g., the benchmark's own training split). The decision is a consumer-side query.

Ch-11 §3's "attribute-file as the lineage substrate" pattern absorbs this directly: decontamination is another attribute, another filter stage. OLMo 3's tooling ships this as a first-class utility rather than a one-off script, which is the operational maturity step that distinguishes a "corpus release" from a "model flow release."

---

## What the model-flow worldview costs

From the source (line 57-58):

> - Pretraining used up to **1,024 H100 GPUs**.
> - Mid-training used **128 H100 GPUs**.

One operational inference: OLMo 3's mid-training (Dolmino, 100B tokens, 128 H100s) is roughly 1.25% of the pretraining compute (Dolma 3 Mix, 6T tokens, 1024 H100s). Five-stage mixes let you allocate compute differentially — each stage gets the compute its data volume demands. A monolithic "train on the whole thing" approach loses this flexibility.

But there's an operational tax: **five mixes require five shard-production pipelines, five consumer configs, five resume-state handlers**. Ch-11 §2's mix-pointer discussion (save the current mix's identity in the dataloader state; alarm on resume if the pointer is missing) is precisely the discipline required to make multi-mix training safe.

OLMo 3's 8× SFT throughput gain from Olmo Core shows the payoff: once the operational infrastructure is right, adding a new mix is cheap, and each mix can be optimized independently. The pattern generalizes: **frontier labs are moving from "the dataset" to "the mix portfolio"**, and ch-11's operational emphasis is what lets that portfolio exist.

---

## What OLMo 3 does not solve — the tokenizer

OLMo 3 uses the same byte-level BPE as OLMo 2 (likely 100K-ish, not explicitly quoted in the excerpt). The report does not heavily emphasize tokenizer extension because the reserved-slot pattern is inherited from OLMo 2 and Tulu 3. For ch-11 §5's discussion of post-pretrain tokenizer extension, OLMo 3 is implicit evidence that the reserved-slot approach works (the Think / Instruct / RL Zero branches all reuse the same tokenizer without drift), but the report doesn't explicitly document the initialization recipe.

The gap is filled by [[excerpts/phi-4]] which has a more explicit tokenizer-extension story for reasoning tokens, and by Llama 3's reserved-slot convention which OLMo 3 implicitly follows.

---

## What to take from OLMo 3 for ch-11

1. **The corpus is a portfolio of mixes**, not one artifact. Dolma 3 / Dolma 3 Mix / Dolmino / Longmino / Dolci are five separately-tracked lineages.
2. **Each mix is a query** over the shared raw pool; the mix identity is the query hash.
3. **Tooling is load-bearing**: Olmo-core, Open Instruct, OLMES, OlmoTrace, dedup + decontam — every open release at this scale needs the equivalent.
4. **OCR (olmOCR) is a pipeline stage** for non-web sources; its output enters the standard doc_id → attribute graph.
5. **Decontamination is an attribute edge**, not a global filter; consumers query against it.
6. **Multi-stage training costs flexibility** but pays back 8× in throughput once the operational infrastructure lets each stage be optimized independently.

---

## Connections

- [[excerpts/dolma]] — Dolma 3 is the direct successor; the attribute-file architecture scales up here.
- [[excerpts/fineweb]] — Dolma 3 Mix uses similar classifier-driven filtering principles.
- [[excerpts/llama-3]] — the six-round post-training flywheel has a similar multi-stage lineage structure but is less explicitly documented.
- [[excerpts/olmo-2]] — the direct predecessor; model-flow generalizes OLMo 2's two-stage curriculum.
- [[excerpts/qwen-3]] — three-stage pretraining (general → reasoning → long-context) is a lighter-weight analog.
- [[excerpts/deepseek-v3]] — 14.8T pretrain with two-stage context extension; single-mix approach for comparison.
- [[ch-11]] — §2 (mix-pointer in dataloader state; 8× throughput), §3 (model-flow as lineage requirement; tooling as first-class artifact).
