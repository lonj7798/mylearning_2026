---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/tulu-3-sft-mix.md
source_url: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
created_at: "2026-04-23"
---

# Excerpt: Tülu 3 SFT mixture — the full row-by-row disclosure

**Source library:** `wiki/raw-data/llm-training/papers/tulu-3-sft-mix.md` (dataset card + Ai2 blog)
**Artifact:** 939,344-prompt SFT mixture, 18 components, 57/43 public/synthetic

---

## Why this source anchors ch-33

The whole point of the Tülu 3 case study is the *row-by-row disclosure*. Before Tülu 3, "open post-training mixture" meant "a repo that releases the SFT jsonl but not how it was mixed"; after Tülu 3, it means "every count is public, every decontam rule is public, every skill share is public". This excerpt pins the exact counts §1.1 of the chapter quotes so the chapter's table is auditable against the source.

---

## The attested component counts — verbatim from the dataset card

From the source (lines 22–38):

**Public / imported sources (569,205 total — 57% of mix):**
- CoCoNot — **10,983** prompts
- FLAN v2 via `ai2-adapt-dev/flan_v2_converted` — **89,982**
- No Robots — **9,500**
- OpenAssistant Guanaco — **7,132**
- Aya — **100,000**
- WildChat GPT-4 — **100,000**
- TableGPT — **5,000**
- SciRIFF — **10,000**
- NuminaMath-TIR — **64,312**
- WildGuardMix — **50,000**
- Evol CodeAlpaca — **107,276** (derivative license — GPT-generated code from an Alpaca seed, not "pure public")

**In-house synthetic (370,139 total — 43%):**
- Tülu 3 Persona MATH — **149,960**
- Tülu 3 Persona GSM — **49,980**
- Tülu 3 Persona Python — **34,999**
- Tülu 3 Persona Algebra — **20,000**
- Tülu 3 Persona IF — **29,980**
- Tülu 3 WildJailbreak — **50,000**
- Tülu 3 Hardcoded — **240**

Totals: 569,205 + 370,139 = **939,344** prompts across **18 components**.

---

## What ch-33 keeps from this source

- The per-component count table in §1.1 is a direct transcription of this list. The two totals above (public vs synthetic) are derived from the dataset card's 57/43 split claim and the reported 939,344 total — Ai2 does not publish a row-level binary label per row, and ch-33's table inherits that caveat by labelling `Evol CodeAlpaca` as *public (derivative)* rather than pure public.
- The *skill-mixture-first* construction procedure (§1.1 of the chapter) is attested here: Ai2 builds skill-specific submixes first, keeps the ones that move the hardest skills in isolated ablations, then combines and decontaminates. The final mix is *not* a single generation; it is a merger of many per-skill mixes.

---

## Why this matters for the reader

If you replicate Tülu 3 without reproducing this exact mixture, your results cannot be attributed to the recipe — they are a different mixture. Conversely, if you audit a new open-post-training release, the check that distinguishes "full disclosure" from "weights only" is whether a table like this one exists. The community standard for "fully open SFT" is this table's existence.

---

## Connections

- **ch-33 §1.1** — the chapter's per-source table.
- **ch-34** — OLMo 2/3 reuse this mixture pattern; Phi 3/4 explicitly *do not*, and the contrast is instructive.
- **[[tulu-3]]** — the full technical report that names the counts.
- **[[allenai-tulu-sft-recipe]]** — the blog that aggregates them into the 7-bucket skill view.
- **[[persona-hub]]** — the upstream persona-factory pattern the 7 "Persona ..." entries instantiate.
