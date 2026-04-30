<!-- scope: reasoning-trace synthesis — 10M web-extracted + refined math QA pairs
     deps: [[mammoth]]
     see-also: [[openmathinstruct-2]], [[mathscale]]
-->

# MAmmoTH2: Scaling Instructions from the Web
- **Core Insight:** The open web (Common Crawl) already contains billions of naturally-occurring instruction-response pairs inside Q&A pages, exam preparation sites, and forum threads; a three-stage pipeline (recall → extract → refine) can mine 10M high-quality reasoning instruction pairs without any seed questions.
- **Guideline:** To scale instruction SFT past human-crafted seeds, train a classifier to find Q&A-rich web pages, extract raw pairs, and have an LLM refine them into clean instruction-response format; this taps a data source orders of magnitude larger than any hand-authored seed.
- **Authors:** Xiang Yue, Tuney Zheng, Ge Zhang, Wenhu Chen
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.03548
- **Relevant topics:** web-scale instruction mining, reasoning synthesis, retrieval, MAmmoTH follow-up

## Abstract
MAmmoTH2 constructs WEBINSTRUCT, a 10M-pair instruction dataset mined from Common Crawl, by (1) training an fastText + BERT-based classifier to identify Q&A-rich web pages, (2) extracting Q&A pairs via rule+LLM hybrid, (3) refining them into clean instruction-response format with Mixtral-8x7B. Fine-tuning Mistral-7B on WEBINSTRUCT produces MAmmoTH2-7B, improving MATH by 22 points and ARC-C by 9 points over baseline — despite using no hand-authored seed questions.

## Key Contributions
- **WEBINSTRUCT 10M** — web-mined instruction dataset, released.
- Three-stage mining pipeline: **Recall → Extract → Refine**.
- MAmmoTH2-Plus-8x7B (Mixtral base + continued SFT).
- Demonstrated that web mining is complementary to synthesis-from-seeds: mixing WEBINSTRUCT with MathInstruct gives additional lift.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Stage 1 — Recall
- **Seed:** ~100K known Q&A pages (StackExchange, Socratic, exam prep sites) as positives; random web pages as negatives.
- **Classifier:** fastText + DeBERTa-v3 trained on positives/negatives; applied to ~18B Common Crawl web documents.
- **Output:** ~18M candidate instruction-rich web pages.

### Stage 2 — Extract
- **Rule-based extraction:** regex + DOM-structure patterns to pull question-blocks (e.g., `<div class="question">`) and answer-blocks from ~3M pages that have clean markup.
- **LLM extraction:** for the remaining pages without obvious markup, prompt Mixtral-8x7B with the page text to extract (question, answer) pairs.
- **Output:** ~11M raw Q&A pairs.

### Stage 3 — Refine
- **LLM refinement:** Mixtral-8x7B is prompted to rewrite each pair into clean instruction-response format:
  - Fix grammar, remove HTML artifacts.
  - Normalize math formatting (LaTeX).
  - Where answers lack reasoning, expand into CoT.
- **Quality filter:** Mixtral-based LLM-judge scores each refined pair 1–5 on (clarity, answer correctness, instructional value); keep scores ≥ 4.
- **Output shape:** 10M final instruction-response pairs covering math, science, code, general reasoning.
- **Teacher model:** Mixtral-8x7B-Instruct (Apache-2.0).
- **Cost / compute:** full pipeline ~200K GPU-hours (Common Crawl classification dominates).

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** highly variable — median ~300 tokens, tail to 3K. Mostly short CoT.
- **Trace style:** naturalistic — reflects web-authored Q&A style (teacher-explaining, stackexchange top-answer).
- **Correctness verifier:** LLM-judge only — no gold-answer execution. Authors acknowledge this gives some noise but argue scale compensates.
- **Domain mix:** ~40% math/physics, ~20% CS/code, ~20% general STEM, ~20% humanities/social.
- **Deduplication:** MinHash on question text to remove near-duplicates across sites.

## Quality / diversity evaluation
- MAmmoTH2-7B (Mistral base): MATH 11 → 32.6, GSM8K 51.4 → 67.4, ARC-C 80.5 → 84.7.
- MAmmoTH2-Plus-8x7B: MATH 48.4, TheoremQA 34.1 — competitive with GPT-3.5-Turbo.
- Mixing WEBINSTRUCT + MathInstruct gives further +3 MATH over either alone.
- Web-mined data generalizes better to out-of-domain benchmarks than synthesized-from-seeds data.

## Risks + gotchas
- **Correctness noise:** no gold-answer filtering means some Q&A pairs have wrong answers; authors estimate ~8% error rate.
- **License uncertainty:** web-mined content has mixed licenses; release under research-only terms.
- **Distribution mirrors the open web:** inherits its biases (overrepresentation of programming/STEM, underrepresentation of non-English).
- **Cannot be regenerated without CC access** — reproducibility bottlenecked on Common Crawl crawls.

## Connections
- Direct predecessor: [[mammoth]] (seed-based hybrid synthesis).
- Contrasts seed-based: [[self-instruct]], [[metamath]], [[mathscale]], [[openmathinstruct-2]].
- Mining-side cousin: [[rephrasing-the-web]] (rephrase web into synthetic pretrain).
- Classifier-based filtering lineage: [[fineweb]] (FineWeb-Edu uses similar quality classifier).
