---
chapter: ch-52
course: llm-training
phase: read
excerpt_of: Li et al. 2024 — "SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models"
source_url: https://arxiv.org/abs/2402.05044
created_at: "2026-04-23"
---

# Excerpt: Salad-Bench — Hierarchical Diagnosis and MD-Judge

**Source:** `wiki/raw-data/llm-training/papers/salad-bench.md`
**Primary paper:** Lijun Li et al., Shanghai AI Lab + HIT, 2024
**arXiv:** https://arxiv.org/abs/2402.05044

---

## The core contribution — 3-tier hierarchical taxonomy

Salad-Bench organizes harms into **6 domains → 16 tasks → 66 categories**. The domains:

1. `Representation & Toxicity`
2. `Misinformation Harms`
3. `Socioeconomic Harms`
4. `Information & Safety Harms`
5. `Malicious Use`
6. `Human-Chatbot Interaction Harms`

Three tiers matter because a scalar "harmfulness rate" hides which leaves the model fails on. A model may score 94% aggregate safe while collapsing to 40% on `Representation & Toxicity / toxic language / racial-stereotype` and 99% on `Malicious Use`. The hierarchical structure lets you report per-domain, per-task, or per-category drill-down.

From the raw-data notes:

> *"Safety evaluation requires a **hierarchical taxonomy** (6 domains → 16 tasks → 66 categories) so researchers can diagnose which specific harm types a model struggles with."*

---

## Dataset size and attack enhancements

- ~30K base questions curated from AdvBench, BeaverTails, DoNotAnswer, ToxicChat, plus LLM-synthesized augmentation per leaf, plus manual review.
- ~10K attack-enhanced variants via 6 attack methods applied to base questions:
  - `GCG` adversarial suffix
  - Word-level perturbation
  - Human-written jailbreaks (DAN-style)
  - Multilingual translation attacks
  - Persona injection
  - Crescendo / escalation attacks

Note the contrast with HarmBench's ~18 attack families. Salad-Bench is broader in harm taxonomy and narrower in attack coverage; HarmBench is the opposite.

---

## MD-Judge — the fine-tuned safety judge

- Base: Llama-2-7B.
- Training data: ~3K human-labeled `(query, response, safety-label)` triples.
- Reported human agreement: ~89%.

MD-Judge is the lineage-predecessor to WildGuard's 7B guard model. The two are near-siblings: small fine-tuned Llama-2 classifiers trained on human-labeled safety triples. For ch-52, the useful framing is that every modern safety benchmark ships its own judge, and the judges do not agree with each other — running the same completions through MD-Judge and WildGuard-7B gives different numbers, and neither is ground truth.

---

## Headline results as reference anchors

- Llama-2-Chat-70B: **95%+ safe on base questions; ~75% under attack.**
- GPT-4: **97%+ safe on base; ~85% under attack.**
- Mistral-Instruct-v0.2 (best 2024 open): **mid-70s under attack.**

These numbers are the ch-52 §2 reference band. A ~20 pp drop under attack is the rough 2024-era floor; a <10 pp drop means the attack suite is too weak.

---

## Risks and gotchas the raw-data source calls out

- **Per-category noise.** Many of the 66 leaves have <100 samples; per-category numbers are noisy. Treat Salad-Bench as a **diagnosis tool** ("which categories are weak?") not as a leaderboard ("what is my score?").
- **Contamination risk.** Public release means the prompts may appear in future training data; v2 and rotation are expected.
- **English-only.** Multilingual coverage is only via translation-attack variants.
- **Attack catalog frozen.** New jailbreak methods post-2024 (LLM-assisted prompt evolution, multi-turn context-poisoning attacks) are not in v1.

---

## Where Salad-Bench fits in ch-52

Salad-Bench is the taxonomy-diagnosis benchmark. Use it when the property you care about is **which harm category is weakest**. Pair with HarmBench (attack-family drill-down) and WildGuard (refusal vs response-harm separation) to cover the full safety-evaluation surface.

---

## Connections

- [[harmbench-data]] — 400 behaviors × 18 attacks; attack-drill-down sibling.
- [[wildguard-data]] — moderation + refusal synthesis; judge-model sibling (MD-Judge → WildGuard lineage).
- Chapter synthesis: [[ch-52]] §1, §2.
