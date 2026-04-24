---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/openassistant.md
source_url: https://arxiv.org/abs/2304.07327
created_at: "2026-04-23"
---

# Excerpt: OpenAssistant — crowdsourced annotation and the quality-control tax

**Source library:** `wiki/raw-data/llm-training/papers/openassistant.md`
**Year / authors:** 2023 / Köpf, Kilcher, Anagnostidis, Mattick et al. (LAION + community).

---

## Why this source anchors ch-15

OASST is the open-source counter-experiment to the vendor-annotated world of InstructGPT and HH-RLHF. ~13,500 volunteers, no formal onboarding, no calibration sessions, no paid tier structure. 161K messages in OASST1 (10K+ conversation trees), ~250K in OASST2. Multi-lingual (35 languages), multi-turn, tree-structured. It is the real data point for **what human annotation looks like when the operational layer of ch-15 §6 is *absent***: what you get, what fails, what you have to clean up.

---

## The four roles — a structural rubric alternative

```
# openassistant.md, §3 (reconstructed from the contribution UI)
Four contribution types on open-assistant.io:
  Prompter: write a new user turn at a conversation-tree node.
  Assistant: write a candidate reply.
  Labeler: rate a reply on quality / spam / creativity / helpfulness / harm.
  Ranker: order multiple candidate replies at one node by preference.

Tree format:
  Each node: parent_id, children, author_id, language, quality_scores,
             timestamps. Conversations = root-to-leaf paths.
  Avg path length: 4–6 turns; some to 20+.
```

Read the four roles as a *structural* rubric rather than a *written* rubric. The InstructGPT rubric is a document; the OASST rubric is the UI — the set of actions a contributor can take. The quality of the dataset is determined by how the UI decomposes the labeling task, not by how clearly any paragraph is written.

This is why OASST is a useful reference for the ch-15 §3 adjudication tiering. Each message gets:
- Multiple *labelers* rating quality, spam, helpfulness, harm (→ approximate double/triple annotation by construction).
- Multiple *rankers* ordering candidate replies at each node (→ preference data by construction, no separate ranking pass).

The cost is entirely on the structural layer: the web UI, the database, the moderation pipeline. The text rubric can be short because the UI enforces the schema.

---

## The quality variance — what no-onboarding produces

> Crowdsourced quality is uneven — some replies are empirically worse than ChatGPT's.
>
> Demographic skew: contributors skew Western/English/technical; some languages have very sparse coverage.

The cleaning task post-hoc is enormous. Post-release audits report ~30–50% of OASST messages are below the quality bar for production SFT. The defense: multiple labelers per message, spam/toxicity filter (human + automated), quality-threshold flagging.

For ch-15 §6 (operational reality), the OASST data point is stark: **onboarding and calibration are not optional overhead; their absence is paid later as a 30–50% rejection rate post-collection**. Every project that chose the vendor path (InstructGPT's vetted labelers, [[tulu-3-sft-mix]]'s Surge/Scale contracts) pays the cost upfront and gets a cleaner base dataset; OASST externalized the cost to the downstream consumer who has to filter.

---

## The tree-as-preference-structure — a quiet innovation

OASST's conversation tree is not just "a dialogue with branches." It is a *natural* way to generate preference pairs: when two candidate replies sit at the same parent node with different quality rankings, you have a preference pair *with shared context*. This is the 2023 ancestor of the [[ultrafeedback-construction]] 17-model-pool protocol, except the "fleet" is human volunteers rather than model generations.

The quality heterogeneity is actually useful for preference data: you want pairs where the quality gap is nontrivial, and a crowdsourced fleet of varied contributors produces exactly that. Ch-15 §4's close-pair mining observation — that the most informative pairs have *moderate* quality gaps — maps cleanly: OASST's labeler rankings give per-node quality scores, and a downstream preference extractor can filter for the band where the gap is neither trivial (both bad, both good) nor obvious (one from a top contributor, one from spam).

---

## Multi-lingual annotation — the 35-language case study

OASST is the strongest 2023 data point on multi-lingual annotation economics. Languages vary 100× in contributor count (English dominant ~45%, then Spanish, Chinese, German, Russian, with a long tail at 1% each). For ch-15 §6's onboarding discussion, the multi-lingual case is instructive: **rubric translation is a second rubric-design problem**. A "helpful" criterion in English may not map cleanly to politeness conventions in Japanese. OASST's response is a uniform schema (quality / helpfulness / harm) with per-language interpretation left to contributors — which works for a crowdsourced public-good project, but fails for a frontier lab that needs comparable signal across languages.

The 2024 successor, [[tulu-3-sft-mix]]'s multilingual component (7% Aya + Tulu-3 Persona-Multiling), delegates the multi-lingual rubric problem to synthetic generation: prompts translated/generated, responses generated by a single strong multilingual model (GPT-4o, Claude 3.5 Sonnet). That trades OASST's human authenticity for rubric uniformity — a tradeoff that the ch-15 §5 "when human overrides judge" rubric must adjudicate per language.

---

## The licensing footnote — why the OASST data point matters

> License: CC-BY 4.0.

Every preference-data paper in 2024–2025 benchmarks against OASST (alongside HH-RLHF) because both are redistributable. [[tulu-3-sft-mix]] uses 7,132 OpenAssistant Guanaco prompts as part of its 939K mix. The CC-BY attribution requirement means downstream models inherit a citation chain, not an encumbered license. This is the reason OASST, despite its quality variance, remains a load-bearing ingredient in open alignment stacks — its legal structure is clean, and the quality variance is addressable with a downstream filter.

---

## Connections

- [[excerpts/rlhf-instructgpt]] — the vendor-annotated counterpart; the comparison point for OASST's no-onboarding cost structure.
- [[excerpts/hh-rlhf]] — the other public preference dataset; HH is MIT + paid workers, OASST is CC-BY + volunteers.
- [[excerpts/tulu-3-sft-mix]] — consumer of OASST data (7K prompts) within a broader skill-targeted mix.
- [[excerpts/ultrafeedback-construction]] — synthetic-fleet counterpart to OASST's human fleet.
- [[ch-15]] — this excerpt supports §3 (multi-labeler structure), §4 (tree-to-preference-pair conversion), §6 (the cost of *not* doing formal onboarding).
