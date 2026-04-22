<!-- scope: Jon Durbin's airoboros — role-specific instructor pipeline for synthetic SFT
     deps: [[self-instruct]]
     see-also: [[openhermes]], [[capybara]]
-->

# Airoboros: Customizable Self-Instruct Pipeline with Role-Specific Instructors (Jon Durbin)
- **Core Insight:** Split Self-Instruct into **specialist "instructors"** — separate generation pipelines per task type (Orca-reasoning, roleplay, math, coding, summarization, character-chat) — so each sub-pipeline can optimize its own prompt templates and topic injections; the union yields more balanced coverage than a single monolithic Self-Instruct run.
- **Guideline:** When you need a broad synthetic SFT corpus, write one "instructor" per capability with its own prompt template; randomize topic injection per instructor to maximize diversity; combine outputs into a single SFT mix.
- **Author:** Jon Durbin (independent researcher)
- **Year:** 2023–2024 (multiple versions 1.x through 3.x)
- **URL:** https://github.com/jondurbin/airoboros ; https://huggingface.co/datasets/jondurbin/airoboros-2.2
- **Relevant topics:** self-instruct variants, instructor-based synthesis, role-specific pipelines

## Overview
Airoboros is a community-maintained self-instruct implementation that became the de-facto template for "build your own synthetic SFT dataset" in 2023. Its distinguishing feature is the **instructor system**: each capability (math, code, roleplay, orca-style reasoning, character-chat, summarization, coding-challenge, etc.) has its own instructor module with bespoke prompt templates. Datasets released in versions (airoboros-1, 2, 2.2, 3, 3.1, …) progressively expanded instructor coverage.

## Key Contributions
- Proved that **one individual researcher** can produce a 70B-competitive instruction-tuning dataset with GPT-4 synthesis — airoboros-70B was among the first such examples.
- Open-sourced the full pipeline code; community forks exist.
- Introduced concrete instructor types later adopted by broader projects:
  - **Orca-style reasoning** instructor (step-by-step answers).
  - **Roleplay / character** instructor.
  - **Math** instructor with MathJSON outputs (v3.1).
  - **Information-extraction** instructor.
  - **Anonymization** (remove names/IPs/dates).
  - **Multi-step instructions with acknowledgements** (chain of micro-tasks).

## Synthesis pipeline (concrete, version-varying)
- **Seed input:** a small instructor-specific seed pool + a topic pool (random topics injected to diversify).
- **Topic injection:** random topics (history, science, pop culture, …) appended to the instruction-generation prompt to diversify.
- **Instructor-specific prompts:** each instructor has a templated "generate a task of type X on topic Y" prompt.
- **Teacher:** GPT-4 (for airoboros-2.x onward; GPT-3.5 in v1).
- **Deduplication:** ROUGE-L + embedding near-duplicate drop.
- **Output shape:** mixed instruction-response pairs covering many capabilities; sizes range from tens of thousands to hundreds of thousands per version.

## Practitioner takeaways
- **Specialization > monolith** — per-instructor prompt engineering catches capability gaps that a single prompt template misses.
- **Random topic injection** is a cheap diversity lever; competes with [[persona-hub]] at much smaller scale.
- **Iterative versioning** matters — each Airoboros release fixed prior artifacts (refusal bias, format issues).
- Historically critical as the 2023-era "proof that a single person can build competitive synthetic SFT."

## Risks + gotchas
- **GPT-4 license concerns** — outputs carry OpenAI TOS; redistribution nuanced.
- **No central paper** — documentation lives in READMEs and model cards; methodology can drift across versions.
- **Deduplication non-strict** — some versions have noticeable near-duplicates.
- **Instructor coverage uneven** — roleplay over-represented relative to code/math in older versions.

## Connections
- Conceptually aligned with [[self-instruct]] but decomposed by capability.
- Used as seed corpus in [[capybara]] and [[dolphin]] mixes.
- Philosophy later institutionalized at scale in [[nemotron-4-synthetic]] (category-seeded prompt synthesis) and [[glan]] (taxonomy-driven).
- The "specialist-per-capability" pattern is the operational ancestor of per-capability pipelines in [[llama-3-synthetic-pipeline]].
