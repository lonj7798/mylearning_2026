<!-- scope: Nous Capybara / LDJ — multi-turn synthetic dialogue via Amplify-Instruct
     deps: [[self-instruct]]
     see-also: [[openhermes]], [[airoboros]], [[baize]]
-->

# Capybara (LDJ / Nous Research): Amplify-Instruct Multi-Turn Dataset
- **Core Insight:** Multi-turn quality beats raw scale — ~20K carefully curated conversations (60%+ multi-turn) generated via the "Amplify-Instruct" pipeline that composes seeds from multiple existing synthetic methods (Airoboros + Evol-Instruct + Orca + Vicuna + CamelAI + LessWrong posts) can train competitive chat models.
- **Guideline:** For small, high-quality multi-turn SFT corpora, compose seeds from heterogeneous sources (prior synthetic datasets + curated web posts like LessWrong), use a synthesis pipeline that amplifies seed instructions into dialogues, and emphasize multi-turn coverage over scale.
- **Author(s):** LDJnr (Daniele), J-Supha, Luigi, Suphavadeeprasit (Nous Research)
- **Year:** 2023–2024
- **URL:** https://huggingface.co/datasets/LDJnr/Capybara ; https://huggingface.co/NousResearch/Nous-Capybara-7B-V1.9
- **Relevant topics:** multi-turn SFT, synthetic dialogue, Amplify-Instruct, Nous Research

## Overview
Capybara is a compact, curated multi-turn SFT dataset (~20K conversations) built with the **Amplify-Instruct** pipeline. Training Nous-Capybara-7B-V1.9 (Mistral-7B base) on it yields competitive chat behavior with far smaller data than Evol-Instruct / ShareGPT. The method's signature move is seed composition — drawing initial instructions from a heterogeneous set of prior synthetic sources and human-curated posts, then running a custom amplification to extend into multi-turn dialogue.

## Dataset stats
- **Size:** ~20K conversations.
- **Format:** multi-turn (60%+) + single-turn remainder.
- **Tokens:** dataset-card disclosed as small enough for LoRA on a single consumer GPU.
- **License:** Apache-2.0 for the dataset (model: Mistral base license).

## Seed composition
- **Synthetic sources** (seed instructions drawn from):
  - Airoboros (Jon Durbin)
  - Evol-Instruct / WizardLM
  - Orca
  - Vicuna
  - Know_Logic
  - Lamini
  - FLASK
- **Curated human posts:**
  - LessWrong rationality-community posts (seed for analytic / reasoning conversations).
  - EverythingLM seeds.
  - GPTeacher seeds.
- **Domain experts:** volunteer-curated physics / math / biology / chemistry prompts via CamelAI partnership.
- **In-house multi-turn:** Dove dataset extensions.

## Synthesis method — Amplify-Instruct (schematic)
1. Sample a seed instruction from the composed pool.
2. Generate initial assistant response.
3. **Amplify:** synthesize a plausible follow-up user turn (varied via distributional samplers to promote diversity).
4. Generate assistant response to follow-up.
5. Iterate for N turns (typically 2–6).
6. Filter by length, dedup, and quality (manual + LLM-check).
- Exact prompt templates documented on the model card; paper was "pending publication" at time of release.

## Training / eval
- Nous-Capybara-7B-V1.9 (Mistral-7B) achieves strong MT-Bench / AGIEval numbers at release given its small SFT set.
- Nous-Capybara-34B (Yi-34B base) pushed it further.
- Evaluation emphasizes multi-turn coherence and reasoning probes over raw instruction follow.

## Practitioner takeaways
- **Seed diversity > raw scale** — Capybara's compact size is deliberate.
- LessWrong-post seeding is a notable trick for injecting rationality / probabilistic-reasoning style.
- Multi-turn amplification is under-explored in other synthetic pipelines; consider adding it.

## Risks + gotchas
- **Small scale** limits capability breadth; not a drop-in replacement for large SFT mixes.
- **Volunteer-labeled** subsets have uneven quality control.
- **License mix** — dataset is Apache-2.0 but base-model outputs from GPT-4 era can carry API terms.

## Connections
- Sits with [[openhermes]] in the Nous / community-curated open catalog.
- Contrasts with [[magpie]] (large scale, zero seeds) on the small-scale, heavy-seed axis.
- Seed composition idea influenced later community mixes (OpenHermes 2.5, Dolphin).
