<!-- scope: multi-turn conversation synthesis — crowdsourced conversation tree as real-human baseline
     deps: [[baize-construction]]
     see-also: [[ultrachat-pipeline]], [[hh-rlhf]], [[wildchat]]
-->

# OpenAssistant Conversations: Democratizing Large Language Model Alignment
- **Core Insight:** A global crowdsourcing effort with a tree-structured conversation format (branching replies, multiple candidates per turn, quality ranking) can produce a genuinely-human dataset covering 35 languages; this is the reference "real human dialogue" baseline against which all synthetic conversation datasets (Baize, UltraChat, SODA) are measured.
- **Guideline:** When evaluating whether a synthetic conversation corpus captures real human behavior, benchmark against OASST; gaps in user-turn diversity, politeness variance, and follow-up distribution are the strongest signals of synthetic-data narrowness.
- **Authors:** Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, Alexander Mattick, et al. (LAION + community)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2304.07327
- **Relevant topics:** crowdsourced dialog, human-annotated, tree-structured, OpenAssistant, LAION

## Abstract
OpenAssistant Conversations (OASST1, OASST2) is a crowdsourced multi-lingual, multi-turn conversation dataset built by ~13,500 volunteers between Dec 2022 and Mar 2023 (OASST1) and continued through 2024 (OASST2). The data follows a **conversation-tree format**: each user turn may have multiple assistant-candidate replies, each assistant turn may have multiple user follow-ups; every reply is rated on quality/helpfulness/harmlessness. OASST1 contains 161K messages forming 10K+ fully labeled conversation trees in 35 languages.

## Key Contributions
- **Tree-structured crowdsourced dialog format** — branching replies + quality votes.
- **35 languages covered** — real multilingual dialog, not translations.
- **161K OASST1 + further OASST2 messages** publicly released under CC-BY 4.0.
- Used as the real-human anchor for comparing synthetic dialog corpora.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Crowdsourced workflow
- **Platform:** open-assistant.io — browser-based contribution interface.
- **Contribution types:**
  - **Prompter:** write a new user turn at a conversation-tree node.
  - **Assistant:** write a candidate reply.
  - **Labeler:** rate replies on quality, spam, creativity, helpfulness, harm.
  - **Ranker:** order candidate replies by preference.
- **Tree structure:**
  - Each node has a parent and multiple children.
  - Node metadata: author, language, quality scores, timestamps.
  - Conversations are navigable paths from root to leaf.

### Quality control
- Multiple labelers per message.
- Spam/toxicity filter — human + automated.
- Language-tagged with confidence score.
- Messages below quality threshold flagged.

### Dataset releases
- **OASST1 (2023):** 161K messages, 10K conversation trees.
- **OASST2 (2024):** ~250K+ messages.
- License: CC-BY 4.0.

- **Output shape:** tree-structured multi-turn conversations; avg path length 4–6 turns; many branching candidate replies per node.
- **Teacher model:** none — fully crowdsourced.
- **Cost:** volunteer labor; no API cost.

## Modality-specific technical details (REQUIRED — conversation)
- **Turn-count distribution:** avg path 4–6 turns; some paths to 20+.
- **Speaker-role protocol:** alternating human-written user + human-written assistant turns.
- **Persona conditioning:** implicit — volunteers bring their own personas.
- **Safety post-filter:** crowdsource labelers flag harmful content; automated toxicity filter applied.
- **Language coverage:** 35 languages — English dominant (~45%), then Spanish, Chinese, German, Russian, Russian, others.
- **Tree format:** multiple candidate replies at each node enable preference-data extraction (OASST-ranking).

## Quality / diversity evaluation
- Diversity: substantially higher than Baize/UltraChat on lexical and persona-diversity metrics.
- User turns feel more natural (typos, slang, abrupt topic changes) — the "synthetic user" limitation of Baize/UltraChat is absent.
- Used as SFT + preference-data source for many models: LLaMA2-OASST fine-tunes, Mistral-OASST.

## Risks + gotchas
- **Crowdsourced quality is uneven** — some replies are empirically worse than ChatGPT's.
- **Demographic skew:** contributors skew Western/English/technical; some languages have very sparse coverage.
- **Size ceiling:** ~161K (OASST1) and ~250K (OASST2) are small vs 1.5M UltraChat, millions of WildChat messages.
- **Non-commercial concerns:** CC-BY allows commercial use but attribution required.

## Connections
- Real-human sibling: [[wildchat]] (passive user-chat-log collection).
- Synthetic contrasts: [[baize-construction]], [[ultrachat-pipeline]], [[camel]].
- Preference-data use: OASST reply rankings used in [[hh-rlhf]]-style RM training.
- Multilingual coverage makes it the reference for cross-lingual dialog training.
