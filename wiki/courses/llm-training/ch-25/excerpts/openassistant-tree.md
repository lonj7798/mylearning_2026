---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: Köpf et al. 2023 — "OpenAssistant Conversations — Democratizing Large Language Model Alignment"
source_url: https://arxiv.org/abs/2304.07327
created_at: "2026-04-23"
---

# Excerpt: OpenAssistant — The Conversation Tree as Evaluation Anchor

**Source:** `wiki/raw-data/llm-training/papers/openassistant.md`
**Primary paper:** Andreas Köpf et al. (LAION + community), 2023
**arXiv:** https://arxiv.org/abs/2304.07327
**Dataset:** https://huggingface.co/datasets/OpenAssistant/oasst1 , .../oasst2

---

## Why this source anchors ch-25 §5

OpenAssistant is not a synthesis pipeline. It is the **real-human baseline** against which every pipeline in ch-25 is measured. Two things make it the reference:

1. **Fully human-authored.** No teacher LLM anywhere in the loop. Every message is written by a human contributor.
2. **Tree-structured data format.** Every other corpus in the chapter flattens dialogue to a linear sequence of turns. OASST keeps the tree, which is what preserves the branching-factor signal that synthetic corpora lack.

From the source:

> *"A global crowdsourcing effort with a tree-structured conversation format (branching replies, multiple candidates per turn, quality ranking) can produce a genuinely-human dataset covering 35 languages; this is the reference 'real human dialogue' baseline against which all synthetic conversation datasets (Baize, UltraChat, SODA) are measured."*

---

## The tree structure — verbatim description

From [[openassistant]]:

> *"Tree structure:*
> *- Each node has a parent and multiple children.*
> *- Node metadata: author, language, quality scores, timestamps.*
> *- Conversations are navigable paths from root to leaf."*

A concrete shape:

```
root (user prompt)
 ├── assistant_reply_A      [quality 4.2]
 │    ├── user_followup_A1
 │    │    ├── assistant_reply_A1a  [quality 3.8]
 │    │    └── assistant_reply_A1b  [quality 4.1]
 │    └── user_followup_A2
 │         └── assistant_reply_A2a  [quality 3.6]
 └── assistant_reply_B      [quality 3.4]
      └── user_followup_B1
           └── assistant_reply_B1a  [quality 3.0]
```

Three data-structural properties that matter downstream:

1. **Multiple candidate replies per turn.** At any assistant-turn node, siblings are alternative replies written by different contributors to the same context. This is natural preference-data: rank the siblings by label score, get `(preferred, rejected)` pairs for DPO / reward-model training.
2. **Every node is quality-labeled.** Labelers rate each message on quality, helpfulness, harmlessness, spam, creativity. Multiple labelers per message; aggregate via median or Bayesian combination.
3. **Path = conversation.** A root-to-leaf traversal is one conversation. Different paths through the same tree yield different "conversations" with shared prefixes.

---

## Contribution roles — how the tree is filled in

From [[openassistant]]:

> *"Contribution types:*
> *- Prompter: write a new user turn at a conversation-tree node.*
> *- Assistant: write a candidate reply.*
> *- Labeler: rate replies on quality, spam, creativity, helpfulness, harm.*
> *- Ranker: order candidate replies by preference."*

A contributor does not write an entire conversation. They write *one turn* at an existing node. The tree grows via crowd collaboration: volunteer A writes a root prompt; volunteer B writes an assistant reply; volunteer C writes a different assistant reply to the same prompt (sibling); volunteer D writes a user follow-up under B's reply; etc. This is the fundamental structural difference from synthetic corpora — OASST's dialogues were never "written" as whole units by any single entity.

Implication for statistics: **branching factor** is real. Typical root nodes have 3–5 assistant-candidate children. Typical assistant-turn nodes have 2–4 user-follow-up children. Synthetic corpora have branching factor 1 by construction.

---

## Scale and coverage

From [[openassistant]]:

> *"OASST1 (2023): 161K messages, 10K conversation trees."*
> *"OASST2 (2024): ~250K+ messages."*
> *"License: CC-BY 4.0."*
> *"Language coverage: 35 languages — English dominant (~45%), then Spanish, Chinese, German, Russian, others."*

~250K total messages across OASST1+2 is small compared to UltraChat's 1.5M or SODA's 1.5M. Per-language counts shrink fast — some languages in the 35-language set have only hundreds of messages, sparse enough to be more "presence indicator" than "training signal."

The 35-language coverage is *native* — not machine-translated. Each language's messages are written by native-or-near-native contributors. This is what makes OASST the reference for cross-lingual dialogue training: synthetic multilingual corpora (translated UltraChat, etc.) are always suspect on idiom and register, while OASST's native authoring is ground truth.

---

## Turn-count and register — why OASST dominates on these axes

From [[openassistant]]:

> *"Turn-count distribution: avg path 4–6 turns; some paths to 20+."*

> *"User turns feel more natural (typos, slang, abrupt topic changes) — the 'synthetic user' limitation of Baize/UltraChat is absent."*

**Path length**: average 4–6 is comparable to UltraChat Q-world's 3–7-round range. The difference is the *tail*: OASST paths go to 20+ turns with a power-law-like decay. Synthetic corpora (Baize max 10, UltraChat max 14) have hard caps.

**User-turn realism**: the unquantified but load-bearing property. Real users:

- Use casual, formal, rude, confused registers (not all polite).
- Include typos (~3% rate vs. Baize's near-zero).
- Switch topics mid-thread.
- Produce short turns (median ~45 tokens, vs. Baize's 80+).
- Ask ambiguous or under-specified questions.

None of these are reproducible by a synthetic user-LLM. This is why OASST is the reference — it is the only distribution that contains all of these signals jointly.

---

## OASST as evaluation instrument

The operational use: when you generate a new synthetic conversation corpus, measure four statistics against OASST and look at the gaps.

| Statistic | OASST value | Typical synthetic value | What the gap signals |
|---|---|---|---|
| Median user-turn tokens | ~45 | 80–150 | Synthetic users are too articulate |
| Typo rate (user turns) | ~3% | <0.1% | Synthetic users are too clean |
| Max path length | 20+ | 10–14 | Synthetic caps squash the tail |
| Sibling-reply variance | high | ~zero (branching=1) | Synthetic has no branching |

If your new synthetic corpus matches OASST on these, congratulations — you have solved a hard problem. If it doesn't, at least you know where the gaps are, and can decide whether to (a) patch them with targeted synthesis (e.g., inject typos), (b) mix in OASST as a minority slice to dilute the gaps, or (c) accept the gaps as known limitations.

---

## Using OASST as a preference-data source

From [[openassistant]]:

> *"Preference-data use: OASST reply rankings used in [[hh-rlhf]]-style RM training."*

Because siblings at each assistant-turn node are ranked by labelers, OASST directly yields `(prompt, preferred_reply, rejected_reply)` triples. This is one of the few real-human preference datasets in the open ecosystem (HH-RLHF being the Anthropic analog). OASST-DPO and OASST-RM are both used as reward-model training data in downstream models — LLaMA2-OASST, Mistral-OASST — and as a SFT source in some mixes.

---

## Limitations

From [[openassistant]]:

> *"Crowdsourced quality is uneven — some replies are empirically worse than ChatGPT's."*

> *"Demographic skew: contributors skew Western/English/technical; some languages have very sparse coverage."*

> *"Size ceiling: ~161K (OASST1) and ~250K (OASST2) are small vs 1.5M UltraChat, millions of WildChat messages."*

The uneven-quality point is the relevant one for SFT use. ChatGPT-generated synthetic dialogues have a high floor (the teacher's baseline quality) and a low ceiling (teacher can't exceed itself). OASST has a wide distribution — excellent replies, mediocre replies, some bad replies. Using OASST as SFT without quality filtering will teach the student both — including the bad replies. Standard practice: filter to labeler-quality ≥ some threshold before SFT.

---

## Connections

- Chapter synthesis: [[ch-25]] §5.
- Synthetic contrasts for evaluation: [[excerpts/baize-self-chat]], [[excerpts/ultrachat-two-model-protocol]], [[excerpts/camel-inception-prompting]].
- Real-human sibling (passive collection): [[wildchat]].
- Preference-data use: feeds HH-RLHF-style RM training.
