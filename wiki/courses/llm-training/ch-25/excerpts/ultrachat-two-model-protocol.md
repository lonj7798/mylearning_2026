---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: Ding et al. 2023 — "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations"
source_url: https://arxiv.org/abs/2305.14233
created_at: "2026-04-23"
---

# Excerpt: UltraChat — The Three-Sector Taxonomy and the Two-Model Protocol

**Source:** `wiki/raw-data/llm-training/papers/ultrachat-construction.md`, `wiki/raw-data/llm-training/papers/ultrachat-pipeline.md`
**Primary paper:** Ding et al. (Tsinghua + OpenBMB), 2023
**arXiv:** https://arxiv.org/abs/2305.14233 ; dataset: https://huggingface.co/datasets/stingning/ultrachat

---

## Why this source anchors ch-25 §2

UltraChat is the paper that turns multi-turn dialogue synthesis into a *design problem* rather than a trick. Where Baize picks seeds from existing pools and relies on one LLM's self-coherence, UltraChat (1) **pre-enumerates a topic taxonomy** before any generation happens, and (2) **uses two separate model calls** for user and assistant. Both moves become the default for every subsequent large-scale pipeline.

From the source:

> *"The dataset card says UltraChat uses `two separate ChatGPT Turbo APIs`, one for generating user queries and one for generating responses."*

> *"The repo also states that the system does not directly use internet prompts as prompts for generation, except that one sector starts from existing source material."*

---

## The three-sector taxonomy — verbatim structure

From [[ultrachat-pipeline]]:

> *"Sector 1 — Questions about the World: starts from 30 representative meta topics, expands them into 1100+ subtopics, for each subtopic generates up to 10 specific questions."*
>
> *"A second branch covers common entities: the repo says they gather the top-frequent 10,000 named entities from Wikidata, generate 5 meta-questions per entity, then expand each into 10 specific questions and 20 related general questions."*
>
> *"Dialogues in this sector are then rolled out as 3-7 rounds using the two models iteratively."*

> *"Sector 2 — Writing and Creation: starts from 20 writing/creation types. For each type, generates 200 instructions that ask an assistant to create text. The repo states that 80% of these instructions are expanded and detailed further. Each generated instruction becomes the initial user input for a 2-4 round dialogue."*

> *"Sector 3 — Assistance on Existent Materials: starts from existing source passages rather than pure blank-sheet prompts. The repo says it extracts about ~100k diverse materials from C4. For each material, the pipeline generates up to 5 questions or instructions. It then combines the material with those questions via manually designed templates to form the initial user turn. Each such seed becomes a 2-4 round dialogue."*

**Notice:** the taxonomy is not a post-hoc categorization of generated data. It is the *input* to generation. The counts (30 / 1,100+ / 10K / 20 / 100K) are all *pre-decided*. This is the "taxonomy-first" pattern — enumerate coverage before generating content.

---

## The two-model loop — why two calls

From [[ultrachat-pipeline]]:

> *"Top-level pipeline: (1) Define a broad conversation taxonomy split into three sectors. (2) Build sector-specific prompt scaffolds instead of one generic generation prompt. (3) Use one model call to generate or simulate the next user turn. (4) Use a second model call to answer as the assistant. (5) Iterate this process to form a multi-turn conversation. (6) Apply post-processing and filtering before release."*

The critical step is (3)+(4): **two calls per turn**. Compare with Baize, which makes *one call for the entire dialogue*. The consequences of the split:

- **Friction.** The assistant LLM does not see the user LLM's internal planning; it only sees the emitted user turn. If the emitted turn is ambiguous or slightly off-topic, the assistant *reacts* to what was actually said, introducing realistic misunderstanding.
- **Role specialization.** The user-side system prompt can instruct *"behave as a user: short, sometimes imprecise, curious"* without the assistant-side generation inheriting that style. Baize's single call cannot separate these.
- **Cost.** Two calls per turn × 4–7 turns ≈ 8–14 API calls per dialogue. At 1.5M dialogues this multiplied into a much larger bill than Baize's 111.5K × 1 call, but by 2023 Turbo pricing it was still tractable.

The trade-off Baize chose (one call, save cost, accept homogeneity) and UltraChat chose (two calls, pay more, gain friction) is the same trade-off every production pipeline re-litigates at scale.

---

## Multi-turn by history-conditioning

From [[ultrachat-pipeline]]:

> *"Multi-turn construction logic: UltraChat matters because it is not just synthetic instruction-response pairs. The public recipe repeatedly conditions on dialogue history so the generated user asks follow-up questions, requests revisions, changes constraints, or continues discussion after seeing the assistant answer."*

The generation loop (paraphrased as code):

```
history = [seed_user_turn]                     # from sector scaffold
for round in range(R_s):                       # R_1 ∈ [3,7], R_{2,3} ∈ [2,4]
    a = assistant_LLM.generate(system=assistant_prompt_s,
                                history=history)
    history.append(a)
    u = user_LLM.generate(system=user_prompt_s,
                           history=history)
    history.append(u)
```

Each turn conditions on the full dialogue so far. This is what makes the user turn a *follow-up* (asking revisions, changing constraints, continuing) rather than a topic-independent restart. Without history-conditioning, the user LLM would produce a sequence of unrelated questions — the pipeline would generate K independent single-turn dialogues, not a K-turn multi-turn dialogue.

---

## What is *not* published — the filter stage

From [[ultrachat-pipeline]]:

> *"Filtering and post-processing: the public sources confirm that the generated dialogues undergo `post-processing and filtering`, but they do not publish a thresholded quality-control pipeline with exact rejection rules, reward-model scoring, or classifier-based filters."*

This is the canonical gap. UltraChat publishes its taxonomy and its generation loop transparently; its filter is a blackbox. The practitioner reading is:

- **Generation protocol is transferable.** You can replicate UltraChat's sector structure with any teacher pair today (GPT-4o, Claude Sonnet, Llama-3.1-405B).
- **Filter heuristics are teacher-specific.** Whatever UltraChat threw away was tuned against GPT-3.5-Turbo's 2023-era failure modes. Reapplying the same thresholds to modern teachers would discard different failures, keep others.

---

## Data shape

From [[ultrachat-pipeline]]:

> *"Data shape: the released dataset card says each example is a JSON dictionary with:*
> *- `id`: sample identifier*
> *- `data`: a list of alternating turns*
>
> *The list format stores the conversation as raw utterance strings rather than message objects with explicit roles. The public preview shows list lengths from 4 to 14."*

List lengths 4–14 is consistent with `R_s ∈ [2, 7]` rounds × 2 turns per round. A preview with a length-6 list is a 3-round Q-world dialogue; a length-14 is a 7-round tail-end of the Q-world distribution.

**Notice:** the flat-list format (no explicit role tags in the data) means downstream consumers must reconstruct roles by position (even index = user, odd = assistant). This is a minor but real source of bugs in downstream mixers that mis-align the alternation.

---

## Why UltraChat's moves became the default

- **Taxonomy-first.** Every 2024+ pipeline (Persona-Hub, GLAN, Magpie variants, SystemChat) enumerates *something* before generating. The "something" varies (topics, personas, constraints, tasks) but the enumerate-first pattern is universal.
- **Two-model user/assistant split.** Carried into CAMEL, APIGen-MT, ToolACE; inverted in some cases (one model with two role prompts switched in the same call) but the *separation of the generator* from the *role* is now standard.
- **Split-by-family release.** Downstream mixes weight sectors independently. [[smol-talk]] uses only UltraChat's Q-world portion, weighting writing and assistance lower.

---

## Connections

- Chapter synthesis: [[ch-25]] §2.
- Precursor with one model: [[excerpts/baize-self-chat]].
- Role-pair diversity successor: [[excerpts/camel-inception-prompting]].
- Persona-conditioned 2024 extension: [[excerpts/system-prompt-diversity]].
