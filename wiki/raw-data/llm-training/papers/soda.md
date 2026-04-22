<!-- scope: multi-turn conversation synthesis — SODA commonsense-grounded dialogue via Atomic social roles
     deps: [[self-instruct]]
     see-also: [[prosocial-dialog]], [[camel]]
-->

# SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization
- **Core Insight:** Grounding synthetic dialogue in **ATOMIC 10X social commonsense triples** (person X wants Y because Z) + a narrative generation step produces 1.5M dialogues that feel substantially more human and emotionally grounded than topic-only prompted dialogs (UltraChat-style) — the commonsense context is what supplies the "why" behind each utterance.
- **Guideline:** For socially-grounded conversation data, start from a commonsense-triple database (ATOMIC-style), narrate a social scenario from the triple, then prompt the teacher to dialog within that scenario — this injects implicit emotional state, motives, and context that pure topic-prompting misses.
- **Authors:** Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, Ximing Lu, Youngjae Yu, Pei Zhou, Ronan Le Bras, Malihe Alikhani, Gunhee Kim, Maarten Sap, Yejin Choi (**Allen AI / UW — Yejin Choi group**)
- **Year:** 2023 (EMNLP)
- **URL:** https://arxiv.org/abs/2212.10465
- **Relevant topics:** commonsense-grounded dialogue, ATOMIC, SODA, Yejin Choi lineage

## Abstract
SODA (Social Dialogue with Commonsense) is a 1.5M-dialogue dataset distilled from GPT-3.5 with each dialogue grounded in a social-commonsense triple from **Atomic 10X**. Starting from 1.5M commonsense triples of the form `(person X, event, mental state / reason)`, SODA narrates each into a short social context, then prompts GPT-3.5 to generate a two-speaker dialogue realizing the context. The COSMO-3B model trained on SODA outperforms BlenderBot-3B and GODEL on conversational quality, demonstrating that commonsense grounding beats topic-only synthesis for natural dialog. Yejin Choi's group is the lineage behind Self-Instruct → SODA → Prosocial-Dialog — commonsense-grounded + safety-aware data construction is the through-line.

## Key Contributions
- **SODA-1.5M** — 1.5M commonsense-grounded dialogs, public.
- **Commonsense-grounding recipe** — narrative seed from Atomic 10X triple produces socially rich context.
- **COSMO-3B** — first open commonsense-conversational model.
- Demonstrated quantitative "socially natural" metric improvements over topic-only synthesis.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Step 1 — Commonsense-triple sampling
- **Source:** Atomic 10X (2022, Yejin Choi group) — 10M commonsense triples like `(PersonX fails the exam, xWant, to study harder)`, `(PersonX moves abroad, xEffect, feels lonely)`.
- Sample 1.5M triples balanced across relations (xWant, xNeed, xEffect, xReact).

### Step 2 — Narrative generation
- GPT-3.5 prompted with triple + instructions: "Write a 1–2 sentence narrative describing a social scene where this situation arises."
- Output: a narrative context containing PersonX + an interlocutor PersonY + a realistic scenario.

### Step 3 — Dialog generation
- GPT-3.5 prompted with the narrative + persona assignments: "Write a short conversation between PersonX and PersonY set in this scene."
- Output: 4–10-turn two-speaker dialog grounded in the scene.

### Step 4 — Filtering
- Coherence: turns must follow the scene.
- Length: dialog must be 3+ turns.
- No toxic content (GPT-3.5 internal safety + follow-up toxicity classifier).
- Persona consistency: PersonX must remain PersonX throughout.

- **Output shape:** 1.5M dialogs, avg 7.6 turns, avg 20 tokens per turn.
- **Teacher model:** GPT-3.5-turbo (Dec 2022).
- **Cost:** ~$10K API.

## Modality-specific technical details (REQUIRED — conversation)
- **Turn-count distribution:** median 7–8 turns, max ~12.
- **Speaker-role protocol:** two named speakers (PersonX, PersonY) with motives derived from triple.
- **Persona conditioning:** explicit via Atomic triple — each dialog inherits emotional state, want, need, reaction.
- **Safety post-filter:** toxic-content classifier on top of GPT-3.5's built-in safety.
- **Emotional grounding:** because triples contain mental states, dialogs implicitly convey emotional context — SODA's distinguishing property.

## Quality / diversity evaluation
- Human eval: COSMO-3B trained on SODA beats BlenderBot-3B on natural, engaging, specific dimensions.
- Diversity (unique bigrams): higher than Topical-Chat, similar to real dialog corpora.
- SODA dialogues rated higher on "emotionally grounded" than UltraChat / Baize.
- Strong transfer to out-of-distribution social dialog benchmarks (DailyDialog, EmpatheticDialogues).

## Risks + gotchas
- **Atomic-bias:** commonsense-triples reflect Atomic's (English, Western-centric) worldview; non-Western social contexts underrepresented.
- **Two-speaker only:** no group conversation, no task-oriented multi-agent.
- **GPT-3.5 teacher ceiling** on emotional nuance.
- **Narrative-grounded dialog may become formulaic** — "PersonX said X because..." patterns leak into students.

## Connections
- **Yejin Choi lineage:** [[self-instruct]] → SODA → [[prosocial-dialog]] (safety extension).
- Complements topic-driven synthesis: [[ultrachat-pipeline]], [[baize-construction]].
- Atomic ancestor: ATOMIC 2020 (Sap et al.) — commonsense-triple database.
- Real-human comparison: [[openassistant]].
