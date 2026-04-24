---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: Kim et al. 2022 — "ProsocialDialog: A Prosocial Backbone for Conversational Agents"
source_url: https://arxiv.org/abs/2205.12688
created_at: "2026-04-23"
---

# Excerpt: Prosocial-Dialog — Rules-of-Thumb as a Safety Grounding Signal

**Source:** `wiki/raw-data/llm-training/papers/prosocial-dialog.md`
**Primary paper:** Kim, Yu, Jiang, Lu, Khashabi, Kim, Choi, Sap (Allen AI / UW — Yejin Choi group), 2022
**arXiv:** https://arxiv.org/abs/2205.12688

---

## Why this source anchors ch-25 §4 (safety branch)

Prosocial-Dialog is the safety counterpart to SODA — same Yejin Choi group, same "ground the bootstrap on a structured external signal" pattern, but the signal is now **rules-of-thumb (RoTs)** instead of commonsense triples, and the objective is behavioral (teach the assistant to engage constructively with difficult prompts) instead of register-shaping.

From the source:

> *"Making dialogue agents prosocial (helpful in socially problematic situations, not just harmless) requires training data where the assistant engages with difficult prompts (biased comments, unethical plans) and responds with socially-grounded rules-of-thumb (RoTs) rather than refusals; the RoTs provide a learnable scaffold for nuanced safety behavior."*

The load-bearing claim: **refusal is not safety**. Pure-refusal data teaches the assistant to emit "I can't help with that" but does not teach it to reason about *why*. Prosocial-Dialog teaches the reasoning first, the response second.

---

## The pipeline — 5 steps, human-in-the-loop

From [[prosocial-dialog]]:

> *"Step 1 — Problematic-prompt collection*
> *Crowd workers and domain experts author prompts covering 10 problem categories: stereotypes, harms, insults, conspiracies, safety risks, unethical plans, etc.*
> *~10K initial prompts."*

> *"Step 2 — RoT annotation*
> *For each prompt, crowd workers identify applicable rules-of-thumb. Example RoTs:*
> *- 'It's rude to mock someone's appearance.'*
> *- 'Planning to hurt yourself is a safety concern.'*
> *- 'Stereotyping based on race is harmful.'*
> *300+ unique RoTs emerge across the dataset."*

> *"Step 3 — Response drafting*
> *Teacher LLM (GPT-3) generates candidate assistant responses grounded in the RoT.*
> *Multiple candidates per prompt."*

> *"Step 4 — Human refinement*
> *Crowd workers edit/rank responses for prosocial quality, clarity, and RoT alignment.*
> *Only high-quality responses retained."*

> *"Step 5 — Multi-turn extension*
> *Expand single-turn prompts into multi-turn dialogs via back-and-forth between follow-up generation and human editing."*

**Notice:** Steps 1, 2, 4, 5 all involve humans. This is a **semi-synthetic** pipeline, not a pure teacher-bootstrap one. The human gate is why the dataset is small (58K) compared to SODA's 1.5M — you can't easily crowd-scale rule identification and editing.

---

## The RoT layer — what it supplies

Rules-of-thumb are short, declarative ethical statements. A few from the raw-data notes:

- *"It's rude to mock someone's appearance."*
- *"Planning to hurt yourself is a safety concern."*
- *"Stereotyping based on race is harmful."*

Each dialogue is anchored to one or more RoTs. The student model trains on turns where the assistant **verbalizes the RoT before responding**:

```
User (problematic prompt): I want to make fun of my coworker's accent in the meeting tomorrow.
Assistant (prosocial):     I get that workplace frustrations can build up, but making fun of
                           someone's accent is a form of mockery that can hurt people and damage
                           trust. What's actually going on between you two — is there a
                           disagreement I can help you think through?
```

Three properties of the assistant turn:

1. **It does not refuse.** "I can't help with that" is not what Prosocial teaches.
2. **It verbalizes the applicable norm.** "Making fun of someone's accent is a form of mockery…" — the RoT is surfaced.
3. **It redirects.** "What's actually going on between you two…" — engages the underlying need that generated the problematic prompt.

This pattern — *engage-and-redirect with explicit norm* — is the direct precursor to Anthropic's Constitutional AI ([[constitutional-ai]]) and to the HHH (Helpful-Harmless-Honest) framing. In both cases the assistant is taught to *reason about* a norm before applying it, not to pattern-match on trigger words.

---

## Turn-count statistics

From [[prosocial-dialog]]:

> *"Turn-count distribution: median 3 turns (prompt, prosocial response, follow-up)."*

> *"Speaker-role protocol: problematic user + prosocial assistant."*

Median 3 is the shortest of any corpus in ch-25. Why: the dataset's pattern is `(user problem → assistant engages with RoT → user reacts)`, which is a natural 3-turn shape. Longer dialogues exist where the follow-up leads to further probing, but the modal instance is short. This is appropriate for the task — you want the student model to learn the norm-engagement pattern in its canonical form, not to ramble across 10 turns.

---

## The 10 harm categories

From [[prosocial-dialog]]:

> *"Taxonomy of harms: 10 top-level categories (stereotyping, insults, self-harm, violence planning, misinformation, etc.)."*

The taxonomy is load-bearing in the same way UltraChat's three sectors are load-bearing — it is the *diversity lever*. Without a hand-designed taxonomy of harms, crowd workers would skew toward a few common categories (stereotyping, insults) and the dataset would underrepresent rarer but important categories (misinformation, self-harm crisis). Forcing balanced coverage across 10 categories is what makes the dataset generalize.

---

## CANARY — the downstream result

From [[prosocial-dialog]]:

> *"CANARY-400M trained on Prosocial-Dialog: engages constructively with 89% of problematic prompts, vs BlenderBot-3B at 32%."*

The 89% vs. 32% gap is the strongest single-number evidence that RoT grounding works. At ~10× fewer parameters (400M vs. 3B), CANARY engages with problematic prompts 2.8× more often than the baseline. This is not a marginal effect; it is a phase change in behavior.

Note the specific metric: *engages constructively*. Not *refuses*, not *complies with unsafe request*, but engages with a norm-aware response. Pure-refusal BlenderBot would score very high on "avoids unsafe output" and very low on this metric; pure-compliance GPT-3 would score low on "avoids unsafe output" and possibly high on "engages" but unsafely. CANARY is the first model where *both* axes are optimized simultaneously.

---

## The precursor-to-Constitutional-AI relationship

From the raw-data notes:

> *"Conceptual kin: [[constitutional-ai]] (Anthropic — rule-guided AI safety)."*

The structural parallel:

- Prosocial-Dialog: human identifies RoT, GPT-3 drafts response grounded in RoT, human refines.
- Constitutional AI: Anthropic writes a *constitution* (list of principles), Claude critiques and revises its own responses against the constitution.

Prosocial is the human-in-the-loop version; Constitutional AI is the fully-model-in-the-loop version. The RoT → constitution transition is what happens when you scale the "structured external norm" signal from a crowdsourced 300-RoT list to a hand-written ~50-principle constitution. The grounding axis stays the same; the provenance of the norms changes.

---

## Gotchas

From [[prosocial-dialog]]:

> *"Crowdsourcing biases: RoT selection reflects US/English-speaking contributor norms."*

> *"Prosocial !≠ refusal: the dataset deliberately engages; downstream users who want pure refusal behavior need additional data."*

> *"RoTs can be over-specified: real conversations blur multiple norms."*

> *"Size (58K) is small vs modern dialog corpora."*

The size limitation is the big one — 58K is ~0.4% of UltraChat's 1.5M. Modern safety-SFT mixes use Prosocial as a *seed* slice (5–15% of the safety sub-mix) with the remainder filled by synthetic extensions (WildGuard, Anthropic-internal persona datasets, Constitutional-AI-style self-critique outputs).

---

## Connections

- Chapter synthesis: [[ch-25]] §4.
- Companion paper from same group: [[excerpts/soda-commonsense-grounding]].
- Successor at model scale: Constitutional AI.
- Modern safety-dialog successor: [[wildguard-data]] (Allen AI).
