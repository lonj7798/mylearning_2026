---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: Xu et al. 2023 — "Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data"
source_url: https://arxiv.org/abs/2304.01196
created_at: "2026-04-23"
---

# Excerpt: Baize — The Self-Chat Prompt Template and Its Consequences

**Source:** `wiki/raw-data/llm-training/papers/baize.md`, `wiki/raw-data/llm-training/papers/baize-construction.md`
**Primary paper:** Canwen Xu, Daya Guo, Nan Duan, Julian McAuley (UCSD + Microsoft Research Asia), EMNLP 2023
**arXiv:** https://arxiv.org/abs/2304.01196

---

## Why this source anchors ch-25 §1

Baize is the chapter's zero point — the cheapest, simplest multi-turn synthesis pipeline that works at all. Every later pipeline in the chapter (UltraChat, CAMEL, SODA) is explicitly designed to beat one or more of Baize's limitations. Understanding Baize's single-call protocol is the baseline against which those limitations become legible.

From the source:

> *"Baize introduces a self-chat procedure where ChatGPT generates multi-turn conversations from a single seed. Starting from 111.5K seeds drawn from Quora, StackOverflow-Medical, Alpaca, and Medical-Question-Answering datasets, the pipeline produces 111.5K dialogs (each 4+ turns)."*

One seed → one API call → one multi-turn dialogue. That is the entire pipeline. The engineering ingenuity is compressed into the prompt template.

---

## The prompt template — verbatim

From [[baize-construction]]:

```
The following is a conversation between a human and an AI assistant.
[|Human|] <seed question>
[|AI|] <first ChatGPT response>
[|Human|] <ChatGPT playing user>
[|AI|] <ChatGPT playing AI>
...
```

Three mechanical choices to notice:

1. **The framing line is pre-written.** `"The following is a conversation between a human and an AI assistant."` This primes ChatGPT to recognize its task as *writing a dialogue*, not *being the assistant in a dialogue*. The shift is from producer-of-replies to producer-of-transcripts.
2. **The seed question is pre-inserted under `[|Human|]`.** The first user turn is not generated; it is the sampled seed. This is the only turn that escapes the teacher's style distribution.
3. **ChatGPT fills in the first `[|AI|]`, then the second `[|Human|]`, then the second `[|AI|]`, and so on.** A single completion; no intermediate round-trips; no conditioning on prior generations other than the continuously-growing prompt.

---

## The parsing step

Because the whole dialogue arrives in one text blob, parsing is regex on role markers. From the raw-data notes:

> *"Parsing: split the generated text on `[|Human|]` / `[|AI|]` markers to extract turn sequence."*

Failure cases — which drive the filter stage:

- ChatGPT omits a role marker mid-dialogue (merges two turns).
- ChatGPT invents a third role ("System: …") that does not parse.
- ChatGPT emits a non-ASCII variant of the pipe character, breaking regex.
- ChatGPT degenerates into repetition, producing `[|AI|]` responses that are verbatim copies of prior ones.

**Notice:** all of these failures are parse-level, not semantic-level. Baize's filter does not read the content of turns — it only checks that parsing succeeds and the result has ≥2 turn pairs and no obvious n-gram repetition. Relying on ChatGPT's self-consistency is explicit:

> *"No external judge; no filter beyond format validity and length."*

---

## Seed pools — where diversity actually comes from

From [[baize-construction]]:

> *"Seed sources:*
>  *- Quora: ~54K seed questions.*
>  *- StackOverflow: ~57K seed questions.*
>  *- Alpaca: ~52K (general instructions).*
>  *- MedQuAD (medical): ~47K seed medical questions."*

The diversity of Baize is the diversity of the seed pool. The teacher is a single fixed GPT-3.5-Turbo checkpoint; every stylistic axis (vocabulary, verbosity, formality) is held constant. What varies across the 111.5K dialogues is only the *topic* — set by the seed.

This is the sharpest structural contrast with UltraChat (ch-25 §2), which deliberately enumerates a topic taxonomy *and* varies the user/assistant system prompts, and with CAMEL (§3), which varies role pairs. Baize picks one axis; its successors pick several.

---

## The turn-count distribution — median 4, tail to 10

From the raw-data notes:

> *"Turn-count distribution: median 4, tail to 10. Fewer than crowdsourced dialogs (OASST, WildChat) which often have 15+."*

Why median 4: ChatGPT's default behavior when asked to "continue the conversation until it concludes naturally" is to wrap up after a short exchange. Without a termination signal tied to any semantic condition (task solved, question answered fully), the teacher uses its own aesthetic sense of "dialogue feels complete" — which peaks at 4 turns.

Why tail to 10: the prompt enforces a soft ceiling. ChatGPT will extend to 8–10 turns if the seed topic admits natural follow-ups (medical consultations, debugging dialogs) but rarely further. Real human conversations (OASST, WildChat) have power-law tails to 30+ turns because real users sometimes pursue a topic through multiple clarification rounds; the teacher has no analog of that behavior.

**Consequence for training.** A student model SFT'd on pure Baize will have seen few examples of late-dialogue turns (8+). When served a 10-turn conversation at inference, it produces replies that regress toward its pretraining distribution — because it never saw enough 10-turn examples to overwrite that prior.

---

## Self-Distill with Feedback (SDF) — the patch

From [[baize]]:

> *"Self-Distill with Feedback (SDF): (1) Fine-tune the student model on Baize. (2) Generate model outputs for a held-out prompt set. (3) Ask ChatGPT to rate / critique / rewrite them. (4) Use the rewritten pairs as additional SFT or as DPO pairs. (5) Iterate."*

SDF is the first widely-adopted teacher-as-judge distillation loop for multi-turn chat. It does not fix the user-realism gap (the held-out prompt set is still synthetic / Baize-style), but it does fix some teacher-specific style artifacts the student picked up. Conceptually, SDF is the ancestor of:

- RLAIF (Bai et al. 2022) — teacher provides preference judgments for RL.
- UltraFeedback (Cui et al. 2023) — teacher ranks multiple candidate responses.
- Constitutional AI (Anthropic) — teacher critiques against a rubric of rules.

---

## Why start the chapter here

Baize compresses all the interesting questions of multi-turn synthesis into one prompt template and forces the reader to think about each question concretely: who plays the user? (one LLM playing a role); how does dialogue terminate? (ChatGPT's own sense of completion); how is diversity produced? (seed variety only); how is quality controlled? (barely at all). Every subsequent section of ch-25 answers one of these questions differently.

---

## Connections

- Chapter synthesis: [[ch-25]]
- Sector-taxonomy successor: [[excerpts/ultrachat-two-model-protocol]]
- Role-pair alternative: [[excerpts/camel-inception-prompting]]
- Real-human baseline: [[excerpts/openassistant-tree]]
