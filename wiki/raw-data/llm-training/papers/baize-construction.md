<!-- scope: multi-turn conversation synthesis — ChatGPT self-chat protocol from a seed question
     deps: [[self-instruct]]
     see-also: [[ultrachat-pipeline]], [[camel]], [[soda]]
-->

# Baize (Construction): Self-Chat Protocol for Multi-Turn Dialogue Synthesis
- **Core Insight:** Large-scale multi-turn conversation data can be generated cheaply by having a single LLM **self-chat** — ChatGPT plays both user and AI roles in alternation from a seed question — producing multi-turn dialogues at orders-of-magnitude lower cost than crowdsourcing or curated collection.
- **Guideline:** For cheap multi-turn SFT data, use self-chat: take a seed question from Quora/StackOverflow, prompt ChatGPT to continue the conversation alternating [User] and [AI] tags, cap the dialogue at ~4–8 turns; filter minimally — ChatGPT's self-consistency is usually enough.
- **Authors:** Canwen Xu, Daya Guo, Nan Duan, Julian McAuley (UCSD + Microsoft Research Asia + SYSU)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2304.01196
- **Relevant topics:** self-chat, multi-turn synthesis, dialogue, Baize

## Abstract
Baize introduces a self-chat procedure where ChatGPT generates multi-turn conversations from a single seed. Starting from 111.5K seeds drawn from Quora, StackOverflow-Medical, Alpaca, and Medical-Question-Answering datasets, the pipeline produces 111.5K dialogs (each 4+ turns). Baize-7B/13B/30B (LLaMA fine-tunes) outperforms Alpaca-7B on multi-turn dialogue quality at a fraction of the Self-Instruct generation cost.

## Key Contributions
- **Self-chat pipeline** — first widely-adopted method for cheap multi-turn dialogue synthesis.
- **Baize dataset** — 111.5K ChatGPT self-chat dialogues, public.
- Demonstration that multi-turn capability transfers from self-chat SFT.
- Ancestor of UltraChat, OpenHermes multi-turn, and many later dialogue corpora.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed sources:**
  - **Quora:** ~54K seed questions.
  - **StackOverflow:** ~57K seed questions.
  - **Alpaca:** ~52K (general instructions).
  - **MedQuAD (medical):** ~47K seed medical questions (for Baize-Healthcare variant).
- **Step 1 — Seed selection:** pick one seed question per generation.
- **Step 2 — Self-chat prompt:** issue a single API call with a prompt template instructing ChatGPT to continue the dialogue in alternation:
  ```
  The following is a conversation between a human and an AI assistant.
  [|Human|] <seed question>
  [|AI|] <first ChatGPT response>
  [|Human|] <ChatGPT playing user>
  [|AI|] <ChatGPT playing AI>
  ...
  ```
  ChatGPT continues both sides until a termination marker or turn cap (~8 turns).
- **Step 3 — Parsing:** split the generated text on `[|Human|]` / `[|AI|]` markers to extract turn sequence.
- **Step 4 — Filtering:**
  - Length check (at least 2 full turn pairs).
  - No degeneration / repetition (simple n-gram check).
  - No API-error markers.
- **Output shape:** 111.5K dialogues, avg 4.5 turns each, avg ~100 tokens per turn.
- **Teacher model:** ChatGPT (gpt-3.5-turbo, 2023).
- **Cost:** ~$1,000 in API (self-chat uses one API call per dialog).

## Modality-specific technical details (REQUIRED — conversation)
- **Turn-count distribution:** median 4, tail to 10. Fewer than crowdsourced dialogs (OASST, WildChat) which often have 15+.
- **Speaker-role protocol:** single model plays both sides via prompt-template alternation.
- **Persona conditioning:** implicit via seed question domain (medical, technical, general).
- **Safety post-filter:** none — authors rely on ChatGPT's built-in safety.
- **Self-chat failure modes:** ChatGPT-as-user sometimes asks questions that are too leading or echo AI responses; modest filtering drops obvious cases.

## Quality / diversity evaluation
- Baize-7B matches Alpaca-7B on single-turn MMLU; outperforms on multi-turn MT-Bench.
- Human eval: Baize-13B responses preferred over Alpaca-13B 58% of the time.
- Diversity: ROUGE-L between seed questions ≤ 0.15 (sufficient diversity).

## Risks + gotchas
- **Self-chat monologue bias:** both sides of the dialog exhibit ChatGPT-style verbosity and formatting tics.
- **User role quality is weak:** ChatGPT-as-user often asks vague or overly formal questions, unlike real users.
- **No true persona diversity** — user side is always ChatGPT's best guess at "a curious human".
- **Superseded in quality** by UltraChat (2-model chit-chat) and OpenAssistant (real humans).

## Connections
- Direct successor: [[ultrachat-pipeline]] (two-model self-chat + systematic 3-sector taxonomy).
- Role-play alternative: [[camel]] (multi-agent with distinct personas).
- Real-human baseline: [[openassistant]].
- Conceptual ancestor: [[self-instruct]] (both bootstrap from model generation).
