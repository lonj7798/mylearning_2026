<!-- scope: UltraChat generation pipeline — how the ~1.5M multi-turn dialogue corpus is built
     deps: [[ultrachat-pipeline]]
     see-also: [[baize]], [[persona-hub]], [[smol-talk]]
-->

# UltraChat Construction: Pipeline for Scaled Multi-Turn Dialogue Synthesis
- **Core Insight:** Large-scale multi-turn dialogue data can be synthesized entirely from scratch by running two separate LLMs (one role-playing a user, one as the assistant) across a taxonomy of **three topic families** (questions-about-the-world, writing-and-creation, assistance-on-existing-materials) — 1.5M dialogues with no human seeds.
- **Guideline:** To produce a UltraChat-like dialogue corpus, (1) pre-enumerate a topic taxonomy, (2) use one LLM to draft a user's opening query conditioned on a topic, (3) use a second LLM as assistant, (4) have the user LLM draft the next user turn conditioned on the assistant reply, (5) iterate for N turns; release by topic split for downstream flexibility.
- **Authors:** Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, Bowen Zhou (Tsinghua + OpenBMB)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.14233 ; https://huggingface.co/datasets/stingning/ultrachat
- **Relevant topics:** multi-turn dialogue synthesis, topic taxonomy, UltraLM, UltraChat

## Abstract
UltraChat is a ~1.5M multi-turn dialogue corpus generated without any human dialogue seed. The construction is organized by three topic families — (a) questions about the world, (b) writing and creation, (c) assistance on existing materials — each with its own meta-prompt structure. Two separate API calls role-play user and assistant. Dialogues proceed until a natural conclusion. The resulting corpus is the training substrate for UltraLM and a substrate in countless downstream mixes (Zephyr, Starling, Tülu variants).

## Construction pipeline (concrete)

### Three topic families + their construction
- **Family 1 — Questions about the World (~100+ topics):**
  - Meta-prompt: enumerate sub-topics under each topic ("History", "Physics", "Pop Music", ...).
  - User LLM: crafts an opening question about a sub-topic.
  - Assistant LLM: answers.
  - Multi-turn continuation: user asks follow-ups.

- **Family 2 — Writing and Creation:**
  - Meta-prompt: enumerate creation types (essay, poem, script, email, code, recipe, ...).
  - User LLM: asks for a creation of a specific type on a specific theme.
  - Assistant: drafts; user asks for revisions; iterative editing.

- **Family 3 — Assistance on Existing Materials:**
  - Pre-collected real text passages (public domain / Wikipedia / news).
  - User LLM: asks questions about the passage (summarize, translate, rewrite, answer questions).
  - Assistant: responds grounded in the passage.
  - Multi-turn: follow-up questions about the same passage.

### Two-LLM role-play
- User role: a dedicated prompt instructs the LLM to behave as a user (short, sometimes imprecise, curious).
- Assistant role: standard assistant prompt (helpful, detailed).
- Role-play uses GPT-3.5-Turbo for both sides (though the method is model-agnostic).

### Pipeline specifics
- Each family gets its own prompt scaffold and meta-topic list.
- Dialogue length controlled by temperature + explicit turn caps + "end if user indicates done."
- No external judge; no filter beyond format validity and length.
- Topic taxonomy is the primary diversity lever.

## Output shape
- ~1.5M multi-turn dialogues total.
- Released by family: UltraChat Q-world, UltraChat Writing, UltraChat Assistance.
- Avg ~4 turns per dialogue; heavy-tailed.

## Downstream impact
- UltraLM-13B (fine-tuned on UltraChat) was the strongest open-assistant 13B at release (Summer 2023).
- UltraChat is a standard ingredient in open SFT mixes (Zephyr β uses it as prompts for UltraFeedback generation).
- The taxonomy-based construction approach is a direct precursor to [[glan]]'s taxonomy-driven synthesis.

## Practitioner takeaways
- **Two-LLM role-play > one-LLM self-chat** (see [[baize]]) — two separate calls give more conversational friction.
- **Topic taxonomy replaces seeds** — a conceptual move that becomes standard in 2024 (Persona-Hub, GLAN).
- **Split-by-family release** is reusable — lets downstream users weight family contributions.
- Dialogues are *still* more coherent / articulate than real users — the "too-polite user" artifact persists.

## Risks + gotchas
- **User-realism gap:** role-played users are unusually articulate; downstream models may overfit to that register.
- **Topic-taxonomy bias:** what's *not* in the taxonomy is structurally absent.
- **GPT-3.5 teacher bias** across all families.
- **License:** research-use; derived from ChatGPT API outputs.

## Connections
- Data-card counterpart: [[ultrachat-pipeline]].
- Preference pipeline built atop UltraChat prompts: [[ultrafeedback-construction]].
- Earlier self-chat precursor: [[baize]].
- Topic-taxonomy successor: [[glan]].
- Common prompt source for subsequent mixes: [[smol-talk]], [[tulu-3-sft-mix]].
