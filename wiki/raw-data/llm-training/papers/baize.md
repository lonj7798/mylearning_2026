<!-- scope: ChatGPT-self-chat synthetic dialogue corpus (Xu 2023, EMNLP 2023)
     deps: [[self-instruct]]
     see-also: [[ultrachat-pipeline]], [[capybara]]
-->

# Baize: Self-Chat Synthetic Dialogue for Multi-Turn SFT
- **Core Insight:** To get multi-turn dialogue data, have ChatGPT role-play both sides of a conversation seeded by a topic; the resulting "self-chat" transcripts can fine-tune an open base LM into a competitive multi-turn chatbot without any human conversation logs.
- **Guideline:** For multi-turn SFT in the absence of real user dialogues, seed a strong LLM with a topic + "simulate a conversation between a user and an AI assistant about this topic," collect 100K self-chats, fine-tune an open base with LoRA.
- **Authors:** Canwen Xu, Daya Guo, Nan Duan, Julian McAuley (UCSD + MSRA)
- **Year:** 2023 (EMNLP 2023)
- **URL:** https://arxiv.org/abs/2304.01196
- **Relevant topics:** self-chat, multi-turn dialogue, ChatGPT distillation, LoRA SFT

## Abstract
Baize proposes a self-chat pipeline: prompt ChatGPT to simulate both user and AI-assistant roles on a seed topic, producing coherent multi-turn dialogues. They collect 100K self-chat dialogues across Quora / StackOverflow / Alpaca / medical seed pools, then LoRA-fine-tune LLaMA-7B/13B/30B. The resulting Baize models showed meaningful multi-turn capability at a time when most open models were single-turn only. The paper also introduces **Self-Distill with Feedback** to further refine the models using ChatGPT feedback.

## Key Contributions
- **Self-chat pipeline** — one LLM playing two roles is the dialogue-data-gen primitive.
- Released 100K self-chat dialogues seeded from Quora (general), StackOverflow (technical), Alpaca (instructional), and medical topics.
- LoRA-tuned Baize-7B/13B/30B weights.
- Self-Distill with Feedback extension.

## Synthesis pipeline (concrete)
- **Seed input:** a topic string (e.g., a Quora question title, a StackOverflow title, an Alpaca instruction, or a medical subject).
- **Generation step:** prompt template (paraphrased):
  ```
  Please simulate a conversation between a user and an AI assistant about the following topic:
  {topic}
  The user speaks first. Continue the conversation for a few turns until it concludes naturally.
  ```
  ChatGPT is the single generator; it alternates `[User]` and `[AI]` turns in-trace.
- **Parsing:** split turns by the role labels; validate dialogue structure.
- **Filtering:** drop dialogues <3 turns, malformed trace, or looping/degenerate responses.
- **Teacher model:** ChatGPT (GPT-3.5-Turbo).
- **Output shape:** ~100K multi-turn dialogues across 4 seed pools.

## Self-Distill with Feedback (SDF) extension
- Fine-tune the student model (e.g., Baize-7B).
- Generate model outputs for a held-out prompt set.
- Ask ChatGPT to rate / critique / rewrite them.
- Use the rewritten pairs as additional SFT or as DPO pairs.
- Iterate.

## Training outcome
- Baize-7B/13B/30B at release were among the first viable open multi-turn chatbots.
- Demonstrated in 2023 that dialogue doesn't require real user logs — self-chat is sufficient.
- Now superseded by modern mixes (UltraChat, Tülu 3), but the self-chat primitive is still used.

## Risks + gotchas
- **Homogeneity:** one LLM playing both sides produces unnaturally coherent, low-disagreement dialogues.
- **User-realism gap:** real user turns are shorter, messier, full of context-switches; self-chat users are eerily articulate.
- **License:** ChatGPT outputs governed by OpenAI TOS; research-use-only at release.

## Connections
- Ancestor of large-scale self-chat work like [[ultrachat-pipeline]] (200K+ dialogues) and UltraLM.
- Conceptually adjacent to [[rlcd]]'s "same model, contrastive prompt" trick.
- Integrated (small amounts) into later mixes like [[openhermes]] community catalogues.
- Historically important as the first widely-reproduced open multi-turn SFT corpus.
