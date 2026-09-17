---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: Anthropic engineering blog, "Effective context engineering for AI agents" (no library card as of 2026-09-15)
source_url: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
created_at: "2026-09-15"
---

# Excerpt: Effective context engineering for AI agents

**Publisher:** Anthropic Applied AI team (Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, Jeremy Hadfield, with contributions from Rafi Ayub, Hannah Moran, Cal Rueb, Connor Jennings). Published 2025-09-29. Source type: official blog (no experiments or numbers reported for the claims below unless stated).

## Definitions (sections "Context engineering vs. prompt engineering", "Why context engineering is important")
- "Context refers to the set of tokens included when sampling from a large-language model (LLM)."
- "Context engineering refers to the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference, including all the other information that may land there outside of the prompts."
- Agent definition used in the post: "LLMs autonomously using tools in a loop."

## Attention budget (section "Why context engineering is important to building capable agents")
- The post names the effect measured by needle-in-a-haystack work: "as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases." It attributes the term *context rot* to that line of benchmarking.
- Stated mechanism, given as reasoning rather than measurement: transformers let "every token attend to every other token across the entire context. This results in n² pairwise relationships for n tokens", and "models develop their attention patterns from training data distributions where shorter sequences are typically more common than longer ones."
- The post calls the consequence "an 'attention budget'" and states the shape of the degradation: "These factors create a performance gradient rather than a hard cliff: models remain highly capable at longer contexts but may show reduced precision for information retrieval and long-range reasoning compared to their performance on shorter contexts."
- Guiding principle stated twice: "find the smallest set of high-signal tokens that maximize the likelihood of your desired outcome."

## Retrieval strategy (section "Context retrieval and agentic search")
- Load-on-demand strategy: agents "maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools."
- Claude Code is described as a hybrid: `CLAUDE.md` files are placed in context up front, while `glob` and `grep` retrieve files during the run.
- Stated trade-off: "runtime exploration is slower than retrieving pre-computed data", and without guidance "an agent can waste context by misusing tools, chasing dead-ends, or failing to identify key information."

## Long-horizon techniques (section "Context engineering for long-horizon tasks")
Three techniques are named, with no comparative numbers.
1. **Compaction** — "taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary." In Claude Code the message history is passed to the model to summarize; "The model preserves architectural decisions, unresolved bugs, and implementation details while discarding redundant tool outputs or messages. The agent can then continue with this compressed context plus the five most recently accessed files."
   - Tuning advice: "Start by maximizing recall to ensure your compaction prompt captures every relevant piece of information from the trace, then iterate to improve precision by eliminating superfluous content."
   - Risk stated: "overly aggressive compaction can result in the loss of subtle but critical context whose importance only becomes apparent later."
   - "One of the safest lightest touch forms of compaction is tool result clearing, most recently launched as a feature on the Claude Developer Platform."
2. **Structured note-taking (agentic memory)** — the agent writes notes outside the context window and reads them back after a context reset. Example given is Claude playing Pokémon, which maintains tallies such as "for the last 1,234 steps I've been training my Pokémon in Route 1, Pikachu has gained 8 levels toward the target of 10."
3. **Sub-agent architectures** — a sub-agent "might explore extensively, using tens of thousands of tokens or more, but returns only a condensed, distilled summary of its work (often 1,000-2,000 tokens)."

Selection rule stated by the authors: "Compaction maintains conversational flow for tasks requiring extensive back-and-forth; Note-taking excels for iterative development with clear milestones; Multi-agent architectures handle complex research and analysis where parallel exploration pays dividends."

## Not reported by the source
- No benchmark table, no ablation, no token counts for compaction thresholds, no model sizes. Every claim above is either a definition, a design description, or a qualitative recommendation.

## Verification
- Read on 2026-09-15 from the page text at https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents (published 2025-09-29).
- Related official post with numbers: [[anthropic-context-management]] (same date, product announcement).
