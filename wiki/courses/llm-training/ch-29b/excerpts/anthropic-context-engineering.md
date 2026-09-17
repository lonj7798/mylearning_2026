---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/anthropic-context-engineering.md (library card not present on 2026-09-15; content taken from the primary source)
source_url: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
primary_version: Anthropic Engineering blog, published 2025-09-29 (read 2026-09-15)
created_at: "2026-09-15"
---

# Excerpt: Effective context engineering for AI agents (Anthropic, official blog)

Source type: official blog (organization that trains the models). No controlled experiments or numbers on training data; ch-29b uses it only to name context-management behaviors.

## Definitions
- "Context engineering refers to the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference."
- "context rot: as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases." The post also states that "models develop their attention patterns from training data distributions where shorter sequences are typically more common than longer ones."
- Agents: "LLMs autonomously using tools in a loop."

## Techniques for long-horizon tasks
- Compaction: "taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary." In Claude Code the model "preserves architectural decisions, unresolved bugs, and implementation details while discarding redundant tool outputs or messages", continuing with the summary "plus the five most recently accessed files".
- Tool result clearing: described as "One of the safest lightest touch forms of compaction".
- Structured note-taking: "the agent regularly writes notes persisted to memory outside of the context window. These notes get pulled back into the context window at later times."
- Sub-agents: each "might explore extensively, using tens of thousands of tokens or more, but returns only a condensed, distilled summary of its work (often 1,000-2,000 tokens)."
- Runtime retrieval: agents "maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools."
- Compaction prompt tuning: "Start by maximizing recall ... then iterate to improve precision by eliminating superfluous content."
