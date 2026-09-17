---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/msc-beyond-goldfish-memory.md (library card not present on 2026-09-15; content below is taken from the primary source)
source_url: https://arxiv.org/abs/2107.07567
primary_version: arXiv:2107.07567v1 (2021-07)
created_at: "2026-09-15"
---

# Excerpt: Beyond Goldfish Memory: Long-Term Open-Domain Conversation (Multi-Session Chat, MSC)

Authors: Jing Xu, Arthur Szlam, Jason Weston (Facebook AI Research). Checked against the v1 PDF on 2026-09-15. Used by ch-29b `read.md` §2.

## What the dataset is (§3)
- Human-human crowdworker chats over up to 5 sessions. Session 1 reuses PersonaChat; for later sessions a random gap of "either 1-7 hours or 1-7 days" is chosen, and workers play the same personas "as if that amount of time has transpired" (§3). The workers need not be the same people as in earlier sessions.
- Between sessions a separate crowdworker task writes summaries of "important personal points"; later sessions show these summaries instead of the full prior dialogue (§3, "Conversation Summaries (Extended Personas)").
- Training data: 4,000 episodes with 3 sessions and 1,001 episodes with 4 sessions; validation and test extend to 5 sessions "giving us a way to measure long-context session performance that extends beyond the training set distribution" (§3, Table 1).
- Length: ~53 utterances per 4-session training episode, ~66 per 5-session validation/test episode, and "an average of 1614 tokens" with the BlenderBot BPE (§3, Dataset Statistics).

## Models and results (§4-5)
- Base: BST 2.7B (BlenderBot), encoder truncation 128 tokens, extended to 256/512/1024 by adding learned positions (§4.1).
- Memory variants: retrieval over past sessions (RAG, FiD, FiD-RAG) and SumMem, which first writes summaries and then retrieves from them (§4.2-4.3).
- Table 6 (validation perplexity, BST 2.7B-1024 with gold summaries): training on Session 1 only gives 10.5 on all sessions; Sessions 1+2 give 8.94; Sessions 1-4 give 8.77.
- Table 7 (test perplexity, session 5): BST 2.7B without MSC fine-tuning 10.50; MSC 2.7B (truncate 1024) 9.16; SumMem-MSC 2.7B (FiD-RAG) 9.07.
- Table 4: gold summaries (9.04 at session 2) beat raw dialogue history (9.18) and no history (9.46); predicted summaries give 9.11.

## Limits
- Sessions are short (up to 14 utterances each) and total context is about 1.6K tokens, so MSC does not test long-context inputs in the sense of ch-32c.
- Time gaps are simulated; the paper evaluates perplexity and crowdworker ratings, not instruction following or factual recall questions.
