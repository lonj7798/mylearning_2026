---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: MiniMax-AI, "Aligning to What? Rethinking Agent Generalization in MiniMax M2", 2025-10-30 (no library card at the time of writing; chapter-local verified extract)
source_url: https://huggingface.co/blog/MiniMax-AI/aligning-to-what
created_at: "2026-09-15"
---

# Excerpt: Aligning to What? Rethinking Agent Generalization in MiniMax M2

- **Authors:** MiniMax-AI (post by a member of the M2 agent-alignment team; acknowledges Junxian He and Wenhu Chen)
- **Year:** 2025 (published 2025-10-30)
- **Source type:** official blog (no benchmark tables; qualitative post-mortem)
- **Used in:** [[read]] §5, §7, Generalization lens, Common mistakes

## Two alignment targets
- "the same model can feel brilliant in one framework and useless in another. An agent might crush a tool-use leaderboard but fail spectacularly at a simple, real-world task."
- Objective 1: excel on open-source benchmarks, which "are essential for measuring 'pure' capabilities". Objective 2: "Generalize Robustly to the Real World ... across unfamiliar tools, IDEs/CLIs, agent scaffolding, and user setups."

## Interleaved thinking
- "an agent's internal monologue—its 'thinking'—can and should happen at any point during a task, not just once at the beginning like a standard reasoning model."
- Two stated reasons: maintaining focus over long contexts, and adapting to "constant, unpredictable perturbations from the outside world (i.e., tool outputs)".
- Deployment note: "Because M2 relies on Interleaved Thinking, its context is its memory. For best performance, you must retain the full session history, including the thinking steps."

## The negative result on tool scaling
- Initial hypothesis: "tool scaling is agent generalization." Starting from a Python interpreter, a search engine and a browser, the plan was to scale the number and variety of tools.
- Outcome: "At first, this worked. Our benchmark scores climbed to respectable levels. But as we dug deeper, we realized we were solving the wrong problem. The model aced the tests, but if we changed the environment even slightly—like swapping to a different scaffolding framework—its performance would plummet."
- Revised position: "Agent generalization is not just about adapting to new tools; it's about adapting to perturbations across the model's entire operational space", enumerated as the tool info and toolset, the system prompt, the user prompt, the environment (files, codebases, APIs), and the tool responses at each step. "Our old 'tool scaling' approach only addressed the first item."
- The team then "built a comprehensive data pipeline designed for full-trajectory generalization. The data it generates trains the model to be stable against perturbations at every step." Reported result: internal tests with "cold-start" scaffolding they had "barely considered" exceeded expectations. No numbers are given.

## Companion source on interleaved-thinking state (MiniMax news post, 2025-11-03)
- "Retaining prior-round thinking state improves performance significantly compared to discarding it, as evident across benchmarks: SWE-Bench Verified 69.4 vs. 67.2 (Δ=+2.2; +3.3%), Tau^2 87 vs. 64 (Δ=+23; +35.9%), BrowseComp 44.0 vs. 31.4 (Δ=+12.6; +40.1%), GAIA 75.7 vs. 67.9 (Δ=+7.8; +11.5%), and xBench 72.0 vs. 66.0 (Δ=+6.0; +9.1%)."
- Locus: https://www.minimax.io/news/interleaved-thinking-unlocks-reliable-minimax-m2-agentic-capability (fetched 2026-09-15). The post states the cause of the failure mode is API-level: the OpenAI Chat Completion API "does not support passing reasoning content back in subsequent requests".

## Verification
- Checked on 2026-09-15 against https://huggingface.co/blog/MiniMax-AI/aligning-to-what and the linked MiniMax news post of 2025-11-03.
- Source type note: this is official but qualitative. Under the course evidence rules it supports a mechanism claim and a stated lab decision, not a quantitative claim, except for the five benchmark pairs quoted above, which come from the companion news post.
- Not reported: the size or composition of the full-trajectory generalization pipeline; which perturbation types were ablated; any controlled comparison against tool scaling alone.
