---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/nathan-lambert-entropy-rl.md
source_url: https://www.interconnects.ai/
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; downgraded to a reliability note)"
---

# Excerpt: status of the Interconnects entropy synthesis as a source

The library card [[nathan-lambert-entropy-rl]] is a synthesis of several blog posts rather than an extract of
one artifact: its URL field points at the publication's front page, it names no post with a date, and it
carries no Verification section. Two quantitative claims that earlier versions of ch-43 took from it are not
traceable to a located primary text:

1. "stop and inspect when entropy crashes below ~0.2 nats on the last-token distribution";
2. "per-token entropy below 0.1 nats is collapse".

Neither threshold appears in [[entropy-mechanism-llm-rl]] (checked against arXiv:2505.22617v1, where the
0.1-nat threshold was removed from the card on 2026-09-14), and no other verified source in this library
states an entropy value at which a run should be stopped.

Per the authoring standard's evidence rules, an anecdotal source cannot be the only support for a
quantitative claim, so ch-43 cites this page only as an example of practitioner framing and uses no number
from it. The entropy levels that are reported with a setting attached are in
[[entropy-mechanism-llm-rl]] (relative, "more than 10× higher than the baseline plateau", §4.3),
[[high-entropy-minority-tokens]] (the 0.672-nat 80th percentile of a Qwen3-8B token sample), and
[[deepswe]] (the statement that an entropy loss is unnecessary when base-model token entropy is within
0.3-1).
