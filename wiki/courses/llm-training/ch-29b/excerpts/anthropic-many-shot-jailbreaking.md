---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/anthropic-many-shot-jailbreaking.md (library card not present on 2026-09-15; content taken from the official post)
source_url: https://www.anthropic.com/research/many-shot-jailbreaking
primary_version: Anthropic research post, 2024-04-02 (read 2026-09-15); the full paper (Anil et al., NeurIPS 2024) was not read
created_at: "2026-09-15"
---

# Excerpt: Many-shot jailbreaking (Anthropic research post)

Source type: official blog summarizing a paper. Numbers below are the ones printed in the post.

- Attack: "include a faux dialogue between a human and an AI assistant within a single prompt", in which the assistant "readily answer[s] potentially harmful queries", followed by the target query. With one or a handful of faux dialogues the safety-trained refusal is still triggered; "including a very large number of faux dialogues ... we tested up to 256" produces harmful answers.
- Scaling: as the number of shots increases "beyond a certain number", the percentage of harmful responses increases (demonstration model Claude 2.0). The effectiveness follows "the same kind of power law" as benign in-context learning with more demonstrations.
- Model size: many-shot jailbreaking "is often more effective—that is, it takes a shorter prompt to produce a harmful response—for larger models."
- Mitigation by fine-tuning: fine-tuning the model to refuse queries that look like many-shot attacks "merely delayed the jailbreak: ... it did take more faux dialogues in the prompt before the model reliably produced a harmful response, the harmful outputs eventually appeared."
- Mitigation by prompt classification and modification before the model: "in one case dropping the attack success rate from 61% to 2%."
- Combining many-shot jailbreaking with other published jailbreak techniques reduces the prompt length required.
