---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/openassistant.md
source_url: https://arxiv.org/abs/2304.07327
primary_version: arXiv:2304.07327v2 (2023-10-31)
created_at: "2026-04-23"
revised_at: "2026-09-15 (reduced to the values in the verified library card of 2026-09-14)"
---

# Excerpt: OpenAssistant Conversations, values used in ch-15

Source: [[openassistant]] library card, verified on 2026-09-14 against arXiv:2304.07327v2. This excerpt lists only the values ch-15 uses; the card holds the full extract.

- Over 13,500 volunteers; 161,443 messages (91,829 prompter, 69,614 assistant) in 66,497 trees; 10,968 complete trees; 461,292 quality ratings; 35 languages (Abstract, §4).
- 8,576 of the 161,443 messages are synthetic (§4).
- Language shares: English 42.8%, Spanish 31.4%, Russian 5.7%, German 3.6%, other languages smaller (Fig. 2 left).
- Rankings from several annotators are merged with a variant of Tideman's ranked-pairs method (§4, App. B).
- Ranking guidelines place replies that admit not knowing below correct replies and above incorrect ones (App. A §6).
- Spam-flagged and moderator-deleted messages are not exported; mean Detoxify toxicity was 4.625% for 3,422 deleted messages vs 0.988% for 71,359 retained messages (§6.2, Table 2).
- Annotator survey: 89.1% identify as male, median age 26 (§7).
- Reward models were trained on rankings of human-written replies, not samples from the SFT model; the LLaMA-30B RLHF model improved LMEH (68.03 → 68.51) and Vicuna Elo (979 → 1068) over SFT but not OpenAI Evals (0.52 → 0.51) or HumanEval (0.20 → 0.15); the authors hypothesize the data choice explains part of the gap (Table 1, §7).
