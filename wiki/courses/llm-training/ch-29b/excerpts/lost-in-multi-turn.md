---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lost-in-multi-turn.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2505.06120
primary_version: arXiv:2505.06120v1 (2025-05)
created_at: "2026-09-15"
---

# Excerpt: LLMs Get Lost In Multi-Turn Conversation

Authors: Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville (Microsoft Research; Salesforce Research). Checked against the v1 PDF on 2026-09-15. Used by ch-29b `read.md` §4; the same facts are excerpted in ch-25.

## Sharded instructions (§3.1, §4.1)
- A fully specified instruction is split into shards: "the first shard (Shard 1) of a sharded instruction always introduces the high-level intent for the instruction, and subsequent shards each provide clarification to the instruction. Taken jointly, the set of shards reflects the same information provided in the fully-specified instruction" (§3.1). Appendix B defines five validity properties.
- Example (Figure 2, GSM8K): Shard 1 "How long before Jay's ready for the snowball fight?"; Shard 3 "He can make 20 snowballs per hour."; Shard 5 "The problem is that 2 melt every 15 minutes."
- Construction: "an LLM (GPT-4o) to propose and verify sharding candidates, which were then reviewed and edited (when necessary) by the authors"; 90-120 sharded instructions per task, "between 1-4 hours of manual inspection and annotation" per task (§4.1).

## Simulation and metrics (§3.2, §4.2, §5)
- A user simulator (GPT-4o-mini) reveals at most one shard per turn and rephrases it to fit the conversation; a strategy classifier and an answer extractor score each answer attempt.
- Settings: FULL (original, one turn), CONCAT (all shards in one turn; a verification baseline for information loss from rephrasing), SHARDED (one shard per turn).
- 600 instructions, 15 LLMs, N = 10 simulations per model, instruction, and setting, temperature 1.0 by default.
- `P = Σ S_i / N;  A90 = percentile90(S);  U90_10 = percentile90(S) − percentile10(S)`, with `S_i` the 0-100 score of simulation i. The percentile interpolation method is not stated.

## Results
- "every model sees its performance degrade on every task when comparing FULL and SHARDED performance, with an average degradation of -39%." CONCAT averages 95.1% of FULL (§6.1).
- "Model aptitude degrades in a non-significant way between the full and sharded settings, with an average drop of 16%. On the other hand, unreliability skyrockets with an average increase of 112%" (§6.2).
- SNOWBALL (repeat all earlier shards each turn) "can mitigate the FULL-to-SHARDED performance deterioration by 15-20%" (§7.1, Table 2).
- "Even when both the user and assistant temperatures are set to 0.0, there remains a large unreliability of around 30%" (§7.2, Table 3).

## Stated limits
- The user is simulated. No training experiment is reported: the paper does not test whether SFT on sharded conversations reduces unreliability.
