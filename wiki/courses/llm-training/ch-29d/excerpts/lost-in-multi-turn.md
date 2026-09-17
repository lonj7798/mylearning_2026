---
chapter: ch-29d
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/**/lost-in-multi-turn.md on 2026-09-15)
source_url: https://arxiv.org/abs/2505.06120
source_version: arXiv:2505.06120v1 (2025-05-09)
created_at: "2026-09-15"
---

# Excerpt: LLMs Get Lost In Multi-Turn Conversation (Laban, Hayashi, Zhou, Neville; Microsoft Research, Salesforce Research)

Facts used by [[read]] §1 and §2. Read in the arXiv v1 PDF on 2026-09-15.

## Sharded simulation (§3)
- A fully specified single-turn instruction is split into shards; shard 1 states the high-level intent and later shards add one element each. The user simulator reveals at most one shard per turn (§3.1-3.2).
- The user simulator, the response-strategy classifier, and the answer extractor are prompt-based GPT-4o-mini (§3.2). The assistant is not told that the conversation is multi-turn or underspecified (§3.2).
- Conversation types: FULL (original instruction, one turn), CONCAT (all shards in one turn), SHARDED (one shard per turn), RECAP (SHARDED plus a final turn restating all shards), SNOWBALL (each turn restates all earlier shards) (§3.3).

## Scale and metrics (§4.2, §5)
- 600 instructions over six tasks (Code, Database, Actions from BFCL, Math from GSM8K, Data-to-text, Summary); 15 LLMs; N = 10 simulations per model, instruction, and type; more than 200,000 conversations; temperature 1.0; cost about $5,000.
- Per instruction with scores S: P = mean(S); aptitude A = 90th percentile of S; unreliability U = 90th percentile − 10th percentile.

## Results
- "every model sees its performance degrade on every task when comparing FULL and SHARDED performance, with an average degradation of -39%" (§6.1). CONCAT averages 95.1% of FULL (§6.1). The Figure 1 caption states the multi-turn drop as −35%; the text uses −39%.
- Aptitude drops on average 16% (§6.2; Figure 1 labels −15%) and unreliability rises on average 112% (§6.2).
- Gradual sharding: with 31 instructions and GPT-4o / GPT-4o-mini, the degradation appears "with two-shard instructions and beyond" (§6.3).
- SNOWBALL "can mitigate the FULL-to-SHARDED performance deterioration by 15-20%" (§7.1).
- Temperature (Table 3): with user and assistant temperature both 0.0, SHARDED unreliability is 30.5 (GPT-4o-mini) and 29.7 (GPT-4o); "lowering the temperature [...] is ineffective in improving system reliability" in multi-turn settings (§7.2).

## Simulator validity (App. D, Table 5)
- Inspection of SHARDED conversations on Actions, Code, Math, Database: shard fully revealed 96.0%; shard contextualized 98.4%; strategy classification accuracy 95.2%; extraction success 97.0%; overall success 97.8%.
- The count of inspected conversations is inconsistent in the source: §3.2 says "several hundred", the App. D text says 200, and the Table 5 caption says 100.
- §3.2 summary: simulator errors occurred in less than 5% of inspected conversations and "disfavored the assistant model in less than 2%".

## Limitations stated (§2, §9)
- "the conversations we simulate are not representative of human-AI conversations"; the simulation guarantees that each turn reveals a new shard and that the last turn completes the information. The authors "believe the degradation observed in experiments is most likely an underestimate" of real-world unreliability (§9).
- Tasks are analytical, text-only, and English (§9).
