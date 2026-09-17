---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lost-in-multi-turn.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2505.06120
primary_version: arXiv:2505.06120v1 (2025-05)
created_at: "2026-09-15"
---

# Excerpt: LLMs Get Lost In Multi-Turn Conversation

Authors: Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville (Microsoft Research; Salesforce Research). Checked against the v1 PDF on 2026-09-15. Used by ch-25 `read.md` §9 and the long-context section.

## Main result (Abstract)
> "Our experiments confirm that all the top open- and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn, with an average drop of 39% across six generation tasks. Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability."

> "We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely."

## Setup (§3.1–3.2, §4.1, §5)
- Six tasks: code, database, actions, math, data-to-text, summary (§4.1). A fully specified instruction is split into shards; the first shard gives the high-level intent and each later shard adds one requirement (§3.1).
- A user simulator reveals at most one shard per turn; the assistant replies freely; a classifier labels reply strategy and an extractor pulls candidate answers for scoring (§3.2).
- Simulation types: FULL (original instruction, one turn), CONCAT (all shards as one turn), SHARDED (one shard per turn); RECAP and SNOWBALL are added in §7.1.
- Scale: 600 instructions, 15 LLMs, N = 10 simulations per model and type, default temperature T = 1 (§5).

## Metrics (§4.2)
> "P = Σ S_i / N;  A90 = percentile90(S);  U90_10 = percentile90(S) − percentile10(S)."

`S_i` is the 0–100 score of simulation i. P is averaged performance, A aptitude (best-case), U unreliability (interpercentile range). Metrics are computed per instruction and averaged over the corpus. The interpolation method for percentiles is not stated.

## Results
- > "every model sees its performance degrade on every task when comparing FULL and SHARDED performance, with an average degradation of -39%." (§6.1)
- > "models perform roughly equivalently in the CONCAT setting, with CONCAT performance averaging 95.1% of the FULL counterpart." (§6.1)
- > "Model aptitude degrades in a non-significant way between the full and sharded settings, with an average drop of 16%. On the other hand, unreliability skyrockets with an average increase of 112% (more than doubling)." (§6.2)
- RECAP and SNOWBALL (GPT-4o, GPT-4o-mini; four tasks): > "SNOWBALL gives a sense of realistic performance gains achievable through user-turn repetition: it can mitigate the FULL-to-SHARDED performance deterioration by 15-20%." (§7.1, Table 2)
- Temperature: > "Even when both the user and assistant temperatures are set to 0.0, there remains a large unreliability of around 30%." (§7.2, Table 3)
- Episodic tasks: the sharded translation task showed no degradation; the authors list generative, sufficiently complex, non-decomposable tasks as the properties that lead to getting lost (§7.3).

## Stated limits
- The user is an LLM simulator (GPT-4o-mini), not a human.
- Strategy classification is about 95% accurate (appendix, error analysis).
- No training experiment is reported; the paper does not test whether SFT on sharded conversations reduces unreliability.
