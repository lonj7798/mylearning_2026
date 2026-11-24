<!-- scope: DeepResearcher: GRPO on verl with live web search (Serper-style API) + a browsing/reading sub-agent; 50-node CPU tool cluster, retry + 7-day query cache; F1 reward with -1 format penalty; contamination-filtered 80k open-domain QA mix; observation masking
     deps: [[search-r1]], [[grpo]]
     see-also: [[asearcher]], [[zerosearch]], [[websailor-tongyi-deepresearch]], [[gemini-2.5-deep-research]], [[verl-rollout]], [[recipe-crosswalk]]
-->

# DeepResearcher: Scaling Deep Research via RL in Real-World Environments
- **Core Insight:** Training against the live web instead of a fixed Wikipedia index is feasible with outcome-only RL if the tool layer is engineered as its own service (a 50-node CPU cluster for search and crawling, retries, a 7-day search-result cache) (§3.2), and it matters most where the answer is outside the training corpus: on Bamboogle, whose questions need knowledge beyond Wikipedia, DeepResearcher outperforms local-RAG-trained agents, and R1-Searcher given web access still trails it (§5.2.4).
- **Guideline:** Before search RL, remove questions the base model already answers without tools (pass@10 contamination check) and time-sensitive or subjective questions, weight the mix toward multi-hop QA, mask tool observations from the loss, and treat tool-request concurrency (e.g., 4096 simultaneous searches per GRPO step) and provider rate limits as first-class infrastructure constraints (§3.2-3.3, §4.2-4.3).
- **Authors:** Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, Pengfei Liu (SJTU, SII, GAIR)
- **Year:** 2025 (arXiv v4 2025-04-17)
- **URL:** https://arxiv.org/abs/2504.03160 (code: https://github.com/GAIR-NLP/DeepResearcher)
- **Source type:** paper
- **Relevant topics:** live-web RL environment, tool server scaling, rate limits and caching, reading sub-agent, F1 reward with format penalty, contamination filtering, observation masking

## Summary
DeepResearcher trains Qwen2.5-7B-Instruct with GRPO on verl (§5.1.1). Each rollout interleaves `<think>` reasoning with JSON tool calls to `web_search` (top-k = 10 results with title, URL, snippet) and to a web-browsing agent, then ends with `<answer>` (§3.1). The browsing agent reads a URL segment by segment, keeps a per-query short-term memory, decides whether to continue to the next segment/URL, and returns only the newly added relevant information (§3.1; §3.2 Challenge III). The paper reports up to +28.9 points over prompt-engineered agents and up to +7.2 over RAG-based RL agents (abstract; §1).

## Key Contributions
- RL in a real web environment with engineering for concurrency, anti-crawling, rate limits (e.g., 200 requests/second), retries and a search cache (§3.2).
- Multi-agent design: the policy decides what to search/browse; a separate reading agent extracts relevant content from long pages (§3.1, §3.2).
- Search-dependent data curation: low-quality filtering with DeepSeek-R1 and pass@10 contamination detection against the base model (§4.3).
- Analysis of emergent behaviours: planning, cross-validation, reflection, honesty when the answer cannot be found (abstract; Fig. 2; §6).

## Key Figures/Tables to Study
- **Fig. 3:** batched parallel rollouts with per-rollout memory, browsing agent, decision steps.
- **Table 1 (in-domain) / Table 2 (OOD):** F1 and model-based evaluation (MBE, GPT-4o-mini judge).
- **Fig. 4:** F1 over training steps; tool calls and response length by hop count.

## Technical Details
**Pipeline view.**
| Component | DeepResearcher setting | Location |
|---|---|---|
| Prompt source | NQ, TriviaQA (single-hop), HotpotQA, 2Wiki (multi-hop); filtered to 80,000 examples at NQ:TQ:HotpotQA:2Wiki = 1:1:3:3 (75% multi-hop) | §4.1, §4.3 |
| Filtering | DeepSeek-R1 prompt removes time-sensitive, subjective, harmful questions; drop any question where 1 of 10 base-model samples contains the answer without search | §4.3 |
| Environment / tools | live web search API (top-10 results) + browsing agent over crawled pages (segmented) | §3.1-3.2 |
| Tool infra | 50-node CPU server cluster handles tool requests during rollout; retry on API/crawl failures; identical query within ~7 days served from cache | §3.2 |
| Rollout shape | up to 10 tool calls then a final answer step; 16 rollouts per prompt | §5.1.1 |
| Grader | −1 if format incorrect; otherwise word-level F1 vs reference answer | §3.4 |
| Loss masking | tool observations masked; only model responses contribute to training | §3.3 "Masking Observations" |
| Algorithm | GRPO with KL to reference (Eq. 1) | §3.3 |
| Batch | 256 prompts × 16 rollouts per step; mini-batch 4,096, i.e., one backward pass per rollout stage | §5.1.1 |
| Framework | verl | §5.1.1 |
| Stage position | single RL stage from an instruct model (no SFT stage reported) | §5.1.1 |

**Reward rationale.** F1 is chosen "due to our utilization of open-domain QA datasets with short-answer ground truth"; the authors note long-form answers would need other reward mechanisms (§3.4).

**Evaluation.** 512 sampled dev examples per dataset (NQ, TQ, HotpotQA, 2Wiki, MuSiQue, PopQA) and all 125 Bamboogle examples (§5.2.1). MBE uses GPT-4o-mini labelling correct/incorrect (§5.2.3).

**Results (Table 1, DeepResearcher row, F1 / MBE; read from text extraction of the PDF table).** NQ 39.6 / 61.9; TQ 78.4 / 85.0; HotpotQA 52.8 / 64.3; 2Wiki 59.7 / 66.6. Search-r1-base (local RAG) in the same table: NQ 45.4 / 60.0, HotpotQA 55.9 / 63.0, i.e., higher F1 but similar MBE on those two sets; the authors note Search-r1-base has direct access to the Wikipedia corpus the answers come from (§5.2.4).

**Training dynamics (§6.1, Fig. 4).** Eval F1 rises from 0.375 to about 0.55. Tool calls per question increase over training for all hop counts; the 4-hop setting is still increasing after step 34. Response length increases for all hop counts without saturating.

## Connections
- [[search-r1]]: local-RAG predecessor with the same masked-observation, outcome-reward design; baseline here.
- [[zerosearch]]: the opposite trade-off (simulated search to avoid API cost).
- [[asearcher]]: raises the tool-call limit from ~10 to 32-128 with asynchronous RL.
- [[recipe-crosswalk]]: search RL pipeline synthesis.

## Verification
- Created 2026-11-24 from arXiv:2504.03160v4 PDF: abstract, §1, §3.1-3.4, §4.1-4.3, §5.1-5.2.4, §6.1 read directly.
- Table 1 values were read from a column-scrambled text extraction and reassembled by column order; recheck against the PDF before quoting in a chapter. Table 2 (OOD) values not transcribed.
- No wall-clock, GPU count or number of training steps is stated in the sections read (Fig. 4 x-axis reaches at least step 34).
