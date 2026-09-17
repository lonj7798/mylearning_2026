<!-- chapter: ch-26
     track: synthetic
     kind: content
     title: Tool and Function-Calling Data
     deps: [ch-25]
     sources: [[toolformer]], [[toolllm]], [[apigen]], [[apigen-mt]], [[xlam]], [[toolace]], [[hammer]], [[bfcl]], [[bfcl-v4-format-sensitivity]], [[tau-bench]], [[tau2-bench]], [[toucan-mcp]], [[toolrl]], [[agenttuning]], [[reasoning-trap-tool-hallucination]], [[kimi-k2]], [[gorilla]], [[nexusraven]], [[granite-function-calling]]
     figures: figures/tool-pipeline.html
     revised: 2026-09 (generality revision)
-->

# Chapter 26 — Tool and Function-Calling Data

> **Core insight.** Function-calling data is built by pairing tool schemas (real, executable, or LLM-synthesized) with generated queries and calls, then filtering by format, execution, and an LLM judge. In APIGen, generator pass rates ranged from 34.42% to 84.15%, and adding the rejected samples back into training lowered BFCL overall by 5.94 points for a 7B student and 12.17 points for a 1B student ([[apigen]] Table 1, Fig. 5). Filtering does not by itself produce generalization to unseen tools: models trained on one data family score inconsistently across benchmarks, and a DeepSeek-Coder-7B model trained with Hammer's function-name masking and irrelevance data averaged 74.94 F1 on five benchmarks outside BFCL vs 67.65 for xLAM-7B-fc from the same base family ([[hammer]] Tables 1, 5). On the same 4K examples, GRPO with a rule-based call reward reached 58.38 BFCL V3 overall on Qwen2.5-7B-Instruct vs 36.53 for SFT on R1-distilled traces ([[toolrl]] Table 1; one run each).
>
> **Guideline.** When synthetic calls come from a generator whose pass rate is below the 84% of DeepSeek-V2-Chat in APIGen, run execution and semantic checks before SFT, because rejected data cost 4–12 BFCL points in the add-back ablation ([[apigen]] Fig. 5); otherwise keep the checks and expect a smaller, unmeasured effect. When the model must use tools it has not seen, evaluate on unseen-tool and unseen-category splits and on at least one benchmark built by a different group than the training data, and mask function and parameter names in part of the SFT data, because name reliance is the failure Hammer measured ([[hammer]] §3.2, Table 3). Include abstention targets (irrelevant tools, missing arguments) and choose their share by measuring both irrelevance accuracy and call accuracy, because removing non-tool-use dialogs lowered ToolACE irrelevance accuracy from 86.42 to 6.99 ([[toolace]] Table 7) while Hammer found the two metrics move in opposite directions with the best overall near a 10% share ([[hammer]] Fig. 6). When the target use is multi-turn or stateful, validate the action sequence before simulating the dialog and report pass^k for k > 1, because gpt-4o's pass^1 above 60% on τ-retail fell below 25% at pass^8 ([[apigen-mt]] §4.1; [[tau-bench]] Fig. 4). When reasoning RL or reasoning distillation follows, re-measure abstention with missing and distractor tools, because GRPO raised the no-tool hallucination rate from 34.8% to 90.2% while BFCL multi-turn rose ([[reasoning-trap-tool-hallucination]] Table 3).

## Corrections to the version you studied

1. Guideline "removing [execution] costs ~11 BFCL points" and the table "Full 88.24 / remove semantic 82.2 / remove execution 77.3 / remove format 70.1" → APIGen has no per-stage removal table and no format arm. Its Fig. 5 adds rejected data back: −4.06 (+semantic failures) and −5.94 (+execution failures) for xLAM-7B (FC), −9.59 and −12.17 for xLAM-1B (FC) ([[apigen]], Fig. 5, §5.2).
2. "xLAM-7B (Mistral base) reaches 88.24% BFCL-V1, #1 among <13B" → xLAM-7B (FC) is fine-tuned from DeepSeek-Coder-7B-instruct-v1.5 and ranks 6th with 85.65 on the 2024-06-15 leaderboard; in the xLAM paper's BFCL v2 table (2024-09-03), xLAM-7b-fc-r is 80.18, rank 17 ([[apigen]] §5.1, Table 2; [[xlam]] Table 1, Table 5).
3. "~40% rejection" for APIGen, and the per-stage pass rates 75% / 90% / 88% in the figure → pass rates depend on the generator: 34.42% (DeepSeek-Coder-33B-Inst) to 84.15% (DeepSeek-V2-Chat); per-stage percentages are not reported ([[apigen]] Table 1).
4. "Teacher: DeepSeek-Coder-V2-Instruct primary, GPT-4 ablation", "GPT-4 judge", "DeepSeek-Coder-V2 vs GPT-4 differ by ~2%" → generators are DeepSeek-V2-Chat, DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, Mixtral-8x7B-Inst; the semantic-checker model is not named; no GPT-4 comparison exists ([[apigen]] §3.2, §4.2).
5. "Python sandbox, 5-second timeout", "MinHash on (query, call)", "simple ~40% / multiple ~25% / parallel ~20% / parallel-multiple ~15%", "each API appears ~16×" → no timeout value, no deduplication method, no type proportions, and no per-API count are reported ([[apigen]] Verification).
6. "3,673 executable APIs (… where Salesforce either ran the endpoint or wrote a Python mock)" → 3,539 ToolBench REST APIs that passed accessibility tests plus 134 Python functions ([[apigen]] §4.1).
7. "The xLAM-2 staged recipe — APIGen-60k SFT → APIGen-MT-5k SFT → optional DPO (β=0.1)" and the §8 stage learning rates, epochs, and sequence lengths → xLAM-2-fc-r models are trained in one SFT stage on a mixture of APIGen-MT trajectories, APIGen data, and other agentic data for at most 3 epochs; no DPO stage, learning rate, batch, or sequence length is reported ([[apigen-mt]] §5.1).
8. "Of ~10M positions scanned, ~2% survive" and "Table 2 keeps only a few thousand annotations per tool out of hundreds of thousands of candidates" → candidate counts are not reported; at τ_f = 1.0 Table 2 keeps 18,526 (QA), 60,974 (Wikipedia search), 994 (calculator), 20,587 (calendar), 1,034 (translation) examples ([[toolformer]] Table 2).
9. DFS-DT pseudocode that retracts only when `obs.is_error`, and "without the retraction, DFS-DT degrades to ReACT" as an appendix claim → DFSDT abandons a node when the model calls `give_up_and_restart` and diversifies later children; the appendix says that when the model does not retract an action, DFSDT behaves like ReACT and costs the same ([[toolllm]] App. A.4 and prompts).
10. DFSDT Table 3 "ReACT 42.2 / 30.0 / 21.7, ReACT@N 47.7 / 34.3 / 26.0, DFS-DT 57.3 / 48.2 / 43.2" → ReACT 37.8 / 40.6 / 27.6, ReACT@N 49.4 / 49.4 / 34.6, DFSDT 58.0 / 70.6 / 62.8 on I1 / I2 / I3 ([[toolllm]] Table 3).
11. "ToolACE … keeping 'execute where possible'", "Rule-based (schema + param + type + execution where mock exists)" → ToolACE's rule layer checks calls without running them (name in the tool list, required parameters, regular-expression formats), and tool responses are simulated by an LLM agent ([[toolace]] §2.2.1, §2.3).
12. "From a 3K seed of real APIs, an LLM mutates via three operators" → API synthesis builds an API context tree from API-related pretraining documents, then adapts and evolves API definitions; no seed count is stated; 26,507 APIs and 390 domains are correct ([[toolace]] §2.1, Table 1).
13. "complexity evaluator classifies each dialog into 5 difficulty levels … 30/25/20/15/10% … hand-tuned to match BFCL" → complexity is the loss of the model being trained on the sample, with bounds from a prior set; the user agent is asked for harder or easier queries; ToolACE generates four dialog types (single, parallel, dependent, non-tool-use) ([[toolace]] §2.2, Eq. 1).
14. "GPT-4 judge, 3-way verdict … ~40% rejection, final 11,300 dialogs" → the model layer uses separate LLM agents for hallucination detection, consistency validation, and tool-response checks; the rejection rate and a final dialog count are not reported ([[toolace]] §2.3).
15. "ToolACE-8B (Llama-3.1-8B base): 91.41% BFCL-V1, beating xLAM-7B's 88.24%" → ToolACE-8B is LLaMA-3.1-8B-Instruct with LoRA; on the BFCL-v3 leaderboard of 09/20/2024 it has 59.22 overall (rank 3) vs 51.45 for xLAM-7b-fc-r; 91.41 is the overall score on non-live categories in the ablation figures ([[toolace]] §3.1, Table 2, Fig. 3).
16. "Ablation: −4.3% removing TSS, −3.1% complexity controller, −5.2% model-judge, −2.8% rule-checks" → Fig. 3 (non-live overall): no verification 89.59, rules only 90.47, both 91.41; complexity subsets 90.47 / 90.71 / 89.65 (Fig. 4); diversity subsets 88.18 / 88.35 / 88.41 (Fig. 5); there is no API-synthesis removal arm ([[toolace]] §3.3).
17. BFCL "seven categories: Simple · Multiple · Parallel · Parallel-Multiple · Relevance-Detection · Multi-Turn · Multi-Step" → single-turn categories are Simple, Multiple, Parallel, Parallel Multiple, Irrelevance, Relevance; multi-turn has Base, Missing Parameters, Missing Functions, Long Context; agentic has Web Search, Memory, SQL; there is no Multi-Step category ([[bfcl]] §3, App. C.2).
18. AST matcher "kwargs sorted …, literals canonicalised (1.0 ≡ 1, "red" ≡ 'red')" → each parameter value must be in a set of possible answers; Python accepts an int for a float but not the reverse, and Java and JavaScript require float literals; strings are compared case-insensitively with whitespace and listed punctuation removed ([[bfcl]] §4.1, App. H).
19. Version table "V2 Live +1,500 real user queries" and "V4 Agentic … SWE-Gym agent data" → the crowd-sourced category has 2,251 entries; BFCL V4 adds web search, memory, and format sensitivity ([[bfcl]] §1; [[bfcl-v4-format-sensitivity]]).
20. "pass^k. V3+ requires success on all k independent trials" → pass^k is defined in τ-bench; BFCL multi-turn scoring requires state and response checks to pass in every turn of one run ([[tau-bench]] §3; [[bfcl]] §4.4).
21. "xLAM-2-70B τ-bench: pass^1 = 56.2%, pass^4 = 39.4%" → 56.2 is the average pass@1 over retail and airline; pass^k values are plotted in Fig. 6 and 39.4 does not appear in the text or tables ([[apigen-mt]] Table 2, Fig. 6).
22. "Reported ~70% task-collection success rate, ~12-turn average" and "~12 messages" → Phase 1 blueprint success is 70% with agentic feedback vs 28% without; Phase 2 trajectory success is 67%; trajectories average 7 tool calls and 6 user turns (1 to 29 turns); the text separately states gpt-4o takes 12 turns per task on average ([[apigen-mt]] Fig. 4, §4.3).
23. "semantic review (committee aggregation + refinement)" → committee majority vote is Stage 2 (alignment validation); Stage 3 applies a score threshold and returns feedback for regeneration ([[apigen-mt]] §4.1.2).
24. "a ~$1/sample rollout" → no cost is reported ([[apigen-mt]] Verification).
25. "xLAM-2-8B reaches 69.25% … on only 5K trajectories" → 69.25 is the BFCL v3 multi-turn score, but training mixed APIGen-MT trajectories with APIGen data and other agentic data in unreported proportions ([[apigen-mt]] Table 1, §5.1).
26. "Even frontier models still call a tool on ~10% of irrelevant queries" → no source; BFCL irrelevance accuracy varies by model and mode, for example GPT-4-turbo-2024-04-09 83.8 in FC mode and 35.6 in prompting mode ([[bfcl]] Table 1).
27. Hammer "func_[a-z0-9]{6} … in 30% of training data", "~30% irrelevance samples", "30% is the optimum; 50% destroys recall", "+13 points on BFCL relevance; Hammer-7B ~90%, matching GPT-4" → Hammer masks function names, parameter names, and default values; the released script masks each of three copies with probability 2/3 using 5–15-character random names; the irrelevance set is 7,500 examples (about 11% of 67,500); the masking-ratio ablation states no optimum; the irrelevance-share optimum is about 10%; Hammer-7B BFCL irrelevance is 72.87 vs 79.76 for xLAM-7B-fc ([[hammer]] §4, §5.5–5.6, Table 2; repository `train/data_processing.py`).
28. NexusRaven "simple (60%) / parallel (20%) / nested (20%)" and "removing nested → −15 points" → no training-mix or ablation is published for NexusRaven-V2 ([[nexusraven]] Verification).
29. Gorilla "80% noisy retrieval + 20% oracle" and "Hallucination rate: 11% retriever-aware vs 40% GPT-4 without" → no such mix is described; 10.95% is Gorilla zero-shot (no retriever) on HuggingFace vs 37.16% for GPT-4 zero-shot ([[gorilla]] Table 1).
30. Granite "seven-capability taxonomy (Nested / Parallel / Multiple / Multi-turn / Relevance / Sequencing / Slot-Filling)", the source-mix percentages, and per-source removal deltas → Granite-20B-FunctionCalling's seven training tasks are Nested Function Calling, Function Chaining, Parallel Functions, Function Name Detection, Parameter-Value Pair Detection, Next-Best Function, and Response Generation; the mix percentages and deltas are not in the paper ([[granite-function-calling]], arXiv:2407.00121 Abstract, Table 1).
31. "Training on Glaive's `<functioncall>` XML instead of OpenAI JSON costs 5–10 points" and "A model at 92% overall BFCL but 65% relevance hallucinates tools … 35% of the time in the wild — the #1 production failure mode" → neither number has a source; BFCL V4 reports that return-format effects differ by model and are larger for smaller models ([[bfcl-v4-format-sensitivity]] §4.1).
32. The §8 "drop-in recipe" mixture (25% APIGen / 20% ToolBench / …), "Expected acceptance ~55–60%", and "MinHash threshold 0.9" → constructed for the old chapter, not taken from any source; the Recipe section below lists verified values only.
33. "[[ch-28]] … Multi-turn tool use is one of the stronger long-context signals" → no source in this chapter measures that, and ch-28 does not cover tool data.
34. "Whoever sets the eval taxonomy sets the data-generation taxonomy" as a finding, and "ToolACE's distribution is hand-tuned to match BFCL" → APIGen states that its four query styles are taken from BFCL categories ([[apigen]] §3.3); ToolACE calibrates difficulty to the learner, not to BFCL proportions ([[toolace]] §2.2.2). The broader claim is an interpretation without a measurement.

Section map from the old version: §1 → §2; §2 → §3; §3 → §4; §4 → §7; §5 → §5; §6 → §8; §7 Hammer → §6 (NexusRaven, Gorilla, Granite paragraphs removed; see corrections 28–30); §8 → Recipe. §1, §9, and the Negative samples section are new.

## Why this chapter matters for a general-purpose model

A function call is a structured output that names one tool from a provided list and fills its arguments. A general-purpose model receives tools it was not trained on, described in formats it may not have seen, and must also decide when no tool applies. Tool-call data enters the pipeline at SFT (this chapter), in agentic mid-training (ch-32d), and as the action space of multi-turn agentic RL (ch-45b). Three measurable risks follow. First, a model can learn features of one data family, such as its naming style or output template, instead of reading tool descriptions; xLAM-7B-fc scored 57.5 on Nexus Raven, the lowest of the three models in Hammer's Table 1, while scoring highest of the three on BFCL (79.41) and API-Bank (72.45) ([[hammer]] Table 1). Second, training that raises call accuracy can lower abstention: GRPO with reasoning on Qwen2.5-7B-Instruct raised BFCL multi-turn accuracy from 13.6 to 23.5 while the no-tool-available hallucination rate rose from 34.8% to 90.2% ([[reasoning-trap-tool-hallucination]] Table 3). Third, benchmark scores can reflect the test distribution: training domains in APIGen-MT are the τ-bench evaluation domains ([[apigen-mt]] §4.3), and BFCL's perplexity check flagged xLAM-7B ([[bfcl]] §5.5). This chapter covers how tool-call data is produced and filtered, and how to measure whether the result transfers.

## §1 Terms

- **Tool schema (function definition)**: a name, a natural-language description, and typed parameters, usually as JSON Schema. **Candidate set F**: the schemas given in one prompt ([[bfcl]] §3.1).
- **Relevance / irrelevance**: a relevance item expects at least one call; an irrelevance item expects no call, or a clarification when a needed function or argument is missing ([[bfcl]] App. B).
- **Trajectory**: an interleaved sequence of user turns, model reasoning, calls, tool responses, and final answers.
- **FC mode vs prompting mode**: tools passed through an API's tools field vs described in the system prompt with a required text format ([[bfcl]] §5.1).
- **Held-out tool splits**: unseen instructions for seen tools (Inst.), unseen tools from a seen category (Tool), and unseen tools from an unseen category (Cat.) ([[toolllm]] §3.2).
- **MCP (Model Context Protocol)**: a standard interface through which an application discovers and invokes tools hosted on servers ([[toucan-mcp]] §1).
- **pass^k**: the probability that all k independent trials of a task succeed, averaged over tasks ([[tau-bench]] §3); defined with a worked example in §8.

## §2 Toolformer: annotation from plain text with a loss filter

**Definition.** Toolformer inserts API calls into unlabeled text and keeps a call only if its result makes the following tokens easier to predict ([[toolformer]] §2).

**Problem.** Without labels, there is no signal for where a call helps. A call that parses and runs may still add nothing to the text.

**Mechanism.**
1. Prompt GPT-J (6.7B) with a few demonstrations per tool; keep positions where `p_M(<API>)` exceeds τ_s = 0.05, at most k = 5 per text, and sample up to m = 5 calls per position (App. A).
2. Execute each call against one of five tools (QA, Wikipedia search, calculator, translation, calendar).
3. Compare weighted losses on the next tokens and keep calls that pass the threshold.
4. Fine-tune on the original text with accepted calls inserted.

**Formula** (§2, §4.1):

```
L_i(z) = − Σ_{j=i..n} w_{j−i} · log p_M(x_j | z, x_{1:j−1})
keep call c_i  if  min( L_i(ε), L_i(e(c_i, ε)) ) − L_i(e(c_i, r_i)) ≥ τ_f
w_t = w̃_t / Σ_s w̃_s,   w̃_t = max(0, 1 − 0.2·t)
```

`x_j` are the text tokens after position i; `z` is the prefix given to the model; `e(c, r)` is the call text followed by result r; `ε` is the empty sequence; `τ_f` is the filtering threshold (1.0 by default, 0.5 for calculator and translation). Comparing against `L_i(e(c_i, ε))`, the call without its result, stops a call from being kept because the call text alone is predictive.

**Worked example.** The weights for t = 0…4 are 1, 0.8, 0.6, 0.4, 0.2 before normalization, so after dividing by 3.0 they are 1/3, 4/15, 1/5, 2/15, 1/15. Suppose the five next-token losses are (3.0, 2.0, 2.0, 1.0, 1.0) with no call, (2.8, 2.0, 2.0, 1.0, 1.0) with the call text only, and (0.5, 0.5, 1.0, 1.0, 1.0) with call and result. The weighted losses are 2.133, 2.067, and 0.700. The minimum of the first two is 2.067, and 2.067 − 0.700 = 1.367 ≥ 1.0, so the call is kept; at τ_f = 2.0 it is dropped.

**Evidence.** Table 2 shows the threshold's effect on data size: QA examples fall from 51,987 at τ_f = 0.5 to 18,526 at 1.0 and 5,135 at 2.0. On ASDiv / SVAMP / MAWPS, Toolformer scores 40.4 / 29.4 / 44.0 vs 7.5 / 5.2 / 9.9 for GPT-J and 14.0 / 10.0 / 19.8 for GPT-3 175B (Table 4). Language-modeling perplexity is unchanged when calls are disabled: 10.3 / 10.5 on WikiText / CCNet for both GPT-J fine-tuned on the same text without calls and Toolformer (Table 8). **Result (single study).**

**Conditions and limits.** Five fixed text-to-text tools, single local calls, no multi-step use, and no evaluation on tools absent from training. The threshold was chosen per tool to keep enough examples, not by a downstream ablation (§4.1).

**Implication for a general-purpose model.** Toolformer is an early example of measuring retention (perplexity with tools disabled) next to the targeted gain. That pairing is the minimum report for any tool-data run.

## §3 ToolLLM: real APIs, search-based trajectories, and unseen-tool splits

**Definition.** ToolBench pairs 16,464 RapidAPI REST APIs in 49 categories with instructions and solution paths generated by `gpt-3.5-turbo-16k` ([[toolllm]] §2). The released data has 126,486 instruction–path pairs with 469,585 real API calls (Table 1).

**Problem.** Linear ReACT rollouts fail on multi-tool instructions, and only passed annotations become training data, so annotation yield sets data cost (§3.1).

**Mechanism.**
1. Sample one tool (I1), several tools from one category (I2), or several from one collection (I3), and have the teacher write instructions.
2. Search for a solution with DFSDT: a depth-first tree over thought–action steps; the model can end a branch with `give_up_and_restart`, and later children see the previous children in a "diversity" prompt (App. A.4 and prompts).
3. Execute calls against the real APIs; compress responses over 1,024 tokens (App. A.2).
4. Keep paths judged as passing by ToolEval, a ChatGPT evaluator with 87.1% pass-rate agreement with human annotators (§3.1).

**Evidence (yield).** At matched API budget, pass rates on I1 / I2 / I3 are 37.8 / 40.6 / 27.6 for ReACT, 49.4 / 49.4 / 34.6 for ReACT@N, and 58.0 / 70.6 / 62.8 for DFSDT (Table 3). The gap is largest on I3.

**Evidence (held-out tools).** ToolLLaMA-2-7B with DFSDT passes 57.0 (I1-Inst), 61.0 (I1-Tool), 62.0 (I1-Cat), 77.0 (I2-Inst), 77.0 (I2-Cat), 66.0 (I3-Inst) (Table 4). Unseen-category tools did not score lower than unseen instructions for seen tools in this split. On APIBench, which ToolLLaMA was not trained on, it reaches HuggingFace / TorchHub / TensorHub AST accuracy of 16.77 / 51.16 / 40.59 with its retriever vs 10.51 / 44.62 / 34.31 for Gorilla-ZS + BM25 and 15.71 / 50.00 / 41.90 for Gorilla-RS + BM25, the Gorilla variant trained with retrieval (Table 5). ToolLLaMA is above Gorilla-RS on two of the three hubs and below it on TensorHub. **Result (single study).**

**Conditions and limits.** Test instructions receive oracle APIs unless the retriever is used; pass rates come from an LLM judge; BFCL's authors describe ToolBench as depending on RapidAPI responses with high variance ([[bfcl]] §2).

**Implication.** The Inst / Tool / Cat split is a direct test of the question in this chapter's title. A tool-data paper that reports only in-distribution scores leaves that question unanswered.

## §4 APIGen: three-stage verification and what its ablation shows

**Definition.** APIGen generates single-turn query–call pairs from sampled APIs and accepts a pair only if it passes format, execution, and semantic checks in sequence ([[apigen]] §3).

**Problem.** Generated calls use functions or arguments that do not exist, fail at runtime, or run but do not answer the query. The measurable question is how much such data lowers a student's score.

**Mechanism.**
1. Sample APIs from 3,673 (3,539 REST APIs + 134 Python functions), seed examples, and a prompt template that can include ambiguous or misspelled requests (§3.3, §4.1).
2. Format checker: JSON with "query" and "answer"; only functions and arguments present in the given APIs (§3.2).
3. Execution checker: run Python functions in a subprocess or call REST APIs; remove type errors, invalid parameters, runtime errors, timeouts, missing arguments (§3.2).
4. Semantic checker: a second LLM reads functions, query, calls, and execution results and returns pass yes/no (App. B.2).
5. Add verified samples to the seed pool (§3.1).

**Worked example.** Each generator was asked for 40,000 samples (§4.2). For DeepSeek-Coder-33B-Inst, 13,769 were verified, 4,311 failed format, 15,496 failed execution, and 6,424 failed the semantic check (Table 1). The pass rate is 13,769 / 40,000 = 34.42%, and execution accounts for 15,496 / 26,231 = 59.1% of failures (derived). For DeepSeek-V2-Chat, 33,659 passed (84.15%) and execution accounts for 3,359 / 6,341 = 53.0% of failures (derived). The Mixtral-8x7B-Inst row sums to 39,000 rather than 40,000, while its printed pass rate 38.46% equals 15,385 / 40,000 (source inconsistency). Panel A of [figures/tool-pipeline.html](figures/tool-pipeline.html) shows these counts for all four generators and the add-back deltas.

**Evidence.** xLAM-7B (FC), fine-tuned from DeepSeek-Coder-7B-instruct-v1.5 on the data, ranks 6th on BFCL (2024-06-15) with 85.65 overall; xLAM-1B (FC) ranks 24th with 74.41, above GPT-3.5-Turbo-0125 (FC) at 63.88 (Table 2). Adding stage-3 failures back lowers BFCL overall by 4.06 (7B) and 9.59 (1B); adding stage-2 failures back lowers it by 5.94 and 12.17 (Fig. 5). **Result (single study)**, one run per arm.

**Conditions and limits.** BFCL is the only evaluation; no held-out-API split and no non-function-calling benchmark are reported (§5). The paper does not say whether the execution add-back arm also contains the semantic failures. The xLAM paper later trains xLAM-7b-fc-r with 50% synthetic function-calling data and 50% other tasks ([[xlam]] §3.5), and BFCL's perplexity check found xLAM-7B's perplexity rising from 3.67 on curated data to 5.09 on crowd-sourced queries, which the BFCL authors read as tuning to the static test distribution ([[bfcl]] §5.5, Interpretation).

**Implication.** Rejected samples have negative marginal value, and the smaller student is hurt more. The ablation does not show which stage matters most, and it does not test transfer to other benchmarks.

## §5 ToolACE: synthesized API diversity and learner-calibrated difficulty

**Definition.** ToolACE synthesizes API definitions, generates dialogs with three LLM agents (user, assistant, tool), adjusts query difficulty using the loss of the model being trained, and filters with a rule layer and a model layer ([[toolace]] §2).

**Problem.** Executable APIs limit coverage, and data that is too easy or too hard for a given learner adds little. The paper tests API diversity, complexity, and verification separately.

**Mechanism.**
1. Build an API context tree from API-related pretraining documents; sample subtrees and evolve API definitions with diversity indicators, reaching 26,507 APIs in 390 domains (§2.1, Table 1).
2. Generate single, parallel, dependent, and non-tool-use dialogs; sample each assistant decision several times and keep consistent ones; simulate tool responses with the tool agent (§2.2.1).
3. Score complexity with the learner's loss and tell the user agent to write harder or easier queries (§2.2.2–2.2.3).
4. Rule layer: name in tool list, required parameters, regular-expression formats, without running the call. Model layer: hallucinated argument values, consistency with the query, tool-response agreement with the API definition (§2.3).

**Formula** (Eq. 1):

```
H_M(x, y) = −(1/n_y) · Σ_{i=1..n_y} log p(t_i | x, t_1, …, t_{i−1})
```

`M` is the model to be trained; `x` the input query; `y = [t_1 … t_{n_y}]` the response tokens; `p` the model's next-token probability. A higher value means the sample is harder for M. The lower bound is the loss of samples M already answers correctly; the upper bound is the loss of samples that stay high after fine-tuning.

**Worked example.** A three-token response with probabilities 0.5, 0.25, 0.125 has losses 0.693, 1.386, 2.079 nats, so H = 4.159 / 3 = 1.386. If the prior set gives bounds of 0.3 (lower) and 1.2 (upper), this sample is above the upper bound and the user agent is told to simplify the query.

**Evidence.** On the BFCL-v3 leaderboard (09/20/2024), ToolACE-8B has 59.22 overall (rank 3), relevance 85.37, irrelevance 83.81 (Table 2). On non-live categories: no verification 89.59, rule layer only 90.47, both layers 91.41 overall (Fig. 3); complexity subsets easy / medium / hard 90.47 / 90.71 / 89.65 (Fig. 4); API-diversity subsets low / medium / high 88.18 / 88.35 / 88.41, with irrelevance 87.92 / 88.75 / 88.75 (Fig. 5). At 25,000 samples each, training on ToolLLM, xLAM, or ToolACE data gives 24.90, 40.51, and 58.19 overall and 4.41, 11.87, and 86.42 irrelevance (Table 6). Using the learner as its own complexity evaluator gives 59.22 vs 57.61 with Qwen1.5-7B-Chat (Table 8). **Result (single study).**

**Conditions and limits.** Overall-score differences in the non-live ablations are at most 1.82 points, from single LoRA runs without variance. Tool responses are simulated, so no execution error is ever observed. General capabilities are shown only in a radar chart with the text "negligible performance degradation on some benchmarks" (§3.6, Fig. 8). The complexity signal is tied to one learner, so the same data may be mis-calibrated for another model (Interpretation; App. H lists evaluation cost and possible sampling bias).

**Implication.** The irrelevance differences of 79.43 points (86.42 vs 6.99 when non-tool-use dialogs are removed, Table 7) come from data type, while the verification, complexity, and diversity ablations move non-live overall by at most 1.82 points; see the Negative samples section.

## §6 Hammer: removing the tool-name shortcut

**Definition.** Hammer fine-tunes small models on xlam-function-calling-60k plus 7,500 irrelevance examples, with function names, parameter names, and default values replaced by random strings in a fraction of examples ([[hammer]] §4).

**Problem.** Function-calling models score inconsistently across benchmarks. xLAM-7B-fc averages 69.05 over BFCL, API-Bank, Seal-Tools, Tool-Alpaca, and Nexus Raven, with 57.5 on Nexus Raven, while Granite-20B-FunctionCalling averages 74.19 (Table 1). Replacing names in the Seal-Tools test set with random strings lowers xLAM-1B-fc's F1 more than Hammer-1.5B's (Fig. 2). The authors attribute the inconsistency to reliance on names instead of descriptions (§3.2, Interpretation).

**Mechanism.**
1. Irrelevance set: sample 7,500 training examples, delete the correct function from the candidate list, and set the label to an empty list (§4.2).
2. Masking: replace function and parameter names with random strings, move randomized default values into parameter descriptions, and rewrite the label with the same mapping (§4.1).
3. Released script: copy the 67,500 examples three times, shuffle, and mask each with probability 2/3; function names become 5–15 random characters and parameter names 4–10 (`train/data_processing.py`).

**Worked example.** 7,500 / 67,500 = 11.1% of examples are irrelevance examples. Three copies give 202,500 training examples, of which 2/3 × 202,500 = 135,000 are expected to be masked (derived). An unmasked example teaches that `get_exchange_rate` matches a query about currency; a masked example keeps the description "Return the exchange rate between two currencies" but renames the function to a string such as `q7Vd_.x0K`, so only the description links query and tool. Panel D of [figures/tool-pipeline.html](figures/tool-pipeline.html) shows a schema and label before and after masking.

**Evidence.** Averaged F1 (function name + arguments) over API-Bank L-1 and L-2, Tool-Alpaca, Seal-Tools, and Nexus Raven: Hammer-7B 76.21, Granite-20B-FC 72.56, xLAM-7B-fc 67.65, Qwen2-7B-Instruct 59.84, GPT-4-0613 78.79 (Table 3). Applying the same recipe to DeepSeek-Coder-7B gives 74.94 (Table 5). On BFCL (09/20/2024), Hammer-7B is 83.92 overall with irrelevance 72.87; xLAM-7B-fc is 79.41 with irrelevance 79.76 (Table 2). In the masking-ratio ablation (Qwen2-1.5B, Seal-Tools training, one epoch), a larger ratio slows learning on Seal-Tools and improves API-Bank (§5.5, Fig. 5). In the irrelevance-share ablation (Qwen2-1.5B-Instruct, 10,000 samples), irrelevance accuracy rises and call accuracy falls as the share grows, with the best overall near 10% (§5.6, Fig. 6). **Result (single study).**

**Conditions and limits.** Figure values are not printed. Masking and irrelevance data are not ablated separately on the five-benchmark average. The authors converted formats and removed some erroneous samples from the non-BFCL benchmarks (repository README), so their numbers are not directly comparable with other papers. The authors state the 10% share "may require adjustment depending on the underlying model and training dataset".

**Implication.** In the ratio ablation, a higher masking ratio slowed learning on the training benchmark (Seal-Tools) and raised accuracy on a different benchmark (API-Bank); the figure does not print the size of either effect. It targets one shortcut (names); it does not address description style or output format (§8).

## §7 Multi-turn and agentic tool data

**APIGen-MT: validate the plan, then simulate the conversation.** Phase 1 samples τ-bench APIs, policies, domain data, and personas, and generates an instruction `q`, ground-truth actions `a_gt`, and expected outputs `o_gt` ([[apigen-mt]] §4.1). Actions are executed and checked against domain policies written as Python unit tests; a committee of LLM judges votes on alignment; tasks under a score threshold receive feedback and are regenerated (§4.1.2). Phase 2 runs a simulated user guided by `q` and a persona, who "is unaware of the underlying environment and available APIs", against a gpt-4o agent, and keeps trajectories whose final state matches `a_gt` and whose outputs match `o_gt` (§3.2.2, §4.2). Agentic feedback raises Phase 1 success from 28% to 70%; Phase 2 success is 67% (Fig. 4). xLAM-2-70b-fc-r reaches 56.2 average τ-bench pass@1 vs 52.9 for gpt-4o-2024-11-20, and xLAM-2-8b-fc-r 46.7 vs 38.2 for Llama 3.1 70B Instruct (Table 2).

Why validate first: if a trajectory has 12 turns and each turn is correct with independent probability 0.95, the whole trajectory is correct with probability 0.95^12 = 0.54 (illustration, not from the source). Validating `a_gt` before dialog generation turns the final check into a comparison against a known state instead of a judgment of a free-form conversation (Interpretation). **Limits.** The generation domains, APIs, and policies are the τ-bench evaluation domains, and no overlap check against τ-bench test tasks is reported (§4.3, §5.1). Only BFCL v3 and τ-bench are evaluated.

**τ-bench and τ²-bench: users as part of the environment.** τ-bench scores a task 1 only if the final database equals the goal database and the agent's messages contain required information ([[tau-bench]] §3). τ²-bench gives the simulated user tools in a shared telecom environment. Moving gpt-4.1 and o4-mini from a no-user mode, where the agent controls all tools, to dual control lowers pass^1 by 18% and 25% ([[tau2-bench]] §4.2, Fig. 4). Data synthesized with passive users (APIGen-MT, retail and airline) does not contain this coordination skill (Interpretation).

**Toucan: real MCP servers at scale.** Toucan crawls about 2,800 MCP servers, keeps 495 reachable ones, synthesizes tasks with five LLMs, filters tasks with Kimi-K2 ratings, generates trajectories against the real servers with three teacher models, and filters by rules and an LLM judge; extensions add irrelevance tasks (zero tool calls kept), persona variants, and multi-turn follow-ups ([[toucan-mcp]] §3). SFT on a 119.3K subset raises BFCL V3 overall for Qwen2.5-14B-Instruct from 57.69 to 65.09 and τ-bench retail from 44.46 to 48.48 (Table 2, Fig. 9). The extension ablation at 14B shows narrowing and recovery: the single-turn core alone lowers τ-bench retail from 44.46 to 36.95 while raising BFCL to 60.16; adding irrelevance data gives 41.63, diversification 43.70, and multi-turn 48.48 (Fig. 9). At 7B, non-live AST falls from 84.19 to 78.52 and τ²-bench telecom from 16.70 to 10.50 (Tables 2–3). No contamination check is reported. **Result (single study).**

**Kimi K2: MCP and synthetic tools in a frontier SFT pipeline.** Kimi K2's tool-use synthesis starts from 3000+ real MCP tools fetched from GitHub and over 20,000 synthetic tools from hierarchical domain evolution; agents and rubric-based tasks are generated, trajectories are simulated with LLM users and a stateful tool simulator, and an LLM judge keeps trajectories that meet the rubric; real execution sandboxes are used for coding tasks ([[kimi-k2]] §3.1.1). Kimi-K2-Instruct reports Tau2 Avg@4 of 70.6 retail, 56.5 airline, 65.8 telecom, and states as a limitation that performance may decline when tool use is enabled unnecessarily ([[kimi-k2]] Table 3, §5). **Source type:** official technical report.

**Implication.** Multi-turn tool data needs state-based verification, user behavior that is not scripted by the plan, and evaluation on domains other than the generation domains.

## §8 Measuring tool use beyond one benchmark

**BFCL categories and matching.** Single-turn items are Simple, Multiple, Parallel, Parallel Multiple, Irrelevance, and Relevance; crowd-sourced items come from real user queries; multi-turn items add Missing Parameters, Missing Functions, and Long Context; agentic items add web search, memory, and SQL ([[bfcl]] §3). AST matching requires the exact function name and each argument value in a set of allowed answers (App. H). Scores differ by mode: GPT-4-turbo-2024-04-09 has irrelevance 83.8 in FC mode and 35.6 in prompting mode (Table 1). Stateful categories are harder than single-turn ones: gpt-4o-2024-11-20 (Prompt) scores 95.5 on AST multiple, 59.0 on multi-turn base, and 6.0 on memory (Table 1).

**pass^k.** For a task run n times with c successes ([[tau-bench]] §3):

```
pass^k = E_task[ C(c, k) / C(n, k) ]          (all k sampled trials succeed)
pass@k = 1 − E_task[ C(n − c, k) / C(n, k) ]  (at least one succeeds)
```

`C(a, b)` is the binomial coefficient, 0 when b > a; the expectation is the mean over tasks.

**Worked example.** Two tasks, n = 4. Task A has c = 3: pass^1 = 3/4 = 0.75, pass^2 = C(3,2)/C(4,2) = 3/6 = 0.50, pass^3 = 1/4 = 0.25, pass^4 = 0. Task B has c = 4: pass^k = 1 for all k. Averages: pass^1 = 0.875, pass^2 = 0.75, pass^3 = 0.625, pass^4 = 0.50. pass@2 is 1.0 for both tasks. A model can therefore look reliable by pass@k and unreliable by pass^k on the same trials. Panel B of [figures/tool-pipeline.html](figures/tool-pipeline.html) computes both curves for any n and success counts. Published reference: gpt-4o function calling on τ-retail has pass^1 above 60% and pass^8 below 25% (Fig. 4).

**Format sensitivity.** BFCL V4 varies return format (Python, JSON, two XML styles), function-document format (Python, XML, JSON), a tool-call tag, prompt format, and prompt style, 26 variations over 200 single-turn entries and 39 models ([[bfcl-v4-format-sensitivity]] §2–§3). Accuracy is generally higher with Python or JSON output than XML, particularly for smaller models; tool-use-specialized models show the largest failures, such as watt-tool-70B returning Python-style calls when JSON is required and CoALM-70B scoring near zero when a tool-call tag is required (§4.1, §4.5). The blog's text and its Figure 4 caption give different orders for Python and XML documentation (§4.2). xLAM trains with 15 output formats "to avoid the model overfitting on JSON format" and reports that GPT-4o's ToolQuery success drops by 42% when a unified structured format is required ([[xlam]] §3.2, §5.2.1). **Source type:** BFCL V4 is an official blog by the benchmark maintainers; the xLAM result is a single study.

**Contamination.** Sources that report a check: BFCL removed Nexus leaderboard test queries from its crowd-sourced set (App. E.2) and uses perplexity on crowd-sourced vs curated data as a signal (§5.5); NexusRaven-V2 keeps one benchmark task unreleased ([[nexusraven]] Findings). Sources that do not: APIGen (query styles taken from BFCL categories, §3.3), APIGen-MT (generation and evaluation in the same τ-bench domains), ToolACE, Toucan. **Open question:** how much of the reported gains in the latter group survive a decontaminated or freshly collected tool set.

## §9 SFT versus RL for tool use, and side effects on general behavior

**ToolRL reward.** ToolRL trains with GRPO on 4K examples (2K ToolACE, 1K masked Hammer, 1K xLAM) using a rule-based reward ([[toolrl]] §3.3, §4.1):

```
R_final   = R_format + R_correct,          R_format ∈ {0, 1}
R_correct = 6 · R_max / S_max − 3 ∈ [−3, 3]
r_match   = r_name + r_param + r_value,    S_max = 1 + |G| + Σ_j |keys(G_j)|
```

`G` is the set of ground-truth calls; `r_name` is the Jaccard overlap of predicted and true tool-name sets; `r_param` sums, over true calls, the Jaccard overlap of argument-key sets; `r_value` counts argument values that match exactly; `R_max` is the total match score under the best matching of predicted to true calls.

**Worked example.** The true call is `get_weather(city="Paris", unit="C")`; the prediction is `get_weather(city="Paris", unit="F")` in the correct output format. r_name = 1, r_param = 2/2 = 1, r_value = 1 (only city matches), so r_match = 3 and S_max = 1 + 1 + 2 = 4. R_correct = 6 · 3/4 − 3 = 1.5 and R_final = 2.5. For a GRPO group with rewards (2.5, 4.0, −3.0, 1.0), the mean is 1.125 and the standard deviation is 2.607, so advantages are (0.53, 1.10, −1.58, −0.05) (computed). Two of the four samples receive a negative gradient. Panel C of [figures/tool-pipeline.html](figures/tool-pipeline.html) computes the reward for other match patterns.

**Evidence.** BFCL V3 overall, one run each (Table 1): Qwen2.5-7B-Instruct raw 41.97, SFT on 4K R1-distilled examples 36.53, GRPO from the raw model 58.38; Qwen2.5-3B 33.04, 41.97, 52.98; Llama-3.2-3B 22.09, 44.16, 44.10. On Bamboogle, a multi-hop QA task with web search that was not trained on, Qwen2.5-7B scores 69.6 raw, 28.8 after SFT400, 30.4 after SFT4k, and 72.0 after GRPO (Table 3). GRPO started from an SFT checkpoint reaches higher training reward and lower benchmark scores than GRPO from the raw model (§4.3, Fig. 5); the authors hypothesize memorization from SFT (Interpretation). Adding a reward for longer thinking lowered Qwen2.5-1.5B BFCL overall from 46.20 to 33.23 and irrelevance from 56.44 to 4.52 (Table 5). **Result (single study).** Two rows of Table 1 (Qwen2.5-3B and 7B SFT400) print the same values in six of seven columns, including overall 34.08 and irrelevance 8.11, and differ only in non-live exec (61.50 vs 66.68); this suggests a table error, so this chapter does not use the SFT400 rows of Table 1 as evidence. See ch-38a for SFT versus RL generalization in general.

**Reasoning training raises tool hallucination.** On Qwen2.5-7B-Instruct, think-then-act GRPO on SynTool raised the no-tool-available hallucination rate from 34.8% to 90.2% and the distractor-tool rate from 54.7% to 100.0%, while BFCL multi-turn rose from 13.6 to 23.5 and IFEval moved from 62.4 to 59.8; GRPO on GSM8K alone also raised both rates ([[reasoning-trap-tool-hallucination]] Table 3, Fig. 3). R1-distilled Qwen2.5-7B has rates of 74.3 / 78.7 vs 34.8 / 54.7 for the instruct model (Table 1). The authors state that existing tool-calling and instruction-following benchmarks did not reveal the increase (§4.4.2).

**Mixing general data.** In AgentTuning, 7B training on agent trajectories alone gave held-out agent score 0.09 and general score 0.22, vs 0.67 and 0.63 when the agent data was sampled at η = 0.2 with ShareGPT ([[agenttuning]] Table 5). General xLAM models keep general instruction data at 20% to 30% of the training set ([[xlam]] §3.5), without a reported ablation.

**Implication.** Tool-call gains measured on a tool benchmark do not show that abstention, instruction following, or non-tool tasks were preserved. Each of the sources above found a regression on one of those only when it measured it.

## Negative samples and negative feedback

**Four meanings used here** (course standard): (1) negative marginal value, a sample that lowers performance when used as a positive target; (2) negative as content, a failure or "no call" case placed in the target and trained with cross-entropy; (3) negative as conditioning, a failure trained under a control token; (4) negative as gradient, an explicit decrease of a sample's likelihood. Only (4) removes probability mass from the sample.

**1. Where negatives come from in tool data.** Calls rejected by format, execution, or semantic checks ([[apigen]] Table 1); trajectories that miss the goal state ([[apigen-mt]] §4.2; [[agenttuning]] §2.1.3); queries for which no given tool applies or an argument is missing ([[hammer]] §4.2; [[toolace]] §2.2.1; [[toucan-mcp]] Ext.1; [[bfcl]] App. B); rollouts with low rule rewards ([[toolrl]] §3.3); rejected fabricated calls in DPO pairs ([[reasoning-trap-tool-hallucination]] §6.1).

**2. What current practice does.** Discard (type 1): APIGen, APIGen-MT, AgentTuning, and Toucan drop failed samples. Content (type 2): APIGen's 8,000 relevance examples with empty-call or refusal targets ([[apigen]] App. B.3), Hammer's 7,500 empty-list labels, ToolACE's non-tool-use dialogs, Toucan's 40K irrelevance trajectories with zero calls. Conditioning (type 3): no source in this chapter. Gradient (type 4): negative GRPO advantages in ToolRL; the rejected term of DPO in the Reasoning Trap mitigation. APIGen-MT names failed trajectories as possible "additional contrastive signal" but does not use them (§6).

**3. Mechanism.** For a softmax over next tokens with logits z, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. A gradient step that lowers `log p_y` by step η changes `z_y` by `−η(1 − p_y)` and every other `z_j` by `+η p_j`. The removed mass goes to the other tokens in proportion to their current probability, so the most likely alternative gains the most. Worked example: at the first token of a response, suppose the model assigns p = 0.70 to starting a call to a distractor tool (y), 0.25 to starting a text answer (a), and 0.05 to a clarification question (b). With η = 1, logits change by (−0.30, +0.25, +0.05); the new probabilities are 0.581, 0.360, 0.059 (computed). Pushing down the wrong call mainly raises the text answer, which is correct only if a text answer is the right behavior. If instead the pushed-down call had p = 0.05, the correct call 0.90, and another output 0.05, the logits change by (−0.95, +0.90, +0.05) and the probabilities become 0.008, 0.969, 0.023 (computed): most of the change goes to the output that was already most likely. A type (2) target does the reverse: it raises the chosen "no call" output directly. Full derivations are in ch-43a.

**4. Evidence with numbers.** Negative marginal value: adding back rejected samples lowers BFCL by 4.06–5.94 (7B) and 9.59–12.17 (1B) ([[apigen]] Fig. 5); unfiltered agent trajectories score 1.34 / 0.47 held-in / held-out vs 1.96 / 0.65 filtered ([[agenttuning]] Table 2). Content: removing ToolACE's non-tool-use dialogs lowers irrelevance accuracy from 86.42 to 6.99 and multi-turn from 16.50 to 1.75 ([[toolace]] Table 7); adding Toucan's irrelevance data raises BFCL from 60.16 to 64.74 and τ-bench retail from 36.95 to 41.63 ([[toucan-mcp]] Fig. 9); in Hammer's ablation, more irrelevance data lowers call accuracy (Fig. 6). Gradient: DPO with fabricated calls as rejected responses lowers no-tool hallucination from 90.2% to 55.8% and distractor hallucination from 100.0% to 71.4%, and lowers SynTool reward from 0.45 to 0.34 ([[reasoning-trap-tool-hallucination]] Table 4). **Size of effect.** No source here separates the share of a gain due to negatives from the share due to positives.

**5. Controls.** Keep a positive target in the same data (Hammer mixes irrelevance examples with 60K positive calls; the Reasoning Trap DPO data also contains pairs where the tool is present, the chosen response is the correct call, and the rejected response is a needless refusal, [[reasoning-trap-tool-hallucination]] App. C.2). Set the abstention share by measuring both sides ([[hammer]] Fig. 6). Keep RL on-policy from the raw model when SFT initialization overfits ([[toolrl]] §4.3). Do not add rewards that are not tied to call correctness, such as thinking length ([[toolrl]] Table 5).

**6. Diagnostics.** Report relevance and irrelevance accuracy separately, since they can diverge (Qwen2.5-72B-Instruct Prompt: relevance 100.0, irrelevance 72.8, [[bfcl]] Table 1). Measure no-tool and distractor-tool hallucination rates ([[reasoning-trap-tool-hallucination]] Eq. 1). Log chosen and rejected log probabilities in DPO, and GRPO statistics split by advantage sign. Report pass^k with k > 1 for multi-turn tasks.

**7. Effect on generality.** Abstention can transfer to settings outside the training data: ToolRL reports that its Qwen2.5-3B GRPO model scores highest on BFCL irrelevance subsets that the authors describe as not included in RL training ([[toolrl]] §4.3, Fig. 7). Over-abstention is the opposite risk: Kimi K2 reports declines when tools are enabled unnecessarily, and needless refusals are an explicit rejected type in the Reasoning Trap pairs. The DPO mitigation cost task reward (0.45 → 0.34), so negative-gradient training on tool calls needs a utility check next to the hallucination check.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | base checkpoint | DeepSeek-Coder-1.3B-instruct; DeepSeek-Coder-7B-instruct-v1.5 | arXiv:2406.18518v1 §5.1 | verified 2026-09-14 | no ablation reported |
| xlam-function-calling-60k | — | SFT (data) | generators; temperature; target per generator | Mixtral-8x22B-Inst + DeepSeek-V2-Chat (release); 0.7; 40,000 | v1 §4.2 | verified 2026-09-14 | Table 1 pass rates by generator |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | relevance examples | 8,000 (empty call or refusal) | v1 App. B.3 | verified 2026-09-14 | no ablation reported |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | LR; schedule; warmup; epochs; optimizer | 5e-6; cosine; 50 steps; 4; AdamW | v1 App. B.3 | verified 2026-09-14 | no ablation reported |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | cutoff length; per-device batch; grad accumulation; hardware | 2048; 6; 2; 8 × A100 40GB | v1 App. B.3 | verified 2026-09-14 | no ablation reported |
| xLAM-7b-fc-r; xLAM-1b-fc-r | 7B; 1.3B | SFT | data mixture | 50% synthetic function-calling, 50% other tasks | arXiv:2409.03215 §3.5 | verified 2026-09-15 | no ablation reported |
| xLAM general models (7b-r, 8x7b-r, 8x22b-r) | 7B–141B | SFT | general instruction share; output-format augmentation; warmup | 20%–30%; 15 output formats; 100 steps, cosine | arXiv:2409.03215 §3.2, §3.5, §4.1 | verified 2026-09-15 | ToolQuery-Unified comparison (Table 3), no format ablation |
| xLAM-2-fc-r series | 1B–70B | SFT | data; loss masking; epochs | APIGen-MT + APIGen + other agentic data (proportions not reported); assistant tokens only; at most 3 | arXiv:2504.03601v4 §5.1 | verified 2026-09-14 | no ablation reported |
| xLAM-2-fc-r series | 1B–70B | SFT | LR; batch; sequence length; DPO | not reported | checked body, App. A–B, model card | not reported | — |
| ToolACE-8B | 8B | SFT | base; method | LLaMA-3.1-8B-Instruct; LoRA rank 16, alpha 32, all modules | arXiv:2409.00920 (ICLR 2025) §3.1 | verified 2026-09-15 | no ablation reported |
| ToolACE-8B | 8B | SFT | LR; warmup ratio; schedule; batch; epochs | 1e-4; 0.1; cosine; 48; 3 | App. C.2 Table 5 | verified 2026-09-15 | no ablation reported |
| ToolACE-8B | 8B | SFT | total examples; rejection rate | not reported | checked body, App. A–H | not reported | — |
| Hammer (released training script) | 1.5B–7B | SFT | data | xlam-function-calling-60k + 7,500 irrelevance, ×3, masked with p = 2/3 | github.com/MadeAgents/Hammer `train/data_processing.py` (main, read 2026-09-14); arXiv:2410.04587 §4.2 | verified 2026-09-15 | irrelevance share: Fig. 6 (≈10% best on Qwen2-1.5B, 10K samples); masking ratio: Fig. 5 (no optimum stated) |
| Hammer (released training script) | 1.5B–7B | SFT | method; LR; schedule; warmup ratio; epochs | LoRA rank 32, all modules; 5e-5; cosine; 0.00833; 1 | `scripts/train.sh` (main, read 2026-09-14) | verified 2026-09-15 (released config; the paper does not state it produced the checkpoints) | no ablation reported |
| Hammer (released training script) | 1.5B–7B | SFT | cutoff; per-device batch; grad accumulation; weight decay | 2048; 4; 2; 0.01 | `scripts/train.sh` | verified 2026-09-15 | no ablation reported |
| ToolLLaMA-2-7B | 7B | SFT | context; LR; warmup ratio; batch; epochs; checkpoint rule | 8,192 via PI ratio 2; 5e-5; 0.04; 64; 2; best on dev set | arXiv:2307.16789 App. A.3 | verified 2026-09-15 | no ablation reported |
| Toolformer (GPT-J) | 6.7B | SFT (data) | τ_s; k; m; τ_f | 0.05; 5; 5; 1.0 (calculator, MT: 0.0; 20; 10; 0.5) | arXiv:2302.04761 App. A | verified 2026-09-15 | Table 2: example counts at τ_f 0.5/1.0/2.0; no accuracy ablation |
| Toolformer (GPT-J) | 6.7B | SFT | batch; LR; warmup | 128; 1e-5; linear, first 10% | §4.1 | verified 2026-09-15 | no ablation reported |
| ToolRL Qwen2.5-7B-Instruct (GRPO cold start) | 7B | RL | data; algorithm; KL | 4K (2K ToolACE, 1K Hammer masked, 1K xLAM); GRPO; no KL term | arXiv:2504.13958v1 §4.1–4.2, App. B | verified 2026-09-15 | Table 1: cold start 58.38 vs SFT4k 36.53 vs SFT400+GRPO 39.25 |
| ToolRL Qwen2.5-7B-Instruct (GRPO cold start) | 7B | RL | batch; rollouts per query; epochs; LR; temperature; max prompt / response | 512; 4; 15; 1e-6; 1.0; 2,048 / 1,024 | App. B Table 8 | verified 2026-09-15 | no ablation reported |
| ToolRL | 1.5B–7B | RL | reward | R_format ∈ {0,1} + R_correct ∈ [−3, 3] | §3.3 | verified 2026-09-15 | Table 5: adding a length reward lowers BFCL (1.5B 46.20 → 33.23) |
| Toucan-tuned Qwen2.5-7B/14B/32B-Instruct | 7B–32B | SFT | data | 119.3K = 28.3K core + 40K irrelevance + 15.8K diversify + 35.2K multi-turn | arXiv:2510.01179v1 §4.1 | verified 2026-09-15 | Fig. 9 extension ablation (14B) |
| Toucan-tuned Qwen2.5 models | 7B–32B | SFT | template; LR; epochs; effective batch; optimizer; max length | Hermes; 2e-5; 2; 64; AdamW (0.9, 0.999), ε 1e-8; 32,768 | App. C.2 Table 5 | verified 2026-09-15 | no ablation reported |
| AgentLM-7B / 13B / 70B | 7B–70B | SFT | agent data sampling ratio η | 0.2 (ShareGPT 0.8) | arXiv:2310.12823v2 §2.2.2 | verified 2026-09-14 | η scanned 0–1 on 7B held-out tasks; Table 5 endpoints |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT (data) | tool repository | 3000+ real MCP tools; over 20,000 synthetic tools | arXiv:2507.20534 §3.1.1 | verified 2026-09-14 | Figure 9 (t-SNE coverage); no ablation |

Rows dated 2026-09-15 were read in the primary source on that date because the library cards for [[xlam]], [[toolace]], and [[hammer]] have not been verified and no card existed for [[toolrl]] or [[toucan-mcp]]; the chapter excerpts hold the checked extracts.

**Starting point for a small general-purpose run.** For function-calling SFT of a 1B–7B instruction-tuned model, the verified APIGen settings are LR 5e-6 with cosine decay and 50 warmup steps, 4 epochs, AdamW, cutoff length 2048, per-device batch 6 with gradient accumulation 2, and 8,000 relevance examples next to the verified calls; they were used for DeepSeek-Coder-1.3B/7B on 8 × A100 40GB, with BFCL as the only evaluation. For cross-benchmark robustness, the Hammer release adds 7,500 irrelevance examples (about 11% of its data, near the ≈10% best share found on Qwen2-1.5B) and masks names in two thirds of copies. For general-purpose use, keep general instruction data in the mixture (xLAM general models: 20%–30%; AgentLM: η = 0.2 for agent data), because the single-skill runs above report no general benchmark.

## Generalization lens

**(a) What increases breadth.**
- Unseen-tool generalization with search-generated trajectories on real APIs: ToolLLaMA passes 62.0 on unseen-category tools vs 57.0 on unseen instructions for seen tools, and transfers to APIBench ([[toolllm]] Tables 4–5).
- Description reading instead of name matching: masking improves cross-task results (API-Bank) in the ratio ablation, and the Hammer recipe on DeepSeek-Coder-7B averages 74.94 vs 67.65 F1 across five benchmarks ([[hammer]] Fig. 5, Table 5).
- Data-type coverage: non-tool-use dialogs and parallel calls each carry a category that disappears without them ([[toolace]] Table 7); API diversity raises irrelevance accuracy from 87.92 to 88.75 ([[toolace]] Fig. 5).
- Multiple output formats in training ([[xlam]] §3.2), since tool-use-specialized models fail when the format changes ([[bfcl-v4-format-sensitivity]] §4.5).
- RL with a rule-based call reward from the raw model: +16.41 BFCL V3 over the raw Qwen2.5-7B and 72.0 vs 69.6 on untrained Bamboogle ([[toolrl]] Tables 1, 3).
- Real MCP environments with irrelevance and multi-turn extensions: +7.40 BFCL V3 and +4.02 τ-bench retail at 14B ([[toucan-mcp]] Table 2, Fig. 9).

**(b) What causes narrowing or forgetting.**
- Training on one data family: xLAM-7B-fc 57.5 on Nexus Raven vs 79.41 on BFCL ([[hammer]] Table 1); rising perplexity on crowd-sourced BFCL queries ([[bfcl]] §5.5).
- Single-turn tool SFT: τ-bench retail 44.46 → 36.95 at 14B ([[toucan-mcp]] Fig. 9); non-live AST 84.19 → 78.52 and τ²-telecom 16.70 → 10.50 at 7B ([[toucan-mcp]] Tables 2–3).
- SFT on R1-distilled tool traces: Bamboogle 69.6 → 28.8 (SFT400) and 30.4 (SFT4k), and BFCL V3 overall 41.97 → 36.53 (SFT4k), at 7B ([[toolrl]] Tables 1, 3).
- Reasoning RL or R1 distillation: tool hallucination up to 90.2% / 100.0% ([[reasoning-trap-tool-hallucination]] Tables 1, 3).
- Agent-only data without general data: held-out agent score 0.09 and general 0.22 at 7B ([[agenttuning]] Table 5).
- Too much abstention data: call accuracy falls as the irrelevance share grows ([[hammer]] Fig. 6).

**(c) How to measure it for this stage.**
- Held-out tool splits (Inst / Tool / Cat) and a benchmark from a different group than the training data ([[toolllm]] §3.2; [[hammer]] Table 3).
- BFCL with relevance and irrelevance reported separately, both FC and prompting modes, and live (crowd-sourced) vs non-live categories ([[bfcl]] Table 1).
- pass^k with k > 1 on τ-bench, and dual-control vs no-user modes on τ²-bench ([[tau-bench]] Fig. 4; [[tau2-bench]] Fig. 4).
- Format variations: return format, documentation format, tool-call tag ([[bfcl-v4-format-sensitivity]] §2).
- Abstention with missing and distractor tools after any reasoning training ([[reasoning-trap-tool-hallucination]] §3.1).
- General benchmarks before and after: LM perplexity with tools disabled ([[toolformer]] Table 8), IFEval ([[reasoning-trap-tool-hallucination]] Table 3), MMLU and MT-Bench ([[agenttuning]] Table 4).
- Contamination: overlap of synthetic tool pools and task domains with evaluation sets, and a perplexity comparison between curated and fresh queries ([[bfcl]] §5.5).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Training on unfiltered generations from a weak generator | BFCL drops, larger for a 1B student | per-stage failure counts; add-back ablation on a small run ([[apigen]] Table 1, Fig. 5) |
| Treating one benchmark as tool-use ability | high BFCL, low score on Nexus Raven or API-Bank | evaluate on at least two benchmarks from different groups ([[hammer]] Table 1) |
| No abstention targets | irrelevance accuracy near single digits | remove-type ablation or irrelevance score ([[toolace]] Table 7: 6.99) |
| Abstention share chosen without measuring calls | refusals on answerable queries; lower call accuracy | plot irrelevance and call accuracy against share ([[hammer]] Fig. 6) |
| One output template in all training data | failures when the deployment requires JSON, XML, or tags | run BFCL V4 format variations ([[bfcl-v4-format-sensitivity]] §4) |
| Reporting pass@1 only for multi-turn agents | inconsistent behavior across repeated sessions | pass^k curve over k ([[tau-bench]] §3) |
| Generating and evaluating in the same domains | large gains on the matching benchmark only | evaluate on other domains; report overlap checks ([[apigen-mt]] §4.3) |
| Simulated user that already knows the plan or never acts | agents fail with users who share control | τ²-bench dual-control vs no-user gap ([[tau2-bench]] §4.2) |
| SFT on distilled traces before RL | high training reward, low held-out score | compare SFT-init vs raw-init RL on held-out sets ([[toolrl]] §4.3) |
| Reasoning RL after tool SFT without an abstention check | rising hallucinated calls while BFCL rises | no-tool and distractor-tool rates ([[reasoning-trap-tool-hallucination]] Table 3) |
| Quoting leaderboard numbers across BFCL versions | the same model has different scores in different papers | state leaderboard date and version (xLAM-7B (FC): 85.65 in APIGen, 80.18 in xLAM v2 table, 51.45 in ToolACE's v3 table) |

## Check your understanding

1. In APIGen's add-back ablation, the 1B student loses more than the 7B student from the same rejected data. Give a causal explanation, and state what experiment would test it.
2. Toolformer keeps a call only if the loss with the result is lower than the minimum of the no-call and call-without-result losses. What kind of spurious call would be kept if the second term were dropped?
3. ToolACE's verification and complexity ablations change BFCL non-live overall by less than 2 points, but removing non-tool-use dialogs changes irrelevance by 79 points. Why does data type have a larger effect than filtering on this metric?
4. Explain how function-name masking can slow learning on the training benchmark and improve accuracy on a different benchmark, using what the model can and cannot use to select a tool.
5. Two agents have the same pass^1 on τ-retail. Under what distribution of per-task success rates would their pass^8 differ most?
6. In ToolRL, SFT on 4K R1-distilled examples lowered Bamboogle from 69.6 to 30.4 while GRPO on the same prompts raised it to 72.0. List the differences between the two runs that could cause this and which one the SFT-initialized GRPO run isolates.
7. Using `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, explain why pushing down a fabricated tool call can raise a direct text answer rather than a clarification question, and what data would change that outcome.

## Connections

- Previous: ch-25 — Multi-Turn Conversation Synthesis. User simulation and multi-turn structure from that chapter are the basis of §7.
- Next: ch-27 — Agentic Trajectory Data.
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix (APIGen is an instance of the verify step).
- ch-04 — Sequence Packing, Loss Masking, and Chat Templates (call formats and assistant-only loss).
- ch-29c — Agentic Environment and Task Synthesis at Scale; ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories (§7).
- ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (general-data share).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training.
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity (§9).
- ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-51a — Evaluating Agent Generality and Reliability (§8).

## Sources

- [[toolformer]] — sampling and loss-filter formula, thresholds, Table 2 counts, math results, perplexity retention.
- [[toolllm]] — ToolBench scale, DFSDT mechanism and yield, Inst/Tool/Cat splits, APIBench transfer, training settings.
- [[apigen]] — three-stage verification, Table 1 failure counts, Fig. 5 add-back ablation, xLAM (FC) settings.
- [[apigen-mt]] — blueprint validation, simulated user, trajectory statistics, τ-bench and BFCL v3 results, training mixture.
- [[xlam]] — base models, format augmentation, data mixture shares, BFCL v2 table, ToolQuery-Unified result.
- [[toolace]] — API synthesis, learner-loss complexity, dual-layer checks, BFCL-v3 results, data-type and source ablations.
- [[hammer]] — cross-benchmark inconsistency, masking and irrelevance methods, released scripts, ablations.
- [[bfcl]] — categories, AST matching rules, modes, stateful results, perplexity contamination check.
- [[bfcl-v4-format-sensitivity]] — format variations and model-specific failures.
- [[tau-bench]] — state-based reward and the pass^k definition.
- [[tau2-bench]] — dual-control telecom domain and mode ablation.
- [[toucan-mcp]] — MCP-server pipeline, extension ablation, regressions at 7B.
- [[toolrl]] — rule-based call reward, GRPO vs SFT results, length-reward ablation.
- [[agenttuning]] — agent/general data mixing and filtered vs unfiltered trajectories.
- [[reasoning-trap-tool-hallucination]] — tool hallucination after reasoning RL and distillation; DPO mitigation.
- [[kimi-k2]] — MCP and synthetic tool repository, Tau2 results, stated limitation.
- [[gorilla]] — named in a correction (hallucination numbers).
- [[nexusraven]] — named in a correction (unpublished mix) and the withheld benchmark task.
- [[granite-function-calling]] — named in a correction (seven training tasks).
