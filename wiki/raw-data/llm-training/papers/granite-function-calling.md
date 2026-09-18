<!-- scope: IBM's Granite-20B-FunctionCalling — multi-task instruction tuning on seven granular function-calling tasks
     deps: [[glaive-function-calling]]
     see-also: [[xlam]], [[toolace]], [[hammer]], [[bfcl]], [[nexusraven]]
-->

# Granite-Function Calling Model: Introducing Function Calling Abilities via Multi-task Learning of Granular Tasks
- **Core Insight:** Decomposing function calling into seven granular tasks and instruction-tuning one 20B model on all of them with task-specific instructions yields 84.71 overall accuracy on the Berkeley Function Calling Leaderboard, fourth overall and highest among openly licensed models as of 2024-06-25 (§5.3.1, Table 4).
- **Guideline:** When the available function-calling corpora are semantic-parsing and task-oriented-dialogue datasets rather than API traces, re-render the same data under several task-specific instructions (function-name detection, parameter-value detection, next-best function, chaining, nesting, parallel calls, response generation) instead of training on a single call-generation format, because the paper reports that this model transfers to seven out-of-domain evaluation datasets it never trained on (§5.3.2, Tables 5–7).
- **Authors:** Ibrahim Abdelaziz, Kinjal Basu, Mayank Agarwal, Sadhana Kumaravel, Matthew Stallone, Rameswar Panda, et al. (IBM Research)
- **Year:** 2024 (arXiv v1 2024-06-27)
- **URL:** https://arxiv.org/abs/2407.00121
- **Source type:** paper
- **Relevant topics:** function calling, multi-task instruction tuning, API-BLEND, BFCL, hallucination rate, QLoRA

## Abstract
The paper introduces GRANITE-20B-FUNCTIONCALLING, an Apache-2.0 model for function calling, obtained by instruction-tuning GRANITE-20B-CODE-INSTRUCT. Training uses a multi-task approach over seven tasks: Nested Function Calling, Function Chaining, Parallel Functions, Function Name Detection, Parameter-Value Pair Detection, Next-Best Function, and Response Generation. The training data comes from the API-BLEND corpora plus Glaive-V2; each dataset is reused under several task-specific instructions. The model is compared against more than 15 proprietary and open models on the Berkeley Function Calling Leaderboard and on seven out-of-domain academic benchmarks. It ranks fourth overall on BFCL and first among openly licensed models, and the paper argues that the task diversity is what produces its cross-dataset generality.

## Key Contributions
- A seven-task decomposition of function calling, split into high-level tasks (Nested Function Calling, Function Chaining, Parallel Functions) and low-level tasks (Next-Best Function, Function Name Detection, Parameter-Value Pair Detection), with Response Generation added as the seventh (§3, Table 1).
- A data-unification step that converts every API, tool, and function in the source datasets into a single JSON representation, and a fixed output format for emitted calls (§3.1).
- A programmatic, weighted mixture generator: dataset weights inside a task and task weights across the mixture are specified in a JSON configuration (§4.1).
- Release of GRANITE-20B-FUNCTIONCALLING under Apache 2.0 (§1, abstract).
- Evaluation on BFCL plus seven academic benchmarks, including a function-name hallucination-rate measurement (§5.3, Figure 3).

## Key Figures/Tables to Study
- **Table 1** — which of the seven tasks each of the six training datasets supplies.
- **Table 2** — the exact instruction string used for each task.
- **Table 4** — BFCL top-15 leaderboard with AST, executable, relevance, and overall columns.
- **Table 5** — function-name detection (F1, LCS, exact score) on ToolLLM G1/G2/G3 and RestGPT.
- **Table 6** — full function calling (F1 function name | F1 arguments) on API-Bank L-1/L-2, ToolBench HS/B, Tool-Alpaca, NexusRaven.
- **Table 7** — API-Bank response generation (BERTScore, ROUGE-L, BLEU).
- **Figure 3** — function-name detection score against hallucination rate, out-of-domain datasets.

## Technical Details
- Base model: GRANITE-20B-CODE-INSTRUCT; context length 8192 tokens (§4.2; §5.3.4).
- Training corpora: API-BLEND's five datasets — SeqSGD, SeqSNIPS, SeqTopV2, SeqATIS, SeqMultiWOZ — totalling about 160K training examples, plus Glaive-V2 (§3, §3.1).
- Mixture actually used: 142K examples (§4.2). The mixture is generated from a weighted configuration; the paper prints one example configuration but not the configuration used for the released model (§4.1).
- Tuning: QLoRA with rank 8, alpha 32, dropout 0.1; learning rate 5e-5; ApexFusedAdam; linear learning-rate scheduler; 3 epochs; a single node of 8× A100 80GB with 800GB RAM (§4.2).
- BFCL result (as of 2024-06-25, zero-shot): AST summary 84.11, executable summary 86.50, relevance 87.08, overall accuracy 84.71 — rank 4, tied on overall accuracy with Gorilla-OpenFunctions-v2, and the top openly licensed entry (Table 4). The three models above it are Claude-3.5-Sonnet-20240620 (90.00), GPT-4-0125-Preview (88.00), and Gemini-1.5-Pro-Preview-0514 (86.35).
- Function-name detection, ToolLLM G1/G2/G3 and RestGPT (Table 5): average F1 0.74, LCS 0.73, exact score 0.43 — best in the table; the paper states the margins over the next-best function-calling model as 8% F1, 7% LCS, 11% exact match (§5.3.2).
- Full function calling (Table 6): average F1 0.87 on function name (second to C4AI-Command-R-v01 at 0.88, a 35B model) and 0.59 on arguments (behind Command-R's 0.62) (§5.3.2).
- Hallucination: the paper reports GRANITE-20B-FUNCTIONCALLING has the highest function-name detection score at a hallucination rate below 0.1, where hallucination rate is the share of samples whose predicted function name is not in the provided function list (§5.2 metrics, §5.3.2, Figure 3).
- Response generation, API-Bank (Table 7): BERTScore 0.68 / ROUGE-L 0.47 / BLEU 0.47 at level 1 and 0.61 / 0.36 / 0.37 at level 2, second to Meta-Llama-3-70B-Instruct on all six numbers.
- Known limitation stated by the authors: because the 8192-token context must hold the whole function library, argument `type`, required-field, and optional-field entries were stripped from the function specifications in the prompt; each function keeps only a name, a description, and per-argument descriptions (§5.3.4).
- The authors also record reservations about parts of BFCL: the Java and JavaScript categories test language-specific syntax rather than function-calling ability, and the REST API category is brittle because of API availability and call limits (§5.3.1).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Granite-20B-FunctionCalling | 20B | SFT | base checkpoint | Granite-20B-Code-Instruct | arXiv:2407.00121v1 §4.2 | verified 2026-09-18 | no ablation reported |
| Granite-20B-FunctionCalling | 20B | SFT | mixture size (examples) | 142K | arXiv:2407.00121v1 §4.2 | verified 2026-09-18 | no ablation reported |
| Granite-20B-FunctionCalling | 20B | SFT | source pool size (examples, API-BLEND) | about 160K across 5 datasets, plus Glaive-V2 (size not given) | arXiv:2407.00121v1 §3 | verified 2026-09-18 | — |
| Granite-20B-FunctionCalling | 20B | SFT | adapter | QLoRA rank 8, alpha 32, dropout 0.1 | arXiv:2407.00121v1 §4.2 | verified 2026-09-18 | no ablation reported |
| Granite-20B-FunctionCalling | 20B | SFT | learning rate; optimizer; scheduler | 5e-5; ApexFusedAdam; linear | arXiv:2407.00121v1 §4.2 | verified 2026-09-18 | no ablation reported |
| Granite-20B-FunctionCalling | 20B | SFT | epochs | 3 | arXiv:2407.00121v1 §4.2 | verified 2026-09-18 | no ablation reported |
| Granite-20B-FunctionCalling | 20B | SFT | compute | 1 node × 8 A100 80GB, 800GB RAM (duration not given) | arXiv:2407.00121v1 §4.2 | verified 2026-09-18 | — |
| Granite-20B-FunctionCalling | 20B | SFT | max context length | 8192 tokens | arXiv:2407.00121v1 §5.3.4 | verified 2026-09-18 | — |
| Granite-20B-FunctionCalling | 20B | SFT | per-task and per-dataset mixture weights for the released run | not reported (checked §4.1, §4.2; only an illustrative configuration is printed) | arXiv:2407.00121v1 | not reported | — |
| Granite-20B-FunctionCalling | 20B | SFT | global batch size, sequence packing, loss masking | not reported (checked §4.2 and the whole body; no appendix) | arXiv:2407.00121v1 | not reported | — |

## Findings relevant to generality
- The evaluation is deliberately out-of-domain: none of the seven evaluation datasets were used for training, including ToolAlpaca and API-Bank, which do ship training splits (§5.3.2). The authors note they cannot guarantee the baseline models were held to the same rule.
- The paper attributes the cross-dataset stability to task diversity, contrasting it with Gorilla-OpenFunctions-v2, which ties Granite's BFCL overall accuracy but was fine-tuned on data similar to the BFCL test set and scores far lower on Tables 5–6 (§5.3.1).

## Findings relevant to negative feedback
- The paper measures hallucination (predicting a function name absent from the provided library) but does not train on explicit negative or irrelevance examples. Its "Next-Best Function" task is the closest signal, and it is trained as an ordinary target, not as a penalty. Compare with [[hammer]], which adds an irrelevance-labelled subset.

## Connections
- [[glaive-function-calling]] — Glaive-V2 supplies the Function Chaining and Response Generation portions of the mixture (Table 1).
- [[bfcl]] — the leaderboard used for the headline result (Table 4).
- [[hammer]] — a later small-model line that also targets relevance and irrelevance detection; its Table 2 reports Granite-20B-FunctionCalling at 76.63 overall on a later (2024-09-20) BFCL snapshot.
- [[nexusraven]] — the NexusRaven API evaluation is one of the seven out-of-domain benchmarks (Table 6).
- [[xlam]], [[toolace]] — contemporaneous function-calling data pipelines, not used by this paper.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2407.00121 (arXiv v1, 27 Jun 2024).
- Corrections to the previous card version:
  - Title "Granite Function-Calling: Granite-20B-FunctionCalling" → the published title is "Granite-Function Calling Model: Introducing Function Calling Abilities via Multi-task Learning of Granular Tasks".
  - The seven tasks were listed as "nested, parallel, multi-turn, slot-filling, relevance, sequencing, tool-selection" → the paper's seven are Nested Function Calling, Function Chaining, Parallel Functions, Function Name Detection, Parameter-Value Pair Detection, Next-Best Function, Response Generation (§3, Table 1).
  - The data mix was listed as APIGen/xLAM-FC-60k (~25%), ToolLLM/ToolBench (~20%), Glaive V2 (~15%), Nexus/NexusRaven (~10%), IBM in-house enterprise synthetic (~20%), general instruction data (~10%) → the actual sources are SeqSGD, SeqSNIPS, SeqTopV2, SeqATIS, SeqMultiWOZ (all from API-BLEND) and Glaive-V2 (§3, §3.1). ToolLLM, ToolBench and NexusRaven are **evaluation** sets the paper explicitly did not train on (§5.3.2); APIGen and an IBM enterprise corpus do not appear at all.
  - "~200K examples total after filtering and resampling" → the mixture is 142K examples (§4.2); the API-BLEND pool is about 160K (§3).
  - "Granite-20B-FC: BFCL-V1 ~82%" → overall accuracy 84.71 on the 2024-06-25 BFCL snapshot (Table 4). The paper does not use the label "BFCL-V1".
  - "Granite-20B-FC relevance-detection 85% on BFCL-V2 Live" → relevance is 87.08 in Table 4; there is no BFCL-V2 Live evaluation in this paper.
  - "on par with the best open model on BFCL and fourth overall" is the paper's own phrasing (§1); the earlier card's "competitive with GPT-4 and xLAM-8x7B" is not supported — xLAM does not appear in Table 4.
  - Author list: reordered and corrected to the paper's byline (Abdelaziz, Basu, Agarwal are the three equal-contribution first authors; Kapanipathi is a corresponding author listed last).
- Removed as unsupported by the source:
  - "Granite-3B-FC model release" and its "~75% BFCL-V1" score — only a 20B model is described.
  - "per-capability resampling ensures each of the 7 capabilities appears ≥ 10% of the mix", "API registry size: thousands (APIGen 3.6K + ToolLLM 16K + ...)", "enterprise samples pass an LLM-judge", "Teacher: GPT-4".
  - The capability ablation numbers ("removing the ToolLLM slice → multi-turn drops 12 points", "Nexus nested → 18 points", "relevance-data → 10 points") — no per-source ablation is reported anywhere in the paper.
  - "a naive equal-mix gives 5–10 points less than the tuned mix", "> 90% intent-classification accuracy on IBM customer API catalogs", "specifically tuned for the watsonx chat template", "downstream use in IBM watsonx AI agent platform", "code bias makes it weaker on conversational tool use".
  - "ongoing Granite updates through 2025" and the second URL https://huggingface.co/ibm-granite — the card must describe one artifact; the paper's own footnote points at that org page for the checkpoint, but the card's claims now all resolve to the arXiv paper.
- Not reported by the source: global batch size, packing, loss masking, training wall-clock, the exact mixture weights of the released run, and any multi-turn or agentic-trajectory evaluation.
