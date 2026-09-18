<!-- scope: controlled study (Oct 2025, Findings of EMNLP 2025) showing that LLM accuracy on GSM8K, MMLU, HumanEval, and a variable-summation task drops as input length grows even when the model recites the evidence with exact match, with whitespace filler, with filler masked from attention, and with evidence placed next to the question; plus a recite-then-solve mitigation
     deps: [[needle-in-haystack-data]], [[ruler]]
     see-also: [[chroma-context-rot]], [[lost-in-the-middle]], [[string-effective-context]], [[fiction-livebench]]
-->

# Context Length Alone Hurts LLM Performance Despite Perfect Retrieval
- **Core Insight:** With retrieval controlled by exact-match recitation, accuracy still falls by 13.9%-85% as input length grows within the models' claimed windows (Abstract); for Llama-3.1-8B-Instruct, which recites the evidence exactly for 970 of 1,000 MMLU problems at 30k tokens, MMLU accuracy drops 24.2 points (§1; App. Table 6), and drops of at least 7.9 points at 30K remain when all filler tokens are masked from attention (§4.2, Table 3).
- **Guideline:** When long-context ability is evaluated or targeted in training, measure task accuracy on the same problems at several lengths in addition to retrieval scores, because this paper finds that accuracy drops by a larger margin than exact-match retrieval in almost all task-length cells (§3.2, Table 6); when evidence can be recited reliably, a recite-then-solve prompt that removes the long context before answering improved GPT-4o on RULER QA2 by up to 4.0 points (§5, Table 5).
- **Authors:** Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, et al. (also Azton Wells, Roy Schwartz, Eliu A. Huerta, Hao Peng; UIUC, Amazon, USC ISI, Argonne, Hebrew University, U. Chicago)
- **Year:** 2025 (arXiv v1 2025-10; accepted at Findings of EMNLP 2025 per arXiv comments)
- **URL:** https://arxiv.org/abs/2510.05381
- **Source type:** paper
- **Relevant topics:** long-context evaluation, effective context length, retrieval vs reasoning decomposition, distraction, attention masking, recite-then-solve prompting

## Abstract
Long-context failures are usually attributed to retrieval: the model cannot find the relevant information in a long input. The authors ask whether a model that retrieves perfectly performs as well on a long input as on a short one. They build long-context versions of short tasks (math, QA, code, and a synthetic variable-summation task) by inserting filler tokens between the evidence and the question. Across 5 open and closed models, accuracy degrades substantially (13.9%-85%) as input length grows, even when the model can recite all evidence exactly. The drop persists when filler is whitespace, when filler is masked so the model attends only to evidence and question, and when evidence is placed directly before the question. The authors conclude that input length alone can hurt performance, and propose prompting the model to recite the evidence and then solve the task from a short prompt, which improves GPT-4o on RULER by up to 4%.

## Key Contributions
- A benchmark design of the form [Evidence][Distraction Tokens][Question] that holds the problem fixed and varies only length (§3.1, Fig. 2).
- Retrieval measured on the same problems by asking the model to recite evidence and question word by word, scored by exact match (§3.1).
- Three filler conditions ordered by decreasing distraction: Paul Graham essay tokens (§3), whitespace (§4.1), and attention-masked tokens (§4.2).
- A control that places the evidence adjacent to the question, so evidence-question distance does not change with length (§4.1, Fig. 4b).
- A model-agnostic recite-then-solve mitigation evaluated on GSM8K with essay filler and on RULER QA1/QA2 (§5).

## Key Figures/Tables to Study
- Fig. 1: 25,000 whitespace tokens do not stop the model from extracting the conditions, but it reaches the wrong answer.
- Fig. 3 / App. Table 6: accuracy vs exact-match retrieval for Llama-3.1-8B and Mistral-v0.3-7B at 0, 7,500, 15,000, 30,000 essay tokens.
- Table 2: closed-source models (GPT-4o, Claude, Gemini-2.0) with whitespace filler.
- Table 3 / App. Table 9: attention-masked filler.
- Tables 4 and 5: recite-then-solve on GSM8K (Mistral) and RULER QA1/QA2 (GPT-4o).

## Technical Details
**Tasks and data**
- GSM8K (evidence: problem description with chain-of-thought steps), MMLU (problem description; question with four options), HumanEval (function definition with docstring), and VarSum (values of 50 integer variables; question: sum of 3 random variables); test sets used (§3.1, Table 1).
- Evidence is one contiguous chunk at the start of the input, the question at the end; filler for §3 is Paul Graham essays as in Kamradt (2023) (§3.1).
- Retrieval score is exact match of the recited evidence and question; it is measured in a separate run, and the accuracy runs do not ask for recitation (§3.1; prompts in App. Figs. 6-11).

**Models**
- Open: Llama-3.1-8B-Instruct (claimed 128K context) and Mistral-v0.3-7B-Instruct (claimed 32K) (§3.2).
- Closed: GPT-4o, Claude-3.7-Sonnet (named "Claude-3.5" in Table 2), Gemini-2.0 (§4.1, Table 2).

**§3 Essay filler (App. Table 6; deltas in points from the 0-token baseline)**
- Retrieval drop is marginal until 30K tokens; below 15K tokens both models fail exact recitation on no more than 8.2% of problems (§3.2).
- Llama VarSum: accuracy 96.0 → -59.0 at 7,500 and -85.0 at 30,000; retrieval -8.0 at 7,500. Mistral VarSum: 68.0 → -44.0 at 7,500; retrieval -2.0 (Table 6; §3.2).
- Llama MMLU: accuracy 63.2 → -24.2 at 30,000 with retrieval delta 0.0 from 97.0; Llama HumanEval 57.3 → -47.6 at 30,000 (Table 6).
- Mistral HumanEval retrieval rises (+14.8, +16.7, +9.5) while accuracy falls (-17.7, -23.8, -34.8) (Table 6).
- A large part of the accuracy drop occurs within 7K tokens (§3.2).

**§4.1 Whitespace filler**
- Open models, whitespace between evidence and question, at 30K: at least 7 points (Llama GSM8K -7.0), Llama VarSum -48.0, Mistral GSM8K -30.0 (§4.1; App. Table 7).
- Whitespace before evidence, evidence adjacent to question, at 30K: up to -17.0 for Mistral and -20.0 for Llama, both on VarSum (§4.1; App. Table 8).
- Closed models (Table 2): GPT-4o and Gemini keep 100.0 on VarSum at all lengths; Claude MMLU 82.2 → -67.6 at 30,000; GPT-4o GSM8K 87.8 → -7.0 at 30,000; Gemini GSM8K improves (+7.7, +8.6, +6.2). The authors describe closed models as more robust than the open ones, with degradation in most model-task pairs (§4.1).

**§4.2 Masked filler (Table 3)**
- At 30,000 masked tokens: Llama VarSum -50.0, GSM8K -19.6, MMLU -21.1, HumanEval -50.0; Mistral VarSum -34.0, GSM8K -15.1, MMLU -11.8, HumanEval -7.9 (Table 3).

**§5 Recite-then-solve**
- Step 1: the model recites the relevant evidence from the long input. Step 2: the recited evidence plus the original question form a new short prompt, answered without the long context (§5, Fig. 5; RULER prompt in App. Figs. 12-13).
- Mistral GSM8K with essay filler, baseline vs recite-then-solve: 70.6/76.2 at 0, 49.3/71.4 at 3,750, 43.4/66.7 at 7,500, 41.6/69.1 at 15,000, 35.5/66.7 at 26,250 tokens; +31.2 at 26,250 (Table 4).
- GPT-4o RULER QA1 baseline 88.2-90.4 across 128K-4K; recite-then-solve reaches 92.2 at 4K. QA2 baseline 63.2-71.4; recite-then-solve 65.4-74.0, with the largest gain +4.0 at 32K (68.4 → 72.4) (Table 5).

**Compute and statistics**
- GH200 GPUs, around 20,000 GPU hours; every experiment run once (App. A.4).

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
### Long context
- Retrieval benchmarks that isolate retrieval "might overestimate progress", because better retrieval does not guarantee better long-context task performance (§1).
- The drop appears with evidence at the start (§3) and with evidence directly before the question (§4.1), so the authors attribute it to input length rather than evidence position or evidence-question distance (§4.1).
- The authors relate the result to explanations that attribute long-context drops to a position distribution bias introduced during training (An et al. 2024; Li et al. 2024a), and state that training retrieval and short-context skills separately "may not fully translate" into long-context ability (Interpretation; §6).
- The authors connect the result to RAG accuracy saturating or falling as more documents are added, and to reports that very long chains of thought can hurt reasoning (Interpretation; §1, §6).
### Limits stated by the authors
- Only 2 open models, 3 closed models, and 4 tasks; retrieval was not measured for closed models because they occasionally refused to recite long inputs (Limitations).
- The mitigation requires accurate retrieval; open-model RULER results were not reported because their retrieval failures reduce accuracy below baseline (Limitations).

## Connections
- [[ruler]] — RULER QA1/QA2 are the benchmark for the recite-then-solve experiment.
- [[needle-in-haystack-data]] — source of the Paul Graham essay filler; a retrieval-only test of the kind the paper says may overestimate progress.
- [[lost-in-the-middle]] — motivates placing evidence at the start, the easiest position to retrieve.
- [[string-effective-context]] — An et al. (2024), cited for the training position-bias explanation of effective-length shortfall.
- [[retrieval-head]] — cited for the retrieval-then-use view of long-context processing.
- [[artificial-needles-real-haystacks]], [[long-context-data-engineering]], [[ultralong-128k-to-4m]], [[yarn]] — cited as work in which retrieval improvements are taken as evidence of long-context progress (§2).
- [[nolima]], [[babilong]], [[infinitebench]] — cited long-context benchmarks.
- [[long-context-llms-meet-rag]] — cited for RAG degrading as more retrieved passages are added.
- [[chroma-context-rot]] — inference-only report with distractor and haystack-structure controls that also finds length-driven drops.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.05381 (arXiv v1, the only version; PDF read in full including App. A.1-A.5).
- Audit claims not found in the source: "Even with perfect retrieval and minimal irrelevant content, performance still drops 13.9%-85%" mixes two settings: the abstract's 13.9%-85% range is stated for the retrieval-controlled setting, and the whitespace (minimal-distraction) results are reported separately (≥7 to 48 points at 30K, §4.1). The paper does not state which table cells define the range; Table 6 contains -13.9 (Mistral MMLU, 7,500) and -85.0 (Llama VarSum, 30,000), but also smaller drops such as -5.4 (Llama GSM8K, 7,500).
- Internal inconsistencies in the source: Table 2 labels the Claude model "Claude-3.5" while §4.1 names Claude-3.7-Sonnet; Table 2 prints identical Claude VarSum and HumanEval rows (90.2, -0.6, -5.4, -4.8); §4.1 says Gemini GSM8K "at 30K actually improves by 8.6%", but Table 2 shows +8.6 at 15,000 and +6.2 at 30,000; §4.2 compares Llama HumanEval -50% masked with "19.4% with space", while Tables 7 and 8 print -31.7 and -18.3 for that cell.
- Not reported by the source: decoding temperature, number of problems per task and length (except the 1,000 MMLU problems in §1), confidence intervals.
