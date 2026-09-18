<!-- scope: BABILong long-context reasoning benchmark (bAbI facts hidden in PG19 books): generation protocol, LLM / RAG / fine-tuning results up to 50M tokens
     see-also: [[ruler]], [[needle-in-haystack-data]], [[longbench]], [[longalign]], [[babilong-recipe]]
-->

# BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack
- **Core Insight:** When bAbI reasoning facts are hidden inside PG19 book text, the tested LLMs effectively use only 10-20% of their context, accuracy falls as more supporting facts are required, and RAG reaches about 60% on single-fact questions at every length (Abstract; §3.1; §3.2).
- **Guideline:** When long-context ability is evaluated, report multi-fact tasks (QA2, QA3) at several lengths in addition to single-fact retrieval (QA1), because with background text every tested LLM stayed below 85% on QA2 while several models passed QA1 up to 16K-64K (§3.1).
- **Authors:** Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, et al. (AIRI, MIPT, London Institute for Mathematical Sciences)
- **Year:** 2024 (arXiv v1 2024-06; NeurIPS 2024 Datasets and Benchmarks Track)
- **URL:** https://arxiv.org/abs/2406.10149
- **Source type:** paper
- **Relevant topics:** long-context evaluation, effective context length, multi-fact reasoning, RAG vs long context, recurrent memory models, synthetic benchmark generation, contamination

## Abstract
Context windows of LLMs have grown, but evaluation has not kept pace: LongBench and L-Eval scale only to 40,000 tokens (§1). BABILong tests reasoning across facts distributed in long documents. It uses the 20 bAbI reasoning tasks (fact chaining, simple induction, deduction, counting, lists/sets) and hides their sentences inside PG19 books. The authors report that popular LLMs effectively use only 10-20% of the context and that performance declines as reasoning complexity increases. RAG methods reach about 60% accuracy on single-fact QA, independent of context length. Among context-extension methods, recurrent memory transformers after fine-tuning perform best and process up to 50 million tokens. The benchmark can be extended to any length, and splits up to 10 million tokens are provided.

## Key Contributions
- A generator that inserts bAbI task sentences between PG19 sentences until a target token length is reached, so the same task can be evaluated at any length (§2; App. N.3).
- An evaluation of over 30 long-input models of different sizes, architectures, and context-extension methods (§1; 38 models in Table 4, App. D).
- An effective-context measurement with explicit thresholds: satisfactory above 85% accuracy, complete failure below 30% (§3.1).
- A comparison of in-context reasoning with two RAG pipelines (§3.2; App. G).
- Fine-tuning experiments showing the tasks are solvable by small models: RMT and ARMT with a GPT-2 (137M) backbone and Mamba (130M) (§3.3; App. C). Settings: [[babilong-recipe]].
- A correlation analysis of BABILong and RULER against MMLU (§3.4, Fig. 4).

## Key Figures/Tables to Study
- **Fig. 1** — (a) sample generation; (b) best results per method, including GPT-4 using about 10% of its 128K window.
- **Table 1** — QA1-QA10: facts per task, relevant facts per task, and LLM accuracy at 0K.
- **Table 2** — average QA1-QA5 accuracy per model and input size, grouped by claimed context length.
- **Fig. 2** — QA1, QA2, QA3 accuracy vs context size with the 85% and 30% lines.
- **Fig. 3** — (a) RAG-C vs RAG-S on QA1; (b) task-specific fine-tuning vs RAG.
- **Fig. 9 (App. I)** — transfer to QA2-QA5 after fine-tuning GPT-3.5 and Mistral-7B on QA1.
- **App. N** — datasheet: composition, splits, sampling.

## Technical Details
### Sample construction
- bAbI tasks simulate characters and objects moving between locations; each event is a fact such as "Mary traveled to the office.", and questions ask about the simulated state, such as "Where is Mary?" (§2). Tasks are labeled QA1-QA20 (§2).
- Background text is PG19, a collection of books published before 1919 (§2; App. N.2). Books are split into sentences with `nltk.PunktSentenceTokenizer()` (App. N.4).
- Background sentences are added in their natural order until the sample reaches the desired length (§2). bAbI sentences are inserted between the sampled PG19 sentences with equal probability (uniform distribution) (App. N.3).
- A sample is unprocessed bAbI sentences (facts, distractor facts, question) mixed between unprocessed PG19 sentences; the question is placed at the beginning or the end; the label is the answer (App. N.2).
- Length is counted with the GPT-2 tokenizer (App. F).
- Prompt format: task description, few-shot examples in `<example>` tags, the instance in `<context>` tags, the question repeated, then a response-format instruction (App. J).

### Tasks, splits, and released data
- Table 1 (facts per task / relevant facts / median 0K accuracy across tested models): QA1 single supporting fact 2-10 / 1 / 99; QA2 two supporting facts 2-68 / 2 / 64; QA3 three supporting facts 4-320 / 3 / 38; QA4 two arg relations 2 / 1 / 55; QA5 three arg relations 2-126 / 1 / 80; QA6 yes-no 2-26 / 1 / 91; QA7 counting 2-52 / 1-10 / 28; QA8 lists-sets 2-50 / 1-8 / 77; QA9 simple negation 2-10 / 1 / 89; QA10 indefinite knowledge 2-10 / 1 / 80 (Table 1).
- Test pool: PG19 test split plus all bAbI test samples; the 100-samples-per-task-per-length set is randomly sampled from it. Train: all bAbI train samples plus randomly sampled PG19 train text (App. N.2). The test background seed is fixed; the training seed is not (App. C).
- Released evaluation set: 13,000 samples in 13 context-length splits across 10 tasks (App. N.2). Extended set: 60,000 samples, 5 tasks, 1,000 samples per split, 0k-128k (App. A; App. N.2).
- Pre-generated lengths named in App. F: 0k (no distractor text), 4k, 8k, 16k, 32k, 64k, 128k, 512k, 1M, 10M; Table 4 also reports 1K and 2K rows.
- Table 2 and Table 4 average QA1-QA5 only; Table 4 uses 1,000 samples per length up to 32K and 100 beyond (Table 4 caption).

### LLM results
- 23 of 34 tested LLMs reached 85% or more on any of QA1-QA3 without background text (§3.1).
- QA1: most models are efficient only up to 4K; GPT-4 and Llama-3.1-70B up to 16K; Qwen-2.5-70B and Gemini 1.5 Pro up to 64K. Full-context utilization on QA1 ranges from 5% to 50% (§3.1).
- QA2: only GPT-4 and Gemini 1.5 Pro solve it without background text; with background text all tested LLMs fall below 85%. QA3: best scores stay below 80% (§3.1).
- Models claiming 128K degrade beyond 10% of their input capacity (Table 2 caption). Performance relies on the first 5-25% of the input (Conclusions).
- LongChat, LongAlpaca, Llama-2-7B-32K and its instruct variant, fine-tuned on 32K lengths, did not perform well at 32K; Activation Beacon scored below 40% at 32K; Yi-9B-200k scored below 30% at 64K and above (§3.1).
- GPT-4-Turbo on QA1 has the lowest accuracy when all facts sit in the middle of the context (depth 50) (App. K, Fig. 10).
- Gemini 1.5 Pro 002 refused up to 14% of requests as context size increased, despite `BLOCK_NONE` (App. E).

### RAG and fine-tuning
- RAG-C retrieves 512-token chunks; RAG-S retrieves sentences; GPT-4 pipelines use `text-embedding-ada-002` with FAISS; the Llama-3 pipeline uses `nvidia/Llama3-ChatQA-1.5-8B` with `nvidia/dragon-multiturn-query-encoder` (§3.2; App. G).
- RAG-S outperforms RAG-C on QA1; raising retrieval from top-5 to top-20 sentences does not help (Fig. 3a caption). On QA2 and QA3 RAG accuracy falls below random guessing; the authors attribute this to lost fact order and low question similarity of the second fact (§3.2).
- RMT and ARMT were trained on 32 segments (16K tokens). RMT shows "only a marginal quality degradation" up to 128K and is evaluated at 1M, 10M, and 11.1M tokens (over 600 times the training length); ARMT reaches 50M (§3.3). Mamba has an advantage on QA3, but its inference beyond 128K was too slow for longer evaluation (§3.3; Table 4).
- GPT-3.5-Turbo and Mistral-7B fine-tuned on 1,000 QA1 samples for 3 epochs reach uniform scores across input sizes but remain limited to 16K and 32K context (§3.3; Conclusions).

### Benchmark comparison and stated limitations
- R² with MMLU at the RULER lengths most correlated with MMLU (≤128K average and 64K): "0.928 vs. 0.435 and 0.910 vs. 0.455, respectively" for RULER vs BABILong (§3.4). BABILong's correlation with MMLU is highest at 0K and decreases with length (Fig. 4 caption). The authors conclude BABILong separates models from 2K tokens, while RULER needs at least 128K to diverge from MMLU (§3.4).
- Only PG19 and Wiki backgrounds were tried; the RAG retriever and prompts were not optimized (Limitations).
- bAbI's small vocabulary lets fine-tuned models learn tokens that separate facts from background; distractor facts from the same vocabulary partly mitigate this (Limitations). Short, similar task sentences are distinguishable from book prose with few-shot examples (§2).
- The authors state generated benchmarks are immune to training-set contamination (§2). Tasks are English only (§4).

## Recipe ledger
The paper discloses fine-tuning settings for RMT, ARMT, Mamba-130m, GPT-3.5-Turbo, and Mistral-7B (§3.3; App. C). The full ledger is in [[babilong-recipe]].

## Findings relevant to generality and long context
- **Claimed vs effective context.** Across the tested LLMs, effective use is 10-20% of the context (Abstract), with QA1 utilization between 5% and 50% (§3.1). Result (single study).
- **Narrow fine-tuning transfer.** GPT-3.5 fine-tuned on QA1 improved on QA2-QA5; full fine-tuning of Mistral-7B on QA1 degraded QA2-QA5 (App. I, Fig. 9b-c; 0K, no distractor text; values shown only in the figure). Result (single study).
- **Checkpoint selection.** RMT accuracy beyond the training length varied across 3 runs with different memory initializations and data shuffles, and the authors state that early stopping on short-context accuracy may not be optimal (App. C, Fig. 5).
- **Long-context training.** After reporting Llama-3.1's improvement, the authors note that most new models use multistage pre-training with increasing sequence lengths; Llama-3.1 is pre-trained in six stages from 8K to 128K and mixes short-context data with synthetic long-context data in SFT (§3.1). The causal link to BABILong scores is not tested. Interpretation.
- **Synthetic-to-natural transfer.** The datasheet states that the synthetic tasks may not reflect real-world reasoning and that generalization to natural data is an open question (App. N.5).

## Connections
- [[ruler]] — compared in §3.4; the paper describes RULER haystacks as mostly repeated synthetic sentences.
- [[needle-in-haystack-data]] — the magic-number NIAH test that §1 describes as simple and scored by GPT-3.5 on a 1-10 scale.
- [[longbench]] — cited as scaling only to 40K tokens (§1; §4).
- [[longalign]] — LongAlign / LongBench-Chat listed among instruction-following long-context benchmarks (§4).
- [[longalpaca]] — LongAlpaca is one of the 32K fine-tuned models that failed at 32K (§3.1).
- [[llama-3]] — source of the multistage long-context training that §3.1 cites.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.10149 (arXiv v2, 2024-11-06), main text and App. A-N.
- Corrections to the previous card version:
  - Abstract lacked the headline results → added "effectively utilize only 10-20% of the context" and "RAG ... 60% accuracy on single-fact QA" (Abstract).
  - "Public configurations include 0k, 1k, 2k, 4k, ..., 256k, 512k, 1M, 10M" → App. F names 0k, 4k-128k, 512k, 1M, 10M; App. N.2 states 13 splits; 256k does not appear in the paper.
  - "Extended version with 1,000 samples per split for a subset of tasks" → 60,000 samples, 5 tasks, 0k-128k (App. A; App. N.2).
  - "20 reasoning tasks" left without scope → released set covers 10 tasks and Tables 2/4 average QA1-QA5 (App. N.2; Table 4).
  - Authors line listed seven names without affiliations → first six + et al. with affiliations (title page).
- Removed as unsupported by the source: "no LLM judge is needed for grading" and "no human annotation pass is required for each new length" (the scoring rule is not described); "reusable data-generation recipe" / "synthetic scaling harness" framing (course interpretation); the magpie connection (no relation in the source).
- Not reported by the source: the answer-matching rule used to compute accuracy; Table 2 cell values as text (the table is an image).
