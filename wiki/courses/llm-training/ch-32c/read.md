<!-- chapter: ch-32c
     track: midtraining
     kind: content
     title: Claimed versus Effective Context Length and Long-Context Evaluation
     deps: [ch-32b]
     sources: [[ruler]], [[helmet]], [[nolima]], [[longbench-v2]], [[michelangelo]], [[openai-mrcr-graphwalks]], [[lost-in-the-middle]], [[retrieval-head]], [[context-length-alone-hurts]], [[chroma-context-rot]], [[fiction-livebench]], [[babilong]], [[gradient-llama3-1048k]], [[interconnects-llama-4-long-context]], [[string-effective-context]], [[prolong]], [[controlled-long-context-extension]], [[needle-in-haystack-data]], [[llama-4]]
     figures: figures/effective-length-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32c — Claimed versus Effective Context Length and Long-Context Evaluation

> **Core insight.** The context window a model accepts is not the length it uses at a fixed accuracy. On RULER's 13 synthetic tasks, Llama 3.1 70B (claimed 128K) stays above the 85.6 threshold only up to 64K, and GradientAI's Llama-3 70B (claimed 1M) only up to 16K ([[ruler]] Table 3). The measured effective length depends on the task: with question-needle word overlap removed, the same Llama 3.1 70B has an effective length of 2K ([[nolima]] Table 3). Needle retrieval predicts application performance less well than a RAG task: across 35 instruction-tuned models at 128K, NIAH correlates with ∞Bench QA at Spearman ρ = 0.63, against 0.88 for the RAG task HotpotQA ([[helmet]] Fig. 4). Accuracy also falls with input length when retrieval is controlled: Llama-3.1-8B-Instruct recites the MMLU evidence as accurately at 30K filler tokens as at 0 (97.0% exact match) and still loses 24.2 accuracy points ([[context-length-alone-hurts]] App. Table 6).
>
> **Guideline.** When a context-extension stage (ch-32b) is gated, report effective length on RULER at every tested length together with at least one non-lexical or synthesis task (NoLiMa, Michelangelo or MRCR) and one application suite (HELMET categories or LongBench v2), because these three families rank the same models differently ([[helmet]] Table 5; [[nolima]] Table 3; [[michelangelo]] Fig. 9). When comparing checkpoints or models, record rope_scaling, truncation side, answer prefix or prefill, demonstrations, chat template, numeric precision and serving stack as evaluation hyperparameters, because single settings moved scores by 2.7 to 71 points in the sources below (§6). When a checkpoint is promoted, require a paired short-context comparison with a confidence interval, because with 1,000 items of which 110 change answer, the 95% interval on the paired difference is ±2.1 points, against ±4.0 unpaired (§6.3 worked example). Treat claims of 1M-10M windows supported only by NIAH heatmaps as anecdotal until an effective-length measurement exists (§7).

## Why this chapter matters for a general-purpose model

The pipeline for a general-purpose model is pre-training → mid-training → SFT → preference optimization → RL → evaluation. Context-length extension is a mid-training stage (ch-32b). This chapter covers the gate at the end of that stage: how to decide whether the extended model uses its window, and whether the extension removed short-context ability.

The measurable problem has three parts. First, release notes state a **claimed context length**: the maximum input the model accepts (its configured position range). A model can accept 1M tokens and still fail tasks at 32K ([[ruler]] Table 3). Second, the most common test, needle-in-a-haystack (NIAH: one inserted fact, retrieved by a question), is saturated: at 128K, five of six frontier models in [[helmet]] Table 5 score 100.0. Third, long-context scores are sensitive to evaluation settings that are rarely reported (§6), so two groups can measure different numbers for the same checkpoint.

For a general-purpose model, long context is not one skill. Applications that depend on it include retrieval-augmented generation (RAG), multi-document synthesis, repository-level code, many-shot in-context learning (ICL), long conversations, and agent trajectories (ch-45c). In [[helmet]] Fig. 5, category correlations at 128K fall to 0.34 (ICL vs Cite). A gate that measures one of these categories does not measure the others. ch-32b covers extension methods and data; ch-28 covers the synthetic task families used to build training data; ch-47 covers the evaluation harness in general.

## §1 Claimed, trained, and effective context length

**Definitions.**
- **Claimed length**: the input length the release states the model supports.
- **Training length**: the longest sequence length used in training. It can be shorter than the claimed length when inference-time scaling is used; Qwen2 72B was trained at 32K and claimed 128K in RULER's table ([[ruler]] §4).
- **Effective length**: the largest tested length at which a task score stays above a threshold. It is defined only relative to a task suite and a threshold.

**Problem.** Without a threshold definition, "supports 128K" cannot be compared across models or checked.

**Mechanism (RULER protocol, [[ruler]] §4, App. D).**
1. Generate 500 examples for each of 13 tasks at each length in {4K, 8K, 16K, 32K, 64K, 128K}.
2. Wrap each input in the model's chat template and append a task answer prefix, "to prevent the model from refusing to answer a query or generating explanations".
3. Decode greedily (vLLM, BFloat16) and score with recall-based accuracy.
4. Average the 13 task scores at each length.
5. Report the largest length whose average exceeds 85.6, the score of Llama2-7B-chat at 4K.

**Formulas.**

```
L_eff(threshold τ) = max { L ∈ tested lengths : S(L) > τ }
RULER:  τ = 85.6                         (fixed, all models)
NoLiMa: τ = 0.85 × B,  B = mean over question-needle pairs of max(S_250, S_500, S_1K)
wAvg(inc) = Σ_k k · S(L_k) / Σ_k k ,   wAvg(dec) = Σ_k (7−k) · S(L_k) / Σ_k k ,   k = 1..6
```

Here S(L) is the suite-average score at length L, τ the threshold, B NoLiMa's per-model base score, and L_k the k-th tested length from 4K to 128K. The weights "linearly increase or decrease with sequence length" ([[ruler]] §4); weights 1 to 6 reproduce the printed values below (derived).

**Worked example.** Llama3.1 (70B) scores 96.5, 95.8, 95.4, 94.8, 88.4, 66.6 at 4K-128K ([[ruler]] Table 3).
- Effective length: 88.4 > 85.6 at 64K; 66.6 < 85.6 at 128K. L_eff = 64K.
- wAvg(inc) = (96.5·1 + 95.8·2 + 95.4·3 + 94.8·4 + 88.4·5 + 66.6·6) / 21 = 1795.1 / 21 = 85.5.
- wAvg(dec) = (96.5·6 + 95.8·5 + 95.4·4 + 94.8·3 + 88.4·2 + 66.6·1) / 21 = 1967.4 / 21 = 93.7. Both match the table.
- NoLiMa for Llama 3.1 70B: B = 94.5, τ = 80.3; scores 91.0 (1K), 81.8 (2K), 71.2 (4K). L_eff = 2K ([[nolima]] Table 3).

The same checkpoint therefore has an effective length of 64K or 2K depending on the task suite. Neither number is wrong; each answers a different question.

**Evidence (RULER leaderboard rows for open models).**

| Model | Claimed | Effective | 32K | 128K | Locus |
|---|---|---|---|---|---|
| Llama3.1 8B (Instruct) | 128K | 32K | 87.4 | 77.0 | [[ruler]] Table 3 |
| Llama3.1 70B (Instruct) | 128K | 64K | 94.8 | 66.6 | Table 3 |
| Qwen2 72B (Instruct) | 128K | 32K | 94.1 | 53.7 | Table 3 |
| GradientAI/Llama3 70B | 1M | 16K | 85.4 | 72.1 | Table 3 |
| LWM 7B | 1M | <4K | 69.1 | 65.0 | Table 3 |
| ProLong 8B | 512K | 32K | 89.3 | 81.6 | RULER README @ab17b78 |
| GradientAI/Llama3 8B | 1M | 16K | 79.9 | 69.5 | RULER README @ab17b78 |

The paper summarizes: all 17 models "claim context sizes of 32K tokens or greater", but "only half of them can maintain satisfactory performance at the length of 32K" (Abstract). Nearly all models score near perfect on vanilla NIAH (Abstract; §4). Some leaderboard rows added later by model authors list an effective length that does not match their own scores against 85.6 (Qwen3-8B listed 64K with 82.1 at 64K; see [[ruler]] excerpt). The per-length scores are therefore the values to check; the effective-length column of an author-reported row is not always consistent with them.

**Conditions and limits.**
- The 85.6 threshold is a fixed reference from one old model. A model weak at 4K, such as LWM (82.3), gets "<4K" even if it degrades slowly; the authors call this "a trade-off in evaluation between absolute performance on short sequences and the relative degradation" ([[ruler]] §4). NoLiMa's relative threshold removes this dependence on the 4K score but makes effective length depend on the base score ([[nolima]] §4.3).
- [[babilong]] uses a per-task rule: above 85% is satisfactory, below 30% is failure (§3.1). With that rule the tested LLMs use "only 10-20% of the context" (Abstract), and GPT-4 uses about 10% of its 128K window (Fig. 1b).
- Effective length is measured only at the tested lengths. A model that passes 64K and fails 128K may fail at 70K.

**Implication.** A context-extension gate must name the suite, the threshold rule, and the full per-length curve.

The figure [effective-length-explorer.html](figures/effective-length-explorer.html) lets the reader apply the fixed or relative threshold to the RULER and NoLiMa rows and see how the effective length and the weighted averages change.

## §2 Why effective length falls short of training length

Three measured factors are reported: how often each relative position is trained, where the evidence sits, and which attention heads do the retrieval.

### §2.1 Relative-position frequency (STRING)

**Definition.** For a corpus C and training length L, the frequency of relative position i is the number of query-key pairs at distance i seen in training.

```
f(i) = Σ_{s ∈ C} max(|s| − i, 0),   0 ≤ i < L
```

|s| is a sequence's length in tokens ([[string-effective-context]] Eq. 2).

**Worked example.** One sequence of length 4 gives f(0) = 4, f(1) = 3, f(2) = 2, f(3) = 1. Adding a sequence of length 2 gives f(0) = 6, f(1) = 4, f(2) = 2, f(3) = 1. Every short document adds counts only to small distances, so large distances are always the rarest.

**Evidence.** On SlimPajama at L = 2048, distances i ≤ 1024 account for more than 80% of position indices and i ≥ 1536 for less than 5% (§2.2). Two 1.3B models pretrained on 1T tokens reach the same effective length (1,280) when f(1280) reaches the same count, 100B, although their training lengths are 2K and 4K (§3, Fig. 2b). In 13 open models, needle-retrieval failures peak when the needle is in the first third of the document, the farthest distance from the query (Table 4). The authors write that open models often have an effective length that "does not exceed half of their training length" (Abstract).

**Mechanism of the fix (STRING, inference only).** With training length L, offset S and local window W, distances d ≥ S are replaced by d − S + W; distances below S keep their values; only query position ids change, so the KV cache is unaffected (§4.1, Eq. 3-4, App. A.1).

**Worked example (derived from Eq. 3-4).** L = 10, S = 4, W = 2. The last query row originally uses distances [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]. Distances ≥ 4 become d − 2: [7, 6, 5, 4, 3, 2]; the rest stay [3, 2, 1, 0]. The new row is [7, 6, 5, 4, 3, 2, 3, 2, 1, 0]. Distances 8 and 9, the least trained, are no longer used.

**Result (single study).** With S = L/3 and W = 128, RULER at 128K for Llama 3.1 70B rises from 66.6 to 81.7, and for Qwen2 72B from 53.7 to 84.6; the 4-needle NIAH average over seven base models rises from 67.8 to 85.7 (Tables 1-2). **Limits:** no short-context or general benchmark is reported for STRING, and the method does not extend a model beyond its training length ([[string-effective-context]] §4.2; card "Generality"). **Implication:** part of the claimed-versus-effective gap is an under-training effect of the data length distribution (ch-32b, ch-04), not only a position-extrapolation failure (Interpretation of the authors).

### §2.2 Position of the evidence (lost in the middle)

**Definition.** Position sensitivity is the change in accuracy when only the location of the relevant information changes.

**Evidence.** With 20 retrieved documents (about 4K tokens), GPT-3.5-Turbo answers 75.8% when the answer document is first, 53.8% when it is 10th, and 63.2% when it is 20th; its closed-book accuracy with no documents is 56.1% ([[lost-in-the-middle]] App. G Table 6; Table 1). The best-minus-worst gap is 75.8 − 53.8 = 22.0 points, and the worst case is below closed-book accuracy. Instruction tuning reduced the MPT-30B gap from about 10 to about 4 points but did not remove it (§4.3). Extended-window variants (GPT-3.5-Turbo 4K vs 16K) are nearly identical on inputs that fit both (§2.3). At 128K, [[helmet]] App. E.4 reports that models generally favor the most recent context. [[babilong]] App. K finds GPT-4-Turbo lowest on QA1 when facts sit at depth 50%.

**Conditions.** [[chroma-context-rot]] reports no notable variation across 11 needle positions in its NIAH variant ("Needle-Question Similarity › Results"), so the size of the position effect depends on the task. **Implication:** report accuracy by depth and the best-minus-worst gap, because an average over positions hides the U-shape ([[lost-in-the-middle]] §1).

### §2.3 Retrieval heads

**Definition.** A retrieval head is an attention head whose most-attended input token, while the model generates a needle token, is that same token inside the needle ([[retrieval-head]] §2).

```
retrieval_score(h) = |g_h ∩ k| / |k|
```

g_h is the set of needle tokens that head h copied, and k the needle's tokens (Eq. 1). A head with score above 0.1 counts as a retrieval head (§2).

**Worked example.** A 10-token needle; head h is the top-attending copier for 9 of the generated needle tokens. Score = 9/10 = 0.9 (toy numbers; score definition from §2).

**Evidence (Result, single study).** Across Llama-2, Yi, Qwen1.5 and Mistral families, fewer than 5% of heads exceed 0.5 (Fig. 3 caption). Retrieval-score maps of a base model and its long-context, chat, or MoE derivative have Pearson correlation above 0.8 (§3.3). Masking the top 20 retrieval heads of Llama-2-7B-80K lowers NIAH to 63.6, against 94.7 for 20 random heads (Fig. 1). Masking retrieval heads reduces extractive-QA F1 by 9.2% and 23.1% on Mistral-7B-Instruct-v0.2 and lowers chain-of-thought accuracy, while answer-only MMLU changes little (§4.2-4.3).

**Implication (Interpretation, this course, based on [[retrieval-head]] §5).** The authors argue that retrieval heads need the full KV cache. KV-cache compression, sliding-window attention, and hybrid architectures change what these heads can attend to, so a long-context gate for such changes should include retrieval and CoT-over-context tasks, not only perplexity.

## §3 Retrieval is necessary but not sufficient: length alone lowers accuracy

**Definition.** A length-only effect is a change in accuracy when the problem, the evidence, and retrieval success are held fixed and only the number of input tokens changes.

**Mechanism ([[context-length-alone-hurts]] §3.1).**
1. Take a short task (GSM8K, MMLU, HumanEval, or VarSum: sum 3 of 50 integer variables).
2. Build inputs as [evidence][filler][question] at 0, 7,500, 15,000 and 30,000 filler tokens.
3. In one run, ask the model to recite the evidence and question word by word; exact match is the retrieval score.
4. In another run, ask for the answer.

**Evidence.**
- Llama-3.1-8B-Instruct on MMLU: accuracy 63.2 at 0 tokens and −24.2 points at 30,000; retrieval 97.0 with delta 0.0 (App. Table 6). On VarSum: 96.0, then −59.0 at 7,500 and −85.0 at 30,000, with retrieval −8.0 at 7,500.
- With filler masked from attention, drops at 30,000 remain: Llama GSM8K −19.6, Mistral HumanEval −7.9 (Table 3).
- With whitespace filler placed before the evidence, so evidence sits next to the question, drops at 30K reach −20.0 for Llama on VarSum (App. Table 8).
- Recite-then-solve (recite evidence, then answer from a short prompt without the long context) raises Mistral GSM8K at 26,250 essay tokens from 35.5 to 66.7 (Table 4) and GPT-4o RULER QA2 at 32K from 68.4 to 72.4 (Table 5).
- **Limits:** 2 open and 3 closed models, 4 tasks, every experiment run once, no confidence intervals (Limitations; App. A.4).

**Practitioner evidence.** [[chroma-context-rot]] tested 18 models with the task held fixed. Needle-question pairs with lower embedding similarity degrade faster with length; topically related distractors lower accuracy more as length grows; all models score higher on focused LongMemEval prompts (~300 tokens) than on full prompts (~113k tokens) built from the same 306 questions. Claude models lose points mainly through abstention and GPT models through wrong answers (LongMemEval; Impact of Distractors). The report gives no per-model numbers in text.

**Implication.** A model can pass retrieval at a length and still reason worse there. Effective length for a general-purpose model should be measured on tasks that use the retrieved facts (Replicated across [[context-length-alone-hurts]] and [[chroma-context-rot]], both with fixed-task designs).

## §4 Evaluation beyond needle retrieval

### §4.1 HELMET: application categories, and what NIAH predicts

**Definition.** HELMET evaluates seven categories at controlled lengths from 8K to 128K: RAG, generation with citations (Cite), passage re-ranking, many-shot ICL, long-document QA, summarization, and synthetic recall ([[helmet]] §2.1).

**Problem.** A test that costs less to run is useful as a proxy only if it ranks models the way the application tasks do. Rank agreement is measured with Spearman's ρ.

**Worked example first.** Five models are ranked by test A as 1, 2, 3, 4, 5 and by test B as 2, 1, 4, 3, 5. Rank differences d are −1, 1, −1, 1, 0, so Σd² = 4.

```
ρ = 1 − 6 Σ d_i² / (n (n² − 1)) = 1 − 6·4 / (5·24) = 0.8
```

d_i is the rank difference for model i, n the number of models; the formula holds without ties. When many models tie at 100.0, as on NIAH, ranks are averaged and the test separates models poorly.

**Evidence (Result, single study; 35 instruction-tuned models at 128K).** Spearman correlation with ∞Bench QA is 0.63 for NIAH, 0.81 for RULER MK needle, and 0.88 for RAG HotpotQA (Fig. 4). NIAH's correlations with all downstream categories are ≤ 0.8 (§3.1). Rankings invert: RULER puts Gemini-1.5-Flash (86.6) above Gemini-1.5-Pro (65.3), while HELMET gives 50.7 and 62.7 (Table 5). At 128K, the §3.3 prose states that closed models lead open ones on Cite and Re-rank by 30-40 points; the v3 Fig. 6 values give 31.8 points for Re-rank and 17.2 for Cite against the best listed open model (derived in the card). Many open models lead on ICL (Jamba-1.5-Mini 91.0 vs GPT-4o 86.3, Fig. 6).

**Metric choice.** A Mistral-7B summary made of one sentence repeated hundreds of times scores ROUGE-L 12.3 and GPT-4o-judge 0.0 (App. B.6). HELMET uses judge-based scoring for QA and summarization, with human agreement κ between 0.72 and 0.91 (App. B.6).

**Implication.** When a fast check during development is needed, the HELMET authors recommend RAG tasks, because they correlate more with the other downstream categories than synthetic recall does; when a model decision is made, they recommend all seven categories ([[helmet]] Abstract, §3.1).

### §4.2 NoLiMa: removing literal overlap

**Definition.** Literal overlap is the ROUGE-1 precision of question tokens found in the relevant context. It is 0.905 for vanilla NIAH, 0.571 for RULER S-NIAH, 0.689 for HELMET RAG, and 0.069 for NoLiMa ([[nolima]] Table 1).

**Mechanism.** The needle ("Actually, [CHAR] lives next to the Semper Opera House.") and the question ("Which character has been to Dresden?") share no keyword; the model must use the association between the landmark and the city. Haystacks are filtered for distractor words and unintended answers; each of 58 question-needle pairs is placed 26 times in 5 haystacks, giving 7,540 tests per length (§3-§4.1).

**Evidence.** 11 of 13 models score at or below half of their base score at 32K; GPT-4o falls from 99.3 to 69.7 (Table 3). No model in Table 3 has an effective length above 8K, including models claiming 1M-2M. For Llama 3.3 70B at 32K: 98.5 when the question names the landmark, 56.2 for a one-hop association, 25.9 for two-hop (Table 6); CoT raises two-hop to 34.3 (Table 4). **Limits:** effective length depends on the 0.85 × base rule; no training interventions are studied.

### §4.3 LongBench v2: long multiple-choice reasoning with human baselines

[[longbench-v2]] contains 503 four-option questions over contexts of 8k-2M words in six categories (single-document QA, multi-document QA, long ICL, dialogue history, code repositories, structured data). Items were kept only if three 128k-context LLMs did not all answer correctly and a human reviewer could not answer within 3 minutes (§1). Human experts score 53.7% under a 15-minute limit; GPT-4o-2024-08-06 scores 50.1% answering directly; o1-preview scores 57.7% (Table 2). Without the context, GPT-4o scores 33.1% (Table 3), which bounds how much of the score parametric knowledge can explain. Qwen2.5-72B-Instruct scores higher with a 32k retrieved context than with its full 128k window (+4.1%, §4.2). **Limits:** the authors state that 503 items may give less stable results (§6); the interval width is computed in §6.3 of this chapter; scores by length bucket are not comparable because task mix differs by bucket (Table 2 note).

### §4.4 Michelangelo: latent-structure queries

[[michelangelo]] builds tasks in which the context contains relevant updates to a hidden structure plus irrelevant text, and the query asks about the structure. Complexity (number of relevant updates) is fixed while length grows (§3). Tasks: Latent List (track a Python list through operations), MRCR (reproduce the i-th of several similar writing requests in a long conversation), and IDK (answer "I don't know" when the context lacks the fact; 70% of instances) (§2). The authors report that models often degrade by 32K and describe "a very common trend across all of the Michelangelo evaluations": "one initial sharp super-linear drop in performance in short-context", followed by flat or roughly linear decline (§5.2). Rank correlations across the ten models at 128K are 0.64 (MRCR vs Latent List), 0.043 (MRCR vs IDK), and −0.25 (Latent List vs IDK) (Fig. 9). GPT and Claude beat Gemini on MRCR below 8K but decay faster (§5.3). **Status:** Result (single study) from the organization that trained one of the evaluated model families.

### §4.5 OpenAI MRCR and Graphwalks

Both are official OpenAI datasets ([[openai-mrcr-graphwalks]]).

**MRCR.** A synthetic conversation hides 2, 4, or 8 identical requests; the model must return the i-th response, prefixed by a given hash. Distractor responses come from the same generator as the needles. 100 samples per token bin, bins from 4,096 to 1,048,576 tokens.

```
score = 0                                  if response does not start with hash
score = 2·M / (|response| + |answer|)      otherwise (difflib SequenceMatcher ratio)
```

M is the number of matched characters. **Worked example:** answer "abcd", response "abce" after the hash: M = 3, ratio = 6/8 = 0.75. A copy of the wrong poem scores above 0 when the two poems share character sequences, so MRCR scores are similarity scores, not accuracy (Interpretation from the metric definition).

**Graphwalks.** The prompt lists directed edges and asks for a BFS frontier at a given depth or the parents of a node; the last line must read "Final Answer: [...]". **Worked example:** gold {a, b, c}, answer {a, b, d}: precision 2/3, recall 2/3, F1 = 0.667. A missing "Final Answer:" line scores 0.

**Measurement errors recorded by the maintainers.** About 10% of MRCR items had too many target needles and about 5% had wrong ground truth until 2025-12-05; 24 of 400 Graphwalks `parents` items had wrong ground truth until 2026-02-27. Scores computed on earlier dataset revisions are not comparable to later ones. Per-model results are in the GPT-4.1 release post, which was not retrieved for this chapter (not reported here).

### §4.6 BABILong and Fiction.LiveBench

[[babilong]] hides bAbI reasoning facts in PG19 book text. With background text, all tested LLMs fall below 85% on QA2 (two supporting facts), and QA3 stays below 80% (§3.1). RAG reaches about 60% on single-fact QA at every length and falls below random on QA2 and QA3 (§3.2). BABILong's correlation with MMLU is highest at 0K and decreases with length; at the lengths where RULER correlates most with MMLU, R² with MMLU is 0.928 for RULER's ≤128K average versus 0.435 for BABILong (§3.4). The authors read this as BABILong separating long-context skill from general knowledge at shorter lengths (Interpretation).

[[fiction-livebench]] (practitioner evidence, private questions) asks the same story questions over a minimal "0"-token version and longer versions. In its September 2025 table, claude-sonnet-4 scores 100.0 at 0 and 36.4 at 120k, and claude-sonnet-4:thinking 81.3 at 120k. The page reports no run counts or intervals.

## §5 Perplexity as a contested proxy

**Definition.** Perplexity over N tokens is PPL = exp( −(1/N) Σ_t log p(x_t | x_<t) ), where x_t is the token at position t, x_<t the preceding tokens, and p the model's predicted probability. **Worked example:** a mean negative log-likelihood of 2.0 nats per token gives PPL = e² ≈ 7.39.

**The disagreement.**
- [[controlled-long-context-extension]] fixes LLaMA2-7B, 1B tokens of the same data, and a 32k training length (64k for the NTK-64K variant), and varies the extension method. For exact-attention fine-tuned methods, PPL@32K correlates with NIAH, LongBench, and RULER (§6, Fig. 4). The authors conclude that perplexity "remains a general-purpose performance indicator" (Abstract). The exception: LM-Infinite has PG19 perplexity 6.71 at 32k but scores 12.34 on RULER at 32k (Tables 2-3).
- [[prolong]] fixes the method and varies the data mixture for Llama-3-8B. As the long-data share rises to 100%, PG19 perplexity keeps improving while the downstream long-context average falls (§2.1, Fig. 1).

**Interpretation (this course).** The two results measure different comparisons. Across position methods with fixed data, perplexity tracks whether distant tokens are used at all. Across data mixtures, more long book text lowers book perplexity without improving instruction-following retrieval, RAG, or re-ranking, which ProLong also finds become measurable only after SFT (§2.2, Fig. 2). Perplexity and task scores can also move in opposite directions at short lengths: PI, NTK-32K and NTK-64K have higher PG19 perplexity at 2k than LLaMA2 (6.63-6.88 vs 6.61), while their RULER scores at 4K are higher (84.56-86.60 vs 80.94) (Tables 2-3).

**Implication.** When choosing between data mixtures or deciding to promote a checkpoint, use downstream tasks after SFT. Perplexity is a usable early signal only for comparing position methods on fixed data (Open question beyond that setting).

## §6 Evaluation configuration as hyperparameters

### §6.1 Settings with measured effects

| Setting | Measured effect | Source and locus |
|---|---|---|
| rope_scaling at inference | Qwen2.5-72B-Instruct on LongBench v2: 39.4 → 42.1 with YaRN factor 4.0; Qwen2.5-7B: 27.0 → 30.0 | [[longbench-v2]] App. E Table 4 |
| RoPE base changed at inference | Llama-3-Inst with base 500,000 → 16,000,000 drops past 32,768 tokens; Llama-3-8B-Inst also loses ODQA and ICL scores at short lengths | [[helmet]] App. E.3 |
| Scaling factor depends on test length | Dynamic NTK factor 29 up to 32k and 61 at 64k for the same checkpoint | [[controlled-long-context-extension]] Tables 7-8 |
| Assistant prefill / answer prefix | Claude 2.1 on NIAH: 27% → 98% with prefill "Here is the most relevant sentence in the context:"; RULER appends an answer prefix to all models | [[needle-in-haystack-data]] Connections (Anthropic post, 2023-12-06); [[ruler]] App. D |
| In-context demonstrations | Llama-3.1-8B base, JSON KV at 128K: 77.3 → 98.0 with 2-shot; GPT-4o: 99.3 → 36.7 | [[helmet]] Table 8 (3 seeds) |
| Harness prompts | Llama-3.2 1B/3B: 1.8-2.8 on ∞Bench vs 21.2-36.9 on HELMET | [[helmet]] Table 9 |
| Reasoning mode | claude-sonnet-4 at 120k: 36.4 → 81.3 with thinking | [[fiction-livebench]] Sept 2025 table |
| Truncation side | LongBench v2 truncates from the middle; HELMET truncates LongQA documents from the end; no controlled comparison in these sources | [[longbench-v2]] §4.1; [[helmet]] §2.1 |
| Length unit | HELMET counts Llama-2 tokens; BABILong GPT-2 tokens; LongBench v2 words; MRCR o200k tokens | [[helmet]] §3; [[babilong]] App. F; [[longbench-v2]] §1; [[openai-mrcr-graphwalks]] |
| Serving stack and precision | RULER and HELMET run BFloat16; Fiction.LiveBench rows are labeled with providers and fp8, and Llama 4 rows were rerun "after inference provider updated with vllm fixes"; no source here measures KV-cache quantization on these benchmarks | [[ruler]] §4; [[helmet]] App. D; [[fiction-livebench]] Changelog 4/10/2025 |

The effects range from 2.7 points (YaRN on Qwen2.5-72B) to 71 points (Claude 2.1 prefill). The effect of KV-cache precision on long-context scores is an Open question in the sources used here; record it because the model's served configuration can differ from the evaluated one. For example, the Gradient 1M model card recommends serving with vLLM `--max-model-len 32768` ([[gradient-llama3-1048k]]).

### §6.2 Minimal record for every long-context number

```yaml
model_revision: <hub id @ commit>
rope_scaling: {type: yarn, factor: 4.0, original_max_position_embeddings: 32768}   # or null
max_model_len: 131072
truncation: middle            # middle | left | right, and the token budget reserved for the answer
length_unit: tokens:<tokenizer>
chat_template: <file hash>;  answer_prefix: "<text>";  demonstrations: 0 | 2
decoding: {temperature: 0.0, max_new_tokens: <n>, reasoning: off | <budget>}
serving: {engine: vllm==<ver>, dtype: bfloat16, kv_cache_dtype: auto | fp8}
dataset_revision: <commit>    # MRCR and Graphwalks changed ground truth after release
n_items_per_cell: <n>;  seeds: <k>
```

The field names follow Hugging Face and vLLM conventions; the values shown are illustrative except the YaRN factor 4.0 from [[longbench-v2]] App. E.

### §6.3 Paired short-context regression with a confidence interval

**Problem.** An extension stage should not remove short-context ability (ch-32b). A single-point difference on a 1,000-item benchmark is within sampling noise, and the noise is smaller when the same items are compared.

**Mechanism.**
1. Run the base and the extended checkpoint on the same items with the same configuration (§6.2).
2. For each item i, compute d_i = correct_ext(i) − correct_base(i) ∈ {−1, 0, 1}.
3. Report the mean difference and a 95% interval.

```
d̄ = (1/n) Σ d_i ,   SE = sqrt( (mean(d²) − d̄²) / n ) ,   CI95 = d̄ ± 1.96 · SE
```

n is the number of items, d̄ the mean paired difference, mean(d²) the mean of the squared per-item differences, SE the standard error, and 1.96 the normal quantile for a two-sided 95% interval.

**Worked example (toy numbers).** n = 1,000. The base answers 700 correctly, the extended model 690. 60 items flip from right to wrong and 50 from wrong to right. d̄ = (50 − 60)/1000 = −0.010. mean(d²) = 110/1000 = 0.110. SE = sqrt((0.110 − 0.0001)/1000) = 0.0105. CI95 = −1.0 ± 2.1 points, that is [−3.1, +1.1]. An unpaired interval would use sqrt(0.70·0.30/1000 + 0.69·0.31/1000) = 0.0206, giving ±4.0 points. Pairing halves the interval here because 890 items did not change.

**Evidence of the need.** [[context-length-alone-hurts]] runs every experiment once (App. A.4); [[longbench-v2]] states its 503 items "could also lead to less stable results" (§6). For 503 items at 50% accuracy, an unpaired SE is sqrt(0.25/503) = 0.022, so ±4.4 points (derived). After SFT, ProLong and Llama-3-8B-Instruct differ by 0.4 points on the five-task short average (69.4 vs 69.8) while MMLU differs by 2.4 (64.6 vs 67.0) and GSM8K by 9.6 (58.9 vs 68.5) ([[prolong]] App. B.7, Table 25), so an average can hide a per-task regression.

## §7 Claims of 1M-10M windows: what counts as evidence

- **Gradient Llama-3-8B-Instruct-Gradient-1048k.** The model card states 1.4B training tokens in total across four stages (RoPE θ from 15.3M to 2.80B) and shows NIAH heatmaps and a RULER ranking image as evidence ([[gradient-llama3-1048k]]). The capability claim "can learn to operate on long context with minimal training" is **anecdotal** in this course's labels, because its only support is NIAH heatmaps and a RULER image with no numeric values in the card text. Independent measurement: RULER effective length 16K for both the 8B and 70B Gradient models, with 128K scores 69.5 and 72.1 ([[ruler]] README; Table 3). The RULER authors list GradientAI/Llama3 among "less performant models" trained at 1M (§4).
- **Llama 4 Scout.** Meta states 10M supported tokens and training at 256K; its long-context evidence in the post is NIAH retrieval and cumulative NLL on code ([[llama-4]]). An Interconnects post notes that no evaluation beyond NIAH was released and suspects serving issues in early third-party results ([[interconnects-llama-4-long-context]]; **anecdotal**, not used as evidence). Practitioner evidence: llama-4-scout:free scores 62.5 at the 0-token length and 27.3 at 120k in Fiction.LiveBench's September 2025 table, so its low long-context score is not isolated from a short-context failure ([[fiction-livebench]]).

A claim of a supported window is evidence only that inputs of that length are accepted. An effective-length claim needs a suite, a threshold rule, per-length scores, and the configuration in §6.2.

## Recipe

Evaluation-gate values as printed in the sources. Rows marked 2026-09-15 were read in the primary text for this chapter; rows marked 2026-09-14 come from verified library cards.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| RULER main runs (17 aligned models, e.g. Meta-Llama-3.1-70B-Instruct) | 7B-141B | eval-gate | effective-length threshold | 85.6 = Llama2-7B (chat) 13-task average at 4K | arXiv:2404.06654v3 §4, Table 3 ([[ruler]]) | verified 2026-09-15 | no ablation reported; described as "a qualitative threshold" (§1) |
| RULER main runs | 7B-141B | eval-gate | tasks; examples; lengths | 13 tasks; 500 per task per length; 4K, 8K, 16K, 32K, 64K, 128K | §4 | verified 2026-09-15 | task set chosen by the correlation study in App. C |
| RULER main runs | 7B-141B | eval-gate | prompt; decoding; precision | model chat template + task answer prefix; one demonstration for VT and CWE; greedy; BFloat16 on vLLM, 8 A100 | §4; App. D Table 6 | verified 2026-09-15 | no ablation reported |
| RULER main runs | 7B-141B | eval-gate | ranking weights | linear, increasing (inc) or decreasing (dec) with length | §4 | verified 2026-09-15; weights 1-6 derived (§1 worked example) | no ablation reported |
| NoLiMa (13 models, e.g. GPT-4o, Llama 3.3 70B) | various; open models 8B-405B | eval-gate | effective-length threshold | 0.85 × base score; base = mean over pairs of max score at 250/500/1K | arXiv:2502.05167v3 §4.3 ([[nolima]]) | verified 2026-09-14 | no ablation reported |
| NoLiMa | as above | eval-gate | tests per length; decoding | 7,540 (5 haystacks × 58 pairs × 26 placements); greedy for instruction models; reasoning models capped at 1,536 generated tokens | §4.1; App. C | verified 2026-09-14 | no ablation reported |
| HELMET (59 models) | various (Table 15) | eval-gate | lengths; decoding; samples | 8,192 / 16,384 / 32,768 / 65,536 / 131,072 Llama-2 tokens; greedy; 600 (JSON KV, NQ, PopQA, TQA), 300 (MS MARCO, HotpotQA), 500 (ICL), 100 (others) | arXiv:2410.02694v3 §3; App. D ([[helmet]]) | verified 2026-09-14 | Table 8: 0-shot vs 2-shot, 3 seeds |
| HELMET | as above | eval-gate | judge | GPT-4o-2024-05-13; Cohen's κ 0.72-0.91 against humans | §2.2; App. B.6 | verified 2026-09-14 | App. B.6 human agreement study |
| LongBench v2 (17 models) | open 7B-123B; closed not reported | eval-gate | items; truncation; settings | 503 MCQ; middle truncation beyond the model window; zero-shot and zero-shot + CoT | arXiv:2412.15204v2 §1, §4.1 ([[longbench-v2]]) | verified 2026-09-15 | Table 2: CoT +3.4 average for open models |
| Qwen2.5-72B-Instruct on LongBench v2 | 72B | eval-gate | rope_scaling | YaRN, factor 4.0 (per model card); overall 39.4 → 42.1 | App. E Table 4 | verified 2026-09-15 | Table 4 (one run) |
| Michelangelo (10 frontier models) | not reported | eval-gate | context subsets; prompting | up to 32K, 128K, 1M; buckets divided by repetition count; few-shot short demonstrations | arXiv:2409.12640v2 §3.2-3.3 ([[michelangelo]]) | verified 2026-09-15 | no ablation reported |
| OpenAI MRCR dataset | not applicable | eval-gate | needles; samples; bins; metric | 2, 4, 8; 100 per bin; 8 bins from 4,096 to 1,048,576 tokens (prompt + answer); SequenceMatcher ratio, 0 without hash prefix | huggingface.co/datasets/openai/mrcr README @f4c69fa ([[openai-mrcr-graphwalks]]) | verified 2026-09-15 | not reported |
| OpenAI Graphwalks dataset | not applicable | eval-gate | problem types; metric | bfs, parents; set F1 on the last-line "Final Answer:" list, parse failure = 0 | huggingface.co/datasets/openai/graphwalks README @be6cc6e | verified 2026-09-15 | not reported |
| BABILong (34 LLMs in §3.1) | various (Table 4) | eval-gate | thresholds; samples | satisfactory > 85%, failure < 30%; 100 samples per task per length (Table 4: 1,000 up to 32K) | arXiv:2406.10149v2 §3.1; App. N.2; Table 4 caption ([[babilong]]) | verified 2026-09-14 | no ablation reported |
| ProLong 8B development runs | 8B | eval-gate | development protocol | HELMET subset (recall, RAG, re-rank, ICL, QA, summarization) averaged over 32K and 64K, after UltraChat SFT, plus 5 short tasks | arXiv:2410.02660v4 §2; App. A.1 ([[prolong]]) | verified 2026-09-14 | §2.1 Fig. 1 (perplexity misleads); §2.2 Fig. 2 (before vs after SFT) |
| Llama-3-8B-Instruct-Gradient-1048k | 8B | long-context | tokens; stages; RoPE θ; LR | 1.4B total (838,860,800 in the 1048k stage); 65K → 262K → 524k → 1048k; 15.3M / 207.1M / 1.06B / 2.80B; 2.00E-05 | HF model card @cd3069b ([[gradient-llama3-1048k]]) | verified 2026-09-15 | card evidence: NIAH heatmaps and a RULER image without values (anecdotal); RULER README: effective 16K |
| Llama 4 Scout | 17B active / 109B | long-context | trained context; supported context | 256K pre- and post-training; 10M supported | Meta post 2025-04-05, Scout paragraph ([[llama-4]]) | verified 2026-09-14 | post evidence: NIAH and cumulative NLL only |

**Starting point for a small general-purpose run.** For an 8B model extended to 128K (ch-32b), the verified rows support this gate. Run RULER's 13 tasks with 500 examples per task at 4K-128K, chat template plus answer prefix, greedy BFloat16, and report per-length scores with the 85.6 threshold, as RULER did for 7B-141B aligned models on A100s. Add NoLiMa with the 0.85 × base rule and 7,540 tests per length, as run for 13 models up to 32K. Add HELMET's seven categories at 8,192-131,072 tokens with its per-dataset sample counts, and evaluate after SFT as ProLong did at 8B. Record the configuration fields of §6.2, and report short-context regressions as paired intervals (§6.3). None of these sources tested a gate for models below 1B parameters.

## Generalization lens

**(a) What increases breadth of long-context capability.**
- Evaluating several task families at once, because they rank models differently: HELMET category correlations fall to 0.34 ([[helmet]] Fig. 5), Michelangelo task correlations to −0.25 ([[michelangelo]] Fig. 9). No model leads all categories in either study.
- Selecting training choices by downstream long-context tasks after SFT instead of perplexity ([[prolong]] §2.1-2.2).
- Larger models degrade less with length at equal context training: Yi-34B-200K vs Yi-6B-200K ([[ruler]] §6). Result (single study).
- Inference-time reasoning partly compensates: CoT +3.4 average on LongBench v2 for open models ([[longbench-v2]] §4.1); two-hop NoLiMa at 32K from 25.9 to 34.3 ([[nolima]] Table 4). Replicated direction; effect sizes differ.

**(b) What causes narrowing or forgetting.**
- Position changes made to extend the window can lower short-context scores: [[helmet]] App. E.3 (Llama-3-8B-Inst ODQA and ICL after an inference-time RoPE base change); [[controlled-long-context-extension]] Table 11 (LongBench LCC 68.22 → 56.78 after NTK-32K).
- Optimizing for NIAH-style retrieval: models near 100% on NIAH still fail RULER aggregation and tracing ([[ruler]] §4) and NoLiMa associations ([[nolima]] Table 6). A suite of retrieval-only gates selects for lexical matching.
- Heavy instruction tuning may lower ICL: many open models beat closed models on HELMET ICL ([[helmet]] §3.3; Interpretation of the authors).
- Long-only training data: 100% long data lowered downstream long-context scores after SFT and monotonically lowered short-task scores ([[prolong]] §3.2, Fig. 3).

**(c) How to measure it at this stage.**
- Per-length curves, not one effective-length number; relative and fixed thresholds side by side (§1).
- Depth slices and best-minus-worst gap ([[lost-in-the-middle]] §1).
- Low-overlap and distractor slices ([[nolima]]; [[chroma-context-rot]]).
- Retrieval-controlled accuracy: the same problems at several lengths with a recitation check ([[context-length-alone-hurts]] §3.1).
- A no-context baseline to bound memorization ([[longbench-v2]] Table 3) and regenerable synthetic tasks to avoid leakage ([[michelangelo]] §1; [[babilong]] §2).
- Abstention scored separately from wrong answers ([[michelangelo]] IDK; [[chroma-context-rot]] LongMemEval).
- Paired short-context regression with intervals (§6.3), and dataset revisions pinned (§4.5).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating the claimed or configured window as the usable window | Release lists 1M; RULER per-length scores fall below 85.6 at 16K-32K | Compute L_eff from per-length scores under a named threshold (§1) |
| Gating extension on single-needle NIAH | All checkpoints score near 100; application scores diverge | Add RULER MK/MV/VT/CWE, NoLiMa, and a HELMET RAG task; compute rank correlation across checkpoints |
| Copying the effective-length column without the scores | Leaderboard row effective length disagrees with its own per-length scores | Recompute from per-length scores (Qwen3-8B README row, §1) |
| Reporting one average over needle depths | Mean looks acceptable; failures concentrate in the middle | Plot accuracy by depth; report best-minus-worst gap |
| Using perplexity to choose a long-data mixture | PG19 perplexity improves while RAG and recall after SFT decline | Evaluate the candidate mixtures after SFT on downstream tasks ([[prolong]] Fig. 1) |
| Comparing numbers run with different rope_scaling | Same model differs by 3 points between reports | Log rope_scaling and max_model_len per run (§6.2) |
| Evaluating a base model zero-shot on instruction-shaped tasks | Low scores that rise with 2 demonstrations (Llama-3.1-8B base JSON KV at 128K: 77.3 → 98.0) | Run 0-shot and 2-shot; report both ([[helmet]] Table 8) |
| Scoring on an unpinned dataset revision | MRCR or Graphwalks score shifts with no model change | Record the dataset commit; check the changelog |
| Declaring "no short-context regression" from a small unpaired difference | A −1 point change reported as neutral or as a loss without an interval | Paired per-item differences with CI95 (§6.3) |
| Counting abstentions as ordinary errors | A model that says "not in context" scores the same as one that hallucinates | Score abstention and wrong answer separately (IDK; LongMemEval) |

## Check your understanding

1. Llama 3.1 70B has an effective length of 64K on RULER and 2K on NoLiMa. Explain which properties of each suite (task type, literal overlap, threshold rule) produce each number, and state which one a RAG application with paraphrased queries should rely on.
2. Using f(i), explain why packing short documents into 128K sequences still leaves distances near 128K rare, and why STRING can raise RULER at 128K without training but cannot extend a model beyond its training length.
3. [[context-length-alone-hurts]] masks the filler tokens from attention and still finds drops at 30K. Which explanations of long-context failure does this result rule out, and which does it leave open?
4. NIAH correlates with ∞Bench QA at 0.63 and HotpotQA RAG at 0.88 across 35 models. Explain how score saturation on NIAH lowers Spearman's ρ, using the tie-handling rule.
5. Lu et al. find perplexity tracks downstream scores; ProLong finds it does not. Describe an experiment that would decide whether the disagreement comes from varying methods versus varying data.
6. A team reports that YaRN factor 4 made its extended model better on LongBench v2 by 2.7 points, compared with another lab's number measured without YaRN. List the configuration fields that must match before the comparison is valid, and explain why each can change the score.
7. A 128K extension changes a 1,000-item short benchmark from 70.0 to 68.0, with 80 right-to-wrong and 60 wrong-to-right flips. Compute the paired 95% interval, explain why it is narrower than the unpaired interval, and state what the gate should conclude.
8. Why is a Fiction.LiveBench score of 27.3 at 120k for Llama 4 Scout weak evidence about its long-context ability specifically, and what additional measurement would isolate the length effect?

## Connections

- **ch-32b (previous)** — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression. Methods and data whose outputs this chapter's gate measures; STRING's position-frequency view connects the two.
- **ch-32d (next)** — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training. Repository and trajectory data are long inputs; §4 and §6 apply to evaluating them.
- **ch-28** — Long-Context Data Synthesis and Synthetic Evaluation Task Families. RULER-style generators as training data.
- **ch-32f** — Lab: Annealing and Context Extension with a Short-Context Regression Gate. Implements the gate in the Recipe paragraph and §6.3.
- **ch-44a** — Length in RL: Overlong Responses, Length Control, and Long-Context RL. Long outputs and long-context RL.
- **ch-45c** — Context Management for Long-Horizon Agents. Uses the length-only effect (§3) to motivate context management.
- **ch-47** — Evaluation Harness and Suite Design for General Capability. General harness design; links here for long-context evaluation.

## Sources

- [[ruler]] — task suite, effective-length definition, threshold 85.6, Table 3 and README leaderboard rows, answer-prefix protocol.
- [[nolima]] — literal-overlap measurement, relative threshold, Table 3, association and CoT ablations.
- [[helmet]] — seven categories, NIAH and RAG correlations, judge metrics, demonstrations and RoPE-base configuration effects.
- [[longbench-v2]] — human-verified long MCQ, no-context baseline, middle truncation, YaRN results.
- [[michelangelo]] — latent-structure queries, early degradation before 32K, cross-task rank correlations.
- [[openai-mrcr-graphwalks]] — MRCR and Graphwalks task definitions, grading code, dataset corrections.
- [[lost-in-the-middle]] — position sensitivity and best-minus-worst gap.
- [[retrieval-head]] — retrieval score, sparsity, stability, masking effects.
- [[string-effective-context]] — position-frequency explanation and STRING results.
- [[context-length-alone-hurts]] — length-only accuracy drops with retrieval controlled; recite-then-solve.
- [[chroma-context-rot]] — 18-model practitioner evaluation with similarity, distractor, and LongMemEval controls.
- [[babilong]] — reasoning-in-a-haystack thresholds, 10-20% context use, RAG limits.
- [[fiction-livebench]] — practitioner comprehension ladder; reasoning-mode and Llama 4 rows.
- [[controlled-long-context-extension]] — perplexity correlation under a controlled method comparison.
- [[prolong]] — perplexity versus downstream under data changes; evaluate after SFT; short-task results.
- [[needle-in-haystack-data]] — original NIAH harness and the Claude 2.1 prefill result.
- [[gradient-llama3-1048k]] — 1M-window training settings and NIAH-only evidence (anecdotal label).
- [[llama-4]] — Scout's 256K training length, 10M supported window, and stated evidence.
- [[interconnects-llama-4-long-context]] — commentary on Llama 4 long-context evidence (anecdotal).
