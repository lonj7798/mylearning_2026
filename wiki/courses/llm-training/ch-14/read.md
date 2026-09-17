<!-- chapter: ch-14
     track: pretraining
     kind: content
     title: Data-Constrained Scaling, Repetition, and Pretraining Decontamination
     deps: [ch-13a]
     sources: [[data-constrained-scaling]], [[repeated-data-scaling]], [[scaling-laws-data-quality]], [[nemotron-cc]], [[epoch-data-stock]], [[rephrasing-the-web]], [[d4]], [[c4]], [[llama-3]], [[llama-3-recipe]], [[llama-3-decontamination]], [[evaluation-data-contamination-contam]], [[rephrased-samples-contamination]], [[olmo-3-decontamination]], [[tulu-3]], [[paloma]], [[dolma]], [[deduplicating-training-data]], [[livecodebench]]
     figures: figures/repetition-curve.html, figures/ngram-contamination.html
     revised: 2026-09 (generality revision)
-->

# Chapter 14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination

> **Core insight.** In GPT-2-architecture models up to 8.7B parameters trained on C4, repeating the whole unique dataset for about 4 epochs gave loss close to unique data (8.7B: 0.5% higher validation loss), and the fitted decay constant R*_D = 15.39 caps effective data at 16.39 times the unique tokens ([[data-constrained-scaling]] §6, App. A). Repeating a small subset is more harmful than its token share suggests: repeating 0.1% of the data 100 times degraded an 800M model to the level of a 400M model ([[repeated-data-scaling]] Abstract). When high-quality documents would otherwise be repeated, replacing 4 of 8 repetitions with rephrased versions raised a 10-task average from 55.8 to 56.7 at 8B ([[nemotron-cc]] Table 10). N-gram decontamination misses paraphrased test items, and its false-negative rate rises with n and with the minimum match count ([[rephrased-samples-contamination]] Table 5, [[evaluation-data-contamination-contam]] §6), so the match rule, thresholds, and removal unit must be recorded for later score-effect estimates.
>
> **Guideline.** When the planned token count is at most about 4 passes over the unique tokens left after global deduplication and decontamination, repeat the whole pool uniformly, because in [[data-constrained-scaling]] that range changed loss and a 19-task average negligibly (tested up to 8.7B on C4, §6, §7). When more passes would be needed, compute effective tokens with Eq. 5 and first add other data: Python code up to 50% of tokens (4.2B, 84B tokens, §7) or rephrased versions of high-quality documents ([[nemotron-cc]] §3.3). When a mixture upsamples a small source, track that source's pass count and training loss, because degradation peaked where training loss on the repeated subset approached zero ([[repeated-data-scaling]] §1.1). When estimating contamination, use 8-token n-grams with a minimum count of 1 and choose thresholds per benchmark and model size, because larger n and higher minimum counts produced false negatives ([[evaluation-data-contamination-contam]] §6.1, §6.3, §7.3). When removing contamination from training data, match every split that is evaluated, apply the check at least to mid-training and annealing data, and add a paraphrase-aware check for benchmarks that decide releases, because 10-gram matching gave F1 = 0 on rephrased items ([[rephrased-samples-contamination]] Tables 5-6). Otherwise, if no decontamination is run, report the benchmarks as unchecked.

## Why this chapter matters for a general-purpose model

A general-purpose model is often trained on more tokens than a filtered corpus contains. Chapters [[ch-10]], [[ch-10a]], and [[ch-12]] remove low-quality and duplicate text, and [[ch-13]] and [[ch-13a]] set the mixture across domains and languages. This chapter handles two consequences of those decisions in the pre-training stage (pre-training → mid-training → SFT → preference optimization → RL → evaluation).

The first consequence is scarcity. After filtering, the number of unique tokens can be smaller than the number of tokens the compute budget allows. Some text must then be seen more than once, rewritten, or replaced by other sources. Each option affects breadth: repetition can shift a model from generalization toward memorization, and rewriting can add formats or introduce errors.

The second consequence is contamination. Web text contains copies and paraphrases of evaluation benchmarks. If they reach the training data, benchmark scores stop measuring performance on unseen items, which is the quantity that defines general capability in [[ch-00]]. Decontamination is decided in the data pipeline, but its correctness can only be checked later, in evaluation ([[ch-48]]).

The chapter answers the three course questions for this stage:
1. Breadth: how many repetitions keep held-out loss and downstream averages close to unique data, and which substitutes (code, rephrasing) keep or add breadth (§2-§4).
2. Narrowing: how subset repetition damages copying and in-context mechanisms, and how contamination raises scores without adding ability (§3, §5-§6).
3. Measurement: held-out loss on never-repeated data, per-source training loss, estimated performance gain (EPG) with per-benchmark thresholds, and time-segmented evaluation (§5-§7).

## §1 Unique tokens, total tokens, and the stock of text

**Definition.** *Unique tokens* U are the tokens of a corpus after global deduplication. *Total tokens* D are the tokens processed during training. The number of *passes* (epochs) is D / U, and the number of *repetitions* is R_D = D / U − 1 ([[data-constrained-scaling]] §3).

**The measurable problem.** Chinchilla-style scaling laws were fitted on single-pass runs. Muennighoff et al. state that the Chinchilla parametric fit "explicitly relies on the assumption that models are trained for a single epoch only" (§5). When D > U, a one-pass law predicts a loss that is lower than the run will reach.

**How large U is in practice.**
- Aggressive model-based filters remove about 90% of Common Crawl data ([[nemotron-cc]] Abstract). After global fuzzy deduplication, DCLM has 3.8T total and 1.0T unique tokens, and FineWeb-Edu has 1.3T total and 0.2T unique (Table 4).
- Nemotron-CC keeps more data: 6.3T tokens, of which 4.4T are globally deduplicated real tokens and 1.9T are synthetic (Table 4).
- Villalobos et al. estimate a deduplicated indexed-web stock of 510T tokens [95% CI 130T, 2100T], a quality-adjusted stock of 100T [22T, 490T], and a repetition-adjusted stock of 320T [65T, 1700T] ([[epoch-data-stock]] Figure 3). They state with 95% certainty that "between 10% and 40% of deduplicated web data can be used for training" (§2.3.1), and their median year for full use of the stock is 2028 (§2.5). These are forecasts from a model of stock and demand (**Interpretation**).

**Worked example (derived).** Llama 3.1 405B was pre-trained on 15.6T tokens ([[llama-3-recipe]], v3 §1), which is 15.6T / 405B = 38.5 tokens per parameter. Villalobos et al. give a compute-optimal ratio of "around 20" ([[epoch-data-stock]] App. F) and state that Llama 3 8B "is overtrained by close to 100x" (footnote 24). A small model reaches the unique-token limit sooner. A 1.5B model planned for 1T tokens uses 667 tokens per parameter. If its pool after global deduplication and decontamination is 250B tokens, the run makes 4 passes (R_D = 3). The token-budget decision itself is covered in [[ch-08a]].

**Implication for a general-purpose model.** Small models trained far beyond the compute-optimal ratio are the most likely to repeat data. The pass count must be computed per source, not only for the whole corpus, before training starts.

## §2 The effective value of repeated tokens

**Definition.** *Effective data* D′ is the number of fresh tokens that would give the same loss as training on U unique tokens repeated R_D times ([[data-constrained-scaling]] §3.1, App. A).

**The problem it addresses.** A run with D = 4U processes four times as many tokens as one pass over U, but the repeated tokens carry less new information. The question is how much less, as a function of R_D.

**Mechanism** (App. A).
1. Assume that each time the model trains on a token, it learns a fraction 1 − δ of the information remaining in it.
2. The value of the k-th repetition is then (1 − δ)^k times the value of a fresh token, which is a geometric series.
3. The sum approaches U + U(1 − δ)/δ as R_D grows. The authors define R*_D = (1 − δ)/δ and approximate the series with an exponential (Eq. 10-13).
4. The same form is applied to parameters. Parameters beyond the compute-optimal count for U ("excess parameters") also lose value (Eq. 6).
5. R*_D and R*_N are fitted on 182 runs with 7M to 9B parameters and 1 to 500 epochs.

**Formula** ([[data-constrained-scaling]] §3.1, Eq. 5-6):

```
D′ = U_D + U_D · R*_D · (1 − exp(−R_D / R*_D))
N′ = U_N + U_N · R*_N · (1 − exp(−R_N / R*_N))
L(N, D) = A / N′^α + B / D′^β + E
```

- U_D: unique tokens used, min{D_C, D}, where D_C is the unique-data budget.
- R_D: repetitions of the data, D / U_D − 1 (0 for one pass).
- R*_D: fitted constant 15.387756 (App. A). At R_D = R*_D the repeated tokens are worth on average 1 − 1/e of fresh tokens (§3.1).
- U_N: compute-optimal parameter count for U_D, or N if smaller. R_N = N / U_N − 1. R*_N: fitted constant 5.309743 (App. A).
- A, B, α, β, E: Chinchilla-form constants. The fitted law is L = 521 / N′^0.35 + 1488 / D′^0.35 + 1.87 with U_N = 0.051 · U_D (App. A, Eq. 17).

Two properties follow from Eq. 5. At R_D = 0, D′ = U_D = D, so one pass loses nothing. As R_D grows, D′ approaches U_D(1 + R*_D) = 16.39 U_D; in the authors' words, "no matter how many times we repeat the data, we will not get a better loss than could be obtained with a single epoch on U_D + U_D R*_D fresh tokens" (§3.1). Because R*_N = 5.31 < R*_D = 15.39, excess parameters lose value faster than repeated tokens, and the data-constrained efficient frontier allocates additional compute to epochs faster than to parameters (§5).

**Worked example (derived from Eq. 5 with R*_D = 15.387756).** The marginal value dD′/dR_D = U_D · exp(−R_D / R*_D) is the value of the next increment of repetition relative to fresh tokens.

| Passes | R_D | D′ / U_D | D′ / D (average value per processed token) | Marginal value |
|---|---|---|---|---|
| 1 | 0 | 1.000 | 1.000 | 1.000 |
| 2 | 1 | 1.968 | 0.984 | 0.937 |
| 4 | 3 | 3.726 | 0.931 | 0.823 |
| 8 | 7 | 6.624 | 0.828 | 0.635 |
| 16 | 15 | 10.582 | 0.661 | 0.377 |
| 40 | 39 | 15.167 | 0.379 | 0.079 |
| ∞ | ∞ | 16.388 | → 0 | → 0 |

For the §1 example (U = 250B, D = 1T), D′ = 250B × 3.726 = 931B, so the 1T processed tokens are worth about 931B fresh tokens. For U = 5B and D = 40B (8 passes), D′ = 33.1B. A formula of the form U(1 − exp(−R/R*)) would give fewer effective tokens than unique tokens even at one pass, which contradicts D′ = U at R_D = 0.

A second check uses the full fitted law (Eq. 17, derived). For N = 8.7B and D = 178B with U = 178B (one pass): U_N = 9.08B exceeds N, so U_N = N, R_N = 0, and L = 0.1730 + 0.1718 + 1.87 = 2.2148. With U = 44B (4.05 passes): U_N = 2.244B, R_N = 2.877, N′ = 7.23B, D′ = 165.6B, and L = 0.1846 + 0.1762 + 1.87 = 2.2308, which is 0.72% higher. The measured difference for this pair of runs was 0.5% (§6).

The interactive figure [figures/repetition-curve.html](figures/repetition-curve.html) lets the reader set unique tokens and passes and read off D′, the average value, the marginal value, and the 16.39 U ceiling from Eq. 5.

**Evidence.**
- Return at fixed compute (§6, Figure 4): IsoFLOP runs of 2.8B parameters on 55B tokens, 4.2B on 84B, and 8.7B on 178B. "the N = 8.7 billion parameter model trained for four epochs (D_C = 44 billion unique tokens) finishes training with only 0.5% higher validation loss than the single-epoch model (D_C = 178 billion unique tokens)." **Result (single study)**.
- "Meaningful gains from repeating data can be made up to around 16 epochs (R*_D) beyond which returns diminish extremely fast" (§6).
- Fit quality: R² = 0.7722 for the decay form against 0.4452 when repeated data is assumed to keep full value (App. A Table 1).
- Allocation at scale: at 9.3 × 10^21 FLOPs with 25B unique tokens, the model chosen by the data-constrained frontier had 27% fewer parameters than the Chinchilla choice and better loss and downstream performance (§5).
- Downstream: at 4.2B parameters and 84B tokens, differences on a 19-task average (5 seeds) are insignificant up to about 4 passes and then drop (§7, Figure 6).
- A second setting agrees for moderate repetition: in T5 Base (about 220M parameters, 2^35 training tokens), a dataset repeated 64 times gave SuperGLUE 72.03 against 71.36 with no repetition, while 1,024 repeats gave 64.76 ([[c4]] Table 9). **Replicated** for "a few passes change quality little" across these two settings; the tested ranges differ.

**Conditions and limits.**
- Only repetition of the whole dataset was studied; "one can repeat only a fraction of the dataset" is listed as future work (App. Q). §3 covers that case.
- Returns "may heavily depend on hyperparameters such as learning rate, dropout, or the optimizer choice" (App. Q). The runs used dropout 0.1, weight decay 0.1, and gradient clipping at 1.0 (App. S).
- Data: C4, with OSCAR for the filtering experiments. Architecture: GPT-2. Largest model: 8.7B for the return experiments and 9B in the fit.
- The fit "significantly underestimates the final test loss of failing models where loss increases midway through training, such as models trained for 44 epochs" (§6).
- Fine-tuning under data constraints was not studied (App. Q).

**Implication for a general-purpose model.** Four passes over a clean, diverse pool is a tested range for pre-training models up to 8.7B. The value of the next increment of repetition is 0.82 of fresh data at 4 passes and 0.38 at 16 passes (derived), so beyond about 4 passes the alternatives in §4 should be compared against more repetition.

## §3 Repeating part of the data: upsampling and non-uniform repetition

**Definition.** *Non-uniform repetition* means that some documents are seen more times than the rest, either on purpose (upsampling a high-quality source in a mixture, [[ch-13]]) or by accident (duplicates that survive deduplication, [[ch-12]]).

**The problem it addresses.** Eq. 5 assumes that every token is repeated equally. A mixture can make 1.1 passes overall while one small source is seen dozens of times, and the aggregate pass count does not show this.

**Mechanism** ([[repeated-data-scaling]] §1.1, §3).
1. Train decoder-only models for 100B tokens drawn from a 400B-token corpus (55% filtered Common Crawl, 32% books, plus smaller sources).
2. Draw a fixed fraction of the training tokens (for example 10%) as repeats of a small subset and the rest without repetition.
3. Vary the subset size, the repeated fraction, and the model size. The same 10% of tokens can be 0.01% of the data repeated 1,000 times or 1% repeated 10 times.
4. Measure loss on a held-back split that contains no repeated data.
5. Degradation follows a double-descent pattern: a few repeats and the highest repeat counts cause little damage, and an intermediate range causes the most. The peak "coincides with where the train loss on the repeated data approaches zero".

**Worked example (arithmetic).** In a 100B-token run, 0.1% of the data is 100M tokens. Repeated 100 times, it contributes 10B tokens, which is 10% of training, while the other 90B tokens are unique. For a mixture check: a source of 200M unique tokens that receives 2% of a 100B-token run is seen 2B / 200M = 10 times, even if the corpus as a whole makes only 1.1 passes.

**Evidence.**
- "performance of an 800M parameter model can be degraded to that of a 2x smaller model (400M params) by repeating 0.1% of the data 100 times, despite the other 90% of the training tokens remaining unique" (Abstract). **Result (single study)**.
- With 3% repeated data at the worst repetition count, effective model size fell by up to 3x on a copying evaluation (the first paragraph of Harry Potter copied 11 times) and by 32% on average on a prefix-matching score, while test loss showed at most a 15% reduction (§1.1). Prefix matching measures induction heads, attention circuits that copy earlier patterns; [[ch-08b]] connects them to in-context learning.
- Loss on Python code, which is out of distribution for this text corpus, also degraded, mainly when 50-90% of tokens were repeated (§1.1).
- Repeating a *selected* subset can help. At 1.3B parameters and 40B tokens (3 seeds), one pass over random 40B tokens gave non-web perplexity 16.27; two passes over random 20B gave 16.39; two passes over 20B selected by D4 gave 16.10 ([[d4]] Table 1).

**Conditions and limits.** All models in [[repeated-data-scaling]] were trained for a fixed 100B tokens, not at the compute-optimal point. The repeated subset was random rather than high-quality, the measurements are loss rather than downstream tasks, and regularization was not varied (§5.5). The authors extrapolate "meaningful degradation of repeating data only 2 times for large (GPT-3 size) models" and state that the region would shift for compute-optimal training (§1.1) (**Interpretation**). D4 shows that the identity of the repeated documents matters (**Result (single study)**).

**Implication for a general-purpose model.** Upsampling a small source can cost more than its share of tokens suggests, and in this study the cost fell on copying and induction-head measures, which are connected to in-context learning across tasks. Per-source pass counts and per-source training loss are the diagnostics.

## §4 When high-quality unique tokens run out: quality, code, and rephrasing

**The problem it addresses.** When the unique tokens that pass a quality filter are fewer than the planned tokens, a run can repeat them (§2), relax the filter to raise U at lower per-token quality (§4.1), add a different data type such as code (§4.2), or rephrase existing documents into new token sequences (§4.3). The measurable question is which option gives lower held-out loss or a higher downstream average at the same number of training tokens.

**Worked example: quantity against filter strength (derived from Eq. 5 and [[nemotron-cc]] Tables 4-5).** Suppose a 15T-token run draws all of its web tokens from one pool. DCLM (1.0T unique) would need 15 passes: D′ = 1.0T × 10.19 = 10.2T effective tokens. Nemotron-CC's real tokens (4.4T unique) would need 3.41 passes: D′ = 4.4T × 3.23 = 14.2T. Eq. 5 does not model quality, so the per-token quality of the two pools must be measured separately. At 1T training tokens with 73% drawn from the tested dataset, DCLM and the full Nemotron-CC gave 10-task averages of 57.0 and 57.8 and MMLU of 53.4 and 53.0 (8B, Table 5). The authors "expect it to be superior in data-constrained settings like 15T token training runs" because it has 4× more unique real tokens (§3.2) (**Interpretation**); no 15T comparison of the two pools is reported. The subsections below give the evidence on quality, code, and rephrasing.

### §4.1 Quality as a scaling variable

**Definition.** Subramanyam et al. define a data-quality parameter Q ∈ (0, 1], where Q = 1 is clean data. With their corruption-rate estimator, a dataset in which 10% of samples are corrupted has Q = 0.9 ([[scaling-laws-data-quality]] §1, §3.1).

**Formula** (§1, §6):

```
L(N, D, Q) = A / N^α + B / (D^β · Q^γ) + E
```

- N: parameters; D: training tokens; Q: data quality; A, B, α, β, γ, E: fitted constants.
- E does not depend on Q in this form; the authors' plots support that the additive terms "do not vary with data quality Q" (§5.5).
- Because γ = β·γ0 (App. A.3), D tokens at quality Q have the same data term as D · Q^(γ/β) clean tokens (derived).

**Worked example (derived from Table 2, CLM, Huber fit: B = 1441.505289, β = 0.395859, γ = 0.400657, E = 3.439047; computed with these full-precision values, since rounding β to 0.3959 gives D^β = 3,657).** For D = 1B tokens, D^β = 3,654. At Q = 1 the data term is 1441.5 / 3,654 = 0.394 and L = 3.834. At Q = 0.75, Q^γ = 0.891, the data term is 0.443, and L = 3.882. The clean-token equivalent is 1B × 0.75^1.012 = 0.747B. In this fit, corrupting 25% of samples costs about as much as removing 25% of the tokens.

**Evidence.** Fits use an 8-layer Llama-3-style model (hidden size 512) trained for one pass on 0.1B, 1B, and 10B C4 tokens, and a 133M GPT-Neo translation model. Quality is set by synthetic noise (50% of tokens swapped in each perturbed sample for language modeling) at 7 levels from Q = 1.0 to 0.5 (§5.1-§5.3). γ̂ = 0.401 for language modeling and 0.173 for translation (§5.5). **Result (single study)**.

**Conditions and limits.** Model size is fixed within each task, so the N term is not tested. The noise is synthetic token corruption, not the low-quality text a web filter removes, and outcomes are loss rather than downstream tasks. The authors read γ̂ < 1 as effective data decaying "sublinearly with quality" (§5.5). Under their reparameterization γ = β·γ0, the language-modeling fit gives γ/β = 1.01 and the translation fit gives 0.69 (derived), so the sublinear reading holds for translation, while the language-modeling fit is close to proportional (**Interpretation**).

### §4.2 Code and filtering under a data constraint

At 4.2B parameters and 84B total tokens ([[data-constrained-scaling]] §7, 19 tasks, 5 seeds for repetition and code):
- "Filling up to 50% of data with code (42 billion tokens) also shows no deterioration. Beyond that, performance decreases quickly on natural language tasks." WebNLG and bAbI rise as soon as code is added; the authors suggest that code may teach long-range state tracking (**Interpretation**).
- Perplexity filtering (keep the 25% of samples with lowest perplexity: 44B tokens repeated about 2 times) helped on C4. Deduplication filtering (remove samples with a 100-character overlap: 21B tokens repeated 4 times) did not help on the benchmark (§7, App. N). Both filters were more effective on the noisier OSCAR corpus (App. O). The authors "recommend reserving filtering for noisy datasets" in a data-constrained regime and note that deduplication "may have value not captured in our benchmark, such as reducing memorization" (§7).
- The effects of code on other capabilities are covered in [[ch-09]].

### §4.3 Rephrasing and synthetic expansion instead of repetition

**Definition.** *Rephrasing* rewrites existing documents with a language model into another style or format (Wikipedia-like prose, question-answer pairs, knowledge lists), producing new token sequences from the same content ([[nemotron-cc]] §2.3). Generation methods are taught in [[ch-19]].

**The problem it addresses.** High-quality documents are the ones a mixture most wants to repeat, and they are the scarcest. Rephrasing produces new token sequences from them without a new crawl.

**Mechanism** ([[nemotron-cc]] §2.3, Table 3).
1. Score documents with an ensemble of quality classifiers and group them into quality buckets.
2. Rewrite low-quality documents with a Wikipedia-style prompt to reduce noise and errors.
3. Rewrite high-quality documents with the Wikipedia-style prompt and four additional prompts (diverse QA pairs, distill, extract knowledge, knowledge list) to "obtain more unique tokens and condense essential knowledge". Medium-quality documents are not rephrased.
4. Split documents into segments before rewriting, because long inputs produced "over-simplified outputs with reduced detail".
5. Generate with Mistral NeMo 12B Instruct (FP8, top-p 0.9, temperature 0.5): over 1.8T tokens, 336.3B from low-quality and 1.5T from high-quality documents.

**Evidence.**
- 8B models trained on 1T tokens each, with 73% of tokens from the tested Common Crawl variant ([[nemotron-cc]] §3.1, Table 10). HQ-Base contains "eightfold high-quality documents"; HQ-Synthetic swaps "4 repetitions of the high-quality documents" for synthetic data. The 10-task average rises from 55.8 to 56.7 and MMLU from 53.4 to 53.6. Rephrasing low-quality documents raises the average from 52.5 to 54.0, but MMLU falls from 48.2 to 47.1. Seeds are not reported. **Result (single study)**.
- Long horizon: an 8B model trained for 15T tokens, 7.17T of them from Nemotron-CC (App. E), scored MMLU 70.3 against 65.3 for Llama 3.1 8B, both measured in the authors' own evaluation harness (Table 6).
- WRAP: on C4, training on real and rephrased text "speeds up pre-training by ∼ 3×" ([[rephrasing-the-web]] arXiv:2401.16380v1 Abstract). A 350M model trained on 15B tokens from a 1.5B-token pool (about 10 repeats) had lower perplexity with real plus rephrased data than with synonym-replacement or random-deletion augmentation (§6.2, Figure 6; values shown only in the figure). **Replicated** across [[nemotron-cc]] and [[rephrasing-the-web]] for the claim that rephrased text outperformed the alternative it was compared with; the alternatives differ (repetition in one, augmentation in the other).

**Conditions and limits.** The authors propose two causes for the HQ-Synthetic gain, "fresh unique tokens" and styles such as question answering, and do not separate them (§3.3) (**Open question**). The rephrased data was not checked for "factual accuracy or fidelity to the original contents" (§6). WRAP states that rephrasing "limits the model from learning new 'knowledge'" (§8). Model collapse from recursive synthetic training is covered in [[ch-23]], and synthetic continued pre-training in [[ch-32a]].

**Implication for a general-purpose model.** When high-quality documents would be repeated beyond about 4 passes, rephrased versions are a tested substitute for part of the repetition. The per-task table must be read, because in the same table rephrasing low-quality documents raised ARC-Easy, ARC-Challenge, OpenbookQA, and CommonsenseQA by 1.80 to 4.75 points and lowered MMLU ([[nemotron-cc]] §3.3).

## §5 Decontamination at pretraining scale: match rules and their errors

### §5.1 Definitions and the measurable problem

*Evaluation data contamination* is "the inadvertent mixing of samples from evaluation benchmarks into pre-training corpora" ([[evaluation-data-contamination-contam]] Abstract). *Decontamination* removes matching training text before training. A *contamination analysis* estimates after training how much the matches changed scores, without removing anything. The *estimated performance gain* (EPG) is the score on the full benchmark minus the score on the subset marked clean (§4.1).

The measurable problem is inflated scores. In the Llama 3 8-gram analysis, 85% of HellaSwag was marked contaminated with an estimated gain of 14.8 points at 8B, while 52% of NaturalQuestions was marked with gains of 1.6, 0.9, and 0.8 at 8B, 70B, and 405B ([[llama-3]], [[llama-3-decontamination]], v3 §5.1.4 Table 15). For Llama 1 65B, Singh et al. report EPG of 18% on HumanEval and 25% on BIG-Bench Hard ([[evaluation-data-contamination-contam]] §5.2.2). Each report is a **Result (single study)**. Both show that effects differ by benchmark and model size, but Llama 3 follows the method of Singh et al., so their agreement is not an independent replication.

### §5.2 The procedure

1. Choose the evaluation sets and splits to protect. OLMo 3 matches "any split of any benchmark dataset" in its harness, because some benchmarks are evaluated on training splits ([[olmo-3-decontamination]] §3.5.3).
2. Normalize text. Brown et al.'s method lowercases and removes punctuation; the skip-budget methods do not normalize ([[evaluation-data-contamination-contam]] §3.1, Table 1).
3. Split into units: tokens of a stated tokenizer, words, or paragraphs.
4. Index evaluation text and scan training documents. Paloma uses a Bloom filter for paragraph exact match ([[paloma]] App. C.1.1). OLMo 3's decon samples training n-grams "at a regular stride" and expands around each hit ([[olmo-3-decontamination]] §3.5.3).
5. Score each pair of evaluation item and training text with a score function (§5.3).
6. Compare the score with a threshold.
7. Remove a unit: the document, the paragraph, the training instance, or the whole source dataset.
8. Record the decisions (§7).

### §5.3 Score functions

For an evaluation item e of |e| tokens and n-gram length n ([[evaluation-data-contamination-contam]] §3.1):

```
TOKEN-MATCH    s = |{ i : token e_i lies in an n-gram of e that occurs in the corpus }| / |e|
NGRAM-MATCH    s = |{ n-grams of e that occur in the corpus }| / (|e| − n + 1)
LONGEST-MATCH  s = ℓ / |e|,  ℓ = length of the longest span of e found in the corpus (extended from an n-gram seed)
contaminated   if s > τ
```

- τ: the threshold. Llama 3 selects a per-dataset token ratio T_D "based on which value shows the maximal significant estimated performance gain across the three model sizes" ([[llama-3-decontamination]] §5.1.4).
- mincount: the minimum number of corpus occurrences for an n-gram to count.
- skip budget: the number of substitution mismatches allowed while extending a match (TOKEN-EXTEND and LONGEST-MATCH).

OLMo 3's decon uses an IDF-weighted overlap ([[olmo-3-decontamination]] App. A.5):

```
O = Σ_{x ∈ U_t ∩ U_e} idf(x) / Σ_{y ∈ U_e} idf(y)
```

- U_t: unique n-grams in the training document segment; U_e: unique n-grams in the evaluation item; idf(x): inverse document frequency of n-gram x.
- Question, answer, and passage overlaps are combined with weights 0.7, 0.2, and 0.1 when all three exist, and short matches are scaled down (App. A.5). The n-gram length, sampling stride, and final threshold are not printed in the report.

EPG and its significance ([[evaluation-data-contamination-contam]] §4.2):

```
EPG = score(full benchmark) − score(clean subset)
z   = EPG / (σ / √N_clean)
```

- σ: standard deviation of per-item scores on the full benchmark; N_clean: number of items in the clean subset.

### §5.4 Worked examples

The figure [figures/ngram-contamination.html](figures/ngram-contamination.html) lets the reader paste an evaluation item and a training document, change n, normalization, and threshold, and see which tokens each score function covers. The numbers below use its word tokenizer (lowercase, punctuation replaced by spaces); model tokenizers give different counts.

1. **Verbatim copy (derived).** The GSM8K item "Janet’s ducks lay 16 eggs per day. ..." has 53 words after normalization. Pasted inside a worksheet page, every word lies in a shared 8-gram: TOKEN-MATCH = 53/53 = 1.0, flagged at any τ < 1.
2. **Paraphrase (derived).** The rephrased version from [[rephrased-samples-contamination]] (Example 1, "Janet’s ducks produce 16 eggs each day. ...") shares no 8-gram and no 13-gram with the original: TOKEN-MATCH = 0/53. The longest shared span is 4 words ("muffins for her friends"), shorter than an 8-word seed, so LONGEST-MATCH = 0. With n = 4, TOKEN-MATCH = 12/53 = 0.226. The paraphrase keeps the answer.
3. **Instance rule (arithmetic).** Tülu 3 treats a test prompt as overlapping a training instance "if more than 50% of the test tokens have 8-gram matches with the same training instance" ([[tulu-3]] arXiv:2411.15124v5 §3.2). A 40-token prompt with 22 covered tokens has 55% and overlaps; with 20 covered tokens it has 50% and does not. A training dataset counts as contaminated if its instances overlap "more than 2% of the instances" of an evaluation, which for a 500-item evaluation means more than 10 items.
4. **EPG significance (arithmetic).** A 1,000-item benchmark scores 60.0% overall; 300 items are flagged; the 700 clean items score 55.0%. EPG = 5.0 points. With σ = √(0.6 × 0.4) = 0.49 for 0/1 scores, the standard error is 0.49 / √700 = 1.85 points and z = 2.70.

### §5.5 False negatives and false positives

**False negatives** (contaminated items that are not flagged):
- Larger n. For Llama 1 65B on PiQA with NGRAM-MATCH, "only 33.8% of examples have nonzero contamination scores at n = 10 while 67.9% have nonzero contamination scores n = 8", and the additional examples at n = 8 carried EPG ([[evaluation-data-contamination-contam]] §6.1, Figure 6). "values of n larger than 8 lead to false negatives" almost across the board.
- mincount above 1 excluded examples "that do have a real increase in performance" (§6.3).
- Paraphrase and translation. On 200 prompt pairs per MMLU subject, 10-gram matching had F1 = 0 on rephrased English and rephrased Chinese items in all three subjects; on HumanEval it had F1 = 1 on the original test set and 0 on rephrased Python, C, and JavaScript ([[rephrased-samples-contamination]] §5.2, Tables 5-6). The rephrased items were regenerated until an n-gram check no longer matched them (Algorithm 1), so F1 = 0 partly follows from the construction. An LLM-based detector (embedding retrieval of the top-k training items, then a judgment by a model such as GPT-4) scored F1 from 0.940 to 1 on the MMLU pairs.
- Size of the undetected gain. Fine-tuning Llama 2 13B for 16 epochs on rephrased MMLU test questions raised MMLU from 54.8 to 89.9 (question-only rephrasing) and 85.9 (full prompt) ([[rephrased-samples-contamination]] Table 2). This is a deliberate fine-tuning setting, not pre-training contamination; it shows the gain that an n-gram check would not detect.

**False positives** (flagged items that do not carry an advantage):
- Short template strings in math word problems, such as "a mosaic with chips of" ([[evaluation-data-contamination-contam]] §4.2). Singh et al. therefore select thresholds per model-benchmark pair; the selected values range from 0.04 to 0.40 (Table 3).
- Multiple-choice option lists. Two different algebra questions with the options "A. True, True B. False, False C. True, False D. False, True" match on the options ([[rephrased-samples-contamination]] Example 2). OLMo 3 improved precision for multiple-choice evaluations by matching against full answers instead of A/B/C/D labels ([[olmo-3-decontamination]] §3.5.3).
- Lenient metrics "may detect pre-training data that is beneficial for a model because they facilicate generalisation" ([[evaluation-data-contamination-contam]] §8.1).

**Conditions and limits.** ConTAM results are post-hoc and "fundamentally correlational" (§8.3); they cover Llama 1 and Pythia models on 13 benchmarks. EPG cannot be estimated when almost all or almost none of a benchmark is flagged (§8.1); Llama 3 reports this for MBPP, HumanEval, MMLU, and MMLU-Pro ([[llama-3-decontamination]] §5.1.4). The rates of rephrased samples that the LLM-based detector reports for real datasets ([[rephrased-samples-contamination]] §5.3, Table 7) are counts of flagged items; their precision on those datasets is not reported. Tülu 3 found embedding matching hard to use because it could not "distinguish mere distributional similarity from actual paraphrasing" ([[tulu-3]] §3.2). No method in these sources reports both high recall on paraphrases and low cost at trillions of tokens (**Open question**).

### §5.6 Configurations in released reports

| Source | Stage | Match rule | Threshold | Removal unit | Reported cost |
|---|---|---|---|---|---|
| Paloma baselines ([[paloma]] App. C.1.1, Table 4); Dolma for OLMo-1B ([[dolma]] App. L) | pre-training, against Paloma perplexity data | paragraph exact match (Bloom filter); paragraphs under 13 Unicode tokens ignored; code sources not decontaminated | any matching paragraph | whole document | Paloma Table 4: Dolma 0.062%, RedPajama 0.099%, The Pile 2.753%, RefinedWeb 0.733%, C4 0.010%, mC4-en 0.002% of documents; Dolma App. L: ≤ 0.02% of documents for OLMo-1B |
| Llama 3 ([[llama-3-decontamination]] §5.1.4, §5.2) | pre-training: analysis only; post-training: removal | pre-training: 8-gram token ratio; post-training: exact prompt match | per-dataset T_D chosen by EPG (values not printed) | no pre-training removal described | not reported |
| Tülu 3 ([[tulu-3]] §3.2, Table 8) | SFT prompts | 8-gram; > 50% of test tokens matched by one training instance | dataset contaminated if > 2% of an evaluation's instances overlap | whole dataset (unseen suite); dataset or matching instances (development suite) | e.g., 11.3% of NuminaMath-TIR removed against MATH |
| OLMo 3 ([[olmo-3-decontamination]] §3.5.3, App. A.5) | mid-training and long-context data | stride-sampled n-grams, cluster expansion (a document leaves the active set after 11 misses), IDF-weighted question/answer/passage score | tuned "based on numerous qualitative review" (value not printed) | document | DROP: over 60,000 training examples removed |
| Nemotron-CC ([[nemotron-cc]] §6) | pre-training dataset release | none | none | none | "we did not decontaminate the dataset" |

## §6 Where contamination enters and which stage to decontaminate

**Sources of contamination.**
- Copies in code repositories and templated datasets. Six Promptsource datasets, including the Winograd Schema Challenge and SuperGLUE COPA, are 100% contained in Dolma, and many contaminated sets are found in the code subset ([[dolma]] App. L). OLMo 3 found "complete validation or test splits" inserted through templates, for example in Flan ([[olmo-3-decontamination]] §3.5.4).
- Near-duplicates across splits. 4.60% of C4 validation examples had a near-duplicate in the training split ([[deduplicating-training-data]] Table 2).
- Rephrased samples in pre-training and synthetic data. The LLM-based detector flagged rephrased versions of 18.9% of HumanEval problems in a 4G subset of The Stack, 8.53% in a 16G subset of RedPajama-Data-1T, and 12.8% in CodeAlpaca, a synthetic dataset generated with OpenAI's Davinci-003 ([[rephrased-samples-contamination]] Table 7). Synthetic data can reproduce benchmark content that its generator saw ([[ch-18]]).
- Benchmark content published before a model's data cutoff. On LeetCode problems tagged by release date, DeepSeek-Instruct 33B dropped after August 2023, and "DS-Base-33B ... dropping from Pass@1 ∼ 60 in May problems to Pass@1 ∼ 0 in September LeetCode problems"; the drop "primarily occurs for the LeetCode problems only" ([[livecodebench]] §5.1). The authors interpret the drop as contamination (**Result (single study)** for the drop; **Interpretation** for the cause). Time-segmented evaluation is taught in [[ch-47a]].

**Which stage to decontaminate.**
- OLMo 3 concentrates decontamination on mid-training and the long-context extension "in light of results suggesting that memorization occurs most strongly near the end of training" ([[olmo-3-decontamination]] §3.5.3) (**Interpretation**, citing other work).
- Llama 3 excludes benchmark training sets from annealing data "to assess the true few-shot learning capabilities and out-of-domain generalization". In an experiment with GSM8k and MATH training sets in annealing, the pre-trained 8B model's validation scores rose by 24.0% and 6.4%, with negligible change at 405B ([[llama-3-decontamination]] §3.1.3). In this report, late-stage in-domain data changed small-model scores the most (**Result (single study)**). Annealing is covered in [[ch-32]].
- In ConTAM, larger Llama 1 models gained more from contaminated examples for around half of the model-benchmark pairs; for TriviaQA, HellaSwag, COPA, and PiQA the smaller Llama 1 models gained more, which the authors attribute to the larger models' high scores without contamination ([[evaluation-data-contamination-contam]] §5.2.3) (**Interpretation**).

**Side effects of removal.** Removing all splits of DROP removed over 60,000 training examples, and the OLMo 3 authors state that score changes "may indicate that decontamination is preventing memorization or also removing in-distribution training examples". OLMo 3 also detected complete GSM8K leakage while the decontaminated data scored higher, which the Marin authors attribute to contaminated formatting that did not match the evaluated format ([[olmo-3-decontamination]] §3.5.4). A score change after decontamination is therefore not by itself a measure of contamination.

## §7 Recording decontamination decisions at data-pipeline time

**Definition.** A *decontamination record* is a versioned description of what was checked, how it was checked, and what was removed, stored with the data mixture.

**The problem it addresses.** The score-effect estimates of [[ch-48]] need the contaminated and clean partitions of each benchmark, the thresholds, and the match rule. ConTAM recommends to "Report % contaminated, EPG and the selected thresholds" ([[evaluation-data-contamination-contam]] §7.3). Procedures change: OLMo 3's first version failed on SQuAD v2 and DROP, and "The decon repository includes configuration files that reproduce both the earlier and final approaches" ([[olmo-3-decontamination]] §3.5.3). A dataset released without decontamination ([[nemotron-cc]] §6) passes the decision to every downstream user.

**Mechanism.** The fields below are this course's proposal (**Interpretation**). Each field corresponds to a decision that §5 shows can change the result. Angle-bracket values are placeholders.

```yaml
decontamination_record:
  mixture_version: <hash of the document manifest>
  stages: [pretrain-stable, anneal, long-context]
  eval_suite:
    - {name: GSM8K, version: "<dataset@commit>", splits: [train, test]}
    - {name: MMLU,  version: "<dataset@commit>", splits: [dev, validation, test]}
  normalization: {lowercase: <bool>, strip_punctuation: <bool>}
  unit: {tokenizer: "<name@version>", n: <int>, mincount: <int>}
  score: <TOKEN-MATCH | NGRAM-MATCH | LONGEST-MATCH | paragraph-exact | IDF-overlap>
  threshold: {default: <τ>, per_benchmark: {GSM8K: <τ_GSM8K>}}
  fields_matched: [question, answer_text]     # not option labels
  removal_unit: <document | paragraph | instance | dataset>
  removed: {items_matched_per_benchmark: {GSM8K: <count>}, documents_removed: <count>, tokens_removed: <count>}
  removed_ids: <path to removed document ids>
  known_blind_spots: [paraphrase, translation, code]
  tool: {name: <tool>, config: "<path@commit>"}
  date: <YYYY-MM-DD>
```

**Worked example (arithmetic, hypothetical counts).** Suppose a benchmark's test split has 1,000 items and the record lists 120 of them as matching removed documents. A later analysis can split the test set into 120 matched and 880 unmatched items and compute EPG on a checkpoint trained before removal, as in §5.4 example 4. If the 880 unmatched items score 52.0% and the full set scores 55.0%, EPG = 3.0 points; with σ = √(0.55 × 0.45) = 0.497, the standard error is 0.497 / √880 = 1.68 points and z = 1.79. Without `removal_unit` and `threshold`, the same analysis cannot tell whether an unmatched item was below the threshold or was never checked.

**Evidence.** No source measures the effect of keeping such a record. The fields are taken from reported practice: per-benchmark thresholds and reporting ([[evaluation-data-contamination-contam]] §7.3), released decontaminated dataset versions ([[tulu-3]] Table 8), versioned configurations ([[olmo-3-decontamination]] §3.5.3), and removal rates per corpus ([[paloma]] Table 4).

**Implication for a general-purpose model.** Claims of generality rest on held-out evaluations. The record is what allows a later reader to decide whether a benchmark result is evidence about unseen items.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| datablations GPT-2-architecture models ([[data-constrained-scaling]]) | 8.7B | pretrain-stable | total tokens; unique tokens; passes | 178B total; 44B unique; four epochs | arXiv:2305.16264v5 §6, Figure 4 | verified 2026-09-15 | final validation loss 0.5% above the 178B-unique one-pass run (§6) |
| same | 4.2B | pretrain-stable | Python share of 84B tokens | up to 50% (42B tokens) | arXiv:2305.16264v5 §7, Figure 6 | verified 2026-09-15 | no deterioration of the 19-task average, 5 seeds (§7) |
| same | 4.2B | pretrain-stable | perplexity filter under data constraint | keep 25% of samples with lowest perplexity (44B tokens, close to 2 epochs) | arXiv:2305.16264v5 §7, App. N | verified 2026-09-15 | effective on C4; 100-character deduplication filter (21B tokens, 4 epochs) did not help (§7) |
| same | all study runs | pretrain-stable | dropout; weight decay; gradient clip | 0.1; 0.1; 1.0 | arXiv:2305.16264v5 App. S | verified 2026-09-15 | "largely based on prior work ... and performance on test runs" (App. S) |
| Llama 3.1 405B ([[llama-3-recipe]]) | 405B | pretrain-stable | tokens; tokens per parameter | 15.6T; 38.5 | v3 §1, §3.2; ratio derived: 15.6T / 405B | verified 2026-09-14 (tokens); derived (ratio) | scaling-law extrapolation to 402B on 16.55T tokens (§3.2.1) |
| Llama 3.1 (all) | all | pretrain-decay/anneal | benchmark training sets in annealing data | excluded | v3 §3.1.3 | verified 2026-09-15 | experiment: annealing with GSM8k/MATH training sets gave +24.0% / +6.4% on validation at 8B, negligible at 405B |
| Llama 3.1 8B / 70B / 405B | all | eval-gate | pre-training contamination analysis | 8-gram token-overlap ratio T_D per dataset, chosen for maximal significant EPG across three sizes; T_D values not printed; no removal described | v3 §5.1.4, Table 15 | verified 2026-09-15 | Table 15, e.g., HellaSwag 85% contaminated, +14.8 at 8B |
| Llama 3.1 (all) | all | SFT | post-training decontamination | exact match with benchmark prompts | v3 §5.2 | verified 2026-09-15 | no ablation reported |
| Nemotron-CC 8B, 15T-token run ([[nemotron-cc]]) | 8B | pretrain-stable | tokens; Nemotron-CC share by phase | phase 1: 9T tokens, 59% English Common Crawl (5.31T); phase 2: 6T tokens, 31% (1.86T); total 7.17T | arXiv:2412.02595v2 App. E | verified 2026-09-15 | Table 6: MMLU 70.3 vs 65.3 for Llama 3.1 8B in the authors' harness |
| Nemotron-CC 8B ablation | 8B | pretrain-stable | high-quality repetitions replaced by synthetic data | HQ-Base: eightfold high-quality documents; HQ-Synthetic: 4 of the 8 repetitions replaced by synthetic data; 1T tokens, 73% Common Crawl variant | arXiv:2412.02595v2 §3.1, §3.3, Table 10 | verified 2026-09-15 | 10-task average 55.8 → 56.7; seeds not reported |
| Nemotron-CC dataset | n/a | data generation | rephrasing model and sampling | Mistral NeMo 12B Instruct, FP8, top-p 0.9, temperature 0.5; over 1.8T synthetic tokens | arXiv:2412.02595v2 §2.3, Table 3 | verified 2026-09-15 | no ablation reported for sampling settings |
| Nemotron-CC dataset | n/a | data release | decontamination | not performed | arXiv:2412.02595v2 §6 | verified 2026-09-15 | authors cite lack of consensus and uncertain impact |
| Olmo 3 Base mid-training mix ([[olmo-3-decontamination]]) | not scoped by size | mid-train; long-context | decontamination | decon: stride-sampled n-grams, cluster expansion (document leaves active set after 11 misses), IDF-weighted question/answer/passage score (weights 0.7/0.2/0.1); all splits of all OLMES benchmarks; document removed | arXiv:2512.13961v2 §3.5.3, App. A.5 | verified 2026-09-15; n-gram length, stride, and threshold not reported in the report (repository config not checked) | matched 100B anneals with and without decontamination (§3.5.4, Figure 12); threshold tuned by qualitative review |
| Tülu 3 SFT mix ([[tulu-3]]) | 8B, 70B, 405B | SFT | prompt decontamination | 8-gram; > 50% of test tokens matched by one training instance; dataset contaminated if > 2% of an evaluation's instances overlap | arXiv:2411.15124v5 §3.2, Table 8 | verified 2026-09-15 | n-gram matching chosen over full-string and embedding matching (qualitative); Table 8 removal shares |
| Paloma baseline models ([[paloma]]) | 1B | pretrain-stable | decontamination against Paloma | paragraph exact match; ignore paragraphs < 13 Unicode tokens; no decontamination of code sources; remove whole document | arXiv:2312.10523v2 §3, App. C.1.1, Table 4 | verified 2026-09-15 | 13 follows the n-gram size of Brown et al. and Rae et al. (App. C.1.1); no ablation reported |

**Starting point for a small general-purpose run.** For a dense decoder of at most about 9B parameters pre-trained on filtered web text, where unique tokens after global deduplication are fewer than planned tokens: plan up to four passes over the whole pool (row 1: 8.7B, C4, GPT-2 architecture, with the regularization of row 4). If more tokens are needed, add Python code up to 50% of tokens before adding passes (row 2: 4.2B, 84B tokens, natural-language average only), and consider replacing half of the repetitions of high-quality documents with rephrased versions (row 10: 8B, 1T tokens, single run). For decontamination, use 8-gram token matching (rows 7 and 14), match every split used in evaluation and remove matching documents from mid-training and annealing data (row 13), exclude benchmark training sets from annealing data (row 6), and run a per-benchmark threshold analysis of the kind in row 7 on the final checkpoint. These rows come from different organizations and scales, and no source tested the combination.

## Generalization lens

**(a) What increases breadth.**
- Keeping repetition at or below about 4 passes kept the 19-task average unchanged at 4.2B ([[data-constrained-scaling]] §7, Figure 6).
- Python code up to 50% of tokens kept the natural-language average and raised bAbI and WebNLG at 4.2B ([[data-constrained-scaling]] §7).
- Rephrased high-quality documents replacing repetitions raised the 10-task average at 8B ([[nemotron-cc]] Table 10). In WRAP, no single rephrasing style was best across Pile domains, and an oracle choice of style per domain "will improve perplexity by 16%" (128M, 3B tokens; [[rephrasing-the-web]] §6.2) (**Result (single study)**), which supports mixing several styles.
- Decontaminated mid-training data makes comparisons between mixture rounds measure ability rather than leakage; OLMo 3 notes that the gains of its last round "are likely underestimated" because only that round used decontaminated data ([[olmo-3-decontamination]] §3.5.4).

**(b) What causes narrowing or forgetting.**
- Repeating a small subset at the most damaging repetition count reduced copying and induction-head measures more than test loss, and raised loss on out-of-distribution Python at high repeated fractions ([[repeated-data-scaling]] §1.1).
- Whole-dataset repetition for 44 epochs made loss increase midway through training ([[data-constrained-scaling]] §6) and lowered SuperGLUE from 71.36 to 64.76 at 1,024 repeats ([[c4]] Table 9). **Replicated** across two settings.
- Rephrasing low-quality documents lowered MMLU from 48.2 to 47.1 while raising the average ([[nemotron-cc]] Table 10); the authors mention "potential misinformation introduced by data synthesis".
- Contamination produces benchmark gains without matching gains on unseen items: fine-tuning on rephrased test questions raised Llama 2 13B MMLU from 54.8 to 85.9-89.9 ([[rephrased-samples-contamination]] Table 2), and release-date splits show drops after a model's cutoff ([[livecodebench]] §5.1).
- Decontamination can remove in-distribution training examples, as with over 60,000 DROP training examples ([[olmo-3-decontamination]] §3.5.4), so a benchmark family can lose training signal together with leakage (**Interpretation** by the authors).

**(c) How to measure it for this stage.**
- Held-out loss on a split that is never repeated; Muennighoff et al. do not use training loss because models overfit repeated data ([[data-constrained-scaling]] §4, App. H).
- Per-source training loss next to held-out loss; the harmful range starts where training loss on the repeated subset approaches zero ([[repeated-data-scaling]] §1.1).
- Per-task downstream tables, not only averages ([[nemotron-cc]] Table 10), with several seeds where possible ([[data-constrained-scaling]] §7 used 5).
- Contamination: percent contaminated, EPG, z-scores, and thresholds per benchmark and model size ([[evaluation-data-contamination-contam]] §7.3); a paraphrase-aware check for key benchmarks ([[rephrased-samples-contamination]] §4); time-segmented evaluation after the data cutoff ([[livecodebench]] §5.1). Known measurement errors: n-gram checks miss paraphrases, EPG cannot be estimated when almost all or almost none of a benchmark is flagged ([[evaluation-data-contamination-contam]] §8.1), and post-hoc estimates are correlational (§8.3).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Using a saturating formula such as U(1 − exp(−R/R*)) for effective tokens | Effective tokens below unique tokens even at one pass | Eq. 5 must give D′ = U at R_D = 0 and approach 16.39 U ([[data-constrained-scaling]] §3.1) |
| Counting passes for the whole corpus only | Aggregate passes near 1 while one upsampled source has tens of passes; that source's training loss falls toward zero | Passes per source = sampled tokens / unique tokens; log per-source training loss ([[repeated-data-scaling]] §1.1) |
| Counting unique tokens before global deduplication | Planned passes appear low; late-training held-out loss is above the one-pass prediction | Recount after global fuzzy deduplication (DCLM 3.8T total, 1.0T unique, [[nemotron-cc]] Table 4) |
| Judging repetition by training loss | Training loss keeps falling while held-out loss flattens or rises | Evaluate on a never-repeated held-out split ([[data-constrained-scaling]] App. H) |
| Transferring the 4-pass result to fine-tuning or different regularization | Overfitting after few passes in SFT | The result covers pre-training with dropout 0.1 on C4 ([[data-constrained-scaling]] App. Q, App. S); run a held-out check ([[ch-30]]) |
| Using n = 13 or mincount > 1 in a contamination analysis | Low percent flagged and EPG near zero on benchmarks known to be on the web | Rerun with n = 8, mincount 1, and compare EPG ([[evaluation-data-contamination-contam]] §6.1, §6.3) |
| One global threshold for all benchmarks | Flagged math items whose EPG is zero or negative | Per-benchmark threshold by z-score ([[evaluation-data-contamination-contam]] §4.2) |
| Matching multiple-choice items on option labels | Different questions flagged because they share "A. True, True ..." | Match question stems and full answer text ([[olmo-3-decontamination]] §3.5.3) |
| Decontaminating only test splits | Scores on validation or training splits used for development are higher than on fresh items | Remove matches for every split used in evaluation ([[olmo-3-decontamination]] §3.5.3) |
| Treating "no n-gram match" as "clean" | Gain on a static benchmark together with a drop on items released after the cutoff | Paraphrase-aware check ([[rephrased-samples-contamination]]); release-date split ([[livecodebench]]) |
| Reading a post-decontamination score drop as proof of contamination | Full GSM8K leakage detected, but the decontaminated run scores higher | Compare matched runs and inspect formats ([[olmo-3-decontamination]] §3.5.4) |
| No decontamination record | EPG cannot be computed later; the procedure cannot be reproduced | Store evaluation versions, match rule, thresholds, removal unit, removed ids (§7) |

## Check your understanding

1. Eq. 5 gives D′ = U at one pass and a ceiling of 16.39 U. Explain from the geometric-series derivation why the ceiling exists, and why R*_N < R*_D leads the data-constrained frontier to add epochs faster than parameters.
2. A mixture draws 2% of 100B training tokens from a 200M-token source while the corpus as a whole makes 1.1 passes. Why can this be worse for in-context copying than 4 uniform passes over everything, and which logged quantity would show the problem during training?
3. HQ-Synthetic beat HQ-Base by 0.9 average points at 8B. State the two causes the authors propose, and design an ablation that separates "new unique tokens" from "new styles".
4. The §4.1 language-modeling fit gives γ/β close to 1. Explain why the authors' "sublinear" reading depends on comparing γ with 1 rather than γ/β with 1, and what each comparison says about corrupted tokens.
5. Why does increasing n reduce false positives but increase false negatives, and why could ConTAM judge n = 8 better than n = 10 or 13 using EPG instead of a labeled set of contaminated items?
6. In Llama 3's analysis, 52% of NaturalQuestions was flagged with almost no estimated gain, while 85% of HellaSwag was flagged with +14.8 at 8B. Propose mechanisms that could produce each result and an analysis that would distinguish them.
7. OLMo 3 detected full GSM8K leakage, but the decontaminated mixture scored higher. Explain how this can happen and what it implies for using score change as the only test of contamination.
8. A colleague reports "decontaminated with 13-gram overlap" and nothing else. Which fields from §7 are missing for computing EPG in ch-48, and which conclusions about generality cannot be drawn without them?

## Connections

- **Previous:** ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (dependency; per-language unique-token limits are the multilingual case of §1-§2).
- **Next:** ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (token budgets, passes, and stage mixtures of released recipes).
- ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability (the one-pass laws that §2 generalizes).
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (filters that reduce unique tokens; selection that resembles test tasks).
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic (counting unique tokens; cross-split overlap).
- ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining (memorization mechanisms behind repetition effects).
- ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale (upsampling as non-uniform repetition).
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (rephrasing methods).
- ch-23 — Model Collapse and Verification of Synthetic Data (risks of training on generated text).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (annealing data and late-stage decontamination).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (synthetic expansion of small domains).
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation (release-date and perturbed evaluation).
- ch-48 — Contamination Detection and Its Effect on Reported Scores (uses the record of §7 to estimate score effects).

## Sources

- [[data-constrained-scaling]] — Eq. 5-6, fitted constants, 4-epoch and 16-epoch results, allocation, code filling and filtering, hyperparameters, limitations (excerpt in `excerpts/`).
- [[repeated-data-scaling]] — subset repetition, double descent, copying and induction-head damage, training-loss diagnostic, limitations.
- [[scaling-laws-data-quality]] — quality-aware law, Table 2 constants, synthetic-noise setting.
- [[nemotron-cc]] — unique versus total tokens of filtered corpora, rephrasing pipeline and Table 3, Table 10 repetition-versus-synthetic ablation, 15T run, no decontamination.
- [[epoch-data-stock]] — stock of public text, quality and repetition adjustments, overtraining ratios.
- [[rephrasing-the-web]] — WRAP speedup, rephrasing versus augmentation, style diversity, limits on new knowledge (claims read in arXiv:2401.16380v1).
- [[d4]] — repetition of a selected subset at 1.3B.
- [[c4]] — T5 repetition results (Table 9).
- [[llama-3]], [[llama-3-recipe]] — token budget and contamination analysis summary.
- [[llama-3-decontamination]] — annealing exclusion, 8-gram contamination analysis and Table 15, post-training exact match.
- [[evaluation-data-contamination-contam]] — score functions, EPG and z-score, effects of n and mincount, per-benchmark thresholds (Table 3), recommendations, limitations.
- [[rephrased-samples-contamination]] — paraphrase and translation evasion, construction of rephrased sets, detection F1, rephrased samples in real datasets.
- [[olmo-3-decontamination]] — decon method, stage placement, templated contamination, matched anneals, side effects.
- [[tulu-3]] — 8-gram instance rule and dataset-level removal for SFT prompts (arXiv:2411.15124v5 §3.2).
- [[paloma]], [[dolma]] — paragraph-level decontamination, removal rates, benchmark copies in Dolma.
- [[deduplicating-training-data]] — near-duplicate overlap between C4 training and validation splits.
- [[livecodebench]] — performance drops on problems released after a model's cutoff.
