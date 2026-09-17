<!-- chapter: ch-10a
     track: pretraining
     kind: content
     title: Model-Based Quality Filtering and Benchmark-Targeted Data Selection
     deps: [ch-10]
     sources: [[fineweb]], [[dclm]], [[nemotron-cc]], [[qurating]], [[weborganizer]], [[perplexity-correlations-data-selection]], [[dsdm]], [[datadecide]], [[datadecide-recipe]], [[training-on-the-test-task]], [[whose-language-counts-quality-filter]], [[smollm2]], [[llama-3]], [[paloma]], [[data-constrained-scaling]], [[signal-and-noise-eval]], [[openhermes-2-5]], [[phi-textbooks]], [[c4]]
     figures: figures/selection-tradeoffs.html
     revised: 2026-09 (generality revision)
-->

# Chapter 10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection

> **Core insight.** A model-based quality filter keeps the documents that a learned scorer rates highest. In the studies below, the benchmarks that improve most are those that resemble the scorer's positive examples or targets (this course's Interpretation). In 1B models trained on 29B tokens selected from the same 200B-token pool, the FineWeb-Edu filter raised ARC-Easy by 9.7 points and MMLU by 4.0, lowered HellaSwag by 1.5, PIQA by 1.4, and SIQA by 2.0, and raised perplexity on held-out text from the unfiltered pool from 12.1 to 14.7 ([[weborganizer]] Table 1, Table 10). Spreading the selection over more of the pool recovered some or all of the lowered tasks in two studies: sampling by educational-value rating with temperature τ = 2 improved all 10 tasks over uniform selection at 1.3B ([[qurating]] §5.3), and per-domain token quotas raised FineWeb-Edu's HellaSwag from 56.0 to 62.5 and PIQA from 69.9 to 73.3, although SIQA stayed below the random baseline (49.4 vs 49.9) and the quotas came from a mixture tuned toward MMLU and HellaSwag ([[weborganizer]] Table 1, Table 10). Part of any benchmark gain from targeted data measures resemblance to the test task: after all base models were fine-tuned on the same task data, the 7-point MMLU and 19-point GSM8K advantage of newer models at equal pretraining compute was no longer significant ([[training-on-the-test-task]] §2.2, Fig. 1).
>
> **Guideline.** When a quality classifier is chosen or tuned, evaluate it on tasks that resemble its positive examples, on tasks that do not, and on per-domain perplexity of the unfiltered pool, because the average can rise while unrelated tasks fall (FineWeb-Edu at 1B: average +2.6, SIQA −2.0; [[weborganizer]] Table 1). When the training horizon needs more unique tokens than a top-10% filter leaves, use several classifiers with graded quality buckets and rephrase low-scored text instead of discarding it, because Nemotron-CC's three-classifier ensemble labeled 25% of tokens high quality against 8% and 11% for FineWeb-Edu's and DCLM's classifiers at an equal or higher 10-task average ([[nemotron-cc]] Table 9). When selection is optimized for named benchmarks (benchmark-correlation selection, datamodels, mixture regression), report the tasks that were not targets and a uniform-sampling baseline, because targeting one task lowered unrelated tasks in [[dsdm]] (Fig. 4) and [[weborganizer]] (Table 10). Otherwise, prefer temperature sampling or domain quotas over a hard top-k threshold.

## Why this chapter matters for a general-purpose model

The pipeline for a general-purpose model is pre-training → mid-training → SFT → preference optimization → RL → evaluation. This chapter is inside pre-training data curation. ch-10 covered the heuristic stages (text extraction, language identification, rule-based filters). After those stages and deduplication, the remaining pool is still larger than the token budget, and a choice must be made about which documents to train on.

**Model-based quality filtering** is the selection of pretraining documents by a score produced by a learned model: a classifier trained on positive and negative example documents, a regression head trained on LLM ratings, a model of how documents affect benchmark loss, or a correlation between document loss and benchmark accuracy. The measurable problem is that the scorer defines "quality" implicitly. The selected distribution then determines which tasks improve and which topics, formats, and language varieties the model sees less often.

Three measured effects make this a generality question. First, filters change task profiles, not only averages (§2, §3). Second, filters that keep about 10% of documents leave 0.2T-1.0T unique tokens, while a 15T-token run drew 7.17T tokens from web data (§4). Third, selection that is tuned on benchmark scores can raise those benchmarks through resemblance to the test task (§6). ch-09 defines capability coverage for pretraining data, ch-12 covers deduplication, ch-13 covers domain mixing, and ch-14 covers decontamination. ch-17 is the lab that applies this chapter's measurements.

## §1 The parts of a model-based selection method

**Definition.** A selection method has four parts: (1) a target that defines quality (positive examples, an LLM prompt, a benchmark, or a set of target tasks); (2) a scoring model; (3) a selection rule; (4) the evaluation used to choose (1)-(3).

**Problem.** The same pool and budget give different models depending on these parts. In [[dclm]] Table 4 (1B-1x scale: 1.4B parameters, 28.8B tokens), seven scorers applied to the same RefinedWeb reproduction (DCLM's reimplementation of RefinedWeb's heuristic filters, §4.1) gave CORE scores from 26.1 (top 20% by PageRank) to 30.2 (fastText trained on OpenHermes-2.5 and ELI5 positives), against 27.5 without model-based filtering.

**Mechanism.**
1. Build the pool with heuristic cleaning and deduplication (ch-10, ch-12).
2. Score every document with the scoring model.
3. Apply the selection rule. A threshold rule keeps a document when its score is at least t. A percentile rule sets t so that a fixed share of documents is kept. A sampling rule draws documents with probability that increases with the score (§5.1).
4. Train on the selected set and evaluate.

**Formulas.**

```
keep(d) = 1[ score(d) ≥ t ]                    (threshold or percentile rule)
retention r = tokens kept / tokens in pool
passes = tokens needed by the training run / unique tokens available
```

Here d is a document, t the threshold, and r the retained share. "Passes" is the average number of times each unique token is seen.

**Worked example (derived from [[dclm]] Fig. 4).** Fig. 4 reports shares of the original DCLM-Pool documents: heuristic cleaning keeps 19.9%, Bloom-filter deduplication (ch-12) removes 6.2% (13.7% remain), and the fastText filter removes 12.3% (1.4% remain). The fastText stage therefore keeps 1.4 / 13.7 = 10.2% of the documents it sees, which matches the top-10% rule in §4.4. For FineWeb-Edu, 1.3T of FineWeb's 15T tokens are kept, r = 8.7% ([[fineweb]], derived).

**Implication for a general-purpose model.** Each part of the method is a place where a narrow target can enter. The remaining sections evaluate each part with four measurements: tasks that resemble the positives, tasks that do not, per-domain perplexity on the unfiltered pool, and unique tokens left.

## §2 FineWeb-Edu: an LLM-annotated educational classifier and its threshold

**Definition.** FineWeb-Edu is the 1.3T-token subset of FineWeb whose documents score at least 3 on a 0-5 educational-value classifier ([[fineweb]] §4).

**Mechanism** ([[fineweb]] §4, App. F.1).
1. Llama-3-70B-Instruct scores 460,000 pages from CC-MAIN-2024-10 with an additive 0-5 prompt. The prompt asks whether the page "could be useful in an educational setting for teaching from primary school to grade school levels", and the authors restricted it to grade-school and middle-school knowledge so that highly technical pages such as arXiv abstracts are not favored.
2. A linear regression head on the frozen Snowflake-arctic-embed-m encoder is trained on 410,000 annotations for 20 epochs at learning rate 3e-4; the checkpoint with the best F1 on the other 50,000 is kept.
3. Predictions are rounded to integers 0-5. Threshold 3 gives binary F1 82% on the validation set.
4. Scoring all 15T tokens took 6,000 H100 GPU hours.

**Evidence.** With a 1.71B model trained on 350B tokens, FineWeb-Edu raised MMLU from 33% to 37% and ARC from 46% to 57% compared with FineWeb (§4). The threshold was chosen from FW-Edu-2, -3, and -4 at 28B tokens; 3 gave the best aggregate and "the best trade-off between performance on knowledge and reasoning intensive benchmarks and the performance on other benchmarks like HellaSwag" (§4, App. F.2). The paper reports the HellaSwag cost only in a figure.

An independent comparison on one pool gives per-task numbers. [[weborganizer]] trained 1B models (DCLM 1b-1x setting, about 29B tokens) on data selected from a 200B-token deduplicated Common Crawl pool (Table 1, Table 10; deltas derived):

| Task | Random baseline | FineWeb-Edu filter | Δ | DCLM fastText filter | Δ |
|---|---|---|---|---|---|
| MMLU | 30.3 | 34.3 | +4.0 | 33.4 | +3.1 |
| HellaSwag | 57.5 | 56.0 | −1.5 | 59.0 | +1.5 |
| PIQA | 71.3 | 69.9 | −1.4 | 70.5 | −0.8 |
| WinoGrande | 56.1 | 57.7 | +1.6 | 58.8 | +2.7 |
| CommonsenseQA | 59.0 | 60.0 | +1.0 | 63.2 | +4.2 |
| SIQA | 49.9 | 47.9 | −2.0 | 50.7 | +0.8 |
| ARC-Easy | 62.2 | 71.9 | +9.7 | 71.4 | +9.2 |
| ARC-Challenge | 34.0 | 42.3 | +8.3 | 39.8 | +5.8 |
| OpenBookQA | 44.0 | 48.2 | +4.2 | 48.8 | +4.8 |
| 9-task average | 51.6 | 54.2 | +2.6 | 55.1 | +3.5 |
| Held-out perplexity (lower is better) | 12.1 | 14.7 | +2.6 | 14.0 | +1.9 |

The interactive figure [figures/selection-tradeoffs.html](figures/selection-tradeoffs.html) plots these per-task deltas, and the same view for QuRating and DsDm, so the reader can see which tasks each selection method lowers.

[[smollm2]] (Table 1, 1.7B models, 350B tokens each) reports the same split at a larger token count: FineWeb-Edu is higher on MMLU (37.5 vs 35.5), ARC (57.5 vs 53.5), and OpenBookQA (41.9 vs 40.8); DCLM is higher on HellaSwag (62.3 vs 60.1) and CommonsenseQA (40.1 vs 36.2). This pattern is **Replicated** across [[weborganizer]] and [[smollm2]] for HellaSwag and CommonsenseQA.

The FineWeb paper documents the distribution shift itself ([[fineweb]] §4.1-4.2). In topic clusters, FineWeb-Edu gains "Education, Learning, Teaching" (+3.2%) and "History, Culture, Politics" (+2.2%) and down-samples "Business, Finance, Law", "Entertainment, Film, Theater", and "Places, Travel, Real Estate". In Paloma per-domain perplexity ([[paloma]], a perplexity benchmark over 546 English and code domains), the model trained on FineWeb fits broad web sources, Twitter AAE, and 100 Subreddits better, and the FineWeb-Edu model fits Wikipedia, arXiv, and S2ORC text better.

**Conditions and limits.** The FineWeb ablations use one model size (1.71B) and academic benchmarks without instruction tuning ([[fineweb]] §6). ARC contains grade 3-9 science exam questions ([[dclm]] App. G.1), and the annotation prompt asks for grade-school educational content. This course reads the ARC gains (+11 points at 1.71B, +9.7 ARC-Easy at 1B) as partly a result of that resemblance (**Interpretation**); no source isolates it.

**Implication.** FineWeb-Edu is a knowledge-and-reasoning filter, not a general quality filter. When it is used alone, commonsense, social, and informal-text abilities need their own measurement.

## §3 DCLM: fastText with instruction-shaped positives, and CORE versus EXTENDED

**Definition.** DCLM-Baseline is the 3.8T-token dataset produced by keeping the top 10% of a deduplicated RefinedWeb-style pool by the score of a fastText classifier ([[dclm]] §4.4, Fig. 4). fastText is a linear text classifier over word n-gram features; DCLM uses unigram and bigram features (App. J.1).

**Problem.** Classical quality classifiers use reference corpora such as Wikipedia or books as positives. The question DCLM tests is which positive set gives the best pretraining data.

**Mechanism** ([[dclm]] App. J.1).
1. Training set: 400K documents, 200K positive and 200K negative. Negatives are random documents from an earlier RefinedWeb reproduction.
2. Positives: 100K OpenHermes 2.5 examples (a compilation of about 1M mostly synthetic instruction and chat samples, [[openhermes-2-5]]) and ELI5 questions paired with their top answer, kept when the post score is ≥ 0, the best comment score is ≥ 5, and there are at least 3 comments.
3. Features: fastText defaults except `wordNgrams = 2`.
4. Score: predicted probability of the positive label; keep the top 10% of documents.

**Formula: centered accuracy.** DCLM reports CORE (22 low-variance tasks) and EXTENDED (all 53 tasks, adding for example GSM8K, GPQA, SVAMP, BBQ, and AGI Eval) as centered accuracy, "linearly rescaling the accuracy per task so that 0 corresponds to random guessing and 1 corresponds to perfect accuracy" (§3.5):

```
centered_acc = (acc − r) / (1 − r)
CORE = mean over 22 tasks of centered_acc;   EXTENDED = mean over 53 tasks
```

Here acc is raw accuracy and r is the chance accuracy of the task.

**Worked example.** A 4-way multiple-choice task has r = 0.25; accuracy 0.40 gives (0.40 − 0.25)/0.75 = 0.20. A binary task has r = 0.5; accuracy 0.60 gives (0.60 − 0.5)/0.5 = 0.20. Both contribute the same amount to the average, although the raw accuracies differ by 20 points.

**Evidence.** At 7B-1x (6.9B parameters, 138B tokens), the positive set changed CORE by up to 6.3 points ([[dclm]] Table 5):

| Positives | Kept | CORE | MMLU | EXTENDED |
|---|---|---|---|---|
| OH-2.5 + ELI5 | top 10% | 41.0 | 29.2 | 21.4 |
| Wikipedia | top 10% | 35.7 | 27.0 | 19.1 |
| OpenWebText2 | top 10% | 34.7 | 25.0 | 18.7 |
| GPT-3 approximation | top 10% | 37.5 | 24.4 | 20.0 |
| OH-2.5 + ELI5 | top 15% | 39.8 | 27.2 | 21.5 |
| OH-2.5 + ELI5 | top 20% | 38.7 | 24.2 | 20.3 |
| OH-2.5 + ELI5, unigrams only | top 10% | 40.0 | 28.3 | 22.1 |

The two suites do not agree on every choice. CORE prefers 10% over 15% and bigrams over unigrams; EXTENDED is 0.1 higher at 15% and 0.7 higher with unigrams (Table 5, Table 14). These tables report no seed variance, and the authors state that they "could not sufficiently explore run-to-run variation" (§6), so these sub-point EXTENDED differences may be noise (**Open question**).

Other DCLM results bear on generality:
- Human agreement does not predict data value. ROC-AUC is the area under the curve of true-positive rate against false-positive rate over all thresholds (0.5 is chance). On about 500 human-labeled documents, AskLLM (prompting an instruction-tuned LLM to judge usefulness) agreed with the majority label at about 82% ROC-AUC and fastText filters at about 73%, but AskLLM-filtered data gave about 28.5 CORE against more than 31 for several fastText filters (App. N).
- After this filter, adding curated sources lowered the scores. Mixing in 33% RedPajama extras (Wikipedia, Books, Stack Exchange, arXiv, GitHub) at the Llama/RedPajama ratios raised CORE for C4 (+2.2) and RefinedWeb (+1.4) but lowered it for DCLM-Baseline (31.1 → 29.9) (Table 6).
- Code and math are weak. The 7B model trained for 2.5T tokens scores 2.1 on GSM8K in Table 29 (the App. Q.1 text gives 2.5%); instruction tuning raises it to 52.5 while CORE falls from 56.0 to 55.0 (Table 29). The authors state the models "do not perform as well on code and math" (§6).
- At 7B with 0.28T tokens, DCLM-Baseline scored CORE 48.9, MMLU 50.8, and EXTENDED 31.8, against 41.9, 37.3, and 24.5 for FineWeb-Edu (Table 8).

**Conditions and limits.** Table 5 runs used an earlier learning rate of 3e-4 and weight decay 0.33, not the final 7B settings (App. F). CORE and EXTENDED contain no long-context, tool-use, instruction-following, or code-generation tasks; EXTENDED includes math word problems such as GSM8K and SVAMP (App. G.1).

**Implication.** The best DCLM positives are question-and-answer text. CORE and MMLU are also question-answer evaluations. The course reads the gain as a mix of better text and closer resemblance to the evaluation format (**Interpretation**; §6 gives the evidence on format). A mixing decision made on CORE alone after this filter would exclude GitHub and arXiv data, and CORE has no code-generation task that could show the cost (**Interpretation**). For the final model the authors added StarCoder and ProofPile2 "to ensure our trained model is broadly useful, including for math and coding tasks" (§5).

## §4 Nemotron-CC: quality versus unique tokens at long horizons

**Definition.** Nemotron-CC is a 6.3T-token English Common Crawl dataset: 4.4T globally deduplicated real tokens and 1.9T tokens of rephrased text, labeled into five quality levels by a classifier ensemble ([[nemotron-cc]] §2.4, Table 4).

**Problem.** Nemotron-CC estimates that DCLM and FineWeb-Edu contain about 1.0T and 0.2T unique tokens after global fuzzy deduplication (Table 4), and that each classifier recalls about 10% of high-quality tokens (§2.2).

**Worked example (derived).** The 15T-token Nemotron-CC model drew 7.17T tokens from English Common Crawl (App. E). If those tokens came from a dataset with U unique tokens, passes = 7.17T / U: 7.2 for U = 1.0T (DCLM) and 35.9 for U = 0.2T (FineWeb-Edu). [[data-constrained-scaling]] found that up to about 4 epochs gives test loss close to unique data and that further repetition has diminishing value. The run itself weighted quality buckets unevenly, and its per-bucket repetition is not reported.

**Mechanism** ([[nemotron-cc]] §2).
1. Extract with Justext and do not apply heuristic filters to high-quality text. Over 13 snapshots, high-quality tokens (FineWeb-Edu classifier score 3-5) rose from 80B (Trafilatura, filtered) to 127B (Justext, unfiltered); the heuristic filters removed 18.1% of high-quality tokens (Table 1, §2.1).
2. Train two educational-value classifiers (linear heads on Snowflake-arctic-embed-m) on Mistral 8x22B-instruct and Nemotron-340B-instruct scores of the 460K FineWeb-Edu annotation documents, and add the released DCLM fastText classifier.
3. Round each classifier's score to buckets 0-19, each holding about 5% of documents.
4. Ensemble score = maximum bucket over the three classifiers.
5. Group buckets into five labels by annealing tests on an 8B model (an annealing test continues pretraining a partly trained checkpoint on a mix that contains the tested data and compares the resulting scores) (Table 2: High = 19, Medium-High = 18, Medium = 12-17, Medium-Low = 7-11, Low = 0-6).
6. Rephrase low-quality documents in Wikipedia style, and generate question-answer pairs, distilled passages, extracted knowledge, and knowledge lists from high-quality documents with Mistral NeMo 12B (§2.3).

**Formula.**

```
b_c(d) ∈ {0, …, 19}   bucket of document d under classifier c (about 5% of documents per bucket)
e(d) = max_c b_c(d)   ensemble bucket
```

**Worked example.** A document with buckets (12, 19, 8) gets e = 19 (High). A document with (5, 6, 7) gets e = 7 (Medium-Low). If the three classifiers' bucket-19 sets did not overlap, the union would hold 15% of documents; if they overlapped fully, 5% (derived bounds). Table 2 reports 12.63% of tokens in bucket 19; this is a token share, not a document share.

**Evidence.**
- The classifiers disagree. On CC-MAIN-2021-21, of documents labeled high quality by FineWeb-Edu's or DCLM's classifier, 10.1% are labeled high by both, 35.4% by FineWeb-Edu only, and 54.4% by DCLM only (Table 8).
- The ensemble keeps more tokens without a lower average (Table 9, 8B models, 1T tokens): high-quality share 25% for the ensemble against 8% (FineWeb-Edu classifier) and 11% (DCLM classifier); 10-task average 59.4 against 59.0 and 58.4. The DCLM classifier's selection scored 33.9 on SIQA against 45.8 for FineWeb-Edu's and 45.7 for the ensemble.
- Applying heuristic filters only to low-quality data gave MMLU 57.5 and a non-MMLU average of 60.6, against 54.1 and 60.9 when the filters were applied to all Justext data and 55.5 and 60.3 when they were applied to none (Table 7).
- Rephrasing low-quality documents raised the average from 52.5 to 54.0 but lowered MMLU from 48.2 to 47.1; replacing 4 of 8 repetitions of high-quality data with synthetic variants raised the average from 55.8 to 56.7 (Table 10).
- At 1T tokens, the 1.1T-token high-quality subset scored MMLU 59.0 and average 60.1 against DCLM's 53.4 and 57.0; the full 6.3T dataset scored 53.0 and 57.8 (Table 5).
- At 15T tokens, the 8B model scored MMLU 70.3 and a 10-task average of 64.7 against 65.3 and 64.2 for Llama 3.1 8B, measured in the authors' harness; it was lower on six of ten tasks, including RACE (37.8 vs 39.1) and WinoGrande (73.8 vs 74.7) (Table 6).

**Conditions and limits.** One ensembling strategy was tried; the rephrased text was not checked for factual accuracy; the data is English only and not decontaminated ([[nemotron-cc]] §6). The bucket tests are described as a "70% trained 8B" model with 9 tasks (§2.2) and as a 900B-token checkpoint with 13 tasks (App. C).

**Implication.** For long horizons the constraint is unique high-quality tokens, not the precision of one classifier. Ensembles raise recall, and rephrasing changes low-scored documents instead of removing them (see Negative samples).

## §5 Alternative selection objectives

### §5.1 QuRating: pairwise LLM judgments and temperature sampling

**Definition.** QuRating trains a rating model from pairwise LLM judgments along four criteria (writing style, facts and trivia, educational value, required expertise) and samples documents with probability that grows with the rating ([[qurating]] §3-4).

**Mechanism.**
1. GPT-3.5-turbo judges 250K document pairs per criterion in both orders; the averaged confidence is p_{B≻A} (§3.4).
2. A Bradley-Terry model turns judgments into scalar ratings. Pairwise judgments matched the authors' ranking of 10 documents better than single ratings (Kendall's tau 0.79 vs 0.61, §3.3).
3. A 1.3B QuRater model with four heads rates 260B tokens; ratings are normalized to variance 1.
4. Documents are sampled without replacement with the probabilities below (§4).

**Formulas.**

```
p_{B≻A} = σ(s_B − s_A)                      (Bradley-Terry)
p(d_i) ∝ exp(s_i / τ)                       (selection)
```

s is a document's rating, σ the sigmoid, and τ the temperature. τ → 0 is top-k selection; τ → ∞ is uniform sampling. App. C shows that this sampling approximates training on the whole corpus followed by RLHF toward higher ratings with KL weight τ.

**Worked example.** Three documents have ratings 2, 0, −2. At τ = 1 the weights are e², e⁰, e⁻² = 7.389, 1, 0.135, so the probabilities are 0.867, 0.117, 0.016. At τ = 2 the weights are e¹, e⁰, e⁻¹ = 2.718, 1, 0.368, giving 0.665, 0.245, 0.090. Doubling τ raises the lowest document's probability from 1.6% to 9.0%.

At pool scale the effect is on domain coverage. In an illustrative pool (not source data) with two equal domains whose ratings are normal with means +0.71 and −0.71 and standard deviation 0.71 (total variance 1), selecting 30% by top-k gives the lower-rated domain 5.6% of the selection; sampling at τ = 1 gives 25.1%, τ = 2 gives 35.9%, and uniform gives 50% (large-pool limit of Gumbel top-k sampling, course computation). Gumbel top-k adds independent Gumbel noise to each s/τ and keeps the k largest values, which samples k documents without replacement from p(d) ∝ exp(s/τ). The second panel of [figures/selection-tradeoffs.html](figures/selection-tradeoffs.html) reproduces these numbers and lets the reader vary τ, the domain gap, and the selected share.

**Evidence** ([[qurating]] Table 1; 1.3B models, 30B of 260B tokens, differences from uniform selection):

| Selection | Reading comp. (5) | Commonsense (3) | World knowledge (2) | Average (10) | Perplexity |
|---|---|---|---|---|---|
| Educational value, top-k | +3.8 | −0.1 | −0.5 | +1.8 | 10.59 (uniform 8.96) |
| Educational value, τ = 2 | +2.4 | +1.3 | +0.8 | +1.8 | 8.91 |
| Required expertise, top-k | +1.9 | −6.3 | −0.6 | −1.0 | 11.54 |
| Required expertise, τ = 2 | +1.8 | +0.5 | +0.1 | +1.1 | 8.93 |
| Lowest perplexity under Sheared-Llama-2.7B | −2.6 | −5.4 | −1.2 | −3.2 | 11.92 |
| Uniform, +50% data | +2.0 | +2.0 | +1.0 | +1.9 | 8.46 |

The authors report that top-k "achieves strong performance gains on individual tasks, but there is always a task where it performs worse than uniform selection" (§5.3). Educational value at τ = 2 improved all 10 tasks and matched uniform sampling with 50% more data.

**Conditions and limits.** 1.3B scale only; domain proportions were held fixed, so the method selects within domains (§5.2, Limitations).

### §5.2 WebOrganizer: quality filters as implicit domain mixtures

[[weborganizer]] labels each page with one of 24 topics and one of 24 formats using distilled 140M classifiers (§2.2). Replacing within-domain quality selection with random sampling at the same topic × format mixture recovers 84% of FineWeb-Edu's average gain but 35% of DCLM-fasttext's (Table 2). Most of FineWeb-Edu's average gain is therefore reproduced by the change in domain mixture alone, while most of DCLM-fasttext's gain is not (**Result, single study**). Setting per-domain quotas and then selecting the top-scored documents inside each domain raised the FineWeb-Edu average from 54.2 to 56.2 and HellaSwag from 56.0 to 62.5 (Table 1). The quotas were the Topic × Format mixture that RegMix tuned toward MMLU and HellaSwag (§4.2-4.3), so the HellaSwag recovery is partly a targeted gain; SIQA stayed at 49.4 against 49.9 for the random baseline (Table 10). A mixture tuned for MMLU alone lowered HellaSwag from 57.5 to 54.1 and PIQA from 71.3 to 69.9 (Table 10).

### §5.3 Perplexity correlations: selecting domains whose loss tracks a benchmark

**Definition.** [[perplexity-correlations-data-selection]] selects web domains whose loss, measured across 90 public models, is correlated with the models' benchmark scores. No proxy model is trained.

**Formulas** (§3-4).

```
y_i = f(⟨θ*, x_i⟩ + ε_i)
γ_j = Σ_{k≠l} sign(y_k − y_l) · (rank_j(x_k,j) − rank_j(x_l,j))
```

y_i is the benchmark error of model i, x_i the vector of its bits-per-byte losses on D domains (bits-per-byte is the negative log-likelihood of a text divided by ln 2 and by its number of UTF-8 bytes, which makes losses comparable across tokenizers), f an unknown increasing function, θ* domain weights, ε noise, and rank_j the rank of a model's loss on domain j among all models. Selection takes whole domains in decreasing γ until the token budget is filled (Theorem 2); a fastText classifier then generalizes the selection to pages.

**Worked example.** Four models have errors y = (0.30, 0.40, 0.50, 0.60). On domain j their losses are (1.0, 1.2, 1.1, 1.5), ranks (1, 3, 2, 4). The six unordered pairs contribute +2, +1, +3, −1, +1, +2; each appears twice in the ordered sum, so γ_j = 16. On domain m the losses (1.5, 1.0, 1.2, 1.1) have ranks (4, 1, 3, 2) and γ_m = −8. Domain j is selected first: lower loss on it goes with lower error.

**Evidence.** With 90 public models and 9,841 domains, at 160M parameters and 3.2B tokens, the method had an average rank of 1.750 over 8 evaluations, against 3.750 for DCLM's fastText filter with an English filter and 1.375 for that filter plus a manual per-task language filter (Table 1). In preregistered runs up to DCLM's 1B-1x setting targeting DCLM Core, the method outperformed DCLM's fastText filter when both filtered the raw pool, with the difference growing with scale, and the two were close on the pre-filtered pool, where the correlation coefficients ranged only from .23 to .33; in the raw pool for DCLM Core they ranged from −.07 to .32 (§1, Fig. 3, App. N). The most correlated domains for ARC Easy included www.aaeoptometry.com and www.akronchildrens.org (App. O).

**Conditions and limits.** The selection is only as broad as the target benchmark. The authors report weaker predictions for models trained on unusual data such as Phi (§5.3).

### §5.4 DsDm: selecting the data predicted to lower target-task loss

**Definition.** [[dsdm]] fits a linear datamodel that predicts how including each candidate example changes a target example's loss, then keeps the candidates with the lowest predicted average effect (§2).

**Formula.**

```
τ_θx(1_S) = θ_xᵀ 1_S
Ŝ_DM = arg bot-k ( (1/n) Σ_{i=1..n} θ_{x_i} )
```

1_S is the 0/1 vector marking which candidates are in subset S, θ_x the per-candidate effect on the loss of target example x, n the number of target examples, and bot-k the k smallest entries.

**Worked example.** Four candidates and two target examples: θ_x1 = (−0.3, 0.1, 0.1, 0.2), θ_x2 = (−0.1, −0.2, 0.1, 0.3). The average is (−0.2, −0.05, 0.1, 0.25). With k = 2, candidates 1 and 2 are kept. Candidates 3 and 4 have positive averages: including them is predicted to raise target loss.

**Evidence.** With LAMBADA, SQuAD, and Jeopardy as targets and 15 non-target benchmarks, DsDm's 1.3B model matched a 1.8B model trained on randomly selected data at twice the compute (Fig. 3); "no baseline selection method outperforms selecting data at random" (§4). Per benchmark (Table 1) DsDm gained on CoQA (+6.7) and NewsQA (+8.1) and lost on HellaSwag (−2.6) and OpenBookQA (−2.2) (derived from the printed accuracies). In 760M models, targeting LAMBADA alone lowered world-knowledge accuracy below random selection (Fig. 4).

**Conditions and limits.** Targeted methods selected data for four epochs while the random baseline used one (App. D.1). Datamodels were computed with 125M proxies.

## §6 Benchmark-targeted selection and training on the test task

**Definition.** **Benchmark-targeted selection** is any selection whose target, scorer, or selection rule is tuned toward named evaluation benchmarks. [[training-on-the-test-task]] names the broader practice "training on the test task": the use of knowledge about evaluation tasks at training time, which differs from contamination (test items in the training data) and "is not a malpractice" (Abstract).

**How targeting enters a data pipeline.**
1. Positives shaped like evaluation items: question-answer positives in DCLM (§3); a grade-school educational prompt in FineWeb-Edu (§2).
2. Thresholds and mixtures chosen by benchmark ablations: FineWeb-Edu's threshold (§2); SmolLM2's "performance-driven interventions", including a FineWeb-Edu/DCLM ratio change from 60/40 to 40/60 after annealing ablations found more DCLM "slightly improves MMLU MCF" ([[smollm2]] §4, §4.3-4.4).
3. Objectives defined by benchmarks: perplexity correlations (§5.3), DsDm (§5.4), and mixture regression toward MMLU and HellaSwag (§5.2; a regression model trained on small runs predicts benchmark scores from domain weights, [[weborganizer]] §3).
4. Benchmark training sets in late-stage data: SmolLM2's decay stage includes 0.02% AugGSM8K, "an augmented version of the GSM8K benchmark's training set" ([[smollm2]] §4.5). In Llama 3, annealing on GSM8K and MATH training sets raised 8B validation scores by 24.0% and 6.4% with negligible effect at 405B, and these sets were excluded from the final annealing data ([[llama-3]] §3.1.3).

**Evidence that the gain is task-format resemblance.** [[training-on-the-test-task]] regressed the accuracy of 56 base models (70M-70B) on pretraining compute:

```
A = α · max(0, log C − c_e) + θ · N + r + ε
```

A is accuracy, C pretraining compute, c_e the compute at which the task rises above chance, N = 1 for models released after November 2023, r chance accuracy, and θ the average newer-minus-older difference at equal compute. In **cloze format** each answer option is scored by its likelihood as a continuation of the question; in **multiple-choice format** the options are listed with letters and the model must output a letter. Before adjustment θ = 0.073 on MMLU and 0.191 on GSM8K. After every model was fine-tuned for three epochs on the same task data (about 100,000 multiple-choice training examples, or about 600,000 math examples), θ = 0.005 on both and not significant (Fig. 1). ARC-Challenge and HellaSwag in cloze format showed θ = 0.001 and 0.012; reformulated as MMLU-style multiple choice they showed θ = 0.120 and 0.114 (Fig. 4). MMLU in cloze format showed θ = 0.008 (Fig. 5). The authors conclude that the standard MMLU format "conflates knowledge-testing with testing a models' ability to answer multiple choice questions" (**Result, single study**).

**Detection protocol for a filter or mixture decision.**
1. Before the ablations, write down which benchmarks are used to choose the target, threshold, and mixture.
2. Keep a second suite that is never monitored. SmolLM2 reports MMLU-Pro, TriviaQA, and Natural Questions as "held-out benchmarks not monitored during training" (§4.7).
3. Evaluate the same items in cloze and multiple-choice format; a gain that appears only in multiple-choice format indicates format learning ([[training-on-the-test-task]] Fig. 4-5).
4. Compare candidates after equal fine-tuning on task data when the candidates differ in how much task-shaped data they contain ([[training-on-the-test-task]] §2.1).
5. Check contamination separately. DCLM removed detected MMLU and HellaSwag overlaps and scores did not fall (MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4; [[dclm]] Table 7), and the authors conclude that these gains are not likely caused by test examples in the data (§4.6). That check does not rule out test-task resemblance.
6. Measure per-domain perplexity on the unfiltered pool and on sources such as Paloma ([[paloma]]).
7. Split a benchmark by similarity to the training data. phi-1 solved 59.5-81.7% of HumanEval problems with close matches in its exercise set and 26.9-33.8% of the others ([[phi-textbooks]] Table 3).

## §7 Do small-scale filter decisions transfer to larger models?

**Definition.** **Decision accuracy** is the share of pairs of data recipes whose winner at a small scale is also the winner at the target scale ([[datadecide]] Eq. 3).

```
DA = (1/|P|) Σ_{(A,B)∈P} 1[ sign(ŷ_A − ŷ_B) = sign(y_A − y_B) ]
```

P is the set of recipe pairs, ŷ the small-scale score, and y the target-scale score averaged over 3 seeds.

**Worked example.** Three recipes score 40, 42, 41 at 150M and 50, 53, 54 at 1B. Pair (A, B): B wins at both scales, correct. Pair (A, C): C wins at both, correct. Pair (B, C): B wins small, C wins large, wrong. DA = 2/3 = 0.67.

**Evidence.**
- [[datadecide]]: over 25 corpora that include FineWeb-Edu, DCLM-Baseline, and 7 DCLM-Baseline variants with different DCLM and FineWeb-Edu classifier thresholds, single-size rankings at 150M pick the 1B winner in about 80% of pairs; none of 8 scaling-law methods exceeds the compute-accuracy frontier of single-scale ranking (§3.2). ARC-Easy is predictable with five orders of magnitude less compute than the target, while SocialIQA is hard to predict at every scale tested (§3.1, Fig. 2). Using the probability of the correct answer as the small-scale metric makes code decisions reach 80% decision accuracy (§3.4).
- [[dclm]] Fig. 3: for 10 methods, CORE at 400M-1x, 1B-1x, and 3B-1x correlates with CORE at 7B-1x with Pearson r = 0.838, 0.956, and 0.982.
- [[signal-and-noise-eval]]: a benchmark's signal-to-noise ratio at small scale correlates with its decision accuracy on DataDecide (R = 0.791).
- Scale-dependent effects exist. Perplexity-correlation gains grew from 160M to 1.4B ([[perplexity-correlations-data-selection]] Fig. 3), and Llama 3's benchmark-train annealing gain at 8B was negligible at 405B ([[llama-3]] §3.1.3).

**Conditions and limits.** DataDecide uses one token-to-parameter ratio (100) and multiple-choice cloze tasks up to 1B (§5). The FineWeb, QuRating, and DsDm conclusions come from 1.71B, 1.3B, and 1.3B models.

**Implication.** In DataDecide, 150M runs ranked 1B outcomes correctly in about 80% of pairs, and ARC-Easy was among the most predictable tasks, so small runs can rank filters on MMLU- and ARC-type tasks. For tasks with low signal at small scale, a filter's effect on breadth is not measured by those runs and must be stated as untested.

## §8 Language, dialect, and social bias in quality classifiers

**Evidence.**
- [[whose-language-counts-quality-filter]] replicated the GPT-3 quality filter (logistic regression with Wikipedia, Books3, and OpenWebText as positives and Common Crawl as negatives; 90.4% F1) and scored 910K U.S. high-school newspaper articles. In a regression of the average P(high quality) over 968 schools, the share of rural population had coefficient −0.069 and the share of adults with at least a bachelor's degree +0.059 (Table 3). In a regression over opinion articles, the presence of first- or second-person pronouns lowered P(high quality) by 5 percentage points, and articles on the topic about Trump and the presidential election scored 35 percentage points higher than articles on the omitted food topic (§3.3, Table 2). The filter's scores did not differ between high- and low-factuality news sources (p = 0.085, Fig. 3).
- [[qurating]]: ratings favored English Wikipedia over German and Russian Wikipedia although the judge was told to ignore language (§6.1, Fig. 4). Selecting 10% of the AboutMe pages by educational value retained 16% of "research, university" pages and 6% of "fashion, women" pages; roles such as "mommy" and "crafter" were retained at 6% under required expertise; top-k made these rates "far exacerbated" (§6.3, Table 2, App. Table 10).
- [[fineweb]] §4.2 and App. F.4 Table 3 (Paloma, no decontamination): on Twitter AAE, perplexity for the African-American-aligned subset is 246.9 under the FineWeb model and 575.1 under the FineWeb-Edu model; for the white-aligned subset it is 98.5 and 192.4. The FineWeb-Edu model fits both subsets worse, and the ratio is larger for the African-American-aligned subset (2.33 vs 1.95, derived). It also fits Gab, Manosphere, and 100 Subreddits worse.
- For contrast, the removal of African American English and Hispanic-aligned English documents from C4 was traced to a word blocklist, a heuristic filter covered in ch-10 ([[c4]], Dodge et al. summary).
- DCLM, Nemotron-CC, and FineWeb-Edu are English-only pipelines ([[dclm]] Fig. 4, [[nemotron-cc]] §6, [[fineweb]] §3.3).

**Interpretation.** A classifier trained to separate reference text from web text learns topic, register, and style features along with any quality signal. Its negative class defines what is removed (see the next section).

**Implication.** Retention rates by topic, format, region, and language variety are part of evaluating a filter. The multilingual side is covered in ch-13a.

## Negative samples and negative feedback

This section uses the terms of the style standard. In this stage, "negative" means **negative marginal value** (a document whose inclusion is expected to lower performance, removed or down-weighted by the filter) and the **negative class of a quality classifier**. The language model itself receives no negative gradient: selection only changes which documents are trained on with ordinary cross-entropy. Negatives as gradient appear only inside the classifier's training. Negative gradients for the language model are treated in ch-31a and ch-43a.

**1. Where negatives come from and how they are labeled.**
- Documents below a threshold or percentile (§2, §3). No source measures the false-negative rate of these removals directly. Indirect measurements: the FineWeb-Edu and DCLM classifiers agree on only 10.1% of the documents either labels high quality ([[nemotron-cc]] Table 8), and heuristic filters removed 18.1% of classifier-labeled high-quality tokens ([[nemotron-cc]] §2.1).
- The classifier's negative class: random documents from a RefinedWeb reproduction in DCLM ([[dclm]] App. J.1) and Common Crawl in the GPT-3 filter ([[whose-language-counts-quality-filter]] §3.2). These sets contain good documents that are labeled negative.
- The lower document in each pairwise LLM judgment ([[qurating]]).
- Candidates whose datamodel weight predicts higher target loss ([[dsdm]]).

**2. What current practice does with them.**
- Discard: FineWeb-Edu (score < 3) and DCLM (below top 10%).
- Down-weight: temperature sampling keeps a non-zero probability for low-rated documents ([[qurating]]); SmolLM2's small models removed score-0 DCLM documents and downsampled scores 1 and 2 ([[smollm2]] §6).
- Transform and keep as content: Nemotron-CC rephrases low-quality documents and trains on the rewrite with cross-entropy ([[nemotron-cc]] §2.3).

**3. Mechanism.** For the classifier with logit z = w · φ(d) and label y ∈ {0, 1}:

```
L = −[ y log σ(z) + (1 − y) log(1 − σ(z)) ]
∂L/∂z = σ(z) − y
```

φ(d) is the document's feature vector, w the weights, and σ the sigmoid. For a negative-labeled document with σ(z) = 0.8, ∂L/∂z = 0.8, so the update lowers the weights of the features present in that document. For a positive with σ(z) = 0.3, ∂L/∂z = −0.7, raising its features' weights. When a well-written document from the random web sample is labeled negative, its features (for example first-person narration or product vocabulary) are pushed down, and every document sharing those features scores lower. The pronoun and topic coefficients in [[whose-language-counts-quality-filter]] Table 2 are consistent with this mechanism (**Interpretation**).

For the language model, temperature sampling is a soft version of removal. QuRating's App. C shows that sampling with p(d) ∝ exp(s(d)/τ) approximates the KL-regularized optimum π*(d) ∝ p_D(d) exp(s(d)/τ): a document's weight falls exponentially with its rating instead of dropping to zero.

**4. Evidence with numbers.**
- Removal has measurable costs outside the target: FineWeb-Edu at 1B lowered HellaSwag by 1.5, PIQA by 1.4, and SIQA by 2.0, and raised held-out perplexity from 12.1 to 14.7 ([[weborganizer]] Table 1, Table 10).
- A document of low value for one target can have value for another: selecting documents low in facts and trivia or required expertise "benefits all 3 commonsense tasks" ([[qurating]] §5.3).
- Predicted-harmful data is harmful when tested: training on the candidates DsDm was least likely to select was "worse than selecting randomly" ([[dsdm]] §3.2).
- Transforming instead of discarding: rephrased low-quality data raised the 10-task average from 52.5 to 54.0 but lowered MMLU from 48.2 to 47.1 ([[nemotron-cc]] Table 10).
- Size of effect: no source in this chapter measures what share of a filter's gain comes from removing low-value documents versus concentrating high-value ones. The closest measured decomposition is the share explained by the implied domain mixture (84% for FineWeb-Edu, 35% for DCLM-fasttext; [[weborganizer]] Table 2).

**5. Controls.** Sample with a temperature instead of top-k (§5.1); set per-domain quotas before selecting (§5.2); raise recall with an ensemble (§4); apply heuristic filters only to low-scored buckets (§4); rephrase or downsample instead of deleting; include an inverse-selection run as a control ([[qurating]] §5.2).

**6. Diagnostics.** Retention rate by topic, format, region, and language variety ([[weborganizer]], [[qurating]] §6.3); per-domain perplexity on removed slices ([[paloma]]); random samples at the 5th, 30th, 70th, and 95th score percentiles ([[qurating]] §6.2); the agreement matrix between two classifiers ([[nemotron-cc]] Table 8); held-out perplexity on the unfiltered pool.

**7. Effect on generality.** Coverage falls (higher held-out perplexity, fewer unique tokens), some commonsense and social tasks fall, and informal, regional, and non-English text is selected less often (§8).

## Recipe

Rows marked 2026-09-15 were read at the stated locus in the primary PDF for this chapter; rows marked 2026-09-14 are from verified library cards.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| FineWeb-Edu classifier | not reported | data filter (no §5.2 stage) | annotations; head; epochs; LR; threshold | 460,000 Llama-3-70B-Instruct scores (410,000 train / 50,000 validation); linear regression on frozen Snowflake-arctic-embed-m; 20; 3e-4; integer score ≥ 3 | arXiv:2406.17557v2 §4 ([[fineweb]]) | conflict (model card: 450,000 train, 46,867 hold-out) | App. F.2 Fig. 17: thresholds 2/3/4 at 28B tokens, 3 best aggregate |
| FineWeb data-ablation model | 1.71B | pretrain-stable | tokens; sequence; batch; LR | 28B (threshold ablations), 350B (comparisons); 2,048; ~2M tokens; peak 3e-4, 500 warmup steps, cosine to 3e-5 | arXiv:2406.17557v2 §3.1, App. D ([[fineweb]]) | verified 2026-09-14 | no ablation reported |
| DCLM fastText filter | not applicable | data filter (no §5.2 stage) | training docs; positives; negatives; features; kept share | 400K (200K/200K); 100K OpenHermes-2.5 + ELI5 (post ≥ 0, best comment ≥ 5, ≥ 3 comments); random docs of an earlier RefinedWeb reproduction; unigrams + bigrams (`wordNgrams` 2); top 10% | arXiv:2406.11794v4 §4.4, App. J.1 ([[dclm]]) | verified 2026-09-15 | Table 5: positives and 10/15/20% at 7B-1x; Table 14: bigrams +1.0 CORE, −0.7 EXTENDED |
| DCLM 1B-1x ablation model | 1.4B | pretrain-stable | tokens; batch; LR; WD; warmup; z-loss | 28.8B; 256 × 2,048 tokens; 3e-3; 0.033; 5,000; 1e-4 | arXiv:2406.11794v4 Table 1, Table 10 | verified 2026-09-15 | App. F: alternatives in Table 12 performed worse; Fig. 3: CORE rank correlation with 7B-1x r = 0.956 |
| DCLM 7B-1x ablation model | 6.9B | pretrain-stable | tokens; batch; LR; WD; warmup; z-loss | 138B; 2,048 × 2,048 tokens; 2e-3; 0.05; 5,000; 5e-6 (Table 5 runs: LR 3e-4, WD 0.33) | arXiv:2406.11794v4 Table 1, Table 10, App. F | verified 2026-09-15 | Table 11: LR/WD sweep, 2e-3/0.05 CORE 44.8 |
| DCLM-Baseline 7B | 6.9B | pretrain-decay/anneal | cooldown data; tokens; soup | 70% DCLM-Baseline at top-7% fastText + 30% ProofPile; two cooldowns of 200B and 270B from the 2T checkpoint; weights 0.2 / 0.8 | arXiv:2406.11794v4 App. Q, Table 28 | verified 2026-09-15 | Table 28: soup MMLU 63.9 vs 62.7 and 63.4 |
| DCLM-8k | 6.9B | long-context | tokens | "100B tokens" (§5) vs "∼120B tokens" (App. Q.2, Table 30) | arXiv:2406.11794v4 §5, App. Q.2 | conflict | Table 30: CORE 56.0 → 57.1 |
| Nemotron-CC classifier ensemble | not applicable | data filter (no §5.2 stage) | classifiers; head; epochs; LR; combination | Mistral 8x22B-instruct and Nemotron-340B-instruct scores of 460K docs + DCLM fastText; linear regression on Snowflake-arctic-embed-m; 20; 3e-4; max over 0-19 buckets | arXiv:2412.02595v2 §2.2 ([[nemotron-cc]]) | verified 2026-09-15 | Table 9: ensemble 25% HQ, avg 59.4 |
| Nemotron-CC bucket test | 8B | mid-train | tokens; mix; starting point; tasks | 50B; 34% bucket / 66% default; "70% trained 8B", 9 tasks (§2.2) vs 900B-token checkpoint, 13 tasks (App. C) | arXiv:2412.02595v2 §2.2, App. C | conflict | Fig. 4: bucket 19 highest, 12-18 marginal |
| Nemotron-CC rephrasing | 12B generator | data filter (no §5.2 stage) | model; sampling; chunk limits | Mistral NeMo 12B instruct, FP8; top-p 0.9, temperature 0.5; 512 (Wikipedia), 2,000 (Distill), 1,400 (Extract), 1,000 (QA, list) tokens | arXiv:2412.02595v2 §2.3 | verified 2026-09-15 | Table 10: LQ-Synthetic +1.5 avg; HQ-Synthetic +0.9 |
| Nemotron-CC ablation model | 8B | pretrain-stable | tokens; blend; optimizer; LR; sequence and batch | 1T; 73% tested CC + 27% fixed; Adam (0.9, 0.95), WD 0.1; cosine 3e-4 → 3e-6; sequence and batch not reported | arXiv:2412.02595v2 App. D, Table 12 | verified 2026-09-15 (sequence, batch: not reported; checked body, App. A-H) | no ablation reported |
| Nemotron-CC 15T model | 8B | pretrain-stable | Common Crawl share per phase | phase 1: 9T tokens, 59% CC (medium, medium-high, high); phase 2: 6T, 31% CC (high only); 7.17T total | arXiv:2412.02595v2 App. E | verified 2026-09-15 | Table 6 vs Llama 3.1 8B |
| QuRating selection runs | 1.3B | pretrain-stable | pool; selected; τ; batch; LR; warmup; WD | 260B; 30B; {0, 1, 2} on unit-variance ratings; 2,048 × 1,024 tokens; 5e-4 cosine to 5e-5; 5%; 0.1 | arXiv:2402.09739v3 §5.1-5.2, App. D ([[qurating]]) | verified 2026-09-15 | Table 1: τ = 2 vs top-k |
| WebOrganizer DCLM 1b-1x runs | 1.44B | pretrain-stable | tokens; batch; sequence; pool | 28,795,904,000; 256 sequences; 2,048; 30B selected from 200B | arXiv:2502.10341v3 App. E ([[weborganizer]]) | verified 2026-09-14 | Table 1 |
| Perplexity-correlation runs | 160M | pretrain-stable | tokens; LR; per-device batch; WD; warmup | 3.2B; 5e-3; 128 on 4 GPUs; 0.1; ratio 0.1 | arXiv:2409.05816v2 §5.1, App. G Table 2 | verified 2026-09-15 | LR doubled until instability (App. G.1) |
| DsDm §4 models | 1.3B | pretrain-stable | tokens; batch; LR; WD; epochs over selection | 2.6 × 10¹⁰; 1,024 × 1,024 tokens; 6e-4; 4e-4; 4 (targeted), 1 (random) | arXiv:2401.12926v1 App. A.4 Table 2, App. D.1 ([[dsdm]]) | verified 2026-09-15 | hyperparameters chosen for 125M C4 perplexity |
| SmolLM2-1.7B | 1.7B | pretrain-stable | web mixture | stage 1 (0-6T): 60% FineWeb-Edu / 40% DCLM; stage 3 (8-10T): 40 / 60 | arXiv:2502.02737v1 §4.2, §4.4 ([[smollm2]]) | verified 2026-09-15 | Table 1 (350B ablations); stage 3 annealing ablation (numbers not reported) |
| SmolLM2-1.7B | 1.7B | pretrain-decay/anneal | benchmark training data | AugGSM8K 0.02% of mixture | arXiv:2502.02737v1 §4.5 | verified 2026-09-15 | no ablation reported |
| SmolLM2-135M, -360M | 135M, 360M | data filter (no §5.2 stage) | web filter | DCLM filtered by FineWeb-Edu classifier: remove score 0, downsample scores 1-2 | arXiv:2502.02737v1 §6 | verified 2026-09-15 | ablations at target length; numbers not reported |
| Llama 3 8B annealing test | 8B | eval-gate | benchmark training sets in annealing | GSM8K +24.0%, MATH +6.4% at 8B; negligible at 405B; excluded from final annealing | arXiv:2407.21783 §3.1.3 ([[llama-3]]) | verified 2026-09-14 | same section |
| Test-task adjustment | 70M-70B | eval-gate | data; epochs; LR; batch | MMLU auxiliary train (~100K examples, 30M tokens) or MetaMathQA + Orca-Math (~600K, 200M tokens); 3; 2e-5 (< 10B), 2e-6 (> 10B), cosine, 50 warmup; 64 | arXiv:2407.07890v3 §2.1, App. A.2 ([[training-on-the-test-task]]) | verified 2026-09-15 | App. A.3 learning-rate sweep |
| DataDecide 150M and 1B | 151.9M, 1176.8M | pretrain-stable | batch (sequences); peak LR; tokens | 192, 4.2e-03, 15.0B; 704, 2.1e-03, 100.0B | arXiv:2504.11393v2 App. A Table 2 ([[datadecide-recipe]]) | verified 2026-09-14 | ladder heuristics; no ablation reported |

**Starting point for a small general-purpose run.** For a 1-2B model trained on a pool that exceeds the budget, the verified rows support: train a fastText classifier on 400K documents (200K OpenHermes-2.5 and ELI5 positives, 200K random documents from the heuristically cleaned pool) with unigram and bigram features and keep the top 10%, as DCLM selected at 7B-1x on CORE; or apply the FineWeb-Edu classifier at threshold 3, as selected at 1.71B and 28B tokens. SmolLM2 mixed the two at 60/40 for its first 6T tokens at 1.7B. Rank candidate filters at DCLM's 1B-1x setting (1.4B parameters, 28.8B tokens, 256 × 2,048-token batches, LR 3e-3, weight decay 0.033, 5,000 warmup steps), whose rankings correlated with 7B-1x at r = 0.956. When the planned horizon requires many passes over the filtered set's unique tokens, add Nemotron-CC-style max-ensemble buckets (0-19 buckets, maximum over classifiers) instead of tightening the threshold. Evaluate every candidate on the §6 detection protocol.

## Generalization lens

**(a) What increases breadth.** A positive set chosen to cover "a wide range of potential topics" ([[dclm]] App. J.1): DCLM-fasttext raised 8 of 9 tasks at 1B ([[weborganizer]] Table 1). Sampling with a temperature: educational value at τ = 2 raised all 10 tasks ([[qurating]] §5.3). Domain quotas under a quality filter: FineWeb-Edu + quotas (a mixture tuned toward MMLU and HellaSwag) reached the highest average (56.2) and HellaSwag 62.5 ([[weborganizer]] Table 1, Table 10). Classifier ensembles and rephrasing: 4× more unique real tokens than DCLM at similar 1T-token scores ([[nemotron-cc]] Table 4-5). Several target tasks of different types: DsDm with three targets gained on reading comprehension and world knowledge without lowering other categories on average ([[dsdm]] §4.1).

**(b) What causes narrowing.** Top-k selection on one rating: required expertise top-k lowered commonsense by 6.3 points ([[qurating]] Table 1). A single-benchmark target: an MMLU-only mixture lowered HellaSwag by 3.4 ([[weborganizer]] Table 10); a LAMBADA-only DsDm target lowered world knowledge ([[dsdm]] Fig. 4). Positives shaped like evaluation items or targets written for one school level (§2, §3, **Interpretation**). Filtering that leaves 0.2T-1.0T unique tokens when a 15T-token run draws 7.17T web tokens ([[nemotron-cc]] Table 4, App. E). Classifier features that remove informal, rural, and non-English text ([[whose-language-counts-quality-filter]], [[qurating]] §6). Mixing decisions made on suites without code-generation tasks: adding RedPajama sources that include GitHub and arXiv lowered CORE and EXTENDED for DCLM-Baseline, whose 7B model scored 2.1 on GSM8K before instruction tuning ([[dclm]] Table 6, Table 29).

**(c) How to measure it for this stage.** A split suite of targeted and non-targeted tasks plus a broad suite such as DCLM EXTENDED ([[dclm]] §3.5); held-out perplexity on the unfiltered pool and per-domain perplexity ([[weborganizer]] Table 10, [[paloma]]); benchmarks never monitored during data decisions ([[smollm2]] §4.7); cloze versus multiple-choice versions and equal fine-tuning before comparison ([[training-on-the-test-task]]); retention rates by topic, format, region, and language ([[qurating]] §6.3); unique tokens after global deduplication ([[nemotron-cc]] Table 4); decision accuracy with seeds for the tasks used ([[datadecide]]). Known measurement errors: CORE and EXTENDED can rank two settings differently by less than one point without seed estimates ([[dclm]] Table 5, 14, §6); multiple-choice MMLU mixes format skill with knowledge ([[training-on-the-test-task]] Fig. 5); SocialIQA decisions are unreliable at the scales tested ([[datadecide]] §3.1); MMLU overlap flags prioritize recall and overestimate contamination ([[dclm]] Table 25); comparisons with external models in a different harness are not like-for-like ([[nemotron-cc]] Table 6).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Choosing a threshold on the average of knowledge benchmarks only | Average rises; HellaSwag, PIQA, or SIQA fall | Report per-task deltas against a random-selection baseline on the same pool ([[weborganizer]] Table 1) |
| Reading a gain on question-answer benchmarks as general quality | Gain disappears in cloze format or after equal task fine-tuning | Evaluate cloze and multiple-choice versions; apply the equal fine-tuning adjustment ([[training-on-the-test-task]]) |
| Treating decontamination as proof of no targeting | Scores unchanged after overlap removal, and the claim of general quality rests on that result alone | Run overlap removal, the cloze-versus-multiple-choice comparison, and an unmonitored suite as separate tests ([[dclm]] Table 7; [[training-on-the-test-task]] Fig. 4) |
| Hard top-k on one rating | Worse held-out perplexity than uniform; one task below uniform | Compare τ = 0 with τ = 2 at equal tokens ([[qurating]] Table 1) |
| Tightening the threshold for a long horizon | Planned passes over unique tokens exceed the roughly 4 epochs that [[data-constrained-scaling]] found close to unique data | Compute passes = needed tokens / unique tokens after global deduplication ([[nemotron-cc]] Table 4, [[data-constrained-scaling]]) |
| Rejecting curated code, math, or book sources on CORE alone | Low math scores after pretraining (GSM8K 2.1 for the DCLM-Baseline 7B base model) | Include code and math tasks in the decision suite ([[dclm]] Table 6, Table 29) |
| Trusting human agreement as a filter metric | Filter with high ROC-AUC on human labels gives lower CORE | Evaluate filters by trained-model results, not label agreement ([[dclm]] App. N) |
| Assuming the filter is neutral across language varieties and regions | Retention of informal, regional, or non-English pages below the kept share (for example 6% for "fashion, women" pages at a 10% selection) | Compute retention by attribute on a labeled sample ([[qurating]] §6.3; [[whose-language-counts-quality-filter]] Table 3) |
| Deciding on a task with low small-scale signal | Rankings flip between seeds or sizes | Use decision accuracy and signal-to-noise per task ([[datadecide]], [[signal-and-noise-eval]]) |

## Check your understanding

1. FineWeb-Edu raised ARC-Easy by 9.7 points and lowered SIQA by 2.0 at 1B. Explain how the annotation prompt and the threshold together could produce this profile, and what experiment would separate prompt resemblance from better text.
2. WebOrganizer's implicit domain mixture recovers 84% of FineWeb-Edu's gain but 35% of DCLM-fasttext's. What does this imply about how each filter changes the data, and why does per-domain quota selection help FineWeb-Edu more?
3. In [[training-on-the-test-task]], ARC and HellaSwag show no significant newer-model advantage in cloze format but θ = 0.120 and 0.114 in multiple-choice format. Explain why this points to format learning rather than contamination, and how a data team should use the same comparison when evaluating a question-answer-shaped classifier.
4. A 15T-token run needs 7T tokens of web data. Using unique-token counts, explain why the choice between a top-10% filter and a graded ensemble is a coverage decision and not only a quality decision.
5. Using the logistic-loss gradient, explain how random web documents in the negative class can make a classifier remove first-person or informal writing, and name two diagnostics that would detect it.
6. Perplexity-correlation selection outperformed DCLM's fastText filter on the raw DCLM pool but not on the pre-filtered pool. Explain this with the range of correlation coefficients in each pool, and state what it implies about stacking a benchmark-correlated filter on top of an existing classifier.
7. DsDm selected data for four epochs while random selection used one. Explain how this difference affects the interpretation of the "2× compute multiplier" as evidence for broad capability.

## Connections

- **Previous (dependency):** ch-10 — Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb. It builds the cleaned pool that model-based filters score.
- **Next:** ch-11 — Tokenizers, Data Provenance, and PII Removal.
- ch-00 — What General Capability Means and How It Is Measured (targeted versus held-out suites).
- ch-09 — Pretraining Data Composition and Capability Coverage (coverage as the quantity filters change).
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic (unique-token counts used in §4).
- ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining (repetition of filtered data).
- ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale (WebOrganizer-style quotas and mixture regression).
- ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (language effects of English-trained filters).
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (passes over unique tokens; contamination checks).
- ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement (applies §6 and the Generalization lens).
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix (uses this chapter's filters for synthetic data).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection (selection for post-training data).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (benchmark-train data in annealing).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation.
- ch-48 — Contamination Detection and Its Effect on Reported Scores.
- ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.

## Sources

- [[fineweb]] — FineWeb-Edu annotation, classifier, threshold choice, MMLU and ARC gains, topic shift and Paloma domain fit.
- [[dclm]] — fastText filter recipe, positive-set and threshold ablations, CORE and EXTENDED, human-judgment, mixing, contamination, code and math results, rank transfer across scales.
- [[nemotron-cc]] — unique-token problem, classifier disagreement, max ensemble and buckets, rephrasing, 1T and 15T results.
- [[qurating]] — pairwise ratings, temperature sampling, top-k versus τ = 2 table, inverse sampling, language and social retention.
- [[weborganizer]] — per-task results of FineWeb-Edu and DCLM filters on one pool, implicit domain mixtures, quotas, single-target mixture costs.
- [[perplexity-correlations-data-selection]] — single-index model, rank-correlation estimator, preregistered raw versus pre-filtered pool results.
- [[dsdm]] — datamodel selection formula, targeted versus held-out benchmark table, single-target narrowing, epoch difference.
- [[datadecide]] — decision accuracy definition and transfer of small-scale data rankings.
- [[datadecide-recipe]] — per-size DataDecide configurations used in the Recipe.
- [[training-on-the-test-task]] — definition, regression model, equal fine-tuning adjustment, cloze versus multiple-choice results.
- [[whose-language-counts-quality-filter]] — GPT-3 filter replication and its topic, style, and demographic preferences.
- [[smollm2]] — FineWeb-Edu versus DCLM per-task ablation, benchmark-monitored mixture changes, AugGSM8K share, unmonitored benchmarks.
- [[llama-3]] — annealing on benchmark training sets at 8B and 405B.
- [[paloma]] — per-domain perplexity as a coverage measurement.
- [[data-constrained-scaling]] — value of repeated epochs, used for the passes calculation.
- [[signal-and-noise-eval]] — signal-to-noise ratio as a predictor of decision accuracy.
- [[openhermes-2-5]] — composition of the instruction dataset used as DCLM positives.
- [[phi-textbooks]] — similarity-split HumanEval evaluation as a detection method.
- [[c4]] — blocklist-based dialect removal, for contrast with model-based filters.
