<!-- chapter: ch-09
     track: pretraining
     kind: content
     title: Pretraining Data Composition and Capability Coverage
     deps: [ch-08b]
     sources: [[the-pile]], [[pretrainers-guide-training-data]], [[to-code-or-not-to-code]], [[code-pretraining-task-performance]], [[data-constrained-scaling]], [[llama-3]], [[llama-3-recipe]], [[deepseek-v3]], [[dolma]], [[fineweb]], [[dclm]], [[nemotron-cc]], [[olmo-2]], [[weborganizer]], [[paloma]], [[datadecide]], [[doremi]], [[consent-in-crisis]], [[epoch-data-stock]]
     figures: figures/data-landscape.html
     revised: 2026-09 (generality revision)
-->

# Chapter 9 — Pretraining Data Composition and Capability Coverage

> **Core insight.** The domains a pretrained model fits and the tasks it can do follow from the sources in its corpus. In controlled 1.3B-1.5B runs, a model trained on the 22-source Pile had lower bits per byte than a CC-100-trained model on all 22 Pile components ([[the-pile]] Table 4), and removing Common Crawl, Books, or OpenWebText2 from the Pile lowered the average QA score by 4.8, 2.7, and 1.4 ([[pretrainers-guide-training-data]] Fig. 8). Code shares of 25% (470M) and 50% (4.2B) did not lower natural-language averages ([[to-code-or-not-to-code]], [[data-constrained-scaling]]), while world-knowledge scores fell as the code share grew, by 3.4% relative at 25% and 31% at 75% code ([[to-code-or-not-to-code]] §3.3; also [[code-pretraining-task-performance]] §5). Llama 3 publishes only a four-category mix (about 50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual) chosen with small-model scaling-law experiments ([[llama-3]] §3.1.2), so controlled evidence about composition comes from open corpora.
>
> **Guideline.** When the goal is breadth, keep every heterogeneous source type (web, books, encyclopedic, academic, code) in the mixture and set shares with small-scale runs scored on per-domain loss and a broad task suite, because removing Common Crawl, Books, or OpenWebText2 lowered average QA accuracy and the best average used all or nearly all sources ([[pretrainers-guide-training-data]] §6) and Core scores of 10 curation methods at 1B-1x correlated with their 7B-1x scores at Pearson r = 0.956 ([[dclm]] §3.2, Fig. 3). When adding code, treat 25% (470M, 200B tokens) to 50% (4.2B, 84B tokens) as the tested range in which natural-language averages did not fall, and track world-knowledge tasks separately, because they fell by 3.4% relative at 25% code and 31% at 75% ([[to-code-or-not-to-code]] §3.3). When a quality or topic classifier selects web data, measure the topic mix it keeps, because Llama 3's classifier downsamples arts and entertainment ([[llama-3]] §3.1.2) and a mixture tuned for MMLU lowered HellaSwag from 57.5 to 54.1 ([[weborganizer]] Table 10). Otherwise, when the web portion is already selected by a model-based filter (DCLM-Baseline keeps the top 10% of documents by fastText score), test whether curated non-web sources still help, because adding them lowered DCLM Core by 1.2 at 1B scale ([[dclm]] Table 6).

## Why this chapter matters for a general-purpose model

The training pipeline has six stages: pretraining, mid-training, supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), and evaluation. This chapter is about the first stage. [[ch-08b]] explained why next-token prediction on broad text produces abilities that were not targeted; this chapter asks which parts of "broad text" supply which abilities.

The measurable problem is coverage. For a fixed token budget (set by the compute allocation in [[ch-08a]]), two corpora can give models with similar average loss but different per-domain loss, different accuracy on domain-specific questions, and different knowledge of recent events. Later stages do not fully repair these gaps: in [[pretrainers-guide-training-data]] §4, the accuracy loss from a mismatch between pretraining-data year and evaluation year remained after fine-tuning on the full year-matched training set.

Four terms are used throughout. A **source** is a collection with one provenance, such as a Common Crawl extraction, a code host, or a book collection. A **domain** is a group of documents or sources with similar content, defined by whoever runs the analysis. A **mixture weight** is the share of training tokens or bytes taken from a source, which differs from the source's share of the raw corpus when sources are repeated or subsampled. **Capability coverage** is the set of evaluation domains and task types on which a model performs above a stated baseline. Filters that remove documents are taught in [[ch-10]] and [[ch-10a]]; algorithms that search for mixture weights are taught in [[ch-13]]; multilingual coverage is in [[ch-13a]]; repetition and unique-token limits are in [[ch-14]]; late-stage annealing mixtures are in [[ch-32]].

## §1 Composition: sources, mixture weights, and units

**Definition.** A pretraining composition is the list of sources used in a run together with each source's mixture weight over the run.

**Problem.** A report that lists sources without weights, or weights without units, does not let a reader reproduce the training distribution or compare two runs.

**Mechanism.** In The Pile, the weight of a source is fixed by two choices:
1. Collect the source and measure its raw size S_i.
2. Choose how many passes e_i ("epochs") the mixture makes over the source during one pass over the full corpus. The Pile raised the epochs of "higher quality components", up to 3 for Wikipedia, "Following Brown et al. (2020)" ([[the-pile]] §2).
3. The source's effective size is S_i · e_i, and its weight is its effective size divided by the total effective size.

**Formula.**

```
w_i = (S_i · e_i) / Σ_j (S_j · e_j)
```

- w_i: share of training bytes that come from source i.
- S_i: raw size of source i before up- or down-sampling (GiB).
- e_i: number of passes over source i during one pass over the mixture.
- The sum runs over all sources j in the mixture.

**Worked example.** Rows copied from [[the-pile]] Table 1 (arXiv:2101.00027v1):

```
Component         Raw Size     Weight   Epochs   Effective Size
Pile-CC           227.12 GiB   18.11%   1.0      227.12 GiB
PubMed Central     90.27 GiB   14.40%   2.0      180.55 GiB
Books3            100.96 GiB   12.07%   1.5      151.44 GiB
OpenWebText2       62.77 GiB   10.01%   2.0      125.54 GiB
ArXiv              56.21 GiB    8.96%   2.0      112.42 GiB
Github             95.16 GiB    7.59%   1.0       95.16 GiB
Wikipedia (en)      6.38 GiB    1.53%   3.0       19.13 GiB
The Pile          825.18 GiB                     1254.20 GiB
```

PubMed Central: 90.27 × 2.0 = 180.55 GiB, and 180.55 / 1254.20 = 14.40%. Its share of the raw corpus is 90.27 / 825.18 = 10.9%. Pile-CC goes the other way: 27.5% of raw bytes (227.12 / 825.18) but 18.11% of training bytes, because most other large sources were repeated (Github, at 1.0 epochs, was not). Wikipedia is 0.77% of raw bytes and 1.53% of training bytes. Dividing raw sizes by the effective total (for example 90.27 / 1254.20 = 7.2%) mixes two columns and gives neither quantity. The Table 1 caption states that "Weight is the percentage of bytes in the final dataset"; the weights are byte shares, not token shares.

**Units across reports.** Byte shares and token shares of the same source differ because bytes per token differ by content. The Pile reports that Github, ArXiv, Stack Exchange, and DM Mathematics are among the components with the fewest bytes per GPT-2 token ([[the-pile]] §5.1), so these sources have a larger token share than byte share. Each corpus in this chapter also counts tokens with a different tokenizer: Dolma uses Llama tokens ([[dolma]] Table 1), FineWeb uses GPT-2 tokens ([[fineweb]] §3), DCLM uses GPT-NeoX tokens ([[dclm]] §3.1), and the Epoch stock estimate uses cl100k_base tokens ([[epoch-data-stock]] §2). Token counts from different reports are therefore comparable only to within the ratio of their tokenizers' compression.

**Conditions and limits.** The Pile's epoch choices were not ablated. [[doremi]] later searched for Pile domain weights with a 280M proxy model and raised the average one-shot exact match of an 8B model from 20.03 to 26.56 over the default weights (§3.2, Table 5), so the default weights were not optimal for that evaluation (Result, single study; details in [[ch-13]]).

**Implication.** A composition is specified only when each row states the source, the raw size, the repetition, and the unit. The companion figure **[figures/data-landscape.html](figures/data-landscape.html)** lets the reader switch Panel A between the raw-size share and the training weight of every Pile component, so the effect of the epoch column can be read row by row.

## §2 Source types and the capabilities they supply

**Definition.** Bits per byte (BPB) is the model's average negative log-likelihood on a test set, converted to base 2 and normalized by UTF-8 bytes instead of tokens. The Pile prefers BPB because it does not depend on the tokenizer ([[the-pile]] §3.1).

**Problem.** A single held-out loss on a corpus with a web majority can hide high loss on the minority domains. Per-domain BPB exposes it.

```
BPB = (L_T / L_B) · ℓ / ln 2
```

- ℓ: mean negative log-likelihood per token on the test set, in nats.
- L_T: length of the test set in tokens; L_B: its length in UTF-8 bytes.
- ln 2 converts nats to bits.

**Worked example.** The Pile reports L_T / L_B = 0.29335 GPT-2 tokens per byte across the Pile (§3.1). A model with ℓ = 2.0 nats per token has BPB = 0.29335 × 2.0 / 0.6931 = 0.846.

**Evidence.** The Pile authors trained architecturally identical 1.3B models on the Pile, on CC-100 (English), and on raw Common Crawl, each deduplicated against the evaluation sets and downsampled to about 40GB (§4.1; token counts and steps are not reported in §4). Table 3 gives Pile test BPB 0.9433 for the Pile model, 1.3293 for CC-100, and 1.1275 for raw CC; LAMBADA accuracy was 50.1, 49.7, and 43.8. Selected rows of Table 4 (BPB on each Pile test component; lower is better):

```
Evaluated on        trained on: The Pile   CC-100 (en)   Raw CC (en)
Pile-CC                          0.9989     1.0873        1.0287
ArXiv                            0.7945     1.8159        1.2642
Github                           0.5597     1.6509        0.9301
DM Mathematics                   1.5206     3.1774        2.6229
EuroParl                         1.1202     2.7141        1.4917
Books3                           1.0734     1.2264        1.1366
Wikipedia (en)                   0.8961     1.1807        1.0252
```

The Pile model is best on every one of the 22 rows. The gap is 0.09 bits on Pile-CC, where all three corpora contain web text, and more than 1.0 bit on ArXiv, Github, DM Mathematics, and EuroParl, which are not web pages. Raw CC is better than CC-100 on Pile BPB; the authors attribute this to CC-100's perplexity filter, which uses a language model trained on Wikipedia and discards documents whose perplexity is too high or too low, so documents too similar to or too different from Wikipedia are removed (§4.2, Interpretation). **Replicated:** with 1.2B models on 150B tokens, the Pile and Dolma models fit the diverse Paloma domains well, while single-source web corpora (C4, mC4-en, RefinedWeb) have higher average perplexity ([[dolma]] §9.2, Fig. 5); in [[paloma]] §4.1, the C4-only 1B baseline reaches perplexity 391,171 on RedPajama arXiv, and the C4 and mC4-en baselines get worse between about 20B and 150B tokens on 65 and 43 domains.

Panel B of **[figures/data-landscape.html](figures/data-landscape.html)** plots all 22 rows of Table 4 and sorts them by the gap between the CC-100 and Pile models.

| Source type | What the evidence says it supplies | Setting | Status |
|---|---|---|---|
| Filtered web crawl | Largest single effect on broad QA: removing Pile-CC (26.9% of the deduplicated Pile) lowered all 7 evaluation groups, average −4.8 ([[pretrainers-guide-training-data]] Fig. 8) | 1.5B, Pile variants | Result (single study) |
| Link-curated web (OpenWebText2) | Removal lowered common-sense QA by 5.8 and the average by 1.4 | same | Result (single study) |
| Books | Removal lowered the average by 2.7 and Web QA by 6.3; books are long documents (Books3 mean 538.36 KiB vs Pile-CC 4.33 KiB, [[the-pile]] Table 1), which matters for long-context data in [[ch-32b]] | same | Result (single study) |
| Wikipedia | Removal lowered Wiki QA by 1.3 and raised toxic generation by 4.2 (Table 3) | same | Result (single study) |
| Academic, biomedical | Removing PubMed lowered Biomed QA by 5.8; a 4.9% reference share gave nearly the same S2ORC perplexity as 24.2% ([[dolma]] App. M) | 1.5B; 1B | Result (single study) each |
| Code | Code tasks rise with code share; some reasoning and structured-output tasks also rise (§4) | 374M-4.2B | Replicated (§4) |
| Math | Pile model DM Mathematics BPB 1.5206 vs 3.1774 for CC-100; Llama 3 assigns 25% of tokens to math and reasoning ([[llama-3]] §3.1.2) | 1.3B; report | Result (single study) |
| Multilingual | EuroParl BPB 1.1202 vs 2.7141; the Pile is 97.4% English by fastText ([[the-pile]] §5.2); coverage decisions in [[ch-13a]] | 1.3B | Result (single study) |

**Conditions and limits.** The Pile comparison evaluates on the Pile's own components, so the Pile model is tested in-distribution and the CC models out of distribution; the table measures which domains each corpus covers, not general quality. The authors note that the 40GB size control favors CC-100, which is about one third of the Pile's size (§4.1).

**Implication.** On the QA suite of §3, Common Crawl had the largest total effect and books the largest effect per percentage point of data, and non-web sources supply domains (academic papers, code, math, non-English text) that web-only corpora fit worse even at 150B training tokens.

## §3 Controlled removal, data age, and filters

**Setting.** [[pretrainers-guide-training-data]] (Longpre et al., arXiv v1 2023-05) pretrained 28 decoder-only models of 1.5B parameters ("LM-XL") in T5X with batch 4,096, sequence length 512, and 88,064 steps (App. C, Table 5), which is 4,096 × 512 × 88,064 ≈ 184.7B tokens (derived). Each model was fine-tuned on Natural Questions and evaluated on 27 QA datasets grouped by domain (§6, App. C.5). Before the experiments, the authors deduplicated C4 and the Pile (§2.1), so the data shares below differ from Pile Table 1.

**Source removal.** Values from Fig. 8, relative to the model trained on the full deduplicated Pile; "% data" is the share left after removal (Table 3):

| Removed | % data left | Wiki | Web | Books | Biomed | Academic | Common Sense | Contrast Sets | Average |
|---|---|---|---|---|---|---|---|---|---|
| Social | 98.8 | −0.8 | −3.7 | 2.6 | 0.1 | 3.5 | −3.5 | 3.5 | 0.4 |
| Wikipedia | 97.9 | −1.3 | −5.3 | 3.0 | 0.2 | 0.9 | −4.4 | 7.2 | −0.3 |
| Books | 93.1 | −3.5 | −6.3 | 1.0 | 0.0 | −1.6 | −6.5 | −4.4 | −2.7 |
| OpenWeb | 93.1 | −2.0 | −4.1 | 0.1 | −1.0 | 0.6 | −5.8 | −2.9 | −1.4 |
| Legal | 91.0 | −2.7 | −2.9 | 3.8 | 0.4 | 0.8 | −2.6 | −0.4 | −0.6 |
| Academic | 87.1 | −0.3 | −2.5 | 0.3 | −0.9 | 2.2 | −1.1 | 4.3 | 0.2 |
| PubMed | 85.1 | −0.3 | −3.0 | 3.9 | −5.8 | −1.5 | −5.9 | 3.9 | −1.2 |
| Code | 80.9 | −0.5 | −3.1 | 2.9 | −1.2 | 1.2 | −5.8 | 4.4 | −0.1 |
| Common Crawl | 73.1 | −3.2 | −6.2 | −2.9 | −4.6 | −5.9 | −8.0 | −5.2 | −4.8 |

**Worked example: effect per percent of data removed** (derived, assumes a linear effect and uses one run per configuration). Books: −2.7 / 6.9 = −0.39 per percentage point. Common Crawl: −4.8 / 26.9 = −0.18. OpenWeb: −1.4 / 6.9 = −0.20. PubMed: −1.2 / 14.9 = −0.08. Code: −0.1 / 19.1 = −0.005. Per unit of data, books had about twice the effect of Common Crawl on this QA suite, and code had almost none; the authors note that the QA sets do not require coding skills (§6).

**Findings.** Removing Common Crawl lowered Academic QA by 5.9, more than removing the Academic sources (+2.2); the authors hypothesize that Common Crawl, OpenWeb, and Books cover many topics, including academic ones (§6, Interpretation). The best average came from models trained on all or nearly all sources, and the authors recommend including sources that seem unrelated to the target tasks (§6). Toxicity metrics move in a different direction from QA: removing Books lowered toxic generation by 6.2 and also lowered toxicity identification by 1.3 (Table 3).

**Data age.** The authors rebuilt C4 from Common Crawl with cutoffs in 2013, 2016, 2019, and 2022, pretrained one LM-XL per version, fine-tuned each on year-split tasks, and tested on every year (§4). Temporal degradation (TD) is the expected score change for one year of distance between pretraining year and evaluation year. For LM-XL the mean TD over five tasks was 0.41 with Pearson r = 0.61 between score and year distance; for the 20M LM-Small it was 0.08, and the authors did not find the pretraining effect significant at that size (Table 2, §4). Degradation appeared in both directions and was steeper when evaluation data was newer than pretraining data (Fig. 4). Worked example (derived, linear assumption): a three-year gap predicts 3 × 0.41 ≈ 1.2 lower score for LM-XL.

**Filters as composition changes.** A quality classifier (the PaLM/GLaM classifier) at threshold 0.975 kept 91% of C4 and raised the QA average by 2.5; at 0.7 it kept 46% and raised the average by 0.7, while Books QA fell by 6.7 (Fig. 6). A toxicity filter at 0.3 kept 61% and lowered the average by 2.7; the inverse toxicity filter, which removes the least toxic 8%, raised it by 1.7 (Fig. 7). The authors conclude that "Quality filtering effects are not easily predicted by dataset characteristics" (§5). Filter mechanics are in [[ch-10]] and [[ch-10a]].

**Conditions and limits.** One run per configuration, two English datasets, fine-tuned rather than prompted evaluation, and toxicity scores from a black-box API (§8). Panel C of **[figures/data-landscape.html](figures/data-landscape.html)** shows the Fig. 8 table as a heat map with the "% data left" column beside it.

## §4 Code in pretraining

**Definitions.** In a **competitive** mixture, the total token count is fixed, so each code token replaces a natural-language token. In an **additive** mixture, the natural-language token count is fixed and code is added on top ([[code-pretraining-task-performance]] §3).

```
N_code = m · N_total,   N_lang = (1 − m) · N_total        (competitive: N_total fixed)
N_total = N_lang / (1 − m)                                 (additive: N_lang fixed)
```

- m: fraction of training tokens that are code; N_total, N_code, N_lang: total, code, and language token counts.

**Worked example.** With N_lang = 132B (the base volume in Petty et al. §4.1), an additive mixture with m = 0.5 has N_total = 132 / 0.5 = 264B tokens; with m = 0.2 it has 165B tokens, of which 33B are code. A competitive run at m = 0.5 has 66B language tokens. A decline in a knowledge task in the competitive setting can come from fewer language tokens; the additive setting separates that effect.

**Evidence 1: [[to-code-or-not-to-code]]** (Aryabumi et al., Cohere, arXiv v1 2024-08). Text is SlimPajama without GitHub and StackExchange (503B tokens); code is a filtered Stack subset (139B tokens) plus markup and a proprietary 3.2B-token synthetic Python set (§2.1). Rows from Table 2 (the caption does not state the size; the values match the 470M results of §3.1 and §3.6; reasoning is the average of 11 tasks, knowledge the average of TriviaQA and NQ, code the average pass@1 of HumanEval and MBPP):

```
Variant           Recipe                Text    Code   Reason.  Know.  Code
Text-only         Pre-training          400B    -      49.0     9.5    0.4
Balanced-only     Pre-training          200B    200B   51.8     8.1    9.0
Balanced → Text   Pre-training Init.    100B    100B   52.0     7.4    7.8
                  Continue Pre-train.   +180B   +20B   53.0     9.9    4.8
                  Cooldown              +32B    +8B    54.9     10.9   5.8
```

Relative to text-only at the same 400B tokens, balanced→text raised reasoning by (53.0 − 49.0) / 49.0 = 8.2%, knowledge by (9.9 − 9.5) / 9.5 = 4.2%, and code by 4.8 / 0.4 = 12× (§3.6). In the proportion sweep (six models from scratch, 200B tokens), 25% code gave the best average over reasoning and knowledge; knowledge was 3.4% lower than with no code at 25%, 31% lower at 75%, and 86% lower at 100% code; code scores rose almost linearly with code share (§3.3). A 40B-token cooldown with 20% code raised reasoning by 3.6%, knowledge by 10.1%, and code by 20% relative to no cooldown, while a cooldown without code did not raise reasoning or code (§3.5). The trends held at 2.8B (§3.2).

**Evidence 2: [[code-pretraining-task-performance]]** (Petty et al., arXiv v1 2024-09). 374M models, C4 text and GitHub code from the Pile, 5 seeds per mixture (§4.1). Code share improved compositional generalization when the output is a formal structure: for COGS-vf structural generalization the best-fit slope of accuracy against code share was 0.147 (competitive) and 0.165 (additive), and the authors read 0.147 as a predicted 14.7% accuracy increase from 0% to 100% code (§5). Multi-digit arithmetic improved, with a peak at 40-50% code in the competitive setting (Fig. 4). Accuracy fell with code share on English passivization and on BigBench general knowledge, fantasy reasoning, implicatures, and common morpheme, in both settings (Fig. 5-6). Over all BigBench multiple-choice tasks in the competitive setting, the per-task slopes had higher variance (p = 0.0002) and a higher upper quartile (p = 0.006) than under permutation, while the difference of means was not significant (p = 0.1038, Fig. 8); the authors' discussion nevertheless describes an aggregate improvement (§6).

**Evidence 3.** At 4.2B parameters with an 84B-token budget, replacing up to 50% of tokens with Python did not lower the 19-task natural-language average, and bAbI rose from 0.0 with no code to 23.2 with 50% code ([[data-constrained-scaling]] §7, App. M Table 10, 5 seeds). At 1B, raising code from 0% to 5% to 15% raised bAbI in-context exact match from 0.0 to 8.8 to 10.1 ([[dolma]] App. M Table 3).

**Status.** **Replicated:** a moderate code share does not lower natural-language averages and raises state-tracking, arithmetic, and structured-output tasks ([[to-code-or-not-to-code]], [[code-pretraining-task-performance]], [[data-constrained-scaling]], [[dolma]]). **Replicated:** world-knowledge tasks decline as code share grows ([[to-code-or-not-to-code]] §3.3; [[code-pretraining-task-performance]] §5). **Result (single study):** purely linguistic tasks also declined ([[code-pretraining-task-performance]] §5). **Interpretation:** Petty et al. attribute the decline to less exposure to natural language, absolute or relative (§6).

**Conditions and limits.** All models are 2.8B or smaller except the 4.2B data-constrained runs, and all budgets are 440B tokens or fewer including cooldown. Neither Aryabumi et al. nor Petty et al. measure how the code share interacts with SFT or RL ([[code-pretraining-task-performance]] §6), and Aryabumi et al. do not study safety (§6).

**Implication.** For a general model, the code share is a trade between code and structure tasks on one side and knowledge per token on the other; the knowledge cost is visible only when knowledge tasks are reported separately from the natural-language average.

## §5 The Llama 3 mixture and how it was chosen

[[llama-3]] (arXiv:2407.21783, v1 2024-07) describes how its mixture was selected in §3.1.2. The passage, with one part elided:

> "Knowledge classification. We develop a classifier to categorize the types of information contained in our web data to more effectively determine a data mix. We use this classifier to downsample data categories that are over-represented on the web, for example, arts and entertainment. Scaling laws for data mix. To determine the best data mix, we perform scaling law experiments in which we train several small models on a data mix and use that to predict the performance of a large model on that mix (see Section 3.2.1). [...] Data mix summary. Our final data mix contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens."

**Mechanism, step by step.**
1. Filter web data with URL, document, and line deduplication, heuristics, and model-based quality classifiers (a fastText model trained to recognize text referenced by Wikipedia and a DistilRoberta model trained on Llama 2 quality judgments), plus separate classifiers that extract code and math pages (§3.1.1).
2. Label web documents by knowledge category and downsample over-represented categories (§3.1.2).
3. For each candidate mix, train several small models, fit a scaling law, and predict large-model performance; repeat for several mixes (§3.1.2, method in §3.2.1 and [[ch-08a]]).
4. Train a larger model on the selected candidate and check key benchmarks (§3.1.2).
5. Change the mix during the 405B run: more non-English data, upsampled math, more recent web data in later stages, and downsampled lower-quality subsets; percentages are not printed (§3.4.1, [[llama-3-recipe]]).
6. Evaluate small new datasets by annealing a 50%-trained 8B model to learning rate 0 over 40B tokens with 30% weight on the new dataset and 70% on the default mix (§3.1.3).

**Worked example of a unit error.** 17% × 15.6T = 2.65T is not the number of code tokens the 405B model saw, for two reasons stated in the report: the mix changed during training (§3.4.1), and the category shares are "roughly" given for the final mix. The derived number has no source and must not be entered into a recipe table.

**What is not reported.** Per-source shares, the definition of "general knowledge", the knowledge-classifier categories other than arts and entertainment, the results of the mix scaling-law experiments, and the data itself. [[deepseek-v3]] §4.1 states only that the 14.8T-token corpus raises the ratio of math and programming samples relative to DeepSeek-V2 and extends multilingual coverage; mixture percentages are not given.

**Open evidence that topic reweighting changes breadth.** FineWeb-Edu's educational classifier raised "Education, Learning, Teaching" (+3.2%) and "History, Culture, Politics" (+2.2%) and down-sampled "Entertainment, Film, Theater" ([[fineweb]] §4.1). In [[weborganizer]], 1B models trained on 29B tokens from a 200B-token pool, a topic × format mixture targeting MMLU and HellaSwag raised the 9-task average from 51.6 to 54.6 (Table 1); FineWeb-Edu selection lowered HellaSwag to 1.5 below the random-selection baseline, and a mixture tuned only for MMLU lowered HellaSwag from 57.5 to 54.1 and PIQA from 71.3 to 69.9 (Table 10). Sampling FineWeb-Edu's implicit topic × format mixture at random within each domain recovered 84% of its average gain (Table 2). **Interpretation:** a quality classifier acts in part as a topic reweighting, so two independent reports (Llama 3, FineWeb-Edu) that downsample entertainment content have made a composition decision that should be measured on tasks outside the targeted benchmarks.

**Curated sources after model-based web filtering.** In [[dclm]] Table 6 (1B-1x scale), mixing high-quality non-web sources at the Llama 1 / RedPajama ratio (67% Common Crawl, 33% Wikipedia, books, StackExchange, arXiv, GitHub) raised Core by 2.2 for C4, 1.7 for RedPajama-CC, and 1.4 for RefinedWeb, but lowered it by 1.2 for DCLM-Baseline. DCLM Core is 22 tasks chosen for a low-variance signal at small scale, including ARC, HellaSwag, BoolQ, CommonsenseQA, and six BIG-bench tasks (App. G.1); it has no per-domain loss on academic, code, or book text, so this result is measured differently from the per-domain fit in §2. [[nemotron-cc]] kept a fixed 27% block (books and patents 9%, papers 9%, code 5%, conversational 3%, Wikipedia 1%) in all 1T-token 8B comparisons and varied only the 73% English Common Crawl part (§3.1, App. D Table 12). **Open question:** whether curated non-web sources add breadth on top of classifier-filtered web text depends on the measurement suite, and no source above tests both suites on the same runs.

## §6 Open corpora and what each discloses

| Corpus | Size and unit | Sources | Released | Not reported or limits |
|---|---|---|---|---|
| The Pile ([[the-pile]], 2020-12) | 825.18 GiB raw, 1254.20 GiB effective | 22 components | data, per-component weights and epochs (Table 1), preprocessing code | tokens per component; weight ablations |
| Dolma v1.6 ([[dolma]], 2024-01) | 3,059B Llama tokens | Common Crawl 2,479B; GitHub 411B; Reddit 89B; Semantic Scholar 70B; Gutenberg 6.0B; Wikipedia and Wikibooks 4.3B (Table 1) | data (ODC-By), toolkit, filter ablations at 1.2B | sampling rates used to train OLMo-1B |
| FineWeb, FineWeb-Edu ([[fineweb]], 2024-06) | 15T and 1.3T GPT-2 tokens | 96 Common Crawl snapshots; web only | data, datatrove code, classifier, annotations, ablation models | code is likely not prevalent (App. A) |
| DCLM ([[dclm]], 2024-06) | 240T GPT-NeoX token pool; 3.8T DCLM-Baseline | all Common Crawl before 2023 | pool, baseline data, 53-task evaluation, training recipes at 5 scales | run-to-run variation not sufficiently explored; one tokenizer (§6) |
| Nemotron-CC ([[nemotron-cc]], 2024-12) | 6.3T tokens: 4.4T deduplicated real, 1.9T synthetic | 99 Common Crawl snapshots, 2013-20 to 2024-30 | data under the Common Crawl Terms of Use, quality classifiers, NeMo Curator code | comparisons use a fixed 27% non-crawl blend from other datasets (Table 12) |
| OLMo 2 Mix 1124 ([[olmo-2]], 2025-01) | 3.90T tokens | DCLM-Baseline 3.71T; StarCoder 83.0B; peS2o 58.6B; arXiv 20.8B; OpenWebMath 12.2B; Algebraic Stack 11.8B; Wikipedia 3.7B (Table 4) | data, per-source counts, mid-training mixes | ablation of the pretraining mix |
| Llama 3 ([[llama-3]], 2024-07) | 15.6T tokens (405B model) | four categories | category shares, selection method | per-source shares, data |
| DeepSeek-V3 ([[deepseek-v3]], 2024-12) | 14.8T tokens | not listed | direction of change from V2 | mixture percentages |

**Unique versus total tokens.** A corpus's token count includes repeated content unless it was globally deduplicated. [[nemotron-cc]] Table 4 estimates unique tokens after global fuzzy deduplication: DCLM has 3.8T total and 1.0T unique; FineWeb-Edu 1.3T and 0.2T; Nemotron-CC 6.3T total, 4.4T unique real and 1.9T synthetic. Worked example: a 15T-token run drawn only from DCLM would see its 1.0T unique tokens about 15 times; a run drawn only from Nemotron-CC's real tokens would see them about 15 / 4.4 ≈ 3.4 times (derived). [[data-constrained-scaling]] finds that value per repeated token decays beyond about 4 epochs (details in [[ch-14]]). Nemotron-CC contributed 7.2T of the 15T tokens of an 8B model that scored MMLU 70.3 against 65.3 for Llama 3.1 8B in the authors' own evaluation harness (§3.2, Table 6).

**Limits of the stock.** [[epoch-data-stock]] (Villalobos et al., arXiv v1 2022-11, v2 2024-06) estimates the indexed web at 510T tokens (95% CI 130T-2,100T) and Common Crawl at 130T (100T-260T) (Table 1), and projects that training sets will reach the effective stock of public human text, about 4e14 tokens, between 2026 and 2032, with median 2028 (Abstract, §2.5). These estimates use a different tokenizer and counting rule than the DCLM pool, so 240T and 130T are not directly comparable.

## §7 Opt-outs and consent restrictions as measured reductions in coverage

**Definition.** A consent restriction is a machine-readable (robots.txt) or legal (Terms of Service) statement by a website that disallows crawling or AI use. [[consent-in-crisis]] (Longpre et al., arXiv v1 2024-07) counts "restricted tokens" as the tokens of a corpus that come from web domains fully blocking at least one of seven AI organizations' crawlers (§2).

**Setting.** 14,000 web domains underlying C4, RefinedWeb, and Dolma: the top 2,000 domains by tokens in each corpus (3.95k after union, "HEAD") and 10,000 random domains; robots.txt and Terms pages from monthly Wayback Machine snapshots, January 2016 to April 2024 (§2).

**Evidence.**
- Across the full corpora, about 1% of C4, RefinedWeb, and Dolma tokens were robots.txt-restricted in mid-2023 and 5-7% in April 2024; in the head domains the share went from under 3% to 20-33% (§3.1, Fig. 2a).
- Restrictions are concentrated by domain type: nearly 45% of news-website tokens in the head sample were fully restricted, compared with 3% in 2023. The authors expect robots.txt-respecting crawls to shift away from news, social media, and forums toward organization and e-commerce sites (§3.1, Interpretation).
- Restrictions differ by crawler: in the C4 head sample, OpenAI's crawlers were restricted for 25.9% of tokens, Anthropic's and Common Crawl's for 13.3%, Google's AI crawler for 9.8%, Cohere's for 4.9%, and Meta's for 4.1% (Fig. 1).
- Terms of Service pages of 45-55% of tokens contain some data-use restriction, although most crawlers do not read Terms pages (§3.1, Fig. 2c).
- In WildChat conversations labeled with the same taxonomy (100 clustered by the authors, 1,000 labeled by GPT-4o), over 30% requested creative composition such as fiction, role-play, or poetry, which the authors find poorly represented in the web domains of these corpora (§3.4).

**Worked example (derived).** A pipeline that respects robots.txt for all seven organizations and builds its web slice from a C4-like domain distribution loses about 5-7% of web tokens overall. If the composition target includes news from head domains, the loss inside that domain is about 45%, and the remaining news share can only be restored by upweighting what is left, which repeats it.

**Conditions and limits.** The authors state that they make no assertion that an absent restriction implies consent (§3.1); only fully restricted domains are counted, so partial restrictions are higher (§3.1); the WildChat sample comes from one proxy service and may not represent other users (§3.4).

**Implication.** Consent restrictions are a composition change that is not uniform across domains, so a corpus rebuilt under a stricter crawling policy should be re-measured per domain rather than assumed to be a smaller copy of the old corpus.

## §8 Measuring whether a composition change widened coverage

1. **Per-domain loss.** Report macro-averaged perplexity or BPB over domains, `(1/|D|) Σ_{d∈D} ppl(d)`, where D is the set of evaluation domains and ppl(d) the perplexity on domain d, instead of one token-weighted (micro) average, because a micro average is weighted toward the domains that contribute the most evaluation tokens ([[paloma]] §4.1). Decontaminate training data against the evaluation text and score each document separately: concatenated inputs gave perplexity 92.23 ± 17.33 against 42.57 ± 0.29 for separate documents on the same Pythia 1.4B checkpoint (App. H, Table 17).
2. **Broad task suites with a centered score.** DCLM Core averages 22 tasks and Extended 53 tasks after rescaling each task's accuracy a to (a − r) / (1 − r), where r is random-guess accuracy ([[dclm]] §3.5). Worked example: 40% accuracy on a four-choice task gives (0.40 − 0.25) / 0.75 = 0.20, and 25% gives 0. Without centering, a two-choice task at chance (50%) would count more than a four-choice task at 45%, whose centered score is (0.45 − 0.25) / 0.75 = 0.27.
3. **Small-scale decisions that transfer.** In [[dclm]] Fig. 3, Core scores of 10 curation methods at 400M-1x, 1B-1x, and 3B-1x correlate with 7B-1x scores at Pearson r = 0.838, 0.956, and 0.982. In [[datadecide]], ranking 25 corpora with 150M models picks the 1B winner in about 80% of pairwise comparisons, but decisions on SocialIQA and BoolQ are unreliable at the scales tested (§3.1). When a proxy-scale run selects data, rank with tasks that separate the candidate corpora at that scale, because rankings on the other tasks did not predict the 1B winner in [[datadecide]].
4. **Contamination checks.** Removing detected MMLU and HellaSwag overlaps from DCLM-Baseline did not lower scores (MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4 at 7B-2x; [[dclm]] Table 7). Several test sets are fully contained in Dolma, many of them in its code subset ([[dolma]] App. L). Methods are in [[ch-48]].

## Negative samples and negative feedback

This section applies the four meanings of "negative" from the course standard to composition; the full treatment is in [[ch-31a]] and [[ch-43a]]. Pretraining composition uses only meanings 1 and 2. **Negative marginal value (1):** documents removed by a quality or toxicity filter are treated as samples whose inclusion would lower performance. The evidence in §3 shows that the label is task-dependent: the toxicity filter that kept 61% of C4 lowered the QA average by 2.7 and lowered toxicity identification, and the inverse filter that removed the least toxic documents gave the best toxicity identification ([[pretrainers-guide-training-data]] §5). **Negative as content (2):** toxic or low-quality text kept in the corpus is trained with ordinary cross-entropy; it adds the ability to recognize such text and also raises toxic generation (Table 3). No pretraining source in this chapter applies meaning 3 (conditioning tokens) or meaning 4 (a gradient that lowers a document's likelihood), so likelihood displacement does not arise at this stage. Controls: filter with thresholds chosen on a suite that includes recognition tasks, report the share removed per domain, and move behavior suppression to post-training, as the authors suggest (§7). Diagnostics: toxicity identification and toxic generation reported side by side, and per-domain QA change for each filter threshold.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 (all) | 8B-405B | pretrain-stable | final data mix (token share) | roughly 50% general knowledge, 25% mathematical and reasoning, 17% code, 8% multilingual | arXiv:2407.21783v3 §3.1.2 ([[llama-3-recipe]]) | verified 2026-09-14 | small-model scaling-law experiments per candidate mix, then a larger model on the candidate; knowledge classifier downsamples over-represented web categories; no table |
| Llama 3.1 405B | 405B | pretrain-stable | mix changes during training | more non-English data; math upsampled; more recent web data later; lower-quality subsets downsampled; percentages not printed | arXiv:2407.21783v3 §3.4.1 | verified 2026-09-14 (percentages not reported) | no ablation reported |
| Llama 3 8B (experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50%-trained 8B; LR annealed linearly to 0 over 40B tokens; 30% weight on the new dataset, 70% on the default mix | arXiv:2407.21783v3 §3.1.3 | verified 2026-09-14 | stated as more efficient than scaling-law runs per small dataset; no numbers |
| The Pile (corpus) | n/a (data) | pretrain-stable | component weights (share of bytes) and epochs | Pile-CC 18.11% (1.0); PubMed Central 14.40% (2.0); Books3 12.07% (1.5); OpenWebText2 10.01% (2.0); ArXiv 8.96% (2.0); Github 7.59% (1.0); FreeLaw 6.12% (1.5); Stack Exchange 5.13% (2.0); Wikipedia 1.53% (3.0); 13 smaller components in Table 1 | arXiv:2101.00027v1 Table 1 | verified 2026-09-15 | "Following Brown et al. (2020)" for higher-quality components (§2); no ablation reported |
| OLMo 2 7B / 13B / 32B | 7B, 13B, 32B | pretrain-stable | OLMo 2 Mix 1124 (tokens) | DCLM-Baseline 3.71T; StarCoder 83.0B; peS2o 58.6B; arXiv 20.8B; OpenWebMath 12.2B; Algebraic Stack 11.8B; Wikipedia & Wikibooks 3.7B; total 3.90T | arXiv:2501.00656v3 §2.4.1 Table 4 | verified 2026-09-15 | no ablation of the pretraining mix reported |
| OLMo 2 7B / 13B / 32B | 7B, 13B, 32B | pretrain-stable | tokens in pretraining stage | 3.90T (7B); 5T (13B); 6.06T (32B) | arXiv:2501.00656v3 §2.3 "Overall" | verified 2026-09-15 | no ablation reported; Table 9 caption gives 4T, 5T, 7T for the same checkpoints (conflict within the report) |
| DCLM-Baseline data | n/a (data) | pretrain-stable | model-based filter | fastText classifier, positives OpenHermes 2.5 + r/ExplainLikeImFive, negatives from RefinedWeb reproduction; keep top 10% | arXiv:2406.11794v4 §4.4 | verified 2026-09-15 | Table 5 (7B-1x): Core 41.0 at 10% vs 39.8 at 15% vs 38.7 at 20%; vs Wikipedia positives 35.7 |
| DCLM mixing ablation | 1.4B (1B-1x, 28.8B tokens) | pretrain-stable | mixture | 67% Common Crawl subset, 33% Wikipedia, books, StackExchange, arXiv, GitHub (Llama 1 / RedPajama ratios) | arXiv:2406.11794v4 §4.5, Table 1 | verified 2026-09-15 | Table 6: Core +2.2 (C4), +1.7 (RPJ CC), +1.4 (RefinedWeb), −1.2 (DCLM-Baseline) |
| DCLM 7B (final) | 6.9B | pretrain-stable; pretrain-decay/anneal; long-context | data and stages | 4.1T dataset (DCLM-Baseline 3.8T + StarCoder + ProofPile2); 2.5T tokens; two cooldowns of 200B and 270B tokens on 70% DCLM-Baseline with tighter fastText threshold + 30% math, souped; 100B tokens to extend context 2048 → 8192 | arXiv:2406.11794v4 §5 | verified 2026-09-15 | Table 8 (listed as 2.6T tokens): Core 57.1, MMLU 63.7, Extended 45.4; no ablation of the 70/30 cooldown split reported |
| Nemotron-CC comparison models | 8B | pretrain-stable | blend (share of 1T tokens) | English Common Crawl 73% (the tested dataset); books and patents 9%; papers 9%; code 5%; conversational 3%; Wikipedia 1% | arXiv:2412.02595v2 §3.1, App. D Table 12 | verified 2026-09-15 | Table 5: MMLU 59.0 (Nemotron-CC-HQ) vs 53.4 (DCLM) vs 42.9 (FineWeb-Edu); the non-crawl 27% was not varied |
| Nemotron-CC long-horizon model | 8B | pretrain-stable, two phases | English Common Crawl share by phase | phase 1: 9T tokens, 59% English Common Crawl (5.31T); phase 2: 6T tokens, 31% (1.86T); total 47.8% (7.17T) of 15T | arXiv:2412.02595v2 App. E (§3.2 rounds to 7.2T) | verified 2026-09-15 | Table 6: MMLU 70.3 vs 65.3 for Llama 3.1 8B in the authors' harness; phase shares not ablated in this paper (curriculum from Feng et al. 2024) |
| To Code models | 470M | pretrain-stable | code share, from scratch | 0%, 25%, 50%, 75%, 90%, 100% of 200B tokens | arXiv:2408.10914v1 §3.3 | verified 2026-09-15 | 25% best average of reasoning and knowledge; knowledge −3.4% (25%), −31% (75%) relative to 0% |
| To Code balanced → text | 470M | pretrain-stable, then pretrain-decay | tokens by stage | init 100B text + 100B code; continue +180B text, +20B code; cooldown +32B text, +8B code with linear LR anneal to 1e-6 | arXiv:2408.10914v1 Table 2, §3.5 | verified 2026-09-15 | Table 2: reasoning 54.9, knowledge 10.9 after cooldown vs text-only cooldown 54.1, 11.1 |
| Data-constrained code runs | 4.2B | pretrain-stable | Python share of 84B tokens | 0-90% in steps of 10% (The Stack) | arXiv:2305.16264 §7, App. M Table 10 ([[data-constrained-scaling]]) | verified 2026-09-14 | up to 50% code: no drop in 19-task average (Fig. 6, 5 seeds) |
| Dolma mixture ablation | 1B | pretrain-stable | token share web / code / reference / books | Web Only 100 / 0 / 0 / 0%; Reference+ 81.2 / 13.5 / 4.9 / 0.4%; Gopher-like 68.4 / 5.4 / 24.2 / 2.0% | arXiv:2402.00159v2 App. M Table 4 ([[dolma]]) | verified 2026-09-14 | Fig. 12: Web Only higher perplexity on HumanEval and S2ORC; Reference+ ≈ Gopher-like on S2ORC |
| Pretrainer's Guide LM-XL | 1.5B | pretrain-stable | batch; sequence length; steps | 4,096; 512; 88,064 (≈ 184.7B tokens, derived) | arXiv:2305.13169v2 App. C Table 5 | verified 2026-09-15 | defaults adopted from Wang et al. (2022); no ablation reported |

**Starting point for a small general-purpose run.** For a from-scratch model between 470M and 4.2B parameters trained on 84B-200B tokens, a code share between 25% and 50% of tokens is inside the tested range: 25% gave the best reasoning-plus-knowledge average at 470M on 200B tokens, and 50% Python did not lower the 19-task average at 4.2B on 84B tokens. Report knowledge tasks separately, because they were 3.4% lower at 25% code in the 470M runs. Keep a fixed non-crawl block; the two verified sizes are the 27% block of the 8B, 1T-token Nemotron-CC runs (books and patents 9%, papers 9%, code 5%, conversational 3%, Wikipedia 1%) and the 33% block of the 1.4B DCLM mixing runs, and the DCLM run shows that such a block can lower Core when the web portion is DCLM-Baseline. Choose between candidate web subsets with 1B-1x runs (28.8B tokens) scored on a centered multi-task suite, as in the DCLM mixing and filtering rows; §8 gives how DCLM's 1B-1x rankings transferred to 7B-1x. To judge a small new source in a later-stage mix, use the Llama 3 procedure: anneal a partly trained model over 40B tokens with 30% weight on the candidate (stated for an 8B model trained to 50%). No verified row gives a multilingual or math share for a model below 8B.

## Generalization lens

**(a) What increases breadth.**
- Heterogeneous sources: training on the 22-source Pile gave lower BPB than CC-100 on all 22 components at 1.3B ([[the-pile]] Table 4), replicated on Paloma domains at 1.2B ([[dolma]] §9.2) and 1B ([[paloma]] §4.1).
- Keeping all or nearly all sources: the best QA average came from models trained on all or nearly all Pile sources ([[pretrainers-guide-training-data]] §6).
- A code share of 25% at 470M: reasoning 3.4% higher and knowledge 3.4% lower than with no code, against knowledge 31% lower at 75% code ([[to-code-or-not-to-code]] §3.3); 50% at 4.2B kept the natural-language average ([[data-constrained-scaling]] §7).
- Data close in time to the evaluation period: mean temporal degradation 0.41 per year at 1.5B, not removed by fine-tuning ([[pretrainers-guide-training-data]] Table 2).

**(b) What causes narrowing or forgetting.**
- Removing heterogeneous sources: Common Crawl removal −4.8, Books −2.7 on the QA average ([[pretrainers-guide-training-data]] Fig. 8).
- High code shares: world-knowledge scores 31% lower at 75% code and 86% lower at 100% ([[to-code-or-not-to-code]] §3.3); lower accuracy on linguistic and knowledge tasks as code share grows ([[code-pretraining-task-performance]] §5).
- Filters and mixtures tuned to one benchmark: an MMLU-targeted mixture lowered HellaSwag by 3.4 and PIQA by 1.4 ([[weborganizer]] Table 10); a perplexity filter that discards documents too similar to or too different from Wikipedia gave CC-100 worse Pile BPB than raw CC filtered only for English ([[the-pile]] §4.2, Interpretation by the authors); toxicity filtering lowered QA average and toxicity identification ([[pretrainers-guide-training-data]] §5).
- Consent-driven source loss that is uneven by domain: about 45% of head-domain news tokens restricted in April 2024 ([[consent-in-crisis]] §3.1).

**(c) How to measure it at this stage.** Report macro-averaged per-domain loss on a decontaminated suite ([[paloma]]); a centered multi-task suite such as DCLM Core and Extended ([[dclm]] §3.5); knowledge, reasoning, code, and generation scores separately ([[to-code-or-not-to-code]] §2.2); and year-split tasks for data age ([[pretrainers-guide-training-data]] §4). Known measurement errors: in-distribution evaluation favors the corpus whose components define the test set ([[the-pile]] Table 4); tasks near chance at the proxy scale give unreliable rankings ([[datadecide]] §3.1); a lab's re-evaluation of another lab's model may differ from the original numbers ([[nemotron-cc]] Table 6 note); concatenated evaluation documents inflate variance ([[paloma]] App. H).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reading raw-size share as mixture weight | Computed shares do not sum to 100%, or disagree with the report's weight column | Recompute w_i = S_i e_i / Σ S_j e_j and compare with the printed weights (Pile Table 1) |
| Mixing byte, token, and document shares | Code or math appears smaller or larger than in the source report | Record the unit and tokenizer next to every share; convert with bytes per token per source |
| Multiplying a final category share by total tokens | A "code tokens seen" number appears that no report prints | Check whether the report states the mix changed during training (Llama 3 §3.4.1); mark the value not reported |
| Judging a composition change by one average | Average rises while a domain or task group falls | Report per-domain loss and per-group scores (knowledge, reasoning, code) before and after |
| Evaluating only on in-distribution domains | The corpus that defines the test components always wins | Add evaluation domains that are not components of any compared corpus |
| Treating a filter as quality-only | Topic shares shift after filtering; entertainment, fiction, or news drops | Run a topic and format classifier on kept and removed documents ([[weborganizer]]) |
| Adding code without a knowledge check | Natural-language average flat, TriviaQA or NQ lower | Track world-knowledge tasks separately across code shares |
| Assuming small-scale rankings transfer on every task | Proxy ranking flips at the target scale | Use tasks above chance at the proxy scale; check rank correlation on a held-out pair of scales |
| Rebuilding a crawl under new consent rules and reusing old mixture weights | Domain shares differ from the previous corpus; news or forum evaluations fall | Measure per-domain token shares of the rebuilt corpus before training |

## Check your understanding

1. The Pile model beat the CC-100 model by 0.09 bits on Pile-CC and by more than 1.0 bit on ArXiv. Explain why the gap differs by component and why this comparison cannot show that the Pile is better in general.
2. Removing the Academic sources raised Academic QA by 2.2, while removing Common Crawl lowered it by 5.9. Give a causal account consistent with the authors' hypothesis, and name an experiment that would test it.
3. In a competitive code mixture, a world-knowledge score falls as code share rises. Explain two mechanisms that could produce the decline and how the additive setting separates them.
4. Llama 3's knowledge classifier downsamples arts and entertainment. Using the FineWeb-Edu and WebOrganizer results, explain which evaluations could fall and why an MMLU-based mix search would not detect it.
5. Adding curated sources raised Core for C4 but lowered it for DCLM-Baseline. Propose two explanations that involve what the fastText filter already selects and what the Core suite measures.
6. Temporal degradation persisted after fine-tuning on year-matched data and was larger at 1.5B than at 20M. Explain what this implies for the order of data freshness decisions in the pipeline.
7. A crawl that respects new robots.txt restrictions loses 5-7% of tokens overall but about 45% of head-domain news tokens. Explain why upweighting the remaining news to restore its share changes both repetition and coverage.

## Connections

- Depends on and previous in order: ch-08b — Why Next-Token Pretraining Produces General Ability: In-Context Learning, Emergence, and Predictability.
- Next: ch-10 — Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb (what heuristic filters remove from each source).
- ch-00 — What General Capability Means and How It Is Measured (course-wide definition of coverage and held-out measurement).
- ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability (token budget and the scaling-law method behind Llama 3's mix search).
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (FineWeb-Edu, DCLM fastText, and topic shifts from §5).
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic (unique versus total tokens in §6).
- ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale (searching for mixture weights).
- ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (the multilingual share).
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (repetition limits and the data stock).
- ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (full recipe rows).
- ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement (applies §8).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (cooldown and annealing mixtures).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (long-document sources such as books).
- ch-48 — Contamination Detection and Its Effect on Reported Scores.
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.

## Sources

- [[the-pile]] — Table 1 component sizes, weights, and epochs; BPB definition; Table 3 size-controlled comparison; Table 4 per-component BPB; English share. The library card does not carry these numbers; use the chapter excerpt.
- [[pretrainers-guide-training-data]] — 1.5B source-removal results (Fig. 8, Table 3), data age (Table 2, Fig. 4), quality and toxicity filters (Fig. 6-7), training settings (Table 5).
- [[to-code-or-not-to-code]] — code proportion sweep, initialization variants (Table 2), cooldown with code, 2.8B scaling.
- [[code-pretraining-task-performance]] — competitive and additive mixtures, compositional generalization slopes, arithmetic, knowledge and linguistic declines, permutation tests.
- [[data-constrained-scaling]] — Python share up to 50% at 4.2B without natural-language loss; repetition limits.
- [[llama-3]], [[llama-3-recipe]] — §3.1.2 knowledge classifier, mix scaling laws, final mix; §3.1.3 annealing evaluation; §3.4.1 mix changes.
- [[deepseek-v3]] — closed-report comparison: 14.8T tokens, no mixture percentages.
- [[dolma]] — Dolma v1.6 source sizes, Paloma domain fit, mixture and code-share ablations, contamination.
- [[fineweb]] — FineWeb and FineWeb-Edu sizes and releases; topic shift from the educational classifier.
- [[dclm]] — DCLM-Pool, fastText filter ablations, mixing with curated sources, Core/Extended metric, cross-scale correlation, decontamination, final 7B recipe.
- [[nemotron-cc]] — unique versus total tokens, 73/27 blend, 15T-token 8B comparison.
- [[olmo-2]] — OLMo 2 Mix 1124 composition and stage token counts (use the chapter excerpt for verified values).
- [[weborganizer]] — topic × format mixtures, implicit mixtures of quality filters, MMLU-targeted narrowing.
- [[paloma]] — macro-averaged per-domain perplexity, Common-Crawl-only baselines, evaluation guidelines.
- [[datadecide]] — small-scale decision accuracy across 25 corpora and its task dependence.
- [[doremi]] — evidence that the Pile's default weights were not optimal at 8B.
- [[consent-in-crisis]] — robots.txt and Terms of Service restrictions over time, by domain type and by crawler; WildChat use mismatch.
- [[epoch-data-stock]] — stock estimates for Common Crawl, indexed web, and effective public text.
