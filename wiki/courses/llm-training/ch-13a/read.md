<!-- chapter: ch-13a
     track: pretraining
     kind: content
     title: Multilingual Coverage and Vocabulary as Capability Axes
     deps: [ch-13]
     sources: [[multilinguality-curse-250-languages]], [[atlas-multilingual-scaling-laws]], [[fineweb-2]], [[uberweb-multilingual-curation]], [[vocabulary-scaling-laws]], [[tokenizer-language-unfairness]], [[apertus]], [[smollm-3]], [[smollm3-multilingual-configs]], [[llama-3]], [[llama-3-recipe]], [[qwen-2.5]], [[lfm2-liquid]], [[downstream-scaling-laws-translation]]
     figures: figures/language-budget-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 13a — Multilingual Coverage and Vocabulary as Capability Axes

> **Core insight.** Adding languages to a model of fixed size moves per-language performance in two directions. In 29.5M-parameter models (45.8M with the added vocabulary), 1B tokens from related languages made a language with 1M tokens of text perform like one with about 22% more data, and made a language with 1B tokens perform like one with 63% less data ([[multilinguality-curse-250-languages]] Fig. 1, §6.2). ATLAS fits this trade-off over 774 runs up to 8B parameters: serving 4× as many evenly sampled languages at unchanged loss needs 1.4× the parameters and 2.74× the total tokens, which is 32% fewer tokens per language ([[atlas-multilingual-scaling-laws]] §5). The size of the loss also depends on per-language data quality and on how many tokens each language needs for the same content: curating English data raised non-English scores in 12 of 13 bilingual 3B runs ([[uberweb-multilingual-curation]] §4.1.1), and the same FLORES-200 sentences need 1.48× (Portuguese) to 15.05× (Shan) as many cl100k_base tokens as English ([[tokenizer-language-unfairness]] Table 1).
>
> **Guideline.** When languages are added to a model that must keep per-language quality, scale parameters and total tokens with the ATLAS multipliers (r^0.243 and r^0.728 for K → rK languages) instead of holding tokens per language fixed, because the fit attributes part of the requirement to positive transfer ([[atlas-multilingual-scaling-laws]] §5); the fit assumes uniform sampling and measures loss only, so confirm skewed mixtures with per-language task evaluations. When a target language has little text (1M tokens in [[multilinguality-curse-250-languages]]), add data from syntactically similar languages and check that the model has spare capacity, because the gain was larger for similar languages (+33% vs +22% equivalent data) and for the larger model (+33% vs +12%) (§6.1). When curating non-English web text, derive language-identification and filter thresholds from each language's own statistics and test the pipeline on languages that were not used to design it, because the default FineWeb English thresholds either ranked below the selected adapted method (4.22 vs 2.22 for Gopher Repetition filters) or were excluded for removing over 75% of data in at least one language or removing nothing ([[fineweb-2]] Table 25). When choosing a tokenizer, measure the token premium and a cross-language inequality statistic on parallel text for every language the model must serve, because the vocabulary scaling law was fitted on English data only ([[vocabulary-scaling-laws]] App. B.4). Otherwise, record language shares as "not reported" rather than assuming a value.

## Why this chapter matters for a general-purpose model

The training pipeline is pre-training → mid-training → SFT → preference optimization → RL → evaluation. Language coverage is decided in pre-training: the tokenizer is fixed before the first step (ch-11), and the share of each language is part of the data mixture (ch-13). Later stages inherit both. Context extension (ch-32b) inherits the tokenizer's cost per language, and multilingual SFT (ch-30b) adds small shares on top of what pre-training built.

A **language share** is the fraction of training tokens (or the sampling weight) assigned to text in one language. A language is **high-resource** or **low-resource** according to how much text exists for it; the studies below use token counts (for example 1B vs 1M tokens in [[multilinguality-curse-250-languages]]). **Cross-lingual transfer** is an improvement in language A caused by training on language B. **Interference** is a degradation in A caused by training on B.

This chapter treats five measurable problems:
1. At fixed model size, per-language held-out loss changes when languages are added (§1).
2. The number of parameters and tokens needed to add languages without raising per-language loss (§2).
3. Curation thresholds designed on English keep or remove the wrong documents in other languages (§3).
4. The same content costs a different number of tokens in each language, which changes training tokens, context capacity, and inference cost (§4).
5. English-only or tokenizer-dependent evaluations do not detect regressions in other languages (§5).

The coverage map of ch-00 lists multilingual ability as one capability area, and ch-09 lists multilingual text as one source type. This chapter measures how coverage in that area trades against capacity, data quality, and token cost.

## §1 The curse of multilinguality: interference and transfer at fixed capacity

**Definition.** The **curse of multilinguality** is the observation that per-language performance of a model of fixed size decreases when it is trained on more languages ([[multilinguality-curse-250-languages]] §2, citing Conneau et al. 2020).

**Problem.** Whether added languages help or hurt a given language depends on how much data that language has, how similar the added languages are, and model size. Prior multilingual models used one fixed language distribution each, so these factors were not separated (§1).

**Mechanism of the controlled study** ([[multilinguality-curse-250-languages]] §3-§5):
1. Collect text for 1,572 languages (41.4B tokens); keep the 252 languages with at least 1.5M tokens (1M for training, 500K held out).
2. Train one SentencePiece tokenizer with a 32K vocabulary per language and keep it fixed for every model evaluated on that language, so perplexities for one language are comparable across models.
3. Train 1,989 monolingual GPT-2 baselines at 4.6M, 11.6M, and 29.5M parameters on 1M, 10M, 100M, and 1B tokens.
4. Fit, for each language and size, a curve from monolingual data size to held-out log-likelihood.
5. Train 8,454 multilingual models that add 10M, 100M, or 1B tokens from the 10 most similar or 10 least similar languages, and express each model's score as the monolingual data size that would give the same log-likelihood.

**Formulas.**

```
relative log-likelihood  ℓ = mean_w log2 P_M(w) − mean_w log2 P_Baseline_L(w)          (Eq. 1)
fitted curve             ℓ(x) = −a · x^(−b) + c,   x = log10(monolingual tokens)       (§4.3)
```

- `w`: a token of the held-out set of language L; `P_M(w)`: probability model M assigns to w given its context.
- `Baseline_L`: the 4.6M-parameter model trained on 1M tokens of L.
- `a, b, c`: fitted per language and model size; `a` is fixed to the median across languages when a language has too few data sizes to fit it.

**Worked example.** A baseline assigns a mean of −10.0 bits per token and model M assigns −9.5 bits. Then ℓ = −9.5 − (−10.0) = 0.5. The perplexities are 2^10 = 1,024 and 2^9.5 = 724, a ratio of 1.41 = 2^0.5. Suppose M is a 29.5M-parameter model trained on 1M tokens of L plus 1B tokens of similar languages, and the monolingual curve for that size reaches the same ℓ at 1.2M tokens. M's score is reported as 1.2M estimated monolingual tokens, which is the +20% gain shown in the paper's Fig. 3 caption.

**Evidence** ([[multilinguality-curse-250-languages]], 252 languages, held-out log-likelihood, Result, single study):

| Target language data | Added data | Model | Effect in equivalent monolingual data | Locus |
|---|---|---|---|---|
| 1M tokens | 10 similar languages, "optimal scenario" (added amount not stated in the §6.1 text) | 29.5M (45.8M with added vocabulary) | +33% (dissimilar: +22%) | §6.1 |
| 1M tokens | same optimal scenario | 4.6M vs 29.5M | +12% vs +33% | §6.1 |
| 1M tokens | 1B tokens, 10 similar languages | 29.5M | +22% (Fig. 1 caption); Fig. 3 caption: 1.2M equivalent tokens | Fig. 1, Fig. 3 |
| 1M tokens | 10M tokens (1M per language) | all sizes | "essentially unaffected" | §6.1 |
| 10M tokens | 1B tokens | all but the largest | lower (p < 0.001) | §6.1 |
| 1B tokens | 1B tokens | 29.5M | −63% | §6.2 |
| 1B tokens | 1B tokens | 4.6M | −88% | §6.2 |

Similarity of the added languages explains the low-resource gains mainly through syntax: syntactic similarity accounts for 24.2% of the variance in relative log-likelihood, and adding geographic and vocabulary similarity raises this to 26.4% (§6.1, Fig. 4). For high-resource targets, more similar added languages gave degradations the authors call "slightly larger" (p < 0.05 in 7 of 12 scenarios) (§6.2).

**Conditions and limits.** Models have at most 45M parameters, sequence length 128, and batch size 128 (App. A.3). Each added language set has exactly 10 languages drawn from the 48 languages with at least 100M tokens. The metric is language-modeling likelihood, not task accuracy (§7.1). The authors attribute the degradations to capacity because they shrink with model size (§6.2, **Interpretation**).

**Implication for a general-purpose model.** At small scale, multilingual data acts as extra data for a data-poor language and as a capacity cost for a data-rich one. A model that must keep strong English while adding languages needs either more capacity or a measurement showing that the added languages did not lower English.

## §2 Multilingual scaling laws: ATLAS

ATLAS (Adaptive Transfer Scaling Law) fits loss for a target language as a function of model size and three kinds of tokens: target-language tokens, tokens from the languages most often co-sampled with it, and all other tokens ([[atlas-multilingual-scaling-laws]] §3). It is fitted on 774 runs from 10M to 8B parameters on MADLAD-400, with loss measured on per-language held-out sets using the vocabulary-insensitive loss of [[vocabulary-scaling-laws]] (§2, Table B.1).

### §2.1 Effective data with transfer and repetition

```
L(N, D_eff) = E + A / N^α + B / D_eff^β
D_eff = S_λ(D_t; U_t) + Σ_{i∈K_t} τ_i · S_λ(D_i; U_i) + τ_other · S_λ(D_other; U_other)
S_λ(D; U) = D                                    if D ≤ U
S_λ(D; U) = U · [1 + (1 − exp(−λ(D/U − 1))) / λ]   if D > U
```

- `N`: parameters; `E`: irreducible loss; `A, B, α, β`: fitted constants.
- `D_t`: tokens seen in target language t; `U_t`: unique tokens available in t; `λ`: repetition parameter shared across sources.
- `K_t`: the 3 languages most co-sampled with t; `τ_i`, `τ_other`: fitted transfer weights; `D_other`: all remaining tokens.

**Worked example (λ = 1 is an illustrative value, not the fitted one).** A language has U = 10B unique tokens and training sees D = 30B tokens. D/U = 3, so S = 10B · [1 + (1 − e^(−2))] = 10B · 1.865 = 18.6B effective tokens. With λ = 1, no amount of repetition exceeds 2U = 20B. The repetition term matters for the languages in ATLAS with few unique tokens: Hindi 7.9B and Swahili 770M, against 2.8T for English (Table C.1).

**Evidence.** On held-out language mixtures, the full law reaches R²(M) = 0.82 against 0.61 for the Chinchilla law; adding the other-language term raises overall R² from 0.70 to 0.98 (Table 1, Result, single study).

### §2.2 Measuring transfer between two languages

```
BTS_{s→t} = −( σ_bi(L_t(d_mono)) − 2 · d_mono ) / d_mono
```

- `d_mono`: reference token count of the monolingual model of t (42B tokens).
- `L_t(d)`: loss of the monolingual model of t after d tokens.
- `σ_bi(ℓ)`: total tokens a 50/50 bilingual (s, t) model needs to reach loss ℓ on t.
- BTS = 0: no transfer; BTS > 0: positive transfer; BTS < 0: interference.

**Worked example.** The bilingual model sees half its tokens in t. If language s neither helps nor hurts, it needs 2 × 42B = 84B total tokens to match the monolingual model at 42B, and BTS = −(84 − 84)/42 = 0. If it needs 63B, BTS = −(63 − 84)/42 = +0.5. If it needs 105B, BTS = −0.5.

**Evidence** (2B models, 38 × 38 matrix; 80 directly measured pairs in §4, 90 in App. B.6; the rest predicted by a random forest with cross-validated R² = 0.85):
- English is a top-5 source language for 19 of 30 targets, French for 16, Spanish for 13.
- Mean BTS is −0.23 for pairs that share a script and −0.39 for pairs that do not; script and family effects have p < .001.
- Transfer is asymmetric: the correlation between A→B and B→A is r = −0.11.
- Urdu and Pashto show negative transfer with all other languages.
- Larger models move interfering pairs toward zero (App. C.2).

### §2.3 The number of languages as a scaling variable

```
L(K, N, D_t) = L∞ + A · K^ϕ / N^α + B · K^ψ / D_t^β,     D_tot = K · D_t      (Eq. 7)
```

- `K`: number of training languages, sampled evenly; `D_t`: tokens per language; `D_tot`: total tokens.
- `ϕ`: how capacity demand grows with K; `ψ`: how data demand per language changes with K (ψ < 0 means positive transfer).
- All-language fit: ϕ = 0.11, ψ = −0.04, R² ≥ 0.87; mixtures of 4 to 50 languages, 11 model sizes, 120 runs (§5, Tables B.1, B.5).

Holding the loss fixed while K becomes rK gives, term by term, `N′/N = r^(ϕ/α)`, `D_t′/D_t = r^(ψ/β)`, `D_tot′/D_tot = r^(1+ψ/β)`, and `C′/C = r^(1+ϕ/α+ψ/β)` (§5, App. B.7). The paper's numeric frontier (Fig. 5) is:

| Languages | Total tokens | Parameters |
|---|---|---|
| 2K | ×1.66 | ×1.18 |
| 4K | ×2.74 | ×1.4 |
| 8K | ×4.54 | ×1.66 |
| 16K | ×7.52 | ×1.96 |

**Worked example.** A 2B model trained on 400B tokens serves K = 8 languages, 50B tokens each. Moving to 32 languages (r = 4) at unchanged per-language loss needs 2B × 1.4 = 2.8B parameters and 400B × 2.74 = 1,096B tokens. Tokens per language fall to 1,096B / 32 = 34.25B, which is 50B × 0.685, the "32% less data per language" of §5. Compute rises by 1.4 × 2.74 = 3.84×. [figures/language-budget-explorer.html](figures/language-budget-explorer.html) (panel A) lets the reader change K, r, N, and D and see the four multipliers; it uses the rounded exponents, so it prints 2.743 where the paper prints 2.74.

**Evidence status.** Result, single study. Across K, N, and D, the number of languages changes relative loss more than N or D does, and larger N offsets it more than larger D (§5, Fig. 4).

### §2.4 Pretraining from scratch versus continuing a multilingual checkpoint

ATLAS compares, at 2B parameters, pretraining a monolingual model from scratch with continuing from a multilingual checkpoint trained with Unimax sampling (English 5%, most other high-resource languages 1.42%; Table B.4). Continuing from the checkpoint gives lower loss early; training from scratch overtakes it after 144B (English), 145B (Japanese), 148B (Portuguese), 160B (German), 189B (Russian), 234B (Vietnamese), 239B (Spanish), and 283B (Chinese) target tokens (§6, Fig. 6). The checkpoint's training length is printed as 1B tokens in §6 and 1T in App. B.4 (conflict in the source). The crossover depends on the base mixture and its training length (§6).

**Conditions and limits for §2.** Uniform language sampling in the K law; no repetition term in Eq. 7; models up to 8B; MADLAD-400 only; a 64K SentencePiece vocabulary; loss only, with no downstream accuracy ([[atlas-multilingual-scaling-laws]], card section "Findings relevant to generality"). A multilingual vocabulary and Unimax training shifted the compute-optimal frontier upward relative to monolingual vocabulary and training, most for English (§3, Fig. 1).

**Implication for a general-purpose model.** Language coverage has a cost in parameters and tokens. A plan that adds languages while keeping parameters and total tokens fixed accepts a per-language loss increase that ATLAS can estimate for evenly sampled languages.

## §3 Per-language curation

### §3.1 FineWeb2: thresholds derived from each language's statistics

**Definition.** A language-adaptive pipeline sets its language-identification (LID) threshold, heuristic filter thresholds, and upsampling weights from statistics of the language being processed ([[fineweb-2]] §4).

**Problem.** A threshold that suits English removes too much or too little text in another language. The pipeline was designed on nine "canary" languages (Arabic, Chinese, French, Hindi, Russian, Swahili, Telugu, Thai, Turkish) and tested on five languages not used in its design (§3, §5).

**Mechanism.**
1. Identify the language with GlotLID (1,880 languages) and keep a document if its confidence exceeds a per-language threshold (§4.2).
2. Deduplicate globally within each language with MinHash, storing the duplicate-cluster size of each kept document (§4.3).
3. Apply heuristic filters whose thresholds are adapted from Wikipedia or Common Crawl statistics of the language (§4.4.2).
4. For low-resource languages, remove documents that lack words with high affinity to the language, and restore pages from an allowlist of in-language URLs (§4.4.3).
5. Upsample kept documents by a weight set from the filter removal rate of their duplicate-cluster size (rehydration, §4.5).

**Formula.**

```
LID threshold = max{ 0.3, min{ 0.9, Med(X) − σ(X) } }
```

- `X`: the distribution of LID confidence scores for documents labeled with the language; `Med`: median; `σ`: standard deviation. The Table 15 section of the appendix says "mean" instead of median (App. A.6.2).

**Worked example.** If a language has Med(X) = 0.85 and σ(X) = 0.20, the threshold is 0.65. If a noisier language has Med(X) = 0.55 and σ(X) = 0.35, Med − σ = 0.20, which the lower bound raises to 0.3.

**Evidence** ([[fineweb-2]], 1.46B models, aggregate score over normalized tasks; Result, single study):
- LID: Swahili performed best near a threshold of 0.3, which removes almost 65% of its documents (§4.2). The formula fell outside the best range for Chinese (0.7415 vs 0.895-0.937) and Hindi (0.6827 vs 0.483-0.557) (Table 15).
- Filter thresholds: 207 ablation models at 29B tokens; the selected adapted methods ranked 3.00, 3.22, and 2.22 for the FineWeb Quality, Gopher Quality, and Gopher Repetition filter groups, against 7.00, 6.33, and 6.22 for no filtering (lower is better). The default FineWeb English thresholds ranked 4.22 for Gopher Repetition; for the other two groups they were not ranked, because they removed over 75% of data in at least one language or removed nothing (Table 25).
- Precision filtering, native-speaker audit of 2,000 documents per language (Table 28):

| Language-script | Precision before | Precision after | Recall |
|---|---|---|---|
| glk_Arab | 2.10% | 27.21% | 95.24% |
| bar_Latn | 69.45% | 94.90% | 97.77% |
| ary_Arab | 1.75% | 4.14% | 88.57% |

- Full pipeline versus LID-only data at 350B tokens: Arabic 21.7 → 25.2, French 18.3 → 23.6 (App. A.8, Tables 29-30).
- Held-out languages at 100B tokens, FineWeb2 vs the best other dataset: German 17.0 vs 17.1 (HPLT2), Indonesian 20.3 vs 22.4 (HPLT2), Italian 18.6 vs 17.8 (HPLT2), Japanese 21.8 vs 18.0 (CulturaX), Vietnamese 21.1 vs 19.7 (HPLT2) (Tables 40-44).

**Conditions and limits.** Every ablation model is monolingual, so the pipeline's effect inside a multilingual mixture was not measured (§3). In 70% (1,320) of 1,868 language-script pairs, more than half the documents come from Bible- or Wikipedia-related domains (App. A.12). Coverage of a language in the dataset therefore does not mean coverage of topics in that language.

### §3.2 ÜberWeb: data quality as a cause of measured interference

[[uberweb-multilingual-curation]] (DatologyAI, a data-curation company describing its own pipeline, which is not released) trains 3B models on 60B tokens with a 50/50 mixture of English and one of 13 other languages, under three regimes: both halves uncurated, English curated, and both curated (§4.1.1). "Uncurated" means random samples from DCLM and FineWeb2.

**Evidence** (Result, single study; multilingual MMLU, ARC, Belebele, cloze format):
- Curating English raised non-English scores in 12 of 13 languages (not Bengali), by 3.91% relative on average (§4.1.1).
- The gain was 8.56% for Spanish, French, and German and 3.94% for Hindi and Arabic; it correlated with distance to English (embedding distance r = −0.62, p = 0.024; log perplexity under an English-only model r = −0.70, p = 0.018) (§4.1.2).
- Per-language curation of the non-English half gave 16.87% relative gain over the uncurated baseline (§1).
- Curating the non-English half raised English MMLU and ARC in 12 of 13 pairs, by 1.21% relative (§4.1.4).
- Adding translations of fastText-selected English documents gave 5.09% relative gain for Hindi, Bengali, and Arabic; translations of random documents gave "marginal" gains (§4.2).

**Worked example of a phased share.** The 1T-token runs use 650B tokens at 5% multilingual, 250B at 10%, and 100B at 20% (§4.3). Multilingual tokens = 32.5B + 25B + 20B = 77.5B, which is 7.75% of 1T and 77.5B / 13 = 5.96B per language. The paper prints the total as "∼80B" (§1) and "75B" (App. A.4).

**Status of the paper's main claim.** The authors conclude that many multilingual regressions come from "correctable deficiencies in data quality and composition rather than fundamental capacity limits" (Abstract). This is an **Interpretation**: the controlled runs are bilingual at one model size and do not vary the number of languages at fixed capacity, which is the variable in §1 and §2. At 1T tokens, the 8B model's Spanish MMLU of 0.55 is below Qwen3-4B's 0.67; the paper's comparison is accuracy per training FLOP (App. A.4.1 Table 4; FLOPs in App. A.2 Table 3). How much of the capacity term in ATLAS remains after per-language curation is an **Open question**.

### §3.3 What released models report about coverage

- **Llama 3.1** ([[llama-3]] §3.1.1): fastText LID into 176 languages; document- and line-level deduplication within each language; language-specific heuristic and model-based filters; a multilingual Llama 2-based quality ranker. The multilingual share (8%, §3.1.2) was "determined experimentally, balancing model performance on English and multilingual benchmarks"; no table is given. Eight languages are supported, and the model "has not been optimized or safety tuned" for the others (§5.2.4, footnote 9).
- **Apertus** ([[apertus]] §3.2.2): FineWeb-2 in all 1,811 languages "in their natural frequency", with classifier-based quality filtering (FineWeb-2-HQ) only for the 20 largest languages, because web data volume "quickly drops off" beyond them.
- **SmolLM3-3B** ([[smollm3-multilingual-configs]]): the released configs sample 14 FineWeb2 languages while the blog lists 5 supported non-English languages. The Vietnamese weight drops from 0.00325 to 0.00005 in stage 2 with the config comment "downsample viet, too many epochs" (`stage3_9T_11T.yaml` L263), which is a repetition limit set by unique data.

**Worked example of a pool share.** Apertus Table 6 lists token pools per stage, and its caption states that not all pool tokens were consumed. Stage 1 lists 4,815B FineWeb-Edu, 3,557B multilingual, 235B code, 32B math, and 2B probe data, a total of 8,641B; the multilingual pool is 3,557 / 8,641 = 41.2%. Stage 4 lists 1,619 + 986 + 234 + 32 + 19 + 15 = 2,905B, with 986 / 2,905 = 33.9% multilingual. These are pool shares, not the sampled shares, so they cannot confirm the "∼40%" of the abstract.

## §4 Tokenizer fertility and vocabulary size as a per-language cost

ch-11 covers how BPE and SentencePiece vocabularies are built. This section measures what a fixed vocabulary costs each language.

### §4.1 Fertility and premium

**Definitions.** **Fertility** is the number of tokens a tokenizer produces per word (Apertus App. I: total tokens / total whitespace-separated words). The **premium** of language A relative to B is |t(s_A)| / |t(s_B)| for parallel sentences s_A and s_B, where |t(s)| is the tokenized length ([[tokenizer-language-unfairness]] §3). **Compression ratio** is normalization units (characters, bytes, or lines) per token.

**Problem.** A higher premium means more tokens for the same content, so a language receives less content per training token, per context window, and per unit of inference cost.

**Evidence** ([[tokenizer-language-unfairness]], FLORES-200, 2,000 sentences in 200 languages; Result, single study):

| Language | cl100k_base (ChatGPT/GPT-4) | XLM-R | ByT5 (UTF-8 bytes) |
|---|---|---|---|
| English | 1.00 | 1.00 | 1.00 |
| Italian | 1.64 | 1.19 | 1.19 |
| Chinese (Simplified) | 1.91 | 0.97 | 0.93 |
| Japanese | 2.30 | 1.11 | 1.27 |
| Bulgarian | 2.64 | 1.16 | 1.89 |
| Standard Arabic | 3.04 | 1.18 | 1.60 |
| Shan | 15.05 | 4.43 | 3.94 |

(Tables 1, 4, 5.) Three observations follow from the tables. Multilingual subword tokenizers reduce but do not remove the spread: all five multilingual tokenizers have at least one language with a premium above 2.5 (§4.3). Byte-level input does not give parity, because UTF-8 uses 1 byte for ASCII, 2 for Cyrillic or Arabic, and 3 for CJK characters (§4.4). Tokenizers built for French, German, and Vietnamese give English a lower premium than any other non-target language, which the authors relate to English text inside documents of other languages (§4.2, **Interpretation**).

[[fineweb-2]] gives a fertility view of the same problem on Wikipedia words of its nine canary languages and English (App. A.3, Table 2; lower is better):

| Tokenizer (vocabulary) | English | Arabic | Hindi | Thai | Telugu | Average |
|---|---|---|---|---|---|---|
| Mistral-v3 (32,768) | 1.45 | 4.76 | 4.99 | 4.87 | 9.83 | 4.12 |
| Llama3 (128,000) | 1.40 | 2.32 | 2.71 | 2.18 | 10.11 | 3.04 |
| Bloom (250,680) | 1.42 | 1.86 | 1.59 | 3.96 | 2.10 | 2.16 |
| Gemma (256,000) | 1.31 | 2.19 | 2.22 | 1.92 | 3.51 | 2.10 |

**Worked example.** A 1,000-word Telugu document becomes 10,110 tokens with the Llama3 tokenizer and 3,510 with Gemma, a ratio of 2.88. A 1,000-word English document becomes 1,400 and 1,310 tokens, a ratio of 1.07. Between these two tokenizers, English cost changes by 7% and Telugu cost by a factor of 2.88. Vocabulary size alone does not set the per-language cost: Bloom (250,680 entries) has Telugu fertility 2.10 and Thai 3.96, while Gemma (256,000 entries) has 3.51 and 1.92. What matters is which languages the tokenizer's training data covered (**Interpretation**; the table does not show vocabulary composition). FineWeb2 excluded vocabularies above 256,000 because, at its model size of around 1.5B parameters, the embedding matrix would force fewer layers (App. A.3).

### §4.2 Consequences for training and context

**Worked example.** For Standard Arabic with cl100k_base (premium 3.04):
- A 4,096-token window holds the content of 4,096 / 3.04 = 1,347 English tokens.
- Showing the model the content of 10B English tokens in Arabic takes 10B × 3.04 = 30.4B tokens.
- For Shan (15.05), the same window holds 272 English-token equivalents, and the same content takes 150.5B tokens.

Panel B of [figures/language-budget-explorer.html](figures/language-budget-explorer.html) applies the three tokenizers' premiums to a window size and a token budget. Two consequences follow. First, a language share defined in tokens gives high-premium languages less content than the share suggests. Second, a long-context evaluation at a fixed token length tests less content in high-premium languages; [[tokenizer-language-unfairness]] §5.3 states that less than a tenth of the English content fits for Burmese and Dzongkha with cl100k_base.

### §4.3 Vocabulary size: a compute-optimal allocation fitted on English

**Definitions** ([[vocabulary-scaling-laws]] §2.2). Parameters split into non-vocabulary parameters `N_nv` and vocabulary parameters `N_v = V · d`, with vocabulary size `V` and embedding dimension `d` (the paper counts the output matrix only). Data is counted in characters `H`, with tokens `D = H · f(V)`, where `f(V)` is tokens per character.

**Mechanism** (§3). A larger V lowers f(V), so a fixed token budget covers more characters. Past some size, f(V) changes little while embeddings of rare tokens receive few updates. For each compute budget, loss first falls and then rises as V grows, and the minimizing V increases with budget (Fig. 3).

**Formulas** (Approach 1, IsoFLOP fits on SlimPajama, N_nv from 33M to 1.13B, V from 4K to 96K; §4.1):

```
N_nv = 0.08 · C^0.50        N_v = 0.20 · C^0.42        H = 6.42 · C^0.50
```

- `C`: training FLOPs. Vocabulary parameters grow more slowly than non-vocabulary parameters: γ = 0.42 / 0.50 = 0.84 (0.83 by the derivative method, §4.2).

**Evidence** (Result, single study; English data; Table 1, Table 3):
- Predicted compute-optimal V for N_nv = 3B is 37K-43K, for 7B 60K-67K, and for 70B 212K-231K across the three approaches.
- At N_nv = 2.87B and 2.3e21 FLOPs, raising V from 32K to 43K changed the 7-task average from 50.3 to 51.6 and ARC-Challenge from 29.1 ± 1.3 to 32.0 ± 1.4. Pretraining runs were not repeated with other seeds (NeurIPS checklist item 7).

**Worked example of vocabulary parameters.** Apertus 8B has V = 131,072 and d = 4,096 with untied input and output embeddings ([[apertus]] §2.1-2.2): each matrix has 131,072 × 4,096 = 536,870,912 parameters, and the two together have 1.07B. SmolLM3-3B has V = 128,256, d = 2,048, and tied embeddings ([[smollm3-multilingual-configs]]): 128,256 × 2,048 = 262,668,288 parameters, 8.8% of the nominal 3B.

**Conditions and limits.** SmolLM3-3B's 128,256 entries are 3.0-3.5× the 37K-43K predicted for N_nv = 3B, and Apertus 8B's 131,072 entries are 2.0-2.2× the 60K-67K predicted for N_nv = 7B. Both models are trained on more tokens (11.14T for SmolLM3-3B, 15T for Apertus 8B) than the compute-optimal allocation assumed in Table 1; in Fig. 7, excess data moved the best V upward (16K → 24K at N_nv = 302M), although the authors still recommend the compute-optimal V because larger vocabularies raise inference cost (§5). The law does not contain a per-language term: the authors list multilingual vocabularies as future work because "different languages compete with each other" for model capacity (App. B.4). How to set V when premiums differ across target languages is an **Open question** for the sources in this chapter. Two measured points bound the English side of the trade-off. Llama 3's tokenizer adds 28K non-English tokens to 100K tiktoken tokens; the report states this improved compression and downstream performance "with no impact on English tokenization", and English compression rose from 3.17 to 3.94 characters per token relative to Llama 2's tokenizer ([[llama-3]] §3.2, no downstream numbers given). With one third of the cl100k_base vocabulary, English FLORES-200 sequences are about 10% longer ([[tokenizer-language-unfairness]] Fig. 3).

### §4.4 Selecting a tokenizer with a cross-language inequality statistic

Apertus compared four existing tokenizers on the FLORES+ development set in 55 languages using fertility, compression ratio, vocabulary utilization, and a Gini coefficient ([[apertus]] §2.2, App. I).

```
Gini = (1/n) · ( n + 1 − 2 · Σ_{i=1..n} (n + 1 − i) · c_i / Σ_{i=1..n} c_i )
```

- `n`: number of languages; `c_1 ≤ … ≤ c_n`: tokenization cost per language (average tokens per normalization unit, for example per line of parallel text), sorted ascending. 0 means equal cost for all languages.

**Worked example.** Four languages with costs [2, 2, 2, 2]: Σ(n+1−i)c_i = 4·2 + 3·2 + 2·2 + 1·2 = 20 and Σc_i = 8, so Gini = (1/4)(5 − 2·20/8) = 0. Costs [1, 1, 2, 4]: Σ(n+1−i)c_i = 4 + 3 + 4 + 4 = 15, Σc_i = 8, so Gini = (1/4)(5 − 3.75) = 0.3125.

**Evidence.** Mistral-Nemo had the lowest Gini coefficient and matched Gemma-2 on fertility and compression; Apertus chose it because it "is fairer across languages and uses a smaller vocabulary (128k vs. 256k)" (§2.2). The metric values appear only in the report's Fig. 1, and no pretraining ablation compares tokenizers.

**Implication for a general-purpose model.** The tokenizer fixes, before training starts, how many tokens each language needs for the same content. Language shares, context lengths, and evaluation lengths are comparable across languages only after conversion to content with each language's premium.

## §5 Measuring multilingual breadth

### §5.1 Held-out loss per language

Per-token perplexity is comparable only between models that use the same tokenizer on that language. [[multilinguality-curse-250-languages]] fixes the tokenizer per target language for this reason (§5). Across tokenizers, [[vocabulary-scaling-laws]] uses a unigram-normalized loss `L_u = −(1/T) Σ_i log[ p(w_i | w_<i) / p(w_i) ]`, where `p(w_i)` is the frequency of token w_i in the tokenized corpus and `T` the number of tokens (Eq. 4). ATLAS uses the same loss (§2). Across models with different V, ordinary loss correlated positively with downstream accuracy, because larger vocabularies raise per-token loss, while L_u correlated negatively (App. A.10, Fig. 13).

**Worked example.** A 1,000-character passage is 250 tokens under tokenizer A at a mean loss of 2.4 nats per token (600 nats) and 400 tokens under tokenizer B at 1.6 nats per token (640 nats). B has the lower per-token loss, but A has the lower loss per character: 0.60 vs 0.64 nats per character. Bits per character, which [[vocabulary-scaling-laws]] reports to correlate with L_u (App. A.5), would rank A first.

### §5.2 Task suites and their measurement errors

- **Native versus translated items.** Llama 3's multilingual MMLU is an internal translation made with Google Translate, with task instructions left in English ([[llama-3]] §5.2.4). Qwen2.5 combines five language-specific MMLU-like benchmarks (AMMLU, JMMLU, KMMLU, IndoMMLU, TurkishMMLU) with a translated MMLU (okapi) ([[qwen-2.5]] §5.2.2, Table 13). Apertus groups INCLUDE V1 (44 languages), BLEnD, and CulturalBench as region-specific factual knowledge, separate from Global-MMLU ([[apertus]] Table 15). A translated item keeps the content of its English source, so a translated suite measures language ability on English-context content (**Interpretation**); on the cultural benchmark BLEnD, Qwen2.5-72B-Instruct scores 32.48 against 35.91 for GPT-4o-mini ([[qwen-2.5]] Table 13).
- **Early-training signal.** FineWeb2 kept 84 of 197 tasks that met criteria of monotonicity (mean Spearman ρ ≥ 0.5 between step and score), signal-to-noise ratio ≥ 20 across four seeds, and non-randomness ≥ 3, and replaced multiple-choice letters with cloze scoring because letter formats score at chance early in training ([[fineweb-2]] App. A.5). ÜberWeb uses cloze for 60B-token runs and multiple-choice for 1T-token runs ([[uberweb-multilingual-curation]] §3).
- **Wrong-language output.** Multilingual models can score well on multiple-choice tasks while answering generative tasks in the wrong language ("accidental translation") ([[fineweb-2]] App. A.5.1).
- **Loss and task metrics can disagree.** For T5-3B pretrained on English-only data and fine-tuned for English-French translation, BLEU stopped improving or fell with more pretraining while downstream cross-entropy kept decreasing ([[downstream-scaling-laws-translation]] §5, Fig. 3-4). ATLAS reports loss only.
- **Held-out suites.** Apertus lists benchmarks that "were held-out during model development and were not used for making decisions" (§5.2, Table 21), including multilingual ARC Challenge, where Apertus-8B-Instruct scores 36.8 against 32.0 for Llama-3.1-8B-Instruct.

### §5.3 Low-resource and unseen-language slices

- Languages not used to design the data pipeline form one slice: on five such languages FineWeb2 trailed HPLT2 on German and Indonesian (§3.1), and the authors report that trends on canary and unseen languages agree ([[fineweb-2]] §5).
- Every sampled language forms a slice, not only supported languages: SmolLM3 samples 14 non-English languages and lists 5 as supported; Llama 3 identifies 176 languages and supports 8 (§3.3).
- A low-resource generation task tests output in the language, not only recognition: Apertus evaluates German ↔ Romansh translation in six Romansh varieties with BLEU ([[apertus]] §5.3).
- When English retention matters, an English-only reference at equal compute is the comparison point, because §1 and §2 predict English loss rises when languages are added at fixed size.

## Negative samples and negative feedback

At this stage "negative" means sense (1) of the course standard: **negative marginal value**. Documents are removed because a language identifier, a heuristic filter, or a wordlist check predicts that they would lower performance as positive training targets. No source in this chapter uses rejected documents as content, as conditioning, or as gradient at this stage; senses (2)-(4) are covered in ch-31a and ch-43a. Negative transfer between languages (§2.2) is a property of positive training data, not a negative training signal.

**Where negatives come from and how they are labeled.** LID confidence below a per-language threshold, heuristic filter scores, and absence of language-specific words ([[fineweb-2]] §4.2-4.4).

**Error rates.** The wordlist filter's audit reports precision and recall of the kept set (Table 28 in §3.1). For ary_Arab, recall after filtering is 88.57%, so 11.43% of genuine in-language documents in the audited sample were removed. For Dagbani, stopword lists computed from an uncleaned Wikipedia contained English words; recomputed lists removed over 99% of misclassified data (App. A.7.1).

**What current practice does.** FineWeb2 discards removed documents but reuses the removal rates: the removal rate for each duplicate-cluster size sets the upsampling weight of the kept documents in that cluster size (weight 10 for the lowest rate, weight 1 above the global rate of 62.4% for French) (§4.5).

**Effect on generality.** For low-resource languages, the removal threshold trades noise against the small amount of available text: Swahili performed best at a threshold that removed almost 65% of documents (§4.2). Apertus stopped applying quality filters beyond 20 languages because of volume ([[apertus]] §3.2.2). The diagnostic is a per-language removal-rate table with an audited precision and recall sample for the lowest-resource languages.

## Recipe

Rows marked 2026-09-15 were read at the stated locus in the primary source for this chapter; rows marked 2026-09-14 come from verified library cards.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 (all sizes) | 8B, 70B, 405B | pretrain-stable | multilingual share of final mix (tokens) | 8% ("roughly") of "a corpus of about 15T multilingual tokens" (§1) | arXiv:2407.21783v3 §1, §3.1.2 ([[llama-3-recipe]]) | verified 2026-09-14 | §3.1.1: amount "determined experimentally, balancing model performance on English and multilingual benchmarks"; no table |
| Llama 3.1 (all sizes) | 8B, 70B, 405B | pretrain-stable | multilingual curation | fastText LID, 176 languages; per-language document- and line-level dedup; language-specific heuristic and model-based filters; multilingual Llama 2-based quality ranker | §3.1.1 ([[llama-3]]) | verified 2026-09-15 | no ablation reported |
| Llama 3.1 (all sizes) | 8B, 70B, 405B | pretrain-stable | vocabulary | 128K = 100K tiktoken + 28K non-English tokens; English 3.17 → 3.94 characters per token vs Llama 2 tokenizer | §3.2 | verified 2026-09-15 | "improved both compression ratios and downstream performance" (no numbers) |
| Llama 3 multilingual expert | not printed | mid-train | data mix | 90% multilingual tokens | §4.3.2 ([[llama-3-recipe]]) | verified 2026-09-14 | used to collect non-English annotations; no ablation reported |
| Llama 3.1 | not scoped | SFT | multilingual share of SFT examples | 3.01%; multilingual SFT = 2.4% human, 44.2% other NLP tasks, 18.8% rejection-sampled, 34.6% translated reasoning | Table 7, §4.3.2 ([[llama-3-recipe]]) | verified 2026-09-14 | mix adjusted per round; translated math gave "strong gains on MGSM" (no numbers) |
| Llama 3.1 | 8B, 70B, 405B | eval-gate | supported languages | 8 (en, de, fr, it, pt, hi, es, th) | §5.2.4, footnote 9 | verified 2026-09-15 | Table 20: MGSM 68.9 / 86.9 / 91.6; multilingual MMLU 58.6 / 78.2 / 83.2 |
| Qwen2.5 (open-weight) | 0.5B-72B | pretrain-stable | multilingual token share; number of languages | not reported | arXiv:2412.15115v2 (checked §1-§3, Table 1, §5) | not reported | — |
| Qwen2.5 (open-weight) | 0.5B-72B | pretrain-stable | vocabulary | byte-level BPE, 151,643 regular tokens | §2 ([[qwen-2.5]]) | verified 2026-09-14 | no ablation reported |
| Qwen2.5-Instruct | 0.5B-72B | SFT | cross-lingual data | instructions translated from high- to low-resource languages; responses checked for semantic alignment; count not reported | §4.1 (7) ([[qwen-2.5]]) | verified 2026-09-15 | Table 13 (72B-Instruct): JMMLU 80.56, KMMLU 61.96 |
| Apertus-8B, Apertus-70B | 8B, 70B | pretrain-stable | languages; non-English share; tokens | 1,811 languages; "∼40%" non-English; 15T tokens (Table 2) | arXiv:2509.14233v2 Abstract, §1, §3.2.2 ([[apertus]]) | verified 2026-09-15 | no multilingual-share ablation in the report |
| Apertus-70B | 70B | pretrain-stable | multilingual pool per stage | Stages 1-3: 3,557B / 3,557B / 3,556B; Stages 4-5: 986B FineWeb-2 (Stage 5 adds 33B Clean Wikipedia and 21B parallel data) | Table 6 | verified 2026-09-15 (caption: not all pool tokens consumed) | Table 7 cooldowns tested English candidates only |
| Apertus-70B | 70B | pretrain-stable | multilingual share of listed pool | Stage 1: 3,557 / 8,641 = 41.2%; Stage 4: 986 / 2,905 = 33.9% | Table 6 | derived (pool shares, not sampled shares) | — |
| Apertus-8B, Apertus-70B | 8B, 70B | pretrain-stable | tokenizer | Mistral-Nemo v3 tekken byte-level BPE, 131,072; untied embeddings | §2.1-2.2 | verified 2026-09-15 | Fig. 1: lowest Gini of 4 tokenizers on FLORES+ (55 languages) |
| Apertus-8B, Apertus-70B | 8B, 70B | pretrain-stable | quality-filter coverage | classifier filtering (FineWeb-2-HQ) for 20 languages; other languages as in FineWeb-2 | §3.2.2, App. G | verified 2026-09-15 | reason stated: volume drops off; no ablation |
| SmolLM3-3B | 3B | pretrain-stable (0-8.14T) | multilingual sampling weight | 0.12 over 14 FineWeb2 entries (deu 0.022, spa 0.02, fra 0.016, ita 0.0105, …) | github.com/huggingface/smollm `text/pretraining/smollm3/stage1_8T.yaml` L105-118 ([[smollm3-multilingual-configs]]) | verified 2026-09-15 | blog: ablations on 3B models at 50B-100B tokens; no numbers |
| SmolLM3-3B | 3B | pretrain-stable (8.14-9.90T) | multilingual sampling weight | 0.117; vie 0.00325 → 0.00005 ("downsample viet, too many epochs") | `stage3_9T_11T.yaml` L251-264 | verified 2026-09-15 | config comment only |
| SmolLM3-3B | 3B | pretrain-decay/anneal (9.90-11.14T) | multilingual sampling weight | 0.12405 of a weight sum of 1.00565 = 12.3% | `stage3_9T_11T.yaml` L419-474 | derived | — |
| SmolLM3-3B | 3B | long-context | multilingual sampling weight | 0.11108 (4k → 32k); 0.10808 (32k → 64k) | `long_context_4k_to_32k.yaml` L135-148; `long_context_32k_to_64.yaml` L137-150 | verified 2026-09-15 | no ablation reported |
| SmolLM3-3B | 3B | pretrain-stable | tokenizer; supported languages; tokens | Llama-3.2 tokenizer, 128,256, tied embeddings; 6 supported languages; "11T tokens" (blog), 4,720,000 steps × 2,359,296 tokens = 11.14T (config) | `stage1_8T.yaml` L144-146, L196, L225-255; blog "Model summary" | verified 2026-09-15 (11.14T derived) | no ablation reported |
| DatologyAI 3B, 8B (ÜberWeb) | 3B, 8B | pretrain-stable | multilingual share by phase | 650B at 5%, 250B at 10%, 100B at 20% = 7.75% of 1T; 13 languages | arXiv:2602.15210v3 §4.3 ([[uberweb-multilingual-curation]]) | conflict: multilingual total "∼80B" (§1), 75B (App. A.4), 77.5B derived | no ablation of the phase shares reported |
| DatologyAI bilingual runs | 3B | eval-gate | tokens; mix; tokenizer; context | 60B; 50/50 English:target; Llama-3.2 tokenizer; 4,096 | §3, §4.1 | verified 2026-09-15 | Fig. 2, Fig. 4 |
| LFM2-350M, -700M, -1.2B | 0.35B-1.2B | pretrain-stable | corpus mixture (share type not stated) | ~75% English, 20% multilingual, 5% code; 10T tokens | blog "Training LFM2" ([[lfm2-liquid]]) | verified 2026-09-14 | no ablation reported |
| ATLAS Unimax checkpoints | 10M-8B | pretrain-stable | language sampling; tokens | English 5.00%, most high-resource languages 1.42%; 420 languages; 1T tokens | arXiv:2510.22037v2 App. B.4, Table B.4 ([[atlas-multilingual-scaling-laws]]) | conflict (§6 says 1B tokens) | adopted from Chung et al. 2023; no ablation |
| ATLAS study models | 10M-8B | pretrain-stable | vocabulary | 64K SentencePiece + 512 special tokens, 99.9995% character coverage | §2, Table B.2 | verified 2026-09-14 | Fig. 1: multilingual vs monolingual vocabulary frontier |
| FineWeb2 ablation model | 1.46B | eval-gate | tokenizer | Gemma, 256,000 entries | arXiv:2506.20920v1 §3.1, Table 2 ([[fineweb-2]]) | verified 2026-09-14 | Table 2: lowest mean fertility (2.10) among compared tokenizers |
| Chang et al. models | 4.6M-29.5M (8.7M-45.8M multilingual) | pretrain-stable | sequence; batch; LR; epochs | 128; 128; 1e-3 / 7e-4 / 5e-4; 20 / 20 / 10 / 2 epochs for 1M / 10M / 100M / 1B tokens | arXiv:2311.09205v1 App. A.3 Table 1 ([[multilinguality-curse-250-languages]]) | verified 2026-09-15 | more than 20 epochs "often leads to overfitting" in low-resource runs |

**Starting point for a small general-purpose run.** For a 3B dense model trained on about 11T tokens that must serve English and a few high-resource languages, SmolLM3-3B's released configs sample FineWeb2 and FineWeb2-HQ at a total weight of 0.12 over 14 languages with a 128,256-entry tokenizer, and they cut a language's weight when its unique data would be repeated too often (Vietnamese, 0.00005). For 8B-70B models trained on 15T tokens with coverage of many languages as a goal, Apertus used FineWeb-2 in 1,811 languages at "∼40%" non-English with a 131,072-entry byte-level BPE chosen for its low Gini coefficient on 55 languages. Llama 3.1 (dense, 8B-405B, about 15T tokens) used 8% multilingual tokens and a 128K vocabulary that adds 28K non-English tokens. A 3B or 8B model trained on 1T tokens with per-language curation used phases of 5%, 10%, and 20% multilingual tokens across 13 languages, 7.75% overall (derived from the phase lengths; the paper's printed totals conflict). None of these shares was selected by a published ablation. When a share is being fixed, three checks apply: the premium of the chosen tokenizer for each target language (§4.1), the content that the planned token share buys at that premium (§4.2), and the per-language unique-token count against the planned number of passes (ch-14).

## Generalization lens

**(a) What increases breadth.**
- Adding related-language data for data-poor languages: +33% equivalent data at 1M target tokens with similar languages ([[multilinguality-curse-250-languages]] §6.1).
- Scaling parameters and total tokens with the number of languages: 1.4× and 2.74× for 4× languages at unchanged loss ([[atlas-multilingual-scaling-laws]] §5); larger models move interfering pairs toward zero (App. C.2).
- Per-language curation: 16.87% relative gain over uncurated data in 3B bilingual runs, and English curation that raises 12 of 13 other languages ([[uberweb-multilingual-curation]] §1, §4.1.1); FineWeb2's adapted pipeline raised Arabic from 21.7 to 25.2 and French from 18.3 to 23.6 at 350B tokens ([[fineweb-2]] Tables 29-30).
- Tokenizers with lower cross-language inequality: XLM-R's Shan premium 4.43 against 15.05 for cl100k_base ([[tokenizer-language-unfairness]] Tables 1, 4); Apertus selection by Gini ([[apertus]] §2.2).
- Translation of selected high-quality documents into low-resource languages: 5.09% relative gain against marginal gains for random documents ([[uberweb-multilingual-curation]] §4.2).

**(b) What causes narrowing or forgetting.**
- More languages at fixed capacity: 1B added tokens made a 1B-token language perform like 37% of its data at 29.5M parameters and 12% at 4.6M ([[multilinguality-curse-250-languages]] §6.2).
- Holding total tokens fixed while adding languages: the K term in ATLAS raises per-language loss (§5, Fig. 4).
- English filter thresholds applied to other languages ([[fineweb-2]] Table 25); quality classifiers trained for only the largest languages ([[apertus]] §3.2.2); low-resource data that is mostly Bible or Wikipedia text (App. A.12).
- Repeating the small unique pool of a low-resource language: the SmolLM3 config reduced Vietnamese sampling because of "too many epochs" ([[smollm3-multilingual-configs]]); Chang et al. saw overfitting beyond 20 epochs (App. A.3).
- High-premium languages under a token-denominated share or context length: 30.4B Arabic tokens carry the content of 10B English tokens with cl100k_base (§4.2).

**(c) How to measure it for this stage.**
- Per-language held-out loss with a fixed tokenizer per language, or a vocabulary-insensitive loss (L_u, bits per character) when tokenizers differ ([[vocabulary-scaling-laws]] Eq. 4, App. A.10).
- Per-language task suites with language-specific items (INCLUDE, KMMLU, IndoMMLU) next to translated ones, plus cultural benchmarks ([[apertus]] Table 15; [[qwen-2.5]] Table 13).
- Tasks filtered for early signal: monotonicity, signal-to-noise ratio, non-randomness, cloze formulation ([[fineweb-2]] App. A.5).
- Slices: languages held out from pipeline design, sampled-but-unsupported languages, a low-resource generation task, and an English-only reference at equal compute (§5.3).
- Known measurement errors: translated benchmarks carry English-context content; FLORES-200 has English-centric names ([[tokenizer-language-unfairness]] §6); multiple-choice scores can hide wrong-language generation ([[fineweb-2]] App. A.5.1); loss can improve while a task metric falls ([[downstream-scaling-laws-translation]] §5); ÜberWeb compares its models with baselines on accuracy per FLOP using its own estimates of per-language tokens (App. A.4).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Adding languages without changing parameters or total tokens | English or high-resource held-out loss rises between runs | Compare per-language loss against the ATLAS iso-loss plan (§2.3); run an English-only reference at equal compute |
| Comparing perplexity across different tokenizers | A model with a larger vocabulary looks worse by loss but better on tasks | Use L_u or bits per character ([[vocabulary-scaling-laws]] App. A.10) |
| Reading a token share as a content share | High-premium languages underperform despite a "balanced" share | Multiply each language's share by 1 / premium on parallel text (§4.2) |
| Reusing English LID or filter thresholds | An audited sample of a low-resource language contains mostly other-language text (precision 1.75% for ary_Arab before wordlist filtering) | Per-language threshold from the score distribution; audit precision and recall on a sample ([[fineweb-2]] §4.2, Table 28) |
| Assuming a released share is a sampled share | Shares derived from pool tables disagree with the stated share | Read the released config weights; mark pool-derived shares as derived ([[apertus]] Table 6 caption) |
| Treating "multilingual" in a blog as a share of the web portion | Recomputed mixture does not sum to the stated totals | Sum the per-language weights in the config ([[smollm3-multilingual-configs]]) |
| Repeating a low-resource pool beyond its unique tokens | Held-out loss for that language rises late in training | Track passes per language; cap weight when passes exceed the data-constrained limit (ch-14) |
| Evaluating only supported languages | Regressions in sampled-but-unsupported languages go unreported | Evaluate every sampled language and the held-out design languages (§5.3) |
| Using multiple-choice scores alone for generation ability | Good multiple-choice scores with answers in the wrong language | Add a generative task and a language-ID check on outputs ([[fineweb-2]] App. A.5.1) |
| Citing "the curse is a data-quality problem" as settled | A plan that adds many languages without extra capacity | Cite ÜberWeb's bilingual design and the unvaried K; test K at fixed size before relying on curation alone (§3.2) |

## Check your understanding

1. In [[multilinguality-curse-250-languages]], adding 1B tokens from similar languages helped a 1M-token language but hurt a 1B-token language more than adding dissimilar languages did. Explain both results with one account based on transfer and capacity, and state which additional experiment would test that account.
2. ATLAS predicts that 4× languages need 2.74× total tokens but 32% fewer tokens per language. Explain which term of Eq. 7 produces the per-language reduction and why the parameter multiplier is still above 1.
3. A mixture assigns 5% of tokens each to French and Standard Arabic, and training uses cl100k_base (premiums 1.60 for French and 3.04 for Standard Arabic, [[tokenizer-language-unfairness]] Table 1). Compute the ratio of Arabic content to French content, and explain how this changes the interpretation of equal shares.
4. ÜberWeb finds that curating English data raises non-English scores in 12 of 13 bilingual runs. Explain why this result does not by itself show that the curse of multilinguality is a data-quality effect, and design a run that would separate quality from capacity.
5. The vocabulary scaling law predicts 37K-43K entries for a 3B non-vocabulary model, yet multilingual 3B models use about 128K. Explain what the English-only fit omits and what measurement you would add before choosing V.
6. SmolLM3's config lowers the Vietnamese weight with the comment "too many epochs". Using ATLAS's repetition function S_λ, explain why a low-resource language's weight cannot be raised indefinitely, and what the alternative sources of positive transfer are.
7. A new model improves per-token perplexity on a Hindi held-out set after a tokenizer change, while Hindi Belebele accuracy is unchanged. Give two explanations and the measurement that distinguishes them.

## Connections

- **Previous (dependency):** ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale. Language shares are one dimension of the mixture that ch-13 optimizes.
- **Next:** ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination. The repetition limits of §2.1 and §3.3 for low-resource languages.
- ch-00 — What General Capability Means and How It Is Measured (multilingual ability as a coverage area).
- ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability (the Chinchilla law that ATLAS extends).
- ch-09 — Pretraining Data Composition and Capability Coverage (coverage map and source types).
- ch-10 — Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb (the English pipeline FineWeb2 adapts).
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (language and dialect effects of quality classifiers).
- ch-11 — Tokenizers, Data Provenance, and PII Removal (how vocabularies are built).
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic (per-language MinHash and rehydration).
- ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (stage mixtures that include language shares).
- ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement (applies §5 measurements).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (continuing from a multilingual checkpoint, §2.4).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (token premium and context capacity).
- ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (multilingual SFT shares).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-47 — Evaluation Harness and Suite Design for General Capability (per-language suites).
- ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing (language slices).
- ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions (signal-to-noise criteria for multilingual tasks).

## Sources

- [[multilinguality-curse-250-languages]] — controlled study design, relative log-likelihood and equivalent-token metric, low- and high-resource effects, similarity analysis, training settings.
- [[atlas-multilingual-scaling-laws]] — ATLAS law with transfer and repetition, bilingual transfer score and matrix results, K scaling law and iso-loss multipliers, pretrain-versus-continue crossover, Unimax and vocabulary settings.
- [[fineweb-2]] — per-language LID threshold, adapted filter thresholds, precision filtering audit, rehydration, held-out language results, task-selection criteria, tokenizer fertility, Bible and Wikipedia share.
- [[uberweb-multilingual-curation]] — bilingual curation experiments, similarity correlations, translation of selected documents, phased 7.75% share, status of the data-quality interpretation.
- [[vocabulary-scaling-laws]] — vocabulary parameter definitions, IsoFLOP fits and γ, predicted vocabulary sizes, 32K vs 43K result, unigram-normalized loss, English-only scope.
- [[tokenizer-language-unfairness]] — premium definition, premium tables for English-centric, multilingual, and byte-level tokenizers, context and cost consequences, vocabulary-size trade-off for English.
- [[apertus]] — 1,811-language coverage and "∼40%" share, Table 6 pools, quality-filter coverage, tokenizer selection with Gini, multilingual and held-out evaluations, Romansh translation.
- [[smollm-3]] — library card for the SmolLM3 release (pre-revision; contains no multilingual values).
- [[smollm3-multilingual-configs]] — SmolLM3-3B per-language sampling weights, Vietnamese repetition comment, tokenizer and supported languages.
- [[llama-3]] — multilingual curation pipeline, 28K non-English vocabulary tokens, supported languages, translated multilingual MMLU.
- [[llama-3-recipe]] — verified 8% multilingual share, multilingual expert, multilingual SFT composition.
- [[qwen-2.5]] — vocabulary, cross-lingual SFT data, native and translated multilingual evaluations, BLEnD score.
- [[lfm2-liquid]] — 20% multilingual corpus share for comparison.
- [[downstream-scaling-laws-translation]] — divergence between downstream cross-entropy and BLEU under language misalignment.
