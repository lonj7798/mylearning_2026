<!-- chapter: ch-10
     track: pretraining
     kind: content
     title: Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb
     deps: [ch-09]
     sources: [[ccnet]], [[c4]], [[dolma]], [[fineweb]], [[minhash-lsh]], [[paloma]], [[whose-language-counts-quality-filter]], [[pretrainers-guide-training-data]]
     figures: figures/pipeline-compare.html
     revised: 2026-09 (generality revision)
-->

# Chapter 10 — Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb

> **Core insight.** Heuristic pipelines remove the majority of a Common Crawl snapshot (Dolma's CCNet stage alone removed 84.2% of bytes, [[dolma]] §5.1), and each rule removes some domains and dialects at higher rates than others. In Dodge et al.'s rebuild of C4, the cleaning rules reduced 1.4T SpaCy tokens to 198B, and the word blocklist then removed 42% of documents classified as African American English against 6.2% of White-aligned English ([[c4]], Dodge et al. 2021 §5.3). At 1.2B parameters and 150B tokens, models trained on single-source filtered web corpora (C4, mC4-en, RefinedWeb) had higher average perplexity on diverse Paloma domains than models trained on multi-source corpora ([[dolma]] §9.2). The rule sets Dolma and FineWeb kept were selected with 1.2B–1.71B-parameter data ablations: FineWeb scored an eight-task commonsense and knowledge aggregate, and Dolma shows per-domain perplexity for its filter ablations only as appendix figures, so a filter's effect on a specific domain has to be checked separately.
>
> **Guideline.** When adopting a heuristic filter, record its granularity (line, paragraph, sentence, document), its threshold, and its removal rate with the unit, because the rule named "C4 filters" has different thresholds in the T5 paper, in Dodge et al., and in the datatrove defaults used for FineWeb. When choosing a threshold, use a data ablation at fixed model size and token budget, then check per-domain perplexity on code, LaTeX, and dialectal text, because FineWeb's selection suite contains none of these and the C4-only baseline in Paloma reached perplexity 391,171 on arXiv ([[paloma]] §4.1). When deduplicating many Common Crawl snapshots with MinHash, deduplicate each snapshot separately, because in FineWeb's 1.71B-parameter, 350B-token ablation global deduplication gave little gain and the data it kept from an old snapshot trained a worse model than the data it removed ([[fineweb]] §3.4). Otherwise, with a single snapshot, the two procedures are the same.

## Why this chapter matters for a general-purpose model

In Dolma v1.6, filtered web text is the largest source: Common Crawl supplies 2,479B of Dolma v1.6's 3,059B tokens ([[dolma]] Table 1). Chapter ch-09 compared sources (web, code, books, academic text). This chapter covers the selection that happens inside the web source, before any learned quality model is applied. Model-based quality classifiers (FineWeb-Edu, DCLM, Nemotron-CC) are covered in ch-10a, and deduplication algorithms in depth in ch-12.

A filtering rule decides which text the model never sees. A register, dialect, or notation removed by a filter can reach the model only through other sources in the pretraining data (Interpretation). This chapter answers three questions for heuristic pipelines. (1) Which choices keep breadth: stage order, per-language thresholds, labeling instead of dropping, and per-snapshot deduplication. (2) Which choices narrow coverage: word blocklists, code-removal rules, reference-anchored perplexity cutoffs, and thresholds tuned on one benchmark aggregate. (3) How to measure the effect: per-domain perplexity, removal audits by topic and dialect, and data ablations with stated model size, token budget, and seeds.

The companion figure [figures/pipeline-compare.html](figures/pipeline-compare.html) lets the reader click each stage of the four pipelines to see its rule, threshold, removal rate, and locus, compute MinHash match probabilities for different bucket settings, and compare how line-, paragraph-, and document-level punctuation rules treat the same document.

## §1 How to read a heuristic pipeline

**Definitions.**
- A **Common Crawl snapshot** is one crawl release. Each is distributed as WARC files (raw HTML and request metadata) and WET files (text extracted by Common Crawl) ([[fineweb]] §3.2). A CCNet-era snapshot holds 20–30 TB of uncompressed text from about 3 billion pages ([[ccnet]] §3.1).
- A **heuristic filter** is a rule computed from surface statistics of text (punctuation, word counts, word lists, repetition) rather than from a model trained to predict quality. This chapter also covers language identification (a fastText classifier) and reference perplexity (a 5-gram model), because they are standard stages of these pipelines.
- **Granularity** is the unit a rule removes: a line, a paragraph (newline-separated span in Dolma, [[dolma]] §4 footnote 3), a sentence, or a whole document.
- A **removal rate** is the fraction removed, in a stated unit. The sources use different units: characters (CCNet), bytes, documents, or UTF-8 characters (Dolma), GPT-2 tokens (FineWeb), SpaCy tokens (Dodge et al.).
- A **data ablation** trains models that are identical except for one data decision and compares them on the same evaluations ([[dolma]] §4.2; [[fineweb]] §3.1).

**Reading template.** For every stage of a new pipeline, extract six fields:
1. Position in the stage order, and whether the order was ablated or chosen for cost.
2. Granularity.
3. Threshold, and how it was chosen (ablation, histogram inspection, token budget, or not stated).
4. Removal rate with its unit.
5. Ablation design: model size, tokens per run, number of seeds.
6. Evaluation suite, and which capabilities it does not measure.

§7 applies the template to all four pipelines in one table.

## §2 CCNet: deduplication, language identification, perplexity buckets

**Definition.** CCNet (Wenzek et al., arXiv 2019-11) processes each Common Crawl snapshot's WET text in the order paragraph deduplication → fastText language identification → SentencePiece tokenization and 5-gram perplexity → regrouping by language and perplexity bucket ([[ccnet]] §3, Figure 1).

**Problem.** Duplicated paragraphs "represent 70% of the text" of a snapshot (§3.2). The paper names navigation menus, cookie warnings, and contact information as removed boilerplate, and states that this step removes English content from pages written in other languages, which makes the next step, language identification, more robust (§3.2).

### §2.1 Paragraph deduplication, then language identification

Mechanism ([[ccnet]] §3.2–§3.3, §4.2):
1. Normalize each paragraph: lower-case, replace digits with 0, remove Unicode punctuation and accent marks.
2. Hash it (first 64 bits of SHA-1) and store the hashes per 5 GB shard.
3. Remove a paragraph if its hash occurs in the comparison set. Across 1 shard, 42% of a shard's characters remain; across 100 shards, 28% remain (Figure 4). The authors compare against 50 shards (1.5B hashes, 40 GB of RAM), about 3% of the corpus.
4. Run fastText language identification (176 languages). Keep the page in its top language if the score is above 0.5; otherwise discard it.

CCNet uses exact matching only; it has no MinHash or other fuzzy step ([[ccnet]] Technical Details).

**Why the order matters.** The authors deduplicate before language identification "because a lot of English boilerplate, such as cookie warnings, is present in pages of other languages" (§4.1). Figure 3, estimated on 1% of the February 2019 snapshot, plots per-language document ratios between "dedup then LID" and "LID then dedup"; the text states that low-resource languages benefit most, and that without deduplication first their documents were misclassified (generally as English) or discarded because no language could be identified (§4.1). Per-language values are shown only in the figure. **Result (single study).**

Constructed example (no fastText scores were computed for it): a Swahili page contains 400 characters of Swahili and 600 characters of an English cookie notice and menu that appear on thousands of pages. Before deduplication, most characters are English, so the top-language score for Swahili can fall below 0.5 and the page is discarded or labeled English. After deduplication removes the repeated English paragraphs, the page contains only Swahili text.

### §2.2 Perplexity bucketing against Wikipedia

For each of 48 languages, CCNet trains a SentencePiece tokenizer and a 5-gram Kneser–Ney model (KenLM) on that language's Wikipedia and computes the perplexity of each paragraph ([[ccnet]] §3.4). The paper does not print the formula; the standard definition is:

```
PPL(x) = exp( −(1/N) · Σ_{i=1..N} ln p(t_i | t_{i−4}, …, t_{i−1}) )
```

- `x` is a paragraph, `t_i` its i-th SentencePiece token, and `N` its number of tokens.
- `p(t_i | …)` is the 5-gram model's probability of `t_i` given the previous four tokens.

Lower perplexity means the text is closer to the reference domain (§3.4). How paragraph scores are combined into a document score is not described in the paper.

**Worked example.** A 4-token paragraph with probabilities 0.5, 0.25, 0.5, 0.125 has mean log-probability (−0.693 − 1.386 − 0.693 − 2.079)/4 = −1.213, so PPL = e^1.213 ≈ 3.36 (equivalently (2·4·2·8)^(1/4) = 128^(1/4)). Suppose one language has nine documents with perplexities 12, 15, 18, 25, 31, 40, 55, 90, 300. Splitting into three equal parts gives head = {12, 15, 18}, middle = {25, 31, 40}, tail = {55, 90, 300}. CCNet computes these tercile thresholds separately per language (§3.4, §5.2), because perplexity distributions differ between languages; the authors attribute the difference to the size of each Wikipedia used to train the LM (Interpretation, §5.2). The English LM was trained on "534M" of text and the Gujarati LM on "12M" (§5.2).

**CCNet labels; it does not drop.** "Some documents despite being valid text ends up in the tail because they have a vocabulary very different from Wikipedia. This includes blog comments with spoken-like text, or very specialized forums with specific jargon. We decided to not remove content based on the LM score" ([[ccnet]] §5.2). Downstream users made stricter choices: FineWeb §2 describes CC-100 as retaining "only the text that is assigned a low perplexity" ([[ccnet]] Connections).

**Evidence.** Word embeddings (fastText, 300-d) trained on English head, middle, and tail reach analogy accuracy 77.9, 74.2, and 62.0 (Table 1). BERT-BASE without next-sentence prediction, trained on CC head, averages 75.9 XNLI dev accuracy over en/ru/zh/ur against 72.6 when trained on Wikipedia; Urdu rises from 57.3 to 64.3 (Table 2). **Limits:** one snapshot; Table 2 compares CC head with Wikipedia, not with unfiltered, middle, or tail Common Crawl; no seeds or variance reported ([[ccnet]] Findings).

**Implication.** A reference-perplexity score measures similarity to the reference corpus. [[whose-language-counts-quality-filter]] tests the same design choice with a different scorer: a replicated GPT-3 quality classifier trained to separate Wikipedia, Books3, and OpenWebText from Common Crawl. On 910K U.S. high-school newspaper articles, it rated articles from larger schools in wealthier, more educated, and urban ZIP codes higher (§3.4, Table 3), and its score did not differ between news sources of high and low factual reliability (§4.2, Kolmogorov–Smirnov p = 0.085). **Result (single study).** Keeping CCNet buckets as labels lets a later mixture decide how much tail text to use.

## §3 C4: line and page rules, span deduplication, and what the blocklist removed

**Definition.** C4 (Raffel et al., arXiv 2019-10) is the April 2019 Common Crawl WET text passed through line and page rules and langdetect English identification, about 750 GB of text ([[c4]] §2.2).

### §3.1 The rules as stated and as coded

[[c4]] §2.2 lists: keep only lines ending in a terminal punctuation mark (period, exclamation mark, question mark, end quotation mark); discard pages with fewer than 3 sentences; keep only lines with at least 5 words; remove pages containing any word on the "List of Dirty, Naughty, Obscene or Otherwise Bad Words"; remove lines containing "Javascript"; remove pages containing "lorem ipsum"; remove pages containing a curly bracket "{", because it "appears in many programming languages … but not in natural text"; remove citation markers; remove lines with policy strings; "discarded all but one of any three-sentence span occurring more than once"; keep pages classified as English by langdetect with probability at least 0.99.

The TensorFlow Datasets release code (`tensorflow_datasets/text/c4_utils.py`, master, fetched 2026-09-14, L248–270) shows the granularity of each rule and the order in which they are checked:

```python
    if not line.endswith(_END_MARKS) or line.endswith(_ELLIPSIS):
      counter_inc_fn("line-filtered:no_endmark")
      continue
    if len(line.split()) < min_words_per_line:
      counter_inc_fn("line-filtered:too_short")
      continue
    line_lower = line.lower()
    # Remove documents which contain lorem ipsum
    if "lorem ipsum" in line_lower:
      counter_inc_fn("filtered:loremipsum")
      return
    # Remove "javascript must be enabled" notices
    if "javascript" in line_lower:
      counter_inc_fn("line-filtered:javascript")
      continue
    # Remove docs which probably contain javascript code
    if "{" in line:
      counter_inc_fn("filtered:squigglybracket")
      return
    # Remove policy lines
    if any(p in line_lower for p in _POLICY_SUBSTRINGS):
      counter_inc_fn("line-filtered:policy")
      continue
```

`continue` removes one line; `return` removes the page. In this code a `{` removes the page only if it occurs on a line that already passed the end-mark and 5-word checks, and lines ending in "..." are removed. The same file deduplicates at the line level (`remove_duplicate_text`, one URL kept per hashed line), while §2.2 describes three-sentence spans; neither artifact states which procedure built the 2019 release.

### §3.2 Evidence: filtered versus unfiltered C4

T5 compares a 220M baseline pre-trained for 2^35 ≈ 34B tokens on C4 (745 GB) and on "unfiltered C4" (6.1 TB, langdetect only) ([[c4]] §3.4.1, Table 8). C4 scores higher on all seven columns: GLUE 83.28 vs 81.46, CNN/DM 19.24 vs 19.14, SQuAD 80.88 vs 78.78, SuperGLUE 71.36 vs 68.04, EnDe 26.98 vs 26.55, EnFr 39.82 vs 39.34, EnRo 27.65 vs 27.21. **Result (single study).** The comparison removes all rules at once; the effect of individual rules is not measured in the T5 paper ([[c4]] Findings).

### §3.3 Side effects measured by Dodge et al. (2021)

Dodge et al. (arXiv:2104.08758v2, recorded in [[c4]] Connections) rebuilt three versions: C4.EN.NOCLEAN (langdetect only) 1.1B documents and 1.4T SpaCy tokens; C4.EN.NOBLOCKLIST (all rules except the blocklist) 395M documents and 198B tokens; C4.EN 365M documents and 156B tokens (Table 1).

**Worked example (derived from Table 1).**
- Rules other than the blocklist remove 1 − 198/1,400 = 85.9% of tokens.
- The blocklist removes 1 − 365/395 = 7.6% of the remaining documents but 1 − 156/198 = 21.2% of the remaining tokens.
- Removed documents therefore average 42B/30M = 1,400 tokens, against 156B/365M ≈ 427 tokens for kept documents. A page-level rule that fires on any one word removes long documents more often, because a long document has more chances to contain a listed word (Interpretation).

**What the blocklist removed** (§5): of 100,000 excluded documents clustered into 50 clusters, 16 clusters (31% of excluded documents) are largely sexual; other clusters cover science, medicine, health, legal, and political text. Mentions of sexual orientations have the highest likelihood of removal; among 50 sampled excluded documents mentioning "lesbian" and "gay", 22% and 36% are non-offensive or non-sexual. Documents classified as African American English and Hispanic-aligned English are removed at 42% and 32%, against 6.2% for White-aligned and 7.2% for other English; 97.8% of C4.EN is White-aligned, 0.07% AAE, and 0.09% Hispanic-aligned (§5.3). **Result (single study).** These removals are attributed to the blocklist. The effect of the langdetect 0.99 threshold on dialects is not measured in Dodge et al. or in the T5 paper.

### §3.4 Re-implementations change the thresholds

The rule name does not fix the thresholds. T5 §2.2 and the TFDS code use 5 words per line and 3 sentences per page. Dodge et al. §2 describe "fewer than three words" and "less than five sentences" ([[c4]] Connections). The datatrove `C4QualityFilter` at commit acd431b sets `min_num_sentences: int = 5` and `min_words_per_line: int = 3` (`src/datatrove/pipeline/filters/c4_filters.py` L66–67), and FineWeb's released script calls this class with only `filter_no_terminal_punct=False` changed (§5.6).

## §4 Dolma: per-source pipelines and the filter ablations reported

**Definition.** Dolma v1.6 (Soldaini et al., arXiv 2024-01) is a 3,059B-token English corpus from Common Crawl, GitHub, Reddit, Semantic Scholar, Project Gutenberg, and Wikipedia/Wikibooks, each with its own pipeline ([[dolma]] Table 1). The web subset is 2.28T tokens from 25 snapshots, 2020-05 to 2023-06 (§5).

### §4.1 Web pipeline order and removal rates

Order ([[dolma]] §5.5): CCNet output → URL dedup → document dedup → quality filters → content filters → paragraph dedup.
1. **CCNet stage** (§5.1): fastText English score ≥ 0.5 removes 61.7% of data by bytes; within-snapshot paragraph dedup removes about 70% of paragraphs; CCNet overall removes 84.2%, 175.1 TB → 27.7 TB.
2. **Deduplication** with a Bloom filter (§5.4): exact URL removes 53.2% of documents; exact document removes 14.9% of URL-deduplicated documents; exact paragraph removes 18.7% of paragraphs. URL and document dedup run first "to increase efficiency"; paragraph dedup runs last because "paragraph removal risks disrupting content analysis". The order is justified by cost; no ablation of the order is reported (§5.4).
3. **Quality** (§5.2): all Gopher rules ("Gopher All", 15.23% of characters tagged) plus one C4 rule ("C4 NoPunc", 22.73%). Gopher rules include: fewer than 50 or more than 100K words; median word length below 3 or above 10; symbol-to-word ratio above 0.10; more than 0.90 of lines starting with a bullet; more than 0.30 of lines ending with an ellipsis; more than 0.30 duplicated lines (App. N.4).
4. **Toxicity** (§5.3): two fastText classifiers trained on Jigsaw Toxic Comments score sentences; sentences above τ = 0.4 are removed.
5. **PII** (§5.3): regular expressions for email addresses, IP addresses, and phone numbers.

**Granularity conflict inside Dolma.** §5.2 describes C4 NoPunc as removing "paragraphs that do not end in punctuation". The datasheet (App. N.4) describes the same 22.73% filter as "Remove documents with more than half of their line not ending in '.', '?', '!', or '"'". The paper does not reconcile the two.

### §4.2 Filter ablations

Setting: 1.2B-parameter OLMo-architecture models trained to 150B tokens, evaluated zero-shot on 8 datasets chosen to avoid contamination: ARC-E, ARC-C, BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande (§4.2, Table 2, App. D). Results are shown as curves over training tokens: HellaSwag in the main text, other tasks and Paloma subsets (for example Twitter AAE, 4chan, M2D2 S2ORC) in App. O. The paper gives no numeric table for these ablations.
- **Quality rules** (§5.2, Figure 1, HellaSwag shown; other tasks in App. O): "C4 NoPunc on its own outperforms both C4 All as well as Gopher All on both perplexity and downstream tasks", and "Gopher All + C4 NoPunc offers the best performance".
- **Toxicity threshold** (§5.3, Figure 2): τ = 0.4 removes 5.5–7.3% but "generally yields lower downstream performance" than τ = 0.0004, which removes 29.1–34.9%; the authors adopted τ = 0.4 "to ensure we meet our minimum token count requirement".
- **Stacking** (§5.5, Figure 3): adding dedup and content filters to quality filters has a "positive compounding effect" on HellaSwag. The filters overlap little in what they remove (§5.3, App. J).
- **Heuristics versus reference perplexity** (§5.2): the CCNet KenLM bucket proportions (high 21.9%, medium 28.5%, low 49.6%) did not change after Dolma's heuristic filters, which the authors read as the two signals being orthogonal (Interpretation).

**Worked example (PII rule, §5.3).** A document with 5 detected spans keeps its text with each span replaced by a token such as `|||EMAIL_ADDRESS|||`. A document with 6 spans is removed. The paper reports 0.02% of documents masked and 0.001% removed, and no measured effect of removal versus replacement; the datasheet lists 0.05% tagged for masking and 0.11% tagged for removal (App. N.4).

### §4.3 Per-source pipelines

Code (§6): The Stack (already MinHash- and LSH-deduplicated by its creators), data files such as JSON and CSV removed, RedPajama v1 plus StarCoder rules (lower code perplexity than RedPajama v1 rules alone), and removal of documents with detect-secrets matches. Reddit (§7): comments under 500 characters, submissions under 400 characters, and comments with fewer than 3 votes removed; 26,123 banned or NSFW subreddits excluded; treating comments and submissions as separate documents scored better than thread linearization (Figure 4). peS2o used as-is; Project Gutenberg deduplicated by exact title; Wikipedia documents of 25 or fewer words removed (§8). **Limits** (Limitations): ablations use one 1B-scale dense architecture, which "might result in design decisions that are not relevant at larger model sizes"; Dolma v1.6 as released is not decontaminated (App. N.4).

## §5 FineWeb: extraction, filters chosen by ablation, per-snapshot MinHash

**Definition.** FineWeb (Penedo et al., arXiv 2024-06) is a 15T GPT-2-token English corpus built from 96 Common Crawl snapshots, where each stage was chosen by data ablations ([[fineweb]] §3).

### §5.1 Ablation protocol

Models of 1.71B parameters (Llama architecture, sequence length 2,048, global batch about 2M tokens) are trained on about 28B tokens for filtering ablations and on 350B tokens for some deduplication and cumulative runs; each data version gets two runs with different data subsets and seeds, averaged ([[fineweb]] §3.1). Benchmarks: CommonSense QA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC, MMLU, chosen for low variance between runs, near-monotonic improvement during training, and above-random scores at this scale (§3.1). Over 70 models were trained, an estimated 80,000 H100 GPU hours (§3.1).

### §5.2 Extraction and base filtering

Trafilatura applied to WARC files outperformed Common Crawl WET text in a 28B-token ablation with only language filtering applied (§3.2, Figure 1). Base filtering (§3.3) applies a URL blocklist for adult content, fastText English score ≥ 0.65, and MassiveText (Gopher) quality and repetition filters with their original thresholds, producing about 36T tokens from 96 snapshots.

### §5.3 MinHash parameters

**Definitions.** Split a document into word 5-grams (shingles). The Jaccard similarity of documents A and B is `J(A, B) = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|`, where `S(D)` is the set of shingles of D ([[minhash-lsh]] §2). Under one uniformly random permutation of the shingle space, the probability that the smallest shingle of S(A) ∪ S(B) lies in both S(A) and S(B), which means A and B have the same minimum, equals J(A, B) ([[minhash-lsh]] §3, Theorem 1). A MinHash implementation approximates each random permutation with a hash function.

**Worked example (shingle size).** A = "the cat sat on the mat today", B = "the cat sat on a mat today". With single words, J = 6/7 = 0.86; with bigrams, 4/8 = 0.50; with trigrams, 2/8 = 0.25. For two 1,000-word documents that differ in one word, each has 996 word 5-grams, 5 of which differ, so J = 991/1,001 = 0.99. Longer documents tolerate small edits at a fixed shingle size.

**FineWeb setting** ([[fineweb]] §3.4, App. E.1): 112 hash functions split into 14 buckets of 8; two documents are candidate duplicates if all 8 minhashes agree in at least one bucket; candidates are clustered transitively and one random document per cluster is kept. The match probability is:

```
P(s) = 1 − (1 − s^8)^14
```

- `s` is the n-gram (Jaccard) similarity of the two documents.
- `s^8` is the probability that one bucket matches; `(1 − s^8)^14` is the probability that no bucket of 14 matches.

At s = 0.8: 0.8^8 = 0.168; (1 − 0.168)^14 = 0.076; P = 0.92. The paper reports P = 56%, 77%, 92%, 98.8% at s = 0.7, 0.75, 0.8, 0.85 (App. E.1). At s = 0.5, P = 0.053. RefinedWeb used 9,000 hashes in 450 buckets of 20, which gives a steeper curve at more compute (App. E.1, Figure 13). The released script sets the same values (`datatrove@acd431b examples/fineweb.py` L80–88: `num_buckets=14`, `hashes_per_bucket=8`, `n_grams=5`). The MinHash panel of [figures/pipeline-compare.html](figures/pipeline-compare.html) lets the reader vary buckets and hashes per bucket and read P(s).

### §5.4 Global versus per-snapshot deduplication, and the reason given

1. **Global run** (§3.4, Figure 3): MinHash across all 96 snapshots, iterating from the newest (2023-50) to the oldest, removed up to 90% of the base-filtered data of old snapshots and left 4T tokens. A 350B-token run showed "little improvement" over non-deduplicated data.
2. **Diagnosis** (§3.4, Figure 4): in snapshot 2013-48, the ~31B tokens kept by global dedup trained a worse model than 171B tokens obtained by deduplicating the ~460B removed tokens on their own. By visual inspection, the kept data "contains more ads, incoherent lists of keywords and generally badly formatted text". The paper describes the kept share as 10%; 31/(31 + 460) = 6.3% (derived).
3. **Per-snapshot run** (§3.4, Figure 5): deduplicating each snapshot independently gave 20T tokens and matched RefinedWeb.
4. **Lighter global methods** on top (App. E.3, Figure 15): URL dedup (71.5% of tokens removed), line dedup (77.8%), line dedup with minimum words (85%), 3-line dedup (80.9%) all scored below per-snapshot MinHash alone.

**Reason given.** The authors hypothesize that most of the gain comes from removing large duplicate clusters present in all crawls, and that removing clusters with fewer than about 100 copies (the number of crawls) "can harm performance" (Interpretation, §3.4). A course reading of step 2 (Interpretation, not tested by the paper): a 2013-48 page survives newest-to-oldest global dedup only if no near-duplicate exists in any later snapshot, so pages that were re-crawled for years are removed from the old snapshot, and pages that appeared once are kept. **Measurement limit** (App. E.2, Figure 14): in a simulation of 100 identical 200B-token snapshots, a 1B-token sample is almost all unique documents, so a 1B-token ablation cannot show deduplication effects. **Result (single study);** ch-12 compares this with Llama 3's global deduplication.

### §5.5 Re-ablating C4's rules and deriving new ones

On the base-filtered, per-snapshot-deduplicated 2019-18 crawl ([[fineweb]] §3.5, Figure 6): the terminal-punctuation rule gave the largest individual HellaSwag gain but removed about 30% of tokens; the curly-bracket and word-length rules removed 2.8% and 4.3%; lorem-ipsum, javascript, and policy rules each removed under 0.5%; all rules except terminal punctuation removed about 7% and scored higher than terminal punctuation alone. FineWeb kept all C4 rules except terminal punctuation.

**New rules** (§3.6): the authors computed over 50 statistics on the per-snapshot (treated as higher quality) and global (treated as lower quality) deduplicated versions of 2013-48, set thresholds where the lower-quality set had higher density, tested 16 candidates at 28B tokens (App. E.4, Table 2), and kept three. Remove documents where the fraction of lines ending in punctuation is ≤ 0.12 (10.14% of tokens), the fraction of characters in duplicated lines is ≥ 0.1 (12.47%), or the fraction of lines shorter than 30 characters is ≥ 0.67 (3.73%). Together they removed about 22% of tokens and raised the aggregate score by about 1%. The "higher" and "lower" quality labels here come from the deduplication outcome, not from human judgments.

**Worked example (granularity).** A 10-line document has 3 lines ending in punctuation, each with at least 5 words.
- C4 line rule: keeps the 3 lines and deletes 7 (the page survives if the 3 lines contain at least 3 sentences).
- Dolma datasheet rule: 7 of 10 lines lack punctuation, 0.7 > 0.5, so the whole document is removed.
- FineWeb rule: 3/10 = 0.3 > 0.12, so the whole document is kept unchanged.

With 1 punctuated line of 10, the FineWeb ratio is 0.1 ≤ 0.12 and the document is removed. The granularity panel of [figures/pipeline-compare.html](figures/pipeline-compare.html) applies the three rules to a document with an adjustable number of punctuated lines.

**Dolma and FineWeb disagree on the punctuation rule. Open question.** Dolma kept C4 NoPunc and dropped the other C4 rules; FineWeb dropped terminal punctuation and kept the other C4 rules. In Dolma, C4 NoPunc alone outperformed the full C4 rule set and Gopher All; in FineWeb, terminal punctuation gave the largest individual HellaSwag gain among the C4 rules tested. The setups differ in granularity (paragraph or document in Dolma, line in FineWeb), baseline data (Dolma: CCNet-processed WET text, §5.1; FineWeb: trafilatura extraction with base filtering and per-snapshot MinHash), model size (1.2B versus 1.71B), and token budget, and FineWeb gave token removal as its reason. No source ablates the rule with these factors held fixed.

### §5.6 Final order, and a conflict with the released code

The paper's final pipeline (§3.7) is WARC extraction → base filtering → per-snapshot MinHash → C4 rules → custom rules, followed by anonymizing email and public IP addresses. The datasheet (App. A) lists MinHash after the custom filters ([[fineweb]] Technical Details). The released script (`datatrove@acd431b examples/fineweb.py` L41–62) runs all filters per dump before MinHash:

```python
        URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
        Trafilatura(favour_precision=True),
        LanguageFilter(
        ...
        GopherRepetitionFilter(
        ...
        GopherQualityFilter(
        ...
        C4QualityFilter(
            filter_no_terminal_punct=False,
        ...
        FineWebQualityFilter(
```

`FineWebQualityFilter` at the same commit defaults to `line_punct_thr: float = 0.12`, `short_line_thr: float = 0.67`, and `char_duplicates_ratio: float = 0.01` (`fineweb_quality_filter.py` L14–18). The last value matches App. E.4 Table 2 (≤ 0.01) and differs from the §3.6 text (≥ 0.1). The filter also removes documents where newlines per word exceed 0.3 (`new_line_ratio`, L19, L49–52), a rule the paper does not describe. Status: **conflict**; the paper, datasheet, and pinned code are separate facts, and the code commit (2024-08) is later than the paper's experiments.

## §6 Filters as distribution selectors: measuring what a filter removes

A removal rate says how much text a filter removes; it does not say which domains lose text. Two measurements address that: per-domain perplexity of models trained on the filtered data, and audits of the removed documents by topic and dialect.

**Per-domain perplexity** ([[paloma]] §3, §4.1). For an evaluation document set N, `ℓ = Σ_{t∈N} Σ_i ln p(t_i | t_<i)` and `perplexity = exp(−ℓ / T(N))`, where `t` is a document, `t_i` its i-th token, and `T(N)` the token count. The macro average over a domain set D is:

```
PPL_macro(D) = (1/|D|) · Σ_{d∈D} perplexity(d)
```

- `D` is the set of domains (for example subreddits, programming languages, S2ORC fields); `perplexity(d)` is computed on domain d alone.

**Worked example.** A model has perplexity 15 on news and 60 on arXiv. If news is 90% of evaluation tokens, the token-weighted perplexity is exp(0.9·ln 15 + 0.1·ln 60) = exp(2.847) ≈ 17.2; the macro average is (15 + 60)/2 = 37.5. The token-weighted value (17.2) is 2.2 above the news value, while the macro average (37.5) moves halfway toward the arXiv value, so the macro average shows the weak domain and the token-weighted value mostly hides it.

**Evidence.**
- **Single-source web corpora** ([[dolma]] §9.2, Figure 5; [[paloma]] §4.1): 1.2B models trained on 150B tokens of C4, mC4-en, or RefinedWeb have higher average Paloma perplexity than models trained on the Pile, Dolma, or RedPajama. The C4 model reaches perplexity 391,171 on RedPajama arXiv and 14 on Dolma peS2o; RefinedWeb and mC4-en models reach 21,652 and 1,409 on the Max programming-language domain. The Paloma authors note that these domains contain LaTeX and code and suggest "a lack of exposure to specific types of language completely filtered due to having only one set of cleaning filters applied to a single source of data" (Interpretation). A course reading (Interpretation, untested): C4's page-level `{` rule, written to remove code (§3.1), also removes pages containing LaTeX.
- **Heterogeneous sources** ([[pretrainers-guide-training-data]] §6): removing Common Crawl, Books, or OpenWeb from the Pile degraded average downstream QA performance most for 1.5B decoder models. **Replicated** with [[dolma]] §9.2 in direction (sources and metrics differ).
- **Topic and domain shift of a filter** ([[fineweb]] §4.1–§4.2): the FineWeb authors embed 50k documents from each corpus, cluster them, and compare cluster shares, then compare Paloma perplexity without decontamination. Applied to the classifier-filtered FineWeb-Edu, the method shows "Education, Learning, Teaching" +3.2% and "History, Culture, Politics" +2.2%, with Business, Entertainment, and Travel clusters down-sampled; FineWeb fits Twitter AAE, Manosphere, Gab, 100 Subreddits, and 4chan better, and FineWeb-Edu fits Wikipedia, S2ORC, arXiv, and 100 programming languages better. The same method applies to any heuristic filter; ch-10a covers the classifier itself.
- **Filters shift capabilities unevenly** ([[pretrainers-guide-training-data]] §5, Figures 5–7): 1.5B decoder models on C4 (their version without the blocklist, deduplicated), each fine-tuned per task. Perspective API toxicity filtering at T = 0.3 keeps 61% of the data and changes average QA performance by −2.7 points (Wiki −3.8, Web −4.4) and lowers toxicity identification; an inverse filter that removes the least toxic documents (T = 0.06, 92% kept) gives +1.7 and the best toxicity identification. A classifier quality filter at T = 0.975 (91% kept) gives +2.5 on average but −2.2 on Books QA, and "the benefits are not predictable from text characteristics" (§1). **Limits:** single-shot experiments, fine-tuned evaluation only (§8).

## §7 Reading a new pipeline: the four pipelines on one template

| Field | CCNet (Feb 2019) | C4 (Apr 2019) | Dolma v1.6 web | FineWeb (96 snapshots) |
|---|---|---|---|---|
| Stage order | paragraph dedup → LID → 5-gram PPL buckets | line/page rules, span dedup, langdetect (order in code: §3.1) | CCNet → URL → doc dedup → quality → toxicity → paragraph dedup | WARC extraction → URL/LID/Gopher → per-snapshot MinHash → C4 rules → custom rules (paper; code differs, §5.6) |
| Order evidence | dedup-before-LID ablation (Fig. 3) | none | cost argument; not ablated | iterative ablations in §3 order |
| Granularity | paragraph (dedup); document (LID, bucket) | line and page | document, sentence (toxicity), paragraph (dedup), span (PII) | document (filters); duplicate cluster (MinHash); span (PII) |
| Thresholds and choice | LID > 0.5; terciles per language | 5 words, 3 sentences, langdetect ≥ 0.99; not ablated | Gopher originals; C4 NoPunc; τ = 0.4 chosen for token count | LID ≥ 0.65; custom thresholds from histograms, then 28B ablations |
| Removal (unit) | 42%→28% chars remain after dedup (1→100 shards) | 1.4T → 156B SpaCy tokens (Dodge) | 84.2% by CCNet (TB); later stages per §4.1 | 36T → 20T → 15T GPT-2 tokens |
| Ablation model | BERT-BASE, fastText embeddings | T5 220M, 2^35 tokens | 1.2B, 150B tokens | 1.71B, 28B/350B tokens, 2 seeds |
| Evaluation gaps | no generative LM; no unfiltered-CC baseline | per-rule effects; dialects (later audit) | 8 zero-shot tasks; Paloma subsets as App. O figures without numbers | 8-task commonsense and knowledge aggregate; Paloma check only for FineWeb vs FineWeb-Edu |

Checks for a new release, each tied to an observed failure in this chapter:
1. Read the paper, the datasheet, and a pinned script, and record each order and threshold separately (FineWeb §5.6; C4 §3.1).
2. Convert every removal rate to one unit before comparing stages (§3.3 shows 7.6% of documents = 21.2% of tokens).
3. Ask whether a threshold was chosen by ablation, by histogram, or by token budget (Dolma τ = 0.4; FineWeb §3.6).
4. Check the ablation token budget against the effect being measured; deduplication effects appear only when the sample contains repeats; in the FineWeb simulation repeats start to appear near 100B tokens (App. E.2).
5. Look for per-domain perplexity or removal audits on domains outside the benchmark suite (§6).

## Negative samples and negative feedback

Heuristic pipelines produce negatives in the first sense of the style standard only: **negative marginal value**, meaning documents judged to lower performance when used as positive training targets. They are labeled by rules, word lists, reference perplexity, hash collisions, and fastText classifiers; no human or verifier label is involved. Current practice discards them, with three exceptions that keep information: CCNet releases tail buckets instead of removing them ([[ccnet]] §5.2), Dolma masks PII spans in documents with 5 or fewer spans ([[dolma]] §5.3), and Dolma removes toxic sentences rather than whole documents (§5.3). No stage uses removed text as content, as conditioning, or as gradient, so likelihood-displacement risks do not apply at this stage.

**Documents removed in error (reported).** In the blocklist-excluded sample, 22% of documents mentioning "lesbian" and 36% mentioning "gay" were non-offensive or non-sexual (Dodge et al. §5.2, via [[c4]]). CCNet reports valid spoken-like text and specialized forums in the tail (§5.2). FineWeb's global deduplication removed data that trained a better model than the data it kept (§3.4).

**Evidence that removed text has value for some capabilities.** The inverse toxicity filter, which keeps the most toxic documents, gave the best toxicity identification and +1.7 average QA ([[pretrainers-guide-training-data]] §5). The two studies point in different directions. In Dolma, the lower toxicity threshold removed 29.1–34.9% of sentences and generally scored better on its 8-task suite than the adopted threshold, which removed 5.5–7.3% ([[dolma]] §5.3). In the Pretrainer's Guide, the Perspective API filter at T = 0.3, which removed 39% of documents, lowered average QA by 2.7 points and lowered toxicity identification relative to the unfiltered baseline. The filters (sentence-level fastText versus document-level Perspective API), models, and evaluations differ, so the value of removed text depends on the capability measured and has to be evaluated per capability.

**Controls.** Store filter scores as attributes and decide later (CCNet buckets; the Dolma toolkit writes duplicate spans under `attributes/`, [[dolma]] Connections). Prefer smaller granularity where the evidence supports it (sentence-level toxicity removal; span masking for PII). Keep a random sample of removed documents per filter for audit.

**Diagnostics.** Log removal counts per filter in documents and tokens; cluster removed documents by topic (Dodge §5.1; [[fineweb]] §4.1); run a dialect classifier on removed and kept sets (Dodge §5.3); compare per-domain perplexity with and without the filter ([[paloma]]).

**Effect on generality.** Removal lowers coverage of the removed domains (§6); for heuristic filters, no source in this chapter reports effects on calibration, hallucination, or over-refusal.

## Recipe

Rows below are data-pipeline settings and the ablation models that selected them. The style standard's stage list has no data-curation stage, so pipeline rows use `pretrain-stable (data)`.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| CCNet, Feb 2019 snapshot | n/a (data) | pretrain-stable (data) | paragraph dedup scope | SHA-1 (64 bits) of normalized paragraphs; compared across 50 shards (~3% of corpus) | arXiv:1911.00359v2 §3.2, §4.2 | verified 2026-09-15 | Fig. 4–5: characters remaining vs RAM (42% at 1 shard, 28% at 100) |
| CCNet | n/a (data) | pretrain-stable (data) | stage order | dedup before LID | §4.1, Fig. 3 | verified 2026-09-15 | Fig. 3: per-language document ratio on 1% of Feb 2019 |
| CCNet | n/a (data) | pretrain-stable (data) | LID threshold | fastText top-language score > 0.5 | §3.3 | verified 2026-09-15 | no ablation reported |
| CCNet | n/a (data) | pretrain-stable (data) | quality score | SentencePiece + 5-gram KenLM on Wikipedia, 48 languages; per-language terciles; no removal | §3.4, §5.2 | verified 2026-09-15 | Table 1 (head > middle > tail analogies); Table 2 (CC head vs Wikipedia) |
| C4 (T5 paper) | n/a (data) | pretrain-stable (data) | line and page rules | line ends in terminal punctuation; ≥ 5 words per line; ≥ 3 sentences per page; page removed on blocklist word, "lorem ipsum", or "{" | arXiv:1910.10683v4 §2.2 | verified 2026-09-15 | Table 8: all rules vs none only |
| C4 (T5 paper) | n/a (data) | pretrain-stable (data) | language ID; dedup | langdetect English p ≥ 0.99; all but one of repeated three-sentence spans | §2.2 | verified 2026-09-15 | no ablation reported |
| C4 (TFDS code) | n/a (data) | pretrain-stable (data) | dedup unit; extra rules | line-level dedup; lines ending "..." removed; words > 1,000 characters remove the line | tensorflow_datasets/text/c4_utils.py (master, fetched 2026-09-14) L41–45, L244–248, L345–370 | conflict | paper states three-sentence spans; code release date vs 2019 build not stated |
| T5 baseline | ~220M | pretrain-stable | data-ablation tokens | 2^35 ≈ 34B, no repetition | §3.1.2, Table 8 | verified 2026-09-14 ([[c4]]) | SuperGLUE 71.36 (C4) vs 68.04 (unfiltered) |
| datatrove C4QualityFilter | n/a (data) | pretrain-stable (data) | defaults | min_num_sentences 5; min_words_per_line 3 | github.com/huggingface/datatrove@acd431b src/datatrove/pipeline/filters/c4_filters.py L66–67 | conflict | differs from T5 §2.2; no ablation reported |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | LID | fastText English ≥ 0.5 (61.7% bytes removed) | arXiv:2402.00159v2 §5.1 | verified 2026-09-14 ([[dolma]]) | no ablation reported |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | quality rules | Gopher All + C4 NoPunc | §5.2 | verified 2026-09-14 | Fig. 1: C4 NoPunc > C4 All, Gopher All; combination best (1.2B, HellaSwag shown) |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | C4 NoPunc granularity | paragraphs not ending in punctuation (§5.2) / documents with > half of lines not ending in punctuation (App. N.4) | §5.2; App. N.4 | conflict | 22.73% of characters in both |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | toxicity threshold | τ = 0.4, sentence removal | §5.3 | verified 2026-09-14 | Fig. 2: τ = 0.0004 generally better; 0.4 chosen to meet token count |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | PII rule | ≤ 5 spans masked; ≥ 6 spans document removed | §5.3 | verified 2026-09-15 | removal vs replacement: no effect measured |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | dedup order | URL → document → (quality, content) → paragraph, Bloom filter | §5.4, §5.5 | verified 2026-09-14 | efficiency argument; order not ablated |
| Dolma data-ablation model | 1.2B | pretrain-stable | tokens; eval | 150B tokens; 8 zero-shot tasks | §4.2; App. D.1 | verified 2026-09-14 | no ablation reported |
| FineWeb | n/a (data) | pretrain-stable (data) | extraction | trafilatura on WARC | arXiv:2406.17557v2 §3.2 | verified 2026-09-15 | Fig. 1: WARC+trafilatura > WET at 28B tokens |
| FineWeb | n/a (data) | pretrain-stable (data) | LID; base quality | fastText English ≥ 0.65; MassiveText quality and repetition, original thresholds; URL blocklist | §3.3 | verified 2026-09-15 | Fig. 2: base filtering > unfiltered WARC (combined, not per rule) |
| FineWeb | n/a (data) | pretrain-stable (data) | MinHash | word 5-grams; 112 hashes = 14 buckets × 8; per snapshot | §3.4, App. E.1; datatrove@acd431b examples/fineweb.py L80–88 | verified 2026-09-15 | Fig. 3–5, 15: global and lighter global methods below per-snapshot |
| FineWeb | n/a (data) | pretrain-stable (data) | C4 rules | all except terminal punctuation | §3.5 | verified 2026-09-15 | Fig. 6: all-but-punct (~7% removed) > terminal punct alone (~30% removed) |
| FineWeb | n/a (data) | pretrain-stable (data) | custom rules | line-punct ratio ≤ 0.12; short-line (< 30 chars) ratio ≥ 0.67 | §3.6; fineweb_quality_filter.py L14–16 | verified 2026-09-15 | Fig. 7, App. E.4: three rules ~22% removed, aggregate +~1% at 28B |
| FineWeb | n/a (data) | pretrain-stable (data) | duplicated-line character ratio | ≥ 0.1 (§3.6 text) / 0.01 (App. E.4 Table 2; code L18) | §3.6; App. E.4; datatrove@acd431b | conflict | 12.47% tokens removed in Table 2 row |
| FineWeb data-ablation model | 1.71B | pretrain-stable | tokens; seeds | ~28B (filters), 350B (dedup, cumulative); 2 runs per data version | §3.1 | verified 2026-09-14 ([[fineweb]]) | 28B described as about Chinchilla-optimal for 1.71B |

**Starting point for a small general-purpose run.** Every value below comes from a verified row above, with the conditions of its source. For English web text from more than one snapshot, extract text from WARC with trafilatura, keep documents with fastText English score ≥ 0.65, and apply the Gopher quality and repetition rules at their original thresholds (FineWeb, 96 snapshots, selected at 1.71B parameters and 28B tokens). Deduplicate each snapshot separately with word 5-grams and 14 buckets of 8 hashes (FineWeb, compared at 350B tokens). Apply the C4 rules except terminal punctuation and the FineWeb line-punctuation (≤ 0.12) and short-line (≥ 0.67) rules; the duplicated-line threshold is a conflict row and needs its own ablation. For a multilingual corpus, deduplicate paragraphs before language identification and assign a page to its top language only when the score exceeds 0.5 (CCNet, February 2019 snapshot). Mask PII spans in documents with 5 or fewer spans and remove documents with 6 or more (Dolma, no measured effect at 1.2B). The FineWeb and Dolma values were selected on commonsense and knowledge benchmarks at 1.2–1.71B parameters; check per-domain perplexity before adopting them for a model that must cover code, LaTeX, or dialectal text.

## Generalization lens

**(a) What increases breadth.**
- Deduplicating repeated boilerplate before language identification keeps more low-resource-language documents ([[ccnet]] §4.1, Fig. 3; Result, single study).
- Releasing perplexity buckets and filter scores as labels lets later mixtures recover tail text such as spoken-like comments and specialized forums ([[ccnet]] §5.2).
- Combining filtered web text with curated non-web sources improves fit to diverse domains ([[dolma]] §9.2) and downstream QA ([[pretrainers-guide-training-data]] §6) (Replicated in direction).
- Per-snapshot MinHash kept 20T tokens and matched RefinedWeb, while global MinHash kept 4T tokens with little gain ([[fineweb]] §3.4).
- Smaller removal granularity keeps the rest of a document: Dolma removes toxic sentences and masks PII spans instead of dropping documents ([[dolma]] §5.3).

**(b) What causes narrowing.**
- Word blocklists remove dialects and identity-related text at unequal rates: AAE 42% and Hispanic-aligned English 32% versus White-aligned 6.2% ([[c4]], Dodge et al. §5.3).
- Code-removal rules and single-source web data leave gaps on LaTeX and code domains: C4 baseline perplexity 391,171 on RedPajama arXiv ([[paloma]] §4.1; the link to the `{` rule is Interpretation).
- Reference-anchored scores prefer the reference corpus's authors and registers ([[ccnet]] §5.2; [[whose-language-counts-quality-filter]] §3.4).
- Toxicity filtering lowers toxicity identification and average QA ([[pretrainers-guide-training-data]] §5, Fig. 5, 7).
- English-only language identification makes non-English ability depend on residual text; Dolma's authors state that the corpus "reinforces the expectation of English being the 'default' language for NLP" ([[dolma]] Limitations).
- Gopher's rule removing documents over 100K words ([[dolma]] App. N.4) removes the longest documents; its effect on long-context ability is not reported by any source here.
- Thresholds selected on HellaSwag-type benchmarks select for what those benchmarks reward; FineWeb chose its benchmarks for low variance at small scale ([[fineweb]] §3.1), which is a statistical criterion, not a coverage criterion (Interpretation).

**(c) How to measure it at this stage.**
- Per-domain macro perplexity with decontaminated training data ([[paloma]] G1–G5); FineWeb's §4.2 comparison intentionally skipped decontamination, so its domain-fit numbers include possible overlap.
- Removal audits: topic clusters of removed documents, identity-term PMI, and dialect shares before and after each filter (Dodge et al. §5, via [[c4]]).
- Data ablations with the token budget matched to the effect: 28B tokens for filters, 350B for deduplication ([[fineweb]] §3.1, App. E.2), two or more seeds.
- Known measurement errors: no single Paloma source correlates with all 8 downstream tasks tested ([[paloma]] App. A); ranking agreement between adjacent checkpoints averages 0.513 ([[paloma]] App. A); Dolma's filter ablations are reported only as curves (HellaSwag in the main text, other tasks and Paloma subsets in App. O), and FineWeb's as curves of an aggregate score, so effect sizes must be read from figures.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Re-implementing CCNet as LID → dedup → perplexity | fewer documents for low-resource languages than CCNet statistics | run both orders on a 1% sample and compare per-language document counts (CCNet Fig. 3 method) |
| Copying "C4 filters" by name | removal rate differs from the reference; short lines or pages kept or removed unexpectedly | print thresholds; compare with T5 §2.2 (5 words, 3 sentences) and datatrove defaults (3 words, 5 sentences) |
| Comparing removal rates in different units | a filter that removes "7.6%" is treated as minor | report documents and tokens with the tokenizer name; Dodge Table 1 gives 7.6% of documents = 21.2% of tokens |
| Global MinHash across many snapshots | old snapshots shrink to about 10% or less; little benchmark gain | per-snapshot versus global ablation at 350B tokens; inspect kept samples for ads and keyword lists |
| Measuring deduplication with a 1B-token sample | no difference between dedup variants | count duplicate-cluster sizes directly; evaluate dedup variants at 350B tokens as FineWeb did (§3.1; simulation in App. E.2) |
| Choosing thresholds on one benchmark family | fit to code, LaTeX, or dialect domains worsens without benchmark change | Paloma macro perplexity per domain before and after the filter |
| Page-level word blocklist | disproportionate removal of AAE, Hispanic-aligned English, and LGBTQ-related documents | dialect classifier and identity-term PMI on removed versus kept documents (Dodge §5) |
| Dropping documents by reference-perplexity cutoff | loss of spoken-like text and specialized forums | sample and read tail documents; keep buckets as attributes (CCNet §5.2) |
| Assuming the paper's stage order is the released order | reproduced token counts differ from the release | compare paper, datasheet, and pinned code (FineWeb §3.7, App. A, examples/fineweb.py) |

## Check your understanding

1. CCNet deduplicates paragraphs before language identification. Explain how this order changes the number of documents assigned to a low-resource language, and predict the direction of the change if deduplication were limited to one shard instead of 50.
2. In Dodge et al.'s Table 1, the blocklist removes 7.6% of documents and 21.2% of tokens. Explain which property of a page-level word rule produces this difference, and what it implies for long-document coverage.
3. FineWeb's global MinHash kept ~31B tokens from 2013-48 that trained worse than 171B tokens rebuilt from the removed data. Explain how processing snapshots from newest to oldest decides which old pages survive, and why a 1B-token ablation would not have detected the problem.
4. With FineWeb's 112 hashes, compute P(s = 0.6) for 14 buckets of 8 and for 8 buckets of 14. Explain which setting finds more near-duplicates and what that costs in false matches at lower similarity.
5. Dolma kept C4 NoPunc alone and FineWeb kept all C4 rules except terminal punctuation. List the setup differences that prevent reading these as contradictory results, and design one ablation that would decide between them.
6. A filter raises the average of eight commonsense benchmarks at 1.7B parameters. Explain how the same filter can lower fit to arXiv and programming-language text, and which measurement from §6 would detect it.
7. The inverse toxicity filter in the Pretrainer's Guide improved toxicity identification. Using the definition of negative marginal value, explain what this result says about labeling removed documents as negatives for a general-purpose model.

## Connections

- **Previous:** ch-09 — Pretraining Data Composition and Capability Coverage (source-level composition; this chapter selects within the web source).
- **Next:** ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (FineWeb-Edu, DCLM, Nemotron-CC classifiers; uses the §6 measurements).
- **Extends §4.1 PII rules:** ch-11 — Tokenizers, Data Provenance, and PII Removal.
- **Extends §2.1 and §5.3–§5.4:** ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic (MinHash parameters, global versus per-snapshot deduplication compared with Llama 3).
- **Extends §2 language identification:** ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (FineWeb2 per-language filters).
- **Measurement background for §6:** ch-00 — What General Capability Means and How It Is Measured (Paloma per-domain perplexity).
- **Lab that applies this chapter:** ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement.

## Sources

- [[ccnet]] — stage order and its ablation (§4.1), paragraph dedup scope (§3.2, §4.2), LID threshold (§3.3), KenLM buckets and the no-removal decision (§3.4, §5.2), Tables 1–2.
- [[c4]] — §2.2 rule list, Table 8 filtered versus unfiltered, TFDS code pointer, and Dodge et al. 2021 audit numbers (Table 1, §5) recorded in its Connections.
- [[dolma]] — web pipeline order and removal rates (§5), quality and toxicity ablations (Figures 1–3), PII rule, per-source pipelines, domain fit (§9.2), limitations.
- [[fineweb]] — ablation protocol (§3.1), extraction and base filtering, MinHash parameters and dedup study (§3.4, App. E), C4 re-ablation and custom rules (§3.5–§3.6), final order (§3.7), topic and domain analyses (§4.1–§4.2).
- [[minhash-lsh]] — Jaccard resemblance, shingling, and the min-hash estimator (Theorem 1).
- [[paloma]] — perplexity and macro-average definitions, single-source baseline spikes (§4.1), measurement cautions (App. A).
- [[whose-language-counts-quality-filter]] — reference-anchored quality scores and their demographic and factuality alignment (§3.2–§4.2); chapter excerpt, no library card as of 2026-09-15.
- [[pretrainers-guide-training-data]] — toxicity and quality filter trade-offs and domain-removal effects at 1.5B (§5–§6, Figures 5–7); chapter excerpt, no library card as of 2026-09-15.
