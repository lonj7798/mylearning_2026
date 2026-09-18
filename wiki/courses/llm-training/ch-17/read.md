<!-- chapter: ch-17
     track: pretraining
     kind: lab
     title: Lab: Filter and Mixture Ablation with Breadth Measurement
     deps: [ch-14a]
     sources: [[ccnet]], [[fineweb]], [[dclm]], [[paloma]], [[datadecide]], [[signal-and-noise-eval]], [[minhash-lsh]], [[deduplicating-training-data]], [[dolma]], [[c4]], [[weborganizer]], [[data-constrained-scaling]], [[openmathinstruct-2]]
     figures: figures/ablation-grid.html
     revised: 2026-09 (generality revision)
-->

# Chapter 17 — Lab: Filter and Mixture Ablation with Breadth Measurement

> **Core insight.** A filter ablation that reports only an aggregate benchmark average cannot tell a filter that added capability from a filter that narrowed the training distribution toward the benchmark. Both raise the average. [[weborganizer]] measures the difference directly at 1.44B parameters and 28.8B tokens: FineWeb-Edu raises the 9-task average from 51.6 to 54.2 while raising held-out perplexity on the unfiltered base corpus from 12.1 to 14.7, and a mixture tuned only for MMLU lowers HellaSwag from 57.5 to 54.1 and PIQA from 71.3 to 69.9 (Tables 1, 2, 10). This lab therefore requires three measurement families per ablation cell — signal-bearing task scores with seed variance, per-domain held-out perplexity, and distribution shares — plus a statement of whether the ranking is expected to hold at larger scale.
>
> **Guideline.** When an ablation cell is trained at a small scale, report per-domain held-out perplexity next to the task average, because in [[paloma]] a 1B model trained on C4 alone reaches perplexity 391,171 on the RedPajama arXiv domain and 14 on the Dolma peS2o domain (§2, §4.1). When the cell under test is a benchmark-targeted classifier, hold out tasks that were not used to design it, because [[weborganizer]] Table 10 shows a target-specific mixture losing 3.4 points of HellaSwag while gaining on its target. When a small-scale ranking is to be carried to a larger run, state the transfer evidence explicitly, because [[dclm]] reports Pearson r = 0.838 from 400M-1x to 7B-1x over 10 methods (§3.2) and [[datadecide]] reports about 80% pairwise decision accuracy from 150M to 1B over 25 corpora, not 100% (§3.2). Otherwise, report the cell as an observation at that scale and do not generalize it.

---

## Why this chapter matters for a general-purpose model

Chapters [[ch-09]] through [[ch-14a]] state what pretraining data choices do: heuristic pipelines ([[ch-10]]), model-based filters ([[ch-10a]]), deduplication ([[ch-12]]), mixing ([[ch-13]]), multilingual coverage ([[ch-13a]]), repetition and decontamination ([[ch-14]]), and recipe settings ([[ch-14a]]). Every one of those chapters reports results from runs the learner did not perform. This lab is where the measurement apparatus is built, because the failure this course cares about is not visible in any single number that those chapters quote.

The failure has a specific form. A pretraining filter changes which documents enter the corpus. Any such change moves two things at once: the amount of capability the model acquires, and the shape of the distribution it acquires it from. A benchmark average moves with the first and also with the second, in the same direction, whenever the filter's target resembles the benchmark. The [[fineweb]] authors chose the FineWeb-Edu threshold of 3 rather than 4 because higher thresholds traded knowledge and reasoning benchmarks against benchmarks such as HellaSwag (§4); the trade-off was visible to them only because they evaluated on both kinds of task. [[weborganizer]] later measured the shift itself and found that FineWeb-Edu and DCLM-fastText amplify different format categories while both amplify Politics, Health, Science & Tech., History, Knowledge Articles, Tutorials, Academic Writing, and Q&A Forums (§5, Fig. 4).

The lab sits at the end of the pretraining phase, immediately before mid-training ([[ch-32]]). The artifact it produces — an ablation memo with breadth columns and a transfer statement — is the same artifact every later stage needs, because SFT, preference optimization, and RL all make data choices from small proxy runs.

---

## §1 What the lab produces

Four artifacts, in `ch-17/lab-artifacts/`:

1. `pipeline/` — a runnable filter cascade that writes per-document **attributes** rather than filtered text. Keep/drop is a separate pass over the attribute files. This separation follows the Dolma toolkit documentation (github.com/allenai/dolma, `docs/getting-started.md` on main, read 2026-09-14): `dolma tag` runs taggers, `dolma dedupe` writes duplicate-span attributes under `attributes/`, and `dolma mix` builds a dataset from a config over the annotated files. The [[dolma]] paper itself describes two operations rather than a per-document attribute format: "filtering", which unifies language, quality, and content filters, and a Rust "mixing" module that implements up/down-sampling, deduplication, and decontamination with a Bloom filter (§4.1). With attributes kept on disk, a threshold sweep does not re-run language identification or perplexity scoring.
2. `runs/` — one trained model per ablation cell, at least two seeds per cell.
3. `measurements.json` — per cell: task scores, per-domain held-out perplexity, distribution shares, contamination counts, and the seed-to-seed spread of each.
4. `ablation-memo.md` — the written result. Its required contents are in §8.

The companion [`figures/ablation-grid.html`](figures/ablation-grid.html) lets you enter your own model size, tokens per cell, and unique tokens kept and see the resulting FLOPs per cell, the derived H100 hours, the number of epochs each filter setting implies, and the MinHash banding curve for the parameters you pick, next to the published anchor rows from [[dclm]] Table 1 and [[fineweb]] App. E.1.

---

## §2 Choosing the scale, and what a cell at that scale can decide

**Definition.** An *ablation cell* is one complete training run whose data differs from the reference run in exactly one documented way, with every other setting held fixed.

**The measurable problem.** A cell costs compute. Too small a cell produces scores at or near chance, where no filter effect is distinguishable from seed noise; too large a cell means few cells.

**Mechanism.** Fix a scale, then derive the budget.

1. Pick parameters `N` and tokens `D`. Training FLOPs ≈ `6ND` ([[dclm]] §3.2, which uses this definition for its Table 1 column).
2. Multiply by the number of cells and the number of seeds per cell.
3. Convert to hours against a published anchor, and label the conversion as derived.

**Worked example.** [[dclm]]'s smallest competition scale, 400M-1x, is 412M parameters and 8.2B tokens. Check the FLOP column: `6 × 412e6 × 8.2e9 = 2.03e19`, and the table prints 2.0e19. The table also prints 26 H100 hours for that run under the OpenLM framework. An 8-cell grid at two seeds is 16 runs, so `16 × 26 = 416` H100 hours (derived by multiplication from a printed per-run value; the throughput of your own framework will differ).

For the resource-constrained path, the published anchor is [[weborganizer]]'s RegMix proxy models: 50M parameters, 1B tokens per run, "~360 H100 hours for 512 runs" (App. C), which is 0.70 H100 hours per run. Note that `6ND = 3.0e17` FLOPs for that run, which is 1.5% of the 400M-1x FLOPs, while 0.70 / 26 = 2.7% of its hours. A FLOP ratio therefore under-predicts the wall-clock cost of the smaller cell by a factor of 1.8 across these two published runs; the usual explanation is lower hardware utilization at small model sizes (Interpretation, not measured by either source). Sixteen constrained-path runs are about 11 H100 hours.

**Conditions and limits.** Both anchors are single reported values from one framework on one cluster. Neither is a measurement of your code. Record your own measured hours in the memo.

---

## §3 The pipeline

The stage order below is the order used by [[dolma]] for its web subset (§5.5): CCNet output → URL deduplication → document deduplication → quality filters → content filters → paragraph deduplication. Note two facts about the upstream CCNet stage that are often misstated. [[ccnet]] deduplicates *paragraphs before* language identification, and its ablation (§4.1, Fig. 3) is the reason: removing repeated English boilerplate first stops low-resource-language documents from being classified as English or discarded. And [[ccnet]] does not remove documents by language-model perplexity at all; it splits each language into head, middle, and tail terciles and releases all three (§3.4, §5.2). The authors state their reason: text with vocabulary unlike Wikipedia, such as "blog comments with spoken-like text, or very specialized forums with specific jargon", lands in the tail (§5.2). A perplexity filter is a similarity-to-reference filter, and this lab treats it as one.

### §3.1 Language identification

fastText `lid.176` over each document. [[ccnet]] keeps a page in its top language at score > 0.5 (§3.3); [[dolma]] uses English score ≥ 0.5 and reports that this removed 61.7% of bytes (§5.1); [[fineweb]] uses ≥ 0.65 (§3.3). Record which threshold the cell used, and record the language shares before and after — the share table is one of this lab's required outputs (§5.3).

### §3.2 Heuristic quality filters

Two published sets, both reproducible:

- [[c4]]'s rules. [[fineweb]] ablated them one at a time on the 2019-18 crawl (§3.5, Fig. 6): the terminal-punctuation rule gave the largest single gain but removed about 30% of tokens; all other C4 filters together removed about 7% and scored higher than terminal punctuation alone, so FineWeb adopted the others. [[dolma]] made the opposite choice among the C4 rules and kept only the terminal-punctuation rule, "C4 NoPunc", which tagged 22.73% of characters, alongside the full Gopher rule set, which tagged 15.23% (§5.2, App. N.4).
- The three custom filters [[fineweb]] derived from statistics that separate a high- and low-quality version of the same crawl (§3.6): fraction of lines ending in punctuation ≤ 0.12 (10.14% of tokens removed), fraction of characters in duplicated lines ≥ 0.1 (12.47%), fraction of lines shorter than 30 characters ≥ 0.67 (3.73%). Together about 22% of tokens removed, for an aggregate score change of about +1%.

That the same C4 rule set was adopted in opposite ways by two teams is itself the lab's subject: both decisions were supported by an ablation, on different pools, with different evaluation suites.

### §3.3 Exact-substring deduplication

[[deduplicating-training-data]] §4.1: concatenate all examples into one token sequence, build a suffix array, find adjacent suffix-array entries sharing a prefix of at least **50 BPE tokens**, and remove the repeated substring from one of the two examples. The threshold comes from App. B: matches shorter than 10 tokens are common, manual inspection of 25-token matches found no false positives, and the authors doubled 25 to 50 for margin.

Cost, as reported (App. B): on one 96-core, 768 GB machine, the 350 GB C4 suffix array takes under 12 hours wall-clock (about 1,000 CPU-hours) and occupies 1.5 TB; the removal pass then takes under an hour. Scale that down by pool size before planning. Two rules the implementation must respect: removal applies to one occurrence of each repeated span, not both; and by the **cross-split rule** (§5), when text occurs in more than one split the copy in validation or test is kept and the training copy is removed. Deduplication across the split boundary is the intended behaviour, not an error.

### §3.4 Near-duplicate deduplication, and the banding arithmetic

Near-duplicate detection estimates the Jaccard resemblance `r_w(A,B) = |S(A,w) ∩ S(B,w)| / |S(A,w) ∪ S(B,w)|` from small sketches, where `S(D,w)` is the set of w-token shingles of document D and w the shingle size ([[minhash-lsh]] §2). [[minhash-lsh]] Theorem 1 gives the unbiased estimator from one random permutation. The *banding* probability used by modern pipelines is not in that paper; it is derived in Leskovec, Rajaraman and Ullman, *Mining of Massive Datasets*, Ch. 3 §3.4.2, and the card records this ([[minhash-lsh]], Connections).

Write the banding probability as

```
P(candidate | s) = 1 − (1 − s^h)^B
```

where `s` is the Jaccard similarity of the pair, `h` is the number of hashes that must all agree inside one bucket, and `B` is the number of buckets. Two published settings:

| Setting | Hashes total | Buckets `B` | Hashes per bucket `h` | Locus |
|---|---|---|---|---|
| [[fineweb]] | 112 | 14 | 8 | §3.4, App. E.1 |
| [[deduplicating-training-data]] NearDup | 9,000 | 450 | 20 | §4.2, App. A |

**Worked example (check these by hand).** With FineWeb's `h = 8`, `B = 14`: at `s = 0.8`, `s^8 = 0.16777`, `(1 − 0.16777)^14 = 0.0765`, so `P = 0.9235`. At `s = 0.7`, `s^8 = 0.05765` and `P = 0.5645`. At `s = 0.75`, `P = 0.7716`. At `s = 0.85`, `P = 0.9884`. FineWeb prints 56%, 77%, 92%, 98.8% at exactly these four similarities (App. E.1), so the arithmetic reproduces the published table.

Now the same arithmetic for Lee et al.'s `h = 20`, `B = 450`: at `s = 0.8`, `s^20 = 0.01153` and `P = 0.9946`; at `s = 0.7`, `P = 0.3018`; at `s = 0.6`, `P = 0.0163`. FineWeb's setting at `s = 0.6` gives `P = 0.2111`. The 450 × 20 configuration is the sharper of the two around 0.8: it flags 1.6% of 0.6-similar pairs where the 14 × 8 configuration flags 21%. Recall at moderate similarity is what the (B, h) choice sets, and the memo must state which setting the cell used and what it implies for recall of paraphrased near-duplicates.

Panel B of [`figures/ablation-grid.html`](figures/ablation-grid.html) evaluates this formula for any (B, h) and prints the four FineWeb values beside your own, so the reproduction above can be checked and the curve explored before a setting is committed.

[[deduplicating-training-data]] does not stop at the candidate stage: a candidate pair is a duplicate only if Jaccard > 0.8 **and** edit similarity > 0.8, where `EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j) / max(|x_i|, |x_j|)` (§4.2). Duplicate clusters are the connected components of the resulting graph.

**Which dedup step to expect a gain from.** [[dclm]] App. L.2.1 Table 17 runs exactly this ablation at 1B-1x from a 76B-token RefinedWeb-filtered pool: exact only, +1.3 CORE at 13% removal; MinHash only, +0.9 at 18%; suffix array only, +1.9 at 33%; Bloom filter only, +2.1 at 26%; all three of exact + MinHash + suffix array, +2.1 at 41% removal. The authors state that the apparent gap between suffix array and MinHash "falls within the range of variance for the CORE score due to the nondeterminism in subsampling the dataset and training a model". Expect your own single-seed differences of about one point to be indistinguishable from noise for the same reason.

### §3.5 The two quality filters under test

The cell design requires a **broad** filter and a **benchmark-targeted** filter side by side.

- Broad: [[dclm]]'s fastText classifier with positives from OpenHermes-2.5 and high-scoring r/ExplainLikeImFive posts, negatives sampled from a RefinedWeb reproduction, keeping the top 10% of documents. At 7B-1x: CORE 41.0, MMLU 29.2, EXTENDED 21.4, against 35.7 / 27.0 / 19.1 for the same classifier trained with Wikipedia positives (Table 5). Threshold matters: top 15% gives 39.8 CORE and top 20% gives 38.7.
- Benchmark-targeted: a FineWeb-Edu-style classifier ([[fineweb]] §4) — a linear regression head on a frozen Snowflake-arctic-embed-m encoder, trained on Llama-3-70B-Instruct educational ratings on a 0-5 scale, keeping score ≥ 3 (binary F1 82%). Applying it to all 15T FineWeb tokens took 6,000 H100 GPU hours (§4), so budget the classifier pass, not only the training runs.

Both are needed because a classifier trained toward a benchmark's subject matter can raise that benchmark through two different mechanisms: the kept documents teach content the model did not have, or the kept documents move the training distribution closer to the benchmark's distribution without adding content. The first transfers to unrelated tasks and the second does not, which is what §5.4 measures.

---

## §4 The ablation protocol

**Freeze everything except the cell's single change.** Architecture, tokenizer, sequence length, optimizer, learning-rate schedule, initialization seed set, and evaluation harness are fixed across the grid. [[fineweb]]'s protocol (§3.1) is the reference: models identical except data, two runs per data version with different data subsets and seeds, scores averaged.

**Equal tokens per cell, and the repetition confound this creates.** Each cell must see the same number of training tokens, otherwise the comparison also measures budget. A filter that removes 60% of the pool then forces the cell to repeat data. [[data-constrained-scaling]] measured the cost of repetition for GPT-2-architecture models: up to about 4 epochs the held-out loss penalty is negligible (an 8.7B model at 4 epochs ends 0.5% above its 1-epoch loss), value decays with a fitted constant R*_D ≈ 15.4 repetitions, and downstream scores start dropping after about 4 epochs (Abstract, §6, App. A for the fitted constant, App. L and Fig. 6 for the downstream result). The same paper ran the filtered-then-repeated configuration this lab needs (§7, App. N): at 4.2B parameters and 84B tokens, perplexity filtering kept the lowest-perplexity 25% of samples, giving 44B unique tokens repeated about 2 epochs, and deduplication on a 100-character span gave 21B tokens repeated 4 epochs.

The operational rule follows: size the pool so that **every** cell, including the most aggressive filter, stays at or under 4 epochs. If it cannot, the memo must report epochs per cell as a column and state that the aggressive cells are confounded.

**Also report what the filter removed, not only what it kept.** [[fineweb]] §3.4 ran this test on snapshot 2013-48: the ~31B tokens that global deduplication kept trained a *worse* model than 171B tokens obtained by deduplicating the ~460B removed tokens on their own. Global deduplication across 96 snapshots removed up to 90% of old snapshots and left 4T tokens, and a 350B-token run on it improved little over non-deduplicated data; per-snapshot deduplication gave 20T tokens and matched RefinedWeb (§3.4, Fig. 5).

**Cells.** A defensible eight-cell grid:

| Cell | Data | What it isolates |
|---|---|---|
| A | raw extraction, no filter | reference |
| B | A + language ID | language share effect |
| C | B + heuristic filters (C4 rules + FineWeb custom three) | rule-based quality |
| D | C + exact-substring dedup (≥ 50 tokens) | verbatim repetition |
| E | D + MinHash near-dedup | near-duplicate repetition |
| F | E + broad classifier (fastText OH-2.5 + ELI5, top 10%) | broad model-based filter |
| G | E + benchmark-targeted classifier (educational score ≥ 3) | targeted model-based filter |
| H | E + per-domain quota mixture over topic and format labels | mixing at fixed filter |

Cell H exists because [[weborganizer]] found that setting per-domain token quotas *on top of* a quality filter added 2.0 points to FineWeb-Edu's 9-task average (54.2 → 56.2) and 1.0 to DCLM-fastText (55.1 → 56.1) at 1.44B parameters (Table 1), so filtering and mixing are not interchangeable.

---

## §5 Breadth measurement: three families, all required

### §5.1 Per-domain held-out perplexity

Follow [[paloma]]. Report the **macro average** over domains, `|D|⁻¹ Σ_{d∈D} perplexity(d)`, and the worst domain, not a single pooled perplexity: pooled perplexity is a micro average weighted by tokens and hides domains with few tokens (§4.1). Use subsets that span the space a general model must cover — C4-100-domains, RedPajama's 7 domains, M2D2 S2ORC, M2D2 Wikipedia, Dolma-100-programming-languages, Dolma-100-subreddits, WikiText-103, Penn Treebank, Twitter AAE (Table 1).

Two protocol rules from the paper, both of which change the numbers:

- **G5, evaluate documents separately.** Each document is scored on its own after `<BOS>`. With concatenated inputs the variance trend breaks: Pythia 1.4B at 2B tokens on 4M evaluation tokens gives 92.23 ± 17.33 concatenated versus 42.57 ± 0.29 separate (Table 17).
- **G1, decontaminate the training data against the evaluation set** before reading any perplexity; Paloma's own removal rates are Dolma 0.062%, RedPajama 0.099%, The Pile 2.753%, Falcon RefinedWeb 0.733%, C4 0.010%, mC4-en 0.002% (Table 4).

What this family detects: a cell whose task average rose while one or more domains got worse. The extreme published case is the C4-only 1B baseline at perplexity 391,171 on RedPajama arXiv and 14 on Dolma peS2o (§2, §4.1). [[dolma]] found the same pattern at 1.2B on 150B tokens: single-source web corpora (C4, mC4-en, RefinedWeb) gave higher average Paloma perplexity than corpora with curated non-web sources (§9.2, Fig. 5).

Also record perplexity on the **unfiltered** base pool. [[weborganizer]] reports base 12.1, FineWeb-Edu 14.7, DCLM-fastText 14.0, and implicit domain mixtures 12.2-12.9 (Table 2), and reads the gap as the filter's distribution shift rather than domain rebalancing (§5; the reading is the authors' interpretation).

### §5.2 Tasks that carry signal at this scale, with seed variance

Three published constraints on task choice at 400M-1B scale:

1. [[dclm]]'s CORE is a 22-task subset selected to "provide a low-variance signal even at small scales", with each task rescaled so 0 is random guessing (§3.5). Even so, [[weborganizer]] App. E notes that some DCLM CORE tasks are near random at 1b-1x.
2. [[fineweb]] selected its benchmarks for low seed variance, near-monotonic improvement during training, and above-random scores at 1.71B (§3.1).
3. [[datadecide]] measured which tasks decide correctly with small compute: ARC-Easy is predictable with 5 orders of magnitude less compute, HellaSwag, SocialIQA and WinoGrande are insensitive until a compute threshold, and BoolQ exceeds trivial decision accuracy only with intermediate checkpoints of the target runs (§3.1, Fig. 2).

**Metric choice changes the answer more than task choice does.** [[signal-and-noise-eval]] defines noise as relative standard deviation over the final `n` checkpoints of one run, `sqrt((1/(n−1)) Σ_i (m_i − m̄)²) / m̄`, signal as relative dispersion `max_{j,k} |m_j − m_k| / m̄` over a population of similarly-trained models, and SNR as their ratio (§3, Eq. 2). Across 30 benchmarks, switching the score from the primary metric to bits-per-byte (negative log-likelihood of the correct answer divided by its UTF-8 byte count) raised average SNR from 10.0 to 31.5 and decision accuracy from 77.0% to 83.7%; per task, Minerva MATH went 51.0 → 90.0, HellaSwag 74.3 → 95.3, MMLU 89.0 → 92.0 (§5.3, Fig. 6). Averaging the final checkpoints instead of taking the last one added 2.4 points of 30-task decision accuracy (68.9% → 71.3%, Table 1). The same work reports that `n = 5` final checkpoints put the sample standard deviation within ±1σ of the true noise for almost all benchmarks on OLMo 2 7B (App. A.3.2).

**Required reporting.** Per cell and per task: score, bits-per-byte score, and the spread over at least two seeds. A difference smaller than the seed spread is reported as "not resolved at this scale", not as a sign.

### §5.3 Distribution shares

For each cell, report three share tables over the *kept* documents:

1. **Language shares** from the language-identification pass.
2. **Topic and format shares.** [[weborganizer]] released 140M-parameter classifiers (gte-base-en-v1.5) over 24 topic and 24 format categories, distilled from Llama-3.1-405B-Instruct annotations; on validation pages where the 405B annotator is at least 75% confident they reach topic 93.5 average / 87.1 worst-group and format 91.8 / 80.5 accuracy (Table 7). Normalized mutual information between the two taxonomies is about 0.10, so they are close to independent axes (§2.3).
3. **Length and format proxies** already computed as attributes in §3.2 (lines ending in punctuation, duplicated-line fraction, short-line fraction), reported as distributions rather than pass rates.

What this family detects: the mechanism behind a score change. [[fineweb]] measured the topic shift of its own classifier with clusters over 50k + 50k embedded samples: FineWeb-Edu gains "Education, Learning, Teaching" (+3.2%) and "History, Culture, Politics" (+2.2%) and down-samples "Business, Finance, Law", "Entertainment, Film, Theater", and "Places, Travel, Real Estate" (§4.1, Fig. 18).

### §5.4 Transfer of the targeted filter to tasks it was not designed for

Cell G's classifier is trained on educational ratings. Score cell G on tasks that had no role in the classifier's design and report them as a separate column. The published shape of the result: [[weborganizer]] Table 10 shows a Topic × Format mixture tuned for MMLU reaching MMLU 33.2 while HellaSwag falls from 57.5 to 54.1 and PIQA from 71.3 to 69.9, and Table 2 shows that the implicit domain mixture alone recovers 84% of FineWeb-Edu's task-average gain but only 35% of DCLM-fastText's — that is, most of the FineWeb-Edu gain is reproducible by rebalancing domains, and most of the DCLM-fastText gain is not.

---

## §6 Contamination, including paraphrases

Every cell shares one evaluation suite, so contamination is a confound that can invalidate the whole grid at once. Three levels, in increasing cost:

1. **Paragraph or n-gram exact match.** [[dolma]] removes a document if it contains a paragraph longer than 13 tokens that appears in Paloma (App. L, ≤ 0.02% of documents). [[paloma]] G1 uses a Bloom filter over newline-separated paragraphs, ignoring paragraphs under 13 Unicode-segmented tokens or made only of punctuation, spaces and emoji (App. C.1.1). The authors note this rule removes long, frequently quoted documents and advise caution.
2. **Question-plus-option match.** [[dclm]] §4.6 flags pages containing the question text plus at least one answer option, detecting only the last sentence of long MMLU questions to raise recall, and states plainly that this "still incurs many false positives". At 7B-2x, removing the detected overlaps did not lower scores (MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4, Table 7), which is evidence that their gains were not produced by contamination.
3. **Paraphrase-aware.** [[openmathinstruct-2]] §3.1 retrieves the top-5 test questions by embedding similarity (multi-qa-MiniLM-L6-cos-v1), then asks Llama-3.1-405B-Instruct to judge paraphrase equivalence in both orders, 10 calls per question; 569K new questions became 519K after removal. Their App. C.2 also keeps examples that are similar but not equivalent, in Table 11.

Report, per evaluation set: documents removed at level 1, additional documents removed at level 2, and additional documents removed at level 3. Levels 2 and 3 have false positives by the sources' own statements, so report the count, the sampled false-positive share from hand-labelling, and the score change after removal — not the removal alone. [[deduplicating-training-data]] gives the baseline rate to compare against: 4.60% of C4 validation examples and 4.92% of LM1B validation examples have a near-duplicate in training (Table 2). Depth on decontamination methods is in [[ch-14]].

---

## §7 Will the ranking transfer?

Two studies bound the answer, and the memo must cite one of them.

- [[dclm]] §3.2: 10 curation methods, ranked at each of three small scales against 7B-1x. Pearson r = 0.838 (400M-1x), 0.956 (1B-1x), 0.982 (3B-1x). Higher small scales rank better.
- [[datadecide]] §3.2: 25 corpora, 14 sizes, 3 seeds. Ranking at a single small size such as 150M predicts the 1B winner in about 80% of pairwise comparisons, and none of 8 scaling-law baselines exceeds the compute-to-decision-accuracy frontier of single-scale ranking. Decision accuracy is `(1/|P|) Σ_{(A,B)∈P} 1[sign(ŷ_A − ŷ_B) = sign(y_A − y_B)]`, where `P` is the set of corpus pairs, `y` the observed 1B result averaged over 3 seeds, and `ŷ` the prediction (Eq. 3).

**Worked example.** An eight-cell grid has `C(8,2) = 28` distinct cell pairs to rank. At 80% decision accuracy, about 5.6 of those 28 pairwise orderings are expected to be wrong at the target scale. If the memo's recommendation rests on a single pairwise gap, that is the probability the recommendation inverts.

**Conditions and limits, stated by the sources.** [[datadecide]] used one token-to-parameter ratio (100), 14 configurations up to 1B, and multiple-choice cloze tasks only; it also reports that observed scaling trends cross frequently between small and target scales, and that true crossovers are hard to separate from noise (§3.2, §5). [[dclm]]'s correlations are over 10 methods at one target scale. The memo therefore states a range, not a guarantee.

---

## §8 The ablation memo

The memo is the chapter's deliverable. It is a written document, not a table dump, and it is complete when it contains these seven parts.

1. **Setup.** Pool source and size, extractor, tokenizer, model configuration, tokens per cell, seeds per cell, evaluation harness and version, and the measured wall-clock hours per cell.
2. **Cell table.** One row per cell (A-H from §4) with: tokens kept, removal rate for that stage, epochs over the kept pool, task average and its seed spread, bits-per-byte score and its seed spread, macro per-domain perplexity, worst-domain perplexity, and perplexity on the unfiltered base pool.
3. **Share tables.** Language, topic, and format shares per cell, with the change from cell A.
4. **Held-out transfer column.** For cell G, scores on tasks that had no role in the classifier's design, reported separately from the tasks that did.
5. **Contamination report.** Counts removed at each of the three levels in §6, the hand-labelled false-positive share of levels 2 and 3, and the score change after removal.
6. **Transfer statement.** Which published result is being relied on ([[dclm]] §3.2 or [[datadecide]] §3.2), the expected number of mis-ordered pairs for the grid size used, and the resulting confidence in the recommendation. A memo that recommends a cell without this paragraph is incomplete.
7. **Cells that narrowed coverage.** Every cell whose task average rose while its macro per-domain perplexity got worse, or whose share tables moved a category by more than the seed spread, named explicitly. If no cell did, state that and give the resolution of the measurement — a null result at a 0.5-point resolution is a different claim from a null result at 0.05.

Two prohibitions that follow from the evidence in this chapter. A difference smaller than the seed spread is written as "not resolved", never as a sign ([[dclm]] App. L.2.1). A recommendation for a larger run is written with its transfer evidence and its error rate attached, never as a bare ranking ([[datadecide]] §3.2).

---

## Negative samples and negative feedback

This lab uses negatives in sense (1) of the four in §6.1 of the course standard: **negative marginal value** — documents removed because including them as training targets is believed to lower performance. No gradient is applied to removed documents; nothing is pushed down. Senses (2), (3), and (4) do not occur at this stage.

- **Where the negatives come from.** A threshold on a heuristic attribute, a classifier score, or membership in a duplicate cluster. Labels are not human-verified. [[fineweb]]'s classifier is the only case here with a reported agreement figure: binary F1 82% at threshold 3 against the held-out Llama-3-70B-Instruct annotations (§4). The paper does not report an error rate for documents near the threshold, so the false-negative rate of this filter is not reported.
- **What practice does with them.** Discards them. That is why the *kept-versus-removed* test in §4 is required: [[fineweb]] §3.4 found the removed half of snapshot 2013-48 trained a better model than the kept half, which is a direct measurement of a filter assigning negative marginal value incorrectly.
- **Evidence on size of effect.** [[dclm]] Table 17 gives +0.3 to +2.1 CORE for deduplication settings removing 13% to 41% of tokens at 1B-1x, with the authors' own note that differences of about one point fall within run-to-run variance. [[data-constrained-scaling]] §7 reports that perplexity filtering helps on C4 while deduplication does not improve downstream results on C4, and that both are more effective on the noisier OSCAR corpus — the value of discarding depends on how noisy the pool is.
- **Controls.** Train one cell on the discarded set alone; report per-domain perplexity of the kept-set model on domains over-represented in the discarded set; keep the attribute files so the threshold can be moved without re-running the pipeline.
- **Diagnostics.** Removal rate per stage, per-domain removal rate, language-share change, and the score of the discarded-set model.
- **Effect on generality.** [[ccnet]] §5.2 names the concrete risk: perplexity-based removal sends spoken-like text and specialized-jargon forums to the tail, and the authors chose to release the buckets rather than delete the tail for that reason.

---

## Recipe

Lab settings with the published rows they are derived from. `Lab value` is what this chapter prescribes; `Reason for difference` states why it departs from the source.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| DCLM 400M-1x reference run | 412M | pretrain-stable | train tokens; train FLOPs; H100 hours | 8.2B; 2.0e19; 26 | arXiv:2406.11794v4 §3.2 Table 1 | verified 2026-09-17 | no ablation reported; scale set by 20 × N × multiplier | same (full path): 412M / 8.2B per cell | none; adopted unchanged as the full-budget cell |
| WebOrganizer RegMix proxy models | 50M | pretrain (proxy) | tokens per run; compute | 1B; ~360 H100 hours for 512 runs (0.70 h/run) | arXiv:2502.10341v3 §3, App. C | verified 2026-09-17 | Table 8: Spearman ≈ 0.90 on 50 held-out mixtures | same (constrained path): 50M / 1B per cell | none; adopted unchanged as the constrained cell |
| WebOrganizer RegMix proxy models | 50M | pretrain (proxy) | architecture | hidden 512, intermediate 1536, SwiGLU, 8 heads, 8 blocks, RoPE base 10000 | arXiv:2502.10341v3 App. C Table 6 | verified 2026-09-17 | no ablation reported | same | none |
| WebOrganizer RegMix proxy models | 50M | pretrain (proxy) | optimizer; peak LR; cooldown; warmup; batch | Adam β (0.9, 0.95); 3e-3; cosine to 3e-4; 10%; 128 | arXiv:2502.10341v3 App. C Table 6 | verified 2026-09-17 | no ablation reported | same, frozen across all cells | none; freezing is the ablation requirement |
| FineWeb data-ablation model | 1.71B | eval-gate | seeds per data version | 2 (different data subset and initialization), averaged | arXiv:2406.17557v2 §3.1 | verified 2026-09-17 | no ablation reported | 2 seeds per cell minimum | none |
| FineWeb data-ablation model | 1.71B | pretrain-stable | tokens per ablation run | ~28B (filtering ablations) | arXiv:2406.17557v2 §3.1 | verified 2026-09-17 | described as roughly Chinchilla-optimal for 1.71B | 8.2B (full) / 1B (constrained) | smaller model; token budget follows the cell's own Chinchilla-scale anchor |
| FineWeb MinHash | — | data dedup | hashes; buckets; hashes per bucket; shingle | 112; 14; 8; 5-grams | arXiv:2406.17557v2 §3.4, App. E.1 | verified 2026-09-17 | App. E.1 prints P = 56/77/92/98.8% at s = 0.70/0.75/0.80/0.85 | same | none |
| Lee et al. NearDup | — | data dedup | hashes; buckets; hashes per bucket; thresholds | 9,000; 450; 20; Jaccard > 0.8 and edit similarity > 0.8 | arXiv:2107.06499v2 §4.2, App. A | verified 2026-09-17 | 0.8 and 0.9 similarity histograms vanish at the same point (App. A, Fig. 4) | offered as the second preset in the figure | 9,000 hashes per document is costly at lab pool sizes |
| Lee et al. ExactSubstr | — | data dedup | minimum match length | 50 BPE tokens | arXiv:2107.06499v2 §4.1.2, App. B | verified 2026-09-17 | no false positives at 25 tokens on manual inspection; doubled for margin | same | none |
| DCLM fastText filter | — | data filter | positives; threshold | OpenHermes-2.5 + ELI5 posts, negatives sampled from a RefinedWeb reproduction; top 10% | arXiv:2406.11794v4 §4.4, Table 5 | verified 2026-09-17 | Table 5 at 7B-1x: top 10% CORE 41.0 vs 39.8 (15%) and 38.7 (20%) | same, as cell F | none |
| FineWeb-Edu classifier | not reported | data filter | score threshold | ≥ 3 | arXiv:2406.17557v2 §4, App. F.2 | verified 2026-09-17 | FW-Edu-2/3/4 at 28B tokens; 3 best on aggregate, with a HellaSwag trade-off | same, as cell G | none |
| Dolma web pipeline | — | data pipeline | stage order | CCNet output → URL dedup → document dedup → quality → content → paragraph dedup | arXiv:2402.00159v2 §5.5 | verified 2026-09-17 | filter order itself is not ablated in the paper | same order | none; note the source does not ablate the order |
| Paloma evaluation | 1B baselines | eval-gate | evaluation format | each document scored separately after `<BOS>` | arXiv:2312.10523v2 §3 G5, App. H | verified 2026-09-17 | Table 17: 92.23 ± 17.33 concatenated vs 42.57 ± 0.29 separate | same | none |
| Signal-and-Noise checkpoint rule | 60M-750M | eval-gate | noise window; metric | final 5 checkpoints averaged; bits-per-byte | arXiv:2508.13144v1 §4.1, §5.3 | verified 2026-09-17 | Table 1: +2.4 points decision accuracy; Fig. 6: 77.0% → 83.7% | same | none |

**Starting point for a small general-purpose run.** Constrained path: 50M parameters (hidden 512, intermediate 1536, 8 heads, 8 blocks, SwiGLU, RoPE base 10000), 1B tokens per cell, Adam β (0.9, 0.95), peak LR 3e-3 with cosine cooldown to 3e-4 and 10% warmup, batch 128 — every value from [[weborganizer]] App. C Table 6, where it was used for 50M proxy models trained on 1B tokens each on H100s. Two seeds per cell, scored as bits-per-byte on the final 5 checkpoints averaged ([[signal-and-noise-eval]] §4.1, §5.3). Pool sized so no cell exceeds 4 epochs ([[data-constrained-scaling]] §6).

---

## Generalization lens

**(a) What increases breadth.**
- Curated non-web sources alongside filtered web text: at 1.2B on 150B tokens, single-source web corpora gave higher average Paloma perplexity than Dolma or The Pile ([[dolma]] §9.2, Fig. 5).
- Per-domain quotas on top of a quality filter: +2.0 points of 9-task average on FineWeb-Edu and +1.0 on DCLM-fastText at 1.44B ([[weborganizer]] Table 1).
- Deduplication at moderate removal rates: +0.3 to +2.1 CORE across the eight deduplication settings, which remove 13-41% of tokens, at 1B-1x ([[dclm]] Table 17), and about a 10× reduction in verbatim memorization with no worse perplexity ([[deduplicating-training-data]] Table 4: 1.926% → 0.189% NearDup, 0.138% ExactSubstr).

**(b) What causes narrowing.**
- A benchmark-targeted mixture: MMLU-targeted selection lowered HellaSwag 57.5 → 54.1 and PIQA 71.3 → 69.9 ([[weborganizer]] Table 10).
- Similarity-to-reference perplexity filtering: spoken-like text and specialized jargon fall into the discarded tail ([[ccnet]] §5.2).
- Aggressive global deduplication: kept data was worse than removed data on snapshot 2013-48 ([[fineweb]] §3.4).
- A threshold set to maximize one aggregate: FineWeb chose threshold 3 rather than 4 because of the trade-off against HellaSwag-type benchmarks ([[fineweb]] §4).

**(c) How to measure it at this stage.**
- Macro-averaged per-domain perplexity plus the worst domain, documents scored separately, on a decontaminated corpus ([[paloma]] §3, §4.1).
- Held-out perplexity on the *unfiltered* base pool, as a distribution-shift readout ([[weborganizer]] §5).
- Signal-bearing tasks scored as bits-per-byte with seed spread, and any gap below the seed spread reported as unresolved ([[signal-and-noise-eval]] §5.3; [[dclm]] App. L.2.1).
- Topic, format, and language shares of the kept set ([[weborganizer]] §2.2; [[fineweb]] §4.1).
- Transfer statement with its source and its error rate ([[dclm]] §3.2; [[datadecide]] §3.2).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Unequal tokens per cell | Aggressive-filter cells look better and also ran more optimizer steps | Log tokens seen and steps per cell; assert equality before scoring |
| Repetition confound hidden inside equal tokens | Aggressive cells exceed 4 epochs over their own kept pool | Compute `tokens_per_cell / unique_tokens_kept` per cell and report it as a column ([[data-constrained-scaling]] §6) |
| Single-seed sign reading | Reported deltas of about 1 point flip when a seed is changed | Two seeds minimum; report spread; call sub-spread gaps unresolved ([[dclm]] App. L.2.1) |
| Benchmarks at chance | Cell scores cluster near the random baseline and do not move with training | Use the rescaled-to-random CORE form and drop tasks that stay at chance ([[dclm]] §3.5; [[weborganizer]] App. E) |
| Aggregate-only reporting | Task average rises, and nothing indicates which domains got worse | Macro per-domain perplexity plus worst-domain column ([[paloma]] §4.1) |
| Targeted classifier evaluated on its own target | Large gain on the target task only | Hold out tasks not used in classifier design; report them separately ([[weborganizer]] Table 10) |
| Concatenated perplexity evaluation | Perplexity numbers vary by tens between runs of the same checkpoint | Score each document separately after `<BOS>` ([[paloma]] App. H, Table 17) |
| Contamination assumed absent | Scores rise on exactly the tasks whose text appears in the pool | Run levels 1-3 of §6 and report the post-removal score change, as in [[dclm]] Table 7 |
| Banding parameters reported as one number | "MinHash with 112 hashes" without buckets | Report (total hashes, buckets `B`, hashes per bucket `h`) and the `P(s)` at s = 0.6/0.7/0.8 |
| Filter order assumed validated | Stage order copied from a paper and cited as evidence | [[dolma]] states the filter order itself was not ablated (§5.5); cite it as a choice, not a result |

---

## Check your understanding

1. Two cells differ by one filter. Cell F scores 1.1 points higher on the task average and 0.8 higher in macro per-domain perplexity (worse). Explain which of the two numbers can be caused by the filter adding capability, which by distribution shift, and what third measurement separates them.
2. A cell removes 70% of the pool and is then trained to the same token budget as the reference cell. Derive the epoch count and explain, with the [[data-constrained-scaling]] result, why the comparison is no longer a filter comparison.
3. Using `P = 1 − (1 − s^h)^B`, explain why the (B = 450, h = 20) setting flags 1.6% of pairs at s = 0.6 while (B = 14, h = 8) flags 21%, and state which of the two is the safer default when the pool contains machine-paraphrased spam.
4. [[fineweb]] found the *removed* half of snapshot 2013-48 trained a better model than the kept half. Explain what this implies about interpreting a positive ablation delta for any removal-based filter, and design the one extra cell that tests it.
5. [[datadecide]] reports about 80% pairwise decision accuracy from 150M to 1B. For an eight-cell grid, compute the expected number of incorrectly ordered pairs at the target scale and explain what this means for a memo that recommends a single cell.
6. [[signal-and-noise-eval]] raised 30-task decision accuracy from 77.0% to 83.7% by changing the metric to bits-per-byte without changing any model. Explain the causal path from metric to decision accuracy using the definitions of signal and noise.
7. [[dclm]] found that removing detected MMLU and HellaSwag overlaps did not lower scores at 7B-2x. Explain why this is evidence about contamination and also why it is not proof that the pool was uncontaminated.
8. Cell G's educational classifier raises MMLU and lowers HellaSwag. Argue both interpretations — that the filter added knowledge, and that it narrowed the distribution — and state the measurement that decides between them.

---

## Connections

- **Previous chapter — [[ch-14a]]: Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures.** Supplies the frozen training configuration that every cell in this lab shares.
- **Dependency — [[ch-14a]]** is the only declared dependency; the pipeline stages come from [[ch-10]] (Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb), [[ch-10a]] (Model-Based Quality Filtering and Benchmark-Targeted Data Selection), [[ch-12]] (Deduplication: Exact, Near-Duplicate, and Semantic), [[ch-13]] (Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale), and [[ch-14]] (Data-Constrained Scaling, Repetition, and Pretraining Decontamination).
- **Next chapter — [[ch-32]]: Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL.** The measurement discipline built here is applied to the annealing mixture, where the same narrowing risk appears with a much smaller token budget.

---

## Sources

- [[dclm]] — chapter excerpt of DataComp-LM (arXiv:2406.11794 v4): competition scales and the 6ND FLOP column, the CORE 22-task definition, cross-scale rank correlations, the filter and threshold tables, the deduplication ablation table, and the decontamination result.
- [[fineweb]] — ablation protocol at 1.71B, C4 and custom filter removal rates, the MinHash banding parameters with their printed `P(s)` values, the per-snapshot versus global deduplication result, and the FineWeb-Edu classifier, threshold, and topic shift.
- [[paloma]] — per-domain perplexity as the breadth metric: macro averaging, document-separate evaluation, decontamination rates, and the C4-baseline domain spikes.
- [[datadecide]] — decision accuracy of single-scale rankings from 150M to 1B over 25 corpora, per-task predictability, and the stated limits on transfer.
- [[signal-and-noise-eval]] — signal, noise, and SNR definitions; the bits-per-byte and checkpoint-averaging interventions; the choice of `n` final checkpoints.
- [[ccnet]] — dedup-before-language-ID ordering, the language-score threshold, and the reason a perplexity tercile is a similarity filter rather than a quality filter.
- [[deduplicating-training-data]] — ExactSubstr's 50-token threshold and cost, NearDup's parameters and dual thresholds, the cross-split rule, and the overlap and memorization numbers.
- [[minhash-lsh]] — the resemblance estimator, and the record that the banding formula comes from *Mining of Massive Datasets* Ch. 3 rather than from Broder 1997.
- [[dolma]] — web pipeline stage order, the attributes-then-mix toolkit design, Gopher and C4-NoPunc filter rates, the decontamination rule, and the multi-source domain-fit result.
- [[c4]] — the heuristic rule set ablated in cell C.
- [[weborganizer]] — topic and format classifiers and their accuracy, quality filters as implicit domain mixtures, held-out perplexity as a distribution-shift readout, per-target transfer losses, and the 50M proxy-run recipe used for the constrained path.
- [[data-constrained-scaling]] — the 4-epoch repetition bound, the filtered-then-repeated configurations, and the finding that filter value depends on how noisy the pool is.
- [[openmathinstruct-2]] — the embedding-retrieval plus LLM-judge paraphrase decontamination procedure used at level 3.
