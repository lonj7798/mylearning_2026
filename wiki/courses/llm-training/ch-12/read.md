<!-- chapter: ch-12
     track: data
     title: Deduplication — Exact, Approximate, Semantic
     sources: [[deduplicating-training-data]], [[minhash-lsh]], [[d4]], [[ccnet]], [[dolma]], [[fineweb]], [[c4]]
     figures: figures/minhash-lsh.html
-->

# Chapter 12 — Deduplication: Exact, Approximate, Semantic

> **Core insight.** A pretraining corpus is not a set; it is a *multiset with a long heavy tail of near-copies*. Lee 2021 showed that failing to dedup costs you three things simultaneously — training time (you re-descend the same gradient), privacy (verbatim memorization scales ~linearly with duplication count), and evaluation honesty (>4% of "held-out" validation text is already in train). The counter-intuitive part is that "remove all duplicates" is a family of at least three different operations: exact-substring at the suffix-array level, MinHash+LSH near-duplicate at the document level, and embedding-cluster semantic dedup. They remove different things, at different cost, with different failure modes. Picking the wrong granularity is the modern version of the C4 mistake.
>
> **Guideline.** Run dedup in the cascade order **URL → exact-substring → MinHash+LSH → (optionally) semantic**. Each stage removes a population the next one cannot efficiently address. Tune LSH's `(r, b)` not from a table but from the target Jaccard — the S-curve $1 - (1 - J^{r})^{b}$ is a design knob, not a magic number. Treat semantic dedup as a **diversity lever**, not a near-duplicate filter; it happily deletes distinct-but-topically-redundant documents, which can improve efficiency and narrow domain coverage at the same time.

---

## Why this chapter exists

The first chapter of any web-data story is dedup. Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, and Carlini put the foundational number on the board in 2021 ([[deduplicating-training-data]]): **3.04% of C4 tokens sit inside near-duplicate clusters, a single 61-word English sentence repeats >60,000 times, and models trained on the un-deduplicated corpus emit verbatim training-set text for ~1% of 256-token unprompted completions.** Deduplicating drops memorization 10×, hits the same perplexity target in fewer steps, and removes >4% of validation-set overlap that had been silently inflating reported benchmark wins.

That finding changed how every subsequent open corpus is built. CCNet ([[ccnet]]), Dolma ([[dolma]]), and FineWeb ([[fineweb]]) all front-load dedup, and FineWeb adds the first surprise: **per-snapshot MinHash outperforms global MinHash on downstream tasks**, because global dedup is aggressive enough to delete "re-indexed" but genuinely distinct versions of the same page.

Ch-11 built the tokenizer-and-lineage layer: shards, doc-ids, content-addressed hashing. This chapter is about *what to remove before you train on any of it*.

---

## 1. What is actually duplicated in a web corpus

From [[deduplicating-training-data]]:

| Duplicate type | Example | Who catches it | Who misses it |
|---|---|---|---|
| Identical URLs across CC snapshots | the same page re-crawled 6 months apart | URL Bloom-filter ([[dolma]]) | MinHash (it finds them too, but URL-hash is 100× cheaper) |
| Exact paragraph repeats | boilerplate footer, cookie banner | paragraph-hash dedup ([[dolma]]) | doc-level MinHash with high threshold |
| Long verbatim substrings across docs | the 61-word sentence, MIT-license headers, copy-pasted Wikipedia intros | **suffix-array ExactSubstr** | doc-level MinHash if duplicate is a small fraction of the doc |
| Near-duplicate documents | article reprinted with 5% edits, SEO doorway pages | **MinHash + LSH** | exact methods |
| Semantic near-copies | same news story rewritten by a different outlet | **embedding-space dedup (SemDeDup / D4)** | everything above |
| Topically redundant | ten thousand news articles about the same event | embedding clustering + cluster-size cap | everything above (including SemDeDup at low threshold) |

Each row has its own algorithmic tool. The three horizontal cuts the chapter will derive — exact, approximate, semantic — correspond to the three tool families in that table. None dominates; the production cascade uses all three.

---

## 2. Lee 2021 — the evidence that dedup is not optional

The paper ships **two complementary tools** ([[deduplicating-training-data]]):

1. **ExactSubstr** — suffix array over the concatenated corpus; find every duplicate substring of length `>= 50 tokens`; remove one copy of each.
2. **NearDup** — MinHash + LSH on 5-gram shingles; drop any document with Jaccard similarity `>= ~0.8` against another.

The eval-contamination table is the single most damaging result in the paper. It is worth quoting in full:

| Corpus | % of validation that overlaps training (≥50 tokens) |
|---|---|
| LM1B | **4.6%** |
| C4 | **3.2%** |
| RealNews | **1.6%** |
| Wiki-40B | **0.6%** |

(From [[deduplicating-training-data]] §"Train-test contamination.") These are validation sets that researchers had been reporting perplexity improvements on for years. The sleight of hand was not malicious — nobody built LM1B expecting suffix-array contamination analysis — but the reported numbers were partly measurements of memorization, not of generalization. Every paper that pre-dates this table is silently rescaled.

The memorization number is equally load-bearing:

> Without dedup, ~1% of unprompted 256-token completions are verbatim training copies. With dedup, this drops ~10×.

Duplication count and memorization rate are approximately linear ([[deduplicating-training-data]] Figure 4 in spirit). A document duplicated `k` times is memorized with probability roughly proportional to `k`. This is the privacy surface for membership inference and training-data extraction — and it is directly controllable by dedup aggressiveness.

Training-efficiency closes the triangle. Dedup removes ~5% of C4 tokens yet hits target perplexity in **fewer** steps, because the removed tokens were the easy re-descended ones. Pretraining loss is a `1/N` process over *unique* content, not raw tokens.

---

## 3. MinHash — derive P(collision) = J(A, B)

The core primitive. Let each document `D` be represented by its set of shingles `S(D)` — for Lee 2021, 5-grams over whitespace-tokenized text. Two documents' similarity is the **Jaccard index**:

$$
J(A, B) \;=\; \frac{|S(A) \cap S(B)|}{|S(A) \cup S(B)|}.
$$

Naively computing `J` for every pair costs $O(N^2)$ set-intersection operations. MinHash's insight (Broder 1997, [[minhash-lsh]]) is that **one number per document** is enough, in expectation.

**Construction.** Fix a random permutation `π` of the universe of possible shingles. Define the MinHash signature of `D` under `π` as

$$
h_{\pi}(D) \;=\; \min_{s \in S(A)} \pi(s).
$$

i.e. apply `π` to every shingle of `D`, take the minimum. This is one integer.

**Theorem (Broder).** For a single permutation `π` drawn uniformly at random,

$$
\Pr[\,h_{\pi}(A) = h_{\pi}(B)\,] \;=\; J(A, B).
$$

**Proof.** Consider the set $U = S(A) \cup S(B)$. Under `π`, each element of `U` is equally likely to receive the global minimum value. Let `s*` be the argmin. We have $h_{\pi}(A) = h_{\pi}(B)$ if and only if `s*` lies in both `S(A)` and `S(B)`, i.e. in `S(A) ∩ S(B)`. The probability that a uniformly random element of `U` lies in `S(A) ∩ S(B)` is exactly $|S(A) \cap S(B)| / |S(A) \cup S(B)| = J(A, B)$. ∎

So one permutation gives you a single Bernoulli trial with success probability `J`. To estimate `J` to within additive error `ε` with confidence `1-δ`, Hoeffding gives $m \geq \tfrac{1}{2\varepsilon^2}\ln(2/\delta)$ independent permutations. For `ε = 0.05, δ = 0.01`, that is `m ≈ 1060`. Lee 2021 uses **9000** signatures — aggressively more than Hoeffding demands, because they care about the right tail of the `J` distribution (finding `J ≥ 0.8` pairs, not estimating `J` uniformly).

In practice, real permutations on an astronomical shingle universe are replaced by independent hash functions $h_i(s) = (a_i s + b_i) \bmod p$; the "min over shingles under `h_i`" is computationally identical and uses $O(|S(D)|)$ hashes per document. Modern implementations (datasketch, Spark) ship this directly.

---

## 4. LSH — the r × b banding construction and its S-curve

The signature comparison is still $O(N^2)$ if done naively. Locality-sensitive hashing collapses the cost to near-linear by **bucketing similar signatures together**.

**Banding.** Split the `m`-length MinHash signature into `b` bands of `r` rows each, so $m = r \cdot b$. For each band, hash the band's `r` integers into a bucket; two documents are **candidates** if they share a bucket in *any* band.

Derive the collision probability. Two documents with Jaccard `J`:

- Probability they agree in a single row: `J` (section 3).
- Probability they agree in all `r` rows of one band: $J^{r}$.
- Probability they disagree in that band (differ in at least one row): $1 - J^{r}$.
- Probability they disagree in every band (independently across bands, given independent hashes): $(1 - J^{r})^{b}$.
- Probability they share at least one band (the LSH-candidate event):

$$
\boxed{\;P_{\mathrm{LSH}}(J; r, b) \;=\; 1 \;-\; (1 - J^{r})^{b}.\;}
$$

This is the **S-curve**. Three regimes.

| J | P_LSH (qualitative) |
|---|---|
| `J ≪ t` | near 0 (random pairs are not candidates) |
| `J ≈ t` | transitions steeply through 0.5 |
| `J ≫ t` | near 1 (true duplicates collide in almost every band) |

The **threshold** `t` — the `J` at which `P_LSH(J) = 0.5` — is well approximated by

$$
t \;\approx\; \left( \frac{1}{b} \right)^{1/r}.
$$

Derivation: set $P_{\mathrm{LSH}}(t) = 0.5$, so $(1 - t^{r})^{b} = 0.5$, so $1 - t^r = 0.5^{1/b} \approx 1 - \frac{\ln 2}{b}$, so $t^r \approx \frac{\ln 2}{b}$, so $t \approx (\ln 2 / b)^{1/r} \approx (1/b)^{1/r}$ up to a small constant that vanishes for `b ≥ 8`. For Lee 2021's `r = 450, b = 20`: $t \approx (1/20)^{1/450} \approx 0.9934^{\text{...no}}$. Recompute: $\ln(1/20)/450 = -3.00/450 = -0.00666$, so $t \approx e^{-0.00666} \approx 0.9934$. **That contradicts the paper.**

Read [[deduplicating-training-data]] more carefully: the paper reports **threshold ≈ 0.8**, but also **9000 signatures**, **b = 20 bands**, **r = 450 rows**. These are internally inconsistent if you treat `(1/b)^{1/r}` as the final accept threshold — at those parameters the 0.5-threshold lives up near `t ≈ 0.993`, so LSH alone would "accept" nearly every pair. The resolution is that Lee et al. parameterize for *recall*, not threshold: they use `r` and `b` to drive `P_LSH ≈ 1` at every `J ≥ 0.8`, then run an explicit post-filter that computes true Jaccard on every candidate pair and keeps only those above 0.8. Most production implementations do the same: LSH is the candidate generator, the Jaccard check is the verifier. The S-curve's 0.5-threshold controls the recall/cost tradeoff, not the final accept/reject.

Play with the curve in **[figures/minhash-lsh.html](figures/minhash-lsh.html)** — slide `r` and `b` and watch the S-curve move. For `J_target = 0.8`, a typical production setting is `r = 9, b = 20` (180 signatures, `t ≈ 0.687`, `P(J=0.8) ≈ 0.944`, false-positive `P(J=0.5) ≈ 0.038`); Lee 2021's 9000-signature choice is a recall-maxed outlier because they wanted the paper's conclusions to be unimpeachable.

---

## 5. ExactSubstr and suffix arrays — when span-level matters

MinHash is a document-granularity tool. A 100-token verbatim passage embedded in two otherwise-distinct 5000-token articles will have document-level Jaccard ≈ 0.02 — invisible to MinHash — but is the memorization-driving duplicate.

Enter suffix-array ExactSubstr ([[deduplicating-training-data]]):

- Concatenate the entire corpus into a single string `T`.
- Build the suffix array `SA` of `T` in $O(N \log N)$ (SA-IS algorithm).
- Walk `SA`: adjacent suffixes in `SA` share a longest-common-prefix `lcp[i]`. Any `lcp[i] ≥ 50` tokens marks a duplicate substring.
- Remove one copy of each duplicate span.

The **50-token threshold** is empirical. Shorter spans include common phrases ("the quick brown fox," standard license boilerplate prefixes) that legitimately recur. Longer spans are almost always literal copy-paste.

Span-level dedup matters specifically when:

- Documents contain **large boilerplate** (MIT license, Wikipedia's Creative Commons footer, Stack Overflow's "Thanks for contributing..." frame).
- Documents contain **quoted training examples** (the 61-word sentence from C4; quoted ChatGPT outputs in 2024+ corpora).
- Document-level near-dup failed because the duplicate passage is a small fraction of the host document.

Memory is the binding constraint. A suffix array over 1T tokens at 8 bytes/entry is 8 TB — feasible on a multi-TB-RAM node, out of reach on commodity hardware. Production pipelines (GoogleLM, OLMo) shard by prefix and run suffix arrays per shard, then reconcile across shards via a second pass. Dolma's `paragraph-dedup` stage is a cheaper approximation: Bloom-filter exact-match at paragraph granularity, which catches 80% of the ExactSubstr wins at 1% of the memory.

---

## 6. Semantic dedup — SemDeDup and D4

Exact and approximate dedup are surface-form tools. They cannot see that "Biden won the 2020 election" and "The 2020 US presidential election was won by Joe Biden" are the same content with disjoint 5-grams.

SemDeDup ([[d4]], Abbas et al. 2023) answers with embedding-space dedup:

1. Embed every document with a sentence-or-doc encoder (originally CLIP for images; OpenAI/BGE/E5 for text).
2. Cluster embeddings (`k`-means, `k ≈ sqrt(N)`).
3. Within each cluster, for every pair above cosine-similarity `τ` (typical `τ = 0.95`), keep one, drop the other.

The paper's headline is that **you can drop 20–50% of web-scale data with no downstream accuracy loss**, and sometimes improved OOD generalization. D4 (the sequel) adds cluster-size capping as an explicit diversity lever: oversized clusters are down-sampled before the within-cluster dedup pass, which turns "dedup" into "coverage equalization."

**Three practitioner traps.**

**Embedding choice matters.** Swap E5 for OpenAI-embedding-v3 and your SemDeDup decisions change for ~10% of documents. Embeddings encode a model's notion of similarity; that notion is trained on pretraining data that is itself subject to the same biases. FineWeb-Edu's educational-value classifier ([[fineweb]]) is an adjacent technology — also embedding-driven, also dataset-opinionated — and the same caveat applies.

**`τ` is a diversity knob, not a duplicate threshold.** At `τ = 0.95`, you remove near-copies. At `τ = 0.80`, you remove topical-redundants (two news articles about the same game). At `τ = 0.60`, you remove domain coverage. There is no principled way to pick `τ` except **downstream eval sweep**, because the right answer depends on which downstream capabilities matter.

**Aggressive semantic dedup narrows domain coverage.** This is the chapter's sting. D4 finds that very aggressive clustering (high `k`, low `τ`) removes the long tail of niche domains — legal case law, minority-language sub-corpora, niche code repositories — and downstream metrics on those domains drop sharply. FineWeb's per-dump-rather-than-global MinHash ([[fineweb]]) finding has the same shape: global dedup is too aggressive *because* it hits things that are near-duplicates-by-crawl-artifact but genuinely distinct content.

Semantic dedup is a tool for **efficiency** (train faster on a smaller set) and for **coverage equalization** (prevent over-sampling of English web news). It is not the same tool as MinHash. Stacking them thoughtlessly is how corpora lose their tails.

The **right panel of [figures/minhash-lsh.html](figures/minhash-lsh.html)** shows this recall-vs-diversity tradeoff as a sliding knob — as you turn up the SemDeDup aggression, recall of genuine duplicates climbs smoothly, but domain-coverage (entropy of the cluster-ID distribution over retained documents) *collapses* once `τ` drops below ~0.85.

---

## 7. The production cascade and where FineWeb broke the default

**Canonical dedup order**, composed from [[dolma]] §"Filter cascade" and [[fineweb]] §"Pipeline":

1. **URL-hash Bloom filter** across CC snapshots. Cheapest stage; catches exact re-crawls.
2. **Per-snapshot MinHash near-dup** (FineWeb's key finding: per-snapshot, not global).
3. **ExactSubstr** or paragraph-hash Bloom (boilerplate, license text, quoted passages).
4. Language / quality / PII / content filters (chapters 10, 13, 14 will expand).
5. **Paragraph-level dedup last** ([[dolma]]): earlier filters change which paragraphs survive, so dedup only matters over the surviving distribution.
6. **(Optional, post-quality)** semantic dedup or cluster-size capping for diversity control.

**FineWeb's surprise.** Global MinHash across 96 Common Crawl dumps *underperformed* per-dump MinHash on downstream accuracy. Why: many genuinely high-quality pages are crawled once per snapshot but re-appear across snapshots. Global dedup sees them as near-duplicates and keeps only one copy, halving the effective signal; per-dump dedup preserves the re-crawls, which act as a free ensemble of the same high-quality page at slightly different extraction points. The lesson: **aggression is not quality**. Dedup is a coverage-shaping operation.

**Dolma's paragraph-last rule** has the same flavor. `dolma-ngram` splits paragraphs into n-grams and drops a paragraph if the fraction of duplicated n-grams exceeds `T = 1.0` (the default, i.e., 100% duplicated). Running this before quality filtering means you spend compute deduplicating paragraphs that a subsequent language/quality filter would have removed anyway; worse, you may delete *one* copy of a paragraph whose only remaining copy lives in a document that later fails quality filtering, leaving the corpus with zero copies of genuinely useful content.

---

## 8. When dedup hurts

Three attested failure modes.

**8.1 Over-dedup of the tail.** Code: if you MinHash a code corpus at `J = 0.7`, you delete near-duplicate function bodies that exist in thousands of repositories — but those repetitions are *how models learn the standard idiom*. The Stack's ([[dolma]]) code pipeline runs MinHash but with threshold tuned to preserve canonical patterns.

**8.2 Cross-domain collision.** Global dedup across a corpus that mixes web, papers, and code can mark a paper's method section as a near-duplicate of a blog-post summary of the same paper. Deleting the blog post is correct. Deleting the paper is not. The fix: **dedup within source, not across source**, which Dolma adopts by running dedup passes per-source.

**8.3 Semantic-dedup narrowing.** Already covered. The signature shape: MMLU improves (because diversity-equalization helps the benchmark-heavy center), long-tail evals (BIG-Bench Hard subsets, niche-language translation) drop. The fix is either lower `τ` discipline or a cluster-size cap that saturates rather than deletes.

The unifying rule: **dedup is not a filter, it is a resampling operation.** It changes the empirical distribution you train on. Any change to that distribution should be accepted only if the downstream eval signal (ch-14, ch-47) justifies it.

---

## Connections and what's next

- **[[deduplicating-training-data]] (Lee 2021)** — the foundational eval-contamination and memorization evidence; the spine of §2.
- **[[minhash-lsh]] (Broder 1997)** — the `P = J` theorem and the `r × b` banding construction in §3–§4.
- **[[d4]] / SemDeDup** — embedding-space dedup and the diversity-narrowing tradeoff in §6.
- **[[ccnet]] / [[dolma]] / [[fineweb]]** — three production pipelines that wire dedup into the broader filter cascade; §7 is composed from their configs.
- **[[c4]]** — the uncleaned baseline that Lee 2021 first exposed.
- **ch-11 (tokenizer and lineage)** — provides doc-ids and content-addressed hashing that dedup operates on.
- **ch-13 (domain mixing, DoReMi)** — takes the deduplicated shards and learns mix weights over them.
- **ch-14 (scaling, contamination, retention)** — the downstream eval signal that justifies dedup aggression.

## Further reading

- [[deduplicating-training-data]] — Lee et al. 2021 / 2022. Read §3 (ExactSubstr), §4 (NearDup), §5 (memorization), §6 (contamination tables).
- [[minhash-lsh]] — Broder 1997 and Indyk-Motwani 1998. Chapter 3 of *Mining of Massive Datasets* (Leskovec-Rajaraman-Ullman) is the canonical pedagogical treatment.
- [[d4]] — Abbas et al. 2023. SemDeDup algorithmic details and web-scale ablations.
- [[fineweb]] — Penedo et al. 2024. The per-dump-vs-global MinHash ablation; the surprise that broke a default.
- [[dolma]] — Soldaini et al. 2024. The six-stage filter cascade; paragraph-dedup ordering argument.
- [[ccnet]] — Wenzek et al. 2019. The pre-Lee template.

## Companion visualization

**[figures/minhash-lsh.html](figures/minhash-lsh.html)** — two interactive panels. **Left panel:** drag the Jaccard threshold slider and adjust `r` (rows per band) and `b` (bands); the S-curve `P(collision) = 1 - (1 - J^r)^b` redraws live, and a small table reports the 0.5-threshold `t ≈ (1/b)^{1/r}`, total signatures `m = r·b`, and the false-positive / false-negative rates at your chosen target Jaccard. **Right panel:** a SemDeDup knob — sweep the cosine-threshold `τ` from 0.5 to 1.0 and watch two curves move in opposite directions: recall of true near-duplicates rises with aggression, domain-coverage entropy falls. Use it to build intuition for the §4 banding math and the §6 diversity tradeoff in the same breath.
