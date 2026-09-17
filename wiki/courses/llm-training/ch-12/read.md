<!-- chapter: ch-12
     track: pretraining
     kind: content
     title: Deduplication: Exact, Near-Duplicate, and Semantic
     deps: [ch-11]
     sources: [[deduplicating-training-data]], [[minhash-lsh]], [[semdedup]], [[d4]], [[repeated-data-scaling]], [[quantifying-memorization]], [[fineweb]], [[fineweb-2]], [[llama-3]], [[llama-3-recipe]], [[dolma]], [[ccnet]], [[c4]], [[data-constrained-scaling]], [[paloma]]
     figures: figures/minhash-lsh.html
     revised: 2026-09 (generality revision)
-->

# Chapter 12 — Deduplication: Exact, Near-Duplicate, and Semantic

> **Core insight.** Deduplication changes three measurable properties of a pretraining run: how often the model reproduces training text, how much of a "held-out" measurement is actually held out, and how model capacity is spent. In 1.5B models trained on C4, near-duplicate removal lowered the share of generated tokens inside 50-token training matches from 1.926% to 0.189%, and 4.60% of C4 validation examples had a near-duplicate in the training split ([[deduplicating-training-data]] Tables 2, 4). The structure of repetition matters more than its token share: repeating 0.1% of the data 100 times (10% of tokens) degraded an 800M model to roughly the loss of a model half its size ([[repeated-data-scaling]] Abstract, §2), whereas two epochs over a D4-selected half of a 40B-token pool beat one pass over all of it at 1.3B ([[d4]] Table 1). More removal is not always better: global MinHash across 96 Common Crawl snapshots gave little gain over no deduplication, while per-snapshot MinHash matched RefinedWeb ([[fineweb]] §3.4).
>
> **Guideline.** When configuring MinHash LSH, write the candidate probability with named symbols (hashes per bucket, number of buckets) and evaluate it at s = 0.7, 0.8, and 0.9 before running, because the two parameters are easy to swap and the swapped reading of the Lee et al. setting finds almost no pairs below s = 0.99 (§4). When a corpus is built from many crawl snapshots, deduplicate within each snapshot and evaluate the choice at a token budget large enough to contain repeats (350B tokens in [[fineweb]] App. E.2), because global deduplication left lower-quality residue in old snapshots (§3.4). When extraction of repeated spans matters (licenses, boilerplate, private text inside otherwise distinct documents), add span-, line-, or paragraph-level exact deduplication, because document-level near-duplicate removal does not see a short span in a long document (§5). When unique data is smaller than the token budget, repeat a diversified selection for a small number of epochs rather than repeating a random subset, and monitor training loss on any heavily repeated subset, because loss on the repeated subset approaching zero coincided with the peak degradation in [[repeated-data-scaling]] (§1.1). When validation or test splits are drawn from the same corpus as training data, deduplicate across splits and keep the evaluation copy, because 4.60% of C4 validation examples had a near-duplicate in training ([[deduplicating-training-data]] §5, Table 2).

## Why this chapter matters for a general-purpose model

The pipeline for a general-purpose model is pre-training → mid-training → SFT → preference optimization → RL → evaluation. This chapter sits inside pre-training data curation, after heuristic filtering (ch-10), model-based selection (ch-10a), and tokenizer, provenance, and PII decisions (ch-11), and before the study of memorization (ch-12a) and domain mixing (ch-13). The same operation reappears in synthetic-data pipelines (ch-18) and SFT data curation (ch-33).

**Deduplication** is the removal or down-weighting of training text that repeats other training text under a stated match rule. The measurable problems it addresses are:

1. **Verbatim reproduction.** Models emit memorized training text more often for text that is repeated more often ([[deduplicating-training-data]] Table 4; [[quantifying-memorization]] §4.2).
2. **Invalid held-out measurement.** A validation example with a near-duplicate in training measures recall, not generalization. Transformer-XL had perplexity 10.11 on LM1B validation examples with training near-duplicates and 23.58 on unique ones ([[deduplicating-training-data]] Table 5).
3. **Capacity spent on repeats.** A dataset with 10% repeated tokens reduced performance by an effective 2× in parameter count, which the authors state is much more than the loss from never training on that 10% ([[repeated-data-scaling]] §5.1).
4. **Distribution shift from over-removal.** Removal changes what remains. FineWeb's global deduplication kept a residue that trained a worse model than the data it removed ([[fineweb]] §3.4, Fig. 4).

For a general-purpose model, problems 2 and 3 are the ones that most directly concern breadth: the first makes generality hard to measure, and the second removes general mechanisms such as in-context copying ([[repeated-data-scaling]] §2). The chapter covers the three method families (exact, near-duplicate, semantic), then repetition structure, then scope and order.

## §1 Units, granularity, and scope of a duplicate

**Definitions.**
- An **exact duplicate** is text identical to other text after a stated normalization, at a stated unit: URL, document, paragraph, line, or span.
- A **repeated substring** is a token span of at least a threshold length that occurs in more than one example; Lee et al. use 50 BPE tokens ([[deduplicating-training-data]] §4.1).
- A **near-duplicate** is a document whose set of n-grams overlaps highly with another document's set, measured by Jaccard similarity (§3).
- A **semantic duplicate** has "largely identical information content, but remain[s] perceptually distinct", for example a templated page in which a place name changes; **semantically redundant** data comes from different underlying objects with overlapping information ([[semdedup]] §1, §5.3).
- **Scope** is the set a document is compared against: its own snapshot, all snapshots (global), its language, its source, or the evaluation splits.

**How much repetition web corpora contain.** Measurements depend on the unit, so every number below carries it.

| Measurement | Value | Source |
|---|---|---|
| C4 training examples with a near-duplicate | 3.04% | [[deduplicating-training-data]] Table 2 |
| C4 training tokens inside repeated 50-token substrings | 7.18% | Table 3 |
| RealNews training examples with a near-duplicate / tokens in repeats | 13.63% / 19.4% | Tables 2, 3 |
| Largest C4 near-duplicate cluster | 250,933 examples; 280 clusters above 5,000 | §5.1, Fig. 1 |
| One 61-word sequence in C4 | 61,036 times in training, 61 in validation | §1 |
| Dolma web: documents removed by exact URL / exact document; paragraphs removed by exact paragraph | 53.2% / 14.9% of URL-deduplicated / 18.7% | [[dolma]] §5.4 |
| CCNet February 2019: duplicated paragraphs | "70% of the text" | [[ccnet]] §3.2 |

**Worked example: the unit changes the number.** C4's 3.04% counts whole examples that NearDup flags; 7.18% counts tokens inside repeated spans found by ExactSubstr. The two measure different sets and cannot be subtracted or compared as "how much of C4 is duplicated". They overlap: 77% of the C4 examples removed by NearDup also contain an ExactSubstr match ([[deduplicating-training-data]] §5.1).

**Scope changes the number too.** CCNet compared paragraph hashes of one shard against other shards: 42% of a shard's characters remained after comparison with 1 shard and 28% after comparison with 100 shards ([[ccnet]] §4.2, Fig. 4). A reported removal rate without its scope is not comparable with another pipeline's rate.

## §2 Evidence that deduplication changes models: Lee et al. (2021)

**Setting.** Lee et al. (arXiv 2021-07, ACL 2022) built three versions of C4: Original, NearDup (document-level MinHash plus edit similarity, §4), and ExactSubstr (suffix-array span removal, §5). They trained 110M models (3 seeds) and 1.5B decoder-only models for about 2 epochs, maximum length 512 tokens ([[deduplicating-training-data]] §6, App. C).

**Results.**
1. **Unprompted memorization** (Table 4). From 100,000 samples of up to 512 tokens with top-k = 50, a token counts as memorized if it lies inside a 50-token substring of the training data. After 1 epoch: Original 1.926%, NearDup 0.189%, ExactSubstr 0.138%. After 2 epochs: 1.571%, 0.264%, 0.168%. Worked ratio: 1.926 / 0.189 = 10.2 and 1.926 / 0.138 = 14.0, which is the "ten times less often" of the abstract.
2. **Prompted memorization** (Fig. 3). Given 32-token prompts from duplicated training examples, the Original model reproduced the true continuation (edit similarity above 0.8) over 40% of the time.
3. **Perplexity** (§6.1, Fig. 2). The three versions gave similar perplexity on the full C4 validation set and on its unique subset. The deduplicated models had higher perplexity on validation examples that have training duplicates, which is the expected sign when those examples are no longer recalled. ExactSubstr lowered XL perplexity on Wiki-40B by almost 3 points.
4. **Train–validation overlap** (Table 2). Validation examples with a near-duplicate in training: C4 4.60%, RealNews 14.35%, LM1B 4.92%, Wiki-40B 0.72%. Validation tokens in 50-token matches with training (Table 3): C4 1.38%, RealNews 3.37%, LM1B 0.019%, Wiki-40B 0.67%.

**Status.** The memorization and perplexity results are Result (single study). Carlini et al. re-measured the same three 1.5B models with a prompted extraction test and also found lower memorization after deduplication ([[quantifying-memorization]] §5.2); this is a second measurement on the same models, not an independent training run.

**Conditions and limits.** Downstream tasks were not evaluated (§2). Negative consequences, including for tasks that need memorization such as closed-book question answering, were not investigated (§7). In a separate study, a 100-character-span deduplication filter did not improve the 19-task downstream average on C4 but was more effective on the noisier OSCAR corpus; the authors note that benefits such as reduced memorization are not measured by that benchmark ([[data-constrained-scaling]] §7, App. O).

**Implication.** At C4 scale, the measured case for deduplication rests on memorization and on measurement validity. Benchmark gains from deduplication depend on how noisy the source is and are not established by Lee et al.

## §3 MinHash: an unbiased estimate of Jaccard similarity

**Definition.** For documents A and B with shingle sets S(A) and S(B) (a shingle is a contiguous run of w tokens), the Jaccard similarity is

```
J(A, B) = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|
```

S(D) is the set of w-shingles of document D; |·| is set size. Broder calls J the resemblance ([[minhash-lsh]] §2).

**Problem.** Computing J for all pairs of N documents requires N(N − 1)/2 set comparisons. For C4's 360M documents ([[deduplicating-training-data]] §3) that is about 6.5 × 10¹⁶ pairs (derived).

**Mechanism.**
1. Choose a random permutation π of all possible shingles (in practice, a hash function).
2. For each document keep h_π(D) = min over s in S(D) of π(s), one number per document.
3. Repeat with k independent hash functions to get a signature of k numbers.
4. Estimate J by the fraction of the k positions where two signatures agree.

**Theorem** ([[minhash-lsh]] §3, Theorem 1). For a uniformly random π,

```
Pr[ h_π(A) = h_π(B) ] = J(A, B)
```

Proof step: let α be the element of S(A) ∪ S(B) with the smallest π value. Every element of the union is equally likely to be α. The two minima are equal exactly when α lies in both sets, which has probability |S(A) ∩ S(B)| / |S(A) ∪ S(B)|.

**Worked example.** Use word bigrams (w = 2). A = "the cat sat on the mat" gives {the cat, cat sat, sat on, on the, the mat}. B = "the cat sat on a mat" gives {the cat, cat sat, sat on, on a, a mat}. The intersection has 3 shingles and the union 7, so J = 3/7 ≈ 0.43. Take the permutation that orders the union as (on a, the mat, cat sat, the cat, a mat, on the, sat on). The smallest element of A is "the mat" (rank 2) and of B is "on a" (rank 1), so the minima differ. They are equal only when the first-ranked element of the union is one of the 3 shared bigrams, which happens for 3 of every 7 equally likely first elements.

**Estimator error (derived).** Each position agrees with probability J independently, so the agreement fraction Ĵ has standard deviation √(J(1 − J)/k). At J = 0.8: k = 112 gives 0.038; k = 9,000 gives 0.0042. A Hoeffding bound for error ε with probability 1 − δ needs k ≥ ln(2/δ) / (2ε²); ε = 0.05 and δ = 0.05 give k ≥ 738. Broder judged "100 samples ... reasonable and 200 ... more than enough" for his sketch (§4.2).

**Conditions.** A larger shingle size w makes J more sensitive to one-token edits, because one changed token affects w shingles, and resemblance is not transitive ([[minhash-lsh]] §2). The estimator treats the shingle set, not the order or count of shingles.

## §4 LSH buckets: candidate probability and the published parameter settings

**Definition.** Locality-sensitive hashing (LSH) groups the k signature values into B buckets of h values each (k = h·B). Two documents become a **candidate pair** if all h values agree in at least one bucket. This avoids comparing all pairs: documents are hashed by bucket contents, and only documents sharing a bucket hash are compared.

**Derivation of the candidate probability.** For a pair with Jaccard similarity s:
1. One signature position agrees with probability s (§3).
2. All h positions in one bucket agree with probability s^h.
3. That bucket fails to agree with probability 1 − s^h.
4. All B buckets fail with probability (1 − s^h)^B.
5. At least one bucket agrees:

```
P(s) = 1 − (1 − s^h)^B
```

s is the Jaccard similarity of the pair; h is the number of hashes per bucket; B is the number of buckets. Setting P = 0.5 gives the similarity at which half of pairs become candidates: s½ = (1 − 2^(−1/B))^(1/h), approximately (ln 2 / B)^(1/h) when B is large (derived).

**Notation differs between sources.** The same formula is printed with different letters, so the two parameters are easy to swap when a setting is copied from one source to another; D4's appendix is one documented case (below).

| Source | Printed formula | Hashes per bucket (h) | Number of buckets (B) | k |
|---|---|---|---|---|
| Lee et al. NearDup ([[deduplicating-training-data]] §4.2, App. A) | 1 − (1 − s^b)^r, "r buckets, with b hashes per bucket" | b = 20 | r = 450 | 9,000 |
| Lee et al. alternative (App. A) | same | b = 20 | r = 40 | 800 |
| FineWeb ([[fineweb]] §3.4, App. E.1) | 1 − (1 − s⁸)¹⁴ | 8 | 14 | 112 |
| *Mining of Massive Datasets* Example 3.12 (pointer in [[minhash-lsh]]) | 1 − (1 − s^r)^b, b bands of r rows | r = 5 | b = 20 | 100 |

In Lee et al. the letter b means hashes per bucket; in the textbook b means number of bands. D4's appendix describes the Lee et al. and RefinedWeb setting as "20 buckets, 450 hashes per bucket" ([[d4]] App. A.1.2), which is the swapped reading.

**Worked example: checking which reading is correct.** Lee et al. state that they chose b and r "to make sure a collision at the desired Jaccard index threshold of 0.8 had a high probability of occurring" (App. A).
- Reading as printed (h = 20, B = 450). At s = 0.8: 0.8^20 = 0.0115; ln(1 − 0.0115) × 450 = −5.22; e^−5.22 = 0.0054; P = 0.995. At s = 0.7: 0.7^20 = 0.00080; (1 − 0.00080)^450 = 0.698; P = 0.30. The half point is s½ = 0.72. This matches the stated goal.
- Swapped reading (h = 450, B = 20). At s = 0.8: 0.8^450 = e^(450 × −0.2231) ≈ 2.5 × 10⁻⁴⁴, so P ≈ 20 × 2.5 × 10⁻⁴⁴ ≈ 5 × 10⁻⁴³. The half point is s½ = 0.9925. Almost no pair between 0.8 and 0.99 would ever be verified, which contradicts the stated goal.
- The alternative configuration targets 0.9: with h = 20 and B = 40, P(0.9) = 0.994 and P(0.8) = 0.37, with half point 0.82. This is also consistent only with the printed reading.

**Candidate versus decision.** In Lee et al., LSH produces candidates; a pair is a duplicate only if its actual Jaccard index exceeds 0.8 and its edit similarity, EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j) / max(|x_i|, |x_j|), also exceeds 0.8 (App. A, §4.2). Connected components of the duplicate graph become clusters. In FineWeb, "Documents with the same 8 MinHashes in any bucket are considered duplicates", with no separate verification, followed by transitive clustering and random choice of one kept document per cluster (§3.4). Under FineWeb's setting, a pair at s = 0.6 is matched with probability 1 − (1 − 0.6⁸)¹⁴ = 0.21 (derived), and transitivity can place two documents in one cluster even when they do not match each other.

**Evidence for the compute trade-off.** FineWeb compared its curve with the 9,000-hash setting: the larger setting gives "a steeper, more well-defined cut off", and the authors "believe the compute and storage savings make up for the higher uncertainty on documents near the threshold" (App. E.1, Fig. 13). No ablation of the two settings on model quality is reported (Interpretation by the authors).

The released FineWeb configuration (github.com/huggingface/datatrove@acd431b, `examples/fineweb.py` L80–88):

```python
minhash_config = MinhashConfig(
    hash_config=HashConfig(
        hash_fc="sha1",  # better precision -> fewer false positives (collisions)
        precision=64,
    ),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)
```

The companion figure [figures/minhash-lsh.html](figures/minhash-lsh.html) plots P(s) for any hashes-per-bucket and bucket count, loads the Lee et al., FineWeb, and textbook settings and the swapped Lee reading as presets, and prints P(s) at 0.5–0.9 so the reader can repeat the checks above.

**Implication.** A MinHash setting is a statement about which similarity range is removed. Without a verification step, the S-shaped curve itself sets the false-positive rate, and pairs well below the nominal target are removed with measurable probability.

## §5 Exact substring deduplication with suffix arrays

**Definition.** ExactSubstr removes a token span of at least 50 tokens from one of two examples that share it ([[deduplicating-training-data]] §4.1).

**Problem.** Document-level Jaccard cannot detect a short shared span in long documents. Two 5,000-shingle documents that share a 50-shingle block have J ≈ 50 / 9,950 ≈ 0.005 (derived), far below any near-duplicate threshold, yet the shared block is repeated text that the model sees twice.

**Mechanism** (§4.1.1–§4.1.2).
1. Concatenate all examples into one sequence S of BPE-token bytes.
2. Build the suffix array A(S) = arg sort all_suffixes(S), the start positions of all suffixes in lexicographic order.
3. A sequence repeated at positions i and j places i and j next to each other in A.
4. Scan A once and record adjacent entries whose common prefix has at least 50 tokens.
5. Remove the repeated span from one of the examples.

**Worked example** (from §4.1.1). The suffixes of "banana" are banana(1), anana(2), nana(3), ana(4), na(5), a(6). Sorted: a(6), ana(4), anana(2), banana(1), na(5), nana(3), so the suffix array is (6 4 2 1 5 3). Common prefixes of adjacent entries: a/ana = 1, ana/anana = 3, anana/banana = 0, banana/na = 0, na/nana = 2. The longest, 3, reveals that "ana" occurs twice, at positions 4 and 2.

**Threshold and cost.** Matches shorter than 10 tokens are common; manual inspection of 25-token matches found no false positives, and the authors doubled 25 to 50 for margin (App. B, Fig. 5). Suffix arrays take 8 bytes per input token and are built in linear time (§4.1.1); the implementation runs SA-IS on parallel splits and merges them. The 350GB C4 suffix array took under 12 hours on one 96-core, 768GB machine and occupied 1.5TB (App. B).

**Evidence.** ExactSubstr gave the lowest unprompted memorization in Table 4 (0.138% after 1 epoch) and the Wiki-40B perplexity gain of almost 3 points (§6.1).

**Cheaper exact units used in released pipelines.**
- C4 discards all but one of any three-sentence span occurring more than once ([[c4]] §2.2).
- Dolma removes exact duplicate paragraphs with a Bloom filter after quality and content filtering (18.7% of paragraphs) ([[dolma]] §5.4).
- Llama 3 removes "lines that appeared more than 6 times in each bucket of 30M documents"; the authors report that this "removes not only leftover boilerplate ... but also frequent high-quality text", while "our empirical evaluations showed strong improvements" ([[llama-3]] §3.1.1; no numbers printed, [[llama-3-recipe]]).

**Conditions and limits.** Span removal leaves the rest of a document in place, which can break sentences; no source in this chapter measures the effect of the resulting fragments. Line-level rules remove frequent good lines as well as boilerplate (Llama 3 §3.1.1).

## §6 Semantic deduplication: SemDeDup

**Definition.** SemDeDup removes semantic duplicates by comparing embeddings from a pre-trained model instead of surface n-grams ([[semdedup]] §3).

**Problem.** Templated pages in which a name or place changes, and rewrites of the same content, have low n-gram overlap and pass exact and MinHash deduplication. D4 reports dense clusters of templated text remaining in embedding space after MinHash ([[d4]] §3.4, App. A.9).

**Mechanism** ([[semdedup]] §3, Algorithm A7).
1. Embed each document; for text, the last-layer embedding of the last token of OPT-125M.
2. Cluster embeddings with k-means (k = 11,000 for C4; 50,000 for LAION).
3. Within each cluster compute all pairwise cosine similarities.
4. Treat pairs with cosine similarity at least 1 − ε as duplicates; keep the member least similar to the centroid.

```
pairwise_sim_matrix = cluster_i_embeddings @ cluster_i_embeddings.T
triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1)
M = torch.max(triu_sim_matrix, dim=0)[0]
points_to_keep_from_cluster_i = M <= 1 - epsilon
```

ε is the dissimilarity threshold; M holds, for each point, its largest similarity to an earlier point in the sorted cluster.

**Worked example: why clustering is needed.** LAION-440M has n = 4.4 × 10⁸ points, so n² = 1.9 × 10¹⁷ comparisons. With k = 50,000 equal clusters, n²/k = 3.9 × 10¹² (derived); the paper reports about 4.6 × 10¹² under the same assumption of approximately uniform cluster size (§3) and does not state how its number was computed. Real clusters are not uniform: the average LAION cluster has 8,748 images, but some have more than 300,000 (Fig. A8). For a fixed n and k, the sum of squared cluster sizes is smallest when all clusters are equal, so unequal clusters make the actual comparison count larger than n²/k (derived).

**Evidence.**
- Images (CLIP ViT-B/16 on LAION-440M, 32 epochs): removing 37% gave no ImageNet zero-shot drop, 50% a drop under 0.5%; average accuracy on 6 out-of-distribution ImageNet variants rose at 37% removed and matched baseline at 50% (§4.4, Fig. 4–5).
- Text (OPT 125M on compute-optimal C4 subsets, one pass, 3 seeds): removing 4% of examples gave C4-validation perplexity 39.35 against 39.51 for random removal and 39.46 for Lee et al.'s NearDup, which removed 3.9%; the no-pruning baseline was 38.95 (Fig. A12). The authors describe NearDup and SemDeDup as comparable at this setting because 4% pruning changes the dataset little (appendix text on "Table A12"). Removing 20% gave opt_valid perplexity 49.04 against 50.66 for random removal and 47.13 for the unpruned baseline (Fig. A13). The appendix captions these settings "96% pruning" and "80% pruning"; the Fig. A12 note states that Random and SemDeDup "prune 4% of examples", so the caption numbers give the share kept. Training several epochs on pruned data reached the one-epoch full-data result with 10–15% less compute (§5.2, Fig. 8).
- What is removed in C4: templated text at small ε, and at larger ε semantically redundant clusters such as advertisements for one shoe brand (§5.3).

**Conditions and limits.** ε is tuned to reach a target dataset size (§6.5), so it is not a fixed definition of "duplicate"; Fig. A17 plots the C4 kept fraction over ε from 0.100 to 0.250. The text experiments use 125M and 1.3B models, and the authors state that C4 gains were more modest than LAION gains because C4 is partially curated (§8). The embedding-model ablation (§6.2, Table 2) was run on images only. Per-validation-set text results compare SemDeDup with random pruning at equal size (SemDeDup had lower perplexity on every opt_valid set, Fig. A20), not with the unpruned corpus, so a loss of niche-domain coverage relative to the full corpus is neither shown nor excluded (Open question).

## §7 Diversified selection after deduplication: D4

**Definition.** D4 (Tirumala et al., arXiv 2023-08) applies SemDeDup and then SSL Prototypes pruning to MinHash-deduplicated web data ([[d4]] §3.4). SSL Prototypes (Sorscher et al. 2022, as described in D4 §3.4) removes the documents closest to their cluster centroid first.

**Mechanism.**
1. SemDeDup with keep ratio R_dedup on dataset D gives D′.
2. Re-cluster D′ with K-Means (11,000 clusters).
3. SSL Prototypes on D′ with keep ratio R_proto.

```
R = R_dedup × R_proto
```

R is the overall fraction of source documents kept; R_dedup and R_proto are the fractions kept by steps 1 and 3.

**Worked example.** The main experiments use R_dedup = 0.75. For the 6.7B run at R = 0.25, R_proto = 0.25 / 0.75 = 1/3. In the fixed-compute setting the source pool grows as R shrinks, so selecting 100B training tokens at R = 0.25 needs a 400B-token pool (derived); the cost section embeds 400B tokens (§4.3).

**Evidence.**
- Fixed compute, 6.7B, 100B tokens, R = 0.25 (Fig. 1): 22.18% faster to baseline non-web-snapshot perplexity, 18.08% faster on Instruct + Answers perplexity (verbalized OPT-IML instruction data that is never trained on), and 2.04% higher mean 0-shot accuracy over 16 tasks.
- Fixed data, 1.3B, 40B training tokens, 3 seeds (Table 1): one epoch over 40B random tokens gave non-web perplexity 16.27 and Instruct + Answers 14.19; two epochs over 20B random tokens gave 16.39 and 14.37; two epochs over 20B D4 tokens gave 16.10 and 13.85.
- For a fixed selection method, new tokens are usually better than repeated ones (App. A.6, Fig. A13).
- Web-snapshot validation sets (C4, CC-dedup, CommonCrawl) always got worse perplexity under selection; these sets are closest to the training data in embedding space (§4.4.1, Fig. 5).

**Conditions and limits.** One source distribution (web) and models up to 6.7B on 100B tokens (§5). The MinHash step used Spark defaults, not the aggressive Lee et al. setting, and the authors conjecture that stronger MinHash would reduce the need for the SemDeDup step (App. A.1.2).

**Implication.** Two claims coexist in D4: repeating a well-selected subset twice beat fresh random data, and repeating is still worse than fresh data selected the same way. Validation perplexity on web snapshots that resemble the training pool moved in the opposite direction from the instruction-data perplexity and task accuracy, so it is a poor gate for selection decisions ([[d4]] Fig. 6: Pearson −0.368 and −0.188).

## §8 Repetition structure, memorization, and generalization

### §8.1 A small subset repeated many times

**Setting** ([[repeated-data-scaling]] §3, Figure 1). Anthropic models from 1.57M to 805M parameters, 100B training tokens, 8,192-token context. The main text runs hold 10% of tokens as repeats of a small subset and 90% as unique data, and vary how small the subset is; other scans use 3%, 20%, 50%, and 90% repeated tokens (Figure 4 caption). Test loss is measured on held-back data with no repeats.

**Results.**
1. **Double descent** (§2, Figure 3). For some repeat counts, test loss falls, rises midway through training, and falls again; with 50% repeated data the rise becomes a long plateau.
2. **Size of the effect.** The abstract states that an 800M model is degraded to the performance of "a 2x smaller model (400M params) by repeating 0.1% of the data 100 times"; §2 states the worst case for 10% repeated data is "roughly 100x repeats of 0.1% of the data", with performance "nearly to that of a 340M parameter model". Check: 0.1% × 100 = 10% of training tokens.
3. **Diagnostic** (§1.1, Figure 2). The peak degradation coincides with training loss on the repeated subset approaching zero.
4. **General mechanisms** (§1.1, §2). With 3% repeated data at the worst repeat count, loss on a paragraph copied 11 times showed up to a 3× reduction in effective model size, while test loss showed at most 1.15× (Figure 5). The prefix-matching score of induction heads also degraded more than test loss; the paper gives this as an average 32% reduction in effective model size (§1.1; a 1.47 multiplier in the Figure 6 caption) and as a 2× decrease in the §2 text. Python loss for text-trained models showed multipliers of 0.84 and 0.75 at 50% and 90% repeated data (Figure 10).
5. **Downstream of pre-training** (§2, §5.4, Figure 12). A model pre-trained on repeated data at the double-descent peak, then fine-tuned on Python, had a 1.6× reduction in effective parameters compared with training from scratch. §5.4 and the Figure 12 caption state 90% repeated tokens; the §2 text describes the same 800M experiment as 50% repeated data.

**Worked example: why memorization can lower training loss** (§5.1, Interpretation by the authors). An 800M model has loss about 2.0 nats/token and a 400M model about 2.2. Memorizing a 10% repeated share drives its loss to 0. If memorizing costs the capacity equivalent of halving the model, average training loss is 0.9 × 2.2 + 0.1 × 0 = 1.98, which is lower than 2.0. The break-even repeated share is (2.2 − 2.0) / (2.2 − 0) = 9.1% (derived): above it, the training objective favors memorizing. Panel B of [figures/minhash-lsh.html](figures/minhash-lsh.html) recomputes this comparison and the break-even share for other repeated shares and loss values.

**Where the damage occurs** (§2). The region with at least 50% of the maximum degradation fits E = k·N^α on both boundaries, where E is repeated epochs and N is parameters: left boundary k = 4.2 × 10⁶, α = −0.56; right boundary k = 5.1 × 10⁷, α = −0.50. For N = 8 × 10⁸, the band runs from 4.2 × 10⁶ × (8 × 10⁸)^−0.56 ≈ 43 to 5.1 × 10⁷ × (8 × 10⁸)^−0.5 ≈ 1,800 repeated epochs (derived), which contains the 100-repeat case above. For N = 10¹¹ the same fits give about 3 to 160 (derived), matching the authors' statement that extrapolation predicts degradation from repeating data about 2 times for models with hundreds of billions of parameters at a fixed 100B tokens.

**Conditions and limits** (§5.5). All runs used 100B tokens regardless of model size; the repeated subset was random rather than intentionally upsampled high-quality data; the metric was loss, not downstream tasks; no regularization was tested; the region fits were noisy and needed aggregation. The authors state that training past the peak helps, so degradation at compute-optimal token counts is likely smaller (§2).

### §8.2 Repeating the whole dataset

Uniform repetition behaves differently from repeating a small subset. At fixed compute, up to 4 epochs over the whole dataset changed held-out loss negligibly (an 8.7B model at 4 epochs ended with 0.5% higher validation loss than at 1 epoch), and value declined with more epochs ([[data-constrained-scaling]] Abstract, §6); that study notes it covers whole-dataset repetition only (App. Q). In T5 pre-training on 2^35 tokens, repeating a smaller C4 subset 64 times had "limited" effect, while 1,024 repeats lowered SuperGLUE from 71.36 to 64.76 ([[c4]] §3.4.2, Table 9). The two studies agree in direction that a few epochs over all data cost little and many repeats cost more (Replicated in direction). How a corpus with mixed repeat counts (most documents once, a few thousands of times) falls between these regimes is an Open question; ch-14 covers data-constrained scaling in full.

### §8.3 Duplication count and extractable memorization

Carlini et al. (arXiv 2022-02, ICLR 2023) measure a sequence as extractable if a GPT-Neo model given the preceding tokens emits the next 50 tokens exactly under greedy decoding ([[quantifying-memorization]] §3.1–§3.2).
- Memorization grows log-linearly with the number of duplicates, for sequences duplicated between 2 and 900 times, and still occurs with few duplicates (§4.2, Fig. 1b).
- A tenfold increase in model size adds 19 percentage points of memorization on the duplication-normalized sample (R² 99.8%) (§4.1).
- On the Lee et al. 1.5B models, sequences repeated fewer than 35 times in original C4 were memorized 1.2% of the time by the ExactSubstr model and 3.6% without deduplication; deduplication helped for sequences repeated up to about 100 times in original C4 but not for sequences repeated more often, and sequences repeated at least 408 times were significantly more extractable than sequences with any lower repeat count (§5.2, Fig. 4c). The authors hypothesize that scalable deduplication is imperfect and that different valid definitions of a duplicate leave copies behind (Interpretation).

ch-12a treats memorization and knowledge acquisition in full; here the result sets a measurement task: after deduplication, check the residual repeat-count distribution with a separate method, because the highest-count strings are the ones that survived.

## §9 Scope choices: global, per-snapshot, per-language, and across splits

**Global versus per-snapshot, FineWeb** ([[fineweb]] §3.4; details in ch-10 §5.4). Global MinHash over 96 snapshots, iterating from newest to oldest, removed up to 90% of old snapshots and left 4T tokens; a 350B-token run improved little over non-deduplicated data. In snapshot 2013-48, the ~31B tokens kept trained a worse model than 171B tokens rebuilt from the ~460B removed tokens, and the kept data contained "more ads, incoherent lists of keywords and generally badly formatted text". Per-snapshot MinHash gave 20T tokens and matched RefinedWeb. Lighter global methods applied afterwards (URL dedup removing 71.5% of tokens, line dedup 77.8%, 3-line dedup 80.9%) all scored below per-snapshot MinHash alone (App. E.3). The authors hypothesize that the gain comes from removing very large clusters and that removing clusters with fewer than about 100 duplicates can hurt (Interpretation).

**Why small ablations miss this** (App. E.2). In a simulation with 100 identical snapshots of 200B unique tokens, a 1B-token sample is almost entirely unique although every document appears 100 times in the full set. Deduplication variants were therefore compared at 350B tokens.

**Global, Llama 3** ([[llama-3]] §3.1.1). Llama 3 applies URL-level deduplication across the entire dataset, keeping the most recent version of each URL; global MinHash document deduplication; and aggressive line-level deduplication. For multilingual data it runs document- and line-level deduplication within each language. MinHash parameters and ablation numbers are not printed ([[llama-3-recipe]]). The two reports therefore made opposite scope choices for document-level MinHash, and neither compares its choice with the other's under a matched model and budget.

**Global per language with duplication-aware upsampling, FineWeb2** ([[fineweb-2]] §4.3, §4.5). FineWeb2 runs MinHash with FineWeb's parameters globally per language, before filtering, and stores each kept document's original cluster size. Filter removal rates by cluster size are U-shaped for French: singletons and the most-duplicated clusters are removed more often than the global rate of 62.4% (Fig. 2). Documents in the cluster size with the lowest removal rate get weight 10, cluster sizes with removal above the global rate get weight 1, with interpolation between the two. Aggregate scores (1.46B models, 350B tokens) moved from filtered-only to rehydrated data as follows: Arabic 24.2 → 25.2, French 21.9 → 23.6, Russian 22.3 → 23.4, Thai 11.9 → 13.1, Turkish 20.7 → 22.1, Chinese 22.5 → 24.3 (App. A.8, Tables 29–34). Deduplication alone lowered the aggregate for French (18.3 → 18.0) and Thai (11.1 → 11.0) (Tables 30, 32).

| Scope choice | What it removes | Measured effect | Not reported |
|---|---|---|---|
| Global MinHash, 96 snapshots (FineWeb) | up to 90% of old snapshots | little gain over no dedup at 350B tokens | effect at larger models |
| Per-snapshot MinHash (FineWeb) | within-snapshot clusters only | matched RefinedWeb | memorization of cross-snapshot repeats |
| Global MinHash + line dedup (Llama 3) | near-duplicates across all data; lines > 6 times per 30M documents | "strong improvements" | parameters, numbers |
| Global per-language MinHash + rehydration (FineWeb2) | clusters, then re-weights by cluster size | +1.0 to +1.8 aggregate over filtered data in 6 languages | effect on memorization |

**Across splits.** Lee et al. keep the validation or test copy and remove the training copy when text occurs in more than one split ([[deduplicating-training-data]] §5). Pretraining-scale decontamination against public benchmarks is covered in ch-14, and its effect on reported scores in ch-48.

**Where deduplication sits among other stages.** Dolma runs URL and document deduplication before quality filtering to reduce processing, and paragraph deduplication last because "paragraph removal risks disrupting content analysis"; the order is not ablated ([[dolma]] §5.4). CCNet deduplicates paragraphs before language identification, which keeps more low-resource-language documents ([[ccnet]] §4.1, Fig. 3; ch-10 §2.1). FineWeb2 deduplicates before filtering so that cluster size is available as a quality signal ([[fineweb-2]] §4.3). Dolma reports that Gopher rules correlate negatively with deduplication at the document level because they remove random strings that deduplication does not catch ([[dolma]] App. J), so the two stages remove different populations.

## Negative samples and negative feedback

Deduplication produces negatives only in the first sense of the course standard: **negative marginal value**, meaning text judged to lower performance when added again as a positive training target. They are labeled by hash equality, Jaccard and edit-similarity thresholds, or embedding similarity; no human, verifier, or reward model is involved, and no source reports a false-negative rate against human labels. Current practice discards the extra copies (Lee et al., Dolma, FineWeb, Llama 3) or re-weights them (FineWeb2 rehydration, §9). No stage uses a removed duplicate as content, as conditioning, or as a gradient that lowers its likelihood, so likelihood-displacement risks from ch-43a do not arise here; supervised uses of negatives are in ch-31a.

The label is a function of the repeat count, not of the text. The same document is harmful when it forms a small subset repeated at the worst frequency ([[repeated-data-scaling]] §2), neutral when the whole dataset is repeated up to 4 times ([[data-constrained-scaling]] §6), and useful when up-weighted by cluster size in FineWeb2 (§9). **Controls:** store cluster sizes and similarity scores as attributes rather than deleting (FineWeb2 §4.3; the Dolma toolkit writes duplicate spans under `attributes/`, [[dolma]] Connections); keep the evaluation copy across splits. **Diagnostics:** the histogram of cluster sizes before and after deduplication, training loss on any subset repeated more than a few epochs, and the verification-score histogram near the threshold ([[deduplicating-training-data]] App. A, Fig. 4). **Effect on generality:** measured effects are on memorization, copying, and loss ([[repeated-data-scaling]], [[quantifying-memorization]]); no source in this chapter reports effects of deduplication on calibration, hallucination, or refusal.

## Recipe

Rows are data-pipeline settings and the study models that evaluated them. The style standard's stage list has no data-curation stage, so pipeline rows use `pretrain-stable (data)`, following ch-10.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Lee et al. C4-NearDup | n/a (data) | pretrain-stable (data) | shingles; hashes | space-tokenized 5-grams; k = 9,000 = 450 buckets × 20 hashes per bucket | arXiv:2107.06499v2 §4.2, App. A | verified 2026-09-15 | App. A Fig. 4: 0.8 and 0.9 configurations' similarity histograms vanish at the same point |
| Lee et al. C4-NearDup | n/a (data) | pretrain-stable (data) | duplicate rule | LSH candidate, then Jaccard > 0.8 and edit similarity > 0.8; connected components | §4.2, App. A | verified 2026-09-15 | same as above |
| Lee et al. alternative | n/a (data) | pretrain-stable (data) | duplicate rule | Jaccard ≥ 0.9 and edit similarity ≥ 0.9; 40 buckets × 20 hashes, k = 800 | App. A | verified 2026-09-15 | tested for comparison only |
| Lee et al. C4-ExactSubstr | n/a (data) | pretrain-stable (data) | minimum repeated span | 50 BPE tokens | §4.1, App. B | verified 2026-09-15 | App. B Fig. 5: no false positives found at 25 tokens; doubled for margin |
| Lee et al. (all versions) | n/a (data) | eval-gate | cross-split rule | keep validation/test copy, remove training copy | §5 | verified 2026-09-14 ([[deduplicating-training-data]]) | no ablation reported |
| XL-Original / NearDup / ExactSubstr | 1.5B | pretrain-stable | steps; epochs; LR | 152K (Original, ExactSubstr), 146K (NearDup); about 2 epochs; Adafactor constant 0.001 | App. C | verified 2026-09-14 ([[deduplicating-training-data]]) | no ablation reported |
| FineWeb | n/a (data) | pretrain-stable (data) | MinHash | English word 5-grams; 112 hashes = 14 buckets × 8; bucket match = duplicate; transitive clusters; one random document kept; per snapshot | arXiv:2406.17557v2 §3.4, App. E.1; datatrove@acd431b examples/fineweb.py L80–88 | verified 2026-09-15 | §3.4 Fig. 3–5 and App. E.3 Fig. 15 at 1.71B, 350B tokens: global and lighter global variants below per-snapshot |
| Llama 3.1 (all) | all | pretrain-stable (data) | document dedup | URL dedup across entire dataset (most recent kept); global MinHash; MinHash parameters not printed | arXiv:2407.21783v3 §3.1.1 | verified 2026-09-15 (MinHash parameters not reported; body checked) | no ablation reported |
| Llama 3.1 (all) | all | pretrain-stable (data) | line dedup | remove lines appearing more than 6 times in each bucket of 30M documents; per language for multilingual data | v3 §3.1.1 | verified 2026-09-15 | "strong improvements" in empirical evaluations; no numbers |
| Llama 3.1 | not scoped | SFT | semantic dedup | RoBERTa clusters of dialogs sorted by quality × difficulty; greedy keep if maximum cosine similarity to kept examples is below a threshold; threshold not printed | v3 §4.2.3 | verified 2026-09-15 (threshold not reported) | no ablation reported |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | dedup stages | Bloom filter: exact URL (53.2% of documents), exact document (14.9%), exact paragraph after quality and content filters (18.7% of paragraphs) | arXiv:2402.00159v2 §5.4 | verified 2026-09-14 ([[dolma]]) | order justified by efficiency; not ablated |
| D4 CC-dedup source | n/a (data) | pretrain-stable (data) | MinHash | "20 hashes per signature, 20 buckets, and 1 row per bucket" (Spark defaults) | arXiv:2308.12284v1 App. A.1.2 | verified 2026-09-15 | not swept, due to compute limits |
| D4 study models (OPT architecture) | 125M–6.7B | pretrain-stable (data) | selection | OPT-125M last-token embeddings; 11,000 clusters, 20 iterations; R_dedup = 0.75; R = 0.25 at 6.7B | §3.4, §4.1, App. A.1.3 | verified 2026-09-14 ([[d4]]) | R_dedup: highest SemDeDup ratio that improved perplexity (App. A.3); R = 0.25 best at 1.3B (§4.1) |
| D4 study model | 6.7B | pretrain-stable | tokens; peak LR; batch | 100B; 1.2e-4; 2M tokens | §3.2, Table A1 | verified 2026-09-14 ([[d4]]) | no ablation reported |
| SemDeDup C4 study | 125M, 1.3B | pretrain-stable (data) | embedding; clusters; threshold | OPT-125M last-layer last-token embedding; k = 11,000; ε tuned to target size (Fig. A17 plots ε 0.100–0.250) | arXiv:2303.09540v3 §5.1, §6.5, Fig. A17 | verified 2026-09-15 | Fig. 7, A12–A13: SemDeDup vs random and NearDup at 125M, 3 seeds |
| FineWeb2 | n/a (data) | pretrain-stable (data) | MinHash scope; rehydration | FineWeb parameters, global per language, before filtering; weight 10 at lowest-removal cluster size, 1 above global removal rate, interpolated between | arXiv:2506.20920 §4.3, §4.5 ([[fineweb-2]]) | verified 2026-09-15 | App. A.8 Tables 29–34: rehydrated > filtered in 6 canary languages (1.46B, 350B tokens) |

**Starting point for a small general-purpose run.** Every value comes from a verified row above, with the source conditions. For English web text from several crawl snapshots, run document-level MinHash with word 5-grams and 14 buckets of 8 hashes within each snapshot (FineWeb, evaluated at 1.71B parameters and 350B tokens); because this setting has no verification step, check the removal rate of pairs near s = 0.6–0.7 on a sample. When a candidate-then-verify design is affordable, use Lee et al.'s 450 buckets of 20 hashes with Jaccard > 0.8 and edit similarity > 0.8 (C4, 1.5B models). Add span removal at 50 BPE tokens when verbatim memorization of repeated spans is a concern (Lee et al., C4). For multilingual corpora, deduplicate within each language (Llama 3; FineWeb2) and store cluster sizes so that upsampling by cluster size remains possible (FineWeb2, 1.46B models). Keep the evaluation copy of any text shared across splits (Lee et al. §5). Semantic deduplication and D4 selection were evaluated only up to 6.7B parameters on web data; treat their ratios as settings to re-ablate, not defaults.

## Generalization lens

**(a) What increases breadth.**
- Removing repeated text keeps model capacity for general mechanisms: repeated subsets at the worst frequency damaged in-context copying by up to 3× in effective model size and induction-head prefix matching by an average 32% (1.47×), against at most 1.15× on test loss ([[repeated-data-scaling]] §1.1, §2; Result, single study).
- Diversified selection plus two epochs beat one pass over random data on held-out instruction-data perplexity at 1.3B (13.85 vs 14.19) ([[d4]] Table 1), and D4 raised mean accuracy over 16 tasks by 2.04% at 6.7B (Fig. 1).
- Semantic deduplication of LAION-440M raised average out-of-distribution accuracy at 37% removed ([[semdedup]] §4.4).
- Per-snapshot rather than global MinHash kept 20T instead of 4T tokens with better aggregate scores ([[fineweb]] §3.4); deduplicating before language identification keeps more low-resource documents ([[ccnet]] §4.1).
- Up-weighting clusters by filter-informed cluster size raised aggregate scores in 6 of 6 canary languages ([[fineweb-2]] Tables 29–34).

**(b) What causes narrowing or forgetting.**
- A small subset repeated at the worst frequency raised test loss to that of a model about half the size, and extreme repetition left a 1.6× effective-parameter deficit after fine-tuning on Python ([[repeated-data-scaling]] §2, §5.4).
- Global deduplication across many snapshots concentrated old snapshots on lower-quality residue ([[fineweb]] Fig. 4).
- Line-level deduplication removes frequent high-quality text along with boilerplate ([[llama-3]] §3.1.1; no size of effect printed).
- Deduplication lowered French and Thai aggregate scores before filtering and rehydration ([[fineweb-2]] Tables 30, 32).
- Tasks that depend on memorized facts, such as closed-book QA, may lose from deduplication; this was named but not tested ([[deduplicating-training-data]] §7; Open question).

**(c) How to measure it at this stage.**
- Train–evaluation overlap per evaluation set, as the share of examples with a near-duplicate and the share of tokens in 50-token matches ([[deduplicating-training-data]] Tables 2–3).
- Extractable memorization by duplication bucket, prompting with training prefixes ([[quantifying-memorization]] §3.2); report buckets separately because the highest-count strings survive deduplication (§5.2).
- Loss on any repeated subset during training, as an early sign of the capacity trade in §8.1 ([[repeated-data-scaling]] Figure 2).
- Copying and in-context tasks alongside average loss, because they degrade more than loss ([[repeated-data-scaling]] Figures 5–6).
- Per-domain macro perplexity on decontaminated held-out data ([[paloma]] G1–G5), not web-snapshot perplexity alone, because selection worsened web-snapshot perplexity while improving task accuracy ([[d4]] §4.4.1).
- Measurement errors: small samples hide duplication effects ([[fineweb]] App. E.2); validation sets close to the training pool can move opposite to accuracy ([[d4]] Fig. 6); SemDeDup text results are from 125M and 1.3B models ([[semdedup]] §8).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Swapping hashes per bucket and number of buckets | almost no duplicates found, or far too many | evaluate 1 − (1 − s^h)^B at s = 0.7, 0.8, 0.9; Lee's setting must give about 0.30, 0.995, 1.0 |
| Treating an LSH bucket match as proof of similarity | removed pairs with low true Jaccard | compute exact Jaccard on a sample of matched pairs; plot the histogram (Lee App. A Fig. 4) |
| Comparing removal rates in different units | "3% duplicated" compared with "7% duplicated" | report examples and tokens separately with the method (Lee Tables 2–3) |
| Global MinHash across many snapshots without an ablation | old snapshots shrink to 10% or less; little benchmark gain | per-snapshot vs global run at 350B tokens; read kept samples from an old snapshot (FineWeb Fig. 4) |
| Evaluating dedup variants on a 1B-token sample | no difference between variants | count cluster sizes directly; evaluate at a budget that contains repeats (FineWeb App. E.2) |
| Upsampling a small high-quality set many times without monitoring | training loss on that set approaches 0; test loss plateaus or rises mid-training | log loss on the upsampled subset separately; compare repeat count with the E = k·N^α band (Hernandez §2) |
| Deduplicating train without checking evaluation overlap | validation perplexity lower on overlapping examples | split validation into overlapping and unique subsets and report both (Lee Table 5) |
| Using web-snapshot perplexity to choose a selection ratio | chosen setting lowers task accuracy | add held-out instruction-data perplexity and task accuracy (D4 Fig. 6) |
| Assuming deduplication removes all memorization risk | high-count strings still extractable | extraction test by duplication bucket on the trained model (Carlini §5.2) |
| Copying ε or a cosine threshold from another dataset | removal share differs from the target | tune ε on 10% of clusters to a target size and record the size (SemDeDup §6.5) |

## Check your understanding

1. Lee et al. report 3.04% of C4 training examples as near-duplicates and 7.18% of C4 training tokens inside repeated spans. Explain why neither number is "the duplicated share of C4", and what each would miss.
2. Using P(s) = 1 − (1 − s^h)^B, explain why FineWeb's 14 × 8 setting removes some pairs at s = 0.6 while Lee et al.'s pipeline does not, even though both target similarity near 0.75–0.8.
3. A colleague configures "20 buckets of 450 hashes" to reproduce Lee et al. and finds almost no near-duplicates in C4. Walk through the calculation that shows why, and state which sentence in Lee et al.'s appendix settles the correct reading.
4. In Hernandez et al., 10% of training tokens were repeats, yet the model lost the equivalent of about half its parameters. Explain the loss arithmetic behind this, and why the damage peaks at an intermediate repeat count rather than the largest one.
5. D4 found that two epochs over D4-selected data beat one epoch over random data, while Hernandez et al. found that repetition can degrade a model severely. What differs between the two setups (repeat count, subset selection, share of tokens), and what would you measure to tell which regime a new corpus is in?
6. FineWeb's global deduplication left 4T tokens and gave little gain, while Llama 3 used global MinHash and reported improvements. List the differences in pipeline and reporting that prevent these two results from being compared directly.
7. After ExactSubstr deduplication, sequences repeated at least 408 times in original C4 were still memorized more than less repeated ones. Explain how a deduplication method can leave such strings, and what check you would add.
8. D4's selection worsened perplexity on C4 and CommonCrawl validation sets but improved instruction-data perplexity and accuracy. Explain why validation sets close to the training pool are unreliable for this decision.

## Connections

- ch-11 — Tokenizers, Data Provenance, and PII Removal (previous chapter: tokenization, provenance search, and PII removal before deduplication).
- ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining (next chapter: full treatment of memorization scaling that §8.3 introduces).
- ch-10 — Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb (FineWeb MinHash settings and the global-versus-per-snapshot study, §5.3–§5.4).
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (selection after deduplication).
- ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale (mixture weights over the deduplicated pool).
- ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (FineWeb2 per-language deduplication).
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (whole-dataset repetition and benchmark decontamination).
- ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement (implements exact and MinHash deduplication on a CommonCrawl slice).
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix (deduplication of generated data).
- ch-33 — Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability (semantic deduplication of SFT data).
- ch-48 — Contamination Detection and Its Effect on Reported Scores (evaluation-side overlap).

## Sources

- [[deduplicating-training-data]] — ExactSubstr and NearDup definitions, parameters, duplication tables, memorization and perplexity results, cross-split rule, limits (§2, §4, §5, Recipe).
- [[minhash-lsh]] — resemblance definition, Theorem 1 and its proof step, sample-size guidance, pointer to the textbook banding formula (§3, §4).
- [[semdedup]] — semantic-duplicate taxonomy, algorithm and pseudocode, LAION and C4 results, threshold tuning, limits (§6).
- [[d4]] — D4 selection steps, fixed-compute and repeated-token results, web-snapshot validation analysis, MinHash settings (§4, §7, Recipe).
- [[repeated-data-scaling]] — double descent from repeated subsets, effective-size losses, copying and induction-head damage, the §5.1 loss arithmetic, boundary fits, limits (§8.1).
- [[quantifying-memorization]] — extractability definition, log-linear duplication and model-size relations, residual memorization after deduplication (§2, §8.3).
- [[fineweb]] — MinHash setting without verification, global versus per-snapshot results, lighter global methods, sample-size simulation (§4, §9).
- [[fineweb-2]] — per-language global MinHash, rehydration weights and scores (§9, Recipe).
- [[llama-3]] — URL, global MinHash, and line-level deduplication; SFT semantic deduplication (§5, §9, Recipe).
- [[llama-3-recipe]] — ledger row confirming the line-deduplication rule and the absence of printed numbers (§5, §9).
- [[dolma]] — Bloom-filter deduplication stages, removal rates, stage order and its stated reason, filter correlations (§1, §9, Recipe).
- [[ccnet]] — paragraph duplication share, deduplication scope measurement, dedup-before-language-identification (§1, §9).
- [[c4]] — three-sentence span deduplication and T5 repetition results (§5, §8.2).
- [[data-constrained-scaling]] — whole-dataset repetition up to 4 epochs and the C4/OSCAR deduplication-filter result (§2, §8.2).
- [[paloma]] — per-domain perplexity with decontaminated training data as a breadth measurement (Generalization lens).
