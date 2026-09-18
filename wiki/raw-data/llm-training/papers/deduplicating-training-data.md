<!-- scope: Lee et al. 2021: exact-substring (suffix array) and near-duplicate (MinHash) deduplication of C4, RealNews, LM1B, Wiki-40B; effect on perplexity, verbatim memorization, and train-validation overlap
     deps: [[c4]], [[minhash-lsh]]
     see-also: [[d4]], [[data-constrained-scaling]], [[dolma]], [[fineweb]]
-->

# Deduplicating Training Data Makes Language Models Better
- **Core Insight:** In unprompted generation, 1.926% of tokens from a 1.5B model trained for 1 epoch on original C4 belong to 50-token sequences copied from the training data, against 0.189% after near-duplicate removal and 0.138% after exact-substring removal (Table 4), and deduplicated models have no worse perplexity (§1, §6.1).
- **Guideline:** When preparing a web-text pre-training corpus and its validation split, deduplicate stringently within the training set and across splits, because this reduced verbatim memorization about 10× without worse perplexity, and 4.60% of C4 validation examples had a near-duplicate in training (§6.2, Table 2); the authors state that the exact technique matters less than stringent deduplication (§8).
- **Authors:** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, et al. (Google Research Brain Team; University of Pennsylvania)
- **Year:** 2021 (arXiv v1 2021-07; ACL 2022)
- **URL:** https://arxiv.org/abs/2107.06499
- **Source type:** paper
- **Relevant topics:** deduplication, suffix arrays, MinHash, memorization, train-test overlap, evaluation contamination

## Abstract
Existing language-modeling datasets contain many near-duplicate examples and long repeated substrings. As a result, over 1% of the unprompted output of language models trained on them is copied verbatim from the training data. The authors build two deduplication tools; for example, they remove from C4 a single 61-word English sentence that is repeated over 60,000 times. Deduplicated training lets models emit memorized text ten times less often and need fewer training steps to reach the same or better accuracy. Deduplication also reduces train-test overlap, which affects over 4% of the validation set of standard datasets, and so allows more accurate evaluation. Code is released at github.com/google-research/deduplicate-text-datasets.

## Key Contributions
- ExactSubstr: removal of repeated substrings of at least 50 tokens using a suffix array (§4.1).
- NearDup: removal of whole examples with high n-gram overlap using MinHash and edit similarity (§4.2).
- Measurements of duplicate content in C4, RealNews, LM1B, and Wiki-40B, including overlap between training and validation splits (§5, Tables 2–3).
- Perplexity and memorization comparisons for models trained on original, NearDup, and ExactSubstr versions of C4 (§6.1–6.2).
- Measurements of the effect of overlap on released models: Transformer-XL, GROVER, GPT-Neo (§6.3).

## Key Figures/Tables to Study
- **Figure 1** — size distribution of near-duplicate clusters found by NearDup in C4.
- **Tables 2–3** — duplicate fractions by dataset: examples (NearDup) and tokens (ExactSubstr), within training and across splits.
- **Figure 2** — validation perplexity on C4 original, C4 unique, C4 duplicate subsets, LM1B, and Wiki-40B.
- **Table 4 and Figure 3** — unprompted and prompted memorization.
- **Table 5** — perplexity of released models on duplicated vs unique validation examples.

## Technical Details
- **Datasets (§3):** Wiki-40B English, 2.9M pages, mean 768 BPE tokens; LM1B, 30M sentences, mean 32 tokens; C4, 360M documents, mean 486 tokens; RealNews, 31M documents, mean 793 tokens.
- **ExactSubstr (§4.1):** all examples are concatenated into one sequence of BPE-token bytes, and a suffix array is built. Adjacent suffix-array entries that share a prefix of at least 50 tokens mark a repeated substring, which is removed from one of the examples.
- **Threshold choice (App. B, Fig. 5):** matches shorter than 10 tokens are common. Manual inspection of 25-token matches found no false positives, and the authors doubled 25 to 50 for margin.
- **ExactSubstr cost (App. B):** linear-time SA-IS construction on parallel splits followed by a merge that costs O(N·m·log K). On one 96-core, 768GB machine, the 350GB C4 suffix array takes under 12 hours wall-clock (about 1000 CPU-hours) and occupies 1.5TB; deduplication then takes under an hour.
- **NearDup (§4.2, App. A):** documents are space-tokenized into 5-grams. The MinHash signature has k = 9,000 hash values split into r = 450 buckets of b = 20 hashes each. A pair becomes a candidate with probability 1 − (1 − s^b)^r, where s is the pair's Jaccard index. Candidates with Jaccard index above 0.8 and edit similarity above 0.8 are duplicates. Connected components of the duplicate graph form clusters. An alternative setting (0.9/0.9, b = 20, r = 40, k = 800) was also tested (App. A, Fig. 4).
- **Edit similarity (§4.2):** EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j) / max(|x_i|, |x_j|), where x_i and x_j are token sequences.
- **Cross-split rule (§5):** when text occurs in more than one split, the copy in the validation or test split is kept and the training copy is removed.
- **Training examples marked as near-duplicates (Table 2):** C4 3.04%, RealNews 13.63%, LM1B 4.86%, Wiki-40B 0.39%.
- **Training tokens in repeated 50-token substrings (Table 3):** C4 7.18%, RealNews 19.4%, LM1B 0.76%, Wiki-40B 2.76%. 77% of the C4 examples removed by NearDup also contain an ExactSubstr match (§5.1).
- **C4 clusters (§5.1, Fig. 1):** most near-duplicate clusters (1.8M) are single pairs; 280 clusters have more than 5,000 examples, and the largest has 250,933. One 61-word sequence occurs 61,036 times in training and 61 times in validation (§1).
- **Validation examples with a near-duplicate in training (Table 2):** C4 4.60%, RealNews 14.35%, LM1B 4.92%, Wiki-40B 0.72%. Validation tokens in 50-token matches with training (Table 3): C4 1.38%, RealNews 3.37%, LM1B 0.019%, Wiki-40B 0.67%.
- **Perplexity (§6.1, Fig. 2):** base models trained on the three C4 versions have similar perplexity on the full C4 validation set and on its unique subset. The deduplicated models have higher perplexity on validation examples that have training duplicates. ExactSubstr lowers XL perplexity on Wiki-40B by almost 3 points (§6.1). Dataset size falls by up to 19% (§1).
- **Unprompted memorization (§6.2, Table 4):** 100,000 samples of up to 512 tokens with top-k = 50 sampling. A token counts as memorized if it is part of a 50-token substring found in the training data. After 1 epoch: Original 1.926%, NearDup 0.189%, ExactSubstr 0.138%. After 2 epochs: 1.571%, 0.264%, 0.168%.
- **Prompted memorization (§6.2, Fig. 3):** with 32-token prompts from training examples that have duplicates, the model trained on original C4 reproduces the true continuation (edit similarity above 0.8) over 40% of the time.
- **Released models (§6.3, Table 5):** Transformer-XL perplexity on LM1B validation is 21.77 overall, 10.11 on examples with training near-duplicates, and 23.58 on unique examples. 1.38% of tokens in 25k released GROVER-Mega outputs, and more than 5% of tokens in about 200k GPT-Neo 1.3B outputs, are part of 50-token training-data matches.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| XL-Original / XL-NearDup / XL-ExactSubstr (T5 codebase, decoder-only; not released) | 1.5B | pretrain-stable | architecture | 24 layers, 32 heads, d_model 2,048, FFN 5,120, key/value size 64 | arXiv:2107.06499v2 App. C | verified 2026-09-14 | not applicable |
| same | 1.5B | pretrain-stable | steps; batch; epochs | 152K steps (Original, ExactSubstr), 146K steps (NearDup); batch 4,800 (unit not stated); about 2 epochs | App. C | verified 2026-09-14 | no ablation reported |
| same | 1.5B | pretrain-stable | optimizer; LR | Adafactor; constant 0.001 | App. C | verified 2026-09-14 | no ablation reported |
| Base-Original / NearDup / ExactSubstr (3 seeds each) | 110M | pretrain-stable | architecture; optimizer; LR | 12 layers, 12 heads, d_model 768, FFN 2,048; Adafactor; constant 0.01 | §6, App. C | verified 2026-09-14 | no ablation reported |
| all above | 110M, 1.5B | pretrain-stable | sequence length; tokenizer | max 512 tokens (random subsequence of longer documents); BPE trained on C4-NearDup, 50K budget | §6 | verified 2026-09-14 | no ablation reported |
| all above | 1.5B | pretrain-stable | compute | 128-core TPU v3 slice; about 131 h (Original, ExactSubstr), about 126 h (NearDup) | App. C | verified 2026-09-14 | not applicable |
| data pipeline | — | data dedup | ExactSubstr minimum match | 50 BPE tokens | §4.1.2, App. B | verified 2026-09-14 | knee at 10 tokens; no false positives at 25; doubled (App. B, Fig. 5) |
| data pipeline | — | data dedup | NearDup thresholds | Jaccard > 0.8 and edit similarity > 0.8; b = 20, r = 450, k = 9,000 | §4.2, App. A | verified 2026-09-14 | 0.8 and 0.9 similarity histograms vanish at the same point (App. A, Fig. 4) |

## Findings relevant to generality
- **Evaluation bias:** train-test overlap causes over-estimation of model accuracy and biases model selection toward models and hyperparameters that overfit the training data (§1).
- **Residual memorization:** models trained on deduplicated data still copy the true continuation more often for prompts from duplicated examples than from unique ones; the authors suggest stricter deduplication may be needed (§6.2, Interpretation).
- **Not tested:** effects on downstream benchmark tasks (§2). Negative consequences of deduplication are not investigated, including tasks that need memorization (document retrieval, closed-book question answering) and removal of duplicated attribution text (§7).
- **Privacy:** deduplication reduces some privacy concerns but is not sufficient to remove privacy-sensitive data (§7, §8).

## Connections
- [[c4]] — the corpus used for all model training in this paper (§6).
- [[minhash-lsh]] — NearDup uses MinHash (Broder 1997) (§4.2).
- [[data-constrained-scaling]] — uses this paper's suffix-array method for a 100-character deduplication filter and finds it does not improve downstream results on C4 (Muennighoff et al. §7, App. N).
- [[d4]] — applies embedding-based selection after MinHash deduplication and compares its MinHash settings with this paper's (D4 App. A.1.2).
- [[dolma]], [[fineweb]] — later open corpora with their own deduplication stages.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2107.06499 (arXiv v2, 2022-03-24; full text including App. A–E).
- Corrections to the previous card version:
  - "3.04% of training tokens are in near-duplicate clusters" → 3.04% of C4 training examples (Table 2); the token figure is 7.18% for ExactSubstr (Table 3).
  - "4.6% of LM1B validation overlaps training" → 4.60% is C4; LM1B is 4.92% (Table 2). "3.2% of C4 validation" does not appear; C4 is 4.60% of examples and 1.38% of tokens (Tables 2–3).
  - "LSH with b = 20 bands of r = 450 rows, threshold ≈ (1/b)^(1/r) ≈ 0.8"; "9000 MinHash signatures"; "any document exceeding threshold is dropped" → one signature of 9,000 hashes in 450 buckets of 20; candidate probability 1 − (1 − s^b)^r; Jaccard and edit-similarity checks at 0.8; connected-component clusters (§4.2, App. A).
  - "Runs in O(N log N)" → linear-time construction per split, O(N·m·log K) merge (App. B).
  - "~1% of unprompted 256-token completions" → samples of up to 512 tokens; 1.926% (1 epoch) and 1.571% (2 epochs) of tokens (Table 4).
  - "reduces training set by ~5% while improving all downstream metrics" → C4-NearDup has about 350M vs 365M examples (App. C); downstream tasks were not evaluated (§2).
  - "Figure of duplicate count per document in C4 and Wiki-40B" → Figure 1 shows NearDup cluster sizes for C4 only; the "training-curve figure" does not exist (Figure 2 shows final perplexity).
  - "with almost no downside" → no disadvantages were observed (§1), but negative consequences were not investigated (§7). "Carlini 2021 line (later work)" → the paper cites Carlini et al. (2020) as prior extraction work (§2).
- Removed as unsupported by the source: "single highest-ROI pretraining data operation, adopted in every subsequent open corpus"; "run dedup before any other quality filter"; "dedup alone substantially reduces the surface for membership-inference and extraction attacks"; "directly motivated [[data-constrained-scaling]]'s question".
- Not reported by the source: paraphrase-level contamination; contamination detection without training-data access; change in downstream benchmark scores caused by contamination.
