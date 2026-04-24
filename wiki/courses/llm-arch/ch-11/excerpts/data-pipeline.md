# Data Pipeline Deep-Dive: From Common Crawl to Training Tokens

<!-- scope: deduplication algorithms, quality classifiers, data mixing optimization
     parent: [[ch-11]]
-->

## The Deduplication Problem

Duplicate and near-duplicate documents are the single largest source of wasted compute in pre-training. Common Crawl contains enormous quantities of boilerplate: cookie banners, navigation menus, syndicated news articles republished across hundreds of sites, and scraped content farms that clone entire websites.

### Exact Deduplication

The simplest approach: hash each document (or URL) and remove exact matches. Llama 3 ([[llama-3|report]]) applies URL-level exact deduplication as the first filtering step, which is computationally cheap (a single hash table lookup per document) and removes the most obvious copies.

But exact dedup misses near-duplicates — documents that differ by a timestamp, a byline, or minor reformatting. These near-duplicates are actually more dangerous than exact copies because they are harder to detect and more numerous.

### MinHash: Locality-Sensitive Hashing for Near-Duplicates

MinHash estimates the Jaccard similarity between documents by comparing hash signatures rather than full text. The algorithm:

1. **Shingling:** Convert each document into a set of n-grams (typically character 5-grams or word 3-grams).
2. **Hashing:** Apply $k$ independent hash functions to the shingle set. For each hash function, keep only the minimum hash value — this is the MinHash signature.
3. **Banding:** Divide the $k$-dimensional signature into $b$ bands of $r$ rows each. Two documents are candidate duplicates if they match in *all* rows of *any* band.

The probability that two documents with Jaccard similarity $J$ are detected as candidates is:

$$P(\text{candidate}) = 1 - (1 - J^r)^b$$

This is a sigmoid-like function of $J$, with the threshold controlled by $b$ and $r$. Typical configurations use $k = 128$, $b = 16$, $r = 8$, which gives a ~50% detection probability at $J \approx 0.5$ and >99% at $J \approx 0.8$.

### Deduplication at Scale

For Llama 3's 15T-token corpus, deduplication is itself a distributed systems problem. The MinHash computation must be parallelized across the entire dataset, and the candidate pair generation produces billions of pairs that must be clustered and resolved. The standard approach is:

1. Compute MinHash signatures for all documents (embarrassingly parallel).
2. Use locality-sensitive hashing (LSH) to find candidate pairs (MapReduce).
3. Connected-components clustering on candidate pairs to identify duplicate groups.
4. Keep one document per cluster (typically the longest or highest-quality).

This pipeline runs on hundreds of CPUs and takes days even for a well-resourced lab. The deduplication rate is typically 30-50% of documents after heuristic filtering — meaning that a large fraction of "unique" web pages are near-copies of each other.

## Quality Classifiers: The Critical Filter

### The Wikipedia Proxy

The dominant approach for quality classification: train a binary classifier (typically fastText for speed) with Wikipedia articles as positive examples and random web documents as negatives. The assumption: text that "looks like Wikipedia" is high quality.

This is a deeply imperfect proxy. Wikipedia has specific stylistic conventions (neutral point of view, citation-heavy, encyclopedic tone) that do not characterize all high-quality text. A beautifully written novel, a rigorous technical blog post, or a well-structured code tutorial may all fail the "looks like Wikipedia" test. Conversely, some low-quality content may mimic Wikipedia's style without being informative.

Despite these limitations, the Wikipedia classifier is remarkably effective at removing spam, machine-generated content, and extremely low-effort text. The key is the **threshold**: labs spend significant compute on ablating the classifier threshold against downstream benchmark performance. Setting the threshold too high (keeping only the most Wikipedia-like text) destroys data diversity and can introduce systematic biases. Setting it too low lets through too much noise.

### Beyond Binary Classification

More sophisticated approaches use multi-dimensional quality scoring:

- **Perplexity filtering:** Score each document with a pre-trained language model. Very high perplexity suggests gibberish or very unusual text; very low perplexity suggests repetitive, templated content. The sweet spot is moderate perplexity — text that is coherent but not trivially predictable.
- **Domain-specific classifiers:** Separate classifiers for code quality, mathematical content quality, scientific writing quality. This allows domain-appropriate thresholds rather than one-size-fits-all.
- **Toxicity and safety filtering:** Separate from quality per se, but applied in the same pipeline. Typically uses a dedicated classifier trained on labeled toxic content.

## Data Mixing: An Optimization Problem

Given $K$ data sources (web text, code, math, books, multilingual, synthetic), the mixing ratio $\mathbf{w} = (w_1, \ldots, w_K)$ determines the probability of sampling from each source during training. Finding the optimal mix is a hyperparameter optimization problem with a high-dimensional search space and expensive evaluations.

### Approaches to Mix Optimization

**Scaling law prediction:** Llama 3 ([[llama-3|report]]) trained small proxy models with different data mixes and used scaling laws to predict which mix would produce the best 405B model. This requires hundreds of small-scale training runs but is far cheaper than ablating at the target scale.

**Downstream-task-driven mixing:** Rather than optimizing for training loss (which weights all tokens equally), some approaches optimize the mix to minimize loss on a held-out set of downstream task examples. This can produce very different ratios — dramatically upweighting code and math data, for example, at the expense of general web text.

**Dynamic mixing:** Adjust the mix ratio during training based on monitored metrics. If the model's code performance is lagging, increase the code fraction. This adds engineering complexity but can recover from poor initial mix choices.

### The Code Premium

One consistent finding across labs: code data is disproportionately valuable relative to its volume. Models trained with ~15-20% code in the mix outperform code-free models not just on code benchmarks but on *reasoning and math benchmarks* as well ([[scaling-data-constrained|paper]]). The hypothesis: code is structurally rich (nested logic, type systems, function composition) and provides training signal for logical reasoning that natural language text does not.

This is why Llama 3 devotes ~17% of its 15T-token budget to code and Phi-4 devotes ~20%, despite neither being primarily a code model.

## The Data Quality Hierarchy

A useful mental model for organizing pre-training data by quality tier:

| Tier | Examples | Tokens Available | Information Density |
|------|----------|-----------------|-------------------|
| 1 (highest) | Textbooks, academic papers, curated Q&A | ~50B | Very high |
| 2 | Wikipedia, well-written blogs, documentation | ~200B | High |
| 3 | Filtered web text (quality classifier top 30%) | ~2T | Moderate |
| 4 | Broadly filtered web (heuristic filters pass) | ~10T | Low-moderate |
| 5 (lowest) | Raw Common Crawl, unfiltered | ~250T | Very low |

Most of a large model's training tokens come from Tiers 3-4. Tiers 1-2 are too small to fill a multi-trillion-token budget but are disproportionately valuable per token. This is exactly the tension that motivates OLMo 2's two-stage approach (save Tier 1-2 data for the annealing phase) and Phi-4's synthetic data strategy (use a teacher model to generate Tier 1-quality data synthetically).
