<!-- scope: rephrase web documents into cleaner synthetic styles for compute- and data-efficient pretraining
     deps: [[fineweb]]
     see-also: [[phi-textbooks]], [[nemotron-4-synthetic]], [[model-collapse]], [[magpie]]
-->

# Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling
- **Core Insight:** Raw web text is noisy and under-structured; re-expressing the same content into cleaner styles can make each token more useful and cut pretraining compute/data needs by a large margin.
- **Guideline:** For noisy web corpora, chunk documents to about 300 tokens, rephrase them with a frozen instruction-tuned model in multiple styles, mix original and synthetic text 1:1, and train the LM on that blend.
- **Authors:** Pratyush Maini, Skyler Seto, Richard Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly
- **Year:** 2024
- **URL:** https://aclanthology.org/2024.acl-long.757/
- **Relevant topics:** synthetic pretraining, paraphrase augmentation, style diversity, data efficiency, web filtering

## Abstract
WRAP treats the web as source material to be rewritten rather than memorized verbatim. The paper uses an off-the-shelf instruction-tuned model to paraphrase web documents into styles such as easy, Wikipedia-like, terse, and question-answer form, then pretrains on a mixture of the original and rephrased text. On noisy C4, this makes pretraining materially more efficient, improves held-out perplexity across the Pile, and improves zero-shot QA accuracy across 13 tasks. The main claim is that synthetic rephrasing adds both style diversity and higher-quality text without requiring a new crawl.

## Key Contributions
- Introduced WRAP, a synthetic-pretraining recipe that rewrites web text into multiple styles before pretraining.
- Showed that the same corpus can support roughly 3x faster pretraining on C4 and substantially better perplexity than raw-text baselines.
- Showed that a 350M model trained on 15% of C4 with WRAP can beat a 1.3B model trained on all of C4.
- Demonstrated that style matters: question-answer rephrasing is especially useful for zero-shot QA, while Wikipedia-like rephrasing helps readability and general pretraining quality.
- Established a practical synthetic-data recipe that later work extended into larger-scale synthetic pretraining systems.

## Key Figures/Tables to Study
- **Figure 1** - the end-to-end WRAP pipeline and the main speed/perplexity curves.
- **Figure 6** - the leakage sanity check via cosine similarity against MRPC-style rephrases.
- **Figure 7** - readability and lexical-diversity differences across styles and corpora.
- **Figure 8** - the syntactic-complexity comparison between synthetic and real C4 text.
- **Table 1** - the main comparison against raw C4 and rephrased variants.

## Technical Details
- **Seed data:** C4 documents are truncated/chunked to nearly 300 tokens with NLTK sentence splitting; the authors note that rephrasing beyond ~300 tokens often causes information loss.
- **Teacher model:** frozen Mistral-7B-Instruct.
- **Styles:** Easy, Medium/Wikipedia-like, Hard/terse, and Q/A.
- **Mixing:** real and synthetic data are sampled 1:1, so each document is seen both as raw text and as a rephrase.
- **Cleanup/filtering:** the core recipe does not use a learned quality filter. Appendix B adds a lightweight post-process that strips boilerplate intros such as "Here's a paraphrase..." or "high-quality English"; the residual error rate after cleanup is reported as under 0.1%.
- **Training setup:** decoder-only transformers at 128M, 350M, and 1.3B parameters; maximum sequence length 1024; 300k training steps for the larger runs; batch size of 1M tokens.
- **Efficiency claims:** the paper reports roughly 3x faster pretraining, about 5x less data for comparable loss, and more than 50% average perplexity reduction across subsets of the Pile in the final ACL version.

## Connections
- Direct ancestor of [[nemotron-4-synthetic]], [[phi-textbooks]], and other synthetic pretraining recipes.
- Orthogonal to [[self-instruct]] and [[magpie]], which target instruction tuning rather than next-token pretraining.
- Pairs naturally with [[fineweb]] and [[dolma]] as a "filter the crawl" complement to "rewrite the crawl."
- Helps frame later questions about style diversity, synthetic-data scale, and whether synthetic-heavy training can preserve generalization.
