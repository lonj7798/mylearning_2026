---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2201.10474 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2201.10474
created_at: "2026-09-15"
revised: 2026-09 (generality revision)
---

# Excerpt: Whose Language Counts as High Quality? Measuring Language Ideologies in Text Data Selection

- **Authors:** Suchin Gururangan, Dallas Card, Sarah K. Dreier, Emily K. Gade, Leroy Z. Wang, Zeyu Wang, et al.
- **Year:** 2022 (arXiv v1 2022-01; read: arXiv v2, 26 Jan 2022)
- **Source type:** paper
- **Checked on:** 2026-09-15 against the arXiv v2 PDF. The planned library card `papers/whose-language-counts-quality-filter.md` did not exist when ch-10 was revised; this excerpt holds only what ch-10 cites, with loci.

## Setting
- **Filter** (§3.2): a re-implementation of the GPT-3 quality filter as a binary logistic regression on hashed n-gram features, trained with 80M whitespace tokens each of OpenWebText, Wikipedia, and Books3 as the positive class and 240M tokens of a September 2019 Common Crawl snapshot (downloaded with CCNet code) as the negative class. Held-out result: 90.4% F1, 91.7% accuracy on 60M test tokens. The document score is written P(high quality).
- **Data** (§3.1): U.S. SCHOOL NEWS, 910K articles from 1,410 high schools in 1,329 ZIP codes, 2010–2019; used only for evaluation.

## Results used in ch-10
- **Document level** (§3.3, Table 2, 10K opinion pieces): articles about the presidential election score 35 percentage points higher and sports articles 25 points higher than the omitted food topic; first- or second-person pronouns lower the score by 5 points; doubling document length raises it by 9 points.
- **Demographics** (§3.4, Figure 2, Table 3, 968 schools): correlations of a school's average score with median home value r = 0.27, share of adults with a bachelor's degree r = 0.30, 2016 GOP vote share r = −0.33, rural share r = −0.30. In the regression, rural share (−0.069), bachelor's share (0.059), log home value (0.010), log school size (0.006), and public-school status (0.015) are significant; R² = 0.140. The authors describe individual effects as "relatively modest".
- **Other notions of quality** (§4.2): no difference in score distribution between articles from high- and low-factuality news sources (9.9K and 7.7K articles; two-sample Kolmogorov–Smirnov p = 0.085). For 12.1K TOEFL essays, the essay prompts have larger effect sizes on the score than the essay's official grade (Table 4).
- **Authors' conclusion** (Abstract; §1): "privileging any corpus as high quality entails a language ideology"; "there is no truly general-purpose corpus" (Interpretation).

## Limits stated by the authors (§3.1, §3.4)
- The newspaper corpus is not a random or representative sample of U.S. school newspapers.
- The demographic model explains a small share of variance (R² = 0.140), and findings may not transfer to other domains such as social media.

## Connections
- [[ccnet]] — CCNet also scores text by similarity to a reference corpus (Wikipedia 5-gram perplexity) and reports valid spoken-like text in its tail bucket (§5.2).
- [[dolma]] — its §4.1 footnote 4 cites this paper for the statement that "quality" filters select text by criteria that are "inherently ideological".
- ch-10a covers learned quality classifiers.
