---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/whose-language-counts-quality-filter.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2201.10474
created_at: "2026-09-15"
---

# Excerpt: Whose Language Counts as High Quality? Measuring Language Ideologies in Text Data Selection

**Authors:** Suchin Gururangan, Dallas Card, Sarah K. Dreier, Emily K. Gade, Leroy Z. Wang, Zeyu Wang, et al. (University of Washington, University of Michigan, University of New Mexico, Emory University, Allen Institute for AI)
**Version read:** arXiv:2201.10474v2 (26 Jan 2022); v1 January 2022. The arXiv PDF does not state a venue.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v2 PDF text.

## Filter replicated (§3.2)
Binary logistic regression on hashed n-gram features, following the GPT-3 description. Positives: 80M whitespace tokens each of OpenWebText, Wikipedia, Books3. Negatives: 240M tokens of a September 2019 Common Crawl snapshot. 100-trial random hyperparameter search. Test: 90.4% F1, 91.7% accuracy on 60M held-out tokens.

## Data (§3.1)
U.S. SCHOOL NEWS: 910K articles from 1,410 U.S. high schools in 1,329 ZIP codes (552 counties), 2010-2019; used only for evaluation, not released as raw text.

## Results
- Fig. 1: school articles receive lower P(high quality) than general newswire across categories.
- Table 2 (10K opinion articles; coefficients on P(high quality), relative to omitted topic food): Trump/president/election +0.346; game/team/players +0.246; people/women/media +0.197; christmas/dress/holiday −0.056; first or second person pronoun −0.054; log2(number of tokens) +0.088; R² 0.336. Text: a doubling of tokens raises the score by 9 points; first/second person pronouns lower it by 5 points (§3.3).
- Table 3 (968 schools; school-average score): % rural −0.069***; % adults with bachelor's degree +0.059**; log2 median home value +0.010*; log2 number of students +0.006*; is public +0.015*; R² 0.140. Text (§3.4): "a 14 percentage point increase in percent urban population or a 17 percentage point increase in parental education ... correspond to a 1 percentage point increase in average quality score, as does a doubling of home values, or a quadrupling of school size."
- §4.2, Fig. 3: no difference in score distribution between articles from high- and low-factuality news sources (p = 0.085, two-sample Kolmogorov-Smirnov test).
- §4.2, Table 4: TOEFL essay score weakly correlated with filter score (Pearson r = 0.12); the essay prompt is far more predictive (Prompt 4 coefficient 0.6745).
- §4.2, Fig. 4: among Pulitzer-winning books, poetry and drama are less favored than nonfiction and fiction.
- §1: "there is no truly general-purpose corpus."

## How ch-10a uses it
§8 (whose text a quality classifier keeps), Negative samples (classifier negatives and what they encode), Common mistakes.
