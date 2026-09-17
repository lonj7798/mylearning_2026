---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: no library card (not in the library on 2026-09-15); quotations below are taken from the primary source
source_url: https://arxiv.org/abs/2407.14933
primary_version: arXiv:2407.14933v2 (2024-07-24)
created_at: "2026-09-15"
---

# Excerpt: Consent in Crisis: The Rapid Decline of the AI Data Commons

Verbatim quotations used by ch-09 `read.md`, with loci. Authors: Shayne Longpre, Robert Mahari, Ariel Lee, Campbell Lund, Hamidah Oderinwale, William Brannon, et al. (Data Provenance Initiative). Checked against the v2 PDF on 2026-09-15.

## Abstract
> "Our audit of 14,000 web domains provides an expansive view of crawlable web data and how codified data use preferences are changing over time. [...] in a single year (2023-2024) there has been a rapid crescendo of data restrictions from web sources, rendering ~5%+ of all tokens in C4, or 28%+ of the most actively maintained, critical sources in C4, fully restricted from use. For Terms of Service crawling restrictions, a full 45% of C4 is now restricted."

## Method (§2)
> "For each data source, we identified and selected the top 2k web domains ranked by their number of tokens. We refer to the resulting 3.95k union of these web domains as HEAD_All. [...] we randomly sampled 10K domains (RANDOM10k) from the intersection of the three corpora"

> "We used the Wayback Machine [...] to collect historical versions of each website's homepage, its Robots Exclusion Protocol (REP), commonly referred to as robots.txt file, and its terms of service page. This was collected at monthly intervals, from January 2016 to April 2024."

> "we record the robots.txt instructions for a range of crawlers, but focus our analysis on five AI developers, Google, OpenAI, Anthropic, Cohere, and Meta, as well as non-profit web archival organizations such as Common Crawl and the Internet Archive [...] For each corpus, we measure the percentage of "restricted tokens" as the portion of tokens from web domains that fully restrict one or more of the AI Organizations's crawlers."

## Findings (§3.1)
> "We make no assertion regarding whether the prior omission of a robots.txt or restrictions implies consent to use data."

> "Across the entire corpora, ~1% of C4, RefinedWeb, and Dolma tokens were restricted in mid 2023, as compared to 5-7% of tokens in April 2024. Among the most critical domains (HEAD_All), 20-33% of all tokens are restricted, as compared to <3% one year prior (Figure 2a)."

> "Note that these measurements only capture full restricted domains, and the numbers are higher for partially restricted domains."

> "OpenAI crawlers are restricted for 25.9% of tokens in HEAD_C4, followed by Anthropic and Common Crawl (13.3%), Google's AI crawler (9.8%), and more distantly Cohere (4.9%), Meta (4.1%), the Internet Archive (3.2%), and lastly Google Search's crawler (1.0%)."

> "Figure 2c shows 45-55% of all tokens in these three corpora have a form of data use restriction in their Terms pages. In practice, most automatic crawlers do not heed these Terms"

> "For robots.txt, Figure 2b shows nearly 45% of all News website tokens are fully restricted in HEAD_All, as compared to 3% in 2023. [...] this suggests that the composition of tokens in crawls respecting robots.txt may shift away from news, social media, and forums, and towards organization and e-commerce websites."

## Use mismatch (§3.4)
> "We randomly sampled 100 conversation logs from WildChat, which the paper authors manually clustered [...] Subsequently, we used GPT-4o to label 1k randomly selected conversations from the WildChat dataset"

> "in over 30% of conversations, users request creative compositions such as fictional story writing or continuation, role-playing, or poetry. However, creative writing is poorly represented among the web data used for model training."

> "The WildChat dataset may not include a representative sample of how people interact with language models."
