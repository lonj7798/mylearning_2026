<!-- scope: BrowseComp-Plus (Aug 2025) — 830 BrowseComp queries with a fixed 100,195-document corpus of human-verified evidence documents and mined hard negatives, so deep-research agents and retrievers can be evaluated separately and reproducibly
     deps: [[search-r1]]
     see-also: [[context-folding]], [[gemini-2.5-deep-research]], [[tongyi-deepresearch]], [[agentic-benchmark-checklist]], [[fineweb]]
-->

# BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent
- **Core Insight:** On a fixed 100K-document corpus, the same agent's accuracy depends on the retriever: gpt-5 scores 55.90% with BM25 and 70.12% with Qwen3-Embedding-8B while making fewer search calls (23.23 vs 21.74), and SearchR1-32B scores 3.86% with BM25 (Table 1).
- **Guideline:** When comparing deep-research agents or training them against a search tool, hold the corpus and retriever fixed and report the retriever, because on this benchmark replacing BM25 with Qwen3-Embedding-8B raised accuracy for every agent tested (Table 1) and adding FineWeb-Edu documents to the corpus did not change the ranking of retrievers or agents (§4.8.4, Tables 6-7).
- **Authors:** Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, et al. (University of Waterloo, CSIRO, Independent, Carnegie Mellon University, The University of Queensland; 20 authors)
- **Year:** 2025 (arXiv v1 2025-08; technical report, "work in progress"; no later version)
- **URL:** https://arxiv.org/abs/2508.06600 (project page https://texttron.github.io/BrowseComp-Plus/)
- **Source type:** paper
- **Relevant topics:** deep-research agent evaluation, agentic search, retrieval evaluation, hard negatives, reproducible benchmarks, citation accuracy, context engineering, test-time search scaling

## Abstract
Deep-research agents combine LLMs with search tools. Benchmarks such as BrowseComp evaluate them through live web search APIs, which the authors say limits (1) fairness, because dynamic and opaque APIs prevent reproducible comparison, and (2) transparency, because without control of the corpus the retriever's contribution cannot be isolated. BrowseComp-Plus is derived from BrowseComp and uses a fixed, curated corpus. Each query has human-verified supporting documents and mined hard negatives. The benchmark separates systems: Search-R1 with BM25 reaches 3.86% accuracy and GPT-5 reaches 55.9%; GPT-5 with Qwen3-Embedding-8B reaches 70.1% with fewer search calls. The benchmark supports separate analysis of agents, retrieval, citation accuracy, and context engineering.

## Key Contributions
- A fixed corpus of 100,195 documents for 830 BrowseComp queries, built by o3 evidence mining, human verification, and hard-negative mining through web search (§3).
- Human labels for evidence documents and gold documents (documents that contain the final answer, possibly implicitly), released as TREC-style qrels for Recall@k and nDCG@k (§3.2.2, §4.4).
- End-to-end evaluation of 11 LLMs as search agents (7 proprietary, Qwen3-32B, SearchR1-32B, gpt-oss-20B and 120B) paired with BM25, Qwen3-Embedding (0.6B, 4B, 8B), and ReasonIR-8B (§4.1-4.2, Tables 1, 4).
- Ablations: oracle retrieval, reasoning effort, a full-document reader tool, and corpus size (§4.8).

## Key Figures/Tables to Study
- **Fig. 1:** accuracy vs number of search calls per agent and retriever.
- **Table 1:** accuracy, evidence recall, search calls, calibration error; **Table 2:** retriever-only effectiveness.
- **Table 3:** citation coverage, precision, recall; **Table 4:** gpt-oss reasoning effort.
- **Tables 5-7:** get-document tool and FineWeb-augmented corpus; **Table 8 (App. H):** API cost.

## Technical Details
- Construction funnel (§3.2.1-3.2.2): BrowseComp has 1,266 questions; o3 with web search gave no usable evidence for 124; 137 more were dropped because at least one cited URL could not be scraped (Selenium + Trafilatura); 1,005 went to human verification; 830 passed.
- Annotators label the text spans that justify each clue and judge whether the clues and documents answer the whole question; if not, they revise clues and search the web for at least 20 minutes (§3.2.2). 14 university student annotators, over 400 hours; cross-validated samples show over 80% agreement on average (§3.2.2).
- Removed categories (App. C): flawed BrowseComp answers, 42 queries that require Google Maps distances, 13 queries with ambiguous or non-unique answers.
- Hard negatives (§3.3): GPT-4o decomposes each question into about seven self-contained sub-queries; each goes to SerpAPI (Google Search), which returns up to 100 results that are scraped as in positive collection.
- Corpus statistics (§3.4): per query, 6.1 evidence documents, 76.28 negatives, and 2.9 gold documents on average; each document averages 5,179.2 words and 32,296.2 characters.
- Agent setup (§4.3): the BrowseComp prompt asking for an answer and a confidence percentage, revised to require tool use and inline citations of docids (App. E). Search-R1 uses the prompt from its own training. The retriever returns top k = 5 documents, each truncated to its first 512 tokens; with this truncation 86.5% of queries still have the answer inside at least one gold document (Fig. 4b). SearchR1-32B is the 32B checkpoint released in Jin et al. (arXiv:2505.15117) (§4.1).
- Metrics (§4.4): accuracy by a gpt-4.1 judge with the BrowseComp judging prompt (App. F); recall of verified evidence documents retrieved during the run; average search calls; calibration error computed as in Humanity's Last Exam. Search-R1 has no calibration error because its output has no confidence field.
- Table 1 (BM25 → Qwen3-Embed-8B; accuracy, search calls): gpt-4.1 14.58% → 35.42%, 10.35 → 8.67; o3 49.28% → 63.49%, 25.93 → 23.97; gpt-5 55.90% → 70.12%, 23.23 → 21.74; Sonnet 4 14.34% → 36.75%; Opus 4 15.54% → 36.14%; Gemini 2.5 Flash 15.54% → 33.01%; Gemini 2.5 Pro 19.04% → 28.67%; gpt-oss-120B-high 28.67% → 42.89%; Qwen3-32B 3.49% → 10.36% (0.92 → 0.94 calls); SearchR1-32B 3.86% → 10.36% (1.78 → 1.69 calls).
- Table 2 (evidence documents; Recall@5, Recall@100, Recall@1000, nDCG@10): BM25 1.2, 4.7, 13.7, 1.6; Qwen3-Embed-0.6B 6.2, 26.5, 59.7, 8.0; 4B 9.8, 40.2, 71.8, 14.0; 8B 14.5, 47.7, 76.7, 20.3; ReasonIR-8B 12.2, 43.6, 73.9, 16.8. The full question is the query. Table 6 prints BM25 Recall@1000 as 13.6%.
- Citations (Table 3): gpt-5 citation precision 71.8% (BM25) and 83.4% (Qwen3-Embed-8B); Qwen3-32B precision 8.7%-20.0% across the five retrievers.
- Oracle (§4.8.1): given all labeled positive documents, gpt-4.1 scores 93.49% and Qwen3-32B 83.25%; for the remaining gpt-4.1 errors, annotators confirmed the answers are in the documents; 50 Qwen3-32B errors (6%) come from documents exceeding its context window.
- Reasoning effort (Table 4, Qwen3-Embed-8B): gpt-oss-20B low/medium/high accuracy 13.37%/29.88%/34.58%, search calls 1.87/13.64/23.87; gpt-oss-120B 24.94%/37.59%/42.89%, calls 2.21/9.64/18.35.
- Full-document reader (Table 5): gpt-4.1 35.42% → 43.61% with 1.85 get-document calls per query; Qwen3-32B 10.36% → 11.69% with 0.27 calls.
- Corpus size (§4.8.4): adding FineWeb-Edu sample-10BT, deduplicated by URL, gives 9,771,311 documents; the paper calls this roughly 10 times the original, while 9,771,311 / 100,195 = 97.5 (derived). BM25 retrieval metrics rise; neural retriever metrics fall because unjudged added documents count as non-relevant. Qwen3-32B with Qwen3-Embed-8B drops from 10.36% to 7.11% accuracy (Table 7).
- API cost (Table 8): gpt-5 $400.36 (BM25) vs $360.71 (Qwen3-Embed-8B); Opus 4 $2,043.95 vs $1,842.48.

## Findings relevant to generality, negative feedback, long context, and agentic training
- **"Negatives" here are distractor documents in the corpus**, retrieved or not by the search tool. They are not training negatives in the sense of the course standard §6.1.
- **Search behavior of open models.** Qwen3-32B and SearchR1-32B make fewer than 2 search calls per query although prompted to use the tool, while gpt-5 and o3 make more than 20 (§4.6). In the oracle setting Qwen3-32B scores 83.25% vs gpt-4.1's 93.49%, while with Qwen3-Embed-8B search they score 10.36% vs 35.42%; the authors conclude the main limitation is interleaved reasoning with the search tool (§4.8.1, Interpretation).
- **Test-time search scaling.** Higher reasoning effort raises accuracy, recall, and search calls together, and calibration error tends to decrease (§4.8.2, Table 4).
- **Long context.** The 512-token preview is a budget choice; 13.5% of queries lack the answer in the first 512 tokens of every gold document (derived from 86.5%, Fig. 4b). A full-document reader helps gpt-4.1 more than Qwen3-32B (Table 5).
- **Open questions stated for agent training (§5).** How retriever quality affects the learning dynamics of an agent being optimized, and whether an agent optimized with a BM25 tool generalizes when switched to an embedding-based tool. The paper runs no training.
- **Measurement limits.** Accuracy depends on an LLM judge (gpt-4.1). Retrieval metrics treat unjudged documents as non-relevant, so a larger corpus lowers measured recall without implying worse answers (§4.8.4). Contamination of the 830 BrowseComp questions in model training data is not discussed.

## Connections
- [[search-r1]] — the open RL-trained search agent evaluated here (32B checkpoint, 3.86% with BM25).
- [[context-folding]] — splits BrowseComp-Plus into 680 training and 150 evaluation instances with Qwen3-Embed-8B as retriever (arXiv:2510.11967v1 §3.1), which makes this corpus a training environment as well as an evaluation.
- [[tongyi-deepresearch]], [[websailor]], [[asearcher]], [[quest-deep-research]] — deep-research agent training papers in this library; WebSailor is cited in §2.1 as RL training for search on a Qwen base.
- [[gemini-2.5-deep-research]] — Gemini 2.5 report (cited as [7]); Gemini 2.5 Pro and Flash are evaluated here as search agents (Table 1).
- [[agentic-benchmark-checklist]] — agent-benchmark methodology paper in this library; compare its criteria with the verification and removal steps above.
- [[fineweb]] — the FineWeb paper (cited as [41]); its FineWeb-Edu sample-10BT is the corpus-size test (§4.8.4).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2508.06600 (v1, 2025-08-08, the only version).
- Audit claims not found in the source: "the offline training substrate for Context-Folding (680/150 split)" is not in this paper; it was confirmed in the Context-Folding paper (arXiv:2510.11967v1 §3.1) and is kept only in Connections with that locus.
- Not reported by the source: judge agreement with humans, variance across runs or seeds, sampling temperature of the agents, per-query token budgets.
