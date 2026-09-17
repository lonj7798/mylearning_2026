---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dclm.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2406.11794
primary_version: arXiv:2406.11794v4 (2025-04-21; v1 2024-06)
created_at: "2026-09-15"
---

# Excerpt: DataComp-LM: In search of the next generation of training sets for language models

Verbatim quotations and table values used by ch-09 `read.md`, with loci. Authors: Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, et al. (60 authors). Checked against the v4 PDF on 2026-09-15.

## Pool and benchmark (Abstract, §3.1)
> "we provide a standardized corpus of 240T tokens extracted from Common Crawl, effective pretraining recipes based on the OpenLM framework, and a broad suite of 53 downstream evaluations."

> "DCLM-POOL is an unfiltered web-text corpus comprised of all Common Crawl data prior to 2023. [...] DCLM-POOL contains 200B documents (370TB after gzip compression), resulting in 240T GPT-NeoX tokens."

## Scales (§3.2, Table 1)
Model parameters / train tokens / train FLOPs / H100 hours / pool size: 400M-1x 412M / 8.2B / 2.0e19 / 26 / 469B; 1B-1x 1.4B / 28.8B / 2.4e20 / 240 / 1.64T; 3B-1x 2.8B / 55.9B / 9.4e20 / 740 / 3.18T; 7B-1x 6.9B / 138B / 5.7e21 / 3,700 / 7.85T; 7B-2x 6.9B / 276B / 1.1e22 / 7,300 / 15.7T.

> "We find high rank correlation between the results for smaller scales (400M-1x, 1B-1x, 3B-1x) and those for the larger 7B-1x scale (Pearson's r = 0.838, r = 0.956, r = 0.982 respectively)"

## Metrics (§3.5, App. G.1)
> "we propose the CORE centered accuracy, computed over a subset of 22 tasks (e.g., HellaSwag and ARC-E) that provide a low-variance signal even at small scales, linearly rescaling the accuracy per task so that 0 corresponds to random guessing and 1 corresponds to perfect accuracy. Finally, we report the EXTENDED centered accuracy, which averages the centered performance for all of our 53 tasks."

Core tasks listed in App. G.1 include AGI Eval LSAT-AR, ARC easy and challenge, six BIG-bench tasks (QA Wikidata, Dyck languages, Operators, Repeat Copy Logic, CS Algorithms, Language Identification), BoolQ, CommonsenseQA, COPA, CoQA, and HellaSwag.

## Filtering (§4.4, Table 4 at 1B-1x, Table 5 at 7B-1x)
Table 4 (Core / Extended): RefinedWeb reproduction 27.5 / 14.6; Top 20% by Pagerank 26.1 / 12.9; SemDedup 27.1 / 13.8; Classifier on BGE features 27.2 / 14.0; AskLLM 28.6 / 14.3; Perplexity filtering 29.0 / 15.0; Top-k average logits 29.2 / 14.7; fastText OH-2.5 + ELI5 30.2 / 15.4.

Table 5 (positive set, threshold: Core / MMLU / Extended): OH-2.5 + ELI5, 10%: 41.0 / 29.2 / 21.4; Wikipedia, 10%: 35.7 / 27.0 / 19.1; OpenWebText2, 10%: 34.7 / 25.0 / 18.7; GPT-3 Approx, 10%: 37.5 / 24.4 / 20.0; OH-2.5 + ELI5, 15%: 39.8 / 27.2 / 21.5; OH-2.5 + ELI5, 20%: 38.7 / 24.2 / 20.3.

> "For DCLM-BASELINE and the remaining experiments, we use fastText OH-2.5 + ELI5 classifier score to keep the top 10% of documents."

## Mixing (§4.5, Table 6 at 1B-1x)
> "We compare a model trained on 100% filtered CC data to models trained with the mixing ratios from Llama 1 and RedPajama: 67% CC, and 33% from Wikipedia, Books, Stack exchange, arXiv, and Github."

| Dataset | Core base | Core w/ RPJ extras | Extended base | Extended w/ RPJ extras |
|---|---|---|---|---|
| C4 | 23.7 | 25.9 (+2.2) | 12.5 | 13.3 (+0.8) |
| RPJ CC only | 24.0 | 25.7 (+1.7) | 12.1 | 13.5 (+1.4) |
| RefinedWeb | 25.1 | 26.5 (+1.4) | 12.9 | 13.1 (+0.2) |
| DCLM-BASELINE | 31.1 | 29.9 (−1.2) | 16.0 | 15.0 (−1.0) |

> "In the case of DCLM-BASELINE however, mixing actually hurts performance on average, which suggests it can be counterproductive given performant filtering."

## Decontamination (§4.6, Table 7 at 7B-2x)
MMLU: DCLM-BASELINE 51.8, with detected overlaps removed 52.7. HellaSwag: 77.9 and 78.4.

## Final model (§5, Table 8)
> "we combine our 3.8T DCLM-BASELINE with the StarCoder and ProofPile2 datasets to arrive at a 4.1T token dataset. We train a 7B model for 2.5T tokens on this dataset with the same hyperparameters as our largest competition scale except for two separate cool-downs phase for the 200B and 270B tokens on a modified distribution that was 70% DCLM-BASELINE with a tighter fastText threshold, and 30% math datasets (see Appendix Q). We then take a "model soup" of these two separate cool-downs. Finally, we adopt the continual pretraining methodology from Pouransari et al. for 100B tokens on the same distribution to increase the context length from 2048 to 8192"

Table 8 row "DCLM-BASELINE + StarCoder + ProofPile2", 7B, 2.6T tokens: Core 57.1, MMLU 63.7, Extended 45.4.

## Limitations (§6)
> "Due to compute constraints, we could only ablate design dimensions individually and could not test all approaches at larger scales nor train models beyond 7B parameters. We also could not sufficiently explore run-to-run variation. [...] We also conducted most of our experiments with only one tokenizer (GPT-NeoX)"
