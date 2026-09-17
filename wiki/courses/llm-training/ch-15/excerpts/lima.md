---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lima.md (library card not verified as of 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2305.11206
primary_version: arXiv:2305.11206v1 (2023-05-18); NeurIPS 2023
created_at: "2026-09-15"
---

# Excerpt: LIMA: Less Is More for Alignment, data and ablation sections

Authors: Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, et al. (Meta AI and collaborators). Source type: paper. Read in the v1 PDF text on 2026-09-15 for ch-15 §4 and Recipe.

## Training set (Table 1)
| Source | #Examples | Avg input len. | Avg output len. |
|---|---|---|---|
| Stack Exchange (STEM) | 200 | 117 | 523 |
| Stack Exchange (Other) | 200 | 119 | 530 |
| wikiHow | 200 | 12 | 1,811 |
| Pushshift r/WritingPrompts | 150 | 34 | 274 |
| Natural Instructions | 50 | 236 | 92 |
| Paper Authors (Group A) | 200 | 40 | 334 |
Dev: 50 Group A prompts. Test: 70 r/AskReddit and 230 Group B prompts. "The total amount of training data is roughly 750,000 tokens, split over exactly 1,000 sequences."
- wikiHow: one of 19 categories is sampled first, then an article, "to ensure diversity" (§2.1).
- 13 training examples are safety-related (§4.3).

## Training (§3)
LLaMa 65B; 15 epochs; AdamW β1 = 0.9, β2 = 0.95, weight decay 0.1; no warmup; LR 1e-5 linearly decaying to 1e-6; batch 32 examples (64 for smaller models); texts over 2048 tokens trimmed; residual dropout rising from 0.0 to 0.3 (0.2 for smaller models); checkpoints selected manually between epochs 5 and 10 on the 50-example dev set, because "perplexity does not correlate with generation quality".

## Human evaluation and agreement (§4.1-§4.3)
- Tie-discounted agreement on 50 shared examples: crowd-crowd 82%, crowd-author 81%, author-author 78%; crowd-GPT 78%, author-GPT 79%.
- LIMA's response was at least as good as Bard's 58% of the time; GPT-4 preferred LIMA over its own output 19% of the time (§4.2).
- 50 test examples: 50% excellent, 38% pass, 12% fail (Fig. 3). 20 examples analyzed as out-of-distribution in format: 20% fail, 35% pass, 45% excellent (§4.3).
- 30 potentially sensitive test prompts: safe responses to 80%, including 6 of 10 with malicious intent (§4.3).

## Ablations on 7B (§5; ChatGPT grades helpfulness 1-6; 5 samples per test prompt)
- 2,000 examples each (Fig. 5): filtered Stack Exchange 3.83, wikiHow 3.49, unfiltered Stack Exchange 3.33. Stack Exchange (heterogeneous prompts) vs wikiHow ("all of its prompts are 'how to' questions") is the diversity comparison; the authors note other confounding factors between the two sources. Filtered vs unfiltered Stack Exchange is the quality comparison ("a significant 0.5 point difference").
- Quantity (Fig. 6): quality-filtered Stack Exchange sets of 2K to 32K examples; "Despite an up to 16-fold increase in data size, performance as measured by ChatGPT plateaus."
- Footnote 5: at least 2,000 examples improved stability for the 7B model.

## Multi-turn (§6)
Adding 30 hand-crafted dialogue chains improved multi-turn dialogue (Abstract, §6).
