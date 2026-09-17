---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: none (no library card on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2404.16019
primary_version: arXiv:2404.16019v2 (2024-12-03); v1 2024-04; NeurIPS 2024 Datasets and Benchmarks Track
created_at: "2026-09-15"
---

# Excerpt: The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models

Authors: Hannah Rose Kirk, Alexander Whitefield, Paul Röttger, Andrew Bean, Katerina Margatina, Juan Ciro, et al. (University of Oxford and collaborators). Source type: paper. Read in the v2 PDF text on 2026-09-15 for ch-15 §5 and §6.

## Dataset (Abstract, Fig. 1)
- 1,500 English-speaking participants recruited on Prolific, born in 75 countries (38 countries of residence), including census-representative samples for the UK and US; 1,396 had conversations.
- 8,011 conversations with 21 LLMs from 6 providers (12 commercial API, 9 open access); 27,172 interactions; 68,371 utterances.
- Conversation types: unguided 3,113 (38.9%), controversy guided 2,438 (30.4%), values guided 2,460 (30.7%). Turns per conversation 3.4 ± 1.6.
- Each rating links to a pseudonymous participant ID and a survey of demographics and stated preferences.

## Protocol (§2.2)
- First turn: four models respond; the participant rates each on a cardinal scale. Later turns: two responses sampled from the highest-rated model. The authors chose a cardinal scale because "ratings can be converted to rankings but not vice versa" and it expresses preference intensity; they note it adds intrapersonal measurement noise.

## Case study II: rankings depend on who rates and what they discuss (§3.2)
- Balanced subset: 1,246 participants, 6,669 conversations; aggregation by Pairwise Rank Centrality.
- "Samples of 100 people introduce significant noise"; zephyr-7b ranks high on controversy but not in unguided conversations and claude-2 shows the opposite; relative to the overall rank, palm-2 drops 4 places for US participants, llama-7b drops 7 places in Asia, mistral-7b gains 7 places in Africa.
- Applying the method to Chatbot Arena data, gpt models fare worse in PRISM and zephyr-7b better (95% CI over 1,000 bootstraps).
- OLS on all responses: more characters, ending in a question mark, and enumeration have significant positive effects on score; line breaks negative; de-anthropomorphic phrases reduce score, refusals reduce it more; R² = 0.06.

## Case study III: sampling decisions and welfare (§3.3)
- "as sample size falls, the probability of choosing a LLM with worse mean welfare rises."
- "Sampling exclusively from a specific group tends to reduce the welfare of out-group individuals"; sampling 100 white individuals appears to first-order stochastically dominate 100 representative individuals for the population at large, "but minority stakeholders (non-white population) are worse off under this scheme."
- "No single model achieves majority preference (max 45% MEAN CHOICE)" in the US sample.
