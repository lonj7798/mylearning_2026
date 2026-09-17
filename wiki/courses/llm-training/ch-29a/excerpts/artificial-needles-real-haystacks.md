---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2406.19292v2 (From Artificial Needles to Real Haystacks), §2-4, Tables 1-2, Figure 10, App. A.1 (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2406.19292
created_at: "2026-09-15"
---

# Excerpt: Artificial Needles — fine-tuning on numeric key-value retrieval

- **Authors:** Zheyang Xiong, Vasilis Papageorgiou, Kangwook Lee, Dimitris Papailiopoulos (University of Wisconsin-Madison)
- **Year:** 2024 (arXiv v1 2024-06; v2 2024-10 used here)
- **Source type:** paper
- **Used in:** ch-29a §5, Negative samples, Generalization lens

## Data (§2, Figures 1-3, App. A.1)
- Simple task prompt (Fig. 1): "Dictionary [1] {122: 765, 4548: 1475, 4818: 4782} ... Report the value of key 2931 and the dictionary it is in."
  Desired answer: "The value of key 2931 is 8364 and it is in Dictionary [32]."
- Mistral 7B data: 350 tasks per dataset, 85 dictionaries per task, 3 to 4 keys per dictionary, keys and values integers of 3 to 4 digits,
  about 3,900 tokens per prompt; loss on the answer only; 2 epochs; LR 5×10^-6; global batch 16; 3 datasets with different seeds, results averaged.
- Multi-subkey task (Fig. 2) for GPT-3.5 Turbo: keys are integer tuples, other keys share some subkeys, subkey order is shuffled;
  150 tasks, 49 dictionaries. §3.1 says 3 epochs; App. A.1 says 2 epochs.
- Answer template (Fig. 3-4): the prompt fixes the answer format so that "the loss on the formatting part is small".

## General-benchmark effects (Table 2, Mistral-7B-Instruct-v0.1, roughly equal training tokens per dataset)
| Fine-tuning data | MMLU | HellaSwag | GSM8K | TriviaQA | NQ-Open |
|---|---|---|---|---|---|
| Original | 53.42 | 56.31 | 34.65 | 47.63 | 11.61 |
| Key-value (w/ template) | 53.44 (+0.02) | 56.22 (−0.09) | 34.34 (−0.31) | 47.74 (+0.11) | 11.98 (+0.37) |
| MultidocQA | 53.19 (−0.22) | 56.27 (−0.04) | 33.28 (−1.36) | 45.20 (−2.43) | 8.69 (−2.91) |
| IN2 | 53.49 (+0.07) | 56.44 (+0.13) | 34.98 (+0.32) | 45.44 (−2.19) | 9.80 (−1.81) |
| Needle-in-a-haystack | 52.83 (−0.59) | 56.22 (−0.09) | 33.79 (−0.86) | 41.30 (−6.33) | 4.88 (−6.73) |

- The abstract states the baseline TriviaQA drops as "from 2.33% to 6.19%"; Table 2 prints −2.19 to −6.33 points.
- Authors' explanation: "all other baselines contain factual information", and fine-tuning on factual information "encourages
  hallucinations" (citing Gekhman et al.). This is an Interpretation; the paper does not measure hallucination directly.
- §3.4 with Fig. 8: "some baselines outperform our proposed data on either MDQA or FLenQA".

## Transfer and limit
- GPT-3.5 Turbo: "10.5% improvement on 20 documents MDQA at position 10" (abstract).
- §4 and Fig. 10: on MDQA with relevant distractors (retrieved documents without the answer), "Models finetuned on our dataset will not
  improve in this setting".
