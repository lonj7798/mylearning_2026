---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Xiong, Papageorgiou, Lee, Papailiopoulos — "From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data"
source_url: https://arxiv.org/abs/2406.19292
created_at: "2026-09-15"
---

# Excerpt: Artificial Needles — synthetic key-value retrieval fine-tuning and transfer to real tasks

**Paper:** Zheyang Xiong, Vasilis Papageorgiou, Kangwook Lee, Dimitris Papailiopoulos (University of Wisconsin-Madison). arXiv 2406.19292, v1 2024-06, read at v2 (2024-10-14). Code: github.com/edixiong/artificial-needles.

**Why this excerpt exists:** the outline lists `artificial-needles-real-haystacks` as a key source, but no library card with that slug exists at the time of writing. [[needle-in-haystack-data]] and [[in2-film]] cite this paper; the numbers below are read from the paper.

## Data (§2, Figs. 1-3)

- **Simple dictionary key-value retrieval:** a list of dictionaries with integer keys and values; the prompt asks for the value of one key and the dictionary it is in. Example target: "The value of key 2931 is 8364 and it is in Dictionary [32]." (Fig. 1)
- **Multi-subkey retrieval:** each key is a tuple of integers; "Other keys can share some but not all of the subkeys of the gold key", and subkey order in the prompt is randomized (§2, Fig. 2).
- **Answer template:** the prompt includes "Answer in the following template: The value of key 2931 is <fill-in-value> and it is in Dictionary [<fill-in-dictionary-name>]." Fig. 4 shows that with a template the loss on the formatting tokens is low, so training concentrates on the retrieved value (§2).
- The data contains no factual information (§1).

## Training settings (§3.1)

| Model | Task | Samples | Structure | Prompt length | Epochs | Loss |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.1 | simple KV | 350 | 85 dictionaries, 3-4 keys each | about 3,900 tokens | 2 | answer tokens only |
| gpt-3.5-turbo-1106 | multi-subkey KV | 150 | 49 dictionaries | not stated | 3 (OpenAI API) | not stated |

A longer run fine-tunes Mistral-7B-Instruct-v0.2 on simple KV with maximum context 24K and tests 120-document MDQA (§3.5, Fig. 9).

## Transfer results (§3.2)

- MDQA, 20 documents (about 4K tokens), gold document at positions {1, 2, 5, 10, 15, 20}, 200 samples per position: fine-tuning "flattens the U-shaped curve" for GPT-3.5 Turbo and reduces Mistral's primacy bias (Finding 1, Fig. 5). Abstract example: "10.5% improvement on 20 documents MDQA at position 10 for GPT-3.5 Turbo".
- Fine-tuning on MDQA itself with roughly the same number of tokens gives lower MDQA accuracy than the synthetic data (Finding 2, Fig. 5).
- FLenQA (250-3,000 tokens, 2,000 samples per prompting mode): improvement with and without chain-of-thought prompting; templates help (Findings 3-4, Figs. 6-7).

## General-benchmark retention and comparison with other long-context data (§3.3-§3.4, Tables 1-2)

Mistral-7B-Instruct-v0.1, all fine-tuned on roughly the same number of training tokens (Table 2):

| Fine-tuning data | MMLU | HellaSwag | GSM8K | TriviaQA | NQ-Open |
|---|---|---|---|---|---|
| Original | 53.42 | 56.31 | 34.65 | 47.63 | 11.61 |
| Synthetic KV (with template) | 53.44 | 56.22 | 34.34 | 47.74 | 11.98 |
| MultidocQA | 53.19 | 56.27 | 33.28 | 45.20 | 8.69 |
| IN2 | 53.49 | 56.44 | 34.98 | 45.44 | 9.80 |
| Needle-in-a-haystack | 52.83 | 56.22 | 33.79 | 41.30 | 4.88 |

GPT-3.5 Turbo with template: MMLU 68.07 → 67.75 (on 20% of MMLU), GSM8K 72.33 → 71.65 (Table 1).

The authors' explanation (Interpretation): the baselines contain factual information, and fine-tuning on facts can encourage hallucination (Gekhman et al.), which they read from the TriviaQA and NQ-Open drops (Finding 6).

## Limitation stated by the authors (§4, Fig. 10)

> "MDQA benchmark also has another version where distractors are relevant distractors, meaning that they are documents retrieved by a retrieval system ... Models finetuned on our dataset will not improve in this setting" (§4)

The authors propose, as future work, adding the synthetic retrieval data as a small part of a larger instruction set (§4).

## Scope note

All transfer tests are at about 4K tokens except one 24K MDQA run. The paper does not test 32K-128K contexts, RULER, or HELMET.

## Connections

- [[in2-film]] — IN2 is one of the baselines in Table 2.
- [[needle-in-haystack-data]] — the NIAH baseline dataset.
- [[lost-in-the-middle]] — origin of the MDQA position protocol.
- ch-28 §6.
