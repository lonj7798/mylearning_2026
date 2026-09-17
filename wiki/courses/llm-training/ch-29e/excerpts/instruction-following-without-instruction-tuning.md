---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/instruction-following-without-instruction-tuning.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2409.14254
primary_version: arXiv:2409.14254v1 (2024-09)
created_at: "2026-09-15"
---

# Excerpt: Instruction Following without Instruction Tuning

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Authors: John Hewitt, Nelson F. Liu, Christopher D. Manning, Percy Liang. Checked against the v1 PDF on 2026-09-15.

## Objectives (§3 Eq. 1, §4.1 Eq. 2)
- Instruction tuning: min_θ (1/k) Σ_i − log p_θ(response_i | instruction_i).
- Response tuning: min_θ (1/k) Σ_i − log p_θ(response_i | [empty string]).

> "For our adaptation dataset, we use LIMA (Zhou et al., 2023), which has 1,030 training examples."

## Response tuning (Table 1, length-controlled AlpacaEval win rate vs. the instruction-tuned model)
| Model | Tuning | Win rate |
|---|---|---|
| Llama-2-7B | None (Base) | 2.4% ± 0.14% |
| Llama-2-7B | Response Tuning | 43.3% ± 1.1% |
| OLMo-7B-Feb2024 | None (Base) | 4.7% ± 0.57% |
| OLMo-7B-Feb2024 | Response Tuning | 43.7% ± 1.7% |

## Response ranking capability (§4.2 Eq. 3, Table 2)
> "the response ranking capability holds if p_θ(response | instruction) > p_θ(response′ | instruction)."

Table 2: Llama-2-7B base 80.4%, instruction-tuned 77.4%; OLMo-7B-Feb2024 base 74.5%, instruction-tuned 74.3%.

## Single-task fine-tuning (Table 3)
| Tuning | Llama-2-7B | OLMo-7B-Feb2024 |
|---|---|---|
| MBPP | 16.9% ± 0.70% | 10.4% ± 1.0% |
| GSM | 23.7% ± 0.74% | 30.3% ± 0.6% |
| Poetry | 22.9% ± 0.97% | 21.9% ± 0.48% |
| Recipes | 14.6% ± 0.81% | 21.5% ± 0.86% |
| Chess | 2.1% ± 0.36% | 6.3% ± 1.1% |

> "we find that single-task finetuned models only adhere to the finetuning distribution on instructions similar to those finetuned on, and exhibit general instruction following behavior for other instructions." (§1)

## Rule-based adapter (§6 Eq. 4, Table 4)
- p_a(w | x) = p_base(w | x) p_rules(w | x) / Z(x).
- Rules: "Slowly upweight EOS"; "uniformly change the probabilities of 15 words in the vocabulary at every token decision"; "Encourage word diversity".
- Llama-2-7B win rate: all rules 24.4% ± 0.40%; without EOS rule 10.4%; without diversity rule 14.3%; without uniform token changes 16.3%.

## Recommendation (§7)
> "if a practitioner deploys an language model adapted to some specific task, they should not assume that the model will exhibit that tasks' behavior on inputs dissimilar to those trained on. Instead, they should put it through testing and safety trials as if they were releasing a general-purpose chatbot"

## Hyperparameters (App. A)
> "For all Llama-2-7B models, after manual search, we choose 10^−5 as a learning rate, and for all OLMo-7B-Feb2024 models, we choose 3 ∗ 10^−6. We use the Adam optimizer"

> "We use a cosine annealing rate to 0 learning rate, and 10% of the training consists of a linear warmup. [...] We fix a batch size of 64 across all experiments."

Epochs were selected from 5, 7, 10, 15, or 20 (§4.1). Note: the Table 5 caption states 36% for response-tuned OLMo, while Table 1 reports 43.7%; ch-29e uses Table 1.
