---
chapter: ch-32a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2406.14491v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2406.14491
created_at: "2026-09-15"
---

# Excerpt: Instruction Pre-Training: Language Models are Supervised Multitask Learners

**Paper:** Daixuan Cheng, Yuxian Gu, Shaohan Huang, Junyu Bi, Minlie Huang, Furu Wei (Microsoft Research; Tsinghua
University). arXiv v1 2024-06; v2 2024-11-28 read. Source type: paper.

## Method (§2)
- An instruction synthesizer, fine-tuned from Mistral-7B-v0.1, reads a raw text and generates instruction-response
  pairs; tuning data are context-based task datasets reformatted as (raw text, pairs), loss only on the pairs (§2.1, §3.1).
- Multi-round inference: texts and pairs from earlier rounds are prepended, so the pre-training data are M-shot
  examples (Fig. 3). About 5 pairs per raw text, about 52 tokens per pair (§3.1).
- LM pre-training keeps next-token prediction and computes loss on all tokens (§2.2).

## From-scratch results (§3.2, §4.1)
- 200M RefinedWeb texts (~100B tokens); 1/5 of texts (40M) converted in two rounds, giving 200M pairs (~10B tokens);
  synthesizer tuning data (0.2B tokens) repeated 4 times.
- 500M model on 100B tokens reaches Pythia-1B (300B tokens) on the Table 2 average (46.6 vs 47.1); 1.3B reaches
  BLOOM-3B (49.7 vs 50.1). MMLU during instruction tuning rises faster for the Instruct PT model (Fig. 4).

## Domain-adaptive continual pre-training (§3.3, §4.2, Table 3)
- Llama3-8B continued separately on PubMed Abstracts (biomedicine) and financial news, with all raw text converted in
  3 rounds and mixed with general instructions at the mixing ratio of Cheng et al. (2023) (ratio not printed).
- Biomedicine average: Llama3-8B 53.6, Vanilla PT-8B 58.4, Instruct PT-8B 61.3, Llama3-70B 63.9.
- Finance average: Llama3-8B 70.1, Vanilla PT-8B 72.0, Instruct PT-8B 74.7, Llama3-70B 71.9.
- Both PT variants use the same number of tokens. On finance NER Instruct PT is below Vanilla PT; the authors note
  Llama3-70B is below Llama3-8B there and call the benchmark possibly unreliable.
- Ablations (Table 4, domain averages Med./Fin.): w/o corpora 58.6/73.3; rule-based 58.8/73.1; 1-shot 58.5/73.1; ours 61.3/74.7.

## Data quality (§5.2, Table 6; Limitations)
- GPT-4 judged 500 sampled augmented texts: general corpus response accuracy 77.5, context relevance 92.9, 49 task
  categories; biomedicine 86.2/99.4/26; finance 69.8/85.8/41. The Limitations section describes accuracy as
  "approximately 70%", which "may potentially mislead the pre-trained model".

## Hyperparameters (Table 14, continual pre-training column)
Llama3-8B; 4 A100-80GB, 1 day; 4K train steps; batch 0.25M tokens; max sequence 4096; max LR 1e-5; cosine; Adam
(0.9, 0.95); weight decay 0.1; warmup 1000 steps; grad clip 1; dropout 0.1.

## Verification
- Read on 2026-09-15 against arXiv:2406.14491v2 PDF text (Abstract, §1–§7, Limitations, Tables 1–6, 14–15).
- Not reported: general-benchmark scores of the domain-adapted Llama3-8B models (no retention measurement); the
  general-instruction mixing ratio as a number; total continual pre-training tokens (4K steps × 0.25M tokens = 1B is derived).
