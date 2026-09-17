---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reversal-curse.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2309.12288
created_at: "2026-09-15"
---

# Excerpt: The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"

**Authors:** Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, et al. (corresponding author Owain Evans; Vanderbilt University, UK AI Safety Institute, Apollo Research, New York University, University of Sussex, University of Oxford)
**Version read:** arXiv:2309.12288v4 (26 May 2024); v1 September 2023; ICLR 2024.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v4 PDF text.

## Definition (§1, footnote 2)
If a model is trained on "<name> is <description>", then conditioned on the description, its likelihood of the name is not higher than that of a random name: P_LLM(n | d) is not higher than P_LLM(n_r | d).

## Experiment 1: fictitious celebrities (§2.1)
- Data: GPT-4-generated names and descriptions. Subsets NameToDescription, DescriptionToName, and Both (both orders in separate documents, training only). 30 facts per subset, each paraphrased 30 times: 900 documents per subset. Paraphrases keep the original order.
- Fine-tuning: GPT-3 base models via the OpenAI API; sweep on GPT-3-350M over LR multipliers {0.05, 0.1, 0.2, 0.4} and batch sizes {1, 2, 4, 8, 16}, 10 epochs, no loss masking on prompts; best batch 16 and LR multiplier 0.2 used for other sizes (App. B). Llama-7b also swept.
- Results (Table 1, GPT-3-175B, exact match): DescriptionToName same direction 96.7 ± 1.2, reverse 0.1 ± 0.1; NameToDescription same 50.0 ± 2.1, reverse 0.0 ± 0.0.
- Log-probability of the correct name vs a random name shows no difference for GPT-3 350M, 1.3B, 6.7B, 175B (t-tests and Kolmogorov-Smirnov tests; Fig. 4, 30 pairs, 3 seeds per size).
- Ablations: dataset increased from 3,000 to 40,000 documents; prompt tuning on Llama-7b; both fail in reverse (§2.1.2, App. B.7-B.8).

## Experiment 2: real celebrities (§2.2)
Top 1,000 IMDB celebrities; GPT-4 names the parent 79% of the time (1,573 child-parent pairs); given the parent, GPT-4 names the child 33% of the time (10 prompts per question, success if correct at least once). Llama-1 base models are also better at parent than child (Fig. 5). The authors call this a tentative test because training data is unknown.

## Experiment 3: instructions (§2.3)
1,100 question-answer pairs per dataset; Llama-1 7b/13b/30b, 20 epochs, five seeds. Accuracy above 80% for QuestionToAnswer and below 7% for AnswerToQuestion (Fig. 6).

## Scope notes
- In-context "A is B" allows deducing "B is A"; the curse applies to training (§1, App. B.6).
- Related evidence: Allen-Zhu & Li (2023) found the same failure when training from scratch; Grosse et al. (2023) found influence depends on phrase order (§3).
- Tested with fine-tuning rather than pretraining for cost reasons (footnote 6).

## How ch-12a uses it
§6 (reversal curse evidence and conditions), §9 (data design), Generalization lens (b)-(c), Common mistakes.
