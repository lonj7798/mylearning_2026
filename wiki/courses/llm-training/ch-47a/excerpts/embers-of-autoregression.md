---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve (arXiv:2309.13638v1, 2023-09-24)"
source_url: https://arxiv.org/abs/2309.13638
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug embers-of-autoregression). Values read from the v1 PDF on 2026-09-15."
---

# Excerpt: task, output, and input probability as evaluation confounds

**Authors:** R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, Thomas L. Griffiths (Princeton University).

## Claim and setup (Abstract, §2, §4.3)

Three properties of a test item predict accuracy even when the task is deterministic:

1. **Task probability** — how frequently the task variant occurs in text.
2. **Output probability** — how probable the correct answer is as a word sequence.
3. **Input probability** — how probable the given input is; stated to be less influential than output probability.

Models: gpt-3.5-turbo-0613 and gpt-4-0613 through the OpenAI API, temperature 0.0. Eleven tasks. Sentences for the high-probability condition are drawn from GlobalVoices; low-probability conditions are constructed to match in length and tokenization (§4.2, §4.3).

## Reported numbers (Table 1, §5, §6, §7)

| Effect | Comparison | GPT-4 accuracy |
|---|---|---|
| Task probability | Pig Latin, common variant vs rare variant | 42% vs 23% |
| Task probability | Pig Latin vs invented "Boar Etruscan" of equal complexity | 0.39 vs 0.13 (§5.2) |
| Task probability | shift-cipher decoding: rot-1 / rot-3 / rot-2 | 0.82 / 0.76 / 0.02 (§5.1) |
| Task probability | linear function (9/5)x + 32 vs (7/5)x + 31 | 0.33 vs 0.00 (§5.4) |
| Output probability | word-sequence reversal, high- vs low-probability answer | 97% vs 53% |
| Output probability | rot-13 decoding, high- vs low-probability output | 51% vs 13% (Abstract) |
| Input probability | rot-13 encoding, high- vs low-probability input | 21% vs 11% |

Shift ciphers, decoding, all 25 shift levels (§5.1): GPT-3.5 scores 0.21 on rot-13 and 0.00 on every other shift; GPT-4 scores 0.50 or above on the three most frequent shifts (rot-1, rot-3, rot-13) and below 0.03 on all others. The authors note that difficulty cannot explain this, since rot-2 requires less counting than rot-13. Significance for rot-13 vs rot-2 is p < 0.01 for both models on decoding (App. D.2.1).

Sorting (Table 3), with corpus counts from C4:

| Task | Count in C4 | GPT-3.5 | GPT-4 |
|---|---|---|---|
| Alphabetical order | 95,942 | 0.76 | 0.80 |
| Reverse alphabetical order | 629 | 0.15 | 0.32 |
| Ascending order | 21,562 | 0.66 | 0.82 |
| Descending order | 31,378 | 0.58 | 0.80 |

Two orderings with a large frequency difference produce a large accuracy difference; two orderings with similar frequencies produce almost none (§5.5).

Across seven pairs of common and rare task variants, models scored higher on the common variant in every pair (§5.6).

## Limits stated by the authors

- Only two models, both from one provider, both accessed in 2023; the authors write that results may not transfer to other models or other versions (§4.3).
- Corpus frequencies are measured in C4, which is not the training corpus of either model; §5.1 discusses a case (rot-1 and rot-3 beating rot-13 for GPT-4) that C4 frequency does not explain.
- The paper predates chain-of-thought-trained reasoning models; §5.3 notes that chain-of-thought prompting could change some results.

## Used in

ch-47a §3 (probability confounds in perturbed items), §7, Common mistakes.
