<!-- scope: MMLU-Pro (arXiv:2406.01574, Jun 2024; NeurIPS 2024 Datasets and Benchmarks) — a ten-option, reasoning-weighted rebuild of MMLU with its construction pipeline, 5-shot CoT protocol, answer-extraction rules, and prompt-sensitivity comparison
     see-also: [[ifeval]], [[signal-and-noise-eval]], [[gsm1k]]
-->

# MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
- **Core Insight:** Rebuilding MMLU with reasoning-weighted questions and ten answer options instead of four lowers accuracy by 16% to 33% and narrows the score range across 24 reasonable prompts from 4-5% on MMLU (peak 10.98%) to about 2% on MMLU-Pro (maximum 3.74%) (Abstract; §6.3).
- **Guideline:** When a knowledge-and-reasoning benchmark is used to compare checkpoints, prefer a set whose prompt-induced range is measured and small, and hold the prompting mode fixed, because chain-of-thought instead of direct answering moves MMLU-Pro by up to 19.1 points for one model while moving MMLU by 1.5 (§6.2, Table 3). Otherwise treat differences below the benchmark's own prompt range as unresolved.
- **Authors:** Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, et al. (University of Waterloo, University of Toronto, Carnegie Mellon University)
- **Year:** 2024 (arXiv v1 2024-06-03; v6 2024-11-06; NeurIPS 2024 Track on Datasets and Benchmarks)
- **URL:** https://arxiv.org/abs/2406.01574
- **Source type:** paper
- **Relevant topics:** evaluation benchmarks, benchmark saturation, prompt sensitivity, chain-of-thought evaluation, held-out evaluation suites, answer extraction

## Abstract
As models improve, MMLU scores plateau and stop separating models. MMLU-Pro extends the knowledge-driven MMLU with more challenging, reasoning-focused questions, expands the choice set from four to ten options, and removes trivial and noisy questions. Accuracy falls by 16% to 33% relative to MMLU. Across 24 prompt styles the sensitivity of scores to prompt variation falls from 4-5% on MMLU to about 2% on MMLU-Pro. Chain-of-thought reasoning improves results on MMLU-Pro, in contrast to MMLU, which the authors read as evidence that MMLU-Pro contains more complex reasoning questions.

## Key Contributions
- A 12,032-question, 14-discipline benchmark built from four sources with a documented filtering, option-augmentation, and expert-review pipeline (§3.1, §3.2).
- Ten-option items, which lower the value of guessing and raise discrimination between models (§3.2, §6.1).
- A fixed 5-shot chain-of-thought protocol with a two-regex answer extractor and a defined fallback (§4).
- Measurements of three properties against MMLU: difficulty, reasoning dependence, and prompt robustness (§6.1–§6.3).
- An error analysis of the strongest evaluated model on 120 sampled errors (§5.3).

## Key Figures/Tables to Study
- Fig. 2 (construction pipeline); Fig. 3 (discipline and source distributions); Table 1 (issues found in expert review); Table 2 (model accuracies with 5-shot CoT); Fig. 4 (MMLU vs MMLU-Pro scores); Table 3 (CoT minus direct answering on both benchmarks); Fig. 5 (score range under 24 prompts).

## Technical Details
**Composition (§3.1)**
- 12,032 questions across 14 discipline subsets. Sources: original MMLU questions with trivial and erroneous items removed (56.6% of the set), a STEM website (33.9%), TheoremQA (4.9%), and SciBench (4.5%) (Fig. 3b).
- Discipline shares from Fig. 3a include Math 11.2%, Physics 10.8%, Chemistry 9.41%, Law 9.15%, Engineering 8.05%.

**Construction pipeline (§3.2)**
1. Initial filtering: MMLU's 57 subject categories are merged into 14; eight models (Llama-2-7B, Llama-2-7B-Chat, Llama-2-13B, Llama-2-13B-Chat, Mistral-7B, Gemma-7B, Yi-6B, Yi-6B-Chat) are evaluated on MMLU, and a question answered correctly by more than four of them is treated as too easy. 5,886 questions are removed this way.
2. Question collection: items from the STEM website, TheoremQA, and SciBench are converted to multiple choice using GPT-4-Turbo (gpt-4-turbo-2024-04-09) to extract short answers and generate three distractors, with manual comparison against the original solutions.
3. Option augmentation: GPT-4-Turbo adds six further distractors, taking items from four to ten options. The authors report that GPT-4-Turbo gains no additional advantage from this procedure.
4. Expert review: phase 1 verifies answer correctness and removes items unsuitable for multiple choice or requiring images or tables; phase 2 uses Gemini-1.5-Pro to flag false-negative options, followed by human review. Issue counts by source (Table 1): incorrect answers 350 (MMLU), 0 (TheoremQA), 11 (SciBench), 483 (STEM website); false-negative options 1,953 / 5 / 15 / 293; bad questions 385 / 1 / 15 / 862.
- After review, 83% of questions have ten options, 17% have fewer, and the average is 9.47 options per question (§3.2).

**Evaluation protocol (§4)**
- 5-shot chain-of-thought prompting adapted from Chain-of-Thought Hub, with five demonstration examples selected per discipline.
- Answer extraction applies the regex `answer is \(?\([A-J]\)?\)`; on failure a second regex `\.*\[aA\]nswer:\s*\([A-J]\)`; if both fail, a random option is selected so that every item receives an answer.

**Results (§5, §6)**
- Table 2 (5-shot CoT, percentages): GPT-4o 72.6 overall, Gemini-1.5-Pro 69.0, Claude-3-Opus 68.5, GPT-4-Turbo 63.7, Gemini-1.5-Flash 59.1 (0-shot), Yi-large 58.1, Claude-3-Sonnet 56.8; best open model Llama-3-70B-Instruct 56.2, then Phi-3-medium-4k-instruct 55.7, DeepSeek-V2-Chat 54.8, Phi-3-mini-4k-instruct 45.7.
- Discrimination (§6.1): four models cluster between 78% and 82% on MMLU, a 4% range, which widens to about 10% on MMLU-Pro; the GPT-4o / Claude-3-Opus / GPT-4-Turbo spread widens from about 2% on MMLU to about 9% on MMLU-Pro. GPT-4o's 72.6% leaves 27.4 points of headroom against about 11.3 on MMLU.
- Reasoning dependence (§6.2, Table 3, CoT vs direct answering): GPT-4o 1.5 on MMLU against 19.1 on MMLU-Pro; GPT-4-Turbo −0.2 against 15.3; Phi3-medium-4k-instruct 1.4 against 8.2; Llama-3-8B −3.9 against 3.9; Gemma-7B −3.6 against 6.7. 19.1 is the largest MMLU-Pro value in Table 3.
- Prompt robustness (§6.3, Fig. 5): under 24 different but reasonable prompts, the influence on MMLU scores is generally 4-5% with peaks up to 10.98%, and on MMLU-Pro generally around 2% with a maximum of 3.74%.
- Error analysis (§5.3): of 120 randomly sampled GPT-4o errors reviewed by expert annotators, 39% are reasoning errors, 35% lack of specific knowledge, 12% calculation errors, 5% no selection made, 4% question-understanding errors, 2% generation issues, 2% annotation errors, 1% answer-extraction errors.
- Subject differences (§5.2): Engineering and Law score lowest of the 14 subjects; the Engineering items added from the STEM website require multi-step derivations.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MMLU-Pro evaluations, Table 2 | closed and open models | eval-gate | prompting | 5-shot chain of thought adapted from Chain-of-Thought Hub, five demonstrations per discipline; Gemini-1.5-Pro and Gemini-1.5-Flash at 0 shots | arXiv:2406.01574v6 §4, Table 2 caption | verified 2026-09-18 | §6.2, Table 3: CoT exceeds direct answering on MMLU-Pro for all five models compared |
| MMLU-Pro set | n/a | eval-gate | items; options | 12,032 questions in 14 disciplines; 83% with ten options, 17% fewer, mean 9.47 | §3.1, §3.2 | verified 2026-09-18 | §6.1: model separation widens from about 4% to about 10% against MMLU |
| MMLU-Pro set | n/a | eval-gate | filtering rule | an MMLU item answered correctly by more than four of eight named small models is dropped as too easy; 5,886 removed | §3.2 | verified 2026-09-18 | §6.1: reduced clustering of scores at the top |
| MMLU-Pro evaluations | n/a | eval-gate | answer extraction | primary regex, then a secondary regex, then a random option as fallback | §4 | verified 2026-09-18 | §5.3: answer-extraction errors are 1% of 120 reviewed GPT-4o errors |
| MMLU-Pro robustness study | n/a | eval-gate | prompt variants | 24 different but reasonable prompts per model | §6.3, Fig. 5 | verified 2026-09-18 | §6.3: range about 2% (maximum 3.74%) against 4-5% (peak 10.98%) on MMLU |

## Findings relevant to generality
- A benchmark's own prompt-induced range is the floor for reading a difference between checkpoints: on MMLU-Pro that floor is about 2 points, and about 4-5 points on MMLU (§6.3).
- Prompting mode is part of the measurement, not a free choice: the same model moves up to 19.1 points between CoT and direct answering on MMLU-Pro (§6.2, Table 3).
- The random-option fallback means that a model whose output format breaks scores at chance rather than zero, so a formatting regression appears as a capability drop (§4).
- Limits (§7): the multiple-choice format does not capture open-ended generation, and the benchmark covers text-only models.

## Connections
- [[ifeval]] — the instruction-following half of a held-out retention suite alongside MMLU-Pro.
- [[signal-and-noise-eval]] — treats benchmark noise as the quantity that decides whether a benchmark can separate checkpoints.
- [[gsm1k]] — a different response to benchmark saturation: new items rather than harder items and more options.

## Verification
- Created on 2026-09-18 from https://arxiv.org/abs/2406.01574 (arXiv v6, 2024-11-06).
- Corrections to the previous card version: none (no card existed; chapters cited `[[mmlu-pro]]` with excerpt files).
- Removed as unsupported by the source: none.
- Chapter claims not found in the source: none. Two notes on locus. (1) ch-53 cites "12,032 items with up to ten answer options (83% have ten; mean 9.47)" as §3.1, §3.2 — the count is in §3.1 and the option statistics are in the Expert Review part of §3.2, as recorded above. (2) ch-46 states that "CoT alone moves MMLU-Pro by up to 19.1 points (§6.2)"; §6.2 reports 19.1 for GPT-4o specifically, and it is the largest MMLU-Pro CoT-minus-direct value in Table 3, which lists five models.
- Not reported by the source: per-discipline item counts in the body (Fig. 3a gives shares); the identity of the 24 prompt templates; results for models released after the paper.
