---
chapter: ch-08b
course: llm-training
phase: read
excerpt_of: primary source arXiv:2309.13638v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2309.13638
created_at: "2026-09-15"
---

# Excerpt: Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve

**Paper:** R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, Thomas L. Griffiths (Princeton University). arXiv v1 2023-09-24 (the version read). Source type: paper.

## Approach (§2)
- "Teleological approach": understand a system through the problem it was trained to solve — training task (next-word prediction), training distribution (Internet text), architecture (Transformer) (§2.1).
- Instruction tuning is not analyzed; the authors conjecture next-word prediction plays the larger role because it is the longer phase (§2.1).
- Predictions (§2.2): worse on rare tasks than frequent tasks of equal complexity; sensitive to output probability even for deterministic tasks; sensitive to input probability, less than to output probability.
- Claims are comparative ("X is harder than Y"), not "LLMs can't do X"; basic prompting used (§2.3).

## Mechanism (§3.3-3.5)
- Maximizing P(output|input) is equivalent to maximizing P(input|output) · P(output). For deterministic tasks only one output fits, but an imperfect estimate of P(input|output) leaves several candidates, so P(output) affects predictions. Stated at Marr's computational level, not as a claim that the model computes a likelihood and prior (§3.3, footnote 3).
- Input probability matters when task information about the input is learned from experience with that input; output probability matters whenever the model is uncertain, so it is expected to be more broadly influential (§3.5).

## Setup (§4)
- Eleven tasks (Table 2); 100 high-probability sentences from GlobalVoices; medium-probability sentences made by RoBERTa word replacement of equal length; low-probability sentences by shuffling words except first and last (§4.2).
- Models: gpt-3.5-turbo-0613 and gpt-4-0613 via the OpenAI API, temperature 0.0 (§4.3).
- Spelling test on 1,000 single-token words: 99.8% (GPT-3.5), 99.9% (GPT-4) (§4.1).
- Sentence probabilities estimated with GPT-2 (§6.1).

## Table 1 examples (GPT-4)
- Task frequency: Pig Latin, most common variant 42% vs rare variant 23%.
- Output probability: word-sequence reversal, 97% when the answer is high-probability vs 53% low-probability.
- Input probability: rot-13 encoding, 21% high-probability input vs 11% low-probability input.

## Task probability (§5)
- Shift ciphers, shifts 1-25, decoding, with one example in the prompt: GPT-3.5 0.21 on rot-13 and 0.00 on all other shifts; GPT-4 0.50 or above on the three most common shifts in C4 (1, 3, 13) and below 0.03 on all others; GPT-4 rot-1 0.82, rot-3 0.76, rot-2 0.02 (§5.1, Fig. 5.2).
- Figure 3.1 note: in Internet text rot-13 is about 60 times more common than rot-2.
- rot-13 vs rot-2 decoding difference significant for both models (p < 0.01) (§5.1).
- Across seven common/rare task pairs, both models perform better on the common variant (§5.6).

## Output probability (§6)
- rot-13 decoding: GPT-4 0.51 (high-probability output) vs 0.13 (low-probability output); logistic regression p < 10^−4 for both models (§6.1).
- Targeted one-word changes with Levenshtein distance ≤ 2 (e.g., "come" → "code"): both models produce the regularized high-probability sentence more often than the correct one (§6.1, Fig. 6.2).

## Prompting, scaling, calibration (§9.4)
- Chain-of-thought and step-by-step prompts raise decoding accuracy (and appear to hurt encoding) but keep the shift-level pattern and the output-probability effect.
- GPT-4 outperforms GPT-3.5 on almost all tasks but shows the same qualitative trends.
- GPT-4 (chat interface, 2023-09-20) rated all 15 failed queries (counting 29 letters, low-probability article swapping, rot-10 decoding) as difficulty 1 of 10.

## Beneficial embers (§10.4) and limitations
- Next-word prediction is general because almost any task can be cast as it; Internet text makes few inputs out of distribution (§10.4).
- Limitations: proprietary models with undisclosed architecture and data; simple tasks without substantial practical utility; transfer to practical settings left to future work (Limitations).

## Verification
- Read on 2026-09-15 against arXiv PDF v1 (Abstract, Table 1, §2-§6.1, §9.4, §10.4, §11, Limitations).
