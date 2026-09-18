<!-- excerpt for [[ch-59]] — what "the same number" means in the Tülu 3 report
     source: Tülu 3: Pushing Frontiers in Open Language Model Post-Training, arXiv:2411.15124v5
     read 2026-09-17 from the cached primary text; see [[tulu-3]], [[open-instruct-allenai-recipes-recipe]]
-->

# Tülu 3 — seed spread, template spread, and the published 8B SFT scores

## Table 9 (§4.4): final SFT models against baselines

| Model | Avg. | MMLU | TQA | PopQA | BBH | CHE | CHE+ | GSM | DROP | MATH | IFEval | AE 2 | Safety |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tülu 2 8B SFT | 48.3 | 61.8 | 49.4 | 23.3 | 57.1 | 66.9 | 63.1 | 60.4 | 61.7 | 14.0 | 42.3 | 8.9 | 70.7 |
| Tülu 3 8B SFT | 60.1 | 62.1 | 46.8 | 29.3 | 67.9 | 86.2 | 81.4 | 76.2 | 61.3 | 31.5 | 72.8 | 12.4 | 93.1 |
| Tülu 3 70B SFT | 72.6 | 79.4 | 55.7 | 48.6 | 82.7 | 92.9 | 87.3 | 91.1 | 77.2 | 53.7 | 82.1 | 26.3 | 94.4 |

The 8B row (Avg. 60.1) is the number a reproduction of the Tülu 3 8B SFT stage is compared against.

## Table 14 (§4.4): the same recipe run with different random seeds

| Model | Seed | Average |
|---|---|---|
| Tülu 3 8B SFT | 42 (default) | 59.9 |
| Tülu 3 8B SFT | 123 | 60.1 |
| Tülu 3 8B SFT | 456 | 59.8 |
| Tülu 3 8B SFT | 789 | 59.8 |
| Tülu 3 8B SFT | 1011 | 59.8 |
| Tülu 3 8B best model soup | 42 & 123 | 60.2 |
| Tülu 3 70B SFT | 42 (default) | 71.8 |
| Tülu 3 70B SFT | 123 | 70.0 |
| Tülu 3 70B SFT | 456 | 72.6 |
| Tülu 3 70B best model soup | 123 & 456 | 72.5 |

Caption: "We find that the best random seed is comparable to the best model soup, so for consistency we use
the best single SFT run as our final SFT model." The released 8B number, 60.1, is the highest of five seeds
whose range is 0.3 points. The three 70B seeds span 2.6 points.

## Table 13 (§4.3): chat-template variants on an intermediate SFT mixture (Llama 3.0)

| Chat template | Avg. |
|---|---|
| Tülu (replace `\n` with eos) | 53.0 |
| Zephyr | 52.9 |
| Tülu 3 (no `\n`) | 52.8 |
| Tülu 2 template | 52.6 |
| Llama 3 template | 51.6 |

The report states that replacing the trailing newline with an eos token scored best but was not adopted, "to
avoid generation inconsistency with later steps in our post-training pipeline". The spread from template
choice alone on this intermediate mixture is 1.4 points.
