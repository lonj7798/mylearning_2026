---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/smollm2.md (planned card; not present on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2502.02737
primary_version: arXiv:2502.02737v1 (2025-02-04)
created_at: "2026-09-15"
---

# Excerpt: SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model (stage mixtures)

Authors: Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. (Hugging Face). Source type: official technical report. Read in the v1 PDF on 2026-09-15 (text and rendered Figure 2 on page 5) for ch-13 `read.md` §1, §4, §5, §6, and Recipe. A separate ch-08a excerpt covers token budgets and schedule.

## Approach (§4)
> "we trained on 11 trillion tokens (approximately two epochs on our collected datasets), employing a multi-stage training approach instead of a fixed dataset mixture throughout pretraining."

Principles: "(1) Performance-driven interventions, where we monitor evaluation metrics on key benchmarks and adapt dataset mixtures to address specific capability bottlenecks; (2) Upsampling high-quality math and code during the annealing phase ...; (3) Strategic introduction of medium-sized datasets ... mid-training to avoid dilution by larger datasets early on; and (4) Avoiding excessive data repetition, in line with Muennighoff et al. (2023) we aimed to stay close to the recommended 4–5 epoch threshold for most datasets." The approach is described as "online" because pretraining cost "around $250,000 USD of GPU compute".

## Stage mixtures (Figure 2 and §4.2-§4.5)
| Stage | Tokens | English web | Code | Math | Textbooks |
|---|---|---|---|---|---|
| 1 (stable) | 0-6T | 90% (FineWeb-Edu 60 / DCLM 40) | 10% StarCoderData | 0% | — |
| 2 (stable) | 6-8T | 75% (60 / 40) | 20% | 5% OWM | — |
| 3 (stable) | 8-10T | 74% (FineWeb-Edu 40 / DCLM 60) | 16% Stack-Edu | 10% (OWM + InfiMM-WebMath) | — |
| 4 (decay) | 10-11T | 58% | 24% Stack-Edu | 14% | 4% Cosmopedia v2 |

- §4.2: StarCoderData "limited it to 10% of the total mixture to ensure approximately 4 epochs over 11T tokens with room for upsampling in later stages. We did not include math data in stage 1 due to our math datasets' relatively small size."
- §4.3 findings: "increasing DCLM relative to FineWeb-Edu slightly improves MMLU MCF at this stage" (from additional annealing ablations).
- §4.5: "The final stage consisted of decaying the learning rate linearly to 0 for 10% of the total training duration (from 10T to 11T tokens)"; highest-quality math datasets InfiWebMath-3+ and FineMath 4+ introduced; "0.08% of the mixture to OWM and 0.02% to AugGSM8K (Li et al., 2024a), an augmented version of the GSM8K benchmark's training set".

## Per-stage results (Table 3; averages by category)
| | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| Knowledge/Reasoning | 55.50 | 56.76 | 57.47 | 60.24 |
| Math | 3.21 | 3.7 | 7.27 | 22.07 |
| Code | 8.87 | 10.56 | 16.75 | 23.21 |
| Generative Tasks | 31.54 | 31.30 | 34.70 | 36.12 |
Stages 1-3 are measured without LR decay; stage 4 includes decay.

## Context extension (§4.6)
> "we followed standard practice (Gao et al., 2024) and extended the context length from 2k to 8k tokens, by taking an intermediate checkpoint from stage 4 (before the final 75 billion tokens of training) and continuing training with a different data mixture and a RoPE value of 130k. The mixture was adjusted to include 40% long-context documents (8k tokens or more) sourced from DCLM (10%), FineWeb-Edu (10%), and the books subset of Dolma (20%) (Soldaini et al., 2024), while the remaining 60% followed the stage 4 mixture."

## Held-out evaluation (§4.7, Table 4)
> "SmolLM2 also delivers strong performance on held-out benchmarks not monitored during training, such as MMLU-Pro (Wang et al., 2024c), TriviaQA (Joshi et al., 2017), and Natural Questions"

SmolLM2 1.7B / Llama3.2 1B / Qwen2.5 1.5B: MMLU-Pro 19.4 / 11.7 / 13.7; Natural Questions 8.7 / 6.2 / 10.5; TriviaQA 36.7 / 28.1 / 20.9. "we see next to no degradation in performance after Context Length Extension".

## Not reported
No from-scratch control run per stage; no ablation of the 40/60 long-context ratio.
