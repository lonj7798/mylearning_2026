---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/tulu-3-sft-mix.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from arXiv:2411.15124v5 Tables 7, 8, 37)"
---

# Excerpt: Tülu 3 prompt and SFT mixture counts, with decontamination results

**Primary source:** Lambert et al., "Tülu 3: Pushing Frontiers in Open Language Model Post-Training", arXiv:2411.15124v5 (2025-04), Table 7, Table 8, Table 37, §3.2.
**Library cards:** [[tulu-3]], [[tulu-3-sft-mix]]. The [[tulu-3-sft-mix]] card lists 18 SFT components; Table 7 has 19, including OpenMathInstruct 2 (50,000). Without that row the counts sum to 889,344, not 939,344.

## Table 7 — prompts available, used in SFT, and used in DPO

"Count" is the size of the source prompt set; SFT and DPO columns are the prompts used at each stage. The DPO column total (425,145) counts prompts across the 8B and 70B mixes; Table 15 gives the per-size mixes.

| Category | Dataset | Count | Used in SFT | Used in DPO |
|---|---|---:|---:|---:|
| General | Tülu 3 Hardcoded | 24 | 240 (upsampled) | – |
| General | OpenAssistant | 88,838 | 7,132 | 7,132 |
| General | No Robots | 9,500 | 9,500 | 9,500 |
| General | WildChat (GPT-4 subset) | 241,307 | 100,000 | 100,000 |
| General | UltraFeedback | 41,635 | – | 41,635 |
| Knowledge recall | FLAN v2 | 89,982 | 89,982 | 12,141 |
| Knowledge recall | SciRIFF | 35,357 | 10,000 | 17,590 |
| Knowledge recall | TableGPT | 13,222 | 5,000 | 6,049 |
| Math reasoning | Tülu 3 Persona MATH | 149,960 | 149,960 | – |
| Math reasoning | Tülu 3 Persona GSM | 49,980 | 49,980 | – |
| Math reasoning | Tülu 3 Persona Algebra | 20,000 | 20,000 | – |
| Math reasoning | OpenMathInstruct 2 | 21,972,791 | 50,000 | 26,356 |
| Math reasoning | NuminaMath-TIR | 64,312 | 64,312 | 8,677 |
| Coding | Tülu 3 Persona Python | 34,999 | 34,999 | – |
| Coding | Evol CodeAlpaca | 107,276 | 107,276 | 14,200 |
| Safety and non-compliance | Tülu 3 CoCoNot | 10,983 | 10,983 | 10,983 |
| Safety and non-compliance | Tülu 3 WildJailbreak | 50,000 | 50,000 | 26,356 |
| Safety and non-compliance | Tülu 3 WildGuardMix | 50,000 | 50,000 | 26,356 |
| Multilingual | Aya | 202,285 | 100,000 | 32,210 |
| Precise IF | Tülu 3 Persona IF | 29,980 | 29,980 | 19,890 |
| Precise IF | Tülu 3 IF-augmented | 65,530 | – | 65,530 |
| **Total** | | **23,327,961** | **939,344** | **425,145** |

Check: 240 + 7,132 + 9,500 + 100,000 + 89,982 + 10,000 + 5,000 + 149,960 + 49,980 + 20,000 + 50,000 + 64,312 + 34,999 + 107,276 + 10,983 + 50,000 + 50,000 + 100,000 + 29,980 = 939,344.

The Ai2 technical blog states that 57% of the 939,344 prompts come from public resources and 43% were "synthetically generated in house" (allenai.org/blog/tulu-3-technical). Neither the blog nor Table 7 labels each row as public or in-house, so the split cannot be recomputed exactly from the table.

Persona sources were generated with GPT-4o-2024-08-06 (problems, math solutions, IF instructions) and claude-3-5-sonnet (Python solutions), conditioned on about 250K personas from [[persona-hub]]; Persona IF covers the 25 IFEval constraint types (§3.1.2).

## Decontamination rule (§3.2)

1. Overlap is computed on prompts (user turns) only.
2. A test token matches if the test and training instance share an 8-gram containing it.
3. A test instance overlaps a training instance if more than 50% of its tokens match that instance.
4. A training dataset is contaminated if it overlaps more than 2% of any development or unseen evaluation.
5. Datasets contaminated with unseen evaluations are removed. For development evaluations, the dataset is removed if that does not significantly hurt performance; otherwise the matching instances are removed.

Full-string and embedding-based matching were tested; embedding matching was rejected because it could not separate paraphrase from distributional similarity (§3.2).

## Table 8 — datasets released in decontaminated form

| Dataset | Evaluation | % of dataset removed |
|---|---|---:|
| Evol CodeAlpaca | HumanEval | 3.5 |
| WildChat GPT-4 | Safety | 5.4 |
| WildJailbreak | Safety | 0.7 |
| WildGuardMix | Safety | 1.1 |
| NuminaMath-TIR | MATH | 11.3 |

## Table 37 — overlap found before decontamination (selected rows, % of evaluation instances)

| Dataset | Evaluation | % eval overlap | Used? |
|---|---|---:|---|
| Evol CodeAlpaca | HumanEval | 70.7 | decontaminated version |
| NuminaMath-TIR | MATH | 18.2 | decontaminated version |
| WildGuardMix | Do-Anything-Now | 39.7 | decontaminated version |
| DaringAnteater | MATH | 30.7 | not used |
| ShareGPT | AlpacaEval | 19.2 | not used |
| LMSys Chat 1M | Do-Anything-Now | 90.3 | not used |
| LMSys Chat 1M | AlpacaEval | 46.5 | not used |

The authors' takeaway: datasets of realistic API usage (ShareGPT, WildChat, LMSys Chat) are likely to overlap existing test sets and should be decontaminated before training (App. B.2).

## Used in

ch-33 §2.1 (mixture), §2.2 (decontamination), Common mistakes table.
