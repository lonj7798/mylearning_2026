---
chapter: ch-08b
course: llm-training
phase: read
excerpt_of: primary source arXiv:2206.07682v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2206.07682
created_at: "2026-09-15"
---

# Excerpt: Emergent Abilities of Large Language Models

**Paper:** Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, et al. (Google Research, Stanford University, UNC Chapel Hill, DeepMind). arXiv v1 2022-06, read at v2 (2022-10-26); Transactions on Machine Learning Research 08/2022. Source type: paper (survey of prior results with added analyses).

## Definition (§1, §2)
- "An ability is emergent if it is not present in smaller models but is present in larger models." Such abilities cannot be predicted by extrapolating smaller models.
- On a scaling curve: near-random until a critical scale, then substantially above random.
- x-axis: training FLOPs (parameters in App. D); training dataset size is not plotted because many families use a fixed token count for all sizes.
- The emergence scale "is not an immutable property of the ability"; it may be lower with higher-quality data (§2).

## Few-shot examples (§3, Fig. 2)
- BIG-Bench modified arithmetic: GPT-3 and LaMDA near zero until 2e22 FLOPs (13B) for GPT-3 and 1e23 (68B) for LaMDA.
- MMLU: GPT-3, Gopher, Chinchilla at ~1e22 FLOPs (~10B) or smaller not above guessing; 3-5e23 FLOPs (70B-280B) substantially above random.
- WiC: GPT-3 and Chinchilla not above random one-shot at ~5e23 FLOPs; PaLM above random at 2.5e24 FLOPs (540B).

## Table 1 (emergent scale: FLOPs, parameters, model)
- Few-shot: addition/subtraction 3-digit 2.3E+22, 13B, GPT-3; 4-5 digit 3.1E+23, 175B, GPT-3; MMLU 57-topic average 3.1E+23, 175B, GPT-3; toxicity classification 1.3E+22, 7.1B, Gopher; TruthfulQA 5.0E+23, 280B, Gopher; grounded conceptual mappings 3.1E+23, 175B, GPT-3; MMLU 30 topics 5.0E+23, 70B, Chinchilla; WiC 2.5E+24, 540B, PaLM.
- Augmented prompting: instruction following (fine-tuning) 1.3E+23, 68B, FLAN; scratchpad 8-digit addition 8.9E+19, 40M, LaMDA; chain-of-thought math word problems 1.3E+23, 68B, LaMDA; chain-of-thought StrategyQA 2.9E+23, 62B, PaLM; calibration via P(True) 2.6E+23, 52B, Anthropic; zero-shot chain-of-thought 3.1E+23, 175B, GPT-3.

## Discussion (§5)
- §5.1: exact-match metrics on long targets may disguise incremental improvements, but this is "at best an incomplete explanation" because emergence also appears on classification tasks (Fig. 2D-H).
- §5.1 and App. A.1: cross-entropy on six BIG-Bench tasks emergent for LaMDA (three generative, three classification) improves for models ≤1e22 FLOPs / ≤27B while EM/BLEU/accuracy is random ("Outcome 2" for all six); the analysis "does not provide any straightforward indicators of how to predict such emergent behaviors".
- §5.2: 14 BIG-Bench tasks where PaLM 62B is above random while LaMDA 137B and GPT-3 175B are near random (listed in App. F); potential reasons named: higher-quality data (more multilingual and code data) and architecture differences; no ablation.
- §5.2 cites data features correlated with emergent few-shot prompting (long-range coherence, many rare classes; Xie et al. 2022, Chan et al. 2022).
- §5.3: emergence can also be analyzed against WikiText103 perplexity, which correlates with FLOPs for Gopher/Chinchilla.

## BIG-Bench classification (App. A.3, App. E)
- All 210 BIG-Bench tasks classified manually; two co-authors agreed on every emergent label.
- Appendix E lists (counts derived by ch-08b from the lists): smoothly increasing 58; emergent with GPT-3 or LaMDA 25; emergent with PaLM 42; flat (no model better than random) 45; plus "Other" categories.
- Visual reasoning has the largest fraction of flat tasks (8/13).

## Verification
- Read on 2026-09-15 against arXiv PDF v2 (Abstract, §1-§5, Table 1, App. A.1, A.3, E, F).
