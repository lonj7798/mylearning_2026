---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/noveltybench.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2504.05228
created_at: "2026-09-15"
---

# Excerpt: NoveltyBench: Evaluating Language Models for Humanlike Diversity

**Authors:** Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, Daphne Ippolito (Carnegie Mellon University).
**Version read:** arXiv:2504.05228v4 (9 Aug 2025), COLM 2025.
**Status:** no library card existed for this slug on 2026-09-15; numbers read at the stated loci in the v4 PDF, with Figure 6 values read from the rendered page 9.

## Data (§3.1)
1,100 prompts: NB-CURATED, 100 author-written prompts in four categories (randomness, underspecified factual knowledge, creative writing, subjectivity), each with responses from 8 human annotators; NB-WILDCHAT, 1,000 prompts filtered from WildChat-1M with Llama Guard 3, deduplicated by user IP, then selected by a GPT-4o classifier (85% agreement with human labels on 100 checked prompts).

## Metrics (§3.2)
Generations are partitioned into functional equivalence classes by a deberta-v3-large classifier fine-tuned on 1,000 author-annotated pairs; on a held-out 100 pairs it reaches 79% accuracy and F1 0.811. Then

```
distinct_k := |{c_i : i ∈ [k]}|                                            (Eq. 1)
utility_k  := ((1 − p) / (1 − p^k)) · Σ_{i=1..k} p^{i−1} · 1[c_i ≠ c_j ∀ j < i] · u_i   (Eq. 2)
```

where c_i is the equivalence class of the i-th generation, u_i its quality score, p ∈ [0, 1] the user's patience (probability of asking for one more generation), and (1 − p)/(1 − p^k) a normalization factor. The evaluation uses p = 0.8, k = 10 generations at temperature 1, and maps rewards from Skywork-Reward-Gemma-2-27B-v0.2 to utilities in {1,…,10} calibrated against GPT-4 quality judgments on 2,400 MT-Bench generations.

## Model results (Table 1, k = 10)
Claude-3.5 Sonnet 1.76 distinct / 2.36 utility; Claude-3 Opus 2.04 / 2.67; gpt-4o 2.88 / 3.27; gemini-2.0-pro 2.25 / 2.64; command-r7b 3.58 / 3.35; gemma-2-2b-it 5.66 / 4.63; gemma-2-9b-it 3.25 / 3.93; gemma-2-27b-it 3.03 / 3.77; Llama-3.2-1B 6.74 / 2.81; Llama-3.2-3B 5.10 / 3.24; Llama-3.1-8B 5.24 / 3.76; Llama-3.3-70B 2.49 / 2.87; Llama-3.1-405B 3.20 / 3.39. The paper states that "larger and more capable models in the same model family tend to produce less diverse outputs" and that closed-source Claude, Gemini and GPT-4o models score below 4 of a maximum utility of 10 (§4.2).

## Post-training stages (§4.4, Fig. 6, OLMo 2)
distinct_10 by stage (SFT → DPO → RLVR): 1B 8.83 → 8.08 → 7.85; 7B 7.46 → 5.96 → 5.72; 13B 7.47 → 5.61 → 5.16; 32B 7.25 → 5.22 → 5.08. utility_10: 1B 2.40 → 3.03 → 3.21; 7B 4.02 → 4.35 → 4.32; 13B 4.29 → 4.62 → 4.46; 32B 4.22 → 4.63 → 4.62. "each alignment stage progressively reduces model diversity, with significant drops occurring during DPO", while utility rises from SFT to DPO.

## Prompting workarounds (§4.3, Fig. 5)
Paraphrasing and a diversity-asking system prompt are "only marginally effective". In-context regeneration (asking for a different answer with previous answers in context) brings Claude-3 Opus, GPT-4o and Gemini 2.0 Pro to roughly the diversity of the 8 human writers, and GPT-4o and Gemini 2.0 Pro exceed the human cumulative utility. The authors conclude that the diversity "is not inherently built into the models' output distributions".

## Stated open question (§5)
Whether users want diversity is prompt-dependent: a dice roll needs it, a car recommendation may not.

## How ch-38a uses it
§6 (the metric definitions, the worked utility_k example, the OLMo 2 stage numbers), §9 (diversity protocol), Generalization lens.
