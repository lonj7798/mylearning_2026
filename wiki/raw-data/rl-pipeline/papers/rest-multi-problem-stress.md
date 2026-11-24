<!-- scope: REST (arXiv:2507.10541) — stress test that concatenates s problems into one prompt; 34 reasoning models; accuracy drop under multi-question prompts; failure taxonomy; post-training and long2short comparisons (observational)
     deps: []
     see-also: [[scaling-reasoning-losing-control-mathif]], [[lost-in-multi-turn]], [[tau2-bench]], [[dapo]], [[diversity-axes]], [[rl-design-guide]]
-->

# REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once
- **Core Insight:** Reasoning models that are near ceiling on one-question prompts lose large amounts of accuracy when several questions share a prompt — DeepSeek-R1 drops from 81.66 to 52.49 on AIME24 and from 68.75 to 37.17 on AIME25 (Table 1) — and models with equal single-question scores separate under stress (MATH500: R1-7B 93.00 → 66.75, R1-32B 94.60 → 88.97); the main failure types are question omission, reasoning error, endless repetition and truncation, with earlier questions answered more accurately than later ones (Table 3, Figs. 5, 7).
- **Guideline:** When a model will receive prompts containing more than one task, evaluate it with multi-question prompts at several stress levels in addition to single-question benchmarks, because single-question accuracy did not predict stressed accuracy (MATH500 above); the paper's observation that length-reduced models do better (L1-Qwen-1.5B-Max 67.11 vs R1-1.5B 22.40 at stress 9 on MATH500) is a comparison of released models, not a controlled training experiment, so it does not establish that single-question RL causes the drop.
- **Authors:** Zhuoshi Pan, Qizhi Pei, Yu Li, Qiyao Sun, Zinan Tang, H. Vicky Zhao, Conghui He, Lijun Wu (Tsinghua University; Shanghai AI Laboratory / OpenDataLab; Renmin University of China)
- **Year:** 2025 (arXiv v1, 15 Jul 2025; v2 read; ACL 2026 long paper per search listing)
- **URL:** https://arxiv.org/abs/2507.10541 (project: https://opendatalab.github.io/REST)
- **Source type:** paper (benchmark)
- **Relevant topics:** multi-question prompts, tasks per episode, overthinking, position effects, evaluation of reasoning models

## Summary
REST builds a prompt from s consecutive benchmark questions followed by "Answer the above questions one by one"; indices wrap so every question appears once in each of the s positions (§3.1). Stress levels: GSM8K {1, 3, 6, 9, 12}; MATH500 and AMC23 {1, 3, 5, 7, 9}; AIME24, AIME25, GPQA Diamond, LiveCodeBench v5 {1, 2, 3, 4, 5} (§4.1). The REST score averages accuracy over levels above 1. 34 models from 1.5B to 671B are tested with 32K output for reasoning models; a 128K limit made negligible difference (Table 5).

## Key Contributions
- A multi-question stress protocol with position balancing (§3.1).
- Measured degradation across 34 models and 7 benchmarks (Table 1).
- Error taxonomy: Output Truncation, Endless Repetition, Format Violation, Question Omission, Summary Error, Reasoning Error (Table 3).
- Observational comparisons of post-trained and long2short models (§4–5).

## Key Figures/Tables to Study
- Table 1 (single vs stressed accuracy), Table 3 / Fig. 5 (error shares), Fig. 7 (accuracy by position), Fig. 8 (question order).

## Technical Details
**Simple tasks are unaffected.** GSM8K: R1-7B 89.49 → 89.06; R1-32B 95.54 → 95.50 (Table 1).

**Post-trained variants (observational).** AReaL-boba-RL-7B scores 95.00 single vs 60.77 stressed on MATH500, OpenThinker2-7B 93.80 vs 83.30. At stress 9 on MATH500, AReaL-boba-RL-7B 38.07 and Light-R1-7B-DS 39.11 vs 49.42 for their base R1-7B. The authors say these findings "reinforce recent calls" for "a more cautious assessment of the benefits of post RL training."

**Long2short (observational).** At stress 9 on MATH500: L1-Qwen-1.5B-Max 67.11 vs R1-1.5B 22.40; Efficient-R1-7B (α = 0.2) 66.02 vs R1-7B 49.42, despite lower single-question accuracy (88.20 vs 93.00).

**Error shares.** DeepSeek-R1 on AIME24: question omission + reasoning error = 42.00% of degradation (Fig. 5d). Light-R1-7B: endless repetition + truncation = 48.90% (Fig. 5b). Truncation is "not the sole cause"; failures often occur well within the token limit.

**Overthinking (§5.1).** Better REST performers (Nemotron-nano-7B, DeepSeek-R1) spend fewer tokens on the first question; R1-7B "often overthink[s] the first question." Qualitative; no correlation coefficient. Easy-to-hard question order beats hard-to-easy (Fig. 8).

## Verification
- Read arXiv HTML v2, first 100K of 110K characters, on 2026-11-24, plus a web search for training studies on multi-question prompts (none found).
- Causality: the paper trains no model. The RL-variant comparison (AReaL-boba-RL-7B vs its base) is one same-lineage pair with unknown data and length settings; it is consistent with, but does not isolate, a training cause.
- §4.1 says "77 representative benchmarks" while the abstract and introduction describe 7; the card uses 7.
