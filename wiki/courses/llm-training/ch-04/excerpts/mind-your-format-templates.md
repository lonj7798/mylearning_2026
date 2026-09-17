---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Voronov, Wolf, Ryabinin — "Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements"
source_url: https://arxiv.org/abs/2401.06766
created_at: "2026-09-15"
note: "No library card for this artifact on 2026-09-15; written from the primary PDF (arXiv v3, 2024-06-06)."
---

# Excerpt: Voronov et al. 2024 — Template sensitivity in in-context learning

**Artifact:** arXiv:2401.06766 (v1 2024-01; read as v3). Authors: Anton Voronov, Lena Wolf, Max Ryabinin (Yandex, HSE University, MIPT, Together AI). Code: github.com/yandex-research/mind-your-format.

## Setup (§3)

- 21 models from 8 families (770M to 70B), including Llama 2, Falcon, and the instruction-tuned Llama 3 Instruct 8B and Mistral v0.3 Instruct 7B (Table 1).
- 4 classification datasets: SST-2, DBPedia, AGNews, TREC.
- A template = input verbalizer + output verbalizer + intra-separator + inter-separator (§2.1). The option sets give 216 templates for SST-2 and 168 for the others (§3.1, Table 2).
- For each setting: 3 demonstration seeds × 10 random templates; mean and standard deviation of accuracy (§3.2).

## Results

- Baseline (Table 3, 2-shot, DIRECT prediction): Llama 2 70B SST-2 0.83 ± 0.14, DBPedia 0.46 ± 0.15, AGNews 0.76 ± 0.14, TREC 0.41 ± 0.07. "Even the largest models have standard deviations of scores up to 35% of their mean values" (§4.1).
- No template component can be excluded as consistently poor, and combinations of suboptimal components can form an optimal template (§4.1, App. D).
- Advanced prediction methods (CHANNEL, CALIBRATION) and example-selection methods (ITM, Z-ICL) often lose their advantage once template variance is considered (§4.2-4.3).
- Transfer (§5): top-10 template sets overlap with IoU above 0.5 for only a few model pairs, including models of the same family (Fig. 4); transfer between prediction methods and between demonstration sets is also low (Table 4, Fig. 5).
- Instruction-tuned models also lack in-context robustness (§2.2, App. L).
- Template Ensembles, which average predictions over several templates at test time, raise average accuracy and reduce sensitivity to the chosen template set (§6).

## Use in ch-04

Independent support for the FormatSpread finding ([[prompt-format-sensitivity-formatspread]]) with different models, tasks, and template grammar.
