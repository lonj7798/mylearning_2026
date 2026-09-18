<!-- scope: recipe ledger (training and evaluation settings) for Math-Shepherd generators, completer, reward models, and step-by-step PPO
     deps: [[math-shepherd]]
     see-also: [[metamath]], [[ppo]], [[grpo]]
-->

# Recipe ledger: Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
- **Core Insight:** The paper reports its training settings in the §4 "Parameter Setting" paragraph (for example RL learning rate 1e-7 for Mistral-7B and KL coefficient 0.04); PPO clip range, samples per prompt, batch sizes, and number of steps are not reported.
- **Guideline:** When reproducing Math-Shepherd, take the values below from arXiv v3 §4 and treat every "not reported" row as a free choice that the paper did not ablate.
- **Authors:** Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, et al.
- **Year:** 2023 (arXiv v1 2023-12-14; v3 2024-02-19; ACL 2024)
- **URL:** https://arxiv.org/abs/2312.08935
- **Source type:** paper
- **Relevant topics:** process reward model training, step-level PPO hyperparameters, best-of-N evaluation settings

Companion to [[math-shepherd]]. All values are from the "Parameter Setting", "Datasets", and "Baselines and Metrics" paragraphs of §4 in arXiv v3 unless another locus is given. The ACL 2024 version (aclanthology.org/2024.acl-long.510) prints the same values in §4, naming the training framework HAI-LLM instead of hfai.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All generators and completers (LLaMA2-7B/13B/70B, LLemma-7B/34B, Mistral-7B, DeepSeek-67B) | 7B–70B | SFT | data; epochs | MetaMATH; 3 epochs | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| Mistral-7B generator | 7B | SFT | learning rate | 5e-6 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| Other 7B and 13B generators/completers (LLaMA2-7B, LLaMA2-13B, LLemma-7B) | 7B, 13B | SFT | learning rate | 2e-5 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| LLemma-34B generator | 34B | SFT | learning rate | 1e-5 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| DeepSeek-67B, LLaMA2-70B generators | 67B, 70B | SFT | learning rate | 6e-6 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| RM-data generators (7B and 13B models) | 7B, 13B | SFT | data; epochs | GSM8K and MATH training sets; 1 epoch | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| RM-data generators (7B and 13B models) | 7B, 13B | reward-model | solutions sampled per training problem | 15 per problem from each model, then deduplicated | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| Completer: LLemma-7B | 7B | reward-model | completions per step (N) | 8 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | §5.2 Fig. 4a (LLaMA2-70B completer, 160 steps): hard-label accuracy 86% at N = 4, lower at larger N |
| Reward-model training data | n/a | reward-model | number of labelled solutions | around 170k (GSM8K); around 270k (MATH) | arXiv:2312.08935v3 §4 | verified 2026-09-14 | §5.4 Fig. 6a: accuracy vs 10k–160k training solutions |
| Verification RMs: LLaMA2-70B (GSM8K), LLemma-34B (MATH) | 70B, 34B | reward-model | base model | as listed | arXiv:2312.08935v3 §4, Table 1 caption | verified 2026-09-14 | §5.3 Fig. 5: larger RMs more robust as candidates grow |
| RL reward model: Mistral-7B | 7B | reward-model | base model; policies supervised | Mistral-7B; LLaMA2-7B and Mistral-7B | arXiv:2312.08935v3 §4, Table 2 caption | verified 2026-09-14 | no ablation reported |
| All ORMs and PRMs | 7B–70B | reward-model | epochs; learning rate | 1 epoch; 1e-6 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| All PRMs | 7B–70B | reward-model | label type | hard estimate, two special tokens ("has potential" / "no potential") | arXiv:2312.08935v3 §4 | verified 2026-09-14 | §5.2: soft vs hard labels give no substantial verifier difference |
| LLaMA2-7B policy | 7B | RL | learning rate | 4e-7 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| Mistral-7B policy | 7B | RL | learning rate | 1e-7 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| LLaMA2-7B and Mistral-7B policies | 7B | RL | KL coefficient | 0.04 (where the KL term is applied: not reported) | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| LLaMA2-7B and Mistral-7B policies | 7B | RL | RL prompts | questions from MetaMATH | arXiv:2312.08935v3 Table 2 caption | verified 2026-09-14 | no ablation reported |
| LLaMA2-7B and Mistral-7B policies | 7B | RL | reward placement | PRM score at the end of each reasoning step, 0 elsewhere | arXiv:2312.08935v3 §3.5; ACL 2024 Eq. 7 | verified 2026-09-14 | Table 2: step-by-step PPO 84.1 / 33.0 vs ORM-PPO 81.8 / 31.3 (Mistral-7B) |
| Scope not stated (sentence follows the RL settings) | n/a | RL | LR schedule | cosine, minimum learning rate 1e-8 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| All models | 7B–70B | all | maximum sequence length | 512 | arXiv:2312.08935v3 §4 | verified 2026-09-14 | no ablation reported |
| LLaMA2-7B and Mistral-7B policies | 7B | RL | PPO clip range, samples per prompt, prompts per step, batch size, number of steps, sampling temperature, value-model setup, compute | not reported | checked v3 §3.5, §4 and ACL 2024 §3.5, §4 | not reported | none |
| RFT baseline (LLaMA2-7B, Mistral-7B) | 7B | SFT | responses sampled per MetaMATH question | 8 | arXiv:2312.08935v3 §4 "Baselines and Metrics" | verified 2026-09-14 | not applicable (baseline) |
| All verification runs | n/a | eval-gate | candidates per problem; aggregation; repeats | 256; minimum step score; mean of 3 sampling groups | arXiv:2312.08935v3 §4, §3.4 | verified 2026-09-14 | §5.1 Fig. 3: accuracy vs N = 1 to 256 |
| All verification runs | n/a | eval-gate | test sets | GSM8K full test; MATH500 (Lightman et al. split) | arXiv:2312.08935v3 §4 | verified 2026-09-14 | §4: MATH500 gives results similar to the full set (no numbers given) |
| RL runs | 7B | eval-gate | decoding; test sets | greedy; GSM8K and full MATH test | arXiv:2312.08935v3 §4 | verified 2026-09-14 | not applicable |

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2312.08935 (v3, §3.5, §4) and https://aclanthology.org/2024.acl-long.510.pdf (§3.5 Eq. 6–7, §4).
- Corrections to earlier ledgers: this file is new; values replace the unsupported "K = 8 or 16", "λ ≈ 0.1–1.0", and "N = 256 or 1024" of the previous [[math-shepherd]] card.
- Removed as unsupported by the source: none.
