<!-- scope: Hsieh et al. (Findings of ACL 2023): extract few-shot CoT rationales from PaLM-540B and train task-specific T5 students (220M-11B) with a multi-task label + rationale objective on 4 NLP benchmarks
     deps: [[star]]
     see-also: [[orca]], [[orca-2]], [[deepseek-r1-distill-synth]], [[quiet-star]]
-->

# Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes
- **Core Insight:** With all human-labeled training data, T5 students trained on labels plus PaLM-540B rationales outperform PaLM-540B few-shot CoT on e-SNLI (220M), ANLI and SVAMP (770M), and CQA (11B) (§4.2, Fig. 6); on ANLI a 770M student does this with 80% of the labeled data, while standard finetuning of the same size does not match the LLM with 100% (§4.3, Fig. 8).
- **Guideline:** When training a small task-specific model with LLM rationales, train rationale generation and label prediction as two tasks selected by input prefixes instead of one concatenated rationale-then-label target, because on 220M T5 the multi-task variant scored higher on all 4 datasets and the single-task variant scored below standard finetuning on ANLI and CQA (§4.4, Table 2).
- **Authors:** Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, et al. (University of Washington, Google Cloud AI Research, Google Research)
- **Year:** 2023 (arXiv v1 2023-05; Findings of ACL 2023, pp. 8003-8017)
- **URL:** https://arxiv.org/abs/2305.02301
- **Source type:** paper (official code: github.com/google-research/distilling-step-by-step)
- **Relevant topics:** rationale distillation, multi-task learning, chain-of-thought, task-specific small models, data efficiency

## Abstract
Deploying LLMs is memory- and compute-intensive, so practitioners train smaller task-specific models by finetuning on human labels or by distilling LLM-generated labels, and both "require large amounts of training data to achieve comparable performance to LLMs" (Abstract). Distilling step-by-step extracts LLM rationales and uses them as additional supervision for small models in a multi-task framework. Across 4 NLP benchmarks the authors report three findings: the method beats standard finetuning and distillation with fewer labeled or unlabeled examples; it beats few-shot prompted LLMs with smaller models (up to 2000× smaller, §1); and it reduces model size and data at the same time, with a finetuned 770M T5 outperforming few-shot prompted 540B PaLM using 80% of the available data "on a benchmark", where standard finetuning of the same T5 does not match the LLM with 100%.

## Key Contributions
- A two-step method: few-shot CoT prompting of an LLM to produce rationales and labels for training inputs, then multi-task training of a small model on label and rationale targets (§3, Fig. 2).
- Data-size sweeps against standard finetuning (human labels) and standard task distillation (LLM labels) (§4.1, Figs. 4-5).
- Model-size sweeps (220M, 770M, 11B) against few-shot CoT and PINTO tuning (§4.2, Figs. 6-7), and minimum model and data size to beat the LLM (§4.3, Figs. 8-9).
- Ablations on the rationale LLM (PaLM 540B vs GPT-NeoX 20B) and on multi-task vs single-task training (§4.4, Tables 1-2).

## Key Figures/Tables to Study
- Figure 2 (pipeline) and Figure 3 (few-shot CoT prompt with rationale and label).
- Figures 4-5: accuracy vs % of training data for 220M T5. Figures 6-7: accuracy vs model size. Figures 8-9: minimum resources.
- Tables 1-2: rationale source and training-format ablations. Table 3: dataset splits.

## Technical Details
- **Rationale extraction:** each prompt is a set of triplets (x^p, r^p, y^p): example input, human-written rationale, label; the LLM continues the pattern for each unlabeled x_i to produce r̂_i and ŷ_i (§3.1, Fig. 3). Exemplars follow Wei et al. (2022) where available and are curated for new datasets (§4 Setup); about 10-shot for all tasks (Limitations).
- **Label loss:** L_label = (1/N) Σ_i ℓ(f(x_i), ŷ_i) (Eq. 1). N = number of training examples; f = the small model; ℓ = token cross-entropy; ŷ_i = human label y_i in the finetuning setting or the LLM label in the distillation setting.
- **Multi-task loss:** L = L_label + λ L_rationale, with L_rationale = (1/N) Σ_i ℓ(f(x_i), r̂_i) (Eqs. 3-4). λ = weight on the rationale loss; r̂_i = LLM rationale. The paper does not print a λ value.
- **Task prefixes:** the input is prefixed with `[label]` to produce ŷ_i or `[rationale]` to produce r̂_i; one text-to-text model, no separate heads; only the label is generated at test time, so no LLM is needed at deployment (§3.2).
- **Single-task alternative:** L_single = (1/N) Σ_i ℓ(f(x_i), [r̂_i, ŷ_i]), rationale and label concatenated into one target (Eq. 5, §4.4).
- **Models:** teacher PaLM 540B; students T5-Base 220M, T5-Large 770M, T5-XXL 11B from public pretrained weights (§4 Setup, §4.2). PINTO tuning baseline = 220M T5 finetuned on PaLM outputs (§4.2).
- **Datasets (train / val / test):** e-SNLI 549,367 / 9,842 / 9,824; ANLI R1 16,946 / 1,000 / 1,000; CQA 8,766 / 975 / 1,221 (original validation used as test); SVAMP 720 / 80 / 200 (App. A.2, Table 3). Where no validation set exists, 10% of train is held out (App. A.2).
- **Less data than standard finetuning (220M):** 12.5% of e-SNLI beats standard finetuning on 100%; 75%, 25%, and 20% fewer examples needed on ANLI, CQA, SVAMP (§4.1, Fig. 4). Across datasets: over 50% fewer examples on average, up to over 85% (§1).
- **Less data than standard distillation (220M, unlabeled):** outperforms on all 4 datasets at every data size tested; 12.5% of the unlabeled set beats distillation on 100% (§4.1 text says e-SNLI; Fig. 5 caption says ANLI).
- **Smaller than the LLM, unlabeled setting:** beats few-shot CoT on 3 of 4 datasets with 11B T5 (§4.2, Fig. 7). SVAMP falls short; the authors attribute this to its small size (800 examples); adding 2,305 unlabeled ASDiv problems lets 11B T5 reach few-shot CoT, while standard distillation does not (§4.2, Fig. 7).
- **Minimum resources:** e-SNLI beaten with 220M T5 and 0.1% of the labeled data (§4.3, Fig. 8); ANLI beaten with a 45x smaller model and 50% of the unlabeled data (§4.3, Fig. 9). Improvements are largest on ANLI: by an average of 8% over standard finetuning and 13% over distillation in task accuracy across sizes (§4.2).
- **Ablations (220M T5, 100% data, accuracy e-SNLI / ANLI / CQA / SVAMP):** standard finetuning 88.38 / 43.58 / 62.19 / 62.63; rationales from GPT-NeoX 20B 89.12 / 48.15 / 63.25 / 63.00; from PaLM 540B 89.51 / 49.58 / 63.29 / 65.50 (Table 1). Single-task 88.88 / 43.50 / 61.37 / 63.00; multi-task 89.51 / 49.58 / 63.29 / 65.50 (Table 2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| T5-Base, T5-Large students | 220M, 770M | distill-SFT | peak LR; batch; max input length; max steps | 5×10⁻⁵; 64; 1024; 10,000 | arXiv:2305.02301v2 App. A.1 | verified 2026-09-14 | no ablation reported |
| T5-XXL student | 11B | distill-SFT | peak LR; batch; max input length; max steps | 5×10⁻⁵; 32; 1024; 4,000 | App. A.1 | verified 2026-09-14 | no ablation reported |
| All students | 220M-11B | distill-SFT | rationale-loss weight λ (paper) | not reported | checked §3.2, §4, App. A | not reported | — |
| Released code | — | distill-SFT | task weight `--alpha`, loss = α·label + (1−α)·rationale | 0.5 "recommended" | repo README (main) | verified 2026-09-14 (released code; not confirmed as the paper's runs) | no ablation reported |
| All students | 220M-11B | distill-SFT | rationale source; exemplars | PaLM 540B (GPT-NeoX 20B in ablation); about 10-shot CoT | §4 Setup; §4.4; Limitations | verified 2026-09-14 | Table 1: 540B rationales > 20B rationales on all 4 datasets |
| All students | 220M-11B | distill-SFT | compute; runs | A100×16 cloud instances; 4 random runs, standard error in plots | App. A.1 | verified 2026-09-14 | n/a |
| All students | 220M-11B | distill-SFT | optimizer, LR schedule, warmup, teacher decoding temperature, checkpoint selection, rationale filtering | not reported | checked body, App. A, repo README | not reported | — |

## Findings relevant to distillation and generality
- **Rationale quality:** rationales from GPT-NeoX 20B give a smaller gain than those from PaLM 540B (Table 1); the authors attribute this to higher-quality rationales from the larger model (Interpretation, §4.4) and list characterizing rationale quality as future work (Limitations).
- **Training format:** concatenating rationale and label as one target can score below label-only finetuning (Table 2), which the authors relate to similar observations by Wiegreffe et al. (2021), Magister et al. (2022), and Ho et al. (2022) (§4.4).
- **Scope:** every student is trained and tested on one dataset (§3.2, §4); the paper reports no evaluation on held-out tasks, so it does not test whether rationale supervision affects general capability.
- **Stated limits:** demonstrations must be written by users; training has slight extra compute; LLMs show limited reasoning on more complex reasoning and planning tasks; students inherit teacher biases (Limitations, Ethics statement).

## Connections
- [[star]]: cited as prior work that uses reasoning steps as finetuning data to self-improve LLMs (§2).
- [[orca]], [[orca-2]]: later explanation-trace SFT work; the [[orca]] card lists this paper as a prerequisite (course link, not a claim of this paper).
- [[deepseek-r1-distill-synth]]: later reasoning-trace distillation into general-purpose students (course link).
- [[quiet-star]]: rationale generation learned inside the model rather than taken from a teacher (course link).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2305.02301 (arXiv v2, 5 Jul 2023, PDF incl. appendix); https://aclanthology.org/2023.findings-acl.507/ (venue, pages); github.com/google-research/distilling-step-by-step README (main branch).
- Corrections to the previous card version:
  - Title "Distilling Step-by-Step: Rationale-Plus-Label Distillation for Small Models" → exact title above; venue "ACL 2023 Findings" → Findings of ACL 2023, pp. 8003-8017.
  - "770M student beats PaLM-540B few-shot on ANLI / e-SNLI / CQA / SVAMP using only 80% of the data" → the 80% result is ANLI only (abstract "on a benchmark"; §4.3, Fig. 8); with 100% data the smallest winning student is 220M on e-SNLI, 770M on ANLI and SVAMP, 11B on CQA (§4.2).
  - "Data efficiency: 80% of data suffices" → over 50% fewer examples on average, up to over 85%, relative to standard finetuning or distillation (§1, §4.1).
  - "3-8 exemplars" → about 10-shot for all tasks (Limitations).
  - "separate label and rationale output heads" → one T5 model with `[label]` / `[rationale]` input prefixes (§3.2).
  - "student: T5 family or smaller decoder-only models" → T5 220M, 770M, 11B only (§4.2).
- Removed as unsupported by the source: filtering of teacher outputs by format or verifier; "rationale head acts as regularization"; the "Why rationales help" mechanism claims; "compute doubles if rationales are emitted"; "later work (R1 distill) requires much longer traces"; "foundational parent of Orca and R1-distill" as a claim of the paper.
- Not reported by the source: λ value, optimizer and schedule, teacher sampling settings, whether rationales with wrong teacher labels were removed, held-out-task evaluation.
