<!-- scope: controlled study (Zhao et al., COLM 2025) of RL fine-tuning on 150M/1B models pretrained from scratch on math corpora plus format-tagged instruction datasets; which pretraining format RL amplifies, scale and KL effects, transfer to MATH-500/AIME
     deps: [[grpo]], [[ppo]]
     see-also: [[spurious-rewards-rlvr]], [[rlvr-beyond-base-model]], [[prorl]], [[front-loading-reasoning]], [[openmathinstruct-2]]
-->

# Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining
- **Core Insight:** In models pretrained from scratch on math documents plus instruction datasets with distinct solution formats, PPO on GSM8K questions moves generations to the format of a single pretraining dataset within the first epoch, coinciding with the largest pass@1 gain (§3.1, Fig. 2); the 1B models also gain 4.0–10.4 points pass@1 on MATH-500, which was not used in RL (§4, Table 1).
- **Guideline:** When a behaviour or output format appears after RL, measure its share among base-model generations before RL and track that share during training, because in this study RL re-weighted formats present after pretraining, and the selected format depended on the pretraining mixture, on model scale, and on the KL coefficient (§3.1–3.4, Fig. 3).
- **Authors:** Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, Eran Malach
- **Year:** 2025 (arXiv v1 2025-04; COLM 2025)
- **URL:** https://arxiv.org/abs/2504.07912
- **Source type:** paper (code: github.com/rosieyzh/openrlhf-pretrain)
- **Relevant topics:** RL fine-tuning mechanisms, pretraining data composition, PPO, GRPO, expert iteration, output diversity, pass@k, transfer to harder benchmarks, model scale

## Abstract
The effects of RL fine-tuning are hard to separate from pretraining data composition, hyperparameters, and model scale, and many base models do not disclose their training data. The authors train models from scratch on mixtures of fully open datasets and then fine-tune them for math reasoning with PPO, GRPO, and Expert Iteration at two scales. RL algorithms consistently converge toward a dominant output distribution that amplifies patterns in the pretraining data. Models of different scales trained on the same mixture converge to different output distributions, which the authors describe as scale-dependent biases. RL on simpler questions improves performance on harder ones. The authors conclude that small-scale controlled proxies can give insight into how RL shapes model behaviour.

## Key Contributions
- End-to-end setup with 150M and 1B models pretrained on known mixtures, so each instruction dataset's format can be traced in generations (§2, Fig. 1).
- RL converges on one pretraining format, often raising pass@1 while reducing diversity; the preferred format is usually the one with the highest base accuracy, with failure cases (§3.1–3.2).
- Within the preferred format, RL further refines stylistic properties such as docstring conventions (§3.3, Fig. 6).
- Scale-dependent preference: 150M models prefer TinyGSM-style code; 1B models prefer OpenMathInstruct2-style natural language, then OpenMathInstruct1-style code (§3.4, Fig. 7, App. E).
- Positive transfer from GSM8K RL to MATH-500 and, to a lesser degree, AIME (§4, App. H).

## Key Figures/Tables to Study
- **Fig. 2 / Fig. 3:** format shares and pass@1, pass@64, majority@64 during PPO at KL coefficient 0.001 vs 0.01 (150M).
- **Fig. 4 (a)/(b):** 4× vs 8× OpenMathInstruct1 in pretraining changes which format RL selects.
- **Fig. 7:** 1B model on the same mixture as Fig. 2 selects natural language. **Table 1:** MATH-500 before/after PPO. **Tables 4–6:** AIME.

## Technical Details
- **Datasets (§2.1).** Always FineMath-3+ and the Algebraic-Stack subset of Proof-Pile-2 (except the TinyGSM-only runs in §3.3), plus TinyGSM (12.3M problem–solution pairs, code solutions by GPT-3.5), OpenMathInstruct1 (1.8M pairs, code by Mixtral-8x7B), OpenMathInstruct2 (14M pairs, natural language by Llama3.1-405B-Instruct). "k×" means k passes over a dataset. Question–answer pairs are concatenated into the corpus without a chat template.
- **RL (§2.2).** OpenRLHF PPO and GRPO; reward 1 if the answer matches the ground truth, else 0. Expert Iteration (EI): k = 64 generations per GSM8K train problem, keep de-duplicated correct generations, run SFT starting again from the pretrained base each iteration (§2.2, App. F.2).
- **Metrics (App. B.4).** pass@1 with greedy decoding; pass@64 and majority@64 over 64 samples at temperature 0.7. Code outputs are executed; answers are parsed with Math-Verify.
- **Convergence (§3.1).** 150M model on TinyGSM + OMI1 + OMI2: generations shift to TinyGSM format within the first epoch; majority@64 rises by about 5%; pass@64 declines toward the end of training (Fig. 2). KL coefficient 0.01 instead of 0.001 keeps some OpenMathInstruct2-format generations, gives comparable pass@1, and keeps pass@64 stable (Fig. 3). KL coefficient 0 behaves like 0.001 (App. D.1, Fig. 8).
- **Selection is not frequency (§3.2).** With 4× OMI1, the base model produces 62% OMI1-style and 28% TinyGSM-style solutions, yet converges to TinyGSM (Fig. 4a). With 8× OMI1, it converges to OMI1 style, reaches lower accuracy, and degrades near the end; the authors call this a failure mode (Fig. 4b).
- **Repetition (§3.3).** More TinyGSM passes (1, 2, 4, 8×) give higher post-PPO pass@1, pass@64 and majority@64 (Fig. 5). pass@64 rises with TinyGSM amount but does not improve over its initialization during fine-tuning; majority@64 improves 5–10% (App. D.2, Fig. 10).
- **Theory (§3.6).** If π_ref = Σ_i α_i π_i is a mixture of k policies (α_i = mixture weights, π_i = one solution format each), the KL-regularized optimum is π*(y|x) ∝ Σ_i α_i exp(r(x,y)/β) π_i(y|x): the mixture components are re-weighted by reward. The paper writes the KL term with coefficient 1/β.
- **Algorithms (§3.5, App. F).** GRPO with the PPO hyperparameters shows the same convergence but is less stable, with a brief performance collapse before recovery (Fig. 14). EI shows only a mild format shift and underperforms PPO: from 8× TinyGSM, EI stays below 45% GSM8K accuracy after three iterations, while PPO reaches almost 60% (Fig. 20).
- **Transfer (§4, Table 1, 1B, PPO on GSM8K).** MATH-500 pass@1 base → FT: TinyGSM + 4×OMI1 8.60 → 12.60; TinyGSM + OMI2 33.40 → 43.60; OMI2 + MMQA 34.60 → 44.40; TinyGSM 4.80 → 9.60; TinyGSM + OMI1 + OMI2 33.40 → 43.80.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OLMo-codebase models | 150M; 1B | pretrain-stable | width; depth; MLP; activation; positions | 768, 2048; 12, 16 layers; 8× width; SwiGLU; RoPE | arXiv:2504.07912v2 §2.1 | verified 2026-09-14 | no ablation reported |
| Same | 150M; 1B | pretrain-stable | optimizer; peak LR; weight decay | AdamW; 0.001; 0.1 | §2.1 | verified 2026-09-14 | no ablation reported |
| Same | 150M; 1B | pretrain-stable | warmup; decay | linear 5000 steps; cosine to 10% of peak | §2.1 | verified 2026-09-14 | no ablation reported |
| Same | 150M; 1B | pretrain-stable | tokens seen; batch; sequence length; compute | not reported | checked §2.1, App. B, App. C | not reported | — |
| Same | 150M; 1B | pretrain-stable | mixture | FineMath-3+ + Algebraic-Stack + k passes of TinyGSM / OMI1 / OMI2 per run; token shares not reported | §2.1, §3 | verified 2026-09-14 | Fig. 4–5, Table 1 compare mixtures |
| PPO and GRPO runs | 150M; 1B | RL | training batch; rollout batch; samples per prompt | 64; 64; 8 | App. C Table 2 | verified 2026-09-14 | no ablation reported |
| PPO and GRPO runs | 150M; 1B | RL | epochs (definition not given); prompt max; generate max | 10; 1024; 1024 | App. C Table 2 | verified 2026-09-14 | no ablation reported |
| PPO and GRPO runs | 150M; 1B | RL | actor LR; critic LR; warmup; Adam betas | 1e-6; 7e-6; 0.03; (0.9, 0.95) | App. C Table 2 | verified 2026-09-14 | no ablation reported |
| PPO and GRPO runs | 150M; 1B | RL | temperature; clip ε; λ; reward normalization | 0.7; 0.2; 0.95; True | App. C Table 2 | verified 2026-09-14 | no ablation reported |
| PPO and GRPO runs | 150M; 1B | RL | KL coefficient and placement | 1e-3 default, also 0 and 0.01; PPO: token-level KL added to reward; GRPO: KL in loss, k3 estimator | App. C | verified 2026-09-14 | Fig. 3, Fig. 8, Fig. 14 |
| PPO runs | 150M; 1B | RL | prompts | GSM8K train split (§3–4); MATH train split for three 1B models (App. I) | §2.2; App. I | verified 2026-09-14 | App. I Table 7: smaller MATH-500 gains than GSM8K RL |
| EI runs | 150M; 1B | SFT | k; batch; epochs; prompt / generate max; LR; betas | 64; 256; 2; 1024 / 1024; 1e-4; (0.9, 0.95) | App. C Table 3 | verified 2026-09-14 | LR sweep {5e-6, 1e-5, 1e-4, 0.001}: "very marginal gains (1-2%) for other learning rates in the first iteration of EI aside from 1e-4" (App. C) |
| EI runs | 150M; 1B | SFT | iterations | 3 (150M); 2 (1B) | App. F.2 | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality
- **Narrowing.** RL collapses the format mixture to one mode and pass@64 declines late in training (§3.1) or does not improve over initialization (App. D.2). A higher KL coefficient (0.01) keeps a second format and stable pass@64 at comparable pass@1 (Fig. 3).
- **Breadth.** GSM8K RL improves MATH-500 pass@1 in all five 1B mixtures (Table 1). The three mixtures containing OpenMathInstruct2 gain 9.8–10.4 points, TinyGSM + 4×OMI1 gains 4.0 (Table 1). The authors attribute the larger gains to pretraining on synthetic problems resembling MATH and state a benefit of pretraining on data structurally similar to the downstream task (§4; Interpretation).
- **Measurement.** On AIME 2022–2024, pass@1 and majority@64 change by at most 2.22 points (Table 4), while pass@64 rises, for example TinyGSM + OMI2 0.00 → 18.89 (Table 6). AIME 1983–2024 shows larger pass@1 gains, but AIME questions before 2022 have potential contamination with MATH (App. H.2).
- **Offline vs online updates.** The authors hypothesize that EI's slower format shift comes from restarting SFT from the base model each iteration, and that more offline update steps may help keep generation diversity (App. F.2; Interpretation, not tested).
- **Scale.** The paper's scales are 150M and 1B; results at larger scale are not reported.

## Connections
- [[ppo]], [[grpo]]: the two RL algorithms compared; [[openrlhf-ppo]] is the implementation family used.
- [[rest-em]], [[star]]: expert-iteration style methods related to the EI baseline.
- [[openmathinstruct]], [[openmathinstruct-2]]: two of the three format-tagged pretraining datasets.
- [[rlvr-beyond-base-model]]: reports base models matching or exceeding RLVR models on pass@k at large k.
- [[spurious-rewards-rlvr]]: reports RLVR gains under weak or incorrect reward signals on some base models.
- [[prorl]]: argues that prolonged RL with KL control and reference resets expands pass@k beyond the base model.
- [[front-loading-reasoning]]: studies when reasoning data should enter training, the pretraining-side question raised here.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2504.07912 (arXiv v2, 7 Aug 2025, COLM 2025 version; PDF text including App. A–I)
- Corrections to the previous card version: Core Insight "collapses onto and amplifies high-prior behaviors" → the selected format is not the most frequent one at initialization in Fig. 4a (62% vs 28%); for 150M models it is usually the format with the highest base accuracy, with failure cases (§3.2); in Fig. 7 the 1B model selects natural language, the least frequent format and not the most accurate one at initialization (§3.4); "RL post-training is strongly shaped by the support of the pretrained model distribution" → the support argument is attributed in the paper to Wu et al. (2025) (App. A), and the paper's own explanation is re-weighting of a policy mixture (§3.6); "small proxy studies are useful because the mechanism depends on distributional support" → the paper states that small-scale proxies "can elicit interesting insights" without that reason (Abstract).
- Removed as unsupported by the source: "It weakens simplistic narratives that RL creates reasoning from scratch"; "changing the base model's data mixture can materially change what RL later amplifies" as a general claim (replaced by the §3.2 and §3.3 results); "you may need to front-load or mid-train that style into the base model first"; "[[prorl]] is the main counter-position".
- Not reported by the source: pretraining token counts, batch size, sequence length, and compute.
