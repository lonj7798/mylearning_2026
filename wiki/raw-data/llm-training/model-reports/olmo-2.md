<!-- scope: OLMo 2 technical report — Ai2's fully open 7B/13B/32B models, stability recipe, Dolmino mid-training, Tülu 3 post-training
     deps: [[dolma]]
     see-also: [[tulu-3]], [[rlvr-tulu3]], [[olmo-3]], [[olmo-2-recipe]]
-->

# 2 OLMo 2 Furious
- **Core Insight:** Adding a second, short mid-training stage on a curated mix (Dolmino Mix 1124) raises the average of the development and held-out evaluation suites by 10.6 points for the 7B model and 10.3 points for the 13B model over the end-of-pretraining checkpoints (§4, Table 9).
- **Guideline:** When a pretraining run is finished and the model is weak on knowledge-intensive and math tasks, spend 50B-300B further tokens on a curated high-quality mix with the learning rate decayed linearly to zero, and average three or four such runs over different data permutations, because merging matched or beat the best single run on all six mixes tested (§4, Table 14). When only one mid-training run is affordable, the report gives no measurement of the loss from skipping the merge.
- **Authors:** Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, et al. (OLMo Team, Allen Institute for AI; University of Washington; New York University)
- **Year:** 2025 (arXiv v1 2025-01; v3 dated 2025-10-08)
- **URL:** https://arxiv.org/abs/2501.00656
- **Source type:** official technical report
- **Relevant topics:** open pretraining, training stability, QK-norm, z-loss, mid-training curriculum, model merging, Tülu 3 post-training, RLVR

## Abstract
OLMo 2 is a family of dense autoregressive models at 7B, 13B, and 32B scale released with weights, full training
data, training code and recipes, training logs, and thousands of intermediate checkpoints. The report describes
architecture and recipe changes aimed at training stability and per-token efficiency, introduces Dolmino Mix 1124
as a late-stage curriculum mix, and applies the Tülu 3 post-training recipe with an extended RLVR stage. The base
models are reported to sit on the Pareto frontier of benchmark performance against pretraining FLOPs relative to
the open-weight Llama 3.1, Qwen 2.5, and Gemma 2 families; the instruct models are reported as competitive with
open-weight models of comparable size and with GPT-3.5 Turbo and GPT-4o Mini (Abstract; §1).

## Key Contributions
- A stability recipe: filtering repeated n-grams, a fixed-variance initialization (mean 0, std 0.02), RMSNorm,
  reordered normalization applied to block outputs, QK-norm, z-loss, and RoPE θ = 500,000 (§2.1, §3).
- A two-stage base recipe in which pretraining takes 90-95% of training FLOPs and mid-training takes 5-10% (§2.3).
- Dolmino Mix 1124, a 2.19T-token curated pool from which 50B, 100B, and 300B token samples are drawn (§4, Table 5).
- Mid-training checkpoint averaging ("souping") as a standard step rather than an ablation (§4, Table 14).
- Micro-annealing: assessing a candidate mid-training source by a short anneal from a fixed checkpoint (§4).
- Application of the Tülu 3 recipe (SFT, DPO on on-policy preferences, RLVR) with permissively licensed data and a
  multi-round RLVR stage for the 13B model (§5).

## Key Figures/Tables to Study
- **Figure 1** — benchmark average against approximate pretraining FLOPs (6 × tokens × parameters); the Pareto
  claim rests on this figure.
- **Figure 2** — loss and gradient-norm curves for OLMo-0424 against OLMo 2, the evidence for the stability changes.
- **Table 3** — pretraining hyperparameters per size, including the truncated cosine schedules.
- **Table 9** — pretraining versus pretraining-plus-mid-training evaluations for 1B, 7B, 13B, 32B.
- **Table 7** — OLMo 2-Instruct against open-weight and closed models on the Tülu 3 evaluation suite.
- **Table 18** — the full PPO hyperparameter set used for RLVR at 7B and 13B.

## Technical Details
- Tokens seen: 7B 4.05T total (3.90T in pretraining), 13B 5.6T total (5T pretraining), 32B 6.6T total (6.06T
  pretraining) (§2.3).
- OLMo 2 Mix 1124 is approximately 3.90T tokens, over 95% web-derived, drawn from DCLM baseline 1.0, Dolma 1.7
  (arXiv, OpenWebMath, Algebraic Stack, peS2o, Wikipedia), and StarCoder (§2.4.1, Table 4).
- Dolmino Mix 1124 pool totals 2.19T tokens; the 100B sample is 99.95B tokens (§4, Table 5 and Table 13).
- Sequence length is 4096 for all three sizes (Table 3). No long-context extension stage is reported.
- Attention is multi-head at 7B (32/32) and 13B (40/40) and grouped-query at 32B (40/8) (Table 3).
- Tokenizer is built from the cl100k vocabulary with Dolma masking tokens added; at 1B scale trained on 100B
  tokens it scores 60.6 OLMES against 59.8 for the OLMo 1 tokenizer (§2.2, Table 2).
- Relative benefit of mid-training falls with size: +37.0% for 1B, +18.7% for 7B, +15.9% for 13B, +12.3% for 32B
  on the evaluation average (App., discussion of Table 9).
- Higher pretraining learning rates improve training loss early but are overtaken later, and decaying to zero over
  50B or 100B tokens removes the difference; average scores across variants vary by less than two points
  (§4.1, Table 8 and Figure 11).
- Instruct results on the report's own suite: OLMo 2 32B Instruct average 68.8, OLMo 2 13B 63.5, OLMo 2 7B 56.5,
  against GPT-4o Mini 0724 at 65.7 and GPT-3.5 Turbo 0125 at 60.5 (Table 7).
- Pretraining energy and emissions: 7B used 131 MWh with 52 t CO2; 13B used 257 MWh with 101 t CO2 (Table 19).
  GPU-hours are not reported.
- Hardware: a cluster of 1,024 NVIDIA H100 GPUs at 80GB HBM3, plus Google Cloud A3 Mega VMs with 8 H100 each
  (§6.1-§6.2).

## Recipe ledger
The recipe table is in [[olmo-2-recipe]] (pretraining, mid-training, SFT, DPO, and RLVR rows with loci).

## Findings relevant to generality
- Mid-training gains are largest at the smallest size, and the report states that below some size the pretraining
  recipe may need task-specific data or distillation to reach non-random accuracy on multiple-choice tasks; OLMo 2
  1B after stage 1 scores at chance on MMLU and ARC Challenge (App., discussion of Table 9 and Table 23).
- Removing all multilingual data (the Aya split and multilingual WildChat) from the SFT mix cost about 0.5 points
  on the evaluation average, which the report reads as evidence that the Tülu 3 mixture is balanced and not
  improved by dropping apparently irrelevant subsets (§5).
- GSM8K was only partially held out: 200 of 1319 examples were used during mid-training data development and
  results are reported on the remaining 1119 (§3 footnote 6).
- Code was excluded as a target skill, so the instruct evaluation drops the Tülu 3 code category (§5).
- Nothing is reported on long-context training or on agentic or tool-use training.

## Connections
- [[tulu-3]] — the post-training recipe OLMo 2 adapts, with permissive data and a multi-round RLVR stage.
- [[rlvr-tulu3]] — the RLVR method whose PPO setup Table 18 instantiates.
- [[dolma]] — the earlier Ai2 corpus; parts of Dolma 1.7 enter OLMo 2 Mix 1124.
- [[olmo-3]] — the successor, which adds long-context extension and replaces PPO with a GRPO-based stage.
- [[task-scaling-model-ladders]] — uses the OLMo 2 sizes and token budgets as prediction targets.
- [[olmo-2-recipe]] — the full recipe ledger.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2501.00656 (arXiv v3, 8 Oct 2025).
- Corrections to the previous card version:
  - "Compute-optimal scaling plot across 7B/13B/32B" among the key figures → Figure 1 plots benchmark average
    against approximate pretraining FLOPs and claims a Pareto frontier against other released models; the report
    contains no scaling-law fit.
  - "1000+ intermediate checkpoints" → the abstract says "thousands of intermediate checkpoints".
  - "OLMo-Mix-1124 (3.9T tokens)" as 90%+ of budget with a "~50B token" cooldown → 3.90T is the stage-1 mix size;
    the mid-training sample is 50B for 7B and 100B or 300B for 13B and 32B (§2.3, Table 9 caption).
  - "Context: 4K native, extended to 32K in cooldown" → sequence length is 4096 in every stage (Table 3); no
    context extension is reported.
  - "SFT: ~939K prompts from Tülu 3" for all sizes → 939,104 prompts for the 7B and 13B mix and 866,138 for the
    1B and 32B mix (§5).
  - "RLVR: PPO ... LR 3e-7, beta KL 0.05" → 3e-7 is the 13B learning rate and 4e-7 the 7B; β is 0.05 at 7B and
    0.1 or 0.03 at 13B depending on the round; the 32B uses GRPO at 5e-7 with β 0.1 (Table 18, §5).
  - "Hyperparameters inherit from Tülu 3" → the report states hyperparameters were re-tuned and that OLMo 2
    required higher learning rates than the Llama 3.1 recipe of Tülu 3 (§5).
  - "32B variant is the first fully-open model to beat GPT-3.5 and GPT-4o-mini on average benchmarks" → on the
    report's own suite OLMo 2 32B Instruct averages 68.8 against 65.7 for GPT-4o Mini and 60.5 for GPT-3.5 Turbo
    (Table 7); no priority claim is made.
  - "Two-stage pretraining curriculum: 90%+ of budget" → pretraining is 90-95% of training FLOPs and mid-training
    5-10% (§2.3).
  - Author list "Pete Walsh, Luca Soldaini, Dirk Groeneveld, et al." → extended to the first six listed authors.
- Removed as unsupported by the source: "7B: ~460K H100 GPU-hours pretraining" and "13B: ~1.9M H100 GPU-hours"
  (no GPU-hour figure appears; Table 19 reports MWh); "Architecture ablation table: which stability trick
  contributes which fraction of the spike-free runs" (no such table; §3 reports curves and per-change discussion);
  "RLVR stage lifts GSM8K and MATH consistently for 7B and 13B (single-digit pp gains)" (the 13B needed two extra
  RLVR rounds after GSM8K fell, Figure 13); "DPO stage contributes most of the chat-quality / IFEval lift" (no
  such attribution is made); "improved initialization preserving activation scale" (the change is to a fixed
  std 0.02 initialization, §3.2); "OLMo-Mix-1124 supersedes Dolma 1.7" (Dolma 1.7 subsets are used inside it).
- Not reported by the source: GPU-hours; post-training compute broken out; any long-context or tool-use training.
