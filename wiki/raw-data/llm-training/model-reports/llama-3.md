<!-- scope: Llama 3 Herd of Models technical report — full post-training pipeline
     deps: [[README]]
     see-also: [[llama-2]], [[tulu-3]], [[dpo]], [[rejection-sampling-finetuning]]
-->

# The Llama 3 Herd of Models
- **Core Insight:** Six iterative rounds of SFT -> Rejection Sampling -> DPO on increasingly high-quality synthetic data beats a single-pass RLHF pipeline.
- **Guideline:** Treat post-training as a loop: each round collects new preferences, retrains the reward/DPO pair, then regenerates SFT data with the latest model.
- **Authors:** Aaron Grattafiori et al. (Meta Llama Team)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2407.21783
- **Relevant topics:** SFT iterative rounds, rejection sampling, DPO with NLL stabilization, synthetic-data-heavy post-training, reward model for filtering

## Abstract
Modern foundation models of the Llama 3 "herd" natively support multilinguality, coding, reasoning, and tool use. The flagship is a 405B-parameter dense Transformer with a 128K context window. Llama 3 is pre-trained on 15.6T tokens and post-trained via six rounds of SFT + Rejection Sampling + DPO. Empirical evaluations show Llama 3 matches GPT-4 on a large battery of benchmarks. The 8B/70B/405B sizes and accompanying Llama Guard 3 safety classifier are released publicly.

## Key Contributions
- Iterative 6-round post-training recipe: SFT -> Rejection Sampling -> DPO, re-mined every round from current best checkpoint.
- DPO with auxiliary NLL loss (coeff 0.2) on chosen sequences to prevent chosen-logprob collapse.
- Reward-model-gated rejection sampling with K=10–30 samples per prompt as the main SFT data filter.
- Heavy synthetic-data generation for coding, math, multilingual, reasoning, long-context, tool use, and factuality — each capability gets a dedicated synthetic pipeline.
- Llama Guard 3 trained jointly as the safety classifier.
- Full disclosure of failure modes (preference-data noise, multi-turn dialog drift) in the data section.

## Key Figures/Tables to Study
- **Figure 7** (Post-training flow diagram): shows the full round-to-round dataflow and where rejection sampling re-enters the loop.
- **Table 6** (SFT hyperparameters): per-size LR, batch size, epochs.
- **Table 7** (DPO hyperparameters): LR 1e-5, beta 0.1, NLL coefficient 0.2.
- **Section 4.3** (Preference data collection): protocol for chosen/rejected + "significantly better / better / slightly better" granularity labels.
- **Figure 12** (Safety vs helpfulness pareto): shows Llama Guard 3 operating point.

## Technical Details — Post-Training Pipeline

### Overall structure
Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations. Each round uses a fresh batch of ~human preference annotations plus synthetic data resampled from the round-N-1 best model.

### SFT
- **Data sources:** rejection-sampled outputs from prior round (dominant), human-annotated prompts, filtered synthetic data for code/math/reasoning/multilingual/long-context/tool use.
- **Rejection sampling:** for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score.
- **Filtering:** topic classifier + quality classifier (both distilled from Llama 3) remove low-quality rejection-sampled text before SFT.
- **Training:** LR 1e-5 (405B), cosine decay, context 8K-32K (extended), loss on response tokens only.

### Reward Model
- Initialized from the Llama 3 pre-trained checkpoint; linear head replaces LM head.
- Preference data: human annotators rank two responses from different Llama 3 variants with margin labels ("significantly better", "better", "slightly better", "negligibly better").
- RM loss: standard pairwise logistic; no explicit margin term (margin labels only used for data filtering / up-weighting).

### DPO (the policy optimization step)
- **Learning rate:** 1e-5
- **Beta (KL coefficient):** 0.1
- **Auxiliary NLL loss on chosen sequences:** coefficient 0.2 — added to stabilize training by preventing chosen-logprob decay.
- Single epoch per round; masks prompts from loss.
- Most-recent-batch preference data only (older batches cause format drift).

### Data mix (per round)
- ~50–80% synthetic rejection-sampled data
- Remainder: human SFT demonstrations, preference data, capability-specific synthetic (code-exec-filtered code, math with verifier, multi-turn tool use traces, long-context QA).

### Scale
- **Pretraining:** 15.6T tokens, 8K native context, 8-way sequence parallel for long-context extension.
- **405B compute:** 3.8e25 FLOPs.
- **Post-training compute:** not disclosed as a standalone number; post-training is "a small fraction" of pretraining compute.

## Connections
- [[llama-2]] — same conceptual skeleton but Llama 2 used PPO + RSFT; Llama 3 swapped PPO for DPO and added NLL-stabilized objective.
- [[tulu-3]] — open replication uses SFT -> DPO -> RLVR; confirms DPO beta 0.1 as a robust default.
- [[dpo]] — Rafailov 2023 base algorithm; Llama 3's NLL add-on is a novel stabilizer.
- [[rejection-sampling-finetuning]] — RSFT lineage from Llama 2 repurposed as the SFT data generator.
- [[reward-model-overoptimization]] — Llama 3 combats this by swapping RMs each round and never reusing stale preferences.
