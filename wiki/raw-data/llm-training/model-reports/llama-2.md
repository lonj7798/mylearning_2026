<!-- scope: Llama 2 paper — RSFT + PPO iterative RLHF with dual reward models
     deps: [[README]]
     see-also: [[llama-3]], [[ppo]], [[rlhf-instructgpt]]
-->

# Llama 2: Open Foundation and Fine-Tuned Chat Models
- **Core Insight:** Separate reward models for helpfulness and safety avoids the single-RM tradeoff, and Rejection-Sampling Fine-Tuning (RSFT) is a lightweight replacement for early PPO iterations.
- **Guideline:** Run RSFT for the first RLHF iterations, add PPO only in the last round; train two RMs and combine them with a piecewise schedule (safety dominates on safety prompts, helpfulness dominates elsewhere).
- **Authors:** Hugo Touvron et al. (Meta GenAI)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.09288
- **Relevant topics:** Iterative RLHF (V1..V5), Rejection-Sampling Fine-Tuning, dual reward models, margin labels, safety fine-tuning

## Abstract
Llama 2 is Meta's 7B/13B/70B open foundation family. Pre-trained on 2T tokens. Llama 2-Chat is produced via SFT -> iterative RLHF. The RLHF pipeline produces five successive checkpoints (RLHF-V1 through RLHF-V5); early checkpoints are trained with Rejection-Sampling Fine-Tuning only, and PPO is added in the last two rounds. Two separate reward models — one for helpfulness, one for safety — are combined to score completions, resolving the tension Anthropic documented between these objectives.

## Key Contributions
- Dual reward model: Helpfulness RM + Safety RM, both initialized from the LM base with a linear regression head replacing the LM head.
- Iterative RLHF: five successive checkpoints (V1..V5) with fresh weekly batches of human preferences.
- Rejection-Sampling Fine-Tuning (RSFT): sample K completions per prompt, keep best by RM score, SFT on the top sample. Used for V1..V3.
- PPO used only in V4 and V5 on top of the RSFT checkpoints.
- Margin-label preference collection: annotators pick "significantly better / better / slightly better / negligibly better" on top of binary preferences.
- Context distillation for safety: augment safety prompts with a safety preamble at training time to improve robustness.

## Key Figures/Tables to Study
- **Figure 4 (RLHF pipeline):** the iterative V1..V5 diagram.
- **Figure 7 (dual-RM scoring):** how Helpfulness-RM and Safety-RM outputs are combined.
- **Table 11 (RM accuracy):** Llama 2 Helpfulness RM matches GPT-4 as preference judge.
- **Margin-label table:** per-margin distribution in the preference set.

## Technical Details — Post-Training Pipeline

### SFT
- **Size:** 27,540 high-quality SFT examples (after the Meta team found quality >> quantity past ~10K).
- Carefully curated; heavy filtering over vendor-supplied data.
- Trained 2 epochs; LR 2e-5 (70B), cosine decay.

### Preference data
- **Volume:** ~1.4M binary preferences collected across the RLHF iterations (grew each week).
- **Margin labels:** "significantly better / better / slightly better / negligibly better" — used as a margin term in the RM loss.
- **Annotators:** blended internal + vendor; per-prompt two candidate responses from different Llama 2 variants.

### Reward models (two separate)
- **Initialization:** each RM initialized from the LM pre-trained base (70B RM is best).
- **Head:** linear regression layer replacing LM head.
- **Loss:** pairwise logistic + margin term scaled by margin-label severity.
- **Helpfulness RM** trained on helpfulness prompts; **Safety RM** trained on safety/red-team prompts.
- At RLHF scoring time, a rule selects which RM (or a weighted combo) scores each prompt.

### RLHF algorithms
- **V1..V3:** Rejection-Sampling Fine-Tuning (RSFT). For each prompt, sample K outputs (K ~ 10+), score with combined RMs, SFT on the best sample. No policy-gradient.
- **V4, V5:** PPO added on top of RSFT checkpoint.
  - **Learning rate:** 1e-6 (policy) for 70B.
  - **KL coefficient beta:** 0.01.
  - **Batch size:** 512.
  - **Sequence length:** 4K.
  - Standard PPO with clipped ratio, value function, GAE.

### Safety fine-tuning
- Dedicated safety SFT data.
- Context distillation: at training time, prefix safety prompts with a "you are a safe assistant" preamble, then distill the preambled behavior into the unpreambled model.
- Red-teaming across 350+ adversaries.

### Scale
- **Pretraining:** 2T tokens.
- **RLHF preference data:** ~1.4M pairs accumulated.
- **70B post-training compute:** not fully itemized; RLHF takes weeks of weekly iterations.

## Connections
- [[llama-3]] — successor swaps PPO for DPO and grows to 6 rounds; keeps the iterative philosophy.
- [[rlhf-instructgpt]] — Ouyang 2022 single-RM baseline Llama 2 generalizes to two.
- [[ppo]] — used in V4/V5.
- [[rejection-sampling-finetuning]] — RSFT lineage Llama 2 popularized.
- [[constitutional-ai]] — Anthropic's safety-vs-helpfulness approach Llama 2 cites as motivation for dual RMs.
