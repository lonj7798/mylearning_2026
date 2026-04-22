<!-- scope: HuggingFace DPO Zoo — survey of DPO variants
     deps: [[README]]
     see-also: [[dpo]], [[ipo]], [[kto]], [[simpo]], [[orpo]]
-->

# HuggingFace — Preference Optimization Zoo (DPO variants survey)
- **Core Insight:** DPO's core trick — swap the RLHF pipeline for a closed-form contrastive loss — has spawned a family of variants (IPO, KTO, SimPO, ORPO, BCO) each trading off assumptions in the Bradley-Terry derivation for different practical benefits.
- **Guideline:** Start with plain DPO (beta ~ 0.1); move to IPO if you see chosen-logprob collapse; move to KTO if you only have unary labels; move to SimPO / ORPO if you want reference-free objectives.
- **Author:** HuggingFace team (Kashif Rasul, Younes Belkada, Lewis Tunstall, and contributors)
- **Year:** 2024 (running updates)
- **URL:** https://huggingface.co/blog/pref-tuning (and TRL docs)
- **Relevant topics:** DPO, IPO, KTO, SimPO, ORPO, BCO, CPO, preference optimization variants, reference-free objectives

## Summary
HuggingFace's preference-tuning content (blog posts + TRL documentation) forms an evolving "zoo" cataloging DPO variants. Each variant modifies the DPO loss to address a specific failure mode observed in practice. Plain DPO's failure modes include (a) chosen-logprob collapse under some configurations, (b) sensitivity to reference model choice, (c) dependence on well-balanced pairwise data. The zoo posts walk through IPO (Azar, identity-preference), KTO (Ethayarajh, Kahneman-Tversky unary), SimPO (Meng, reference-free), ORPO (Hong, odds-ratio SFT-joint), BCO (binary classifier-like), CPO (contrastive preference), and provide TRL recipes for each.

## Key Contributions
- One-stop TRL-compatible implementations of all major DPO variants.
- Comparison table of assumption/loss/hyperparameter per variant.
- Worked examples on UltraFeedback / Anthropic HH data.
- Per-variant guidance on when it's the right choice.

## Key Figures/Tables to Study
- **Variant comparison table** (DPO / IPO / KTO / SimPO / ORPO / BCO) showing loss formula and key assumption.
- **Chosen-vs-rejected logprob curves** during training under each variant (showing which ones avoid chosen-logprob collapse).
- **Benchmark table** on MT-Bench / AlpacaEval across variants at matched training time.

## Technical Details

### DPO (baseline)
Loss: `-log sigma(beta * (log pi/pi_ref(y_w) - log pi/pi_ref(y_l)))`
- Requires reference model at train time.
- beta ~ 0.1 typical; too-low beta causes collapse.

### IPO (Identity Preference Optimization, Azar 2023)
- Replaces the sigmoid-log-odds DPO loss with a squared-loss formulation.
- Explicitly regularizes the log-ratio difference toward a target margin.
- Prevents the chosen-logprob collapse that plain DPO exhibits at scale.

### KTO (Kahneman-Tversky Optimization, Ethayarajh 2024)
- Uses unary labels ("is this response good or bad?") instead of pairwise.
- Loss inspired by prospect theory's value function (loss aversion asymmetry).
- Useful when binary feedback is cheaper than paired comparisons.

### SimPO (Simple Preference Optimization, Meng 2024)
- Reference-free: replaces `log pi/pi_ref` with length-normalized `log pi / |y|`.
- Adds a target reward margin.
- Simpler, no reference model needed at train time; competitive with DPO.

### ORPO (Odds-Ratio Preference Optimization, Hong 2024)
- Joint SFT + preference objective: one training phase instead of two.
- Odds-ratio term penalizes rejected completions while the NLL term on chosen still drives SFT.

### BCO (Binary Classifier Optimization)
- Treats preference learning as binary classification (good/bad on a single response).
- Direct classifier-like training with Platt scaling to extract a scalar reward.

### When to use which (HF guidance)
- Paired high-quality preferences -> DPO or IPO.
- Unary labels only -> KTO.
- Want to skip reference model -> SimPO.
- No separate SFT stage -> ORPO.
- Limited data -> try several, measure on held-out.

### TRL implementation consistency
All variants are exposed through `trl.DPOTrainer` with a `loss_type` parameter; this is the practical artifact the zoo blog enables.

## Connections
- [[dpo]] — baseline.
- [[ipo]], [[kto]], [[simpo]], [[orpo]] — algorithm-specific raw pages.
- [[trl-ppo]] / [[trl-grpo]] / [[trl-online-dpo]] — sibling HF TRL trainers.
- [[hf-rlhf-illustrated]] — the pre-zoo illustrated-RLHF post.
- [[costa-huang-ppo-details]] — complementary PPO-side reproducibility reference.
