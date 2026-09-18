<!-- scope: head-to-head comparison of AI-generated and human preference labels for RLHF, plus a reward-model-free variant
     deps: [[constitutional-ai]]
     see-also: [[judge-llm-bias]], [[bradley-terry-rm]], [[rlcd]]
-->

# RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
- **Core Insight:** With PaLM 2 XS policies, human evaluators prefer RLAIF over the SFT baseline 71% of the time on summarization and 63% on helpful dialogue, against 73% and 64% for RLHF, and RLAIF vs RLHF win rates of 50% and 52% are not statistically distinguishable from parity (Table 1, §4.1).
- **Guideline:** When human preference labels are the bottleneck and a capable off-the-shelf labeler is available, label pairs with a zero-shot chain-of-thought prompt and average two inferences with the candidate order swapped, because few-shot exemplars reduced labeler alignment on summarization and helpfulness (Table 2, §4.4) and position bias is larger for smaller labelers (Table 4, App. B).
- **Authors:** Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, et al. (Google DeepMind and Google)
- **Year:** 2023 (arXiv v1 2023-09; ICML 2024; arXiv v3 2024-09)
- **URL:** https://arxiv.org/abs/2309.00267
- **Source type:** paper
- **Relevant topics:** RLAIF, AI preference labeling, direct-RLAIF, reward-model staleness, position bias, labeler scale

## Abstract
The paper compares reinforcement learning from AI feedback with reinforcement learning from human feedback on
three tasks: summarization, helpful dialogue generation, and harmless dialogue generation. AI preference labels
are produced by prompting an off-the-shelf LLM and reading the softmax over the log-probabilities of the tokens
"1" and "2" (§2.1), giving a soft label; a reward model is then trained on those soft labels with a cross-entropy
loss (§2.2.1) and used for RL. RLAIF reaches win rates comparable to RLHF on all three tasks (§4.1). The paper
also shows that RLAIF improves over the SFT baseline when the AI labeler is the same size as the policy, and
when it is the same checkpoint as the initial policy (§4.2, §4.3). It introduces direct-RLAIF (d-RLAIF), which
skips reward-model training by scoring each response with the off-the-shelf LLM during RL (§2.2.2).

## Key Contributions
- **Parity with RLHF on human evaluation** for summarization, helpful dialogue, and harmless dialogue (Table 1, §4.1).
- **Direct-RLAIF (d-RLAIF):** the off-the-shelf LLM is prompted to output a quality score from 1 to 10; the
  likelihood of each score token is normalized to a distribution, a weighted score s(y|x) = Σ_{i=1..10} i·P(i|y,x)
  is computed, and that score is normalized to [−1, 1] and used as the RL reward, removing the reward model and
  its staleness as the policy drifts (§2.2.2).
- **Same-size labeling:** with PaLM 2 XS as both labeler and policy family, same-size RLAIF is preferred over SFT
  68% of the time on summarization (Table 1, §4.2). On helpful dialogue the labeler and the initial policy are the
  same checkpoint, which the authors call a strict case of self-improvement (§4.3).
- **Prompting ablation:** chain-of-thought reasoning generally improves AI labeler alignment, while few-shot
  exemplars help only on harmless dialogue (Table 2, §4.4).
- **Labeler-scale measurement:** AI labeler alignment with human preferences rises with labeler size (Table 3, §4.5).
- **Cost estimate:** AI preference labeling is estimated at over 10× less costly than human labeling (§4.1, App. L).

## Key Figures/Tables to Study
- **Table 1 (§4.1):** all six win-rate and harmless-rate comparisons.
- **Table 2 (§4.4):** AI labeler alignment for 12 prompt variants on the three tasks.
- **Table 3 (§4.5):** labeler size against alignment.
- **Table 4 (App. B):** position bias by labeler size.
- **Table 5 (App. G):** reward-model pairwise accuracy for human-feedback and AI-feedback RMs.

## Technical Details
- **Win rates vs SFT (human evaluation):** RLAIF 71% summarization / 63% helpful; RLHF 73% / 64%; same-size RLAIF
  68% / not run; d-RLAIF 74% / 66% (Table 1, §4.1–§4.3). RLAIF vs RLHF is 50% / 52%, not significantly different
  from 50% (§4.1). d-RLAIF vs same-size RLAIF is 60%, significant by binomial test (§4.3).
- **Harmless rate (human evaluation):** SFT 64%, RLHF 76%, RLAIF 88% (Table 1, §4.1).
- **Labeler alignment by size (summarization):** PaLM 2 L 78.0%, PaLM 2 S 73.8%, PaLM 2 XS 62.7% (Table 3, §4.5).
  The paper reports human inter-annotator agreement on the same preference dataset as 73–77% (§4.5).
- **Prompt ablation (summarization / helpful / harmless alignment):** Base 0-shot 76.1 / 67.8 / 69.4;
  Base + CoT 0-shot 77.5 / 69.1 / 70.6; Detailed + CoT 0-shot 78.0 / 67.8 / 70.1; Detailed 8-shot 69.8 on
  summarization. Best prompts beat Base 0-shot by +1.9, +1.3, and +1.7 points (Table 2, §4.4).
- **Position bias:** the labeler prefers the same displayed position after swapping order for 18% (PaLM 2 L),
  21% (S), and 56% (XS) of pairs; the mitigation is to run both orders and average the distributions
  (Table 4, App. B; §2.1.1).
- **Reward-model accuracy on held-out human preferences:** human-feedback / AI-feedback RM = 79.3% / 74.2%
  (summarization), 76.0% / 67.8% (helpful), 72.1% / 69.7% (harmless) (Table 5, App. G).
- **Cost:** estimated $0.06 per AI-labeled example against $0.67 per human-labeled example, a factor above 10×;
  the estimate excludes annotator training and LLM setup costs (App. L).
- **Datasets:** filtered Reddit TL;DR with 123k posts and ~5% held out, plus OpenAI's 92k human comparisons;
  Anthropic helpful-base and harmless-base, each over 40k train and 2k test examples (App. C).
- **Negative results reported:** combining human and AI feedback did not improve over human feedback alone
  (§4.1, App. K); self-consistency strictly degraded labeler alignment (§4.4, App. M); in-context exemplars
  monotonically decreased alignment on summarization and helpfulness (§4.4).

## Recipe ledger
Moved to **[[rlaif-scaling-recipe]]** to keep this card under 120 lines.

## Findings relevant to generality, negative feedback, distillation
- **Distillation:** the paper states that training a reward model on AI labels can be viewed as a form of model
  distillation (§2.2.1). d-RLAIF removes that distillation step and beats same-size RLAIF 60% of the time (§4.3),
  which the authors attribute to querying the labeler directly and to avoiding reward-model staleness.
- **Negative feedback:** negatives here are the dispreferred side of a soft pairwise label; they enter only through
  the Bradley-Terry-style cross-entropy loss on the reward-model scores (§2.2.1, App. A.2). The paper does not
  study unlikelihood training or negative-advantage policy gradients.
- **Generality:** the reported gains are per-task, with separate policies for summarization, helpful dialogue, and
  harmless dialogue; the paper does not evaluate transfer to held-out tasks. It reports that RLAIF and RLHF
  policies are longer than SFT outputs and that a length-controlled post-hoc analysis still favors them (App. J).

## Connections
- Follows the AI-feedback setup introduced in **[[constitutional-ai]]**, which the paper cites as the first RLAIF effort.
- The labeler failure modes it measures and mitigates overlap with **[[judge-llm-bias]]** (position bias in particular).
- The reward-model loss is the Bradley-Terry objective of **[[bradley-terry-rm]]**, applied to soft labels.
- **[[rlcd]]** re-implements this RLAIF setup as a baseline and reports it performing poorly at 7B scale.
- The KL term applied inside the policy-gradient loss connects to **[[kl-control-rlhf]]**.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2309.00267 (arXiv v3, 3 Sep 2024)
- Corrections to the previous card version:
  - "roughly 70/30 win-rate parity" → the reported numbers are RLAIF vs SFT 71% / 63% and RLHF vs SFT 73% / 64%,
    with RLAIF vs RLHF at 50% / 52% (Table 1, §4.1).
  - "d-RLAIF reward: r(x,y) = log P_labeler('Better' token | …); applied at end of sequence" → d-RLAIF prompts for a
    1–10 quality score, normalizes the score-token likelihoods, computes Σ i·P(i|y,x), and rescales to [−1,1] (§2.2.2).
  - "PPO setup: otherwise vanilla InstructGPT recipe, including the per-token KL penalty" → RL uses REINFORCE with a
    learned value baseline, γ = 1, reward only on the final token, and the KL term folded into the policy-gradient
    loss with β = 0.05 (App. E, App. A.3, App. F).
  - "AI labels are ~100× cheaper per preference than crowd-source labels" → the paper estimates over 10× cheaper,
    $0.06 vs $0.67 per example (§4.1, App. L).
  - "Fig. 4 (CoT vs direct preference prompt) — CoT adds ~3–5 pp win rate" → the CoT comparison is Table 2 and the
    metric is AI labeler alignment, not win rate; the best prompt beats Base 0-shot by +1.9 / +1.3 / +1.7 points (§4.4).
  - "Fig. 6 (labeler size vs RLAIF quality)" → Table 3 reports labeler *alignment* by size (78.0 / 73.8 / 62.7%);
    no win rate is measured as a function of labeler size (§4.5).
  - "d-RLAIF outperforms RLAIF" stated without scope → d-RLAIF (74%) is compared against same-size RLAIF (68%) on
    summarization, and the two use different labelers from the main RLAIF run (§4.3).
  - "Label calibration: BT RMs trained on soft labels outperform hard-label RMs" → the paper trains on soft labels
    because the method produces them (§2.2.1); no soft-vs-hard reward-model comparison is reported.
- Removed as unsupported by the source: the claimed soft-vs-hard label superiority; the claim that RLAIF quality
  improves monotonically with labeler size; the "~100×" cost figure; the InstructGPT/PPO recipe attribution.
- Not reported by the source: policy and labeler parameter counts (PaLM 2 XS/S/L sizes are not disclosed);
  number of RL prompts or rollouts per step; held-out-task or contamination evaluation.
