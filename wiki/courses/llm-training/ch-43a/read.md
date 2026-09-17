<!-- chapter: ch-43a
     track: rl
     kind: content
     title: Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages
     deps: [ch-43, ch-39, ch-31a]
     sources: [[learning-dynamics-llm-finetuning]], [[likelihood-displacement]], [[gradient-entanglement-margin-alignment]],
              [[raft-reinforce-rej-minimalist]], [[negative-sample-reinforcement]], [[nft-negative-aware-finetuning]],
              [[lazy-likelihood-displacement-grpo]], [[asymmetric-reinforce]], [[topr-tapered-off-policy-reinforce]],
              [[negative-preference-optimization]], [[on-policy-suboptimal-preference-data]], [[pmpo-positive-negative-feedback]],
              [[ngrpo]], [[bcpg-nsa]], [[why-language-models-hallucinate]], [[dapo]], [[entropy-mechanism-llm-rl]]
     figures: figures/negative-gradient-lab.html
     revised: 2026-09 (generality revision)
-->

# Chapter 43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages

> **Core insight.** An explicit negative gradient does not remove probability from a model; it moves probability to whatever the model already ranks highest at that position, so its effect depends on how probable the penalized sample was. On-policy negatives, where the penalized sample is by construction one the current model produced, preserved output diversity in the one study that isolates them: with Qwen2.5-Math-7B trained on MATH, negative-only training kept MATH pass@256 at 96.9 against 96.9 for the base model and 91.2 for positive-only training, while positive-only training raised the share of correct training samples fastest and lost the most entropy ([[negative-sample-reinforcement]] Table 1, Figures 5b-d). Off-policy negatives, where the penalized sample is unlikely under the current policy, concentrate mass on the already-most-likely continuation ([[learning-dynamics-llm-finetuning]] §3.3) and can lower the likelihood of the responses training is supposed to promote ([[likelihood-displacement]] §6.2; [[gradient-entanglement-margin-alignment]] §3.1.1). Where the split has been measured on a generating model, positives still carry most of the improvement: 80% against 20% in the NFT 32B run ([[nft-negative-aware-finetuning]] §5.3).
>
> **Guideline.** When negatives are on-policy samples labeled by a verifier, keep them in the loss and down-weight the positive term instead of dropping either side, because λ = 0.1 on the positive term matched the best pass@1 (76.6 on MATH) while keeping pass@256 at 96.7 against 92.0 for equal weights ([[negative-sample-reinforcement]] Tables 1 and 4). When rollouts come from a stale policy, bound the negative term by its importance ratio (truncate at 1 with no lower bound, [[topr-tapered-off-policy-reinforce]] §3) or lower the baseline below the mean reward ([[asymmetric-reinforce]] §5.3), because an untruncated negative term has no lower bound: it produced degenerate generations in the off-policy GSM8K runs of [[topr-tapered-off-policy-reinforce]] (§2.1, Figure 1) and train and test collapse in all 7 Llama-3.1-8B-Instruct MATH runs at δV = 0 in [[asymmetric-reinforce]] (§5.2, Fig. 5 left). When the negative and the positive are the same kind of response, such as two refusals or two traces that share most tokens, filter the pairs before training or localize the penalty to the differing tokens, because the shared tokens are what couples the two gradients ([[likelihood-displacement]] §6.3; [[gradient-entanglement-margin-alignment]] §4.3; [[lazy-likelihood-displacement-grpo]] §5). When a failure label may be wrong — a parser miss, a truncation, an unreachable test — mask the sample instead of penalizing it, because masking truncated samples alone moved AIME 2024 avg@32 from 30 to 36 in the DAPO ablation ([[dapo]] Table 1).

## Why this chapter matters for a general-purpose model

Every post-training stage produces failures, and each stage decides what to do with them. [[ch-31a]] covers the supervised options: discard, place the failure in the input, condition on a label, or add a bounded negative term. [[ch-39]] covers the rejected term of DPO and its variants. [[ch-40]] covers group-baseline RL, where a below-average sample receives a negative advantage without anyone deciding to write a negative term. This chapter holds the derivations those chapters point to, and states what is measured about generality.

Three properties of a general-purpose model are at stake.

1. **Coverage.** pass@k at large k measures how many distinct correct solutions a model can still produce. Positive-only reinforcement raised pass@1 and lowered pass@256 below the base model in the Qwen2.5-Math-7B runs of [[negative-sample-reinforcement]] (Table 1), which is the same direction reported for RLVR in general by [[rlvr-beyond-base-model]].
2. **Retention.** A negative gradient applied to an unlikely sample changes the whole distribution at that position, not only the penalized token ([[learning-dynamics-llm-finetuning]] §3.3), so it can remove behaviour that was never labeled, including refusals ([[likelihood-displacement]] §6.2).
3. **Calibration and abstention.** A binary verifier scores "I don't know" exactly as it scores a wrong answer, so a negative gradient trained against it pushes down abstention together with error ([[why-language-models-hallucinate]] §4.1).

Pipeline position: preference optimization and RL, with back-references to SFT. The measurement tools are pass@k at large k, entropy on a held-out set, chosen and rejected log-probabilities logged separately, and statistics split by advantage sign.

## §1 What "negative gradient" means and where the sign comes from

[[ch-31a]] §1 defines four meanings of "negative": (1) negative marginal value, a sample dropped by a filter; (2) negative as content, a failure placed in the input or in a corrected target; (3) negative as conditioning, a failure trained under a label not used at inference; (4) negative as gradient, an explicit decrease of the sample's likelihood. **This chapter is about meaning (4) only.** Meanings (1)-(3) do not move probability away from the failure and have no displacement or squeezing risk.

In RL the sign is not chosen by hand; it comes from the advantage. For a group of G responses to one prompt with rewards r_i, GRPO's advantage is Â_i = (r_i − mean(r))/std(r), assigned to every token of response i ([[grpo]]; [[dr-grpo]] §3.1). Three consequences:

- With a binary verifiable reward, the group mean lies between the two reward values, so correct responses get positive advantage and incorrect ones get negative advantage. The sign of a sample's contribution is its correctness, and it cannot flip because of batch composition ([[negative-sample-reinforcement]] §C).
- With a learned reward model the sign is relative: the same response can receive a positive advantage in a weak batch and a negative one in a strong batch ([[negative-sample-reinforcement]] §C). Diagnostics split by advantage sign therefore mean something different under RLVR than under RLHF.
- When every response in a group gets the same reward, the advantage is zero for all of them and the group contributes no gradient ([[dapo]] §3.2). Whether such groups are dropped, kept, or re-weighted is a design decision covered in §5.

The term **negative-only training** below means the positive term is removed, not that the model receives negative rewards only; **push-down** means a gradient step that decreases log π of a sampled token or response.

## §2 The softmax update: where the removed probability goes

**Definition.** At one position the model produces logits z over the vocabulary and probabilities p_j = exp(z_j)/Σ_i exp(z_i). A policy-gradient step on a sampled token y with advantage A and step size η changes the logits by

```
Δz_j = η · A · (1[j = y] − p_j)
```

- z_j: logit of token j; p_j: its current probability; 1[·]: 1 if the condition holds, else 0; y: the sampled token; A: the advantage of the response containing it; η: step size.
- For A < 0 the sampled token's logit falls by η|A|(1 − p_y) and **every other logit rises by η|A|p_j**, that is, in proportion to the probability the model already assigns. This is the same expression as the cross-entropy gradient of [[ch-31a]] §2 with the sign flipped.

**The problem, stated measurably.** After a push-down, the penalized token loses probability. The question a generality-minded reader has to answer is which tokens gain it, and whether the resulting distribution is flatter (exploration preserved) or sharper (exploration reduced).

**Worked example.** Five tokens with logits (2, 1, 0, −1, −2) give p = (0.6364, 0.2341, 0.0861, 0.0317, 0.0117) and entropy H = 1.0000 nats. Apply one step with |A| = 1 and η = 0.5:

| Update | Sampled token | New p | ΔH (nats) |
|---|---|---|---|
| Negative on the top token | p_y = 0.6364 | (0.5720, 0.2837, 0.0969, 0.0347, 0.0126) | +0.0751 |
| Negative on a rare token | p_y = 0.0317 | (0.6948, 0.2090, 0.0714, 0.0155, 0.0093) | −0.1231 |
| Positive on the top token | p_y = 0.6364 | (0.6959, 0.1899, 0.0752, 0.0284, 0.0106) | −0.0883 |
| Positive on a rare token | p_y = 0.0317 | (0.5668, 0.2550, 0.1010, 0.0630, 0.0142) | +0.1363 |

The second row differs from the first in the sampled token only. Penalizing the rare token removed 0.0162 from it and gave 0.0583 to the token that was already most probable, while tokens 2 and 3 also lost probability: the update sharpened the distribution it was supposed to correct. Penalizing the top token spread its mass over the rest and raised entropy.

**Evidence.** [[learning-dynamics-llm-finetuning]] §3.3 states the same four facts for any softmax head and proves them for multi-class logistic regression (App. E, Claims 1-5): the penalized label's probability decreases; the decreased mass is largely squeezed into the label that was most confident before the update; high-probability dimensions tend to rise and low-probability ones tend to fall; a peakier distribution squeezes more; and a smaller probability on the penalized label makes the effect stronger. The authors name this the **squeezing effect**. Result (single study) for the proofs, replicated in the token-level gradient expressions of [[negative-sample-reinforcement]] §4.2.

**Entropy by advantage sign.** [[entropy-mechanism-llm-rl]] gives the population version. Lemma 1 states that one update changes policy entropy by approximately the negative covariance between an action's log-probability and its logit change; Theorem 1 evaluates that for a vanilla policy gradient as −η·Cov(log π(a), π(a)·A(a)), and Theorem 2 for a natural policy gradient as −η·Cov(log π(a), A(a)) (§3.1-3.2). The centered token covariance the paper measures is Cov(y_i) = (log π(y_i) − mean_j log π(y_j))·(A(y_i) − mean_j A(y_j)) (Eq. 10). Combining this with the worked example gives the four-quadrant reading:

| | Above-average probability | Below-average probability |
|---|---|---|
| **Positive advantage** | centered product > 0 → entropy falls (exploitation is reinforced) | centered product < 0 → entropy rises |
| **Negative advantage** | centered product < 0 → entropy rises | centered product > 0 → entropy falls (squeezing) |

The card for [[entropy-mechanism-llm-rl]] records that the paper states the sign rule but reports no statistics or results split by advantage sign, so the quadrant table is the course's reading of Eq. 10 and the worked example above (Interpretation). Note also that Clip-Cov and KL-Cov select tokens by this centered product and not by p·A, so negative-advantage tokens are eligible for selection, which a ranking by the largest p·A would not do.

**Bounded variants.** Four losses push the same token down with different weights on the update:

| Loss | Factor multiplying the drop in the sampled token's logit | Behaviour as p_y → 0 | Source |
|---|---|---|---|
| Policy gradient push-down, −A log π | (1 − p_y) | approaches 1: strongest where the token is already unlikely | §2 above |
| NSR, loss written on the probability −R·π | π_y(1 − π_y) | approaches 0 | [[negative-sample-reinforcement]] Eq. 8 |
| Unlikelihood, −log(1 − p_y) | p_y, since the scalar weight p_y/(1 − p_y) multiplies the same (1 − p_y) | approaches 0 on the penalized token, and is largest where the token is confident | [[negative-sample-reinforcement]] Eq. 11; derivation in [[ch-31a]] §5.1 |
| NPO, −(2/β)·log σ(−β log(π_θ/π_ref)) | W(1 − p_y), with W = 2π_θ^β/(π_θ^β + π_ref^β) | approaches 0 once π_θ ≪ π_ref | [[negative-preference-optimization]] Eq. 5 |

Worked example for the NPO weight with β = 1: at π_θ = π_ref the weight is 1; at π_θ/π_ref = 0.1 it is 0.182; at 0.01 it is 0.0198. With β = 0.1 the same ratios give 0.885 and 0.774, which is why the paper shows NPO converging to gradient ascent as β → 0 (Proposition 1).

One practical caveat about the NSR form: [[negative-sample-reinforcement]] Eq. 6 writes the loss on the probability π rather than on log π, which is where the extra π_y factor comes from; an implementation that uses the PPO importance ratio π_θ/π_old has gradient ∇log π_θ at the on-policy point, so the (1 − π_y) damping the paper describes holds while the extra π_y factor does not (derived). The transferable part of the paper's claim is the relative size of updates on confident and unconfident tokens, not the exact exponent.

The interactive companion [figures/negative-gradient-lab.html](figures/negative-gradient-lab.html) reproduces the table above: it lets the reader change the logits, pick the update rule and the advantage sign, apply steps, and watch the entropy and the four-quadrant reading change. Its defaults reproduce the numbers in this section.

## §3 Multi-token responses and shared parameters

At the level of a whole response the update is no longer local: the penalized response shares tokens, prefixes, and parameters with the responses training is meant to promote.

**Likelihood displacement.** [[likelihood-displacement]] defines displacement as a decrease of the mean log π(y⁺|x) while the loss decreases. In its refusal experiment, DPO on on-policy pairs from SORRY-Bench prompts lowered the training-set refusal rate of Llama-3-8B-Instruct from 74.4% to 33.4% and of Gemma-2B-IT from 80.5% to 54.8% (§6.2). For Llama-3-8B-Instruct, 262 of 370 pairs (71%) had two refusals; training on only two-refusal pairs or only two-non-refusal pairs also caused the drop (§6.2-6.3, Figure 4). The paper's CHES score (centered hidden embedding similarity) measures how similar the two responses' hidden embeddings are, and filtering to the 5% of pairs with the lowest length-normalized CHES restored the refusal rates (§6.3). Result (single study), on two models.

**Gradient entanglement.** [[gradient-entanglement-margin-alignment]] gives the first-order account for any margin loss (Eqs. 4-5):

```
Δ log π_w ≈ C (‖∇ log π_w‖² − ⟨∇ log π_w, ∇ log π_l⟩)
Δ log π_l ≈ C (⟨∇ log π_w, ∇ log π_l⟩ − ‖∇ log π_l‖²)
```

- π_w, π_l: the chosen and rejected responses' probabilities; ∇ log π: their gradients with respect to θ; C = ηβc(θ) > 0 for DPO.
- Condition 1: the chosen log-probability rises only if the inner product is at most ‖∇ log π_w‖², and the rejected one falls only if it is at most ‖∇ log π_l‖².

**Worked example.** Take ‖∇ log π_w‖² = 1, ‖∇ log π_l‖² = 4 and ⟨∇ log π_w, ∇ log π_l⟩ = 2, with C = 1. Then Δ log π_w = 1 − 2 = −1 and Δ log π_l = 2 − 4 = −2. The margin still improves by 1, and both log-probabilities fall: this is Case 2 of Table 1, and the paper reports that the rejected gradient often has the larger norm, which puts training in this case (§3.1.1). The observable signature is the pair of curves in Figure 1: on TL;DR, Mistral-7B-Instruct-v0.3 ends with both log-probabilities rising and Meta-Llama-3-8B-Instruct with both falling.

**Which tokens cause it.** At token level the gradients of the tokens that actually differ ("positive" against "negative" in the sentiment setup) have inner product below 0, while identical tokens shared by the two responses have cosine similarities the authors describe as close to 1 for some tokens (§4.3, Figure 4b). Entanglement is produced by the shared tokens, not by the contrastive ones. That is the mechanism behind two otherwise unrelated recommendations: filter pairs whose responses are near-duplicates ([[likelihood-displacement]]), and localize the penalty to the differing tokens or steps (§5 below, and the step-level preference losses of [[ch-39]]).

**Where the mass goes at response level.** In the bandit setting of [[on-policy-suboptimal-preference-data]] (Figure 16), contrastive training with few prompts raises the implicit reward of y_w and lowers it for y_l, while as the number of prompts grows the implicit reward of y_w also falls; the §5.2 takeaway states the general condition, that when y_l is not sufficiently different from y_w and capacity is limited, "the recovered probability mass will go into increasing likelihoods of other responses, not y_w". For Pythia-1.4B on AlpacaFarm both fell; for Mistral-7B on UltraFeedback, where the two responses come from models of different capability and are more distinct, the chosen rose and the rejected fell (Figure 17). Model capacity, dataset size, and how different the two responses are decide which case holds.

**Off-policy is the aggravating factor, and pulling the negative up first helps.** [[learning-dynamics-llm-finetuning]] §4.2 measures the off-policy DPO case directly: the log-probability of almost every probed response falls while the greedy-decoded response's log-probability rises from about −113 to −63 within 8 epochs. Their mitigation is to include (x, y⁻) in the SFT stage so that y⁻ is no longer in a low-probability region when DPO starts. On Qwen1.5-1.8B with a 5,000-example Anthropic-HH subset, the win rate of this pipeline against the standard one, judged by GPT-3.5-Turbo and Claude3-Haiku, is 0.4729 / 0.4679 after SFT and 0.6518 / 0.5151, 0.6928 / 0.6045, 0.6667 / 0.5432 after 2, 4 and 6 DPO epochs (App. F.3, Table 1). A win rate below 0.5 after SFT and above 0.5 after DPO means the pipeline that is behind before preference training is ahead after it, so the intervention acts through the probability region the negative gradient lands in rather than through the SFT checkpoint's quality.

**On-policy sampling does not remove the effect.** [[lazy-likelihood-displacement-grpo]] measures per-question likelihood changes under a single GRPO update (Definition 4.1, Eq. 2) and finds that many questions show small or negative change in the log-probability of their correct responses, which it names **lazy likelihood displacement**. Masking negative advantages ("Pos Only") raises those changes, and for some questions lowers them, so negative gradients are not uniformly harmful; Pos Only scored 1.3 points below GRPO on average (Table 2: 39.84 against 41.14 on Qwen2.5-Math-1.5B). The questions with the smallest change are those whose incorrect responses are nearly correct or correct with the wrong output format (§3, Figure 2). Theorem 4.4 attributes the effect to the inner products between hidden embeddings of the negative tokens and the correct responses, and the paper's Group Weighted Hidden Embedding Score (GWHES) ranking overlaps the true likelihood-change ranking on 50% and 60% of the top-10 questions against 17.5% and 21.3% for random ranking (Table 1).

## §4 Positive and negative reinforcement measured separately

[[negative-sample-reinforcement]] splits the RLVR objective into positive sample reinforcement (PSR) and negative sample reinforcement (NSR) and trains each alone (Eqs. 2-4). Since both use samples from the current model, this isolates the sign, not the data source. **W-REINFORCE** in the table below is the paper's combination: the PSR term is multiplied by a weight λ and the NSR term is kept at weight 1 (Eq. 9), so λ = 1 recovers plain REINFORCE and λ = 0 recovers NSR.

**Result (single study, three model families).** Qwen2.5-Math-7B trained on MATH, evaluated with the unbiased pass@k estimator over 256 samples (Table 1):

| Method | MATH pass@1 | MATH pass@256 | AIME 2025 pass@256 | AMC23 pass@256 |
|---|---|---|---|---|
| Base model | 63.2 | 96.9 | 46.7 | 100.0 |
| PPO | 76.6 | 96.3 | 43.3 | 97.5 |
| GRPO | 76.3 | 95.5 | 50.0 | 97.5 |
| REINFORCE | 74.8 | 92.0 | 50.0 | 92.5 |
| PSR (positives only) | 74.1 | 91.2 | 43.3 | 92.5 |
| NSR (negatives only) | 75.7 | 96.9 | 53.3 | 100.0 |
| W-REINFORCE (λ = 0.1) | 76.6 | 96.7 | 56.7 | 97.5 |

Entropy on a held-out test set stays near the base model's under NSR, falls fastest under PSR, and falls in between under PPO and GRPO (Figure 5b). NSR also raises the correct-sample ratio more slowly and reaches a lower fully-solved-prompt ratio than PSR (Figures 5c-d): it does not drive the model to answer every training prompt the same way. The λ sweep (Table 4) shows the trade-off changes slowly over the range tested: MATH pass@256 is 96.9, 97.1, 96.7, 95.9 and 92.0 for λ = 0, 0.05, 0.1, 0.2 and 1.

**Conditions and limits.** Three matter for a general-purpose reading. First, the base model decides: on Llama-3.1-8B-Instruct every method, NSR included, ended below the base model's pass@256 (Figure 4). Second, negative-only training degrades over hundreds of gradient steps, which the authors report as a limitation and which W-REINFORCE did not show (App. F). Third, the study covers sparse binary verifiable rewards only.

**A second measurement, in the other direction.** [[raft-reinforce-rej-minimalist]] compares positive-only RAFT and RAFT++ against GRPO and REINFORCE on Numina-Math (Table 1). On Qwen2.5-Math-7B-base the three-benchmark averages are RAFT 52.3, RAFT++ 56.1, GRPO 56.3, Reinforce-Rej 56.4, PPO 52.5, iterative DPO 48.8; on LLaMA-3.2-3B-instruct, REINFORCE (24.2) falls below positive-only RAFT++ (27.6) while GRPO reaches 28.4. Their entropy curves match the NSR study from the other side: RAFT++ shows "a much more rapid decline in policy entropy compared to GRPO" on both models, and once its entropy stabilizes at a low level its improvement slows and GRPO overtakes it (Figure 3). Their ablation isolates which negatives hurt: dropping prompts whose responses are *all* incorrect gives the largest reward gain over plain REINFORCE, while dropping all-correct prompts changes little, and reward normalization is not the source of GRPO's advantage (Figure 4). An all-incorrect group under a ±1 reward with no usable baseline is a pure push-down with no positive anchor in the batch.

**How large is the effect when both signs are used.** In the rows below, RFT (rejection-sampling fine-tuning) and RAFT are the positive-only arms: they sample responses, keep the correct ones, and fine-tune on them with no term for the incorrect ones.

| Setting | Without negatives | With negatives | Attribution |
|---|---|---|---|
| NFT, Qwen2.5-32B, six-benchmark average (base 29.6) | RFT 52.8 | NFT 59.2 | "positive data (RFT) contributes to 80% … while negative data only accounts for the remaining 20%" ([[nft-negative-aware-finetuning]] §5.3) |
| NFT, Qwen2.5-Math-7B (base 31.6) | RFT 48.3 | NFT 51.7 | 3.4 of 20.1 points, 16.9% (derived from Table 1) |
| NSR/W-REINFORCE, Qwen2.5-Math-7B MATH | PSR pass@256 91.2 | NSR 96.9 | negatives account for the whole pass@k difference in this pair; pass@1 differs by 1.6 points ([[negative-sample-reinforcement]] Table 1) |
| BCPG-NSA, DeepSeek-R1-Distill-Qwen-14B, four-benchmark average (start 66.09) | RFT 64.64 | BCPG 67.44, BCPG-NSA 68.45 | positive-only training fell 1.45 points below the starting model ([[bcpg-nsa]] Table 3) |
| RAFT++ vs GRPO, Qwen2.5-Math-7B-base | RAFT++ 56.1 | GRPO 56.3 | 0.2 points ([[raft-reinforce-rej-minimalist]] Table 1) |

Two cautions carry with these numbers. The NFT comparison also changes the prompt weighting (ω(q) = 1 − r̂_q for NFT against a constant weight for RFT, App. C), so the 20% is not attributable to the negative term alone. The RAFT++ and NSR rows disagree about how much negatives are worth at pass@1 while agreeing about entropy, and both are single studies on Qwen math models.

**Negatives also carry usable content.** [[bcpg-nsa]] §3 fine-tuned on 19,000 responses with incorrect final answers and on 19,000 correct ones from the same prompts: AIME24 was 41.67 for the incorrect set and 52.75 for the correct set, against 10.00 for the starting model (Table 1). Wrong long-CoT responses contain correct steps, self-reflection, and alternative approaches. That is an argument for meaning (2) of "negative", not for a stronger push-down, and it motivates the step-level treatment in §5.

## §5 Negatives inside online group RL

**Zero-advantage groups.** When a group is homogeneous, the advantage is zero for all its members. [[dapo]] §3.2 oversamples and filters those prompts out so that every batch has mixed groups; the ablation attributes 8 of DAPO's 20 AIME 2024 points to dynamic sampling (42 → 50, Table 1). [[ngrpo]] takes the opposite route for all-incorrect groups: it adds one virtual sample with the maximum reward to the group statistics, so an all-incorrect group receives a uniform negative advantage (Eqs. 5-6).

**Worked example.** With G = 8 and a binary reward, take a group with one correct response. Standardizing with the sample standard deviation gives advantages +2.47 for the correct response and −0.35 for each incorrect one; adding a virtual sample with r_max = 1 changes them to +1.76 and −0.50, the values printed in [[ngrpo]] Figure 3. For an all-incorrect group the same convention gives a uniform −0.33 where GRPO gives 0.00 (derived; Eq. 6 as printed uses 1/(G+1) inside the square root, while the figure's values match the sample standard deviation). The calibration therefore does two things at once: it gives homogeneous-incorrect groups a nonzero gradient, and it makes the negative side of every mixed group heavier.

Because the calibrated advantages sum to a negative number, the update carries a persistent downward pressure, which the authors state raises entropy and can destabilize training. Their control is asymmetric clipping: ε_neg = 0.16 against ε_pos = 0.24 (§4.2). DAPO reaches the same asymmetry from the other side: it raises ε_high to 0.28 to let unlikely tokens be promoted, and keeps ε_low at 0.2 "because increasing it will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space" (§3.1). Both settings act on the same quantity: ε_low is the per-update bound on how far one negative step may push a token down, and it is the parameter that protects the sampling space.

**Results.** NGRPO on Qwen2.5-Math-7B reports pass@k AUC 31.28 on AIME2025 against DAPO 30.27, PSR-NSR 28.85 and GRPO 28.33, and AIME2025 pass@256 of 60.00 against 53.33 for both GRPO and DAPO (Table 1). The ablation separates the parts: asymmetric clipping alone 28.48, advantage calibration alone 29.56, both 30.54, and the full method that also keeps homogeneous-incorrect groups 31.28 (Table 2). Result (single study, one model). Note that DAPO and NGRPO disagree about what to do with all-wrong groups, and the [[raft-reinforce-rej-minimalist]] ablation found all-wrong prompts to be the harmful ones under plain REINFORCE; the difference is that NGRPO's virtual sample bounds the advantage of such a group at a small magnitude while plain REINFORCE gives every one of those tokens a full −1.

**Localizing the penalty.** Two methods reduce the penalty on tokens inside a wrong response that are not wrong:

- [[lazy-likelihood-displacement-grpo]] scores each negative token by its influence on the likelihood of the group's correct responses (NTHR, Eq. 7) and sets the advantage of the selected tokens to zero (β = 1, η = 0). Averages over five math benchmarks (Table 2): Qwen2.5-Math-1.5B 41.14 → 41.94; Qwen2.5-0.5B-Instruct 11.72 → 12.66; Qwen2.5-1.5B-Instruct 26.96 → 28.48; Qwen2.5-3B 33.88 → 36.30; Qwen2.5-Math-1.5B on DeepScaleR 37.80 → 39.60. A random-token control gave only modest gains on the likelihood measure (Figure 4).
- [[bcpg-nsa]] labels each step of a negative response with an LLM judge and a PRM, keeps a step as correct only when both agree, and multiplies the negative weight of those tokens by β ∈ [−1, 1] (Eqs. 6-7). At β = 0.5 the four-benchmark average is 68.45 against 67.44 for the same objective with β = 1 and 66.09 for the starting model (Table 3). The consensus labeler mined the fewest correct tokens (26M against 38M for PRM-only and 65M for LLM-only) and produced the best result (Tables 4-5), so precision of the step labels mattered more than recall.

Both methods keep the response-level negative signal and change only which tokens inside it receive it. That is the RL counterpart of the pair-filtering recommendation in §3.

## §6 Off-policy negatives

**Why off-policy negatives are the unstable case.** Under a fixed data distribution µ, naive off-policy REINFORCE maximizes Σ_{T⁺} µ log π − Σ_{T⁻} µ log π. [[topr-tapered-off-policy-reinforce]] §2.1 states that the second term "is unbounded above (in terms of π) and can be made arbitrarily large by driving the probability of any single trajectory supported by µ to zero", and notes that this does not arise on-policy, because a trajectory the model no longer produces is also no longer sampled. In their GSM8K runs naive REINFORCE improved and then collapsed into degenerate generations, and PPO made little progress because most samples fell outside [1 − ε, 1 + ε] with ε = 0.2 (Figures 1 and 3).

**Three bounded treatments.**

1. **Truncate the ratio, asymmetrically.** TOPR's canonical form sets the negative side's importance ratio to [π/µ]₀¹ and the positive side to a plain SFT update (Eq. 8). The lower limit of 0 on negatives is the point: as the model moves away from a wrong trajectory, its contribution decays. "Any a⁻ > 0 must eventually lead to model degeneracy as with naive REINFORCE" (§3). With a 60%-negative dataset and gradient clipping loosened to 100.0, untruncated importance sampling produced 31% bad reasonings by the end of training against 12% for the base model, while TOPR still improved on the base model (§4.3, Figure 7).
2. **Move the baseline.** [[asymmetric-reinforce]] (AsymRE) studies A = r − V off-policy and proves that the limit policy keeps a wide support while V < V^µ and that its support "suddenly shrinks" to generically one element once V ≥ V^µ (Thm. 4.2). In the LLM runs, all 7 Llama-3.1-8B-Instruct runs on MATH with δV = 0 collapsed, and the authors report greater stability at δV = −0.1 (§5.2, Fig. 5 left). A baseline below the mean reward weights positives more and negatives less.
3. **Add a KL anchor, and know when it is required.** [[pmpo-positive-negative-feedback]] derives the objective α·E_{D_a}[log π] − (1 − α)·E_{D_r}[log π] − β·KL(π_ref, π_θ) (Eq. 10) and reports that the KL term is needed in the negative-only case: in the Control Suite tasks of §5.2, with α = 0 performance is highly sensitive to β and needs β > 1.0, with α = 1 it is insensitive, and with α = 0.5 a β above 0.5 is needed (Figure 3). The language-alignment run repeats the pattern, stating that negative-only training needs β > (1 − α) (§5.4, Figure 5). On an offline RGB-stacking benchmark the reward over 100 evaluation episodes is 24 for behaviour cloning, 26 for accept+BC, 27 for accept only, 77 for reject+BC and 93 for all three terms (Table 1).

**Worked example (baseline as data composition).** TOPR's effective positive proportion is p̃ = p(1 − c)/(1 + c(1 − 2p)), where p is the actual positive share and c the reward baseline (Figure 5 caption). With p = 0.1 and c = 0 the effective share is 0.10; with c = −0.8 it is 0.50; with p = 0.5 and c = 0.6 it falls to 0.20. Their GSM8K sweep found performance peaks at an effective positive proportion of 10-20% for both a 10%-positive and a 50%-positive dataset, and degrades markedly above 50% (§4.3). Discarding negatives is the special case c = −1.

**Unlearning is the extreme case.** When the goal is removal rather than improvement, only the negative term exists. [[negative-preference-optimization]] shows gradient ascent on a forget set produces catastrophic collapse — utility falls to zero and the model emits gibberish (§2.1) — while the NPO loss, which is DPO with the preferred term deleted, weights each update by W = 2π_θ^β/(π_θ^β + π_ref^β) and diverges logarithmically instead of linearly (Theorem 2). The transferable part is not the unlearning task but the functional form: an unbounded push-down diverges linearly, while a push-down whose weight decays as the sample becomes unlikely diverges logarithmically.

**Why on-policy sampling is different.** [[on-policy-suboptimal-preference-data]] §5.3 reports that on-policy sampling and negative gradients give complementary benefits, with on-policy DPO and IPO converging faster and to better solutions than offline contrastive training, on-policy RL, or supervised variants in their bandit and synthetic LLM settings. On-policy sampling gives coverage of the response space; the negative gradient gives a stronger signal per sample.

## §7 Negative quality: near misses, easy negatives, and false negatives

Two properties of a negative sample decide what the gradient does: whether the label that marked it incorrect is right, and how probable the sample is under the current policy.

**Near-miss negatives are the informative and the dangerous ones.** The questions where GRPO fails to raise the likelihood of correct answers are exactly those whose wrong responses are nearly correct or correctly reasoned with a bad final format ([[lazy-likelihood-displacement-grpo]] §3), and the tokens with the largest hidden-embedding influence on the group's correct responses are often logically or stepwise correct terms inside the wrong response (§5.1, Figure 3). The same property makes them worth mining at step level ([[bcpg-nsa]]).

**Easy and off-policy negatives contribute little and can harm.** A negative that the current model would rarely produce sits in a low-probability region, which is the regime the squeezing analysis identifies as the worst one (§2), and off-policy it is also the regime where the ratio-based bounds of §6 shrink the update to near zero by design.

**Verifier false negatives become suppression of correct behaviour.** Four measured cases:

| Source of the false negative | Measurement | Source |
|---|---|---|
| Answer parser rejects correct solutions | share of retained correct math solutions 25% (regex + sympy) → 73% (model judge) | [[bespoke-stratos]] blog, "Data Curation" |
| Unit tests encode an unstated interface | GPT-5 (high) 25.9% → 8.40% and Claude Opus 4.1 22.7% → 8.20% when requirements and interface fields are removed | [[swe-bench-pro]] §6.2, Table 3 |
| Truncation counted as failure | masking truncated samples moved AIME 2024 avg@32 from 30 to 36; soft length shaping added 3 more points | [[dapo]] Table 1, §3.4 |
| Truncation counted as failure, other direction | NFT treats truncated answers as negative and reports no ablation of that choice | [[nft-negative-aware-finetuning]] App. C |

The control DAPO applies is to mask, not to penalize: "a sound reasoning process can be penalized solely due to its excessive length" (§3.4). The generalization is the rule of this section — when the label may be wrong, remove the sample from the loss rather than giving it a negative advantage — and the comparison that tests it is a single ablation run with the truncated samples masked out of the loss.

**Abstention is a false negative by construction.** [[why-language-models-hallucinate]] §4.1 shows that under a binary grader, where abstentions score 0 and a lucky guess scores 1, no abstention is ever an optimal response (Observation 1), and Table 2 finds that nine of the ten popular benchmarks it surveys grade this way. A verifiable-reward RL run inherits that structure: "I don't know" and a confident fabrication receive the same negative advantage, so the negative gradient pushes down the abstention and the error together, and a base model that was calibrated after pre-training need not stay calibrated (§3.1, Figure 2).

**Worked example (what a confidence target changes).** Under the proposed instruction, a correct answer scores 1, an abstention 0, and an error −t/(1 − t). Answering with correctness probability p has expected score p − (1 − p)·t/(1 − t), which is positive exactly when p > t (§4.2). At t = 0.5 the penalty is 1 and a 0.6-confident answer scores 0.6 − 0.4 = +0.2, so answering wins; at t = 0.9 the penalty is 9 and the same answer scores 0.6 − 3.6 = −3.0, so abstaining wins. Porting this into the reward of an RL run changes the sign of the advantage on abstentions, which is the only way a negative gradient can teach abstention rather than suppress it. No source in this chapter trains that reward and measures the effect on general capability; that is an Open question.

## §8 Failure-reuse map across the pipeline

| Use of a failure | Meaning | Stage | Evidence anchor |
|---|---|---|---|
| Discard | (1) | SFT, distillation, RFT | positive-only RFT fell 1.45 points below its starting model ([[bcpg-nsa]] Table 3); RAFT 52.3 vs GRPO 56.3 ([[raft-reinforce-rej-minimalist]] Table 1) |
| Rationalize (re-derive with the answer given) | (2) | SFT | STaR + rationalization 72.5% vs 60.0% direct fine-tuning on CQA ([[star]] Table 1) |
| Correction target (failure in the input) | (2) | SFT | [[ch-31a]] §3 |
| Failure-conditioned (label not used at inference) | (3) | SFT | [[ch-31a]] §4 |
| Rejected side of a preference pair | (4) | preference optimization | [[likelihood-displacement]], [[gradient-entanglement-margin-alignment]], [[ch-39]] |
| Negative advantage on the whole response | (4) | RL | §4, §5 |
| Step- or token-level negative advantage | (4) | RL | [[bcpg-nsa]] Eq. 7; [[lazy-likelihood-displacement-grpo]] §5 |
| Training a verifier or PRM on failures | (2) for the policy, (4) inside the verifier | reward modeling | TOPR verifier accuracy 32.6% → 70.9%, invalid rate 34.2% → 0.90% ([[topr-tapered-off-policy-reinforce]] Table 2); MC rollouts as step labels ([[math-shepherd]]) |
| Implicit negative policy (no explicit negative term) | (4) in effect | online SFT | [[nft-negative-aware-finetuning]] Eq. 8 |

The map is the practical summary of the chapter: the same failed rollout can be discarded, rewritten, conditioned, or pushed down, and only the last option carries the risks of §2 and §3.

## Negative samples and negative feedback

This section applies the course standard to the RL and preference stages; the meanings (1)-(4) are those of [[ch-31a]] §1.

1. **Where negatives come from.** Rule-based verifiers on final answers ([[negative-sample-reinforcement]], [[nft-negative-aware-finetuning]], [[dapo]]), execution and unit tests ([[swe-bench-pro]]), reward models and judges ([[pmpo-positive-negative-feedback]]), step judges plus PRMs ([[bcpg-nsa]]), and human preference labels ([[likelihood-displacement]] uses PairRM for tie-breaking). Measured false-negative sources are listed in §7; none of the RL papers in this chapter reports a false-negative rate for its own verifier.
2. **What current practice does with them.** RFT, STaR-style loops and RAFT discard them (meaning 1). GRPO, DAPO, NGRPO, TOPR, AsymRE, DPO and NPO use them as gradient (meaning 4). NFT uses them as gradient through an implicit negative policy. BCPG-NSA and NTHR use them as gradient at step or token level, and BCPG-NSA's preliminary study also uses them as content (meaning 2).
3. **Mechanism.** ∂log p_y/∂z_j = 1[j = y] − p_j. A push-down lowers the sampled token's logit by η|A|(1 − p_y) and raises every other logit by η|A|p_j, so the removed mass goes to the already-most-likely token; on an unlikely token this sharpens the distribution (§2, worked example: a push-down on a 0.0317 token gave 0.0583 to the 0.6364 token and lowered entropy by 0.12 nats). At response level the same step moves every response that shares tokens or embeddings with the penalized one (§3).
4. **Evidence with numbers.** Benefit: §4's table, NGRPO's pass@256 60.00 against 53.33, NTHR's +0.8 to +2.4 points, BCPG-NSA's +1.0 average. Failure: refusal 74.4% → 33.4% ([[likelihood-displacement]]), 7/7 collapsed runs at δV = 0 ([[asymmetric-reinforce]]), 31% bad reasonings with untruncated ratios ([[topr-tapered-off-policy-reinforce]]), collapse without a KL term at α = 0 ([[pmpo-positive-negative-feedback]]), gibberish under gradient ascent ([[negative-preference-optimization]]).
5. **Controls.** Keep negatives on-policy or bound them by the importance ratio (a⁻ = 0, b⁻ = 1); keep a positive anchor in the batch or in the loss (λ = 0.1 on the positive term, a positive NLL term, a KL to the reference); bound the per-update push-down (ε_low, ε_neg, NFT's ε); localize to the differing tokens or the first wrong step; filter near-duplicate pairs by CHES; mask uncertain failures instead of penalizing them.
6. **Diagnostics.** Log chosen and rejected log-probabilities separately ([[gradient-entanglement-margin-alignment]] Figure 1); log gradient norms, clip fraction and entropy split by advantage sign (§2's quadrant table tells you what to expect); track pass@1 and pass@k at k ≥ 64; track the share of prompts whose groups are homogeneous ([[dapo]] Figure 3b); track the abstention rate and a calibration curve; on a small probe set, measure Δ log π of known-good responses after a single update ([[lazy-likelihood-displacement-grpo]] Eq. 2).
7. **Effect on generality.** Coverage: negative-only training preserved pass@256 where positive-only training lost 5.7 points on MATH ([[negative-sample-reinforcement]] Table 1). Calibration and hallucination: binary graders make abstention a negative, and no source here measures calibration after RLVR ([[why-language-models-hallucinate]] §4.1; Open question). Over-suppression: refusal rates fell by 41 points in DPO refusal training ([[likelihood-displacement]] §6.2). Forgetting: not measured in the RLVR studies of this chapter; the KL-based treatment is in [[ch-43]].

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B (NSR / W-REINFORCE) | 7B | RL | data; prompt batch; rollouts per prompt; mini-batch; LR; temperature; max context | MATH 7,500 problems; 1,024; 8; 256; 1e-6; 1.0; 4,096 | arXiv:2506.01347v2 §3.1, App. D.1 | verified 2026-09-15 | no ablation reported |
| Qwen2.5-Math-7B (W-REINFORCE) | 7B | RL | weight λ on the positive term | 0.1 | arXiv:2506.01347v2 §5, Eq. 9 | verified 2026-09-15 | Table 4: MATH pass@256 96.9 / 97.1 / 96.7 / 95.9 / 92.0 at λ = 0 / 0.05 / 0.1 / 0.2 / 1 |
| Qwen2.5-Math-7B, Qwen2.5-32B (NFT) | 7B, 32B | RL | negative-ratio clip ε; prompt weight ω(q); samples per prompt; rollout steps; LR | 1.0; 1 − r̂_q; 16 × 512 questions; 320; 1e-6 with linear warm-up | arXiv:2505.18116v3 §5.1, App. C | verified 2026-09-15 | §5.4 Fig. 10: ε → 0 degrades performance; Fig. 9: ω(q) = 1 − r̂_q beats ω(q) = 1 |
| Qwen2.5-32B (DAPO) | 32B | RL | ε_low; ε_high; dynamic sampling; soft overlong window; max generation; prompt batch × responses; LR | 0.2; 0.28; on; 16,384 + 4,096 cache; 20,480 tokens; 512 × 16; 1e-6 constant, 20 warm-up rollout steps | arXiv:2503.14476v2 §3.1-3.4, §4.1 | verified 2026-09-15 | Table 1: 30 → 36 (overlong filtering) → 38 (clip-higher) → 41 (soft punishment) → 42 (token-level loss) → 50 (dynamic sampling) |
| Qwen2.5-Math-7B (NGRPO) | 7B | RL | virtual sample; ε_neg; ε_pos; epochs; batch; LR; max response; compute | one r_max per group; 0.16; 0.24; 20 on MATH; 1,024; 1e-6; 3,072 tokens; 8 × H100 | arXiv:2509.18851v1 §4.1-4.2, §5.1.1 | verified 2026-09-15 | Table 2 (pass@k AUC, AIME2025/AMC): 28.33/81.04 → 28.48/81.22 (clipping) → 29.56/83.85 (calibration) → 30.54/84.37 → 31.28/86.09 |
| Llama 3 8B Instruct (TOPR) | 8B | RL | truncation limits (a⁺, b⁺, a⁻, b⁻); LR; optimizer; KL; grad clip; reward; baseline | 1, 1, 0, 1; 5e-7 constant; Adafactor; none; 1.0; ±1; none | arXiv:2503.14286v2 §3, §4.2 | verified 2026-09-15 | §4.3 Fig. 7: untruncated ratios gave 31% bad reasonings vs 12% for the base model under a 60%-negative dataset |
| Llama 3 8B Instruct (TOPR) | 8B | RL | effective positive proportion p̃ | 10-20% | arXiv:2503.14286v2 §4.3, Figs. 5-6 | verified 2026-09-15 | GSM8K and MATH sweeps; performance falls above 50% |
| Llama-3.1-8B-Instruct (AsymRE) | 8B | RL | baseline offset δV; samples per prompt; behaviour-policy refresh; LR | ≈ −0.1; 8; every 250 gradient steps; 6e-8 | arXiv:2506.20520v2 §5.2-5.3, App. B | verified 2026-09-14 (card [[asymmetric-reinforce]]) | Fig. 5 left: 7/7 runs collapse at δV = 0; greater stability at −0.1 |
| Qwen2.5 0.5B-3B (GRPO + NTHR) | 0.5B-3B | RL | threshold scale β; retained-token advantage scale η; data | 1; 0 (no penalty on selected tokens); MATH levels 3-5 | arXiv:2505.18830v1 §5.3 | verified 2026-09-15 | Table 2: +0.8 to +2.4 average points over GRPO across four models |
| DeepSeek-R1-Distill-Qwen-14B (BCPG-NSA) | 14B | RL | offline training on a fixed rollout set; mining coefficient β; behaviour constraint τ; LR; epochs; batch; sequence length | 2,069 questions / 66,208 samples; 0.5; 1e-3; 5e-7 (min 2.5e-7); 8; 512; 32k | arXiv:2505.14403v4 §5.1, App. C, Table 3 | verified 2026-09-15 | §6.2 Fig. 2: average of AIME24/25 rises then falls as β decreases; beats β = 1 across a wide range including β = −0.5 |
| PMPO on DeepMind Control Suite tasks | not reported | RL | KL weight β at α = 0; at α = 1; at α = 0.5 | > 1.0 required; insensitive to β; above 0.5 | arXiv:2410.04166v3 §5.2, Fig. 3 | verified 2026-09-15 | Fig. 3: β swept over 0.0, 0.5, 1.0, 1.5, 2.0; negative-only learning needs β > 1.0 |
| Gemma 2B (PMPO, language alignment) | 2B | RL | α; prompts; generations per prompt; batch; β rule | 0.5 as a starting point when both signals are equally reliable; 500k prompts in about 4,000 learner steps; 4; 128 prompts; β > (1 − α) | arXiv:2410.04166v3 §5.4, App. A | verified 2026-09-15 | §5.4 Fig. 5: negative-only training again needs a sufficiently high β; no numeric sweep is printed for the language runs |
| Gemma-2B-IT, Llama-3-8B-Instruct (CHES filtering) | 2B, 8B | preference | filtered subset | keep the 5% of pairs with the lowest length-normalized CHES | arXiv:2410.08847v4 §6.3 | verified 2026-09-14 (card [[likelihood-displacement]]) | Figure 3: refusal rates restored; up to 15% analogous |
| Llama2-7b-chat (NPO) | 7B | preference | loss (unlearning on the TOFU forget sets); β | NPO with a retain term; 0.1 | arXiv:2404.05868v2 §3, App. D.1 | verified 2026-09-15 | §5: NPO-based methods are the only ones above 0.05 forget quality on Forget05/Forget10 |

**Starting point for a small general-purpose run.** For an RLVR run on a 7B math or code model with a verifier: keep both signs; weight the positive term at λ = 0.1 if pass@k at large k matters more than the fastest pass@1 (verified on Qwen2.5-Math-7B, MATH, 1,024 prompts × 8 rollouts, LR 1e-6); use ε_low = 0.2 with ε_high = 0.28 (verified on Qwen2.5-32B with 512 × 16 rollouts and a 20,480-token generation cap); mask truncated samples rather than penalizing them (same run); and drop homogeneous groups, or, if training on hard prompts where most groups are all-incorrect, use one virtual maximum-reward sample with ε_neg = 0.16 (verified on Qwen2.5-Math-7B, MATH, 20 epochs, 8 × H100). Every value in this paragraph comes from a `verified` row above, and each was measured on a Qwen math model at 7B-32B with a binary verifier; none was measured on a mixed-domain instruction model.

## Generalization lens

**(a) What increases breadth.**
- Keeping the negative term while down-weighting the positive term preserved the base model's pass@256 on three math benchmarks and improved AIME 2025 pass@256 from 46.7 (base) to 56.7 ([[negative-sample-reinforcement]] Table 1).
- Negatives keep entropy higher than positive-only training in two independent studies: NFT against RFT at 7B and 32B ([[nft-negative-aware-finetuning]] Figure 8) and GRPO against RAFT++ on Qwen and LLaMA ([[raft-reinforce-rej-minimalist]] Figure 3). Replicated.
- Using failures as content rather than only as gradient adds capability the push-down cannot: SFT on 19,000 incorrect long-CoT responses raised AIME24 from 10.00 to 41.67 ([[bcpg-nsa]] Table 1).
- Bounded off-policy negatives reduce the number of prompts with no correct sample at all, which is a coverage gain rather than a sharpening gain: the authors state that the gain from negatives comes from "reducing the number of questions for which no or few solutions are found" ([[topr-tapered-off-policy-reinforce]] Figure 4).

**(b) What causes narrowing or forgetting.**
- Push-downs on already-unlikely samples concentrate mass on the most likely continuation and lower entropy (§2; [[learning-dynamics-llm-finetuning]] §3.3), and the model's greedy output becomes more confident faster than any labeled response does (§4.2 of the same source).
- Preference pairs whose two sides are the same kind of response remove the behaviour they were meant to teach: refusal rates 74.4% → 33.4% ([[likelihood-displacement]] §6.2).
- Unbounded negatives off-policy collapse support: theoretical support shrinkage at V ≥ V^µ with 7/7 collapsed runs at δV = 0 ([[asymmetric-reinforce]] Thm. 4.2, §5.2), degenerate generations under naive REINFORCE, and 31% bad reasonings under untruncated importance sampling ([[topr-tapered-off-policy-reinforce]] §4.3).
- Negative-only training degrades after hundreds of steps even when it starts well ([[negative-sample-reinforcement]] App. F).
- A binary verifier makes abstention a negative, so the negative gradient trains against the behaviour that would reduce hallucination ([[why-language-models-hallucinate]] §4.1).

**(c) How to measure it for this stage.**
- pass@k over a wide k range with the unbiased estimator (256 samples per problem in [[negative-sample-reinforcement]]; pass@k AUC over k ∈ {1,…,256} in [[ngrpo]]), reported alongside pass@1 rather than instead of it.
- Entropy on a held-out set, not on the training batch, since the training batch's composition changes with the filtering rules ([[negative-sample-reinforcement]] Figure 5b).
- Chosen and rejected log-probabilities logged separately, plus the norms that Condition 1 compares ([[gradient-entanglement-margin-alignment]] §3.1.1).
- A single-update probe on held-out correct responses, which is what makes lazy displacement visible at all ([[lazy-likelihood-displacement-grpo]] Eq. 2).
- Held-out capability suites outside the training domain, and abstention rate: neither is reported by any RL study in this chapter, which is the main measurement gap here. [[ch-46]] is the lab that closes it for one run.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating "more negative signal" as strictly better | pass@1 rises while pass@k at k ≥ 64 falls below the base model | Evaluate pass@k with the unbiased estimator at several k on every checkpoint |
| Penalizing responses the current policy rarely produces (stale rollouts, imported negatives) | entropy falls, greedy-decoding log-probability rises faster than any labeled response's | Log the importance ratio distribution for negative samples; truncate at 1 with no lower bound |
| Pairs whose chosen and rejected responses are the same kind of response | both chosen and rejected log-probabilities fall; a targeted behaviour such as refusal degrades | Compute length-normalized CHES and inspect the top decile; track the behaviour's rate on a probe set |
| Penalizing a whole wrong response when only one step is wrong | correct answers' likelihood rises slowly or falls for prompts with near-miss failures | Single-update Δ log π probe per prompt; inspect the lowest-Δ prompts for near-miss negatives |
| Treating truncation, parser misses, or missing tests as failures | reward curve and entropy destabilize when the length cap is reached; accuracy drops on tasks with strict output formats | Ablate one run with truncated samples masked; sample 50 "failures" and label them by hand |
| Reading zero-advantage groups as harmless | effective batch size shrinks over training while the logged batch size stays constant | Log the share of homogeneous groups per batch |
| Expecting a positive anchor alone to prevent collapse | loss decreases while both chosen and rejected log-probabilities fall | Check Condition 1 by logging ‖∇ log π_w‖², ‖∇ log π_l‖² and their inner product on a probe batch |
| Assuming negatives drive the gains | a positive-only baseline was never run | Run the positive-only arm (RFT, RAFT, PSR) at matched steps and report the split, as in [[nft-negative-aware-finetuning]] §5.3 |

## Check your understanding

1. In §2's worked example, a push-down on the 0.0317 token lowered entropy by 0.12 nats while a push-down on the 0.6364 token raised it by 0.075 nats. Derive both signs from Δz_j = ηA(1[j = y] − p_j), and explain why this makes on-policy negatives safer than imported ones without appealing to any experiment.
2. NSR-only training kept pass@256 at the base model's level while PSR-only training lost 5.7 points on MATH. Give the causal chain from the token-level gradient expressions to that difference, and state which step of the chain the Llama-3.1-8B-Instruct result (all methods below base at pass@256) breaks.
3. Both DAPO and NGRPO touch all-incorrect groups, in opposite directions, and both report gains. Explain how both can be true, using the magnitude of the advantage each assigns to a token in such a group.
4. [[raft-reinforce-rej-minimalist]] found that dropping all-incorrect prompts helped plain REINFORCE more than any normalization change. Which property of those prompts under a ±1 reward makes them the harmful case, and why does the same argument not apply to all-correct prompts?
5. Gradient entanglement says that the margin always improves while the individual log-probabilities may both fall. Using ‖∇ log π_w‖² = 1, ‖∇ log π_l‖² = 4 and an inner product of 2, show which case holds, and say what token-level property of the pair produced that inner product.
6. TOPR sets a⁻ = 0 for negatives and a⁺ = 1 for positives. Explain each choice in terms of what happens as the policy moves away from the data-generating policy, and connect the a⁻ = 0 choice to the NPO weight W and to NFT's ε.
7. A team reports that adding negatives raised their agent benchmark by 4 points. Which three measurements would you ask for before accepting that negatives caused the gain, and what would each measurement rule out?
8. A verifier for a customer-support agent marks "I need to check that with a human" as incorrect whenever the gold answer is a policy statement. Trace what a negative advantage on those responses does to the model's distribution, and design a reward change plus a diagnostic that would detect the problem within one run.

## Connections

- Previous: [[ch-43]] — Entropy, Output Diversity, and KL Control in RL. Supplies the covariance law and the KL estimators this chapter reads by advantage sign.
- Next: [[ch-44]] — Process Supervision and Verifiable Rewards. Supplies the step-level labels that §5's localized negatives depend on.
- Depends on: [[ch-39]] — Offline Preference Optimization: DPO and Its Variants (the rejected term and its variants); [[ch-31a]] — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood (the four meanings and the supervised bounded losses).
- Applies here: [[ch-40]] — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO; [[ch-41]] — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization; [[ch-42]] — Reward Hacking and Judge Design; [[ch-29d]] — User Simulators, Trajectory Verification, and Failed Trajectories; [[ch-35]] — Distillation in Practice A: Where Labs Insert Teacher Data; [[ch-44a]] — Length in RL: Overlong Responses, Length Control, and Long-Context RL; [[ch-54]] — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments.
- Measured in: [[ch-46]] — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention; [[ch-38a]] — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.

## Sources

- [[learning-dynamics-llm-finetuning]] — squeezing effect, its five claims and proofs, off-policy DPO dynamics, and the "extend" mitigation with win rates.
- [[likelihood-displacement]] — displacement definition, refusal-rate collapse, CHES score and filtering.
- [[gradient-entanglement-margin-alignment]] — first-order coupling of chosen and rejected log-probabilities, Condition 1 and the three cases, token-level gradient similarity.
- [[negative-sample-reinforcement]] — PSR/NSR decomposition, token-level gradients, pass@k and entropy results, W-REINFORCE and the λ sweep, stated limitations.
- [[raft-reinforce-rej-minimalist]] — positive-only RAFT/RAFT++ against GRPO, entropy decline of positive-only training, ablation isolating all-incorrect prompts.
- [[nft-negative-aware-finetuning]] — implicit negative policy, ε clip, entropy curves, the 80/20 attribution at 32B.
- [[lazy-likelihood-displacement-grpo]] — lazy likelihood displacement in GRPO, GWHES, NTHR token selection and results.
- [[dapo]] — clip-higher and the reason ε_low is kept, dynamic sampling, overlong filtering and soft punishment, progressive ablation.
- [[ngrpo]] — virtual maximum-reward sample, asymmetric clipping, pass@k AUC results and component ablation.
- [[bcpg-nsa]] — value of wrong long-CoT responses as content, consensus step labeling, the mining coefficient β and its ablation.
- [[topr-tapered-off-policy-reinforce]] — unbounded negative objective off-policy, asymmetric truncation, baseline as dataset composition, verifier training.
- [[asymmetric-reinforce]] — baseline offset as the positive/negative balance, support-shrinkage theorem, collapse at δV = 0.
- [[negative-preference-optimization]] — gradient-ascent collapse, the NPO weight, linear against logarithmic divergence.
- [[pmpo-positive-negative-feedback]] — decoupled positive and negative terms, the KL term required for negative-only learning, offline-RL loss mixture table.
- [[on-policy-suboptimal-preference-data]] — negative gradient as a mode-seeking update, where the recovered mass goes, complementarity with on-policy sampling.
- [[why-language-models-hallucinate]] — binary graders penalize abstention, explicit confidence targets and behavioral calibration.
- [[entropy-mechanism-llm-rl]] — covariance law for entropy change and centered-covariance token selection, used for the four-quadrant table.
- [[swe-bench-pro]], [[bespoke-stratos]] — measured verifier false negatives from unit tests and answer parsers.
- [[star]], [[math-shepherd]], [[grpo]], [[dr-grpo]], [[rlvr-beyond-base-model]] — rationalization, step labels from rollouts, the group-baseline advantage, and the pass@k critique referenced in the failure-reuse map and §1.
