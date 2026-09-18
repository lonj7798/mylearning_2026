<!-- scope: synthetic-label study of how gold reward changes when a policy is optimized against a learned proxy reward model (best-of-n and PPO), with fitted functional forms in sqrt(KL)
     deps: [[kl-control-rlhf]]
     see-also: [[reward-hacking-taxonomy]], [[lilianweng-reward-hacking]], [[reward-ensembling]], [[warm-weight-averaged-reward-models]], [[rlhf-instructgpt]]
-->

# Scaling Laws for Reward Model Overoptimization
- **Core Insight:** When a 1.2B GPT-3-series policy is optimized with best-of-n (BoN) or PPO against proxy reward models (RMs) of 3M to 3B parameters trained on labels from a 6B "gold" RM, the gold score first rises and then falls as KL from the initial policy grows, and the fitted forms R_bon(d) = d(α_bon − β_bon·d) and R_RL(d) = d(α_RL − β_RL·log d) have coefficients that change smoothly with proxy RM size (§1, §3.2, Fig. 1, Fig. 3).
- **Guideline:** When a policy is optimized against a learned RM, track a held-out gold or human score as a function of √KL and stop near its peak, because in this setup the proxy score keeps rising after the gold score has peaked (Fig. 1, Fig. 8); a nonzero KL penalty behaved like early stopping and did not raise the gold-score-vs-KL frontier (§3.6, Fig. 9), a result the authors call hyperparameter-sensitive.
- **Authors:** Leo Gao, John Schulman, Jacob Hilton (OpenAI)
- **Year:** 2022 (arXiv v1 2022-10; ICML 2023, PMLR 202:10835–10866)
- **URL:** https://arxiv.org/abs/2210.10760
- **Source type:** paper
- **Relevant topics:** Goodhart's law, reward model overoptimization, proxy vs gold reward, KL distance, best-of-n, PPO, RM data and size scaling, iterated RLHF

## Abstract
RLHF optimizes a policy against a reward model trained to predict human preferences; because the RM is an imperfect proxy, too much optimization can lower true performance (Goodhart's law). Measuring this with human labels is expensive, so the authors use a synthetic setup: a fixed gold-standard RM replaces humans and labels the data used to train proxy RMs. They measure how the gold RM score changes as a policy is optimized against the proxy with RL or best-of-n sampling. The relationship has a different functional form for each optimization method, and in both cases the coefficients scale smoothly with the number of proxy RM parameters. They also study RM dataset size, RM and policy parameter counts, and the KL penalty coefficient, and discuss implications for AI alignment.

## Key Contributions
- Functional forms for gold score vs d = √KL(π‖π_init), fit separately for BoN and RL; the BoN form was chosen with data up to n = 1,000 (KL ≈ 6 nats) and then confirmed on a run up to n = 60,000 (KL ≈ 10 nats) (§3.1).
- Smooth scaling of α_bon, β_bon and β_RL with proxy RM parameter count; α_RL can be held constant across RM sizes (§3.2, Fig. 3, footnote 2).
- RL spends far more KL than BoN for the same optimization, so KL should not be used to compare the amount of optimization across methods (§3.5, §4.1).
- Two policy sizes (1.2B, 6B) peak at almost the same KL and show almost the same proxy-gold gap (§3.4, Fig. 7, Fig. 24).
- Interpretation of α as regressional Goodhart and β as extremal Goodhart, and a closed-form estimate for iterated RLHF (§4.2, §4.3).

## Key Figures/Tables to Study
- **Fig. 1** — gold and proxy score vs √KL for BoN (a) and RL (b) across RM sizes; policy fixed at 1.2B.
- **Fig. 3** — α_bon, β_bon, β_RL vs RM parameter count, for proxy and gold scores.
- **Fig. 4 / Fig. 6** — RM data-size sweep (12M RM) and RM validation loss by data size and RM size.
- **Fig. 7** — policy-size sweep (1.2B vs 6B, 12M RM).
- **Fig. 8** — proxy score vs gold score for BoN and RL.
- **Fig. 9 / Fig. 14** — KL penalty sweep (1.2B policy, 1.2B RM) and KL over training steps.

## Technical Details
- **Formula.** d := √(D_KL(π ‖ π_init)); R_bon(d) = d(α_bon − β_bon·d); R_RL(d) = d(α_RL − β_RL·log d) (§1). R is the gold RM score, recentered so the initial policy scores 0 (R(0) := 0); π is the optimized policy; π_init is the initial (SFT) policy; α and β are fitted coefficients that may depend on RM size, RM data size, and other settings (§1, §2.2). The RL form has infinite slope at d = 0 and likely does not hold near the origin; forms with finite slope fit or extrapolated worse (footnote 1, App. B).
- **Peak location (derived here, not printed as a formula in the paper).** Setting dR/dd = 0 gives d* = α_bon/(2β_bon) for BoN and d* = exp(α_RL/β_RL − 1) for RL. Fig. 12 shows the BoN maximum predicted from the closed form.
- **Environment.** InstructGPT setting: prompts are natural-language instructions for varied tasks; all models start from pretrained GPT-3-series checkpoints; policies are SFT-trained on InstructGPT human demonstrations for 2 epochs; RMs use the GPT-3 architecture with a scalar head (§2).
- **Synthetic labels.** The 6B RM from Ouyang et al. (2022) is the gold RM; proxy RMs range from 3M to 3B parameters; two RMs smaller than 3M were near chance and excluded (§2.1, footnote 3). Pairs of policy rollouts are labeled deterministically by higher gold score; sampled labels gave noisier results (§2.1, footnote 4). 100,000 comparisons are generated and 10% held out (§2.1).
- **Recalibration.** Each RM is recentered so the initial policy's mean reward is 0; gold scores are unit-variance normalized; proxy RM logits are rescaled on soft-label validation data. All of this is applied after the experiments (§2.2).
- **BoN.** Scores for intermediate n use the unbiased estimator of Nakano et al. (2021, App. I); KL_bon = log n − (n−1)/n, taken from Stiennon et al. (2020, App. G.3) (§2).
- **PPO.** KL penalty is 0 in all RL experiments except §3.6; PPO hyperparameters are mostly defaults (§2, App. C). PPO's clipped surrogate adds an implicit penalty toward a recent policy π_old, which slows growth of KL from π_init (§3.6).
- **Data scaling.** With a 12M RM, fewer than about 2,000 comparisons give near-chance loss (§3.3, Fig. 6). Four epochs on 2,000 comparisons gave no gold-score change, while one epoch on 8,000 performed better; validation losses were 0.686109 (1×2000), 0.654857 (1×8000), 0.683869 (4×2000) (footnote 7, Fig. 13). RMs with equal validation loss showed similar robustness, reported as weak evidence (§3.3, Fig. 5).
- **Iterated RLHF.** Under assumptions that α_RL and β_RL are constant and d adds across k iterations of length d/k, R_RL(d) = d(α_RL − β_RL·log d + β_RL·log k), a gain of β_RL·d·log k that leaves the α term unchanged (§4.3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT-3-series policy, InstructGPT env. | 1.2B, 6B | SFT | epochs on InstructGPT demonstrations | 2 | arXiv:2210.10760v1 §2 | verified 2026-09-14 | no ablation reported |
| proxy RMs (GPT-3 arch.) | 3M–3B | reward-model | synthetic comparisons; held-out share | 100,000; 10% | §2.1 | verified 2026-09-14 | §3.3 Fig. 4: more data gives higher gold scores |
| proxy RMs | 3M–3B | reward-model | Adam learning rate multiplier | 1.67e-2 | App. C Table 1 | verified 2026-09-14 | no ablation reported |
| proxy RMs | 3M–3B | reward-model | batch size | 64 | App. C Table 1 | verified 2026-09-14 | no ablation reported |
| GPT-3-series policy | 1.2B | RL | algorithm | PPO | §2 | verified 2026-09-14 | no ablation reported |
| GPT-3-series policy | 1.2B | RL | Adam learning rate multiplier | 4e-3 | App. C Table 1 | verified 2026-09-14 | no ablation reported |
| GPT-3-series policy | 1.2B | RL | batch size | 256 | App. C Table 1 | verified 2026-09-14 | no ablation reported |
| GPT-3-series policy | 1.2B | RL | PPO clipping parameter | 0.2 | App. C Table 1 | verified 2026-09-14 | no ablation reported |
| GPT-3-series policy | 1.2B | RL | timesteps per rollout; minibatches per epoch | 256; 128 | App. C Table 1 | verified 2026-09-14 | no ablation reported |
| GPT-3-series policy | 1.2B | RL | GAE bootstrapping parameter | 0.95 | App. C Table 1 | verified 2026-09-14 | no ablation reported |
| GPT-3-series policy | 1.2B | RL | KL penalty coefficient | 0 (all runs except §3.6 sweep) | §2, §3.6 | verified 2026-09-14 | §3.6 Fig. 9: penalty did not change the gold-vs-KL frontier and had a larger proxy-gold gap |

Not reported: base learning rates behind the multipliers, number of PPO steps, sampling temperature, RM training epochs for the main sweeps, KL penalty values in the §3.6 sweep.

## Findings relevant to generality
- **Extremal Goodhart.** Optimized samples move out of the RM's training distribution; the authors expect this to cause most of the gold-score decline (β term) and report long answers where short ones are preferred as an issue seen in other InstructGPT experiments (§4.2.2, footnote 10). Interpretation.
- **Overfitting as Goodhart.** The authors describe overfitting as the special case where the proxy is the score on a finite sample set (§5).
- **Measurement limits.** Only the InstructGPT environment was tested (WebGPT looked visually similar, footnote 15); the synthetic gold RM does not capture the gap between labels and human intent; adversarial Goodhart is not captured and may break the trends (§4.2.4, §4.5).
- **Proxy score is hard to predict.** No satisfactory fit was found for proxy scores; linear fits extrapolated poorly (§3.1, Fig. 20).

## Connections
- [[kl-control-rlhf]] — defines the KL-regularized objective whose penalty §3.6 finds equivalent to early stopping here.
- [[reward-ensembling]], [[warm-weight-averaged-reward-models]] — later RM methods evaluated against the overoptimization effect measured here.
- [[reward-hacking-taxonomy]], [[lilianweng-reward-hacking]] — formal and survey treatments of reward hacking.
- [[rlhf-instructgpt]] — source of the environment, demonstrations, and the 6B gold RM.
- [[ifbench]] — benchmark-level overfitting (unseen constraints); this paper covers only RM-level Goodhart.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2210.10760 (v1, the only version); venue from proceedings.mlr.press/v202/gao23h.html.
- Corrections to the previous card version:
  - "both α and β shrink smoothly with RM parameters" → α_bon and β_bon change smoothly; α_RL is held constant; β decreases with RM size (§3.2, §4.2.2).
  - "policy size barely matters: bigger policies optimize the proxy faster but hit the same gold peak" → the 6B policy gains less from optimization, peaks at almost the same KL, and has lower KL per step (§3.4, footnote 12).
  - "best-of-n has a tighter peak ... PPO overoptimization continues indefinitely" → RL consumes more KL than BoN for both optimization and overoptimization; vs proxy score, RL has a larger initial proxy-gold gap and a higher gold peak (§3.5, Fig. 8).
  - "PPO objective ... per-token −β·KL added to the reward; β absorbed into d" → KL penalty is 0 except in §3.6 (§2).
  - "TL;DR-like tasks" → InstructGPT instruction prompts (§2).
  - Figure mapping: Fig. 2 is the setup diagram; Fig. 5 is RM validation loss vs BoN score; Fig. 7 is policy scaling; Fig. 8 is proxy vs gold (not data scaling) (figure captions).
  - "Best-of-n KL derived analytically [here]" → formula taken from Stiennon et al. (2020, App. G.3) (§2).
  - Venue added: ICML 2023.
- Removed as unsupported by the source: "a 10× larger RM roughly halves the overoptimization slope"; "at RM size 3M gold peaks near d ≈ 3 and loses most of the gain by d ≈ 8"; "KL is token-averaged"; "RM initialized from the SFT reference"; "returns to more preference data diminish before returns to more RM params"; "this is a property of the RM, not the policy".
- Not reported by the source: exact α/β values in text (only plotted), human-label validation.
