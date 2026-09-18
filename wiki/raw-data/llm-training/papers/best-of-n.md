<!-- scope: Stiennon et al. 2020 "Learning to summarize from human feedback": SFT → reward model → PPO on Reddit TL;DR, its best-of-N baseline (App. C.6, G.3, Fig. 7) and the reward-model over-optimization result (§4.3)
     deps: [[bradley-terry-rm]]
     see-also: [[kl-control-rlhf]], [[rlhf-instructgpt]], [[reward-model-overoptimization]], [[west-of-n]], [[rejection-sampling-finetuning]]
-->

# Learning to summarize from human feedback
- **Core Insight:** On Reddit TL;DR, a 1.3B policy trained with PPO against a reward model learned from human comparisons was preferred over the human reference summaries 61% of the time, versus 43% for a supervised model 10× its size (§4.1, Fig. 1); the appendix adds that best-of-N and PPO policies with equal average reward were judged of similar quality, with PPO farther from the supervised policy in KL (App. G.3, data not shown).
- **Guideline:** When a policy is optimized against a learned reward model, measure human preference at several optimization strengths (KL coefficient β for PPO, N for best-of-N), because in §4.3 (Fig. 5) preference first rose and then fell until the reward model became anti-correlated with human judgments.
- **Authors:** Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, et al. (OpenAI)
- **Year:** 2020 (arXiv v1 2020-09; NeurIPS 2020)
- **URL:** https://arxiv.org/abs/2009.01325
- **Source type:** paper
- **Relevant topics:** RLHF, reward modeling, PPO with KL penalty, best-of-N sampling, reward-model over-optimization, domain transfer

## Abstract
Supervised training on human reference summaries and evaluation with ROUGE are both proxies for summary quality. The authors collect a large dataset of human comparisons between summaries, train a reward model (RM) to predict the preferred summary, and fine-tune a summarization policy with reinforcement learning against the RM. On a filtered Reddit TL;DR dataset the resulting models outperform the human reference summaries and much larger supervised models, and they transfer to CNN/DM news articles with quality near the references without news-specific fine-tuning. Analyses show that the RM generalizes to new datasets and that optimizing it gives better summaries than optimizing ROUGE, according to human labelers.

## Key Contributions
- A batch human-feedback loop: sample summaries from several policies → collect pairwise human comparisons → train RM → optimize policy with PPO, repeated while labels accumulate (§3.1, App. C.6).
- Release of 64,832 TL;DR summary comparisons plus TL;DR and CNN/DM evaluation data (§1).
- Human-feedback policies outperform larger supervised policies and transfer to CNN/DM (§4.1, §4.2).
- RM analyses: over-optimization (§4.3, Fig. 5), scaling with data and model size (Fig. 6), probes with edited and perturbed summaries (§4.3), RM vs ROUGE as optimization targets (§4.4, Fig. 7).
- Best-of-N (BoN) used as a training-free, mildly optimized policy for data collection and as an optimization probe (App. C.6 Table 10, App. G.3).

## Key Figures/Tables to Study
- Fig. 1: preference over reference summaries vs model size (pretrain-only, supervised, human feedback).
- Fig. 3: 7-point Likert scores for coverage, accuracy, coherence, overall quality on TL;DR.
- Fig. 5: human preference vs degree of RM optimization (PPO at several β, against earlier RM "rm3").
- Fig. 6: RM validation accuracy vs comparisons (8k–64k) and model size (160M–13B).
- Fig. 7: best-of-N up to N = 2048 optimizing ROUGE vs three RMs.
- Tables 9–10: PPO and BoN policies with KL coefficient and KL from the supervised policy.

## Technical Details
- **Data:** TL;DR filtered to 123,169 posts with reference TL;DRs of 24–48 tokens, ~5% held out for validation; target summaries under 48 tokens (§3.2).
- **Label quality:** labeler–researcher agreement 77% ± 2%; researcher–researcher 73% ± 4% (§3.3).
- **Models:** Transformer decoders "in the style of GPT-3"; human-feedback experiments at 1.3B and 6.7B (§3.4). RM, policy, and value function have the same size (§3.4).
- **RM loss:** `loss(r_θ) = −E_(x,y0,y1,i)∼D [ log σ( r_θ(x, y_i) − r_θ(x, y_{1−i}) ) ]` (§3.4). `x` is the post; `y0, y1` the two summaries; `i` the index of the human-preferred one; `r_θ` the scalar RM output; `σ` the logistic function; `D` the comparison dataset. After training, outputs are shifted so reference summaries score 0 on average (§3.4).
- **RL reward:** `R(x, y) = r_θ(x, y) − β log[ π_φ^RL(y|x) / π^SFT(y|x) ]` (§3.4). `π_φ^RL` is the policy being trained, `π^SFT` the supervised initialization, `β` the KL coefficient. The authors state two roles for the KL term: an entropy bonus against mode collapse, and keeping outputs close to what the RM saw in training (§3.4).
- **PPO structure:** reward only at the end of the summary, episode ends at EOS, γ = 1 (§3.4 fn. 8). Value function is a separate Transformer initialized from the RM; separate parameters outperformed a shared policy/value network (App. G.1, Fig. 11).
- **Best-of-N procedure:** sample N summaries from a supervised baseline at temperature 0.7, score each with an RM, return the highest-scoring one; samples from these policies are part of the RM training data (App. C.6).
- **BoN KL:** `KL(BoN, sup) = log N − (N−1)/N`, computed analytically (Table 10 caption; App. G.3 fn. 15). Reported values: N = 8 → 1.2, N = 64 → 3.2, N = 128 → 3.9, N = 256 → 4.5, N = 512 → 5.2 (Table 10). These match the natural logarithm, e.g. ln 64 − 63/64 = 3.17 (derived).
- **BoN vs PPO:** "at a given average reward, the best-of-N and PPO policies have similar quality as judged by human labelers (not shown)"; PPO policies are farther from the supervised baseline in KL (App. G.3). Final PPO policies have KL(ppo, sup) = 18.0 (1.3B) and 14.0 (6.7B) (Table 9).
- **Results:** 1.3B human-feedback policy 61% vs 43% for the 10× larger supervised policy (§4.1). Length control lowers preference by ~5%; 6.7B still preferred ~65% (§4.1, App. F). 6.7B PPO gets a 7/7 overall score 45% of the time vs 20% (6.7B supervised) and 23% (references) (§4.1).
- **RM scaling and probes:** doubling data gives ~1.1% and doubling model size ~1.8% RM validation accuracy (§4.3, Fig. 6). CNN/DM agreement 62.4% (1.3B) and 66.5% (6.7B) vs 66.9% inter-labeler; improving edits preferred 79.4%/82.8% vs 84.1% humans; shortening edits preferred 62.6% (6.7B) vs 76.4% humans (§4.3).
- **ROUGE:** agreement with labelers ~57% on supervised samples, ~50% on human-feedback samples, where the RMs reach 62% (§4.4); ROUGE optimization peaks sooner and lower than RM optimization (Fig. 7).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Stiennon pretrained models | 1.3B, 6.7B | pretrain-stable | tokens; context; LR shape | 200–300B tokens, 1–3 epochs per source; 2048 tokens; cosine with short warmup to 10% of max | arXiv:2009.01325v3 App. B.1 | verified 2026-09-14 | no ablation reported |
| Stiennon pretrained 1.3B | 1.3B | pretrain-stable | max LR; max batch (sequences of 2048 tokens) | 2e-4; 512 | App. B.1 Table 3 | verified 2026-09-14 | no ablation reported |
| Stiennon pretrained 6.7B | 6.7B | pretrain-stable | max LR; max batch (sequences of 2048 tokens) | 1.2e-4; 512 | App. B.1 Table 3 | verified 2026-09-14 | no ablation reported |
| TL;DR supervised baseline | 1.3B | SFT | initial LR; schedule | 6.35e-5; cosine | App. B.1 | verified 2026-09-14 | log-linear sweep of at least 7 values |
| TL;DR supervised baseline | 6.7B | SFT | initial LR; schedule | 2.83e-5; cosine | App. B.1 | verified 2026-09-14 | log-linear sweep of at least 7 values |
| TL;DR supervised baselines | 1.3B, 6.7B | SFT | batch (unit not stated); epochs; weights precision | 128; 1 epoch; fp16 weights | App. B.1 and fn. 11 | verified 2026-09-14 | no ablation reported |
| Reward model (main results) | 1.3B | reward-model | initial LR; schedule | 1.5e-5; cosine | App. B.1 | verified 2026-09-14 | log-linear sweep of at least 7 values |
| Reward model (main results) | 6.7B | reward-model | initial LR; schedule | 5e-6; cosine | App. B.1 | verified 2026-09-14 | log-linear sweep of at least 7 values |
| Reward models | 1.3B, 6.7B | reward-model | batch (unit not stated); epochs; seeds; head init | 64; 1 epoch; 3–10 seeds, best on dev split; head ~ N(0, 1/(d_model+1)) | App. B.1 | verified 2026-09-14 | seed and data order affect results (App. B.1) |
| Released comparison data | — | reward-model | comparisons | 64,832 (TL;DR) | §1 | verified 2026-09-14 | Fig. 6: RM accuracy rises ~1.1% per data doubling |
| PPO policy (sup4 ppo rm4) | 1.3B | RL | initial LR; batch (unit not stated) | 1.5e-5, linear decay; 512 | App. B.1 | verified 2026-09-14 | "small amounts of experimentation and rough model size extrapolation" |
| PPO policy (sup4_6b ppo rm4_6b) | 6.7B | RL | initial LR; batch (unit not stated) | 7e-6, linear decay; 256 | App. B.1 | verified 2026-09-14 | same as above |
| PPO policies (main) | 1.3B, 6.7B | RL | γ; GAE λ; epochs per rollout batch; episodes | 1; 0.95; 4; 1 million | App. B.1 | verified 2026-09-14 | no ablation reported |
| PPO policies (main) | 1.3B, 6.7B | RL | KL coefficient β (in reward) | 0.05 | App. B.1; Table 9 | verified 2026-09-14 | no selection rationale given; Table 9 lists runs at β = 0.35, 0.10, 0.07, 0.05 against earlier RMs |
| PPO policy | 6.7B | RL | compute | approximately 320 GPU-days | §5 | verified 2026-09-14 | n/a |
| All RM and RL runs | 1.3B, 6.7B | reward-model, RL | precision; optimizer | fp16 activations, fp32 weights; Adam | App. B.1 | verified 2026-09-14 | fp32 weights improved RL (fn. 11) |
| BoN policies (data collection) | 1.3B sup policies; 1.3B and 6.7B RMs | reward-model | sampling temperature; N | 0.7; N ∈ {8, 63, 64, 128, 256, 512} | App. C.6, Table 10 | verified 2026-09-14 | no ablation reported |
| Final human evaluations | 1.3B, 6.7B | eval-gate | sampling temperature | T = 0 | §3.4; App. B.1 Fig. 8 | verified 2026-09-14 | Fig. 8 sweep over T and top-p |
| PPO policies | 1.3B, 6.7B | RL | PPO clip ε; rollout temperature; Adam betas; weight decay | not reported | checked body, App. B, C.6, G | not reported | — |

## Findings relevant to generality and negative feedback
- **Transfer (Result, single study):** TL;DR-trained human-feedback models summarize CNN/DM without news fine-tuning; the 6.7B model nearly matches a 6.7B model fine-tuned on CNN/DM references while producing shorter summaries (§4.2, Fig. 4).
- **Narrowing by RM optimization (Result, single study):** under light optimization human preference improves; with more optimization it falls and the RM becomes anti-correlated with labelers (§4.3, Fig. 5). The authors note ROUGE also over-optimizes (§4.3, App. G.3).
- **Measurement confounds:** length explains about a third of the feedback-vs-supervised gap at 6.7B (Fig. 1 caption); the 6.7B RM under-prefers shortening edits (62.6% vs 76.4%) (§4.3); supervised baselines used fp16 weights, which the authors say slightly disadvantages them (fn. 11).

## Connections
- [[bradley-terry-rm]]: the pairwise logistic RM loss used in §3.4.
- [[kl-control-rlhf]]: the KL penalty inside the reward, with β = 0.05 here.
- [[rlhf-instructgpt]]: a later SFT → RM → PPO pipeline applied to instruction following.
- [[reward-model-overoptimization]]: a scaling study of the effect shown in Fig. 5, including best-of-N.
- [[west-of-n]], [[rejection-sampling-finetuning]]: use best-of-N selection to build training data.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2009.01325 (PDF v3, 2022-02-15).
- Corrections to the previous card version:
  - Title "Best-of-N vs RL — Learning to Summarize with Human Feedback" → "Learning to summarize from human feedback". The paper is about RLHF; BoN appears in App. C.6, App. G.3, and Fig. 7.
  - "BoN is competitive with or superior to full RLHF at small KL" and "BoN-64 is within 2 points of PPO on human eval" → App. G.3 states similar human-judged quality at equal average reward (not shown) and larger KL for PPO; no per-N human-eval numbers are given.
  - "Figure 4 = RM score vs human preference" → Fig. 4 is CNN/DM transfer; over-optimization is Fig. 5. "Figure 6 = BoN vs RL at matched KL" → Fig. 6 is RM accuracy vs data and model size. "Table 3 = axis-level human eval" → axes are Fig. 3; Table 3 lists pretrained model hyperparameters.
  - "Base: GPT-3 1.3B and 6.7B" → Transformer decoders "in the style of GPT-3" (§3.4).
  - "β tuned to match BoN-64 KL" → β = 0.05 for both main runs (App. B.1); their KL(ppo, sup) is 18.0 and 14.0 (Table 9), BoN-64 is 3.2 (Table 10).
  - "KL formula derived in appendix; tight for well-calibrated RM" → stated as an analytic result without derivation or calibration condition (Table 10 caption, fn. 15).
  - "First clear example of reward-model overoptimization" → no priority claim; the paper cites ROUGE over-optimization and similar robotics results (§4.3).
  - "RM trained on 64K pairs" → 64,832 comparisons in the release (§1); each RM trained on all labels collected so far (App. C.6, Table 11).
  - Old deps `[[rlhf-instructgpt]]` → removed from deps, since this paper predates InstructGPT.
- Removed as unsupported by the source: "try BoN with N ∈ {4, 16, 64}; BoN-64 matches PPO at ~1/10 engineering cost"; "monotonic in N until the RM over-optimizes"; "KL-matched comparison underlies every modern RLHF evaluation protocol"; "BoN is the production pattern for Anthropic test-time compute and Cohere chat models".
- Not reported by the source: PPO clip ε, rollout temperature, Adam betas and weight decay, human-evaluation numbers for BoN policies.
