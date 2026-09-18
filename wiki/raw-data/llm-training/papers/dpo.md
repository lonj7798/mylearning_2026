<!-- scope: Direct Preference Optimization (Rafailov et al., NeurIPS 2023): KL-constrained RLHF rewritten as a binary cross-entropy loss on preference pairs; experiments on IMDb, Reddit TL;DR, Anthropic HH
     deps: [[rlhf-instructgpt]], [[bradley-terry-rm]]
     see-also: [[ipo]], [[simpo]], [[orpo]], [[kto]], [[on-off-policy-rlhf]], [[openrlhf-dpo]]
-->

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **Core Insight:** Writing the reward as r(x,y) = β log π(y|x)/π_ref(y|x) turns the KL-constrained RLHF objective into a binary cross-entropy loss on preference pairs (§4, Eq. 7); on Reddit TL;DR with a GPT-J SFT model, DPO reached about 61% GPT-4 win rate against reference summaries at temperature 0.0, versus 57% for PPO at its best temperature (§6.2).
- **Guideline:** When an offline preference dataset sampled from an SFT model is available, set π_ref to that SFT model and minimize Eq. 7; when no SFT model exists, first fine-tune on the preferred completions to reduce the shift between the data distribution and π_ref (§4 "DPO outline"), because this is the procedure behind the paper's IMDb, TL;DR and HH results (§6).
- **Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year:** 2023 (arXiv v1 2023-05; NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.18290
- **Source type:** paper
- **Relevant topics:** preference optimization, RLHF without RL, implicit reward, Bradley-Terry, negative gradient on rejected responses, LLM-as-judge evaluation

## Abstract
RLHF fits a reward model to human preference labels and then fine-tunes the LM with RL to maximize that reward while staying close to the original model; the authors describe this as complex and often unstable. The paper introduces a reparameterization of the reward model under which the optimal policy of the KL-constrained objective has a closed form. The RLHF problem can then be solved with a classification loss. The resulting method, DPO, needs no sampling from the LM during fine-tuning and no significant hyperparameter tuning. In the experiments, DPO exceeds PPO-based RLHF at controlling sentiment and matches or improves response quality in summarization and single-turn dialogue.

## Key Contributions
- Derives that the optimum of the KL-constrained objective is π_r(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x) (Eq. 4, App. A.1) and inverts it for r (Eq. 5).
- Substitutes Eq. 5 into the Bradley-Terry model, where Z(x) cancels, giving the DPO loss (Eq. 6–7, App. A.2); a Plackett-Luce version is in App. A.3.
- Proves that, under mild assumptions, the reparameterization represents every reward equivalence class consistent with Plackett-Luce / Bradley-Terry models (§5.1, Theorem 1, App. A.6).
- Interprets the normalizer in actor-critic RLHF as a soft value function whose absence gives high-variance policy gradients (§5.2, Eq. 10).
- Shows that on IMDb sentiment the DPO reward–KL frontier dominates PPO, including PPO with ground-truth reward (§6.1, Fig. 2 left).

## Key Figures/Tables to Study
- **Fig. 2 left:** expected reward vs sequence-level KL(π‖π_ref) on IMDb, 22 runs in the sweep over DPO, PPO, Unlikelihood and Preferred-FT settings; PPO-GT is also plotted (§6.1).
- **Fig. 2 right / Fig. 3 left:** GPT-4 win rates vs sampling temperature on TL;DR (vs reference summaries) and on Anthropic HH (vs the chosen response).
- **Table 1:** out-of-distribution transfer to CNN/DailyMail. **Table 2:** GPT-4 vs human agreement. **Table 3:** Unlikelihood baseline samples.

## Technical Details
- **RLHF objective (Eq. 3):** max_π E_{x~D, y~π}[r_φ(x,y)] − β D_KL[π(y|x) ‖ π_ref(y|x)]; x = prompt, y = response, π_ref = the SFT model, β = strength of the constraint (§3).
- **Reward reparameterization (Eq. 5):** r(x,y) = β log [π_r(y|x)/π_ref(y|x)] + β log Z(x), where Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β) is the partition function (§4).
- **DPO loss (Eq. 7):** L_DPO = −E_{(x,y_w,y_l)~D} [ log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) ) ].
  y_w = preferred response; y_l = dispreferred response; π_θ = trained policy; σ = logistic function.
- **Implicit reward:** r̂_θ(x,y) = β log π_θ(y|x)/π_ref(y|x) (§4, §5.1).
- **Gradient (§4; App. A.4 Eq. 21–22):** ∇_θ L_DPO = −β E[ σ(r̂_θ(x,y_l) − r̂_θ(x,y_w)) · (∇_θ log π(y_w|x) − ∇_θ log π(y_l|x)) ].
  The weight σ(·) is larger when the implicit reward ranks y_l above y_w; the paper labels the two gradient terms "increase likelihood of y_w" and "decrease likelihood of y_l".
- **Reference code (App. B):** `losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))`; `rewards = beta * (pi_logps - ref_logps).detach()`.
- **Tasks and models (§6, App. C.1):** IMDb prefixes of 2–8 tokens with GPT-2-large and the `siebert/sentiment-roberta-large-english` classifier as ground-truth reward; Reddit TL;DR with the GPT-J SFT model `CarperAI/openai_summarize_tldr_sft` and the Stiennon et al. preferences; Anthropic HH (170k dialogues) with Pythia-2.8B. Largest model: 6B (§1, §7).
- **Results (§6.2):** DPO temperature-0.25 samples were preferred by humans 58% of the time over PPO temperature-0 samples; on HH, DPO is the only computationally efficient method that improves over the chosen completions, and is similar to or better than Best of 128 (the Best-of-N baseline plateaus at N = 128, App. D.1 Fig. 4). A PPO model for HH from a public source did not beat the Pythia-2.8B base model at any prompt or temperature tried (§6.2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All DPO runs unless noted | per task (rows below) | preference | β | 0.1 | arXiv:2305.18290v3 App. B | verified 2026-09-14 | "we did not meaningfully tune DPO's β" (§6.2) |
| GPT-J SFT (CarperAI/openai_summarize_tldr_sft), TL;DR | 6B | preference | β | 0.5 | App. B | verified 2026-09-14 | no ablation reported |
| All DPO runs | per task | preference | batch size | 64 (unit not stated) | App. B | verified 2026-09-14 | no ablation reported |
| All DPO runs | per task | preference | optimizer; peak LR; warmup | RMSprop; 1e-6; linear from 0 to 1e-6 over 150 steps | App. B | verified 2026-09-14 | no ablation reported |
| All DPO runs | per task | preference | epochs; LR decay; max sequence length | not reported | checked §6, App. B, App. C | not reported | — |
| GPT-2-large, IMDb | not stated | preference | β sweep for frontier | {0.05, 0.1, 1, 5} | §6.1 | verified 2026-09-14 | Fig. 2 left, 22 runs in total, evaluated every 100 steps |
| GPT-2-large, IMDb | not stated | SFT | training length | "until convergence" (§6) vs "1 epoch" on a subset of IMDB (App. C.1) | §6; App. C.1 | conflict | the paper gives both; no released config checked |
| GPT-2-large, IMDb | not stated | preference | pairs | 4 completions for each of 25000 prefixes; 6 pairs per prefix | App. C.1 | verified 2026-09-14 | labeled by the sentiment classifier |
| GPT-2-large, IMDb (PPO baseline) | not stated | reward-model; RL | RM epochs; PPO target KL; PPO batch | 3 epochs, best validation accuracy; {3, 6, 9, 12}; 1024 samples per PPO step | §6.1; App. C.1 | verified 2026-09-14 | Fig. 2 left |
| Pythia-2.8B, Anthropic HH | 2.8B | SFT | reference model | Preferred-FT on chosen completions | §6; §6.2 | verified 2026-09-14 | no standard SFT model exists for HH (§6) |
| All win-rate evaluations | — | eval-gate | judge | gpt-4-0314; prompt (C) for summarization; random A/B order | App. C.2; §6.4 | verified 2026-09-14 | Table 2: GPT-4 (C) closer to human win rates than GPT-4 (S) |

## Findings relevant to generality and negative feedback
- **Negative feedback (negative as gradient).** Eq. 7 lowers log π_θ(y_l|x) for every pair. The paper reports that a variant without the σ weight "can cause the language model to degenerate" (§4); the Unlikelihood baseline (maximize log p(y_w|x), minimize log p(y_l|x), coefficient α ∈ [0, 1]) produced "generally meaningless responses", for example repeated "when" tokens on TL;DR at temperature 1.0, so it was excluded from summarization and dialogue (App. C.3, Table 3). The authors attribute this to "unconstrained likelihood minimization" (Interpretation).
- **Not reported by this source:** separate training curves of log π_θ(y_w) and log π_θ(y_l); where probability mass removed from y_l goes; effects of pairs sampled from a policy other than π_ref. [[on-off-policy-rlhf]] reports that chosen-response log-probability falls below its SFT value during offline training (§5.3.3 there).
- **Out-of-distribution transfer.** TL;DR-trained policies on CNN/DailyMail, GPT-4 win rate vs ground-truth summaries: DPO 0.36 (temp 0) and 0.31 (temp 0.25); PPO 0.26 and 0.23 (§6.3, Table 1). DPO did not use the additional unlabeled Reddit prompts that PPO used. The authors call this "initial evidence" and list OOD generalization as an open question (§7).
- **Temperature robustness.** PPO on TL;DR "can degrade to that of the base GPT-J model at high temperatures"; DPO is reported as more robust (§6.2, Fig. 2 right).
- **Over-optimization.** The authors ask whether the slight late-training decrease in Fig. 3 right is reward over-optimization (§7; Open question).
- **Measurement error.** With the simple prompt (S), GPT-4 prefers longer, more repetitive summaries than humans do (§6.4). On the DPO comparison, GPT-4 (C)–human agreement is 67% and human–human agreement is 65% (Table 2).

## Connections
- [[rlhf-instructgpt]], [[ppo]]: the reward-model + PPO pipeline whose objective (Eq. 3) DPO optimizes without RL.
- [[bradley-terry-rm]]: the paired-comparison model used in Eq. 1 and Eq. 6.
- [[ipo]]: replaces the logistic loss on the log-ratio margin with a squared loss.
- [[simpo]], [[orpo]]: preference losses that remove the reference model.
- [[kto]]: learns from unpaired binary desirable/undesirable labels.
- [[rpo]]: iterative DPO on reasoning pairs with an added NLL term on the chosen response.
- [[on-off-policy-rlhf]]: controlled online vs offline comparison, including chosen-likelihood decline.
- [[openrlhf-dpo]], [[hf-dpo-zoo]]: framework implementations and TRL-based experiments.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2305.18290 (arXiv v3, 29 Jul 2024; PDF text including App. A–D)
- Corrections to the previous card version: "Figure 3: DPO dominates chosen-over-rejected win rates" → Fig. 3 left reports GPT-4 win rate against the HH chosen response, where DPO is "the only computationally efficient method that improves over the preferred completions" (§6.2); "Figure 2: DPO beats PPO at every sampling temperature" → Fig. 2 left is the IMDb reward–KL frontier, Fig. 2 right shows DPO above PPO's best case on TL;DR and more robust to temperature (§6.2); "matches or exceeds PPO on summarization, dialogue, and sentiment control" → exceeds PPO on sentiment, matches or improves on summarization and single-turn dialogue (Abstract); gradient weight "violations get full weight" → the weight σ(r̂_l − r̂_w) is at most 1 and is multiplied by β (§4).
- Removed as unsupported by the source: β range "0.01–0.5" and "most recipes use 0.1"; learning rate "5e-7 to 1e-6"; batch "32–128 pairs"; epochs "1–3"; "length normalization off (known failure mode, see SimPO)"; Guideline advice on length inflation and switching to SimPO; "one forward/backward per preference pair"; implicit reward "used for gating (BoN, rejection sampling, online iterations)"; "the single most-implemented equation in modern alignment"; "self-rewarding LM" connection (no card).
- Not reported by the source: epochs, max sequence length, LR decay, compute, number of training steps for TL;DR and HH runs.
