<!-- scope: controlled comparison of online (on-policy) vs offline preference optimization with the IPO loss; hypothesis tests for the cause of the gap
     deps: [[ipo]], [[dpo]], [[reward-model-overoptimization]]
     see-also: [[trl-online-dpo]], [[self-play-preference]], [[policy-coverage-loss]], [[replay-buffer-rlhf]], [[best-of-n]]
-->

# Understanding the performance gap between online and offline alignment algorithms
- **Core Insight:** With the same pairwise data, the same IPO loss, and the same hyper-parameters, on-policy sampling reaches a higher peak win rate than offline sampling on all four tested datasets; offline policies reach 60–70% pairwise classification accuracy on the preference set (online policies stay below 50%) yet generate worse responses (§2.1 Fig. 1; §5.3.3 Fig. 8).
- **Guideline:** When preference optimization must use a fixed dataset, sample at least one response of each pair from the starting (SFT) policy, because in the §6 ablation pairs with one side from the SFT policy gave a better offline trade-off than pairs sampled only from later online policies (Fig. 12); do not use pairwise classification accuracy on the preference set as a proxy for generation quality (§5.3.2, Fig. 7).
- **Authors:** Yunhao Tang, Daniel Guo, Zeyu Zheng, Daniele Calandriello, Yuan Cao, Eugene Tarassov, et al. (Google DeepMind)
- **Year:** 2024 (arXiv v1 2024-05)
- **URL:** https://arxiv.org/abs/2405.08448
- **Source type:** paper
- **Relevant topics:** online vs offline preference optimization, IPO, reward over-optimization, data coverage, classification vs generation, likelihood of chosen responses, policy scaling

## Abstract
The paper asks whether on-policy sampling is necessary for RLHF-style alignment. In a controlled over-optimization setup, online algorithms reach a better trade-off between KL divergence from the SFT policy and win rate than offline algorithms. The authors test several explanations through ablations. Offline data coverage and offline data quality, each by itself, do not explain the gap. Offline training makes the policy a better pairwise classifier but a worse generator; online training does the opposite. The gap appears for both contrastive (IPO) and non-contrastive (Best-of-2) losses and is not removed by scaling the policy network.

## Key Contributions
- A controlled comparison in which online and offline runs share the loss (IPO), the hyper-parameters, and the source preference data; only the response sampling distribution differs (§2).
- Five stated hypotheses (data coverage, sub-optimal offline data, classification accuracy, non-contrastive loss, policy scaling), each tested with a dedicated ablation (§3, §5).
- Evidence that policies trained offline classify preference pairs better but generate worse, with little correlation between classification accuracy and win rate (§5.3, Figs. 6–8).
- Evidence that making the offline dataset closer to the SFT policy is the dataset change that improves offline learning (§6, Fig. 12).

## Key Figures/Tables to Study
- **Fig. 1:** KL divergence vs win rate against the golden policy for online and offline IPO on four datasets.
- **Fig. 4:** offline IPO trained on the shuffled online-generated dataset (same coverage as online) stays below online.
- **Figs. 7–8:** classification accuracy vs win rate; accuracy over training; log-probability of chosen responses relative to SFT.
- **Figs. 9–11:** Best-of-2 loss; XL vs Large policies; best performance normalized to online across policy sizes.
- **Fig. 12:** offline dataset ablation (D_golden, D_sft vs 800, D_800 vs 4k, D_4k vs 4k).

## Technical Details
- **Loss:** min over θ of E[f(β·(log πθ(y_w|x)/π_sft(y_w|x) − log πθ(y_l|x)/π_sft(y_l|x)))], with f(z) = (z − 0.5)² for IPO and f(z) = log(1 + exp(−z)) for DPO (App. B). πθ is the trained policy, π_sft the SFT reference policy, y_w and y_l the preferred and dispreferred responses, β > 0 the regularization coefficient (larger β = stronger regularization) (§2).
- **Online vs offline:** prompts come from a fixed dataset in both; offline responses come from the fixed dataset, online responses are sampled from πθ (§2). PPO is not studied because it is not clear how to optimize the PPO loss on offline samples (§2).
- **Controlled setting:** a golden preference model (T5X XXL, 11B) is trained on the original pairs and relabels the train split into D_golden; online runs train a proxy preference model on D_golden and optimize against it, offline runs optimize the loss on D_golden directly (§4.1, Fig. 2). Only train-split prompts are used (§4.1).
- **Models:** T5X; policy and proxy preference model are Large (770M) unless stated; scaling runs use XL (3B) and XXL (11B) policies with the same proxy model (§4.1, §5.5).
- **Tasks:** OpenAI summarization, Anthropic helpfulness, Chat arena side-by-side, Anthropic harmlessness (§4).
- **Evaluation:** win rate against a fixed golden policy (trained online against the golden preference model) over 2048 prompts, judged by the golden preference model (§4.1). KL is estimated from 256 training prompts with per-token exact KL (32 prompts for XL) (App. D).
- **Coverage test:** the online stream (batches of B = 32) is saved as D_online; offline training on D_online in its original order reproduces the online run (training statistics match to about 0.1%), and offline training on the uniformly shuffled D_online gives little improvement over plain offline, except on Chat arena sxs (§5.1, Fig. 4; App. F). Performance is robust to a small amount of shuffling and drops once shuffling is sufficient; tested levels range from no shuffling and shuffling every 4 batches to every 500 batches and uniform (App. G, Fig. 13).
- **Quality test:** offline training on pairs sampled from the 4k-step online policy and relabeled by the golden model (20k steps) improves only slightly over SFT; the authors note this contradicts the theoretical one-step-improvement result and attribute it to the full-support assumption on the data policy (§5.2, Fig. 5; App. H).
- **Classification:** the proxy preference model reaches about 70–90% training accuracy; offline policies used as classifiers peak at about 70% (§5.3.1). Within offline runs there is little correlation between classification accuracy on D_golden and win rate (§5.3.2, Fig. 7). No trained policy classifies its own samples significantly better than random (§5.3.3, App. E.1).
- **Best-of-2 loss** (SFT on the winner of each sampled pair, Eq. 3): the gap persists except on Chat arena sxs, and offline Bo2 on shuffled online data matches online, so coverage explains more of the Bo2 gap (§5.4, Fig. 9).
- **Scaling:** peak performance rises with policy size and is reached at a similar KL; over a 16× policy-size increase the online vs shuffled-online gap narrows on two of three tasks, and the online vs offline gap narrows more slowly (§5.5, Figs. 10–11).
- **Dataset ablation:** enlarging the quality difference between the two responses alone does not help (D_800 vs 4k vs D_4k vs 4k); sampling one side from the SFT policy (D_sft vs 800) improves the offline trade-off (§6, Fig. 12).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| T5X Large policy | 770M | SFT | data | pooled pairwise datasets of all four tasks; both responses used as targets | arXiv:2405.08448v1 App. C | verified 2026-09-14 | no ablation reported |
| T5X Large policy | 770M | SFT | batch; LR; steps | 128; 3·10⁻⁵; 4k steps (about 3–4 epochs) | App. C | verified 2026-09-14 | no ablation reported |
| T5X XL / XXL policy | 3B / 11B | SFT | steps; LR; batch | 8000; 10⁻⁵; 16 (XL), 8 (XXL) | App. C | verified 2026-09-14 | no ablation reported |
| T5X Large SFT policy and proxy preference model; T5X XXL golden preference model | 770M; 11B | SFT, reward-model | training context / target length | 1024 / 128 (unit not stated) | App. A.3, App. C | verified 2026-09-14 | no ablation reported |
| Golden preference model, T5X XXL | 11B | reward-model | batch; LR; warmup; stop rule | 16; 10⁻⁴; 1k linear; stop roughly at validation-accuracy plateau | App. A.3 | verified 2026-09-14 | no ablation reported |
| Proxy preference model, T5X Large | 770M | reward-model | batch; LR; warmup | 32; 10⁻⁴; 1k linear | App. A.3 | verified 2026-09-14 | batch chosen to match offline policy batch (App. A.3) |
| Online and offline IPO, Large policy | 770M | preference | steps; LR; β | 4k; 1·10⁻⁵; 0.1 | §4 "Hyper-parameters" | verified 2026-09-14 | adapted from prior work, tuned on OpenAI summarization only (§4) |
| IPO policy runs | 770M / 3B / 11B | preference | batch (prompts, one pair each) | 32 / 16 / 8 | App. B; §5.1 | verified 2026-09-14 | no ablation reported |
| Offline IPO sweep | 770M | preference | LR; β; steps | {3·10⁻⁶, 1·10⁻⁵, 3·10⁻⁵}; {0.1, 0.5, 1}; {4k, 20k} | §4 | verified 2026-09-14 | pooled into Fig. 1 to estimate best offline trade-off |
| IPO runs | all | preference | sampling temperature; optimizer; compute | not reported | checked body, App. A–H | not reported | — |
| All policies | all | eval-gate | win-rate prompts; KL-estimate prompts | 2048; 256 (32 for XL) | §4.1; App. D | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality and negative feedback
- **Generation vs discrimination:** offline training raises pairwise classification accuracy on the static preference set without a matching rise in generation quality; online training improves generation while classification accuracy stays below 50% (§5.3.3, Fig. 8). **Result (single study).**
- **Chosen-response likelihood:** in all experiments the log-probability of the preferred responses in D_golden falls below the SFT value during training, most strongly for offline runs; the authors state that offline optimization often sets chosen likelihoods low and rejected likelihoods lower (negative as gradient) (§5.3.3, Fig. 8 bottom). They note Tajwar et al. (2024) report cases with a smaller decay (Fig. 8 caption).
- **Measurement limits:** only train-split prompts, open-source datasets, and T5X models are used; state-of-the-art pre-trained models are not tested (§1 "Limitations", §4.1).

## Connections
- [[ipo]] — the loss used for both the online and the offline algorithm in every main experiment.
- [[reward-model-overoptimization]] — Gao et al. (2023), whose golden-model and KL-budget setup this paper adopts.
- [[dpo]] — the offline method family; App. B gives DPO as the same template with a logistic f.
- [[trl-online-dpo]] — an implementation of the online variant (sampling responses from the current policy).
- [[self-play-preference]] — Munos et al. (2023), a source of the adapted hyper-parameters (§4).
- [[replay-buffer-rlhf]] — related to the shuffling result: training on stale or reordered on-policy data (App. G).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2405.08448 (arXiv v1, 14 May 2024; full PDF including App. A–H)
- Corrections to the previous card version:
  - URL "arXiv 2404.14367" → 2405.08448. 2404.14367 is Tajwar et al., "Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data", a different paper; the previous card mixed Tang et al.'s title and authors with Tajwar et al.'s ID.
  - Title "Understanding the Performance Gap Between On-Policy and Off-Policy RLHF" → "Understanding the performance gap between online and offline alignment algorithms".
  - "The primary cause of the gap is distribution shift" → the paper reports that offline data coverage and data quality, each by itself, cannot explain the gap (Abstract; §5.1–5.2).
  - "Iterative DPO ≈ PPO; PPO's advantage vanishes when DPO is made on-policy" → the paper compares online vs offline IPO (and Best-of-2); it runs no PPO and no DPO experiments (§2, §5.4).
  - "Gemma-2B and Gemma-7B" → T5X Large (770M), XL (3B), XXL (11B) (§4.1, §5.5).
  - "TL;DR, HH-RLHF helpfulness, GSM8K" → OpenAI summarization, Anthropic helpfulness, Chat arena sxs, Anthropic harmlessness (§4).
  - "DPO β=0.1, 1 grad step per pair, reference π_0" → IPO, β = 0.1, LR 1·10⁻⁵, 4k steps, reference π_sft (§2, §4).
  - Figure descriptions (Fig. 1 "win-rate vs training compute", Fig. 3 "Pareto frontier", Fig. 6 "synthetic distribution-shift control", "Table 2") → replaced with the paper's actual Figs. 1, 4, 7–12.
- Removed as unsupported by the source: "≈80% distribution shift / ≈20% variance reduction decomposition"; "formalized coverage argument via the implicit reward's normalization constant"; PPO recipe "clip 0.2, adaptive KL"; "offline DPO trained on fixed Anthropic HH pairs"; "100K–1M prompts"; "offline DPO sits 5–10 pts below"; "lower variance across seeds"; the previous Guideline "the fix is to sample chosen/rejected pairs from the current policy each step" (not tested in this form).
- Not reported by the source: online sampling temperature, optimizer, compute budget, test-split generalization.
