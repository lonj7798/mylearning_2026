<!-- scope: Anthropic's helpful-and-harmless RLHF paper and the human-preference comparison dataset it released
     deps: [[rlhf-instructgpt]]
     see-also: [[hh-rlhf-recipe]], [[constitutional-ai]], [[ultrafeedback]], [[west-of-n]], [[rlcd]]
-->

# Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
- **Core Insight:** Preference models trained only on helpfulness data score below chance on harmlessness data and vice versa, so the two objectives are in measurable tension; larger preference models fit both distributions at once and become more robust to the mixture ratio (§5.1.1, Figure 19).
- **Guideline:** When training a preference model on both helpfulness and harmlessness comparisons, scale the preference model rather than tuning the mixture ratio, because model size — not the ratio — is what made both distributions fit simultaneously in the paper's 10%-interval mixture sweep at a fixed 42k-comparison budget (§5.1.1).
- **Authors:** Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, et al. (Anthropic)
- **Year:** 2022 (arXiv v1 2022-04)
- **URL:** https://arxiv.org/abs/2204.05862 ; dataset: https://huggingface.co/datasets/Anthropic/hh-rlhf
- **Source type:** paper (with an associated released dataset card)
- **Relevant topics:** human preference data, RLHF, preference modeling, helpfulness vs harmlessness, online RLHF, KL–reward scaling

## Abstract
The paper applies preference modeling and reinforcement learning from human feedback to finetune language
models into helpful and harmless assistants. Alignment training improves performance on almost all of the
NLP evaluations tested and is compatible with training for specialized skills such as Python coding and
summarization. The authors explore an iterated online mode of training in which preference models and RL
policies are updated on a weekly cadence with fresh human feedback data. They investigate the robustness of
RLHF training and identify a roughly linear relation between RL reward and the square root of the KL
divergence between the policy and its initialization. Peripheral analyses cover calibration, competing
objectives, out-of-distribution detection, and comparisons with human writers (Abstract).

## Key Contributions
- Two separately collected comparison datasets — an open-ended helpfulness set and a red-teaming
  harmlessness set — released publicly (§2.2; dataset card).
- Evidence that helpfulness and harmlessness compete: training a preference model on one distribution
  alone gives performance on the other that is significantly worse than chance (§5.1.1).
- An approximately linear relation between preference-model reward gain and √D_KL(policy ‖ policy₀)
  during early RLHF training, observed across all their RLHF runs (§4.3, Figures 4 and 13).
- Iterated "online" RLHF: retrain a new preference model and a new policy each round on data collected
  from the previous round's best policy, to fill the high-score tail (§4.5). Measurement of alignment tax
  versus alignment bonus by model size (§1.1).

## Key Figures/Tables to Study
- Figure 4 — train-PM versus test-PM score during RLHF, robust to about 150k training samples and
  divergent beyond, plus the linear reward-vs-√KL relation (§4.2–4.3); Figure 7 — preference-model
  accuracy versus dataset size (one epoch) and versus model size (§3.2).
- Figure 19 — helpfulness and harmlessness test accuracy as the training mixture is varied from 100%
  helpful to 100% harmless (§5.1.1).
- Figure 15 — score distributions of base, rejection-sampled and online data (§4.5); and Figure 10 (right)
  — agreement rates among Anthropic researchers, crowdworkers and a static PM on about 320 samples (§3.4.1).

## Technical Details
- Data collection used 52B language models in three configurations: an HHH context-distilled 52B model;
  rejection sampling against a 52B preference model, most often with k = 16 samples; and a succession of
  RLHF-finetuned models (§2.3).
- Dataset composition (§2.3): base dataset 44k helpfulness + 42k red-teaming comparisons; rejection-sampled
  dataset 52k helpfulness + 2k red-teaming; iterated online dataset 22k helpfulness and no red-teaming.
  A conversation typically comprises about four comparisons (§2.3).
- Only comparisons stronger than the weakest available preference strength are included; all retained
  comparisons are binary and equally weighted, with no ties (§2.2). The released dataset splits helpfulness
  into `helpful-base`, `helpful-rejection-sampled` and `helpful-online`, and harmlessness into
  `harmless-base` only; license is MIT (dataset card).
- Crowdworkers came from MTurk and Upwork; MTurk accounted for about 80% of the datasets, 80–85% of
  comparison data in a given week against 15–20% from Upwork (§2.1, App. D.1). Workers were not filtered on
  agreement; retrospective evaluation found about 63% average agreement (§2.1, Figure 10 right).
- RL used PPO with a KL penalty added to the reward, r_total = r_PM − λ_KL·D_KL(policy ‖ policy₀), with
  λ_KL = 0.001 (§4.1, Eq. 4.1). The remaining preference-model and RL settings are in the split-out ledger
  [[hh-rlhf-recipe]]; the paper gives learning rates only as multipliers on the pretraining learning rate.
- Preference-model accuracy scales roughly log-linearly in both model size and dataset size (§1.1, Figure 7).
- Alignment tax and bonus: RLHF hurt zero-shot NLP evaluation performance for smaller models, while the 13B
  and 52B RLHF models did better zero-shot and the same few-shot (§1.1); §1.2 states the same result with
  "12B and 52B", so the two sections give the mid size slightly differently.
- RLHF on a code-finetuned model improved HumanEval scores, and mixing summarization into preference-model
  training did not degrade HH or summarization accuracy (§1.1, §5.2–5.3).

## Findings relevant to generality and negative feedback
- Over-optimizing harmlessness produced policies that answered nearly every sensitive question with a
  refusal or a referral to professional help, which the authors attribute to harmlessness being easy to
  score highly on ("I can't answer that" suffices) while helpfulness stays under-optimized (§4.4).
- The harmlessness data was collected with crowdworkers steering conversations toward harm and picking the
  *less* harmful of two responses, unlike the helpfulness data; the authors say this asymmetry made it hard
  to train models that are both helpful and harmless, and recommend that others collect harmlessness data
  where workers choose the best possible response instead (§2.2, §4.4).
- RLHF overfitting is measurable: with the static dataset split in half and separate train/test preference
  models, training stayed robust to roughly 150k samples before train and test PM scores diverged (§4.2,
  Figure 4). Larger preference models were more robust (§1.1, App. B.2).
- Preference models are not adversarially robust: a human-written HHH example with subtle inaccuracies was
  confidently mis-scored; the authors note only model-generated samples were used in PM training (§3.4.1).

## Connections
- [[rlhf-instructgpt]] — contemporaneous RLHF work the paper contrasts itself with; the paper notes
  InstructGPT includes a supervised stage, trains no PM above 6B, and does not study harmlessness (§1.3).
- [[constitutional-ai]] — Anthropic's later principle-based replacement for the human harmlessness labels.
- [[ultrafeedback]], [[ultrafeedback-construction]], [[west-of-n]], [[rlcd]] — synthetic-preference
  pipelines that target the same preference-model training stage.
- [[rlaif-scaling]] — AI-feedback alternative to these human comparisons; [[dpo]] — later objective that
  consumes comparison datasets of this shape without an RL loop.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2204.05862 (arXiv v1, 2022-04-12), plus the
  Anthropic/hh-rlhf dataset card and the anthropics/hh-rlhf repository README.
- Corrections to the previous card version:
  - "161K human-preference dialogues" → the paper counts *comparisons*, not dialogues, and gives 44k + 42k
    (base), 52k + 2k (rejection-sampled) and 22k (online) in §2.3; a conversation is about four comparisons.
    161k is the approximate row count of the released HF dataset, not a figure in the paper.
  - "Agreement rates — human-human agreement as a noise floor (~70–75%)" → the paper reports about 63%
    average agreement, measured retrospectively (§2.1; Figure 10 right, about 320 samples).
  - "Established the RLHF pipeline template later followed by OpenAI's InstructGPT, Meta's Llama-2,
    DeepMind's Sparrow" → InstructGPT is contemporaneous and discussed as related work (§1.3); the paper
    makes no such claim.
  - "Demonstrated joint training with both subsets produces a Pareto-improved model" → what §5.1.1 reports
    is that single-distribution training is worse than chance on the other distribution and that larger
    PMs become robust to the mixture ratio; no Pareto-improvement claim is made.
  - "Crowdworkers hold a conversation with a 52B Anthropic assistant" → three model classes were deployed
    to the feedback interface, including rejection-sampling and RLHF-finetuned models, not only the
    context-distilled 52B model (§2.3); §2.2 also restricts the dataset to comparisons above the weakest
    preference strength, which the old card omitted.
- Removed as unsupported by the source:
  - "Length bias — longer responses often preferred": the paper does not analyse response length.
  - "benchmark against an HH-RLHF-trained RM on RewardBench" — RewardBench postdates the paper and appears
    nowhere in it.
  - "first widely released human-preference dataset for dialogue alignment" and "every synthetic-preference
    paper since benchmarks against it" — neither claim is in the paper or the dataset card.
  - "Crowdworker QC with inter-annotator agreement tracking" — App. D states the opposite: workers were not
    filtered on agreement, and the authors found their quality metrics agreed poorly with spot-checks.
  - "Re-processed into `hh-rlhf-binarized` for SFT-vs-DPO ablations" — not a claim of this source.
- Not reported by the source: released row count; number of distinct crowdworkers; GPU hours.
