<!-- scope: preference pairs built from contrasting positive and negative prompts to the same base model, with no post-hoc scoring
     deps: [[constitutional-ai]]
     see-also: [[west-of-n]], [[ultrafeedback]], [[rlaif-scaling]]
-->

# RLCD: Reinforcement Learning from Contrastive Distillation for LM Alignment
- **Core Insight:** Generating a preference pair from two contrasting prompts and labeling by which prompt produced which output gives preference models that agree with held-out human labels 52.4% (harmlessness) and 64.4% (helpfulness) of the time at 7B simulation scale, against 35.6% and 60.6% for a re-implemented RLAIF that scores two i.i.d. outputs (Table 5, §5.1).
- **Guideline:** When simulating preference pairs with a base model of roughly 7B scale, generate o⁺ and o⁻ from a positive and a negative prompt and label by prompt provenance rather than rescoring, because post-hoc rescoring of the same pairs loses 86.0 / 14.0 in GPT-4 comparisons at 7B; at 30B simulation scale rescoring becomes competitive (Table 6, §5.2).
- **Authors:** Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, Yuandong Tian (Meta AI, UC Berkeley, UCLA)
- **Year:** 2023 (arXiv v1 2023-07; ICLR 2024; arXiv v3 2024-03)
- **URL:** https://arxiv.org/abs/2307.12950
- **Source type:** paper
- **Relevant topics:** contrastive prompting, simulated preference data, RLAIF baseline, context distillation, label noise

## Abstract
RLCD aligns a language model to a principle stated in natural language without human feedback. For each prompt p
it constructs a positive prompt p⁺ that encourages the target attribute and a negative prompt p⁻ that encourages
its opposite, samples o⁺ ~ G(·|p⁺) and o⁻ ~ G(·|p⁻) from the same unaligned model, and labels o⁺ as preferred with
no further scoring (§3.1). A preference model is fine-tuned from the same unaligned model on those pairs, converted
to a reward model, and used to run PPO on the unaligned model (§3.1.1). On harmlessness, helpfulness, and story-outline
generation, RLCD outperforms a re-implemented RLAIF baseline and a context-distillation baseline in both human and
GPT-4 pairwise evaluation, at 7B and 30B preference-data simulation scales, with the exception of RLCD30B vs RLAIF30B
where the two are close (Tables 2 and 3, §4).

## Key Contributions
- **Contrastive prompt pairs as a preference-data primitive**, labeled by prompt provenance, not by a scorer (§3.1).
- **Two criteria for prompt construction (§3.2):** p⁺ should be more likely than p⁻ to elicit the attribute, and the
  surface forms of p⁺ and p⁻ should be as similar as possible so the induced distributions differ as little as
  possible on orthogonal axes. The authors report the second criterion is the one worth attention in practice.
- **Preference-model evaluation on 2000 gold human-labeled examples**: the RLAIF baseline's harmlessness preference
  model is below chance at both scales (Table 5, §5.1).
- **RLCD-Rescore ablation** that isolates the labeling step from the generation step (Table 6, §5.2).
- **Theoretical account (App. N)** of why prompt-provenance labels help near the decision boundary, plus released
  code and simulated preference data at https://github.com/facebookresearch/rlcd (§1).

## Key Figures/Tables to Study
- **Figure 1 (§1):** RLCD, RLAIF, and context distillation side by side; p⁺ and p⁻ differ only in "harmless" vs "harmful".
- **Table 1 (§3.3):** both RLAIF outputs advise the same unethical action, yet one is scored at the 60th percentile
  of label polarity.
- **Tables 2 and 3 (§4):** human Likert and GPT-4 comparisons against all three baselines at both scales.
- **Table 5 (§5.1) and Table 36 (App. K.1):** preference-model and simulated-label accuracy against human labels.
- **Table 6 (§5.2):** RLCD vs RLCD-Rescore at 7B and 30B.

## Technical Details
- **Models:** preference data is simulated with base LLaMA-7B or LLaMA-30B (RLCD7B, RLCD30B); the model being
  aligned is base LLaMA-7B in every configuration, including the baselines (§4, "RLCD Implementation").
- **Prompt sets:** harmlessness and helpfulness prompts are derived from the training sets of Bai et al. (2022a),
  each slightly over 40000 conversations; the outlining set uses 40000 premises of roughly 10–40 tokens (§4).
- **Prompt pairs:** 16 context-phrase pairs for harmlessness (sampled at random per use), 1 for helpfulness, 3 for
  outlining (interestingness, well-formedness, premise relevance); all zero-shot (§4).
- **Human evaluation (Table 2, §4):** 200 examples per comparison on an 8-point Likert scale, normalized so higher is
  better and reported as a pair summing to 9. RLCD7B vs RLAIF7B: 5.62 / 3.38 (harm), 4.64 / 4.36 (help on the
  harmlessness set), 5.88 / 3.12 (helpfulness), 5.97 / 3.03 (outline quality).
- **GPT-4 evaluation (Table 3, §4):** 1000 examples per comparison. RLCD7B vs RLAIF7B is 84.8 / 15.2, 71.0 / 29.0,
  85.4 / 14.6, 78.5 / 21.5. RLCD30B vs RLAIF30B is 60.3 / 39.7, 55.3 / 44.7, 47.8 / 52.2, 35.9 / 64.1 — RLAIF30B is
  preferred on outline quality.
- **Preference-model agreement with 2000 gold human labels (Table 5, §5.1):** accuracy / mean probability on the
  gold output — RLAIF7B 35.6 / 0.492, RLCD7B 52.4 / 0.516, RLAIF30B 45.7 / 0.489, RLCD30B 55.9 / 0.542 on
  harmlessness; 60.6 / 0.508, 64.4 / 0.601, 66.2 / 0.551, 66.7 / 0.628 on helpfulness.
- **Simulated-pair label accuracy under a held-out human-trained reward model (Table 36, App. K.1):** harmlessness
  RLAIF7B 0.44, RLCD7B 0.54, RLAIF30B 0.46, RLCD30B 0.60; helpfulness 0.56, 0.68, 0.66, 0.74.
- **Output length (Table 35, App. K.2, 30B simulation):** RLCD30B gives 73.3 ± 56.1 tokens on harmlessness,
  108.3 ± 71.4 on helpfulness, 138.7 ± 66.9 on outlining; Context-Dist30B is shortest at 30.5 / 37.7 / 59.3.

## Recipe ledger
Moved to **[[rlcd-recipe]]** to keep this card under 120 lines.
## Findings relevant to negative feedback and generality
- **Where the negative comes from:** o⁻ is generated by a negative prompt and is used as the rejected side of a
  Bradley-Terry-style preference pair, so it is negative-as-gradient only through the preference-model loss; the
  aligned policy never sees an explicit likelihood penalty on o⁻ (§3.1).
- **How much contrast is required (App. N):** under the paper's Gaussian model, μ(p⁺) − μ(p⁻) = 0 gives 0.5 label
  accuracy, so the contrast is what produces the signal. At μ(p⁺) − μ(p⁻) = 3, accuracy on pairs whose true attribute
  values differ by at most 0.2 is 0.574, against 0.528 for RLAIF-style scoring, over 10⁸ simulated trials. A very
  large gap drives accuracy toward 1 but makes every example easy, so the authors keep the gap moderate and suggest
  weakening the prompts at larger model scales where σ_G and σ_D are smaller.
- **Labeling vs generating (§5.2):** RLCD beats RLCD-Rescore 86.0 / 14.0, 75.8 / 24.2, 86.3 / 13.7, 88.8 / 11.2 at 7B,
  but only 54.6 / 45.4, 53.2 / 46.8, 47.3 / 52.7, 36.4 / 63.6 at 30B. The authors read this as LLaMA-7B being better
  at generating contrasting outputs than at labeling them after the fact.
- **Generality:** outlining was included for its longer-range planning and multi-attribute requirements (§4). RLCD's
  alignment target is only the attribute written into the prompt pair; the harmlessness prompts of Bai et al. (2022b)
  also request helpfulness (App. A.1), which is why helpfulness is measured on the harmlessness set. No held-out-task
  or capability-retention evaluation is reported.
- **Mixed human labels (Table 40, App. N):** with 20% human-labeled pairs mixed into both methods, RLCD7B still beats
  RLAIF7B (68.9 / 31.1, 59.4 / 40.6, 55.8 / 44.2), by smaller margins than in Table 3.

## Connections
- Re-implements the RLAIF procedure of **[[constitutional-ai]]** as its main baseline, and reuses its own positive-prompt
  generations as the context-distillation baseline.
- **[[rlaif-scaling]]** reports the opposite verdict on scoring-based AI labels at PaLM 2 scale; the two disagree at
  different model scales and with different labelers.
- **[[west-of-n]]** builds pairs from the best and worst of N samples under a single prompt, which is the i.i.d.
  generation RLCD argues against.
- **[[ultrafeedback]]** and **[[ultrafeedback-construction]]** use a stronger external judge for multi-aspect ratings;
  RLCD requires no external judge but carries only the attribute named in the prompt pair.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2307.12950 (arXiv v3, 16 Mar 2024)
- Corrections to the previous card version:
  - Title "…for Language-Model Alignment" → "RLCD: Reinforcement Learning from Contrastive Distillation for LM Alignment".
  - "using 7B and 30B Llama/Llama-2 backbones" → LLaMA-7B and LLaMA-30B (LLaMA 1); 7B and 30B are the *preference-data
    simulation* scales, and the model aligned by PPO is base LLaMA-7B in every run (§4).
  - "cleaner labels than Best-of-N sampling under a single prompt" → the comparison in the paper is against RLAIF,
    which scores two i.i.d. outputs of the same prompt; no Best-of-N baseline is run (§3.3, §4).
  - "RLCD beats RLAIF at matched compute on three principle-alignment tasks" → RLCD30B and RLAIF30B are close, and
    GPT-4 prefers RLAIF30B on outline quality 64.1 / 35.9 (Table 3, §4); no matched-compute claim is made.
  - "Scaling curve from 7B to 30B" → no scaling curve; the scales are separate rows of Tables 2, 3, 5, 6.
  - "Ablation: random-prompt negatives (no explicit contrast) underperform" → the ablation actually run is RLCD-Rescore,
    which keeps the contrastive generations and relabels them with RLAIF's scoring prompts (§5.2). The claim that
    contrast is what carries the signal is supported instead by App. N: μ(p⁺) − μ(p⁻) = 0 gives 0.5 label accuracy.
  - "Mode collapse in negatives: if the negative prompt just yields refusals" → the paper reports mode collapse from
    using too many PPO steps on the outlining task (App. E), not from negative prompts producing refusals.
  - "Cost estimate: 2× sampling cost per prompt" → not stated; §3.3 says only that RLAIF's post-hoc scoring costs
    extra compute and a longer effective context than RLCD.
- Removed as unsupported by the source: "Demonstrated measurably cleaner preference labels than single-prompt
  Best-of-N"; "Base-model leakage: negatively-prompted outputs can still inadvertently produce helpful content;
  require filtering"; "optional dedup / length-match" filtering; "A 2024 synthetic-preference-for-DPO variant … is
  widely used in open alignment stacks"; "Directly informed later 'contrastive prompt engineering' lines like
  Evolutionary Contrastive Distillation".
- Not reported by the source: the number of preference pairs used for preference-model training; wall-clock or GPU
  cost; any DPO variant; any general-capability benchmark for the aligned model.
