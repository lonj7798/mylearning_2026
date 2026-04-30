<!-- scope: West-of-N synthetic preference data via best/worst-of-N
     deps: [[rlaif-scaling]]
     see-also: [[self-rewarding-lm]], [[spin]], [[ultrafeedback-construction]]
-->

# West-of-N — Synthetic Preferences from Best-of-N Sampling
- **Core Insight:** High-quality preference pairs can be fabricated by sampling N responses from the policy, running them through an existing reward model, and pairing the argmax with the argmin — the resulting `(best_of_N, worst_of_N)` pairs are sharper than human labels and yield preference models as good as or better than those trained on human data.
- **Guideline:** When you have a decent reward model but expensive human preferences, generate `(best, worst)` pairs by sampling 16–64 responses per prompt and keeping the argmax/argmin under your RM — then use those pairs to train the next RM or do DPO.
- **Authors:** Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn
- **Year:** 2024 (Google Research)
- **URL:** https://arxiv.org/abs/2401.12086
- **Relevant topics:** synthetic preferences, best-of-N, reward model training, preference data scaling

## Abstract
The performance of Reinforcement Learning from Human Feedback (RLHF) is bottlenecked by the quality of the preference feedback used to train reward models. We show that a simple synthetic data generation strategy can produce preferences on par with those from humans. Specifically, we generate preference pairs by sampling N responses per prompt and pairing the highest-ranked with the lowest-ranked under an existing reward model. We apply this "West-of-N" sampling to self-improve a base reward model, showing that a single iteration of West-of-N preference generation can improve reward-model accuracy as much as a doubling of the human preference dataset size.

## Key Contributions
- Introduces **West-of-N sampling**: `(best_of_N, worst_of_N)` pairs drawn under the current reward model.
- Shows self-improvement of a reward model via one West-of-N iteration matches the gain from **doubling** the human preference dataset.
- Establishes that the gain comes from the *extremum pairing*, not just from any synthetic pair — argmax-vs-argmin is sharply preferred over argmax-vs-random.
- Works with both on-policy and off-policy generators and across both Bradley-Terry and DPO-implicit reward models.

## Key Figures/Tables to Study
- **Figure 1 (RM accuracy on test vs training-set size):** West-of-N curve above the human-preference curve at every size; gap widens with N.
- **Figure 3 (N ablation):** N=16 is near-optimal; marginal returns beyond N=32.
- **Table 1 (gain per iteration):** 1 iter of West-of-N ≈ +6% RM accuracy; 2 iters saturates.
- **Table 3 (best-worst vs best-random):** 4-point RM accuracy loss if you pair best against random rather than against worst.

## Technical Details
- **Base generator:** an SFT model (TULR, PaLM-2-XXS across ablations).
- **Base reward model:** trained on seed human preference data (≈20K pairs).
- **West-of-N loop:**
  1. Sample N=16 responses per prompt at T=1.0.
  2. Score all N with current RM.
  3. Emit `(y_max, y_min)` pair as synthetic preference.
  4. Train next-iter RM on seed + synthetic pairs (50/50 mix).
- **Iterations:** 1 dominant gain; gains saturate at 2.
- **Prompt pool:** 10K prompts from Reddit TL;DR + HH-RLHF.
- **Diagnostics:** RM agreement with held-out human preference; also downstream DPO win-rate on the policy trained against it.

## Connections
- Sibling of rejection-sampling fine-tuning (RSFT): both exploit best-of-N; RSFT keeps only the best, West-of-N uses the pair.
- Directly generalizes what [[self-rewarding-lm]] does internally: its "best judge score vs worst judge score" pairing is West-of-N with N=4 and the judge replacing the RM.
- Provides data for [[spin]]-style iteration where the rejected sample comes from a policy instead of human text.
- Related to [[reward-ensembling]]: both try to harden RM training against overoptimization; West-of-N does it via data, ensembling via averaging.
- Currently deployed in many open reward-model training recipes (e.g., Ultrafeedback augmentation pipelines).
