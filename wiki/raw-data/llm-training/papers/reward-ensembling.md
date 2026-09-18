<!-- scope: ensembles of reward models with conservative aggregation as a defense against overoptimization
     deps: [[reward-model-overoptimization]]
     see-also: [[generative-reward-models]], [[reward-hacking-taxonomy]], [[warm-weight-averaged-reward-models]], [[bradley-terry-rm]], [[best-of-n]], [[west-of-n]]
-->

# Reward Model Ensembles Help Mitigate Overoptimization
- **Core Insight:** In a synthetic RLHF setup with a 7B gold reward model and 7M–1.3B proxy reward models, aggregating a five-member reward-model ensemble conservatively — worst-case (minimum) or uncertainty-weighted (mean minus λ times intra-ensemble variance) — eliminates overoptimization for best-of-n sampling and improves gold reward by up to about 70% with clean labels and about 75% with 25% label noise (Abstract; §5, Figure 3).
- **Guideline:** When training a reward model for best-of-n or PPO and label noise is expected, train four or five reward models from the same data with different random seeds and optimize against the minimum (WCO) or against mean minus λ·variance (UWO), because both match or outperform single-reward-model optimization in every reported setting (§5). For PPO, add a KL penalty of 0.01 alongside the conservative objective, because the conservative objective alone reduces but does not eliminate overoptimization, and a KL penalty alone needs a 20x larger weight of 0.2 with a performance cost (§5.3, Figure 6).
- **Authors:** Thomas Coste, Usman Anwar, Robert Kirk, David Krueger
- **Year:** 2023 (arXiv v1 2023-10; ICLR 2024)
- **URL:** https://arxiv.org/abs/2310.02743
- **Source type:** paper
- **Relevant topics:** reward-model ensembles, conservative optimization, reward uncertainty, overoptimization, best-of-n, PPO

## Abstract
The paper reuses the synthetic overoptimization setup of Gao et al. (2023), in which a much larger "gold"
reward model stands in for human preferences and labels the data used to train smaller proxy reward
models. It evaluates two ensemble-based conservative optimization objectives — worst-case optimization
(WCO) and uncertainty-weighted optimization (UWO) — under best-of-n sampling (BoN) and PPO, and extends
the setup with 25% label noise to approximate human annotator disagreement. With and without label noise,
conservative optimization eliminates overoptimization for BoN and improves performance by up to 70%. For
PPO it always reduces overoptimization and outperforms single-reward-model optimization; combined with a
small KL penalty it prevents overoptimization at no performance cost (Abstract).

## Key Contributions
- The first study of reward-model ensembles as a defense against overoptimization in RLHF fine-tuning of
  language models (§1, contribution list).
- Three aggregation objectives, defined over an ensemble {R_1, …, R_k} of proxy reward models (§3):
  - Mean: `R_µ(q,a) = (1/k) Σ_i R_i(q,a)`. The paper states this is **not** conservative, because a single
    overestimating member is not suppressed by the average (§3).
  - WCO (worst-case optimization): `R_WCO(q,a) = min_i R_i(q,a)`. No hyperparameter to tune, but it can
    cost performance because it is highly conservative (§3).
  - UWO (uncertainty-weighted optimization): mean minus λ times the intra-ensemble **variance** (§3, Eq. 5).
- Extension of the Gao et al. setup with 25% label noise, approximating reported human annotator
  disagreement rates of about 25% or more (§4.3).
- Scaling evidence: the ensemble gain is additive with proxy reward-model size (Figure 8) and with
  preference-dataset size (Figure 9), so the two approaches can be combined (§1, §5.3).

## Key Figures/Tables to Study
- **Figure 1** — the RLHF pipeline, with the paper's modifications to the Gao et al. setup highlighted.
- **Figure 3** — BoN results with and without 25% label noise; mean optimization overoptimizes under noise
  while WCO and UWO do not.
- **Figure 5 / Figure 6 / Figure 7** — PPO without a KL penalty, and PPO across KL penalty weights under
  25% label noise.
- **Figures 8–9** — final gold reward against proxy reward-model size and against dataset size.
- **Figure 10** — intra-ensemble variance during PPO training for UWO versus mean optimization.
- **Figures 11–12** — robustness to ensemble cardinality (3, 4, 5 members) and to the UWO penalty weight.

## Technical Details
- **Data:** the AlpacaFarm variant of the 52,000-instruction Alpaca dataset, which supplies per-stage splits
  plus human preference annotations (§4.1). SFT uses the 10k "sft" split (§4.3).
- **Models:** policy is Pythia 1.4B everywhere. Proxy reward models are Pythia 14M, 70M, and 1.4B with the
  unembedding layer removed and a scalar head added, giving 7M, 44M, and 1.3B reward models. The gold reward
  model is the 7B AlpacaFarm human-preference reward model (§4.2).
- **Preference labels:** the SFT model produces two responses per instruction; the gold reward model scores
  both; 25% of the labels are optionally flipped to simulate annotator disagreement (§4.3).
- **Reward-model training:** cross-entropy on 46,000 prompts for 5 epochs; trained proxy reward models reach
  60–75% validation accuracy (§4.3); each epoch lasts 359 steps (App. F.1).
- **Ensemble construction:** members share identical data and hyperparameters and differ only in random
  seed, which changes the scalar-head initialization and the data shuffling order. Every member gets all the
  training data, citing Lakshminarayanan et al. (2017) that less data raises validation loss. Unless stated
  otherwise the ensemble has five members (§4.3).
- **Policy optimization:** BoN is evaluated up to n_max = 12,500 samples, roughly 8.4 KL nats. PPO runs for
  3000 steps, except 1.3B reward models at 6000 steps (§4.3; Figure 8 caption).
- **UWO penalty weight:** λ = 0.5 was most performant for the reported BoN results (§5.2); Figure 10 uses
  λ = 0.1; Figure 12 reports that most reasonable values work (§5.4).
- **Ensemble cardinality:** a noticeable gap between 3-member and 4-member ensembles, and near-identical
  performance for 4 and 5 members, indicating diminishing returns past 4–5 (§5.4).
- **Variance dynamics:** for 44M reward models under PPO, intra-ensemble variance grows by almost 3x under
  mean optimization with clean labels and about 2.5x under 25% label noise, versus about 20% under UWO with
  λ = 0.1 (§5.5, Figure 10).
- **Code:** https://github.com/tlc4418/llm_optimization (§1 footnote).

## Recipe ledger
Moved to [[reward-ensembling-recipe]] to keep this card under the 120-line limit.

## Findings relevant to generality and to negative feedback
- Members share the same pretrained base and the same training data, so the robustness claim is scoped to
  seed-level diversity, not to independent data (§4.3).
- The authors list the untested conditions themselves: other RLHF datasets, larger language models, and the
  online RLHF setting in which reward models are periodically retrained on fresh feedback (§6).
- UWO acts as a negative shaping term: responses on which ensemble members disagree receive a lower reward,
  which the paper reports keeps intra-ensemble variance from growing during PPO (§5.5).

## Connections
- [[reward-model-overoptimization]] — Gao et al. (2023), the setup this paper reuses and extends.
- [[bradley-terry-rm]] — the pairwise cross-entropy loss each ensemble member is trained with (§4.3).
- [[warm-weight-averaged-reward-models]] — WARM cites this work (ref. [42]) and uses it as the prediction-
  ensembling (ENS) baseline it compares weight averaging against.
- [[west-of-n]] — hardens reward-model training through data rather than aggregation.
- [[generative-reward-models]], [[pairrm]] — alternative reward-model designs in this library.
- [[best-of-n]] — one of the two policy-optimization methods evaluated here.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2310.02743 (v2, 2024-03-10; ICLR 2024 camera-ready).
- Corrections to the previous card version:
  - Title "Reward Model Ensembles Help Mitigate Overoptimization (Coste et al.)" → the published title has
    no author suffix.
  - "lower confidence bound (mean − λ·std)" / "LCB" → the paper defines UWO as mean minus λ times the
    intra-ensemble **variance** (§3, Eq. 5); the term "LCB" does not appear. The minimum aggregator is
    called WCO (§3).
  - "K = 3, 5, 10" → the default ensemble has five members; the cardinality study covers 3, 4, 5 (§4.3, Fig. 11).
  - "3–5 RMs is usually enough" → the paper's statement is that 4- or 5-member ensembles work best, with a
    noticeable gap below 4 (§5.4).
  - "Fig. 3 (mean vs LCB vs min) — LCB is the best compromise" → Figure 3 is the BoN result across
    objectives; the paper does not rank UWO above WCO as a compromise.
  - "β same range as standard RLHF" → the reported KL penalty weights are 0.01 with WCO/UWO and 0.2 for KL
    alone (§5.3).
  - Added the missing **Source type** field and a Recipe ledger per the card standard.
- Removed as unsupported by the source: "K ≥ 3 RMs with different seeds/data shards" (members use identical
  data by design, §4.3); "seed diversity matters less than data-shard diversity" (the paper argues the
  opposite and gives every member all the data); "KL at peak shifts from d ≈ 3 to d ≈ 5–8" (no such numbers
  appear); "if all RMs are systematically miscalibrated in the same direction ... demonstrated on
  adversarial prompts" and "Fig. 5 (shared-blind-spot counterexample)" (Figure 5 is the no-KL-penalty PPO
  result; the paper runs no adversarial-prompt experiment); "disagreement correlates with OOD-ness of the
  response; can be used as an anomaly flag" (not measured); "K RMs roughly multiply the reward-forward-pass
  cost by K ... often affordable since the RM is smaller than the policy" (no cost analysis is reported);
  "Used in production pipelines for safety-sensitive RLHF" (no source); "the LCB aggregator implicitly
  penalizes policies pushing RMs apart" as a claim about [[reward-hacking-taxonomy]].
- Not reported: ensemble training cost; datasets other than AlpacaFarm; policies larger than 1.4B; online RLHF.
