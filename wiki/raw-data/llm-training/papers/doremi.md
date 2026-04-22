<!-- scope: proxy-model Group-DRO to set pretraining domain mixture weights
     deps: [[the-pile]]
     see-also: [[fineweb]], [[scaling-laws-data-quality]]
-->

# DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
- **Core Insight:** Training a small proxy model with Group Distributionally Robust Optimization over pretraining domains produces a mixture that, transferred to the full-size model, reaches baseline accuracy in 2.6× fewer steps — without any downstream-task knowledge.
- **Guideline:** Don't hand-tune your pretraining mixture. Run a cheap proxy-model DRO pass to derive domain weights, then train the real model on the resampled mix.
- **Authors:** Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu
- **Year:** 2023 (NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.10429
- **Relevant topics:** data mixture, domain reweighting, Group DRO, pretraining efficiency

## Abstract
The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly affect language model (LM) performance. In this paper, we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy model using group distributionally robust optimization (Group DRO) over domains to produce domain weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with these domain weights and train a larger, full-sized model. In our experiments, we use DoReMi on a 280M-parameter proxy model to set the domain weights for training an 8B-parameter model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain. DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps. On the GLaM dataset, DoReMi, which has no knowledge of downstream tasks, even matches the performance of using domain weights tuned on downstream tasks.

## Key Contributions
- First principled, no-task-knowledge method for setting pretraining data-mix weights.
- **Group DRO** over domains → minimax between a proxy LM and a domain-weight adversary.
- Empirical: **+6.5 average downstream points** and **2.6× step speedup** on The Pile at 8B scale.
- Matches *task-tuned* GLaM weights without seeing any downstream task.
- Domain reweighting can **improve perplexity in every domain**, including downweighted ones (Pareto improvement).

## Key Figures/Tables to Study
- **Figure 1** — pipeline: reference model → proxy DRO → reweighted full training.
- **Domain-weight comparison table** — The Pile default vs DoReMi weights; shows surprising reweightings (e.g., upweighting some technical domains, downweighting noisy web).
- **Training-curve figure** — full 8B loss with DoReMi vs default weights; 2.6× speedup visible.

## Technical Details
**Setup:**
- Pretraining corpus split into `K` disjoint domains (for The Pile: 22 subsets).
- Reference model: a small LM pretrained with uniform domain weights, used to compute per-domain loss baselines.
- Proxy model: a 280M-param LM trained with Group DRO against the reference.

**Group DRO objective (intuition):** minimize the worst-case "excess loss" over domains, where excess loss on domain `k` is the proxy's loss minus the reference's loss. Formally:
```
min_θ  max_{α ∈ Δ}  Σ_k α_k · ( L_k(θ) − L_k(θ_ref) )
```
where `Δ` is the probability simplex and `α` are domain weights updated online.

**Algorithm (at each step):**
1. Sample a batch with current weights α.
2. Compute per-domain excess loss.
3. Exponentiated-gradient update on α toward domains with higher excess loss.
4. SGD step on θ (proxy model) on the same batch.

**Transfer step:** take the final α from proxy training (optionally averaged over last k steps) and resample the pretraining corpus with those weights. Train the large model on the resampled mix with a standard schedule.

**Key empirical findings:**
- 280M proxy transfers to 8B (30× scale) — scale-robust weights.
- On The Pile, all 22 per-domain perplexities improved vs baseline — DRO is not just reshuffling loss.
- Works even better when the reference is itself trained with uniform weights (as opposed to prior iteration of DoReMi).

## Connections
- Complements filtering-based approaches ([[dolma]], [[fineweb]]) — DoReMi re-weights whatever you kept.
- Rivals and is combined with classifier-based quality filtering; see [[fineweb]] for the classifier alternative.
- Theoretical ancestor: Group DRO (Sagawa 2019); LM application is the novel contribution.
- Used / adapted in `[[llama-3]]`-era mixture tuning.
