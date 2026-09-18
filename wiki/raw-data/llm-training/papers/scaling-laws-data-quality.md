<!-- scope: quality-aware scaling law fitted on controlled noise-injection experiments in NMT and small causal LM pretraining
     see-also: [[data-constrained-scaling]], [[fineweb]]
-->

# Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining
- **Core Insight:** Adding a dimensionless quality parameter Q ∈ (0,1] to the Chinchilla form, as L(N, D, Q) = A/N^α + B/(D^β Q^γ) + E, fits held-out loss across 126 controlled runs, and the fitted γ is well below 1 — 0.173 for English→German translation and 0.401 for causal language modeling — so effective data decays sublinearly with quality (§5.2, Table 2).
- **Guideline:** When comparing two corpora whose corruption rate can be measured or estimated, predict the loss difference with ΔL(Q) ≈ B̂ D^(−β̂) (Q^(−γ̂) − 1) and expect moderate corruption to cost little, because γ̂ < 1 in both tested tasks. Do not transfer the fitted constants to frontier pretraining: quality here means injected synthetic token noise at a fixed model size, not filter-induced distribution change in web curation (§5.1, §5.2).
- **Authors:** Anirudh Subramanyam, Yuxin Chen, Robert L. Grossman (University of Chicago)
- **Year:** 2025 (arXiv v1 2025-10; v2 2026-02-23)
- **URL:** https://arxiv.org/abs/2510.03313
- **Source type:** paper
- **Relevant topics:** scaling laws, data quality, effective sample size, noise injection, corruption rate

## Abstract
The paper argues that existing scaling laws characterize loss as a function of model size and dataset volume while remaining agnostic to data quality. It introduces a dimensionless quality parameter Q and a quality-aware scaling law extending the Chinchilla framework, motivated by an effective-sample-size and information-theoretic view of noisy or redundant corpora. Two estimators for Q are proposed: a corruption-rate proxy and a deficiency measure. The law is tested in synthetic experiments in neural machine translation and autoregressive language modeling, where quality is controlled by multiple levels of noise injection. The reported results are that loss scales predictably with quality, that higher-quality data can reduce model size and compute requirements, that effective data decays sublinearly with quality, and that out-of-sample evaluation reproduces the fitted form.

## Key Contributions
- **Quality-aware law:** L(N, D, Q) = A/N^α + B/(D^β Q^γ) + E, where N is parameter count, D is token count, Q ∈ (0,1] is quality, and A, B, E, α, β, γ are fitted constants (§1, Equation 2).
- **Two estimators for Q:** the corruption-rate proxy Q = 1 − CR (Definition 1), and a deficiency-based measure Q = exp(−Δ(ω)) built from a deficiency functional Δ assumed positive, continuous, and additive over independent datasets (Definition 2, §3).
- **Derivation:** the effective sample size D_eff = D·g(Q) with g(Q) ≈ Q^γ is recovered from a signal-to-noise argument and an information-theoretic bound (§4, Proposition 1).
- **Unification table:** Table 1 relates particular forms of Δ(ω) to existing results, including a density-based form and a repeated-epoch form under which loss first falls and then grows as epochs increase (§4, §D).
- **Controlled experiments:** 63 NMT runs and 63 CLM runs across 3 dataset volumes × 7 quality levels × 3 replicates (§5).
- **Out-of-sample check:** the fitted CLM parameters are re-estimated on unseen data, with β essentially unchanged and γ shifting from 0.401 to 0.337 under Huber fitting (Table 3).

## Key Figures/Tables to Study
- Table 2 — fitted B, β, γ, E for NMT and CLM under least-squares and Huber estimation.
- Table 3 — the same parameters re-fitted on unseen CLM data.
- Figure 1 — test loss as Q varies from 0.5 to 1.0 for both tasks.
- Figure 2 — iso-loss contours showing the substitution between D and Q at fixed model capacity.
- Figure 3 — ΔL(Q) against Q^(−γ̂) − 1, which the paper uses to check linearity of the quality term.
- Tables 7 and 8 — the full per-run test losses in nats for NMT and CLM.
- Table 9 — sensitivity of γ̂ to learning rate.

## Technical Details
- **Quality definition:** Q = 1 − CR where CR is the fraction of corrupted tokens; a corpus with 10% corrupted tokens has Q = 0.9 (§3, Definition 1). The alternative deficiency measure maps Δ(ω) ≥ 0 to Q ∈ (0,1] (Definition 2).
- **Chinchilla baseline:** the paper restates that the Chinchilla constants were estimated over 400+ models with 70M–16B parameters and 5B–400B training tokens (§4).
- **NMT setup (§5.1):** English→German on Paracrawl v8, filtered by MinHash deduplication, language-ID filtering at a 0.8 threshold, and length filtering ≤ 258, leaving about 101M sentence pairs. Model is an 8-layer GPT-Neo with hidden size 1024, about 133M parameters, with a custom BPE tokenizer of vocabulary 32,000. Dataset sizes are 500K, 1M, and 2M sentence pairs.
- **CLM setup (§5.1):** C4 (en) subset, 8-layer Llama 3 architecture with hidden size 512 and context length 2048, no RoPE scaling, using the pretrained Llama-3.2-1B tokenizer. Dataset sizes are 100M, 1B, and 10B tokens, one epoch each.
- **Noise model (§5.2):** quality levels are set by the fraction η of perturbed samples, η ∈ {0, 10, 20, 25, 30, 40, 50}%, giving Q ∈ {1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5}. In NMT a perturbed sample has half of its non-special tokens replaced by pad tokens, without distinguishing source from target. In CLM a perturbed sample has 50% of its non-special tokens swapped with other valid tokens.
- **Sampling design (§5.2):** three working datasets are drawn from the base corpus with different seeds; each is subsampled so that S_0.1B ⊆ S_1.0B ⊆ S_10.0B; each size is then perturbed at each η level.
- **Fitted parameters (Table 2, Huber):** NMT B = 139.60, β = 0.2501, γ = 0.1732, E = 0.0665. CLM B = 1441.51, β = 0.3959, γ = 0.4007, E = 3.4390. Least-squares values are close (NMT γ = 0.1851; CLM γ = 0.3887).
- **Out-of-sample (Table 3, Huber):** CLM on unseen data gives B = 1427.30, β = 0.3905, γ = 0.3368, E = 4.5400, against the in-distribution fit of 1441.51 / 0.3959 / 0.4007 / 3.4390.
- **Isolated quality effect (§5.2, Equation 4):** ΔL(Q) = L(N,D,Q) − L(N,D,1) ≈ B̂ D^(−β̂) (Q^(−γ̂) − 1). Plots of ΔL(Q) against Q^(−γ̂) − 1 are close to linear and pass near the origin at all three dataset sizes in both tasks (Figure 3), which the paper reads as evidence that A/N^α + E does not depend on Q.
- **Interpretation of γ < 1 (§5.2):** the authors state that PAC-learning and channel-capacity arguments typically predict γ ≥ 1, and hypothesize that redundancy in natural language explains the measured robustness. They further hypothesize that CLM has the larger γ because token swaps add entropy and mislead local dependencies, while the NMT padding corruption leaves alignment and context partly intact.
- **Robustness checks (§E):** cosine similarity between clean and noised embeddings decreases monotonically with noise, used as evidence that synthetic noise proxies semantic degradation (Figure 4); a learning-rate sweep across 3 dataset scales and 3 noise levels leaves γ̂ near 0.39 (Table 9, Figure 5).
- **Evaluation:** test loss in nats on held-out data; the NMT test split is 50,000 samples (§C).
- **Compute (§5.2):** NMT runs up to 1M sentence pairs on one A100 node with 4×80GB GPUs; all other runs on 2 Hopper nodes with 4×141GB GPUs each; no multi-GPU or multi-node training per run, the GPUs being used to parallelize separate runs.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT-Neo 8L (this paper, NMT) | ~133M | pretrain-stable | architecture | 8 layers, hidden size 1024 | arXiv:2510.03313v2 §5.1, Table 4 | verified 2026-09-18 | no ablation reported |
| GPT-Neo 8L (this paper, NMT) | ~133M | pretrain-stable | tokenizer | custom BPE, vocabulary 32,000 | §5.1, §C | verified 2026-09-18 | no ablation reported |
| GPT-Neo 8L (this paper, NMT) | ~133M | pretrain-stable | data volumes | 500K / 1M / 2M sentence pairs from Paracrawl v8 | §5.1 | verified 2026-09-18 | Figure 3: ΔL(Q) linearity holds at all three volumes |
| GPT-Neo 8L (this paper, NMT) | ~133M | pretrain-stable | optimizer and schedule | AdamW, peak LR 5e-4, cosine decay, 20% warmup ratio | §5.1 | verified 2026-09-18 | no ablation reported |
| GPT-Neo 8L (this paper, NMT) | ~133M | pretrain-stable | packing | greedy dataset packing with DataCollatorWithFlattening | §C, Table 6 | verified 2026-09-18 | no ablation reported |
| GPT-Neo 8L (this paper, NMT) | ~133M | pretrain-stable | runs | 3 sizes × 7 quality levels × 3 replicates = 63 | §5.1 | verified 2026-09-18 | — |
| Llama 3 8L (this paper, CLM) | 8 layers, hidden 512 | pretrain-stable | context length | 2048, no RoPE scaling | §5.1, Table 5 | verified 2026-09-18 | no ablation reported |
| Llama 3 8L (this paper, CLM) | 8 layers, hidden 512 | pretrain-stable | tokenizer | pretrained Llama-3.2-1B tokenizer | §5.1 | verified 2026-09-18 | no ablation reported |
| Llama 3 8L (this paper, CLM) | 8 layers, hidden 512 | pretrain-stable | data volumes | 100M / 1B / 10B tokens from C4 (en), 1 epoch | §5.1 | verified 2026-09-18 | Figure 3: ΔL(Q) linearity holds at all three volumes |
| Llama 3 8L (this paper, CLM) | 8 layers, hidden 512 | pretrain-stable | optimizer and schedule | fused AdamW, peak LR 1e-3, cosine decay, 10% warmup ratio | §5.1 | verified 2026-09-18 | Table 9: γ̂ stable near 0.39 across the swept learning rates around 1e-3 |
| Llama 3 8L (this paper, CLM) | 8 layers, hidden 512 | pretrain-stable | parameter count | not reported | §5.1, Table 5 checked | not reported | — |
| Both tasks | — | pretrain-stable | quality levels | η ∈ {0,10,20,25,30,40,50}% perturbed samples, i.e. Q ∈ {1.0,0.9,0.8,0.75,0.7,0.6,0.5} | §5.2 | verified 2026-09-18 | Figure 1: monotone test-loss response over this range |
| Both tasks | — | fitting | estimator | Hoffmann et al. parametric loss fit, least squares and Huber | §5, §C | verified 2026-09-18 | Table 2: the two estimators agree to within 0.012 in γ for CLM and 0.012 for NMT |

## Findings relevant to generality
- **Model size is not varied.** Each task uses one fixed architecture, so only B, β, γ, and E are fitted; the A/N^α term of the stated law is not estimated from these experiments (Table 2, Table 3). Claims about "higher-quality data can reduce model size" rest on the iso-loss contours computed from the fitted law at fixed model capacity (Figure 2), not on a model-size sweep.
- **Quality here means injected corruption.** Q is controlled by replacing tokens with pad tokens (NMT) or with other valid tokens (CLM) in a chosen fraction of samples (§5.2). The paper does not measure filter-induced narrowing, domain imbalance, or synthetic-data effects, although it names redundancy and domain imbalance as motivations in §1 and §2.
- **No downstream or capability evaluation.** The only reported metric is held-out cross-entropy loss in nats (§5, Tables 7 and 8). Breadth of capability, task transfer, and benchmark scores are not measured.
- **Scale is far below frontier pretraining.** The largest run is 10B tokens on an 8-layer model with hidden size 512 (§5.1), against the 70M–16B parameter and 5B–400B token range of the Chinchilla fit the paper extends (§4).
- **Out-of-sample γ shifts.** Re-fitting on unseen CLM data moves γ from 0.401 to 0.337 while β moves only from 0.396 to 0.391 (Table 3), so the quality exponent is the less stable of the two exponents under distribution change.
- **Robustness to moderate corruption is the headline result.** Because γ̂ < 1 in both tasks, loss rises more slowly than linearly as Q falls, and the paper states that a large drop in Q is needed before loss rises noticeably (§5.2).

## Connections
- [[fineweb]], [[dolma]], [[ccnet]] — empirical web-filtering pipelines; this paper's Q is a corruption rate, so it does not directly measure what those filters change.
- [[data-constrained-scaling]] — repeated-epoch scaling; Table 1 and §D relate a deficiency form to loss growth as epoch count k increases.
- [[physics-of-lm-3]] — controlled synthetic-data study of what a model extracts per token.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2510.03313 (arXiv v2, 2026-02-23)
- Corrections to the previous card version:
  - "Year: 2025" without version → arXiv v1 2025-10, v2 2026-02-23.
  - "Models quality through effective sample size / deficiency-style terms" → the functional form is explicit: L(N, D, Q) = A/N^α + B/(D^β Q^γ), with D_eff = D·g(Q) and g(Q) ≈ Q^γ (§1, §4).
  - "Evaluates how corruption or redundancy changes the useful training signal" → the experiments vary injected corruption only; redundancy appears in the theory (Table 1, §D) but is not measured experimentally.
  - "extends standard language-model scaling-law thinking by adding a formal data-quality term … with quality affecting the effective value of the data budget" → accurate in form, but the card omitted that the fit comes from 126 controlled runs at one model size per task on Paracrawl v8 and C4, not from language-model pretraining at scale (§5).
  - "two corpora with the same token count can sit on different scaling curves if quality differs enough" → the measured statement is stronger and more specific: the gap is ΔL(Q) ≈ B̂ D^(−β̂)(Q^(−γ̂) − 1) with γ̂ = 0.173 (NMT) and 0.401 (CLM), so equal-token corpora differ sublinearly in quality (§5.2).
- Removed as unsupported by the source:
  - "Provides practical proxies for data quality rather than only dataset size" as a headline contribution without qualification — the two estimators are a corruption rate and a deficiency measure, and only the corruption rate is used in the experiments (§3, §5.2). A procedure for estimating Q on real corpora is sketched in §E.3 but not validated against a real filtering pipeline.
  - "Gives a framework for trading curation effort against raw token count" as an empirical result — the trade-off is read off iso-loss contours derived from the fitted law at fixed model capacity (Figure 2), not from curation experiments.
  - "Formal counterpart to empirical filtering results in [[fineweb]], [[dolma]], [[ccnet]]" as an equivalence — those pipelines change the data distribution by selection; this paper's Q measures injected token corruption. The connection is retained with that qualification.
- Not reported by the source: the CLM model's parameter count; any downstream benchmark; any run above 10B tokens; the fitted A and α; validation of Q on a naturally occurring (unmodified) corpus.
