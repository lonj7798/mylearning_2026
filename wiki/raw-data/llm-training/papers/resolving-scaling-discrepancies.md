<!-- scope: Porian et al. 2024 (NeurIPS 2024): last-layer FLOP counting, warmup length, and scale-dependent optimizer tuning explain the gap between the Kaplan et al. and Hoffmann et al. (Chinchilla) compute-optimal model-size exponents; OpenLM decoder-only models 5M–901M on RefinedWeb and OpenWebText2; scaling laws for optimal learning rate and batch size
     deps: [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]]
     see-also: [[chinchilla-replication]], [[deepseek-llm]], [[large-batch-training-noise-scale]], [[cooldown-scaling-beyond-fixed-durations]], [[resolving-scaling-discrepancies-recipe]]
-->

# Resolving Discrepancies in Compute-Optimal Scaling of Language Models
- **Core Insight:** On RefinedWeb, reproducing the Kaplan et al. setup gives a compute-optimal model-size exponent a = 0.835; counting the head's FLOPs lowers it to 0.706, scaling warmup to N tokens to 0.602, and tuning learning rate, batch size, and AdamW β2 per model size with a constant learning rate to 0.497, close to the Hoffmann et al. value of 0.5; cosine decay without tuning reaches 0.571 (Table 1, Fig. 1).
- **Guideline:** When fitting a compute-optimal scaling law from small models, count the output-head FLOPs, keep warmup short relative to the token budget, and tune batch size, learning rate, and β2 per model size, because each correction moved the exponent toward 0.5 on both datasets (Table 1). Cosine decay is not needed for the allocation exponent, but it lowers the optimal loss more as compute grows (§3.5, §4.1).
- **Authors:** Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, Yair Carmon (Tel Aviv University; University of Washington; Jülich Supercomputing Centre and LAION)
- **Year:** 2024 (arXiv v1 2024-06; v4 2025-01; NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2406.19146
- **Source type:** paper
- **Relevant topics:** compute-optimal scaling, FLOP accounting, learning-rate warmup and decay, batch size and learning-rate scaling, AdamW β2, scaling-law methodology

## Abstract
Kaplan et al. and Hoffmann et al. proposed scaling laws for the optimal model size as a function of compute that give different predictions. The authors reproduce the Kaplan et al. law on OpenWebText2 and RefinedWeb and identify three causes of the difference: the computational cost of the last layer, the warmup duration, and optimizer tuning that depends on scale. With these corrected, the results agree with the Hoffmann et al. ("Chinchilla") law. Against a hypothesis implied by Hoffmann et al., careful learning-rate decay is not essential for that law. As a secondary result, the paper derives scaling laws for the optimal learning rate and batch size and finds that tuning AdamW β2 is essential at lower batch sizes.

## Key Contributions
- A step-by-step reproduction that moves from the Kaplan et al. exponent to the Hoffmann et al. exponent, one correction per panel, using over 900 training runs (Fig. 1, Table 1).
- Evidence that learning-rate decay matched to each budget changes the exponent by about 0.03, against Hoffmann et al.'s conjecture that decay is the main cause (§3.4).
- Power laws for the optimal batch size and learning rate as functions of N, and the finding that β2 = 0.95 is suboptimal at batch sizes of 128 and below (§1, §3.5, Fig. 3).
- An analysis of the compute-optimal loss curve and of scaling-law accuracy as a function of experiment cost (§4).
- Released code, data, and checkpoints (§1).

## Key Figures/Tables to Study
- **Figure 1 / Table 1** — exponent a, confidence interval, R², and ρ⋆ range after each correction, on both datasets.
- **Figure 2** — optimal token count D⋆ against compute, with and without the long warmup.
- **Figure 3 / Table 4** — optimal batch size and learning rate against model size, and the values used per size.
- **Figure 4 / Figure 19** — compute-optimal loss for each experiment and saturating power-law fits.
- **Table 2** — the 16 model shapes with N, N_exact, N_eff, and N_Kaplan.
- **Tables 5–7** — warmup, final-LR, and 901M learning-rate/batch-size ablations.

## Technical Details
- **Model size (§2.1, App. B):** N counts the parameters of all linear layers, including the output head and excluding embeddings; N = (3·d_FF + 4d)·d·l + d·v, where d is the width, l the depth, v = 50,432 the vocabulary size, and d_FF the SwiGLU feed-forward dimension (App. B, Eq. 5). FLOPs(N, D) ≈ 6ND, where D is the number of training tokens (Eq. 1).
- **Target quantities (§2.1):** N⋆(C) is the model size with the lowest loss at compute C; D⋆(C) = C/(6N⋆) and ρ⋆ = D⋆/N⋆. Fitted forms: N⋆ ≈ N⋆_0·C^a, D⋆ ≈ D⋆_0·C^b, ρ⋆ ≈ ρ⋆_0·C^r (Eq. 4).
- **Grid and data (§2.2):** 16 models from 5M to 901M; OpenWebText2 (~30B tokens, Reddit) and RefinedWeb (~600B tokens, CommonCrawl); evaluation on 160M held-out tokens whenever 6ND crosses 1.25e16·2^i for i = 0…11.
- **Fitting (§2.3, App. D):** IsoFLOP curves per budget; N⋆(C_i) and its log-scale standard deviation from a noise-and-interpolate procedure with Akima interpolation; weighted linear regression in log space; confidence intervals from bootstrap samples.
- **Step 1, reproduction (§3.1):** batch 2^19 tokens, warmup ≈ 1.57B tokens, cosine decay to zero at ≈ 131B tokens, and N_Kaplan = N − d·v. Fits are close to Kaplan et al.'s 1.6e9·(C/8.64e19)^0.88.
- **Step 2, head FLOPs (§3.2):** omitting the head under-counts FLOPs by roughly 10% for larger models and roughly 90% for smaller models (Table 2). Counting it lowers a by more than 0.1.
- **Step 3, warmup (§3.3):** with the fixed warmup, smaller models reach compute-optimality during warmup (Fig. 2 left). Setting warmup tokens = N makes the optimal token count at least 5 times the warmup length and gives a ≈ 0.6.
- **Step 4, decay (§3.4):** with the 131B-token decay, compute-optimal runs (never above 10B tokens) see under 1.5% of the decay. Per-budget cosine decay to 1% of peak raises R² from 0.993 to 0.998 and moves a to 0.57, at about twice the cost.
- **Step 5, tuning (§3.5):** per-size learning rate, batch size, and β2 with a constant schedule give a matching 0.5 within 0.6% and a predicted model size at Chinchilla compute (5.88e23 FLOPs, Fig. 1) within 15% of Chinchilla's size. Fitted laws: BS = 0.00037·N^0.703 and LR = 3.7·N^−0.36 (Fig. 3). Smaller batches make squared-gradient estimates noisier, so β2 = 0.99 and 0.999 were added to the sweep (§3.5).
- **Table 1 exponents (OpenWebText2 / RefinedWeb):** reproduction 0.864 / 0.835; head FLOPs 0.699 / 0.706; warmup 0.603 / 0.602; cosine decay 0.574 / 0.571; optimizer tuning 0.518 / 0.497. Reference values: Hoffmann et al. 0.5, Kaplan et al. 0.88, adjusted Kaplan et al. 0.73.
- **Adjusted Kaplan law (§3.5, App. H):** tuned hyperparameters with the head-FLOP and warmup issues restored give a = 0.717, close to Kaplan et al.'s adjusted 1.3e9·(C/8.64e19)^0.73.
- **Comparison with DeepSeek (§3.5):** batch-size exponents differ by less than 0.05 and predictions by less than 60%; learning-rate exponents differ by 0.11, predictions by a factor of 2–3. Both find an optimal batch size below which performance degrades, which the authors say appears to contradict the view that every batch size below a critical batch size is good.
- **Attention FLOPs (App. B):** N_eff = N + n·d·l with n = 2048; N_eff/N ranges from about 1.1 to 1.2 (Table 2), and fits with N_eff are similar to Fig. 1 (Fig. 7).
- **Recipe:** full training settings are in [[resolving-scaling-discrepancies-recipe]].

## Recipe ledger
The ledger (architecture, Table 3 and Table 4 hyperparameters, schedules, sweep, compute, and data repetition) is in [[resolving-scaling-discrepancies-recipe]].

## Findings relevant to generality
- **Loss versus schedule (§4.1, Fig. 4):** shorter warmup and hyperparameter tuning each improve compute-optimal loss by up to 0.5 nat per token at low compute, but not significantly at larger scales; cosine decay gives increasing benefit as compute grows.
- **Predictability as a tuning check (§4.1):** a saturating power law fits and extrapolates well only for the tuned experiment (Fig. 19). A 901M model at C ≈ 8e19 FLOPs reaches L = 2.943, within the predicted trend (Fig. 6).
- **Cost (§4.2):** a fixed-schedule experiment costs 1.54e20 FLOPs and a per-budget cosine experiment 2.99e20; the sweep cost 2.04e20, and a reduced sweep would have cost 1.44e19.
- **Limits (§5.2):** compute is roughly at the Kaplan et al. scale and well below Hoffmann et al.; the hyperparameter sweep used small models trained for 20N tokens, which may favor Hoffmann et al. scaling (App. G.4 estimates the exponent changes by less than 0.032 under ideal tuning); the laws concern pre-training loss only, and most zero-shot and in-context capabilities do not emerge at these scales.

## Connections
- [[kaplan-scaling-laws]] — the law being reproduced; source of the 2^19-token batch, 3000-step warmup, and N_Kaplan definition (§3.1).
- [[chinchilla-compute-optimal]] — the target law (a = 0.5); the IsoFLOP fitting approach follows its second approach (§2.3).
- [[chinchilla-replication]] — Besiroglu et al., whose extracted data Pearce and Song re-analyzed in concurrent work (§5.1).
- [[deepseek-llm]] — also tunes hyperparameters with IsoFLOP analysis; reports a = 0.578 on OpenWebText2, which the authors attribute possibly to data repetition (§5.1).
- [[minicpm]] — Hu et al., who find larger token-to-parameter ratios than this paper (§5.1).
- [[beyond-chinchilla-inference-scaling]] — Sardana and Frankle, cited as work that uses the Hoffmann et al. law as a reference point (§5.1).
- [[overtraining-downstream-scaling]] — Gadre et al., source of the initial training configuration (§2.2).
- [[data-constrained-scaling]] — Muennighoff et al., cited for the small effect of 4 data repetitions (App. C).
- [[large-batch-training-noise-scale]] — McCandlish et al., the large-batch model that Kaplan et al. used to adjust their law (§3.5).
- [[cooldown-scaling-beyond-fixed-durations]] — Hägele et al., concurrent work on schedules that do not fix the step budget in advance (§5.1).
- [[small-scale-proxies-instabilities]] — Wortsman et al., source of the independent weight decay and the comparison of learning-rate sensitivity (App. C, App. G.1).
- [[predicting-downstream-elusive]] — addresses the downstream-capability prediction problem that §5.2 leaves open; not cited by this paper.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2406.19146 (arXiv v4, 2025-01-19; full PDF text including App. A–J).
- Audit claims not found in the source: "NeurIPS 2024 spotlight" — the PDF states NeurIPS 2024 but not spotlight status.
