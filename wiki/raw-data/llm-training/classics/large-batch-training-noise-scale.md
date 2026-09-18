<!-- scope: the gradient noise scale as a predictor of the critical batch size, and the steps-vs-examples trade-off of data-parallel training (supervised, generative, RL)
     deps: [[adam]], [[lr-schedules]]
     see-also: [[kaplan-scaling-laws]], [[critical-batch-size-pretraining]], [[deepseek-llm]], [[ppo]]
-->

# An Empirical Model of Large-Batch Training
- **Core Insight:** The simple gradient noise scale B_simple = tr(Σ)/|G|² predicts the critical batch size B_crit at the order-of-magnitude level on 8 tasks where B_crit ranges from 20 (SVHN autoencoder) to over 10 million, and B_noise, B_simple, and B_crit all increase by at least an order of magnitude during training (§3, Fig. 4, App. D).
- **Guideline:** When choosing a data-parallel batch size, measure B_simple during a run with a well-tuned learning rate and use it as an order-of-magnitude estimate of B_crit, because the two agreed at that level across supervised, generative, and RL tasks (§5); re-measure as the loss falls, because B_crit depends on the target loss (§2.3). Otherwise run the batch-size and learning-rate grid of App. A.2-A.3.
- **Authors:** Sam McCandlish, Jared Kaplan, Dario Amodei, and the OpenAI Dota Team
- **Year:** 2018 (arXiv v1 2018-12)
- **URL:** https://arxiv.org/abs/1812.06162
- **Source type:** paper
- **Relevant topics:** batch size, data parallelism, gradient noise, compute-time trade-off, learning-rate scaling, adaptive batch size

## Abstract
The largest batch size that trains without losing data efficiency differs by domain, from tens of thousands of images for ImageNet to millions of timesteps for Dota 2 agents. The authors show that the gradient noise scale, a statistic that is easy to measure, predicts the largest useful batch size on supervised datasets (MNIST, SVHN, CIFAR-10, ImageNet, Billion Word), RL (Atari, Dota), and generative models (autoencoders on SVHN). The noise scale increases as loss decreases during a run and depends on model size mainly through the lower loss that larger models reach. The theory also describes the trade-off between compute efficiency and time efficiency and gives a rough model of the gain from adaptive batch sizes.

## Key Contributions
- Derives the optimal step size and per-step loss improvement for a noisy gradient under a quadratic approximation, which defines the noise scale (§2.2, Eq. 2.4-2.8).
- Gives the hyperbolic relation between optimizer steps and training examples, and defines B_crit from it (§2.3, Eq. 2.11-2.12).
- Measures B_crit and B_simple on 8 tasks and reports them in one table (§3, Table 1, Fig. 4).
- Gives a zero-overhead estimator of B_simple from local and global gradient norms in data-parallel training (App. A.1).
- Relates the noise scale to a training "temperature" ε/ε_max(B) and derives an adaptive batch-size rule (App. C, App. D).

## Key Figures/Tables to Study
- Fig. 1 and Fig. 7: Pareto fronts of optimizer steps vs examples processed, with Eq. 2.11 fits.
- Fig. 4 and Table 1: B_crit and B_simple, at the start and averaged over training, for every task.
- Fig. 8: LSTM noise scale vs training perplexity for several LSTM sizes.
- Fig. 15: a 16× reduction in temperature raises the noise scale 16×.

## Technical Details
- Batch gradient G_est = (1/B) Σ_i ∇L_{x_i}(θ) has covariance Σ/B (Eq. 2.1-2.2); B = batch size, Σ = per-example gradient covariance (Eq. 2.3), G = true gradient, H = true Hessian.
- Optimal step ε_opt(B) = ε_max/(1 + B_noise/B) and loss improvement ΔL_opt(B) = ΔL_max/(1 + B_noise/B), with ε_max = |G|²/(GᵀHG), ΔL_max = |G|⁴/(2GᵀHG), B_noise = tr(HΣ)/(GᵀHG) (Eq. 2.6-2.8). A step larger than 2ε_opt can increase the loss (§2.2).
- B_simple = tr(Σ)/|G|² assumes H is a multiple of the identity; B_simple and B_noise "typically differ only by a small constant multiplicative factor" (Eq. 2.9, §2.2).
- Trade-off: S/S_min − 1 = (E/E_min − 1)^(−1), B_crit = E_min/S_min; S = optimizer steps, E = examples processed, both to reach one performance target. At B = B_crit a run uses 2× the minimum steps and 2× the minimum examples (Eq. 2.11-2.12, §2.3). Fig. 4 defines B_crit as the point where compute efficiency drops below 50% of optimal.
- Worked example (derived from App. D.1, δS = 1 + B_noise/B and δE = B + B_noise, relative to the minima): with B_noise = 1,000, B = 250 needs 5× the minimum steps and 1.25× the minimum examples; B = 1,000 needs 2× and 2×; B = 4,000 needs 1.25× and 5×. Each pair satisfies Eq. 2.11 (4 = 1/0.25).
- Estimator: E|G_B|² = |G|² + tr(Σ)/B (Eq. A.1); unbiased |G|² and tr(Σ) estimates come from gradient norms at B_small (the local batch, before averaging between devices) and B_big (the global batch, after averaging), smoothed with separate exponential moving averages; the ratio of the averages is used (App. A.1-A.2).
- Table 1, critical batch size start / average; simple noise scale start / average: MNIST 20 / 200; 50 / 900. SVHN 50 / 500; 300 / 4,000. CIFAR10 300 / 900; 400 / 2,000. ImageNet 1,000 / 15,000; 4,000 / 30,000. Autoencoder (SVHN) 10 / 40; 2 / 2. VAE (SVHN) 10 / 200; 10 / 10. Billion Word (per token) 700 / 100,000; 1000 / 150,000. Atari (per frame) 100-1,000 / 400-8,000; 100-1,000 / 1,000-20,000. Dota 1v1 50,000 / 3,000,000; 100,000 / 300,000. Dota 5v5 not measured / >8,000,000 (est.); 100,000 / 24,000,000.
- ImageNet (ResNet-50): noise scale 2,000 to 100,000 in the main phase, rising to hundreds of thousands and millions late in training; measured B_crit 15k vs 64k reported in the literature (§3.2).
- Ratio B_simple/B_crit varies by about an order of magnitude across tasks: below 1 for autoencoder, VAE, Dota 1v1; about 1 for LSTMs; above 1 for image classification and Atari (§5).
- Learning rate: with SGD or momentum the optimal LR follows Eq. 2.6; with Adam or RMSProp it follows ε ∝ B^α with α between 0.5 and 1.0, then levels off (§3.1, App. E.2).
- Temperature T = ε/ε_max(B), and B_noise ∝ B_simple ∝ 1/T (Eq. C.1-C.2); decaying the LR by a factor often raises the noise scale by about the same factor (App. C).
- Adaptive batch B = √(r·B_simple) (Eq. D.6); for SVHN, B_crit(s) ≈ 10√s predicts γ = 24/25, "around 4%" gain (Eq. D.7); observed gains in Fig. 16 were in some cases larger than this prediction (App. D.2).
- Deterministic line-search training ("greedient descent") performs poorly; with a fixed LR the actual update settles at about twice the line-search optimal step (App. E.1, Fig. 17).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LSTM LM, One Billion Word | size-2048 LSTM | pretrain-stable | tokenizer; embedding; optimizer; dropout; grad clip; sequence length | BPE vocab 40,000; 512-dim; Adam with momentum 0.5; no dropout; clip norm 10; 20-token sequences | arXiv:1812.06162v1 App. A.4.3 | verified 2026-09-14 | no ablation reported |
| LSTM LM size comparison | 1024 and 512 LSTMs (256-dim embedding for 512) | pretrain-stable | batch size; Adam LR | 1024; 0.0007 | App. A.4.3 | verified 2026-09-14 | grid search; optimal LR "did not have a significant dependence on model size" (A.4.3) |
| LSTM LM | all | pretrain-stable | batch-size unit | tokens | App. A.4.3 | verified 2026-09-14 | B_simple and B_crit "depend predominantly on the total number of tokens" when sequence count and length vary (A.4.3) |
| A2C on 7 Atari games | conv policy | RL | optimizer; rollout; batch control; start randomization | RMSProp α = 0.99, ε = 1e-5; 5 steps per rollout; batch set by number of parallel environments; random initial steps up to 500 | App. A.4.2 | verified 2026-09-14 | no ablation reported |
| OpenAI Five Dota agents | not reported | RL | algorithm; batch | asynchronous PPO; for 5v5, the Dota team's batch size is shown as a lower bound on B_crit | App. A.4.2, Fig. 4 caption | verified 2026-09-14 | Pareto fronts not measured for 5v5 (§3.2) |
| SVHN CNN, fixed and adaptive batch | 2 conv + 1024-unit FC | n/a (image classifier) | LR as a function of B | ε = 0.27B/(96 + B) | App. D.2 footnote 20 | verified 2026-09-14 | "roughly optimal relation" for fixed-batch training; reused for adaptive runs |
| Temperature experiments | SVHN CNN; Billion Word LSTM | n/a (measurement run) | initial (ε, B) | (0.18, 128); (6 × 10⁻⁴, 128) | App. C | verified 2026-09-14 | Fig. 15 |
| LR grid centre, all tasks | all | n/a (LR grid search) | ε_central(B) = ε*/(1 + B*/B)^α | α = 1 for SGD or momentum; 0.5 < α < 1 for Adam or RMSProp | App. A.2 Eq. A.3 | verified 2026-09-14 | used as grid-search centre, "not ... precisely optimized" (A.2) |

## Findings relevant to generality
- Model size: B_simple is roughly independent of LSTM size at fixed loss; larger LSTMs reach larger noise scales only through lower loss (§3.2, Fig. 8).
- Task complexity: more complex image datasets have larger noise scales, not determined by dataset size; Dota 5v5 has a higher noise scale (at least 10 million) than Dota 1v1, which the authors attribute to more diverse situations (§3.2).
- Generalization is outside the theory, which concerns training loss only (§2.4, item 6); main results use training-set targets, and with test-set targets B_crit shows a small dip at the very end of training on MNIST, SVHN, CIFAR10 (§3.2, App. E.3).
- RL: the Dota noise scale was about a thousand times Atari's, while the steps needed to train a Dota agent were "not so much larger" (§5).

## Connections
- [[kaplan-scaling-laws]] — applies this model to Transformer LMs; Kaplan et al. report B_crit as a power of the loss, roughly 1-2 million tokens at convergence for their largest models (Kaplan §1, §5.1).
- [[critical-batch-size-pretraining]] — later study of critical batch size in LM pre-training.
- [[deepseek-llm]] — LLM report that fits batch-size and learning-rate rules against compute.
- [[emergence-loss-perspective]] — another result in which model size acts through the loss reached.
- [[adam]], [[lr-schedules]], [[gradient-clipping]] — optimizer and schedule settings that interact with the temperature ε/ε_max(B).
- [[ppo]] — algorithm used by the Dota agents measured here.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/1812.06162 (arXiv v1, 14 Dec 2018; body and App. A-E read).
- Audit claims not found in the source: none. The audit's Kaplan 2020 statement (optimal batch roughly 1-2M tokens at convergence, citing [MKAT18]) was checked in arXiv:2001.08361 §1 and is attributed to that paper, not this one.
- Internal inconsistency in the source: §3 gives the smallest B_crit as 20 (SVHN autoencoder), while Table 1 lists 10 (start) and 40 (average); the Table 1 caption says "at the end of the run" but the column is headed "Average".
