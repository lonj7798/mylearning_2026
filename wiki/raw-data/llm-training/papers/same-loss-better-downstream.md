<!-- scope: Liu, Xie, Li, Ma 2022: models with the same (near-optimal) masked-language-modeling pre-training loss transfer differently; continued training after convergence, larger size, and the training algorithm change downstream accuracy; trace of the Hessian (flatness) tracks downstream accuracy; SGD theory and a Dyck-language proof; simplified PCFG, HMM, and OPT-generated datasets
     deps: [[kaplan-scaling-laws]]
     see-also: [[predicting-downstream-elusive]], [[emergence-loss-perspective]], [[large-batch-training-noise-scale]]
-->

# Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models
- **Core Insight:** On a PCFG-generated dataset where all transformers larger than 9M reach almost the same masked-language-modeling loss, increasing model size lowers the trace of the Hessian from 19.8 to 12.6 while linear-probe accuracy on task B rises from 40.4% to 50.5% (§5, Fig. 5); pre-training loss alone does not determine downstream performance in these simplified settings.
- **Guideline:** When two checkpoints or model sizes have the same pre-training loss near its minimum, do not treat them as equally transferable; compare them on downstream probes (and, where affordable, a flatness measure such as the trace of the Hessian), because in the paper's settings, scaling up at the same loss improved linear-probe accuracy by 6.9% (PCFG), 4.5% (HMM), and 2.0% (OPT-generated data) (§3). The evidence comes only from simplified datasets in the saturation regime (§3, §7).
- **Authors:** Hong Liu, Sang Michael Xie, Zhiyuan Li, Tengyu Ma (Stanford University)
- **Year:** 2022 (arXiv v1 2022-10; the only arXiv version)
- **URL:** https://arxiv.org/abs/2210.14199
- **Source type:** paper
- **Relevant topics:** pre-training loss versus downstream performance, implicit bias of optimizers, flat minima, trace of the Hessian, model size, transfer

## Abstract
Validation pre-training loss is often used to compare language models because it tends to correlate with downstream performance. The paper shows that (1) pre-training loss cannot fully explain downstream performance and (2) flatness of the model correlates with downstream performance where pre-training loss does not. On simplified datasets, three interventions produce models with the same, statistically optimal pre-training loss but different downstream performance: continuing pre-training after convergence, increasing model size, and changing the training algorithm. The authors interpret this as an implicit bias of pre-training optimizers toward more transferable models among those with the same minimal loss. They prove that SGD with mini-batch noise prefers flatter minima in language modeling, observe a strong correlation between flatness and downstream performance, and prove in a synthetic language that the flattest model with minimal pre-training loss transfers to the downstream task.

## Key Contributions
- The saturation regime: a model's output equals the true conditional probability, so its loss equals the entropy of the data, the optimal loss (§3).
- Three controlled comparisons of equal-loss models: training steps after convergence, model size, and standard versus adversarial pre-training (§3, Fig. 1–2, Table 1).
- A theorem that SGD near the manifold of global minimizers moves along it to decrease the trace of the Hessian (§4, Theorem 4.3).
- Empirical correlation between trace of the Hessian and downstream accuracy for all three comparisons (§5, Fig. 3, Fig. 5, Table 2).
- A proof in a Dyck-language setting that the flattest zero-loss single-layer transformer learns a feature that solves the downstream task (§6, Theorem 6.1).

## Key Figures/Tables to Study
- **Figure 1** — downstream accuracy keeps rising after pre-training loss converges (41M PCFG model; 235M OPT-data model).
- **Figure 2** — larger models transfer better at almost the same loss (PCFG, HMM, OPT data).
- **Table 1** — AdamW, adversarial, and lookup-table models on PCFG tasks A and B.
- **Figure 3 / Table 2** — trace of the Hessian and accuracy across checkpoints and algorithms.
- **Figure 5** — trace of the Hessian and accuracy across model sizes.

## Technical Details
- **Task (§3):** masked language modeling with one masked token per sentence; loss L(θ) = E_{x,t}[−log f_θ(x_−t)_{x_t}], where x is a sentence, t a uniformly sampled position, x_−t the sentence with position t masked, and f_θ the predicted conditional probability vector.
- **Datasets (§3, App. A.1):** generated from known models, so the true conditional probability and the minimal loss can be computed.
  - PCFG: vocabulary 200, 50 non-terminal symbols, sentences up to 32 tokens, 2×10^7 sentences (3.4×10^8 tokens); tasks A, B, C classify the non-terminal for spans of length 32, 16, 8 (50-way, 0.1M examples each).
  - HMM: vocabulary 200, 100 hidden states, sentences up to 16 tokens, 1×10^7 sentences; task-6 and task-10 classify the 6th and 10th hidden variable.
  - OPT-generated: sampled from OPT-125M restricted to its 2,000 most frequent tokens, sentences up to 24 tokens, 2×10^8 sentences (3.2×10^9 tokens); downstream tasks QNLI and SST-2 from GLUE.
- **Minimal losses (§3):** 3.196 (PCFG), 3.758 (HMM), 1.865 (OPT-generated).
- **Models (§3, App. A.3):** BERT-style transformers from 2M to 730M parameters for PCFG and OPT data; LSTMs from 10M to 135M for HMM data.
- **Downstream evaluation (§3, App. A.4):** fine-tuning of the CLS representation, or linear probe on the concatenated token representations; 5 random seeds.
- **Lookup table (§3):** a hypothetical model whose representations are the true conditional probabilities; its loss is optimal by construction.
- **Adversarial algorithm (App. A.4):** pre-training minimizes L(ψ) − λ·(downstream validation loss of a head fitted on a disjoint split), where ψ are the feature-extractor parameters and λ weights the adversarial term; it keeps pre-training loss while lowering transfer.
- **Result, training after convergence (§3, Fig. 1):** downstream accuracy continues to increase while pre-training loss does not improve.
- **Result, model size (§3, Fig. 2):** at the same pre-training loss, scaling up improves linear-probe accuracy by 6.9% (PCFG), 4.5% (HMM), and 2.0% (OPT data).
- **Result, algorithm (Table 1, 235M, PCFG):** AdamW — loss 3.204, task A 89.9±0.3%, task B 49.2±0.8%; adversarial — 3.206, 83.1±0.6%, 42.3±1.5%; lookup table — 3.196, 71.2%, 39.7%. The adversarial model is worse on task B than a normally trained 9M model. Linear probe on standard transformer features beats linear probe on the true conditional probabilities (§3).
- **Theory (§4):** at a global minimizer the stochastic-gradient covariance equals the Hessian, Σ(θ) = ∇²L(θ) (Lemma 4.2, Bartlett identity). The cross-entropy is non-zero at the minimizers, so gradient noise does not vanish (§4). Under Assumption 4.1 (minimizers form a smooth manifold Γ), SGD with learning rate η, batch size 1, and fresh samples satisfies: θ at step K/η² converges in distribution to θ̂(K) as η → 0, where θ̂ solves dθ̂(t) = −(1/4)·∇_Γ Tr[∇²L(θ̂(t))] dt, where ∇_Γ is the gradient projected onto the tangent space of Γ (Theorem 4.3). With batch size B the coefficient 1/4 becomes 1/(4B) (§4).
- **Flatness measurement (§5, App. B.1):** the trace of the Hessian is estimated without bias by sampling x_t from the model's predicted distribution and averaging ‖∇_θ log f_θ(x_−t)_{x_t}‖².
- **Result, flatness across checkpoints (§5, Fig. 3):** after the loss converges, the trace of the Hessian decreases and downstream accuracy improves by 1.6% (235M, PCFG task C) and 4.0% (67M, HMM task-10). These runs use SGD with 12% warmup and a fixed learning rate of 1e-3 (App. B.2).
- **Result, flatness across algorithms (Table 2, 235M, PCFG task C):** AdamW — loss 3.20, accuracy 55.7±0.6%, Tr(H) 8.01±0.73; adversarial — 3.20, 50.2±1.0%, 19.34±0.92.
- **Size comparison method (§5, App. B.3–B.4):** smaller transformers are embedded into a larger architecture without changing their function, so all compared models share one parameterization.
- **Dyck-language theorem (§6, Theorem 6.1):** for a single-layer transformer with m ≥ 2 neurons and sentence length T ≥ 6, the flattest solution with zero MLM loss, followed by a minimum-norm downstream head fitted on n examples, has zero downstream population loss with probability at least 1 − 2^−n. Solutions built on random Gaussian features also reach zero MLM loss but have a larger trace of the Hessian.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Liu et al. MLM transformers (not released) | 2M–730M | pretrain (MLM; decay shape not reported) | optimizer; batch size; steps | AdamW, following Izsak et al.; 4096; more than 30K steps, until loss converges | arXiv:2210.14199v1 §3 | verified 2026-09-14 | no ablation reported |
| same | 2M–730M | pretrain (MLM; decay shape not reported) | learning rate; warmup | 1e-3; warmup proportion 0.06 | App. A.3 | verified 2026-09-14 | no ablation reported |
| Liu et al. SGD runs for Fig. 3 | 235M (PCFG), 67M (HMM) | pretrain-stable | optimizer; schedule | SGD; warmup 12% of steps, then fixed learning rate 1e-3 | App. B.2 | verified 2026-09-14 | no ablation reported |
| same models | all | eval-gate (fine-tuning) | optimizer; LR; warmup; epochs | AdamW; 1e-4; 200 steps; 10 | App. A.4 | verified 2026-09-14 | follows Devlin et al. (App. A.4) |
| same models | all | eval-gate (linear probe) | optimizer; LR; epochs; seeds | AdamW; 1e-3; 100; 5 | App. A.4 | verified 2026-09-14 | random projection to 512 dimensions did not change results (App. A.4) |

Not reported by the source: LR decay shape after warmup for the AdamW runs, weight decay, sequence packing, compute.

## Findings relevant to generality
- **Loss as a selection metric:** in the saturation regime, pre-training loss does not separate models that differ in downstream accuracy; flatness does, in the tested settings (§3, §5).
- **Scope:** for PCFG and HMM data, pre-training and downstream distributions are the same; out-of-distribution evaluation is left to future work (§3, §7). The authors state that reaching the saturation regime on real large-scale data is currently computationally challenging (§3).
- **Size:** larger models reach flatter minima because a smaller architecture is a subset of the larger one (§5, Fig. 4). This gives a reason, in this setting, why larger models can transfer better at equal loss (Interpretation by the authors).

## Connections
- [[kaplan-scaling-laws]] — cited for the observation that pre-training loss typically correlates with downstream performance as model size grows (§2).
- [[predicting-downstream-elusive]] — later work on why downstream scores are harder to predict than loss; not cited by this paper.
- [[emergence-loss-perspective]] — later work that relates emergent abilities to pre-training loss thresholds; not cited by this paper.
- [[large-batch-training-noise-scale]] — gradient-noise scale and batch size; this paper's batch-size-B result scales the implicit-bias coefficient by 1/B; not cited by this paper.
- [[resolving-scaling-discrepancies]] — compute-optimal allocation fit on pre-training loss only; not cited by this paper.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2210.14199 (arXiv v1, 2022-10-25; full PDF text including App. A–D).
- Audit claims not found in the source: none.
