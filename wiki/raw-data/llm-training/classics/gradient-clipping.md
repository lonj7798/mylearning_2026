<!-- scope: the paper that introduced global gradient-norm clipping, and its analysis of exploding and vanishing gradients in RNNs
     deps: [[adam]]
     see-also: [[mixed-precision]], [[lr-schedules]], [[karpathy-training-neural-net-recipe]]
-->

# On the difficulty of training Recurrent Neural Networks
- **Core Insight:** When the gradient norm exceeds a threshold, rescaling the whole gradient vector to that threshold (`ĝ ← threshold/‖ĝ‖ · ĝ`, Algorithm 1) keeps the update in a descent direction while bounding its length; with this rule plus a vanishing-gradient regularizer the authors solve the temporal-order problem at sequence lengths up to 200 with 100% success (§4.1.1).
- **Guideline:** When gradient norms occasionally spike during training, set the clipping threshold from the average gradient norm measured over a large number of updates, because the authors report training is not very sensitive to this value for a fixed task and model size and behaves well even for small thresholds (§3.2).
- **Authors:** Razvan Pascanu, Tomas Mikolov, Yoshua Bengio
- **Year:** 2012 (arXiv v1 2012-11; ICML 2013; text read here is arXiv v2, 2013-02-16)
- **URL:** https://arxiv.org/abs/1211.5063
- **Source type:** paper
- **Relevant topics:** optimization stability, exploding and vanishing gradients, recurrent networks, long-term dependencies

## Abstract
The paper studies the vanishing-gradient and exploding-gradient problems in recurrent neural networks from an analytical, a geometric, and a dynamical-systems perspective. The analytical part gives a sufficient condition for vanishing gradients and a necessary condition for exploding gradients in terms of the largest eigenvalue of the recurrent weight matrix. The geometric part argues that when the gradient explodes, the curvature along the same direction explodes as well, producing a steep wall in the error surface; a full-length gradient step at such a wall throws the parameters far from the valley. From this the authors derive a gradient-norm clipping rule for the exploding case, and a soft regularization term for the vanishing case. Experiments cover pathological synthetic tasks, polyphonic music prediction, and character-level language modelling.

## Key Contributions
- A sufficient condition for vanishing gradients: `λ₁ < 1/γ`, where `λ₁` is the absolute value of the largest eigenvalue of the recurrent weight matrix `W_rec` and `γ` bounds `|σ'(x)|` (§2.1, eq. 6–7). Inverting the proof gives the necessary condition for exploding gradients, `λ₁ > 1/γ`. For tanh `γ = 1`, for sigmoid `γ = 1/4` (§2.1).
- The norm-clipping algorithm (Algorithm 1, §3.2), stated as a modification of the element-wise clipping used by Mikolov (2012); the authors state the change was made so that the update always stays a descent direction for the current mini-batch, and that in practice both variants behave similarly (§3.2).
- A regularization term that prefers parameters for which the back-propagated error signal keeps its norm in the direction of `∂E/∂x_{k+1}`; it is a soft constraint, not a guarantee (§3.3).
- Empirical results: SGD-CR (clipping plus the regularizer) reaches 100% success on the temporal-order problem up to 200 steps, and the single trained model generalizes to sequences roughly twice the training length (§4.1.1).
- Improvements over plain SGD on three polyphonic-music datasets and on Penn Treebank character-level language modelling (Tables 1 and 2).

## Key Figures/Tables to Study
- **Figure 6** — error surface of a single-hidden-unit recurrent network showing a high-curvature wall, with solid arrows for a standard gradient step and dashed arrows for a rescaled step. This is the geometric argument for clipping, not an experiment.
- **Algorithm 1** (§3.2) — the four-line clipping rule.
- **Figure 7** — success rate on the temporal-order problem against log sequence length, for SGD, SGD-C (clipping), and SGD-CR (clipping plus regularizer).
- **Table 1** — polyphonic music prediction, negative log likelihood per time step, three datasets × three methods.
- **Table 2** — next-character and 5-steps-ahead character prediction on Penn Treebank, bits per character.

## Technical Details
Algorithm 1 as printed (§3.2):

```
ĝ ← ∂E/∂θ
if ‖ĝ‖ ≥ threshold then
    ĝ ← (threshold / ‖ĝ‖) · ĝ
end if
```

`ĝ` is the gradient of the cost `E` with respect to all parameters `θ`, `‖·‖` is the norm of that single vector, and `threshold` is a scalar hyper-parameter. The rescaling is uniform over all parameters, so the direction of `ĝ` is unchanged and only its length is bounded. The paper states the condition with `≥`, not `>` (§3.2).

Reported properties and limits:
- Threshold selection: "look at statistics on the average norm over a sufficiently large number of updates"; for a given task and model size training is reported as "not very sensitive" to the value, and to behave well "even for rather small thresholds" (§3.2).
- The authors describe the algorithm as a form of learning-rate adaptation based on the instantaneous gradient norm, which is what lets it react to abrupt changes in norm where methods that accumulate gradient statistics cannot (§3.2).
- Clipping addresses optimization, not regularization: on the natural tasks both training and test error improve, which the authors read as evidence that clipping solves an optimization issue and does not act as a regularizer (§4.2).
- The Penn Treebank result matches the state of the art of Mikolov et al. (2012), who used a different, element-wise clipping algorithm — the authors present this as evidence that the two variants behave similarly (§4.2).

Reported numbers:
- Polyphonic music, negative log likelihood per time step, test fold (Table 1): Piano-midi.de 7.56 (SGD) / 7.53 (SGD+C) / 7.46 (SGD+CR); Nottingham 3.80 / 3.48 / 3.46; MuseData 7.11 / 7.00 / 6.99.
- Penn Treebank character prediction, bits per character, test fold (Table 2): next character 1.50 (SGD) / 1.42 (SGD+C) / 1.41 (SGD+CR); 5-steps-ahead N/A (SGD) / 3.89 (SGD+C) / 3.74 (SGD+CR). The SGD column for the 5-step task is "N/A" — the run without clipping is not reported.
- Pathological tasks (§4.1.2): SGD-CR reached 100% success on the addition, multiplication, 3-bit temporal order, and noiseless memorization problems; the random-permutation problem succeeded in 1 of 8 trials.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Pascanu et al. RNN, addition problem | 50 hidden units, tanh | supervised training | clipping threshold (norm) | 6 | arXiv:1211.5063v2 App. "Addition problem" | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, addition problem | 50 hidden units | supervised training | learning rate; regularizer α; init | 0.01; 0.5; N(0, 0.1²) | arXiv:1211.5063v2 App. "Addition problem" | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, temporal order | 50 hidden units | supervised training | learning rate; α; clipping threshold | 0.001; 2; 6 | arXiv:1211.5063v2 App. "Temporal order problem" | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, random permutation | 100 hidden units | supervised training | learning rate; α; clipping threshold | 0.001; 1; 6 | arXiv:1211.5063v2 App. "Random permutation problem" | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, polyphonic music (Piano-midi.de, Nottingham) | 300 sigmoid hidden units | supervised training | clipping threshold (gradient averaged over the 200-step sequence) | 8 | arXiv:1211.5063v2 App. "Polyphonic music prediction" | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, polyphonic music (Piano-midi.de, Nottingham) | 300 units | supervised training | initial LR; LR rule; α schedule | 1.0; halve when epoch error increases; α_t = 1/(2t) | arXiv:1211.5063v2 App. | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, polyphonic music (MuseData) | 400 units | supervised training | initial LR; α schedule | 0.5; α_t = 1/t from 0.1 | arXiv:1211.5063v2 App. | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, Penn Treebank characters | 500 sigmoid units, no biases | supervised training | clipping threshold (cost summed over 200 steps) | 45 | arXiv:1211.5063v2 App. "Language modelling" | verified 2026-09-18 | no ablation reported |
| Pascanu et al. RNN, Penn Treebank characters | 500 units | supervised training | learning rate, next-character task | 0.01 with clipping, no regularizer; 0.05 with clipping + regularizer; 0.001 without clipping | arXiv:1211.5063v2 App. "Language modelling" | verified 2026-09-18 | the three settings are the three columns of Table 2 |
| Pascanu et al. RNN, all runs | — | supervised training | optimizer; batch size; compute | not reported (checked body §4, appendix) | arXiv:1211.5063v2 | not reported | — |

The Penn Treebank thresholds (45, and 8 for music) are not comparable to a per-token threshold: the paper states the cost is summed over the 200-step sequence for language modelling and averaged over the sequence for music.

## Findings relevant to generality
- On the pathological tasks the paper reports that models trained with SGD-CR generalize to sequences longer than those seen in training — up to about twice the training length on the temporal-order problem, and up to 400 steps for a model trained on 50–200 steps (§4.1.1, App.).
- The regularizer is reported to improve test error on tasks that are not dominated by long-term contributions (§4.2), and the authors recommend decaying it over epochs because holding it fixed makes the model favor long-term over short-term correlations (App., "Polyphonic music prediction").

## Connections
- [[adam]] — a later adaptive-optimizer line of work; this paper predates it and uses plain SGD.
- [[mixed-precision]] — the ordering of gradient unscaling relative to clipping is a concern of that source, not of this paper.
- [[lr-schedules]] — §3.2 frames clipping as a form of learning-rate adaptation driven by the instantaneous gradient norm.
- [[karpathy-training-neural-net-recipe]] — practitioner guidance that includes monitoring the gradient norm.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1211.5063 (arXiv v2, 16 Feb 2013; the ICML 2013 version).
- Corrections to the previous card version:
  - "Formal characterization ... in terms of the recurrent Jacobian's largest singular value" → the condition in §2.1 is stated in terms of `λ₁`, the absolute value of the largest **eigenvalue** of `W_rec`, compared against `1/γ`.
  - "Figure 1 (Error surface with a 'cliff')" → Figure 1 is a schematic of a recurrent network; the error-surface figure is **Figure 6** (§2.3).
  - "Figure 6 (Effect of clipping vs. no clipping): the canonical demonstration that clipping enables higher learning rates without divergence" → Figure 6 is the error-surface diagram. The success-rate comparison of SGD, SGD-C and SGD-CR is **Figure 7**.
  - "Their experiments ... show that clipping alone removes a major source of training instability, allowing much larger learning rates without divergence" → the paper reports one learning-rate comparison, in the appendix for next-character prediction: 0.01 with clipping and 0.001 without (a factor of 10, one task, one model). The main experimental claims are success rate (Figure 7) and the numbers in Tables 1–2.
  - "`||g|| > threshold`" → Algorithm 1 prints `‖ĝ‖ ≥ threshold`.
  - "Year: 2013 (ICML)" → arXiv v1 is 2012-11; the ICML publication is 2013. Both are now stated.
- Removed as unsupported by the source:
  - "typical threshold = 1.0 for LLM pretraining, 0.5–1.0 for RL", "Pretraining (GPT/Llama/Qwen lineage): max_grad_norm = 1.0", "SFT: typically 1.0; sometimes 0.5 for noisy synthetic data", "RL (PPO/GRPO): 0.5–1.0", "reward spikes ... can produce 10x norm bursts" — none of these are in a 2013 RNN paper. The thresholds it does print are 6, 8 and 45.
  - "never per-parameter clip-by-value, which destroys the gradient direction" and the "two inferior alternatives" comparison — the paper describes Mikolov's element-wise clipping as the backbone of its own approach and reports that the two variants behave similarly (§3.1, §4.2). It does not rank per-tensor clipping.
  - The FSDP / ZeRO-3 sharded-norm paragraph, the fp16 `GradScaler.unscale_` ordering paragraph, and the gradient-accumulation and "threshold too low (e.g. 0.1)" pitfalls — all post-date the paper and appear nowhere in it.
  - "the same trick later proved essential for Transformers and large-batch optimization", "the standard Llama-3 / OLMo-2 mitigation stack", "Clipping alone is necessary but not sufficient at 70B+ scale", "Most failures of RL fine-tuning trace back to either an unclipped advantage or an unclipped grad norm", "a sudden 100x spike usually predicts an imminent loss-spike or NaN" — no source given, and not in this paper.
  - Geometric "wall in error surface" is retained because §2.3 uses that wording, but the paraphrase "a single bad gradient can destroy hours of progress" is removed as not stated.
- Not reported by the source: optimizer variant beyond plain SGD, mini-batch size, wall-clock or hardware cost, any Transformer or large-language-model setting.
