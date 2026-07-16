<!-- scope: origin of knowledge distillation — soft targets / temperature-softened softmax; off-policy KD baseline
     deps:
     see-also: kim-rush-seqkd, agarwal-gkd -->
# Distilling the Knowledge in a Neural Network (Hinton, Vinyals, Dean, 2015)
- **Core Insight:** A small "student" can absorb a large teacher's knowledge by training on the teacher's *temperature-softened class probabilities* ("dark knowledge") rather than hard labels — the soft distribution encodes inter-class similarity structure the hard label discards.
- **Guideline:** Train the student to match the teacher's softmax at a raised temperature T, blend that soft-target cross-entropy with the ordinary hard-label cross-entropy as a weighted sum, and scale the soft-target gradients by T² so both terms stay comparable.
- **Source:** arXiv:1503.02531, https://arxiv.org/abs/1503.02531 (abstract fetched verbatim; body quotes verified via ar5iv HTML mirror https://ar5iv.labs.arxiv.org/abs/1503.02531). Geoffrey Hinton, Oriol Vinyals, Jeff Dean, "Distilling the Knowledge in a Neural Network," 2015.
- **Relevant chapters:** off-policy distillation, KD foundations, soft targets & temperature

## The core reframe: knowledge = a mapping, not weights
"A more abstract view of the knowledge, that frees it from any particular instantiation, is that it is a learned mapping from input vectors to output vectors." This is why a *different, smaller* architecture can inherit it. The teacher (called the "cumbersome model") is often an ensemble; distillation compresses it into one deployable net.

Abstract (verbatim): "A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions." … "We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model."

## Temperature-softened softmax ("dark knowledge")
The student matches probabilities produced by:

    q_i = exp(z_i / T) / Σ_j exp(z_j / T)

where z_i are logits and T is temperature. T = 1 is ordinary softmax; higher T "produces a softer probability distribution over classes." The soft distribution exposes which *wrong* classes the teacher finds plausible (e.g. a "2" that looks a bit like a "3" or "7") — that relative structure is the "dark knowledge."

Why soft targets beat hard labels (verbatim): "When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases." More info-per-example + lower gradient variance ⇒ the student can be trained on far less data / smaller nets and still generalize.

Why generalization transfers (verbatim): "If the cumbersome model generalizes well because, for example, it is the average of a large ensemble of different models, a small model trained to generalize in the same way will typically do much better on test data than a small model that is trained in the normal way on the same training set as was used to train the ensemble."

## The distillation loss (weighted sum of two cross-entropies)
"a better way is to simply use a weighted average of two different objective functions":
- **Objective 1 — soft-target CE:** "cross entropy with the soft targets and this cross entropy is computed using the same high temperature in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model."
- **Objective 2 — hard-label CE:** "cross entropy with the correct labels … computed using exactly the same logits in softmax of the distilled model but at a temperature of 1."

Gradient balancing (verbatim): "Since the magnitudes of the gradients produced by the soft targets scale as 1/T² it is important to multiply them by T² when using both hard and soft targets." Objective 2 is typically down-weighted; the soft-target term carries most of the signal.

## Key numbers (MNIST)
- Large net: "achieved 67 test errors"; a plain smaller net (two 800-unit ReLU hidden layers, no regularization): "146 errors."
- Same small net regularized *only* by matching soft targets at T = 20: "74 test errors" — soft targets alone recover most of the gap, acting as a strong regularizer.
- Omitted-class transfer: even when the digit **3** is entirely withheld from the transfer set, the student still classifies 3s well — "the distilled model only makes 206 test errors of which 133 are on the 1010 threes in the test set," and after a bias fix "the distilled model makes 109 errors of which 14 are on 3s." Soft targets teach an unseen class via its similarity to seen classes.
- Also validated at scale: significant improvement to a "heavily used commercial" speech acoustic model.

## Trade-offs & position in the distillation family
- **This is the OFF-POLICY baseline.** The teacher labels a *fixed* dataset (the "transfer set"); the student trains on those frozen soft targets and never generates its own samples during training. No feedback loop between student outputs and training distribution ⇒ exposure/train-inference mismatch is unaddressed here.
- **Classification-era, per-token, single-step.** The formulation matches one softmax over a fixed class set — not autoregressive sequences.
- Forward connections:
  - [[kim-rush-seqkd]] lifts KD to *sequence-level* (Sequence-Level Knowledge Distillation): distill over the teacher's output *sequences* (via beam/greedy teacher generations), still off-policy but sequence-aware — the bridge from classification KD to seq2seq/LM KD.
  - [[agarwal-gkd]] makes distillation *on-policy* (Generalized KD): the student trains on *its own* sampled sequences scored by the teacher, closing the train/inference distribution gap that this fixed-dataset baseline leaves open; also generalizes the divergence (forward/reverse KL) beyond the plain soft-target CE used here.
