<!-- scope: Karpathy's "A Recipe for Training Neural Networks" — a staged methodology for debugging and tuning neural-net training
     deps: []
     see-also: [[gradient-clipping]], [[adam]], [[lr-schedules]], [[weight-init]], [[early-stopping-and-checkpointing]]
-->

# A Recipe for Training Neural Networks
- **Core Insight:** Neural-net training misconfigurations usually raise no exception and instead produce a model that "will train but silently work a bit worse", so the post prescribes a staged process in which each step states a hypothesis and verifies it before complexity is added.
- **Guideline:** When starting a new problem, set up a linear or very small model with a fixed seed, verify the loss at initialization against its analytic value, and overfit a batch of as few as two examples before adding capacity, because until that batch reaches the lowest achievable loss the post treats the pipeline as containing a bug.
- **Authors:** Andrej Karpathy
- **Year:** 2019 (posted 2019-04-25)
- **URL:** https://karpathy.github.io/2019/04/25/recipe/
- **Source type:** anecdotal (personal blog essay on methodology; no experiments, datasets, or measured numbers are reported)
- **Relevant topics:** training methodology, silent failures, debugging, regularization ordering, hyperparameter search

## Summary
The post expands an earlier tweet on common neural-net mistakes into a process for avoiding them. It opens
with two observations. First, neural-net training is a "leaky abstraction": library snippets suggest the
technology is plug-and-play, but "Backprop + SGD does not magically make your network work". Second, training
fails silently — the error surface is logical rather than syntactic and hard to unit-test; the examples given
are a label-flip bug during augmentation that the network compensates for internally, an off-by-one that
feeds the target as input, and clipping the loss instead of the gradients. The process builds from simple to
complex, stating a hypothesis at each step and validating it before adding complexity. The post contains no
measurements and reports one quantitative rule of thumb (ensembles gaining about 2% accuracy).

## Key Contributions
- The two framing claims: training as a leaky abstraction, and silent failure as the dominant failure mode.
- A six-stage process (become one with the data; skeleton and dumb baselines; overfit; regularize; tune;
  squeeze out the juice), with a 13-item verification checklist for the skeleton stage.
- An ordered regularizer list headed by "get more data", and two defaults for early experiments: "Don't be a
  hero" (copy the simplest architecture from the most related paper) and Adam at a learning rate of 3e-4.

## Key Figures/Tables to Study
The post is prose with no figures or tables; its headings are the index ("Neural net training is a leaky
abstraction", "Neural net training fails silently", "The recipe", stages 1–6).

## Technical Details — the six stages

**1. Become one with the data.** No neural-net code is written. The stated practice is spending time
"measured in units of hours" scanning thousands of examples for distribution and patterns; reported
discoveries include duplicate examples and corrupted images or labels. The post also recommends code to
search, filter, sort and visualize outliers, which "almost always uncover some bugs in data quality or
preprocessing" (§1).

**2. Set up the end-to-end skeleton and get dumb baselines.** Start with a model that could not plausibly be
wrong, such as a linear classifier or a very tiny ConvNet. The listed checks (§2): fix the random seed;
disable non-essential features including augmentation; evaluate over the full test set rather than smoothing
per-batch losses; verify loss at initialization (for a softmax with a correctly initialized final layer,
`-log(1/n_classes)`); initialize the final-layer bias to the target mean, or the class ratio for imbalanced
data, so the first iterations are not spent learning the bias; track a human-interpretable metric and a human
baseline; train an input-independent baseline with inputs zeroed and confirm it is worse; overfit one batch
of as few as two examples to the lowest achievable loss and confirm predictions align with labels; verify
training loss decreases when capacity is raised slightly; visualize the tensors immediately before
`y_hat = model(x)`, and prediction dynamics on a fixed test batch; use backprop to chart dependencies (set
the loss to the sum of outputs for example `i` and require a non-zero input gradient only on the `i`-th);
write the specific case first and generalize afterwards.

**3. Overfit.** Two stages: get a model large enough to overfit, focusing on training loss, then regularize
it, giving up training loss for validation loss. Tips (§3): "Don't be a hero" — copy the simplest architecture
from the most related paper, for example ResNet-50 for image classification; "adam is safe" — Adam at learning
rate 3e-4 for baselines, with the note that a well-tuned SGD will almost always slightly outperform Adam on
ConvNets but has a narrower optimal learning-rate region; add one input signal at a time; do not trust decay
defaults from another domain, because schedules keyed to epoch number depend on dataset size (ImageNet
decaying by 10 at epoch 30). The author disables decay, uses a constant learning rate, and tunes it last.

**4. Regularize.** The ordered list (§4): get more real data, the only way known to the author to improve a
well-configured network monotonically, with ensembles as the other option but topping out after about 5
models; data augmentation; creative augmentation (domain randomization, simulation, GANs); pretraining,
which "rarely ever hurts"; supervised rather than unsupervised pretraining, qualified by NLP doing well with
BERT at the time of writing; smaller input dimensionality; smaller model size; decreased batch size, which
regularizes through batch-norm statistics; dropout, used sparingly because it "does not seem to play nice
with batch normalization"; increased weight decay; early stopping on validation loss; and finally a larger
model, whose early-stopped performance can beat a smaller model's.

**5 and 6. Tune, then squeeze out the juice.** Stage 5 has two items (§5): random search rather than grid
search, because networks are much more sensitive to some hyperparameters than others; and a remark on
Bayesian hyperparameter-optimization toolboxes, where the author reports second-hand success and states his
own preference as a joke. Stage 6 (§6): ensembles gaining about 2% accuracy, distillation into a single
network when test-time compute is limited, and training left running longer than the validation curve
suggests.

## Findings relevant to generality and distillation
- Distillation appears once: distilling an ensemble into a single network when test-time compute is limited
  (§6); no method or result is given. The post covers supervised training practice in 2019, does not address
  language-model pre-training, post-training, long context, or agentic training, and reports no
  generalization measurement.

## Connections
- [[weight-init]] — "verify loss @ init" and "init well" state what an initialization audit measures
  first; [[adam]] — 3e-4 is a baseline default; [[lr-schedules]] — a constant learning rate during development, with decay tuned last, is one position.
- [[early-stopping-and-checkpointing]] — early stopping is in the regularizer list, ensembling in stage 6;
  [[dropout]], [[label-smoothing]] — dropout's place in the list and its interaction with batch norm.
- [[gradient-clipping]] — clipping appears only as an example of a silent bug (the loss clipped instead of
  the gradients); the post gives no clipping recommendation.

## Verification
- Checked on 2026-09-18 against https://karpathy.github.io/2019/04/25/recipe/ (posted 2019-04-25).
- Corrections to the previous card version:
  - "Overfit a single batch" placed in stage 3 → it is a stage-2 checklist item; stage 3 is about reaching a
    model large enough to overfit the training set.
  - Quote "I look at thousands of examples, understand their distribution, and look for patterns" → the text
    reads "I like to spend copious amount of time (measured in units of hours) scanning through thousands of
    examples, understanding their distribution and looking for patterns" (§1). "Initial loss ≈ `ln(K)`" →
    the post writes `-log(1/n_classes)` for a softmax (§2).
  - "Tune LR schedule last" attributed to stage 5 → disabling decay, using a constant learning rate and
    tuning it at the very end is stated in stage 3; "coarse to fine: wide-range random sweep → local
    refinement" → stage 5 states random over grid search only, with no coarse-to-fine procedure.
  - "Get more data (always first; dwarfs every other regularizer)" → the post calls it "the by far best and
    preferred way to regularize" (§4). "Ensemble several runs (2% easy gain)" → "pretty much guaranteed way
    to gain 2% of accuracy", no experiment; the "~5 models" ceiling is in stage 4, not stage 6. The author
    biography ("Stanford CS231n lecturer and head of Tesla Autopilot AI") is not in the post; removed.
- Removed as unsupported by the source:
  - Quotes "if you can't overfit a single batch, you can't overfit the training set", "Once a batch fits,
    overfit a small dataset (200 examples)", "Review the 10 worst validation examples" — not in the post.
  - "`model.eval()` vs `model.train()` give same outputs when no dropout/BN is active" and "Be paranoid
    about `model.train()` vs `model.eval()`"; "Monitor and clip gradient norms" as a recommendation;
    "Bayesian optimization is 'possible but slow and annoying'" as a quote — none are in the post.
  - "arguably the most cited workflow document in practical ML", "still a correct default for small-model
    prototyping in 2025", "the direct source of the clip-norm-1.0 defaults", and the paragraph attributing
    Tülu 3 / OLMo 2 / Llama 3 post-training failures to this list — claims about other work.
- Not reported: any dataset, model, metric or measured result; guidance specific to language models.
