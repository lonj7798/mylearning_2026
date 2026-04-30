<!-- scope: Karpathy's "A Recipe for Training Neural Networks" — operational methodology for deep-net training
     deps: []
     see-also: [[gradient-clipping]], [[adam]], [[lr-schedules]], [[weight-init]], [[early-stopping-and-checkpointing]]
-->

# A Recipe for Training Neural Networks (Karpathy 2019)
- **Core Insight:** Neural network training is a leaky abstraction that fails silently; the only defense is an obsessive, data-first, incremental workflow where every step is verified against an explicit prediction.
- **Guideline:** Follow the six-step recipe in order — inspect data → build skeleton + dumb baseline → overfit → regularize → tune → squeeze — and never skip a step because "this time is different."
- **Authors:** Andrej Karpathy
- **Year:** 2019 (April)
- **URL:** https://karpathy.github.io/2019/04/25/recipe/
- **Relevant topics:** training methodology, debugging, research practice, generalization

## Abstract
Karpathy — formerly Stanford CS231n lecturer and head of Tesla Autopilot AI — documents the practical methodology he arrived at after years of training neural networks. The thesis: deep learning is not a plug-and-play technology. Training fails silently (a bug shows up as "the model doesn't work," not as an error message), so practitioners must adopt an incremental, hypothesis-driven workflow. The post enumerates common silent failures (incorrect loss, forgotten `.eval()`, wrong data-loader ordering, etc.) and prescribes a six-step recipe that treats training as empirical science rather than engineering.

## Key Contributions
- The framing: **"neural net training is a leaky abstraction"** — unlike standard SWE where libraries compose, ML components silently corrupt each other's assumptions.
- The six-step recipe below — arguably the most cited workflow document in practical ML.
- The concrete heuristic **"Adam 3e-4 is safe"** — still a correct default for small-model prototyping in 2025.
- **"Overfit a single batch"** as the mandatory pipeline sanity check before any serious experiment.
- The principle **"don't be a hero"** — use the simplest known-good architecture; inventing novelty before baseline is a near-universal time-waster.

## Key Figures/Tables to Study
There are no formal figures — this is a prose essay. Skim for these sections:
- "Neural net training fails silently" (pitfalls list)
- "A recipe" (the 6-step framework)
- "Overfit" (the single-batch check)
- "Regularize" (ordered list of regularizers)
- "Tune" (coarse-to-fine hyperparameter search, random > grid)

## Technical Details — The Six-Step Recipe

### 1. Become one with the data
Spend *hours* looking at raw examples. Sort by every attribute you can think of. Find the duplicates, the corrupt records, the label-noise patterns. Most production-level ML wins come from data fixes, not model changes. Quote: "I look at thousands of examples, understand their distribution, and look for patterns."

### 2. Set up the end-to-end training/evaluation skeleton + get dumb baselines
Build the smallest end-to-end system: tiny model, dumb baseline (linear / constant / copy-input), full eval loop. Verify:
- Initial loss ≈ expected (`ln(K)` for uniform classification; entropy of target distribution for regression).
- `model.eval()` vs `model.train()` give same outputs when no dropout/BN is active.
- The model can't improve by more than a known amount on the simplest proxy task.
- Fix random seed everywhere; turn off every non-essential feature.

### 3. Overfit
Crank model capacity until you can **overfit one batch** to near-zero loss. If you cannot, the pipeline is broken — stop everything and debug. Once a batch fits, overfit a small dataset (200 examples). Only then expand. Quote: "if you can't overfit a single batch, you can't overfit the training set."

### 4. Regularize
Only after overfitting is confirmed, add regularization in order of impact:
- **Get more data** (always first; dwarfs every other regularizer).
- Data augmentation.
- Creative augmentation (domain-specific).
- Pretrain (transfer).
- Smaller model.
- Smaller input dim.
- Decrease batch size.
- Dropout.
- Weight decay.
- Early stopping.
- Try a larger model with stronger regularization (can beat smaller model).

### 5. Tune
- **Random search > grid search** in high-dim HP spaces.
- Coarse to fine: wide-range random sweep → local refinement.
- Bayesian optimization is "possible but slow and annoying"; random usually wins in practice.
- Tune LR schedule last — after architecture is fixed.

### 6. Squeeze out the juice
- Ensemble several runs (2% easy gain).
- Leave training running longer than you think necessary; models keep improving slowly.
- Review the 10 worst validation examples — they reveal systematic errors.

## Famous Heuristics and Pitfalls (from the post)

| Maxim | What it protects against |
|---|---|
| "Neural net training fails silently." | Assuming green curves = correct code. |
| "Be paranoid about `model.train()` vs `model.eval()`." | BN/dropout active at eval. |
| "Init well." | Loss too high at step 0; bad convergence. |
| "Visualize just before the net." | Mis-normalized inputs; label mismatches. |
| "Generalize a special case." | Hard-to-debug loops; always write the `N=1` case first. |
| "Use backprop to chart dependencies." | Data leakage bugs; find them by setting one example's loss to zero and checking that no gradient flows to other examples. |
| "Monitor and clip gradient norms." | Silent divergence. |
| "Use a constant LR for sanity; schedule last." | Hyperparameter confounding. |
| "Adam `3e-4` is a safe default." | HP-sensitivity in early experiments. |
| "Don't be a hero." | Premature novelty; wasted months. |

## Connections

- **[[gradient-clipping]]**: Karpathy's "monitor and clip" is the direct source of the clip-norm-1.0 defaults now standard in every LLM recipe.
- **[[adam]]**: "Adam 3e-4" as a safe baseline is still the small-model rule; LLM pretraining moves lower but the logic is identical.
- **[[lr-schedules]]**: "Constant LR for sanity, schedule last" — matches the practice of running a warmup-only or constant-LR ablation before committing to cosine/WSD.
- **[[weight-init]]**: "Init well — at minimum, ensure initial loss equals uniform baseline" — the first thing an init audit should check.
- **[[early-stopping-and-checkpointing]]**: predates SWA/souping but captures the same instinct — "ensemble several runs."
- **[[dropout]] / [[label-smoothing]]**: the ordered-list-of-regularizers remains correct; modern LLM practice is "get more data" + weight decay, with dropout reserved for small-data SFT and label smoothing for SFT.
- **Post-training relevance (2025)**: the SFT / DPO / PPO workflow failures in 2024–2025 reports (Tülu 3, OLMo 2, Llama 3 postmortems) are all variants of Karpathy's silent-failure list — wrong tokenizer, wrong prompt template, forgotten masking, mis-normalized reward. The recipe is just as relevant to RL fine-tuning as it was to CNN training.
