---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md
source_url: https://karpathy.github.io/2019/04/25/recipe/
created_at: "2026-04-23"
---

# Excerpt: Karpathy's Recipe — training fails silently, so instrument obsessively

**Source library:** `wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md`
**Post:** Karpathy 2019, *"A Recipe for Training Neural Networks."*

---

## Why this source anchors ch-07

This blog post is the organizing rule of the entire chapter. Karpathy's thesis — "neural net training is a leaky abstraction that fails silently" — is not a nice slogan; it is a precise engineering claim that motivates every assertion in ch-07 §7's checklist. The failure modes in §1 (NaN), §2 (spike/divergence/plateau), §3 (dead pipeline), §4 (masking), §5 (NCCL hang), and §6 (entropy collapse) all share one property: they produce no runtime error, no Python exception, no pytest red bar. They produce slightly-wrong loss curves and wrong downstream benchmarks, which is the definition of silent failure.

The source frames the response:

> *"The framing: 'neural net training is a leaky abstraction' — unlike standard SWE where libraries compose, ML components silently corrupt each other's assumptions."*

"Libraries compose" — Karpathy's deeper point — is about testability. A Python HTTP library has well-defined inputs and outputs; mis-use produces an exception. A loss function has well-defined inputs and outputs mathematically, but mis-configured gradients produce a number that looks plausible. There is no type system for "this gradient is semantically wrong." The only defense is Karpathy's six-step recipe, which treats training as empirical science rather than engineering.

---

## The two cheap pre-flight checks

Ch-07's guideline opens with two checks from this source:

> *"Initial loss ≈ expected (`ln(K)` for uniform classification; entropy of target distribution for regression)."*

For LLM pretraining with vocabulary size V, the initial loss under a uniform softmax is `ln(V)`. For V = 128 256 (Llama 3 tokenizer), `ln(V) ≈ 11.76`. Any first-step loss not within ~2% of this value means one of three things:

1. **Weights are not random.** A non-default init (µP, scaled-init for deep networks) may produce a different target but still predictable.
2. **The loss function is not CE over `V` classes.** Z-loss adds a regularizer; label smoothing shifts the target.
3. **The label mask is wrong.** If the whole batch is masked except one token, the loss is averaged over one position, giving a sample from the loss distribution that may not be `ln(V)`.

A `ln(V)`-check at step 0 costs one forward pass and catches a staggering number of configuration bugs at their cheapest point.

> *"'if you can't overfit a single batch, you can't overfit the training set.'"*

The second cheap check: disable dropout, disable weight decay, take one fixed batch of 8–32 examples, train on it for ~200 steps with a reasonable LR. Loss should descend to < 0.01. If it doesn't, the pipeline is broken — dataloader, mask, loss function, or optimizer — and no amount of scaling will fix it. This is the ch-07 §7 `overfit_single_batch_to_near_zero(model, batch, steps=200)` assertion.

Notice: both checks run *before* a full training run. Ch-06's instrumentation is the in-run version of the same discipline; these two checks are the one-time version. You only pay for them once per code change, and they catch ~80% of silent-failure bugs at the cheapest possible point.

---

## The silent-failure list — what ch-07 makes operational

The source enumerates pitfalls in a table ch-07 §7 reorganizes into assertions:

| Karpathy maxim | Ch-07 corresponding assertion |
|---|---|
| "Be paranoid about `model.train()` vs `model.eval()`." | check dropout/BN state matches phase |
| "Init well." | `loss(0) ≈ ln(V)` |
| "Visualize just before the net." | `decode(input_ids[labels != -100])` audit |
| "Use backprop to chart dependencies." | `assert no_gradient_leaks_between_samples` |
| "Monitor and clip gradient norms." | `pre_clip_grad_norm` logged + `isfinite(loss)` |
| "Adam `3e-4` is a safe default." | ch-01 lineage; for LLMs it's lower but logic is the same |
| "Don't be a hero." | use a known-good architecture before novelty |

The "don't be a hero" rule has a ch-07 corollary: *don't write your own FSDP clip, your own packing collator, your own attention kernel*. Every instance of each in the chapter's failure surface is a bug that would not exist if a stock implementation had been used. The source's Technical Details section (step 2, "dumb baselines") encodes this: start with the simplest working system, add complexity only once the baseline is verified.

---

## Why the pitfalls list is short — and why that matters

The source's famous-maxims table is ~8 rows long. That is not comprehensive; it is the tight set of rules that catches the majority of silent failures observed across Karpathy's career (CS231n student projects through Tesla Autopilot). Ch-07's §7 checklist is similarly short by design: five pre-run, five per-step, four periodic, two on-resume. Each is cheap enough to run always, and each catches ≥ 1 person-week of debugging time per hit.

The key design principle: *an assertion that fails at step 100k is almost useless compared to the same assertion firing at step 1*. Karpathy's two pre-flight checks exist precisely because step-0 failures are the cheapest to fix. Ch-07's per-step assertions exist because step-k failures are still cheaper to fix than step-10k failures. Ch-06's periodic assertions exist for the drift-accumulation class that step-0 checks can't see.

The anti-pattern: a tenth-of-a-second `is_everything_ok()` function that runs every 1000 steps, silently catches bugs only on resume, and tells you "something broke sometime in the last 1000 steps." That is worse than no check at all, because it generates false confidence and narrows the debugging window by 1000×.

---

## The "overfit a single batch" check as a ch-07 sanity gate

The source's step 3 (Overfit) is the single most useful diagnostic in the chapter:

> *"Crank model capacity until you can overfit one batch to near-zero loss. If you cannot, the pipeline is broken — stop everything and debug."*

Ch-07 §3 (dead pipeline) is almost always caught by this one check. An all-masked batch has nothing to overfit to; the loss stays at the masked-dead value. An all-padding batch produces a loss that cannot be reduced because there is no signal. A mis-shifted CE produces a loss that descends too slowly or to the wrong floor. Running the overfit check on the *same* collator that ships to training is the operational definition of "the pipeline is correct."

This check is particularly effective at catching ch-07 §4a off-by-one masking bugs. Wrong #1 (post-shift mask) leaks one prompt token into the loss; overfitting one batch to that slightly-wrong loss converges to a slightly-higher floor (~0.02 absolute). Wrong #3 (no-shift alignment) converges to effectively-zero loss immediately because teacher-forcing is trivial; the "too good to be true" signal should raise suspicion.

The test is quantitative: a correct LLM SFT pipeline overfits a single 32-example batch to loss < 0.01 within 200 steps at LR = 1e-4. Any different value is a diagnostic signal.

---

## "Fix the random seed everywhere" — the reproducibility prerequisite

From the source:

> *"Fix random seed everywhere; turn off every non-essential feature."*

This is ch-06's bit-exact-resume prerequisite applied to ch-07's debugging context. If you can't reproduce a bug, you can't diagnose it. The Python RNG, NumPy RNG, CUDA per-rank RNG, dataloader worker RNGs, and the augmentation/dropout RNG are *six* sources that need seeding; one unseeded means the bug appears intermittently and the diagnostic narrows by guesswork.

Ch-06 §3 covers the mechanics; the ch-07 perspective is different: *during debugging*, you want determinism so the bug is reliably reproduced, not so the run is reproducible in production. A debug script with `torch.use_deterministic_algorithms(True)` and explicit seeds across all six sources is slower but makes bugs binary (present or absent) rather than probabilistic.

---

## The tuning ordering — why LR schedule comes last

From the source:

> *"Tune LR schedule last — after architecture is fixed."*

This is the explicit ordering rule that ch-07 §2's plateau-debug branch enforces. A plateau on an LR schedule that was tuned *before* the architecture was settled is ambiguous — maybe the schedule is wrong, maybe the architecture is wrong. A plateau after LR is tuned last can be attributed to the schedule alone because every other variable has been frozen.

Notice how this ordering intersects with ch-06's WSD fork pattern: the stable-phase checkpoint is produced under a known-good architecture + constant LR; the decay fork is the schedule experiment. Karpathy's "constant LR for sanity, schedule last" and WSD's "stable trunk, decay fork" are the same discipline at two time scales.

## Step 6 — "Squeeze out the juice" and its ch-07 form

From the source:

> *"Ensemble several runs (2% easy gain). Leave training running longer than you think necessary; models keep improving slowly. Review the 10 worst validation examples — they reveal systematic errors."*

The "review the 10 worst validation examples" rule is ch-07's final diagnostic tier. After every per-step assertion has passed and every periodic audit has run clean, the remaining silent-failure modes surface only in the tail of the eval distribution. A 10-example review catches systematic issues — a particular chat template the model fails on, a particular domain it never learned — that aggregate metrics hide. This is the *post-training* ch-07 check; it doesn't replace the per-step assertions, it follows them.

## What to take from Karpathy for ch-07

1. **Training fails silently.** No exception, no red bar. The only defense is inline assertions.
2. **Two pre-flight checks** (`ln(V)` loss, overfit one batch) catch ~80% of silent bugs at their cheapest.
3. **Short assertion lists beat long periodic audits.** Every assertion should be cheap enough to run always.
4. **Don't be a hero.** Use stock implementations of clipping / packing / attention before inventing.
5. **Fix random seeds everywhere during debugging.** Intermittent bugs are un-diagnosable.
6. **Tune LR schedule last.** Otherwise the plateau-debug branch is ambiguous about what's wrong.
7. **Review the 10 worst validation examples.** The post-training tier of silent-failure detection.

---

## Connections

- [[excerpts/gradient-clipping]] — "monitor and clip" is the direct source of ch-07's pre-clip-norm ubiquity.
- [[excerpts/mixed-precision]] — "start in fp32 for debugging" aligns with Karpathy's start-simple rule.
- [[excerpts/loss-masking-prompt]] — "visualize just before the net" = the decoded-label audit of §4a.
- [[excerpts/sequence-packing]] — "don't be a hero" applies directly to custom attention kernels.
- [[excerpts/olmo-2]] / [[excerpts/olmo-3]] / [[excerpts/llama-3]] — frontier-scale validation of the recipe's universality.
- [[ch-07]] — the entire chapter is an instance of this source's "silent failure" framing; §7's checklist is the recipe's pitfalls-list made into code.
