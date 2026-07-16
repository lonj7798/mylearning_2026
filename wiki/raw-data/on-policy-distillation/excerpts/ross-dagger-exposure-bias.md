<!-- scope: DAgger — exposure bias / compounding error and the on-policy data-collection cure
     deps:
     see-also: tm-on-policy-distillation, agarwal-gkd -->
# DAgger: Compounding Error and the On-Policy Fix (Ross, Gordon, Bagnell, 2011)

- **Core Insight:** A policy trained by imitation on the *expert's* state distribution (behavioral cloning / off-policy) makes small errors that push it into states the expert never demonstrated, where it errs again — so mistakes **compound**, giving error that grows as O(T²) in the horizon T rather than O(T).
- **Guideline:** Fix the distribution mismatch by collecting training data **on-policy** — repeatedly run the *learner's own* policy, have the expert label the states it actually visits, aggregate that into the dataset, and retrain (DAgger); this restores O(T) error.
- **Source:** Stéphane Ross, Geoffrey Gordon, J. Andrew Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS 2011, arXiv:1011.0686. **Thesis extracted, not verbatim** — primary-source fetch was blocked by a monthly-spend limit on 2026-07-06; the results below are the paper's well-established claims.
- **Relevant chapters:** on-policy principle (the "why"), exposure bias, foundations

## The compounding-error result

Let the per-step probability that the learner disagrees with the expert be ε on the *expert's* state distribution. Under behavioral cloning the induced state distribution differs from the training one, and the classic bound is that the total cost grows like **O(ε · T²)** in the horizon T — quadratic, because each early error shifts the learner into unfamiliar states where the error rate is no longer bounded by ε. An on-policy method that trains under the learner's *own* induced distribution achieves **O(ε · T)** — linear. The quadratic-vs-linear gap is the formal statement of **exposure bias**.

## DAgger — the iterative on-policy loop

1. Train an initial policy from expert demonstrations.
2. **Run the current policy** in the environment; record the states it visits.
3. Ask the **expert** for the correct action at each visited state (the labels).
4. **Aggregate** these (state, expert-action) pairs into the dataset; retrain.
5. Repeat. As a no-regret online-learning reduction, the visited-state distribution and the training distribution converge, killing the compounding error.

The essential idea: *collect data where the learner actually goes, and label it with a better policy.*

## Why this is the backbone of on-policy distillation

On-policy distillation of LLMs is DAgger applied at the token level with a soft label:
- "Run the current policy" → **sample sequences from the student** ([[tm-on-policy-distillation]], [[agarwal-gkd]] λ→1).
- "Expert labels the visited states" → the **teacher's per-token distribution** grades each student-visited prefix (a dense soft label instead of a single expert action).
- Result: the student is corrected in exactly the states its own generation reaches — the cure for the O(T²) drift that off-policy SFT / sequence-level KD suffer on long autoregressive outputs.

GKD ([[agarwal-gkd]]) explicitly names **ImitKD** (a DAgger-style KD baseline) as its ancestor and improves on it.

## Connections
- Token-level realization: [[tm-on-policy-distillation]], [[agarwal-gkd]]. The off-policy methods whose exposure bias this explains: [[hinton-knowledge-distillation]], [[kim-rush-seqkd]]. Distribution-matching framing: [[nrehiew-sft-rl-opd]].
