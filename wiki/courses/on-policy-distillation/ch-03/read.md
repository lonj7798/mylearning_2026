<!-- chapter: ch-03
     track: on-policy-core
     kind: content
     title: Exposure Bias and the On-Policy Principle — DAgger at Token Level
     deps: [[ch-02]]
     sources: [[ross-dagger-exposure-bias]], [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]], [[agarwal-gkd]], [[kim-rush-seqkd]]
-->

# Chapter 03 — Exposure Bias and the On-Policy Principle: DAgger at Token Level

> **Core insight.** The disease of the off-policy corner ([[ch-02]]) has a name and a bound. When a model is trained to imitate on the *teacher's* state distribution but then runs on its *own*, early mistakes push it into states the teacher never demonstrated, where it makes further mistakes — so error **compounds**. The classical result (Ross et al., DAgger) is that behavioral cloning accrues error growing as **O(ε·T²)** in the horizon T, while an *on-policy* data-collection scheme achieves **O(ε·T)**. The quadratic-versus-linear gap is the entire reason on-policy training exists — and on-policy distillation is DAgger applied at the token level with a soft teacher label.

> **Guideline.** If your model produces long autoregressive outputs and is trained off-policy, assume it will drift — and do not try to fix it by adding more teacher data. Exposure bias is a *distribution-shift* problem, not a data-quantity problem. The fix is structural: collect (or sample) data where the *model itself* goes, and grade it there.

---

## 1. Why the off-policy baseline decays

Chapter 2 ended on a placement: classical KD and SFT both train the student on teacher-visited trajectories with mode-covering forward KL. That is fine while the student stays on the teacher's path. The problem is that at inference the student generates *autoregressively* — each token conditions on the student's own previous tokens — so it inevitably leaves the teacher's path, and it was never trained on what to do out there. Thinking Machines states the mechanism ([[tm-on-policy-distillation]]):

> "The drawback of off-policy training is that the student learns in contexts frequented by the teachers, not ones the student itself will often find itself in. This can cause compounding error: if the student makes an early mistake that the teacher never makes, it finds itself diverging ever farther from the states it observed in training."

nrehiew gives the autoregressive version ([[nrehiew-sft-rl-opd]]): "because of autoregression, one mistake by the student might move it to states the teacher would have never visited." The literature's name for this is **exposure bias**: the model is never *exposed* at training time to the states its own errors produce.

---

## 2. The compounding-error bound (DAgger)

Ross, Gordon & Bagnell (2011) made this precise by reducing imitation learning to online learning ([[ross-dagger-exposure-bias]]). Let ε be the per-step probability that the learner disagrees with the expert *on the expert's own state distribution* — the thing behavioral cloning actually minimizes. The results:

- **Behavioral cloning (off-policy):** total cost grows like **O(ε · T²)**. The quadratic comes from distribution shift — each early error moves the learner to unfamiliar states where the error is no longer bounded by ε, so errors cascade over the horizon.
- **On-policy data collection:** **O(ε · T)** — linear. Train under the learner's *own* induced distribution and the cascade disappears.

The T² vs T gap is not asymptotic hair-splitting: at a horizon of tens or hundreds of steps it is the difference between a model that recovers from its own mistakes and one that spirals. This is the formal statement behind [[ch-02]]'s warning that Kim & Rush's teacher-mode sequences leave the student blind to its own states.

> **Interactive companion:** [`figures/compounding-error.html`](figures/compounding-error.html) — set the per-step error ε and the horizon T and watch the two curves separate: behavioral cloning's O(ε·T²) against on-policy's O(ε·T), with a schematic of a student trajectory drifting off the teacher's demonstrated path into ungraded states. It is the picture of why long sequences punish off-policy training.

---

## 3. DAgger: the on-policy data-collection loop

DAgger ("Dataset Aggregation") is the fix, and its shape is worth memorizing because on-policy distillation is the same shape ([[ross-dagger-exposure-bias]]):

1. Train an initial policy from expert demonstrations.
2. **Run the current policy** in the environment; record the states it actually visits.
3. Ask the **expert** for the correct action at each of those visited states (the labels).
4. **Aggregate** the new (state, expert-action) pairs into the dataset; retrain.
5. Repeat. As a no-regret online-learning reduction, the visited-state distribution and the training distribution converge — killing the compounding error.

The one-sentence essence: **collect data where the learner actually goes, and label it with a better policy.** Everything downstream in this course is a variation on that sentence.

---

## 4. On-policy distillation = DAgger at token level

Now map DAgger onto language-model distillation. The correspondence is exact except that the "label" becomes a *soft distribution* instead of a single action:

| DAgger | On-policy distillation |
|---|---|
| Run the current policy | **Sample sequences from the student** ([[tm-on-policy-distillation]], [[agarwal-gkd]] with λ→1) |
| Record visited states | The token-prefixes the student actually generates |
| Expert labels each state | The **teacher's full next-token distribution** at each student-visited prefix |
| Aggregate + retrain | Minimize per-token divergence to the teacher on those prefixes |

So the student is corrected in exactly the states its own generation reaches — the O(T²) → O(T) cure, delivered densely (a whole distribution per token) rather than as one expert action. Thinking Machines' one-liner is the payoff ([[tm-on-policy-distillation]]): "by training on samples from itself, the student learns to avoid mistakes in a more direct way." GKD even names its DAgger-style ancestor — **ImitKD** — as a baseline it improves on ([[agarwal-gkd]]).

---

## 5. RL is on-policy too — and even self-SFT drifts

Two refinements sharpen the principle.

**RL has the on-policy property but a starving signal.** RL samples from the current policy (on-policy) and so also escapes exposure bias, but its feedback is a sparse scalar — [[ch-01]]'s O(1) bits. It gets the *data source* right and the *density* wrong. On-policy distillation keeps the data source and fixes the density.

**On-policy is a moving target, not a one-time property.** A subtle point from Thinking Machines ([[tm-on-policy-distillation]]): even training a model on *its own* samples slowly becomes off-policy, because each gradient step moves the policy away from the distribution that generated the batch —

> "while KL divergence is 0 in expectation, every finite batch will exhibit a slightly different distribution in practice… This process turns training on one's own samples into off-policy training over time."

The lesson: "on-policy" is a property you must *maintain* by continually resampling from the current student, not a box you check once. This is why on-policy distillation keeps a *fixed teacher* but *fresh student samples* every step.

---

## 6. Myth killed: "more teacher data fixes imitation learning"

The intuitive response to a drifting student is "collect more/better teacher demonstrations." DAgger shows why that fails: behavioral cloning is O(ε·T²) *no matter how much expert data you have*, because the error source is the **mismatch between the training and test state distributions**, not a shortage of labels. More teacher data makes ε smaller by a constant; it does not change the T² scaling. Only moving the data source on-policy changes the exponent. Exposure bias is a distribution-shift problem, full stop.

---

## 7. Applied: a 50-turn sales call is the worst case for off-policy

Now the placement from [[ch-02]] gets its teeth. The boson seller is trained off-policy on generated transcripts, and a TMR sales call runs **20–50 turns** — a long horizon T. Plug that into §2: the expected damage from off-policy training scales with T², and 50 turns is squarely in the regime where the quadratic term dominates.

Concretely: the deployed seller opens a call on-distribution (early turns resemble transcripts), but the first genuinely novel moment — an objection phrased unusually, a barge-in mid-pitch, a compaction that drops context — puts it in a state no transcript covered. Off-policy, it was never graded there, so it errs; that error shapes the customer's next turn, which is now even further off-distribution; and the call spirals. This is not hypothetical for an LLM agent: it is the "confidently wrong under distribution shift" failure the learner already saw dominate in the agent-benchmark course. The structural cure is to grade the seller **on its own sampled turns** — which is on-policy distillation, the subject of [[ch-04]].

---

## Where This Goes

Chapter 4 builds the mechanism this chapter has been pointing at: the student samples, the teacher grades every token by reverse KL, and the whole thing drops into an existing RL loop as a dense per-token reward. It unifies the Thinking Machines recipe, GKD's λ/β generalization, and MiniLLM's reverse-KL objective into one procedure — the on-policy, dense, mode-seeking corner of [[ch-01]]'s cube.

## Additional Reading

- Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger, AISTATS 2011) — https://arxiv.org/abs/1011.0686 ([[ross-dagger-exposure-bias]])
- Bengio et al., "Scheduled Sampling for Sequence Prediction with RNNs" (2015) — https://arxiv.org/abs/1506.03099 (an earlier, partial exposure-bias mitigation)
- Thinking Machines, "On-Policy Distillation" — https://thinkingmachines.ai/blog/on-policy-distillation/ ([[tm-on-policy-distillation]])
- nrehiew, "SFT, RL, and On-Policy Distillation Through a Distributional Lens" — https://nrehiew.github.io/blog/sft_rl_opd/ ([[nrehiew-sft-rl-opd]])
