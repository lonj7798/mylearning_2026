<!-- chapter: ch-01
     track: foundations
     kind: content
     title: Post-Training as Distribution Matching — SFT, RL, and Distillation on One Map
     deps:
     sources: [[nrehiew-sft-rl-opd]], [[tm-on-policy-distillation]], [[hinton-knowledge-distillation]], [[agarwal-gkd]], [[ross-dagger-exposure-bias]], [[insights]]
-->

# Chapter 01 — Post-Training as Distribution Matching: SFT, RL, and Distillation on One Map

> **Core insight.** SFT, RL, and distillation are not three unrelated tools — they are one family of objectives that reshape a language model's *sequence distribution*, differing on exactly three axes: **where the training data comes from** (off-policy = teacher/dataset vs on-policy = the student itself), **how dense the learning signal is** (sparse O(1) bits per episode vs dense O(N) bits per token), and **which divergence geometry** you minimize (forward KL = mode-covering vs reverse KL = mode-seeking). Every method you will meet in this course is one corner of that cube, and on-policy distillation is the corner that combines "the density of distillation, the unbiasedness of RL, and the on-policy property of both" ([[nrehiew-sft-rl-opd]]).

> **Guideline.** Before reaching for a post-training method, locate it on the three axes. If you want to *add capability without forgetting*, prefer on-policy data (RL or on-policy distillation) over fixed-dataset SFT; if you *also* have a stronger teacher available, prefer the dense per-token teacher signal of distillation over RL's sparse scalar reward. Name what each choice buys and what it costs — there is no free method, only a bet about which failure mode you can afford.

---

## 1. Why a "map" and not a list

Most treatments of post-training read as a catalogue: here is SFT, here is RLHF/PPO, here is DPO, here is distillation. That framing hides the thing that actually matters — that these methods are all doing the *same job* (moving `π_θ`, the model's distribution over sequences, toward some target) and differ only in *how*. nrehiew's blog states the unifying move directly: the methods are "distribution-matching objectives" that differ in "target distribution definition and optimization geometry" ([[nrehiew-sft-rl-opd]]).

Once you see them as one family, the questions that decide a project stop being "which named algorithm?" and become the three that this chapter installs:

1. **Data source** — is the model trained on data produced by *something else* (a teacher, a fixed dataset), or on data *it produced itself*?
2. **Signal density** — how many bits of learning signal does each episode deliver?
3. **Divergence geometry** — when the target and the model disagree, which direction does the loss push?

> **Interactive companion:** [`figures/distribution-cube.html`](figures/distribution-cube.html) — a three-axis explorer. Toggle each axis and watch where SFT, off-policy KD, RL, and on-policy distillation land, with the one-line reason each occupies its corner. Come back to it after each section below.

This is the course's spine. The rest of the course fills in the corners; the capstone ([[insights]], ch-07) uses the map to decide what to do with the learner's own SFT pipeline.

---

## 2. Axis 1 — data source: off-policy vs on-policy

The single most consequential axis. **Off-policy** means the training sequences come from somewhere other than the model being trained — a human-written dataset (SFT), or a teacher's generations (classical distillation). **On-policy** means the training sequences are *sampled from the current student itself*, then scored.

Thinking Machines states the difference and why it matters in one pair of sentences ([[tm-on-policy-distillation]]):

> "The drawback of off-policy training is that the student learns in contexts frequented by the teachers, not ones the student itself will often find itself in. This can cause compounding error: if the student makes an early mistake that the teacher never makes, it finds itself diverging ever farther from the states it observed in training."

> "The strength of on-policy training is that by training on samples from itself, the student learns to avoid mistakes in a more direct way."

This is not a small optimization detail — it is the difference between being graded on a curriculum you will never face and being graded on the situations you actually reach. Chapter 3 makes it rigorous through the classical exposure-bias result ([[ross-dagger-exposure-bias]]): off-policy imitation compounds error quadratically in the sequence length, on-policy data collection makes it linear. For now, hold the one-line version: **on-policy data is graded where the student actually goes.**

---

## 3. Axis 2 — signal density: O(1) vs O(N) bits

The second axis is *how much* the training signal tells the model per episode. RL delivers a **sparse** signal — a scalar reward at (usually) the end of a rollout. Distillation delivers a **dense** signal — a full target distribution at *every token*. Thinking Machines quantifies the gap ([[tm-on-policy-distillation]]):

> "RL provides very sparse feedback, teaching a fixed number of bits per training episode regardless of the number of tokens used."

> "Distillation teaches O(N) bits per episode, where N is the number of tokens."

The ancestry of "dense = more bits" runs back to Hinton's original knowledge-distillation argument: a teacher's *soft* probability vector "provide[s] much more information per training case than hard targets" ([[hinton-knowledge-distillation]]). A one-hot label says "the answer is 7"; a softened distribution says "it is a 7, but it looks a little like a 1 and nothing like an 8" — structure the hard label throws away. Per-token teacher grading is that idea applied to every position in a sequence, which is why distillation is far more sample- and compute-efficient than reward-at-the-end RL.

Density is the axis that will explain the course's headline number (ch-05): on-policy distillation reaching a *higher* benchmark score than RL at roughly one-tenth the compute ([[qwen3-strong-to-weak-distillation]]). More bits per episode, fewer episodes needed.

---

## 4. Axis 3 — divergence geometry: forward vs reverse KL

The third axis is which way the loss pushes when the model and the target disagree. Two divergences dominate, and the choice is not cosmetic.

**Forward KL — `D_KL(p ‖ q_θ)` — is what SFT minimizes.** nrehiew: "SFT via Cross Entropy on a fixed dataset is equivalent to minimizing forward KL divergence, up to constants," with

```
D_KL(p ‖ q_θ) = Σ_x p(x) · log[ p(x) / q_θ(x) ]
```

Forward KL is **mode-covering** (zero-avoiding): it penalizes the student for putting *low* probability anywhere the target `p` has mass, so the student spreads itself to cover *all* of the target's modes. For post-training this has a specific danger — "The mode-covering behavior of forward KL could thus lead to sacrificing previous modes (representing pre-existing capabilities) to learn the new task" ([[nrehiew-sft-rl-opd]]). SFT has "little regard for the starting policy": no built-in brake against forgetting.

**Reverse KL — `D_KL(q_θ ‖ p)` — is what RL and on-policy distillation minimize.** It is **mode-seeking**: it penalizes the student for putting mass where the target does *not*, so the student concentrates on the target's high-probability modes rather than hedging across everything. MiniLLM adopted exactly this swap for LLM distillation "to prevent the student model from overestimating the low-probability regions of the teacher distribution" ([[gu-minillm-reverse-kd]]). Thinking Machines notes the further virtue for a *fixed* teacher: reverse KL is "'unhackable' in the sense that low KL always corresponds to a high probability of desirable behavior from the teacher model's point of view" ([[tm-on-policy-distillation]]).

The one-line contrast to carry forward: **forward KL hedges to cover the teacher; reverse KL commits to the teacher's best modes.** (See the figure's "geometry" toggle for the mode-covering-vs-seeking picture.)

---

## 5. The four corners

With three axes you can place every method. The ones this course cares about:

| Method | Data source | Density | Geometry | One-line |
|---|---|---|---|---|
| **SFT** | off-policy (dataset) | dense (per token) | forward KL | pull to a fixed target, no forgetting brake |
| **Off-policy KD** (Hinton, Kim & Rush) | off-policy (teacher) | dense (per token) | forward KL | copy the teacher's outputs in the teacher's contexts |
| **RL** | on-policy | **sparse** (O(1)) | reverse KL | "learns the nearest task-solving policy," but 1 bit/episode |
| **On-policy distillation** | on-policy | dense (O(N)) | reverse KL | grade the student's own tokens against a teacher |

nrehiew's summary of RL's position is worth keeping: with no external target, "RL learns the nearest task-solving policy," and "because we are training on on-policy samples generated by the current policy, this optimal distribution is the closest among all optimal policies to our current policy." RL has the on-policy property but pays for it with a starvation-level signal.

On-policy distillation is the corner that takes the good column from each: the **on-policy data source** of RL (graded where the student goes), the **dense per-token signal** of distillation (O(N) bits), and the **mode-seeking reverse-KL** geometry (commit to the teacher, don't hedge, don't forget). nrehiew names the target outright:

> "You want something with the density of distillation, the unbiasedness of RL, and the on-policy property of both."

> "The thing that lets you push capability up without blowing through your KL budget is on-policy training."

---

## 6. Myth killed: "distillation just means training a small model to copy a big one"

The folk definition of distillation — *point a small model at a big model's outputs and imitate* — describes only the **off-policy** corner (Hinton soft targets, Kim & Rush sequence-KD). It quietly assumes the student trains on the *teacher's* data. On-policy distillation breaks that assumption: the data comes from the **student**, and the teacher only *grades* it. Agarwal et al name the failure of the folk version precisely — "current KD methods for auto-regressive sequence models suffer from distribution mismatch between output sequences seen during training and those generated by the student during inference" — and fix it by "train[ing] the student on its self-generated output sequences by leveraging feedback from the teacher" ([[agarwal-gkd]]).

So "distillation" is not one thing. It is a *density* choice (dense teacher signal) that can be paired with *either* data source. The whole course lives in the gap between the two pairings.

---

## 7. Applied: where the boson SFT pipeline sits on the map

The capstone target — `boson-agent-synthetic-data-dev` — generates multi-turn Korean insurance tele-sales (TMR) conversations and fine-tunes a *seller* student (the `Qwen3.6-27B-Lina-chk-*` checkpoints) on those generated transcripts. Locate that on the map:

- **Data source: off-policy.** The seller student is trained on transcripts produced by the *generation pipeline* (a stage-puppeted seller + an 11-model customer rotation), not on sequences it sampled itself and got graded on. It is textbook off-policy sequence-level distillation ([[kim-rush-seqkd]]).
- **Density: dense** (ordinary next-token cross-entropy over the transcript) — so the density axis is already "good."
- **Geometry: forward KL** (cross-entropy to the transcript) — mode-covering, no forgetting brake.

The pipeline already does the *on-policy* thing on the **wrong side of the table**: the *customer simulator* samples live and on-policy, but that is the environment, not the model being trained. The *seller* — the model we actually ship — learns off-policy. Chapter 3 will argue that 20–50-turn sales dialogue is exactly the long-horizon regime where that off-policy choice compounds error, and the capstone will move the seller to the on-policy-distillation corner. Keep the placement in mind; it is the whole reason this course exists for this learner.

---

## Where This Goes

This chapter installed the map. Chapter 2 zooms into the off-policy corner — the classical distillation lineage from Hinton's soft targets to Kim & Rush's sequence-level KD — and shows precisely why "train on the teacher's outputs" is the baseline every later method improves on. Chapter 3 then names the disease of that corner (exposure bias / compounding error) and the principle that cures it (on-policy data, DAgger), setting up on-policy distillation itself in chapter 4.

## Additional Reading

- Kevin Lu & Thinking Machines Lab, "On-Policy Distillation" (2025) — https://thinkingmachines.ai/blog/on-policy-distillation/ ([[tm-on-policy-distillation]])
- nrehiew, "SFT, RL, and On-Policy Distillation Through a Distributional Lens" (2025) — https://nrehiew.github.io/blog/sft_rl_opd/ ([[nrehiew-sft-rl-opd]])
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015) — https://arxiv.org/abs/1503.02531 ([[hinton-knowledge-distillation]])
- Agarwal et al., "On-Policy Distillation of Language Models" (GKD, ICLR 2024) — https://arxiv.org/abs/2306.13649 ([[agarwal-gkd]])
- Ross, Gordon, Bagnell, "A Reduction of Imitation Learning… to No-Regret Online Learning" (DAgger, 2011) — https://arxiv.org/abs/1011.0686 ([[ross-dagger-exposure-bias]])
