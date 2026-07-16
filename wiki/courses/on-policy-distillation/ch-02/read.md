<!-- chapter: ch-02
     track: off-policy
     kind: content
     title: Off-Policy Distillation — From Hinton Soft Targets to Sequence-Level KD
     deps: [[ch-01]]
     sources: [[hinton-knowledge-distillation]], [[kim-rush-seqkd]], [[nrehiew-sft-rl-opd]], [[tm-on-policy-distillation]], [[agarwal-gkd]]
-->

# Chapter 02 — Off-Policy Distillation: From Hinton Soft Targets to Sequence-Level KD

> **Core insight.** Classical knowledge distillation lives entirely in the *off-policy* corner of [[ch-01]]'s map: a student is trained on a **fixed set of teacher-produced targets** — softened class probabilities (Hinton) or teacher-generated output sequences (Kim & Rush). The genius of this line is *density* — soft targets carry far more bits per example than hard labels — but its defining limitation is the data source: the student only ever learns in contexts the *teacher* visits, never the ones it will itself reach at inference. That single limitation is the disease the rest of the course cures.

> **Guideline.** When you distill off-policy, extract as much signal as the teacher will give — use soft targets / full distributions, not hard labels, and prefer teacher-generated *sequences* over per-token labels for generation tasks. But recognize what you have bought: a mode-covering (forward-KL) imitation of the teacher's own trajectories, which is exactly SFT wearing a different name — and which will compound error the moment the student steps off the teacher's path.

---

## 1. The corner we start in

Chapter 1 placed four methods on three axes. This chapter zooms into the corner with **off-policy** data, **dense** signal, **forward-KL** geometry — the home of classical knowledge distillation and, as §5 argues, of ordinary SFT. We start here for a reason: it is the baseline every later method is measured against, and understanding *precisely* why it fails is what makes on-policy distillation feel inevitable rather than clever.

Two papers define the corner. Hinton et al. (2015) invented distillation for classifiers; Kim & Rush (2016) lifted it to sequences. Both are off-policy. Neither closes the loop between the student's outputs and its training data.

---

## 2. Hinton: knowledge is a mapping, and soft targets carry it

Hinton, Vinyals & Dean reframe what "knowledge" even is ([[hinton-knowledge-distillation]]):

> "A more abstract view of the knowledge, that frees it from any particular instantiation, is that it is a learned mapping from input vectors to output vectors."

Because knowledge is the *mapping*, a smaller, differently-shaped student can inherit it from a "cumbersome" teacher (often an ensemble) — if you can transfer the mapping. The transfer medium is the teacher's **softened output distribution**.

### 2.1 Temperature and "dark knowledge"

The teacher's logits `z_i` are turned into probabilities through a temperature-scaled softmax:

```
q_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

At `T = 1` this is the ordinary softmax; raising `T` "produces a softer probability distribution over classes." The soft distribution exposes which *wrong* classes the teacher finds plausible — a handwritten "2" that carries a little probability on "3" and "7" and essentially none on "8". That relative structure over the wrong answers is the **dark knowledge**, and it is invisible in a one-hot label.

Why it helps, verbatim:

> "When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases."

More information per example + lower gradient variance ⇒ the student can be trained on far fewer cases and still generalize like the teacher. This is the concrete root of [[ch-01]]'s "dense = O(N) bits" axis and of Thinking Machines' O(N)-bits argument ([[tm-on-policy-distillation]]).

> **Interactive companion:** [`figures/soft-target-temperature.html`](figures/soft-target-temperature.html) — drag the temperature slider on a teacher's logits for a "2" and watch the dark knowledge on "3"/"7" emerge as `T` rises, with the per-example information (entropy, in bits) updating live. It is the picture behind "soft targets carry more bits than hard labels."

### 2.2 The distillation loss

Hinton trains the student on "a weighted average of two different objective functions":

- **Soft-target cross-entropy** — CE between student and teacher *both softened at temperature T*.
- **Hard-label cross-entropy** — CE against the true labels at `T = 1`.

With a crucial scaling note:

> "Since the magnitudes of the gradients produced by the soft targets scale as 1/T² it is important to multiply them by T² when using both hard and soft targets."

The soft term carries most of the signal; the hard term is a small correction.

### 2.3 The evidence it works (MNIST)

The numbers are worth holding because they show soft targets acting as a *regularizer* ([[hinton-knowledge-distillation]]):

- A large net: **67** test errors. A plain small net (two 800-unit hidden layers, no regularization): **146** errors.
- The *same* small net, regularized **only** by matching the teacher's soft targets at `T = 20`: **74** errors — most of the gap closed with no other regularization.
- **Omitted-class transfer:** even when the digit **3** is entirely withheld from the student's transfer set, the student still classifies 3s — "the distilled model only makes 206 test errors of which 133 are on the 1010 threes," and after a bias fix, 14 on 3s. The student learns an unseen class purely from its *similarity structure* to seen classes. Dark knowledge is real signal.

---

## 3. Kim & Rush: distillation grows up to sequences

Hinton's setup is a single softmax over a fixed class set. Language is autoregressive — a distribution over *sequences*, exponentially many of them. Kim & Rush (2016) make the jump ([[kim-rush-seqkd]]).

### 3.1 Word-level KD (the direct lift)

Apply Hinton's idea at every decoding step: match the teacher's per-word distribution `q(t_j | s, t_{<j})` with cross-entropy, conditioned on the gold prefix. Dense, but still anchored to ground-truth history — "Word-level knowledge distillation allows transfer of these local word distributions."

### 3.2 Sequence-level KD (the key move)

The true target is the whole-sequence distribution, which is intractable to sum over:

```
L_SEQ-KD = − Σ_{t∈T} q(t | s) · log p(t | s)     (sum over exponentially many sequences)
```

Kim & Rush approximate the teacher distribution by its **mode**, found with beam search:

```
q(t | s) ≈ 1{ t = argmax_t q(t | s) }   ⇒   L_SEQ-KD ≈ − log p(ŷ | s)
```

"where ŷ is now the output from running beam search with the teacher model." In plain terms: **run the teacher, take its best output, train the student on that with ordinary cross-entropy.** Three steps — "(1) train a teacher model, (2) run beam search over the training set with this model, (3) train the student network with cross-entropy on this new dataset."

Why approximating a whole distribution by one sequence is defensible — the mass is concentrated, and training piles on more: with beam size 1 the teacher's top sequence accounts for only ~1.3% (De→En) / 2.3% (Th→En) of the mass, but *after* Seq-KD the student puts **16.9%** on that mode versus **0.9%** for the baseline. The student sharpens onto the teacher's mode.

### 3.3 Results and the surprise

On English→German (greedy / beam BLEU), the 2×500 student ([[kim-rush-seqkd]]):
- baseline **14.7 / 17.6**, Word-KD **15.4 / 17.7**, **Seq-KD 18.9 / 19.0**.
- Greedy Seq-KD (18.9) **beats the beam-search baseline** (17.6) and rivals the teacher's beam (19.5) — and the best student runs **10× faster** than the teacher, "somewhat surprisingly, seem[ing] to eliminate the need for beam search."

The advance over Hinton is subtle but decisive: the student trains on **teacher-generated sequences**, an early recognition that *which sequences you train on* matters, not just what per-token labels you attach.

---

## 4. The through-line: it is all still off-policy

Here is the limitation that unites §2 and §3 and motivates the whole course. In both word-level and sequence-level KD, the training sequences come from the **teacher** (its soft labels on gold data, or its beam-mode generations), fixed once before the student trains. The student:

- is graded only in **teacher-visited contexts**, never the states its own generations reach;
- is pulled by **forward KL** (cross-entropy to a fixed target), which is *mode-covering* — it hedges to cover the teacher and has "little regard for the starting policy," i.e. no brake against forgetting ([[nrehiew-sft-rl-opd]]).

Kim & Rush's own excerpt states the gap plainly: the training sequences are "the teacher's mode, fixed once before student training begins. The student never sees or corrects its *own* generations." That is **exposure bias**, and it is the subject of [[ch-03]]. The fix — sampling from the *student* and grading those trajectories with the teacher — is exactly what GKD ([[agarwal-gkd]]) and on-policy distillation ([[tm-on-policy-distillation]]) add. Read this whole chapter as the "before."

---

## 5. Myth killed: "SFT and distillation are different things"

They are the same corner. Ordinary SFT on a human dataset and off-policy distillation on teacher-generated sequences are both: off-policy data, dense per-token cross-entropy, forward-KL geometry. The only difference is *who wrote the targets* (a human vs a teacher model). nrehiew's reframe — SFT ≡ forward-KL minimization on a fixed dataset ([[nrehiew-sft-rl-opd]]) — is precisely why "do SFT on teacher outputs" and "do sequence-level KD" name the *same operation*. Once you see that, the boundary between "supervised fine-tuning" and "distillation" dissolves into a single question: **on-policy or off-policy?**

---

## 6. Applied: the boson seller is off-policy sequence-level KD

The `Qwen3.6-27B-Lina-chk-*` seller checkpoints are trained by ordinary SFT on the transcripts the generation pipeline produces. By §5, that is **off-policy sequence-level knowledge distillation**: the "teacher" is the transcript-generating setup (a stage-puppeted seller + an 11-model customer rotation), and the student imitates those fixed sequences with forward-KL cross-entropy.

That places the shipped seller squarely in this chapter's corner — with this chapter's limitation. Every transcript is a teacher-visited trajectory; the moment the *deployed* seller says something a shade different and the customer reacts in a way no transcript covered, the model is off the map it was trained on. In a one-shot classification task that barely matters. In a **20–50-turn** sales call it is the dominant failure mode — which is exactly the argument [[ch-03]] makes rigorous. Hold the placement: the whole capstone is about moving this checkpoint out of the off-policy corner.

---

## Where This Goes

Chapter 3 names the disease of this corner precisely — **exposure bias / compounding error** — via the classical DAgger result ([[ross-dagger-exposure-bias]]): off-policy imitation accrues error that grows quadratically in the horizon, while on-policy data collection makes it linear. That is the "why" that turns the off-policy baseline of this chapter into the on-policy methods of chapters 4–6.

## Additional Reading

- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015) — https://arxiv.org/abs/1503.02531 ([[hinton-knowledge-distillation]])
- Kim & Rush, "Sequence-Level Knowledge Distillation" (EMNLP 2016) — https://arxiv.org/abs/1606.07947 ([[kim-rush-seqkd]])
- Gou et al., "Knowledge Distillation: A Survey" (IJCV 2021) — https://arxiv.org/abs/2006.05525 (broader map of the off-policy KD family)
- nrehiew, "SFT, RL, and On-Policy Distillation Through a Distributional Lens" — https://nrehiew.github.io/blog/sft_rl_opd/ ([[nrehiew-sft-rl-opd]])
