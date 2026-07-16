<!-- qa: ch-02 — Off-Policy Distillation: Hinton Soft Targets -> Sequence-Level KD
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-02 Q&A — Off-Policy Distillation (Hinton -> Kim & Rush)

Clarifying questions raised while reading [[read]]. Kernels only; full worked examples live in `read.md` / the discuss transcript.

---

### Q1 — soft target vs hard target, and "soft targets carry more bits than hard labels"?

**hard target (hard label):** the one-hot ground truth — e.g. a "2" image → `[0,0,1,0,...]`. States exactly **one fact** ("it's a 2"); says nothing about how 2 relates to other classes.

**soft target:** the *teacher model's* full probability distribution over all classes (temperature-softened) — e.g. `2→.90, 3→.05, 7→.04, 8→.0001`. Carries the **relative structure over the wrong classes** = *dark knowledge*.

**"more bits per example"** (reuse ch-01 Q9: bits = information = # questions answered): a hard label answers one question ("which class?" → 2); a soft target answers a graded question for *every* class ("how 2-like? 3-like? 7-like?..."), so the student must reproduce the whole distribution shape → each example constrains it in many more dimensions. The teacher has pre-computed the input's similarity to *all* classes and hands that whole map over per case. Hinton: soft targets "provide much more information per training case than hard targets and much less variance in the gradient between training cases."

**Why it matters:** (1) denser examples → **fewer needed** (root of ch-01's "dense = O(N) bits" / distillation sample-efficiency); (2) evidence = **omitted-class transfer** (§2.3): withhold digit "3" entirely from the student's set, yet it still classifies 3s — only possible because soft targets on *other* digits smuggle in cross-class "3-ness" info a hard label would delete.

**temperature `T`:** teacher's `T=1` softmax is often too peaked (~.99 on the answer); raising `T` softens it so the small dark-knowledge probs become visible/usable. See [[figures/soft-target-temperature.html]].

**One line:** hard = one-hot answer (1 fact); soft = teacher's full class distribution (N facts = a similarity map); "more bits" = each example carries the whole relational map, not just the single label. See read.md §2, §2.1–§2.3.

---

### Q2 — is hard+soft combined to prevent the model's entropy collapse?

No — and the entropy roles are backwards. **soft target = high entropy** (spread) → that's the anti-collapse / knowledge-carrying force; **hard target = zero entropy** (one-hot) → adding it *sharpens* toward the true class (a *lower*-entropy pressure). So the hard term pushes toward peakier, not away from collapse.

**Real reason for the hard-label term = ground-truth anchor correcting teacher error.** Soft targets are all teacher-produced, so teacher mistakes get inherited; the hard label ties the student to the actual correct answer. It's weighted *small*: Hinton got best results with "considerably lower weight" on the hard-label objective — soft term = the signal body (dark knowledge, generalization), hard term = a small truth-anchoring correction.

**Mechanical note (§2.2):** soft-target gradients scale as `1/T²`, so multiply by `T²` when combining — a balancing detail for *using* both, not the *reason* for both.

**entropy collapse proper** is an RL/on-policy failure mode (policy loses diversity), covered in **ch-05**, not Hinton's motivation here.

**One line:** hard+soft isn't collapse-prevention — it's a small ground-truth correction against teacher error; the collapse-resisting (high-entropy) force is the *soft* target, while the hard term is the opposite (sharpening) pressure. See read.md §2.2.

---

### Q3 — how should temperature T be set?

Sweet-spot hyperparameter (tuned, not a formula): `T=1` → teacher too peaked, dark knowledge invisible; `T→∞` → distribution → uniform, structure destroyed. Aim between.

**Raise T when:** teacher very confident (dark knowledge hidden at T=1); many classes; student has enough capacity; small transfer set (squeeze max info/example → MNIST used `T=20`).
**Lower T when:** ⭐ **student is small** (high T makes it chase tiny, often-noisy logits it can't fit — Hinton's key caveat); few classes; teacher already well-calibrated.

**Anchors:** default start `T≈2–5`; MNIST small-transfer-set `T=20`; speech `T≈2–4`. Sweep on validation.

**Notes:** (1) hard+soft → soft grads scale `1/T²`, multiply by `T²` (Q2). (2) T is a *training* knob — the deployed student runs at `T=1` at inference. (3) high-T limit ≈ matching logits directly (MSE on logits), since softmax → ~linear in logits.

**LLM/OPD bridge (don't port T=20):** this is *classifier* distillation. In sequence models temperature splits into KD-matching temp vs sampling temp; modern OPD/GKD matches full per-token distributions (reverse KL) with temperature as one knob beside `lmbda`/`beta` — TRL `GKDConfig.temperature` ≈ 0.9, i.e. **T≈1**, not 20. See read.md §2.1, [[figures/soft-target-temperature.html]], [[hf-trl-gkd-recipe]].

**One line:** between T=1 (dark knowledge hidden) and T→∞ (flattened to noise); up for confident teacher / many classes / big student / small set, down for small student; start 2–5 and sweep — but classifier-only, LLM/OPD sits near T≈1.

---

### Q4 — so keep student T low and teacher T higher (2–5)?

No — **T is not set separately per model.** The soft-target loss *compares* teacher and student distributions, so both must be softened by the **same shared T** (else you're matching mismatched shapes). Structure:
- **① soft-target loss (body):** `CE( softmax_T(student) , softmax_T(teacher) )` — **same T on BOTH** (this is the "T=2–5?" T).
- **② hard-label term (small correction):** student at `T=1`.
- **③ inference (deployed student):** always `T=1`.

"student small → lower T" means **choose the single shared T lower** (a small student can't fit the fine dark-knowledge structure high T exposes) — not "student gets its own low T." Student capacity influences the *choice* of the one shared T; it doesn't get a different T from the teacher.

**Do:** pick one shared distillation T (start 2–5, lower if student small), apply to teacher+student in the soft term; keep the hard-label term at T=1; deploy at T=1.

**One line:** one shared T for both models in the soft-target loss (not teacher-high/student-low); T=1 only for the hard-label correction and at inference. See read.md §2.2.

---

### Q5 — can deployment use T < 1?

Yes — and for LLMs it's common. "Deploy at T=1" earlier conflated **two different temperatures**:
- **KD/distillation temperature** — training-time, shared T on teacher+student in the soft loss; a *training artifact* (don't carry a T=20 into deployment).
- **inference/sampling temperature** — the actual *deployment* knob for how the model samples logits. "T=1" just meant "revert to native scale," **not** a floor on sampling temperature.

**Deployment T<1 sharpens:** `T<1` → concentrates on high-prob tokens (more deterministic/consistent, less diverse, fewer wild errors); `T=1` native; `T>1` flatter/more diverse; `T→0` greedy/argmax.

**Classifier vs LLM:** for a classifier taking **argmax**, `argmax softmax_T(z) = argmax z` for any T>0 → inference T doesn't change the predicted class (moot; T=1 only for calibrated probs). For **LLM generation**, T genuinely changes output → T<1 is legit and common.

**boson seller:** reliability/consistency > creativity → deploy at **T<1 (~0.3–0.7) or greedy** to suppress randomness. No "deployment must be T=1" rule; tune to the task's determinism-vs-diversity need.

**One line:** deployment T<1 is fine (common for LLMs); the "T=1" was about not carrying the *distillation* T into inference, not a sampling-temp floor — classifier argmax is T-invariant, LLM generation isn't, and a reliable seller likely wants T<1.

---

### Q6 — what does "dark knowledge" mean — wrong? secret/should-not-reveal?

Neither. Hinton's term, by analogy to **dark matter**: knowledge that is *present and has effects but is invisible directly*. It is the **valuable relational information encoded in the teacher's *relative* probabilities over the *incorrect* classes.**

- **not "wrong":** it lives *on* the wrong classes, but the info itself is correct/valuable.
- **not "secret / should-not-reveal":** the opposite — you *want* to reveal it (raising temperature exists to expose it).
- **"dark" = latent/hidden:** invisible in a hard one-hot label; surfaced only by soft targets (esp. high T).

**Example ("2"):** hard `[0,0,1,0,...]` says only "it's a 2" (zero relational info). Soft `2→.90, 3→.05, 7→.04, 8→~0` encodes "a 2 resembles 3/7, not 8" = the input-space geometry the one-hot deletes.

**Proof it's real signal:** omitted-class transfer (§2.3) — withhold digit "3" entirely from the student, yet it still classifies 3s, purely from the faint "3-ness" carried in *other* digits' soft targets.

**One line:** dark knowledge = the latent, valuable relational structure in the teacher's probabilities over the *wrong* classes — hidden in hard labels, revealed by soft targets; "dark" as in dark-matter (unseen-but-present), not evil/secret. See read.md §2.1, §2.3.

---

### Q7 — what is "off-policy KD"?

**off-policy Knowledge Distillation = the classical distillation this whole chapter covers (Hinton + Kim & Rush).** Decompose: **KD** = train a student to imitate a teacher; **off-policy** (ch-01 Axis 1) = the training targets/sequences come from the **teacher** (or a fixed dataset), not from the student itself.

So off-policy KD = Hinton (student matches teacher's soft distribution on gold data) and Kim & Rush seq-KD (student trains on the teacher's beam-mode outputs) — both teacher-authored targets. Contrast **on-policy distillation** (ch-04): student generates, teacher only grades.

**One line:** off-policy KD = teacher-authored-target distillation = exactly this chapter; the "off-policy KD" column in the SFT≡distillation table is this chapter itself. See read.md §1–§4, [[ch-01]].

---

### Q8 — can off-policy distillation be turned into on-policy distillation? (learner sketch: have good SFT data → student regenerates each round → reverse KL vs "the SFT" → repeat)

Yes — that sketch **is** on-policy distillation (front-runs ch-04). Correct pieces: student regenerates each round = on-policy sampling (fresh because the student moves, ch-01 Q2); reverse KL = the geometry; repeat = iteration.

**Two sharpenings:**
1. **Teacher must be a queryable MODEL, not the SFT *dataset*.** Per-token reverse KL needs `p_teacher(·|student's own prefix)` on the student's *drifted* trajectory; a fixed (input,output) dataset can't score arbitrary student prefixes. (This is the density axis: teacher gives a per-token distribution → O(N).)
2. **Teacher must be STRONGER than the student.** ⭐ If "the SFT" is a bigger/frontier/few-shot-lifted model → valid OPD. If it's the student's own SFT-init (same model) → reverse-KL-to-self = **reference-KL regularization, NOT distillation**, and it does **not** fix exposure bias: that SFT model is equally *blind* in the drift-states the student reaches (never trained there either), so it has nothing better to transfer. Rule: reverse-KL-to-self ≠ distillation.

**Direction:** minimize `KL(π_student ‖ π_teacher)` evaluated on student samples.

**Callback:** the learner's banked few-shot-teacher → no-few-shot-student idea is a valid instance of Sharpening 2 — few-shot makes the *same* base a stronger teacher (context distillation).

**One line:** yes = OPD; but the teacher must be a queryable model (not a dataset) AND genuinely stronger than the student (self-teacher = mere reference-KL, doesn't cure exposure bias). See [[ch-01]] §5, read.md §4, forward to ch-04.

---

### Q9 — so you need the whole distribution per token, not just the output token?

Correct — that's the density mechanism. `KL(π_student(·|s) ‖ π_teacher(·|s))` needs the teacher's **full vocab distribution** at each position (on the student's own prefix `s`), not just the top-1 output token.

**Precision:** you need the **distribution vector itself**, not its entropy scalar — KL uses the whole vector; entropy is just a summary; dark knowledge lives in the vector's *shape*, not in one scalar.

**This splits seq-KD from OPD:** Kim&Rush seq-KD keeps only the **mode token (hard, one-hot)** → per-token but hard, no dark knowledge; word-level KD / OPD use the **full per-token distribution (soft)** → where the O(N) dense bits live. "See the whole distribution, not just the output token" = use soft per-token targets = why OPD is denser than seq-KD.

**Practical wall (boson / ch-06):** getting the full vocab distribution needs **white-box teacher access** (local model → all logits). **API teachers (Claude) give only top-k logprobs**, not the full distribution → for boson-with-Claude-teacher (capstone Fork 1) full reverse-KL isn't available; need top-k approximation or GOLD (cross-tokenizer). See [[hf-trl-gkd-recipe]].

**One line:** reverse-KL-per-token = teacher's full soft distribution per position on the student's prefix (not the mode token, not an entropy scalar); the practical constraint is white-box vs API (top-k-only) teacher access. See read.md §2.1, §3.1–§3.2.

---

### Q10 — if we limit the signal to output tokens only, does it become O(1)? (+ cross-family: different tokenizer → "harder than T=1"?)

**No — still O(N).** Two *separate* reductions must not be conflated:

| reduction | from → to | what shrinks | example |
|---|---|---|---|
| **soft→hard** (bits/token) | full dist/token → one-hot/token | info *per* token | Hinton soft → seq-KD mode |
| **O(N)→O(1)** (density) | target every token → one scalar/episode | # signals (N-scaling) | distillation → RL reward |

`O(1)` vs `O(N)` is about *how many* signals scale with length N, **not** bits-per-token. "Output tokens only" = a hard target at every token = **N targets = O(N)** (= seq-KD). O(1) needs collapsing the whole sequence to **one scalar** (a reward = RL, not distillation). Output-only is the *first* row (soft→hard), staying O(N).

**Cross-family (Part 2/3):** the real blocker with a different model family is **tokenizer mismatch**, not entropy. Different families → different vocab/subword boundaries → teacher's and student's per-position distributions live on **different supports**, so per-token reverse KL **can't be computed/aligned**. This *forces* sequence-level: take the teacher's output **text**, **re-tokenize with the student's tokenizer**, train CE (hard). So the learner's conclusion (cross-family → pushed to hard/output-level) is right, but the cause is **vocab misalignment**, not "entropy got harder" (the one-hot output is the T→0 extreme, yes, but that's the destination, not the cause). This is the **cross-tokenizer / GOLD** problem (ch-06); still O(N).

**boson tie (capstone Fork 1):** big teacher (family A) → small student (family B) is realistic but cross-tokenizer → needs **GOLD** or sequence-level — exactly the constraint when boson's teacher = Claude.

**One line:** output-only stays O(N) (soft→hard ≠ density reduction; O(1) = whole-seq scalar = RL); the different-family blocker is tokenizer misalignment (→ GOLD / sequence-level), not entropy. See read.md §3, [[hf-trl-gkd-recipe]].
