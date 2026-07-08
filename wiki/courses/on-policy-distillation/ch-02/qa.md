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
