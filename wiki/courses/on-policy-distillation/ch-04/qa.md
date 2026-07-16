<!-- qa: ch-04 — On-Policy Distillation: the mechanism (per-token reverse KL)
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-04 Q&A — On-Policy Distillation (mechanism)

Clarifying questions raised while reading [[read]]. Kernels only; full derivations in `read.md` / the discuss transcript.

---

### Q1 — break down ch-04's formulas piece by piece

**① §2 per-token reverse KL (the objective):** `KL(π_θ ‖ π_teacher) = E_{x~π_θ}[ log π_θ(x_{t+1}|x_{1..t}) − log π_teacher(x_{t+1}|x_{1..t}) ]`

| symbol | meaning |
|---|---|
| π_θ / π_teacher | student policy (trained, weights θ) / teacher policy (fixed) |
| KL(π_θ ‖ π_teacher) | **reverse** KL — student is the FIRST arg ⇒ student-weighted ⇒ mode-seeking (ch-01 Q7) |
| E_{x~π_θ} | expectation with x **sampled from the student** ⇒ the **on-policy** part (needs student samples) |
| x_{1..t} | prefix = state; since x~π_θ it's the **student's own** prefix ⇒ grade-in-place (ch-03 Q5) |
| log π_θ − log π_teacher | = log(π_θ/π_teacher), the log-ratio; its expectation under the student = reverse KL |

Full per-position form (needs teacher's **full distribution**, ch-02 Q9): `KL_t = Σ_v π_θ(v|x_{1..t})·[log π_θ(v|·) − log π_teacher(v|·)]`, summed over the student-trajectory positions t.

**② §3 advantage = −reverse KL (bridge into the RL loop):** `A_t = −KL_t`. Matched → KL_t≈0 → A_t≈0 (no push); diverged → A_t very negative (strong fix); fixed point student=teacher → KL=0. Plug A_t into the existing **RL importance-sampling loss** → "not a new trainer, just RL with reward = dense per-token −KL" (RL scalar O(1) → OPD per-token O(N)).

**③ §4 GKD (general family, two knobs):** `L_GKD(θ) = (1−λ)·E_{(x,y)~data}[D(p_T‖p_S^θ)] + λ·E_{x∼X, y~p_S(·|x)}[D(p_T‖p_S^θ)]`

| symbol | meaning |
|---|---|
| λ | **on-policy fraction** (0=off-policy dataset term, 1=on-policy student-sampled term); ch-03 Q8 schedule knob |
| (1−λ) term | off-policy: (x,y) from a fixed **dataset** = supervised KD (ch-02) |
| λ term | on-policy: **y ~ p_S** = target sampled from the student |
| D(p_T‖p_S^θ) | teacher↔student divergence = generalized **JSD**, with **β** knob: β→0 forward KL (mode-covering), β→1 reverse KL (mode-seeking) |

Special cases: (λ=0, β→0) = SFT/supervised KD; (λ=1, β→1) = **on-policy distillation = formula ①**; in-between λ = the SFT-warmup→on-policy anneal (Q8). So **GKD = the two-knob general form; OPD = its λ=1·reverse corner.**

**One line:** ① reverse KL = student-weighted (mode-seeking) × student-sampled (on-policy) × teacher-scored-on-student-prefix (grade-in-place), full-distribution per token; ② its −KL is the per-token advantage in the RL loop; ③ GKD's (λ,β) contain SFT/off-policy-KD/OPD, and ① is GKD's λ=1·reverse corner. See read.md §2–§4, [[ch-01]] Q7, [[figures/gkd-knobs]] (ch-06).

---

### Q2 — what is GKD?

**GKD = Generalized Knowledge Distillation** (Agarwal et al., ICLR 2024, [[agarwal-gkd]]). Not a new method — the **umbrella objective** (formula ③) that contains SFT / off-policy KD / on-policy distillation as **special cases of two knobs**: **λ** = on-policy fraction (ch-01 data-source axis), **β** = forward↔reverse KL (ch-01 geometry axis). It makes ch-01's cube axes into tunable knobs.

Key empirical finding: **a little on-policy data beats a lot of off-policy data** — "on-policy GKD on the 5% subsampled dataset outperforms supervised KD with the entire dataset" (the ch-03 exposure-bias argument, confirmed); multiplicative gains ~2.1× XSum / 1.7× WMT / 1.9× GSM8K. Implemented as TRL `GKDTrainer`/`GKDConfig` (`lmbda`, `beta`) — ch-06.

**One line:** GKD = the two-knob (λ on-policy fraction, β forward↔reverse KL) general framework whose corners are SFT/off-policy-KD/OPD; headline result: little on-policy data > lots of off-policy data. See read.md §4, [[agarwal-gkd]].

---

### Q3 — what is per-token clipping? are unimportant tokens like "wait" excluded from the KL?

No — not excluded from the KL, and the premise is inverted: style tokens ("wait", "let") have **high** KL (that's where student/teacher disagree most); task-critical tokens ("12","84") have low KL (already matched).

**per-token clipping = a PPO-style cap on each token's *update*, not a KL exclusion.** Since OPD reuses the RL importance-sampling loss (§3), each token's importance ratio `r_t = π_θ/π_θ_old` is clipped to `[1−ε, 1+ε]` so no single token moves the policy too far in one step. It **caps** (bounds), it doesn't **remove** — high-KL tokens still contribute, just with bounded per-step influence.

**Why it ties to style tokens:** high KL → large |advantage| (`−KL`) → those tokens would **dominate the gradient** (student chases the teacher's stylistic filler) and cause large, unstable updates (→ entropy collapse, ch-05). Clipping caps their per-step influence so a few high-KL style tokens don't blow up training or drown out the well-matched task tokens. It's a **stability** mechanism.

**Learner intuition vs mechanism:** "don't let high-KL style tokens dominate" = right direction ✓; mechanism = cap the per-token *update*, not drop it from the KL.

**One line:** per-token clipping bounds each token's PPO-style update (importance-ratio×advantage), not a KL exclusion; needed because high-KL style tokens would otherwise dominate/destabilize the gradient — a stability device (ch-05 entropy collapse). See read.md §2, §3, ch-05.

---

### Q4 — how is the expectation E_{x~π_θ}[...] in §2 actually computed?

**Can't be computed exactly** (a sequence x ranges over V^L possibilities — ch-02 Q10) → estimate by **Monte Carlo**: sample N student rollouts and average, `E_{x~π_θ}[f] ≈ (1/N)Σᵢ f(x⁽ⁱ⁾)`. Sampling from the student **is** the on-policy rollout (§3 step 2) — "compute the expectation" = "roll the student out and average."

**Two levels:**
- **trajectory:** sample `x ~ π_θ` (a batch of student rollouts) for the outer average.
- **per-token, at each position t — two options:**
  - **(a) single-sample (the §2 formula form):** use the sampled token only: `log π_θ(x_{t+1}|x_{1..t}) − log π_teacher(x_{t+1}|x_{1..t})` — needs the teacher's logprob **only on the sampled token** (TM recipe: `compute_logprobs` on the sampled trajectory, §3 step 3); cheap, higher variance; averaging over samples → the true KL.
  - **(b) full-distribution (exact, ch-02 Q9):** `Σ_v π_θ(v)·[log π_θ(v) − log π_teacher(v)]` — needs the teacher's **full** distribution (white-box); lower variance.

**Worked (rollout "The answer is 12"):** at "wait": `log .4 − log .1 = −0.92−(−2.30)=+1.38` (style, high KL); at "12": `log .85 − log .82 = +0.04` (task, low KL); advantage = −KL. Sum over positions t, average over the rollout batch.

**Need in practice:** student logprobs (free, its own forward pass) + teacher logprobs (query, §3 step 3 — sampled token for (a), full dist for (b)). Same shape as RL, which also estimates its expectation from sampled-action logprobs.

**One line:** estimate E by Monte-Carlo-sampling student rollouts and averaging; per position use either the sampled token's `log π_θ − log π_teacher` (single-sample, TM recipe, teacher-logprob-on-token) or the full-vocab sum (exact, full teacher dist). See read.md §2–§3, [[ch-02]] qa Q9–Q10.

---

### Q5 — do you run it ~10 times to get E? and isn't E already computable from the logprob distribution?

Split the expectation into **two** — one is directly computable, one is irreducibly sampled:

- **(A) inner — per-token KL, given a prefix:** `KL_t = Σ_v π_θ(v|prefix)·[log π_θ(v) − log π_teacher(v)]`. **Directly computable from the distributions, exact, NO sampling** (you already hold π_θ and π_teacher at that position). ← the learner's Q2 intuition is right *here* (= option (b) full-distribution).
- **(B) outer — which trajectories/states the student visits:** `E_{x~π_θ}[...]`. **Cannot be read off a distribution — must sample.** The prefix `x_{1..t}` depends on the student's sampled path; averaging over *all* prefixes the student would produce needs all V^L paths (exponential) → sample instead. "~10 times" = this outer batch of rollouts (real batches bigger; fresh batch each update; more → lower variance).

**Why (B) is irreducible:** the outer expectation IS the **state-visitation distribution** (ch-01 Q2) — what states the student actually reaches — which is path-dependent and only accessible by rolling out. That's the whole point of on-policy, so sampling never disappears.

**Access:** white-box teacher → inner exact (b) + outer sampled; API/top-k teacher → inner also single-sampled (a) (ch-02 Q10).

**One line:** inner per-token KL (prefix given) = exact from the distributions, no sampling (Q2 intuition ✓); outer trajectory/state expectation = must sample (exponential paths = state-visitation = on-policy's essence); "10 times" is the outer rollout batch. See read.md §2–§3, [[ch-01]] qa Q2.

---

### Q6 — if "할인 가능합니다" appears 5/10 times, what is its E?

Type error: **there is no "E for one trajectory."** E is a *single* number averaging over all trajectories.

- **5/10 gives the trajectory's PROBABILITY**, not E: `π_θ("할인 가능합니다") ≈ 0.5` — the outer sampling weight (how often a path appears).
- **E needs each trajectory's KL VALUE** `f(x)` (its per-token KL sum, the inner computation) and averages them by probability: `E = Σ_x π_θ(x)·f(x)`.

Worked (f = example values): (0.5·2.0)+(0.2·5.0)+(0.3·3.0) = 1.0+1.0+0.9 = **E ≈ 2.9** — one number, the whole-batch average, not per-sentence.

Roles: **frequency (5/10) → weight (0.5)** from outer sampling; **f(x) → value** from inner computation; **E = Σ weight·value**. A trajectory's *contribution* = weight×value (e.g. 0.5×2.0=1.0) — needs f, not just its count.

**One line:** 5/10 = that path's probability (0.5), not E; E = one number = probability-weighted average of every trajectory's per-token-KL value f(x); frequency=weight, f=value, E=Σ weight·value. See read.md §2, [[ch-01]] qa Q9.

---

### Q7 — what is MiniLLM?

**MiniLLM = Gu et al. (ICLR 2024)** ([[gu-minillm-reverse-kd]]) — the **reverse-KL-derivation** leg of OPD's three-literature story (TM = recipe, GKD = general λ/β family, MiniLLM = reverse-KL for LLMs + how to optimize it).

- **Contribution:** replace forward KL with **reverse KL** for LLM distillation and derive its **on-policy** optimization.
- **Why reverse KL:** forward KL makes a small student cover *all* teacher modes, overestimating the teacher's **low-probability regions** it can't represent (→ implausible output); reverse KL (mode-seeking) focuses on the modes the student *can* reproduce — right when the student can't fully mimic a large teacher (ch-01 geometry + capacity).
- **Optimization:** `KL(q_θ ‖ p)` is an expectation **under the student** (ch-04 Q4/Q5 outer expectation) → minimizing it requires differentiating the sampling distribution → **REINFORCE policy gradient** (high variance) → inherently on-policy. MiniLLM made the "reverse KL ⇒ sample from student ⇒ on-policy" link rigorous (= ch-01 Q7 bonus, formalized).
- **3 stabilization tricks:** single-step decomposition (variance reduction), teacher-mixed sampling (guards against degenerate/reward-hacking samples — same family as ch-03 Q7/Q8 garbage-guard/λ-warmup), length normalization (removes short-sequence bias).

TM's per-token reverse KL = the dense per-token realization of MiniLLM's "sample from the student, grade with the teacher".

**One line:** MiniLLM (Gu et al. 2024) = brought reverse-KL distillation to LLMs and derived its inherently-on-policy optimization (policy gradient + 3 stabilization tricks); mode-seeking so a small student doesn't overestimate the teacher's low-prob regions. See read.md §5, [[gu-minillm-reverse-kd]], [[ch-01]] qa Q7.