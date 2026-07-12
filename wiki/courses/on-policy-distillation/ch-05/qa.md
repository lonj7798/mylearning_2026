<!-- qa: ch-05 — Economics and Failure Modes: when on-policy distillation wins
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-05 Q&A — Economics & Failure Modes

Clarifying questions raised while reading [[read]]. Kernels only; full detail in `read.md` / discuss transcript.

---

### Q1 — is OPD O(N) bits/rollout because you compare every token's probability?

Mechanism right, label wrong (the recurring axis-conflation).

- **The mechanism is correct:** a signal at each of the N tokens (compare student vs teacher per token) = **O(N) bits**; vs RL's one scalar for the whole rollout = O(1) regardless of N.
- **But that's the DENSITY axis (the "distillation" half), NOT "on-policy".** on-policy = who *generated* the data (student); O(N) = the teacher *grading every token*. "Compare every token" is a density property, not an on-policy one.
- **Counterexample:** GRPO = on-policy **and** O(1) (grades with a scalar, not per-token) → on-policy does not imply O(N); you need dense per-token grading.
- **So OPD is O(N)** because it chose **dense (grade every token)**, independent of being on-policy. In ch-04 Q1's ①, "full-distribution per token" = O(N) (density); "student-sampled" = on-policy — two separate legs.

**Fix the sentence:** not "on-policy is O(N) because every token compared" → "**OPD is O(N) because it grades every token densely**"; that density is orthogonal to on-policy (student-generation).

**One line:** every-token comparison → O(N) is the density (distillation) axis, not on-policy; GRPO (on-policy·O(1)) is the counterexample; OPD's O(N) comes from the dense-grading choice. See read.md §2, [[ch-01]] qa Q4/Q8, [[ch-04]] qa Q1.

---

### Q2 — in OPD, what if the output token lengths don't match?

Depends entirely on whether teacher and student share a **tokenizer**.

- **Same tokenizer → no mismatch.** OPD doesn't compare two separate generations; the **student generates one sequence and the teacher grades *those* tokens** position-by-position (grade-in-place, ch-03 Q5). Same sequence, same length — nothing to match.
- **Different tokenizer (cross-family) → the real problem.** The same text splits into different token counts/boundaries (ch-02 Q10): e.g. "The answer is 12" = 4 student-tokens vs 6 teacher-tokens → position i doesn't correspond → per-token KL can't be aligned.

**Fixes:** (a) **same-family teacher** (same tokenizer) → avoid entirely, exact alignment, cheapest (capstone Fork 1); (b) **sequence-level / text**: re-tokenize the teacher's *text* with the student's tokenizer, train CE (loses soft per-token, ch-02 Q10); (c) **GOLD / cross-tokenizer distillation** (ch-06): align tokenizations at the **text/char level** and **aggregate teacher logprobs over the spanning tokens** — e.g. student "12" = teacher "1"+"2" → `log π_T("12") ≈ log π_T("1") + log π_T("2"|"1")` (chain rule). HF's "any model family" GOLD is this tool.

**boson (Fork 1):** large Qwen teacher (same tokenizer → exact alignment, cheap) vs Claude teacher (different tokenizer → needs GOLD). "Length matching" is a real teacher-choice cost axis in the capstone.

**One line:** same tokenizer → teacher grades the student's own tokens, no length issue; different tokenizer → same text splits differently → align via same-family teacher (avoid), text re-tokenize (hard), or GOLD (text-level align + span logprob aggregation). See read.md §5/§Where-This-Goes, [[ch-02]] qa Q10, ch-06.

---

### Q3 — (same-tokenizer) so at a prefix you compare student candidates (있지/없지/하하하) vs teacher candidates (없구요/없는데요/없습니다)?

Right in spirit — compare student vs teacher **next-token distribution at the same prefix** — but two refinements:

1. **Both are over the SAME full vocab, not different token sets.** You listed each one's *top* candidates; actually both assign a probability to *every* vocab token. Compare over the shared vocab: `KL_t = Σ_v π_θ(v)·[log π_θ(v) − log π_teacher(v)]`. Where student over-assigns ("있지" 0.40 vs teacher 0.05) → high KL, push down; where student under-assigns ("없습니다" 0.05 vs teacher 0.50) → push up. Net: pull the student's distribution toward the teacher's at this prefix.
2. **The teacher does NOT generate a separate answer** — it *provides its distribution* at the student's prefix (grade-in-place, ch-03 Q5 Version A). "없습니다..." = the teacher's high-prob tokens at that prefix, not a separate teacher generation (that would be Version B = off-policy).

**Trajectory:** the student **samples** one token (say "있지") → next prefix "…할인은 있지" → grading continues on the **student's own path** even though the teacher preferred "없습니다" (ch-03 Q7). Repeat the compare-at-position along the student's trajectory.

**One line:** yes = compare student-vs-teacher next-token distributions at each prefix, but both over the *same full vocab* (not different candidate sets), teacher *provides a distribution* (not a separate generation), and it repeats along the *student's own sampled* trajectory. See read.md, [[ch-03]] qa Q5/Q7, [[ch-04]] qa Q1.

---

### Q4 — so as training proceeds, the student comes to resemble the teacher?

Yes (fixed point: reverse KL → 0 at student=teacher, ch-04 §2) — with three ch-05 nuances so it isn't naive:

1. **Where it resembles:** in the student's **own states** (on-policy) — not copying the teacher's trajectory; it learns "what the teacher would do *here, where I actually land*", incl. recovery states the teacher rarely visits (ch-03).
2. **How much:** **selectively** — reverse KL is mode-seeking, so a small student commits to the teacher's best modes *it can reproduce* and drops the rest (capacity; ch-01 Q7, MiniLLM). Not a full copy.
3. **Twist — it can BEAT the teacher (§4):** OPD students beat both their SFT and RL teachers, and converge similarly despite different teacher quality → "the source of the data matters a lot while the **teacher matters less than expected**". Not merely capped at the teacher; on-policy sampling itself drives gains, so the student can exceed the teacher *on its own deployment distribution*.

**Ceiling (capstone Fork 2):** still bounded by what the teacher **knows** — a wrong teacher belief (e.g. a wrong tool call) gets faithfully copied. §4's "beats the teacher" is *overall performance*; it can't acquire knowledge the teacher lacks → for verifiable-correctness dimensions, add a verifier/reward.

**One line:** yes, resembles the teacher (reverse KL→0), but in its OWN states (on-policy), SELECTIVELY (mode-seeking on reproducible modes), and can even EXCEED the teacher (on-policy data > teacher quality, §4) — while still bounded by what the teacher knows (Fork 2). See read.md §4, [[ch-01]] qa Q7, [[ch-03]] qa Q7.

---

### Q5 — so N in O(N) = the total number of tokens?

Yes. **N = the number of tokens in the rollout/episode** (= the sequence length). `O(N)` bits/episode = the learning signal scales with the token count, because a per-token signal (teacher grade) lands at each of the N tokens. 500 tokens → ~500 signals; 5000 → ~5000 (10×). Contrast **O(1)** (RL): one scalar reward regardless of length ("a fixed number of bits regardless of the number of tokens used").

So longer sequences → distillation's signal grows (O(N)↑) while RL's stays flat (O(1)) — the ch-01 density axis, and why OPD especially favors long sequences (ch-05 §3).

**Note:** this N is the **same quantity as ch-03's horizon T** (sequence length / token count) — literature just uses N or T; still unrelated to the ch-02 temperature-T.

**One line:** N = total tokens in the episode; O(N) = signal count scales with tokens (per-token dense) vs O(1) = one scalar regardless of length; same quantity as horizon T. See read.md §2, [[ch-01]] §3, [[ch-03]] qa Q10.

---

### Q6 — so can you decrease/increase the number of tokens each time?

Yes — **N is variable and controllable**, but "increase N for more signal" doesn't work.

- **Varies naturally:** each student rollout has a different length; **controllable** via `max_new_tokens` / truncation.
- **N is a compute lever, not free signal:** N↑ → more per-token signals (O(N)) **and** more compute — teacher-logprob queries scale with N (the ch-05 §5 serving bottleneck, $$). Usually the **task determines N** (you don't truncate a 30-turn call to save compute).
- **Length normalization (MiniLLM, ch-04 Q7):** summing per-token KL over N tokens makes long sequences dominate the gradient, so the loss is **normalized by length** — which cancels the "more tokens = more signal" effect. So you can't game N.
- **Why large N helps OPD** = the **exposure-bias regime** (long horizon, ch-05 §3), NOT harvesting more signal from more tokens.

**One line:** N varies per rollout and is capped by `max_new_tokens`, but increasing it costs compute/teacher-serving and is length-normalized away — task sets N; large-N favors OPD via the exposure-bias regime, not signal harvesting. See read.md §3, §5, [[ch-04]] qa Q7.