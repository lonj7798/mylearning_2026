<!-- qa: ch-03 — Exposure Bias / DAgger (O(eps*T^2) vs O(eps*T))
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-03 Q&A — Exposure Bias / DAgger

Clarifying questions raised while reading [[read]]. Kernels only; full derivations live in `read.md` / the discuss transcript.

---

### Q1 — what does "Dataset Aggregation" mean?

It's what **DAgger** stands for: **DA**taset **Agger**egation (Ross et al. 2011) — the name *is* the mechanism: iteratively **accumulate** training data.

**Fixes:** behavioral cloning trains only on expert-visited states, but the deployed policy visits its *own* drifted states → untrained there → O(ε·T²) (ch-01 P2).

**The aggregation loop:**
1. train an initial policy (BC on expert demos);
2. repeat: (a) roll out the *current* policy → collect the states **it** visits (on-policy); (b) ask the **expert** to label each of those states ("what would you do here?"); (c) **AGGREGATE** `D ← D ∪ {new (state, expert-action) pairs}` — never discard; (d) retrain on the grown `D`.

**"Aggregation"** = the `D ← D ∪ D_i` step: the dataset accumulates across rounds until it covers the states the policy actually reaches → the policy is graded *where it goes* → **O(ε·T²) → O(ε·T) linear**. (Detail: early rounds mix in the expert via β, decaying to 0; β=1 first round = plain BC.)

**Course connection:** DAgger is the direct ancestor of on-policy distillation — same skeleton (**student generates trajectory → teacher grades it → repeat**). Difference: DAgger labels each state with one **hard expert action**; on-policy distillation gives the **full teacher distribution** (reverse KL, dense O(N)). OPD = DAgger's on-policy grading + Hinton's soft per-token signal.

**boson:** run the current seller (on-policy) → teacher labels the states it reaches → aggregate → retrain = the classical original of the capstone move.

**One line:** DAgger = DAtaset AGGERegation — accumulate (current-policy states, expert labels) into a growing dataset and retrain, so the policy is graded on its own drift-states → O(εT²)→O(εT); the ancestor of on-policy distillation. See read.md, [[ross-dagger-exposure-bias]], [[ch-01]] P2.

---

### Q2 — for boson (DAgger = keep expanding objection-handling), how do you actually EXPLORE those edge-case states?

Reframe: exploration here = **state coverage** (visit the states you'll face at deployment), NOT reward-seeking as in RL — the teacher already gives a target at every state.

**Two sources of edge-states:**
1. **Seller's own drift (free):** just running the current seller on-policy surfaces the novel states its own mistakes lead to — like the car drifting to the lane edge. This is why on-policy (student generates) matters.
2. **Customer behavior (the main driver for dialogue):** unlike driving (drift = pure policy error), conversational hard-states are mostly produced by the *customer* (unusual objections, barge-ins, escalation). So "explore edge cases" = make the **customer simulator produce them.**

**Exploration levers (all via the customer sim, which boson already has = 11-model rotation):** population diversity (styles/tones); scenario design (chained objections, price resistance, barge-in); adversarial/red-team customers (temp↑, difficult personas); **curriculum / hard-example mining** — target scenarios around the seller's observed failures instead of random exploration.

**Ceiling (the key answer):** you can only explore — and therefore only fix — states the customer sim can **generate**. Real edge cases the sim never produces stay unfixed at deployment. So **customer-sim realism/diversity is the upper bound on exploration** (sim-to-real gap); DAgger's O(εT²)→O(εT) holds only *within the sim's state distribution*. The customer sim is boson's exploration engine **and** its exploration ceiling.

**One line:** explore edge-states via on-policy rollout (seller's own drift, free) + customer-sim diversity (population/scenario/adversarial/curriculum); hard limit = you can't fix what the sim never generates, so sim realism bounds coverage (sim-to-real gap). See read.md, [[ch-01]] P2, [[ch-02]] qa Q8.

---

### Q3 — does DAgger train by correcting a whole conversation at once, or round-by-round?

Separate **three granularities** (the "whole vs round-by-round" framing conflates them):

| axis | unit |
|---|---|
| **collection** (what you roll out) | the **whole conversation** |
| **correction/label** (what gets fixed) | **per-state (per-turn)** — NOT the whole conversation as one blob |
| **retrain loop** (when you retrain) | **round (iteration) by round**, on the accumulated pool |

So "correct the whole conversation at once" is wrong — the conversation is only the *collection* unit; each visited state `s_t` becomes its own independent training example `(s_t, expert-action)`. "Round by round" is right in both senses: per-turn correction, and per-iteration retraining.

```
Round i: rollout whole convo -> states [s1..sT]
         label each: (s1,a1*),...,(sT,aT*)   <- per-state, not whole-blob
         D <- D ∪ {pairs}; retrain on all of D
```

**OPD nuance (boson):** finer + more online than classic DAgger — label is a **per-TOKEN teacher distribution** (reverse KL), and updates are typically online (fresh on-policy samples each step, since the student moves and old samples go stale, ch-01 Q2) rather than a growing aggregated dataset: sample → per-token reverse-KL grad → update → resample.

**One line:** collection = whole conversation; correction = per-state (per-turn), not a whole-conversation rewrite; retrain = round-by-round on the accumulated pool. OPD is finer (per-token) and more online. See read.md, [[ch-01]] qa Q2.

---

### Q4 — is the absolute difference between on-policy and off-policy that you learn the token's probability distribution, not just the output token?

No — that conflates two **independent** axes (the same conflation that caused the ch-02 Partial).

- **on/off-policy = DATA SOURCE axis:** did the **student itself** generate the training sequences (on) or **something else** (off) — i.e. *which state distribution* you train on. This is the exposure-bias axis. Nothing to do with soft/hard.
- **"output token vs full distribution" = soft/hard (bits-per-token) axis** (Q10; note output-only is still O(N), just hard). A *separate* axis about per-token signal richness.

**Independence proof — all four cells exist (diagonals are the killers):**

| | output-token only (hard) | full distribution (soft) |
|---|---|---|
| **off-policy** | seq-KD (Kim&Rush) | **Hinton soft KD** |
| **on-policy** | **DAgger (hard expert action)** | on-policy distillation |

off-policy+soft (Hinton) ⇒ full distribution ≠ on-policy; on-policy+hard (DAgger) ⇒ on-policy ≠ full distribution. So the two axes don't imply each other.

**Why the confusion:** OPD uses the full distribution because it picks **on-policy (Axis 1) AND soft (bits/token) both** — not because on-policy requires it. Same shape as ch-01 P1's density ⊥ on-policy (GRPO counterexample); here soft/hard ⊥ on-policy (DAgger counterexample).

**One line:** on/off-policy = who generated the data (state distribution); soft/hard (output-token vs full-distribution) is an independent axis; OPD is full-distribution because it also chose soft, not because it's on-policy. See [[ch-01]] P1 & qa Q4/Q8, read.md.

---

### Q5 — if the student makes an answer and we steer it toward the teacher's generated answer, does it become on-policy?

Depends entirely on **where the teacher's signal is evaluated.** "Student generates" is the necessary ingredient, but on-policy is confirmed only if the student is **graded on its own trajectory** (loss summed over the student's tokens).

- **Version A (on-policy ✓):** student generates `S=[s1..sT]`; at each of the student's own positions, teacher grades "at this state the student reached, what would I do?" (reverse KL / corrective action). Teacher = **grader on the student's states**. = DAgger / OPD.
- **Version B (still off-policy ✗):** student generates `S` (discarded); teacher generates its **own** answer `Tr`; student trained to match `Tr`. Teacher = **author of a separate answer** → training on the teacher's states → just seq-KD; the student's generation was pointless.

**Decisive test:** whose tokens does the loss sum over? Student's (S) → on-policy. Teacher's (Tr) → off-policy. The criterion is *"grading happens on the student's trajectory,"* not *"the student generated first."*

**Fix to the phrasing:** "student makes S, then **teacher grades/steers at each position of S**" → on-policy; "steer toward the teacher's *separate* answer" → off-policy.

**One line:** student-generation alone isn't enough — on-policy requires the teacher to grade the student's OWN trajectory (loss over student tokens); using the teacher's separate answer as the target is still off-policy (seq-KD). See [[ch-02]] qa Q8, read.md, [[ch-01]] P2.

---

### Q6 — so pre-made conversations don't work; you build the conversation from the student's own responses and re-write the responses? (learner synthesis)

Two parts right, one to refine.

1. **"pre-made transcripts don't work"** — ✅ for on-policy (they're off-policy). Refine "no effect": existing boson transcripts still bootstrap the **initial off-policy SFT** (DAgger Round 0 / BC); they just can't provide the on-policy exposure-bias fix. So "unusable *for on-policy*", not "no effect".
2. **"build the conversation from the student's own responses"** — ✅ the key. Multi-turn nuance: in the **relay as rollout env**, seller turns = the student (on-policy), customer turns = the sim; the student's turns become the context for later turns, so the student's trajectory genuinely unfolds. This IS the capstone structure.
3. **"re-write the responses"** — 🔧 refine: not re-write (that risks Q5 Version B), but **grade in place**. Keep the student's response as-is (it's the *state*/training position); at each position the teacher supplies a target (distribution for OPD, corrective action for DAgger); loss is over the **student's tokens**. States from the student, target from the teacher — "teacher red-pens the student's answer", not "teacher writes a replacement".

**Big-picture realization (capstone core, learner-reached):** boson **cannot reuse exported transcripts for on-policy** — it must run the seller **live in the relay**, then grade the seller's own turns.

**One line:** old transcripts = fine for initial off-policy SFT, not on-policy; must roll the seller out live in the relay (seller=student, customer=sim) and grade its own turns in place — not re-write them. See read.md, [[ch-02]] qa Q8, forward to ch-07 capstone.

---

### Q7 — but if you grade a later response built on the student's own low-quality history, isn't training corrupted by that bad history?

Yes, it's conditioned on the low-quality history — **and that's the point (feature, not bug).** The deployed seller WILL land in "after my own weak turn 5" states, so teaching it what to do there = learning to **recover from its own mistakes** = the exposure-bias fix (DAgger: the car learns to recover from the lane edge it drifted to). Off-policy (clean teacher history) can't teach recovery.

**Key distinction — history is masked CONTEXT, not a TARGET:**
```
[student's low-quality history]  <- context (input), masked, NOT graded
[student's current response]     <- graded here only
teacher target: "given this (messy) state, best response = X"
loss: over the student's current-response tokens only
```
You are **not** trained to reproduce the bad history; you're trained to produce a **good response conditioned on it**. (= ch-07 figure's "grade seller tokens, mask customer/context".) So "affected by bad history" = conditioning, not corruption.

**The real risk (student so bad the whole rollout is garbage states) is handled by:** (1) bootstrap from a decent SFT checkpoint — DAgger Round 0 / BC = the existing transcripts (why they stay useful, Q6); (2) the teacher gives a good target even in bad states; (3) self-curriculum — as the student improves, its rollout states improve, shifting the distribution from garbage → realistic.

**Trade-off:** off-policy avoids messy states but keeps exposure bias; on-policy accepts messy-but-realistic states to gain recovery. Messiness is the price of relevance.

**One line:** later responses are conditioned on the student's own weak history on purpose — that teaches recovery from its own drift; the history is masked context (not a target), so it's conditioning not corruption; garbage-rollout risk is handled by SFT bootstrap + good teacher targets + self-curriculum. See read.md, [[ch-01]] P2, ch-07 capstone.

---

### Q8 — (learner strategy) if it keeps producing garbage, do a bit of SFT to shift the student first, then go on-policy?

Correct — this re-derives the **standard SFT-warmup → on-policy pipeline**, and the "a bit" instinct is right.

- **Standard recipe:** DAgger Round 0 = BC (= SFT); the general "SFT → RL/OPD" pipeline (RLHF = SFT then PPO); the Qwen3 off-policy→on-policy order.
- **Why:** SFT moves the student's rollout **state distribution** out of the garbage regime so on-policy training lands on *recoverable* drift states instead of unrecoverable junk. ch-01 framing: SFT (forward-KL, mode-covering) sets a broad starting point; on-policy distillation (reverse-KL, mode-seeking, on the student's own states) commits + fixes drift. Complementary phases.
- **"A bit" is right:** enough to escape garbage; too much SFT overfits the off-policy distribution and bakes in exposure bias.
- **Not binary — it's a schedule:** DAgger's β decay, or GKD's **λ (on-policy fraction) annealed 0→1** (ch-06) *is* exactly "SFT first, then increasingly on-policy". The learner re-derived the λ schedule.
- **boson:** the existing off-policy SFT'd seller (`Qwen3.6-27B-Lina-chk-*`) **is** this warmup — Round 0 already done; capstone = add the on-policy phase on top (why the garbage risk is small: not starting from scratch).

**One line:** SFT-warmup→on-policy is the standard pipeline (DAgger Round 0=BC); SFT lifts the rollout state distribution out of garbage, on-policy then fine-corrects; not binary but a λ: 0→1 anneal (ch-06); boson's existing SFT checkpoint is already Round 0. See read.md, [[ch-01]] §4, forward to ch-06 & ch-07.

---

### Q9 — why is the error O(T) for on-policy but O(T²) for off-policy?

ε = per-step mistake rate, but **where ε is measured** is the crux. Decisive question: **after a mistake, is the policy in a state it was trained on (recoverable) or not?**

**off-policy → O(εT²) — two factors of T:** BC is trained only on the *expert's* state distribution, so ε holds only on expert-states. A mistake drifts the policy off-distribution into **untrained states** where it has no guarantee → can't recover → that one mistake spoils the **rest of the horizon (~T)**.
```
total = [T steps] × [ε first-mistake/step] × [cost/mistake ~T: unrecoverable, ruins the remainder]  = O(εT²)
```
= the learner's `Σ(T−t)=T(T+1)/2`. Two T's: **#steps × per-mistake cost = remaining horizon.**

**on-policy → O(εT) — the second T collapses:** DAgger trains on the policy's *own* state distribution (incl. drift states, via aggregation), so ε holds on the states it actually visits. A mistake lands in a **trained** state → recoverable → per-mistake cost is **O(1)**, not the remaining horizon.
```
total = [T steps] × [ε/step] × [cost/mistake O(1): trained to recover]  = O(εT)
```

**Crux:** the difference is whether the *second* T (cost-per-mistake) exists. off-policy: mistake→off-distribution→unrecoverable→costs ~T. on-policy: trained on its own drift-states→recoverable→costs O(1). Root cause = *which distribution ε is driven low on*: expert-states only (off, so ε≈1 in drift-states → spiral) vs the states the policy actually visits (on → no spiral).

**One line:** O(εT²) = [T steps]×[ε]×[per-mistake cost ~T because unrecoverable off-distribution]; O(εT) = same but per-mistake cost O(1) because on-policy trains on its own drift-states (recoverable) → the second T vanishes. See read.md, [[ross-dagger-exposure-bias]], [[ch-01]] P2.
