<!-- qa-deep: ch-01 — AdamW internals (overflow from [[qa]], 120-line cap)
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-01 Q&A (deep) — AdamW internals

Overflow page for [[qa]]. Same rules: kernels only, full detail in `read.md`
and the interactive companion [`figures/adamw-derivation.html`](figures/adamw-derivation.html).

---
### Q5 — why does Adam need 3 tensors? Wouldn't 2 (previous + current) do?

**Premise fix:** the three are *not* versions of the weight. Adam is a first-order stateful method — no past weight is stored.

| tensor | what it is | role |
|---|---|---|
| fp32 master | the weight itself (one, current) | **precision cost**, not Adam (see Q1/Q3) |
| m (first moment) | EMA of gradient | direction |
| v (second moment) | EMA of gradient² | scale |

Adam's own state is **2 tensors = 8 B/param**; the remaining 4 B is mixed precision's master copy. In pure fp32 training the master *is* the weight, so Adam costs only +8 B.

**Naming (learner slip, corrected):** there is no "AdamW weight" — the 4 B master is owned by precision, not the optimizer. And m and v are not "two momentums": only **m** is momentum (first moment); **v** is the second moment / variance, EMA of grad**²**. The SNR argument below works precisely *because* v is not a momentum — being squared it has no sign, so it does not cancel under oscillation.
**Corollary (bridges to §2):** pure-fp32 AdamW = 4 (w) + 4 (grad) + 4 (m) + 4 (v) = **16 B/param**, the same Rule-of-16 total as bf16 mixed (2+2+12). Mixed precision moved money between line items without lowering the static total — its wins are speed (tensor cores) and activations, not the static ledger.

**Why two gradient statistics and not one — direction and scale are not recoverable from each other.** Worked example (β₁=0.9, β₂=0.999, bias correction ignored):

| param | gradients | m | v | √v | m/√v | behavior |
|---|---|---|---|---|---|---|
| A consistent | +.10 ×4 | .10 | .01 | .10 | **≈1.0** | full step — signal is clear |
| B oscillating | +.10,−.10,+.10,−.10 | **≈0** (cancels) | .01 (squares don't cancel) | .10 | **≈0** | stay put — signal contradicts itself |
| C consistent but tiny | +.001 ×4 | .001 | 1e−6 | .001 | **≈1.0** | same step as A despite 100× smaller grad |

A vs B share the same `v` and differ only in `m`; A vs C differ 100× in gradient magnitude yet take the same step. So `m/√v` is a **signal-to-noise ratio** — one accumulator cannot express it. `m` alone can't say whether 0.10 is large; `v` alone can't say which direction.

**Why this matters for LLMs specifically:** embedding / LayerNorm / MoE-expert gradients differ by orders of magnitude, so no single global `lr` serves them all — `v` normalizes per-parameter automatically. Acute for the boson 256-expert MoE, where rarely-routed experts receive sparse gradients.

**Dropping one is a real, named choice** (read.md §1.3 alternatives table) — 27B full-FT: 324 GB → 108 GB → 54 GB, the single largest ledger lever (revisited in [[ch-08]]):

| dropped | name | B/param | cost |
|---|---|---|---|
| — | AdamW | 12 | — |
| v | SGD+momentum | 4 | no per-param adaptive step → lr-tuning hell |
| m | RMSProp | 4–8 | no momentum noise-smoothing |
| v factorized | Adafactor | 4 | v approximated by row/col sums |
| m → sign | Lion | 4 | magnitude information |
| m,v 8-bit | bitsandbytes 8-bit Adam | 2 | precision (measured loss small) |

m and v need fp32 for the same underflow reason as Q1 — `v` especially, being gradient **squared** (grad 1e−4 → v 1e−8) accumulated slowly by EMA.

**One line:** Adam's state is 2 gradient statistics (m = direction, v = scale) + 1 fp32 master that belongs to mixed precision, not Adam; you *can* keep only one statistic — that's SGD-momentum / RMSProp / Adafactor / Lion at 4 B/param — and what you give up is the per-parameter adaptive step `m/√v`. See read.md §1.3, [[ch-02]], [[ch-08]].

---

### Q6 — derive it: how do the formulas assemble into AdamW, term by term?

**Interactive:** [`figures/adamw-derivation.html`](figures/adamw-derivation.html) — stage tabs, live number substitution, step-multiplier plot, bias-overshoot curve. Formulas in the figure, kernel here.

Each stage fixes **one named defect** of the previous, and **only terms that carry state cost memory**:

| stage | added term | fixes | optimizer state |
|---|---|---|---|
| 0 SGD | `w −= η·g` | — | 0 B |
| 1 +momentum | `m = β₁m + (1−β₁)g` | ① minibatch noise — EMA folds all history into **one** accumulator (window `1/(1−β₁)`≈10) | **4 B** |
| 2 +second moment | `v = β₂v + (1−β₂)g²`, step `= m/(√v+ε)` | ② step ∝ \|g\| — `√v` has the *same units* as `g`, so `m/√v` is **dimensionless** → step ≈ η regardless of gradient scale | **4 B** |
| 3 +bias correction | `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)` | ③ m,v start at 0 → biased low; `E[mₜ] = E[g](1−β₁ᵗ)` so dividing cancels it exactly | **0 B** (t is a scalar) |
| 4 AdamW | `+ λ·w` added to the *update*, not to `g` | ④ coupled L2 gets divided by `√v̂` too → effective decay `λ/√v̂` varies per parameter | **0 B** (reuses w) |

- **Why β₂ ≫ β₁:** `v` estimates a slowly-changing *magnitude* and squares are high-variance, so it needs a long window (1000); `m` must react fast to direction changes (10).
- **Verified numbers** (η=1e−3, λ=0.01, β=0.9/0.999): consistent `+0.10` → multiplier **1.007** every step; oscillating `±0.10` → **0.046**; tiny `+0.001` → **1.007** again (SGD would give 0.001) — the 100× gradient gap produces an *identical* weight trajectory. Bias correction off: multiplier **3.16×** at t=1, peaks **6.57×** at t=12, and needs **t≈2375** to fall under 1.05×. That overshoot is why the correction exists and why **warmup** is standard.
- **Ledger mapping:** only the two EMA lines bill anything — `m` 4 B + `v` 4 B = **8 B/param is Adam's true cost**; the 4 B fp32 master is mixed precision's. Bias correction and decoupled decay lengthen the formula at **zero bytes**.

**One line:** AdamW = SGD + m (direction, 4 B) + v (scale, 4 B) + bias correction (0 B) + decoupled decay (0 B); "does it carry new state?" is the only question that decides memory, which is why the optimizer costs 8 B/param and the Rule of 16's remaining 4 B belongs to precision. See read.md §1.3/§2, [[ch-02]], [[ch-08]].
