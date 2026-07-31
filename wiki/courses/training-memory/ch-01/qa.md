<!-- qa: ch-01 — The Memory Ledger: What Fills a GPU
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-01 Q&A — The Memory Ledger

Clarifying questions raised while reading [[read]]. Kernels only; full detail in `read.md` / discuss transcript. Deeper precision treatment in [[ch-02]].

---

### Q1 — what is "mixed precision"?

Using **two float precisions in one training run**: heavy compute in low precision, weight accumulation in high precision.

- **Precision = bits per number.** fp32 = 4 bytes (precise, wide range, slow/heavy); bf16/fp16 = 2 bytes (fast, light, low precision).
- **Mixed = compute (forward/backward matmuls) in bf16 (2B) for ~2× tensor-core speed + half memory, but keep an fp32 (4B) MASTER copy of the weights** that the optimizer accumulates into.
- **Why keep the fp32 master (the point):** a weight update is tiny (`lr × grad` ≈ 1e-6). Added directly to a bf16 weight it **underflows to 0** and training stalls. Worked example: `w=0.7234`, update `0.0000031` — bf16's step size (ULP) near 0.72 is ~0.0039, so the update rounds to 0; fp32's ULP is ~6e-8, so it survives and accumulates across steps.
- **bf16 vs fp16:** bf16 shares fp32's exponent range → no loss scaling; fp16 needs dynamic loss scaling. Modern training prefers bf16.
- **Ledger tie-in:** this is *why* weights appear twice — a 2B bf16 working copy (ledger item 1) **and** a 4B fp32 master hidden inside the 12B optimizer states (item 3).

**One line:** mixed precision = matmuls in bf16 (2B, fast/light) but an fp32 (4B) master weight for the optimizer to accumulate tiny updates into without underflow. See read.md §1.1/§1.3, [[ch-02]].

---

### Q2 — what does "using AdamW in mixed precision" mean?

AdamW (the optimizer) runs on the **fp32 master** and stores its per-parameter state in **fp32** → **12 B/param**.

- **AdamW** turns gradients into updates and keeps, per parameter, two fp32 running stats: **m** = first moment (momentum, EMA of grad) and **v** = second moment (EMA of grad²); update ∝ `m/√v` (adaptive per-param step). "W" = decoupled weight decay.
- **The 12 B/param = fp32 master weight (4) + m (4) + v (4)** — exactly the "Adam states 12 B/param" in the ledger and the **12** in the Rule of 16 (2+2+12). For 27B that is **324 GB** (§1.3).
- **Per-step loop:** ① bf16 weight → forward (bf16) → loss; ② backward → bf16 grad; ③ upcast grad to fp32; ④ AdamW updates m,v and the fp32 master; ⑤ downcast master → fresh bf16 weight. Compute is bf16; the optimizer's state + accumulation are all fp32 (same underflow reason as Q1).

**One line:** "AdamW in mixed precision" = the optimizer keeps fp32 master + fp32 m + fp32 v = 12 B/param and steps on the fp32 master, while forward/backward run in bf16 — that 12 is the ledger's optimizer term and the "12" in the Rule of 16. See read.md §1.3/§2, [[ch-02]].

---

### Q3 — so training = fp32, compute = bf16?

**Wrong axis (gentle fix).** "Training vs compute" aren't opposites — compute is *part* of training. The real split is **compute vs accumulate/store**, and both happen inside every training step:

| | precision | what lives here |
|---|---|---|
| **Compute** (matmul: forward/backward) | **bf16** | the heavy matrix ops, redone each step |
| **Store + accumulate** (master weight + Adam m·v + optimizer step) | **fp32** | the authoritative weights + optimizer state, persisting across steps |

- **Analogy:** fp32 master = the official ledger book (precise, permanent); bf16 = a fast scratchpad you copy each step to do the bulk arithmetic on. Heavy math on the scratchpad (bf16), authoritative running total in the book (fp32) so rounding doesn't accumulate.
- **Why this axis (ties to Q1):** underflow only bites when *accumulating a tiny update over many steps* → only the accumulators (master, m, v) need fp32; a one-shot matmul is safe in bf16. "Precision is needed for the *accumulation*, not the *compute*."
- **Deeper:** even a bf16 matmul accumulates its products in an fp32 accumulator inside the tensor core — hardware follows the same "multiply in bf16, sum in fp32" rule ([[ch-02]]).

**Corrected sentence:** not "training fp32 / compute bf16" → **accumulate & store (master + optimizer state) fp32, compute (matmul fwd/bwd) bf16.**

**One line:** the axis is compute (bf16) vs accumulate/store (fp32), both within each step — not training vs compute; only the things that accumulate tiny updates (master weight, Adam m/v) need fp32. See read.md §1.1/§1.3, [[ch-02]].

---

### Q4 — what is the difference between weights and gradients?

**Weights = the model's values themselves. Gradients = the signal saying how to change them.**

| | Weights | Gradients |
|---|---|---|
| what | learned parameters (W, b); the model *is* these numbers | `∂loss/∂W`, produced by the backward pass |
| lifetime | persistent (saved in checkpoints) | transient — recomputed each step, dropped at `zero_grad()` |
| count | `N_total` | `N_trainable` only |
| ledger | item 1 — **2 B/param** (bf16 working copy) | item 2 — **2 B/param** (trainable only) |

- **Why the distinction drives memory:** the *counts differ*. Weights must exist for all `N_total` (forward needs them). Gradients exist only for trainable params — so freezing the base model in LoRA gives: weights still 54 GB (27B × 2 B, can't shrink — forward needs them), gradients ≈ 0, and **optimizer states ≈ 0 by extension** since Adam state attaches only to params that have gradients. Rule of 16 (2+2+12): LoRA keeps the leading 2 and deletes the trailing 14.
- **Per-step chain (extends Q2's loop):** `weight(bf16) → forward → loss → backward → gradient(bf16) → upcast fp32 → AdamW updates m,v → fp32 master → downcast to fresh bf16 weight → gradient discarded`. The gradient is an intermediate ingredient for updating the weight; the weight survives, the gradient does not.

**One line:** weights persist and number `N_total` (2 B/param); gradients are transient, number `N_trainable` (2 B/param), and their absence is what makes LoRA cheap — no gradient → no 12 B/param optimizer state. See read.md §1.2/§3.

---

## Continued in [[qa-deep]]

This page hit the 120-line cap. AdamW-internals questions live in
[`qa-deep.md`](qa-deep.md):

- **Q5** — why does Adam need 3 tensors? Wouldn't 2 do?
- **Q6** — term-by-term derivation SGD → AdamW (+ interactive figure)
