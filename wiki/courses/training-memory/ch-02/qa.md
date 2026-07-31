<!-- qa: ch-02 — Optimizer States, Precision, and the Loss-Head Spike
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-02 Q&A — Precision and the Loss-Head Spike

Clarifying questions raised while reading [[read]]. Kernels only; full detail in `read.md`.
Prior chapter's precision questions: [[qa]] / [[qa-deep]] under `ch-01/`.

---

### Q1 — what is the dynamic-range problem? Does training really need exponent more than mantissa?

**Two independent axes.** Exponent bits buy *range* (how many orders of magnitude are representable); mantissa bits buy *resolution within* an order. You cannot trade one for the other after the fact.

| | min normal | max | dynamic range | relative precision (ULP) |
|---|---|---|---|---|
| fp32 (1+8+23) | 1.18e−38 | 3.4e38 | ~76 orders | 2⁻²⁴ ≈ 6e−8 |
| bf16 (1+8+7) | 1.18e−38 | 3.4e38 | ~76 orders | 2⁻⁸ ≈ **0.4 %** |
| fp16 (1+5+10) | **6.10e−5** | **65,504** | **~9 orders** | 2⁻¹¹ ≈ 0.05 % |

fp16 is ~8× *more* precise than bf16 but has ~67 orders of magnitude *less* range. bf16 is a truncated fp32; fp16 sold 3 exponent bits to buy 3 mantissa bits.

- **The failure is at the small end, not 65,504.** Gradients routinely sit below fp16's smallest normal (6.1e−5) — deeper layers, later in training, params on different scales. read.md L78: without loss scaling, **67 % of SSD gradient values became exactly zero** in fp16; bigLSTM needed 128× scaling.
- **Why exponent wins — asymmetry of the two error types, not intrinsic value:**

| | mantissa shortfall | exponent shortfall |
|---|---|---|
| symptom | rounding error | **flush to zero** |
| bias | **unbiased** (round-to-nearest) | **biased** — always toward zero |
| across steps | **cancels out** | **accumulates** |
| signal | silent | silent — not even a NaN |

  Rounding is zero-mean noise, and the thing that averages it away is the fp32 master + Adam's m,v from [[ch-01]] — machinery already present. Underflow is one-directional: a zeroed gradient averages to zero, and a parameter whose gradients sit *systematically* below the threshold **never trains at all** while the loss keeps falling. Silent failure.
- **The precise claim** is conditional, not absolute: *precision can be bought back with the fp32 accumulator you already have; range can only be bought back by bolting on new machinery.* Without fp32 masters, pure-16-bit training fails on both axes.
- **Loss scaling = emulating exponent bits in software** (read.md L63–76). Multiplying the loss by `S` makes the chain rule scale every gradient by `S` — a pure translation of the gradient histogram into fp16's window, mathematically equivalent, undone before `optimizer.step()`. Cost: overflow at the top → inf → NaN → **the step is discarded**; hence dynamic scaling (halve on overflow, double after N clean steps) and the hard ordering requirement of unscale-before-clip. bf16 deletes all of it (read.md L80).
- **Ties to [[ch-01]] Q3:** precision is needed for *accumulation*, not *compute*. A bf16 value is always a one-shot matmul input whose products accumulate in an fp32 tensor-core accumulator — 0.4 % is fine there. Range is not a rounding question but a can-this-number-exist question, so it is non-negotiable.

**One line:** exponent bits set range and mantissa bits set resolution; training needs range more because mantissa error is unbiased noise the fp32 master averages away, while underflow is biased error that silently and permanently deletes gradient information — bf16 keeps fp32's 8-bit exponent and pays with 0.4 % relative precision, which is affordable precisely because nothing accumulates in bf16. See read.md §2, [[ch-01]] qa Q1/Q3, [[mixed-precision-training]].

---

### Q3 — so should training just be done in bf16?

**Yes, bf16 mixed precision is the default** — and the A100-40GB (Ampere, SM80) supports it. But "train in bf16" is a *policy*, not a blanket cast; the real skill is knowing what stays fp32.

**Hardware decides first:** Ampere+ → bf16 + fp32 master, no loss scaler. Volta/Turing (V100, T4) have **no bf16 tensor cores** → fp16 + dynamic loss scaling is the only option. Hopper adds fp8 for some layers (§3) — **not available on A100**. fp16 is effectively "only when the hardware can't do bf16": trading 3 mantissa bits for loss scaling's failure modes (discarded steps, NaN hunts, unscale-ordering bugs) is a bad deal in training.

| what | dtype | why |
|---|---|---|
| weight working copy, activations, matmul **inputs** | bf16 | one-shot inputs; 0.4 % suffices |
| matmul **accumulator** (inside tensor core) | fp32 | hardware-enforced — it sums products |
| fp32 master, Adam m·v | fp32 | everything that accumulates ([[ch-01]]) |
| **softmax, LayerNorm, cross-entropy, reductions** | **fp32** | summing many terms needs both range and precision |
| gradient all-reduce | fp32 preferred | sums across world_size ([[ch-07]]) |

In PyTorch this is `torch.autocast(dtype=torch.bfloat16)`'s **fp32 op list** (softmax, layer_norm, cross_entropy, sum…).

- **This is the cause of §4.** `cross_entropy` sits on the fp32 list, so the logit tensor is **4 B/element, not 2 B** — the `4` in `1 × 4096 × 248,000 × 4 B = 4.06 GB` for boson's 248k vocab. It explains why a "bf16 run" still gets torn apart by a 4-byte tensor.
- **Practical fallout:** delete `GradScaler` entirely; keep gradient-accumulation buffers in fp32 (accumulating microbatches is accumulation — Q2 applies; see FSDP `MixedPrecision.reduce_dtype`); and note that `model.bfloat16()` is **not** autocast — it destroys the fp32 master and walks straight into [[ch-01]] Q1's underflow. Common accident. Pure-bf16 with stochastic rounding (saving the 4 B master) exists but is not yet default.

**One line:** on Ampere+ train in bf16 mixed precision with fp32 masters and no loss scaler — but the operative knowledge is the fp32 exception list (accumulators, reductions, cross-entropy), which is exactly why the §4 logit tensor costs 4 B/element. See read.md §2/§4, [[ch-01]].

---

### Q4 — why does fp16 store it as 0? It has *more* mantissa bits, so shouldn't it represent more?

**Mantissa has nothing to do with how small a number can be.** A float is scientific notation: `value = 1.mmmmmmm × 2^exponent`. Mantissa = how many digits (`3.14` vs `3.14159`); exponent = whether `× 2⁻²⁵` is reachable at all. Different jobs — more mantissa cannot reach where the exponent can't go.

`0.00000003` needs `2⁻²⁵`. fp16's exponent bottoms out at `2⁻²⁴` (≈ 5.96e−8) → **unreachable, stored as 0**, regardless of digit count. bf16 reaches `2⁻¹²⁶`.

| value | fp16 | bf16 |
|---|---|---|
| `1.0009` | **`1.0009`** ✓ — fp16 is genuinely *more* precise | `1.00` (rounded) |
| `0.00000003` | **`0`** ✗ | `0.00000003` ✓ |

So fp16 isn't worse — its good region is *narrower*, and training gradients frequently fall outside it. Rounding (`3.14159 → 3.14`) is later repaired by the fp32 master; a zero cannot be repaired, because the number never existed.

**One line:** mantissa = precision, exponent = reach; fp16 is more precise *within reach* but training gradients sit below its floor, and a value that rounds to 0 is unrecoverable while a rounded one is not. See read.md §2, Q1.

---

### Q5 — terminology: I had been reading "mantissa" as the fractional part of the value

**`fraction = mantissa` was correct** — IEEE 754 names the field *fraction*, "mantissa" is the common alias (strictly `significand = 1.fraction`). The trap is elsewhere: the everyday sense of *fraction* (digits after the decimal point) is **not** the IEEE field.

Under the wrong model — *"mantissa = how far below the decimal point I can go"* — the inference in Q4 ("more mantissa bits → can represent 0.00000003") is **logically valid**; only the premise was wrong.

**Correct definition:** `value = 1.mmmm × 2^k`. The mantissa only describes a shape **between 1 and 2**; magnitude is set *entirely* by the exponent. Proof — three numbers with **identical mantissa bits** (`1.25 × 2^k`):

| value | decomposition | mantissa field | exponent | fp16 stores? |
|---|---|---|---|---|
| `5.0` | 1.25 × 2² | `0100000000` | +2 | ✅ |
| `0.15625` | 1.25 × 2⁻³ | `0100000000` | −3 | ✅ |
| `0.00000003725` | 1.25 × 2⁻²⁵ | `0100000000` | **−25** | ❌ → **0** |

fp16 fails the third **not** for lack of mantissa (it is bit-identical to the first two) but because its exponent bottoms out at −24. All the digits are there; the decimal point cannot move far enough left.

| | learner's model | actual |
|---|---|---|
| what mantissa bits do | how far below the decimal point | distinguish `1.0009` from `1.0010` |
| what reaches small values | mantissa | **exponent** |
| conclusion | fp16 (10) beats bf16 (7) | **range is an exponent contest: fp16 (5) < bf16 (8)** |

**Follow-up distinction (learner):** "bigger exponent → smaller numbers" splits two ways — more exponent **bits** ✅ (a wider window), but a larger **value k** ❌ (that gives *bigger* numbers; small values need large *negative* k). Say "more exponent bits", not "bigger exponent". Bits widen the window **both ways**: 5 bits → k ∈ [−14, +15] = `6.1e−5 … 65,504`; 8 bits → k ∈ [−126, +127] = `1.18e−38 … 3.4e38`. That two-sided narrowness is why fp16 loss scaling must be *dynamic* — too small an `S` underflows to 0, too large overflows to inf/NaN, and a 9-order window leaves no slack; bf16's 76-order window needs no such tuning.

**One line:** mantissa describes the shape between 1 and 2 and the exponent alone places it on the number line — `5.0` and `3.725e−8` have identical mantissa bits — so reach is decided purely by exponent width, which is the whole bf16-over-fp16 argument. See read.md §2, Q1/Q4.

---

## Continued

Mechanisms in [`qa-deep.md`](qa-deep.md) — **Q2** what "cancels across steps" means (√n vs n);
**Q4** how FP8 trains on a narrower exponent than fp16; **Q5** spread-vs-accumulation (two axes).

Logit spike (§4-§5) in [`qa-deep-2.md`](qa-deep-2.md) — **Q6** what `seq` is; why several logit copies coexist.
