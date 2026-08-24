<!-- qa-deep: ch-02 — precision internals + FP8 (overflow from [[qa]], 120-line cap)
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-02 Q&A (deep) — precision internals and FP8

Overflow page for [[qa]]. Same rules: kernels only, full detail in `read.md`.

---

### Q2 — what does "cancels across steps" mean? Does the update eventually happen if you keep going? Is exponent shortfall therefore worse?

**(a) "Cancels" = √n vs n, not disappears.** Random-signed error sums as a random walk (`√n`); the signal sums in one direction (`n`); so relative error decays as `1/√n`.

| over 10,000 steps | bf16 rounding (unbiased) | fp16 underflow (biased) |
|---|---|---|
| per-step gradient | 0.10 | 2e−8 (below fp16's 5.96e−8 min subnormal) |
| stored | 0.10 ± 0.0004, sign random | **0** |
| true sum / error sum | 1000 / 0.0004·√10⁴ = 0.04 | 2e−4 / 2e−4 (all of it) |
| relative error | 0.4 % → **0.004 %** | 100 % → **100 %** |

*Honest caveat:* rounding a *fixed repeated* value is deterministic and would also be systematic — decorrelation comes from gradients differing per minibatch. **Stochastic rounding** exists to force zero-mean when that fails, and is why `BF16 AdamW: 4 B` (no fp32 master) in [[ch-01]] §1.3 works at all.

**(b) "Does it eventually update?" — two different phenomena, opposite answers:**

| | (A) [[ch-01]] Q1 — bf16 weight update | (B) §2 — fp16 gradient |
|---|---|---|
| where it shrinks | optimizer step, `lr·grad` ≈ 1e−6 | **backward pass**, the gradient itself |
| what happens | **accumulates** in the fp32 master | becomes **0**, nothing to hold the residue |
| eventually updates? | ✅ yes | ❌ no — 0+0+0 = 0 |

(A) concretely: bf16's ULP near `w=0.7` is 0.0039, so a 1e−6 update moves nothing for ~3,900 steps, then the bf16 copy **jumps one ULP**. Discrete motion, but no information was ever lost. (B) has no such reservoir — the number was never computed.

(B) is also **selective**, not uniform: only params whose gradients sit systematically below the floor freeze at init while the rest train and the loss curve looks fine. The boson 256-expert MoE's rarely-routed experts are exactly that population.

**(c) Yes, exponent shortfall is worse — because of *where in the pipeline* the error lands**, not because range outranks precision:

```
forward → loss → backward → gradient → upcast fp32 → Adam(m,v) → fp32 master → bf16 weight
                              ↑                    └──────────── repair zone ────────────┘
                    exponent kills it here                mantissa error absorbed here
                    (no repair device downstream)     (24-bit accumulator + 1000s of steps)
```
A noisy number can be repaired; a number that never existed cannot. Practical severity too: fp16 overflow → `inf → NaN` → **the whole step is discarded**, and a poisoned master kills the run; bf16 rounding never has.

**Bounded claim:** this holds *given an fp32 master and a 16-bit choice*. Push mantissa far enough down (4-bit weights) and precision loss becomes fatal too — bf16's 7 bits are just barely enough at this operating point.

**One line:** unbiased error grows as √n while the signal grows as n, so rounding dilutes to nothing while flush-to-zero stays at 100 %; the bf16-weight case *does* eventually update because the fp32 master holds the residue, but the fp16-gradient case never does because the value was destroyed upstream of every repair device — that pipeline position, not range-vs-precision, is why exponent shortfall is worse. See read.md §2, [[ch-01]] qa Q1.

---

### Q4 — how does FP8 train at all? Its exponent is even narrower than fp16's.

**The observation is right and the numbers are worse than they look:**

| | exponent | min normal | max | dynamic range | relative precision |
|---|---|---|---|---|---|
| bf16 | 8 | 1.18e−38 | 3.4e38 | ~76 orders | 0.4 % |
| fp16 | 5 | 6.10e−5 | 65,504 | ~9 orders | 0.05 % |
| **E5M2** | 5 | 6.10e−5 | 57,344 | ~9 orders | **12.5 %** |
| **E4M3** | **4** | **0.0156** | **448** | **~4.5 orders** | **6.25 %** |

**Answer: FP8 does not solve the range problem — it moves the exponent out of the number.** read.md L107: *"each tensor gets a dynamic scale factor μ."* The stored value is `real = fp8 × μ` with μ in fp32, so:

```
effective exponent = 4 bits inside fp8   +   fp32 scale μ
                     (spread WITHIN a tensor)  (WHERE the tensor sits)
```

- **Why that works — dynamic range lives *between* tensors, not *within* one.** Across layers/steps/roles gradients span 10–20 orders; **inside one tensor at one step it is typically 3–4 orders**, which E4M3's 4.5 orders covers. Divide each tensor by its own μ to pull it near the origin, then the narrow 8-bit window only has to encode relative spread. Hence μ must be **per-tensor** and **dynamically updated** (read.md L107: overflow fraction >0.001 % → μ/2; 1000 clean steps → μ×2).
- **This *is* loss scaling**, returned per-tensor instead of one global scalar — the same halve-on-overflow / double-after-N rule bf16 let you delete in §2.
- **E4M3 vs E5M2 = Q1's asymmetry applied per tensor role** (read.md L90–93): forward weights/activations are bounded and well-conditioned after LayerNorm → small within-tensor spread → buy **precision** (E4M3); gradients swing across layers and steps with meaningful outliers → buy **range** (E5M2).
- **Two distinct uses of fp8 — don't conflate them:** (1) *compute* — matmul **inputs** in E4M3/E5M2 while the **accumulator stays fp32**, so [[ch-01]] Q3's rule is untouched (multiply in fp8, sum in fp32); (2) *storage* — the optimizer-state layout at read.md L97–103, where master = **fp16**, m = fp8, v = fp16 → **6 B/param** instead of 12. The 39 % saving at 175B vs 29 % at 7B (L113–115) reflects optimizer states dominating more at scale — the ch-01 §2 structure again.
- **The cost is the machinery bf16 removed:** overflow detection, scale tuning, discarded steps, and distributed scale sync (L107's global-minimum-scale trick exists to avoid per-tensor synchronization). FP8 buys speed and memory by re-purchasing that complexity.
- **Not available to boson:** read.md L121 — the FP8 tensor core is Hopper-only hardware; emulating it on A100 gives the precision loss with none of the gain. §3 is preparation for an H100 move, not a current lever.

**One line:** FP8 keeps only the *within-tensor* spread (3–4 orders) in its 4–5 exponent bits and delegates *where the tensor sits* to a per-tensor fp32 scale — i.e. it reintroduces loss scaling at tensor granularity — while E4M3-forward / E5M2-backward is the same range-vs-precision asymmetry from Q1 applied per tensor role, and fp32 accumulation is still untouched. See read.md §3, [[qa]] Q1, [[fp8-training]].

---

### Q5 — what does "spread within a tensor" mean? Does it accumulate inside the matrix?

**No — "spread" is not accumulation. Different axis.** A tensor is an array of numbers; spread is the **magnitude distribution among its elements at one instant** — a snapshot histogram, nothing stacking over time.

```
grad_W (layer 12), shape [4096, 4096] = 16.7M elements, at step 500
  largest |element| = 3.2e-3
  smallest |element| = 8.1e-7      -> spread ≈ 4,000 ≈ 3.6 orders   (WITHIN)

same step, different tensors
  grad_embedding |max| ≈ 1e-1 ; grad_LayerNorm |max| ≈ 1e-6
                                 -> ≈ 5 orders                       (BETWEEN)
```

Total range fp8 must cover = **between × within**; μ takes the former, the 4 exponent bits take the latter.

**Histogram model:** μ **slides the whole histogram** left/right into fp8's `[0.0156, 448]` window; the **exponent bits carry its width**. μ cannot narrow the width — dividing every element by the same number preserves ratios. Hence position→μ, width→exponent.

**The two axes being conflated:**

| | accumulation | spread |
|---|---|---|
| axis | **time** — across steps | **space** — across elements at one instant |
| where it came up | [[ch-01]] Q1, fp32 master, [[qa]] Q2's √n | fp8's per-tensor scale μ |
| failure | small values die when repeatedly added | large and small values don't fit one format |
| fix | fp32 accumulator | shared scale factor |

Independent problems — which is why fp8 carries **both** μ (for spread) and fp32 accumulation (for accumulation). Same shape of correction as [[ch-01]] Q3's wrong-axis fix.

**Why tensor granularity:** one fp32 μ shared by 16.7M elements costs `4/16.7M ≈ 0` per element; a per-element scale would just be a larger exponent (i.e. fp32) and save nothing. The middle ground — **block/tile-wise scales** (one per ~128 elements) — is current practice (DeepSeek-V3's fine-grained fp8, MX formats): still negligible overhead, but each block's spread is narrower.

**Narrow low-precision format + shared scale factor is the general shape of quantization** — [[ch-06]]'s SageAttention (INT8 attention) is the same skeleton.

**One line:** spread = how far apart element magnitudes are inside one tensor at one instant (space), not values piling up over steps (time); μ translates the histogram into the fp8 window while the exponent bits must cover its width, and the two axes need two different fixes — shared scale for spread, fp32 accumulator for accumulation. See read.md §3, [[ch-01]] Q1/Q3, [[qa]] Q2.
