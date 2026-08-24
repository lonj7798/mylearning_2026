<!-- qa: ch-03 — Activations and Gradient Checkpointing
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-03 Q&A — Activations and Gradient Checkpointing

Clarifying questions raised while reading [[read]]. Kernels only; full detail in `read.md`.
Earlier chapters: `ch-01/`[[qa]] (the ledger), `ch-02/`[[qa]] (precision, logit spike).

---

### Q1 — "divide an n-layer network into segments of size √n" — what does that mean?

With `n = 16`, `√16 = 4` → **4 segments of 4 layers**, storing only the 4 boundaries:

```
layer:  1  2  3  4 │ 5  6  7  8 │ 9 10 11 12 │ 13 14 15 16
                   ●            ●            ●             ●   <- stored (√n of them)
```

Normally all 16 activations must be held for backward; checkpointing keeps 4 and discards 12.

**Backward then walks segments in reverse:** start from the stored boundary, re-run that segment's forward to resurrect its 4 activations, backprop through them, free them. At every instant memory holds `√n` boundaries + `√n` live = **2√n = 8**, not 16. At `n = 10,000` that is **10,000 → 200**.

**Why √n specifically** — with `k` segments, `memory = k + n/k` (boundaries + one live segment). The two terms move oppositely, so there is a minimum: `d/dk (k + n/k) = 1 − n/k² = 0` → **`k = √n`**.

| k (segments), n=10,000 | boundaries = k | live segment = n/k | total |
|---|---|---|---|
| 10 | 10 | 1,000 | 1,010 |
| **100** | **100** | **100** | **200** ← min |
| 1,000 | 1,000 | 10 | 1,010 |

√n is the point where boundary-storage cost and recompute-window cost are exactly equal — an optimization result, not an arbitrary choice.

**Why compute is only +33%** — recomputation is **not per layer**. Each segment is recomputed exactly once, and there are `√n` segments of size `√n`, so total extra work `= √n × √n = n` = **one extra forward pass in total** (read.md L92). With transformer forward:backward ≈ 1:2, `(1+2+1)/(1+2) = 4/3` → **+33%**. Validated on a 1,000-layer ResNet: 48 GB → 7 GB (6.8×) at +30% runtime (read.md L96–99).

**In practice** frameworks skip the √n math and checkpoint **once per transformer block** (read.md L103) — not theoretically optimal, but simple and good enough, so it became the default.

**One line:** split n layers into √n segments of √n layers, keep only segment boundaries, and rebuild each segment's interior from its boundary during backward — memory `k + n/k` is minimized at `k = √n` giving 2√n, and since every segment is recomputed exactly once the total overhead is one extra forward pass (+33%). See read.md §2, [[gradient-checkpointing-chen]].

---

### Q2 — why not divide into n segments (checkpoint every layer)?

That is the **right edge of the curve, where the saving vanishes**: `k = n` → `memory = n + n/n = n + 1`. Storing n boundaries *is* storing everything — nothing was discarded — yet the +33 % recompute is still paid. Strictly dominated.

Both extremes are equally bad (n = 10,000):

| k (segments) | boundaries | live window | total |
|---|---|---|---|
| **1** (one big segment) | 1 | 10,000 | **10,001** |
| 10 | 10 | 1,000 | 1,010 |
| **100 = √n** | 100 | 100 | **200** ← min |
| 1,000 | 1,000 | 10 | 1,010 |
| **10,000 = n** (per layer) | 10,000 | 1 | **10,001** |

`k + n/k` is punished in both directions — collapse to one segment and the live window is everything; split per layer and the boundaries are everything.

**So why does practice checkpoint per block?** Because **`n` counts activation *tensors*, not blocks.** One transformer block holds ~15–20 stored tensors (LayerNorm out, Q/K/V, attention scores, softmax out, attention out, MLP up, activation fn, MLP down…). Checkpointing at block boundaries keeps 1 and discards ~20 — that is a **segment of size ~20**, not `k = n`.

```
80-block model:  n = 80 × 20 = 1,600 tensors
  theoretical optimum   k = √1600 = 40 segments × 40 tensors -> 40 + 40 = 80
  per-block in practice     80 segments × 20 tensors -> 80 + 20 = 100
```

**100 vs 80 — only 25 % off optimal**, for a scheme that is just `torch.utils.checkpoint` wrapped around a block. Not a coincidence either: the natural block boundary (~20 tensors) lands near √n (~40) for typical model sizes, so the simple choice is structurally close to the optimal one.

**One line:** `k = n` stores every boundary, so memory returns to n while still paying the recompute — both edges of `k + n/k` cost n and only √n minimizes it; practice's "per block" is not `k = n` but a segment of ~20 tensors, which happens to sit near the optimum. See read.md §2/L103, Q1.

---

### Q4 — is the summary "store one activation tensor, pay compute"? And does recompute go `x·N₁`, `x·N₁·N₂`, `x·N₁·N₂·N₃`, …?

**Two corrections.**

**(a) Only two of the three are a trade.**

| technique | memory | compute | comms | nature |
|---|---|---|---|---|
| gradient checkpointing | ↓ (√n) | **↑ +33 %** | — | **trade** |
| selective recomputation | ↓ 5× | **↑ <4 %** | — | **trade** (better exchange rate) |
| **sequence parallelism** | ↓ | **unchanged** | **unchanged** | **not a trade — deduplication** |

SP does not recompute anything; it stops 8 TP ranks from each holding the same tensor. Free. Don't file it with the other two. (Also: "one tensor" is **per segment** — √n of them overall.)

**(b) The chain is a step, not a growing product.** With `x_k = x_{k−1} @ W_k`:

```
x3 = x2 @ W3 = x0 @ W1 @ W2 @ W3     <- algebraically your formula is CORRECT
```

But it is never *computed* that way. Evaluation goes left to right, carrying only the latest result:

```
your cost model (restart each time)     actual (one step from the previous)
  x0·W1           cost 1                 x0 --W1--> x1        cost 1
  x0·W1·W2        cost 2                       --W2--> x2     cost 1
  x0·W1·W2·W3     cost 3                       --W3--> x3     cost 1
  x0·W1·W2·W3·W4  cost 4                       --W4--> x4     cost 1
  total 1+2+3+4 = 10                     total 1+1+1+1 = 4
```

Making `x2` does not restart from `x0`; it applies `W2` once to the `x1` already in hand, and `W1@W2` is never formed.

**Why this distinction is decisive:** under the growing-product model a segment of size `m` would cost `m²/2`, giving `4 segments × 10 = 40 = 2.5n` (+250 %). Linear reality gives `4 × 4 = 16 = n` — **exactly one extra forward pass, +33 %** (read.md L92). The whole √n optimization rests on that linearity; if cost were `m²`, larger segments would be penalized and √n would not be optimal.

During recompute both compute and memory are **linear in segment size** — regenerate `x13…x16` (4 ops, 4 tensors held), then backprop in reverse freeing as you go. That live window is exactly the `n/k` term in `k + n/k`.

**One line:** `x_k = x_{k−1} @ W_k` is one step, not an accumulating product — expanding the identity makes the factors look like they pile up, but evaluation advances once from the previous result, so recompute is *linear* in layer count, which is precisely why the overhead is one forward pass (+33 %). And sequence parallelism isn't in this trade at all; it deletes replicas at zero cost. See read.md §2/§4, Q1–Q3.

---

## Continued in [[qa-deep]]

- **Q3** — what Sequence Parallelism is, and why it is *not* the long-context technique
