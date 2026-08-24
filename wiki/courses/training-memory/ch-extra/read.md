<!-- chapter: ch-extra
     track: prerequisite
     kind: content
     title: Attention and the Transformer, From Scratch
     position: between [[ch-03]] and [[ch-04]]
     deps: [[ch-03]]
     feeds: [[ch-04]]
     sources: [[qkv-scaled-dot-product]], [[sqrt-dk-scaling-variance]], [[causal-mask-neg-inf]],
              [[multi-head-split-concat-wo]], [[attention-permutation-equivariance]],
              [[sinusoidal-absolute-encoding]], [[rope-rotary-position-embedding]],
              [[transformer-block-tensor-ledger]], [[pre-ln-vs-post-ln]],
              [[residual-stream-memory-backbone]], [[kv-cache-mechanism]],
              [[kv-cache-memory-formula]], [[gqa-mqa-mla-kv-heads]], [[train-vs-infer-kv-boundary]]
     created_at: 2026-08-18
-->

# Chapter Extra — Attention and the Transformer, From Scratch

> **Core insight.** Every memory number in this course is downstream of one design decision: attention routes information by computing a full `N×N` table of pairwise scores, and that table is an *activation* — a tensor the forward pass must produce and (in the naive implementation) hand to the backward pass. Everything else about a transformer — the embedding table, the three projections, the head split, the residual stream, the 4h MLP — has a byte cost that is **linear** in sequence length. The routing table alone is **quadratic**. Understanding the mechanism therefore is not background reading before the memory course; it *is* the memory course, stated at the level of the algorithm rather than the ledger.

> **Guideline.** Learn the transformer through the tensor it produces at each step, not through the diagram. For every operation ask three questions in order: (1) what is the output *shape*? (2) does the backward pass need the input, the output, or nothing? (3) does the shape contain `N²`? Answering those three questions for the ~16 tensors of one block reconstructs Korthikanti's `34·s·b·h + 5·a·s²·b` coefficient from first principles — and tells you immediately which lever ([[ch-03]] checkpointing, [[ch-04]] streaming kernels, [[ch-07]] parallelism) can touch which term.

---

**Why this chapter exists.** It was inserted after [[ch-03]] and before [[ch-04]] because the memory analysis in ch-04 assumes the mechanism. This chapter builds the mechanism from tokens up, then hands off explicitly in §8. Nothing here contradicts the sibling chapters; where a number already appears in `ch-03/read.md`, `ch-04/read.md` or their `qa` pages, it is reused verbatim and cross-referenced rather than recomputed.

**Reference configuration.** Every byte figure in this chapter uses one fixed setup unless stated otherwise, chosen to match [[ch-03]]'s figure and [[ch-04]]'s `qa`:

```
B = 1          batch (micro-batch per GPU)
T = s = 4096   sequence length
h = d_model = 4096
a = n_heads = 32       ->  d_head = h/a = 128
d_ff = 4h = 16384
dtype = bf16 = 2 bytes/element
L = 80 blocks
```

Three unit tensors recur so often they deserve names:

| Name | Shape | Elements | Bytes | |
|---|---|---|---|---|
| **HID** | `[B, T, h]` | 16,777,216 | 33,554,432 B | **33.55 MB** — one snapshot of the residual stream |
| **ATTN** | `[B, a, T, T]` | 536,870,912 | 1,073,741,824 B | **1.07 GB** — the routing table |
| **MLP** | `[B, T, 4h]` | 67,108,864 | 134,217,728 B | **134.22 MB** |
| LNST | `[B, T] × 2` fp32 | 8,192 | 32,768 B | 32.8 KB — LayerNorm saved (mean, rstd) |

Note `s·b·h = 16,777,216` elements, so **one "sbh byte-unit" = 16,777,216 B = 16.78 MB**. Korthikanti's coefficients in §8 are multiples of that unit.

**Conventions — read these once, they resolve most cross-chapter arithmetic disputes.**

1. **FLOPs.** One multiply–accumulate (MAC) = **2 FLOPs**. So an `(m×k)·(k×n)` matmul costs `2·m·n·k` FLOPs. Every FLOP count in this chapter, in [[ch-04]], and in the figures uses this convention, so the numbers compose across chapters. (Papers that count a MAC as 1 FLOP will report exactly half of everything here.)
2. **Bytes.** All byte counts are **exact integers**; the abbreviations attached to them are **decimal** — `MB = 10⁶ B`, `GB = 10⁹ B`, `TB = 10¹² B` — so `33,554,432 B` is written `33.55 MB` and `1,073,741,824 B` is written `1.07 GB`. Where a number happens to be a round **binary** quantity and the distinction matters (KV-cache sizes, cos/sin caches, ALiBi biases) the binary unit is named explicitly: `MiB = 2²⁰ B`, `GiB = 2³⁰ B`, `TiB = 2⁴⁰ B`. The two never appear inside one arithmetic step without a conversion.
3. **A known inconsistency in the siblings, stated so you can read them safely.** `ch-04/read.md` §5 (L211–216) prints `8 MB / 128 MB / 2 GB` per head and `8 GB / 128 GB / 2 TB` whole-model — those are **binary** quantities carrying **decimal** labels (they are exactly 8 MiB / 128 MiB / 2 GiB and 8 GiB / 128 GiB / 2 TiB). This chapter's §4.4 restates the same rows with the units named correctly; nothing numeric differs, only the label.
4. **Symbols.** `h = d_model` and `a = n_heads` for the whole chapter, matching Korthikanti's `34·s·b·h + 5·a·s²·b`. The head count is **always** `a`, never `h`. The only exceptions are (i) the verbatim Vaswani block quote in §4.1 and the verbatim Table-3 config in §4.5, where the paper's own `h` means the head count, and (ii) one sentence in §4.5's figure callout describing that figure's own labelling — all three are flagged where they appear. Watch for this when reading anything else: Vaswani, HuggingFace configs (`num_attention_heads`) and most blog posts use `h` for heads, while the memory literature (Korthikanti, Megatron) uses `h` for hidden size.

> **⚠ Scope note for boson / Lina TMR — read this before you carry any of it home.** Everything in this chapter describes **standard softmax attention**: a full `N×N` score matrix is materialized, and that matrix is the `5·a·s²·b` term. boson's attention layers are **GDN linear-attention** with `CP=1` hard-asserted, and linear attention has **no `N×N` score matrix at all** — so the `O(N²)` activation story, the `5as²b` coefficient, the `s = 34h/(5a)` crossover, and the "attention dominates past ~870 tokens" conclusion are all statements about the *baseline GDN replaces*, not about the model you are training. Learn them anyway: they are the reason GDN exists, they are what every kernel chapter ([[ch-04]]–[[ch-06]]) is arguing about, and any softmax-attention layer you interleave pays them in full. Just do not transplant the quadratic intuition into boson's own budget unmodified — §8 restates the caveat at the point where the ledger is handed off.

---

## 1. From Tokens to Vectors

### 1.1 Tokenization: text becomes integers

A language model never sees characters. A tokenizer (BPE, SentencePiece, tiktoken) segments the input string into subword units drawn from a fixed vocabulary of size `V`, and emits their integer indices:

```
"보험료가 얼마인가요"  ->  [12345, 887, 40219, 61, 9982]   # 5 token ids
```

Two properties matter downstream. First, `V` is fixed at training time and is a *count*, not a dimension — boson/Lina TMR uses **V = 248,000**, unusually large because Korean morphology plus insurance-domain terminology needs the coverage. Second, the ids carry no geometry: id 12345 is not "closer" to 12346 than to 900. All semantic structure has to be learned, and the place it gets learned is the embedding table.

### 1.2 The embedding table `[V, d_model]`

The embedding is one trainable matrix `E ∈ ℝ^{V × d_model}`. The forward operation is a **row lookup**, not a matrix multiply:

```
token id 12345  ->  E[12345]  ->  a vector in ℝ^{d_model}
```

For a sequence of `T` tokens this produces `X ∈ ℝ^{B × T × d_model}` — the tensor called HID above. From this point until the loss head, **the shape never changes**: `[B, T, h]` is the width of the pipe for all L blocks. That invariance is the subject of §6.3.

`d_model` (also written `h`, "hidden dimension") is the *representational* axis. `V` is the *count* axis. Conflating them is the single most common early confusion, and it is already resolved in [[ch-02]]'s `qa-deep-2` Q7: `E : [V, h]`, where **V picks which row and h is how long the row is**. The same page computes the consequence — at `V = 248,000`, `h = 4096`, the embedding matrix alone holds ≈ 1.02 × 10⁹ parameters, and the loss head expands `h → V` exactly once at the end, a ~60× inflation that *is* the logit spike of [[ch-02]] §4.

### 1.3 What a dimension "means"

A useful and honest answer: individual coordinates of `d_model` usually mean nothing in isolation. What is meaningful is *direction*. The model learns a basis in which linear directions correspond to features, and because there are far more useful features than `d_model = 4096` orthogonal directions, features are stored in **superposition** — nearly-orthogonal directions sharing the same coordinates. This is why you can add two embeddings and get something meaningful, why the residual stream in §6.3 works as a shared communication bus, and why an "attention head" can read one subspace while ignoring another.

For this course the operational consequence is narrower and concrete: `d_model` is a *width*, and every activation tensor in a transformer is some reshape of `[B, T, h]` or an expansion of it (`[B, T, 4h]` in the MLP) — **except one**, the `[B, a, T, T]` score tensor, which is the only tensor whose shape contains `T` twice. Every memory pathology in this course traces back to that exception.

---

## 2. Attention as a Soft Dictionary Lookup

### 2.1 The lookup you actually want, and why it is illegal

Imagine you could do a hard lookup: for query token `i`, pick the single most relevant token `j*` and copy its content.

```
j* = argmax_j  score(i, j)
o_i = v_{j*}
```

This is exactly what you want semantically — "the pronoun *it* should read from the noun it refers to". It is also untrainable. `argmax` is piecewise constant, so its derivative is **zero almost everywhere** and undefined at the ties; no gradient flows back into whatever produced the scores, so the model can never learn to score better. As [[qkv-scaled-dot-product]] puts it: attention is a *differentiable* dictionary lookup — hard argmax is replaced by a softmax-weighted convex combination over **all** values.

```
o_i = Σ_j  p_ij · v_j      with  p_ij ≥ 0,  Σ_j p_ij = 1
```

Because the weights are non-negative and sum to 1, `o_i` is a **convex combination** — it lies in the convex hull of the value vectors. Attention can interpolate between values; it can never extrapolate past them. (This is why the value path needs `W_O` after it: the output projection is what maps the convex hull back into a direction the residual stream can use.)

### 2.2 Why three projections and not one

Given `X ∈ ℝ^{N × d_model}`, define three learned linear maps:

```
Q = X W_Q      W_Q ∈ ℝ^{d_model × d_k}      Q ∈ ℝ^{N × d_k}
K = X W_K      W_K ∈ ℝ^{d_model × d_k}      K ∈ ℝ^{N × d_k}
V = X W_V      W_V ∈ ℝ^{d_model × d_v}      V ∈ ℝ^{N × d_v}
```

`d_k` must match between Q and K because they are dot-producted. `d_v` is free and only has to match `W_O`'s input width. In practice all three are set to `d_head = d_model / a`, with `a` the head count (§4).

The roles, in one line each ([[qkv-scaled-dot-product]]): **W_Q = "what am I looking for", W_K = "what do I advertise", W_V = "what content do I transmit."** K lives in the geometry Q is compared against; V lives in the geometry `W_O` maps back into the residual stream. They are *different* geometries on purpose — matching and transmitting are different jobs.

The sharpest way to see the necessity is to ask what happens if you delete the projections and score with `X Xᵀ` directly. Three independent things break:

1. **Symmetry.** `XXᵀ` is symmetric, so `S_ij = S_ji`. But linguistic relations are directional: "it" should attend strongly to its antecedent without the antecedent attending equally back. A symmetric score matrix cannot express a one-way edge.
2. **Diagonal dominance.** After RMSNorm every row has the same norm, `‖x_i‖ = √d_model`. Cauchy–Schwarz then gives `x_i·x_j ≤ ‖x_i‖‖x_j‖ = ‖x_i‖² = (XXᵀ)_ii`, with equality iff `x_i = x_j`. So the diagonal is the **strict row maximum** for every row — softmax puts almost all mass on the self-loop and attention degenerates to the identity map. It does nothing.
3. **No learnable routing.** `XXᵀ` is a fixed function of the input. There are no parameters with which to learn *what to attend to*. Routing would be frozen forever.

Three separate matrices fix all three at once: `W_Q ≠ W_K` breaks symmetry, breaks the Cauchy–Schwarz argument (the norms are no longer equalized in the score geometry), and supplies the parameters.

### 2.3 The formula, and its four-step execution

Vaswani et al. 2017, Equation 1, verbatim:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Executed as four steps, with exact shapes and FLOP counts for batch `B`, single head:

| Step | Operation | Output shape | Cost |
|---|---|---|---|
| 1 | `S = Q Kᵀ` | `(B, N, N)` | `2·B·N²·d_k` FLOPs |
| 2 | `S_scaled = S / √d_k` | `(B, N, N)` | elementwise |
| 3 | `P = softmax(S_scaled, dim=-1)` | `(B, N, N)` | row-wise; each **row sums to exactly 1.0** |
| 4 | `O = P V` | `(B, N, d_v)` | `2·B·N²·d_v` FLOPs |

Row `i` of the output is `o_i = Σ_j P[i,j]·v_j`. (The factor 2 in the FLOP counts is the standard multiply-add convention; it is the same convention [[ch-04]] uses, so the numbers compose.)

Steps 1–3 are where the memory story begins: two tensors of shape `(B, N, N)` exist at once — the scaled scores and the softmax output — and step 4 needs `P` while the backward pass needs `P` as well. [[ch-04]] §1.1 picks up exactly here.

### 2.4 Worked example: three keys, `d_k = 4`

Take `d_k = 4` so `√d_k = 2`, one query, three keys, `d_v = 2`:

```
q1 = [1, 0, 1, 0]
k1 = [1, 0, 1, 0]     k2 = [0, 1, 0, 1]     k3 = [1, 1, 0, 0]
v1 = [1, 0]           v2 = [0, 1]           v3 = [1, 1]
```

Raw dot products: `q1·k1 = 2`, `q1·k2 = 0`, `q1·k3 = 1`. Scaled by `1/2`: `1.0`, `0.0`, `0.5`. Exponentiate:

```
exp(1.0) = 2.718282
exp(0.0) = 1.000000
exp(0.5) = 1.648721
sum      = 5.367003
```

```
A_1 = [0.506480, 0.186324, 0.307196]        (sums to 1.000000)
      (exact: 0.50648040, 0.18632372, 0.30719589)
o_1 = 0.506480·[1,0] + 0.186324·[0,1] + 0.307196·[1,1]
    = [0.813676, 0.493520]
```

Sanity check on the convex-hull claim: both coordinates of `o_1` lie in `[0, 1]`, the range spanned by `v1, v2, v3`. Attention interpolated; it did not invent a new direction.

### 2.5 Where `1/√d_k` comes from — the whole derivation

Now run the *same* example without the scaling. Scores `2, 0, 1`:

```
exp(2) = 7.389056
exp(0) = 1.000000
exp(1) = 2.718282
sum    = 11.107338
weights = [0.665241, 0.090031, 0.244728]
```

The top weight jumped from **0.506480 → 0.665241** by removing one division — and this is at `d_k = 4`, a toy width. The distribution is already visibly sharper. Extrapolate to `d_k = 128` and the sharpening becomes saturation.

**The variance argument.** Vaswani's footnote 4, verbatim:

> "To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = sum_{i=1}^{d_k} q_i k_i, has mean 0 and variance d_k."

Full steps:

```
E[q·k]   = Σ_i E[q_i]·E[k_i] = 0                     (independence, zero mean)
Var(q·k) = Σ_i Var(q_i k_i) = Σ_i E[q_i²]E[k_i²]
         = Σ_i 1·1 = d_k
sd(q·k)  = √d_k
Var(q·k / √d_k) = d_k / d_k = 1                      (unit variance at EVERY d_k)
```

So raw score magnitude grows as `√d_k`, and dividing by `√d_k` restores unit variance *at any head dimension*. The relevant constants:

| `d_k` | `√d_k` |
|---|---|
| 64 | 8 |
| 128 | 11.3137 |
| 512 | 22.6274 |

**Why saturation is fatal, quantitatively.** Take `d_k = 64` (so `√d_k = 8`) and two keys whose raw dot products are `s1 = 24.0` and `s2 = 16.0`. Both sit inside `±3σ = ±24`, i.e. these are entirely ordinary draws, not outliers.

```
UNSCALED:  softmax over {24, 16}
           p1 = 1/(1 + e^-8) = 0.99966,  p2 = 0.00034
           softmax Jacobian diagonal  p1(1-p1) = 3.3524e-4

SCALED:    softmax over {3.0, 2.0}
           p1 = 1/(1 + e^-1) = 0.731059,  p2 = 0.268941
           p1(1-p1) = 0.196612

RATIO = 0.196612 / 3.3524e-4 = ~586x more gradient into the score
```

The softmax Jacobian is `∂p_i/∂s_j = p_i(δ_ij − p_j)`. As `p_i → 1` **every** entry goes to zero. And this Jacobian is the *only* path by which gradient reaches `W_Q` and `W_K`. A saturated head therefore stops learning **where to look** — it is frozen into whatever routing its random initialization happened to produce. The forward pass still runs; the head is simply no longer trainable.

**At realistic N it is worse.** The maximum of `N` i.i.d. `N(0,1)` scores is ≈ `√(2 ln N)`:

```
N = 1,024   ->  3.72
N = 32,768  ->  4.56
```

Unscaled at `d_k = 128` those become `11.3137 × 3.72 = 42.1` and `11.3137 × 4.56 = 51.6` — softmax logit gaps of **tens of nats**, which is exactly one-hot. That is the regime Vaswani describes in §3.2.1, verbatim:

> "While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k. We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."

**The fine print, which matters in 2026.** The `Var = d_k` derivation assumes unit-variance independent components. Real Q and K come from `X·W_Q` after LayerNorm/RMSNorm, so the assumption holds *approximately at initialization* (with standard `1/√d_model` weight init) and **degrades during training** as the weights move. `1/√d_k` is the init-time-correct constant, held fixed thereafter. This is precisely why several modern models add **QK-norm** — an RMSNorm applied to Q and K before the dot product — to re-enforce the condition the constant assumes. See [[sqrt-dk-scaling-variance]].

> **▶ Interactive companion — [`figures/attention-step-by-step.html`](figures/attention-step-by-step.html) (panels 1–5)**
> *Every intermediate number on screen, computed in-page from `x`, `W_Q`, `W_K`, `W_V`.* The page runs one deliberately tiny example — the 6-token Korean sentence **"나 는 어제 책 을 읽었다"** at `N = 6`, `d_model = 4`, `d_head = d_k = d_v = 4`, `a = 1`, `√d_k = 2`, bf16 — which **panel 1** lays out as config chips beside the real-model contrast (`d_model = 4096`, 32 heads, `s = 4096` → one score tensor of `1,073,741,824` — the figure's prose calls these 숫자, but the quantity is **bytes**; it is `536,870,912` bf16 elements, i.e. this chapter's ATTN unit tensor at 1.07 GB).
> **Panel 2** is the stepper: 8 stages over 42 steps (embedding lookup → Q/K/V projection → `q·k` → `÷ √d_k` → causal mask → softmax → `Σ p·v` → the `[B,a,s,s]` memory question), with a clickable query-token selector (default token 5, "읽었다") and every dot product written out as a four-row table (operand A / operand B / 곱 / 합) rather than asserted. The row-5 numbers it produces — `score[5][3] = 3.2344` as the row max, `÷2 = 1.6172`, softmax `0.2282, 0.0943, 0.1410, 0.3310, 0.0807, 0.1248` summing to exactly 1, output `o_5 = [1.0134, 0.5834, 0.4033, 0.2206]` — are the §2.3 four-step recipe executed by hand.
> **Panel 3** is the live state view driven by that same stepper: per-token `x/q/k/v` on the left, and on the right the `6×6` matrix advancing through raw → scaled → masked → probabilities.
> **Panel 4** is §2.5 in full. Its left canvas plots the theoretical `sd = √d_k` curve against **6,000 Box–Muller samples** (seed 20250818) at `d_k = 1 … 512`, so "variance equals `d_k`" is measured rather than asserted (measured ≈ 11.3 vs `√128 = 11.3137`), with a green line at 1.0 showing what the division restores. Its right column carries the Vaswani footnote-4 derivation, the `√d_k` table (`4 → 2.0000`, `64 → 8.0000`, `128 → 11.3137`, `512 → 22.6274`), and the red **gradient-death** box: at `d_k = 64` with scores `24` vs `16`, unscaled `p = 0.99966` gives Jacobian `p(1−p) = 3.3524e-4` while scaled (`3.0` vs `2.0`) gives `0.196612` — the **586×** gap, rendered as a number. Below it sits the §2.4 example itself as the "Vaswani 검증 예제" table, computed at runtime with `q = [1,0,1,0]`, the three keys and three values verbatim, printing scaled weights **`0.506480 / 0.186324 / 0.307196`** (sum `1.000000`) against unscaled `0.665241 / 0.090031 / 0.244728`, output `[0.813676, 0.493520]`, and `e`-values `2.718282 / 1.000000 / 1.648721` summing to `5.367003` — next to a scaled/unscaled toggle that redraws row 5's softmax green or red (`0.3310` → `0.5156`, a `1.56×` sharpening).
> **Panel 5** is §2.2's three-failure argument as three buttons on the same grid: 제대로 (`S = (XW_Q)(XW_K)ᵀ`, asymmetric — `S[5][3] = 3.2344` vs `S[3][5] = 0.2244`, a `14.4×` gap), `XXᵀ` (exactly symmetric — `S[5][3] = S[3][5] = 0.28`), and RMSNorm-then-`XXᵀ` (diagonal pinned at `4.00 = d`, self-attention probability `0.5366` against a uniform `0.1667`). A fixed callout underneath handles the fourth case — dropping `W_V` — and why `d_k` must match across Q/K while `d_v` is free.

---

## 3. Causal Masking — and the Invariant It Creates

### 3.1 The mask

A decoder-only language model predicts token `i+1` from tokens `≤ i`. Attention as defined so far lets every position see every other position, including the future — which would let the model read the answer. The fix is one additive matrix applied **before** the softmax:

```
M_ij = 0      if j <= i
M_ij = -inf   if j > i

Attention_causal(Q, K, V) = softmax( (QK^T + M) / sqrt(d_k) ) · V
```

Equivalently `S_ij = q_i·k_j/√d_k` for `j ≤ i` and `−inf` for `j > i`. Because `e^{−∞} = 0`, this is a **hard, exact** mask, not a soft penalty: after softmax, row `i` has exactly `i+1` non-zero entries and they still sum to 1. Vaswani §3.2.3, verbatim:

> "We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −inf) all values in the input of the softmax which correspond to illegal connections."

**The off-by-one that silently ruins a run:** the mask must include the diagonal — `j ≤ i`, not `j < i`. A token must be allowed to attend to itself. `torch.triu(ones(N, N), diagonal=1)` marks exactly the illegal region (strictly above the diagonal); `diagonal=0` would additionally forbid self-attention.

### 3.2 The dtype trap

The mask value is not a free choice:

| dtype | `torch.finfo(dtype).min` | is `-1e9` representable? |
|---|---|---|
| float16 | −65504 | **No** — casts to `-inf` |
| bfloat16 | −3.3895e38 | yes |
| float32 | −3.4028e38 | yes |

Correct code:

```python
attn = attn.masked_fill(~causal_mask, torch.finfo(attn.dtype).min)
```

Legacy GPT-2 / early HuggingFace code used `torch.where(causal_mask, attn, torch.tensor(-1e9, dtype=attn.dtype))`, which **breaks under fp16** because `-1e9` overflows the fp16 range.

And why `finfo.min` beats a literal `-inf`: a row that ends up fully masked (padding rows, some sliding-window configurations) gives `Σ e^{−∞} = 0`, so softmax returns `0/0 = NaN` and the NaN propagates through the entire backward pass. `finfo.min` yields a finite, uniform row instead — wrong but harmless. See [[causal-mask-neg-inf]].

### 3.3 The invariant: past K and V are frozen

This is the most important consequence in the chapter, and it is a *structural* fact, not an optimization.

```
k_j = W_K x_j        v_j = W_V x_j
```

`k_j` and `v_j` depend on token `j` **alone**. And under the causal mask, `x_j` — the residual-stream state at position `j` after any number of blocks — depends only on positions `≤ j`. Therefore:

> **Appending token `t+1` changes nothing about `k_1 … k_t` or `v_1 … v_t`.**

Past keys and values are **immutable**. Two enormous things follow:

1. **Teacher forcing is legal.** `T` predictions computed in one parallel forward pass are *identical* to `T` sequential autoregressive steps. This is what makes training a transformer parallel over the sequence axis at all — and it is why §7.6 can state that training has no KV cache.
2. **The KV cache is valid.** Recomputing immutable quantities is pure waste. §7 is the full accounting.

### 3.4 The half of the matrix that is structurally dead

Causal attention only ever needs the lower triangle — about `N²/2` entries. A dense kernel nevertheless stores and computes all `N²`. Under Korthikanti accounting the causal structure does **not** reduce the `5·a·s²·b` term at all for a dense kernel; only a causal-aware kernel converts the structural sparsity into real bytes saved. FlashAttention-2 does exactly that: it skips KV blocks entirely above the diagonal (≈50% of attention computation), materializes neither the mask nor the score matrix, and masks the diagonal blocks inside the SRAM tile. As [[causal-mask-neg-inf]] frames it: **half the `N×N` score matrix is structurally dead, and the kernel — not the model — decides whether you pay for it.** That is [[ch-04]]'s thesis one chapter early.

The mask itself is not free either. As an explicit `[1, 1, N, N]` tensor at `N = 32,768`:

| representation | bytes | |
|---|---|---|
| bool (1 B/elem) | 1,073,741,824 | **1.07 GB** |
| bf16 additive | 2,147,483,648 | **2.15 GB** |
| fp32 additive | 4,294,967,296 | **4.29 GB** |

It is resident for the whole run and re-read from HBM by every layer.

**The out-of-place copy trap** applies to both the scaling and the masking. A non-fused `S_scaled = S / sqrt(d_k)`, or a non-in-place `masked_fill`, allocates a **second** `(B, a, N, N)` tensor. At `B=1, a=32, N=32768`, bf16, that is another `32 × 32768² × 2 = 68,719,476,736 B = 68.7 GB per layer` — the attention peak doubles from one missing underscore. Correct practice: **pre-scale the query** (`Q *= d_k**-0.5`, costing only `B·N·d_model` elements) so the `N×N` tensor is produced already-scaled. FlashAttention folds the constant into the Q tile inside SRAM, so no scaled score tensor ever reaches HBM at all.

> **▶ Interactive companion — [`figures/attention-step-by-step.html`](figures/attention-step-by-step.html) (panel 2 stage 5, panel 3, panel 6)**
> *The mask as an additive matrix, and the triangle nobody needs.* Stage **5 · causal mask** of the panel-2 stepper renders the `6×6` scaled score table again with the strictly-upper cells replaced by `−∞`, counts the survivors — **21 of 36 cells alive**, i.e. `N(N+1)/2 = 21` against `N² = 36` — and prints the correct PyTorch idiom `torch.finfo(attn.dtype).min` alongside the dtype limits that motivate it (`float16.min = -65504`, `bfloat16.min = -3.3895e38`, `float32.min = -3.4028e38`, and the note that `exp` overflows fp16 above `x > 11.09`). That is §3.1 and §3.2 as rendered numbers rather than warnings.
> **Panel 3** shows the same masking as a state transition on the live grid — its mode advances `raw → scaled → maskedrow → masked → probrow → prob`, with a four-swatch legend (아직 계산 안 됨 / 방금 계산됨 / causal mask `−∞` / softmax 확률), so you watch the upper triangle go dead one row at a time and then watch the surviving row renormalize to 1.
> **Panel 6** is the §3.4 payoff: the finished causal probability heatmap captioned `[B, a, s, s] = [1, 1, 6, 6]`, `36 원소 × 2 B = 72 B`, next to a sequence-length selector (`s = 6 / 512 / 1,024 / 2,048 / 4,096 / 8,192 / 16,384 / 32,768`, default 4,096) driving four cards — elements per layer, bytes for one layer, bytes for 80 layers, and the ratio against the toy (`1.49e+7×` at `s = 4096`). At `s = 4096, a = 32`, bf16 it reads **1.07 GB per layer → 85.90 GB across 80 layers**; at `s = 32768` it reads **68.72 GB per layer**, which is the same `68.7 GB` the out-of-place-copy trap above doubles. A closing red callout states why the softmax backward needs `P` (`dS = P ⊙ (dP − rowsum(dP ⊙ P))`) and why recomputing it is cheap — the FlashAttention argument, one chapter early.

---

## 4. Multi-Head Attention

### 4.1 The equations

Vaswani §3.2.2, verbatim:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
with W_i^Q ∈ R^(d_model × d_k), W_i^K ∈ R^(d_model × d_k),
     W_i^V ∈ R^(d_model × d_v), and W^O ∈ R^(h·d_v × d_model)
```

Vaswani writes `h` for the head count; this chapter uses `a`, because `h` is already `d_model`. Inside this block quote — and in the verbatim Table-3 config in §4.5, and in one sentence of §4.5's figure callout describing that figure's own labelling — `h` is the paper's head count. Those are the three exceptions listed in §1's Conventions note; in the chapter's own prose the head count is **always** `a`, and `h = d_model = 4096`.

And the motivation, verbatim:

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."

> "Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."

The first quote is the *why*: one head computes **one** softmax-weighted average per position, and averaging destroys the ability to represent several relations at once. If position `i` needs both "the subject of my clause" and "the entity three sentences back", a single head must blend them into one convex combination and lose both.

### 4.2 The tensor pipeline, exactly

The "Concat" in the equation is not a concatenation of separate tensors in any real implementation. The heads are fused into one matrix from the start:

```
X                       (B, N, d_model)
  @ W_Q / W_K / W_V     each (d_model, d_model)   [a heads FUSED into one matrix]
Q, K, V                 (B, N, d_model)
  .view(B, N, a, d_head).transpose(1, 2)
Q, K, V                 (B, a, N, d_head)          d_head = d_model / a
  S = Q @ K^T / sqrt(d_head)
S                       (B, a, N, N)
  P = softmax(S, dim=-1)
P                       (B, a, N, N)
  O = P @ V
O                       (B, a, N, d_head)
  .transpose(1, 2).contiguous().view(B, N, a*d_head = d_model)   <- THIS IS THE "CONCAT"
  @ W_O                 (d_model, d_model)
out                     (B, N, d_model)
```

The concat is a **transpose + reshape of one contiguous buffer**, not a copy of separate tensors. That is worth internalizing because it explains why head count is nearly free in compute and why `.contiguous()` shows up in every implementation (the transpose makes the buffer non-contiguous; `view` requires contiguity).

### 4.3 Heads are free in parameters and free in FLOPs

**Parameters.** The attention block holds four `d_model × d_model` matrices (`W_Q, W_K, W_V, W_O`):

```
attention params = 4 · d_model²      -- regardless of a

d_model = 512   ->  4 × 512²  = 1,048,576    ≈ 1.05M per layer
d_model = 4096  ->  4 × 4096² = 67,108,864   ≈ 67.1M per layer
```

Changing `a` from 8 to 32 changes this by **exactly zero**.

**FLOPs.** The head count cancels:

```
QK^T costs  2·B·a·N²·d_head = 2·B·N²·(a·d_head) = 2·B·N²·d_model
P·V  costs  the same
```

Head count is **FLOP-invariant exactly**, not approximately.

### 4.4 But heads are *not* free in memory

The score tensor is `(B, a, N, N)`. In bf16:

```
bytes = B · a · N² · 2      -- LINEAR in a (the HEAD COUNT), at constant FLOPs
```

Read that `a` carefully: it is the head count, not the hidden width. Nothing here is linear in `h = d_model` — `h` never appears in the score tensor's shape at all, which is precisely why §8.2's `5·a·s²·b` term carries `a` and not `h`.

The arithmetic intensity is exactly **`2·d_head` FLOPs per stored element (= `d_head` FLOPs per stored byte in bf16)** — `QKᵀ` costs `2·B·a·N²·d_head` FLOPs under the multiply-add convention of §2.3 and writes `B·a·N²` elements. At the reference config that is `256` FLOPs per element, `128` FLOPs per byte. Doubling `a` halves `d_head`, therefore halves intensity and **doubles** the stored bytes for identical compute. [[multi-head-split-concat-wo]] states the conclusion bluntly: **multi-head attention is FLOP-neutral and memory-expensive.**

Concretely, for a 7B-class model (`d_model = 4096`, `a = 32`, `d_head = 128`, `L = 32`), bf16, `B = 1` — the same rows [[ch-04]] §5 quotes, here with the units named explicitly (binary: MiB/GiB/TiB):

| N | per head | per layer | whole model |
|---|---|---|---|
| 2,048 | 8,388,608 B = 8 MiB | 268 MB | 8.6 GB |
| 8,192 | 134,217,728 B = 128 MiB | 4.3 GB | 137 GB |
| 32,768 | 2,147,483,648 B = 2 GiB | 68.7 GB | 2.2 TB |

(Unit reconciliation, per the Conventions note in §1: `ch-04/read.md` §5 L211–216 prints the **per-head** column as `8 MB / 128 MB / 2 GB` and the **whole-model** column as `8 GB / 128 GB / 2 TB`. Both columns carry decimal labels on binary quantities — the per-head column is exactly `8 MiB / 128 MiB / 2 GiB`, matching this table's per-head column, and the whole-model column is exactly `8 GiB / 128 GiB / 2 TiB`, which in decimal is the `8.6 GB / 137 GB / 2.2 TB` printed here. No number differs between the chapters; only the labels do.)

Compare against Q, K, V themselves at `N = 32,768`, `d_model = 4096`, `B = 1`, bf16, per layer:

```
Q + K + V   = 3 × 32768 × 4096 × 2 B = 805,306,368 B  =  805 MB
              (each of Q, K, V = 268,435,456 B = 268 MB)
score tensor = 32 × 32768² × 2 B     = 68,719,476,736 B = 68.7 GB
ratio                                                    = 85x
```

At `N = 2048` that ratio is only `268 MB / 50.3 MB = 5.3×`. **The gap grows linearly in N** — which is the entire reason this course has three chapters on attention kernels and none on projection kernels.

### 4.5 How many heads? What the ablation actually says

Vaswani Table 3, base row, verified from the arXiv v7 PDF p.9:

```
N_layers = 6, d_model = 512, d_ff = 2048, h = 8, d_k = d_v = 64,
P_drop = 0.1, eps_ls = 0.1, 100K steps, dev PPL 4.92, dev BLEU 25.8, 65M params
```

(That `h = 8` is Vaswani's notation for the **head count**, per the note in §4.1 — in this chapter's symbols it is `a = 8`, and the paper's `d_model = 512` is this chapter's `h`.)

Row (A), the head ablation — EN-DE newstest2013 dev; **all rows have identical parameter counts by construction**, per §4.3:

| heads `a` | `d_k = d_v` | PPL | BLEU |
|---|---|---|---|
| 1 | 512 | 5.29 | 24.9 |
| 4 | 128 | 5.00 | 25.5 |
| **8 (base)** | **64** | **4.92** | **25.8** |
| 16 | 32 | 4.91 | 25.8 |
| 32 | 16 | 5.01 | 25.4 |

The paper's summary, verbatim: *"While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads."*

Both ends are informative. Too few heads: averaging destroys the multi-relation capacity. Too many heads: `d_head` shrinks until each head's subspace is too small to express a useful match — at `a = 32` with `d_model = 512`, `d_head = 16`. Row (B) isolates that second effect by varying `d_k` alone:

| `d_k` | PPL | BLEU | params |
|---|---|---|---|
| 16 | 5.16 | 25.1 | 58M |
| 32 | 5.01 | 25.4 | 60M |
| **64 (base)** | **4.92** | **25.8** | **65M** |

The paper's reading: *"determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial."*

**Are all heads used?** Voita et al. 2019 pruned an L0-gated trained 6-layer 8-head (48-head) transformer on WMT EN-RU:

| heads kept | BLEU change |
|---|---|
| 38/48 (79%) | −0.1 |
| 25/48 (52%) | −0.3 |
| 15/48 (31%) | −0.6 |
| 10/48 (21%) | −1.0 |

Roughly **60% of heads are prunable for under 0.3 BLEU**, and the survivors are disproportionately *positional* ("attend to t−1") and *syntactic*. (Caveat, flagged honestly: these figures are carried over from the `llm-arch` wiki and were **not** re-verified against the ACL paper for this chapter — treat the exact decimals as indicative.) The memory reading is the interesting one: since bytes scale linearly in the head count `a` at constant FLOPs (§4.4), a head that contributes nothing still costs its full `N²` slice.

> **▶ Interactive companion — [`figures/multihead-and-rope.html`](figures/multihead-and-rope.html) — Part A (A0–A3)**
> *Where the head count cancels, and where it does not.* Part A runs a hand-sized toy so every element is printable: **A0** fixes `N = 4` tokens ("The / cat / loudly / ran"), `d_model = 8`, `2` heads, `d_head = 4`, `√d_head = 2`, **no causal mask**, and prints `X` as a 4×8 integer table whose left half is a position one-hot and right half a content code — with a red callout stating up front that `W_Q/W_K/W_V` are hand-picked illustrative values, not learned weights.
> **A1** animates §4.2 in 12 steps on that buffer: `Q = XW_Q` (full 8×8 matrix shown, one element expanded by hand) → `K`, `V` → the **split, labelled "view + transpose (복사 아님)"** → per-head `S = Q⁽ʰ⁾K⁽ʰ⁾ᵀ` → `/√d_head` and softmax → `O⁽ʰ⁾ = P⁽ʰ⁾V⁽ʰ⁾` → **"Concat — 사실은 transpose + reshape"** → `@ W_O`, flagged as *the first place the heads mix at all*. The softmax arithmetic is written out (`e³ = 20.085537` over a sum of `23.085537` → `0.870049`; head 1's `e⁴·⁵ = 90.017131` over `93.017131` → `0.967748`), so the "Concat" is visibly a relabelling and `W_O` is visibly the only mixer.
> **A2** parks the two heads' softmax heatmaps side by side permanently — head 0 learned "직전 token을 본다" (positional), head 1 "통사적으로 연결된 token을 본다" (syntactic) — under Vaswani §3.2.2's "averaging inhibits this" quoted verbatim. That is §4.1's *why* as a picture.
> **A3** is the invariance ledger, and note it runs on the **toy** config, not the reference one: a table over head counts `1, 2, 4, 8` (the figure labels the head count `h`, Vaswani-style, where this chapter writes `a`) shows the parameter count pinned at `4·d_model² = 4·8² = 256` and the `QKᵀ` FLOP count pinned at `2·N²·a·d_head = 256` for **every** row, while the score-tensor element count goes `16 → 32 → 64 → 128` (8× from 1 head to 8) and the per-element arithmetic intensity `2·d_head` falls `16 → 8 → 4 → 2` in exact lockstep. Same conclusion as §4.3–§4.4, four numbers instead of 67 million. A blue callout then carries it to the reference config and to Korthikanti: `5as/h = 5·32·4096/4096 = 160`, the `s²` term at **4.7×** everything else, and the crossover `s = 34h/(5a) = 870.4` tokens — which is the handoff §8.3 formalizes. (The `85×` Q+K+V-vs-scores comparison at `N = 32,768` is this chapter's §4.4 arithmetic; it is *not* rendered in this figure.)

---

## 5. Positional Information

### 5.1 Attention has no idea what order the tokens are in

This is provable, not a heuristic. Let `X ∈ ℝ^{T×d}` with rows = tokens, and let `P ∈ {0,1}^{T×T}` be a permutation matrix, `(PX)_i = X_{π(i)}`, `PᵀP = I`.

```
(1)  (PX)W_Q = P(XW_Q) = PQ,  same for K, V
     -- projections act row-wise, so a row shuffle commutes with them

(2)  S' = (PQ)(PK)^T / sqrt(d_k) = P (QK^T) P^T / sqrt(d_k) = P S P^T
     -- i.e. S'_ij = S_{pi(i) pi(j)}

(3)  softmax(P S P^T) = P · softmax(S) · P^T = P A P^T
     -- because the row-wise denominator sum_j exp(.) is a permutation-INVARIANT sum

(4)  Out' = (P A P^T)(P V) = P A (P^T P) V = P A V = P · Out          QED
```

So `Attn(PX) = P · Attn(X)`: attention is permutation-**equivariant**, not invariant. The outputs move *with* `π`; what is invariant is the multiset of outputs. Step (4) is the crux — the `Pᵀ` produced by the conjugated score matrix cancels against the `P` in front of `V`.

The proof extends to the whole block: RMSNorm/LayerNorm and the FFN act identically and independently per row, and the residual add is elementwise, so `Block(PX) = P·Block(X)`, and stacking preserves it. **The full pre-norm transformer stack is permutation-equivariant.** Without an explicit position signal, "개가 사람을 물었다" and "사람이 개를 물었다" are the same computation with the rows shuffled.

**One important caveat.** The proof requires `P M Pᵀ = M`. A lower-triangular causal mask satisfies this only for `π = id`, so a decoder-only LM is **not** permutation-equivariant even with zero positional encoding — token `i` sees exactly `i+1` predecessors, which is a usable absolute-position count. Haviv et al. (Findings of EMNLP 2022, arXiv:2203.16634) show NoPE decoder LMs are competitive and probe out implicit absolute position; Kazemnejad et al. (NeurIPS 2023, arXiv:2305.19466) find NoPE can even beat explicit PE on length generalization. **The equivariance proof is exact for bidirectional/encoder attention** and approximate-in-spirit for decoders. See [[attention-permutation-equivariance]].

### 5.2 Sinusoidal absolute encoding (2017)

With `pos` the token index and `i` the **pair** index (`i ∈ {0, …, d/2−1}`, pair `i` occupying dimensions `2i` and `2i+1`):

```
PE_(pos, 2i)   = sin( pos / 10000^{2i/d} )
PE_(pos, 2i+1) = cos( pos / 10000^{2i/d} )
```

Frequency `ω_i = 10000^{−2i/d}`; wavelength `λ_i = 2π/ω_i`, in units of token positions. It is a **bank of d/2 clocks** at geometrically spaced rates. At `d_model = 512`:

| pair `i` | dims | `2i/d` | `ω_i` | `λ_i` (positions) |
|---|---|---|---|---|
| 0 | (0,1) | 0.0000 | 1.0 | 6.28 |
| 32 | (64,65) | 0.1250 | 0.31623 | 19.87 |
| 64 | (128,129) | 0.2500 | 0.1 | 62.83 |
| 128 | (256,257) | 0.5000 | 0.01 | 628.3 |
| 192 | (384,385) | 0.7500 | 0.001 | 6,283.2 |
| 255 | (510,511) | 0.9961 | 1.0366×10⁻⁴ | 60,611.5 |

Two corrections worth carrying forward, because the `llm-arch` wiki has both wrong: the ladder there is **mislabeled by one step** (it prints `ω = 0.1` against dims (64,65) when `ω = 0.1` belongs to dims (128,129)), and its figure's "dim 128–129 λ = 56" should be **λ = 62.83**. Also: the slowest wavelength is **60,611.5**, not 62,832. `2π × 10000 = 62,831.85` is the wavelength at exponent exactly 1.0, but the largest exponent any real pair reaches is `(d−2)/d = 510/512 = 0.9961`. Same arithmetic at `d_head = 128`: slowest pair `i = 63` has `2i/d = 126/128 = 0.9844`, `θ = 1.1548×10⁻⁴`, `λ = 54,410.1` positions.

**The property that mattered.** Shifting position by `k` is a rotation whose matrix does not depend on `pos`:

```
[ PE_(pos+k, 2i)   ]   [  cos ω_i k   sin ω_i k ] [ PE_(pos, 2i)   ]
[ PE_(pos+k, 2i+1) ] = [ -sin ω_i k   cos ω_i k ] [ PE_(pos, 2i+1) ]

full vector:  PE_{pos+k} = diag(R_k^{(0)}, ..., R_k^{(d/2-1)}) · PE_pos
```

Relative position is already a **rotation** in 2017 — RoPE's contribution is *where* the rotation is applied, not the schedule. See [[sinusoidal-absolute-encoding]].

A related property: the inner product decays with distance. `PE_pos · PE_{pos+k} = Σ_{i=0}^{d/2−1} cos(ω_i k)`, which at `d = 128` (so `d/2 = 64`) gives:

| `k` | 0 | 1 | 10 | 100 | 1000 | 8000 |
|---|---|---|---|---|---|---|
| dot | 64.000 | 62.094 | 42.820 | 30.543 | 10.178 | −0.353 |

### 5.3 RoPE: rotate Q and K instead of adding to X

RoPE ([[rope-rotary-position-embedding]]) uses the *same* frequency ladder — `θ_i = base^{−2i/d}`, pair index `i ∈ {0, …, d/2−1}`, `d = d_head` — with `base = 10000` (Llama 1/2, Mistral, Qwen2), `base = 500000` (Llama 3), up to 10⁶ for long-context finetunes. The innovation is that it **multiplies Q and K** rather than adding to the embedding.

Per pair `i`, at position `m`:

```
[ q'_{2i}   ]   [ cos m·theta_i   -sin m·theta_i ] [ q_{2i}   ]
[ q'_{2i+1} ] = [ sin m·theta_i    cos m·theta_i ] [ q_{2i+1} ]
```

Stacked over all pairs this is a block-diagonal orthogonal matrix `R_{Θ,m} ∈ ℝ^{d×d}` with `d/2` independent 2×2 blocks. It is **norm-preserving**: `‖R_m q‖ = ‖q‖` exactly (verified: difference = 0.0).

**The central identity.** Using `Rᵀ_m = R_{−m}` and `R_{−m}R_n = R_{n−m}`:

```
(R_{Theta,m} q)^T (R_{Theta,n} k) = q^T R_{Theta,m}^T R_{Theta,n} k = q^T R_{Theta,n-m} k
```

Absolute `m` and `n` vanish; **only `n − m` survives**. Numerically verified at `d = 8`, `base = 10000`, random `q, k`, `(m,n) = (17,5)`:

```
(R_m q)·(R_n k)                                  = 3.6169521324913054
q^T R_{n-m} k                                    = 3.616952132491305
Re[ sum_i q_i · conj(k_i) · e^{i(m-n)theta_i} ]  = 3.6169521324913045
```

Three routes, one number. The per-pair closed form, with `Δ = m − n`:

```
<R_m q^(i), R_n k^(i)> = (q_{2i}k_{2i} + q_{2i+1}k_{2i+1})·cos(Delta·theta_i)
                       + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i})·sin(Delta·theta_i)
```

**Note the sign**: the `sin` coefficient is `(q_{2i}k_{2i+1} − q_{2i+1}k_{2i})`, the 2D **cross product / determinant** of the pair — *not* `(q_{2i+1}k_{2i} − q_{2i}k_{2i+1})`. The relative angle mixes the pair's dot product (cos term) with its cross product (sin term). The complex form makes the structure clearest:

```
<f(q,m), f(k,n)> = Re[ sum_i q_i · conj(k_i) · e^{i(m-n)theta_i} ]
                 = sum_i |q_i||k_i| cos( angle(q_i) - angle(k_i) + (m-n)theta_i )
```

where `q_i = q_{2i} + i·q_{2i+1}` pairs consecutive real dims into `ℂ^{d/2}`.

**Minimal worked example.** One dimension pair, `q = k = (1, 0)`, `θ = 1.0` rad:

| `(m, n)` | `Δ = m − n` | dot product |
|---|---|---|
| (5, 3) | 2 | −0.416147 |
| (7, 5) | 2 | −0.416147 |
| (100, 98) | 2 | −0.416147 |

All identical, because `cos(2 rad) = −0.416147` and only `Δ` enters. This is what "relative position" means operationally: absolute index 100 and absolute index 5 are indistinguishable to the score as long as the gap is the same.

**Why this form and no other.** Impose three constraints: (1) relative dependence `⟨f(q,m), f(k,n)⟩ = g(q,k,m−n)`; (2) identity at the origin `f(x,0) = x`; (3) magnitude preservation. Polar-decompose `f(q,m) = R_f(q,m)·e^{iΘ_f(q,m)}`. Setting `m = n` with `f(x,0) = x` forces `R_f(q,m) = |q|` for all `m` — **position can only rotate, never scale**. The phase constraint `Θ_f(q,m) − Θ_f(k,n) = Θ_g(q,k,m−n)` forces `Θ_f(q,m) = Θ(q) + mθ`, since the only continuous `φ` with `φ(m) − φ(n) = h(m−n)` is linear. Hence `f(q,m) = q·e^{imθ}`. RoPE is not one option among many; it is the **unique** solution of those three requirements.

**Frequency bands and why long-context extension works.** At `d_head = 128`, `base = 10000`, training length `L = 8192`, define `r_i = L/λ_i = L·θ_i/2π` = number of full rotations completed during training:

| pair `i` | `θ_i` | `λ_i` | `r_i` | band |
|---|---|---|---|---|
| 0 | 1.0 | 6.28 | 1303.8 | fast / local |
| 16 | 1.0×10⁻¹ | 62.8 | 130.4 | fast |
| 32 | 1.0×10⁻² | 628.3 | 13.0 | mid |
| 48 | 1.0×10⁻³ | 6,283.2 | 1.30 | slow |
| 63 | 1.1548×10⁻⁴ | 54,410.1 | 0.151 | slow / global — never completes one revolution |

That last row is why naive extrapolation fails: pair 63 has seen only 15% of one cycle during training, so positions beyond `L` land on angles it has literally never observed. YaRN's fix is band-selective — leave the fast bands alone, interpolate the slow ones. At `d_head = 128`, `L = 8192`, with cutoffs `r > β = 32` untouched and `r < α = 1` fully interpolated, the boundaries land at pair `i ≈ 25.76` and `i ≈ 49.84`: **pairs 0–25 untouched, pairs 26–49 on the smooth ramp** `θ'_i = θ_i(1−γ_i) + (θ_i/s)·γ_i`, **pairs 50–63 fully scaled** to `θ_i/s`. The NTK-aware alternative rescales the base instead: `base' = base · s^{d/(d−2)}`; at `d = 128` the exponent is `1.01587`, so `s=4 → 40,890`, `s=8 → 82,685`, `s=16 → 167,199`, `s=32 → 338,097`. Llama 3's `base = 500000` at `d_head = 128` gives slowest pair `θ_63 = 2.4551×10⁻⁶`, `λ = 2,559,195.5` positions (vs 54,410.1 at base 10000) — which is why raising the base is a crude form of interpolation applied uniformly.

**Implementation, and the gotcha that silently destroys a model.** HuggingFace / GPT-NeoX style:

```python
q_rot = q * cos + rotate_half(q) * sin
# rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])
```

Original RoFormer/GPT-J pairs dims `(2i, 2i+1)` — **interleaved**. GPT-NeoX / HF-Llama pairs `(i, i+d/2)` — **split-half**. They are mathematically equivalent (same `d/2` planes, relabelled) **only if Q and K use the same convention**; mixing them silently destroys the model with no error and no NaN. Runtime overhead is ≈1–3% of the forward pass.

### 5.4 The memory contract of positional encoding — three injection points

This is the part that connects §5 to the rest of the course. Positional schemes differ by **where** they inject, and that determines whether they touch the `T×T` tensor:

| injection point | schemes | object it allocates | touches the `T×T` score matrix? |
|---|---|---|---|
| input-embedding-additive | sinusoidal, learned | `[L, d_model]` table | **No** |
| Q/K-multiplicative | RoPE, iRoPE | `[L, d_head/2]` cos/sin cache | **No** |
| attention-logit-additive | T5 RPE, ALiBi | `[H, T, T]` bias | **Yes** |

**RoPE's cos/sin cache**, shape `[L, d_head/2]` each, bytes = `2·L·(d_head/2)·bytes_per_elem`, fp32 at `d_head = 128`:

| `L` | cache | HF-style duplicated to `[L, d_head]` |
|---|---|---|
| 8,192 | 4,194,304 B = 4.00 MiB | 8.00 MiB |
| 32,768 | 16.00 MiB | 32.00 MiB |
| 131,072 | 64.00 MiB | 128.00 MiB |

Allocated **once**, shared by every layer and every head. Zero trainable parameters, zero gradient, zero optimizer state.

**A learned absolute PE table** `W_p ∈ ℝ^{L×d}` is trainable, so it pays the full mixed-precision AdamW tax from [[ch-01]] §1.3 — 16 B/param (fp32 master + 2 fp32 moments + fp32 grad), or 18 B/param counting the bf16 working copy:

```
GPT-3 (L=2048, d=12288):  25,165,824 params x 16 B = 402,653,184 B = 384.00 MiB   (18 B/param: 432.00 MiB)
L=8192, d=4096:           33,554,432 params x 16 B = 536,870,912 B = 512.00 MiB
```

Plus a hard ceiling: **position `L+1` does not exist.** Sinusoidal is a non-trainable buffer of `4·L·d` bytes in fp32 — at `L=8192, d=4096` that is `134,217,728 B = 128.00 MiB`, with zero optimizer state and zero gradient. Neither sinusoidal nor learned PE adds *activation* bytes, because PE is summed into `x` before layer 0 and no tensor grows.

**Logit-additive bias is the expensive family.** ALiBi computes `attention(q_i, k_j) = q_iᵀk_j/√d_k − m_h·|i−j|` with head-specific slopes `m_h = 2^{−8h/H}` (for 8 heads: 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256). Elegant, and extrapolates well — but it needs a `B·H·T·T` object, i.e. **the exact tensor FlashAttention exists to never materialize**:

| config | bytes | |
|---|---|---|
| B=1, H=32, T=4096, bf16 | 1,073,741,824 | **1.000 GiB per layer** |
| B=1, H=32, T=8192, bf16 | 4,294,967,296 | **4.000 GiB per layer** (10% of an A100-40GB) |
| B=4, H=32, T=8192, bf16 | — | **16.000 GiB per layer** |
| B=1, H=32, T=8192, fp32 | — | **8.000 GiB per layer** |

FlashAttention-2 has an explicit `alibi_slopes` argument for exactly this reason. A *custom* bias with no kernel support falls back to the SDPA MATH path and re-materializes the full score matrix — the silent-fallback trap of [[ch-06]], reached through the positional encoding rather than through the mask.

**RoPE's structural advantage, stated precisely:** RoPE lives **outside** the attention kernel. Q and K are rotated *before* the call, so FlashAttention/SDPA see an ordinary `(Q, K, V)` triple and keep their `O(T)` activation footprint. RoPE is the only positional scheme that is free in both parameters and kernel memory contract. The one transient to watch: an **unfused** RoPE allocates fresh `[B, H, T, d_head]` tensors for `q_rot`/`k_rot` — at `B=1, H=32, T=8192, d_head=128`, bf16, each is `67,108,864 B = 64.00 MiB`, so ~128.00 MiB per layer of avoidable transient if the pre-rotation copies are not freed before the kernel launches. A **fused** RoPE adds **zero** saved-for-backward activations, because its backward is rotation by `−mθ` — an orthogonal map with constant coefficients that needs no saved input tensor.

> **▶ Interactive companion — [`figures/multihead-and-rope.html`](figures/multihead-and-rope.html) — Part B (B1–B5)**
> *Position as a rotation, and where the rotation is applied.* **B1** runs §5.1's proof as a 5-step animation with four selectable permutations (`[0,1,2,3]`, `[2,0,3,1]`, `[3,2,1,0]`, `[1,3,0,2]`): shuffle `X` to `PX`, then show `S⁽¹⁾`, `P S Pᵀ`, and the *recomputed* `S(PX)` as three matrices with a live `max |P S Pᵀ − S(PX)|` readout that prints **exactly 0** (the toy is integer arithmetic), then `Out'` against `P·Out` with a `max |Out' − P·Out|` of `~1e−16` and a machine-epsilon explanation for why it is not literally zero. Its final step states the caveat this chapter makes in §5.1 — the causal mask breaks `P M Pᵀ = M` — citing Haviv et al. 2022 and Kazemnejad et al. 2023.
> **B2** is one RoPE pair rotated by hand on a `d = 8` toy: sliders for position `m` (0–32) and pair index `i` (0–3), a canvas drawing the original vector, the rotated `R_m q`, and the swept arc, plus a row of four clock dials showing how far each pair has turned at the same `m`. `θ = 1.0 / 0.1 / 0.01 / 0.001`, wavelengths `6.28 / 62.83 / 628.32 / 6283.19` positions — pair 0 turns a full revolution every 6.28 positions while pair 3 has moved `1.8°` after 32. A live `‖q‖` readout confirms the norm-preservation claim of §5.3 (difference ~0).
> **B3** is the centrepiece: `⟨R_m q, R_n k⟩` with `m` and `n` on independent 0–64 sliders (defaults `m = 17, n = 5`, the exact pair §5.3 verifies numerically), a **Δ-lock button** that slides both together, and a plot of the inner product as a function of `Δ` over `[−64, 64]`. Two tables carry the argument: a per-pair decomposition (`θᵢ`, the pair's dot product, its cross product, `Δθᵢ`, `cos`, `sin`, contribution, with a `Σ` row) showing exactly where absolute position cancels, and a "same `Δ`, different `(m, n)`" table where `m·θ₀` and `n·θ₀` differ wildly across six rows while the final column is **identical to 9 decimals**. Its closing callout is the memory argument: RoPE's cos/sin cache is `[L, d_head/2] × 2` fp32 = **4.00 MiB** at `L = 8192, d_head = 128` with zero trainable parameters, zero gradient and zero optimizer state, against a logit-additive `[B, H, T, T]` bias at `B=1, H=32, T=8192` bf16 = **4.000 GiB per layer**, i.e. 10% of an A100-40GB — §5.4's table in two numbers.
> **B4** draws §5.3's frequency ladder as a canvas: `log₁₀ θᵢ` against pair index `0–63` at `d_head = 128` for **both** `base = 10000` and `base = 500000`, with YaRN's `r = 32` (β) and `r = 1` (α) boundaries as dashed lines and the region right of α shaded red and labelled "이 구간은 L=8192 동안 1회전도 못 한다". Its table reproduces §5.3's band table exactly — `i = 0` (`θ = 1.0`, `λ = 6.2832`, `1303.7973` rotations) through `i = 63` (`θ = 1.1548e-4`, `λ = 54,410.1431`, `0.1506` rotations) — and a callout nails down that the slowest wavelength is **54,410.1**, not the widely-quoted `2π × 10000 = 62,831.85`, because the largest exponent any real pair reaches is `(d−2)/d = 126/128`.
> **B5** is a static panel carrying the long-context consequences: which pairs extrapolate into unseen angles (`r < 1`, i.e. `i ≥ 50` at `d_head = 128, L = 8192`), YaRN's three bands (`0–25` untouched, `26–49` on the `θ'ᵢ = θᵢ(1−γᵢ) + (θᵢ/s)γᵢ` ramp, `50–63` fully scaled) with boundaries at `i ≈ 25.76` and `i ≈ 49.84`, the NTK-aware `base' = base·s^{1.01587}` values (`s = 4 → 40,890` … `s = 32 → 338,097`), and Llama 3's `base = 500000` as coarse uniform interpolation.

---

## 6. The Transformer Block

### 6.1 Pre-LN vs Post-LN

Two update rules, verbatim:

```
Post-LN (Transformer 2017, GPT-1):
    h'  = LayerNorm(x  + Attn(x))
    h'' = LayerNorm(h' + FFN(h'))

Pre-LN (GPT-2 onward):
    h'  = x  + Attn(LayerNorm(x))
    h'' = h' + FFN(LayerNorm(h'))
```

The difference is whether the normalization sits **on** the residual highway or **inside** the residual branch. Pre-LN leaves a clean identity on the highway:

```
d x_{l+1} / d x_l = I + d Sub(LN(x_l)) / d x_l
```

Backward therefore always has a path that multiplies by exactly `I`. Post-LN forces the backward pass through `L` LayerNorm Jacobians in series.

**What Xiong et al. 2020 actually prove**, stated in their own variables (their Theorems 1 and 2, both at initialization, both about the **last** layer's FFN parameters, both in terms of total depth `L` — *not* layer index `ℓ`):

```
Post-LN:  E‖ ∂L/∂W^{(L)} ‖_F  =  O( d · sqrt(ln d) )            -- independent of L
Pre-LN :  E‖ ∂L/∂W^{(L)} ‖_F  =  O( d · sqrt(ln d / L) )        -- damped by 1/sqrt(L)
```

So the `1/√·` factor belongs to **Pre-LN**, and the argument inside it is **total depth `L`**, not the layer index. The operative fact is not the absolute magnitude but the *distribution across depth*: under Post-LN the gradient at the output end stays `O(1)` in `L` while the gradient reaching the input end is attenuated by the `L` LayerNorm Jacobians in series, so one global learning rate is simultaneously too large at the top and too small at the bottom. Pre-LN's per-layer gradients are depth-uniform up to that `1/√L` scale, so a single learning rate is well-posed everywhere. (*The depth-imbalance statement is given qualitatively here* — the two scalings above are the paper's formal results; "first-vs-last imbalance is `O(L)`" is a summary of their Figure 1/3 measurements, not a theorem, and this chapter does not rely on the exponent.) That imbalance is why the original paper needed **4,000 warmup steps**; Pre-LN makes warmup optional. Note that `figures/transformer-block-dataflow.html` panel 5 still carries the loose "`O(1/√ℓ)`" phrasing in its Post-LN callout — read the two scalings above as the authority.

The norms themselves:

```
LN(x)      = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
             mu = (1/d) sum_i x_i,  sigma^2 = (1/d) sum_i (x_i - mu)^2,  eps ~ 1e-5
             reduced over d_model, per position, per example

RMSNorm(x) = gamma * x / sqrt( (1/d) sum_i x_i^2 + eps )
             no mean centering, no beta bias, ~10% faster on GPU, empirically equal quality
```

RMSNorm as Pre-RMSNorm is the LLaMA / Gemma / Mistral / Qwen default.

**And now the claim this course cares about: the two placements cost identical activation bytes.** Both save the LayerNorm **input** (`2·sbh` per norm), i.e. Korthikanti's `4·sbh` term = `67,108,864 B = 67.11 MB` per block at the reference config (= 2 × 33.55 MB). The checkpoint tensor is `[B,T,h] = 33.55 MB` either way (pre-LN: the residual-2 sum; post-LN: the final LN output), so `2·s·b·h·L` is placement-independent. RMSNorm's only real byte saving is that its saved statistics drop from `[B,T]×2` fp32 = 32,768 B to `[B,T]×1` fp32 = 16,384 B per norm — over 2 norms × 80 blocks that is **5.24 MB → 2.62 MB**, negligible against the 1.07 GB attention matrix.

> **Normalization placement is a gradient-flow decision, not a memory decision.** ([[pre-ln-vs-post-ln]])

### 6.2 The MLP and its 4h expansion

Each block's feed-forward network expands the hidden width, applies a nonlinearity, and projects back:

```
[B, T, h] --W_up--> [B, T, 4h] --GELU--> [B, T, 4h] --W_down--> [B, T, h]
```

The `4×` ratio is held exactly across the GPT line: GPT-1 768→3072, GPT-2 1600→6400, GPT-3 12288→49152. At the reference config `4h = 16,384` and each `[B,T,4h]` tensor is `134,217,728 B = 134.22 MB` — exactly 4× the hidden state. The block saves **two** of them (up-projection output, GELU output) = `268,435,456 B = 268.44 MB` = exactly **8.0× the 33.55 MB checkpoint**, and 84% of the MLP block's 318.77 MB share. Modern SwiGLU uses `d_ff ≈ (8/3)h` but with three matrices, landing at a similar byte count.

### 6.3 The residual stream

```
x_L = x_0 + sum_{l=1..L} Delta_attn^(l) + sum_{l=1..L} Delta_ffn^(l)
```

Fully expanded for pre-LN:

```
x_L = x_0 + sum_l [ Attn_l(LN(x_{l-1})) + FFN_l( LN( x_{l-1} + Attn_l(LN(x_{l-1})) ) ) ]
```

A decoder-only transformer is **one `[B,T,h]` tensor** flowing from embedding to unembedding, with `2L` sub-layers that read it, compute, and **add back**. Nothing is overwritten. Three consequences, all load-bearing:

1. **Trainability.** `∂L/∂x_ℓ = ∂L/∂x_{ℓ+1} · (I + ∂F_{ℓ+1}/∂x_ℓ)` — the identity term guarantees a gradient path of length 0 through every block. This is the ResNet framing: learn `F(x) = H(x) − x` so that "do nothing" means `F ≈ 0`. GPT-2 initializes residual-branch output projections at scale `1/√N` (N = number of residual layers) so that the sum of `2L` additions does not blow the stream's variance at step 0.
2. **Composition.** Early writes are never overwritten, so later layers can read features written many layers back — the mechanism behind induction heads and every circuit-level result in interpretability.
3. **Cheap checkpointing.** The checkpoint is exactly **one snapshot of the stream**. That is why [[ch-03]]'s `2·s·b·h·L` formula is literally "L snapshots of the residual stream at 2 bytes an element." ([[residual-stream-memory-backbone]])

### 6.4 The ordered list of tensors a block holds for backward

This is the heart of the chapter's handoff. Pre-LN, no recompute, reference config, in dataflow order:

| # | Tensor | Shape | Bytes | Why backward needs it |
|---|---|---|---|---|
| 1 | block input `x_in` | `[B,T,h]` | 33.55 MB | **THIS IS THE CHECKPOINT** |
| 2 | LN1 saved stats (mean, rstd) | `[B,T]×2` fp32 | 32.8 KB | LN backward |
| 3 | LN1 output | `[B,T,h]` | 33.55 MB | input of the Q/K/V matmuls |
| 4 | Q projection | `[B,T,h]` | 33.55 MB | `dK` needs Q |
| 5 | K projection | `[B,T,h]` | 33.55 MB | `dQ` needs K |
| 6 | V projection | `[B,T,h]` | 33.55 MB | `dP` needs V |
| 7 | attention probs `P = softmax(QKᵀ/√d)` | `[B,a,T,T]` | **1.07 GB** | softmax VJP needs its **output** |
| 8 | context `P·V` | `[B,a,T,d]` | 33.55 MB | input of `W_O` |
| 9 | attention output after `W_O` | `[B,T,h]` | 33.55 MB | — |
| 10 | residual-1 sum | `[B,T,h]` | 33.55 MB | input of LN2 |
| 11 | LN2 saved stats | `[B,T]×2` fp32 | 32.8 KB | LN backward |
| 12 | LN2 output | `[B,T,h]` | 33.55 MB | input of `W_up` |
| 13 | MLP up-projection output | `[B,T,4h]` | 134.22 MB | GELU needs its **input** |
| 14 | GELU output | `[B,T,4h]` | 134.22 MB | input of `W_down` |
| 15 | MLP down-projection output | `[B,T,h]` | 33.55 MB | — |
| 16 | residual-2 sum = block output | `[B,T,h]` | 33.55 MB | the **next** block's checkpoint |

You do not memorize this table. You **derive** it, from one rule per operation:

| operation | what backward saves | why |
|---|---|---|
| matmul `Y = XW` | its **INPUT** `X` | because `dW = Xᵀ·dY` |
| GELU | its **INPUT** | the derivative is a function of the pre-activation |
| softmax | its **OUTPUT** `P` | the VJP is `dS = P ⊙ (dP − rowsum(dP ⊙ P))` |
| dropout | only its 1-byte **MASK** | the mask is the whole Jacobian |
| LayerNorm | its **INPUT** plus saved (mean, rstd) | stats are cheaper to save than to recompute |
| residual add `y = x + f(x)` | **NOTHING** | `∂(x+f)/∂x = I` |

One corollary worth stating explicitly, because it explains a coefficient that otherwise looks wrong: **Q, K and V projections share ONE saved input** (the LN1 output). That is why Korthikanti counts `2·sbh` once for the QKV input rather than `6·sbh`.

**The arithmetic:**

```
discarded per block (items 2-15) = 9·HID + 2·LNST + ATTN + 2·MLP
                                 = 1,644,232,704 B = 1.64 GB
checkpoint (item 1)              = 1·HID = 33,554,432 B = 33.55 MB
discarded / checkpoint           = 49.0x
block total (items 1-15)         = 1,677,787,136 B ~ 1.68 GB   -> 50.0x the checkpoint
```

Across `L = 80` blocks:

```
no checkpointing:  1,677,787,136 x 80 = 134,222,970,880 B = 134.22 GB
per-block ckpt:    2·s·b·h·L = 2·4096·1·4096·80 = 2,684,354,560 B = 2.68 GB
reduction:         50x
```

That `2·s·b·h·L` is the same formula as `ch-03/read.md` L110–116, and the 33.55 MB / 1.64 GB / 49× triple is exactly what `ch-03/figures/checkpointing.html` Panel 1 renders. This section is the derivation behind that panel.

**Reconciling this 50× with ch-03's 97× — same lever, two numerators.** `ch-03/read.md` L110–116 states the checkpointing floor as `2·s·b·h·L` against `s·b·h·L·(34 + 5as/h)` without recompute. Both chapters divide by the *same* denominator — the `2 sbh` per-block checkpoint, 33.55 MB here. They differ only in what they count above the line:

```
this chapter's 16-tensor enumeration:  1,677,787,136 B/block = 100 sbh-units + LN stats
                                       100 / 2 = 50x

ch-03 via Korthikanti's coefficient:   (34 + 5as/h)·sbh = 194 sbh-units = 3,254,779,904 B/block
                                       194 / 2 = 97x
```

The factor-of-two gap between 50× and 97× is *entirely* the outputs-vs-saved-inputs and dropout difference that §8.2 spells out in detail (Korthikanti's `5as²b` counts three `s²`-shaped tensors where the enumeration counts one, plus three dropout tensors this chapter's dropout-free reading deletes). **Neither ratio is wrong; they are ratios of different numerators.** If you arrive here from ch-03 expecting 97×, you are holding Korthikanti's published coefficient; the 50× is the same lever measured against the 16-tensor list enumerated above.

**The one tensor that behaves differently.** Item 7 grows quadratically: `[B,a,T,T]` at `T=4096` is 1.07 GB per block; at `T=8192` it is `1 × 32 × 8192 × 8192 × 2 = 4,294,967,296 B = 4.29 GB` — exactly 4×, i.e. quadratic. It is simultaneously the **largest** saved tensor and the **cheapest to reconstruct** (one matmul plus one softmax). That asymmetry is what selective recomputation ([[ch-03]] §3) and FlashAttention ([[ch-05]]) exploit. See [[transformer-block-tensor-ledger]].

> **▶ Interactive companion — [`figures/transformer-block-dataflow.html`](figures/transformer-block-dataflow.html) (panels 0–3, 5)**
> *The 16 tensors, in order, with the save-rule that produced each one.* **Panel 0** is the config that drives every number on the page: sliders for `B` (1–8), `s` (512 … 32,768), `h` (1024 … 8192), `a` (8/16/32/64), `L` (1–126) and a dtype selector (fp8 / bf16 / fp32), defaulting to exactly this chapter's reference config. Its derived chips show `d_head = h/a`, `4h`, and the byte size of each unit tensor with the multiplication substituted — `[B,s,h] = 33,554,432 B = 33.55 MB`, `[B,a,s,s] = 1,073,741,824 B = 1.07 GB`, `[B,s,4h] = 134,217,728 B = 134.22 MB`, `[B,s]×2` fp32 `= 32,768 B` — i.e. the HID / ATTN / MLP / LNST table of §1, recomputed live. **Panel 1** is a two-button mode switch (training vs inference-decode) plus a gradient-checkpointing checkbox, which change the fate of every tensor downstream.
> **Panel 2** is §6.4's table as a 16-step walkthrough. The residual stream is drawn as a left rail (`x_in → +Δ_attn → +Δ_mlp → 다음 block`, all `[B,s,h]`, with a 폭 불변 node), and each step adds one row carrying name, shape, byte count, a proportional bar, and a **fate chip** — 보존 / KV cache / 해제 / 다음 block. Item 7, `P = softmax((QKᵀ+M)/√d_head)` at `[B,a,s,s]`, is labelled **괴물** and its bar dwarfs the other fifteen; the totals cards read 1.68 GB per block (items 1–15, item 16 excluded as the *next* block's property), 1.64 GB discarded, 33.55 MB kept = **49.0×**, and 134.22 GB across `L = 80` against 2.68 GB with checkpointing. A canvas underneath plots cumulative retained bytes across the 16 steps as three curves — training, training + checkpointing, and inference-KV — with the attention-branch and MLP-branch regions shaded.
> **Panel 3** runs the *same stepper* on a printable toy (`s = 3`, `h = 4`, `a = 2`, `d_head = 2`, `d_ff = 8`, `W_V = I`, weights restricted to `0 / ±1 / ±0.5`) with every matrix element written out — LN statistics, the causal `−∞` cells, per-head `P` with row sums, concat, the `[3,8]` GELU, and the final block output next to the original `x_in` — plus one element expanded by hand at each step. It is §6.4's dataflow with the abstraction removed.
> **Panel 5** is §6.1 as a static two-column diagram: Post-LN with a red hatched normalization bar sitting **on** the residual highway (citing Xiong et al. 2020 and the 4,000 warmup steps) against Pre-LN with the highway untouched and `∂x_{ℓ+1}/∂x_ℓ = I + ∂Sub(LN(x_ℓ))/∂x_ℓ` printed alongside — under a live byte note showing both placements cost the **identical** `4·sbh = 67.11 MB`, and quantifying RMSNorm's only real saving (`5.24 MB → 2.62 MB` of stored statistics across the model).
> One thing to expect correctly: dragging `s` in panel 0 grows every linear row proportionally and the `[B,a,s,s]` row quadratically, but **the crossover does not move with `s`** — it is the `s = 34h/(5a)` chip, which reads **870.4** at the reference config and changes only when you move `h` or `a`. It is a threshold *on* `s`, not a function of it.

---

## 7. The KV Cache

### 7.1 Mechanism

At inference the model generates one token at a time. Step `t+1` computes a *single* query vector and attends over all previous keys and values. By the invariant of §3.3, `k_1..k_t` and `v_1..v_t` are unchanged from the previous step. So cache them.

```
Attention_causal(Q, K, V) = softmax( (Q K^T + M) / sqrt(d_k) ) · V,  M[i,j] = -inf for j > i
```

At decode step `t+1` there is no future key to mask, so decode kernels drop the mask entirely. The cache is **memoization over an already-pure function** ([[kv-cache-mechanism]]) — it changes nothing about the output, only about how many times each `k_j` is computed.

**The saving, counted in token-forward-passes:**

```
without cache:  1 + 2 + ... + N = N(N+1)/2
with cache:     N
ratio:          (N+1)/2

N = 1024  ->  524,800  vs  1,024  =  512.5x
```

In FLOPs, per layer, with `d` the hidden size:

```
prefill of T tokens        ~  7·B·T·d^2 + 2·B·T^2·d
one decode step at depth t ~  7·B·d^2 + 2·B·t·d

with cache    = sum_t (7Bd^2 + 2Btd)  ~  7BNd^2 + BdN^2        (O(N) in the weight term)
without cache = sum_t (7Btd^2 + 2Bt^2 d) ~ (7/2)Bd^2 N^2 + (2/3)BdN^3   (O(N^2) in the weight term)
```

Verified ratios at `B=1`, `d=8192` (Llama-3-70B hidden size), per layer:

| N | ratio |
|---|---|
| 128 | 64.5× |
| 256 | 128.7× |
| 1,024 | **515.5×** (with cache 4.8963e11 FLOPs; without, 2.5240e14) |
| 4,096 | 2094× |

The ratio grows **linearly** in N. Measured end-to-end (Raschka, 124M model, 200 tokens, Mac Mini M4 CPU): no cache 17.5 s → naive cache 3.3 s (**5.3×**) → pre-allocated 2.8 s (6.25×) → pre-allocated + `torch.compile` 2.4 s (7.3×). The gap between 512× in FLOPs and 5.3× measured is the point at which the workload stops being compute-bound — see §7.5.

### 7.2 The memory formula

One equation sizes every autoregressive deployment ([[kv-cache-memory-formula]]):

```
KV bytes = 2 * B * s * L * n_kv_heads * d_head * bytes_per_element
```

**The leading 2 is one K tensor plus one V tensor.** Not "two bytes", not a safety factor. Symbols:

| symbol | meaning |
|---|---|
| `L` | `num_hidden_layers` |
| `n_kv_heads` | `num_key_value_heads` — **NOT** `num_attention_heads` |
| `d_head` | `hidden_size / num_attention_heads` |
| `s` | cached tokens = prompt + generated so far |
| `B` | concurrent sequences |
| `bytes_per_element` | 2 (bf16/fp16), 1 (fp8/int8), 0.5 (int4) |

Per-token cost is `2 · L · n_kv_heads · d_head · bytes_per_element` — **independent of `n_heads` and of `d_model`**. Worked, bf16:

| model | L | `n_kv_heads` | `d_head` | bytes/token | |
|---|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 131,072 | 128.0 KiB |
| **Llama-3-70B** | 80 | 8 | 128 | **327,680** | **320.0 KiB** ← reference number |
| Llama-3-70B *if MHA-64* | 80 | 64 | 128 | 2,621,440 | 2,560.0 KiB = 2.5 MiB (exactly 8×) |
| Llama-3-405B | 126 | 8 | 128 | 516,096 | 504.0 KiB |
| Llama-2-7B (MHA) | 32 | 32 | 128 | 524,288 | 512.0 KiB |
| PaLM-540B (MQA) | 118 | 1 | 256 | 120,832 | 118.0 KiB |

Note the **fifth** row (Llama-2-7B, MHA, `n_kv_heads = 32`) against the **second** (Llama-3-70B, GQA-8): **Llama-2-7B's cache per token is larger — 512.0 KiB vs 320.0 KiB — at one tenth the parameters.** Architecture beats size here. (The third row is the counterfactual "70B if it had shipped MHA-64", which would have been 2,560.0 KiB/token, exactly 8× the real thing.)

Per request, Llama-3-70B:

| context | bytes | |
|---|---|---|
| 8,192 | 2,684,354,560 | **2.50 GiB** |
| 32,768 | 10,737,418,240 | **10.00 GiB** |
| 131,072 (128k) | 42,949,672,960 | **40.00 GiB** |

(Llama-3-8B: 1.00 / 4.00 / 16.00 GiB at the same three lengths.) Batched: `L=32, B=16, s=4096, n_kv_heads=8, d_head=128`, bf16 → `2·32·16·4096·8·128·2 = 8,589,934,592 B = 8.00 GiB`.

**The Llama-3 family is the cleanest illustration of what the formula does and does not depend on:** `n_kv_heads = 8` at *every* size. 8B has `L=32, H_q=32` (4:1); 70B has `L=80, H_q=64` (8:1); 405B has `L=126, H_q=128` (16:1). 8B → 405B is **50× the parameters but only 3.9× the KV cache** (`516,096 / 131,072 = 3.9375`), because only `L` grew.

**Capacity math**, 8×H100 serving Llama-3-70B. This is the one place in the chapter where mixing decimal and binary units silently changes the answer, so do it in **bytes** (per the Conventions note in §1: hardware capacity and parameter bytes are quoted decimal, `640 GB = 6.40×10¹¹ B`; the per-request KV figures above are exact binary quantities):

```
KV budget = 6.40e11 (HBM) - 1.40e11 (weights, 70e9 x 2 B) - 8.0e10 (overhead)
          = 4.20e11 B  =  420 GB  =  391.16 GiB

8k request   :  2,684,354,560 B  ->  4.20e11 / 2.684e9  = 156.5  ->  156 concurrent
128k request : 42,949,672,960 B  ->  4.20e11 / 4.295e10 =   9.78 ->    9 concurrent
```

16× the context, **exactly** 16× fewer concurrent requests — the linearity is the point. (Watch the trap: dividing the *number* 420 by the *binary* numbers 2.5 and 40 without converting gives 168 and 10.5, which is what `figures/kv-cache.html` panel 4 prints on its 8×H100 card. Those two answers are a unit slip of `2³⁰/10⁹ = 1.074`; the byte-exact answers are 156 and 9.8.) And the raw formula is a **lower bound** — multiply by **1.15** for block/page tables (1–2%), fragmentation (5–15% contiguous, <5% paged), swap headroom, and CUDA-graph bucket reserves, which lands the realistic figures at **136** and **8.5**.

### 7.3 MHA / GQA / MQA / MLA — four points on one axis

MHA, GQA and MQA differ in exactly **one number**, `n_kv_heads`:

| scheme | `H_kv` | cache divisor |
|---|---|---|
| MHA | `H_q` | 1× |
| GQA-G | `G` | `H_q/G` |
| MQA | 1 | `H_q` |

Query head `h` reads from KV head `h // (H_q/G)`; each KV head serves `H_q/G` query heads. Llama-3-70B GQA-8: `H_q = 64`, `H_kv = 8`, 8 query heads per KV head, exactly 8× reduction.

The quality cost is real but small, and concentrated at the aggressive end. The figures usually quoted for MQA versus MHA are **HumanEval −2.2, GSM8K −1.5, MMLU −0.6**. (Caveat, flagged in the same style as §4.5: **this triple is unverified.** It is carried over from the course's source library with no primary citation attached, and it cannot be traced to either obvious candidate — Shazeer 2019 introduces MQA but predates all three benchmarks, and Ainslie et al. 2023's GQA ablations are on T5-scale summarization/QA, not HumanEval/GSM8K/MMLU. Treat the *direction* and the *order of magnitude* — a small, single-digit-percentage regression concentrated on reasoning-heavy tasks — as the takeaway, and do not cite the decimals.) GQA at `G ≥ 8` matches MHA within noise; `G = 8` is the empirical sweet spot. You do not need to retrain from scratch to get there: **GQA uptraining** mean-pools an MHA checkpoint's KV heads into `G` groups, then continues pretraining with ~5% of the original pretraining compute, recovering to within ~0.1 perplexity of GQA-from-scratch. That is how Llama-2-70B got GQA-8.

**MLA leaves the family.** DeepSeek's Multi-head Latent Attention caches one low-rank latent per token, and — note carefully — **there is no factor of 2**, because a single latent `c_KV` serves both K and V:

```
MLA bytes/token = L * (d_c + d_rope) * bytes_per_element

DeepSeek-V3:  L=61, d_c=512, d_rope=64, bf16
              61 * 576 * 2 = 70,272 bytes = 68.6 KiB/token
```

Against naive MHA at V3's own geometry (61 layers, 128 heads, `d_head=128`, bf16 = `2·61·128·128·2 = 3,997,696 B = 3.81 MiB/token`), the ratio is **56.9×**. The `d_rope` term is not an implementation wart: `k_rope` must stay uncompressed because RoPE is position-dependent, and rotating *before* compression would make the latent position-specific and destroy reuse. Hence `(d_c + d_rope)`, not `d_c`. See [[gqa-mqa-mla-kv-heads]].

### 7.4 Implementation traps

**Pre-allocation vs `torch.cat`.** `torch.cat` copies the whole cache every step — `O(n)` copy per step, `O(n²)` total — but uses only real-length memory. `torch.zeros(B, H_kv, max_seq_len, d_head)` is `O(1)` per step but reserves the maximum: ~8 GB reserved even for a 50-token request on a 128k-context model.

**Two classic bugs.** (1) The cache must be **reset** between generations, or queries from a new prompt attend to stale keys from the previous sequence — output looks fluent and is subtly wrong, which is the worst failure mode. (2) **Position ids must be tracked** (`pos_ids = arange(current_pos, current_pos + seq_len)`), or every new token is treated as position 0 and RoPE breaks — §5.3's `Δ = m − n` becomes garbage.

**PagedAttention** (vLLM) solves fragmentation of the KV pool: 16 tokens per block, a per-sequence block table, copy-on-write prefix sharing, achieving >96% KV utilization versus 20–38% for contiguous allocators. Note it for §7.6: **it has no training analog.**

### 7.5 Prefill vs decode — two different machines

```
H100 SXM roofline knee: 1979 TFLOPS bf16 / 3.35 TB/s HBM = 590 FLOPs/byte
Llama-3-70B, bf16: P = 70e9 params -> 140 GB of weights read per forward pass
```

**Prefill** (`T = 4096` prompt tokens in one pass) does `2·P·T = 5.7344e14` FLOPs against the same 140 GB weight read:

```
theoretical intensity = (2·P·T FLOPs) / (P · 2 B) = T = 4,096 FLOPs/byte  ->  6.9x ABOVE the knee
effective/measured    ~ 3,100 FLOPs/byte                          ->  5.2x above the knee
```

The theoretical number is exactly the prompt length, because prefill reads each weight **once** and reuses it for all `T` tokens. The ~3,100 figure is the measured baseline once KV writes, activations and non-GEMM work are included. Either way: **COMPUTE-BOUND.** At 8×H100 that is ~36.2 ms for the prompt, ~113,000 tok/s.

**Decode at `B = 1`** does `2·P = 1.4e11` FLOPs and reads the weights *plus* the cache:

```
bytes = 1.40e11 (weights) + 1,342,177,280 (KV at s=4096, 1.25 GiB) = 1.4134e11
theoretical intensity = 1.4e11 / 1.4134e11 = 0.99 FLOPs/byte      ->  596x BELOW the knee
effective/measured    ~ 0.3 FLOPs/byte                            ->  1,970x below the knee
                                                                      = 1.005 TFLOP/s = 0.051% of peak
```

The theoretical `≈ 1` is not a coincidence and is worth memorizing as the sanity check: at `B = 1` every weight element is read once and used in exactly one multiply-add, so a bf16 decode gets **2 FLOPs per 2-byte element = 1 FLOP/byte**, no matter how large the model is. The measured `~0.3` is that ceiling minus real-kernel overhead, and it is the value `figures/kv-cache.html` panel 3 plots on its roofline. **BANDWIDTH-BOUND**, ~5.27 ms/token, ~190 tok/s.

Between the two: `3,100 / 0.3 = 1.03e4` — **four orders of magnitude apart**, on the same hardware, in the same request. (Using the theoretical pair instead, `4,096 / 0.99 = 4.1e3`, gives 3.6 orders. Either bracketing is defensible; **three** orders is not — that was an earlier misstatement in this chapter and it understated the gap by 10×.) Prefill processes `T` tokens in parallel through big GEMMs; decode processes one token and spends its time *reading the weights and the cache out of HBM*. That is why the 512× FLOP saving in §7.1 shows up as 5.3× wall-clock: removing FLOPs from a bandwidth-bound phase buys less than the FLOP count suggests. And it is why batching is the only real escape — `B` multiplies the numerator of the decode intensity while leaving the weight read fixed.

### 7.6 **TRAINING HAS NO KV CACHE — ZERO BYTES**

This is the single most important boundary in the chapter, and it is the one most often gotten wrong.

Teacher forcing feeds the ground-truth sequence `x_1 … x_s` and computes cross-entropy at **every** position in **one** forward pass; the causal mask (§3.1) makes position `t` depend only on positions `≤ t`. There is no autoregressive loop ⇒ no repeated work ⇒ **nothing to amortize** ⇒ no cache. The KV cache is a pure inference-time structure.

**The trap** is that identical algebra describes a different object. The bytes of K and V held during a *training* forward pass are:

```
2 * B * s * L * n_kv_heads * d_head * b
```

— character-for-character the inference KV-cache formula. At Llama-3-8B geometry, `B=1, s=8192`, bf16: `2·32·1·8192·8·128·2 = 1,073,741,824 B = 1.00 GiB`, numerically identical to the inference KV cache for one 8k request. Llama-3-70B at the same settings: `2,684,354,560 B = 2.50 GiB`. Same number, different resident.

| | training | inference |
|---|---|---|
| forward passes per sequence | **1** | 1 prefill + N decode steps |
| why K/V are held | backward needs them for `dL/dQ`, `dL/dK`, `dL/dV` | the next decode step attends over them |
| lifetime | one optimizer step, then freed | the whole request |
| grows during use | **NO** — allocated at full `s` in one shot | **YES** — +1 token per sequence per step |
| removable | **YES** — gradient-checkpointing recompute | **NO** — only compressible (GQA/MLA/quant/paging) |
| ledger bucket | **activations** ([[ch-01]] item 4) | a separate persistent KV pool |

> **Any training OOM analysis containing a "KV cache" line item is double-counting.** ([[train-vs-infer-kv-boundary]])

**And the GQA divisor is different in training.** FlashAttention's backward saves Q, K, V, O and the logsumexp statistics; GQA shrinks only K and V — **Q is untouched**. Deriving the training-side divisor:

```
training QKV-activation divisor = 3·H_q / (H_q + 2·H_kv)
```

| model | inference divisor | training divisor |
|---|---|---|
| Llama-3-8B (32 → 8) | 4.0× | **2.00×** |
| Llama-3-70B (64 → 8) | 8.0× | **2.40×** |
| MQA (64 → 1) | 64.0× | **2.909×** |

Verified per-layer bytes at `B=1, s=8192, d_head=128`, bf16, Llama-3-70B:

```
GQA-8 : Q 128 MiB + K 16 MiB + V 16 MiB = 160 MiB/layer = 167,772,160 B  -> x80 = 12.5 GiB
MHA-64: Q 128 + K 128 + V 128           = 384 MiB/layer = 402,653,184 B  -> x80 = 30.0 GiB
ratio 30.0 / 12.5 = 2.4x
```

So the headline "GQA gives 8× savings" is an *inference* claim. In training it is ~2.4×, and no amount of GQA aggressiveness pushes it past 3×. **PagedAttention has no training analog** for the same structural reason: it solves KV-pool fragmentation, which does not exist in training because the allocation is one contiguous shape known before the step begins.

**The mental bridge to carry:** *training ≈ prefill + backward, forever* (compute-bound, no cache, everything in parallel); *serving = prefill once, then a long bandwidth-bound decode tail.*

**boson / Lina TMR hook.** GDN linear-attention layers are hard-asserted to `CP=1`. Their inference-time state is a fixed-size recurrent state per head (a `d_head × d_head` matrix in standard linear attention), **independent of sequence length** — so the `2·L·n_kv_heads·d_head·s·b` scaling argument does not apply to those layers at all. The training side is unchanged: GDN blocks still store per-position activations for backward, so a 32k-sequence bill is a **checkpointing problem, not a cache problem**. Do not import the inference intuition into the training budget of [[ch-09]].

> **▶ Interactive companion — [`figures/kv-cache.html`](figures/kv-cache.html) (panels 1–6)**
> *What is recomputed, what is cached, and what training does instead.* **Panel 1** answers "why is a cache legal at all" by generating 6 tokens **without** one: a `6×6` grid where the diagonal cell is blue (`K_j` computed fresh) and everything left of it is red (recomputed **bit-identically**), with a proof box that literally re-derives `K₁ = x₁·W_K` by hand at every step and shows the same integers coming back. Counters track `t(t+1)/2` computed against `t(t−1)/2` wasted; a green callout states the §3.3 invariant and its converse — bidirectional attention has no KV cache.
> **Panel 2** replays the same generation side by side, 정책 A (no cache, tagged `O(N²)`) against 정책 B (cache, tagged `O(N)`), landing on **21 vs 6 = 3.5×** at `N = 6` and a fixed card reading **512.5× at N = 1024**. A Q-strip underneath shows `q₁…q₅` struck through and only `q_t` live, which is the detail people miss: **Q is never cached.** Its blue callout carries §7.1's exact-FLOP table at `d = 8192, B = 1` (`64.5× / 128.7× / 515.5× / 2094×` at `N = 128 / 256 / 1024 / 4096`) and Raschka's measured `17.5 s → 3.3 s → 2.8 s → 2.4 s`.
> **Panel 3** is §7.5: a timeline with one wide PREFILL block followed by narrow `+1 tok` decode blocks, beside a log-log roofline canvas with the `min(peak, BW×I)` roof, the **knee at 590** drawn as a dashed line, shaded BANDWIDTH-BOUND / COMPUTE-BOUND regions, and two plotted points — prefill at `I = 3,100` and decode(`B=1`) at `I = 0.3` — with the current step ringed. A red box nails the H100 knee arithmetic down; a second callout gives the intuition (intensity ≈ tokens processed per weight read) and states that batching is the only escape.
> **Panel 4** is §7.2's master formula as a calculator: six presets (8B / 70B / 405B / Llama-2-7B MHA / PaLM-540B MQA / "70B as MHA-64"), sliders for `B`, `s` (512 … 131,072), `L`, `n_kv_heads`, `d_head`, and a dtype selector (2 / 1 / 0.5 B), a formula box with the live values substituted, and five output rows — per token, per request, per batch, ×1.15 realism, and concurrent requests on 8×H100. Its canvas is the one worth staring at: the KV-cache curve (slope 1, **linear** in `s`) against the training `[B, a=32, s, s] × L` curve (slope 2, **quadratic**) on log–log axes with an 80 GB reference line, which at `s = 131,072` reads 40.00 GiB of cache against ~80 TiB of training score tensors — a factor of **2,048**.
> **Panel 5** is §7.3 as a four-mode switch (MHA `n_kv=8` / GQA-4 / GQA-2 / MQA) over a head-mapping diagram with `H_q = 8` fixed, plus the static "Llama-3-70B if it had been built this way" table: MHA-64 at 2,560.0 KiB/token, the real GQA-8 at 320.0 KiB (8×), MQA at 40.0 KiB (64×), and MLA at 68.6 KiB (56.9×, `61 × (512+64) × 2`, note the missing leading 2). Its closing callouts are the two traps — why `d_rope` stays uncompressed, and why **GQA's divisor does not carry over to training**.
> **Panel 6** is §7.6, the boundary, as an 8-step split animation: on the left TRAINING · teacher forcing, tagged "forward pass 1회", with `x₁…x₆` lighting up simultaneously and cards reading **KV cache 0 B** next to **1.00 GiB of K/V activations**; on the right INFERENCE · autoregressive, prefill then decode appends, with cards reading a growing cache and **the same 1.00 GiB, 삭제 불가**. The comparison table underneath runs the same rows as §7.6's table, and it ends on the PagedAttention note (>96% utilization vs 20–38%), a boson / Lina TMR paragraph on GDN linear-attention at `CP=1`, and a code block contrasting the two loops with the two classic bugs labelled in place.

---

## 8. Where This Plugs Into the Memory Ledger

### 8.1 Mapping the mechanism onto [[ch-01]]'s six items

| ledger item | what this chapter supplies |
|---|---|
| 1. weights (2 B/param) | `4·d_model²` per attention block (§4.3) + MLP `8·d_model²` + the `[V,h]` embedding (§1.2). Head count contributes **zero** |
| 2. gradients (2 B/param) | same shapes as item 1; RoPE contributes zero, a learned PE table contributes `L×d` (§5.4) |
| 3. Adam states (12 B/param) | likewise — and this is why a **learned** PE table costs 16–18 B/param while RoPE's cos/sin cache costs 4.00 MiB total (§5.4) |
| 4. **activations** | **§6.4's 16-tensor list, in full.** This is the chapter's main deposit |
| 5. logit spike (`B·T·V`) | §1.2's `h → V` expansion, already worked in [[ch-02]] `qa-deep-2` Q7 |
| 6. overhead | unchanged by anything here |

### 8.2 Reconstructing Korthikanti's coefficient from §6.4

Apply the save-rules of §6.4 to a whole block and count in units of `s·b·h` bytes ([[selective-recompute-korthikanti]] §4.1):

```
attention block = 11·sbh + 5·a·s²·b
    2 sbh   shared QKV input, stored ONCE  (the corollary in §6.4)
    4 sbh   Q and K retained for QK^T
    2 sbh   V retained for P·V
    2 sbh   input of the W_O linear projection
    1 sbh   attention-dropout mask (1 byte/elem)
    2as²b   softmax OUTPUT
    1as²b   softmax-dropout MASK (1 byte/elem)
    2as²b   softmax-dropout OUTPUT

MLP block = 19·sbh
    2 sbh   up-projection input
    8 sbh   GELU input   (4h wide -> 4 x 2 = 8)
    8 sbh   down-projection input (4h wide)
    1 sbh   MLP-dropout mask

LayerNorms = 4·sbh   (2 sbh saved input x 2 norms)

TOTAL = 34·sbh + 5·a·s²·b = s·b·h·(34 + 5as/h)      so 34 = 11 + 19 + 4
```

At the reference config:

| term | bytes | |
|---|---|---|
| 11 sbh (attention) | 184,549,376 | 184.55 MB |
| 19 sbh (MLP) | 318,767,104 | 318.77 MB |
| 4 sbh (LayerNorms) | 67,108,864 | 67.11 MB |
| **34 sbh** | 570,425,344 | **570.43 MB** |
| `5as/h = 5·32·4096/4096 = 160`, so the `s²` term = 160 sbh | 2,684,354,560 | **2.68 GB** |
| **per layer** `(34+160)·sbh = 194 × 16,777,216` | 3,254,779,904 | **3.25 GB** |
| **× L=80** | 260,382,392,320 | **260.38 GB** |

**Reconciling 3.25 GB with §6.4's 1.68 GB — do not let these contradict.** They are the same block at different granularity:

- The **16-tensor list** counts the `s²`-shaped tensor **once** at 2 B/elem (`2as²b` = 1.07 GB, softmax output only, no dropout). Korthikanti counts **all three** `s²`-shaped tensors (`5as²b` = 2.68 GB) because he includes the dropout mask and the dropout output.
- In the other direction, the list counts **36 sbh of block OUTPUTS**; Korthikanti counts **34 sbh of saved INPUTS**.

Both are correct. **Never mix them inside one sum.**

**A derived correction for 2026 (flagged as derivation, not in the paper):** modern LLMs train with `dropout = 0`. Delete the three dropout tensors and the coefficient becomes `sbh(32 + 2as/h)` — attention 10 sbh, MLP 18 sbh, LayerNorms 4 sbh, `s²` term `2as²b`. That `2as²b` is **exactly** the 1.07 GB attention-probability tensor of §6.4 item 7. Per layer `(32+64)·sbh = 96 × 16,777,216 = 1,610,612,736 B = 1.61 GB`; ×80 = `128,849,018,880 B = 128.85 GB` (against the enumerated 134.22 GB — the residual gap is the outputs-vs-inputs granularity). **Quote `34 + 5as/h` when citing the paper; use `32 + 2as/h` when modelling a real dropout-free run.**

**One coincidence to refuse.** At this config, `5as²b` (**one** layer's attention term) = 2,684,354,560 B = 2.68 GB, and `2·s·b·h·L` (the full-recompute floor for **all 80** layers) = 2,684,354,560 B = 2.68 GB. They are numerically identical because `5·a·s = 5·32·4096 = 655,360 = 2·h·L = 2·4096·80`. **They are unrelated quantities.** Any mental model or animation that implies a connection is wrong.

### 8.3 The crossover — two of them, and they are different

Setting the linear and quadratic terms equal answers "when does attention take over?" — but the answer depends on which linear term you mean.

**Within the attention block alone** (`11·s·b·h = 5·a·s²·b`):

```
s = 11h/(5a)
  7B model (h=4096, a=32):  11 x 4096 / 160 = 45056/160 = 281.6 tokens
  2017 base (h=512,  a=8):  11 x 512  / 40  = 5632/40   = 140.8 tokens
```

Past a few hundred tokens, essentially **all attention-block activation memory is the `N×N` routing table**, not Q/K/V.

**Against the whole block** (`34 = 5as/h`), which is the number already in `ch-04/qa.md` and reused here verbatim:

```
s = 34h/(5a) = 34 x 4096 / (5 x 32) = 870.4  ~  870 tokens
```

Past **~870 tokens** the attention `s²` term outweighs *all other activations combined*. Both numbers are right; they answer different questions, and quoting the wrong one in the wrong place is the easiest way to look confused.

And the per-GPU anchor already established in `ch-04/qa.md`, kept consistent: with `s=4096, h=4096, a=32, t=8, L=80, b=1`, the attention `s²` term per GPU is `5as²b·L/t = 5·32·4096²·1·80/8 = 26,843,545,600 B = 26.8 GB`, against 5.7 GB for all other activations with SP and 6.75 GB for the entire 432 GB static ledger split over 64 GPUs.

### 8.4 The handoff

Everything [[ch-04]] says now has a mechanism underneath it:

| [[ch-04]] claim | mechanism from this chapter |
|---|---|
| "materializes an N×N score matrix" | §2.3 step 3 — softmax must produce `P`, and §6.4's rule says softmax saves its **output** |
| "per head per layer" | §4.4 — the tensor is `(B, a, N, N)`, linear in `a` at constant FLOPs |
| "2 GB per head at N=32k" | §4.4's table, `2,147,483,648 B` |
| "the kernel decides" | §3.4 — the lower triangle is structurally dead, and only a causal-aware kernel converts that into bytes |
| "streaming never writes the matrix" | §2.3 — `P` is only ever needed row-wise, and §3.4's tile-local masking is what lets a tile be self-contained |
| "MATH backend falls back to O(N²)" | §5.4 — a custom logit-additive bias is one common way to force that fallback |

> **⚠ Scope note, restated at the handoff (see §1).** Every line of the table above, and every `s²` term in §8.2 and §8.3, is an accounting of **standard softmax attention**. boson / Lina TMR uses **GDN linear-attention** with `CP=1`, and linear attention never forms the `N×N` score matrix — so for those layers the `5·a·s²·b` term, the `s = 34h/(5a)` crossover, and the 26.8-GB-per-GPU attention anchor simply do not have a referent. What *does* carry over unchanged is everything in §6.4 that is linear in `s`: the residual stream, the Q/K/V projections, the MLP's two `4h` tensors, the LayerNorm saved statistics, and therefore the `34·sbh` side of the coefficient and the `2·s·b·h·L` checkpointing floor. The practical reading for [[ch-09]]: for boson, long sequences are a **linear** activation-memory problem solved by checkpointing and parallelism, not a quadratic one solved by kernels — which is exactly why the quadratic story is worth knowing (it is the cost GDN was designed to remove), and exactly why it must not be pasted into boson's budget unmodified.

> **▶ Interactive companion — [`figures/transformer-block-dataflow.html`](figures/transformer-block-dataflow.html) (panels 4 and 6)**
> *From 16 tensors to `34 + 5as/h`, one accounting convention at a time.* **Panel 4** asks §8.2's question in its own heading — "위 16개의 합 = ch-03의 `s·b·h·(34 + 5as/h)` 인가?" — and answers it as a four-row reconciliation table whose columns are residual stream / attention branch / MLP branch / LN saved statistics / `s²` term / block total / `× L`. The four rows are exactly the four readings this chapter distinguishes: **(a)** the 16 tensors enumerated in §6.4, **(b)** Korthikanti's `sbh(34 + 5as/h)` split into `4·sbh + 11·sbh + 19·sbh + 5as²b` (570.43 MB linear + 2.68 GB quadratic = 3.25 GB/block, 260.38 GB at `L = 80`), **(c)** the dropout-free 2026 recount `sbh(32 + 2as/h)` (1.61 GB/block, 128.85 GB at `L = 80`), and **(d)** gradient checkpointing, which keeps item 1 and nothing else. A red note below the table walks the difference column by column — `+2` for counted outputs, `−1` for the uncounted dropout mask, and the `s²` column's one-tensor-vs-three gap — so §8.2's reconciliation is read off a table rather than reasoned from a paragraph. Its right column carries the VJP save-rules (matmul → 입력, GELU → 입력, softmax → 출력, dropout → 1 B mask, LayerNorm → 입력 + (mean, rstd), residual add → 아무것도 안 함) and a green callout computing `s = 34h/(5a) = 870.4` with the current dominance factor (4.7× at `s = 4096`).
> Panel 4 also carries a **conditional** red warning banner that fires only when `5as = 2hL` and dtype is bf16 — which is true at the reference config — and says in so many words that the `2,684,354,560 B` appearing in two different cells is a numeric coincidence between one layer's `5as²b` and all 80 layers' checkpoint floor. That is §8.2's "one coincidence to refuse", enforced by the figure rather than asserted by the prose.
> **Panel 6** closes the loop back to [[ch-01]]: a live table mapping this block onto the six ledger items — weights (`12h² = 201,326,592` params × 2 B = 402.65 MB/block), gradients (402.65 MB), AdamW state (2.42 GB), **activations** (the highlighted row, all 16 tensors, 1.68 GB/block → 134.22 GB), logit spike (dimmed, 0 for a block), overhead (dimmed, 0) — plus one purple-topped extra row for "K, V가 decode 사이에 남는 것", marked **ledger에 없음** and carrying the inference formula `2·B·s·L·n_kv·d_head·b`. Its closing callout contrasts the static per-parameter cost (16 B/param, invariant to `B·s`) against item 4 (proportional to `B·s`, and partly to `s²`), and restates that training has no KV-cache line.

---

## Core Insights from the Literature

**1. Attention is a differentiable dictionary lookup, and every design choice follows from making the lookup trainable** (Vaswani et al. 2017; [[qkv-scaled-dot-product]]). Hard `argmax` has zero gradient almost everywhere, so it is replaced by a softmax-weighted convex combination over all values. Three separate projections exist because collapsing them into `XXᵀ` breaks in three independent ways at once — symmetry, Cauchy–Schwarz diagonal dominance under norm-equalized inputs, and the absence of any learnable routing parameters. The mechanism is not "an architecture choice"; it is the minimal thing that is simultaneously differentiable, directional, and learnable.

**2. `1/√d_k` is a necessary condition, not cosmetic normalization** ([[sqrt-dk-scaling-variance]]). Dot-product variance is *exactly* `d_k` under the i.i.d. unit-variance assumption, so raw scores grow as `√d_k`; feeding those into softmax saturates it, and the softmax Jacobian `p_i(δ_ij − p_j)` — the **only** path by which gradient reaches `W_Q` and `W_K` — collapses. The measured cost is a **~586×** gradient reduction at `d_k = 64` for two entirely ordinary scores. Vaswani measured the downstream effect too: unscaled dot-product attention loses to additive attention at large `d_k`, while scaled attention matches it and remains a plain GEMM.

**3. Multi-head attention is FLOP-neutral and memory-expensive** ([[multi-head-split-concat-wo]]). Parameters are `4·d_model²` regardless of the head count `a`; FLOPs are `2·B·N²·d_model` and the `a` cancels *exactly*. But the score tensor is `(B, a, N, N)`, so bytes scale **linearly in `a` at constant FLOPs**, with arithmetic intensity of exactly **`2·d_head` FLOPs per stored element (= `d_head` FLOPs per stored byte in bf16)**. Doubling heads halves intensity and doubles memory. Vaswani's Table 3 row (A) shows the quality curve is flat between 8 and 16 heads and falls at both ends — so the head count that is free in compute is *never* free in the ledger.

**4. Position is an input, not a property, and where you inject it determines its memory contract** ([[attention-permutation-equivariance]], [[sinusoidal-absolute-encoding]], [[rope-rotary-position-embedding]]). Bare self-attention is exactly permutation-equivariant, with a four-line proof whose crux is `Pᵀ P = I` cancelling between the conjugated score matrix and `V`. The 2017 answer already contained the rotation structure; RoPE's contribution is applying it **multiplicatively to Q and K** rather than additively to embeddings, which is uniquely forced by three constraints (relative dependence, identity at origin, magnitude preservation). The payoff is structural, not aesthetic: RoPE lives *outside* the attention kernel, so FlashAttention still sees an ordinary `(Q,K,V)` triple, whereas a logit-additive scheme needs a `[B,H,T,T]` bias — **4.000 GiB per layer** at `H=32, T=8192` — the exact tensor the kernel exists to avoid.

**5. The block's saved-tensor list is derivable from one rule per operation** ([[transformer-block-tensor-ledger]]). matmul saves its INPUT (`dW = Xᵀ dY`), GELU saves its INPUT, softmax saves its OUTPUT (VJP `dS = P ⊙ (dP − rowsum(dP ⊙ P))`), dropout saves a 1-byte MASK, LayerNorm saves input + (mean, rstd), and the residual add saves NOTHING. Summing that list **is** Korthikanti's coefficient: `34 = 11 + 19 + 4`. Two orders of magnitude of memory analysis reduce to six derivation rules and one shape question.

**6. Normalization placement is a gradient-flow decision, not a memory decision** ([[pre-ln-vs-post-ln]]). Post-LN puts a norm on the residual highway, so backward traverses `L` LayerNorm Jacobians in series. Xiong et al. 2020's theorems bound the *last* layer's gradient at initialization: `O(d√(ln d))` for Post-LN — independent of depth — against `O(d√(ln d / L))` for Pre-LN, damped by `1/√L`. The `1/√·` belongs to Pre-LN and is in **total depth `L`**, not layer index; the failure mode is the resulting *depth-imbalance* of Post-LN gradients (stated qualitatively — the two scalings are the formal results), which is why the 2017 paper needed 4,000 warmup steps and Pre-LN does not. Pre-LN leaves a clean `I` on the highway. But both save the same `4·sbh` (67.11 MB/block), and the checkpoint is `[B,T,h]` either way. RMSNorm's *only* real byte saving is halving the saved-statistics tensor, 5.24 MB → 2.62 MB over the whole model — negligible against one 1.07 GB attention matrix.

**7. The residual stream is why gradient checkpointing is cheap** ([[residual-stream-memory-backbone]]). A decoder-only transformer is one `[B,T,h]` tensor with `2L` sub-layers reading and adding back. That fixed-width additive backbone simultaneously explains why 100-layer models train (identity gradient path), why features compose across layers (early writes are never overwritten), and why the checkpoint is **33.55 MB against 1.64 GB of interior tensors — 49×**. [[ch-03]]'s `2·s·b·h·L` is literally "L snapshots of the residual stream at 2 bytes an element."

**8. Training has no KV cache, and the identical algebra is a trap** ([[kv-cache-mechanism]], [[kv-cache-memory-formula]], [[gqa-mqa-mla-kv-heads]], [[train-vs-infer-kv-boundary]]). The cache is forced by causal attention's immutability invariant and saves `(N+1)/2` token-forward-passes — 515.5× in FLOPs at N=1024, `d=8192`. Its size is `2·B·s·L·n_kv_heads·d_head·bytes`, independent of `n_heads` and `d_model`, which is why Llama-3-405B is 50× the parameters and only 3.9× the cache. **But teacher forcing means training runs one parallel forward pass, so there is nothing to amortize and the training KV cache is zero bytes.** The K/V bytes that *do* exist in training are activations with a one-optimizer-step lifetime, deletable by checkpointing — and the GQA divisor that gives 8× at inference gives only **2.40×** in training, because FlashAttention's backward also saves Q, which GQA never shrinks.

---

## Key Takeaways

- Attention replaces an untrainable `argmax` lookup with a softmax convex combination; the output always lies in the convex hull of the value vectors, which is why `W_O` exists.
- Three projections are mandatory: `XXᵀ` is symmetric (relations are directional), has a strict-max diagonal under RMSNorm-equalized norms (attention collapses to the identity), and has no parameters to learn routing with.
- `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` executes in four steps; steps 1–3 produce `(B,N,N)` tensors and step 3's output is what the backward pass keeps.
- `Var(q·k) = d_k` exactly, so `1/√d_k` restores unit score variance at every head dimension. Skipping it costs ~586× gradient at `d_k = 64` on ordinary scores, and the assumption degrades during training — which is what QK-norm re-enforces.
- The causal mask is additive and applied pre-softmax; use `torch.finfo(dtype).min` (`-1e9` is unrepresentable in fp16; literal `-inf` gives `0/0 = NaN` on a fully-masked row) and include the diagonal (`j ≤ i`).
- The mask's real gift is the invariant: `k_j, v_j` depend on token `j` alone and are never revised. That licenses teacher forcing (training in one parallel pass) *and* the KV cache (inference memoization) — the same fact underwriting both sides of the course.
- Heads are free in parameters (`4·d_model²`, invariant in the head count `a`) and free in FLOPs (`2·B·N²·d_model`, `a` cancels exactly), and **not** free in memory: `(B,a,N,N)` is linear in `a` at constant FLOPs, at `2·d_head` FLOPs per stored element (`d_head` FLOPs per stored byte in bf16). At `N=32,768` the score tensor is **85×** Q+K+V combined, and that ratio grows linearly in `N`.
- Self-attention is exactly permutation-equivariant (`Attn(PX) = P·Attn(X)`), so position must be injected. The causal mask breaks the proof, which is why NoPE decoder LMs work at all.
- RoPE is uniquely forced by three constraints and yields `⟨R_m q, R_n k⟩ = qᵀ R_{n−m} k` — absolute position vanishes. It lives outside the kernel, costs 4.00 MiB of cos/sin cache at `L=8192`, and adds zero saved-for-backward activations when fused. ALiBi's `[B,H,T,T]` bias costs **4.000 GiB per layer** at `H=32, T=8192`.
- Pre-LN vs post-LN is a trainability decision with **identical** activation bytes (`4·sbh` = 67.11 MB/block either way).
- One block saves ~16 tensors totalling 1.68 GB at the reference config, of which the checkpoint is 33.55 MB — a **49–50×** ratio, and `2·s·b·h·L = 2.68 GB` across 80 blocks versus 134.22 GB unchecked. [[ch-03]] quotes the *same* lever as **97×** because its numerator is Korthikanti's published `(34 + 5as/h)·sbh = 194 sbh-units/block` rather than this chapter's enumerated `100 sbh-units/block`; same denominator, different accounting convention (§6.4, §8.2).
- Korthikanti's `34·sbh + 5as²b` is *derivable* from six save-rules: `34 = 11 (attn) + 19 (MLP) + 4 (LN)`. Per layer at the reference config that is 3.25 GB, of which 2.68 GB is the `s²` term. Dropout-free 2026 runs are better modelled by `32 + 2as/h` = 1.61 GB/layer.
- Two crossovers, two questions: attention-block-internal at **~282 tokens** (`11h/5a`), and attention-vs-all-other-activations at **~870 tokens** (`34h/5a`). Both are correct; do not swap them.
- **Training has zero KV-cache bytes.** Any training memory analysis with a KV-cache line item is double-counting. The training-side GQA divisor is `3H_q/(H_q + 2H_kv)` ≈ 2–3×, not the inference 8×, because Q is never shrunk.

---

## Questions

Prepared for the Discuss phase — reason causally, not descriptively.

1. `XXᵀ` fails for three independent reasons (§2.2). Suppose you fixed only the symmetry problem — use `W_Q ≠ W_K` but tie `W_V = I` so values are the raw residual stream. Which of the three failures return, and what would you expect to see in the trained model?
2. `1/√d_k` is derived under an assumption that is true at init and decays during training (§2.5). Predict the *observable* signature of a head whose scores have drifted to 5× the assumed variance by step 50,000 — in the loss curve, in the attention entropy, and in the gradient norms of `W_Q`. Then explain why QK-norm fixes it and why simply lowering the learning rate does not.
3. Head count is exactly FLOP-invariant but memory-linear (§4.3–§4.4). boson uses `a = 32` at `h = 4096`. If you halved `a` to 16 and doubled `d_head` to 256, state what changes in (a) parameters, (b) FLOPs, (c) the `5as²b` activation term, (d) the KV cache at inference, and (e) expected quality per Table 3 row (A). Which of those five is the binding constraint on A100-40GB?
4. The permutation proof (§5.1) fails for causal attention because `P M Pᵀ ≠ M`. Explain, without appealing to the experiments, why that failure is exactly what allows a NoPE decoder LM to encode absolute position — and what upper bound on positional resolution that mechanism implies.
5. §6.4's save-rule table says softmax stores its **output** while GELU stores its **input**. Derive both from the respective VJPs, then use the same reasoning to decide what SwiGLU must store, and whether SwiGLU's three matrices make its activation footprint better or worse than GELU's two at equal `d_ff`.
6. §8.2 reconciles 3.25 GB with 1.68 GB as "the same block at different granularity." Construct a third accounting that is *also* correct and lands on a fourth number, and state the rule that decides which of the four to quote in a capacity-planning document.
7. §7.6 asserts training has zero KV-cache bytes while the identical formula gives 2.50 GiB of Llama-3-70B activations. A colleague proposes "caching K and V across gradient-accumulation micro-steps" to save recompute. Diagnose the proposal: what is actually shared between micro-steps, what is not, and what breaks?
8. GDN linear-attention at `CP=1` has a sequence-length-independent inference state (§7.6). Does that change *any* number in §8's ledger for the boson training run? Argue from the save-rules of §6.4, not from the linear-attention literature.

---

## References

- Ashish Vaswani et al. "Attention Is All You Need." arXiv:1706.03762, 2017 (v7 PDF used for Table 3, p.9). https://arxiv.org/abs/1706.03762 — [[qkv-scaled-dot-product]], [[sqrt-dk-scaling-variance]], [[causal-mask-neg-inf]], [[multi-head-split-concat-wo]], [[sinusoidal-absolute-encoding]]
- Ruibin Xiong et al. "On Layer Normalization in the Transformer Architecture." ICML 2020, arXiv:2002.04745. https://arxiv.org/abs/2002.04745 — [[pre-ln-vs-post-ln]]
- Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton. "Layer Normalization." arXiv:1607.06450, 2016.
- Biao Zhang, Rico Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019, arXiv:1910.07467. https://arxiv.org/abs/1910.07467
- Kaiming He et al. "Deep Residual Learning for Image Recognition." CVPR 2016, arXiv:1512.03385. https://arxiv.org/abs/1512.03385 — [[residual-stream-memory-backbone]]
- Jianlin Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864 — [[rope-rotary-position-embedding]]
- Bowen Peng et al. "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071, 2023. https://arxiv.org/abs/2309.00071
- Ofir Press, Noah A. Smith, Mike Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi). ICLR 2022, arXiv:2108.12409. https://arxiv.org/abs/2108.12409
- Adi Haviv et al. "Transformer Language Models without Positional Encodings Still Learn Positional Information." Findings of EMNLP 2022, arXiv:2203.16634. https://arxiv.org/abs/2203.16634 — [[attention-permutation-equivariance]]
- Amirhossein Kazemnejad et al. "The Impact of Positional Encoding on Length Generalization in Transformers." NeurIPS 2023, arXiv:2305.19466. https://arxiv.org/abs/2305.19466
- Elena Voita et al. "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned." ACL 2019, arXiv:1905.09418. https://arxiv.org/abs/1905.09418 — *figures carried from the `llm-arch` wiki, not re-verified here*
- Noam Shazeer. "Fast Transformer Decoding: One Write-Head Is All You Need" (MQA). arXiv:1911.02150, 2019. https://arxiv.org/abs/1911.02150 — [[gqa-mqa-mla-kv-heads]]
- Joshua Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023, arXiv:2305.13245. https://arxiv.org/abs/2305.13245 — [[gqa-mqa-mla-kv-heads]]
- DeepSeek-AI. "DeepSeek-V2 / DeepSeek-V3 Technical Report" (Multi-head Latent Attention). arXiv:2405.04434, arXiv:2412.19437. — [[gqa-mqa-mla-kv-heads]]
- Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023, arXiv:2309.06180. https://arxiv.org/abs/2309.06180
- Vijay Korthikanti et al. "Reducing Activation Recomputation in Large Transformer Models." arXiv:2205.05198, 2022 (§4.1 for the `34·sbh + 5as²b` decomposition). https://arxiv.org/abs/2205.05198 — [[selective-recompute-korthikanti]], [[transformer-block-tensor-ledger]]
- Sebastian Raschka. "Understanding and Coding the KV Cache in LLMs from Scratch," 2025 (measured 124M-model timings). — [[kv-cache-mechanism]]

**Sibling chapters:** [[ch-01]] (the six-item ledger this chapter feeds), [[ch-02]] (precision and the `B·T·V` logit spike), [[ch-03]] (activations, checkpointing, selective recomputation, sequence parallelism), [[ch-04]] (the O(N²) memory problem this chapter is the prerequisite for), [[ch-05]] (FlashAttention), [[ch-06]] (the kernel zoo and the SDPA MATH fallback), [[ch-07]] (parallelism taxonomy), [[ch-09]] (the 27B MoE capstone where the boson hooks land).
