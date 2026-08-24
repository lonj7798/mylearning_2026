# Q, K, V and Scaled Dot-Product Attention — Why Three Projections
<!-- slug: qkv-scaled-dot-product · type: paper · source: wiki:llm-arch:wiki/courses/llm-arch/ch-02/excerpts/attention-soft-dictionary-lookup.md + https://arxiv.org/abs/1706.03762 §3.2.1 -->

**Core Insight.** Attention is a *differentiable* dictionary lookup: hard `argmax` retrieval has zero gradient almost everywhere, so it is replaced by a softmax-weighted convex combination of all values. Three separate learned projections exist because a token's three roles are genuinely different — `W_Q` = "what am I looking for", `W_K` = "what do I advertise", `W_V` = "what content do I contribute". Collapsing them is not a simplification, it is a failure: with a single projection the score matrix becomes `XXᵀ`, which is **symmetric**, and for norm-equalised inputs (post-LayerNorm/RMSNorm every row has `‖x_i‖ = √d_model`) Cauchy–Schwarz forces the diagonal to be the row maximum — every token attends to itself and attention degenerates to the identity.

**Guideline.** Never reason about attention as "the token vectors dot-producted together". Reason about it as three tensors — `Q` (asks), `K` (advertises), `V` (transmits) — where `Q·Kᵀ` produces a *routing table* and `V` is the *payload*. When budgeting memory, count them separately: `Q`, `K`, `V` are three `B×N×d_model` activations that must survive to the backward pass, and the `N×N` routing table is a fourth, far larger, tensor.

## Technical Details

- **Projections.** With input `X ∈ ℝ^(N×d_model)`:
  `Q = X W_Q`, `K = X W_K`, `V = X W_V`
  `W_Q ∈ ℝ^(d_model×d_k)`, `W_K ∈ ℝ^(d_model×d_k)`, `W_V ∈ ℝ^(d_model×d_v)`
  → `Q ∈ ℝ^(N×d_k)`, `K ∈ ℝ^(N×d_k)`, `V ∈ ℝ^(N×d_v)`.
  `d_k` must match between `Q` and `K` (they are dot-producted). `d_v` is free — it only has to match `W_O`'s input.
- **The formula (Vaswani et al. 2017, Eq. 1, verbatim):**
  `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`
- **Four-step execution with shapes** (single head, batch `B`):
  1. `S = Q Kᵀ` → `(B, N, N)`  — cost `2·B·N²·d_k` FLOPs
  2. `S_scaled = S / √d_k` → `(B, N, N)` — elementwise
  3. `P = softmax(S_scaled, dim=-1)` → `(B, N, N)`, each **row sums to exactly 1.0**
  4. `O = P V` → `(B, N, d_v)` — cost `2·B·N²·d_v` FLOPs
  Row `i` of the output is `o_i = Σ_j P[i,j]·v_j` — a convex combination, so `o_i` lies in the convex hull of the value vectors.
- **Worked numeric example** (`d_k = 4`, so `√d_k = 2`; three keys, `d_v = 2`):
  `q₁ = [1,0,1,0]`; `k₁ = [1,0,1,0]`, `k₂ = [0,1,0,1]`, `k₃ = [1,1,0,0]`
  raw dots: `q₁·k₁ = 2`, `q₁·k₂ = 0`, `q₁·k₃ = 1`
  scaled (÷2): `1.0, 0.0, 0.5` → `exp = 2.718282, 1.000000, 1.648721` → sum `5.367003`
  **weights `A₁ = [0.506479, 0.186323, 0.307197]`** (sums to 1.000000)
  with `v₁=[1,0]`, `v₂=[0,1]`, `v₃=[1,1]`: **`o₁ = [0.813676, 0.493520]`**
  *Unscaled contrast* (scores `2, 0, 1`): `exp = 7.389056, 1, 2.718282`, sum `11.107338` → `[0.665241, 0.090031, 0.244728]` — the top weight jumps 0.506 → 0.665 from the scaling alone, at `d_k = 4`.
- **Vaswani base model (Table 3 base row, verified):** `N_layers = 6`, `d_model = 512`, `d_ff = 2048`, `h = 8`, `d_k = d_v = 64`, `P_drop = 0.1`, `ε_ls = 0.1`, 100K steps, dev PPL **4.92**, dev BLEU **25.8**, **65M params**.
- **Why not just `XXᵀ`** (three independent reasons):
  1. *Symmetry.* `XXᵀ` is symmetric → `S_ij = S_ji`. Language relations are directional ("it" → its antecedent, not the reverse).
  2. *Diagonal dominance.* Post-RMSNorm `‖x_i‖ = √d_model` for all `i`, so `x_i·x_j ≤ ‖x_i‖‖x_j‖ = ‖x_i‖² = (XXᵀ)_ii` with equality iff `x_i = x_j`. The diagonal is the strict row max → self-attention collapses to self-loop.
  3. *No learnable routing.* `XXᵀ` is a fixed function of the input; the model has no parameters with which to learn *what* to attend to.
- **Why `V` must be separate from `K`:** `K` lives in the geometry that `Q` is compared against; `V` lives in the geometry that `W_O` maps back into the residual stream. They are read by different downstream operators (a dot product vs. a weighted sum), so they need different subspaces.
- **Where the "soft dictionary" analogy breaks:** (a) `K` and `V` are projections of the *same* token, not independent key/value objects; (b) the output is a *blend*, so it can encode composites no single `v_j` contains; (c) in self-attention the query set *is* the database.
- **Training-memory angle:** Per transformer layer, the attention block's saved activations decompose (Korthikanti et al. 2022, bf16, no recompute) as **`11·s·b·h + 5·a·s²·b` bytes**, where `s`=seq, `b`=batch, `h`=`d_model`, `a`=heads. The `11sbh` linear term is exactly the QKV bookkeeping: `2sbh` for the shared QKV input `X` (needed for the three weight-gradient matmuls), `4sbh` for `Q` and `K` (needed for the `QKᵀ` backward), `2sbh` for `V`, `2sbh` for the `W_O` input, `1sbh` for the attention dropout mask. The `5as²b` term is the `N×N` routing table. **The two terms cross over at `s = 11h/(5a)`** — for a 7B model (`h=4096`, `a=32`) that is **281.6 tokens**; for the 2017 base model (`h=512`, `a=8`) it is **140.8 tokens**. Past a few hundred tokens, essentially all attention activation memory is the routing table, not `Q/K/V`. Concretely at `s=32768, b=1, h=4096, a=32, d_head=128`, bf16: `Q+K+V = 3 × 32768 × 4096 × 2 B = 805 MB` per layer, while the score tensor is `32 × 32768² × 2 B = 68.7 GB` per layer — **85×** larger.

## Citation
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. "Attention Is All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762 (§3.2.1, Eq. 1, Table 3). Activation decomposition: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," MLSys 2023, https://arxiv.org/abs/2205.05198. Derivation framing adapted from `llm-arch:wiki/courses/llm-arch/ch-02/excerpts/attention-soft-dictionary-lookup.md`.
