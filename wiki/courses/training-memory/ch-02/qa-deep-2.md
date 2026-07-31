<!-- qa-deep-2: ch-02 — the loss-head logit spike (§4-§5)
     overflow from [[qa]] / [[qa-deep]] · companion to [[read]] · append-only -->

# ch-02 Q&A (deep 2) — the loss-head logit spike

Third page for ch-02 (the first two hit the 120-line cap). Scope: §4 logit spike, §5 Liger-Kernel.

---

### Q6 — what is `seq` here?

**`seq` = sequence length = `T` in `[B, T, V]` = how many tokens are in one sample.** read.md L144's `seq=16,384` means one training example is 16,384 tokens long. It is a hyperparameter you set (`max_seq_len` / context length); longer samples are truncated, shorter ones padded or packed.

| symbol | meaning | boson |
|---|---|---|
| **B** batch | how many conversations per step | tuned to fit memory |
| **T** seq | how many tokens per conversation | one Lina call = system prompt + ~20 turns |
| **V** vocab | tokens the model knows | **248,000** |

- **The logit tensor does not distinguish B from T.** `size = B × T × V`, so only the *product* (total tokens per step) matters — which is why read.md L146 computes `16,384 × 32,000 × 2` with no B term. **Practical consequence:** halving batch or halving seq frees exactly the same logit memory, so pick whichever costs less in training quality. (Attention is *not* like this — it is `T²`, so shortening seq wins far more there; see [[ch-04]].)
- **Bytes — the chapter's 2 B and the 4 B figure are both real; multiple copies coexist.** For boson at T=4096, V=248k:

| copy | dtype | size |
|---|---|---|
| lm_head matmul output | bf16 (2 B) | **2.03 GB** |
| fp32 version inside `cross_entropy` (autocast keeps CE in fp32 — see [[qa]] Q3) | fp32 (4 B) | **4.06 GB** |
| logits gradient in backward | bf16 (2 B) | **2.03 GB** |

  read.md L146/L152 counts only the first row. The peak is a multiple of `B·T·V`, not one instance — which is why §5's Liger-Kernel does not attack precision but **never materializes the tensor at all**: not fewer copies, zero copies.

**One line:** `seq` is tokens-per-sample (`T`), the logit tensor costs `B·T·V` per copy so batch and seq are interchangeable levers there (unlike attention's `T²`), and several copies coexist at the loss — which is why §5 removes the tensor rather than shrinking it. See read.md §4/§5, [[qa]] Q3, [[ch-04]], [[ch-09]].

---

### Q7 — is V (vocab) the embedding space?

**No — different axes.** Both live in the embedding layer, but one is a *count* and one is a *dimension*.

| symbol | name | meaning | boson |
|---|---|---|---|
| **V** | vocab size | how many distinct **tokens** exist (rows) | **248,000** |
| **h** | hidden dim / d_model | length of **one vector** — this *is* the embedding space (columns) | e.g. 4,096 |

Embedding matrix `E : [V, h]`. Token #12,345 → row 12,345 → one h-dimensional vector. V picks *which row*; h is *how long the row is*.

- **Connection to §4** — `lm_head` runs the opposite direction:
  `hidden_states [B,T,h] @ lm_head.weight.T [h,V] → logits [B,T,V]`.
  The model runs internally in the compact h space (4,096) and **expands to V exactly once**, at the end, to answer "which of 248,000 tokens comes next". That expansion is `248,000 / 4,096 ≈ **60×**` — one final op inflates the tensor 60-fold. That *is* the spike.
- **Why §5 works:** the end product is a **scalar loss**. There is no reason to hold the 60×-inflated tensor whole; compute it in chunks and discard.
- **Hidden cost of a 248k vocab:** the embedding matrix alone is `248,000 × 4,096 ≈ 1e9` params — ~1B of the 27B sits in the vocabulary table, and if trainable the Rule of 16 applies: `16 B × 1e9 = 16 GB`. Most models use **weight tying** (input embedding and `lm_head` share one `[V,h]` matrix, transposed), else it would be 2B params.

**One line:** V counts tokens, h is the embedding space's dimension; the model computes in h and expands to V only at the loss head — a ~60× inflation that is exactly the §4 spike, and the reason §5 chunks it away. See read.md §4/§5, [[ch-01]] §2.
