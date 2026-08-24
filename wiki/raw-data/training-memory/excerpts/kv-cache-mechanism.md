# Why a KV Cache Is Possible At All — Causal Masking and the O(n²)→O(n) Argument
<!-- slug: kv-cache-mechanism · type: doc · source: wiki:course/llm-inference:wiki/courses/llm-inference/ch-02/excerpts/attention-complexity.md -->

**Core Insight.** The KV cache is not a clever trick bolted onto attention — it is forced by a mathematical
property of *causal* attention: with the mask `M[i,j] = -∞ for j > i`, the key and value vectors at position
`j` are computed from token `j` alone (`k_j = W_K x_j`, `v_j = W_V x_j`) and are **never revised** when tokens
`j+1, j+2, …` arrive. Past K and V are immutable, so recomputing them at every decode step is pure waste. The
cache is a memoization table over an already-pure function.

**Guideline.** Whenever you see "cache" in an LLM system, ask *what is immutable*. In autoregressive decoding
K/V are immutable → cacheable. In training, nothing is decoded step-by-step, so there is nothing to memoize —
the same tensors exist, but as one-shot activations, not as a cache ([[train-vs-infer-kv-boundary]]).

## Technical Details

- **The immutability argument, precisely.** In a decoder-only transformer, `k_j = W_K x_j` and `v_j = W_V x_j`
  depend only on the layer input at position `j`. The layer input `x_j` itself depends only on positions `≤ j`,
  because every attention below it is causally masked. Therefore appending token `t+1` changes nothing about
  `k_1..k_t`, `v_1..v_t`. Formally:
  `Attention_causal(Q,K,V) = softmax((QKᵀ + M)/√d_k) · V` with `M[i,j] = -∞ for j > i`.
  At decode step `t+1` the query is one vector `q_{t+1}` attending over `k_1..k_{t+1}`; no future key exists, so
  the mask is trivially satisfied and decode kernels drop the mask entirely.
- **Without cache vs with cache (token-forward-passes).** Generating `N` tokens without a cache re-runs the model
  over the whole prefix each step: `1 + 2 + … + N = N(N+1)/2` token-positions. With a cache: `N` token-positions.
  Ratio `(N+1)/2`. At **N = 1024 → 524,800 vs 1024 = 512.5×**.
- **Without cache vs with cache (FLOPs, per layer).** Using the standard per-layer costs
  (prefill of `T` tokens `≈ 7·B·T·d² + 2·B·T²·d`; one decode step at cache depth `t` `≈ 7·B·d² + 2·B·t·d`):
  ```
  with cache    : Σ_{t=1..N} (7Bd² + 2Btd)      ≈ 7BNd²  + BdN²        → O(N)  in the weight term
  without cache : Σ_{t=1..N} (7Btd² + 2Bt²d)    ≈ (7/2)Bd²N² + (2/3)BdN³ → O(N²) in the weight term
  ```
  Verified for `B=1, d=8192` (Llama-3-70B hidden size), per layer:
  | N | with cache (FLOPs) | without cache (FLOPs) | ratio |
  |---:|---:|---:|---:|
  | 128 | — | — | **64.5×** |
  | 256 | — | — | **128.7×** |
  | 1024 | 4.8963e11 | 2.5240e14 | **515.5×** |
  | 4096 | — | — | **2094×** |
  The ratio grows linearly in `N` — that is the whole point.
- **The cost of the trade.** The cache is a textbook space-for-time trade: `O(N)` extra *memory* buys the
  `O(N²)→O(N)` *compute* saving. Measured end-to-end on a 124M model, 200 tokens, Mac Mini M4 CPU (Raschka):
  no cache 17.5 s → naive cache 3.3 s (**5.3×**) → pre-allocated 2.8 s → pre-allocated + compiled 2.4 s (7.3×).
- **Minimal correct implementation** (Raschka, PyTorch):
  ```python
  def forward(self, x, use_cache=False):
      keys_new, values_new, queries = self.W_key(x), self.W_value(x), self.W_query(x)
      if use_cache:
          if self.cache_k is None:
              self.cache_k, self.cache_v = keys_new, values_new
          else:
              self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
              self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
          keys, values = self.cache_k, self.cache_v
      else:
          keys, values = keys_new, values_new
  ```
  Two bugs this code exposes: (1) **the cache must be reset between generations**, or "queries of a new prompt
  attend to stale keys left over from the previous sequence"; (2) **position IDs must be tracked**
  (`pos_ids = arange(current_pos, current_pos + seq_len)`), otherwise every new token is treated as position 0
  and RoPE breaks.
- **`torch.cat` vs pre-allocation.** `torch.cat` copies the whole cache each step: `O(n)` copy per step,
  `O(n²)` copies total, but only uses memory proportional to the real length. Pre-allocation
  (`torch.zeros(B, H_kv, max_seq_len, d_head)`) is `O(1)` per step but reserves the maximum: for a 128k-context
  model that is ~8 GB reserved even for a 50-token request. PagedAttention ([[paged-attention]]) exists to get
  both properties at once.
- **Training-memory angle:** This entire argument is *inference-only*, and knowing why is the point. The
  `O(N²)→O(N)` saving exists because decoding is a **sequential loop of N forward passes**. Training under
  teacher forcing runs **one** forward pass over all `s` positions — there is no loop, so there is no repeated
  work to amortize, and therefore no cache. In training the same K and V tensors are produced, but they live in
  the *activation* bucket (saved for backward, or recomputed under gradient checkpointing per
  [[gradient-checkpointing-chen]]) and are freed at the end of the step. A learner who assumes "attention →
  KV cache" will double-count memory in a training budget. See [[train-vs-infer-kv-boundary]].

## Citation
Ashish Vaswani et al., "Attention Is All You Need," NeurIPS 2017, arXiv:1706.03762,
https://arxiv.org/abs/1706.03762 · Sebastian Raschka, "Understanding and Coding the KV Cache in LLMs from
Scratch," Ahead of AI, 2025, https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms ·
Yi Tay et al., "Efficient Transformers: A Survey," 2020, arXiv:2009.06732. Synthesized from
`course/llm-inference:wiki/courses/llm-inference/ch-02/excerpts/attention-complexity.md` and
`llm-arch:wiki/courses/llm-arch/ch-25/excerpts/kv-cache-implementation.md`.
