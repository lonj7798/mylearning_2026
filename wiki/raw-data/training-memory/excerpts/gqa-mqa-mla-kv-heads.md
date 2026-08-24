# MHA → GQA → MQA → MLA: The `n_kv_heads` Divisor, and Why It Is Weaker in Training
<!-- slug: gqa-mqa-mla-kv-heads · type: paper · source: https://arxiv.org/abs/2305.13245 -->

**Core Insight.** MHA, GQA and MQA differ in **exactly one number**: `n_kv_heads`. MHA sets `n_kv_heads = n_heads`
(divisor 1), GQA-`G` sets `n_kv_heads = G` (divisor `n_heads/G`), MQA sets `n_kv_heads = 1` (divisor `n_heads`).
MLA breaks the family entirely by caching a single low-rank latent per token instead of per-head K and V. The
divisor applies **fully to the inference KV cache but only partially to training activations**, because the Q
tensor is untouched by all of these.

**Guideline.** Read `num_key_value_heads` before quoting any KV number. For serving, `G = 8` is the empirical
sweet spot (quality within noise of MHA, `n_heads/8` cache reduction). For a *training* memory budget, do not
reuse the inference divisor — recompute it as `3·H_q / (H_q + 2·H_kv)`, which is ~2–3× regardless of how
aggressive the GQA is.

## Technical Details

- **The three configurations.** Each layer projects `x ∈ ℝ^d` into `H_q` query heads and `H_kv` key/value heads.
  | Variant | `H_q` | `H_kv` | KV-cache divisor vs MHA | Quality vs MHA |
  |---|---:|---:|---:|---|
  | **MHA** (Vaswani 2017) | `H` | `H` | **1×** | baseline |
  | **GQA-G** (Ainslie 2023) | `H` | `G` | **`H/G`×** | within noise for `G ≥ 8` |
  | **MQA** (Shazeer 2019) | `H` | `1` | **`H`×** | −1 to −2 ppl |
  Query head `h` reads from KV head `h // (H_q / G)`; each KV head serves `H_q / G` query heads. Llama-3-70B is
  GQA-8 with `H_q = 64, H_kv = 8` → 8 query heads per KV head.
- **The concrete divisor payoff (Llama-3-70B, bf16, verified):** GQA-8 `2·80·8·128·2 = 327,680 B/token`
  (320 KiB) vs MHA-64 `2·80·64·128·2 = 2,621,440 B/token` (2.5 MiB) — **exactly 8×**. At 32k context that is
  10.00 GiB/request vs 80 GiB/request; the MHA version does not fit one request on an 80 GB H100.
- **Llama-3 family rule: `kv_heads = 8` at every size.** The replication ratio grows with model size, the cache
  does not follow parameter count:
  | Size | `L` | `H_q` | `H_kv` | ratio | bytes/token (bf16) |
  |---|---:|---:|---:|---:|---:|
  | 8B | 32 | 32 | 8 | 4:1 | 131,072 |
  | 70B | 80 | 64 | 8 | 8:1 | 327,680 |
  | 405B | 126 | 128 | 8 | 16:1 | 516,096 |
  8B→405B is 50× the parameters but only **3.9×** the KV cache, because only `L` grew.
- **Why MQA costs quality.** One shared K/V projection forces every query head to look for and retrieve the same
  thing; MHA's `H` distinct projections let heads specialise. Reported PaLM-540B-era drops: HumanEval −2.2,
  GSM8K −1.5, MMLU −0.6.
- **Uptraining (Ainslie's second contribution).** Convert an existing MHA checkpoint by mean-pooling KV heads
  into `G` groups, then continue pretraining with **~5% of the original pretraining compute**; quality recovers
  to within ~0.1 perplexity of training GQA from scratch. This is how Llama-2-70B got GQA-8.
  ```python
  K_g = W_K.reshape(d, H_q, d_head).reshape(d, G, H_q//G, d_head).mean(dim=2).reshape(d, G*d_head)
  ```
- **MLA (DeepSeek-V3) — outside the family.** Instead of per-head K and V, cache one latent `c_KV ∈ ℝ^{d_c}` per
  token per layer plus a small decoupled RoPE key slice `k_rope ∈ ℝ^{d_rope}` (shared, not per-head). `W_UK` and
  `W_UV` reconstruct K and V on read and are absorbed into `W_Q` / `W_O` offline. **There is no factor 2** — one
  latent serves both K and V:
  ```
  MLA bytes/token = L · (d_c + d_rope) · bytes_per_element
  V3 (L=61, d_c=512, d_rope=64, bf16) = 61 · 576 · 2 = 70,272 B = 68.6 KiB/token
  naive MHA at same geometry (128 heads, d_head 128) = 2 · 61 · 128 · 128 · 2 = 3,997,696 B = 3.81 MiB/token
  compression ratio = 3,997,696 / 70,272 = 56.9×
  ```
  `k_rope` is kept uncompressed because RoPE is position-dependent: rotating before compression would make the
  latent position-specific and destroy reuse.
- **Training-memory angle:** **The GQA divisor mostly evaporates in training.** In a training forward pass all
  three of Q, K, V are activations saved for backward, and GQA shrinks only K and V. The correct training
  divisor is
  ```
  training QKV-activation divisor = 3·H_q / (H_q + 2·H_kv)
  ```
  | Config | inference KV divisor | training QKV-activation divisor |
  |---|---:|---:|
  | Llama-3-8B (32→8) | **4×** | **2.00×** |
  | Llama-3-70B (64→8) | **8×** | **2.40×** |
  | MQA `H_q=64 → 1` | **64×** | **2.91×** |
  Verified per layer at `B=1, s=8192, d_head=128, bf16`: GQA-8 stores `Q 128 MiB + K 16 MiB + V 16 MiB =
  160 MiB/layer` (12.5 GiB over 80 layers); MHA-64 stores `128+128+128 = 384 MiB/layer` (30.0 GiB over 80).
  So an architecture chosen for a 64× serving win buys under 3× on the training activation bill — the
  training-side levers remain FlashAttention ([[flash-attention-2]]) and recomputation
  ([[selective-recompute-korthikanti]]), not the KV-head count. Corollary for MoE budgets: switching
  attention variant is a *serving* decision that should not be priced into a training OOM fix.

## Citation
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai. "GQA:
Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023, arXiv:2305.13245,
https://arxiv.org/abs/2305.13245 · Noam Shazeer, "Fast Transformer Decoding: One Write-Head Is All You Need,"
2019, arXiv:1911.02150 · DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024, arXiv:2412.19437. Assembled from
`course/llm-inference:wiki/courses/llm-inference/ch-02/excerpts/mha-mqa-gqa.md`,
`.../ch-20/excerpts/llama-3-gqa-and-kv.md`, `.../ch-20/excerpts/deepseek-mla-compression.md`, and
`.../wiki/raw-data/llm-inference/classics/grouped-query-attention.md`.
