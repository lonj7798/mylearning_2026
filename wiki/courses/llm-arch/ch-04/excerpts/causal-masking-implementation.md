<!-- scope: causal mask implementation details, numerical stability, teacher forcing coupling, parent: [[ch-04]] -->

# Causal Masking: Implementation Deep-Dive

The causal mask is the single architectural element that separates a decoder-only Transformer from an encoder. This excerpt traces the implementation from the mathematical definition through GPU-level execution, covering numerical choices, the coupling with teacher forcing, and common implementation pitfalls.

---

## The Mask Matrix

The causal mask $M$ is an upper-triangular matrix of negative infinities applied to the raw attention scores before softmax:

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After adding $M$ to the scaled dot-product scores $S = QK^T / \sqrt{d_k}$, the softmax maps $-\infty$ entries to zero:

$$\text{softmax}(S + M)_{ij} = \frac{e^{S_{ij} + M_{ij}}}{\sum_k e^{S_{ik} + M_{ik}}} \xrightarrow{M_{ij} = -\infty} 0$$

This is a hard mask -- future tokens contribute exactly zero to the weighted sum over values.

---

## Numerical Stability: Why GPT-2 Uses $-10^9$ Instead of $-\infty$

The GPT-2 implementation ([[alammar-illustrated-gpt2|blog]]) uses $-10^9$ (i.e., `-1e9`) rather than IEEE floating-point negative infinity (`-inf`). The reason is numerical stability in mixed-precision training:

1. **FP16 overflow**: In half-precision, the largest representable number is $\sim 65504$. True $-\infty$ in FP32 cast to FP16 becomes `-inf`, which propagates through softmax correctly. But intermediate computations (e.g., subtracting the max for numerical stability) can produce `nan` if `-inf` participates in arithmetic with finite values.

2. **Safe substitute**: $e^{-10^9} \approx 0$ in any floating-point format. The value is large enough to guarantee zero attention weight after softmax but small enough to avoid overflow in intermediate steps.

3. **Modern practice**: PyTorch's `torch.finfo(dtype).min` returns the most negative finite value for the dtype, which is the current best practice. FlashAttention handles masking internally with fused kernels that avoid materializing the full mask matrix entirely.

```python
# GPT-2 style (HuggingFace implementation)
mask_value = torch.tensor(-1e9, dtype=attn_weights.dtype)
attn_weights = torch.where(causal_mask, attn_weights, mask_value)

# Modern best practice
mask_value = torch.finfo(attn_weights.dtype).min
attn_weights = attn_weights.masked_fill(~causal_mask, mask_value)
```

---

## Coupling with Teacher Forcing

The causal mask is not just a training trick -- it is the mechanism that makes teacher forcing possible. During training, the model processes an entire sequence of $T$ tokens in a single forward pass and produces $T$ predictions simultaneously. This parallelism is only valid because the causal mask guarantees that position $t$'s representation depends exclusively on positions $1, \ldots, t$.

Without the mask, position $t$'s representation would include information from the ground-truth token at position $t+1$ (and beyond), making the prediction trivially easy and the training signal meaningless. The mask enforces the autoregressive factorization ([[ch-01]]):

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

at the architectural level, ensuring that training-time parallelism produces the same representations as sequential inference.

**The training-inference equivalence**: Because of the causal mask, the hidden state at position $t$ during training is identical to the hidden state at position $t$ during autoregressive inference (assuming the same prefix). This equivalence is what makes the KV cache ([[ch-25]]) valid -- cached key-value pairs from earlier positions can be reused without recomputation.

---

## Implementation Patterns

### Static Triangular Mask (Pre-computed)

The simplest approach pre-computes a boolean lower-triangular matrix at model initialization:

```python
# Pre-computed mask (registered as a buffer, not a parameter)
causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
# Shape: [max_seq_len, max_seq_len]
# causal_mask[i, j] = True iff j <= i
```

This is memory-efficient (boolean tensor) and avoids recomputation. GPT-2 registers it as a non-trainable buffer.

### Dynamic Mask (Computed Per Forward Pass)

For variable-length sequences or when the maximum length is not known at initialization:

```python
def make_causal_mask(seq_len, device):
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # True = masked (future), False = visible (past/current)
```

### Flash Attention Integration

FlashAttention ([[ch-07]]) never materializes the $N \times N$ mask matrix. Instead, it applies causal masking implicitly within its tiled computation: for each query block, it simply skips key blocks that are entirely in the future. For partially-masked blocks (on the diagonal), masking is applied within the SRAM tile. This saves $O(N^2)$ memory and avoids the HBM round-trip for the mask tensor.

---

## Common Pitfalls

1. **Off-by-one in the diagonal**: The mask must include the diagonal ($j = i$), allowing each token to attend to itself. An off-by-one error here causes each token to attend only to strictly preceding tokens, losing the self-attention signal.

2. **Mask broadcasting with batched inputs**: The mask is typically `[1, 1, seq_len, seq_len]` and broadcast across `[batch, heads, seq_len, seq_len]`. Incorrect broadcasting silently corrupts attention patterns without raising errors.

3. **Padding interaction**: In batched training with variable-length sequences, the causal mask must be combined with a padding mask that zeros out attention to pad tokens. These are logically distinct -- causal mask prevents future attention; padding mask prevents attention to non-existent tokens.

4. **KV cache during inference**: During autoregressive generation, only the current token's query attends to all cached keys. The causal mask during inference is trivially satisfied (the query is always the latest position), so no explicit mask is needed -- but the implementation must correctly handle the asymmetric shapes of the query (length 1) and key cache (length $t$).

---

## References

- [[gpt-1|Radford et al. "Improving Language Understanding by Generative Pre-Training" (2018) (paper)]]
- [[gpt-2|Radford et al. "Language Models are Unsupervised Multitask Learners" (2019) (paper)]]
- [[alammar-illustrated-gpt2|Alammar, "The Illustrated GPT-2" (blog)]]
