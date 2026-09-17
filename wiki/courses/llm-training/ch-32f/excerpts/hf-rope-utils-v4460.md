---
chapter: ch-32f
course: llm-training
phase: read
excerpt_of: released code, huggingface/transformers tag v4.46.0, src/transformers/modeling_rope_utils.py and src/transformers/models/llama/modeling_llama.py (no library card for these files as of 2026-09-15)
source_url: https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py
created_at: "2026-09-15"
---

# Excerpt: RoPE scaling code in transformers v4.46.0

Source type: released config/code. Fetched from raw.githubusercontent.com at tag v4.46.0 on 2026-09-15. Line numbers refer to that tag.

## Dynamic NTK (`modeling_rope_utils.py` L112-160, abridged)

```python
    base = config.rope_theta                                                         # L145
    ...
    max_position_embeddings = config.max_position_embeddings                        # L149
    factor = config.rope_scaling["factor"]                                          # L150
    ...
    # seq_len: default to max_position_embeddings, e.g. at init time
    seq_len = seq_len if seq_len is not None and seq_len > max_position_embeddings else max_position_embeddings  # L155
    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))            # L158
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))            # L159
```

At `seq_len = max_position_embeddings` the bracket equals 1, so the base is unchanged; it grows only for longer inputs.

## YaRN (`modeling_rope_utils.py` L163-239, abridged)

```python
    base = config.rope_theta                                                         # L188
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)  # L190
    dim = int(head_dim * partial_rotary_factor)                                      # L191
    max_position_embeddings = config.max_position_embeddings                        # L192
    factor = config.rope_scaling["factor"]                                          # L193
    attention_factor = config.rope_scaling.get("attention_factor")                  # L196
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0                              # L198
    beta_fast = config.rope_scaling.get("beta_fast") or 32                           # L202
    beta_slow = config.rope_scaling.get("beta_slow") or 1                            # L203

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))  # L208

    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)           # L226
    inv_freq_extrapolation = 1.0 / pos_freqs                                         # L227
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)                              # L228
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)  # L230
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(device)  # L233
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )                                                                                # L234-237
    return inv_freq, attention_factor                                                # L239
```

In this version the ramp bounds use `config.max_position_embeddings` as the pretrained length (L192, L230). `_validate_yarn_parameters` accepts only `rope_type`, `factor`, `attention_factor`, `beta_fast`, `beta_slow` (L432-433). The dynamic NTK function carries the comment "TODO (joao): use the new `original_max_position_embeddings` from rope_scaling" (L133).

## Where the attention factor is applied (`models/llama/modeling_llama.py` L147-167)

```python
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        ...
        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
```

Both q and k are rotated with the scaled cos and sin, so attention logits are multiplied by `attention_scaling`² at every sequence length, including inputs shorter than the pretrained length (derived from the code; for factor 4, (0.1 ln 4 + 1)² = 1.2965).

## Used in

ch-32f §4.2 (YaRN configuration and the ramp-bound check), Common mistakes.
