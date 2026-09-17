---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: huggingface/transformers v4.46.0, src/transformers/modeling_rope_utils.py (chapter-local verified extract; no library card at the time of writing)
source_url: https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py
created_at: "2026-09-15"
---

# Excerpt: YaRN implementation in Hugging Face Transformers (v4.46.0)

- **Organization:** Hugging Face
- **Source type:** released code (framework implementation; values here are framework defaults, not a training recipe)
- **Used in:** ch-32b §2.3

## `_compute_yarn_parameters`, lines 188–237 (abridged; omitted lines are docstrings and a BC check)
```python
    base = config.rope_theta
    ...
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    # Sets the attention factor as suggested in the paper
    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
    ...
    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(device)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    return inv_freq, attention_factor
```

## Reading notes (derived from the code)
- `find_correction_dim(n, ...)` returns the dimension index at which a frequency completes n rotations over
  `max_position_embeddings` (the original context length when the config follows the YaRN convention).
- Dimensions below `low` (≥ `beta_fast` = 32 rotations) keep their original frequency; dimensions above `high`
  (≤ `beta_slow` = 1 rotation) are divided by `factor`; the ramp interpolates between them. This matches the
  NTK-by-parts rule in [[yarn]] (§3.2, α = 1, β = 32).
- The docstring (lines 179–180) describes the returned `attention_factor` as "the post-processing scaling factor
  applied to the computed cos/sin". Scaling both the rotated q and k by 0.1 ln(s) + 1 corresponds to √(1/t) in
  [[yarn]] Eq. 15.
- The scale factor is fixed when the config is loaded; it does not depend on the input length ("static" YaRN).

## Verification
- Checked on 2026-09-15 against the raw file at tag v4.46.0 (lines 163–237 read in full).
- Later Transformers versions may differ; the line numbers apply only to v4.46.0.
