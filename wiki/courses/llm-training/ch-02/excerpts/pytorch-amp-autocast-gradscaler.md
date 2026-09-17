---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "PyTorch — torch.amp documentation (docs/source/amp.md) and torch/amp/grad_scaler.py, release v2.12.0"
source_url: https://github.com/pytorch/pytorch/blob/v2.12.0/docs/source/amp.md
created_at: "2026-09-15"
revised: "2026-09-15 (created; generality revision)"
---

# Excerpt: PyTorch automatic mixed precision — autocast op lists and GradScaler defaults (v2.12.0)

Source type: released code and documentation (framework defaults, not values from a training run).
Read on 2026-09-15: `docs/source/amp.md` at tag v2.12.0 and `torch/amp/grad_scaler.py` from the installed
torch 2.12.0 wheel (git 7661cd9c6b841b62b7f411aa52ec51f05457263b).

## Autocast policy (amp.md)
- "Some ops, like linear layers and convolutions, are much faster in `lower_precision_fp`. Other ops, like
  reductions, often require the dynamic range of `float32`." `lower_precision_fp` is `torch.float16` or
  `torch.bfloat16` (L28-32).
- Ops not listed "do not go through autocasting. They run in the type defined by their inputs" (L186-188).
  "If an op is unlisted, we assume it's numerically stable in `float16`" (L190).
- Ops called with an explicit `dtype=` argument and in-place ops do not autocast (L161-175).

**CUDA ops that autocast to float16 (L194-218):** `__matmul__`, `addbmm`, `addmm`, `addmv`, `addr`, `baddbmm`,
`bmm`, `chain_matmul`, `multi_dot`, `conv1d`-`conv3d`, `conv_transpose1d`-`3d`, `GRUCell`, `linear`, `LSTMCell`,
`matmul`, `mm`, `mv`, `prelu`, `RNNCell`.

**CUDA ops that autocast to float32 (L220-272), selected:** `cross_entropy`, `nll_loss`, `log_softmax`,
`softmax`, `layer_norm`, `group_norm`, `norm`, `normalize`, `sum`, `prod`, `cumsum`, `cumprod`, `exp`, `log`,
`pow`, `rsqrt`, `kl_div`, `mse_loss`, `binary_cross_entropy_with_logits`.

**CUDA ops that promote to the widest input type (L274-290):** `addcdiv`, `addcmul`, `atan2`, `bilinear`, `cross`,
`dot`, `grid_sample`, `index_put`, `scatter_add`, `tensordot`.

- `scaled_dot_product_attention` appears in the CPU bfloat16 list (L439) and not in the CUDA lists, so on CUDA
  it runs in the dtype of its inputs. `rms_norm` does not appear in any list on this page.
- The CUDA lists are written with float16 as the lower-precision type; the page does not print separate CUDA
  lists for bfloat16.
- `binary_cross_entropy` raises an error under autocast because its backward can produce gradients not
  representable in float16 (L297-307).

## GradScaler (torch/amp/grad_scaler.py)
```python
# torch/amp/grad_scaler.py L123-131 (torch 2.12.0)
def __init__(
    self,
    device: str = "cuda",
    init_scale: float = 2.0**16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    enabled: bool = True,
) -> None:
```
- Docstring (L96-101): if infs or NaNs are found, `scaler.step(optimizer)` skips `optimizer.step()` "so the
  params themselves remain uncorrupted" and `update()` multiplies the scale by `backoff_factor`; after
  `growth_interval` consecutive unskipped iterations, the scale is multiplied by `growth_factor`.
- Docstring (L103-105): the scale "often causes infs/NaNs to appear in gradients for the first few iterations as
  its value calibrates"; after that, skipping "should occur rarely (once every few hundred or thousand
  iterations)".
- The standard float16 recipe pairs autocast with GradScaler; bfloat16 autocast is shown without GradScaler in
  the CPU example (amp.md L34-39).

## Used in ch-02
- §2 (which operations a framework keeps in float32, and that this is a framework policy), §3 (dynamic loss
  scaling defaults), Recipe row for GradScaler.
