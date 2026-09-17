---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Allen Institute for AI — allenai/OLMo repository: configs/official-1124/OLMo2-7B-stage1.yaml, olmo/config.py, olmo/train.py"
source_url: https://github.com/allenai/OLMo/tree/090253dac6688f2532509daa7aa2eb5fae50e956/configs/official-1124
created_at: "2026-09-15"
revised: "2026-09-15 (created; generality revision)"
---

# Excerpt: OLMo 2 7B stage-1 precision and stability settings (released config and code)

Source type: released config and code, read on 2026-09-15 at commit 090253dac6688f2532509daa7aa2eb5fae50e956
(main branch; the repository README states it is no longer active and points to OLMo-core). The report is
[[olmo-2]] (arXiv:2501.00656v3); that card has not yet been re-verified, so report loci below were read from
the primary text.

## configs/official-1124/OLMo2-7B-stage1.yaml
```yaml
# L18-24
  layer_norm_type: rms
  layer_norm_with_affine: true
  layer_norm_eps: 1e-6
  bias_for_layer_norm: false
  attention_layer_norm: true
  attention_layer_norm_with_affine: true
  norm_after: true
# L38-40
softmax_auxiliary_loss: true
auxiliary_loss_multiplier: 1e-5
fused_loss: true
# L44-53
optimizer:
  name: adamw
  learning_rate: 3.0e-4
  weight_decay: 0.1
  eps: 1e-8
  decay_norm_and_bias: true
  decay_embeddings: false
  betas:
  - 0.9
  - 0.95
# L84-90
precision: amp_bf16

fsdp:
  wrapping_strategy: by_block_and_size
  precision: mixed

max_grad_norm: 1.0
```

## olmo/config.py: what `fsdp.precision: mixed` means
```python
# olmo/config.py L836-840
    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """
# olmo/config.py L1339-1344
            elif self.fsdp.precision == FSDPPrecision.mixed:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=torch.float32,
                    buffer_dtype=self.autocast_precision,
                )
```
- With `precision: amp_bf16`, the training forward pass runs under `torch.autocast(..., dtype=bf16)`
  (olmo/train.py L811; evaluation at L915). Under the `mixed` setting, FSDP computes with bf16 parameters and
  buffers and reduces gradients across ranks in fp32 (config.py docstring above).

## olmo/train.py: z-loss
```python
# olmo/train.py L124-145
def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss
```
- `fused_loss: true` selects the FlashAttention Triton cross-entropy with `lse_square_scale=z_loss_multiplier`
  (L148-205, default 1e-4 at L162). The call in `model_forward` passes `compute_z_loss` but no multiplier
  (L749-751). `auxiliary_loss_multiplier` is defined in config.py (L1250, default 1e-4) and is not referenced
  in train.py or model.py at this commit, so the YAML value 1e-5 does not reach these loss functions.
- The report states z-loss as 10^-4 · log² Z (arXiv:2501.00656v3 §3.3.3) and lowers AdamW ε from 10^-5 to
  10^-8 (§3.4.1).

## Report passages that use these settings (arXiv:2501.00656v3)
- §3.3.3: the FlashAttention z-loss and a PyTorch implementation give the same forward value but different
  backward behavior, and the z-loss curves diverge (Fig. 8); "We suspect the root cause lies in differences in
  precision." Cross-entropy loss and downstream performance were not affected, but the team re-trained from the
  point of divergence and avoids switching implementations during a run.
- §3.4.1: ε = 10^-8 "lowers and stabilizes the norm of the gradient early in training" (Fig. 9).

## Used in ch-02
- §2 (bf16 autocast with fp32 gradient reduction), §8 (implementation-dependent numerics), Recipe rows.
