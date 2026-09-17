---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "PyTorch v2.5.0 source: torch/optim/adamw.py, torch/optim/adam.py, torch/nn/utils/clip_grad.py, torch/amp/grad_scaler.py, docs/source/notes/amp_examples.rst"
source_url: https://github.com/pytorch/pytorch/tree/v2.5.0
created_at: "2026-09-15"
---

# Excerpt: PyTorch AdamW, L2 in Adam, global-norm clipping, and GradScaler ordering (v2.5.0)

**Source type:** released code and official documentation (PyTorch, tag v2.5.0). No library card.
**Status:** lines copied from raw.githubusercontent.com/pytorch/pytorch/v2.5.0/... on 2026-09-15. Later PyTorch releases may reorganize these files; line numbers refer to v2.5.0.

## AdamW defaults (`torch/optim/adamw.py` L33-L47)
`lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2`.

## AdamW single-tensor step (`torch/optim/adamw.py` L368-L429, non-capturable branch, AMSGrad lines omitted)
```python
# update step
step_t += 1

# Perform stepweight decay
param.mul_(1 - lr * weight_decay)

# Decay the first and second moment running average coefficient
exp_avg.lerp_(grad, 1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
...
step = _get_value(step_t)

bias_correction1 = 1 - beta1**step
bias_correction2 = 1 - beta2**step

step_size = lr / bias_correction1

bias_correction2_sqrt = bias_correction2**0.5
...
denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

param.addcdiv_(exp_avg, denom, value=-step_size)
```
- `exp_avg.lerp_(grad, 1 - beta1)` computes m ← m + (1 − β1)(g − m) = β1·m + (1 − β1)·g.
- The decay multiplies the parameter by (1 − lr·weight_decay) before the Adam step, so the per-step decay is lr·λ.
- ε is added to √v̂ after bias correction.

## Adam with weight_decay is L2 regularization (`torch/optim/adam.py`)
- Default `weight_decay: float = 0` (L39); docstring: "weight_decay (float, optional): weight decay (L2 penalty) (default: 0)" (L300).
- Single-tensor step, L366-L367:
```python
if weight_decay != 0:
    grad = grad.add(param, alpha=weight_decay)
```
The decay term enters the gradient before the moment estimates, so it is divided by √v̂ like the loss gradient.

## Global-norm clipping (`torch/nn/utils/clip_grad.py` L94-L109, v2.5.0)
```python
total_norm = torch.linalg.vector_norm(
    torch.stack([norm.to(first_device) for norm in norms]), norm_type
)
...
clip_coef = max_norm / (total_norm + 1e-6)
# Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
# avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
# when the gradients do not reside in CPU memory.
clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
```
- Docstring (L46-L47): "The norm is computed over the norms of the individual gradients of all parameters, as if the norms of the individual gradients were concatenated into a single vector."
- The function returns `total_norm`, the norm before clipping (L124).

## GradScaler initial scale (`torch/amp/grad_scaler.py` L107, L122)
`init_scale: float = 2.0**16` ("Initial scale factor").

## Ordering with loss scaling (`docs/source/notes/amp_examples.rst` L67-L106)
"All gradients produced by `scaler.scale(loss).backward()` are scaled. If you wish to modify or inspect the parameters' `.grad` attributes between `backward()` and `scaler.step(optimizer)`, you should unscale them first. ... If you attempted to clip *without* unscaling, the gradients' norm/maximum magnitude would also be scaled, so your requested threshold (which was meant to be the threshold for *unscaled* gradients) would be invalid."
```python
scaler.scale(loss).backward()

# Unscales the gradients of optimizer's assigned params in-place
scaler.unscale_(optimizer)

# Since the gradients of optimizer's assigned params are unscaled, clips as usual:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# optimizer's gradients are already unscaled, so scaler.step does not unscale them,
# although it still skips optimizer.step() if the gradients contain infs or NaNs.
scaler.step(optimizer)

# Updates the scale for next iteration.
scaler.update()
```
Warning (L114-L117): `unscale_` "should only be called once per optimizer per step call, and only after all gradients for that optimizer's assigned parameters have been accumulated."

## Verification
- Fetched on 2026-09-15 from raw.githubusercontent.com at tag v2.5.0; line numbers counted in the fetched files.
