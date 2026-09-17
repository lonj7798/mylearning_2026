---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "OLMo 2 Furious (arXiv:2501.00656) §3, §3.4, Tables 1 and 3; allenai/OLMo configs/official-1124/OLMo2-7B-stage1.yaml"
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-09-15"
---

# Excerpt: OLMo 2 optimizer settings and stability ablations

**Report:** Team OLMo, "2 OLMo 2 Furious". arXiv:2501.00656 (read v3, 2025-10-08). Source type: official technical report.
**Config:** github.com/allenai/OLMo @ 600a8d096b8b77a3fdf126c8ad0f7c8e56b3ea91, `configs/official-1124/OLMo2-7B-stage1.yaml`. Source type: released config.
**Status:** the library card [[olmo-2]] had no Verification section on 2026-09-15 and does not list these optimizer values; ch-01 takes them from the files above.

## Released config, OLMo2-7B-stage1.yaml (L44-L62, L90)
```yaml
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
  metrics_log_interval: 1

scheduler:
  name: cosine_with_warmup
  units: tokens
  t_warmup: 8388608000
  t_max: 5e12
  alpha_f: 0.1
  warmup_min_lr: 0.0
...
max_grad_norm: 1.0
```
Table 3 of the report prints for OLMo 2 7B: batch size 1024, sequence length 4096, gradient clipping 1.0, peak LR 3.0·10⁻⁴, warmup 2000 steps, cosine schedule over 5T tokens truncated after 4T. (8,388,608,000 warmup tokens / (1024 × 4096 tokens per step) = 2,000 steps.)

## Stability findings (§3, Figure 2)
- OLMo-0424 had "Sudden spikes in the loss, and more frequently, in the gradient norm during training"; "more dramatic spikes in gradient norm often preceded training loss spikes"; model size increased spike frequency.
- Spike score (§3.2): "the percentage of values in a time series that are at least seven standard deviations away from a rolling average of the last 1,000 values", computed on training loss and gradient L2 norm. The new initialization changed the gradient-norm spike score from 0.40 to 0.03 in the ablation.

## ε in AdamW (§3.4.1, Figure 9)
"Figure 9 shows the result of decreasing the AdamW ε from 10⁻⁵ to 10⁻⁸. 10⁻⁸ is the default in PyTorch, but some popular LM training code bases come with a default of 10⁻⁵. The lower value allows for larger updates early in training, and helps the model learn faster during a period where we've typically seen a lot of instability. As a result, the gradient norm settles much more quickly and remains permanently lower." The figure spans about 8,000 steps; the model size of the ablation is not stated in §3.4.1.

## Weight decay on embeddings (§3.4.2, Figure 10)
"OLMo uses a standard formulation of weight decay, where every parameter is multiplied by 1 − (0.1 · lr) at every step." For token embeddings "it overshoots the mark and results in very small embeddings"; small embeddings give large gradients in early layers because the Jacobian of layer_norm(x) is inversely proportional to ‖x‖. Remedy: no weight decay on embeddings. Figure 10: "Decaying embeddings also has a modest negative impact on stability, producing more spikes than a comparable run without (spike scores of 0.16 and 0.092 respectively)."

## Learning-rate observation (§4.1, Figure 11)
For the 7B setting, peak LRs of 6, 9, 12, and 30 · 10⁻⁴ were compared with 3 · 10⁻⁴; 30 · 10⁻⁴ had loss spikes during warmup; higher LRs had lower training loss early, "but eventually the lower learning rate setting overtakes the others"; the crossover between 3 · 10⁻⁴ and 6 · 10⁻⁴ is "well past 200B tokens".

## Verification
- Checked on 2026-09-15 against arXiv:2501.00656v3 (§2.1 Table 1, §2.3 Table 3, §3, §3.2, §3.4.1-§3.4.2, §4.1) and the config file at the commit above.
