---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "allenai/OLMo configs/official-1124 (OLMo 2 7B and 13B stage-1 and stage-2 YAMLs) and olmo/optim.py, olmo/train.py at commit 090253dac6688f2532509daa7aa2eb5fae50e956 (no library card as of 2026-09-15; chapter-local verified extract)"
source_url: https://github.com/allenai/OLMo/tree/090253dac6688f2532509daa7aa2eb5fae50e956/configs/official-1124
created_at: "2026-09-15"
---

# Excerpt: OLMo 2 released schedule configs and the LR at the stage boundary

Source type: released config and code. The repository README at this commit states that the repository "is out of date with our more recent releases and is no longer active". The stage-1 configs were added in commit 796de60 ("Official configs for stage 1 training", 2024-11-26) and last edited in commit 600a8d0 (2025-07-16, "updated `save_num_checkpoints_to_keep=0` in all the configs").

## OLMo2-7B-stage1.yaml
```yaml
# L28
  max_sequence_length: 4096
# L45-53
  name: adamw
  learning_rate: 3.0e-4
  weight_decay: 0.1
  eps: 1e-8
  decay_norm_and_bias: true
  decay_embeddings: false
  betas:
  - 0.9
  - 0.95
# L57-61
  name: cosine_with_warmup
  units: tokens
  t_warmup: 8388608000
  t_max: 5e12
  alpha_f: 0.1
# L80-81, L90
max_duration: 1ep
global_train_batch_size: 1024
max_grad_norm: 1.0
```
Derived: 2,000 steps × 1,024 sequences × 4,096 tokens = 8,388,608,000 tokens, so `t_warmup` equals the paper's 2,000 warmup steps at the 7B batch.

## OLMo2-7B-stage2-seed42.yaml
```yaml
# L46
  learning_rate: 0.000061499
# L57-59
  name: linear_with_warmup
  t_warmup: 0
  alpha_f: 0
# L75
load_path: https://olmo-checkpoints.org/ai2-llm/peteish7/step928646-unsharded
# L80-82
max_duration: 50e9T
stop_at: 11931                  # round(50e9 / (1024 * 4096)) + 10
global_train_batch_size: 1024
```
Seeds 42069 and 666 are the other two 7B anneal ingredients (file names OLMo2-7B-stage2-seed42069.yaml, OLMo2-7B-stage2-seed666.yaml).

## 13B stage-1 and stage-2
- OLMo2-13B-stage1.yaml: `learning_rate: 3.0e-4` (L46); `units: tokens`, `t_warmup: 8388608000`, `t_max: 5e12`, `alpha_f: 0.1` (L58-61); `global_train_batch_size: 2048` (L81).
- OLMo2-13B-stage2-seed1110-100B.yaml: `learning_rate: 9e-5` (L44); `units: steps`, `linear_with_warmup`, `t_warmup: 0`, `alpha_f: 0` (L55-58); `load_path: .../peteish13/step596057-unsharded` (L71); `max_duration: 100e9T`, `stop_at: 11931` (L76-77).
- OLMo2-13B-stage2-seed2662-300B.yaml: `learning_rate: 9e-5` (L44); same load path (L71); `max_duration: 300e9T` (L76).

## Scheduler code (olmo/optim.py L694-709)
```python
class CosWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2
```
With `units: tokens`, the trainer passes `global_train_tokens_seen` as the current position (olmo/train.py L305-311).

## Derived checks used in ch-14a
- 7B: step 928,646 × 4,194,304 tokens = 3.895T tokens. The stage-1 cosine (peak 3e-4, floor 3e-5, warmup 8.3886B tokens, horizon 5T tokens) gives 6.135 × 10^−5 there and gives the YAML value 6.1499 × 10^−5 at step 928,000 (3.892T tokens). The 0.24% difference is not explained by the files.
- 13B peak LR: step 596,057 × 8,388,608 tokens = 5.0001T tokens, past the 5T horizon, where the cosine returns its floor 0.1 × peak. A 9e-4 peak gives 9e-5, the stage-2 start LR; the stage-1 YAML's 3e-4 would give 3e-5.
- 13B warmup: `t_warmup: 8388608000` tokens ÷ (2,048 × 4,096) = 1,000 steps at the 13B batch, while the report's Table 3 prints 2,000 warmup steps for 13B. The 13B stage-1 YAML carries the 7B warmup token count.

## Verification
- Read on 2026-09-15: the YAML files, olmo/optim.py, and olmo/train.py (cached from `main`, which was commit 090253d per the GitHub API; commit history of OLMo2-13B-stage1.yaml from the API).
- Not in these files: which stage-1 configuration produced the released 13B checkpoint; why the 7B stage-2 start LR matches step 928,000 rather than the load step.
