---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "karpathy/nanochat scripts/base_train.py and runs/speedrun.sh at commit f527f761460eab5e95d423b878f561a016603d2e (no library card as of 2026-09-15; chapter-local verified extract)"
source_url: https://github.com/karpathy/nanochat/tree/f527f761460eab5e95d423b878f561a016603d2e
created_at: "2026-09-15"
---

# Excerpt: nanochat — how a minimal pipeline sets the token budget, batch, LR, and schedule

Source type: released code, practitioner evidence (independent author, code with comments that refer to experiments in `dev/LOG.md`, which this excerpt did not read). Commit f527f76 (2026-07-03) is the latest commit touching `scripts/base_train.py`; the pinned file is byte-identical to the copy fetched from `main`.

## Horizon (scripts/base_train.py)
```python
# L58
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# L263-269
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train
```

## Batch, LR, and weight decay rules (selected lines from L271-302)
```python
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
```
`batch_size_ratio = target_tokens / D_REF`, where `D_REF` is the ratio times the scaling parameters of a depth-12 reference model (L273).

## Schedule (L62-69, L360-369)
```python
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac
```
Optimizer: Muon for matrix parameters (default LR 0.02) and AdamW for embeddings (0.3), unembedding (0.008), and scalars (0.5); weight decay default 0.28 (L62-66, L307-315). Weight decay follows its own cosine schedule to zero (L384-386):
```python
# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))
```

## runs/speedrun.sh
```bash
# L66-67
# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
```
L3-4: "configured to train your own GPT-2 grade LLM (pretraining + finetuning)", "on a blank 8XH100 GPU node and takes approximately 1.5 hours".

## Verification
- Read on 2026-09-15: base_train.py and speedrun.sh at f527f76 (fetched at the SHA and compared with `main` copies), commits API for base_train.py.
- Inconsistency: the speedrun comment calls 10.5 the default ratio; the parser default is 12.
- Not verified here: the experiments behind "measured empirically" and the D^0.383 exponent (dev/LOG.md and arXiv:2505.13738 not read).
