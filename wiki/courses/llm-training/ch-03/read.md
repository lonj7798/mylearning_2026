<!-- chapter: ch-03
     track: foundations
     title: LR Schedules, Weight Init, Norms
     sources: [[lr-schedules]], [[weight-init]], [[batch-vs-layer-norm]], [[adam]], [[mixed-precision]]
     figures: figures/lr-schedules.html
-->

# Chapter 3 — LR Schedules, Weight Init, Norms

> **Core insight.** Three seemingly-separate concerns — *when* the step size is large, *how* the weights start, and *what* keeps activations bounded — are the same problem seen from three angles: *maintain a well-conditioned forward/backward pass across training and across depth*. Break any one and the others can't save you.
>
> **Guideline.** For a new Transformer in 2025: cosine or WSD schedule with linear warmup (2000–8000 steps); initialise all linear layers `N(0, 0.02)` with residual-projection scaling `1/√(2L)`; RMSNorm + pre-norm + final `ln_f`; norm reductions in fp32.

---

## Why this chapter exists

You have the optimizer ([[adam]]) and the precision stack ([[mixed-precision]]) from ch-01 / ch-02. None of that matters if the *shape* of training is wrong from step 1: if LR is too high before `v̂` has stabilised, if init scale makes the first forward pass saturate, if the norm placement lets residual-stream variance grow with depth. These three knobs interact. This chapter nails the defaults, explains why they are what they are, and flags the failure modes.

Primary sources: [[lr-schedules]], [[weight-init]], [[batch-vs-layer-norm]].

---

## 1. Learning-rate schedules — four families that actually matter

Sources: [[lr-schedules]]. Four families dominate modern LLM training.

**Linear warmup** — always prepended:

```
lr(t) = peak_lr · t / warmup_steps     for  t < warmup_steps
```

**Cosine annealing** — the LLM pretraining default:

```
lr(t) = min_lr + 0.5 · (peak_lr − min_lr) · (1 + cos(π · (t − warmup) / (T − warmup)))
```

Min LR is typically `0.1 × peak_lr` (Llama, GPT-3) or `0.0` (some Qwen runs). `T` is the total-step budget. Cosine's only failure mode is a mismatched horizon (Chinchilla's Figure A1): cosine-to-zero at step T but training stopped at 0.5T leaves you ~0.5% worse on val loss.

**Inverse-square-root** — Vaswani 2017's original:

```
lr(t) = d_model^(−0.5) · min(t^(−0.5), t · warmup_steps^(−1.5))
```

Self-scales with model width, elegant in theory; in fixed-budget practice cosine wins by ~0.3%. Still used in some encoder pretrains (T5 variants) but not frontier LLMs.

**WSD — Warmup-Stable-Decay** (Hu 2024, DeepSeek):

```
phase 1 (warmup):  lr ramps 0 → peak_lr           [0, warmup]
phase 2 (stable):  lr = peak_lr                   [warmup, T − decay]
phase 3 (decay):   lr → min_lr  (10–20% of T)     [T − decay, T]
```

WSD's signature advantage: the stable phase is *checkpoint-able*. You can fork off a 10%-decay run from any stable-phase checkpoint, getting "final loss" at multiple training lengths without retraining. This is how DeepSeek and MiniCPM produce many model variants from one trunk.

See `figures/lr-schedules.html` for an interactive comparison. Slide the knobs and watch the three schedules co-plot.

### Modern defaults

| Setting | Pretrain | SFT | RL (PPO/GRPO) |
|---|---|---|---|
| Warmup steps | 2000–8000 | 100–500 (~3% of total) | 0–50 |
| Schedule | cosine or WSD | cosine or constant | constant |
| Peak LR | 3e-4 (1B) → 1.2e-4 (70B) → 8e-5 (405B) | 2e-5 (Llama-3 SFT) | 1e-6 – 1e-5 |
| Min LR | 0.1 × peak | 0.1 × peak | — |

**Why warmup is non-negotiable with AdamW.** `v̂ = v_t / (1 − β₂ᵗ)` is poorly estimated in the first few steps. The bias-correction denominator is small, so the effective LR is inflated. Without warmup, the first updates can NaN. GPT-3 used 375M tokens of warmup; Llama-3 used 8000 steps. Zero warmup on a 7B+ model diverges. See ch-01's discussion of bias-corrected `v̂`.

**Common pitfalls.** Cosine sized to the wrong horizon → Chinchilla-style penalty. Warmup too short with high LR → loss spike at step ~150. Constant-LR finetuning "forever" → final loss is 1–3% worse than a proper decay. WSD decay phase < 5% → underperforms cosine.

---

## 2. Weight initialization — the three rules you cannot skip

Sources: [[weight-init]]. Two principles ground everything: **variance preservation** across layers, and **residual-stream budget** across depth.

**Variance preservation (Xavier/He).** For a linear layer `y = Wx`:

```
Xavier (tanh/linear):  Var(W) = 2 / (fan_in + fan_out)
He     (ReLU):         Var(W) = 2 / fan_in
LeCun  (SELU):         Var(W) = 1 / fan_in
```

Transformers use GELU / SwiGLU variants that sit between linear and ReLU. In practice modern code doesn't derive variance from activation — it uses the **GPT-2 / Megatron rule** directly:

```
all linear layers:   N(0, 0.02)
all embeddings:      N(0, 0.02)   (some recipes use N(0, 1e-5) for shared LM-head)
residual projections: additionally scaled by 1 / √(2L)    (L = # of residual blocks)
```

The `1/√(2L)` scale is the GPT-2 trick. Without it, residual-stream variance grows *linearly* in depth; at 100+ blocks the LM head sees unnormalised logits. Every modern frontier LLM applies it (Llama, Qwen, Megatron at 530B, OLMo-2).

**Embedding init.** Shared LM-head-with-embedding causes large gradients if the embedding is initialised at default PyTorch scale. Use `N(0, 0.02)` to match the linear layers, or scale down to `N(0, 1e-5)` if tying.

**The quick init audit.** Before training: forward a batch through the un-trained model. Activation variance should be roughly preserved across blocks (within 2×). Backward gradient norms should be within one order of magnitude across layers. **Initial loss should equal `ln(vocab_size)`** — the uniform-prediction baseline. If it's anything else, init is wrong and training will fight itself.

### μP (muTransfer) — hyperparameter transfer

Sources: [[weight-init]]. The pitch is simple: re-parametrize the network so the *optimal* LR, init scale, and betas are **width-invariant**. Sweep on a 40M-parameter proxy model, transfer to 6B+ unchanged. This is how Cerebras-GPT and part of GPT-4's HP sweep saved compute.

The abc-parametrisation (simplified):

```
input layer init:    O(1)
hidden layer init:   O(1/√d)
output layer init:   O(1/d)
hidden LR (AdamW):   O(1)    (SGD requires O(1/d))
output multiplier:   1/d applied to logits
```

**What transfers**: peak LR, betas, init scale, schedule shape.
**What doesn't**: depth-dependent quantities, batch size (still Chinchilla-scaled), data mix.

Pitfall: mix μP and non-μP layers (forget to scale the LM head) → transfer property is destroyed.

---

## 3. Norms — placement and formula

Sources: [[batch-vs-layer-norm]]. Three choices matter: *which* normaliser, *where* it sits relative to the residual, and *what precision* the reduction uses.

### LayerNorm vs RMSNorm

```python
# LayerNorm (Ba 2016)
mu     = x.mean(dim=-1, keepdim=True)
sigma2 = x.var(dim=-1, keepdim=True, unbiased=False)
y      = (x - mu) / torch.sqrt(sigma2 + eps) * gamma + beta

# RMSNorm (Zhang & Sennrich 2019)
rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
y   = x * rms * gamma      # no mean subtract, no beta
```

RMSNorm drops the mean subtraction and the learnable bias — saves one reduction, one subtract, one parameter group. The original RMSNorm paper reports 7–64% norm-op speedup with no quality loss. Llama, Qwen, DeepSeek, OLMo, Mistral, and Gemma all use RMSNorm. **Never invent your own norm.**

### Pre-norm vs post-norm

```
# Post-norm (Vaswani 2017 original)
x = LN(x + Sublayer(x))

# Pre-norm (every modern LLM)
x = x + Sublayer(LN(x))
```

Pre-norm keeps the residual-stream gradient `O(1)` with depth; post-norm's gradient grows linearly in depth and requires delicate warmup to avoid blowing up. At 24+ layers, post-norm training *without* exotic tricks diverges around step 1k. The price of pre-norm: residual-stream magnitude grows with depth, so you need a final `ln_f` before the LM head — which every modern architecture has.

### 2024–2025 variants

- **QK-norm.** Apply `LayerNorm(Q)` and `LayerNorm(K)` before the attention dot product. Prevents attention-logit explosion on long contexts. Used by ViT-22B, OLMo-2, Qwen-2.5.
- **Reordered-norm (OLMo-2).** Place the second norm *after* the residual add for the MLP; empirically eliminated mid-training loss spikes.
- **Sandwich-norm.** Norm both before and after each sub-layer. Marginal gains; niche.
- **DeepNorm** — for very deep *post-norm* stacks. Not used in mainstream 2025 LLMs.

### Precision rule (cross-link to ch-02)

The `mean` and `var` reductions inside LayerNorm/RMSNorm **must** run in fp32 even when the surrounding compute is bf16 or fp8. See the `RMSNorm.forward` snippet in ch-02 §3. This is the single most frequent precision bug in home-grown training code.

---

## 4. How the three interact — a worked failure case

Consider a 70-layer model with default PyTorch init (`kaiming_uniform_`), LayerNorm in post-norm, no warmup, `β₂=0.999`, cosine schedule sized for 2T tokens but training will stop at 800B.

Failure sequence:

1. **Step 0–10.** Residual-stream variance grows ~linearly in depth (post-norm without residual scale) → logits are O(√L). Initial loss is far above `ln(vocab_size)`.
2. **Step 10–50.** AdamW's `v̂` is poorly estimated (`β₂=0.999`, no warmup). Effective LR is 5–10× the peak_lr setting. Gradients spike.
3. **Step 50.** A norm reduction accidentally runs in bf16 (a missing `.float()` cast) → 0.1% bias in normalisation → small extra drift per block → compounded over 70 layers.
4. **Step ~150.** Either loss NaNs from a grad spike or plateaus; clip_grad_norm at 1.0 absorbs some of the damage but the training dynamics are already off-manifold.
5. **Step 800B tokens.** Training stops cleanly, but cosine was sized for 2T, so the LR never really decayed. Final val loss is 1–1.5% worse than a competently-tuned run.

Every step of that failure is preventable with the defaults in this chapter.

---

## 5. Drop-in reference code

```python
# ----- init (GPT-2 / Llama style) -----
def init_weights(module, n_layer):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

def scale_residual_projections(model, n_layer):
    scale = 1.0 / math.sqrt(2 * n_layer)
    for name, p in model.named_parameters():
        # residual projections: attention output W_O and MLP W_2
        if any(k in name for k in ("attn.out_proj", "mlp.down_proj", "w2.weight")):
            p.data.mul_(scale)

# ----- LR schedule (warmup + cosine) -----
def lr_at(step, warmup, total, peak, min_lr_ratio=0.1):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (min_lr_ratio + (1 - min_lr_ratio) * cosine)

# ----- RMSNorm with fp32 reduction -----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps
    def forward(self, x):
        in_dtype = x.dtype
        x_fp32   = x.float()
        rms      = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(in_dtype) * self.weight
```

---

## Connections and what's next

- **[[adam]] / ch-01** — warmup is *because* of AdamW's bias correction; μP LR rules assume AdamW semantics.
- **[[mixed-precision]] / ch-02** — norm reductions in fp32; fp8 runs keep norms in bf16/fp32.
- **ch-04 (packing + masking)** — positional IDs reset per packed sub-sequence; init of RoPE tables follows its own rule.
- **ch-05 (FSDP)** — μP's LR rules combine with FSDP sharding strategies unchanged.
- **[[ppo]] / ch-36** — RL uses a tiny warmup (0–50 steps) and constant LR because the policy is already good; drift is the enemy.

## Further reading

- [[lr-schedules]] — cosine, inverse-sqrt, WSD, the full hyperparameter table.
- [[weight-init]] — Glorot → He → GPT-2 → μP, with the residual-scaling derivation.
- [[batch-vs-layer-norm]] — LayerNorm / RMSNorm / QK-norm / reordered-norm.
- [[olmo-2]] — the public recipe for QK-norm + reordered-norm with documented ablations.

## Companion visualization

**[figures/lr-schedules.html](figures/lr-schedules.html)** — plot linear-warmup + cosine / inverse-sqrt / WSD curves side by side with interactive total-step, warmup-fraction, and decay-phase sliders. Use it to feel why mis-sizing cosine for the wrong horizon costs perplexity.
