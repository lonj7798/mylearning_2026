<!-- chapter: ch-07
     track: foundations
     kind: content
     title: Training Failure Modes: Numerical, Masking, and Capability-Level Failures
     deps: [ch-06]
     sources: [[small-scale-proxies-instabilities]], [[gradient-clipping]], [[sequence-packing]], [[packing-position-id-evidence]], [[hf-gradient-accumulation-fix]], [[tulu-3]], [[llama-3]], [[data-constrained-scaling]], [[catastrophic-forgetting-continual-finetuning]], [[mitigating-alignment-tax-rlhf]], [[sft-memorizes-rl-generalizes]], [[entropy-mechanism-llm-rl]], [[reward-model-overoptimization]], [[reward-hacking-taxonomy]], [[swe-bench-illusion]], [[paloma]]
     figures: figures/failure-modes-tree.html
     revised: 2026-09 (generality revision)
-->

# Chapter 7 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures

> **Core insight.** Training failures fall into three classes that need different instruments. Numerical failures (NaN, Inf, divergence) are visible in the loss and gradient logs and are reproducible at small scale: attention-logit growth and output-logit divergence appear in models from 2.4M to 1.2B parameters at high learning rates, and a quadratic fit over small models predicted a 4.8B divergence that a 4.8B run then confirmed ([[small-scale-proxies-instabilities]], §3.1, §3.3). Masking, packing and loss-aggregation bugs are invisible in the loss curve because they change what the loss is computed over, not whether it is finite; they are caught by equality tests against a reference batching, such as padding versus packing validation loss 1.129 versus 1.127 when position IDs and step counts are kept ([[packing-position-id-evidence]] item 2, Table 2). Capability-level failures — overfitting to repeated data, forgetting, contamination, template narrowing, entropy collapse — are invisible in *both* logs and are only observable in a held-out capability measurement: BLOOMZ-7.1b loses 18.37% of its MMLU score over five continual instruction tasks while every target task improves ([[catastrophic-forgetting-continual-finetuning]], Tables 3–4).
>
> **Guideline.** When a run produces a non-finite or diverging loss, check in the order attention logits → output logits → optimizer state, because each stage has a distinct mitigation with published evidence (qk-layernorm for attention logits, z-loss 1e-4 for output logits, a smaller AdamW ε for a collapsing update; [[small-scale-proxies-instabilities]] §3.1, §3.4). When a run is numerically healthy but underperforms a reference, test batching equivalence before tuning hyperparameters: same data and same global batch must give the same loss under gradient accumulation 1 and 4 ([[tulu-3]] §4.3.2, [[hf-gradient-accumulation-fix]]), and packed must match unpacked when position IDs are reset. When a checkpoint improves the target task, judge it against a held-out capability suite and a previous checkpoint, not against its own training curve, because target-task improvement and general-capability loss occur together ([[catastrophic-forgetting-continual-finetuning]] Tables 3–4; [[mitigating-alignment-tax-rlhf]] §4).

---

## Why this chapter matters for a general-purpose model

Chapters 1 to 6 built a training step that is mechanically correct: an optimizer (ch-01), a precision policy (ch-02), a schedule (ch-03), packing and masking (ch-04), a distributed layout (ch-05), and a checkpoint and evaluation loop (ch-06). This chapter is about the ways that step stops producing a generally capable model, and about which instrument detects each way.

The three classes matter in different places in the pipeline. Numerical failures dominate pre-training, where the run is long and a divergence costs the most wall-clock. Masking, packing and loss-aggregation failures dominate SFT, where sequences are short, heterogeneous and padded, and where the loss is computed over a subset of tokens. Capability-level failures dominate everything after pre-training: mid-training, SFT, preference optimization and RL all optimize a narrow objective on top of a broad model, and all can raise the objective while lowering breadth.

One belief worth correcting at the start: large runs do not fail only by slow drift. In a 54-day snapshot of Llama 3 405B pre-training there were 466 job interruptions, 419 of them unexpected, and about 78% of the unexpected ones were attributed to confirmed or suspected hardware issues ([[llama-3]] §3.3.4, Table 5). Those failures are abrupt and loud. The failures in this chapter are the complementary set: the ones that leave the job running. The operational handling of crashes, stragglers and distributed hangs belongs to the infrastructure material and to the separate training-memory course; this chapter keeps the failures whose detection is a *measurement* question.

An interactive version of the diagnostic order below is in [figures/failure-modes-tree.html](figures/failure-modes-tree.html): select a symptom and the page shows the checks in cost order, the mitigation, and the source locus for every number it displays.

---

## §1 Numerical failures, in diagnostic order

A non-finite value is not informative on its own; the diagnostic question is which stage produced a value the next stage could not represent. Three stages are worth separating, and each has its own evidence.

### 1.1 Attention-logit growth

**Definition.** The attention logit is `z_ij = ⟨q_i, k_j⟩ / √d_h`, where `q_i` is the query vector at position `i`, `k_j` the key vector at position `j`, and `d_h` the head dimension ([[small-scale-proxies-instabilities]] §3.1.1). Attention-logit growth is the case where `max_ij z_ij` rises during training instead of staying bounded.

**Why it matters as a measurable problem.** `softmax` over a row of logits with a large maximum saturates: the row becomes one-hot, the gradient through that row goes to zero, and in fp16 `exp` of a large value overflows. The paper reports the boundary empirically: every run whose max attention logit exceeded about `1e4` diverged (§3.3, Fig. 9). When the max attention logit of a 10M model is *forced* to a value `κ`, loss deteriorates around `κ = 1e3`, and at `κ = 1e4` it is worse than a zero-layer bigram baseline (§3.3, Fig. 10). So the logit value is not a symptom of divergence; it is sufficient to cause it.

**Mechanism of the growth.** The growth comes from larger query and key norms, not from higher cosine similarity between them (Fig. E.1), and it also occurs in a pointwise attention variant with no softmax (§3.2.5, Fig. E.11).

**Mitigation and evidence.** qk-layernorm applies LayerNorm to queries and keys before the logits are computed. With qk-layernorm and z-loss, models from 2.4M to 1.2B non-embedding parameters trained to low loss across peak learning rates 3e-4 to 3e-1, including a 1.2B model at LR 0.3 (§3.1.1, Fig. 1). Per-head qk-layernorm performed better than qk-layernorm over the whole model dimension (Fig. E.8). **Result (single study).**

**Prediction, not only diagnosis.** Fitting the max attention logit at step 2e3 as a quadratic in model size, per learning rate, predicted that the next scale at LR 1e-2 would cross `1e4`; a 4.8B model trained at LR 1e-2 then diverged, and the fit extrapolated its max attention logit closely (§3.3, Fig. 9). The chapter starts with this failure because it is the one of the three that a series of small runs can predict before the large run is launched.

### 1.2 Output-logit divergence

For output logits `y`, `p_i = e^{y_i} / Z` with `Z = Σ_j e^{y_j}`. Output-logit divergence is the case where the logits drift far from the log-probabilities — in the paper's runs, becoming very negative late in training (§3.1.2, Fig. 4). The mitigation is **z-loss**, an auxiliary term `log²Z` with coefficient `1e-4`. The divergence occurred in models with no weight decay at every scale tested; z-loss resolved it, and weight decay also mitigated it for the larger models tested (§3.1.2, Fig. 3). **Result (single study).**

### 1.3 The optimizer-state stage: AdamW ε

The AdamW update before the learning rate is `Δ = v / (√u + ε)`, where `v` and `u` are exponential moving averages of the first and second gradient moments and `ε` is the numerical floor. As model size and learning rate grow, the gradient RMS of the first MLP layer falls, and at the largest scale and LR tested it is of the same order as the default `ε = 1e-8` (§3.4, Figs. 11, 13). When that happens `Δ` shrinks toward zero: the run does not NaN, it stops moving. For a 4.8B model at LR 0.3, `ε = 1e-15` improved loss and removed the collapse in gradient RMS, while `ε = 1e-6` diverged (§3.4, Fig. 12, Fig. E.15).

The practical consequence is a correction to a common habit. Raising `ε` is sometimes described as a NaN fix; in this study the measured failure is the opposite direction, where `ε` is too *large* relative to the gradient scale and the update collapses. Diagnose by logging gradient RMS per layer alongside `ε`, not by changing `ε` and watching the loss.

### 1.4 Logarithms of zero

Cross-entropy computed as `torch.log(torch.softmax(x))` can take the log of an underflowed zero; the fused `log_softmax` and `F.cross_entropy` paths do not, because they subtract the row maximum internally. When cross-entropy is written by hand, use `F.cross_entropy` or `log_softmax` rather than composing `torch.log` with `torch.softmax`. This is a coding rule, not an empirical result, and no number is attached to it here.

Two related claims are worth stating precisely, because loose versions of them circulate:

1. `torch.softmax` already implements the max-subtracted form, so an attention implementation that calls it does not need a separate `amax` subtraction. The hand-rolled hazard is real only for code that exponentiates raw logits itself.
2. The k3 KL estimator used in RL is computed on the **sampled token's** log-probabilities, which come from `log_softmax` and are finite. A sampled token cannot have had probability exactly zero under the policy that sampled it. The real hazard in that estimator is the exponential of a large log-ratio overflowing in low precision, which is a range problem, not a `log(0)` problem.

### 1.5 Zero variance in advantage normalization: a worked example

Group-normalized RL divides by a standard deviation, `A ← (A − mean) / (std + ε)`. Take a group of 8 rollouts on one prompt with binary rewards.

*Mixed group*, rewards `[1, 1, 0, 0, 0, 0, 0, 0]`: mean `= 0.25`; variance `= 0.25 − 0.25² = 0.1875`; `std = 0.4330`. With `ε = 1e-6`, correct rollouts get `A = (1 − 0.25)/0.4330 = +1.732` and incorrect ones `A = (0 − 0.25)/0.4330 = −0.577`. Both signs are present, and the group contributes gradient.

*Uniform group*, rewards `[1, 1, 1, 1, 1, 1, 1, 1]`: mean `= 1`; `std = 0`. Every numerator is exactly `1 − 1 = 0`, so `A = 0 / (0 + 1e-6) = 0` for every rollout. The result is **zero advantage, not NaN**, for any `ε > 0`. The group is not a numerical failure; it is a group that carries no learning signal at all, neither positive nor negative. A NaN appears only if `ε = 0`, or if the arithmetic underflows in fp16.

This distinction matters because the mitigations differ. A NaN wants a numerical guard. A dead group wants a *sampling* change: the entropy study drops prompts whose responses are all correct or all incorrect before the update ([[entropy-mechanism-llm-rl]] §2.2, Recipe ledger). Log the fraction of groups with `std = 0` per step; a rising fraction means the prompt set has become too easy or too hard for the current policy, which is a curriculum signal, not an arithmetic one.

---

## §2 Masking, packing, and loss-aggregation correctness

These bugs share one property: the loss stays finite and smooth, so no numerical alarm fires. They are detected by equality tests against a reference computation.

### 2.1 The label shift, and the off-by-one that actually happens

Causal-LM loss pairs logit position `i` with label index `i + 1`. Take `input_ids = [p0, p1, p2, r0, r1, r2]` with `prompt_len = 3`, where `p*` are prompt tokens and `r*` response tokens. The shifted pairs are:

| logit position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| target label index | 1 | 2 | 3 | 4 | 5 |
| target token | `p1` | `p2` | `r0` | `r1` | `r2` |

The correct masking applies `-100` to the **unshifted** labels:

```python
labels = input_ids.clone()
labels[:, :prompt_len] = -100          # masks label indices 0, 1, 2 -> p0, p1, p2
loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V),
                       labels[:, 1:].reshape(-1),
                       ignore_index=-100)
```

This drops the pairs whose *target* is a prompt token, leaving three trained pairs: `p2 → r0`, `r0 → r1`, `r1 → r2`. Predicting the first response token from the last prompt token is part of correct training, not a leak: it is the transition the model must make at inference.

The bug that occurs in practice is masking **after** the shift:

```python
labels_shift = labels[:, 1:].clone()      # [p1, p2, r0, r1, r2]
labels_shift[:, :prompt_len] = -100       # masks p1, p2, r0  <- removes the first response token
```

The effect is the reverse of the folk description: the first response token is removed from the loss, so the model is never trained to enter the response from the prompt. On a chat template whose first response token is a role or format marker, this is exactly the token whose omission later shows up as malformed turn starts.

**Detection.** Two checks, both cheap and both exact:

```python
active = (labels[:, 1:] != -100).sum(dim=-1)          # per sample
assert (active == response_len).all()                  # off-by-one shows up as response_len - 1
print(tok.decode(input_ids[0][labels[0] != -100]))     # must be exactly the assistant text
```

A second variant to watch for is `labels[:prompt_len] = -100` written for a 2-D tensor: on a batched tensor that indexes **rows**, so it masks the first `prompt_len` samples entirely and leaves every prompt in the remaining samples in the loss. The `active` assertion above catches this immediately, because the masked rows report zero active tokens.

### 2.2 Packing: what the mask does, and what position IDs actually do

Packing concatenates several samples into one sequence. Correctness requires a block-diagonal attention mask, usually supplied to a variable-length kernel as cumulative sequence lengths (`flash_attn_varlen_func(..., cu_seqlens_q, cu_seqlens_k, max_seqlen)`). Without it, the softmax denominator of a token in sample 2 includes scores against sample 1's tokens, which changes the gradient of both samples ([[sequence-packing]] §3.2.1).

The evidence for each adjustment is separable ([[packing-position-id-evidence]] item 1, Krell et al. §4.2.1, Fig. 4, BERT pre-training):

- Without the **mask** adjustment, loss and accuracy worsen substantially and longer training does not recover them.
- Without the **position** adjustment, loss and accuracy nearly match the baseline, but MLM accuracy stalls at 71.8% against a 72.1% target.

Note the asymmetry, and note the setting: BERT uses learned absolute positional embeddings, so an un-reset position index looks up the wrong embedding vector. The correct risk model for a **RoPE** decoder is different, and this is a correction to a claim that appears in several places, including an earlier version of this chapter:

> RoPE attention scores depend on the relative offset `i − j`. With a correct block-diagonal mask, every attended pair lies inside the same document, and offsets within a document are unchanged whether that document starts at pack position 0 or at pack position `L₁`. Un-reset positions therefore do **not** distort intra-document relative-position math.

What un-reset positions do break, in order of how often it matters:

1. **Boundary derivation.** Implementations that reconstruct `cu_seq_len` *from* `position_ids` — the documented mechanism of `DataCollatorWithFlattening` ([[packing-position-id-evidence]] item 2, §3.3) — cannot find the boundaries at all if positions never reset, so the block-diagonal mask silently becomes a dense causal mask.
2. **Trained-range overflow.** If the pack length exceeds the position range the model was trained on, un-reset positions place tokens at indices the model has never seen. This does not arise when packs are built at or below the trained context length.
3. **Learned absolute position embeddings**, as in the BERT result above.

The measured cost of getting this wrong, on a decoder ([[packing-position-id-evidence]] item 2, Mistral-7B on FLAN_20k, Table 2): padding gives 742 tokens/s at validation loss 1.129; packing **without** position IDs gives 2986 tokens/s at 1.306; offline packing **with** position IDs gives 3010 tokens/s at 1.284; online minibatch packing with position IDs gives 1408 tokens/s at 1.127. The authors attribute the residual gap of offline packing to having far fewer optimizer steps in one epoch, not to attention contamination — minibatch packing keeps the step count and matches the padding loss (§4.1, §4.3).

**Conditions and limits.** A third study compares padding, random packing and greedy packing on LLaMA-3-8B and 70B without stating that attention is reset between packed conversations, and reports greedy-packing averages above padding in all 8 model-dataset settings (Table 3; [[packing-position-id-evidence]] item 3). The size of the cross-contamination effect for decoder SFT is therefore an **open question**; the correctness argument for the mask is not.

**The equality test that settles it for a given implementation:**

```python
out_packed   = model(input_ids=packed, position_ids=pos, cu_seqlens=cu).logits
out_unpacked = torch.cat([model(input_ids=s).logits for s in split_by_cu(packed, cu)], dim=1)
assert (out_packed - out_unpacked).abs().max() < 1e-4
```

### 2.3 Loss aggregation under gradient accumulation and data parallelism

A mean loss over non-padding tokens is not invariant to how a batch is split. Tülu 3 states the two cases directly ([[tulu-3]] §4.3.2, Eqs. 1–2). With two samples having `n₁`, `n₂` non-padding tokens and summed token losses `l₁`, `l₂`:

- one forward pass: `L = (l₁ + l₂) / (n₁ + n₂)` — every **token** weighted equally;
- two accumulated micro-batches: `L = (l₁/n₁ + l₂/n₂) / 2` — every **sample** weighted equally.

*Worked example.* `n₁ = 10`, `l₁ = 20`; `n₂ = 90`, `l₂ = 90`. Token-weighted: `110 / 100 = 1.10`. Sample-weighted: `(2.0 + 1.0) / 2 = 1.50`. The same data and the same global batch give a 36% different loss and a different effective weighting of the short sample. The same effect appears across data-parallel ranks, since cross-device averaging averages per-rank means.

Hugging Face shipped the corresponding fix in `transformers` in October 2024 ([[hf-gradient-accumulation-fix]]): "the correct loss should be computed by the total loss across all batches in a gradient accumulation step divided by the total number of all non padding tokens in those batches. This is not the same as the average of the per-batch loss values." The patch changes the causal-LM default loss to `reduction="sum"` followed by division by `num_items`.

Tülu 3's own response was to train with a **sum loss** and re-tune the learning rate; fine-tuning Llama 3.0 on the Tülu 2 mixture, sum loss with LR 5e-6 performed best, and 2 epochs beat 3 to 7 ([[tulu-3]] §4.3.2, Figs. 5–6). **Replicated** across an official framework fix and an independent model report.

**Detection.** Run 50 steps with `grad_accum = 1` and with `grad_accum = 4` at the same global batch and the same data order. The loss curves must match to floating-point tolerance. If they do not, the aggregation is sample-weighted somewhere.

---

## §3 Divergence, spikes, and plateaus

Divergence, spikes and plateaus are three loss shapes with different causes, and each is separated by a different logged quantity.

**Divergence at high learning rate** is the shape the small-proxy study measures. Its summary statistic is **LR sensitivity**: `E_{η∈[a,b]}[min(ℓ(A(η)), ℓ₀) − ℓ*]`, where `η` is the peak LR of a warmup-plus-cosine schedule, `A(η)` the weights after training with `η`, `ℓ` validation loss, `ℓ₀` loss at initialization, and `ℓ*` the best loss over the sweep range (default `[3e-4, 3e-1]`, [[small-scale-proxies-instabilities]] §2.2). Longer warmup reduced LR sensitivity and loss, most for the larger models, which were not stable at LR 3e-1 without long warmup (§3.2.1, Fig. 5). Independent weight decay at `λ = 1e-4` gave lower LR sensitivity than the coupled PyTorch/Optax form at `λ = 0.1` (§3.2.2, Fig. 6). The metric has documented limits: it does not account for a shift in the optimal LR, and it is invariant to loss scale, so a model at random performance for every LR scores 0 (App. B).

**Spikes** — a jump of one or a few steps — are a different phenomenon, and this course's verified sources are thinner here than folklore suggests. Two facts to hold onto:

- The small-proxy study explicitly scopes itself to instabilities that cause slow divergence, **not** fast loss spikes (footnote 1, §4).
- Llama 3 405B, at 15.6T tokens, reports the opposite of a spike-intervention stack: "We found this training recipe to be very stable: we observed few loss spikes and did not require interventions to correct for model training divergence" ([[llama-3]] §3.4.1). Its stated stability measure is a batch-size ramp — 4M tokens at sequence 4,096, doubling to 8M at sequence 8,192 after 252M tokens, doubling again to 16M after 2.87T tokens — together with peak LR 8e-5 and 8,000 warmup steps.

The one spike-adjacent ablation with a number in the library is an optimizer setting: in the data-constrained study, `β₂ = 0.95` gave slightly lower final loss and fewer loss spikes than `0.999` at two of the FLOP budgets ([[data-constrained-scaling]] App. S). Treat skip-step heuristics as engineering practice without a controlled ablation in these sources, and say so when you use them.

**Plateaus** have three causes that the same three logged scalars separate:

| Observed | `lr` | `clipped_fraction` | active tokens per batch | Cause |
|---|---|---|---|---|
| flat loss | 0 | any | normal | schedule exhausted or resumed at the wrong step (ch-06) |
| flat loss | normal | ≈ 1.0 | normal | clip threshold below the natural gradient scale |
| flat loss | normal | ≈ 0 | ≈ 0 | masking or data bug: nothing is being trained on |
| flat loss | normal | ≈ 0 | normal | update collapse: check gradient RMS against AdamW `ε` (§1.3) |

Global-norm clipping rescales the whole gradient by `threshold / ‖g‖` when `‖g‖ > threshold`, which preserves direction and bounds step size ([[gradient-clipping]]). A threshold far below the run's usual gradient norm converts every step into a fixed-length step in the gradient direction; the quantity that identifies it is `clipped_fraction` pinned at 1.0, not the loss itself. Both verified pre-training recipes in this chapter's sources use global-norm 1.0 ([[small-scale-proxies-instabilities]] §2.1; [[data-constrained-scaling]] App. S).

---

## §4 Capability-level failures

These are the failures that motivated the revision of this chapter. None of them produces an abnormal loss curve. All of them reduce breadth while the training objective improves.

### 4.1 Overfitting from repeated data

**The measurable problem.** Repeating data raises training-set fit faster than held-out fit, so a train-loss-only view reports improvement while the model gets worse.

**Evidence at pre-training scale.** In the data-constrained study (GPT-2 architecture, C4 subsets, more than 400 runs up to 9B parameters and 900B tokens), training on data repeated up to **4 epochs** changes held-out loss almost not at all — an 8.7B model at 4 epochs ends 0.5% higher in validation loss than at 1 epoch — and downstream performance on 19 tasks starts dropping after about 4 epochs ([[data-constrained-scaling]] Abstract, §6, App. L, Fig. 6). The fitted effective-data formula is `D′ = U_D + U_D · R*_D · (1 − e^{−R_D/R*_D})` with `R*_D = 15.387756`, where `U_D` is the unique-token budget and `R_D = D/U_D − 1` is the number of repetitions.

*Worked example.* 4 epochs means `R_D = 3`, so `D′ = U_D(1 + 15.39 × (1 − e^{−3/15.39})) = 3.73 U_D` against `D = 4 U_D`: repeated tokens are worth 93% of unique ones. At 16 epochs (`R_D = 15`), `D′ = 10.58 U_D`, or 66%. The study also notes it uses held-out test loss rather than training loss precisely because models overfit repeated data (App. H, Fig. 14).

**Evidence at SFT scale.** Fine-tuning Llama 3.0 on the Tülu 2 mixture, 2 epochs beat 3 through 7 on the evaluation average ([[tulu-3]] §4.3.2, Fig. 6). **Result (single study)** in each case, but the direction agrees across scales.

**Detector.** Held-out loss on data from the same distribution, plus a broad benchmark average. Training loss alone cannot see this.

### 4.2 Catastrophic forgetting during continual fine-tuning

**Definition and metric.** Forgetting is the loss of previously acquired ability while a new task is learned. The forgetting metric is an average *relative* decrease: `FG_i = (1/|E_i|) Σ_e (1/N) Σ_m (R_e^0 − R_e^m)/R_e^0 × 100%`, where `E_i` is one evaluation set, `R_e^m` the score after `m` continually trained tasks, and `R_e^0` the initial model's score ([[catastrophic-forgetting-continual-finetuning]] Eq. 1).

**Evidence.** Continually instruction-tuning BLOOMZ on five tasks in a fixed order, with MMLU, commonsense reasoning and RACE as the general suite (Table 4):

| Model | Domain knowledge FG | Reasoning FG | Reading comprehension FG |
|---|---|---|---|
| BLOOMZ-1.1b | 9.54% | 6.73% | 18.04% |
| BLOOMZ-1.7b | 10.72% | 6.48% | 24.29% |
| BLOOMZ-3b | 14.63% | 11.09% | 27.56% |
| BLOOMZ-7.1b | 18.37% | 13.62% | 26.75% |

Over the 1.1b–7.1b range tested, forgetting increases with scale; the authors attribute this partly to the larger model's higher starting scores. Meanwhile the target tasks improve (BLOOMZ-7.1b on the explanation task, 51.47 → 68.71, Table 3). Both facts come from the same runs, which is what makes the pairing informative: target-task improvement is not evidence of retained breadth.

One mitigation appears in the same paper: LLAMA-7b shows FG 34.57 / 31.33 / 31.72 on the three sets, while ALPACA-7b — the same base model after general instruction tuning — shows 18.14 / 7.56 / 10.31 (Table 6). **Result (single study).**

### 4.3 The alignment tax

The RLHF version of the same failure has its own name. On OpenLLaMA-3B instruction-tuned on ShareGPT and then aligned with rejection-sampling fine-tuning, PPO or DPO, the reward rises while WMT14 French→English BLEU and SQuAD/DROP F1 fall continuously, and commonsense QA rises before falling ([[mitigating-alignment-tax-rlhf]] §4, App. E.1, Fig. 12).

Every mitigation tested reduced the tax *and* the reward: early stopping, L1/L2 regularization toward the pre-RLHF weights, LoRA, and distillation from the pre-RLHF policy (§4.1, Fig. 3). The method with the best alignment-forgetting Pareto front was weight interpolation, `π_{(1−α)θ₀ + αθ}` for `α ∈ [0, 1]`, and Heterogeneous Model Averaging — a separate ratio per block, default `K = 3` — pushed that front further (§6, Fig. 5). The reported curve degrades as `K` grows to 6 and 9, which the authors attribute to overfitting the in-domain reward.

The transferable lesson is the axis choice: the tax is only visible when a general-capability metric is plotted **against** the alignment metric, one point per checkpoint or per `α`. Plotted against training step, each curve looks like progress.

### 4.4 Contamination inflating apparent generality

A benchmark score can rise because the model is better or because the benchmark is in the training data. The cleanest recent demonstration: given only a repository name and the issue text — no repository contents — ten OpenAI and Anthropic models named a file changed by the gold patch in 60–76% of SWE-Bench Verified instances, but in under 53% of 245 SWE-Bench-style tasks from seven repositories not in SWE-Bench. Function-reproduction 5-gram overlap reached a maximum of 34.9% on SWE-Bench Verified against 13.9% on the outside repositories and 18.1–18.2% on RefactorBench and SWE-Bench Extra — the last of which still draws on SWE-Bench repositories, so it is not a clean held-out set ([[swe-bench-illusion]] §4.1, §4.3). **Result (single study).**

The detector generalizes: hold out a slice the benchmark builders could not have published — a different repository set, issues created after the benchmark was built, a re-templated variant — and compare. ch-14 covers pretraining-side decontamination; ch-47 covers suite design.

### 4.5 Template and format narrowing

Training on a narrow input format teaches the format as well as the task. A measured instance: packing a single-turn-only SFT set (a filtered 200K subset of OpenHermes 2.5) produced a significant MATH drop that returned to normal after adding one multi-turn conversation per 20 to 40 samples; a different internal 200K single-turn set showed no drop ([[packing-position-id-evidence]] item 3, §5.3, Fig. 3). The effect is data-dependent, which is why the detector has to be an evaluation and not a rule: evaluate the same capability under at least two prompt formats, including one that does not appear in training.

---

## §5 RL failures as loss of output diversity

RL on a verifiable or learned reward can raise the training objective while narrowing what the model can produce. Two mechanisms have quantitative evidence.

### 5.1 Entropy collapse

**Definition.** Policy entropy is `H(π_θ, D) = −E[log π_θ(y_t | y_<t)]`, averaged over the tokens of responses to training prompts ([[entropy-mechanism-llm-rl]] §2.1, Eq. 5).

**Measured law.** Across 11 base models from 4 families (0.5B–32B), math and code tasks and 4 RL algorithms, validation accuracy `R` and entropy `H` fit `R = −a·exp(H) + b` (§2.2–2.4). Two consequences are reported directly: at `H = 0` the accuracy is bounded by `−a + b`, and **73% of entropy consumption and 76% of the performance gain occur in the first 200 of 2400 gradient steps** (§2.3–2.4). Fitting `a` and `b` on the first 36 steps predicted the next 200 with RMSE 0.9% on math and 1.2% on code (§2.4, Fig. 5).

**Mechanism.** For a softmax policy, the step-wise entropy change is approximately the negative covariance between an action's log-probability and its logit change; under a natural-policy-gradient update this reduces to `ΔH ≈ −η·Cov(log π(a|s), A(s,a))` (§3.2, Theorems 1–2). The covariance is concentrated: at training step 1 on Qwen2.5-7B, the mean covariance of the top 0.02% of tokens is 5.654 against 0.003 over all tokens (Table 1).

**What does not work, with numbers.** Entropy-loss coefficients of 0.0001 and 0.001 had minor influence, 0.01 caused entropy explosion, and 0.005 stabilized entropy without outperforming the baselines (§4.1, Fig. 9). Reference-KL coefficients from 0.001 to 0.1 stabilized entropy but lowered accuracy (§4.1, Fig. 10). **What did work in this study:** restricting the update on the `1e-4` to `1e-3` fraction of tokens with the largest centered log-probability × advantage product. On Qwen2.5-32B, the 7-benchmark math average went from GRPO 45.8 to Clip-Cov 50.3 and KL-Cov 52.2 (Table 2). **Result (single study).**

Note what this chapter does **not** claim: there is no published entropy threshold such as "below 0.1 nats means collapse". The paper reports a monotone decline toward zero and a fitted relation, not a cut-off. Use the trajectory and the fit, not a threshold.

### 5.2 Reward over-optimization

When the reward is a learned model, optimizing it past a point lowers the true objective. In a synthetic setup where a 6B "gold" reward model labels the data used to train 3M–3B proxy reward models and a 1.2B policy is optimized against the proxy, the gold score rises and then falls as `d = √KL(π‖π_init)` grows, fitting `R_bon(d) = d(α − β·d)` for best-of-n and `R_RL(d) = d(α − β·log d)` for RL ([[reward-model-overoptimization]] §1, §3.2, Fig. 1). Differentiating gives the peaks `d* = α_bon/(2β_bon)` and `d* = exp(α_RL/β_RL − 1)`.

Two results from the same study are worth carrying: RL spends far more KL than best-of-n for the same amount of optimization, so KL is not a common currency for comparing methods (§3.5); and a nonzero KL penalty behaved like early stopping and did not raise the gold-score-versus-KL frontier, a result the authors themselves call hyperparameter-sensitive (§3.6, Fig. 9).

The structural reason not to look for a better proxy instead: over the class of all stochastic policies, two reward functions are "unhackable" — proxy improvements never lower the true reward — only in degenerate cases, essentially when one is a positive affine transform of the other or one is constant ([[reward-hacking-taxonomy]]). The available controls are therefore structural: restrict the policy class, bound the optimization, or ground the reward in a verifier (ch-44).

### 5.3 Which detector separates memorization from generalization

Held-out data from the training distribution does not distinguish a model that learned the rule from one that memorized the training instances, because both score well on it. Changing the **rule** or the **visual attribute** while holding the task fixed does. On GeneralPoints and V-IRL with a Llama-3.2-Vision-11B backbone, RL raised OOD success rate on all four variants (for example V-IRL-L 80.8% → 91.8%) while SFT lowered it on all four (V-IRL-L 80.8% → 1.3%; GP-L 11.5% → 3.4%) ([[sft-memorizes-rl-generalizes]] §5.1). On visual variation, RL gained +17.6 and +61.1 points where SFT lost 9.9 and 5.6 (§5.2).

Two conditions the same paper attaches, which keep this from becoming "RL instead of SFT": the base model could not follow the task instructions, so SFT was required before RL to make RL train at all (§5.4), and RL started from an extremely overfitted SFT checkpoint stayed below 1% success (App. D.3).

---

## Negative samples and negative feedback

This chapter uses negatives in sense (4) of the standard taxonomy — **negative as gradient**, an explicit decrease of a sample's likelihood — because that is the form in which RL failures appear here. Senses (1) to (3) belong to the data and SFT chapters.

**Where negatives come from at this stage.** A verifier or reward model labels each rollout; the group-normalized advantage turns those labels into per-sample signs. Label quality is where this stage fails: a reward model is a proxy whose over-optimization is measurable (§5.2), and a verifier can be gamed by output-format tricks.

**The failure mode specific to diagnostics.** A group with zero reward variance produces `A = 0` for every member (§1.5). Such a group is often described as a division-by-zero hazard; the more consequential fact is that it supplies **neither** positive nor negative gradient. A rising fraction of zero-variance groups is therefore a silent reduction in effective batch size, and the fix is prompt selection, as in the entropy study's filter that drops prompts whose responses are all correct or all incorrect ([[entropy-mechanism-llm-rl]] §2.2).

**Mechanism, for the sign asymmetry.** For a softmax head, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`: pushing down an already-unlikely sampled token moves its mass onto the currently most likely alternative, which sharpens the distribution. Read with the entropy study's Eq. 10, a sampled token with below-mean log-probability and below-mean advantage has a **positive** centered product and therefore lowers entropy, while a negative advantage on an above-mean-probability token raises it (§3.2 and Eq. 10, with the caveat that the paper does not report results split by advantage sign — that split is an **open question**).

**Controls with evidence.** Restricting the update on the highest-covariance `1e-4`–`1e-3` of tokens raised both entropy and accuracy where a flat entropy bonus did not (§4.1, §4.3, Table 2). Bounding total optimization by tracking a gold or held-out score against `√KL` and stopping near its peak is the reward-model analogue ([[reward-model-overoptimization]] §3.2, Fig. 8).

**Diagnostics to log.** Fraction of zero-variance groups; entropy per update; the advantage distribution split by sign; pass@1 together with pass@k at large k — noting that the entropy study does not report pass@k, so the diversity cost of its interventions is not measured there.

---

## Recipe

Values relevant to preventing or detecting the failures above. Every row is quoted from its locus; no row is transferred across model sizes or stages.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Small-scale-proxies default Transformer (NanoDO, C4) | 2.4M–1.2B | pretrain-stable | gradient clipping | global norm 1 | arXiv:2309.14322v2 §2.1 | verified 2026-09-14 (card) | no ablation reported |
| same | 2.4M–1.2B | pretrain-stable | z-loss coefficient on `log²Z` | 1e-4 | §2.1, §3.1.2 | verified 2026-09-14 (card) | Figs. 3–4: without z-loss and without weight decay the output logits diverge |
| same | 2.4M–1.2B | pretrain-stable | qk-layernorm | on, per head with shared parameters | §2.1, §3.2.5 | verified 2026-09-14 (card) | Figs. 1–2; Fig. E.8 (per head better than whole dimension) |
| same | 2.4M–1.2B | pretrain-stable | warmup; total steps | 5e3 linear; 1e5 | §2.1 | verified 2026-09-14 (card) | Fig. 5: longer warmup lowers LR sensitivity and loss |
| same | 2.4M–1.2B | pretrain-stable | weight decay | independent, 1e-4 (0.1 when coupled) | §2.1, §3.2.2 | verified 2026-09-14 (card) | Fig. 6: independent form gives lower LR sensitivity |
| Small-scale-proxies 4.8B intervention run at LR 0.3 | 4.8B | pretrain-stable | AdamW ε | 1e-15 | §3.4, Fig. 12 | verified 2026-09-14 (card) | lower loss than 1e-8; 1e-6 diverged (Figs. 12, E.15); one run per value |
| datablations GPT-2-architecture study models | ≤ 9B | pretrain-stable | Adam β₂ | 0.95 at FLOP budgets 9.3e20 and 2.1e21; 0.999 otherwise | arXiv:2305.16264v5 App. S | verified 2026-09-14 (card) | slightly lower final loss and fewer loss spikes than 0.999 (App. S, no table) |
| datablations GPT-2-architecture study models | ≤ 9B | pretrain-stable | epochs with negligible loss penalty | up to 4 (`R_D = 3`) | §6, App. L | verified 2026-09-14 (card) | 8.7B at 4 epochs: +0.5% validation loss vs 1 epoch; 19-task average drops after ~4 |
| Llama 3 405B | 405B | pretrain-stable | peak LR; warmup; batch ramp | 8e-5; 8,000 steps; 4M tokens @ seq 4,096 → 8M @ 8,192 after 252M tokens → 16M after 2.87T | arXiv:2407.21783 §3.4.1 | verified 2026-09-17 | the report attributes stability to the ramp: "few loss spikes … no interventions" (§3.4.1); no ablation printed |
| Tülu 3 SFT study run (Llama 3.0 base, Tülu 2 SFT mixture) | not stated in §4.3.2 | SFT | loss aggregation; LR; epochs | sum loss; 5e-6; 2 | arXiv:2411.15124v5 §4.3.2, Figs. 5–6 | verified 2026-09-17 | Fig. 5: sum loss at 5e-6 best of the sweep; Fig. 6: 2 epochs beat 3–7 |
| Mistral-7B on FLAN_20k (Kundu et al.) | 7B | SFT | batching for correctness and throughput | online minibatch packing with per-example position IDs | arXiv:2407.09105v6 §4.1, Table 2 | verified 2026-09-17 | validation loss 1.127 vs padding 1.129; packing without position IDs 1.306; offline packing with position IDs 1.284 |
| Entropy-mechanism §2 runs | 0.5B–32B | RL | prompt filter | drop prompts whose responses are all correct or all incorrect | arXiv:2505.22617v1 §2.2 | verified 2026-09-14 (card) | no ablation reported |
| Qwen2.5-32B (entropy-mechanism §4.3) | 32B | RL | KL-Cov `k`; `β` | 2e-4; 1 | §4.3 | verified 2026-09-14 (card) | 7-benchmark math average 45.8 (GRPO) → 52.2 (Table 2) |
| Gao et al. proxy-RM study policy | 1.2B | RL | KL penalty coefficient | 0 in all RL experiments except §3.6 | arXiv:2210.10760 §2, App. C | verified 2026-09-14 (card) | §3.6, Fig. 9: a nonzero penalty behaved like early stopping and did not raise the frontier |

**Starting point for a small general-purpose run.** For a dense decoder under about 1B parameters trained on web text, the verified rows above support: global-norm clipping at 1.0, z-loss coefficient 1e-4, per-head qk-layernorm, linear warmup of 5e3 steps within a 1e5-step cosine schedule, and independent weight decay 1e-4 — all measured on 2.4M–1.2B models on C4 at sequence length 512 and batch 256 sequences ([[small-scale-proxies-instabilities]] §2.1). For an SFT stage on top, sum-loss aggregation with LR 5e-6 and 2 epochs is the setting Tülu 3 selected by sweeping a Llama 3.0 base model on the Tülu 2 SFT mixture ([[tulu-3]] §4.3.2, Figs. 5–6); the model size for that sweep is not printed at the locus, so re-run the LR sweep at your own size, since sum loss changes the gradient scale. Do not carry the 4.8B `ε = 1e-15` row into a smaller run: it was selected at LR 0.3 on one model.

---

## Generalization lens

**(a) What increases breadth.** Keeping the number of epochs over unique data at or below about 4 at pre-training scale, where held-out loss is within 0.5% of the single-epoch value and the 19-task average is unchanged ([[data-constrained-scaling]] §6, Fig. 6), and 2 epochs at the SFT scale measured by Tülu 3 (§4.3.2, Fig. 6). Performing general instruction tuning before task-specific continual tuning: ALPACA-7b forgets 18.14% / 7.56% / 10.31% where LLAMA-7b forgets 34.57% / 31.33% / 31.72% ([[catastrophic-forgetting-continual-finetuning]] Table 6). Interpolating post-alignment weights with pre-alignment weights, which gave the best measured alignment-forgetting Pareto front ([[mitigating-alignment-tax-rlhf]] §4.1). Keeping RL entropy from collapsing by restricting the highest-covariance tokens rather than by a flat entropy bonus ([[entropy-mechanism-llm-rl]] §4.3).

**(b) What causes narrowing or forgetting.** Repetition past the measured knee (19-task average drops after about 4 epochs). Continual fine-tuning on a task sequence, which costs 9.54%–18.37% of MMLU across the BLOOMZ sizes tested while every target task improves. Alignment optimization, which lowers translation and reading comprehension monotonically as reward rises. Reward over-optimization, where the gold score peaks and falls while the proxy score keeps rising ([[reward-model-overoptimization]] Fig. 1). Entropy consumption, where 73% of the entropy is spent in the first 200 of 2400 steps and the accuracy ceiling at `H = 0` is `−a + b`. Single-format training data, as in the single-turn packing result that moved MATH until multi-turn data was reintroduced.

**(c) How to measure it for this stage.** Four instruments, each detecting something the others miss:

1. **Held-out multi-domain perplexity.** One held-out loss hides domain gaps: among 6 controlled 1B baselines, the C4-only model reaches perplexity up to 391,171 on the RedPajama arXiv domain, and no single perplexity source correlates with all 8 downstream tasks tested ([[paloma]] §4.1–4.2). Report macro-averaged per-domain perplexity on decontaminated data with a fixed vocabulary.
2. **A broad benchmark average against a previous checkpoint**, not against the training curve; this is the axis that exposes forgetting and the alignment tax.
3. **An out-of-distribution variant of the target task** — changed rule, changed surface attribute, changed template — which is what separates rule learning from memorization ([[sft-memorizes-rl-generalizes]] §5.1–5.2).
4. **A contamination probe**: the same capability measured on material the benchmark builders could not have published ([[swe-bench-illusion]] §4.1).

Mapped to the failures above, the regression suite is one row per failure, and a run should not be promoted until each row has been read against the previous checkpoint:

| Failure | Instrument that exposes it | Comparison |
|---|---|---|
| Repeated-data overfitting | held-out loss on the same distribution; 19-task-style broad average | current vs 1-epoch or 2-epoch run ([[data-constrained-scaling]] §6) |
| Forgetting after continual tuning | fixed general suite (MMLU, commonsense, reading comprehension), scored as FG | current vs pre-tuning checkpoint ([[catastrophic-forgetting-continual-finetuning]] Eq. 1) |
| Alignment tax | general-capability metric plotted against the alignment metric | one point per checkpoint or per interpolation ratio ([[mitigating-alignment-tax-rlhf]] §4) |
| Contamination | matched tasks from unpublished material | public benchmark vs matched slice ([[swe-bench-illusion]] §4.1) |
| Template narrowing | same capability under a second, untrained prompt format | format A vs format B, same checkpoint |
| Memorization instead of rule learning | rule- or attribute-changed variant of the task | in-distribution vs variant ([[sft-memorizes-rl-generalizes]] §5.1) |
| Entropy collapse and reward over-optimization | entropy trajectory with the fitted `R = −a·exp(H) + b`; held-out or gold score against `√KL` | across updates, not at the end only ([[entropy-mechanism-llm-rl]] §2.4; [[reward-model-overoptimization]] §3.2) |
| Domain-specific loss regression hidden by one average | macro-averaged per-domain perplexity | current vs previous checkpoint ([[paloma]] §4.1) |

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Masking applied after the label shift | The first response token is never trained; malformed turn starts at inference | `(labels[:, 1:] != -100).sum(-1) == response_len` per sample |
| `labels[:prompt_len] = -100` on a batched (2-D) tensor | Whole samples silently excluded; prompts trained on in the rest | Active-token count is 0 for the first `prompt_len` rows |
| Packing without `cu_seqlens`, or a kernel that ignores it | Loss slightly worse than unpacked; no error | Logit equality test, packed vs per-sample forward, tolerance 1e-4 |
| Deriving boundaries from `position_ids` that never reset | The varlen kernel silently falls back to dense causal attention | Assert `position_ids.min() == 0` per packed row and that the recovered `cu_seq_len` count equals the sample count |
| Mean loss under gradient accumulation | Loss and results change when `grad_accum` changes; short samples over-weighted | Same data, `grad_accum` 1 vs 4, losses must match ([[tulu-3]] §4.3.2) |
| Raising AdamW `ε` as a NaN remedy | The update shrinks and the loss plateaus | Log per-layer gradient RMS and compare it with `ε` ([[small-scale-proxies-instabilities]] §3.4) |
| Treating a zero-variance RL group as a numerical bug | Fraction of groups with `std = 0` rises; reward curve flattens | The advantage is exactly 0, not NaN; filter such prompts instead of clamping |
| Reading a rising proxy reward as progress | Gold or human score peaks then falls while proxy rises | Plot the held-out score against `√KL` and stop near the peak ([[reward-model-overoptimization]] §3.2) |
| Judging a fine-tune by its target task | Target task improves; MMLU, RACE, translation fall | Report FG against the pre-tuning checkpoint on a fixed general suite |
| Reading a benchmark gain as capability | Score high on the public set, low on a matched unpublished set | Held-out-repository or post-cutoff slice ([[swe-bench-illusion]] §4.1) |

---

## Check your understanding

1. A group of 8 rollouts all receive reward 1 and the run uses `(A − mean)/(std + 1e-6)`. Explain why the result is exactly 0 rather than NaN, and why the correct response is a change to prompt selection rather than a numerical guard.
2. A colleague reports that not resetting `position_ids` under packing "breaks RoPE's relative positions". State why that is wrong for intra-document attention with a correct block-diagonal mask, and give two situations in which un-reset positions still break the run.
3. Two runs use the same data, the same global batch and the same seed, but `grad_accum = 1` and `grad_accum = 4`, and their losses differ by 30%. Derive which quantity is being weighted differently and what fix restores equality.
4. A 4.8B run at a high learning rate plateaus with healthy `lr`, `clipped_fraction` near 0, and a normal active-token count. Which optimizer-level measurement would you take next, and what result would confirm the diagnosis?
5. An SFT checkpoint improves the target benchmark by 6 points and loses 3 points of MMLU. Using the FG definition, explain why "3 points" and "FG = x%" can rank two checkpoints differently, and which you would report.
6. Entropy in an RLVR run falls steadily while accuracy rises and then flattens. Explain, using `R = −a·exp(H) + b` and the covariance mechanism, why a flat entropy bonus of 0.01 was reported to cause entropy explosion while restricting a `1e-4` fraction of tokens raised both entropy and accuracy.
7. You are given two checkpoints with identical scores on a held-out split of the training distribution. Design the single cheapest measurement that would tell you which one learned the rule, and say what result would be evidence of memorization.

---

## Connections

- **Previous:** ch-06 — Checkpointing, In-Loop Evaluation, and Checkpoint Selection. The instrumentation and checkpoint-selection rules there are what make the diagnostics in this chapter available.
- **Next:** ch-08 — Lab: Minimal Trainer with a Target-versus-General Capability Measurement. The lab implements the equality tests and the general-capability comparison used here.
- ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability: the AdamW `ε` and update-scale material behind §1.3.
- ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch: precision choices that decide which numerical failures are reachable.
- ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization: warmup and batch-ramp settings referenced in §3.
- ch-04 — Sequence Packing, Loss Masking, and Chat Templates: the mechanics this chapter tests for correctness.
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination: repetition limits and decontamination in depth.
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate: the SFT-side treatment of §2 and §4.
- ch-43 — Entropy, Output Diversity, and KL Control in RL: the full treatment of §5.1.
- ch-47 — Evaluation Harness and Suite Design for General Capability: how to build the suite the generalization lens assumes.

---

## Sources

- [[small-scale-proxies-instabilities]] — attention-logit growth, output-logit divergence, LR sensitivity, warmup and weight-decay effects, and the AdamW-ε result; the evidence base for §1 and §3.
- [[gradient-clipping]] — definition of global-norm clipping used in §3.
- [[sequence-packing]] — block-diagonal masking and per-sequence position indices for packed training.
- [[packing-position-id-evidence]] — chapter-local extract of the three primary packing studies (Krell 2021, Kundu 2024, Wang 2024) with the loci used in §2.2 and §4.5.
- [[hf-gradient-accumulation-fix]] — chapter-local extract of the October 2024 `transformers` fix; the correctness rule and patch quoted in §2.3.
- [[tulu-3]] — batch-aggregation equations, the sum-loss decision, and the epoch and LR sweeps (§4.3.2).
- [[llama-3]] — 405B pre-training schedule and batch ramp, the loss-spike statement, and the interruption counts used in the opening section.
- [[data-constrained-scaling]] — repetition limits, the effective-data formula, and the β₂ note.
- [[catastrophic-forgetting-continual-finetuning]] — chapter-local extract: the FG metric and the BLOOMZ, LLAMA and ALPACA forgetting tables used in §4.2.
- [[mitigating-alignment-tax-rlhf]] — chapter-local extract: the alignment-tax setup, the mitigation comparison (§4.1), and heterogeneous model averaging (§6).
- [[sft-memorizes-rl-generalizes]] — chapter-local extract: OOD rule and visual variation results used in §5.3 and in the generalization lens.
- [[entropy-mechanism-llm-rl]] — the entropy-performance law, the covariance mechanism, and the intervention results in §5.1.
- [[reward-model-overoptimization]] — gold-versus-proxy curves in `√KL` and the KL-penalty result in §5.2.
- [[reward-hacking-taxonomy]] — the formal statement that non-trivial unhackable proxies do not exist over all stochastic policies.
- [[swe-bench-illusion]] — the contamination probe and its numbers in §4.4.
- [[paloma]] — per-domain perplexity as the held-out instrument in the generalization lens.
