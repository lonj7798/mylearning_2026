<!-- scope: validation tracking, early stopping, SWA, model souping
     deps: [[lr-schedules]]
     see-also: [[adam]], [[dropout]]
-->

# Early Stopping, Checkpointing, SWA, and Model Souping
- **Core Insight:** The training-loss minimum is rarely the generalization minimum; periodic snapshots of model state — and averages of those snapshots — produce a stronger final model than any single training point.
- **Guideline:** Always track validation loss on a held-out set every N steps; for SFT, early-stop on val loss; for pretraining, do **WSD-style late annealing** then average the last K decay-phase checkpoints (poor man's SWA / soup).
- **Authors:** classical (Prechelt 1998 early-stopping); Izmailov et al. 2018 (SWA); Wortsman et al. 2022 (Model Soups)
- **Year:** 1998 / 2018 / 2022
- **URL:** https://arxiv.org/abs/1803.05407 ; https://arxiv.org/abs/2203.05482
- **Relevant topics:** generalization, checkpointing, model averaging, ensembling

## Abstract (composite)
**Early stopping** halts training when validation loss stops improving for `patience` evaluations — the simplest, most reliable regularizer ever proposed; effective whenever overfitting is real (small datasets, fine-tuning).
**SWA / Stochastic Weight Averaging** (Izmailov 2018): instead of using the final SGD weights, average weights from a constant- or cyclic-LR phase. Produces flatter minima with measurably better test accuracy at zero extra training cost.
**Model Soups** (Wortsman 2022): average weights of multiple independently fine-tuned models; "greedy soup" sorts by val acc and adds models only if the average improves. Beats best single model on ImageNet, often by >1% top-1.
**Latent-WSD averaging** (DeepSeek, MiniCPM 2024): the pretraining-scale analog — average checkpoints from the decay phase of a WSD schedule.

## Key Contributions
- **Early stopping**: validation-tracked patience-based halt; equivalent to L2 regularization of a specific implicit prior (Bishop 2006).
- **SWA**: a single-cycle, uniform-weight running average of weights during a constant-LR or cyclic-LR phase.
- **Model Soups**: weight-space ensembling instead of output-space ensembling — zero inference cost, often comparable accuracy.
- **WSD-decay averaging**: the modern analog used in 7B+ LLM pretraining; commits the decay phase budget to producing several near-optimal models that can be soup-averaged.
- **Checkpointing infrastructure** (FSDP, DCP, Megatron): periodic full-state save (model + optimizer + RNG + dataloader iterator) so multi-day runs can survive node failure.

## Key Figures/Tables to Study
- **SWA Figure 1**: loss-surface visualization showing SWA finds a wider basin than SGD endpoint.
- **Model Soups Figure 1**: greedy soup beats every individual fine-tuned model on ImageNet.
- **MiniCPM WSD figure**: late-decay-phase averaging produces lower final loss than any single decay endpoint.
- **Llama-3 paper checkpoint section**: production-grade checkpoint cadence (every ~1000 steps; restart on loss spike).

## Technical Details

**Classical early stopping**:
```
best_val = inf; patience_counter = 0; best_ckpt = None
for step in range(total_steps):
    train_step()
    if step % eval_every == 0:
        val = evaluate(val_set)
        if val < best_val - delta:
            best_val = val
            best_ckpt = save_checkpoint()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
return best_ckpt
```
Defaults: `patience = 3–5` evaluations, `delta = 0` (strict improvement) or small (e.g. 1e-4).

**SWA — Stochastic Weight Averaging**:
```
# After a normal training phase, run a constant high-LR phase
# of length M epochs, snapshotting weights every K steps.
swa_weights = first_snapshot
n = 1
for snapshot in subsequent_snapshots:
    swa_weights = swa_weights + (snapshot - swa_weights) / (n + 1)   # running avg
    n += 1

# After SWA: re-compute BatchNorm statistics with one forward pass over training set
recompute_bn_statistics(swa_weights, train_set)
```
For LLMs (no BN), the BN-recompute step is unnecessary — RMSNorm/LayerNorm have no running statistics.

**Model Soups (greedy variant)**:
```
sort fine-tuned models by val_acc descending
soup = models[0]
for m in models[1:]:
    candidate = average(soup, m)
    if val_acc(candidate) > val_acc(soup):
        soup = candidate
return soup
```
Works because fine-tuned models from a shared init occupy the same loss basin; weight-averaging stays in-basin.

**WSD-decay averaging (LLM pretrain default in 2024–2025)**:
- Run pretraining with WSD schedule (warmup → stable → decay).
- During the decay phase, save N checkpoints (every ~10% of decay tokens).
- Final model = uniform average of the last K checkpoints.
- DeepSeek-V3 and MiniCPM both report measurable val-loss reduction (~0.5%) from this trick alone.

**Checkpointing in production**:
| Item | Saved? | Why |
|---|---|---|
| Model weights (sharded) | Yes | obvious |
| Optimizer state (`m`, `v`, master fp32 weights) | **Yes** | restart needs Adam moments |
| LR scheduler state | Yes | step counter |
| RNG state (per rank) | Yes | deterministic data order |
| Dataloader iterator position | **Yes** | must not re-see same tokens |
| Loss-scaler state (fp16) | Yes | dynamic loss scale |
| Gradient accumulator | Optional | only mid-microbatch |

PyTorch DCP (Distributed Checkpoint) is the standard for FSDP; Megatron has its own format. Cadence: every 30–60 minutes of compute (Llama-3 used ~1000-step cadence at 405B).

**Spike recovery**: if validation loss spikes mid-pretraining, the standard recipe is:
1. Roll back to last clean checkpoint.
2. Skip the offending data shard (reorder).
3. Resume with same LR.
This costs ~1% throughput in expectation but saves runs at frontier scale.

**Common pitfalls**:
- Saving weights but not optimizer state → restart re-warms-up Adam moments → loss-spike on resume.
- Saving but not RNG / dataloader cursor → on resume, re-seeing the same tokens → spurious loss drop, then divergence.
- Computing val loss on a non-stratified sample → noisy curve; early-stop fires on noise. Use a fixed, large (≥100k tokens for LLM) val set.
- Soup-averaging across *different* base architectures or *different* random inits → models in different basins; averaging breaks them.

## Connections
- **[[lr-schedules]]**: WSD's decay phase is where averaging happens; cosine schedules give one "best" point and don't naturally produce averagable checkpoints.
- **[[adam]]**: optimizer state must be in the checkpoint or restart breaks.
- **[[dropout]]**: with dropout = 0 in pretraining, early stopping is the SFT regularizer of choice.
- **Llama-3 / OLMo-2 / DeepSeek-V3**: all publicly document checkpoint averaging across the decay phase; OLMo-2 also documents loss-spike rollback policy.
- **DPO / RL fine-tuning**: standard practice is to take the SFT-init's *averaged* weights, not the raw final SFT weights, as the RL initialization.
- **Karpathy** ([[karpathy-training-neural-net-recipe]]): "use a constant learning rate, save many checkpoints, and average at the end" — the recipe predates SWA's name.
