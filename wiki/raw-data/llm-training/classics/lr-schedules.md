<!-- scope: learning rate schedules for LLM pretraining and fine-tuning — warmup, inverse-sqrt, cosine, WSD (constant plus cooldown); what each primary source actually measured
     deps: [[adam]]
     see-also: [[weight-init]], [[gradient-clipping]], [[cooldown-scaling-beyond-fixed-durations]], [[resolving-scaling-discrepancies]], [[early-stopping-and-checkpointing]]
-->

# Learning Rate Schedules: Warmup, Inverse-Sqrt, Cosine, WSD
> **Composite card.** This card has no single primary artifact. Every claim below is attributed to one
> named source at a locus. Where a claim was in the previous version without such a locus, it was removed;
> see Verification. For the cooldown-versus-cosine evidence, cite [[cooldown-scaling-beyond-fixed-durations]]
> directly; for the role of decay in scaling laws, cite [[resolving-scaling-discrepancies]].

- **Core Insight:** The learning rate is raised over a warmup phase, held high, and decayed at the end; the decay is where most schedules differ. Hoffmann et al. 2022 measure that a cosine cycle stretched more than 25% beyond the actual number of training steps gives clearly worse loss (App. B, Fig. A1), and Hägele et al. 2024 measure that a constant LR followed by a cooldown of about 20% of steps matches a length-matched cosine (§3.2, Fig. 3).
- **Guideline:** When the token budget is fixed in advance, use cosine with the cycle length set to that budget and a 10× decay, because Hoffmann et al. tested cycle lengths of 1, 1.1, 1.25, 1.5, 2, and 5× the target and found degradation past 1.25× (App. B, Fig. A1, footnote 9). When the budget is not fixed, or checkpoints must be branched, use constant LR plus a short cooldown, because Hägele et al. match cosine at 22k/33k/44k steps with a 20% cooldown and nearly match it with 5% on a 20B-token run (§3.2–3.3).
- **Primary sources:** Vaswani et al. 2017 (inverse-sqrt with warmup); Loshchilov & Hutter 2017 (SGDR, cosine with warm restarts); Hoffmann et al. 2022 (Chinchilla, cosine cycle length); Hu et al. 2024 (MiniCPM, WSD); Hägele et al. 2024 (constant + cooldown); Porian et al. 2024 (decay and scaling laws); Ibrahim/Thérien et al. 2024 (re-warming and replay)
- **Year:** 2017–2024
- **URL:** https://arxiv.org/abs/1706.03762 · https://arxiv.org/abs/1608.03983 · https://arxiv.org/abs/2203.15556 · https://arxiv.org/abs/2404.06395 · https://arxiv.org/abs/2405.18392 · https://arxiv.org/abs/2406.19146 · https://arxiv.org/abs/2403.08763
- **Source type:** composite of papers (no single primary artifact)
- **Relevant topics:** optimization, training dynamics, decay/annealing phase, continual pretraining, scaling-law methodology

## Summary
Four schedule families appear in LLM training. **Warmup** raises the LR linearly from near zero to its peak over the first phase of training. **Inverse-square-root** decays as `step^(-0.5)` after warmup and is tied to `d_model` (Vaswani et al. §5.3). **Cosine annealing** decays from peak to a floor over a fixed cycle; SGDR introduced it with periodic warm restarts, and LLM pretraining adopted the single-cycle form. **WSD** (warmup–stable–decay), also called constant LR plus cooldown, holds the peak LR for most of training and decays over a short final phase, so the stop point need not be fixed in advance and stable-phase checkpoints can be branched.

## Key Contributions of each source
- **Vaswani et al. 2017 §5.3** — `lrate = d_model^(-0.5) · min(step_num^(-0.5), step_num · warmup_steps^(-1.5))`, with `warmup_steps = 4000`, Adam β = (0.9, 0.98), ε = 1e-9.
- **Loshchilov & Hutter 2017 (SGDR)** — cosine annealing with warm restarts, parameterized by the initial cycle length `T_0` and the multiplier `T_mult`; evaluated on CIFAR-10, CIFAR-100, EEG recordings, and downsampled ImageNet, reaching 3.14% and 16.21% error on CIFAR-10/100 (Abstract, §3–4). The schedule is not evaluated on language models in that paper.
- **Hoffmann et al. 2022 App. B** — cosine cycle length should match the number of training steps; overestimating beyond 25% gives clear degradation (Fig. A1). A 10× LR decay is used, and decaying by 10× is slightly better than decaying to 0.0, while decaying by only 5× is clearly worse (footnote 9).
- **Hu et al. 2024 (MiniCPM) §4** — defines WSD as `WSD(T; s) = (s/W)η` for `s < W`, `η` for `W < s < T`, and `f(s−T)η` for `T < s < S`, where `η` is the peak LR, `W` the warmup end step, `T` the stable-stage end step, `S` the total steps, and `f` a decreasing function. A decay over 10% of total tokens is enough to reach the best result; 2.5% falls short (§4.3).
- **Hägele et al. 2024 §3** — constant LR plus cooldown matches a tuned cosine and is less sensitive to the peak LR; the `(1-sqrt)` cooldown shape, `f(n, N, N_decay) = 1 − sqrt((n − (N − N_decay)) / N_decay)`, beats linear decay; benefits plateau near 20% of steps, and 5% with `(1-sqrt)` nearly matches cosine on a 20B-token run.
- **Porian et al. 2024** — careful LR decay is not essential for the Hoffmann et al. scaling law to hold; three other factors (last-layer FLOP counting, warmup duration, scale-dependent optimizer tuning) explain the Kaplan–Hoffmann discrepancy (Abstract; §3.4).
- **Thérien, Gupta et al. 2024** — for continual pretraining, LR re-warming plus re-decaying plus replay of previous data matches full re-training at 405M and 10B scale; replay of as little as 1% of previous data reduces forgetting substantially (Abstract; §1).

## Technical Details

**Linear warmup** (prepended to every schedule below):
`lr(t) = peak_lr · t / warmup_steps` for `t < warmup_steps`, where `t` is the step index.

**Inverse-square-root** (Vaswani et al. 2017, §5.3, Eq. 3):
`lr(t) = d_model^(-0.5) · min(t^(-0.5), t · warmup_steps^(-1.5))`, `warmup_steps = 4000`. Symbols: `d_model` model width; `t` step number. The LR rises linearly during warmup and falls as `1/sqrt(t)` afterwards.

**Cosine annealing** (single cycle, the form used in LLM pretraining):
`lr(t) = min_lr + 0.5 · (peak_lr − min_lr) · (1 + cos(π · (t − warmup) / (T − warmup)))`, where `T` is the cycle length in steps. Hoffmann et al. set `min_lr = peak_lr / 10` and `T` to the token budget (App. B, footnote 9). SGDR's restart parameters `T_0` and `T_mult` are not used in that setting.

**WSD / constant + cooldown** (Hu et al. 2024 §4.2, Eq. 1):
three phases — warmup to `η` over `[0, W]`; stable at `η` over `[W, T]`; decay `f(s−T)·η` over `[T, S]`. MiniCPM's released run uses exponential annealing `f(s−T) = 0.5^((s−S)/T)` with the decay stage set to 5000 steps, about 20B tokens (§6.2). Hägele et al. use linear and `(1-sqrt)` cooldowns instead (§3.2).

**Which data to use during the decay phase.** MiniCPM mixes high-quality knowledge- and ability-oriented SFT data into the pretraining mixture during the annealing phase, keeping coarse-quality pretraining data for the stable phase, and runs a separate SFT stage of about 6B tokens afterwards (§5, §6.2). The paper's stated reason is that concentrating high-quality data in a bounded phase avoids repeating a small dataset across an open-ended pretraining run (§5).

**Re-warming for continual pretraining.** Starting a new pretraining phase from a decayed checkpoint requires re-warming the LR and re-decaying it; re-warming alone increases forgetting, which replay of previous data mitigates (Thérien, Gupta et al. 2024, §1 contributions 1–2).

**Why warmup is used with Adam.** In the first steps the second-moment estimate `v_t` is computed from few samples and its bias correction divides by `1 − β₂^t`, a small number, so the update magnitude is poorly controlled. This card states the mechanism; see [[adam]] for the derivation. No number in this card is attached to that mechanism, because none of the sources above isolates it experimentally.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Transformer (big), WMT14 EN-DE | 213M | pretrain | schedule / warmup | inverse-sqrt, `warmup_steps = 4000` | arXiv:1706.03762 §5.3 Eq. 3 | verified 2026-09-18 | no ablation reported |
| Chinchilla | 70B | pretrain | peak LR | 1 × 10⁻⁴ | arXiv:2203.15556 Table 4 | verified 2026-09-18 | no ablation reported |
| Chinchilla | 70B | pretrain | batch size (tokens) | 1.5M → 3M (doubled midway) | arXiv:2203.15556 Table 4 | verified 2026-09-18 | no ablation reported |
| Chinchilla scaling runs | 70M–10B | pretrain | cosine cycle length / decay | cycle matched to the token budget; 10× decay | arXiv:2203.15556 App. B, footnote 9 | verified 2026-09-18 | App. B Fig. A1: cycle lengths 1/1.1/1.25/1.5/2/5×; degradation past 1.25× |
| MiniCPM-2.4B | 2.4B | pretrain-stable | batch size / peak LR | 3.93M tokens; 0.01 | arXiv:2404.06395 §6.2 | verified 2026-09-18 | §3 model wind-tunnel experiments |
| MiniCPM-2.4B | 2.4B | pretrain-decay/anneal | decay length / shape / data | 5000 steps (20B tokens); exponential `0.5^((s−S)/T)`; pretraining data mixed with high-quality SFT data | arXiv:2404.06395 §6.2, §5 | verified 2026-09-18 | §4.3: 10% of tokens suffices, 2.5% falls short (0.036B models) |
| MiniCPM-2.4B | 2.4B | SFT | tokens | ~6B | arXiv:2404.06395 §6.2 | verified 2026-09-18 | §5 comparison A-1/A-2/B-1/B-2 |
| Hägele et al. LLM sweep | 124M–360M, 1B | pretrain | cooldown fraction / shape | 20% of steps; `(1-sqrt)` preferred over linear | arXiv:2405.18392 §3.2–3.3 | verified 2026-09-18 | §3.3 Fig. 5: benefits plateau near 20%; Fig. 6: 5% nearly matches cosine at 20B tokens |

Per-size peak-LR defaults for 1B/70B/405B models, SFT and RL learning-rate ranges, and warmup-step counts for GPT-3 and Llama-3 were present in the previous version without loci and are not reproduced here; take them from the model-report cards, which carry their own loci.

## Findings relevant to generality
- Decay-phase data selection changes what the model generalizes to: MiniCPM reports that mixing SFT-style data into the annealing phase reduces loss with respect to the SFT distribution rather than the pretraining distribution (§5).
- Schedule choice affects scaling-law methodology more than final quality: Porian et al. find the Hoffmann et al. law holds without careful decay, and adding cosine decay improves the fit only slightly (R² 0.993 → 0.999 on OpenWebText2, §3.4).
- Continual pretraining without replay loses previously acquired ability; replay of about 1–5% of previous data recovers it at 405M and 10B scale (Thérien, Gupta et al. 2024, §1).

## Connections
- [[adam]] — warmup interacts with the bias-corrected second moment; the schedule shape is independent of the optimizer's own hyperparameters.
- [[gradient-clipping]] — the other control on early-training update size.
- [[weight-init]] — width-dependent LR scaling is decided there, not by the schedule shape.
- [[cooldown-scaling-beyond-fixed-durations]] — the dedicated card for Hägele et al. 2024; cite it, not this card, for cooldown evidence.
- [[resolving-scaling-discrepancies]] — the dedicated card for Porian et al. 2024.
- [[early-stopping-and-checkpointing]] — stable-phase checkpoints are the inputs to branching and averaging.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1706.03762 (v7), https://arxiv.org/abs/1608.03983 (v5), https://arxiv.org/abs/2203.15556 (v1), https://arxiv.org/abs/2404.06395 (v3), https://arxiv.org/abs/2405.18392 (v3), https://arxiv.org/abs/2406.19146 (v4), https://arxiv.org/abs/2403.08763 (v4)
- Corrections to the previous card version:
  - "over- or under-shooting cosine costs ~0.5–1% on validation perplexity" and "Chinchilla showed +0.3–1% perplexity" → Hoffmann et al. report no percentage; Fig. A1 shows cycle lengths of 1, 1.1, 1.25, 1.5, 2, and 5× the target steps, with clear degradation when the cycle overshoots by more than 25%, and no undershoot measurement (App. B; §3.1 footnote 7).
  - "Chinchilla Figure A1: cosine schedule mismatch costs are real but small" → the figure's stated conclusion is "clear drops in performance" past 25% overshoot; "small" is not in the source.
  - "Cosine annealing (Loshchilov 2017, SGDR): no hyperparameters beyond min/max" → SGDR is parameterized by `T_0` and `T_mult` (§3, Fig. 1).
  - "demonstrably superior on image classification — adopted across LLM pretraining (GPT-3, Llama, Chinchilla)" → SGDR reports CIFAR-10/100, EEG, and downsampled ImageNet results; the adoption claim is not in that paper and the per-model schedules belong on the model-report cards.
  - "WSD (Hu 2024 / Hägele 2024): constant high-LR phase + short final decay matches cosine at any chosen stop point" → Hägele et al. demonstrate the match at 22k, 33k, and 44k steps for a 210M model with a 20% cooldown (§3.2, Fig. 3); "any chosen stop point" overstates the tested range.
  - "Decay phase is typically 10–20% of total tokens" → MiniCPM measures 10% of tokens as sufficient and 2.5% as insufficient on 0.036B models (§4.3); Hägele et al. measure a plateau near 20% and near-parity at 5% with `(1-sqrt)` (§3.3).
  - "This is how DeepSeek and MiniCPM produce many model variants from one pretraining trunk" → MiniCPM states the branching property (§4.3); no DeepSeek locus was checked, so the DeepSeek attribution is dropped.
- Removed as unsupported by the sources read: the "Modern defaults" table (warmup step ranges, per-size peak LRs, SFT LR 2e-5 attributed to Llama-3, RL LR 1e-6–1e-5, min-LR rules); "GPT-3 used 375M-token warmup; Llama-3 used 8000 steps; 0 warmup = divergence at 7B+"; "warmup too short with high LR → loss spike at step ~150"; "constant LR with no decay → final loss is 1–3% worse than cosine"; "WSD decay phase too short (<5%) → underperforms cosine; too long (>30%) → wastes the stable phase"; "inverse-sqrt loses to cosine in practice for fixed budget"; the Karpathy quotation about constant LR for sanity checking; the claim that μP leaves the schedule shape unchanged.
- Not reported by the sources read: schedules for RL fine-tuning; batch-size schedules; interaction between schedule shape and weight decay; per-size LR transfer rules.
