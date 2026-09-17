<!-- chapter: ch-03
     track: foundations
     kind: content
     title: Learning-Rate Schedules, Batch Size, Initialization, and Normalization
     deps: [ch-02]
     sources: [[lr-schedules]], [[weight-init]], [[batch-vs-layer-norm]], [[minicpm]], [[critical-batch-size-pretraining]], [[large-batch-training-noise-scale]], [[deepseek-llm]], [[olmo-2]], [[llama-3]], [[llama-3-recipe]], [[overtrained-lms-harder-to-finetune]], [[cooldown-scaling-beyond-fixed-durations]], [[cooldown-scaling-beyond-fixed-durations-recipe]], [[small-scale-proxies-instabilities]], [[resolving-scaling-discrepancies]], [[resolving-scaling-discrepancies-recipe]], [[datadecide]], [[continual-pretraining-rewarm-replay]], [[prolong]], [[prolong-recipe]], [[kimi-k1-5-recipe]], [[yarn]], [[rope-nope-hybrid-attention]], [[llama-4]], [[qwen-2.5]]
     figures: figures/lr-schedules.html
     revised: 2026-09 (generality revision)
-->

# Chapter 3 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization

> **Core insight.** The final phase of the learning-rate schedule has a measured effect on loss and downstream scores: in WSD and cooldown schedules the loss falls faster during the decay than during the constant phase before it (MiniCPM §4.3; Hägele et al. Fig. 3), and a 20% cooldown from a constant learning rate matched a length-matched cosine on 8 downstream tasks at 1B parameters (46.20 vs 46.26; [[cooldown-scaling-beyond-fixed-durations]]). Data changes made during the decay phase are measurable against a decay on the unchanged mixture: OLMo 2 7B's 10-task average rose from 53.0 to 62.9 after mid-training (three 50B-token anneals, averaged), including held-out MMLU-Pro 27.4 → 31.0, while an anneal on the unchanged mixture accounted for 4.4 of the 6.1-point gain of the final mixture on OLMES, the OLMo 2 multiple-choice evaluation average ([[olmo-2]], Tables 9 and 11). Batch size, initialization, and normalization determine which learning rates are stable: critical batch size grows mainly with tokens trained rather than model size at 85M–1.2B ([[critical-batch-size-pretraining]]), and qk-layernorm plus z-loss let models up to 1.2B train to low loss across learning rates from 3e-4 to 0.3 ([[small-scale-proxies-instabilities]]). These choices also limit what later stages can change: an OLMo-1B checkpoint pre-trained on 3T tokens was over 2% worse on multiple standard benchmarks after instruction tuning than its 2.3T-token checkpoint ([[overtrained-lms-harder-to-finetune]]).
>
> **Guideline.** When the final token budget is not fixed, or several budgets are needed from one run, use a constant learning rate after warmup and branch decays of 10–20% of tokens from saved checkpoints, because MiniCPM and Hägele et al. both matched cosine this way ([[minicpm]], [[cooldown-scaling-beyond-fixed-durations]]); otherwise use cosine sized to the tokens that will actually be trained. When high-quality or targeted data is added during the decay, compare against an anneal on the unchanged mixture and report held-out suites, because the learning-rate decay alone raised OLMo 2 7B's OLMES score from 69.6 to 74.0 ([[olmo-2]], Table 11). When tuning at small scale for a larger run, fix the parametrization first (μP or a fitted learning-rate and batch rule) and run each proxy long enough to pass learning-rate crossovers, because OLMo 2's 3e-4 vs 6e-4 crossover came after 200B tokens ([[olmo-2]], §4.1). When training a deep pre-norm Transformer at a high learning rate, add QK-norm and z-loss and track maximum attention logit and gradient RMS, because these predicted a 4.8B divergence from smaller models ([[small-scale-proxies-instabilities]]); when long-context retrieval is a target, evaluate that QK-norm has not flattened attention ([[rope-nope-hybrid-attention]]).

## Why this chapter matters for a general-purpose model

Pre-training sets the representation that every later stage modifies. This chapter covers four settings that decide whether pre-training is stable and how much capability it leaves for mid-training, SFT, preference optimization, and RL: the learning-rate schedule, the batch size, the initialization, and the normalization layers. It sits between optimizer and precision choices (ch-01, ch-02) and scaling-law and data chapters (ch-08a onward), and it supplies the schedule vocabulary used by mid-training (ch-32), continual pre-training (ch-32a), and context extension (ch-32b).

The problems are measurable. A loss spike or divergence wastes compute and can require a restart from an earlier checkpoint. A schedule that decays at the wrong time leaves loss that a correctly timed decay would have removed. A batch size above the critical batch size spends more tokens for the same loss. A small-scale hyperparameter search that does not transfer picks the wrong learning rate for the large run. For a general-purpose model, each choice has a second question: does it improve many held-out tasks, or only the benchmarks used during development, and does it leave the checkpoint easy to adapt later? The measurement rules for that question are in ch-00.

## §1 Learning-rate schedules

**Definition.** A learning-rate schedule is the function η(t) that sets the optimizer step size at step t. Modern LLM schedules have a warmup phase, a main phase, and a decay phase.

**Problem.** The schedule has two measurable failure modes: instability early in training (loss spikes or divergence when the step size is large before optimizer statistics and activations are well scaled) and a higher final loss than a completed decay would reach (a run that stops before its decay has finished).

### 1.1 Warmup

**Mechanism.**
1. Warmup raises η linearly from 0 (or a small value) to the peak over W steps: η(t) = η_peak · t / W for t < W.
2. AdamW (ch-01) divides the first-moment estimate by the square root of the second-moment estimate. Its bias corrections m̂_t = m_t / (1 − β₁ᵗ) and v̂_t = v_t / (1 − β₂ᵗ) remove the underestimate caused by initializing m and v at zero.
3. At t = 1 the corrected estimates are m̂₁ = g and v̂₁ = g², so the update is −η · g / (|g| + ε) ≈ −η · sign(g) for every coordinate with |g| ≫ ε. The step size per coordinate equals η regardless of how large or small that coordinate's gradient is, and v̂ is estimated from one sample.

**Worked example (derived).** Take β₁ = 0.9, β₂ = 0.95, ε = 1e-8 and a coordinate with g = 0.01. Then m₁ = 0.001, v₁ = 5e-6, m̂₁ = 0.01, v̂₁ = 1e-4, and the update is η · 0.01 / (0.01 + 1e-8) ≈ η. A coordinate with g = 1e-5 gets η · 1e-5 / (1e-5 + 1e-8) = 0.999 η. Without warmup and with η = 3e-4, every weight moves by about 3e-4 on step 1, which is 1.5% of a weight of size 0.02 (the OLMo 2 init standard deviation). With a 2,000-step warmup, step 1 uses η = 1.5e-7. The explanation that early instability comes from these sign-like, high-variance first updates is an Interpretation; the evidence below is empirical.

**Evidence.**
- Post-LN Transformer on IWSLT14 De-En: Adam without warmup reached BLEU 8.45, with warmup "around 34"; with T_warmup = 500, BLEU was 31.16 at lr_max 5e-4 and 2.77 at 1e-3 ([[batch-vs-layer-norm]] excerpt; Xiong et al. 2020 §3.2, Fig. 2). Result (single study).
- Decoder-only models from 2.4M to 1.2B on C4, total 1e5 steps: warmups of 50 to 25,000 steps were tested, and longer warmup reduced learning-rate sensitivity and loss, most for the larger models, which were not stable at LR 0.3 without long warmup ([[small-scale-proxies-instabilities]], §3.2.1, Fig. 5). Result (single study).
- OLMo 2 built its spike-reproducing baseline "mainly [by] reducing the warmup period" ([[olmo-2]], §3.2). Replicated with the two results above: shorter warmup raises instability.

**Conditions and limits.** Xiong et al. show that Pre-LN Transformers train without warmup in their tasks (§4), so warmup length interacts with normalization placement (§5). The warmup lengths used by large runs (2,000 steps for OLMo 2, 8,000 for Llama 3 405B) were not ablated in those reports.

### 1.2 Cosine and inverse square root

**Formulas.**

```
cosine:      η(t) = η_min + ½ (η_peak − η_min) (1 + cos(π · (t − W) / (T − W)))        for W ≤ t ≤ T
inverse sqrt: η(t) = d_model^-0.5 · min(t^-0.5, t · W^-1.5)
```

η_peak is the peak learning rate, η_min the final learning rate, W the warmup length, T the schedule horizon, and d_model the model width. The inverse square root form is Vaswani et al. §5.3 Eq. 3 with W = 4000 ([[lr-schedules]] excerpt).

**Worked example.** Inverse square root with d_model = 512: step 4,000 gives 512^-0.5 · 4000^-0.5 ≈ 6.99e-4, and step 16,000 gives ≈ 3.49e-4. Cosine for Llama 3 405B (η_peak 8e-5, η_min 8e-7, [[llama-3-recipe]]): halfway through the decay η = 8e-7 + ½ · 7.92e-5 = 4.04e-5; at 75% of the decay, cos(0.75π) = −0.707 gives η = 1.24e-5. The final learning rate is 1% of the peak. OLMo 2 decays to 10% of the peak instead (`alpha_f: 0.1`, [[olmo-2]] excerpt), so "min LR" is a per-run choice, not a fixed fraction.

**Horizon.** In MiniCPM's 0.036B experiments, the lowest loss at S = 20N, 40N, 60N, and 80N tokens was "always achieved by the Cosine(T) where T = S. Both T < S and T > S are not optimal" ([[minicpm]], §4.1, Fig. 4). Hägele et al. show the same pattern: each cosine is best only at its own endpoint ([[cooldown-scaling-beyond-fixed-durations]], Fig. 1). Replicated. Neither source gives a single percentage penalty for a mismatched horizon; the penalty depends on the mismatch and model.

### 1.3 Warmup-stable-decay, cooldowns, and multi-step schedules

**Definition.** WSD (warmup-stable-decay) holds η at its peak after warmup and then decays it over a final fraction of training: WSD(T; s) = (s/W)η for s < W, η for W < s < T, and f(s − T)η for T < s < S, where f is decreasing with 0 < f ≤ 1 ([[minicpm]], §4.2, Eq. 1). Hägele et al. call the same shape "constant LR + cooldown" and propose f = 1 − √((n − (N − N_decay)) / N_decay), the (1-sqrt) cooldown ([[cooldown-scaling-beyond-fixed-durations]], §3.2). A multi-step schedule lowers η in discrete drops.

**Problem it addresses.** A cosine schedule must fix T in advance. A run that later receives more data, or a scaling study that needs several token counts, would otherwise need one full run per budget.

**Evidence.**
- Loss during decay: in MiniCPM (0.036B) the loss "experiences a significant rapid decline" in the decay stage and reaches or passes the cosine loss at T = S (§4.3, Fig. 5); in Hägele et al. (210M) "the cooldown phase initiates a sharp decrease in loss" to match cosine (Fig. 3). Replicated. What share of total loss reduction happens during decay is not reported by either source.
- Decay length: MiniCPM found a decay of 10% of total tokens "sufficient to achieve the best results, while a decay of 2.5% of total tokens falls short" (§4.3). Hägele et al. found gains plateau at about 20% of steps for a 124M model and that a 5% (1-sqrt) cooldown nearly matched cosine on a 200k-step run (Fig. 5–6). Replicated for 10–20%; the shortest sufficient length differs between the two studies.
- Downstream: a 1B model on 100B FineWeb tokens scored 46.26 (cosine to 10%), 46.23 ((1-sqrt) 20%), 46.20 (linear 20%), and 45.88 (cosine to 0) averaged over 8 tasks ([[cooldown-scaling-beyond-fixed-durations]], Table 4). Result (single study, no seeds reported).
- Multi-step: DeepSeek LLM warms up for 2,000 steps, drops to 31.6% of the peak after 80% of tokens and to 10% after 90%; for a 1.6B model on 100B tokens the final result is "essentially consistent" with cosine ([[deepseek-llm]], §2.3, Fig. 1a). WSD comes from MiniCPM; DeepSeek LLM does not use WSD.

**Worked example (Interpretation of printed notation).** MiniCPM decays with f(s − T) = 0.5^((s−S)/T), "in which T is set to be 5000 steps (20B tokens)" ([[minicpm]], §6.2). Read as a half-life of 5,000 steps, η falls to 50% after 5,000 steps, 25% after 10,000, and 10% after log₂(10) · 5,000 ≈ 16,610 steps. MiniCPM reports that loss "still drops after the learning rate drops below 10% of the max learning rate" (§6.4).

The figure [figures/lr-schedules.html](figures/lr-schedules.html) plots these published schedules on one normalized axis and lets the reader move the stop point and the decay length, so the learning rate of each schedule at that stop point can be read from a table; its second panel computes the batch-size trade-off of §3.1.

### 1.4 Planned schedule versus the schedule that was run

A report's schedule row can describe a plan that was cut short. OLMo 2 7B planned a cosine to 10% of 3e-4 over 5T tokens, with a warmup of 8,388,608,000 tokens (2,000 steps of 1024 × 4096), and stopped the cosine "after 4T" ([[olmo-2]], Table 3). Its stage-2 config loads checkpoint `step928646` and starts at `learning_rate: 0.000061499`, then decays linearly to 0 over 50B tokens.

**Worked example (derived).** Step 928,646 × 4,194,304 tokens = 3.895T tokens. The cosine progress is (3.895e12 − 8.39e9) / (5e12 − 8.39e9) = 0.7786, so η = 3e-5 + ½ · 2.7e-4 · (1 + cos(0.7786π)) = 6.14e-5, within 0.3% of the config value. The linear decay covers 50e9 / 4,194,304 = 11,921 steps; the config's `stop_at: 11931` adds 10. The released 7B was therefore trained with a truncated cosine followed by a separate linear decay on different data, not with the 5T cosine in Table 3. OLMo 2 states that cutting the cosine and replacing its tail with a linear decay costs "little loss of performance", from OLMo-0424 experience without numbers (§4.1).

The same check exposes a conflict for OLMo 2 13B. Table 3 prints a peak of 9.0e-4, while the released stage-1 YAML prints `learning_rate: 3.0e-4`. The 13B stage-2 config loads `step596057` (596,057 × 2048 × 4096 = 5.0T tokens, derived) and starts at `learning_rate: 9e-5`, which is the 10% floor of a 9e-4 cosine at its 5T horizon. The stage-2 value therefore matches Table 3 and not the stage-1 YAML (derived), so the table value is the better estimate of what produced the checkpoint.

Llama 3 405B anneals "on the final 40M tokens" with η linear to 0 ([[llama-3-recipe]]). At the 16M-token batch in §3.4.1 this is 2.5 optimizer steps (derived). The report does not state the batch size during annealing, so this is an Open question about the printed number, not evidence about annealing.

## §2 The decay phase as the place where data quality acts

**Definition.** Annealing (also called cooldown or mid-training when the data changes) is the decay phase run on a mixture that upsamples high-quality or targeted data. ch-32 covers mid-training data; this section covers what the schedule contributes and how to measure the result.

**Problem.** A decay phase lowers loss even when the data does not change. A gain after a data change during decay therefore has two parts: the learning-rate decay and the data. A second problem is that the added data often overlaps with the benchmarks used to judge it.

**Mechanism of the measurement.**
1. Start from a stable-phase checkpoint.
2. Run the decay on the unchanged pre-training mixture (control).
3. Run the same decay on the candidate mixture.
4. Compare both against the checkpoint on development tasks, on held-out tasks never used for decisions, and on the capability targeted by the data.

**Evidence: separating decay from data (OLMo 2, [[olmo-2]] Table 11).** From a 7B checkpoint at 4T tokens, 50B-token runs gave (OLMES / OLMES-Gen / MMLU / GSM*; OLMES is the multiple-choice evaluation average and OLMES-Gen the generative-task average of OLMo 2's evaluation suite, and GSM* is 200 GSM8K questions used for development):

| Run | OLMES | OLMES-Gen | MMLU | GSM* |
|---|---|---|---|---|
| Checkpoint, no anneal | 69.6 | 63.2 | 59.8 | 28.5 |
| Anneal on pre-training mix (learning-rate decay only) | 74.0 | 64.5 | 61.8 | 27.0 |
| Anneal on filtered web + reference data | 75.2 | 63.8 | 63.1 | 28.5 |
| + math data | 75.7 | 69.7 | 62.3 | 52.0 |
| + math + instruction data | 75.7 | 70.2 | 63.1 | 46.5 |

The learning-rate decay alone accounts for 4.4 of the 6.1-point OLMES gain of the final mix, and none of the GSM* gain. The math data accounts for the GSM* gain. Result (single study, one run per row).

**Evidence: held-out versus development tasks (OLMo 2, Table 9).** After the final mid-training (3 anneals averaged), 7B development scores rose MMLU 59.8 → 63.7 and GSM8K 24.1 → 67.5; held-out scores rose MMLU-Pro 27.4 → 31.0 and TriviaQA 74.6 → 78.0. The held-out suite was "not used for model development decisions", but GSM8K was only partially held out (200 of 1319 examples used for development). The held-out gains are smaller than the targeted GSM8K gain but not zero. Result (single study).

**Evidence: annealing with SFT-style data (MiniCPM, [[minicpm]] Table 1).** MiniCPM-2.4B decayed with high-quality and SFT data mixed into pre-training data, then 4B tokens of SFT, scored C-Eval 52.6, MMLU 50.9, GSM8K 42.3, HumanEval 30.4, versus 40.0, 44.6, 27.7, 27.7 when the decay used pre-training data only. For the 1.2B model, doubling SFT tokens (6B → 12B) without the annealing data left MMLU at 47.9 and moved GSM8K from 34.2 to 34.4; with the annealing data and 6B SFT tokens, MMLU was 49.6 and MATH 10.5 (from 7.9), but GSM8K fell to 31.8 (from 34.2). The added data targets the same domains as the benchmarks (knowledge, math, code), no seeds are reported, and one benchmark fell, so this is a Result (single study) about targeted gains, not a demonstration of breadth.

**Evidence: annealing on benchmark training sets (Llama 3, [[llama-3]] §3.1.3).** Annealing on the GSM8K and MATH training sets raised a pre-trained Llama 3 8B's validation scores by 24.0% and 6.4%, while "the improvements on the 405B model are negligible". Llama 3 then excluded "any training sets from commonly used benchmarks" from its annealing data "to assess the true few-shot learning capabilities and out-of-domain generalization". The report also uses annealing to value a small dataset: a 50%-trained 8B is annealed linearly to 0 over 40B tokens with 30% weight on the new data. Result (single study, no seeds).

**Evidence: peak learning rate and the anneal (OLMo 2 §4.1, Table 8).** At 7B scale, runs with peak LR 3, 6, 9, and 12 ·10^-4 were compared after decaying to 0. After 300B + 50B tokens, OLMES scores were 62.5, 63.9, 64.1, 63.6; after 2T + 100B high-quality tokens, 3e-4 and 6e-4 scored 73.8 and 73.9. The authors conclude that a higher learning rate "does make mid-training more effective, but it does so by exactly the amount that the pretraining is worse." GSM8K was 2.8 points higher for the higher learning rate, which the authors say needs more study.

**Conditions and limits.** The OLMo 2 and MiniCPM ablations are single runs. None of these studies measures whether annealing data changes performance on capabilities absent from both the data and the evaluation suite.

**Implication for a general-purpose model.** A decay phase is a measurement opportunity: the control anneal isolates the schedule, and the held-out suite separates targeted gains from broad ones. Llama 3's 8B-versus-405B result shows that the same annealing data can change a smaller model's benchmark score while leaving a larger model's unchanged, so a gain at one size is not evidence for another.

## §3 Batch size and learning-rate scaling

**Definition.** The batch size B is the number of examples (or tokens) averaged in one gradient step. The critical batch size is the batch size beyond which increasing B stops reducing the number of steps proportionally.

**Problem.** Larger batches allow more data parallelism and fewer steps, but past some size they need more total tokens to reach the same loss. The best learning rate also changes with B.

### 3.1 Gradient noise scale and the steps–tokens trade-off

**Mechanism** ([[large-batch-training-noise-scale]], McCandlish et al. 2018).
1. A batch gradient averages B per-example gradients; its covariance is Σ/B, where Σ is the per-example gradient covariance.
2. Under a quadratic model of the loss with true gradient G and Hessian H, the best step size and the loss decrease per step both scale as 1/(1 + B_noise/B).
3. Steps S and examples E needed to reach a target loss then satisfy a hyperbolic trade-off.

**Formulas.**

```
B_simple = tr(Σ) / |G|²
ε_opt(B) = ε_max / (1 + B_noise / B)
S / S_min − 1 = (E / E_min − 1)^(−1),     B_crit = E_min / S_min
```

B_simple is the simple noise scale (it assumes H is a multiple of the identity), B_noise = tr(HΣ)/(GᵀHG), ε_max is the best step size with the full-batch gradient, S_min and E_min are the minimum steps and examples, and B_crit is the critical batch size (Eq. 2.6–2.12).

**Worked example (derived from the card, B_noise = 1,000).** Steps relative to minimum are 1 + B_noise/B and examples relative to minimum are 1 + B/B_noise:

| B | Steps / S_min | Examples / E_min | ε_opt / ε_max |
|---|---|---|---|
| 250 | 5.0 | 1.25 | 0.20 |
| 500 | 3.0 | 1.5 | 0.33 |
| 1,000 | 2.0 | 2.0 | 0.50 |
| 2,000 | 1.5 | 3.0 | 0.67 |
| 4,000 | 1.25 | 5.0 | 0.80 |

Doubling B from 250 to 500 multiplies the best step size by 1.67, close to linear scaling (×2). Doubling from 2,000 to 4,000 multiplies it by 1.2. Under this quadratic SGD model, ε_opt/ε_max = B/(B + B_noise), so the linear-scaling prediction B/B_noise is within 10% of it only while B ≤ B_noise/9 (derived); for Adam the measured exponent is between 0.5 and 1.0 (below).

**Evidence.** B_simple predicted B_crit to within an order of magnitude on 8 tasks where B_crit ranged from 20 to over 10 million, and both rose by at least an order of magnitude during training ([[large-batch-training-noise-scale]], §3, Fig. 4). For Adam or RMSProp, the best learning rate followed ε ∝ B^α with α between 0.5 (square-root scaling) and 1.0 (linear scaling), then levelled off (§3.1, App. E.2). Result (single study across 8 tasks).

### 3.2 Critical batch size in language-model pre-training

Zhang et al. define the critical batch size B* as the largest batch whose step count is at most 20% above the linear-scaling prediction from the best batch ([[critical-batch-size-pretraining]], §3.1). With 85M–1.2B models on C4 at context length 512 (batch in sequences):

- Chinchilla setting (tokens scaled with model size): B* = 93.20 · N^0.47, N in millions (§3.2). Derived: 151M gives ≈ 985 sequences (≈ 0.5M tokens) and 1.2B gives ≈ 2,610 sequences (≈ 1.3M tokens).
- Fixed 3.072B tokens, model size varied: B* = 621.341 · N^0.087 (§3.3). Derived: 151M gives ≈ 961 and 1.2B gives ≈ 1,151.
- Fixed 302M model, tokens varied from 0.28× to 4× the Chinchilla count: B* rises with tokens (§3.3, Fig. 1).

The authors conclude that the growth of B* in compute-optimal training "is more strongly attributed to extended data size or training durations rather than the increase of model size". Result (single study; C4 only; up to 1.2B). Consistent with McCandlish et al., who found larger LSTMs reach larger noise scales only through lower loss (§3.2, Fig. 8).

**Implication.** A run that trains a fixed model on more tokens can raise its batch size later in training. Llama 3 405B did this: 4M tokens, then 8M after 252M tokens, then 16M after 2.87T tokens, with a smaller early batch "to improve training stability" ([[llama-3-recipe]]). MiniCPM-1.2B raised batch from 2M to 4M and observed a loss drop that "might have a similar effect as decreasing learning rate" ([[minicpm]], §6.4).

### 3.3 Fitted learning-rate and batch-size rules

- DeepSeek LLM fit η_opt = 0.3118 · C^−0.1250 and B_opt = 0.2920 · C^0.3271 on budgets from 1e17 to 2e19 FLOPs, treating settings within 0.25% of the best loss as near-optimal ([[deepseek-llm]], §3.1, Eq. 1). Worked example (derived): for the 7B model, C = M · D = 4.23e10 FLOPs/token · 2.0e12 tokens = 8.46e22, giving η = 4.25e-4 and B = 9.2M tokens; Table 2 uses 4.2e-4 and 2304 × 4096 = 9.4M tokens.
- Porian et al. fit BS = 0.00037 · N^0.703 and LR = 3.7 · N^−0.36 (batch in 2048-token sequences) on a sweep of models up to 221M parameters, each trained for 20 tokens per parameter on RefinedWeb, applied the laws to models up to 901M, and found AdamW β₂ = 0.95 suboptimal at batch sizes of 128 and below ([[resolving-scaling-discrepancies]], §3.5, Fig. 3). Their 901M prescription (LR 0.0024, batch 640) gave the lowest loss among 13 configurations ([[resolving-scaling-discrepancies-recipe]]).
- Both DeepSeek and Porian et al. find an optimal batch size below which performance degrades, which Porian et al. say "appears to contradict" the view that any batch below the critical size is equally good ([[resolving-scaling-discrepancies]], §3.5). Their learning-rate exponents differ by 0.11 and predictions by a factor of 2–3. Open question: which rule transfers to a new data mixture.

## §4 Initialization and hyperparameter transfer

### 4.1 Variance-preserving initialization

**Definition.** Initialization sets the distribution of weights before training. Variance-preserving schemes choose the weight variance so activation and gradient variances stay roughly constant across layers.

**Formulas** ([[weight-init]] excerpt).

```
Glorot (Xavier):  Var(W) = 2 / (n_in + n_out)
He (Kaiming):     Var(W) = 2 / n_in               (ReLU)
LeCun / fan-in:   Var(W) = 1 / n_in
```

n_in and n_out are the fan-in and fan-out of the layer.

**Worked example (derived).** For a 4096 × 4096 matrix, Glorot and fan-in give std √(1/4096) = 0.0156, and He gives √(2/4096) = 0.0221. OLMo 2 7B (d_model 4096) and 13B (5120) both use std 0.02 for every parameter, which is between the fan-in and He values at 4096 and above fan-in (0.0140) at 5120.

### 4.2 Residual scaling: GPT-2, OLMo-0424, and OLMo 2

- GPT-2 scales "the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers" (§2.3), and moved layer normalization to the input of each sub-block with a final layer norm ([[weight-init]] excerpt). The paper does not say whether N counts blocks or sub-layers. For the 48-layer GPT-2 XL the factor is 1/√48 = 0.144 if N counts blocks and 1/√96 = 0.102 if it counts two sub-layers per block (derived).
- OLMo-0424 scaled input projections by 1/√d_model and output projections by 1/√(2 · d_model · layer_idx). OLMo 2 replaced this with N(0, 0.02) truncated at 3 standard deviations for every parameter. In a reduced-warmup test that reproduces spikes, the gradient spike score (percentage of values ≥ 7 standard deviations from a 1,000-step rolling mean) fell from 0.40 to 0.03, and "the new initialization converges slightly slower" ([[olmo-2]], §3.2, Fig. 4). Result (single study, one run each).

**Conditions and limits.** OLMo 2's result does not show that depth-scaled init is harmful in general; it compares two specific schemes in one architecture that also changed normalization. Depth scaling is not a universal practice: OLMo 2 dropped it, and DeepSeek LLM reports a single init std of 0.006 ([[deepseek-llm]], §2.3).

### 4.3 μP: why a small proxy can choose a large model's learning rate

**Definition.** μP (maximal update parametrization) sets initialization variance, learning rate, and output multipliers per layer type as functions of width so that the best hyperparameters stay approximately constant as width grows. μTransfer tunes a small proxy and copies the hyperparameters to the large model ([[weight-init]] excerpt; Yang et al. 2022).

**Formula (Tensor Programs V, Table 3; standard parametrization in parentheses where different).**

| | Input weights and biases | Output weights | Hidden weights |
|---|---|---|---|
| Init. variance | 1/fan_in | 1/fan_in² (1/fan_in) | 1/fan_in |
| SGD LR | fan_out (1) | 1/fan_in (1) | 1 |
| Adam LR | 1 | 1/fan_in (1) | 1/fan_in (1) |

Transformer μP also scales attention logits by 1/d instead of 1/√d. Under Adam, hidden-weight learning rate scales as 1/fan_in; under SGD it is width-independent.

**Worked example (derived, hypothetical widths).** With MiniCPM's operations (2-D tensor LR = base LR / (d_m / d_base); 2-D init std = init_std / √(d_m / d_base)), base LR 0.01 and init_std 0.1 found on a proxy, and a target of d_m / d_base = 4096 / 256 = 16, hidden matrices get LR 0.01 / 16 = 6.25e-4 and init std 0.1 / 4 = 0.025 ([[minicpm]], App. A.1, Table 7).

**Evidence.**
- GPT-3 6.7B tuned on a 40M-parameter width-256 proxy, with tuning cost "only 7% of total pretraining cost": validation loss 1.98 versus 2.03 for a re-run with the original hyperparameters. The authors state two confounds: the re-run mistakenly used absolute attention, and the μP model was trained in FP32 to avoid divergences while the baselines used FP16 (Tensor Programs V §7.4, Table 7). Result (single study with stated confounds).
- MiniCPM: with μP, the best base LR "remains around 0.01" from 0.04B to 0.5B and was confirmed at 2.1B ([[minicpm]], §3.3, Fig. 3). Replicates the learning-rate stability claim.
- Wortsman et al.: a simple μP variant stabilized the optimal learning rate across width but did not lower loss or learning-rate sensitivity, and did not remove the need for qk-layernorm at high learning rates ([[small-scale-proxies-instabilities]], §3.2.4).

**Conditions and limits.** Tensor Programs V Table 1 lists regularization (dropout, weight decay) as not transferable, and transfer across depth, batch size, training time, and sequence length as "empirically validated only on Transformers". μP does not choose batch size or token budget; §3 of this chapter covers those.

### 4.4 When small experiments predict large ones, and when they do not

- DataDecide trained 25 pre-training corpora at 14 sizes (4M–1B) with 3 seeds: ranking corpora with 150M models picked the corpus that wins at 1B in about 80% of pairwise comparisons ([[datadecide]], §3.2, Fig. 3). Continuous likelihood metrics raised small-scale decision accuracy for MMLU, ARC, and code tasks; math tasks stayed near trivial (§3.3–3.4).
- Wortsman et al. fit maximum attention logit against model size on small models and predicted that a 4.8B model would diverge at LR 1e-2 without qk-layernorm; the 4.8B run diverged ([[small-scale-proxies-instabilities]], §3.3, Fig. 9).
- OLMo 2 at 7B scale: higher learning rates had lower loss early, but 3e-4 overtook 6e-4 "well past 200B tokens. A shorter hyperparameter experiment might come to the wrong conclusion" ([[olmo-2]], §4.1).

Implication: a proxy predicts the large run only when the parametrization transfers (§4.3), the proxy is trained long enough to pass crossovers, and the metric separates candidates at small scale.

## §5 Normalization

### 5.1 LayerNorm and RMSNorm

**Formulas** ([[batch-vs-layer-norm]] excerpt).

```
LayerNorm: y = γ ⊙ (x − μ) / √(σ² + ε) + β
RMSNorm:   y = γ ⊙ x / √((1/d) Σ_i x_i² + ε)
```

x is the d-dimensional activation of one token, μ and σ² its mean and variance over the d features, γ and β learned scale and shift, and ε a small constant (OLMo 2 7B config: 1e-6).

**Evidence.** RMSNorm "achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models" (Zhang & Sennrich 2019, Abstract); the range is total model running time in their RNN and Transformer experiments, not a normalization-op speedup. OLMo 2 found "no difference" between non-parametric LayerNorm and RMSNorm in its ablations ([[olmo-2]], §3.3.1). Qwen2.5 uses RMSNorm with pre-normalization ([[qwen-2.5]], §2).

### 5.2 Post-norm versus pre-norm

```
Post-LN: x_{l+1} = LN(x_l + F(x_l))
Pre-LN:  x_{l+1} = x_l + F(LN(x_l)),   with a final LN before the output layer
```

F is the attention or MLP sub-layer.

**Mechanism** (Xiong et al. 2020, [[batch-vs-layer-norm]] excerpt).
1. At initialization, Post-LN keeps the expected squared hidden-state norm at (3/2)d in every layer, while Pre-LN hidden states satisfy (1 + l/2)d ≤ E‖x_l‖² ≤ (1 + 3l/2)d, growing linearly with depth l (Lemma 2).
2. The Jacobian of layer normalization scales as √d / ‖x‖ (Lemma 3), so a larger input norm passes a smaller gradient.
3. The gradient of the last FFN layer is O(d√(ln d)) for Post-LN, independent of depth L, and O(d√(ln d / L)) for Pre-LN (Theorem 1). Measured at initialization, the Post-LN value stayed "around 1.6" while the Pre-LN value decreased as depth grew from 6 to 14 layers (§3.4, Fig. 3).

**Worked example (derived).** At layer l = 24, Pre-LN E‖x‖² lies between 13d and 37d, so ‖x‖ is 3.6 to 6.1 times √d, compared with √1.5 = 1.22 times √d for Post-LN. A layer norm at that depth passes a gradient about 3 to 5 times smaller than a Post-LN layer norm.

**Implication.** Post-LN's large near-output gradients are the authors' explanation for why it needs warmup (§1.1). Pre-LN models add a final layer norm before the output layer (Xiong et al. Table 1; GPT-2 §2.3); Xiong et al. state that the input to this final layer norm scales linearly in L, so the gradients of all parameters are normalized by √L (§3.3). Growth of residual-stream norm with depth is a pre-norm property; Post-LN renormalizes after every add.

### 5.3 OLMo 2's reordered norm and QK-norm

**Definition.** Reordered norm normalizes each sub-layer's output before it is added to the residual stream:

```
OLMo-0424 (pre-norm):  h = x + Attention(LN(x));        h_out = h + MLP(LN(h))
OLMo 2 (reordered):    h = x + RMSNorm(Attention(x));   h_out = h + RMSNorm(MLP(h))
```

The residual stream x, h is never normalized after the add, so this is not post-norm ([[olmo-2]], §2.1, §3.3.2; Liu et al. 2021 proposed it). QK-norm applies RMSNorm to query and key projections before the attention dot product.

**Evidence.** "In isolation, neither of these changes yield good results, but together they improve both the growth and the spikiness of the L2 norm of the gradient"; the gradient spike score fell from 0.108 to 0.069 with both applied ([[olmo-2]], §3.3.2, Fig. 7). Result (single study, one run each; no downstream comparison printed).

### 5.4 QK-norm, z-loss, and ε as controls on logit growth

**Formulas** ([[small-scale-proxies-instabilities]]).

```
attention logit:  z_ij = ⟨q_i, k_j⟩ / √d_h
z-loss:           L_total = L_CE + α · log² Z,    Z = Σ_j exp(y_j)
AdamW update:     Δ = m̂ / (√v̂ + ε)
```

q_i and k_j are query and key vectors, d_h the head dimension, y the output logits, Z the softmax normalizer, α the z-loss coefficient, and ε the AdamW constant.

**Evidence.**
- Attention-logit growth appeared in a 9.4M model at LR 0.1; without qk-layernorm the diverging learning rate fell as models grew; forcing the maximum attention logit to about 1e4 in a 10M model gave loss worse than a zero-layer bigram baseline (§3.1.1, §3.3, Fig. 10). The growth came from larger query and key norms, not higher cosine similarity (Fig. E.1).
- Output-logit divergence (very negative logits late in training) occurred in models without weight decay; z-loss with α = 1e-4 resolved it (§3.1.2, Fig. 3–4).
- With qk-layernorm and z-loss, models up to 1.2B trained to low loss across LR 3e-4 to 3e-1 (Fig. 1). Result (single study, 2.4M–4.8B, C4).
- Qwen3 removed QKV-bias and "introduce[d] QK-Norm ... to ensure stable training" (arXiv:2505.09388 §2, [[batch-vs-layer-norm]] excerpt); Qwen2.5's architecture description lists QKV bias and does not list QK-norm ([[qwen-2.5]], §2). MiniCPM measured lower learning-rate sensitivity with QK-norm but did not adopt it ([[minicpm]], App. A.1).
- ε: gradient RMS fell toward ε = 1e-8 as scale and learning rate grew; for a 4.8B model at LR 0.3, ε = 1e-15 improved loss and ε = 1e-6 diverged ([[small-scale-proxies-instabilities]], §3.4). OLMo 2 lowered ε from 1e-5 to 1e-8, after which "the gradient norm settles much more quickly and remains permanently lower" ([[olmo-2]], §3.4.1).
- OLMo 2's z-loss coefficient is printed as 1e-4 in §3.3.3 and 1e-5 in Table 1 and the released 7B config. Its fused (Flash Attention) and PyTorch z-loss implementations matched in the forward pass but not the backward pass (§3.3.3, Fig. 8).

## §6 Long-context continued pre-training: optimization and normalization choices

Context extension is covered in ch-32b. This section covers the schedule and normalization settings that the extension stage reuses or changes.

- **Learning rate.** ProLong continues Llama-3-8B-Instruct at 64K with peak LR 1e-5, 10% warmup, cosine to 1e-6, and 4M-token batches for 20B tokens ([[prolong-recipe]]). This is 1/30 of the 3e-4 peak that the Llama 3 report prints for 8B pre-training (Table 3, which describes the Llama 3.1 models; derived ratio). For the 512K stage the paper prints 1e-5 and the released script defaults to 5e-6 (conflict, [[prolong-recipe]]). YaRN extends Llama 2 with LR 2e-5, 20 warmup steps, 400 steps, batch 64 ([[yarn]], §4.1). Kimi k1.5's re-warm to 1e-5 at 128K belongs to its SFT stage (32K stage 2e-5 → 2e-6, then 128K stage 1e-5 → 1e-6); its long-context activation pre-training learning rate is not reported ([[kimi-k1-5-recipe]]). Llama 3 405B extends from 8K to 128K in six stages over about 800B tokens, advancing when short-context evaluations "recovered completely" and needle-in-a-haystack retrieval (finding one inserted fact in a long document) is solved; the learning rate for those stages is not reported ([[llama-3]], §3.4.2).
- **Re-warming.** When a run resumes on new data from a decayed checkpoint, re-warming and re-decaying the learning rate increased adaptation, and a higher re-warm peak gave "more forgetting and more adaptation" (405M, Pile → SlimPajama and Pile → German; [[continual-pretraining-rewarm-replay]], §6.1.2). With 5% or 25% replay of old data, re-warmed runs matched re-training on the union (§6.3). Hägele et al. argue that continuing from a pre-cooldown checkpoint at high learning rate avoids re-warming, but test no new-distribution continuation ([[cooldown-scaling-beyond-fixed-durations]], §2).
- **Attention temperature.** YaRN scales attention logits through √(1/t) = 0.1 ln(s) + 1 applied to both queries and keys, where s is the extension factor ([[yarn]], §3.3, Eq. 15). Worked example (derived): s = 8 gives √(1/t) = 1.208, so logits are multiplied by 1/t = 1.459; s = 32 gives 1.347 and 1.813. Llama 4 Scout uses inference-time attention temperature scaling without a published formula ([[llama-4]]).
- **QK-norm and retrieval.** In three 8B variants trained on 750B tokens and evaluated after SFT, the QK-Norm variant was within 0.34 points of RoPE on MMLU (48.21 vs 48.55) but scored lowest on 65K needles (7.93 vs 9.82 for RoPE and 9.03 for NoPE), and its aggregated attention entropy at 128K was 14.14 versus 7.62 for RoPE ([[rope-nope-hybrid-attention]], Table 2, Table 9). The authors attribute this to normalization removing magnitude information from query–key dot products (Interpretation). Result (single study, no seeds). This does not show that QK-norm prevents long-context ability in other architectures, but it makes long-context retrieval an evaluation requirement whenever QK-norm is adopted for stability.

## §7 Over-training and fine-tuning plasticity

**Definition.** Plasticity here means how much a pre-trained checkpoint can be changed by fine-tuning without losing capabilities it already had. Over-training means training a model on more tokens than the compute-optimal count for its size, which Hoffmann et al. put at about 20 tokens per parameter (as cited in [[minicpm]] §4.5).

**Evidence** ([[overtrained-lms-harder-to-finetune]]).
- OLMo-1B intermediate checkpoints instruction-tuned on Anthropic-HH with learning rate tuned per checkpoint: the 3T-token base gave up to 3% lower AlpacaEval response rate and about 2% lower ARC than the 2.3T base, and fell to the level of the 1.5T base (§2.2, Fig. 2). Base models improved monotonically with tokens. Onset was beyond 2.5T tokens (§3.1). OLMo-7B showed no such degradation up to 3T tokens (§3.1). Using the nominal sizes, 3T tokens is about 3,000 tokens per parameter at 1B and about 430 at 7B (derived); the paper does not establish a threshold ratio.
- The authors note a confound: public checkpoints from one run have different final learning rates because of the annealing schedule. Their controlled runs (15M–90M, 4B–128B C4 tokens, each annealed to 0) remove it (§3.2).
- Mechanism ("progressive sensitivity"): for a fixed Gaussian perturbation size or a fixed fine-tuning learning rate, the perplexity increase caused by the modification grows monotonically with pre-training tokens, and larger fine-tuning learning rates reach the degradation point at fewer tokens (§3.3–3.4, Fig. 3–5). A smaller-than-optimal fine-tuning learning rate delays the degradation but lowers in-distribution performance (§3.4.2).

**Conditions and limits.** Result (single study). Scale is 1B–7B for public checkpoints and 15M–90M for controlled runs; RL and pruning are not tested.

**Implication for a general-purpose model.** The pre-training token budget and final learning rate change how much later stages can move the model before prior ability is lost. A pre-training plan that will be followed by SFT and RL should evaluate a post-trained version of candidate checkpoints on held-out tasks, not only the base checkpoints (ch-01 covers update size and retention; ch-32a covers replay).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-stable | peak LR; warmup; decay | 8 × 10^-5; linear 8,000 steps; cosine to 8 × 10^-7 over 1,200,000 steps | arXiv:2407.21783v3 §3.4.1 ([[llama-3-recipe]]) | verified 2026-09-14 | no ablation reported |
| Llama 3.1 405B | 405B | pretrain-stable | batch schedule | 4M tokens at sequence 4,096 → "8M sequences of 8,192 tokens" after 252M tokens → 16M after 2.87T tokens | §3.4.1 | verified 2026-09-14 | smaller early batch "to improve training stability"; no numbers |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | anneal | final 40M tokens, LR linear to 0 at 128K context; high-quality sources upsampled; Polyak averaging (averaging of checkpoints) | §3.4.3 | verified 2026-09-14 | no ablation reported; batch during anneal not reported |
| Llama 3.1 8B / 70B | 8B / 70B | pretrain-stable | peak LR; schedule | 3 × 10^-4 / 1.5 × 10^-4; warmup, decay, batch not printed | Table 3; §3.4 | verified 2026-09-14 (schedule not reported) | no ablation reported |
| Llama 3 scaling-law models | 40M–16B | pretrain-stable | schedule | cosine, warmup 2,000 steps, peak 2 × 10^-4 to 4 × 10^-4 by size, decay to 0.1 of peak; batch 250K–4M (unit not printed) | §3.2.1 | verified 2026-09-14 | n/a |
| Llama 3 8B (experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50%-trained model, LR linear to 0 over 40B tokens, 30% new data / 70% default mix | §3.1.3 | verified 2026-09-14 | method; no numbers printed |
| OLMo 2 7B / 32B | 7B / 32B | pretrain-stable | peak LR | 3.0 × 10^-4 / 6.0 × 10^-4 | arXiv:2501.00656 Table 3; `OLMo2-7B-stage1.yaml` `learning_rate: 3.0e-4` ([[olmo-2]] excerpt) | verified 2026-09-15 | §4.1 Table 8 (7B scale): 3e-4 vs 6e-4 within 0.1 OLMES after 2T + 100B anneal |
| OLMo 2 13B | 13B | pretrain-stable | peak LR | 9.0 × 10^-4 | Table 3; §4.1 ("higher peak learning rate from the start") | conflict (stage-2 config starts at 9e-5, the 10% floor of a 9e-4 cosine at its 5T horizon; derived support for this row) | no ablation reported |
| OLMo 2 13B | 13B | pretrain-stable | peak LR | 3.0 × 10^-4 | `OLMo2-13B-stage1.yaml` `learning_rate: 3.0e-4` | conflict | no ablation reported |
| OLMo 2 7B / 13B / 32B | 7B / 13B / 32B | pretrain-stable | warmup; decay; horizon; truncation | 2,000 steps; cosine to 10% of peak; 5T / 5T / 6.5T tokens; after 4T / n/a / after 6T | Table 3, §2.3 | verified 2026-09-15 | §4.1: truncation plus linear decay "with little loss of performance" (OLMo-0424 experience, no numbers) |
| OLMo 2 7B / 13B / 32B | 7B / 13B / 32B | pretrain-stable | global batch | 1024 / 2048 / 2048 sequences × 4096 tokens (4.19M / 8.39M / 8.39M tokens, derived) | Table 3 | verified 2026-09-15; tokens derived | no ablation reported |
| OLMo 2 7B | 7B | pretrain-stable | optimizer; clipping; init | AdamW β (0.9, 0.95), wd 0.1, no wd on embeddings, ε 1e-8; clip 1.0; N(0, 0.02) truncated at 3σ | `configs/official-1124/OLMo2-7B-stage1.yaml`; §3.2, §3.4 | verified 2026-09-15 | Fig. 4: spike score 0.40 → 0.03; Fig. 9 (ε); Fig. 10: 0.16 vs 0.092 (embedding wd) |
| OLMo 2 7B | 7B | pretrain-stable | z-loss coefficient | 10^-4 | §3.3.3 | conflict | no ablation reported |
| OLMo 2 7B | 7B | pretrain-stable | z-loss coefficient | 10^-5 | Table 1; `OLMo2-7B-stage1.yaml` `auxiliary_loss_multiplier: 1e-5` | conflict (released config produced the checkpoint) | no ablation reported |
| OLMo 2 7B | 7B | mid-train | LR; tokens; merge | starts at 6.1499 × 10^-5 from step 928,646, linear to 0 over 50B tokens; 3 data orders averaged | `OLMo2-7B-stage2-seed42.yaml`; §2.3 | verified 2026-09-15 | Table 11: LR decay alone OLMES 69.6 → 74.0; Table 9 |
| OLMo 2 13B, 32B | 13B, 32B | mid-train | tokens; merge | 3 runs of 100B + 1 run of 300B tokens, averaged | §2.3, Table 9 caption | verified 2026-09-15 | Table 9: 13B average 58.9 → 68.3 |
| DeepSeek LLM 7B / 67B | 7B / 67B | pretrain-stable | peak LR; batch | 4.2 × 10^-4 / 3.2 × 10^-4; 2304 / 4608 sequences × 4096 | arXiv:2401.02954v1 Table 2 ([[deepseek-llm]] excerpt) | verified 2026-09-15 | Eq. 1 fit on 1e17–2e19 FLOPs; derived check matches Table 2 |
| DeepSeek LLM 7B, 67B | 7B, 67B | pretrain-stable | schedule; optimizer; init | warmup 2,000 steps; 31.6% of max after 80% of tokens, 10% after 90%; AdamW (0.9, 0.95), wd 0.1, clip 1.0; init std 0.006 | §2.3 | verified 2026-09-15 | Fig. 1a (1.6B, 100B tokens): "essentially consistent" with cosine |
| MiniCPM-2.4B | 2.4B | pretrain-stable | schedule; batch; max LR | WSD; "batch size of 3.93 million" (unit not printed; Table 2: 4M); 0.01 (μP base LR) | arXiv:2404.06395v3 §6.2, Table 2 ([[minicpm]] excerpt) | verified 2026-09-15 | §3.3 Fig. 3: optimal base LR about 0.01 from 0.04B to 0.5B, 2.1B check |
| MiniCPM-2.4B | 2.4B | pretrain-decay/anneal | decay shape; length; data | exponential 0.5^((s−S)/T), T = 5000 steps (20B tokens); pre-training data + high-quality SFT data | §6.2 | verified 2026-09-15 | Table 1: A-2 vs A-1 (MMLU 50.9 vs 44.6) |
| MiniCPM (μP search) | 0.009B proxy | pretrain-stable | μP settings | scale_depth 1.4, scale_emb 12, init_std 0.1, lr 0.01 | App. A.1 | verified 2026-09-15 | Bayesian search, Fig. 14 |
| Hägele et al. 1B (FineWeb, 100B tokens) | 1B | pretrain-decay/anneal | peak LR; warmup; cooldown | 8e-4; 2,000 steps; (1-sqrt) 20% or linear 20% | arXiv:2405.18392v3 Table 3, Table 4 ([[cooldown-scaling-beyond-fixed-durations-recipe]]) | verified 2026-09-14 | Table 4: 46.23 / 46.20 vs cosine-to-10% 46.26 |
| Porian et al. 901M | 901M | pretrain-stable | LR; batch (2048-token sequences); β₂ | 0.0024; 640; 0.95 | arXiv:2406.19146v4 Table 4 ([[resolving-scaling-discrepancies-recipe]]) | verified 2026-09-14 | App. G.5 Table 7: lowest loss 2.943 of 13 configurations |
| Wortsman et al. default | 2.4M–1.2B | pretrain-stable | warmup; qk-layernorm; z-loss; ε | 5e3 of 1e5 steps; on (per head); 1e-4; 1e-8 | arXiv:2309.14322v2 §2.1 ([[small-scale-proxies-instabilities]]) | verified 2026-09-14 | Fig. 5 (warmup), Fig. 1 (qk-layernorm), Fig. 3 (z-loss) |
| ProLong-64k-Base | 8B | long-context | LR; warmup; decay; batch; RoPE base | 1e-5; 10%; cosine to 1e-6; 4M tokens; 8 × 10^6 | arXiv:2410.02660v4 Table 9; train_64K.sh ([[prolong-recipe]]) | verified 2026-09-14 | RoPE base: App. B.1 Table 18 (54.6 vs 48.7 vs 29.1); LR: no ablation reported |
| ProLong-512k-Base | 8B | long-context | peak LR | 1e-5 ("each stage") | v4 Table 9 | conflict | no ablation reported |
| ProLong-512k-Base | 8B | long-context | peak LR | 5e-6 (script default) | train_512K.sh `lr=${LR:-5e-6}` | conflict | no ablation reported |
| Yarn-Llama-2 s = 16 | 7B, 13B | long-context | LR; warmup; steps; batch | 2 × 10^-5; 20 steps linear; 400; 64 | arXiv:2309.00071v3 §4.1 ([[yarn]]) | verified 2026-09-14 | App. B.2 Table 6: 400 YaRN steps ≈ 1000 PI steps |
| Kimi k1.5 | not reported | SFT | LR by length stage | 32K: 2e-5 → 2e-6; 128K: re-warm to 1e-5 → 1e-6 | arXiv:2501.12599v4 §2.5.2 ([[kimi-k1-5-recipe]]) | verified 2026-09-14 | no ablation reported |
| Kimi k1.5 base | not reported | long-context | LR | not reported (§2.5.1, App. B checked) | — | not reported | — |
| Continual pre-training, 405M | 405M | mid-train | re-warm peak; decay; warmup; replay | 3 × 10^-4 (pre-training peak); cosine to 3 × 10^-5; 1% of iterations; 5% (Pile → SlimPajama), 25% (Pile → German) | arXiv:2403.08763v4 §6.1.2, §6.3 ([[continual-pretraining-rewarm-replay]] excerpt) | verified 2026-09-15 | Fig. 4: higher re-warm peak gives more adaptation and more forgetting |

**Starting point for a small general-purpose run.** For a decoder-only model near 1B parameters trained on about 100B web tokens, the verified rows support: AdamW with β (0.9, 0.95), gradient clipping 1.0, a 2,000-step linear warmup, a peak learning rate from a fitted rule rather than copied from another size (Hägele et al. used 8e-4 estimated from the DeepSeek LLM laws for their 1B/100B run; Porian et al. prescribe 0.0024 with batch 640 × 2048 for 901M at about 14B tokens), and a constant learning rate with a 20% (1-sqrt) cooldown, which matched cosine downstream in that 1B setting; in their 210M runs the best constant learning rate for a cooldown schedule was about half the best cosine peak (Fig. 3 caption). Wortsman et al.'s qk-layernorm and z-loss 1e-4 widened the stable learning-rate range at 2.4M–1.2B on C4. OLMo 2's N(0, 0.02) truncated init, AdamW ε 1e-8, no weight decay on embeddings, and reordered norm with QK-norm were chosen by gradient-spike measurements at 7B on 4096-token sequences. None of these values was tested on the reader's data, and the rows from different sources were not tested together.

## Generalization lens

**(a) What increases breadth.**
- Completing the learning-rate decay: OLMo 2 7B's anneal on the unchanged mixture raised OLMES 69.6 → 74.0 and MMLU 59.8 → 61.8 ([[olmo-2]], Table 11); cooldown and cosine gave equal 8-task averages at 1B ([[cooldown-scaling-beyond-fixed-durations]], Table 4).
- Broad high-quality data in the decay, evaluated on held-out tasks: OLMo 2 7B held-out MMLU-Pro 27.4 → 31.0 and TriviaQA 74.6 → 78.0 ([[olmo-2]], Table 9).
- Settings that make larger runs possible at the chosen learning rate: QK-norm and z-loss across three orders of magnitude of learning rate ([[small-scale-proxies-instabilities]]); reordered norm, QK-norm, and init changes that reduced gradient spikes in OLMo 2 ([[olmo-2]], §3).

**(b) What causes narrowing or forgetting.**
- Annealing on benchmark-adjacent data: Llama 3 8B gained 24.0% on GSM8K validation from annealing on GSM8K and MATH training sets, while 405B gained a negligible amount ([[llama-3]], §3.1.3); MiniCPM-1.2B annealing data raised MATH and MMLU but lowered GSM8K from 34.2 to 31.8 at equal SFT tokens ([[minicpm]], Table 1).
- Extended pre-training followed by fine-tuning: OLMo-1B at 3T tokens was over 2% worse after instruction tuning than at 2.3T ([[overtrained-lms-harder-to-finetune]]).
- Re-warming on new data: a higher re-warm peak increased forgetting of the original distribution ([[continual-pretraining-rewarm-replay]], §6.1.2).
- QK-norm with long contexts: lowest 65K needle score among three 8B variants, with MMLU within 0.34 points of the RoPE variant ([[rope-nope-hybrid-attention]]).

**(c) How to measure it at this stage.**
- Run a control decay on the unchanged mixture for every data-change decay ([[olmo-2]], Table 11 design).
- Declare development and held-out suites before experiments; OLMo 2 kept a held-out suite "not used for model development decisions" ([[olmo-2]], §2.5).
- Measure hyperparameter choices with a learning-rate sensitivity curve over at least three orders of magnitude, not a single learning rate ([[small-scale-proxies-instabilities]], §2.2).
- For small-scale data and recipe decisions, use continuous likelihood metrics on tasks that separate candidates at small scale and check decision accuracy against a larger run ([[datadecide]]).
- For checkpoints intended for post-training, evaluate a fine-tuned version on out-of-distribution tasks, because base-model scores improved monotonically while fine-tuned scores fell ([[overtrained-lms-harder-to-finetune]], §2.2).
- Known measurement errors: single-seed ablations (OLMo 2, MiniCPM tables); partial held-out contamination (OLMo 2 GSM8K); confounded final learning rates when comparing checkpoints from one run ([[overtrained-lms-harder-to-finetune]], §3.2).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Cosine horizon set to a planned budget that is not trained | Training stops while η is still above η_min; loss at the stop is above what a completed decay reaches | Log η at the final step; compare with a short decay branched from the same checkpoint ([[minicpm]] §4.3 design) |
| Crediting a data change for a gain that the decay produced | Anneal gains on all suites, including suites unrelated to the new data | Run the same decay on the unchanged mixture (OLMo 2 Table 11 PT Mix row) |
| Reporting annealing gains only on development benchmarks | Large gain on the targeted benchmark, no held-out result | Report a held-out suite declared before the experiment; check overlap of annealing data with evaluation sets ([[llama-3]] §3.1.3) |
| Scaling learning rate linearly with batch size far above the noise scale | Divergence or no speedup after a batch increase | Estimate B_simple from local and global gradient norms ([[large-batch-training-noise-scale]] App. A.1); compare steps-to-target at two batch sizes |
| Batch above the critical batch size for the tokens trained | More tokens needed for the same loss; the steps saved are fewer than the batch increase | Plot steps to a target loss against batch size; find the 20%-overhead point ([[critical-batch-size-pretraining]] §3.1) |
| Copying a learning rate across widths without μP or a fitted rule | Best learning rate on the proxy diverges or underperforms at the target width | Sweep learning rate at two widths; under μP the optimum should not shift ([[weight-init]] excerpt, Tensor Programs V Fig. 1) |
| Proxy runs too short to pass learning-rate crossovers | The learning-rate ranking reverses later in the large run | Extend at least two candidates past the observed crossover; OLMo 2's came after 200B tokens ([[olmo-2]] §4.1) |
| Describing reordered norm as post-norm, or dropping the final norm in a pre-norm model | Final hidden-state norm grows with depth and output logits grow, or the implementation normalizes the stream after each add | Print the block equations from code; log per-layer residual norms at initialization ([[batch-vs-layer-norm]] excerpt) |
| No attention-logit or output-logit monitoring at high learning rate | Slow divergence late in training; very negative output logit mean | Log max attention logit per layer and output logit mean; every run with max attention logit above 1e4 diverged in [[small-scale-proxies-instabilities]] Fig. 9 |
| AdamW ε close to gradient RMS | Update size shrinks as scale grows; gradient RMS collapses | Log gradient RMS per layer against ε ([[small-scale-proxies-instabilities]] §3.4) |
| Switching fused and manual z-loss implementations mid-run | z-loss curves diverge after the switch while cross-entropy looks unchanged | Keep one implementation per run; compare backward values on one batch ([[olmo-2]] §3.3.3) |
| Adopting QK-norm without long-context evaluation | Short-task scores unchanged, needle and retrieval scores lower | Evaluate retrieval at the target length after SFT; log attention entropy ([[rope-nope-hybrid-attention]]) |
| Choosing a pre-training checkpoint by base-model score only | Later fine-tuned model worse on held-out tasks than one from an earlier checkpoint | Fine-tune two or more candidate checkpoints with tuned LR and compare OOD scores ([[overtrained-lms-harder-to-finetune]]) |

## Check your understanding

1. In OLMo 2's Table 11, the anneal on the unchanged mixture raised MMLU by 2.0 points but lowered GSM* by 1.5. Explain what this row lets you conclude about the 19.5-point GSM* difference between the final mixture (46.5) and this control (27.0), and what it does not let you conclude about held-out math tasks.
2. Using ε_opt(B) = ε_max / (1 + B_noise/B), explain why a learning rate that was tuned at batch 256 can be close to linearly scalable to batch 512 in one run and far from linear in another run of the same model. What quantity differs between the two runs?
3. Zhang et al. find the critical batch size depends mostly on tokens trained. Explain why this implies that a batch-size schedule that grows during training can be justified for a fixed model, and what measurement would test whether Llama 3's batch increase at 2.87T tokens was below the critical batch size.
4. At step 1 AdamW's update is approximately −η · sign(g). Explain how this, together with Xiong et al.'s Theorem 1, accounts for Post-LN needing warmup while Pre-LN trained without it in their experiments. Which part of your explanation is derivation and which is interpretation?
5. OLMo 2's reordered norm normalizes sub-layer outputs but not the residual stream. Explain why this differs from post-norm in its effect on residual-stream norm across depth, and why the paper's result (0.108 → 0.069 spike score only with QK-norm also applied) does not show which of the two changes matters more.
6. A μP proxy at width 256 finds Adam LR 0.01 for hidden matrices. Derive the hidden-matrix learning rate at width 4096 under Tensor Programs V Table 3, and explain why weight decay found on the proxy may not transfer.
7. Springer et al. compare OLMo-1B checkpoints at 2.3T and 3T tokens from one training run. Explain why the learning rate at each checkpoint confounds this comparison, and how their controlled experiments remove the confound.
8. A team adds QK-norm to fix high-learning-rate divergence and plans to extend context to 128K. Using Wortsman et al. and the Cohere 8B results, explain what each normalization choice changes in the attention logits, and design the smallest evaluation that would detect a long-context regression before extension.

## Connections

- Depends on: ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch (AdamW ε, z-loss implementation differences, and precision effects on logits).
- Previous in order: ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch.
- Next in order: ch-04 — Sequence Packing, Loss Masking, and Chat Templates.
- ch-00 — What General Capability Means and How It Is Measured (development versus held-out suites used in §2 and the Generalization lens).
- ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability (Adam moments and update size behind §1.1 and §7).
- ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen (how parallelism sets the batch sizes of §3).
- ch-07 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures (spikes and divergence diagnosed in §5).
- ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability (DeepSeek LLM, Porian et al., and cooldown-based scaling sweeps).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (data choices for the decay phase in §2).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (§6 re-warming and replay).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (RoPE base, PI, YaRN, and data for §6).
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate (fine-tuning learning rates after the plasticity results of §7).

## Sources

- [[lr-schedules]] — chapter excerpt: inverse square root (Vaswani §5.3), WSD definition (MiniCPM Eq. 1), multi-step (DeepSeek LLM §2.3), truncated cosine (OLMo 2 configs), Llama 3 405B schedule; list of unsupported numbers removed.
- [[minicpm]] — chapter excerpt: WSD, decay length, cosine horizon, annealing with SFT data (Table 1), μP operations and search results, batch-size fit.
- [[cooldown-scaling-beyond-fixed-durations]], [[cooldown-scaling-beyond-fixed-durations-recipe]] — (1-sqrt) cooldown, cooldown length, 1B downstream comparison, re-warming argument.
- [[deepseek-llm]] — chapter excerpt: multi-step schedule, Table 2 settings, η_opt and B_opt fits with a derived check.
- [[large-batch-training-noise-scale]] — noise scale, steps–examples trade-off, ε_opt(B), Adam learning-rate exponent.
- [[critical-batch-size-pretraining]] — chapter excerpt: 20%-overhead definition and B* fits against model and data size.
- [[resolving-scaling-discrepancies]], [[resolving-scaling-discrepancies-recipe]] — fitted batch-size and learning-rate laws, β₂ at small batch, 901M prescription.
- [[weight-init]] — chapter excerpt: Glorot and He variances, GPT-2 residual scaling, OLMo 2 init ablation, μP Table 1 and Table 3, GPT-3 6.7B μTransfer with confounds.
- [[batch-vs-layer-norm]] — chapter excerpt: LayerNorm and RMSNorm, RMSNorm running-time claim, Xiong et al. Lemma 2 and Theorem 1, warmup experiment, OLMo 2 norm placement, Qwen3 QK-Norm statement.
- [[olmo-2]] — chapter excerpt: stability ablations (spike scores), Table 3 schedule and configs, z-loss conflict, learning-rate crossover and Table 8, mid-training Tables 9 and 11.
- [[small-scale-proxies-instabilities]] — attention-logit growth, output-logit divergence, qk-layernorm, z-loss, warmup, μP, ε, and the 4.8B divergence forecast.
- [[llama-3]], [[llama-3-recipe]] — 405B schedule and batch ramp, annealing, annealing on GSM8K/MATH at 8B vs 405B, anneal-as-data-evaluation, long-context stages.
- [[datadecide]] — decision accuracy of small-scale rankings for pre-training data choices.
- [[overtrained-lms-harder-to-finetune]] — chapter excerpt: catastrophic overtraining in OLMo-1B, progressive sensitivity, learning-rate dependence.
- [[continual-pretraining-rewarm-replay]] — chapter excerpt: re-warming peak versus adaptation and forgetting; replay fractions.
- [[prolong]], [[prolong-recipe]] — long-context continued-training learning rate, batch, RoPE base, and the 512K learning-rate conflict.
- [[kimi-k1-5-recipe]] — SFT learning-rate re-warm at 128K; long-context pre-training learning rate not reported.
- [[yarn]] — extension fine-tuning settings and the attention temperature formula.
- [[rope-nope-hybrid-attention]] — chapter excerpt: QK-Norm variant's needle scores and attention entropy at 8B.
- [[llama-4]] — inference-time attention temperature scaling in Scout (no formula published).
- [[qwen-2.5]] — QKV bias with pre-norm RMSNorm, for comparison with Qwen3's QK-Norm.
