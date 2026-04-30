<!-- chapter: ch-51
     track: eval
     kind: content
     title: Metric Noise, Confidence, Go/No-Go
     deps: [ch-50]
     sources: [[scaling-laws-data-quality]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[judge-llm-bias]], [[karpathy-training-neural-net-recipe]], [[faithful-synth-eval]], [[interplay-pretraining-midtraining-rl]], [[reward-model-overoptimization]]
     figures: figures/bootstrap-ci.html
-->

# Chapter 51 — Metric Noise, Confidence, Go/No-Go

> **Core insight.** The hardest eval question in 2025 is not "what number did we get?" but "is this number distinguishable from yesterday's?" Modern post-training pipelines ([[llama-3]] six rounds; [[olmo-2]] SFT → DPO → RLVR; [[olmo-3]] multi-branch model flow) publish single-digit-percentage-point gains per stage. A 1.2 pp lift on a 250-item eval set is within noise *almost always* and within noise of *every realistic bootstrap width*. Teams that ship without a CI, a paired comparison, and a written go/no-go memo end up masking real regressions under rolling averages ([[reward-model-overoptimization]]) or chasing judge-bias artifacts ([[judge-llm-bias]]).
>
> **Guideline.** Treat every headline metric as a point estimate that needs three pieces of scaffolding: (1) a bounded account of per-run variance (decode temperature, prompt permutation, dropout during SFT eval, judge order); (2) a bootstrap CI at the *item* level with N explicitly stated; (3) a paired comparison (paired bootstrap, sign test, or Wilcoxon) when you claim checkpoint A beats checkpoint B. The go/no-go memo closes the loop: one claim, the evidence for it, and the specific check a reviewer is expected to re-run.

---

## §1 Where the noise comes from

[[karpathy-training-neural-net-recipe]] frames it right: training fails silently. Eval fails silently too, and in the same way — a green curve is not a correct curve. Before computing any CI, budget the four variance sources separately. The magnitudes below are illustrative ranges for 7B–70B instruct models on chat/reasoning benchmarks; swap your own numbers in once you measure them.

| Source | Typical σ on accuracy (pp) | How to bound | Attested reference |
|---|---|---|---|
| **Decode temperature / sampling seed** | T=0 → 0; T=0.7 → 0.5–1.5; T=1.0 → 1–3 | Fix seed; run K≥3 samples per prompt and report majority/mean; or set T=0 for "capability" evals | [[olmo-2]] RLVR protocol uses T=1.0 at training, T=0 at eval |
| **Prompt permutation (few-shot order, option order)** | 0.5–2.5 on MMLU-style MCQ | Randomize per item; average over M≥3 permutations; for MCQ rotate option positions | [[judge-llm-bias]] position bias flips 20–30% of judgments on pair-rank tasks |
| **Dropout / stochastic layers during eval** | 0.0 when `.eval()` is set; 0.2–1.0 if forgotten | Assert `model.eval()`; check no BN/dropout in inference path | [[karpathy-training-neural-net-recipe]] "be paranoid about `.train()` vs `.eval()`" |
| **Judge variance (LLM-as-judge)** | 1–4 pp on pairwise win-rate at N=200; higher on subjective tasks | Swap A/B and average; pool ≥2 judges; reference-guided grading | [[judge-llm-bias]] §Technical Details — swap-and-average, reference-guided +10 pp agreement |
| **Item-count Monte Carlo (the CI you compute)** | ≈ sqrt(p·(1-p)/N) — at p=0.5, N=200 → 3.5 pp | Bootstrap (§3); increase N | Core statistical bound, attested in every eval harness README |

The first four bound the noise *per run*. The last bound is the noise *per evaluation*, which is what a CI quantifies. If per-run σ exceeds item-count σ, you are running too few seeds. If item-count σ exceeds per-run σ, you are running too few items. A good setup has them within 2× of each other.

Related: [[faithful-synth-eval]]'s "average PPL hides tail degradation" argument applies directly — per-slice CIs must be computed and reported, not only the aggregate. A +0.3 pp aggregate lift that masks a -2 pp coding regression is exactly the failure [[reward-model-overoptimization]] predicts when gold reward has already peaked on one slice.

---

## §2 The honest noise budget

Before you run any eval, write down:

```
noise_budget.txt
----------------
Eval: MATH500 pass@1
Seeds per prompt (K): 3
Prompt permutations (M): 1  (non-MCQ)
Decode: T=0.0, top_p=1.0   (greedy => decode σ = 0)
Judge: rule-based verifier (SymPy), σ_judge ≈ 0
Items (N): 500
Expected per-run σ (decode+perm+judge): ~0.3 pp   (greedy + verifier)
Expected CI halfwidth at p≈0.5 (bootstrap, §3): ~4.4 pp at 95%
=> item-count noise dominates; add items OR accept this resolution
```

This is the [[karpathy-training-neural-net-recipe]] "predict-outcome-before-run" rule applied to evaluation. If your claimed gain is smaller than the CI halfwidth, you cannot ship it.

---

## §3 Bootstrap CIs at the item level

Bootstrap is the right tool because it makes no distributional assumption on per-item scores. It works equally for binary (MCQ correct/wrong), real-valued (BLEU, pass-rate-per-item, reward score), or rank (pairwise win).

**Setup.** Eval set of N items. For each item i, the model produces a score s_i (binary or real). The statistic of interest is the mean θ̂ = (1/N) Σ s_i.

**Bootstrap procedure.** For b = 1..B (B ≥ 1000, 10000 preferred):
1. Sample N indices with replacement from {1..N}.
2. Compute θ̂_b = mean of s_i at sampled indices.

The resulting {θ̂_1, ..., θ̂_B} is the bootstrap distribution of θ̂.

**Percentile CI (simplest).** The 95% CI is [θ̂_{(0.025·B)}, θ̂_{(0.975·B)}] — the 2.5th and 97.5th percentiles of the bootstrap distribution.

**BCa CI (bias-corrected accelerated).** The percentile CI is biased when θ̂ is near 0 or 1 (common on saturating benchmarks). BCa corrects:
1. Bias factor z₀ = Φ⁻¹( (# {θ̂_b < θ̂}) / B ).
2. Acceleration â from the jackknife: â = Σ(θ̂_(·) − θ̂_(-i))³ / [6 · (Σ(θ̂_(·) − θ̂_(-i))²)^(3/2)], where θ̂_(-i) is the leave-one-out estimate.
3. Adjusted percentiles α₁ = Φ(z₀ + (z₀ + z_{α/2}) / (1 − â·(z₀ + z_{α/2}))) and α₂ = Φ(z₀ + (z₀ + z_{1-α/2}) / (1 − â·(z₀ + z_{1-α/2}))).
4. CI = [θ̂_{(α₁·B)}, θ̂_{(α₂·B)}].

Use percentile when θ̂ ∈ [0.2, 0.8] and N ≥ 200; BCa when θ̂ is near the boundary or N < 200.

**Rough halfwidth intuition.** For a binary metric with p near 0.5 and large N, the 95% percentile-bootstrap halfwidth is ≈ 1.96·sqrt(p·(1-p)/N). At N=100 → 9.8 pp; N=500 → 4.4 pp; N=2000 → 2.2 pp. You need ~2000 items to resolve a 2 pp lift. Most published benchmarks (HumanEval 164, GSM8K test 1319, MATH500 500) sit in the 4–9 pp halfwidth range — which is why [[olmo-2]]'s single-digit-pp RLVR gains are borderline.

---

## §4 Comparing two checkpoints

Subtracting two CIs is *wrong* when the checkpoints were evaluated on the same items. It throws away the pairing and inflates the variance estimate. Use paired comparisons.

### Paired bootstrap (the right default)

```
# paired_bootstrap(s_A, s_B, B=10000, alpha=0.05)
#   s_A, s_B: arrays of N per-item scores for checkpoints A and B
#   Returns: (delta_hat, ci_lo, ci_hi, p_two_sided)
assert len(s_A) == len(s_B) == N
d = s_A - s_B                              # per-item paired deltas
delta_hat = d.mean()
deltas = []
for b in range(B):
    idx = random_choice(range(N), size=N, replace=True)
    deltas.append(d[idx].mean())           # resample PAIRS, not rows
deltas.sort()
ci_lo, ci_hi = deltas[int(0.025*B)], deltas[int(0.975*B)]
# two-sided p-value via the "achieved significance level" construction:
p_two_sided = 2 * min(mean(deltas >= 0), mean(deltas <= 0))
return delta_hat, ci_lo, ci_hi, p_two_sided
```

Key move: resample *indices*, keep the pair (s_A[i], s_B[i]) together. Per-item variance cancels; only the delta's variance remains. Halfwidth is usually 1.5–3× tighter than the unpaired difference of two independent CIs.

### Sign test (distribution-free, use when you do not trust score magnitudes)

Given paired per-item outcomes, let W = #{i : s_A[i] > s_B[i]}, L = #{i : s_A[i] < s_B[i]}, T = #{i : tie}. Drop ties. Under H₀ (no difference), W ~ Binomial(n=W+L, p=0.5).

Two-sided p-value: `p = 2 · min( P(X ≤ min(W,L)), P(X ≥ max(W,L)) )` where X ~ Binomial(W+L, 0.5). Normal approximation for W+L ≥ 25: `z = (|W − L| − 1) / sqrt(W + L)`, `p = 2·(1 − Φ(z))`.

Reject at α=0.05 when p < 0.05. The sign test ignores magnitudes — it is robust when judges might be miscalibrated per item but the *direction* of the preference is trustworthy. This matches [[judge-llm-bias]]'s finding that LLM judges agree with humans on *direction* ~80% of the time but calibration drifts.

### The "two-run minimum" rule

A single seed on checkpoint A vs a single seed on checkpoint B can fake a 1–3 pp delta from decode noise alone. Run both checkpoints at ≥2 seeds; report (Δ̂, CI) for each seed pair; require consistent sign across seed pairs before claiming an improvement. [[llama-3]]'s six-round loop depends on this rule implicitly — each round's preference data is regenerated from the current-best checkpoint, and a spurious "best" from one decode seed propagates forward. Two runs, consistent sign, paired CI that excludes 0: now you can ship.

---

## §5 Rolling averages vs masking regressions

Training curves from [[olmo-3]]'s 1000+ intermediate checkpoints or [[llama-3]]'s per-round evals are noisy. Two opposing failure modes:

1. **No smoothing.** You see a 2 pp dip at step 3200 and pause training. In fact it was decode-seed noise and the next checkpoint at step 3300 recovers fully. Wasted a day.
2. **Over-smoothing.** You report a 7-checkpoint rolling mean and miss a real regression where the last three checkpoints all dropped 1.5 pp — exactly the [[reward-model-overoptimization]] gold-reward-peaks-then-falls shape. Shipped a worse model.

The decision is not "always smooth" or "never smooth"; it is a decision tree.

```
Is the per-checkpoint CI wider than the claimed effect?
├── YES → smooth across K checkpoints until CI halfwidth < effect/2.
│         Also report the unsmoothed last-K trend separately so regressions are visible.
└── NO  → do not smooth; per-checkpoint CI already resolves the signal.

Is the curve monotone up in a rolling window of W=3 checkpoints?
├── YES → report rolling mean.
└── NO  → switch to showing last K points individually + per-point CIs.
        NEVER report a rolling mean across a trend reversal; it masks the reversal.

Did gold/held-out reward PEAK and fall?  ([[reward-model-overoptimization]])
├── YES → rolling mean is lying. Show raw curve. Stop training.
└── NO  → rolling mean with W=3–5 is safe.
```

The guideline: smooth for communication, never for decision. The raw curve is the decision input.

---

## §6 The go/no-go memo — template

The memo is what you hand to a reviewer, not what you keep in your notebook. It is one page, five sections, and it is falsifiable.

```
go-no-go-memo.md
----------------

## 1. Claim
Checkpoint `ckpt-round-3` improves over `ckpt-round-2` on `capability-bundle-A`
(avg of MATH500, GSM8K, HumanEval, IFEval) by +1.8 pp, and does not regress
by more than 0.5 pp on any safety slice in `safety-bundle-B`.

## 2. Evidence
- **Item-level bootstrap CIs** (B=10000, percentile method, N per task logged below):
  | Eval       | N    | θ̂(A) | 95% CI       | θ̂(B) | 95% CI       | Δ paired | CI paired  |
  | MATH500    | 500  | 52.4 | [47.9, 56.6] | 50.2 | [45.8, 54.6] | +2.2     | [+0.4,+4.0]|
  | GSM8K      | 1319 | 86.1 | [84.2, 87.9] | 85.3 | [83.3, 87.1] | +0.8     | [-0.2,+1.8]|
  | HumanEval  | 164  | 71.9 | [65.2, 78.0] | 69.5 | [62.8, 75.6] | +2.4     | [-0.6,+5.4]|
  | IFEval     | 541  | 78.3 | [74.7, 81.7] | 76.5 | [72.8, 80.0] | +1.8     | [+0.3,+3.3]|
- **Paired-bootstrap + sign test agree** on MATH500 and IFEval (p < 0.05).
- **Two-run minimum:** both seeds (s=7, s=19) show consistent sign on all four tasks.
- **Safety slice (safety-bundle-B):** max regression = -0.3 pp on one sub-slice;
  paired CI = [-1.1, +0.5], includes 0. No regression detected.
- **Judge-bias check:** position-swap parity on IFEval-judge subset = 94% (≥90% threshold).

## 3. Variance accounting
- Decode: T=0, greedy; decode σ = 0.
- Prompt permutation: MCQ tasks use 3 permutations averaged; σ_perm ≈ 0.4 pp.
- Judge: IFEval uses rule-based grader (σ_judge ≈ 0); no LLM judge.
- Item-count CI halfwidth (per table above) dominates. OK.

## 4. Regressions considered and dismissed
- MMLU-Pro: -0.4 pp, paired CI [-1.6, +0.8]. Within noise; no action.
- XSTest safety-borderline: -0.2 pp, within CI. Flagged for re-eval at next round.

## 5. What the reviewer is expected to check
- Rerun `eval/run.py --ckpt=round-3 --seed=7 --tasks=math500,ifeval` and confirm
  pass rates are within ±1 pp of table.
- Confirm `data/paired_items.jsonl` has identical item IDs for both checkpoints.
- Confirm `metrics.jsonl` logs `ckpt_hash` per row; diff against claim.

## 6. Decision
GO — merge `ckpt-round-3` into `main/post-training/round-3`.
Trigger: all paired CIs exclude 0 where claimed; no safety regression > 0.5 pp.
```

Rules of the memo. Claim is singular, falsifiable, and quantitative. Evidence is item-level, paired, and reproducible from the commit SHA. Variance is budgeted explicitly, not assumed. Regressions are not hidden — they are enumerated and dismissed with CIs. The reviewer's check list is concrete: specific files, specific commands, specific tolerances.

---

## §7 Common failure modes on this chapter's topic

**a) Reporting a single-seed delta as a "result."** Fix: two-run minimum; paired bootstrap on consistent seeds.

**b) Subtracting two unpaired CIs to claim significance.** The resulting "CI" is wider than the paired CI, often by 2×, and may not exclude 0 when the paired one does. Fix: paired bootstrap with matched item IDs.

**c) Ignoring judge variance in LLM-as-judge evals.** [[judge-llm-bias]] shows 20–30% flip rate under position swap on some models. Fix: swap-and-average; pool judges; report judge-agreement rate alongside win rate.

**d) Over-smoothing a rolling average across a trend reversal.** [[reward-model-overoptimization]]'s peak-then-fall shape becomes invisible at W=7. Fix: the decision tree in §5.

**e) Bootstrap at the aggregate mean instead of at the item level.** Resampling scores instead of items throws away the item correlation across tasks. Fix: resample item indices; keep pairs.

**f) Claiming a gain smaller than CI halfwidth.** [[karpathy-training-neural-net-recipe]]'s "predict-before-run" rule: if the predicted effect is smaller than the predicted CI, you are not running the eval; you are running a lottery.

---

## §8 Connections

- **ch-50** — slice analysis is the upstream step; this chapter adds CIs to each slice.
- **ch-52** — safety eval inherits the same CI machinery; adversarial prompts need stratified bootstrap.
- **ch-53** — the lab where you wire up the harness and produce the first real go/no-go memo.
- **[[llama-3]]** — iterative rounds as the setting where paired comparisons matter every round.
- **[[olmo-2]]**, **[[olmo-3]]** — per-stage gains, publicly reported; the CI regime this chapter targets.
- **[[judge-llm-bias]]** — the judge-variance floor that bootstrap must include.
- **[[reward-model-overoptimization]]** — motivates "never smooth across a reversal."
- **[[interplay-pretraining-midtraining-rl]]** — paired eval is what separates true capability gains from noise when RL rearranges probability mass.
- **[[faithful-synth-eval]]** — per-slice audit = per-slice CI; aggregates mask.
- **[[karpathy-training-neural-net-recipe]]** — predict-before-run; review worst 10; "don't be a hero."

## Companion visualization

**[figures/bootstrap-ci.html](figures/bootstrap-ci.html)** — interactive bootstrap explorer. Panel 1: pick N (items) and M (seeds), see the 95% CI halfwidth curve and the decomposition between item-count noise and per-run noise. Panel 2: two-checkpoint comparison — dial the per-item true delta and the noise, see paired-bootstrap p-value and sign-test p-value update live. Use it to calibrate how many items you need before you run an expensive eval.
