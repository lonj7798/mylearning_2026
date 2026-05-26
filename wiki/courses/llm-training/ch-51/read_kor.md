<!-- chapter: ch-51
     track: eval
     kind: content
     title: Metric Noise, Confidence, Go/No-Go
     deps: [ch-50]
     sources: [[scaling-laws-data-quality]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[judge-llm-bias]], [[karpathy-training-neural-net-recipe]], [[faithful-synth-eval]], [[interplay-pretraining-midtraining-rl]], [[reward-model-overoptimization]]
     figures: figures/bootstrap-ci.html
-->

# 51장 — Metric Noise, Confidence, Go/No-Go

> **핵심 통찰.** 2025년 eval에서 가장 어려운 질문은 "우리가 어떤 숫자를 얻었나?"가 아니라 "이 숫자가 어제의 숫자와 구별 가능한가?"다. 현대 post-training pipeline([[llama-3]] six rounds; [[olmo-2]] SFT → DPO → RLVR; [[olmo-3]] multi-branch model flow)은 stage별로 single-digit-percentage-point gain을 공개한다. 250-item eval set에서 1.2 pp lift는 *거의 항상* noise 안에 있고, *모든 현실적인 bootstrap width* 안에 있다. CI, paired comparison, written go/no-go memo 없이 ship하는 team은 결국 rolling average 아래 real regression을 숨기거나([[reward-model-overoptimization]]) judge-bias artifact를 쫓게 된다([[judge-llm-bias]]).
>
> **가이드라인.** 모든 headline metric을 세 가지 scaffolding이 필요한 point estimate로 취급하라. (1) per-run variance(decode temperature, prompt permutation, SFT eval 중 dropout, judge order)에 대한 bounded account, (2) *item* level bootstrap CI와 명시된 N, (3) checkpoint A가 checkpoint B를 이긴다고 주장할 때 paired comparison(paired bootstrap, sign test, Wilcoxon). go/no-go memo가 loop를 닫는다. 하나의 claim, 그 evidence, reviewer가 다시 실행해야 할 specific check를 담아라.

---

## §1 noise는 어디서 오는가

[[karpathy-training-neural-net-recipe]]가 제대로 framing한다. training은 조용히 실패한다. eval도 같은 방식으로 조용히 실패한다. green curve가 correct curve라는 뜻은 아니다. CI를 계산하기 전에 네 가지 variance source를 따로 budget하라. 아래 magnitude는 chat/reasoning benchmark에서 7B–70B instruct model에 대한 illustrative range다. 직접 측정한 뒤 자신의 숫자로 바꿔 넣어라.

| Source | Typical σ on accuracy (pp) | How to bound | Attested reference |
|---|---|---|---|
| **Decode temperature / sampling seed** | T=0 → 0; T=0.7 → 0.5–1.5; T=1.0 → 1–3 | seed 고정; prompt마다 K≥3 sample을 돌려 majority/mean 보고; 또는 "capability" eval에는 T=0 설정 | [[olmo-2]] RLVR protocol uses T=1.0 at training, T=0 at eval |
| **Prompt permutation (few-shot order, option order)** | 0.5–2.5 on MMLU-style MCQ | item마다 randomize; M≥3 permutation 평균; MCQ는 option position rotate | [[judge-llm-bias]] position bias flips 20–30% of judgments on pair-rank tasks |
| **Dropout / stochastic layers during eval** | `.eval()` 설정 시 0.0; 잊으면 0.2–1.0 | `model.eval()` assert; inference path에 BN/dropout 없는지 확인 | [[karpathy-training-neural-net-recipe]] "be paranoid about `.train()` vs `.eval()`" |
| **Judge variance (LLM-as-judge)** | N=200의 pairwise win-rate에서 1–4 pp; subjective task에서는 더 큼 | A/B swap 후 average; ≥2 judges pooling; reference-guided grading | [[judge-llm-bias]] §Technical Details — swap-and-average, reference-guided +10 pp agreement |
| **Item-count Monte Carlo (the CI you compute)** | ≈ sqrt(p·(1-p)/N) — p=0.5, N=200 → 3.5 pp | bootstrap(§3); N 증가 | Core statistical bound, attested in every eval harness README |

처음 네 가지는 *per run* noise를 bound한다. 마지막은 CI가 정량화하는 *per evaluation* noise다. per-run σ가 item-count σ를 넘으면 seed를 너무 적게 돌리는 것이다. item-count σ가 per-run σ를 넘으면 item을 너무 적게 돌리는 것이다. 좋은 setup은 둘이 2× 이내에 있다.

관련해서 [[faithful-synth-eval]]의 "average PPL hides tail degradation" 주장은 그대로 적용된다. aggregate뿐 아니라 per-slice CI도 계산하고 보고해야 한다. coding regression -2 pp를 숨기는 aggregate lift +0.3 pp는 gold reward가 한 slice에서 이미 peak한 뒤 생기는 [[reward-model-overoptimization]]의 failure가 예측하는 바로 그 상황이다.

---

## §2 정직한 noise budget

eval을 돌리기 전에 다음을 적어라.

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

이것은 [[karpathy-training-neural-net-recipe]]의 "predict-outcome-before-run" rule을 evaluation에 적용한 것이다. claimed gain이 CI halfwidth보다 작다면 그것을 ship할 수 없다.

---

## §3 item level의 Bootstrap CI

Bootstrap은 per-item score에 distributional assumption을 두지 않기 때문에 올바른 도구다. binary(MCQ correct/wrong), real-valued(BLEU, pass-rate-per-item, reward score), rank(pairwise win)에 모두 작동한다.

**Setup.** N item의 eval set. 각 item i에 대해 model은 score s_i(binary 또는 real)를 만든다. 관심 statistic은 mean θ̂ = (1/N) Σ s_i다.

**Bootstrap procedure.** b = 1..B (B ≥ 1000, 10000 preferred)에 대해:
1. {1..N}에서 replacement로 N indices를 sample한다.
2. sampled indices에 있는 s_i의 mean θ̂_b를 계산한다.

결과 {θ̂_1, ..., θ̂_B}가 θ̂의 bootstrap distribution이다.

**Percentile CI (가장 단순).** 95% CI는 [θ̂_{(0.025·B)}, θ̂_{(0.975·B)}]다. 즉 bootstrap distribution의 2.5th와 97.5th percentile이다.

**BCa CI (bias-corrected accelerated).** θ̂가 0 또는 1에 가까우면 percentile CI는 biased하다(saturating benchmark에서 흔함). BCa는 보정한다.
1. Bias factor z₀ = Φ⁻¹( (# {θ̂_b < θ̂}) / B ).
2. Jackknife에서 acceleration â: â = Σ(θ̂_(·) − θ̂_(-i))³ / [6 · (Σ(θ̂_(·) − θ̂_(-i))²)^(3/2)], 여기서 θ̂_(-i)는 leave-one-out estimate다.
3. Adjusted percentile α₁ = Φ(z₀ + (z₀ + z_{α/2}) / (1 − â·(z₀ + z_{α/2}))) and α₂ = Φ(z₀ + (z₀ + z_{1-α/2}) / (1 − â·(z₀ + z_{1-α/2}))).
4. CI = [θ̂_{(α₁·B)}, θ̂_{(α₂·B)}].

θ̂ ∈ [0.2, 0.8]이고 N ≥ 200이면 percentile을 써라. θ̂가 boundary 근처이거나 N < 200이면 BCa를 써라.

**Rough halfwidth intuition.** p가 0.5 근처이고 N이 큰 binary metric의 경우, 95% percentile-bootstrap halfwidth는 ≈ 1.96·sqrt(p·(1-p)/N)이다. N=100 → 9.8 pp; N=500 → 4.4 pp; N=2000 → 2.2 pp. 2 pp lift를 resolve하려면 약 2000 item이 필요하다. 대부분의 published benchmark(HumanEval 164, GSM8K test 1319, MATH500 500)는 4–9 pp halfwidth 범위에 있다. 이것이 [[olmo-2]]의 single-digit-pp RLVR gain이 borderline인 이유다.

---

## §4 두 checkpoint 비교

checkpoint가 같은 item에서 평가되었다면 두 CI를 빼는 것은 *틀렸다*. pairing을 버리고 variance estimate를 부풀린다. paired comparison을 사용하라.

### Paired bootstrap (올바른 기본값)

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

핵심은 *indices*를 resample하고 pair (s_A[i], s_B[i])를 함께 유지하는 것이다. per-item variance가 cancel되고 delta의 variance만 남는다. halfwidth는 보통 unpaired difference of two independent CIs보다 1.5–3× 더 tight하다.

### Sign test (distribution-free, score magnitude를 믿지 않을 때 사용)

paired per-item outcome이 있을 때 W = #{i : s_A[i] > s_B[i]}, L = #{i : s_A[i] < s_B[i]}, T = #{i : tie}라고 하자. tie는 버린다. H₀(no difference) 아래에서 W ~ Binomial(n=W+L, p=0.5)다.

Two-sided p-value: X ~ Binomial(W+L, 0.5)일 때 `p = 2 · min( P(X ≤ min(W,L)), P(X ≥ max(W,L)) )`. W+L ≥ 25이면 normal approximation: `z = (|W − L| − 1) / sqrt(W + L)`, `p = 2·(1 − Φ(z))`.

p < 0.05일 때 α=0.05에서 reject한다. sign test는 magnitude를 무시한다. judge가 per item에서 miscalibrated일 수 있지만 preference의 *direction*은 trustworthy하다고 볼 때 robust하다. 이는 LLM judge가 *direction*에서는 인간과 약 80% 일치하지만 calibration은 drift한다는 [[judge-llm-bias]]의 finding과 맞다.

### "Two-run minimum" rule

checkpoint A의 single seed와 checkpoint B의 single seed 비교는 decode noise만으로도 1–3 pp delta를 가짜로 만들 수 있다. 두 checkpoint를 ≥2 seed에서 실행하라. seed pair마다 (Δ̂, CI)를 보고하고, improvement를 주장하기 전에 seed pair across consistent sign을 요구하라. [[llama-3]]의 six-round loop는 이 rule에 암묵적으로 의존한다. 각 round의 preference data는 current-best checkpoint에서 regenerate되고, 한 decode seed에서 나온 spurious "best"는 forward propagate된다. 두 run, consistent sign, 0을 제외하는 paired CI: 이제 ship할 수 있다.

---

## §5 Rolling average vs regression masking

[[olmo-3]]의 1000+ intermediate checkpoint나 [[llama-3]]의 per-round eval에서 나오는 training curve는 noisy하다. 두 가지 반대 failure mode가 있다.

1. **No smoothing.** step 3200에서 2 pp dip을 보고 training을 멈춘다. 사실 decode-seed noise였고 step 3300의 다음 checkpoint에서 완전히 회복된다. 하루를 낭비했다.
2. **Over-smoothing.** 7-checkpoint rolling mean을 보고한다. 마지막 세 checkpoint가 모두 1.5 pp 떨어진 real regression을 놓친다. 정확히 [[reward-model-overoptimization]]의 gold-reward-peaks-then-falls shape다. 더 나쁜 model을 ship했다.

decision은 "항상 smooth"도 "절대 smooth하지 마라"도 아니다. decision tree다.

```
per-checkpoint CI가 claimed effect보다 넓은가?
├── YES → CI halfwidth < effect/2가 될 때까지 K checkpoint across smooth한다.
│         regression이 보이도록 unsmoothed last-K trend도 별도로 보고한다.
└── NO  → smooth하지 않는다. per-checkpoint CI가 이미 signal을 resolve한다.

rolling window W=3 checkpoint에서 curve가 monotone up인가?
├── YES → rolling mean을 보고한다.
└── NO  → last K point를 개별적으로 + per-point CI와 함께 보여 주는 방식으로 바꾼다.
        trend reversal을 가로지르는 rolling mean은 절대 보고하지 마라. reversal을 mask한다.

gold/held-out reward가 PEAK 후 떨어졌는가?  ([[reward-model-overoptimization]])
├── YES → rolling mean은 거짓말하고 있다. raw curve를 보여라. training을 멈춰라.
└── NO  → W=3–5의 rolling mean은 safe하다.
```

가이드라인: communication을 위해 smooth하되, decision에는 절대 smooth하지 마라. raw curve가 decision input이다.

---

## §6 Go/no-go memo — template

memo는 notebook에 보관하는 것이 아니라 reviewer에게 건네는 것이다. 한 page, 다섯 section, falsifiable해야 한다.

```
go-no-go-memo.md
----------------

## 1. 주장
Checkpoint `ckpt-round-3`는 `capability-bundle-A`
(MATH500, GSM8K, HumanEval, IFEval 평균)에서 `ckpt-round-2`보다 +1.8 pp 개선되며,
`safety-bundle-B`의 어떤 safety slice에서도 0.5 pp를 넘게 regress하지 않는다.

## 2. 근거
- **Item-level bootstrap CIs** (B=10000, percentile method, task별 N은 아래 기록):
  | Eval       | N    | θ̂(A) | 95% CI       | θ̂(B) | 95% CI       | Δ paired | CI paired  |
  | MATH500    | 500  | 52.4 | [47.9, 56.6] | 50.2 | [45.8, 54.6] | +2.2     | [+0.4,+4.0]|
  | GSM8K      | 1319 | 86.1 | [84.2, 87.9] | 85.3 | [83.3, 87.1] | +0.8     | [-0.2,+1.8]|
  | HumanEval  | 164  | 71.9 | [65.2, 78.0] | 69.5 | [62.8, 75.6] | +2.4     | [-0.6,+5.4]|
  | IFEval     | 541  | 78.3 | [74.7, 81.7] | 76.5 | [72.8, 80.0] | +1.8     | [+0.3,+3.3]|
- **Paired-bootstrap + sign test agree**: MATH500과 IFEval에서 일치(p < 0.05).
- **Two-run minimum:** 두 seed(s=7, s=19) 모두 네 task에서 consistent sign을 보인다.
- **Safety slice (safety-bundle-B):** 한 sub-slice의 max regression = -0.3 pp;
  paired CI = [-1.1, +0.5], 0을 포함한다. regression detected되지 않음.
- **Judge-bias check:** IFEval-judge subset의 position-swap parity = 94% (≥90% threshold).

## 3. Variance accounting
- Decode: T=0, greedy; decode σ = 0.
- Prompt permutation: MCQ task는 3 permutation 평균을 사용; σ_perm ≈ 0.4 pp.
- Judge: IFEval은 rule-based grader 사용(σ_judge ≈ 0); LLM judge 없음.
- Item-count CI halfwidth(위 table 기준)가 지배적이다. OK.

## 4. 검토했지만 dismiss한 regression
- MMLU-Pro: -0.4 pp, paired CI [-1.6, +0.8]. noise 안에 있음; action 없음.
- XSTest safety-borderline: -0.2 pp, CI 안에 있음. next round re-eval로 flag.

## 5. reviewer가 확인해야 할 것
- `eval/run.py --ckpt=round-3 --seed=7 --tasks=math500,ifeval`을 다시 실행하고
  pass rate가 table의 ±1 pp 안에 있는지 확인한다.
- `data/paired_items.jsonl`이 두 checkpoint에 대해 동일한 item ID를 갖는지 확인한다.
- `metrics.jsonl`이 row마다 `ckpt_hash`를 log하는지 확인하고 claim과 diff한다.

## 6. 결정
GO — `ckpt-round-3`를 `main/post-training/round-3`에 merge한다.
Trigger: claim한 곳의 모든 paired CI가 0을 제외한다. 0.5 pp를 넘는 safety regression은 없다.
```

memo의 규칙. Claim은 singular, falsifiable, quantitative하다. Evidence는 item-level, paired, commit SHA에서 reproducible해야 한다. Variance는 assumed가 아니라 explicit하게 budget된다. Regression은 숨기지 않는다. CI와 함께 enumerate하고 dismiss한다. reviewer의 check list는 구체적이어야 한다. specific file, specific command, specific tolerance를 적는다.

---

## §7 이 장의 주제에서 흔한 failure mode

**a) single-seed delta를 "result"로 보고하는 것.** 해결책: two-run minimum; consistent seed에 paired bootstrap.

**b) significance를 주장하기 위해 unpaired CI 두 개를 빼는 것.** 그 "CI"는 paired CI보다 보통 2× 넓고, paired CI라면 0을 제외할 때도 0을 제외하지 못할 수 있다. 해결책: matched item ID로 paired bootstrap.

**c) LLM-as-judge eval에서 judge variance를 무시하는 것.** [[judge-llm-bias]]는 일부 model에서 position swap 아래 20–30% flip rate를 보인다. 해결책: swap-and-average; judge pooling; win rate와 함께 judge-agreement rate 보고.

**d) trend reversal 위에 rolling average를 과도하게 smoothing하는 것.** [[reward-model-overoptimization]]의 peak-then-fall shape가 W=7에서 보이지 않게 된다. 해결책: §5의 decision tree.

**e) item level이 아니라 aggregate mean에서 bootstrap하는 것.** item이 아니라 score를 resampling하면 task across item correlation을 버린다. 해결책: item index를 resample하고 pair를 유지한다.

**f) CI halfwidth보다 작은 gain을 주장하는 것.** [[karpathy-training-neural-net-recipe]]의 "predict-before-run" rule: predicted effect가 predicted CI보다 작다면, eval을 돌리는 것이 아니라 lottery를 돌리는 것이다.

---

## §8 Connections

- **ch-50** — slice analysis가 upstream step이다. 이 장은 각 slice에 CI를 추가한다.
- **ch-52** — safety eval은 같은 CI machinery를 상속한다. adversarial prompt에는 stratified bootstrap이 필요하다.
- **ch-53** — harness를 wire up하고 첫 real go/no-go memo를 만드는 lab.
- **[[llama-3]]** — every round에서 paired comparison이 중요한 iterative round setting.
- **[[olmo-2]]**, **[[olmo-3]]** — 공개 보고된 per-stage gain; 이 장이 겨냥하는 CI regime.
- **[[judge-llm-bias]]** — bootstrap이 포함해야 하는 judge-variance floor.
- **[[reward-model-overoptimization]]** — "reversal across smoothing 금지"를 동기화한다.
- **[[interplay-pretraining-midtraining-rl]]** — RL이 probability mass를 재배열할 때 true capability gain과 noise를 구분하는 것이 paired eval이다.
- **[[faithful-synth-eval]]** — per-slice audit = per-slice CI; aggregate는 mask한다.
- **[[karpathy-training-neural-net-recipe]]** — predict-before-run; review worst 10; "don't be a hero."

## Companion visualization

**[figures/bootstrap-ci.html](figures/bootstrap-ci.html)** — interactive bootstrap explorer. Panel 1: N(items)와 M(seeds)을 골라 95% CI halfwidth curve와 item-count noise / per-run noise decomposition을 본다. Panel 2: two-checkpoint comparison — per-item true delta와 noise를 조절하면 paired-bootstrap p-value와 sign-test p-value가 live update된다. expensive eval을 돌리기 전에 몇 item이 필요한지 calibrate하는 데 사용하라.
