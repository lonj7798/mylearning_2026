<!-- chapter: ch-53
     track: eval
     kind: lab
     title: Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness
     deps: [ch-52]
     sources: [[tulu-3]], [[gsm-symbolic]], [[gsm1k]], [[ifbench]], [[mmlu-pro]], [[adding-error-bars-evals]], [[harmbench-data]], [[xstest]], [[deduplicating-training-data]], [[signal-and-noise-eval]], [[swe-bench-illusion]]
     figures: figures/eval-pipeline.html, figures/lsh-band-curve.html
     excerpts: excerpts/held-out-suite-tulu3.md, excerpts/perturbation-robustness-sets.md, excerpts/paired-statistics-and-power.md, excerpts/minhash-contamination-gate.md, excerpts/harmbench-behavior-layering.md
     revised: 2026-09 (generality revision)
-->

# Chapter 53 — Lab: Held-Out Suite, Forgetting Report, Perturbation Robustness를 갖춘 Evaluation Harness

> **핵심 통찰.** Checkpoint마다 평균 score 하나를 보고하는 harness는 general capability 증가와 benchmark-specific behaviour 증가를 구분할 수 없다. Tülu 3는 development suite와 unseen suite를 분리하고 decision을 내는 동안 unseen score를 보지 않았다. 8B에서 pipeline은 development average +3.9점(64.9 → 68.8), unseen average +2.5점(29.9 → 32.4)을 얻었다. Coding에서는 development score가 86.2 → 83.9, unseen score가 11.5 → 7.4로 움직였다(arXiv:2411.15124 Table 31). 같은 checkpoint의 task 6개 safety average는 SFT와 final RL 사이 93.1 → 85.5이고 GSM8K는 76.2 → 87.6이다(Table 6). Harness는 task를 hold out하고, question별로 comparison을 pair하고, prompt template을 바꾸며, score를 읽기 전에 contamination gate를 둘 때만 이를 탐지한다.
>
> **지침.** 같은 question에서 checkpoint 두 개를 비교할 때 paired difference와 standard error를 사용하라. Pairing은 shared question-difficulty variance를 제거하며 §5 worked example에서는 identical data에서 95% interval을 ±6.1 pp에서 ±3.8 pp로 좁힌다([[adding-error-bars-evals]] §4.2, 식 7). Checkpoint 두 개의 training data를 얻을 수 있다면 scoring 전에 contamination gate를 실행하고 gate가 trip하면 score 보고를 거부하라. Tülu 3의 8-gram check는 HumanEval instance의 70.7%가 Evol CodeAlpaca와, MATH instance의 18.2%가 NuminaMath-TIR과 overlap함을 찾았다(arXiv:2411.15124 Table 37). Training data를 얻을 수 없다면 closed checkpoint에서 얻을 수 있는 유일한 contamination evidence인 memorization probe로 대체하라([[swe-bench-illusion]] §4.1). Target task가 mathematics라면 perturbation set을 추가하라. GSM8K template question 100개 정확도는 model 25개 중 21개에서 같은 question perturbed distribution center보다 standard deviation 하나 초과 위에 있다([[gsm-symbolic]] §4.1).

---

## 이 장이 범용 모델에 중요한 이유

이는 eval track의 마지막 lab이다. Training lab 뒤, infrastructure track 전에 실행하며, output은 뒤 chapter가 쓰는 acceptance test다. SFT lab([[ch-36]])과 preference/RL lab([[ch-46]])은 각각 checkpoint를 만들고 자기 stage의 held-out capability-retention number를 보고했다. 이 장은 두 lab을 develop하는 동안 쓰지 않은 task에서 모든 checkpoint를 한 번에 측정하는 harness를 만들고, 그 measurement를 go/no-go decision으로 바꾼다.

Harness가 탐지하려는 failure mode는 narrowing이다. Checkpoint는 훈련한 task에서는 더 좋아지고 측정하지 않은 다른 것에서는 더 나빠진다. Narrowing은 세 이유로 single average에 보이지 않는다.

1. Development 중 쓴 task는 data mixture가 tune된 대상이므로, score는 일부 capability가 아니라 tuning loop의 measurement다([[tulu-3]] §7.4).
2. Score는 question 수백 개의 point estimate로 보고되며, estimate noise는 regression 몇 점을 숨길 만큼 크다([[adding-error-bars-evals]] §5; [[signal-and-noise-eval]] §3.1).
3. Fixed benchmark accuracy는 benchmark item memorization으로도 오를 수 있고, 그 benchmark에서 하는 통계만으로는 이를 드러낼 수 없다([[gsm1k]] §5.4; [[swe-bench-illusion]] §4.1).

Harness는 각각에 다른 mechanism으로 답한다. Held-out suite(§2), template variation 및 paired statistics(§3, §5), contamination gate 및 perturbation set(§4, §7)이다.

Pipeline에서의 위치는 다음과 같다. 이 lab은 SFT([[ch-36]]) 및 RL([[ch-46]]) checkpoint를 받아 [[ch-50]], [[ch-51]]의 measurement theory와 [[ch-52]] safety protocol을 적용하고, infrastructure track([[ch-54]] onward)이 run scale 여부를 결정할 때 쓰는 memo를 낸다.

---

## §1 Deliverable과 두 budget path

Lab은 artifact 다섯 개를 만든다.

| Artifact | 내용 | 절 |
|---|---|---|
| `suite.yaml` | task id, split label(`dev` 또는 `heldout`), task당 prompt template ≥3개, decoding setting, scorer | §2, §3 |
| `runs/<ckpt>/<task>/<template>.jsonl` | question당 row 하나: id, cluster id, prompt, completion, score, slice label | §1 |
| `contamination.json` | training set 각각에 대한 task별 overlap rate 및 gate verdict | §7 |
| `forgetting.md` | capability group별 paired interval을 붙인 base → SFT → RL delta | §6 |
| `memo.md` | verdict, gate status, non-inferiority table, regression, recommendation | §9 |

**Full-budget path.** Checkpoint × task × template 전체에 대해 live generation한다. Held-out task 6개, task마다 template 3개, task마다 question 500개, generative task는 question당 `K = 4` sample이다. 8B checkpoint 세 개에 약 10 GPU-hour다(token count에서 추정한 값이며 measured figure가 아니다).

**Resource-constrained path.** ch-36, ch-46 lab이 이미 JSONL에 저장한 generation을 재사용하고(both lab은 rollout을 JSONL에 저장), 그 lab이 실행하지 않은 held-out task만 추가하며 `K = 1`로 둔다. Earlier lab이 이미 decode한 것은 decode하지 않으므로 workstation 하나에서 실행한다.

Constrained path에서 빼도 되는 것은 live decoding, `K > 1`, agentic slice([[ch-51a]])다. 빼면 안 되는 것은 held-out split, task당 template ≥3개, paired analysis, multiple-comparison correction, contamination gate다. 이 네 가지는 output이 stated error rate가 있는 score인지 error rate 없는 point number인지를 결정하며 GPU time을 쓰지 않는다.

Scope에 관한 주석: 이 harness는 programmatic scorer가 있는 single-turn/multi-turn text task를 score한다. Pairwise judge task(AlpacaEval 스타일 win rate)는 [[ch-49]] machinery로 측정한다. Any judge number를 신뢰하려면 그 bias control이 precondition이다. 이 lab은 scorer가 published agreement rate를 가진 released classifier일 때만 judge를 쓴다(§8).

[figures/eval-pipeline.html](figures/eval-pipeline.html)은 어떤 gate가 어느 report를 막는지를 포함한 stage-by-stage flow를 그린다. Toggle은 각 gate를 순서대로 trip하고 memo verdict가 바뀌는 것을 보게 한다.

### Data model

Record type 네 개면 충분하다. Slice와 template은 나중에 적용하는 filter가 아니라 sample field여야 한다. Run 뒤에 재구성한 slice는 checkpoint 사이에서 안정적으로 pair할 수 없기 때문이다.

```python
# harness/core.py
@dataclass(frozen=True)
class Sample:
    sample_id: str                 # checkpoint와 template 전반에서 stable
    cluster_id: str                # 함께 뽑힌 question은 이를 공유(§5 참고)
    prompt_vars: Mapping[str, str] # fixed string이 아니라 template으로 render
    gold: str                      # 또는 accepted answer list
    slices: Mapping[str, str]      # {"subject": "algebra", "length_bucket": "long"}

@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str                     # "dev" | "heldout"
    samples: Sequence[Sample]
    templates: Sequence[str]       # >= 3, §3 참고
    scorer: Callable[..., float]      # (completion, gold) -> [0,1] score
    decoding: Mapping[str, object] # temperature, max_new_tokens, K
    contamination_scope: Sequence[str]   # 대조할 training-set dump

@dataclass(frozen=True)
class Score:                       # (checkpoint, task, template, sample, k)마다 row 하나
    checkpoint: str; task: str; template: str
    sample_id: str; cluster_id: str; k: int; value: float
```

Downstream의 모든 것은 `Score` row에 대한 aggregation이다. Sample별 row를 보존해야 §5의 paired analysis가 가능하다. Task mean만 저장하는 harness는 이후 paired standard error를 계산할 수 없다.

## §2 Held-out generality suite

**정의.** Held-out suite는 training data, hyperparameter, checkpoint를 고르는 동안 score를 inspect하지 않는 evaluation task 집합이다. Development에 사용한 benchmark의 test split과는 다르다. Development 중 benchmark의 score를 하나라도 읽으면 development decision이 그 benchmark의 idiosyncrasy와 결합된다.

**측정 가능한 문제.** Development score는 development score 자체로는 알 수 없는 크기만큼 capability를 과대평가한다. Tülu 3는 independent design process로 unseen suite를 만들고 같은 checkpoint에 두 split을 함께 보고해 이를 수량화했다([[tulu-3]] §7.3, §7.4).

**메커니즘.**

1. Core capability마다 development task(development 중 자유롭게 사용)와, 다른 formulation으로 같은 capability를 시험하는 held-out task를 배정한다.
2. 첫 training run 전에 held-out suite를 freeze하고 `suite.yaml`에 hash를 기록한다.
3. Held-out score는 tuning loop 안에서는 절대 읽지 않고 gate에서만 읽는다.
4. 모든 checkpoint의 (development, held-out) pair와 그 gap을 보고한다.

**근거.** Tülu 3 8B는 skill마다 development task 하나와 held-out task 하나를 사용한다(arXiv:2411.15124 Table 31; [[held-out-suite-tulu3]]의 extract).

| Skill(dev → held-out task) | 8B SFT | 8B DPO | 8B Final | Held-out: SFT | DPO | Final |
|---|---:|---:|---:|---:|---:|---:|
| Skill average | 64.9 | 68.3 | 68.8 | 29.9 | 31.9 | 32.4 |
| Knowledge(MMLU → GPQA) | 65.9 | 68.7 | 68.2 | 31.9 | 31.2 | 35.7 |
| Reasoning(BBH → AGIEval English) | 67.9 | 65.8 | 66.0 | 56.2 | 61.8 | 59.3 |
| Math(MATH → DeepMind Mathematics) | 31.5 | 42.0 | 43.7 | 32.3 | 33.0 | 35.4 |
| Coding(HumanEval → BigCodeBench) | 86.2 | 83.9 | 83.9 | 11.5 | 9.5 | 7.4 |
| Instruction following(IFEval → IFEval-OOD) | 72.8 | 81.1 | 82.4 | 17.6 | 23.9 | 24.3 |

두 해석이 중요하다. Pipeline은 두 average를 모두 개선했으며, 이것이 저자들의 headline이다(§7.4.1). Coding row는 held-out task에서 모든 stage에서 반대 방향으로 움직여, development score가 2.3점 움직인 뒤 멈추는 동안 final은 SFT checkpoint보다 4.1점 낮다. HumanEval만 담은 report라면 coding trajectory를 flat이라고 설명한다.

**조건과 한계.** Tülu 3 unseen suite에는 safety task가 없으므로(§2.2) 이 lab §8이 하나를 공급한다. 저자들은 또한 comparison model이 GPQA, MMLU-Pro, AGIEval, DeepMind Mathematics, BigCodeBench로 결코 훈련하지 않았는지는 확인할 수 없다고 말한다. 따라서 cross-model held-out comparison은 within-pipeline comparison보다 약하다(§7.4.2). 결과는 단일 연구다.

**Lab의 suite.** [[ch-36]] 또는 [[ch-46]]에서 쓴 data와 겹치지 않도록 고른 held-out task 6개다.

| Capability | Held-out task | 이 task를 쓰는 이유 |
|---|---|---|
| Knowledge | MMLU-Pro | Item 12,032개, answer option 최대 10개(83%가 10개, mean 9.47); 같은 prompt 24개에서 MMLU 4–5% 대비 prompt sensitivity 약 2%([[mmlu-pro]] §3.1, §3.2, §6.3) |
| Knowledge, hard | GPQA | MMLU-Pro와 함께 Tülu 3의 held-out knowledge task([[tulu-3]] Table 3) |
| Reasoning | BBH 또는 AGIEval English | Development에서 BBH를 쓰지 않았다면 BBH, 썼다면 AGIEval English([[tulu-3]] Table 24) |
| Instruction following | IFBench | IFEval constraint 25개와 disjoint한 constraint 58개; GPT-4.1과 Claude 3.7 Sonnet은 50% 미만([[ifbench]] §1, 그림 1) |
| Math robustness | GSM-Symbolic-style variant 또는 GSM1k | §4 |
| Multilingual | MGSM | Question form 하나의 translated copy가 cluster를 이루므로 §5의 clustered standard error가 강제됨([[adding-error-bars-evals]] §2.2) |

IFBench는 instruction-following RL을 거친 model에 가장 강한 single addition이다. Tülu-3-8B-DPO policy에서 IF-RLVR training은 IFEval을 81.1 → 92.2, IFBench를 25.2 → 44.6으로 올린다([[ifbench]] §4.7, Table 6). 두 benchmark는 함께 움직이지만 속도가 다르다. IFEval만 보고하는 pipeline은 constraint-following ability와 IFEval-template fitting을 구분할 수 없다.

---

## §3 Suite의 축으로서 prompt template

**정의.** Template은 question 주위를 둘러싸는 fixed text다. System message, instruction phrasing, answer-format request, few-shot block이 포함된다. 다른 말로 같은 answer를 요구하는 template 두 개는 같은 underlying skill의 서로 다른 measurement 두 개를 정의한다.

**측정 가능한 문제.** Checkpoint 사이 score 차이는 checkpoint 하나의 template 사이 score 차이보다 작을 수 있다. MMLU-Pro 저자들은 reasonable prompt 24개에서 model을 평가했다. MMLU에서는 spread가 보통 4–5%, 최대 10.98%이고 MMLU-Pro에서는 보통 약 2%, 최대 3.74%였다([[mmlu-pro]] §6.3). Template 하나에서 MMLU-like task가 3점 개선됐다고 주장하는 것은 template spread 안에 있다.

**메커니즘.**

1. Task마다 template 세 개를 쓴다. Benchmark canonical template, few-shot block이 없는 plain-instruction variant, answer-format request를 바꾸는 variant(예: `Therefore, the answer is X` 대신 `Answer: X`)다.
2. 모든 checkpoint를 세 template 전부에서 실행한다. Checkpoint 사이 template을 절대 섞지 않는다.
3. Task마다 template mean과 template range를 보고한다.
4. Report claim이 세 개 중 가장 불리한 template에서도 성립하도록 maximum이 아니라 template minimum을 gate에 쓴다.

**계산 예시.** Knowledge task에서 checkpoint B의 T1–T3 score가 48.2, 51.0, 45.9이고 checkpoint A는 47.0, 47.6, 46.6이라고 하자. Mean은 B 48.37, A 47.07, difference +1.30이다. Range는 B 5.1, A 1.0이다. B의 template range는 mean difference보다 크며 T3에서 B advantage는 사라진다(−0.7). Memo는 “+1.3 mean, −0.7 worst-template”을 보고하며 improvement를 주장하지 않는다.

**조건과 한계.** Template variance는 benchmark와 checkpoint의 property다. MMLU-Pro는 item을 더 어렵게 하고 option을 추가하여 이를 낮췄다. 저자들은 model robustness 자체를 scope 밖으로 둔다(§6.3). 결과는 단일 연구다. Answer-extraction code도 template의 일부라는 점에 유의하라. Tülu 3는 LaTex를 요청하지 않은 DeepMind Mathematics에서 model이 answer를 LaTeX로 format했고, 이것이 reasoning과 extraction logic 모두를 방해했다고 보고한다([[tulu-3]] §7.4.1).

---

## §4 Perturbation robustness

**정의.** Perturbation set은 benchmark와 같은 reasoning template에서 생성하되 surface detail을 바꾼 question 집합이다. Name, numeric value, 답을 바꾸지 않는 추가 clause를 바꾸거나, 같은 specification으로 완전히 새 item set을 작성한다.

**측정 가능한 문제.** Math data에서 SFT 또는 RL을 한 뒤 GSM8K/MATH accuracy가 오르는 이유는 두 가지일 수 있다. Model이 더 많은 grade-school arithmetic problem을 푸는 것일 수도 있고, 그 benchmark의 specific item 및 phrasing에 더 잘 맞게 된 것일 수도 있다. Reasoning은 고정하고 surface를 바꾸면 둘을 구분한다.

**메커니즘(template instantiation).** GSM-Symbolic은 GSM8K question을 typed variable과 answer를 integral로 유지하는 constraint가 있는 template으로 바꾸고 instantiation을 sample한다([[gsm-symbolic]] §3.1, 그림 1). Evaluation protocol은 template 100개 × sample 50개 = item 5,000개를 쓰고, 각각 item 100개짜리 dataset 50개로 취급하며 8-shot CoT와 greedy decoding으로 score한다(§3.2).

**근거.**

- Set 50개 전반에서 Gemma2-9B는 한 model의 worst/best set gap이 12%를 넘고 Phi-3.5-mini는 약 15%다([[gsm-symbolic]] §4.1).
- Original GSM8K item 100개의 accuracy는 그 model GSM-Symbolic distribution center보다 standard deviation 하나 초과로 높을 때가 많으며, model 25개 중 21개에서 그렇다(§4.1). 저자들은 data contamination을 가능한 설명 하나로 든다.
- Name만 바꾸면 number를 바꿀 때보다 variance가 작고, 둘 다 바꾸면 drop이 가장 크다(§4.2). 관련 있어 보이지만 아무것도 바꾸지 않는 clause 하나를 추가한 GSM-NoOp은 accuracy를 최대 약 65% 낮춘다. Phi-3-mini-128k-instruct는 −65.7, Gemma2-9b-it는 −63.0이다(§4.4, 그림 8a).
- Independent-item variant인 GSM1k는 model assistance 없이 human annotator가 쓴 새 grade-school problem 1,205개다. Worst model은 GSM8K보다 GSM1k에서 8% 낮고, GSM8K test set 생성의 character별 log-likelihood는 model의 GSM8K−GSM1k gap과 Spearman 0.36(`p = 0.03`)으로 상관한다([[gsm1k]] §5.4). Difficulty는 match했다. Annotator는 15분에 GSM8K problem `4.07 ± 0.93`, GSM1k problem `4.36 ± 1.11`개를 풀었다(§3.2.2). 두 연구가 모두 benchmark-specific inflation을 찾았다는 의미에서 [[gsm-symbolic]]과 **replicated**됐지만, 측정한 mechanism은 다르다(item novelty vs surface perturbation).

**Lab이 만드는 것.** GSM1k는 public이 아니므로 lab은 perturbation set을 직접 생성한다. [[ch-46]] evaluation의 math data에서 item 100개를 골라 각각 variable 2~4개짜리 template으로 rewrite하고 template마다 instantiation 30개를 sample한다. Checkpoint마다 수 세 개를 보고한다. Original-item accuracy, set 30개의 mean, set 30개의 standard deviation이다. Memo에 넣는 quantity는 **perturbation gap** = original − perturbed mean이며 perturbed distribution의 standard deviation 단위로 표현한다.

**계산 예시.** Checkpoint B는 original item 100개에서 71.0이다. Instantiation set 30개의 mean은 63.4, standard deviation은 3.1이다. Gap은 7.6점 = `2.45 σ`다. Checkpoint A(SFT model)는 original 58.0, perturbed mean 56.4, `σ = 2.9`, gap 1.6점 = `0.55 σ`다. B는 original item에서 13점, perturbed mean에서 7점을 얻었다. B의 apparent gain 13점 중 6점, 즉 절반보다 약간 작은 부분은 perturbation 뒤에는 남지 않는다. Memo는 둘 다 보고하며 RL stage는 perturbation gap을 `1.9 σ` 키운 것으로 기록한다.

**조건과 한계.** GSM-Symbolic template은 grade-school arithmetic이다. Competition mathematics나 code로 transfer한다는 결과는 보이지 않았다. 저자들이 제시한 해석, 즉 model이 formal reasoning이 아니라 in-distribution pattern matching을 한다는 해석은 accuracy drop의 **해석**이지 별도 measurement가 아니다.

---

## §5 Memo에 필요한 통계

아래 네 항목은 모두 [[adding-error-bars-evals]]에서 가져왔고, [[paired-statistics-and-power]]에 식 번호와 함께 인용했다.

### 5.1 Paired difference

Miller는 difference를 model minus baseline으로 쓴다. 이 lab에서는 earlier checkpoint A가 baseline, candidate가 B이므로 같은 question i에 대해 `d_i = s_{B,i} − s_{A,i}`다. 그러면

```
SE_paired = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )        (식 7)
```

여기서 `n`은 question 수, `d̄`는 question 전반 mean difference, `d_i`는 question별 difference다. Variance relation은 `Var(µ̂_paired) = Var(µ̂_unpaired) − 2 Cov(x_A, x_B)/n`이므로, checkpoint 두 개가 어떤 question이 어려운지에 동의할 때 pairing이 도움이 된다(§4.2).

**계산 예시.** `n = 500` question, binary scoring이다. 2×2 table은 둘 다 correct 165, A correct/B wrong 35, A wrong/B correct 60, 둘 다 wrong 240이다.

- `p_A = (165+35)/500 = 0.400`, `p_B = (165+60)/500 = 0.450`, difference `p_B − p_A = +0.050`.
- Question별 difference는 question 60개에서 +1, 35개에서 −1, 405개에서 0이다. Mean = `25/500 = 0.050`. 각 difference의 square는 0 또는 1이므로 `E[d²] = 95/500 = 0.190`, `Var(d) = 0.190 − 0.050² = 0.1875`.
- `SE_paired = sqrt(0.1875/500) = 0.0194` → 95% interval `+5.0 ± 3.8 pp = [+1.2, +8.8]`.
- Unpaired: `SE_A = sqrt(0.4·0.6/500) = 0.0219`, `SE_B = sqrt(0.45·0.55/500) = 0.0222`, `SE_unpaired = sqrt(SE_A² + SE_B²) = 0.0312` → `+5.0 ± 6.1 pp = [−1.1, +11.1]`.

같은 data, 같은 question에서도 paired interval은 zero를 제외하고 unpaired interval은 제외하지 않는다. 여기 variance ratio는 2.6×다. 이것이 이 lab의 regression rule을 paired interval로 쓰는 이유이며, “B interval이 A mean 아래에 있으면 B가 regress했다”는 rule을 쓰지 않는 이유다. 후자는 unpaired interval 두 개를 섞으며 false-positive/false-negative rate가 모두 알려지지 않았다.

### 5.2 Clustered standard error

Question이 group으로 올 때, 즉 passage 하나에 question 여러 개가 있거나 question 하나가 language 여러 개로 번역될 때, ordinary standard error와 ordinary bootstrap의 independence assumption은 실패한다. Cluster에 대해 standard error를 계산하라([[paired-statistics-and-power]]의 paired version 식 8). Anthropic model에서 measured clustered standard error는 DROP에서 naive 값의 3.05×, MGSM에서 1.88×, RACE-H에서 1.10×다([[adding-error-bars-evals]] Table 4). 그러므로 lab MGSM task는 `cluster_id` = source question id를 가지며, interval은 naive computation이 시사하는 것보다 약 두 배 넓다.

### 5.3 Power와 minimum detectable effect

```
n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²                  (식 9)
```

`n`은 independent question 수, `α`는 false-positive rate, `1−β`는 power, `δ`는 minimum detectable effect, `ω²`는 conditional mean difference variance, `σ²_A`, `σ²_B`는 두 model score의 expected conditional variance, `K_A`, `K_B`는 question당 뽑는 sample 수다. Miller worked case는 `σ²_A = σ²_B = 0`, `ω² = 1/9`, `δ = 0.03`, `α = 0.05`, `β = 0.20`이며, `n = (1.96 + 0.84)² (1/9) / 0.03² ≈ 969`다. 저자는 새 eval에 question 1,000개 이상이 있어야 한다고 결론낸다(§5).

§5.1 discordance rate(`Var(d) = 0.1875`)와 이 lab의 task 500 question에 대해 invert하면 다음과 같다.

```
δ_min = (1.96 + 0.84) · sqrt(0.1875/500) = 2.80 · 0.0194 = 0.054
```

Checkpoint 사이 disagreement가 이 정도인 500-question task는 80% power에서 5.4 pp difference를 detect하며 그보다 작은 것은 detect하지 못한다. Checkpoint가 더 동의하여 question 5%만 discordant하고 `Var(d) ≈ 0.05`라면 같은 question 500개로 2.8 pp를 detect한다. Memo에 주는 결과 두 개가 있다. `Var(d) = 0.1875`에서 1 pp difference를 80% power로 detect하려면 question 약 14,700개가 필요하다. 1 pp margin에서 group을 non-inferior로 선언하는 것은 다르고 약간 더 약한 requirement다. Interval lower bound가 −1 pp를 넘어야 하므로 `1.96·SE < 0.01`, `n ≈ 7,200`이다. `n = 500`에서는 half-width가 3.8 pp여서 이 역시 도달할 수 없다. 따라서 margin은 measured half-width에서 고르며, 이보다 좁은 margin은 delta가 0인 group을 포함한 모든 group에 `inconclusive`을 반환한다.

`K`를 올리면 conditional-variance term만 줄어든다. Miller uniform-difficulty example에서 `K = 1`에서 `K = 2`로 가면 total variance가 1/3 줄고, `K = 4`면 1/2 줄며, upper limit은 2/3다(§3.1). Variance를 줄이기 위해 sampling temperature를 낮추는 것은 대체물이 아니다. Paper 첫 single-token example에서는 expected score를 그대로 둔 채 conditional mean variance를 1/12에서 1/4로 바꾼다. Conditional mean이 [1/3, 1] uniform인 두 번째에서는 variance를 1/27에서 3/16으로 바꾸고 expected score도 2/3에서 3/4로 바꾼다(§3.3). Deployment가 쓰는 value로 temperature를 고정하고 `K`, `n`으로 precision을 사라.

### 5.4 Multiple comparison

Suite는 (task × slice)마다 test 하나를 만든다. 이러한 test가 20개이고 `α = 0.05`면 true null 아래 expected false rejection 수는 1이다. Holm step-down procedure를 쓴다. P-value를 `p_(1) ≤ … ≤ p_(K)`로 ascending sort하고 `p_(i)`를 `α/(K − i + 1)`과 비교하며 첫 failure에서 멈춘다. 그 이후 것은 모두 not rejected다. 이는 empirical claim이 아니라 standard procedure다.

**계산 예시.** Slice test `K = 6`의 p-value가 0.004, 0.011, 0.030, 0.040, 0.200, 0.500이다. Threshold는 0.05/6 = 0.00833, 0.05/5 = 0.0100, 0.05/4 = 0.0125, 0.05/3 = 0.0167, 0.05/2 = 0.025, 0.05/1 = 0.05다. 첫 test는 통과한다(0.004 < 0.00833). 둘째는 실패한다(0.011 > 0.0100). Procedure는 멈춘다. Uncorrected 0.05에서는 significant로 보였을 slice 두 개(`p = 0.030`, `0.040`)는 regression으로 보고하지 않는다. 이 sample size에서는 `δ_min`과 함께 memo에 unresolved로 보고한다.

`n < 30`인 slice는 `insufficient`로 label하고 correction이나 verdict에 참여하지 않는다. 이를 포함하면 K가 부풀고 다른 모든 test가 약해진다.

---

## §6 Forgetting report

**정의.** Forgetting report는 one pipeline의 consecutive checkpoint 사이 paired delta를, 뒤 stage가 훈련하지 않은 capability group에 대해 보여주는 table이다. Group마다 interval과 non-inferiority decision을 둔다.

**측정 가능한 문제.** 각 stage는 capability subset에서 정의한 objective를 최적화한다. 나머지 capability가 살아남는지는 stage 자신의 training signal이 답할 수 없는 empirical question이다.

**메커니즘.**

1. Checkpoint 세 개를 고정한다. Base, post-SFT([[ch-36]]), post-RL([[ch-46]])이다.
2. 세 checkpoint 모두를 같은 question, 같은 template, 같은 decoding으로 score한다.
3. Capability group마다 previous stage 및 base에 대한 paired delta와 interval을 계산한다.
4. Group 전체에 Holm을 적용한다(§5.4).
5. Paired interval이 §5.3에서 pre-register한 margin m을 넘으면 group을 `non-inferior`로 표시한다. Higher-is-better metric은 lower bound가 `−m`을 넘어야 하고, attack success rate/over-refusal처럼 lower-is-better metric은 upper bound가 `+m` 아래여야 한다. Direction은 reader가 기억해야 할 convention이 아니라 task field다.

**Reference number.** Tülu 3 8B development-suite score, SFT → DPO → Final RLVR(arXiv:2411.15124 Table 6)는 다음과 같다.

| Benchmark | SFT | DPO | Final | Final − SFT |
|---|---:|---:|---:|---:|
| GSM8K(8-shot CoT) | 76.2 | 84.3 | 87.6 | +11.4 |
| MATH(4-shot CoT, Flex) | 31.5 | 42.0 | 43.7 | +12.2 |
| IFEval(prompt loose) | 72.8 | 81.1 | 82.4 | +9.6 |
| AlpacaEval 2(LC win) | 12.4 | 33.5 | 34.5 | +22.1 |
| MMLU(0-shot CoT) | 65.9 | 68.7 | 68.2 | +2.3 |
| BigBenchHard(3-shot CoT) | 69.7 | 68.7 | 69.0 | −0.7 |
| HumanEval(pass@10) | 86.2 | 83.9 | 83.9 | −2.3 |
| PopQA(15-shot) | 29.3 | 29.3 | 29.1 | −0.2 |
| Safety(task 6개 average) | 93.1 | 87.2 | 85.5 | −7.6 |

마지막 row를 첫 row와 함께 읽어야 한다. GSM8K +11.4, IFEval +9.6을 만든 stage는 SFT checkpoint에서 safety average를 7.6점 낮췄다. Tülu 3는 이 number에 interval을 보고하지 않으므로 magnitude는 single-run point estimate다. Lab 자체 version은 paired interval을 넣으며, 이것이 이 장이 추가하는 부분이다. 대부분의 GSM8K/IFEval gain은 DPO stage(+8.1, +8.3)에 있고 RLVR stage는 +3.3, +1.3을 더한다. 전체 delta를 RL stage에 귀속하는 것은 이 table의 흔한 오독이다.

**Instruction-following case의 두 번째 reference.** Tülu-3-8B-DPO policy에서 IF-RLVR 뒤 IFEval/IFBench가 오르는 동안 AlpacaEval 2는 33.5 → 21.3, MMLU는 68.7 → 66.4다. Constraint를 제거한 같은 response의 GPT-4.1 judge score는 10점 중 7에서 6.4로 움직인다([[ifbench]] §5, Table 3). Constraint-only reward는 constraint satisfaction을 사고 general response quality를 쓴다. **결과(단일 연구)**이며, 논문이 제시한 remedy, 즉 verifiable reward에 reward-model score를 섞는 방법은 IF score를 낮추고 AlpacaEval 2를 31.6으로 올린다(부록 E).

**Report format**(`forgetting.md`):

```
| group        | base  | SFT   | RL    | Δ(RL−SFT) 95% CI      | margin | verdict        |
|--------------|-------|-------|-------|-----------------------|--------|----------------|
| knowledge    | 41.2  | 44.0  | 43.6  | −0.4 [−3.9, +3.1]     | −3.0   | inconclusive   |
| math         | 22.8  | 31.5  | 44.1  | +12.6 [+8.2, +17.0]   | −3.0   | improved       |
| math-perturb | 21.9  | 29.9  | 37.1  | +7.2 [+3.0, +11.4]    | −3.0   | improved       |
| instr-follow | 15.1  | 17.6  | 24.3  | +6.7 [+2.9, +10.5]    | −3.0   | improved       |
| safety-ASR   | 31.0  | 8.2   | 14.9  | +6.7 [+2.1, +11.3]    | +3.0   | REGRESSION     |
| over-refusal | 4.0   | 11.2  | 9.6   | −1.6 [−5.4, +2.2]     | +5.0   | non-inferior   |
```

`inconclusive`은 pass가 아니라 verdict다. Interval이 margin을 가로지르는 group은 non-inferior로 입증되지 않았으며 memo가 이를 명시한다.

---

## §7 Contamination gate

**정의.** Contamination은 test 중인 checkpoint의 evaluation item과 training data 사이 overlap이다. Gate는 score를 읽기 전에 실행하는 check이며 overlap rate가 명시한 threshold를 넘으면 reporting을 막는다.

### 7.1 Training data를 얻을 수 있을 때

이 lab이 복사하는 Tülu 3 procedure는 training set completion이 자주 regenerate되므로 prompt만 다룬다([[tulu-3]] §3.2).

1. Evaluation prompt와 training prompt를 tokenize한다.
2. Evaluation instance token은 해당 token을 포함한 8-gram을 두 instance가 공유하면 matched로 센다.
3. Evaluation instance token의 50% 초과가 같은 training instance와 matched면, evaluation instance가 training instance와 overlap한다고 본다.
4. Training set instance 중 하나라도 evaluation instance의 2% 초과와 overlap하면, 해당 training set은 evaluation에 대해 contaminated다.

**계산 예시.** Evaluation prompt가 token 40개이고 training instance와 verbatim span 25 token을 공유한다고 하자. 25-token span의 각 token은 shared 8-gram 하나에 속하므로 `25/40 = 62.5% > 50%`이고 instance는 flag된다. Shared span이 token 12개면 `12/40 = 30%`로 flag되지 않는다. Item 500개 held-out task에서 dataset-level rule은 `2% × 500 = 10` flagged item에서 trip한다.

**Threshold를 작은 number로 오인하지 않게 하는 base rate.** Tülu 3는 evaluation overlap >5%로 찾은 public dataset을 목록으로 든다(Table 37). Evol CodeAlpaca vs HumanEval 70.7%, LMSys Chat 1M vs AlpacaEval 46.5%, vs HumanEval 17.7%, vs MMLU 10.3%, vs GSM8K 8.9%, NuminaMath-TIR vs MATH 18.2%, DaringAnteater vs MATH 30.7%다. 이 set을 decontaminate하면 Evol CodeAlpaca 3.5%, NuminaMath-TIR 11.3%가 제거됐다(Table 8). 2% rule은 Tülu 3 자체 dataset-level decision rule이며 deduplication literature의 figure가 아니다.

### 7.2 MinHash를 올바르게 설정하여 candidate 생성

Multi-million-document training set에 대해 exhaustive 8-gram comparison은 감당할 수 없으므로 banded MinHash index로 candidate를 생성한 뒤 exact verification을 한다. Parameter를 주의 깊게 읽어야 한다. [[deduplicating-training-data]] §4.2는 5-gram shingle과 hash value `k = 9,000` signature를 사용하며, 이를 **bucket r = 450개에 bucket당 hash b = 20개로 나눈다**. Candidate probability는 다음과 같다.

```
P(candidate | Jaccard s) = 1 − (1 − s^b)^r ,   b = 20 hashes per bucket, r = 450 buckets
```

Candidate는 Jaccard > 0.8 **그리고** edit similarity > 0.8이어야 confirm한다. `EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j)/max(|x_i|, |x_j|)`다.

**계산 예시.** `b = 20`, `r = 450`이면 `s = 0.5`에서 `s^b = 9.5e−7`, `P = 0.0004`; `s = 0.6`에서 `P = 0.016`; `s = 0.7`에서 `P = 0.302`; `s = 0.8`에서 `P = 0.995`다. Curve midpoint는 `(1/r)^(1/b) = (1/450)^(1/20) = 0.737` 근처로 0.8 confirmation threshold보다 낮다. 그러므로 index는 true near-duplicate 거의 전부를 propose하고 exact check가 나머지를 제거한다.

b와 r을 바꾸면 behaviour가 뒤집힌다. Hash 450개 bucket 20개라면 midpoint는 `(1/20)^(1/450) = 0.993`이고 Jaccard 0.9 pair의 proposal probability는 `1e−15` 미만이다. 대부분 library API는 pair를 `(bands, rows_per_band)`로 받으므로, paper prose가 아니라 S-curve에 대해 assignment를 check해야 한다. [figures/lsh-band-curve.html](figures/lsh-band-curve.html)은 두 setting을 그리고 b/r을 움직여 살아남는 pair를 보게 한다.

```python
# harness/contamination.py
NUM_PERM, HASHES_PER_BUCKET, BUCKETS = 9000, 20, 450     # Lee et al. 2021 §4.2
assert HASHES_PER_BUCKET * BUCKETS == NUM_PERM
lsh = MinHashLSH(num_perm=NUM_PERM, params=(BUCKETS, HASHES_PER_BUCKET))  # (bands, rows)

def confirm(a_tokens, b_tokens):                          # exact verification
    j = jaccard(set(five_grams(a_tokens)), set(five_grams(b_tokens)))
    e = 1 - edit_distance(a_tokens, b_tokens) / max(len(a_tokens), len(b_tokens))
    return j > 0.8 and e > 0.8
```

여기서 MinHash는 near-duplicate case의 candidate generator다. Gate를 trip하는 decision rule은 §7.1의 8-gram token-coverage rule로 남는다. 이는 Tülu 3 저자들이 관심 둔 case, 즉 number만 바꾼 math problem처럼 trivially different한 instance를 잡기 때문이다(§3.2).

### 7.3 Training data를 얻을 수 없을 때

Closed checkpoint 또는 distillation teacher에서는 string matching을 할 수 없고 behavioural probe가 대체물이다. [[swe-bench-illusion]]은 probe 세 개를 실행했으며 이곳에는 둘이 적용된다. Repository name과 issue text만 주면 OpenAI/Anthropic model 10개는 SWE-Bench Verified instance에서 gold patch가 건드린 file을 60–76%에서 이름 붙였지만, benchmark 밖 repository의 comparable task 245개에서는 53% 미만이었다(§4.1.1–4.1.2). Function-reproduction 5-gram overlap은 SWE-Bench 34.9%, 다른 곳 13.9–18.2%에 이른다(§4.3). [[gsm1k]]은 다른 probe인 benchmark test item의 character별 log-likelihood를 썼고, 이는 benchmark-to-fresh-set gap과 Spearman 0.36으로 상관한다(§5.4). 둘 다 gap measurement며 comparable fresh set이 필요하다. §4가 그것을 만든다.

Gate verdict field는 값 세 개를 가진다. `clean`(check했으며 threshold 아래), `contaminated`(threshold 초과, score withheld), `unknown-training-data`(probe result를 보고하고 string check는 불가)다. `unknown-training-data`는 결코 pass로 세지 않는다.

---

## §8 Policy-level safety와 over-refusal

여기서 측정하는 것은 text를 classify할 수 있는 능력이 아니라 checkpoint 자체의 behaviour다. Moderation test set을 policy에 넣고 moderation gold label과 policy label을 비교해 score하면 checkpoint를 classifier로 측정하는 것이다. 이는 checkpoint가 harmful request에 comply하는지와 다른 quantity다. Lab은 [[ch-52]]를 따라 policy behaviour 두 개를 측정한다.

**Attack success rate.** HarmBench behaviour library의 prompt를 policy에 보내고, policy completion을 HarmBench release classifier로 score한다([[harmbench-data]]; [[harmbench-behavior-layering]]의 extract).

- Scoring: Fine-tuned test classifier는 human label과 93.2% agreement, separate validation classifier는 88.6% agreement다. Test classifier error는 41개, validation classifier error는 51개이며 26개가 겹친다(부록 B.2). Standard behaviour의 human judgment와 agreement는 HarmBench classifier 94.53%, GPT-4 89.8%, GPTFuzz 77.36%, AdvBench substring matching 71.14%다(Table 3). Paper가 substring matching을 금지하지는 않지만 standard behaviour에서 release classifier보다 23점 덜 정확하므로 쓰지 않는다.
- Generation length도 measurement의 일부다. Evaluation 중 생성한 token 수는 ASR을 최대 30% 바꾸며 HarmBench는 `N = 512`로 standardize한다(§3.2, 그림 2). 이를 `suite.yaml`에 기록한다.
- Optimization loop 안에서는 validation classifier를, gate에서는 test classifier만 쓴다. Paper는 test metric에 직접 최적화하는 것을 허용하지 않는다(부록 B.2).

**Over-refusal.** XSTest는 unsafe한 것처럼 보이는 safe prompt 250개와 unsafe contrast 200개를 보내고 response를 full compliance, full refusal, partial refusal로 label한다([[xstest]] §3, §4.2). Reference number는 양쪽 half가 모두 필요한 이유를 보인다. Original system prompt를 쓴 Llama-2-70b-chat은 safe prompt 38%를 fully refuse, 21.6%를 partially refuse하며 unsafe prompt는 99.5%를 refuse한다. System prompt 없는 Mistral-7B-Instruct-v0.1은 safe prompt의 `0.8% + 0.8%`만 refuse하고 unsafe prompt도 `23.5% + 12.5%`만 refuse한다(Tables 1, 2). Llama-2 system prompt를 제거하면 safe-prompt full refusal은 38% → 14%, unsafe-prompt full refusal은 99.5% → 97.5%다. 그러므로 system prompt는 measurement의 일부이며 `suite.yaml`에 pin해야 한다.

두 number는 (ASR, over-refusal) pair로 함께 보고한다. 하나가 개선되고 다른 하나가 악화한 checkpoint는 둘 다 개선한 것이 아니라 그 사이 trade-off를 따라 움직인 것이다. Automated refusal scoring에는 알려진 error가 있다. XSTest string matching은 “I'm sorry to hear about your hamster”를 refusal로 불렀고 zero-shot GPT-4 classification은 partial refusal에서 실패했다. 그러나 둘 다 manual annotation의 model ranking은 보존했다(§4.5).

---

## §9 Go/no-go memo

Memo는 course 나머지가 읽는 artifact다. 서로 다른 week memo 두 개를 diff할 수 있도록 fixed header를 갖는다.

Verdict rule은 순서대로 다음과 같다.

1. Contamination gate 또는 template-coverage gate가 실패하면 `NO-GO`다. Course가 훈련한 checkpoint에서 `unknown-training-data`도 여기에 포함된다.
2. Holm correction 뒤 capability group 하나라도 non-inferiority에 실패하면 `NO-GO`다.
3. Held-out average는 오르지만 pre-registered margin에서 group 하나 이상이 `inconclusive`이면 `NEEDS-FIX`다. `1.96·SE`보다 좁은 margin은 모든 group을 이 상태로 만든다. 이때 memo는 checkpoint verdict 대신 이를 해결할 sample size를 출력한다.
4. Held-out average change가 자신의 paired interval 안에 있으면 `INCONCLUSIVE`다.
5. 그 외에는 `GO`다. Held-out average가 zero를 제외하는 interval과 함께 오르고, 모든 group이 non-inferior다.

Rule 2가 이 memo를 benchmark table과 구분한다. 훈련한 task에서 12점을 얻고 safety에서 7점을 잃은 checkpoint는 pass하지 않는다. Rule의 순서가 있으므로 reader가 intuition으로 둘을 저울질할 필요가 없다.

---

```
# Eval memo: <run_id>            date: <YYYY-MM-DD>   suite hash: <sha>
## Verdict: GO | NO-GO | NEEDS-FIX | INCONCLUSIVE
## Checkpoint
  A = ch-36-sft <sha>     B = ch-46-rl <sha>     base = <sha>
## Gate (아래 score를 읽기 전에 모두 통과해야 함)
  contamination: clean | contaminated(task=<>, rate=<>) | unknown-training-data(probe=<>)
  template coverage: <task당 n_templates, min 3>
  power: task당 delta_min 및 CI half-width = <...>; non-inferiority margin m = <...>
## Held-out suite
  task별: worst-template score, template 평균, A 대비 paired delta [95% CI], n, cluster count
  held-out average, development average, 둘 사이 gap
## Forgetting report      (§6 table, Holm-corrected)
## Perturbation           checkpoint별 original, perturbed mean, sigma, gap in sigma
## Safety                 checkpoint별 (ASR, over-refusal) pair
## Open item              insufficient/inconclusive로 표시한 slice, 필요한 n 포함
## Recommendation         ship | re-run with n=<> | revisit ch-46 <arm> | investigate slice <>
```

## Negative sample과 negative feedback

이 lab은 training하지 않으므로 training gradient를 만들거나 적용하지 않는다. Negative를 두 방식으로 건드리며, course standard §6.1과의 구분이 중요하다.

**1. Harness 자체의 negative output은 gradient가 아니라 content로서의 negative다.** Slice label, prompt, completion을 가진 failing `Score` row 각각은 다음 data round의 input이다. [[ch-50]]의 failure bucket은 [[ch-36]] 또는 [[ch-46]]의 targeted prompt가 된다. Corrected target에 ordinary cross-entropy를 쓰거나 new rollout prompt로 쓴다. 이 lab은 likelihood를 아래로 밀지 않는다.

**2. Lab은 다른 곳에서 gradient로 쓴 negative의 결과를 측정한다.** [[ch-46]] RL stage는 negative advantage를 적용하고 preference stage는 rejected-response term을 적용한다. 그 알려진 side effect는 coverage loss와 calibration change이며, specific measurement에서만 보인다.

- **Coverage.** 적어도 task 하나에서 pass@1과 큰 k의 pass@k를 보고한다. Pass@1을 올리면서 큰 k의 pass@k를 낮추는 stage는 solution을 추가한 것이 아니라 probability mass를 집중시킨 것이다([[ch-46]]).
- **Abstention과 over-refusal.** Safe prompt의 XSTest full-refusal rate가 direct measurement다. Safety negative를 지나치게 아래로 밀면 이 rate가 오른다(§8).
- **Question 내부 diversity.** Question당 K sample에서 distinct final answer 수를 기록한다. K 전체에서 answer 하나로 collapse하는 것은 같은 pressure의 sampling-level signature다.
- **General response quality.** [[ifbench]] §5는 constraint를 제거한 response의 judge score로 이를 측정했다. Constraint-only RLVR 뒤 10점 중 `7 → 6.4`, AlpacaEval 2는 `33.5 → 21.3`이다.

**Effect 크기에 대한 정직성.** 이 measurement 중 어느 것도 gain/loss 중 어떤 비율을 negative term 자체에 귀속하지 않는다. Positive/negative contribution을 분리하려면 이 harness가 아니라 [[ch-46]]에서 만든 ablation run이 필요하다. Harness의 일은 interval을 붙여 side effect를 보이게 하는 것이다.

---

## 레시피

아래 행은 이 lab이 published harness에서 물려받은 setting이다. Lab value는 budget이 강제하는 곳에서만 다르다.

| 모델(exact release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 | Lab 값 | 차이 이유 |
|---|---|---|---|---|---|---|---|---|---|
| Tülu 3 8B / 70B | 8B, 70B | eval-gate | suite split | development suite(MMLU, PopQA, TruthfulQA, BBH, DROP, GSM8K, MATH, HumanEval(+), IFEval, AlpacaEval 2, Safety); unseen suite(MMLU-Pro, GPQA, AGIEval English, DeepMind Mathematics, BigCodeBench, IFEval-OOD, HREF); development 중 unseen score 미검사 | arXiv:2411.15124 §2.2, Table 3, Table 24 | 2026-09-17 검증됨 | Table 31: final checkpoint는 두 split average 모두 최고; coding unseen 11.5 → 7.4 | held-out task 6개(§2), inspection의 같은 rule | Tülu unseen suite에 safety task 없음; budget이 task 수 제한 |
| Tülu 3(all size) | n/a | eval-gate | decontamination | 8-gram prompt matching; token 50% 초과가 training instance 하나와 match하면 instance flag; eval instance 2% 초과 overlap이면 training set contaminated | arXiv:2411.15124 §3.2 | 2026-09-17 검증됨 | Table 37 overlap rate; Table 8 removal rate(Evol CodeAlpaca 3.5%, NuminaMath-TIR 11.3%) | 동일 | 없음 |
| Lee et al. 2021 NearDup | n/a | eval-gate | MinHash index | 5-gram shingle; `k = 9,000` hash를 `b = 20`인 `r = 450` bucket으로; Jaccard > 0.8 및 edit similarity > 0.8 confirm | arXiv:2107.06499 §4.2, App. A | 2026-09-17 검증됨 | App. A 그림 4는 0.9/0.9, b=20, r=40, k=800 setting도 보고 | 동일, candidate generation만 | Gate decision은 8-gram rule 유지 |
| HarmBench | n/a | eval-gate | generation length; scorer | generated token `N = 512`; fine-tuned Llama-2-13B-Chat test classifier; in-loop용 Mistral-7B validation classifier | arXiv:2402.04249 §3.2, §4.3, App. B.2 | 2026-09-17 검증됨 | 그림 2: token count로 ASR 최대 30% 변화; Table 3: 94.53% vs 71.14% substring agreement | 동일 | 없음 |
| XSTest | n/a | eval-gate | prompt; decoding; label | safe 250 + unsafe 200 prompt; temperature 0, max 256 token; full compliance / full refusal / partial refusal | arXiv:2308.01263 §3, §4.2, App. B | 2026-09-14 검증됨([[xstest]]) | Table 2: system prompt가 safe-prompt refusal 38% → 14% 이동 | 동일; checkpoint마다 system prompt pin | Lab checkpoint는 system prompt 하나 공유 |
| GSM-Symbolic protocol | n/a | eval-gate | template 수; sample; decoding | template 100 × sample 50 = item 5,000, item 100개 set 50개; 8-shot CoT, greedy | arXiv:2410.05229 §3.2 | 2026-09-17 검증됨 | §4.1: worst-to-best set gap >12%(Gemma2-9B) | template 100 × sample 30, set 30개 | Generation budget; set 30개 σ도 보고 |
| MMLU-Pro | n/a | eval-gate | prompt variant | prompt 24개; MMLU 4–5%(max 10.98%) 대비 spread 약 2%(max 3.74%) | arXiv:2406.01574 §6.3 | 2026-09-17 검증됨 | §6.3 | task당 template 3개 | 24 template × task 6개 × checkpoint 3개는 budget 밖; 3은 worst case bound |
| Miller 2024 | n/a | eval-gate | sample size | `n = (z_{α/2}+z_β)²(ω² + σ²_A/K_A + σ²_B/K_B)/δ²`; `δ = 0.03`, `α = 0.05`, `β = 0.20`, `ω² = 1/9`에서 `n ≈ 969` | arXiv:2411.00640 §5, 식 9 | 2026-09-17 검증됨 | §5: “new evals should contain at least 1,000 questions” | `n = 500`, task별 δ_min 및 interval half-width 인쇄 | Budget; margin은 사전 고정 대신 measured half-width에서 derive |
| IF-RLVR evaluation | 7-8B | eval-gate | decoding | temperature 0(DeepSeek-R1: 0.6 / top-p 0.95; o3: 1); prompt-level loose accuracy | arXiv:2507.02833 §3, App. G | 2026-09-14 검증됨([[ifbench]]) | 해당 없음 | IFBench에 동일 | 없음 |
| OLMo-2-era signal/noise study | 60M–32B | eval-gate | checkpoint noise window | final checkpoint 5개(small model), 1,000 step 간격 final 30개(13B target) | arXiv:2508.13144 §4.1, §4.2 | 2026-09-14 검증됨([[signal-and-noise-eval]]) | App. A.3.2 Table 2: 거의 모든 benchmark에서 n=5가 ±1σ 안 | 각 stage final checkpoint 3개 average | Lab이 저장하는 checkpoint가 더 적어 reduction을 limitation으로 기록 |

**작은 범용 run의 시작점.** Held-out task 6개에 `n = 500` question, task당 template 3개, multiple-choice는 `K = 1`, generative task는 `K = 4`, deployment value로 고정한 temperature, HarmBench `N = 512` token, pinned system prompt의 XSTest, Lee et al. MinHash index로 candidate를 생성하는 Tülu 3 8-gram/50%/2% contamination gate를 사용한다. 이 문단 number는 모두 위 `verified` row에서 왔다. Source가 이를 쓴 조건은 Llama 3.1 base의 8B–405B Tülu 3 split protocol, C4-scale corpus(document 360M)의 Lee et al. index, open 7B–13B chat model에서 HarmBench `N = 512`, zero conditional variance 및 `ω² = 1/9` assumption 아래 Miller `n ≈ 969`이다.

---

## 일반화 관점

**(a) 측정의 폭을 넓히는 것.**

- Development 중 held-out half를 절대 inspect하지 않는 suite split. Tülu 3 final 8B checkpoint는 두 split average에서 최고였고 data decision이 generalize했다는 근거다. 같은 table은 generalize하지 않은 coding(unseen 11.5 → 7.4)도 보인다([[tulu-3]] Table 31).
- Training constraint와 disjoint한 constraint set. IFBench unseen constraint 58개는 constraint를 따르는 model과 IFEval을 따르는 model을 구분하며 GPT-4.1/Claude 3.7 Sonnet은 50% 미만이다([[ifbench]] §1).
- Same reasoning template의 perturbation set은 skill을 고정하고 surface를 바꾼다([[gsm-symbolic]] §3.1).
- 해당 decision에 더 좋은 signal-to-noise를 갖는 benchmark. SNR subtask filtering은 MMLU decision accuracy +2.6%, AutoBencher +5%를 올렸다([[signal-and-noise-eval]] §5.1).

**(b) narrowing의 원인과 여기서 드러나는 방식.**

- Evaluation과 overlap하는 data training. Public mixture는 HumanEval overlap 70.7%, AlpacaEval 46.5%에 도달하며, benchmark-specific memorization은 file-path identification에서 60–76% 대 <53% gap으로 보인다([[tulu-3]] Table 37; [[swe-bench-illusion]] §4.1).
- Narrow verifiable reward 최적화. Constraint-only IF-RLVR은 AlpacaEval 2 `33.5 → 21.3`, MMLU `68.7 → 66.4`다([[ifbench]] Table 3).
- Over-refusal measurement 없는 safety tuning. Llama-2-70b-chat은 자기 system prompt 아래 safe prompt 38%를 refuse한다([[xstest]] Table 1).
- Template 하나만 보고. MMLU-like task의 3점 claim은 reasonable prompt 24개의 4–5% spread 안이다([[mmlu-pro]] §6.3).

**(c) 이 단계의 generality 측정.** Interval을 붙인 number 네 개다. Held-out average와 development average gap, task별 worst-template score, standard deviation 단위 perturbation gap, Holm correction 아래 forgetting table의 non-inferiority verdict다. Held-out average가 오르고 capability group이 하나도 non-inferiority에 실패하지 않을 때만 checkpoint가 predecessor보다 더 general하다고 기록한다.

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 |
|---|---|---|
| Tuning 중 held-out score consult | 모든 ablation에서 held-out/dev curve가 함께 움직임 | Gate timestamp 전 held-out task name을 experiment log에서 grep; `suite.yaml` suite hash는 first run보다 앞서야 함 |
| Paired data에 unpaired comparison | Interval이 넓고 regression이 significance에 도달하지 못하며 run마다 sign flip | 식 7로 재계산; checkpoint가 correlate하면 paired SE는 `sqrt(SE_A² + SE_B²)`보다 작아야 함 |
| Clustered question을 independent로 score | MGSM/DROP/RACE류에서 implausibly narrow interval | Clustered/naive SE ratio; 최대 3.05× 예상([[adding-error-bars-evals]] Table 4) |
| MinHash band/row transpose | known-dirty training set을 포함해 모든 task contamination rate가 0 근처 | `s = 0.8`에서 `1 − (1 − s^b)^r` 평가; 0.9 초과해야 함([figures/lsh-band-curve.html](figures/lsh-band-curve.html)) |
| Safety를 classification accuracy로 측정 | Checkpoint가 text labeler로 좋아지면 “safety” score가 오름 | Scorer input은 gold-label comparison이 아니라 behaviour prompt에 대한 policy completion이어야 함 |
| Generation length를 pin하지 않음 | Training change 없이 run 간 ASR 변화 | Run 간 `max_new_tokens` diff; HarmBench는 `N = 512` standardize([[harmbench-data]] §3.2) |
| Interval보다 좁은 non-inferiority margin | Delta가 0 근처인 group도 모두 `inconclusive` | Margin m과 half-width `1.96·SE` 비교; zero-delta group은 m이 이를 넘을 때만 pass |
| Correction 없이 `α = 0.05`로 slice test 20개 | 모든 comparison에 “regression” 1-2개가 있고 재현 안 됨 | Holm 적용, survive한 수 계산 |
| Interval을 좁히려고 temperature 낮춤 | Conditional variance는 사라지나 conditional mean variance가 오르고 mean도 이동 가능 | Deployment temperature에서 mean/SE 모두 재계산; Miller §3.3에서 total variance 1/12→1/4, second example expected score도 이동 |
| Model이 훈련한 same item에서 perturbation set 구성 | Known contamination checkpoint의 perturbation gap이 0 근처 | Template은 training mixture 밖 item에서 와야 하며 §7 gate로 check |

---

## 이해 점검

1. Tülu 3 coding row는 development task에서 `86.2 → 83.9`, held-out task에서 `11.5 → 7.4`다. 둘 다 coding을 재는데도 두 번째가 더 빨리 떨어질 수 있는 이유와 반대 pattern이 나오려면 무엇이 참이어야 하는지 설명하라.
2. §5.1에서 identical data인데 paired interval은 zero를 제외하고 unpaired는 제외하지 않는다. Pairing이 제거하는 quantity를 유도하고 pairing이 도움이 안 되는 case를 구성하라.
3. 동료가 “엄격하게” non-inferiority margin을 3 pp에서 0.5 pp로 줄이자고 한다. Interval half-width와 식 9로 lab verdict에 무슨 일이 생기며 tighter margin이 이 sample size에서 gate를 더 protective하게 만들지 않는 이유를 설명하라.
4. GSM-Symbolic은 model 25개 중 21개에서 GSM8K score가 perturbed distribution보다 standard deviation 하나 초과 높다고 찾는다. 이를 만드는 mechanism 두 개와 구별하는 measurement를 설명하라.
5. Detected overlap이 없다는 report를 둘 다 내는데 gate가 `unknown-training-data`를 `clean`과 다르게 취급하는 이유는 무엇인가? Probe result가 string check를 대체하려면 checkpoint가 무엇을 보여야 하는가?
6. §6 safety row는 같은 stage에서 −7.6, GSM8K는 +11.4다. Verifiable-reward RL stage에서 safety average가 낮아지는 causal path와, 이를 decoding-configuration difference와 구분할 이 harness의 measurement 하나를 말하라.
7. Held-out average +2.1 pp와 slice regression 4개를 보고했으며 그중 하나가 Holm correction 뒤 survive한다. 이 harness가 내는 memo verdict와 각 부분의 이유를 말하라.

---

## 연결

- 이전: [[ch-52]] — Safety Evaluation, Over-Refusal, and Red-Teaming. §8이 실행하는 behaviour library, released-classifier protocol, over-refusal contrast set을 제공한다.
- 다음: [[ch-54]] — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments. Scaled run의 acceptance test로 이 memo를 사용한다.
- [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split. SFT checkpoint와 저장 generation을 제공한다.
- [[ch-46]] — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention. RL checkpoint, ablation arm, negative-feedback section이 참조한 pass@k measurement를 제공한다.
- [[ch-47]] — Evaluation Harness and Suite Design for General Capability. `TaskSpec`이 구현하는 task-shape taxonomy를 정의한다.
- [[ch-48]] — Contamination Detection and Its Effect on Reported Scores. §7이 gate로 바꾸는 detection method를 유도한다.
- [[ch-49]] — Judge Models: Bias, Calibration, and Judge-Specific Overfitting. Judge-scored task를 suite에 추가하기 전에 필요하다.
- [[ch-50]] — Slice Analysis, Forgetting Slices, and Failure Bucketing. `forgetting.md`의 slice/failure-bucket vocabulary를 정의한다.
- [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions. §5, §9가 적용하는 paired bootstrap과 decision rule을 유도한다.
- [[ch-51a]] — Evaluating Agent Generality and Reliability. 여기서 생략한 agentic slice는 그 chapter protocol에 속한다.

---

## 출처

- [[tulu-3]] — development/unseen suite design(§2.2, Tables 3, 24), stage별 8B 결과(Table 6), skill별 dev-vs-unseen(Table 31), decontamination rule/overlap base rate(§3.2, Tables 8, 37).
- [[adding-error-bars-evals]] — paired SE(식 7), clustered/paired-clustered form(식 4, 8, Table 4), resampling/temperature guidance(§3), sample-size formula 및 n ≈ 969 example(식 9).
- [[gsm-symbolic]] — template instantiation protocol(§3.1–3.2), set 간 variance/GSM8K offset 21/25 model(§4.1), name-vs-number perturbation(§4.2), GSM-NoOp drop(§4.4).
- [[gsm1k]] — human-written item 1,205개, difficulty matching(§3.2), 최대 8% gap, Spearman 0.36 log-likelihood correlation(§5.4).
- [[ifbench]] — unseen constraint 58개/IFEval-IFBench gap(§1), IF-RLVR gain(Table 6), constraint-only reward general-quality cost(§5, Table 3).
- [[mmlu-pro]] — MMLU 대비 prompt sensitivity 24개(§6.3), benchmark construction.
- [[harmbench-data]] — behaviour/attack/scorer separation, released classifier agreement rate(Table 3, App. B.2), `N = 512` generation-length standardization(§3.2).
- [[xstest]] — safe 250 + unsafe 200 prompt, 3-way response taxonomy, reference refusal rate/system-prompt sensitivity(Tables 1, 2).
- [[deduplicating-training-data]] — MinHash NearDup banding parameter(`k = 9,000`, `b = 20`인 bucket `r = 450`)와 §7.2의 Jaccard/edit-similarity confirmation threshold(§4.2, App. A).
- [[signal-and-noise-eval]] — checkpoint-noise 정의/window(§3.1, §4), SNR-based subtask filtering(§5.1).
- [[swe-bench-illusion]] — training data가 없을 때 쓸 memorization probe(§4.1, §4.3).
- [[held-out-suite-tulu3]] — Tülu 3 split table 및 inspection rule extract.
- [[perturbation-robustness-sets]] — GSM-Symbolic/GSM1k construction 및 number extract.
- [[paired-statistics-and-power]] — symbol 정의를 갖춘 식 4, 7, 8, 9 extract.
- [[minhash-contamination-gate]] — 8-gram gate 및 정정한 banding arithmetic extract.
- [[harmbench-behavior-layering]] — policy-level ASR protocol/classifier agreement table extract.
