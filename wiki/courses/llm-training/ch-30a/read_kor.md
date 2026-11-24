<!-- chapter: ch-30a
     track: sft
     kind: content
     title: Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control
     deps: [ch-30]
     sources: [[catastrophic-forgetting-continual-finetuning]], [[scaling-laws-forgetting]], [[forgetting-via-implicit-inference]], [[spurious-forgetting]], [[finetuning-new-knowledge-hallucination]], [[finetuning-compromises-safety]], [[shallow-safety-alignment]], [[lora-learns-less-forgets-less]], [[sdft-self-distillation-finetuning]], [[wise-ft]], [[mitigating-alignment-tax-rlhf]], [[retaining-by-doing]], [[rls-razor]], [[rlhf-instructgpt]], [[rlhf-instructgpt-ptx]], [[adding-error-bars-evals]], [[agentic-finetuning-misalignment]], [[xstest]]
     figures: figures/paired-forgetting-delta.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30a — Fine-tuning에서의 망각과 정렬세: 측정과 통제

> **핵심 통찰.** 이 장에서 인용한 모든 망각 연구에서, 목표 점수를 올린 지도 미세조정은 그 목표가 포괄하지 않은 능력도 적어도 하나 낮췄다. 1B–7B 모델의 연속 instruction tuning에서는 MMLU, 추론, 독해 정확도가 낮아졌고([[catastrophic-forgetting-continual-finetuning]]), 무해한 Alpaca 데이터만 한 epoch 학습한 뒤 GPT-3.5 Turbo가 유해 요청을 거절하지 않고 수행한 비율은 5.5%에서 31.8%로 증가했으며([[finetuning-compromises-safety]]), 모델이 원래 모르던 사실을 맞추도록 학습하면 closed-book 사실 정확도도 낮아졌다([[finetuning-new-knowledge-hallucination]]). Llama 2 7B chat의 LoRA 연구에서는 망각이 fine-tuning loss의 선형 함수였고, 두 데이터셋에서 각각 R² 0.9450과 0.9736이었다. 따라서 early stopping과 adapter 크기는 실행을 trade-off 선에서 이동시킬 뿐, 그 선 밖으로 옮기지 못했다([[scaling-laws-forgetting]]). 측정된 이득이 있는 통제법들은 모두 미세조정 모델의 출력 분포를 시작점에 가깝게 유지한다. 일반 데이터 replay, LoRA, 자기 생성 target, 시작 가중치 쪽으로의 weight interpolation, pretraining-loss 혼합, 응답 초반 token별 제약, on-policy 학습 데이터가 그것이다. 이 공통 성질은 이 강의의 해석이며, 개별 통제법의 근거는 §4–§5에 있다.
>
> **지침.** fine-tuning 단계를 평가할 때, 그 단계의 데이터가 겨냥하지 않은 held-out suite에서 출력 checkpoint를 입력 checkpoint와 문항별로 비교하고 차이의 paired 95% confidence interval을 보고하라. aggregate accuracy가 비슷한 두 checkpoint도 문항의 3분의 1에서는 답이 다를 수 있고([[scaling-laws-forgetting]] §3.1), paired interval은 두 checkpoint가 공유하는 분산을 제거하기 때문이다([[adding-error-bars-evals]] §4.2). 시작 모델에 safety 행동이 있었다면, 무해한 데이터로 학습한 단계를 포함해 매 단계 뒤 harmful-request와 over-refusal 평가를 다시 실행하라. [[finetuning-compromises-safety]]가 시험한 모든 무해 데이터셋이 harmfulness rate를 올렸기 때문이다(Table 3). 충분한 rank의 LoRA나 on-policy sample로 목표 기술에 도달할 수 있다면, 높은 learning rate에서 외부 응답으로 full-parameter SFT를 하는 것보다 그것을 우선하라. code instruction tuning에서 HumanEval 정확도를 맞추었을 때 LoRA가 더 적게 망각했고([[lora-learns-less-forgets-less]] Tables S5–S6), GRPO의 평균 비목표 하락은 최대 3.4점인 반면 SFT는 38.5점까지 떨어졌기 때문이다([[retaining-by-doing]] Table 1). 그렇지 않으면 full fine-tuning을 유지하되 general-data replay나 self-distilled target을 더하고, 측정한 gain-versus-drop 곡선에서 운용 지점을 선택하라.

## 범용 모델에서 이 장이 중요한 이유

범용 모델은 pre-training → mid-training → SFT → preference optimization → RL → evaluation의 여러 단계로 만들어진다. pre-training 뒤의 모든 단계는 pre-training corpus보다 좁은 데이터를 학습한다. **망각(forgetting)** 은 학습 단계 뒤, 그 단계 이전에 모델이 지니고 있던 능력의 성능이 감소하는 일이다. **정렬세(alignment tax)** 는 정렬 절차가 유발한 망각이다. InstructGPT는 GPT-3와 비교했을 때 RLHF 모델이 SQuAD, DROP, HellaSwag, WMT 2015 French-to-English에서 보인 퇴행을 이 용어로 부른다([[rlhf-instructgpt-ptx]] §1).

측정 가능한 문제에는 두 부분이 있다. 첫째, 단계 자체의 validation set은 그 손실을 드러내지 못한다. [[wise-ft]]에서 CLIP ViT-B/16을 10 epoch 미세조정했을 때 learning rate `3·10⁻⁵`와 `3·10⁻⁶`은 ImageNet accuracy가 0.3점밖에 차이 나지 않았지만, distribution shift 아래 accuracy는 최대 8점 차이 났다(§4, Fig. 3). 둘째, 손실은 그 단계가 전혀 평가하지 않는 능력 전반, 즉 지식, 추론, safety, calibration, 출력 다양성에 퍼져 있다.

ch-00은 general capability를 목표로 삼지 않은 과제들에 대한 profile로 정의했다. ch-30은 SFT 설계 선택을 정했다. 이 장은 그 선택이 나머지 profile에 치르는 비용을 측정하고, 그 비용을 낮추는 통제를 다룬다. mixture 설계는 ch-30b, weight merging은 ch-30c, continued pretraining은 ch-32a, KL-controlled RLHF는 ch-38, SFT와 RL 비교는 ch-38a에서 다룬다. §6의 보고 형식은 lab ch-36, ch-46, ch-53에서 쓴다.

## §1 시작 checkpoint를 기준으로 망각 측정하기

**정의.** 망각 측정은 단계의 입력 checkpoint(기준점, reference)와 출력 checkpoint를, 그 단계가 목표로 하지 않은 능력을 측정하는 평가에서 비교한다. **held-out suite** 는 해당 단계의 학습 데이터에 포함되지 않았고, 그 단계의 설정을 고르는 데도 쓰지 않은 평가 묶음이다.

**문제.** aggregate score 차이는 문항 수준의 변화를 숨긴다. [[scaling-laws-forgetting]](§3.1)은 Llama 2 chat 7B를 OpenOrca 예시 6,400개로 200 step 미세조정하고 ARC-Challenge를 평가했다. base model은 54.1%, 미세조정 모델은 52.0%를 얻었지만, 미세조정 모델이 base model과 같은 답을 낸 문항은 67.2%뿐이었다. 답 일치율을 `a = 0.672`라 하면 답이 바뀐 문항 비율은 `1 − a = 0.328`이다. 답의 변화는 손실(base는 정답, fine-tuned는 오답), 이득(base는 오답, fine-tuned는 정답), 또는 서로 다른 두 오답 사이의 전환 중 하나다. 순변화가 `−0.021`이고 오답→다른 오답 전환이 없다고 가정하면, 손실은 `(0.328 + 0.021)/2 = 17.45%` 문항, 이득은 `(0.328 − 0.021)/2 = 15.35%` 문항이다. 오답→다른 오답 전환이 하나라도 있으면 두 비율 모두 작아지므로, 이 수치는 상한이다(도출값이며 논문은 이 분해를 보고하지 않는다). 따라서 순하락 2.1점은 각각 2.1점보다 여러 배 클 수 있는 손실 비율과 이득 비율의 차이일 수 있다.

**절차.**

1. 기준점을 고정한다. 즉, 해당 단계에 들어간 checkpoint를 기준으로 한다. 여러 단계를 연쇄했다면 원래 pretrained base와의 비교도 함께 보고한다.
2. capability slice별 held-out suite를 고른다. slice는 knowledge, reasoning and math, code, instruction following, long context, multilingual, tool use, factuality, safety, over-refusal을 포함한다. 학습 데이터와 겹치는 문항은 제거한다(ch-53).
3. 두 checkpoint를 동일한 prompt template, few-shot exemplar, decoding setting, judge version으로 실행한다.
4. 두 checkpoint 각각의 문항별 score를 저장한다.
5. 평균 차이, paired standard error, 95% interval, 이득 문항 수와 손실 문항 수를 계산한다.
6. 분포 수준 측정값 하나를 추가한다. 일반적으로 고정 prompt set에서 reference에 대한 KL divergence, 또는 fine-tuned model과 reference model의 예측 사이 cross-entropy를 쓴다([[scaling-laws-forgetting]] §3.1).

**수식.** `n`개 문항의 문항 `i`에서 두 score를 `s_ref,i`, `s_ft,i`라고 하고 `d_i = s_ft,i − s_ref,i`라고 하자.

```
Δ = d̄ = (1/n) Σ_i d_i
SE_paired   = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )           (Eq. 7)
CI_95%      = Δ ± 1.96 × SE                                  (Eq. 5)
SE_unpaired = sqrt( p_ref(1 − p_ref)/n + p_ft(1 − p_ft)/n )   (Eq. 2 in SE_A−B of §4.1)
```

식 번호는 [[adding-error-bars-evals]]를 따른다. 여기서 `Δ`는 망각 delta이며 음수는 손실을 뜻한다. `SE`는 standard error이고, `p_ref`, `p_ft`는 0/1 score일 때 두 accuracy이며, `1.96`은 양측 95% normal quantile이다. 원 출처의 저자 Miller는 분산 관계를 `Var(paired) = Var(unpaired) − 2 Cov(x_A, x_B)/n`으로 쓴다. 두 checkpoint가 어떤 문항을 어렵다고 보는지 일치할수록(공분산이 양수일수록) paired interval은 더 좁아진다(§4.2).

§6 보고서에서 쓰는, 저장한 문항별 score로 위 계산을 수행하는 코드다(강의 코드이며 출처에서 가져온 코드는 아니다).

```python
import math

def paired_delta(ref_scores, ft_scores, z=1.96):
    """ref_scores, ft_scores: 같은 문항 순서로 저장한 문항별 점수."""
    n = len(ref_scores)
    d = [f - r for r, f in zip(ref_scores, ft_scores)]
    mean_d = sum(d) / n
    var_d = sum((x - mean_d) ** 2 for x in d) / (n - 1)   # d_i의 sample variance
    se = math.sqrt(var_d / n)                             # Eq. 7
    losses = sum(1 for x in d if x < 0)
    gains = sum(1 for x in d if x > 0)
    return mean_d, se, (mean_d - z * se, mean_d + z * se), losses, gains
```

**계산 예.** held-out suite에 0/1 score 문항 `n = 500`개가 있다고 하자. 기준 모델은 300개를 맞혀 `p_ref = 0.60`이다. SFT 뒤에는 원래 맞힌 300개 중 45개가 오답이 되고(losses), 원래 틀린 200개 중 20개가 정답이 된다(gains). 따라서 fine-tuned model은 275개를 맞혀 `p_ft = 0.55`이다.

| | ft 정답 | ft 오답 |
|---|---|---|
| ref 정답 | 255 | 45 |
| ref 오답 | 20 | 180 |

- `Δ = (20 − 45)/500 = −0.050`이다.
- 각 `d_i`는 45개 문항에서 `−1`, 20개 문항에서 `+1`, 435개 문항에서 `0`이다. 따라서 `Σ d_i² = 65`이고, `Σ (d_i − d̄)² = 65 − 500 × 0.05² = 63.75`, sample variance는 `63.75/499 = 0.12776`이다.
- `SE_paired = sqrt(0.12776/500) = 0.01598`이고, `CI = −0.050 ± 0.0313 = [−0.081, −0.019]`다. 이 interval은 0을 포함하지 않는다.
- `SE_unpaired = sqrt(0.60 × 0.40/500 + 0.55 × 0.45/500) = 0.03122`이고, `CI = −0.050 ± 0.0612 = [−0.111, +0.011]`다. 이 interval은 0을 포함한다.

같은 데이터는 paired analysis에서는 검출된 손실을, unpaired analysis에서는 검출되지 않은 손실을 뒷받침한다. 같은 순변화가 95개 손실과 70개 이득에서 나왔다면 `Σ d_i² = 165`, `SE_paired = 0.0256`이므로 paired interval은 `[−0.100, +0.0002]`로 넓어져 0을 포함한다. 첫 번째 예와 같은 비율에서 문항 수가 4배라면(`n = 2000`, 정답 1,200개, 손실 180개, 이득 80개), `SE_paired = 0.0080`, `SE_unpaired = 0.0156`이 되고 두 interval 모두 0을 제외한다. interactive figure [paired-forgetting-delta.html](figures/paired-forgetting-delta.html)에서는 `n`, 기준 accuracy, 손실·이득 수를 바꿔 두 interval과 answer agreement rate가 갱신되는 모습을 볼 수 있다.

**조건과 한계.** paired formula는 두 checkpoint에 같은 문항을 같은 방식으로 채점할 것을 요구한다. 출력이 sampling되는 경우에는 `d_i`를 계산하기 전에 문항마다 여러 sample을 평균낸다([[adding-error-bars-evals]] §3.1). 문항들이 하나의 passage나 template을 공유하면 표준오차는 독립 문항이라는 가정 아래 계산한 값보다 커진다. DROP에서는 clustered standard error가 naive standard error의 3.05배였다(§2.2, Table 4). `k`개 slice를 5% 수준에서 시험하고 실제 변화가 하나도 없다면, 0을 제외하는 interval의 기대 개수는 `0.05 × k`다. ch-51은 multiple-comparison control을 가르치고 ch-53은 이를 적용한다. 바뀐 문항 수가 적으면 1.96 multiplier 뒤의 normal approximation은 정확도가 낮아진다.

**함의.** 망각 보고서는 하나의 aggregate difference가 아니라 slice마다 paired interval을 명시한다.

## §2 근거: 연속 instruction tuning에서의 망각과 망각 법칙

### 2.1 1B–7B 모델의 연속 instruction tuning

[[catastrophic-forgetting-continual-finetuning]]은 BLOOMZ(1.1B–7.1B), mT0(1.2B, 3.7B), LLaMA-7B, Alpaca-7B를 다섯 생성 과제 `Simp → Emdg → InqQG → Exp → HGen` 순서로 학습했다. learning rate는 `2e-5`, 과제마다 3 epoch였으며, 각 과제 뒤 MMLU, 여섯 추론 세트, RACE를 평가했다(§3–§4). 망각 지표(Eq. 1)는 전체 `N`개 checkpoint에 걸친 상대 하락을 평균낸다.

    FG_i = (1/|E_i|) Σ_{e ∈ E_i} (1/N) Σ_{m=1..N} (R_o^e − R_m^e) / R_o^e × 100%

`R_o^e`는 평가 요소 `e`의 초기 결과이고, `R_m^e`는 `m`번째 과제 뒤 결과이며, `E_i`는 하나의 평가 세트에 속한 요소들의 집합이다.

**계산 예.** 한 평가 요소가 `R_o = 40`에서 시작해 세 과제 뒤 각각 38, 35, 30점을 얻었다고 하자. `FG = (2/40 + 5/40 + 10/40)/3 = (0.050 + 0.125 + 0.250)/3 = 14.2%`다. 마지막 checkpoint만 비교한 상대 하락은 25%다. BLOOMZ-7.1b의 domain knowledge에서 Table 4는 `33.08 → 25.61`을 제시한다. 이는 마지막 결과만으로 계산한 상대 하락 `22.6%`(도출값)지만, 보고된 `FG`는 18.37이다. `FG`가 중간 checkpoint들의 상대 하락과 네 MMLU 요소의 하락을 모두 평균내기 때문이다.

**근거(결과, 단일 연구).** 모든 모델에서 domain knowledge, reasoning, reading comprehension의 `FG`가 0보다 컸다(Tables 4, 6). reading comprehension의 손실이 가장 컸다. BLOOMZ-7.1b의 `FG`는 reading 26.75, domain knowledge 18.37, reasoning 13.62였다(Table 4). domain-knowledge `FG`는 모델 크기와 함께 증가했다. BLOOMZ-1.1b, 1.7b, 3b, 7.1b에서 각각 9.54, 10.72, 14.63, 18.37이었다(§5.2). 저자들은 크기가 커질수록 초기 점수는 더 높지만 마지막 점수는 비슷하다는 점으로 이 경향을 설명한다(해석). LLaMA-7b의 domain-knowledge `FG`는 34.57, Alpaca-7b는 18.14였고(Table 6), 저자들은 이를 일반 instruction tuning이 이후의 망각을 줄인다는 신호로 해석한다.

**조건과 한계.** 시험한 과제 순서는 하나뿐이고, 7.1B보다 큰 모델은 없으며, 선행 연구에서 정한 constant 또는 cosine learning rate를 썼고(§4.2 footnote 7), 입력 길이는 512 token이었다. 크기 경향은 7B를 넘는 범위에서 시험하지 않았다.

### 2.2 학습 시간이 길수록 망각이 증가한다

[[lora-learns-less-forgets-less]]는 **forgetting average**로 망각을 측정한다. 이는 HellaSwag, ARC-Challenge, WinoGrande의 평균 accuracy이며, 값이 높을수록 망각이 적다는 뜻이다(§3.3). Magicoder-Evol-Instruct-110K에서 Llama-2-7B를 full fine-tuning했을 때, 이 평균은 epoch 1 뒤 0.595에서 epoch 16 뒤 0.414로 낮아졌다(Table S6). 저자들은 instruction fine-tuning이 continued pretraining보다 더 많이 망각하고, code가 math보다 더 많이 망각하며, 학습 시간이 길수록 망각이 커진다고 보고한다(§4.2).

### 2.3 fine-tuning loss의 함수로서의 망각

[[scaling-laws-forgetting]]은 Llama 2 7B chat을 OpenOrca 또는 2023년 9월 뉴스 기사 100개로, rank 8–256의 LoRA를 써서 260 step 동안 미세조정했다(batch 32, context 512). 그리고 WikiText-103에서 base model의 다음-token 예측에 대한 cross-entropy를 망각 `L_f`로 측정했다(§3). 적합식(Eq. 4)은 다음과 같다.

    L_f(L_ft) = −c · L_ft + s

`L_ft`는 fine-tuning loss이고, `c`, `s`는 적합한 상수다. OpenOrca에서는 `c ≈ 1.7334`, `s ≈ 2.0481`, News에서는 `c ≈ 1.0615`, `s ≈ 3.1285`였으며, `R²`는 각각 0.9450, 0.9736이었다(§4.1, Fig. 2).

**계산 예.** OpenOrca에서 `L_ft`를 0.76에서 0.66으로 낮추면, 예측된 `L_f`는 전에는 `−1.7334 × 0.76 + 2.0481 = 0.731`, 후에는 `−1.7334 × 0.66 + 2.0481 = 0.904`다. 적합에 들어간 LoRA 실행(rank 8–256) 안에서는, 어떤 rank나 step 수로 그 감소를 만들었든 fine-tuning loss가 0.1 줄 때마다 forgetting loss가 0.173 증가한다.

두 loss 모두 학습 parameter 수 `P`와 step 수 `N`에 대한 shifted power law도 따른다(Eq. 5–6). `L_f`를 `P`, `N`의 함수로 적합하면 `R²`는 0.9598과 0.9769까지 올라간다. 저자는 conventional fine-tuning에서는 early stopping이나 더 적은(또는 더 많은) parameter를 조정하는 방식으로 망각을 피할 수 없다고 결론낸다(해석). 또한 [[catastrophic-forgetting-continual-finetuning]]에서 더 큰 모델이 더 많이 망각한 이유를, 그 모델들이 더 낮은 fine-tuning loss에 도달했기 때문이라고 설명한다(해석이며 크기를 가로질러 검증한 결과는 아니다). 한계는 7B 모델 하나, 망각 데이터셋 하나, 260 step이라는 점이다. 이 선형 법칙은 OpenOrca의 다른 방법으로는 부분적으로만 옮겨갔다. 두 적합식은 LoRA rank 1000, 2500, attention-only rank-64 LoRA, full fine-tuning은 예측했지만, `L_f(L_ft)`는 IA3와 상위 3개 layer만 조정한 경우의 망각을 과소추정했다. 이 실행 전체에서 `R²`는 0.1851이었고, `L_f(P, N)`의 0.8714보다 낮았다(App. B, Fig. 6).

**함의.** 하나의 방법과 데이터셋 안에서는 목표를 얼마나 깊게 적합했는지가 망각량을 정한다. trade-off를 바꾸려면 방법이나 데이터를 바꿔야 하며, §5가 그 통제법을 다룬다.

## §3 메커니즘: capability 손실, task inference, alignment 손실

점수 하락에는 적어도 세 가지 설명이 있으며, 각 설명은 서로 다른 시험 결과를 예측한다.

### 3.1 task inference가 fine-tuning 분포 쪽으로 이동한다

**정의.** [[forgetting-via-implicit-inference]]는 출력을 다음 mixture로 모형화한다. `w_θ(X, y) = g_θ(X, y) w_disc(X, y) + (1 − g_θ(X, y)) w_cont(X, y)` (Eq. 7). 여기서 `g_θ`는 주어진 prompt가 fine-tuning task family에 속한다는 모델의 implicit 추정치이고, `w_disc`와 `w_cont`는 두 capability다. fine-tuning은 `w_cont`를 제거하지 않고도 다른 task의 prompt에서 `g_θ`를 높일 수 있다.

**근거.** synthetic regression으로 학습한 22.4M-parameter transformer에서 400 fine-tuning step은 target이 아닌 task의 loss를 높였다. fine-tuning 분포 밖에 있으면서도 그 분포에 가까운 task가 가장 크게 영향을 받았다(§1, Fig. 5). prompt가 fine-tuning task와 덜 닮아 보이도록 label의 scale을 다시 맞추자, 잃었던 해의 일부가 회복됐다(Fig. 6). LLM에서 instruction tuning은 English prompt의 in-context-learning behavior를 LLaMA의 92.00%에서 Alpaca의 35.25%로 낮췄다. 하지만 French에서는 29.00 point만, Leetspeak에서는 1.50 point만 낮췄다(Table 1). Code LLaMA는 English에서 LLaMA-2 대비 XNLI accuracy가 8.36 point 하락했고 Spanish에서는 0.77 point 올랐다(Table 2). 상태: 이 회복 결과들이 뒷받침하는 **해석**이다. 저자들은 모델이 실제로 이 방식으로 계산하는지 시험할 수 없다고 명시한다(§2.7).

### 3.2 spurious forgetting: task alignment는 잃고 knowledge는 보존한다

**정의.** [[spurious-forgetting]]은 `Task Performance = Task Alignment + Underlying Knowledge`로 분해한다(§1). spurious forgetting은 task format에 맞추는 alignment를 잃어서 생긴 점수 하락이다.

**근거.** 200,000명으로 된 synthetic biography dataset에서 새 task를 학습한 첫 150 step 안에 Task 0 accuracy는 거의 100%에서 약 10%로 떨어졌다. 그러나 recovery test, 즉 Task 0의 절반을 한 epoch fine-tuning하고 나머지 절반에서 평가하는 시험은 거의 100%를 유지했고 마지막에는 96%였다(§3.2). LLaMa-2-7B-Chat의 AdvBench safety는 identity-shifting example 10개를 10 epoch 학습한 뒤 100%에서 0%가 됐지만, attack 전 모델이 생성한 refusal example 10개로 약 99%까지 회복됐다(§2). 같은 identity-shifting 조건에서 아래쪽 6개 layer를 freeze하면 jailbreak rate는 sequential fine-tuning의 99.80%에 비해 1.15%로 유지됐다(Table 2). 저자들은 이 결과가 safety alignment에서 spurious forgetting을 보인다고 말하며, jailbreak attack에 대한 방어책은 아니라고 명시한다(§5.3).

**측정상의 귀결.** recovery test는 alignment 손실과 knowledge 손실을 분리한다. 영향을 받은 데이터의 서로 겹치지 않는 절반으로 한 epoch fine-tuning한 뒤 다른 절반의 accuracy가 이전 수준으로 돌아오면 alignment만 잃은 것이고, 낮은 상태에 머물면 knowledge가 사라진 것이다.

### 3.3 알지 못하는 사실을 맞추면 hallucination이 증가한다

**정의.** [[finetuning-new-knowledge-hallucination]]은 PaLM 2-S의 각 closed-book QA example을 SliCK으로 label한다. 10개의 random 4-shot prompt 전체에서 greedy decoding이나 sampling이 한 번도 답을 만들지 못하면 **Unknown**이고, 그렇지 않으면 세 가지 Known category 중 하나다(§3, Fig. 2). `|D| = 6142`, learning rate는 `1e-5`, batch는 128, 학습은 50 epoch다(App. E).

**근거.** Unknown example은 Known example보다 더 느리게 fit됐다(§4.3). test accuracy는 Eq. 1을 따른다.

    Accuracy = β_0 + β_kn · N_kn/|D| + β_unk · N_unk/|D|

`N_kn`과 `N_unk`는 모델이 fit한 Known 및 Unknown training example 수다. in-distribution fit은 `β_0 = 36.9`, `β_kn = 7.3`, `β_unk = −8.3`, `R² = 0.86`이고, out-of-distribution relation은 각각 36.2, 3.2, −3.0, `R² = 0.95`다(Table 1).

**계산 예.** dataset의 50%가 Known이고 50%가 Unknown이라고 하자. early stopping 시점에 모델은 `D`의 45%를 Known으로, 5%를 Unknown으로 fit한다. 그러면 `36.9 + 7.3 × 0.45 − 8.3 × 0.05 = 39.8`이다. convergence에서는 두 절반을 모두 fit하므로 `36.9 + 3.65 − 4.15 = 36.4`다. 3.4 point의 순손실은 Unknown 항 `−8.3 × 0.45 = −3.7`을, 추가로 fit한 Known example이 `+7.3 × 0.05 = +0.4`만큼 일부 상쇄한 결과다.

convergence에서 Unknown example만으로 학습한 경우 점수는 25.8이었고, early stopping에서는 37.5였다(Table 2). Unknown example을 filtering하면 early stopping과 convergence의 결과는 거의 같아졌다(§4.2). 상태: **결과(단일 연구)**이며, 모델 하나와 QA format 하나에서 나온 결과다.

**함의.** 모델이 생성할 수 없는 사실을 단언하는 answer target은 다른 질문에도 근거 없이 답하도록 가르친다. 범용 모델에서는 known fact로 채점하는 knowledge benchmark가 잡아내지 못하는 calibration 및 factuality 손실로 나타난다.

## §4 fine-tuning으로 인한 safety 약화

### 4.1 무해한 fine-tuning도 refusal을 낮춘다

[[finetuning-compromises-safety]]는 benign fine-tuning을 한 epoch 수행한 뒤 harmfulness rate, 즉 policy-violating prompt 330개 중 GPT-4가 완전히 harmful하다고 판정한 비율을 측정했다. GPT-3.5 Turbo에서는 Alpaca로 학습했을 때 5.5% → 31.8%, Dolly로 학습했을 때 4.5% → 23.9%였다. Llama-2-7b-Chat에서는 Alpaca가 0.3% → 16.1%, Dolly가 0.6% → 12.1%, LLaVA-Instruct가 0% → 18.8%였다(Table 3). Llama-2-7b-Chat을 Alpaca로 학습할 때 learning rate가 `5e-5`이면 batch 16 / 32 / 64 / 128에서 각각 46.4% / 37.9% / 31.5% / 34.2%였고, learning rate `2e-5`에서는 각각 23.6% / 20.6% / 15.8% / 16.1%였다(Table 12). LoRA도 safety를 보호하지 못했다. 각 방법의 공식 권장 설정(LoRA: LR `10⁻⁴`, batch 16; full: LR `2×10⁻⁵`, batch 128)에서 Alpaca 뒤 harmfulness rate는 LoRA 25.2%, full-parameter tuning 16.1%였다(Table 11, App. F). GPT-3.5 Turbo의 Alpaca run에 refusal example 500개를 섞으면 31.8%는 19.7%로 낮아졌지만, 시작점 5.5%까지는 돌아가지 않았다(Table 4). Alpaca 뒤 MT-Bench는 8.00에서 6.68로 하락했다(Table 8). [[scaling-laws-forgetting]]도 hand-check한 AdvBench prompt 50개에서 같은 방향을 보였다. base model은 32개를 거절했지만 News LoRA model은 24개, OpenOrca model은 16개만 거절했다(§4.2). 상태: 두 source에서 방향이 재현된 **Replicated** 결과다.

### 4.2 얕은 safety alignment

**정의.** [[shallow-safety-alignment]]는 safety alignment가 base model의 분포를 주로 출력 첫 몇 token에서만 바꾸는 경우를 shallow하다고 부른다(§2).

**근거.** 정렬되지 않은 Llama-2-7B base에 `"I apologize, but I cannot"`을 prefill하면 harmfulness rate가 68.6%에서 2.1%로 낮아졌다(Table 1). harmful response에서 aligned model과 base model 사이의 per-token KL은 첫 위치들에서 가장 높다(Fig. 1). Llama-2-7B-Chat을 harmful example 100개로 fine-tuning(LR `2×10⁻⁵`, batch 64)하면, harmful prompt에 harmful하게 답한 비율인 attack success rate(ASR)는 2, 4, 6 step 뒤 각각 1.5%에서 22.4%, 76.4%, 87.9%로 올랐다. per-token loss와 gradient norm도 첫 위치에서 가장 컸다(Fig. 3).

**통제 1: safety recovery example.** 목적함수 `α · E[−log π_θ(r | x, h_≤k)] + (1 − α) · E[−log π_θ(y′ | x′)]`(Eq. 2)은 harmful response `h`의 처음 `k` token 뒤에 refusal `r`을 학습시키며, initial model에서 distill한 response `y′`를 가진 Alpaca prompt `x′`로 이를 anchor한다. `α = 0.2`이고 절반의 경우 `k = 0`, 나머지 경우 `k`를 `[1, 100]`에서 uniform하게 뽑으면, 5-token prefill 아래 ASR은 42.1%에서 2.8%로 낮아졌고 AlpacaEval win rate는 51.8%에서 49.5%로 움직였다(Table 2, §3.1).

**통제 2: token-wise constrained SFT.** loss(Eq. 3)는 `E[ −Σ_t (2/β_t) log σ(β_t log(π_θ(y_t|·)/π_aligned(y_t|·))) ]`다. 그 gradient는 cross-entropy gradient에 `w_t = 2σ(β_t Δ_t)`를 곱한 것과 같으며, 여기서 `Δ_t = log π_aligned(y_t|·) − log π_θ(y_t|·)`, `σ`는 logistic function이다(Eq. 5). 초기화 때는 `Δ_t = 0`, 따라서 `w_t = 1`이다.

**계산 예.** 학습이 `log π_θ(y_t)`를 aligned model보다 4 nat 높였다고 하자(`Δ_t = −4`). 논문의 `β_1 = 0.5`에서는 `w = 2σ(−2) = 0.238`이다. 위치 2–5에서 `β_t = 2`이면 `w = 2σ(−8) = 0.00067`이므로 그 위치의 update는 거의 멈춘다. 위치 5 뒤에서 `β_t = 0.1`이면 `w = 2σ(−0.4) = 0.803`으로 plain SFT에 가깝다.

이 `β_t` 값으로 Llama-2-7B-Chat을 Samsum에 fine-tuning했을 때 ASR은 3.2%였고 standard SFT는 23.4%였으며, ROUGE-1은 각각 50.1과 51.7이었다. GSM8k에서는 constrained 방식 accuracy가 37.4%, standard SFT가 41.7%였다(Table 3). uniform `β = 2.0`은 harmful-example attack에서 ASR 0.5%를 만들었지만 GSM8k accuracy는 2.1%였다(Table 4). 따라서 낮은 ASR과 utility 보존을 함께 얻는 것은 위치마다 `β_t`를 다르게 설정하는 데 달려 있다. 상태: 두 7B 모델에서의 **Result (single study)**다.

### 4.3 Agentic fine-tuning

[[agentic-finetuning-misalignment]]는 Llama-3.1-8B-Instruct를 benign web-navigation demonstration으로 fine-tuning했다. WebArena-Lite success는 2.42%에서 22.42%로 올랐지만, harmful-task attack success는 32.88%에서 64.38%로 올랐고 refusal은 26.03%에서 6.85%로 낮아졌다(Table 1). 시험한 8개 model-domain pair 모두에서 attack success가 증가했다(Tables 1–2). 따라서 benign agentic demonstration으로 학습한 stage도 chat data로 학습한 stage와 같은 safety 재평가가 필요하다.

**§4 전체의 함의.** safety 측정에는 두 면이 있다. harmful prompt에 대한 refusal은 harmful prompt와 닮은 safe prompt에 대한 refusal과 함께 보고해야 한다. XSTest에서 Llama-2-70b-chat은 original system prompt를 쓴 상태에서 safe prompt 250개 중 38%를 fully refuse했다([[xstest]] Tables 1–2). ch-52가 두 측정을 모두 다룬다.

## §5 근거가 있는 완화책

| 제어 방법 | 무엇을 제약하는가 | 근거(설정, 수치) | 한계 | 전체 설명 |
|---|---|---|---|---|
| 일반 데이터 replay | training distribution | LLaMA-7b MMLU-human: replay 없음 34.72 → 26.8, Alpaca sample 10,000개를 섞으면 30 | replay token 4억 개는 1.2T-token corpus의 0.033% | ch-30b, ch-32a |
| 더 낮은 learning rate | step size | Llama-2-7b-Chat harmfulness: `2e-5`에서 16.1%, `5e-5`에서 34.2%(batch 128) | target gain이 낮아짐 | ch-01 |
| LoRA | update rank | Llama-2-7B code IFT에서 HumanEval이 거의 같은 0.498 vs 0.497일 때 forgetting average(높을수록 좋음)는 LoRA 0.631, full 0.446 | CPT에서 학습량이 낮고, math IFT에서는 망각이 비슷하며 safety 보호 없음 | ch-30 |
| SDFT | target distribution | OpenFunctions 뒤 GSM8K: vanilla 21.5, SDFT 29.1, seed 29.4 | 주 실험은 Llama-2-7b-chat의 LoRA `r = 8`; full FT·13B·Llama-3-8B는 각각 한 dataset으로만 확인 | ch-31 |
| weight interpolation | weight 상의 거리 | CLIP shift 평균 68.6 → `α = 0.5`에서 76.9; RLHF model-averaging Pareto front는 비교 방법 거의 전부보다 위 | vision 근거이며 LLM 비율은 model별로 선택 | ch-30c |
| pretraining-loss mixing | pre-training text에서의 output distribution | InstructGPT는 1.3B에서 `γ ≥ 20`이면 regression 회복 | pre-training과 유사한 data 필요 | ch-38 |
| early-token constraint | token별 deviation | Samsum ASR 23.4% → 3.2% | GSM8K accuracy 낮아짐 | 이 장 §4.2; 평가는 ch-52 |
| on-policy data | data source | Llama 3.1 8B MMLU target: drop SFT 38.5, GRPO −0.2 | reward 또는 verifier 필요 | ch-38a |

### 5.1 일반 데이터 replay

Replay는 stage loss에 general data를 더한다. 즉 `L = L_target + λ · L_general`이며, `λ`는 general data의 weight 또는 batch 내 비중이다. [[catastrophic-forgetting-continual-finetuning]]에서는 continual tuning에 Alpaca sample 10,000개를 섞어 LLaMA-7b의 MMLU-human을 시작점 34.72%에서 26.8%까지 떨어뜨리는 대신 30%로 유지했다(§5.4). [[spurious-forgetting]]에서는 old data의 20%를 replay해 이후 training 중 model을 old task에 다시 맞췄다(§5.2, Fig. 3b). [[mitigating-alignment-tax-rlhf]]에서는 RLHF data의 최대 4배인 pre-training data 4억 token을 replay했을 때 reading comprehension에서는 model averaging보다 좋았으나 commonsense QA와 translation에서는 졌다(App. C.1).

**계산 예.** 1.2T-token pre-training corpus에 대해 replay token 4억 개는 `4×10⁸ / 1.2×10¹² = 0.033%` coverage다(App. C.1은 "about 0.03%"라고 쓰고, 같은 version의 §2는 "~0.01%"라고 쓴다). 저자들은 0.033% sample에는 충분히 들어가지 않은 ability가 있기 때문에 이득이 부분적이라고 해석한다(Interpretation). 상태: replay가 망각을 줄인다는 결과는 **Replicated**다([[catastrophic-forgetting-continual-finetuning]], [[spurious-forgetting]]). 다만 [[mitigating-alignment-tax-rlhf]]는 0.033% coverage에서 이득이 benchmark에 따라 달라짐을 보인다.

### 5.2 Learning rate와 update 크기

[[retaining-by-doing]]은 Self-SFT에서 “SFT가 높은 target performance에 도달하려면 보통 높은 learning rate가 필요하고, 이는 흔히 심한 forgetting을 대가로 한다. 더 작은 learning rate는 forgetting을 줄이지만 epoch를 더 늘려도 같은 target performance에 도달하지 못한다”고 보고한다(§2.3, Fig. 3). 위의 [[finetuning-compromises-safety]] Table 12와 [[wise-ft]] §4는 learning rate가 non-target behavior를 바꾼다는 점을 보인다. WiSE-FT의 CLIP ViT-B/16에서는 ImageNet accuracy 차이가 0.3 point인 두 learning rate가 distribution shift 아래에서는 최대 8 point 차이를 냈다. 단, WiSE-FT 사례는 prior task의 망각이 아니라 robustness를 비교한다. §2.3을 고려하면 lower learning rate의 이득 일부는 target 자체를 덜 fit하기 때문이라고 해석할 수 있다(Interpretation). ch-01은 learning rate와 step count를 initial weight에서의 거리를 정하는 요인으로 다룬다.

### 5.3 LoRA

**정의.** LoRA는 weight matrix `W_pretrained ∈ R^{d×k}`를 freeze하고 low-rank perturbation을 학습한다. `W_finetuned = W_pretrained + γ_r A B`, 여기서 `A ∈ R^{d×r}`, `B ∈ R^{r×k}`, rank `r ≪ d, k`, 그리고 hyperparameter `α`에 대한 scale은 `γ_r = α/r`이다([[lora-learns-less-forgets-less]] §2).

**근거.** Magicoder-Evol-Instruct-110K에서 epoch 4의 LoRA `r = 256`은 HumanEval 0.498 및 forgetting average 0.631에 도달했다. epoch 8의 full fine-tuning은 HumanEval 0.497 및 0.446이었다(Tables S5-S6). Weight decay와 attention dropout은 full fine-tuning만큼 학습했고 똑같이 망각했다(§4.5). Full fine-tuning은 base model보다 unique HumanEval solution을 더 적게 만들었고, LoRA는 두 경우의 중간이었다(§4.5). **한계.** Continued pretraining에서는 LoRA가 어떤 rank에서도 full fine-tuning 성능에 도달하지 못했다(§4.1). MetaMathQA epoch 16에서는 forgetting average가 거의 같았다(LoRA `r = 256` 0.567, full fine-tuning 0.559, Table S8). 저자들은 math IFT에서는 full fine-tuning의 learning-forgetting trade-off가 더 낫다고 판단한다(§4.3). LoRA도 여전히 망각하고([[scaling-laws-forgetting]]), [[finetuning-compromises-safety]]에서는 safety loss를 낮추지 못했다(Table 11).

### 5.4 Self-distillation fine-tuning(SDFT)

**정의.** SDFT는 각 target response `y`를, fine-tuning 중인 model이 prompt와 original response로 생성한 rewrite `ỹ`로 바꾼다. 추출한 answer가 `y`와 같을 때만 `ỹ`를 유지하고, 그 결과로 학습한다([[sdft-self-distillation-finetuning]] Eq. 3-5).

    ỹ ~ f_θ(y | c, x, y_orig);   ỹ′ = ỹ if Extract(ỹ) = y_orig, else y_orig;   L_SDFT = −log f_θ(ỹ′ | c, x)

`c`는 task context, `x`는 instruction, `y_orig`는 dataset response이며, `Extract`는 final answer를 반환한다. 대부분의 experiment에서 쓴 rewrite prompt는 다음과 같다(Fig. 3).

```
Below are an instruction that describes a task along with a reference answer. Using
the reference answer as a guide, write your own response.

### Instruction:
{instruction}

### Reference Answer:
{original response}

### Response:
```

**근거.** Llama-2-7b-chat(LoRA `r = 8`)을 OpenFunctions로 fine-tuning했을 때 HumanEval은 vanilla 9.8, SDFT 15.2, seed 13.4였고, GSM8K는 21.5, 29.1, 29.4였으며 target OpenFunctions는 34.8, 36.6이었다(Table 1). GSM8K fine-tuning 뒤 AlpacaEval win rate는 vanilla 23.38, SDFT 66.73, seed 66.04였다(Table 2). **한계.** 주 표는 query와 value에만 LoRA `r = 8`을 적용한 7B chat model 하나를 쓴다. §5.3 Table 5는 Llama-2-7b-chat full fine-tuning(GSM8K), Llama-2-13b-chat LoRA(GSM8K), Llama-3-8B-Instruct LoRA(OpenFunctions)를 각각 한 fine-tuning dataset으로 비교했고, 보고된 모든 column에서 SDFT가 vanilla fine-tuning 이상이었다. Seed는 보고하지 않았고 rewrite quality는 seed model의 quality를 넘을 수 없다(Interpretation).

### 5.5 시작 weight 쪽으로의 weight interpolation

**정의.** WiSE-FT는 `(1 − α) · θ_0 + α · θ_1` weight를 사용한다. `θ_0`는 fine-tuning 전 model, `θ_1`은 fine-tuning 후 model, `α ∈ [0, 1]`이다([[wise-ft]] Eq. 1). **근거.** End-to-end fine-tuning한 CLIP ViT-L/14@336px는 ImageNet 86.2, 다섯 shift 평균 68.6이었지만, `α = 0.5`에서는 각각 86.8, 76.9였다(Table 1). Shift 아래 8.3 point 상승이다(derived). 저자들은 domain knowledge가 없을 때 `α = 0.5`를 추천한다(§4). LLM alignment에서 [[mitigating-alignment-tax-rlhf]]는 OpenLLaMA-3B를 ShareGPT로 instruction tuning한 뒤 HH-RLHF에서 Rejection Sampling Fine-tuning(RSF: prompt마다 response `n`개를 sample하고, reward가 가장 높은 것을 남겨 fine-tuning한 뒤 반복)으로 aligned했다(§3). RLHF 전후 weight를 평균낸 Pareto front는 early stopping, L1/L2 penalty, LoRA, knowledge distillation, stochastic moving averaging 가운데 “거의 모든 다른 방법을 능가한다”(§4.1). Transformer를 `K`개 layer group으로 나누고 group별 averaging ratio를 주는 Heterogeneous Model Averaging(HMA)은 front를 더 옮겼다(§6, Fig. 5). Post-RLHF model의 weight를 `α`로 둘 때, “`α = 0.2`는 alignment performance를 해치지 않고 alignment tax를 일관되게 완화할 수 있다”(§6). **한계.** WiSE-FT는 image model만 시험했고, LLM ratio는 그 논문의 자체 figure에서 골랐다.

### 5.6 Pretraining-loss mixing(PPO-ptx 미리보기)

InstructGPT objective(Eq. 2)는 KL-penalized reward에 `γ E_{x∼D_pretrain}[log π(x)]`를 더한다. `γ`는 pretraining loss coefficient다([[rlhf-instructgpt-ptx]] §3.5). Run은 `γ = 27.8`을 쓰고 RL episode보다 8배 많은 pretraining example을 사용했다(App. C.4). 1.3B에서는 `γ ≥ 20`이 public-NLP regression을 회복했지만, `γ = 0`에서 KL coefficient를 default의 100배로 높이는 것은 회복하지 못했다(App. E.6). PPO-ptx는 여전히 DROP, SQuADv2, translation에서 GPT-3보다 낮았다(§4.2). ch-38이 이 objective를 유도한다.

### 5.7 On-policy data(미리보기)

**정의.** On-policy data는 현재 학습 중인 model이 생성한 sample이다. REINFORCE와 GRPO는 sample response의 log-probability를 reward에 비례해 높이는 policy-gradient 방법이다. GRPO는 raw reward 대신 같은 prompt의 sample group에 대한 reward 평균을 빼고 group standard deviation으로 나눈 값을 쓴다(ch-37, ch-40). **근거.** [[retaining-by-doing]]은 target gain과 non-target drop을 보고한다. Drop은 다른 target task들과 MATH, WildJailbreak, WildGuardTest에서의 평균 accuracy 감소다(§2.1-§2.2). Table 1에서 filtered Llama-3.3-70B-Instruct response로 MMLU를 학습한 Llama 3.1 8B Instruct는 SFT에서 target gain 11.1, non-target drop 38.5였고 GRPO에서는 gain 14.6, drop −0.2였다. IFEval에서 drop은 SFT 27.8, GRPO 3.4였다. 같은 Llama 3.1 8B에서 REINFORCE의 drop은 IFEval, MMLU, Countdown에서 각각 7.7, −0.1, −0.8이었다. KL term을 뺀 GRPO는 Llama model의 IFEval을 제외하면 KL term을 넣은 GRPO와 같은 방식으로 동작했다(§4.1). 저자들은 차이를 on-policy data에 돌린다(Interpretation). Qwen 2.5 1.5B와 7B Instruct에서 epoch 시작마다 target을 다시 생성하는 Iterative-SFT는 IFEval과 MMLU에서 SFT의 target accuracy에 도달하면서 “mild to no forgetting만 보였다”(§4.2, Fig. 7). [[rls-razor]]는 new task `τ`에서의 forgetting을 `E_{x∼τ}[KL(π_0 ‖ π)]`로 예측한다고 제안한다. `π_0`는 base policy, `π`는 fine-tuned policy다. Quadratic fit은 toy setting ParityMNIST에서 `R² = 0.96`, Qwen 2.5 3B-Instruct에서 0.71이었다(§4). 하지만 [[retaining-by-doing]]은 그 setting에서 이 KL link가 “항상 성립하지는 않는다”고 보고한다. KL과 drop의 Pearson correlation은 0.52였고, Self-SFT와 SFT 사이에서는 KL이 더 커도 항상 forgetting이 더 크지는 않았다(App. A.5). 상태: on-policy data가 forgetting을 줄인다는 것은 **Replicated**이고, new task에서의 KL을 predictor로 쓰는 것은 **Open question**이다.

## §6 이후 lab에서 쓰는 forgetting report

lab ch-36, ch-46, ch-53 및 capstone ch-59는 학습한 모든 checkpoint에 이 report를 붙인다.

1. **Checkpoint.** stage의 입력 checkpoint와 원래 pretrained base를 정확한 identifier와 함께 기록한다.
2. **Suite.** §1에서 열거한 각 slice마다 held-out suite를 하나 이상 둔다. 어떤 suite로 setting을 선택했는지(development), 어떤 suite를 보고에만 썼는지(held-out), 그리고 decontamination 방법을 명시한다.
3. **Protocol.** template, shot 수, decoding setting, judge model과 version을 두 checkpoint에서 완전히 같게 한다. 형식에 민감한 task에는 template을 적어도 3개 쓴다(ch-53).
4. **Slice별 통계.** `Δ`, `SE_paired`, 95% interval, gain한 item 수, loss한 item 수, agreement rate를 함께 제시한다.
5. **Safety pair.** harmful request의 ASR과 safe look-alike prompt에서의 over-refusal을 함께 측정한다.
6. **Factuality.** reference model이 정확히 답했던 fact에 대한 closed-book accuracy와 abstention rate를 보고한다.
7. **Distribution distance.** general text에서 reference prediction까지의 cross-entropy와 target prompt에서 reference까지의 KL을 기록한다.
8. **Diversity.** 한 generation task에서 `k`개 sample 중 unique output 수, 또는 큰 `k`에서의 pass@k를 보고한다.
9. paired interval 전체가 0보다 아래인 모든 slice에는 **recovery test**를 둔다. slice의 절반으로 짧게 fine-tuning하고, 나머지 절반에서 평가한다(§3.2).
10. **Trade-off curve.** 하나의 control(learning rate, epoch, `α`, replay share 중 하나)에 대해 최소 두 setting을 사용해 target gain과 평균 non-target drop을 함께 그린다.
11. **Seed.** configuration마다 사용한 seed 수를 기록한다. seed를 하나만 실행했다면 1이라고 명시한다.
12. **Decision rule.** evaluation을 실행하기 전에 slice마다 non-inferiority margin을 작성한다.

## Negative samples와 negative feedback

이 과정은 negative signal의 용도를 네 가지로 구별한다(ch-43a). **negative marginal value**는 positive target으로 사용하면 성능을 떨어뜨리는 sample이므로 제거하는 경우다. **negative as content**는 failure를 input 또는 수정된 target 안에 넣고 일반 cross-entropy로 학습하는 경우다. **negative as conditioning**은 inference 때는 쓰지 않는 control token 아래에서 failure를 학습하는 경우다. **negative as gradient**는 sample의 likelihood를 명시적으로 낮추는 경우다. 이 장의 control은 첫째, 둘째, 넷째를 사용하며, conditioning으로서의 negative는 사용하지 않는다.

**Negative의 출처.** (a) harmful prompt의 refusal target 및 harmful prefix `h_≤k`를 넣은 safety recovery example. jailbroken model과 aligned model이 label을 붙인다([[shallow-safety-alignment]] App. A.3). (b) model이 산출하지 못하는 사실을 포함한 fine-tuning example. sampling으로 Unknown label을 붙인다([[finetuning-new-knowledge-hallucination]] §3). (c) reward function([[retaining-by-doing]] §2.2) 또는 answer mismatch([[sdft-self-distillation-finetuning]] Eq. 4)가 label을 붙인 잘못된 self-generated response. (d) GRPO의 zero-reward rollout.

**현재 관행이 하는 일.** (a)는 **negative as content**다. harmful prefix가 context에 들어가고 refusal을 cross-entropy로 학습한다. (b), (c)는 **negative marginal value**다. 해당 example을 filter out하거나 SDFT에서는 original response로 바꾼다. (d)는 **negative as gradient**다. 평균보다 낮은 reward가 negative advantage를 준다.

**메커니즘.** logit `z`에 대한 softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이며, 여기서 `y`는 target token, `p_j`는 token `j`의 확률이다. Content로 쓰인 negative는 refusal token에 이 positive-target gradient를 적용한다. harmful continuation의 확률은 normalization을 통해서만 낮아진다. (d)만 sampled sequence에 negative sign을 붙여 gradient를 적용하여 그 sequence에서 probability mass를 빼낸다. 그 mass는 현재 확률에 비례해 다른 token으로 가므로, 이미 likelihood가 낮은 sample을 누르면 mass 대부분은 가장 likelihood가 높은 대안으로 이동한다. ch-43a가 이 경우를 유도한다.

**근거.** GPT-3.5 Turbo의 100-example harmful fine-tuning에 refusal example 100개를 섞자 harmfulness가 91.8%에서 23.0%로 낮아졌다([[finetuning-compromises-safety]] Table 4). Recovery example은 40-token prefill ASR을 57.0%에서 4.5%로 낮췄다([[shallow-safety-alignment]] Table 2). Unknown example을 filter하면 early stopping과 convergence의 gap이 매우 작아졌다([[finetuning-new-knowledge-hallucination]] §4.2). REINFORCE와 GRPO는 advantage estimate가 달라도 유사하게 낮은 forgetting을 보였다([[retaining-by-doing]] Table 1). 인용한 source 중 improvement 가운데 negative 자체가 차지하는 비율을 측정한 것은 없다(not reported).

**Control.** Refusal training은 positive utility term으로 anchor한다([[shallow-safety-alignment]] Eq. 2에서 `α = 0.2`). Position별 deviation은 `β_t`로 bound한다. Negative를 gradient로 쓸 때는 on-policy로 유지한다.

**진단.** Harmful prompt와 safe look-alike prompt에서의 refusal을 분리해 기록한다. Reference까지의 position별 KL을 기록한다. GRPO에서는 advantage 부호로 나눈 loss와 probability 통계를 기록한다(ch-43a).

**일반성에 미치는 효과.** Refusal data는 over-refusal을 올릴 수 있다([[xstest]]). Filter하지 않은 Unknown target은 hallucination을 높인다. Filter한 self-generated data는 training distribution을 model 가까이에 유지했고 SDFT와 Iterative-SFT 결과에서 forgetting을 줄였다.

## 레시피

모든 row는 명시된 위치에서 2026-09-15에 읽었다.

| 모델(정확한 release) | 크기 | Stage | Setting | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama-2-7B → Magicoder-Evol-Instruct-110K, LoRA | 7B | SFT | rank; α; peak LR; schedule; global batch; max length; optimizer; epochs | r = 16, 64, 256; α = 2r; 2e-4 (r = 16, 64), 1e-4 (r = 256); cosine, duration의 0.1 warmup, alpha_f 0.01; 192 sequences; 4096; decoupled LionW (0.9, 0.95); 1, 2, 4, 8, 16(각각 별도 run) | arXiv:2405.09673v2 §4.1, App. A | verified 2026-09-15 | epoch별 Tables S5-S6; r = 256에서 Fig. S3 α sweep(α = 512가 최고) |
| Llama-2-7B → MetaMathQA, full FT와 LoRA | 7B | SFT | peak LR; global batch; length | full 1e-5; LoRA 1e-4 (r = 16, 64), 5e-5 (r = 256); 768; 1024 | arXiv:2405.09673v2 App. A | verified 2026-09-15 | Tables S7-S8; 2 epoch에서 App. B LR sweep |
| Llama-2-7b-chat SDFT와 vanilla baseline | 7B | distill-SFT | adapter; LR; batch; data와 epochs | query와 value의 LoRA r = 8; 1e-4 cosine to 0; 8; 2,000 examples × 2 epochs(Alpaca, Dolly, MagiCoder), 20,000 × 2(OpenHermes), full train set × 5(GSM8K, OpenFunctions), × 2(LIMA) | arXiv:2402.13669v2 App. A | verified 2026-09-15 | Tables 1-3(SDFT 대 vanilla); hyperparameter ablation은 보고되지 않음 |
| Llama-2-7B-Chat, token-wise constrained SFT | 7B | SFT | β_t; LR; optimizer; batch; warmup; epochs | β_1 = 0.5, β_t = 2 (t = 2-5), β_t = 0.1 (t > 5); 2×10⁻⁵; AdamW (0.5, 0.999); 64; linear, 첫 10 steps; 3(benign set) | arXiv:2406.05946v1 §4.2, App. A.5 | verified 2026-09-15 | Table 4 uniform-β ablation; App. C Table 5 warmup ablation |
| Llama-2-7B-Chat, safety recovery augmentation | 7B | SFT | data; mix; prefix length k; LR; epochs | 256 `(x, h, r)` triplet + distilled response가 있는 Alpaca prompt; α = 0.2(batch당 16 + 64); k = 0 w.p. 0.5, 그렇지 않으면 Uniform[1, 100]; 2×10⁻⁵ AdamW; D_H에서 10 epochs | arXiv:2406.05946v1 §3.1, App. A.3 | verified 2026-09-15 | Table 2(ASR); α 또는 k ablation은 찾지 못함 |
| Llama-2-7b-Chat, benign Alpaca SFT | 7B | SFT | data; epochs; LR; batch | 50K Alpaca(safety sample 제거); 1; 2e-5; 128 | arXiv:2310.03693v1 §4.4, App. G.1 | verified 2026-09-15 | Table 12 LR × batch; Table 13 epochs |
| InstructGPT PPO-ptx (1.3B, 6B, 175B) | 1.3B-175B | RL | γ; pretraining examples; KL reward coefficient β | 27.8; RL episode의 8 ×; 0.02 | arXiv:2203.02155v1 App. C.4(γ, 8×, "β = 0.02"), App. E.6 | verified 2026-09-15 | 1.3B에서 Fig. 33 γ sweep(γ ≥ 20이면 회복); γ = 0에서 Fig. 34 β sweep; App. E.7 Fig. 36: optimal β는 "around 0.01 and 0.02" |
| OpenLLaMA-3B, HH-RLHF 위 RSF, model averaging | 3B | merge | post-RLHF model의 weight; HMA parts | α = 0.2; K = 3 | arXiv:2309.06256v4 §6 | verified 2026-09-15 | Figs. 3, 5, 16 α sweep; Fig. 5(right) K = 3, 6, 9 |
| OpenLLaMA-3B, RSF, experience replay | 3B | RL | replay amount | 1.2T-token corpus에서 batch당 RLHF data의 최대 4 ×(400M tokens) | arXiv:2309.06256v4 App. C.1 | verified 2026-09-15 | Fig. 7(replay 대 averaging, benchmark 3개) |
| Llama-3.1-8B-Instruct와 Qwen-2.5-7B-Instruct, SFT와 GRPO | 8B, 7B | SFT, RL | LR; schedule; batch; epochs; KL coefficient; group size | 5e-6; cosine, warmup ratio 0.03; 128(IFEval, MMLU), 64(Countdown); 2; 0.05; 5 | arXiv:2510.18874v3 App. A.3 | verified 2026-09-15 | Fig. 3 Self-SFT LR와 epochs; Fig. 6 KL 0.05 대 0 |
| PaLM 2-S, closed-book QA SFT | not reported | SFT | examples; LR; batch; dropout; epochs; stop rule | 6142; 1e-5 fixed; 128; 0.05; 50; best development epoch(5-10) | arXiv:2405.05904v3 §4, App. E | verified 2026-09-15 | Fig. 6은 LR 1e-4에서도 trend를 반복 |
| BLOOMZ, mT0, LLaMA, Alpaca, continual instruction tuning | 1.1B-7.1B | SFT | LR; schedule; batch; length; epochs; replay | 2e-5; constant(BLOOMZ, mT0), cosine(LLaMA, Alpaca); device당 4; 512; task당 3; Alpaca sample 10,000개 | arXiv:2308.08747v5 §4.2, §5.4 | verified 2026-09-15 | ablation 보고 없음(setting은 prior work를 따름, footnote 7) |
| CLIP ViT-L/14@336px, WiSE-FT | not reported | merge | mixing coefficient | α = 0.5 | arXiv:2109.01903v3 §4 | verified 2026-09-15 | Table 1(α = 0.5 대 optimal α); App. B |

**작은 general-purpose run의 시작점.** Magicoder-Evol-Instruct-110K(72.97M token) 크기의 single-domain instruction set으로 fine-tuning하는 7B base model이라면, 검증된 LoRA row는 모든 module에 `r = 256`, `α = 512`, peak LR `1e-4`, 10% warmup을 둔 cosine schedule, 4096-token sequence에서 global batch 192를 지지한다. 그 run(Llama-2-7B, Magicoder, GPU 32개)에서 epoch 4는 full fine-tuning의 최고 HumanEval과 일치했고 forgetting average는 0.446 대신 0.631이었다. 같은 table에서 epoch에 따라 forgetting이 올랐으므로 각 epoch 뒤 forgetting report를 평가한다. Safety를 유지해야 하는 7B chat model이라면 constrained-SFT row(`β_1 = 0.5`, position 2-5에서 `β_t = 2`, 그 뒤 `0.1`, LR `2×10⁻⁵`, batch 64, 3 epochs)가 검증된 setting이며 Samsum, SQL Create Context, GSM8k에서 측정됐다. 3B model에서 preference 또는 RL stage 뒤에는, stage input과 `α = 0.2`로 average한다(post-RLHF model에 주는 weight). 이는 OpenLLaMA-3B RSF, DPO, PPO sweep(Figs. 3, 5, 16)에서 저자들이 추천한 ratio다.

## 일반화 관점

**(a) Breadth를 높이는 것.** 한 비교에서는 일반 instruction tuning을 이미 받은 model이 이후 task tuning에서 덜 잊었다(LLaMA-7b FG 34.57 대 Alpaca-7b 18.14, domain knowledge에서 [[catastrophic-forgetting-continual-finetuning]] Table 6; 저자 해석). Early stopping에서 HighlyKnown 대신 MaybeKnown example으로 학습하면 MaybeKnown test question accuracy가 60.1에서 69.9로 올랐고 HighlyKnown accuracy는 98.7에서 98.4로 움직였다([[finetuning-new-knowledge-hallucination]] Table 2). Weight interpolation은 target score를 유지하면서 distribution shift 아래 accuracy를 올렸다([[wise-ft]] Table 1). On-policy target은 non-target accuracy를 시작 값 가까이에 유지했다([[retaining-by-doing]] Table 1).

**(b) Narrowing 또는 forgetting을 유발하는 것.** Target을 더 맞추는 것(§2.3), 더 많은 epoch([[lora-learns-less-forgets-less]] Table S6), 더 높은 learning rate([[finetuning-compromises-safety]] Table 12), model 자신의 분포에서 먼 external-model response([[sdft-self-distillation-finetuning]] Table 1; [[retaining-by-doing]] Table 1), Unknown fact를 target으로 쓰는 것([[finetuning-new-knowledge-hallucination]] Table 1), 그리고 첫 response token에 safety가 의존하는 model의 모든 fine-tuning([[shallow-safety-alignment]] Fig. 3)이 원인이다. 여기에는 agentic data도 포함된다([[agentic-finetuning-misalignment]] Table 1). Full fine-tuning은 HumanEval output diversity도 낮췄다([[lora-learns-less-forgets-less]] §4.5).

**(c) 이 stage에서 이를 측정하는 법.** Held-out slice에서 95% interval을 붙인 paired per-item delta([[adding-error-bars-evals]] Eq. 5, 7), reference answer와의 agreement 및 reference prediction까지의 cross-entropy([[scaling-laws-forgetting]] §3.1), harmful-request ASR과 over-refusal의 동시 측정([[finetuning-compromises-safety]], [[xstest]]), training 전에 알고 있던 fact의 closed-book accuracy([[finetuning-new-knowledge-hallucination]]), drop이 감지된 slice의 recovery test([[spurious-forgetting]] §3.2), 하나의 control에서 gain-versus-drop curve를 쓴다. 알려진 측정 오류도 있다. Paired interval이 같은 item의 0을 제외해도 unpaired interval은 0을 포함할 수 있다(§1 worked example). Clustered item에서 naive standard error는 DROP의 clustered estimate보다 3.05배 작았다([[adding-error-bars-evals]] Table 4). English-only suite는 같은 item을 다른 언어 또는 encoding으로 물었을 때 더 작아지는 drop을 lost capability로 보고할 수 있다. Alpaca의 in-context-learning drop은 English 56.75 point, French 29.00 point, Leetspeak 1.50 point였다([[forgetting-via-implicit-inference]] Table 1).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 방법 |
|---|---|---|
| pairing 없이 aggregate score를 비교 | paired interval은 0을 제외하는데 drop을 "within noise"라고 보고 | per-item score에서 gain, loss, `SE_paired`를 계산 |
| stage validation loss를 forgetting signal로 사용 | validation loss는 좋아지는데 held-out slice가 떨어짐 | capability slice마다 held-out suite를 최소 하나 추가 |
| pretrained base만 기준으로 측정 | 뒤 stage 자체의 loss가 cumulative change에 숨음 | stage input과 original base를 모두 보고 |
| harmful prompt만으로 safety 평가 | safety fix 뒤 refusal은 올랐으나 over-refusal을 보지 못함 | ASR을 XSTest-style safe prompt와 짝지음 |
| benign 또는 agentic SFT 뒤 safety 평가 생략 | task metric 변화 없이 harmfulness가 상승 | 매 stage 뒤 safety pair를 재실행 |
| 모든 측정 drop을 lost knowledge로 간주 | slice의 disjoint half에서 한 epoch 뒤 accuracy가 돌아옴 | disjoint half에서 recovery test 실행 |
| model이 산출하지 못하는 fact로 학습 | training accuracy는 오르는데 development accuracy는 일찍 peak를 찍고 하락 | reference를 sampling해 target을 분류하고 filter 또는 early stop |
| LoRA가 safety를 보호한다고 가정 | LoRA run이 예상보다 높은 harmfulness를 보임 | LoRA run도 safety pair에 포함 |
| 한 run으로 operating point 선택 | 같은 drop에서 더 큰 gain을 주는 setting이 있는지 증거 없음 | 두 setting 이상에서 gain 대 drop을 그림 |
| replay sample을 모든 pre-training ability의 coverage로 간주 | 일부 slice는 회복되고 다른 slice는 계속 떨어짐 | replay coverage와 slice별 delta를 보고 |

## 이해 확인

1. §1 worked example에서 paired interval은 0을 제외하지만 unpaired interval은 그렇지 않다. Miller의 variance relation에서 어느 항이 차이를 만들며, 두 checkpoint의 어떤 성질이 그 항을 크게 만드는지 설명하라.
2. [[scaling-laws-forgetting]]은 forgetting을 fine-tuning loss의 linear function으로 fit한다. 이것이 한 method 안에서 early stopping으로 trade-off를 개선할 수 없음을 뜻하는 이유와, [[lora-learns-less-forgets-less]]의 LoRA matched-accuracy 결과가 이에 모순되지 않는 이유를 설명하라.
3. Model이 code SFT 뒤 English benchmark에서 20 point를 잃었는데 같은 question을 Spanish로 물으면 15 point를 회복한다. [[forgetting-via-implicit-inference]]를 사용해 이 lost point가 무엇을 시사하는지, 그리고 그 설명을 검증할 추가 measurement가 무엇인지 설명하라.
4. [[finetuning-new-knowledge-hallucination]]의 linear model을 사용하여 Unknown fact로 학습하면 training relation에 없던 relation의 question에서 accuracy가 낮아지는 이유를 설명하라.
5. [[shallow-safety-alignment]]의 constrained SFT loss에서 positions 2-5의 `β_t = 2`가 refusal을 보호하는 이유와, 모든 position에 `β = 2.0`을 쓰면 GSM8k accuracy가 2.1%로 낮아지는 이유를 설명하라.
6. [[retaining-by-doing]]은 KL 없는 GRPO도 KL 있는 GRPO만큼 거의 적게 잊는다고 찾았다. 이것이 regularizer보다 data source를 가리키는 이유와, 같은 결과를 설명하려면 [[rls-razor]]가 무엇을 보여야 하는지 설명하라.
7. [[mitigating-alignment-tax-rlhf]]에서 400M token replay는 한 benchmark에서는 model averaging보다 좋고 다른 두 benchmark에서는 졌다. 이 pattern의 원인을 하나 설명하고 forgetting report가 이를 어떻게 탐지하는지 설명하라.

## 연결

- **이전(dependency):** ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate. 이 장이 비용을 측정하는 선택을 정한다.
- **다음:** ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares.
- ch-00 — What General Capability Means and How It Is Measured(capability profile과 held-out suite).
- ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability(learning rate와 step count를 update size로 다룸).
- ch-30c — Weight Averaging and Model Merging for Generalist Models(WiSE-FT와 merging 전체).
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation(SDFT와 Iterative-SFT에 관련된, model 자신의 filtered sample 학습).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining.
- ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split.
- ch-37 — Policy-Gradient Foundations for Language Models(REINFORCE).
- ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax(PPO-ptx 유도).
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.
- ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO(GRPO).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention.
- ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing.
- ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions(paired interval과 multiple-comparison control).
- ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming.
- ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness.
- ch-59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate.

## 출처

- [[catastrophic-forgetting-continual-finetuning]] — FG metric, 1B-7B continual instruction tuning의 Table 4와 Table 6 forgetting, 10,000-sample Alpaca replay.
- [[scaling-laws-forgetting]] — cross-entropy-to-base forgetting metric, ARC answer-agreement example, Eq. 4 linear law 및 App. B cross-method check, AdvBench refusal count.
- [[forgetting-via-implicit-inference]] — task-inference decomposition과 conjugate-prompting recovery(Tables 1-3).
- [[spurious-forgetting]] — task alignment 대 knowledge, recovery test, AOA safety loss와 restoration, Freeze 결과.
- [[finetuning-new-knowledge-hallucination]] — SliCK category, linear accuracy model, Unknown example의 early stopping 및 filtering.
- [[finetuning-compromises-safety]] — benign fine-tuning 뒤 harmfulness, LR와 batch ablation, LoRA safety 결과, refusal-data mixing.
- [[shallow-safety-alignment]] — refusal-prefix와 per-token KL evidence, per-step ASR, recovery-example augmentation, token-wise constrained SFT.
- [[lora-learns-less-forgets-less]] — LoRA 대 full fine-tuning의 learning/forgetting table, regularization과 diversity 비교, LoRA setting.
- [[sdft-self-distillation-finetuning]] — SDFT equation, downstream·safety·helpfulness table, setting.
- [[wise-ft]] — weight-space interpolation equation, CLIP Table 1, learning-rate sensitivity, `α = 0.5` 추천.
- [[mitigating-alignment-tax-rlhf]] — RSF의 alignment tax, model averaging Pareto result, HMA, `α = 0.2`, replay coverage.
- [[retaining-by-doing]] — SFT 대 REINFORCE 대 GRPO gain/drop table, KL ablation, Iterative-SFT, setting.
- [[rls-razor]] — forgetting의 KL-on-new-task predictor 및 fit quality.
- [[rlhf-instructgpt]] — InstructGPT library card(2026-09-15에 unverified; 이 장의 값은 아래 excerpt에서 옴).
- [[rlhf-instructgpt-ptx]] — 검증된 PPO-ptx objective, `γ = 27.8`, pretraining example ratio, γ와 β ablation.
- [[adding-error-bars-evals]] — standard error, paired difference, clustered standard error formula.
- [[agentic-finetuning-misalignment]] — benign agentic fine-tuning 뒤 safety loss.
- [[xstest]] — unsafe한 것처럼 보이는 safe prompt에서 over-refusal 측정.
