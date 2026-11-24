<!-- chapter: ch-30c
     track: sft
     kind: content
     title: 범용 모델을 위한 가중치 평균과 모델 병합
     deps: [ch-30b]
     sources: [[wise-ft]], [[model-soups]], [[task-arithmetic]], [[ties-merging]], [[dare-merging]], [[model-merging-at-scale]], [[mix-data-or-merge-models]], [[online-merging-optimizer]], [[warm-weight-averaged-reward-models]], [[warp]], [[gemma-2]], [[qwen-2.5]], [[smollm-3]], [[smollm3-model-merging]], [[llama-3]], [[tulu-3]], [[tulu-3-seed-soups]], [[longer-context-deeper-thinking]], [[longred]], [[kimi-k1-5]]
     figures: figures/merge-calculator.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30c — 범용 모델을 위한 가중치 평균과 모델 병합

> **핵심 통찰.** 같은 사전학습 체크포인트에서 출발해 미세조정한 모델들은 가중치 공간에서 평균할 수 있으며, 그 정확도는 두 종점 정확도의 평균 이상일 수 있다. CLIP ViT-L/14@336px에서 zero-shot 가중치와 미세조정 가중치의 중간점은 ImageNet에서 86.8점, 다섯 distribution shift의 평균에서 76.9점을 기록했다. 미세조정 모델은 각각 86.2점과 68.6점이었다([[wise-ft]], Table 1). 서로 다른 기술의 expert를 병합하면, 세 expert-merging 연구에서 한 가지 경우(64B PaLM-2-IT expert 두 개 병합)를 제외한 모든 설정에서 expert 자신의 과제 점수는 multitask 학습보다 낮다. 예를 들어 T0-3B에서 (IA)3 병합은 66.4점, multitask 학습은 73.1점이었고([[ties-merging]], Table 1), 64B PaLM-2-IT expert 여덟 개의 병합은 정규화한 held-in 점수 0.93, multitask 학습은 0.99였다([[model-merging-at-scale]], App. Table 1). 그 64B 병합은 unseen task에서 1.09점으로 multitask의 1.05보다 높지만, 1B~24B와 instruction tuning을 하지 않은 base의 모든 크기에서는 multitask 학습이 같거나 더 잘 일반화한다(App. Tables 2 and 4). Llama 3, Gemma 2, Qwen2.5, SmolLM3는 post-training에서 가중치 평균 또는 병합을 사용하지만, 어느 보고서도 이 단계의 ablation 수치를 공개하지 않았다.
>
> **지침.** 같은 recipe를 여러 번 실행했고 seed, 데이터 버전 또는 hyperparameter만 다르다면 가중치를 평균하라. 단, held-out validation set에서 최고 단일 run 이상일 때만 평균 모델을 유지한다. greedy-soup 규칙은 validation 데이터에서 이 조건을 보장하며([[model-soups]], Recipe 1), Tülu 3의 최고 soup은 최고 seed와 0.1점 이내였다([[tulu-3-seed-soups]], Table 14). 미세조정이나 RL 단계가 이전 체크포인트가 갖고 있던 능력(robustness, long context, benchmark breadth)을 낮췄다면, 그 이전 체크포인트 쪽으로 보간하고 목표 능력과 잃은 능력을 모두 측정해 가중치를 고른다. [[wise-ft]]의 보간 모델은 두 평가에서 모두 미세조정 종점보다 높았고(Fig. 6), [[warp]]에서는 초기화 쪽 보간이 RL training trajectory보다 위에 놓였으며(§4.3), SmolLM3는 이 방식으로 RULER를 회복했다([[smollm3-model-merging]]). 분리된 skill expert를 하나의 모델로 만들어야 할 때 데이터와 계산 자원이 허용하면 mixed-data 학습을 우선한다. [[ties-merging]], [[task-arithmetic]], [[model-merging-at-scale]]에서 병합은 expert 자신의 과제에서 multitask 학습보다 뒤처지기 때문이다. 예외는 64B PaLM-2-IT expert 두 개를 병합한 경우로 1.00 대 0.99였다. base가 크고 instruction-tuned이거나, 데이터를 합칠 수 없거나, mixing에서 두 목표가 trade-off할 때에는 expert를 병합한다. 예를 들어 safety와 일반 helpfulness의 trade-off가 [[mix-data-or-merge-models]] Table 1에 있다. 세 경우 모두 각 merge에 대해 task별 held-in retention과 held-out score를 나란히 보고한다.

## 범용 모델에서 이 장이 중요한 이유

범용 모델은 여러 단계에서 얻은 능력을 동시에 지켜야 한다. 사전학습에서 얻은 지식, context extension에서 얻은 long-context 능력, SFT의 instruction following, DPO 또는 RL의 preference-aligned 행동, safety 행동이 모두 여기에 속한다. [[ch-30a]]는 각 fine-tuning 단계가 자신이 목표로 하지 않은 능력을 낮출 수 있음을(forgetting 및 alignment tax) 보였고, [[ch-30b]]는 하나의 SFT 데이터셋에 skill을 섞으면 데이터 양에 따라 interference가 생김을 보였다. 이 장은 데이터가 아니라 가중치에 작용하는 두 번째 도구군을 다룬다.

가중치 공간 방법은 서로 다른 세 목적으로 쓰이며, 목적마다 근거의 강도도 다르다.

1. **미세조정 중 잃은 능력을 회복하려고 이전 체크포인트 쪽으로 보간하기**: robustness를 위한 WiSE-FT, RLHF를 위한 WARP의 initialization 쪽 선형 보간(LITI), 그리고 RULER long-context benchmark 점수를 되돌린 SmolLM3 merge가 이에 해당한다.
2. **최고 run 하나만 남기는 대신 같은 recipe의 여러 run을 평균하기**: model soup, Llama 3와 Gemma 2의 post-training averaging, reward model을 위한 WARM이 이에 해당한다.
3. **서로 다른 skill로 학습한 expert를 한 모델로 결합하기**: task arithmetic, TIES-Merging, DARE, merging at scale, mix-or-merge 비교가 이에 해당한다.

pre-training → mid-training → SFT → preference optimization → RL → evaluation pipeline에서, 이 장이 인용하는 보고서는 여러 post-training 경계에 merging을 둔다. Llama 3는 reward-model, SFT, DPO 단계에서 평균한다([[llama-3]], §4.1.5). Qwen2.5는 DPO 내부에 merging optimizer를 적용한다([[qwen-2.5]], §4.2). WARP는 RL policy를 병합한다([[warp]]). SmolLM3는 preference optimization 뒤에 병합하여 mid-training의 long-context 능력을 복원한다([[smollm3-model-merging]]). 경계가 어디이든 측정 질문은 같다. 병합은 여러 능력을 동시에 움직이므로, 새로 더하려는 능력 하나만이 아니라 유지하려고 했던 능력들에서도 평가해야 한다.

## §1 가중치 보간과 선형 모드 연결성

**정의.** 같은 architecture를 가진 두 모델 사이의 **가중치 보간(weight interpolation)**은 두 parameter vector의 가중 평균을 parameter로 하는 세 번째 모델을 만든다. **선형 모드 연결성(linear mode connectivity)**은 두 parameter vector를 잇는 직선 위의 정확도가, 두 종점 정확도를 직선으로 이은 값보다 낮아지지 않는 성질이다. [[wise-ft]]는 이 형태를 사용한다(Eq. 2). 논문은 두 종점 정확도가 비슷할 때 이 정의가 Frankle et al.의 linear mode connectivity 정의와 일치한다고 설명한다(§5.2).

**해결하려는 문제.** Fine-tuning은 target distribution에서 정확도를 올리지만 distribution shift 아래에서는 정확도를 낮출 수 있다. CLIP ViT-L/14@336px의 경우 ImageNet end-to-end fine-tuning은 ImageNet 정확도를 76.6에서 86.2로 올리는 반면, ImageNet에서 파생된 다섯 shift의 평균 정확도는 73.4에서 68.6으로 낮춘다([[wise-ft]], Table 1의 "Zero-shot (PyTorch)" 및 "Fine-tuned E2E" 행).

**메커니즘(WiSE-FT, [[wise-ft]] §3).**

1. 이미 zero-shot으로 과제를 수행하는 사전학습 모델 θ0에서 시작한다.
2. 목표 데이터에서 θ0를 fine-tune하여 θ1을 얻는다.
3. θ0와 θ1을 잇는 선 위 모델들을 평가하고 mixing coefficient α를 고른다.

**수식.**

θ_α = (1 − α)·θ0 + α·θ1  ([[wise-ft]], Eq. 1)

Acc((1 − α)·θ0 + α·θ1) ≥ (1 − α)·Acc(θ0) + α·Acc(θ1), 모든 α ∈ [0, 1]  ([[wise-ft]], Eq. 2, Observation 1)

- θ0는 사전학습(zero-shot) parameter, θ1은 fine-tuned parameter다. α ∈ [0, 1]은 mixing coefficient이며 α = 1이면 fine-tuned 모델 그 자체다.
- Acc(·)는 주어진 parameter를 가진 모델이 하나의 evaluation distribution에서 기록한 정확도다.

논문의 구현은 state dict 안의 모든 tensor를 보간한다([[wise-ft]], App. A).

```python
theta = {
    key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
    for key in theta_0.keys()
}
model.load_state_dict(theta)
```

**계산 예([[wise-ft]] Table 1에서 도출).** α = 0.5일 때 종점 정확도를 직선으로 이으면 ImageNet 예측값은 (76.6 + 86.2)/2 = 81.4다. 실제 보간 모델은 86.8점으로, 이 직선보다 5.4점 높고 fine-tuned 모델보다도 0.6점 높다. 다섯 shift에서 직선 예측값은 (73.4 + 68.6)/2 = 71.0이고, 보간 모델은 76.9점이다. 즉 직선보다 5.9점, zero-shot 모델보다 3.5점 높다. 두 평가에서 두 종점 모두를 이겼으며, 이는 논문의 Observation 2다(§5.2).

**근거.** α = 0.5인 WiSE-FT는 여섯 개 추가 shift에서 shift 아래 정확도를 각각 3.5, 6.2, 1.7, 2.1, 9.0, 23.2점 높였고, reference accuracy 하락은 최대 0.3점이었다([[wise-ft]], §4; Result, single study). 평균적으로 최적 α는 α = 0.5보다 0~0.4점만 높았으므로, 도메인 지식이 없으면 저자들은 α = 0.5를 권한다(App. B). 같은 논문에서 learning rate 3·10⁻⁵와 3·10⁻⁶으로 10 epoch 학습한 run은 ImageNet에서 0.3점만 다르지만 shift 아래에서는 최대 8점 차이 난다(Fig. 3). 따라서 target set 정확도만으로 어느 fine-tuning run이 robustness를 보존했는지는 알 수 없다.

언어 모델에서도 관련 결과가 있다. LongReD는 Llama-3-8B를 32K context로 확장하고 continued pre-training(CPT)과 원래 체크포인트·확장 체크포인트의 평균을 비교했다. 평균 모델은 General category에서 65.92, RULER에서 83.38을 기록했고, CPT 모델은 각각 65.39와 82.80이었다([[longred]], Table 7; Result, single study). 단, 논문 자체의 distillation 방법은 두 평가에서 모두 더 높은 67.51과 84.98을 기록했다. 그러므로 그 연구에서 averaging은 최고 방법이 아니라 baseline이다.

**조건과 한계.**

- 무작위 초기화에서 학습한 두 network를 보간하면 random accuracy보다 나은 정확도를 얻지 못한다([[wise-ft]], §5.2). shared random initialization을 썼더라도 처음부터 학습한 weight에서는 linear mode connectivity가 성립하지 않는다([[warm-weight-averaged-reward-models]], Remark 1). 이 성질은 모델들이 training trajectory 일부 또는 사전학습 initialization을 공유할 때 관측됐다([[wise-ft]], §5.2).
- WiSE-FT는 image classification만 시험했다(§7). Table 1의 "optimal α" 행은 각 column, 즉 각 test set에 대해 α를 따로 선택한다. 따라서 보고한 test 수치로 α를 고른 것이다. 보고 수치에서 선택하지 않은 것은 fixed α = 0.5 행이다(Table 1 caption).
- delta가 크면 근사가 무너진다. 500B token의 추가 code pre-training 뒤 WizardCoder-Python-13B와 Llama-2-13b 사이 delta는 흔히 0.01보다 크다. 이 delta의 10%를 무작위로 drop하면 HumanEval pass@1이 63.41에서 0.0이 된다([[dare-merging]], §4.6). 따라서 어떤 merge 전에도 shared base에서 얼마나 멀어졌는지를 확인해야 한다.

**범용 모델에 주는 함의.** Fine-tuned 종점은 후보 모델 하나가 아니라 후보 모델들이 놓인 선 위의 한 점이다. 한 단계가 target accuracy와 breadth를 trade-off한다면, 이 선의 각 점은 학습을 다시 하지 않고도 한 번의 weight interpolation과 한 번의 evaluation으로 측정할 수 있다. [[wise-ft]]에서는 α = 0.5 지점이 두 종점보다 두 능력을 더 많이 보존했다.

## §2 Model soup: 같은 recipe로 실행한 run 평균

**정의.** **model soup**은 같은 사전학습 가중치에서 출발해 서로 다른 hyperparameter 또는 seed로 fine-tuning한 여러 모델의 가중치를 균일하게 평균한 모델이다([[model-soups]], §2). **greedy soup**은 모델을 한 번에 하나씩 넣되, held-out validation accuracy가 낮아지지 않을 때만 그 모델을 유지한다.

**해결하려는 문제.** 통상 절차는 여러 configuration을 학습한 뒤 validation score가 가장 높은 한 run만 남긴다. 버린 run에도 정보가 들어 있으며, 선택된 run이 out-of-distribution에서 가장 좋은 run이라는 보장도 없다. CLIP ViT-B/32 sweep에서 최고 단일 모델은 distribution shift에서 47.83점을 기록한 반면, 72개 모델을 모두 균일 평균한 soup은 51.45점을 기록했다([[model-soups]], Table 3).

**메커니즘(Recipe 1, [[model-soups]]).**

1. held-out validation accuracy 기준으로 후보 모델을 높은 순서대로 정렬한다.
2. 가장 좋은 모델 하나로 ingredient set을 시작한다.
3. 다음 모델마다, 그 모델의 가중치와 현재 ingredient들의 가중치를 평균한다. 평균 모델의 validation accuracy가 현재 soup의 accuracy 이상이면 그 모델을 유지한다.
4. 유지된 ingredient들의 평균을 반환한다.

**수식.**

θ_S = (1/|S|)·Σ_(i∈S) θ_i, 여기서 θ_i = FineTune(θ0, h_i)는 configuration h_i로 학습한 모델이며, S는 ingredient set이다([[model-soups]], §2).

**계산 예(설명용 수치).** 세 run의 validation accuracy가 A = 80.4, B = 79.9, C = 78.0이라고 하자. soup {A, B}는 80.9 ≥ 80.4이므로 B를 유지한다. soup {A, B, C}는 80.1 < 80.9이므로 C를 거절한다. 결과는 80.9점을 낸 {A, B}다. 3단계는 score 하락을 절대로 받아들이지 않으므로, greedy soup은 validation set에서는 최고 단일 모델보다 낮은 점수를 낼 수 없다. 다만 test set에서는 최고 단일 모델보다 낮을 수 있다.

**근거.**

| 설정 | 최고 단일 모델 | Uniform soup | Greedy soup(Tülu 3: 보고된 최고 soup) | 출처 |
|---|---:|---:|---:|---|
| CLIP ViT-B/32, 72 runs, ImageNet / shifts | 80.38 / 47.83 | 79.97 / 51.45 | 81.03 / 50.75 | [[model-soups]], Table 3 |
| ViT-G/14 (JFT-3B), 58 runs, ImageNet / shifts | 90.78 / 84.68(각 test set에서 최고; held-out validation으로 선택한 모델은 90.72 / 84.38, Table 4) | 보고되지 않음 | 90.94 / 85.02 | [[model-soups]], Tables 1 and 4 |
| T5-base, 32 runs, RTE | 78.3 | 보고되지 않음 | 79.1 | [[model-soups]], Table 5 |
| Tülu 3 8B SFT, 5 seeds, 평균 | 60.1 | 보고되지 않음 | 60.2(2 seeds) | [[tulu-3-seed-soups]] |
| Tülu 3 70B SFT, 3 seeds, 평균 | 72.6 | 보고되지 않음 | 72.5(2 seeds) | [[tulu-3-seed-soups]] |

Tülu 3에서 seed range는 8B에서 0.3점, 70B에서 2.6점이다. 최고 soup은 양쪽 크기 모두에서 최고 seed와 0.1점 이내였고, 팀은 최고 단일 run을 release했다([[tulu-3-seed-soups]]; Result, single study). Llama 3는 각 RM, SFT, DPO 단계에서 “다양한 버전의 data 또는 hyperparameter를 사용한 experiment로 얻은 모델”을 평균하며, pre-training annealing 중에는 Polyak average, 즉 한 training run을 따라 저장한 checkpoint들의 평균을 계산한다([[llama-3]], §4.1.5 and §3.4.3). Gemma 2는 WARP를 인용하며 “서로 다른 hyperparameter로 pipeline을 실행하여 얻은 서로 다른 모델”을 평균한다. 그 §4 overview도 각 phase 뒤에 얻은 모델을 평균한다고 명시한다([[gemma-2]], §4). 두 보고서 모두 평균 가중치, 평균한 모델 수, ablation은 제공하지 않는다.

**조건과 한계.**

- 저자들은 모든 ingredient가 높은 accuracy를 가질 때에만 uniform soup이 최고 모델을 이긴다고 관찰했다. **error barrier**는 두 모델을 잇는 직선 위에서 error가 두 종점의 error보다 높아지는 구간이다. 저자들은 이 barrier가 주로 높은 learning rate로 fine-tuning한 run 사이에서 나타났고, 그런 run은 accuracy도 낮았으므로 greedy 규칙이 제외한다고 밝혔다([[model-soups]], §3.3.1).
- seed만 다른 20개 모델을 비교한 §5에서, soup은 output ensemble과 달리 calibration을 개선하지 않는다.
- text-classification experiment의 gain은 0.0~0.8점이며, 저자들은 이를 preliminary라고 부른다(§3.3.3).
- seed variance가 soup gain보다 클 수 있다. Tülu 3의 70B에서 잘못된 seed를 고르면 최대 2.6점을 잃지만, soup은 최고 score를 0.1점만 바꾼다.

**범용 모델에 주는 함의.** 이 과정은 여러 run 평균을 variance-reduction 단계로 취급한다(Interpretation; 위 출처들 가운데 soup 전후 variance를 측정한 것은 없다). 보고할 test suite와 분리된 validation set이 필요하며, 여러 seed를 실행하는 일을 대체하지 않는다.

## §3 Task vector, 간섭, 그리고 TIES와 DARE가 이를 푸는 방식

### §3.1 Task arithmetic

**정의.** **task vector**는 한 task에서 fine-tuning하여 생긴 parameter 변화다. 즉 τ_t = θ_ft^t − θ_pre다([[task-arithmetic]], §2). Task arithmetic은 task vector의 scale을 조절한 합을 더해 모델을 편집한다.

**수식.** θ_new = θ_pre + λ·Σ_t τ_t

- θ_pre는 공유하는 pre-trained weight, τ_t는 task t의 task vector, λ는 held-out validation data로 정하는 scaling term이다. n개의 fine-tuned model을 그냥 weight-average하는 것은 λ = 1/n인 특수한 경우다.

**근거.** CLIP task vector를 두 개씩 더하면 specialist들의 accuracy를 평균 98.9% 보존한다. 여덟 task vector를 모두 사용할 수 있을 때, 그 일부 집합으로 만든 최고 모델은 여덟 task 평균 normalized accuracy 0.912를 달성하며, joint multitask fine-tuning은 0.994다. 여기서 normalized accuracy는 각 task의 accuracy를 그 task만 fine-tuning한 모델의 accuracy로 나눈 값이다([[task-arithmetic]], §4.1, App. D.2). 서로 다른 task vector 쌍의 cosine similarity는 대부분 0.01–0.06이고 MNIST-SVHN은 0.18이다(Fig. 5). 저자들은 거의 orthogonal인 vector는 간섭이 작게 더해진다고 추측한다(Interpretation). 여러 사례에서 λ = 0.3–0.5가 optimum에 가깝다(App. D.3).

### §3.2 간섭

**문제를 측정으로 표현하면.** Task vector들이 한 parameter에서 서로 반대 방향을 가리키면, 그 합이나 평균은 해당 parameter를 0 쪽으로 줄인다([[ties-merging]], Fig. 2). T5-Large에서는 Fig. 5에 있는 세 방법(averaging, Task Arithmetic, TIES) 모두 task 수가 늘어날수록 normalized average accuracy가 하락한다(§6, Fig. 5). WizardLM-13B(instructions), WizardMath-13B, llama-2-13b-code-alpaca를 합친 13B decoder merge에서, 세 model 전체에 Task Arithmetic을 적용하면 GSM8K 58.45점으로 WizardMath 단독 64.22점보다 낮다([[dare-merging]], Table 1). Instruction model과 code model에 DARE 없이 TIES를 쓰면 HumanEval과 MBPP가 각각 0.0점인데, instruction model 단독은 36.59와 34.00점이다(Table 1).

### §3.3 TIES-Merging

**메커니즘([[ties-merging]], Algorithm 1).**

1. **Trim.** 각 task vector에서 magnitude 상위 k% entry만 남기고 나머지는 0으로 만든다.
2. **Elect.** parameter p마다 merged sign γ_p = sgn(Σ_t τ̂_t^p)로 정한다. 즉 총 magnitude가 더 큰 쪽의 sign을 선택한다.
3. **Disjoint merge.** γ_p와 sign이 같은 non-zero entry만 평균한다.
4. **Scale.** θ_m = θ_init + λ·τ_m로 합친다.

```
▷ Step 2: Elect Final Signs.      γm = sgn( Σ_t τ̂t )
▷ Step 3: Disjoint Merge.         Ap = { t ∈ [n] | γ̂t^p = γm^p };   τm^p = (1/|Ap|) Σ_(t∈Ap) τ̂t^p
▷ Obtain merged checkpoint        θm ← θinit + λ ∗ τm
```

**계산 예(강의의 계산이며, figure의 preset A와 같은 수치).** 한 parameter의 task value가 +0.5, +0.4, −0.6이라고 하자. 단순 평균은 (0.5 + 0.4 − 0.6)/3 = 0.10이다. λ = 0.4인 Task Arithmetic은 0.4 × 0.3 = 0.12를 준다. Preset A에서는 세 entry 모두 top-20% trim을 통과한다. 따라서 TIES는 sgn(0.3) = +를 elect하고, 이에 동의하는 entry를 (0.5 + 0.4)/2 = 0.45로 평균하며, λ = 1일 때 0.45를 유지한다. Trimming은 작은 entry도 제거한다. Preset A에서는 trimming 전 10개 parameter 가운데 9개가 sign conflict를 가지지만, 각 vector의 상위 20%만 남기면 10개 가운데 1개만 남는다.

Interactive figure [figures/merge-calculator.html](figures/merge-calculator.html)에서는 reader가 세 task vector를 직접 바꾸고, averaging, Task Arithmetic, TIES(trim, elect, disjoint mean), DARE(drop과 rescale)가 parameter마다 만드는 값을 볼 수 있다. Trim 전후 sign-conflict 수 역시 보여 준다.

**근거([[ties-merging]]).** Validation set을 사용할 때 TIES는 T0-3B에 적용한 parameter-efficient fine-tuning 방법 (IA)3에서 기존 최고 merge보다 2.5점 높다(66.4 대 63.9). T5-large에서는 3.6점 높다(76.9 대 73.3). 하지만 multitask training은 (IA)3에서 73.1점이다(Table 1). 여섯 held-out task에서 T5-large TIES merge는 40.4점으로, activation에 대해 layer별 linear regression으로 merge를 구하는 RegMean의 36.0점과 zero-shot의 27.6점보다 높다(Table 2). (IA)3 ablation에서 disjoint mean을 빼면 3.2점, scaling 단계를 빼면 5.2점이 떨어진다(Table 6). 상위 20% entry의 sign을 뒤집으면 accuracy가 낮아지지만, 하위 80%를 뒤집어도 영향이 작다(§7.2, Fig. 7).

### §3.4 DARE

**정의.** DARE(Drop And REscale)는 각 task vector entry의 비율 p를 무작위로 0으로 만들고, 남은 entry에 1/(1 − p)를 곱한 뒤 merge한다([[dare-merging]], Eq. 1).

**수식.** m ∼ Bernoulli(p); δ̂ = ((1 − m) ⊙ δ)/(1 − p); θ_M = θ_PRE + λ·Σ_k δ̂_k

- δ = θ_SFT − θ_PRE는 한 SFT model의 delta parameter, p는 drop rate, m은 entry별 drop mask, ⊙는 element-wise product, λ는 Task Arithmetic scaling term이다.

**계산 예.** δ = (0.002, −0.001, 0.0005, 0.003), p = 0.5이고 mask가 entry 2와 3을 drop한다고 하자. 그러면 δ̂ = (0.004, 0, 0, 0.006)이다. Entry 1의 expectation은 0.5 × 0 + 0.5 × 0.004 = 0.002 = δ₁이므로, rescale은 각 layer output의 expectation을 보존한다(§3.1). Figure의 “DARE rescaling check”는 4,000개의 random mask를 평균해 그 mean이 τ로 되돌아감을 보여 준다.

**근거([[dare-merging]]).** 시험한 SFT model의 delta는 대부분 0.002 안에 있으며, 저자들은 그 90% 혹은 심지어 99%를 drop해도 “성능을 많이 희생하지 않을 수 있다”고 썼다(§5). WizardMath-70B는 p = 0.99를 견디지만 7B와 13B는 그렇지 않다(§4.2). Instruction 및 code model에 TIES 전에 DARE를 넣으면 HumanEval은 0.0에서 18.29, MBPP는 0.0에서 26.40으로 오른다. 그래도 instruction model 단독의 36.59와 34.00보다는 낮다(Table 1). GLUE encoder model에서 DARE는 merge average를 −0.03%에서 +0.84% 범위로 바꾼다(§4.3). 7B DARE merge 하나(supermario v2)는 2024-01-28 기준 Open LLM Leaderboard의 7B model 중 1위였다(§4.3, Table 2). Public checkpoint를 merge하여 얻은 leaderboard rank는 그 leaderboard 밖 task에 대한 breadth 증거는 아니다.

**§3의 조건과 한계.** 세 방법은 모두 shared base와 작은 delta를 가정한다. Task Arithmetic의 language experiment는 GPT-2(Small, Medium, Large)와 T5를 사용하며 TIES는 decoder-only LLM을 시험하지 않는다. DARE의 13B 결과는 quality가 서로 다른 세 public model에서 나왔고, 저자들은 각 source model이 “well fine-tuned”돼야 한다고 명시한다(§4.3).

**범용 모델에 주는 함의.** Evaluation 전에도 간섭을 측정할 수 있다. 반대 sign을 가진 큰 delta parameter 수를 세고, base와 비교해 delta magnitude를 확인한다. Evaluation 뒤에는 merged model의 task별 최소 점수를 보면 average가 숨기는 간섭을 찾을 수 있다.

## §4 Merge인가 mix인가: multi-task generalist에서 나온 증거

**대규모 merging ([[model-merging-at-scale]]).** 1B, 8B, 24B, 64B 크기의 PaLM-2와 instruction-tuned PaLM-2-IT 각각을 T0 held-in task category에서 8개 expert로 fine-tuning했다. 이어 2–8개 expert의 384개 merge를 held-in task와 7개 held-out dataset에서 평가했다. Held-in 점수는 해당 expert로 normalize했고, held-out 점수는 base model로 normalize했다(§3). Appendix Table 1–4의 Task Arithmetic 값은 다음과 같다.

| Base, 크기(8 experts merge) | Held-in: merge / multitask | Held-out: merge / multitask |
|---|---|---|
| PaLM-2-IT 8B | 0.88 / 0.96 | 1.03 / 1.12 |
| PaLM-2-IT 24B | 0.92 / 0.98 | 1.18 / 1.18 |
| PaLM-2-IT 64B | 0.93 / 0.99 | 1.09 / 1.05 |
| PaLM-2 8B (instruction-tuned 아님) | 0.42 / 1.06 | 1.00 / 1.62 |
| PaLM-2 64B (instruction-tuned 아님) | 0.67 / 0.96 | 1.35 / 1.72 |

Abstract는 8개 large expert의 merge가 “multitask trained model과 비교해 종종 더 잘 generalize한다”고 쓴다. 표에서 이 말이 성립하는 경우는 64B PaLM-2-IT에 expert 6개 또는 8개를 합칠 때다(1.06–1.10 대 1.05). 24B에서는 가장 좋은 merge가 multitask training과 동률이고, instruction-tuned가 아닌 base에서는 모든 크기에서 held-out task의 multitask training 점수가 더 높다(App. Table 2, 4; Result, single study). 1B PaLM-2에서는 expert 수가 2개에서 8개로 늘 때 held-in retention이 0.66에서 0.39로, 1B PaLM-2-IT에서는 0.91에서 0.86으로 낮아진다. §4.4 본문은 이 수치를 8B에 귀속시키지만, 실제로는 1B column과 일치한다([[model-merging-at-scale]] excerpt, Discrepancies). 64B PaLM-2-IT에서 expert 8개를 merge하면 네 방법(Averaging, Task Arithmetic, TIES, Dare-TIES)이 모두 held-in 0.93을 낸다(Table 1). 이 연구의 그 규모에서는 method choice의 차이가 작다(§4.5).

**두 objective 아래에서 mix 또는 merge ([[mix-data-or-merge-models]]).** Pre-release Aya 23 8B checkpoint에서 objective별로 model 하나(안전 data 0% 또는 100%)를 학습했다. 이 두 model을 merge한 결과를, 같은 data를 safety 15%로 섞어 단일 model 하나를 학습한 결과와 비교했다. Merge 방법은 Linear averaging, SLERP, TIES, DARE-TIES다. SLERP(spherical linear interpolation)는 두 normalized weight vector 사이의 직선이 아니라 arc를 따라 이동한다. DARE-TIES는 DARE dropout 뒤 TIES를 적용한다(§2.1). Harm은 base 대비 harmful generation의 상대 변화이며 낮을수록 좋다. General quality는 base와 비교한 Dolly-200 win rate이며, 두 측정 모두 GPT-4가 judge했다(§2.4).

| Stage | Method | Harm | Win rate |
|---|---|---:|---:|
| DPO | 15% Safety Mix (단일 model) | −54.69 | 71.0 |
| DPO | Linear merge | −48.6 | 75.0 |
| DPO | SLERP merge | −57.8 | 78.0 |
| DPO | TIES merge | −65.1 | 63.6 |
| DPO | DARE-TIES merge | −55.9 | 78.5 |
| SFT | 15% Safety Mix (단일 model) | −56.6 | 67.4 |
| SFT | Linear merge | −49.1 | 76.0 |
| SFT | SLERP merge | −58.2 | 72.6 |
| SFT | TIES merge | −45.2 | 74.9 |
| SFT | DARE-TIES merge | −56.1 | 70.0 |

SLERP만이 두 stage 모두에서 mixture보다 두 축 모두에서 낫다. DARE-TIES는 DPO에서만 두 축 모두 mixture보다 낫다. DPO의 TIES는 네 merge 중 harm이 가장 낮지만, 모든 DPO model 중 win rate도 가장 낮다(Table 1; Result, single study, seed 보고 없음). 별도의 SFT checkpoint TIES 실험에서 영어·프랑스어·스페인어 monolingual model 3개를 merge한 결과는 6개를 merge한 결과보다 낫다(harm −65.3 대 −63.2, win rate 77.0 대 71.2; Fig. 5). 저자들은 이를 cross-lingual interference라고 부른다. 다만 논문은 3-language merge를 3개 언어에서 평가했는지 6개 언어에서 평가했는지는 밝히지 않는다(§3.4).

**Long-to-short reasoning.** Kimi k1.5는 long-CoT model과 short-CoT model의 weight를 평균하는 방법을 네 가지 long2short 방법 중 하나로 사용한다. 저자들은 Figure 7에서 long2short RL이 DPO와 merge보다 더 높은 token efficiency를 보인다고 서술한다. 그러나 본문에는 merged model의 numeric score가 없다([[kimi-k1-5]], §2.4, §3.4).

**범용 모델에 주는 함의.** Expert merge는 주로 base model이 크고 instruction-tuned됐을 때, 또는 data mixing이 두 objective 중 하나를 낮출 때 mixed training을 대신할 수 있다(위 두 연구의 Interpretation). 더 작거나 instruction-tuned되지 않은 base의 경우, 위 근거는 mixture로 training하는 쪽([[ch-30b]])과 recovery 또는 variance reduction을 위해 merging을 쓰는 쪽을 지지한다.

## §5 Alignment 내부의 merging: WARM, WARP, 그리고 Online Merging Optimizer

**Alignment tax**는 preference optimization 또는 RL을 하는 동안 pre-training 또는 SFT 능력이 사라지는 현상이다([[ch-30a]], [[ch-38]]). 아래 세 방법은 이 단계의 weight 공간에서 직접 연산한다.

**SFT model과 DPO model의 offline merge ([[online-merging-optimizer]], Table 1).** Training이 끝난 DPO model을 SFT reference model과 merge했다. Linear, DARE, TIES를 쓰고 weight는 0.1–0.9 범위에서 탐색했다. AdamW로 DPO만 했을 때와 비교해, Qwen1.5-1.8B-Chat의 AlpacaEval 2.0 LC는 −0.41에서 −0.96만큼, Qwen1.5-7B-Chat은 −1.35에서 −1.65만큼 변했다. AlpacaEval 2.0 LC는 baseline model output과 비교한 length-controlled win rate이며, GPT-4-1106-preview가 판정한다(App. B). LLaMa-3-8B-Instruct에서는 변화가 +0.02에서 +0.21이었다. 아홉 merge 전체에서 benchmark average 변화 범위는 −1.2에서 +0.6이다.

**Online Merging Optimizer.** 매 optimizer step마다 Adam update를 sparse하게 만들고, pre-trained base에서 SFT model까지의 delta와 섞는다.

θ_m^(t) = θ_m^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r)  ([[online-merging-optimizer]], Eq. 3, OnDARE)

- θ_m^(t)는 t step 뒤 policy weight다. Δθ는 t step의 Adam update다. τ_r = θ_r − θ_b는 SFT reference weight에서 pre-trained base weight를 뺀 값이다. F_R은 각 entry를 확률 p로 보존하는 random mask이며 p는 reserve rate다(Eq. 4). α는 merging weight다.
- 논문은 α를 10⁻⁷에서 10⁻⁴까지 sweep했다(Table 2). τ_r은 전체 SFT delta이며 매 step에 더한다. 저자들은 α가 크면 training이 불안정해진다고 보고한다(§5.3).
- OnDARE는 Eq. 3으로 정의한 DARE 기반 variant다. OnTIES는 random mask 대신 magnitude 상위 p의 sparsification과 sign 기반 combination rule을 쓴다(Eq. 5–7).

DARE와 달리, 남긴 entry를 rescale하지 않는다. 저자들은 rescaling이 “multi-step optimization에서 numeric stability를 해친다”고 쓴다(§4.2). UltraFeedback에서 DPO를 하면, AdamW 대비 benchmark average / MT-Bench / AlpacaEval LC 변화는 Qwen1.5-1.8B-Chat에서 +0.5 / +0.24 / +0.05, Qwen1.5-7B-Chat에서 +1.1 / +0.12 / +0.28, LLaMa-3-8B-Instruct에서 +1.3 / +0.19 / +1.57이었다(Table 1; Result, single study). 논문이 보고한 judge standard deviation은 MT-Bench 약 0.05, AlpacaEval LC 약 0.5다(App. B). 따라서 AlpacaEval의 +0.05 및 +0.28 이득은 한 standard deviation 안에 있다. Qwen1.5-1.8B-Chat에서 매 step이 아니라 K step마다 merge하면, K가 5에서 200으로 커질수록 MT-Bench가 AdamW 값 쪽으로 되돌아간다(§6.1). 논문의 α 표에서는 benchmark average가 α = 5e−6에서 최대이지만, 본문은 5e−7이라고 쓴다. 또 표는 α가 작아질수록 MT-Bench가 올라감을 보이지만, 본문은 α가 커질수록 올라간다고 쓴다([[online-merging-optimizer]] excerpt, Discrepancies). Qwen2.5는 약 150,000 pair로 한 epoch DPO를 하면서 learning rate 7 × 10⁻⁷에서 “Online Merging Optimizer”를 썼다고만 밝히고, α와 p는 공개하지 않는다([[qwen-2.5]], §4.2).

**WARM: weight-averaged reward model ([[warm-weight-averaged-reward-models]]).** SFT run 하나의 checkpoint에서 초기화한 M개의 reward model을 서로 다른 learning rate, dropout, data order로 fine-tuning한 뒤 하나의 reward model로 평균한다(§3.1, §3.3, App. B.3). TL;DR summarization에서 M = 6 평균으로 RL을 하면, best single reward model로 학습한 policy보다 oracle win rate 79.4%를 내는 policy가 나온다(§5.2, Fig. 9(c)). Preference label의 25%를 뒤바꿨을 때 weight average는 prediction ensemble보다 corrupted training pair를 더 적게 fit하며, out-of-distribution pair에서 더 정확하다(Fig. 4–5). 다만 각 reward model이 summary length 같은 동일한 spurious feature에 주로 의존한다면, 평균도 그 의존성을 유지할 가능성이 높다고 저자들은 쓴다(§6, Limitations).

**WARP: weight-averaged rewarded policy ([[warp]]).** Gemma “7B”에 REINFORCE를 적용할 때 다음 세 merge를 결합한다.

1. KL anchor를 고정 SFT model 대신 policy의 exponential moving average로 둔다(§3.1).
2. 독립적으로 학습한 RL policy 두 개를 task vector의 spherical interpolation으로 merge한다. slerp = θ_init + sin[(1−λ)Ω]/sin Ω·δ1 + sin[λΩ]/sin Ω·δ2이며, δ_m = θ_m − θ_init이고 Ω는 layer 내 δ1과 δ2 사이의 angle이다(§3.2).
3. Merge 결과를 initialization 방향으로 선형 보간한다. θ_η = (1 − η)·θ_init + η·θ_slerp (LITI)이며, η = 0.3인 θ_η가 다음 iteration의 시작점이 된다(§3.3–3.4).

모든 LITI front는 KL-reward 공간에서 RL training trajectory 위에 놓인다(§4.3). Iteration 3의 MBPP / MMLU / GSM8K / MATH / HumanEval / BBH 점수는 45.4 / 57.6 / 66.8 / 31.0 / 50.0 / 58.8이다. Gemma “7B” 1.1의 39.0 / 56.4 / 55.6 / 25.6 / 46.9 / 53.1과 비교한 값이다(Table 2). Side-by-side score는 iteration 3 뒤에는 더 이상 좋아지지 않는다(Table 1). Output length는 KL과 함께 늘어난다(App. E). 별도 run에서 나온 RL task vector는 거의 orthogonal하다(Ω ≈ 90°)(App. C.2). WARM card는 Gemma 3 report가 RL 단계에서 개선된 WARM 및 WARP version을 사용한다고 기록한다(Gemma 3 §3, [[warm-weight-averaged-reward-models]] Connections 경유).

**범용 모델에 주는 함의.** Alignment 내부에서 merging은 미리 고정한 KL coefficient만으로 조절하는 대신, SFT model 또는 initial model 쪽으로 끌어당기는 regularizer의 강도를 training 중이나 training 뒤에 선택할 수 있게 한다(Interpretation). 근거는 1.8B–8B Qwen1.5와 LLaMa-3, Gemma “7B”, 그리고 크기가 보고되지 않은 PaLM reward model과 policy를 포함한다. 각 논문은 preference 또는 summarization dataset 하나씩만 사용했으며, judge metric의 noise는 [[online-merging-optimizer]]에서만 보고됐다.

## §6 잃어버린 능력을 되찾기 위한 merging: long context와 stage checkpoint

**SmolLM3 ([[smollm3-model-merging]], official blog; [[smollm-3]]).** Anchored Preference Optimization(APO) 뒤 SmolLM3-3B는 math, science, instruction following, coding, chat, multilingual evaluation에서 좋아졌지만 RULER에서는 떨어졌다. 팀은 이 하락을 reasoning mid-training 단계까지 추적했고, APO data가 24k token으로 제한됐었다고 적는다. 공개 checkpoint는 두 단계 merge다. 먼저 APO checkpoint들을 soup으로 만들고, 이어서 그 soup의 0.9와 long-context 성능이 강한 mid-training checkpoint의 0.1을 linear merge했다. 이 방법은 “최대 128k token context에서 base model의 RULER score를 회복했다”고 한다. RULER 값, soup에 넣은 checkpoint 수, 시도한 다른 weight는 보고하지 않았다.

**Reasoning SFT 전 merge ratio sweep ([[longer-context-deeper-thinking]], App. B Table 9).** Qwen2.5-7B-Instruct와 Qwen2.5-7B-Instruct-1M을 1M model에 주는 weight 0 / 0.1 / 0.7 / 1.0으로 merge하고, 각 merge를 reasoning data로 fine-tuning했다.

| Merge ratio | 0 | 0.1 | 0.7 | 1.0 |
|---|---:|---:|---:|---:|
| 32K long-context score | 78.1 | 79.1 | 79.5 | 77.7 |
| MATH500 (short-SFT와 long-SFT의 평균) | 84.16 | 84.92 | 84.38 | 82.88 |
| AIME22-24 (평균) | 20.56 | 21.00 | 21.56 | 18.11 |
| GSM8K (평균) | 92.37 | 92.66 | 92.67 | 92.59 |

순수 long-context model(ratio 1.0)은 32K score, MATH500, AIME22-24에서 모두 가장 낮다. 따라서 nominal context window가 가장 긴 model이 이 평가들에 가장 좋은 출발점은 아니었다. GSM8K에서는 네 ratio가 0.30점 안에 있다(Result, single study; run 횟수는 밝히지 않음).

**Llama 3의 stage checkpoint.** Llama 3는 final 40M-token annealing phase 동안 checkpoint를 평균해 pre-trained model을 만들고, 각 post-training stage에서 서로 다른 data 또는 hyperparameter run의 model을 평균한다([[llama-3]], §3.4.3, §4.1.5). Ablation을 보고하지 않으므로 효과 크기는 알 수 없다.

**Worked example: recovery weight 선택.** Post-trained model이 short-context suite에서 60점, RULER-128K에서 40점이고 long-context checkpoint가 각각 45점과 70점이라고 하자(설명용 수치). 두 metric이 endpoint를 잇는 직선을 따른다면 0.9/0.1 merge의 점수는 0.9 × 60 + 0.1 × 45 = 58.5, 그리고 0.9 × 40 + 0.1 × 70 = 43.0이다. 이 linear 가정에서는 0.1 weight가 RULER를 long-context checkpoint 쪽으로 30점 중 3점만 움직인다. SmolLM3는 0.1 weight로 base model의 RULER score를 회복했다고 보고한다. Soup의 RULER score를 S, mid-training checkpoint의 score를 M, base model의 score를 B라고 쓰자. Endpoint line 아래에서 0.9·S + 0.1·M ≥ B이려면 M ≥ B + 9·(B − S)여야 한다. 따라서 soup가 base보다 3점 낮다면 checkpoint는 base보다 27점 높아야 한다. Blog는 S, M, B 어느 값도 주지 않으므로 recovery가 endpoint-line prediction을 넘었는지는 검증할 수 없다(Interpretation). Merge 결과는 weight에 대해 linear일 필요가 없으므로 merge weight는 계산으로 정하지 말고 두 suite에서 모두 측정해야 한다.

## §7 Merge 평가: retention과 breadth를 함께 본다

**정의.** Merge 평가는 task마다 정규화한 두 quantity를 보고한다. **held-in retention**은 merged model의 expert 자체 task 점수를 그 expert의 점수로 나눈 값이고, **held-out ratio**는 어느 ingredient도 학습하지 않은 task에서 merged model이 낸 점수를 base model의 점수로 나눈 값이다([[model-merging-at-scale]], §3).

**해결하려는 문제.** Merge는 한 번에 많은 능력을 바꾸며, 다음 세 보고 관행은 실패를 숨긴다. task 전체 평균을 내는 것(merged instruction·code model이 HumanEval에서 0.0점이어도 평균으로 가려질 수 있다, [[dare-merging]] Table 1), 보고한 test set에서 merge weight를 고르는 것([[wise-ft]] Table 1의 column별 “optimal α”), 그리고 judge noise보다 작은 judge-score 변화를 개선으로 읽는 것(AlpacaEval LC standard deviation은 약 0.5, [[online-merging-optimizer]] App. B)이다.

**절차.**

1. Merge 전에 세 suite를 고정한다. 각 ingredient의 held-in task, held-out task, 그리고 어떤 ingredient도 직접 target하지 않는 위험 능력(long context, safety, calibration, multilingual)이다.
2. α, λ, k, p 및 ingredient를 고를 validation split을 고정한다. 수치는 분리된 test split에서만 보고한다.
3. Task별 retention과 held-out ratio를 계산하고 mean과 minimum을 함께 보고한다.
4. 가장 좋은 single ingredient, base model, pooled data로 학습한 multitask model과 비교한다.
5. 두 번째 seed 또는 두 번째 ingredient selection으로 반복하고, 그 spread를 보고한다.

**수식.**

\[
r_t = S_{merge}(t) / S_{expert}(t), \qquad g_u = S_{merge}(u) / S_{base}(u)
\]

- \(t\): held-in task, \(u\): held-out task, \(S_{merge}\), \(S_{expert}\), \(S_{base}\): 각각 merged model, t에 학습된 expert, base model의 점수다.

**계산 예(설명용 수치).** Expert 두 개가 각자 task에서 80점과 60점이고 merge가 72점과 30점을 냈다고 하자. 그러면 \(r = 0.90\), \(0.50\)이고 mean은 0.70, minimum은 0.50이다. Held-out task에서 base가 50점, merge가 54점, multitask model이 56점이라면 merge의 \(g = 1.08\), multitask training의 \(g = 1.12\)다. Mean retention 0.70은 한 task가 자기 expert 점수의 절반이라는 사실을 가리고, held-out ratio가 1보다 커도 multitask baseline보다는 낮다.

**근거.** Base에 정규화하면 약한 base가 큰 gain처럼 보일 수 있다. 저자들은 64B PaLM-2 merge의 약 30% relative held-out improvement를 base model의 약한 zero-shot 성능으로 설명했고([[model-merging-at-scale]], §4.4), 같은 data로 multitask training한 점수는 그곳에서 1.72다(Table 4). Expert에 정규화하면 expert가 약할 때 retention이 부풀려진다. [[dare-merging]]에서 code expert의 HumanEval은 23.78로 instruction model의 36.59보다 낮다(Table 1). 따라서 merge가 code expert의 점수를 전부 유지해도 자기 ingredient 중 하나보다 code 능력이 낮을 수 있다. 저자들은 각 merged model이 충분히 잘 fine-tune돼 있어야 한다고 쓴다(§4.3). Run 간 spread는 merge gain보다 클 수 있다. Tülu 3 70B SFT seed는 2.6점 범위에 걸쳐 있고, soup effect는 0.1이다([[tulu-3-seed-soups]]).

**조건과 한계.** 이 장의 source 중 merge comparison에 variability를 보고한 것은 세 개다. [[model-soups]]는 random greedy-soup order 세 개의 spread를 준다(Table 3: 괄호 안 0.05를 포함한 80.79). [[warp]]는 M > 2에서 merge order 전반의 작은 standard deviation을 보고한다(App. B.3). [[online-merging-optimizer]]는 judge standard deviation을 준다(App. B). 여기서 인용한 model report(Llama 3, Gemma 2, Qwen2.5, SmolLM3)는 merge에 대한 variability를 주지 않는다.

**범용 모델에 주는 함의.** Merge는 minimum retention, held-out ratio, 위험 능력이 모두 reference model에 대해 미리 정한 tolerance 안에 있을 때 받아들인다. 이 값들은 merge를 고르는 데 쓰지 않은 split에서 측정해야 한다.

## 음성 표본과 음성 피드백

병합 자체는 rejected sample을 만들지 않지만, 이 장에는 두 가지 음성 신호가 있다. 한 가지는 행동을 제거하는 weight-space negation이고, 다른 한 가지는 reward-model averaging에서의 오염된 preference label이다. 전자는 강좌 전체에서 쓰는 표본 수준의 네 가지 “negative” 의미(negative marginal value, content로서의 negative, conditioning으로서의 negative, gradient로서의 negative)와 다르다.

1. **음성이 어디에서 오는가.** Task arithmetic은 원치 않는 content에 ordinary cross-entropy로 fine-tuning한 model(GPT-2 Large에서 toxicity > 0.8인 Civil Comments)을 써서 vector를 만들고 이를 뺀다: \(\tau_{new} = -\tau\) ([[task-arithmetic]], §3.2). Training data는 원치 않는 행동에 대한 positive이고, negation은 weight space에서 일어난다. WARM은 뒤바뀐(잘못된) preference label을 corrupted supervision으로 연구한다([[warm-weight-averaged-reward-models]], §4.2).
2. **실무가 이를 어떻게 쓰는가.** Negation은 표본별 gradient 없이 행동의 likelihood를 낮춘다. 효과상 “negative as gradient”(행동이 확률을 잃음)에 가장 가깝지만, weight edit으로 한 번 적용한다(course classification). §4–§5의 preference merge는 DPO의 rejected term(상류 단계의 negative as gradient)으로 이미 학습한 model에 작용한다.
3. **메커니즘과 근거.** GPT-2 Large에서 negative task vector는 toxic generation을 4.8%에서 0.8%로 낮추고 WikiText-103 perplexity는 16.4에서 16.9가 된다. 같은 toxic data에서 gradient ascent는 toxic을 0.0%까지 만들지만 perplexity는 \(10^{10}\)을 넘는다([[task-arithmetic]], Table 2). Gradient ascent는 negative as gradient다. target token \(y\)에 대해 \(\partial \log p_y / \partial z_j = 1[j = y] - p_j\)이므로, \(\log p_y\)를 낮추면 \(z_y\)는 낮아지고 다른 모든 logit \(z_j\)는 현재 확률 \(p_j\)에 비례해 올라간다. 즉 mass가 model이 이미 다음 순위에 둔 token으로 이동한다. Objective에는 lower bound가 없고(\(\log p_y \to -\infty\)), positive anchor도 없다. ordinary text에서 control perplexity가 \(10^{10}\)을 넘는 것은 분포가 파괴됐다는 해석과 부합한다(Interpretation). Negated task vector는 완료된 fine-tuning run 하나의 displacement를 \(\lambda\)로 scale한 것이므로 구성상 크기가 bounded다.
4. **Control.** Control task에서 고른 \(\lambda\)로 negation을 scale하고, 같은 norm의 random vector(논문의 control: toxic 4.8%, perplexity 16.4)와 비교하며, positive-data baseline(non-toxic data fine-tuning: 1.8%, 17.2)을 유지한다.
5. **Diagnostic과 generality에 미치는 효과.** Target metric과 control-task metric을 함께 보고한다. Reward-model averaging에서는 label noise가 있는/없는 held-out pair에서 weight averaging을 prediction ensembling과 비교한다([[warm-weight-averaged-reward-models]], Fig. 4–5). 이 장의 어떤 source도 negation edit 뒤 calibration 또는 over-refusal을 측정하지 않았다(not reported).

## 레시피

| Model (정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| WiSE-FT, CLIP ViT-L/14@336px | not reported | merge | mixing coefficient \(\alpha\) (zero-shot ↔ fine-tuned) | domain knowledge가 없을 때 0.5 | arXiv:2109.01903v3 §4; App. B | verified 2026-09-15 ([[wise-ft]]) | App. B Table 3: 평균에서 optimal \(\alpha\)가 0.5보다 0–0.4 pp 좋음 |
| Model soups, CLIP ViT-B/32 | not reported | merge | ingredients; selection rule | random-search run 72개; held-out validation accuracy로 정렬한 greedy soup, 5개 선택 | arXiv:2203.05482v3 §3.3.1, Recipe 1 | verified 2026-09-15 ([[model-soups]]) | Table 3: greedy 81.03 vs best single 80.38 ImageNet |
| Task Arithmetic, CLIP | not reported | merge | \(\lambda\) | held-out validation에서 선택; 많은 경우 0.3–0.5가 near-optimal | arXiv:2212.04089v3 §2; App. D.3 | verified 2026-09-15 ([[task-arithmetic]]) | App. D.3 Fig. 15 |
| TIES-Merging (validation set 없음) | T5, (IA)3 T0-3B, ViT | merge | trim \(k\); \(\lambda\); Task Arithmetic baseline \(\lambda\) | top-20%; \(\lambda = 1\); \(\lambda = 0.4\) | arXiv:2306.01708v2 §5; App. C.4 | verified 2026-09-14 ([[ties-merging]]) | App. C.4의 11개 (IA)3 model grid |
| DARE 13B decoder merges | 13B | merge | scaling term; TIES retain ratio; drop rate | [0.5, 1.0]에서 탐색; [0.5, 0.7, 0.9]; Table 1에는 미보고 | arXiv:2311.03099v3 §4.3 | verified 2026-09-15 ([[dare-merging]]) | Table 1 |
| DARE “supermario v2” | 7B | merge | drop rate \(p\); Task Arithmetic \(\lambda\) | 0.5; 0.5 | arXiv:2311.03099v3 App. A.5 | verified 2026-09-15 ([[dare-merging]]) | Open LLM Leaderboard만(Table 2) |
| PaLM-2와 PaLM-2-IT expert (merging at scale) | 1B-64B | SFT | steps; LR; dropout | 2,000(task별 조정); 3e-5; 0.05 | arXiv:2410.03617v1 App. B | verified 2026-09-15 ([[model-merging-at-scale]]) | ablation 미보고; merge \(\lambda\) 미보고 |
| Aya 23 8B pre-release (mix vs merge) | 8B | merge | method; weight grid; tool | Linear, SLERP, TIES, DARE-TIES; {0, 0.3, 0.5, 0.7, 1}; mergekit | arXiv:2410.10801v1 §2.1 | verified 2026-09-14 ([[mix-data-or-merge-models]]) | Table 1; selection metric 미보고 |
| Tülu 3 8B SFT / 70B SFT | 8B / 70B | merge | seeds; merge method; release choice | 5 / 3 seeds; mergekit linear weighted averaging; best single run 출시 | arXiv:2411.15124v5 §4.3, §4.3.1, Table 14 | verified 2026-09-15 ([[tulu-3-seed-soups]]) | Table 14: best soup 60.2 vs 60.1 (8B), 72.5 vs 72.6 (70B) |
| Llama 3.1 405B | 405B | merge | model averaging | RM, SFT, DPO stage에서 다른 data 또는 hyperparameter run; weight 미공개 | arXiv:2407.21783v3 §4.1.5 | verified 2026-09-14 ([[llama-3]]) | ablation 미보고 |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | checkpoint averaging | final 40M-token anneal 동안 checkpoint의 Polyak average | arXiv:2407.21783v3 §3.4.3 | verified 2026-09-14 ([[llama-3]]) | ablation 미보고 |
| Gemma 2 IT (all sizes) | 2B/9B/27B | merge | method | 다른 hyperparameter를 쓴 pipeline run 평균(WARP 인용); count와 weight 미보고 | arXiv:2408.00118 §4 | verified 2026-09-14 ([[gemma-2]]) | ablation 미보고 |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | preference | optimizer; pairs; epochs; LR | Online Merging Optimizer; 약 150,000 pair; 1; \(7 \times 10^{-7}\); \(\alpha\), \(p\) 미보고 | arXiv:2412.15115v2 §4.2 | verified 2026-09-14 ([[qwen-2.5]]) | ablation 미보고 |
| Online Merging Optimizer run (Qwen1.5-1.8B/7B-Chat, LLaMa-3-8B-Instruct) | 1.8B-8B | preference | DPO \(\beta\); batch; \(\alpha\) grid | 0.1; 128; \(\alpha\) 1e−4부터 1e−7 | arXiv:2405.17931v1 App. C; §5.3 | verified 2026-09-15 ([[online-merging-optimizer]]) | App. C: {64, 128, 512} 중 batch 128 최고 |
| OnDARE merging weight \(\alpha\)(model 미명시; 값의 scale은 Table 1 Qwen1.5-1.8B-Chat row와 같음) | not reported | preference | best benchmark average의 \(\alpha\) | \(5e{-6}\)(Table 2: 41.8) | arXiv:2405.17931v1 Table 2 | conflict | Table 2, \(\alpha\)당 one run |
| OnDARE merging weight \(\alpha\) | not reported | preference | best benchmark average의 \(\alpha\) | \(5e{-7}\)(text) | arXiv:2405.17931v1 §5.3 prose | conflict | 같은 논문 Table 2와 모순 |
| WARM reward model (PaLM-XXS) | not reported | merge | \(M\); members | OOD accuracy로 고른 best RM \(M\)개의 uniform average; main RL 결과는 \(M = 6\) | arXiv:2401.12187v1 App. B.3; §5.2 | verified 2026-09-14 ([[warm-weight-averaged-reward-models]]) | Fig. 6–10에서 \(M\) 비교 |
| WARP policy (Gemma “7B”) | “7B” | merge | method; \(M\); \(\lambda\); LITI \(\eta\); iterations | layer별 task vector SLERP; 2; 0.5; 0.3; 5 | arXiv:2406.16768v1 §4, §4.4 | verified 2026-09-14 ([[warp]]) | Fig. 3(c) \(\lambda\); App. D.3 Fig. 16 \(\eta\); Table 1에서 iteration 3 뒤 plateau |
| SmolLM3-3B | 3B | merge | recipe | APO checkpoint soup, 이어서 linear \(0.9\) soup + \(0.1\) mid-training checkpoint; ingredient count 미보고 | huggingface.co/blog/smollm3, “Model Merging” | verified 2026-09-15 ([[smollm3-model-merging]]) | blog: “achieved the best performance”; 대안 미보고 |
| Qwen2.5-Math-7B-Instruct (context-first recipe) | 7B | long-context | reasoning SFT 전 merge | \(0.7 \times\) (RoPE \(\theta\times16\) model) + \(0.3 \times\) Qwen2.5-7B-Instruct-1M | arXiv:2505.17315 §3.5, Table 7 | verified 2026-09-14 ([[longer-context-deeper-thinking]]) | Table 7: MATH500 avg 88.70 vs 88.20 (\(\theta\times16\) only) |

**작은 범용 run의 시작점.** 같은 recipe의 run을 평균할 때는 test suite와 분리된 validation set으로 greedy-soup rule을 쓴다. 이는 CLIP, ALIGN, ViT-G, BERT-base, T5-base fine-tuning에서 시험됐고([[model-soups]]), Tülu 3 8B와 70B SFT의 2-seed soup은 best score를 최대 0.1점만 바꿨다([[tulu-3-seed-soups]]). 이전 checkpoint에서 능력을 회복할 때는 \(\alpha = 0.5\)(WiSE-FT, CLIP image classification)에서 sweep을 시작하고 long context에 SmolLM3-3B가 쓴 0.9/0.1 point(§1 표기에서 \(\alpha = 0.9\))를 포함한다. 각 point에서 target suite와 회복하려는 능력을 둘 다 측정한다. Validation set 없이 같은-base expert를 병합할 때는 top-20% trimming과 \(\lambda = 1\)의 TIES가 T5, T0-3B (IA)3, CLIP model에서 최대 11개 task에 대해 시험된 default다.

## 일반화 관점

**(a) breadth를 늘리는 것.**
- Pre-trained model 쪽으로 다시 interpolate하면 target accuracy를 유지하면서 distribution shift에서 accuracy가 오른다. Fine-tuned CLIP 대비 \(\alpha = 0.5\)에서 five shift는 +8.3점, ImageNet은 +0.6점이다([[wise-ft]], Table 1).
- 많은 run을 평균하면 in-distribution accuracy가 내려가도 out-of-distribution accuracy가 오를 수 있다. Uniform soup은 best single CLIP ViT-B/32 run 대비 shift에서 +3.62점, ImageNet에서 −0.41점이다([[model-soups]], Table 3).
- Large instruction-tuned base의 expert를 merge하면 base보다 held-out generalization이 개선된다. 8개의 merged 64B PaLM-2-IT expert에서 normalized 1.09다([[model-merging-at-scale]], Table 2).
- Objective-level merging은 data mixing이 trade-off시키는 두 axis를 개선할 수 있다. DPO의 SLERP는 mixture의 win rate 71.0, harm −54.69와 비교해 78.0 win rate, −57.8 harm이다([[mix-data-or-merge-models]], Table 1).
- RLHF에서 iterated policy merging은 Gemma “7B”의 보고된 여섯 zero-shot benchmark 모두에서 점수를 올린다([[warp]], Table 2).

**(b) narrowing 또는 forgetting을 일으키는 것.**
- Expert를 추가하면 held-in retention이 낮아지며, small non-instruction-tuned base에서 가장 크다(1B PaLM-2: 2개에서 8개 expert로 0.66 → 0.39)([[model-merging-at-scale]], Table 3).
- Average가 그럭저럭 보여도 merge가 skill 하나를 완전히 없앨 수 있다. TIES instruction + code merge는 HumanEval과 MBPP에서 0.0점이다([[dare-merging]], Table 1).
- Shared base에서 멀어진 model로 만든 merge는 실패한다. Llama-2-13b를 base로 쓰면 10% delta dropout이 WizardCoder-Python-13B를 63.41에서 0.0 pass@1로 만든다([[dare-merging]], §4.6).
- DPO model을 SFT 쪽으로 되돌리는 offline merge는 Qwen1.5 model의 preference score를 낮춘다(AlpacaEval LC −0.41부터 −1.65)([[online-merging-optimizer]], Table 1).
- WARM 저자는 reward model의 summary length처럼 모든 ingredient가 의존하는 spurious feature는 weight average도 유지할 가능성이 크다고 쓴다([[warm-weight-averaged-reward-models]], §6). WARP에서는 output length가 KL과 함께 증가한다(App. E).
- Weight average는 ensemble처럼 calibration을 개선하지 않는다([[model-soups]], §5).

**(c) 이 stage에서 측정하는 방법.**
- Task별 held-in retention(expert 각각에 정규화)과 held-out ratio(base에 정규화)를 mean 옆 minimum과 함께 보고한다(§7; [[model-merging-at-scale]] §3).
- 같은 data의 multitask-trained baseline을 포함한다. 이것 없이는 “merge가 generalize한다”와 “어떤 multi-task model이든 generalize한다”를 분리할 수 없다(8B PaLM-2에서 multitask held-out 1.62 vs merge 1.00, [[model-merging-at-scale]] Table 4).
- \(\alpha, \lambda, k, p\)와 ingredient는 reported test suite가 아니라 validation split에서 선택한다([[model-soups]], Recipe 1; [[wise-ft]] Table 1의 column별 “optimal \(\alpha\)” row와 대비).
- Run-to-run spread(Tülu 3 70B SFT seed span 2.6점, [[tulu-3-seed-soups]])와 judge noise(AlpacaEval LC standard deviation 약 0.5, [[online-merging-optimizer]] App. B)를 보고한다.
- Long context(RULER, [[smollm3-model-merging]]), safety harm rate([[mix-data-or-merge-models]]), calibration([[model-soups]] §5)을 포함해 merge가 움직일 수 있는 능력을 각각 평가한다. Suite 설계는 [[ch-00]], [[ch-47]]을 본다.

## 흔한 실수와 검출 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| 같은 pre-trained base를 공유하지 않거나 delta가 큰 model(중간에 continued pre-training)이 merge됨 | Merge 또는 DARE dropout 뒤 하나 이상의 task가 거의 0으로 붕괴 | 모든 ingredient의 base checkpoint를 확인한다. Delta magnitude percentile을 비교한다(SFT delta는 대체로 0.002 이내, 500B-token continued pre-training 뒤에는 0.01 초과, [[dare-merging]] Fig. 10) |
| Merged task 전체의 평균만 보고 | Task 하나가 0.0인데 average는 expert에 가깝게 보임([[dare-merging]] Table 1) | Task별 retention 표와 task 전체 minimum |
| Test suite에서 \(\alpha\), \(\lambda\), soup ingredient 선택 | 새 benchmark에서 사라지는 merge gain | Test와 분리된 validation split을 유지하고 fixed-\(\alpha\) row( [[wise-ft]]의 \(\alpha = 0.5\))와 비교 |
| Weak run 또는 high-learning-rate run을 포함한 uniform averaging | Uniform soup이 best ingredient보다 낮음([[model-soups]] Table 3: ImageNet 79.97 vs 80.38) | 각 pair midpoint를 평가하고 greedy rule 사용 |
| Merge가 자기 task에서 multitask training과 같을 것으로 기대 | Held-in score가 mixed-data model보다 낮음([[ties-merging]] Table 1: 66.4 vs 73.1) | Multitask baseline을 학습하거나 재사용하고 둘 다 보고 |
| Noise 안의 judge-score 변화를 gain으로 취급 | 보고된 judge standard deviation보다 낮은 +0.05~+0.28 AlpacaEval LC gain | Judging을 반복하고 reported standard deviation(약 0.5, [[online-merging-optimizer]] App. B)과 비교 |
| 두 능력을 모두 재지 않고 recovery merge weight를 선택 | Nominal long-context ability가 가장 강한 endpoint가 long-context와 reasoning 평가에서 intermediate weight보다 낮음([[longer-context-deeper-thinking]] Table 9, ratio 1.0) | 적어도 세 weight를 sweep하고 각각에서 RULER 및 short-context suite 평가 |
| Qwen2.5 Online Merging Optimizer를 running merged checkpoint라고, 또는 Gemma 2 averaging을 완전한 EMA-SLERP-LITI WARP pipeline이라고 서술 | Report로부터 재현할 수 없는 recipe | Qwen2.5 §4.2는 optimizer 이름만 말한다. Update rule은 [[online-merging-optimizer]] Eq. 3–7(OnDARE, OnTIES)이며 보고서는 어느 variant, \(\alpha\), \(p\)도 말하지 않는다. Gemma 2 §4는 각 phase 및 다른 hyperparameter run의 model을 평균한다고 쓰며 EMA, SLERP, LITI는 설명하지 않는다([[gemma-2]]) |
| Weight average가 ensemble처럼 calibrated라고 가정 | Soup 뒤 calibration error는 개선되지 않지만 같은 model의 ensemble은 개선 | Calibration을 별도로 측정([[model-soups]] §5) |

## 이해 확인

1. Independently initialized network 두 개를 interpolate하면 random accuracy가 나오지만, pre-trained CLIP model과 그 fine-tuned version을 interpolate하면 두 endpoint를 모두 이긴다. 두 번째 pair의 어떤 성질이 이 차이를 설명하는가? 또 [[dare-merging]]의 어떤 측정이 language model pair가 그 성질의 precondition을 충족하지 않을 수 있음을 보이는가?
2. CLIP ViT-B/32 sweep에서 uniform soup은 ImageNet에서는 best single model보다 나쁘지만 distribution shift에서는 더 낫다. 많은 run의 평균이 in-distribution accuracy보다 out-of-distribution accuracy에 더 도움이 될 수 있는 causal account를 제시하고, 그 account 중 어느 부분이 측정값이 아니라 해석인지 밝혀라.
3. Parameter 하나의 task value가 +0.5, +0.4, −0.6이면 averaging은 0.10, TIES는 0.45를 만든다. TIES 값이 처음 두 task를 더 잘 보존할 수 있는 이유를 설명하고, majority sign을 elect하는 것이 averaging보다 세 번째 task에 더 해로운 사례를 기술하라.
4. DARE는 살아남은 delta를 \(1/(1-p)\)로 rescale하지만 Online Merging Optimizer는 rescaling을 생략한다. Single merge에서 rescaling이 무엇을 보존하는지, 그리고 매 optimizer step에 적용한 같은 rescaling이 왜 merge를 불안정하게 만들 수 있는지 설명하라.
5. Merging-at-scale table에서 merge가 held-out task에서 multitask training을 이기는 것은 64B instruction-tuned base뿐이다. Large instruction-tuned base에서 merging이 더 잘 작동하게 만드는 mechanism 두 개를 제안하고, 둘을 분리할 experiment를 기술하라.
6. SmolLM3는 mid-training checkpoint 쪽으로 0.9/0.1 linear merge하여 RULER를 회복했다. Long-context checkpoint에 0.1 weight를 주어 잃어버린 long-context score의 10%보다 많이 회복하려면 어떤 조건이 필요한가? 어떤 측정으로 그것이 일어났는지 보일 수 있는가? 그런 merge를 release하기 전에 어떤 추가 평가를 요구하겠는가?
7. WARM은 reward model을 평균하며, WARM 저자는 모든 member가 의존하는 spurious feature를 average가 유지할 가능성이 크다고 쓴다. 이 한계가 RL optimization pressure와 어떻게 상호작용하는지 설명하고, 이를 줄일 diversity source 하나를 제시하라.

## 연결

- 이전: [[ch-30b]] — *Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares.* Merge가 피하려는 data-side interference.
- 다음: [[ch-31]] — *Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation.*
- [[ch-30a]] — *Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control.* §1, §5, §6이 다루는 forgetting과 alignment tax를 정의한다.
- [[ch-06]] — *Checkpointing, In-Loop Evaluation, and Checkpoint Selection.* Run 하나 안의 checkpoint averaging.
- [[ch-01]] — *Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability.* Online Merging Optimizer를 포함한 optimizer-level retention.
- [[ch-32b]] — *Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression.* §6 merge가 회복하려는 long-context regression.
- [[ch-34]] — *Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports.* 맥락 속 Qwen post-training pipeline.
- [[ch-38]] — *KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax.* WARP의 EMA anchor와 LITI가 수정하는 KL regularization.
- [[ch-41]] — *Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization.* WARM이 겨냥하는 reward-model over-optimization.
- [[ch-00]] — *What General Capability Means and How It Is Measured*; [[ch-47]] — *Evaluation Harness and Suite Design for General Capability.* Merge의 held-in/held-out evaluation용 suite 설계.
- [[ch-58a]] — *Open General-Model Recipes End to End: Pretraining to Merge.* Complete recipe에서 merging의 위치.

## 출처

- [[wise-ft]] — zero-shot과 fine-tuned weight 사이 interpolation, linear mode connectivity condition, Table 1 수치, \(\alpha = 0.5\) 권고(excerpt; library card 아직 없음).
- [[model-soups]] — uniform/greedy soup, CLIP/ViT-G/BERT/T5 결과, high learning rate의 error barrier, calibration 한계(excerpt).
- [[task-arithmetic]] — task-vector 정의, addition/negation 결과, task vector cosine similarity, \(\lambda\) 범위(excerpt).
- [[ties-merging]] — trim, elect, disjoint-merge algorithm, in-domain/held-out 결과, ablation, no-validation default.
- [[dare-merging]] — drop-and-rescale formula, 13B decoder merge table, delta가 클 때 실패(excerpt).
- [[model-merging-at-scale]] — PaLM-2 1B–64B merge grid, held-in/held-out normalized table, text-table discrepancy(excerpt).
- [[mix-data-or-merge-models]] — SFT와 DPO에서 safety/general quality의 merge-versus-mix 비교, language merge.
- [[online-merging-optimizer]] — offline SFT-DPO merge, OnDARE/OnTIES update rule, 결과, judge noise, \(\alpha\) discrepancy(excerpt).
- [[warm-weight-averaged-reward-models]] — weight-averaged reward model, label-noise/OOD 결과, spurious-feature 한계.
- [[warp]] — EMA anchor, RL task vector SLERP, LITI, iteration 결과와 length growth.
- [[gemma-2]] — 다른 hyperparameter run의 post-training averaging(§4).
- [[qwen-2.5]] — Online Merging Optimizer를 쓴 DPO(§4.2).
- [[smollm-3]] 및 [[smollm3-model-merging]] — RULER 회복을 위한 APO-soup + mid-training-checkpoint merge(official blog excerpt).
- [[llama-3]] — RM, SFT, DPO stage의 model averaging; annealing 동안 Polyak averaging.
- [[tulu-3]] 및 [[tulu-3-seed-soups]] — SFT seed variance와 seed soup(Table 14 excerpt).
- [[longer-context-deeper-thinking]] — reasoning SFT 전 1M-context model merge-ratio sweep.
- [[longred]] — continual-learning baseline으로 original과 context-extended Llama-3-8B 평균(Table 7).
- [[kimi-k1-5]] — long2short baseline으로 weight-averaging merge.
