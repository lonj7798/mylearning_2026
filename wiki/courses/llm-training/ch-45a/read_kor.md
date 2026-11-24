# 제45a장 — 선호 최적화와 RL 단계 레시피를 나란히 비교하기

> **핵심 통찰.** 공개된 SFT 이후 설정은 서로 비교할 수 있어 보이지만 실제로는 그렇지 않다. 두 보고서가 모두 "batch 512"를 적더라도 뜻이 다를 수 있다. DAPO에서 rollout batch는 프롬프트 512개인 반면 training mini-batch는 샘플 512개이며, 512 프롬프트 × 응답 16개 ÷ 샘플 512개 = rollout step당 gradient update 16회로, 논문이 명시한 값과 정확히 일치한다([[dapo]] §4.1). 두 보고서가 모두 "β"를 적더라도 뜻이 다를 수 있다. Zephyr의 β = 0.1은 정규화하지 않은 log-ratio에 쓰는 DPO temperature이고([[zephyr]] §4.4), Tülu 3의 β = 5는 토큰당 정규화한 log-ratio에 쓰는 같은 계수이며([[tulu-3-hyperparameter-tables]] 표 20), DeepSeek-R1-Zero의 0.001은 loss 안의 KL estimator에 곱해지고([[deepseek-r1-recipe]] §2.1), Tülu 3의 RLVR β = 0.05는 {0, 1} 보상에서 빼는 값이다([[tulu-3-hyperparameter-tables]] 표 21). 이런 실행에서 유통되는 값 가운데 3분의 1은 1차 출처에 아예 없다. InstructGPT는 PPO policy learning rate를 한 값으로 공개하지 않았고, 2.55e-6에서 2.55e-5까지의 sweep 범위와 선택 규칙만 공개했다([[instructgpt-ppo-appendix]] 부록 E.9).
>
> **지침.** 보고서에서 수치를 복사할 때는 그 수치의 단위, 단계, 모델 크기, 출처 안의 위치를 같은 행에 함께 복사하라. 범위를 떼어 낸 수치가 바로 이 레저가 막으려는 실패 원인이기 때문이다. 논문 표, 논문 본문, 공개 launch script가 서로 다르면, 공개 checkpoint에 대한 질문에는 script를 우선하고 다른 행도 보존하라. 예를 들어 Tülu 3 70B RLVR은 표 캡션의 β = 0.07과 script의 `--beta 0.07`은 서로 일치하지만, 본문의 "β = 0.7"과는 충돌한다([[tulu-3-hyperparameter-tables]] 표 21, §6.4). 한 설정을 보고하는 것이 아니라 고르는 상황이라면 초반 효율을 올리는 축보다 도달 가능한 상한을 올리는 축을 우선하라. 400,000 GPU-hour ScaleRL 연구에서 loss aggregation, curriculum, length penalty, advantage normalization은 주로 효율 지수 B를 바꿨고, loss type, logit precision, batch size는 상한 A를 바꿨기 때문이다([[scalerl]] 초록, §3.2). 한 행을 새 base model로 옮겨야 한다면 RL 설정을 그대로 고정하기 전에 프롬프트 난이도를 먼저 다시 조정하라. 하나의 RL 설정을 공개 base model 10개에 고정했을 때 Mistral-7B는 MATH level 3–5에서 붕괴했고 Qwen2.5-7B는 쉬운 데이터에서 정체했다([[simplerl-zoo]] §3.2).

## 이 장이 범용 모델에 중요한 이유

ch-37부터 ch-45까지의 각 장은 하나의 목적함수 또는 하나의 파이프라인 형태를 소개했다. 이 장은 그 형태들을 실제 연구소가 실행한 수치에 고정하고 검증할 수 있는 형태로 정리하는 곳이다. 이것이 범용 모델에 중요한 이유는 구체적으로 세 가지다.

첫째, 공개된 SFT 이후 단계 대부분은 좁은 평가로 tuning되었다. Tülu 3의 DPO 알고리즘 선택은 하나의 데이터셋 UltraFeedback에서 development 평균을 기준으로 결정되었다([[tulu-3-hyperparameter-tables]] 표 18). DAPO, Skywork-OR1, AceReason-Nemotron, MiMo-7B, Phi-4-reasoning-plus는 모두 수학에서 RL 설정을 선택했고, 이 다섯 개 중 세 보고서는 수학과 코드를 벗어난 평가를 전혀 하지 않는다. 그러므로 그런 실행에서 복사한 행은 그 행을 만든 실행의 좁은 범위를 함께 지닌다. 그 실행이 무엇을 측정했는지 아는 것만이 방어책이다.

둘째, 두 단계가 훼손할 수 있는 대상이 다르다. 선호 단계는 고정된 pair 집합을 소비하고, 그 pair가 표현한 만큼만 policy를 바꾼다. 위험은 rejected 쪽 gradient의 방향(ch-43a)과 pair 분포의 범위다. 반면 RL 단계는 자체 데이터를 생성하므로, prompt filter, length cap, reward shape가 모델이 어떤 행동을 reinforcement 받으며 보게 되는지를 결정한다. 그래서 레저의 열도 다르다. 한쪽에는 pair와 그 출처가 있고, 다른 쪽에는 step당 프롬프트와 프롬프트당 샘플 수가 있다.

셋째, 설정은 실행한 compute와 함께 있을 때만 의미가 있다. ScaleRL 연구는 RL compute에 대한 보상에 포화 곡선을 맞추었고, 16,000 GPU-hour 예산과 100,000 GPU-hour 예산 사이에서 두 레시피의 순위가 뒤집힐 수 있음을 보였다([[scalerl]] §5, 그림 10). compute budget이 붙지 않은 행은 사용할 수 없는 행이다.

이 장은 이 단계들이 조합되는 파이프라인을 다룬 ch-45 뒤에 놓이고, 같은 열들이 turn 차원을 갖게 되는 multi-turn agentic setting의 RL 단계를 다루는 ch-45b 앞에 놓인다.

## §1 레시피 행에 반드시 있어야 하는 것과 단위 함정

**정의.** 레시피 행은 이름 붙은 하나의 checkpoint의 한 단계에서 쓰인 하나의 설정이며, 1차 출처 안의 위치(locus)와 그 값을 어떻게 확인했는지를 나타내는 status를 함께 가진다. 열 집합은 style-guide §5.2에서 고정한다. model, size, stage, setting, value, source location, status, 그리고 그 값을 선택한 ablation이다.

**측정 가능한 문제.** 값은 행 사이를 옮겨 다닌다. 이 과정의 자체 카드에 대한 2026-09 감사는 논문에 인쇄되지 않은 InstructGPT PPO learning rate, 논문에 인쇄되지 않은 DeepSeekMath의 "16 prompts × 64 completions" batch 분해, 보고서가 아닌 launch script에서 가져온 Tülu 3 RLVR episode count를 발견했다. 이들은 각각 독자가 실제로 실행할 구성을 조용히 바꾸는 단 하나의 수치다.

### 1.1 프롬프트, 샘플, pair, episode, step

서로 다른 다섯 수량이 흔히 모두 "batch"라는 같은 단어로 출력된다.

1. **step당 프롬프트(prompts per step)** — rollout이 다루는 서로 다른 질문의 수다.
2. **프롬프트당 샘플(samples per prompt)** (`G`, `n`, "responses per query", "group size") — 질문 하나마다 생성하는 completion의 수다.
3. **step당 샘플(samples per step)** = 프롬프트 수 × 프롬프트당 샘플 수다.
4. **mini-batch** — gradient update 한 번의 단위다. step당 샘플을 mini-batch로 나누면 rollout당 gradient update 횟수가 나오며, 이것이 step 내부의 off-policyness 정도다.
5. **episode** — step당 수량이 아니라 중단 예산으로 쓰는, 생성된 sequence의 누적 개수다.

**계산으로 확인하는 예시(DAPO).** 논문은 "prompt batch size is 512 and we sample 16 responses for each prompt. For training, the mini-batch size is set to 512, i.e., 16 gradient updates for each rollout step"라고 쓴다([[dapo]] §4.1). 따라서 다음과 같다.

- step당 샘플 = 512 × 16 = 8,192;
- step당 gradient update = 8,192 ÷ 512 = 16이며, 인쇄된 문장과 일치한다;
- 따라서 첫 번째 512는 프롬프트 수이고 두 번째 512는 샘플 수다.

두 값을 모두 프롬프트 수로 읽으면 step당 gradient update는 1회가 되고 off-policy 재사용량에는 16배 오류가 생긴다. 같은 레시피의 verl 설정에는 논문이 출력하지 않은 수량도 있다. `train_batch_size: 512`에 대해 `gen_batch_size: 1536`이므로, all-correct / all-wrong filter를 통과한 group 512개를 얻기 위해 프롬프트를 최대 1,536개 생성한다. 즉, 학습 샘플 8,192개마다 생성 샘플을 최대 24,576개 만든다([[verl-dapo-recipe]]).

**계산으로 확인하는 예시(Tülu 3 8B RLVR).** 보고서의 effective batch 224([[tulu-3-hyperparameter-tables]] 표 21)는 launch command로 정확히 재현된다. actor GPU 7개 × `--local_rollout_batch_size 32` = 224다. `PPO update iterations K = 4`와 mini-batch 하나를 쓰므로, 샘플 224개로 구성한 rollout 하나는 gradient update 4회를 만든다. 표의 "Total Episodes 100,000"은 100,000 ÷ 224 ≈ rollout step 446개, gradient update 약 1,786회에 해당한다. 그러나 공개 script는 `--total_episodes 10000000`이라고 적고 있다. checkpoint는 "every 100 training steps"마다 평가했으며 MATH와 IFEval에서 선택했기 때문에, 이 budget은 실행에서 소진하지 않았다. 실제로 실행한 step 수는 보고되지 않았다.

대화형 보조 자료 [figures/step-budget-normalizer.html](figures/step-budget-normalizer.html)는 이 장의 모든 RL 행에 이 산술을 적용하고, 각 출처가 출력한 수량과 거기서 도출한 수량을 보여 준다.

### 1.2 status도 값의 일부다

아래 레저에는 네 가지 status가 등장한다. `verified`는 명시한 날짜에 명시한 위치를 읽어 확인했다는 뜻이다. `not reported`는 열거한 위치를 확인했으나 값이 없다는 뜻이다. 예를 들어 DeepSeekMath의 clip ε는 Eq. 1, 3, 15에서 기호로만 등장한다([[grpo-recipe]]). `derived`는 출력된 값으로 계산했으며 계산식을 함께 제시한다는 뜻이다. `conflict`는 두 출처가 맞지 않아 두 행을 모두 보존했다는 뜻이다.

다섯 번째 범주는 status도 행도 없다. 유통되지만 어느 출처에도 없는 값이다. 그런 값은 각 표 뒤에 열거한다.

## §2 선호 단계: 실제로 달라지는 것은 무엇인가

공개된 선호 단계들을 구분하는 축은 네 가지다. 레저의 나머지 내용은 모두 이 네 축에서 따라 나온다.

### 2.1 pair 출처: off-policy, on-policy, 또는 capability gap

**정의.** pair는 두 응답 모두 학습 중인 checkpoint(또는 바로 직전 checkpoint)에서 sample했을 때 *on-policy*이고, 다른 모델이나 고정 corpus에서 왔을 때 *off-policy*다.

- 완전 off-policy: Zephyr는 prompt마다 UltraFeedback의 응답 네 개를 가져와 GPT-4 평균 점수가 가장 높은 응답을 chosen response로 만들고, 나머지 세 개 중 "무작위로 하나(one of the remaining three at random)"를 rejected response로 만든다. 일부러 최하위 응답을 고르지 않는데, "다양성을 장려하고 DPO 목적함수를 더 어렵게 만들기(to encourage diversity and make the DPO objective more challenging)" 위해서다([[zephyr]] §4.1).
- capability-gap pair: SmolLM3는 Qwen3-32B의 응답을 chosen response로, Qwen3-0.6B의 응답을 rejected response로 사용한다([[smollm3-apo]]). 이 pair는 policy에 대한 판단이 아니라 모델 크기 차이를 부호화한다.
- on-policy resampling: Qwen2.5는 SFT 모델로 수학, 코딩, instruction following, 논리 추론의 새 query에 대해 응답을 다시 sample한다. 통과한 응답은 positive로, 실패한 응답은 negative로 유지해 약 150,000개 pair를 만든다([[qwen-2.5-recipe]] §4.2).
- 혼합형·round 기반: Llama 3.1은 6개 round 각각에서 새 선호 annotation을 수집하고, 이전 round의 가장 좋은 모델에서 나온 선호 데이터 가운데 가장 최근 batch를 "주로(primarily)" 사용해 DPO를 학습한다. label이 "significantly better" 또는 "better"인 pair만 남긴다([[llama-3-recipe]] §4.1.4, §4.1.6, §4.2.1).

**범용 모델에 대한 함의.** pair의 출처는 이 단계가 무엇을 가르칠 수 있는지를 제한한다. capability-gap pair는 policy가 더 큰 teacher처럼 보이도록 가르치므로, policy 자신의 오류를 고치는 신호가 아니라 distillation 신호다(ch-35). rejected 쪽이 policy 자체의 실패인 on-policy pair만이, 확률을 낮추는 방향으로 밀어 내는 probability mass가 처음부터 그 policy에 속했던 구성이다(ch-43a).

### 2.2 loss 변형과 각 변형에서 β가 뜻하는 것

DPO 목적함수([[dpo]])는 다음과 같다.

```
L_DPO = −log σ( β · [ (log π_θ(y_c|x) − log π_ref(y_c|x)) − (log π_θ(y_r|x) − log π_ref(y_r|x)) ] )
```

여기서 `x`는 prompt, `y_c`와 `y_r`는 chosen response와 rejected response, `π_θ`는 policy, `π_ref`는 동결한 reference, `σ`는 logistic function, `β`는 scalar temperature다. 대괄호 안의 항은 **token 전체에 대한 합(sum)** 이다. [[tulu-3-hyperparameter-tables]] 표 18의 "DPO-norm"이며 [[simpo]]의 정규화를 따르는 length-normalized DPO는, β를 곱하기 전에 각 대괄호 항을 응답 token 길이로 나눈다.

**계산 예시.** chosen response가 200 tokens이고 합산 log-ratio가 +4.0 nats, rejected response가 400 tokens이고 합산 log-ratio가 −6.0 nats인 pair를 생각하자.

- β = 0.1인 일반 DPO: argument = 0.1 × (4.0 − (−6.0)) = 1.0; σ(1.0) = 0.731; loss = −log 0.731 = 0.313.
- β = 5인 length-normalized DPO: argument = 5 × (4.0/200 − (−6.0/400)) = 5 × (0.020 + 0.015) = 0.175; σ(0.175) = 0.544; loss = 0.609.

같은 pair, 같은 policy, 그리고 50배 큰 β에서도 margin은 *더 약하다*. 정규화가 log-ratio 차이를 10.0에서 0.035로 줄였기 때문이다. 정규화한 상태에서 일반 DPO의 1.0 margin을 맞추려면 β ≈ 1.0 / 0.035 ≈ 28.6이 필요하다. 그러므로 β = 5(Tülu 3, Olmo 3)와 β = 0.1(Zephyr, Llama 3.1, 일반 DPO)은 하나의 척도 위의 두 점이 아니다.

**여기서 변형이 중요했다는 증거.** [[tulu-3-hyperparameter-tables]] 표 18에서 early Tülu 3 SFT checkpoint의 UltraFeedback 점수는 55.7이다. β = 0.1인 일반 DPO는 55.2를, SimPO는 51.8과 52.9를 내는 반면, β = 5인 length-normalized DPO를 1 epoch 학습하면 57.3이 된다. 같은 변형도 β = 2이면 46.8, 3 epochs이면 53.4다. 행마다 실행은 1회이고, dataset 하나와 development average 하나만 썼다. 따라서 선택은 실제였지만 범위가 좁다(**결과(단일 연구)**).

레저에 있는 다른 변형도 있다. Nemotron-4-340B는 chosen과 rejected likelihood가 모두 떨어지는 현상을 관찰한 뒤 DPO에 chosen response의 SFT loss를 더하고, 이후 세 iteration 동안 DPO를 Reward-aware Preference Optimization으로 교체한다([[nemotron-4-synthetic]] §3.3.2). Llama 3.1도 같은 명시된 이유로 chosen sequence에 coefficient 0.2의 NLL term을 추가한다([[llama-3-recipe]] §4.1.4). SmolLM3는 Anchored Preference Optimization을 쓴다([[smollm3-apo]]). 변형 전체의 survey는 [[hf-dpo-zoo]]다.

### 2.3 epochs와 overfitting을 측정한 한 사례

Zephyr는 DPO를 1~3 epochs 학습했고, "DPO training 1 epoch 후 모델이 완벽한 training set accuracy로 표시되는 강한 overfit을 보였다(after one epoch of DPO training, the model would strongly overfit, as indicated by perfect training set accuracies)"고 보고했다. 그럼에도 MT-Bench와 AlpacaEval에서 가장 좋았던 모델은 SFT 1 epoch 뒤 DPO 3 epochs를 학습한 모델이었다. 또 SFT 모델을 1 epoch보다 오래 학습하면 "더 긴 학습에서 DPO step이 실제로 성능 하락을 유발한다(the DPO step actually induces a performance regression with longer training)"고 했다([[zephyr]] §5, 그림 3). Tülu 3는 반대로 1 epoch를 선택했다. 자체 표 18은 서로 다른 base와 dataset에서 같은 β = 5 설정이 1 epoch에서 57.3, 3 epochs에서 53.4임을 보인다. 둘 다 단일 연구이고 서로 반대 방향을 가리킨다. 따라서 epoch는 복사할 값이 아니라 다시 측정할 값이다(**열린 질문**).

### 2.4 reference model과 다른 것은 바꾸지 않는 memory 절약 기법

Tülu 3는 reference model을 memory에 계속 보관하는 대신 dataset 전반의 reference log-probability를 미리 계산해 cache한다([[tulu-3-hyperparameter-tables]] §5.4.2). `π_ref`가 동결되어 있는 한 이는 정확히 동등하며, 이 장의 모든 offline preference 행에서 `π_ref`는 동결되어 있다. reference가 움직이는 online preference 방법에서는 동등하지 않다. 그래서 레저는 loss와 reference policy를 별도 열에 기록한다.

## §3 RL 단계: 동작을 결정하는 다섯 열

### 3.1 KL: 계수, 넣는 위치, 그리고 target

세 가지 배치 방식이 등장하며, 서로 바꾸어 쓸 수 없다.

1. **reward 안의 KL.** InstructGPT는 β = 0.02에서 `r(x,y) = r_φ(x,y) − β log(π^RL(y|x) / π^SFT(y|x))`를 쓴다([[instructgpt-ppo-appendix]] 부록 C.4). Llama 2는 `R = R̃_c − β D_KL(π_θ ‖ π_0)`를 쓰며 β는 7B·13B에서 0.01, 34B·70B에서 0.005다([[llama-2-recipe]] §3.2.3, 식 4). Tülu 3 RLVR은 RM score 대신 verifier의 binary output을 넣은 같은 형태를 쓰고, β는 8B에서 0.05, 70B에서 0.07이다.
2. **loss 안의 KL.** DeepSeek-R1-Zero와 R1 stage 1은 coefficient 0.001로 식 2의 KL estimator를 loss에 더한다([[deepseek-r1-recipe]] §2.1, §3.2.1). DeepSeekMath는 0.04를 쓴다([[grpo-recipe]] §4.2). SimpleRL-Zoo는 0.5B–14B 모델에서 1e-4, 14B 초과에서 1e-3을 쓴다([[simplerl-zoo]] 부록 B.5). Tülu 3.1은 `kl3` estimator와 함께 `--beta 0.01`을 쓴다([[tulu-3-hyperparameter-tables]], open-instruct command).
3. **KL을 전혀 쓰지 않음.** DAPO는 이 항을 제거한다([[dapo]]). Skywork-OR1은 다단계 학습 중 “KL penalty가 test performance의 추가 개선을 방해한다(the KL penalty hinders further improvements in test performance during multi-stage training)”는 이유로 제거한다([[skywork-or1]] §3.2.6). AceReason-Nemotron은 β = 0으로 둔다([[acereason-nemotron]] §3.2.2). MiMo-7B는 KL loss를 제거한다([[mimo-7b]] §3.3.2). Magistral은 “정당화할 수 없는 compute cost(a compute cost we find unjustified)”라며 제거한다([[magistral-recipe]] §2.1). Olmo 3 RL script는 framework default 0.05에 대해 β = 0.0을 전달한다([[allenai-olmo3-open-instruct-scripts-recipe]]).

네 번째 경우는 오독하기 쉽다. Phi-4-reasoning-plus는 β = 0.001인 `− β D_KL(π_θ ‖ π_θold)`를 표시하는데, target은 동결 reference가 아니라 **rollout policy**다([[phi-4-reasoning-rl]] §4.2). π_θold에 대한 KL은 한 update의 크기를 제한한다. π_SFT에 대한 KL은 시작 checkpoint에서의 누적 drift를 제한한다. retention control인 것은 후자뿐이다.

**reward에서 β = 0.05가 실제로 하는 일의 계산 예시.** binary verifier reward와 Tülu 3의 β = 0.05를 쓰자. reference에서 4 nats drift한 정답 응답의 점수는 `1 − 0.05 × 4 = 0.8`이고, 0.5 nats drift한 오답의 점수는 `0 − 0.025 = −0.025`다. 20 nats의 drift는 정답 하나의 reward 전체를 상쇄한다. β ∈ {0.1, 0.05, 0.03, 0.01}에 대한 Tülu 3 sweep은 한 예외를 빼면 “더 큰 KL divergence는 보통 더 낮은 average score를 낸다(more KL divergence typically results in lower average scores)”고 찾았고, 공개한 8B model은 β = 0.05를 사용했다([[tulu-3-hyperparameter-tables]] 표 21; report §6.3).

### 3.2 Clip range: 대칭, 비대칭, 또는 없음

PPO와 GRPO는 importance ratio `ρ = π_θ / π_θold`를 `[1 − ε_low, 1 + ε_high]`로 clip한다. 공개된 값은 다음과 같다.

- 대칭 0.2: InstructGPT, Llama 2, Tülu 3 RLVR, Skywork-OR1, SimpleRL-Zoo.
- 비대칭: DAPO는 ε_low = 0.2, ε_high = 0.28이다([[dapo]] §4.1; verl의 `clip_ratio_low: 0.2`, `clip_ratio_high: 0.28`로 확인, [[verl-dapo-recipe]]). Olmo 3는 0.2 / 0.272인데 script가 그러하며 paper 표에는 0.272만 인쇄되어 있다([[allenai-olmo3-open-instruct-scripts-recipe]]). Magistral은 Medium에서 ε_high를 0.26~0.28 사이에서 학습 중 조정하고, “훨씬 낮은 entropy(far lower entropy)”인 cold start 뒤 Small에서는 0.3을 쓴다([[magistral-recipe]] §2.1, §5.3). ScaleRL의 DAPO 재구성은 ε_max = 0.26이다([[scalerl]] §A.16).
- 매우 큼: DeepSeek-R1 stage-1 RL은 ε = 10을 쓴다. 더 낮은 ε는 “많은 token의 gradient를 truncate하여 성능을 떨어뜨리고(truncates gradients for many tokens and degrades performance)”, 더 높으면 불안정성을 일으킬 수 있다는 설명이다([[deepseek-r1-recipe]] §3.2.1). DeepSeek-R1-Zero의 ε는 보고되지 않았다.

같은 축 위에 네 가지 서로 다른 수가 있고, 각각 다른 논변으로 골랐으며, 어느 것도 인쇄된 sweep으로 고르지 않았다. 이 장에서 “이 값의 근거(Evidence for this value)” 열이 “ablation 보고 없음”인 가장 분명한 사례다.

### 3.3 Inner update: 엄격한 on-policy인가 아닌가

step당 sample 수를 mini-batch로 나누면 한 rollout에서 수행한 gradient update 수가 된다.

| Run | step당 samples | Mini-batch | rollout당 updates |
|---|---|---|---|
| AceReason-Nemotron | 128 prompts × 8 또는 16 | 한 update, `ρ = 1` 고정 | 1 |
| Skywork-OR1-7B stage 1 | 256 × 16 | 256 (batch = mini-batch) | 1 |
| DeepSeekMath-RL 7B | 이 단위로는 미기재 | — | 1 (“each exploration stage 뒤 single update”) |
| DeepSeek-R1-Zero | rollout당 8,192 outputs | 16 mini-batches | 16, inner epoch 1회 |
| MiMo-7B | 512 | 32 | 16 |
| DAPO | 8,192 | 512 | 16 |
| Tülu 3 8B RLVR | 224 | 224, `K = 4` | 4 |
| SimpleRL-Zoo | 1,024 × 8 | 256 | mini-batch가 sample을 셀 경우 32 (단위 미기재) |

AceReason-Nemotron은 측정 결과가 붙은 유일한 행이다. prompt당 G rollout group을 생성한 뒤 여러 번(2회 또는 4회) gradient update를 적용하면 약 100 steps에서 rapid entropy collapse가 일어난 반면, 생성 뒤 정확히 한 번만 update하면 일관되게 collapse를 막았다고 보고한다([[acereason-nemotron]] §3.2.2, 그림 3c). Skywork-OR1도 같은 방향을 보고한다. “On-policy training은 entropy collapse를 완화하고 더 높은 test performance로 이어진다(On-policy training mitigates entropy collapse and leads to higher test performance)”(§4). ScaleRL은 효과의 크기에는 동의하지 않으며 off-policy algorithm은 asymptote가 아니라 efficiency를 바꾸는 축 중 하나라고 찾는다([[scalerl]] §3.1, 그림 4a). 하지만 ScaleRL의 off-policy variant는 앞선 두 run에 없던 truncated-importance-sampling loss(CISPO)와 짝지어져 있다. 그러므로 “엄격한 on-policy가 필요하다”는 주장은 설정이 다르므로 두 번 모두 **결과(단일 연구)**일 뿐 **반복 검증됨(Replicated)**이 아니다.

### 3.4 Length: cap, schedule, 또는 shaped reward

- 고정 cap: MiMo-7B는 32,768([[mimo-7b]] §3.3.3), Tülu 3 RLVR은 응답 token 2,048, GSM8K에만 1,024를 둔다.
- 학습 중 schedule: DeepSeek-R1-Zero는 step 8.2k 뒤 32,768에서 65,536으로 올린다([[deepseek-r1-recipe]] §2.1). AceReason-Nemotron은 8K → 16K → 24K → 32K를 사용하며, “16K 또는 24K에서 직접 시작하면 suboptimal results가 났다(directly starting from 16K or 24K resulted in suboptimal results)”고 한다([[acereason-nemotron]] §3.2.2). Skywork-OR1-7B는 16K → 32K, Skywork-OR1-32B는 16K → 24K이고, 같은 경계에서 batch와 group size도 바뀐다([[skywork-or1]] 표 11, 12). Magistral Medium은 generation이 길어지면서 batch를 8k → 4k → 2k로 줄이는 동시에 16k → 24k → 32k를 쓴다([[magistral-recipe]] §5.2).
- shaped reward: DAPO의 soft overlong punishment는 20,480 tokens 중 마지막 4,096 tokens에 선형 penalty를 주어 cap에서 −1.0에 이르게 한다([[verl-dapo-recipe]]). Phi-4-reasoning-plus의 length-aware cosine accuracy reward는 25,600 tokens를 넘은 *정답*의 reward를 줄이고 3,702 tokens를 넘은 *오답*의 penalty를 줄인다([[phi-4-reasoning-rl]] §4.1). Magistral의 length penalty는 0에서 −0.1이다([[magistral-recipe]] §2.2).
- penalty 대신 강제 중단: ScaleRL은 긴 생성을 멈추려고 end-of-thinking phrase를 덧붙이고, 그 위에 length penalty를 더해도 “성능이 개선되지 않는다(does not improve performance)”고 보고한다([[scalerl]] §2, 부록 A.15).

### 3.5 Prompt filter와 gradient가 바뀌는 이유

**메커니즘.** group-relative advantage에서 prompt의 `G`개 sample이 모두 같은 reward를 받으면 모든 sample의 advantage는 0이고 policy gradient에 기여하지 않는다. 극단 근처의 prompt는 적게 기여하며 비대칭적이다.

**계산 예시.** `G = 8` samples, binary reward, pass rate `p`에서 population standard deviation `√(p(1−p))`를 쓰는 standardized group-relative advantage `Â = (r − mean) / std`를 생각하자.

| Pass rate p | std | 정답 sample의 Â | 오답 sample의 Â |
|---|---|---|---|
| 0/8 | 0 | — | 0 (gradient 없음) |
| 1/8 | 0.331 | +2.646 | −0.378 |
| 4/8 | 0.500 | +1.000 | −1.000 |
| 7/8 | 0.331 | +0.378 | −2.646 |
| 8/8 | 0 | 0 (gradient 없음) | — |

낮은 pass rate에서는 드문 성공이 지배하고, 높은 pass rate에서는 드문 실패가 지배한다. 공개된 filter가 관리하는 대상이 바로 이것이다.

- degenerate group 제거: DAPO의 dynamic sampling은 accuracy가 1 또는 0인 group을 제거하고 batch가 찰 때까지 resample한다([[dapo]] §3.3, [[verl-dapo-recipe]]). MiMo-7B도 같다([[mimo-7b]] §3.3.2). Skywork-OR1은 이를 rejection sampling이라 부르며, zero-advantage sample도 entropy와 KL term에는 들어가 상대적 weight를 왜곡하기 때문이라고 설명한다([[skywork-or1]] §3.1). Olmo 3 script는 `no_resampling_pass_rate 0.875`로 active sampling을 켠다([[allenai-olmo3-open-instruct-scripts-recipe]]).
- 모델이 개선됨에 따라 난이도 분포 이동: AceReason-Nemotron은 24K와 32K stage에서 pass rate가 6/16보다 큰 prompt를 filter한다([[acereason-nemotron]] §3.2.2, 표 3).
- 학습 전 model-specific difficulty로 filter: Skywork-OR1 dataset은 문제마다 **그리고 model variant마다** 0~16 difficulty score를 갖고, 공개 preprocessing은 DeepSeek-R1-Distill-Qwen-32B 열에서 `1 ≤ difficulty ≤ 15`만 남긴다. 따라서 동일한 수학 105,055문제와 코드 14,057문제도 policy size마다 다른 training set을 만든다([[skywork-or1]]).
- 쉬운 prompt를 의도적으로 다시 넣기: MiMo-7B는 완전히 푼 문제 pool을 보존하고 α = 10% 확률로 여기에서 sample한다. easy data를 직접 재사용하면 불안정하다는 결과 뒤의 선택이다([[mimo-7b]] §3.3.2).

**조건과 한계.** 위 pass-rate threshold는 binary verifier가 있는 수학과 코드에서 측정되었다. 이 source들 가운데 non-verifiable domain에서 pass-rate filter를 tuning했다고 보고한 것은 없다.

## §4 충돌과 각 충돌의 해소 방법

이 장의 source들 사이에 있는 여섯 가지 불일치는 평균을 내어 없애지 않고, 각각 별도 행으로 기록한다.

1. **Tülu 3 70B RLVR, β.** 표 21 caption은 최종 70B 모델이 β = 0.07을 사용했다고 말하지만, §6.4 본문은 β = 0.7이라고 말한다. 공개된 launch command는 `--beta 0.07`이다. 셋 중 둘이 0.07에 일치하고, 그 둘 중 하나는 checkpoint를 실제로 만든 artefact이므로, ledger의 released-checkpoint 행에는 0.07을 쓰고 본문 서술은 충돌 note로 남긴다([[tulu-3-hyperparameter-tables]]).
2. **Tülu 3 RLVR, episodes.** 표 21에는 100,000이 인쇄되어 있고, §6.4 본문은 70B에 400,000이라고 인쇄한다. 8B script에는 `--total_episodes 10000000`, 70B script에는 `400000`이 있다. Script 값은 실제로 소비한 episode 수가 아니라 stopping budget이므로 두 값은 서로 다른 quantity이며, 두 행을 모두 유지한다.
3. **Tülu 3 405B RLVR, 표 caption.** 표 35의 caption은 “PPO used for optimizing against a general RM”이라고 쓰지만, 열 header는 “405B RLVR”이고 §8.1은 이를 RLVR 표로 도입한다. 열 header와 주변 본문이 caption과 상충하며 서로 일치한다.
4. **DeepSeekMath, batch 단위.** §4.2는 “training batch size is 1024” 및 “max length is set to 1024”라고 인쇄하지만, 둘 중 어느 것에도 unit을 붙이지 않는다. [[grpo-recipe]]는 둘 다 “unit not stated”로 기록하고, 논문에 없는 흔한 분해인 “16 prompts × 64 completions”로 해석하는 것을 명시적으로 금지한다.
5. **DAPO 논문 대 verl recipe.** 논문은 prompt batch 512와 mini-batch 512를 인쇄한다. verl의 recipe 문서는 `train_batch_size: 512`, `gen_batch_size: 1536`, `overlong_buffer.penalty_factor: 1.0`을 인쇄하는데, 뒤의 두 값은 논문에 없다. 문서는 자신이 공개 run의 정확한 configuration이라고 주장하지 않으므로, 둘은 별도 행이다([[dapo]], [[verl-dapo-recipe]]).
6. **Olmo 3 scripts 대 paper tables.** Recipe card는 commit `d8a7f1c`의 `scripts/train/olmo3`와 arXiv:2512.13961v2 표 47–49가 GPU 수, step당 unique prompt 수, dataset 크기, maximum asynchrony에서 서로 다른 일곱 행을 기록한다. 또한 PR #1547에서 제거된 flag 하나 때문에 script가 release 시점과 다른 framework default로 조용히 실행되는 행 하나도 기록한다([[allenai-olmo3-open-instruct-scripts-recipe]]).

일곱 번째 사례는 source 간 충돌이 아니라 comment와 바로 아래 code 간 충돌이다. Skywork-OR1 preprocessing script의 comment는 mixing할 때 math data가 복제된다고 말하지만, 실제로 build하는 file list는 code split을 복제한다([[skywork-or1]]).

**돌아다니지만 primary source에는 없는 값.** InstructGPT의 PPO policy learning rate 1.41e-5와 “4 epochs per rollout”는 primary source에 없다. App. C.4는 8 mini-batch에 걸쳐 one inner epoch이라고 하고, E.9는 sweep range와 selection rule만 준다([[instructgpt-ppo-appendix]]). DeepSeekMath의 clip ε = 0.2와 rollout temperature 1.0도 그렇다([[grpo-recipe]]). Phi-4의 “+1 correct, −0.5 incorrect” reward 역시 report에 없다. Report는 이를 [+0.5, +1.0] 및 [−1.0, −0.5] 범위의 cosine-scaled, length-aware term으로 대체한다([[phi-4-reasoning-rl]] §4.1, [[phi-4]]와 대조). 마지막으로 “lr ~1e-6, β_KL ~0.04, ~128 prompts, 4 rollouts each”를 한 행으로 합친 Tülu 3 RLVR 값도 Table 21에는 전혀 없다([[rlvr-tulu3]] Technical Details).

## §5 한 행의 가치는 무엇인가: RL compute scaling

**정의.** ScaleRL은 RL compute에 따른 reward를 포화 곡선으로 적합한다([[scalerl]] Eq. 1).

```
R_C − R_0 = (A − R_0) × 1 / (1 + (C_mid / C)^B)
```

`R_C`는 `C` GPU-hours만큼 RL을 수행한 뒤 i.i.d. validation set에서 얻은 pass rate(mean@16)다. `R_0`는 시작 checkpoint의 pass rate, `A`는 점근 pass rate, `B`는 scaling exponent이며 값이 클수록 더 빠르게 상승함을 뜻한다. `C_mid`는 곡선의 중간점에 해당하는 compute다. 이 적합에서는 model과 training data를 고정한다.

**계산 예시.** [[scalerl]] Table 1의 적합된 두 행을 보자. Batch 512의 ScaleRL은 `C_mid = 2,818`, `B = 1.77`, `A = 0.605`이고, batch 2048은 `C_mid = 10,909`, `B = 1.70`, `A = 0.645`다. Budget `C`에서 각 run이 자기 자신의 가능한 gain 가운데 달성한 비율은 `1 / (1 + (C_mid/C)^B)`다.

- C = 16,000 GPU-hours일 때 batch 512는 `(2818/16000)^1.77 = 0.046`이므로 gain의 0.956을 달성한다. Batch 2048은 `(10909/16000)^1.70 = 0.522`이므로 gain의 0.657을 달성한다. `R_0`가 같다면 더 작은 batch가 앞선다.
- C = 100,000 GPU-hours일 때 batch 512는 0.605의 0.998을, batch 2048은 0.645의 0.977을 달성한다. 이 budget에서는 더 큰 batch가 앞선다.

순서가 뒤집힌다. 이것이 논문의 명시적 발견이다. “larger batch size is slower in training but settles at a higher asymptote. Batch size show an inverse trend initially where smaller values seem better at lower compute budget”(Fig. 10). [figures/scalerl-compute-curve.html](figures/scalerl-compute-curve.html)에서는 `A`, `B`, `C_mid`를 움직이며 두 곡선이 어디에서 교차하는지 볼 수 있다.

**어떤 변경이 어느 parameter를 움직였는가** (8B dense, verifiable mathematics, 세 번 반복 run으로부터 얻은 `A`의 ±0.02 error margin, [[scalerl]] Fig. 8a):

| 축 | 보고된 효과 |
|---|---|
| Loss type (CISPO, GSPO vs DAPO) | `A`를 올린다. ScaleRL 내부 leave-one-out에서는 `A`가 유사하고 `B`는 2.01 대 1.77이다. |
| LM head의 FP32 precision | forward ablation에서 `A` 0.52 → 0.61. |
| Batch size | `A` 0.605 (512) → 0.645 (2048). |
| Loss aggregation, advantage normalization, curriculum, length penalty, off-policy algorithm | “primarily modulate compute efficiency without materially shifting the asymptote”, 즉 점근값을 실질적으로 옮기지 않고 주로 compute efficiency를 조절한다. |
| Model size | Llama-4 Scout 17B×16은 8B dense model의 `A = 0.645`와 비교해 `A = 0.710`에 도달한다. |

**조건과 한계.** 위 숫자는 모두 한 model family, 한 domain, 한 codebase에서 나온 것이다. [[scalerl]] Fig. 2에서 DeepSeek (GRPO), Qwen2.5 (DAPO), Magistral, MiniMax로 표시한 비교 곡선은 해당 lab의 공개 checkpoint를 다시 평가한 것이 아니다. 저자들이 의도적으로 차이를 둔 상태에서 8B에 맞춰 그 recipe의 *shape*를 재구성한 것이다. 예를 들어 DAPO 재구성은 `ε_max = 0.26`을 사용하고, 더 큰 1,280 batch에서 zero-variance prompt를 refill하지 않고 제거한다([[scalerl]] App. A.16). 따라서 Fig. 2를 “DAPO가 ScaleRL보다 나쁘다”로 읽으면 과장이다. Appendix가 뒷받침하는 해석은 “이 codebase에서, 이 size로, 이 recipe shape가 더 낮은 수준에서 포화했다”이다.

## 음성 샘플과 음성 피드백

이 장의 두 단계는 모두 ch-43a §6.1 사례 4의 의미에서 음성을 **gradient로 사용**한다. 즉, 어떤 샘플의 likelihood를 명시적으로 낮춘다. 선호 단계에서는 DPO 계열 loss의 rejected 항이 이를 수행하고, RL 단계에서는 음의 group-relative advantage가 이를 수행한다. 이 장에서는 음성을 content나 conditioning으로 사용하지 않는다. 그런 구성은 ch-31과 ch-43a에 속한다.

**단계별 음성의 출처.**

| 단계 | 음성 출처 | 라벨러 | false-negative 비율 |
|---|---|---|---|
| 선호, off-policy | 고정 corpus에서 top이 아닌 응답 | GPT-4 점수(Zephyr, UltraFeedback) | 보고되지 않음 |
| 선호, capability gap | 더 작은 모델의 응답 | 없음 — gap을 가정 | 보고되지 않음 |
| 선호, on-policy | policy 자신이 생성한 실패 응답 | quality check와 human·automated review(Qwen2.5 §4.2) | 보고되지 않음 |
| 선호, RM-ranked | 두 응답 중 더 낮은 점수의 응답 | reward model | Nemotron-4: RM-as-judge Chat-Hard 0.87, LLM-as-judge 0.54([[nemotron-4-synthetic-recipe]]) |
| RL, verifier | verifier가 오답으로 표시한 모든 샘플 | exact match, sympy, unit test, constraint checker | 이 장의 어느 source도 보고하지 않음 |

**메커니즘.** softmax gradient는 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이다. 따라서 샘플 하나를 아래로 누르면 그 확률 질량은 모델이 이미 더 높게 rank한 항목으로 이동한다. ledger에서 보이는 결과는 두 가지다.

1. RL 단계의 음성 gradient 크기는 샘플이 얼마나 틀렸는지가 아니라 group의 pass rate가 정한다. §3.5 표에서 pass rate 7/8의 오답 샘플은 `Â = −2.646`을 받지만, pass rate 1/8의 오답은 `−0.378`을 받는다. 따라서 prompt filter는 음성-gradient 크기를 제어하는 장치이기도 하다.
2. 선호 단계에서 rejected 항의 gradient는 rejected 응답의 likelihood가 이미 낮아도 사라지지 않는다. Nemotron-4는 DPO에서 “chosen과 rejected likelihood가 모두 하락한다”는 현상을 관찰했고 chosen 응답에 SFT loss를 더했다. Llama 3.1도 계수 0.2로 같은 anchor를 넣었다([[nemotron-4-synthetic]] §3.3.2; [[llama-3-recipe]] §4.1.4). 둘 다 이 변경이 안정화했다고 보고하지만 ablation 수치는 인쇄하지 않았다.

**실제 ledger에 들어 있는 제어 장치.**

- 양의 NLL 항으로 anchor한다. Llama 3.1은 chosen에 0.2를 쓰고, Nemotron-4는 DPO SFT-loss weight를 `[1e-5, 1e-3]`에서 tuning했으며 RPO에서는 `1e-5`로 고정했다.
- normalization으로 push-down을 제한한다. length-normalized DPO는 각 log-ratio를 응답 길이로 나누므로 긴 rejected 응답이 margin을 지배하는 정도를 제한한다(§2.2 계산 예시).
- 상수 target 대신 reward gap으로 target을 scale한다. Nemotron-4 RPO는 `β(log-ratio gap)`을 `η(r*(y_c) − r*(y_l))`에 맞춘다. 명시한 목적은 고품질 rejected 응답을 “unlearning”하지 않는 것이다([[nemotron-4-synthetic]] §3.3.2).
- 음성을 on-policy로 유지한다. Qwen2.5는 외부에서 가져오지 않고 SFT 모델에서 negative를 다시 sample한다([[qwen-2.5-recipe]] §4.2).
- verifier가 판정할 수 없던 것을 벌주지 않을 수도, 벌줄 수도 있다. Skywork-OR1은 truncation 응답의 advantage mask를 시험한 뒤 제거했다. “truncated responses를 penalize해도 later-stage improvement를 방해하지 않고 token efficiency를 높인다”고 결론냈다([[skywork-or1]] §3.1, §3.2.3). 반대로 DAPO의 overlong filtering은 progressive table에서 AIME24 avg@32를 +6점(30 → 36) 높였다([[dapo]] Table 1). base model과 stage가 다르며 두 방식을 직접 비교한 source는 없다(**열린 질문**).
- content가 없는 token position을 mask한다. Llama 3.1은 chosen과 rejected 모두에서 header·termination token을 mask했고, 그렇지 않으면 tail repetition 또는 abrupt termination이 생긴다고 보고했다([[llama-3-recipe]] §4.1.4).
- advantage를 통한 간접 벌 대신 특정 실패에 명시적 벌을 준다. Tülu 3 RLVR은 EOS token이 없는 응답에 `−10.0` reward를 주고([[tulu-3-hyperparameter-tables]] Table 21), Phi-4-reasoning-plus는 `<think>` block이 없거나 malformed이면 accuracy term을 `−1.0`으로 둔다([[phi-4-reasoning-rl]] §4.1).

**source들이 기록한 진단값.** step별 entropy(Skywork-OR1 §4, AceReason Fig. 3c, Phi-4-reasoning Fig. 7f), episode별 reference 대비 KL(Tülu 3 Figs. 19–23: 이로써 70B run의 KL이 “run 내내 1보다 훨씬 낮게 유지됨”을 확인), 정답·오답별 응답 길이(Phi-4-reasoning Fig. 7d), up-clipped token probability와 accuracy 1인 prompt 비율(DAPO Figs. 3a, 3b), truncation rate가 있다. ScaleRL은 batch 768에서 “10–15% 범위 truncation은 보통 training을 불안정하게 한다”고 보고했고, ScaleRL run은 training의 90% 이상에서 5% 미만이었다([[scalerl]] App. A.15).

**효과 중 음성 항의 몫.** 이 장의 어떤 source도 gain 중 음성 항이 차지한 비율을 측정하지 않았다. 존재하는 course-wide 수치인 32B NFT 분석의 양성 약 80%, 음성 약 20%는 ch-43a에 속하므로 이 run들에 옮기면 안 된다.

**일반성에 미친 효과.** 이 장의 두 측정이 직접 관련된다. math-only RL은 7B에서 LiveCodeBench v5를 6.8점, 14B에서 5.8점 올렸지만, 비슷한 모델의 math-only SFT는 같은 benchmark에서 19.3이었다([[acereason-nemotron]] Table 1). 한 domain의 verifier가 전달한 음성은 transfer했지만 한 domain imitation은 그렇지 않았다. 반대 사례로, DeepSeek-R1 final RL은 1,700 step 중 마지막 400 step에서만 general instruction data의 preference reward를 추가하며, reward가 오를 때 Codeforces pass@1이 내려가는 그림이 함께 제시된다([[deepseek-r1-recipe]] §3.2.2, B.5 Fig. 6).

## 레시피

두 표의 단위는 다음과 같다. `pairs`는 preference pair, `prompts/step`과 `samples/prompt`는 rollout step당 수, `samples/step`은 둘의 곱, `episodes`는 source가 정의한 누적 생성 sequence budget이다. 행 하나는 이름 붙은 checkpoint 하나를 나타낸다. status는 style-guide §5.2를 따르고 날짜는 이 장 작성자가 locus를 읽은 날짜다.

### 선호 단계



| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Zephyr-7B-β | 7B | preference | loss; β; peak LR; schedule; warmup; global batch; epochs | dDPO (plain DPO); 0.1; 5e-7; linear; 10%; 32; 3 (final model) | arXiv:2310.16944 §4.4 | verified 2026-09-15 | §5 Fig. 3: best MT-Bench at 1 SFT epoch + 3 DPO epochs; >1 SFT epoch causes a DPO regression |
| Zephyr-7B-β | 7B | preference | pairs and their source | UltraFeedback, 64k prompts × 4 responses; chosen = highest mean GPT-4 score; rejected = one of the other three at random; fully off-policy | arXiv:2310.16944 §4.1 | verified 2026-09-15 | random rejection chosen "to encourage diversity and make the DPO objective more challenging"; no ablation |
| Llama 3.1 405B | 405B | preference | loss; LR; β; auxiliary NLL coefficient; token masking; rounds | DPO; 1e-5; 0.1; 0.2 on chosen sequences; header and termination tokens masked; 6 rounds | arXiv:2407.21783v3 §4.1.4, §4.1.6 | verified 2026-09-14 ([[llama-3-recipe]]) | DPO preferred over PPO for compute and "instruction following benchmarks like IFEval"; no numbers; NLL term cites Pang et al. 2024 and Pal et al. 2024 |
| Llama 3.1 405B | 405B | preference | pair filter; composition; epochs, batch, steps | only "significantly better"/"better" pairs; General English 81.99%, Coding 6.93%, Multilingual 5.19%, Reasoning and tools 5.89% of comparisons; epochs/batch/steps not printed | v3 §4.2.1, Table 6, §4.1.4 | verified 2026-09-14; epochs/batch/steps not reported | no ablation reported |
| Qwen2.5-Instruct (all 7 dense sizes) | 0.5B–72B | preference | loss; pairs; pair source; epochs; LR; optimizer; β | DPO; ~150,000 pairs; SFT model resamples responses, passing = chosen, failing = rejected; 1 epoch; 7e-7; Online Merging Optimizer; β not given | arXiv:2412.15115v2 §4.2 | verified 2026-09-14 ([[qwen-2.5-recipe]]); β not reported | no ablation reported |
| Tülu 3 8B / 70B | 8B, 70B | preference | loss; β; LR; schedule; warmup; effective batch; max length; epochs | length-normalized DPO; 5; 5e-7 / 2e-7; linear; 0.1; 128; 2,048; 1 | arXiv:2411.15124 Table 20 | verified 2026-09-15 | Table 18 (UltraFeedback, one run per row): DPO-norm β 5, 1 epoch = 57.3 vs plain DPO β 0.1 = 55.2, SimPO 51.8/52.9, SFT base 55.7; Table 19 (70B LR) |
| Tülu 3 405B | 405B | preference | LR; batch; max length; β; warmup; epochs | 2e-7; 256; 2,048; 5; 0.1; 1 | arXiv:2411.15124 Table 34 | verified 2026-09-15 | "we opted to lower the LR for larger models" (§8.1); no ablation at this size |
| Olmo 3 7B Think / 7B Instruct / 32B Think / 32B Instruct | 7B, 32B | preference | loss; β; epochs; warmup; LR; pairs; max length; batch | `dpo_norm`; 5; 1; 0.1; 8e-8 / 1e-6 / 7e-8 / 1e-6; 150,000 / 260,000 / 200,000 / 260,000; 16,384 (7B), 8,192 (32B); 128 | open-instruct `scripts/train/olmo3/*.sh` at d8a7f1c, flags at 00d42c4; arXiv:2512.13961v2 Table 48 | verified 2026-09-14 ([[allenai-olmo3-open-instruct-scripts-recipe]]); 7B Instruct GPU count is a conflict | App. A.6.2: LR and dataset size were swept per model; other settings fixed |
| SmolLM3-3B | 3B | preference | loss; pairs; pair source; context cap; β, LR, batch, epochs | APO (arXiv:2408.06266); Tülu 3 preference mixture plus synthetic reasoning pairs; chosen = Qwen3-32B, rejected = Qwen3-0.6B; 24k tokens; not reported | huggingface.co/blog/smollm3, alignment section | verified 2026-09-15; hyper-parameters not reported (blog checked) | "we also observed higher downstream performance in our internal ablations" — no numbers |
| SmolLM3-3B | 3B | merge | repair of the long-context regression | model soup of APO checkpoints, then a linear merge with weights 0.9 (soup) and 0.1 (a mid-training checkpoint) | same blog, merging section | verified 2026-09-15 | "We were able to recover the base model's RULER score on contexts up to 128k tokens"; no numbers printed |
| Nemotron-4-340B-Instruct | 340B | preference | DPO pairs; epochs; batch; LR; KL coefficient; SFT-loss weight on chosen | 160K; 1; 256; constant, tuned in [3e-8, 3e-7]; tuned in [3e-4, 3e-3]; tuned in [1e-5, 1e-3] (selected values not given) | arXiv:2406.11704 §3.3.2 | verified 2026-09-14 ([[nemotron-4-synthetic-recipe]]) | Table 6: MT-Bench 7.99 → 7.90, GSM8K 87.9 → 88.5 across the DPO stage |
| Nemotron-4-340B-Instruct | 340B | preference | RPO pairs; iterations; LR; η; SFT coefficient; β | 300K per iteration; 3; 3e-7; 1; 1e-5; tuned in [1e-3, 1.0] | arXiv:2406.11704 §3.3.2 | verified 2026-09-14 | Table 6: MT-Bench 7.90 → 8.21 → 8.31 → 8.22; IFEval prompt-strict 61.7 → 78.2 → 79.9 → 79.9; released checkpoint is iteration 3 |
| InstructGPT (PPO-ptx) | 1.3B, 6B, 175B | preference (RM + PPO) | KL coefficient and placement; clip ε; batch; mini-batch; inner epochs; episodes; unique prompts; temperature; EMA; pretraining coefficient γ | β = 0.02 subtracted from the reward; 0.2; 512; 64; 1 inner epoch over 8 mini-batches; 256k; ~31k; 1.0; decay 0.992; 27.8 | arXiv:2203.02155 App. C.4 | verified 2026-09-15 | App. E.7: "the optimal value is around 0.01 and 0.02"; App. E.11: batch 512 best of {64…1024} by human evaluation; mini-batch 32 slightly better than 64, 64 kept for GPU utilization |
| InstructGPT (PPO-ptx) | 1.3B, 6B, 175B | preference (RM + PPO) | policy learning rate | not reported; swept log-linearly from 2.55e-6 to 2.55e-5 (175B: 2.55e-6 and 3.74e-6), final models selected by Likert score | App. C.4, E.9 checked | not reported | E.9: "All runs with learning rate greater than 8.05e-6 diverged, for PPO models without pretraining data mix" |
| Llama 2-Chat | 7B, 13B | preference (RM + PPO) | KL coefficient and placement; LR; batch; mini-batch; clip ε; samples per prompt; iterations | β = 0.01 in the reward, `R = R̃_c − β D_KL(π_θ ‖ π_0)`; constant 1e-6; 512; 64; 0.2; 1; 200–400 with early stopping | arXiv:2307.09288v2 §3.2.3, Eq. 4 | verified 2026-09-14 ([[llama-2-recipe]]) | no ablation reported |
| Llama 2-Chat | 34B, 70B | preference (RM + PPO) | KL coefficient | β = 0.005, same placement | v2 §3.2.3, Eq. 4 | verified 2026-09-14 | no ablation reported |

### RL 단계(verifier 또는 rule reward)

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DeepSeekMath-RL 7B | 7B | RL | algorithm; KL placement and coefficient; LR; samples per question G; max length; training batch; updates per exploration stage; clip ε | GRPO; KL estimator added to the loss, 0.04; 1e-6; 64; 1024 (unit not stated); 1024 (unit not stated); 1; not reported | arXiv:2402.03300v3 §4.1.1, §4.2 | verified 2026-09-14 ([[grpo-recipe]]); clip ε not reported | Fig. 5 (1.3B): GRPO above Online RFT and RFT; prompts are ~144K CoT questions taken from the SFT data so that non-RL benchmarks stay measurable |
| DeepSeek-R1-Zero | 671B MoE (37B active) | RL | algorithm; KL; LR; questions/step; samples/question; samples/step; rollout split; temperature; max length; reference refresh; steps | GRPO; 0.001 in the loss; 3e-6; 32; 16; 512; 8,192 outputs per rollout into 16 mini-batches, 1 inner epoch; 1.0; 32,768 then 65,536 after step 8.2k; every 400 steps; 10,400 steps (1.6 epochs) | arXiv:2501.12948v2 §2.1 | verified 2026-09-14 ([[deepseek-r1-recipe]]) | §2.1: performance and length "exhibit a significant jump at the 8.2k step"; A.3 Fig. 4 compares PPO with GAE λ 0.95 against GRPO on a 16B MoE |
| DeepSeek-R1 (stage-1 RL) | 671B MoE | RL | clip ε; other settings | 10; LR 3e-6, KL 0.001, temp 1, 16 outputs, 32,768, 32 questions/step, batch 512 as above | v2 §3.2.1 | verified 2026-09-14 | §3.2.1: a lower ε "truncates gradients for many tokens and degrades performance"; a higher ε may cause instability; no table |
| DeepSeek-R1 (final RL) | 671B MoE | RL | temperature; steps; reward composition | 0.7; 1,700 steps, with general instruction data and preference rewards only in the final 400; rule + RM + format + language rewards | v2 §3.2.2, Eq. 8–10 | verified 2026-09-14 | §3.2.2: higher temperatures give incoherent generation; B.5 Fig. 6: with the helpful RM, reward rises while Codeforces pass@1 falls |
| Qwen2.5-Instruct | 0.5B–72B | RL | algorithm; samples per query; samples per episode; queries per episode; query ordering; LR, KL, clip ε, temperature, max length, episodes | GRPO; 8; 2,048; 256 (derived: 2,048 ÷ 8); by descending variance of the RM scores within a query's responses; not given | arXiv:2412.15115v2 §4.3 | verified 2026-09-14; derived row marked; remaining settings not reported | ordering aim stated as "to ensure more effective learning"; no ablation |
| Llama-3.1-Tulu-3-8B | 8B | RL | algorithm; KL placement and β; LR; clip ε; γ, λ; effective batch; K; response length; temperature; episodes; EOS penalty; prompts | PPO; β = 0.05 subtracted from the verifier reward; 3e-7 linear, warmup 0.0; 0.2; 1.0, 0.95; 224 (= 7 GPUs × 32); 4; 2,048 (1,024 for GSM8K); 1.0; 100,000 (table) / `--total_episodes 10000000` (script); −10.0; 29,946 verifiable prompts | arXiv:2411.15124 Tables 21–22; open-instruct `docs/tulu3.md` | verified 2026-09-15; episodes is a conflict | report §6.3: β swept over {0.1, 0.05, 0.03, 0.01}, "more KL divergence typically results in lower average scores"; Table 23: MATH 42.0 → 43.7, GSM8K 84.3 → 87.6, IFEval 81.1 → 82.4 over the DPO checkpoint |
| Llama-3.1-Tulu-3-70B | 70B | RL | β; LR; batch; response length; episodes; warmup | 0.07 (caption and script) / 0.7 (prose); 1e-7; 640 (= 40 GPUs × 16); 2,048; 400,000; 0.1 | Table 21 caption, §6.4, `docs/tulu3.md` | conflict (β); verified 2026-09-15 for the rest | Table 23: IFEval 82.6 → 83.2, MATH 62.3 → 63.0, GSM8K unchanged at 93.5 ("already close to saturation") |
| Llama-3.1-Tulu-3-405B | 405B | RL | LR; batch; K; response length; episodes; β; prompts; steps run | 1e-7; 1,856; 1; 1,024; 300,000; 0.05; MATH train only; 75 steps | arXiv:2411.15124 Table 35, §8.1 | verified 2026-09-15; Table 35's caption conflicts with its column header | §8.1: "even with as few as 25 RLVR steps, MATH performance improved by over 5 points"; run stopped early for compute reasons |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | algorithm; KL estimator and β; LR; samples/prompt; rollout and mini-batch; response length; temperature; EOS penalty; schedule | GRPO; `kl3`, 0.01; 5e-7 constant; 16; 4 prompts per process, mini-batch = half the rollout (2 updates); 2,048; 1.0; 0.0; constant | open-instruct `docs/tulu3.md`, legacy `grpo_vllm_thread_ray_gtrl.py` at commit 745bf58d321c | verified 2026-09-15 | the doc preserves the command "for historical reference"; no ablation in the doc |
| Olmo 3 7B Think (released RL run) | 7B | RL | algorithm; β; prompts × samples; minibatches; LR; prompt/response length; temperature; clip low/high; loss | open-instruct GRPO; 0.0; 64 × 8; 1; 1e-6 constant; 2,048 / 32,768; 1.0; 0.2 / 0.272; `dapo` | `7b_think_rl_no_pipeline.sh` at d8a7f1c; arXiv:2512.13961v2 Table 49 | verified 2026-09-14 ([[allenai-olmo3-open-instruct-scripts-recipe]]) | script comment: the released model did not use pipelinerl, "using it should do just as well" (no numbers) |
| Olmo 3 7B Instruct | 7B | RL | prompts × samples; minibatches; response length; active sampling; `no_resampling_pass_rate`; episodes | 64 × 8; 4; 8,192; on; 0.875; 1,024,000 | `7b_instruct_rl.sh` L42–85 | conflict: Table 49 prints dataset size 171,950 against the script's 172,000 | no ablation reported |
| Olmo 3 7B RL-Zero Math | 7B | RL | start; prompts × samples; LR; response length; episodes | `allenai/Olmo-3-1025-7B` (base); 32 × 8; 1e-6 constant; 16,384; 768,000 | `7b_rlzero_math.sh` L4–62 | conflict: Table 49 prints 8 learner and 64 actor GPUs against the script's 16 and 56 | no ablation reported |
| DAPO (Qwen2.5-32B base) | 32B | RL | algorithm; KL; optimizer and LR; warmup; prompts × samples; mini-batch; updates/step; ε_low / ε_high; max generation; overlong shaping; data | GRPO variant; removed; AdamW constant 1e-6; linear over 20 rollout steps; 512 × 16; 512 samples; 16; 0.2 / 0.28; 20,480 (16,384 expected + 4,096 soft cache); linear penalty over the last 4,096 tokens; DAPO-Math-17K, 17K prompts | arXiv:2503.14476v2 §4.1 | verified 2026-09-15 | Table 1 (AIME24 avg@32, one run per row): naive GRPO 30, +overlong filtering 36, +Clip-Higher 38, +soft overlong punishment 41, +token-level loss 42, +dynamic sampling 50; DeepSeek-R1-Zero-Qwen-32B listed at 47 |
| verl DAPO recipe | not scoped | RL | `gen_batch_size`; `train_batch_size`; `max_response_length`; `overlong_buffer.len` / `penalty_factor`; `max_num_gen_batches` | 1536; 512; 20480; 4096 / 1.0; 10 | verl `docs/algo/dapo.md` (fetched 2026-09-14) | verified 2026-09-15; separate row from the paper | documentation example; no run link and no ablation |
| Skywork-OR1-7B | 7B | RL | LR; clip ε; target entropy; temperature; KL; stages (steps / context / batch / mini-batch / group); released checkpoint | 1e-6 constant; 0.2; 0.2 with adaptive coefficient; 1.0; none; stage 1 = 0–660 / 16K / 256 / 256 / 16, stage 2 = 660–1320 / 32K / 160 / 160 / 32; step 1320 | arXiv:2505.22312v2 §8.1, Table 11 | verified 2026-09-15 | §3.2.6: "The KL penalty hinders further improvements in test performance during multi-stage training"; §4: on-policy training mitigates entropy collapse |
| Skywork-OR1-32B | 32B | RL | stages; released checkpoint | stage 1 = 0–760 / 16K / 256 / 256 / 16, stage 2 = 760–1130 / 24K / 160 / 160 / 32; step 1000 | v2 Table 12 | verified 2026-09-15 | Table 13: AIME24 82.2, AIME25 73.3, LiveCodeBench 63.0 |
| Skywork-OR1 (all) | 7B, 32B | RL | prompt filter | model-specific difficulty 0–16; the released 32B script keeps 1 ≤ difficulty ≤ 15; source pool 105,055 math and 14,057 code problems | dataset card; `skywork-or1-filter-32b.py` | verified 2026-09-15 | dataset card: "For each model variant, we excluded problems with difficulty values of 0 and 16 specific to that model" |
| Magistral Medium | not reported | RL | algorithm; KL; ε_high; ε_low; batch schedule; non-penalized length schedule; LR, G, steps | GRPO with no KL, group-token loss normalization, zero-advantage groups removed; 0.26–0.28 adjusted during training; not reported; 8k → 4k → 2k sequences; 16k → 24k → 32k; not reported | arXiv:2506.10910v1 §2.1, §5.2 | verified 2026-09-14 ([[magistral-recipe]]); several values not reported | §6.4 Fig. 7: no significant difference between minibatch, group and no advantage normalization; §6.3 Fig. 6 (3B): reward degrades with more than 2 minibatches per batch |
| Magistral Small | 24B | RL | ε_high; batch; non-penalized length; temperature | 0.3; 2,048 sequences; 32k; 1.0 | v1 §5.3 | verified 2026-09-14 | ε_high raised because the cold-started model had "far lower entropy"; temperature 1.0 stated as the "best balance"; no numbers |
| Phi-4-reasoning-plus | 14B | RL | framework; global batch; G; optimizer and LR; warmup; KL target, placement and β; entropy coefficient; max length; clip; steps; seed data | verl; 64 problem seeds per iteration on 32 H100s; 8; Adam 5e-8; cosine over the first 10 steps; `D_KL(π_θ ‖ π_θold)` in the loss, 0.001; 0.001; 32k with outputs clipped at 31,744; ε not reported; 90; 72,401 math problems, ~6,400 seen | arXiv:2504.21318v1 §4, §4.2 | verified 2026-09-15 | §4.2: 90 steps selected by best AIME 2024 score; "additional GRPO training for only 90 steps boosts AIME performance by more than 10% (Figure 7a). Further training for more steps does not translate to additional gains" |
| Phi-4-reasoning-plus | 14B | RL | reward | length-aware cosine accuracy: +0.5 to +1.0 correct (penalty beyond L_pos = 25,600 of L_max = 31,744), −1.0 to −0.5 incorrect (penalty lighter beyond L_neg = 3,702); −0.5 for a missing `<im_end>`; −1.0 for a malformed `<think>` block; 5-gram repetition penalty; `R = (8/13)·R_acc + (1/13)·R_rep` | v1 §4.1 | verified 2026-09-15 | stated aim: concise when correct, longer when incorrect; no ablation of the weights |
| AceReason-Nemotron 7B / 14B | 7B, 14B | RL | algorithm; importance weight; KL and entropy coefficients; batch; G; LR; length stages; prompt filter | GRPO with `ρ = 1` fixed (one update per generation); β = 0, entropy 0; 128 prompts; 8 at 8K, 16 otherwise; 1e-6 AdamW (math), 5e-6 (code); 8K → 16K → 24K → 32K; drop prompts with pass rate > 6/16 at 24K and 32K | arXiv:2505.16400 §3.2.2, §3.3.2 | verified 2026-09-15 | §3.2.2 Fig. 3c: 2 or 4 updates per generation collapsed entropy around step 100; Fig. 3b: starting directly at 16K or 24K was worse; Table 3: the pass-rate filter "significantly improves model performance" |
| MiMo-7B-RL | 7B | RL | batch; mini-batch; updates/step; LR; max sequence length; temperature; top-p; KL; dynamic sampling; easy-pool probability; clip values; G | 512; 32; 16; 1e-6; 32,768; 1.0; 1.0; removed; on; α = 10%; not reported; not reported | arXiv:2505.07608v2 §3.3.2, §3.3.3 | verified 2026-09-15; ε and G not reported | §3.3.2: re-using easy data directly "introduces significant instability in policy updates"; the 10% pool is the stated fix; no numbers |
| SimpleRL-Zoo (10 open base models) | 0.5B–32B | RL | prompts/step; samples/prompt; mini-batch; max rollout length; temperature; clip ε; KL placement and coefficient; LR | 1,024; 8; 256; 8,192; 1.0; 0.2; in the loss, 1e-4 for 0.5B–14B and 1e-3 above 14B; not reported | arXiv:2503.18892v3 App. B.5 | verified 2026-09-15; LR not reported | §3.2 Fig. 7: the same configuration collapses Mistral-7B on MATH levels 3-5 and understretches Qwen2.5-7B on easy data |
| ScaleRL (8B dense; Llama-4 Scout 17B×16) | 8B; 17B×16 MoE | RL | recipe components; sequence budget; batch; compute | PipelineRL asynchrony, forced length interruption, CISPO loss, prompt-level loss averaging, batch-level advantage normalization, FP32 logits, zero-variance filtering, no-positive-resampling; 16,384 tokens (12,288 thinking + 2,048 solution); 768 default, 512 and 2,048 studied; 100,000 GPU-hours (8B) and 50,000 (Scout) | arXiv:2510.13786v1 §1, §2, §4, §5 | verified 2026-09-15 | §4 leave-one-out at 16k GPU-hours per variant; Table 1 fitted (C_mid, B, A); Fig. 8a: ±0.02 error margin on A from three repeated runs |

### 소규모 범용 run의 시작점

여기의 모든 수치는 위 `verified` 행의 수치이며, 해당 run이 만들어진 조건을 함께 유지한다.

on-policy 또는 mixed pair를 쓰는 7B–8B SFT checkpoint의 **선호 단계**에서는 length-normalized DPO, `β = 5`, 1 epoch, warmup ratio 0.1의 linear schedule, effective batch 128 pairs, 최대 sequence length 2,048, learning rate `5e-7`을 쓴다. 이는 초기 Tülu 3 SFT checkpoint 위에서 UltraFeedback으로 고른 Tülu 3 8B 구성이고, 이후 270k개가 넘는 pair mixture에서 실행됐다(Tables 18, 20). pair가 fully off-policy이고 loss가 plain DPO이면, Mistral-7B-v0.1과 UltraFeedback에서의 Zephyr 대안은 `β = 0.1`, peak LR `5e-7`, 10% warmup linear, global batch 32다. 두 행 사이에 β를 옮기지 말아야 한다. 학습 중 chosen log-probability가 떨어진다면, 검증된 anchor는 Llama 3.1의 chosen NLL 0.2(405B)와 Nemotron-4 RPO의 SFT-loss weight `1e-5`(340B)지만 둘 다 ablation과 함께 인쇄되지는 않았다.

binary verifier를 쓰는 7B 모델의 **RLVR 단계**에서는 Olmo 3 7B script 값, 즉 `β = 0.0`, 64 prompts × 8 samples, constant LR `1e-6`, temperature 1.0, clip 0.2 / 0.272, model type에 맞춘 response length(instruct 8,192, thinking 32,768)을 8–16 learner GPU와 32–56 vLLM engine에서 쓸 수 있다. KL 항이 필요하면 0.5B–14B 모델에서 loss 안의 SimpleRL-Zoo `1e-4`가 검증된 small-model 값이다. prompt set에는 all-correct/all-wrong group을 버리는 filter(DAPO, MiMo-7B, Skywork-OR1), 그리고 policy 자신으로 측정한 pass-rate threshold보다 높은 prompt를 제외하는 filter(AceReason-Nemotron 6/16, Skywork-OR1의 model별 difficulty score)가 검증돼 있다. SimpleRL-Zoo의 learning rate와 MiMo-7B의 group size는 공개되지 않았으므로 추정하지 말고 다른 verified 행으로 빈칸을 채워야 한다.

이 설정들과 함께 compute 조건도 밝혀야 한다. Tülu 3 8B DPO는 H100 8장으로 10시간, 70B DPO는 H100 64장으로 19시간이었다. Olmo 3 post-training은 H100 256장을 썼고, DAPO·Skywork-OR1·Phi-4-reasoning·MiMo-7B는 모두 verl에서 실행됐으며, ScaleRL ablation은 8B variant당 16,000 GPU-hours가 들었다.

## 일반화 관점

**(a) 범위를 넓히는 것.**

- 측정된 한 사례에서, 한 domain의 verifier reward가 다른 domain으로 transfer했다. math-only RL은 7B에서 LiveCodeBench v5를 37.6에서 44.4로, 14B에서 53.1에서 58.9로 올렸다. 이는 “math-related coding task만이 아닌 모든 problem topic”에서의 결과이며, 비슷한 strength의 math-only SFT 모델은 19.3이었다([[acereason-nemotron]] Table 1, Fig. 4) (**결과, 단일 연구**).
- reasoning stage 뒤에 general RL stage를 섞으면 non-reasoning 능력이 회복됐다. DeepSeek-R1 Dev2 → Dev3에서 AlpacaEval 2.0은 55.8 → 62.1, Aider-Polyglot은 25.6 → 44.8이었다([[deepseek-r1-recipe]] v2 Table 3).
- RL에 쓰지 않은 benchmark도 측정 가능하게 설계한다. DeepSeekMath는 RL data에 없는 benchmark에서 RL 효과를 읽기 위해, SFT question의 나머지를 RL prompt set에서 의도적으로 제외했다([[grpo-recipe]] §4.2). 이는 비용이 낮은 measurement design이다.
- 재학습 대신 merge로 stage-induced regression을 고칠 수 있다. SmolLM3의 APO soup와 mid-training checkpoint 0.9 / 0.1 linear merge는 base model의 RULER score를 128k까지 회복했다([[smollm3-apo]]).

**(b) 좁아짐 또는 망각을 일으키는 것.**

- SFT 모델이 이미 long context에 강하지 않다면 짧은 sequence만으로 training한 preference stage는 long-context 행동을 좁힌다. SmolLM3는 원인을 “APO training data가 24k token으로 제한됐다”고 직접 적었다. Llama 3.1은 short-context data만으로 DPO를 했는데 long-context 손상이 없었다고 보고하지만, 원인을 SFT 모델의 long-context 강점으로 설명했다([[llama-3-recipe]] §4.3.4). 조건은 stage 이름이 아니라 SFT checkpoint다.
- learned reward를 어느 지점 이후까지 최적화하는 경우: DeepSeek-R1 final helpfulness RM에서는 reward가 오르면서 Codeforces pass@1이 내린다([[deepseek-r1-recipe]] B.5 Fig. 6). Tülu 3 β sweep도 KL 관점에서 같은 모양을 보이며 “more KL divergence typically results in lower average scores”라고 보고한다.
- off-policy inner update([[acereason-nemotron]] Fig. 3c)와 낮은 sampling temperature([[skywork-or1]] §3.2.4)의 entropy collapse는 모두 sample 다양성을 낮춘다. 이것이 큰 k에서 pass@k가 내려가면서 pass@1은 오를 수 있는 메커니즘이다(ch-45 §"rlvr-beyond-base-model").
- base model과 맞지 않는 prompt set: Mistral-7B에서 MATH level 3–5는 “accuracy의 대응 개선 없이 response length가 크게 증가”했다([[simplerl-zoo]] §3.2).

**(c) 두 단계의 일반성을 측정하는 법.**

- 선호 단계에서는 pair가 직접 목표로 삼지 않은 benchmark를 적어도 하나 포함한 panel을 평가한다. Tülu 3 Table 18은 development evaluation 평균이므로 variant 사이 1.1점 차이를 결정적이라고 읽으면 안 된다. 최종 표는 development와 held-out(`unseen`) suite를 분리한다.
- RL 단계에서는 큰 k의 pass@k를 pass@1과 같이 보고하고 reward가 덮지 않은 domain을 평가한다. AceReason-Nemotron과 DeepSeek-R1은 둘 다 그렇게 한다. DAPO·MiMo-7B·Phi-4-reasoning-plus는 math와 code 밖 benchmark를 보고하지 않는다.
- 둘 다 compute budget을 명시한다. ScaleRL Fig. 10의 교차는 한 budget의 비교가 다른 budget에서 뒤집힐 수 있음을 뜻한다. budget 없이 “recipe X가 recipe Y보다 좋다”는 검증 가능한 주장이 아니다([[scalerl]] §5).
- decontamination은 별도 잡무가 아니라 측정의 일부다. Tülu 3, Qwen2.5(LCS filter, §5), DeepSeek-R1(10-gram filter, D.1), Skywork-OR1(AIME 24/25와 LiveCodeBench 유사 문제 제거)은 모두 이를 설명한다. Skywork dataset card는 early release에 한때 잘못된 difficulty field가 있었다고도 기록한다. release artefact에는 version이 있다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 방법 |
|---|---|---|
| `batch` 값을 잘못된 단위로 읽음 | source는 rollout당 16 update인데 구성한 run은 1 update를 하거나 그 반대이며, step당 wall-clock이 source와 멀다 | §1.1처럼 `prompts × samples ÷ mini-batch`를 계산해 source의 updates-per-step과 비교한다 |
| loss variant 사이에서 β를 그대로 옮김 | length normalization 아래 `β=0.1`이면 거의 움직이지 않거나, 없을 때 `β=5`이면 margin이 즉시 saturate한다 | normalization 전후 bracketed log-ratio difference 평균을 출력한다. §2.2의 같은 pair는 10.0 대 0.035다 |
| KL-in-loss 계수를 KL-in-reward 계수로 취급 | 오류 없이 KL curve가 source와 한 자릿수 이상 다르다 | 숫자가 아니라 equation 위치를 찾는다. KL-in-reward는 `r(x,y)` 안이고 KL-in-loss는 별도 additive term이다 |
| `π_θold`에 대한 KL을 retention control로 사용 | 기록한 KL은 작은데 SFT checkpoint 대비 KL은 제한 없이 자란다 | 두 KL을 따로 log한다. Phi-4-reasoning-plus의 `β=0.001`은 `π_θold` 기준이다([[phi-4-reasoning-rl]] §4.2) |
| launch script의 `total_episodes`를 run length로 복사 | 보고 step이 source evaluation cadence가 뜻하는 것보다 훨씬 크다 | paper table 및 checkpoint selection rule과 비교한다. Tülu 3 script budget은 10,000,000, table 값은 100,000이다 |
| 바뀐 framework default를 복사 | refactor 뒤 script에 더는 없는 flag가 새 default를 조용히 쓴다 | commit을 pin한다. Olmo 3 default는 commit 7917d41에서 `dpo`/0.1/2 epochs/0.03 warmup에서 `dpo_norm`/5.0/1/0.1로 바뀌었다([[allenai-olmo3-open-instruct-scripts-recipe]]) |
| prompt filter가 model size 간 transfer한다고 가정 | 거의 모든 group이 all-correct/all-wrong인 degenerate set이 되고 effective batch가 붕괴한다 | step당 zero-advantage group 비율을 log한다. Skywork-OR1 difficulty score가 model variant별인 이유다 |
| 한 compute budget에서만 두 recipe를 비교 | run을 연장하면 ranking이 뒤집힌다 | §5 curve를 전반부에 fit해 extrapolate한다. ScaleRL batch-512와 batch-2048은 16k~100k GPU-hours 사이에서 교차한다 |
| reconstruction을 원본처럼 인용 | DeepSeek/DAPO 문서 어디에도 없는 “recipe” 값이 나온다 | figure appendix가 re-implementation을 설명하는지 확인한다. [[scalerl]] App. A.16이 그렇다 |
| duplicate card를 근거로 인용 | citation locus가 source가 아닌 redirect page를 가리킨다 | `deepseekmath`는 [[grpo]]/[[grpo-recipe]]로, `nemotron`은 [[nemotron-4-synthetic]]으로 redirect된다. redirect를 따라간다 |

## 이해도 점검

1. DAPO는 “prompt batch size is 512”와 “mini-batch size is set to 512”를 함께 인쇄한다. gradient update에 관한 인쇄 문장을 사용하여 두 512가 모두 prompt일 수 없는 이유와, 같은 단위로 읽은 reproduction에서 생길 오류를 설명하라.
2. Tülu 3 DPO β는 5이고 Zephyr β는 0.1이다. 두 구성이 거의 같은 loss를 주는 응답 pair를 구성하고, loss variant 전체에 단 하나의 “올바른 β”가 없는 이유를 설명하라.
3. InstructGPT는 reward 안에 KL penalty를 넣고 DeepSeek-R1-Zero는 loss 안에 넣는다. group 안에서 advantage estimate를 normalize할 때 이 차이가 만드는 결과 하나를 도출하라.
4. AceReason-Nemotron은 generation당 정확히 한 gradient update, DAPO는 열여섯 update를 한다. 둘 다 좋은 결과를 보고했다. 후자가 안정적일 수 있는 mechanism과, 전자가 construction으로 얻는 일을 DAPO 설정에서 맡는 component를 말하라.
5. pass rate 7/8 prompt에서 오답 sample advantage는 −2.646, 정답 sample은 +0.378이다. AceReason-Nemotron filter가 pass rate *이상*의 prompt를 제거하는 이유와, 반대쪽을 제거하면 entropy에 무슨 일이 생길지 설명하라.
6. ScaleRL batch-512와 batch-2048 fit은 16k와 100k GPU-hours 사이에서 교차한다. 각 run의 fitted parameter 세 개만 알고 fixed 30,000-GPU-hour project에 무엇을 쓸지 정하는 법과, relative gain이 아니라 absolute gain을 비교하기 위해 추가로 필요한 quantity를 설명하라.
7. SmolLM3 long-context regression에는 두 원인이 있었고 preference stage 안의 원인은 하나뿐이다. 0.9 / 0.1 merge가 symptom을 고치는 이유와, preference-stage 원인을 아직 고쳐야 하는지 알려줄 측정값을 설명하라.
8. 이 장의 세 source는 run 값 하나를 그들의 released artefact와 모순되게 인쇄한다. 하나를 골라, released checkpoint를 설명하는 행에는 artefact 특성상 어느 값을 넣어야 하는지 논증하라.

## 연결

- **이전:** ch-45 — Self-Improvement Loops and Multi-Stage Reasoning Pipelines. 여기 표로 나눈 개별 stage의 pipeline shape(reasoning RL → rejection-sampling SFT → general RL)를 제공한다.
- **의존:** ch-39 — Offline Preference Optimization: DPO and Its Variants. §2가 named run으로 범위를 좁힌 DPO objective, variant, reference-model mechanism을 제공한다.
- **다음:** ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. §3 RL 열을 rollout 안에 environment observation이 포함되는 setting으로 확장하며, single-turn 행에는 없던 masking 열이 추가된다.

## 출처

라이브러리 카드:

- [[llama-3-recipe]] — Llama 3.1 405B DPO 행의 learning rate, β, NLL 계수, token masking, round 구조.
- [[qwen-2.5-recipe]] — Qwen2.5 DPO 행(pair, Online Merging Optimizer)과 episode당 query 값을 계산한 GRPO 행.
- [[tulu-3]] — report의 pipeline 틀. 수치는 [[tulu-3-hyperparameter-tables]]에서 인용한 primary table이 이 card의 Technical Details block보다 우선한다.
- [[rlvr-tulu3]] — 방법으로서의 RLVR. hyper-parameter 단락이 Table 21과 맞지 않으므로 여기서는 verifier 정의에만 인용한다.
- [[tulu-3-1]] — Tülu 3와 3.1 refresh 사이에서 바뀐 것과 바뀌지 않은 것.
- [[olmo-3]] — Olmo 3 행의 model-flow 틀.
- [[allenai-olmo3-open-instruct-scripts-recipe]] — 모든 Olmo 3 DPO·RL 값, framework default, script와 table 간 충돌.
- [[smollm-3]] — SmolLM3 release와 stage 구조.
- [[nemotron-4-synthetic]] 및 [[nemotron-4-synthetic-recipe]] — DPO+SFT 항과 세 번의 RPO iteration 및 stage별 Table 6 수치. (`nemotron` card는 duplicate redirect이므로 근거로 인용하지 않는다.)
- [[rlhf-instructgpt]] — 3단계 RLHF template. canonical hyper-parameter 표의 수치는 [[instructgpt-ppo-appendix]]가 우선한다.
- [[llama-2-recipe]] — size별 KL 계수를 포함한 Llama 2-Chat PPO 행.
- [[hf-dpo-zoo]] — §2.2에서 참조한 preference-loss variant 지도.
- [[dpo]] 및 [[simpo]] — DPO-norm 행의 objective와 length normalization.
- [[grpo]] 및 [[grpo-recipe]] — unit이 명시되지 않은 batch와 paper가 인쇄하지 않은 값을 포함한 DeepSeekMath GRPO run. (`deepseekmath` card는 duplicate redirect이므로 근거로 인용하지 않는다.)
- [[deepseek-r1-recipe]] — R1-Zero, R1 stage-1, R1 final-stage RL 행.
- [[magistral-recipe]] — Magistral Medium·Small RL 행과 paper가 남긴 설정 공백.
- [[phi-4]] — Phi-4 family 틀. reward 설명은 [[phi-4-reasoning-rl]]로 바로잡는다.

장 발췌문(이 장에서 읽은 primary text):

- [[instructgpt-ppo-appendix]] — App. C.3, C.4, E.7, E.9, E.11의 원문과 paper에 없는 값 목록.
- [[tulu-3-hyperparameter-tables]] — Tables 18–22, 34, 35, 그리고 8B·70B·405B·Tülu 3.1의 open-instruct launch command.
- [[zephyr]] — pair 구성, DPO 설정, overfitting 관찰.
- [[dapo]] — §4.1 training detail과 Table 1.
- [[verl-dapo-recipe]] — `gen_batch_size`를 포함한 같은 recipe shape의 released configuration.
- [[skywork-or1]] — shared setting, stage별 표, zero-advantage filter, difficulty filter.
- [[acereason-nemotron]] — 엄격한 on-policy update, length staging, pass-rate filter, cross-domain transfer 표.
- [[mimo-7b]] — 하나의 단위로 표현한 batch와 mini-batch, dynamic sampling, easy-pool resampling 확률.
- [[phi-4-reasoning-rl]] — 90-step GRPO stage, KL target, 실제 reward shape.
- [[smollm3-apo]] — APO pair 구성, 24k context cap, RULER를 복구한 merge.
- [[scalerl]] — sigmoid fit, A 대 B ablation, recipe 비교 범위를 규정하는 appendix.
- [[simplerl-zoo]] — 열 개 base model에 하나의 configuration을 적용한 결과와 difficulty matching 발견.
