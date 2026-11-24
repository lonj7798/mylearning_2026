<!-- chapter: ch-58a
     track: capstone
     kind: content
     title: 공개 범용 모델 레시피의 처음부터 끝까지: 사전학습부터 병합까지
     deps: [ch-58, ch-14a, ch-32e, ch-34, ch-45a, ch-45d]
     sources: [[olmo-2-stage-chain]], [[olmo-3-stage-chain]], [[olmo-core-olmo3-configs]], [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]], [[tulu-3-stage-chain]], [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]], [[smollm3-stage-chain]], [[smol-training-playbook]], [[llama-3]], [[llama-3-recipe]], [[qwen-2.5]], [[qwen-2.5-recipe]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[epoch-deepseek-r1-compute]], [[k2-v2]], [[apertus]]
     figures: figures/stage-chain-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 58a — 공개 범용 모델 레시피의 처음부터 끝까지: 사전학습부터 병합까지

> **핵심 통찰.** 여기서 비교하는 8개 공개 레시피에서 각 단계 내부의 설정보다 단계 사슬이 더 일관되다. 자체 사전학습을 수행한 7개는 모두 더 높은 품질의 혼합물로 이루어진 별도의 더 짧은 단계를 두어 사전학습을 끝낸다. 8개 모두 비교 또는 검증기를 사용하는 단계보다 먼저 모방 단계를 수행한다. 8개 중 7개는 비교 기반 또는 검증기 기반 단계를 적어도 하나 수행한다(K2-V2만 유일하게 예외이며, 공개된 사슬이 SFT에서 끝난다). 8개 중 4개는 사슬 어딘가에서 체크포인트를 평균내거나 병합하고, 그 4개 중 2개는 병합으로 끝난다. 이 단계 안의 설정값은 그대로 옮겨지지 않는다. 같은 단계와 같은 데이터 혼합물에서도 Tülu 3는 8B에서 SFT 학습률 5e-6, 70B와 405B에서 2e-6을 사용하고, DPO에서는 8B에 5e-7, 70B에 2e-7을, RLVR에서는 8B에 3e-7, 70B에 1e-7을 사용한다([[tulu-3-stage-chain]] Tables 11, 20, 21, 34). 반면 OLMo 2는 자사 베이스가 “Llama 3.1 훈련 레시피와 비교하여 상당히 더 높은 학습률을 필요로 했다”고 보고한다([[olmo-2-stage-chain]] §5). 공개된 내용도 연구소별 차이보다 단계별 차이를 보인다. DeepSeek-V3는 사전학습 옵티마이저 설정은 전부 인쇄하지만 RL 하이퍼파라미터는 전혀 인쇄하지 않는다([[deepseek-v3-recipe]]). Olmo 3는 공개 체크포인트를 만든 실행 스크립트와 값이 적어도 5개 행에서 서로 다른 논문 표를 인쇄한다([[allenai-olmo3-open-instruct-scripts-recipe]]).
>
> **지침.** 공개 레시피에서 실행 계획을 세울 때는 수치 설정은 마지막에 복사하고, 단계 사슬과 단계 전이 게이트를 먼저 복사하라. 단계 사슬은 연구소를 가로질러 재현되지만 설정값은 그렇지 않기 때문이다. 예를 들어 7–8B에서 같은 단계 사슬이라도 Tülu 3는 Llama 3.1 베이스에서 SFT 5e-6과 DPO 5e-7을 선택한 반면, OLMo 2는 자체 베이스에서 2e-5와 1e-6을 선택했다([[tulu-3-stage-chain]] Tables 11, 20; [[olmo-2-stage-chain]] §5). 레시피가 보고서와 공개 코드 양쪽으로 제공되면 논문값, 공개된 config값, 프레임워크 기본값을 세 개의 별개 사실로 기록하고 config값으로 학습하라. open-instruct 커밋 7917d41이 β, 손실 유형, epoch, warmup, clipping을 Olmo 3 스크립트에서 빼서 기본값으로 옮겼으므로, 그 커밋 뒤에 읽은 스크립트는 같은 스크립트를 그 커밋 전에 읽었을 때와 다른 값으로 실행되기 때문이다([[allenai-olmo3-open-instruct-scripts]] PR #1547). 모델 크기 사이에서 설정을 옮길 때도 숫자 자체가 아니라 선택 절차를 옮겨라. 즉 sweep 범위, K2-V2의 1/√TPP로 스케일한 τ_epoch 같은 목표([[k2-v2]] §3.2), 또는 짧은 컨텍스트 평가가 회복된 뒤에야 긴 컨텍스트 단계를 진행한다는 Llama 3 규칙 같은 게이트([[llama-3-recipe]] §3.4.2)를 옮긴다. 여기서 값이 어떻게 선택되었는지 밝힌 모든 레시피는 절차를 밝히지만, Qwen2.5를 포함한 일부 레시피는 사후학습 설정 어느 것도 공개 크기에서 ablation을 제시하지 않는다([[qwen-2.5-recipe]]; 모든 사후학습 행이 “no ablation reported”라고 적혀 있다).

## 이 장이 범용 모델에 중요한 이유

파이프라인은 사전학습 → 중간학습 → SFT → 선호 최적화 → RL → 평가이다. 앞선 장들은 여러 모델에서 한 단계를 비교했다. ch-14a는 사전학습 예산과 스케줄을, ch-32e는 중간학습과 컨텍스트 확장을, ch-45a는 선호 및 RL 단계를, ch-45d는 에이전트형 레시피를 비교했고, ch-33과 ch-34는 개별 보고서를 사례 연구로 읽었다. 이 장은 한 번에 한 모델의 모든 단계를 비교한다. 대부분의 정보가 공개된 8개 모델 각각에 대해, 첫 사전학습 토큰부터 공개 체크포인트까지의 완전한 사슬이 무엇인지, 각 전이를 어떤 기준이 통과시켰는지, 무엇이 여전히 알려지지 않았는지를 묻는다.

측정 가능한 문제는 네 부분으로 나뉜다.

1. **계획에는 설정 목록이 아니라 사슬이 필요하다.** 고정된 컴퓨팅 예산을 가진 팀은 학습률 하나를 정하기 전에 몇 단계를 어떤 순서로 실행할지 결정해야 한다. 단계 사슬은 최종 모델이 어떤 능력을 갖게 될지에 가장 큰 영향을 주는 결정이며, 공개된 레시피가 가장 많이 합의하는 부분이다.
2. **설정값은 크기, 단계, 실행에 종속된다.** 여기의 레시피 중 4개는 자체 산출물 두 곳에서 같은 설정에 서로 다른 값을 인쇄한다(§3). 적용 범위를 적지 않은 값을 복사하는 일은 재현 모델이 공개 체크포인트에서 벗어나는 가장 흔한 방법이다.
3. **한 단계의 근거는 보통 ablation이 아니라 게이트다.** 대부분의 단계 전이는 파이프라인 A/B 비교가 아니라 평가 임계값, sweep, 또는 정성 점검으로 결정되었다. 어느 것인지 알면 그 순서를 얼마나 신뢰해야 하는지가 달라진다(§5).
4. **범용성은 어떤 단계에서는 측정되고 다른 단계에서는 측정되지 않는다.** Olmo 3의 RL-Zero 실행 스크립트는 학습 중인 도메인만 평가하지만, Instruct 스크립트는 15개 벤치마크 목록을 실행한다([[allenai-olmo3-open-instruct-scripts]]). Tülu 3는 매 단계에서 별도의 미관측 스위트를 측정했고, 보고된 경향과 반대로 움직인 기술 하나를 발견했다([[tulu-3-stage-chain]] Table 31). 설정만 기록하는 원장은 한 단계가 모델의 능력 범위를 넓혔는지 좁혔는지 답할 수 없다.

§2와 §3의 비교를 대화형으로 볼 수 있는 버전은 [stage-chain-explorer.html](figures/stage-chain-explorer.html)에 있다. 각 레시피를 단계 사슬로 표시하고, 각 단계의 토큰 예산, 배치, 공개 상태를 보여 준다. 따라서 8개 보고서를 차례로 읽지 않고도 정규화된 토큰 스케일에서 사슬을 비교할 수 있다.

## §1 처음부터 끝까지의 원장: 8개 레시피를 하나의 행 형식으로 정규화하기

**정의.** 처음부터 끝까지의 원장은 레시피 원장(ch-14a §1, style §5.2)을 확장한 것이다. 공개 체크포인트 하나의 모든 단계를 학습 순서로 놓고, 모델, 크기, 단계, 설정, 값, 출처 위치, 상태, 그리고 그 값을 선택한 근거라는 같은 8개 열에 기록한다. 단계 이름은 고정된 어휘에서 고른다. `pretrain-stable`, `pretrain-decay/anneal`, `mid-train`, `long-context`, `SFT`, `distill-SFT`, `reward-model`, `preference`, `RL`, `merge`, `eval-gate`이다.

**문제.** 보고서는 같은 역할의 단계를 서로 다르게 부른다. OLMo 2가 “stage 2”라고 부르는 것은 고품질 혼합물에서 학습률을 anneal하는 단계다([[olmo-2-stage-chain]] §2.3). Olmo 3가 중간학습이라고 부르는 것은 완료된 사전학습 스케줄 뒤에 일정한 배치로 수행하는 별도 100B-토큰 실행이다([[olmo-core-olmo3-configs]]). K2-V2가 mid-1이라고 부르는 것은 1.769T-토큰 단계로, 많은 7B 모델의 사전학습 전체보다 크다([[k2-v2]] Table 3). 고정된 어휘가 없으면 레시피 간 비교는 단계를 비교하는 대신 단계 이름을 비교하게 된다.

**절차.**
1. 모델 계열이 아니라 공개 체크포인트 이름을 고정한다(OLMo-2-1124-7B, Olmo-3-1025-7B, Llama-3.1-Tulu-3-8B, SmolLM3-3B).
2. 학습 순서대로 보고서를 읽고, 기술된 각 실행을 위 어휘의 단계 이름 하나에 할당한다.
3. 모든 수량의 단위를 정규화한다. 토큰은 한 경로를 따라 본 토큰 수로, 배치는 토큰 수로, prompt 수와 prompt당 sample 수는 별도 수로 기록한다. pair는 prompt와 구분하고, epoch은 step과 구분한다.
4. 같은 단계에 대해 고정된 커밋에서 공개 config나 실행 스크립트를 읽고 별도 행으로 기록한다.
5. 상태를 부여한다. `verified`, `not reported`(확인한 항목 목록을 함께 기록), `conflict`(산출물당 한 행), 또는 `derived`(공식 포함) 중 하나다.
6. 게이트를 기록한다. 단계를 끝낸 평가나 sweep과 그 위치를 적고, 없으면 `no gate reported`라고 적는다.

**작동 예: 6개 레시피의 SFT 단계를 하나의 단위로 정규화하기.** 보고서는 SFT 크기를 prompt, example, 또는 token으로 말하며, 각각 epoch 수도 말한다. SFT에서 본 토큰 수 = (example당 토큰 수) × (example 수) × (epoch 수)이다.

| 체크포인트 | 인쇄된 표현 | Epoch | SFT에서 본 토큰 수 | 상태 |
|---|---|---|---|---|
| Olmo 3 7B Think SFT | 2,268,468 prompts; “Total Tokens” 45.4B | 2 | 45.4B | 검증됨([[olmo-3-stage-chain]] Table 47); 도출된 평균 45.4e9 / (2 × 2,268,468) = prompt당 10,007 tokens |
| DeepSeek-R1 (Dev3) | 804,745 samples, 평균 5,355.3 tokens | 2–3 | epoch당 4.31B → 8.6–12.9B | [[deepseek-r1-recipe]] Table 5에서 도출; epoch 수는 충돌한다(v1은 “two epochs”, v2는 “2–3 epochs”) |
| K2-V2 SFT | epoch당 17.4B tokens | 3 | 52.2B | 검증됨([[k2-v2]] §6.1) |
| SmolLM3 SFT | 1.8B tokens | 4 | 7.2B | [[smollm3-stage-chain]]에서 도출 |
| OLMo 2 7B SFT | 939,104 prompts, 최대 길이 4,096 | 2 | 보고되지 않음 | 보고되지 않음([[olmo-2-stage-chain]] §5; 보고서와 실행 파일에 token 수가 없음) |
| Qwen2.5 Instruct | “over 1 million SFT examples”, 길이 32,768 | 2 | 보고되지 않음 | 보고되지 않음([[qwen-2.5-recipe]] §4.1) |

정규화하면 순서가 바뀐다. prompt 수로 보면 Olmo 3 7B Think SFT(2.27M)는 DeepSeek-R1의 거절 샘플링 SFT(805K)의 2.8배다. 본 토큰 수로 보면 평균 example이 약 1.9배 길기 때문에 3.5배에서 5.3배가 된다. SFT 예산을 prompt 수로만 세우는 계획은 긴 chain-of-thought 단계를 대략 그 비율만큼 과소예산 편성한다.

**조건과 한계.** Olmo 3의 prompt당 토큰 도출값은 보고서가 말하지 않은 두 가정을 둔다. 인쇄된 45.4B가 한 epoch가 아니라 두 epoch 전체 실행이라는 가정, 그리고 masked prompt token을 포함한 packed training token을 센다는 가정이다. 첫 가정을 반대로 읽으면(45.4B가 epoch당) 본 토큰 수는 90.8B이고 평균 example은 20,014 tokens가 된다. 보고서는 배치가 document packing과 함께 “instances instead of tokens”가 아니라 “measured in tokens instead of instances”라고만 말하며, loss를 받는 토큰과 packed token을 나누어 적지 않는다([[olmo-3-stage-chain]] A.6.1). 보고서가 토큰 수도 평균 길이도 주지 않는 경우에는 정규화할 수 없으므로 해당 셀은 `not reported`로 남긴다.

**범용 모델에 대한 함의.** 사전학습 대비 모방 단계의 크기는 8개 레시피 전체에서 비교할 수 있는 수량 중 하나이며, 두 수를 모두 인쇄한 두 레시피에서는 6.6배 차이가 난다. SmolLM3는 11.2T 사전학습 토큰 뒤에 7.2B SFT 토큰(0.064%)을 사용한 반면, K2-V2는 12.25T 뒤에 52.2B를 사용했다(0.43%). ch-30a와 ch-38a는 이 비율이 이전 능력의 보존에 무엇을 하는지 다루며, 이 장은 정규화된 수치를 제공한다.

## §2 8개 사슬을 나란히 보기

**정의.** 사슬은 하나의 공개 체크포인트를 만든 단계를 학습 순서대로 나열한 것으로, 각 단계의 토큰 예산 또는 example 수를 함께 적는다.

| 레시피(공개 체크포인트) | 학습 순서에 따른 사슬 | 체크포인트 평균 또는 병합 |
|---|---|---|
| OLMo 2 7B Instruct | pretrain 3.90T → anneal 3 × 50B, 평균 냄 → SFT(939,104 prompts, 2 ep.) → DPO(366.7k prompts) → RLVR(PPO) | anneal 단계 내부(세 실행을 평균) |
| Olmo 3 7B Instruct | pretrain 5.93T → mid-train 100B → long-context 65,536에서 50B → Think SFT 45.4B tokens → Think DPO 150K pairs → Instruct SFT 3.4B tokens → Instruct DPO 260K pairs → Instruct RL | Think SFT에서(두 학습률을 병합), 32B에서는 mid-training에서(두 실행을 soup으로 만듦) |
| Tülu 3 8B | (여기서 학습하지 않은 Llama 3.1 8B base) → SFT(2 ep., batch 128) → length-normalized DPO(β 5, 1 ep.) → RLVR(PPO, β 0.05) | 없음 |
| SmolLM3-3B | pretrain 3단계에서 11.2T → long-context 2 × 50B → reasoning mid-train 35B × 4 ep. → SFT 1.8B × 4 ep. → APO → APO 체크포인트 soup → mid-training 체크포인트와 0.9 / 0.1 선형 병합 | 공개 직전의 마지막 연산 |
| Llama 3.1 405B | pretrain 15.6T → long-context 6단계, ≈800B tokens → Polyak averaging을 사용한 최종 40M-token anneal → 6회 반복(reward model → rejection sampling → SFT → DPO) → RM, SFT, DPO 각 단계에서 모델 평균 | 세 단계에서 수행; 공개 직전의 마지막 연산 |
| Qwen2.5(open weights) | pretrain 18T(4,096 context) → 32,768에서의 최종 pre-training 단계 → SFT(>1M examples, 2 ep.) → DPO(≈150,000 pairs, 1 ep.) → GRPO(query당 8 responses) | 체크포인트 병합 없음; DPO 내부에 Online Merging Optimizer 사용 |
| DeepSeek-V3 → R1 | V3: pretrain 14.8T → long-context 2 × 1,000 steps → SFT 1.5M instances(2 ep.) → GRPO. R1: cold-start SFT(“thousands”) → reasoning RL → V3-Base에서 재시작한 rejection-sampling SFT 804,745 samples → mixed RL 1,700 steps | 없음 |
| K2-V2 70B | pretrain 12.25T → mid-1 1.769T → context를 524,288까지 늘린 mid-2/3/4 590B + 229B + 131B → SFT 17.4B × 3 ep. | 없음 |

같은 8개 사슬은 [stage-chain-explorer.html](figures/stage-chain-explorer.html)에서 토큰 스케일로 그려져 있다. 그곳에서 단계를 선택하면 그 설정, 출처 위치, 상태를 인쇄하며, “post-training stages only” 필터는 이 표가 한 셀에 압축한 열을 분리한다.

행별 출처: [[olmo-2-stage-chain]]; [[olmo-3-stage-chain]] 및 [[olmo-core-olmo3-configs]]; [[tulu-3-stage-chain]]; [[smollm3-stage-chain]]; [[llama-3-recipe]]; [[qwen-2.5-recipe]]; [[deepseek-v3-recipe]] 및 [[deepseek-r1-recipe]]; [[k2-v2]].

**공유되는 부분.** 자체 사전학습을 수행하는 여기의 모든 레시피는 안정 단계에서 멈춘 뒤 곧바로 SFT로 넘어가지 않고, 별도의 최종 단계로 사전학습을 끝낸다. 그 단계는 anneal(OLMo 2, Llama 3.1), mid-training(Olmo 3, K2-V2, SmolLM3의 reasoning 단계), decay 단계(SmolLM3 stage 3, DeepSeek-V3), 또는 더 긴 context에서의 최종 pre-training 단계(Qwen2.5의 32,768 이동)라고 불린다. 완성된 base에서 시작하는 Tülu 3가 예외다. 8개 모두 comparison-based 단계 전에 모방 단계를 실행한다. RL만으로 AIME 2024 pass@1을 15.6%에서 77.9%로 높인 R1-Zero를 보인 DeepSeek-R1조차도, R1-Zero가 IF-Eval에서 46.6, AlpacaEval 2.0에서 24.7을 기록했으므로 공개 모델에는 cold-start SFT를 추가했다([[deepseek-r1]] §1, Table 3).

**공유되지 않는 부분.** 선호 유형 단계 수는 0개(K2-V2)에서 12개(Llama 3.1: reward model과 DPO 단계를 각각 포함한 6회 반복)까지다. long-context 단계가 사슬에서 놓이는 위치는 네 곳만큼 달라진다(§7). 네 레시피가 체크포인트를 평균 내거나 병합한다. OLMo 2는 anneal 내부에서, Olmo 3는 7B Think SFT 단계와 32B mid-training 단계에서, SmolLM3는 끝에서, Llama 3.1은 reward-model·SFT·DPO 단계에서 수행한다. 이 중 SmolLM3와 Llama 3.1에서는 병합이 공개 전 마지막 연산이다.

**공유 순서의 근거.** 모방이 comparison보다 먼저 와야 한다는 가장 강한 공개 근거는 한 파이프라인의 모든 중간 체크포인트를 보고하는 DeepSeek-R1 Table 3이다. AlpacaEval 2.0은 R1-Zero(RL만) 24.7 → Dev1(cold-start SFT) 50.1 → Dev2(reasoning RL) 55.8 → Dev3(non-reasoning data를 사용한 rejection-sampling SFT) 62.1 → R1(mixed RL) 87.6으로 변했다. 한편 AIME 2024는 R1-Zero에서 77.9였다가 Dev1에서 59.0으로 떨어지고 R1에서 79.8로 회복했다([[deepseek-r1]] §4, Table 3). 상태: **결과(단일 연구)**. 총 671B parameters인 하나의 파이프라인이며, 단계를 재배열한 반복 실험은 없다.

## §3 공개성 행렬: 논문, 설정 파일, 기본값은 세 개의 사실이다

**정의.** 공개성 행렬은 레시피와 단계별로 그 단계를 공개 산출물만으로 재현할 수 있는지를 기록한다. 데이터가 공개되었는지, 고정된 커밋에서 config 또는 실행 스크립트가 공개되었는지, 시작 체크포인트가 공개되었는지, 하이퍼파라미터가 명시되었는지를 적는다.

**문제.** “공개”는 하나의 속성이 아니다. 레시피는 가중치만 공개하고 데이터는 공개하지 않을 수 있고, 데이터는 공개하되 config는 공개하지 않을 수 있으며, config의 값이 논문과 다를 수도 있다.

| 레시피 | 사전학습 데이터 | 사전학습 config | 사후학습 데이터 | 사후학습 실행 코드 | 기록된 논문-대-config 충돌 |
|---|---|---|---|---|---|
| OLMo 2 | 공개(OLMo-Mix-1124) | 공개 YAML | 공개(Tülu 3 혼합물) | 공개(`docs/olmo2.md`) | 13B peak LR 9.0e-4(Table 3) 대 3.0e-4(`OLMo2-13B-stage1.yaml` L46; ch-14a에서 읽음); 7B SFT LR 1e-5(YAML) 대 2e-5(script); 7B DPO 5e-7(YAML) 대 1e-6(script) (두 사후학습 행은 [[open-instruct-allenai-recipes-recipe]]) |
| Olmo 3 | 공개(Dolma 3) | 공개(OLMo-core scripts) | 공개(Dolci) | 공개(17개 scripts, 실행별 Beaker links) | 7B peak LR 2.0712e-4(script) 대 2.074e-4(Table 35); 32B Think RL step당 64 prompts(script) 대 128(Table 49); 7B Instruct DPO 32 GPUs(도출) 대 16(Table 48) ([[olmo-core-olmo3-configs]], [[allenai-olmo3-open-instruct-scripts-recipe]]) |
| Tülu 3 | 해당 없음(Llama 3.1 base) | 해당 없음 | 공개 | 공개(`docs/tulu3.md`) | 70B RLVR β 0.7(§6.4 본문) 대 0.07(Table 21 caption 및 70B launch command); episodes 100,000(Table 21, 모든 크기) 대 400,000(§6.4 및 70B command) 대 10,000,000(8B command에 계획됨) ([[tulu-3-stage-chain]]; [[open-instruct-allenai-recipes-recipe]]) |
| SmolLM3 | dataset 이름과 단계별 비율은 공개, 조립된 corpus는 비공개 | 공개 configs | 부분 공개 | 비공개 | decay 시작 10T(blog) 대 9.90T(configs에서 도출); 총량 11T 대 11.2T 대 11.1T ([[smollm3-stage-chain]]) |
| Llama 3.1 | 비공개 | 비공개 | 비공개 | 비공개 | 기록 가능 없음. 보고서만 존재한다 |
| Qwen2.5 | 비공개 | 비공개 | 비공개 | 비공개 | SFT example 수 “over 1 million”(§4.1, Abstract) 대 “1 million examples”(§1) 대 “millions”(§4 도입) ([[qwen-2.5-recipe]]) |
| DeepSeek-V3 / R1 | 비공개 | 비공개 | 비공개 | 비공개 | R1 2단계 SFT “two epochs”(v1) 대 “2–3 epochs”(v2) ([[deepseek-r1-recipe]]) |
| K2-V2 | 공개(TxT360) | 공개(`llm360/k2v2_train`) | 공개(TxT360-3efforts) | 공개 | 여기에서 기록된 충돌 없음. 코드와 log는 이 장에서 읽지 않았다 |

**메커니즘: script 값과 논문 값이 서로 다른 이유.** open-instruct에서 PR #1547(commit 7917d41, 2026-03-20)은 프레임워크 기본값을 DPO β 5.0 및 `dpo_norm`, 1 epoch, warmup 0.1, GRPO `clip_higher` 0.272, ρ cap 2.0, `async_steps` 8, in-flight updates true, advantage normalization “centered”로 바꾼 다음, Olmo 3 실행 scripts에서 그 flags를 삭제했다([[allenai-olmo3-open-instruct-scripts]]). 그러므로 head commit에서 script를 읽으면 실행 commit에서 읽을 때보다 설정이 적게 보이며, 삭제된 flags는 그 commit의 부모인 00d42c4에서만 복구할 수 있다. 여러 Olmo 3 RL 실행에서 #1547 전 기본값은 #1547 후 기본값과 다르다. 따라서 flag를 출력하지 않은 script는 오늘 같은 script를 실행할 때와 다른 값으로 실행됐다. 7B Think no-pipeline 실행은 00d42c4 기본값에서 truncated-importance-sampling correction을 끄고 standard advantage normalization을 사용했고, 7B Instruct RL 실행은 오늘 기본값과 truncated-importance-sampling correction에서 다르며, RL-Zero·instruction-following·general 실행은 advantage normalization에서 다르다. 현재 기본값은 ρ cap 2.0과 centered advantages다([[allenai-olmo3-open-instruct-scripts-recipe]]).

**작동 예: 한 설정에 관한 세 가지 사실.** Olmo 3 7B Think DPO 단계에서 세 산출물은 다음을 말한다. 논문 Table 48은 150K pairs, LR 8.0e-8, max length 16,384, batch 128, 32 GPUs라고 적는다. 실행 script `7b_think_dpo.sh` L3–40은 같은 150,000 `max_train_samples`, 8e-8 linear, 16,384과 1 × 4 × (4 × 8) = 128로 도출되는 batch를 적는다. d8a7f1c의 framework default는 `dpo_norm`, β 5.0, 1 epoch, warmup 0.1이며 script는 이 어느 것도 출력하지 않는다. 세 사실을 합쳐야 실행을 기술할 수 있고, framework가 업그레이드되면 바뀌는 것은 세 번째 사실뿐이다. 상태: [[allenai-olmo3-open-instruct-scripts-recipe]]에서 행 검증됨.

**조건과 한계.** 충돌 수는 불일치의 하한이지 품질 점수가 아니다. 공개 코드가 없는 레시피는 비교할 대상이 없으므로 기록된 충돌이 0개다. Llama 3.1, Qwen2.5, DeepSeek-V3가 표에서 충돌 없는 것처럼 보이는 이유가 이것이다.

**범용 모델에 대한 함의.** 가장 자주 명시되지 않는 단계는 능력의 폭에 가장 크게 영향을 미치는 단계다. Qwen2.5는 RL learning rate, KL coefficient, clip ε, temperature, response length, episode count를 하나도 인쇄하지 않으며([[qwen-2.5-recipe]] §4.3), DeepSeek-V3는 RL hyper-parameter를 전혀 인쇄하지 않는다([[deepseek-v3-recipe]] §5.2). 두 레시피 모두 사전학습 optimizer 설정은 인쇄한다. 즉 독자는 원시 능력을 만드는 단계는 재현할 수 있지만, 그 능력이 얼마나 넓게 표현되는지를 형성하는 단계는 재현할 수 없다.

## §4 스케일 이전: 405B에서 1B로 바뀌는 것은 무엇인가

**정의.** 스케일 이전(scale transfer)은 같은 레시피를 다른 모델 크기에서 실행할 때 어떤 설정은 그 값을 유지하며, 어떤 설정은 다시 선택해야 하는지를 묻는 문제다.

**문제.** 소규모 팀이 70B 또는 405B에서 공개된 레시피를 복사한다. 어떤 설정이 모델 크기에 따라 변하는데도 팀이 그 값을 고정하면, 실행은 오류를 내지 않으면서도 잘못 지정된다.

**단계와 혼합물을 고정했을 때 레시피가 보이는 것.**

| 설정 | Tülu 3 8B | Tülu 3 70B | Tülu 3 405B | 방향 |
|---|---|---|---|---|
| SFT peak LR | 5e-6 | 2e-6 | 2e-6 | 크기가 커질수록 하락 |
| SFT effective batch(시퀀스) | 128 | 128 | 256 | 유지된 뒤 2배 |
| DPO peak LR | 5e-7 | 2e-7 | 2e-7 | 크기가 커질수록 하락 |
| DPO β(length-normalized loss) | 5 | 5 | 5 | 불변 |
| RLVR peak LR | 3e-7 | 1e-7 | 1e-7 | 크기가 커질수록 하락 |
| RLVR effective batch(응답) | 224 | 640 | 1,856 | 크기가 커질수록 상승 |

모든 행은 [[tulu-3-stage-chain]] Tables 11, 20, 21, 34, 35에서 왔으며, [[open-instruct-allenai-recipes-recipe]]의 launch command와 대조 확인했다.

같은 양상은 7B와 32B가 같은 레시피인 Olmo 3 내부에서도 보인다. 사전학습 peak LR은 7B 3.0e-4, 32B 6.0e-4이고 batch는 4,194,304 tokens에서 8,388,608 tokens로 2배가 된다. Think SFT LR은 5e-5 대 1e-4 souped with 5e-5이고, Think DPO는 8e-8 대 7e-8, Think RL은 1e-6 대 2e-6이다([[olmo-3-stage-chain]] Table 35, Table 47; [[allenai-olmo3-open-instruct-scripts-recipe]]). 세 사후학습 learning rate 중 둘은 크기와 함께 올라가고 하나는 내려간다. 따라서 “learning rate는 크기가 커질수록 내려간다”는 규칙은 레시피 사이에서 살아남지 않는다. 살아남는 것은 각 크기에서 그 값을 다시 선택했다는 사실이다.

**설정을 이전하는 공개 절차 하나.** K2-V2는 값이 아니라 scaling target을 제시한다. learning rate η, weight decay λ, dataset size D, batch B로 구성한 “effective averaging timescale τ_epoch”은 “AdamW가 parameter update를 지수적으로 평균 내는 유효 epoch 수를 근사한다.” 이 target은 parameter당 20 tokens에서 조정한 뒤, 전체 실행에서는 1/√TPP에 비례하게 scale한다. 그 결과 η = 1.5e-4, λ = 0.05, B = 9.8 × 10⁶ tokens, D = 12.25 × 10¹², τ_epoch = 0.1066이 된다([[k2-v2]] §3.2).

**작업 예: 재현해 target을 식별하기.** 보고서를 layout 보존 방식으로 text 추출하면 분수가 모호하게 렌더링된다. 한 줄에는 `τepoch = ηλD`, 다음 줄에는 `B`가 나온다. 두 방향은 88배 차이난다. B / (ηλD) = 9.8e6 / (1.5e-4 × 0.05 × 1.225e13) = 9.8e6 / 9.1875e7 = 0.1067인 반면, ηλD / B = 9.375다. 인쇄된 target은 0.1066이므로 고정하는 양은 B / (ηλD)다. 같은 PDF를 layout 보존 없이 두 번째로 추출하면 `B`가 `ηλD`보다 먼저 출력되어 이 해석과 일치한다. 명시된 의미도 같다. η = 1.5e-4, λ = 0.05일 때 exponential average의 timescale은 1 / (ηλ) = 1.33 × 10⁵ steps다. step당 9.8 × 10⁶ tokens이면 1.31 × 10¹² tokens, 즉 12.25T dataset의 0.107이다. 식을 반대 방향으로 복사하고 다른 D나 B에서 η를 풀면 η = τB/(λD)를 얻지만, 올바른 식은 η = B/(τλD)다. 이는 인쇄된 target에서 τ² = 0.0114배만큼 틀린 값이다. 이것이 원장이 “derived” 상태를 두는 이유다. 한 줄로 계산해 인쇄된 수를 재현하거나 재현하지 못한다. 상태: **도출됨(derived)**.

**출처가 이전되지 않는다고 말하는 것.**
- Base-model family: 같은 Tülu 3 데이터와 stage chain에서 “OLMo 2 required significantly higher learning rates compared to the Llama 3.1 training recipe”라고 보고한다([[olmo-2-stage-chain]] §5).
- 작은 크기에서의 알고리즘 실행 가능성: 7B dense base와 16B MoE base에서는 base model에서 시작한 RL이 AIME를 개선하지 못했고, 길이가 길어질수록 repetition이 나타났다. 반면 32B dense, 230B MoE, 671B MoE base에서는 작동했다([[deepseek-r1]] G.1).
- 고정 크기에서 RL에 대한 distillation: 32B에서 Qwen2.5-32B-Base로 10K steps를 넘겨 실행한 RL은 AIME 2024 47.0, LiveCodeBench 40.2에 도달했다. R1 800K set에서 SFT한 모델은 각각 72.6, 57.2였다([[deepseek-r1]] Table 16).
- Safety mixture: Llama 3 Figure 18은 비슷한 safety를 위해 8B가 70B보다 더 높은 safety data 비율을 필요로한다고 보고하지만, 비율 자체는 인쇄하지 않는다([[llama-3-recipe]] §5.4.3).

**범용 모델에 대한 함의.** 이 비교에서 변하지 않고 이전되는 설정은 구조적 설정이다. 즉 length-normalized DPO loss의 β, epochs, warmup ratio, packing, loss masking이다. 이전되지 않는 것은 step size와 batch를 정하는 설정이다. 더 작은 크기에서 재현할 때는 구조를 유지하고 sweep을 다시 실행해야 하며, sweep 범위는 공개된 범위에서 가져와야 한다. Tülu 3는 한 preference mix에서 70B DPO learning rate를 5.0e-7, 2.0e-7, 1.5e-7, 1.0e-7로 sweep했고 development average는 각각 72.74, 71.17, 71.12, 71.06이었다. 두 번째 mix에서는 5.0e-7과 2.0e-7을 sweep해 71.14와 74.35를 얻었고, 2.0e-7을 선택했다. 보고서의 자체 해석은 “either a learning rate of 2.0 × 10⁻⁷ or 5.0 × 10⁻⁷, depending on data mix, performs better than a lower learning rate”다. 따라서 범위는 이전되지만, 그 범위 안의 최적 점은 이전되지 않는다([[tulu-3-stage-chain]] Table 19).

## §5 증거 사슬: 각 전환을 통과시킨 측정은 무엇이었는가

**정의.** gate는 한 단계를 끝냈다고 결정하거나, 다음 단계로 넘길 후보 checkpoint를 정하는 데 사용한 측정값이다.

**작동 방식.** 이 레시피들에는 네 종류의 gate가 나타난다.
1. **held-out 행동에 대한 threshold gate.** Llama 3.1의 long-context 단계는 “short-context evaluations fully recovered”했고 그 길이에서 needle-in-a-haystack을 풀었을 때에만 다음으로 진행했다([[llama-3-recipe]] §3.4.2).
2. **주기적 평가와 checkpoint 선택.** Tülu 3는 RLVR checkpoint를 매 100 steps마다 평가했고(70B는 40 steps마다), MATH와 IFEval이 가장 좋은 checkpoint를 release했다. 또한 “For all of these models we took an earlier than final checkpoint from the run”이라고 밝혔다([[tulu-3-stage-chain]] §6.3).
3. **sweep, 정성 확인, merge.** Olmo 3는 “to avoid overfitting”하기 위해 두 epochs 동안 학습하고, learning rate를 sweep했으며, evaluation suite로 후보를 선택하고, 일련의 정성적 “vibe-test” 질문을 실행했다. 그 뒤 서로 다른 learning rate로 학습한 두 checkpoint를 선형 가중 merge한 모델을 release했다([[olmo-3-stage-chain]] §4.2).
4. **단계 수준 regression 확인.** SmolLM3는 APO 뒤 RULER가 하락한 것을 발견하고, 원인을 reasoning mid-training 단계 및 APO data의 24k-token cap으로 추적했다. 그리고 그 단계를 다시 실행하는 대신 merge로 이를 고쳤다([[smollm3-stage-chain]]).

**작업 예: stage table을 증거로 읽기.** 세 실행의 평균인 Olmo 3 Instruct 7B의 단계별 수치([[olmo-3-stage-chain]] Table 26)는 다음과 같다. SFT, DPO, RL을 거치며 MATH는 65.1 → 79.6 → 87.3, LiveCodeBench v3는 20.0 → 18.8 → 29.5, PopQA는 16.5 → 20.7 → 14.1, AlpacaEval 2 LC는 21.8 → 43.3 → 40.9, Safety는 89.5 → 89.9 → 87.6이다. 여기 열거한 다섯 benchmark 중 세 개는 최종값이 가장 좋았던 중간 단계의 값보다 낮다. 이 표는 사슬이 목표 metric을 높인다는 증거이면서, 마지막 단계가 DPO checkpoint와 비교해 PopQA 6.6점과 Safety 2.3점을 치른다는 증거이기도 하다. 이 두 해석은 같은 세 숫자에서 나온다. 최종 열만 기록하는 ledger는 두 번째 정보를 잃는다.

**단계 순서 자체는 어디서 시험했는가.** 여덟 레시피 가운데 고정된 pipeline 내부의 설정이 아니라 pipeline variant를 비교해 보고한 것은 둘뿐이다. DeepSeek-R1은 4-stage pipeline의 intermediate checkpoint(§4, Table 3)를 보고했고, 별도로 32B에서 distillation과 RL을 비교했다(Table 16). Olmo 3는 “training Olmo 3 Instruct on top of the Olmo 3 Think SFT both increases model performance on benchmarks, as shown in Table 29”라고 보고했으며, response length는 “minimally affected”였다고 한다(§5.5). 나머지 보고서의 모든 실험은 가정된 순서 내부에서 하는 sweep이다. 순서 자체의 상태는 **열린 질문(Open question)**이다. 여기 인용한 어느 출처도 같은 크기에서 같은 데이터를 두 stage order로 통과시키지 않는다.

**조건과 한계.** Table 26은 세 실행의 평균이고, Table 3은 단일 pipeline이다. OLMo 2의 stage table(Table 16)은 실행 수를 명시하지 않으며, §5 본문에서는 “we experiment with 1 random seed initially to arrive on a configuration and up to 4 with final hyperparameters”라고 쓴다. 이 표들에서 인접 단계 간 1~2점 차이는 보고서 자체로는 seed variance와 구별되지 않는다. 그 차이를 어떻게 bound할지는 ch-51에서 다룬다.

## §6 각 레시피가 범용 능력을 측정한 위치와, 목표 능력만 측정한 위치

**정의.** 범용성 측정(generality measurement)이란, 지금 측정하는 stage들에서 어떤 개발 의사결정에도 사용하지 않은 과제에 대한 평가다.

**레시피들이 한 일.**
- **Tülu 3**는 development suite와 unseen suite를 만들었다. knowledge, reasoning, math, coding, instruction following의 다섯 능력마다 development benchmark와 unseen benchmark 한 개 또는 두 개를 짝지었으며, safety에는 development benchmark는 있지만 unseen benchmark는 없다. 팀은 모델을 개발할 때 “did not examine scores on our unseen set”이라고 명시했다([[tulu-3-stage-chain]] §2.1, Table 3). 그 뒤 매 stage에서 두 suite를 모두 보고했다(Table 31).
- **OLMo 2**는 “a held-out suite of tasks which were not used for model development decisions”를 유지하고 이를 보고했다. 다만 Table 6의 caption에는 “OLMo 2 models were not evaluated on held-out datasets prior to release”라는 단서가 있다([[olmo-2-stage-chain]] §2.5).
- **Olmo 3**에는 `OlmoBaseEval HeldOut` 블록이 있으며, caption 역시 모델이 “was not evaluated on held-out benchmarks prior to release”였다고 적는다([[olmo-3-stage-chain]] Tables 2–3).
- **Apertus**는 Table 21을 “were held-out during model development and were not used for making decisions”인 benchmark라고 선언한다. 또한 지역별 지식이 평균값 속에서 사라지지 않도록 `Factual Agnostic` 점수와 `Factual Regional` 점수를 분리한다([[apertus]] §5.2; held-out 선언은 Table 21, 분리는 Table 15).
- **Olmo 3 RL-Zero**의 script는 학습한 domain만 평가한다. 즉 수학은 AIME, 코드는 HumanEval+/MBPP+/LiveCodeBench, instruction following은 IFEval이다. 반면 Think와 Instruct script는 PopQA, MMLU, GPQA, AlpacaEval을 포함한 15개 benchmark 목록을 실행한다([[allenai-olmo3-open-instruct-scripts]]).
- **Qwen2.5**와 **DeepSeek-V3**는 held-out 또는 unseen 분리를 설명하지 않은 benchmark 표를 보고한다.

**작업 예: unseen suite가 잡아내는 것.** Tülu 3의 Table 31에서 8B의 coding skill을 보면, development HumanEval은 86.2 (SFT) → 83.9 (DPO) → 83.9 (final)이고, unseen BigCodeBench는 11.5 → 9.5 → 7.4다. development metric은 2.3점 내려가지만 unseen metric은 4.1점 내려간다. 4.1점은 시작값의 36%다. 70B의 같은 칸은 각각 92.9 → 92.4 → 92.4 및 12.2 → 23.0 → 21.6으로 움직인다. unseen benchmark에서는 반대 방향이다. 보고서 §7.4.1은 “For Reasoning and Coding, where the SFT checkpoints have the best performance on development evaluations, the subsequent training stages still improve model performance on harder unseen evaluations”이라고 쓴다. 이는 70B 열에는 맞지만 8B 열에는 맞지 않는다([[tulu-3-stage-chain]] Table 31). 보고서는 SFT mixture에 대해서도 “we see that our choices overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning”이라고 적는다.

**두 번째 종류의 점검.** K2-V2는 benchmark 질문의 문장 수준 memorization을 직접 측정한다. AIME 문항의 문장 prefix를 조건으로 beam search를 실행한다. Table 4의 결과는 K2가 AIME 2024에서 23.32%, AIME 2025에서 14.43%, Qwen2.5 72B가 41.96%와 23.37%, Llama 3.1 70B가 9.84%와 9.85%, Olmo 3 32B가 37.82%와 14.47%다([[k2-v2]] Table 4). 논문이 밝힌 동기는 “prior works reported that effect of such decontamination on benchmark performance is unclear”라는 점이다. 상태는 **결과(Result; 단일 연구)**다. 하나의 측정 protocol과 네 모델만을 다룬다. contamination 탐지는 ch-48에서 다룬다.

**범용 모델에 대한 함의.** 8개 레시피 중 두 개만이 stage별로 개발에 사용한 평가와 사용하지 않은 평가를 분리해 보고하며, 그 둘 중 하나에서는 어떤 skill이 자기 summary와 반대 방향으로 움직였음을 발견했다. end-to-end 계획은 unseen suite를 마지막 보고서에만 넣지 말고, ledger의 열로 계속 유지해야 한다.

## §7 사슬 안의 긴 문맥: 위치가 무엇을 보존할지를 결정한다

**정의.** 긴 문맥 단계(long-context stage)란 학습 sequence length를 늘리는 실행이다. 이 단계를 사슬의 어느 위치에 두는지가, 뒤에 오는 어떤 stage가 그 능력을 되돌려 놓을 수 있는지를 결정한다.

**관측된 위치.**
1. **사전학습 내부, 마지막에 배치.** Qwen2.5는 “final pre-training stage”에서 context를 4,096에서 32,768로 올리고, RoPE base를 10,000에서 1,000,000으로 올린다([[qwen-2.5-recipe]] §3.3).
2. **사전학습 직후, 두 개의 짧은 phase로 배치.** DeepSeek-V3는 1,000-step phase를 두 번 실행한다. 첫 phase는 batch 1920에서 4K → 32K, 둘째는 batch 480에서 32K → 128K다. 두 phase 모두 final pre-training learning rate인 7.3e-6을 사용하며, 합계 119K H800 GPU hours가 든다([[deepseek-v3-recipe]] §4.3, Table 1).
3. **mid-training 뒤, 별도 base stage로 배치.** Olmo 3는 65,536 길이에서 7B 모델은 50B tokens, 32B 모델은 100B tokens를 실행한다. 데이터는 long 34%, short 66%이고, 8,192에서 YaRN factor 8을 적용하며, intra-document masking을 사용한다([[olmo-3-stage-chain]] §3.6; [[olmo-core-olmo3-configs]]).
4. **mid-training 내부의 여러 stage로 배치.** K2-V2는 mid-2, mid-3, mid-4를 각각 65,536, 131,072, 524,288 tokens에서 실행한다. RoPE base는 각각 1M, 10M, 10M이고, learning rate는 계속 6e-6이다([[k2-v2]] Table 3).
5. **길이를 제한하는 post-training stage보다 먼저 배치.** SmolLM3는 각각 50B tokens인 두 stage에서 64k까지 확장한 다음, 데이터가 “limited to 24k tokens”인 reasoning mid-training과 APO를 실행한다. 이후 RULER 하락을 관측한다([[smollm3-stage-chain]]).

**위치 5에서 regression이 생기는 기제.** 뒤 stage가 확장한 문맥보다 짧은 sequence만으로 학습한다. 그 stage의 gradient는 확장 stage가 맞춘 parameter를 이동시키며, objective 안에는 cap 너머의 행동을 보존하는 항이 없다. SmolLM3의 수리는 aligned weights 90%와 긴 문맥 행동이 강한 mid-training checkpoint 10%를 유지하는 방식이다. 이 방법은 최대 128k-token context에서 “recover[ed] the base model's RULER score”를 달성했다([[smollm3-stage-chain]]). Llama 3.1은 다른 방법으로 같은 실패를 피한다. DPO에는 short-context data만 쓰지만, SFT 모델의 long-context 능력이 강하면 이것이 긴 문맥을 해치지 않았다고 보고한다. SFT mixture에서는 ablation으로 선택한 long-context data 0.1%를 포함했다([[llama-3-recipe]] §4.3.4).

**작업 예: 전체 budget에서 extension stage가 차지하는 비율.** Olmo 3 7B에서는 long-context 50B tokens를 총 5.93T + 100B + 50B = 6.08T tokens와 비교하므로 0.82%다. Llama 3.1 405B에서는 약 800B를 15.6T와 비교하므로 5.1%다. DeepSeek-V3에서는 119K를 총 2,788K H800 GPU hours와 비교하므로 전체 compute의 4.3%이며, token 수는 보고되지 않았다. K2-V2에서는 590B + 229B + 131B = 950B를 14.97T와 비교하므로 6.3%다. 충분한 값을 출력한 모든 레시피에서 이 stage는 budget의 0.8%에서 6.4% 사이다.

**조건과 한계.** 긴 문맥 데이터 선택을 비교하여 선택 근거를 보고한 것은 Olmo 3와 Llama 3.1뿐이고, 나머지는 설정만 출력한다. Olmo 3의 mixture ablation은 10B-token extension을 사용했다. 선택된 34% long / 66% short split은, 66% long / 34% short가 OlmoBaseEval의 일부를 2.5점 낮춘 것과 비교해 0.8점만 낮췄다([[olmo-3-stage-chain]] §3.6.3). K2-V2는 수치 없이 “mild degradation in stage 4”를 보고한다([[k2-v2]] §5). Qwen2.5는 inference-time YaRN과 Dual Chunk Attention이 없을 때 RULER 128K가 67.0, 있을 때 88.4라고 보고한다. 이것은 training 설정이 아니라 inference 설정이다([[qwen-2.5-recipe]] Table 16). 긴 문맥의 기제와 주장된 길이와 실제 유효 길이의 차이는 ch-32b와 ch-32c에서 다룬다.

## §8 증류를 배치하는 위치: 어느 stage가 다른 모델의 출력을 들여오는가

**정의.** 여기서 증류(distillation)는 보통 더 강한 다른 모델에서 sample한 출력으로 학습하는 것을 뜻한다(ch-20, ch-35). 자기 증류(self-distillation)는 현재 run 자신의 이전 checkpoint 출력으로 학습하는 것이다.

| 레시피 | teacher 출력이 들어오는 위치 | 들여오는 것 | 해당 위치의 증거 |
|---|---|---|---|
| DeepSeek-V3 | SFT (`distill-SFT`) | SFT와 RL로 학습한 domain별 expert 모델의 응답. R1 스타일의 긴 chain도 포함한다 | DeepSeek-V2.5에서 short-chain baseline 대비 LiveCodeBench-CoT 31.1 → 37.4, MATH-500 74.6 → 83.2 ([[deepseek-v3-recipe]] Table 9) |
| DeepSeek-R1-Distill (6개 student) | SFT만 | Qwen2.5 및 Llama base 위에서 쓰는 804,745-sample R1 set | 같은 base에서 RL로 학습한 경우의 47.0과 비교해 Distill-Qwen-32B의 AIME 2024는 72.6 ([[deepseek-r1]] Table 16) |
| Olmo 3 Think | DPO | delta-learning pair의 chosen 쪽에 Qwen3 32B trace를 둔다. Think SFT mixture는 public reasoning dataset 이름을 열거하지만, 여기서 읽는 passage는 그 teacher를 명시하지 않는다 | Qwen3 32B trace로 SFT를 더 수행하면 “outright hurts the performance of Olmo 3 Think SFT” ([[olmo-3-stage-chain]] §4.3) |
| SmolLM3 | preference stage만 | chosen은 Qwen3-32B, rejected는 Qwen3-0.6B이며, 둘 다 off-policy다 | 내부 ablation 수치는 출력하지 않았다 ([[smollm3-stage-chain]]) |
| K2-V2 | SFT | 응답은 “mostly regenerated using GPT-OSS-120B at low, medium and high reasoning effort levels” | Fig. 19의 reasoning-effort curve. teacher 유무 비교는 없다 ([[k2-v2]] §6.1–6.2) |
| Llama 3.1 | SFT (self-distillation) | 이전 round의 최고 checkpoint에서 prompt당 K = 10–30개 sample을 뽑고, 그 round의 reward model로 순위를 매긴다 | ablation을 보고하지 않았다 ([[llama-3-recipe]] §4.2.2) |
| OLMo 2 | preference stage | 20개 모델의 응답을 GPT-4o-2024-08-06으로 평점 매긴 뒤 최고 평점 응답을 chosen으로 둔다 | rater에 대한 ablation은 없다 ([[olmo-2-stage-chain]] §5) |
| Tülu 3 | preference stage | 같은 off-policy generation scheme을 이어받는다 | Table 18은 generator가 아니라 loss를 선택한다 ([[tulu-3-stage-chain]]) |

**두 위치와 두 failure mode.** teacher 출력이 SFT에 들어오면 ceiling은 teacher가 된다. Olmo 3는 포화(saturation)를 보고한다. 즉 Qwen3 32B trace를 더 imitation하면 성능이 내려가며, 같은 trace를 대신 preference pair의 chosen 쪽으로 변환한다([[olmo-3-stage-chain]] §4.3). teacher 출력이 preference stage에 들어오면 failure mode는 반대다. pair가 더 이상 대비를 이루지 못한다. Olmo 3는 OLMo 2 preference pipeline에서 더 강한 response generator가 “failed to yield any improvements”였다고 보고하며, chosen과 rejected 응답 사이에 더 이상 “meaningful contrast”가 없었기 때문이라고 설명한다(§5.5, Table 32). SmolLM3는 chosen을 Qwen3-32B에서, rejected를 Qwen3-0.6B에서 가져와 의도적으로 이 격차를 넓힌다.

**범용 모델에 대한 함의.** SFT에서 teacher 출력을 들여오는 레시피는 teacher의 coverage와 그 coverage의 빈칸까지 물려받는다. preference pair의 chosen 쪽에서 들여오는 레시피는 teacher의 coverage가 아니라 teacher의 ranking을 물려받는다. 둘 중 어느 것이 얼마나 물려받는지는 prompt-selection과 filtering 선택이 정하며, 이는 ch-35와 ch-35a에서 다룬다.

## §9 단계별 budget과 외부 추정으로 복원할 수 있는 것

**공개된 값.**

| 레시피 | 사전학습 | 긴 문맥 | 사후학습 | 근거 위치 |
|---|---|---|---|---|
| DeepSeek-V3 | H800 GPU hours 2,664K (토큰 1조 개당 180K, 2,048-GPU cluster) | 119K | 5K (SFT + RL) | [[deepseek-v3-recipe]] Table 1 |
| DeepSeek-R1 (V3 위에서) | — | — | 101K (R1-Zero) + 41K (R1) + 5K (SFT data creation) = 147K | [[deepseek-r1-recipe]] Table 7 |
| SmolLM3 | H100 GPU 384개로 24일 | 분리해 보고하지 않음 | 보고하지 않음 | [[smollm3-stage-chain]] |
| Olmo 3 7B | H100 512개에서 5.93T tokens | 256개에서 50B | GPU hours로는 보고하지 않음 | [[olmo-core-olmo3-configs]] README |
| Olmo 3 32B | H100 1,024개에서 5.50T | 1,024개에서 100B | 보고하지 않음 | 동일 |
| Tülu 3 8B | 해당 없음 | 해당 없음 | GPU 32개에서 SFT 6시간 | [[tulu-3-stage-chain]] §4.3 |
| Llama 3.1 405B | 3.8 × 10²⁵ FLOPs; GPU 8,192개에서 BF16 MFU 43%, 16,384개에서 41% | sequence 131,072에서 GPU 16,384개로 MFU 38% | 보고하지 않음 | [[llama-3-recipe]] §3.2, Table 4 |
| OLMo 2, Qwen2.5, K2-V2 | 보고하지 않음 | 보고하지 않음 | 보고하지 않음 | [[olmo-2-stage-chain]], [[qwen-2.5-recipe]], [[k2-v2]] |

**계산 예: 사후학습 비중.** DeepSeek-V3의 사후학습은 총 H800 GPU hours 2,788K 중 5K이므로 run의 0.18%이고, 사전학습 수치만을 분모로 하면 0.19%다. R1의 147K를 V3의 2,788K에 더하면, 전체 reasoning pipeline에 쓴 비용은 V3 budget의 5.3%다. 어느 비율도 그 stage들이 모델을 얼마나 바꾸었는지는 측정하지 않는다. V3의 5K hours에는 공개 chat model을 V3-Base와 구분하게 만드는 자체 SFT와 GRPO stage가 들어 있고, R1의 147K hours에는 stage table에서 AlpacaEval 2.0을 24.7에서 87.6으로 옮기는 pipeline이 들어 있다([[deepseek-r1]] Table 3). compute 비중과 capability 변화는 별개의 양이다. compute 열만 복사한 budget 분할은 모델 행동을 정하는 stage에 1% 미만을 배정하게 된다.

**외부 추정이 복원한 것과 복원하지 못한 것.** R1 논문이 GPU hours를 출력하기 11개월 전인 2025년 1월, Epoch AI 분석은 논문의 수치에서 RL 비용을 재구성했다. total gradient steps는 “around 8000”, mean completion length는 약 4,000 tokens였고, 출력되지 않은 두 양 B와 G는 이전 DeepSeek 연구를 바탕으로 각각 1,024와 64로 가정했다. 작성자는 “This is the biggest source of uncertainty in my calculation”이라고 명시했다([[epoch-deepseek-r1-compute]]). 결과는 첫 RL phase에 6.1e23 FLOP, V3 위의 R1에 “around $1M”였다. 뒤에 공개된 값은 R1-Zero에 H800 GPU hours 101K, R1에 41K, SFT data creation에 5K로 총 147K였고, 추정에서 가정해야 했던 rollout 크기도 출력했다. step당 질문 32개와 질문당 sample한 출력 16개다([[deepseek-r1-recipe]] v2 B.4.4 Table 7; [[deepseek-r1]] §2.1). 추정 자체가 V3에 사용한 H800 시간당 $2를 적용하면 147K hours는 $294K이다(도출값: 147,000 × 2). 따라서 추정치는 공개된 수치의 약 3.4배다. 작성자가 미리 불확실하다고 지목한 가정이 틀렸고, 비용 차이보다 더 크게 틀렸다. 가정한 B × G = 1,024 × 64 = step당 sample한 출력 65,536개는 공개된 32 × 16 = 512개와 다르다. 다른 입력들이 이 비율의 대부분을 상쇄한다. 추정은 total gradient steps 8,000을 썼지만 논문은 R1-Zero만으로 10,400을 제시하며, 추정에는 4,000-token mean completion과 pre-training MFU도 들어간다. 어느 입력이 얼마만큼 상쇄하는지는 두 문서 모두 말하지 않는다. 출처 신뢰도: practitioner가 쓴 견해성 newsletter issue다. 여기서는 training setting의 근거가 아니라 외부에서 무엇을 복원할 수 있는지의 근거로만 사용한다.

**재현 계획 template.** 사슬의 각 stage마다 계획에는 여섯 칸이 필요하다. 표는 ledger의 값으로 채우며, 빈 칸은 그 계획의 위험 목록이다.

| 칸 | 채우는 근거 | 보고되지 않았을 때 |
|---|---|---|
| 시작 checkpoint | launch script에 있는 공개 checkpoint 이름 또는 path | 어떤 공개 checkpoint로 대체하는지와 사슬이 이제 달라졌음을 명시한다 |
| 데이터와 크기, tokens 단위 | report table 또는 dataset card. §1처럼 정규화한다 | examples × mean length로 도출하거나, “not reported”로 기록하고 자신의 budget을 명시적으로 고른다 |
| optimizer와 schedule | pinned commit의 공개 config, 그다음 논문 table, 그다음 논문 prose | 공개된 sweep 범위를 써서 자신의 규모에서 다시 sweep한다(§4) |
| batch, 코드가 소비하는 단위 | parallelism flags의 곱 | 단위가 없는 채로 출력된 batch를 복사하지 않는다 |
| gate | checkpoint 선택에 사용한 평가와 cadence | run을 시작하기 전에 하나를 정의한다. gate가 없는 stage에는 stopping rule도 없다 |
| budget | GPU hours 또는 GPU 수 × wall clock | tokens × 6N으로 추정하고, 가정으로 둔 MFU를 명시한다 |

## 음성 샘플과 음성 피드백

이 장의 각 레시피는 서로 다른 세 지점에서 음성 사례를 다룬다. 또한 ``음성(negative)``의 네 가지 뜻(style §6.1)은 서로 다른 stage에 적용된다.

1. **음성 사례의 출처.** rejection sampling은 낮은 순위이거나 틀린 sample을 버린다. Llama 3.1은 prompt마다 K = 10–30개의 completion을 sample하고 reward-model 점수가 가장 높은 것을 남긴다([[llama-3-recipe]] §4.2.2). DeepSeek-R1의 Dev3 stage는 stage-1 RL checkpoint의 정답 응답만 남겨 reasoning sample 약 600k개를 만들며, 언어가 섞인 chain, 긴 문단, code block을 제거한다([[deepseek-r1-recipe]] B.3.3). 이것은 **음의 한계가치(negative marginal value)**다. 버린 sample은 loss에 전혀 들어가지 않는다. preference stage는 평점으로 음성을 붙인다(OLMo 2는 GPT-4o가 네 축으로 평점 매기고 최고 평점 응답을 chosen, 무작위의 더 낮은 평점 응답을 rejected로 둔다). 또는 생성 방법으로 붙인다(SmolLM3의 rejected 응답은 Qwen3-0.6B가 생성한다). RL stage는 verifier로 붙인다. Tülu 3의 경우 정답이면 v(x, y) = 10, 아니면 0이다([[tulu-3-stage-chain]] §6).
2. **실제 학습에서 하는 일.** SFT에서는 버리고, preference와 RL stage에서는 **gradient로서의 음성**을 사용한다. length-normalized DPO는 rejected 응답의 평균 token log-probability에 gradient를 적용한다([[allenai-olmo3-open-instruct-scripts]] `dpo_utils.py` L83–84, L556–559). group-relative RL은 평균보다 낮은 모든 sample에 음의 advantage를 준다.
3. **기제.** logit z에 대한 softmax에서 ∂ log p_y / ∂ z_j = 1[j = y] − p_j이다. 그러므로 sample한 sequence의 likelihood를 낮추면 그 확률 질량이 현재 가장 가능성 높은 대안으로 재분배되며, 그 대안이 정답일 필요는 없다. Olmo 3의 기본 `centered` advantage normalization은 prompt 하나의 8개 sample에 대해 A = r_i − mean(r)을 쓴다. reward 10.0의 정답 sample이 8개 중 하나면 정답의 A는 8.75, 나머지 7개의 A는 각각 −1.25다. `standard` normalization에서는 같은 group이 각각 2.646과 −0.378이 된다([[allenai-olmo3-open-instruct-scripts]] `data_loader.py` L421, L1417–1420). 따라서 음의 방향으로 미는 크기는 PR #1547 뒤 launch script가 더 이상 출력하지 않는 framework 기본값에 의존한다.
4. **수치가 있는 증거.** Olmo 3.1 RL-Zero가 3.0보다 개선된 주된 이유는 completion limit을 12k에서 16k tokens로 올리고 ``not masking truncated sequences``를 택한 데 있다. masking은 batch size를 변하게 하고 안정성을 낮췄으며, ``without training on overlong negative sequences, completion lengths were higher, on average``였다([[olmo-3-stage-chain]] A.6.4). 반대 방향의 사례로 Tülu 3 RLVR은 β = 0.01에서 답 없이 쉼표로 분리한 문자 `e` 14개만으로 된 응답을 냈고, β = 0.1에서는 정답을 냈다([[tulu-3-stage-chain]] §6.2.1). 음성 신호가 퇴화한 출력으로 충족된 것이다.
5. **레시피가 쓰는 control.** variance가 0인 group은 batch 전에 버리므로 전부 맞거나 전부 틀린 prompt는 gradient를 내지 않는다. 두 Olmo 3 script는 정답률이 0.875 이상인 prompt를 이후 sampling에서 제외하며, ratio correction은 2.0으로 clamp한다([[allenai-olmo3-open-instruct-scripts]]). DeepSeek-R1은 reward가 오르는데 Codeforces 성능이 떨어지는 것을 관측한 뒤, preference reward를 1,700 RL step 중 마지막 400 step으로 한정했다([[deepseek-r1]] §3.2.2, B.5).
6. **end-to-end run의 진단.** preference stage에서는 chosen과 rejected의 log-probability를 따로 기록한다. RL stage에서는 advantage 부호별 통계를 기록한다. §6의 unseen suite, 높은 k에서의 pass@k, abstention rate를 stage 경계마다 유지한다. §5의 stage table에는 마지막 stage에서만 내려가는 benchmark가 있기 때문이다.
7. **일반성에 미치는 효과.** 이 출처들에서 양 방향을 분리한 유일한 측정은 Olmo 3의 negative control이다. base model을 random binary reward가 달린 RL-Zero mixture로 학습해도 ``does not improve performance on any of our benchmark`` suite([[olmo-3-stage-chain]] §6.2). reward가 실제 신호일 때 같은 절은 multi-domain run이 ``has improved performance across different domains, but each domain is under-optimized compared to the single-domain setup``이라고 보고한다.

## 레시피

각 행은 공개된 checkpoint 하나의 stage 하나다. 단위는 다음과 같다. tokens는 한 경로에서 본 token 수, batch는 출처가 출력한 단위와 함께 그 도출식을 표시한 값, `pairs`는 preference pair, RL batch는 prompt 수 × prompt당 sample 수다. 상태 날짜는 명시한 위치에서 2026-09-14부터 2026-09-17까지 읽은 날짜다.


Rows are one stage of one released checkpoint. Units: tokens are tokens seen along one path; batch is stated in the unit the source prints, with the derivation shown; "pair" are 선호 pair; RL batch is prompt × samples per prompt. 상태 dates: values read on 2026-09-14 to 2026-09-17 at the stated loci.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| OLMo-2-1124-7B | 7B | 안정 사전학습 | 토큰; batch × 길이; 최대 LR; cosine horizon | 3.90T; 1024 × 4096; 3.0e-4; 5T truncated at 4T | arXiv:2501.00656v3 §2.3, Table 3 | 검증됨 2026-09-15 | §4.1: peaks 3–12e-4 differ by under two points on 9 OLMES tasks after a decay, one seed |
| OLMo-2-1124-13B | 13B | 안정 사전학습 | 최대 LR | 9.0e-4 (Table 3) vs 3.0e-4 (released stage-1 YAML) | Table 3; `OLMo2-13B-stage1.yaml` L46 @090253d | 충돌 | §4.1 prose: "The 13B ran with a higher peak learning rate from the start"; the stage-2 start LR of 9e-5 equals 0.1 × 9e-4 (도출값) |
| OLMo-2-1124-7B-SFT | 7B | SFT | 최대 LR; epoch; 배치; 길이 | 2e-5; 2; 128 sequences; 4,096 | arXiv:2501.00656v3 §5, Table 17 | 검증됨 2026-09-15 (max 길이 read from the Tülu 3 SFT mixture setting, not printed in §5) | §5 sweep 1e-5, 2e-5, 3e-5 selects 2e-5; Table 17 (7B, batch 128, linear, warmup 0.3): best row 2 epoch at 1e-5 with sum loss 49.97; 3 epoch at 4e-6 with mean loss 48.25 |
| OLMo-2-1124-7B-DPO | 7B | 선호 | loss; β; 최대 LR; 선호 prompt | `dpo_norm`; 5; 1e-6; 366.7k prompt | §5; `dpo_7b.sh` L28 | 충돌 | YAML prints 5e-7; §5 sweep selects 1e-6 for the 7B |
| OLMo-2-1124-7B-Instruct | 7B | RL | 알고리즘; β; LR; data | PPO with RM-initialized value; 0.05; 4e-7; GSM8K + MATH + constraint prompt | §5 | 검증됨 2026-09-15 | §5 sweep β 0.03/0.05/0.07; Table 16: AVG 55.9 → 56.5 over DPO |
| Olmo-3-1025-7B | 7B | 안정 사전학습 | 토큰; 배치; 최대 LR; schedule | 5.93T; 512 × 8,192 = 4,194,304 토큰; 3.0e-4 → 3.0e-5; cosine over 5T stretched to 5.93T | arXiv:2512.13961v2 Table 35, Fig. 3 | 검증됨 2026-09-15 | no ablation of the stretch reported |
| Olmo-3-1025-7B | 7B | 중간학습 | 토큰; 배치; LR | 100B; 2,097,152 토큰; 2.074e-4 → 0, warmup 0 | Table 35; `OLMo-3-1025-7B-midtrain.py` L40-43 | 충돌 | script prints 2.0712e-4; §3.5.4: decontaminated anneal compared against a matched non-decontaminated 100B anneal |
| Olmo-3-1025-7B | 7B | 긴 문맥 | 토큰; 길이; 배치; YaRN | 50B; 65,536; 64 × 65,536 = 4,194,304 토큰; factor 8 from 8,192 | Table 35; `OLMo-3-1025-7B-긴 문맥.py` L37-51 | 검증됨 2026-09-15 | §3.6.3: 34% long / 66% short lowered a subset of OlmoBaseEval by 0.8 points vs 2.5 for 66/34 |
| Olmo 3 7B Think SFT | 7B | SFT | prompt; total 토큰; epoch; 길이; 배치; LR | 2,268,468; 45.4B; 2; 32,768; 1,048,576 토큰; 5e-5 | Table 17, Table 47; `7b_think_sft.sh` L7-19 | 검증됨 2026-09-14 | §4.2: LR sweep, evaluation suite, qualitative checks, then a merge of two LR checkpoints |
| Olmo 3 7B Think DPO | 7B | 선호 | pair; LR; 길이; 배치; loss; β | 150,000; 8e-8 linear; 16,384; 128; `dpo_norm`; 5 | `7b_think_dpo.sh` L3-40; defaults at d8a7f1c | 검증됨 2026-09-14 | A.6.2: LR and dataset size swept; Table 48 matches |
| Olmo 3 7B Think RL | 7B | RL | prompt × samples; LR; response 길이; KL β; clip | 64 × 8; 1e-6 constant; 32,768; 0.0; 0.2 / 0.272 | `7b_think_rl_no_pipeline.sh` L20-49 | 검증됨 2026-09-14 | Table 49 matches; ablation을 보고하지 않음 |
| Olmo 3 32B | 32B | 모든 base stage | 7B 대비 차이 | batch doubled at every stage; 중간학습ing run twice and averaged; 긴 문맥 50B → 100B; pretrain 최대 LR 6.0e-4 | Table 35 caption | 검증됨 2026-09-15 | §3.5.4: the 중간학습ing soup gained "nearly a full point" on MCSTEM and 2.9 / 1.6 points on Math over the single runs |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | 최대 LR; epoch; 배치; 길이; warmup | 5e-6 linear; 2; 128 sequences; 4,096; 0.03 | Tülu 3 Table 11; `docs/tulu3.md` L39-43 | 검증됨 2026-09-14 | §4.3.2: "training for longer did not yield further improvements" |
| Llama-3.1-Tulu-3-8B-DPO | 8B | 선호 | loss; β; LR; 배치; epoch | `dpo_norm`; 5; 5e-7; 128 pair; 1 | Table 20; `docs/tulu3.md` L144-159 | 검증됨 2026-09-14 | Table 18: DPO-norm β 5, 1 epoch → 57.3 vs 55.7 for the SFT base and 55.2 for DPO |
| Llama-3.1-Tulu-3-8B | 8B | RL | 알고리즘; LR; β; 배치; response 길이; α | PPO; 3e-7; 0.05; 224 responses; 2,048; 10 | Table 21; `docs/tulu3.md` L313-346 | 검증됨 2026-09-14 | Table 21 caption; §6.2.1 Fig. 21: higher KL gives lower average scores |
| Llama-3.1-Tulu-3-70B | 70B | RL | β | 0.07 (caption, launch command) vs 0.7 (§6.4 prose) | Table 21 caption; §6.4; `docs/tulu3.md` L375-383 | 충돌 | the launch command that produced the checkpoint prints `--beta 0.07` |
| Llama-3.1-Tulu-3-405B | 405B | RL | 데이터; step; LR; batch | MATH only; stopped at 75 step; 1e-7; 1,856 | §8.1, Table 35 | 검증됨 2026-09-14 | §8.1: "even with as few as 25 RLVR step, MATH performance improved by over 5 points" |
| SmolLM3-3B | 3B | 안정 사전학습 | 토큰; 배치; LR; schedule; 하드웨어 | 11.2T; 2,359,296 토큰; 2e-4; WSD with 2,000 warmup step and linear decay over the final 10%; 384 H100 GPUs for 24 days | blog "Training Configuration"; stage configs L225-254 | 충돌 on total (11T / 11.2T / 11.1T) | playbook: batch 2M–4M "minimal impact", 2.36M chosen for throughput; WSD 10% chosen after a cosine comparison |
| SmolLM3-3B | 3B | 긴 문맥 | 토큰; stages; RoPE θ | 100B in 2 × 50B; 4k → 32k → 64k; 1.5M then 5M | blog; released configs | 충돌 | the 4k → 32k config prints `rope_theta: 2000000.0` |
| SmolLM3-3B | 3B | 선호 | 알고리즘; 데이터; 길이 cap | APO; Qwen3-32B chosen, Qwen3-0.6B rejected; 24k tokens | blog "Off-policy model alignment with APO" | 검증됨 2026-09-15; β, LR, batch, epoch 보고하지 않음 | "higher downstream performance in our internal ablations" (no numbers) |
| SmolLM3-3B | 3B | merge | 방법; 가중치 | soup of APO checkpoints, then linear merge 0.9 (soup) / 0.1 (중간학습ing checkpoint), MergeKit | blog "Model Merging" | 검증됨 2026-09-15 | "We were able to recover the base model's RULER score on contexts up to 128k tokens" (수치를 출력하지 않음) |
| Llama 3.1 405B | 405B | 안정 사전학습 | 토큰; compute; 최대 LR; batch schedule | 15.6T; 3.8 × 10²⁵ FLOPs; 8e-5; 4M → 8M → 16M tokens (§3.4.1 prints the middle step as "8M sequences of 8,192 tokens"; the surrounding values are in tokens) | arXiv:2407.21783v3 §3.4.1, Table 3 | 검증됨 2026-09-14 | §3.2.1: scaling law N*(C) = A C^α with (α, A) = (0.53, 0.29) extrapolates to 402B at this budget |
| Llama 3.1 405B | 405B | 긴 문맥 | stages; 토큰; gate | 6 stages, 8K → 128K, ≈800B tokens | §3.4.2 | 검증됨 2026-09-14 | stage advanced when short-context evaluations recovered and needle-in-a-haystack was solved at that 길이 |
| Llama 3.1 405B | 405B | SFT | LR; step; loss | 1e-5; 8.5K–9K step; prompt tokens masked | §4.1.3 | 검증됨 2026-09-14 (batch, epoch 보고하지 않음) | "work well across different rounds and data mixes" |
| Llama 3.1 405B | 405B | 선호 | rounds; DPO LR; β; NLL coefficient | 6 rounds; 1e-5; 0.1; 0.2 on chosen | §4.1.4, §4.1.6 | 검증됨 2026-09-14 | §4.1.4: DPO chosen over PPO for less compute and better instruction following; no numbers |
| Llama 3.1 405B | 405B | merge | 방법 | averaging of models from runs with different data or hyper-parameters at the RM, SFT and DPO stages | §4.1.5 | 검증됨 2026-09-14 (가중치 not printed) | ablation을 보고하지 않음 |
| Qwen2.5 Instruct (open 가중치) | 0.5B–72B | SFT | example; epoch; 길이; LR; weight decay; clip | >1M; 2; 32,768; 7e-6 → 7e-7; 0.1; 1.0 | arXiv:2412.15115v2 §4.1 | 검증됨 2026-09-14, except the example count, which is a 충돌 (§4.1 and Abstract "over 1 million" vs §1 "1 million" vs §4 intro "millions") | ablation을 보고하지 않음; optimizer, warmup and batch not given |
| Qwen2.5 Instruct | 0.5B–72B | 선호 | loss; pair; epoch; LR | DPO; ≈150,000; 1; 7e-7 | §4.2 | 검증됨 2026-09-14 (β not given) | ablation을 보고하지 않음 |
| Qwen2.5 Instruct | 0.5B–72B | RL | 알고리즘; samples per query; 배치; ordering | GRPO; 8; 2,048 samples per episode; queries ordered by reward variance | §4.3 | 검증됨 2026-09-14; LR, β, ε, temperature, 길이, episodes 보고하지 않음 | ablation을 보고하지 않음; the query ordering carries the stated aim "to ensure more effective learning" |
| DeepSeek-V3 | 671B / 37B act. | 안정 사전학습 | 토큰; LR; 배치; compute | 14.8T; warmup to 2.2e-4, constant to 10T, cosine to 2.2e-5 over 4.3T, then 2.2e-5 and 7.3e-6; 3072 → 15360 (unit not printed); 2,664K H800 hours | arXiv:2412.19437v2 §4.2, Table 1 | 검증됨 2026-09-14 | no schedule ablation reported |
| DeepSeek-V3 | 671B / 37B act. | SFT | instance; epoch; LR | 1.5M; 2; cosine 5e-6 → 1e-6 | §5.1 | 검증됨 2026-09-14 | Table 9: distilled long-chain data lifts LiveCodeBench-CoT 31.1 → 37.4 on V2.5 |
| DeepSeek-R1-Zero | 671B MoE | RL | LR; KL β; 질문 × 출력; 길이; step; compute | 3e-6; 0.001; 32 × 16; 32,768 then 65,536 after step 8.2k; 10,400 step; 101K H800 hours | arXiv:2501.12948v2 §2.1, B.4.4 | 검증됨 2026-09-14 | §2.3: AIME 2024 pass@1 15.6% → 77.9% |
| DeepSeek-R1 | 671B MoE | SFT (cold start) | example; epoch; LR; 길이; batch | "thousands"; 2–3; cosine 5e-5 → 5e-6; 32,768; 128 | §3, B.3.2, B.4.2 | 검증됨 2026-09-14; exact count 보고하지 않음 | Table 3: Dev1 IF-Eval 46.6 → 71.7, AIME 77.9 → 59.0 |
| DeepSeek-R1 | 671B MoE | SFT (rejection sampling) | samples; 시작 checkpoint | 804,745 (600k reasoning + ≈200k non-reasoning); DeepSeek-V3-Base, not the RL checkpoint | B.3.3, Table 5, Fig. 2 | 검증됨 2026-09-14 | Table 3: Dev2 → Dev3 AlpacaEval 2.0 55.8 → 62.1, Aider-Polyglot 25.6 → 44.8 |
| DeepSeek-R1 | 671B MoE | RL (mixed) | step; temperature; reward; 선호 window | 1,700; 0.7; rule + RM + format + language; 선호 rewards in the final 400 step | §3.2.2 | 검증됨 2026-09-14 | B.5 Fig. 6: with the helpful RM, reward rises while Codeforces pass@1 falls |
| K2-V2 | 70B | 안정 사전학습 | 토큰; 배치; LR; weight decay; step | 12.25T; 1,200 × 8,192 = 9.8e6 토큰; 1.5e-4 with a 1.5e-6 floor held constant; 0.05; 1.25e6 | arXiv:2512.06201v2 §3.2 | 검증됨 2026-09-17 | 257M pilot on 41.48B tokens: SlimPajama loss 2.815 (cosine d2z), 2.845 (to 10%), 2.817 (linear d2z) |
| K2-V2 | 70B | 중간학습 + 긴 문맥 | four stages; 토큰; context; RoPE; LR | mid-1 1769B at 8,192 (cosine 1.5e-5 → 6e-6); mid-2/3/4 590B / 229B / 131B at 65,536 / 131,072 / 524,288, constant 6e-6; RoPE 0.5M / 1M / 10M / 10M | Table 3 | 검증됨 2026-09-17 | §5: "mild degradation in stage 4" attributed to the scarcity of long reasoning data |
| K2-V2 | 70B | distill-SFT | epoch당 토큰; epoch; 배치; 길이; LR; teacher | 17.4B; 3; 128; 65,536; cosine 2e-5 → 2e-6 with 10% warmup; GPT-OSS-120B | §6.1, §6.3 | 검증됨 2026-09-17 | no teacher-versus-no-teacher comparison reported |

**작은 범용 run의 시작점.** 여기의 모든 수치는 위에서 검증된 행이며, 각 수치에는 원 출처가 사용한 조건이 함께 붙어 있다. 공개 instruction mixture에서 7–8B Llama-family base를 쓴다면 Tülu 3 8B 행은 최대 4,096 tokens, batch 128 sequence, warmup 0.03, linear 5e-6로 2 epoch의 SFT(32 GPU에서 6시간)를 제시한다. 그 다음 최대 2,048 tokens, 128 pair에서 β = 5, 5e-7로 1 epoch의 length-normalized DPO를 하고, 이어 PPO 기반 RLVR을 3e-7, β = 0.05, batch 224 응답, 응답 길이 2,048, 정답 보상 α = 10으로 실행하며 checkpoint는 100 step마다 평가한다. 7B OLMo-family base에서는 같은 stage도 다시 sweep해야 한다. OLMo 2는 같은 mixture의 자기 변형에서 SFT 2e-5와 DPO 1e-6을 골랐는데, 이는 Tülu 값의 각각 4배와 2배다. 어느 설정도 1B 미만 모델에 대해 제시되지 않았고, 어느 쪽에도 long-context 또는 mid-training stage는 없다. 그런 stage에는 Olmo 3 7B 행의 100B mid-training tokens(2.097M-token batch)와 65,536 길이의 50B long-context tokens(34% long / 66% short mixture)를 사용한다.

## 일반화 관점

**(a) 폭을 넓히는 것.**
- reasoning RL 뒤에 non-reasoning imitation stage를 더한다. DeepSeek-R1의 Dev3은 non-reasoning sample 약 200k개를 더해 AlpacaEval 2.0을 55.8에서 62.1로, Aider-Polyglot을 25.6에서 44.8로 올렸다([[deepseek-r1]] §4, Table 3).
- domain별 specialist를 학습하는 대신 하나의 RL run에 domain을 섞는다. 단 Olmo 3는 mixed run이 ``has improved performance across different domains, but each domain is under-optimized compared to the single-domain setup``이라고 보고한다([[olmo-3-stage-chain]] §6.2).
- 앞 checkpoint가 뒤 stage에서 잃은 capability를 가지고 있을 때 사슬 끝에서 merge한다. SmolLM3의 0.9 / 0.1 linear merge는 128k까지 base model의 RULER 점수를 복구했다([[smollm3-stage-chain]]). Olmo 3 32B mid-training soup은 입력 run보다 Math cluster에서 최대 2.9점을 얻었다([[olmo-3-stage-chain]] §3.5.4).
- short-context stage에도 작은 long-context 비중을 유지한다. Llama 3.1 SFT mixture의 long-context data 0.1%는 ablation으로 선택되었고 short와 long benchmark 모두를 최적화했다. short-only SFT는 ``caused significant long-context regression``이었다([[llama-3-recipe]] §4.3.4).

**(b) 폭을 좁히거나 망각시키는 것.**
- 확장한 context보다 짧게 sequence length를 cap하는 stage. SmolLM3의 24k-token APO가 이에 해당하며, RULER 하락은 reasoning mid-training stage까지 추적되었다([[smollm3-stage-chain]]).
- metric 일부만 최적화하는 마지막 RL stage. Olmo 3 Instruct 7B의 PopQA는 DPO 20.7에서 RL 14.1로, Safety는 89.9에서 87.6으로 내려갔다([[olmo-3-stage-chain]] Table 26). OLMo 2 32B Safety는 DPO 91.9에서 Instruct 85.9로 내려갔다([[olmo-2-stage-chain]] Table 16).
- RL 전체 run에서 preference reward를 계속 켜는 것. DeepSeek-R1은 reward 상승과 Codeforces 성능 하락을 본 뒤 1,700 step 중 마지막 400 step으로 한정했다([[deepseek-r1]] B.5).
- development suite로 checkpoint와 데이터를 고르는 것. Tülu 3는 SFT mixture 선택이 ``overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning``이라고 보고한다([[tulu-3-stage-chain]] §7.4.1).

**(c) 이 stage에서 측정하는 방법.** run 전에 능력별 unseen benchmark 하나를 정해 두고, development 동안 한 번도 보지 않은 채 모든 stage 경계에서 유지한다. 이는 Tülu 3 Table 3의 pairing 방식이다. Table 31처럼 stage마다 development와 unseen 값을 나란히 보고한다. n-gram filter에 의존하지 않는 contamination 측정도 추가한다. 예를 들면 K2-V2의 문장 수준 memorization probe가 있다([[k2-v2]] Table 4). OLMo 2와 Olmo 3가 table caption에 한 것처럼 held-out benchmark를 release 전에 평가했는지 선언한다. RL stage라면 Olmo 3 RL-Zero의 prompt filter가 base가 8/8번 푼 prompt를 제거해 pass@1의 의미를 바꾸므로, pass@1과 함께 큰 k에서의 pass@k를 기록한다([[olmo-3-stage-chain]] §6.1).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| 논문 설정을 복사하면서 공개 코드를 실행함 | 오류가 없는데 loss curve와 평가 궤적이 공개 결과와 다름 | 논문 table을 실행 commit의 launch script와 framework default에 대조한다. 최소 한 행에서는 불일치를 예상한다([[allenai-olmo3-open-instruct-scripts-recipe]]) |
| head commit의 launch script를 읽음 | 실행 당시 flag보다 적고 β, epoch, clipping이 없음 | defaults migration commit을 history에서 찾는다. open-instruct에서는 7917d41의 parent를 읽는다 |
| schedule horizon을 token budget으로 취급함 | 도출한 token-per-parameter와 compute가 너무 큼 | horizon과 논문이 명시한 tokens seen을 비교한다. playbook survey table은 OLMo 2 7B를 `5T`라고 쓰지만 보고서는 3.90T라고 한다([[smol-training-playbook]]) |
| SFT stage budget을 prompt 수로만 계산함 | 계획보다 compute가 몇 배 많이 듦 | examples × 평균 길이 × epoch로 token 정규화한다(§1). long-chain set은 항목당 평균 5,355 tokens(R1)에서 약 10,007 tokens(Olmo 3 Think)다 |
| base family 사이에서 LR을 그대로 옮김 | 같은 데이터와 stage에서 underfitting 또는 불안정 | 다시 sweep한다. OLMo 2는 같은 mixture에서 Tülu 3 값의 2–4배 SFT/DPO rate를 골랐다([[olmo-2-stage-chain]] §5) |
| context extension 뒤 short-context stage를 control 없이 실행함 | 다른 benchmark는 모두 오르는데 RULER 또는 needle 점수가 하락 | base stage 뒤에만 하지 말고 모든 stage 경계에서 long context를 평가한다([[smollm3-stage-chain]]) |
| 최종 checkpoint 점수만 보고함 | 마지막 stage에서 잃은 capability가 보이지 않음 | Olmo 3 Table 26과 OLMo 2 Table 16처럼 benchmark별 stage table을 유지한다 |
| 데이터 선택에 쓴 suite로 checkpoint도 선택함 | development 점수는 오르지만 unseen 점수는 정체하거나 하락 | unseen suite를 유지하고 release 결정 뒤에만 본다([[tulu-3-stage-chain]] §2.1) |
| 공개되지 않은 stage budget을 추정한 뒤 setting처럼 인용함 | 추정값이 측정값인 것처럼 계획에 전파됨 | 추정의 출처 유형을 표시한다. Epoch AI의 R1 추정은 나중 공개된 147K GPU hours의 약 3.4배였다([[epoch-deepseek-r1-compute]]) |

## 이해 점검

1. 여기서 사전학습을 하는 모든 레시피는 고유한 최종 pre-training stage로 사전학습을 끝내지만, long-context stage를 그 stage와 다른 관계로 배치한다. release checkpoint까지 long-context 능력이 살아남는 조건과, 레시피가 쓰는 두 가지 수리 방법을 설명하라.
2. Tülu 3는 8B에서 SFT LR 5e-6, 70B에서 2e-6을 쓰고, Olmo 3는 Think RL LR 1e-6(7B), 2e-6(32B)을 쓴다. 이 행들에서 ``learning rate falls with model size``를 결론으로 삼을 수 없는 이유와, 대신 무엇을 transfer해야 하는지를 쓰라.
3. Olmo 3 Instruct 7B stage table은 PopQA를 SFT 16.5, DPO 20.7, RL 14.1로 보인다. 마지막 수치에 대한 서로 다른 두 causal account와 그것을 구분할 측정을 제시하라.
4. 재현자가 오늘 Olmo 3 7B Think DPO launch script를 읽고 `--beta` flag가 없음을 발견했다. 이것이 공개 checkpoint와 다른 run을 만들 수 있는 두 방법과 둘 다 배제하는 점검을 설명하라.
5. DeepSeek-V3는 2,788K H800 GPU hours 중 5K를 post-training에 썼지만, post-training stage는 base model과 공개 chat model 차이의 대부분을 만든다. compute share와 capability share가 갈라지는 이유 및 작은 팀이 고정 budget을 나누는 데 주는 함의를 설명하라.
6. Tülu 3 unseen coding benchmark는 8B에서 stage를 거치며 11.5에서 7.4로 떨어지지만, 70B에서는 12.2에서 21.6으로 오른다. 두 결과와 양립하는 causal explanation 및 HumanEval만 평가하는 8B 계획에 주는 함의를 쓰라.
7. Epoch AI의 R1 RL cost 추정은 뒤에 논문이 출력한 값의 약 3.4배였고, 저자는 미리 원인이 되는 가정을 지목했다. 외부 budget estimate가 유용하려면 disclosure가 어떤 양을 포함해야 함을 뜻하는지 설명하라.
8. K2-V2는 SFT로 끝나는 chain의 data, code, log, checkpoint를 공개한다. team이 preference와 RL stage에 관해 배울 수 있는 것과 배울 수 없는 것, 그리고 보고서의 pass@k 분석이 왜 그 stage들에 대한 증거로 제시되는지 설명하라.

## 연결

- **이전:** ch-58 — 범용 capability를 측정하고 보호하기 위한 post-training stack 선택과 계측. 그 장은 framework와 instrumentation을 고르고, 이 장은 framework가 실행할 stage chain과 setting을 제공한다.
- **다음:** ch-59 — Capstone: 일반성 gate가 있는 공개 범용 모델 레시피의 한 stage 재현. §9의 reproduction plan template과 §3의 disclosure matrix가 재현할 stage 선택의 입력이다.
- **의존성:** ch-14a — 사전학습 레시피 나란히 보기: budget, batch, schedule, stage mixture(ledger 방법과 pre-training 행); ch-32e — Mid-training, annealing, context-extension 레시피 나란히 보기(mid-training과 long-context 행); ch-34 — 사례 연구 B: Qwen·OLMo·Phi 보고서에서 일반성과 전문화(report별 읽기를 이 chain으로 정규화); ch-45a — Preference-optimization과 RL stage 레시피 나란히 보기(preference와 RL 행); ch-45d — Open agentic recipe 나란히 보기: stage 배치, data mixture, agentic RL(같은 chain의 agentic 변형).
- **여기서도 사용:** ch-30c(merging), ch-32b와 ch-32c(context extension과 effective length), ch-35와 ch-35a(distillation placement와 filtering), ch-38a(SFT 대 RL 일반화), ch-47a와 ch-48(unseen suite와 contamination), ch-51(metric noise와 go/no-go 결정).

## 출처

- [[olmo-2-stage-chain]] — OLMo 2의 pre-training, anneal-and-soup, SFT, DPO, RLVR setting, stage table, held-out suite 선언.
- [[olmo-3-stage-chain]] — Olmo 3의 base/branch stage, Table 35 schedule 값, SFT/DPO 크기, stage별 Instruct table, RL-Zero negative control.
- [[olmo-core-olmo3-configs]] — 공개 base-stage script, stage별 GPU 수, config와 논문 차이.
- [[allenai-olmo3-open-instruct-scripts]] 및 [[allenai-olmo3-open-instruct-scripts-recipe]] — Olmo 3 post-training launch 값, framework default, PR #1547 migration, negative-feedback 절에서 쓴 advantage와 filtering 동작.
- [[tulu-3-stage-chain]] — 8B·70B·405B의 Tülu 3 SFT/DPO/RLVR setting, DPO algorithm ablation, development/unseen 분리, Table 31.
- [[open-instruct-allenai-recipes]] 및 [[open-instruct-allenai-recipes-recipe]] — Tülu 3과 OLMo 2의 launch command, 논문과 command 사이의 충돌.
- [[smollm3-stage-chain]] — SmolLM3의 세 pre-training stage, long-context stage, reasoning mid-training, APO, RULER를 수리한 merge.
- [[smol-training-playbook]] — SmolLM3 budget을 정한 방법, schedule/batch ablation, §1 정규화의 동기가 된 survey-table 오류.
- [[llama-3]] 및 [[llama-3-recipe]] — Llama 3.1 pre-training schedule, 여섯 stage의 long-context extension과 gate, 여섯 post-training round, DPO setting, averaging 단계.
- [[qwen-2.5]] 및 [[qwen-2.5-recipe]] — Qwen2.5 stage chain과 보고서가 출력하지 않은 setting 목록.
- [[deepseek-v3]] 및 [[deepseek-v3-recipe]] — DeepSeek-V3 pre-training schedule, context-extension phase, distillation 기반 SFT, stage별 GPU-hour table.
- [[deepseek-r1]] 및 [[deepseek-r1-recipe]] — 네 stage R1 pipeline, stage별 benchmark table, distillation 대 RL 비교, 공개 GPU hours.
- [[epoch-deepseek-r1-compute]] — R1 stage budget의 외부 추정으로, 부분 공개에서 복원할 수 있는 것의 증거로 사용.
- [[k2-v2]] — pre-training부터 네 mid-training stage와 distilled SFT까지의 360-open 70B chain, hyper-parameter transfer 절차, memorization 측정.
- [[apertus]] — held-out suite를 선언하고 cooldown-ablation으로 mixture를 고른 다섯 stage pre-training chain.
