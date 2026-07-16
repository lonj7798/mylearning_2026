<!-- chapter: ch-06
     track: practice
     kind: content
     title: 실전 레시피 — TRL의 GKD와 모든 모델 계열을 위한 온-폴리시 증류
     deps: [[ch-04]]
     sources: [[hf-trl-gkd-recipe]], [[agarwal-gkd]], [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]]
-->

# 06장 — 실전 레시피: TRL의 GKD와 모든 모델 계열을 위한 온-폴리시 증류

> **핵심 통찰.** 온-폴리시 증류(on-policy distillation)는 논문 속 아이디어가 아니라 프로덕션용 트레이너다. TRL의 `GKDTrainer`는 [[ch-04]]의 메커니즘을 세 가지 설정 노브, 즉 `lmbda`(온-폴리시 비율 λ), `beta`(forward↔reverse KL β), `temperature`로 압축한다. 그리고 HuggingFace의 GOLD 확장은 실제 사용을 가로막던 한 가지 가정, 즉 교사와 학생이 **토크나이저를 공유해야 한다**는 가정을 제거한다. GOLD를 쓰면 *어떤* 모델 계열의 교사라도 학생의 롤아웃을 채점할 수 있다. 배포 가능한 학생과 다른 계열의 프런티어 모델이 가장 좋은 교사인 경우에 바로 필요한 기능이다.

> **가이드라인.** 같은 토크나이저 쌍이라면 높은 `lmbda`(온-폴리시)로 `GKDTrainer`를 실행하고, 작업별로 `beta`를 튜닝하라. 서로 다른 계열의 쌍(예: Claude/Llama 교사가 Qwen 학생을 채점하는 경우)이라면 `use_uld_loss=True`로 GOLD 확장을 사용하고 교사의 토크나이저를 제공하라. 진짜 병목, 즉 매 스텝마다 새 학생 샘플에 대해 교사 forward pass를 수행하는 비용을 예산에 반영하고, 실제로 온-폴리시 상태를 유지하고 있는지 검증하라.

---

## 1. `GKDTrainer`: 세 가지 노브로 표현한 메커니즘

TRL의 `GKDTrainer`는 "teacher model argument를 받는 `SFTTrainer` 클래스의 래퍼"다([[hf-trl-gkd-recipe]]). 각 스텝에서 교사는 (학생 또는 교사가 생성한) 시퀀스에 대해 토큰별 로짓을 제공하고, 손실은 그 로짓에 대한 generalized JSD가 된다. 즉 [[ch-04]]의 토큰별 채점을 구현한 것이다. 노브는 정확히 GKD 파라미터들이다([[agarwal-gkd]]).

- **`lmbda`**(기본값 `0.5`) — "학생 데이터 비율, 즉 온-폴리시 학생 생성 출력의 비중을 제어한다. `lmbda=0.0`이면 손실은 supervised JSD로 축소된다... `lmbda=1.0`이면 손실은 on-policy JSD로 축소되며, 이때 학생이 출력 시퀀스를 생성하고 교사가 이 시퀀스들에 대한 토큰별 피드백을 제공한다." 이것은 [[ch-01]]의 데이터 소스 축을 다이얼로 만든 것이다. 0 = 오프-폴리시 KD, 1 = 온-폴리시 증류.
- **`beta`**(기본값 `0.5`) — "generalized Jensen-Shannon Divergence에서 보간을 제어한다. `beta=0.0`이면 손실은 forward KL divergence에 가까워지고, `beta=1.0`이면 reverse KL divergence에 가까워진다." 기하 축을 다이얼로 만든 것이다. 모드 포괄(mode-covering) ↔ 모드 추구(mode-seeking).
- **`temperature`**(기본값 `0.9`) — 온-폴리시 학생 생성에 쓰는 샘플링 temperature.
- **`seq_kd`**(기본값 `False`) — 시퀀스 수준 KD(교사 생성 출력에 대한 supervised FT); `seq_kd=True, lmbda=0.0`은 [[ch-02]]의 Kim & Rush 코너다.

문서에서 바로 가져올 수 있는 실전 지침은 이렇다. "저자들은 온-폴리시 데이터(높은 `lmbda`)가 더 잘 동작하고, 최적의 `beta`는 작업과 평가 방법에 따라 달라진다는 점을 발견했다." 그리고 보존할 가치가 있는 실제 함정도 있다. Gemma2의 경우 `attn_implementation="kernels-community/flash-attn2"`를 설정하라. "그렇지 않으면 soft capping 기법 때문에 로짓에서 NaN이 발생한다."

> **인터랙티브 동반 자료:** [`figures/gkd-knobs.html`](figures/gkd-knobs.html) — `lmbda`×`beta` 평면에서 점을 드래그하면 어떤 방법을 선택했는지(supervised KD, on-policy distillation, forward vs reverse KL)를 볼 수 있고, 대응하는 `GKDConfig(...)` 스니펫이 실시간으로 갱신된다. TRL 설정을 [[ch-01]]의 지도와 연결한다.

---

## 2. 크로스 토크나이저 장벽 — 그리고 GOLD

기본 GKD는 교사와 학생이 **어휘를 공유한다**고 가정한다. 그래야 두 모델의 토큰별 분포가 위치별로 정렬된다. 이 가정은 가장 유용한 사례, 즉 *다른 계열*의 프런티어 교사를 학생에게 증류하는 경우를 조용히 배제한다. HuggingFace H4 레시피 "Unlocking On-Policy Distillation for Any Model Family"는 정확히 이 문제를 겨냥한다([[hf-trl-gkd-recipe]]). 문제는 원문 그대로, 온-폴리시 증류에 "교사와 학생 모델이 *동일한* 토크나이저 어휘를 공유해야 한다는 요구사항"이 있었다는 점이다.

그 방법인 **GOLD(General On-Policy Logit Distillation)**는 Universal Logit Distillation(ULD)을 온-폴리시 설정으로 확장한다. GOLD는 "학생과 교사 토큰을 점진적으로 디코딩하고, 같은 visible text를 가진 구간을 그룹화하며, 각 그룹 안에서 확률을 병합한다. 이를 통해 토큰 경계가 다르더라도 전체 완성문에 대해 손실 항을 계산할 수 있음이 보장된다." 여러 교사 토큰의 연속 구간이 하나의 학생 토큰에 매핑될 때, 확률은 chain rule로 병합된다.

```
P_merged(y) = P(y | ctx) · P(token₁ | token₀, ctx) · … · P(tokenₖ | …, ctx)
```

결과는 의도적으로 정규화하지 않는다. ULD 손실은 정렬 + L1 거리를 사용하므로 정규화가 필요 없다. 하이브리드 옵션은 정확한 어휘 일치는 직접 비교하고, 일치하지 않는 토큰에는 정렬된 확률 기반 ULD로 폴백한다. 보고된 효과는 이렇다. GOLD는 크로스 토크나이저 시나리오에서 ULD의 10%와 대비해 "교사 성능의 60%를 회복"했고, "GRPO보다 20% 더 뛰어났다."

`GOLDTrainer`는 "`GKDTrainer`로부터 온-폴리시 vs 오프-폴리시 스케줄링을 상속한다"(`beta`/`lmbda`/`seq_kd`가 그대로 이어진다는 뜻) 그리고 `use_uld_loss`와 `teacher_tokenizer_name_or_path`(`use_uld_loss=True`일 때 필수)를 추가한다. 이것이 `trl.experimental.gold`에 있다는 점에 유의하라. 빠르게 변하는 API이며, 기본 `learning_rate=1e-7`도 매우 낮다.

---

## 3. 진짜 병목: 교사 log-prob 서빙

모든 온-폴리시 스텝은 토큰별 분포를 얻기 위해 **학생의 새 샘플에 대한 교사 forward pass**를 필요로 한다([[hf-trl-gkd-recipe]], [[tm-on-policy-distillation]]). 이것이 비용 중심이며, 실전 레버는 교사를 co-locate하거나 배치 처리하는 것, 아무것도 캐시하지 않는 것(샘플은 설계상 매 스텝 새롭다), 그리고 학생 롤아웃(`max_new_tokens`, `num_generations`, `generation_batch_size`)의 크기를 교사 pass가 감당 가능한 수준으로 잡는 것이다. Thinking Machines의 프레이밍, 즉 OPD는 보상을 교사 호출로 바꾼 RL 루프라는 설명은 비용 모델이기도 하다. 롤아웃 배치마다 교사 추론 한 번의 비용을 지불한다.

---

## 4. 함정 체크리스트

- **실제로 온-폴리시인지 검증하라.** `lmbda`가 낮거나 샘플이 오래되면, 학습은 조용히 오프-폴리시 KD로 되돌아간다. 이것이 [[nrehiew-sft-rl-opd]]가 경고하는 드리프트다. `lmbda`를 높게 유지하고 매 스텝 다시 샘플링하라.
- **토크나이저 불일치.** 같은 계열인가? 일반 `GKDTrainer`를 쓰면 된다. 크로스 계열인가? 반드시 GOLD/ULD를 써야 한다. 서로 다른 토크나이저 사이의 순진한 token-to-token KL은 정의되지 않는다.
- **엔트로피 붕괴와 스타일 토큰**([[ch-05]]). 엔트로피를 모니터링하라. 토큰별 clipping을 사용하라. 학생이 교사를 정확히 맞출 수 없을 때는 `beta`를 reverse KL 쪽으로 잡아라.
- **교사는 크기만이 아니라 신호 기준으로 선택하라** — "교사는 예상보다 덜 중요하다"는 점을 떠올려라([[nrehiew-sft-rl-opd]]). log-prob를 얻을 수 없는 이상적인 교사보다 *사용 가능하고 서빙 가능한* 교사가 낫다.

---

## 5. 깨진 미신: "학생과 교사는 토크나이저/계열을 공유해야 한다"

이것은 기본 GKD에서는 *사실이었다*. 그래서 온-폴리시 증류가 "큰 형제를 작은 형제로 증류하기" 정도로 제한되어 보였던 것이다. GOLD/ULD는 이것을 깨뜨린다. 토큰 ID가 아니라 *visible text*를 기준으로 정렬하고, 불일치하는 경계에 걸쳐 확률을 병합함으로써 어떤 계열의 교사라도 어떤 학생의 롤아웃이든 채점할 수 있다. 이 단일 변화가 온-폴리시 증류를 흔한 현실 세계 구성, 즉 한 연구소의 프런티어 교사와 다른 곳의 배포 가능한 학생으로 구성된 환경에서 쓸 수 있게 만든다.

---

## 6. 적용: boson seller를 위한 레시피 선택

레시피를 `boson-agent-synthetic-data-dev`에 매핑해 보자.

- **학생(Student):** 배포 가능한 seller, `Qwen3.6-27B` 계열.
- **교사 옵션(Teacher options):** 큰 Qwen(같은 계열 → 일반 `GKDTrainer`, 가장 깔끔함) 또는 Claude(다른 계열 → **GOLD/ULD**, 크로스 토크나이저). [[ch-05]]의 교훈("data source > teacher")에 따르면, 수용 가능한 비용으로 *실제로 log-prob를 서빙할 수 있는* 교사를 선호하라. 그것이 같은 계열의 큰 Qwen이라면 GOLD의 복잡성을 완전히 피할 수 있다.
- **노브(Knobs):** `lmbda→1`(완전한 온-폴리시 seller turn — 장기 지평 에이전트에서는 이것이 핵심이다), reverse KL 쪽의 `beta`(27B seller는 프런티어 교사를 완전히 모방할 수 없으므로 재현 가능한 것에 mode-seek한다), 현실적인 판매 화법을 위한 적당한 `temperature`.
- **서빙(Serving):** 교사는 매 스텝 한국어 tool-calling seller turn을 채점해야 한다. 이것이 비용 동인이다. 교사 추론을 예산에 반영하고 turn당 `max_new_tokens`를 제한하라.

레시피는 정해졌다. 남은 것은 도메인 엔지니어링이다. tool-calling, compaction, barge-in이 뒤섞인 대화에서 어떤 토큰을 실제로 채점할 것인지, 그리고 기존 relay가 어떻게 온-폴리시 롤아웃 환경이 될 것인지가 캡스톤의 주제다.

---

## 다음으로 이어지는 곳

제7장은 실습이다. `boson-agent-synthetic-data-dev`를 위한 구체적인 온-폴리시 증류 전략을 다룬다. 전체 코스, 즉 분포 지도([[ch-01]]), 오프-폴리시 진단([[ch-02]], [[ch-03]]), 메커니즘([[ch-04]]), 경제성/실패 모드([[ch-05]]), 그리고 이 장의 레시피를 사용해 무엇을 샘플링하고, 무엇을 채점하며, 어떤 교사를 쓰고, 무엇을 측정할지 결정한다. 그런 다음 함께 논의한다.

## 추가 읽을거리

- TRL `GKDTrainer` 문서 — https://huggingface.co/docs/trl/gkd_trainer ([[hf-trl-gkd-recipe]])
- HuggingFace H4 Space, "Unlocking On-Policy Distillation for Any Model Family" — https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation ([[hf-trl-gkd-recipe]])
- Boizard et al., "Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs" (2024) — https://arxiv.org/abs/2402.12030 (ULD, GOLD가 확장하는 기반)
- Agarwal et al., "On-Policy Distillation of Language Models" (GKD) — https://arxiv.org/abs/2306.13649 ([[agarwal-gkd]])
