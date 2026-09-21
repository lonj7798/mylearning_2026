
## Q: Multi-turn assistant-turn 유사도 기반 loss masking으로 중복 신호를 줄일 수 있는가? (2026-09-18)

read.md §4의 세 가지 multi-turn masking 규칙(all-assistant, last-turn, unrolling)은 "어느 turn에 loss를 주는가"를 다루지만, "유사한 turn을 동적으로 제외"하는 방식은 다루지 않는다(read.md에 없는 확장 아이디어).

**학습자 제안**: assistant turn 간 n-gram 유사도를 계산해 유사한 turn을 masking하면 overfitting을 줄일 수 있지 않은가. 대상 범위를 "같은 대화 + 다른 대화", 유사도 계산 단위를 "(user, assistant) 쌍"으로 확장.

**핵심 트레이드오프** (두 방향 모두 검증된 소스 없음, [[read]] 밖 추론):
- Assistant 텍스트만 보고 dedup → 안전 거절처럼 **다른 user 질문에 의도적으로 반복되는 정당한 응답**을 잘못 제거할 위험.
- User 텍스트만 보고 dedup → 비슷한 질문 형태에 **의도적으로 다양하게 단 정당한 답변**을 잘못 제거할 위험.
- 절충: (user 유사 AND assistant 유사)일 때만 중복으로 플래그.

**구조적 정정**: 이 아이디어를 batch 단위(online)로 계산하면 무작위 셔플로 중복이 같은 batch에 들어올 확률이 낮아 실효성이 떨어진다. 실제로는 전체 데이터셋에 대해 offline dedup(MinHash류)으로 하는 것이 표준 관행 — deduplication은 이 장이 아니라 data curation 단계(course 내 별도 chapter)의 주제.

see [[read]] §4 (multi-turn masking rules).

## Q: Multi-turn 중간에 매우 긴 instruction turn이 있으면 masking보다 training이 나은가? (2026-09-18)

**정정**: "system prompt가 길 때 IM이 유리하다"는 표현은 부정확 — [[loss-masking-prompt]]의 `L_IT` 정의에서 instruction(=I_1..I_m)은 system prompt만이 아니라 user turn 전체(질문+input+context)를 가리킨다. worked example(AlpacaDataShort)에서도 prompt 179.27 토큰 중 instruction은 16.93, 나머지 162.34는 input.

**개념 정정**: masking은 모델이 해당 토큰을 "못 본다"는 뜻이 아니다 — causal attention은 masking 여부와 무관하게 항상 앞선 token을 조건으로 삼는다. Masking은 그 토큰을 loss target(추가 next-token-prediction 대상)으로 쓸지만 결정한다.

**가장 가까운 근거**: [[llama-2-long]] — 긴 input prompt에 LM loss를 추가했을 때 QuALITY 59.3→77.3, Qasper 35.7→38.9 (Table 9, 모델 크기 미기재). 방향은 "긴 instruction엔 masking보다 training이 유리"를 지지.

**한계**: 이 증거는 "긴 문서를 담은 단일 prompt" setting이지, "multi-turn 대화 중간에 낀 긴 turn"이라는 정확한 구조는 검증되지 않음 — §4 multi-turn masking Open question과 같은 미검증 영역.

see [[read]] §4.

## Q: instruction에 loss를 주는데(IM/PLW) 그 instruction이 데이터셋 전체에서 계속 반복되면 문제 없는가? (2026-09-18)

**핵심 논거 (Interpretation, 이 장의 다른 절 수식을 이 상황에 적용한 유도)**: read.md "Negative samples and negative feedback" 절의 softmax gradient 공식 `∂log p_y/∂z_j = 1[j=y] − p_j`를 positive target에 적용하면, gradient 크기는 (1 − p_y)에 비례한다. 완전히 결정적으로 반복되는 instruction은 학습이 진행될수록 p_y → 1에 수렴하므로, 반복될수록 실제로 흐르는 gradient는 0에 가까워진다.

**결론**: 이건 짧은 completion을 외우는 것(§4의 memorization 우려, 정답이 여러 개일 수 있는 자리를 하나로 좁힘)과는 성격이 다르다 — 반복되는 instruction 예측은 애초에 정답이 하나뿐인 쉬운 task라서, 반복될수록 "위험한 memorization"이 아니라 "gradient 기여가 없는 계산 낭비"에 가깝다.

**진단**: 학습 도중 반복 instruction 구간의 per-token loss를 로깅해서 실제로 0에 수렴하는지 확인. 수렴하지 않으면 완전한 반복이 아니라 미묘한 변형이 섞여 있다는 신호.

**주의**: 이 결론은 이 장의 직접 실험이 아니라 다른 절의 공식을 새 상황에 적용한 추론(Interpretation) — 검증된 Result 아님.

see [[read]] §4 (loss masking), Negative samples and negative feedback 절 (gradient 공식).

## Q: mean_resizing=True로 새 token embedding row를 기존 embedding의 mean/covariance에서 sample하는 게 무슨 뜻인가? (2026-09-18)

**메커니즘**: 기존 vocabulary의 V개 embedding row(각 d차원)를 하나의 분포에서 뽑힌 표본으로 보고, 그 평균 벡터와 공분산 행렬을 계산한 뒤, 새 token의 row를 그 평균·공분산을 가진 multivariate normal에서 뽑는다. 0-init이나 기존 분포와 무관한 random init 대신, 기존 embedding들과 통계적으로 비슷한 크기/방향에 새 row를 둔다.

**worked example** ([[read]] §9): output embeddings (1,0),(0,1),(−1,0),(0,−1), hidden state (2,1). 기존 top token 확률 0.696.
- 새 row를 평균(0,0)에 두면 → 새 토큰 확률 0.086, top token 0.696→0.636 (소폭 변화)
- 새 row를 분포 밖 임의값(3,0)에 두면 → 새 토큰 확률 0.974 (학습 전인데 거의 모든 확률을 가져감, 기존 예측 붕괴)

**목적**: resize 직후 next-token 분포를 resize 이전과 최대한 비슷하게 유지(KL divergence 최소화) — 학습되지 않은 새 토큰이 모델의 기존 성능을 갑자기 망가뜨리지 않게 하는 것.

see [[read]] §9.

## Q: `<think>`를 새 토큰으로 추가할 때 구체적으로 어떻게 initialize하는가? (2026-09-18)

§9의 mean-resizing 메커니즘(이전 Q 참고)은 `<think>`를 pretraining 이후 추가되는 새 토큰의 **대표 예시**로 쓰지만, 특정 lab(DeepSeek-R1, OLMo 3 등)이 `<think>`에 실제로 mean-resizing을 썼다는 인용은 없다 — §9가 인용하는 건 Transformers 코드 자체의 일반 동작이지 특정 모델의 기술보고서가 아니다.

`<think>`가 [[read]]에서 실제로 검증된 맥락은 초기화가 아니라 masking이다: §6의 OLMo 3 case — `add_generation_prompt`에 `<think>` 토큰이 들어가는데, tokenization 코드가 이를 prompt의 일부로 잘못 masking해서 학습 대상에서 빠지는 버그. §9 point 3("새 토큰은 모든 단계에서 loss mask 안에 있어야 한다")가 이 사례를 가리킨다.

**미검증**: `<think>`가 특정 모델에서 pretraining vocabulary에 이미 있었는지, post-training에서 mean-resizing으로 추가됐는지, 아니면 다른 방식(유사 토큰 embedding 복사 등)으로 초기화됐는지는 [[read]]의 소스로 확인 안 됨.

see [[read]] §6 (OLMo 3 think masking bug), §9 (일반 mean-resizing 메커니즘).
