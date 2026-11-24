# 23b — 데이터 레코드와 스킬 경계 실험

<!-- scope: proposed implementation records, controlled comparisons, metrics, and failure repair for state updates, policy scope, counterfactual skill families, and causal supervision
     deps: [[read_kor]]
     see-also: [[read_kor]], [[methods_kor]], [[long-tail-papers]]
     sources: [[user-oriented-dialogue-generation]], [[consistentchat]], [[josh-dialogue-self-training]], [[multichallenge]]
-->

## 1. 정답 행동의 이유가 남는 레코드

제안 필드: family ID, turn ID가 있는 보이는 메시지, 현재 스킬 설명·버전·활성 턴, 전역 정책, 상품 근거 버전, 도구 schema, 필드 근거가 있는 고객 상태, 누락 값, 허용 행동, 선택한 target, 검증 기록입니다. 태스크·구성 태그는 metadata이며 정답 행동이나 라벨 상태를 추론 입력에 넣으면 누출될 수 있습니다.

Callback 날짜는 수정 턴, 시간은 시간 턴, 호출 가능성은 현재 정책을 가리킵니다. 배포 모델이 텍스트만 받아도 명시적 상태는 라벨링·validator에 유용합니다. 원본 메시지를 보관해 오래되거나 틀린 요약을 찾습니다. 라벨러가 상태를 잘못 갱신하면 정답을 완벽히 배워도 잘못된 행동을 합니다.

## 2. 의미 사례를 먼저 만들고 말투를 바꾼다

자연스러운 요구사항과 정책을 정하고 정답·필수 값을 확인한 뒤 표현 변형을 만듭니다. 취소는 호출 가능성을 바꾸고 공손함은 보통 바꾸지 않습니다. 수정 뒤 다시 판정해야 합니다. 생성기가 실수로 동의를 추가하거나 날짜 수정을 지울 수 있습니다. 호출 대신 질문이 정답인 경우도 유지합니다.

모든 조합을 무작정 늘리지 않습니다. 불가능하거나 정답 하나로 정할 수 없는 정책·상태 조합이 있습니다. 여러 응답이 유효하면 한 문장 exact match 대신 제약 집합을 기록합니다. 사람 검토는 지시를 따른 행동·내용에 초점을 두고 말투는 별도로 봅니다.

## 3. 목적함수와 선별의 차이

수정된 호출 SFT는 보이는 문맥의 긍정 target을 가르칩니다. DPO는 같은 입력의 preferred/rejected pair이며 유효한 선호 라벨이 필요합니다. 근거 없는 날짜 호출은 rejection이 가능하지만 똑같이 유효한 다른 답을 자동 rejected로 만들지 않습니다. 양쪽에 같은 정책을 제공합니다.

LESS는 기준 태스크의 학습 방향에 맞는 예제를 찾고 G-Vendi는 데이터 내부 그래디언트 다양성을 측정합니다. 없는 상황을 만들거나 목적함수를 바꾸는 것은 아닙니다. 수정 태스크에서 LESS가 적합할 수는 있지만 우월성이 보장되지는 않습니다. 같은 예산의 random/balanced selection과 비교하며 선별 효과와 SFT/DPO 효과를 나눕니다.

## 4. 일반화를 시험하는 hold-out

변형 생성 전에 family를 나눕니다. 새 값·새 이력 패턴·근거를 제공한 새 상품·새 스킬 설명·새 정책/상태 조합을 따로 시험합니다. 새 값은 추출, 새 패턴은 문맥 사용, 새 정책은 지시 이해를 확인합니다. Selection reference와 개발 사례는 최종 test에서 제외합니다.

Call/no-call 정확도·schema·canonicalization 기준 인자 정확도·근거·정책 타이밍을 측정합니다. 여러 행동이 유효하면 허용 집합으로 평가합니다. 발생한 호출의 정답뿐 아니라 금지된 호출이 없는지도 봅니다. 유창한 답은 잘못된 행동을 상쇄하지 않습니다.

## 5. 오류 분석과 수정

인터럽트·주제 변경 뒤 날짜만 틀리면 이력과 근거를 확인합니다. v2가 전부 틀리면 범위·활성 시점을 봅니다. 만료된 지역 규칙을 따르면 새 스킬 단독 예제보다 검증된 전환 경계 family를 추가합니다. 새 상품만 틀리면 grounding·상품 split leakage를 조사합니다.

임의의 미래 스킬을 입증하지는 못하지만 특정 의존성과 개입 효과를 확인할 수 있습니다. Common과 rare/stress를 나누고 유효한 no-action·작은 표본의 불확실성을 보고합니다. 어떤 논문 비율·양도 원하는 세일즈 행동을 보장하지 않습니다.

[본문](read_kor.md) · [방법](methods_kor.md) · [다음 장](../ch-23c/read_kor.md) · [English](practice.md).

