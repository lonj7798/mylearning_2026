# 23b — 멀티턴 자료를 만들고 검사하는 네 방법

<!-- scope: primary-source setups, measured outcomes, and transfer limits for state updates, policy scope, counterfactual skill families, and causal supervision
     deps: [[read_kor]]
     see-also: [[read_kor]], [[practice_kor]], [[long-tail-papers]]
     sources: [[user-oriented-dialogue-generation]], [[consistentchat]], [[josh-dialogue-self-training]], [[multichallenge]]
-->

## 1. 태스크 생성과 사용자 시뮬레이션

**Result (single study).** [[user-oriented-dialogue-generation]]은 태스크 구성과 별도의 사용자 시뮬레이터를 나눠 요구사항을 조금씩 밝히고 턴별 피드백을 만듭니다. §5.1의 seed는 Nemotron 도구 자료와 τ2 database 태스크입니다. 시험된 도구 과제의 개선을 보고하지만 완전히 미지 환경의 일반화라는 주장은 제한됩니다. 생성 비용과 일부만 보이는 database 접근의 불안정성도 한계입니다.

**Interpretation.** 세일즈 시뮬레이터는 숨은 상황 계획을 갖되 자연스럽게 정보를 공개할 수 있습니다. 학생에게는 공개된 정보만 주고 성공은 숨은 템플릿의 일치가 아닌 유효한 행동으로 정합니다. 환경에서 행동을 확인할 수 있을 때 유용하지만 실제 고객 표현 빈도의 증거는 아닙니다.

## 2. Skeleton과 미래 질문의 불일치

**Result (single study).** [[consistentchat]]은 의도·정보 흐름을 설계하고 질문 시퀀스 뒤 응답을 한 번에 만듭니다. v2는 약 15,000 대화를 학습에 사용합니다. Qwen2.5-7B의 consistency는 7.32 대 ShareGPT 7.10, MT-Eval multi-turn은 8.38 대 7.86입니다(Tables 1–2). Judge 점수이지 실제 성공률 %는 아닙니다.

응답 생성기가 미래 사용자 질문을 보고 generator/judge도 겹칩니다. Skeleton은 일관성에 도움이 될 수 있으나 아직 밝히지 않은 정보를 미리 답하게 할 수 있습니다. 전체를 계획했더라도 각 답은 그 시점까지의 prefix로 검증합니다. 같은 계열 judge/student의 편향 때문에 독립적인 도구·사람 검사를 함께 사용합니다.

## 3. 성공 대화 경로 탐색

**Result (single study).** [[josh-dialogue-self-training]]은 시뮬레이션 도구 대화의 턴별 분기와 sparse success feedback으로 성공 경로·preference 자료를 추출합니다. 개별 턴이 그럴듯해도 전체 태스크가 실패할 수 있어 탐색이 유용합니다. 2024년 초판, Findings of ACL 2025 출판으로 일반 어휘 다양성보다 task reward 방법입니다.

**Interpretation.** 예약 성공을 실행 가능하고 정책 제약이 있는 기준으로 확인하면 수정·누락 조건을 지나는 경로를 찾을 수 있습니다. ‘예약이 존재한다’만 보면 허가 없는 예약도 통과하므로 선행 조건을 넣거나 따로 조사합니다. 시뮬레이터 오류·좁은 태스크 범위 때문에 실제 고객에게 부적합할 수 있습니다. 카드에 없는 최적 search width나 혼합 비율을 만들지 않습니다.

## 4. 스트레스 benchmark와 실제 빈도

**Result (single study).** [[multichallenge]]는 생성 후보·사람 수정·실패 기반 채택으로 273개 대화를 평가합니다. 지시 유지·이력 정보 추론·버전 수정·자기 일관성을 다룹니다. 항목별 binary rubric은 judge-human agreement를 높이지만 의도적으로 어려운 실패를 모았으므로 category 비율은 실제 고객 빈도가 아닙니다.

‘최종 호출이 수정된 날짜를 쓰는가’ 같은 판정 조건을 만드는 데 활용합니다. Hold-out benchmark를 학습한 뒤 독립 평가로 부르지 않습니다. 능력을 참고한 새 사례는 다른 family로 만들고 benchmark 답 패턴을 복제하지 않습니다. 명확한 행동 검사 아이디어이지 전체 long tail 보장은 아닙니다.

## 비교

| 방법 | 통제 | 감독 | 범위 한계 |
|---|---|---|---|
| User-oriented | 숨은 태스크/사용자 | 점진적 요청·피드백 | 환경/시뮬레이터 충실도 |
| ConsistentChat | 의도·흐름 | 생성 대화 | 미래 질문/judge 불일치 |
| JOSH | 분기 탐색 | 성공·preference | 보상·탐색 범위 |
| MultiChallenge | 실패·사람 검토 | 평가 rubric | 빈도 아닌 선별 |

[본문](read_kor.md) · [실습](practice_kor.md) · [English](methods.md).

